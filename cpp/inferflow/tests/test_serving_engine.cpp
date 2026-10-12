#include "fake_runtime.h"
#include "inferflow/bounded_queue.h"
#include "inferflow/serving_engine.h"

#include <chrono>
#include <condition_variable>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace inferflow;
using namespace std::chrono_literals;

namespace {

void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

RequestContext request(std::string id, float marker = 1.0f) {
    return {std::move(id),
            {{1, 2, 8}, std::vector<float>(16, marker), "float32"},
            RequestState::kAccepted, std::nullopt};
}

ServingEngine::Clock::time_point later() {
    return ServingEngine::Clock::now() + 60s;
}
Result take(std::future<Result>& future) {
    check(future.wait_for(2s) == std::future_status::ready,
          "future did not complete");
    return future.get();
}

struct GateState {
    std::mutex mu;
    std::condition_variable cv;
    std::vector<float> seen;
    bool released{false};

    bool wait_for_calls(std::size_t count) {
        std::unique_lock<std::mutex> lock(mu);
        return cv.wait_for(lock, 2s, [&] { return seen.size() >= count; });
    }

    void open() {
        {
            std::lock_guard<std::mutex> lock(mu);
            released = true;
        }
        cv.notify_all();
    }

    std::vector<float> observed() {
        std::lock_guard<std::mutex> lock(mu);
        return seen;
    }
};

class GateRuntime final : public IRuntime {
public:
    explicit GateRuntime(std::shared_ptr<GateState> state)
        : state_(std::move(state)) {}

    TensorOutput infer(const TensorInput& input) override {
        std::unique_lock<std::mutex> lock(state_->mu);
        state_->seen.push_back(input.values.at(0));
        state_->cv.notify_all();
        state_->cv.wait(lock, [&] { return state_->released; });
        return {input.shape, input.values};
    }

private:
    std::shared_ptr<GateState> state_;
};

struct ReleaseOnExit {
    std::shared_ptr<GateState> state;
    ~ReleaseOnExit() { state->open(); }
};

void test_queue_contract() {
    BoundedQueue<int> queue(1);
    check(queue.try_push(7) == PushResult::kOk, "first push failed");
    check(queue.try_push(8) == PushResult::kFull, "full queue accepted item");
    queue.close();
    check(queue.try_push(9) == PushResult::kClosed, "closed queue accepted item");
    check(queue.wait_pop() == std::optional<int>(7), "close lost queued item");
    check(!queue.wait_pop().has_value(), "closed drained queue did not end");

    BoundedQueue<int> empty(1);
    auto waiter = std::async(std::launch::async, [&] {
        return empty.wait_pop();
    });
    empty.close();
    check(waiter.wait_for(2s) == std::future_status::ready,
          "close did not wake empty waiter");
    check(!waiter.get().has_value(), "empty closed queue returned item");
}

void test_normal_and_runtime_error() {
    auto stats = std::make_shared<RuntimeStats>();
    ServingEngine engine(std::make_unique<FakeRuntime>(stats), 1);
    auto ok = engine.submit(request("ok"), later());
    check(take(ok).code == StatusCode::kOk, "normal request failed");
    check(stats->calls == 1, "normal request did not call runtime once");
    engine.close();
    auto rejected = engine.submit(request("after-close"), later());
    check(take(rejected).code == StatusCode::kUnavailable,
          "submit after close was not rejected");

    auto failed_stats = std::make_shared<RuntimeStats>();
    ServingEngine failing(
        std::make_unique<FakeRuntime>(failed_stats, true), 1);
    auto failure = failing.submit(request("failure"), later());
    check(take(failure).code == StatusCode::kRuntimeError,
          "runtime exception was not converted");
}

void test_full_and_duplicate() {
    auto gate = std::make_shared<GateState>();
    ServingEngine engine(std::make_unique<GateRuntime>(gate), 1);
    ReleaseOnExit guard{gate};
    auto a = engine.submit(request("A", 1), later());
    check(gate->wait_for_calls(1), "A did not enter runtime");
    auto b = engine.submit(request("B", 2), later());
    auto duplicate = engine.submit(request("B", 9), later());
    check(take(duplicate).code == StatusCode::kInvalidArgument,
          "duplicate ID was accepted");
    auto c = engine.submit(request("C", 3), later());
    check(take(c).code == StatusCode::kResourceExhausted,
          "third waiting request was not rejected");
    gate->open();
    check(take(a).code == StatusCode::kOk, "A failed");
    check(take(b).code == StatusCode::kOk, "B failed");
    check(gate->observed() == std::vector<float>({1, 2}),
          "rejected request reached runtime");
}

void test_queued_expiry() {
    auto gate = std::make_shared<GateState>();
    ServingEngine engine(std::make_unique<GateRuntime>(gate), 1);
    ReleaseOnExit guard{gate};
    auto a = engine.submit(request("A", 1), later());
    check(gate->wait_for_calls(1), "A did not enter runtime");
    const auto deadline = ServingEngine::Clock::now() + 30ms;
    auto b = engine.submit(request("B", 2), deadline);
    std::this_thread::sleep_until(deadline + 1ms);
    gate->open();
    check(take(a).code == StatusCode::kOk, "A failed");
    check(take(b).code == StatusCode::kDeadlineExceeded,
          "queued B did not expire");
    check(gate->observed() == std::vector<float>({1}),
          "expired B reached runtime");
}

void test_queued_and_running_cancel() {
    {
        auto gate = std::make_shared<GateState>();
        ServingEngine engine(std::make_unique<GateRuntime>(gate), 1);
        ReleaseOnExit guard{gate};
        auto a = engine.submit(request("A", 1), later());
        check(gate->wait_for_calls(1), "A did not enter runtime");
        auto b = engine.submit(request("B", 2), later());
        check(engine.cancel("B"), "queued B not found");
        check(!engine.cancel("missing"), "unknown request was found");
        gate->open();
        check(take(a).code == StatusCode::kOk, "A failed");
        check(take(b).code == StatusCode::kCancelled,
              "queued B not cancelled");
        check(gate->observed() == std::vector<float>({1}),
              "queued cancelled B reached runtime");
    }
    {
        auto gate = std::make_shared<GateState>();
        ServingEngine engine(std::make_unique<GateRuntime>(gate), 1);
        ReleaseOnExit guard{gate};
        auto a = engine.submit(request("running", 1), later());
        check(gate->wait_for_calls(1), "running request did not start");
        check(engine.cancel("running"), "running request not found");
        gate->open();
        check(take(a).code == StatusCode::kCancelled,
              "running cancellation result wrong");
        check(gate->observed() == std::vector<float>({1}),
              "running cancellation incorrectly skipped runtime");
    }
}

void test_drain() {
    auto gate = std::make_shared<GateState>();
    ServingEngine engine(std::make_unique<GateRuntime>(gate), 1);
    ReleaseOnExit guard{gate};
    auto a = engine.submit(request("A", 1), later());
    check(gate->wait_for_calls(1), "A did not enter runtime");
    auto b = engine.submit(request("B", 2), later());
    std::promise<void> closing_started;
    auto started = closing_started.get_future();
    auto closer = std::async(std::launch::async, [&] {
        closing_started.set_value();
        engine.close();
    });
    started.wait();
    const auto still_waiting = closer.wait_for(20ms);
    gate->open();
    check(still_waiting == std::future_status::timeout,
          "close returned before in-flight request finished");
    check(closer.wait_for(2s) == std::future_status::ready,
          "close did not complete");
    closer.get();
    check(take(a).code == StatusCode::kOk, "drain lost A");
    check(take(b).code == StatusCode::kOk, "drain lost B");
    check(gate->observed() == std::vector<float>({1, 2}),
          "drain did not execute accepted items");
    auto c = engine.submit(request("C", 3), later());
    check(take(c).code == StatusCode::kUnavailable,
          "close still accepted new request");
}

}  // namespace

int main() {
    try {
        test_queue_contract();
        test_normal_and_runtime_error();
        test_full_and_duplicate();
        test_queued_expiry();
        test_queued_and_running_cancel();
        test_drain();
        std::cout << "All serving engine tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
