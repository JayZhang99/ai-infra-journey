#include "fake_runtime.h"
#include "inferflow/service_core.h"
#include "inferflow/serving_engine.h"

#include <chrono>
#include <future>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace inferflow;
using namespace std::chrono_literals;

namespace {

void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

RequestContext request(std::string id, float marker,
                       std::int64_t rows = 1,
                       std::int64_t sequence = 2) {
    return {std::move(id),
            {{rows, sequence, 8},
             std::vector<float>(static_cast<std::size_t>(
                 rows * sequence * 8), marker), "float32"},
            RequestState::kAccepted, std::nullopt};
}

ServingEngine::Clock::time_point later() {
    return ServingEngine::Clock::now() + 5s;
}

Result take(std::future<Result>& future) {
    check(future.wait_for(2s) == std::future_status::ready,
          "batch future did not complete");
    return future.get();
}

void test_count_trigger_and_scatter() {
    auto stats = std::make_shared<RuntimeStats>();
    ServingEngine engine(std::make_unique<FakeRuntime>(stats), 4,
                         {3, 1s});
    auto a = engine.submit(request("a", 3.0f, 2), later());
    auto b = engine.submit(request("b", 7.0f), later());
    auto ar = take(a);
    auto br = take(b);
    check(stats->calls == 1, "two compatible requests were not batched");
    check(ar.code == StatusCode::kOk && br.code == StatusCode::kOk,
          "batched request failed");
    check(ar.output->shape == std::vector<std::int64_t>({2, 2, 8}) &&
          br.output->shape == std::vector<std::int64_t>({1, 2, 8}),
          "batch output shape was not restored");
    check(ar.output->values.front() == 3.0f &&
          br.output->values.front() == 7.0f,
          "batch output mapped to wrong request");
}

void test_time_and_drain_triggers() {
    auto stats = std::make_shared<RuntimeStats>();
    ServingEngine engine(std::make_unique<FakeRuntime>(stats), 4,
                         {4, 20ms});
    auto one = engine.submit(request("timer", 1.0f), later());
    check(take(one).code == StatusCode::kOk, "time trigger failed");
    check(stats->calls == 1, "time trigger missed runtime");

    auto drain_stats = std::make_shared<RuntimeStats>();
    ServingEngine draining(std::make_unique<FakeRuntime>(drain_stats), 4,
                           {4, 5s});
    auto pending = draining.submit(request("drain", 2.0f), later());
    draining.close();
    check(take(pending).code == StatusCode::kOk,
          "drain did not flush partial batch");
    check(drain_stats->calls == 1, "drain did not call runtime");
}

void test_shape_boundary_and_deadline() {
    auto stats = std::make_shared<RuntimeStats>();
    ServingEngine engine(std::make_unique<FakeRuntime>(stats), 4,
                         {2, 200ms});
    auto a = engine.submit(request("t2", 1.0f, 1, 2), later());
    auto b = engine.submit(request("t3", 2.0f, 1, 3), later());
    check(take(a).code == StatusCode::kOk, "first shape failed");
    check(take(b).code == StatusCode::kOk, "second shape failed");
    check(stats->calls == 2, "incompatible sequences merged");

    auto deadline_stats = std::make_shared<RuntimeStats>();
    ServingEngine deadlines(std::make_unique<FakeRuntime>(deadline_stats), 4,
                            {3, 1s});
    auto expired = deadlines.submit(
        request("expiring", 1.0f),
        ServingEngine::Clock::now() + 30ms);
    auto live = deadlines.submit(request("live", 2.0f), later());
    check(take(expired).code == StatusCode::kDeadlineExceeded,
          "expired member reached runtime");
    check(take(live).code == StatusCode::kOk,
          "live member was lost when peer expired");
    check(deadline_stats->calls == 1,
          "deadline filtering called runtime wrong number of times");
}

void test_cancel_and_failure() {
    auto stats = std::make_shared<RuntimeStats>();
    ServingEngine engine(std::make_unique<FakeRuntime>(stats), 4,
                         {2, 1s});
    auto cancelled = engine.submit(request("cancel", 1.0f), later());
    check(engine.cancel("cancel"), "bucketed item not cancellable");
    auto live = engine.submit(request("live", 2.0f), later());
    check(take(cancelled).code == StatusCode::kCancelled,
          "cancelled member reached runtime");
    check(take(live).code == StatusCode::kOk,
          "live member failed after peer cancellation");
    check(stats->calls == 1, "cancelled member caused an extra call");

    auto failure_stats = std::make_shared<RuntimeStats>();
    ServingEngine failing(
        std::make_unique<FakeRuntime>(failure_stats, true), 4,
        {2, 1s});
    auto a = failing.submit(request("a", 1.0f), later());
    auto b = failing.submit(request("b", 2.0f), later());
    check(take(a).code == StatusCode::kRuntimeError &&
          take(b).code == StatusCode::kRuntimeError,
          "batch runtime failure was not broadcast");
    check(failure_stats->calls == 1, "failed batch was executed twice");
}

void test_core_validation_isolation() {
    auto stats = std::make_shared<RuntimeStats>();
    ServiceCore core(std::make_unique<FakeRuntime>(stats));
    auto good_a = request("a", 1.0f);
    auto bad = request("bad", 2.0f);
    bad.input.dtype = "float64";
    auto good_b = request("b", 3.0f);
    auto results = core.execute_batch({&good_a, &bad, &good_b});
    check(results.size() == 3 &&
          results[0].code == StatusCode::kOk &&
          results[1].code == StatusCode::kInvalidArgument &&
          results[2].code == StatusCode::kOk,
          "invalid member polluted valid batch");
    check(stats->calls == 1, "validation isolation lost batching");
    check(results[0].output->values.front() == 1.0f &&
          results[2].output->values.front() == 3.0f,
          "core scatter order is wrong");
}

}  // namespace

int main() {
    try {
        test_count_trigger_and_scatter();
        test_time_and_drain_triggers();
        test_shape_boundary_and_deadline();
        test_cancel_and_failure();
        test_core_validation_isolation();
        std::cout << "All dynamic batching tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
