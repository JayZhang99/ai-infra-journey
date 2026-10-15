#include "inferflow/serving_engine.h"
#include "inferflow/batch_bucket.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace inferflow {
namespace {

Result error_result(StatusCode code, const char* message) {
    return {code, std::nullopt, message};
}

double elapsed_ms(ServingEngine::Clock::time_point begin,
                  ServingEngine::Clock::time_point end) {
    if (begin == ServingEngine::Clock::time_point{} ||
        end == ServingEngine::Clock::time_point{} || end < begin) {
        return 0.0;
    }
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

TimingMs summarize_timing(const RequestTiming& timing) {
    TimingMs result;
    result.queue_time = elapsed_ms(timing.enqueued_at, timing.dequeued_at);
    result.pre_bucket_wait = elapsed_ms(timing.dequeued_at,
                                        timing.bucketed_at);
    result.batch_wait = elapsed_ms(timing.bucketed_at,
                                   timing.dispatched_at);
    result.runtime_time = elapsed_ms(timing.runtime_begin,
                                     timing.runtime_end);
    result.end_to_end = elapsed_ms(timing.submitted_at,
                                   timing.completed_at);
    return result;
}

ServingEngine::BatchPolicy checked_policy(ServingEngine::BatchPolicy policy) {
    if (policy.max_batch_rows == 0 ||
        policy.max_wait < ServingEngine::Clock::duration::zero()) {
        throw std::invalid_argument("invalid batch policy");
    }
    return policy;
}

std::optional<std::int64_t> batch_sequence(const RequestContext& ctx,
                                            std::size_t max_batch_rows) {
    const auto& input = ctx.input;
    if (input.dtype != "float32" || input.shape.size() != 3 ||
        input.shape[0] <= 0 || input.shape[1] <= 0 ||
        input.shape[2] != 8 ||
        static_cast<std::uint64_t>(input.shape[0]) > max_batch_rows) {
        return std::nullopt;
    }
    return input.shape[1];
}

}  // namespace

ServingEngine::ServingEngine(std::unique_ptr<IRuntime> runtime,
                             std::size_t queue_capacity)
    : ServingEngine(std::move(runtime), queue_capacity, BatchPolicy{}) {}

ServingEngine::ServingEngine(std::unique_ptr<IRuntime> runtime,
                             std::size_t queue_capacity,
                             BatchPolicy policy)
    : core_(std::move(runtime)),
      queue_(queue_capacity),
      policy_(checked_policy(policy)),
      worker_([this] { worker_loop(); }) {}

ServingEngine::~ServingEngine() {
    close();
}

std::future<Result> ServingEngine::submit(
    RequestContext ctx,
    Clock::time_point deadline
) {
    auto item = std::make_shared<Pending>();
    item->ctx = std::move(ctx);
    item->deadline = deadline;
    item->ctx.timing.submitted_at = Clock::now();
    auto future = item->promise.get_future();
    auto complete_immediately = [&](Result result) {
        item->ctx.timing.completed_at = Clock::now();
        result.timing.end_to_end = elapsed_ms(
            item->ctx.timing.submitted_at, item->ctx.timing.completed_at);
        item->ctx.result = result;
        item->promise.set_value(std::move(result));
    };

    if (Clock::now() >= deadline) {
        complete_immediately(error_result(
            StatusCode::kDeadlineExceeded, "deadline already expired"));
        return future;
    }

    {
        // 注册请求和入队放在同一个临界区：
        // cancel(id) 不会看见一个最终入队失败的请求。
        std::lock_guard<std::mutex> lock(pending_mu_);

        const auto& id = item->ctx.request_id;
        if (pending_.count(id) != 0) {
            complete_immediately(error_result(
                StatusCode::kInvalidArgument, "duplicate request_id"));
            return future;
        }

        pending_.emplace(id, item);
        item->ctx.state = RequestState::kQueued;
        item->ctx.timing.enqueued_at = Clock::now();
        const PushResult pushed = queue_.try_push(item);

        if (pushed != PushResult::kOk) {
            pending_.erase(id);
            complete_immediately(
                pushed == PushResult::kFull
                    ? error_result(StatusCode::kResourceExhausted,
                                   "waiting queue is full")
                    : error_result(StatusCode::kUnavailable,
                                   "engine is closed"));
            return future;
        }
    }

    return future;
}

bool ServingEngine::cancel(std::string_view request_id) {
    std::lock_guard<std::mutex> lock(pending_mu_);
    auto it = pending_.find(std::string(request_id));
    if (it == pending_.end()) return false;

    // 只发出取消请求；由 Worker 独占 promise 的完成权。
    it->second->cancel_requested.store(true);
    return true;
}

void ServingEngine::worker_loop() {
    using Item = std::shared_ptr<Pending>;
    BatchBucket bucket(policy_.max_batch_rows, policy_.max_wait);
    std::vector<Item> group;
    std::optional<Item> deferred;
    std::optional<std::int64_t> group_sequence;

    auto complete = [this](const Item& item, Result result) {
        item->ctx.timing.completed_at = Clock::now();
        result.timing = summarize_timing(item->ctx.timing);
        item->ctx.result = result;
        {
            std::lock_guard<std::mutex> lock(pending_mu_);
            item->promise.set_value(std::move(result));
            auto it = pending_.find(item->ctx.request_id);
            if (it != pending_.end() && it->second == item) {
                pending_.erase(it);
            }
        }
    };

    auto run_group = [&] {
        std::vector<RequestContext*> ready;
        std::vector<Item> ready_items;
        for (const Item& item : group) {
            if (item->cancel_requested.load()) {
                item->ctx.state = RequestState::kCancelled;
                complete(item, error_result(StatusCode::kCancelled,
                                            "cancelled before inference"));
            } else if (Clock::now() >= item->deadline) {
                item->ctx.state = RequestState::kExpired;
                complete(item, error_result(StatusCode::kDeadlineExceeded,
                                            "expired before inference"));
            } else {
                ready.push_back(&item->ctx);
                ready_items.push_back(item);
            }
        }
        if (!ready.empty()) {
            // ServiceCore retains validation and Runtime ownership.
            const auto dispatched_at = Clock::now();
            for (const Item& item : ready_items) {
                item->ctx.timing.dispatched_at = dispatched_at;
            }
            auto results = core_.execute_batch(ready);
            for (std::size_t i = 0; i < ready_items.size(); ++i) {
                const Item& item = ready_items[i];
                Result result = std::move(results[i]);
                if (item->cancel_requested.load()) {
                    item->ctx.state = RequestState::kCancelled;
                    result = error_result(StatusCode::kCancelled,
                                          "cancelled during inference");
                } else if (Clock::now() >= item->deadline) {
                    item->ctx.state = RequestState::kExpired;
                    result = error_result(StatusCode::kDeadlineExceeded,
                                          "expired during inference");
                }
                complete(item, std::move(result));
            }
        }
        group.clear();
        bucket.take();
        group_sequence.reset();
    };

    while (true) {
        if (group.empty()) {
            std::optional<Item> next;
            if (deferred) {
                next = std::move(deferred);
                deferred.reset();
            } else {
                next = queue_.wait_pop();
                if (next) (*next)->ctx.timing.dequeued_at = Clock::now();
            }
            if (!next) break;
            const auto sequence = batch_sequence((*next)->ctx,
                                                 policy_.max_batch_rows);
            group.push_back(std::move(*next));
            if (!sequence) {
                run_group();
                continue;
            }
            group_sequence = sequence;
            group.front()->ctx.timing.bucketed_at = Clock::now();
            bucket.add({group.front()->ctx.request_id,
                        static_cast<std::size_t>(
                            group.front()->ctx.input.shape[0])},
                       group.front()->ctx.timing.bucketed_at);
        }

        if (bucket.ready(Clock::now(), queue_.closed_and_empty())) {
            run_group();
            continue;
        }

        auto wake_at = *bucket.due();
        for (const Item& item : group) {
            wake_at = std::min(wake_at, item->deadline);
        }
        auto next = queue_.wait_pop_until(wake_at);
        if (!next) {
            run_group();  // timer, deadline, or drain trigger
            continue;
        }
        (*next)->ctx.timing.dequeued_at = Clock::now();
        const auto sequence = batch_sequence((*next)->ctx,
                                             policy_.max_batch_rows);
        const auto rows = sequence
            ? static_cast<std::size_t>((*next)->ctx.input.shape[0]) : 0;
        if (sequence != group_sequence || bucket.would_overflow(rows)) {
            deferred = std::move(*next);
            run_group();
            continue;
        }
        (*next)->ctx.timing.bucketed_at = Clock::now();
        bucket.add({(*next)->ctx.request_id, rows},
                   (*next)->ctx.timing.bucketed_at);
        group.push_back(std::move(*next));
    }
}

void ServingEngine::close() {
    std::lock_guard<std::mutex> lock(close_mu_);
    queue_.close();                  // 先关 admission
    if (worker_.joinable()) {
        worker_.join();              // 再等已有请求处理完
    }
    core_.stop();                    // 最后停止旧的同步 ServiceCore
}

}  // namespace inferflow
