#pragma once

#include "inferflow/bounded_queue.h"
#include "inferflow/service_core.h"

#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>

namespace inferflow {

class ServingEngine {
public:
    using Clock = std::chrono::steady_clock;
    struct BatchPolicy {
        std::size_t max_batch_rows{1};
        Clock::duration max_wait{Clock::duration::zero()};
    };

    ServingEngine(std::unique_ptr<IRuntime> runtime,
                  std::size_t queue_capacity);
    ServingEngine(std::unique_ptr<IRuntime> runtime,
                  std::size_t queue_capacity, BatchPolicy policy);
    ~ServingEngine();

    ServingEngine(const ServingEngine&) = delete;
    ServingEngine& operator=(const ServingEngine&) = delete;

    std::future<Result> submit(RequestContext ctx,
                               Clock::time_point deadline);
    bool cancel(std::string_view request_id);
    void close();  // drain：不接新请求，处理完已接收请求

private:
    struct Pending {
        RequestContext ctx;
        std::promise<Result> promise;
        Clock::time_point deadline;
        std::atomic<bool> cancel_requested{false};
    };

    void worker_loop();

    ServiceCore core_;
    BoundedQueue<std::shared_ptr<Pending>> queue_;
    BatchPolicy policy_;

    std::mutex pending_mu_;
    std::unordered_map<std::string, std::shared_ptr<Pending>> pending_;

    std::mutex close_mu_;
    std::thread worker_;
};

}  // namespace inferflow
