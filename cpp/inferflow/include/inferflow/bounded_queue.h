#pragma once

#include <condition_variable>
#include <chrono>
#include <cstddef>
#include <deque>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>

namespace inferflow {

enum class PushResult { kOk, kFull, kClosed };

template <typename T>
class BoundedQueue {
public:
    explicit BoundedQueue(std::size_t capacity) : capacity_(capacity) {
        if (capacity == 0) {
            throw std::invalid_argument("queue capacity must be positive");
        }
    }

    PushResult try_push(T item) {
        std::lock_guard<std::mutex> lock(mu_);

        if (closed_) return PushResult::kClosed;
        if (items_.size() >= capacity_) return PushResult::kFull;

        items_.push_back(std::move(item));
        cv_.notify_one();
        return PushResult::kOk;
    }

    std::optional<T> wait_pop() {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait(lock, [this] {
            return closed_ || !items_.empty();
        });

        // close 后仍继续取出已有请求；只有“关闭且排空”才退出。
        if (items_.empty()) return std::nullopt;

        T item = std::move(items_.front());
        items_.pop_front();
        return item;
    }

    template <typename Clock, typename Duration>
    std::optional<T> wait_pop_until(
        const std::chrono::time_point<Clock, Duration>& deadline) {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait_until(lock, deadline, [this] {
            return closed_ || !items_.empty();
        });
        if (items_.empty()) return std::nullopt;
        T item = std::move(items_.front());
        items_.pop_front();
        return item;
    }

    bool closed_and_empty() const {
        std::lock_guard<std::mutex> lock(mu_);
        return closed_ && items_.empty();
    }

    void close() {
        {
            std::lock_guard<std::mutex> lock(mu_);
            closed_ = true;
        }
        cv_.notify_all();
    }

private:
    const std::size_t capacity_;
    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::deque<T> items_;
    bool closed_{false};
};

}  // namespace inferflow
