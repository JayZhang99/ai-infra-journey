#pragma once

#include <chrono>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace inferflow {

// A clock-injected, single-shape bucket. It does not own or execute requests.
class BatchBucket {
public:
    using Clock = std::chrono::steady_clock;
    using TimePoint = Clock::time_point;

    struct Item {
        std::string request_id;
        std::size_t rows;
    };

    BatchBucket(std::size_t max_batch_rows,
                Clock::duration max_wait)
        : max_batch_rows_(max_batch_rows), max_wait_(max_wait) {
        if (max_batch_rows == 0 || max_wait < Clock::duration::zero()) {
            throw std::invalid_argument("invalid batch policy");
        }
    }

    bool empty() const { return items_.empty(); }
    std::size_t rows() const { return rows_; }

    bool would_overflow(std::size_t incoming_rows) const {
        return incoming_rows > max_batch_rows_ - rows_;
    }

    void add(Item item, TimePoint now) {
        if (item.rows == 0 || item.rows > max_batch_rows_ ||
            would_overflow(item.rows)) {
            throw std::invalid_argument("request does not fit bucket");
        }
        if (items_.empty()) first_at_ = now;
        rows_ += item.rows;
        items_.push_back(std::move(item));
    }

    std::optional<TimePoint> due() const {
        if (!first_at_) return std::nullopt;
        return *first_at_ + max_wait_;
    }

    bool ready(TimePoint now, bool draining = false) const {
        return !empty() &&
            (draining || rows_ >= max_batch_rows_ || now >= *due());
    }

    std::vector<Item> take() {
        std::vector<Item> result = std::move(items_);
        items_.clear();
        rows_ = 0;
        first_at_.reset();
        return result;
    }

private:
    std::size_t max_batch_rows_;
    Clock::duration max_wait_;
    std::vector<Item> items_;
    std::size_t rows_{0};
    std::optional<TimePoint> first_at_;
};

}  // namespace inferflow
