#pragma once

#include "inferflow/request_context.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace inferflow {

struct BatchSlice {
    std::string request_id;
    std::size_t row_start;
    std::size_t row_count;
};

struct BatchPlan {
    TensorInput merged;
    std::vector<BatchSlice> slices;
};

inline std::size_t checked_product(std::size_t a, std::size_t b) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::invalid_argument("tensor size overflow");
    }
    return a * b;
}

// Pure logic: no thread, queue, runtime, or ownership of RequestContext.
// The caller keeps contexts alive until this function returns.
inline BatchPlan make_batch_plan(
    const std::vector<const RequestContext*>& requests
) {
    if (requests.empty()) {
        throw std::invalid_argument("empty batch");
    }

    BatchPlan plan;
    std::size_t total_rows = 0;
    std::int64_t common_t = -1;

    for (const RequestContext* ctx : requests) {
        if (!ctx || ctx->request_id.empty()) {
            throw std::invalid_argument("missing request or request_id");
        }
        const TensorInput& x = ctx->input;
        if (x.shape.size() != 3 || x.dtype != "float32") {
            throw std::invalid_argument("expected rank-3 float32 input");
        }
        const auto [b, t, d] =
            std::tuple{x.shape[0], x.shape[1], x.shape[2]};
        if (b <= 0 || t <= 0 || d != 8) {
            throw std::invalid_argument("expected positive B/T and D=8");
        }
        if (common_t != -1 && t != common_t) {
            throw std::invalid_argument("mixed sequence lengths");
        }
        common_t = t;

        const auto rows = static_cast<std::size_t>(b);
        const auto elems_per_row = checked_product(
            static_cast<std::size_t>(t), static_cast<std::size_t>(d));
        if (x.values.size() != checked_product(rows, elems_per_row)) {
            throw std::invalid_argument("shape/data size mismatch");
        }
        if (rows > static_cast<std::size_t>(
                std::numeric_limits<std::int64_t>::max()) - total_rows) {
            throw std::invalid_argument("batch rows overflow");
        }

        plan.slices.push_back({ctx->request_id, total_rows, rows});
        total_rows += rows;
        plan.merged.values.insert(
            plan.merged.values.end(), x.values.begin(), x.values.end());
    }

    plan.merged.shape = {
        static_cast<std::int64_t>(total_rows), common_t, 8};
    plan.merged.dtype = "float32";
    return plan;
}

inline std::vector<TensorOutput> split_batch_output(
    const BatchPlan& plan,
    const TensorOutput& output
) {
    if (output.shape != plan.merged.shape ||
        output.values.size() != plan.merged.values.size()) {
        throw std::invalid_argument("batched output contract mismatch");
    }

    const std::size_t elems_per_row = checked_product(
        static_cast<std::size_t>(plan.merged.shape[1]),
        static_cast<std::size_t>(plan.merged.shape[2]));
    std::vector<TensorOutput> parts;
    parts.reserve(plan.slices.size());
    for (const BatchSlice& slice : plan.slices) {
        const std::size_t begin = slice.row_start * elems_per_row;
        const std::size_t end = begin + slice.row_count * elems_per_row;
        parts.push_back({
            {static_cast<std::int64_t>(slice.row_count),
             plan.merged.shape[1], plan.merged.shape[2]},
            {output.values.begin() + static_cast<std::ptrdiff_t>(begin),
             output.values.begin() + static_cast<std::ptrdiff_t>(end)}
        });
    }
    return parts;
}

}  // namespace inferflow
