#include "inferflow/serving_engine.h"

#include <algorithm>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using namespace std::chrono_literals;
using namespace inferflow;

struct RuntimeStats {
    std::atomic<std::size_t> calls{0};
};

class DelayedRuntime final : public IRuntime {
public:
    DelayedRuntime(std::shared_ptr<RuntimeStats> stats,
                   std::chrono::milliseconds delay)
        : stats_(std::move(stats)), delay_(delay) {}

    TensorOutput infer(const TensorInput& input) override {
        ++stats_->calls;
        std::this_thread::sleep_for(delay_);
        return {input.shape, input.values};
    }

private:
    std::shared_ptr<RuntimeStats> stats_;
    std::chrono::milliseconds delay_;
};

struct Row {
    int concurrency;
    std::size_t max_batch_rows;
    std::int64_t sequence;
    int client;
    int request;
    StatusCode status;
    TimingMs timing;
};

const char* status_name(StatusCode code) {
    switch (code) {
        case StatusCode::kOk: return "ok";
        case StatusCode::kInvalidArgument: return "invalid_argument";
        case StatusCode::kRuntimeError: return "runtime_error";
        case StatusCode::kUnavailable: return "unavailable";
        case StatusCode::kResourceExhausted: return "queue_full";
        case StatusCode::kDeadlineExceeded: return "deadline_exceeded";
        case StatusCode::kCancelled: return "cancelled";
    }
    return "unknown";
}

double percentile(std::vector<double> values, double p) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const auto rank = static_cast<std::size_t>(std::ceil(p * values.size()));
    const auto index = std::max<std::size_t>(1, rank) - 1;
    return values[std::min(index, values.size() - 1)];
}

void run_case(std::ofstream& csv, int concurrency,
              std::size_t max_batch_rows, std::int64_t sequence,
              int requests_per_client) {
    constexpr auto runtime_delay = 3ms;
    const auto queue_capacity =
        std::max<std::size_t>(8, static_cast<std::size_t>(concurrency * 4));
    auto stats = std::make_shared<RuntimeStats>();
    ServingEngine engine(std::make_unique<DelayedRuntime>(stats, runtime_delay),
                         queue_capacity,
                         {max_batch_rows, 2ms});

    std::mutex rows_mu;
    std::vector<Row> rows;
    std::vector<std::thread> clients;
    Clock::time_point last_completion{};
    std::barrier start_gate(concurrency + 1);

    for (int client = 0; client < concurrency; ++client) {
        clients.emplace_back([&, client] {
            start_gate.arrive_and_wait();
            for (int request = 0; request < requests_per_client; ++request) {
                const std::string id = std::to_string(client) + "-" +
                                       std::to_string(request);
                RequestContext ctx{
                    id,
                    {{1, sequence, 8},
                     std::vector<float>(static_cast<std::size_t>(sequence * 8),
                                        1.0f),
                     "float32"}
                };
                auto future = engine.submit(std::move(ctx), Clock::now() + 5s);
                Result result = future.get();
                const auto completed_at = Clock::now();
                std::lock_guard<std::mutex> lock(rows_mu);
                rows.push_back({concurrency, max_batch_rows, sequence,
                                client, request, result.code, result.timing});
                last_completion = std::max(last_completion, completed_at);
            }
        });
    }
    const auto begin = Clock::now();
    start_gate.arrive_and_wait();
    for (auto& client : clients) client.join();
    engine.close();

    std::vector<double> all_latency;
    std::vector<double> successful_latency;
    std::size_t accepted = 0;
    std::size_t completed = 0;
    std::size_t successful = 0;
    for (const Row& row : rows) {
        all_latency.push_back(row.timing.end_to_end);
        if (row.status != StatusCode::kResourceExhausted &&
            row.status != StatusCode::kUnavailable) {
            ++accepted;
            ++completed;  // each accepted future has reached a terminal result
        }
        if (row.status == StatusCode::kOk) {
            ++successful;
            successful_latency.push_back(row.timing.end_to_end);
        }
        csv << "synthetic," << concurrency << ',' << max_batch_rows << ','
            << sequence << ',' << row.client << ',' << row.request << ','
            << status_name(row.status) << ',' << row.timing.queue_time << ','
            << row.timing.pre_bucket_wait << ',' << row.timing.batch_wait << ','
            << row.timing.runtime_time << ',' << row.timing.end_to_end << '\n';
    }
    csv.flush();

    const double elapsed_s =
        std::chrono::duration<double>(last_completion - begin).count();
    const double responded_qps = elapsed_s > 0.0 ? rows.size() / elapsed_s : 0.0;
    const double completed_qps = elapsed_s > 0.0 ? completed / elapsed_s : 0.0;
    const double success_qps = elapsed_s > 0.0 ? successful / elapsed_s : 0.0;
    std::cout << "load_model=closed_loop evidence=synthetic"
              << " concurrency=" << concurrency
              << " max_batch_rows=" << max_batch_rows
              << " sequence=" << sequence
              << " offered=" << rows.size()
              << " accepted=" << accepted
              << " completed=" << completed
              << " success=" << successful
              << " responded_qps=" << responded_qps
              << " completed_qps=" << completed_qps
              << " success_qps=" << success_qps
              << " samples=" << all_latency.size()
              << " p50_all_ms=" << percentile(all_latency, 0.50)
              << " p95_all_ms=" << percentile(all_latency, 0.95)
              << " p99_all_ms=" << percentile(all_latency, 0.99)
              << " p95_success_ms=" << percentile(successful_latency, 0.95)
              << " runtime_calls=" << stats->calls.load() << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    const std::string output = argc > 1 ? argv[1] :
        "benchmarks/inferflow/10.11_capacity_samples.csv";
    const auto parent = std::filesystem::path(output).parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent);
    std::ofstream csv(output);
    if (!csv) {
        std::cerr << "cannot open output CSV: " << output << '\n';
        return 1;
    }
    csv << "evidence,concurrency,max_batch_rows,sequence,client,request,status,"
           "queue_ms,pre_bucket_ms,batch_wait_ms,runtime_ms,end_to_end_ms\n";

    for (const auto sequence : {16, 64}) {
        for (const auto batch_rows : {std::size_t{1}, std::size_t{4}}) {
            for (const int concurrency : {1, 4, 8}) {
                run_case(csv, concurrency, batch_rows, sequence, 50);
            }
        }
    }
    return 0;
}
