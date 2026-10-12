#include "fake_runtime.h"
#include "inferflow/service_core.h"

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace inferflow;

void check(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

RequestContext valid_request(std::string id = "request-1") {
    return {
        std::move(id),
        {{1, 2, 8}, std::vector<float>(16, 1.0f), "float32"},
        RequestState::kAccepted,
        std::nullopt,
    };
}

int main() {
    auto stats = std::make_shared<RuntimeStats>();

    {
        ServiceCore core(std::make_unique<FakeRuntime>(stats));

        // 正常请求
        auto normal = valid_request();
        Result ok = core.execute_one(normal);
        check(ok.code == StatusCode::kOk, "normal request failed");
        check(ok.output.has_value(), "normal output missing");
        check(ok.output->values.size() == 16, "wrong output size");
        check(normal.state == RequestState::kSucceeded, "wrong state");
        check(stats->calls == 1, "runtime should be called once");

        // 非法 rank：不能调用 Runtime
        auto bad_rank = valid_request("bad-rank");
        bad_rank.input.shape = {2, 8};
        check(core.execute_one(bad_rank).code ==
                  StatusCode::kInvalidArgument,
              "bad rank was accepted");
        check(stats->calls == 1, "bad rank reached runtime");

        // shape 与数据量不匹配：不能调用 Runtime
        auto bad_size = valid_request("bad-size");
        bad_size.input.values.pop_back();
        check(core.execute_one(bad_size).code ==
                  StatusCode::kInvalidArgument,
              "bad size was accepted");
        check(stats->calls == 1, "bad size reached runtime");

        // dtype 不支持：不能调用 Runtime
        auto bad_dtype = valid_request("bad-dtype");
        bad_dtype.input.dtype = "float64";
        check(core.execute_one(bad_dtype).code ==
                  StatusCode::kInvalidArgument,
              "bad dtype was accepted");
        check(stats->calls == 1, "bad dtype reached runtime");

        // 同一个 RequestContext 不执行第二次
        check(core.execute_one(normal).code == StatusCode::kOk,
              "completed result changed");
        check(stats->calls == 1, "request executed twice");

        // 多个请求复用同一个 Runtime
        for (int i = 0; i < 3; ++i) {
            auto request = valid_request(
                "reuse-" + std::to_string(i)
            );
            check(core.execute_one(request).code == StatusCode::kOk,
                  "reuse request failed");
        }
        check(stats->constructed == 1,
              "runtime constructed more than once");
        check(stats->calls == 4, "unexpected inference count");

        // 停止后，新请求被拒绝
        core.stop();
        auto after_stop = valid_request("after-stop");
        check(core.execute_one(after_stop).code ==
                  StatusCode::kUnavailable,
              "request accepted after stop");
        check(stats->calls == 4,
              "request after stop reached runtime");
    }

    check(stats->destroyed == 1,
          "runtime not destroyed with ServiceCore");

    // Runtime 异常必须变成 Result
    auto failing_stats = std::make_shared<RuntimeStats>();
    ServiceCore failing_core(
        std::make_unique<FakeRuntime>(failing_stats, true)
    );

    auto failing_request = valid_request("runtime-error");
    Result failure = failing_core.execute_one(failing_request);
    check(failure.code == StatusCode::kRuntimeError,
          "runtime exception not converted");
    check(failing_request.state == RequestState::kRuntimeFailed,
          "wrong failure state");

    std::cout << "All inferflow tests passed\n";
}
