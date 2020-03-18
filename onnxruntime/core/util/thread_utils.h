#include "core/platform/threadpool.h"
#include "core/session/onnxruntime_c_api.h"
#include <memory>
#include <string>

namespace onnxruntime {
namespace concurrency {


std::unique_ptr<ThreadPool> CreateThreadPool(Env* env, ThreadPoolParams options,
                                             Eigen::Allocator* allocator = nullptr);
}  // namespace concurrency
}  // namespace onnxruntime