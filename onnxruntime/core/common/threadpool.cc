
/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <memory>

#include "core/platform/threadpool.h"
#include "core/common/common.h"
#ifndef EIGEN_USE_THREADS
#define EIGEN_USE_THREADS
#endif
#include "core/util/eigen_common_wrapper.h"
#include "core/platform/EigenNonBlockingThreadPool.h"
#include "core/platform/ort_mutex.h"

namespace onnxruntime {
namespace {
class BlockingCounter {
 public:
  BlockingCounter(int initial_count) : state_(initial_count << 1), notified_(false) {
    ORT_ENFORCE(initial_count >= 0);
#ifndef NDEBUG
    ORT_ENFORCE(((initial_count << 1) >> 1) == initial_count);
#endif
  }

  ~BlockingCounter() = default;

  inline void DecrementCount() {
    unsigned int v = state_.fetch_sub(2, std::memory_order_acq_rel) - 2;
    if (v != 1) {
#ifndef NDEBUG
      ORT_ENFORCE(((v + 2) & ~1) != 0);
#endif
      return;  // either count has not dropped to 0, or waiter is not waiting
    }
    std::lock_guard<OrtMutex> l(mu_);
    notified_ = true;
    cond_var_.notify_all();
  }

  inline void Wait() {
    unsigned int v = state_.fetch_or(1, std::memory_order_acq_rel);
    if ((v >> 1) == 0)
      return;
    std::unique_lock<OrtMutex> l(mu_);
    while (!notified_) {
      cond_var_.wait(l);
    }
  }
  // Wait for the specified time, return false iff the count has not dropped to
  // zero before the timeout expired.
  inline bool WaitFor(std::chrono::milliseconds ms) {
    unsigned int v = state_.fetch_or(1, std::memory_order_acq_rel);
    if ((v >> 1) == 0)
      return true;
    std::unique_lock<OrtMutex> l(mu_);
    while (!notified_) {
      const std::cv_status status = cond_var_.wait_for(l, ms);
      if (status == std::cv_status::timeout) {
        return false;
      }
    }
    return true;
  }

 private:
  OrtMutex mu_;
  OrtCondVar cond_var_;
  std::atomic<int> state_;  // low bit is waiter flag
  bool notified_;
};
}  // namespace
namespace concurrency {

ThreadPool::ThreadPool(Env* env, const ThreadOptions& thread_options, const NAME_CHAR_TYPE* name, int num_threads,
                       bool low_latency_hint, Eigen::Allocator* allocator)
    : thread_options_(thread_options) {
  ORT_ENFORCE(num_threads >= 1);
  eigen_threadpool_ =
      onnxruntime::make_unique<ThreadPoolTempl<Env>>(name, num_threads, low_latency_hint, *env, thread_options_);
  underlying_threadpool_ = eigen_threadpool_.get();
  threadpool_device_ =
      onnxruntime::make_unique<Eigen::ThreadPoolDevice>(underlying_threadpool_, num_threads, allocator);
}

ThreadPool::ThreadPool(Eigen::ThreadPoolInterface* user_threadpool, Eigen::Allocator* allocator)
    : thread_options_(ThreadOptions()) {
  underlying_threadpool_ = user_threadpool;
  threadpool_device_ = onnxruntime::make_unique<Eigen::ThreadPoolDevice>(
      underlying_threadpool_, underlying_threadpool_->NumThreads(), allocator);
}

ThreadPool::~ThreadPool() = default;
void ThreadPool::SimpleParallelFor(std::ptrdiff_t total, std::function<void(std::ptrdiff_t)> fn) {
  if (total <= 0)
    return;

  if (total == 1) {
    fn(0);
    return;
  }

  Barrier barrier(static_cast<unsigned int>(total));
  std::function<void(std::ptrdiff_t)> handle_iteration = [&barrier, &fn](std::ptrdiff_t iteration) {
    fn(iteration);
    barrier.Notify();
  };

  for (std::ptrdiff_t id = 0; id < total; ++id) {
    Schedule([=, &handle_iteration]() { handle_iteration(id); });
  }

  barrier.Wait();
}

void ThreadPool::Schedule(std::function<void()> fn) {
  ORT_ENFORCE(fn != nullptr);
  underlying_threadpool_->Schedule(std::move(fn));
}

int ThreadPool::NumShardsUsedByFixedBlockSizeScheduling(const int64_t total, const int64_t block_size) {
  if (block_size <= 0 || total <= 1 || total <= block_size || NumThreads() == 1) {
    return 1;
  }
  // TODO:check overflow?
  return static_cast<int>((total + block_size - 1) / block_size);
}

void ThreadPool::ParallelFor(std::ptrdiff_t total, const SchedulingParams& scheduling_params,
                   const std::function<void(std::ptrdiff_t, std::ptrdiff_t)>& fn) {
  switch (scheduling_params.strategy()) {
    case SchedulingStrategy::kAdaptive: {
      if (scheduling_params.cost_per_unit().has_value()) {
        ParallelFor(total, static_cast<double>(scheduling_params.cost_per_unit().value()), fn);
      }
      break;
    }
    case SchedulingStrategy::kFixedBlockSize: {
      if (scheduling_params.block_size().has_value()) {
        ParallelForFixedBlockSizeScheduling(total, scheduling_params.block_size().value(), fn);
      }
      break;
    }
  }
}


// This functionality is similar to parallelFor, except that reasoning about
// the number of shards used is significantly easier.
void ThreadPool::ParallelForFixedBlockSizeScheduling(const int64_t total, const int64_t block_size,
                                                     const std::function<void(ptrdiff_t, ptrdiff_t)>& fn) {
  const int num_shards_used = NumShardsUsedByFixedBlockSizeScheduling(total, block_size);
  if (num_shards_used == 1) {
    fn(0, total);
    return;
  }

  // Adapted from Eigen's parallelFor implementation.
  BlockingCounter counter(num_shards_used);
  std::function<void(ptrdiff_t, ptrdiff_t)> handle_range = [=, &handle_range, &counter, &fn](int64_t first,
                                                                                             int64_t last) {
    while (last - first > block_size) {
      // Find something near the midpoint which is a multiple of block size.
      const int64_t mid = first + ((last - first) / 2 + block_size - 1) / block_size * block_size;
      Schedule([=, &handle_range]() { handle_range(mid, last); });
      last = mid;
    }
    // Single block or less, execute directly.
    fn(first, last);
    counter.DecrementCount();  // The shard is done.
  };
  if (num_shards_used <= NumThreads()) {
    // Avoid a thread hop by running the root of the tree and one block on the
    // main thread.
    handle_range(0, total);
  } else {
    // Execute the root in the thread pool to avoid running work on more than
    // numThreads() threads.
    Schedule([=, &handle_range]() { handle_range(0, total); });
  }
  counter.Wait();
}

void ThreadPool::ParallelFor(std::ptrdiff_t total, const TensorOpCost& cost_per_unit,
                             const std::function<void(std::ptrdiff_t first, std::ptrdiff_t)>& fn) {
  static_assert(sizeof(onnxruntime::TensorOpCost) == sizeof(Eigen::TensorOpCost), "TensorOpCost size mismatch");
  threadpool_device_->parallelFor(total, *reinterpret_cast<const Eigen::TensorOpCost*>(&cost_per_unit), fn);
}
void ThreadPool::ParallelFor(std::ptrdiff_t total, double cost_per_unit,
                             const std::function<void(std::ptrdiff_t first, std::ptrdiff_t)>& fn) {
  ORT_ENFORCE(total >= 0);
  threadpool_device_->parallelFor(total, Eigen::TensorOpCost(0, 0, static_cast<double>(cost_per_unit)),
                                  [&fn](std::ptrdiff_t first, std::ptrdiff_t last) { fn(first, last); });
}

void ThreadPool::ParallelForWithWorkerId(std::ptrdiff_t total, int64_t cost_per_unit,
                                         const std::function<void(std::ptrdiff_t, std::ptrdiff_t, int)>& fn) {
  ORT_ENFORCE(total >= 0);
  ORT_ENFORCE(total == (int64_t)(std::ptrdiff_t)total);

  threadpool_device_->parallelFor(total, Eigen::TensorOpCost(0, 0, static_cast<double>(cost_per_unit)),
                                  [this, &fn](int64_t start, int64_t limit) {
                                    // ParallelFor may use the current thread to
                                    // do some work synchronously. When calling
                                    // CurrentThreadId() from outside of the
                                    // thread pool, we get -1, so we can shift
                                    // every id up by 1.
                                    int id = CurrentThreadId() + 1;
                                    fn(start, limit, id);
                                  });
}

void ThreadPool::ParallelForWithWorkerId(std::ptrdiff_t total, const SchedulingParams& scheduling_params,
                                         const std::function<void(std::ptrdiff_t, std::ptrdiff_t, int)>& fn) {
  ParallelFor(total, scheduling_params, [this, &fn](int64_t start, int64_t limit) {
    // We may use the current thread to do some work synchronously.
    // When calling CurrentThreadId() from outside of the thread
    // pool, we get -1, so we can shift every id up by 1.
    int id = CurrentThreadId() + 1;
    fn(start, limit, id);
  });
}

int ThreadPool::NumThreads() const {
  return underlying_threadpool_->NumThreads();
}

int ThreadPool::CurrentThreadId() const {
  return underlying_threadpool_->CurrentThreadId();
}

void ThreadPool::ScheduleWithHint(std::function<void()> fn, int start, int limit) {
  underlying_threadpool_->ScheduleWithHint(std::move(fn), start, limit);
}

void ThreadPool::SetStealPartitions(const std::vector<std::pair<unsigned, unsigned>>& partitions) {
  // ThreadPool::SetStealPartitions is only called in the constructor of
  // RunHandlerPool::Impl, which currently instantiates ThreadPool using a
  // constructor that does not take user_threadpool. Thus we assume
  // eigen_threadpool_ is not null here.
  ORT_ENFORCE(eigen_threadpool_ != nullptr);
  eigen_threadpool_->SetStealPartitions(partitions);
}

Eigen::ThreadPoolInterface* ThreadPool::AsEigenThreadPool() const {
  ORT_ENFORCE(underlying_threadpool_ != nullptr);
  return underlying_threadpool_;
}
}  // namespace concurrency
}  // namespace onnxruntime