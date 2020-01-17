// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/atomic/common.cuh"
#include "reduction_utils.cuh"
#include "reduction_all.h"

namespace onnxruntime {
namespace cuda {

template<typename T>
__global__ void _ScalarSqrtImpl(T* input, T* output) {
  *output = _Sqrt(*input);
};

template<typename T>
void ScalarSqrt(T* input, T* output) {
  _ScalarSqrtImpl<<<1, 1, 0>>>(input, output);
};

template void ScalarSqrt(float* input, float* output);
template void ScalarSqrt(half* input, half* output);

template <typename TIn, typename TOut, typename TBuf, typename TInOp, typename TOutOp>
__global__ void _MultiTensorReduceImpl(ChunkGroup<1> chunk_group, TOut* output) {
  const int group_index = chunk_group.block_index_to_tensor_group_index[blockIdx.x];
  const int tensor_size = chunk_group.tensor_sizes[group_index];
  const int chunk_size = chunk_group.chunk_size;
  const int chunk_start = chunk_group.block_index_to_chunk_start_index[blockIdx.x];
  const TIn* w = reinterpret_cast<const TIn*>(chunk_group.tensor_ptrs[0][group_index]) + chunk_start;
  TOut* w_norm = output;

  TBuf w_sum = TBuf(0.f);
  constexpr int load_count_per_thread = 4;
  for (int i = threadIdx.x; i < chunk_size && i + chunk_start < tensor_size; i += blockDim.x * load_count_per_thread) {
#pragma unroll
    for (int j = 0; j < load_count_per_thread; ++j) {
      const int index_in_chunk = i + j * blockDim.x;
      const int index_in_tensor = chunk_start + index_in_chunk;
      if (index_in_chunk < chunk_size && index_in_tensor < tensor_size) {
        const TBuf w_element = TBuf(w[index_in_chunk]);
        w_sum += TInOp()(w_element);
      }
    }
  }

  // Thread count in a block must be a multiple of 32.
  constexpr int warp_size = 32;
#pragma unroll
  for (int stride = warp_size / 2; stride > 0; stride /= 2) {
    w_sum += __shfl_down_sync(0xFFFFFFFF, w_sum, stride);
  }

  const int warp_count_in_block = blockDim.x / warp_size;
  const int lid = threadIdx.x % warp_size;
  const int wid = threadIdx.x / warp_size;

  // Shape is 2 x warp_count_in_block.
  extern __shared__ unsigned char shared_memory_[];
  TBuf* shared_memory = reinterpret_cast<TBuf*>(shared_memory_);

  if (lid == 0) {
    shared_memory[wid] = w_sum;
  }

  __syncthreads();

#pragma unroll
  for (int stride = warp_count_in_block / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      shared_memory[threadIdx.x] += shared_memory[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    atomic_add(w_norm, TOutOp()(shared_memory[0]));
  }
};

template <typename TIn, typename TOut, typename TBuf, typename TInOp, typename TOutOp>
void MultiTensorReduce(ChunkGroup<1> chunk_group, TOut* output) {
  // thread count per block.
  constexpr int thread_count = ChunkGroup<1>::thread_count_per_block;
  // warp size of GPU.
  constexpr int warp_size = 32;
  // shared memory's size per block.
  const int shared_memory_size = thread_count / warp_size * sizeof(TBuf);

  // Enforce assumptions used inside this reduction CUDA kernel.
  ORT_ENFORCE(thread_count % warp_size == 0);
  ORT_ENFORCE((thread_count & (thread_count - 1)) == 0);

  _MultiTensorReduceImpl<TIn, TOut, TBuf, TInOp, TOutOp><<<chunk_group.chunk_count, thread_count, shared_memory_size>>>(chunk_group, output);
}

template <typename TIn, typename TOut>
void MultiTensorReduceL2<TIn, TOut>::operator()(ChunkGroup<1> chunk_group, TOut* output) {
  typedef typename ToBuffer<TIn>::Type TBuf;
  MultiTensorReduce<TIn, TOut, TBuf, Square<TBuf, TIn>, Cast<TOut, TBuf>>(chunk_group, output);
}

#define INSTANTIATE_MULTI_TENSOR_REDUCTION_L2_FUNCTOR(TIn, TOut) \
  template void MultiTensorReduceL2<TIn, TOut>::operator()(ChunkGroup<1> chunk_group, TOut* output);

INSTANTIATE_MULTI_TENSOR_REDUCTION_L2_FUNCTOR(double, float)
INSTANTIATE_MULTI_TENSOR_REDUCTION_L2_FUNCTOR(float, float)
INSTANTIATE_MULTI_TENSOR_REDUCTION_L2_FUNCTOR(half, float)
INSTANTIATE_MULTI_TENSOR_REDUCTION_L2_FUNCTOR(float, half)
INSTANTIATE_MULTI_TENSOR_REDUCTION_L2_FUNCTOR(half, half)

}  // namespace cuda
}  // namespace onnxruntime