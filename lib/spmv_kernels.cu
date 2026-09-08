#include "../include/spmv_kernels.h"

// Hybrid SpMV kernels only: hybrid adaptive (v1) and hybrid v2.
// Older kernels (simple, value-parallel, vector, adaptive) were removed;
// see git history for their implementations.

// Hybrid kernel that uses pre-classified row arrays
__global__ void hybrid_adaptive_spmv_optimized(
    const dtype *csr_values, const int *csr_row_ptr, const int *csr_col_indices,
    const dtype *vec, dtype *res, int n, const int *short_rows,
    const int *long_rows, int num_short, int num_long, int short_blocks) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int lane_id = tid & 31;

  // Phase 1: Handle short rows with thread-per-row
  if (blockIdx.x < short_blocks) { // First half of blocks for short rows
    int row_idx = tid;
    if (row_idx < num_short) {
      int row = short_rows[row_idx];
      int start = csr_row_ptr[row];
      int end = csr_row_ptr[row + 1];
      int row_length = end - start;

      if (row_length == 0) {
        res[row] = 0.0;
        return;
      }

      dtype sum = 0.0;

      // Simple unrolled loop for better performance
      int j = start;
      for (; j + 3 < end; j += 4) {
        sum += csr_values[j] * __ldg(&vec[csr_col_indices[j]]);
        sum += csr_values[j + 1] * __ldg(&vec[csr_col_indices[j + 1]]);
        sum += csr_values[j + 2] * __ldg(&vec[csr_col_indices[j + 2]]);
        sum += csr_values[j + 3] * __ldg(&vec[csr_col_indices[j + 3]]);
      }

      // Handle remaining elements
      for (; j < end; j++) {
        sum += csr_values[j] * __ldg(&vec[csr_col_indices[j]]);
      }

      res[row] = sum;
    }
  }
  // Phase 2: Handle long rows with warp-per-row
  else {
    int warp_id =
        (blockIdx.x - short_blocks) * (blockDim.x >> 5) + (threadIdx.x >> 5);

    if (warp_id < num_long) {
      int row = long_rows[warp_id];
      int start = csr_row_ptr[row];
      int end = csr_row_ptr[row + 1];

      dtype thread_sum = 0.0;

      // Coalesced memory access with stride
      for (int j = start + lane_id; j < end; j += 32) {
        thread_sum += csr_values[j] * __ldg(&vec[csr_col_indices[j]]);
      }

      // Optimized warp reduction using __shfl_down_sync
      thread_sum += __shfl_down_sync(0xFFFFFFFF, thread_sum, 16);
      thread_sum += __shfl_down_sync(0xFFFFFFFF, thread_sum, 8);
      thread_sum += __shfl_down_sync(0xFFFFFFFF, thread_sum, 4);
      thread_sum += __shfl_down_sync(0xFFFFFFFF, thread_sum, 2);
      thread_sum += __shfl_down_sync(0xFFFFFFFF, thread_sum, 1);

      if (lane_id == 0) {
        res[row] = thread_sum;
      }
    }
  }
}

//------------------------ HYBRID V2 ----------------------------------------

int hybrid_v2_rows_per_block(int block_size, int lane_class) {
  return (block_size / WARP_SIZE) * (WARP_SIZE >> lane_class);
}

// Sum of the row segment [start, end) over a group of L consecutive lanes.
// Every lane of the warp must call this (full-mask shuffles); lanes with an
// empty range contribute zero. The result is valid in every lane of the group.
template <int L>
__device__ __forceinline__ dtype hv2_group_sum(const dtype *vals,
                                               const int *cols,
                                               const dtype *vec, int start,
                                               int end, int lane_in_group) {
  dtype sum = 0;
  for (int j = start + lane_in_group; j < end; j += L) {
    sum += vals[j] * __ldg(&vec[cols[j]]);
  }
#pragma unroll
  for (int offset = L / 2; offset > 0; offset >>= 1) {
    sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset);
  }
  return sum;
}

// One block of lane class C (L = 1 << C lanes per row). In DIRECT mode the
// block walks rows [first, first + rows_per_block) of the matrix and skips
// rows that belong to another class; otherwise rows come from `rows_list`.
template <int C, bool DIRECT>
__device__ __forceinline__ void
hv2_lane_class_block(const dtype *vals, const int *row_ptr, const int *cols,
                     const dtype *vec, dtype *res, const int *rows_list,
                     int count, int local_block, int huge_threshold) {
  constexpr int L = 1 << C;
  constexpr int GROUPS_PER_WARP = WARP_SIZE / L;

  const int lane = threadIdx.x & (WARP_SIZE - 1);
  const int warp = threadIdx.x >> 5;
  const int lane_in_group = lane & (L - 1);
  const int group = lane >> C;
  const int rows_per_block = (blockDim.x >> 5) * GROUPS_PER_WARP;
  const int idx = local_block * rows_per_block + warp * GROUPS_PER_WARP + group;

  int row = -1;
  int start = 0;
  int end = 0;
  if (idx < count) {
    row = DIRECT ? idx : rows_list[idx];
    start = row_ptr[row];
    end = row_ptr[row + 1];
    if (DIRECT && hv2_row_class(end - start, huge_threshold) != C) {
      row = -1; // belongs to another class: contribute nothing, write nothing
      end = start;
    }
  }

  dtype sum = hv2_group_sum<L>(vals, cols, vec, start, end, lane_in_group);

  if (row >= 0 && lane_in_group == 0) {
    res[row] = sum;
  }
}

// Block-wide sum; the result is valid in thread 0 only.
__device__ __forceinline__ dtype hv2_block_sum(dtype value,
                                               dtype *warp_sums) {
  const int lane = threadIdx.x & (WARP_SIZE - 1);
  const int warp = threadIdx.x >> 5;
  const int num_warps = blockDim.x >> 5;

#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(0xFFFFFFFF, value, offset);
  }
  if (lane == 0) {
    warp_sums[warp] = value;
  }
  __syncthreads();

  dtype total = 0;
  if (warp == 0) {
    total = (lane < num_warps) ? warp_sums[lane] : 0;
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
      total += __shfl_xor_sync(0xFFFFFFFF, total, offset);
    }
  }
  return total;
}

// One block reduces one chunk [start, end) of a huge row. Values and column
// indices are streamed exactly once, so they bypass L1 (__ldcs) and leave it
// to the gathers from vec.
__device__ __forceinline__ void hv2_chunk_block(const dtype *vals,
                                                const int *cols,
                                                const dtype *vec, int start,
                                                int end, dtype *partial_out,
                                                dtype *warp_sums) {
  dtype sum = 0;
  for (int j = start + threadIdx.x; j < end; j += blockDim.x) {
    sum += __ldcs(&vals[j]) * __ldg(&vec[__ldcs(&cols[j])]);
  }
  sum = hv2_block_sum(sum, warp_sums);
  if (threadIdx.x == 0) {
    *partial_out = sum;
  }
}

#define HV2_DISPATCH_CLASS(C)                                                  \
  if (direct) {                                                                \
    hv2_lane_class_block<C, true>(csr_values, csr_row_ptr, csr_col_indices,   \
                                  vec, res, nullptr, args.n, local_block,     \
                                  args.huge_threshold);                        \
  } else {                                                                     \
    hv2_lane_class_block<C, false>(csr_values, csr_row_ptr, csr_col_indices,  \
                                   vec, res, args.class_rows[C],              \
                                   args.class_count[C], local_block,          \
                                   args.huge_threshold);                       \
  }

__global__ void hybrid_v2_spmv(const dtype *csr_values, const int *csr_row_ptr,
                               const int *csr_col_indices, const dtype *vec,
                               dtype *res, dtype *chunk_partials,
                               HV2DeviceArgs args) {
  __shared__ dtype warp_sums[WARP_SIZE];
  const int block = blockIdx.x;

  // Segment 0: chunks of huge rows (largest work units, scheduled first).
  if (block < args.block_begin[1]) {
    hv2_chunk_block(csr_values, csr_col_indices, vec, args.chunk_start[block],
                    args.chunk_end[block], &chunk_partials[block], warp_sums);
    return;
  }

  // Segments 1..6: lane classes, widest first. Uniform per block.
  int segment = 1;
  while (segment < HV2_NUM_LANE_CLASSES && block >= args.block_begin[segment + 1]) {
    segment++;
  }
  const int lane_class = HV2_NUM_LANE_CLASSES - segment;
  const int local_block = block - args.block_begin[segment];
  const bool direct = (lane_class == args.direct_class);

  switch (lane_class) {
  case 5:
    HV2_DISPATCH_CLASS(5);
    break;
  case 4:
    HV2_DISPATCH_CLASS(4);
    break;
  case 3:
    HV2_DISPATCH_CLASS(3);
    break;
  case 2:
    HV2_DISPATCH_CLASS(2);
    break;
  case 1:
    HV2_DISPATCH_CLASS(1);
    break;
  default:
    HV2_DISPATCH_CLASS(0);
    break;
  }
}

#undef HV2_DISPATCH_CLASS

__global__ void hybrid_v2_finalize(const dtype *chunk_partials,
                                   const int *huge_rows,
                                   const int *huge_chunk_begin, int num_huge,
                                   dtype *res) {
  const int global_warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & (WARP_SIZE - 1);
  if (global_warp >= num_huge) {
    return; // whole warp exits together: no shuffle below is reached partially
  }

  const int begin = huge_chunk_begin[global_warp];
  const int end = huge_chunk_begin[global_warp + 1];
  dtype sum = 0;
  for (int k = begin + lane; k < end; k += WARP_SIZE) {
    sum += chunk_partials[k];
  }
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset);
  }
  if (lane == 0) {
    res[huge_rows[global_warp]] = sum;
  }
}
