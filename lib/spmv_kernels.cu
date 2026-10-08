#include "../include/spmv_kernels.h"

// Hybrid SpMV kernels: adaptive CSR only.
// Older kernels (simple, value-parallel, vector) were removed;
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
