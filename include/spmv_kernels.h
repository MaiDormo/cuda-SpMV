#ifndef SPMV_KERNELS_H
#define SPMV_KERNELS_H

#include "spmv_type.h"
#include "hybrid_v2_plan.h"

#define WARP_SIZE 32
#define BLOCK_SIZE 1024

/**
 * Simple CSR SpMV kernel that maps one thread to one row
 * Works best for matrices with few non-zeros per row
 */
__global__ void spmv_simple(const dtype *csr_values, const int *csr_row_ptr, 
                          const int *csr_col_indices, const dtype *vec, 
                          dtype *res, int num_rows);

/**
 * @brief kernel that maps each thread to each non-zero element
 * and later uses atomic add in order to calculate the result
 */
__global__ void value_parallel_sequential_spmv(const dtype *csr_values, const int *csr_row_ptr, 
    const int *csr_col_indices, const dtype *vec, 
    dtype *res, int nnz, int num_rows);

/**
 * @brief kernel that maps each thread to a consecutive block of element
 * so that each thread reads the lenght of stride element and only cares about those
 */

__global__ void value_parallel_blocked_spmv(const dtype *csr_values, const int *csr_row_ptr, 
    const int *csr_col_indices, const dtype *vec,
    dtype *res, int nnz, int num_rows, int stride);

//------------------------ NOT PART OF THE FIRST DELIVERABLE ----------------

/**
 * Standard vector-based CSR SpMV kernel
 * Each warp processes one row
 */
__global__ void vector_csr(const dtype *csr_values, const int *csr_row_ptr, 
                          const int *csr_col_indices, const dtype *vec, 
                          dtype *res, int n);

/**
 * Adaptive CSR SpMV kernel that handles both dense and sparse rows efficiently
 * Uses either a warp per row or an entire block per row depending on density
 */
__global__ void adaptive_csr(const dtype *csr_values, const int *csr_row_ptr,
                           const int *csr_col_indices, const dtype *vec,
                           dtype *res, const int *row_blocks, int n);

/**
 * @brief Double-buffered vector CSR kernel
 */
__global__ void vector_csr_double_buffer(const dtype *csr_values, const int *csr_row_ptr, 
                                        const int *csr_col_indices, const dtype *vec, 
                                        dtype *res, int n);

/**
 * @brief Hybrid adaptive kernel that employs both scalar and 
 * vector approaches based on matrix statistics and row lengths.
 */
__global__ void hybrid_adaptive_spmv_optimized(const dtype *csr_values, const int *csr_row_ptr,
                                              const int *csr_col_indices, const dtype *vec,
                                              dtype *res, int n, const int *short_rows,
                                              const int *long_rows, int num_short,
                                              int num_long, int short_blocks);

//------------------------ HYBRID V2 ----------------------------------------

/**
 * Device-side view of an HV2Plan (see hybrid_v2_plan.h), passed by value.
 *
 * Grid layout of hybrid_v2_spmv, in launch order:
 *   segment 0                 : one block per chunk of a huge row
 *   segment s = 1..6          : lane class c = HV2_NUM_LANE_CLASSES - s
 *                               (32 lanes per row first, 1 lane per row last)
 * block_begin[s] is the first block of segment s; block_begin[7] is the grid size.
 * The direct class (if any) has class_rows == NULL and walks all n rows.
 */
struct HV2DeviceArgs {
  const int *class_rows[HV2_NUM_LANE_CLASSES];
  int class_count[HV2_NUM_LANE_CLASSES];
  int block_begin[HV2_NUM_LANE_CLASSES + 2];
  const int *chunk_start;
  const int *chunk_end;
  int direct_class;
  int huge_threshold;
  int n;
};

/**
 * @brief Rows handled by one block for lane class `lane_class` (host helper).
 */
int hybrid_v2_rows_per_block(int block_size, int lane_class);

/**
 * @brief Multi-granularity CSR SpMV: 1..32 lanes per row chosen per row,
 * plus block-per-chunk processing of huge rows. chunk_partials must hold one
 * dtype per chunk; when the plan has huge rows, hybrid_v2_finalize must run
 * afterwards to write those rows of res.
 */
__global__ void hybrid_v2_spmv(const dtype *csr_values, const int *csr_row_ptr,
                               const int *csr_col_indices, const dtype *vec,
                               dtype *res, dtype *chunk_partials,
                               HV2DeviceArgs args);

/**
 * @brief Sums the chunk partials of every huge row (one warp per row).
 */
__global__ void hybrid_v2_finalize(const dtype *chunk_partials,
                                   const int *huge_rows,
                                   const int *huge_chunk_begin, int num_huge,
                                   dtype *res);


#endif // SPMV_KERNELS_H