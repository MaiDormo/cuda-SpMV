#ifndef SPMV_KERNELS_H
#define SPMV_KERNELS_H

#include "spmv_type.h"

#define WARP_SIZE 32

/**
 * @brief Hybrid adaptive kernel: short rows use thread-per-row,
 * long rows use warp-per-row.
 */
__global__ void hybrid_adaptive_spmv_optimized(const dtype *csr_values, const int *csr_row_ptr,
                                              const int *csr_col_indices, const dtype *vec,
                                              dtype *res, int n, const int *short_rows,
                                              const int *long_rows, int num_short,
                                              int num_long, int short_blocks);

#endif // SPMV_KERNELS_H
