#ifndef SPMV_UTILS_H
#define SPMV_UTILS_H

#include <stddef.h>
#include "spmv_type.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Determine block distribution for adaptive CSR SpMV
 *
 * @param csr_row_ptr Row pointers array from CSR matrix
 * @param rows Number of rows
 * @param row_blocks Array to store row block indices (preallocated)
 * @param warp_size Size of a warp
 * @param block_size Size of a block
 * @return Number of row blocks
 */
int adaptive_row_selection(const int *csr_row_ptr, int rows, int *row_blocks, int warp_size, int block_size);

/**
 * Calculate bandwidth and GFLOPS for standard SpMV operations
 *
 * @param n Number of rows
 * @param m Number of columns
 * @param nnz Number of non-zero elements
 * @param col_indices Column indices array
 * @param avg_time Average execution time in seconds
 * @param bandwidth Output: Memory bandwidth in GB/s
 * @param gflops Output: Computational performance in GFLOPS
 */
void calculate_bandwidth(int n, int m, int nnz, const int *col_indices, 
                        double avg_time, double *bandwidth, double *gflops);

/**
 * Calculate bandwidth and GFLOPS for hybrid SpMV operations
 *
 * @param n Number of rows
 * @param m Number of columns
 * @param nnz Number of non-zero elements
 * @param col_indices Column indices array
 * @param num_short Number of short rows
 * @param num_long Number of long rows
 * @param avg_time Average execution time in seconds
 * @param bandwidth Output: Memory bandwidth in GB/s
 * @param gflops Output: Computational performance in GFLOPS
 */
void calculate_hybrid_bandwidth(int n, int m, int nnz, const int *col_indices, 
                               int num_short, int num_long, double avg_time, 
                               double *bandwidth, double *gflops);

/**
 * Calculate bandwidth and GFLOPS for hybrid SpMV operations
 *
 * @param n Number of rows
 * @param m Number of columns
 * @param nnz Number of non-zero elements
 * @param col_indices Column indices array
 * @param num_short Number of short rows
 * @param num_long Number of long rows
 * @param avg_time Average execution time in seconds
 * @param bandwidth Output: Memory bandwidth in GB/s
 * @param gflops Output: Computational performance in GFLOPS
 */
void calculate_adaptive_bandwidth(int n, int m, int nnz, const int *col_indices,
                                int optimal_num_blocks, double avg_time,
                                double *bandwidth, double *gflops);

/**
 * Bandwidth/GFLOPS with an explicit amount of extra bytes read per SpMV
 * (row lists, chunk tables, ...), for kernels with their own metadata.
 */
void calculate_bandwidth_with_extra(int n, int m, int nnz, const int *col_indices,
                                    double avg_time, size_t extra_bytes_read,
                                    double *bandwidth, double *gflops);

/** Median of `count` doubles (input is left untouched). Returns 0 if count <= 0. */
double median_of(const double *values, int count);

/** Default relative tolerance for verify_spmv_result. */
#define SPMV_VERIFY_REL_TOL 1e-3

/**
 * Compares `result` with a double-precision CPU SpMV of csr * vec.
 * The error of a row is |ref - got| / sum_j |a_ij * x_j| (absolute error
 * when that sum is zero).
 *
 * @param max_rel_err  receives the largest row error (may be NULL)
 * @param num_bad_rows receives the number of rows above rel_tol (may be NULL)
 * @return 0 if all rows are within rel_tol, 1 otherwise, -1 on invalid input
 */
int verify_spmv_result(const struct CSR *csr, const dtype *vec,
                       const dtype *result, double rel_tol,
                       double *max_rel_err, int *num_bad_rows);

/** Runs verify_spmv_result with the default tolerance and prints a PASS/FAIL line. */
int verify_and_report(const char *implementation_name, const struct CSR *csr,
                      const dtype *vec, const dtype *result);

/**
 * Calculate matrix statistics for optimization decisions
 *
 * @param csr_matrix CSR matrix structure
 * @return Matrix statistics structure
 */
struct MAT_STATS calculate_matrix_stats(const struct CSR *csr_matrix);

/**
 * Print matrix statistics for profiling and understanding
 *
 * @param matrix CSR matrix structure
 */
void print_matrix_stats(const struct CSR *matrix);

/**
 * Print standardized performance information for SpMV operations
 * 
 * @param implementation_name Name of the SpMV implementation (e.g., "GPU Simple CSR")
 * @param matrix_path Path to the matrix file used
 * @param n Number of rows
 * @param m Number of columns
 * @param nnz Number of non-zero elements
 * @param avg_time Average execution time in seconds
 * @param bandwidth Memory bandwidth in GB/s
 * @param gflops Computational performance in GFLOPS
 * @param result_vector Pointer to the result vector to print samples from
 * @param max_samples Maximum number of non-zero samples to print
 */
void print_spmv_performance(
    const char* implementation_name,
    const char* matrix_path,
    int n, 
    int m, 
    int nnz, 
    double avg_time, 
    double bandwidth, 
    double gflops, 
    const dtype* result_vector,
    int max_samples
);

#ifdef __cplusplus
}
#endif

#endif // SPMV_UTILS_H