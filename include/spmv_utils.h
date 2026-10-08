#ifndef SPMV_UTILS_H
#define SPMV_UTILS_H

#include "spmv_type.h"

#ifdef __cplusplus
extern "C" {
#endif

void calculate_bandwidth(int n, int m, int nnz, const int *col_indices,
                         double avg_time, double *bandwidth, double *gflops);

void calculate_hybrid_bandwidth(int n, int m, int nnz, const int *col_indices,
                                int num_short, int num_long, double avg_time,
                                double *bandwidth, double *gflops);

#define SPMV_VERIFY_REL_TOL 1e-3

int verify_spmv_result(const struct CSR *csr, const dtype *vec,
                       const dtype *result, double rel_tol,
                       double *max_rel_err, int *num_bad_rows);

int verify_and_report(const char *implementation_name, const struct CSR *csr,
                      const dtype *vec, const dtype *result);

struct MAT_STATS calculate_matrix_stats(const struct CSR *csr_matrix);

void print_matrix_stats(const struct CSR *matrix);

void print_spmv_performance(const char *implementation_name,
                            const char *matrix_path, int n, int m, int nnz,
                            double avg_time, double bandwidth, double gflops,
                            const dtype *result_vector, int max_samples);

#ifdef __cplusplus
}
#endif

#endif // SPMV_UTILS_H
