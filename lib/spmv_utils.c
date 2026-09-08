#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "../include/spmv_utils.h"
#include "../include/spmv_type.h"

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

// Helper function to count unique column indices
static size_t count_unique_columns(int nnz, int m, const int *col_indices) {
    int *unique_cols = (int*)calloc(m, sizeof(int));
    size_t unique_count = 0;
    
    if (unique_cols != NULL) {
        for (int i = 0; i < nnz; i++) {
            if (col_indices[i] >= 0 && col_indices[i] < m) {
                if (unique_cols[col_indices[i]] == 0) {
                    unique_cols[col_indices[i]] = 1;
                    unique_count++;
                }
            }
        }
        free(unique_cols);
    } else {
        // Fallback: assume all vector elements are accessed
        unique_count = m;
    }
    
    return unique_count;
}

// Generic bandwidth calculation function
static void calculate_bandwidth_generic(int n, int m, int nnz, const int *col_indices,
                                       double avg_time, size_t extra_bytes_read,
                                       double *bandwidth, double *gflops) {
    // Input validation
    if (avg_time <= 0.0 || !col_indices || n <= 0 || m <= 0 || nnz < 0) {
        *bandwidth = 0.0;
        *gflops = 0.0;
        return;
    }
    
    // Calculate memory access patterns
    size_t bytes_read_vals_cols = (size_t)nnz * (sizeof(dtype) + sizeof(int));
    size_t bytes_read_row_ptr = (size_t)(n + 1) * sizeof(int);
    size_t unique_count = count_unique_columns(nnz, m, col_indices);
    size_t bytes_read_vec = unique_count * sizeof(dtype);
    
    // Total memory traffic
    size_t bytes_read = bytes_read_vals_cols + bytes_read_row_ptr + bytes_read_vec + extra_bytes_read;
    size_t bytes_written = (size_t)n * sizeof(dtype);
    size_t total_bytes = bytes_read + bytes_written;

    // Calculate performance metrics
    *bandwidth = (double)total_bytes / (avg_time * 1.0e9);
    *gflops = (2.0 * nnz) / (avg_time * 1.0e9);
}

// Comparison function for qsort
static int compare_ints(const void *a, const void *b) {
    int ia = *(const int*)a;
    int ib = *(const int*)b;
    return (ia > ib) - (ia < ib);
}

struct MAT_STATS calculate_matrix_stats(const struct CSR *csr_matrix) {
    struct MAT_STATS stats = {0};
    
    if (!csr_matrix || !csr_matrix->row_pointers || csr_matrix->num_rows <= 0) {
        return stats;
    }
    
    int rows = csr_matrix->num_rows;
    int cols = csr_matrix->num_cols;
    const int *csr_row_ptr = csr_matrix->row_pointers;
    
    int *row_nnz = malloc(rows * sizeof(int));
    if (!row_nnz) {
        return stats;
    }
    
    // Calculate NNZ per row and basic statistics
    for (int i = 0; i < rows; i++) {
        row_nnz[i] = csr_row_ptr[i + 1] - csr_row_ptr[i];
        stats.total_nnz += row_nnz[i];
        
        if (row_nnz[i] == 0) {
            stats.empty_rows++;
        }
        
        // Update min/max on the fly
        if (i == 0 || row_nnz[i] < stats.min_nnz_per_row) {
            stats.min_nnz_per_row = row_nnz[i];
        }
        if (i == 0 || row_nnz[i] > stats.max_nnz_per_row) {
            stats.max_nnz_per_row = row_nnz[i];
        }
    }
    
    // Calculate derived statistics
    stats.mean_nnz_per_row = (double)stats.total_nnz / rows;
    
    // Calculate variance
    double variance = 0.0;
    for (int i = 0; i < rows; i++) {
        double diff = row_nnz[i] - stats.mean_nnz_per_row;
        variance += diff * diff;
    }
    stats.std_dev_nnz_per_row = sqrt(variance / rows);
    
    // Calculate median
    qsort(row_nnz, rows, sizeof(int), compare_ints);
    stats.median_nnz_per_row = row_nnz[rows / 2];
    
    // Calculate sparsity ratio
    long long total_elements = (long long)rows * cols;
    stats.sparsity_ratio = 1.0 - ((double)stats.total_nnz / total_elements);
    
    free(row_nnz);
    return stats;
}

void print_matrix_stats(const struct CSR *csr_matrix) {
    if (!csr_matrix) {
        printf("Error: NULL CSR matrix provided\n");
        return;
    }
    
    struct MAT_STATS stats = calculate_matrix_stats(csr_matrix);
    
    if (stats.total_nnz == 0) {
        printf("Error: Unable to calculate matrix statistics\n");
        return;
    }
    
    printf("\n=== Matrix Statistics ===\n");
    printf("Matrix dimensions: %d x %d\n", csr_matrix->num_rows, csr_matrix->num_cols);
    printf("Total NNZ: %d\n", stats.total_nnz);
    printf("Mean NNZ per Row: %.2f\n", stats.mean_nnz_per_row);
    printf("Standard Deviation NNZ per Row: %.2f\n", stats.std_dev_nnz_per_row);
    printf("Min NNZ per Row: %d, Max NNZ per Row: %d\n", 
           stats.min_nnz_per_row, stats.max_nnz_per_row);
    printf("Median NNZ per Row: %d\n", stats.median_nnz_per_row);
    printf("Sparsity Ratio: %.4f (%.2f%% sparse)\n", 
           stats.sparsity_ratio, stats.sparsity_ratio * 100.0);
    printf("========================\n\n");
}

void print_spmv_performance(const char* implementation_name, const char* matrix_path,
                           int n, int m, int nnz, double avg_time, double bandwidth,
                           double gflops, const dtype* result_vector, int max_samples) {
    // Input validation
    if (!implementation_name || !matrix_path || !result_vector || 
        n <= 0 || m <= 0 || nnz < 0 || avg_time <= 0 || max_samples <= 0) {
        printf("Error: Invalid parameters for performance printing\n");
        return;
    }
    
    // Extract matrix name from path
    const char* matrix_name = strrchr(matrix_path, '/');
    matrix_name = matrix_name ? matrix_name + 1 : matrix_path;
    
    // Calculate additional statistics
    double avg_nnz_per_row = (double)nnz / n;
    double percentage_nnz = ((double)nnz / ((long long)n * m)) * 100.0;
    
    printf("\n=== SpMV Performance Results ===\n");
    printf("Implementation: %s\n", implementation_name);
    printf("Matrix: %s\n", matrix_name);
    printf("Matrix size: %d x %d with %d non-zero elements\n", n, m, nnz);
    printf("Average NNZ per Row: %.2f\n", avg_nnz_per_row);
    printf("Percentage of NNZ: %.4f%%\n", percentage_nnz);
    printf("Average execution time: %.6f seconds\n", avg_time);
    printf("Memory bandwidth (estimated): %.4f GB/s\n", bandwidth);
    printf("Computational performance: %.6f GFLOPS\n", gflops);
    
    // Print sample of result vector
    printf("\nFirst few non-zero elements of result vector:\n");
    int count = 0;
    for (int i = 0; i < n && count < max_samples; i++) {
        if (result_vector[i] != 0.0) {
            printf("%.6f ", result_vector[i]);
            count++;
        }
    }
    
    if (count == 0) {
        printf("Result vector is all zeros or first %d elements are zero.", max_samples);
    }
    printf("\n===============================\n\n");
}

void calculate_hybrid_bandwidth(int n, int m, int nnz, const int *col_indices, 
                               int num_short, int num_long, double avg_time, 
                               double *bandwidth, double *gflops) {
    // Calculate extra bytes for hybrid approach (row arrays)
    size_t extra_bytes = (size_t)num_short * sizeof(int) + (size_t)num_long * sizeof(int);
    calculate_bandwidth_generic(n, m, nnz, col_indices, avg_time, extra_bytes, bandwidth, gflops);
}

void calculate_bandwidth(int n, int m, int nnz, const int *col_indices, 
                        double avg_time, double *bandwidth, double *gflops) {
    // No extra bytes for standard approach
    calculate_bandwidth_generic(n, m, nnz, col_indices, avg_time, 0, bandwidth, gflops);
}


void calculate_bandwidth_with_extra(int n, int m, int nnz, const int *col_indices,
                                    double avg_time, size_t extra_bytes_read,
                                    double *bandwidth, double *gflops) {
    calculate_bandwidth_generic(n, m, nnz, col_indices, avg_time, extra_bytes_read,
                                bandwidth, gflops);
}

static int compare_doubles(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) - (da < db);
}

double median_of(const double *values, int count) {
    if (!values || count <= 0) return 0.0;
    double *sorted = malloc(count * sizeof(double));
    if (!sorted) return values[0];
    memcpy(sorted, values, count * sizeof(double));
    qsort(sorted, count, sizeof(double), compare_doubles);
    double median = (count % 2 == 1)
        ? sorted[count / 2]
        : 0.5 * (sorted[count / 2 - 1] + sorted[count / 2]);
    free(sorted);
    return median;
}

int verify_spmv_result(const struct CSR *csr, const dtype *vec,
                       const dtype *result, double rel_tol,
                       double *max_rel_err, int *num_bad_rows) {
    if (!csr || !csr->row_pointers || !csr->col_indices || !csr->values ||
        !vec || !result || csr->num_rows <= 0 || rel_tol <= 0.0) {
        return -1;
    }

    const int n = csr->num_rows;
    const int *row_ptr = csr->row_pointers;
    const int *cols = csr->col_indices;
    const dtype *vals = csr->values;
    double worst = 0.0;
    int bad = 0;

    #pragma omp parallel for reduction(max:worst) reduction(+:bad) schedule(dynamic, 4096)
    for (int i = 0; i < n; i++) {
        double ref = 0.0, scale = 0.0;
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
            double term = (double)vals[j] * (double)vec[cols[j]];
            ref += term;
            scale += fabs(term);
        }
        double err = fabs(ref - (double)result[i]);
        if (scale > 0.0) err /= scale;
        if (err > worst) worst = err;
        if (err > rel_tol) bad++;
    }

    if (max_rel_err) *max_rel_err = worst;
    if (num_bad_rows) *num_bad_rows = bad;
    return bad ? 1 : 0;
}

int verify_and_report(const char *implementation_name, const struct CSR *csr,
                      const dtype *vec, const dtype *result) {
    double max_err = 0.0;
    int bad = 0;
    int status = verify_spmv_result(csr, vec, result, SPMV_VERIFY_REL_TOL,
                                    &max_err, &bad);
    const char *name = implementation_name ? implementation_name : "SpMV";
    if (status < 0) {
        printf("Verification (%s): SKIPPED (invalid input)\n", name);
    } else {
        printf("Verification (%s): %s (max relative error %.3e, %d row(s) above %.1e)\n",
               name, status == 0 ? "PASS" : "FAIL", max_err, bad, SPMV_VERIFY_REL_TOL);
    }
    return status;
}
