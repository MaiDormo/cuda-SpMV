#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>
#include <algorithm>
#include <unordered_map>

#include "../include/read_file_lib.h"
#include "../include/spmv_type.h"
#include "../include/csr_conversion.h"
#include "../include/spmv_utils.h"
#include "../include/spmv_kernels.h"


// Function to classify rows and create row arrays
void classify_rows(const int *row_ptr, int n, int **short_rows, int **long_rows, 
                   int *num_short, int *num_long, int threshold) {
    
    // First pass: count short and long rows
    *num_short = 0;
    *num_long = 0;
    
    for (int i = 0; i < n; i++) {
        int row_length = row_ptr[i + 1] - row_ptr[i];
        if (row_length <= threshold) {
            (*num_short)++;
        } else {
            (*num_long)++;
        }
    }
    
    // Allocate arrays
    *short_rows = (int*)malloc(*num_short * sizeof(int));
    *long_rows = (int*)malloc(*num_long * sizeof(int));
    
    if (!*short_rows || !*long_rows) {
        printf("Error: Failed to allocate memory for row classification\n");
        return;
    }
    
    // Second pass: populate arrays
    int short_idx = 0, long_idx = 0;
    for (int i = 0; i < n; i++) {
        int row_length = row_ptr[i + 1] - row_ptr[i];
        if (row_length <= threshold) {
            (*short_rows)[short_idx++] = i;
        } else {
            (*long_rows)[long_idx++] = i;
        }
    }
}

// Updated launch configuration function with improved adaptive logic
void get_hybrid_launch_config(int n, int nnz, const int *row_ptr, 
                             int &blocks, int &threads, double &strategy_ratio,
                             int **short_rows, int **long_rows, 
                             int *num_short, int *num_long, int *short_blocks_limit) {
    
    // Use existing matrix analysis function
    struct CSR temp_csr;
    temp_csr.row_pointers = (int*)row_ptr;
    temp_csr.num_rows = n;
    temp_csr.num_non_zeros = nnz;
    
    struct MAT_STATS stats = calculate_matrix_stats(&temp_csr);
    
    printf("Matrix analysis:\n");
    printf("  Mean NNZ per row: %.2f\n", stats.mean_nnz_per_row);
    printf("  Std deviation: %.2f\n", stats.std_dev_nnz_per_row);
    printf("  Max NNZ per row: %d\n", stats.max_nnz_per_row);
    printf("  Empty rows: %d (%.2f%%)\n", stats.empty_rows, 100.0 * stats.empty_rows / n);
    
    // Improved adaptive configuration based on analysis results
    int threshold;
    
    if (stats.mean_nnz_per_row < 3.0) {
        // Very sparse matrices (like 662_bus, mawi)
        threads = 128;
        threshold = 16;  // Lower threshold for very sparse matrices
        printf("Strategy: Very sparse matrix - using 128 threads, threshold 16\n");
    }
    else if (stats.mean_nnz_per_row < 8.0) {
        // Sparse matrices (like CurlCurl_4)
        threads = 128;
        threshold = 16;  // Keep low threshold for moderate sparsity
        printf("Strategy: Sparse matrix - using 128 threads, threshold 16\n");
    }
    else if (stats.mean_nnz_per_row < 35.0) {
        // Medium density matrices (like Goodwin_127, Zd_Jac3_db)
        if (stats.std_dev_nnz_per_row > 50.0) {
            // High variance like Zd_Jac3_db
            threads = 256;
            threshold = 32;
            printf("Strategy: Medium density with high variance - using 256 threads, threshold 32\n");
        } else {
            // Regular medium density like Goodwin_127
            threads = 256;
            threshold = 256;  // Higher threshold for more uniform medium density
            printf("Strategy: Medium density matrix - using 256 threads, threshold 256\n");
        }
    }
    else {
        // Dense matrices (like ML_Geer)
        threads = 128;  // Fewer threads but higher threshold
        threshold = 256;
        printf("Strategy: Dense matrix - using 128 threads, threshold 256\n");
    }
    
    // Special case for extremely large matrices with very low density
    if (n > 100000000 && stats.mean_nnz_per_row < 2.0) {
        // Like mawi - huge but extremely sparse
        threads = 128;
        threshold = 32;  // Slightly higher threshold for better load balancing
        printf("Strategy: Extremely large sparse matrix - using 128 threads, threshold 32\n");
    }
    
    classify_rows(row_ptr, n, short_rows, long_rows, num_short, num_long, threshold);
    
    printf("Final configuration: THREADS=%d THRESHOLD=%d\n", threads, threshold);
    printf("Row classification (threshold=%d):\n", threshold);
    printf("  Short rows: %d (%.1f%%)\n", *num_short, 100.0 * (*num_short) / n);
    printf("  Long rows: %d (%.1f%%)\n", *num_long, 100.0 * (*num_long) / n);
    
    // Calculate launch configuration
    *short_blocks_limit = (*num_short + threads - 1) / threads;
    int long_blocks = (*num_long + (threads / WARP_SIZE) - 1) / (threads / WARP_SIZE);
    
    blocks = *short_blocks_limit + long_blocks;
    strategy_ratio = (double)*num_short / n;
    
    printf("Launch config: %d blocks (%d short + %d long), %d threads\n", 
           blocks, *short_blocks_limit, long_blocks, threads);
}

// Refactored function to calculate hybrid-specific bandwidth
double calculate_hybrid_bandwidth(int n, int m, int nnz, const int *col_indices, 
                                 int num_short, int num_long, double avg_time) {
    // Calculate memory bandwidth for hybrid approach - similar to adaptive approach
    size_t bytes_read_vals_cols = (size_t)nnz * (sizeof(dtype) + sizeof(int));  // CSR values + column indices
    size_t bytes_read_row_ptr = (size_t)(n + 1) * sizeof(int);                 // Row pointers (all accessed)
    
    // Row arrays access: each block reads its portion
    size_t bytes_read_short_rows = (size_t)num_short * sizeof(int);
    size_t bytes_read_long_rows = (size_t)num_long * sizeof(int);

    // Count unique column indices (more accurate than assuming all vector elements)
    int* unique_cols = (int*)calloc(m, sizeof(int));
    size_t unique_count = 0;
    for (int i = 0; i < nnz; i++) {
        if (unique_cols[col_indices[i]] == 0) {
            unique_cols[col_indices[i]] = 1;
            unique_count++;
        }
    }
    free(unique_cols);

    size_t bytes_read_vec = unique_count * sizeof(dtype);
    size_t bytes_read = bytes_read_vals_cols + bytes_read_row_ptr + bytes_read_vec + 
                       bytes_read_short_rows + bytes_read_long_rows;
    size_t bytes_written = (size_t)n * sizeof(dtype);
    size_t total_bytes = bytes_read + bytes_written;

    return total_bytes / (avg_time * 1.0e9);
}

// --- Main Function ---
int main(int argc, char ** argv) { 
    
    if (argc != 2) {
        printf("Usage: <./bin/spmv_hybrid_adaptive> <path/to/file.mtx>\n");
        return -1;
    }

    // --- Host Data Structures ---
    struct COO h_coo;
    struct CSR h_csr;
    dtype *h_vec = NULL;
    dtype *h_res = NULL;

    // --- Read Matrix ---
    read_from_file_and_init(argv[1], &h_coo);
    int n = h_coo.num_rows;
    int m = h_coo.num_cols;
    int nnz = h_coo.num_non_zeros;

    // --- Allocate Host Memory ---
    h_vec = (dtype*)malloc(m * sizeof(dtype));
    h_res = (dtype*)malloc(n * sizeof(dtype));
    h_csr.values = (dtype*)malloc(nnz * sizeof(dtype));
    h_csr.col_indices = (int*)malloc(nnz * sizeof(int));
    h_csr.row_pointers = (int*)calloc(n + 1, sizeof(int));

    if (!h_vec || !h_res || !h_csr.values || !h_csr.col_indices || !h_csr.row_pointers) {
        perror("Failed to allocate host memory");
        free(h_coo.a_val); free(h_coo.a_row); free(h_coo.a_col);
        free(h_vec); free(h_res);
        free(h_csr.values); free(h_csr.col_indices); free(h_csr.row_pointers);
        return -1;
    }

    // --- Initialize Host Vectors ---
    for (int i = 0; i < m; i++) h_vec[i] = 1.0;
    memset(h_res, 0, n * sizeof(dtype));

    // --- Convert COO to CSR ---
    if (coo_to_csr(&h_coo, &h_csr) != 0) {
         fprintf(stderr, "Error during COO to CSR conversion.\n");
         free(h_coo.a_val); free(h_coo.a_row); free(h_coo.a_col);
         free(h_vec); free(h_res);
         free(h_csr.values); free(h_csr.col_indices); free(h_csr.row_pointers);
         return -1;
    }

    // Free original COO data
    free(h_coo.a_val); h_coo.a_val = NULL;
    free(h_coo.a_row); h_coo.a_row = NULL;
    free(h_coo.a_col); h_coo.a_col = NULL;

    // --- Determine Hybrid Launch Configuration ---
    int hybrid_blocks, hybrid_threads;
    double strategy_ratio;
    int *h_short_rows = NULL, *h_long_rows = NULL;
    int num_short, num_long;
    int short_blocks;
    
    get_hybrid_launch_config(n, nnz, h_csr.row_pointers, 
                           hybrid_blocks, hybrid_threads, strategy_ratio,
                           &h_short_rows, &h_long_rows, &num_short, &num_long, &short_blocks);

    printf("hybrid threads: %d\n", hybrid_threads);
    // --- Allocate Device Memory for Row Arrays ---
    int *d_short_rows = NULL, *d_long_rows = NULL;
    
    if (num_short > 0) {
        cudaMalloc(&d_short_rows, num_short * sizeof(int));
        cudaMemcpy(d_short_rows, h_short_rows, num_short * sizeof(int), cudaMemcpyHostToDevice);
    }
    
    if (num_long > 0) {
        cudaMalloc(&d_long_rows, num_long * sizeof(int));
        cudaMemcpy(d_long_rows, h_long_rows, num_long * sizeof(int), cudaMemcpyHostToDevice);
    }

    // --- Device Data Structures ---
    struct CSR d_csr;
    dtype *d_vec = NULL, *d_res = NULL;

    // --- Allocate Device Memory ---
    cudaMalloc(&d_vec, m * sizeof(dtype));
    cudaMalloc(&d_res, n * sizeof(dtype));
    cudaMalloc(&d_csr.values, nnz * sizeof(dtype));
    cudaMalloc(&d_csr.col_indices, nnz * sizeof(int));
    cudaMalloc(&d_csr.row_pointers, (n + 1) * sizeof(int));
    d_csr.num_rows = n;
    d_csr.num_cols = m;
    d_csr.num_non_zeros = nnz;

    // --- Copy Data to Device ---
    cudaMemcpy(d_vec, h_vec, m * sizeof(dtype), cudaMemcpyHostToDevice);
    cudaMemcpy(d_res, h_res, n * sizeof(dtype), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csr.values, h_csr.values, nnz * sizeof(dtype), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csr.col_indices, h_csr.col_indices, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csr.row_pointers, h_csr.row_pointers, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);

    // --- Timing Setup ---
    const int NUM_RUNS = 50;
    dtype total_time = 0.0;
    dtype times[NUM_RUNS];
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    // --- Warmup Run ---
    hybrid_adaptive_spmv_optimized<<<hybrid_blocks, hybrid_threads>>>(
        d_csr.values, d_csr.row_pointers, d_csr.col_indices, d_vec, d_res, n,
        d_short_rows, d_long_rows, num_short, num_long, short_blocks
    );
    
    cudaDeviceSynchronize();
    
    // Check for kernel errors
    cudaError_t cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cuda_err));
        return -1;
    }
    
    // --- Timed Runs ---
    for (int run = 0; run < NUM_RUNS; run++) {
        cudaEventRecord(start);

        hybrid_adaptive_spmv_optimized<<<hybrid_blocks, hybrid_threads>>>(
            d_csr.values, d_csr.row_pointers, d_csr.col_indices, d_vec, d_res, n,
            d_short_rows, d_long_rows, num_short, num_long, short_blocks
        );
            
        cudaEventRecord(end);
        cudaEventSynchronize(end);

        float millisec = 0.0;
        cudaEventElapsedTime(&millisec, start, end);
        times[run] = millisec * 1e-3;
    }

    // --- Copy Result Back ---
    cudaMemcpy(h_res, d_res, n * sizeof(dtype), cudaMemcpyDeviceToHost);

    // --- Performance Calculation ---
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    for (int i = 0; i < NUM_RUNS; i++) {
        total_time += times[i];
    }
    dtype avg_time = total_time / NUM_RUNS;


    double bandwidth, gflops;
    calculate_hybrid_bandwidth(n, m, nnz, h_csr.col_indices, num_short, num_long, avg_time, &bandwidth, &gflops);

    // --- Print Matrix Statistics ---
    print_matrix_stats(&h_csr);

    // --- Use existing performance printing function ---
    print_spmv_performance(
        "Hybrid Adaptive CSR", 
        argv[1],
        n, 
        m, 
        nnz, 
        avg_time, 
        bandwidth, 
        gflops, 
        h_res,
        10
    );

    // Print hybrid-specific configuration info
    printf("\n=== Hybrid Configuration Details ===\n");
    printf("Strategy distribution: %.1f%% short rows, %.1f%% long rows\n", 
           100.0 * num_short / n, 100.0 * num_long / n);
    printf("Launch configuration: %d blocks (%d short + %d long), %d threads\n", 
           hybrid_blocks, short_blocks, hybrid_blocks - short_blocks, hybrid_threads);

    // Correctness check against a double-precision CPU reference
    verify_and_report("Hybrid Adaptive CSR", &h_csr, h_vec, h_res);

    // --- Cleanup ---
    if (d_short_rows) cudaFree(d_short_rows);
    if (d_long_rows) cudaFree(d_long_rows);
    if (h_short_rows) free(h_short_rows);
    if (h_long_rows) free(h_long_rows);

    cudaFree(d_vec);
    cudaFree(d_res);
    cudaFree(d_csr.values);
    cudaFree(d_csr.col_indices);
    cudaFree(d_csr.row_pointers);

    free(h_vec);
    free(h_res);
    free(h_csr.values);
    free(h_csr.col_indices);
    free(h_csr.row_pointers);

    return 0;
}