#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

#include "../include/read_file_lib.h"
#include "../include/spmv_type.h"
#include "../include/csr_conversion.h"
#include "../include/spmv_utils.h"
#include "../include/spmv_kernels.h"


// Optimized launch function that takes pre-calculated parameters
void launch_adaptive_spmv_optimized(
    const CSR *matrix, 
    const dtype *vec, 
    dtype *res,
    const int *row_blocks,
    int num_blocks,
    int block_size,
    size_t shared_mem_size
) {
    adaptive_csr<<<num_blocks, block_size, shared_mem_size>>>(
        matrix->values, matrix->row_pointers, matrix->col_indices,
        vec, res, row_blocks, matrix->num_rows);
}

int main(int argc, char ** argv) { 
    
    if (argc != 2) {
        printf("Usage: <./bin/spmv_*> <path/to/file.mtx>\n");
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
    int *h_block_rows = (int*)calloc(n, sizeof(int));

    if (!h_vec || !h_res || !h_csr.values || !h_csr.col_indices || !h_csr.row_pointers || !h_block_rows) {
        perror("Failed to allocate host memory");
        free(h_coo.a_val); free(h_coo.a_row); free(h_coo.a_col);
        free(h_vec); free(h_res);
        free(h_csr.values); free(h_csr.col_indices); free(h_csr.row_pointers);
        free(h_block_rows);
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
         free(h_block_rows);
         return -1;
    }

    // Free original COO data (host)
    free(h_coo.a_val); h_coo.a_val = NULL;
    free(h_coo.a_row); h_coo.a_row = NULL;
    free(h_coo.a_col); h_coo.a_col = NULL;

    // --- Calculate Launch Configuration FIRST ---
    int optimal_block_size;
    int optimal_num_blocks;
    size_t optimal_shared_mem_size;

    // Calculate optimal block size based on matrix characteristics only
    struct MAT_STATS stats = calculate_matrix_stats(&h_csr);
    if (stats.mean_nnz_per_row < 16) {
    optimal_block_size = 128;  // Very sparse matrices
    } else if (stats.mean_nnz_per_row < 32) {
        optimal_block_size = 256;  // Sparse matrices
    } else if (stats.mean_nnz_per_row < 64) {
        optimal_block_size = 512;  // Medium density matrices
    } else {
        optimal_block_size = 1024; // Dense matrices - utilize full block capacity
    }


    // Check shared memory limits and adjust if necessary
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    size_t requested_shared_mem = optimal_block_size * sizeof(dtype);
    if (requested_shared_mem > prop.sharedMemPerBlock) {
        printf("Warning: Requested shared memory (%zu bytes) exceeds limit (%zu bytes)\n", 
            requested_shared_mem, prop.sharedMemPerBlock);
        optimal_shared_mem_size = prop.sharedMemPerBlock;
        optimal_block_size = optimal_shared_mem_size / sizeof(dtype);
    } else {
        optimal_shared_mem_size = requested_shared_mem;
    }

    // Now use the determined optimal block size for row selection
    int countRowBlocks = adaptive_row_selection(h_csr.row_pointers, n, h_block_rows, WARP_SIZE, optimal_block_size);
    printf("Number of rowBlocks: %d\n", countRowBlocks);

    // Calculate final number of blocks
    optimal_num_blocks = countRowBlocks - 1;

    printf("Final launch config: %d blocks, %d threads per block, %zu bytes shared memory\n",
        optimal_num_blocks, optimal_block_size, optimal_shared_mem_size);

    // --- Device Data Structures ---
    struct CSR d_csr;
    dtype *d_vec = NULL, *d_res = NULL;
    int * d_block_rows;

    // --- Allocate Device Memory ---
    cudaMalloc(&d_vec, m * sizeof(dtype));
    cudaMalloc(&d_res, n * sizeof(dtype));
    cudaMalloc(&d_csr.values, h_csr.num_non_zeros * sizeof(dtype));
    cudaMalloc(&d_csr.col_indices, h_csr.num_non_zeros * sizeof(int));
    cudaMalloc(&d_csr.row_pointers, (n + 1) * sizeof(int));
    cudaMalloc(&d_block_rows, countRowBlocks * sizeof(int));
    d_csr.num_rows = n;
    d_csr.num_cols = m;
    d_csr.num_non_zeros = h_csr.num_non_zeros;

    // --- Copy Data to Device ---
    cudaMemcpy(d_vec, h_vec, m * sizeof(dtype), cudaMemcpyHostToDevice);
    cudaMemcpy(d_res, h_res, n * sizeof(dtype), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csr.values, h_csr.values, h_csr.num_non_zeros * sizeof(dtype), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csr.col_indices, h_csr.col_indices, h_csr.num_non_zeros * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csr.row_pointers, h_csr.row_pointers, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_block_rows, h_block_rows, countRowBlocks * sizeof(int), cudaMemcpyHostToDevice);

    

    // --- Timing Setup ---
    const int NUM_RUNS = 50;
    dtype total_time = 0.0;
    dtype times[NUM_RUNS];
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    // --- Warmup Run ---
    launch_adaptive_spmv_optimized(&d_csr, d_vec, d_res, d_block_rows, 
                                  optimal_num_blocks, optimal_block_size, optimal_shared_mem_size);
    cudaDeviceSynchronize();
    
    // --- Timed Runs ---
    for (int run = 0; run < NUM_RUNS; run++) {
        cudaEventRecord(start);

        launch_adaptive_spmv_optimized(&d_csr, d_vec, d_res, d_block_rows, 
                                      optimal_num_blocks, optimal_block_size, optimal_shared_mem_size);
            
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
   calculate_adaptive_bandwidth(n,m,nnz,h_csr.col_indices, optimal_num_blocks, avg_time, &bandwidth, &gflops);

    // --- Print Other Stats ---
    print_matrix_stats(&h_csr);

    // --- Print Results ---
    print_spmv_performance(
        "Adaptive CSR", 
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

    // Correctness check against a double-precision CPU reference
    verify_and_report("Adaptive CSR", &h_csr, h_vec, h_res);

    // --- Cleanup ---
    cudaFree(d_vec);
    cudaFree(d_res);
    cudaFree(d_csr.values);
    cudaFree(d_csr.col_indices);
    cudaFree(d_csr.row_pointers);
    cudaFree(d_block_rows);

    free(h_vec);
    free(h_res);
    free(h_csr.values);
    free(h_csr.col_indices);
    free(h_csr.row_pointers);
    free(h_block_rows);

    return 0;
}