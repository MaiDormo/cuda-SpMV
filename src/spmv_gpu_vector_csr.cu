#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

#include "../include/csr_conversion.h"
#include "../include/read_file_lib.h"
#include "../include/spmv_kernels.h"
#include "../include/spmv_type.h"
#include "../include/spmv_utils.h"

#define WARP_SIZE 32

// --- Main Function ---
int main(int argc, char **argv) {

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
  h_vec = (dtype *)malloc(m * sizeof(dtype));
  h_res = (dtype *)malloc(n * sizeof(dtype));
  h_csr.values = (dtype *)malloc(nnz * sizeof(dtype));
  h_csr.col_indices = (int *)malloc(nnz * sizeof(int));
  h_csr.row_pointers =
      (int *)calloc(n + 1, sizeof(int)); // Zero initialization is important

  if (!h_vec || !h_res || !h_csr.values || !h_csr.col_indices ||
      !h_csr.row_pointers) {
    perror("Failed to allocate host memory");
    // Free any successfully allocated memory before exiting
    free(h_coo.a_val);
    free(h_coo.a_row);
    free(h_coo.a_col);
    free(h_vec);
    free(h_res);
    free(h_csr.values);
    free(h_csr.col_indices);
    free(h_csr.row_pointers);
    return -1;
  }

  // --- Initialize Host Vectors ---
  for (int i = 0; i < m; i++)
    h_vec[i] = 1.0;
  memset(h_res, 0, n * sizeof(dtype));

  // --- Convert COO to CSR ---
  if (coo_to_csr(&h_coo, &h_csr) != 0) {
    fprintf(stderr, "Error during COO to CSR conversion.\n");
    // Free memory and exit
    free(h_coo.a_val);
    free(h_coo.a_row);
    free(h_coo.a_col);
    free(h_vec);
    free(h_res);
    free(h_csr.values);
    free(h_csr.col_indices);
    free(h_csr.row_pointers);
    return -1;
  }

  // Free original COO data (host)
  free(h_coo.a_val);
  h_coo.a_val = NULL;
  free(h_coo.a_row);
  h_coo.a_row = NULL;
  free(h_coo.a_col);
  h_coo.a_col = NULL;

  // --- Calculate Optimal Launch Configuration ---
  int optimal_block_size;

  // Analyze matrix characteristics to determine optimal block size
  struct MAT_STATS stats = calculate_matrix_stats(&h_csr);
  printf("Matrix analysis: mean_nnz_per_row = %.2f\n", stats.mean_nnz_per_row);

  // Determine block size based on matrix density
  if (stats.mean_nnz_per_row < 16) {
    optimal_block_size = 128; // Very sparse matrices - smaller blocks
  } else if (stats.mean_nnz_per_row < 32) {
    optimal_block_size = 256; // Sparse matrices
  } else if (stats.mean_nnz_per_row < 64) {
    optimal_block_size = 512; // Medium density matrices
  } else {
    optimal_block_size =
        1024; // Dense matrices - larger blocks for better occupancy
  }

  // Check device limits and adjust if necessary
  int device;
  cudaGetDevice(&device);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);

  if (optimal_block_size > prop.maxThreadsPerBlock) {
    printf("Warning: Optimal block size (%d) exceeds device limit (%d)\n",
           optimal_block_size, prop.maxThreadsPerBlock);
    optimal_block_size = prop.maxThreadsPerBlock;
  }

  // Ensure block size is a multiple of warp size
  optimal_block_size =
      ((optimal_block_size + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;

  // Calculate launch configuration for vector CSR double buffer kernel
  int warps_per_block = optimal_block_size / WARP_SIZE;
  int rows_per_block =
      warps_per_block * 2; // 2 rows per warp in double buffer kernel
  int optimal_block_num = (n + rows_per_block - 1) / rows_per_block;

  // Ensure minimum occupancy
  // int min_blocks = prop.multiProcessorCount;
  // if (optimal_block_num < min_blocks) {
  //     optimal_block_num = min_blocks;
  // }

  // // Cap maximum blocks to avoid overhead
  // int max_blocks = prop.multiProcessorCount * 8;
  // if (optimal_block_num > max_blocks) {
  //     optimal_block_num = max_blocks;
  // }

  printf("Device properties: %d SMs, %d max threads per block\n",
         prop.multiProcessorCount, prop.maxThreadsPerBlock);
  printf("Calculated launch config:\n");
  printf("  Block size: %d threads (%d warps)\n", optimal_block_size,
         warps_per_block);
  printf("  Grid size: %d blocks\n", optimal_block_num);
  printf("  Rows per block: %d (2 rows per warp)\n", rows_per_block);

  // --- Device Data Structures ---
  struct CSR d_csr; // Holds device pointers
  dtype *d_vec = NULL, *d_res = NULL;

  // --- Allocate Device Memory ---
  cudaMalloc(&d_vec, m * sizeof(dtype));
  cudaMalloc(&d_res, n * sizeof(dtype));
  cudaMalloc(&d_csr.values, h_csr.num_non_zeros * sizeof(dtype));
  cudaMalloc(&d_csr.col_indices, h_csr.num_non_zeros * sizeof(int));
  cudaMalloc(&d_csr.row_pointers, (n + 1) * sizeof(int));

  d_csr.num_rows = n;
  d_csr.num_cols = m;
  d_csr.num_non_zeros = h_csr.num_non_zeros;

  // Check for CUDA errors
  cudaError_t cuda_err = cudaGetLastError();
  if (cuda_err != cudaSuccess) {
    fprintf(stderr, "CUDA memory allocation failed: %s\n",
            cudaGetErrorString(cuda_err));
    return -1;
  }

  // --- Copy Data to Device ---
  cudaMemcpy(d_vec, h_vec, m * sizeof(dtype), cudaMemcpyHostToDevice);
  cudaMemcpy(d_res, h_res, n * sizeof(dtype), cudaMemcpyHostToDevice);
  cudaMemcpy(d_csr.values, h_csr.values, h_csr.num_non_zeros * sizeof(dtype),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_csr.col_indices, h_csr.col_indices,
             h_csr.num_non_zeros * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_csr.row_pointers, h_csr.row_pointers, (n + 1) * sizeof(int),
             cudaMemcpyHostToDevice);

  // --- Timing Setup ---
  const int NUM_RUNS = 50;
  dtype total_time = 0.0;
  dtype times[NUM_RUNS];
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  // --- Warmup Run ---
  vector_csr_double_buffer<<<optimal_block_num, optimal_block_size>>>(
      d_csr.values, d_csr.row_pointers, d_csr.col_indices, d_vec, d_res, n);

  cudaDeviceSynchronize();

  // Check for kernel launch errors
  cuda_err = cudaGetLastError();
  if (cuda_err != cudaSuccess) {
    fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cuda_err));
    return -1;
  }

  // --- Timed Runs ---
  for (int run = 0; run < NUM_RUNS; run++) {
    // Reset result vector to ensure correctness
    // cudaMemset(d_res, 0, n * sizeof(dtype));

    // Add synchronization before starting timing
    cudaDeviceSynchronize();

    cudaEventRecord(start);

    vector_csr_double_buffer<<<optimal_block_num, optimal_block_size>>>(
        d_csr.values, d_csr.row_pointers, d_csr.col_indices, d_vec, d_res, n);

    cudaEventRecord(end);
    cudaEventSynchronize(end);

    // Check for kernel execution errors
    cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess) {
      fprintf(stderr, "Kernel execution failed at run %d: %s\n", run,
              cudaGetErrorString(cuda_err));
      break;
    }

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
  calculate_bandwidth(n, m, nnz, h_csr.col_indices, avg_time, &bandwidth,
                      &gflops);
  // Also call the standard print function for consistency
  print_spmv_performance("Vector CSR Double Buffer", argv[1], n, m, nnz,
                         avg_time, bandwidth, gflops, h_res,
                         10 // Print up to 10 samples
  );

  // Correctness check against a double-precision CPU reference
  verify_and_report("Vector CSR Double Buffer", &h_csr, h_vec, h_res);

  // --- Cleanup ---
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
