#include <cusparse.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

#include "../include/csr_conversion.h"
#include "../include/read_file_lib.h"
#include "../include/spmv_type.h"
#include "../include/spmv_utils.h"

// --- Main Function ---
int main(int argc, char **argv) {
  if (argc != 2) {
    printf("Usage: <./bin/spmv_gpu_cusparse_csr> <path/to/file.mtx>\n");
    return -1;
  }

  const dtype alpha = 1.0f;
  const dtype beta = 0.0f;

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
    h_vec[i] = 1.0f;
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

  // --- Device Data Structures ---
  struct CSR d_csr; // Holds device pointers
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
  cudaMemcpy(d_res, h_res, n * sizeof(dtype),
             cudaMemcpyHostToDevice); // Copy initial zeros
  cudaMemcpy(d_csr.values, h_csr.values, nnz * sizeof(dtype),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_csr.col_indices, h_csr.col_indices, nnz * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_csr.row_pointers, h_csr.row_pointers, (n + 1) * sizeof(int),
             cudaMemcpyHostToDevice);

  // --- Initialize cuSPARSE ---
  cusparseHandle_t handle;
  cusparseCreate(&handle);

  // Create matrix/vector descriptors using the new generic API
  cusparseSpMatDescr_t matA;
  cusparseDnVecDescr_t vecX, vecY;
  void *dBuffer = NULL;
  size_t bufferSize = 0;

  // Create sparse matrix A in CSR format
  cusparseCreateCsr(&matA, n, m, nnz, d_csr.row_pointers, d_csr.col_indices,
                    d_csr.values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

  // Create dense vector X
  cusparseCreateDnVec(&vecX, m, d_vec, CUDA_R_32F);

  // Create dense vector Y
  cusparseCreateDnVec(&vecY, n, d_res, CUDA_R_32F);

  // Allocate an external buffer if needed
  cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                          matA, vecX, &beta, vecY, CUDA_R_32F,
                          CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);
  cudaMalloc(&dBuffer, bufferSize);

  // --- Timing Setup ---
  const int NUM_RUNS = 50;
  dtype total_time = 0.0;
  dtype times[NUM_RUNS];
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  // --- Warmup Run ---
  cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX,
               &beta, vecY, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer);

  cudaDeviceSynchronize();

  // --- Timed Runs ---
  for (int run = 0; run < NUM_RUNS; run++) {
    cudaEventRecord(start);

    cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX,
                 &beta, vecY, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer);

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
  calculate_bandwidth(n, m, nnz, h_csr.col_indices, avg_time, &bandwidth,
                      &gflops);

  // Also call the standard print function for consistency
  print_spmv_performance("cuSPARSE CSR", argv[1], n, m, nnz, avg_time,
                         bandwidth, gflops, h_res,
                         10 // Print up to 10 samples
  );

  // Correctness check against a double-precision CPU reference
  verify_and_report("cuSPARSE CSR", &h_csr, h_vec, h_res);

  // --- Cleanup cuSPARSE ---
  cusparseDestroySpMat(matA);
  cusparseDestroyDnVec(vecX);
  cusparseDestroyDnVec(vecY);
  if (dBuffer)
    cudaFree(dBuffer);
  cusparseDestroy(handle);

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
