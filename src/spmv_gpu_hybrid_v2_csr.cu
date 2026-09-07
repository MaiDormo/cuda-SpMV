#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "../include/csr_conversion.h"
#include "../include/hybrid_v2_plan.h"
#include "../include/read_file_lib.h"
#include "../include/spmv_kernels.h"
#include "../include/spmv_type.h"
#include "../include/spmv_utils.h"

// Hybrid V2 CSR SpMV driver.
//
//   usage: spmv_gpu_hybrid_v2_csr.exec <file.mtx> [block_size] [huge_threshold] [flush_l2]
//
//   block_size     threads per block, multiple of 32 (default 256)
//   huge_threshold rows with more non-zeros are split into chunks of this
//                  many non-zeros, one block per chunk (default 2048)
//   flush_l2       1 = evict the L2 cache before every timed run so small
//                  matrices report HBM bandwidth instead of L2 bandwidth
//                  (default 0)

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err__ = (call);                                                \
    if (err__ != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err__));                                      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

static const int NUM_RUNS = 50;
static const int DEFAULT_BLOCK_SIZE = 256;
static const size_t L2_FLUSH_BYTES = 64u << 20; // > 24 MB L2 of the A30

struct Options {
  const char *matrix_path;
  int block_size;
  int huge_threshold;
  int flush_l2;
};

static int parse_options(int argc, char **argv, struct Options *opt) {
  if (argc < 2 || argc > 5) {
    fprintf(stderr,
            "Usage: %s <path/to/file.mtx> [block_size=%d] [huge_threshold=%d] "
            "[flush_l2=0]\n",
            argv[0], DEFAULT_BLOCK_SIZE, HV2_DEFAULT_HUGE_THRESHOLD);
    return -1;
  }
  opt->matrix_path = argv[1];
  opt->block_size = argc > 2 ? atoi(argv[2]) : DEFAULT_BLOCK_SIZE;
  opt->huge_threshold = argc > 3 ? atoi(argv[3]) : HV2_DEFAULT_HUGE_THRESHOLD;
  opt->flush_l2 = argc > 4 ? atoi(argv[4]) : 0;

  if (opt->block_size < WARP_SIZE || opt->block_size > 1024 ||
      opt->block_size % WARP_SIZE != 0) {
    fprintf(stderr, "block_size must be a multiple of %d in [%d, 1024]\n",
            WARP_SIZE, WARP_SIZE);
    return -1;
  }
  if (opt->huge_threshold < WARP_SIZE) {
    fprintf(stderr, "huge_threshold must be at least %d\n", WARP_SIZE);
    return -1;
  }
  return 0;
}

static double wall_seconds(void) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}

// Device copies of the plan arrays plus the grid layout.
struct DevicePlan {
  HV2DeviceArgs args;
  int *huge_rows;
  int *huge_chunk_begin;
  dtype *chunk_partials;
  int num_huge_rows;
  int total_blocks;
};

static int *upload_ints(const int *host, int count) {
  int *dev = NULL;
  size_t bytes = (size_t)(count > 0 ? count : 1) * sizeof(int);
  CUDA_CHECK(cudaMalloc(&dev, bytes));
  if (count > 0) {
    CUDA_CHECK(cudaMemcpy(dev, host, bytes, cudaMemcpyHostToDevice));
  }
  return dev;
}

static void build_device_plan(const struct HV2Plan *plan, int block_size,
                              struct DevicePlan *dev) {
  memset(dev, 0, sizeof(*dev));
  dev->args.n = plan->n;
  dev->args.huge_threshold = plan->huge_threshold;
  dev->args.direct_class = plan->direct_class;
  dev->num_huge_rows = plan->num_huge_rows;

  for (int c = 0; c < HV2_NUM_LANE_CLASSES; c++) {
    dev->args.class_count[c] = plan->class_count[c];
    dev->args.class_rows[c] =
        (c == plan->direct_class) ? NULL
                                  : upload_ints(plan->class_rows[c],
                                                plan->class_count[c]);
  }
  dev->args.chunk_start = upload_ints(plan->chunk_start, plan->num_chunks);
  dev->args.chunk_end = upload_ints(plan->chunk_end, plan->num_chunks);
  dev->huge_rows = upload_ints(plan->huge_rows, plan->num_huge_rows);
  dev->huge_chunk_begin =
      upload_ints(plan->huge_chunk_begin, plan->num_huge_rows + 1);
  CUDA_CHECK(cudaMalloc(&dev->chunk_partials,
                        (size_t)(plan->num_chunks > 0 ? plan->num_chunks : 1) *
                            sizeof(dtype)));

  // Grid layout: chunks, then lane classes from 32 lanes down to 1 lane.
  dev->args.block_begin[0] = 0;
  dev->args.block_begin[1] = plan->num_chunks;
  for (int segment = 1; segment <= HV2_NUM_LANE_CLASSES; segment++) {
    int c = HV2_NUM_LANE_CLASSES - segment;
    int rows = (c == plan->direct_class) ? plan->n : plan->class_count[c];
    int rows_per_block = hybrid_v2_rows_per_block(block_size, c);
    int blocks = (rows + rows_per_block - 1) / rows_per_block;
    dev->args.block_begin[segment + 1] = dev->args.block_begin[segment] + blocks;
  }
  dev->total_blocks = dev->args.block_begin[HV2_NUM_LANE_CLASSES + 1];
}

static void free_device_plan(struct DevicePlan *dev) {
  for (int c = 0; c < HV2_NUM_LANE_CLASSES; c++) {
    cudaFree((void *)dev->args.class_rows[c]);
  }
  cudaFree((void *)dev->args.chunk_start);
  cudaFree((void *)dev->args.chunk_end);
  cudaFree(dev->huge_rows);
  cudaFree(dev->huge_chunk_begin);
  cudaFree(dev->chunk_partials);
}

static void launch_hybrid_v2(const struct CSR *d_csr, const dtype *d_vec,
                             dtype *d_res, const struct DevicePlan *dev,
                             int block_size) {
  hybrid_v2_spmv<<<dev->total_blocks, block_size>>>(
      d_csr->values, d_csr->row_pointers, d_csr->col_indices, d_vec, d_res,
      dev->chunk_partials, dev->args);
  if (dev->num_huge_rows > 0) {
    int warps_per_block = block_size / WARP_SIZE;
    int blocks = (dev->num_huge_rows + warps_per_block - 1) / warps_per_block;
    hybrid_v2_finalize<<<blocks, block_size>>>(
        dev->chunk_partials, dev->huge_rows, dev->huge_chunk_begin,
        dev->num_huge_rows, d_res);
  }
}

static void print_plan_summary(const struct HV2Plan *plan,
                               const struct DevicePlan *dev, int block_size,
                               double build_seconds) {
  printf("\n=== Hybrid V2 plan ===\n");
  printf("Block size: %d, huge threshold: %d nnz, chunk: %d nnz\n", block_size,
         plan->huge_threshold, plan->chunk_nnz);
  printf("Rows per lane class (1,2,4,8,16,32 lanes):");
  for (int c = 0; c < HV2_NUM_LANE_CLASSES; c++) {
    printf(" %d", plan->class_count[c]);
  }
  printf("\nDirect class: %s", plan->direct_class >= 0 ? "" : "none");
  if (plan->direct_class >= 0) {
    printf("%d lane(s) per row (%.1f%% of rows, no row list)",
           1 << plan->direct_class,
           100.0 * plan->class_count[plan->direct_class] / plan->n);
  }
  printf("\nHuge rows: %d split into %d chunks\n", plan->num_huge_rows,
         plan->num_chunks);
  printf("Grid: %d blocks (%d chunk blocks), finalize kernel: %s\n",
         dev->total_blocks, plan->num_chunks,
         plan->num_huge_rows > 0 ? "yes" : "no");
  printf("Plan metadata read per SpMV: %.2f MB\n",
         hv2_plan_device_bytes(plan) / 1e6);
  printf("Plan build time (host): %.3f ms\n", build_seconds * 1e3);
}

int main(int argc, char **argv) {
  struct Options opt;
  if (parse_options(argc, argv, &opt) != 0) {
    return -1;
  }

  // --- Read matrix and convert to CSR ---
  struct COO h_coo;
  struct CSR h_csr;
  read_from_file_and_init((char *)opt.matrix_path, &h_coo);
  const int n = h_coo.num_rows;
  const int m = h_coo.num_cols;
  const int nnz = h_coo.num_non_zeros;

  dtype *h_vec = (dtype *)malloc(m * sizeof(dtype));
  dtype *h_res = (dtype *)malloc(n * sizeof(dtype));
  h_csr.values = (dtype *)malloc(nnz * sizeof(dtype));
  h_csr.col_indices = (int *)malloc(nnz * sizeof(int));
  h_csr.row_pointers = (int *)calloc(n + 1, sizeof(int));
  if (!h_vec || !h_res || !h_csr.values || !h_csr.col_indices ||
      !h_csr.row_pointers) {
    perror("Failed to allocate host memory");
    return -1;
  }
  for (int i = 0; i < m; i++) {
    h_vec[i] = 1.0;
  }
  memset(h_res, 0, n * sizeof(dtype));

  if (coo_to_csr(&h_coo, &h_csr) != 0) {
    fprintf(stderr, "Error during COO to CSR conversion.\n");
    return -1;
  }
  free(h_coo.a_val);
  free(h_coo.a_row);
  free(h_coo.a_col);

  // --- Build the row plan (host preprocessing) ---
  struct HV2Plan plan;
  double t0 = wall_seconds();
  if (hv2_build_plan(h_csr.row_pointers, n, opt.huge_threshold,
                     opt.huge_threshold, &plan) != 0) {
    fprintf(stderr, "Failed to build the hybrid v2 plan.\n");
    return -1;
  }
  double build_seconds = wall_seconds() - t0;

  // --- Device memory ---
  struct CSR d_csr = h_csr;
  dtype *d_vec = NULL, *d_res = NULL;
  CUDA_CHECK(cudaMalloc(&d_vec, m * sizeof(dtype)));
  CUDA_CHECK(cudaMalloc(&d_res, n * sizeof(dtype)));
  CUDA_CHECK(cudaMalloc(&d_csr.values, nnz * sizeof(dtype)));
  CUDA_CHECK(cudaMalloc(&d_csr.col_indices, nnz * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_csr.row_pointers, (n + 1) * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_vec, h_vec, m * sizeof(dtype), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_res, 0, n * sizeof(dtype)));
  CUDA_CHECK(cudaMemcpy(d_csr.values, h_csr.values, nnz * sizeof(dtype),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_csr.col_indices, h_csr.col_indices, nnz * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_csr.row_pointers, h_csr.row_pointers,
                        (n + 1) * sizeof(int), cudaMemcpyHostToDevice));

  struct DevicePlan dev;
  build_device_plan(&plan, opt.block_size, &dev);

  void *d_flush = NULL;
  if (opt.flush_l2) {
    CUDA_CHECK(cudaMalloc(&d_flush, L2_FLUSH_BYTES));
  }

  // --- Warmup and error check ---
  launch_hybrid_v2(&d_csr, d_vec, d_res, &dev, opt.block_size);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // --- Timed runs ---
  cudaEvent_t start, end;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&end));
  double times[NUM_RUNS];
  for (int run = 0; run < NUM_RUNS; run++) {
    if (d_flush) {
      CUDA_CHECK(cudaMemsetAsync(d_flush, run & 0xFF, L2_FLUSH_BYTES));
    }
    CUDA_CHECK(cudaEventRecord(start));
    launch_hybrid_v2(&d_csr, d_vec, d_res, &dev, opt.block_size);
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, end));
    times[run] = ms * 1e-3;
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(end));

  CUDA_CHECK(cudaMemcpy(h_res, d_res, n * sizeof(dtype), cudaMemcpyDeviceToHost));

  // --- Statistics ---
  double mean_time = 0.0, min_time = times[0];
  for (int i = 0; i < NUM_RUNS; i++) {
    mean_time += times[i];
    if (times[i] < min_time) {
      min_time = times[i];
    }
  }
  mean_time /= NUM_RUNS;
  double median_time = median_of(times, NUM_RUNS);

  size_t extra_bytes = hv2_plan_device_bytes(&plan);
  double bandwidth, gflops;
  calculate_bandwidth_with_extra(n, m, nnz, h_csr.col_indices, mean_time,
                                 extra_bytes, &bandwidth, &gflops);
  double median_bandwidth, median_gflops;
  calculate_bandwidth_with_extra(n, m, nnz, h_csr.col_indices, median_time,
                                 extra_bytes, &median_bandwidth, &median_gflops);

  print_matrix_stats(&h_csr);
  print_spmv_performance("Hybrid V2 CSR", opt.matrix_path, n, m, nnz, mean_time,
                         bandwidth, gflops, h_res, 10);
  printf("\nMedian execution time: %.6f seconds (min %.6f, %d runs, L2 flush: %s)\n",
         median_time, min_time, NUM_RUNS, opt.flush_l2 ? "on" : "off");
  printf("Median memory bandwidth (estimated): %.4f GB/s\n", median_bandwidth);
  printf("Median computational performance: %.6f GFLOPS\n", median_gflops);

  print_plan_summary(&plan, &dev, opt.block_size, build_seconds);
  int verify_status = verify_and_report("Hybrid V2 CSR", &h_csr, h_vec, h_res);

  // --- Cleanup ---
  free_device_plan(&dev);
  cudaFree(d_flush);
  cudaFree(d_vec);
  cudaFree(d_res);
  cudaFree(d_csr.values);
  cudaFree(d_csr.col_indices);
  cudaFree(d_csr.row_pointers);
  hv2_free_plan(&plan);
  free(h_vec);
  free(h_res);
  free(h_csr.values);
  free(h_csr.col_indices);
  free(h_csr.row_pointers);

  return verify_status == 0 ? 0 : 2;
}
