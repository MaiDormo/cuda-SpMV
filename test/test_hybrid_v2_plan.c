/*
 * Host-only test for the hybrid_v2 row plan. Builds the plan for a CSR row
 * pointer array and checks that every row is covered exactly once and that
 * chunks tile the huge rows exactly.
 *
 * Build:  gcc -std=c11 -O2 -fopenmp -o test_hybrid_v2_plan \
 *             test/test_hybrid_v2_plan.c lib/hybrid_v2_plan.c \
 *             lib/read_file_lib.c lib/coo_to_csr.c -lm
 * Usage:  test_hybrid_v2_plan [file.mtx ...]
 * With no arguments only the synthetic cases run.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/csr_conversion.h"
#include "../include/hybrid_v2_plan.h"
#include "../include/read_file_lib.h"
#include "../include/spmv_type.h"

static int failures = 0;

#define CHECK(cond, ...)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      failures++;                                                              \
      printf("  FAIL: " __VA_ARGS__);                                          \
      printf("\n");                                                            \
    }                                                                          \
  } while (0)

static void check_plan(const int *row_ptr, int n, int huge_threshold,
                       int chunk_nnz, const char *label) {
  struct HV2Plan plan;
  printf("case %s (n=%d, huge>%d, chunk=%d)\n", label, n, huge_threshold,
         chunk_nnz);
  CHECK(hv2_build_plan(row_ptr, n, huge_threshold, chunk_nnz, &plan) == 0,
        "build failed");

  int *seen = calloc(n, sizeof(int));
  long covered = 0;

  /* rows from lists */
  for (int c = 0; c < HV2_NUM_LANE_CLASSES; c++) {
    if (c == plan.direct_class) {
      CHECK(plan.class_rows[c] == NULL, "direct class %d has a list", c);
      continue;
    }
    for (int k = 0; k < plan.class_count[c]; k++) {
      int r = plan.class_rows[c][k];
      CHECK(r >= 0 && r < n, "class %d row %d out of range", c, r);
      int nnz = row_ptr[r + 1] - row_ptr[r];
      CHECK(hv2_row_class(nnz, huge_threshold) == c,
            "row %d (nnz %d) listed in class %d", r, nnz, c);
      if (k > 0)
        CHECK(plan.class_rows[c][k - 1] < r, "class %d list not ascending", c);
      seen[r]++;
      covered++;
    }
  }

  /* rows from the direct class: implied by class membership */
  if (plan.direct_class >= 0) {
    int direct_count = 0;
    for (int r = 0; r < n; r++) {
      int nnz = row_ptr[r + 1] - row_ptr[r];
      if (hv2_row_class(nnz, huge_threshold) == plan.direct_class) {
        seen[r]++;
        direct_count++;
      }
    }
    CHECK(direct_count == plan.class_count[plan.direct_class],
          "direct class count %d vs %d", direct_count,
          plan.class_count[plan.direct_class]);
    CHECK((double)direct_count > HV2_DIRECT_MIN_SHARE * n,
          "direct class below share threshold");
    covered += direct_count;
  }

  /* huge rows and chunks */
  CHECK(plan.num_huge_rows == plan.class_count[HV2_CLASS_HUGE],
        "huge count mismatch");
  CHECK(plan.huge_chunk_begin[0] == 0, "first chunk begin not 0");
  CHECK(plan.huge_chunk_begin[plan.num_huge_rows] == plan.num_chunks,
        "chunk prefix does not end at num_chunks (%d vs %d)",
        plan.huge_chunk_begin[plan.num_huge_rows], plan.num_chunks);
  for (int h = 0; h < plan.num_huge_rows; h++) {
    int r = plan.huge_rows[h];
    CHECK(r >= 0 && r < n, "huge row %d out of range", r);
    int nnz = row_ptr[r + 1] - row_ptr[r];
    CHECK(nnz > huge_threshold, "huge row %d has nnz %d", r, nnz);
    seen[r]++;
    covered++;
    int cb = plan.huge_chunk_begin[h], ce = plan.huge_chunk_begin[h + 1];
    CHECK(ce > cb, "huge row %d has no chunks", r);
    CHECK(plan.chunk_start[cb] == row_ptr[r], "row %d first chunk start", r);
    CHECK(plan.chunk_end[ce - 1] == row_ptr[r + 1], "row %d last chunk end", r);
    for (int k = cb; k < ce; k++) {
      int len = plan.chunk_end[k] - plan.chunk_start[k];
      CHECK(len > 0 && len <= chunk_nnz, "chunk %d has length %d", k, len);
      if (k > cb)
        CHECK(plan.chunk_start[k] == plan.chunk_end[k - 1],
              "chunk %d not contiguous", k);
    }
  }

  CHECK(covered == n, "covered %ld rows of %d", covered, n);
  for (int r = 0; r < n; r++)
    CHECK(seen[r] == 1, "row %d seen %d times", r, seen[r]);

  printf("  direct class: %d, counts:", plan.direct_class);
  for (int c = 0; c <= HV2_CLASS_HUGE; c++)
    printf(" %d", plan.class_count[c]);
  printf(", chunks: %d, plan bytes: %zu\n", plan.num_chunks,
         hv2_plan_device_bytes(&plan));

  free(seen);
  hv2_free_plan(&plan);
}

static void synthetic_cases(void) {
  /* rows: 0,1,2,3,5,9,17,33,100,5000 nnz -> exercises every class */
  int lens[] = {0, 1, 2, 3, 5, 9, 17, 33, 100, 5000};
  int n = sizeof(lens) / sizeof(lens[0]);
  int row_ptr[11];
  row_ptr[0] = 0;
  for (int i = 0; i < n; i++)
    row_ptr[i + 1] = row_ptr[i] + lens[i];
  check_plan(row_ptr, n, 64, 1000, "synthetic-all-classes");
  check_plan(row_ptr, n, 4096, 2048, "synthetic-no-huge");
  check_plan(row_ptr, n, 1, 1, "synthetic-chunk-1");

  /* mostly 1-nnz rows with one giant row: the mawi shape */
  int big_n = 1000;
  int *big = malloc((big_n + 1) * sizeof(int));
  big[0] = 0;
  for (int i = 0; i < big_n; i++)
    big[i + 1] = big[i] + (i == 500 ? 100000 : (i % 3 == 0 ? 0 : 1));
  check_plan(big, big_n, 2048, 2048, "synthetic-mawi-shape");
  free(big);

  /* invalid input must fail cleanly */
  struct HV2Plan plan;
  CHECK(hv2_build_plan(NULL, 10, 2048, 2048, &plan) == -1, "NULL accepted");
  CHECK(hv2_build_plan(row_ptr, 0, 2048, 2048, &plan) == -1, "n=0 accepted");
  CHECK(hv2_build_plan(row_ptr, n, 0, 2048, &plan) == -1, "huge=0 accepted");
  hv2_free_plan(&plan);
}

static void file_case(char *path) {
  struct COO coo;
  struct CSR csr;
  read_from_file_and_init(path, &coo);
  csr.values = malloc(coo.num_non_zeros * sizeof(dtype));
  csr.col_indices = malloc(coo.num_non_zeros * sizeof(int));
  csr.row_pointers = calloc(coo.num_rows + 1, sizeof(int));
  if (coo_to_csr(&coo, &csr) != 0) {
    printf("  conversion failed for %s\n", path);
    failures++;
    return;
  }
  check_plan(csr.row_pointers, csr.num_rows, HV2_DEFAULT_HUGE_THRESHOLD,
             HV2_DEFAULT_CHUNK_NNZ, path);
  check_plan(csr.row_pointers, csr.num_rows, 8, 8, path);
  free(coo.a_val);
  free(coo.a_row);
  free(coo.a_col);
  free(csr.values);
  free(csr.col_indices);
  free(csr.row_pointers);
}

int main(int argc, char **argv) {
  synthetic_cases();
  for (int i = 1; i < argc; i++)
    file_case(argv[i]);
  printf(failures ? "\n%d check(s) FAILED\n" : "\nall checks passed\n",
         failures);
  return failures ? 1 : 0;
}
