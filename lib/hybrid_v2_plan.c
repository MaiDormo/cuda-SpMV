#include <limits.h>
#include <stdlib.h>
#include <string.h>

#include "../include/hybrid_v2_plan.h"

static void reset_plan(struct HV2Plan *plan, int n, int huge_threshold,
                       int chunk_nnz) {
  memset(plan, 0, sizeof(*plan));
  plan->n = n;
  plan->huge_threshold = huge_threshold;
  plan->chunk_nnz = chunk_nnz;
  plan->direct_class = -1;
}

/* First pass: per-class row counts and the total number of chunks. */
static int count_classes(const int *row_ptr, int n, int huge_threshold,
                         int chunk_nnz, int *class_count, long *num_chunks) {
  *num_chunks = 0;
  for (int i = 0; i < n; i++) {
    int nnz = row_ptr[i + 1] - row_ptr[i];
    if (nnz < 0)
      return -1;
    int c = hv2_row_class(nnz, huge_threshold);
    class_count[c]++;
    if (c == HV2_CLASS_HUGE)
      *num_chunks += (nnz + chunk_nnz - 1) / chunk_nnz;
  }
  return (*num_chunks > INT_MAX) ? -1 : 0;
}

static int pick_direct_class(const int *class_count, int n) {
  for (int c = 0; c < HV2_NUM_LANE_CLASSES; c++) {
    if ((double)class_count[c] > HV2_DIRECT_MIN_SHARE * (double)n)
      return c;
  }
  return -1;
}

static int allocate_plan_arrays(struct HV2Plan *plan) {
  for (int c = 0; c < HV2_NUM_LANE_CLASSES; c++) {
    if (c == plan->direct_class || plan->class_count[c] == 0)
      continue;
    plan->class_rows[c] = (int *)malloc(plan->class_count[c] * sizeof(int));
    if (!plan->class_rows[c])
      return -1;
  }
  int num_huge = plan->num_huge_rows;
  plan->huge_rows = (int *)malloc((num_huge > 0 ? num_huge : 1) * sizeof(int));
  plan->huge_chunk_begin = (int *)malloc((num_huge + 1) * sizeof(int));
  int num_chunks = plan->num_chunks > 0 ? plan->num_chunks : 1;
  plan->chunk_start = (int *)malloc(num_chunks * sizeof(int));
  plan->chunk_end = (int *)malloc(num_chunks * sizeof(int));
  if (!plan->huge_rows || !plan->huge_chunk_begin || !plan->chunk_start ||
      !plan->chunk_end)
    return -1;
  return 0;
}

/* Second pass: fill the row lists and the chunk table. */
static void fill_plan(const int *row_ptr, struct HV2Plan *plan) {
  int fill[HV2_NUM_LANE_CLASSES] = {0};
  int huge_idx = 0;
  int chunk_idx = 0;

  for (int i = 0; i < plan->n; i++) {
    int row_start = row_ptr[i];
    int row_end = row_ptr[i + 1];
    int c = hv2_row_class(row_end - row_start, plan->huge_threshold);

    if (c == HV2_CLASS_HUGE) {
      plan->huge_rows[huge_idx] = i;
      plan->huge_chunk_begin[huge_idx] = chunk_idx;
      for (int s = row_start; s < row_end; s += plan->chunk_nnz) {
        int e = s + plan->chunk_nnz;
        plan->chunk_start[chunk_idx] = s;
        plan->chunk_end[chunk_idx] = e < row_end ? e : row_end;
        chunk_idx++;
      }
      huge_idx++;
    } else if (c != plan->direct_class) {
      plan->class_rows[c][fill[c]++] = i;
    }
  }
  plan->huge_chunk_begin[huge_idx] = chunk_idx;
}

int hv2_build_plan(const int *row_ptr, int n, int huge_threshold, int chunk_nnz,
                   struct HV2Plan *plan) {
  if (!plan)
    return -1;
  reset_plan(plan, n, huge_threshold, chunk_nnz);
  if (!row_ptr || n <= 0 || huge_threshold <= 0 || chunk_nnz <= 0)
    return -1;

  long num_chunks = 0;
  if (count_classes(row_ptr, n, huge_threshold, chunk_nnz, plan->class_count,
                    &num_chunks) != 0)
    return -1;

  plan->num_huge_rows = plan->class_count[HV2_CLASS_HUGE];
  plan->num_chunks = (int)num_chunks;
  plan->direct_class = pick_direct_class(plan->class_count, n);

  if (allocate_plan_arrays(plan) != 0) {
    hv2_free_plan(plan);
    reset_plan(plan, n, huge_threshold, chunk_nnz);
    return -1;
  }

  fill_plan(row_ptr, plan);
  return 0;
}

void hv2_free_plan(struct HV2Plan *plan) {
  if (!plan)
    return;
  for (int c = 0; c < HV2_NUM_LANE_CLASSES; c++) {
    free(plan->class_rows[c]);
    plan->class_rows[c] = NULL;
  }
  free(plan->huge_rows);
  free(plan->huge_chunk_begin);
  free(plan->chunk_start);
  free(plan->chunk_end);
  plan->huge_rows = NULL;
  plan->huge_chunk_begin = NULL;
  plan->chunk_start = NULL;
  plan->chunk_end = NULL;
}

size_t hv2_plan_device_bytes(const struct HV2Plan *plan) {
  if (!plan)
    return 0;
  size_t bytes = 0;
  for (int c = 0; c < HV2_NUM_LANE_CLASSES; c++) {
    if (c != plan->direct_class)
      bytes += (size_t)plan->class_count[c] * sizeof(int);
  }
  /* chunk table read by the main kernel, partials written then read back,
   * huge-row table read by the finalize kernel */
  bytes += (size_t)plan->num_chunks * (2 * sizeof(int) + 2 * sizeof(float));
  bytes += (size_t)plan->num_huge_rows * 2 * sizeof(int) + sizeof(int);
  return bytes;
}
