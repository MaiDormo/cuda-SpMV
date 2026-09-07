#ifndef HYBRID_V2_PLAN_H
#define HYBRID_V2_PLAN_H

#include <stddef.h>

/*
 * Host-side row classification for the hybrid_v2 SpMV kernel.
 *
 * Every row is assigned to one of:
 *   - a lane class c in [0, HV2_NUM_LANE_CLASSES): (1 << c) lanes of a warp
 *     cooperate on the row, so each lane handles at most two non-zeros except
 *     in the widest class;
 *   - the huge class: the row is split into fixed-size chunks of non-zeros,
 *     one thread block per chunk, and the partial sums are combined by a
 *     small finalize kernel.
 *
 * If one lane class holds more than HV2_DIRECT_MIN_SHARE of the rows it is
 * processed "directly" (blocks walk the row range and skip rows of other
 * classes), which avoids reading a row-index list for the common case.
 */

#define HV2_NUM_LANE_CLASSES 6 /* lanes per row: 1, 2, 4, 8, 16, 32 */
#define HV2_CLASS_HUGE HV2_NUM_LANE_CLASSES
#define HV2_DEFAULT_HUGE_THRESHOLD 2048
#define HV2_DEFAULT_CHUNK_NNZ 2048
#define HV2_DIRECT_MIN_SHARE 0.5

#ifdef __CUDACC__
#define HV2_INLINE __host__ __device__ static inline
#else
#define HV2_INLINE static inline
#endif

/* Lane class for a row with `nnz` non-zeros (ignoring the huge threshold). */
HV2_INLINE int hv2_lane_class(int nnz) {
  if (nnz <= 2)
    return 0;
  if (nnz <= 4)
    return 1;
  if (nnz <= 8)
    return 2;
  if (nnz <= 16)
    return 3;
  if (nnz <= 32)
    return 4;
  return 5;
}

/* Full class: lane class or HV2_CLASS_HUGE. */
HV2_INLINE int hv2_row_class(int nnz, int huge_threshold) {
  return nnz > huge_threshold ? HV2_CLASS_HUGE : hv2_lane_class(nnz);
}

struct HV2Plan {
  int n;
  int huge_threshold;
  int chunk_nnz;
  int direct_class; /* lane class processed without a row list, or -1 */
  int class_count[HV2_NUM_LANE_CLASSES + 1];
  int *class_rows[HV2_NUM_LANE_CLASSES]; /* NULL for the direct class */
  int num_huge_rows;
  int *huge_rows;        /* [num_huge_rows] */
  int *huge_chunk_begin; /* [num_huge_rows + 1] prefix into chunk arrays */
  int num_chunks;
  int *chunk_start; /* [num_chunks] first nnz index of the chunk */
  int *chunk_end;   /* [num_chunks] one past the last nnz index */
};

#ifdef __cplusplus
extern "C" {
#endif

/* Builds the plan. Returns 0 on success, -1 on invalid input or allocation
 * failure (the plan is left empty and safe to pass to hv2_free_plan). */
int hv2_build_plan(const int *row_ptr, int n, int huge_threshold, int chunk_nnz,
                   struct HV2Plan *plan);

void hv2_free_plan(struct HV2Plan *plan);

/* Bytes of plan metadata the kernels read per SpMV (for bandwidth accounting). */
size_t hv2_plan_device_bytes(const struct HV2Plan *plan);

#ifdef __cplusplus
}
#endif

#endif /* HYBRID_V2_PLAN_H */
