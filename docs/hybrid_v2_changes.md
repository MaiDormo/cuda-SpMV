# Hybrid V2 SpMV: what changed and why

This document explains every change made when moving from the **Hybrid Adaptive**
kernel (`hybrid_adaptive_spmv_optimized`) to **Hybrid V2** (`hybrid_v2_spmv`),
with code excerpts from the repository. Paths are relative to the repository root.

---

## 1. Where the previous kernel lost time

Measured on the A30 (HBM2 peak about 933 GB/s, practical ceiling about 800 GB/s)
with `results/spmv_benchmark_results_second_deliverable.csv`. The "floor" is the
minimum bytes the kernel must move (values + column indices + row pointers +
touched vector entries + result) divided by 800 GB/s.

| Matrix      | Hybrid Adaptive | Floor      | Share of floor | Cause                                              |
|-------------|-----------------|------------|----------------|----------------------------------------------------|
| mawi        | 226 ms          | ~6 ms      | ~3 %           | one row with 97 M nnz handled by a single warp     |
| Zd_Jac3_db  | 19 µs           | < 8 µs     | ~40 %          | 31-nnz rows handled by one thread each (latency)   |
| Goodwin_127 | 111 µs          | ~60 µs     | ~55 %          | row-list indirection, scattered result writes      |
| ML_Geer     | 1.87 ms         | ~1.13 ms   | ~60 %          | same                                               |
| CurlCurl_4  | 237 µs          | ~180 µs    | ~76 %          | already close                                      |
| 662_bus     | 6 µs            | n/a        | n/a            | pure launch latency                                |

Two failure modes dominate: **load imbalance** on huge rows and a **latency chain**
on medium rows. The changes below address them in that order.

---

## 2. Kernel changes

### 2.1 Lanes per row chosen from the row's own length

**Before.** Hybrid Adaptive had two bins. A "short" row (below a threshold picked
from a per-matrix table) was processed by one thread; a "long" row by one warp.

```cuda
// lib/spmv_kernels.cu — hybrid_adaptive_spmv_optimized (short-row path)
int row = short_rows[row_idx];
int start = csr_row_ptr[row];
int end   = csr_row_ptr[row + 1];
dtype sum = 0.0;
int j = start;
for (; j + 3 < end; j += 4) {              // one thread, all non-zeros of the row
  sum += csr_values[j]     * __ldg(&vec[csr_col_indices[j]]);
  sum += csr_values[j + 1] * __ldg(&vec[csr_col_indices[j + 1]]);
  sum += csr_values[j + 2] * __ldg(&vec[csr_col_indices[j + 2]]);
  sum += csr_values[j + 3] * __ldg(&vec[csr_col_indices[j + 3]]);
}
for (; j < end; j++)
  sum += csr_values[j] * __ldg(&vec[csr_col_indices[j]]);
res[row] = sum;
```

The unrolling does not help here: each `vec[col]` gather is a dependent global
load, so a thread on a 31-nnz row waits for about 31 memory latencies in
sequence. Every warp in the grid is stuck in the same chain, so the GPU cannot
hide it by switching warps. That is why Zd_Jac3_db (mean 31 nnz/row, threshold
32 → almost all rows "short") stayed at ~40 % of its floor. Escalating to a full
warp was the only alternative, and a warp on a 5-nnz row leaves 27 lanes idle.

**After.** Six classes: 1, 2, 4, 8, 16 or 32 lanes per row, chosen per row so
that a lane handles at most two non-zeros except in the widest class.

```c
// include/hybrid_v2_plan.h — shared by host (gcc) and device (nvcc)
HV2_INLINE int hv2_lane_class(int nnz) {
  if (nnz <= 2)  return 0;   // 1 lane
  if (nnz <= 4)  return 1;   // 2 lanes
  if (nnz <= 8)  return 2;   // 4 lanes
  if (nnz <= 16) return 3;   // 8 lanes
  if (nnz <= 32) return 4;   // 16 lanes
  return 5;                  // 32 lanes (a full warp)
}
```

The class boundaries are fixed and depend only on the row, not on a per-matrix
heuristic table, so no tuning is needed for a new matrix. The row segment is
summed by a group of `L` consecutive lanes; the reduction width is a template
parameter so the shuffle loop unrolls completely:

```cuda
// lib/spmv_kernels.cu
template <int L>
__device__ __forceinline__ dtype hv2_group_sum(const dtype *vals, const int *cols,
                                               const dtype *vec, int start, int end,
                                               int lane_in_group) {
  dtype sum = 0;
  for (int j = start + lane_in_group; j < end; j += L)
    sum += vals[j] * __ldg(&vec[cols[j]]);
#pragma unroll
  for (int offset = L / 2; offset > 0; offset >>= 1)
    sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset);   // stays inside the group
  return sum;
}
```

A warp therefore processes 32 one-nnz rows at once, or 8 four-lane rows, or one
wide row, and the per-lane dependent-load chain is at most two loads long.

### 2.2 Huge rows split across thread blocks

**Before.** Any row above the threshold went to exactly one warp:

```cuda
// lib/spmv_kernels.cu — hybrid_adaptive_spmv_optimized (long-row path)
int row = long_rows[warp_id];
for (int j = start + lane_id; j < end; j += 32)       // 97 M nnz / 32 lanes = 3 M iterations
  thread_sum += csr_values[j] * __ldg(&vec[csr_col_indices[j]]);
```

On mawi the longest row has 97,373,173 non-zeros. One warp looped three million
times while the other 55 SMs finished everything else and idled. This single row
is the whole reason mawi ran at ~3 % of its floor.

**After.** Rows above `huge_threshold` (default 2048 nnz) are cut on the host
into fixed chunks; the kernel gives each chunk a whole block:

```c
// lib/hybrid_v2_plan.c — fill_plan()
if (c == HV2_CLASS_HUGE) {
  plan->huge_rows[huge_idx] = i;
  plan->huge_chunk_begin[huge_idx] = chunk_idx;
  for (int s = row_start; s < row_end; s += plan->chunk_nnz) {
    int e = s + plan->chunk_nnz;
    plan->chunk_start[chunk_idx] = s;
    plan->chunk_end[chunk_idx]   = e < row_end ? e : row_end;
    chunk_idx++;
  }
  huge_idx++;
}
```

```cuda
// lib/spmv_kernels.cu — one block reduces one chunk to a single partial sum
__device__ __forceinline__ void hv2_chunk_block(const dtype *vals, const int *cols,
                                                const dtype *vec, int start, int end,
                                                dtype *partial_out, dtype *warp_sums) {
  dtype sum = 0;
  for (int j = start + threadIdx.x; j < end; j += blockDim.x)
    sum += __ldcs(&vals[j]) * __ldg(&vec[__ldcs(&cols[j])]);
  sum = hv2_block_sum(sum, warp_sums);          // shuffle + shared-memory reduction
  if (threadIdx.x == 0) *partial_out = sum;
}
```

The 97 M non-zeros now spread over about 47,500 independent blocks that fill the
GPU. A second, tiny kernel gives each huge row one warp to sum its partials:

```cuda
// lib/spmv_kernels.cu
__global__ void hybrid_v2_finalize(const dtype *chunk_partials, const int *huge_rows,
                                   const int *huge_chunk_begin, int num_huge, dtype *res) {
  const int global_warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (global_warp >= num_huge) return;          // whole warp exits together
  const int begin = huge_chunk_begin[global_warp];
  const int end   = huge_chunk_begin[global_warp + 1];
  dtype sum = 0;
  for (int k = begin + lane; k < end; k += WARP_SIZE) sum += chunk_partials[k];
  for (int offset = 16; offset > 0; offset >>= 1)
    sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset);
  if (lane == 0) res[huge_rows[global_warp]] = sum;
}
```

Why two passes instead of `atomicAdd` into `res[row]`: no memset of the output
is needed before every multiply, the result is bit-identical on every run, and
the finalize launch is skipped entirely when a matrix has no huge rows.

### 2.3 The dominant class runs without a row list

**Before.** Every row index went through an explicit list, and the kernel read the
list entry before it could read the row pointers:

```cuda
// hybrid_adaptive_spmv_optimized
int row   = short_rows[row_idx];       // 4 extra bytes per row, then...
int start = csr_row_ptr[row];          // ...a dependent load
```

That is 4 extra bytes per row per multiply. On mawi (226 M rows, 240 M nnz) the
list alone is about 0.9 GB out of roughly 4.6 GB of total traffic. When short and
long rows interleave, the row-pointer reads and the `res[row]` writes are also
scattered instead of coalesced.

**After.** The plan builder checks whether one lane class holds more than half of
the rows. If so, that class is the *direct* class and gets no list:

```c
// lib/hybrid_v2_plan.c
static int pick_direct_class(const int *class_count, int n) {
  for (int c = 0; c < HV2_NUM_LANE_CLASSES; c++)
    if ((double)class_count[c] > HV2_DIRECT_MIN_SHARE * (double)n)   // > 50 %
      return c;
  return -1;
}
```

Blocks of the direct class walk the row range in order. They read the row
pointers they were going to read anyway, derive the class from the row length,
and simply contribute nothing for rows that belong elsewhere:

```cuda
// lib/spmv_kernels.cu — hv2_lane_class_block<C, DIRECT>
int row = -1, start = 0, end = 0;
if (idx < count) {
  row   = DIRECT ? idx : rows_list[idx];
  start = row_ptr[row];
  end   = row_ptr[row + 1];
  if (DIRECT && hv2_row_class(end - start, huge_threshold) != C) {
    row = -1;          // another class owns this row: no work, no write
    end = start;
  }
}
dtype sum = hv2_group_sum<L>(vals, cols, vec, start, end, lane_in_group);
if (row >= 0 && lane_in_group == 0) res[row] = sum;
```

Consecutive threads now touch consecutive row pointers and consecutive result
entries, so both accesses are coalesced, and the list vanishes for the common
case. Minority classes keep ascending lists, so their reads coalesce too.
Skipped rows only cost idle lanes, negligible when the class covers most rows.

Note that `row = -1; end = start;` is used instead of `return`: every lane of
the warp must still reach the `__shfl_xor_sync` calls, so lanes with nothing to
do run the loop zero times and shuffle a zero.

### 2.4 Longest work scheduled first

The block scheduler hands out blocks in index order. If the heaviest blocks come
last they start when most of the GPU is already draining and produce a long
tail. Hybrid Adaptive launched short-row blocks first and long-row blocks after,
the reverse of what tail behaviour wants.

```c
// src/spmv_gpu_hybrid_v2_csr.cu — build_device_plan(): grid layout
dev->args.block_begin[0] = 0;
dev->args.block_begin[1] = plan->num_chunks;                 // segment 0: chunk blocks
for (int segment = 1; segment <= HV2_NUM_LANE_CLASSES; segment++) {
  int c = HV2_NUM_LANE_CLASSES - segment;                    // 32 lanes first ... 1 lane last
  int rows = (c == plan->direct_class) ? plan->n : plan->class_count[c];
  int rows_per_block = hybrid_v2_rows_per_block(block_size, c);
  int blocks = (rows + rows_per_block - 1) / rows_per_block;
  dev->args.block_begin[segment + 1] = dev->args.block_begin[segment] + blocks;
}
```

The kernel maps `blockIdx.x` back to its segment with a short uniform loop and
dispatches to the right template instantiation:

```cuda
// lib/spmv_kernels.cu — hybrid_v2_spmv
if (block < args.block_begin[1]) { /* chunk */ return; }
int segment = 1;
while (segment < HV2_NUM_LANE_CLASSES && block >= args.block_begin[segment + 1]) segment++;
const int lane_class = HV2_NUM_LANE_CLASSES - segment;
const int local_block = block - args.block_begin[segment];
const bool direct = (lane_class == args.direct_class);
switch (lane_class) { case 5: HV2_DISPATCH_CLASS(5); break; /* ... */ }
```

One launch covers every class; only the finalize kernel is a separate launch,
and only when huge rows exist.

### 2.5 Load hints and reductions

- In the chunk path every value and column index is touched exactly once, so
  they are loaded with `__ldcs` (streaming, evict-first). That keeps L1 free for
  the `vec` gathers, which *are* reused. The Makefile's global
  `--def-load-cache=ca` still applies to the lane classes, where a thread
  revisits the same 32-byte sector across iterations and L1 caching helps.
- All reductions use `__shfl_xor_sync` instead of `__shfl_down_sync`. Every lane
  ends up holding the full sum, so no `if (lane == 0)` is needed inside the
  reduction and no lane carries a half-reduced value.

Compiler report for the new kernels (`nvcc -Xptxas -v`, sm_80):

| Kernel               | Registers | Spills | Shared memory |
|----------------------|-----------|--------|---------------|
| `hybrid_v2_spmv`     | 32        | 0      | 128 B         |
| `hybrid_v2_finalize` | 31        | 0      | 0             |

---

## 3. Measuring correctly

### 3.1 Correctness check in every GPU driver

**Before.** No GPU driver compared its output with anything; they printed the
first ten non-zero results. A kernel with a broken reduction, a skipped row or
a race can be fast and still look plausible in ten printed numbers.

**After.** Every driver (and the cuSPARSE baseline) recomputes the product on
the CPU in double precision and reports the worst row:

```c
// lib/spmv_utils.c — verify_spmv_result()
#pragma omp parallel for reduction(max:worst) reduction(+:bad) schedule(dynamic, 4096)
for (int i = 0; i < n; i++) {
  double ref = 0.0, scale = 0.0;
  for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
    double term = (double)vals[j] * (double)vec[cols[j]];
    ref   += term;
    scale += fabs(term);              // sum of |terms|: the natural error scale
  }
  double err = fabs(ref - (double)result[i]);
  if (scale > 0.0) err /= scale;      // relative to the row's magnitude
  if (err > worst) worst = err;
  if (err > rel_tol) bad++;
}
```

The error is scaled by the sum of the absolute terms, not by the result, so a
row that legitimately cancels to near zero does not raise a false alarm. Each
driver prints one line such as

```
Verification (Hybrid V2 CSR): PASS (max relative error 2.4e-07, 0 row(s) above 1.0e-03)
```

and the Hybrid V2 driver exits with a non-zero code on FAIL so batch scripts
can notice.

### 3.2 Median next to the mean

Fifty runs averaged arithmetically let one outlier (clock ramp, another job on
the shared node) shift the number. The driver keeps printing the mean under the
label the extraction script already parses, and adds:

```
Median execution time: 0.000229 seconds (min 0.000226, 50 runs, L2 flush: off)
Median memory bandwidth (estimated): 671.2 GB/s
```

### 3.3 Optional L2 flush

The A30 has a 24 MB L2. Zd_Jac3_db needs ~6 MB for its whole matrix and 662_bus
far less, so after the warm-up run they sit entirely in L2 and the reported
"memory bandwidth" is L2 bandwidth, not HBM. The driver can now evict L2 before
each timed run, outside the timed window:

```cuda
// src/spmv_gpu_hybrid_v2_csr.cu — timed loop
if (d_flush)
  CUDA_CHECK(cudaMemsetAsync(d_flush, run & 0xFF, L2_FLUSH_BYTES));   // 64 MB > 24 MB L2
CUDA_CHECK(cudaEventRecord(start));
launch_hybrid_v2(&d_csr, d_vec, d_res, &dev, opt.block_size);
CUDA_CHECK(cudaEventRecord(end));
```

`scripts/run_spmv_hybrid_v2.sh` runs every matrix both warm and cold so the
report can show both numbers honestly.

### 3.4 Bandwidth accounting that includes the plan

The old formula counted the two row lists. The new kernel reads different
metadata, so a helper computes exactly what it moves:

```c
// lib/hybrid_v2_plan.c
size_t hv2_plan_device_bytes(const struct HV2Plan *plan) {
  size_t bytes = 0;
  for (int c = 0; c < HV2_NUM_LANE_CLASSES; c++)
    if (c != plan->direct_class)                       // direct class has no list
      bytes += (size_t)plan->class_count[c] * sizeof(int);
  bytes += (size_t)plan->num_chunks * (2 * sizeof(int) + 2 * sizeof(float)); // chunk table + partials
  bytes += (size_t)plan->num_huge_rows * 2 * sizeof(int) + sizeof(int);
  return bytes;
}
```

This is passed to `calculate_bandwidth_with_extra()` so the reported GB/s
includes the kernel's real overhead rather than flattering it.

---

## 4. Build, test and pipeline

### 4.1 Preprocessing in plain C with a unit test

The classification is the part most likely to hide a bug that silently drops or
duplicates a row. It lives in `lib/hybrid_v2_plan.c` with no CUDA dependency, so
it compiles with gcc and sanitizers on a machine without a GPU.
`test/test_hybrid_v2_plan.c` builds plans for synthetic row shapes (including a
mawi-shaped one with a single giant row) and for real matrices, then checks:

```c
// test/test_hybrid_v2_plan.c — invariants checked per plan
CHECK(covered == n, "covered %ld rows of %d", covered, n);          // every row assigned...
for (int r = 0; r < n; r++)
  CHECK(seen[r] == 1, "row %d seen %d times", r, seen[r]);          // ...exactly once
CHECK(plan.chunk_start[cb] == row_ptr[r],     "row %d first chunk start", r);
CHECK(plan.chunk_end[ce - 1] == row_ptr[r + 1], "row %d last chunk end", r);
CHECK(plan.chunk_start[k] == plan.chunk_end[k - 1], "chunk %d not contiguous", k);
```

Run it with `make test-plan` (add `MTX="a.mtx b.mtx"` for real matrices).

### 4.2 Scripts and README

- `scripts/run_spmv_hybrid_v2.sh`: SLURM script; forwards `block_size` and
  `huge_threshold` from its command line and runs warm and cold variants.
- `scripts/run_all_benchmarks.sh`: the new script is registered. The entry
  `run_cusparse.h` was corrected to `run_cusparse.sh`; the typo meant the
  cuSPARSE baseline was never submitted by the full run.
- `scripts/extract_spmv_data.sh`: detects `Hybrid V2 CSR` output and labels it
  `gpu_hybrid_v2` in the CSV.
- `README.md`: describes the kernel, its parameters, the verification line and
  the host-only test.

---

## 5. What is still unmeasured

Everything compiles for sm_80 and the host logic is tested, but the kernel has
not yet run on the A30. Expected outcome from the analysis:

| Matrix      | Hybrid Adaptive | Target for Hybrid V2         |
|-------------|-----------------|------------------------------|
| mawi        | 226 ms          | < 20 ms                      |
| Zd_Jac3_db  | 19 µs           | < 12 µs                      |
| others      | see §1          | no regression, small gains   |

Ideas deliberately left for a profiler-guided pass on the cluster: dropping
`--def-load-cache=ca` from the Makefile, 128-bit vectorised loads in the
32-lane class, and a merge-path CSR variant as a further comparison point.

To validate:

```bash
make clean && make
sbatch scripts/run_spmv_hybrid_v2.sh          # defaults: 256 threads, huge > 2048 nnz
sbatch scripts/run_spmv_hybrid_v2.sh 512 4096 # parameter variants
grep "Verification" hybrid_v2_spmv-*.out      # every line must say PASS
```
