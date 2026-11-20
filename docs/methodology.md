# Measurement methodology (read before touching a GPU)

Numbers without a method are decoration. Every claim in the README, thesis,
or CV must be traceable to a run that follows these rules.

## The three rules

1. **Oracle per kernel.** Every GPU kernel is qualified against an
   independent reference, never against another GPU kernel. Here the oracle
   is the double-precision CPU CSR reference; every driver prints
   `Verification (...): PASS/FAIL`. A benchmark row with FAIL (or without
   the line) does not exist — do not record it, do not average it in.
2. **Fixed fixtures, recorded conditions.** Matrices are fixtures: record
   name, rows, nnz, longest row (see `scripts/download_matrices.sh` for the
   canonical set). Every result row must carry: GPU model, CUDA
   driver/runtime, `sm_XX` build flag, clocks state (`nvidia-smi` locked or
   not), cache-flush policy, run count, warmup policy. Same fixture +
   same conditions or the comparison is void.
3. **Evidence rules.** Median over ≥5 timed runs, warmup excluded (drivers
   already do a warmup run — keep it). Report GFLOPS *and* GB/s *and* % of
   the cuSPARSE baseline on the same fixture, same line. Stop measuring once
   alternatives are distinguished; more runs do not make a weak claim strong.
   Nsight (`ncu`) answers kernel questions (top 3 limiters); end-to-end
   claims need end-to-end timing, never kernel-only math.

## Already in place

- 50 timed runs per matrix, warm-up excluded, averaged (legacy Deliverable 2
  drivers).
- CSV extract scripts (`scripts/extract_spmv_data.sh`, `extract_test.sh`).
- cuSPARSE vendor baseline (`test/cusparse.cu` via `test/compile.sh`).
- Cloud build path: `make local` (direct `nvcc`, no UniTN modules).

## Gaps (fix before claiming more)

- [ ] FP64 CPU reference + `Verification (...): PASS/FAIL` is **not** in the
      legacy drivers on `main`. The experimental V2 branch has a verifier; do not
      attribute it to the Deliverable 2 numbers.
- [ ] Lock GPU clocks during timed runs (`sudo nvidia-smi -lgc`), record it.
- [ ] Pin the fixture set: minimum publishable = 3 matrices (small/L2-resident
      e.g. `662_bus`, one medium, one huge-row e.g. `mawi`) + longest-row stat.
- [ ] State the FP32-vs-FP64 pass tolerance next to the PASS/FAIL line.
- [ ] Nsight top-3 limiters per kernel (DRAM-bound? latency? imbalance?).
- [ ] Update order: repo README first, CV second, site config third.

## First-GPU-run checklist (for agents)

1. `deviceQuery` (or `nvidia-smi`) → record GPU, driver, CUDA version.
2. Set `RELEASE_NV_OPT` to the card's arch (`sm_80` A30/A100, `sm_86`
   3090/A10/A40, `sm_89` 4090/L40S, `sm_75` T4), `make clean && make local`.
3. `./scripts/download_matrices.sh` (if `data/` empty).
4. Correctness pass: every driver must print PASS on all fixtures. Stop on FAIL.
5. Timing pass: Hybrid Adaptive binary, each matrix x5, median.
6. Baseline pass: `test/compile.sh` + `run_cusparse.sh`, same fixtures.
7. Roofline pass (`ncu`, where available): achieved % of peak + top 3 limiters.
8. Write the per-matrix table (rows, nnz, longest row, median ms, GFLOPS,
   GB/s, % of cuSPARSE, limiter) into the README. Only then touch the CV.
