# Hybrid V2 SpMV: what changed and why

> **How to read this doc:** if you haven't looked at the code in a while,
> read only §0–§2. That gives you the full idea with no code.
> §1B is a visual CUDA crash-course: every V2 choice maps to one principle there.
> §3 gives the visual before/after for each fix with the full logic chain.
> §4 onward is the code-level reference for when you need exact details.

---

## 0. The idea in 30 seconds

SpMV computes `y = A * x` where `A` is sparse (mostly zeros). Each row of `A`
has a different number of non-zeros: one row may have 2, the next 30, one
freak row may have 97 million.

**Hybrid V2 is a recipe for giving every row the right number of GPU threads:**

1. **Right-sized teams.** Instead of "1 thread or 1 full warp (32 threads)"
   per row, pick 1, 2, 4, 8, 16, or 32 threads based on the row's own length,
   so nobody idles and nobody waits in a long serial chain.
2. **Split giants.** Rows above ~2048 non-zeros are chopped into fixed-size
   chunks. Each chunk gets a whole thread block, and a tiny second kernel
   adds up the partial sums. One giant row can no longer serialize the GPU.
3. **Skip the guest list.** If >50% of rows fall in one size class (the common
   case), don't store a list of "which rows are in this class" — just walk
   rows in order and skip the others. Saves memory traffic and keeps accesses
   coalesced.
4. **Biggest jobs first.** Launch chunk blocks first, then 32-thread rows,
   down to 1-thread rows, so the long tail doesn't hold up the end of the kernel.

Everything else in this doc is just those 4 ideas spelled out.

**When you run it, this happens:**

```text
CPU once per matrix          GPU every SpMV
───────────────────          ──────────────
count non-zeros per row  →   one big kernel (chunks + all 6 size classes)
sort rows into 6 piles   →   tiny cleanup kernel, only if giant rows exist
chop giants into chunks
```

The CPU sorting ("plan") is built once in plain C (`lib/hybrid_v2_plan.c`),
uploaded to the GPU, and reused for all 50 timed runs.

---

## 1. Quick refresher (CSR, warps, why it is hard)

**CSR format.** We store only non-zeros: `values[]`, `col_indices[]`, and
`row_ptr[]` where row `i` owns entries `row_ptr[i] .. row_ptr[i+1]-1`.
Row `i` computes `sum(values[j] * x[col_indices[j]])`.

Example with 4 rows:

```text
row 0: 2 non-zeros   → needs little work
row 1: 3 non-zeros   → needs little work
row 2: 30 non-zeros  → needs medium work
row 3: 5000 non-zeros → needs a lot of work
```

Concrete memory layout for a toy matrix:

```text
values:      [ a b | c d e | f ... (30 vals) ... | g ... (5000 vals) ... ]
col_indices: [ 0 5 | 1 2 9 | 3 ...               ... | 7 ...               ... ]
                    ▲         ▲                       ▲
row_ptr:     [ 0 , 2 ,     5 ,                     35 ,                    5035 ]
row:            0       1                         2                         3

Row i reads values[row_ptr[i] .. row_ptr[i+1]-1].
So row length nnz(i) = row_ptr[i+1] - row_ptr[i] is known with 2 loads.
```

**GPU vocabulary used below** (one paragraph, that's all):

| Term | Plain meaning |
|------|---------------|
| thread / lane | one worker |
| warp | 32 workers that always move in lockstep |
| block | a team of warps (e.g. 256 threads = 8 warps) that can cooperate |
| coalesced | consecutive workers reading consecutive memory — fast; scattered reads — slow |
| gather | `x[col]` — unpredictable address, the expensive part of SpMV |

**The core tension:** short rows want few threads (otherwise 31 of 32 threads
sit idle), long rows want many threads (otherwise one thread loops forever
while everyone else waits). No single choice fits a matrix with both.

---

## 1B. CUDA principles V2 relies on (visual crash-course)

Every V2 decision below cites one of these principles as `P1..P7`.
If you understand these 7 pictures, you understand why V2 looks the way it does.

### P1 — SIMT lockstep: a warp is 32 lanes moving as one

```text
time ──►

lane 0:  [ load ] [ mul ] [ load ] [ mul ] [ shuffle ] [ write ]
lane 1:  [ load ] [ mul ] [ load ] [ mul ] [ shuffle ] [  idle ]
lane 2:  [ load ] [ mul ] [ load ] [ mul ] [ shuffle ] [  idle ]
...
lane 31: [ load ] [ mul ] [ load ] [ mul ] [ shuffle ] [  idle ]
           ▲ same instruction, same cycle, 32 lanes at once
```

Consequences enforced in V2:

- Lanes that do nothing still **execute** the instruction (predicated off).
  Giving 32 lanes to a 2-nnz row wastes 30/32 of the issued work.
- Lanes that take different branches **diverge**: first one side runs with the
  other side masked, then vice versa. V2 therefore never branches *inside* a
  cooperating group on data — the only `if (DIRECT ...)` test in
  `hv2_lane_class_block` resolves identically for all lanes working on the same
  row, and the `row = -1` idle path still falls through to the same shuffle
  calls (see §4.3) so the warp never diverges at `__shfl_xor_sync`.
- Shuffle reductions (`__shfl_xor_sync`) require **all 32 lanes to arrive**.
  An early `return` by some lanes would hang or corrupt the reduction. That is
  why idle lanes run the loop zero times with `sum = 0` instead of returning.

### P2 — Coalesced access: consecutive lanes → consecutive addresses → one transaction

Global memory moves in 32-byte sectors (128 bytes per fully coalesced warp
access for 4-byte data). The hardware coalesces only when lane `k` touches
address `base + k`:

```text
COALESCED (fast: 1–2 transactions)          SCATTERED (slow: up to 32)

lane:   0  1  2  3 ... 31                   lane:   0  1  2  3 ... 31
addr:  [a0][a1][a2][a3]...[a31]             addr:  [a0]  ..[a91]..[a7]..[a400]..
        ▲ one contiguous 128 B line                  ▲ each lane pulls its own line
```

Enforced in V2:

- `values[]` / `col_indices[]` inside a row are contiguous, and consecutive
  lanes read `start + lane_in_group + k*L` — consecutive addresses. Good.
- `row_ptr[row]` / `res[row]` are coalesced **only if consecutive threads
  handle consecutive rows**. That is exactly what the sorted per-class lists
  and the direct-class linear walk guarantee (§3.3). An interleaved row list
  (`short_rows = [0, 1005, 7, ...]`) would scatter both.
- `x[col]` (the gather) is inherently uncoalesced — column indices are random.
  Nothing can fix that; the strategy is to not make it worse and to keep L1
  free for it (§4.5).

### P3 — Latency hiding through occupancy (Little's law)

A global load costs ~400–800 cycles. The SM hides that by swapping to another
warp while one waits — but only if there *is* another warp ready:

```text
Low occupancy (1 warp/SM):        High occupancy (many warps/SM):

warp A: [load........wait........][use]    warp A: [load....wait...]
warp -:  idle all that time                warp B:      [load....wait...]
                                           warp C:           [compute]
                                           warp A:                [use] ← data arrived
```

Enforced in V2:

- One thread looping over 31 dependent gathers (`sum += vals[j]*x[col[j]]`)
  exposes ~31 serial latencies with **zero** independent work between them.
  Splitting the row over 16 lanes turns 31 serial waits into ~2 parallel waits
  (§3.1). Same bytes, ~15× less exposed latency.
- Giant rows handled by one warp leave 55 of 56 SMs idle (§3.2). Splitting one
  row into ~47,500 blocks gives the scheduler enough independent work to keep
  every SM fed.
- 32 registers / 0 spill / 128 B shared (compiler report in §4.5) means many
  blocks can be resident simultaneously — high occupancy is *possible*. A
  register-hungry kernel would cap it regardless of how clever the mapping is.

### P4 — Memory hierarchy: registers > shared/L1 > L2 > HBM, and each has a job

```text
                   A30 numbers (approx.)
                   ─────────────────────
 per-thread regs:  255 max, V2 uses 32        ← thread sums live here
 per-block shared: 48–164 KB, V2 uses 128 B   ← only chunk/final reductions
 L1 / RO cache:    192 KB per SM              ← reused vals + x[] gathers
 L2:               24 MB chip-wide            ← whole small matrices fit here!
 HBM2:             24 GB, ~933 GB/s peak      ← what we benchmark against
```

Enforced in V2:

- Per-lane partial sums stay in **registers**; intra-group combine uses
  **shuffles** (register-to-register, no shared memory, no barrier) whenever
  the group fits in a warp (all 6 lane classes do).
- **Shared memory** is used only where cooperation must cross warp boundaries:
  the chunk path (`hv2_block_sum`: one `__syncthreads` per block) — 128 B total.
- `__ldg()` marks the `x[]` gather path as read-only-cache friendly; `__ldcs()`
  marks streamed `values/col` in the chunk path as evict-first so one-pass data
  does not thrash L1 that the gathers need (§4.5).
- The 24 MB L2 explains the `flush_l2` benchmarking knob (§5.3): matrices under
  ~6 MB never touch HBM after warm-up unless you evict L2 deliberately.

### P5 — No stragglers: the kernel ends when the slowest block ends

```text
GPU with 4 SMs, 5 blocks (block 5 is huge):

SM0: [blk0][blk4]
SM1: [blk1][idle........]
SM2: [blk2][idle........]   ← 3 SMs drain, then wait for the straggler
SM3: [blk3][idle........]
                 [blk5............................] on SM0
time ──────────────────────────────────────────────►
```

Enforced in V2 as **schedule biggest work first** (§3.4): CUDA issues blocks in
index order, so segment 0 = chunks, then 32-lane rows down to 1-lane rows.
Heavy blocks start while all SMs are still full; tiny 1-lane rows fill the
drain at the end instead of a giant chunk landing on an otherwise empty GPU.
Hybrid Adaptive did the reverse (short rows first) and paid the tail penalty.

### P6 — Deterministic reduction beats atomics for SpMV

```text
ATOMICS (one address, many writers):        V2 TWO-PASS (disjoint writers):

thr0 ──╲                                     block0 → partial[0] ──╲
thr1 ───►► atomicAdd(&y[row])  contention,   block1 → partial[1] ───► warp sums
thr2 ──╱   float non-associativity,          block2 → partial[2] ──╱  in index order
         need y[] zeroed first               (bit-identical every run, no zeroing)
```

Enforced in V2: chunk blocks write **disjoint** `partial[chunk]` slots; the
finalize kernel sums each row's partials in order with one warp. No `atomicAdd`
anywhere, no `memset(y,0)` before every run, no run-to-run bit variation, and
the finalize launch is skipped entirely when there are no huge rows.

### P7 — Launch overhead and metadata are real costs on small matrices

Kernel launch ≈ several microseconds; `662_bus` total floor is ~6 µs. Two
consequences enforced in V2:

- **One launch covers all 6 lane classes** (grid segments + `switch` dispatch,
  §4.4). Only huge rows pay a second launch, and only when they exist.
- **Row lists cost 4 B/row/run** plus a dependent load before `row_ptr` can be
  read. On `mawi` (226 M rows) that is ~0.9 GB of pure overhead (§3.3) — hence
  the direct class that deletes the list for the majority class.

Quick map (used as margin notes in §3–§4):

| V2 choice | Principles enforced |
|---|---|
| 1/2/4/8/16/32 lanes per row | P1 (lockstep/shuffle), P3 (cut serial latency), P2 (consecutive lanes) |
| Huge-row chunking + 2-pass | P3 (feed all SMs), P6 (no atomics), P5 (kill straggler) |
| Direct class, sorted lists | P2 (coalesce row_ptr/res), P7 (delete 4 B/row + dependent load) |
| Biggest-first grid order | P5 (tail), P7 (single launch) |
| `__ldg`/`__ldcs`, `shfl_xor`, templates, 32 regs | P4 (hierarchy), P1 (uniform shuffles), P3 (occupancy) |

---

## 2. What Hybrid Adaptive did, and where it broke

Hybrid Adaptive (the previous kernel) had **two bins**: rows below a
per-matrix threshold got 1 thread, rows above got a full warp (32 threads).
The threshold was hand-tuned per matrix.

That fixed the average case but left two failure modes, measured on the A30
(HBM2 peak ~933 GB/s, practical ceiling ~800 GB/s). "Floor" = minimum bytes
that must move ÷ 800 GB/s:

| Matrix      | Hybrid Adaptive | Floor      | Share of floor | Cause in plain words |
|-------------|-----------------|------------|----------------|----------------------|
| mawi        | 226 ms          | ~6 ms      | ~3 %           | one 97 M-nnz row stuck on a single warp |
| Zd_Jac3_db  | 19 µs           | < 8 µs     | ~40 %          | 31-nnz rows each stuck on a single thread |
| Goodwin_127 | 111 µs          | ~60 µs     | ~55 %          | extra row-list traffic + scattered writes |
| ML_Geer     | 1.87 ms         | ~1.13 ms   | ~60 %          | same |
| CurlCurl_4  | 237 µs          | ~180 µs    | ~76 %          | already close |
| 662_bus     | 6 µs            | n/a        | n/a            | pure launch latency, nothing to fix |

In short: **giant rows starved the GPU, medium rows waited in line, and the
row lists cost extra traffic.** V2 fixes them in that order.

---

## 3. Hybrid V2 in plain English (the 4 fixes, with pictures)

### 3.1 Right-sized teams instead of 1-thread-or-32

Give each row `1, 2, 4, 8, 16, or 32` cooperating threads ("lanes"), chosen
from its own length so each lane handles at most ~2 non-zeros (except rows
with 33–2048 nnz, which share 32 lanes and loop a few times):

| Row length | Lanes assigned | Work per lane |
|------------|---------------|---------------|
| 1–2        | 1             | ≤ 2           |
| 3–4        | 2             | ≤ 2           |
| 5–8        | 4             | ≤ 2           |
| 9–16       | 8             | ≤ 2           |
| 17–32      | 16            | ≤ 2           |
| 33–2048    | 32 (a warp)   | loops, ~1–64 each |
| > 2048     | split (see §3.2) | —          |

Concrete: with 256 threads/block (8 warps), one block handles 256 one-nnz
rows at once, or 8 wide rows at once — the hardware stays full either way.
No per-matrix tuning: the boundaries are fixed.

Why it matters: a 31-nnz row previously waited for ~31 dependent `x[col]`
loads in a row on one thread. Now 16 lanes do ~2 loads each in parallel,
then combine with fast warp shuffles.

**Visual 1 — what goes wrong with "1 thread per short row" (P1, P3).**

A 6-nnz row on 1 thread vs 4 lanes. Each `x[col]` is a dependent load
(`→` = stall waiting for memory):

```text
BEFORE (1 thread, 6 serial gathers):

thread: [ld→][mul][ld→][mul][ld→][mul][ld→][mul][ld→][mul][ld→][mul][wr]
         ─────────────────── ~6 latencies back-to-back ─────────────────
         warp cannot switch away: all 32 threads are in the same chain.

AFTER (4 lanes, same row, hv2_group_sum<4>):

lane0:  [ld→][mul] [ld→][mul] ╲
lane1:  [ld→][mul] [ld→][mul] ──► [xor16][xor8] → every lane holds sum → lane0 writes
lane2:  [ld→][mul] [ld→][mul] ╱   ▲ register shuffles, ~log2(4) steps, no memory
lane3:  [ld→][mul] [ld→][mul] ╱
         ── ~2 latencies ──  ▲ fully parallel, consecutive addresses (P2)
```

Logic chain, stated fully:

1. SpMV's inner loop is **gather-bound**: `vals[j] * x[col[j]]` needs `x[col]`
   before the multiply can issue, so each iteration is a pointer chase.
2. On one thread, `n` non-zeros = `n` dependent latencies in series (P3). No
   instruction-level parallelism helps because iteration `k+1` does not depend
   on `k` arithmetically but still cannot overlap its load usefully — the
   thread has nothing else to do while waiting.
3. Warp-level parallelism *could* hide it, but every warp is stuck in the same
   serial chain on its own row, so there is no ready warp to swap to.
4. Splitting the row over `L` lanes cuts the chain from `n` to `ceil(n/L)`
   parallel waits. Choosing `L ≈ n/2` caps the chain at ~2 for all rows up to
   32 nnz. That is why the boundaries double (2, 4, 8, 16, 32): each class
   keeps `nnz/L ≤ 2`.
5. `L` must divide 32 and be a power of two: a warp then splits into an integer
   number of equal groups (`GROUPS_PER_WARP = 32/L`), all groups run the same
   shuffle pattern with no divergence (P1), and the `xor` reduction depth is
   exactly `log2(L)`.

**Visual 2 — what goes wrong with "a full warp per row" on narrow rows (P1).**

A 5-nnz row on 32 lanes (old long-row path) vs 4 lanes (V2 class 2):

```text
OLD: 32 lanes, 5 do work, 27 idle the whole row:

lane 0-4:  [work][work]...   useful
lane 5-31: [idle][idle]...   still consume issue slots (lockstep, P1)
efficiency ≈ 5/32 ≈ 16 %

V2: the same warp handles 8 such rows at once (8 groups × 4 lanes):

warp: [rowA:lanes0-3][rowB:lanes4-7][rowC:lanes8-11]...[rowH:lanes28-31]
       ▲ 32/32 lanes useful, rows_per_block = 8 warps × 8 groups = 64 rows/block
```

The general formula in code — `hybrid_v2_rows_per_block(block, class)` —
is just "warps per block × groups per warp":

```text
block=256 (8 warps):  class0 (L=1): 8×32=256 rows/block ... class5 (L=32): 8×1=8 rows/block
block=512 (16 warps): class0: 512 rows/block ... class5: 16 rows/block
```

**Why not other granularities (e.g. 3, 6, 12 lanes)?** Three reasons, all P1:
non-powers-of-two do not divide 32 evenly (leftover lanes diverge), shuffle
reduction needs power-of-two depth to leave the sum in every lane without
branches, and lane-index math (`lane & (L-1)`, `lane >> C`) compiles to single
bitwise ops only for powers of two.

### 3.2 Giants are chopped up

Any row above `huge_threshold` (default 2048) is cut on the CPU into
2048-nnz chunks. Each chunk gets a **whole block**; the block reduces its
chunk to one number. A second tiny kernel gives each giant row one warp to
add up its chunk partials.

The 97 M-nnz row in `mawi` becomes ~47,500 independent blocks that fill all
56 SMs instead of blocking one warp for 3 M iterations.

Why two passes instead of `atomicAdd` directly into `y[row]`: no need to
zero the output before every run, results are bit-identical across runs,
and the second kernel is skipped entirely when there are no giant rows.

Note: the chunk size currently equals `huge_threshold` (one CLI knob controls
both — see `hv2_build_plan(row_ptr, n, huge_threshold, huge_threshold)` in
`src/spmv_gpu_hybrid_v2_csr.cu`).

**Visual 3 — the mawi straggler (P3, P5, P6).**

```text
BEFORE (one warp owns the 97 M-nnz row, WARP_SIZE=32 → 3.04 M iterations):

SM 0:  [giant row warp: iter 0 ... iter 3,041,000.....................]  226 ms
SM 1:  [all other 226 M rows.................................] idle....
SM 2:  [................................] idle....
...
SM 55: [................................] idle....
        ▲ 55 SMs finish in ~6 ms then wait ~220 ms for SM 0 (P5 straggler).
        Only 32 threads of ~10,000+ do useful work (P3 starvation).

AFTER (97.4 M / 2048 ≈ 47,543 chunk blocks + 6 lane-class segments):

SM 0:  [chk0][chk56][chk112]...
SM 1:  [chk1][chk57][chk113]...
...
SM 55: [chk55][chk111]...[tail: 1-lane rows]
        ▲ every SM fed from the same queue; finalize = 1 warp, ~47 k adds.
        Grid order (P5): chunks first so no giant chunk lands on an empty GPU.
```

Logic chain, stated fully:

1. Work per row varies by 7 orders of magnitude (2 vs 97 M). Any scheme that
   assigns a *bounded* number of threads per row (1 thread, 1 warp) has an
   unbounded serial tail on the maximum row (P5).
2. The only way to bound the tail is to bound the work *per block*: cut the row
   into fixed `chunk_nnz` pieces so every block does ≤ `chunk_nnz` gathers.
   Now parallelism scales with row length instead of being capped at 32.
3. One block (not one warp) per chunk because a chunk of 2048 with 256 threads
   is 8 gathers per thread — enough to saturate the block's memory pipeline
   (P3) while keeping the block count high enough (~47 k) to fill 56 SMs many
   times over. One warp per chunk would need 8× more blocks and finer partials
   with no benefit.
4. Two passes instead of atomics (P6): `atomicAdd` to one address from 47 k
   blocks serializes on that address, needs `y[row]=0` before every run
   (extra pass + extra traffic), and float atomics are order-dependent
   (bit-varying results). Disjoint `partial[chunk]` writes have no contention;
   the finalize warp sums them in index order — deterministic, no zeroing, and
   zero cost when `num_huge_rows == 0` (launch skipped).
5. Why default 2048: it equals ~8 iterations of a 256-thread block
   (`2048/256 = 8`), a sweet spot between "too many tiny blocks → finalize and
   launch overhead dominate" and "too few huge blocks → tail again". It is
   exposed as a CLI knob (`huge_threshold`) precisely so the sweep script can
   verify this trade-off per matrix instead of trusting the default.

**Why not send giant rows to cuSPARSE / a separate kernel launch?** Same-grid
placement (segment 0 of the one big launch) means no extra launch latency (P7),
no separate H2D of matrix data, and the scheduler interleaves chunks with wide
rows automatically. A separate kernel would serialize behind the first launch
on the same stream anyway.

### 3.3 The common case skips the row list

Before, every row index went through an explicit list (`short_rows[i]`),
costing 4 extra bytes/row plus a dependent load before the row pointers
could even be read. On `mawi` (226 M rows) that's ~0.9 GB of pure overhead.

V2 checks: does one size class hold > 50% of rows? If yes, that class is
"direct" — its blocks just walk rows `0..n-1` in order, read the row
pointers they needed anyway, and skip rows belonging to other classes.
No list stored, and row-pointer + result accesses become consecutive
(coalesced). Minority classes keep sorted lists, so they stay coalesced too.

**Visual 4 — pointer chasing vs linear walk (P2, P7).**

```text
BEFORE (indirect: list → row_ptr → values; note the dependent load):

thread k reads:  short_rows[k] ──► row ──► row_ptr[row], row_ptr[row+1] ──► values
                 4 B extra/row     ▲ random if short/long rows interleave
                                   row_ptr reads scatter, res[row] writes scatter (P2 ✗)

  short_rows: [   0, 1042,    7,   9001, ... ]   ← ascending but sparse
  row_ptr reads:  ptr[0], ptr[1042], ptr[7], ... ← jumps all over HBM
  res writes:     res[0], res[1042], res[7], ... ← same scatter

AFTER (direct class, e.g. class 0 holds 80 % of rows; P2 ✓, P7 saves 4 B/row):

thread k reads:  row = k ──► row_ptr[k], row_ptr[k+1] ──► values (or skip)
                 no list at all     ▲ consecutive threads, consecutive addresses

  block handles rows [1024..1279]: ptr[1024..1280] = one 1 KB stream, res[1024..1279] same.
  Rows of other classes: lanes compute nnz, see class ≠ C, contribute 0, write nothing.
  Cost of a skip ≈ a few predicated-off instructions; negligible at 80 % majority.
```

Logic chain, stated fully:

1. The list exists to *compact* each class: without it, a block assigned to
   class `C` would have to scan all rows to find its own. Compaction restores
   coalescing for minority classes at the price of 4 B/row + one dependent
   load (P7) — the load matters more than the bytes because `row_ptr[row]`
   cannot issue until `rows_list[idx]` returns.
2. When one class is the majority (>50 %), compaction buys nothing for it: its
   rows are already dense in `0..n-1`. Walking all `n` rows and skipping the
   minority keeps `row_ptr`/`res` perfectly consecutive (P2 optimal) while
   deleting the list entirely.
3. Why 50 % and not higher/lower (`HV2_DIRECT_MIN_SHARE`): below half, the
   direct walk wastes more skip-slots than the list would cost; above half, the
   list costs more traffic than the skips waste. 50 % is the break-even where
   `skipped_rows × idle_cost < listed_rows × 4 B + dependent_load`. Minority
   classes keep ascending lists so *their* accesses stay coalesced too — both
   paths are P2-clean.
4. Skipped rows must still participate in shuffles (P1): hence
   `row = -1; end = start;` instead of `return` — the lane sums an empty range
   (zero) and shuffles a zero, so the group's `xor` reduction stays uniform.

### 3.4 Biggest work is scheduled first

CUDA hands out blocks in index order. If heavy blocks come last, most of the
GPU drains and then waits for the stragglers (long tail). V2 orders the grid:
chunk blocks → 32-lane rows → … → 1-lane rows. (Hybrid Adaptive did the
reverse: short rows first.) One launch covers everything; only the finalize
pass is a second launch, and only when needed.

**Visual 5 — grid order and the tail (P5, P7).**

```text
Grid layout (blockIdx.x increases ──►):

[ seg0: #chunks chunk blocks ][ seg1: 32-lane ][ seg2: 16-lane ]...[ seg6: 1-lane ]
  ▲ heaviest, most parallel      ▲ ... decreasing work per block ... ▲ lightest

GOOD (V2: heavy first):                    BAD (heavy last, old order):

SMs: ██████████ (all full on chunks)       SMs: ██████████ (finish small work fast)
     ████████░░ (chunks + wide rows)            ██████░░░░ (drain...)
     ████░░░░░░ (narrow rows fill drain)        ██░░░░░░░░ (drain...)
     ░░░░░░░░░░ done. No straggler.             █░░░░░░░░░ one heavy block holds SM0
```

Logic chain, stated fully:

1. The hardware block scheduler is FIFO by `blockIdx.x` (documented behavior
   relied upon, not guaranteed round-robin — hence "hint", but in practice
   exact on NVIDIA GPUs). Whatever is at low indices starts first on full SMs.
2. Tail duration = `max_block_time − average_block_time` after the queue
   drains. Putting the longest blocks first overlaps them with the maximum
   number of other blocks; putting them last exposes their full duration with
   no overlap (P5 picture above).
3. Within lane classes the same argument orders 32-lane before 1-lane: a
   32-lane row does up to 2048 gathers, a 1-lane row ≤ 2. The old short-first
   order was optimal for *latency of first result* but pessimal for *makespan*,
   which is what the benchmark measures.
4. One launch for everything (P7): the `block_begin[8]` table + `switch` on
   `lane_class` costs a few instructions per block and saves a launch (~µs)
   per class — decisive on `662_bus`-scale matrices where the whole SpMV is
   ~6 µs.

---

## 3B. End-to-end worked example (8 rows, block 128, threshold 2048)

To pin down every logic passage, trace a toy matrix through the plan:

```text
row:      0   1   2   3   4   5   6   7
nnz:      1   3   7  12  20  40 100 3000
class:    0   1   2   3   4   5   5  HUGE   (hv2_lane_class / hv2_row_class)
lanes:    1   2   4   8  16  32  32  chunk
```

Host plan (`hv2_build_plan`, `lib/hybrid_v2_plan.c`):

```text
counts: c0=1 c1=1 c2=1 c3=1 c4=1 c5=2 HUGE=1, chunks = ceil(3000/2048) = 2
direct? largest share = c5 with 2/8 = 25 % < 50 % → direct_class = -1 (no direct)
lists:  class_rows[0]=[0], [1]=[1], [2]=[2], [3]=[3], [4]=[4], [5]=[5,6]
huge:   huge_rows=[7], chunk table: [row7+0 .. row7+2048), [row7+2048 .. row7+3000)
```

Device grid (`build_device_plan`, block 128 = 4 warps):

```text
rows/block: c0:128 c1:64 c2:32 c3:16 c4:8 c5:4   (warps × groups, §3.1)
blocks: seg0 chunks=2 | c5: ceil(2/4)=1 | c4:1 | c3:1 | c2:1 | c1:1 | c0:1 → 8 blocks

blockIdx:  0   1  |  2  |  3  |  4  |  5  |  6  |  7
segment:   chunk   | c5  | c4  | c3  | c2  | c1  | c0
```

Execution:

```text
blk0: 128 threads cooperatively reduce values[s..s+2048) → partial[0]
blk1: same for remainder → partial[1]
blk2: warp0: lanes0-31 reduce row5; warp1: lanes0-31 reduce row6; warps2-3 idle-but-shuffling
blk3: 1 group of 16 lanes reduces row4, other 7 groups' warps... (1 block, 8 rows capacity, 1 used)
...
blk7: 128 groups of 1 lane handle row0 (127 groups idle-but-converged)
finalize: 1 warp sums partial[0..1] → res[7]
```

Note what this exposes: on tiny matrices the tail blocks are under-filled —
that is *expected and fine* (P7: one launch still beats six launches). The
design pays off when each segment has hundreds of blocks, which is every real
matrix in `data/`.

---

## 4. Kernel changes (code reference)

You can stop here for the concept. Below is the exact code behind each idea.

### 4.1 Lanes per row chosen from the row's own length

**Before.** Two bins; short rows ran serially on one thread:

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

Why `__shfl_xor_sync` and not `__shfl_down_sync` (used by the old kernels):
`xor` is a butterfly — after `log2(L)` steps **every** lane holds the full sum,
so no `if (lane == 0)` is needed inside the reduction and no lane carries a
half-reduced value into the write-back test. `down` leaves partial sums in
upper lanes and needs the extra branch. Both are correct; `xor` keeps the warp
converged (P1) and compiles to the same shuffle throughput.

Why a template `<int L>` instead of a runtime loop bound: `L` as a compile-time
constant lets `#pragma unroll` fully unroll the `log2(L)` shuffle chain and lets
`lane & (L-1)` / `lane >> C` fold to immediates. The `switch (lane_class)` in
`hybrid_v2_spmv` instantiates exactly 6 copies — ~6× code for the reduction but
each copy is a few instructions and stays in I-cache.

### 4.2 Huge rows split across thread blocks

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

Note the `if (global_warp >= num_huge) return;` placement: it is *before* any
shuffle, and warps are whole (32 consecutive threads), so either all 32 lanes
return together or all 32 proceed to the shuffles. A per-lane early exit after
shuffles started would violate P1; here it cannot happen.

### 4.3 The dominant class runs without a row list

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

Lane↔row mapping detail (why the bit math works): for class `C` with
`L = 1<<C` lanes per row, `lane_in_group = lane & (L-1)` and
`group = lane >> C` partition the warp into `32/L` groups with consecutive
lanes per group. Because groups are consecutive and rows are consecutive
(`idx = local_block*rows_per_block + warp*GROUPS + group`), consecutive lanes
(except at group boundaries) touch consecutive rows' pointers — coalesced (P2).
`threadIdx.x & 31` / `>> 5` assume nothing about block size except divisibility
by 32, enforced by the driver's CLI validation.

### 4.4 Longest work scheduled first

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

Why the `while` loop over ≤6 segments is cheap: it is *uniform* — every thread
in the block computes the same `segment` from the same `blockIdx.x`, so no
divergence (P1); 6 iterations of integer compares are negligible next to
hundreds of gather latencies. The alternative (separate kernel per class, or a
per-block class table in global memory) would cost launches (P7) or an extra
dependent load per block.

### 4.5 Load hints and reductions

- In the chunk path every value and column index is touched exactly once, so
  they are loaded with `__ldcs` (streaming, evict-first). That keeps L1 free for
  the `vec` gathers, which *are* reused. The Makefile's global
  `--def-load-cache=ca` still applies to the lane classes, where a thread
  revisits the same 32-byte sector across iterations and L1 caching helps.
- All reductions use `__shfl_xor_sync` instead of `__shfl_down_sync`. Every lane
  ends up holding the full sum, so no `if (lane == 0)` is needed inside the
  reduction and no lane carries a half-reduced value.

```text
Cache-policy picture (P4):

LANE-CLASS path (row reused across loop trips):   CHUNK path (each element once):

  vals[j], cols[j]: __ldg / cached (ca)             vals[j], cols[j]: __ldcs (streaming)
   └─ same 32 B sector revisited next trip           └─ never revisited → don't pollute L1
  x[col]: __ldg (read-only cache)                   x[col]: __ldg (read-only cache)
   └─ only reuse in the whole kernel → give it L1    └─ same, now with zero competition
```

Compiler report for the new kernels (`nvcc -Xptxas -v`, sm_80):

| Kernel               | Registers | Spills | Shared memory |
|----------------------|-----------|--------|---------------|
| `hybrid_v2_spmv`     | 32        | 0      | 128 B         |
| `hybrid_v2_finalize` | 31        | 0      | 0             |

Why 32 registers matters (P3): an A30 SM has 65,536 registers; at 32 regs/thread
a 256-thread block uses 8,192 — up to 8 such blocks could be resident per SM in
the register dimension, so occupancy is limited by other factors (shared, warps)
rather than registers. Any spill (register → local memory → HBM) would add
hidden traffic to the most latency-sensitive loop; `--warn-on-spills` in the
Makefile turns that into a build-time error signal.

---

## 5. Measuring correctly

### 5.1 Correctness check in every GPU driver

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

### 5.2 Median next to the mean

Fifty runs averaged arithmetically let one outlier (clock ramp, another job on
the shared node) shift the number. The driver keeps printing the mean under the
label the extraction script already parses, and adds:

```
Median execution time: 0.000229 seconds (min 0.000226, 50 runs, L2 flush: off)
Median memory bandwidth (estimated): 671.2 GB/s
```

### 5.3 Optional L2 flush

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

```text
L2-residency picture (P4):

  HBM (24 GB)                      L2 (24 MB)                     SM
  ┌──────────────┐   warm-up run   ┌──────────────┐   timed run   ┌────┐
  │ 662_bus      │ ──────────────► │ 662_bus      │ ────────────► │ OK │
  │ (~1 MB)      │   copy to L2    │ (fits whole) │  L2 hit!      │    │
  └──────────────┘                 └──────────────┘               └────┘
  Reported "bandwidth" is L2 bandwidth (~2–3 TB/s capable) unless flushed.
  flush_l2=1 writes 64 MB of garbage → evicts the matrix → timed run hits HBM.
```

### 5.4 Bandwidth accounting that includes the plan

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

## 6. Build, test and pipeline

### 6.1 Preprocessing in plain C with a unit test

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

### 6.2 Scripts and README

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

## 7. What is still unmeasured

Everything compiles for sm_80 and the host logic is tested, but the kernel has
not yet run on the A30. Expected outcome from the analysis:

| Matrix      | Hybrid Adaptive | Target for Hybrid V2         |
|-------------|-----------------|------------------------------|
| mawi        | 226 ms          | < 20 ms                      |
| Zd_Jac3_db  | 19 µs           | < 12 µs                      |
| others      | see §2          | no regression, small gains   |

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
