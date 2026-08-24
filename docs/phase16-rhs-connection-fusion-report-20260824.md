# Phase 16: RHS connection fusion experiment

Date: 2026-08-24

## Scope

`compute_rhs_bssn` computes 18 second-kind Christoffel components in the
pointwise block at `src/bssn_rhs.f90:294-316`.  The original code uses 18
whole-array assignments.  This phase tested whether those assignments could
share one traversal of `gup**` and the metric derivatives.  The change was
kept behind `AMSS_ENABLE_RHS_CONNECTION_FUSION`; its default remains `OFF`.

The experiment used the accepted OpenMP-only geometry: 30 workers bound to
cores, static target/thread count 24, moving target/thread count 30,
`dynamic,1`, direct same-level Sync enabled, pairwise `prolong3` enabled, and
`-O3 -g -fno-omit-frame-pointer`.

## Experiment A: one 18-output loop

All 18 assignments were put in one `do k/j/i` loop with `!$omp simd` on the
contiguous `i` dimension.  The two OFF runs averaged **29.045 s** Evolve;
the two ON runs averaged **29.626 s**, a **2.0% regression**.  All four
outputs passed the course check and were bitwise identical.

The counters explain the regression rather than a numerical problem.  ON
reduced retired instructions from about 2.98e12 to 2.94e12, but IPC fell from
1.42 to 1.39.  LLC miss rate rose from about 49.3% to 50.2% and dTLB misses
from about 3.7% to 4.3%.  The source-mapped disassembly showed a long
interleaved vector loop with many live vector values and pointer reloads.

## Experiment B: smaller groups

The same pointwise formulas were split into three loops containing 3, 6, and
9 outputs.  This was intended to lower the live-vector count without giving
up input reuse.  In the complete quiet A/B job `160166`, OFF averaged
**29.285 s** and ON averaged **29.490 s**, still a **0.70% regression**.  All
runs were bitwise identical and passed validation.

The ON counters again showed fewer instructions but lower IPC (1.39 versus
1.42), with L1D misses increasing from 4.13% to 4.24% and LLC misses from
49.4% to about 50.0%.  GCC reported SIMD for the new loops, so the failure is
not lack of vectorization; the fused loop's larger working set and scheduling
pressure are the limiting factors.

## Decision

The 18 assignments are already well optimized as independent compiler-
generated array loops.  Combining them saves loop traversals but prevents the
compiler from keeping each short expression's dataflow compact.  Both a
single large fusion and a smaller grouped fusion were measured, so this is
not retained as a production optimization.  `AMSS_ENABLE_RHS_CONNECTION_FUSION`
and the sweep script remain as an opt-in reproducible experiment, with the
default OFF and no change to numerical behavior.

The next RHS candidates should reduce memory traffic without creating a wide
multi-output SIMD body, for example a small producer-consumer fusion around a
single Ricci component or an explicitly tiled, bounded subset.  A full
six-Ricci fusion is not justified by this result and would likely increase
register pressure further.
