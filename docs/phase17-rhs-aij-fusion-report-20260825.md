# Phase 17: RHS Aij index-raising fusion

Date: 2026-08-25

## Scope

Before the Ricci calculation, `compute_rhs_bssn` raises the six symmetric
components of `Aij` and stores them in `Rxx/Ryy/Rzz/Rxy/Rxz/Ryz`.  The original
code uses six whole-array assignments.  This phase tested one explicit
`do k/j/i` loop with `!$omp simd` for those six independent pointwise outputs.
The build option is `AMSS_ENABLE_RHS_AIJ_FUSION`, default `OFF`.

The A/B runs used the accepted OpenMP-only geometry (30 workers bound to
cores, static 24, moving 30, `dynamic,1`, direct Sync ON, pairwise prolong3
ON), `-O3 -g -fno-omit-frame-pointer`, and `t=0..4`.

## Result

In job `160189`, all four runs passed the course check and were bitwise
identical:

| candidate | Evolve runs (s) | mean (s) |
|---|---:|---:|
| OFF | 29.094, 29.2491 | 29.1716 |
| ON | 29.1276, 29.4966 | 29.3121 |

The explicit loop therefore regressed by about **0.76%**, within the range
where node variation is relevant but not a positive signal.  The counters
show the same tradeoff as the earlier connection experiment: ON retired
slightly fewer instructions, but IPC was around 1.40–1.41 versus 1.41–1.42
for OFF.  L1D and LLC miss rates stayed near 4.1% and 49.5%; there was no
evidence of a synchronization or branch problem.

## Decision

The six independent array expressions are already handled efficiently by
GCC's array-expression lowering.  The explicit multi-output loop does not
make the end-to-end run faster, so the option remains disabled and no change
is included in the production path.  The experiment is kept as a reproducible
fallback for future compiler or data-layout changes.

The next candidate is deliberately different: fuse the chi covariant-
derivative correction, scalar `f`, and six Ricci updates using local values.
Those intermediate arrays are dead immediately after the block, so that
fusion can remove stores rather than merely combine six output expressions.
