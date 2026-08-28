# Phase 18: RHS chi-Ricci producer-consumer fusion

Date: 2026-08-25

## Motivation

After `fdderivs(ex,chi,...)`, the code applies six covariant corrections to
`fxx..fzz`, forms the scalar `f`, and updates six Ricci components.  The
corrected derivative arrays and `f` are overwritten by the next
`fdderivs(ex,Lap,...)` call, so they are dead after this block.  This made the
block a better memory-traffic candidate than the previously rejected
multi-output connection and Aij loops.

The option `AMSS_ENABLE_RHS_CHI_RICCI_FUSION` is disabled by default.  Both
experiments used 30 bound OpenMP workers, static 24, moving 30, `dynamic,1`,
direct Sync ON, pairwise prolong3 ON, and `-O3 -g`.

## Experiment A: full producer-consumer loop

One SIMD loop kept six corrected derivatives, `f`, and all six Ricci updates
in SIMD-private scalars.  Job `160221` produced four bitwise-identical PASS
runs:

| candidate | Evolve runs (s) | mean (s) |
|---|---:|---:|
| OFF | 29.1751, 29.2414 | 29.2083 |
| ON | 29.6503, 29.6263 | 29.6383 |

This was a **1.47% regression**.  Retired instructions decreased by about
0.5%, but IPC fell from 1.41 to 1.39 and dTLB misses rose from about 3.75%
to 4.42%.  The compiler did vectorize the loop, so the loss came from the
larger live SIMD expression and register/cache pressure.

## Experiment B: fuse only intermediate production

The second implementation kept one loop for the six corrected derivatives and
`f`, wrote those seven arrays, then used the original six array expressions
for the Ricci updates.  This reduced the live output set in the long loop.
Job `160239` again produced four bitwise-identical PASS runs:

| candidate | Evolve runs (s) | mean (s) |
|---|---:|---:|
| OFF | 29.4263, 29.3812 | 29.4038 |
| ON | 29.5197, 29.5044 | 29.5121 |

The remaining difference was a **0.37% regression**, not a reproducible
speedup.  ON still lowered IPC from about 1.42 to 1.41 and raised dTLB miss
rate from 3.74% to 3.94%.  The lower regression confirms that splitting the
Ricci updates helped, but the producer loop itself still carries enough live
data to offset the saved array passes.

## Decision

Neither implementation is a production optimization.  The option remains
`OFF`, and the original array expressions remain the fallback.  The two
experiments establish a useful boundary: reducing array stores is not enough
when the replacement loop increases the SIMD live set.  Further algebraic
fusion inside this block is unlikely to help without a data-layout or tile
change that lowers the number of simultaneously active fields.

The next useful direction is therefore bounded tiling/field layout or outer
OpenMP task geometry.  It should be measured with cache/TLB counters before
changing the arithmetic again.
