# Phase 24: current profile and RHS/RK4 fusion analysis

Date: 2026-08-25

## Valid profile

The first attempt at this profile was cancelled because a wrapper submitted
through `env` did not expose its `#HPC --cpu=60` header and received only two
CPUs.  That run is excluded.  The valid rerun used wrapper
`hpc_abe_current_profile.sh`, job `160888`, with the intended 60 logical CPUs
(30 physical cores), 30 bound OpenMP workers, and the accepted configuration:
arena ON, direct same-level Sync ON, THP/LTO/SVE OFF, direct AMR OFF, pairwise
prolong3 ON, and rejected RHS fusions OFF.

The `perf stat` pass reported:

- Evolve `t=0..4`: **28.8042 s**;
- total run including initialization: `30.6296 s`;
- `23.998` CPUs utilized, zero migrations and zero context switches;
- IPC `1.41`, branch miss `0.47%`, L1D miss `4.17%`;
- LLC miss `49.31%`, dTLB miss `1.43%` after the arena change;
- no lost samples in `perf record`.

The record pass measured 29.5284 s for Evolve; this is expected to be slower
because sampling and call-graph unwinding run concurrently.  It is used for
the hotspot map, not as the A/B timing number.

## Hotspot map

| flat self samples | share | interpretation |
|---|---:|---|
| `compute_rhs_bssn_` | 51.05% | dominant arithmetic and array traffic |
| `__memcpy_sve` | 9.68% | ghost/sync and transfer copies |
| `lopsided_core_` | 8.83% | advection stencil and boundary staging |
| `__memset_sve_zva64` | 5.37% | derivative/boundary initialization |
| `fdderivs_` | 4.63% | second derivatives |
| `fderivs_` | 2.54% | first derivatives |
| `prolong3_pair_kernel_` | 2.52% | AMR interpolation |
| `kodis_` | 2.46% | dissipation stencil |
| `rungekutta4_rout_` | 1.94% | state/RHS update |
| `enforce_ga_` | 1.63% | algebraic constraint enforcement |
| `restrict3_` | 1.41% | AMR restriction |

The call graph assigns the local data paths as follows:

- `omp_execute_cached_sync` is 6.57% of the worker path; its `copy_` child is
  5.28% and reaches `memcpy`;
- `omp_local_transfer` is 6.34%; `prolong3` contributes 3.60% and
  `restrict3` 1.94%;
- inside RHS, lopsided, fdderivs, fderivs and kodis remain separate kernels,
  so their samples are not a single hidden MPI/synchronization cost.

The hottest source lines are the six large Ricci expressions in
`bssn_rhs.f90` (for example lines 655, 711, 750, 789, and 828), followed by
the derivative calls and their boundary work.  These expressions repeatedly
load connection and metric-derivative fields; they are the best remaining
arithmetic target, but large fusion risks register spill as the previous
connection/Aij/chi-Ricci experiments demonstrated.

OpenMP diagnostics show the expected hierarchy: level 7 utilization is 86.4%
and level 8 is 88.6%, while levels 0--4 have too few blocks to fill 30 workers.
The moving levels therefore have useful but not perfect balance; simply
increasing worker count or changing to SMT was already measured as a large
regression.  The remaining gap is primarily level dependencies, Sync/transfer
work, and the serial portion of each block's RHS, not failed core binding.

## Why RHS/RK4 fusion was not implemented

At RK stage 0, `rungekutta4_rout` writes the predictor state but must preserve
the raw RHS.  At stages 1 and 2 it updates that same RHS in place with
`f_rhs = f_rhs + 2*f1`; stage 3 consumes both arrays for the final combination.
Between stages, `SynchList_pre`/`SynchList_cor` are synchronized and their field
lists are swapped.  The RHS is also needed by later stages and is not a
single-consumer temporary.  Therefore a fused `compute_rhs -> RK` loop would
still have to store the full RHS, or would have to recompute it and violate the
RK dataflow.  It would add a second large live array set to the already large
RHS loop and likely create spills, while eliminating only one later read.

This is not a safe “remove one memcpy” transformation.  The measured RK4
routine is only 1.94% flat, and the dependency proof gives no credible path to
the 1% acceptance threshold.  It is therefore rejected without code changes.

## Remaining useful work

1. Keep the contiguous per-block field arena as the production memory change;
   it reduced dTLB misses from about 3.7% to 1.4% and improved Evolve by about
   0.65%.
2. Keep THP hints, broad SVE flags, and LTO disabled: THP regressed 0.35% on
   the arena, SVE regressed 13.5%, and LTO improved only 0.32%.
3. Further gains require local RHS work: reduce repeated loads or tile one
   Ricci/connection producer-consumer pair while checking vector width and
   spills after every change.  Do not fuse all six Ricci expressions.
4. The next practical target is the 9.68% copy path, but only through a new
   geometry proof that removes additional bytes beyond the already-enabled
   direct same-level Sync path.  Generic allocation or MPI changes will not
   address this remaining cost.

Artifacts: `profile/abe-20260825T031721Z-15`.
