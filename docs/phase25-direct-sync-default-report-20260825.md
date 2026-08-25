# Phase 25: make validated direct Sync the default

Date: 2026-08-25

## Scope

The same-level OpenMP Sync direct path was already implemented and validated in
Phase 8, but the generic CMake option and the generic profile script still
defaulted to the old pack/unpack path.  This phase changes only those defaults:

- `AMSS_ENABLE_OMP_DIRECT_SYNC` in `CMakeLists.txt`: `OFF` -> `ON`;
- `DIRECT_SYNC` in `hpc_abe_profile.sh`: `OFF` -> `ON`.

The direct path still performs the existing rectangle-overlap proof and falls
back to pack/unpack for unsafe operations.  No numerical kernel or task
partition was changed.  Explicit `-DAMSS_ENABLE_OMP_DIRECT_SYNC=OFF` remains
available for regression comparisons.

## A/B experiment

HPC job `161231` used one OpenMP-only process, 30 bound workers, static/moving
thread geometry `24/30`, `dynamic,1`, `-O3 -g`, and `t=0..4`.  The run order was
`pack direct direct pack`.

| candidate | Evolve runs (s) | mean (s) | mean active CPUs |
|---|---:|---:|---:|
| pack/unpack | 29.3067, 29.2014 | 29.25405 | 24.188 |
| direct Sync | 28.9582, 28.9544 | 28.95630 | 24.005 |

The direct path is about **1.02% faster** on the application Evolve timer.  The
slightly lower average CPU count is not a regression: the candidate finishes
sooner, while IPC and cache/TLB rates remain effectively unchanged.

All four runs passed the course checker.  The four numerical output files were
bitwise identical after ignoring their timestamp headers.  The direct-path
diagnostics reported `direct_safe=1` for the major same-level plans.

## Decision

This is a low-risk production improvement because it removes one intermediate
workspace copy only where the existing geometry proof establishes that direct
source-to-ghost writes are safe.  The default is now ON.  AMR transfer direct
output remains OFF, and the lopsided/fderivs batch experiments remain OFF.

Artifacts: `profile/abe-sync-direct-20260825T043044Z-14`.
