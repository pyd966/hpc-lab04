# Phase 26: two-field first-connection RHS fusion

Date: 2026-08-25

## Scope and hardware limitation

The planned 60-physical-core calibration is still unavailable. The lab4
request provides 60 logical CPUs, corresponding to 30 physical ARM cores with
SMT. This phase therefore uses the accepted 30-worker OpenMP configuration:
`OMP_PLACES=cores`, `OMP_PROC_BIND=close`, `dynamic,1`, and static/moving
block/thread targets `24/30`.

## Source analysis

In `src/bssn_rhs.f90`, the first-kind connection fields `gxxx`, `gxxy`, and
`gxxz` were produced by three whole-array expressions immediately before the
physical connection update. Each expression is pointwise independent and all
three outputs are consumed later, so a loop over the contiguous `i` direction
is mathematically safe. However, fusing all three creates three simultaneous
output streams and did not reduce the SIMD width.

Two variants were tested behind separate CMake switches:

- `AMSS_ENABLE_RHS_FIRST_CONNECTION_FUSION`: fuse all three fields;
- `AMSS_ENABLE_RHS_FIRST_CONNECTION_PAIR_FUSION`: fuse only `gxxx` and `gxxy`,
  while leaving `gxxz` as the original array expression.

The pair version keeps the transformation local and leaves the later dataflow,
stencil calls, and arithmetic order unchanged. The new loop is marked
`!$omp simd`; GCC reports 128-bit vectorization on both the original and fused
paths. No `-Ofast`, fast-math, or new ISA flag was used.

## Three-field experiment: rejected

HPC job `161351`, artifact
`profile/abe-rhs-first-connection-20260825T045027Z-15`, used the same fixed
input and run order `off on on off`.

| candidate | runs (Evolve seconds) | mean | mean CPUs |
|---|---:|---:|---:|
| original | 28.7643, 28.8453 | 28.8048 | 22.528 |
| three-field fusion | 29.0092, 28.9206 | 28.9649 | 22.534 |

The all-three version regressed by about **0.56%**. It passed the course
checker and was bitwise identical, but instructions increased slightly and the
compiler still used 128-bit vectors. The RHS function stack frame stayed at
`0x5e0`, so this was not a large stack-spill event; the likely cost is the
additional address/load/store organization in a loop with three output streams.
The switch remains OFF.

## Two-field experiment: accepted

HPC job `161401`, artifact
`profile/abe-rhs-first-connection-20260825T045813Z-14`, used the same run order.

| candidate | runs (Evolve seconds) | mean | mean CPUs |
|---|---:|---:|---:|
| original | 28.7093, 28.7903 | 28.7498 | 22.416 |
| pair fusion (`gxxx/gxxy`) | 28.4214, 28.4270 | 28.4242 | 22.266 |

The pair version is **1.13% faster** than its interleaved baseline. Every run
passed the course checker and the four numerical output files were bitwise
identical after their timestamp headers. Hardware counters were stable or
slightly better:

| metric | original mean | pair mean | interpretation |
|---|---:|---:|---|
| IPC | 1.41 | 1.42 | no throughput collapse |
| L1D miss | 4.18% | 4.15% | slight improvement |
| LLC load miss | 49.32% | 49.14% | slight improvement |
| dTLB miss | 1.44% | 1.44% | unchanged |

The pair function has the same `sub sp, sp, #0x5e0` frame allocation as the
baseline. Its code is about 0.16% larger, but the measured instruction count is
marginally lower and no new stack traffic is visible. The gain is consistent
with eliminating one full-grid traversal without over-expanding the live
output set.

## Production decision

The pair fusion is now the default (`ON`) in `CMakeLists.txt` and the profile
wrappers. The three-field variant remains available only as an explicit
experiment and defaults to `OFF`. The generic profile wrapper records both
switches so a future rollback is one CMake/environment change.

This phase does not change OpenMP task geometry or synchronization. It removes
one pointwise array pass inside `compute_rhs_bssn`; the post-commit profile will
verify that the flat RHS share and copy path remain stable under the default
configuration.

Artifacts: `profile/abe-rhs-first-connection-20260825T045813Z-14` and
`profile/abe-rhs-first-connection-20260825T045027Z-15`.
