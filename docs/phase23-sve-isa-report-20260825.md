# Phase 23: ABE-only SVE ISA flags

Date: 2026-08-25

## Motivation

The node exposes SVE, while the default ABE build reports 16-byte vector
loops.  The earlier `-mcpu=native` experiment was a broad 9% regression, so
this phase isolated the ISA capability from microarchitecture scheduling:
only the ABE ON build received `-march=armv8.2-a+sve`; all other settings,
including arena ON and THP OFF, were unchanged.

## Experiment

Job `160755` ran `OFF, ON, ON, OFF` with 30 bound OpenMP workers, static 24,
moving 30, direct Sync ON, prolong3 pair reuse ON, rejected RHS fusions OFF,
and `-O3 -g`.  All four output comparisons were bitwise PASS.

| candidate | Evolve runs (s) | mean (s) |
|---|---:|---:|
| default ISA | 29.1181, 29.0556 | 29.0869 |
| `-march=armv8.2-a+sve` | 32.9673, 33.0608 | 33.0141 |

The explicit SVE build was **13.5% slower**.  `perf stat` explains the
regression: IPC fell from about 1.41 to 1.00, L1D load miss rate rose from
4.17--4.19% to 5.47--5.51%, and cycles increased by roughly 10% even though
retired instructions decreased by about 22%.  LLC miss rate stayed near 49%
and dTLB stayed near 1.4--1.6%, so SVE did not solve the dominant memory
working-set problem.  GCC's report changed many loops from fixed 16-byte
vectors to **variable-length vectors**, adding strip-mining/predicate and
register pressure costs.  The result is a wider-looking ISA path with lower
throughput on this workload.

## Decision

Keep the default ABE architecture flags empty.  Do not enable SVE,
`-mcpu=native`, or `-funroll-loops` globally; the earlier native result and
this isolated ISA result agree that wider/more aggressive code generation is
not a win for the current RHS and transfer mix.  Future SIMD work should stay
local to kernels where the compiler's fixed-width vector report and counters
show a benefit.

Artifacts: `profile/abe-sve-20260825T024947Z-15`.
