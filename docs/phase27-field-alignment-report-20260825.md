# Phase 27: contiguous arena cache-line alignment

Date: 2026-08-25

## Candidate

The persistent floating-point fields already use one contiguous per-block arena
(`AMSS_ENABLE_BLOCK_FIELD_ARENA=ON`). The candidate changed only the arena base
allocation from `malloc` to `posix_memalign(..., 64, ...)`, leaving field order,
first-touch order, and all numerical kernels unchanged. It was controlled by a
temporary `AMSS_ENABLE_BLOCK_FIELD_ALIGN` switch and did not use `-Ofast` or a
new ISA flag.

This was a low-risk P8 experiment, but the current allocation is already served
by a cache-line-aligned allocator on the tested platform often enough that a
benefit was not guaranteed. The experiment was run from `/tmp` so perf output
would not consume the nearly full home filesystem.

## Measurement

The first exploratory job `161546` ran one OFF and one ON pass:

| candidate | Evolve (s) | average CPUs | L1D miss | LLC miss | dTLB miss |
|---|---:|---:|---:|---:|---:|
| OFF | 28.4756 | 24.112 | 4.15% | 49.09% | 1.43% |
| ON | 28.8769 | 23.853 | 4.20% | 49.51% | 1.46% |

Because that was not interleaved, it was treated as exploratory only. The
required repeated job `161573` used `OFF ON ON OFF` with the same fixed input,
30 bound OpenMP workers, and static/moving block targets `24/30`:

| candidate | runs (Evolve seconds) | mean | mean CPUs |
|---|---:|---:|---:|
| OFF | 28.7518, 28.5935 | 28.6727 | 24.030 |
| ON | 28.9255, 28.9159 | 28.9207 | 23.816 |

The aligned version is **0.86% slower**. Its average L1D miss was 4.19% vs
4.14%, LLC miss 49.51% vs 49.24%, and dTLB miss 1.46% vs 1.445%. All four runs
passed the course checker and the output rows were bitwise identical.

## Decision

The alignment switch and implementation were removed. The existing
`malloc`-backed contiguous arena remains the production path. The result is
consistent with the current single-NUMA-node placement: base alignment alone
does not improve locality, and the extra allocation path slightly perturbs
placement. Future NUMA work should use a real cross-node allocation and
first-touch experiment, not reintroduce this local-only alignment change.

Artifacts: `profile/abe-field-align-20260825T052035Z-14` and
`profile/abe-field-align-20260825T052415Z-14`.
