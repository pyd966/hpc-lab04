# Phase 20: contiguous floating-point field arena

Date: 2026-08-25

## Motivation and implementation

The previous THP experiment showed that each `fgfs[i]` allocation was too
small for a 2 MiB page to be useful.  The next experiment changed only the
owner of those allocations: with `AMSS_ENABLE_BLOCK_FIELD_ARENA=ON`, one
`malloc` reserves all floating-point fields of a block and `fgfs[i]` points to
its fixed offset inside that range.  The old per-field allocation path remains
available and is now the production default.  Destruction,
field swapping, indexing, GPU shadow pointers, and arithmetic are unchanged.

This tests allocator/address locality independently of THP; the THP hint was
explicitly OFF in both candidates.

## Experiment

Job `160644` used one OpenMP-only process, 30 bound workers, static target and
thread count 24, moving target and thread count 30, `dynamic,1`, direct
same-level Sync ON, prolong3 pair reuse ON, rejected RHS fusions OFF, and
`-O3 -g`.  Run order was `OFF, ON, ON, OFF`; all four output comparisons were
bitwise PASS.

| candidate | Evolve runs (s) | mean (s) |
|---|---:|---:|
| OFF: separate field allocations | 29.4707, 29.2128 | 29.3418 |
| ON: one block arena | 29.1912, 29.1078 | 29.1495 |

The arena is **0.65% faster** on the application timer.  The kernel-level
`perf stat` elapsed time averages 33.78 s (OFF) versus 33.22 s (ON), about
1.7% faster.  The counters explain why the improvement is modest but real:

| metric | separate allocations | block arena |
|---|---:|---:|
| IPC | 1.42 | 1.41--1.42 |
| L1D load miss rate | 4.12--4.14% | 4.18--4.19% |
| LLC load miss rate | 49.48--49.50% | 49.24--49.33% |
| dTLB load miss rate | 3.73--3.74% | 1.41--1.43% |
| branch miss rate | 0.47% | 0.42--0.43% |

The large dTLB reduction is consistent across both ON runs.  LLC misses are
nearly unchanged, so this is not a general cache-locality transformation;
fewer independent allocation ranges and more regular page traversal reduce
translation pressure.  IPC is unchanged, confirming that the computation
itself was not altered.  The small L1D increase is within the expected cost of
the denser layout and does not offset the TLB benefit.

## Decision and next test

This is a useful optimization and is a candidate for the production default
after one additional combined test.  The next experiment keeps the arena and
enables `MADV_HUGEPAGE` on the whole range.  A positive result would make the
arena and hint a compound optimization; a neutral or negative result will
still leave the arena enabled and the hint disabled.

Artifacts: `profile/abe-field-arena-20260825T022210Z-15`.
