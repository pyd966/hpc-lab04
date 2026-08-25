# Phase 21: transparent huge pages on the contiguous arena

Date: 2026-08-25

## Question

Phase 20 made all floating-point fields of one block contiguous and reduced
dTLB misses substantially.  This phase tested whether Linux THP could add a
second benefit now that the range is large enough.  Both candidates retained
`AMSS_ENABLE_BLOCK_FIELD_ARENA=ON`; only `AMSS_ENABLE_HUGEPAGE_HINT` changed.

## Experiment

Job `160682` used the same fixed configuration and run order `OFF, ON, ON,
OFF`.  The node policy was `[always] madvise never` with 2 MiB huge pages.
All four output comparisons were bitwise PASS.

| candidate | Evolve runs (s) | mean (s) |
|---|---:|---:|
| arena, THP hint OFF | 28.8783, 28.9152 | 28.8968 |
| arena, THP hint ON | 28.9988, 28.9953 | 28.9971 |

The hint was **0.35% slower** on the application timer.  `perf stat` showed
the same behavior at the hardware level: dTLB miss rate was 1.42--1.43% for
both candidates, LLC miss rate was about 49.2--49.3%, and IPC stayed at 1.41.
The node's `AnonHugePages` counter was nonzero, but the application-level
counter did not improve, so the presence of some huge pages is not evidence
that the hot field accesses were promoted.

## Decision

Keep the contiguous arena enabled and the THP hint disabled.  The arena has
already reduced translation pressure enough that THP has no measurable room to
help; the hint adds page-management cost without changing the hot counters.
Further THP work would require OS-level policy changes and is not justified by
this workload.  The optimization switch remains available for experiments but
defaults to `OFF`.

Artifacts: `profile/abe-arena-thp-20260825T023036Z-14`.
