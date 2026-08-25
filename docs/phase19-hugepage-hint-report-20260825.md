# Phase 19: persistent-field transparent huge-page hint

Date: 2026-08-25

## Motivation

The ABE profile still reports roughly 49% LLC-load misses and 3.7% dTLB
load misses.  `Block` allocates every persistent floating-point grid field
with a separate `malloc`, then keeps it for the whole evolution.  Linux
transparent huge pages (THP) can reduce page-table pressure, so this phase
tested a low-risk `madvise(MADV_HUGEPAGE)` hint on those allocations.

The change is controlled by `AMSS_ENABLE_HUGEPAGE_HINT` and is **OFF by
default**.  It does not change allocation, indexing, arithmetic, or output
ordering.  The profile script records the node THP policy and memory-page
counters so that a timing result is not interpreted without the OS context.

## Experiment

Job `160292` used the accepted configuration: one OpenMP-only process, 30
bound workers, static target/thread count 24, moving target/thread count 30,
`dynamic,1`, direct same-level Sync ON, prolong3 pair reuse ON, all rejected
RHS fusions OFF, and `-O3 -g`.  The fixed input was run in the order
`OFF, ON, ON, OFF`; every output comparison was bitwise PASS.

| candidate | Evolve runs (s) | mean (s) |
|---|---:|---:|
| OFF | 29.1341, 28.9943 | 29.0642 |
| ON | 29.3870, 29.5826 | 29.4848 |

The hint was **1.45% slower** on the two-run mean.  The corresponding
`perf stat` measurements were effectively unchanged:

| metric | OFF | ON |
|---|---:|---:|
| IPC | 1.42 | 1.41--1.42 |
| L1D load miss rate | 4.14--4.15% | 4.13--4.15% |
| LLC load miss rate | 49.29--49.31% | 49.34--49.43% |
| dTLB load miss rate | 3.75% | 3.70--3.76% |

The node reported THP policy `[always] madvise never` and 2 MiB huge pages.
The largest block is `48*96*24` cells, so one `double` field is about 0.84
MiB.  Since each field is a separate allocation smaller than 2 MiB, the
kernel has little opportunity to promote an individual allocation to a full
huge page.  The unchanged TLB rate confirms that this is what happened in
practice.  The small regression is consistent with page-promotion/allocator
bookkeeping cost without a compensating reduction in page walks.

## Decision

The THP hint remains disabled in production.  This is not evidence that huge
pages are useless for the application; it shows that the current per-field
allocation layout prevents the hint from helping.  The follow-up experiment
is to allocate all fields of one block from one contiguous arena, then test
that layout with and without the hint.  That separates allocator/locality
effects from THP effects and creates multi-megabyte ranges that can actually
be promoted.

Artifacts: `profile/abe-hugepage-20260825T003424Z-13`.
