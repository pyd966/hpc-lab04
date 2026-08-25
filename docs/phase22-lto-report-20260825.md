# Phase 22: link-time optimization for ABE

Date: 2026-08-25

## Motivation

The ABE executable is a mixed C++/Fortran program with many small wrapper
calls around large Fortran kernels.  GCC LTO can inline or simplify across
translation units, so it was tested as a low-risk compiler experiment.  The
option `AMSS_ENABLE_LTO` adds `-flto` to ABE compile and link steps only; it is
OFF by default and does not affect TwoPuncture or the GPU target.

## Experiment

Job `160721` used arena ON, THP OFF, one OpenMP-only process, 30 bound
workers, 24 static and 30 moving workers, `dynamic,1`, direct Sync ON,
prolong3 pair reuse ON, rejected RHS fusions OFF, and `-O3 -g`.  Run order was
`OFF, ON, ON, OFF`; all four output comparisons were bitwise PASS.

| candidate | Evolve runs (s) | mean (s) |
|---|---:|---:|
| LTO OFF | 28.7387, 28.7092 | 28.7240 |
| LTO ON | 28.6061, 28.6558 | 28.6310 |

The application timer improved by **0.32%**, below the 1% acceptance
threshold.  Hardware counters were consistent with a small code-generation
change rather than a memory improvement: retired instructions fell by about
0.9%, IPC moved from about 1.41 to 1.40, LLC miss rate stayed near 49.4%, and
dTLB miss rate stayed near 1.4%.  The LTO executable was smaller (about 3.7 MB
versus 5.0 MB in this build), but no measured cache/TLB counter shows a large
working-set reduction.  The perf wall time includes initialization and was
more variable; the application Evolve timer is the relevant comparison.

## Decision

LTO remains available for experiments but stays **OFF** in the production
configuration.  The measured gain is real in direction but too small to justify
less transparent profiling and longer, more memory-intensive builds.  A future
LTO revisit should only happen together with a larger algorithmic change or a
different compiler; changing link flags alone is exhausted for this target.

Artifacts: `profile/abe-lto-20260825T024051Z-14`.
