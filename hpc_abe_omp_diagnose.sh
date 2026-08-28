#!/bin/bash
#HPC --partition=lab4
#HPC --cpu=60
#HPC --mem=100Gi
#HPC --time=30m
#HPC --name=abe-omp-diagnose
#HPC --output=profile/hpc_%x_%j.log
set -euo pipefail

ROOT_DIR="${AMSS_ROOT_DIR:-$PWD}"
if [[ ! -f "$ROOT_DIR/CMakeLists.txt" ]]; then
    echo "submit this script from the repository root" >&2
    exit 2
fi
cd "$ROOT_DIR"

RUN_ID="${HPC_JOB_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$BASHPID}"
OUT_DIR="$ROOT_DIR/profile/abe-omp-diagnose-$RUN_ID"
BUILD_DIR="$OUT_DIR/build"
PREP_ROOT="$OUT_DIR/prepare"
CACHE_DIR="$ROOT_DIR/profile/twopuncture-cache"
RUN_DIR="$OUT_DIR/run"
mkdir -p "$OUT_DIR" "$PREP_ROOT" "$CACHE_DIR" "$RUN_DIR/binary_output"
exec > >(tee "$OUT_DIR/job.log") 2>&1

OMP_THREADS="${ABE_OMP_DIAG_THREADS:-30}"
STATIC_BLOCKS="${ABE_OMP_DIAG_STATIC_BLOCKS:-24}"
STATIC_THREADS="${ABE_OMP_DIAG_STATIC_THREADS:-24}"
MOVING_BLOCKS="${ABE_OMP_DIAG_MOVING_BLOCKS:-30}"
MOVING_THREADS="${ABE_OMP_DIAG_MOVING_THREADS:-30}"
EVOLVE_TIME="${ABE_OMP_DIAG_TIME:-4.0}"

echo "git revision: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "OpenMP threads: $OMP_THREADS"
echo "static blocks/threads: $STATIC_BLOCKS/$STATIC_THREADS"
echo "moving blocks/threads: $MOVING_BLOCKS/$MOVING_THREADS"
echo "evolution interval: t=0..$EVOLVE_TIME"

"$ROOT_DIR/scripts/collect_cpu_info.sh" > "$OUT_DIR/cpu-info.txt"

export JOBS="$(nproc)"
export OMP_NUM_THREADS="$OMP_THREADS"
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export AMSS_OMP_STATIC_BLOCK_TARGET="$STATIC_BLOCKS"
export AMSS_OMP_STATIC_THREADS="$STATIC_THREADS"
export AMSS_OMP_MOVING_BLOCK_TARGET="$MOVING_BLOCKS"
export AMSS_OMP_MOVING_THREADS="$MOVING_THREADS"
export AMSS_OMP_ONLY_RUN=1
export AMSS_BUILD_DIR="$BUILD_DIR"

./compile.sh \
    -DAMSS_ENABLE_GPU=OFF \
    -DAMSS_ENABLE_OPENMP=ON \
    -DAMSS_ENABLE_OMP_ONLY=ON \
    -DAMSS_ENABLE_OMP_DIAGNOSTICS=ON \
    -DAMSS_ENABLE_TWOPUNCTURE_OPENMP=ON \
    -DAMSS_OPT='-O3 -g -fno-omit-frame-pointer' \
    -DAMSS_TWOPUNCTURE_OPT=-O3 \
    -DAMSS_TWOPUNCTURE_ARCH_FLAGS=-march=native \
    2>&1 | tee "$OUT_DIR/build.log"

export AMSS_OUTPUT_ROOT="$PREP_ROOT"
export AMSS_CACHE_DIR="$CACHE_DIR"
export AMSS_NCKU_PREPARE_ONLY=1
./run.sh --twop-cache
unset AMSS_NCKU_PREPARE_ONLY

BASE_DIR="$PREP_ROOT/GW250118/AMSS_NCKU_output"
cp "$BUILD_DIR/ABE" "$RUN_DIR/ABE"
cp "$BASE_DIR/Ansorg.psid" "$RUN_DIR/Ansorg.psid"
sed -E \
    "s/^(ABE::TotalTime[[:space:]]*=[[:space:]]*).*/\\1$EVOLVE_TIME/" \
    "$BASE_DIR/input.par" > "$RUN_DIR/input.par"

perf stat -d -d -o "$OUT_DIR/perf-stat.txt" -- \
    bash -c 'cd "$1" && exec ./ABE < /dev/null' _ "$RUN_DIR" \
    2>&1 | tee "$OUT_DIR/run.log"

rg '^OMP_DIAG' "$OUT_DIR/run.log" > "$OUT_DIR/omp-diagnostics.txt"
set +e
./check.sh "$RUN_DIR/binary_output" | tee "$OUT_DIR/check.txt"
set -e

echo "=== OpenMP diagnostics ==="
column -t "$OUT_DIR/omp-diagnostics.txt" || cat "$OUT_DIR/omp-diagnostics.txt"
echo "Artifacts: $OUT_DIR"
