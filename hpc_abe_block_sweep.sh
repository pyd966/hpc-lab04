#!/bin/bash
#HPC --partition=lab4
#HPC --cpu=60
#HPC --mem=100Gi
#HPC --time=30m
#HPC --name=abe-block-sweep
#HPC --output=profile/hpc_%x_%j.log
set -euo pipefail

ROOT_DIR="${AMSS_ROOT_DIR:-$PWD}"
if [[ ! -f "$ROOT_DIR/CMakeLists.txt" ]]; then
    echo "submit this script from the repository root" >&2
    exit 2
fi
cd "$ROOT_DIR"

RUN_ID="${HPC_JOB_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$BASHPID}"
OUT_DIR="$ROOT_DIR/profile/abe-block-sweep-$RUN_ID"
BUILD_DIR="$OUT_DIR/build"
PREP_ROOT="$OUT_DIR/prepare"
CACHE_DIR="$ROOT_DIR/profile/twopuncture-cache"
mkdir -p "$OUT_DIR" "$PREP_ROOT" "$CACHE_DIR"
exec > >(tee "$OUT_DIR/job.log") 2>&1

OMP_THREADS="${ABE_SWEEP_OMP_THREADS:-30}"
EVOLVE_TIME="${ABE_SWEEP_TIME:-4.0}"
TARGETS="${ABE_SWEEP_TARGETS:-18 24 30 36}"

echo "git revision: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "OpenMP threads: $OMP_THREADS"
echo "evolution interval: t=0..$EVOLVE_TIME"
echo "block targets: $TARGETS"

export JOBS="$(nproc)"
export OMP_NUM_THREADS="$OMP_THREADS"
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export AMSS_BUILD_DIR="$BUILD_DIR"

./compile.sh \
    -DAMSS_ENABLE_GPU=OFF \
    -DAMSS_ENABLE_OPENMP=ON \
    -DAMSS_ENABLE_OMP_ONLY=ON \
    -DAMSS_ENABLE_TWOPUNCTURE_OPENMP=ON \
    -DAMSS_OPT='-O3 -g -fno-omit-frame-pointer' \
    -DAMSS_TWOPUNCTURE_OPT=-O3 \
    -DAMSS_TWOPUNCTURE_ARCH_FLAGS=-march=native \
    2>&1 | tee "$OUT_DIR/build.log"

export AMSS_OUTPUT_ROOT="$PREP_ROOT"
export AMSS_CACHE_DIR="$CACHE_DIR"
export AMSS_NCKU_PREPARE_ONLY=1
export AMSS_OMP_ONLY_RUN=1
./run.sh --twop-cache
unset AMSS_NCKU_PREPARE_ONLY

BASE_DIR="$PREP_ROOT/GW250118/AMSS_NCKU_output"
: > "$OUT_DIR/timings.tsv"
printf 'target\tevolve_seconds\ttotal_seconds\taverage_cpus\tcourse_check\n' \
    >> "$OUT_DIR/timings.tsv"

for target in $TARGETS; do
    run_dir="$OUT_DIR/target-$target"
    mkdir -p "$run_dir/binary_output"
    cp "$BUILD_DIR/ABE" "$run_dir/ABE"
    cp "$BASE_DIR/Ansorg.psid" "$run_dir/Ansorg.psid"
    sed -E \
        "s/^(ABE::TotalTime[[:space:]]*=[[:space:]]*).*/\\1$EVOLVE_TIME/" \
        "$BASE_DIR/input.par" > "$run_dir/input.par"

    export AMSS_OMP_BLOCK_TARGET="$target"
    echo "=== Block target $target ==="
    perf stat -o "$run_dir/perf-stat.txt" -- \
        bash -c 'cd "$1" && exec ./ABE < /dev/null' _ "$run_dir" \
        2>&1 | tee "$run_dir/run.log"

    ./check.sh "$run_dir/binary_output" | tee "$run_dir/check.txt"
    course_check="$(awk '/^FINAL:/ {result=$2} END {print result}' "$run_dir/check.txt")"

    evolve="$(awk '/Total Evolve Time:/ {value=$4} END {print value}' "$run_dir/run.log")"
    total="$(awk '/Total Running Time:/ {value=$4} END {print value}' "$run_dir/run.log")"
    task_ms="$(awk '/task-clock/ {gsub(/,/, "", $1); print $1}' "$run_dir/perf-stat.txt")"
    elapsed="$(awk '/seconds time elapsed/ {gsub(/,/, "", $1); print $1}' "$run_dir/perf-stat.txt")"
    avg="$(awk -v task="$task_ms" -v elapsed="$elapsed" 'BEGIN {printf "%.3f", task / 1000 / elapsed}')"
    printf '%s\t%s\t%s\t%s\t%s\n' "$target" "$evolve" "$total" "$avg" "$course_check" \
        | tee -a "$OUT_DIR/timings.tsv"
done

echo "=== Summary ==="
column -t -s $'\t' "$OUT_DIR/timings.tsv" || cat "$OUT_DIR/timings.tsv"
echo "Artifacts: $OUT_DIR"
