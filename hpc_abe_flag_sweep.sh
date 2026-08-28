#!/bin/bash
#HPC --partition=lab4
#HPC --cpu=60
#HPC --mem=100Gi
#HPC --time=30m
#HPC --name=abe-flags
#HPC --output=profile/hpc_%x_%j.log
set -euo pipefail

ROOT_DIR="${AMSS_ROOT_DIR:-$PWD}"
if [[ ! -f "$ROOT_DIR/CMakeLists.txt" ]]; then
    echo "submit this script from the repository root" >&2
    exit 2
fi
cd "$ROOT_DIR"

RUN_ID="${HPC_JOB_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$BASHPID}"
OUT_DIR="$ROOT_DIR/profile/abe-flag-sweep-$RUN_ID"
PREP_ROOT="$OUT_DIR/prepare"
CACHE_DIR="$ROOT_DIR/profile/twopuncture-cache"
mkdir -p "$OUT_DIR" "$PREP_ROOT" "$CACHE_DIR"
exec > >(tee "$OUT_DIR/job.log") 2>&1

OMP_THREADS="${ABE_FLAG_OMP_THREADS:-30}"
STATIC_TARGET="${ABE_FLAG_STATIC_TARGET:-24}"
MOVING_TARGET="${ABE_FLAG_MOVING_TARGET:-30}"
EVOLVE_TIME="${ABE_FLAG_TIME:-4.0}"

names=(o3 native native-unroll)
arch_flags=(
    ""
    "-mcpu=native"
    "-mcpu=native -funroll-loops"
)

echo "git revision: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "OpenMP threads: $OMP_THREADS"
echo "static blocks/threads: $STATIC_TARGET"
echo "moving blocks/threads: $MOVING_TARGET"
echo "evolution interval: t=0..$EVOLVE_TIME"
echo "candidates: ${names[*]}"

"$ROOT_DIR/scripts/collect_cpu_info.sh" > "$OUT_DIR/cpu-info.txt"
{
    gcc -Q -mcpu=native --help=target
    echo "=== predefined macros ==="
    gcc -mcpu=native -dM -E -x c /dev/null
} > "$OUT_DIR/native-target.txt" 2>&1

export JOBS="$(nproc)"
export OMP_NUM_THREADS="$OMP_THREADS"
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export AMSS_OMP_STATIC_BLOCK_TARGET="$STATIC_TARGET"
export AMSS_OMP_MOVING_BLOCK_TARGET="$MOVING_TARGET"
export AMSS_OMP_STATIC_THREADS="$STATIC_TARGET"
export AMSS_OMP_MOVING_THREADS="$MOVING_TARGET"
export AMSS_OMP_ONLY_RUN=1

BASE_DIR=""
for index in "${!names[@]}"; do
    name="${names[$index]}"
    arch="${arch_flags[$index]}"
    build_dir="$OUT_DIR/build-$name"
    export AMSS_BUILD_DIR="$build_dir"

    echo "=== Build $name: -O3 -g -fno-omit-frame-pointer $arch ==="
    ./compile.sh \
        -DAMSS_ENABLE_GPU=OFF \
        -DAMSS_ENABLE_OPENMP=ON \
        -DAMSS_ENABLE_OMP_ONLY=ON \
        -DAMSS_ENABLE_TWOPUNCTURE_OPENMP=ON \
        -DAMSS_OPT="-O3 -g -fno-omit-frame-pointer" \
        -DAMSS_ARCH_FLAGS="$arch" \
        -DAMSS_TWOPUNCTURE_OPT=-O3 \
        -DAMSS_TWOPUNCTURE_ARCH_FLAGS=-march=native \
        2>&1 | tee "$OUT_DIR/build-$name.log"

    if [[ -z "$BASE_DIR" ]]; then
        export AMSS_OUTPUT_ROOT="$PREP_ROOT"
        export AMSS_CACHE_DIR="$CACHE_DIR"
        export AMSS_NCKU_PREPARE_ONLY=1
        ./run.sh --twop-cache
        unset AMSS_NCKU_PREPARE_ONLY
        BASE_DIR="$PREP_ROOT/GW250118/AMSS_NCKU_output"
    fi

    run_dir="$OUT_DIR/run-$name"
    mkdir -p "$run_dir/binary_output"
    cp "$build_dir/ABE" "$run_dir/ABE"
    cp "$BASE_DIR/Ansorg.psid" "$run_dir/Ansorg.psid"
    sed -E \
        "s/^(ABE::TotalTime[[:space:]]*=[[:space:]]*).*/\\1$EVOLVE_TIME/" \
        "$BASE_DIR/input.par" > "$run_dir/input.par"

    echo "=== perf stat: $name ==="
    perf stat -d -d -o "$run_dir/perf-stat.txt" -- \
        bash -c 'cd "$1" && exec ./ABE < /dev/null' _ "$run_dir" \
        2>&1 | tee "$run_dir/run.log"

    set +e
    ./check.sh "$run_dir/binary_output" | tee "$run_dir/check.txt"
    set -e
done

printf 'candidate\tarch_flags\tevolve_seconds\ttotal_seconds\taverage_cpus\tipc\tbranch_miss_pct\tL1D_miss_pct\tLLC_load_miss_pct\tdTLB_miss_pct\tbitwise_vs_o3\tcourse_check\n' \
    > "$OUT_DIR/results.tsv"

reference="$OUT_DIR/run-o3/binary_output"
for index in "${!names[@]}"; do
    name="${names[$index]}"
    arch="${arch_flags[$index]}"
    run_dir="$OUT_DIR/run-$name"
    stat_file="$run_dir/perf-stat.txt"

    bitwise=yes
    if [[ "$name" != o3 ]]; then
        for output in bssn_ADMQs.dat bssn_BH.dat bssn_constraint.dat bssn_psi4.dat; do
            cmp -s <(tail -n +3 "$reference/$output") \
                <(tail -n +3 "$run_dir/binary_output/$output") || bitwise=no
        done
    fi

    evolve="$(awk '/Total Evolve Time:/ {value=$4} END {print value}' "$run_dir/run.log")"
    total="$(awk '/Total Running Time:/ {value=$4} END {print value}' "$run_dir/run.log")"
    task_ms="$(awk '/task-clock/ {gsub(/,/, "", $1); print $1}' "$stat_file")"
    elapsed="$(awk '/seconds time elapsed/ {gsub(/,/, "", $1); print $1}' "$stat_file")"
    average="$(awk -v task="$task_ms" -v elapsed="$elapsed" 'BEGIN {printf "%.3f", task / 1000 / elapsed}')"
    ipc="$(awk '/instructions:u/ {print $4}' "$stat_file")"
    branch="$(awk '/branch-misses:u/ {gsub(/%/, "", $4); print $4}' "$stat_file")"
    l1d="$(awk '/L1-dcache-load-misses:u/ {gsub(/%/, "", $4); print $4}' "$stat_file")"
    llc="$(awk '/LLC-load-misses:u/ {gsub(/%/, "", $4); print $4}' "$stat_file")"
    dtlb="$(awk '/dTLB-load-misses:u/ {gsub(/%/, "", $4); print $4}' "$stat_file")"
    course="$(awk '/^FINAL:/ {value=$2} END {print value}' "$run_dir/check.txt")"

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$name" "${arch:-none}" "$evolve" "$total" "$average" "$ipc" \
        "$branch" "$l1d" "$llc" "$dtlb" "$bitwise" "$course" \
        >> "$OUT_DIR/results.tsv"
done

echo "=== Summary ==="
column -t -s $'\t' "$OUT_DIR/results.tsv" || cat "$OUT_DIR/results.tsv"
echo "Artifacts: $OUT_DIR"
