#!/bin/bash
#HPC --partition=lab4
#HPC --cpu=60
#HPC --mem=100Gi
#HPC --time=30m
#HPC --name=abe-moving-block
#HPC --output=profile/hpc_%x_%j.log
set -euo pipefail

ROOT_DIR="${AMSS_ROOT_DIR:-$PWD}"
if [[ ! -f "$ROOT_DIR/CMakeLists.txt" ]]; then
    echo "submit this script from the repository root" >&2
    exit 2
fi
cd "$ROOT_DIR"

RUN_ID="${HPC_JOB_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$BASHPID}"
OUT_DIR="$ROOT_DIR/profile/abe-moving-block-$RUN_ID"
PREP_ROOT="$OUT_DIR/prepare"
CACHE_DIR="$ROOT_DIR/profile/twopuncture-cache"
mkdir -p "$OUT_DIR" "$PREP_ROOT" "$CACHE_DIR"
exec > >(tee "$OUT_DIR/job.log") 2>&1

EVOLVE_TIME="${ABE_GEOMETRY_TIME:-4.0}"
RUN_ORDER="${ABE_GEOMETRY_RUN_ORDER:-base move60 move60 base}"
BASE_OMP_SCHEDULE="${ABE_GEOMETRY_SCHEDULE:-dynamic,1}"

echo "git revision: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "evolution interval: t=0..$EVOLVE_TIME"
echo "run order: $RUN_ORDER"
echo "schedule: $BASE_OMP_SCHEDULE"
"$ROOT_DIR/scripts/collect_cpu_info.sh" > "$OUT_DIR/cpu-info.txt"
gfortran --version > "$OUT_DIR/compiler.txt"

export JOBS="$(nproc)"
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_SCHEDULE="$BASE_OMP_SCHEDULE"
export AMSS_OMP_ONLY_RUN=1

names=(base move60)
threads=(30 30)
static_targets=(24 24)
moving_targets=(30 60)
static_threads=(24 24)
moving_threads=(30 30)

for index in "${!names[@]}"; do
    name="${names[$index]}"
    build_dir="$OUT_DIR/build-$name"
    export AMSS_BUILD_DIR="$build_dir"
    echo "=== Build $name: OMP=${threads[$index]}, static=${static_targets[$index]}/${static_threads[$index]}, moving=${moving_targets[$index]}/${moving_threads[$index]} ==="
    ./compile.sh \
        -DAMSS_ENABLE_GPU=OFF \
        -DAMSS_ENABLE_OPENMP=ON \
        -DAMSS_ENABLE_OMP_ONLY=ON \
        -DAMSS_ENABLE_FDERIVS_SIMD=ON \
        -DAMSS_ENABLE_OMP_DIRECT_SYNC=ON \
        -DAMSS_ENABLE_OMP_DIRECT_AMR_TRANSFER=OFF \
        -DAMSS_ENABLE_OMP_DIRECT_AMR_SPLIT=OFF \
        -DAMSS_ENABLE_TWOPUNCTURE_OPENMP=ON \
        -DAMSS_OPT='-O3 -g -fno-omit-frame-pointer -fopt-info-vec-optimized' \
        -DAMSS_TWOPUNCTURE_OPT=-O3 \
        -DAMSS_TWOPUNCTURE_ARCH_FLAGS=-march=native \
        2>&1 | tee "$OUT_DIR/build-$name.log"
done

export OMP_NUM_THREADS=30
export AMSS_OMP_STATIC_BLOCK_TARGET=24
export AMSS_OMP_MOVING_BLOCK_TARGET=30
export AMSS_OMP_STATIC_THREADS=24
export AMSS_OMP_MOVING_THREADS=30
export AMSS_BUILD_DIR="$OUT_DIR/build-base"
export AMSS_OUTPUT_ROOT="$PREP_ROOT"
export AMSS_CACHE_DIR="$CACHE_DIR"
export AMSS_NCKU_PREPARE_ONLY=1
./run.sh --twop-cache
unset AMSS_NCKU_PREPARE_ONLY

BASE_DIR="$PREP_ROOT/GW250118/AMSS_NCKU_output"
printf 'sequence\tcandidate\tomps\tstatic_target\tmoving_target\tstatic_threads\tmoving_threads\tevolve_seconds\ttotal_seconds\taverage_cpus\tipc\tbranch_miss_pct\tL1D_miss_pct\tLLC_load_miss_pct\tdTLB_miss_pct\tbitwise_vs_base\tcourse_check\n' > "$OUT_DIR/results.tsv"

reference=""
sequence=0
for name in $RUN_ORDER; do
    case "$name" in
        base) index=0 ;;
        move60) index=1 ;;
        *) echo "invalid candidate in run order: $name" >&2; exit 2 ;;
    esac
    sequence=$((sequence + 1))
    OMP_NUM_THREADS="${threads[$index]}"
    AMSS_OMP_STATIC_BLOCK_TARGET="${static_targets[$index]}"
    AMSS_OMP_MOVING_BLOCK_TARGET="${moving_targets[$index]}"
    AMSS_OMP_STATIC_THREADS="${static_threads[$index]}"
    AMSS_OMP_MOVING_THREADS="${moving_threads[$index]}"
    export OMP_NUM_THREADS AMSS_OMP_STATIC_BLOCK_TARGET AMSS_OMP_MOVING_BLOCK_TARGET
    export AMSS_OMP_STATIC_THREADS AMSS_OMP_MOVING_THREADS

    run_dir="$OUT_DIR/run-$(printf '%02d' "$sequence")-$name"
    mkdir -p "$run_dir/binary_output"
    cp "$OUT_DIR/build-$name/ABE" "$run_dir/ABE"
    cp "$BASE_DIR/Ansorg.psid" "$run_dir/Ansorg.psid"
    sed -E "s/^(ABE::TotalTime[[:space:]]*=[[:space:]]*).*/\\1$EVOLVE_TIME/" \
        "$BASE_DIR/input.par" > "$run_dir/input.par"

    echo "=== perf stat run $sequence: $name (OMP=$OMP_NUM_THREADS, static=$AMSS_OMP_STATIC_THREADS, moving=$AMSS_OMP_MOVING_THREADS) ==="
    perf stat -d -d -o "$run_dir/perf-stat.txt" -- \
        bash -c 'cd "$1" && exec ./ABE < /dev/null' _ "$run_dir" \
        2>&1 | tee "$run_dir/run.log"

    set +e
    ./check.sh "$run_dir/binary_output" | tee "$run_dir/check.txt"
    set -e

    if [[ "$reference" == "" && "$name" == "base" ]]; then
        reference="$run_dir/binary_output"
    fi
    bitwise=yes
    if [[ "$run_dir/binary_output" != "$reference" ]]; then
        for output in bssn_ADMQs.dat bssn_BH.dat bssn_constraint.dat bssn_psi4.dat; do
            cmp -s <(tail -n +3 "$reference/$output") \
                <(tail -n +3 "$run_dir/binary_output/$output") || bitwise=no
        done
    fi

    stat_file="$run_dir/perf-stat.txt"
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

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$sequence" "$name" "$OMP_NUM_THREADS" "$AMSS_OMP_STATIC_BLOCK_TARGET" "$AMSS_OMP_MOVING_BLOCK_TARGET" \
        "$AMSS_OMP_STATIC_THREADS" "$AMSS_OMP_MOVING_THREADS" "$evolve" "$total" "$average" "$ipc" \
        "$branch" "$l1d" "$llc" "$dtlb" "$bitwise" "$course" >> "$OUT_DIR/results.tsv"
    if [[ "$run_dir/binary_output" != "$reference" ]]; then
        rm -rf "$run_dir/binary_output"
    fi
    rm -f "$run_dir/ABE" "$run_dir/Ansorg.psid" "$run_dir/input.par"
done

printf 'candidate\truns\tmean_evolve_seconds\tmin_evolve_seconds\tmax_evolve_seconds\tmean_total_seconds\tmean_average_cpus\n' > "$OUT_DIR/summary.tsv"
for name in "${names[@]}"; do
    awk -F '\t' -v candidate="$name" '
        NR > 1 && $2 == candidate {
            count++; evolve += $8; total += $9; cpus += $10
            if (count == 1 || $8 < min) min = $8
            if (count == 1 || $8 > max) max = $8
        }
        END {
            printf "%s\t%d\t%.6f\t%.6f\t%.6f\t%.6f\t%.3f\n",
                candidate, count, evolve / count, min, max, total / count, cpus / count
        }
    ' "$OUT_DIR/results.tsv" >> "$OUT_DIR/summary.tsv"
done

echo "=== Per-run results ==="
column -t -s $'\t' "$OUT_DIR/results.tsv" || cat "$OUT_DIR/results.tsv"
echo "=== Summary ==="
column -t -s $'\t' "$OUT_DIR/summary.tsv" || cat "$OUT_DIR/summary.tsv"
echo "Artifacts: $OUT_DIR"
