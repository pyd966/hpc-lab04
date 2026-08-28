#!/bin/bash
#HPC --partition=lab4
#HPC --cpu=60
#HPC --mem=100Gi
#HPC --time=30m
#HPC --name=abe-rhs-first-connection
#HPC --output=profile/hpc_%x_%j.log
set -euo pipefail

ROOT_DIR="${AMSS_ROOT_DIR:-$PWD}"
[[ -f "$ROOT_DIR/CMakeLists.txt" ]] || exit 2
cd "$ROOT_DIR"

RUN_ID="${HPC_JOB_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$BASHPID}"
OUT_DIR="$ROOT_DIR/profile/abe-rhs-first-connection-$RUN_ID"
PREP_ROOT="$OUT_DIR/prepare"
CACHE_DIR="$ROOT_DIR/profile/twopuncture-cache"
mkdir -p "$OUT_DIR" "$PREP_ROOT" "$CACHE_DIR"
exec > >(tee "$OUT_DIR/job.log") 2>&1

OMP_THREADS="${ABE_FIRST_CONNECTION_OMP_THREADS:-30}"
STATIC_TARGET="${ABE_FIRST_CONNECTION_STATIC_TARGET:-24}"
MOVING_TARGET="${ABE_FIRST_CONNECTION_MOVING_TARGET:-30}"
EVOLVE_TIME="${ABE_FIRST_CONNECTION_TIME:-4.0}"
RUN_ORDER="${ABE_FIRST_CONNECTION_RUN_ORDER:-off on on off}"
MODE="${ABE_FIRST_CONNECTION_MODE:-three}"
case "$MODE" in
    three) FUSION_OPTION=AMSS_ENABLE_RHS_FIRST_CONNECTION_FUSION ;;
    pair) FUSION_OPTION=AMSS_ENABLE_RHS_FIRST_CONNECTION_PAIR_FUSION ;;
    *) echo "invalid mode: $MODE" >&2; exit 2 ;;
esac

echo "git revision: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "OpenMP threads: $OMP_THREADS"
echo "static blocks/threads: $STATIC_TARGET"
echo "moving blocks/threads: $MOVING_TARGET"
echo "evolution interval: t=0..$EVOLVE_TIME"
echo "run order: $RUN_ORDER"
echo "fusion mode: $MODE"

"$ROOT_DIR/scripts/collect_cpu_info.sh" > "$OUT_DIR/cpu-info.txt"
gfortran --version > "$OUT_DIR/compiler.txt"

export JOBS="$(nproc)"
export OMP_NUM_THREADS="$OMP_THREADS" OMP_PLACES=cores OMP_PROC_BIND=close OMP_SCHEDULE=dynamic,1
export AMSS_OMP_STATIC_BLOCK_TARGET="$STATIC_TARGET" AMSS_OMP_MOVING_BLOCK_TARGET="$MOVING_TARGET"
export AMSS_OMP_STATIC_THREADS="$STATIC_TARGET" AMSS_OMP_MOVING_THREADS="$MOVING_TARGET" AMSS_OMP_ONLY_RUN=1

for value in OFF ON; do
    name=off; [[ "$value" == ON ]] && name=on
    export AMSS_BUILD_DIR="$OUT_DIR/build-$name"
    echo "=== Build $name: $FUSION_OPTION=$value ==="
    ./compile.sh \
        -DAMSS_ENABLE_GPU=OFF \
        -DAMSS_ENABLE_OPENMP=ON \
        -DAMSS_ENABLE_OMP_ONLY=ON \
        -DAMSS_ENABLE_FDERIVS_SIMD=ON \
        -DAMSS_ENABLE_OMP_DIRECT_SYNC=ON \
        -DAMSS_ENABLE_OMP_DIRECT_AMR_TRANSFER=OFF \
        -DAMSS_ENABLE_OMP_DIRECT_AMR_SPLIT=OFF \
        -DAMSS_ENABLE_PROLONG3_PAIRWISE=ON \
        -DAMSS_ENABLE_PROLONG3_SIMD=OFF \
        -DAMSS_ENABLE_RHS_METRIC_FUSION=ON \
        -DAMSS_ENABLE_RHS_GAMMA_FUSION=ON \
        -DAMSS_ENABLE_RHS_CONNECTION_FUSION=OFF \
        -DAMSS_ENABLE_RHS_AIJ_FUSION=OFF \
        -DAMSS_ENABLE_RHS_CHI_RICCI_FUSION=OFF \
        -DAMSS_ENABLE_RHS_FIRST_CONNECTION_FUSION=OFF \
        -DAMSS_ENABLE_RHS_FIRST_CONNECTION_PAIR_FUSION=OFF \
        -D"$FUSION_OPTION"="$value" \
        -DAMSS_ENABLE_TWOPUNCTURE_OPENMP=ON \
        -DAMSS_OPT='-O3 -g -fno-omit-frame-pointer -fopt-info-vec-optimized' \
        -DAMSS_TWOPUNCTURE_OPT=-O3 \
        -DAMSS_TWOPUNCTURE_ARCH_FLAGS=-march=native \
        2>&1 | tee "$OUT_DIR/build-$name.log"
    objdump -d --disassemble=f_compute_rhs_bssn_ "$OUT_DIR/build-$name/ABE" > "$OUT_DIR/compute-rhs-$name.asm" || true
done

export AMSS_BUILD_DIR="$OUT_DIR/build-off"
export AMSS_OUTPUT_ROOT="$PREP_ROOT" AMSS_CACHE_DIR="$CACHE_DIR" AMSS_NCKU_PREPARE_ONLY=1
./run.sh --twop-cache
unset AMSS_NCKU_PREPARE_ONLY

BASE_DIR="$PREP_ROOT/GW250118/AMSS_NCKU_output"
printf 'sequence\tcandidate\tevolve_seconds\ttotal_seconds\taverage_cpus\tipc\tbranch_miss_pct\tL1D_miss_pct\tLLC_load_miss_pct\tdTLB_miss_pct\tbitwise_vs_off\tcourse_check\n' > "$OUT_DIR/results.tsv"
reference=""
sequence=0
for name in $RUN_ORDER; do
    case "$name" in off|on) ;; *) echo "invalid candidate: $name" >&2; exit 2 ;; esac
    sequence=$((sequence + 1))
    run_dir="$OUT_DIR/run-$(printf '%02d' "$sequence")-$name"
    mkdir -p "$run_dir/binary_output"
    cp "$OUT_DIR/build-$name/ABE" "$run_dir/ABE"
    cp "$BASE_DIR/Ansorg.psid" "$run_dir/Ansorg.psid"
    sed -E "s/^(ABE::TotalTime[[:space:]]*=[[:space:]]*).*/\\1$EVOLVE_TIME/" \
        "$BASE_DIR/input.par" > "$run_dir/input.par"

    echo "=== perf stat run $sequence: $name ==="
    perf stat -d -d -o "$run_dir/perf-stat.txt" -- \
        bash -c 'cd "$1" && exec ./ABE < /dev/null' _ "$run_dir" \
        2>&1 | tee "$run_dir/run.log"
    set +e
    ./check.sh "$run_dir/binary_output" | tee "$run_dir/check.txt"
    set -e

    [[ -z "$reference" && "$name" == off ]] && reference="$run_dir/binary_output"
    bitwise=yes
    if [[ "$run_dir/binary_output" != "$reference" ]]; then
        for output in bssn_ADMQs.dat bssn_BH.dat bssn_constraint.dat bssn_psi4.dat; do
            cmp -s <(tail -n +3 "$reference/$output") <(tail -n +3 "$run_dir/binary_output/$output") || bitwise=no
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
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$sequence" "$name" "$evolve" "$total" "$average" "$ipc" "$branch" "$l1d" "$llc" "$dtlb" "$bitwise" "$course" >> "$OUT_DIR/results.tsv"
done

printf 'candidate\truns\tmean_evolve_seconds\tmin_evolve_seconds\tmax_evolve_seconds\tmean_average_cpus\n' > "$OUT_DIR/summary.tsv"
for name in off on; do
    awk -F '\t' -v candidate="$name" '
        NR > 1 && $2 == candidate {
            count++; sum += $3; cpus += $5
            if (count == 1 || $3 < min) min = $3
            if (count == 1 || $3 > max) max = $3
        }
        END { printf "%s\t%d\t%.6f\t%.6f\t%.6f\t%.3f\n", candidate, count, sum / count, min, max, cpus / count }
    ' "$OUT_DIR/results.tsv" >> "$OUT_DIR/summary.tsv"
done

printf '=== Per-run results ===\n'
column -t -s $'\t' "$OUT_DIR/results.tsv" || cat "$OUT_DIR/results.tsv"
printf '=== Summary ===\n'
column -t -s $'\t' "$OUT_DIR/summary.tsv" || cat "$OUT_DIR/summary.tsv"
echo "Artifacts: $OUT_DIR"
