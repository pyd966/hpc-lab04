#!/bin/bash
#HPC --partition=lab4
#HPC --cpu=60
#HPC --mem=100Gi
#HPC --time=30m
#HPC --name=abe-omp-initial-constraint
#HPC --output=profile/hpc_%x_%j.log
set -euo pipefail

ROOT_DIR="${AMSS_ROOT_DIR:-$PWD}"
if [[ ! -f "$ROOT_DIR/CMakeLists.txt" ]]; then
    echo "submit this script from the repository root" >&2
    exit 2
fi
cd "$ROOT_DIR"

RUN_ID="${HPC_JOB_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$BASHPID}"
OUT_DIR="$ROOT_DIR/profile/abe-omp-initial-constraint-$RUN_ID"
PREP_ROOT="$OUT_DIR/prepare"
CACHE_DIR="$ROOT_DIR/profile/twopuncture-cache"
mkdir -p "$OUT_DIR" "$PREP_ROOT" "$CACHE_DIR"
exec > >(tee "$OUT_DIR/job.log") 2>&1

OMP_THREADS="${ABE_OMP_SWEEP_THREADS:-30}"
STATIC_TARGET="${ABE_OMP_SWEEP_STATIC_TARGET:-24}"
MOVING_TARGET="${ABE_OMP_SWEEP_MOVING_TARGET:-30}"
EVOLVE_TIME="${ABE_OMP_SWEEP_TIME:-4.0}"
RUN_ORDER="${ABE_OMP_SWEEP_RUN_ORDER:-initial-off initial-on initial-on initial-off}"

echo "git revision: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "OpenMP threads: $OMP_THREADS"
echo "static blocks/threads: $STATIC_TARGET"
echo "moving blocks/threads: $MOVING_TARGET"
echo "evolution interval: t=0..$EVOLVE_TIME"
echo "run order: $RUN_ORDER"

"$ROOT_DIR/scripts/collect_cpu_info.sh" > "$OUT_DIR/cpu-info.txt"
export JOBS="$(nproc)"
export OMP_NUM_THREADS="$OMP_THREADS"
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export AMSS_OMP_STATIC_BLOCK_TARGET="$STATIC_TARGET"
export AMSS_OMP_MOVING_BLOCK_TARGET="$MOVING_TARGET"
export AMSS_OMP_STATIC_THREADS="$STATIC_TARGET"
export AMSS_OMP_MOVING_THREADS="$MOVING_TARGET"
export AMSS_OMP_ONLY_RUN=1

names=(initial-off initial-on)
values=(OFF ON)
for index in "${!names[@]}"; do
    name="${names[$index]}"
    value="${values[$index]}"
    build_dir="$OUT_DIR/build-$name"
    export AMSS_BUILD_DIR="$build_dir"

    echo "=== Build $name: AMSS_ENABLE_OMP_INITIAL_CONSTRAINT_PARALLEL=$value ==="
    ./compile.sh \
        -DAMSS_ENABLE_GPU=OFF \
        -DAMSS_ENABLE_OPENMP=ON \
        -DAMSS_ENABLE_OMP_ONLY=ON \
        -DAMSS_ENABLE_OMP_DIAGNOSTICS=ON \
        -DAMSS_ENABLE_OMP_CONSTRAINT_PARALLEL=ON \
        -DAMSS_ENABLE_OMP_INITIAL_CONSTRAINT_PARALLEL="$value" \
        -DAMSS_ENABLE_TWOPUNCTURE_OPENMP=ON \
        -DAMSS_OPT='-O3 -g -fno-omit-frame-pointer' \
        -DAMSS_TWOPUNCTURE_OPT=-O3 \
        -DAMSS_TWOPUNCTURE_ARCH_FLAGS=-march=native \
        2>&1 | tee "$OUT_DIR/build-$name.log"
done

export AMSS_BUILD_DIR="$OUT_DIR/build-initial-off"
export AMSS_OUTPUT_ROOT="$PREP_ROOT"
export AMSS_CACHE_DIR="$CACHE_DIR"
export AMSS_NCKU_PREPARE_ONLY=1
./run.sh --twop-cache
unset AMSS_NCKU_PREPARE_ONLY

BASE_DIR="$PREP_ROOT/GW250118/AMSS_NCKU_output"
printf 'sequence\tcandidate\tbefore_evolve_seconds\tevolve_seconds\ttotal_seconds\tcompute_constraint_seconds\tinitial_interp_seconds\tconstraint_out_seconds\taverage_cpus\tbitwise_vs_initial_off\tcourse_check\n' > "$OUT_DIR/results.tsv"

reference=""
sequence=0
for name in $RUN_ORDER; do
    if [[ "$name" != initial-off && "$name" != initial-on ]]; then
        echo "invalid candidate in run order: $name" >&2
        exit 2
    fi
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

    if [[ -z "$reference" && "$name" == initial-off ]]; then
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
    before="$(awk '/Before Evolve, it takes/ {value=$5} END {print value}' "$run_dir/run.log")"
    evolve="$(awk '/Total Evolve Time:/ {value=$4} END {print value}' "$run_dir/run.log")"
    total="$(awk '/Total Running Time:/ {value=$4} END {print value}' "$run_dir/run.log")"
    compute="$(awk '/^OMP_DIAG_COMPUTE_CONSTRAINT/ {value=$5} END {print value}' "$run_dir/run.log")"
    interp="$(awk '/^OMP_DIAG_INITIAL_INTERP_CONSTRAINT/ {value=$5} END {print value}' "$run_dir/run.log")"
    constraint="$(awk '/^OMP_DIAG_CONSTRAINT / {value=$5} END {print value}' "$run_dir/run.log")"
    task_ms="$(awk '/task-clock/ {gsub(/,/, "", $1); print $1}' "$stat_file")"
    elapsed="$(awk '/seconds time elapsed/ {gsub(/,/, "", $1); print $1}' "$stat_file")"
    average="$(awk -v task="$task_ms" -v elapsed="$elapsed" 'BEGIN {printf "%.3f", task / 1000 / elapsed}')"
    course="$(awk '/^FINAL:/ {value=$2} END {print value}' "$run_dir/check.txt")"

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$sequence" "$name" "$before" "$evolve" "$total" "$compute" \
        "$interp" "$constraint" "$average" "$bitwise" "$course" \
        >> "$OUT_DIR/results.tsv"
done

printf 'candidate\truns\tmean_before_evolve_seconds\tmean_evolve_seconds\tmean_total_seconds\tmean_compute_constraint_seconds\tmean_initial_interp_seconds\tmean_average_cpus\n' > "$OUT_DIR/summary.tsv"
for name in "${names[@]}"; do
    awk -F '\t' -v candidate="$name" '
        NR > 1 && $2 == candidate {
            count++; before += $3; evolve += $4; total += $5;
            compute += $6; interp += $7; cpus += $9
        }
        END {
            printf "%s\t%d\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.3f\n",
                candidate, count, before / count, evolve / count, total / count,
                compute / count, interp / count, cpus / count
        }
    ' "$OUT_DIR/results.tsv" >> "$OUT_DIR/summary.tsv"
done

echo "=== Per-run results ==="
column -t -s $'\t' "$OUT_DIR/results.tsv" || cat "$OUT_DIR/results.tsv"
echo "=== Summary ==="
column -t -s $'\t' "$OUT_DIR/summary.tsv" || cat "$OUT_DIR/summary.tsv"
echo "Artifacts: $OUT_DIR"
