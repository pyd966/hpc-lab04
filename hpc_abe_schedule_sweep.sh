#!/bin/bash
#HPC --partition=lab4
#HPC --cpu=60
#HPC --mem=100Gi
#HPC --time=30m
#HPC --name=abe-schedule-sweep
#HPC --output=profile/hpc_%x_%j.log
#
# Compare ABE OpenMP block scheduling strategies on one node and one prepared
# input.  Each case is run sequentially so node-to-node variation cannot hide
# small scheduling effects.
set -euo pipefail

ROOT_DIR="${AMSS_ROOT_DIR:-$PWD}"
if [[ ! -f "$ROOT_DIR/CMakeLists.txt" ]]; then
    echo "submit this script from the repository root" >&2
    exit 2
fi

cd "$ROOT_DIR"
mkdir -p profile

RUN_ID="${HPC_JOB_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$BASHPID}"
SWEEP_DIR="$ROOT_DIR/profile/schedule-sweep-$RUN_ID"
NORMAL_BUILD="$SWEEP_DIR/build-normal"
PREP_ROOT="$SWEEP_DIR/prepare"
CACHE_DIR="$ROOT_DIR/profile/twopuncture-cache"
EVOLVE_TIME="${ABE_SWEEP_TIME:-4.0}"
OMP_THREADS="${ABE_SWEEP_THREADS:-30}"
STATIC_TARGET="${ABE_SWEEP_STATIC_TARGET:-$((OMP_THREADS * 4 / 5))}"
STATIC_THREADS="${ABE_SWEEP_STATIC_THREADS:-$((OMP_THREADS * 4 / 5))}"
MOVING_THREADS="${ABE_SWEEP_MOVING_THREADS:-$OMP_THREADS}"
ABE_OPT="${AMSS_ABE_PROFILE_OPT:--O3 -g -fno-omit-frame-pointer}"
mkdir -p "$SWEEP_DIR" "$PREP_ROOT" "$CACHE_DIR"
exec > >(tee "$SWEEP_DIR/job.log") 2>&1

export JOBS="$(nproc)"
export OMP_NUM_THREADS="$OMP_THREADS"
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
export AMSS_OMP_STATIC_THREADS="$STATIC_THREADS"
export AMSS_OMP_MOVING_THREADS="$MOVING_THREADS"

echo "sweep directory: $SWEEP_DIR"
echo "git revision: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "evolution interval: t=0..$EVOLVE_TIME"
echo "OpenMP threads: $OMP_THREADS"
echo "default static block target: $STATIC_TARGET"
echo "static/moving threads: $STATIC_THREADS/$MOVING_THREADS"
"$ROOT_DIR/scripts/collect_cpu_info.sh" > "$SWEEP_DIR/cpu-info.txt"

build_variant() {
    local build_dir="$1"
    export AMSS_BUILD_DIR="$build_dir"
    ./compile.sh \
        -DAMSS_ENABLE_GPU=OFF \
        -DAMSS_ENABLE_OPENMP=ON \
        -DAMSS_ENABLE_OMP_ONLY=ON \
        -DAMSS_ENABLE_OMP_DIAGNOSTICS=ON \
        -DAMSS_ENABLE_TWOPUNCTURE_OPENMP=ON \
        -DAMSS_OPT="$ABE_OPT" \
        -DAMSS_ARCH_FLAGS= \
        -DAMSS_TWOPUNCTURE_OPT=-O3 \
        -DAMSS_TWOPUNCTURE_ARCH_FLAGS=-march=native \
        -DAMSS_ENABLE_RHS_METRIC_FUSION=ON \
        -DAMSS_ENABLE_RHS_GAMMA_FUSION=ON \
        -DAMSS_ENABLE_RHS_SANITY_CHECK=OFF
}

echo "=== Build normal-team variant ==="
build_variant "$NORMAL_BUILD" 2>&1 | tee "$SWEEP_DIR/build-normal.log"

echo "=== Prepare fixed course input ==="
export AMSS_BUILD_DIR="$NORMAL_BUILD"
export AMSS_OUTPUT_ROOT="$PREP_ROOT"
export AMSS_CACHE_DIR="$CACHE_DIR"
export AMSS_NCKU_PREPARE_ONLY=1
export AMSS_MPIEXEC="mpiexec --allow-run-as-root --map-by core --bind-to core"
./run.sh --twop-cache
unset AMSS_NCKU_PREPARE_ONLY

BASE_DIR="$PREP_ROOT/GW250118/AMSS_NCKU_output"
if [[ ! -f "$BASE_DIR/input.par" || ! -f "$BASE_DIR/Ansorg.psid" ]]; then
    echo "prepared ABE input is incomplete: $BASE_DIR" >&2
    exit 1
fi

stage_run() {
    local target="$1"
    local executable="$2"
    mkdir -p "$target/binary_output"
    cp "$executable" "$target/ABE"
    cp "$BASE_DIR/Ansorg.psid" "$target/Ansorg.psid"
    sed -E \
        "s/^(ABE::TotalTime[[:space:]]*=[[:space:]]*).*/\\1$EVOLVE_TIME/" \
        "$BASE_DIR/input.par" > "$target/input.par"
}

run_case() {
    local name="$1"
    local executable="$2"
    local schedule="$3"
    local moving_target="$4"
    local static_target="$5"
    local run_dir="$SWEEP_DIR/$name"

    stage_run "$run_dir" "$executable"
    export OMP_SCHEDULE="$schedule"
    export AMSS_OMP_STATIC_BLOCK_TARGET="$static_target"
    export AMSS_OMP_MOVING_BLOCK_TARGET="$moving_target"

    echo "=== $name: schedule=$schedule, static-target=$static_target, moving-target=$moving_target ==="
    set +e
    (
        cd "$run_dir"
        perf stat -d -d -o perf-stat.txt -- ./ABE < /dev/null 2>&1 | tee run.log
    )
    local run_rc=${PIPESTATUS[0]}
    set -e
    if (( run_rc != 0 )); then
        echo "$name failed with status $run_rc" >&2
        return "$run_rc"
    fi

    # Keep one numerical reference only.  The remaining large output files are
    # compared immediately and removed to keep the shared home quota usable.
    if [[ "$name" != "normal-static-24" ]]; then
        local reference="$SWEEP_DIR/normal-static-24/binary_output"
        local result=identical
        for output in bssn_ADMQs.dat bssn_BH.dat bssn_constraint.dat bssn_psi4.dat; do
            if ! cmp -s <(tail -n +3 "$reference/$output") \
                       <(tail -n +3 "$run_dir/binary_output/$output"); then
                result=different
                break
            fi
        done
        printf '%s\t%s\n' "$name" "$result" >> "$SWEEP_DIR/numerical-summary.tsv"
        rm -rf "$run_dir/binary_output"
    fi
}

printf 'case\tresult\n' > "$SWEEP_DIR/numerical-summary.tsv"

# Scheduling policy at the current 30-block granularity.
run_case normal-static-24 "$NORMAL_BUILD/ABE" static 30 24
run_case normal-static1-24 "$NORMAL_BUILD/ABE" static,1 30 24
run_case normal-dynamic1-24 "$NORMAL_BUILD/ABE" dynamic,1 30 24
run_case normal-guided1-24 "$NORMAL_BUILD/ABE" guided,1 30 24

# More blocks expose additional work to the same 30 bound workers.
run_case normal-static-60 "$NORMAL_BUILD/ABE" static 60 60
run_case normal-dynamic1-60 "$NORMAL_BUILD/ABE" dynamic,1 60 60
run_case normal-guided1-60 "$NORMAL_BUILD/ABE" guided,1 60 60
run_case normal-dynamic1-90 "$NORMAL_BUILD/ABE" dynamic,1 90 90

{
    printf 'case\tevolve_seconds\ttotal_seconds\n'
    for run_dir in "$SWEEP_DIR"/normal-*; do
        [[ -f "$run_dir/run.log" ]] || continue
        evolve="$(sed -n 's/.*Total Evolve Time: \([^ ]*\).*/\1/p' "$run_dir/run.log" | tail -1)"
        total="$(sed -n 's/.*Total Running Time: \([^ ]*\).*/\1/p' "$run_dir/run.log" | tail -1)"
        printf '%s\t%s\t%s\n' "$(basename "$run_dir")" "$evolve" "$total"
    done
} | sort > "$SWEEP_DIR/timing-summary.tsv"

echo "=== Timing summary ==="
cat "$SWEEP_DIR/timing-summary.tsv"
echo "=== Numerical summary ==="
cat "$SWEEP_DIR/numerical-summary.tsv"
echo "sweep complete: $SWEEP_DIR"
