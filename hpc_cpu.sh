#!/bin/bash
#HPC --partition=lab4
#HPC --cpu=60
#HPC --mem=100Gi
#HPC --time=30m
#HPC --name=amss-cpu
#HPC --output=profile/hpc_%x_%j.log
#
# Submit from the repository root. The #HPC directives above are parsed when
# the script is submitted without positional arguments. Select a mode with:
#   hpc submit ./hpc_cpu.sh
#   AMSS_JOB_MODE=stat hpc submit ./hpc_cpu.sh
#   AMSS_JOB_MODE=record hpc submit ./hpc_cpu.sh
#   AMSS_JOB_MODE=topology hpc submit ./hpc_cpu.sh
set -euo pipefail

ROOT_DIR="${AMSS_ROOT_DIR:-$PWD}"
if [[ ! -f "$ROOT_DIR/CMakeLists.txt" ]]; then
    echo "submit this script from the repository root" >&2
    exit 2
fi
ROOT_DIR="$(cd -- "$ROOT_DIR" && pwd)"
MODE="${1:-${AMSS_JOB_MODE:-baseline}}"

case "$MODE" in
    baseline|stat|record|topology) ;;
    *)
        echo "AMSS_JOB_MODE must be one of: baseline, stat, record, topology" >&2
        exit 2
        ;;
esac

cd "$ROOT_DIR"
mkdir -p profile profile/runs

RUN_ID="${HPC_JOB_ID:-${SLURM_JOB_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$$}}"
PROFILE_DIR="$ROOT_DIR/profile/$MODE-$RUN_ID"
RUN_ROOT="$ROOT_DIR/profile/runs/$MODE-$RUN_ID"
mkdir -p "$PROFILE_DIR" "$RUN_ROOT"

exec > >(tee "$PROFILE_DIR/job.log") 2>&1

echo "mode: $MODE"
echo "run id: $RUN_ID"
echo "repository: $ROOT_DIR"
echo "profile directory: $PROFILE_DIR"
echo "simulation output root: $RUN_ROOT"
echo "git revision: $(git rev-parse HEAD 2>/dev/null || echo unknown)"

"$ROOT_DIR/scripts/collect_cpu_info.sh" > "$PROFILE_DIR/cpu-info.txt"
cat "$PROFILE_DIR/cpu-info.txt"

if [[ "$MODE" == topology ]]; then
    exit 0
fi

# Count unique physical cores inside the job's CPU affinity mask. This gives
# 30 on the current SMT-enabled lab4 node and 60 on the physical-core-only
# evaluation allocation, without hard-coding either topology.
physical_cores="$($ROOT_DIR/scripts/count_available_physical_cores.sh)"
export AMSS_MPIEXEC="${AMSS_JOB_MPIEXEC:-mpiexec --allow-run-as-root}"
omp_threads="${OMP_NUM_THREADS:-$physical_cores}"
if [[ ! "$omp_threads" =~ ^[1-9][0-9]*$ ]]; then
    echo "OMP_NUM_THREADS must be one positive integer: $omp_threads" >&2
    exit 2
fi
export OMP_NUM_THREADS="$omp_threads"
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export OMP_PLACES="${OMP_PLACES:-cores}"

# Keep more block tasks than workers so a worker finishing an inexpensive block
# can take another one.  The tested 30-core configuration is 60 blocks with
# 24 static-level workers and 30 moving-level workers.  Keep that validated
# target on the 60-core evaluation node too; a new 120-block decomposition
# would change numerical block boundaries without having been validated.
static_threads=$((omp_threads * 4 / 5))
static_threads=$((static_threads > 0 ? static_threads : 1))
block_target=$((omp_threads < 60 ? 60 : omp_threads))
export AMSS_OMP_BLOCK_SCHEDULE="${AMSS_OMP_BLOCK_SCHEDULE:-dynamic,1}"
export OMP_SCHEDULE="${OMP_SCHEDULE:-$AMSS_OMP_BLOCK_SCHEDULE}"
export AMSS_OMP_STATIC_BLOCK_TARGET="${AMSS_OMP_STATIC_BLOCK_TARGET:-$block_target}"
export AMSS_OMP_MOVING_BLOCK_TARGET="${AMSS_OMP_MOVING_BLOCK_TARGET:-$block_target}"
export AMSS_OMP_STATIC_THREADS="${AMSS_OMP_STATIC_THREADS:-$static_threads}"
export AMSS_OMP_MOVING_THREADS="${AMSS_OMP_MOVING_THREADS:-$omp_threads}"
export AMSS_OMP_ONLY_RUN=1
export AMSS_OUTPUT_ROOT="$RUN_ROOT"
export AMSS_CACHE_DIR="$ROOT_DIR/profile/twopuncture-cache"
export JOBS="$(nproc)"
unset AMSS_BUILD_DIR
if [[ "$MODE" == record ]]; then
    export AMSS_BUILD_DIR="$ROOT_DIR/build-cpu-profile"
    BUILD_OPT='-O3 -g -fno-omit-frame-pointer'
else
    export AMSS_BUILD_DIR="$ROOT_DIR/build-cpu-baseline"
    BUILD_OPT='-O3'
fi

echo "OpenMP policy: schedule=$OMP_SCHEDULE, total=$OMP_NUM_THREADS, static blocks/threads=" \
     "$AMSS_OMP_STATIC_BLOCK_TARGET/$AMSS_OMP_STATIC_THREADS, " \
     "moving blocks/threads=" \
     "$AMSS_OMP_MOVING_BLOCK_TARGET/$AMSS_OMP_MOVING_THREADS"

echo "=== Build ==="
./compile.sh \
    -DAMSS_ENABLE_GPU=OFF \
    -DAMSS_ENABLE_OPENMP=ON \
    -DAMSS_ENABLE_OMP_ONLY=ON \
    -DAMSS_ENABLE_TWOPUNCTURE_OPENMP=ON \
    -DAMSS_OPT="$BUILD_OPT" \
    -DAMSS_ARCH_FLAGS= \
    -DAMSS_TWOPUNCTURE_OPT=-O3 \
    -DAMSS_TWOPUNCTURE_ARCH_FLAGS=-march=native

echo "=== Run ==="
case "$MODE" in
    baseline)
        start_epoch="$(date +%s)"
        ./run.sh
        end_epoch="$(date +%s)"
        echo "wall_seconds=$((end_epoch - start_epoch))" | tee "$PROFILE_DIR/time.txt"
        ;;
    stat)
        perf stat -d -d -o "$PROFILE_DIR/perf-stat.txt" -- ./run.sh
        ;;
    record)
        perf record -m 1 -F 49 --call-graph fp \
            -o "$PROFILE_DIR/perf.data" -- ./run.sh
        perf report --stdio --no-children --sort comm,dso,symbol \
            --percent-limit 0.5 -i "$PROFILE_DIR/perf.data" \
            > "$PROFILE_DIR/perf-report-flat.txt"
        perf report --stdio --children --sort comm,dso,symbol \
            --percent-limit 0.5 -i "$PROFILE_DIR/perf.data" \
            > "$PROFILE_DIR/perf-report-children.txt"
        ;;
esac

echo "=== Correctness ==="
./check.sh "$RUN_ROOT/GW250118/AMSS_NCKU_output" \
    | tee "$PROFILE_DIR/check.txt"

echo "=== Key timing ==="
grep -R -I -E 'This Program Cost|Elapsed \(wall clock\)|seconds time elapsed|wall_seconds=' \
    "$PROFILE_DIR" "$RUN_ROOT/GW250118/AMSS_NCKU_output" \
    2>/dev/null || true

echo "Artifacts: $PROFILE_DIR"
echo "Results: $RUN_ROOT/GW250118/AMSS_NCKU_output"
