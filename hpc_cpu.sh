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

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
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

# lab4 currently exposes 60 logical CPUs, i.e. 30 physical cores with SMT
# siblings. The fixed baseline input launches 30 ranks, one rank per core.
export AMSS_MPIEXEC="${AMSS_JOB_MPIEXEC:-mpiexec --allow-run-as-root --map-by core --bind-to core --report-bindings}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export OMP_PLACES="${OMP_PLACES:-cores}"
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

echo "=== Build ==="
./compile.sh -DAMSS_ENABLE_GPU=OFF -DAMSS_OPT="$BUILD_OPT"

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
