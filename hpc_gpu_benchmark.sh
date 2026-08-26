#!/bin/bash
#HPC --partition=lab4g10
#HPC --cpu=16
#HPC --gpu=1
#HPC --mem=24Gi
#HPC --time=30m
#HPC --name=amss-gpu-time
#HPC --output=profile/hpc_%x_%j.log
#HPC --export=NONE
#
# Repeat the official end-to-end workload without the TwoPuncture cache:
#   hpc submit ./hpc_gpu_benchmark.sh
# Override settings with, for example:
#   hpc submit -e AMSS_BENCHMARK_RUNS=3,AMSS_BENCHMARK_TIME=5 ./hpc_gpu_benchmark.sh
set -euo pipefail

ROOT_DIR="${AMSS_ROOT_DIR:-$PWD}"
if [[ ! -f "$ROOT_DIR/CMakeLists.txt" ]]; then
    echo "submit this script from the repository root" >&2
    exit 2
fi

cd "$ROOT_DIR"
export AMSS_EXECUTION_MODE=gpu
mkdir -p profile

RUN_ID="${HPC_JOB_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$BASHPID}"
ARTIFACT_DIR="$ROOT_DIR/profile/gpu-benchmark-$RUN_ID"
WORK_ROOT="$ARTIFACT_DIR/work"
RUNS="${AMSS_BENCHMARK_RUNS:-3}"
BENCHMARK_TIME="${AMSS_BENCHMARK_TIME:-}"
if ! [[ "$RUNS" =~ ^[1-9][0-9]*$ ]]; then
    echo "AMSS_BENCHMARK_RUNS must be a positive integer" >&2
    exit 2
fi
mkdir -p "$ARTIFACT_DIR" "$WORK_ROOT"
exec > >(tee "$ARTIFACT_DIR/job.log") 2>&1

export JOBS="$(nproc)"
export AMSS_BUILD_DIR="${AMSS_GPU_BUILD_DIR:-$ROOT_DIR/build-gpu-baseline}"
export AMSS_OUTPUT_ROOT="$WORK_ROOT"
export AMSS_CACHE_DIR="$ROOT_DIR/profile/twopuncture-cache"
export AMSS_MPIEXEC="${AMSS_JOB_MPIEXEC:-mpiexec --allow-run-as-root --bind-to core}"
export OMP_PLACES="${OMP_PLACES:-cores}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
unset AMSS_NCKU_TWOP_CACHE
if [[ -n "$BENCHMARK_TIME" ]]; then
    export AMSS_NCKU_FINAL_TIME="$BENCHMARK_TIME"
else
    unset AMSS_NCKU_FINAL_TIME
fi

echo "git revision: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "benchmark repetitions: $RUNS"
echo "evolution end time: ${BENCHMARK_TIME:-official input}"
"$ROOT_DIR/scripts/collect_gpu_info.sh" > "$ARTIFACT_DIR/system-info.txt"

echo "=== Build outside measured region ==="
./compile.sh \
    -DAMSS_ENABLE_GPU=ON \
    -DCMAKE_CUDA_ARCHITECTURES=80 \
    -DAMSS_ENABLE_OPENMP=OFF \
    -DAMSS_ENABLE_OMP_ONLY=OFF \
    -DAMSS_ENABLE_TWOPUNCTURE_OPENMP=ON \
    -DAMSS_OPT=-O3 \
    2>&1 | tee "$ARTIFACT_DIR/build.log"

printf 'run\tprogram_seconds\touter_wall_seconds\tcheck\n' \
    > "$ARTIFACT_DIR/timings.tsv"
TIMEFORMAT=$'outer_wall_seconds=%R\nuser_cpu_seconds=%U\nsystem_cpu_seconds=%S'

for ((run = 1; run <= RUNS; ++run)); do
    RUN_LOG="$ARTIFACT_DIR/run-$run.log"
    CHECK_LOG="$ARTIFACT_DIR/check-$run.txt"
    echo "=== Timed run $run/$RUNS ==="
    { time ./run.sh; } 2>&1 | tee "$RUN_LOG"

    ./check.sh "$WORK_ROOT/GW250118/AMSS_NCKU_output" | tee "$CHECK_LOG"
    program_seconds="$(awk '/This Program Cost =/{value=$(NF-1)} END{print value}' "$RUN_LOG")"
    outer_seconds="$(awk -F= '/^outer_wall_seconds=/{value=$2} END{print value}' "$RUN_LOG")"
    check_status="$(awk '/^FINAL:/{value=$2} END{print value}' "$CHECK_LOG")"
    printf '%s\t%s\t%s\t%s\n' \
        "$run" "$program_seconds" "$outer_seconds" "$check_status" \
        >> "$ARTIFACT_DIR/timings.tsv"
done

awk -F '\t' 'NR > 1 {sum += $2; sumsq += $2*$2; n += 1}
    END {
      if (n == 0) exit 1;
      mean = sum / n;
      variance = (n > 1) ? (sumsq - sum*sum/n)/(n-1) : 0;
      printf "runs=%d\nmean_program_seconds=%.6f\nsample_stddev_seconds=%.6f\n", n, mean, sqrt(variance)
    }' "$ARTIFACT_DIR/timings.tsv" | tee "$ARTIFACT_DIR/summary.txt"

echo "artifacts: $ARTIFACT_DIR"
