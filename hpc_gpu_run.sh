#!/bin/bash
#HPC --partition=lab4g10
#HPC --cpu=16
#HPC --gpu=1
#HPC --mem=24Gi
#HPC --time=30m
#HPC --name=amss-gpu-run
#HPC --output=profile/hpc_%x_%j.log
#HPC --export=NONE
#
# Build and run the fixed full GPU workload once. Submit from the repository:
#   hpc submit ./hpc_gpu_run.sh
set -euo pipefail

# hpc executes a submitted copy from /tmp while preserving the submission cwd.
ROOT_DIR="${AMSS_ROOT_DIR:-$PWD}"
if [[ ! -f "$ROOT_DIR/CMakeLists.txt" ]]; then
    echo "submit this script from the repository root" >&2
    exit 2
fi

cd "$ROOT_DIR"
mkdir -p profile

RUN_ID="${HPC_JOB_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$BASHPID}"
ARTIFACT_DIR="$ROOT_DIR/profile/gpu-run-$RUN_ID"
mkdir -p "$ARTIFACT_DIR"
exec > >(tee "$ARTIFACT_DIR/job.log") 2>&1

export JOBS="$(nproc)"
export AMSS_BUILD_DIR="${AMSS_GPU_BUILD_DIR:-$ROOT_DIR/build-gpu-baseline}"
export AMSS_OUTPUT_ROOT="${AMSS_GPU_OUTPUT_ROOT:-$ROOT_DIR}"
export AMSS_CACHE_DIR="${AMSS_GPU_CACHE_DIR:-$ROOT_DIR/profile/twopuncture-cache}"
export AMSS_MPIEXEC="${AMSS_JOB_MPIEXEC:-mpiexec --allow-run-as-root --bind-to core}"
export OMP_PLACES="${OMP_PLACES:-cores}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
unset AMSS_NCKU_TWOP_CACHE

echo "git revision: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "artifact directory: $ARTIFACT_DIR"
"$ROOT_DIR/scripts/collect_gpu_info.sh" | tee "$ARTIFACT_DIR/system-info.txt"

echo "=== Build ==="
./compile.sh \
    -DAMSS_ENABLE_GPU=ON \
    -DCMAKE_CUDA_ARCHITECTURES=80 \
    -DAMSS_ENABLE_OPENMP=OFF \
    -DAMSS_ENABLE_TWOPUNCTURE_OPENMP=ON \
    -DAMSS_OPT=-O3 \
    2>&1 | tee "$ARTIFACT_DIR/build.log"

echo "=== Full run ==="
TIMEFORMAT=$'outer_wall_seconds=%R\nuser_cpu_seconds=%U\nsystem_cpu_seconds=%S'
{ time ./run.sh; } 2>&1 | tee "$ARTIFACT_DIR/run.log"

echo "=== Correctness ==="
./check.sh "$AMSS_OUTPUT_ROOT/GW250118/AMSS_NCKU_output" \
    | tee "$ARTIFACT_DIR/check.txt"

grep -E 'This Program Cost|outer_wall_seconds|user_cpu_seconds|system_cpu_seconds' \
    "$ARTIFACT_DIR/run.log" | tee "$ARTIFACT_DIR/timing-summary.txt"
echo "artifacts: $ARTIFACT_DIR"
