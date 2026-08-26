#!/bin/bash
#HPC --partition=lab4g10
#HPC --cpu=16
#HPC --gpu=1
#HPC --mem=24Gi
#HPC --time=30m
#HPC --name=amss-gpu-profile
#HPC --output=profile/hpc_%x_%j.log
#HPC --export=NONE
#
# Profile only ABEGPU after preparing the fixed initial data. Examples:
#   hpc submit -e AMSS_PROFILE_TOOL=vtune ./hpc_gpu_profile.sh
#   hpc submit -e AMSS_PROFILE_TOOL=nsys ./hpc_gpu_profile.sh
#   hpc submit -e AMSS_PROFILE_TOOL=ncu,NCU_KERNEL_REGEX=rhs_kernel ./hpc_gpu_profile.sh
set -euo pipefail

ROOT_DIR="${AMSS_ROOT_DIR:-$PWD}"
if [[ ! -f "$ROOT_DIR/CMakeLists.txt" ]]; then
    echo "submit this script from the repository root" >&2
    exit 2
fi

TOOL="${1:-${AMSS_PROFILE_TOOL:-nsys}}"
case "$TOOL" in
    vtune|nsys|ncu) ;;
    *)
        echo "AMSS_PROFILE_TOOL must be one of: vtune, nsys, ncu" >&2
        exit 2
        ;;
esac

find_vtune() {
    if command -v vtune >/dev/null 2>&1; then
        command -v vtune
        return
    fi
    local candidate
    for candidate in /opt/intel/oneapi/vtune/*/bin64/vtune; do
        if [[ -x "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return
        fi
    done
    return 1
}

cd "$ROOT_DIR"
mkdir -p profile

ARTIFACT_ROOT="${AMSS_PROFILE_ROOT:-$ROOT_DIR/profile}"
RUN_ID="${HPC_JOB_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$BASHPID}"
ARTIFACT_DIR="$ARTIFACT_ROOT/gpu-$TOOL-$RUN_ID"
PREP_ROOT="$ARTIFACT_DIR/prepare"
BUILD_DIR="${AMSS_GPU_PROFILE_BUILD_DIR:-$ROOT_DIR/build-gpu-profile}"
CACHE_DIR="$ROOT_DIR/profile/twopuncture-cache"
PROFILE_TIME="${AMSS_PROFILE_TIME:-4.0}"
mkdir -p "$ARTIFACT_ROOT" "$ARTIFACT_DIR" "$PREP_ROOT" "$CACHE_DIR"
exec > >(tee "$ARTIFACT_DIR/job.log") 2>&1

export JOBS="$(nproc)"
export AMSS_BUILD_DIR="$BUILD_DIR"
export AMSS_OUTPUT_ROOT="$PREP_ROOT"
export AMSS_CACHE_DIR="$CACHE_DIR"
export AMSS_MPIEXEC="${AMSS_JOB_MPIEXEC:-mpiexec --allow-run-as-root --bind-to core}"
export OMP_PLACES="${OMP_PLACES:-cores}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"

echo "profile tool: $TOOL"
echo "git revision: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "artifact directory: $ARTIFACT_DIR"
echo "profile evolution interval: t=0..$PROFILE_TIME"
"$ROOT_DIR/scripts/collect_gpu_info.sh" | tee "$ARTIFACT_DIR/system-info.txt"

echo "=== Build with source correlation ==="
./compile.sh \
    -DAMSS_ENABLE_GPU=ON \
    -DCMAKE_CUDA_ARCHITECTURES=80 \
    -DAMSS_ENABLE_OPENMP=OFF \
    -DAMSS_ENABLE_TWOPUNCTURE_OPENMP=ON \
    -DAMSS_OPT='-O3 -g -fno-omit-frame-pointer' \
    2>&1 | tee "$ARTIFACT_DIR/build.log"

echo "=== Prepare fixed input (outside measured region) ==="
export AMSS_NCKU_FINAL_TIME="$PROFILE_TIME"
export AMSS_NCKU_PREPARE_ONLY=1
./run.sh --twop-cache 2>&1 | tee "$ARTIFACT_DIR/prepare.log"
unset AMSS_NCKU_PREPARE_ONLY

RUN_DIR="$PREP_ROOT/GW250118/AMSS_NCKU_output"
if [[ ! -x "$RUN_DIR/ABEGPU" || ! -f "$RUN_DIR/input.par" || \
      ! -f "$RUN_DIR/Ansorg.psid" ]]; then
    echo "prepared GPU input is incomplete: $RUN_DIR" >&2
    exit 1
fi

MPI_CMD=(mpiexec --allow-run-as-root --bind-to core -n 1)
cd "$RUN_DIR"

case "$TOOL" in
    vtune)
        VTUNE_BIN="$(find_vtune)" || {
            echo "vtune executable was not found" >&2
            exit 127
        }
        "$VTUNE_BIN" -collect hotspots \
            -knob sampling-mode=sw \
            -result-dir "$ARTIFACT_DIR/vtune-result" \
            -knob enable-stack-collection=true \
            -- "${MPI_CMD[@]}" ./ABEGPU < /dev/null \
            2>&1 | tee "$ARTIFACT_DIR/vtune-collect.log"
        "$VTUNE_BIN" -report summary -format text \
            -result-dir "$ARTIFACT_DIR/vtune-result" \
            -report-output "$ARTIFACT_DIR/vtune-summary.txt"
        "$VTUNE_BIN" -report hotspots -format csv \
            -result-dir "$ARTIFACT_DIR/vtune-result" \
            -report-output "$ARTIFACT_DIR/vtune-hotspots.csv"
        ;;
    nsys)
        command -v nsys >/dev/null
        nsys profile \
            --trace=cuda,nvtx,osrt,mpi \
            --mpi-impl=openmpi \
            --sample=cpu \
            --cpuctxsw=process-tree \
            --force-overwrite=true \
            --output="$ARTIFACT_DIR/nsys" \
            "${MPI_CMD[@]}" ./ABEGPU < /dev/null \
            2>&1 | tee "$ARTIFACT_DIR/nsys-collect.log"
        nsys stats \
            --report=cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum \
            --format=csv \
            "$ARTIFACT_DIR/nsys.nsys-rep" \
            > "$ARTIFACT_DIR/nsys-stats.csv"
        ;;
    ncu)
        command -v ncu >/dev/null
        KERNEL_REGEX="${NCU_KERNEL_REGEX:-rhs_kernel}"
        LAUNCH_SKIP="${NCU_LAUNCH_SKIP:-0}"
        LAUNCH_COUNT="${NCU_LAUNCH_COUNT:-1}"
        echo "kernel regex: $KERNEL_REGEX"
        echo "launch skip/count: $LAUNCH_SKIP/$LAUNCH_COUNT"
        ncu \
            --target-processes all \
            --kernel-name-base demangled \
            --kernel-name "regex:$KERNEL_REGEX" \
            --launch-skip "$LAUNCH_SKIP" \
            --launch-count "$LAUNCH_COUNT" \
            --kill yes \
            --clock-control none \
            --set full \
            --import-source yes \
            --force-overwrite \
            --export "$ARTIFACT_DIR/ncu" \
            "${MPI_CMD[@]}" ./ABEGPU < /dev/null \
            2>&1 | tee "$ARTIFACT_DIR/ncu-collect.log"
        ncu --import "$ARTIFACT_DIR/ncu.ncu-rep" \
            --page details --csv > "$ARTIFACT_DIR/ncu-details.csv"
        ;;
esac

cd "$ROOT_DIR"
if [[ "$TOOL" == "ncu" ]]; then
    echo "=== Correctness skipped ==="
    echo "ncu --kill yes intentionally terminates after the requested launch count;"
    echo "validate the same binary/input with the nsys or unprofiled workflow."
else
    echo "=== Correctness ==="
    ./check.sh "$RUN_DIR" | tee "$ARTIFACT_DIR/check.txt"
fi

if [[ "$TOOL" == "nsys" ]]; then
    echo "=== Nsys summary ==="
    cat "$ARTIFACT_DIR/nsys-stats.csv"
elif [[ "$TOOL" == "vtune" ]]; then
    echo "=== VTune summary ==="
    cat "$ARTIFACT_DIR/vtune-summary.txt"
fi

echo "artifacts: $ARTIFACT_DIR"
