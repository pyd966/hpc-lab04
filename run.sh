#!/bin/bash
# run.sh — run the AMSS-NCKU Lab 4 driver.
#
# Path layout (relative values resolve against the lab root):
#   AMSS_BUILD_DIR    build output          (default: <lab4>/build)
#   AMSS_OUTPUT_ROOT  run directory parent  (default: <lab4>)
#   AMSS_CACHE_DIR    TwoPuncture cache root (default: <lab4>/twopuncture_cache)
#   AMSS_MPIEXEC      MPI launcher          (default: mpiexec)
set -euo pipefail

# Ansorg-TwoPuncture allocates large Fortran automatic arrays.
ulimit -s unlimited

# OJ executes this script directly.  CPU OpenMP-only is therefore the default
# execution mode; GPU workflows set AMSS_EXECUTION_MODE=gpu explicitly.
AMSS_EXECUTION_MODE="${AMSS_EXECUTION_MODE:-cpu}"
case "$AMSS_EXECUTION_MODE" in
  cpu|gpu) ;;
  *) echo "AMSS_EXECUTION_MODE must be 'cpu' or 'gpu'" >&2; exit 2 ;;
esac
export AMSS_EXECUTION_MODE

ROOT_DIR="$(pwd)"

if [[ "$AMSS_EXECUTION_MODE" == "cpu" ]]; then
  # Count physical cores in the scheduler-provided cpuset.  Explicit
  # OMP_NUM_THREADS remains an escape hatch for profiling and sweeps.
  if [[ -x "$ROOT_DIR/scripts/count_available_physical_cores.sh" ]]; then
    detected_cores="$($ROOT_DIR/scripts/count_available_physical_cores.sh)"
  else
    detected_cores="$(nproc 2>/dev/null || echo 1)"
  fi
  [[ "$detected_cores" =~ ^[1-9][0-9]*$ ]] || detected_cores=1
  OMP_NUM_THREADS="${OMP_NUM_THREADS:-$detected_cores}"
  if ! [[ "$OMP_NUM_THREADS" =~ ^[1-9][0-9]*$ ]]; then
    echo "OMP_NUM_THREADS must be one positive integer: $OMP_NUM_THREADS" >&2
    exit 2
  fi
  static_threads=$((OMP_NUM_THREADS * 4 / 5))
  (( static_threads > 0 )) || static_threads=1
  export OMP_NUM_THREADS
  export OMP_PLACES="${OMP_PLACES:-cores}"
  export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
  export OMP_SCHEDULE="${OMP_SCHEDULE:-dynamic,1}"
  export OMP_DYNAMIC="${OMP_DYNAMIC:-FALSE}"
  export AMSS_OMP_ONLY_RUN=1
  export AMSS_OMP_STATIC_THREADS="${AMSS_OMP_STATIC_THREADS:-$static_threads}"
  export AMSS_OMP_MOVING_THREADS="${AMSS_OMP_MOVING_THREADS:-$OMP_NUM_THREADS}"
  export AMSS_OMP_STATIC_BLOCK_TARGET="${AMSS_OMP_STATIC_BLOCK_TARGET:-$static_threads}"
  export AMSS_OMP_MOVING_BLOCK_TARGET="${AMSS_OMP_MOVING_BLOCK_TARGET:-$OMP_NUM_THREADS}"
fi

# HPC jobs run inside a root container even when submitted by a regular user.
# Open MPI 5.x (prterun) therefore needs its explicit container opt-in. These
# variables are ignored for non-root launches and preserve a caller-supplied
# AMSS_MPIEXEC.
export OMPI_ALLOW_RUN_AS_ROOT="${OMPI_ALLOW_RUN_AS_ROOT:-1}"
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="${OMPI_ALLOW_RUN_AS_ROOT_CONFIRM:-1}"
AMSS_MPIEXEC="${AMSS_MPIEXEC:-mpiexec --allow-run-as-root}"

PYTHON="${PYTHON:-python3}"

resolve_under_root() {
  case "$1" in
    /*) printf '%s' "$1" ;;
     *) printf '%s/%s' "$ROOT_DIR" "$1" ;;
  esac
}

AMSS_BUILD_DIR="$(resolve_under_root "${AMSS_BUILD_DIR:-$ROOT_DIR/build}")"
AMSS_OUTPUT_ROOT="$(resolve_under_root "${AMSS_OUTPUT_ROOT:-$ROOT_DIR}")"
AMSS_CACHE_DIR="$(resolve_under_root "${AMSS_CACHE_DIR:-$ROOT_DIR/twopuncture_cache}")"
AMSS_MPIEXEC="${AMSS_MPIEXEC:-mpiexec}"
export AMSS_BUILD_DIR AMSS_OUTPUT_ROOT AMSS_CACHE_DIR AMSS_MPIEXEC

if [[ "$AMSS_EXECUTION_MODE" == "cpu" ]]; then
  # A direct OJ invocation may start from a clean checkout.  Reconfigure when
  # the selected build is missing or was compiled for GPU/MPI execution.
  cpu_build_ok=1
  cpu_cache="$AMSS_BUILD_DIR/CMakeCache.txt"
  if [[ ! -x "$AMSS_BUILD_DIR/ABE" ||
        ! -x "$AMSS_BUILD_DIR/TwoPunctureABE" ||
        ! -f "$cpu_cache" ]]; then
    cpu_build_ok=0
  else
    grep -q '^AMSS_ENABLE_GPU:BOOL=OFF$' "$cpu_cache" || cpu_build_ok=0
    grep -q '^AMSS_ENABLE_OPENMP:BOOL=ON$' "$cpu_cache" || cpu_build_ok=0
    grep -q '^AMSS_ENABLE_OMP_ONLY:BOOL=ON$' "$cpu_cache" || cpu_build_ok=0
  fi
  if (( cpu_build_ok == 0 )); then
    echo "==> CPU build is missing or incompatible; configuring OpenMP-only ABE"
    "$ROOT_DIR/compile.sh" \
      -DAMSS_ENABLE_GPU=OFF \
      -DAMSS_ENABLE_OPENMP=ON \
      -DAMSS_ENABLE_OMP_ONLY=ON \
      -DAMSS_ENABLE_TWOPUNCTURE_OPENMP=ON \
      -DAMSS_OPT="${AMSS_RUN_BUILD_OPT:--O3}" \
      -DAMSS_ARCH_FLAGS="" \
      -DAMSS_TWOPUNCTURE_OPT="${AMSS_RUN_TWOP_OPT:--O3}" \
      -DAMSS_TWOPUNCTURE_ARCH_FLAGS="${AMSS_RUN_TWOP_ARCH:--march=native}"
  fi
fi

if [[ "${1:-}" == "--twop-cache" ]]; then
  export AMSS_NCKU_TWOP_CACHE=1
  shift
fi
if (( $# > 0 )); then
  echo "usage: ./run.sh [--twop-cache]" >&2
  exit 1
fi

echo "==> Build    : $AMSS_BUILD_DIR"
echo "==> Output   : $AMSS_OUTPUT_ROOT"
echo "==> Cache    : $AMSS_CACHE_DIR"
echo "==> MPI exec : $AMSS_MPIEXEC"
echo "==> Execution: $AMSS_EXECUTION_MODE"
if [[ "$AMSS_EXECUTION_MODE" == "cpu" ]]; then
  echo "==> OpenMP   : threads=$OMP_NUM_THREADS places=$OMP_PLACES bind=$OMP_PROC_BIND schedule=$OMP_SCHEDULE"
  echo "==> OMP work : static=$AMSS_OMP_STATIC_THREADS/$AMSS_OMP_STATIC_BLOCK_TARGET moving=$AMSS_OMP_MOVING_THREADS/$AMSS_OMP_MOVING_BLOCK_TARGET"
fi

cd "$ROOT_DIR"
"$PYTHON" AMSS_NCKU_Program.py
