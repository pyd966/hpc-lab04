#!/bin/bash
# run.sh — run the AMSS-NCKU Lab 4 driver.
#
# Path layout (relative values resolve against the lab root):
#   AMSS_BUILD_DIR    build output          (default: <lab4>/build)
#   AMSS_OUTPUT_ROOT  run directory parent  (default: <lab4>)
#   AMSS_CACHE_DIR    TwoPuncture cache root (default: <lab4>/twopuncture_cache)
#   AMSS_MPIEXEC      MPI launcher          (default: mpiexec)
set -euo pipefail

# This marker is intentionally kept in an OJ-accepted file. It makes it
# possible to identify the submitted CPU launch wrapper when the checkout has
# no .git directory.
CPU_ROUTE_MARKER="omp-30-geometry-v3"

# Ansorg-TwoPuncture allocates large Fortran automatic arrays.
ulimit -s unlimited

# Resolve the repository from the submission working directory. hpc may run
# a copied script from /tmp, while the OJ invokes this file from the checkout;
# AMSS_ROOT_DIR is exported below so the in-process launcher can be called
# later from ABE's output directory.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${AMSS_ROOT_DIR:-$PWD}"
if [[ ! -f "$ROOT_DIR/CMakeLists.txt" && -f "$SCRIPT_DIR/CMakeLists.txt" ]]; then
  ROOT_DIR="$SCRIPT_DIR"
fi
ROOT_DIR="$(cd -- "$ROOT_DIR" && pwd)"

read_allowed_cpu_list() {
  if [[ -r /proc/self/status ]]; then
    awk '/^Cpus_allowed_list:/ { print $2; exit }' /proc/self/status
  fi
}

# Preserve the scheduler cpuset before Python imports OpenMP-enabled libraries.
inherited_cpu_list="$(read_allowed_cpu_list)"
scheduler_cpu_list="${AMSS_SCHEDULER_CPU_LIST:-$inherited_cpu_list}"
if [[ -z "$scheduler_cpu_list" ||
      ! "$scheduler_cpu_list" =~ ^[0-9]+(-[0-9]+)?(,[0-9]+(-[0-9]+)?)*$ ]]; then
  echo "Invalid scheduler CPU list: ${scheduler_cpu_list:-empty}" >&2
  exit 2
fi
export AMSS_SCHEDULER_CPU_LIST="$scheduler_cpu_list"

file_digest() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    printf '%s' "missing"
  elif command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$path" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$path" | awk '{print $1}'
  else
    cksum "$path" | awk '{print $1 ":" $2}'
  fi
}

print_submission_identity() {
  local git_revision="unavailable"
  local allowed_cpus="unavailable"
  local online_cpus="unavailable"
  if [[ -d "$ROOT_DIR/.git" ]] && command -v git >/dev/null 2>&1; then
    git_revision="$(git -C "$ROOT_DIR" rev-parse --short HEAD 2>/dev/null || printf '%s' unavailable)"
  fi
  if [[ -r /proc/self/status ]]; then
    allowed_cpus="$(awk '/^Cpus_allowed_list:/ {print $2}' /proc/self/status 2>/dev/null || true)"
    [[ -n "$allowed_cpus" ]] || allowed_cpus="unavailable"
  fi
  if command -v nproc >/dev/null 2>&1; then
    online_cpus="$(nproc 2>/dev/null || printf '%s' unavailable)"
  elif command -v getconf >/dev/null 2>&1; then
    online_cpus="$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf '%s' unavailable)"
  fi
  echo "==> CPU route marker : $CPU_ROUTE_MARKER"
  echo "==> Repository root   : $ROOT_DIR"
  echo "==> Git revision      : $git_revision"
  echo "==> run.sh sha256     : $(file_digest "$ROOT_DIR/run.sh")"
  echo "==> CMakeLists sha256 : $(file_digest "$ROOT_DIR/CMakeLists.txt")"
  echo "==> compile.sh sha256 : $(file_digest "$ROOT_DIR/compile.sh")"
  if [[ -f "$ROOT_DIR/AMSS_NCKU_Input.py" ]]; then
    echo "==> input sha256      : $(file_digest "$ROOT_DIR/AMSS_NCKU_Input.py")"
  fi
  echo "==> Host              : $(hostname 2>/dev/null || printf '%s' unavailable)"
  echo "==> Online CPUs       : $online_cpus"
  echo "==> Allowed CPU list  : $allowed_cpus"
  echo "==> Process ID        : $$"
}

print_openmp_diagnostics() {
  local cache_file="$AMSS_BUILD_DIR/CMakeCache.txt"
  local key line
  echo "==> Build flags"
  for key in AMSS_ENABLE_GPU AMSS_ENABLE_OPENMP AMSS_ENABLE_OMP_ONLY \
             AMSS_ENABLE_TWOPUNCTURE_OPENMP AMSS_OPT AMSS_TWOPUNCTURE_OPT \
             CMAKE_CXX_COMPILER CMAKE_CXX_FLAGS; do
    line="$(grep -E "^${key}(:|=)" "$cache_file" 2>/dev/null | head -n 1 || true)"
    [[ -n "$line" ]] && echo "    $line"
  done
  if [[ -x "$AMSS_BUILD_DIR/ABE" ]]; then
    echo "==> ABE binary       : $AMSS_BUILD_DIR/ABE"
    if command -v file >/dev/null 2>&1; then
      echo "==> ABE file          : $(file "$AMSS_BUILD_DIR/ABE")"
    fi
    if command -v ldd >/dev/null 2>&1; then
      local dependencies
      dependencies="$(ldd "$AMSS_BUILD_DIR/ABE" 2>&1 || true)"
      if printf '%s\n' "$dependencies" | grep -Eq 'libgomp|libomp'; then
        echo "==> OpenMP runtime    : $(printf '%s\n' "$dependencies" | grep -E 'libgomp|libomp' | tr '\n' ';')"
      else
        echo "==> OpenMP runtime    : not found by ldd (runtime may be static)"
      fi
      if printf '%s\n' "$dependencies" | grep -Eq 'libmpi|libopen-rte|libopen-pal'; then
        echo "==> MPI runtime        : PRESENT ($(printf '%s\n' "$dependencies" | grep -E 'libmpi|libopen-rte|libopen-pal' | tr '\n' ';'))"
      else
        echo "==> MPI runtime        : absent from ABE dependencies"
      fi
    else
      echo "==> Dynamic libraries  : ldd unavailable"
    fi
  fi
  echo "==> OMP environment"
  for key in OMP_NUM_THREADS OMP_DYNAMIC OMP_PROC_BIND OMP_PLACES OMP_SCHEDULE \
             OMP_DISPLAY_ENV OMP_WAIT_POLICY AMSS_OMP_ONLY_RUN \
             AMSS_OMP_STATIC_THREADS AMSS_OMP_MOVING_THREADS \
             AMSS_OMP_STATIC_BLOCK_TARGET AMSS_OMP_MOVING_BLOCK_TARGET; do
    printf '    %s=%s\n' "$key" "${!key-<unset>}"
  done
}

# The OJ keeps its own Python launcher and does not collect scripts/*.py. Its
# launcher can still call `$AMSS_MPIEXEC ... ./ABE`; this mode consumes the
# MPI-shaped arguments and executes the one OpenMP process directly.
if [[ "${1:-}" == "--amss-omp-launch" ]]; then
  shift
  executable=""
  saw_env=0
  while (( $# > 0 )); do
    argument="$1"
    shift
    if [[ "$argument" == "env" ]]; then
      saw_env=1
      continue
    fi
    if (( saw_env )); then
      if [[ "$argument" == *=* ]]; then
        export "$argument"
        continue
      fi
      executable="$argument"
      break
    fi
    if [[ "$argument" == "./ABE" || "$argument" == "./ABEGPU" ||
          "$argument" == "./TwoPunctureABE" ]]; then
      executable="$argument"
      break
    fi
  done
  if [[ -z "$executable" ]]; then
    echo "CPU launcher could not find an executable in MPI-shaped command" >&2
    exit 2
  fi
  echo "==> CPU launcher marker: $CPU_ROUTE_MARKER"
  echo "==> CPU launcher mode  : direct OpenMP exec (MPI mapping bypassed)"
  echo "==> CPU launcher cwd   : $PWD"
  echo "==> CPU launcher exe   : $executable"
  echo "==> CPU launcher PID   : $$"
  echo "==> CPU launcher OMP   : threads=${OMP_NUM_THREADS:-<unset>} places=${OMP_PLACES:-<unset>} bind=${OMP_PROC_BIND:-<unset>} schedule=${OMP_SCHEDULE:-<unset>} display=${OMP_DISPLAY_ENV:-<unset>}"
  launcher_cpu_list="$(read_allowed_cpu_list)"
  echo "==> CPU launcher affinity: inherited=${launcher_cpu_list:-unknown} target=${AMSS_SCHEDULER_CPU_LIST:-unknown}"
  if [[ -n "${AMSS_SCHEDULER_CPU_LIST:-}" &&
        "$launcher_cpu_list" != "$AMSS_SCHEDULER_CPU_LIST" ]]; then
    if command -v taskset >/dev/null 2>&1; then
      export AMSS_AFFINITY_PREPARED=1
      exec taskset --cpu-list "$AMSS_SCHEDULER_CPU_LIST" "$executable" "$@"
    fi
    echo "Warning: taskset is unavailable; $executable will self-restore affinity" >&2
  else
    export AMSS_AFFINITY_PREPARED=1
  fi
  exec "$executable" "$@"
fi

unset AMSS_AFFINITY_PREPARED

# OJ executes this script directly. CPU OpenMP-only is therefore the default
# execution mode; GPU workflows set AMSS_EXECUTION_MODE=gpu explicitly.
AMSS_EXECUTION_MODE="${AMSS_EXECUTION_MODE:-cpu}"
case "$AMSS_EXECUTION_MODE" in
  cpu|gpu) ;;
  *) echo "AMSS_EXECUTION_MODE must be 'cpu' or 'gpu'" >&2; exit 2 ;;
esac
export AMSS_EXECUTION_MODE AMSS_ROOT_DIR="$ROOT_DIR"

if [[ "$AMSS_EXECUTION_MODE" == "cpu" ]]; then
  # Count physical cores in the scheduler-provided cpuset. Explicit
  # OMP_NUM_THREADS remains an escape hatch for profiling and sweeps.
  if [[ -x "$ROOT_DIR/scripts/count_available_physical_cores.sh" ]]; then
    detected_cores="$($ROOT_DIR/scripts/count_available_physical_cores.sh)"
  else
    detected_cores="$(nproc 2>/dev/null || echo 1)"
    # The OJ only collects selected files, so the topology helper is absent.
    # This workload's measured optimum is 24/30 blocks and workers; scaling
    # the decomposition to all 60 allocated cores makes the AMR tasks too fine.
    if [[ "$detected_cores" =~ ^[1-9][0-9]*$ ]] && (( detected_cores > 30 )); then
      detected_cores=30
    fi
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
  # Ask the OpenMP runtime itself to echo the settings it actually consumes.
  export OMP_DISPLAY_ENV="${OMP_DISPLAY_ENV:-TRUE}"
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
if [[ "$AMSS_EXECUTION_MODE" == "cpu" ]]; then
  # Override any scheduler-provided mpiexec command. This is needed even
  # when the OJ's unsubmitted Python helper ignores AMSS_OMP_ONLY_RUN.
  AMSS_MPIEXEC="$ROOT_DIR/run.sh --amss-omp-launch"
else
  AMSS_MPIEXEC="${AMSS_MPIEXEC:-mpiexec}"
fi
export AMSS_BUILD_DIR AMSS_OUTPUT_ROOT AMSS_CACHE_DIR AMSS_MPIEXEC

if [[ "$AMSS_EXECUTION_MODE" == "cpu" ]]; then
  # A direct OJ invocation may start from a clean checkout. Reconfigure when
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
  echo "==> CPU set  : inherited=${inherited_cpu_list:-unknown} restore=${AMSS_SCHEDULER_CPU_LIST:-unknown}"
  echo "==> OpenMP   : threads=$OMP_NUM_THREADS places=$OMP_PLACES bind=$OMP_PROC_BIND schedule=$OMP_SCHEDULE"
  echo "==> OMP work : static=$AMSS_OMP_STATIC_THREADS/$AMSS_OMP_STATIC_BLOCK_TARGET moving=$AMSS_OMP_MOVING_THREADS/$AMSS_OMP_MOVING_BLOCK_TARGET"
  print_submission_identity
  print_openmp_diagnostics
fi

cd "$ROOT_DIR"
runtime_log="$(mktemp "${TMPDIR:-/tmp}/amss-runtime.XXXXXX.log")"
set +e
"$PYTHON" AMSS_NCKU_Program.py 2>&1 | tee "$runtime_log"
pipeline_status=$?
set -e

echo "==> AMSS performance recap"
grep -E 'TwoPunctures\.C (affinity|OpenMP|Solve wall time)|ABE (affinity|OpenMP)|Before Evolve|Total Evolve Time|Total Running Time' \
  "$runtime_log" || echo "    expected native timing markers were not found"
awk '
  /AMSS_STEP_TIMING/ {
    if (count == 0) first = $0
    last = $0
    count++
  }
  END {
    if (count > 0) {
      print "    first: " first
      print "    last : " last
      print "    timed evolution steps: " count
    }
  }
' "$runtime_log"
rm -f "$runtime_log"
exit "$pipeline_status"
