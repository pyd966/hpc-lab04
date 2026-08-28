#!/bin/bash
# compile.sh — portable build driver for AMSS-NCKU Lab 4.
#
# Toolchain comes from environment (CC/CXX/FC/MPI_CXX_COMPILER/CUDACXX) or
# PATH (mpicxx/mpifort). AMSS_* vars are forwarded to CMake only when set.
# Build directory: AMSS_BUILD_DIR (default <lab-root>/build), relative to
# lab root. Extra args (e.g. -DAMSS_ENABLE_GPU=ON) override at the end.
set -euo pipefail

ROOT_DIR="$(pwd)"
CMAKE="${CMAKE:-cmake}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 2)}"

# Resolve build directory relative to the lab root.
BUILD_DIR="${AMSS_BUILD_DIR:-$ROOT_DIR/build}"
case "$BUILD_DIR" in
  /*) : ;;
   *) BUILD_DIR="$ROOT_DIR/$BUILD_DIR" ;;
esac

cmake_args=()
# MPI wrapper wins over plain CXX — the wrapper carries the right MPI flags
# and include paths. Setting both to different values is almost always a
# mistake, so warn instead of silently letting CXX override the wrapper.
if [[ -n "${MPI_CXX_COMPILER:-}" && -n "${CXX:-}" && "${MPI_CXX_COMPILER}" != "${CXX}" ]]; then
  echo "warning: both MPI_CXX_COMPILER ($MPI_CXX_COMPILER) and CXX ($CXX) are set;" >&2
  echo "         using MPI_CXX_COMPILER for the C++ compiler (MPI wrapper)." >&2
fi
if [[ -n "${MPI_CXX_COMPILER:-}" ]]; then
  cmake_args+=("-DCMAKE_CXX_COMPILER=$MPI_CXX_COMPILER")
elif [[ -n "${CXX:-}" ]]; then
  cmake_args+=("-DCMAKE_CXX_COMPILER=$CXX")
fi
[[ -n "${FC:-}" ]]                && cmake_args+=("-DCMAKE_Fortran_COMPILER=$FC")
[[ -n "${CUDACXX:-}" ]]           && cmake_args+=("-DCMAKE_CUDA_COMPILER=$CUDACXX")
[[ -n "${AMSS_ENABLE_GPU:-}" ]]    && cmake_args+=("-DAMSS_ENABLE_GPU=$AMSS_ENABLE_GPU")
[[ -n "${AMSS_CUDA_ARCHITECTURES:-}" ]] && cmake_args+=("-DCMAKE_CUDA_ARCHITECTURES=$AMSS_CUDA_ARCHITECTURES")
[[ -n "${AMSS_ARCH_FLAGS:-}" ]]    && cmake_args+=("-DAMSS_ARCH_FLAGS=$AMSS_ARCH_FLAGS")
[[ -n "${AMSS_ENABLE_OPENMP:-}" ]] && cmake_args+=("-DAMSS_ENABLE_OPENMP=$AMSS_ENABLE_OPENMP")
[[ -n "${AMSS_ENABLE_OMP_ONLY:-}" ]] && cmake_args+=("-DAMSS_ENABLE_OMP_ONLY=$AMSS_ENABLE_OMP_ONLY")
[[ -n "${AMSS_ENABLE_TWOPUNCTURE_OPENMP:-}" ]] && cmake_args+=("-DAMSS_ENABLE_TWOPUNCTURE_OPENMP=$AMSS_ENABLE_TWOPUNCTURE_OPENMP")
[[ -n "${AMSS_MPI_CUDA_AWARE:-}" ]] && cmake_args+=("-DAMSS_MPI_CUDA_AWARE=$AMSS_MPI_CUDA_AWARE")

# The CPU and GPU OJs collect this same script. Explicit mode/arguments win;
# on a clean automatic build, the allocated NVIDIA device selects the GPU
# target and a host without one selects the OpenMP-only CPU target.
compile_mode="${AMSS_EXECUTION_MODE:-auto}"
gpu_requested=0
gpu_choice_explicit=0
case "$compile_mode" in
  auto) ;;
  cpu) gpu_choice_explicit=1 ;;
  gpu) gpu_requested=1; gpu_choice_explicit=1 ;;
  *) echo "AMSS_EXECUTION_MODE must be 'auto', 'cpu' or 'gpu'" >&2; exit 2 ;;
esac
case "${AMSS_ENABLE_GPU:-}" in
  ON|on|TRUE|true|1) gpu_requested=1; gpu_choice_explicit=1 ;;
  OFF|off|FALSE|false|0) gpu_requested=0; gpu_choice_explicit=1 ;;
esac
for arg in "$@"; do
  case "$arg" in
    -DAMSS_ENABLE_GPU=ON|-DAMSS_ENABLE_GPU:BOOL=ON)
      gpu_requested=1; gpu_choice_explicit=1 ;;
    -DAMSS_ENABLE_GPU=OFF|-DAMSS_ENABLE_GPU:BOOL=OFF)
      gpu_requested=0; gpu_choice_explicit=1 ;;
  esac
done
if (( gpu_choice_explicit == 0 )) &&
   command -v nvidia-smi >/dev/null 2>&1 &&
   nvidia-smi -L >/dev/null 2>&1; then
  gpu_requested=1
fi
if (( gpu_requested == 0 )); then
  cmake_args+=("-DAMSS_ENABLE_GPU=OFF" "-DAMSS_ENABLE_OPENMP=ON" "-DAMSS_ENABLE_OMP_ONLY=ON")
else
  # Do not inherit CPU defaults when a build directory is reused for GPU.
  # Explicit trailing arguments can still override these defaults.
  cmake_args+=("-DAMSS_ENABLE_GPU=ON" "-DAMSS_ENABLE_OPENMP=OFF" "-DAMSS_ENABLE_OMP_ONLY=OFF")
fi

echo "==> Configure: $CMAKE -S \"$ROOT_DIR\" -B \"$BUILD_DIR\" ${cmake_args[*]:-} $*"
"$CMAKE" -S "$ROOT_DIR" -B "$BUILD_DIR" "${cmake_args[@]}" "$@"

echo "==> Build: $CMAKE --build \"$BUILD_DIR\" -j $JOBS"
"$CMAKE" --build "$BUILD_DIR" -j "$JOBS"

echo "==> Built executables:"
ls -lh "$BUILD_DIR/TwoPunctureABE" "$BUILD_DIR/ABE"
if [[ -f "$BUILD_DIR/ABEGPU" ]]; then
  ls -lh "$BUILD_DIR/ABEGPU"
fi
