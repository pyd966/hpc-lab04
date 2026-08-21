#!/bin/bash
# Record the allocation and toolchain details needed to reproduce a GPU run.
set -euo pipefail

echo "=== Job allocation ==="
echo "host: $(hostname)"
echo "kernel: $(uname -srmo)"
echo "available processing units: $(nproc)"
grep -E '^(Cpus_allowed_list|Mems_allowed_list):' /proc/self/status

echo
echo "=== CPU topology ==="
lscpu

echo
echo "=== GPU allocation ==="
nvidia-smi -L
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,compute_cap \
    --format=csv,noheader

echo
echo "=== Toolchain ==="
cmake --version | head -n 1
gcc --version | head -n 1
gfortran --version | head -n 1
nvcc --version | tail -n 1
mpiexec --version | head -n 3

for tool in nsys ncu; do
    echo
    echo "=== $tool ==="
    if command -v "$tool" >/dev/null 2>&1; then
        "$tool" --version 2>&1 | head -n 4
    else
        echo "$tool is unavailable"
    fi
done

echo
echo "=== vtune ==="
VTUNE_BIN="$(command -v vtune 2>/dev/null || true)"
if [[ -z "$VTUNE_BIN" ]]; then
    for candidate in /opt/intel/oneapi/vtune/*/bin64/vtune; do
        if [[ -x "$candidate" ]]; then
            VTUNE_BIN="$candidate"
            break
        fi
    done
fi
if [[ -n "$VTUNE_BIN" ]]; then
    "$VTUNE_BIN" --version 2>&1 | head -n 4
    echo "path: $VTUNE_BIN"
else
    echo "vtune is unavailable"
fi
