#!/bin/bash
# Print CPU and NUMA information relevant to an HPC job allocation.
set -euo pipefail

echo "=== Job allocation ==="
echo "host: $(hostname)"
echo "kernel: $(uname -srmo)"
echo "available processing units: $(nproc)"
grep -E '^(Cpus_allowed_list|Mems_allowed_list):' /proc/self/status

echo
echo "=== CPU topology and ISA ==="
lscpu

echo
echo "=== CPU / core / NUMA mapping ==="
lscpu -e=CPU,NODE,SOCKET,CORE,ONLINE,MAXMHZ,MINMHZ

echo
echo "=== Cache hierarchy ==="
lscpu -C

echo
echo "=== NUMA topology ==="
if command -v numactl >/dev/null 2>&1; then
    numactl --hardware
else
    echo "numactl is unavailable"
fi

echo
echo "=== Toolchain ==="
cmake --version | head -n 1
gcc --version | head -n 1
gfortran --version | head -n 1
mpiexec --version | head -n 3
perf --version
