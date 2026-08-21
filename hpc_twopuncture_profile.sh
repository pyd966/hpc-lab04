#!/bin/bash
#HPC --partition=lab4
#HPC --cpu=60
#HPC --mem=100Gi
#HPC --time=30m
#HPC --name=twop-profile
#HPC --output=profile/twopuncture_%x_%j.log
#
# Profile only the shared TwoPuncture initial-data solver.  TwoPunctureABE is
# explicitly built without OpenMP; the 60-CPU allocation is used for a stable
# node and does not imply 60 MPI ranks.
set -euo pipefail

# hpc submits a temporary copy of a script (often under /tmp), while keeping
# the submit directory as the job working directory. Resolve paths from PWD,
# with an explicit override for interactive/debug use.
ROOT_DIR="${AMSS_REPO_ROOT:-$PWD}"
ROOT_DIR="$(cd -- "$ROOT_DIR" && pwd)"
RUN_ID="${HPC_JOB_ID:-${SLURM_JOB_ID:-$(date -u +%Y%m%dT%H%M%SZ)-${BASHPID:-0}}}"
PROFILE_DIR="$ROOT_DIR/profile/twopuncture-$RUN_ID"
RUN_DIR="$PROFILE_DIR/run"
BUILD_DIR="$ROOT_DIR/build-cpu-twop-profile"
mkdir -p "$PROFILE_DIR" "$RUN_DIR"
exec > >(tee "$PROFILE_DIR/job.log") 2>&1

echo "run id: $RUN_ID"
echo "git revision: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "host: $(hostname)"
echo "cpuset: $(awk '/Cpus_allowed_list/ {print $2}' /proc/self/status)"

"$ROOT_DIR/scripts/collect_cpu_info.sh" > "$PROFILE_DIR/cpu-info.txt"

echo "=== Generate TwoPuncture input ==="
export TWOP_RUN_DIR="$RUN_DIR"
python3 - <<'PY'
import os
import AMSS_NCKU_Input as input_data

input_data.File_directory = os.path.abspath(os.environ["TWOP_RUN_DIR"])
from scripts import generate_TwoPuncture_input
generate_TwoPuncture_input.generate_AMSSNCKU_TwoPuncture_input()
PY
cp "$RUN_DIR/AMSS-NCKU-TwoPuncture.input" "$RUN_DIR/TwoPunctureinput.par"

echo "=== Build with debug symbols ==="
export AMSS_BUILD_DIR="$BUILD_DIR"
export JOBS="$(nproc)"
./compile.sh -DAMSS_ENABLE_GPU=OFF \
    -DAMSS_ENABLE_OPENMP=OFF \
    -DAMSS_ENABLE_TWOPUNCTURE_OPENMP=OFF \
    -DAMSS_OPT="-O3 -g -fno-omit-frame-pointer"
cp "$BUILD_DIR/TwoPunctureABE" "$RUN_DIR/TwoPunctureABE"

cd "$RUN_DIR"
echo "=== perf stat: one serial solver run ==="
perf stat -d -d -o "$PROFILE_DIR/perf-stat.txt" -- ./TwoPunctureABE \
    2>&1 | tee "$PROFILE_DIR/twopuncture-stat.log"

echo "=== perf record: one serial solver run ==="
perf record -m 1 -F 99 --call-graph fp -o "$PROFILE_DIR/perf.data" \
    -- ./TwoPunctureABE 2>&1 | tee "$PROFILE_DIR/twopuncture-record.log"

echo "=== perf report ==="
perf report --stdio --no-children --sort comm,dso,symbol \
    --percent-limit 0.5 -i "$PROFILE_DIR/perf.data" \
    > "$PROFILE_DIR/perf-report-flat.txt"
perf report --stdio --children --sort comm,dso,symbol \
    --percent-limit 0.5 -i "$PROFILE_DIR/perf.data" \
    > "$PROFILE_DIR/perf-report-callgraph.txt"
perf report --stdio --call-graph none --no-children --sort srcline,symbol \
    --percent-limit 0.1 -i "$PROFILE_DIR/perf.data" \
    > "$PROFILE_DIR/perf-report-lines.txt"

echo "=== correctness artifacts ==="
test -s "$RUN_DIR/Ansorg.psid"
test -s "$RUN_DIR/puncture_parameters_new.txt"
sha256sum "$RUN_DIR/Ansorg.psid" "$RUN_DIR/puncture_parameters_new.txt" \
    | tee "$PROFILE_DIR/output-sha256.txt"
echo "Artifacts: $PROFILE_DIR"
