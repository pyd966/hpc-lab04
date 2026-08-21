#!/bin/bash
#HPC --partition=lab4
#HPC --cpu=60
#HPC --mem=100Gi
#HPC --time=30m
#HPC --name=twop-openmp
#HPC --output=profile/twopuncture_%x_%j.log
set -euo pipefail

ROOT_DIR="${AMSS_REPO_ROOT:-$PWD}"
ROOT_DIR="$(cd -- "$ROOT_DIR" && pwd)"
RUN_ID="${HPC_JOB_ID:-${SLURM_JOB_ID:-$(date -u +%Y%m%dT%H%M%SZ)-${BASHPID:-0}}}"
PROFILE_DIR="$ROOT_DIR/profile/twopuncture-openmp-$RUN_ID"
BUILD_DIR="$ROOT_DIR/build-cpu-twop-openmp"
mkdir -p "$PROFILE_DIR"
exec > >(tee "$PROFILE_DIR/job.log") 2>&1

echo "run id: $RUN_ID"
echo "git revision: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "host: $(hostname)"
echo "cpuset: $(awk '/Cpus_allowed_list/ {print $2}' /proc/self/status)"
"$ROOT_DIR/scripts/collect_cpu_info.sh" > "$PROFILE_DIR/cpu-info.txt"

logical_cpus="$(nproc)"
physical_cores="$(python3 - <<'PY'
import os
import subprocess

allowed = os.sched_getaffinity(0)
cores = set()
for line in subprocess.check_output(["lscpu", "-p=CPU,CORE"], text=True).splitlines():
    if line.startswith("#"):
        continue
    cpu, core = (int(value) for value in line.split(",")[:2])
    if cpu in allowed:
        cores.add(core)
print(len(cores))
PY
)"
test "$physical_cores" -gt 0
half_cores=$(( (physical_cores + 1) / 2 ))

export OMP_DYNAMIC=FALSE
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_WAIT_POLICY=ACTIVE
echo "logical CPUs in affinity: $logical_cpus"
echo "physical cores in affinity: $physical_cores"
echo "binding: OMP_PLACES=$OMP_PLACES OMP_PROC_BIND=$OMP_PROC_BIND"
TWOP_OPT="${AMSS_TWOPUNCTURE_OPT:--O3}"
echo "TwoPuncture optimization flags: $TWOP_OPT"

INPUT_DIR="$PROFILE_DIR/input"
mkdir -p "$INPUT_DIR"
export TWOP_RUN_DIR="$INPUT_DIR"
python3 - <<'PY'
import os
import AMSS_NCKU_Input as input_data

input_data.File_directory = os.path.abspath(os.environ["TWOP_RUN_DIR"])
from scripts import generate_TwoPuncture_input
generate_TwoPuncture_input.generate_AMSSNCKU_TwoPuncture_input()
PY
cp "$INPUT_DIR/AMSS-NCKU-TwoPuncture.input" "$INPUT_DIR/TwoPunctureinput.par"

cmake -S "$ROOT_DIR" -B "$BUILD_DIR" \
  -DAMSS_ENABLE_GPU=OFF \
  -DAMSS_ENABLE_OPENMP=OFF \
  -DAMSS_ENABLE_TWOPUNCTURE_OPENMP=ON \
  -DAMSS_TWOPUNCTURE_OPT="$TWOP_OPT" \
  -DAMSS_OPT="-O3 -g -fno-omit-frame-pointer"
cmake --build "$BUILD_DIR" --target TwoPunctureABE -j "$logical_cpus"

if [[ -n "${AMSS_OMP_CANDIDATES:-}" ]]; then
  read -r -a candidates <<< "$AMSS_OMP_CANDIDATES"
else
  candidates=(1 4 8 16 "$half_cores" "$physical_cores" "$logical_cpus")
fi
declare -A seen=()
echo -e "threads\tseconds\ttask_clock_ms\tcycles\tinstructions\tipc\tbitwise\tmax_abs\teligible" > "$PROFILE_DIR/candidates.tsv"

reference_dir=""
for threads in "${candidates[@]}"; do
  if [[ -n "${seen[$threads]:-}" || "$threads" -gt "$logical_cpus" ]]; then
    continue
  fi
  seen[$threads]=1
  run_dir="$PROFILE_DIR/run-t$threads"
  mkdir -p "$run_dir"
  cp "$BUILD_DIR/TwoPunctureABE" "$run_dir/TwoPunctureABE"
  cp "$INPUT_DIR/TwoPunctureinput.par" "$run_dir/TwoPunctureinput.par"

  export OMP_NUM_THREADS="$threads"
  export OMP_DISPLAY_AFFINITY=FALSE
  echo "=== perf stat: $threads OpenMP thread(s) ==="
  (
    cd "$run_dir"
    perf stat -d -d -o "$PROFILE_DIR/perf-stat-t$threads.txt" -- ./TwoPunctureABE \
      2>&1 | tee "$PROFILE_DIR/twopuncture-t$threads.log"
  )

  bitwise=yes
  max_abs=0
  eligible=yes
  if [[ -z "$reference_dir" ]]; then
    reference_dir="$run_dir"
  else
    cmp -s "$reference_dir/puncture_parameters_new.txt" "$run_dir/puncture_parameters_new.txt" || eligible=no
    diff -q <(tail -n +2 "$reference_dir/Ansorg.psid") \
            <(tail -n +2 "$run_dir/Ansorg.psid") >/dev/null || bitwise=no
    max_abs="$(paste <(tail -n +26 "$reference_dir/Ansorg.psid") \
                       <(tail -n +26 "$run_dir/Ansorg.psid") | \
      awk '{d=$1-$2; if(d<0)d=-d; if(d>m)m=d} END {printf "%.17g", m}')"
    awk -v error="$max_abs" 'BEGIN {exit !(error <= 1e-12)}' || eligible=no
  fi

  stat_file="$PROFILE_DIR/perf-stat-t$threads.txt"
  seconds="$(awk '/seconds time elapsed/ {print $1}' "$stat_file")"
  task_clock="$(awk '/task-clock:u/ {gsub(/,/, "", $1); print $1}' "$stat_file")"
  cycles="$(awk '/cycles:u/ {gsub(/,/, "", $1); print $1}' "$stat_file")"
  instructions="$(awk '/instructions:u/ {gsub(/,/, "", $1); print $1}' "$stat_file")"
  ipc="$(awk '/instructions:u/ {print $4}' "$stat_file")"
  echo -e "$threads\t$seconds\t$task_clock\t$cycles\t$instructions\t$ipc\t$bitwise\t$max_abs\t$eligible" | \
    tee -a "$PROFILE_DIR/candidates.tsv"
done

selected="$(awk -F '\t' '
  NR > 1 && $9 == "yes" {
    ++count; threads[count]=$1+0; seconds[count]=$2+0
    if (!best || seconds[count] < best) best=seconds[count]
  }
  END {
    limit=best*1.02
    for (i=1; i<=count; ++i)
      if (seconds[i] <= limit && (!selected || threads[i] < selected))
        selected=threads[i]
    print selected
  }' "$PROFILE_DIR/candidates.tsv")"
test -n "$selected"
echo "$selected" | tee "$PROFILE_DIR/selected-threads.txt"

export OMP_NUM_THREADS="$selected"
export OMP_DISPLAY_AFFINITY=TRUE
selected_run="$PROFILE_DIR/run-t$selected"
echo "=== perf record: selected $selected thread(s) ==="
(
  cd "$selected_run"
  perf record -m 1 -F 99 --sample-cpu --call-graph fp -o "$PROFILE_DIR/perf.data" \
    -- ./TwoPunctureABE 2>&1 | tee "$PROFILE_DIR/twopuncture-record.log"
)

perf report --stdio --no-children --sort comm,dso,symbol --percent-limit 0.5 \
  -i "$PROFILE_DIR/perf.data" > "$PROFILE_DIR/perf-report-flat.txt"
perf report --stdio --children --sort comm,dso,symbol --percent-limit 0.5 \
  -i "$PROFILE_DIR/perf.data" > "$PROFILE_DIR/perf-report-callgraph.txt"
perf report --stdio --call-graph none --no-children --sort srcline,symbol --percent-limit 0.1 \
  -i "$PROFILE_DIR/perf.data" > "$PROFILE_DIR/perf-report-lines.txt"
perf report --stdio --call-graph none --no-children --sort pid --percent-limit 0.1 \
  -i "$PROFILE_DIR/perf.data" > "$PROFILE_DIR/perf-report-threads.txt"
perf report --stdio --call-graph none --no-children --sort pid,symbol --percent-limit 0.01 \
  -i "$PROFILE_DIR/perf.data" > "$PROFILE_DIR/perf-report-thread-symbols.txt"
perf report --stdio --call-graph none --no-children --sort cpu --percent-limit 0.1 \
  -i "$PROFILE_DIR/perf.data" > "$PROFILE_DIR/perf-report-cpus.txt"

sha256sum "$selected_run/Ansorg.psid" "$selected_run/puncture_parameters_new.txt" | \
  tee "$PROFILE_DIR/output-sha256.txt"
echo "Artifacts: $PROFILE_DIR"
