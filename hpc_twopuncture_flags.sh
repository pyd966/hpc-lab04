#!/bin/bash
#HPC --partition=lab4
#HPC --cpu=60
#HPC --mem=100Gi
#HPC --time=30m
#HPC --name=twop-flags
#HPC --output=profile/twopuncture_%x_%j.log
set -euo pipefail

ROOT_DIR="${AMSS_REPO_ROOT:-$PWD}"
ROOT_DIR="$(cd -- "$ROOT_DIR" && pwd)"
RUN_ID="${HPC_JOB_ID:-${SLURM_JOB_ID:-$(date -u +%Y%m%dT%H%M%SZ)-${BASHPID:-0}}}"
PROFILE_DIR="$ROOT_DIR/profile/twopuncture-flags-$RUN_ID"
mkdir -p "$PROFILE_DIR"
exec > >(tee "$PROFILE_DIR/job.log") 2>&1

echo "run id: $RUN_ID"
echo "git revision: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "host: $(hostname)"
echo "cpuset: $(awk '/Cpus_allowed_list/ {print $2}' /proc/self/status)"
"$ROOT_DIR/scripts/collect_cpu_info.sh" > "$PROFILE_DIR/cpu-info.txt"

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

names=(fast-march fast-cpu)
opts=(
  "-Ofast -g -fno-omit-frame-pointer"
  "-Ofast -g -fno-omit-frame-pointer"
)
archs=("-march=native" "-mcpu=native")
reference="${names[0]}"

echo -e "candidate\tseconds\tcycles\tinstructions\tipc\tbitwise\tmax_abs\teligible" > "$PROFILE_DIR/candidates.tsv"

for index in "${!names[@]}"; do
  name="${names[$index]}"
  opt="${opts[$index]}"
  arch="${archs[$index]}"
  build_dir="$ROOT_DIR/build-cpu-twop-flags-$name"
  run_dir="$PROFILE_DIR/run-$name"
  mkdir -p "$run_dir"

  echo "=== Build $name: AMSS_OPT='$opt' AMSS_ARCH_FLAGS='$arch' ==="
  cmake -S "$ROOT_DIR" -B "$build_dir" -DAMSS_ENABLE_GPU=OFF -DAMSS_ENABLE_OPENMP=OFF -DAMSS_ENABLE_TWOPUNCTURE_OPENMP=OFF -DAMSS_TWOPUNCTURE_OPT="$opt" -DAMSS_TWOPUNCTURE_ARCH_FLAGS="$arch"
  cmake --build "$build_dir" --target TwoPunctureABE -j "$(nproc)"
  cp "$build_dir/TwoPunctureABE" "$run_dir/TwoPunctureABE"
  cp "$INPUT_DIR/TwoPunctureinput.par" "$run_dir/TwoPunctureinput.par"

  echo "=== perf stat: $name ==="
  (
    cd "$run_dir"
    perf stat -d -d -o "$PROFILE_DIR/perf-stat-$name.txt" -- ./TwoPunctureABE 2>&1 | tee "$PROFILE_DIR/twopuncture-$name.log"
  )

  bitwise=yes
  max_abs=0
  eligible=yes
  if [[ "$name" != "$reference" ]]; then
    cmp -s "$PROFILE_DIR/run-$reference/puncture_parameters_new.txt" "$run_dir/puncture_parameters_new.txt" || eligible=no
    diff -q <(tail -n +2 "$PROFILE_DIR/run-$reference/Ansorg.psid") <(tail -n +2 "$run_dir/Ansorg.psid") >/dev/null || bitwise=no
    max_abs="$(paste <(tail -n +26 "$PROFILE_DIR/run-$reference/Ansorg.psid") <(tail -n +26 "$run_dir/Ansorg.psid") | awk '{d=$1-$2; if(d<0)d=-d; if(d>m)m=d} END {printf "%.17g", m}')"
    awk -v error="$max_abs" 'BEGIN {exit !(error <= 1e-12)}' || eligible=no
  fi

  stat_file="$PROFILE_DIR/perf-stat-$name.txt"
  seconds="$(awk '/seconds time elapsed/ {print $1}' "$stat_file")"
  cycles="$(awk '/cycles:u/ {print $1}' "$stat_file")"
  instructions="$(awk '/instructions:u/ {print $1}' "$stat_file")"
  ipc="$(awk '/instructions:u/ {print $4}' "$stat_file")"
  echo -e "$name\t$seconds\t$cycles\t$instructions\t$ipc\t$bitwise\t$max_abs\t$eligible" | tee -a "$PROFILE_DIR/candidates.tsv"
done

selected="$(awk -F '\t' 'NR > 1 && $8 == "yes" && (!best || $2 + 0 < best) {best=$2+0; name=$1} END {print name}' "$PROFILE_DIR/candidates.tsv")"
test -n "$selected"
echo "$selected" | tee "$PROFILE_DIR/selected.txt"

SELECTED_RUN="$PROFILE_DIR/run-$selected"
echo "=== perf record: selected $selected ==="
(
  cd "$SELECTED_RUN"
  perf record -m 1 -F 99 --call-graph fp -o "$PROFILE_DIR/perf.data" -- ./TwoPunctureABE 2>&1 | tee "$PROFILE_DIR/twopuncture-record.log"
)

perf report --stdio --no-children --sort comm,dso,symbol --percent-limit 0.5 -i "$PROFILE_DIR/perf.data" > "$PROFILE_DIR/perf-report-flat.txt"
perf report --stdio --children --sort comm,dso,symbol --percent-limit 0.5 -i "$PROFILE_DIR/perf.data" > "$PROFILE_DIR/perf-report-callgraph.txt"
perf report --stdio --call-graph none --no-children --sort srcline,symbol --percent-limit 0.1 -i "$PROFILE_DIR/perf.data" > "$PROFILE_DIR/perf-report-lines.txt"

sha256sum "$SELECTED_RUN/Ansorg.psid" "$SELECTED_RUN/puncture_parameters_new.txt" | tee "$PROFILE_DIR/output-sha256.txt"
echo "Artifacts: $PROFILE_DIR"
