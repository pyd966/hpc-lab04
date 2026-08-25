#!/bin/bash
#HPC --partition=lab4
#HPC --cpu=60
#HPC --mem=100Gi
#HPC --time=30m
#HPC --name=abe-sve
#HPC --output=profile/hpc_%x_%j.log
set -euo pipefail

ROOT_DIR="${AMSS_ROOT_DIR:-$PWD}"
[[ -f "$ROOT_DIR/CMakeLists.txt" ]] || exit 2
cd "$ROOT_DIR"
RUN_ID="${HPC_JOB_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$BASHPID}"
OUT_DIR="$ROOT_DIR/profile/abe-sve-$RUN_ID"
PREP_ROOT="$OUT_DIR/prepare"
CACHE_DIR="$ROOT_DIR/profile/twopuncture-cache"
mkdir -p "$OUT_DIR" "$PREP_ROOT" "$CACHE_DIR"
exec > "$OUT_DIR/job.log" 2>&1

EVOLVE_TIME="${ABE_SVE_TIME:-4.0}"
RUN_ORDER="${ABE_SVE_RUN_ORDER:-off on on off}"
export JOBS="$(nproc)" OMP_NUM_THREADS=30 OMP_PLACES=cores OMP_PROC_BIND=close
export OMP_SCHEDULE=dynamic,1 AMSS_OMP_ONLY_RUN=1
export AMSS_OMP_STATIC_BLOCK_TARGET=24 AMSS_OMP_MOVING_BLOCK_TARGET=30
export AMSS_OMP_STATIC_THREADS=24 AMSS_OMP_MOVING_THREADS=30
"$ROOT_DIR/scripts/collect_cpu_info.sh" > "$OUT_DIR/cpu-info.txt"

for value in OFF ON; do
  name=off; arch_flags=""
  if [[ "$value" == ON ]]; then
    name=on; arch_flags='-march=armv8.2-a+sve'
  fi
  export AMSS_BUILD_DIR="$OUT_DIR/build-$name"
  ./compile.sh \
    -DAMSS_ENABLE_GPU=OFF -DAMSS_ENABLE_OPENMP=ON -DAMSS_ENABLE_OMP_ONLY=ON \
    -DAMSS_ARCH_FLAGS="$arch_flags" -DAMSS_ENABLE_LTO=OFF \
    -DAMSS_ENABLE_BLOCK_FIELD_ARENA=ON -DAMSS_ENABLE_HUGEPAGE_HINT=OFF \
    -DAMSS_ENABLE_FDERIVS_SIMD=ON -DAMSS_ENABLE_OMP_DIRECT_SYNC=ON \
    -DAMSS_ENABLE_OMP_DIRECT_AMR_TRANSFER=OFF -DAMSS_ENABLE_OMP_DIRECT_AMR_SPLIT=OFF \
    -DAMSS_ENABLE_PROLONG3_PAIRWISE=ON -DAMSS_ENABLE_PROLONG3_SIMD=OFF \
    -DAMSS_ENABLE_RHS_METRIC_FUSION=ON -DAMSS_ENABLE_RHS_GAMMA_FUSION=ON \
    -DAMSS_ENABLE_RHS_CONNECTION_FUSION=OFF -DAMSS_ENABLE_RHS_AIJ_FUSION=OFF \
    -DAMSS_ENABLE_RHS_CHI_RICCI_FUSION=OFF -DAMSS_ENABLE_TWOPUNCTURE_OPENMP=ON \
    -DAMSS_OPT='-O3 -g -fno-omit-frame-pointer -fopt-info-vec-optimized' \
    -DAMSS_TWOPUNCTURE_OPT=-O3 -DAMSS_TWOPUNCTURE_ARCH_FLAGS=-march=native \
    > "$OUT_DIR/build-$name.log" 2>&1
done

export AMSS_BUILD_DIR="$OUT_DIR/build-off"
export AMSS_OUTPUT_ROOT="$PREP_ROOT" AMSS_CACHE_DIR="$CACHE_DIR" AMSS_NCKU_PREPARE_ONLY=1
./run.sh --twop-cache > "$OUT_DIR/prepare.log" 2>&1
unset AMSS_NCKU_PREPARE_ONLY
BASE_DIR="$PREP_ROOT/GW250118/AMSS_NCKU_output"
[[ -f "$BASE_DIR/input.par" && -f "$BASE_DIR/Ansorg.psid" ]] || exit 1

reference=""
: > "$OUT_DIR/results.tsv"
for name in $RUN_ORDER; do
  case "$name" in
    off) value=OFF; build_name=off ;;
    on) value=ON; build_name=on ;;
    *) echo "invalid run order entry: $name" >&2; exit 2 ;;
  esac
  run_dir="$OUT_DIR/run-$name-$(date +%s%N)"
  mkdir -p "$run_dir/binary_output"
  cp "$OUT_DIR/build-$build_name/ABE" "$run_dir/ABE"
  cp "$BASE_DIR/Ansorg.psid" "$run_dir/Ansorg.psid"
  sed -E "s/^(ABE::TotalTime[[:space:]]*=[[:space:]]*).*/\\1$EVOLVE_TIME/" \
      "$BASE_DIR/input.par" > "$run_dir/input.par"
  perf stat -d -d -o "$run_dir/perf-stat.txt" -- \
      bash -c 'cd "$1" && exec ./ABE < /dev/null' _ "$run_dir" \
      > "$run_dir/run.log" 2>&1
  set +e
  ./check.sh "$run_dir/binary_output" > "$run_dir/check.txt" 2>&1
  set -e
  if [[ -z "$reference" && "$name" == off ]]; then reference="$run_dir/binary_output"; fi
  bitwise=yes
  if [[ -n "$reference" && "$run_dir/binary_output" != "$reference" ]]; then
    for output in bssn_ADMQs.dat bssn_BH.dat bssn_constraint.dat bssn_psi4.dat; do
      cmp -s <(tail -n +3 "$reference/$output") <(tail -n +3 "$run_dir/binary_output/$output") || bitwise=no
    done
  fi
  evolve="$(awk '/Total Evolve Time:/ {v=$4} END {print v}' "$run_dir/run.log")"
  printf '%s\t%s\t%s\t%s\n' "$name" "$value" "$evolve" "$bitwise" >> "$OUT_DIR/results.tsv"
done

printf 'candidate\tbuild\tevolve_seconds\tbitwise_vs_off\n'
cat "$OUT_DIR/results.tsv"
echo "Artifacts: $OUT_DIR"
