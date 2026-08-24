#!/bin/bash
#HPC --partition=lab4g10
#HPC --cpu=16
#HPC --gpu=1
#HPC --mem=24Gi
#HPC --time=30m
#HPC --name=amss-rhs-pair
#HPC --output=profile/hpc_%x_%j.log
#HPC --export=NONE
set -euo pipefail

ROOT_DIR="${AMSS_ROOT_DIR:-$PWD}"
BASE_REV="${AMSS_PAIR_BASE_REV:-master}"
CANDIDATE_REV="${AMSS_PAIR_CANDIDATE_REV:?set AMSS_PAIR_CANDIDATE_REV}"
BENCHMARK_TIME="${AMSS_PAIR_TIME:-5}"
RUN_ID="${HPC_JOB_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$BASHPID}"
ARTIFACT_DIR="$ROOT_DIR/profile/gpu-rhs-pair-$RUN_ID"
PAIR_TMP="${TMPDIR:-/tmp}/amss-rhs-pair-$RUN_ID"

mkdir -p "$ARTIFACT_DIR" "$PAIR_TMP/base" "$PAIR_TMP/candidate"
exec > >(tee "$ARTIFACT_DIR/job.log") 2>&1

git -C "$ROOT_DIR" archive "$BASE_REV" | tar -xf - -C "$PAIR_TMP/base"
git -C "$ROOT_DIR" archive "$CANDIDATE_REV" | tar -xf - -C "$PAIR_TMP/candidate"

build_revision() {
    local label="$1"
    local source_dir="$PAIR_TMP/$label"
    (
        cd "$source_dir"
        export AMSS_BUILD_DIR="$source_dir/build-gpu"
        export JOBS="$(nproc)"
        ./compile.sh \
            -DAMSS_ENABLE_GPU=ON \
            -DCMAKE_CUDA_ARCHITECTURES=80 \
            -DAMSS_ENABLE_OPENMP=OFF \
            -DAMSS_ENABLE_TWOPUNCTURE_OPENMP=ON \
            -DAMSS_OPT=-O3
    ) 2>&1 | tee "$ARTIFACT_DIR/build-$label.log"
}

run_revision() {
    local label="$1"
    local sequence="$2"
    local source_dir="$PAIR_TMP/$label"
    local output_root="$ARTIFACT_DIR/$sequence-$label"
    local run_log="$ARTIFACT_DIR/$sequence-$label.log"
    local check_log="$ARTIFACT_DIR/$sequence-$label-check.txt"
    mkdir -p "$output_root"
    (
        cd "$source_dir"
        export AMSS_BUILD_DIR="$source_dir/build-gpu"
        export AMSS_OUTPUT_ROOT="$output_root"
        export AMSS_CACHE_DIR="$ROOT_DIR/profile/twopuncture-cache"
        export AMSS_MPIEXEC="mpiexec --allow-run-as-root --bind-to core"
        export AMSS_NCKU_FINAL_TIME="$BENCHMARK_TIME"
        export OMP_PLACES=cores
        export OMP_PROC_BIND=close
        TIMEFORMAT=$'outer_wall_seconds=%R\nuser_cpu_seconds=%U\nsystem_cpu_seconds=%S'
        { time ./run.sh --twop-cache; }
    ) 2>&1 | tee "$run_log"
    "$source_dir/check.sh" "$output_root/GW250118/AMSS_NCKU_output" | tee "$check_log"
    awk -v label="$label" -v sequence="$sequence" '
        /This Program Cost =/ {program=$(NF-1)}
        /^outer_wall_seconds=/ {split($0, value, "="); outer=value[2]}
        END {printf "%s\t%s\t%s\t%s\n", sequence, label, program, outer}
    ' "$run_log" >> "$ARTIFACT_DIR/timings.tsv"
}

echo "base revision: $(git -C "$ROOT_DIR" rev-parse "$BASE_REV")"
echo "candidate revision: $(git -C "$ROOT_DIR" rev-parse "$CANDIDATE_REV")"
echo "evolution end time: $BENCHMARK_TIME"
"$ROOT_DIR/scripts/collect_gpu_info.sh" > "$ARTIFACT_DIR/system-info.txt"
build_revision base
build_revision candidate
printf 'sequence\tvariant\tprogram_seconds\touter_wall_seconds\n' > "$ARTIFACT_DIR/timings.tsv"

# Alternate order to reduce warm-up and frequency-drift bias.
run_revision base 1
run_revision candidate 2
run_revision candidate 3
run_revision base 4

awk -F '\t' 'NR > 1 {sum[$2] += $3; count[$2] += 1}
    END {for (variant in sum) printf "%s_mean_program_seconds=%.6f\n", variant, sum[variant]/count[variant]}' \
    "$ARTIFACT_DIR/timings.tsv" | sort | tee "$ARTIFACT_DIR/summary.txt"
echo "artifacts: $ARTIFACT_DIR"
