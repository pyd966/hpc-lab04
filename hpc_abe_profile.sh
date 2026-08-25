#!/bin/bash
#HPC --partition=lab4
#HPC --cpu=60
#HPC --mem=100Gi
#HPC --time=30m
#HPC --name=abe-profile
#HPC --output=profile/hpc_%x_%j.log
#
# Profile the CPU ABE evolution without including TwoPuncture in the measured
# region. The default t=0..4 window covers several complete AMR/RK4 cycles.
set -euo pipefail

# hpc executes a submitted copy from /tmp but preserves the submission cwd.
ROOT_DIR="${AMSS_ROOT_DIR:-$PWD}"
if [[ ! -f "$ROOT_DIR/CMakeLists.txt" ]]; then
    echo "submit this script from the repository root" >&2
    exit 2
fi

# mpiexec uses this entry point for each rank so perf PIDs can be mapped back
# to MPI ranks. exec preserves the PID recorded in the map.
if [[ "${1:-}" == "--rank-worker" ]]; then
    RUN_DIR="$2"
    RANK_MAP="$3"
    cd "$RUN_DIR"
    rank="${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-unknown}}"
    printf '%s\t%s\t%s\n' "$rank" "$$" "$(taskset -pc $$ 2>/dev/null || true)" >> "$RANK_MAP"
    exec ./ABE < /dev/null
fi

cd "$ROOT_DIR"
mkdir -p profile

RUN_ID="${HPC_JOB_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$BASHPID}"
PROFILE_DIR="$ROOT_DIR/profile/abe-$RUN_ID"
BUILD_DIR="$PROFILE_DIR/build"
PREP_ROOT="$PROFILE_DIR/prepare"
CACHE_DIR="$ROOT_DIR/profile/twopuncture-cache"
mkdir -p "$PROFILE_DIR" "$PREP_ROOT" "$CACHE_DIR"

exec > >(tee "$PROFILE_DIR/job.log") 2>&1

RANKS="${ABE_PROFILE_RANKS:-1}"
OMP_THREADS="${ABE_PROFILE_OMP_THREADS:-30}"
OPENMP_ENABLE="${ABE_PROFILE_OPENMP:-ON}"
OMP_ONLY="${ABE_PROFILE_OMP_ONLY:-ON}"
OMP_DIAGNOSTICS="${ABE_PROFILE_OMP_DIAGNOSTICS:-${AMSS_OMP_DIAGNOSTICS:-ON}}"
BLOCK_SCHEDULE="${ABE_PROFILE_BLOCK_SCHEDULE:-${AMSS_OMP_BLOCK_SCHEDULE:-dynamic,1}}"
BLOCK_TARGET="${ABE_PROFILE_BLOCK_TARGET:-${AMSS_OMP_BLOCK_TARGET:-}}"
STATIC_BLOCK_TARGET="${ABE_PROFILE_STATIC_BLOCK_TARGET:-${AMSS_OMP_STATIC_BLOCK_TARGET:-}}"
MOVING_BLOCK_TARGET="${ABE_PROFILE_MOVING_BLOCK_TARGET:-${AMSS_OMP_MOVING_BLOCK_TARGET:-}}"
STATIC_THREADS="${ABE_PROFILE_STATIC_THREADS:-${AMSS_OMP_STATIC_THREADS:-}}"
MOVING_THREADS="${ABE_PROFILE_MOVING_THREADS:-${AMSS_OMP_MOVING_THREADS:-}}"
RHS_METRIC_FUSION="${AMSS_RHS_METRIC_FUSION:-ON}"
RHS_GAMMA_FUSION="${AMSS_RHS_GAMMA_FUSION:-ON}"
RHS_CONNECTION_FUSION="${AMSS_RHS_CONNECTION_FUSION:-OFF}"
RHS_AIJ_FUSION="${AMSS_RHS_AIJ_FUSION:-OFF}"
RHS_SANITY_CHECK="${AMSS_RHS_SANITY_CHECK:-OFF}"
DIRECT_SYNC="${AMSS_ENABLE_OMP_DIRECT_SYNC:-OFF}"
DIRECT_AMR="${AMSS_ENABLE_OMP_DIRECT_AMR_TRANSFER:-OFF}"
DIRECT_AMR_SPLIT="${AMSS_ENABLE_OMP_DIRECT_AMR_SPLIT:-OFF}"
LOPSIDEDIFF_BATCH="${AMSS_ENABLE_LOPSIDEDIFF_BATCH:-OFF}"
PROLONG3_SIMD="${AMSS_ENABLE_PROLONG3_SIMD:-OFF}"
PROLONG3_PAIRWISE="${AMSS_ENABLE_PROLONG3_PAIRWISE:-ON}"
LOPSIDEDIFF_GHOST_ONLY="${AMSS_ENABLE_LOPSIDEDIFF_GHOST_ONLY:-OFF}"
SYMMETRY_BD_NO_CLEAR="${AMSS_ENABLE_SYMMETRY_BD_NO_CLEAR:-ON}"
if [[ "$OMP_ONLY" == "ON" || "$OMP_ONLY" == "1" ]]; then
    RANKS=1
fi
if [[ -z "$STATIC_BLOCK_TARGET" ]]; then
    STATIC_BLOCK_TARGET=$((OMP_THREADS * 4 / 5))
    STATIC_BLOCK_TARGET=$((STATIC_BLOCK_TARGET > 0 ? STATIC_BLOCK_TARGET : 1))
fi
if [[ -z "$MOVING_BLOCK_TARGET" ]]; then
    MOVING_BLOCK_TARGET=$OMP_THREADS
fi
if [[ -z "$STATIC_THREADS" ]]; then
    STATIC_THREADS=$((OMP_THREADS * 4 / 5))
fi
if [[ -z "$MOVING_THREADS" ]]; then
    MOVING_THREADS=$OMP_THREADS
fi
EVOLVE_TIME="${ABE_PROFILE_TIME:-4.0}"
ABE_OPT="${AMSS_ABE_PROFILE_OPT:--O3 -g -fno-omit-frame-pointer -fopt-info-vec-optimized}"
ABE_ARCH="${AMSS_ABE_ARCH_FLAGS:-}"
TWOP_OPT="${AMSS_TWOPUNCTURE_OPT:--O3}"
TWOP_ARCH="${AMSS_TWOPUNCTURE_ARCH_FLAGS:--march=native}"

echo "profile directory: $PROFILE_DIR"
echo "git revision: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "MPI ranks: $RANKS"
echo "OpenMP enabled: $OPENMP_ENABLE"
echo "OpenMP threads: $OMP_THREADS"
echo "OpenMP-only ABE: $OMP_ONLY"
echo "OpenMP diagnostics: $OMP_DIAGNOSTICS"
echo "OpenMP block schedule: $BLOCK_SCHEDULE"
echo "OpenMP block target: ${BLOCK_TARGET:-thread-count default}"
echo "OpenMP static block target: ${STATIC_BLOCK_TARGET:-global/default}"
echo "OpenMP moving block target: ${MOVING_BLOCK_TARGET:-global/default}"
echo "OpenMP static threads: ${STATIC_THREADS:-global default}"
echo "OpenMP moving threads: ${MOVING_THREADS:-global default}"
echo "profile evolution interval: t=0..$EVOLVE_TIME"
echo "ABE flags: $ABE_OPT $ABE_ARCH"
echo "TwoPuncture flags: $TWOP_OPT $TWOP_ARCH"
echo "RHS metric fusion: $RHS_METRIC_FUSION"
echo "RHS Gamma fusion: $RHS_GAMMA_FUSION"
echo "RHS connection fusion: $RHS_CONNECTION_FUSION"
echo "RHS Aij fusion: $RHS_AIJ_FUSION"
echo "RHS sanity check: $RHS_SANITY_CHECK"
echo "Direct same-level Sync: $DIRECT_SYNC"
echo "Direct AMR transfer: $DIRECT_AMR"
echo "Direct AMR split: $DIRECT_AMR_SPLIT"
echo "symmetry_bd full clear: $SYMMETRY_BD_NO_CLEAR"
echo "Lopsided batch: $LOPSIDEDIFF_BATCH"
echo "prolong3 SIMD: $PROLONG3_SIMD"
echo "prolong3 pairwise: $PROLONG3_PAIRWISE"
echo "Lopsided ghost-only: $LOPSIDEDIFF_GHOST_ONLY"

"$ROOT_DIR/scripts/collect_cpu_info.sh" > "$PROFILE_DIR/cpu-info.txt"

export JOBS="$(nproc)"
export OMP_NUM_THREADS="$OMP_THREADS"
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_SCHEDULE="$BLOCK_SCHEDULE"
if [[ -n "$BLOCK_TARGET" ]]; then
    export AMSS_OMP_BLOCK_TARGET="$BLOCK_TARGET"
else
    unset AMSS_OMP_BLOCK_TARGET
fi
if [[ -n "$STATIC_BLOCK_TARGET" ]]; then
    export AMSS_OMP_STATIC_BLOCK_TARGET="$STATIC_BLOCK_TARGET"
else
    unset AMSS_OMP_STATIC_BLOCK_TARGET
fi
if [[ -n "$MOVING_BLOCK_TARGET" ]]; then
    export AMSS_OMP_MOVING_BLOCK_TARGET="$MOVING_BLOCK_TARGET"
else
    unset AMSS_OMP_MOVING_BLOCK_TARGET
fi
if [[ -n "$STATIC_THREADS" ]]; then
    export AMSS_OMP_STATIC_THREADS="$STATIC_THREADS"
else
    unset AMSS_OMP_STATIC_THREADS
fi
if [[ -n "$MOVING_THREADS" ]]; then
    export AMSS_OMP_MOVING_THREADS="$MOVING_THREADS"
else
    unset AMSS_OMP_MOVING_THREADS
fi
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
export AMSS_BUILD_DIR="$BUILD_DIR"

echo "=== Build with symbols ==="
set +e
./compile.sh \
    -DAMSS_ENABLE_GPU=OFF \
    -DAMSS_ENABLE_OPENMP="$OPENMP_ENABLE" \
    -DAMSS_ENABLE_OMP_ONLY="$OMP_ONLY" \
    -DAMSS_ENABLE_OMP_DIAGNOSTICS="$OMP_DIAGNOSTICS" \
    -DAMSS_ENABLE_TWOPUNCTURE_OPENMP=ON \
    -DAMSS_OPT="$ABE_OPT" \
    -DAMSS_ARCH_FLAGS="$ABE_ARCH" \
    -DAMSS_TWOPUNCTURE_OPT="$TWOP_OPT" \
    -DAMSS_TWOPUNCTURE_ARCH_FLAGS="$TWOP_ARCH" \
    -DAMSS_ENABLE_RHS_METRIC_FUSION="$RHS_METRIC_FUSION" \
    -DAMSS_ENABLE_OMP_DIRECT_AMR_SPLIT="$DIRECT_AMR_SPLIT" \
    -DAMSS_ENABLE_RHS_GAMMA_FUSION="$RHS_GAMMA_FUSION" \
    -DAMSS_ENABLE_RHS_CONNECTION_FUSION="$RHS_CONNECTION_FUSION" \
    -DAMSS_ENABLE_RHS_AIJ_FUSION="$RHS_AIJ_FUSION" \
    -DAMSS_ENABLE_OMP_DIRECT_AMR_TRANSFER="$DIRECT_AMR" \
    -DAMSS_ENABLE_RHS_SANITY_CHECK="$RHS_SANITY_CHECK" \
    -DAMSS_ENABLE_SYMMETRY_BD_NO_CLEAR="$SYMMETRY_BD_NO_CLEAR" \
    -DAMSS_ENABLE_OMP_DIRECT_SYNC="$DIRECT_SYNC" \
    -DAMSS_ENABLE_LOPSIDEDIFF_BATCH="$LOPSIDEDIFF_BATCH" \
    -DAMSS_ENABLE_LOPSIDEDIFF_GHOST_ONLY="$LOPSIDEDIFF_GHOST_ONLY" \
    -DAMSS_ENABLE_PROLONG3_SIMD="$PROLONG3_SIMD" \
    -DAMSS_ENABLE_PROLONG3_PAIRWISE="$PROLONG3_PAIRWISE" \
    2>&1 | tee "$PROFILE_DIR/build.log"
build_rc=${PIPESTATUS[0]}
set -e
if (( build_rc != 0 )); then
    exit "$build_rc"
fi

echo "=== Prepare fixed course input ==="
export AMSS_OUTPUT_ROOT="$PREP_ROOT"
export AMSS_CACHE_DIR="$CACHE_DIR"
export AMSS_NCKU_PREPARE_ONLY=1
export AMSS_MPIEXEC="mpiexec --allow-run-as-root --map-by core --bind-to core"
./run.sh --twop-cache
unset AMSS_NCKU_PREPARE_ONLY

BASE_DIR="$PREP_ROOT/GW250118/AMSS_NCKU_output"
if [[ ! -f "$BASE_DIR/input.par" || ! -f "$BASE_DIR/Ansorg.psid" ]]; then
    echo "prepared ABE input is incomplete: $BASE_DIR" >&2
    exit 1
fi

stage_run() {
    local target="$1"
    mkdir -p "$target/binary_output"
    cp "$BUILD_DIR/ABE" "$target/ABE"
    cp "$BASE_DIR/Ansorg.psid" "$target/Ansorg.psid"
    sed -E \
        "s/^(ABE::TotalTime[[:space:]]*=[[:space:]]*).*/\\1$EVOLVE_TIME/" \
        "$BASE_DIR/input.par" > "$target/input.par"
}

STAT_RUN="$PROFILE_DIR/stat-run"
RECORD_RUN="$PROFILE_DIR/record-run"
stage_run "$STAT_RUN"
stage_run "$RECORD_RUN"

if [[ "$OMP_ONLY" == "ON" || "$OMP_ONLY" == "1" ]]; then
    MPI_CMD=()
else
    MPI_CMD=(mpiexec --allow-run-as-root --map-by "ppr:${RANKS}:node:PE=${OMP_THREADS}" --bind-to core --report-bindings -n "$RANKS")
fi

echo "=== perf stat ==="
: > "$PROFILE_DIR/stat-rank-pids.tsv"
set +e
perf stat -d -d -o "$PROFILE_DIR/perf-stat.txt" -- \
    "${MPI_CMD[@]}" "$ROOT_DIR/hpc_abe_profile.sh" --rank-worker \
    "$STAT_RUN" "$PROFILE_DIR/stat-rank-pids.tsv" \
    2>&1 | tee "$PROFILE_DIR/stat-run.log"
stat_rc=${PIPESTATUS[0]}
set -e
if (( stat_rc != 0 )); then
    exit "$stat_rc"
fi

echo "=== perf record ==="
: > "$PROFILE_DIR/record-rank-pids.tsv"
set +e
perf record -m 1 -F 99 --call-graph fp -o "$PROFILE_DIR/perf.data" -- \
    "${MPI_CMD[@]}" "$ROOT_DIR/hpc_abe_profile.sh" --rank-worker \
    "$RECORD_RUN" "$PROFILE_DIR/record-rank-pids.tsv" \
    2>&1 | tee "$PROFILE_DIR/record-run.log"
record_rc=${PIPESTATUS[0]}
set -e
if (( record_rc != 0 )); then
    exit "$record_rc"
fi

echo "=== Reports ==="
REPORT_WARNINGS="$PROFILE_DIR/perf-report-warnings.log"
: > "$REPORT_WARNINGS"

perf report --stdio --no-children --no-call-graph --sort comm,dso,symbol \
    --percent-limit 0.1 -i "$PROFILE_DIR/perf.data" \
    > "$PROFILE_DIR/perf-report-flat.txt" 2>> "$REPORT_WARNINGS"
perf report --stdio --children --sort comm,dso,symbol \
    --percent-limit 0.2 -i "$PROFILE_DIR/perf.data" \
    > "$PROFILE_DIR/perf-report-callgraph.txt" 2>> "$REPORT_WARNINGS"
perf report --stdio --no-children --no-call-graph --sort comm,dso,symbol,srcline \
    --percent-limit 0.1 -i "$PROFILE_DIR/perf.data" \
    > "$PROFILE_DIR/perf-report-lines.txt" 2>> "$REPORT_WARNINGS"
perf report --stdio --no-children --no-call-graph --sort pid,comm,dso,symbol \
    --percent-limit 0.05 -i "$PROFILE_DIR/perf.data" \
    > "$PROFILE_DIR/perf-report-pids.txt" 2>> "$REPORT_WARNINGS"
perf report --stdio --no-children --no-call-graph --sort dso \
    --percent-limit 0.05 -i "$PROFILE_DIR/perf.data" \
    > "$PROFILE_DIR/perf-report-dso.txt" 2>> "$REPORT_WARNINGS"
perf report --stdio --no-children --no-call-graph --field-separator ';' \
    --fields overhead,pid,dso --sort pid,dso --percent-limit 0 \
    -i "$PROFILE_DIR/perf.data" \
    > "$PROFILE_DIR/perf-report-rank-dso.csv" 2>> "$REPORT_WARNINGS"

{
    echo "=== perf stat pass ==="
    grep -E "Before Evolve|Timestep|Total Evolve|Total Running" \
        "$PROFILE_DIR/stat-run.log" || true
    echo "=== perf record pass ==="
    grep -E "Before Evolve|Timestep|Total Evolve|Total Running" \
        "$PROFILE_DIR/record-run.log" || true
} > "$PROFILE_DIR/timing-summary.txt"

{
    for output in bssn_ADMQs.dat bssn_BH.dat bssn_constraint.dat bssn_psi4.dat; do
        stat_output="$STAT_RUN/binary_output/$output"
        record_output="$RECORD_RUN/binary_output/$output"
        if cmp -s <(tail -n +3 "$stat_output") <(tail -n +3 "$record_output"); then
            echo "$output: numerical rows identical"
        else
            echo "$output: DIFFERENT"
        fi
    done
} > "$PROFILE_DIR/numerical-reproducibility.txt"

sort -n "$PROFILE_DIR/stat-rank-pids.tsv" -o "$PROFILE_DIR/stat-rank-pids.tsv"
sort -n "$PROFILE_DIR/record-rank-pids.tsv" -o "$PROFILE_DIR/record-rank-pids.tsv"

echo "profile complete: $PROFILE_DIR"
