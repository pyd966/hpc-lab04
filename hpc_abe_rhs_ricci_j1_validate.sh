#!/bin/bash
#HPC --partition=lab4
#HPC --cpu=60
#HPC --mem=100Gi
#HPC --time=30m
#HPC --name=abe-ricci-j1-validate
#HPC --output=profile/hpc_%x_%j.log
set -euo pipefail

ROOT_DIR="${AMSS_ROOT_DIR:-$PWD}"
export ABE_RICCI_BUILD_SET="off j1"
export ABE_RICCI_RUN_ORDER="off j1 j1 off off j1"
exec bash "$ROOT_DIR/hpc_abe_rhs_ricci_tile_sweep.sh"
