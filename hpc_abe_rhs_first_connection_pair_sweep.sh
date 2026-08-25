#!/bin/bash
#HPC --partition=lab4
#HPC --cpu=60
#HPC --mem=100Gi
#HPC --time=30m
#HPC --name=abe-rhs-first-pair
#HPC --output=profile/hpc_%x_%j.log
set -euo pipefail
export ABE_FIRST_CONNECTION_MODE=pair
exec "${AMSS_ROOT_DIR:-$PWD}/hpc_abe_rhs_first_connection_sweep.sh"
