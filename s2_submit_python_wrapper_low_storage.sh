#!/bin/bash
# ===== SGE directives =====
#$ -N dRSA_py_low
#$ -S /bin/bash
#$ -cwd
#$ -o logs/$JOB_NAME.$JOB_ID.out
#$ -e logs/$JOB_NAME.$JOB_ID.err
#$ -j y
# Resources (adjust per cluster policy)
#$ -pe smp 8
#$ -l h_vmem=24G

set -euo pipefail

# --- Paths (edit WD to point at the repository root)
WD="/mnt/storage/tier2/morwur/Projects/DAMIANO/SpeDiction/meg-masc"
GLOVE="$WD/derivatives/Models/glove/glove.6B.300d.txt"

mkdir -p logs
cd "$WD"

# --- Activate micromamba env (no ~/.bashrc dependence)
export MAMBA_ROOT_PREFIX="$HOME/micromamba"
eval "$($HOME/bin/micromamba shell hook -s bash)"
micromamba activate drsa311
export PYTHONNOUSERSITE=1

# --- Sanity prints (optional)
python -V
which python

# --- Run the low-storage wrapper
python pipeline_wrapper_low_storage.py \
  --subjects 1-23 \
  --glove-path "$GLOVE" \
  --continue-on-error \
  --keep-reports
