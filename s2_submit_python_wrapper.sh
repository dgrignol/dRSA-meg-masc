#!/bin/bash
# ===== SGE directives =====
#$ -N dRSA_py_full
#$ -S /bin/bash
#$ -cwd
#$ -o logs/$JOB_NAME.$JOB_ID.out
#$ -e logs/$JOB_NAME.$JOB_ID.err
#$ -j y
# Resources (tune as needed)
#$ -pe smp 8
#$ -l h_vmem=24G

set -euo pipefail

# --- Paths (edit WD to your repo root that contains pipeline_wrapper.py)
WD="/mnt/storage/tier2/morwur/Projects/DAMIANO/SpeDiction/meg-masc"
GLOVE="/mnt/storage/tier2/morwur/Projects/DAMIANO/SpeDiction/meg-masc/derivatives/Models/glove/glove.6B.300d.txt"

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

# --- Run your exact command
python pipeline_wrapper.py \
  --subjects 2-23 \
  --glove-path "$GLOVE"