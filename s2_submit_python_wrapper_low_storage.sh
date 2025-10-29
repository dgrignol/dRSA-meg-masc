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

analysis_name="${ANALYSIS_NAME:-drsa_${JOB_ID}}"
if [[ -n "${ANALYSIS_NAME:-}" ]]; then
  echo ">>> Using analysis name from \$ANALYSIS_NAME: ${analysis_name}"
else
  echo ">>> Defaulting analysis name to ${analysis_name}"
fi

# --- Run the low-storage wrapper
python pipeline_wrapper_low_storage.py \
  --subjects 1-27 \
  --glove-path "$GLOVE" \
  --analysis-name "$analysis_name" \
  --continue-on-error \
  --keep-reports \
  "$@"
