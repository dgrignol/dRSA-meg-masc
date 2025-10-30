#!/bin/bash
#$ -N dRSA_group
#$ -S /bin/bash
#$ -cwd
#$ -o logs/$JOB_NAME.$JOB_ID.out
#$ -e logs/$JOB_NAME.$JOB_ID.err
#$ -j y
#$ -pe smp 4
#$ -l h_vmem=24G

set -euo pipefail

WD="/mnt/storage/tier2/morwur/Projects/DAMIANO/SpeDiction/meg-masc"

mkdir -p logs
cd "$WD"

export MAMBA_ROOT_PREFIX="$HOME/micromamba"
eval "$($HOME/bin/micromamba shell hook -s bash)"
micromamba activate drsa311
export PYTHONNOUSERSITE=1

python -V
which python

if [[ -n "${ANALYSIS_NAME:-}" ]]; then
  analysis_name="$ANALYSIS_NAME"
  echo ">>> Using analysis name from \$ANALYSIS_NAME: ${analysis_name}"
else
  if ! analysis_name=$(python - <<'PY'
from functions.generic_helpers import find_latest_analysis_directory
from pathlib import Path
import sys

root = Path("$WD") / "results"
latest = find_latest_analysis_directory(root)
if latest is None:
    sys.exit(1)
print(latest.name)
PY
  ); then
    echo "!!! Unable to determine the latest analysis directory under $WD/results" >&2
    echo "!!! Set ANALYSIS_NAME or pass --analysis-name manually." >&2
    exit 1
  fi
  echo ">>> Defaulting to latest analysis: ${analysis_name}"
fi

python D1_group_cluster_analysis.py \
  --subjects $(seq -w 1 27) \
  --models "Envelope" "Phoneme Voicing" "Word Frequency" "GloVe" "GloVe Norm" \
  --analysis-name "${analysis_name}" \
  --results-root results \
  --lag-metric correlation \
  "$@"
