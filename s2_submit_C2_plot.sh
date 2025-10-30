#!/bin/bash
#$ -N C2_plot
#$ -S /bin/bash
#$ -cwd
#$ -o logs/$JOB_NAME.$JOB_ID.$TASK_ID.out
#$ -e logs/$JOB_NAME.$JOB_ID.$TASK_ID.err
#$ -j y
#$ -pe smp 4
#$ -l h_vmem=24G
#$ -t 1-27

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

if [[ "${1:-}" == "--" ]]; then
  shift
fi

extra_args=("$@")
analysis_name_arg_present=false
for arg in "${extra_args[@]}"; do
  if [[ "$arg" == "--analysis-name" ]]; then
    analysis_name_arg_present=true
    break
  fi
done

if [[ "$analysis_name_arg_present" == false ]]; then
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
      echo "!!! Set ANALYSIS_NAME or pass --analysis-name explicitly." >&2
      exit 1
    fi
    echo ">>> Defaulting to latest analysis: ${analysis_name}"
  fi
  extra_args+=("--analysis-name" "${analysis_name}")
fi

subject=$(printf "%02d" "$SGE_TASK_ID")
subject_label="sub-${subject}"
echo ">>> Rendering dRSA plot for ${subject_label}"
python C2_plot_dRSA.py \
  "${subject_label}" \
  "${extra_args[@]}"
