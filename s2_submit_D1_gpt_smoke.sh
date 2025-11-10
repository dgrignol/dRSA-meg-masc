#!/bin/bash
#$ -N D1_gpt_smoke
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

# Resolve analysis name
analysis_name="${ANALYSIS_NAME:-gpt_smoke}"
echo ">>> Using analysis name: ${analysis_name}"

# Resolve subjects list (space-separated; default 01..27)
if [[ -n "${SUBJECTS:-}" ]]; then
  subjects=( ${SUBJECTS} )
else
  mapfile -t subjects < <(seq -w 1 27)
fi
echo ">>> Subjects: ${subjects[*]}"

# Run D1 for GPT-only models
python D1_group_cluster_analysis.py \
  --analysis-name "${analysis_name}" \
  --subjects "${subjects[@]}" \
  --models "GPT Next-Token" "GPT Surprisal" \
  "$@"

