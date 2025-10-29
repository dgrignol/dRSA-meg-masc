#!/bin/bash
#$ -N A3_resample
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

subject=$(printf "%02d" "$SGE_TASK_ID")
subject_label="sub-${subject}"
echo ">>> Resampling concatenated data for ${subject_label}"
python A3_resample_concatenated_data.py \
  --subject "${subject_label}" \
  "$@"
