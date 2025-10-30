#!/bin/bash
#$ -N C1_dRSA
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

# Allow users to separate qsub options from script args via `--`.
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
    analysis_name="drsa_${JOB_ID}"
    echo ">>> Defaulting analysis name to ${analysis_name}"
  fi
  extra_args+=("--analysis-name" "${analysis_name}")
fi

subject=$(printf "%02d" "$SGE_TASK_ID")
subject_label="sub-${subject}"
echo ">>> Running C1_dRSA for ${subject_label}"
python C1_dRSA_run.py \
  "${subject_label}" \
  "${extra_args[@]}"
