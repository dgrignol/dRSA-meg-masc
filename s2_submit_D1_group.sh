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

python D1_group_cluster_analysis.py \
  --subjects $(seq -w 1 27) \
  --models "Envelope" "Phoneme Voicing" "Word Frequency" "GloVe" "GloVe Norm" \
  --results-dir results \
  --lag-metric correlation