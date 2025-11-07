#!/bin/bash
#$ -N B6_gpt_next
#$ -S /bin/bash
#$ -cwd
#$ -o logs/$JOB_NAME.$JOB_ID.$TASK_ID.out
#$ -e logs/$JOB_NAME.$JOB_ID.$TASK_ID.err
#$ -j y
#$ -pe smp 4
#$ -l h_vmem=24G
#$ -t 1-27

set -euo pipefail

# Adjust to the absolute path of this repo on the cluster filesystem
WD="/mnt/storage/tier2/morwur/Projects/DAMIANO/SpeDiction/meg-masc"

mkdir -p logs
cd "$WD"

# Activate cluster environment
export MAMBA_ROOT_PREFIX="$HOME/micromamba"
eval "$($HOME/bin/micromamba shell hook -s bash)"
micromamba activate drsa311
export PYTHONNOUSERSITE=1
# Limit BLAS threads to reduce contention/memory
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}

python -V
which python

# Preflight dependency check (fail fast with guidance)
python - <<'PY'
import sys
mods = ["torch", "transformers", "numpy", "scipy", "sklearn"]
missing = []
for m in mods:
    try:
        __import__(m)
    except Exception:
        missing.append(m)
if missing:
    print("ERROR: Missing Python packages:", ", ".join(missing))
    sys.exit(1)
print("All required Python packages found.")
PY
if [[ $? -ne 0 ]]; then
  echo "\nInstall them once into the micromamba env, e.g.:" >&2
  echo "  micromamba activate drsa311" >&2
  echo "  micromamba install -y pytorch cpuonly -c pytorch -c conda-forge" >&2
  echo "  micromamba install -y transformers scikit-learn scipy numpy -c conda-forge" >&2
  exit 1
fi

# Allow users to separate qsub options from script args via `--`.
if [[ "${1:-}" == "--" ]]; then
  shift
fi

subject=$(printf "%02d" "$SGE_TASK_ID")
subject_label="sub-${subject}"
echo ">>> Running B6 logprob-SVD for ${subject_label}"

# Ensure global SVD basis is created by the first task; others will wait until it exists.
GLOBAL_SVD_PATH="${WD}/derivatives/Models/gpt_next/global_svd_basis.pkl"

if [[ "$SGE_TASK_ID" -eq 1 ]]; then
  echo ">>> Fitting/using global SVD basis and building story caches (task 1)."
  python B6_gpt_next_logprob_svd.py \
    --subjects "${subject_label}" \
    --hf-model derivatives/Models/gpt2 \
    --context-tokens 512 \
    --components 64 \
    --topk 4096 --topk-mass 0.99 \
    --svd-fit-sample 100000 \
    --plot \
    --overwrite \
    "$@"
else
  echo ">>> Waiting for global SVD basis to be created at: ${GLOBAL_SVD_PATH}"
  for i in $(seq 1 120); do
    if [[ -f "${GLOBAL_SVD_PATH}" ]]; then
      echo ">>> Global SVD basis found. Proceeding."
      break
    fi
    sleep 10
  done
  if [[ ! -f "${GLOBAL_SVD_PATH}" ]]; then
    echo "ERROR: Global SVD basis not found after waiting. Exiting." >&2
    exit 1
  fi

  # Optional: wait for story caches (task-0..task-3) to be present to avoid partial reads
  # Adjust the list if your task labels differ.
  for story in task-0 task-1 task-2 task-3; do
    CACHE_FILE="${WD}/derivatives/Models/gpt_next/story_cache/${story}/reduced_tokens.npy"
    echo ">>> Waiting for cache: ${CACHE_FILE}"
    for i in $(seq 1 120); do
      if [[ -f "${CACHE_FILE}" ]]; then
        echo ">>> Cache ready: ${CACHE_FILE}"
        break
      fi
      sleep 5
    done
  done

  python B6_gpt_next_logprob_svd.py \
    --subjects "${subject_label}" \
    --hf-model derivatives/Models/gpt2 \
    --context-tokens 512 \
    --components 64 \
    --plot \
    --overwrite \
    "$@"
fi
