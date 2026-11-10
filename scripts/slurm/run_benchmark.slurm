#!/bin/bash
# ======================================================================
# Slurm template for scripts/run_benchmark.sh (one architecture, one seed).
# Adapt the #SBATCH lines and the variables below to your cluster, then:
#
#   sbatch --export=ALL,ARCH=genexplorer,SEED=2020 scripts/slurm/run_benchmark.slurm
#   sbatch --export=ALL,ARCH=dnabert2camp,SEED=1969,TASKS=ecoli,CONFIGS=A scripts/slurm/run_benchmark.slurm
#
# Submit from the repository root (logs/ must exist: mkdir -p logs).
# ======================================================================
#SBATCH --job-name=dna_bench
#SBATCH --partition=<partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/bench_%x_%j.out
#SBATCH --error=logs/bench_%x_%j.err

set -Eeuo pipefail

# ---- settings to adapt -------------------------------------------------
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
export DATA_DIR="${DATA_DIR:-$REPO_DIR/datasets}"
export OUT_DIR="${OUT_DIR:-$REPO_DIR}"
export PYTHON="${PYTHON:-python}"            # e.g. $HOME/miniconda3/envs/<env>/bin/python

# Optional container for DNABERT2-CAMP (transformers + triton). Leave empty to use $PYTHON.
DNABERT2_SIF="${DNABERT2_SIF:-}"
if [ -n "$DNABERT2_SIF" ]; then
    export DNABERT2_PYTHON="singularity exec --nv -B $REPO_DIR:$REPO_DIR -B $DATA_DIR:$DATA_DIR $DNABERT2_SIF python"
fi
# Optional local copy of the DNABERT-2 weights (otherwise downloaded from Hugging Face).
# export DNABERT2_MODEL_PATH=/path/to/DNABERT-2-117M
# ------------------------------------------------------------------------

echo "[i] node: $(hostname) | ARCH=${ARCH:-?} | SEED=${SEED:-2020}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

bash "$REPO_DIR/scripts/run_benchmark.sh"
