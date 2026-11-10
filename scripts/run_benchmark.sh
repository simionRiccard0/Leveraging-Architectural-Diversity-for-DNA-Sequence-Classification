#!/bin/bash
# ======================================================================
# Train + evaluate one architecture over the benchmark grid for one seed.
#
# Usage (from anywhere):
#   ARCH=genexplorer SEED=2020 bash scripts/run_benchmark.sh
#
# Environment variables
#   ARCH      fulle | genexplorer | nucleictransformer | dnabert2camp | helixpr   (required)
#   SEED      run seed (default 2020); also sets the training/validation
#             partition of ECO, ENH and OOD
#   TASKS     space-separated subset of: viral ecoli enhancer anellovirus  (default: all)
#   CONFIGS   space-separated subset of: A B C D  (default: all; HelixPR: A)
#   DATA_DIR  dataset root (default: ./datasets)
#   OUT_DIR   where checkpoints and logits are written (default: repository root)
#   PYTHON    Python interpreter (default: python)
#   DNABERT2_PYTHON  interpreter for DNABERT2-CAMP, e.g. a container wrapper:
#             "singularity exec --nv -B /data:/data dnabert2.sif python"  (default: $PYTHON)
#   EXTRA_TRAIN_ARGS  extra arguments passed to the training script (e.g. "--epochs 2")
#
# For every (task, config) the script
#   1. skips the pair if all validation and test logits already exist;
#   2. moves aside a checkpoint without its .done marker (interrupted run);
#   3. trains (writing the .done marker only on success);
#   4. evaluates on validation and on every test set of the task
#      (for viral: VM test + the 15 SIM sets).
# ======================================================================
set -Eeuo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

source "$REPO_DIR/scripts/bench_common.sh"

ARCH="${ARCH:?set ARCH (fulle | genexplorer | nucleictransformer | dnabert2camp | helixpr)}"
SEED="${SEED:-2020}"
DATA_DIR="${DATA_DIR:-./datasets}"
PYTHON="${PYTHON:-python}"
DNABERT2_PYTHON="${DNABERT2_PYTHON:-$PYTHON}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"
read -r -a TASK_LIST <<< "${TASKS:-viral ecoli enhancer anellovirus}"

PY="$PYTHON"
BATCH_ARGS=1
case "$ARCH" in
    fulle)              TRAIN_SCRIPT="pipelines/train_fulle.py";               BASE_BS=32 ;;
    genexplorer)        TRAIN_SCRIPT="pipelines/train_genexplorer.py";         BASE_BS=32 ;;
    nucleictransformer) TRAIN_SCRIPT="pipelines/train_nucleic_transformer.py"; BASE_BS=64 ;;
    dnabert2camp)       TRAIN_SCRIPT="pipelines/train_dnabert2_camp.py";       BASE_BS=32; PY="$DNABERT2_PYTHON" ;;
    helixpr)            TRAIN_SCRIPT="pipelines/train_helixpr.py";             BASE_BS=64; BATCH_ARGS=0 ;;
    *) echo "[ERROR] unknown ARCH: $ARCH" >&2; exit 1 ;;
esac

if [ "$ARCH" == "helixpr" ]; then
    read -r -a CONFIG_LIST <<< "${CONFIGS:-A}"
else
    read -r -a CONFIG_LIST <<< "${CONFIGS:-A B C D}"
fi

mkdir -p "$(ckpt_dir "$SEED")" "$(logits_dir "$SEED")" "$(val_logits_dir "$SEED")"

echo "======================================================================"
echo "BENCHMARK | ARCH=$ARCH | SEED=$SEED | TASKS=${TASK_LIST[*]} | CONFIGS=${CONFIG_LIST[*]}"
echo "DATA_DIR=$DATA_DIR | OUT_DIR=$OUT_DIR"
echo "======================================================================"

for TASK in "${TASK_LIST[@]}"; do
    # The small balanced benchmarks use twice the batch size.
    if [ "$TASK" == "ecoli" ] || [ "$TASK" == "enhancer" ]; then
        BS=$((BASE_BS * 2))
    else
        BS=$BASE_BS
    fi
    BS_OPT=()
    if [ "$BATCH_ARGS" -eq 1 ]; then BS_OPT=(--batch_size "$BS"); fi

    for CFG in "${CONFIG_LIST[@]}"; do
        CKPT=$(ckpt_file "$ARCH" "$SEED" "$TASK" "$CFG")

        if is_complete "$ARCH" "$SEED" "$TASK" "$CFG"; then
            echo "[SKIP] $ARCH | $TASK | $CFG | seed $SEED (complete)"
            continue
        fi

        if [ -f "$CKPT" ] && [ ! -f "${CKPT}.done" ]; then
            BAK="${CKPT}.partial_$(date +%Y%m%d_%H%M%S)"
            echo "[WARN] $CKPT has no .done marker (interrupted run): moved to $BAK"
            mv "$CKPT" "$BAK"
        fi

        if [ ! -f "$CKPT" ]; then
            echo "[$(date '+%F %T')] TRAIN | $ARCH | $TASK | $CFG | seed $SEED"
            # shellcheck disable=SC2086
            $PY "$TRAIN_SCRIPT" \
                --config "$CFG" \
                --task "$TASK" \
                --seed "$SEED" \
                --data_folder "$DATA_DIR" \
                --checkpoint_dir "$(ckpt_dir "$SEED")" \
                ${BS_OPT[@]+"${BS_OPT[@]}"} $EXTRA_TRAIN_ARGS
            touch "${CKPT}.done"
        fi

        echo "[$(date '+%F %T')] EVAL  | $ARCH | $TASK | $CFG | seed $SEED"
        # shellcheck disable=SC2086
        $PY evaluation/evaluate.py \
            --arch "$ARCH" \
            --task "$TASK" \
            --config "$CFG" \
            --checkpoint "$CKPT" \
            --seed "$SEED" \
            --data_folder "$DATA_DIR" \
            --logits_dir "$(logits_dir "$SEED")" \
            --val_logits_dir "$(val_logits_dir "$SEED")" \
            --batch_size "$BS"

        if ! is_complete "$ARCH" "$SEED" "$TASK" "$CFG"; then
            echo "[ERROR] missing outputs after evaluation:" >&2
            missing_outputs "$ARCH" "$SEED" "$TASK" "$CFG" | sed 's/^/    /' >&2
            exit 1
        fi
    done
done

echo "[✓] Benchmark completed: $ARCH, seed $SEED"
