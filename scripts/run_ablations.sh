#!/bin/bash
# ======================================================================
# Ablations and additional baselines of the paper.
#
# Usage:
#   bash scripts/run_ablations.sh c1        # preliminary C1 ablation on ECO/ENH: RC and focal alpha
#   bash scripts/run_ablations.sh get       # GenExplorer without TransformerBranch (GE-T), config C1
#   bash scripts/run_ablations.sh classical # k-mer + Logistic Regression / Random Forest
#
# Same environment variables as scripts/run_benchmark.sh
# (SEED, DATA_DIR, OUT_DIR, PYTHON, DNABERT2_PYTHON).
# ======================================================================
set -Eeuo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

SEED="${SEED:-2020}"
DATA_DIR="${DATA_DIR:-./datasets}"
OUT_DIR="${OUT_DIR:-.}"
PYTHON="${PYTHON:-python}"
DNABERT2_PYTHON="${DNABERT2_PYTHON:-$PYTHON}"
WHAT="${1:?usage: run_ablations.sh c1 | get | classical}"

train_script() {
    case "$1" in
        fulle)              echo "pipelines/train_fulle.py" ;;
        genexplorer)        echo "pipelines/train_genexplorer.py" ;;
        nucleictransformer) echo "pipelines/train_nucleic_transformer.py" ;;
        dnabert2camp)       echo "pipelines/train_dnabert2_camp.py" ;;
    esac
}

# One C1 run (config A) of one architecture on one task, with extra training options.
#   $1 variant name  $2 arch  $3 task  $4.. extra training arguments
run_c1_variant() {
    local variant=$1 arch=$2 task=$3; shift 3
    local base="${OUT_DIR}/ablation_c1/${variant}"
    local ckdir="${base}/checkpoints_seed_${SEED}"
    local py="$PYTHON" bs=64
    [ "$arch" == "nucleictransformer" ] && bs=128
    [ "$arch" == "dnabert2camp" ] && py="$DNABERT2_PYTHON"
    mkdir -p "$ckdir" "${base}/global_logits_${SEED}" "${base}/global_logits_val_${SEED}"

    echo "[$(date '+%F %T')] C1 ablation | ${variant} | ${arch} | ${task} | seed ${SEED}"
    # shellcheck disable=SC2086
    $py "$(train_script "$arch")" --config A --task "$task" --seed "$SEED" \
        --data_folder "$DATA_DIR" --checkpoint_dir "$ckdir" --batch_size "$bs" "$@"
    # shellcheck disable=SC2086
    $py evaluation/evaluate.py --arch "$arch" --task "$task" --config A \
        --checkpoint "${ckdir}/best_${arch}_${task}_A.pt" --seed "$SEED" --data_folder "$DATA_DIR" \
        --logits_dir "${base}/global_logits_${SEED}" --val_logits_dir "${base}/global_logits_val_${SEED}" \
        --batch_size "$bs"
}

case "$WHAT" in
    c1)
        # (i) original C1 with RC, (ii) without RC, (iii) symmetric focal weighting alpha=0.5.
        # Without RC is the default on ECO (and the setting of the benchmark grid);
        # the Nucleic Transformer already uses alpha=0.5 on balanced data, so (iii) is not run for it.
        for TASK in ecoli enhancer; do
            for ARCH in fulle genexplorer nucleictransformer dnabert2camp; do
                run_c1_variant rc_on  "$ARCH" "$TASK" --rc on
                run_c1_variant rc_off "$ARCH" "$TASK" --rc off
                if [ "$ARCH" != "nucleictransformer" ]; then
                    run_c1_variant alpha05 "$ARCH" "$TASK" --focal_alpha 0.5
                fi
            done
        done
        ;;
    get)
        # GE-T = Pattern + Frequency branches (FP); the full model is FPT.
        for TASK in viral ecoli enhancer anellovirus; do
            $PYTHON pipelines/train_ge_ablation.py --task "$TASK" --branches FP --seed "$SEED" \
                --data_folder "$DATA_DIR" \
                --checkpoint_dir "${OUT_DIR}/ablation_get/checkpoints_seed_${SEED}" \
                --logits_dir "${OUT_DIR}/ablation_get/global_logits_${SEED}"
        done
        ;;
    classical)
        $PYTHON pipelines/train_classical_baselines.py --data_folder "$DATA_DIR" --seed "$SEED"
        ;;
    *)
        echo "usage: run_ablations.sh c1 | get | classical" >&2; exit 1 ;;
esac
