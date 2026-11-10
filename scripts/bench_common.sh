#!/bin/bash
# ======================================================================
# Shared helpers of the benchmark launchers (sourced, not executed).
# Output layout, relative to $OUT_DIR (default: repository root):
#   checkpoints_seed_<seed>/best_<arch>_<task>_<cfg>.pt   (+ .done marker)
#   global_logits_<seed>/<arch>_cfg<cfg>_<task>_test_<split>.txt
#   global_logits_val_<seed>/<arch>_cfg<cfg>_<task>_val[_VM].txt
# Function signature: <arch> <seed> <task> <config>
# ======================================================================

OUT_DIR="${OUT_DIR:-.}"

val_tag() {
    if [ "$1" == "viral" ]; then echo "val_VM"; else echo "val"; fi
}

ckpt_dir()       { echo "${OUT_DIR}/checkpoints_seed_${1}"; }
logits_dir()     { echo "${OUT_DIR}/global_logits_${1}"; }
val_logits_dir() { echo "${OUT_DIR}/global_logits_val_${1}"; }

ckpt_file() {
    echo "$(ckpt_dir "$2")/best_${1}_${3}_${4}.pt"
}

val_file() {
    echo "$(val_logits_dir "$2")/${1}_cfg${4}_${3}_$(val_tag "$3").txt"
}

# Expected test outputs of one (arch, seed, task, config):
#   viral -> test_VM + 15 SIM sets (noise 1/5/10%, 5 realizations each)
#   other -> test_<task>
expected_test_files() {
    local arch=$1 seed=$2 task=$3 cfg=$4
    local d p n k
    d=$(logits_dir "$seed")
    p="${arch}_cfg${cfg}_${task}"
    if [ "$task" == "viral" ]; then
        echo "${d}/${p}_test_VM.txt"
        for n in 1 5 10; do
            for k in 1 2 3 4 5; do
                echo "${d}/${p}_test_sim_noise_${n}_seed_$((n * 1000 + k))_fullset_test.txt"
            done
        done
    else
        echo "${d}/${p}_test_${task}.txt"
    fi
}

# Prints missing or empty outputs (validation + test); no output = complete.
missing_outputs() {
    local f
    f=$(val_file "$@")
    [ -s "$f" ] || echo "$f"
    while read -r f; do
        [ -s "$f" ] || echo "$f"
    done < <(expected_test_files "$@")
}

is_complete() {
    [ -z "$(missing_outputs "$@")" ]
}
