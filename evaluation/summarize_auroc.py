#!/usr/bin/env python3
"""
AUROC summary of the benchmark logits of one seed.

Reads the files written by evaluation/evaluate.py
    <logits_dir>/<arch>_cfg<C>_<task>_test_<split>.txt
    <val_logits_dir>/<arch>_cfg<C>_<task>_val[_VM].txt
rebuilds the labels from the dataset files (same split as the run, given its seed)
and prints validation and test AUROC (%) for every architecture and configuration.
SIM values are averaged over the noise realizations of each level.

Example:
    python evaluation/summarize_auroc.py --seed 2020 \
        --logits_dir global_logits_2020 --val_logits_dir global_logits_val_2020
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.data_utils import load_seq_label_dataframe, resolve_task_splits  # noqa: E402

TASKS = {"viral": "VM", "ecoli": "ECO", "enhancer": "ENH", "anellovirus": "OOD"}
TEST_TAG = {"viral": "test_VM", "ecoli": "test_ecoli", "enhancer": "test_enhancer",
            "anellovirus": "test_anellovirus"}
ARCHS = ["fulle", "genexplorer", "nucleictransformer", "dnabert2camp", "helixpr"]
CONFIGS = "ABCD"
NOISE_LEVELS = [1, 5, 10]


def locate_sim(data_folder):
    """SIM test files: tag 'test_sim_noise_<n>_seed_<s>_fullset_test' -> path."""
    sim = {}
    for path in glob.glob(os.path.join(data_folder, "**", "*.csv"), recursive=True):
        norm = path.replace("\\", "/")
        m = (re.search(r"sim_noise_(\d+)_seed_(\d+)_fullset_test", os.path.basename(norm))
             or re.search(r"noise_(\d+)/seed_(\d+)/fullset_test\.csv", norm))
        if m:
            sim.setdefault(f"test_sim_noise_{m.group(1)}_seed_{m.group(2)}_fullset_test", path)
    return sim


def auroc(labels, path):
    scores = np.loadtxt(path)
    if scores.ndim != 1 or len(scores) != len(labels):
        raise ValueError(f"{path}: {scores.shape} scores vs {len(labels)} labels")
    return 100.0 * roc_auc_score(labels, scores)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=2020, help="seed of the run (sets the ECO/ENH/OOD validation split)")
    ap.add_argument("--logits_dir", required=True)
    ap.add_argument("--val_logits_dir", required=True)
    ap.add_argument("--data_folder", default=os.environ.get("DATA_DIR", "./datasets"))
    ap.add_argument("--json_out", default=None, help="optional JSON file with all values")
    args = ap.parse_args()

    val_labels, test_labels = {}, {}
    for task in TASKS:
        _, val_df, _, test_paths = resolve_task_splits(task, args.data_folder, seed=args.seed)
        val_labels[task] = val_df["label"].values
        test_labels[(task, TEST_TAG[task])] = load_seq_label_dataframe(test_paths[TEST_TAG[task]])["label"].values
    sim = locate_sim(args.data_folder)
    for tag, path in sim.items():
        test_labels[("viral", tag)] = load_seq_label_dataframe(path)["label"].values

    results, problems = {}, []
    for arch in ARCHS:
        for cfg in CONFIGS:
            for task in TASKS:
                vt = "val_VM" if task == "viral" else "val"
                fv = os.path.join(args.val_logits_dir, f"{arch}_cfg{cfg}_{task}_{vt}.txt")
                ft = os.path.join(args.logits_dir, f"{arch}_cfg{cfg}_{task}_{TEST_TAG[task]}.txt")
                if not os.path.exists(ft):
                    continue
                try:
                    results[(arch, cfg, TASKS[task], "test")] = auroc(test_labels[(task, TEST_TAG[task])], ft)
                    if os.path.exists(fv):
                        results[(arch, cfg, TASKS[task], "val")] = auroc(val_labels[task], fv)
                except ValueError as err:
                    problems.append(str(err))
            for n in NOISE_LEVELS:
                vals = []
                for tag in sorted(t for t in sim if re.search(rf"_noise_{n}_seed_", t)):
                    f = os.path.join(args.logits_dir, f"{arch}_cfg{cfg}_viral_{tag}.txt")
                    if os.path.exists(f):
                        try:
                            vals.append(auroc(test_labels[("viral", tag)], f))
                        except ValueError as err:
                            problems.append(str(err))
                if vals:
                    results[(arch, cfg, f"SIM{n}", "test")] = float(np.mean(vals))
                    results[(arch, cfg, f"SIM{n}", "n_rep")] = len(vals)

    cols = list(TASKS.values()) + [f"SIM{n}" for n in NOISE_LEVELS]
    print(f"Seed {args.seed} | SIM sets found: {len(sim)}")
    print("Test AUROC (%)  [validation AUROC in brackets]\n")
    print(f"{'model':<26}" + "".join(f"{c:>16}" for c in cols))
    for arch in ARCHS:
        for cfg in CONFIGS:
            if not any(k[0] == arch and k[1] == cfg for k in results):
                continue
            cells = []
            for c in cols:
                t = results.get((arch, cfg, c, "test"))
                v = results.get((arch, cfg, c, "val"))
                cell = "--" if t is None else f"{t:.2f}" + (f" [{v:.2f}]" if v is not None else "")
                cells.append(f"{cell:>16}")
            print(f"{arch + ' C' + str(CONFIGS.index(cfg) + 1):<26}" + "".join(cells))

    if problems:
        print("\nProblems:")
        print("\n".join(problems))
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump({"|".join(map(str, k)): v for k, v in results.items()}, fh, indent=1)


if __name__ == "__main__":
    main()
