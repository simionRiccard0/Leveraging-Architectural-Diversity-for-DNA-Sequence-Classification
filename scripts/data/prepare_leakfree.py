#!/usr/bin/env python
"""Convert the released leakage-free splits (datasets_leakfree/seed_<seed>/*.txt) into the
folder layout read by the training and evaluation scripts.

Output: <out>/seed_<seed>/ecoli/{ecoli_train_NT.csv, ecoli_independent_test_NT.csv}
        <out>/seed_<seed>/enhancer/{bert_enhancer_dataset.csv, bert_enhancer_ind_test_set.csv}
Each run must use the folder of its own seed, e.g.
    DATA_DIR=datasets_leakfree_csv/seed_2020 SEED=2020 TASKS="ecoli enhancer" ARCH=fulle bash scripts/run_benchmark.sh
The training scripts read *_trainval.txt and hold out the first 20% after shuffling it with the run seed
(pandas DataFrame.sample(frac=1, random_state=seed)), as for the original benchmarks; this hold-out is
exactly the one released in *_val.txt, and the remaining 80% is *_train.txt.

Usage: python scripts/data/prepare_leakfree.py [--src datasets_leakfree] [--out datasets_leakfree_csv]
"""
import argparse
import glob
import os

import pandas as pd

FILES = {
    "ECO_trainval.txt": "ecoli/ecoli_train_NT.csv",
    "ECO_test.txt": "ecoli/ecoli_independent_test_NT.csv",
    "ENH_trainval.txt": "enhancer/bert_enhancer_dataset.csv",
    "ENH_test.txt": "enhancer/bert_enhancer_ind_test_set.csv",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="datasets_leakfree")
    ap.add_argument("--out", default="datasets_leakfree_csv")
    args = ap.parse_args()
    seeds = sorted(glob.glob(os.path.join(args.src, "seed_*")))
    if not seeds:
        raise SystemExit(f"no seed_* folder in {args.src}")
    for sd in seeds:
        for txt, rel in FILES.items():
            df = pd.read_csv(os.path.join(sd, txt), sep="\t")
            dst = os.path.join(args.out, os.path.basename(sd), rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            df[["sequence", "label"]].to_csv(dst, index=False, header=False)
            print(f"{dst}: {len(df)} sequences ({int(df.label.sum())} positives)")


if __name__ == "__main__":
    main()
