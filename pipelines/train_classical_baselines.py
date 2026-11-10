#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classical (non-deep) baselines: L1-normalized k-mer frequency vectors with
Logistic Regression and Random Forest, k in {4, 5, 6} selected on validation.
Reports the test AUROC on VM, ECO, ENH and OOD.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.data_utils import seed_everything, resolve_task_splits, load_seq_label_dataframe

TASKS = ["viral", "ecoli", "enhancer", "anellovirus"]
TASK_NAMES = {"viral": "VM", "ecoli": "ECO", "enhancer": "ENH", "anellovirus": "OOD"}


def extract_kmers(train_seqs, val_seqs, test_seqs, k=4):
    """
    L1-normalized k-mer counts (relative frequencies per sequence).
    """
    vectorizer = CountVectorizer(analyzer="char", ngram_range=(k, k), lowercase=False)
    X_tr = vectorizer.fit_transform(train_seqs).astype(np.float32)
    X_va = vectorizer.transform(val_seqs).astype(np.float32)
    X_te = vectorizer.transform(test_seqs).astype(np.float32)

    X_tr = normalize(X_tr, norm="l1", axis=1)
    X_va = normalize(X_va, norm="l1", axis=1)
    X_te = normalize(X_te, norm="l1", axis=1)

    return X_tr, X_va, X_te, vectorizer


def evaluate_classical_baselines(data_folder="datasets", seed=2020):
    seed_everything(seed)

    results = {
        "kmer_LR": {},
        "kmer_RF": {},
        "best_k_LR": {},
        "best_k_RF": {},
    }

    for task in TASKS:
        tname = TASK_NAMES[task]
        print(f"\n=================== Benchmark: {tname} ({task}) ===================")

        train_df, val_df, seq_len, cascade_test_paths = resolve_task_splits(task, data_folder, seed=seed)

        test_key = [k for k in cascade_test_paths.keys() if task in k or tname.lower() in k.lower()][0]
        test_path = cascade_test_paths[test_key]
        test_df = load_seq_label_dataframe(test_path)

        tr_seqs, tr_y = train_df["sequence"].values, train_df["label"].values
        va_seqs, va_y = val_df["sequence"].values, val_df["label"].values
        te_seqs, te_y = test_df["sequence"].values, test_df["label"].values

        best_val_auc_lr, best_test_auc_lr, best_k_lr = -1, -1, None
        best_val_auc_rf, best_test_auc_rf, best_k_rf = -1, -1, None

        for k in [4, 5, 6]:
            print(f"--- k = {k} ---")
            X_tr, X_va, X_te, _ = extract_kmers(tr_seqs, va_seqs, te_seqs, k=k)

            # Logistic Regression
            lr = LogisticRegression(max_iter=1000, C=1.0, random_state=seed, solver="lbfgs")
            lr.fit(X_tr, tr_y)
            val_probs_lr = lr.predict_proba(X_va)[:, 1]
            test_probs_lr = lr.predict_proba(X_te)[:, 1]
            val_auc_lr = roc_auc_score(va_y, val_probs_lr)
            test_auc_lr = roc_auc_score(te_y, test_probs_lr)

            if val_auc_lr > best_val_auc_lr:
                best_val_auc_lr = val_auc_lr
                best_test_auc_lr = test_auc_lr
                best_k_lr = k

            # Random Forest
            rf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
            rf.fit(X_tr, tr_y)
            val_probs_rf = rf.predict_proba(X_va)[:, 1]
            test_probs_rf = rf.predict_proba(X_te)[:, 1]
            val_auc_rf = roc_auc_score(va_y, val_probs_rf)
            test_auc_rf = roc_auc_score(te_y, test_probs_rf)

            if val_auc_rf > best_val_auc_rf:
                best_val_auc_rf = val_auc_rf
                best_test_auc_rf = test_auc_rf
                best_k_rf = k

            print(f"  k={k} | LR -> Val AUC: {val_auc_lr*100:.2f}%, Test AUC: {test_auc_lr*100:.2f}%")
            print(f"  k={k} | RF -> Val AUC: {val_auc_rf*100:.2f}%, Test AUC: {test_auc_rf*100:.2f}%")

        results["kmer_LR"][tname] = best_test_auc_lr * 100
        results["kmer_RF"][tname] = best_test_auc_rf * 100
        results["best_k_LR"][tname] = best_k_lr
        results["best_k_RF"][tname] = best_k_rf

        print(f"Selected for LR ({tname}): k={best_k_lr} (Test AUROC: {best_test_auc_lr*100:.2f}%)")
        print(f"Selected for RF ({tname}): k={best_k_rf} (Test AUROC: {best_test_auc_rf*100:.2f}%)")

    print("\n================ Test AUROC (%), k selected on validation ================")
    print(f"{'Method':<14}" + "".join(f"{TASK_NAMES[t]:>10}" for t in TASKS))
    for key, name in (("kmer_LR", "k-mer + LR"), ("kmer_RF", "k-mer + RF")):
        print(f"{name:<14}" + "".join(f"{results[key][TASK_NAMES[t]]:>10.2f}" for t in TASKS))
    for key, name in (("best_k_LR", "best k (LR)"), ("best_k_RF", "best k (RF)")):
        print(f"{name:<14}" + "".join(f"{results[key][TASK_NAMES[t]]:>10}" for t in TASKS))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classical k-mer baselines (LR, RF).")
    parser.add_argument("--data_folder", type=str, default=os.environ.get("DATA_DIR", "./datasets"), help="Dataset root.")
    parser.add_argument("--seed", type=int, default=2020, help="Seed (also sets the training/validation partition of ECO, ENH, OOD).")
    args = parser.parse_args()

    evaluate_classical_baselines(data_folder=args.data_folder, seed=args.seed)
