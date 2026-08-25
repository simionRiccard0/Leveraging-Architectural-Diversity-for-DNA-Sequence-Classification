#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Universal Standalone Inference Script:
Loads trained checkpoints and extracts logit files for all test benchmarks[cite: 16].
Supports: GenExplorer, Nucleic Transformer, DNABERT2-CAMP, and FullE Ensemble[cite: 16].
"""

import argparse
import os
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef
import torch
from torch.utils.data import DataLoader

from utils.data_utils import load_seq_label_dataframe, resolve_task_splits, SequenceClassificationDataset


def evaluate_loader(model, loader, is_ce, is_dnabert, is_fulle, device):
    model.eval()
    all_logits, all_targets = [], []
    with torch.no_grad():
        for batch in loader:
            if is_dnabert:
                out = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            else:
                out = model(batch["data"].to(device))
                if not is_ce:
                    out = out.squeeze(-1)

            logits_val = (out[:, 1] - out[:, 0]) if is_ce else out
            all_logits.extend(logits_val.cpu().numpy().flatten())
            all_targets.extend(batch["label"].numpy().flatten())

    logits_arr = np.array(all_logits)
    targets_arr = np.array(all_targets)
    probs_arr = 1.0 / (1.0 + np.exp(-logits_arr))

    auc = roc_auc_score(targets_arr, probs_arr) if len(np.unique(targets_arr)) > 1 else 0.5
    prc = average_precision_score(targets_arr, probs_arr) if len(np.unique(targets_arr)) > 1 else 0.0
    mcc = matthews_corrcoef(targets_arr, (probs_arr >= 0.5).astype(int)) if len(np.unique(targets_arr)) > 1 else 0.0
    return auc, prc, mcc, logits_arr


def main():
    parser = argparse.ArgumentParser(description="Universal Model Evaluation & Logits Extraction")
    parser.add_argument("--arch", type=str, required=True, choices=["genexplorer", "nucleictransformer", "dnabert2camp", "fulle"])
    parser.add_argument("--task", type=str, required=True, choices=["viral", "ecoli", "enhancer", "anellovirus"])
    parser.add_argument("--config", type=str, required=True, choices=["A", "B", "C"])
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint (.pt/.pth)")
    parser.add_argument("--data_folder", default="/home/simionricc/datasets")
    parser.add_argument("--logits_dir", default="./global_logits")
    parser.add_argument("--test_csv", default=None, help="Optional direct test CSV path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, seq_len, cascade_dict = resolve_task_splits(args.task, args.data_folder)

    if args.test_csv:
        cascade_dict = {"custom_test": args.test_csv}

    is_ce = (args.config == "C")
    num_classes = 2 if is_ce else 1
    is_dnabert = (args.arch == "dnabert2camp")
    is_fulle = (args.arch == "fulle")
    tokenizer = None

    # Lazy-loading modulare delle architetture per isolare dipendenze e ambienti
    if args.arch == "genexplorer":
        from models.genexplorer import GenExplorer
        model = GenExplorer(out_features=num_classes, max_len=max(512, seq_len))
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
        model.to(device).eval()

    elif args.arch == "nucleictransformer":
        from models.nucleic_transformer import NucleicTransformer
        model = NucleicTransformer(ntoken=4, nclass=num_classes, ninp=512)
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
        model.to(device).eval()

    elif args.arch == "dnabert2camp":
        from transformers import AutoTokenizer
        from models.dnabert2_camp import DNABERT2CAMP
        tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
        model = DNABERT2CAMP(num_classes=num_classes)
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
        model.to(device).eval()

    elif args.arch == "fulle":
        from models.fulle import FullEBranch, FullEMerger
        ckpt = torch.load(args.checkpoint, map_location=device)
        b_freq = FullEBranch(in_channels=5, out_channels=1000, kernel_size=8, pool_type="avg", num_classes=num_classes).to(device)
        b_pat = FullEBranch(in_channels=5, out_channels=1200, kernel_size=11, pool_type="max", num_classes=num_classes).to(device)
        b_lp = FullEBranch(in_channels=5, out_channels=1200, kernel_size=11, pool_type="lp", num_classes=num_classes).to(device)
        merger = FullEMerger([b_freq, b_pat, b_lp], num_classes=num_classes).to(device)

        b_freq.load_state_dict(ckpt["branch_freq"])
        b_pat.load_state_dict(ckpt["branch_pat"])
        b_lp.load_state_dict(ckpt["branch_lp"])
        merger.load_state_dict(ckpt["merger_full"])
        model = merger.eval()

    os.makedirs(args.logits_dir, exist_ok=True)
    prefix_map = {
        "genexplorer": "genexplorer",
        "nucleictransformer": "nucleictransformer",
        "dnabert2camp": "dnabert2camp",
        "fulle": "fulle"
    }

    print(f"\n--- Cascade Evaluation: {args.arch} (Config {args.config}, Task {args.task}) ---")
    for d_name, d_path in cascade_dict.items():
        if not os.path.exists(d_path):
            continue
        eval_df = load_seq_label_dataframe(d_path)
        eval_ds = SequenceClassificationDataset(
            eval_df["sequence"].tolist(),
            eval_df["label"].tolist(),
            args.arch,
            seq_len,
            args.config,
            is_train=False,
            tokenizer=tokenizer
        )
        eval_loader = DataLoader(eval_ds, batch_size=64, shuffle=False)

        auc, prc, mcc, raw_logits = evaluate_loader(model, eval_loader, is_ce, is_dnabert, is_fulle, device)
        print(f" -> [{d_name}] AUROC: {auc:.4f} | AUPRC: {prc:.4f} | MCC: {mcc:.4f}")

        out_fname = f"{prefix_map[args.arch]}_cfg{args.config}_{args.task}_{d_name}.txt"
        out_fpath = os.path.join(args.logits_dir, out_fname)
        np.savetxt(out_fpath, raw_logits, fmt="%.6f")
        print(f"    Logits saved: {out_fname}")


if __name__ == "__main__":
    main()