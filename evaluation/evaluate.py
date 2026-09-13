#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Universal Standalone Inference Script:
Extracts logit files across all benchmark test sets.
Supports: GenExplorer, Nucleic Transformer, DNABERT2-CAMP, and FullE 10-Model Sum-Rule Ensemble.
"""

import argparse
import os
import sys
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.data_utils import load_seq_label_dataframe, resolve_task_splits, SequenceClassificationDataset


def get_default_model_path():
    local_p = "/nfsd/nannidl/simionricc/models/dnabert2_camp/model"
    return local_p if os.path.exists(local_p) else "zhihan1996/DNABERT-2-117M"


def evaluate_loader(model, loader, is_ce, is_dnabert, is_fulle, device):
    # INVARIATO rispetto alla versione precedente: nessuna modifica in questa funzione.
    model.eval()
    all_logits, all_targets = [], []
    with torch.no_grad():
        for batch in loader:
            if is_dnabert:
                out = model(batch["input_ids"].to(device, non_blocking=True), batch["attention_mask"].to(device, non_blocking=True))
            else:
                out = model(batch["data"].to(device, non_blocking=True))
                if not is_ce and not is_fulle:
                    out = out.squeeze(-1)

            logits_val = (out[:, 1] - out[:, 0]) if (is_ce and not is_fulle) else out
            all_logits.extend(logits_val.cpu().numpy().flatten())
            all_targets.extend(batch["label"].numpy().flatten())

    logits_arr = np.nan_to_num(np.array(all_logits), nan=0.0)
    targets_arr = np.array(all_targets).astype(int)
    probs_arr = 1.0 / (1.0 + np.exp(-logits_arr))

    auc = roc_auc_score(targets_arr, probs_arr) if len(np.unique(targets_arr)) > 1 else 0.5
    prc = average_precision_score(targets_arr, probs_arr) if len(np.unique(targets_arr)) > 1 else 0.0
    mcc = matthews_corrcoef(targets_arr, (probs_arr >= 0.5).astype(int)) if len(np.unique(targets_arr)) > 1 else 0.0
    return auc, prc, mcc, logits_arr


def main():
    parser = argparse.ArgumentParser(description="Universal Model Evaluation & Logits Extraction")
    parser.add_argument("--arch", type=str, required=True, choices=["genexplorer", "nucleictransformer", "dnabert2camp", "fulle"])
    parser.add_argument("--task", type=str, required=True, choices=["viral", "ecoli", "enhancer", "anellovirus"])
    parser.add_argument("--config", type=str, required=True, choices=["A", "B", "C", "D"])
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint (.pt/.pth)")
    parser.add_argument("--model_name_or_path", default=get_default_model_path())
    parser.add_argument("--data_folder", default="/home/simionricc/datasets")
    parser.add_argument("--logits_dir", default="./global_logits")
    parser.add_argument("--test_csv", default=None, help="Optional direct test CSV path")
    # --- Nuovi argomenti, usati SOLO da arch=nucleictransformer (vedi piu' sotto) ---
    parser.add_argument("--kmers", type=int, nargs="+", default=[2, 3, 4, 5, 6],
                         help="[Solo nucleictransformer] Kernel size(s) del checkpoint da caricare. "
                              "Non ha alcun effetto su genexplorer/dnabert2camp/fulle.")
    parser.add_argument("--suffix", type=str, default="",
                         help="[Solo nucleictransformer] Suffisso per il nome del file di logit "
                              "(es. _k13 o _nmute40), per non sovrascrivere run di ablazione diversi. "
                              "Non ha alcun effetto su genexplorer/dnabert2camp/fulle.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, seq_len, cascade_dict = resolve_task_splits(args.task, args.data_folder)

    if args.test_csv:
        cascade_dict = {"custom_test": args.test_csv}

    is_ce = (args.config in ["C", "D"])
    num_classes = 2 if is_ce else 1
    is_dnabert = (args.arch == "dnabert2camp")
    is_fulle = (args.arch == "fulle")
    tokenizer = None

    if args.arch == "genexplorer":
        # INVARIATO
        from models.genexplorer import GenExplorer
        model = GenExplorer(out_features=num_classes, max_len=max(512, seq_len))
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
        model.to(device).eval()

    elif args.arch == "nucleictransformer":
        # UNICA MODIFICA: kmers=args.kmers invece di lasciare il default della classe.
        # Con args.kmers=[2,3,4,5,6] (default di questo script) il comportamento e'
        # identico a prima: NucleicTransformer(ntoken=4, nclass=num_classes, ninp=512)
        # usa gia' [2,3,4,5,6] come default interno.
        from models.nucleic_transformer import NucleicTransformer
        model = NucleicTransformer(ntoken=4, nclass=num_classes, ninp=512, kmers=args.kmers)
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
        model.to(device).eval()

    elif args.arch == "dnabert2camp":
        # INVARIATO
        from transformers import AutoTokenizer
        from models.dnabert2_camp import DNABERT2CAMP
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        model = DNABERT2CAMP(model_name_or_path=args.model_name_or_path, num_classes=num_classes)
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
        model.to(device).eval()

    elif args.arch == "fulle":
        # INVARIATO
        from models.fulle import FullEBranch, FullEMerger, FullE10Ensemble
        ckpt = torch.load(args.checkpoint, map_location=device)

        branches = {
            "freq": FullEBranch(in_channels=5, out_channels=1000, kernel_size=8, pool_type="avg", num_classes=num_classes).to(device),
            "freq_paper": FullEBranch(in_channels=5, out_channels=1000, kernel_size=8, pool_type="avg", num_classes=num_classes).to(device),
            "pat": FullEBranch(in_channels=5, out_channels=1200, kernel_size=11, pool_type="max", num_classes=num_classes).to(device),
            "pat_paper": FullEBranch(in_channels=5, out_channels=1200, kernel_size=11, pool_type="max", num_classes=num_classes).to(device),
            "lp": FullEBranch(in_channels=5, out_channels=1200, kernel_size=11, pool_type="lp", num_classes=num_classes).to(device),
        }
        for name, b in branches.items():
            if "branches" in ckpt and name in ckpt["branches"]:
                b.load_state_dict(ckpt["branches"][name])
            elif f"branch_{name}" in ckpt:
                b.load_state_dict(ckpt[f"branch_{name}"])
            else:
                b.load_state_dict(ckpt[name])

        mergers = {
            "merger_fpl": FullEMerger([branches["freq"], branches["pat"], branches["lp"]], num_classes=num_classes).to(device),
            "merger_fl": FullEMerger([branches["freq"], branches["lp"]], num_classes=num_classes).to(device),
            "merger_pl": FullEMerger([branches["pat"], branches["lp"]], num_classes=num_classes).to(device),
            "merger_vm": FullEMerger([branches["freq"], branches["pat"]], num_classes=num_classes).to(device),
            "merger_vm_paper": FullEMerger([branches["freq_paper"], branches["pat_paper"]], num_classes=num_classes).to(device),
        }
        for name, m in mergers.items():
            if "mergers" in ckpt and name in ckpt["mergers"]:
                m.load_state_dict(ckpt["mergers"][name])
            elif f"merger_{name}" in ckpt:
                m.load_state_dict(ckpt[f"merger_{name}"])
            else:
                m.load_state_dict(ckpt[name])

        model = FullE10Ensemble(branches, mergers, num_classes=num_classes).to(device).eval()

    os.makedirs(args.logits_dir, exist_ok=True)
    prefix_map = {
        "genexplorer": "genexplorer",
        "nucleictransformer": "nucleictransformer",
        "dnabert2camp": "dnabert2camp",
        "fulle": "fulle"
    }

    # Suffisso applicato al nome file SOLO per nucleictransformer. Per tutti gli
    # altri arch, effective_suffix resta "" a prescindere da cosa contiene
    # args.suffix: il nome del file di logit per genexplorer/dnabert2camp/fulle
    # e' quindi bit-per-bit identico a prima, in ogni caso.
    effective_suffix = args.suffix if (args.arch == "nucleictransformer" and args.suffix) else ""

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

        out_fname = f"{prefix_map[args.arch]}_cfg{args.config}{effective_suffix}_{args.task}_{d_name}.txt"
        out_fpath = os.path.join(args.logits_dir, out_fname)
        np.savetxt(out_fpath, raw_logits, fmt="%.6f")
        print(f"    Logits saved: {out_fname}")


if __name__ == "__main__":
    main()