#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef, accuracy_score
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.sequencecnn import SequenceCNN
from models.genexplorer import GenExplorer
from models.nucleic_transformer import NucleicTransformer
from models.fulle import build_fulle_10ensemble
from utils.data_utils import seed_everything, resolve_task_splits, SequenceClassificationDataset, load_seq_label_dataframe


def load_fulle_weights(model, ckpt):
    """Carica i pesi di FullE gestendo sia formati flat (branch_*/merger_*) sia model_state_dict."""
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        return

    if isinstance(ckpt, dict) and "branches" in ckpt and isinstance(ckpt["branches"], dict):
        flat = {}
        for b_name, s_dict in ckpt["branches"].items():
            for p_name, p_val in s_dict.items():
                flat[f"branches.{b_name}.{p_name}"] = p_val
        for m_name, s_dict in ckpt["mergers"].items():
            for p_name, p_val in s_dict.items():
                flat[f"mergers.{m_name}.{p_name}"] = p_val
        model.load_state_dict(flat)
        return

    if isinstance(ckpt, dict) and any(k.startswith("branch_") for k in ckpt.keys()):
        flat = {}
        for top_k, s_dict in ckpt.items():
            if top_k.startswith("branch_"):
                b_name = top_k[len("branch_"):]
                for p_name, p_val in s_dict.items():
                    flat[f"branches.{b_name}.{p_name}"] = p_val
            elif top_k.startswith("merger_"):
                m_name = top_k[len("merger_"):]
                for p_name, p_val in s_dict.items():
                    flat[f"mergers.{m_name}.{p_name}"] = p_val
        model.load_state_dict(flat)
        return

    model.load_state_dict(ckpt)


def main():
    parser = argparse.ArgumentParser(description="Universal Model Evaluation Script")
    parser.add_argument("--arch", type=str, required=True, choices=["sequencecnn", "genexplorer", "nucleictransformer", "dnabert2camp", "fulle"])
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--config", type=str, required=True, choices=["A", "B", "C", "D"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_folder", default=os.environ.get("DATA_DIR", "./datasets"))
    parser.add_argument("--logits_dir", default="./global_logits")
    parser.add_argument("--model_name_or_path", default=os.environ.get("DNABERT2_MODEL_PATH", "zhihan1996/DNABERT-2-117M"))
    parser.add_argument("--input_rep", type=str, default="onehot", choices=["onehot", "physhelical"])
    parser.add_argument("--period_T", type=float, default=None)
    parser.add_argument("--use_length_feature", action="store_true")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--test_csv", type=str, default=None)
    parser.add_argument("--kmers", nargs="+", type=int, default=[2, 3, 4, 5, 6])
    parser.add_argument("--suffix", type=str, default="")
    args = parser.parse_args()

    seed_everything(2020)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.logits_dir, exist_ok=True)

    is_ce = (args.config in ["C", "D"])
    num_classes = 2 if is_ce else 1

    _, _, seq_len, cascade_test_paths = resolve_task_splits(args.task, args.data_folder, seed=2020)
    eval_dict = {"custom_test": args.test_csv} if args.test_csv else cascade_test_paths

    if not eval_dict:
        print("[!] Nessun set di test trovato.")
        return

    tokenizer = None
    if args.arch == "dnabert2camp":
        from transformers import AutoTokenizer
        from models.dnabert2_camp import DNABERT2CAMP
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    in_ch = 4 if args.input_rep == "physhelical" else (5 if args.arch == "fulle" else 4)
    safe_max_len = max(2048, seq_len + 64)
    is_variable_len = (args.task.lower() == "basith" or args.use_length_feature)

    ckpt = torch.load(args.checkpoint, map_location=device)
    raw_state = ckpt["model_state_dict"] if (isinstance(ckpt, dict) and "model_state_dict" in ckpt) else ckpt

    if args.arch == "sequencecnn":
        model = SequenceCNN(in_channels=4, out_channels=num_classes, use_length_feature=is_variable_len)
        model.load_state_dict(raw_state)
    elif args.arch == "genexplorer":
        if isinstance(raw_state, dict) and "transformer.pos_enc.weight" in raw_state:
            ge_max_len = raw_state["transformer.pos_enc.weight"].shape[0] - 1
        else:
            ge_max_len = 512
        model = GenExplorer(in_channels=4, out_features=num_classes, max_len=ge_max_len, use_length_feature=is_variable_len)
        model.load_state_dict(raw_state)
    elif args.arch == "nucleictransformer":
        model = NucleicTransformer(ntoken=4, nclass=num_classes, ninp=512, kmers=args.kmers, max_len=max(2000, seq_len * 6))
        model.load_state_dict(raw_state)
    elif args.arch == "dnabert2camp":
        model = DNABERT2CAMP(model_name_or_path=args.model_name_or_path, num_classes=num_classes)
        model.load_state_dict(raw_state)
    elif args.arch == "fulle":
        model = build_fulle_10ensemble(in_channels=in_ch, num_classes=num_classes, use_length_feature=is_variable_len)
        load_fulle_weights(model, ckpt)

    model.to(device).eval()

    for test_tag, test_path in eval_dict.items():
        print(f"\n--- Valutazione {args.arch.upper()} | Task: {args.task} | Set: {test_tag} ---")
        df_te = load_seq_label_dataframe(test_path)
        te_ds = SequenceClassificationDataset(
            df_te["sequence"].tolist(), df_te["label"].tolist(),
            arch=args.arch, target_len=seq_len, config=args.config, is_train=False,
            tokenizer=tokenizer, task=args.task, input_rep=args.input_rep, period_T=args.period_T
        )
        te_loader = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())

        preds, logits_list, targets = [], [], []
        with torch.no_grad():
            for batch in te_loader:
                if args.arch == "dnabert2camp":
                    ids = batch["input_ids"].to(device, non_blocking=True)
                    att = batch["attention_mask"].to(device, non_blocking=True)
                    out = model(ids, att)
                else:
                    xb = batch["data"].to(device, non_blocking=True)
                    mb = batch["mask"].to(device, non_blocking=True) if ("mask" in batch and is_variable_len) else None
                    lb = batch.get("len_feat", None)
                    if lb is not None and is_variable_len:
                        lb = lb.to(device, non_blocking=True)

                    if args.arch in ["sequencecnn", "genexplorer", "fulle"]:
                        out = model(xb, mask=mb, lengths=lb) if mb is not None else model(xb)
                    else:
                        out = model(xb)

                # Fix dimensionalità: usa softmax solo se out ha effettivamente 2 o più colonne
                if is_ce and (out.ndim > 1 and out.shape[-1] > 1):
                    probs = F.softmax(out, dim=1)[:, 1].cpu().numpy()
                    raw_logits = (out[:, 1] - out[:, 0]).cpu().numpy()
                else:
                    probs = torch.sigmoid(out.view(-1)).cpu().numpy()
                    raw_logits = out.view(-1).cpu().numpy()

                preds.extend(probs)
                logits_list.extend(raw_logits)
                targets.extend(batch["label"].numpy())

        preds_arr = np.nan_to_num(np.array(preds), nan=0.5)
        targets_arr = np.array(targets)
        logits_arr = np.array(logits_list)

        auc = roc_auc_score(targets_arr, preds_arr) if len(np.unique(targets_arr)) > 1 else 0.5
        prc = average_precision_score(targets_arr, preds_arr) if len(np.unique(targets_arr)) > 1 else 0.0
        mcc = matthews_corrcoef(targets_arr, (preds_arr >= 0.5).astype(int)) if len(np.unique(targets_arr)) > 1 else 0.0
        acc = accuracy_score(targets_arr, (preds_arr >= 0.5).astype(int))

        print(f"[{test_tag}] AUROC: {auc:.4f} | AUPRC: {prc:.4f} | MCC: {mcc:.4f} | Acc: {acc:.4f}")

        rep_suffix = f"_{args.input_rep}" if args.input_rep != "onehot" else ""
        out_name = f"{args.arch}_cfg{args.config}{rep_suffix}_{args.task}_{test_tag}.txt"
        np.savetxt(os.path.join(args.logits_dir, out_name), logits_arr, fmt="%.6f")
        print(f"[✓] Logit salvati in: {os.path.join(args.logits_dir, out_name)}")


if __name__ == "__main__":
    main()