#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Training and evaluation pipeline for GenExplorer Ablation Study.
Runs strictly under Configuration C1 (Config A native: Focal Loss, AdamW, Cosine Warmup).
"""

import argparse
import math
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from models.genexplorer_ablation import GenExplorerAblation
from utils.data_utils import (
    RC_CHOICES,
    parse_rc,
    seed_everything,
    resolve_task_splits,
    load_seq_label_dataframe,
    SequenceClassificationDataset
)


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, smoothing: float = 0.05):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.view(-1)
        targets = targets.view(-1)
        targets_smooth = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        bce = F.binary_cross_entropy_with_logits(logits, targets_smooth, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1.0 - targets) * (1.0 - probs)
        alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
        return (alpha_t * ((1.0 - p_t) ** self.gamma) * bce).mean()


def evaluate(model, dataloader, device, return_logits: bool = False):
    model.eval()
    all_preds, all_targets, all_logits = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            x = batch["data"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            logits = model(x).view(-1)
            probs = torch.sigmoid(logits)

            all_preds.extend(probs.cpu().numpy().flatten())
            all_targets.extend(y.cpu().numpy().flatten())
            all_logits.extend(logits.cpu().numpy().flatten())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_logits = np.array(all_logits)

    if len(np.unique(all_targets)) < 2:
        auc_score, ap_score, mcc_score = 0.5, 0.0, 0.0
    else:
        auc_score = roc_auc_score(all_targets, all_preds)
        ap_score = average_precision_score(all_targets, all_preds)
        mcc_score = matthews_corrcoef(all_targets, (all_preds >= 0.5).astype(int))

    if return_logits:
        return auc_score, ap_score, mcc_score, all_logits
    return auc_score, ap_score, mcc_score


def main():
    parser = argparse.ArgumentParser(description="GenExplorer Ablation Study (Config C1)")
    parser.add_argument("--task", type=str, required=True, choices=["viral", "ecoli", "enhancer", "anellovirus"])
    parser.add_argument("--branches", type=str, required=True, choices=["F", "P", "FP", "FPT"],
                        help="Active branches: 'F', 'P', 'FP' (dual CNN), 'FPT' (full GE)")
    parser.add_argument("--data_folder", default=os.environ.get("DATA_DIR", "./datasets"))
    parser.add_argument("--checkpoint_dir", default="./checkpoints")
    parser.add_argument("--logits_dir", default="./global_logits_ablation")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--rc", choices=RC_CHOICES, default="auto", help="Reverse complement (auto: off on ECO)")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.logits_dir, exist_ok=True)

    # 1. Splits
    train_df, val_df, seq_len, cascade_dict = resolve_task_splits(args.task, args.data_folder, args.seed)
    print(f"[i] Ablation GenExplorer | Task: '{args.task}' | Branches: '{args.branches}' | Input: {seq_len} bp | Train: {len(train_df)} | Val: {len(val_df)}")

    # 2. Datasets with the C1 (config A) augmentation
    tr_ds = SequenceClassificationDataset(
        train_df["sequence"].tolist(), train_df["label"].tolist(),
        arch="genexplorer", target_len=seq_len, config="A", is_train=True,
        task=args.task, use_rc=parse_rc(args.rc)
    )
    val_ds = SequenceClassificationDataset(
        val_df["sequence"].tolist(), val_df["label"].tolist(),
        arch="genexplorer", target_len=seq_len, config="A", is_train=False
    )

    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())

    # 3. Model
    model = GenExplorerAblation(
        branches=args.branches,
        cnn_drop=0.25,
        trans_drop=0.1,
        out_features=1,
        max_len=max(512, seq_len)
    ).to(device)

    criterion = BinaryFocalLoss(alpha=0.75, gamma=2.0, smoothing=0.05)

    # 4. AdamW with a separate learning rate for the Transformer branch (if present)
    if "T" in args.branches:
        trans_p = list(model.transformer.parameters())
        other_p = [p for n, p in model.named_parameters() if not n.startswith("transformer")]
        optimizer = torch.optim.AdamW([
            {"params": other_p, "lr": 1e-4},
            {"params": trans_p, "lr": 5e-5}
        ], weight_decay=1e-4)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # 5-epoch warmup + cosine schedule
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda ep: (ep + 1) / 5 if ep < 5 else 0.5 * (1.0 + math.cos(math.pi * (ep - 5) / (args.epochs - 5)))
    )

    # Transformer branch frozen during the first 5 epochs
    if "T" in args.branches:
        for p in model.transformer.parameters():
            p.requires_grad = False

    best_auc, patience_cnt = 0.0, 0
    patience = 15
    branch_tag = "GE" if args.branches == "FPT" else f"GE_{args.branches}"
    ckpt_path = os.path.join(args.checkpoint_dir, f"best_ablation_{branch_tag}_{args.task}.pt")

    # 5. Training loop
    for epoch in range(1, args.epochs + 1):
        if "T" in args.branches and epoch == 6:
            for p in model.transformer.parameters():
                p.requires_grad = True
            print("[i] Transformer unfreezed post-warmup.")

        model.train()
        total_loss, total_samples = 0.0, 0

        for batch in tr_loader:
            x_b = batch["data"].to(device, non_blocking=True)
            y_b = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            out = model(x_b)
            loss = criterion(out, y_b.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * len(x_b)
            total_samples += len(x_b)

        scheduler.step()
        val_auc, val_ap, val_mcc = evaluate(model, val_loader, device)
        train_loss = total_loss / max(1, total_samples)

        print(f"Epoch {epoch:03d}/{args.epochs:03d} | Train Loss: {train_loss:.4f} | Val AUC: {val_auc:.4f} | Val AP: {val_ap:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            patience_cnt = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    print(f"[✓] Best checkpoint saved: {ckpt_path} (Val AUC: {best_auc:.4f})")

    # 6. Cascade Evaluation sui test set
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"\n--- Cascade Evaluation for Ablation [{branch_tag}] ({args.task}) ---")

    for d_name, d_path in cascade_dict.items():
        if not os.path.exists(d_path):
            continue

        eval_df = load_seq_label_dataframe(d_path)
        eval_ds = SequenceClassificationDataset(
            eval_df["sequence"].tolist(), eval_df["label"].tolist(),
            arch="genexplorer", target_len=seq_len, config="A", is_train=False
        )
        eval_loader = DataLoader(eval_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())

        eval_auc, eval_ap, eval_mcc, raw_logits = evaluate(model, eval_loader, device, return_logits=True)
        print(f" -> [{d_name}] AUROC: {eval_auc:.4f} | AUPRC: {eval_ap:.4f} | MCC: {eval_mcc:.4f}")

        # save the test logits
        out_fname = f"ge_ablation_{branch_tag}_{args.task}_{d_name}.txt"
        out_fpath = os.path.join(args.logits_dir, out_fname)
        np.savetxt(out_fpath, raw_logits, fmt="%.6f")
        print(f"    Logits saved: {out_fpath}")


if __name__ == "__main__":
    main()
