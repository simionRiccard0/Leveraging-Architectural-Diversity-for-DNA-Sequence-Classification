#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dedicated training pipeline for Nucleic Transformer (He et al.).
Fully aligned with 4x3 Cross-Benchmarking (Strategies A/C1, B/C2, C/C3).
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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.nucleic_transformer import NucleicTransformer
from utils.data_utils import seed_everything, resolve_task_splits, SequenceClassificationDataset


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, smoothing: float = 0.05):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.view(-1)
        targets = targets.view(-1)
        targets_smooth = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        bce = F.binary_cross_entropy_with_logits(logits, targets_smooth, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1 - targets) * (1 - probs)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        return (alpha_t * ((1 - p_t) ** self.gamma) * bce).mean()


def main():
    parser = argparse.ArgumentParser(description="Nucleic Transformer Training Pipeline")
    parser.add_argument("--config", type=str, required=True, choices=["A", "B", "C"], help="Strategy: A (C1), B (C2), C (C3)")
    parser.add_argument("--task", type=str, default="viral", choices=["viral", "ecoli", "enhancer", "anellovirus"])
    parser.add_argument("--data_folder", default="/home/simionricc/datasets")
    parser.add_argument("--checkpoint_dir", default="./checkpoints")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2020)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_map = {
        "A": {"epochs": 120, "early_stop": True, "patience": 15, "loss": "focal", "opt": "adamw", "sched": "cosine", "drop": 0.25, "out": 1},
        "B": {"epochs": 30, "early_stop": False, "patience": 0, "loss": "bce", "opt": "adam", "sched": "step", "drop": 0.2, "out": 1},
        "C": {"epochs": 60, "early_stop": False, "patience": 0, "loss": "ce", "opt": "adam", "sched": "aiayn", "drop": 0.1, "out": 2},
    }
    cfg = config_map[args.config]
    is_ce = (args.config == "C")

    train_df, val_df, seq_len, _ = resolve_task_splits(args.task, args.data_folder, args.seed)
    print(f"[i] Training Nucleic Transformer | Task: '{args.task}' | Config: '{args.config}' | Input: {seq_len} bp | Train: {len(train_df)} | Val: {len(val_df)}")

    tr_ds = SequenceClassificationDataset(train_df["sequence"].tolist(), train_df["label"].tolist(), "nucleictransformer", seq_len, args.config, is_train=True)
    val_ds = SequenceClassificationDataset(val_df["sequence"].tolist(), val_df["label"].tolist(), "nucleictransformer", seq_len, args.config, is_train=False)

    pin_mem = torch.cuda.is_available()
    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=pin_mem)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=2, pin_memory=pin_mem)

    model = NucleicTransformer(
        ntoken=4,
        nclass=cfg["out"],
        ninp=512,
        nhead=8,
        nhid=2048,
        nlayers=6,
        kmer_aggregation=True,
        kmers=[2, 3, 4, 5, 6],
        dropout=cfg["drop"],
        max_len=max(2000, seq_len + 100)
    ).to(device)

    # Definizione Loss coerente
    if cfg["loss"] == "focal":
        criterion = BinaryFocalLoss(alpha=0.75, gamma=2.0)
    elif cfg["loss"] == "bce":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    if cfg["opt"] == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    else:
        lr_base = 0.001 if args.config == "B" else 1.0
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_base)

    if cfg["sched"] == "cosine":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda ep: (ep + 1) / 5 if ep < 5 else 0.5 * (1 + math.cos(math.pi * (ep - 5) / (cfg["epochs"] - 5))))
    elif cfg["sched"] == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: (512**-0.5) * min(max(1, s)**-0.5, max(1, s) * (4000**-1.5)))

    best_auc, patience_cnt = 0.0, 0
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_path = os.path.join(args.checkpoint_dir, f"best_nucleictransformer_{args.task}_{args.config}.pt")

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        total_loss = 0.0

        for batch in tr_loader:
            optimizer.zero_grad(set_to_none=True)
            src = batch["data"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            
            output = model(src)
            
            if is_ce:
                loss = criterion(output, labels.view(-1).long())
            else:
                loss = criterion(output.view(-1), labels.view(-1).float())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            if cfg["sched"] == "aiayn":
                scheduler.step()
                
            total_loss += loss.item() * len(src)

        if cfg["sched"] != "aiayn":
            scheduler.step()

        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                src = batch["data"].to(device, non_blocking=True)
                out = model(src)
                probs = F.softmax(out, dim=1)[:, 1] if is_ce else torch.sigmoid(out.view(-1))
                preds.extend(probs.cpu().numpy().flatten())
                targets.extend(batch["label"].numpy().flatten())

        preds_arr = np.nan_to_num(np.array(preds), nan=0.5)
        targets_arr = np.array(targets)
        
        val_auc = roc_auc_score(targets_arr, preds_arr) if len(np.unique(targets_arr)) > 1 else 0.5
        val_prc = average_precision_score(targets_arr, preds_arr) if len(np.unique(targets_arr)) > 1 else 0.0
        val_mcc = matthews_corrcoef(targets_arr, (preds_arr >= 0.5).astype(int)) if len(np.unique(targets_arr)) > 1 else 0.0
        
        print(f"Epoch {epoch:03d}/{cfg['epochs']:03d} | Train Loss: {total_loss/len(tr_ds):.4f} | Val AUC: {val_auc:.4f} | Val PRC: {val_prc:.4f} | Val MCC: {val_mcc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            patience_cnt = 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "val_auc": val_auc}, best_path)
        else:
            patience_cnt += 1
            if cfg["early_stop"] and patience_cnt >= cfg["patience"]:
                print(f"Early stopping at epoch {epoch}.")
                break

    print(f"[✓] Checkpoint saved: {best_path} (Best Val AUC: {best_auc:.4f})")


if __name__ == "__main__":
    main()