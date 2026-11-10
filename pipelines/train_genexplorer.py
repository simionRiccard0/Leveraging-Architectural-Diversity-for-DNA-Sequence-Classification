#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.genexplorer import GenExplorer
from utils.data_utils import seed_everything, resolve_task_splits, SequenceClassificationDataset, RC_CHOICES, parse_rc


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, smoothing: float = 0.05):
        super().__init__()
        self.alpha, self.gamma, self.smoothing = alpha, gamma, smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.view(-1)
        targets = targets.view(-1)
        targets_smooth = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        bce = F.binary_cross_entropy_with_logits(logits, targets_smooth, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1.0 - targets) * (1.0 - probs)
        alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
        return (alpha_t * ((1.0 - p_t) ** self.gamma) * bce).mean()


def main():
    parser = argparse.ArgumentParser(description="GenExplorer Training Pipeline")
    parser.add_argument("--epochs", type=int, default=None, help="Override default epochs")
    parser.add_argument("--config", type=str, required=True, choices=["A", "B", "C", "D"])
    parser.add_argument("--task", type=str, default="viral")
    parser.add_argument("--data_folder", default=os.environ.get("DATA_DIR", "./datasets"))
    parser.add_argument("--checkpoint_dir", default="./checkpoints")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--input_rep", type=str, default="onehot", choices=["onehot", "komlos"],
                        help="komlos: Komlos-Hadamard encoding (comparison with the literature)")
    parser.add_argument("--use_length_feature", action="store_true", help="Masked pooling + length prior (variable-length inputs)")
    parser.add_argument("--rc", choices=RC_CHOICES, default="auto", help="Reverse complement in config A (auto: off on ECO)")
    parser.add_argument("--focal_alpha", type=float, default=0.75, help="Focal-loss alpha (config A)")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs_map = {"A": 120, "B": 30, "C": 60, "D": 20}
    total_epochs = args.epochs if args.epochs is not None else epochs_map[args.config]
    is_ce = (args.config in ["C", "D"])
    num_classes = 2 if is_ce else 1

    train_df, val_df, seq_len, _ = resolve_task_splits(args.task, args.data_folder, args.seed)
    print(f"[i] Training GenExplorer | Task: '{args.task}' | Config: '{args.config}' (Classes: {num_classes}) | "
          f"Rep: {args.input_rep} | Length Feature: {args.use_length_feature} | Train: {len(train_df)} | Val: {len(val_df)}")

    tr_ds = SequenceClassificationDataset(
        train_df["sequence"].tolist(), train_df["label"].tolist(),
        arch="genexplorer", target_len=seq_len, config=args.config, is_train=True,
        task=args.task, input_rep=args.input_rep, use_rc=parse_rc(args.rc)
    )
    val_ds = SequenceClassificationDataset(
        val_df["sequence"].tolist(), val_df["label"].tolist(),
        arch="genexplorer", target_len=seq_len, config=args.config, is_train=False,
        task=args.task, input_rep=args.input_rep
    )

    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=torch.cuda.is_available())

    in_ch = 3 if args.input_rep == "komlos" else 4

    safe_max_len = max(2048, seq_len + 64)
    model = GenExplorer(
        in_channels=in_ch,
        out_features=num_classes,
        max_len=safe_max_len,
        use_length_feature=args.use_length_feature
    ).to(device)

    if args.config == "A":
        criterion = BinaryFocalLoss(alpha=args.focal_alpha, gamma=2.0)
        trans_p = list(model.transformer.parameters())
        other_p = list(model.pattern.parameters()) + list(model.frequency.parameters()) + list(model.fc.parameters())
        optimizer = torch.optim.AdamW([
            {"params": other_p, "lr": 1e-4},
            {"params": trans_p, "lr": 5e-5}
        ], weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda ep: (ep + 1) / 5.0 if ep < 5 else 0.5 * (1.0 + math.cos(math.pi * (ep - 5) / max(1, total_epochs - 5)))
        )
    elif args.config == "B":
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    elif args.config == "C":
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-6)
    else:
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-6)

    best_auc, patience_cnt = 0.0, 0
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    suffix = f"_{args.input_rep}" if args.input_rep != "onehot" else ""
    best_path = os.path.join(args.checkpoint_dir, f"best_genexplorer_{args.task}_{args.config}{suffix}.pt")

    for epoch in range(1, total_epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in tr_loader:
            optimizer.zero_grad(set_to_none=True)
            x_b = batch["data"].to(device, non_blocking=True)
            y_b = batch["label"].to(device, non_blocking=True)
            mask_b = batch.get("mask", None)
            if mask_b is not None:
                mask_b = mask_b.to(device, non_blocking=True)
            len_b = batch.get("len_feat", None)
            if len_b is not None:
                len_b = len_b.to(device, non_blocking=True)

            out = model(x_b, mask=mask_b, lengths=len_b) if mask_b is not None else model(x_b)
            loss = criterion(out, y_b.long() if is_ce else y_b.float().unsqueeze(1))

            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y_b)

        if scheduler is not None:
            scheduler.step()

        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                x_b = batch["data"].to(device, non_blocking=True)
                mask_b = batch.get("mask", None)
                if mask_b is not None:
                    mask_b = mask_b.to(device, non_blocking=True)
                len_b = batch.get("len_feat", None)
                if len_b is not None:
                    len_b = len_b.to(device, non_blocking=True)

                out = model(x_b, mask=mask_b, lengths=len_b) if mask_b is not None else model(x_b)
                probs = F.softmax(out, dim=1)[:, 1] if is_ce else torch.sigmoid(out.view(-1))
                preds.extend(probs.cpu().numpy().flatten())
                targets.extend(batch["label"].numpy().flatten())

        preds_arr = np.nan_to_num(np.array(preds), nan=0.5)
        targets_arr = np.array(targets)

        val_auc = roc_auc_score(targets_arr, preds_arr) if len(np.unique(targets_arr)) > 1 else 0.5
        val_prc = average_precision_score(targets_arr, preds_arr) if len(np.unique(targets_arr)) > 1 else 0.0
        val_mcc = matthews_corrcoef(targets_arr, (preds_arr >= 0.5).astype(int)) if len(np.unique(targets_arr)) > 1 else 0.0
        print(f"Epoch {epoch:02d}/{total_epochs:02d} | Train Loss: {total_loss/len(tr_ds):.4f} | Val AUC: {val_auc:.4f} | Val PRC: {val_prc:.4f} | Val MCC: {val_mcc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            patience_cnt = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience_cnt += 1
            if args.config in ["A", "D"] and patience_cnt >= args.patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    print(f"[✓] Checkpoint saved: {best_path} (Best Val AUC: {best_auc:.4f})")


if __name__ == "__main__":
    main()
