#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dedicated training pipeline for Nucleic Transformer (C1/A, C2/B, C3/C strategies)[cite: 24].
Decoupled from inference: saves best validated model to checkpoint_dir[cite: 24].
"""

import argparse
import math
import os
import time
import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from models.nucleic_transformer import NucleicTransformer
from utils.data_utils import seed_everything, resolve_task_splits, SequenceClassificationDataset


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, smoothing: float = 0.05):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets_smooth = targets.float() * (1 - self.smoothing) + 0.5 * self.smoothing
        bce = F.binary_cross_entropy_with_logits(logits.float(), targets_smooth, reduction="none")
        bce = torch.clamp(bce, min=1e-7, max=50.0)
        return (self.alpha * (1 - torch.exp(-bce)) ** self.gamma * bce).mean()


def main():
    parser = argparse.ArgumentParser(description="Nucleic Transformer Training Pipeline")
    parser.add_argument("--config", type=str, required=True, choices=["A", "B", "C"], help="Strategy: A (C1), B (C2), C (C3)")
    parser.add_argument("--task", type=str, default="viral", choices=["viral", "ecoli", "enhancer", "anellovirus"])
    parser.add_argument("--data_folder", default="/home/simionricc/datasets")
    parser.add_argument("--checkpoint_dir", default="./checkpoints")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--ninp", type=int, default=512)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--nhid", type=int, default=2048)
    parser.add_argument("--nlayers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=2020)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_map = {
        "A": {"epochs": 120, "early_stop": True, "patience": 15, "loss": "focal", "opt": "adamw", "sched": "cosine", "drop": 0.1, "nclass": 1},
        "B": {"epochs": 30, "early_stop": False, "patience": 0, "loss": "bce", "opt": "adam", "sched": "step", "drop": 0.2, "nclass": 1},
        "C": {"epochs": 60, "early_stop": False, "patience": 0, "loss": "ce", "opt": "adam", "sched": "aiayn", "drop": 0.1, "nclass": 2},
    }
    cfg = config_map[args.config]

    train_df, val_df, seq_len, _ = resolve_task_splits(args.task, args.data_folder, args.seed)
    print(f"[i] Training Nucleic Transformer | Task: '{args.task}' | Config: '{args.config}' | Input: {seq_len} bp | Train: {len(train_df)} | Val: {len(val_df)}")

    tr_ds = SequenceClassificationDataset(train_df["sequence"].tolist(), train_df["label"].tolist(), "nucleictransformer", seq_len, args.config, is_train=True)
    val_ds = SequenceClassificationDataset(val_df["sequence"].tolist(), val_df["label"].tolist(), "nucleictransformer", seq_len, args.config, is_train=False)

    pin_mem = torch.cuda.is_available()
    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=pin_mem)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2, pin_memory=pin_mem)

    model = NucleicTransformer(
        ntoken=4,
        nclass=cfg["nclass"],
        ninp=args.ninp,
        nhead=args.nhead,
        nhid=args.nhid,
        nlayers=args.nlayers,
        dropout=cfg["drop"]
    ).to(device)

    if cfg["loss"] == "focal":
        criterion = BinaryFocalLoss()
    elif cfg["loss"] == "bce":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss(reduction="none")

    if cfg["opt"] == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    else:
        lr_base = 0.001 if args.config == "B" else 1.0
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_base, weight_decay=0, eps=1e-8)

    if cfg["sched"] == "cosine":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda ep: (ep + 1) / 5 if ep < 5 else 0.5 * (1 + math.cos(math.pi * (ep - 5) / (cfg["epochs"] - 5))))
    elif cfg["sched"] == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    else:
        warmup = 3200
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: (args.ninp**-0.5) * min(max(1, s)**-0.5, max(1, s) * (warmup**-1.5)) * 0.1)

    scaler = GradScaler()
    best_auc, patience_cnt = 0.0, 0
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_path = os.path.join(args.checkpoint_dir, f"best_nucleictransformer_{args.task}_{args.config}.pt")

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        total_loss = 0.0
        t0 = time.time()

        for batch in tr_loader:
            src = batch["data"].to(device, non_blocking=True).long()
            labels = batch["label"].to(device, non_blocking=True)
            y_loss = labels.long() if cfg["loss"] == "ce" else labels.float().unsqueeze(1)

            optimizer.zero_grad()
            with autocast():
                output = model(src).float()
                loss = torch.mean(criterion(output, y_loss)) if cfg["loss"] == "ce" else criterion(output, y_loss)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            # Corretto: aggiornamento scheduler per Noam/AIAYN dopo step dell'ottimizzatore
            if cfg["sched"] == "aiayn":
                scheduler.step()

        if cfg["sched"] != "aiayn":
            scheduler.step()

        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                src = batch["data"].to(device, non_blocking=True).long()
                out = model(src).float()
                probs = F.softmax(out, dim=1)[:, 1] if cfg["loss"] == "ce" else torch.sigmoid(out)
                probs_np = np.nan_to_num(probs.cpu().numpy().flatten(), nan=0.5)
                preds.extend(probs_np)
                targets.extend(batch["label"].numpy().flatten())

        val_auc = roc_auc_score(targets, preds) if len(np.unique(targets)) > 1 else 0.5
        elapsed = time.time() - t0
        print(f"Epoch {epoch:02d}/{cfg['epochs']:02d} | Train Loss: {total_loss/len(tr_loader):.4f} | Val AUC: {val_auc:.4f} | Time: {elapsed:.1f}s")

        if val_auc > best_auc:
            best_auc = val_auc
            patience_cnt = 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "val_auc": val_auc}, best_path)
        else:
            patience_cnt += 1
            if cfg["early_stop"] and patience_cnt >= cfg["patience"]:
                print("Early stopping triggered.")
                break

    if not cfg["early_stop"]:
        torch.save({"model_state_dict": model.state_dict(), "epoch": cfg["epochs"], "val_auc": val_auc}, best_path)

    print(f"Checkpoint saved successfully: {best_path} (Best Val AUC: {best_auc:.4f})")


if __name__ == "__main__":
    main()