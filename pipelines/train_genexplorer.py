#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dedicated training script for GenExplorer.
"""

import argparse
import math
import os
import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from models.genexplorer import GenExplorer
from utils.data_utils import seed_everything, resolve_task_splits, SequenceClassificationDataset


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, smoothing: float = 0.05):
        super().__init__()
        self.alpha, self.gamma, self.smoothing = alpha, gamma, smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets_smooth = targets.unsqueeze(1) * (1 - self.smoothing) + 0.5 * self.smoothing
        bce = F.binary_cross_entropy_with_logits(logits, targets_smooth, reduction="none")
        return (self.alpha * (1 - torch.exp(-bce)) ** self.gamma * bce).mean()


def main():
    parser = argparse.ArgumentParser(description="GenExplorer Training")
    parser.add_argument("--config", type=str, required=True, choices=["A", "B", "C", "D"])
    parser.add_argument("--task", type=str, default="viral", choices=["viral", "ecoli", "enhancer", "anellovirus"])
    parser.add_argument("--data_folder", default="/home/simionricc/datasets")
    parser.add_argument("--checkpoint_dir", default="./checkpoints")
    parser.add_argument("--seed", type=int, default=2020)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_map = {
        "A": {"epochs": 120, "early_stop": True, "patience": 15, "batch_size": 128, "loss": "focal", "opt": "adamw", "sched": "cosine", "drop": 0.25, "out": 1},
        "B": {"epochs": 30, "early_stop": False, "patience": 0, "batch_size": 128, "loss": "bce", "opt": "adam", "sched": "step", "drop": 0.2, "out": 1},
        "C": {"epochs": 60, "early_stop": False, "patience": 0, "batch_size": 128, "loss": "ce", "opt": "adam", "sched": "aiayn", "drop": 0.1, "out": 2},
        "D": {"epochs": 20, "early_stop": True, "patience": 8, "batch_size": 128, "loss": "ce", "opt": "adamw", "sched": "warmup_cosine", "drop": 0.5, "out": 2},
    }
    cfg = config_map[args.config]

    train_df, val_df, seq_len, _ = resolve_task_splits(args.task, args.data_folder, args.seed)
    print(f"[i] Training GenExplorer | Task: '{args.task}' | Config: '{args.config}' (Classes: {cfg['out']}) | Input: {seq_len} bp | Train: {len(train_df)} | Val: {len(val_df)}")

    tr_ds = SequenceClassificationDataset(train_df["sequence"].tolist(), train_df["label"].tolist(), "genexplorer", seq_len, args.config, is_train=True)
    val_ds = SequenceClassificationDataset(val_df["sequence"].tolist(), val_df["label"].tolist(), "genexplorer", seq_len, args.config, is_train=False)

    tr_loader = DataLoader(tr_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())

    model = GenExplorer(cnn_drop=cfg["drop"], trans_drop=0.1, out_features=cfg["out"], max_len=max(512, seq_len)).to(device)
    criterion = BinaryFocalLoss() if cfg["loss"] == "focal" else (nn.BCEWithLogitsLoss() if cfg["loss"] == "bce" else nn.CrossEntropyLoss())
    total_steps = len(tr_loader) * cfg["epochs"]

    if cfg["opt"] == "adamw":
        trans_p = list(model.transformer.parameters())
        other_p = list(model.pattern.parameters()) + list(model.frequency.parameters()) + list(model.fc.parameters())
        wd = 0.05 if args.config == "D" else 1e-4
        optimizer = torch.optim.AdamW([{"params": other_p, "lr": 1e-4}, {"params": trans_p, "lr": 5e-5}], weight_decay=wd)
    else:
        lr_base = 0.001 if args.config == "B" else 1.0
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_base)

    if cfg["sched"] == "cosine":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda ep: (ep + 1) / 5 if ep < 5 else 0.5 * (1 + math.cos(math.pi * (ep - 5) / (cfg["epochs"] - 5))))
    elif cfg["sched"] == "warmup_cosine":
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.10), num_training_steps=total_steps)
    elif cfg["sched"] == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: (256**-0.5) * min(max(1, s)**-0.5, max(1, s) * (4000**-1.5)))

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_path = os.path.join(args.checkpoint_dir, f"best_genexplorer_{args.task}_{args.config}.pt")
    best_auc, patience_cnt = 0.0, 0

    if args.config == "A":
        for p in model.transformer.parameters():
            p.requires_grad = False

    for epoch in range(1, cfg["epochs"] + 1):
        if args.config == "A" and epoch == 6:
            for p in model.transformer.parameters():
                p.requires_grad = True
            print("[i] Transformer unfreezed post-warmup.")

        model.train()
        total_loss = 0.0
        for batch in tr_loader:
            x_b, y_b = batch["data"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            out = model(x_b)
            loss = criterion(out, y_b.long() if cfg["loss"] == "ce" else (y_b.unsqueeze(1) if cfg["loss"] == "bce" else y_b))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.8 if args.config == "D" else 1.0)
            optimizer.step()
            if cfg["sched"] in ["aiayn", "warmup_cosine"]:
                scheduler.step()
            total_loss += loss.item() * len(x_b)

        if cfg["sched"] not in ["aiayn", "warmup_cosine"]:
            scheduler.step()

        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                out = model(batch["data"].to(device))
                probs = F.softmax(out, dim=1)[:, 1] if cfg["loss"] == "ce" else torch.sigmoid(out)
                preds.extend(probs.cpu().numpy().flatten())
                targets.extend(batch["label"].numpy().flatten())

        val_auc = roc_auc_score(targets, preds) if len(np.unique(targets)) > 1 else 0.5
        print(f"Epoch {epoch:03d}/{cfg['epochs']:03d} | Train Loss: {total_loss/len(tr_ds):.4f} | Val AUC: {val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            patience_cnt = 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "val_auc": val_auc}, best_path)
        else:
            patience_cnt += 1
            if cfg["early_stop"] and patience_cnt >= cfg["patience"]:
                print(f"Early stopping at epoch {epoch}.")
                break

    if not cfg["early_stop"]:
        torch.save({"model_state_dict": model.state_dict(), "epoch": cfg["epochs"], "val_auc": val_auc}, best_path)

    print(f"Checkpoint saved: {best_path} (Best Val AUC: {best_auc:.4f})")


if __name__ == "__main__":
    main()