#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dedicated training pipeline for FullE Complete 10-Subnetwork Ensemble (Eq. 3 in Paper).
Trains: 5 Branches (freq, freq_paper, pat, pat_paper, lp) + 5 Mergers (f+p+l, f+l, p+l, viraminer, viraminer_paper).
"""

import argparse
import math
import os
import sys
import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.fulle import FullEBranch, FullEMerger
from utils.data_utils import seed_everything, resolve_task_splits, SequenceClassificationDataset


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, smoothing: float = 0.05):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets_smooth = targets.unsqueeze(1) * (1 - self.smoothing) + 0.5 * self.smoothing
        bce = F.binary_cross_entropy_with_logits(logits.unsqueeze(1), targets_smooth, reduction="none")
        return (self.alpha * (1 - torch.exp(-bce)) ** self.gamma * bce).mean()


def train_single_module(model, tr_loader, val_loader, cfg, config_name, device, freeze_branches=False):
    is_ce = (config_name == "C")
    
    if freeze_branches and isinstance(model, FullEMerger):
        for b in model.branches:
            for p in b.parameters():
                p.requires_grad = False

    if cfg["loss"] == "focal":
        criterion = BinaryFocalLoss()
    elif cfg["loss"] == "bce":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if cfg["opt"] == "adamw":
        optimizer = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=1e-4)
    else:
        lr_base = 0.001 if config_name == "B" else 1.0
        optimizer = torch.optim.Adam(trainable_params, lr=lr_base)

    if cfg["sched"] == "cosine":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda ep: (ep + 1) / 5 if ep < 5 else 0.5 * (1 + math.cos(math.pi * (ep - 5) / (cfg["epochs"] - 5))))
    elif cfg["sched"] == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: (512**-0.5) * min(max(1, s)**-0.5, max(1, s) * (4000**-1.5)))

    best_auc, best_state = 0.0, None
    patience_cnt = 0

    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        for batch in tr_loader:
            x_b, y_b = batch["data"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            y_loss = y_b.long() if is_ce else y_b.float()
            out = model(x_b)
            loss = criterion(out, y_loss)
            loss.backward()
            optimizer.step()
            if cfg["sched"] == "aiayn":
                scheduler.step()

        if cfg["sched"] != "aiayn":
            scheduler.step()

        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                out = model(batch["data"].to(device))
                probs = F.softmax(out, dim=1)[:, 1] if is_ce else torch.sigmoid(out)
                preds.extend(probs.cpu().numpy().flatten())
                targets.extend(batch["label"].numpy().flatten())

        auc = roc_auc_score(targets, preds) if len(np.unique(targets)) > 1 else 0.5
        if auc >= best_auc:
            best_auc = auc
            patience_cnt = 0
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if cfg["early_stop"] and patience_cnt >= cfg["patience"]:
                break

    model.load_state_dict(best_state)
    return model, best_auc


def main():
    parser = argparse.ArgumentParser(description="FullE Complete 10-Subnetwork Training Pipeline")
    parser.add_argument("--config", type=str, required=True, choices=["A", "B", "C"])
    parser.add_argument("--task", type=str, default="viral", choices=["viral", "ecoli", "enhancer", "anellovirus"])
    parser.add_argument("--data_folder", default="/home/simionricc/datasets")
    parser.add_argument("--checkpoint_dir", default="./checkpoints")
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

    train_df, val_df, seq_len, _ = resolve_task_splits(args.task, args.data_folder, args.seed)
    print(f"[i] Training FullE Ensemble (10 Models) | Task: '{args.task}' | Config: '{args.config}' | Input: {seq_len} bp | Train: {len(train_df)} | Val: {len(val_df)}")

    tr_ds = SequenceClassificationDataset(train_df["sequence"].tolist(), train_df["label"].tolist(), "fulle", seq_len, args.config, is_train=True)
    val_ds = SequenceClassificationDataset(val_df["sequence"].tolist(), val_df["label"].tolist(), "fulle", seq_len, args.config, is_train=False)

    pin_mem = torch.cuda.is_available()
    tr_loader = DataLoader(tr_ds, batch_size=128, shuffle=True, num_workers=2, pin_memory=pin_mem)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=2, pin_memory=pin_mem)

    # 1. I 5 Rami Convoluzionali (Sezione 3.2.2)
    branches = {
        "freq": FullEBranch(in_channels=5, out_channels=1000, kernel_size=8, pool_type="avg", dropout=cfg["drop"], num_classes=cfg["out"]).to(device),
        "freq_paper": FullEBranch(in_channels=5, out_channels=1000, kernel_size=8, pool_type="avg", dropout=cfg["drop"], num_classes=cfg["out"]).to(device),
        "pat": FullEBranch(in_channels=5, out_channels=1200, kernel_size=11, pool_type="max", dropout=cfg["drop"], num_classes=cfg["out"]).to(device),
        "pat_paper": FullEBranch(in_channels=5, out_channels=1200, kernel_size=11, pool_type="max", dropout=cfg["drop"], num_classes=cfg["out"]).to(device),
        "lp": FullEBranch(in_channels=5, out_channels=1200, kernel_size=11, pool_type="lp", dropout=cfg["drop"], num_classes=cfg["out"]).to(device),
    }

    print("\n--- Training 5 Base Convolutional Branches ---")
    for name, branch_mod in branches.items():
        print(f"Training branch '{name}'...")
        branch_mod, score = train_single_module(branch_mod, tr_loader, val_loader, cfg, args.config, device)
        print(f" -> Branch '{name}' Best Val AUC: {score:.4f}")

    # 2. I 5 Merger Multi-Ramo (Sezione 3.2.2[cite: 9])
    mergers = {
        "merger_fpl": FullEMerger([branches["freq"], branches["pat"], branches["lp"]], dropout=cfg["drop"], num_classes=cfg["out"]).to(device),
        "merger_fl": FullEMerger([branches["freq"], branches["lp"]], dropout=cfg["drop"], num_classes=cfg["out"]).to(device),
        "merger_pl": FullEMerger([branches["pat"], branches["lp"]], dropout=cfg["drop"], num_classes=cfg["out"]).to(device),
        "merger_vm": FullEMerger([branches["freq"], branches["pat"]], dropout=cfg["drop"], num_classes=cfg["out"]).to(device),
        "merger_vm_paper": FullEMerger([branches["freq_paper"], branches["pat_paper"]], dropout=cfg["drop"], num_classes=cfg["out"]).to(device),
    }

    print("\n--- Training 5 Multi-Branch Mergers ---")
    for name, merger_mod in mergers.items():
        print(f"Training merger '{name}'...")
        merger_mod, m_score = train_single_module(merger_mod, tr_loader, val_loader, cfg, args.config, device, freeze_branches=True)
        print(f" -> Merger '{name}' Best Val AUC: {m_score:.4f}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(args.checkpoint_dir, f"best_fulle_{args.task}_{args.config}.pt")

    save_dict = {}
    for name, b in branches.items():
        save_dict[f"branch_{name}"] = b.state_dict()
    for name, m in mergers.items():
        save_dict[f"merger_{name}"] = m.state_dict()

    torch.save(save_dict, ckpt_path)
    print(f"\n[✓] FullE Complete 10-Model Checkpoint saved: {ckpt_path}")


if __name__ == "__main__":
    main()