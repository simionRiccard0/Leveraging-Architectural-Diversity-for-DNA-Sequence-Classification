#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dedicated training pipeline for FullE Convolutional Ensemble (Branches + Mergers)[cite: 4].
Decoupled from inference: saves trained sub-network weights to checkpoint_dir[cite: 4].
"""

import argparse
import os
import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.fulle import FullEBranch, FullEMerger
from utils.data_utils import seed_everything, resolve_task_splits, SequenceClassificationDataset


def train_single_module(model, tr_loader, val_loader, epochs, lr, device):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_auc, best_state = 0.0, None

    for ep in range(1, epochs + 1):
        model.train()
        for batch in tr_loader:
            x_b, y_b = batch["data"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            out = model(x_b).squeeze(-1)
            loss = criterion(out, y_b)
            loss.backward()
            optimizer.step()

        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                out = model(batch["data"].to(device)).squeeze(-1)
                preds.extend(torch.sigmoid(out).cpu().numpy())
                targets.extend(batch["label"].numpy())

        auc = roc_auc_score(targets, preds) if len(np.unique(targets)) > 1 else 0.5
        if auc >= best_auc:
            best_auc = auc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, best_auc


def main():
    parser = argparse.ArgumentParser(description="FullE Modular Ensemble Training Pipeline")
    parser.add_argument("--config", type=str, required=True, choices=["A", "B", "C"])
    parser.add_argument("--task", type=str, default="viral", choices=["viral", "ecoli", "enhancer", "anellovirus"])
    parser.add_argument("--data_folder", default="/home/simionricc/datasets")
    parser.add_argument("--checkpoint_dir", default="./checkpoints")
    parser.add_argument("--seed", type=int, default=2020)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_map = {
        "A": {"epochs": 40, "lr": 1e-4, "drop": 0.25},
        "B": {"epochs": 30, "lr": 1e-3, "drop": 0.2},
        "C": {"epochs": 30, "lr": 1e-3, "drop": 0.1}
    }
    cfg = config_map[args.config]

    train_df, val_df, seq_len, _ = resolve_task_splits(args.task, args.data_folder, args.seed)
    print(f"[i] Training FullE Ensemble | Task: '{args.task}' | Config: '{args.config}' | Input: {seq_len} bp | Train: {len(train_df)} | Val: {len(val_df)}")

    tr_ds = SequenceClassificationDataset(train_df["sequence"].tolist(), train_df["label"].tolist(), "fulle", seq_len, args.config, is_train=True)
    val_ds = SequenceClassificationDataset(val_df["sequence"].tolist(), val_df["label"].tolist(), "fulle", seq_len, args.config, is_train=False)

    pin_mem = torch.cuda.is_available()
    tr_loader = DataLoader(tr_ds, batch_size=128, shuffle=True, num_workers=2, pin_memory=pin_mem)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=2, pin_memory=pin_mem)

    # 1. Base Convolutional Branches
    branches = {
        "freq": FullEBranch(in_channels=5, out_channels=1000, kernel_size=8, pool_type="avg", dropout=cfg["drop"]).to(device),
        "pat": FullEBranch(in_channels=5, out_channels=1200, kernel_size=11, pool_type="max", dropout=cfg["drop"]).to(device),
        "lp": FullEBranch(in_channels=5, out_channels=1200, kernel_size=11, pool_type="lp", dropout=cfg["drop"]).to(device),
    }

    print("\n--- Training Base Convolutional Branches ---")
    for name, branch_mod in branches.items():
        print(f"Training branch '{name}'...")
        branch_mod, score = train_single_module(branch_mod, tr_loader, val_loader, cfg["epochs"], cfg["lr"], device)
        print(f" -> Branch '{name}' Best Val AUC: {score:.4f}")

    # 2. Multi-Branch Merger Network
    print("\n--- Training Multi-Branch Merger ---")
    merger_full = FullEMerger([branches["freq"], branches["pat"], branches["lp"]], dropout=cfg["drop"]).to(device)
    merger_full, m_score = train_single_module(merger_full, tr_loader, val_loader, cfg["epochs"], cfg["lr"], device)
    print(f" -> Full Merger Best Val AUC: {m_score:.4f}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(args.checkpoint_dir, f"best_fulle_{args.task}_{args.config}.pt")

    torch.save({
        "branch_freq": branches["freq"].state_dict(),
        "branch_pat": branches["pat"].state_dict(),
        "branch_lp": branches["lp"].state_dict(),
        "merger_full": merger_full.state_dict()
    }, ckpt_path)

    print(f"FullE checkpoint saved successfully: {ckpt_path}")


if __name__ == "__main__":
    main()