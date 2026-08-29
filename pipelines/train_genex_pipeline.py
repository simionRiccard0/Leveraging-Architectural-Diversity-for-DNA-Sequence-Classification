#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Integrated training and cascade evaluation pipeline for GenExplorer.
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

from models.genexplorer import GenExplorer
from utils.data_utils import seed_everything, resolve_task_splits, load_seq_label_dataframe, SequenceClassificationDataset


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, smoothing: float = 0.05):
        super().__init__()
        self.alpha, self.gamma, self.smoothing = alpha, gamma, smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.view(-1)
        targets = targets.view(-1)
        targets_smooth = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        bce = F.binary_cross_entropy_with_logits(logits, targets_smooth, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1 - targets) * (1 - probs)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        return (alpha_t * ((1 - p_t) ** self.gamma) * bce).mean()


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def evaluate_model(model, dataloader, is_ce, device):
    model.eval()
    all_logits, all_targets = [], []
    with torch.no_grad():
        for batch in dataloader:
            x_b = batch["data"].to(device, non_blocking=True)
            y_b = batch["label"].to(device, non_blocking=True)
            out = model(x_b)
            logits_val = (out[:, 1] - out[:, 0]) if is_ce else out.view(-1)
            all_logits.extend(logits_val.cpu().numpy().flatten())
            all_targets.extend(y_b.cpu().numpy().flatten())

    logits_arr = np.array(all_logits)
    targets_arr = np.array(all_targets).astype(int)
    probs_arr = 1.0 / (1.0 + np.exp(-logits_arr))

    auc = roc_auc_score(targets_arr, probs_arr) if len(np.unique(targets_arr)) > 1 else 0.5
    prc = average_precision_score(targets_arr, probs_arr) if len(np.unique(targets_arr)) > 1 else 0.0
    mcc = matthews_corrcoef(targets_arr, (probs_arr >= 0.5).astype(int)) if len(np.unique(targets_arr)) > 1 else 0.0
    return auc, prc, mcc, logits_arr


def main():
    parser = argparse.ArgumentParser(description="GenExplorer Training & Cascade Pipeline")
    parser.add_argument("--config", type=str, required=True, choices=["A", "B", "C", "D"])
    parser.add_argument("--task", type=str, default="viral", choices=["viral", "ecoli", "enhancer", "anellovirus"])
    parser.add_argument("--data_folder", default="/home/simionricc/datasets")
    parser.add_argument("--checkpoint_dir", default="./checkpoints")
    parser.add_argument("--logits_dir", default="./global_logits")
    parser.add_argument("--seed", type=int, default=2020)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_map = {
        "A": {"epochs": 120, "early_stop": True, "patience": 15, "loss": "focal", "opt": "adamw", "sched": "cosine", "drop": 0.25, "trans_drop": 0.1, "out": 1},
        "B": {"epochs": 30, "early_stop": False, "patience": 0, "loss": "bce", "opt": "adam", "sched": "step", "drop": 0.2, "trans_drop": 0.2, "out": 1},
        "C": {"epochs": 60, "early_stop": False, "patience": 0, "loss": "ce", "opt": "adam", "sched": "aiayn", "drop": 0.1, "trans_drop": 0.1, "out": 2},
        "D": {"epochs": 20, "early_stop": True, "patience": 8, "loss": "ce", "opt": "adamw", "sched": "warmup_cosine", "drop": 0.5, "trans_drop": 0.1, "out": 2},
    }
    cfg = config_map[args.config]
    is_ce = (args.config in ["C", "D"])

    train_df, val_df, seq_len, cascade_dict = resolve_task_splits(args.task, args.data_folder, args.seed)
    print(f"[i] GenExplorer Pipeline | Task: '{args.task}' | Config: '{args.config}' (Classes: {cfg['out']}) | Length: {seq_len} bp | Train: {len(train_df)} | Val: {len(val_df)}")

    tr_ds = SequenceClassificationDataset(train_df["sequence"].tolist(), train_df["label"].tolist(), "genexplorer", seq_len, args.config, is_train=True)
    val_ds = SequenceClassificationDataset(val_df["sequence"].tolist(), val_df["label"].tolist(), "genexplorer", seq_len, args.config, is_train=False)

    tr_loader = DataLoader(tr_ds, batch_size=128, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())

    model = GenExplorer(cnn_drop=cfg["drop"], trans_drop=cfg["trans_drop"], out_features=cfg["out"], max_len=max(512, seq_len)).to(device)
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
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: (512**-0.5) * min(max(1, s)**-0.5, max(1, s) * (4000**-1.5)))

    best_auc, patience_cnt = 0.0, 0
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_path = os.path.join(args.checkpoint_dir, f"best_genexplorer_{args.task}_{args.config}.pt")

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
            optimizer.zero_grad(set_to_none=True)
            x_b = batch["data"].to(device, non_blocking=True)
            y_b = batch["label"].to(device, non_blocking=True)

            out = model(x_b)
            loss = criterion(out, y_b.view(-1).long() if is_ce else y_b.view(-1).float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.8 if args.config == "D" else 1.0)
            optimizer.step()

            if cfg["sched"] in ["aiayn", "warmup_cosine"]:
                scheduler.step()
            total_loss += loss.item() * len(x_b)

        if cfg["sched"] not in ["aiayn", "warmup_cosine"]:
            scheduler.step()

        val_auc, val_prc, val_mcc, _ = evaluate_model(model, val_loader, is_ce, device)
        print(f"Epoch {epoch:03d}/{cfg['epochs']:03d} | Loss: {total_loss/len(tr_ds):.4f} | Val AUC: {val_auc:.4f} | Val PRC: {val_prc:.4f} | Val MCC: {val_mcc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            patience_cnt = 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "val_auc": val_auc}, best_path)
        else:
            patience_cnt += 1
            if cfg["early_stop"] and patience_cnt >= cfg["patience"]:
                print(f"Early stopping at epoch {epoch}.")
                break

    print(f"\nTraining complete. Best Checkpoint: {best_path} (AUC: {best_auc:.4f})")
    model.load_state_dict(torch.load(best_path, map_location=device)["model_state_dict"])

    os.makedirs(args.logits_dir, exist_ok=True)
    print("\n--- Running Cascade Evaluation ---")
    for d_name, d_path in cascade_dict.items():
        if not os.path.exists(d_path):
            continue
        eval_df = load_seq_label_dataframe(d_path)
        eval_ds = SequenceClassificationDataset(eval_df["sequence"].tolist(), eval_df["label"].tolist(), "genexplorer", seq_len, args.config, is_train=False)
        eval_loader = DataLoader(eval_ds, batch_size=128, shuffle=False)

        auc, prc, mcc, raw_logits = evaluate_model(model, eval_loader, is_ce, device)
        print(f" -> [{d_name}] AUROC: {auc:.4f} | AUPRC: {prc:.4f} | MCC: {mcc:.4f}")
        out_fpath = os.path.join(args.logits_dir, f"genexplorer_cfg{args.config}_{args.task}_{d_name}.txt")
        np.savetxt(out_fpath, raw_logits, fmt="%.6f")
        print(f"    Logits saved: {out_fpath}")


if __name__ == "__main__":
    main()