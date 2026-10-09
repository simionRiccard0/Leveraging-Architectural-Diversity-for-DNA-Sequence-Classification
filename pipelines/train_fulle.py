#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import os
import gc
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.fulle import FullEBranch, FullEMerger, FullE10Ensemble
from utils.data_utils import seed_everything, resolve_task_splits, SequenceClassificationDataset


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, smoothing: float = 0.05):
        super().__init__()
        self.alpha, self.gamma, self.smoothing = alpha, gamma, smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets_smooth = targets.unsqueeze(1) * (1 - self.smoothing) + 0.5 * self.smoothing
        bce = F.binary_cross_entropy_with_logits(logits.unsqueeze(1), targets_smooth, reduction="none")
        return (self.alpha * (1 - torch.exp(-bce)) ** self.gamma * bce).mean()


def train_single_subnetwork(model, tr_loader, val_loader, cfg, config_name, device, freeze_branches=False, is_variable_len=False):
    is_ce = (config_name in ["C", "D"])
    if freeze_branches and isinstance(model, FullEMerger):
        for b in model.branches:
            for p in b.parameters():
                p.requires_grad = False

    criterion = BinaryFocalLoss() if cfg["loss"] == "focal" else (nn.BCEWithLogitsLoss() if cfg["loss"] == "bce" else nn.CrossEntropyLoss())
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if cfg["opt"] == "adamw":
        wd = 0.05 if config_name == "D" else 1e-4
        optimizer = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=wd)
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
            x_b = batch["data"].to(device, non_blocking=True)
            y_b = batch["label"].to(device, non_blocking=True)
            mask_b = batch["mask"].to(device, non_blocking=True) if ("mask" in batch and is_variable_len) else None
            len_b = batch.get("len_feat", None)
            if len_b is not None and is_variable_len:
                len_b = len_b.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            out = model(x_b, mask=mask_b, lengths=len_b) if mask_b is not None else model(x_b)
            loss = criterion(out, y_b.long() if is_ce else y_b.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.8 if config_name == "D" else 1.0)
            optimizer.step()

        scheduler.step()

        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                x_b = batch["data"].to(device, non_blocking=True)
                mask_b = batch["mask"].to(device, non_blocking=True) if ("mask" in batch and is_variable_len) else None
                len_b = batch.get("len_feat", None)
                if len_b is not None and is_variable_len:
                    len_b = len_b.to(device, non_blocking=True)

                out = model(x_b, mask=mask_b, lengths=len_b) if mask_b is not None else model(x_b)
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
    parser = argparse.ArgumentParser(description="FullE 10-Subnetwork Training Pipeline")
    parser.add_argument("--epochs", type=int, default=None, help="Override default epochs")
    parser.add_argument("--config", type=str, required=True, choices=["A", "B", "C", "D"])
    parser.add_argument("--task", type=str, default="viral")
    parser.add_argument("--data_folder", default=os.environ.get("DATA_DIR", "./datasets"))
    parser.add_argument("--checkpoint_dir", default="./checkpoints")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--input_rep", type=str, default="onehot", choices=["onehot", "physhelical"])
    parser.add_argument("--period_T", type=float, default=None)
    parser.add_argument("--use_length_feature", action="store_true")
    parser.add_argument("--seed", type=int, default=2020)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_map = {
        "A": {"epochs": 120, "early_stop": True, "patience": 15, "loss": "focal", "opt": "adamw", "sched": "cosine", "drop": 0.25, "out": 1},
        "B": {"epochs": 30, "early_stop": False, "patience": 0, "loss": "bce", "opt": "adam", "sched": "step", "drop": 0.2, "out": 1},
        "C": {"epochs": 60, "early_stop": False, "patience": 0, "loss": "ce", "opt": "adam", "sched": "aiayn", "drop": 0.1, "out": 2},
        "D": {"epochs": 20, "early_stop": True, "patience": 8, "loss": "ce", "opt": "adamw", "sched": "warmup_cosine", "drop": 0.5, "out": 2},
    }
    cfg = config_map[args.config]
    if args.epochs is not None:
        cfg["epochs"] = args.epochs

    is_variable_len = (args.task.lower() == "basith" or args.use_length_feature)
    in_channels = 4 if args.input_rep == "physhelical" else 5
    batch_size = args.batch_size if args.batch_size is not None else (16 if is_variable_len else 128)

    train_df, val_df, seq_len, _ = resolve_task_splits(args.task, args.data_folder, args.seed)
    print(f"[i] FullE Train | Task: '{args.task}' | Config: '{args.config}' (Classes: {cfg['out']}) | "
          f"Length: {seq_len} bp | Rep: {args.input_rep} ({in_channels} ch) | BS: {batch_size} | Masked Pooling: {is_variable_len}")

    tr_ds = SequenceClassificationDataset(
        train_df["sequence"].tolist(), train_df["label"].tolist(), "fulle", seq_len, args.config,
        is_train=True, task=args.task, input_rep=args.input_rep, period_T=args.period_T
    )
    val_ds = SequenceClassificationDataset(
        val_df["sequence"].tolist(), val_df["label"].tolist(), "fulle", seq_len, args.config,
        is_train=False, task=args.task, input_rep=args.input_rep, period_T=args.period_T
    )

    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())

    branches = {
        "freq": FullEBranch(in_channels=in_channels, out_channels=1000, kernel_size=8, pool_type="avg", dropout=cfg["drop"], num_classes=cfg["out"], use_length_feature=is_variable_len).to(device),
        "freq_paper": FullEBranch(in_channels=in_channels, out_channels=1000, kernel_size=8, pool_type="avg", dropout=cfg["drop"], num_classes=cfg["out"], use_length_feature=is_variable_len).to(device),
        "pat": FullEBranch(in_channels=in_channels, out_channels=1200, kernel_size=11, pool_type="max", dropout=cfg["drop"], num_classes=cfg["out"], use_length_feature=is_variable_len).to(device),
        "pat_paper": FullEBranch(in_channels=in_channels, out_channels=1200, kernel_size=11, pool_type="max", dropout=cfg["drop"], num_classes=cfg["out"], use_length_feature=is_variable_len).to(device),
        "lp": FullEBranch(in_channels=in_channels, out_channels=1200, kernel_size=11, pool_type="lp", dropout=cfg["drop"], num_classes=cfg["out"], use_length_feature=is_variable_len).to(device),
    }

    print("\n--- Step 1/2: Training 5 Base Branches ---")
    for name, b in branches.items():
        gc.collect()
        torch.cuda.empty_cache()
        b, score = train_single_subnetwork(b, tr_loader, val_loader, cfg, args.config, device, is_variable_len=is_variable_len)
        print(f" -> Branch '{name}' Best Val AUC: {score:.4f}")

    mergers = {
        "merger_fpl": FullEMerger([branches["freq"], branches["pat"], branches["lp"]], dropout=cfg["drop"], num_classes=cfg["out"], use_length_feature=is_variable_len).to(device),
        "merger_fl": FullEMerger([branches["freq"], branches["lp"]], dropout=cfg["drop"], num_classes=cfg["out"], use_length_feature=is_variable_len).to(device),
        "merger_pl": FullEMerger([branches["pat"], branches["lp"]], dropout=cfg["drop"], num_classes=cfg["out"], use_length_feature=is_variable_len).to(device),
        "merger_vm": FullEMerger([branches["freq"], branches["pat"]], dropout=cfg["drop"], num_classes=cfg["out"], use_length_feature=is_variable_len).to(device),
        "merger_vm_paper": FullEMerger([branches["freq_paper"], branches["pat_paper"]], dropout=cfg["drop"], num_classes=cfg["out"], use_length_feature=is_variable_len).to(device),
    }

    print("\n--- Step 2/2: Training 5 Multi-Branch Mergers ---")
    for name, m in mergers.items():
        gc.collect()
        torch.cuda.empty_cache()
        m, score = train_single_subnetwork(m, tr_loader, val_loader, cfg, args.config, device, freeze_branches=True, is_variable_len=is_variable_len)
        print(f" -> Merger '{name}' Best Val AUC: {score:.4f}")

    ensemble = FullE10Ensemble(branches, mergers, num_classes=cfg["out"]).to(device).eval()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    rep_suffix = f"_{args.input_rep}" if args.input_rep != "onehot" else ""
    ckpt_path = os.path.join(args.checkpoint_dir, f"best_fulle_{args.task}_{args.config}{rep_suffix}.pt")
    save_dict = {
        "model_state_dict": ensemble.state_dict(),
        "branches": {k: v.state_dict() for k, v in branches.items()},
        "mergers": {k: v.state_dict() for k, v in mergers.items()}
    }
    torch.save(save_dict, ckpt_path)
    print(f"\n[✓] Checkpoint Ensemble (10 reti) salvato: {ckpt_path}")


if __name__ == "__main__":
    main()