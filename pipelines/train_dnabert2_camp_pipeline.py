#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Integrated training and cascade evaluation pipeline for DNABERT2-CAMP.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.dnabert2_camp import DNABERT2CAMP
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
        p_t = targets * probs + (1.0 - targets) * (1.0 - probs)
        alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
        return (alpha_t * ((1.0 - p_t) ** self.gamma) * bce).mean()


def get_default_model_path():
    return os.environ.get("DNABERT2_MODEL_PATH", "zhihan1996/DNABERT-2-117M")


def evaluate_model(model, dataloader, is_ce, device):
    model.eval()
    all_logits, all_targets = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            logits = model(input_ids, mask)
            logits_val = (logits[:, 1] - logits[:, 0]) if is_ce else logits.view(-1)
            all_logits.extend(logits_val.cpu().numpy().flatten())
            all_targets.extend(labels.cpu().numpy().flatten())

    logits_arr = np.nan_to_num(np.array(all_logits), nan=0.0)
    targets_arr = np.array(all_targets).astype(int)
    probs_arr = 1.0 / (1.0 + np.exp(-logits_arr))

    auc = roc_auc_score(targets_arr, probs_arr) if len(np.unique(targets_arr)) > 1 else 0.5
    prc = average_precision_score(targets_arr, probs_arr) if len(np.unique(targets_arr)) > 1 else 0.0
    mcc = matthews_corrcoef(targets_arr, (probs_arr >= 0.5).astype(int)) if len(np.unique(targets_arr)) > 1 else 0.0
    return auc, prc, mcc, logits_arr


def main():
    parser = argparse.ArgumentParser(description="DNABERT2-CAMP Fine-tuning & Cascade Pipeline")
    parser.add_argument("--config", type=str, required=True, choices=["A", "B", "C", "D"])
    parser.add_argument("--task", type=str, default="viral", choices=["viral", "ecoli", "enhancer", "anellovirus"])
    parser.add_argument("--data_folder", default=os.environ.get("DATA_DIR", "./datasets"), help="Path to dataset folder (default: './datasets' or value of DATA_DIR environment variable)")
    parser.add_argument("--model_name_or_path", default=get_default_model_path())
    parser.add_argument("--checkpoint_dir", default="./checkpoints")
    parser.add_argument("--logits_dir", default="./global_logits_monolithic")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--backbone_lr", type=float, default=2e-5)
    parser.add_argument("--head_lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--patience", type=int, default=8)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs_map = {"A": 40, "B": 20, "C": 30, "D": 20}
    total_epochs = epochs_map[args.config]
    is_ce = (args.config in ["C", "D"])
    num_classes = 2 if is_ce else 1

    train_df, val_df, seq_len, cascade_dict = resolve_task_splits(args.task, args.data_folder, args.seed)
    print(f"[i] DNABERT2-CAMP Pipeline | Task: '{args.task}' | Config: '{args.config}' (Classes: {num_classes}) | Model: {args.model_name_or_path} | Train: {len(train_df)} | Val: {len(val_df)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    tr_ds = SequenceClassificationDataset(train_df["sequence"].tolist(), train_df["label"].tolist(), "dnabert2camp", seq_len, args.config, is_train=True, tokenizer=tokenizer)
    val_ds = SequenceClassificationDataset(val_df["sequence"].tolist(), val_df["label"].tolist(), "dnabert2camp", seq_len, args.config, is_train=False, tokenizer=tokenizer)

    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())

    model = DNABERT2CAMP(model_name_or_path=args.model_name_or_path, num_classes=num_classes).to(device)

    wd_val = 0.05 if args.config == "D" else 0.01
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    param_groups = [
        {"params": [p for n, p in model.bert.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad], "lr": args.backbone_lr, "weight_decay": wd_val},
        {"params": [p for n, p in model.bert.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad], "lr": args.backbone_lr, "weight_decay": 0.0},
        {"params": [p for n, p in model.classifier.named_parameters() if p.requires_grad], "lr": args.head_lr, "weight_decay": wd_val},
        {"params": [p for n, p in model.camp.named_parameters() if p.requires_grad], "lr": args.head_lr, "weight_decay": wd_val},
    ]

    total_steps = len(tr_loader) * total_epochs

    if args.config == "A":
        criterion = BinaryFocalLoss(alpha=0.75, gamma=2.0)
        optimizer = torch.optim.AdamW(param_groups, eps=1e-8)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.10), num_training_steps=total_steps)
    elif args.config == "B":
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(param_groups, eps=1e-8)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, len(tr_loader) * 5), gamma=0.5)
    elif args.config == "C":
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(param_groups, eps=1e-8)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.10), num_training_steps=total_steps)
    else:
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(param_groups, lr=args.backbone_lr, weight_decay=0.05, eps=1e-8)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.10), num_training_steps=total_steps)

    best_auc, patience_cnt = 0.0, 0
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_path = os.path.join(args.checkpoint_dir, f"best_dnabert2camp_{args.task}_{args.config}.pt")

    for epoch in range(1, total_epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in tr_loader:
            optimizer.zero_grad(set_to_none=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            logits = model(input_ids, mask)
            loss = criterion(logits, labels.view(-1).long() if is_ce else labels.view(-1).float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.8 if args.config == "D" else 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item() * len(input_ids)

        val_auc, val_prc, val_mcc, _ = evaluate_model(model, val_loader, is_ce, device)
        print(f"Epoch {epoch:02d}/{total_epochs:02d} | Train Loss: {total_loss/len(tr_ds):.4f} | Val AUC: {val_auc:.4f} | Val PRC: {val_prc:.4f} | Val MCC: {val_mcc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            patience_cnt = 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "val_auc": val_auc}, best_path)
        else:
            patience_cnt += 1
            if args.config in ["A", "D"] and patience_cnt >= args.patience:
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
        eval_ds = SequenceClassificationDataset(eval_df["sequence"].tolist(), eval_df["label"].tolist(), "dnabert2camp", seq_len, args.config, is_train=False, tokenizer=tokenizer)
        eval_loader = DataLoader(eval_ds, batch_size=args.batch_size * 2, shuffle=False)

        auc, prc, mcc, raw_logits = evaluate_model(model, eval_loader, is_ce, device)
        print(f" -> [{d_name}] AUROC: {auc:.4f} | AUPRC: {prc:.4f} | MCC: {mcc:.4f}")
        out_fpath = os.path.join(args.logits_dir, f"dnabert2camp_cfg{args.config}_{args.task}_{d_name}.txt")
        np.savetxt(out_fpath, raw_logits, fmt="%.6f")
        print(f"    Logits saved: {out_fpath}")


if __name__ == "__main__":
    main()