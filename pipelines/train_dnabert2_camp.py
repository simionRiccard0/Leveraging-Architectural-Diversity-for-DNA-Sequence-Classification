#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dedicated fine-tuning pipeline for DNABERT2-CAMP genomic foundation model.
Decoupled from inference: saves best model checkpoint to checkpoint_dir.
"""

import argparse
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from models.dnabert2_camp import DNABERT2CAMP
from utils.data_utils import seed_everything, resolve_task_splits, SequenceClassificationDataset


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1.0 - targets) * (1.0 - probs)
        alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
        return (alpha_t * ((1.0 - p_t) ** self.gamma) * bce).mean()


def main():
    parser = argparse.ArgumentParser(description="DNABERT2-CAMP Fine-tuning Pipeline")
    parser.add_argument("--config", type=str, required=True, choices=["A", "B", "C"], help="Strategy: A (C1), B (C2), C (C3)")
    parser.add_argument("--task", type=str, default="viral", choices=["viral", "ecoli", "enhancer", "anellovirus"])
    parser.add_argument("--data_folder", default="/home/simionricc/datasets")
    parser.add_argument("--model_name_or_path", default="zhihan1996/DNABERT-2-117M")
    parser.add_argument("--checkpoint_dir", default="./checkpoints")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--backbone_lr", type=float, default=2e-5)
    parser.add_argument("--head_lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--patience", type=int, default=8)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs_map = {"A": 40, "B": 20, "C": 30}
    total_epochs = epochs_map[args.config]

    train_df, val_df, seq_len, _ = resolve_task_splits(args.task, args.data_folder, args.seed)
    print(f"[i] Fine-tuning DNABERT2-CAMP | Task: '{args.task}' | Config: '{args.config}' | Input: {seq_len} bp | Train: {len(train_df)} | Val: {len(val_df)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    tr_ds = SequenceClassificationDataset(train_df["sequence"].tolist(), train_df["label"].tolist(), "dnabert2camp", seq_len, args.config, is_train=True, tokenizer=tokenizer)
    val_ds = SequenceClassificationDataset(val_df["sequence"].tolist(), val_df["label"].tolist(), "dnabert2camp", seq_len, args.config, is_train=False, tokenizer=tokenizer)

    pin_mem = torch.cuda.is_available()
    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=pin_mem)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=2, pin_memory=pin_mem)

    model = DNABERT2CAMP(model_name_or_path=args.model_name_or_path).to(device)

    # Differential Learning Rate Setup
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    param_groups = [
        {"params": [p for n, p in model.bert.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad], "lr": args.backbone_lr, "weight_decay": 0.01},
        {"params": [p for n, p in model.bert.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad], "lr": args.backbone_lr, "weight_decay": 0.0},
        {"params": [p for n, p in model.classifier.named_parameters() if p.requires_grad], "lr": args.head_lr, "weight_decay": 0.01},
        {"params": [p for n, p in model.camp.named_parameters() if p.requires_grad], "lr": args.head_lr, "weight_decay": 0.01},
    ]

    total_steps = len(tr_loader) * total_epochs

    if args.config == "A":
        criterion = BinaryFocalLoss(alpha=0.75, gamma=2.0)
        optimizer = torch.optim.AdamW(param_groups, eps=1e-8)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.10), num_training_steps=total_steps)
    elif args.config == "B":
        num_pos = sum(train_df["label"].values)
        pos_weight = torch.tensor([(len(train_df) - num_pos) / max(num_pos, 1)], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(param_groups, eps=1e-8)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, len(tr_loader) * 5), gamma=0.5)
    else:  # Config C
        num_pos = sum(train_df["label"].values)
        pos_weight = torch.tensor([(len(train_df) - num_pos) / max(num_pos, 1)], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(param_groups, eps=1e-8)
        warmup_steps = max(1, int(total_steps * 0.10))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: (float(s + 1) / float(warmup_steps)) if (s < warmup_steps) else ((float(warmup_steps) / float(s + 1)) ** 0.5))

    scaler = GradScaler()
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

            with autocast():
                logits = model(input_ids, mask)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item() * len(input_ids)

        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                mask = batch["attention_mask"].to(device, non_blocking=True)
                logits = model(input_ids, mask)
                probs = torch.sigmoid(logits)
                preds.extend(probs.cpu().numpy().flatten())
                targets.extend(batch["label"].numpy().flatten())

        val_auc = roc_auc_score(targets, preds) if len(np.unique(targets)) > 1 else 0.5
        val_prc = average_precision_score(targets, preds) if len(np.unique(targets)) > 1 else 0.0
        val_mcc = matthews_corrcoef(targets, (np.array(preds) >= 0.5).astype(int)) if len(np.unique(targets)) > 1 else 0.0
        print(f"Epoch {epoch:02d}/{total_epochs:02d} | Train Loss: {total_loss/len(tr_ds):.4f} | Val AUC: {val_auc:.4f} | Val PRC: {val_prc:.4f} | Val MCC: {val_mcc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            patience_cnt = 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "val_auc": val_auc}, best_path)
        else:
            patience_cnt += 1
            if args.config == "A" and patience_cnt >= args.patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    print(f"Checkpoint saved successfully: {best_path} (Best Val AUC: {best_auc:.4f})")


if __name__ == "__main__":
    main()