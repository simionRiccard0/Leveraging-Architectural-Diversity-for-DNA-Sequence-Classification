#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Training Pipeline for Nucleic Transformer across Configs A, B, C, D.
"""

import os
import sys
import copy
import math
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, matthews_corrcoef

from models.nucleic_transformer import NucleicTransformer
from utils.data_utils import resolve_task_splits, SequenceClassificationDataset


# -----------------------------------------------------------------------------
# Loss Functions
# -----------------------------------------------------------------------------

class BinaryFocalLoss(nn.Module):
    """Focal Loss numericamente stabile (forza il calcolo in FP32)."""
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, smoothing: float = 0.05):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.float().view(-1)
        targets = targets.float().view(-1)

        targets_smooth = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        bce = F.binary_cross_entropy_with_logits(logits, targets_smooth, reduction="none")
        probs = torch.sigmoid(logits)

        p_t = targets * probs + (1.0 - targets) * (1.0 - probs)
        alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
        modulating_factor = (1.0 - p_t).clamp(min=0.0, max=1.0) ** self.gamma

        return (alpha_t * modulating_factor * bce).mean()


# -----------------------------------------------------------------------------
# Learning Rate Schedulers
# -----------------------------------------------------------------------------

class lr_AIAYN:
    """Scheduler Noam (Attention Is All You Need) per Config C, come in LrScheduler.py."""
    def __init__(self, optimizer, d_model: int = 512, warmup_steps: int = 3200, factor: float = 0.1):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = (self.d_model ** -0.5) * min(
            self.step_num ** -0.5,
            self.step_num * (self.warmup_steps ** -1.5)
        ) * self.factor
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class WarmupThenPlateau:
    """
    Wrapper per Config B: warmup lineare per warmup_steps step, poi passa il controllo
    a ReduceLROnPlateau (chiamato manualmente a fine epoca su val_auc).
    Risolve il collasso a AUC 0.5 osservato quando lr=1e-3 veniva applicato di colpo
    dall'epoca 1 su un transformer Post-LN.
    """
    def __init__(self, optimizer, base_lr: float, warmup_steps: int, plateau_kwargs: dict):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.warmup_steps = max(1, warmup_steps)
        self.step_num = 0
        self.warmup_done = False
        self.plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **plateau_kwargs)
        for pg in self.optimizer.param_groups:
            pg['lr'] = 0.0

    def step_batch(self):
        if self.step_num < self.warmup_steps:
            self.step_num += 1
            lr = self.base_lr * (self.step_num / self.warmup_steps)
            for pg in self.optimizer.param_groups:
                pg['lr'] = lr
        else:
            self.warmup_done = True

    def step_epoch(self, val_metric: float):
        if self.warmup_done:
            self.plateau.step(val_metric)


# -----------------------------------------------------------------------------
# Data Augmentation (esclusiva per Config C)
# -----------------------------------------------------------------------------

def mutate_dna_sequence(sequences: torch.Tensor, nmute: int) -> torch.Tensor:
    """Mutazione stocastica di basi, applicata solo a Config C (Yin et al.)."""
    if nmute <= 0:
        return sequences
    mutated = sequences.clone()
    B, L = mutated.shape
    nmute = min(nmute, L - 1)

    for i in range(B):
        mute_indices = torch.randperm(L, device=sequences.device)[:nmute]
        random_tokens = torch.randint(0, 4, (nmute,), dtype=mutated.dtype, device=sequences.device)
        mutated[i, mute_indices] = random_tokens
    return mutated


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------

def evaluate(model, dataloader, device, is_ce: bool, amp_dtype: torch.dtype):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            x = batch["data"].to(device)
            y = batch["label"].to(device)

            with torch.cuda.amp.autocast(enabled=True, dtype=amp_dtype):
                logits = model(x)
                if is_ce:
                    probs = torch.softmax(logits.float(), dim=-1)[:, 1]
                else:
                    probs = torch.sigmoid(logits.float().view(-1))

            all_preds.extend(probs.cpu().numpy().flatten())
            all_targets.extend(y.cpu().numpy().flatten())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    if len(np.unique(all_targets)) < 2 or np.isnan(all_preds).any():
        return 0.5, 0.0, 0.0

    auc_score = roc_auc_score(all_targets, all_preds)
    precision, recall, _ = precision_recall_curve(all_targets, all_preds)
    prc_score = auc(recall, precision)
    binary_preds = (all_preds >= 0.5).astype(int)
    mcc_score = matthews_corrcoef(all_targets, binary_preds)

    return auc_score, prc_score, mcc_score


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train Nucleic Transformer with AMP and Model Soup")
    parser.add_argument("--task", type=str, required=True, choices=["viral", "anellovirus", "ecoli", "enhancer"])
    parser.add_argument("--config", type=str, required=True, choices=["A", "B", "C", "D"])
    parser.add_argument("--data_folder", type=str, default="/home/simionricc/datasets")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=None, help="Override numero epoche (default: A:120, B:30, C:100, D:30)")
    parser.add_argument("--nmute", type=int, default=40, help="Mutazioni stocastiche per Config C (paper: 40 per il task virale, 15 per ecoli)")
    parser.add_argument("--kmers", type=int, nargs="+", default=[2, 3, 4, 5, 6],
                         help="Kernel size(s) per la k-mer aggregation. Default: multi-k [2,3,4,5,6] "
                              "(comportamento della repo pubblica). Passare un solo valore (es. --kmers 13) "
                              "per replicare il regime single-k del paper (Fig. 2a, sez. Best Hyperparameters).")
    parser.add_argument("--suffix", type=str, default="",
                         help="Suffisso esplicito per checkpoint (es. _nmute40, _k13). Ha priorita' "
                              "sull'auto-derivazione da --kmers: usalo sempre quando cambi SOLO nmute "
                              "(o qualsiasi parametro diverso da kmers), altrimenti il checkpoint "
                              "sovrascrive quello del run standard perche' il nome dipenderebbe solo da kmers.")
    parser.add_argument("--top_k_swa", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--amp_dtype", type=str, default="fp16", choices=["fp16", "bf16"],
                         help="bf16 e' piu' stabile su sequenze lunghe (task virale/anellovirus, L_tot=1485) se la GPU lo supporta")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16

    # 1. Split e dataframe via data_utils
    train_df, val_df, seq_len, _ = resolve_task_splits(args.task, args.data_folder, seed=args.seed)

    # 2. Dropout differenziato per config (fedele al design A/B/C/D originale)
    dropout_map = {"A": 0.25, "B": 0.2, "C": 0.1, "D": 0.5}
    drop = dropout_map[args.config]

    is_ce = (args.config == "C")
    nclass = 2 if is_ce else 1

    # 3. Dataset
    train_ds = SequenceClassificationDataset(
        train_df["sequence"].tolist(), train_df["label"].tolist(),
        arch="nucleictransformer", target_len=seq_len, config=args.config, is_train=True
    )
    val_ds = SequenceClassificationDataset(
        val_df["sequence"].tolist(), val_df["label"].tolist(),
        arch="nucleictransformer", target_len=seq_len, config=args.config, is_train=False
    )

    # 4. Sampler pesato SOLO per Config C su task fortemente sbilanciati (viral/anellovirus)
    #    Evita la gradient starvation: con batch=24 e ~2% di positivi, la probabilita'
    #    che un batch non contenga alcun positivo e' ~62%, il che porta il modello
    #    a collassare nel minimo banale (predire sempre negativo).
    num_pos = float(train_df["label"].sum())
    num_neg = float(len(train_df) - num_pos)
    imbalance_ratio = num_neg / max(1.0, num_pos)
    use_weighted_sampler = is_ce and imbalance_ratio > 5.0

    if use_weighted_sampler:
        labels_arr = train_df["label"].to_numpy()
        class_sample_weight = np.where(labels_arr > 0.5, num_neg / max(1.0, num_pos), 1.0)
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(class_sample_weight, dtype=torch.double),
            num_samples=len(train_ds),
            replacement=True
        )
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                                   num_workers=2, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                   num_workers=2, pin_memory=True)

    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=2, pin_memory=True)

    default_epochs = {"A": 120, "B": 30, "C": 100, "D": 30}[args.config]
    total_epochs = args.epochs if args.epochs is not None else default_epochs

    print(f"[i] Training Nucleic Transformer | Task: '{args.task}' | Config: '{args.config}' | "
          f"Loss: {'CrossEntropy (2 classes)' if is_ce else 'Binary'} | Input: {seq_len} bp | "
          f"Train: {len(train_ds)} | Val: {len(val_ds)} | nmute: {args.nmute if is_ce else 0} | "
          f"kmers: {args.kmers} | suffix: '{args.suffix}' | epochs: {total_epochs} | dropout: {drop} | "
          f"weighted_sampler: {use_weighted_sampler} | top_k_swa: {args.top_k_swa} | AMP dtype: {args.amp_dtype}")

    # 5. Modello
    model = NucleicTransformer(
        ntoken=4, nclass=nclass, ninp=512, nhead=8, nhid=2048, nlayers=6,
        kmer_aggregation=True, kmers=args.kmers, dropout=drop,
        max_len=max(2000, seq_len * 6)
    ).to(device)

    # 6. Ottimizzatori, scheduler, loss
    total_steps = len(train_loader) * total_epochs
    custom_scheduler = False
    warmup_plateau = None

    if args.config == "A":
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.05), num_training_steps=total_steps)
        # Alpha dinamico in base allo sbilanciamento del task, con tetto abbassato
        # (0.90 invece di 0.95) per ridurre il rischio di overflow FP16 su viral.
        alpha = min(0.90, max(0.5, imbalance_ratio / (imbalance_ratio + 1.0)))
        criterion = BinaryFocalLoss(alpha=alpha, gamma=2.0)
    elif args.config == "B":
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)
        # Warmup lineare (~1 epoca) prima di passare a ReduceLROnPlateau: senza questo,
        # il Post-LN transformer collassa a AUC 0.5 fin dall'epoca 1 (visto nei log).
        warmup_steps_b = min(len(train_loader), 2000)
        warmup_plateau = WarmupThenPlateau(
            optimizer, base_lr=1e-3, warmup_steps=warmup_steps_b,
            plateau_kwargs=dict(mode="max", factor=0.5, patience=3)
        )
        custom_scheduler = True
        criterion = nn.BCEWithLogitsLoss()
    elif args.config == "C":
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0, weight_decay=0.0)
        aiayn_sched = lr_AIAYN(optimizer, d_model=512, warmup_steps=3200, factor=0.1)
        custom_scheduler = True
        criterion = nn.CrossEntropyLoss()
    elif args.config == "D":
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-3)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.05), num_training_steps=total_steps)
        criterion = nn.BCEWithLogitsLoss()

    scaler = torch.cuda.amp.GradScaler(enabled=(amp_dtype == torch.float16))

    saved_checkpoints = []
    best_val_auc = -1.0
    patience = 15 if args.config in ["A", "C"] else 30
    patience_counter = 0

    # 7. Training loop
    for epoch in range(1, total_epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0

        for batch in train_loader:
            x = batch["data"].to(device)
            y = batch["label"].to(device)

            if is_ce and args.nmute > 0:
                x = mutate_dna_sequence(x, args.nmute)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=True, dtype=amp_dtype):
                logits = model(x)
                if is_ce:
                    loss = criterion(logits.float(), y.long().view(-1))
                else:
                    loss = criterion(logits.float().view(-1), y.float().view(-1))

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.8 if args.config == "D" else 1.0)
            scaler.step(optimizer)
            scaler.update()

            if args.config == "C":
                aiayn_sched.step()
            elif args.config == "B":
                warmup_plateau.step_batch()
            elif args.config in ["A", "D"]:
                scheduler.step()

            total_loss += loss.item() if not math.isnan(loss.item()) else 0.0
            steps += 1

        train_loss = total_loss / max(steps, 1)
        val_auc, val_prc, val_mcc = evaluate(model, val_loader, device, is_ce=is_ce, amp_dtype=amp_dtype)

        if args.config == "B":
            warmup_plateau.step_epoch(val_auc)

        print(f"Epoch {epoch:03d}/{total_epochs:03d} | Train Loss: {train_loss:.4f} | "
              f"Val AUC: {val_auc:.4f} | Val PRC: {val_prc:.4f} | Val MCC: {val_mcc:.4f}")

        if val_auc > 0.55:
            state_dict_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
            saved_checkpoints.append((val_auc, epoch, state_dict_cpu))
            saved_checkpoints.sort(key=lambda item: item[0], reverse=True)
            if len(saved_checkpoints) > args.top_k_swa:
                saved_checkpoints.pop()

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    # 8. Model Soup / SWA
    # --suffix esplicito ha priorita': usalo sempre per isolare un run di ablazione
    # che cambia un parametro diverso da kmers (es. nmute), altrimenti il nome
    # verrebbe derivato solo da kmers e un run con --nmute 40 (kmers di default)
    # sovrascriverebbe silenziosamente il checkpoint del run standard con nmute=18.
    if args.suffix:
        ckpt_suffix = args.suffix if args.suffix.startswith("_") else f"_{args.suffix}"
    else:
        kmers_tag = "-".join(str(k) for k in args.kmers)
        default_kmers_tag = "2-3-4-5-6"
        ckpt_suffix = "" if kmers_tag == default_kmers_tag else f"_k{kmers_tag}"
    out_ckpt = os.path.join(args.checkpoint_dir, f"best_nucleictransformer_{args.task}_{args.config}{ckpt_suffix}.pt")

    if saved_checkpoints:
        print(f"\n[i] Calcolo Weight Averaging (Model Soup) sui migliori {len(saved_checkpoints)} checkpoint:")
        for idx, (score, ep, _) in enumerate(saved_checkpoints):
            print(f"   -> Top {idx+1}: Epoca {ep:03d} (Val AUC: {score:.4f})")

        soup_state = copy.deepcopy(saved_checkpoints[0][2])
        for key in soup_state.keys():
            for i in range(1, len(saved_checkpoints)):
                soup_state[key] += saved_checkpoints[i][2][key]
            soup_state[key] = soup_state[key] / float(len(saved_checkpoints))

        torch.save(soup_state, out_ckpt)
        print(f"[v] Checkpoint Model Soup salvato: {out_ckpt} (Peak Val AUC: {best_val_auc:.4f})")
    else:
        torch.save(model.state_dict(), out_ckpt)
        print(f"[!] Nessun checkpoint qualificato (>0.55). Salvato ultimo stato in: {out_ckpt}")


if __name__ == "__main__":
    main()