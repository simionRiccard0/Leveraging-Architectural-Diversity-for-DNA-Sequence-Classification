#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import sys
import glob
import math
import random
import shutil
import logging
import argparse
from typing import Tuple, Dict, Any, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup

logging.basicConfig(
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# =====================================================================
# 1. RIPRODUCIBILITÀ GLOBALE (SEED 2020)
# =====================================================================

def seed_everything(seed: int = 2020) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"[i] Seed globale fissato a: {seed}")


# =====================================================================
# 2. PARSING CSV (HEADERLESS & WITH-HEADER)
# =====================================================================

def _is_dna_sequence(val: Any) -> bool:
    if not isinstance(val, str) or len(val) < 10:
        return False
    val_set = set(val.upper().strip())
    return val_set.issubset({"A", "C", "G", "T", "N", "U"})


def load_seq_label_dataframe(file_path: str) -> pd.DataFrame:
    """Carica CSV/TSV identificando in modo euristico la sequenza e la label binaria."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File non trovato: {file_path}")

    sample = pd.read_csv(file_path, nrows=5, header=None)
    sep = "\t" if sample.shape[1] == 1 and "\t" in str(sample.iloc[0, 0]) else ","
    df_raw = pd.read_csv(file_path, sep=sep, header=None, low_memory=False)

    first_row_dna = _is_dna_sequence(df_raw.iloc[0, 0]) or _is_dna_sequence(df_raw.iloc[0, 1])
    if not first_row_dna:
        df_raw = pd.read_csv(file_path, sep=sep, header=0, low_memory=False)
        df_raw.columns = [f"col_{i}" for i in range(df_raw.shape[1])]

    seq_col = None
    label_col = None

    for col in df_raw.columns:
        sample_vals = df_raw[col].dropna().astype(str).head(10).tolist()
        if any(_is_dna_sequence(v) for v in sample_vals):
            seq_col = col
            break

    for col in df_raw.columns:
        if col == seq_col:
            continue
        try:
            numeric_vals = pd.to_numeric(df_raw[col].dropna().head(10))
            if set(numeric_vals.unique()).issubset({0, 1, 0.0, 1.0}):
                label_col = col
                break
        except (ValueError, TypeError):
            continue

    if seq_col is None:
        seq_col = df_raw.columns[0]
    if label_col is None:
        label_col = df_raw.columns[1] if len(df_raw.columns) > 1 else df_raw.columns[0]

    return pd.DataFrame({
        "sequence": df_raw[seq_col].astype(str).str.upper().str.strip(),
        "label": pd.to_numeric(df_raw[label_col], errors="coerce").fillna(0).astype(int)
    })


# =====================================================================
# 3. RISOLUZIONE DEI DATASET CON RICERCA SCOPED PER TASK
# =====================================================================

def find_file_scoped(base_dir: str, candidate_subdirs: List[str], candidate_files: List[str]) -> Optional[str]:
    """Cerca i file primariamente nelle sottodirectory dedicate al task."""
    for sub in candidate_subdirs:
        sub_path = os.path.join(base_dir, sub) if sub else base_dir
        if os.path.exists(sub_path) and os.path.isdir(sub_path):
            for cand in candidate_files:
                direct = os.path.join(sub_path, cand)
                if os.path.exists(direct):
                    return direct
            for root, _, files in os.walk(sub_path):
                for cand in candidate_files:
                    if cand in files:
                        return os.path.join(root, cand)

    # Fallback diretto nella radice base_dir (senza walk per evitare cross-contamination)
    for cand in candidate_files:
        direct = os.path.join(base_dir, cand)
        if os.path.exists(direct):
            return direct
    return None


def resolve_datasets(data_folder: str, task: str, seed: int = 2020) -> Tuple[pd.DataFrame, pd.DataFrame, int, Dict[str, str]]:
    """
    Risolve i file di training/test e gestisce lo split in-code 80/20 dove necessario.
    """
    task = task.lower()
    cascade_eval_paths = {}

    if task == "viral":
        seq_len = 300
        subdirs = ["viral", "VM", "DS-VM", ""]
        train_p = find_file_scoped(data_folder, subdirs, ["fullset_train.csv"])
        val_p = find_file_scoped(data_folder, subdirs, ["fullset_validation.csv"])

        if not train_p or not val_p:
            raise FileNotFoundError(f"File viral 'fullset_train.csv' o 'fullset_validation.csv' non trovati in {data_folder}.")

        train_df = load_seq_label_dataframe(train_p)
        val_df = load_seq_label_dataframe(val_p)

        test_vm_p = find_file_scoped(data_folder, subdirs, ["fullset_test.csv"])
        if test_vm_p:
            cascade_eval_paths["test_VM"] = test_vm_p

        ood_p = find_file_scoped(data_folder, ["anellovirus", "anello", "unseen_anell", ""], ["test_data_anell_300_binary.csv"])
        if ood_p:
            cascade_eval_paths["test_anellovirus"] = ood_p

        # Scansione dei 15 file di rumore (DS-SIM: 3 noise levels x 5 seeds)
        sim_base_dir = os.path.join(data_folder, "SIM")
        if not os.path.exists(sim_base_dir):
            sim_base_dir = find_file_scoped(data_folder, ["SIM", ""], ["SIM"]) or sim_base_dir

        if os.path.exists(sim_base_dir) and os.path.isdir(sim_base_dir):
            sim_files = sorted(glob.glob(os.path.join(sim_base_dir, "**", "*.csv"), recursive=True))
            for sf in sim_files:
                rel_path = os.path.relpath(sf, sim_base_dir)
                tag_name = "test_sim_" + os.path.splitext(rel_path)[0].replace(os.sep, "_")
                cascade_eval_paths[tag_name] = sf

    elif task == "ecoli":
        seq_len = 81
        subdirs = ["ecoli", "DS-ECO", "promoter", ""]
        train_p = find_file_scoped(data_folder, subdirs, ["ecoli_train_NT.csv"])
        test_p = find_file_scoped(data_folder, subdirs, ["ecoli_independent_test_NT.csv", "ecoli_test_NT.csv"])

        if not train_p:
            raise FileNotFoundError(f"File ecoli 'ecoli_train_NT.csv' non trovato in {data_folder}.")

        full_df = load_seq_label_dataframe(train_p)
        df_shuffled = full_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n_val = int(len(df_shuffled) * 0.2)
        val_df = df_shuffled.iloc[:n_val].reset_index(drop=True)
        train_df = df_shuffled.iloc[n_val:].reset_index(drop=True)

        if test_p:
            cascade_eval_paths["test_ecoli"] = test_p

    elif task == "enhancer":
        seq_len = 200
        subdirs = ["enhancer", "DS-ENH", "bert_enhancer", ""]
        train_p = find_file_scoped(data_folder, subdirs, ["bert_enhancer_train.csv", "bert_enhancer_dataset.csv"])
        test_p = find_file_scoped(data_folder, subdirs, ["bert_enhancer_ind_test_set.csv", "bert_enhancer_test.csv"])

        if not train_p:
            raise FileNotFoundError(f"Dataset enhancer 'bert_enhancer_train.csv' non trovato in {data_folder}.")

        full_df = load_seq_label_dataframe(train_p)
        df_shuffled = full_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n_val = int(len(df_shuffled) * 0.2)
        val_df = df_shuffled.iloc[:n_val].reset_index(drop=True)
        train_df = df_shuffled.iloc[n_val:].reset_index(drop=True)

        if test_p:
            cascade_eval_paths["test_enhancer"] = test_p

    elif task == "anellovirus":
        seq_len = 300
        subdirs = ["anellovirus", "anello", "unseen_anell", "DS-OOD", ""]
        train_p = find_file_scoped(data_folder, subdirs, ["training_data_anell_300_binary.csv"])
        test_p = find_file_scoped(data_folder, subdirs, ["test_data_anell_300_binary.csv"])

        if not train_p:
            raise FileNotFoundError(f"File anellovirus 'training_data_anell_300_binary.csv' non trovato in {data_folder}.")

        full_df = load_seq_label_dataframe(train_p)
        df_shuffled = full_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n_val = int(len(df_shuffled) * 0.2)
        val_df = df_shuffled.iloc[:n_val].reset_index(drop=True)
        train_df = df_shuffled.iloc[n_val:].reset_index(drop=True)

        if test_p:
            cascade_eval_paths["test_anellovirus"] = test_p

    else:
        raise ValueError(f"Task sconosciuto: {task}")

    return train_df, val_df, seq_len, cascade_eval_paths


# =====================================================================
# 4. GESTIONE I/O E CHECKPOINTING ATOMICO SU NFS
# =====================================================================

def check_disk_space(path: str, required_mb: float = 1500.0) -> None:
    target_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(target_dir, exist_ok=True)
    _, _, free = shutil.disk_usage(target_dir)
    free_mb = free / (1024 * 1024)
    if free_mb < required_mb:
        raise IOError(f"Spazio disco insufficiente ({free_mb:.1f} MB liberi, richiesti >= {required_mb:.1f} MB).")


def safe_save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, val_metrics: Dict[str, float], path: str) -> bool:
    tmp_path = path + ".tmp"
    try:
        check_disk_space(path, required_mb=1500.0)
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": val_metrics
        }
        torch.save(state, tmp_path)
        if (os.path.getsize(tmp_path) / (1024 * 1024)) < 50.0:
            raise IOError("File temporaneo di checkpoint corrotto o incompleto.")
        os.replace(tmp_path, path)
        logger.info(f"[✓] Checkpoint salvato con successo: {path}")
        return True
    except Exception as e:
        logger.error(f"[!] Errore salvataggio checkpoint: {str(e)}")
        if os.path.exists(tmp_path):
            try: os.remove(tmp_path)
            except OSError: pass
        return False


def safe_load_checkpoint(model: nn.Module, path: str, device: torch.device) -> nn.Module:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        raise FileNotFoundError(f"Checkpoint non valido o a 0 byte: {path}")
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    logger.info(f"[✓] Best checkpoint ricaricato da: {path}")
    return model


# =====================================================================
# 5. DATA AUGMENTATION (CONFORME A C1, C2, C3)
# =====================================================================

def reverse_complement(dna_seq: str) -> str:
    comp = {"A": "T", "C": "G", "G": "C", "T": "A", "N": "N", "U": "A"}
    return "".join(comp.get(b, "N") for b in reversed(dna_seq))


def positional_shift(dna_seq: str, max_shift: int = 5) -> str:
    """Positional shift: trim + pad con 'N' (Eq. 8 del paper)."""
    shift = random.randint(-max_shift, max_shift)
    if shift > 0:
        return dna_seq[shift:] + "N" * shift
    elif shift < 0:
        return "N" * (-shift) + dna_seq[:shift]
    return dna_seq


def nucleic_transformer_mutation(dna_seq: str) -> str:
    """Mutation Augmentation: K = min(40, floor(0.13 * L)) (Eq. 9 del paper)."""
    L = len(dna_seq)
    k = min(40, math.floor(0.13 * L))
    if k <= 0 or L == 0:
        return dna_seq
    bases = ["A", "C", "G", "T"]
    indices = random.sample(range(L), k)
    seq_list = list(dna_seq)
    for idx in indices:
        curr = seq_list[idx]
        choices = [b for b in bases if b != curr]
        seq_list[idx] = random.choice(choices)
    return "".join(seq_list)


class GenomicSequenceDataset(Dataset):
    def __init__(self, sequences: List[str], labels: List[int], tokenizer: Any, max_length: int, config: str, is_train: bool):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.config = config.upper()
        self.is_train = is_train

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx: int):
        seq = self.sequences[idx]

        if self.is_train:
            if self.config == "A":  # C1: RC (50%) + Positional Shift (±5 bp)
                if random.random() < 0.5:
                    seq = reverse_complement(seq)
                seq = positional_shift(seq, max_shift=5)
            elif self.config == "B":  # C2: Nessuna augmentation
                pass
            elif self.config == "C":  # C3: Mutation augmentation (min(40, floor(0.13*L)))
                seq = nucleic_transformer_mutation(seq)

        encoded = self.tokenizer(
            seq,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32)
        }


# =====================================================================
# 6. ARCHITETTURA MODELLO & LOSS FUNCTIONS
# =====================================================================

class BinaryFocalLoss(nn.Module):
    """Focal Loss con alpha=0.75, gamma=2.0 (C1 / Config A)."""
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


class DNABERT2ForSequenceClassification(nn.Module):
    """Backbone DNABERT-2 con Mean Pooling e Classification Head."""
    def __init__(self, model_path: str, dropout: float = 0.2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        hidden_size = self.bert.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def mean_pooling(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeds = torch.sum(last_hidden_state * mask, dim=1)
        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
        return sum_embeds / sum_mask

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.mean_pooling(outputs[0], attention_mask)
        return self.classifier(pooled).squeeze(-1)


# =====================================================================
# 7. OTTIMIZZATORE E SCHEDULER DIFFERENZIATI (A / B / C)
# =====================================================================

def configure_optimization(
    model: nn.Module,
    config: str,
    backbone_lr: float,
    head_lr: float,
    total_steps: int,
    epochs: int,
    pos_weight: Optional[torch.Tensor] = None
):
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    grouped_params = [
        {"params": [p for n, p in model.bert.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad], "lr": backbone_lr, "weight_decay": 0.01},
        {"params": [p for n, p in model.bert.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad], "lr": backbone_lr, "weight_decay": 0.0},
        {"params": [p for n, p in model.classifier.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad], "lr": head_lr, "weight_decay": 0.01},
        {"params": [p for n, p in model.classifier.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad], "lr": head_lr, "weight_decay": 0.0},
    ]

    cfg = config.upper()
    if cfg == "A":  # C1: Focal Loss (alpha=0.75, gamma=2.0), AdamW, Cosine Warmup
        criterion = BinaryFocalLoss(alpha=0.75, gamma=2.0)
        optimizer = torch.optim.AdamW(grouped_params, eps=1e-8)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.10), num_training_steps=total_steps)

    elif cfg == "B":  # C2: BCE pos_weight, Adam, StepLR decay
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(grouped_params, eps=1e-8)
        steps_per_epoch = max(1, total_steps // epochs)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=steps_per_epoch * 5, gamma=0.5)

    elif cfg == "C":  # C3: BCE pos_weight, Adam, Noam / Inverse Sqrt decay
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(grouped_params, eps=1e-8)
        warmup_steps = max(1, int(total_steps * 0.10))
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: (float(step + 1) / float(warmup_steps)) if (step < warmup_steps) else ((float(warmup_steps) / float(step + 1)) ** 0.5)
        )
    else:
        raise ValueError(f"Configurazione '{config}' non valida.")

    return criterion, optimizer, scheduler


# =====================================================================
# 8. VALUTAZIONE ED ESTRAZIONE DEI LOGIT
# =====================================================================

def evaluate_and_extract_logits(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            total_loss += loss.item() * input_ids.size(0)
            all_logits.extend(logits.detach().cpu().numpy())
            all_targets.extend(labels.detach().cpu().numpy())

    all_logits = np.array(all_logits)
    all_targets = np.array(all_targets)
    all_probs = 1.0 / (1.0 + np.exp(-all_logits))
    avg_loss = total_loss / len(dataloader.dataset)

    if len(np.unique(all_targets)) > 1:
        try: auroc = roc_auc_score(all_targets, all_probs)
        except ValueError: auroc = 0.5000
        try: auprc = average_precision_score(all_targets, all_probs)
        except ValueError: auprc = 0.0
        mcc = matthews_corrcoef(all_targets, (all_probs >= 0.5).astype(int))
    else:
        auroc, auprc, mcc = 0.5000, 0.0, 0.0

    return avg_loss, auroc, auprc, mcc, all_logits


# =====================================================================
# 9. MAIN PIPELINE
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="DNABERT2-CAMP Fine-tuning & Evaluation")
    parser.add_argument("--config", type=str, required=True, choices=["A", "B", "C"])
    parser.add_argument("--task", type=str, required=True, choices=["viral", "ecoli", "enhancer", "anellovirus"])
    parser.add_argument("--data_folder", type=str, default="/home/simionricc/datasets")
    parser.add_argument("--model_path", type=str, default="/nfsd/nannidl/simionricc/models/dnabert2_camp/model")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--backbone_lr", type=float, default=2e-5)
    parser.add_argument("--head_lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--logits_dir", type=str, default="./global_logits")
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.epochs is None:
        epochs_map = {"A": 40, "B": 20, "C": 30}
        args.epochs = epochs_map[args.config.upper()]

    train_df, val_df, seq_len, cascade_eval_dict = resolve_datasets(args.data_folder, args.task, args.seed)

    logger.info(f"[i] DNABERT-2 CAMP | Task: '{args.task}' | Config: '{args.config}' | Seq Length: {seq_len} bp | Train: {len(train_df)} | Val: {len(val_df)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    train_ds = GenomicSequenceDataset(train_df["sequence"].tolist(), train_df["label"].tolist(), tokenizer, seq_len, args.config, is_train=True)
    val_ds = GenomicSequenceDataset(val_df["sequence"].tolist(), val_df["label"].tolist(), tokenizer, seq_len, args.config, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    num_pos = sum(train_df["label"].values)
    num_neg = len(train_df) - num_pos
    pos_weight = torch.tensor([num_neg / max(num_pos, 1)], device=device)

    model = DNABERT2ForSequenceClassification(args.model_path).to(device)
    total_steps = len(train_loader) * args.epochs

    criterion, optimizer, scheduler = configure_optimization(
        model=model,
        config=args.config,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        total_steps=total_steps,
        epochs=args.epochs,
        pos_weight=pos_weight
    )

    scaler = torch.cuda.amp.GradScaler()
    best_val_auc = 0.0
    best_epoch = 0
    patience_counter = 0

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(args.checkpoint_dir, f"best_dnabert2camp_{args.task}_{args.config}.pt")

    # Training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.item() * input_ids.size(0)

        epoch_loss = train_loss / len(train_ds)
        val_loss, val_auc, val_auprc, val_mcc, _ = evaluate_and_extract_logits(model, val_loader, criterion, device)

        logger.info(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"Loss: {epoch_loss:.4f} | "
            f"Val AUC: {val_auc:.4f} | "
            f"Val AUPRC: {val_auprc:.4f} | "
            f"Val MCC: {val_mcc:.4f}"
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            patience_counter = 0
            safe_save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_metrics={"val_loss": val_loss, "val_auc": val_auc, "val_auprc": val_auprc, "val_mcc": val_mcc},
                path=best_model_path
            )
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"Early stopping all'epoca {epoch} (Best Val AUC: {best_val_auc:.4f} a epoca {best_epoch})")
                break

    # Cascade Evaluation e Salvataggio Logits
    logger.info("\n--- Automatic Cascade Evaluation ---")
    try:
        model = safe_load_checkpoint(model, best_model_path, device)
    except Exception as e:
        logger.warning(f"[!] Impossibile ripristinare il checkpoint ({e}). Uso lo stato corrente.")

    os.makedirs(args.logits_dir, exist_ok=True)

    for d_name, d_path in cascade_eval_dict.items():
        logger.info(f"[Cascade] Valutazione su '{d_name}' ({d_path})...")
        eval_df = load_seq_label_dataframe(d_path)
        eval_ds = GenomicSequenceDataset(eval_df["sequence"].tolist(), eval_df["label"].tolist(), tokenizer, seq_len, args.config, is_train=False)
        eval_loader = DataLoader(eval_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers)

        _, auroc, auprc, mcc, raw_logits = evaluate_and_extract_logits(model, eval_loader, criterion, device)
        logger.info(f" -> [{d_name}] AUROC: {auroc:.4f} | AUPRC: {auprc:.4f} | MCC: {mcc:.4f}")

        logit_file = os.path.join(args.logits_dir, f"dnabert2camp_cfg{args.config}_{args.task}_{d_name}.txt")
        np.savetxt(logit_file, raw_logits, fmt="%.6f")
        logger.info(f" -> Logit salvati in: {logit_file}")

    logger.info("Pipeline completed successfully.")


if __name__ == "__main__":
    main()