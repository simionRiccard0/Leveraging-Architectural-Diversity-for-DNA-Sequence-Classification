#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Shared data utilities
"""

import os
import glob
import math
import random
from typing import Tuple, Dict, List, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

MAPPING_4D = {"A": [1,0,0,0], "C": [0,1,0,0], "G": [0,0,1,0], "T": [0,0,0,1], "N": [0,0,0,0]}
MAPPING_5D = {"A": [1,0,0,0,0], "C": [0,1,0,0,0], "G": [0,0,1,0,0], "T": [0,0,0,1,0], "N": [0,0,0,0,1]}
NUC_TO_ID = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 0}


def seed_everything(seed: int = 2020) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _peek_columns(path: str) -> Tuple[int, bool]:
    with open(path, "r") as fh:
        first_line = fh.readline().strip()
    fields = first_line.split(",")
    has_header = not any(c.isdigit() for c in fields[-1])
    return len(fields), has_header


def load_seq_label_dataframe(path: str) -> pd.DataFrame:
    """Robust heuristic loader handling headerless, headers, 2-col, and 3-col CSVs."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    ncols, has_header = _peek_columns(path)
    df = pd.read_csv(path) if has_header else pd.read_csv(path, header=None)
    if df.shape[1] == 2:
        seqs = df.iloc[:, 0].astype(str).values
        labels = df.iloc[:, 1].astype(float).values
    else:
        seqs = df.iloc[:, 1].astype(str).values
        labels = df.iloc[:, 2].astype(float).values

    seqs = np.array([str(s).upper().strip() for s in seqs])
    labels = np.array([float(l) for l in labels])
    return pd.DataFrame({"sequence": seqs, "label": labels})


def find_file_scoped(base_dir: str, candidate_subdirs: List[str], candidate_files: List[str]) -> Optional[str]:
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
    for cand in candidate_files:
        direct = os.path.join(base_dir, cand)
        if os.path.exists(direct):
            return direct
    return None


def resolve_task_splits(task: str, data_folder: str, seed: int = 2020) -> Tuple[pd.DataFrame, pd.DataFrame, int, Dict[str, str]]:
    """Resolves train/val splits and cascade test targets for any benchmark task."""
    task = task.lower()
    cascade_test_paths = {}

    if task == "viral":
        seq_len = 300
        subdirs = ["viral", "DNA_data", "VM", "DS-VM", ""]
        train_p = find_file_scoped(data_folder, subdirs, ["fullset_train.csv"])
        val_p = find_file_scoped(data_folder, subdirs, ["fullset_validation.csv"])
        if not train_p or not val_p:
            raise FileNotFoundError(f"Viral fullset_train/validation not found in {data_folder}")
        train_df = load_seq_label_dataframe(train_p)
        val_df = load_seq_label_dataframe(val_p)

        test_vm = find_file_scoped(data_folder, subdirs, ["fullset_test.csv"])
        if test_vm:
            cascade_test_paths["test_VM"] = test_vm

        # Scan 15 synthetic noise sets (DS-SIM)[cite: 1, 8]
        sim_dir = find_file_scoped(data_folder, ["SIM", ""], ["SIM"]) or os.path.join(data_folder, "SIM")
        if os.path.exists(sim_dir) and os.path.isdir(sim_dir):
            for sf in sorted(glob.glob(os.path.join(sim_dir, "**", "*.csv"), recursive=True)):
                rel = os.path.relpath(sf, sim_dir)
                tag = "test_sim_" + os.path.splitext(rel)[0].replace(os.sep, "_")
                cascade_test_paths[tag] = sf

    elif task == "ecoli":
        seq_len = 81
        subdirs = ["ecoli", "DS-ECO", "promoter", ""]
        train_p = find_file_scoped(data_folder, subdirs, ["ecoli_train_NT.csv"])
        test_p = find_file_scoped(data_folder, subdirs, ["ecoli_independent_test_NT.csv", "ecoli_test_NT.csv"])
        if not train_p:
            raise FileNotFoundError(f"E. coli training set not found in {data_folder}")
        full_df = load_seq_label_dataframe(train_p).sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n_val = int(len(full_df) * 0.2)
        val_df = full_df.iloc[:n_val].reset_index(drop=True)
        train_df = full_df.iloc[n_val:].reset_index(drop=True)
        if test_p:
            cascade_test_paths["test_ecoli"] = test_p

    elif task == "enhancer":
        seq_len = 200
        subdirs = ["enhancer", "DS-ENH", "bert_enhancer", ""]
        train_p = find_file_scoped(data_folder, subdirs, ["bert_enhancer_dataset.csv", "bert_enhancer_train.csv"])
        test_p = find_file_scoped(data_folder, subdirs, ["bert_enhancer_ind_test_set.csv", "bert_enhancer_test.csv"])
        if not train_p:
            raise FileNotFoundError(f"Enhancer training set not found in {data_folder}")
        full_df = load_seq_label_dataframe(train_p).sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n_val = int(len(full_df) * 0.2)
        val_df = full_df.iloc[:n_val].reset_index(drop=True)
        train_df = full_df.iloc[n_val:].reset_index(drop=True)
        if test_p:
            cascade_test_paths["test_enhancer"] = test_p

    elif task == "anellovirus":
        seq_len = 300
        subdirs = ["unseen_anell", "anellovirus", "DS-OOD", ""]
        train_p = find_file_scoped(data_folder, subdirs, ["training_data_anell_300_binary.csv"])
        test_p = find_file_scoped(data_folder, subdirs, ["test_data_anell_300_binary.csv"])
        if not train_p:
            raise FileNotFoundError(f"Anellovirus training set not found in {data_folder}")
        full_df = load_seq_label_dataframe(train_p).sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n_val = int(len(full_df) * 0.2)
        val_df = full_df.iloc[:n_val].reset_index(drop=True)
        train_df = full_df.iloc[n_val:].reset_index(drop=True)
        if test_p:
            cascade_test_paths["test_anellovirus"] = test_p

    else:
        raise ValueError(f"Unknown task: {task}")

    return train_df, val_df, seq_len, cascade_test_paths


def reverse_complement(dna_seq: str) -> str:
    comp = {"A": "T", "C": "G", "G": "C", "T": "A", "N": "N", "U": "A"}
    return "".join(comp.get(b, "N") for b in reversed(dna_seq))


def positional_shift(dna_seq: str, max_shift: int = 5) -> str:
    shift = random.randint(-max_shift, max_shift)
    if shift > 0:
        return dna_seq[shift:] + "N" * shift
    elif shift < 0:
        return "N" * (-shift) + dna_seq[:shift]
    return dna_seq


def mutate_sequence(dna_seq: str) -> str:
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


class SequenceClassificationDataset(Dataset):
    """Generic dataset providing 4D one-hot, 5D one-hot, integer-tokens, or tokenizer embeddings[cite: 7, 8]."""
    def __init__(self, sequences: List[str], labels: List[float], arch: str, target_len: int, config: str, is_train: bool, tokenizer=None):
        self.sequences = sequences
        self.labels = labels
        self.arch = arch.lower()
        self.target_len = target_len
        self.config = config.upper()
        self.is_train = is_train
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx: int):
        seq = str(self.sequences[idx]).upper().strip()

        if self.is_train:
            if self.config == "A":
                if random.random() < 0.5:
                    seq = reverse_complement(seq)
                seq = positional_shift(seq, max_shift=5)
            elif self.config == "C":
                seq = mutate_sequence(seq)

        y = float(self.labels[idx])

        if self.arch == "dnabert2camp":
            enc = self.tokenizer(seq, padding="max_length", truncation=True, max_length=self.target_len, return_tensors="pt")
            return {"input_ids": enc["input_ids"].squeeze(0), "attention_mask": enc["attention_mask"].squeeze(0), "label": torch.tensor(y, dtype=torch.float32)}

        elif self.arch == "nucleictransformer":
            ids = [NUC_TO_ID.get(b, 0) for b in seq[:self.target_len]]
            while len(ids) < self.target_len:
                ids.append(0)
            target_tensor = torch.tensor(int(y), dtype=torch.long) if self.config == "C" else torch.tensor(y, dtype=torch.float32)
            return {"data": torch.tensor(ids, dtype=torch.long), "label": target_tensor}

        elif self.arch == "fulle":
            enc = [MAPPING_5D.get(b, [0,0,0,0,1]) for b in seq[:self.target_len]]
            while len(enc) < self.target_len:
                enc.append([0,0,0,0,1])
            return {"data": torch.tensor(enc, dtype=torch.float32).transpose(0, 1), "label": torch.tensor(y, dtype=torch.float32)}

        else:  # GenExplorer
            enc = [MAPPING_4D.get(b, [0,0,0,0]) for b in seq[:self.target_len]]
            while len(enc) < self.target_len:
                enc.append([0,0,0,0])
            target_tensor = torch.tensor(int(y), dtype=torch.long) if self.config == "C" else torch.tensor(y, dtype=torch.float32)
            return {"data": torch.tensor(enc, dtype=torch.float32), "label": target_tensor}