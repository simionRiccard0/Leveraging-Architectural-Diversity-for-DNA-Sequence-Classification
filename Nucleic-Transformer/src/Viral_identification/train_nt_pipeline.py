import argparse
import glob
import math
import os
import random
import shutil
import time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
from torch.cuda.amp import GradScaler, autocast
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

torch.backends.cudnn.benchmark = True

from Functions import *
from LrScheduler import *
from Network import *

NUC_TO_INT = {"A": 0, "C": 1, "G": 2, "T": 3}
CLEAN_TRANS = str.maketrans(
    {
        "N": "A",
        "R": "A",
        "Y": "C",
        "S": "G",
        "W": "A",
        "K": "G",
        "M": "A",
        "B": "C",
        "D": "A",
        "H": "A",
        "V": "A",
    }
)

NUC_MAP = np.zeros(256, dtype=np.int64)
NUC_MAP[ord("A")] = 0
NUC_MAP[ord("C")] = 1
NUC_MAP[ord("G")] = 2
NUC_MAP[ord("T")] = 3

parser = argparse.ArgumentParser(
    description="Optimized Nucleic Transformer Multi-Task Pipeline"
)
parser.add_argument(
    "--config",
    type=str,
    required=True,
    choices=["A", "B", "C"],
    help="Training configuration strategy (A, B, C)",
)
parser.add_argument(
    "--task",
    type=str,
    default="viral",
    choices=["viral", "ecoli", "enhancer", "anellovirus"],
    help="Target biological training task (default: viral)",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=16,
    help="Batch size for training (default: 16)",
)
parser.add_argument(
    "--data_folder",
    "-d",
    default="/home/simionricc/datasets",
    help="Root directory for external benchmark datasets",
)
parser.add_argument(
    "--viraminer_folder",
    default="..",
    help="Path to native fullset_train/validation/test.csv files",
)
parser.add_argument("--gpu_id", type=str, default="0")
parser.add_argument("--ntoken", type=int, default=4)
parser.add_argument("--ninp", type=int, default=512)
parser.add_argument("--nhead", type=int, default=8)
parser.add_argument("--nhid", type=int, default=2048)
parser.add_argument("--nlayers", type=int, default=6)
parser.add_argument("--kmers", type=int, nargs="+", default=[2, 3, 4, 5, 6])
parser.add_argument(
    "--kmer_aggregation", dest="kmer_aggregation", action="store_true"
)
parser.set_defaults(kmer_aggregation=True)
parser.add_argument("--warmup_steps", type=int, default=3200)
parser.add_argument("--lr_scale", type=float, default=0.1)
args = parser.parse_args()

CONFIG_MAP = {
    "A": {
        "epochs": 120,
        "early_stop": True,
        "patience": 12,
        "loss": "focal",
        "optimizer": "adamw",
        "lr_sched": "cosine",
        "augment": "rc_shift",
        "dropout": 0.1,
        "nclass": 1,
    },
    "B": {
        "epochs": 30,
        "early_stop": False,
        "patience": 0,
        "loss": "bce",
        "optimizer": "adam",
        "lr_sched": "step",
        "augment": "none",
        "dropout": 0.2,
        "nclass": 1,
    },
    "C": {
        "epochs": 60,
        "early_stop": False,
        "patience": 0,
        "loss": "ce",
        "optimizer": "adam",
        "lr_sched": "aiayn",
        "augment": "mutation",
        "dropout": 0.1,
        "nclass": 2,
    },
}
cfg = CONFIG_MAP[args.config]

GLOBAL_LOGITS_DIR = "/home/simionricc/global_logits"
os.makedirs(GLOBAL_LOGITS_DIR, exist_ok=True)
checkpoints_folder = f"/tmp/checkpoints_nt_{args.task}_cfg_{args.config}_{os.getpid()}"
os.makedirs(checkpoints_folder, exist_ok=True)
best_model_path = os.path.join(checkpoints_folder, "best_model.pt")


def _peek_columns(path):
    with open(path, "r") as fh:
        first_line = fh.readline().strip()
    fields = first_line.split(",")
    has_header = not any(c.isdigit() for c in fields[-1])
    return len(fields), has_header


def load_seq_label_dataframe(path):
    ncols, has_header = _peek_columns(path)
    df = pd.read_csv(path) if has_header else pd.read_csv(path, header=None)
    if df.shape[1] == 2:
        seqs = df.iloc[:, 0].astype(str).values
        labels = df.iloc[:, 1].astype(float).values
    else:
        seqs = df.iloc[:, 1].astype(str).values
        labels = df.iloc[:, 2].astype(float).values

    seqs = np.array([s.upper() for s in seqs])
    return pd.DataFrame({0: range(len(seqs)), 1: seqs, 2: labels})


def get_task_dataframes(task_name, base_dir, viraminer_folder):
    if task_name == "viral":
        train_path = os.path.join(viraminer_folder, "fullset_train.csv")
        val_path = os.path.join(viraminer_folder, "fullset_validation.csv")
        if not os.path.exists(train_path):
            train_path = os.path.join(base_dir, "DNA_data", "fullset_train.csv")
            val_path = os.path.join(
                base_dir, "DNA_data", "fullset_validation.csv"
            )
        if not os.path.exists(train_path):
            train_path = os.path.join(base_dir, "fullset_train.csv")
            val_path = os.path.join(base_dir, "fullset_validation.csv")
        train_df = load_seq_label_dataframe(train_path)
        val_df = load_seq_label_dataframe(val_path)
        return train_df, val_df, 300

    elif task_name == "anellovirus":
        train_path = os.path.join(
            base_dir, "unseen_anell", "training_data_anell_300_binary.csv"
        )
        full_df = load_seq_label_dataframe(train_path)
        full_df = full_df.sample(frac=1, random_state=2020).reset_index(drop=True)
        n_val = int(len(full_df) * 0.2)
        val_df = full_df.iloc[:n_val].reset_index(drop=True)
        train_df = full_df.iloc[n_val:].reset_index(drop=True)
        return train_df, val_df, 300

    elif task_name == "ecoli":
        train_path = os.path.join(base_dir, "ecoli", "ecoli_train_NT.csv")
        target_len = 81
    elif task_name == "enhancer":
        train_path = os.path.join(
            base_dir, "enhancer", "bert_enhancer_dataset.csv"
        )
        target_len = 200
    else:
        raise ValueError(f"Invalid task: {task_name}")

    full_df = load_seq_label_dataframe(train_path)
    full_df = full_df.sample(frac=1, random_state=2020).reset_index(drop=True)
    n_val = int(len(full_df) * 0.2)
    val_df = full_df.iloc[:n_val].reset_index(drop=True)
    train_df = full_df.iloc[n_val:].reset_index(drop=True)
    return train_df, val_df, target_len


class FastNTDataset(Dataset):
    def __init__(self, dataframe, augment_type="none", target_len=300):
        raw_seqs = dataframe.iloc[:, 1].values
        labels = dataframe.iloc[:, 2].values
        self.augment_type = augment_type
        self.target_len = target_len

        cleaned_list = []
        for s in raw_seqs:
            s_clean = str(s).upper().translate(CLEAN_TRANS)
            if len(s_clean) < target_len:
                s_clean = s_clean.ljust(target_len, "A")
            else:
                s_clean = s_clean[:target_len]
            cleaned_list.append(s_clean)

        self.labels = torch.tensor(labels, dtype=torch.float32)

        full_bytes = "".join(cleaned_list).encode("ascii")
        byte_matrix = np.frombuffer(full_bytes, dtype=np.uint8).reshape(
            len(cleaned_list), target_len
        )
        encoded_matrix = NUC_MAP[byte_matrix]
        self.data_tensor = torch.from_numpy(encoded_matrix).long()

    def __len__(self):
        return len(self.data_tensor)

    def __getitem__(self, idx):
        x = self.data_tensor[idx]
        label = self.labels[idx]

        if self.augment_type == "none":
            return {"data": x, "labels": label}

        if self.augment_type == "rc_shift":
            x = x.clone()
            if random.random() > 0.5:
                x = 3 - torch.flip(x, dims=[0])
            shift = random.randint(-5, 5)
            if shift > 0:
                x = torch.cat([x[shift:], torch.zeros(shift, dtype=torch.long)])
            elif shift < 0:
                x = torch.cat([torch.zeros(-shift, dtype=torch.long), x[:shift]])
            return {"data": x, "labels": label}

        elif self.augment_type == "mutation":
            x = x.clone()
            n_mut = min(40, max(1, int(self.target_len * 0.13)))
            mut_indices = torch.randperm(self.target_len)[:n_mut]
            x[mut_indices] = torch.randint(0, 4, (n_mut,), dtype=torch.long)
            return {"data": x, "labels": label}

        return {"data": x, "labels": label}


train_df, val_df, TASK_SEQ_LEN = get_task_dataframes(
    args.task, args.data_folder, args.viraminer_folder
)

train_batch_size = args.batch_size
print(
    f"[i] Task: '{args.task}' | Config: '{args.config}' | Input Length: {TASK_SEQ_LEN} bp | "
    f"Train Batch Size: {train_batch_size} | Train: {len(train_df)} | Val: {len(val_df)}",
    flush=True,
)

train_loader = DataLoader(
    FastNTDataset(
        train_df, augment_type=cfg["augment"], target_len=TASK_SEQ_LEN
    ),
    batch_size=train_batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
)

val_loader = DataLoader(
    FastNTDataset(val_df, augment_type="none", target_len=TASK_SEQ_LEN),
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, smoothing=0.05):
        super().__init__()
        self.alpha, self.gamma, self.smoothing = alpha, gamma, smoothing

    def forward(self, logits, targets):
        targets_smooth = targets.float() * (
            1 - self.smoothing
        ) + 0.5 * self.smoothing
        bce = F.binary_cross_entropy_with_logits(
            logits.float(), targets_smooth, reduction="none"
        )
        bce = torch.clamp(bce, min=1e-7, max=50.0)
        return (self.alpha * (1 - torch.exp(-bce)) ** self.gamma * bce).mean()


seed_everything(2020)

# DEI Watchdog VRAM Guard
if torch.cuda.is_available():
    _vram_guard = torch.empty((1000, 1024, 1024), dtype=torch.float32, device="cuda")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = NucleicTransformer(
    args.ntoken,
    cfg["nclass"],
    args.ninp,
    args.nhead,
    args.nhid,
    args.nlayers,
    args.kmer_aggregation,
    kmers=args.kmers,
    dropout=cfg["dropout"],
).to(device)

if cfg["loss"] == "focal":
    criterion = FocalLoss()
elif cfg["loss"] == "bce":
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.CrossEntropyLoss(reduction="none")

if cfg["optimizer"] == "adamw":
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-4, weight_decay=1e-4
    )
else:
    lr_base = 0.001 if args.config == "B" else 1.0
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr_base, weight_decay=0, eps=1e-8
    )

if cfg["lr_sched"] == "cosine":

    def lr_lambda(epoch):
        if epoch < 5:
            return (epoch + 1) / 5
        return 0.5 * (1 + math.cos(math.pi * (epoch - 5) / (cfg["epochs"] - 5)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
elif cfg["lr_sched"] == "step":
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=5, gamma=0.5
    )
else:
    scheduler = lr_AIAYN(
        optimizer, args.ninp, args.warmup_steps, args.lr_scale
    )

scaler = GradScaler()
best_auc = 0.0
counter = 0

for epoch in range(cfg["epochs"]):
    model.train()
    total_loss = 0
    t0 = time.time()

    for step, data in enumerate(train_loader):
        if cfg["lr_sched"] == "aiayn":
            scheduler.step()

        src = data["data"].to(device, non_blocking=True).long()
        if cfg["loss"] == "ce":
            labels = data["labels"].to(device, non_blocking=True).long()
        else:
            labels = (
                data["labels"].to(device, non_blocking=True).float().unsqueeze(1)
            )

        optimizer.zero_grad()

        with autocast():
            output = model(src).float()
            loss = (
                torch.mean(criterion(output, labels))
                if cfg["loss"] == "ce"
                else criterion(output, labels)
            )

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    if cfg["lr_sched"] != "aiayn":
        scheduler.step()

    model.eval()
    val_preds, val_targets = [], []
    with torch.no_grad():
        for data in val_loader:
            src = data["data"].to(device, non_blocking=True).long()
            out = model(src).float()
            probs = (
                F.softmax(out, dim=1)[:, 1]
                if cfg["loss"] == "ce"
                else torch.sigmoid(out)
            )
            probs_np = np.nan_to_num(
                probs.cpu().numpy().flatten(), nan=0.5, posinf=1.0, neginf=0.0
            )
            val_preds.extend(probs_np)
            val_targets.extend(data["labels"].numpy().flatten())

    auc = roc_auc_score(val_targets, val_preds)
    elapsed = time.time() - t0
    print(
        f"Epoch {epoch + 1:02d}/{cfg['epochs']} | Loss: {total_loss / len(train_loader):.4f} | "
        f"Val AUC: {auc:.4f} | Time: {elapsed:.1f}s",
        flush=True,
    )

    if auc > best_auc:
        best_auc = auc
        counter = 0
        torch.save(model.state_dict(), best_model_path)
    else:
        counter += 1

    if cfg["early_stop"] and counter >= cfg["patience"]:
        print("Early stopping triggered.", flush=True)
        break

if not cfg["early_stop"]:
    torch.save(model.state_dict(), best_model_path)

if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("\n--- Generating Clean Logits ---", flush=True)
model.load_state_dict(torch.load(best_model_path, map_location=device))
model.eval()


def _resolve_test_file(folder, default_name):
    target = os.path.join(folder, default_name)
    if os.path.exists(target):
        return target
    all_csvs = sorted(glob.glob(os.path.join(folder, "*.csv")))
    test_csvs = [f for f in all_csvs if "test" in os.path.basename(f).lower()]
    return test_csvs[0] if test_csvs else (all_csvs[0] if all_csvs else None)


def _loader_from_csv(path, target_len, batch_size=32):
    df_extra = load_seq_label_dataframe(path)
    return DataLoader(
        FastNTDataset(df_extra, augment_type="none", target_len=target_len),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )


DATASETS_TO_EVAL = {}

if args.task == "viral":
    vm_test_path = os.path.join(args.data_folder, "fullset_test.csv")
    if not os.path.exists(vm_test_path):
        vm_test_path = os.path.join(args.data_folder, "DNA_data", "fullset_test.csv")
    if not os.path.exists(vm_test_path):
        vm_test_path = os.path.join(args.viraminer_folder, "fullset_test.csv")

    if os.path.exists(vm_test_path):
        DATASETS_TO_EVAL["test_VM"] = _loader_from_csv(
            vm_test_path, target_len=300
        )

    _sim_base_dir = os.path.join(args.data_folder, "SIM")
    if os.path.exists(_sim_base_dir):
        for _f in sorted(
            glob.glob(os.path.join(_sim_base_dir, "**", "*.csv"), recursive=True)
        ):
            _rel_noext = os.path.splitext(os.path.relpath(_f, _sim_base_dir))[0]
            _key = "test_sim_" + _rel_noext.replace(os.sep, "_")
            DATASETS_TO_EVAL[_key] = _loader_from_csv(_f, target_len=300)

elif args.task == "anellovirus":
    anell_test = os.path.join(
        args.data_folder, "unseen_anell", "test_data_anell_300_binary.csv"
    )
    if os.path.exists(anell_test):
        DATASETS_TO_EVAL["test_anellovirus"] = _loader_from_csv(
            anell_test, target_len=300
        )

elif args.task == "ecoli":
    ecoli_test = _resolve_test_file(
        os.path.join(args.data_folder, "ecoli"), "ecoli_independent_test_NT.csv"
    )
    if ecoli_test:
        DATASETS_TO_EVAL["test_ecoli"] = _loader_from_csv(ecoli_test, target_len=81)

elif args.task == "enhancer":
    enhancer_test = _resolve_test_file(
        os.path.join(args.data_folder, "enhancer"), "bert_enhancer_ind_test_set.csv"
    )
    if enhancer_test:
        DATASETS_TO_EVAL["test_enhancer"] = _loader_from_csv(
            enhancer_test, target_len=200
        )

with torch.no_grad():
    for name, loader in DATASETS_TO_EVAL.items():
        logits_list = []
        for data in loader:
            src = data["data"].to(device, non_blocking=True).long()
            out = model(src).float()
            if cfg["loss"] == "ce":
                logits_val = out[:, 1] - out[:, 0]
            else:
                logits_val = out
            probs_flat = np.nan_to_num(
                logits_val.cpu().numpy().flatten(), nan=0.0
            )
            logits_list.extend(probs_flat)

        filename = f"nucleictransformer_cfg{args.config}_{args.task}_{name}.txt"
        save_path = os.path.join(GLOBAL_LOGITS_DIR, filename)
        np.savetxt(save_path, np.array(logits_list), fmt="%.6f")
        print(
            f" Logits saved ({len(logits_list)} samples): {filename}",
            flush=True,
        )

if os.path.exists(checkpoints_folder):
    shutil.rmtree(checkpoints_folder)
print("Pipeline completata con successo.", flush=True)