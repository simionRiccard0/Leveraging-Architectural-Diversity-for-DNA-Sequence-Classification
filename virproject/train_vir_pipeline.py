import argparse
import glob
import math
import os
import random
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

RC_TRANS = str.maketrans("ACGT", "TGCA")


def build_seq_set(seq_iterable):
    """Generates a fast lookup set containing both forward and reverse-complement sequences."""
    s = set()
    for seq in seq_iterable:
        seq_str = str(seq).upper()
        s.add(seq_str)
        s.add(seq_str.translate(RC_TRANS)[::-1])
    return s


parser = argparse.ArgumentParser(
    description="ViraExplorer Multi-Task Training Pipeline (DNA Sequence Classification)"
)
parser.add_argument(
    "--config",
    type=str,
    required=True,
    choices=["A", "B", "C"],
    help="Configurazione di training (A, B, C)",
)
parser.add_argument(
    "--task",
    type=str,
    default="viral",
    choices=["viral", "ecoli", "enhancer", "anellovirus"],
    help="Task biologico su cui addestrare il modello (default: viral)",
)
parser.add_argument(
    "--data_folder",
    "-d",
    default="/home/simionricc/datasets",
    help="Path alla cartella principale dei dataset",
)
args = parser.parse_args()


CONFIG_MAP = {
    "A": {  # ViraExplorer Native Strategy
        "epochs": 120,
        "early_stop": True,
        "patience": 15,
        "batch_size": 128,
        "loss": "focal",
        "optimizer": "adamw",
        "lr_sched": "cosine",
        "augment": "rc_shift",
        "cnn_dropout": 0.25,
        "trans_dropout": 0.1,
        "out_features": 1,
    },
    "B": {  # FullE Native Strategy
        "epochs": 30,
        "early_stop": False,
        "patience": 0,
        "batch_size": 128,
        "loss": "bce",
        "optimizer": "adam",
        "lr_sched": "step",
        "augment": "none",
        "cnn_dropout": 0.2,
        "trans_dropout": 0.2,
        "out_features": 1,
    },
    "C": {  # Nucleic Transformer Native Strategy
        "epochs": 60,
        "early_stop": False,
        "patience": 0,
        "batch_size": 128,
        "loss": "ce",
        "optimizer": "adam",
        "lr_sched": "aiayn",
        "augment": "mutation",
        "cnn_dropout": 0.1,
        "trans_dropout": 0.1,
        "out_features": 2,
    },
}
cfg = CONFIG_MAP[args.config]

SAVE_DIR = "/home/simionricc/virproject/"
BEST_MODEL_PATH = os.path.join(
    SAVE_DIR, f"best_vira_{args.task}_cfg_{args.config}.pth"
)
GLOBAL_LOGITS_DIR = "/home/simionricc/global_logits"
os.makedirs(GLOBAL_LOGITS_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)


def _peek_columns(path):
    with open(path, "r") as fh:
        first_line = fh.readline().strip()
    fields = first_line.split(",")

    def _is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    has_header = not _is_number(fields[-1])
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


def get_task_dataframes(task_name, base_dir):
    if task_name == "viral":
        train_path = "ViraMiner/data/DNA_data/fullset_train.csv"
        val_path = "ViraMiner/data/DNA_data/fullset_validation.csv"
        if not os.path.exists(train_path):
            train_path = os.path.join(base_dir, "fullset_train.csv")
            val_path = os.path.join(base_dir, "fullset_validation.csv")
        train_df = load_seq_label_dataframe(train_path)
        val_df = load_seq_label_dataframe(val_path)
        return train_df, val_df

    elif task_name == "ecoli":
        train_path = os.path.join(base_dir, "ecoli", "ecoli_train_NT.csv")
    elif task_name == "enhancer":
        train_path = os.path.join(
            base_dir, "enhancer", "bert_enhancer_dataset.csv"
        )
    elif task_name == "anellovirus":
        train_path = os.path.join(
            base_dir, "unseen_anell", "training_data_anell_300_binary.csv"
        )
        full_df = load_seq_label_dataframe(train_path)

        # Deduplicate Anellovirus training set against its test set (Remove overlapping sequences)
        anell_test_path = os.path.join(
            base_dir, "unseen_anell", "test_data_anell_300_binary.csv"
        )
        if os.path.exists(anell_test_path):
            anell_test_df = load_seq_label_dataframe(anell_test_path)
            test_seq_set = build_seq_set(anell_test_df.iloc[:, 1].values)
            keep_mask = [
                (str(s).upper() not in test_seq_set)
                for s in full_df.iloc[:, 1].values
            ]
            removed_count = len(full_df) - sum(keep_mask)
            full_df = full_df[keep_mask].reset_index(drop=True)
            print(
                f"[i] Filtered {removed_count} test-overlapping sequences from Anellovirus training split.",
                flush=True,
            )

        full_df = full_df.sample(frac=1, random_state=2020).reset_index(drop=True)
        n_val = int(len(full_df) * 0.2)
        val_df = full_df.iloc[:n_val].reset_index(drop=True)
        train_df = full_df.iloc[n_val:].reset_index(drop=True)
        return train_df, val_df
    else:
        raise ValueError(f"Task non valida: {task_name}")

    full_df = load_seq_label_dataframe(train_path)
    full_df = full_df.sample(frac=1, random_state=2020).reset_index(drop=True)
    n_val = int(len(full_df) * 0.2)
    val_df = full_df.iloc[:n_val].reset_index(drop=True)
    train_df = full_df.iloc[n_val:].reset_index(drop=True)
    return train_df, val_df


train_df, val_df = get_task_dataframes(args.task, args.data_folder)
print(
    f"[i] Task: '{args.task}' | Config: '{args.config}' | Train samples: {len(train_df)} | Val samples: {len(val_df)}",
    flush=True,
)

mapping = {
    "A": [1, 0, 0, 0],
    "C": [0, 1, 0, 0],
    "G": [0, 0, 1, 0],
    "T": [0, 0, 0, 1],
    "N": [0, 0, 0, 0],
}
MAX_LEN = 300


def mutate_sequence(seq, n_mutations=40):
    seq_list = list(seq)
    seq_len = len(seq_list)
    nucleotides = ["A", "C", "G", "T"]
    pos_to_mutate = random.sample(range(seq_len), min(n_mutations, seq_len))
    for pos in pos_to_mutate:
        current_nuc = seq_list[pos]
        choices = [n for n in nucleotides if n != current_nuc]
        seq_list[pos] = random.choice(choices)
    return "".join(seq_list)


def encode_sequence(seq, augment_type):
    seq = seq.upper()
    if augment_type == "rc_shift":
        complement = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
        if random.random() > 0.5:
            seq = "".join(complement.get(b, "N") for b in reversed(seq))
        shift = random.randint(-5, 5)
        seq = seq[max(0, shift) :] if shift > 0 else "N" * (-shift) + seq
    elif augment_type == "mutation":
        seq = mutate_sequence(seq, n_mutations=40)

    seq = seq[:MAX_LEN]
    encoded = [mapping.get(b, [0, 0, 0, 0]) for b in seq]
    while len(encoded) < MAX_LEN:
        encoded.append([0, 0, 0, 0])
    return encoded


class DNADataset(Dataset):
    def __init__(self, dataframe, augment_type="none"):
        self.sequences = dataframe[1].values
        self.labels = dataframe[2].values
        self.augment_type = augment_type

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x = torch.tensor(
            encode_sequence(self.sequences[idx], self.augment_type),
            dtype=torch.float32,
        )
        if cfg["loss"] == "ce":
            y = torch.tensor(self.labels[idx], dtype=torch.long)
        else:
            y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y


train_loader = DataLoader(
    DNADataset(train_df, augment_type=cfg["augment"]),
    batch_size=cfg["batch_size"],
    shuffle=True,
    num_workers=2,
    pin_memory=True,
)
val_loader = DataLoader(
    DNADataset(val_df, augment_type="none"),
    batch_size=cfg["batch_size"],
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)


class PatternBranch(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.conv6, self.conv12 = nn.Conv1d(
            4, 512, 6, padding="same"
        ), nn.Conv1d(4, 512, 12, padding="same")
        self.conv18, self.conv24 = nn.Conv1d(
            4, 512, 18, padding="same"
        ), nn.Conv1d(4, 512, 24, padding="same")
        self.bn, self.pool = nn.BatchNorm1d(2048), nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(2048, 512), nn.GELU(), nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = torch.cat(
            [
                F.gelu(self.conv6(x)),
                F.gelu(self.conv12(x)),
                F.gelu(self.conv18(x)),
                F.gelu(self.conv24(x)),
            ],
            dim=1,
        )
        x = self.pool(x).squeeze(-1)
        x = self.bn(x)
        return self.fc(x)


class FrequencyBranch(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.conv6, self.conv12 = nn.Conv1d(
            4, 256, 6, padding="same"
        ), nn.Conv1d(4, 256, 12, padding="same")
        self.conv18, self.conv24 = nn.Conv1d(
            4, 256, 18, padding="same"
        ), nn.Conv1d(4, 256, 24, padding="same")
        self.bn, self.pool = nn.BatchNorm1d(1024), nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(1024, 512), nn.GELU(), nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = torch.cat(
            [
                F.gelu(self.conv6(x)),
                F.gelu(self.conv12(x)),
                F.gelu(self.conv18(x)),
                F.gelu(self.conv24(x)),
            ],
            dim=1,
        )
        x = self.pool(x).squeeze(-1)
        x = self.bn(x)
        return self.fc(x)


class TransformerBranch(nn.Module):
    def __init__(
        self, d_model=256, nhead=8, num_layers=4, max_len=300, dropout=0.1
    ):
        super().__init__()
        self.embedding, self.pos_enc = nn.Linear(4, d_model), nn.Embedding(
            max_len + 1, d_model
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.norm, self.out = nn.LayerNorm(d_model), nn.Linear(d_model, 512)

    def forward(self, x):
        B, L, _ = x.shape
        x = self.embedding(x) + self.pos_enc(
            torch.arange(1, L + 1, device=x.device).unsqueeze(0)
        )
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        return self.out(self.norm(self.transformer(x))[:, 0, :])


class ViraExplorer(nn.Module):
    def __init__(self, cnn_drop, trans_drop, out_features):
        super().__init__()
        self.pattern = PatternBranch(cnn_drop)
        self.frequency = FrequencyBranch(cnn_drop)
        self.transformer = TransformerBranch(dropout=trans_drop)
        self.fc = nn.Sequential(
            nn.Linear(1536, 512),
            nn.GELU(),
            nn.Dropout(cnn_drop),
            nn.LayerNorm(512),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(cnn_drop),
            nn.Linear(128, out_features),
        )

    def forward(self, x):
        return self.fc(
            torch.cat(
                [self.pattern(x), self.frequency(x), self.transformer(x)], dim=1
            )
        )



class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, smoothing=0.05):
        super().__init__()
        self.alpha, self.gamma, self.smoothing = alpha, gamma, smoothing

    def forward(self, logits, targets):
        targets_smooth = targets.unsqueeze(1) * (
            1 - self.smoothing
        ) + 0.5 * self.smoothing
        bce = F.binary_cross_entropy_with_logits(
            logits, targets_smooth, reduction="none"
        )
        return (self.alpha * (1 - torch.exp(-bce)) ** self.gamma * bce).mean()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViraExplorer(
    cnn_drop=cfg["cnn_dropout"],
    trans_drop=cfg["trans_dropout"],
    out_features=cfg["out_features"],
).to(device)

if cfg["loss"] == "focal":
    criterion = FocalLoss()
elif cfg["loss"] == "bce":
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.CrossEntropyLoss()

if cfg["optimizer"] == "adamw":
    transformer_params = list(model.transformer.parameters())
    other_params = (
        list(model.pattern.parameters())
        + list(model.frequency.parameters())
        + list(model.fc.parameters())
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": other_params, "lr": 1e-4},
            {"params": transformer_params, "lr": 5e-5},
        ],
        weight_decay=1e-4,
    )
else:
    lr_base = 0.001 if args.config == "B" else 1.0
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_base)

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
    d_model = 256
    warmup_steps = 4000

    def aiayn_lambda(current_step):
        step = max(1, current_step)
        return (d_model**-0.5) * min(step**-0.5, step * (warmup_steps**-1.5))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, aiayn_lambda)


best_auc = 0
counter = 0
scaler = torch.amp.GradScaler("cuda") if torch.cuda.is_available() else None

if args.config == "A":
    for p in model.transformer.parameters():
        p.requires_grad = False

for epoch in range(cfg["epochs"]):
    if args.config == "A" and epoch == 5:
        for p in model.transformer.parameters():
            p.requires_grad = True
        print("Transformer sbloccato dopo il warmup.", flush=True)

    model.train()
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()

        if cfg["loss"] == "bce":
            y_batch_loss = y_batch.unsqueeze(1)
        else:
            y_batch_loss = y_batch

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                loss = criterion(model(X_batch), y_batch_loss)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = criterion(model(X_batch), y_batch_loss)
            loss.backward()
            optimizer.step()

        if cfg["lr_sched"] == "aiayn":
            scheduler.step()

    if cfg["lr_sched"] != "aiayn":
        scheduler.step()

    model.eval()
    preds, targets_list = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            out = model(X_batch.to(device))
            if cfg["loss"] == "ce":
                probs = F.softmax(out, dim=1)[:, 1]
            else:
                probs = torch.sigmoid(out)
            preds.extend(probs.cpu().numpy().flatten())
            targets_list.extend(y_batch.cpu().numpy().flatten())

    auc = roc_auc_score(targets_list, preds)
    print(f"Epoca {epoch + 1}/{cfg['epochs']} | Val AUC: {auc:.4f}", flush=True)

    if auc > best_auc:
        best_auc = auc
        counter = 0
        torch.save(model.state_dict(), BEST_MODEL_PATH)
    else:
        counter += 1

    if cfg["early_stop"] and counter >= cfg["patience"]:
        print(f"Early stopping intervenuto all'epoca {epoch + 1}", flush=True)
        break

if not cfg["early_stop"]:
    torch.save(model.state_dict(), BEST_MODEL_PATH)


print("\nInizio estrazione logit in formato TXT...", flush=True)
model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
model.eval()


def _find_single_csv(folder, label):
    all_csvs = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if len(all_csvs) == 0:
        print(f"[SKIP] '{label}': nessun CSV trovato in {folder}.", flush=True)
        return None
    test_csvs = [f for f in all_csvs if "test" in os.path.basename(f).lower()]
    if test_csvs:
        return test_csvs[0]
    return all_csvs[0]


DATASETS_TO_EVAL = {
    "val": (
        "ViraMiner/data/DNA_data/fullset_validation.csv"
        if os.path.exists("ViraMiner/data/DNA_data/fullset_validation.csv")
        else os.path.join(args.data_folder, "fullset_validation.csv")
    ),
    "test_VM": (
        "ViraMiner/data/DNA_data/fullset_test.csv"
        if os.path.exists("ViraMiner/data/DNA_data/fullset_test.csv")
        else os.path.join(args.data_folder, "fullset_test.csv")
    ),
}

_ecoli_csv = _find_single_csv(
    os.path.join(args.data_folder, "ecoli"), "test_ecoli"
)
if _ecoli_csv:
    DATASETS_TO_EVAL["test_ecoli"] = _ecoli_csv

_enhancer_csv = _find_single_csv(
    os.path.join(args.data_folder, "enhancer"), "test_enhancer"
)
if _enhancer_csv:
    DATASETS_TO_EVAL["test_enhancer"] = _enhancer_csv

# Dynamic deduplication of Anellovirus test set against active train dataset
_anell_test = os.path.join(
    args.data_folder, "unseen_anell", "test_data_anell_300_binary.csv"
)
if os.path.exists(_anell_test):
    anell_test_raw = load_seq_label_dataframe(_anell_test)
    train_seq_set = build_seq_set(train_df.iloc[:, 1].values)

    keep_mask = [
        (str(s).upper() not in train_seq_set)
        for s in anell_test_raw.iloc[:, 1].values
    ]
    anell_clean_df = anell_test_raw[keep_mask].reset_index(drop=True)
    removed_overlap = len(anell_test_raw) - len(anell_clean_df)
    print(
        f"[i] Anellovirus test evaluation dataset cleaned: {len(anell_clean_df)} / {len(anell_test_raw)} samples retained (removed {removed_overlap} train overlap).",
        flush=True,
    )

    clean_anell_tmp = os.path.join(
        SAVE_DIR, f"clean_test_anell_{args.task}.csv"
    )
    anell_clean_df.to_csv(clean_anell_tmp, index=False)
    DATASETS_TO_EVAL["test_anellovirus"] = clean_anell_tmp

_sim_base_dir = os.path.join(args.data_folder, "SIM")
_sim_files = sorted(
    glob.glob(os.path.join(_sim_base_dir, "**", "*.csv"), recursive=True)
)
_seen_sim_keys = {}
for _f in _sim_files:
    _rel_noext = os.path.splitext(os.path.relpath(_f, _sim_base_dir))[0]
    _key = "test_sim_" + _rel_noext.replace(os.sep, "_")
    if _key in _seen_sim_keys:
        print(
            f"[WARN] Chiave duplicata '{_key}' per i file SIM:\n"
            f"       {_seen_sim_keys[_key]}\n"
            f"       {_f}\n"
            f"       Il primo file verrà sovrascritto: verifica la struttura di {_sim_base_dir}.",
            flush=True,
        )
    _seen_sim_keys[_key] = _f
    DATASETS_TO_EVAL[_key] = _f

with torch.no_grad():
    for name, path in DATASETS_TO_EVAL.items():
        if not os.path.exists(path):
            print(
                f"[SKIP] '{name}': path non trovato ({path}). Aggiorna DATASETS_TO_EVAL.",
                flush=True,
            )
            continue

        df = load_seq_label_dataframe(path)
        loader = DataLoader(
            DNADataset(df, augment_type="none"),
            batch_size=64,
            shuffle=False,
            num_workers=2,
        )

        logits_list = []
        for X, _ in loader:
            out = model(X.to(device))
            if cfg["loss"] == "ce":
                logits_equivalent = out[:, 1] - out[:, 0]
                logits_list.extend(logits_equivalent.cpu().numpy().flatten())
            else:
                logits_list.extend(out.cpu().numpy().flatten())

        task_prefix = f"_{args.task}" if args.task != "viral" else ""
        filename = f"viraexplorer_cfg{args.config}{task_prefix}_{name}.txt"

        save_path = os.path.join(GLOBAL_LOGITS_DIR, filename)
        np.savetxt(save_path, np.array(logits_list), fmt="%.6f")
        print(f"Logit salvati in: {save_path}", flush=True)


if os.path.exists(BEST_MODEL_PATH):
    os.remove(BEST_MODEL_PATH)
clean_tmp_file = os.path.join(SAVE_DIR, f"clean_test_anell_{args.task}.csv")
if os.path.exists(clean_tmp_file):
    os.remove(clean_tmp_file)
print("Pesi temporanei (.pth) e file temporanei rimossi con successo.", flush=True)