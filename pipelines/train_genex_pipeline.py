import argparse
import glob
import math
import os
import random
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------
# ARGUMENT PARSER
# ---------------------------------------------------------
parser = argparse.ArgumentParser(
    description="GenExplorer (ViraExplorer) Multi-Task Training Pipeline (DNA Sequence Classification)"
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

# ---------------------------------------------------------
# CONFIGURATION MAP (C1, C2, C3)
# ---------------------------------------------------------
CONFIG_MAP = {
    "A": {  # Strategy C1: Focal Loss + Cosine Warmup
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
    "B": {  # Strategy C2: BCE + StepLR (Deterministic)
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
    "C": {  # Strategy C3: CE + Noam (AIAYN) + Mutation Augmentation
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
    SAVE_DIR, f"best_genexplorer_{args.task}_cfg_{args.config}.pth"
)
GLOBAL_LOGITS_DIR = "/home/simionricc/global_logits"
os.makedirs(GLOBAL_LOGITS_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)


def seed_everything(seed=2020):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


seed_everything(2020)

# ---------------------------------------------------------
# DATA LOADING & PREPROCESSING
# ---------------------------------------------------------
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


def get_task_dataframes(task_name, base_dir):
    if task_name == "viral":
        train_path = "ViraMiner/data/DNA_data/fullset_train.csv"
        val_path = "ViraMiner/data/DNA_data/fullset_validation.csv"
        if not os.path.exists(train_path):
            train_path = os.path.join(base_dir, "DNA_data", "fullset_train.csv")
            val_path = os.path.join(base_dir, "DNA_data", "fullset_validation.csv")
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
        train_path = os.path.join(base_dir, "enhancer", "bert_enhancer_dataset.csv")
        target_len = 200
    else:
        raise ValueError(f"Task non valida: {task_name}")

    full_df = load_seq_label_dataframe(train_path)
    full_df = full_df.sample(frac=1, random_state=2020).reset_index(drop=True)
    n_val = int(len(full_df) * 0.2)
    val_df = full_df.iloc[:n_val].reset_index(drop=True)
    train_df = full_df.iloc[n_val:].reset_index(drop=True)
    return train_df, val_df, target_len


train_df, val_df, TASK_SEQ_LEN = get_task_dataframes(args.task, args.data_folder)
print(
    f"[i] Task: '{args.task}' | Config: '{args.config}' | Seq Length L: {TASK_SEQ_LEN} bp | "
    f"Train samples: {len(train_df)} | Val samples: {len(val_df)}",
    flush=True,
)

MAPPING = {
    "A": [1, 0, 0, 0],
    "C": [0, 1, 0, 0],
    "G": [0, 0, 1, 0],
    "T": [0, 0, 0, 1],
    "N": [0, 0, 0, 0],
}


def mutate_sequence(seq, n_mutations=40):
    seq_list = list(seq)
    seq_len = len(seq_list)
    nucleotides = ["A", "C", "G", "T"]
    n_mut = min(n_mutations, max(1, int(seq_len * 0.13)))
    pos_to_mutate = random.sample(range(seq_len), n_mut)
    for pos in pos_to_mutate:
        current_nuc = seq_list[pos]
        choices = [n for n in nucleotides if n != current_nuc]
        seq_list[pos] = random.choice(choices)
    return "".join(seq_list)


def encode_sequence(seq, augment_type, target_len):
    seq = seq.upper()
    if augment_type == "rc_shift":
        complement = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
        if random.random() > 0.5:
            seq = "".join(complement.get(b, "N") for b in reversed(seq))
        shift = random.randint(-5, 5)
        seq = seq[max(0, shift) :] if shift > 0 else "N" * (-shift) + seq
    elif augment_type == "mutation":
        seq = mutate_sequence(seq, n_mutations=40)

    seq = seq[:target_len]
    encoded = [MAPPING.get(b, [0, 0, 0, 0]) for b in seq]
    while len(encoded) < target_len:
        encoded.append([0, 0, 0, 0])
    return encoded


class DNADataset(Dataset):
    def __init__(self, dataframe, augment_type="none", target_len=300):
        self.sequences = dataframe[1].values
        self.labels = dataframe[2].values
        self.augment_type = augment_type
        self.target_len = target_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x = torch.tensor(
            encode_sequence(
                self.sequences[idx], self.augment_type, self.target_len
            ),
            dtype=torch.float32,
        )
        if cfg["loss"] == "ce":
            y = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        else:
            y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y


train_loader = DataLoader(
    DNADataset(train_df, augment_type=cfg["augment"], target_len=TASK_SEQ_LEN),
    batch_size=cfg["batch_size"],
    shuffle=True,
    num_workers=2,
    pin_memory=True,
)
val_loader = DataLoader(
    DNADataset(val_df, augment_type="none", target_len=TASK_SEQ_LEN),
    batch_size=cfg["batch_size"],
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

# ---------------------------------------------------------
# GENEXPLORER ARCHITECTURE
# ---------------------------------------------------------
class PatternBranch(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.conv6 = nn.Conv1d(4, 512, 6, padding="same")
        self.conv12 = nn.Conv1d(4, 512, 12, padding="same")
        self.conv18 = nn.Conv1d(4, 512, 18, padding="same")
        self.conv24 = nn.Conv1d(4, 512, 24, padding="same")
        self.bn = nn.BatchNorm1d(2048)
        self.pool = nn.AdaptiveMaxPool1d(1)
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
        self.conv6 = nn.Conv1d(4, 256, 6, padding="same")
        self.conv12 = nn.Conv1d(4, 256, 12, padding="same")
        self.conv18 = nn.Conv1d(4, 256, 18, padding="same")
        self.conv24 = nn.Conv1d(4, 256, 24, padding="same")
        self.bn = nn.BatchNorm1d(1024)
        self.pool = nn.AdaptiveAvgPool1d(1)
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
        self, d_model=256, nhead=8, num_layers=4, max_len=512, dropout=0.1
    ):
        super().__init__()
        self.embedding = nn.Linear(4, d_model)
        self.pos_enc = nn.Embedding(max_len + 1, d_model)
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
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, 512)

    def forward(self, x):
        B, L, _ = x.shape
        x = self.embedding(x) + self.pos_enc(
            torch.arange(1, L + 1, device=x.device).unsqueeze(0)
        )
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        return self.out(self.norm(self.transformer(x))[:, 0, :])


class GenExplorer(nn.Module):
    def __init__(self, cnn_drop, trans_drop, out_features, max_len=512):
        super().__init__()
        self.pattern = PatternBranch(cnn_drop)
        self.frequency = FrequencyBranch(cnn_drop)
        self.transformer = TransformerBranch(
            dropout=trans_drop, max_len=max_len
        )
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


# ---------------------------------------------------------
# OPTIMIZER & SCHEDULER SETUP
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GenExplorer(
    cnn_drop=cfg["cnn_dropout"],
    trans_drop=cfg["trans_dropout"],
    out_features=cfg["out_features"],
    max_len=max(512, TASK_SEQ_LEN),
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

# ---------------------------------------------------------
# TRAINING LOOP
# ---------------------------------------------------------
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
        print("[i] Transformer unfreezed post-warmup.", flush=True)

    model.train()
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()

        y_batch_loss = (
            y_batch.unsqueeze(1) if cfg["loss"] == "bce" else y_batch
        )

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
            probs = (
                F.softmax(out, dim=1)[:, 1]
                if cfg["loss"] == "ce"
                else torch.sigmoid(out)
            )
            preds.extend(probs.cpu().numpy().flatten())
            targets_list.extend(y_batch.cpu().numpy().flatten())

    auc = roc_auc_score(targets_list, preds)
    print(f"Epoch {epoch + 1}/{cfg['epochs']} | Val AUC: {auc:.4f}", flush=True)

    if auc > best_auc:
        best_auc = auc
        counter = 0
        torch.save(model.state_dict(), BEST_MODEL_PATH)
    else:
        counter += 1

    if cfg["early_stop"] and counter >= cfg["patience"]:
        print(f"Early stopping at epoch {epoch + 1}", flush=True)
        break

if not cfg["early_stop"]:
    torch.save(model.state_dict(), BEST_MODEL_PATH)

# ---------------------------------------------------------
# TARGETED TEST EVALUATION (CORRETTO & CONDIZIONATO)
# ---------------------------------------------------------
print("\n--- Automatic Cascade Evaluation ---", flush=True)
model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
model.eval()


def _resolve_test_file(folder, default_name):
    target = os.path.join(folder, default_name)
    if os.path.exists(target):
        return target
    all_csvs = sorted(glob.glob(os.path.join(folder, "*.csv")))
    test_csvs = [f for f in all_csvs if "test" in os.path.basename(f).lower()]
    return test_csvs[0] if test_csvs else (all_csvs[0] if all_csvs else None)


DATASETS_TO_EVAL = {}

if args.task == "viral":
    vm_test = os.path.join(args.data_folder, "fullset_test.csv")
    if not os.path.exists(vm_test):
        vm_test = os.path.join(args.data_folder, "DNA_data", "fullset_test.csv")
    if os.path.exists(vm_test):
        DATASETS_TO_EVAL["test_VM"] = vm_test

    _sim_base_dir = os.path.join(args.data_folder, "SIM")
    if os.path.exists(_sim_base_dir):
        for _f in sorted(
            glob.glob(os.path.join(_sim_base_dir, "**", "*.csv"), recursive=True)
        ):
            _rel_noext = os.path.splitext(os.path.relpath(_f, _sim_base_dir))[0]
            _key = "test_sim_" + _rel_noext.replace(os.sep, "_")
            DATASETS_TO_EVAL[_key] = _f

elif args.task == "anellovirus":
    anell_test = os.path.join(
        args.data_folder, "unseen_anell", "test_data_anell_300_binary.csv"
    )
    if os.path.exists(anell_test):
        DATASETS_TO_EVAL["test_anellovirus"] = anell_test

elif args.task == "ecoli":
    ecoli_test = _resolve_test_file(
        os.path.join(args.data_folder, "ecoli"), "ecoli_independent_test_NT.csv"
    )
    if ecoli_test:
        DATASETS_TO_EVAL["test_ecoli"] = ecoli_test

elif args.task == "enhancer":
    enh_test = _resolve_test_file(
        os.path.join(args.data_folder, "enhancer"),
        "bert_enhancer_ind_test_set.csv",
    )
    if enh_test:
        DATASETS_TO_EVAL["test_enhancer"] = enh_test

with torch.no_grad():
    for name, path in DATASETS_TO_EVAL.items():
        if not path or not os.path.exists(path):
            continue

        df_eval = load_seq_label_dataframe(path)
        eval_len = (
            81
            if "ecoli" in name
            else (200 if "enhancer" in name else TASK_SEQ_LEN)
        )

        loader = DataLoader(
            DNADataset(df_eval, augment_type="none", target_len=eval_len),
            batch_size=64,
            shuffle=False,
            num_workers=2,
        )

        logits_list = []
        for X, _ in loader:
            out = model(X.to(device))
            logits_val = (
                (out[:, 1] - out[:, 0]) if cfg["loss"] == "ce" else out
            )
            logits_list.extend(logits_val.cpu().numpy().flatten())

        filename = f"genexplorer_cfg{args.config}_{args.task}_{name}.txt"
        save_path = os.path.join(GLOBAL_LOGITS_DIR, filename)
        np.savetxt(save_path, np.array(logits_list), fmt="%.6f")
        print(
            f"  Logits salvati ({len(logits_list)} campioni): {filename}",
            flush=True,
        )

if os.path.exists(BEST_MODEL_PATH):
    os.remove(BEST_MODEL_PATH)
print("Pipeline completed successfully.", flush=True)