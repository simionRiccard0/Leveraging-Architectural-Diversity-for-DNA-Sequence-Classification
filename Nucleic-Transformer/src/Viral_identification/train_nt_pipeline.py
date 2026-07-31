import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import random
import argparse
import math
import glob
import shutil
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score

torch.backends.cudnn.benchmark = True

from Functions import *
from Network import *
from LrScheduler import *

# ========================================== #
# 0. CONSTANTS & FAST TRANSLATION TABLES     #
# ========================================== #
MAX_LEN = 300
NUC_TO_INT = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

CLEAN_TRANS = str.maketrans({
    'N': 'A', 'R': 'A', 'Y': 'C', 'S': 'G', 'W': 'A',
    'K': 'G', 'M': 'A', 'B': 'C', 'D': 'A', 'H': 'A', 'V': 'A'
})
RC_TRANS = str.maketrans('ACGT', 'TGCA')

NUC_MAP = np.zeros(256, dtype=np.int64)
NUC_MAP[ord('A')] = 0
NUC_MAP[ord('C')] = 1
NUC_MAP[ord('G')] = 2
NUC_MAP[ord('T')] = 3

# ========================================== #
# 1. CLI ARGUMENT PARSING                    #
# ========================================== #
parser = argparse.ArgumentParser(
    description="Nucleic Transformer Multi-Task High-Performance Pipeline"
)
parser.add_argument('--config', type=str, required=True, choices=['A', 'B', 'C'],
                     help="Training configuration strategy (A, B, C)")
parser.add_argument('--task', type=str, default='viral',
                     choices=['viral', 'ecoli', 'enhancer', 'anellovirus'],
                     help="Target biological training task (default: viral)")
parser.add_argument('--data_folder', '-d', default='/home/simionricc/datasets',
                     help="Root directory for external benchmark datasets")
parser.add_argument('--viraminer_folder', default='..',
                     help="Path to native fullset_train/validation/test.csv files")
parser.add_argument('--gpu_id', type=str, default='0')
parser.add_argument('--ntoken', type=int, default=4)
parser.add_argument('--ninp', type=int, default=512)
parser.add_argument('--nhead', type=int, default=8)
parser.add_argument('--nhid', type=int, default=2048)
parser.add_argument('--nlayers', type=int, default=6)
parser.add_argument('--kmers', type=int, nargs='+', default=[2, 3, 4, 5, 6])
parser.add_argument('--kmer_aggregation', dest='kmer_aggregation', action='store_true')
parser.set_defaults(kmer_aggregation=True)
parser.add_argument('--warmup_steps', type=int, default=3200)
parser.add_argument('--lr_scale', type=float, default=0.1)
opts = parser.parse_args()
args = opts

# ========================================== #
# 2. HYPERPARAMETER CONFIGURATIONS           #
# ========================================== #
CONFIG_MAP = {
    'A': {  # ViraExplorer Native Strategy
        'epochs': 120, 'early_stop': True, 'patience': 15, 'batch_size': 16,
        'loss': 'focal', 'optimizer': 'adamw', 'lr_sched': 'cosine', 'augment': 'rc_shift',
        'dropout': 0.1, 'nclass': 1
    },
    'B': {  # FullE Native Strategy
        'epochs': 30, 'early_stop': False, 'patience': 0, 'batch_size': 16,
        'loss': 'bce', 'optimizer': 'adam', 'lr_sched': 'step', 'augment': 'none',
        'dropout': 0.2, 'nclass': 1
    },
    'C': {  # Nucleic Transformer Native Strategy
        'epochs': 60, 'early_stop': False, 'patience': 0, 'batch_size': 16,
        'loss': 'ce', 'optimizer': 'adam', 'lr_sched': 'aiayn', 'augment': 'mutation',
        'dropout': 0.1, 'nclass': 2
    }
}
cfg = CONFIG_MAP[args.config]

GLOBAL_LOGITS_DIR = f"./global_logits_temp_{args.task}_{args.config}"
os.makedirs(GLOBAL_LOGITS_DIR, exist_ok=True)

checkpoints_folder = f'checkpoints_{args.task}_cfg_{args.config}'
os.makedirs(checkpoints_folder, exist_ok=True)
best_model_path = os.path.join(checkpoints_folder, 'best_model.pt')

# ========================================== #
# 3. DATA LOADING HELPERS                    #
# ========================================== #
def _peek_columns(path):
    with open(path, 'r') as fh:
        first_line = fh.readline().strip()
    fields = first_line.split(',')

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


def get_task_dataframes(task_name, base_dir, viraminer_folder):
    if task_name == 'viral':
        train_path = os.path.join(viraminer_folder, "fullset_train.csv")
        val_path = os.path.join(viraminer_folder, "fullset_validation.csv")

        if not os.path.exists(train_path):
            alt_train_dna = os.path.join(base_dir, "DNA_data", "fullset_train.csv")
            alt_val_dna = os.path.join(base_dir, "DNA_data", "fullset_validation.csv")
            if os.path.exists(alt_train_dna):
                train_path, val_path = alt_train_dna, alt_val_dna
            else:
                train_path = os.path.join(base_dir, "fullset_train.csv")
                val_path = os.path.join(base_dir, "fullset_validation.csv")

        train_df = load_seq_label_dataframe(train_path)
        val_df = load_seq_label_dataframe(val_path)
        return train_df, val_df

    elif task_name == 'ecoli':
        train_path = os.path.join(base_dir, "ecoli", "ecoli_train_NT.csv")
    elif task_name == 'enhancer':
        train_path = os.path.join(base_dir, "enhancer", "bert_enhancer_dataset.csv")
    elif task_name == 'anellovirus':
        train_path = os.path.join(base_dir, "unseen_anell", "training_data_anell_300_binary.csv")
    else:
        raise ValueError(f"Invalid task: {task_name}")

    full_df = load_seq_label_dataframe(train_path)
    full_df = full_df.sample(frac=1, random_state=2020).reset_index(drop=True)
    n_val = int(len(full_df) * 0.2)
    val_df = full_df.iloc[:n_val].reset_index(drop=True)
    train_df = full_df.iloc[n_val:].reset_index(drop=True)
    return train_df, val_df


# ========================================== #
# 4. DATASET                                 #
# ========================================== #
class FastNTDataset(Dataset):
    def __init__(self, dataframe, augment_type='none', target_len=300):
        raw_seqs = dataframe.iloc[:, 1].values
        labels = dataframe.iloc[:, 2].values
        self.augment_type = augment_type
        self.target_len = target_len

        cleaned_list = []
        for s in raw_seqs:
            s_clean = str(s).upper().translate(CLEAN_TRANS)
            if len(s_clean) < target_len:
                s_clean = s_clean.ljust(target_len, 'A')
            else:
                s_clean = s_clean[:target_len]
            cleaned_list.append(s_clean)

        self.sequences = cleaned_list
        self.labels = torch.tensor(labels, dtype=torch.float32)

        full_bytes = "".join(cleaned_list).encode('ascii')
        byte_matrix = np.frombuffer(full_bytes, dtype=np.uint8).reshape(len(cleaned_list), target_len)
        encoded_matrix = NUC_MAP[byte_matrix]

        self.data_tensor = torch.from_numpy(encoded_matrix)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        if self.augment_type == 'none':
            return {
                'data': self.data_tensor[idx],
                'labels': self.labels[idx]
            }

        seq = self.sequences[idx]
        label = self.labels[idx]

        if self.augment_type == 'rc_shift':
            if random.random() > 0.5:
                seq = seq.translate(RC_TRANS)[::-1]
            shift = random.randint(-5, 5)
            if shift > 0:
                seq = seq[shift:] + 'A' * shift
            elif shift < 0:
                seq = 'A' * (-shift) + seq[:shift]

        elif self.augment_type == 'mutation':
            seq_list = list(seq)
            n_mut = min(40, len(seq_list))
            for pos in random.sample(range(len(seq_list)), n_mut):
                seq_list[pos] = random.choice(['A', 'C', 'G', 'T'])
            seq = "".join(seq_list)

        encoded = [NUC_TO_INT.get(b, 0) for b in seq]
        return {
            'data': torch.tensor(encoded, dtype=torch.long),
            'labels': label
        }


# Load dataset splits
train_df, val_df = get_task_dataframes(args.task, args.data_folder, args.viraminer_folder)

# Micro-batching: batch size 8 for 'viral' to guarantee zero OOM on self-attention
train_batch_size = 8 if args.task == 'viral' else cfg['batch_size']

print(f"[i] Task: '{args.task}' | Config: '{args.config}' | Input Length: {MAX_LEN} bp | "
      f"Train Batch Size: {train_batch_size} | Train: {len(train_df)} | Val: {len(val_df)}", flush=True)

train_loader = DataLoader(
    FastNTDataset(train_df, augment_type=cfg['augment']),
    batch_size=train_batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)

val_loader = DataLoader(
    FastNTDataset(val_df, augment_type='none'),
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

# ========================================== #
# 5. LOSS FUNCTIONS & MODEL SETUP            #
# ========================================== #
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, smoothing=0.05):
        super().__init__()
        self.alpha, self.gamma, self.smoothing = alpha, gamma, smoothing

    def forward(self, logits, targets):
        logits = logits.float()
        targets = targets.float()
        targets_smooth = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        bce = F.binary_cross_entropy_with_logits(logits, targets_smooth, reduction='none')
        bce = torch.clamp(bce, min=1e-7, max=50.0)
        return (self.alpha * (1 - torch.exp(-bce)) ** self.gamma * bce).mean()


seed_everything(2020)
os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_id
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NucleicTransformer(opts.ntoken, cfg['nclass'], opts.ninp, opts.nhead, opts.nhid,
                            opts.nlayers, opts.kmer_aggregation, kmers=opts.kmers,
                            dropout=cfg['dropout']).to(device)

if cfg['loss'] == 'focal':
    criterion = FocalLoss()
elif cfg['loss'] == 'bce':
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.CrossEntropyLoss(reduction='none')

if cfg['optimizer'] == 'adamw':
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
else:
    lr_base = 0.001 if args.config == 'B' else 1.0
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_base, weight_decay=0, eps=1e-8)

if cfg['lr_sched'] == 'cosine':
    def lr_lambda(epoch):
        if epoch < 5:
            return (epoch + 1) / 5
        return 0.5 * (1 + math.cos(math.pi * (epoch - 5) / (cfg['epochs'] - 5)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
elif cfg['lr_sched'] == 'step':
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
else:
    scheduler = lr_AIAYN(optimizer, opts.ninp, opts.warmup_steps, opts.lr_scale)

# ========================================== #
# 6. TRAINING LOOP                           #
# ========================================== #
# Accumulate 4 steps for 'viral' (8 x 4 = effective batch size of 32)
GRAD_ACCUM_STEPS = 4 if args.task == 'viral' else 2
scaler = GradScaler()
best_auc = 0
counter = 0

for epoch in range(cfg['epochs']):
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    for step, data in enumerate(train_loader):
        if cfg['lr_sched'] == 'aiayn':
            scheduler.step()

        src = data['data'].to(device, non_blocking=True).long()
        if cfg['loss'] == 'ce':
            labels = data['labels'].to(device, non_blocking=True).long()
        else:
            labels = data['labels'].to(device, non_blocking=True).float().unsqueeze(1)

        with autocast():
            output = model(src).float()  # Force FP32 output for numerical stability
            if cfg['loss'] == 'ce':
                loss = torch.mean(criterion(output, labels))
            else:
                loss = criterion(output, labels)

            loss = loss / GRAD_ACCUM_STEPS

        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad()
            continue

        scaler.scale(loss).backward()

        if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * GRAD_ACCUM_STEPS

    if cfg['lr_sched'] != 'aiayn':
        scheduler.step()

    model.eval()
    val_preds, val_targets = [], []
    with torch.no_grad():
        for data in val_loader:
            src = data['data'].to(device, non_blocking=True).long()
            out = model(src).float()  # Stable FP32 evaluation
            if cfg['loss'] == 'ce':
                probs = F.softmax(out, dim=1)[:, 1]
            else:
                probs = torch.sigmoid(out)

            probs_np = np.nan_to_num(probs.cpu().numpy().flatten(), nan=0.5, posinf=1.0, neginf=0.0)
            val_preds.extend(probs_np)
            val_targets.extend(data['labels'].numpy().flatten())

    auc = roc_auc_score(val_targets, val_preds)
    print(f"Epoch {epoch + 1}/{cfg['epochs']} | Loss: {total_loss / len(train_loader):.4f} | Val AUC: {auc:.4f}", flush=True)

    if auc > best_auc:
        best_auc = auc
        counter = 0
        torch.save(model.state_dict(), best_model_path)
    else:
        counter += 1

    if cfg['early_stop'] and counter >= cfg['patience']:
        print("Early stopping triggered.", flush=True)
        break

if not cfg['early_stop']:
    torch.save(model.state_dict(), best_model_path)

if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ========================================== #
# 7. CASCADE EVALUATION                      #
# ========================================== #
print("\nStarting high-speed logit extraction...", flush=True)
model.load_state_dict(torch.load(best_model_path, map_location=device))
model.eval()


def _find_single_csv(folder, label):
    all_csvs = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if len(all_csvs) == 0:
        print(f"[SKIP] '{label}': No CSV found in {folder}.", flush=True)
        return None
    test_csvs = [f for f in all_csvs if 'test' in os.path.basename(f).lower()]
    if test_csvs:
        return test_csvs[0]
    return all_csvs[0]


def _loader_from_csv(path, batch_size=32):
    df_extra = load_seq_label_dataframe(path)
    return DataLoader(
        FastNTDataset(df_extra, augment_type='none'),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )


DATASETS_TO_EVAL = {}

val_native_path = os.path.join(args.viraminer_folder, "fullset_validation.csv")
test_native_path = os.path.join(args.viraminer_folder, "fullset_test.csv")

if not os.path.exists(val_native_path):
    for candidate in [os.path.join(args.data_folder, "DNA_data", "fullset_validation.csv"),
                      os.path.join(args.data_folder, "fullset_validation.csv")]:
        if os.path.exists(candidate):
            val_native_path = candidate
            break

if not os.path.exists(test_native_path):
    for candidate in [os.path.join(args.data_folder, "DNA_data", "fullset_test.csv"),
                      os.path.join(args.data_folder, "fullset_test.csv")]:
        if os.path.exists(candidate):
            test_native_path = candidate
            break

if os.path.exists(val_native_path):
    DATASETS_TO_EVAL['val'] = _loader_from_csv(val_native_path)
else:
    print(f"[SKIP] 'val': Path not found ({val_native_path}).", flush=True)

if os.path.exists(test_native_path):
    DATASETS_TO_EVAL['test_VM'] = _loader_from_csv(test_native_path)
else:
    print(f"[SKIP] 'test_VM': Path not found ({test_native_path}).", flush=True)

_ecoli_csv = _find_single_csv(os.path.join(args.data_folder, "ecoli"), "test_ecoli")
if _ecoli_csv:
    DATASETS_TO_EVAL['test_ecoli'] = _loader_from_csv(_ecoli_csv)

_enhancer_csv = _find_single_csv(os.path.join(args.data_folder, "enhancer"), "test_enhancer")
if _enhancer_csv:
    DATASETS_TO_EVAL['test_enhancer'] = _loader_from_csv(_enhancer_csv)

_anell_test = os.path.join(args.data_folder, "unseen_anell", "test_data_anell_300_binary.csv")
if os.path.exists(_anell_test):
    DATASETS_TO_EVAL['test_anellovirus'] = _loader_from_csv(_anell_test)
else:
    print(f"[SKIP] 'test_anellovirus': Path not found ({_anell_test}).", flush=True)

_sim_base_dir = os.path.join(args.data_folder, "SIM")
_sim_files = sorted(glob.glob(os.path.join(_sim_base_dir, "**", "*.csv"), recursive=True))

for _f in _sim_files:
    _rel_noext = os.path.splitext(os.path.relpath(_f, _sim_base_dir))[0]
    _key = "test_sim_" + _rel_noext.replace(os.sep, "_")
    DATASETS_TO_EVAL[_key] = _loader_from_csv(_f)

with torch.no_grad():
    for name, loader in DATASETS_TO_EVAL.items():
        logits_list = []
        for data in loader:
            src = data['data'].to(device, non_blocking=True).long()
            out = model(src).float()  # Stable FP32 evaluation
            if cfg['loss'] == 'ce':
                logits_equivalent = out[:, 1] - out[:, 0]
                probs_flat = np.nan_to_num(logits_equivalent.cpu().numpy().flatten(), nan=0.0)
            else:
                probs_flat = np.nan_to_num(out.cpu().numpy().flatten(), nan=0.0)
            logits_list.extend(probs_flat)

        task_prefix = f"_{args.task}" if args.task != 'viral' else ""
        filename = f"nucleictransformer_cfg{args.config}{task_prefix}_{name}.txt"
        np.savetxt(os.path.join(GLOBAL_LOGITS_DIR, filename), np.array(logits_list), fmt='%.6f')
        print(f"Logits saved: {filename}", flush=True)

# ========================================== #
# 8. CHECKPOINT CLEANUP                      #
# ========================================== #
if os.path.exists(checkpoints_folder):
    shutil.rmtree(checkpoints_folder)
print("Temporary checkpoints folder removed successfully.", flush=True)