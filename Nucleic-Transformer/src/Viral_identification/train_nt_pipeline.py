import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import random
import argparse
import math
import glob
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score

from Functions import *
from Dataset import *
from Network import *
from LrScheduler import *

# ========================================== #
# 1. PARSING DEGLI ARGOMENTI                 #
# ========================================== #
parser = argparse.ArgumentParser(
    description="Nucleic Transformer Multi-Task Training Pipeline (DNA Sequence Classification)"
)
parser.add_argument('--config', type=str, required=True, choices=['A', 'B', 'C'],
                     help="Configurazione di training")
parser.add_argument('--task', type=str, default='viral',
                     choices=['viral', 'ecoli', 'enhancer', 'anellovirus'],
                     help="Task biologico su cui addestrare il modello (default: viral)")
parser.add_argument('--data_folder', '-d', default='/home/simionricc/datasets',
                     help="Path alla cartella condivisa con ecoli/enhancer/SIM/unseen_anell")
parser.add_argument('--viraminer_folder', default='..',
                     help="Path (relativo alla working directory dello script, tipicamente "
                          "Viral_identification/) ai file nativi fullset_train/validation/test.csv")
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
# 2. MAPPATURA IPERPARAMETRI DELLA TABELLA   #
# ========================================== #
CONFIG_MAP = {
    'A': {  # Configurazione stile ViraExplorer
        'epochs': 120, 'early_stop': True, 'patience': 15, 'batch_size': 128,
        'loss': 'focal', 'optimizer': 'adamw', 'lr_sched': 'cosine', 'augment': 'rc_shift',
        'dropout': 0.1, 'nclass': 1
    },
    'B': {  # Configurazione stile FullE
        'epochs': 30, 'early_stop': False, 'patience': 0, 'batch_size': 128,
        'loss': 'bce', 'optimizer': 'adam', 'lr_sched': 'step', 'augment': 'none',
        'dropout': 0.2, 'nclass': 1
    },
    'C': {  # Configurazione nativa Nucleic Transformer
        'epochs': 60, 'early_stop': False, 'patience': 0, 'batch_size': 128,
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
# 3. HELPER CARICAMENTO DATASET & SPLIT      #
# ========================================== #
def _peek_columns(path):
    """Legge solo la prima riga per capire se c'è un header e quante colonne ci sono."""
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
    """Autodetect del formato CSV con normalizzazione uppercase delle sequenze."""
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
    """Ritorna i DataFrame di train e validation per la task selezionata con fallback sui path."""
    if task_name == 'viral':
        train_path = os.path.join(viraminer_folder, "fullset_train.csv")
        val_path = os.path.join(viraminer_folder, "fullset_validation.csv")

        # Fallback automatico se il file non esiste nel percorso relativo
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
        raise ValueError(f"Task non valida: {task_name}")

    full_df = load_seq_label_dataframe(train_path)
    full_df = full_df.sample(frac=1, random_state=2020).reset_index(drop=True)
    n_val = int(len(full_df) * 0.2)
    val_df = full_df.iloc[:n_val].reset_index(drop=True)
    train_df = full_df.iloc[n_val:].reset_index(drop=True)
    return train_df, val_df


# ========================================== #
# 4. DATA AUGMENTATION & DATASET ADATTIVO    #
# ========================================== #
def mutate_string_sequence(seq, n_mutations=40):
    seq_list = list(seq)
    seq_len = len(seq_list)
    nucleotides = ['A', 'C', 'G', 'T']
    pos_to_mutate = random.sample(range(seq_len), min(n_mutations, seq_len))
    for pos in pos_to_mutate:
        current_nuc = seq_list[pos].upper()
        choices = [n for n in nucleotides if n != current_nuc]
        seq_list[pos] = random.choice(choices)
    return "".join(seq_list)


class ParametricViraminerDataset(Dataset):
    def __init__(self, dataframe, augment_type='none'):
        self.sequences = dataframe.iloc[:, 1].values
        self.labels = dataframe.iloc[:, 2].values
        self.augment_type = augment_type

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]

        if self.augment_type == 'rc_shift':
            complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
            if random.random() > 0.5:
                seq = ''.join(complement.get(b, 'N') for b in reversed(seq))
            shift = random.randint(-5, 5)
            seq = seq[max(0, shift):] if shift > 0 else 'N' * (-shift) + seq
            seq = seq[:300]
        elif self.augment_type == 'mutation':
            seq = mutate_string_sequence(seq, n_mutations=40)

        temp_ds = ViraminerDataset([seq], [label])
        return temp_ds[0]


# Caricamento dati per la task selezionata
train_df, val_df = get_task_dataframes(args.task, args.data_folder, args.viraminer_folder)
print(f"[i] Task: '{args.task}' | Config: '{args.config}' | "
      f"Train samples: {len(train_df)} | Val samples: {len(val_df)}")

train_loader = DataLoader(ParametricViraminerDataset(train_df, augment_type=cfg['augment']),
                           batch_size=cfg['batch_size'], shuffle=True, num_workers=2)
val_loader = DataLoader(ParametricViraminerDataset(val_df, augment_type='none'),
                         batch_size=cfg['batch_size'] * 2, shuffle=False)


# ========================================== #
# 5. LOSS FUNCTIONS & SETUP INIZIALE         #
# ========================================== #
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, smoothing=0.05):
        super().__init__()
        self.alpha, self.gamma, self.smoothing = alpha, gamma, smoothing

    def forward(self, logits, targets):
        targets_smooth = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        bce = F.binary_cross_entropy_with_logits(logits, targets_smooth, reduction='none')
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

# Setup Ottimizzatore ed LR Scheduler
if cfg['optimizer'] == 'adamw':
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
else:
    lr_base = 0.001 if args.config == 'B' else 1.0
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_base, weight_decay=0)

if cfg['lr_sched'] == 'cosine':
    def lr_lambda(epoch):
        if epoch < 5:
            return epoch / 5
        return 0.5 * (1 + math.cos(math.pi * (epoch - 5) / (cfg['epochs'] - 5)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
elif cfg['lr_sched'] == 'step':
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
else:
    scheduler = lr_AIAYN(optimizer, opts.ninp, opts.warmup_steps, opts.lr_scale)

# ========================================== #
# 6. LOOP DI TRAINING CON SCALER (AMP NATIVO)#
# ========================================== #
scaler = GradScaler()
best_auc = 0
counter = 0

for epoch in range(cfg['epochs']):
    model.train()
    total_loss = 0
    for step, data in enumerate(train_loader):
        if cfg['lr_sched'] == 'aiayn':
            scheduler.step()

        src = data['data'].to(device).long()
        if cfg['loss'] == 'ce':
            labels = data['labels'].to(device).long()
        else:
            labels = data['labels'].to(device).float().unsqueeze(1)

        optimizer.zero_grad()

        with autocast():
            output = model(src)
            if cfg['loss'] == 'ce':
                loss = torch.mean(criterion(output, labels))
            else:
                loss = criterion(output, labels)

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()

    if cfg['lr_sched'] != 'aiayn':
        scheduler.step()

    model.eval()
    val_preds, val_targets = [], []
    with torch.no_grad():
        for data in val_loader:
            src = data['data'].to(device).long()
            out = model(src)
            if cfg['loss'] == 'ce':
                probs = F.softmax(out, dim=1)[:, 1]
            else:
                probs = torch.sigmoid(out)
            val_preds.extend(probs.cpu().numpy().flatten())
            val_targets.extend(data['labels'].numpy().flatten())

    auc = roc_auc_score(val_targets, val_preds)
    print(f"Epoca {epoch + 1}/{cfg['epochs']} | Loss: {total_loss / len(train_loader):.4f} | Val AUC: {auc:.4f}")

    if auc > best_auc:
        best_auc = auc
        counter = 0
        torch.save(model.state_dict(), best_model_path)
    else:
        counter += 1

    if cfg['early_stop'] and counter >= cfg['patience']:
        print("Early stopping attivato.")
        break

if not cfg['early_stop']:
    torch.save(model.state_dict(), best_model_path)

# ========================================== #
# 7. VALUTAZIONE IN CASCATA SUI DATASET      #
# ========================================== #
print("\nInizio estrazione logit in formato TXT...")
model.load_state_dict(torch.load(best_model_path, map_location=device))
model.eval()


def _find_single_csv(folder, label):
    """Trova il CSV di test in una cartella."""
    all_csvs = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if len(all_csvs) == 0:
        print(f"[SKIP] '{label}': nessun CSV trovato in {folder}.")
        return None
    test_csvs = [f for f in all_csvs if 'test' in os.path.basename(f).lower()]
    if test_csvs:
        return test_csvs[0]
    return all_csvs[0]


def _loader_from_csv(path):
    df_extra = load_seq_label_dataframe(path)
    return DataLoader(ParametricViraminerDataset(df_extra, augment_type='none'),
                       batch_size=cfg['batch_size'] * 2, shuffle=False)


DATASETS_TO_EVAL = {}

# val/test_VM nativi ViraMiner con ricerca fallback
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
    print(f"[SKIP] 'val': path non trovato ({val_native_path}).")

if os.path.exists(test_native_path):
    DATASETS_TO_EVAL['test_VM'] = _loader_from_csv(test_native_path)
else:
    print(f"[SKIP] 'test_VM': path non trovato ({test_native_path}).")

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
    print(f"[SKIP] 'test_anellovirus': path non trovato ({_anell_test}).")

_sim_base_dir = os.path.join(args.data_folder, "SIM")
_sim_files = sorted(glob.glob(os.path.join(_sim_base_dir, "**", "*.csv"), recursive=True))

_seen_sim_keys = {}
for _f in _sim_files:
    _rel_noext = os.path.splitext(os.path.relpath(_f, _sim_base_dir))[0]
    _key = "test_sim_" + _rel_noext.replace(os.sep, "_")
    _seen_sim_keys[_key] = _f
    DATASETS_TO_EVAL[_key] = _loader_from_csv(_f)

with torch.no_grad():
    for name, loader in DATASETS_TO_EVAL.items():
        logits_list = []
        for data in loader:
            src = data['data'].to(device).long()
            out = model(src)
            if cfg['loss'] == 'ce':
                logits_equivalent = out[:, 1] - out[:, 0]
                logits_list.extend(logits_equivalent.cpu().numpy().flatten())
            else:
                logits_list.extend(out.cpu().numpy().flatten())

        task_prefix = f"_{args.task}" if args.task != 'viral' else ""
        filename = f"nucleictransformer_cfg{args.config}{task_prefix}_{name}.txt"
        np.savetxt(os.path.join(GLOBAL_LOGITS_DIR, filename), np.array(logits_list), fmt='%.6f')
        print(f"Logit salvati temporaneamente: {filename}")

# ========================================== #
# 8. PULIZIA AUTOMATICA DEL CHECKPOINT       #
# ========================================== #
import shutil
if os.path.exists(checkpoints_folder):
    shutil.rmtree(checkpoints_folder)
print("Cartella pesi pesanti rimossa per preservare lo spazio sul cluster.")