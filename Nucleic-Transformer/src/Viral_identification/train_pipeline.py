import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import random
import argparse
import math
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score

from Functions import *
from Dataset import *
from Network import *
from LrScheduler import *

# Parsing degli argomenti da Slurm
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, choices=['A', 'B', 'C'], help="Configurazione di training")
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
# FIX: il resto dello script (mappatura CONFIG_MAP, nome cartella checkpoint,
# nome file logit, scelta lr_base) referenzia "args.config" in più punti, ma
# solo "opts" veniva assegnata da parser.parse_args() — NameError garantito,
# indipendentemente da qualunque altro problema di ambiente/container.
# Alias minimo, non tocca nessuno degli usi esistenti di opts.xxx (ninp,
# nhead, kmers, ecc.).
args = opts

# ========================================== #
# 1. MAPPATURA IPERPARAMETRI DELLA TABELLA   #
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

# NOTA: path locale relativo, non assoluto. Anche questo script gira dentro un
# container Singularity con bind SOLO su $SLURM_SUBMIT_DIR:/workspace, quindi
# un path assoluto come /home/simionricc/global_logits non esiste nel
# filesystem del container. È lo script .slurm che, dopo l'uscita dal
# container, sposta il contenuto di questa cartella nella directory condivisa
# globale (vedi run_nt_configs.slurm).
GLOBAL_LOGITS_DIR = "./global_logits_temp"
os.makedirs(GLOBAL_LOGITS_DIR, exist_ok=True)

checkpoints_folder = f'checkpoints_cfg_{args.config}'
os.makedirs(checkpoints_folder, exist_ok=True)
best_model_path = os.path.join(checkpoints_folder, 'best_model.pt')


# ========================================== #
# 2. DATA AUGMENTATION & DATASET ADATTIVO    #
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
            complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N',
                          'a': 't', 't': 'a', 'c': 'g', 'g': 'c', 'n': 'n'}
            if random.random() > 0.5:
                seq = ''.join(complement.get(b, 'N') for b in reversed(seq))
            shift = random.randint(-5, 5)
            seq = seq[max(0, shift):] if shift > 0 else 'N' * (-shift) + seq
            seq = seq[:300]
        elif self.augment_type == 'mutation':
            seq = mutate_string_sequence(seq, n_mutations=40)

        temp_ds = ViraminerDataset([seq], [label])
        return temp_ds[0]


# Caricamento dati
train_df = pd.read_csv(os.path.join("..", "fullset_train.csv"))
val_df = pd.read_csv(os.path.join("..", "fullset_validation.csv"))

train_loader = DataLoader(ParametricViraminerDataset(train_df, augment_type=cfg['augment']),
                           batch_size=cfg['batch_size'], shuffle=True, num_workers=2)
val_loader = DataLoader(ParametricViraminerDataset(val_df, augment_type='none'),
                         batch_size=cfg['batch_size'] * 2, shuffle=False)


# ========================================== #
# 3. LOSS FUNCTIONS & SETUP INIZIALE         #
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
# 4. LOOP DI TRAINING CON SCALER (AMP NATIVO)#
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
# 5. VALUTAZIONE IN CASCATA SUI DATASET      #
# ========================================== #
print("\nInizio estrazione logit in formato TXT...")
model.load_state_dict(torch.load(best_model_path, map_location=device))
model.eval()

test_df = pd.read_csv(os.path.join("..", "fullset_test.csv"))
test_loader = DataLoader(ParametricViraminerDataset(test_df, augment_type='none'),
                          batch_size=cfg['batch_size'] * 2, shuffle=False)

import glob


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
    """
    Autodetect dei due formati osservati in ~/datasets/:
      - ecoli/enhancer: header 'sequence,label', 2 colonne (seq, label)
      - unseen_anell/SIM: nessun header, 3 colonne (id, seq, label)
    Restituisce sempre un DataFrame a 3 colonne posizionali [0,1,2] =
    [id fittizio, sequenza, label], compatibile con ParametricViraminerDataset
    (che usa iloc[:,1]=seq, iloc[:,2]=label per posizione, non per nome colonna).
    """
    ncols, has_header = _peek_columns(path)
    df = pd.read_csv(path) if has_header else pd.read_csv(path, header=None)
    if df.shape[1] == 2:
        seqs = df.iloc[:, 0].astype(str).values
        labels = df.iloc[:, 1].astype(float).values
    else:
        seqs = df.iloc[:, 1].astype(str).values
        labels = df.iloc[:, 2].astype(float).values
    return pd.DataFrame({0: range(len(seqs)), 1: seqs, 2: labels})


# ---------------------------------------------------------------------- #
# CASCATA COMPLETA SUI DATASET, basata sulla cartella condivisa          #
# ~/datasets/ (fuori da virproject/Viral_identification, comune ai 3    #
# metodi). Struttura vista su OneDrive: ecoli/, enhancer/, unseen_anell/,#
# SIM/ (niente deepsea -> aggiungerlo manualmente sotto quando il path   #
# sarà noto).                                                            #
# ---------------------------------------------------------------------- #
BASE_DATASETS_DIR = "/home/simionricc/datasets"


def _find_single_csv(folder, label):
    """Trova il CSV di test in una cartella. Preferisce file con 'test' nel nome
    (per evitare di scegliere per caso il file di training quando ce ne sono
    più di uno, come in enhancer/: bert_enhancer_dataset.csv vs
    bert_enhancer_ind_test_set.csv - alfabeticamente il primo vincerebbe ma è
    il training set)."""
    all_csvs = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if len(all_csvs) == 0:
        print(f"[SKIP] '{label}': nessun CSV trovato in {folder}.")
        return None
    test_csvs = [f for f in all_csvs if 'test' in os.path.basename(f).lower()]
    if test_csvs:
        if len(test_csvs) > 1:
            print(f"[WARN] '{label}': più file con 'test' nel nome in {folder}, uso il primo: "
                  f"{test_csvs[0]} (altri: {test_csvs[1:]}). Verifica manualmente.")
        return test_csvs[0]
    print(f"[WARN] '{label}': nessun file con 'test' nel nome in {folder}, uso il primo CSV "
          f"disponibile ({all_csvs[0]}) - VERIFICA che non sia il training set!")
    return all_csvs[0]


DATASETS_TO_EVAL = {
    'val':     val_loader,
    'test_VM': test_loader,
}


def _loader_from_csv(path):
    df_extra = load_seq_label_dataframe(path)
    return DataLoader(ParametricViraminerDataset(df_extra, augment_type='none'),
                       batch_size=cfg['batch_size'] * 2, shuffle=False)


_ecoli_csv = _find_single_csv(os.path.join(BASE_DATASETS_DIR, "ecoli"), "test_ecoli")
if _ecoli_csv:
    DATASETS_TO_EVAL['test_ecoli'] = _loader_from_csv(_ecoli_csv)

_enhancer_csv = _find_single_csv(os.path.join(BASE_DATASETS_DIR, "enhancer"), "test_enhancer")
if _enhancer_csv:
    DATASETS_TO_EVAL['test_enhancer'] = _loader_from_csv(_enhancer_csv)

# Anellovirus: escludiamo esplicitamente il file di training (typo "birany",
# sospetto duplicato ViraMiner) e usiamo solo il test noto.
_anell_test = os.path.join(BASE_DATASETS_DIR, "unseen_anell", "test_data_anell_300_binary.csv")
if os.path.exists(_anell_test):
    DATASETS_TO_EVAL['test_anellovirus'] = _loader_from_csv(_anell_test)
else:
    print(f"[SKIP] 'test_anellovirus': path non trovato ({_anell_test}). "
          f"Verifica il nome file dopo l'upload sul cluster.")

# DeepSea: nessuna cartella "deepsea" nel listing datasets/ condiviso al momento
# della scrittura di questo script. Aggiungi qui il path reale appena disponibile:
# DATASETS_TO_EVAL['test_deepsea'] = _loader_from_csv(os.path.join(BASE_DATASETS_DIR, "deepsea", "<file>.csv"))

# SIM: 15 test set rumorosi, auto-discovery ricorsiva (stessa logica degli altri script).
_sim_base_dir = os.path.join(BASE_DATASETS_DIR, "SIM")
_sim_files = sorted(glob.glob(os.path.join(_sim_base_dir, "**", "*.csv"), recursive=True))
if not _sim_files:
    print(f"[WARN] Nessun CSV trovato in {_sim_base_dir} (test rumore SIM).")

# FIX: chiave basata sul path relativo a SIM/ (non solo sul basename), per
# evitare che sottocartelle di livello di rumore diverse con file omonimi
# si sovrascrivano a vicenda in DATASETS_TO_EVAL (stesso fix applicato
# negli altri due script).
_seen_sim_keys = {}
for _f in _sim_files:
    _rel_noext = os.path.splitext(os.path.relpath(_f, _sim_base_dir))[0]
    _key = "test_sim_" + _rel_noext.replace(os.sep, "_")
    if _key in _seen_sim_keys:
        print(f"[WARN] Chiave duplicata '{_key}' per i file SIM:\n"
              f"       {_seen_sim_keys[_key]}\n"
              f"       {_f}\n"
              f"       Il primo file verrà sovrascritto: verifica la struttura di {_sim_base_dir}.")
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

        # FIX naming: "nucleictransformerA_..." si leggeva male. Ora il formato
        # è uniforme tra i tre script: {architettura}_cfg{X}_{dataset}.txt
        filename = f"nucleictransformer_cfg{args.config}_{name}.txt"
        np.savetxt(os.path.join(GLOBAL_LOGITS_DIR, filename), np.array(logits_list), fmt='%.6f')
        print(f"Logit salvati temporaneamente: {filename}")

# ========================================== #
# 6. PULIZIA AUTOMATICA DEL CHECKPOINT       #
# ========================================== #
import shutil
if os.path.exists(checkpoints_folder):
    shutil.rmtree(checkpoints_folder)
print("Cartella pesi pesanti rimossa per preservare lo spazio sul cluster.")
