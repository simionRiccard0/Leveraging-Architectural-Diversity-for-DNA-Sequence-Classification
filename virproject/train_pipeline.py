import os
import math
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

# Parsing degli argomenti da Slurm
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, choices=['A', 'B', 'C'], help="Configurazione di training")
args = parser.parse_args()

# ========================================== #
# 1. MAPPATURA COMPLETA DELLA TABELLA        #
# ========================================== #
CONFIG_MAP = {
    'A': {  # ViraExplorer
        'epochs': 120, 'early_stop': True, 'patience': 15, 'batch_size': 128,
        'loss': 'focal', 'optimizer': 'adamw', 'lr_sched': 'cosine', 'augment': 'rc_shift',
        'cnn_dropout': 0.25, 'trans_dropout': 0.1, 'out_features': 1
    },
    'B': {  # FullE Ensemble
        'epochs': 30, 'early_stop': False, 'patience': 0, 'batch_size': 128,
        'loss': 'bce', 'optimizer': 'adam', 'lr_sched': 'step', 'augment': 'none',
        'cnn_dropout': 0.2, 'trans_dropout': 0.2, 'out_features': 1
    },
    'C': {  # Nucleic Transformer
        'epochs': 60, 'early_stop': False, 'patience': 0, 'batch_size': 128,
        'loss': 'ce', 'optimizer': 'adam', 'lr_sched': 'aiayn', 'augment': 'mutation',
        'cnn_dropout': 0.1, 'trans_dropout': 0.1, 'out_features': 2  # CE classica richiede 2 classi
    }
}
cfg = CONFIG_MAP[args.config]

SAVE_DIR = "/home/simionricc/virproject/"
BEST_MODEL_PATH = os.path.join(SAVE_DIR, f"best_vira_cfg_{args.config}.pth")
GLOBAL_LOGITS_DIR = "/home/simionricc/global_logits"
os.makedirs(GLOBAL_LOGITS_DIR, exist_ok=True)

train_df = pd.read_csv('ViraMiner/data/DNA_data/fullset_train.csv', header=None)
val_df   = pd.read_csv('ViraMiner/data/DNA_data/fullset_validation.csv', header=None)

# ========================================== #
# 2. DATA AUGMENTATION AVANZATA              #
# ========================================== #
mapping = {"A": [1, 0, 0, 0], "C": [0, 1, 0, 0], "G": [0, 0, 1, 0], "T": [0, 0, 0, 1], "N": [0, 0, 0, 0]}
MAX_LEN = 300


def mutate_sequence(seq, n_mutations=40):
    """Implementazione esatta della random nucleotide mutation (Config C)"""
    seq_list = list(seq)
    seq_len = len(seq_list)
    nucleotides = ['A', 'C', 'G', 'T']
    pos_to_mutate = random.sample(range(seq_len), min(n_mutations, seq_len))
    for pos in pos_to_mutate:
        current_nuc = seq_list[pos]
        choices = [n for n in nucleotides if n != current_nuc]
        seq_list[pos] = random.choice(choices)
    return "".join(seq_list)


def encode_sequence(seq, augment_type):
    # FIX CRITICO: mapping e complement hanno solo chiavi maiuscole. Senza
    # questa normalizzazione, sequenze minuscole (es. ecoli_independent_test_NT.csv,
    # interamente in minuscolo) venivano silenziosamente codificate come
    # vettore nullo per OGNI base (mapping.get(b, [0,0,0,0]) non trova mai
    # una corrispondenza) - equivalente a passare al modello un input
    # completamente azzerato, senza alcun errore visibile. Confermato
    # empiricamente: somma totale encoding = 0 su una sequenza reale di
    # ecoli senza questo fix. I logit già salvati in
    # viraexplorer_cfg{A,B,C}_test_ecoli.txt sono stati calcolati PRIMA di
    # questo fix e vanno ricalcolati.
    seq = seq.upper()

    if augment_type == 'rc_shift':  # Config A
        complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
        if random.random() > 0.5:
            seq = ''.join(complement.get(b, 'N') for b in reversed(seq))
        shift = random.randint(-5, 5)
        seq = seq[max(0, shift):] if shift > 0 else 'N' * (-shift) + seq
    elif augment_type == 'mutation':  # Config C
        seq = mutate_sequence(seq, n_mutations=40)

    seq = seq[:MAX_LEN]
    encoded = [mapping.get(b, [0, 0, 0, 0]) for b in seq]
    while len(encoded) < MAX_LEN:
        encoded.append([0, 0, 0, 0])
    return encoded


class DNADataset(Dataset):
    def __init__(self, dataframe, augment_type='none'):
        self.sequences = dataframe[1].values
        self.labels = dataframe[2].values
        self.augment_type = augment_type

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x = torch.tensor(encode_sequence(self.sequences[idx], self.augment_type), dtype=torch.float32)
        if cfg['loss'] == 'ce':
            y = torch.tensor(self.labels[idx], dtype=torch.long)
        else:
            y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y


train_loader = DataLoader(DNADataset(train_df, augment_type=cfg['augment']), batch_size=cfg['batch_size'],
                           shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(DNADataset(val_df, augment_type='none'), batch_size=cfg['batch_size'], shuffle=False,
                         num_workers=2, pin_memory=True)


# ========================================== #
# 3. MODELLO VIRAEXPLORER DINAMICO           #
# ========================================== #
class PatternBranch(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.conv6, self.conv12 = nn.Conv1d(4, 512, 6, padding='same'), nn.Conv1d(4, 512, 12, padding='same')
        self.conv18, self.conv24 = nn.Conv1d(4, 512, 18, padding='same'), nn.Conv1d(4, 512, 24, padding='same')
        self.bn, self.pool = nn.BatchNorm1d(2048), nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(nn.Linear(2048, 512), nn.GELU(), nn.Dropout(dropout))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = torch.cat([F.gelu(self.conv6(x)), F.gelu(self.conv12(x)), F.gelu(self.conv18(x)), F.gelu(self.conv24(x))],
                      dim=1)
        # FIX: il ramo "if" era irraggiungibile (pool(x) ha sempre 3 dim -> [B, C, 1]),
        # quindi la BatchNorm non veniva mai applicata. Ora si applica sempre dopo il pooling.
        x = self.pool(x).squeeze(-1)  # [B, 2048]
        x = self.bn(x)
        return self.fc(x)


class FrequencyBranch(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.conv6, self.conv12 = nn.Conv1d(4, 256, 6, padding='same'), nn.Conv1d(4, 256, 12, padding='same')
        self.conv18, self.conv24 = nn.Conv1d(4, 256, 18, padding='same'), nn.Conv1d(4, 256, 24, padding='same')
        self.bn, self.pool = nn.BatchNorm1d(1024), nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(nn.Linear(1024, 512), nn.GELU(), nn.Dropout(dropout))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = torch.cat([F.gelu(self.conv6(x)), F.gelu(self.conv12(x)), F.gelu(self.conv18(x)), F.gelu(self.conv24(x))],
                      dim=1)
        # Stessa correzione applicata a PatternBranch
        x = self.pool(x).squeeze(-1)  # [B, 1024]
        x = self.bn(x)
        return self.fc(x)


class TransformerBranch(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=4, max_len=300, dropout=0.1):
        super().__init__()
        self.embedding, self.pos_enc = nn.Linear(4, d_model), nn.Embedding(max_len + 1, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
                                                     dropout=dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm, self.out = nn.LayerNorm(d_model), nn.Linear(d_model, 512)

    def forward(self, x):
        B, L, _ = x.shape
        x = self.embedding(x) + self.pos_enc(torch.arange(1, L + 1, device=x.device).unsqueeze(0))
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        return self.out(self.norm(self.transformer(x))[:, 0, :])


class ViraExplorer(nn.Module):
    def __init__(self, cnn_drop, trans_drop, out_features):
        super().__init__()
        self.pattern = PatternBranch(cnn_drop)
        self.frequency = FrequencyBranch(cnn_drop)
        self.transformer = TransformerBranch(dropout=trans_drop)
        self.fc = nn.Sequential(
            nn.Linear(1536, 512), nn.GELU(), nn.Dropout(cnn_drop), nn.LayerNorm(512),
            nn.Linear(512, 128), nn.GELU(), nn.Dropout(cnn_drop),
            nn.Linear(128, out_features)
        )

    def forward(self, x):
        return self.fc(torch.cat([self.pattern(x), self.frequency(x), self.transformer(x)], dim=1))


# ========================================== #
# 4. CONFIGURAZIONE LOSS ED OTTIMIZZATORI    #
# ========================================== #
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, smoothing=0.05):
        super().__init__()
        self.alpha, self.gamma, self.smoothing = alpha, gamma, smoothing

    def forward(self, logits, targets):
        targets_smooth = targets.unsqueeze(1) * (1 - self.smoothing) + 0.5 * self.smoothing
        bce = F.binary_cross_entropy_with_logits(logits, targets_smooth, reduction='none')
        return (self.alpha * (1 - torch.exp(-bce)) ** self.gamma * bce).mean()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViraExplorer(cnn_drop=cfg['cnn_dropout'], trans_drop=cfg['trans_dropout'],
                      out_features=cfg['out_features']).to(device)

if cfg['loss'] == 'focal':
    criterion = FocalLoss()
elif cfg['loss'] == 'bce':
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.CrossEntropyLoss()

if cfg['optimizer'] == 'adamw':
    transformer_params = list(model.transformer.parameters())
    other_params = list(model.pattern.parameters()) + list(model.frequency.parameters()) + list(
        model.fc.parameters())
    optimizer = torch.optim.AdamW([{"params": other_params, "lr": 1e-4}, {"params": transformer_params, "lr": 5e-5}],
                                   weight_decay=1e-4)
else:
    lr_base = 0.001 if args.config == 'B' else 1.0
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_base)

if cfg['lr_sched'] == 'cosine':
    def lr_lambda(epoch):
        if epoch < 5:
            return epoch / 5
        return 0.5 * (1 + math.cos(math.pi * (epoch - 5) / (cfg['epochs'] - 5)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
elif cfg['lr_sched'] == 'step':
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
else:
    d_model = 256
    warmup_steps = 4000
    def aiayn_lambda(current_step):
        step = max(1, current_step)
        return (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, aiayn_lambda)

# ========================================== #
# 5. TRAINING LOOP ADATTIVO                  #
# ========================================== #
best_auc = 0
counter = 0
scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

if args.config == 'A':
    for p in model.transformer.parameters():
        p.requires_grad = False

for epoch in range(cfg['epochs']):
    if args.config == 'A' and epoch == 5:
        for p in model.transformer.parameters():
            p.requires_grad = True
        print("Transformer sbloccato dopo il warmup.")

    model.train()
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                loss = criterion(model(X_batch), y_batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()

        if cfg['lr_sched'] == 'aiayn':
            scheduler.step()

    if cfg['lr_sched'] != 'aiayn':
        scheduler.step()

    model.eval()
    preds, targets_list = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            out = model(X_batch.to(device))
            if cfg['loss'] == 'ce':
                probs = F.softmax(out, dim=1)[:, 1]
            else:
                probs = torch.sigmoid(out)
            preds.extend(probs.cpu().numpy().flatten())
            targets_list.extend(y_batch.cpu().numpy().flatten())

    auc = roc_auc_score(targets_list, preds)
    print(f"Epoca {epoch + 1}/{cfg['epochs']} | Val AUC: {auc:.4f}")

    if auc > best_auc:
        best_auc = auc
        counter = 0
        torch.save(model.state_dict(), BEST_MODEL_PATH)
    else:
        counter += 1

    if cfg['early_stop'] and counter >= cfg['patience']:
        print(f"Early stopping intervenuto all'epoca {epoch + 1}")
        break

if not cfg['early_stop']:
    torch.save(model.state_dict(), BEST_MODEL_PATH)

# ========================================== #
# 6. VALUTAZIONE IN CASCATA E SALVATAGGIO    #
# ========================================== #
print("\nInizio estrazione logit in formato TXT...")
model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
model.eval()

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
      - unseen_anell/SIM: nessun header, 3 colonne (id, seq, label) -> stesso
        formato di fullset_train.csv
    Restituisce sempre un DataFrame a 3 colonne posizionali [0,1,2] =
    [id fittizio, sequenza, label], compatibile con DNADataset (che si
    aspetta dataframe[1]=seq, dataframe[2]=label).
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
# ~/datasets/ (fuori da virproject, comune ai 3 metodi). Struttura vista #
# su OneDrive: ecoli/, enhancer/, unseen_anell/, SIM/ (niente deepsea -> #
# aggiungerlo manualmente sotto quando il path sarà noto).               #
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
    'val':     'ViraMiner/data/DNA_data/fullset_validation.csv',
    'test_VM': 'ViraMiner/data/DNA_data/fullset_test.csv',
}

_ecoli_csv = _find_single_csv(os.path.join(BASE_DATASETS_DIR, "ecoli"), "test_ecoli")
if _ecoli_csv:
    DATASETS_TO_EVAL['test_ecoli'] = _ecoli_csv

_enhancer_csv = _find_single_csv(os.path.join(BASE_DATASETS_DIR, "enhancer"), "test_enhancer")
if _enhancer_csv:
    DATASETS_TO_EVAL['test_enhancer'] = _enhancer_csv

# Anellovirus: la cartella unseen_anell contiene sia train che test.
# Usiamo esplicitamente il file di test noto (visto in dataset/ sul cluster):
# training_data_anell_300_birany.csv è escluso di proposito (typo "birany" =
# probabile duplicato del file originale ViraMiner, da NON usare come test).
_anell_test = os.path.join(BASE_DATASETS_DIR, "unseen_anell", "test_data_anell_300_binary.csv")
if os.path.exists(_anell_test):
    DATASETS_TO_EVAL['test_anellovirus'] = _anell_test
else:
    print(f"[SKIP] 'test_anellovirus': path non trovato ({_anell_test}). "
          f"Verifica il nome file dopo l'upload sul cluster.")

# DeepSea: nessuna cartella "deepsea" nel listing datasets/ condiviso al momento
# della scrittura di questo script. Aggiungi qui il path reale appena disponibile:
# DATASETS_TO_EVAL['test_deepsea'] = os.path.join(BASE_DATASETS_DIR, "deepsea", "<file>.csv")

# SIM: 15 test set rumorosi (5 per livello di rumore), valutati con il modello
# allenato su dati puliti. Non conosciamo i nomi/sottocartelle esatti -> li
# scopriamo automaticamente e ogni file produce una riga di logit a parte,
# nominata in base al proprio filename.
_sim_files = sorted(glob.glob(os.path.join(BASE_DATASETS_DIR, "SIM", "**", "*.csv"), recursive=True))
if not _sim_files:
    print(f"[WARN] Nessun CSV trovato in {os.path.join(BASE_DATASETS_DIR, 'SIM')} (test rumore SIM).")
for _f in _sim_files:
    _key = f"test_sim_{os.path.splitext(os.path.basename(_f))[0]}"
    DATASETS_TO_EVAL[_key] = _f

with torch.no_grad():
    for name, path in DATASETS_TO_EVAL.items():
        if not os.path.exists(path):
            print(f"[SKIP] '{name}': path non trovato ({path}). Aggiorna DATASETS_TO_EVAL.")
            continue

        # val/test_VM sono già nel formato nativo (id,seq,label, no header);
        # tutti gli altri dataset passano dal normalizzatore di formato.
        if name in ('val', 'test_VM'):
            df = pd.read_csv(path, header=None)
        else:
            df = load_seq_label_dataframe(path)
        loader = DataLoader(DNADataset(df, augment_type='none'), batch_size=64, shuffle=False, num_workers=2)

        logits_list = []
        for X, _ in loader:
            out = model(X.to(device))
            if cfg['loss'] == 'ce':
                logits_equivalent = out[:, 1] - out[:, 0]
                logits_list.extend(logits_equivalent.cpu().numpy().flatten())
            else:
                logits_list.extend(out.cpu().numpy().flatten())

        filename = f"viraexplorer{args.config}_{name}.txt"
        np.savetxt(os.path.join(GLOBAL_LOGITS_DIR, filename), np.array(logits_list), fmt='%.6f')
        print(f"Logit salvati in: {os.path.join(GLOBAL_LOGITS_DIR, filename)}")

# ========================================== #
# 7. PULIZIA AUTOMATICA DEL PESO             #
# ========================================== #
if os.path.exists(BEST_MODEL_PATH):
    os.remove(BEST_MODEL_PATH)
print("File dei pesi pesanti (.pth) rimosso per preservare lo spazio su disco.")
