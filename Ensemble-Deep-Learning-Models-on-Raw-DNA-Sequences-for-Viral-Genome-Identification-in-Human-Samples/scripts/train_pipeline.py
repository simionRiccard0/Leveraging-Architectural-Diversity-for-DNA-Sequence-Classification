from pathlib import Path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
import torch
import torch.nn as nn
import json
import copy
import argparse
import math
import numpy as np
from src.dataset.csv_dataset import CSVDataset
from src.utils.metrics import auroc as score_fn
from src.utils.create_model import create_model
from src.utils.train_model import train_model

# NOTA: rimosso `from sklearn.model_selection import ParameterGrid`, importato
# nello script originale ma mai usato in main(). Se l'intento era generare
# gli originali "10 nets" dell'ensemble FullE variando LR/dropout sulla
# griglia della tabella (LR in {0.1,0.01,0.001}, dropout in {0.1,0.2,0.5}),
# quella logica va reintrodotta esplicitamente: qui, come negli altri due
# script, si allena UN modello per config (A/B/C), non un intero ensemble
# di 10 reti. Verificare con chi ha scritto lo script originale quale fosse
# l'intento prima di lanciare gli esperimenti.


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, choices=['A', 'B', 'C'],
                         help="Configurazione di training dalla tabella")
    parser.add_argument("--data_folder", "-d", default="dataset", help="Path to data folder containing csv files")
    parser.add_argument("--path", "-p", default="ready_to_train_files/final/onehot/branch/frequency/init+norm.json",
                         help="Path to base JSON config")
    return parser.parse_args()


# ========================================== #
# SEED PER RIPRODUCIBILITA'                  #
# ========================================== #
# FIX: nello script originale solo Nucleic Transformer seedava random/np/torch.
# Senza questo, mutation e reverse-complement+shift (applicati sui tensori
# one-hot qui sotto) non erano riproducibili tra run diverse con la stessa config.
def seed_everything(seed=2020):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ========================================== #
# DATA AUGMENTATION ON ONE-HOT TENSORS       #
# ========================================== #
def apply_tensor_augmentation(X, augment_type):
    """Applica le mutazioni e il reverse complement direttamente sui vettori One-Hot"""
    if augment_type == 'none':
        return X

    X = X.clone()
    B = X.shape[0]

    # FIX: confermato via debug print che CSVDataset produce un encoding a
    # 5 canali (A,C,G,T,N) in formato channel-first [B, 5, 300], non 4 canali
    # come assunto in origine (mutuato da ViraExplorer, che codifica N come
    # vettore nullo su 4 canali invece di un canale dedicato). Il vecchio
    # controllo `if X.shape[1] == 4` era quindi sempre falso qui, cadendo nel
    # ramo che assumeva channel-last [B,300,4] - causa del RuntimeError di
    # shape mismatch. Ora gestiamo esplicitamente il caso a 5 canali:
    # la permutazione di reverse-complement (A<->T, C<->G) diventa
    # [3,2,1,0,4] per lasciare il canale N (indice 4) invariato.
    n_channels = X.shape[1] if X.shape[1] in (4, 5) else X.shape[2]
    channel_first = X.shape[1] in (4, 5)
    rc_perm = [3, 2, 1, 0, 4] if n_channels == 5 else [3, 2, 1, 0]

    if channel_first:  # Shape: [B, C, 300], C in {4,5}
        L = X.shape[2]
        if augment_type == 'rc_shift':
            for i in range(B):
                if np.random.rand() > 0.5:
                    X[i] = torch.flip(X[i], dims=[1])
                    X[i] = X[i][rc_perm, :]
                shift = np.random.randint(-5, 6)
                if shift != 0:
                    X[i] = torch.roll(X[i], shifts=shift, dims=1)
        elif augment_type == 'mutation':
            for i in range(B):
                pos = np.random.choice(L, min(40, L), replace=False)
                for p in pos:
                    current_channel = torch.argmax(X[i, :, p]).item()
                    # Muta solo tra i 4 canali di base (A,C,G,T); se il canale
                    # corrente è N (indice 4, quando presente) lo lasciamo
                    # invariato invece di mutarlo verso una base concreta.
                    if current_channel >= 4:
                        continue
                    new_channel = np.random.choice([c for c in range(4) if c != current_channel])
                    X[i, :, p] = 0.0
                    X[i, new_channel, p] = 1.0

    else:  # Shape: [B, 300, C], C in {4,5}
        L = X.shape[1]
        if augment_type == 'rc_shift':
            for i in range(B):
                if np.random.rand() > 0.5:
                    X[i] = torch.flip(X[i], dims=[0])
                    X[i] = X[i][:, rc_perm]
                shift = np.random.randint(-5, 6)
                if shift != 0:
                    X[i] = torch.roll(X[i], shifts=shift, dims=0)
        elif augment_type == 'mutation':
            for i in range(B):
                pos = np.random.choice(L, min(40, L), replace=False)
                for p in pos:
                    current_channel = torch.argmax(X[i, p, :]).item()
                    if current_channel >= 4:
                        continue
                    new_channel = np.random.choice([c for c in range(4) if c != current_channel])
                    X[i, p, :] = 0.0
                    X[i, p, new_channel] = 1.0
    return X


class AugmentedDataLoader:
    def __init__(self, dataloader, augment_type):
        self.dataloader = dataloader
        self.augment_type = augment_type

    def __len__(self):
        return len(self.dataloader)

    @property
    def dataset(self):
        # Passthrough esplicito (oltre al __getattr__ generico sotto) per
        # chiarezza: train_model() (libreria FullE di Nanni) fa internamente
        # len(dataloader.dataset).
        return self.dataloader.dataset

    def __getattr__(self, name):
        # FIX: train_model() sonda più attributi di DataLoader di quanto
        # inizialmente previsto - prima .dataset, poi anche .batch_size
        # (AttributeError: 'AugmentedDataLoader' object has no attribute
        # 'batch_size'). Invece di aggiungere una property per ogni nuovo
        # attributo scoperto uno alla volta, questo __getattr__ generico
        # inoltra qualunque attributo non definito esplicitamente qui al
        # DataLoader sottostante. __getattr__ (a differenza di
        # __getattribute__) scatta SOLO se l'attributo non è già stato
        # trovato normalmente, quindi non interferisce con __init__,
        # __len__, __iter__, dataset già definiti sopra.
        return getattr(self.dataloader, name)

    def __iter__(self):
        for batch in self.dataloader:
            if isinstance(batch, dict) and 'data' in batch:
                batch['data'] = apply_tensor_augmentation(batch['data'], self.augment_type)
                yield batch
            elif isinstance(batch, (tuple, list)):
                X_aug = apply_tensor_augmentation(batch[0], self.augment_type)
                yield (X_aug, batch[1])
            else:
                yield batch


# ========================================== #
# CUSTOM FOCAL LOSS PER OUTPUT PROBABILISTICI#
# ========================================== #
class ProbabilityFocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, smoothing=0.05):
        super().__init__()
        self.alpha, self.gamma, self.smoothing = alpha, gamma, smoothing

    def forward(self, probs, targets):
        targets = targets.view_as(probs)
        targets_smooth = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        probs = torch.clamp(probs, 1e-6, 1.0 - 1e-6)
        loss = -self.alpha * ((1 - probs) ** self.gamma) * targets_smooth * torch.log(probs) \
               - (1 - self.alpha) * (probs ** self.gamma) * (1 - targets_smooth) * torch.log(1 - probs)
        return loss.mean()


class SafeBCELoss(nn.Module):
    """FIX PREVENTIVO: nn.BCELoss() diretto non allinea la shape di target e
    output, causando lo stesso ValueError di shape mismatch visto in
    ViraExplorer-B (nn.BCEWithLogitsLoss() su [batch,1] vs target [batch]).
    ProbabilityFocalLoss lo evita già con targets.view_as(probs); questo
    wrapper applica la stessa protezione anche per le config B/C, che usano bce."""
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, probs, targets):
        return self.bce(probs, targets.view_as(probs).float())


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


def _normalize_to_temp_csv(path):
    """
    Autodetect dei due formati osservati in ~/datasets/:
      - ecoli/enhancer: header 'sequence,label', 2 colonne (seq, label)
      - unseen_anell/SIM: nessun header, 3 colonne (id, seq, label)
    Scrive un CSV temporaneo nello STESSO formato assunto per fullset_train.csv
    (nessun header, 3 colonne: id, seq, label) prima di passarlo a CSVDataset,
    che legge direttamente da file e non accetta un DataFrame in memoria.
    NOTA: questa normalizzazione assume che fullset_train.csv non abbia header;
    se make_eval_loader dovesse fallire su un dataset normalizzato, verifica
    con `head -2 dataset/fullset_train.csv` che l'assunzione sia corretta.
    """
    import pandas as pd
    ncols, has_header = _peek_columns(path)
    df = pd.read_csv(path) if has_header else pd.read_csv(path, header=None)
    if df.shape[1] == 2:
        seqs = df.iloc[:, 0].astype(str).values
        labels = df.iloc[:, 1].astype(float).values
    else:
        seqs = df.iloc[:, 1].astype(str).values
        labels = df.iloc[:, 2].astype(float).values

    # FIX: CSVDataset.onehot_encoding usa una mappa di soli caratteri
    # maiuscoli (A,C,G,T,N) - confermato da KeyError: 't' su ecoli, il cui
    # file sorgente ha sequenze interamente minuscole. Senza questa
    # normalizzazione il training/eval crasha su ecoli. NOTA: lo stesso
    # problema esiste in ViraExplorer (mapping = {"A":...,...} solo
    # maiuscole), ma lì NON crasha: mapping.get(b, [0,0,0,0]) restituisce
    # silenziosamente un vettore nullo per ogni base minuscola non trovata,
    # equivalente a trattare l'intera sequenza come "N" (ambigua). I file
    # viraexplorer_cfg*_test_ecoli.txt già calcolati potrebbero quindi
    # essere basati su input di fatto azzerato, non su sequenze reali - da
    # verificare e ricalcolare separatamente, indipendentemente da questo fix.
    seqs = np.array([s.upper() for s in seqs])

    import tempfile
    tmp_path = os.path.join(tempfile.gettempdir(), f"norm_{os.path.basename(path)}")
    pd.DataFrame({0: range(len(seqs)), 1: seqs, 2: labels}).to_csv(tmp_path, header=False, index=False)
    return tmp_path


def make_eval_loader(csv_path, normalize=False):
    """Helper per costruire un DataLoader di valutazione con lo stesso onehot encoding del training/val/test."""
    if normalize:
        csv_path = _normalize_to_temp_csv(csv_path)
    ds = CSVDataset(csv_path, transform="onehot_encoding")
    return ds.get_dataloader(batch_size=128, shuffle=False)


def main():
    args = parse_args()
    seed_everything(2020)

    # Mapping esatto iperparametri FullE dalla tabella.
    # NOTA: per la config C la tabella indica Cross-Entropy, ma il modello FullE
    # (via create_model) restituisce un'unica probabilità sigmoidale a 1 output
    # (si veda l'inversione logit = log(p/(1-p)) più sotto), quindi non è
    # compatibile con una CE a 2 classi. Si usa BCE come equivalente binario
    # obbligato dal vincolo architetturale: è una deviazione intenzionale dalla
    # tabella "uniformata", da annotare esplicitamente nel paper/metodologia.
    CONFIG_MAP = {
        'A': {'epochs': 120, 'dropout': 0.25, 'loss': 'focal', 'optimizer': 'adamw', 'lr_sched': 'cosine', 'augment': 'rc_shift'},
        'B': {'epochs': 30,  'dropout': 0.2,  'loss': 'bce',   'optimizer': 'adam',  'lr_sched': 'step',   'augment': 'none'},
        'C': {'epochs': 60,  'dropout': 0.1,  'loss': 'bce',   'optimizer': 'adam',  'lr_sched': 'aiayn',  'augment': 'mutation'}
    }
    cfg = CONFIG_MAP[args.config]

    LOCAL_LOGITS_DIR = "./global_logits_temp"
    os.makedirs(LOCAL_LOGITS_DIR, exist_ok=True)

    model_infos = json.loads(Path(args.path).read_text())
    if isinstance(model_infos, list):
        model_infos = model_infos[0]

    model_infos['hyperparameters']['epochs'] = [cfg['epochs']]
    model_infos['hyperparameters']['batch_size'] = [128]
    if 'dropout' in model_infos['parameters']:
        model_infos['parameters']['dropout'] = [cfg['dropout']]

    train_file = os.path.join(args.data_folder, "fullset_train.csv")
    valid_file = os.path.join(args.data_folder, "fullset_validation.csv")
    test_file = os.path.join(args.data_folder, "fullset_test.csv")

    train_df = CSVDataset(f"{train_file}", transform="onehot_encoding")
    valid_df = CSVDataset(f"{valid_file}", transform="onehot_encoding")
    test_df = CSVDataset(f"{test_file}", transform="onehot_encoding")

    dl_train = AugmentedDataLoader(train_df.get_dataloader(batch_size=128, shuffle=True), cfg['augment'])
    dl_valid = valid_df.get_dataloader(batch_size=128, shuffle=False)
    dl_test = test_df.get_dataloader(batch_size=128, shuffle=False)

    model = create_model(model_infos)

    if cfg['loss'] == 'focal':
        loss_fn = ProbabilityFocalLoss()
    else:
        loss_fn = SafeBCELoss()

    if cfg['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    else:
        # FIX: prima qui c'era sempre lr=0.001 fisso, anche per la config C
        # (AIAYN). ViraExplorer e Nucleic Transformer usano invece lr_base=1.0
        # per la config C come base moltiplicata dalla lambda dello scheduler
        # (pattern "Noam" di Attention Is All You Need) e lr_base=0.001 solo
        # per la config B. Con lr fisso a 0.001, il LR di picco per FullE-C
        # risultava ~1000x più piccolo di quello usato da VE-C/NT-C, rendendo
        # il confronto tra le tre architetture sulla stessa "colonna" non valido.
        lr_base = 0.001 if args.config == 'B' else 1.0
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_base)

    if cfg['lr_sched'] == 'cosine':
        # FIX: la versione precedente usava CosineAnnealingLR "nudo" (nessun
        # warmup), mentre ViraExplorer e Nucleic Transformer usano un warmup
        # lineare di 5 epoche seguito da cosine annealing per la stessa
        # config A. Stesso nome ("cosine") ma curva di LR diversa tra le tre
        # architetture rompeva l'uniformità richiesta. Ora è allineato.
        def lr_lambda(epoch):
            if epoch < 5:
                return epoch / 5
            return 0.5 * (1 + math.cos(math.pi * (epoch - 5) / (cfg['epochs'] - 5)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif cfg['lr_sched'] == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    else:
        steps_per_epoch = max(1, len(train_df) // 128)
        def aiayn_epoch_lr(epoch):
            step = max(1, epoch * steps_per_epoch)
            return (512 ** -0.5) * min(step ** -0.5, step * (4000 ** -1.5))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, aiayn_epoch_lr)

    print(f"[i] Avvio Training FullE - Configurazione: {args.config} (Epoche: {cfg['epochs']})")
    # ATTENZIONE (da verificare con Daniel/Nanni): train_model non espone qui un
    # salvataggio/reload esplicito del checkpoint migliore, a differenza degli
    # altri due script dove il reload del best model prima dell'estrazione
    # logit è esplicito. Se train_model non gestisce internamente il best-model
    # selection, il modello usato in fase di eval sotto potrebbe essere quello
    # dell'ultima epoca e non quello a AUROC massimo — stesso tipo di rischio
    # silenzioso già riscontrato con get_best_weights_from_fold() in NT.
    train_model(model, train_dl=dl_train, loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler,
                epochs=cfg['epochs'], epsilon=model_infos['epsilon'], min_score=0,
                validation_fn=score_fn, val_dl=dl_valid, verbose=2)

    # ========================================== #
    # EXTRAZIONE E SALVATAGGIO LOGIT (FORMATO TXT)#
    # ========================================== #
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # ---------------------------------------------------------------------- #
    # CASCATA COMPLETA SUI DATASET, basata sulla cartella condivisa          #
    # ~/datasets/ (fuori dal repo FullE, comune ai 3 metodi). Struttura      #
    # vista su OneDrive: ecoli/, enhancer/, unseen_anell/, SIM/ (niente      #
    # deepsea -> aggiungerlo manualmente sotto quando il path sarà noto).    #
    # ---------------------------------------------------------------------- #
    import glob
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

    datasets_eval = {'val': dl_valid, 'test_VM': dl_test}

    _ecoli_csv = _find_single_csv(os.path.join(BASE_DATASETS_DIR, "ecoli"), "test_ecoli")
    if _ecoli_csv:
        datasets_eval['test_ecoli'] = make_eval_loader(_ecoli_csv, normalize=True)

    _enhancer_csv = _find_single_csv(os.path.join(BASE_DATASETS_DIR, "enhancer"), "test_enhancer")
    if _enhancer_csv:
        datasets_eval['test_enhancer'] = make_eval_loader(_enhancer_csv, normalize=True)

    # Anellovirus: escludiamo esplicitamente il file di training (typo "birany",
    # sospetto duplicato ViraMiner) e usiamo solo il test noto. Formato già
    # nativo (id,seq,label, no header) ma passiamo comunque da normalize=True
    # per sicurezza: costa pochissimo ed evita sorprese se il formato cambia.
    _anell_test = os.path.join(BASE_DATASETS_DIR, "unseen_anell", "test_data_anell_300_binary.csv")
    if os.path.exists(_anell_test):
        datasets_eval['test_anellovirus'] = make_eval_loader(_anell_test, normalize=True)
    else:
        print(f"[SKIP] 'test_anellovirus': path non trovato ({_anell_test}). "
              f"Verifica il nome file dopo l'upload sul cluster.")

    # DeepSea: nessuna cartella "deepsea" nel listing datasets/ condiviso al momento
    # della scrittura di questo script. Aggiungi qui il path reale appena disponibile:
    # datasets_eval['test_deepsea'] = make_eval_loader(os.path.join(BASE_DATASETS_DIR, "deepsea", "<file>.csv"), normalize=True)

    # SIM: 15 test set rumorosi, auto-discovery ricorsiva (stessa logica degli altri script).
    _sim_base_dir = os.path.join(BASE_DATASETS_DIR, "SIM")
    _sim_files = sorted(glob.glob(os.path.join(_sim_base_dir, "**", "*.csv"), recursive=True))
    if not _sim_files:
        print(f"[WARN] Nessun CSV trovato in {_sim_base_dir} (test rumore SIM).")

    # FIX: chiave basata sul path relativo a SIM/ (non solo sul basename), per
    # evitare che sottocartelle di livello di rumore diverse con file
    # omonimi si sovrascrivano a vicenda in datasets_eval (stesso fix
    # applicato negli altri due script).
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
        datasets_eval[_key] = make_eval_loader(_f, normalize=True)

    print("\n[i] Estrazione logit in corso...")
    with torch.no_grad():
        for name, dl in datasets_eval.items():
            logits_list = []
            for batch in dl:
                X = batch['data'].to(device) if isinstance(batch, dict) else batch[0].to(device)
                raw_out = model(X)
                # FIX: model(X) restituisce una tupla (confermato via debug:
                # 2 elementi, entrambi shape [B,1]). train_model() gestisce
                # già lo spacchettamento internamente durante il training;
                # qui nel ciclo manuale va fatto esplicitamente. Elemento 0
                # confermato essere l'output principale (probabilità).
                model_out = raw_out[0] if isinstance(raw_out, (tuple, list)) else raw_out
                probs = model_out.cpu().numpy().flatten()

                probs = np.clip(probs, 1e-6, 1.0 - 1e-6)
                logits = np.log(probs / (1.0 - probs))
                logits_list.extend(logits)

            # FIX naming: "fulleA_..." si leggeva male. Ora il formato è
            # uniforme tra i tre script: {architettura}_cfg{X}_{dataset}.txt
            filename = f"fulle_cfg{args.config}_{name}.txt"
            np.savetxt(os.path.join(LOCAL_LOGITS_DIR, filename), np.array(logits_list), fmt='%.6f')
            print(f" -> File salvato correttamente in formato TXT piatto: {filename}")


if __name__ == "__main__":
    main()
