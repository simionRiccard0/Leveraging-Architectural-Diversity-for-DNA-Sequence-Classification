import os
import re
import shutil
import pandas as pd
import numpy as np

NFOLDS = 5
TOP = 1  # coerente col default usato da evaluate.py / test_ind.py (foldXtop1.ckpt)

os.makedirs('best_weights', exist_ok=True)

for fold in range(NFOLDS):
    csv_file = f'log_fold{fold}.csv'
    ckpt_dir = f'checkpoints_fold{fold}'

    if not os.path.isdir(ckpt_dir):
        print(f"[fold {fold}] SALTATO: {ckpt_dir} non esiste (gia' ripulita o mai creata)")
        continue

    history = pd.read_csv(csv_file)

    # se il file contiene piu' run accodati (stesso fold rilanciato piu' volte),
    # tieni solo l'ULTIMA occorrenza di ciascun epoch: e' quella del run piu' recente,
    # coerente con i checkpoint effettivamente presenti in checkpoints_fold{fold}/ ora.
    n_before = len(history)
    history = history.drop_duplicates(subset='epoch', keep='last').reset_index(drop=True)
    if len(history) != n_before:
        print(f"[fold {fold}] log conteneva {n_before} righe (piu' run accodati), "
              f"ridotte a {len(history)} tenendo solo l'ultima occorrenza di ogni epoch")

    # epoch effettivamente salvati su disco (non quelli teorici del log)
    saved_epochs = set()
    for fname in os.listdir(ckpt_dir):
        m = re.match(r'epoch(\d+)\.ckpt$', fname)
        if m:
            saved_epochs.add(int(m.group(1)))

    if not saved_epochs:
        print(f"[fold {fold}] ERRORE: nessun checkpoint trovato in {ckpt_dir}, salto")
        continue

    # tra i soli epoch salvati, prendi quelli col miglior val_acc
    mask = history.epoch.isin(saved_epochs)
    available_history = history[mask].reset_index(drop=True)
    if available_history.empty:
        print(f"[fold {fold}] ERRORE: nessuna riga di log corrisponde ai checkpoint salvati, salto")
        continue

    scores = np.asarray(available_history.val_acc)
    top_idx = scores.argsort()[-TOP:][::-1]

    copied_ok = []
    for i, idx in enumerate(top_idx):
        best_epoch = available_history.epoch[idx]
        src = os.path.join(ckpt_dir, f'epoch{best_epoch}.ckpt')
        dst = os.path.join('best_weights', f'fold{fold}top{i+1}.ckpt')
        shutil.copy2(src, dst)
        ok = os.path.isfile(dst) and os.path.getsize(dst) > 0
        copied_ok.append(ok)
        print(f"[fold {fold}] epoch scelto = {best_epoch} (val_acc={scores[idx]:.4f}) -> {dst} "
              f"({'OK' if ok else 'FALLITO'})")

    if all(copied_ok):
        shutil.rmtree(ckpt_dir)
        print(f"[fold {fold}] copia verificata, {ckpt_dir} rimossa")
    else:
        print(f"[fold {fold}] ATTENZIONE: copia non verificata, {ckpt_dir} NON rimossa (per sicurezza)")

print("\nFatto. Contenuto di best_weights/:")
print(sorted(os.listdir('best_weights')))
