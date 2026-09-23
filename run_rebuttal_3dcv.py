#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from utils.representation_utils import CGR3DProjector, get_shuffled_tetrahedron, DEFAULT_TETRAHEDRON
from utils.attention_rollout import compute_attention_rollout, save_rollout_figure
from models.model_3dcv import Model3DCV

class CGR3DDataset(Dataset):
    def __init__(self, sequences, labels, resolution=128, vertices=None):
        self.sequences = sequences
        self.labels = labels
        self.projector = CGR3DProjector(resolution=resolution, vertices=vertices)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        planes = self.projector.encode(self.sequences[idx])
        return torch.tensor(planes, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)

def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            out = model(x.to(device)).squeeze(-1)
            preds.extend(torch.sigmoid(out).cpu().numpy().flatten())
            targets.extend(y.numpy().flatten())
    return roc_auc_score(targets, preds) * 100.0 if len(np.unique(targets)) > 1 else 50.0

def main():
    parser = argparse.ArgumentParser(description="3DCV Rebuttal Experiments")
    parser.add_argument("--mode", type=str, required=True, choices=["ablation", "shuffle_coords", "rollout"])
    parser.add_argument("--data_file", type=str, default="/home/simionricc/datasets/DNA_data/fullset_test.csv")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--out_dir", type=str, default="./rebuttal_outputs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.data_file, header=None)
    seqs = df[0].values if df.shape[1] == 2 else df[1].values
    labs = df[1].values if df.shape[1] == 2 else df[2].values

    # Hold-out per validazione rapida
    n_tr = int(len(seqs) * 0.8)
    tr_s, tr_y = seqs[:n_tr], labs[:n_tr]
    te_s, te_y = seqs[n_tr:], labs[n_tr:]

    if args.mode == "ablation":
        modes = ['full', 'p1', 'p2', 'p3', 'p1_p2', 'p1_p3', 'p2_p3']
        results = {}
        tr_ds = CGR3DDataset(tr_s, tr_y, resolution=args.resolution)
        te_ds = CGR3DDataset(te_s, te_y, resolution=args.resolution)
        tr_l = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
        te_l = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

        for m in modes:
            print(f"[*] Training modalità proiezione: {m.upper()}")
            model = Model3DCV(mode=m, img_size=args.resolution).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
            crit = nn.BCEWithLogitsLoss()
            for ep in range(args.epochs):
                model.train()
                for x_b, y_b in tr_l:
                    opt.zero_grad()
                    loss = crit(model(x_b.to(device)).squeeze(-1), y_b.to(device))
                    loss.backward()
                    opt.step()
            auc = evaluate(model, te_l, device)
            results[m] = auc
            print(f"  -> {m}: {auc:.2f}% AUROC")

        out_res = os.path.join(args.out_dir, "projection_ablations.txt")
        with open(out_res, "w") as f:
            for k, v in results.items(): f.write(f"{k}: {v:.2f}\n")
        print(f"[✓] Risultati salvati in: {out_res}")

    elif args.mode == "shuffle_coords":
        seeds = [101, 202, 303, 404, 505]
        scores = []
        for s in seeds:
            shuf = get_shuffled_tetrahedron(seed=s)
            tr_ds = CGR3DDataset(tr_s, tr_y, resolution=args.resolution, vertices=shuf)
            te_ds = CGR3DDataset(te_s, te_y, resolution=args.resolution, vertices=shuf)
            tr_l = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
            te_l = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

            model = Model3DCV(mode='full', img_size=args.resolution).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
            crit = nn.BCEWithLogitsLoss()
            for ep in range(args.epochs):
                model.train()
                for x_b, y_b in tr_l:
                    opt.zero_grad()
                    loss = crit(model(x_b.to(device)).squeeze(-1), y_b.to(device))
                    loss.backward()
                    opt.step()
            auc = evaluate(model, te_l, device)
            scores.append(auc)
            print(f"  -> Permutazione (Seed {s}): {auc:.2f}%")

        out_res = os.path.join(args.out_dir, "shuffled_coords.txt")
        with open(out_res, "w") as f:
            f.write(f"Mean: {np.mean(scores):.2f} +/- {np.std(scores):.2f}\n")
            for s, sc in zip(seeds, scores): f.write(f"Seed {s}: {sc:.2f}\n")
        print(f"[✓] Risultati shuffle salvati in: {out_res}")

    elif args.mode == "rollout":
        model = Model3DCV(mode='full', img_size=args.resolution).to(device).eval()
        pos_idx = np.where(np.array(te_y) == 1)[0][0]
        neg_idx = np.where(np.array(te_y) == 0)[0][0]
        proj = CGR3DProjector(resolution=args.resolution)

        for label_name, idx in [("Positive_Viral", pos_idx), ("Negative_Host", neg_idx)]:
            planes = proj.encode(te_s[idx])
            x_t = torch.tensor(planes, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                _, attns = model(x_t, return_attns=True)

            r_dict = {k: compute_attention_rollout(attns[k]) for k in ['p1', 'p2', 'p3']}
            out_img = os.path.join(args.out_dir, f"rollout_{label_name}.png")
            save_rollout_figure(planes, r_dict, label_name, out_img, resolution=args.resolution)
            print(f"[✓] Mappa salvata: {out_img}")

if __name__ == "__main__":
    main()