#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, matthews_corrcoef
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from utils.representation_utils import HelixPREncoder, FCGR2DEncoder
from models.helix_cnn import HelixCNN
from models.capsnet_2d import CapsEnhancerModel

class EncodingDataset(Dataset):
    def __init__(self, sequences, labels, enc_type="helixpr", target_len=300, padding_mode="zero"):
        self.sequences = sequences
        self.labels = labels
        self.enc_type = enc_type
        self.target_len = target_len
        self.padding_mode = padding_mode

        if enc_type == "helixpr":
            self.encoder = HelixPREncoder(T=10.0, alpha=0.05)
        elif enc_type == "fcgr":
            self.encoder = FCGR2DEncoder(resolution=64)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        if self.enc_type == "helixpr":
            arr = self.encoder.encode(seq, target_len=self.target_len, padding_mode=self.padding_mode)
        elif self.enc_type == "fcgr":
            arr = self.encoder.encode(seq)
        return torch.tensor(arr, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)

def main():
    parser = argparse.ArgumentParser(description="Input Representation Comparison for Thesis")
    parser.add_argument("--enc", type=str, required=True, choices=["helixpr", "fcgr"])
    parser.add_argument("--data_file", type=str, default="/home/simionricc/datasets/DNA_data/fullset_test.csv")
    parser.add_argument("--padding_mode", type=str, default="zero", choices=["zero", "circular"])
    parser.add_argument("--target_len", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Inizializzazione: Encoding={args.enc} | Padding={args.padding_mode} | Device={device}")

    df = pd.read_csv(args.data_file, header=None)
    seqs = df[0].values if df.shape[1] == 2 else df[1].values
    labs = df[1].values if df.shape[1] == 2 else df[2].values

    n_tr = int(len(seqs) * 0.8)
    tr_ds = EncodingDataset(seqs[:n_tr], labs[:n_tr], args.enc, args.target_len, args.padding_mode)
    te_ds = EncodingDataset(seqs[n_tr:], labs[n_tr:], args.enc, args.target_len, args.padding_mode)
    tr_l = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    te_l = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    if args.enc == "helixpr":
        model = HelixCNN(in_dim=4, hidden_dim=128).to(device)
        crit = nn.BCEWithLogitsLoss()
    else:
        model = CapsEnhancerModel(in_channels=1, num_classes=2).to(device)
        crit = nn.BCELoss()

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    for ep in range(1, args.epochs + 1):
        model.train()
        t_loss = 0.0
        for x_b, y_b in tr_l:
            opt.zero_grad()
            out = model(x_b.to(device)).squeeze(-1)
            loss = crit(out, y_b.to(device))
            loss.backward()
            opt.step()
            t_loss += loss.item() * len(x_b)

        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for x_b, y_b in te_l:
                out = model(x_b.to(device)).squeeze(-1)
                p = out if args.enc == "fcgr" else torch.sigmoid(out)
                preds.extend(p.cpu().numpy().flatten())
                targets.extend(y_b.numpy().flatten())

        auc = roc_auc_score(targets, preds) * 100.0
        mcc = matthews_corrcoef(targets, (np.array(preds) >= 0.5).astype(int))
        print(f"Epoch {ep:02d} | Train Loss: {t_loss/len(tr_ds):.4f} | Test AUROC: {auc:.2f}% | MCC: {mcc:.4f}")

if __name__ == "__main__":
    main()