import os
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

# Configurazione ambiente deep learning
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_viraexplorer_v2.pth"
BASE_NOISE_DIR = "data_profs/SIM"

# ========================================== #
# ARCHITETTURA COMPLETA ViraExplorer         #
# ========================================== #

mapping = {"A":[1,0,0,0], "C":[0,1,0,0], "G":[0,0,1,0], "T":[0,0,0,1], "N":[0,0,0,0]}
MAX_LEN = 300

def encode_sequence(seq):
    seq = seq[:MAX_LEN]
    encoded = [mapping.get(b, [0,0,0,0]) for b in seq]
    while len(encoded) < MAX_LEN:
        encoded.append([0,0,0,0])
    return encoded

class DNADataset(Dataset):
    def __init__(self, dataframe):
        self.sequences = dataframe[1].values
        self.labels    = dataframe[2].values
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, idx):
        x = torch.tensor(encode_sequence(self.sequences[idx]), dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y

class PatternBranch(nn.Module):
    def __init__(self, dropout=0.25):
        super().__init__()
        self.conv6  = nn.Conv1d(4, 512, 6,  padding='same')
        self.conv12 = nn.Conv1d(4, 512, 12, padding='same')
        self.conv18 = nn.Conv1d(4, 512, 18, padding='same')
        self.conv24 = nn.Conv1d(4, 512, 24, padding='same')
        self.bn   = nn.BatchNorm1d(2048)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(nn.Linear(2048, 512), nn.GELU(), nn.Dropout(dropout))
    def forward(self, x):
        x = x.permute(0,2,1)
        x = torch.cat([F.gelu(self.conv6(x)), F.gelu(self.conv12(x)), F.gelu(self.conv18(x)), F.gelu(self.conv24(x))], dim=1)
        x = self.bn(x) # Applica BatchNorm
        return self.fc(self.pool(x).squeeze(-1)) # Corretto!

class FrequencyBranch(nn.Module):
    def __init__(self, dropout=0.25):
        super().__init__()
        self.conv6  = nn.Conv1d(4, 256, 6,  padding='same')
        self.conv12 = nn.Conv1d(4, 256, 12, padding='same')
        self.conv18 = nn.Conv1d(4, 256, 18, padding='same')
        self.conv24 = nn.Conv1d(4, 256, 24, padding='same')
        self.bn   = nn.BatchNorm1d(1024)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(nn.Linear(1024, 512), nn.GELU(), nn.Dropout(dropout))
    def forward(self, x):
        x = x.permute(0,2,1)
        x = torch.cat([F.gelu(self.conv6(x)), F.gelu(self.conv12(x)), F.gelu(self.conv18(x)), F.gelu(self.conv24(x))], dim=1)
        x = self.bn(x) # Applica BatchNorm
        return self.fc(self.pool(x).squeeze(-1)) # Corretto!

class TransformerBranch(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=4, max_len=300, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(4, d_model)
        self.pos_enc   = nn.Embedding(max_len+1, d_model)
        self.cls_token = nn.Parameter(torch.randn(1,1,d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, dropout=dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.out  = nn.Linear(d_model, 512)
    def forward(self, x):
        B, L, _ = x.shape
        x = self.embedding(x) + self.pos_enc(torch.arange(1, L+1, device=x.device).unsqueeze(0))
        x = torch.cat([self.cls_token.expand(B,-1,-1), x], dim=1)
        return self.out(self.norm(self.transformer(x))[:,0,:])

class ViraExplorer(nn.Module):
    def __init__(self, cnn_dropout=0.25, transformer_dropout=0.1):
        super().__init__()
        self.pattern     = PatternBranch(cnn_dropout)
        self.frequency   = FrequencyBranch(cnn_dropout)
        self.transformer = TransformerBranch(dropout=transformer_dropout)
        self.fc = nn.Sequential(
            nn.Linear(1536, 512), nn.GELU(), nn.Dropout(cnn_dropout), nn.LayerNorm(512),
            nn.Linear(512, 128), nn.GELU(), nn.Dropout(cnn_dropout), nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.fc(torch.cat([self.pattern(x), self.frequency(x), self.transformer(x)], dim=1))

# ========================================== #
# PIPELINE DI VALUTAZIONE DINAMICA          #
# ========================================== #

def run_evaluation():
    print(f"Dispositivo rilevato per l'inferenza: {device}")
    model = ViraExplorer().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    noise_levels = ['1', '5', '10']
    summary = {}
    
    print("\n==================================================")
    print("🚀 AVVIO VALUTAZIONE STRUTTURALE COMPLETA (SIM)")
    print("==================================================")
    
    for lvl in noise_levels:
        auroc_seeds = []
        
        for seed_idx in range(1, 6):
            if lvl == '1':
                seed_folder = f"seed_100{seed_idx}"
            elif lvl == '5':
                seed_folder = f"seed_500{seed_idx}"
            elif lvl == '10':
                seed_folder = f"seed_1000{seed_idx}"
                
            file_path = os.path.join(BASE_NOISE_DIR, f"noise_{lvl}", seed_folder, "fullset_test.csv")
            
            if not os.path.exists(file_path):
                print(f"⚠️ Mappatura fallita per: {file_path}")
                continue
                
            df_noise = pd.read_csv(file_path, header=None)
            dataset = DNADataset(df_noise)
            loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=2)
            
            preds, targets = [], []
            with torch.no_grad():
                for X_batch, y_batch in loader:
                    probs = torch.sigmoid(model(X_batch.to(device)))
                    preds.extend(probs.cpu().numpy().flatten())
                    targets.extend(y_batch.numpy())
            
            auroc = roc_auc_score(targets, preds)
            auroc_seeds.append(auroc)
            print(f"-> Livello {lvl}% | {seed_folder} | AUROC: {auroc:.4f}")
            
        if auroc_seeds:
            summary[lvl] = {
                'mean': np.mean(auroc_seeds),
                'std': np.std(auroc_seeds)
            }
            
    print("\n==================================================")
    print("📊 RISULTATI AGGREGATI PER I PROFESSORI")
    print("==================================================")
    for lvl, m in summary.items():
        print(f"Rumore {lvl:2s}% -> AUROC Media: {m['mean']:.4f} (± {m['std']:.4f})")

if __name__ == "__main__":
    run_evaluation()
