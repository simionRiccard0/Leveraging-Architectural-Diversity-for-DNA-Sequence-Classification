import os
os.listdir('ViraMiner/data/DNA_data/')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

import pandas as pd

train = pd.read_csv(
    'ViraMiner/data/DNA_data/fullset_train.csv',
    header=None
)

val = pd.read_csv(
    'ViraMiner/data/DNA_data/fullset_validation.csv',
    header=None
)

test = pd.read_csv(
    'ViraMiner/data/DNA_data/fullset_test.csv',
    header=None
)

print("Train shape:", train.shape)
print(train.head())

print("Sequence length:", len(train[1][0]))

print("\nClass balance:")
print(train[2].value_counts())

"""DNA sequences encoding:"""

import torch
from torch.utils.data import Dataset
import random # Added for sequence augmentation

#DNA one-hot encoding
mapping = {
    "A": [1,0,0,0],
    "C": [0,1,0,0],
    "G": [0,0,1,0],
    "T": [0,0,0,1],
    "N": [0,0,0,0]
}

MAX_LEN = 300

def encode_sequence(seq, augment=False):
    complement = {'A':'T','T':'A','C':'G','G':'C','N':'N'}
    if augment and random.random() > 0.5:
        seq = ''.join(complement.get(base, 'N') for base in reversed(seq))
    if augment:
        shift = random.randint(-5, 5)
        seq = seq[max(0, shift):] if shift > 0 else 'N'*(-shift) + seq

    seq = seq[:MAX_LEN]

    encoded = [
        mapping.get(base, [0,0,0,0])
        for base in seq
    ]

    while len(encoded) < MAX_LEN:
        encoded.append([0,0,0,0])

    return encoded


class DNADataset(Dataset):

    def __init__(self, dataframe, augment=False):

        self.sequences = dataframe[1].values
        self.labels = dataframe[2].values
        self.augment = augment # Added for augmentation

    def __len__(self):

        return len(self.sequences)

    def __getitem__(self, idx):

        seq = self.sequences[idx]
        label = self.labels[idx]

        x = torch.tensor(
            encode_sequence(seq, self.augment), # Pass augment flag
            dtype=torch.float32
        )

        y = torch.tensor(
            label,
            dtype=torch.float32
        )

        return x, y

"""Preparing the dataset (DataLoader):"""

from torch.utils.data import DataLoader

train_dataset = DNADataset(train, augment=True) # Added augment=True
val_dataset   = DNADataset(val)
test_dataset  = DNADataset(test)

train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=2
)

X_batch, y_batch = next(iter(train_loader))

print(X_batch.shape)
print(y_batch.shape)

"""CNN Branches"""

import torch.nn as nn
import torch.nn.functional as F

class PatternBranch(nn.Module):
    def __init__(self, dropout): # Added dropout parameter
        super().__init__()
        self.conv6  = nn.Conv1d(4, 512, 6,  padding='same')
        self.conv12 = nn.Conv1d(4, 512, 12, padding='same')
        self.conv18 = nn.Conv1d(4, 512, 18, padding='same')
        self.conv24 = nn.Conv1d(4, 512, 24, padding='same')

        total = 512 * 4
        self.bn   = nn.BatchNorm1d(total)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc   = nn.Sequential(
            nn.Linear(total, 512),
            nn.GELU(),
            nn.Dropout(dropout) # Used passed dropout
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        branches = [
            F.gelu(self.conv6(x)),
            F.gelu(self.conv12(x)),
            F.gelu(self.conv18(x)),
            F.gelu(self.conv24(x)),
        ]
        x = torch.cat(branches, dim=1)
        x = self.bn(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

class FrequencyBranch(nn.Module):
    def __init__(self, dropout): # Added dropout parameter
        super().__init__()
        self.conv6  = nn.Conv1d(4, 256, 6,  padding='same')
        self.conv12 = nn.Conv1d(4, 256, 12, padding='same')
        self.conv18 = nn.Conv1d(4, 256, 18, padding='same')
        self.conv24 = nn.Conv1d(4, 256, 24, padding='same')

        total = 256 * 4
        self.bn   = nn.BatchNorm1d(total)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc   = nn.Sequential(
            nn.Linear(total, 512),
            nn.GELU(),
            nn.Dropout(dropout) # Used passed dropout
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        branches = [
            F.gelu(self.conv6(x)),
            F.gelu(self.conv12(x)),
            F.gelu(self.conv18(x)),
            F.gelu(self.conv24(x)),
        ]
        x = torch.cat(branches, dim=1)
        x = self.bn(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

"""Transformer branch"""

class TransformerBranch(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=4, max_len=300, dropout=0.1): # Added dropout parameter
        super().__init__()
        self.embedding = nn.Linear(4, d_model)
        self.pos_enc   = nn.Embedding(max_len + 1, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, # Used passed dropout
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.out  = nn.Linear(d_model, 512)

    def forward(self, x):
        B, L, _ = x.shape
        x = self.embedding(x)
        positions = torch.arange(1, L + 1, device=x.device).unsqueeze(0)
        x = x + self.pos_enc(positions)
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)
        x = self.transformer(x)
        x = self.norm(x)
        return self.out(x[:, 0, :])

"""ViraExplorer Model (CNN Pattern branch + CNN Frequency branch + Transformer)"""

class ViraExplorer(nn.Module):
    def __init__(self, cnn_dropout=0.25, transformer_dropout=0.1): # Modified dropout parameters
        super().__init__()
        self.pattern     = PatternBranch(cnn_dropout)     # Used cnn_dropout
        self.frequency   = FrequencyBranch(cnn_dropout)   # Used cnn_dropout
        self.transformer = TransformerBranch(dropout=transformer_dropout) # Used transformer_dropout
        self.fc = nn.Sequential(
            nn.Linear(1536, 512),
            nn.GELU(),
            nn.Dropout(cnn_dropout), # Used cnn_dropout
            nn.LayerNorm(512),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(cnn_dropout), # Used cnn_dropout
            nn.Linear(128, 1)
        )

    def forward(self, x):
        p = self.pattern(x)
        f = self.frequency(x)
        t = self.transformer(x)
        x = torch.cat([p, f, t], dim=1)
        return self.fc(x)

"""Focal Loss"""

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, smoothing=0.05): # Added smoothing parameter
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing # Stored smoothing parameter

    def forward(self, logits, targets):
        targets = targets.unsqueeze(1)
        # Apply label smoothing
        targets_smooth = targets * (1 - self.smoothing) + 0.5 * self.smoothing # Applied label smoothing
        bce     = F.binary_cross_entropy_with_logits(logits, targets_smooth, reduction='none') # Used smoothed targets
        pt           = torch.exp(-bce)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()

"""Training Loop"""

import math # Added for lr_lambda

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model = ViraExplorer(cnn_dropout=0.25, transformer_dropout=0.1).to(device) # Modified model instantiation with new dropout parameters
criterion = FocalLoss(alpha=0.75, gamma=2.0, smoothing=0.05) # Added smoothing to FocalLoss

transformer_params = list(model.transformer.parameters())
other_params = list(model.pattern.parameters()) + list(model.frequency.parameters()) + list(model.fc.parameters())

# Define lr_lambda function for scheduler
def lr_lambda(epoch):
    warmup = 5
    if epoch < warmup:
        return epoch / warmup  # linear warmup
    # cosine decay after warmup
    progress = (epoch - warmup) / (EPOCHS - warmup)
    return 0.5 * (1 + math.cos(math.pi * progress))

optimizer = torch.optim.AdamW([
    {"params": other_params, "lr": 1e-4},
    {"params": transformer_params, "lr": 5e-5}, # Raised transformer LR
], weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda) # Changed scheduler to LambdaLR
scaler = torch.amp.GradScaler('cuda')

best_auc, patience, counter, EPOCHS = 0, 15, 0, 120

# Curriculum training: Freeze transformer at start
for p in model.transformer.parameters():
    p.requires_grad = False

WARMUP_EPOCHS = 5 # Defined WARMUP_EPOCHS

for epoch in range(EPOCHS):
    # Unfreeze transformer after warmup
    if epoch == WARMUP_EPOCHS:
        for p in model.transformer.parameters():
            p.requires_grad = True
        print("Transformer unfrozen")

    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()

    scheduler.step()
    model.eval()
    preds, targets_list = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            probs = torch.sigmoid(model(X_batch))
            preds.extend(probs.cpu().numpy().flatten())
            targets_list.extend(y_batch.numpy())

    auc = roc_auc_score(targets_list, preds)
    print(f"Epoch {epoch+1:3d} | Loss: {total_loss/len(train_loader):.4f} | Val AUC: {auc:.4f}")

    if auc > best_auc:
        best_auc = auc
        counter = 0
        torch.save(model.state_dict(), "best_viraexplorer_v2.pth")
    else:
        counter += 1
        if counter >= patience: break
