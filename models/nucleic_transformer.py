#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nucleic Transformer (Yin et al.) 
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, d_model]
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class K_mer_aggregate(nn.Module):
    """
    K-mer Aggregator originale (Yin et al.):
    applica Conv1d con padding=0 per ogni k-mer (out_channels = ninp)
    e concatena lungo l'asse temporale della sequenza.
    """
    def __init__(self, kmers: list, ninp: int = 512):
        super().__init__()
        self.kmers = kmers
        self.convs = nn.ModuleList([
            nn.Conv1d(ninp, ninp, kernel_size=k, padding=0)
            for k in kmers
        ])
        self.norm = nn.LayerNorm(ninp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, ninp] -> permute per Conv1D: [B, ninp, L]
        x_conv = x.permute(0, 2, 1)
        
        outputs = []
        for conv in self.convs:
            # conv(x_conv) shape: [B, ninp, L - k + 1]
            outputs.append(conv(x_conv))
        
        # Concatena lungo la dimensione della sequenza (dim=2):
        # L_tot = sum(L - k + 1). Con L=300 e k=[2,3,4,5,6] -> 299+298+297+296+295 = 1485 token
        fused = torch.cat(outputs, dim=2)  # [B, ninp, L_tot]
        fused = fused.permute(0, 2, 1)     # [B, L_tot, ninp]
        return self.norm(F.relu(fused))


class NucleicTransformer(nn.Module):
    def __init__(
        self,
        ntoken: int = 4,
        nclass: int = 1,
        ninp: int = 512,
        nhead: int = 8,
        nhid: int = 2048,
        nlayers: int = 6,
        kmer_aggregation: bool = True,
        kmers: list = None,
        dropout: float = 0.1,
        max_len: int = 2000
    ):
        super().__init__()
        self.nclass = nclass
        self.ninp = ninp
        self.kmer_aggregation = kmer_aggregation

        # ntoken=4 senza padding_idx: Adenina (indice 0) sempre attiva e allenabile
        self.embedding = nn.Embedding(ntoken, ninp)
        self.pos_encoder = PositionalEncoding(ninp, max_len=max_len, dropout=dropout)

        if self.kmer_aggregation:
            self.k_mer_aggregate = K_mer_aggregate(
                kmers=kmers if kmers is not None else [2, 3, 4, 5, 6],
                ninp=ninp
            )

        # Post-LN (norm_first=False) e ReLU coerenti con lo scheduler AIAYN Noam
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=ninp,
            nhead=nhead,
            dim_feedforward=nhid,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.norm = nn.LayerNorm(ninp)
        self.decoder = nn.Linear(ninp, nclass)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        # src: [B, L]
        emb = self.embedding(src)       # [B, L, ninp]
        
        # 1. Positional Encoding applicato sui nucleotidi (come da paper)
        emb = self.pos_encoder(emb)     # [B, L, ninp]

        # 2. Aggregazione multi-k-mer (espande a L_tot token da 512 feature ciascuno)
        if self.kmer_aggregation:
            emb = self.k_mer_aggregate(emb)  # [B, L_tot, ninp]

        # 3. Transformer Encoder con Self-Attention globale su tutti i k-mer
        output = self.transformer_encoder(emb)  # [B, L_tot, ninp]
        output = self.norm(output)

        # 4. Global Mean Pooling (Eq. 6 del paper) e classificazione finale
        pooled = torch.mean(output, dim=1)      # [B, ninp]
        logits = self.decoder(pooled)           # [B, nclass]

        return logits.squeeze(-1) if self.nclass == 1 else logits