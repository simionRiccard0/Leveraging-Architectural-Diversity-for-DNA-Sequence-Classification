#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nucleic Transformer (Yin et al.) - Exact Paper Architecture with GeM Pooling.
Features:
- K-mer aggregation with padding=0 concatenated along the sequence length axis (L_tot=1485).
- Positional Encoding applied on nucleotide embeddings before k-mer aggregation.
- Post-LN (norm_first=False) and standard ReLU activations.
- Embedding without padding_idx to ensure full Adenine gradient updates.
- Generalized Mean Pooling (GeM, p=3) matching Network.py (replaces plain torch.mean).
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
            outputs.append(conv(x_conv))
        
        # Concatena lungo la dimensione della sequenza (dim=2): L_tot = sum(L - k + 1)
        fused = torch.cat(outputs, dim=2)  # [B, ninp, L_tot]
        fused = fused.permute(0, 2, 1)     # [B, L_tot, ninp]
        return self.norm(F.relu(fused))


class GeM(nn.Module):
    """
    Generalized Mean Pooling (GeM) da Network.py (Yin et al.):
    f(x) = ( 1/L * sum( clamp(x, min=eps)^p ) )^(1/p)
    Preserva i motivi locali rari evitando la cancellazione da media globale su L=1485.
    """
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, ninp, L]
        x_clamped = x.clamp(min=self.eps).pow(self.p)
        pooled = F.adaptive_avg_pool1d(x_clamped, 1).pow(1.0 / self.p)
        return pooled.squeeze(-1)  # [B, ninp]


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

        # ntoken=4 senza padding_idx: Adenina sempre attiva e allenabile
        self.embedding = nn.Embedding(ntoken, ninp)
        self.pos_encoder = PositionalEncoding(ninp, max_len=max_len, dropout=dropout)

        if self.kmer_aggregation:
            self.k_mer_aggregate = K_mer_aggregate(
                kmers=kmers if kmers is not None else [2, 3, 4, 5, 6],
                ninp=ninp
            )

        # Post-LN (norm_first=False) e ReLU
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

        # GeM Pooling al posto di torch.mean semplice
        self.pool_layer = GeM(p=3.0)
        self.decoder = nn.Linear(ninp, nclass)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        # src: [B, L]
        emb = self.embedding(src)           # [B, L, ninp]
        emb = self.pos_encoder(emb)         # Positional encoding su nucleotidi

        if self.kmer_aggregation:
            emb = self.k_mer_aggregate(emb) # [B, L_tot, ninp]

        output = self.transformer_encoder(emb)  # [B, L_tot, ninp]
        output = self.norm(output)

        # GeM pooling lungo la dimensione temporale
        pooled = self.pool_layer(output.permute(0, 2, 1))  # [B, ninp]
        logits = self.decoder(pooled)                      # [B, nclass]

        return logits.squeeze(-1) if self.nclass == 1 else logits