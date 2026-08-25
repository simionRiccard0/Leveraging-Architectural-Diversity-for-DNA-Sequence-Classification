"""
Nucleic Transformer: 1D-Convolutional k-mer Tokenization with Multi-Head Self-Attention[cite: 1, 4].
"""

import math
from typing import List
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding for nucleotide sequences[cite: 1, 4]."""

    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (Batch, Seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]


class KMerTokenizer(nn.Module):
    """Continuous k-mer tokenization via parallel 1D convolutional layers[cite: 1, 4]."""

    def __init__(self, d_model: int, kmers: List[int]):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(d_model, d_model // len(kmers), kernel_size=k, stride=1, padding=k // 2)
            for k in kmers
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (Batch, Length, d_model) -> Permute to (Batch, d_model, Length)
        x_t = x.permute(0, 2, 1)
        outputs = [c(x_t) for c in self.convs]
        out = torch.cat(outputs, dim=1)
        return out.permute(0, 2, 1)


class NucleicTransformer(nn.Module):
    """Nucleic Transformer sequence classification model[cite: 1, 4]."""

    def __init__(
        self,
        ntoken: int = 4,
        nclass: int = 1,
        ninp: int = 512,
        nhead: int = 8,
        nhid: int = 2048,
        nlayers: int = 6,
        kmers: List[int] = [2, 3, 4, 5, 6],
        dropout: float = 0.1
    ):
        super().__init__()
        self.embedding = nn.Embedding(ntoken, ninp)
        self.kmer_tokenizer = KMerTokenizer(ninp, kmers)
        self.pos_encoder = PositionalEncoding(ninp)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=ninp,
            nhead=nhead,
            dim_feedforward=nhid,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=nlayers)
        self.decoder = nn.Linear(ninp, nclass)
        self.ninp = ninp

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        # src: Integer nucleotide indices tensor (Batch, Length)
        emb = self.embedding(src) * math.sqrt(self.ninp)
        emb = self.kmer_tokenizer(emb)
        emb = self.pos_encoder(emb)
        encoded = self.transformer_encoder(emb)
        # Global Spatial Mean Pooling
        pooled = torch.mean(encoded, dim=1)
        return self.decoder(pooled)