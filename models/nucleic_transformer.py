"""
Nucleic Transformer: DNA Sequence Classification with 1D Convolutional k-mer Tokenization and Self-Attention (He et al.)[cite: 9].
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


class MultiKmerTokenizer(nn.Module):
    """
    1D Convolutional k-mer Tokenizer aggregating multi-scale k-mer representations[cite: 9].
    Ensures strict length alignment (L) across all kernel sizes (k in [2, 3, 4, 5, 6])[cite: 9].
    """
    def __init__(self, ninp: int = 512, kmers: list = None):
        super().__init__()
        if kmers is None:
            kmers = [2, 3, 4, 5, 6]
        self.kmers = kmers
        self.num_kmers = len(kmers)
        
        split_size = ninp // self.num_kmers
        remainder = ninp % self.num_kmers
        self.out_channels = [split_size + (1 if i < remainder else 0) for i in range(self.num_kmers)]

        self.convs = nn.ModuleList([
            nn.Conv1d(ninp, self.out_channels[i], kernel_size=k, padding="same")
            for i, k in enumerate(self.kmers)
        ])
        self.norm = nn.LayerNorm(ninp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, ninp] -> permute for Conv1D: [B, ninp, L]
        L = x.size(1)
        x_conv = x.permute(0, 2, 1)
        
        outputs = []
        for conv in self.convs:
            out = conv(x_conv)
            # Garantisce la dimensione temporale identica L per qualsiasi parità di k
            if out.size(2) != L:
                out = out[:, :, :L]
            outputs.append(out)
            
        fused = torch.cat(outputs, dim=1)  # [B, ninp, L]
        fused = fused.permute(0, 2, 1)     # [B, L, ninp]
        return self.norm(F.gelu(fused))


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

        self.embedding = nn.Embedding(ntoken + 1, ninp, padding_idx=0)
        
        if self.kmer_aggregation:
            self.kmer_tokenizer = MultiKmerTokenizer(ninp=ninp, kmers=kmers if kmers is not None else [2, 3, 4, 5, 6])
        else:
            self.kmer_tokenizer = nn.Identity()

        self.pos_encoder = PositionalEncoding(ninp, max_len=max_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=ninp,
            nhead=nhead,
            dim_feedforward=nhid,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.norm = nn.LayerNorm(ninp)
        self.decoder = nn.Linear(ninp, nclass)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        # src: [B, L]
        emb = self.embedding(src) * math.sqrt(self.ninp)  # [B, L, ninp]
        
        if self.kmer_aggregation:
            emb = self.kmer_tokenizer(emb)
            
        emb = self.pos_encoder(emb)
        output = self.transformer_encoder(emb)
        output = self.norm(output)
        
        # Global Mean Pooling (Eq. 6 del paper)[cite: 9]
        pooled = torch.mean(output, dim=1)
        logits = self.decoder(pooled)
        
        return logits.squeeze(-1) if self.nclass == 1 else logits