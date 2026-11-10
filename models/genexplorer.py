"""
GenExplorer: Hybrid Multi-Branch Neural Architecture for DNA Sequence Classification.
Combines multi-scale convolutions (Pattern & Frequency) with a self-attention Transformer branch.
Updated with Masked Pooling, Transformer Attention Padding Mask, and Length Prior.
Supports 3-channel (Komlós-Hadamard), 4-channel (one-hot, HelixPR) and 5-channel inputs.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatternBranch(nn.Module):
    def __init__(self, in_channels: int = 4, dropout: float = 0.25):
        super().__init__()
        self.conv6 = nn.Conv1d(in_channels, 512, kernel_size=6, padding="same")
        self.conv12 = nn.Conv1d(in_channels, 512, kernel_size=12, padding="same")
        self.conv18 = nn.Conv1d(in_channels, 512, kernel_size=18, padding="same")
        self.conv24 = nn.Conv1d(in_channels, 512, kernel_size=24, padding="same")
        self.bn = nn.BatchNorm1d(2048)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.dim() == 3 and x.shape[-1] in (3, 4, 5):
            x = x.permute(0, 2, 1)

        c = torch.cat([
            F.gelu(self.conv6(x)),
            F.gelu(self.conv12(x)),
            F.gelu(self.conv18(x)),
            F.gelu(self.conv24(x)),
        ], dim=1)

        if mask is None:
            pooled = self.pool(c).squeeze(-1)
        else:
            m = mask.unsqueeze(1).float()
            c_masked = c.masked_fill(m == 0, -1e9)
            pooled = torch.max(c_masked, dim=-1)[0]

        return self.fc(self.bn(pooled))


class FrequencyBranch(nn.Module):
    def __init__(self, in_channels: int = 4, dropout: float = 0.25):
        super().__init__()
        self.conv6 = nn.Conv1d(in_channels, 256, kernel_size=6, padding="same")
        self.conv12 = nn.Conv1d(in_channels, 256, kernel_size=12, padding="same")
        self.conv18 = nn.Conv1d(in_channels, 256, kernel_size=18, padding="same")
        self.conv24 = nn.Conv1d(in_channels, 256, kernel_size=24, padding="same")
        self.bn = nn.BatchNorm1d(1024)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if x.dim() == 3 and x.shape[-1] in (3, 4, 5):
            x = x.permute(0, 2, 1)

        c = torch.cat([
            F.gelu(self.conv6(x)),
            F.gelu(self.conv12(x)),
            F.gelu(self.conv18(x)),
            F.gelu(self.conv24(x)),
        ], dim=1)

        if mask is None:
            pooled = self.pool(c).squeeze(-1)
        else:
            m = mask.unsqueeze(1).float()
            real_l = m.sum(dim=-1).clamp(min=1.0)
            pooled = torch.sum(c * m, dim=-1) / real_l

        return self.fc(self.bn(pooled))


class TransformerBranch(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        max_len: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embedding = nn.Linear(in_channels, d_model)
        self.pos_enc = nn.Embedding(max_len + 1, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, 512)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, l, _ = x.shape
        positions = torch.arange(1, l + 1, device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_enc(positions)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        h = torch.cat([cls_tokens, h], dim=1)

        src_key_padding_mask = None
        if mask is not None:
            cls_mask = torch.zeros((b, 1), dtype=torch.bool, device=x.device)
            pad_mask = (mask == 0)
            src_key_padding_mask = torch.cat([cls_mask, pad_mask], dim=1)

        h_enc = self.transformer(h, src_key_padding_mask=src_key_padding_mask)
        cls_out = self.norm(h_enc[:, 0, :])
        return self.out(cls_out)


class GenExplorer(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        cnn_drop: float = 0.25,
        trans_drop: float = 0.1,
        out_features: int = 1,
        max_len: int = 512,
        use_length_feature: bool = False
    ):
        super().__init__()
        self.use_length_feature = use_length_feature
        self.pattern = PatternBranch(in_channels=in_channels, dropout=cnn_drop)
        self.frequency = FrequencyBranch(in_channels=in_channels, dropout=cnn_drop)
        self.transformer = TransformerBranch(
            in_channels=in_channels,
            dropout=trans_drop,
            max_len=max_len
        )

        fc_in = 1536 + (1 if use_length_feature else 0)
        self.fc = nn.Sequential(
            nn.Linear(fc_in, 512),
            nn.GELU(),
            nn.Dropout(cnn_drop),
            nn.LayerNorm(512),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(cnn_drop),
            nn.Linear(128, out_features)
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if x.shape[-1] == 5:
            x_in = x[..., :4]
        else:
            x_in = x

        h_pat = self.pattern(x_in, mask=mask)
        h_freq = self.frequency(x_in, mask=mask, lengths=lengths)
        h_trans = self.transformer(x_in, mask=mask)

        h_fused = torch.cat([h_pat, h_freq, h_trans], dim=1)

        if self.use_length_feature:
            if lengths is None and mask is not None:
                lengths = mask.sum(dim=-1).float()
            if lengths is not None:
                log_l = torch.log(lengths.unsqueeze(1).clamp(min=1.0) + 1.0)
            else:
                log_l = torch.zeros((x.shape[0], 1), dtype=h_fused.dtype, device=h_fused.device)
            h_fused = torch.cat([h_fused, log_l], dim=1)

        return self.fc(h_fused)
