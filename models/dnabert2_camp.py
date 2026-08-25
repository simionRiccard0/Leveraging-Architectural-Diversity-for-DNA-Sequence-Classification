"""
DNABERT2-CAMP: Pre-trained Genomic Foundation Model integrated with a CAMP Motif Refinement Module[cite: 1, 2].
"""

import torch
import torch.nn as nn
from transformers import AutoModel


class CAMPModule(nn.Module):
    """CAMP Module: 4-Layer 1D CNN cascade + Multi-Head Self-Attention + Mean Pooling[cite: 2]."""

    def __init__(self, in_dim: int = 768):
        super().__init__()
        self.conv_cascade = nn.Sequential(
            nn.Conv1d(in_dim, 512, kernel_size=3, padding="same"),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 384, kernel_size=3, padding="same"),
            nn.BatchNorm1d(384),
            nn.ReLU(),
            nn.Conv1d(384, 256, kernel_size=3, padding="same"),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=3, padding="same"),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.mha = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        # sequence_output: (Batch, Length, 768) -> Permute to (Batch, 768, Length)
        x = sequence_output.permute(0, 2, 1)
        c = self.conv_cascade(x)         # (Batch, 128, Length)
        c_seq = c.permute(0, 2, 1)       # (Batch, Length, 128)
        attn_out, _ = self.mha(c_seq, c_seq, c_seq)
        return torch.mean(attn_out, dim=1)  # (Batch, 128)


class DNABERT2CAMP(nn.Module):
    """DNABERT2-CAMP Classifier: Merges global Transformer embeddings with localized motif descriptors[cite: 2]."""

    def __init__(self, model_name_or_path: str = "zhihan1996/DNABERT-2-117M", dropout: float = 0.5):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
        hidden_dim = self.bert.config.hidden_size  # 768
        
        self.pooler = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        self.camp = CAMPModule(in_dim=hidden_dim)
        
        # Feature Fusion: 768 (Global Transformer) + 128 (Local CAMP) = 896
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + 128, 128),
            nn.LayerNorm(128),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        seq_output = outputs[0]  # (Batch, Length, 768)
        
        # Global Transformer Embedding (CLS token projection)
        trans_pooled = self.pooler(seq_output[:, 0, :])  # (Batch, 768)
        
        # Local CAMP Refinement
        camp_pooled = self.camp(seq_output)             # (Batch, 128)
        
        # Concatenation and Projection
        fused = torch.cat([trans_pooled, camp_pooled], dim=1)  # (Batch, 896)
        return self.classifier(fused).squeeze(-1)