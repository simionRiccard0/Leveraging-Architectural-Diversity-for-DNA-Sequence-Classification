"""
DNABERT2-CAMP: Pre-trained Genomic Foundation Model integrated with a CAMP Motif Refinement Module[cite: 1, 2].
"""

import torch
import torch.nn as nn
from transformers import AutoModel


class CAMPModule(nn.Module):
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
        x = sequence_output.permute(0, 2, 1)
        c = self.conv_cascade(x)
        c_seq = c.permute(0, 2, 1)
        attn_out, _ = self.mha(c_seq, c_seq, c_seq)
        return torch.mean(attn_out, dim=1)


class DNABERT2CAMP(nn.Module):
    def __init__(self, model_name_or_path: str = "zhihan1996/DNABERT-2-117M", num_classes: int = 1, dropout: float = 0.5):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
        hidden_dim = self.bert.config.hidden_size  # 768
        self.num_classes = num_classes
        
        self.pooler = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        self.camp = CAMPModule(in_dim=hidden_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + 128, 128),
            nn.LayerNorm(128),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        seq_output = outputs[0]
        
        trans_pooled = self.pooler(seq_output[:, 0, :])
        camp_pooled = self.camp(seq_output)
        
        fused = torch.cat([trans_pooled, camp_pooled], dim=1)
        out = self.classifier(fused)
        return out.squeeze(-1) if self.num_classes == 1 else out