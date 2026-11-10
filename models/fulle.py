"""
FullE: 10-Subnetwork Convolutional Neural Network Ensemble with Parameterized Lp Pooling.
Supports variable-length masking and length prior exclusively when active.
"""

from typing import List, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class PowerAveragePooling1d(nn.Module):
    def __init__(self, init_p: float = 2.0):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(init_p)))

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        p = torch.clamp(self.p, min=1.0, max=15.0)
        x_clamped = torch.clamp(x, min=1e-6)

        if mask is None:
            x_pow = torch.pow(x_clamped, p)
            avg_pow = torch.mean(x_pow, dim=-1)
            return torch.pow(avg_pow + 1e-6, 1.0 / p)

        m = mask.unsqueeze(1).float()
        real_l = lengths.unsqueeze(1).float() if lengths is not None else m.sum(dim=-1).clamp(min=1.0)
        x_pow = torch.pow(x_clamped, p) * m
        avg_pow = torch.sum(x_pow, dim=-1) / real_l
        return torch.pow(avg_pow + 1e-6, 1.0 / p)


class FullEBranch(nn.Module):
    def __init__(
        self,
        in_channels: int = 5,
        out_channels: int = 1200,
        kernel_size: int = 11,
        pool_type: str = "max",
        dropout: float = 0.2,
        num_classes: int = 1,
        use_length_feature: bool = False
    ):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding="same")
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(out_channels)
        self.pool_type = pool_type.lower()
        self.num_classes = num_classes
        self.use_length_feature = use_length_feature

        if self.pool_type == "max":
            self.pool = nn.AdaptiveMaxPool1d(1)
        elif self.pool_type == "avg":
            self.pool = nn.AdaptiveAvgPool1d(1)
        elif self.pool_type == "lp":
            self.pool = PowerAveragePooling1d(init_p=2.0)
        else:
            raise ValueError(f"Unsupported pooling type: {pool_type}")

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(out_channels, 1000)

        cls_in = 1000 + (1 if use_length_feature else 0)
        self.classifier = nn.Linear(cls_in, num_classes)

    def extract_features(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if x.dim() == 3 and x.shape[-1] in (4, 5):
            x = x.permute(0, 2, 1)

        if x.shape[1] == 4 and self.conv.in_channels == 5:
            pad_n = torch.zeros((x.shape[0], 1, x.shape[2]), dtype=x.dtype, device=x.device)
            x = torch.cat([x, pad_n], dim=1)

        h = self.relu(self.conv(x))
        h = self.bn(h)

        if mask is None:
            if self.pool_type == "lp":
                h_pooled = self.pool(h)
            else:
                h_pooled = self.pool(h).squeeze(-1)
        else:
            m = mask.unsqueeze(1).float()
            real_l = lengths.unsqueeze(1).float() if lengths is not None else m.sum(dim=-1).clamp(min=1.0)

            if self.pool_type == "max":
                h_masked = h.masked_fill(m == 0, -1e9)
                h_pooled = torch.max(h_masked, dim=-1)[0]
            elif self.pool_type == "avg":
                h_pooled = torch.sum(h * m, dim=-1) / real_l
            elif self.pool_type == "lp":
                h_pooled = self.pool(h, mask=mask, lengths=lengths)

        h_feat = self.dropout(h_pooled)
        return self.relu(self.fc(h_feat))

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        feat = self.extract_features(x, mask=mask, lengths=lengths)

        if self.use_length_feature:
            if lengths is None and mask is not None:
                lengths = mask.sum(dim=-1).float()
            if lengths is not None:
                log_l = torch.log(lengths.unsqueeze(1).clamp(min=1.0) + 1.0)
            else:
                log_l = torch.zeros((x.shape[0], 1), dtype=feat.dtype, device=feat.device)
            feat = torch.cat([feat, log_l], dim=1)

        out = self.classifier(feat)
        return out.squeeze(-1) if self.num_classes == 1 else out


class FullEMerger(nn.Module):
    def __init__(
        self,
        branches: List[FullEBranch],
        dropout: float = 0.2,
        num_classes: int = 1,
        use_length_feature: bool = False
    ):
        super().__init__()
        self.branches = nn.ModuleList(branches)
        self.num_classes = num_classes
        self.use_length_feature = use_length_feature

        total_dim = (1000 * len(branches)) + (1 if use_length_feature else 0)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(total_dim, 500),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(500, num_classes)
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        feats = [b.extract_features(x, mask=mask, lengths=lengths) for b in self.branches]
        h_fused = torch.cat(feats, dim=1)

        if self.use_length_feature:
            if lengths is None and mask is not None:
                lengths = mask.sum(dim=-1).float()
            if lengths is not None:
                log_l = torch.log(lengths.unsqueeze(1).clamp(min=1.0) + 1.0)
            else:
                log_l = torch.zeros((x.shape[0], 1), dtype=h_fused.dtype, device=h_fused.device)
            h_fused = torch.cat([h_fused, log_l], dim=1)

        out = self.classifier(h_fused)
        return out.squeeze(-1) if self.num_classes == 1 else out


class FullE10Ensemble(nn.Module):
    def __init__(
        self,
        branches: Dict[str, FullEBranch],
        mergers: Dict[str, FullEMerger],
        num_classes: int = 1
    ):
        super().__init__()
        self.branches = nn.ModuleDict(branches)
        self.mergers = nn.ModuleDict(mergers)
        self.num_classes = num_classes

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        probs = []
        is_ce = (self.num_classes == 2)
        all_models = list(self.branches.values()) + list(self.mergers.values())

        for m in all_models:
            out = m(x, mask=mask, lengths=lengths)
            p = F.softmax(out, dim=1)[:, 1] if is_ce else torch.sigmoid(out)
            probs.append(p)

        avg_prob = torch.stack(probs, dim=0).mean(dim=0)
        eps = 1e-7
        avg_prob = torch.clamp(avg_prob, min=eps, max=1.0 - eps)
        logit = torch.log(avg_prob / (1.0 - avg_prob))
        return logit


def build_fulle_10ensemble(
    in_channels: int = 5,
    num_classes: int = 1,
    dropout: float = 0.2,
    use_length_feature: bool = False
) -> FullE10Ensemble:
    """Builds the 10-network FullE ensemble (5 branches + 5 mergers)."""
    branches = {
        "freq": FullEBranch(in_channels=in_channels, out_channels=1000, kernel_size=8, pool_type="avg", dropout=dropout, num_classes=num_classes, use_length_feature=use_length_feature),
        "freq_paper": FullEBranch(in_channels=in_channels, out_channels=1000, kernel_size=8, pool_type="avg", dropout=dropout, num_classes=num_classes, use_length_feature=use_length_feature),
        "pat": FullEBranch(in_channels=in_channels, out_channels=1200, kernel_size=11, pool_type="max", dropout=dropout, num_classes=num_classes, use_length_feature=use_length_feature),
        "pat_paper": FullEBranch(in_channels=in_channels, out_channels=1200, kernel_size=11, pool_type="max", dropout=dropout, num_classes=num_classes, use_length_feature=use_length_feature),
        "lp": FullEBranch(in_channels=in_channels, out_channels=1200, kernel_size=11, pool_type="lp", dropout=dropout, num_classes=num_classes, use_length_feature=use_length_feature),
    }
    mergers = {
        "merger_fpl": FullEMerger([branches["freq"], branches["pat"], branches["lp"]], dropout=dropout, num_classes=num_classes, use_length_feature=use_length_feature),
        "merger_fl": FullEMerger([branches["freq"], branches["lp"]], dropout=dropout, num_classes=num_classes, use_length_feature=use_length_feature),
        "merger_pl": FullEMerger([branches["pat"], branches["lp"]], dropout=dropout, num_classes=num_classes, use_length_feature=use_length_feature),
        "merger_vm": FullEMerger([branches["freq"], branches["pat"]], dropout=dropout, num_classes=num_classes, use_length_feature=use_length_feature),
        "merger_vm_paper": FullEMerger([branches["freq_paper"], branches["pat_paper"]], dropout=dropout, num_classes=num_classes, use_length_feature=use_length_feature),
    }
    return FullE10Ensemble(branches, mergers, num_classes=num_classes)

FullEEnsemble = FullE10Ensemble
