#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Official SequenceCNN architecture from xiangShi22/HelixPR."""

import torch
import torch.nn as nn


class Block(nn.Module):

  def __init__(self, in_channels, out_channels, kernel_size):
    super().__init__()
    self.conv = nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size, padding="same"),
        nn.ReLU(),
        nn.BatchNorm1d(out_channels),
    )

  def forward(self, x):
    return self.conv(x)


class SequenceCNN(nn.Module):

  def __init__(
      self,
      num_classes=2,
      input_dim=4,
      hidden_channels=(128, 256, 128, 64),
      kernel_sizes=(3, 9, 4, 4),
  ):
    super().__init__()
    h1, h2, h3, h4 = hidden_channels
    k1, k2, k3, k4 = kernel_sizes

    self.conv = nn.Sequential(
        Block(input_dim, h1, k1),
        Block(h1, h2, k2),
        Block(h2, h3, k3),
        Block(h3, h4, k4),
    )
    self.pool = nn.AdaptiveAvgPool1d(1)
    self.classifier = nn.Sequential(
        nn.Linear(h4, h4 // 2),
        nn.ReLU(),
        nn.Linear(h4 // 2, num_classes),
    )

  def forward(self, x):
    # x shape in ingresso: (Batch, Length, Channels) -> converte in (Batch, Channels, Length)
    x = x.permute(0, 2, 1)
    x = self.conv(x)
    x = self.pool(x).squeeze(-1)
    return self.classifier(x)


# Alias per compatibilità con il resto della codebase
HelixCNN = SequenceCNN