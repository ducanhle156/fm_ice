"""Light temporal head over the frozen embedding sequence.

Input:  (B, T, D) clip embeddings, optionally with air temperature concatenated
        as an extra channel (D -> D+1).
Output: (B, T) per-step ice-state logits. Onset/breakup are read off as the
        transitions in the predicted state sequence.

Two variants (config: temporal_head.type):
  tcn          MS-TCN style dilated temporal conv stack with a smoothing loss.
  transformer  small causal/bidirectional transformer encoder.

Keep this small: the encoder is frozen, the dataset is tiny, and the published
design economics (Presto, VadCLIP) favor a light head on strong frozen features.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TCNBlock(nn.Module):
    def __init__(self, ch: int, dilation: int, dropout: float):
        super().__init__()
        pad = dilation
        self.conv1 = nn.Conv1d(ch, ch, 3, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(ch, ch, 3, padding=pad, dilation=dilation)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                      # x: (B, C, T)
        y = F.relu(self.conv1(x))[..., : x.size(-1)]
        y = self.drop(F.relu(self.conv2(y))[..., : x.size(-1)])
        return x + y


class TemporalHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, layers: int = 4,
                 dropout: float = 0.1, kind: str = "tcn"):
        super().__init__()
        self.kind = kind
        self.proj = nn.Linear(in_dim, hidden)
        if kind == "tcn":
            self.body = nn.ModuleList([TCNBlock(hidden, 2 ** i, dropout) for i in range(layers)])
        elif kind == "transformer":
            enc = nn.TransformerEncoderLayer(hidden, nhead=4, dim_feedforward=hidden * 2,
                                             dropout=dropout, batch_first=True)
            self.body = nn.TransformerEncoder(enc, num_layers=layers)
        else:
            raise ValueError(kind)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x):                       # x: (B, T, in_dim)
        h = self.proj(x)
        if self.kind == "tcn":
            h = h.transpose(1, 2)               # (B, hidden, T)
            for blk in self.body:
                h = blk(h)
            h = h.transpose(1, 2)
        else:
            h = self.body(h)
        return self.out(h).squeeze(-1)          # (B, T) logits


def smoothing_loss(logits: torch.Tensor) -> torch.Tensor:
    """MS-TCN truncated-MSE smoothing on the temporal dimension to kill flicker."""
    logp = F.logsigmoid(logits)
    d = (logp[:, 1:] - logp[:, :-1]) ** 2
    return torch.clamp(d, max=16.0).mean()


def total_loss(logits, target, w_smooth: float = 0.15, pos_weight=None):
    """BCE (optionally class-balanced via pos_weight) + MS-TCN smoothing."""
    bce = F.binary_cross_entropy_with_logits(logits, target.float(), pos_weight=pos_weight)
    return bce + w_smooth * smoothing_loss(logits)


if __name__ == "__main__":
    m = TemporalHead(in_dim=1024 + 1, kind="tcn")
    x = torch.randn(2, 120, 1025)
    y = m(x)
    print("logits:", y.shape)   # (2, 120)
