"""Train the temporal head on cached embeddings. CPU-friendly by design.

Reads cached embeddings (data/cache/<encoder>/<station>/<winter>.npy) + the clip
manifest with labels, builds per-winter sequences, trains TemporalHead, and
writes predictions + event dates for evaluation.

This is a skeleton: the data-loading and label wiring are marked TODO because
they depend on the labeling decision in assemble_clips. The model, loss, and
loop are ready.

Usage:
  python -m fm_ice.models.train --encoder vjepa2 --train-station cedarburg \
      --test-winter 2024-2025 --epochs 50
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from fm_ice.config import load_yaml
from fm_ice.models.temporal_head import TemporalHead, total_loss


def load_sequence(encoder: str, station: str, winter: str):
    """TODO: load (T, D) embeddings, per-step labels, and air-temp channel.

    Returns (X: Tensor[T, D(+1)], y: Tensor[T], index: DataFrame).
    """
    raise NotImplementedError("Wire to data/cache + labeled clip manifest.")


def fit(encoder: str, train_station: str, test_winter: str, epochs: int, kind: str):
    cfg = load_yaml("pipeline.yaml")
    th = cfg["temporal_head"]
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])

    # TODO: assemble train/val/test splits per stations.yaml 'splits'.
    # X_train, y_train = ...
    model = TemporalHead(in_dim=None, hidden=th["hidden"], layers=th["layers"],
                         dropout=th["dropout"], kind=kind)  # in_dim set after load
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for ep in range(epochs):
        model.train()
        # logits = model(X_train); loss = total_loss(logits, y_train, th['smoothing_loss_weight'])
        # opt.zero_grad(); loss.backward(); opt.step()
        pass
    print("TODO: implement loop body once load_sequence is wired.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", default="vjepa2", choices=["vjepa2", "dinov2"])
    ap.add_argument("--train-station", default="cedarburg")
    ap.add_argument("--test-winter", default="2024-2025")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--head", default="tcn", choices=["tcn", "transformer"])
    args = ap.parse_args()
    fit(args.encoder, args.train_station, args.test_winter, args.epochs, args.head)


if __name__ == "__main__":
    main()
