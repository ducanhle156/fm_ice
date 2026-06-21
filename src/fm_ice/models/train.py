"""Phase 3 -- train the temporal head and read onset/breakup off its output.

Pipeline (all CPU, reads only the cached .npy embeddings + the labeled clip
manifest -- no GPU, no network):

  load_sequence   per winter: cached embeddings in time order, the air-temperature
                  channel, and the dense per-step ice label.
  fit             train TemporalHead (TCN default) on the train winter(s) with a
                  class-balanced BCE + MS-TCN smoothing loss.
  evaluate        predict the ice-state sequence on the held-out winter, read
                  onset/breakup with fm_ice.models.events, and score timing error
                  in hours against the references in results/reference_events.csv.

H1 anchor: alongside the temporal head we fit a per-frame logistic probe on the
SAME features and read events from it the same way. The head-vs-probe gap is the
direct test of H1 ("temporal modeling beats per-frame thresholding"). The
RIce-Net baseline (the other H1 anchor) is produced separately on the GPU box and
dropped into the same timing table.

Split protocol (IMPORTANT -- deviates from the literal plan for a data reason):
the plan says "hold out splits.test_winter on the train station", but the train
station (cedarburg) has only two winters and splits.test_winter (2024-2025)
belongs to bismarck, which is the Phase-5 transfer station and must stay untouched.
So Phase 3 uses LEAVE-ONE-WINTER-OUT on cedarburg: for each held-out cedarburg
winter, train on the remaining cedarburg winter(s) and report its timing error.
Bismarck is never read here.

Usage:
  python -m fm_ice.models.train --encoder vjepa2            # LOO over cedarburg winters
  python -m fm_ice.models.train --encoder vjepa2 --test-winter 2023-2024
  python -m fm_ice.models.train --encoder dinov2 --head transformer
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fm_ice.config import load_yaml
from fm_ice.evaluation.metrics import timing_error_hours
from fm_ice.evaluation.probe_separability import load_probe_matrix
from fm_ice.models.events import min_run_steps_from_hours, read_events
from fm_ice.models.temporal_head import TemporalHead, total_loss


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def discover_winters(station: str) -> list[str]:
    """Winters with a built clip manifest for a station, chronological."""
    cfg = load_yaml("pipeline.yaml")
    interim = Path(cfg["paths"]["interim"]) / station
    winters = sorted(p.stem.replace("clips_", "") for p in interim.glob("clips_*.parquet"))
    if not winters:
        raise SystemExit(f"no clip manifests under {interim}")
    return winters


def load_sequence(encoder: str, station: str, winter: str, stem: str | None = None):
    """One winter as a time-ordered sequence.

    Returns (emb, air, y, times):
      emb   (T, D) float32 embeddings in chronological order
      air   (T,)   air-temperature-per-step (interpolated over short gaps)
      y     (T,)   int ice label (0/1)
      times (T,)   tz-aware UTC clip start times
    """
    X, df = load_probe_matrix(encoder, station, winter, stem=stem)
    order = np.argsort(pd.to_datetime(df["t_start_utc"], utc=True).to_numpy(), kind="stable")
    df = df.iloc[order].reset_index(drop=True)
    emb = X[order].astype("float32")
    times = pd.to_datetime(df["t_start_utc"], utc=True)

    # Air temperature: interpolate the handful of NA steps in time order, then
    # fall back to the series mean if a whole tail is missing.
    air = (df["air_tmpc_mean"].astype("float64")
           .interpolate(limit_direction="both"))
    air = air.fillna(air.mean()).to_numpy(dtype="float32")
    y = df["ice_flag"].astype(int).to_numpy()
    return emb, air, y, times


class FeatureScaler:
    """Standardize embeddings + air temp using TRAIN statistics only (no leakage)."""

    def __init__(self, use_air_temp: bool):
        self.use_air_temp = use_air_temp

    def fit(self, embs: list[np.ndarray], airs: list[np.ndarray]):
        E = np.concatenate(embs, axis=0)
        self.emb_mean = E.mean(0, keepdims=True)
        self.emb_std = E.std(0, keepdims=True) + 1e-6
        if self.use_air_temp:
            A = np.concatenate(airs, axis=0)
            self.air_mean = float(A.mean())
            self.air_std = float(A.std() + 1e-6)
        return self

    def transform(self, emb: np.ndarray, air: np.ndarray) -> np.ndarray:
        z = (emb - self.emb_mean) / self.emb_std
        if self.use_air_temp:
            za = ((air - self.air_mean) / self.air_std)[:, None]
            z = np.concatenate([z, za.astype("float32")], axis=1)
        return z.astype("float32")

    @property
    def out_dim(self) -> int:
        return self.emb_mean.shape[1] + (1 if self.use_air_temp else 0)


# --------------------------------------------------------------------------- #
# Train / predict
# --------------------------------------------------------------------------- #
def _seq_tensor(feat: np.ndarray) -> torch.Tensor:
    return torch.tensor(feat, dtype=torch.float32).unsqueeze(0)   # (1, T, D)


def fit_head(train_feats, train_labels, in_dim, th, epochs, kind, seed, verbose=True):
    """Train TemporalHead on a list of full-winter sequences."""
    torch.manual_seed(seed)
    model = TemporalHead(in_dim=in_dim, hidden=th["hidden"], layers=th["layers"],
                         dropout=th["dropout"], kind=kind)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    n_pos = sum(int(y.sum()) for y in train_labels)
    n_neg = sum(int((1 - y).sum()) for y in train_labels)
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)

    seqs = [(_seq_tensor(f), torch.tensor(y, dtype=torch.long).unsqueeze(0))
            for f, y in zip(train_feats, train_labels)]
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for x, y in seqs:
            logits = model(x)
            loss = total_loss(logits, y, w_smooth=th["smoothing_loss_weight"], pos_weight=pos_weight)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        if verbose and (ep == 0 or (ep + 1) % 10 == 0 or ep == epochs - 1):
            print(f"  epoch {ep + 1:3d}/{epochs}  loss={tot / len(seqs):.4f}")
    return model


@torch.no_grad()
def predict_head(model, feat: np.ndarray) -> np.ndarray:
    model.eval()
    logits = model(_seq_tensor(feat)).squeeze(0)
    return torch.sigmoid(logits).numpy()


def fit_perframe(train_feats, train_labels):
    """Per-frame logistic probe -- the 'per-frame thresholding' H1 anchor."""
    from sklearn.linear_model import LogisticRegression
    X = np.concatenate(train_feats, axis=0)
    y = np.concatenate(train_labels, axis=0)
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    clf.fit(X, y)
    return clf


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def load_references(station: str, winter: str, results_dir: Path) -> dict:
    """Reference onset/breakup per source for a station-winter, as Timestamps."""
    path = results_dir / "reference_events.csv"
    refs: dict[str, dict] = {}
    if not path.exists():
        return refs
    ref = pd.read_csv(path)
    sub = ref[(ref["station"] == station) & (ref["winter"] == winter)]
    for _, r in sub.iterrows():
        def _ts(v):
            return pd.Timestamp(v) if isinstance(v, str) and v.strip() else None
        refs[r["source"]] = {"onset": _ts(r["onset_utc"]), "breakup": _ts(r["breakup_utc"])}
    return refs


def evaluate_prediction(pred_events: dict, station: str, winter: str,
                        results_dir: Path) -> list[dict]:
    """Timing error of one prediction vs every available reference."""
    out = []
    for source, ref in load_references(station, winter, results_dir).items():
        err = timing_error_hours(pred_events, ref)
        out.append({"reference": source,
                    "onset_err_h": round(err["onset"], 1) if "onset" in err else None,
                    "breakup_err_h": round(err["breakup"], 1) if "breakup" in err else None,
                    "ref_onset": ref["onset"], "ref_breakup": ref["breakup"]})
    return out


def run_one(encoder: str, station: str, test_winter: str, train_winters: list[str],
            head: str, epochs: int, cfg: dict, results_dir: Path,
            temp_guard: bool = False, tag: str = "") -> list[dict]:
    th = cfg["temporal_head"]
    ev_cfg = cfg["events"]
    seed = cfg["seed"]
    np.random.seed(seed)
    threshold = ev_cfg["ice_state_threshold"]
    min_run = min_run_steps_from_hours(ev_cfg["min_event_separation_hours"],
                                       cfg["clip"]["stride_hours"])
    guard_c = ev_cfg.get("onset_freeze_guard_c", 0.0) if temp_guard else None

    print(f"\n=== {encoder} | test={station} {test_winter} | "
          f"train={[f'{station} {w}' for w in train_winters]} ===")

    # Build train + test feature matrices, scaled on train stats only.
    tr_emb, tr_air, tr_y = [], [], []
    for w in train_winters:
        e, a, y, _ = load_sequence(encoder, station, w)
        tr_emb.append(e); tr_air.append(a); tr_y.append(y)
    scaler = FeatureScaler(th["use_air_temp"]).fit(tr_emb, tr_air)
    tr_feats = [scaler.transform(e, a) for e, a in zip(tr_emb, tr_air)]

    te_emb, te_air, te_y, te_times = load_sequence(encoder, station, test_winter)
    te_feat = scaler.transform(te_emb, te_air)

    # --- temporal head ---
    model = fit_head(tr_feats, tr_y, scaler.out_dim, th, epochs, head, seed)
    prob_head = predict_head(model, te_feat)

    # --- per-frame probe (H1 anchor) ---
    clf = fit_perframe(tr_feats, tr_y)
    prob_pf = clf.predict_proba(te_feat)[:, 1]

    rows = []
    pred_dump = {"t_start_utc": te_times.astype(str), "ice_flag": te_y, "air_tmpc": te_air}
    for name, prob in (("temporal_head", prob_head), ("perframe_probe", prob_pf)):
        ev = read_events(te_times, prob, threshold=threshold, min_run_steps=min_run,
                         air=te_air, max_run_mean_air=guard_c)
        print(f"  [{name:14s}] onset={ev['onset']}  breakup={ev['breakup']}  "
              f"ice_steps={ev['n_ice_steps']}")
        for r in evaluate_prediction(ev, station, test_winter, results_dir):
            rows.append({"encoder": encoder, "model": name, "head": head,
                         "temp_guard": temp_guard,
                         "station": station, "test_winter": test_winter,
                         "pred_onset": ev["onset"], "pred_breakup": ev["breakup"],
                         **r})
        pred_dump[f"prob_{name}"] = prob
        pred_dump[f"state_{name}"] = ev["state"]

    # dump the per-step prediction for plotting / error analysis (Phase 6)
    dump = results_dir / f"phase3_pred_{encoder}_{station}_{test_winter}_{head}{tag}.csv"
    pd.DataFrame(pred_dump).to_csv(dump, index=False)
    return rows


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", default="vjepa2", choices=["vjepa2", "dinov2"])
    ap.add_argument("--train-station", default="cedarburg")
    ap.add_argument("--test-winter", default=None,
                    help="held-out cedarburg winter; default = leave-one-out over all")
    ap.add_argument("--head", default=None, choices=["tcn", "transformer"],
                    help="override temporal_head.type from the config")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--temp-guard", action="store_true",
                    help="apply the physical freezing guard (events.onset_freeze_guard_c)")
    ap.add_argument("--results", default="results")
    args = ap.parse_args()

    cfg = load_yaml("pipeline.yaml")
    head = args.head or cfg["temporal_head"]["type"]
    guard_suffix = "_guard" if args.temp_guard else ""
    file_tag = f"_{head}{guard_suffix}"          # disambiguates parallel runs
    results_dir = Path(args.results)
    results_dir.mkdir(parents=True, exist_ok=True)

    station = args.train_station
    winters = discover_winters(station)
    test_winters = [args.test_winter] if args.test_winter else winters

    all_rows = []
    for tw in test_winters:
        train_winters = [w for w in winters if w != tw]
        if not train_winters:
            print(f"[skip] {tw}: no other winter to train on")
            continue
        all_rows += run_one(args.encoder, station, tw, train_winters,
                            head, args.epochs, cfg, results_dir,
                            temp_guard=args.temp_guard, tag=guard_suffix)

    if not all_rows:
        raise SystemExit("no evaluations produced")
    df = pd.DataFrame(all_rows)
    out = results_dir / f"phase3_timing_{args.encoder}{file_tag}.csv"
    df.to_csv(out, index=False)
    _report(df)
    print(f"\n[phase3] wrote {out}")


def _report(df: pd.DataFrame) -> None:
    print("\n================ Phase 3 (H1): event-timing error (hours) ================")
    hdr = f"{'test-winter':14s} {'model':15s} {'reference':16s} {'onset_err':>9s} {'breakup_err':>11s}"
    print(hdr); print("-" * len(hdr))
    for _, r in df.iterrows():
        oe = f"{r['onset_err_h']:.1f}" if pd.notna(r["onset_err_h"]) else "  -"
        be = f"{r['breakup_err_h']:.1f}" if pd.notna(r["breakup_err_h"]) else "  -"
        print(f"{r['test_winter']:14s} {r['model']:15s} {r['reference']:16s} {oe:>9s} {be:>11s}")
    print("\nLower is better. 'perframe_probe' is the per-frame-thresholding H1 anchor;")
    print("'temporal_head' should beat it. RIce-Net timing drops into this table once run.")


if __name__ == "__main__":
    main()
