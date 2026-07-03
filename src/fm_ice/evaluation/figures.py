"""Phase 6 figures, regenerated entirely from this script (no hand editing).

Two figures, both saved as PNG under results/figures/ (created on demand):

  h3        UMAP of the frozen embeddings (load_probe_matrix) for one
            station-winter, two panels: left coloured by ice_flag (does the
            representation separate ice from open water?), right coloured by
            night_frac (is that structure just the day/night lighting axis?). This
            is the visual companion to the GATE-A linear probe. The pooled
            linear-probe AUC is added to the title when available from
            results/gate_a_<encoder>.csv.

  timeline  Per held-out winter event timeline from a phase3_pred_*.csv:
            prob_temporal_head over time, the binarized state shaded, and vertical
            lines for the reference onset/breakup (reference_events.csv) and the
            predicted onset/breakup (read_events). Shows at a glance whether the
            head fires at the right time.

matplotlib uses the Agg backend (headless). umap-learn is imported lazily so the
timeline path works on a box without umap; both guarded imports print a clear
install hint if missing. CPU-only.

Usage:
  python -m fm_ice.evaluation.figures --figure h3 --encoder vjepa2 \
         --station cedarburg --winter 2022-2023
  python -m fm_ice.evaluation.figures --figure timeline \
         --pred results/phase3_pred_vjepa2_cedarburg_2023-2024_tcn.csv
  python -m fm_ice.evaluation.figures --figure all --encoder vjepa2   # h3 for all winters
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fm_ice.config import load_yaml
from fm_ice.evaluation.evaluate import parse_pred_name
from fm_ice.evaluation.probe_separability import (
    NIGHT_FRAC_THRESH,
    PAPER_WINTERS,
    load_probe_matrix,
)

FIG_DIR_NAME = "figures"


def _mpl():
    """Import matplotlib with the Agg backend, or a clear install hint."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError as e:  # pragma: no cover
        raise SystemExit(f"matplotlib required for figures ({e}); pip install matplotlib")


def _umap_embed(X: np.ndarray, seed: int) -> np.ndarray:
    try:
        import umap
    except ImportError as e:  # pragma: no cover
        raise SystemExit(f"umap-learn required for the h3 figure ({e}); pip install umap-learn")
    n_neighbors = int(min(15, max(2, len(X) - 1)))
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.1,
                        metric="euclidean", random_state=seed)
    return reducer.fit_transform(X)


def _gate_a_auc(encoder: str, station: str, winter: str, results_dir: Path) -> float | None:
    """Pooled linear-probe AUC from results/gate_a_<encoder>.csv, if present."""
    f = results_dir / f"gate_a_{encoder}.csv"
    if not f.exists():
        return None
    g = pd.read_csv(f)
    row = g[(g["station"] == station) & (g["winter"] == winter)]
    if row.empty or "pooled_auc" not in row or pd.isna(row["pooled_auc"].iloc[0]):
        return None
    return float(row["pooled_auc"].iloc[0])


def fig_h3_umap(encoder: str, station: str, winter: str, results_dir: Path,
                fig_dir: Path, seed: int = 1337) -> Path:
    plt = _mpl()
    X, df = load_probe_matrix(encoder, station, winter)
    emb = _umap_embed(X, seed)
    ice = df["ice_flag"].astype(int).to_numpy()
    night = df["night_frac"].fillna(0).to_numpy()
    auc = _gate_a_auc(encoder, station, winter, results_dir)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    # panel 1: ice vs open water
    for val, color, lab in [(0, "#1f77b4", "open water"), (1, "#d62728", "ice")]:
        m = ice == val
        axes[0].scatter(emb[m, 0], emb[m, 1], s=8, c=color, alpha=0.6, label=lab)
    axes[0].legend(loc="best", framealpha=0.9)
    t0 = f"{encoder} {station} {winter}: embeddings by ice_flag"
    if auc is not None:
        t0 += f"  (pooled probe AUC={auc:.3f})"
    axes[0].set_title(t0)

    # panel 2: lighting axis
    sc = axes[1].scatter(emb[:, 0], emb[:, 1], s=8, c=night, cmap="viridis",
                         alpha=0.7, vmin=0, vmax=1)
    fig.colorbar(sc, ax=axes[1], label="night_frac")
    axes[1].set_title(f"same embeddings by night_frac (>{NIGHT_FRAC_THRESH} = night)")
    for ax in axes:
        ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")

    fig.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / f"h3_umap_{encoder}_{station}_{winter}.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"[figures] wrote {out}")
    return out


def _ref_lines(results_dir: Path, station: str, winter: str) -> dict[str, dict]:
    f = results_dir / "reference_events.csv"
    if not f.exists():
        return {}
    g = pd.read_csv(f)
    sub = g[(g["station"] == station) & (g["winter"] == winter)]
    out = {}
    for _, r in sub.iterrows():
        out[r["source"]] = {
            "onset": pd.to_datetime(r["onset_utc"], utc=True) if pd.notna(r["onset_utc"]) else None,
            "breakup": pd.to_datetime(r["breakup_utc"], utc=True) if pd.notna(r["breakup_utc"]) else None,
        }
    return out


def fig_event_timeline(pred_csv: Path, results_dir: Path, fig_dir: Path) -> Path:
    plt = _mpl()
    from fm_ice.evaluation.evaluate import _events_cfg  # reuse event-rule config
    from fm_ice.models.events import min_run_steps_from_hours, read_events

    meta = parse_pred_name(pred_csv)
    df = pd.read_csv(pred_csv)
    df["t_start_utc"] = pd.to_datetime(df["t_start_utc"], utc=True)
    df = df.sort_values("t_start_utc").reset_index(drop=True)
    t = df["t_start_utc"]

    cfg = load_yaml("pipeline.yaml")
    ev = cfg.get("events", {})
    threshold = ev.get("ice_state_threshold", 0.5)
    stride = cfg["clip"]["stride_hours"]
    min_run_steps = min_run_steps_from_hours(8.0, stride)
    air = df["air_tmpc"].to_numpy() if meta["guard"] and "air_tmpc" in df else None
    max_air = ev.get("onset_freeze_guard_c", 0.0) if meta["guard"] else None
    pred = read_events(t.to_numpy(), df["prob_temporal_head"].to_numpy(),
                       threshold=threshold, min_run_steps=min_run_steps,
                       air=air, max_run_mean_air=max_air)

    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(t, df["prob_temporal_head"], lw=1.2, color="#1f77b4",
            label="prob_temporal_head")
    ax.fill_between(t, 0, df["state_temporal_head"], step="mid", alpha=0.15,
                    color="#1f77b4", label="state (binarized)")
    if "ice_flag" in df:
        ax.plot(t, df["ice_flag"], lw=0.8, color="0.4", alpha=0.6,
                drawstyle="steps-mid", label="ice_flag (truth)")
    ax.axhline(threshold, color="0.7", ls=":", lw=0.8)

    refs = _ref_lines(results_dir, meta["station"], meta["winter"])
    ref_styles = {"usgs_ice_flag": "#2ca02c", "stage_breakpoint": "#9467bd"}
    for src, ev_d in refs.items():
        c = ref_styles.get(src, "0.3")
        for kind in ("onset", "breakup"):
            if ev_d[kind] is not None:
                ax.axvline(ev_d[kind], color=c, ls="--", lw=1.3, alpha=0.9,
                           label=f"ref {src} {kind}")
    for kind, color in (("onset", "#d62728"), ("breakup", "#ff7f0e")):
        if pred[kind] is not None:
            ax.axvline(pred[kind], color=color, ls="-", lw=1.8, alpha=0.9,
                       label=f"pred {kind}")

    ax.set_ylim(-0.05, 1.1)
    ax.set_xlabel("t_start_utc"); ax.set_ylabel("ice probability / state")
    guard = " +guard" if meta["guard"] else ""
    ax.set_title(f"{meta['encoder']} {meta['station']} {meta['winter']} "
                 f"({meta['head']}{guard}) event timeline")
    ax.legend(loc="upper center", ncol=4, fontsize=7, framealpha=0.9)
    fig.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / f"timeline_{pred_csv.stem.replace('phase3_pred_', '')}.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"[figures] wrote {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figure", choices=["h3", "timeline", "all"], required=True)
    ap.add_argument("--encoder", default="vjepa2", choices=["vjepa2", "dinov2"])
    ap.add_argument("--station")
    ap.add_argument("--winter")
    ap.add_argument("--pred", help="phase3_pred_*.csv for the timeline figure")
    ap.add_argument("--results", default="results")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    results_dir = Path(args.results)
    fig_dir = results_dir / FIG_DIR_NAME

    if args.figure == "timeline":
        if not args.pred:
            raise SystemExit("--figure timeline needs --pred <phase3_pred_*.csv>")
        fig_event_timeline(Path(args.pred), results_dir, fig_dir)
    elif args.figure == "h3":
        if not (args.station and args.winter):
            raise SystemExit("--figure h3 needs --station and --winter")
        fig_h3_umap(args.encoder, args.station, args.winter, results_dir, fig_dir, args.seed)
    else:  # all -> one h3 panel per paper winter for the chosen encoder
        for station, winter in PAPER_WINTERS:
            try:
                fig_h3_umap(args.encoder, station, winter, results_dir, fig_dir, args.seed)
            except SystemExit as e:
                print(f"[figures] skip {station} {winter}: {e}")


if __name__ == "__main__":
    main()
