"""Phase 6 error analysis: WHERE does the per-frame ice call fail?

The timing numbers (evaluate.py) say how well we hit onset/breakup; this says which
imaging regimes drive the per-frame mistakes, so the failure stories in the writeup
("misses ice at night / under glare / in low-quality clips") are backed by counts
and example frames rather than vibes.

For every `results/phase3_pred_<encoder>_<station>_<winter>_<head>[_guard].csv` we
join the per-step rows to the clip manifest QC columns by t_start_utc (both UTC),
flag each step as correct/incorrect via `state_temporal_head != ice_flag`, then
stratify the error rate by three regimes:

  night   night_frac > NIGHT_FRAC_THRESH        vs day
  glare   glare_frac > GLARE_FRAC_THRESH        vs none
  quality valid_fraction < per-file median      vs high (low-quality clips)

Each regime row carries n, n_error, error_rate, and the delta vs its complement so
the dominant driver is obvious. We also list the worst-error example clips per
regime with their frame_paths for the figure / appendix.

Output: results/phase6_error_analysis.csv (the regime table) and a printed summary
naming the regime with the largest error gap. CPU-only, cached CSVs only.

Usage:
  python -m fm_ice.evaluation.error_analysis                  # all pred files
  python -m fm_ice.evaluation.error_analysis --glob '*tcn.csv'
  python -m fm_ice.evaluation.error_analysis --examples 3     # examples per regime
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fm_ice.config import load_yaml
from fm_ice.evaluation.evaluate import parse_pred_name
from fm_ice.evaluation.probe_separability import GLARE_FRAC_THRESH, NIGHT_FRAC_THRESH

# Which per-step state column is "the model" for error analysis (the headline head).
STATE_COL = "state_temporal_head"
QC_COLS = ["night_frac", "glare_frac", "occluded_frac", "valid_fraction",
           "brightness_mean", "frame_paths"]


def _paths():
    return load_yaml("pipeline.yaml")["paths"]


def _load_joined(path: Path, interim: Path) -> tuple[pd.DataFrame, dict]:
    """Pred frame joined to manifest QC on t_start_utc. Returns (df, meta)."""
    meta = parse_pred_name(path)
    pred = pd.read_csv(path)
    pred["t_start_utc"] = pd.to_datetime(pred["t_start_utc"], utc=True)

    man = pd.read_parquet(interim / meta["station"] / f"clips_{meta['winter']}.parquet")
    man = man[["t_start_utc"] + [c for c in QC_COLS if c in man.columns]].copy()
    man["t_start_utc"] = pd.to_datetime(man["t_start_utc"], utc=True)

    df = pred.merge(man, on="t_start_utc", how="left", validate="m:1")
    df["error"] = (df[STATE_COL].astype("Int64") != df["ice_flag"].astype("Int64"))
    # Drop steps with no truth label or no QC join (can't attribute those).
    df = df[df["ice_flag"].notna() & df["night_frac"].notna()].reset_index(drop=True)
    return df, meta


def _regime_masks(df: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """{regime: (mask, complement_mask)} over the joined frame."""
    night = df["night_frac"].to_numpy() > NIGHT_FRAC_THRESH
    glare = df["glare_frac"].to_numpy() > GLARE_FRAC_THRESH
    vf = df["valid_fraction"].to_numpy()
    med = np.nanmedian(vf) if len(vf) else np.nan
    low = vf < med
    return {
        "night": (night, ~night),
        "glare": (glare, ~glare),
        "low_quality": (low, ~low),
    }


def _rate(err: pd.Series, mask: np.ndarray) -> tuple[int, int, float]:
    e = err[mask]
    n = int(len(e))
    n_err = int(e.sum())
    return n, n_err, (n_err / n if n else float("nan"))


def analyze_file(path: Path, interim: Path, n_examples: int) -> tuple[list[dict], list[dict]]:
    df, meta = _load_joined(path, interim)
    if df.empty:
        return [], []
    err = df["error"]
    overall_n, overall_err, overall_rate = _rate(err, np.ones(len(df), bool))
    masks = _regime_masks(df)

    rows, examples = [], []
    for regime, (m, mc) in masks.items():
        n, ne, rate = _rate(err, m)
        cn, cne, crate = _rate(err, mc)
        rows.append({
            **meta, "regime": regime,
            "n": n, "n_error": ne, "error_rate": round(rate, 4),
            "complement_n": cn, "complement_error_rate": round(crate, 4),
            "error_rate_delta": round(rate - crate, 4) if not (np.isnan(rate) or np.isnan(crate)) else float("nan"),
            "overall_error_rate": round(overall_rate, 4),
        })
        # worst example error clips IN this regime (most-confident wrong calls first
        # if a probability column is present, else just the erroring rows).
        sub = df[m & err]
        if "prob_temporal_head" in sub.columns:
            # distance of the ice probability from the truth: bigger = worse miss.
            conf = (sub["prob_temporal_head"] - sub["ice_flag"].astype(float)).abs()
            sub = sub.assign(_severity=conf).sort_values("_severity", ascending=False)
        for _, r in sub.head(n_examples).iterrows():
            fp = r.get("frame_paths")
            first_frame = fp[0] if isinstance(fp, (list, np.ndarray)) and len(fp) else ""
            examples.append({
                **meta, "regime": regime,
                "t_start_utc": r["t_start_utc"], "ice_flag": int(r["ice_flag"]),
                "state": int(r[STATE_COL]),
                "prob": round(float(r.get("prob_temporal_head", float("nan"))), 4),
                "night_frac": round(float(r["night_frac"]), 3),
                "glare_frac": round(float(r["glare_frac"]), 3),
                "valid_fraction": round(float(r["valid_fraction"]), 3),
                "first_frame": first_frame,
            })
    return rows, examples


def run(results_dir: Path, glob: str, encoder: str | None, n_examples: int) -> pd.DataFrame:
    interim = Path(_paths()["interim"])
    files = sorted(results_dir.glob(f"phase3_pred_{glob}"))
    if encoder:
        files = [f for f in files if f"_{encoder}_" in f.name]
    if not files:
        raise SystemExit(f"no pred files matching phase3_pred_{glob} in {results_dir}")

    all_rows, all_ex = [], []
    for f in files:
        try:
            rows, ex = analyze_file(f, interim, n_examples)
            all_rows.extend(rows); all_ex.extend(ex)
        except Exception as e:  # noqa: BLE001
            print(f"[error_analysis] WARN {f.name}: {e}")

    out = pd.DataFrame(all_rows)
    dest = results_dir / "phase6_error_analysis.csv"
    out.to_csv(dest, index=False)
    if all_ex:
        ex_dest = results_dir / "phase6_error_examples.csv"
        pd.DataFrame(all_ex).to_csv(ex_dest, index=False)
        print(f"[error_analysis] wrote {len(all_ex)} example clips -> {ex_dest}")
    _report(out)
    print(f"\n[error_analysis] wrote {len(out)} regime rows -> {dest}")
    return out


def _report(df: pd.DataFrame) -> None:
    if df.empty:
        print("[error_analysis] no rows produced.")
        return
    print("\n=============== PHASE 6 error analysis (per-frame, temporal head) ===============")
    hdr = (f"{'enc':7s} {'station':9s} {'winter':9s} {'head':12s} {'g':1s} "
           f"{'regime':12s} {'n':>5s} {'err%':>6s} {'compl%':>6s} {'delta':>6s}")
    print(hdr); print("-" * len(hdr))
    for _, r in df.iterrows():
        print(f"{r['encoder']:7s} {r['station']:9s} {r['winter']:9s} {r['head']:12s} "
              f"{'Y' if r['guard'] else 'n':1s} {r['regime']:12s} {r['n']:5d} "
              f"{100*r['error_rate']:6.1f} {100*r['complement_error_rate']:6.1f} "
              f"{100*r['error_rate_delta']:+6.1f}")
    # Which regime drives errors overall (largest mean positive delta across files)?
    g = df.groupby("regime")["error_rate_delta"].mean().sort_values(ascending=False)
    print("\nMean error-rate delta vs complement, by regime (positive = regime is worse):")
    for regime, d in g.items():
        print(f"  {regime:12s} {100*d:+6.1f} pp")
    top = g.index[0]
    print(f"=> '{top}' drives the most per-frame error (+{100*g.iloc[0]:.1f} pp on average).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--glob", default="*.csv",
                    help="suffix glob after phase3_pred_ (e.g. '*tcn.csv')")
    ap.add_argument("--encoder", default=None, help="filter to one encoder")
    ap.add_argument("--examples", type=int, default=3,
                    help="worst-error example clips to list per regime")
    ap.add_argument("--selftest", action="store_true",
                    help="run the cheap masking self-test and exit")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return
    run(Path(args.results), args.glob, args.encoder, args.examples)


def _selftest() -> None:
    """Cheap self-test: regime masking + rate math on a synthetic frame (no data)."""
    df = pd.DataFrame({
        "night_frac": [0.9, 0.9, 0.1, 0.1],
        "glare_frac": [0.0, 0.2, 0.0, 0.0],
        "valid_fraction": [0.2, 0.2, 0.8, 0.8],
        "error": [True, True, False, False],
    })
    masks = _regime_masks(df)
    n, ne, rate = _rate(df["error"], masks["night"][0])
    assert n == 2 and ne == 2 and rate == 1.0
    n2, ne2, rate2 = _rate(df["error"], masks["night"][1])
    assert n2 == 2 and ne2 == 0 and rate2 == 0.0
    print("error_analysis self-test OK")


if __name__ == "__main__":
    main()
