"""Consolidate the Phase 5 transfer runs into one generalization table.

Reads every results/phase5_transfer_*.csv produced by fm_ice.models.transfer (one
per encoder x head x guard) and prints / writes a single table of the bismarck
held-out-STATION timing error next to the cedarburg leave-one-winter-out anchor
and the transfer gap, against the solid USGS ice-flag reference. This is the
Phase 5 generalization headline -- the test of whether the temporal-head result
holds on a station the model never trained on.

Mirrors fm_ice.evaluation.phase3_report (which consolidates the Phase-3 LOO runs).

Usage:
  python -m fm_ice.evaluation.phase5_report
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REF = "usgs_ice_flag"   # the solid reference; the cedarburg anchor is computed vs the same


def load_all(results_dir: Path) -> pd.DataFrame:
    files = sorted(results_dir.glob("phase5_transfer_*.csv"))
    if not files:
        raise SystemExit(
            f"no phase5_transfer_*.csv in {results_dir} -- run fm_ice.models.transfer first")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return df[(df["reference"] == REF) & (df["model"] == "temporal_head")].copy()


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """One row per config: bismarck onset/breakup error, cedarburg LOO mean, gap."""
    def _cfg(r) -> str:
        guard = " +guard" if str(r["temp_guard"]) == "True" else ""
        return f"{r['encoder']} / {r['head']} / temporal-head{guard}"

    df = df.copy()
    df["config"] = df.apply(_cfg, axis=1)
    cols = ["config", "onset_err_h", "breakup_err_h", "cedarburg_loo_mean", "transfer_gap_h"]
    out = df[cols].rename(columns={
        "onset_err_h": "bismarck onset",
        "breakup_err_h": "bismarck breakup",
        "cedarburg_loo_mean": "cedarburg LOO mean",
        "transfer_gap_h": "transfer gap",
    })
    out["bismarck mean"] = out[["bismarck onset", "bismarck breakup"]].mean(axis=1).round(1)
    out = out[["config", "bismarck onset", "bismarck breakup", "bismarck mean",
               "cedarburg LOO mean", "transfer gap"]]
    return out.sort_values("bismarck mean").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    results_dir = Path(args.results)

    df = load_all(results_dir)
    table = summarize(df)
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print(f"\n===== Phase 5 (transfer): bismarck timing error vs cedarburg LOO, hours (vs {REF}) =====")
    print("(head trained on BOTH cedarburg winters, evaluated on held-out bismarck; lower is better)\n")
    print(table.to_string(index=False))
    print("\ntransfer gap = bismarck - cedarburg LOO mean (positive = worse on the unseen station).")
    out = results_dir / "phase5_summary.csv"
    table.to_csv(out, index=False)
    print(f"\n[phase5] wrote {out}")


if __name__ == "__main__":
    main()
