"""Consolidate the Phase 3 timing runs into one H1 table.

Reads every results/phase3_timing_*.csv produced by fm_ice.models.train (one per
encoder x head x guard configuration) and prints / writes a single comparison of
onset and breakup timing error, per held-out winter, against the solid USGS
ice-flag reference. This is the H1 headline figure for the temporal head.

Usage:
  python -m fm_ice.evaluation.phase3_report
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REF = "usgs_ice_flag"   # the solid reference; stage_breakpoint is a rough cross-check


def load_all(results_dir: Path) -> pd.DataFrame:
    files = sorted(results_dir.glob("phase3_timing_*.csv"))
    if not files:
        raise SystemExit(f"no phase3_timing_*.csv in {results_dir} -- run fm_ice.models.train first")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if "temp_guard" not in df:
        df["temp_guard"] = False
    return df[df["reference"] == REF].copy()


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    df["config"] = (df["encoder"] + " / " + df["head"] + " / "
                    + df["model"].str.replace("_", "-")
                    + df["temp_guard"].map({True: " +guard", False: ""}))
    # one row per config, columns = onset/breakup error per winter + the mean.
    rows = []
    for cfg, g in df.groupby("config", sort=False):
        row = {"config": cfg}
        errs = []
        for _, r in g.iterrows():
            w = r["test_winter"]
            row[f"{w} onset"] = r["onset_err_h"]
            row[f"{w} breakup"] = r["breakup_err_h"]
            errs += [r["onset_err_h"], r["breakup_err_h"]]
        row["mean_err_h"] = round(pd.Series(errs).mean(), 1)
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("mean_err_h").reset_index(drop=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    results_dir = Path(args.results)

    df = load_all(results_dir)
    table = summarize(df)
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print(f"\n========= Phase 3 (H1) onset/breakup timing error in hours vs {REF} =========")
    print("(leave-one-winter-out on cedarburg; lower is better; sorted by mean)\n")
    print(table.to_string(index=False))
    print("\nper-frame-probe rows are the per-frame-thresholding H1 anchor;")
    print("temporal-head should beat its matched per-frame probe. +guard = physical freezing guard.")
    out = results_dir / "phase3_summary.csv"
    table.to_csv(out, index=False)
    print(f"\n[phase3] wrote {out}")


if __name__ == "__main__":
    main()
