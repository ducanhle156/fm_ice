#!/usr/bin/env bash
# Download every data stream for both stations, all available winters.
# Run from the repo root after `pip install -e .`.
# Use STEP 0 (dry-run) first to see image counts and volume before pulling GBs.
set -euo pipefail

echo "== STEP 0: dry-run image counts (no download) =="
python -m fm_ice.data.download_images --station cedarburg --all-winters --dry-run
python -m fm_ice.data.download_images --station bismarck  --all-winters --dry-run

read -r -p "Proceed to full download? [y/N] " ok
[[ "${ok:-N}" == "y" ]] || { echo "Stopped after dry-run."; exit 0; }

echo "== STEP 1: stage (NWIS IV) =="
python -m fm_ice.data.download_stage --station cedarburg --all-winters
python -m fm_ice.data.download_stage --station bismarck  --all-winters

echo "== STEP 2: air temperature (IEM ASOS) =="
python -m fm_ice.data.download_temperature --station cedarburg --all-winters
python -m fm_ice.data.download_temperature --station bismarck  --all-winters

echo "== STEP 3: USGS ice flags (NWIS DV qualifiers) =="
python -m fm_ice.data.download_ice_flags --station cedarburg --all-winters
python -m fm_ice.data.download_ice_flags --station bismarck  --all-winters

echo "== STEP 4: camera images (the big one; use --size small to cut volume) =="
python -m fm_ice.data.download_images --station cedarburg --all-winters --size overlay
python -m fm_ice.data.download_images --station bismarck  --all-winters --size overlay

echo "== STEP 5: assemble clips per station/winter =="
for w in 2022-2023 2023-2024 2024-2025 2025-2026; do
  python -m fm_ice.data.assemble_clips --station cedarburg --winter "$w" || true
done
for w in 2024-2025 2025-2026; do
  python -m fm_ice.data.assemble_clips --station bismarck --winter "$w" || true
done

echo "Done. Raw data in data/raw, clip manifests in data/interim."
