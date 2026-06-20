# CLAUDE.md — guide for Claude Code working in this repo

This repo detects river-ice onset and breakup with a temporal model over frozen
video foundation-model embeddings. Read IMPLEMENTATION_PLAN.md for the phases and
DATA.md for the data. This file is the operating manual.

## What this project is
- Contribution is a measured result, not an architecture: beat the RIce-Net
  threshold baseline on event timing error. The number is the headline.
- Encoder is frozen V-JEPA (no fine-tuning). Only a light temporal head trains.
- Real goal: evidence of CV-scientist skill. Reproducibility and a clean ablation
  matter more than a fancy venue.

## Repo conventions
- Package lives in `src/fm_ice`. Install once with `pip install -e .`, then run
  modules as `python -m fm_ice.<area>.<module>`.
- Config over flags: stations and pipeline settings live in `configs/*.yaml`.
  Do not hardcode station IDs, paths, or thresholds in code.
- Data is never committed. `data/`, `results/`, `*.npy`, `*.pt` are gitignored.
  Cache embeddings, not images.
- Times are UTC everywhere. Convert only for display.
- Keep the encoder the only GPU dependency. Everything downstream runs on CPU.

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
python -m fm_ice.evaluation.metrics   # self-test, no data or network needed
```

## First run (smoke-test the downloaders on a networked machine)
These scripts were built against verified API specs but not run in the authoring
environment. Verify once before a full pull:
```bash
python -m fm_ice.data.download_stage       --station bismarck --winter 2025-2026
python -m fm_ice.data.download_temperature  --station bismarck --winter 2025-2026
python -m fm_ice.data.download_images       --station bismarck --winter 2025-2026 --dry-run
```
Expect: a stage CSV with rows, a temperature CSV with rows, and a printed image
count with an MB estimate. If images 403 with OVER_RATE_LIMIT, set `USGS_API_KEY`
(https://api.waterdata.usgs.gov/signup/).

## Key commands
```bash
bash scripts/download_all.sh                          # all streams, both stations
python -m fm_ice.data.assemble_clips --station cedarburg --winter 2024-2025
sbatch scripts/extract_embeddings_ccast.sbatch        # GPU only, on CCAST
python -m fm_ice.models.train --encoder vjepa2 --test-winter 2024-2025
```

## CCAST / HPC notes
- Edit `scripts/extract_embeddings_ccast.sbatch`: partition name, `module load`
  lines, and the activate command must match the cluster. The placeholders are
  guesses.
- Embedding extraction is the only job that needs the GPU. Run it there, cache the
  `.npy` files, copy them back, and train the head locally.
- V-JEPA 2 loads from Hugging Face (`facebook/vjepa2-vitl-fpc64-256`). The newer
  V-JEPA 2.1 (released 2026-03) is torch.hub only and not in `transformers` yet; do
  not assume the HF API exposes it.
- If CCAST compute nodes have no internet, pre-download the checkpoint on a login
  node (`huggingface-cli download facebook/vjepa2-vitl-fpc64-256`) and set
  `HF_HOME` to a shared path.

## Guardrails for the agent
- Do not invent station IDs, gauge numbers, or API parameters. They are verified in
  configs/stations.yaml and DATA.md. If something is missing, query the HIVIS API
  (`/cameras?siteId=...`) rather than guessing.
- Do not change the clip definition after embeddings are cached. It invalidates them.
- A negative H2 result (DINOv2 ties or beats V-JEPA) is a valid outcome. Report it.
- Stubs are marked with `TODO(...)`. Implement them where the plan says, do not
  silently stub over a phase.

## Git and pushing to GitHub, then cloning to CCAST
```bash
# From the repo root (this folder):
git init
git add .
git commit -m "FM_ice scaffold: data download, embeddings, baselines, plan"
git branch -M main
git remote add origin git@github.com:<you>/fm_ice.git   # create the empty repo first
git push -u origin main

# On CCAST:
git clone https://github.com/<you>/fm_ice.git
cd fm_ice
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```
Notes before pushing:
- `.gitignore` already excludes data, caches, checkpoints, and the loose sample
  images in this folder. Verify with `git status` that no `.jpg`, `.png`, `.npy`,
  or `.pt` is staged.
- Never commit an API key. Use the `USGS_API_KEY` env var.
- The legacy `SAM2_Model1.ipynb` and the `test*.png` / `WI_Milwaukee_*.jpg` samples
  are exploratory. They are gitignored; move the notebook to
  `notebooks/exploratory/` if you want it tracked.
