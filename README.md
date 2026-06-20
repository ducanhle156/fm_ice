# FM_ice

Temporal foundation-model detection of river-ice onset and breakup from USGS
ground cameras. The method replaces RIce-Net's per-frame coverage threshold with
a light temporal head over frozen V-JEPA embeddings, and detects onset and breakup
as events in time. The headline result is the reduction in event-timing error
against the threshold baseline.

## Start here
- DATA.md — what to download, from where, how much (stations, cameras, volumes).
- IMPLEMENTATION_PLAN.md — the 7-phase build, with decision gates and acceptance checks.
- CLAUDE.md — setup, commands, CCAST notes, and the git / clone-to-HPC runbook.

## Quick start
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
python -m fm_ice.evaluation.metrics                 # self-test, no data needed
python -m fm_ice.data.download_images --station bismarck --winter 2025-2026 --dry-run
```

## Layout
```
configs/      stations.yaml (verified), pipeline.yaml
src/fm_ice/
  data/       download_images, download_stage, download_temperature,
              download_ice_flags, assemble_clips
  features/   extract_embeddings (V-JEPA 2 / DINOv2, frozen)
  baselines/  ricenet_baseline (threshold anchor), changepoint (BOCPD, BEAST)
  models/     temporal_head (TCN / transformer), train
  evaluation/ metrics (timing error, event F1, covering)
scripts/      download_all.sh, extract_embeddings_ccast.sbatch
```

## Status
Data download layer and evaluation metrics are implemented and tested offline.
The encoder, temporal head, and parts of assembly/training are scaffolded with
`TODO` markers; see IMPLEMENTATION_PLAN.md for what each phase must complete.
