# STATUS.md — working status of FM_ice

Last updated: 2026-06-20. Companion to IMPLEMENTATION_PLAN.md (the plan) and
DATA.md (the data spec). This file tracks *what is actually done* right now.

**Phase 1 (Data assembly) is essentially complete.** Downloads, clip assembly,
per-clip labels, per-clip QC, and reference events are done and verified. The
only open item is *running* the RIce-Net baseline (Acceptance #2), which needs an
external model + weights — see "Open decision" below.

---

## Environment

- **conda env `fm-ice`** (Python 3.12). Activate: `conda activate fm-ice`.
  Recreate: `conda create -y -n fm-ice python=3.12 && conda activate fm-ice && pip install -e .`
- Installed and used downstream on CPU: numpy, pandas, pyarrow, scipy, sklearn,
  **opencv (cv2)**, Pillow, requests, pyyaml.
- NOT installed (only needed to *run* the RIce-Net baseline): `torch`,
  `segmentation-models-pytorch`. The V-JEPA encoder (Phase 2) and RIce-Net
  inference are the only torch users; both can run on CCAST.
- System python is 3.9 (too old). Always use the conda env.
- This machine HAS outbound network, so downloads run here.
- **conda/env path:** `/mmfs1/projects/chau.le/Computer_Vision_FM/miniconda3` (env at
  `.../envs/fm-ice`). Activate in batch jobs via that miniconda's
  `etc/profile.d/conda.sh` then `conda activate fm-ice`.
- **GPU stack (for Phase 2):** torch 2.12.1+cu130, transformers 5.12.1 installed
  in fm-ice, plus numpy/pandas/pyarrow/pillow/scipy/scikit-learn. V-JEPA 2
  checkpoint cached under `data/external/hf_home`.
- **`~/.local` LEAK — set `PYTHONNOUSERSITE=1` for any torch run.** The user-site
  `~/.local` holds a stale `nvidia-nccl-cu12` that shadows fm-ice's nccl and makes
  torch fail to import (`undefined symbol: ncclCommResume`). fm-ice is now
  self-contained for the embedding path; isolate it with `PYTHONNOUSERSITE=1`
  (baked into `job.pbs`). CPU-only data steps work either way.

---

## Data scope (the "paper winters")

| Station | Winters in use | Role |
|---|---|---|
| cedarburg (04086600) | 2022-2023, 2023-2024 | train |
| bismarck (06342500) | 2024-2025 | transfer test |

Other winters' CSV streams are also on disk; their images were not pulled.

---

## Download status — COMPLETE

All four streams. Image size = **overlay** (full 1920×1080). Under `data/raw/`.

| Stream | cedarburg 2022-2023 | cedarburg 2023-2024 | bismarck 2024-2025 |
|---|---|---|---|
| Stage (NWIS IV, 00065+00060) | ✅ | ✅ | ✅ |
| Temperature (IEM ASOS) | ✅ | ✅ | ✅ |
| Ice flags (NWIS DV) | ✅ 48 ice days | ✅ 26 ice days | ✅ 97 ice days |
| Images (overlay, hourly) | ✅ | ✅ | ✅ |

---

## Clip assembly — COMPLETE & VERIFIED

`data/interim/<station>/clips_<winter>.parquet`. Clip def (configs/pipeline.yaml):
16 hourly frames, 4 h stride, ≥12 real frames.

| Station-winter | clips | ice-labeled | mean valid_fraction |
|---|---|---|---|
| cedarburg 2022-2023 | 1099 | 261 (24%) | 0.44 |
| cedarburg 2023-2024 | 1173 | 156 (13%) | 0.43 |
| bismarck 2024-2025 | 1130 | 582 (52%) | 0.44 |

Columns: clip_id, t_start_utc, t_end_utc, n_frames, max_gap_hours, frame_paths,
station, cam_id, air_tmpc_mean, stage_mean, **ice_flag** (Int64 label),
ice_day_frac, ice_explicit, ice_estimated, n_usable, valid_fraction, night_frac,
glare_frac, occluded_frac, brightness_mean.

**Acceptance #1 (no intra-clip gaps beyond min_valid): PASS.** Every clip has
12–16 real frames, max_gap_hours ≤ 5, valid_fraction ≤ 1.

Validation cross-checks (all pass): ice-labeled clips are colder (e.g. cedarburg
2022-2023 −3.9 °C vs +4.6 °C; bismarck −8.2 vs +5.1), stage is higher under ice
(bismarck 8.75 vs 4.90 ft), night frames are darker than day.

---

## Reference events — DONE

`results/reference_events.csv` — onset/breakup per station-winter from two refs:

- **usgs_ice_flag** (solid): first sustained ice run → onset; day after last ice
  day → breakup. cedarburg 2022-2023 Dec 19 → Feb 27; 2023-2024 Nov 27 → Feb 2;
  bismarck Dec 10 → Mar 17. Physically sensible.
- **stage_breakpoint** (rough cross-check, label-free, via the project's BOCPD):
  crude — at cedarburg 2022-2023 it picks the late-Feb melt freshet as "onset".
  Treat as a sanity reference only; proper change-point tuning is Phase 4.

---

## Code: what changed this session

Data plumbing fixes (earlier): `download_ice_flags` (estimated-`e` ice encoding),
`assemble_clips` (winter filter + UTC tz), `http_util.get` (no 4xx retry),
`download_images` (skip phantom 404, resume-safe manifest).

Phase-1 completion (this session):
1. **`data/labels.py`** (new) — per-clip ice label from the daily flag, mapped
   through the local-standard-time (CST, −6 h) day boundary. Nullable Int64 so
   parquet keeps NA distinct from 0.
2. **`data/qc.py`** (new) — NOAA solar-elevation night detection + photometric
   glare/occlusion on a ¼-res decode; cached per frame, resume-safe. `to_hourly()`
   collapses burst captures to one frame per hour (nearest the top of the hour).
3. **`evaluation/reference_events.py`** (new) — ice-flag + stage-breakpoint
   onset/breakup; reindexes to a contiguous daily calendar (robust to outages)
   and masks USGS −999999 stage sentinels.
4. **`assemble_clips.py`** — integrated labels + QC; added max_gap_hours;
   stage-sentinel masking; QC-cache dedup; valid_fraction = NA (not 0) when a clip
   has no QC coverage.
5. **`baselines/ricenet_baseline.py`** — implemented `segment_ice_coverage`
   faithfully to the verified RIce-Net recipe (1152×640, polygon mask, pickled
   `torch.load(weights_only=False)`, IC = ice_px/pixels×100); fixed unconditional
   /255 and a full-trailing-window persistence rule; weight downloader.
6. **configs** — Cedarburg `ricenet_mask` (polygon/crop/pixels) + a `ricenet`
   block (input size, ImageNet norm, taus); station lat/lon; `usgs_dv_utc_offset_hours`.
7. **`.gitignore`** — added `data/external/` and `*.pth`.

A 5-dimension adversarial review (multi-agent) found 13 defects; all 13 are fixed
and re-verified. The two data-corruption bugs (stage −999999, ice-flag gaps)
did not affect current outputs because our three winters happen to be clean, but
the fixes harden the code for other station-winters.

---

## RIce-Net baseline (Phase 1 Acceptance #2) — DEFERRED to the CCAST GPU session

**Decision (2026-06-20): run the RIce-Net baseline on CCAST during Phase 2**, where
torch is already installed for the encoder. The run commands are wired into
`scripts/extract_embeddings_ccast.sbatch`. The baseline CODE is implemented and
unit-tested; this is purely a run/execution step.

Verified facts (m-ayyad/RIce-Net, MIT; HydroShare CC BY 4.0, DOI
ff4e9c4e87ef4d7d923efe77f5ed2b83):
- Cedarburg ships a river mask (now in `configs/stations.yaml`); **Bismarck does
  not** — its baseline would need a hand-digitized river polygon.
- The classifier ice/no-ice gate is not wired (config flag `false`); IC comes from
  the segmenter alone. `run()` refuses to proceed if the flag is set true.
- Model input is 1152×640; weights are a pickled full model (load weights_only=False).

Run (on CCAST, torch + smp present):
```bash
python -m fm_ice.baselines.ricenet_baseline --download
python -m fm_ice.baselines.ricenet_baseline --station cedarburg --winter 2022-2023
python -m fm_ice.baselines.ricenet_baseline --station cedarburg --winter 2023-2024
# then compare results/ricenet_events.csv against results/reference_events.csv
```

---

## Phase 2 — IN PROGRESS (FULL RUN SUBMITTED & RUNNING; smoke PASSED)

**FULL RUN SUBMITTED 2026-06-20:** `qsub job_full.pbs` → job **698177.bright04**,
state **R** on `gpu0006` (A100, queue `gpus`). Extracts one V-JEPA embedding per
clip for all three winters → `data/cache/vjepa2/<station>/<winter>.npy`. Expected
rows: cedarburg 2022-2023 → (1099,1024); 2023-2024 → (1173,1024); bismarck
2024-2025 → (1130,1024). PBS copies `logs/vjepa_full.{out,err}` back only at job
end. A background watcher (`scripts/_watch_vjepa_full.sh`) polls qstat, then
verifies every shape on completion.

**GATE A probe is BUILT and dry-run-validated** (see below), so the moment the
embeddings land, run:
```bash
conda activate fm-ice
PYTHONNOUSERSITE=1 python -m fm_ice.evaluation.probe_separability --all
```



The only GPU step: one frozen V-JEPA embedding per clip, cached so Phases 3-5 run
on CPU. Then GATE A (H3) decides whether the representation is good enough to model.

- **SMOKE TEST PASSED (2026-06-20):** `qsub job.pbs` → job 698176 ran on an
  A100-40GB (torch 2.12.1+cu130, cuda True), loaded `vjepa2-vitl-fpc64-256` from
  the offline cache, and wrote `(32, 1024)` float32 embeddings for the first 32
  clips of cedarburg 2022-2023. The full GPU path is validated end to end.
- **`job.pbs`** = the SMOKE launcher (PBS, queue `gpus`, 1 GPU): activates fm-ice,
  sets `PYTHONNOUSERSITE=1` + `HF_HOME` + `HF_HUB_OFFLINE=1`, runs
  `extract_embeddings ... --limit 32` (tagged output, never clobbers the cache).
- **`job_full.pbs`** = the FULL launcher (staged, ready to submit): V-JEPA
  embeddings for ALL three winters → `data/cache/vjepa2/<station>/<winter>.npy`
  (no `_smoke` tag). ~3,400 clips total; fits easily in the 4 h walltime on one A100.
- `extract_embeddings.py` gained a `--limit N` smoke flag and a device report.
- Encoder detail (verified): our 16-frame clips are accepted by the fpc64 (64-frame)
  checkpoint; output is 1024-d (ViT-L), mean-pooled over tokens.
- DINOv2 ablation (`--encoder dinov2`, for H2/Phase 4) and the deferred RIce-Net
  baseline are NOT in `job_full.pbs` yet — they need `facebook/dinov2-large`
  pre-downloaded and torch/smp respectively; add as a follow-up GPU session.
- Note: `scripts/extract_embeddings_ccast.sbatch` is a Slurm/`.venv` scaffold; on
  THIS cluster (PBS) use `job.pbs` / `job_full.pbs`.

### GATE A (H3) probe — IMPLEMENTED, awaiting the full embeddings
`src/fm_ice/evaluation/probe_separability.py` (new). A *linear* readout (logistic
regression, time-blocked 5-fold OOF — random CV would leak since clips overlap)
that asks per station-winter:
1. **pooled** ice/water AUC; 2. **lighting invariance** — refit within day-only
and night-only clips; 3. **lighting-only confound** baseline (night_frac +
brightness) so the embedding's margin over lighting is the honest signal.
Heuristic verdict printed (PASS if pooled≥0.80 & day≥0.70; REVIEW if pooled≥0.70;
else FAIL) — the human makes the gate call. Writes `results/gate_a_<encoder>.{csv,json}`.
Code path dry-run-validated on the 32-clip smoke `.npy` (correctly reported
"insufficient labeled support" — those clips are all pre-ice November).

### NEXT ACTION: when job 698177 finishes (watcher will report shapes)
```bash
conda activate fm-ice
PYTHONNOUSERSITE=1 python -m fm_ice.evaluation.probe_separability --all   # GATE A table
```
Pass → Phase 3 (temporal head). Fail → fix preprocessing (river crop, water mask,
clip width) BEFORE modeling. NB: the clip definition is now FROZEN — changing it
invalidates the cache.

## Phase 3 — temporal head — IMPLEMENTED & FIRST RESULT (2026-06-20)

Code: `models/events.py` (new, event reader, self-tested), `models/train.py`
(rewritten from skeleton), `models/temporal_head.py` (loss gained `pos_weight`).
Run (CPU, no GPU): `python -m fm_ice.models.train --encoder {vjepa2,dinov2}`.

**Split protocol (deviation, documented in train.py):** plan says hold out
`splits.test_winter`=2024-2025 on the train station, but cedarburg has only 2
winters and 2024-2025 is bismarck (Phase-5 transfer, untouched). So Phase 3 uses
**leave-one-winter-out on cedarburg**. No separate val winter → fixed 60 epochs,
config hyperparams, no test peeking.

**Result vs USGS ice flag (hours; lower better):**

| fold (test) | encoder | model | onset_err | breakup_err |
|---|---|---|---|---|
| 2023-2024 | vjepa2 | temporal | **18** | **22** |
| 2023-2024 | vjepa2 | per-frame | 1122 | 58 |
| 2023-2024 | dinov2 | temporal | 22 | 30 |
| 2023-2024 | dinov2 | per-frame | 22 | 30 |
| 2022-2023 | vjepa2 | temporal | 1146 | **2** |
| 2022-2023 | vjepa2 | per-frame | 726 | 70 |
| 2022-2023 | dinov2 | temporal | 710 | **2** |
| 2022-2023 | dinov2 | per-frame | 718 | 2 |

Readings (honest, n=2 winters so fragile):
- **Breakup timing is excellent & robust everywhere: 2–30 h.** Strong H1 signal.
- **Onset is bimodal.** 2023-2024 fold: ~18–22 h (nails the Nov-27 onset).
  2022-2023 fold: 710–1146 h — an **early-season (November) false-positive ice
  block**. It appears for BOTH encoders AND the linear per-frame probe, so it is a
  feature/label-transfer confound (Nov 2022 scenes look ice-like to a probe
  trained on the other winter), not merely TCN overfit — though the V-JEPA TCN
  amplifies it (Nov 1 vs Nov 19). The air-temp channel is underused; Nov is too
  warm for sustained ice.
- **H1 (temporal vs per-frame):** one clearly-supporting cell (vjepa2 2023-2024
  onset 18 vs 1122 — per-frame misses the brief late-Nov onset, temporal catches
  it); elsewhere ≈ tied. Not a clean sweep.
- **H2 (V-JEPA vs DINOv2):** tied on timing; consistent with the negative GATE-A.
- Headline "beat RIce-Net" still pending the RIce-Net run (deferred GPU step).

### Phase 3 UPDATE (2026-06-21): freezing guard + head ablation grid
Added a **physical freezing guard** to event reading (`events.onset_freeze_guard_c`,
default 0 deg C): a sustained ice run only counts if its mean air temp is at/below
freezing. Domain prior, NOT test-tuned. Ran the full `{vjepa2,dinov2} x {tcn,
transformer} x {guard off,on}` grid in parallel.

**Best model: vjepa2 / tcn / +guard -> mean 122 h** (was 297 h without guard). The
guard moved cedarburg 2022-2023 onset from Nov 1 -> Nov 30 (ref Dec 19), cutting
that onset error 1146 -> 446 h. Ranking by mean timing error: V-JEPA+guard 122 <
DINOv2 191 < V-JEPA per-frame 494. So (interim, n=2): temporal head beats per-frame
(H1 supported); V-JEPA edges DINOv2 on TIMING (tentative H2 positive, opposite the
static GATE-A). Transformer head overfits worse than TCN. Consolidate:
`python -m fm_ice.evaluation.phase3_report` -> results/phase3_summary.csv.
Figures: results/figures/discussion/{fig_timing,fig_nov_guard}.png.
Interim discussion PDF: docs/discussion_interim.tex -> discussion_interim.pdf
(compile with TinyTeX pdflatex; tables-only, grfext.sty missing for \includegraphics).

## Phase 4 — RUN + GATE B CALLED (2026-06-21)
H2 head-to-head table built: `results/phase4_h2_table.csv`. cedarburg LOO, vs usgs_ice_flag.
Mean timing error (h, lower better): V-JEPA temporal TCN+guard **122** | DINOv2 temporal
TCN(+/-guard) **191** | per-frame (DINOv2) 193 | BEAST 950 | BOCPD-pc1 1216 |
BOCPD-diffnorm 1262. RIce-Net row omitted (still paused).

**GATE B (H2): guard-dependent.** With the freezing guard (a domain prior applied
identically to both encoders), V-JEPA beats DINOv2 by 69 h (122 vs 191) -> keep the
video-FM framing. WITHOUT the guard, V-JEPA loses by 106 h (297 vs 191). The win rests
on the guard + the single weak 2022-2023 onset fold. Table shows guard-on AND guard-off
rows per encoder; human makes the final call. Label-free change-point baselines are all
far worse than the temporal head (thesis: temporal head over FM embeddings beats both
per-frame thresholding and label-free change detection).

Bug fixed this pass: `phase4_table.py` was reading STALE un-tagged `phase3_timing_<enc>.csv`
(no-guard, 297 h) -> would have flipped GATE B to a false negative. Now globs the
head/guard-tagged `phase3_timing_<enc>_*.csv`, emits both guard rows, discovers
winters/station (no hardcoding). Stale un-tagged timing CSVs deleted. BEAST threshold:
cpOccPr is diffuse (max ~0.18 over ~1100 steps), so the old 0.5 found 0 cps -> all-NaN;
default lowered to 0.1 and exposed as `--beast-thresh`. Change-point baseline computed on
the V-JEPA embedding stream only (the table collapses BEAST/BOCPD labels across encoder).

To reproduce: regenerate `results/changepoint_events.csv` (vjepa2; bocpd pc1+diffnorm,
beast pc1; both cedarburg winters) then `python -m fm_ice.evaluation.phase4_table`.

## Phases 5-6 — CODE COMPILED (2026-06-21), experiments NOT yet run
Written in parallel by sub-agents; all import-clean, CLIs parse, cheap self-tests pass.
- **Phase 4 (DINOv2 ablation half):** runs via `train.py --encoder dinov2` (done; in the table).
- **Phase 5:** `models/transfer.py` (train head on BOTH cedarburg winters, eval on
  bismarck 2024-2025, no retrain/no-leak scaler, report transfer gap vs phase3_summary).
- **Phase 6:** `evaluation/metrics.py` (+per_frame_agreement, per_frame_auc,
  event_f1_at_tolerances), `evaluation/evaluate.py` (master metrics -> phase6_metrics.csv),
  `evaluation/error_analysis.py` (error rate by night/glare/quality regime),
  `evaluation/figures.py` (H3 UMAP + event timelines, matplotlib Agg).
NOTE: these are SCAFFOLDS to refine when running each phase. matplotlib installed in
fm-ice (2026-06-21).

## RIce-Net — STILL PAUSED (user decision 2026-06-21)
Weights downloaded (data/external/ricenet/segmentation_model.pth, 97 MB, official
HydroShare). Two blockers: (1) loading needs torch.load(weights_only=False) =
external-pickle code execution (sandbox blocked it); (2) the pickle is incompatible
with installed smp 0.5.0 (PAN GAUBlock.interpolation_mode). User chose "don't run
now". To resume: authorize the unpickle + pin smp to RIce-Net's version OR rebuild
PAN(resnet50,2) from state_dict.

## Later phases
- Phase 4 DONE (GATE B called, guard-dependent V-JEPA win). Phase 5 transfer experiment
  (code ready). Phase 6 eval + figures.
- After Phase 5: MVP freeze.

---

## How to resume / rebuild

```bash
conda activate fm-ice
# rebuild a winter end to end (QC is cached; --force to rescore images):
python -m fm_ice.data.qc            --station cedarburg --winter 2022-2023
python -m fm_ice.data.assemble_clips --station cedarburg --winter 2022-2023
python -m fm_ice.evaluation.reference_events --all
```
