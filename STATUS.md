# STATUS.md — working status of FM_ice

Last updated: 2026-07-03. Companion to docs/FM_ice_plan_v2.md (the plan of
record; IMPLEMENTATION_PLAN.md is superseded) and DATA.md (the data spec).
This file tracks *what is actually done* right now.

**Phase 1 (Data assembly) is essentially complete.** Downloads, clip assembly,
per-clip labels, per-clip QC, and reference events are done and verified. The
only open item is *running* the RIce-Net baseline (Acceptance #2), which needs an
external model + weights — see "Open decision" below.

---

## Plan-v2 task list (2026-07-03 session, merged to main one PR at a time)

1. DONE — phase4-gate-b merged to main (metrics self-test passed first);
   docs/FM_ice_plan_v2.md + docs/IDEAS.md committed to main.
2. DONE — CLAUDE.md repointed to plan v2; IMPLEMENTATION_PLAN.md marked SUPERSEDED.
3. DONE — guard audit + AFDD re-derivation (section below).
4. DONE — entropy_jam.py smoke test (section below).
5. DONE — chippewa_bruce + mohawk_schenectady added; downloads running in the
   background (section below).
6. DONE — degree-day baseline row filled (`fm_ice.baselines.degreeday`,
   AFDD/ATDD, train-winter calibration, LOO folds → results/degreeday_events.csv;
   in the phase4 table: onset mean 936 h, breakup mean 168 h — the weak thermal
   control, as intended). BOCPD row was NOT empty (plan v2 stale on this):
   changepoint_events.py + results/changepoint_events.csv already existed and
   the BOCPD-pc1/diffnorm/BEAST rows are in the regenerated table.
7. DONE — RIce-Net UNBLOCKED and run on both cedarburg winters (section below);
   the named fallback was NOT needed.

## Guard audit (plan-v2 addendum item 2) — DONE 2026-07-03

**How the freezing guard was derived.** `onset_freeze_guard_c: 0.0` is the
physical freezing point, hardcoded in configs/pipeline.yaml — not fitted to any
data and not a calendar-date window. HOWEVER: (a) it was introduced in a single
commit (d2f23ac) *after* all three station-winters — including both LOO test
folds and the bismarck transfer winter — had been scored, so the decision to
INCLUDE the guard was made with test results visible; (b) it filters sustained
runs for onset AND breakup, though its physical rationale is onset-only.
Verdict: value is a prior, inclusion was test-visible ⇒ re-derived blind.

**Re-derivation (AFDD).** New `fm_ice.data.degree_days` + `read_events(...,
afdd, min_onset_afdd)`: onset can only fire after accumulated freezing
degree-days exceed tau = 0.5 x min(AFDD at reference onset) over CALIBRATION
winters — cedarburg train winters only, leave-one-out respected per fold
(scoring winter W calibrates on the OTHER cedarburg winter; bismarck calibrates
on both). Onset-only; breakup is never gated. A run straddling arming keeps its
true start. No retraining: head probabilities are guard-invariant, so events
were re-read from the existing pred dumps (`fm_ice.evaluation.reextract_events`
→ results/reextract/timing.csv; the meanair rows reproduce the published
446 h / 1146 h cedarburg 2022-2023 onsets exactly).

**Gate B rerun (results/phase4_h2_table.csv), onset / breakup SEPARATE, hours
vs usgs_ice_flag, cedarburg LOO:**

| method (TCN) | 22-23 onset | 22-23 brkup | 23-24 onset | 23-24 brkup | onset mean | brkup mean |
|---|---|---|---|---|---|---|
| V-JEPA no guard | 1146 | 2 | 18 | 22 | 582 | 12 |
| V-JEPA +meanair (legacy) | 446 | 2 | 18 | 22 | 232 | 12 |
| V-JEPA +AFDD (blind) | 738 | 2 | 1118 | 22 | 928 | 12 |
| DINOv2 no guard | 710 | 2 | 22 | 30 | 366 | 16 |
| DINOv2 +meanair | 710 | 2 | 22 | 30 | 366 | 16 |
| DINOv2 +AFDD | 710 | 2 | 1122 | 30 | 916 | 16 |

**Reading.** (1) Guard-free (primary evidence): DINOv2 wins onset by 216 h,
V-JEPA wins breakup by 4 h — in-station H2 is negative-to-mixed. (2) The legacy
meanair guard is the ONLY variant under which V-JEPA wins onset; that win does
not survive the audit. (3) The blind AFDD guard hurts BOTH encoders on
2023-2024: the true onset is a short late-November cold-snap run (Nov 28–30,
AFDD 14.7) followed by a mild December (AFDD flat ~24 until Dec 12), so the
fold-blind tau of 29.5 kills the genuine onset run. With two train winters the
calibration is too brittle; the deliberate choice is to report this honestly
rather than tune the fraction post-hoc. (4) Breakup is guard-independent
everywhere, as expected from an onset-only prior. Consequence for the paper:
the freezing guard is dropped from the headline method; guard rows stay in the
ablation. The cleanest H2 evidence remains the guard-free bismarck transfer
(V-JEPA 18/122 vs DINOv2 22/150 onset/breakup).

**Known pooled-mean remnants (flagged, not yet fixed):** transfer.py
`cedarburg_loo_mean`/`transfer_gap_h` and phase3_summary.csv still pool
onset+breakup internally; phase4_h2_table.csv and GATE B no longer do.

## Entropy jam smoke test (plan-v2 Sec. 6 + addendum item 4) — DONE 2026-07-03

`fm_ice.evaluation.entropy_jam` (vjepa2/tcn, out-of-fold pred dumps, no
retraining). tau_H = 0.648 nats = p99 of Bernoulli entropy pooled over the
cedarburg train winters, FROZEN before any event station is scored. Detector:
H > tau_H for >= 6 clips chained while above-threshold clips are < 8 h apart
(timestamp-based; QC holes >= 8 h split windows).

False-alarm table (results/entropy_jam/false_alarm_table.csv) — no jams
documented in the cached winters, so every detection would be a false alarm:

| station-winter | clips | H > tau_H clips | detections | FAR /100 d |
|---|---|---|---|---|
| cedarburg 2022-2023 | 1099 | 19 | 0 | 0.0 |
| cedarburg 2023-2024 | 1173 | 4 | 0 | 0.0 |
| bismarck 2024-2025 | 1130 | 16 | 0 | 0.0 |

Zero false alarms: the 24 h persistence rule kills every isolated
high-entropy clip. Outputs: `results/entropy_jam/<station>_<winter>_entropy.npy`
(row-aligned to the pred CSVs), `detections.json` (frozen params + windows).
Caveat for the paper: on the calibration winters ~1% of clips exceed tau_H by
construction; bismarck (and future stations) are the honest FAR rows.

## New stations (plan-v2 Sec. 3 / week 1) — ADDED 2026-07-03

Both cameras verified live via the HIVIS siteId query the plan mandates.

- `chippewa_bruce` (05356500): WI_Chippewa_River_near_Bruce, 24/7, 60-min,
  archive 2022-10-06, 4 winters. ASOS RCX (Ladysmith, 21 km).
- `mohawk_schenectady` (01354500): the site lists 15 cameras; 13 are hidden PTZ
  presets stale since Nov-Dec 2024. Live fixed camera
  NY_Mohawk_River_at_Freemans_Bridge_at_Schenectady runs daylight/60-min since
  **2023-09-18**, so THREE camera winters exist (plan registry said 2 from
  2024-09-27 — that date belongs to the stale PTZ cams). Second live viewpoint
  NY_Mohawk_River_at_Stockade_at_Schenectady (24/7, 5-min, 2024-08-21) recorded
  as `stockade` role, not downloaded (post-freeze IDEAS row). ASOS: SCH (2 km)
  is part-time (~8 obs/day, warm-biased daily means) → primary is ALB (13 km,
  hourly 24/7); per-station `usgs_dv_utc_offset_hours: -5` (EST vs the CST
  default), honored by a new `config.station_utc_offset_hours` helper wired
  into assemble_clips, reference_events, changepoint_events, degree_days.
- `defaults.image_size` = `small` (plan v2 fixed decision); download_images now
  reads the config default instead of hardcoding overlay. Caveat: existing
  cedarburg/bismarck caches were extracted from overlay-tier frames; plan v2
  forbids re-extraction.
- Downloads: stage/temperature/ice-flags/images for all winters of both
  stations running in the background → logs/download_new_stations_20260703.log.

**Ice-flag gate (plan-v2 Sec. 11, week 1):** flagged days / contiguous blocks /
longest block, per winter:

| station-winter | flagged | blocks | longest | gate |
|---|---|---|---|---|
| chippewa_bruce 2022-2023 | 120 | 3 | 116 | PASS |
| chippewa_bruce 2023-2024 | 91 | 3 | 56 | PASS |
| chippewa_bruce 2024-2025 | 106 | 1 | 106 | PASS |
| chippewa_bruce 2025-2026 | 111 | 1 | 111 | PASS |
| mohawk_schenectady 2023-2024 | 0 | 0 | 0 | FAIL |
| mohawk_schenectady 2024-2025 | 0 | 0 | 0 | FAIL |
| mohawk_schenectady 2025-2026 | 75 | 3 | 69 | PASS |

Mohawk published no ice qualifier at all in its first two camera winters.
Per the plan's pre-written fallback: manual dates from the spot audit for
test-only stations. Mohawk is `event_test` — its C2 references come from the
CRREL/dashboard event audit (week 2), so C2 is unaffected; only its
onset/breakup rows need audited dates. All 4 chippewa (train) winters pass.

## RIce-Net baseline — UNBLOCKED AND RUN 2026-07-03 (was paused since 06-21)

**The fix (11 minutes into the half-day cap; recipe pinned in
`requirements-ricenet.txt`):** separate venv `.venv-ricenet` with
`segmentation-models-pytorch==0.3.4` + `timm==0.9.7` + `torch==2.12.1+cpu` /
`torchvision==0.27.1+cpu` (both from the pytorch cpu index — mixing indexes
breaks torchvision import), then `torch.load(weights_only=False)` on the
trusted HydroShare checkpoint. The pickled PAN loads and forward-passes.
Verified before the full run: 3 classes = {0: masked background, 1: ice,
2: water}; a peak-winter frame (2023-02-05) segments 83% ice, spring frames 0%.
Full runs: 4445 + 4704 hourly frames, CPU, ~4-6 h/winter wall (GPFS stalls).

**Events (results/ricenet_events.csv, published 15%/20% + 8 h rule):**

| winter | RIce-Net onset | RIce-Net breakup | onset err | breakup err |
|---|---|---|---|---|
| 2022-2023 | 2022-12-04 22:00 | 2022-12-04 23:00 | 344 h | 2023 h |
| 2023-2024 | 2023-11-27 21:00 | 2023-11-27 22:00 | 15 h | 1592 h |

**Reading (onset and breakup separately, per the reporting rule):** the
threshold rule is a genuinely strong FIRST-APPEARANCE onset detector — its
onset mean (179.5 h) is the best onset row in the in-station table, and 15 h
on 2023-2024 beats everything. But it cannot time breakup at all: "first
falling edge after onset" fires ONE HOUR after onset on intermittent coverage
(both winters), giving a 1807.5 h breakup mean vs the temporal head's 12 h.
Coverage series sanity-checked (correct seasonal structure, max 100%). This is
the H1 head-to-head the paper needs: threshold coverage rules vs temporal
reasoning, strongest contrast on breakup.

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

## Phase 5 — transfer test DONE (2026-06-21), MVP FROZEN
Refined `models/transfer.py`: `_cedarburg_loo_anchor` now reads `phase3_summary.csv`
correctly (exact config match + per-winter onset/breakup means; takes `temp_guard`) — it
previously read non-existent columns and mixed temporal/perframe/guard rows, so the gap
came out blank. Pred dump now written under the CANONICAL
`phase3_pred_<enc>_bismarck_<winter>_<head>[_guard].csv` schema so Phase 6 scores the
held-out station with no code change; `phase5_transfer_*` tagged per config (no overwrite);
gap defined on `usgs_ice_flag` only; added `--selftest`. New `evaluation/phase5_report.py`
-> `phase5_summary.csv` (mirrors `phase3_report.py`).

Protocol (no leakage): head + per-frame probe + FeatureScaler fit on BOTH cedarburg winters
ONLY; bismarck 2024-2025 transformed + predicted, never trained on. Full matrix
(vjepa2/dinov2 x guard/no-guard, TCN). Timing error vs USGS ice flag (hours):

| config                    | bismarck onset | bismarck breakup | cedarburg LOO mean | gap (mean) |
|---------------------------|---------------:|-----------------:|-------------------:|-----------:|
| vjepa2 / tcn / +guard     | 18             | 122              | 122                | -52        |
| vjepa2 / tcn (no guard)   | 18             | 122              | 297                | -227       |
| dinov2 / tcn (+/- guard)  | 22             | 150              | 191                | -105       |

Findings (honest):
- **Transfer succeeds.** A cedarburg-only head predicts bismarck onset within 18 h (V-JEPA)
  / 22 h (DINOv2) and breakup within 122 h / 150 h, no retraining. Bismarck mean timing
  (70 h V-JEPA, 86 h DINOv2) is LOWER than the cedarburg LOO mean, so the gap is negative —
  but largely because the cedarburg LOO is inflated by the Nov-2022 onset confound, which
  bismarck does not share. The absolute bismarck numbers are the real story.
- **Onset transfers BETTER than breakup — the OPPOSITE of the interim prediction.** Onset is
  essentially solved on the unseen station (18-22 h); breakup is the harder transfer
  (122-150 h vs ~2-22 h on cedarburg).
- **The freezing guard is cedarburg-specific.** Bismarck onset = 18 h with OR without the
  guard (icier 52%, colder, no warm-November false positive). The guard only moves the
  cedarburg anchor — on bismarck the head needs no freezing prior.
- **H2 on the held-out station is CLEAN and guard-independent:** V-JEPA beats DINOv2 on
  bismarck (onset 18 vs 22 h, breakup 122 vs 150 h, mean 70 vs 86 h, pf_auc 0.995 vs 0.981),
  and unlike Cedarburg's GATE B this does NOT rest on the guard. The clearest V-JEPA>DINOv2
  signal in the project.
- **H1 on transfer:** temporal head >> per-frame probe on bismarck onset (18 h vs 306 h at
  the 48 h event rule) — temporal modeling generalizes.

Outputs: `results/phase5_summary.csv`, `results/phase5_transfer_<enc>_<head>[_guard].csv`,
`results/phase3_pred_<enc>_bismarck_2024-2025_<head>[_guard].csv`. Logs in `logs/phase5/`.

## Phase 6 — metrics + error analysis DONE (now incl. bismarck)
`evaluation/evaluate.py` -> `phase6_metrics.csv` (80 rows: all cedarburg LOO configs +
bismarck transfer; timing + event-F1@24/48/72 + per-frame acc/AUC vs both references).
`evaluation/error_analysis.py` -> `phase6_error_analysis.csv` (60 regime rows). **Night**
drives per-frame error overall (+2.8 pp vs day; +2.2 pp on bismarck); glare is negative
(glare clips skew to daytime open water). Bismarck pf_auc 0.99 (V-JEPA) / 0.98 (DINOv2):
static separability (H3) transfers. `evaluation/figures.py` (H3 UMAP + timelines) optional.
matplotlib installed in fm-ice. These were SCAFFOLDS; refined and run this pass.

## RIce-Net — STILL PAUSED (user decision 2026-06-21)
Weights downloaded (data/external/ricenet/segmentation_model.pth, 97 MB, official
HydroShare). Two blockers: (1) loading needs torch.load(weights_only=False) =
external-pickle code execution (sandbox blocked it); (2) the pickle is incompatible
with installed smp 0.5.0 (PAN GAUBlock.interpolation_mode). User chose "don't run
now". To resume: authorize the unpickle + pin smp to RIce-Net's version OR rebuild
PAN(resnet50,2) from state_dict.

## Later phases
- Phase 4 DONE (GATE B called, guard-dependent V-JEPA win). Phase 5 transfer DONE
  (clean guard-independent V-JEPA>DINOv2 on the held-out station). Phase 6 metrics +
  error analysis DONE; figures optional.
- **MVP FROZEN (2026-06-21).** Per the plan, only evaluation, figures, and writing proceed
  from here. Remaining optional/unblocked-only work: RIce-Net anchor (paused), Phase 6
  figures, H4 forecast head (stretch, cut if it threatens the freeze).

---

## How to resume / rebuild

```bash
conda activate fm-ice
# rebuild a winter end to end (QC is cached; --force to rescore images):
python -m fm_ice.data.qc            --station cedarburg --winter 2022-2023
python -m fm_ice.data.assemble_clips --station cedarburg --winter 2022-2023
python -m fm_ice.evaluation.reference_events --all
```
