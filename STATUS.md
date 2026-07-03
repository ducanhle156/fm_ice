# STATUS.md — working status of FM_ice

Last updated: 2026-07-03, end of the plan-v2 task-list session. Companion to
**docs/FM_ice_plan_v2.md** (the plan of record; IMPLEMENTATION_PLAN.md is
superseded, reference only) and DATA.md (the data spec). The pre-v2 phase log
(phases 1–6 in detail) lives in git history at tag-commit `d2f23ac` and in the
merged branch `phase4-gate-b`.

**Do not start H4 or any docs/IDEAS.md item; those are post-freeze. The
Nebraska station waits for the owner's manual event audit (plan Sec. 5).**

---

## Phase / task table

| # | Work item | Status | Evidence |
|---|---|---|---|
| P1 | Data assembly (cedarburg ×2, bismarck ×1 winters: downloads, clips, QC, labels, reference events) | DONE (pre-v2) | data/interim, results/reference_events.csv |
| P2 | Embeddings, V-JEPA 2 + DINOv2, frozen ViT-L | DONE (pre-v2) | data/cache/{vjepa2,dinov2} |
| P3 | Temporal head (TCN/transformer), LOO on cedarburg | DONE (pre-v2) | results/phase3_* |
| P4 | Gate A (probe) PASS both encoders; Gate B called | DONE (pre-v2), **superseded by guard audit** | results/gate_a_*, phase4_h2_table.csv |
| P5 | Transfer to bismarck, no retraining | DONE (pre-v2), MVP frozen | results/phase5_* |
| P6 | Metrics + error analysis | DONE (pre-v2) | results/phase6_* |
| V2-1 | Merge phase4-gate-b → main; land plan v2 + IDEAS | DONE 07-03 | main `da55a42`, `67f6e78` |
| V2-2 | Repoint CLAUDE.md; SUPERSEDED header on old plan | DONE 07-03 | main `3cbd6e8` |
| V2-3 | Guard audit + AFDD re-derivation + Gate B rerun | DONE 07-03 | main `efba729`; results/reextract/timing.csv |
| V2-4 | entropy_jam.py smoke test (C2) | DONE 07-03 | main `f41b1dd`; results/entropy_jam/ |
| V2-5 | chippewa_bruce + mohawk_schenectady; downloads; ice-flag gate | DONE 07-03 | main `84a380c`; data/raw complete |
| V2-6 | Degree-day baseline; BOCPD verification | DONE 07-03 | main `7bb20e4`; results/degreeday_events.csv |
| V2-7 | RIce-Net unblock (half-day cap) + both winters run | DONE 07-03, fallback not needed | main `7b35122`; results/ricenet_events.csv |
| next | Week-2 CRREL event audit (plan Sec. 5) — critical path | NOT STARTED (owner) | — |
| next | Assemble clips + embeddings for chippewa/mohawk winters | NOT STARTED | images on disk, GPU job needed |

## Guard audit (plan-v2 addendum item 2) — DONE 2026-07-03

**How the freezing guard was derived.** `onset_freeze_guard_c: 0.0` is the
physical freezing point, hardcoded in configs/pipeline.yaml — not fitted to any
data and not a calendar-date window. HOWEVER: (a) it was introduced in a single
commit (d2f23ac) *after* all three station-winters — including both LOO test
folds and the bismarck transfer winter — had been scored, so the decision to
INCLUDE the guard was made with test results visible; (b) it filters sustained
runs for onset AND breakup, though its physical rationale is onset-only.
Verdict: the value is a prior, the inclusion was test-visible ⇒ re-derived blind.

**Re-derivation (AFDD).** `fm_ice.data.degree_days` + `read_events(..., afdd,
min_onset_afdd)`: onset can only fire after accumulated freezing degree-days
exceed tau = 0.5 × min(AFDD at reference onset) over CALIBRATION winters —
cedarburg train winters only, leave-one-out respected per fold (scoring winter
W calibrates on the OTHER cedarburg winter; bismarck calibrates on both).
Onset-only; breakup never gated; a run straddling arming keeps its true start.
No retraining: head probabilities are guard-invariant, so events were re-read
from the existing pred dumps (`fm_ice.evaluation.reextract_events` →
results/reextract/timing.csv; the meanair rows reproduce the published
446 h / 1146 h cedarburg 2022-2023 onsets exactly).

## The 6-method table (results/phase4_h2_table.csv), onset / breakup SEPARATE

Hours vs usgs_ice_flag, cedarburg leave-one-winter-out; lower is better.

| method | 22-23 onset | 22-23 brkup | 23-24 onset | 23-24 brkup | onset mean | brkup mean |
|---|---|---|---|---|---|---|
| V-JEPA temporal (TCN) | 1146 | 2 | 18 | 22 | 582 | 12 |
| V-JEPA +meanair guard (legacy) | 446 | 2 | 18 | 22 | 232 | 12 |
| V-JEPA +AFDD guard (blind) | 738 | 2 | 1118 | 22 | 928 | 12 |
| DINOv2 temporal (TCN) | 710 | 2 | 22 | 30 | 366 | 16 |
| DINOv2 +meanair guard | 710 | 2 | 22 | 30 | 366 | 16 |
| DINOv2 +AFDD guard | 710 | 2 | 1122 | 30 | 916 | 16 |
| per-frame (V-JEPA) | 726 | 70 | 1122 | 58 | 924 | 64 |
| per-frame (DINOv2) | 718 | 2 | 22 | 30 | 370 | 16 |
| BOCPD-pc1 | 434 | 1186 | 1514 | 1730 | 974 | 1458 |
| BOCPD-diffnorm | 1650 | 630 | 1506 | — | 1578 | 630 |
| BEAST | 426 | — | — | 1474 | 426 | 1474 |
| degree-day (AFDD/ATDD) | 720 | 144 | 1152 | 192 | 936 | 168 |
| RIce-Net (published rule) | 344 | 2023 | 15 | 1592 | 179.5 | 1807.5 |

**GATE B reading (onset and breakup separately; pooled means are banned):**
1. Guard-free (primary evidence): DINOv2 wins onset by 216 h, V-JEPA wins
   breakup by 4 h — in-station H2 is negative-to-mixed.
2. The legacy meanair guard is the ONLY variant under which V-JEPA wins onset;
   that win does not survive the audit.
3. The blind AFDD guard hurts BOTH encoders on 2023-2024: the true onset is a
   short late-November cold-snap run (Nov 28–30, AFDD 14.7) followed by a mild
   December (AFDD flat ~24 until Dec 12), so the fold-blind tau of 29.5 kills
   the genuine onset run. With two train winters the calibration is too
   brittle; reported honestly rather than tuning the fraction post-hoc.
   Consequence: the freezing guard is dropped from the headline method; guard
   rows stay in the ablation.
4. Breakup is guard-independent everywhere. The temporal heads own breakup
   (12–16 h vs 64+ h for everything else); RIce-Net owns first-appearance
   onset in-station (179.5 h mean, 15 h on 23-24) but its breakup is
   degenerate — "first falling edge after onset" fires ONE HOUR after onset on
   intermittent coverage (both winters). The cleanest H2 evidence remains the
   guard-free bismarck transfer: V-JEPA 18/122 vs DINOv2 22/150 (onset/breakup).

**Known pooled-mean remnants (flagged, not fixed):** transfer.py
`cedarburg_loo_mean`/`transfer_gap_h` and phase3_summary.csv still pool
onset+breakup internally; phase4_h2_table.csv and GATE B no longer do.

## Entropy jam smoke test (plan-v2 Sec. 6 + addendum 4) — DONE 2026-07-03

`fm_ice.evaluation.entropy_jam` (vjepa2/tcn, out-of-fold pred dumps, no
retraining). **tau_H = 0.648 nats** = p99 of Bernoulli entropy pooled over the
cedarburg train winters, FROZEN before any event station is scored. Detector:
H > tau_H for ≥ 6 clips chained while above-threshold clips are < 8 h apart
(timestamp-based; QC holes ≥ 8 h split windows).

| station-winter | clips | H > tau_H clips | detections | FAR /100 d |
|---|---|---|---|---|
| cedarburg 2022-2023 | 1099 | 19 | 0 | 0.0 |
| cedarburg 2023-2024 | 1173 | 4 | 0 | 0.0 |
| bismarck 2024-2025 | 1130 | 16 | 0 | 0.0 |

Zero false alarms; the 24 h persistence rule kills every isolated
high-entropy clip. Outputs: results/entropy_jam/{*_entropy.npy,
detections.json, false_alarm_table.csv}. Paper caveat: on the calibration
winters ~1% of clips exceed tau_H by construction; bismarck and future
stations are the honest FAR rows.

## New stations (plan-v2 Sec. 3, week 1) — ADDED 2026-07-03

- `chippewa_bruce` (05356500): WI_Chippewa_River_near_Bruce, 24/7, 60-min,
  archive 2022-10-06, 4 winters, ASOS RCX. All streams downloaded
  (10,791 images, small tier).
- `mohawk_schenectady` (01354500): live fixed camera
  NY_Mohawk_River_at_Freemans_Bridge_at_Schenectady, daylight/60-min since
  **2023-09-18** → 3 camera winters. The site's other 13 cameras are hidden
  PTZ presets, stale since Nov–Dec 2024 (the plan registry's 2024-09-27 date
  belongs to those). Second live viewpoint (Stockade, 24/7, 5-min) recorded as
  a non-default role. ASOS = ALB (SCH is part-time, ~8 obs/day). Per-station
  `usgs_dv_utc_offset_hours: -5` (EST), honored via the new
  `config.station_utc_offset_hours` used by assemble_clips, reference_events,
  changepoint_events, degree_days. All streams downloaded (4,481 images).
- `defaults.image_size` = `small` (plan v2 fixed decision); `download_images`
  now reads the config default. Existing cedarburg/bismarck embedding caches
  came from overlay-tier frames; plan v2 forbids re-extraction.

**Ice-flag gate (plan-v2 Sec. 11):**

| station-winter | flagged | blocks | longest | gate |
|---|---|---|---|---|
| chippewa_bruce 2022-2023 | 120 | 3 | 116 | PASS |
| chippewa_bruce 2023-2024 | 91 | 3 | 56 | PASS |
| chippewa_bruce 2024-2025 | 106 | 1 | 106 | PASS |
| chippewa_bruce 2025-2026 | 111 | 1 | 111 | PASS |
| mohawk_schenectady 2023-2024 | 0 | 0 | 0 | FAIL |
| mohawk_schenectady 2024-2025 | 0 | 0 | 0 | FAIL |
| mohawk_schenectady 2025-2026 | 75 | 3 | 69 | PASS |

Mohawk published no ice qualifier in its first two camera winters → the plan's
pre-written fallback applies (manual dates from the spot audit for test-only
stations). Mohawk is `event_test`; its C2 references come from the week-2
CRREL/dashboard event audit, so C2 is unaffected.

## RIce-Net baseline — UNBLOCKED AND RUN 2026-07-03

**The fix (11 minutes into the half-day cap; recipe in
`requirements-ricenet.txt`):** separate venv `.venv-ricenet` with
`segmentation-models-pytorch==0.3.4` + `timm==0.9.7` + `torch==2.12.1+cpu` /
`torchvision==0.27.1+cpu` (both from the pytorch cpu index — mixing indexes
breaks torchvision import), then `torch.load(weights_only=False)` on the
trusted HydroShare checkpoint. Classes verified on real frames
({0: masked background, 1: ice, 2: water}; peak-winter frame = 83% ice).
Full runs: 4445 + 4704 hourly frames, CPU, ~4–6 h/winter (GPFS stalls).
Events in the 6-method table above; coverage series in
results/ricenet_coverage_*.csv. The named fallback was NOT needed.

## Environment

- **conda env `fm-ice`** (Python 3.12) at
  `/mmfs1/projects/chau.le/Computer_Vision_FM/miniconda3/envs/fm-ice`.
  Activate via that miniconda's `etc/profile.d/conda.sh`. System python is 3.9
  — never use it. This machine has outbound network; cluster batch is PBS
  (job.pbs / job_full.pbs / job_dinov2.pbs; the sbatch script is a Slurm
  scaffold for a different cluster).
- **`~/.local` LEAK — set `PYTHONNOUSERSITE=1` for any torch run** (stale
  nvidia-nccl-cu12 shadows fm-ice's nccl; baked into job.pbs).
- **RIce-Net only:** `.venv-ricenet` per `requirements-ricenet.txt`. Keep it
  out of fm-ice.
- GPU stack for embeddings: torch 2.12.1+cu130 + transformers 5.12.1 in
  fm-ice; V-JEPA 2 checkpoint cached under `data/external/hf_home`.

## How to resume / rebuild

```bash
conda activate fm-ice
python -m fm_ice.evaluation.metrics                 # self-test, no data needed
# rebuild a winter end to end (QC cached; --force to rescore):
python -m fm_ice.data.qc             --station chippewa_bruce --winter 2024-2025
python -m fm_ice.data.assemble_clips --station chippewa_bruce --winter 2024-2025
python -m fm_ice.evaluation.reference_events --all
# guard audit / table / baselines / entropy (all CPU, no retraining):
python -m fm_ice.evaluation.reextract_events
python -m fm_ice.baselines.degreeday --all
python -m fm_ice.evaluation.phase4_table
python -m fm_ice.evaluation.entropy_jam
```

---

## Deviations from FM_ice_plan_v2.md (one line + reason each)

1. **No mvp_ice subfolder** — owner chose "don't move code" when shown that it
   invalidates the plan's own paths and the cached-artifact layout.
2. **"PRs" are local `git merge --no-ff` + push** — no gh CLI on the cluster;
   owner approved; history is equivalent and auditable.
3. **Mohawk registry corrected** — live camera archive starts 2023-09-18 (3
   winters, not 2); the plan's 2024-09-27 belongs to stale hidden PTZ cams;
   found via the siteId query the plan itself mandates.
4. **Mohawk ASOS is ALB, not the nearest (SCH)** — SCH is part-time (~8
   obs/day), which biases daily means; ALB is 13 km, hourly, 24/7.
5. **Mohawk ice-flag gate FAILS 2023-24 and 2024-25** (no qualifier published)
   — pre-written fallback invoked: spot-audit dates for test-only stations.
6. **BOCPD row was never empty** — changepoint_events.py existed and its rows
   were already in the table; plan v2's "empty rows" claim was stale.
7. **AFDD guard re-derived and reported, but it DEGRADES the mild winter**
   (1118 h vs 18 h onset on 23-24) — reported honestly per the plan's own
   rule instead of tuning `afdd_guard_frac` post-hoc; guard dropped from the
   headline method, kept in the ablation.
8. **RIce-Net compute finished at ~10 h wall, near the half-day cap** — the
   cap governed unblocking effort, which took 11 minutes; the rest was bounded
   CPU inference (two winters, GPFS stalls), so the fallback was not invoked.
9. **Pooled means remain inside transfer.py/phase3_summary internals** —
   flagged for follow-up; no new pooled numbers are reported anywhere.
10. **image_size `small` applies to future downloads only** — existing
    cedarburg/bismarck caches are overlay-tier; plan v2 Sec. 2 forbids
    re-extraction, so the mixed tier is documented rather than fixed.

**STOPPED HERE for owner review (per instruction). Next per plan v2: the
week-2 CRREL event audit (critical path), then clips + embeddings for the new
stations.**
