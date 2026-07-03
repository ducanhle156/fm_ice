# FM_ice plan of record v2. Two-in-one paper, solo, 20 h/week, 14 weeks

Written 2026-07-02 (evening). Supersedes `FM_ice_MVP_plan.md`, which supersedes
the 16-task runbook and `IMPLEMENTATION_PLAN.md`. One plan of record. This one.

Decisions locked with Anh on 2026-07-02: two-in-one framing, 5 stations,
CCAST GPU confirmed working, 20 hours per week.

Hard truth up front: 20 h/week and two contributions do not fit in 12 weeks.
This plan runs 14 weeks (Jul 6 to Oct 11), arXiv submission target Oct 9,
hard stop Oct 16. If you want 12 weeks, use the drop order in Section 12.

---

## Status addendum, 2026-07-02 late evening (read this first on CCAST)

A CCAST Claude session executed the full spine today, but against the OLD
`IMPLEMENTATION_PLAN.md` scope (cedarburg + bismarck only). Results:

- Phases 1 to 6 done: 3 station-winters assembled with QC; V-JEPA 2 and DINOv2
  embeddings cached; Gate A PASS for both encoders (pf_auc 0.98 to 0.99).
- H1 supported: temporal head far beats the per-frame probe (bismarck onset
  18 h vs 306 h).
- Transfer works, no retraining: bismarck onset within 18 to 22 h. Onset
  transfers better than breakup, the opposite of the prediction.
- Gate B is guard-dependent: V-JEPA beats DINOv2 by ~69 h WITH the freezing
  guard, loses by ~106 h without. On the guard-free transfer station V-JEPA
  wins (70 vs 86 h mean).
- RIce-Net baseline paused: weights downloaded, loading blocked (torch pickle
  policy + segmentation-models-pytorch 0.5.0 incompatibility).
- Branch `phase4-gate-b` pushed; PR not yet opened.

What this changes, in priority order:

1. **Merge to main first.** Open the pending PR, merge, then sync this file and
   `IDEAS.md` into the repo and repoint `CLAUDE.md` here. The cluster agent
   proposed H4 (forecast head) because it reads the old plan. H4 stays cut.
2. **Guard audit before anything is written.** The freezing guard must be
   calibrated on train winters only and stated as a physical prior, ideally
   AFDD-based (onset can fire only after accumulated freezing degree-days pass
   a threshold), not an ad hoc date window. If it was tuned with test winters
   visible, re-derive it blind or drop it and report both numbers. Gate B
   currently flips on this one component; it decides the H2 story.
3. **Reporting rule, reaffirmed:** onset and breakup separately, never a pooled
   mean. The "122 h mean" hides breakup at 2 to 22 h. The guard-free transfer
   comparison is the primary H2 evidence; in-station H2 is guard-confounded.
4. **C2 smoke test is now nearly free.** The trained head exists. Dump per-clip
   probability vectors, compute H(t) for every cached winter, set tau_H on
   cedarburg train winters, count false alarms on all anchor winters. One day.
   Debug `entropy_jam.py` on data you already have, before event stations land.
5. **Event audit (Sec. 5) is the critical path now.** Weeks 3 to 8 of the
   original schedule are largely pre-paid on 2 stations. Next data work is the
   CRREL audit, then chippewa, mohawk, and the NE camera.
6. **Cheap missing baselines:** degree-day and BOCPD rows are still empty. CPU
   only, about 1.5 days combined.
7. **RIce-Net unblock, capped at one more half day:** try
   `torch.load(path, weights_only=False)` (torch >= 2.6 changed the default;
   acceptable for a trusted checkpoint) and pin
   `segmentation-models-pytorch==0.3.4` with a matching `timm` in a separate
   venv. Still broken after half a day: invoke the named fallback (threshold
   rule on our features, labeled as such) and stop.

---

## 1. The paper

**Working title:** "Beyond thresholds: label-light river ice monitoring with
frozen foundation models, from seasonal phenology to ice-jam anomalies."

**Contribution C1 (statistical backbone).** A small temporal head on frozen
V-JEPA 2 video embeddings times river ice onset and breakup better than the
RIce-Net coverage-threshold rule (Ayyad et al. 2025), a degree-day control, and
a BOCPD change-point baseline, evaluated on held-out winters and a held-out
station, across ~15 station-winters. Ablation: V-JEPA (video FM) vs DINOv2
(image FM), identical head, identical splits.

**Contribution C2 (novelty hook).** Ice-jam detection with zero extra labels:
predictive entropy of the trained head spikes when the river is physically
ambiguous (piled, jammed, partial ice). Validated as case studies on documented
camera-era jam events, with a false-alarm rate measured on jam-free anchor
winters. A stage-fusion variant (entropy AND stage-rise) is the data-fusion
element. Nobody has published camera-based jam detection with FM features or
predictive entropy (checked 2026-07-02: prior camera work is segmentation and
concentration only: Pei 2023, Ansari IceMaskNet, RIce-Net).

**Why two-in-one.** Confirmed camera-era jams number 2 to 5. Too few to headline
alone. Phenology across 15 station-winters carries the statistics; jams carry
the novelty. One pipeline serves both: C2 reuses C1's trained head outputs.

**Venue.** arXiv first (target Oct 9). Then pick one: Environmental Modelling &
Software (RIce-Net's venue, natural head-to-head) or a WACV/EarthVision-style
workshop for the CV audience. Decide in week 12, not now.

**Reviewer's definition of done.** Clone repo, one command, reproduce: the
6-method timing table, the transfer number, the jam detection table, 5 figures.

---

## 2. Ground rules (inherited from the MVP plan, still binding)

1. Spine first. One station-winter end to end before any scaling.
2. Gates are decisions, not suggestions. Each has a written fallback (Sec. 11).
3. Modeling freezes end of week 8. After that: C2 evaluation, figures, writing.
4. 20 h/week is the budget. Log hours weekly. Two weeks >25% over budget means
   invoke the drop order (Sec. 12), do not "catch up."
5. Fixed decisions stay fixed: clip.frames = 16, stride 4 h, `small` (720 px)
   image tier, ViT-L, mean pooling. No re-extraction for marginal gains.

---

## 3. Station registry: which, why, and the honest caveats

| Key | River, location | NWIS | Archive start | Winters | Role |
|---|---|---|---|---|---|
| `cedarburg` | Milwaukee R, Cedarburg WI | 04086600 | ~2022 | 4 | train |
| `chippewa_bruce` | Chippewa R nr Bruce WI | 05356500 | 2022-10-06 | 4 | train |
| `mohawk_schenectady` | Mohawk R at Freemans Bridge, Schenectady NY | 01354500 | 2024-09-27 | 2 | event test |
| `nebraska_event` (pick in W2) | Platte or Elkhorn cam, NE | see Sec. 5 | 2023+ | 3 | event test |
| `bismarck` | Missouri R at Bismarck ND | 06342500 | 2024-03-13 | 2 | transfer + jam watch |

Camera IDs, ASOS ids, and YAML templates: runbook Task 1. Verify each camera in
week 1 with the per-site query (empty `[]` means no camera at that gage):

```
https://api.waterdata.usgs.gov/nims/v0/cameras?siteId=05356500
```

Config state check (verified 2026-07-02): `configs/stations.yaml` currently
contains only `cedarburg` and `bismarck`, and `image_size` is set to `overlay`.
Week 1 therefore includes: add `chippewa_bruce` and `mohawk_schenectady` using
the runbook Task 1 YAML template, set `image_size: small`, and add
`nebraska_event` after the week-2 audit picks it.

**Why each station.**

- `cedarburg`: RIce-Net site. Their published PAN ResNet50 weights and river
  mask apply directly, so the head-to-head baseline is cheap and exact. 4
  winters of training data on a river with a clean seasonal ice cycle.
- `chippewa_bruce`: second training river, 4 winters, reliable sheet-ice cycle.
  Prevents the head from memorizing one camera. Its winters also serve as
  jam-free anchor winters for the C2 false-alarm rate.
- `mohawk_schenectady`: the strongest event station in the country. USGS runs a
  dedicated Mohawk River Ice Jam Monitoring program on this reach (dashboard:
  ny.water.usgs.gov/maps/mohawk-icejam). Jams are near-annual at channel
  constrictions and locks. Camera era covers winters 2024-25 and 2025-26.
  Bonus viewpoint if needed: `NY_Mohawk_River_at_Lock_9_at_Rotterdam_Junction`
  (NWIS 01354230).
- `nebraska_event`: the Platte-Elkhorn-Loup system jammed in Feb 2025 (warnings)
  and Feb 2026 (confirmed jam near Venice NE, ~1.5 mi long, lowland flooding).
  Which camera actually SAW a jam is unknown until the week-2 audit. Candidate
  gages to sweep for cameras: 06770500 (Platte nr Grand Island), 06799510
  (Elkhorn nr Winslow), 06800500 (Elkhorn at Waterloo), 06796000 (Platte at
  North Bend), 06805500 (Platte at Louisville). Pick the one camera whose frames
  visibly show a jam.
- `bismarck`: RIce-Net site (weights and mask apply), so it is the transfer
  station. Jam watch only. **Correction of a runbook error:** the famous
  National Guard helicopter jam with the emergency declaration was Feb 29,
  2024. The Bismarck camera went live 2024-03-13, two weeks LATER. There was no
  verified spring-2025 Bismarck jam. Do not claim the 2024 event as imaged.
  Check winters 2024-25 and 2025-26 in the week-2 audit; NWS rated jam risk low
  through mid-Jan 2026, so expect Bismarck to contribute anchors, not events.

**Dropped from the old 7-station registry:** `red_cedar_colfax`,
`saginaw_baycity`, `fargo`. Re-add only if you are ahead at week 8 gate.

**Anchor vs event winters.** Every claim in C2 needs both columns filled:

| Type | Definition | Expected count |
|---|---|---|
| Event winters | station-winter with an audited, camera-visible jam | 2 to 5 |
| Anchor winters | station-winter with no documented jam (train winters + quiet test winters) | 10 to 12 |

Detection rate comes from event winters. False-alarm rate comes from anchor
winters. The week-2 audit fills this table; it is Figure 1's foundation.

---

## 4. Label strategy. Answer: no segmentation labels needed.

You never draw ice polygons. Label sources, all cheap:

| Label | Source | Effort | Used for |
|---|---|---|---|
| Daily ice flags | USGS ice qualifier on discharge (scripted: `download_ice_flags`) | 0 h | onset/breakup reference |
| Stage breakpoints | stage CSV (scripted), backwater rise at freeze-up | 0 h | cross-check ice flags |
| Jam events | CRREL database + USGS Mohawk dashboard + NWS/news, 2-of-3 rule | in W2 audit | C2 reference |
| Spot audit | 20 days per station-winter, day-level label by eye, use `labeling_guide/` | ~1 h per winter, ~15 h total | trust in reference |
| Lighting audit | 40 frames (10 sunny AM, 10 sunny PM, 10 overcast, 10 snow) | 1 h | shadow robustness table |
| River ROI mask | SAM2 point prompts, one polygon per camera (notebook exists: `code_base/SAM2_Model1.ipynb`) | 30 min per camera | crop for features |

Segmentation appears only inside the RIce-Net baseline, and there you use their
published weights and masks. Zero new segmentation training.

---

## 5. Week-2 event audit. The step a high-schooler can follow.

This is the make-or-break step for C2. Do it before bulk downloads.

1. **Pull CRREL events.** Go to https://icejam.sec.usace.army.mil/ . Filter
   State = NE, then NY, then ND, then WI, water years 2023 to 2026. View as
   table, Actions > Download, save CSV per state to `data/raw/ice_jams/`.
   Keep columns: jam date, water year, jam type, latitude, longitude, river,
   city, description, damages. Ignore the `Gage number` column, it has known
   data-entry errors.
2. **Distance filter.** Keep events within 15 km of a candidate camera:

   ```python
   from math import radians, sin, cos, asin, sqrt
   def km(lat1, lon1, lat2, lon2):
       dlat, dlon = radians(lat2-lat1), radians(lon2-lon1)
       a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
       return 12742 * asin(sqrt(a))
   ```
3. **Camera-era filter.** Keep events dated after the camera `archive_start`
   from the siteId query. This is why Bismarck's 2024 Guard jam drops out.
4. **Stage cross-check.** For each surviving event, pull instantaneous stage
   ±10 days around the date (downloader exists). A real jam shows an abrupt
   stage rise and often an ice-affected qualifier. No stage signature within
   ±3 days: mark the event weak.
5. **Eyeball the frames.** Download frames for event date ±3 days only
   (targeted, not bulk). Look for piled, ridged, chaotic ice, not smooth sheet.
   Save 3 example frames per event to `docs/event_audit/`. If you cannot see
   the jam, the camera cannot either. Drop it.
6. **Verdict table** in `docs/event_audit.md`: event, station, date, stage
   signature (y/n), visible in frames (y/n), keep/drop.

**Gate 2 (end of week 2), decide and write it down:**

- 2+ kept events across 2+ stations: C2 proceeds as planned.
- Exactly 1 kept event: C2 becomes one deep case study plus breakup ice-run
  detection (every station-winter has a breakup run; entropy spikes there too;
  jams are the motivating extreme case).
- 0 kept events: C2 drops to a Discussion subsection. Paper reverts to C1-only
  MVP framing. Painful but pre-decided, no drift.

Also in W2: pick `nebraska_event` = the NE camera with the best kept event.

---

## 6. Method, concrete

**Pipeline (per station-winter).** Hourly frames Nov 1 to May 15, `small` tier
-> QC (drop night by solar elevation < 5 deg via `astral`, drop glare by
saturated-pixel fraction, drop moved-view frames by SSIM to a reference frame)
-> 16-frame clips, stride 4 h -> frozen V-JEPA 2 ViT-L -> mean-pooled 1024-d
embedding per clip -> temporal head (small GRU or 2-layer Transformer, air
temperature concatenated per step) -> per-clip p(ice state).

**Onset and breakup (C1).** Smooth p(ice) with a short median filter. Onset =
first sustained upward crossing (hysteresis: up-threshold 0.7, down 0.3, min
dwell 48 h). Breakup = last sustained downward crossing. Score timing error in
hours against the ice-flag reference; also event F1 at 72 h tolerance
(`evaluation/metrics.py` self-test exists, run it first).

**Jam detection (C2), predictive entropy.**

```python
probs = model.predict_proba(embeddings)          # (T, C)
H = -(probs * np.log(probs + 1e-8)).sum(axis=1)  # (T,)
```

- Threshold `tau_H` = 99th percentile of H on anchor train winters. Freeze it
  before touching test stations. No post-hoc tuning, say so in the paper.
- Candidate window: H > tau_H for >= 6 consecutive clips (24 h at stride 4 h),
  merge gaps < 8 h. The 24 h persistence is what separates jams (days) from
  breakup ice runs (hours) and from any diurnal artifact.
- Hit: window overlaps an audited event date within ±72 h.
- Report: hits, misses, false alarms per anchor winter, lead time relative to
  the CRREL/NWS date.

**Spatial-disorder entropy (C2 robustness variant, training-free).** K-means
(K = 8) on DINOv2 patch tokens inside the river ROI, fit on 10k random train
patches. Per frame: histogram over clusters, Shannon entropy, daily median.
Physically: jams are spatially chaotic, sheet ice and open water are uniform.
If it agrees with predictive entropy, the claim is not an artifact of one head.

**Stage fusion (the data-fusion element).** Variant detector: entropy window
AND stage residual rise > 2 sigma within 48 h. Report detection and false-alarm
rates with and without fusion. Stage is already downloaded; cost is one script.
Leakage caveat: the week-2 audit uses stage spikes to confirm events, so a
stage-fused detector is partially circular. Entropy-only is therefore the
primary C2 claim; fusion goes in the ablation as a FAR-reduction variant.

**Encoders.** V-JEPA 2 ViT-L is the method. DINOv2-L, identical head and
splits, is the H2 ablation. Gate B: if DINOv2 ties or wins, reframe honestly as
"temporal modeling beats per-frame thresholding" and report it.

---

## 7. Pixel-to-area. Answer: you do not need it.

Every headline metric (timing error, event F1, jam hit rate, false alarms) is
computed on time series and event dates, never on square meters. Ice fraction
inside a fixed camera ROI is scale-free, and for a fixed camera it is monotone
in true area. RIce-Net itself reports percent coverage inside an image-space
mask, so the head-to-head is like for like. State this in one paragraph in the
paper and move on.

Optional 3-hour demo, week 14 only if ahead of schedule, one station
(Bismarck, bridge piers make good control points): mark 4 to 6 fixed points
visible in both a frame and Google Earth (pier bases, ramp corners), convert
lat/lon to UTM meters (`pyproj`), `cv2.findHomography(img_pts, utm_pts)`, warp
the ROI mask to a 1 m grid, count pixels for m². Sanity check: computed river
width within 10% of the width measured in Google Earth. If reviewers ask for
metric area, this is the appendix.

---

## 8. Shadow vs ice. The failure modes and the cheap answers.

Failure modes: low-sun shadows on water read dark and can mimic wet ice edges;
specular glint reads bright like snow-covered ice; night and IR frames shift
the whole distribution.

Defenses, in the order they act:

1. QC drops night (solar elevation) and glare (saturation fraction) before
   anything else sees the frame.
2. FM features, not pixels. This is the point of Gate A: a kNN/linear probe on
   embeddings must separate ice from water across lighting strata. If accuracy
   drops more than 10 points on any stratum (sunny AM, sunny PM, overcast,
   snow), fix the ROI crop first, then add that stratum's days to the probe
   train set. Day labels only, still no segmentation.
3. Temporal aggregation. Daily series use the median over 10:00 to 14:00 local
   clips, when shadows are shortest.
4. Persistence rules. A shadow artifact lasts hours and recurs with the sun; a
   24 h sustained entropy window does not come from a shadow.
5. Paper evidence: the 40-frame lighting audit becomes a small confusion table.
   One hour of labeling buys the reviewer's trust.

---

## 9. Baselines and metrics

| Method | Source | Cost cap |
|---|---|---|
| RIce-Net threshold (15%/20%, 8 h) | published PAN ResNet50 weights + masks, HydroShare | 2 days, then fallback |
| Degree-day (AFDD/ATDD) | new `baselines/degreeday.py`, CPU | 0.5 day |
| BOCPD change-point | implemented, needs runner | 1 day |
| Ours (V-JEPA + head) | spine | n/a |
| Ours (DINOv2 + head) | H2 ablation | extraction + retrain |

RIce-Net fallback (pre-decided, MVP rule): if the HydroShare weights fight you
past the 2-day cap, apply their threshold rule to your own probe-based coverage
estimate and label it "threshold rule, our features" in the table.

Metrics: onset MAE (h), breakup MAE (h), event F1@72h, transfer gap on
Bismarck, jam hits@72h, false alarms per anchor winter, lead time (h).

---

## 10. Week-by-week. 20 h each, Mon Jul 6 to Sun Oct 11.

| Wk | Dates | Work | Definition of done |
|---|---|---|---|
| 1 | Jul 6-12 | env self-test; add missing stations to `stations.yaml`, set `image_size: small`; smoke-test stage/temp/image downloaders on one winter; ice-flag gate ALL stations ALL winters; CCAST V-JEPA forward pass on 1 clip | yaml updated; flags table written; 3 downloaders produce rows; GPU job ran |
| 2 | Jul 13-19 | event audit (Sec. 5); pick `nebraska_event`; Gate 2 decision; start Cedarburg winter-1 download in background | `docs/event_audit.md` verdict table; NE station chosen |
| 3 | Jul 20-26 | `assemble_clips`: label-from-flags+stage, night/glare/SSIM QC; clip manifest winter 1 | manifest CSV with QC drop stats |
| 4 | Jul 27-Aug 2 | V-JEPA embeddings winter 1; Gate A lighting probe; wire `load_sequence`; train; first onset/breakup print | spine alive, Gate A numbers logged |
| 5 | Aug 3-9 | bulk downloads all stations/winters; queue embeddings on CCAST | image tree complete; cache filling |
| 6 | Aug 10-16 | embeddings done; train on Cedarburg+Chippewa, newest winter of each held out; eval | preliminary H1 table |
| 7 | Aug 17-23 | degree-day + BOCPD baselines | both rows filled |
| 8 | Aug 24-30 | RIce-Net baseline (2-day cap); transfer eval on Bismarck; FREEZE modeling | 6-method table v1 + transfer number |
| 9 | Aug 31-Sep 6 | `evaluation/entropy_jam.py`; tau_H frozen on anchors; detection + FAR + stage-fusion variant | jam table done |
| 10 | Sep 7-13 | DINOv2 ablation, same splits; spatial-entropy variant if time | H2 row filled |
| 11 | Sep 14-20 | 5 figures, each from a script | all pngs regenerate |
| 12 | Sep 21-27 | write Results + Methods first; pick venue | draft with real numbers |
| 13 | Sep 28-Oct 4 | write Intro/Related/Discussion; repo cleanup; leakage red-team (tau_H frozen? splits clean? no test tuning?) | repo public-ready |
| 14 | Oct 5-11 | polish; arXiv by Oct 9; SLB application package same week | preprint live |

Figures (5, cut from the runbook's 8): (1) map + anchor/event winter timeline,
(2) method schematic with the entropy branch, (3) 6-method ablation table,
(4) one winter's embedding projection + H(t) with the jam window shaded, the
money figure, (5) jam case study: frames, stage spike, temperature trace,
entropy spike, four panels.

---

## 11. Gates and their pre-written fallbacks

| Gate | When | Pass | Fallback |
|---|---|---|---|
| Ice flags | Wk 1 | contiguous flagged block per winter | drop offending station, or manual dates from spot audit for test-only stations |
| Event audit | Wk 2 | 2+ visible camera-era jams | 1 event: case study + breakup-run detection. 0: C1-only paper (Sec. 5) |
| Gate A probe | Wk 4 | ice/water separate across lighting | fix ROI crop, re-probe; if still failing, DINOv2 features instead of V-JEPA and re-gate |
| Gate B encoders | Wk 10 | any outcome | report honestly either way |
| CCAST queue death | any | jobs run within 48 h | DINOv2 on Colab becomes the method; V-JEPA becomes the ablation on a subset |

## 12. Drop order when (not if) you fall behind

Drop in this order, never reorder: (1) homography demo, (2) spatial-entropy
variant, (3) stage-fusion ablation, (4) oldest Chippewa winter, (5) DINOv2
ablation shrinks to one-station mini-ablation. Never drop: the spine, the event
audit, the RIce-Net comparison (or its named fallback), the preprint.

## 13. Deliberately out of scope (future-work fodder, do not start)

SAR or satellite fusion, forecast head, ice velocity via optical flow,
cross-station contrastive adaptation, dataset DOI release with datasheets,
stations 6 to 8, any fine-tuning of encoders. Each is a sentence in the
Discussion, not a work item.

## 14. Why this beats the RIce-Net paper, in one paragraph you can reuse

RIce-Net detects ice by thresholding per-frame coverage from a supervised
segmenter trained on hand-labeled masks. It cannot time discrete mechanical
events, and degree-day methods cannot either, because jams are not thermal.
This work replaces per-frame thresholds with temporal reasoning over frozen
foundation-model features, needs no segmentation labels, transfers across
stations, and detects jams as entropy anomalies from the same trained head at
zero additional labeling cost, validated against CRREL-documented events.
