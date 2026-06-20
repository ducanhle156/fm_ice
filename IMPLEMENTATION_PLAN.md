# IMPLEMENTATION_PLAN.md

Temporal foundation-model detection of river-ice onset and breakup.
Executable backlog for an AI coding agent and two human owners. Phases run in
order. Two decision gates can redirect the work. Freeze the MVP after Phase 5.

Hypotheses this plan must test:
- H1: temporal modeling on FM embeddings beats per-frame thresholding on timing error.
- H2: video-FM (V-JEPA) beats image-FM (DINOv2) for this task.
- H3: FM embeddings separate ice from open water across lighting better than pixels.
- H4 (optional): a forecast head gives non-trivial lead time before onset.

Repo modules map 1:1 to phases. Each task lists Inputs, Outputs, and Acceptance.
"Done" means the Acceptance check passes, not that code exists.

---

## Phase 0. Environment and repo (0.5 week)

Goal: a reviewer can clone and run, on a laptop for everything except embeddings.

Tasks
- Create venv, `pip install -e .`, confirm `python -m fm_ice.evaluation.metrics` runs.
- Confirm GPU path on CCAST: `transformers` + `torch` import, a V-JEPA 2 checkpoint loads.
- Move the legacy SAM2 notebook to `notebooks/exploratory/` (out of the package).

Acceptance
- `python -c "import fm_ice"` works in the venv.
- `bash scripts/download_all.sh` reaches the dry-run prompt without error.

---

## Phase 1. Data assembly (weeks 1-3, start in week 1)

Goal: aligned clips with exogenous series, plus a reproduced RIce-Net baseline.
This is the hidden time sink. Start it first.

Tasks
- Download all four streams for both stations, all winters (`scripts/download_all.sh`).
- Resolve the cadence decision: confirm the true per-winter cadence from
  `_manifest.csv` timestamps. It is hourly (verified). Fix `clip.frames` and
  `clip.stride_hours` in `configs/pipeline.yaml` and do not change them later.
- Build clips and joins: `fm_ice.data.assemble_clips`. Complete the two TODOs:
  per-clip ice label (from ice flags and stage breakpoints) and QC flags
  (night, glare, occlusion, valid-frame fraction).
- Reproduce the RIce-Net threshold baseline: `fm_ice.baselines.ricenet_baseline`.
  Wire `segment_ice_coverage` to the published PAN weights and the per-site river
  mask from the RIce-Net HydroShare release. The threshold-persistence rule is
  already implemented and unit-tested.

Inputs: configs/stations.yaml, configs/pipeline.yaml.
Outputs: data/raw/*, data/interim/<station>/clips_<winter>.parquet, baseline
onset/breakup dates per station-winter in results/.
Acceptance
- Clip manifest has no time gaps inside a clip beyond `min_valid_frames`.
- Baseline reproduces RIce-Net flag behavior on at least one station-winter,
  agreeing with the USGS ice flag in the same ballpark RIce-Net reported.

---

## Phase 2. Frozen V-JEPA features (weeks 3-4)

Goal: one cached embedding per clip. The only GPU step.

Tasks
- Run `fm_ice.features.extract_embeddings --encoder vjepa2` on CCAST for every
  station-winter (`scripts/extract_embeddings_ccast.sbatch`).
- Verify tensor shapes against the checkpoint (frame count, resolution). Fix the
  attentive-vs-mean pooling choice; attentive probing is the published, stronger
  readout. Mean pooling is the cheap fallback.
- Cache to `data/cache/vjepa2/...`, pull back to the laptop.

Inputs: clip manifests, image files.
Outputs: data/cache/vjepa2/<station>/<winter>.npy + row-aligned index csv.
Acceptance
- Embedding row count equals clip count for every station-winter.
- Re-running is a no-op (cache hit), so the head trains without the GPU.

### GATE A: do FM embeddings separate ice from open water across lighting? (H3)
Probe ice-vs-water on the embeddings under lighting splits (day/night, glare).
- Pass: continue to Phase 3.
- Fail: return to preprocessing (river crop, water mask, clip width) before modeling.

---

## Phase 3. Temporal head (weeks 5-7)

Goal: an ice-state sequence per winter, with onset and breakup read off transitions.

Tasks
- Wire `fm_ice.models.train.load_sequence` to cached embeddings + per-step labels +
  the air-temperature channel.
- Train `TemporalHead` (TCN default) with the MS-TCN smoothing loss to suppress
  flicker. Implement event reading from the predicted state sequence.
- Hold out the test winter (`splits.test_winter`) on the train station.

Inputs: cached embeddings, labeled clip manifest.
Outputs: trained head, predicted ice-state series, predicted onset/breakup dates.
Acceptance (H1)
- On the held-out winter, timing error beats the RIce-Net baseline by a clear
  margin. Report onset and breakup separately, in hours.

---

## Phase 4. Ablations and change-point baselines (weeks 6-8)

Goal: the decisive H2 comparison and the label-free baselines.

Tasks
- DINOv2 ablation: `--encoder dinov2`, identical temporal head, identical splits.
- Change-point baselines on the embedding stream: `fm_ice.baselines.changepoint`
  BOCPD (online) and BEAST (offline). Both implemented; tune `hazard_lambda` to the
  expected segment length in hours and validate on the train winters.
- Keep every knob identical across encoders so the comparison is clean.

Inputs: cached vjepa2 and dinov2 embeddings.
Outputs: head-to-head timing-error table, change-point onset/breakup dates.
Acceptance (H2)
- A single table compares V-JEPA vs DINOv2 vs change-point vs RIce-Net on timing
  error. This is the central figure for a CV audience.

### GATE B: does V-JEPA beat DINOv2?
- Yes: keep the video-FM framing; motion and time help.
- No: reframe honestly as "temporal modeling beats per-frame thresholding." Still
  true, still publishable. Do not bury the negative result.

---

## Phase 5. Transfer test, then MVP freeze (weeks 8-9)

Goal: evidence of generalization, then stop adding scope.

Tasks
- Evaluate the trained head on Bismarck, the entirely held-out station. No retraining.
- Report the transfer gap (performance drop versus the train station).
- Freeze the MVP. After this point, only evaluation, figures, and writing proceed.

Acceptance
- Transfer numbers reported honestly, gap quantified, no leakage of Bismarck into training.

---

## Phase 6. Evaluation and error analysis (weeks 9-11)

Goal: a defensible results section.

Tasks
- Compute all metrics with `fm_ice.evaluation.metrics`: timing error, event F1 at
  tolerance windows (24/48/72 h), per-frame flag agreement and AUC, transfer gap.
- Score against both references: USGS ice flag and stage breakpoints.
- Error analysis: where it fails (glare, frazil, night, partial ice), with examples.
- H3 figure: UMAP plus a linear probe under lighting splits.

Acceptance
- Every headline claim has a number and a reference. Figures regenerate from a script.

---

## Phase 7. Writeup and release (weeks 10-12)

Goal: arXiv preprint and a repo a reviewer can run end to end.

Tasks
- Paper: method, the H2 ablation as the central figure, honest H2 outcome, transfer.
- Release: data-prep scripts, cached embeddings, figure scripts, configs, one-command repro.
- README with exact commands. The repo will be read as closely as the PDF.

Acceptance (success criteria from the research plan)
- Beats RIce-Net on onset and breakup timing error by a clear margin.
- Clean H2 ablation reported either way.
- Demonstrated transfer to Bismarck.
- Reproducible end to end.

### H4 (optional stretch, only if Phase 1-5 finished early)
Add a forecast head that predicts the next embedding (or ice state) and report a
single, clearly caveated lead-time figure. Cut it the moment it threatens the freeze.

---

## Owner split (from the research plan)
- Person A: data assembly and alignment, RIce-Net baseline, change-point baselines, transfer.
- Person B: V-JEPA pipeline, embedding analysis (H3), temporal head, DINOv2 ablation.
- Both: evaluation, figures, error analysis, writing, code release.

## Risk reminders
- Label scarcity: lean on frozen features, a light head, and the label-free
  change-point baselines. Start data in week 1.
- Negative H2: reframe, do not hide. The threshold-vs-temporal result still holds.
- Scope creep: H4 is the only optional task. Freeze the MVP after Phase 5.
