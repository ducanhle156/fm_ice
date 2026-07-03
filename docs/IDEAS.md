# Ideas ledger (parking lot)

Rule: nothing in this file enters the plan before the week-8 freeze review.
Two exceptions only: a gate fails and its written fallback points here, or a
reviewer asks. New ideas get one line here and zero hours. This file exists so
good ideas stop competing with execution.

## Cheap if asked (post-freeze, roughly a day each, embeddings are cached)

- Head ablation: GRU vs 2-layer Transformer, same cached embeddings and splits.
- Stride-1 h re-scoring of event timing (compute-only, finer resolution).
- Second Mohawk viewpoint (Lock 9, NWIS 01354230): does the jam detection
  replicate on a different camera of the same reach? Nice robustness row.

## Paper v2 / journal extension (after the preprint, not before)

- Live deployment, winter 2026-27: run the frozen detector in real time from
  Nov 2026 and report prospective performance. Strongest possible SLB demo
  (deployment, not just benchmarks) and the natural journal-version delta.
- Forecast head (predict onset/breakup ahead of time). Needs more events than
  15 station-winters give. Was cut for a reason.
- SAR / Sentinel-2 fusion for cloud-independent coverage.
- Cross-station contrastive adaptation of the frozen features.
- Re-add stations: fargo, saginaw_baycity, red_cedar_colfax; dataset release
  with DOI and datasheet.

## Rejected, with reasons (do not resurrect silently)

- Fine-tuning the encoders: kills the label-light story, GPU cost, and the
  frozen-feature claim is the paper.
- Discharge as a model input: winter discharge is ice-corrupted and is the
  source of the ice-flag reference. Using it as input leaks the label.
- Stage as a headline detector input: partially circular with the event audit
  (see plan Sec. 6 caveat). Ablation only.
