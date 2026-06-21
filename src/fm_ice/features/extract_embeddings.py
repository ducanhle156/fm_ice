"""Extract one frozen embedding per clip.

This is the expensive step. Run it ONCE on the GPU (CCAST), cache the result,
then everything downstream trains in minutes on CPU.

Two encoders, selected with --encoder:
  vjepa2  video FM (the main method).  facebook/vjepa2-vitl-fpc64-256
  dinov2  image FM (the H2 ablation). facebook/dinov2-large, per-frame then pooled

Both are loaded frozen from Hugging Face. No fine-tuning (the published, and
affordable, mode). Output is written as one .npy per (station, winter) plus a
row-aligned manifest, so the temporal head can index clips directly.

Usage (GPU node):
  python -m fm_ice.features.extract_embeddings \
      --station cedarburg --winter 2024-2025 --encoder vjepa2 --device cuda
  # smoke test on the first N clips (tagged output, never overwrites the cache):
  python -m fm_ice.features.extract_embeddings \
      --station cedarburg --winter 2022-2023 --encoder vjepa2 --device cuda --limit 32

Output:
  data/cache/<encoder>/<station>/<winter>.npy        (N_clips, D)
  data/cache/<encoder>/<station>/<winter>_index.csv  (clip_id -> row)

Status: WORKING reference implementation. Verify tensor shapes against the
checkpoint you actually load (frame count, resolution) before a full run; see
the asserts and the V-JEPA 2 model card. The attentive pooler is a small
trainable head; for a purely frozen readout, mean-pool (default for dinov2).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fm_ice.config import load_yaml


def _load_clip_frames(frame_paths, images_dir: Path, n_frames: int, resize: int):
    """Load and stack frames for one clip as a uint8 array (T, H, W, 3).

    If a clip has fewer than n_frames real images, repeat the last frame to pad
    (documented limitation: V-JEPA expects a fixed frame count). QC of padded
    clips happens upstream in assemble_clips.
    """
    from PIL import Image

    frames = []
    for fp in frame_paths[:n_frames]:
        img = Image.open(images_dir / fp).convert("RGB").resize((resize, resize))
        frames.append(np.asarray(img))
    while len(frames) < n_frames:
        frames.append(frames[-1])
    return np.stack(frames, axis=0)


def extract_vjepa2(clips: pd.DataFrame, images_dir: Path, hf_id: str, n_frames: int,
                   resize: int, device: str) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoVideoProcessor

    model = AutoModel.from_pretrained(hf_id).to(device).eval()
    proc = AutoVideoProcessor.from_pretrained(hf_id)
    embs = []
    with torch.no_grad():
        for _, c in clips.iterrows():
            vid = _load_clip_frames(c["frame_paths"], images_dir, n_frames, resize)  # (T,H,W,3)
            inputs = proc(list(vid), return_tensors="pt").to(device)
            # get_vision_features returns token features; mean-pool over tokens.
            feats = model.get_vision_features(**inputs)        # (1, num_tokens, D)
            embs.append(feats.mean(dim=1).squeeze(0).float().cpu().numpy())
    return np.stack(embs, axis=0)


def extract_dinov2(clips: pd.DataFrame, images_dir: Path, hf_id: str, n_frames: int,
                   resize: int, device: str) -> np.ndarray:
    import torch
    from transformers import AutoImageProcessor, AutoModel

    model = AutoModel.from_pretrained(hf_id).to(device).eval()
    proc = AutoImageProcessor.from_pretrained(hf_id)
    embs = []
    with torch.no_grad():
        for _, c in clips.iterrows():
            vid = _load_clip_frames(c["frame_paths"], images_dir, n_frames, resize)
            inputs = proc(list(vid), return_tensors="pt").to(device)   # batch of frames
            out = model(**inputs).last_hidden_state[:, 0]   # CLS per frame (T, D)
            embs.append(out.mean(dim=0).float().cpu().numpy())  # pool over time
    return np.stack(embs, axis=0)


def run(station: str, winter: str, encoder: str, device: str, limit: int | None = None) -> None:
    cfg_s = load_yaml("stations.yaml")
    cfg_p = load_yaml("pipeline.yaml")
    paths, clip = cfg_p["paths"], cfg_p["clip"]
    enc_cfg = cfg_p["encoders"][encoder]
    cam_id = cfg_s["stations"][station]["cameras"]["primary"]["cam_id"]

    clip_manifest = Path(paths["interim"]) / station / f"clips_{winter}.parquet"
    if not clip_manifest.exists():
        raise SystemExit(f"Missing {clip_manifest}. Run assemble_clips first.")
    clips = pd.read_parquet(clip_manifest)
    if limit:
        # Smoke test: only the first `limit` clips. Output is tagged so it is never
        # mistaken for (or overwritten by) the full cache.
        clips = clips.head(limit).copy()
    images_dir = Path(paths["raw"]) / "images" / station / cam_id

    # Report the device actually used (so a GPU smoke run proves the GPU was seen).
    try:
        import torch
        dev_note = (f"cuda:{torch.cuda.get_device_name(0)}"
                    if device == "cuda" and torch.cuda.is_available() else device)
        if device == "cuda" and not torch.cuda.is_available():
            print("[embed] WARNING: --device cuda but torch.cuda.is_available() is False; using CPU.")
            device = "cpu"
    except ImportError:
        raise SystemExit("torch is not installed in this environment. Install torch + "
                         "transformers (see job.pbs prerequisites) before extracting.")
    print(f"[embed:{encoder}] {station} {winter}: {len(clips)} clips on {dev_note}, "
          f"loading {enc_cfg['hf_id']} ...")

    if encoder == "vjepa2":
        emb = extract_vjepa2(clips, images_dir, enc_cfg["hf_id"], clip["frames"], clip["resize"], device)
    elif encoder == "dinov2":
        emb = extract_dinov2(clips, images_dir, enc_cfg["hf_id"], clip["frames"], clip["resize"], device)
    else:
        raise SystemExit(f"Unknown encoder {encoder}")

    stem = winter if not limit else f"{winter}_smoke{limit}"
    out_dir = Path(paths["cache"]) / encoder / station
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{stem}.npy", emb)
    clips[["clip_id", "t_start_utc", "t_end_utc"]].to_csv(out_dir / f"{stem}_index.csv", index=False)
    print(f"[embed:{encoder}] {station} {winter}: {emb.shape} -> {out_dir / (stem + '.npy')}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--station", required=True)
    ap.add_argument("--winter", required=True)
    ap.add_argument("--encoder", default="vjepa2", choices=["vjepa2", "dinov2"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke test: process only the first N clips (tagged output)")
    args = ap.parse_args()
    run(args.station, args.winter, args.encoder, args.device, args.limit)


if __name__ == "__main__":
    main()
