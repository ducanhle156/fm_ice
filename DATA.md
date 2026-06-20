# DATA.md — what to download, from where, how much

All facts below were verified against the live USGS HIVIS API on 2026-06-20.
Run any download with `--dry-run` first; the script prints exact image counts
and volume before pulling anything.

## 1. Stations (verified)

| Role | Key | NWIS gauge | Primary camera ID | Cadence | Archive starts | Winters |
|---|---|---|---|---|---|---|
| Train | `cedarburg` | 04086600 | `WI_Milwaukee_River_near_Cedarburg` | hourly (24/7) | 2022-09-12 | 4 |
| Transfer | `bismarck` | 06342500 | `ND_Missouri_River_at_Bismarck` | hourly (24/7) | 2024-03-13 | 2 |

Cedarburg also has a second camera `WI_Milwaukee_River_near_Cedarburg_DOWNSTREAM`
(daylight only, from 2023-01-06). Treat it as a bonus viewpoint, not the main feed.

Why these two: they are the RIce-Net stations, so results are directly comparable.
Cedarburg has the longer record, so it trains. Bismarck is the entirely held-out
station for the transfer test.

## 2. Two findings that change the plan

1. Cadence is hourly, not 15-min. Both cameras report `ingest.intr = 60`, and the
   sample filenames sit at :01 past the hour. The research plan's "16-frame,
   4-hour clip" assumed 15-min images. That mapping is wrong. The repo redefines a
   clip as N consecutive hourly frames (`configs/pipeline.yaml: clip`). Decide the
   clip width once and keep it fixed, because the V-JEPA checkpoint wants a fixed
   frame count. See IMPLEMENTATION_PLAN.md Phase 1.
2. Bismarck has only ~2 winters of imagery. That is enough to test transfer, not to
   train. Do not split Bismarck into train/test.

## 3. The four data streams

| Stream | Source | Service | Script |
|---|---|---|---|
| Camera images | USGS HIVIS / NIMS | `api.waterdata.usgs.gov/nims/v0` + public S3 | `fm_ice.data.download_images` |
| River stage (00065) | USGS NWIS | Instantaneous Values (IV) | `fm_ice.data.download_stage` |
| Air temperature | IEM ASOS (nearest airport) | `mesonet.agron.iastate.edu` | `fm_ice.data.download_temperature` |
| Ice flag reference | USGS NWIS | Daily Values (DV) ice qualifier | `fm_ice.data.download_ice_flags` |

Nearest ASOS for temperature: Cedarburg uses `MWC` (Milwaukee Timmerman), Bismarck
uses `BIS` (co-located). Override with `--asos` if you prefer a closer station.

## 4. Volume estimate (plan for this)

At hourly cadence over a Nov 1 to May 15 season (~196 days), a 24/7 camera yields
roughly 4,000 to 4,700 images per winter. Full-size (`overlay`) JPGs are ~0.27 MB.

| Selection | Images (approx) | Size at `overlay` | Size at `small` (720px) |
|---|---|---|---|
| Cedarburg primary, 4 winters | ~18,000 | ~5 GB | ~1.1 GB |
| Bismarck primary, 2 winters | ~9,000 | ~2.5 GB | ~0.6 GB |
| Cedarburg DOWNSTREAM, 3 winters | ~6,000 | ~1.6 GB | ~0.4 GB |
| Core set (both primaries) | ~27,000 | ~7.5 GB | ~1.7 GB |

Stage, temperature, and ice flags are tiny (a few MB total). The images dominate.
Use `--size small` if disk or transfer to CCAST is a concern; 720px is enough for
a 256px encoder input.

## 5. Commands

```bash
# See counts and volume first (no download):
python -m fm_ice.data.download_images --station cedarburg --all-winters --dry-run

# One winter, full size:
python -m fm_ice.data.download_images --station cedarburg --winter 2024-2025

# Everything, end to end (prompts before the big image pull):
bash scripts/download_all.sh
```

Per stream, single winter:

```bash
python -m fm_ice.data.download_stage       --station cedarburg --winter 2024-2025
python -m fm_ice.data.download_temperature --station cedarburg --winter 2024-2025
python -m fm_ice.data.download_ice_flags   --station cedarburg --winter 2024-2025
```

## 6. Folder layout after download

```
data/
  raw/
    images/cedarburg/WI_Milwaukee_River_near_Cedarburg/
        WI_..._2025-01-12T08-01-03Z.jpg ...
        _manifest.csv                 # filename, timestamp_utc, url, bytes
    stage/cedarburg/04086600_2024-2025_iv.csv
    temperature/cedarburg/MWC_2024-2025_tmp.csv
    ice_flags/cedarburg/04086600_2024-2025_iceflag.csv
  interim/cedarburg/clips_2024-2025.parquet   # after assemble_clips
  cache/vjepa2/cedarburg/2024-2025.npy        # after extract_embeddings (GPU)
```

## 7. Caveats and checks

- API key: the `/cameras` and `/listFiles` endpoints currently serve unauthenticated
  low-volume requests. If you hit `OVER_RATE_LIMIT`, get a free key at
  https://api.waterdata.usgs.gov/signup/ and `export USGS_API_KEY=...`. The image
  files themselves are on public S3 and need no key.
- Timezones: HIVIS filenames are UTC (trailing Z). Both stations are US/Central.
  Everything is aligned in UTC; convert only for reporting.
- Night and glare: a 24/7 camera produces dark night frames. Keep them for now; flag
  and handle them in clip QC (assemble_clips TODO), not at download time.
- Ice flag reference is coarse (daily) and can lag. Treat it as one reference among
  several. Cross-check against stage breakpoints. Do not treat it as ground truth.
- The download scripts were built against the verified API specs but could not be
  run from this environment (no outbound network here). Smoke-test them once on a
  networked machine before a full pull; see CLAUDE.md "First run".
