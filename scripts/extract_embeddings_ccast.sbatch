#!/usr/bin/env bash
#SBATCH --job-name=fmice_embed
#SBATCH --partition=gpu              # adjust to the CCAST GPU partition name
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=results/embed_%j.log

# Frozen embedding extraction on a CCAST GPU node. Edit module loads to match
# the cluster. This is the ONLY GPU step; everything else runs on CPU.
set -euo pipefail

module purge
module load python/3.10 cuda/12.1 || true     # adjust to available CCAST modules

source .venv/bin/activate                      # or: conda activate fm_ice

# Cache must be on fast scratch then synced back, if your scratch is separate.
# NOTE: only the three "paper winters" have assembled clips + images so far
# (cedarburg 2022-2023, 2023-2024; bismarck 2024-2025). The other lines below
# need their images downloaded + assemble_clips run first, or they will no-op.
python -m fm_ice.features.extract_embeddings --station cedarburg --winter 2022-2023 --encoder vjepa2 --device cuda
python -m fm_ice.features.extract_embeddings --station cedarburg --winter 2023-2024 --encoder vjepa2 --device cuda
python -m fm_ice.features.extract_embeddings --station cedarburg --winter 2024-2025 --encoder vjepa2 --device cuda
python -m fm_ice.features.extract_embeddings --station cedarburg --winter 2025-2026 --encoder vjepa2 --device cuda
python -m fm_ice.features.extract_embeddings --station bismarck  --winter 2024-2025 --encoder vjepa2 --device cuda
python -m fm_ice.features.extract_embeddings --station bismarck  --winter 2025-2026 --encoder vjepa2 --device cuda

# Repeat with --encoder dinov2 for the H2 ablation.
echo "Embeddings cached under data/cache/. Pull these back to your laptop; the head trains on CPU."

# --- RIce-Net threshold baseline (deferred here from Phase 1) ----------------
# torch + segmentation-models-pytorch are already installed for the encoder, so
# run the anchor baseline in the same session. Only Cedarburg ships a river mask
# (Bismarck would need a hand-digitized polygon). GPU optional; input is 1152x640.
python -m fm_ice.baselines.ricenet_baseline --download   # ~93 MB HydroShare weights (CC BY 4.0)
python -m fm_ice.baselines.ricenet_baseline --station cedarburg --winter 2022-2023
python -m fm_ice.baselines.ricenet_baseline --station cedarburg --winter 2023-2024
echo "RIce-Net onset/breakup -> results/ricenet_events.csv. Compare against results/reference_events.csv."
