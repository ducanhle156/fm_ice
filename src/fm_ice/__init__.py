"""FM_ice: temporal foundation-model detection of river-ice onset and breakup.

Package layout:
    fm_ice.data        download + assemble raw data (images, stage, temp, flags)
    fm_ice.features    frozen V-JEPA / DINOv2 embedding extraction
    fm_ice.baselines   RIce-Net threshold baseline + change-point baselines
    fm_ice.models      temporal head (TCN / transformer) and training
    fm_ice.evaluation  timing error, event F1, covering metric
"""

__version__ = "0.1.0"
