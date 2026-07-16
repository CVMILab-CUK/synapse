from .eeg_AE import *
try:
    # Legacy DreamDiffusion-style LDM; depends on optional ldm/taming/clip packages.
    from .eeg_LDM import *
except ImportError as _e:  # pragma: no cover
    import warnings
    warnings.warn(f"Skipping legacy models.eeg_LDM (optional deps missing): {_e}")
from .eeg_LDM2 import *

from .clip_embedder import *
from .eeg_diffusion_pipeline import *