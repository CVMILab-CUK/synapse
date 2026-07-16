import warnings
from .base_trainer import *
try:
    # Legacy AE trainer; depends on optional dc_ldm package.
    from .eeg_ae_trainer import *
except ImportError as _e:  # pragma: no cover
    warnings.warn(f"Skipping legacy trainer.eeg_ae_trainer (optional deps missing): {_e}")
try:
    # Legacy DreamDiffusion LDM trainer; depends on optional dc_ldm/eLDM packages.
    from .eeg_ldm_trainer import *
except ImportError as _e:  # pragma: no cover
    warnings.warn(f"Skipping legacy trainer.eeg_ldm_trainer (optional deps missing): {_e}")