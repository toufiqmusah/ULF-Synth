"""
Enhance ULF MRI volumes using pretrained restoration models.

Downloads weights from HuggingFace on first use, then runs nnU-Net
translation inference to produce enhanced (HF-like) volumes.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "ulfsynth")
WEIGHTS_REPO = "https://huggingface.co/toufiqmusah/ulfsynth-weights/resolve/main"
MODEL_DIR = "Dataset2023/nnUNetTrainerMRCT_kspace__nnResUNetPlans__3d_fullres"
WEIGHT_FILES = [
    f"{MODEL_DIR}/dataset.json",
    f"{MODEL_DIR}/dataset_fingerprint.json",
    f"{MODEL_DIR}/plans.json",
    f"{MODEL_DIR}/fold_all/checkpoint_best.pth",
    f"{MODEL_DIR}/fold_all/checkpoint_final.pth",
]

_BUNDLED_FORK = Path(__file__).resolve().parent / "_nnunet_src"


def _ensure_nnunet():
    """Import nnunetv2 (fork with MRCT kspace trainer), installing from bundled source if needed."""
    try:
        import nnunetv2  # noqa: F401
    except ImportError:
        if _BUNDLED_FORK.is_dir() and (_BUNDLED_FORK / "setup.py").exists():
            print("Installing nnunetv2 fork from bundled source...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", str(_BUNDLED_FORK)],
            )
        else:
            raise ImportError(
                "nnunetv2 (fork with MRCT kspace trainer) is required for enhancement.\n\n"
                "Reinstall ulfsynth with the bundled fork:\n\n"
                "  pip install --force-reinstall 'ulfsynth'\n"
            )
    # Verify the custom trainer is available (not the upstream nnunetv2)
    try:
        from nnunetv2.training.nnUNetTrainer.nnUNetTrainerMRCT_kspace import (  # noqa: F401
            nnUNetTrainerMRCT_kspace,
        )
    except ImportError:
        raise ImportError(
            "You have the upstream nnunetv2 installed, but ulfsynth needs the custom\n"
            "fork with the MRCT kspace trainer.\n\n"
            "  pip uninstall nnunetv2\n"
            "  pip install --force-reinstall 'ulfsynth'\n"
        )