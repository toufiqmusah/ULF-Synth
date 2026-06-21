"""
Enhance ULF MRI volumes using pretrained restoration models.
"""

import contextlib
import os
import subprocess
import sys
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
    try:
        import nnunetv2  # noqa: F401
    except ImportError:
        if _BUNDLED_FORK.is_dir() and (_BUNDLED_FORK / "setup.py").exists():
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "--quiet", str(_BUNDLED_FORK)],
            )
        else:
            raise ImportError(
                "Required translation module not found.\n"
                "Reinstall ulfsynth:\n\n"
                "  pip install --force-reinstall 'ulfsynth'\n"
            )
    try:
        from nnunetv2.training.nnUNetTrainer.nnUNetTrainerMRCT_kspace import (  # noqa: F401
            nnUNetTrainerMRCT_kspace,
        )
    except ImportError:
        raise ImportError(
            "Incompatible translation module version.\n"
            "Reinstall ulfsynth:\n\n"
            "  pip install --force-reinstall 'ulfsynth'\n"
        )


def _download_weights(force=False):
    os.makedirs(CACHE_DIR, exist_ok=True)
    for rel_path in WEIGHT_FILES:
        dest = os.path.join(CACHE_DIR, rel_path)
        if os.path.exists(dest) and not force:
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        url = f"{WEIGHTS_REPO}/{rel_path}"
        urllib.request.urlretrieve(url, dest)
    return CACHE_DIR


def _run_inference(input_path, output_path, device="cuda"):
    import torch
    _ensure_nnunet()
    weights_dir = _download_weights()

    from nnunetv2.inference.predict_from_raw_data import (
        nnUNetPredictor,
    )

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device(device),
        verbose=False,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        model_folder=os.path.join(weights_dir, MODEL_DIR),
        use_folds=("all",),
        checkpoint_name="checkpoint_best.pth",
    )
    with open(os.devnull, "w") as _null, contextlib.redirect_stdout(_null):
        predictor.predict_from_files(
            [[input_path]],
            [output_path],
            save_probabilities=False,
            overwrite=True,
            num_processes=2,
        )


def enhance_file(input_path, output_path, device="cuda", verbose=True):
    _run_inference(input_path, output_path, device=device)


def enhance_folder(in_dir, out_dir, device="cuda", verbose=True):
    """Enhance all NIfTI files in a folder."""
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(in_dir) if f.endswith(('.nii', '.nii.gz')))
    if not files:
        raise FileNotFoundError(f"No NIfTI files found in {in_dir}")
    for fname in files:
        input_path = os.path.join(in_dir, fname)
        output_path = os.path.join(out_dir, fname)
        enhance_file(input_path, output_path, device=device, verbose=verbose)
