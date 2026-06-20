"""
Enhance ULF MRI volumes using pretrained restoration models.

Downloads weights from HuggingFace on first use, then runs nnU-Net
translation inference to produce enhanced (HF-like) volumes.
"""

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


def _download_weights(force=False):
    """Download pretrained model weights from HuggingFace."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    for rel_path in WEIGHT_FILES:
        dest = os.path.join(CACHE_DIR, rel_path)
        if os.path.exists(dest) and not force:
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        url = f"{WEIGHTS_REPO}/{rel_path}"
        print(f"Downloading {url} -> {dest}")
        urllib.request.urlretrieve(url, dest)
    return CACHE_DIR


def _run_inference(input_path, output_path, device="cuda"):
    """Run nnU-Net inference for a single file."""
    _ensure_nnunet()
    weights_dir = _download_weights()
    os.environ["nnUNet_results"] = weights_dir
    os.environ["nnUNet_raw"] = weights_dir
    os.environ["nnUNet_preprocessed"] = weights_dir

    from nnunetv2.inference.predict_from_raw_data import (
        nnUNetPredictor,
    )

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=device,
        verbose=False,
    )
    predictor.initialize_from_trained_model_folder(
        model_folder=os.path.join(weights_dir, MODEL_DIR),
        use_folds=("all",),
        checkpoint_name="checkpoint_best.pth",
    )
    predictor.predict_from_files(
        [[input_path]],
        [output_path],
        save_probabilities=False,
        overwrite=True,
        num_processes=2,
    )


def enhance_file(input_path, output_path, device="cuda", verbose=True):
    """Enhance a single ULF NIfTI file using the pretrained model."""
    _run_inference(input_path, output_path, device=device)
    if verbose:
        print(f"  {os.path.basename(input_path)} -> {output_path}")


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
