"""
Enhance ULF MRI volumes using pretrained restoration models.

If you use ULF-Synth enhancement in your work, please cite:

    Musah et al. "ULF-Synth: Physics-Guided Ultra-Low-Field MRI
    Enhancement for Pediatric Neuroimaging." arXiv:2605.24625, 2026.
"""

import contextlib
import os
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

_CITED = False


def _citation():
    global _CITED
    if not _CITED:
        _CITED = True
        print(
            "If you use ULF-Synth enhancement in your work, please cite:\n"
            "  Musah et al. \"ULF-Synth: Physics-Guided Ultra-Low-Field MRI\n"
            "  Enhancement for Pediatric Neuroimaging.\" arXiv:2605.24625, 2026.\n",
            flush=True,
        )


def _ensure_nnunet():
    nn_path = str(_BUNDLED_FORK)
    if nn_path not in sys.path:
        sys.path.insert(0, nn_path)
    import nnunetv2  # noqa: F401
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainerMRCT_kspace import (  # noqa: F401
        nnUNetTrainerMRCT_kspace,
    )


def _download_weights(force=False):
    os.makedirs(CACHE_DIR, exist_ok=True)
    new = []
    for rel_path in WEIGHT_FILES:
        dest = os.path.join(CACHE_DIR, rel_path)
        if os.path.exists(dest) and not force:
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        url = f"{WEIGHTS_REPO}/{rel_path}"
        new.append((url, dest))
    if new:
        print("Downloading enhancement weights... ", end="", flush=True)
        for url, dest in new:
            urllib.request.urlretrieve(url, dest)
        print("done.", flush=True)
    return CACHE_DIR


def _run_inference(input_path, output_path, device="cuda"):
    import torch
    os.environ.setdefault("nnUNet_raw", CACHE_DIR)
    os.environ.setdefault("nnUNet_preprocessed", CACHE_DIR)
    os.environ.setdefault("nnUNet_results", CACHE_DIR)
    _ensure_nnunet()
    _download_weights()

    _dev = torch.device(device)
    from nnunetv2.inference.predict_from_raw_data import (
        nnUNetPredictor,
    )

    with open(os.devnull, "w") as _null, contextlib.redirect_stdout(_null):
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=True,
            device=_dev,
            verbose=False,
            allow_tqdm=False,
        )
        predictor.initialize_from_trained_model_folder(
            model_training_output_dir=os.path.join(CACHE_DIR, MODEL_DIR),
            use_folds=("all",),
            checkpoint_name="checkpoint_best.pth",
        )
        predictor.predict_from_files(
            [[input_path]],
            [output_path],
            save_probabilities=False,
            overwrite=True,
            num_processes_preprocessing=2,
            num_processes_segmentation_export=2,
        )

    for f in Path(output_path).parent.glob("predict_from_raw_data_args.json"):
        f.unlink(missing_ok=True)
    for f in Path(output_path).parent.glob("dataset.json"):
        f.unlink(missing_ok=True)
    for f in Path(output_path).parent.glob("plans.json"):
        f.unlink(missing_ok=True)


def enhance_file(input_path, output_path, device="cuda", verbose=True):
    _citation()
    _run_inference(input_path, output_path, device=device)


def enhance_folder(in_dir, out_dir, device="cuda", verbose=True):
    _citation()
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(in_dir) if f.endswith(('.nii', '.nii.gz')))
    if not files:
        raise FileNotFoundError(f"No NIfTI files found in {in_dir}")
    for fname in files:
        input_path = os.path.join(in_dir, fname)
        output_path = os.path.join(out_dir, fname)
        enhance_file(input_path, output_path, device=device, verbose=verbose)
