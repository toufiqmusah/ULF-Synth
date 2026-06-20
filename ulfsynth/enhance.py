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


def _ensure_nnunet():
    """Import nnunetv2 or raise a clear error with install instructions."""
    try:
        import nnunetv2  # noqa: F401
    except ImportError:
        # Look for the bundled fork relative to this package
        here = Path(__file__).resolve().parent
        for candidate in [here.parent / "src" / "nn-translation",
                          here / "_nnunet",
                          Path.cwd() / "src" / "nn-translation"]:
            if (candidate / "setup.py").exists() or (candidate / "pyproject.toml").exists():
                print(f"Installing nnunetv2 fork from {candidate}...")
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "-e", str(candidate)],
                )
                import nnunetv2  # noqa: F401
                return
        raise ImportError(
            "nnunetv2 is required for enhancement. Install it with:\n\n"
            f"    pip install -e {here.parent / 'src' / 'nn-translation'}\n\n"
            "Or if you installed ulfsynth from PyPI:\n\n"
            "    pip install ulfsynth[enhance]\n"
            "    pip install nnunetv2\n"
        )


def _download_weights(force=False):
    """Download model weights from HuggingFace to cache directory."""
    weights_dir = os.path.join(CACHE_DIR, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    missing = []
    for fname in WEIGHT_FILES:
        fpath = os.path.join(weights_dir, fname)
        if not os.path.isfile(fpath) or force:
            missing.append(fname)

    if missing:
        for fname in missing:
            url = f"{WEIGHTS_REPO}/{fname}"
            fpath = os.path.join(weights_dir, fname)
            os.makedirs(os.path.dirname(fpath), exist_ok=True)
            print(f"Downloading {fname}...")
            urllib.request.urlretrieve(url, fpath)
        print("Weights downloaded successfully.")
    else:
        print("Weights already cached.")

    return weights_dir


def enhance_file(input_path, output_path, device="cuda", verbose=True):
    """Enhance a single ULF NIfTI file.

    Args:
        input_path: Path to input .nii or .nii.gz file.
        output_path: Path for the enhanced output file.
        device: Device for inference ('cuda', 'cpu', 'mps').
        verbose: Print progress information.
    """
    input_path = os.path.abspath(input_path)
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="ulfsynth_") as tmpdir:
        # nnUNet expects files named {case}_0000.nii.gz
        basename = os.path.basename(input_path)
        if basename.endswith(".nii.gz"):
            stem = basename[:-7]
        else:
            stem = basename[:-4]
        nnunet_input = os.path.join(tmpdir, "input")
        nnunet_output = os.path.join(tmpdir, "output")
        os.makedirs(nnunet_input, exist_ok=True)
        os.makedirs(nnunet_output, exist_ok=True)

        # Copy input with _0000 suffix
        shutil.copy2(input_path, os.path.join(nnunet_input, f"{stem}_0000.nii.gz"))

        # Load model
        weights_dir = _download_weights()
        model_folder = os.path.join(weights_dir, MODEL_DIR)
        _ensure_nnunet()
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        import torch

        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,

            perform_everything_on_device=True,
            device=torch.device(device),
            verbose=verbose,
            allow_tqdm=verbose,
            verbose_preprocessing=verbose,
        )
        predictor.initialize_from_trained_model_folder(
            model_folder,
            use_folds=["all"],
            checkpoint_name="checkpoint_best.pth",
        )

        # Run inference
        if verbose:
            print(f"Enhancing {basename}...")
        predictor.predict_from_files(
            nnunet_input,
            nnunet_output,
            save_probabilities=False,
            overwrite=True,
            num_processes_preprocessing=1,
            num_processes_segmentation_export=1,
            reconstruction_mode="mean",
        )

        # Find the NIfTI output (filter out JSON summaries added by nnUNet)
        out_candidates = [f for f in os.listdir(nnunet_output)
                          if f.endswith('.nii.gz')]
        if not out_candidates:
            raise RuntimeError(
                f"Enhancement produced no NIfTI output. Found: {os.listdir(nnunet_output)}"
            )

        out_file = os.path.join(nnunet_output, out_candidates[0])
        shutil.move(out_file, output_path)

    if verbose:
        print(f"  -> {output_path}")


def enhance_folder(input_dir, output_dir, device="cuda", verbose=True):
    """Enhance all NIfTI files in a directory.

    Args:
        input_dir: Directory containing .nii/.nii.gz files.
        output_dir: Directory for enhanced outputs.
        device: Device for inference.
        verbose: Print progress information.
    """
    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(
        f for f in os.listdir(input_dir)
        if f.endswith(".nii") or f.endswith(".nii.gz")
    )
    if not files:
        raise FileNotFoundError(f"No NIfTI files found in {input_dir}")

    with tempfile.TemporaryDirectory(prefix="ulfsynth_") as tmpdir:
        nnunet_input = os.path.join(tmpdir, "input")
        nnunet_output = os.path.join(tmpdir, "output")
        os.makedirs(nnunet_input, exist_ok=True)
        os.makedirs(nnunet_output, exist_ok=True)

        # Copy all inputs with _0000 suffix
        name_map = {}
        for fname in files:
            if fname.endswith(".nii.gz"):
                stem = fname[:-7]
            else:
                stem = fname[:-4]
            src = os.path.join(input_dir, fname)
            dst = os.path.join(nnunet_input, f"{stem}_0000.nii.gz")
            shutil.copy2(src, dst)
            name_map[stem] = fname

        # Load model
        weights_dir = _download_weights()
        model_folder = os.path.join(weights_dir, MODEL_DIR)

        _ensure_nnunet()
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        import torch

        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=True,
            device=torch.device(device),
            verbose=verbose,
            allow_tqdm=verbose,
            verbose_preprocessing=verbose,
        )
        predictor.initialize_from_trained_model_folder(
            model_folder,
            use_folds=["all"],
            checkpoint_name="checkpoint_best.pth",
        )

        if verbose:
            print(f"Enhancing {len(files)} files...")
        predictor.predict_from_files(
            nnunet_input,
            nnunet_output,
            save_probabilities=False,
            overwrite=True,
            num_processes_preprocessing=1,
            num_processes_segmentation_export=1,
            reconstruction_mode="mean",
        )

        # Rename outputs back (skip JSON summaries)
        for of in os.listdir(nnunet_output):
            if not of.endswith(".nii.gz"):
                continue
            stem = of[:-7].replace("_0000", "")
            original_name = name_map.get(stem, of)
            shutil.move(
                os.path.join(nnunet_output, of),
                os.path.join(output_dir, original_name),
            )

    if verbose:
        print(f"Done. {len(files)} files written to {output_dir}")
