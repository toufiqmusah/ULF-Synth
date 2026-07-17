"""
Enhance ULF MRI volumes using pretrained restoration models.

If you are using ULF-Synth, please cite the following publication:
Musah, T., Calcagno, S., Salanitri, F.P., Li, X., Adewole, M.
and Elbatel, M., 2026. ULF-Synth: Physics-Guided Ultra-Low-Field
MRI Enhancement for Pediatric Neuroimaging.
arXiv preprint arXiv:2605.24625.
"""

import contextlib
import os
import sys
from pathlib import Path

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "ulfsynth")
MODEL_DIR = "Dataset2023/nnUNetTrainerMRCT_kspace__nnResUNetPlans__3d_fullres"
WEIGHT_FILES = [
    f"{MODEL_DIR}/dataset.json",
    f"{MODEL_DIR}/dataset_fingerprint.json",
    f"{MODEL_DIR}/plans.json",
    f"{MODEL_DIR}/fold_all/checkpoint_best.pth",
    f"{MODEL_DIR}/fold_all/checkpoint_final.pth",
]

_BUNDLED_NNUNET = Path(__file__).resolve().parent / "_nnunet_src"
_BUNDLED_DYNARCH = Path(__file__).resolve().parent / "_dynarch_src"

_CITED = False


def _citation():
    global _CITED
    if not _CITED:
        _CITED = True
        print(
    "If you are using ULF-Synth, please cite the following publication: "
    "Musah, T., Calcagno, S., Salanitri, F.P., Li, X., Adewole, M. "
    "and Elbatel, M., 2026. ULF-Synth: Physics-Guided Ultra-Low-Field "
    "MRI Enhancement for Pediatric Neuroimaging. "
    "arXiv preprint arXiv:2605.24625.\n",
            flush=True,
        )


def _ensure_nnunet():
    for p in [str(_BUNDLED_DYNARCH), str(_BUNDLED_NNUNET)]:
        if p not in sys.path:
            sys.path.insert(0, p)
    # Force reimport of bundled packages (evict any pre-loaded system copies)
    for mod in list(sys.modules.keys()):
        if 'dynamic_network_architectures' in mod or 'dynarch' in mod:
            del sys.modules[mod]
    import dynamic_network_architectures  # noqa: F401
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
        new.append(rel_path)
    if new:
        print("Downloading enhancement weights... ", end="", flush=True)
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        from huggingface_hub import hf_hub_download
        for rel_path in new:
            hf_hub_download(
                repo_id="toufiqmusah/ulfsynth-weights",
                filename=rel_path,
                repo_type="model",
                local_dir=CACHE_DIR,
            )
        print("done.", flush=True)
    return CACHE_DIR


def _run_inference(input_path, output_path, device="cuda"):
    import torch
    os.environ.setdefault("nnUNet_raw", CACHE_DIR)
    os.environ.setdefault("nnUNet_preprocessed", CACHE_DIR)
    os.environ.setdefault("nnUNet_results", CACHE_DIR)
    _ensure_nnunet()
    _download_weights()

    # Normalise output path: resolve to absolute, ensure parent exists
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # nnUNet appends file_ending to output_path, so strip it if present
    file_ending = ".nii.gz"
    if output_path.endswith(file_ending):
        output_path = output_path[: -len(file_ending)]
    elif output_path.endswith(".nii"):
        output_path = output_path[: -len(".nii")]

    _dev = torch.device(device)
    from nnunetv2.inference.predict_from_raw_data import (
        nnUNetPredictor,
    )

    with open(os.devnull, "w") as _null, contextlib.redirect_stdout(_null):
        predictor = nnUNetPredictor(
            tile_step_size=1.0,
            use_gaussian=True,
            use_mirroring=False,
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
