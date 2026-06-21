# Enhancement

The enhancement module restores ULF volumes using a pretrained
nnUNet-based translation model trained with the ULF-Synth simulation
pipeline (synthetic paired data).

## Usage

```python
from ulfsynth.enhance import enhance_file, enhance_folder

# Single file (CPU)
enhance_file("ulf_scan.nii.gz", "enhanced.nii.gz", device="cpu")

# Batch folder (GPU)
enhance_folder("ulf_scans/", "enhanced_scans/", device="cuda")
```

## How it works

1. **Weight download** — on first call, pretrained model weights are
   downloaded from HuggingFace and cached at ``~/.cache/ulfsynth/``.
2. **nnUNet fork** — a custom nnUNetv2 fork with the MRCT k-space
   trainer is bundled inside the package and loaded directly from the
   vendored source.
3. **Inference** — the pretrained model runs sliding-window prediction
   with the appropriate reconstruction mode (mean aggregation).

## Device selection

```python
enhance_file("input.nii.gz", "output.nii.gz", device="cuda")   # GPU
enhance_file("input.nii.gz", "output.nii.gz", device="cpu")    # CPU
enhance_file("input.nii.gz", "output.nii.gz", device="mps")    # Apple Silicon
```

Weights and intermediate tensors are kept on the selected device
(``perform_everything_on_device=True``).
