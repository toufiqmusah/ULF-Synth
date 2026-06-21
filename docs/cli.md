# Command-Line Interface

The `ulfsynth` package provides three CLI commands.

## `ulfsynth simulate`

Synthesise ULF MRI volumes from high-field inputs.

```bash
# Single volume
ulfsynth simulate input.nii.gz output.nii.gz

# Batch folder processing
ulfsynth simulate /path/to/hf/scans/ /path/to/ulf/scans/

# Reproducible seed
ulfsynth simulate input.nii.gz output.nii.gz --seed 42

# Suppress per-file logging
ulfsynth simulate input.nii.gz output.nii.gz --quiet
```

## `ulfsynth enhance`

Restore ULF volumes using a pretrained enhancement model.

```bash
# Single file (CPU)
ulfsynth enhance --device cpu input.nii.gz output.nii.gz

# Batch folder (GPU)
ulfsynth enhance /path/to/ulf/scans/ /path/to/enhanced/scans/

# Quiet mode
ulfsynth enhance --quiet input.nii.gz output.nii.gz
```

Weights are downloaded from HuggingFace on first use and cached at
`~/.cache/ulfsynth/`.

## `ulfsynth download-weights`

Pre-download the pretrained model weights.

```bash
ulfsynth download-weights

# Force re-download even if cached
ulfsynth download-weights --force
```
