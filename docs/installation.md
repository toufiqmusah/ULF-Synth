# Installation

## From PyPI

```bash
pip install ulfsynth
```

This installs the simulation pipeline, CLI, and Python API. The enhancement
module (nnUNet-based ULF→HF restoration) is bundled inside the package and
auto-loaded on first use — no extra installation step needed.

## From source (development)

```bash
git clone https://github.com/toufiqmusah/ULF-Synth.git
cd ULF-Synth
pip install -e .
```

An editable install picks up source changes immediately and is recommended for
development.

## Dependencies

- Python ≥ 3.10
- PyTorch ≥ 2.1
- NumPy, SciPy, NiBabel

PyTorch is a core dependency and will be installed automatically. If you need a
CPU-only or CUDA-specific build, pre-install PyTorch first following the
[official instructions](https://pytorch.org/get-started/locally/).

## Verifying the installation

```bash
ulfsynth --help
python -c "from ulfsynth import __version__; print(__version__)"
```
