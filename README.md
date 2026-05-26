<p align="center">
  <img src="assets/method.png" alt="ULF-Synth" width="100%">
</p>

<h1 align="center">ULF-Synth: Physics-Guided Ultra-Low-Field MRI Enhancement for Pediatric Neuroimaging</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2605.24625v1"><img src="https://img.shields.io/badge/arXiv-2605.24625-b31b1b.svg" alt="arXiv"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.9%2B-blue" alt="Python"></a>
</p>

## Abstract

Ultra-low-field (ULF) MRI enables portable and accessible neuroimaging, but suffers from low signal-to-noise ratio and limited spatial resolution relative to high-field (HF) systems. Acquiring paired ULF–HF data for supervised enhancement is often infeasible, particularly in resource-limited settings. We introduce **ULF-Synth**, a framework combining: (i) acquisition-based synthesis of realistic ULF images from HF volumes for large-scale paired training, and (ii) a spatial-frequency domain objective that prioritizes recovery of high-frequency anatomical detail. The formulation is architecture-agnostic, consistently improving structural similarity and perceptual fidelity across encoder-decoder, adversarial, and diffusion-based translation models. Trained exclusively on synthetic data, our models generalize to real 64 mT ULF acquisitions, improving multiclass brain segmentation and achieving higher radiologist preference and diagnostic acceptability in a blinded reader study.

---

## Synthesis Pipeline

The ULF synthesis module models the key physical phenomena distinguishing ULF from HF acquisitions:

|  | Effect | Implementation |
|:---:|---|---|
| 1 | **Signal scaling** | $(B_{{ULF}}/B_{{HF}})^2$ polarization ratio |
| 2 | **T2\* decay & B0 inhomogeneity** | Spatially-varying exponential decay from random B0 field maps |
| 3 | **Thermal noise** | Gaussian noise scaled to SNR 15–50 |
| 4 | **k-space cropping** | Reduced resolution (45–55%) |
| 5 | **k-space undersampling** | Accelerated acquisition (2×–3×) with center-out sampling |
| 6 | **B0 off-resonance distortion** | Phase distortion from random B0 field maps |

---

## Qualitative Results

<p align="center">
  <img src="assets/results.png" alt="Sample results" width="95%">
</p>

---

## Getting Started

### Installation

```bash
git clone https://github.com/toufiqmusah/ULF-Synth.git
cd ULF-Synth
pip install -r requirements.txt
```

### CLI

```bash
# Single volume
python synthesize_ulf.py input.nii.gz output.nii.gz

# Folder of NIfTI files
python synthesize_ulf.py /path/to/hf/scans/ /path/to/ulf/scans/

# Reproducible seed
python synthesize_ulf.py input.nii.gz output.nii.gz --seed 42
```

### Python API

```python
from synthesize_ulf import simulate_ulf

# Generate one ULF volume with random parameters
ulf_volume, affine, header, params = simulate_ulf("input.nii.gz")

# With a fixed seed
ulf_volume, affine, header, params = simulate_ulf("input.nii.gz", seed=42)

# Custom parameters
from synthesize_ulf import sample_params
params = sample_params()
params["signal_target"] = 30
ulf_volume, affine, header, params = simulate_ulf("input.nii.gz", params=params)
```

Output preserves the input affine and header metadata.

---

## Roadmap

- [x] Physics-guided ULF synthesis pipeline
- [ ] Pre-trained enhancement weights — ULF→HF restoration models
- [ ] Python package — `pip install ulf-synth`
- [ ] Docker image — zero-config containerized pipeline

---

## Citation

```bibtex
@misc{musah2026ulfsynth,
  title        = {ULF-Synth: Physics-Guided Ultra-Low-Field MRI Enhancement for Pediatric Neuroimaging},
  author       = {Toufiq Musah and Salvatore Calcagno and Federica Proietto Salanitri and Xiaomeng Li and Maruf Adewole and Marawan Elbatel},
  year         = {2026},
  eprint       = {2605.24625},
  archivePrefix = {arXiv},
  url          = {https://arxiv.org/abs/2605.24625}
}
```

---

## License

[MIT](LICENSE)
