# Simulation Pipeline

The ULF synthesis module models the key physical phenomena that distinguish
ultra-low-field (0.064 T) from high-field (1.5 T) acquisitions:

| Step | Effect                  | Implementation                                |
|------|-------------------------|-----------------------------------------------|
| 1    | Signal scaling          | $(B_{ULF}/B_{HF})^2$ polarisation ratio      |
| 2    | T2\* decay & B0 map    | Spatially-varying exponential decay           |
| 3    | Thermal noise           | Gaussian noise (SNR 15–50)                    |
| 4    | k-space cropping        | Reduced resolution (45–55 %)                  |
| 5    | k-space undersampling   | Accelerated acquisition (2×–3×)               |
| 6    | B0 off-resonance        | Phase distortion from random field maps       |

## Example

```python
from ulfsynth.simulate import simulate_ulf, simulate_file, simulate_folder

# Get the ULF volume + metadata
ulf_vol, affine, header, params = simulate_ulf("hf_scan.nii.gz", seed=42)

# Save directly
params = simulate_file("hf_scan.nii.gz", "ulf_scan.nii.gz", seed=42)

# Process a folder
results = simulate_folder("hf_scans/", "ulf_scans/", seed=42)
```

## Parameter sampling

Each simulation draws from randomised parameter distributions.
Override specific parameters:

```python
from ulfsynth.simulate import sample_params, simulate_ulf

params = sample_params()
params["signal_target"] = 30
params["T2"] = 0.080

ulf_vol, affine, header, _ = simulate_ulf("hf_scan.nii.gz", params=params)
```

## Reproducibility

Pass a fixed ``seed`` to any simulation function.  All randomisation is
seeded deterministically, so the same input + seed always yields the same
ULF output.
