"""
Physics-guided ULF MRI simulation from high-field volumes.

If you use ULF-Synth in your work, please cite:

    Musah et al. "ULF-Synth: Physics-Guided Ultra-Low-Field MRI
    Enhancement for Pediatric Neuroimaging." arXiv:2605.24625, 2026.
"""

import numpy as np
import nibabel as nib
from scipy.ndimage import gaussian_filter
from numpy.fft import fftn, ifftn, fftshift, ifftshift
import os

B_HF = 1.5
B_ULF = 0.064
POLAR_SCALE = (B_ULF / B_HF) ** 2

_CITED = False


def _citation():
    global _CITED
    if not _CITED:
        _CITED = True
        print(
            "If you use ULF-Synth simulation in your work, please cite:\n"
            "  Musah et al. \"ULF-Synth: Physics-Guided Ultra-Low-Field MRI\n"
            "  Enhancement for Pediatric Neuroimaging.\" arXiv:2605.24625, 2026.\n",
            flush=True,
        )


def to_kspace(vol):
    return fftshift(fftn(vol))


def to_ispace(K):
    return np.real(ifftn(ifftshift(K)))


def to_complex_ispace(K):
    return ifftn(ifftshift(K))


def object_mask(vol, thresh=0.05):
    normed = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
    return normed > thresh


def apply_b0_t2star_decay(I_complex, T2, TE, b0_strength, smoothness, k, eps=1e-6):
    H, W, D = I_complex.shape
    B0_map = gaussian_filter(np.random.randn(H, W, D), sigma=smoothness) * b0_strength
    gx, gy, gz = np.gradient(B0_map)
    grad_B0 = np.sqrt(gx**2 + gy**2 + gz**2)
    grad_B0 /= (np.percentile(grad_B0, 95) + eps)
    T2star = 1.0 / ((1.0 / T2) + k * grad_B0 + eps)
    decay = np.exp(-TE / T2star)
    return I_complex * decay


def noisy_kspace(K, I_ref, signal_target):
    alpha = 0.05
    K_scaled = alpha * K
    mask = object_mask(I_ref)
    I_scaled = np.abs(ifftn(ifftshift(K_scaled)))
    signal_power = np.mean(I_scaled[mask] ** 2)
    sigma = np.sqrt(signal_power / (signal_target + 1e-8))
    noise_img = sigma * np.ones_like(mask, dtype=float) * (
        np.random.randn(*I_scaled.shape) + 1j * np.random.randn(*I_scaled.shape)
    )
    K_noisy = K_scaled + fftshift(fftn(noise_img))
    return K_noisy


def kspace_crop(K, crop_ratio):
    H, W, D = K.shape
    ch, cw, cd = int(H * crop_ratio), int(W * crop_ratio), int(D * crop_ratio)
    h0, w0, d0 = (H - ch) // 2, (W - cw) // 2, (D - cd) // 2
    K_cropped = np.zeros_like(K, dtype=complex)
    K_cropped[h0:h0+ch, w0:w0+cw, d0:d0+cd] = K[h0:h0+ch, w0:w0+cw, d0:d0+cd]
    return K_cropped


def undersample_kspace(K, acceleration, center_fraction):
    H, W, D = K.shape
    mask = np.zeros((H, W, D), dtype=bool)
    ch, cw, cd = int(H * center_fraction), int(W * center_fraction), int(D * center_fraction)
    h0, w0, d0 = (H - ch) // 2, (W - cw) // 2, (D - cd) // 2
    mask[h0:h0+ch, w0:w0+cw, d0:d0+cd] = True
    outer = ~mask
    n_sample = int(np.sum(outer) / acceleration)
    coords = np.where(outer)
    idx = np.random.choice(len(coords[0]), n_sample, replace=False)
    mask[coords[0][idx], coords[1][idx], coords[2][idx]] = True
    K_us = K * mask
    I_original = np.abs(ifftn(ifftshift(K)))
    I_us = np.abs(ifftn(ifftshift(K_us))) * object_mask(I_original, thresh=0.05)
    return fftshift(fftn(I_us))


def apply_b0_inhomogeneity(K, distortion_strength, smoothness):
    I_complex = ifftn(ifftshift(K))
    H, W, D = I_complex.shape
    field_map = gaussian_filter(np.random.randn(H, W, D), sigma=smoothness) * distortion_strength
    I_distorted = I_complex * np.exp(2j * np.pi * field_map)
    return fftshift(fftn(I_distorted))


def sample_params(rng=None):
    """Sample random simulation parameters.

    Returns a dictionary of physics parameters that control the ULF
    degradation pipeline.  Each call draws fresh random values from
    pre-defined distributions that model realistic 0.064 T acquisitions.

    Args:
        rng: Optional random number generator (``numpy.random`` or
            ``numpy.random.Generator``).  Defaults to ``numpy.random``.

    Returns:
        dict with keys ``T2``, ``TE``, ``b0_strength``, ``smoothness``,
        ``k``, ``signal_target``, ``crop_ratio``, ``acceleration``,
        ``center_fraction``.
    """
    if rng is None:
        rng = np.random
    return {
        "T2":               rng.uniform(0.070, 0.090),
        "TE":               rng.uniform(0.100, 0.130),
        "b0_strength":      rng.uniform(0.020, 0.040),
        "smoothness":       rng.uniform(30, 45),
        "k":                rng.uniform(10, 15),
        "signal_target":    rng.uniform(15, 50),
        "crop_ratio":       rng.uniform(0.45, 0.55),
        "acceleration":     int(rng.choice([2, 3], p=[0.7, 0.3])),
        "center_fraction":  rng.uniform(0.20, 0.30),
    }


def simulate_ulf(hf_path, seed=None, params=None):
    """Synthesize a ULF volume from a high-field NIfTI file.

    Applies the full physics-guided degradation pipeline: polarisation
    scaling, T2\\* decay with B0 inhomogeneity, thermal noise,
    k-space cropping, undersampling, and off-resonance distortion.

    Args:
        hf_path: Path to the input high-field NIfTI file.
        seed: Random seed for reproducibility.  ``None`` = non-deterministic.
        params: Parameter dict (see :func:`sample_params`).
            ``None`` = sample fresh parameters.

    Returns:
        Tuple of ``(ulf_volume, affine, header, params)`` where
        ``ulf_volume`` is a ``(H, W, D)`` float32 array, ``affine``
        and ``header`` are copied from the input, and ``params`` is
        the parameter dict used.
    """
    if seed is not None:
        np.random.seed(seed)
    if params is None:
        params = sample_params()

    hf_img = nib.load(hf_path)
    I_hf = hf_img.get_fdata()
    affine = hf_img.affine
    header = hf_img.header.copy()

    I_scaled = POLAR_SCALE * I_hf
    K_hf = to_kspace(I_scaled)
    I_complex = to_complex_ispace(K_hf)

    I_b0 = apply_b0_t2star_decay(
        I_complex,
        T2=params["T2"],
        TE=params["TE"],
        b0_strength=params["b0_strength"],
        smoothness=params["smoothness"],
        k=params["k"],
    )

    K_b0 = to_kspace(I_b0)
    K_noisy = noisy_kspace(K_b0, I_scaled, signal_target=params["signal_target"])

    K_crop = kspace_crop(K_noisy, crop_ratio=params["crop_ratio"])

    K_under = undersample_kspace(
        K_crop,
        acceleration=params["acceleration"],
        center_fraction=params["center_fraction"],
    )

    K_final = apply_b0_inhomogeneity(
        K_under,
        distortion_strength=params["b0_strength"],
        smoothness=max(1, int(params["smoothness"] // 10)),
    )

    I_ulf = np.abs(to_ispace(K_final)).astype(np.float32)
    return I_ulf, affine, header, params


def simulate_file(hf_path, out_path, seed=None, verbose=True):
    """Simulate ULF from a single file and save the result.

    Args:
        hf_path: Input high-field NIfTI path.
        out_path: Output ULF NIfTI path.
        seed: Random seed (``None`` = non-deterministic).
        verbose: Print a summary line when done.

    Returns:
        The parameter dict used for this simulation.
    """
    _citation()
    I_ulf, affine, header, params = simulate_ulf(hf_path, seed=seed)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    nib.save(nib.Nifti1Image(I_ulf, affine, header), out_path)
    if verbose:
        print(f"  {os.path.basename(hf_path)} -> {out_path}  shape={I_ulf.shape}  range=[{I_ulf.min():.3f}, {I_ulf.max():.3f}]")
    return params


def simulate_folder(in_dir, out_dir, seed=None, verbose=True):
    """Simulate ULF for every NIfTI file in a folder.

    Args:
        in_dir: Directory containing input high-field NIfTI files.
        out_dir: Directory for output ULF NIfTI files (created if
            needed).
        seed: Base random seed.  Each file gets a deterministic
            offset derived from its filename.
        verbose: Print a summary line per file.

    Returns:
        List of parameter dicts, one per file.
    """
    _citation()
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(in_dir) if f.endswith(('.nii', '.nii.gz')))
    if not files:
        raise FileNotFoundError(f"No NIfTI files found in {in_dir}")
    results = []
    for fname in files:
        hf_path = os.path.join(in_dir, fname)
        out_path = os.path.join(out_dir, fname)
        file_seed = None if seed is None else seed + abs(hash(fname)) % (2**31)
        params = simulate_file(hf_path, out_path, seed=file_seed, verbose=verbose)
        results.append(params)
    return results
