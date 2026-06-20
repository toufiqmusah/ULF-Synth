# Publishing ULF-Synth to PyPI

## Prerequisites

1. A [PyPI account](https://pypi.org/account/register/) (one-time)
2. Build tools installed:

```bash
pip install build twine
```

---

## 1. Check package name availability

```bash
# Returns 404 if available, JSON if taken
curl -s https://pypi.org/pypi/ulfsynth/json | head -1
```

If the name is taken, update `name = "ulfsynth"` in `pyproject.toml` and this repo.

---

## 2. Version bump

Update the version string in **both** files:

- `pyproject.toml` — `version = "0.1.0"`
- `setup.py` — `version="0.1.0"`

Follow [semver](https://semver.org/): `MAJOR.MINOR.PATCH`.

---

## 3. Decide on nnunetv2 strategy

The bundled fork (`src/nn-translation/`) is the main complication. Pick one:

### Option A — Publish fork separately (recommended)

1. Fork the upstream nnunetv2, apply the MRCT kspace changes, publish as `nnunetv2-ulfsynth` on PyPI
2. In `setup.py`, update:

```python
def get_install_requires():
    return [
        "numpy>=1.21.0",
        "nibabel>=3.2.0",
        "scipy>=1.7.0",
        "torch>=2.1.0",
    ]

def get_extras_require():
    return {"enhance": ["nnunetv2-ulfsynth>=2.5"]}
```

3. Remove `src/nn-translation/` from `MANIFEST.in` — no need to bundle it
4. Remove the fork-detection logic from `setup.py` (the `NNUNET_FORK` block)

### Option B — Skip PyPI nnunetv2

Keep the fork bundled in the repo. Users install it manually:

```bash
pip install ulfsynth
pip install -e src/nn-translation/   # for enhancement
```

Update `MANIFEST.in` to **exclude** `src/` before publishing (avoids bloat):

```diff
-recursive-include src/nn-translation *
```

---

## 4. Build

```bash
# Clean previous builds
rm -rf dist/ build/ *.egg-info/

# Build source distribution + wheel
python -m build
```

Check the built archive contents:

```bash
tar --list -f dist/ulfsynth-0.1.0.tar.gz
```

---

## 5. Upload to TestPyPI (recommended first)

```bash
twine upload --repository testpypi dist/*
```

Install from TestPyPI to verify:

```bash
pip install --index-url https://test.pypi.org/simple/ ulfsynth
```

---

## 6. Upload to PyPI

```bash
twine upload dist/*
```

---

## 7. Tag the release

```bash
git tag -a v0.1.0 -m "v0.1.0 — initial release"
git push --tags
```

---

## Appendix: CI tips

### Skip CI on a push

Add `[skip ci]` to the commit message to skip all workflow runs:

```bash
git commit -m "typo fix [skip ci]"
git push
```

Alternative markers: `[ci skip]`, `[no ci]`, `skip-checks: true`.

### PyPI propagation time

After `twine upload` or a successful CI publish, wait **1–2 minutes** before:

```bash
pip install ulfsynth
```

PyPI's CDN (Fastly) caches package metadata for a short time. If you get a "404 Not Found" immediately, wait 60s and try again.

To install the very latest version after a publish:

```bash
pip install --upgrade ulfsynth
```

## Summary checklist

| Step | Done? |
|------|-------|
| PyPI account created | ☐ |
| Package name confirmed available | ☐ |
| Version bumped | ☐ |
| nnunetv2 strategy decided (A or B) | ☐ |
| `MANIFEST.in` updated accordingly | ☐ |
| `python -m build` succeeds | ☐ |
| TestPyPI upload + verify | ☐ |
| PyPI upload | ☐ |
| Git tag pushed | ☐ |
