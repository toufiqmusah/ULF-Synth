"""Setup script for ULF-Synth.

Local editable installs automatically discover and include the bundled
nnunetv2 fork (src/nn-translation/) as a dependency.  For PyPI builds
the fork is handled via the ``[enhance]`` extra.
"""

import sys
from pathlib import Path

from setuptools import setup

if sys.version_info < (3, 10):
    sys.exit(f"ulfsynth requires Python >= 3.10, found {sys.version}")

HERE = Path(__file__).resolve().parent
NNUNET_FORK = HERE / "src" / "nn-translation"


def _is_local_dev():
    """Return True when installing from the repo (editable), False for PyPI builds."""
    return (HERE / ".git").exists()


def get_install_requires():
    base = [
        "numpy>=1.21.0",
        "nibabel>=3.2.0",
        "scipy>=1.7.0",
        "torch>=2.1.0",
    ]
    has_fork = (NNUNET_FORK / "setup.py").exists() or (NNUNET_FORK / "pyproject.toml").exists()
    if has_fork and _is_local_dev():
        base.append(f"nnunetv2 @ {NNUNET_FORK.resolve().as_uri()}")
    return base


def get_extras_require():
    return {
        "enhance": ["nnunetv2>=2.5"],
    }


setup(
    version="0.1.0",
    description="Physics-Guided Ultra-Low-Field MRI Enhancement & Simulation",
    install_requires=get_install_requires(),
    extras_require=get_extras_require(),
)
