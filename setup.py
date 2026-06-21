"""Setup script for ULF-Synth.

The nnunetv2 fork (with MRCT kspace trainer) is bundled inside the
package at ``ulfsynth/_nnunet_src/`` and auto-installed on first use.
"""

import sys
from setuptools import setup

if sys.version_info < (3, 10):
    sys.exit(f"ulfsynth requires Python >= 3.10, found {sys.version}")

setup(
    version="0.1.5",
    description="Physics-Guided Ultra-Low-Field MRI Enhancement & Simulation",
    install_requires=[
        "numpy>=1.21.0",
        "nibabel>=3.2.0",
        "scipy>=1.7.0",
        "torch>=2.1.0",
    ],
)
