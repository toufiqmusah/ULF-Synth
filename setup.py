"""Setup script for ULF-Synth.

The nnunetv2 fork (with MRCT kspace trainer) is bundled inside the
package at ``ulfsynth/_nnunet_src/`` and auto-installed on first use.
"""

import sys
from setuptools import setup

if sys.version_info < (3, 10):
    sys.exit(f"ulfsynth requires Python >= 3.10, found {sys.version}")

setup(
    version="0.1.12",
    description="Physics-Guided Ultra-Low-Field MRI Enhancement & Simulation",
    install_requires=[
        "numpy>=1.21.0",
        "nibabel>=3.2.0",
        "scipy>=1.7.0",
        "torch>=2.1.0",
        "batchgenerators>=0.25",
        "acvl-utils>=0.2",
        "SimpleITK>=2.0.0",
        "tqdm>=4.60.0",
        "requests>=2.25.0",
        "huggingface-hub>=0.20.0",
    ],
)
