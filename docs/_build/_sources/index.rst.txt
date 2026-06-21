ULF-Synth
=========

.. image:: https://img.shields.io/badge/arXiv-2605.24625-b31b1b.svg
   :target: https://arxiv.org/abs/2605.24625v1
   :alt: arXiv

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

.. image:: https://img.shields.io/badge/python-3.10%2B-blue
   :alt: Python

Physics-guided ultra-low-field MRI simulation and enhancement.

**ULF-Synth** simulates realistic ultra-low-field (0.064 T) MRI from
conventional high-field (1.5 T) volumes, and provides pretrained
models for enhancing real ULF acquisitions.  The pipeline models
signal loss, T2\* decay, thermal noise, k-space cropping,
undersampling, and B0 off-resonance — all without requiring paired
ULF-HF training data.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   cli
   api
   simulation
   enhancement
   citation



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
