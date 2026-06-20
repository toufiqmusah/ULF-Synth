"""
Command-line interface for ULF-Synth.

Usage:
    ulfsynth simulate input output [--seed N] [--quiet]
    ulfsynth enhance input output [--device DEVICE] [--quiet]
    ulfsynth download-weights [--force]
"""

import argparse
import os


def add_simulate_parser(subparsers):
    p = subparsers.add_parser("simulate", help="Synthesize ULF MRI from HF volumes")
    p.add_argument("input", help="Path to .nii/.nii.gz file or folder of NIfTI files")
    p.add_argument("output", help="Output file path or folder")
    p.add_argument("--seed", type=int, default=None, help="Random seed")
    p.add_argument("--quiet", action="store_true", help="Suppress per-file output")
    p.set_defaults(func=_run_simulate)


def _run_simulate(args):
    from ulfsynth.simulate import simulate_file, simulate_folder
    verbose = not args.quiet
    if os.path.isdir(args.input):
        simulate_folder(args.input, args.output, seed=args.seed, verbose=verbose)
    elif os.path.isfile(args.input):
        simulate_file(args.input, args.output, seed=args.seed, verbose=verbose)
    else:
        raise FileNotFoundError(f"Input not found: {args.input}")


def add_enhance_parser(subparsers):
    p = subparsers.add_parser("enhance", help="Enhance ULF MRI using pretrained model")
    p.add_argument("input", help="Path to .nii/.nii.gz file or folder of NIfTI files")
    p.add_argument("output", help="Output file path or folder")
    p.add_argument("--device", type=str, default="cuda",
                   choices=["cuda", "cpu", "mps"],
                   help="Device for inference (default: cuda)")
    p.add_argument("--quiet", action="store_true", help="Suppress progress output")
    p.set_defaults(func=_run_enhance)


def _run_enhance(args):
    from ulfsynth.enhance import enhance_file, enhance_folder
    verbose = not args.quiet
    if os.path.isdir(args.input):
        enhance_folder(args.input, args.output, device=args.device, verbose=verbose)
    elif os.path.isfile(args.input):
        enhance_file(args.input, args.output, device=args.device, verbose=verbose)
    else:
        raise FileNotFoundError(f"Input not found: {args.input}")


def add_download_parser(subparsers):
    p = subparsers.add_parser("download-weights",
                              help="Download pretrained model weights from HuggingFace")
    p.add_argument("--force", action="store_true",
                   help="Force re-download even if cached")
    p.set_defaults(func=_run_download)


def _run_download(args):
    from ulfsynth.enhance import _download_weights
    _download_weights(force=args.force)


def main():
    parser = argparse.ArgumentParser(
        description="ULF-Synth: Physics-Guided Ultra-Low-Field MRI Enhancement & Simulation"
    )
    subparsers = parser.add_subparsers(title="commands", dest="command")
    subparsers.required = True

    add_simulate_parser(subparsers)
    add_enhance_parser(subparsers)
    add_download_parser(subparsers)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
