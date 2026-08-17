"""Reliable module launcher for WaveCT, including non-GUI self-check."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from wave_ct.algorithm_registry import (
    WAVECT_VERSION,
    validate_algorithm_registry,
)


def _self_check() -> int:
    validate_algorithm_registry()
    missing = [
        module
        for module in ("customtkinter", "numpy", "scipy", "PIL")
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        print("WaveCT dependency check failed: " + ", ".join(missing))
        return 1
    entry = Path(__file__).resolve().parents[1] / "WaveCT.py"
    if not entry.is_file():
        print(f"WaveCT entry file not found: {entry}")
        return 1
    print(f"WaveCT {WAVECT_VERSION} self-check passed")
    print(f"Entry: {entry}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m wave_ct",
        description="Launch the WaveCT desktop application.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate the installation without opening the GUI",
    )
    args = parser.parse_args()
    if args.check:
        return _self_check()

    from WaveCT import main as launch_gui

    launch_gui()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

