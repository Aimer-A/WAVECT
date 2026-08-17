"""Command-line entry point for WaveCT's adapter-driven dataset importer."""

from __future__ import annotations

import argparse
from pathlib import Path

from wave_ct.dataset_import import import_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="自动发现原始数据、生成标准走时CSV和WaveCT项目"
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    projects = import_dataset(args.dataset_root, args.output_root)
    print(f"Imported projects: {len(projects)}")
    for project in projects:
        print(project)


if __name__ == "__main__":
    main()
