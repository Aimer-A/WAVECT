"""Summarize the Stage-3 Gaussian ray-tube experiment.

This tool reads completed WaveCT result directories, computes paired validation
and model-roughness diagnostics, and writes reproducible CSV/JSON/PNG artifacts.
It does not run or alter an inversion.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np


def read_key_values(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("RMS history"):
            break
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def masked_rms_difference(
    velocity: np.ndarray, mask: np.ndarray, axis: int
) -> np.ndarray:
    delta = np.diff(velocity, axis=axis)
    left = [slice(None)] * velocity.ndim
    right = [slice(None)] * velocity.ndim
    left[axis] = slice(None, -1)
    right[axis] = slice(1, None)
    pair_mask = mask[tuple(left)] & mask[tuple(right)]
    return delta[pair_mask]


def model_diagnostics(model_path: Path) -> Dict[str, float]:
    with np.load(model_path, allow_pickle=False) as model:
        velocity = np.asarray(model["velocity"], dtype=np.float64)
        density = np.asarray(model["ray_density"], dtype=np.float64)
        support = np.asarray(
            model["kernel_support_density"]
            if "kernel_support_density" in model.files
            else density,
            dtype=np.float64,
        )
        reliable = float(np.asarray(model["reliable_coverage"]).item())
    all_lateral = np.concatenate(
        [np.diff(velocity, axis=0).ravel(), np.diff(velocity, axis=1).ravel()]
    )
    reliable_mask = density >= reliable
    reliable_parts = [
        masked_rms_difference(velocity, reliable_mask, axis=0),
        masked_rms_difference(velocity, reliable_mask, axis=1),
    ]
    reliable_parts = [part for part in reliable_parts if part.size]
    reliable_lateral = (
        np.concatenate(reliable_parts) if reliable_parts else np.empty(0)
    )
    return {
        "lateral_roughness_all_mps": float(
            np.sqrt(np.mean(all_lateral**2)) if all_lateral.size else 0.0
        ),
        "lateral_roughness_reliable_mps": float(
            np.sqrt(np.mean(reliable_lateral**2))
            if reliable_lateral.size
            else float("nan")
        ),
        "velocity_std_all_mps": float(np.std(velocity)),
        "velocity_std_reliable_mps": float(
            np.std(velocity[reliable_mask]) if np.any(reliable_mask) else float("nan")
        ),
        "centerline_covered_fraction": float(np.count_nonzero(density) / density.size),
        "kernel_supported_fraction": float(np.count_nonzero(support) / support.size),
    }


def collect_result(directory: Path) -> Dict[str, object]:
    values = read_key_values(directory / "slice_report.txt")
    result: Dict[str, object] = {
        "directory": directory.name,
        "sigma_xy_cells": float(values.get("ray_kernel_sigma_xy_cells", "0")),
        "sigma_z_cells": float(values.get("ray_kernel_sigma_z_cells", "0")),
        "train_rms_ms": 1000.0 * float(values["solution_train_rms_s"]),
        "validation_rms_ms": 1000.0 * float(values["solution_validation_rms_s"]),
        "kernel_nnz_ratio": float(values.get("ray_kernel_nnz_ratio", "1")),
        "row_sum_error_m": float(
            values.get("ray_kernel_row_sum_error_max_m", "0")
        ),
    }
    result.update(model_diagnostics(directory / "velocity_model.npz"))
    return result


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总高斯胖射线阶段3实验")
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--candidate-sigma", type=float, default=0.25)
    parser.add_argument(
        "--minimum-mean-improvement-percent",
        type=float,
        default=0.5,
        help="生产晋级要求的跨划分平均验证改善百分比",
    )
    args = parser.parse_args()
    root = args.experiment_root

    fixed: List[Dict[str, object]] = []
    for directory in root.glob("fixed_sigma_xy_*"):
        if (directory / "slice_report.txt").is_file():
            fixed.append(collect_result(directory))
    fixed.sort(key=lambda item: float(item["sigma_xy_cells"]))

    robust: List[Dict[str, object]] = []
    pattern = re.compile(r"robust_s(?P<seed>\d+)_sigma_(?P<width>.+)")
    for directory in root.glob("robust_s*_sigma_*"):
        match = pattern.fullmatch(directory.name)
        if not match or not (directory / "slice_report.txt").is_file():
            continue
        item = collect_result(directory)
        item["seed"] = int(match.group("seed"))
        robust.append(item)
    robust.sort(key=lambda item: (int(item["seed"]), float(item["sigma_xy_cells"])))

    paired: List[Dict[str, object]] = []
    for seed in sorted({int(item["seed"]) for item in robust}):
        seed_rows = [item for item in robust if int(item["seed"]) == seed]
        baseline = next(
            (item for item in seed_rows if np.isclose(float(item["sigma_xy_cells"]), 0.0)),
            None,
        )
        candidate = next(
            (
                item
                for item in seed_rows
                if np.isclose(
                    float(item["sigma_xy_cells"]), args.candidate_sigma
                )
            ),
            None,
        )
        if baseline is None or candidate is None:
            continue
        base_val = float(baseline["validation_rms_ms"])
        candidate_val = float(candidate["validation_rms_ms"])
        paired.append(
            {
                "seed": seed,
                "baseline_validation_rms_ms": base_val,
                "candidate_validation_rms_ms": candidate_val,
                "candidate_minus_baseline_ms": candidate_val - base_val,
                "improvement_percent": 100.0 * (base_val - candidate_val) / base_val,
                "baseline_train_rms_ms": float(baseline["train_rms_ms"]),
                "candidate_train_rms_ms": float(candidate["train_rms_ms"]),
                "baseline_roughness_reliable_mps": float(
                    baseline["lateral_roughness_reliable_mps"]
                ),
                "candidate_roughness_reliable_mps": float(
                    candidate["lateral_roughness_reliable_mps"]
                ),
            }
        )

    improvements = np.asarray(
        [float(item["improvement_percent"]) for item in paired], dtype=np.float64
    )
    deltas = np.asarray(
        [float(item["candidate_minus_baseline_ms"]) for item in paired],
        dtype=np.float64,
    )
    roughness_change = np.asarray(
        [
            100.0
            * (
                float(item["candidate_roughness_reliable_mps"])
                - float(item["baseline_roughness_reliable_mps"])
            )
            / float(item["baseline_roughness_reliable_mps"])
            for item in paired
        ],
        dtype=np.float64,
    )
    mean_improvement = float(np.mean(improvements)) if improvements.size else float("nan")
    improved_splits = int(np.count_nonzero(deltas < 0.0))
    gate_passed = bool(
        improvements.size >= 4
        and mean_improvement >= args.minimum_mean_improvement_percent
        and improved_splits >= 3
        and np.all(np.asarray([float(item["row_sum_error_m"]) for item in robust]) <= 1e-8)
    )

    summary = {
        "candidate_sigma_xy_cells": args.candidate_sigma,
        "paired_split_count": int(improvements.size),
        "improved_split_count": improved_splits,
        "mean_validation_improvement_percent": mean_improvement,
        "mean_candidate_minus_baseline_ms": float(np.mean(deltas)) if deltas.size else float("nan"),
        "median_candidate_minus_baseline_ms": float(np.median(deltas)) if deltas.size else float("nan"),
        "mean_reliable_roughness_change_percent": float(np.mean(roughness_change))
        if roughness_change.size
        else float("nan"),
        "minimum_mean_improvement_percent": args.minimum_mean_improvement_percent,
        "promotion_gate_passed": gate_passed,
        "promotion_decision": "promote" if gate_passed else "do_not_promote",
        "fixed_results": fixed,
        "paired_results": paired,
    }

    write_csv(root / "fixed_kernel_scan.csv", fixed)
    write_csv(root / "robust_paired_comparison.csv", paired)
    (root / "fat_ray_experiment_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3), dpi=160)
    if fixed:
        widths = np.asarray([float(item["sigma_xy_cells"]) for item in fixed])
        axes[0].plot(
            widths,
            [float(item["validation_rms_ms"]) for item in fixed],
            "o-",
            label="validation",
        )
        axes[0].plot(
            widths,
            [float(item["train_rms_ms"]) for item in fixed],
            "s--",
            label="training",
        )
        axes[0].set_xlabel(r"Gaussian tube $\sigma_{xy}$ (cells)")
        axes[0].set_ylabel("event-centered RMS (ms)")
        axes[0].set_title("Fixed held-out events")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        axes[2].plot(
            widths,
            [float(item["lateral_roughness_reliable_mps"]) for item in fixed],
            "o-",
            color="#7C3AED",
        )
        axes[2].set_xlabel(r"Gaussian tube $\sigma_{xy}$ (cells)")
        axes[2].set_ylabel("reliable-area lateral roughness (m/s)")
        axes[2].set_title("Model roughness")
        axes[2].grid(alpha=0.3)

    if paired:
        seeds = [str(item["seed"]) for item in paired]
        paired_delta = [float(item["candidate_minus_baseline_ms"]) for item in paired]
        colors = ["#16A34A" if value < 0.0 else "#DC2626" for value in paired_delta]
        axes[1].bar(seeds, paired_delta, color=colors, alpha=0.85)
        axes[1].axhline(0.0, color="#111827", linewidth=1.0)
        axes[1].set_xlabel("grouped-split random seed")
        axes[1].set_ylabel("candidate - Siddon RMS (ms)")
        axes[1].set_title("Paired validation delta (negative is better)")
        axes[1].grid(alpha=0.3)

    fig.suptitle(
        "WaveCT Stage 3: finite-frequency-inspired Gaussian ray tube",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(root / "fat_ray_stage3_validation.png", dpi=180)
    plt.close(fig)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
