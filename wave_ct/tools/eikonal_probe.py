"""Forward-only gate for promoting nonlinear eikonal tomography into WaveCT."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict

import numpy as np

from wave_ct.eikonal import (
    build_forward_grid,
    interpolate_inverse_slowness_to_forward_grid,
    predict_first_arrivals_by_receiver,
)
from wave_ct.inversion import build_siddon_matrix, load_inversion_rows


def event_centered_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
    source_ids: np.ndarray,
) -> Dict[str, float]:
    residual = np.asarray(observed) - np.asarray(predicted)
    centered_parts = []
    for source_id in np.unique(source_ids):
        part = residual[source_ids == source_id]
        if part.size >= 2:
            centered_parts.append(part - np.median(part))
    centered = np.concatenate(centered_parts) if centered_parts else residual
    return {
        "ray_count": int(residual.size),
        "metric_ray_count": int(centered.size),
        "raw_rms_ms": float(np.sqrt(np.mean(residual**2)) * 1000.0),
        "event_centered_rms_ms": float(np.sqrt(np.mean(centered**2)) * 1000.0),
        "event_centered_mae_ms": float(np.mean(np.abs(centered)) * 1000.0),
        "residual_median_ms": float(np.median(residual) * 1000.0),
    }


def subset_metrics(
    observed: np.ndarray,
    source_ids: np.ndarray,
    validation_ids: np.ndarray,
    predictions: Dict[str, np.ndarray],
) -> Dict[str, dict]:
    masks = {
        "all_qc": np.ones(source_ids.size, dtype=bool),
        "training": ~np.isin(source_ids, validation_ids),
        "validation": np.isin(source_ids, validation_ids),
    }
    result: Dict[str, dict] = {}
    for subset_name, mask in masks.items():
        result[subset_name] = {
            name: event_centered_metrics(observed[mask], values[mask], source_ids[mask])
            for name, values in predictions.items()
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare straight Siddon and two-grid fast-sweeping first arrivals"
    )
    parser.add_argument("model_npz", type=Path)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--xy-refinement", type=int, default=2)
    parser.add_argument("--z-refinement", type=int, default=4)
    parser.add_argument("--padding-cells", type=int, default=2)
    parser.add_argument("--max-cycles", type=int, default=16)
    parser.add_argument("--tolerance", type=float, default=2e-6)
    parser.add_argument(
        "--contrast-scales",
        default="1.0",
        help="comma-separated slowness-anomaly scale factors for continuation tests",
    )
    args = parser.parse_args()
    if args.xy_refinement < 1 or args.z_refinement < 1:
        raise ValueError("refinement factors must be positive")
    contrast_scales = sorted({
        float(item.strip()) for item in args.contrast_scales.split(",") if item.strip()
    })
    if not contrast_scales or any(
        not np.isfinite(value) or value < 0.0 or value > 1.0
        for value in contrast_scales
    ):
        raise ValueError("contrast scales must be finite values in [0, 1]")

    started = time.perf_counter()
    with np.load(args.model_npz, allow_pickle=False) as model:
        velocity = np.asarray(model["velocity"], dtype=np.float64)
        xc = np.asarray(model["xc"], dtype=np.float64)
        yc = np.asarray(model["yc"], dtype=np.float64)
        zc = np.asarray(model["zc"], dtype=np.float64)
        xnodes = np.asarray(model["xnodes"], dtype=np.float64)
        ynodes = np.asarray(model["ynodes"], dtype=np.float64)
        znodes = np.asarray(model["znodes"], dtype=np.float64)
        used_rows = np.asarray(model["used_observation_row_indices"], dtype=np.int64)
        validation_ids = np.asarray(model["validation_source_ids"], dtype=np.int64)
        background_velocity = float(model["background_velocity_mps"])

    source_ids, sx, sy, sz, rx, ry, rz, observed = load_inversion_rows(
        args.input_csv, allow_event_time_correction=True
    )
    valid_rows = used_rows[(used_rows >= 0) & (used_rows < source_ids.size)]
    source_ids = source_ids[valid_rows]
    observed = observed[valid_rows]
    sources = np.column_stack([sx[valid_rows], sy[valid_rows], sz[valid_rows]])
    receivers = np.column_stack([rx[valid_rows], ry[valid_rows], rz[valid_rows]])

    nx, ny, nz = velocity.shape
    slowness = 1.0 / velocity.ravel(order="F")
    straight_matrix, _ = build_siddon_matrix(
        sources[:, 0], sources[:, 1], sources[:, 2],
        receivers[:, 0], receivers[:, 1], receivers[:, 2],
        xnodes, ynodes, znodes, nx, ny, nz,
    )
    inside_length = np.asarray(straight_matrix.sum(axis=1)).ravel()
    full_length = np.linalg.norm(receivers - sources, axis=1)
    outside_length = np.maximum(full_length - inside_length, 0.0)
    straight_prediction = (
        np.asarray(straight_matrix @ slowness).ravel()
        + outside_length / background_velocity
    )

    inverse_spacing = (
        float(np.median(np.diff(xnodes))),
        float(np.median(np.diff(ynodes))),
        float(np.median(np.diff(znodes))),
    )
    requested_spacing = (
        inverse_spacing[0] / args.xy_refinement,
        inverse_spacing[1] / args.xy_refinement,
        inverse_spacing[2] / args.z_refinement,
    )
    model_bounds = (
        float(xnodes[0]), float(xnodes[-1]),
        float(ynodes[0]), float(ynodes[-1]),
        float(znodes[0]), float(znodes[-1]),
    )
    forward_grid = build_forward_grid(
        model_bounds, sources, receivers, requested_spacing,
        padding_cells=args.padding_cells,
    )
    background_slowness = 1.0 / background_velocity
    fine_slowness = interpolate_inverse_slowness_to_forward_grid(
        slowness, xc, yc, zc, model_bounds, forward_grid, background_slowness
    )
    eikonal_raw, model_diagnostics = predict_first_arrivals_by_receiver(
        fine_slowness, forward_grid, sources, receivers,
        max_cycles=args.max_cycles, tolerance=args.tolerance,
    )

    uniform_slowness = np.full(forward_grid.shape, background_slowness, dtype=np.float64)
    uniform_numerical, uniform_diagnostics = predict_first_arrivals_by_receiver(
        uniform_slowness, forward_grid, sources, receivers,
        max_cycles=args.max_cycles, tolerance=args.tolerance,
    )
    uniform_analytic = full_length / background_velocity
    discretization_error = uniform_numerical - uniform_analytic
    eikonal_corrected = eikonal_raw - discretization_error

    predictions = {
        "straight_siddon": straight_prediction,
        "uniform_background_analytic": uniform_analytic,
        "eikonal_raw": eikonal_raw,
        "eikonal_uniform_corrected": eikonal_corrected,
    }
    continuation_diagnostics = {}
    for contrast_scale in contrast_scales:
        label = f"{contrast_scale:.2f}".replace(".", "p")
        if np.isclose(contrast_scale, 0.0):
            predictions[f"eikonal_contrast_{label}_corrected"] = uniform_analytic
            continuation_diagnostics[label] = uniform_diagnostics
            continue
        if np.isclose(contrast_scale, 1.0):
            predictions[f"eikonal_contrast_{label}_corrected"] = eikonal_corrected
            continuation_diagnostics[label] = model_diagnostics
            continue
        scaled_slowness = background_slowness + contrast_scale * (
            slowness - background_slowness
        )
        scaled_fine = interpolate_inverse_slowness_to_forward_grid(
            scaled_slowness, xc, yc, zc, model_bounds, forward_grid,
            background_slowness,
        )
        scaled_raw, scaled_diagnostics = predict_first_arrivals_by_receiver(
            scaled_fine, forward_grid, sources, receivers,
            max_cycles=args.max_cycles, tolerance=args.tolerance,
        )
        predictions[f"eikonal_contrast_{label}_corrected"] = (
            scaled_raw - discretization_error
        )
        continuation_diagnostics[label] = scaled_diagnostics
    result = {
        "experiment": "two_grid_fast_sweeping_forward_gate",
        "model_npz": str(args.model_npz.resolve()),
        "input_csv": str(args.input_csv.resolve()),
        "runtime_seconds": float(time.perf_counter() - started),
        "inverse_grid_cells": [int(nx), int(ny), int(nz)],
        "forward_grid_nodes": [int(value) for value in forward_grid.shape],
        "forward_grid_spacing_m": [float(value) for value in forward_grid.spacing],
        "forward_grid_bounds": {
            "x": [float(forward_grid.x[0]), float(forward_grid.x[-1])],
            "y": [float(forward_grid.y[0]), float(forward_grid.y[-1])],
            "z": [float(forward_grid.z[0]), float(forward_grid.z[-1])],
        },
        "quality_control_rays": int(source_ids.size),
        "unique_receivers": int(np.unique(receivers, axis=0).shape[0]),
        "validation_source_ids": [int(value) for value in validation_ids],
        "uniform_grid_error": {
            "rms_ms": float(np.sqrt(np.mean(discretization_error**2)) * 1000.0),
            "mae_ms": float(np.mean(np.abs(discretization_error)) * 1000.0),
            "max_abs_ms": float(np.max(np.abs(discretization_error)) * 1000.0),
            "bias_ms": float(np.mean(discretization_error) * 1000.0),
        },
        "model_solver": {
            "cycles": [int(value) for value in model_diagnostics["cycles"]],
            "final_changes_s": [float(value) for value in model_diagnostics["final_changes_s"]],
        },
        "uniform_solver": {
            "cycles": [int(value) for value in uniform_diagnostics["cycles"]],
            "final_changes_s": [float(value) for value in uniform_diagnostics["final_changes_s"]],
        },
        "contrast_scales": contrast_scales,
        "continuation_solver": {
            label: {
                "cycles": [int(value) for value in diagnostics["cycles"]],
                "final_changes_s": [
                    float(value) for value in diagnostics["final_changes_s"]
                ],
            }
            for label, diagnostics in continuation_diagnostics.items()
        },
        "metrics": subset_metrics(
            observed, source_ids, validation_ids, predictions
        ),
        "prediction_difference": {
            "corrected_eikonal_minus_straight_rms_ms": float(
                np.sqrt(np.mean((eikonal_corrected - straight_prediction) ** 2)) * 1000.0
            ),
            "corrected_eikonal_minus_straight_median_ms": float(
                np.median(eikonal_corrected - straight_prediction) * 1000.0
            ),
        },
        "interpretation_guardrail": (
            "This is a forward-only gate. It does not yet prove that nonlinear "
            "bent-ray inversion improves the recovered velocity model."
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "runtime_seconds": result["runtime_seconds"],
        "forward_grid_nodes": result["forward_grid_nodes"],
        "uniform_grid_error": result["uniform_grid_error"],
        "validation": result["metrics"]["validation"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
