"""Input-driven feature extraction and conservative WaveCT strategy priors."""

from __future__ import annotations

from pathlib import Path
from statistics import median
from typing import Dict, Sequence, Tuple

import numpy as np

from wave_ct.inversion import (
    build_siddon_matrix,
    estimate_event_centered_qc,
    load_inversion_rows,
)


def _percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q)) if values.size else float("nan")


def _normalized_entropy(values: np.ndarray, bins: int) -> float:
    counts, _ = np.histogram(values, bins=bins)
    counts = counts[counts > 0]
    if counts.size <= 1:
        return 0.0
    probability = counts / counts.sum()
    return float(-np.sum(probability * np.log(probability)) / np.log(bins))


def infer_endpoint_bounds(
    sources: np.ndarray,
    receivers: np.ndarray,
    padding_fraction: float = 0.02,
) -> Tuple[float, float, float, float, float, float]:
    """Return finite endpoint bounds with small, scale-aware padding."""
    points = np.vstack([sources, receivers]).astype(np.float64)
    lower = np.min(points, axis=0)
    upper = np.max(points, axis=0)
    span = upper - lower
    nonzero = span[span > 0]
    reference = float(np.median(nonzero)) if nonzero.size else 1.0
    pad = np.maximum(span * padding_fraction, reference * 1e-4)
    lower -= pad
    upper += pad
    return (
        float(lower[0]), float(upper[0]),
        float(lower[1]), float(upper[1]),
        float(lower[2]), float(upper[2]),
    )


def recommend_grid_nodes(
    n_rays: int,
    bounds: Sequence[float],
    target_cells_per_ray: float = 0.8,
    minimum_cells: Sequence[int] = (6, 6, 3),
    maximum_cells: Sequence[int] = (40, 40, 20),
) -> Tuple[int, int, int]:
    """Recommend aspect-aware nodes while limiting unsupported parameters."""
    spans = np.asarray(
        [bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]],
        dtype=np.float64,
    )
    if np.any(~np.isfinite(spans)) or np.any(spans <= 0.0):
        raise ValueError("automatic grid bounds must have positive finite spans")
    minimum = np.asarray(minimum_cells, dtype=np.int64)
    maximum = np.asarray(maximum_cells, dtype=np.int64)
    target = int(np.clip(round(max(n_rays, 1) * target_cells_per_ray), 120, 4000))
    geometric_span = float(np.prod(spans) ** (1.0 / 3.0))
    shape = spans / geometric_span

    best = minimum.copy()
    best_error = abs(int(np.prod(best)) - target)
    for scale in np.linspace(0.25, 80.0, 5000):
        cells = np.clip(np.rint(scale * shape).astype(int), minimum, maximum)
        error = abs(int(np.prod(cells)) - target)
        if error < best_error:
            best = cells
            best_error = error
    return tuple(int(value + 1) for value in best)


def recommend_workface_grid(
    n_rays: int,
    boundary_bounds: Sequence[float],
    slice_z: Sequence[float],
    target_cells_per_ray: float = 1.65,
    minimum_total_cells: int = 240,
    maximum_total_cells: int = 2000,
) -> Tuple[Tuple[float, float, float, float, float, float], Tuple[int, int, int]]:
    """Allocate a compact 3-D grid around requested workface slices.

    Ray endpoints may remain outside this box: the inversion clips Siddon paths
    to the model.  Expanding Z to every source/receiver would spend most of the
    parameter budget far away from the requested slices and make the horizontal
    workface image unnecessarily coarse.
    """
    if len(boundary_bounds) != 4:
        raise ValueError("workface boundary must contain x_min, x_max, y_min, y_max")
    x_min, x_max, y_min, y_max = (float(value) for value in boundary_bounds)
    if not np.all(np.isfinite([x_min, x_max, y_min, y_max])) or x_max <= x_min or y_max <= y_min:
        raise ValueError("workface boundary must have positive finite spans")

    targets = sorted({float(value) for value in slice_z if np.isfinite(value)})
    positive_steps = [
        upper - lower for lower, upper in zip(targets, targets[1:])
        if upper - lower > 1e-6
    ]
    if positive_steps:
        z_step = float(median(positive_steps))
    else:
        # A single requested level still needs several cells for 3-D ray paths.
        z_step = max(min(x_max - x_min, y_max - y_min) / 64.0, 1.0)
    if targets:
        z_min = min(targets) - 0.5 * z_step
        z_max_requested = max(targets) + 0.5 * z_step
    else:
        z_min = -1.5 * z_step
        z_max_requested = 1.5 * z_step
    nz_cells = max(3, int(np.ceil((z_max_requested - z_min) / z_step - 1e-12)))
    z_max = z_min + nz_cells * z_step

    target_cells = int(np.clip(
        round(max(int(n_rays), 1) * target_cells_per_ray),
        minimum_total_cells,
        maximum_total_cells,
    ))
    horizontal_budget = max(48.0, target_cells / nz_cells)
    aspect = max((x_max - x_min) / (y_max - y_min), 0.1)
    nx_cells = max(8, int(round(np.sqrt(horizontal_budget * aspect))))
    ny_cells = max(6, int(np.ceil(horizontal_budget / nx_cells)))
    bounds = (x_min, x_max, y_min, y_max, z_min, z_max)
    nodes = (nx_cells + 1, ny_cells + 1, nz_cells + 1)
    return bounds, nodes


def event_centered_velocity_features(
    source_ids: np.ndarray,
    distance: np.ndarray,
    travel_time_s: np.ndarray,
    reference_velocity: float,
) -> Dict[str, float]:
    """Estimate velocity dispersion after removing event origin-time shifts.

    Passive-source arrival times contain an event-wise additive origin-time
    term.  Treating that term as path heterogeneity inflates the apparent
    velocity spread and can bias automatic regularization.  Only events with
    at least two observations contribute because a single arrival cannot
    separate origin time from propagation time.
    """
    groups = np.asarray(source_ids)
    ray_distance = np.asarray(distance, dtype=np.float64)
    observed = np.asarray(travel_time_s, dtype=np.float64)
    if groups.shape != ray_distance.shape or groups.shape != observed.shape:
        raise ValueError("source ids, distances and travel times must have equal shape")
    if not np.isfinite(reference_velocity) or reference_velocity <= 0.0:
        raise ValueError("event-centered reference velocity must be positive and finite")

    _, counts = np.unique(groups, return_counts=True)
    eligible_sources = np.unique(groups)[counts >= 2]
    corrected_time, corrected_velocity, event_shifts = estimate_event_centered_qc(
        groups,
        ray_distance,
        observed,
        reference_velocity,
    )
    eligible = (
        np.isin(groups, eligible_sources)
        & np.isfinite(corrected_time)
        & (corrected_time > 0.0)
        & np.isfinite(corrected_velocity)
        & (corrected_velocity > 0.0)
    )
    values = corrected_velocity[eligible]
    if values.size:
        q25, median_velocity, q75 = np.percentile(values, [25.0, 50.0, 75.0])
        iqr = max(float(q75 - q25), 1e-12)
        outlier = (
            (values < q25 - 3.0 * iqr)
            | (values > q75 + 3.0 * iqr)
        )
        velocity_mad = float(np.median(np.abs(values - median_velocity)))
        robust_cv = 1.4826 * velocity_mad / max(float(median_velocity), 1e-12)
    else:
        median_velocity = float("nan")
        outlier = np.empty(0, dtype=bool)
        robust_cv = float("nan")
    event_offset = np.asarray(
        [
            float(np.median(event_shifts[groups == source_id]))
            for source_id in eligible_sources
        ],
        dtype=np.float64,
    )
    return {
        "event_centered_reference_velocity_mps": float(reference_velocity),
        "event_centered_eligible_ray_fraction": float(np.mean(eligible)),
        "event_centered_apparent_velocity_p05_mps": _percentile(values, 5.0),
        "event_centered_apparent_velocity_median_mps": float(median_velocity),
        "event_centered_apparent_velocity_p95_mps": _percentile(values, 95.0),
        "event_centered_apparent_velocity_robust_cv": float(robust_cv),
        "event_centered_apparent_velocity_outlier_fraction": (
            float(np.mean(outlier)) if outlier.size else float("nan")
        ),
        "event_centered_event_offset_std_ms": (
            1000.0 * float(np.std(event_offset))
            if event_offset.size else float("nan")
        ),
    }


def extract_dataset_features(
    input_csv: Path,
    reference_velocity: float | None = None,
) -> Dict[str, object]:
    """Extract observable, model-free features from a standard WaveCT CSV."""
    source_ids, sx, sy, sz, rx, ry, rz, travel_time_s = load_inversion_rows(
        input_csv, allow_event_time_correction=True
    )
    sources = np.column_stack([sx, sy, sz])
    receivers = np.column_stack([rx, ry, rz])
    vectors = receivers - sources
    distance = np.linalg.norm(vectors, axis=1)
    valid_time = np.isfinite(travel_time_s) & (travel_time_s > 0.0)
    valid_velocity = valid_time & (distance > 0.0)
    apparent_velocity = distance[valid_velocity] / travel_time_s[valid_velocity]
    unique_sources, source_counts = np.unique(source_ids, return_counts=True)
    rounded_receivers = np.round(receivers, decimals=6)
    unique_stations = np.unique(rounded_receivers, axis=0)

    all_points = np.vstack([sources, receivers])
    coordinate_span = np.ptp(all_points, axis=0)
    lateral_span = max(float(coordinate_span[0]), float(coordinate_span[1]), 1e-12)
    thickness_ratio = float(coordinate_span[2] / lateral_span)
    azimuth = np.mod(np.arctan2(vectors[:, 1], vectors[:, 0]), np.pi)
    azimuth_entropy = _normalized_entropy(azimuth, bins=12)
    vertical_ratio = np.abs(vectors[:, 2]) / np.maximum(distance, 1e-12)

    if apparent_velocity.size:
        q25, median_velocity, q75 = np.percentile(apparent_velocity, [25.0, 50.0, 75.0])
        iqr = max(float(q75 - q25), 1e-12)
        velocity_outlier = (
            (apparent_velocity < q25 - 3.0 * iqr)
            | (apparent_velocity > q75 + 3.0 * iqr)
        )
        velocity_mad = float(np.median(np.abs(apparent_velocity - median_velocity)))
        robust_velocity_cv = 1.4826 * velocity_mad / max(float(median_velocity), 1e-12)
        baseline_residual = travel_time_s - distance / float(median_velocity)
        event_offsets = np.asarray(
            [
                np.median(baseline_residual[source_ids == item])
                for item in unique_sources
            ],
            dtype=np.float64,
        )
    else:
        median_velocity = float("nan")
        velocity_outlier = np.empty(0, dtype=bool)
        robust_velocity_cv = float("nan")
        event_offsets = np.empty(0, dtype=np.float64)
    event_reference = (
        float(reference_velocity)
        if reference_velocity is not None
        and np.isfinite(reference_velocity)
        and reference_velocity > 0.0
        else float(median_velocity)
    )
    centered_features = event_centered_velocity_features(
        source_ids,
        distance,
        travel_time_s,
        event_reference,
    )

    bounds = infer_endpoint_bounds(sources, receivers)
    nodes = recommend_grid_nodes(source_ids.size, bounds)
    return {
        "input_csv": str(input_csv.resolve()),
        "n_rays": int(source_ids.size),
        "n_sources": int(unique_sources.size),
        "n_stations": int(unique_stations.shape[0]),
        "rays_per_source_min": int(np.min(source_counts)),
        "rays_per_source_median": float(np.median(source_counts)),
        "rays_per_source_max": int(np.max(source_counts)),
        "positive_travel_time_fraction": float(np.mean(valid_time)),
        "distance_p05_m": _percentile(distance, 5.0),
        "distance_median_m": _percentile(distance, 50.0),
        "distance_p95_m": _percentile(distance, 95.0),
        "travel_time_p05_ms": 1000.0 * _percentile(travel_time_s[valid_time], 5.0),
        "travel_time_median_ms": 1000.0 * _percentile(travel_time_s[valid_time], 50.0),
        "travel_time_p95_ms": 1000.0 * _percentile(travel_time_s[valid_time], 95.0),
        "apparent_velocity_p05_mps": _percentile(apparent_velocity, 5.0),
        "apparent_velocity_median_mps": float(median_velocity),
        "apparent_velocity_p95_mps": _percentile(apparent_velocity, 95.0),
        "apparent_velocity_robust_cv": float(robust_velocity_cv),
        "apparent_velocity_outlier_fraction": float(np.mean(velocity_outlier))
        if velocity_outlier.size
        else float("nan"),
        "event_offset_std_ms": 1000.0 * float(np.std(event_offsets))
        if event_offsets.size
        else float("nan"),
        **centered_features,
        "x_span_m": float(coordinate_span[0]),
        "y_span_m": float(coordinate_span[1]),
        "z_span_m": float(coordinate_span[2]),
        "thickness_to_lateral_ratio": thickness_ratio,
        "azimuth_entropy_12bin": azimuth_entropy,
        "vertical_path_ratio_median": float(np.median(vertical_ratio)),
        "inferred_bounds": list(bounds),
        "recommended_grid_nodes": list(nodes),
        "recommended_grid_cells": int(np.prod(np.asarray(nodes) - 1)),
    }


def add_coverage_features(
    features: Dict[str, object],
    input_csv: Path,
    bounds: Sequence[float] | None = None,
    nodes: Sequence[int] | None = None,
) -> Dict[str, object]:
    """Add Siddon coverage diagnostics on one proposed grid."""
    source_ids, sx, sy, sz, rx, ry, rz, _ = load_inversion_rows(
        input_csv, allow_event_time_correction=True
    )
    use_bounds = tuple(float(value) for value in (bounds or features["inferred_bounds"]))
    use_nodes = tuple(int(value) for value in (nodes or features["recommended_grid_nodes"]))
    xnodes = np.linspace(use_bounds[0], use_bounds[1], use_nodes[0])
    ynodes = np.linspace(use_bounds[2], use_bounds[3], use_nodes[1])
    znodes = np.linspace(use_bounds[4], use_bounds[5], use_nodes[2])
    matrix, density = build_siddon_matrix(
        sx,
        sy,
        sz,
        rx,
        ry,
        rz,
        xnodes,
        ynodes,
        znodes,
        use_nodes[0] - 1,
        use_nodes[1] - 1,
        use_nodes[2] - 1,
    )
    density = np.asarray(density, dtype=np.float64)
    positive = density[density > 0.0]
    row_nnz = np.diff(matrix.indptr)
    total_cells = density.size
    reliable_threshold = max(3.0, float(np.percentile(positive, 50.0))) if positive.size else 3.0
    result = dict(features)
    result.update(
        {
            "coverage_grid_nodes": list(use_nodes),
            "coverage_grid_cells": int(total_cells),
            "ray_to_cell_ratio": float(source_ids.size / total_cells),
            "covered_cell_fraction": float(np.count_nonzero(density) / total_cells),
            "reliable_cell_fraction": float(np.count_nonzero(density >= reliable_threshold) / total_cells),
            "coverage_positive_median_rays": float(np.median(positive)) if positive.size else 0.0,
            "coverage_positive_p90_rays": _percentile(positive, 90.0),
            "coverage_cv": float(np.std(positive) / max(np.mean(positive), 1e-12))
            if positive.size
            else float("nan"),
            "ray_cells_crossed_median": float(np.median(row_nnz)),
            "recommended_hierarchical_split_rays": int(
                np.clip(round(reliable_threshold), 5, 8)
            ),
            "recommended_hierarchical_min_block": 3,
        }
    )
    return result


def candidate_priorities(features: Dict[str, object]) -> Dict[str, float]:
    """Return transparent priors; pilot CV remains the final selector."""
    coverage = float(features.get("covered_cell_fraction", 1.0))
    ray_to_cell = float(features.get("ray_to_cell_ratio", 1.0))
    sources = int(features["n_sources"])
    # Keep the production prior on the previously validated raw dispersion.
    # The event-centred diagnostic is reported separately until multi-dataset
    # grouped-CV demonstrates that it improves selection or regularization.
    heterogeneity = float(features.get("apparent_velocity_robust_cv", 0.0))
    hierarchy = 1.0 + max(0.0, 0.45 - coverage) + max(0.0, 0.8 - ray_to_cell)
    haar = 1.0 + min(max(sources - 20, 0) / 100.0, 0.5) + min(heterogeneity, 0.5)
    tv = 1.0 + (0.4 if sources < 20 else 0.0)
    total = hierarchy + haar + tv
    return {
        "hierarchical_tv": hierarchy / total,
        "tv_haar": haar / total,
        "tv": tv / total,
    }


def recommend_inversion_controls(features: Dict[str, object]) -> Dict[str, float | int | bool]:
    """Derive conservative solver controls from observable dataset properties.

    These are data-driven starting controls, not globally frozen hyperparameters.
    The grouped-event pilot still decides between candidate model families.
    """
    coverage = float(features.get("covered_cell_fraction", 0.3))
    ray_to_cell = float(features.get("ray_to_cell_ratio", 1.0))
    # Retain the cross-dataset-validated raw dispersion mapping for solver
    # strength; event-centred dispersion remains a reported diagnostic.
    velocity_cv = float(features.get("apparent_velocity_robust_cv", np.nan))
    if not np.isfinite(velocity_cv):
        velocity_cv = float(features.get("event_centered_apparent_velocity_robust_cv", 0.15))
    event_offset_ms = float(features.get("event_centered_event_offset_std_ms", np.nan))
    if not np.isfinite(event_offset_ms):
        event_offset_ms = float(features.get("event_offset_std_ms", 0.0))
    time_median_ms = max(float(features.get("travel_time_median_ms", 20.0)), 1.0)

    # Robust loss transition follows observed dispersion, with engineering-safe
    # bounds so one noisy dataset cannot disable robust weighting.
    huber_ms = float(np.clip(0.05 * time_median_ms + 18.0 * velocity_cv, 2.0, 12.0))
    sparse_penalty = max(0.0, 0.30 - coverage) + 0.5 * max(0.0, 0.8 - ray_to_cell)
    alpha_reg = float(np.clip(1.4 + 3.5 * velocity_cv + 5.0 * sparse_penalty, 1.4, 6.0))
    model_damping = float(np.clip(80.0 * sparse_penalty, 0.0, 35.0))
    reliable_coverage = int(np.clip(round(
        float(features.get("coverage_positive_median_rays", 5.0))
    ), 3, 8))
    return {
        "huber_delta_ms": huber_ms,
        # The inversion argument is dimensionless: the actual transition is
        # ``huber_delta * robust_sigma``.  Keep the millisecond estimate as a
        # diagnostic, but do not pass seconds into that multiplier.
        "huber_delta_sigma": 1.5,
        "alpha_reg": alpha_reg,
        "model_damping": model_damping,
        "min_ray_coverage": reliable_coverage,
        "enable_differential_times": bool(event_offset_ms >= max(3.0, 0.35 * huber_ms)),
        "enable_hierarchical_candidate": bool(coverage < 0.45 or ray_to_cell < 1.2),
        "diagnostic_event_offset_std_ms": event_offset_ms,
        "diagnostic_velocity_robust_cv": velocity_cv,
    }
