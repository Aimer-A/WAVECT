"""Experimental Eikonal-gradient bent-ray Jacobians.

The first-arrival field is solved on a fine forward grid.  Gradient-descent
paths are then converted back to path lengths in WaveCT's coarse inverse cells,
so existing sparse inversion backends can be relinearized without confusing a
fine forward grid with additional model resolution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import csr_matrix

from wave_ct.eikonal import ForwardGrid, solve_first_arrival
from wave_ct.inversion import build_siddon_matrix


@dataclass(frozen=True)
class BentRayMatrixResult:
    path_matrix: csr_matrix
    outside_length_m: np.ndarray
    full_path_length_m: np.ndarray
    eikonal_prediction_s: np.ndarray
    solver_cycles: np.ndarray
    solver_final_changes_s: np.ndarray
    tracing_failures: int
    path_point_counts: np.ndarray


def _trace_batch(
    travel_time: np.ndarray,
    grid: ForwardGrid,
    start_points: np.ndarray,
    target_point: np.ndarray,
    step_length_m: float,
    max_steps: int,
) -> tuple[list[np.ndarray], int]:
    """Vectorized gradient back-tracing for sources sharing one receiver."""

    starts = np.asarray(start_points, dtype=np.float64)
    target = np.asarray(target_point, dtype=np.float64)
    if starts.ndim != 2 or starts.shape[1] != 3 or target.shape != (3,):
        raise ValueError("trace points must have shape (n, 3) and (3,)")
    if not np.isfinite(step_length_m) or step_length_m <= 0.0:
        raise ValueError("trace step length must be positive")

    axes = (grid.x, grid.y, grid.z)
    field_interpolator = RegularGridInterpolator(
        axes,
        np.asarray(travel_time, dtype=np.float64),
        bounds_error=True,
    )
    gradients = np.gradient(
        travel_time,
        *grid.spacing,
        edge_order=1,
    )
    gradient_interpolators = [
        RegularGridInterpolator(axes, component, bounds_error=True)
        for component in gradients
    ]
    points = starts.copy()
    paths: list[list[np.ndarray]] = [[point.copy()] for point in points]
    active = np.ones(points.shape[0], dtype=bool)
    stop_distance = max(
        2.0 * step_length_m,
        1.25 * max(grid.spacing),
    )
    failures = 0

    initially_finished = np.linalg.norm(points - target, axis=1) <= stop_distance
    for row in np.flatnonzero(initially_finished):
        paths[row].append(target.copy())
    active[initially_finished] = False

    lower = np.asarray([grid.x[0], grid.y[0], grid.z[0]])
    upper = np.asarray([grid.x[-1], grid.y[-1], grid.z[-1]])
    for _ in range(max_steps):
        active_rows = np.flatnonzero(active)
        if active_rows.size == 0:
            break
        current_points = points[active_rows]
        current_time = np.asarray(
            field_interpolator(current_points), dtype=np.float64
        )
        gradient = np.column_stack(
            [
                np.asarray(interpolator(current_points), dtype=np.float64)
                for interpolator in gradient_interpolators
            ]
        )
        gradient_norm = np.linalg.norm(gradient, axis=1)
        valid_gradient = np.isfinite(gradient_norm) & (gradient_norm > 1.0e-14)
        direction = np.zeros_like(gradient)
        direction[valid_gradient] = (
            -gradient[valid_gradient] / gradient_norm[valid_gradient, None]
        )
        accepted = np.zeros(active_rows.size, dtype=bool)
        accepted_points = current_points.copy()
        trial = np.full(active_rows.size, step_length_m, dtype=np.float64)
        unresolved = valid_gradient.copy()
        for _ in range(8):
            candidate_rows = np.flatnonzero(unresolved)
            if candidate_rows.size == 0:
                break
            candidate = np.clip(
                current_points[candidate_rows]
                + trial[candidate_rows, None] * direction[candidate_rows],
                lower,
                upper,
            )
            candidate_time = np.asarray(
                field_interpolator(candidate), dtype=np.float64
            )
            improved = (
                np.isfinite(candidate_time)
                & (candidate_time < current_time[candidate_rows] - 1.0e-12)
            )
            if np.any(improved):
                local_rows = candidate_rows[improved]
                accepted[local_rows] = True
                unresolved[local_rows] = False
                accepted_points[local_rows] = candidate[improved]
            rejected_rows = candidate_rows[~improved]
            trial[rejected_rows] *= 0.5

        failed_local = np.flatnonzero(~accepted)
        for local_row in failed_local:
            global_row = active_rows[local_row]
            paths[global_row].append(target.copy())
            active[global_row] = False
            failures += 1

        for local_row in np.flatnonzero(accepted):
            global_row = active_rows[local_row]
            points[global_row] = accepted_points[local_row]
            paths[global_row].append(points[global_row].copy())
            if np.linalg.norm(points[global_row] - target) <= stop_distance:
                paths[global_row].append(target.copy())
                active[global_row] = False
    if np.any(active):
        for global_row in np.flatnonzero(active):
            paths[global_row].append(target.copy())
            failures += 1
    return [np.asarray(path, dtype=np.float64) for path in paths], failures


def build_bent_ray_matrix(
    fine_slowness: np.ndarray,
    forward_grid: ForwardGrid,
    source_points: np.ndarray,
    receiver_points: np.ndarray,
    xnodes: Sequence[float],
    ynodes: Sequence[float],
    znodes: Sequence[float],
    step_length_m: float = 5.0,
    max_trace_steps: int = 3000,
    max_solver_cycles: int = 16,
    solver_tolerance: float = 2.0e-6,
) -> BentRayMatrixResult:
    """Build a coarse-cell Jacobian from fine-grid first-arrival paths."""

    sources = np.asarray(source_points, dtype=np.float64)
    receivers = np.asarray(receiver_points, dtype=np.float64)
    if (
        sources.ndim != 2
        or receivers.shape != sources.shape
        or sources.shape[1] != 3
    ):
        raise ValueError("source and receiver arrays must both have shape (n, 3)")
    x_edges = np.asarray(xnodes, dtype=np.float64)
    y_edges = np.asarray(ynodes, dtype=np.float64)
    z_edges = np.asarray(znodes, dtype=np.float64)
    nx, ny, nz = x_edges.size - 1, y_edges.size - 1, z_edges.size - 1
    if min(nx, ny, nz) < 1:
        raise ValueError("inverse-grid edge arrays must define positive cells")

    unique_receivers, receiver_inverse = np.unique(
        receivers, axis=0, return_inverse=True
    )
    paths: list[np.ndarray | None] = [None] * sources.shape[0]
    predictions = np.empty(sources.shape[0], dtype=np.float64)
    cycles = []
    final_changes = []
    failures = 0
    for receiver_index, receiver in enumerate(unique_receivers):
        travel_time, cycle_count, final_change = solve_first_arrival(
            fine_slowness,
            forward_grid,
            receiver,
            max_cycles=max_solver_cycles,
            tolerance=solver_tolerance,
        )
        rows = np.flatnonzero(receiver_inverse == receiver_index)
        interpolator = RegularGridInterpolator(
            (forward_grid.x, forward_grid.y, forward_grid.z),
            travel_time,
            bounds_error=True,
        )
        predictions[rows] = np.asarray(
            interpolator(sources[rows]), dtype=np.float64
        )
        receiver_paths, receiver_failures = _trace_batch(
            travel_time,
            forward_grid,
            sources[rows],
            receiver,
            step_length_m=step_length_m,
            max_steps=max_trace_steps,
        )
        for row, path in zip(rows, receiver_paths):
            paths[int(row)] = path
        failures += int(receiver_failures)
        cycles.append(int(cycle_count))
        final_changes.append(float(final_change))

    segment_starts = []
    segment_ends = []
    segment_ray_ids = []
    point_counts = np.empty(sources.shape[0], dtype=np.int64)
    for ray_index, path_value in enumerate(paths):
        if path_value is None or path_value.shape[0] < 2:
            raise RuntimeError(f"no bent path was produced for ray {ray_index}")
        point_counts[ray_index] = path_value.shape[0]
        segment_starts.append(path_value[:-1])
        segment_ends.append(path_value[1:])
        segment_ray_ids.append(
            np.full(path_value.shape[0] - 1, ray_index, dtype=np.int32)
        )
    starts = np.vstack(segment_starts)
    ends = np.vstack(segment_ends)
    ray_ids = np.concatenate(segment_ray_ids)
    segment_matrix, _ = build_siddon_matrix(
        starts[:, 0],
        starts[:, 1],
        starts[:, 2],
        ends[:, 0],
        ends[:, 1],
        ends[:, 2],
        x_edges,
        y_edges,
        z_edges,
        nx,
        ny,
        nz,
    )
    segment_count = starts.shape[0]
    aggregation = csr_matrix(
        (
            np.ones(segment_count, dtype=np.float64),
            (ray_ids, np.arange(segment_count, dtype=np.int32)),
        ),
        shape=(sources.shape[0], segment_count),
    )
    path_matrix = (aggregation @ segment_matrix).tocsr()
    segment_lengths = np.linalg.norm(ends - starts, axis=1)
    full_length = np.bincount(
        ray_ids,
        weights=segment_lengths,
        minlength=sources.shape[0],
    ).astype(np.float64)
    inside_length = np.asarray(path_matrix.sum(axis=1)).ravel()
    outside_length = np.maximum(full_length - inside_length, 0.0)
    return BentRayMatrixResult(
        path_matrix=path_matrix,
        outside_length_m=outside_length,
        full_path_length_m=full_length,
        eikonal_prediction_s=predictions,
        solver_cycles=np.asarray(cycles, dtype=np.int64),
        solver_final_changes_s=np.asarray(final_changes, dtype=np.float64),
        tracing_failures=failures,
        path_point_counts=point_counts,
    )
