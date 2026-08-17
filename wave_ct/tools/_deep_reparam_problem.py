"""Shared preparation for deep-reparameterization research tools."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from wave_ct.inversion import (
    build_grid,
    build_siddon_matrix,
    estimate_event_centered_qc,
    estimate_velocity_ranges,
    load_inversion_rows,
)


@dataclass(frozen=True)
class PreparedTomographyProblem:
    input_csv: Path
    source_ids: np.ndarray
    sx: np.ndarray
    sy: np.ndarray
    sz: np.ndarray
    rx: np.ndarray
    ry: np.ndarray
    rz: np.ndarray
    observed_roi_s: np.ndarray
    observed_full_s: np.ndarray
    ray_distance_m: np.ndarray
    inside_length_m: np.ndarray
    outside_length_m: np.ndarray
    observation_row_indices: np.ndarray
    path_matrix: csr_matrix
    ray_density: np.ndarray
    xnodes: np.ndarray
    ynodes: np.ndarray
    znodes: np.ndarray
    xc: np.ndarray
    yc: np.ndarray
    zc: np.ndarray
    nx: int
    ny: int
    nz: int
    qc_velocity_min: float
    qc_velocity_max: float
    model_velocity_min: float
    model_velocity_max: float
    background_velocity: float
    dropped_by_velocity_qc: int
    dropped_by_roi: int


def prepare_standard_problem(
    input_csv: Path,
    bounds: tuple[float, float, float, float, float, float],
    nodes: tuple[int, int, int],
    background_velocity: float = 4200.0,
) -> PreparedTomographyProblem:
    """Mirror WaveCT's production QC and straight-ray ROI preparation."""

    source_ids, sx, sy, sz, rx, ry, rz, observed_full_s = load_inversion_rows(
        input_csv,
        allow_event_time_correction=False,
    )
    observation_row_indices = np.arange(source_ids.size, dtype=np.int64)
    ray_distance = np.sqrt(
        (rx - sx) ** 2 + (ry - sy) ** 2 + (rz - sz) ** 2
    )
    qc_time, qc_velocity, _ = estimate_event_centered_qc(
        source_ids,
        ray_distance,
        observed_full_s,
        background_velocity,
    )
    finite = (
        (qc_time > 0.0)
        & (ray_distance > 1.0)
        & np.isfinite(qc_velocity)
        & (qc_velocity > 0.0)
    )
    (
        qc_velocity_min,
        qc_velocity_max,
        model_velocity_min,
        model_velocity_max,
        _,
    ) = estimate_velocity_ranges(qc_velocity[finite])
    valid = (
        finite
        & (qc_velocity >= qc_velocity_min)
        & (qc_velocity <= qc_velocity_max)
    )
    dropped_by_velocity_qc = int(np.count_nonzero(~valid))
    arrays = [
        source_ids,
        sx,
        sy,
        sz,
        rx,
        ry,
        rz,
        observed_full_s,
        ray_distance,
        observation_row_indices,
    ]
    (
        source_ids,
        sx,
        sy,
        sz,
        rx,
        ry,
        rz,
        observed_full_s,
        ray_distance,
        observation_row_indices,
    ) = [item[valid] for item in arrays]

    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    nx_nodes, ny_nodes, nz_nodes = nodes
    (
        xnodes,
        ynodes,
        znodes,
        nx,
        ny,
        nz,
        xc,
        yc,
        zc,
    ) = build_grid(
        xmin,
        xmax,
        ymin,
        ymax,
        zmin,
        zmax,
        (xmax - xmin) / (nx_nodes - 1),
        (ymax - ymin) / (ny_nodes - 1),
        (zmax - zmin) / (nz_nodes - 1),
        nx_nodes,
        ny_nodes,
        nz_nodes,
    )
    path_matrix, _ = build_siddon_matrix(
        sx,
        sy,
        sz,
        rx,
        ry,
        rz,
        xnodes,
        ynodes,
        znodes,
        nx,
        ny,
        nz,
    )
    inside_length = np.asarray(path_matrix.sum(axis=1)).ravel()
    outside_length = np.maximum(ray_distance - inside_length, 0.0)
    observed_roi_s = observed_full_s - outside_length / background_velocity
    roi_valid = (
        (inside_length > 1.0e-6)
        & (inside_length <= ray_distance + 1.0e-5)
        & np.isfinite(observed_roi_s)
        & (observed_roi_s > 0.0)
    )
    dropped_by_roi = int(np.count_nonzero(~roi_valid))
    arrays = [
        source_ids,
        sx,
        sy,
        sz,
        rx,
        ry,
        rz,
        observed_roi_s,
        observed_full_s,
        ray_distance,
        inside_length,
        outside_length,
        observation_row_indices,
    ]
    (
        source_ids,
        sx,
        sy,
        sz,
        rx,
        ry,
        rz,
        observed_roi_s,
        observed_full_s,
        ray_distance,
        inside_length,
        outside_length,
        observation_row_indices,
    ) = [item[roi_valid] for item in arrays]
    path_matrix = path_matrix[roi_valid].tocsr()
    ray_density = np.asarray((path_matrix > 0).sum(axis=0)).ravel()

    return PreparedTomographyProblem(
        input_csv=input_csv,
        source_ids=source_ids,
        sx=sx,
        sy=sy,
        sz=sz,
        rx=rx,
        ry=ry,
        rz=rz,
        observed_roi_s=observed_roi_s,
        observed_full_s=observed_full_s,
        ray_distance_m=ray_distance,
        inside_length_m=inside_length,
        outside_length_m=outside_length,
        observation_row_indices=observation_row_indices,
        path_matrix=path_matrix,
        ray_density=ray_density,
        xnodes=xnodes,
        ynodes=ynodes,
        znodes=znodes,
        xc=xc,
        yc=yc,
        zc=zc,
        nx=nx,
        ny=ny,
        nz=nz,
        qc_velocity_min=qc_velocity_min,
        qc_velocity_max=qc_velocity_max,
        model_velocity_min=model_velocity_min,
        model_velocity_max=model_velocity_max,
        background_velocity=background_velocity,
        dropped_by_velocity_qc=dropped_by_velocity_qc,
        dropped_by_roi=dropped_by_roi,
    )
