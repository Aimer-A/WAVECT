"""Deterministic coordinate-contract checks for workface rendering."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def _range(values: np.ndarray) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    return [float(np.min(array)), float(np.max(array))]


def coordinate_alignment_audit(
    velocity: np.ndarray,
    xc: np.ndarray,
    yc: np.ndarray,
    zc: np.ndarray,
    xnodes: np.ndarray,
    ynodes: np.ndarray,
    znodes: np.ndarray,
    source_rows: np.ndarray,
    station_rows: np.ndarray,
    cad_segments: Iterable[np.ndarray],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    target_z: Iterable[float],
    cad_offset: tuple[float, float],
    boundary: np.ndarray | None = None,
) -> dict[str, object]:
    """Return an auditable model-to-map coordinate contract.

    ``velocity[ix, iy, iz]`` is the quantitative convention.  Rendering uses
    ``field.T`` only because Matplotlib rows represent Y and columns represent
    X; this is not an X/Y swap in physical coordinates.
    """
    velocity = np.asarray(velocity)
    xc = np.asarray(xc, dtype=np.float64)
    yc = np.asarray(yc, dtype=np.float64)
    zc = np.asarray(zc, dtype=np.float64)
    xnodes = np.asarray(xnodes, dtype=np.float64)
    ynodes = np.asarray(ynodes, dtype=np.float64)
    znodes = np.asarray(znodes, dtype=np.float64)
    expected_shape = (xc.size, yc.size, zc.size)
    shape_ok = velocity.shape == expected_shape
    monotonic = {
        "x": bool(np.all(np.diff(xc) > 0.0)),
        "y": bool(np.all(np.diff(yc) > 0.0)),
        "z": bool(np.all(np.diff(zc) > 0.0)),
    }
    plot_matches_nodes = bool(
        np.allclose([xnodes[0], xnodes[-1]], xlim, atol=1e-3)
        and np.allclose([ynodes[0], ynodes[-1]], ylim, atol=1e-3)
    )

    source_rows = np.asarray(source_rows, dtype=np.float64)
    station_rows = np.asarray(station_rows, dtype=np.float64)

    def inside_fraction(points: np.ndarray) -> float | None:
        if points.size == 0:
            return None
        inside = (
            (points[:, 0] >= xlim[0]) & (points[:, 0] <= xlim[1])
            & (points[:, 1] >= ylim[0]) & (points[:, 1] <= ylim[1])
        )
        return float(np.mean(inside))

    segment_list = [
        np.asarray(segment, dtype=np.float64)
        for segment in cad_segments
        if np.asarray(segment).ndim == 2 and len(segment) >= 2
    ]
    if segment_list:
        cad_points = np.vstack(segment_list)
        cad_bounds: list[list[float]] | None = [
            _range(cad_points[:, 0]), _range(cad_points[:, 1])
        ]
        cad_status = "PASS"
    else:
        cad_bounds = None
        cad_status = "SKIPPED"

    targets = np.asarray(list(target_z), dtype=np.float64)
    target_in_model = [
        bool(znodes[0] <= value <= znodes[-1]) for value in targets
    ]
    failures: list[str] = []
    warnings: list[str] = []
    if not shape_ok:
        failures.append("velocity shape does not match (len(xc), len(yc), len(zc))")
    if not all(monotonic.values()):
        failures.append("one or more model axes are not strictly increasing")
    if not plot_matches_nodes:
        warnings.append(
            "plot X/Y limits differ from model node bounds; rendering continues "
            "so the complete workface/CAD context remains visible"
        )
    if targets.size and not all(target_in_model):
        warnings.append("one or more requested Z slices are outside model node bounds")
    if cad_bounds is None:
        warnings.append("CAD input missing; model-to-basemap alignment is SKIPPED")

    # A valid volume shape does not establish that a CAD basemap uses the same
    # origin.  Compare extents only as a *registration diagnostic*: drawings
    # can contain extra overview geometry, so an inferred translation must not
    # be applied silently.  The boundary is the only explicit workface extent
    # available to the renderer and is therefore the appropriate reference.
    registration: dict[str, object] = {
        "status": "SKIPPED",
        "reason": "requires both CAD segments and a workface boundary",
    }
    if cad_bounds is not None and boundary is not None:
        boundary_array = np.asarray(boundary, dtype=np.float64)
        if boundary_array.ndim == 2 and boundary_array.shape[0] >= 3 and boundary_array.shape[1] >= 2:
            boundary_x = _range(boundary_array[:, 0])
            boundary_y = _range(boundary_array[:, 1])
            cad_x, cad_y = cad_bounds
            boundary_span = np.asarray(
                [boundary_x[1] - boundary_x[0], boundary_y[1] - boundary_y[0]],
                dtype=np.float64,
            )
            cad_span = np.asarray(
                [cad_x[1] - cad_x[0], cad_y[1] - cad_y[0]], dtype=np.float64
            )
            boundary_center = np.asarray(
                [(boundary_x[0] + boundary_x[1]) / 2.0, (boundary_y[0] + boundary_y[1]) / 2.0],
                dtype=np.float64,
            )
            cad_center = np.asarray(
                [(cad_x[0] + cad_x[1]) / 2.0, (cad_y[0] + cad_y[1]) / 2.0],
                dtype=np.float64,
            )
            suggested_offset = cad_center - boundary_center
            normalized_shift = np.divide(
                np.abs(suggested_offset), boundary_span,
                out=np.full(2, np.inf), where=boundary_span > 1.0e-9,
            )
            span_ratio = np.divide(
                cad_span, boundary_span, out=np.full(2, np.nan), where=boundary_span > 1.0e-9
            )
            needs_review = bool(
                np.any(normalized_shift > 0.15) or np.any(np.abs(span_ratio - 1.0) > 0.15)
            )
            registration = {
                "status": "REVIEW_REQUIRED" if needs_review else "CONSISTENT_BY_EXTENT",
                "boundary_bounds_m": [boundary_x, boundary_y],
                "cad_bounds_m": cad_bounds,
                "boundary_to_cad_span_ratio": span_ratio.tolist(),
                "centre_shift_m": suggested_offset.tolist(),
                "centre_shift_fraction_of_boundary_span": normalized_shift.tolist(),
                "suggested_translation_offset_m": suggested_offset.tolist(),
                "suggested_offset_convention": "displayed_CAD = raw_CAD - suggested_offset",
                "automatic_application": "disabled: bounds alone cannot prove CAD feature correspondence",
            }
            if needs_review:
                warnings.append(
                    "CAD/workface extents imply a material registration difference; "
                    "review cad_x_offset/cad_y_offset before interpreting anomaly position"
                )

    status = "FAIL" if failures else ("WARN" if warnings else "PASS")
    return {
        "schema_version": 1,
        "status": status,
        "failures": failures,
        "warnings": warnings,
        "coordinate_convention": {
            "volume_index_order": "velocity[ix, iy, iz]",
            "physical_axis_order": ["X", "Y", "Z"],
            "rendering_matrix": "field.T gives rows=Y and columns=X",
            "x_y_swap_applied": False,
            "x_mirror_applied": False,
            "y_mirror_applied": False,
            "z_sign_flip_applied": False,
        },
        "shape": {
            "actual": list(velocity.shape),
            "expected": list(expected_shape),
            "matches": shape_ok,
        },
        "axes_strictly_increasing": monotonic,
        "model_centers_m": {"x": _range(xc), "y": _range(yc), "z": _range(zc)},
        "model_nodes_m": {"x": _range(xnodes), "y": _range(ynodes), "z": _range(znodes)},
        "plot_window_m": {"x": list(map(float, xlim)), "y": list(map(float, ylim))},
        "plot_window_matches_model_nodes": plot_matches_nodes,
        "source_xy_inside_plot_fraction": inside_fraction(source_rows),
        "station_xy_inside_plot_fraction": inside_fraction(station_rows),
        "cad": {
            "status": cad_status,
            "segment_count": len(segment_list),
            "displayed_bounds_m": cad_bounds,
            "configured_offset_m": list(map(float, cad_offset)),
            "offset_convention": "displayed_CAD = raw_CAD - configured_offset",
        },
        "cad_workface_registration": registration,
        "requested_z_m": targets.tolist(),
        "requested_z_inside_model": target_in_model,
    }
