"""Experimental two-grid first-arrival eikonal forward modelling.

The inverse model remains on the coarse WaveCT cells while the factored
first-arrival approximation is evaluated on an independent, finer Cartesian
grid.  A second-order upwind fast-sweeping solver is used as a validation gate for
curved-ray tomography; it is deliberately kept separate from the production
Siddon solver until grouped holdout tests demonstrate a stable benefit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.interpolate import RegularGridInterpolator

try:
    from numba import njit
except ImportError:  # pragma: no cover - slow compatibility fallback
    def njit(*args, **kwargs):
        def decorate(function):
            return function
        return decorate


_FAR_TIME = 1.0e20


@dataclass(frozen=True)
class ForwardGrid:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray

    @property
    def spacing(self) -> Tuple[float, float, float]:
        return (
            float(self.x[1] - self.x[0]),
            float(self.y[1] - self.y[0]),
            float(self.z[1] - self.z[0]),
        )

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.x.size, self.y.size, self.z.size


def _regular_axis(low: float, high: float, spacing: float) -> np.ndarray:
    if not np.isfinite([low, high, spacing]).all() or high <= low or spacing <= 0.0:
        raise ValueError("invalid forward-grid bounds or spacing")
    count = max(2, int(np.ceil((high - low) / spacing)) + 1)
    return np.linspace(low, high, count, dtype=np.float64)


def build_forward_grid(
    model_bounds: Tuple[float, float, float, float, float, float],
    source_points: np.ndarray,
    receiver_points: np.ndarray,
    spacing: Tuple[float, float, float],
    padding_cells: int = 2,
    max_nodes: int = 2_000_000,
) -> ForwardGrid:
    """Cover model and endpoints on a fine grid with a small safe margin."""
    sources = np.asarray(source_points, dtype=np.float64)
    receivers = np.asarray(receiver_points, dtype=np.float64)
    if sources.ndim != 2 or receivers.ndim != 2 or sources.shape[1] != 3 or receivers.shape[1] != 3:
        raise ValueError("source and receiver points must have shape (n, 3)")
    if padding_cells < 0:
        raise ValueError("padding_cells must be non-negative")
    xmin, xmax, ymin, ymax, zmin, zmax = model_bounds
    dx, dy, dz = spacing
    combined = np.vstack([sources, receivers])
    lows = np.minimum(np.asarray([xmin, ymin, zmin]), np.min(combined, axis=0))
    highs = np.maximum(np.asarray([xmax, ymax, zmax]), np.max(combined, axis=0))
    pads = padding_cells * np.asarray([dx, dy, dz], dtype=np.float64)
    lows -= pads
    highs += pads
    grid = ForwardGrid(
        _regular_axis(float(lows[0]), float(highs[0]), dx),
        _regular_axis(float(lows[1]), float(highs[1]), dy),
        _regular_axis(float(lows[2]), float(highs[2]), dz),
    )
    if int(np.prod(grid.shape)) > max_nodes:
        raise MemoryError(
            f"forward grid {grid.shape} exceeds the {max_nodes:,}-node safety limit"
        )
    return grid


def interpolate_inverse_slowness_to_forward_grid(
    slowness: np.ndarray,
    xc: np.ndarray,
    yc: np.ndarray,
    zc: np.ndarray,
    model_bounds: Tuple[float, float, float, float, float, float],
    forward_grid: ForwardGrid,
    background_slowness: float,
) -> np.ndarray:
    """Trilinearly map the coarse inverse field into the fine forward domain."""
    nx, ny, nz = len(xc), len(yc), len(zc)
    values = np.asarray(slowness, dtype=np.float64).reshape((nx, ny, nz), order="F")
    interpolator = RegularGridInterpolator(
        (np.asarray(xc), np.asarray(yc), np.asarray(zc)),
        values,
        bounds_error=False,
        fill_value=None,
    )
    gx, gy, gz = np.meshgrid(
        forward_grid.x, forward_grid.y, forward_grid.z, indexing="ij"
    )
    points = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    fine = np.asarray(interpolator(points), dtype=np.float64)
    xmin, xmax, ymin, ymax, zmin, zmax = model_bounds
    inside = (
        (points[:, 0] >= xmin) & (points[:, 0] <= xmax)
        & (points[:, 1] >= ymin) & (points[:, 1] <= ymax)
        & (points[:, 2] >= zmin) & (points[:, 2] <= zmax)
    )
    fine[~inside] = background_slowness
    fine = np.where(np.isfinite(fine) & (fine > 0.0), fine, background_slowness)
    return fine.reshape(forward_grid.shape)


@njit(cache=True)
def _local_eikonal_update(
    travel_time: np.ndarray,
    slowness_value: float,
    ix: int,
    iy: int,
    iz: int,
    hx: float,
    hy: float,
    hz: float,
) -> float:
    nx, ny, nz = travel_time.shape
    ax = _FAR_TIME
    ay = _FAR_TIME
    az = _FAR_TIME
    hx_effective = hx
    hy_effective = hy
    hz_effective = hz
    if ix > 0:
        ax = travel_time[ix - 1, iy, iz]
    if ix + 1 < nx and travel_time[ix + 1, iy, iz] < ax:
        ax = travel_time[ix + 1, iy, iz]
        if ix + 2 < nx and travel_time[ix + 2, iy, iz] <= ax:
            ax = (4.0 * ax - travel_time[ix + 2, iy, iz]) / 3.0
            hx_effective = 2.0 * hx / 3.0
    elif ix > 1 and travel_time[ix - 2, iy, iz] <= ax:
        ax = (4.0 * ax - travel_time[ix - 2, iy, iz]) / 3.0
        hx_effective = 2.0 * hx / 3.0
    if iy > 0:
        ay = travel_time[ix, iy - 1, iz]
    if iy + 1 < ny and travel_time[ix, iy + 1, iz] < ay:
        ay = travel_time[ix, iy + 1, iz]
        if iy + 2 < ny and travel_time[ix, iy + 2, iz] <= ay:
            ay = (4.0 * ay - travel_time[ix, iy + 2, iz]) / 3.0
            hy_effective = 2.0 * hy / 3.0
    elif iy > 1 and travel_time[ix, iy - 2, iz] <= ay:
        ay = (4.0 * ay - travel_time[ix, iy - 2, iz]) / 3.0
        hy_effective = 2.0 * hy / 3.0
    if iz > 0:
        az = travel_time[ix, iy, iz - 1]
    if iz + 1 < nz and travel_time[ix, iy, iz + 1] < az:
        az = travel_time[ix, iy, iz + 1]
        if iz + 2 < nz and travel_time[ix, iy, iz + 2] <= az:
            az = (4.0 * az - travel_time[ix, iy, iz + 2]) / 3.0
            hz_effective = 2.0 * hz / 3.0
    elif iz > 1 and travel_time[ix, iy, iz - 2] <= az:
        az = (4.0 * az - travel_time[ix, iy, iz - 2]) / 3.0
        hz_effective = 2.0 * hz / 3.0

    values = np.empty(3, dtype=np.float64)
    weights = np.empty(3, dtype=np.float64)
    values[0], values[1], values[2] = ax, ay, az
    weights[0] = 1.0 / (hx_effective * hx_effective)
    weights[1] = 1.0 / (hy_effective * hy_effective)
    weights[2] = 1.0 / (hz_effective * hz_effective)
    for left in range(2):
        smallest = left
        for right in range(left + 1, 3):
            if values[right] < values[smallest]:
                smallest = right
        if smallest != left:
            tmp = values[left]
            values[left] = values[smallest]
            values[smallest] = tmp
            tmp = weights[left]
            weights[left] = weights[smallest]
            weights[smallest] = tmp

    sum_w = 0.0
    sum_wa = 0.0
    sum_waa = 0.0
    candidate = _FAR_TIME
    for dimension in range(3):
        if values[dimension] >= _FAR_TIME * 0.5:
            break
        sum_w += weights[dimension]
        sum_wa += weights[dimension] * values[dimension]
        sum_waa += weights[dimension] * values[dimension] * values[dimension]
        discriminant = sum_wa * sum_wa - sum_w * (sum_waa - slowness_value * slowness_value)
        if discriminant < 0.0:
            discriminant = 0.0
        candidate = (sum_wa + np.sqrt(discriminant)) / sum_w
        if candidate < values[dimension]:
            candidate = values[dimension]
        if dimension == 2 or values[dimension + 1] >= _FAR_TIME * 0.5 or candidate <= values[dimension + 1]:
            break
    return candidate


@njit(cache=True)
def _fast_sweep_numba(
    slowness: np.ndarray,
    travel_time: np.ndarray,
    fixed: np.ndarray,
    hx: float,
    hy: float,
    hz: float,
    max_cycles: int,
    tolerance: float,
) -> Tuple[np.ndarray, int, float]:
    nx, ny, nz = slowness.shape
    final_change = _FAR_TIME
    completed_cycles = 0
    for cycle in range(max_cycles):
        maximum_change = 0.0
        for x_reverse in range(2):
            for y_reverse in range(2):
                for z_reverse in range(2):
                    for ax in range(nx):
                        ix = nx - 1 - ax if x_reverse else ax
                        for ay in range(ny):
                            iy = ny - 1 - ay if y_reverse else ay
                            for az in range(nz):
                                iz = nz - 1 - az if z_reverse else az
                                if fixed[ix, iy, iz]:
                                    continue
                                old_value = travel_time[ix, iy, iz]
                                new_value = _local_eikonal_update(
                                    travel_time, slowness[ix, iy, iz],
                                    ix, iy, iz, hx, hy, hz,
                                )
                                if new_value < old_value:
                                    travel_time[ix, iy, iz] = new_value
                                    if old_value < _FAR_TIME * 0.5:
                                        change = old_value - new_value
                                        if change > maximum_change:
                                            maximum_change = change
                                    else:
                                        maximum_change = _FAR_TIME
        completed_cycles = cycle + 1
        final_change = maximum_change
        if maximum_change < tolerance:
            break
    return travel_time, completed_cycles, final_change


def solve_first_arrival(
    slowness: np.ndarray,
    grid: ForwardGrid,
    source_point: np.ndarray,
    max_cycles: int = 16,
    tolerance: float = 2e-6,
) -> Tuple[np.ndarray, int, float]:
    """Solve ``|grad(T)| = slowness`` from an off-grid point source."""
    slow = np.asarray(slowness, dtype=np.float64)
    if slow.shape != grid.shape or not np.isfinite(slow).all() or np.any(slow <= 0.0):
        raise ValueError("slowness must be finite, positive and match the forward grid")
    point = np.asarray(source_point, dtype=np.float64)
    if point.shape != (3,):
        raise ValueError("source point must contain x, y, z")
    axes = (grid.x, grid.y, grid.z)
    base = []
    for coordinate, axis in zip(point, axes):
        if coordinate < axis[0] or coordinate > axis[-1]:
            raise ValueError("source point lies outside the forward grid")
        lower = int(np.searchsorted(axis, coordinate, side="right") - 1)
        base.append(min(max(lower, 0), axis.size - 2))

    travel_time = np.full(grid.shape, _FAR_TIME, dtype=np.float64)
    fixed = np.zeros(grid.shape, dtype=np.bool_)
    for ix in (base[0], base[0] + 1):
        for iy in (base[1], base[1] + 1):
            for iz in (base[2], base[2] + 1):
                node = np.asarray([grid.x[ix], grid.y[iy], grid.z[iz]])
                travel_time[ix, iy, iz] = float(np.linalg.norm(node - point) * slow[ix, iy, iz])
                fixed[ix, iy, iz] = True
    hx, hy, hz = grid.spacing
    return _fast_sweep_numba(
        slow, travel_time, fixed, hx, hy, hz, max_cycles, tolerance
    )


def sample_trilinear(field: np.ndarray, grid: ForwardGrid, points: np.ndarray) -> np.ndarray:
    """Sample a scalar forward-grid field at arbitrary in-domain points."""
    values = np.asarray(field, dtype=np.float64)
    query = np.atleast_2d(np.asarray(points, dtype=np.float64))
    if values.shape != grid.shape or query.shape[1] != 3:
        raise ValueError("field/grid or point dimensions do not match")
    output = np.empty(query.shape[0], dtype=np.float64)
    axes = (grid.x, grid.y, grid.z)
    for row, point in enumerate(query):
        lower = []
        fraction = []
        for coordinate, axis in zip(point, axes):
            if coordinate < axis[0] or coordinate > axis[-1]:
                raise ValueError("sample point lies outside the forward grid")
            index = int(np.searchsorted(axis, coordinate, side="right") - 1)
            index = min(max(index, 0), axis.size - 2)
            lower.append(index)
            fraction.append((coordinate - axis[index]) / (axis[index + 1] - axis[index]))
        ix, iy, iz = lower
        fx, fy, fz = fraction
        total = 0.0
        for ox in range(2):
            wx = fx if ox else 1.0 - fx
            for oy in range(2):
                wy = fy if oy else 1.0 - fy
                for oz in range(2):
                    wz = fz if oz else 1.0 - fz
                    total += wx * wy * wz * values[ix + ox, iy + oy, iz + oz]
        output[row] = total
    return output


def trace_travel_time_gradient(
    travel_time: np.ndarray,
    grid: ForwardGrid,
    start_point: np.ndarray,
    target_point: np.ndarray,
    step_fraction: float = 0.4,
    max_steps: int = 20_000,
) -> np.ndarray:
    """Back-trace a first-arrival path by descending the travel-time field."""
    if not 0.05 <= step_fraction <= 1.0:
        raise ValueError("step_fraction must be in [0.05, 1]")
    field = np.asarray(travel_time, dtype=np.float64)
    if field.shape != grid.shape:
        raise ValueError("travel-time field does not match the forward grid")
    point = np.asarray(start_point, dtype=np.float64).copy()
    target = np.asarray(target_point, dtype=np.float64)
    hx, hy, hz = grid.spacing
    step_length = step_fraction * min(hx, hy, hz)
    stop_distance = 1.25 * max(hx, hy, hz)
    gradients = np.gradient(field, hx, hy, hz, edge_order=1)
    path = [point.copy()]

    for _ in range(max_steps):
        if np.linalg.norm(point - target) <= stop_distance:
            break
        gradient = np.asarray([
            sample_trilinear(component, grid, point)[0]
            for component in gradients
        ])
        norm = float(np.linalg.norm(gradient))
        if not np.isfinite(norm) or norm < 1e-14:
            raise RuntimeError("travel-time gradient vanished before reaching the target")
        direction = -gradient / norm
        current_time = float(sample_trilinear(field, grid, point)[0])
        trial_step = step_length
        accepted = False
        for _ in range(8):
            candidate = point + trial_step * direction
            candidate[0] = np.clip(candidate[0], grid.x[0], grid.x[-1])
            candidate[1] = np.clip(candidate[1], grid.y[0], grid.y[-1])
            candidate[2] = np.clip(candidate[2], grid.z[0], grid.z[-1])
            candidate_time = float(sample_trilinear(field, grid, candidate)[0])
            if candidate_time < current_time - 1e-12:
                point = candidate
                path.append(point.copy())
                accepted = True
                break
            trial_step *= 0.5
        if not accepted:
            raise RuntimeError("gradient back-tracing could not find a decreasing step")
    else:
        raise RuntimeError("gradient back-tracing exceeded max_steps")

    if np.linalg.norm(path[-1] - target) > 1e-10:
        path.append(target.copy())
    return np.asarray(path)


def predict_first_arrivals_by_receiver(
    slowness: np.ndarray,
    grid: ForwardGrid,
    source_points: np.ndarray,
    receiver_points: np.ndarray,
    max_cycles: int = 16,
    tolerance: float = 2e-6,
) -> Tuple[np.ndarray, dict]:
    """Use reciprocity so one field is solved for each unique receiver."""
    sources = np.asarray(source_points, dtype=np.float64)
    receivers = np.asarray(receiver_points, dtype=np.float64)
    unique_receivers, inverse = np.unique(receivers, axis=0, return_inverse=True)
    prediction = np.empty(sources.shape[0], dtype=np.float64)
    cycles = []
    changes = []
    for receiver_index, receiver in enumerate(unique_receivers):
        field, count, change = solve_first_arrival(
            slowness, grid, receiver, max_cycles=max_cycles, tolerance=tolerance
        )
        rows = np.flatnonzero(inverse == receiver_index)
        prediction[rows] = sample_trilinear(field, grid, sources[rows])
        cycles.append(count)
        changes.append(change)
    diagnostics = {
        "unique_receivers": int(unique_receivers.shape[0]),
        "cycles": np.asarray(cycles, dtype=np.int64),
        "final_changes_s": np.asarray(changes, dtype=np.float64),
    }
    return prediction, diagnostics
