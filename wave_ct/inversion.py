"""
作用:
- 读取反演输入 CSV(震源坐标、台站坐标、传播时间)，执行 3D P 波速度层析反演。
- 采用 Siddon 射线长度矩阵 + 平滑正则 + LSQR 迭代求解速度场。
- 生成多张速度切片图、RMS 收敛图、射线覆盖图，并输出切片统计。

输入:
- --input-csv: 反演输入 CSV 路径。
- --output-dir: 结果目录。
- --x-min/--x-max 等: 反演体空间范围，必须覆盖震源、台站和主要射线路径。
- --dx/--dy/--dz: 反演网格间距。
- --n-outer/--n-lsqr: 外层迭代与 LSQR 内迭代参数。

输出:
- 5 张切片 PNG。
- 反演统计文本(射线覆盖、RMS 变化等)。
"""

import argparse
import csv
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import BSpline, RectBivariateSpline, RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from scipy.optimize import differential_evolution
from scipy.sparse import csr_matrix, diags, hstack, vstack
from scipy.sparse.linalg import lsqr

from wave_ct.sirt_solver import solve_sirt


matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 220

CSV_COLUMNS = [
    "\u9707\u6e90\u7f16\u53f7",
    "\u9707\u6e90\u5750\u6807-x",
    "\u9707\u6e90\u5750\u6807-y",
    "\u9707\u6e90\u5750\u6807-z",
    "\u53d1\u9707\u65f6\u523bt",
    "\u53f0\u7ad9\u5750\u6807-x",
    "\u53f0\u7ad9\u5750\u6807-y",
    "\u53f0\u7ad9\u5750\u6807-z",
    "\u53f0\u7ad9P\u6ce2\u5230\u65f6",
    "\u9707\u6e90-\u53f0\u7ad9\u4f20\u64ad\u65f6\u95f4",
    "\u9707\u6e90\u4e8b\u4ef6\u6587\u4ef6\u540d",
]


def parse_float(text: str) -> float:
    return float(str(text).strip())


def validate_dataset_contract(
    path: Path,
    bounds: Tuple[float, float, float, float, float, float],
    expected_sources: int = 0,
    expected_stations_per_source: int = 0,
    allow_outside_endpoints: bool = False,
    allow_event_time_correction: bool = False,
) -> Dict[str, float]:
    """Validate only observable input fields and user-specified constraints."""
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if list(reader.fieldnames or []) != CSV_COLUMNS:
            raise ValueError(
                "输入CSV字段或字段顺序不符合反演数据集格式参考.csv。"
                f"\n期望字段: {','.join(CSV_COLUMNS)}"
            )
        rows = list(reader)

    if not rows:
        raise ValueError("输入CSV没有数据记录。")

    source_counts: Counter[int] = Counter()
    station_sets: Dict[int, set] = {}
    max_time_identity_error_ms = 0.0

    for line_no, row in enumerate(rows, start=2):
        try:
            source_id = int(float(row[CSV_COLUMNS[0]]))
            sx, sy, sz = (parse_float(row[CSV_COLUMNS[i]]) for i in (1, 2, 3))
            t0_ms = parse_float(row[CSV_COLUMNS[4]])
            rx, ry, rz = (parse_float(row[CSV_COLUMNS[i]]) for i in (5, 6, 7))
            pick_ms = parse_float(row[CSV_COLUMNS[8]])
            travel_ms = parse_float(row[CSV_COLUMNS[9]])
            event_name = str(row[CSV_COLUMNS[10]]).strip()
        except Exception as exc:
            raise ValueError(f"CSV第{line_no}行存在无效数值: {exc}") from exc

        if source_id < 1:
            raise ValueError(f"CSV第{line_no}行震源编号必须为正整数。")
        if (pick_ms <= t0_ms or travel_ms <= 0) and not allow_event_time_correction:
            raise ValueError(f"CSV第{line_no}行P波到时必须晚于发震时刻。")

        identity_error = abs(travel_ms - (pick_ms - t0_ms))
        max_time_identity_error_ms = max(max_time_identity_error_ms, identity_error)
        if identity_error > 1e-3:
            raise ValueError(
                f"CSV第{line_no}行传播时间不等于P波到时减发震时刻，"
                f"误差为{identity_error:.6f}ms。"
            )
        if not event_name:
            raise ValueError(f"CSV第{line_no}行震源事件文件名为空。")
        xmin, xmax, ymin, ymax, zmin, zmax = bounds
        if not allow_outside_endpoints and not (
            xmin <= sx <= xmax and ymin <= sy <= ymax and zmin <= sz <= zmax
            and xmin <= rx <= xmax and ymin <= ry <= ymax and zmin <= rz <= zmax
        ):
            raise ValueError(
                f"CSV第{line_no}行震源或台站坐标超出当前反演范围，"
                "请扩大模型范围后重试。"
            )

        source_counts[source_id] += 1
        station_sets.setdefault(source_id, set()).add((rx, ry, rz))

    if expected_sources > 0 and len(source_counts) != expected_sources:
        raise ValueError(
            f"期望{expected_sources}个震源，CSV实际包含{len(source_counts)}个。"
        )
    if expected_stations_per_source > 0:
        bad_counts = {
            sid: count for sid, count in source_counts.items()
            if count != expected_stations_per_source
        }
        if bad_counts:
            raise ValueError(
                f"每个震源应有{expected_stations_per_source}条记录，异常震源: {bad_counts}"
            )
        bad_station_sets = {
            sid: len(values) for sid, values in station_sets.items()
            if len(values) != expected_stations_per_source
        }
        if bad_station_sets:
            raise ValueError(
                f"每个震源应对应{expected_stations_per_source}个不同台站，"
                f"异常震源: {bad_station_sets}"
            )
    return {
        "row_count": float(len(rows)),
        "source_count": float(len(source_counts)),
        "stations_per_source_min": float(min(source_counts.values())),
        "stations_per_source_max": float(max(source_counts.values())),
        "max_time_identity_error_ms": max_time_identity_error_ms,
    }


def load_inversion_rows(
    path: Path,
    allow_event_time_correction: bool = False,
) -> Tuple[np.ndarray, ...]:
    source_id_list: List[int] = []
    sx_list: List[float] = []
    sy_list: List[float] = []
    sz_list: List[float] = []
    rx_list: List[float] = []
    ry_list: List[float] = []
    rz_list: List[float] = []
    tt_list: List[float] = []

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {
            "震源编号",
            "震源坐标-x",
            "震源坐标-y",
            "震源坐标-z",
            "台站坐标-x",
            "台站坐标-y",
            "台站坐标-z",
            "台站P波到时",
            "发震时刻t",
            "震源-台站传播时间",
        }
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError("输入 CSV 字段不完整，缺少反演所需列。")

        for row in reader:
            source_id = int(float(row["震源编号"]))
            sx = parse_float(row["震源坐标-x"])
            sy = parse_float(row["震源坐标-y"])
            sz = parse_float(row["震源坐标-z"])
            rx = parse_float(row["台站坐标-x"])
            ry = parse_float(row["台站坐标-y"])
            rz = parse_float(row["台站坐标-z"])

            t_ms_text = str(row["震源-台站传播时间"]).strip()
            if t_ms_text:
                tt_ms = parse_float(t_ms_text)
            else:
                tp_ms = parse_float(row["台站P波到时"])
                t0_ms = parse_float(row["发震时刻t"])
                tt_ms = tp_ms - t0_ms

            tt_s = tt_ms / 1000.0
            if not np.isfinite(tt_s) or (
                tt_s <= 0 and not allow_event_time_correction
            ):
                continue

            source_id_list.append(source_id)
            sx_list.append(sx)
            sy_list.append(sy)
            sz_list.append(sz)
            rx_list.append(rx)
            ry_list.append(ry)
            rz_list.append(rz)
            tt_list.append(tt_s)

    if not sx_list:
        raise RuntimeError("输入中没有可用于反演的有效记录。")

    return (
        np.array(source_id_list, dtype=np.int64),
        np.array(sx_list, dtype=np.float64),
        np.array(sy_list, dtype=np.float64),
        np.array(sz_list, dtype=np.float64),
        np.array(rx_list, dtype=np.float64),
        np.array(ry_list, dtype=np.float64),
        np.array(rz_list, dtype=np.float64),
        np.array(tt_list, dtype=np.float64),
    )


def build_grid(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    zmin: float,
    zmax: float,
    dx: float,
    dy: float,
    dz: float,
    nx_nodes: int = 0,
    ny_nodes: int = 0,
    nz_nodes: int = 0,
):
    def axis_nodes(
        low: float,
        high: float,
        requested_spacing: float,
        requested_nodes: int,
    ) -> np.ndarray:
        """Create an exact, uniform grid without an accidental extra edge cell."""
        if not (np.isfinite(low) and np.isfinite(high) and np.isfinite(requested_spacing)):
            raise ValueError("grid bounds and spacing must be finite")
        if high <= low or requested_spacing <= 0.0:
            raise ValueError("grid upper bound and spacing must be positive")
        if requested_nodes:
            if requested_nodes < 2:
                raise ValueError("grid node count must be at least 2")
            return np.linspace(low, high, requested_nodes, dtype=np.float64)
        span = high - low
        ratio = span / requested_spacing
        nearest = max(1, int(round(ratio)))
        # The GUI persists spacing as decimal text.  Treat a value that is
        # numerically within five parts per million of an integer cell count
        # as that exact count; otherwise a harmless text round-off creates an
        # unintended extra row/column and changes the inverse problem.
        if np.isclose(ratio, nearest, rtol=5e-6, atol=5e-8):
            cell_count = nearest
        else:
            cell_count = max(1, int(np.ceil(ratio)))
        return np.linspace(low, high, cell_count + 1, dtype=np.float64)

    xnodes = axis_nodes(xmin, xmax, dx, nx_nodes)
    ynodes = axis_nodes(ymin, ymax, dy, ny_nodes)
    znodes = axis_nodes(zmin, zmax, dz, nz_nodes)

    nx = len(xnodes) - 1
    ny = len(ynodes) - 1
    nz = len(znodes) - 1
    xc = (xnodes[:-1] + xnodes[1:]) / 2.0
    yc = (ynodes[:-1] + ynodes[1:]) / 2.0
    zc = (znodes[:-1] + znodes[1:]) / 2.0

    return xnodes, ynodes, znodes, nx, ny, nz, xc, yc, zc


def idx3(ix: int, iy: int, iz: int, nx: int, ny: int) -> int:
    return ix + iy * nx + iz * nx * ny


def build_siddon_matrix(
    sx: np.ndarray,
    sy: np.ndarray,
    sz: np.ndarray,
    rx: np.ndarray,
    ry: np.ndarray,
    rz: np.ndarray,
    xnodes: np.ndarray,
    ynodes: np.ndarray,
    znodes: np.ndarray,
    nx: int,
    ny: int,
    nz: int,
) -> Tuple[csr_matrix, np.ndarray]:
    row_list: List[int] = []
    col_list: List[int] = []
    val_list: List[float] = []

    n_rays = sx.size

    for i in range(n_rays):
        p1 = np.array([sx[i], sy[i], sz[i]], dtype=np.float64)
        p2 = np.array([rx[i], ry[i], rz[i]], dtype=np.float64)
        d = p2 - p1
        length = np.linalg.norm(d)
        if length < 1e-12:
            continue

        # Siddon parameter values must include both ray endpoints.  Without
        # 0 and 1 the first and last partial cells are silently omitted.
        alphas = [np.array([0.0, 1.0], dtype=np.float64)]
        if abs(d[0]) > 1e-12:
            alphas.append((xnodes - p1[0]) / d[0])
        if abs(d[1]) > 1e-12:
            alphas.append((ynodes - p1[1]) / d[1])
        if abs(d[2]) > 1e-12:
            alphas.append((znodes - p1[2]) / d[2])
        aa = np.unique(np.concatenate(alphas))
        aa = aa[(aa >= -1e-10) & (aa <= 1 + 1e-10)]
        aa = np.clip(aa, 0.0, 1.0)
        aa = np.unique(aa)
        if aa.size < 2:
            continue

        for k in range(aa.size - 1):
            amid = 0.5 * (aa[k] + aa[k + 1])
            pmid = p1 + amid * d

            ix_arr = np.where((pmid[0] >= xnodes[:-1]) & (pmid[0] < xnodes[1:]))[0]
            iy_arr = np.where((pmid[1] >= ynodes[:-1]) & (pmid[1] < ynodes[1:]))[0]
            iz_arr = np.where((pmid[2] >= znodes[:-1]) & (pmid[2] < znodes[1:]))[0]

            ix = ix_arr[0] if ix_arr.size > 0 else (nx - 1 if abs(pmid[0] - xnodes[-1]) < 1e-6 else -1)
            iy = iy_arr[0] if iy_arr.size > 0 else (ny - 1 if abs(pmid[1] - ynodes[-1]) < 1e-6 else -1)
            iz = iz_arr[0] if iz_arr.size > 0 else (nz - 1 if abs(pmid[2] - znodes[-1]) < 1e-6 else -1)
            if ix < 0 or iy < 0 or iz < 0:
                continue

            seg = (aa[k + 1] - aa[k]) * length
            if seg <= 1e-12:
                continue

            row_list.append(i)
            col_list.append(idx3(ix, iy, iz, nx, ny))
            val_list.append(seg)

    gmat = csr_matrix(
        (
            np.array(val_list, dtype=np.float64),
            (np.array(row_list, dtype=np.int32), np.array(col_list, dtype=np.int32)),
        ),
        shape=(n_rays, nx * ny * nz),
    )
    ray_density = np.asarray((gmat > 0).sum(axis=0)).ravel()
    return gmat, ray_density


def broaden_ray_matrix_gaussian(
    gmat: csr_matrix,
    nx: int,
    ny: int,
    nz: int,
    sigma_xy_cells: float,
    sigma_z_cells: float = 0.0,
    relative_cutoff: float = 1e-4,
) -> csr_matrix:
    """Build a finite-frequency-inspired Gaussian ray-tube sensitivity matrix.

    The input Siddon row contains piecewise-constant path lengths.  Each row is
    convolved on the model grid and then renormalized to its original sum, so
    broadening changes only the spatial sensitivity distribution and never the
    modeled in-volume path length.  This is an empirical fat-ray approximation,
    not a Born/Fréchet banana-doughnut kernel.
    """
    sigmas = np.asarray(
        [sigma_xy_cells, sigma_xy_cells, sigma_z_cells], dtype=np.float64
    )
    if np.any(~np.isfinite(sigmas)) or np.any(sigmas < 0.0):
        raise ValueError("Gaussian ray-kernel sigma must be finite and non-negative")
    if not np.isfinite(relative_cutoff) or not 0.0 <= relative_cutoff < 1.0:
        raise ValueError("Gaussian ray-kernel relative cutoff must be in [0, 1)")
    expected_columns = nx * ny * nz
    if gmat.shape[1] != expected_columns:
        raise ValueError(
            f"ray matrix has {gmat.shape[1]} columns, expected {expected_columns}"
        )
    if not np.any(sigmas > 0.0) or gmat.nnz == 0:
        return gmat.copy().tocsr()

    source = gmat.tocsr()
    row_parts: List[np.ndarray] = []
    col_parts: List[np.ndarray] = []
    value_parts: List[np.ndarray] = []
    for row_index in range(source.shape[0]):
        start, stop = source.indptr[row_index], source.indptr[row_index + 1]
        if start == stop:
            continue
        original_sum = float(np.sum(source.data[start:stop]))
        if original_sum <= 0.0:
            continue
        field = np.zeros(expected_columns, dtype=np.float64)
        field[source.indices[start:stop]] = source.data[start:stop]
        field = field.reshape((nx, ny, nz), order="F")
        broadened = gaussian_filter(
            field,
            sigma=tuple(float(item) for item in sigmas),
            mode="constant",
            cval=0.0,
            truncate=3.0,
        ).ravel(order="F")
        peak = float(np.max(broadened))
        if peak <= 0.0:
            continue
        if relative_cutoff > 0.0:
            broadened[broadened < peak * relative_cutoff] = 0.0
        broadened_sum = float(np.sum(broadened))
        if broadened_sum <= 0.0:
            continue
        broadened *= original_sum / broadened_sum
        columns = np.flatnonzero(broadened > 0.0)
        row_parts.append(np.full(columns.size, row_index, dtype=np.int32))
        col_parts.append(columns.astype(np.int32, copy=False))
        value_parts.append(broadened[columns])

    if not value_parts:
        return csr_matrix(source.shape, dtype=np.float64)
    return csr_matrix(
        (
            np.concatenate(value_parts),
            (np.concatenate(row_parts), np.concatenate(col_parts)),
        ),
        shape=source.shape,
    )


def build_regularization(
    nx: int,
    ny: int,
    nz: int,
    dx: float = 1.0,
    dy: float = 1.0,
    dz: float = 1.0,
) -> csr_matrix:
    """Build a first-derivative operator with physical spacing normalization."""
    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []
    cnt = 0

    spacings = np.asarray([dx, dy, dz], dtype=np.float64)
    if not np.all(np.isfinite(spacings)) or np.any(spacings <= 0.0):
        raise ValueError("regularization grid spacing must be finite and positive")
    reference_spacing = float(np.median(spacings))
    wx, wy, wz = reference_spacing / spacings

    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                j = idx3(ix, iy, iz, nx, ny)
                if ix > 0:
                    rows += [cnt, cnt]
                    cols += [j, idx3(ix - 1, iy, iz, nx, ny)]
                    vals += [wx, -wx]
                    cnt += 1
                if ix < nx - 1:
                    rows += [cnt, cnt]
                    cols += [j, idx3(ix + 1, iy, iz, nx, ny)]
                    vals += [wx, -wx]
                    cnt += 1
                if iy > 0:
                    rows += [cnt, cnt]
                    cols += [j, idx3(ix, iy - 1, iz, nx, ny)]
                    vals += [wy, -wy]
                    cnt += 1
                if iy < ny - 1:
                    rows += [cnt, cnt]
                    cols += [j, idx3(ix, iy + 1, iz, nx, ny)]
                    vals += [wy, -wy]
                    cnt += 1
                if iz > 0:
                    rows += [cnt, cnt]
                    cols += [j, idx3(ix, iy, iz - 1, nx, ny)]
                    vals += [wz, -wz]
                    cnt += 1
                if iz < nz - 1:
                    rows += [cnt, cnt]
                    cols += [j, idx3(ix, iy, iz + 1, nx, ny)]
                    vals += [wz, -wz]
                    cnt += 1

    lmat = csr_matrix(
        (
            np.array(vals, dtype=np.float64),
            (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32)),
        ),
        shape=(cnt, nx * ny * nz),
    )
    return lmat


def build_curvature_regularization(
    nx: int,
    ny: int,
    nz: int,
    dx: float = 1.0,
    dy: float = 1.0,
    dz: float = 1.0,
    z_factor: float = 0.0,
) -> csr_matrix:
    """Build a centered second-derivative roughness operator."""
    spacings = np.asarray([dx, dy, dz], dtype=np.float64)
    if not np.all(np.isfinite(spacings)) or np.any(spacings <= 0.0):
        raise ValueError("curvature grid spacing must be finite and positive")
    if not np.isfinite(z_factor) or z_factor < 0.0:
        raise ValueError("curvature z factor must be finite and non-negative")
    reference_spacing = float(np.median(spacings))
    wx, wy, wz = (reference_spacing / spacings) ** 2
    wz *= z_factor

    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []
    row = 0
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                center = idx3(ix, iy, iz, nx, ny)
                if 0 < ix < nx - 1:
                    rows.extend([row, row, row])
                    cols.extend([
                        idx3(ix - 1, iy, iz, nx, ny), center,
                        idx3(ix + 1, iy, iz, nx, ny),
                    ])
                    vals.extend([wx, -2.0 * wx, wx])
                    row += 1
                if 0 < iy < ny - 1:
                    rows.extend([row, row, row])
                    cols.extend([
                        idx3(ix, iy - 1, iz, nx, ny), center,
                        idx3(ix, iy + 1, iz, nx, ny),
                    ])
                    vals.extend([wy, -2.0 * wy, wy])
                    row += 1
                if wz > 0.0 and 0 < iz < nz - 1:
                    rows.extend([row, row, row])
                    cols.extend([
                        idx3(ix, iy, iz - 1, nx, ny), center,
                        idx3(ix, iy, iz + 1, nx, ny),
                    ])
                    vals.extend([wz, -2.0 * wz, wz])
                    row += 1
    return csr_matrix(
        (
            np.asarray(vals, dtype=np.float64),
            (np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32)),
        ),
        shape=(row, nx * ny * nz),
    )


def build_coverage_weights(
    ray_density: np.ndarray,
    reliable_coverage: float,
    exponent: float = 1.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return confidence and background-prior weights from observed rays.

    ``ray_density`` is a geometry diagnostic, not a source of new data.  The
    confidence reaches one only after a cell meets the reliable-ray threshold;
    cells with fewer rays are progressively shrunk toward the reference
    model.  The exponent is deliberately greater than one in the engineering
    default so one- or two-ray cells cannot retain a strong isolated anomaly.

    The second result is a *row weight* for the quadratic background prior,
    rather than another coverage estimate.  It is the square root of the
    missing-confidence fraction so that the least-squares penalty is
    proportional to the missing support instead of its square.
    """
    density = np.asarray(ray_density, dtype=np.float64)
    if density.ndim != 1:
        raise ValueError("ray density must be a one-dimensional array")
    if not np.isfinite(reliable_coverage) or reliable_coverage <= 0.0:
        raise ValueError("reliable coverage must be finite and positive")
    if not np.isfinite(exponent) or exponent <= 0.0:
        raise ValueError("coverage weight exponent must be finite and positive")
    clipped_density = np.clip(np.nan_to_num(density, nan=0.0), 0.0, None)
    normalised = np.clip(clipped_density / float(reliable_coverage), 0.0, 1.0)
    confidence = np.power(normalised, float(exponent))
    confidence[clipped_density <= 0.0] = 0.0
    background_prior = np.sqrt(np.clip(1.0 - confidence, 0.0, 1.0))
    return confidence, background_prior


def build_bspline_projection(
    nx: int,
    ny: int,
    control_nx: int,
    control_ny: int,
    degree: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a tensor cubic B-spline design matrix and its pseudoinverse."""
    if degree < 1 or degree > 5:
        raise ValueError("B-spline degree must be between 1 and 5")

    def one_axis(n_points: int, n_control: int) -> np.ndarray:
        if n_points < 2:
            raise ValueError("B-spline projection needs at least two model cells")
        if n_control < degree + 1 or n_control > n_points:
            raise ValueError(
                f"B-spline controls must be in [{degree + 1}, {n_points}]"
            )
        internal_count = n_control - degree - 1
        internal = (
            np.linspace(0.0, 1.0, internal_count + 2, dtype=np.float64)[1:-1]
            if internal_count > 0
            else np.empty(0, dtype=np.float64)
        )
        knots = np.concatenate([
            np.zeros(degree + 1, dtype=np.float64),
            internal,
            np.ones(degree + 1, dtype=np.float64),
        ])
        locations = np.linspace(0.0, 1.0, n_points, dtype=np.float64)
        basis = np.asarray(
            BSpline(knots, np.eye(n_control), degree, extrapolate=False)(locations),
            dtype=np.float64,
        )
        basis[np.abs(basis) < 1e-15] = 0.0
        basis /= np.sum(basis, axis=1, keepdims=True)
        return basis

    basis_x = one_axis(nx, control_nx)
    basis_y = one_axis(ny, control_ny)
    design = np.kron(basis_y, basis_x)
    inverse = np.linalg.pinv(design, rcond=1e-10)
    return design, inverse


def apply_bspline_projection(
    slowness: np.ndarray,
    background_slowness: float,
    nx: int,
    ny: int,
    nz: int,
    design: np.ndarray,
    inverse: np.ndarray,
) -> np.ndarray:
    """Project each elevation's slowness anomaly onto a smooth spline basis."""
    values = np.asarray(slowness, dtype=np.float64)
    if values.size != nx * ny * nz:
        raise ValueError("slowness size does not match spline projection grid")
    if design.shape[0] != nx * ny or inverse.shape != (design.shape[1], nx * ny):
        raise ValueError("B-spline projection matrices have incompatible shapes")
    model = values.reshape((nx, ny, nz), order="F")
    projected = np.empty_like(model)
    for iz in range(nz):
        anomaly = (model[:, :, iz] - background_slowness).ravel(order="F")
        projected_anomaly = design @ (inverse @ anomaly)
        projected[:, :, iz] = projected_anomaly.reshape((nx, ny), order="F") + background_slowness
    return projected.ravel(order="F")


def build_hierarchical_block_basis(
    training_ray_density: np.ndarray,
    nx: int,
    ny: int,
    nz: int,
    split_ray_threshold: float,
    min_block_x: int,
    min_block_y: int,
    dx: float = 1.0,
    dy: float = 1.0,
) -> tuple[csr_matrix, np.ndarray, np.ndarray]:
    """Build a training-geometry-only adaptive binary block basis.

    Blocks containing a cell at or above ``split_ray_threshold`` are bisected
    along their longest splittable physical direction.  Empty/low-coverage
    siblings remain coarse, while dense branches refine down to the requested
    minimum block dimensions.  Elevation slices are partitioned independently.
    """
    density = np.asarray(training_ray_density, dtype=np.float64)
    if density.size != nx * ny * nz or np.any(~np.isfinite(density)):
        raise ValueError("training ray density does not match the model grid")
    if split_ray_threshold <= 0.0 or not np.isfinite(split_ray_threshold):
        raise ValueError("hierarchical split threshold must be positive and finite")
    if not 1 <= min_block_x <= nx or not 1 <= min_block_y <= ny:
        raise ValueError("hierarchical minimum block dimensions are invalid")
    if dx <= 0.0 or dy <= 0.0 or not np.isfinite(dx + dy):
        raise ValueError("hierarchical physical spacings must be positive and finite")

    density3d = density.reshape((nx, ny, nz), order="F")
    leaves: List[Tuple[int, int, int, int, int, float, float, float]] = []

    def visit(iz: int, x0: int, x1: int, y0: int, y1: int) -> None:
        block = density3d[x0:x1, y0:y1, iz]
        block_max = float(np.max(block)) if block.size else 0.0
        can_split_x = (x1 - x0) > min_block_x
        can_split_y = (y1 - y0) > min_block_y
        if block_max >= split_ray_threshold and (can_split_x or can_split_y):
            span_x = (x1 - x0) * dx if can_split_x else -1.0
            span_y = (y1 - y0) * dy if can_split_y else -1.0
            if span_x >= span_y:
                middle = x0 + (x1 - x0) // 2
                visit(iz, x0, middle, y0, y1)
                visit(iz, middle, x1, y0, y1)
            else:
                middle = y0 + (y1 - y0) // 2
                visit(iz, x0, x1, y0, middle)
                visit(iz, x0, x1, middle, y1)
            return
        leaves.append(
            (
                iz, x0, x1, y0, y1, block_max,
                float(np.mean(block)) if block.size else 0.0,
                float(np.sum(block)) if block.size else 0.0,
            )
        )

    for iz in range(nz):
        visit(iz, 0, nx, 0, ny)

    labels = np.empty((nx, ny, nz), dtype=np.int64)
    rows: List[int] = []
    cols: List[int] = []
    for column, (iz, x0, x1, y0, y1, _, _, _) in enumerate(leaves):
        labels[x0:x1, y0:y1, iz] = column
        for iy in range(y0, y1):
            for ix in range(x0, x1):
                rows.append(idx3(ix, iy, iz, nx, ny))
                cols.append(column)
    basis = csr_matrix(
        (
            np.ones(len(rows), dtype=np.float64),
            (np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32)),
        ),
        shape=(nx * ny * nz, len(leaves)),
    )
    leaf_table = np.asarray(leaves, dtype=np.float64)
    return basis, labels, leaf_table


def build_common_source_difference(source_ids: np.ndarray) -> csr_matrix:
    """Build normalized all-pair differences within each source event.

    For an event with ``k`` observations, each pair receives ``1/sqrt(k)``.
    Consequently, the sum of squared differential residuals equals the sum of
    squared event-centred residuals.  Event origin-time terms cancel exactly.
    """
    groups = np.asarray(source_ids)
    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []
    row = 0
    for source_id in np.unique(groups):
        indices = np.flatnonzero(groups == source_id)
        if indices.size < 2:
            continue
        scale = 1.0 / np.sqrt(float(indices.size))
        for left_pos in range(indices.size - 1):
            for right_pos in range(left_pos + 1, indices.size):
                rows.extend([row, row])
                cols.extend([int(indices[left_pos]), int(indices[right_pos])])
                vals.extend([scale, -scale])
                row += 1
    return csr_matrix(
        (
            np.asarray(vals, dtype=np.float64),
            (np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32)),
        ),
        shape=(row, groups.size),
    )


def haar_forward_2d(
    field: np.ndarray,
    levels: int,
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Return a padded orthonormal 2-D Haar transform.

    Edge padding to a multiple of ``2**levels`` keeps constant fields free of
    artificial detail coefficients on the odd 29 x 13 grids used by the mine
    data.  The returned metadata is consumed by :func:`haar_inverse_2d`.
    """
    values = np.asarray(field, dtype=np.float64)
    if values.ndim != 2 or values.size == 0:
        raise ValueError("Haar transform input must be a non-empty 2-D array")
    if levels < 1 or levels > 6:
        raise ValueError("wavelet levels must be between 1 and 6")

    nx, ny = values.shape
    block = 2**levels
    pad_x = (-nx) % block
    pad_y = (-ny) % block
    coeff = np.pad(values, ((0, pad_x), (0, pad_y)), mode="edge").copy()
    active_x, active_y = coeff.shape
    root_two = np.sqrt(2.0)

    for _ in range(levels):
        current = coeff[:active_x, :active_y].copy()
        low_x = (current[0::2, :] + current[1::2, :]) / root_two
        high_x = (current[0::2, :] - current[1::2, :]) / root_two
        x_transformed = np.concatenate([low_x, high_x], axis=0)
        low_y = (x_transformed[:, 0::2] + x_transformed[:, 1::2]) / root_two
        high_y = (x_transformed[:, 0::2] - x_transformed[:, 1::2]) / root_two
        coeff[:active_x, :active_y] = np.concatenate([low_y, high_y], axis=1)
        active_x //= 2
        active_y //= 2

    return coeff, (nx, ny, levels)


def haar_inverse_2d(
    coeff: np.ndarray,
    metadata: tuple[int, int, int],
) -> np.ndarray:
    """Invert :func:`haar_forward_2d` and remove its edge padding."""
    transformed = np.asarray(coeff, dtype=np.float64).copy()
    if transformed.ndim != 2 or transformed.size == 0:
        raise ValueError("Haar coefficients must be a non-empty 2-D array")
    nx, ny, levels = metadata
    if levels < 1 or levels > 6:
        raise ValueError("wavelet levels must be between 1 and 6")
    if transformed.shape[0] % (2**levels) or transformed.shape[1] % (2**levels):
        raise ValueError("Haar coefficient shape is incompatible with the level count")

    root_two = np.sqrt(2.0)
    active_x = transformed.shape[0] // (2**levels)
    active_y = transformed.shape[1] // (2**levels)
    for _ in range(levels):
        full_x = active_x * 2
        full_y = active_y * 2
        current = transformed[:full_x, :full_y].copy()

        low_y = current[:, :active_y]
        high_y = current[:, active_y:full_y]
        y_restored = np.empty((full_x, full_y), dtype=np.float64)
        y_restored[:, 0::2] = (low_y + high_y) / root_two
        y_restored[:, 1::2] = (low_y - high_y) / root_two

        low_x = y_restored[:active_x, :]
        high_x = y_restored[active_x:full_x, :]
        restored = np.empty((full_x, full_y), dtype=np.float64)
        restored[0::2, :] = (low_x + high_x) / root_two
        restored[1::2, :] = (low_x - high_x) / root_two
        transformed[:full_x, :full_y] = restored
        active_x = full_x
        active_y = full_y

    return transformed[:nx, :ny]


def apply_joint_sparsity_wavelet(
    slowness: np.ndarray,
    reference_slowness: Union[float, np.ndarray],
    nx: int,
    ny: int,
    nz: int,
    levels: int,
    threshold_factor: float,
) -> tuple[np.ndarray, float, float]:
    """Proximal Haar shrinkage for a wavelet + TV joint-sparsity model.

    Only detail coefficients of the slowness perturbation are thresholded; the
    coarsest approximation is retained.  ``threshold_factor`` multiplies the
    median magnitude of non-zero detail coefficients, making the tuning stable
    across data sets with different background velocities.
    """
    model = np.asarray(slowness, dtype=np.float64)
    if model.size != nx * ny * nz:
        raise ValueError("slowness size does not match the inversion grid")
    if not np.isfinite(threshold_factor) or threshold_factor < 0.0:
        raise ValueError("wavelet threshold factor must be finite and non-negative")
    reference = np.broadcast_to(
        np.asarray(reference_slowness, dtype=np.float64), model.shape
    )
    perturbation = (model - reference).reshape((nx, ny, nz), order="F")

    transformed_slices: List[Tuple[np.ndarray, Tuple[int, int, int], np.ndarray]] = []
    detail_values: List[np.ndarray] = []
    for iz in range(nz):
        coeff, metadata = haar_forward_2d(perturbation[:, :, iz], levels)
        coarse_x = coeff.shape[0] // (2**levels)
        coarse_y = coeff.shape[1] // (2**levels)
        detail_mask = np.ones(coeff.shape, dtype=bool)
        detail_mask[:coarse_x, :coarse_y] = False
        transformed_slices.append((coeff, metadata, detail_mask))
        detail_values.append(np.abs(coeff[detail_mask]))

    all_details = np.concatenate(detail_values) if detail_values else np.empty(0)
    positive_details = all_details[all_details > 1e-15]
    robust_scale = float(np.median(positive_details)) if positive_details.size else 0.0
    absolute_threshold = threshold_factor * robust_scale
    nonzero_before = int(np.count_nonzero(all_details > 1e-15))
    nonzero_after = 0

    restored = np.empty_like(perturbation)
    for iz, (coeff, metadata, detail_mask) in enumerate(transformed_slices):
        if absolute_threshold > 0.0:
            details = coeff[detail_mask]
            coeff[detail_mask] = np.sign(details) * np.maximum(
                np.abs(details) - absolute_threshold, 0.0
            )
        nonzero_after += int(np.count_nonzero(np.abs(coeff[detail_mask]) > 1e-15))
        restored[:, :, iz] = haar_inverse_2d(coeff, metadata)

    retained_fraction = (
        float(nonzero_after / nonzero_before) if nonzero_before else 1.0
    )
    result = reference + restored.ravel(order="F")
    return result, absolute_threshold, retained_fraction


def interpolate_z_slice(volume: np.ndarray, zc: np.ndarray, target_z: float) -> np.ndarray:
    """Return an exact target-elevation slice by linear interpolation."""
    if target_z <= zc[0]:
        return volume[:, :, 0].copy()
    if target_z >= zc[-1]:
        return volume[:, :, -1].copy()

    upper = int(np.searchsorted(zc, target_z, side="right"))
    lower = upper - 1
    span = float(zc[upper] - zc[lower])
    weight = 0.0 if span <= 0 else float((target_z - zc[lower]) / span)
    return (1.0 - weight) * volume[:, :, lower] + weight * volume[:, :, upper]


def parse_slice_targets(text: str, zmin: float, zmax: float) -> List[float]:
    if text.strip():
        return [float(item.strip()) for item in text.split(",") if item.strip()]
    span = zmax - zmin
    return [float(zmin + span * frac) for frac in (0.20, 0.40, 0.50, 0.60, 0.80)]


def estimate_velocity_ranges(v_app: np.ndarray) -> Tuple[float, float, float, float, float]:
    """Estimate robust QC/model limits from observed apparent velocities."""
    values = np.asarray(v_app, dtype=np.float64)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size < 5:
        raise ValueError("有效表观速度不足，无法自动估计波速范围。")

    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    robust_sigma = max(1.4826 * mad, median * 0.005)
    q_low, q_high = np.percentile(values, [0.5, 99.5])

    # Keep the central robust distribution.  The previous min/max directions
    # expanded the interval toward outliers and could produce ranges such as
    # 100-8300 m/s from a 3600 m/s dataset.
    qc_low = max(100.0, max(float(q_low), median - 6.0 * robust_sigma))
    qc_high = min(float(q_high), median + 6.0 * robust_sigma)
    qc_values = values[(values >= qc_low) & (values <= qc_high)]
    if qc_values.size < max(5, int(values.size * 0.5)):
        qc_low, qc_high = float(values.min()), float(values.max())
        qc_values = values

    q5, q95 = np.percentile(qc_values, [5.0, 95.0])
    padding = max(median * 0.08, float(q95 - q5) * 0.25)
    padding = min(padding, median * 0.20)
    model_low = max(100.0, float(q5 - padding))
    model_high = float(q95 + padding)
    return qc_low, qc_high, model_low, model_high, float(np.median(qc_values))


def estimate_event_centered_qc(
    source_ids: np.ndarray,
    ray_dist: np.ndarray,
    travel_s: np.ndarray,
    reference_velocity: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove only each event's common time shift for observation QC.

    The returned corrected times are used to decide whether an individual pick
    is an outlier.  The inversion still receives the original observed times;
    common shifts are represented by explicit source-static parameters.
    """
    if reference_velocity <= 0:
        raise ValueError("逐事件走时质控需要正的参考背景速度。")
    shift_per_ray = np.zeros(travel_s.size, dtype=np.float64)
    for source_id in np.unique(source_ids):
        mask = source_ids == source_id
        if np.count_nonzero(mask) < 2:
            continue
        residual = travel_s[mask] - ray_dist[mask] / reference_velocity
        shift_per_ray[mask] = float(np.median(residual))
    qc_travel_s = travel_s - shift_per_ray
    qc_velocity = np.full(travel_s.shape, np.nan, dtype=np.float64)
    positive = qc_travel_s > 0
    qc_velocity[positive] = ray_dist[positive] / qc_travel_s[positive]
    return qc_travel_s, qc_velocity, shift_per_ray


def validate_slice_targets(targets: List[float], zc: np.ndarray, dz: float) -> None:
    low = float(zc[0] - dz / 2.0)
    high = float(zc[-1] + dz / 2.0)
    bad = [zt for zt in targets if zt < low or zt > high]
    if bad:
        bad_text = ", ".join(f"{zt:g}" for zt in bad)
        raise ValueError(
            f"切片标高 {bad_text} 超出当前反演体 Z 范围 [{low:g}, {high:g}]。"
            "请修改切片标高或扩大模型 Z 范围。"
        )


def save_rms_curve(
    output_dir: Path,
    rms_hist: List[Tuple[int, float, float, int, int]],
    validation_hist: List[Tuple[int, float]],
) -> None:
    if not rms_hist:
        return
    iters = [item[0] for item in rms_hist]
    before = [item[1] * 1000.0 for item in rms_hist]
    after = [item[2] * 1000.0 for item in rms_hist]

    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=150, facecolor="white")
    ax.plot(iters, before, marker="o", linewidth=1.8, label="迭代前 RMS")
    ax.plot(iters, after, marker="s", linewidth=1.8, label="迭代后 RMS")
    if validation_hist:
        ax.plot(
            [item[0] for item in validation_hist],
            [item[1] * 1000.0 for item in validation_hist],
            marker="^",
            linewidth=1.8,
            label="验证集 RMS",
        )
    ax.set_title("反演 RMS 收敛曲线", fontsize=14, fontweight="bold")
    ax.set_xlabel("外层迭代次数")
    ax.set_ylabel("RMS 走时残差 (ms)")
    ax.grid(True, color="#CBD5E1", alpha=0.55)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "rms_convergence.png", dpi=150)
    plt.close(fig)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description="执行层析反演并导出目标切片图")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=project_root / "反演输入数据集" / "煤矿反演数据集_自动生成_高置信.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "反演结果切片",
    )
    parser.add_argument("--mode", type=str, default="generic", choices=["generic"])
    parser.add_argument("--expected-sources", type=int, default=0)
    parser.add_argument("--expected-stations-per-source", type=int, default=0)
    parser.add_argument(
        "--plot-style",
        type=str,
        default="rectangular",
        choices=["rectangular", "cube"],
        help="通用矩形模型成果图样式；cube为兼容旧配置的别名",
    )

    parser.add_argument("--x-min", type=float, default=0.0)
    parser.add_argument("--x-max", type=float, default=1000.0)
    parser.add_argument("--y-min", type=float, default=0.0)
    parser.add_argument("--y-max", type=float, default=1000.0)
    parser.add_argument("--z-min", type=float, default=0.0)
    parser.add_argument("--z-max", type=float, default=1000.0)
    parser.add_argument("--dx", type=float, default=50.0)
    parser.add_argument("--dy", type=float, default=50.0)
    parser.add_argument("--dz", type=float, default=50.0)
    parser.add_argument("--nx-nodes", type=int, default=0, help="X方向精确节点数；0表示按dx生成")
    parser.add_argument("--ny-nodes", type=int, default=0, help="Y方向精确节点数；0表示按dy生成")
    parser.add_argument("--nz-nodes", type=int, default=0, help="Z方向精确节点数；0表示按dz生成")
    parser.add_argument(
        "--initial-model",
        type=Path,
        default=None,
        help="可选velocity_model.npz；插值到当前网格后作为多尺度反演初始模型",
    )
    parser.add_argument("--slice-z", type=str, default="", help="逗号分隔的切片标高；为空时按反演体范围自动取 5 层")
    parser.add_argument("--vmin-qc", type=float, default=0.0, help="0表示根据走时自动估计")
    parser.add_argument("--vmax-qc", type=float, default=0.0, help="0表示根据走时自动估计")
    parser.add_argument("--vmin-model", type=float, default=0.0, help="0表示根据走时自动估计")
    parser.add_argument("--vmax-model", type=float, default=0.0, help="0表示根据走时自动估计")
    parser.add_argument(
        "--background-velocity",
        type=float,
        default=0.0,
        help="覆盖不足区域的参考背景速度；0表示使用表观速度中位数",
    )
    parser.add_argument(
        "--min-ray-coverage",
        type=float,
        default=0.0,
        help="可靠网格所需射线数；0表示根据覆盖分布自动估计",
    )

    parser.add_argument(
        "--coverage-weight-exponent",
        type=float,
        default=1.5,
        help=(
            "覆盖率置信权重指数；大于1会更快抑制低覆盖单元的异常，"
            "仅作用于覆盖先验和展示权重"
        ),
    )
    parser.add_argument(
        "--deep-reparameterization",
        action="store_true",
        default=False,
        help="Use the validated untrained coordinate-MLP parameterization",
    )
    parser.add_argument(
        "--no-deep-reparameterization",
        action="store_false",
        dest="deep_reparameterization",
    )
    parser.add_argument("--deep-reparam-width", type=int, default=24)
    parser.add_argument("--deep-reparam-depth", type=int, default=3)
    parser.add_argument("--deep-reparam-learning-rate", type=float, default=0.004)
    parser.add_argument("--deep-reparam-full-epochs", type=int, default=350)
    parser.add_argument("--deep-reparam-max-epochs", type=int, default=1600)
    parser.add_argument("--deep-reparam-starts", type=int, default=3)
    parser.add_argument("--deep-reparam-huber-ms", type=float, default=8.0)
    parser.add_argument("--deep-reparam-output-l2", type=float, default=0.0)
    parser.add_argument(
        "--deep-reparam-fourier-bands",
        type=int,
        default=0,
        help="DNR多尺度坐标正弦/余弦频带数；0保持原始xyz参数化",
    )
    parser.add_argument(
        "--deep-reparam-differential-loss-fraction",
        type=float,
        default=0.0,
        help="同事件差分走时损失占比，范围[0,1]；0保持原始损失",
    )
    parser.add_argument(
        "--deep-reparam-tv",
        type=float,
        default=0.0,
        help="DNR相对速度异常的TV边缘保持权重；0表示关闭",
    )
    parser.add_argument(
        "--deep-reparam-tv-epsilon",
        type=float,
        default=1.0e-3,
        help="DNR TV的Charbonnier平滑参数",
    )
    parser.add_argument(
        "--deep-reparam-event-static-shrinkage",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--deep-reparam-receiver-statics",
        action="store_true",
        default=False,
        help="实验性DNR台站静校正；须通过按事件留出验证后再用于正式结果",
    )
    parser.add_argument(
        "--deep-reparam-receiver-static-shrinkage",
        type=float,
        default=20.0,
    )
    parser.add_argument(
        "--deep-reparam-receiver-static-max-ms",
        type=float,
        default=12.0,
    )
    parser.add_argument(
        "--deep-reparam-static-profile-iterations",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--deep-reparam-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )

    parser.add_argument("--n-outer", type=int, default=18)
    parser.add_argument(
        "--solver-method",
        choices=("sirt", "lsqr"),
        default="sirt",
        help="线性增量求解器；SIRT 是当前 WaveCT 主反演入口，LSQR 用于回退对照",
    )
    parser.add_argument(
        "--script-compatible-sirt",
        action="store_true",
        help="复现 CT_shi_SIRT_Automatic_tuning_global_optimum.py 的10米绝对慢度SIRT流程",
    )
    parser.add_argument(
        "--reference-profile",
        choices=("auto", "probe_728", "de"),
        default="auto",
        help="兼容 SIRT 参数档案；auto 仅在匹配 6.9-6.16 探针几何时使用 probe_728，否则执行 DE 调参",
    )
    parser.add_argument(
        "--reference-dataset-root",
        type=Path,
        default=None,
        help=(
            "optional directory containing supplied Surfer velocity TXT grids; "
            "used only for post-solve candidate selection, never as inversion input"
        ),
    )
    parser.add_argument("--n-lsqr", type=int, default=160)
    parser.add_argument(
        "--sirt-iterations",
        type=int,
        default=300,
        help="每轮 SIRT 内迭代次数；0 表示复用 --n-lsqr",
    )
    parser.add_argument(
        "--sirt-omega",
        type=float,
        default=0.03,
        help="SIRT 松弛因子，默认沿用候选脚本在728数据上的稳定量级",
    )
    parser.add_argument(
        "--sirt-step-damp",
        type=float,
        default=1.0,
        help="SIRT 外层增量步长；接受判据仍由训练/验证走时控制",
    )
    parser.add_argument(
        "--sirt-tolerance",
        type=float,
        default=1.0e-8,
        help="SIRT 增广方程相对残差停止阈值",
    )
    parser.add_argument(
        "--sirt-auto-tune",
        action="store_true",
        default=True,
        help="在首轮用事件留出目标自动搜索 alpha 和 SIRT 松弛因子",
    )
    parser.add_argument(
        "--no-sirt-auto-tune",
        action="store_false",
        dest="sirt_auto_tune",
    )
    parser.add_argument("--sirt-tune-maxiter", type=int, default=15)
    parser.add_argument("--sirt-tune-popsize", type=int, default=6)
    parser.add_argument(
        "--sirt-tune-iterations",
        type=int,
        default=0,
        help="自动调参试算内迭代次数；0 表示复用正式 SIRT 迭代次数",
    )
    parser.add_argument("--min-rays", type=int, default=20)
    parser.add_argument("--alpha-reg", type=float, default=5.0)
    parser.add_argument("--step-damp", type=float, default=0.2)
    parser.add_argument("--early-stop-patience", type=int, default=2)
    parser.add_argument("--early-stop-tol", type=float, default=1e-6)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument(
        "--validation-source-ids",
        type=str,
        default="",
        help="逗号分隔的固定验证震源编号；为空时按随机种子分组抽取",
    )
    parser.add_argument(
        "--huber-delta",
        type=float,
        default=1.5,
        help="Huber转折点相对于当前稳健尺度sigma的无量纲倍数，不是毫秒",
    )
    parser.add_argument("--background-damping", type=float, default=1.5)
    parser.add_argument(
        "--model-damping",
        type=float,
        default=0.0,
        help="对全部网格施加相对背景慢度的零阶阻尼；0表示关闭",
    )
    parser.add_argument(
        "--regularize-total-model",
        action="store_true",
        default=True,
        help="将一阶和二阶正则作用于更新后的总慢度场；关闭时兼容旧版增量平滑",
    )
    parser.add_argument(
        "--no-regularize-total-model",
        action="store_false",
        dest="regularize_total_model",
    )
    parser.add_argument(
        "--curvature-reg-factor",
        type=float,
        default=0.25,
        help="相对于alpha-reg的二阶曲率权重；0表示关闭",
    )
    parser.add_argument(
        "--curvature-z-factor",
        type=float,
        default=0.5,
        help="Z向二阶曲率相对系数；三层薄层模型默认0，仅约束横向曲率",
    )
    parser.add_argument(
        "--spline-projection",
        action="store_true",
        default=False,
        help="将每个标高的慢度异常投影到低维三次B样条控制网格",
    )
    parser.add_argument(
        "--no-spline-projection",
        action="store_false",
        dest="spline_projection",
    )
    parser.add_argument("--spline-control-nx", type=int, default=15)
    parser.add_argument("--spline-control-ny", type=int, default=7)
    parser.add_argument(
        "--spline-projection-strength",
        type=float,
        default=1.0,
        help="每轮向B样条子空间收缩的比例，范围[0,1]；1表示完全投影",
    )
    parser.add_argument(
        "--hierarchical-parameterization",
        action="store_true",
        default=False,
        help="仅按训练事件射线覆盖构建自适应层次块并在约化参数空间求解",
    )
    parser.add_argument(
        "--no-hierarchical-parameterization",
        action="store_false",
        dest="hierarchical_parameterization",
    )
    parser.add_argument(
        "--hierarchical-split-rays",
        type=float,
        default=5.0,
        help="层次块继续细分所需的训练射线最大覆盖阈值",
    )
    parser.add_argument("--hierarchical-min-block-x", type=int, default=2)
    parser.add_argument("--hierarchical-min-block-y", type=int, default=2)
    parser.add_argument(
        "--ray-kernel-sigma-xy-cells",
        type=float,
        default=0.0,
        help=(
            "有限频率启发的高斯射线管横向标准差（网格单元数）；"
            "0表示使用原始Siddon细射线"
        ),
    )
    parser.add_argument(
        "--ray-kernel-sigma-z-cells",
        type=float,
        default=0.0,
        help="高斯射线管Z向标准差（网格单元数）；薄层模型建议先保持0",
    )
    parser.add_argument(
        "--ray-kernel-relative-cutoff",
        type=float,
        default=1e-4,
        help="高斯射线管稀疏截断阈值（相对每行最大权重）",
    )
    parser.add_argument("--ray-length-normalization", action="store_true", default=False)
    parser.add_argument("--no-ray-length-normalization", action="store_false", dest="ray_length_normalization")
    parser.add_argument("--edge-preserving-tv", action="store_true", default=False)
    parser.add_argument("--no-edge-preserving-tv", action="store_false", dest="edge_preserving_tv")
    parser.add_argument(
        "--tv-epsilon",
        type=float,
        default=0.0,
        help="慢度梯度TV平滑参数(s/m)；0表示每轮稳健自动估计",
    )
    parser.add_argument("--joint-sparsity", action="store_true", default=False)
    parser.add_argument("--no-joint-sparsity", action="store_false", dest="joint_sparsity")
    parser.add_argument(
        "--wavelet-levels",
        type=int,
        default=2,
        help="逐标高二维Haar小波分解层数(1-6)",
    )
    parser.add_argument(
        "--wavelet-threshold-factor",
        type=float,
        default=0.80,
        help="小波细节软阈值相对系数；建议通过按事件留出验证选择",
    )
    parser.add_argument("--differential-times", action="store_true", default=False)
    parser.add_argument(
        "--no-differential-times", action="store_false", dest="differential_times"
    )
    parser.add_argument(
        "--differential-weight",
        type=float,
        default=1.0,
        help="同震源台站差分走时在联合目标函数中的权重",
    )
    parser.add_argument("--random-seed", type=int, default=20260713)
    parser.add_argument("--invert-source-statics", action="store_true", default=True)
    parser.add_argument("--no-source-statics", action="store_false", dest="invert_source_statics")
    parser.add_argument("--global-time-damping", type=float, default=1.0)
    parser.add_argument("--source-static-damping", type=float, default=10.0)
    parser.add_argument("--max-time-correction", type=float, default=0.05)
    parser.add_argument(
        "--event-centered-qc",
        action="store_true",
        default=False,
        help="质控时先去除同一事件各台站共有的起时偏差，原始走时仍进入联合反演",
    )
    parser.add_argument(
        "--no-event-centered-qc",
        action="store_false",
        dest="event_centered_qc",
    )
    parser.add_argument(
        "--allow-nonpositive-observed-times",
        action="store_true",
        default=False,
        help=(
            "允许名义发震时刻晚于人工P标记；必须同时启用逐事件质控和震源时刻联合反演"
        ),
    )
    parser.add_argument(
        "--allow-outside-rays",
        action="store_true",
        default=False,
        help=(
            "允许震源或台站位于反演体外；体外路径按背景速度计算，"
            "只反演穿过目标区域的路径"
        ),
    )
    parser.add_argument(
        "--no-outside-rays",
        action="store_false",
        dest="allow_outside_rays",
    )
    parser.add_argument(
        "--export-deliverables",
        action="store_true",
        default=True,
        help="导出定量速度模型、Surfer兼容TXT和成果清单",
    )
    parser.add_argument(
        "--no-export-deliverables",
        action="store_false",
        dest="export_deliverables",
        help="仅供自动选模的临时交叉验证运行使用",
    )

    args = parser.parse_args()

    if args.script_compatible_sirt:
        if args.solver_method != "sirt":
            raise ValueError("script-compatible-sirt requires --solver-method sirt")
        from wave_ct.reference_sirt import run_reference_sirt

        run_reference_sirt(
            args.input_csv,
            args.output_dir,
            de_maxiter=args.sirt_tune_maxiter,
            de_popsize=args.sirt_tune_popsize,
            tune_iterations=args.sirt_tune_iterations or 20,
            final_iterations=args.sirt_iterations or 300,
            profile=args.reference_profile,
            reference_dataset_root=args.reference_dataset_root,
        )
        return

    if args.n_outer < 1 or args.n_lsqr < 1:
        raise ValueError("n-outer and n-lsqr must be positive")
    if args.sirt_iterations < 0:
        raise ValueError("sirt-iterations must be non-negative")
    if (
        not np.isfinite(args.sirt_omega)
        or not 0.0 < args.sirt_omega <= 1.0
        or not np.isfinite(args.sirt_step_damp)
        or not 0.0 < args.sirt_step_damp <= 1.0
        or not np.isfinite(args.sirt_tolerance)
        or args.sirt_tolerance <= 0.0
    ):
        raise ValueError("SIRT omega, step damp and tolerance are invalid")
    if (
        args.sirt_tune_maxiter < 1
        or args.sirt_tune_popsize < 2
        or args.sirt_tune_iterations < 0
    ):
        raise ValueError("SIRT auto-tuning iteration settings are invalid")

    if args.deep_reparameterization:
        if args.deep_reparam_width < 2 or args.deep_reparam_depth < 1:
            raise ValueError("deep reparameterization width/depth are invalid")
        if (
            args.deep_reparam_full_epochs < 1
            or args.deep_reparam_max_epochs < 1
            or args.deep_reparam_starts < 1
        ):
            raise ValueError("deep reparameterization epoch/start counts must be positive")
        if (
            not np.isfinite(args.deep_reparam_tv)
            or args.deep_reparam_tv < 0.0
            or not np.isfinite(args.deep_reparam_tv_epsilon)
            or args.deep_reparam_tv_epsilon <= 0.0
        ):
            raise ValueError("deep reparameterization TV settings are invalid")
        if (
            args.deep_reparam_fourier_bands < 0
            or not 0.0 <= args.deep_reparam_differential_loss_fraction <= 1.0
        ):
            raise ValueError("deep reparameterization multiscale settings are invalid")
        if (
            not np.isfinite(args.deep_reparam_receiver_static_shrinkage)
            or args.deep_reparam_receiver_static_shrinkage < 0.0
            or not np.isfinite(args.deep_reparam_receiver_static_max_ms)
            or args.deep_reparam_receiver_static_max_ms <= 0.0
            or args.deep_reparam_static_profile_iterations < 1
        ):
            raise ValueError(
                "deep reparameterization receiver-static settings are invalid"
            )
        if not args.invert_source_statics:
            raise ValueError(
                "deep reparameterization requires source-time profiling"
            )
        if (
            args.ray_kernel_sigma_xy_cells > 0.0
            or args.ray_kernel_sigma_z_cells > 0.0
        ):
            raise ValueError(
                "the validated deep candidate requires Siddon centerline rays"
            )
        # These LSQR-only switches are deliberately disabled so the report
        # cannot claim that regularizers unused by the neural solver ran.
        args.hierarchical_parameterization = False
        args.joint_sparsity = False
        args.differential_times = False
        args.spline_projection = False
        args.ray_length_normalization = False
        args.edge_preserving_tv = False
        args.regularize_total_model = False
        args.model_damping = 0.0
        args.event_centered_qc = True

    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")

    contract_stats = validate_dataset_contract(
        args.input_csv,
        bounds=(args.x_min, args.x_max, args.y_min, args.y_max, args.z_min, args.z_max),
        expected_sources=args.expected_sources,
        expected_stations_per_source=args.expected_stations_per_source,
        allow_outside_endpoints=args.allow_outside_rays,
        allow_event_time_correction=(
            args.allow_nonpositive_observed_times
            and args.event_centered_qc
            and args.invert_source_statics
        ),
    )
    allow_event_time_correction = (
        args.allow_nonpositive_observed_times
        and args.event_centered_qc
        and args.invert_source_statics
    )
    source_ids, sx, sy, sz, rx, ry, rz, tt_s = load_inversion_rows(
        args.input_csv,
        allow_event_time_correction=allow_event_time_correction,
    )
    observation_row_indices = np.arange(source_ids.size, dtype=np.int64)
    ray_dist = np.sqrt((rx - sx) ** 2 + (ry - sy) ** 2 + (rz - sz) ** 2)
    v_app = np.divide(
        ray_dist,
        tt_s,
        out=np.full_like(ray_dist, np.nan),
        where=np.abs(tt_s) > np.finfo(np.float64).eps,
    )

    qc_reference_velocity = args.background_velocity
    if qc_reference_velocity <= 0:
        preliminary = v_app[np.isfinite(v_app) & (v_app > 0)]
        if preliminary.size < 5:
            raise ValueError("有效表观速度不足，无法估计逐事件质控参考速度。")
        q10, q90 = np.percentile(preliminary, [10.0, 90.0])
        central = preliminary[(preliminary >= q10) & (preliminary <= q90)]
        qc_reference_velocity = float(np.median(central))
    if args.event_centered_qc:
        qc_tt_s, v_app_qc, qc_event_shift_s = estimate_event_centered_qc(
            source_ids, ray_dist, tt_s, qc_reference_velocity
        )
    else:
        qc_tt_s = tt_s.copy()
        v_app_qc = v_app.copy()
        qc_event_shift_s = np.zeros(tt_s.size, dtype=np.float64)

    finite_base = (
        (qc_tt_s > 0)
        & (ray_dist > 1.0)
        & np.isfinite(v_app_qc)
        & (v_app_qc > 0)
    )
    auto_qc_low, auto_qc_high, auto_model_low, auto_model_high, auto_background = (
        estimate_velocity_ranges(v_app_qc[finite_base])
    )
    qc_auto = args.vmin_qc <= 0 or args.vmax_qc <= 0
    model_auto = args.vmin_model <= 0 or args.vmax_model <= 0
    if qc_auto:
        args.vmin_qc, args.vmax_qc = auto_qc_low, auto_qc_high
    if model_auto:
        args.vmin_model, args.vmax_model = auto_model_low, auto_model_high
    if args.vmin_qc >= args.vmax_qc or args.vmin_model >= args.vmax_model:
        raise ValueError("波速上下限无效：下限必须小于上限，或全部填写0启用自动估计。")

    valid = (
        (qc_tt_s > 0)
        & (ray_dist > 1.0)
        & np.isfinite(v_app_qc)
        & (v_app_qc >= args.vmin_qc)
        & (v_app_qc <= args.vmax_qc)
    )
    dropped_by_velocity_qc = int(np.sum(~valid))
    sx = sx[valid]
    source_ids = source_ids[valid]
    sy = sy[valid]
    sz = sz[valid]
    rx = rx[valid]
    ry = ry[valid]
    rz = rz[valid]
    tt_s = tt_s[valid]
    qc_tt_s = qc_tt_s[valid]
    v_app = v_app[valid]
    v_app_qc = v_app_qc[valid]
    qc_event_shift_s = qc_event_shift_s[valid]
    ray_dist = ray_dist[valid]
    observation_row_indices = observation_row_indices[valid]

    filtered_lengths = {
        sx.size, sy.size, sz.size, rx.size, ry.size, rz.size, tt_s.size,
        v_app.size, v_app_qc.size, qc_event_shift_s.size, ray_dist.size,
    }
    if len(filtered_lengths) != 1:
        raise RuntimeError("质控筛选后的射线字段长度不一致。")

    if sx.size < args.min_rays:
        raise RuntimeError(
            f"有效射线过少: {sx.size}，不足以稳定反演。"
            f" 请先检查输入到时；当前速度质控范围为 {args.vmin_qc}-{args.vmax_qc} m/s。"
        )

    if not (args.x_min < args.x_max and args.y_min < args.y_max and args.z_min < args.z_max):
        raise ValueError("模型范围无效：x/y/z 的最小值必须小于最大值。")

    background_velocity = (
        args.background_velocity if args.background_velocity > 0 else auto_background
    )
    v0 = float(np.clip(background_velocity, args.vmin_model, args.vmax_model))

    xnodes, ynodes, znodes, nx, ny, nz, xc, yc, zc = build_grid(
        args.x_min, args.x_max,
        args.y_min, args.y_max,
        args.z_min, args.z_max,
        args.dx, args.dy, args.dz,
        args.nx_nodes, args.ny_nodes, args.nz_nodes,
    )
    grid_dx = float(xnodes[1] - xnodes[0])
    grid_dy = float(ynodes[1] - ynodes[0])
    grid_dz = float(znodes[1] - znodes[0])

    gmat, ray_density = build_siddon_matrix(
        sx, sy, sz, rx, ry, rz, xnodes, ynodes, znodes, nx, ny, nz
    )
    if gmat.nnz == 0:
        raise RuntimeError("G 矩阵为空，未形成有效射线路径。")

    traced_length = np.asarray(gmat.sum(axis=1)).ravel()
    full_tt_s = tt_s.copy()
    outside_length = np.maximum(ray_dist - traced_length, 0.0)
    roi_dropped_rays = 0
    if args.allow_outside_rays:
        modeled_tt_s = tt_s - outside_length / v0
        qc_modeled_tt_s = qc_tt_s - outside_length / v0
        roi_time_is_positive = (
            qc_modeled_tt_s > 0
            if allow_event_time_correction
            else modeled_tt_s > 0
        )
        roi_valid = (
            (traced_length > 1e-6)
            & (traced_length <= ray_dist + 1e-5)
            & np.isfinite(modeled_tt_s)
            & np.isfinite(qc_modeled_tt_s)
            & roi_time_is_positive
        )
        roi_dropped_rays = int(np.count_nonzero(~roi_valid))
        if roi_dropped_rays:
            source_ids = source_ids[roi_valid]
            sx, sy, sz = sx[roi_valid], sy[roi_valid], sz[roi_valid]
            rx, ry, rz = rx[roi_valid], ry[roi_valid], rz[roi_valid]
            v_app = v_app[roi_valid]
            v_app_qc = v_app_qc[roi_valid]
            qc_event_shift_s = qc_event_shift_s[roi_valid]
            ray_dist = ray_dist[roi_valid]
            observation_row_indices = observation_row_indices[roi_valid]
            full_tt_s = full_tt_s[roi_valid]
            qc_tt_s = qc_tt_s[roi_valid]
            traced_length = traced_length[roi_valid]
            outside_length = outside_length[roi_valid]
            modeled_tt_s = modeled_tt_s[roi_valid]
            gmat = gmat[roi_valid]
        tt_s = modeled_tt_s
        ray_density = np.asarray((gmat > 0).sum(axis=0)).ravel()
        if tt_s.size < args.min_rays:
            raise RuntimeError(
                f"穿过目标反演区域且具有正走时的射线过少: {tt_s.size}。"
                "请扩大模型范围或检查事件时刻/P波标记。"
            )
        path_length_error = np.abs(traced_length + outside_length - ray_dist)
    else:
        path_length_error = np.abs(traced_length - ray_dist)
        max_path_length_error = float(path_length_error.max())
        if max_path_length_error > 1e-5:
            raise RuntimeError(
                "射线路径长度守恒检查失败："
                f"最大误差 {max_path_length_error:.6f} m。"
                "请确认反演网格完整覆盖所有震源和台站坐标，"
                "或启用 --allow-outside-rays 只反演目标区域内路径。"
            )
    max_path_length_error = float(path_length_error.max()) if path_length_error.size else 0.0

    centerline_gmat = gmat
    centerline_ray_density = ray_density.copy()
    centerline_nnz = int(centerline_gmat.nnz)
    gmat = broaden_ray_matrix_gaussian(
        centerline_gmat,
        nx,
        ny,
        nz,
        sigma_xy_cells=args.ray_kernel_sigma_xy_cells,
        sigma_z_cells=args.ray_kernel_sigma_z_cells,
        relative_cutoff=args.ray_kernel_relative_cutoff,
    )
    kernel_support_density = np.asarray((gmat > 0).sum(axis=0)).ravel()
    kernel_row_sum_error = np.abs(
        np.asarray(gmat.sum(axis=1)).ravel()
        - np.asarray(centerline_gmat.sum(axis=1)).ravel()
    )
    max_kernel_row_sum_error = (
        float(np.max(kernel_row_sum_error)) if kernel_row_sum_error.size else 0.0
    )
    if max_kernel_row_sum_error > 1e-8:
        raise RuntimeError(
            "高斯射线管路径长度守恒检查失败："
            f"最大误差 {max_kernel_row_sum_error:.12f} m。"
        )
    # Coverage/reliability remains a centerline acquisition diagnostic.  The
    # broadened support is saved separately and must not be described as new
    # observed ray coverage.
    ray_density = centerline_ray_density
    kernel_nnz_ratio = float(gmat.nnz / max(centerline_nnz, 1))

    lmat = build_regularization(nx, ny, nz, grid_dx, grid_dy, grid_dz)
    cmat = build_curvature_regularization(
        nx,
        ny,
        nz,
        grid_dx,
        grid_dy,
        grid_dz,
        z_factor=args.curvature_z_factor,
    )
    spline_design = None
    spline_inverse = None
    if args.spline_projection:
        spline_design, spline_inverse = build_bspline_projection(
            nx,
            ny,
            args.spline_control_nx,
            args.spline_control_ny,
        )
    slowness = np.ones(nx * ny * nz, dtype=np.float64) / v0
    initial_model_source = "uniform_background"
    if args.initial_model is not None:
        if not args.initial_model.is_file():
            raise FileNotFoundError(f"初始模型不存在: {args.initial_model}")
        with np.load(args.initial_model, allow_pickle=False) as initial_model:
            required = {"velocity", "xc", "yc", "zc"}
            missing = required.difference(initial_model.files)
            if missing:
                raise KeyError(
                    f"初始模型缺少字段 {sorted(missing)}: {args.initial_model}"
                )
            initial_velocity = np.asarray(initial_model["velocity"], dtype=np.float64)
            initial_xc = np.asarray(initial_model["xc"], dtype=np.float64)
            initial_yc = np.asarray(initial_model["yc"], dtype=np.float64)
            initial_zc = np.asarray(initial_model["zc"], dtype=np.float64)
        expected_shape = (initial_xc.size, initial_yc.size, initial_zc.size)
        if initial_velocity.shape != expected_shape:
            raise ValueError(
                f"初始模型velocity形状为{initial_velocity.shape}，应为{expected_shape}"
            )
        initial_background = float(np.nanmedian(initial_velocity))
        initial_interp = RegularGridInterpolator(
            (initial_xc, initial_yc, initial_zc),
            initial_velocity,
            bounds_error=False,
            fill_value=initial_background,
        )
        gx, gy, gz = np.meshgrid(xc, yc, zc, indexing="ij")
        query_points = np.column_stack([
            gx.ravel(order="F"), gy.ravel(order="F"), gz.ravel(order="F")
        ])
        resampled_velocity = np.asarray(initial_interp(query_points), dtype=np.float64)
        resampled_velocity = np.where(
            np.isfinite(resampled_velocity), resampled_velocity, v0
        )
        resampled_velocity = np.clip(
            resampled_velocity, args.vmin_model, args.vmax_model
        )
        slowness = 1.0 / np.maximum(resampled_velocity, 100.0)
        initial_model_source = str(args.initial_model.resolve())
        print(
            "multiscale initial model: "
            f"{args.initial_model} -> {nx}x{ny}x{nz} cells"
        )

    if not 0.0 <= args.validation_fraction < 0.5:
        raise ValueError("validation-fraction 必须在 [0, 0.5) 范围内。")
    if not 1 <= args.wavelet_levels <= 6:
        raise ValueError("wavelet-levels 必须在 [1, 6] 范围内。")
    if not np.isfinite(args.wavelet_threshold_factor) or args.wavelet_threshold_factor < 0.0:
        raise ValueError("wavelet-threshold-factor 必须为非负有限数。")
    if not np.isfinite(args.differential_weight) or args.differential_weight < 0.0:
        raise ValueError("differential-weight 必须为非负有限数。")
    if not np.isfinite(args.model_damping) or args.model_damping < 0.0:
        raise ValueError("model-damping 必须为非负有限数。")
    if (
        not np.isfinite(args.coverage_weight_exponent)
        or args.coverage_weight_exponent <= 0.0
    ):
        raise ValueError("coverage-weight-exponent must be finite and positive")
    if not np.isfinite(args.curvature_reg_factor) or args.curvature_reg_factor < 0.0:
        raise ValueError("curvature-reg-factor 必须为非负有限数。")
    if not np.isfinite(args.curvature_z_factor) or args.curvature_z_factor < 0.0:
        raise ValueError("curvature-z-factor 必须为非负有限数。")
    if args.spline_projection and (
        args.spline_control_nx < 4 or args.spline_control_ny < 4
    ):
        raise ValueError("三次B样条每个方向至少需要4个控制点。")
    if (
        not np.isfinite(args.spline_projection_strength)
        or not 0.0 <= args.spline_projection_strength <= 1.0
    ):
        raise ValueError("spline-projection-strength 必须在 [0, 1] 范围内。")
    if args.hierarchical_parameterization:
        if (
            not np.isfinite(args.hierarchical_split_rays)
            or args.hierarchical_split_rays <= 0.0
        ):
            raise ValueError("hierarchical-split-rays 必须为正有限数。")
        if args.hierarchical_min_block_x < 1 or args.hierarchical_min_block_y < 1:
            raise ValueError("层次最小块尺寸必须为正整数。")
        if args.joint_sparsity or args.spline_projection:
            raise ValueError("层次参数化不能与Haar或B样条投影同时启用。")
        if args.initial_model is not None:
            raise ValueError("层次参数化暂不接受外部初始模型。")
    if (
        not np.isfinite(args.ray_kernel_sigma_xy_cells)
        or args.ray_kernel_sigma_xy_cells < 0.0
        or not np.isfinite(args.ray_kernel_sigma_z_cells)
        or args.ray_kernel_sigma_z_cells < 0.0
    ):
        raise ValueError("射线核标准差必须为非负有限数。")
    if (
        not np.isfinite(args.ray_kernel_relative_cutoff)
        or not 0.0 <= args.ray_kernel_relative_cutoff < 1.0
    ):
        raise ValueError("射线核相对截断阈值必须在 [0, 1) 范围内。")
    rng = np.random.default_rng(args.random_seed)
    split_sources = np.unique(source_ids)
    requested_validation_sources = np.asarray(
        [
            int(item.strip())
            for item in args.validation_source_ids.split(",")
            if item.strip()
        ],
        dtype=np.int64,
    )
    if requested_validation_sources.size:
        validation_sources = np.unique(requested_validation_sources)
        missing_validation_sources = np.setdiff1d(
            validation_sources, split_sources
        )
        if missing_validation_sources.size:
            missing_text = ",".join(str(int(item)) for item in missing_validation_sources)
            raise ValueError(f"固定验证震源不在有效数据中: {missing_text}")
        if validation_sources.size >= split_sources.size:
            raise ValueError("固定验证震源必须少于全部有效震源数。")
    elif args.validation_fraction > 0 and split_sources.size < 2:
        raise RuntimeError("按震源分组验证至少需要两个不同震源事件。")
    else:
        n_val_sources = int(round(split_sources.size * args.validation_fraction))
        n_val_sources = max(1, n_val_sources) if args.validation_fraction > 0 else 0
        n_val_sources = min(n_val_sources, max(split_sources.size - 1, 0))
        validation_sources = rng.permutation(split_sources)[:n_val_sources]
    validation_mask = np.isin(source_ids, validation_sources)
    val_idx = np.flatnonzero(validation_mask)
    train_idx = np.flatnonzero(~validation_mask)
    n_val = val_idx.size
    g_train = gmat[train_idx]
    d_train = tt_s[train_idx]
    g_val = gmat[val_idx] if n_val else None
    d_val = tt_s[val_idx] if n_val else None
    n_model = slowness.size
    hierarchical_training_ray_density = np.asarray(
        (centerline_gmat[train_idx] > 0).sum(axis=0)
    ).ravel()
    model_basis = None
    hierarchical_leaf_index = np.full((nx, ny, nz), -1, dtype=np.int64)
    hierarchical_leaf_table = np.empty((0, 8), dtype=np.float64)
    if args.hierarchical_parameterization:
        model_basis, hierarchical_leaf_index, hierarchical_leaf_table = (
            build_hierarchical_block_basis(
                hierarchical_training_ray_density,
                nx,
                ny,
                nz,
                args.hierarchical_split_rays,
                args.hierarchical_min_block_x,
                args.hierarchical_min_block_y,
                grid_dx,
                grid_dy,
            )
        )
        n_model_parameters = model_basis.shape[1]
    else:
        n_model_parameters = n_model
    # The linear system contains only the path inside the target volume.  In
    # ROI mode the source/receiver may be outside, so normalization must use
    # the modeled path length rather than the full source-receiver distance.
    train_ray_dist = traced_length[train_idx]
    val_ray_dist = traced_length[val_idx] if n_val else None
    reference_ray_length = float(np.median(train_ray_dist))
    if args.ray_length_normalization:
        train_ray_scale = reference_ray_length / np.maximum(train_ray_dist, 1.0)
        train_ray_scale = np.clip(train_ray_scale, 0.25, 4.0)
        val_ray_scale = (
            np.clip(reference_ray_length / np.maximum(val_ray_dist, 1.0), 0.25, 4.0)
            if n_val else None
        )
    else:
        train_ray_scale = np.ones_like(train_ray_dist)
        val_ray_scale = np.ones_like(val_ray_dist) if n_val else None

    unique_sources, source_inverse = np.unique(source_ids, return_inverse=True)
    n_sources = unique_sources.size
    receiver_coordinates, receiver_inverse = np.unique(
        np.column_stack([rx, ry, rz]),
        axis=0,
        return_inverse=True,
    )
    receiver_ids = receiver_inverse.astype(np.int64, copy=False)
    train_source_mask = np.isin(unique_sources, np.unique(source_ids[train_idx]))
    if args.invert_source_statics:
        source_design = csr_matrix(
            (np.ones(sx.size), (np.arange(sx.size), source_inverse)),
            shape=(sx.size, n_sources),
        )
        global_design = csr_matrix(np.ones((sx.size, 1), dtype=np.float64))
        time_design = hstack([global_design, source_design], format="csr")
    else:
        time_design = csr_matrix((sx.size, 0), dtype=np.float64)
    time_train = time_design[train_idx]
    time_val = time_design[val_idx] if n_val else None
    time_parameters = np.zeros(time_design.shape[1], dtype=np.float64)
    if args.differential_times and args.differential_weight > 0.0:
        differential_operator = build_common_source_difference(source_ids[train_idx])
        g_differential = differential_operator @ g_train
        d_differential = np.asarray(differential_operator @ d_train).ravel()
    else:
        differential_operator = csr_matrix((0, train_idx.size), dtype=np.float64)
        g_differential = csr_matrix((0, g_train.shape[1]), dtype=np.float64)
        d_differential = np.empty(0, dtype=np.float64)
    n_differential = d_differential.size
    if time_parameters.size and args.event_centered_qc:
        initial_source_shift = np.asarray(
            [
                float(np.median(qc_event_shift_s[source_ids == source_id]))
                for source_id in unique_sources
            ],
            dtype=np.float64,
        )
        initial_source_shift = np.clip(
            initial_source_shift, -args.max_time_correction, args.max_time_correction
        )
        initial_source_shift[~train_source_mask] = 0.0
        train_shift = initial_source_shift[train_source_mask]
        initial_global_shift = float(np.mean(train_shift)) if train_shift.size else 0.0
        time_parameters[0] = initial_global_shift
        time_parameters[1:] = initial_source_shift - initial_global_shift
        time_parameters[1:][~train_source_mask] = 0.0

    positive_coverage = ray_density[ray_density > 0]
    if args.min_ray_coverage > 0:
        reliable_coverage = float(args.min_ray_coverage)
    elif positive_coverage.size:
        reliable_coverage = float(max(5.0, np.percentile(positive_coverage, 25.0)))
    else:
        reliable_coverage = 5.0
    coverage_weight, background_weight = build_coverage_weights(
        ray_density,
        reliable_coverage,
        exponent=args.coverage_weight_exponent,
    )
    covered_fraction = float(np.count_nonzero(ray_density) / max(ray_density.size, 1))
    reliable_fraction = float(np.count_nonzero(ray_density >= reliable_coverage) / max(ray_density.size, 1))
    if covered_fraction < 0.05:
        print(
            "WARNING: 射线仅覆盖 "
            f"{covered_fraction * 100.0:.2f}% 的三维网格；未覆盖区域显示的是背景模型，"
            "不能解释为真实均匀速度场。请缩小目标区域、降低网格数量或增加有效射线。"
        )
    background_matrix = diags(background_weight, format="csr")

    zero_reg = np.zeros(lmat.shape[0], dtype=np.float64)
    zero_curvature = np.zeros(cmat.shape[0], dtype=np.float64)
    n_time = time_parameters.size
    zero_time_for_smoothing = csr_matrix((lmat.shape[0], n_time))
    zero_time_for_curvature = csr_matrix((cmat.shape[0], n_time))
    zero_time_for_background = csr_matrix((n_model, n_time))
    background_model_matrix = (
        background_matrix @ model_basis
        if model_basis is not None
        else background_matrix
    )
    background_aug = hstack([
        args.background_damping * background_model_matrix,
        zero_time_for_background,
    ], format="csr")
    model_damping_matrix = model_basis if model_basis is not None else diags(
        np.ones(n_model, dtype=np.float64), format="csr"
    )
    model_damping_aug = hstack([
        args.model_damping * model_damping_matrix,
        zero_time_for_background,
    ], format="csr")
    if n_time:
        time_rows = []
        time_cols = []
        time_vals = []
        time_rows.append(0)
        time_cols.append(0)
        time_vals.append(args.global_time_damping)
        for j in range(n_sources):
            time_rows.append(1 + j)
            time_cols.append(1 + j)
            time_vals.append(args.source_static_damping)
            time_rows.append(1 + n_sources)
            time_cols.append(1 + j)
            time_vals.append(20.0 / np.sqrt(max(n_sources, 1)))
        time_reg_local = csr_matrix(
            (time_vals, (time_rows, time_cols)),
            shape=(n_sources + 2, n_time),
        )
        time_reg_aug = hstack([
            csr_matrix((n_sources + 2, n_model_parameters)),
            time_reg_local,
        ], format="csr")
    else:
        time_reg_aug = csr_matrix((0, n_model_parameters), dtype=np.float64)

    def predict(g_part: csr_matrix, time_part: csr_matrix, model: np.ndarray, timing: np.ndarray) -> np.ndarray:
        result = np.asarray(g_part @ model).ravel()
        if timing.size:
            result = result + np.asarray(time_part @ timing).ravel()
        return result

    def grouped_validation_rms(model: np.ndarray, timing: np.ndarray) -> tuple[float, int]:
        """Validate spatial travel-time structure on completely held-out events."""
        if not n_val:
            return float("nan"), 0
        prediction = np.asarray(g_val @ model).ravel()
        if timing.size:
            prediction += float(timing[0])
        residual = d_val - prediction
        val_source_ids = source_ids[val_idx]
        centered_parts = []
        for source_id in np.unique(val_source_ids):
            part = residual[val_source_ids == source_id]
            if part.size < 2:
                continue
            centered_parts.append(part - np.median(part))
        if not centered_parts:
            return float(np.sqrt(np.mean(residual**2))), int(residual.size)
        centered = np.concatenate(centered_parts)
        return float(np.sqrt(np.mean(centered**2))), int(centered.size)

    rms_hist: List[Tuple[int, float, float, int, int]] = []
    validation_hist: List[Tuple[int, float]] = []
    best_slowness = slowness.copy()
    best_time_parameters = time_parameters.copy()
    best_rms = float(np.sqrt(np.mean((d_train - predict(g_train, time_train, slowness, time_parameters)) ** 2)))
    initial_validation_rms, validation_metric_rays = grouped_validation_rms(
        slowness, time_parameters
    )
    best_validation_rms = initial_validation_rms if n_val else best_rms
    no_improve = 0
    tv_epsilon = 0.0
    wavelet_threshold = 0.0
    wavelet_retained_fraction = 1.0
    differential_downweighted = 0
    if model_basis is not None:
        inversion_method = (
            "hierarchical_tv_sirt"
            if args.solver_method == "sirt"
            else "hierarchical_tv_lsqr"
        )
    else:
        inversion_method = "tv_sirt" if args.solver_method == "sirt" else "tv_lsqr"
    deep_selected_start = -1
    deep_parameter_count = 0
    deep_velocity_uncertainty = np.zeros(n_model, dtype=np.float64)
    deep_ensemble_velocity = np.empty((0, n_model), dtype=np.float64)
    deep_start_train_rms_ms = np.empty(0, dtype=np.float64)
    deep_coverage_gate_min = float("nan")
    deep_coverage_gate_mean = float("nan")
    deep_coverage_gate_max = float("nan")
    training_reliable_coverage = float("nan")
    sirt_tuned_alpha = float("nan")
    sirt_tuned_omega = float("nan")
    sirt_tuned_metric = float("nan")
    sirt_tuning_evaluations = 0
    sirt_tuning_status = "disabled" if args.solver_method != "sirt" else "not_run"

    if args.deep_reparameterization:
        from wave_ct.deep_reparam import (
            event_centered_rms,
            normalized_cell_coordinates,
            profile_event_time_corrections,
            training_coverage_gate,
        )

        inversion_method = "deep_coordinate_mlp_reparameterization"
        coordinates = normalized_cell_coordinates(xc, yc, zc)
        training_positive_coverage = hierarchical_training_ray_density[
            hierarchical_training_ray_density > 0
        ]
        if args.min_ray_coverage > 0:
            training_reliable_coverage = float(args.min_ray_coverage)
        elif training_positive_coverage.size:
            training_reliable_coverage = float(
                max(5.0, np.percentile(training_positive_coverage, 25.0))
            )
        else:
            training_reliable_coverage = 5.0
        deep_coverage_gate, _ = training_coverage_gate(
            g_train,
            nx,
            ny,
            nz,
            reliable_coverage=training_reliable_coverage,
            coverage_exponent=0.5,
        )
        deep_coverage_gate_min = float(np.min(deep_coverage_gate))
        deep_coverage_gate_mean = float(np.mean(deep_coverage_gate))
        deep_coverage_gate_max = float(np.max(deep_coverage_gate))
        payload = {
            "matrix_data": g_train.data,
            "matrix_indices": g_train.indices,
            "matrix_indptr": g_train.indptr,
            "matrix_shape": np.asarray(g_train.shape, dtype=np.int64),
            "observed_s": d_train,
            "source_ids": source_ids[train_idx],
            "coordinates": coordinates,
            "background_velocity": np.asarray(v0),
            "velocity_min": np.asarray(args.vmin_model),
            "velocity_max": np.asarray(args.vmax_model),
            "coverage_gate": deep_coverage_gate,
        }
        if args.deep_reparam_receiver_statics:
            payload["receiver_ids"] = receiver_ids[train_idx]
        if n_val:
            payload.update(
                {
                    "validation_matrix_data": g_val.data,
                    "validation_matrix_indices": g_val.indices,
                    "validation_matrix_indptr": g_val.indptr,
                    "validation_matrix_shape": np.asarray(
                        g_val.shape, dtype=np.int64
                    ),
                    "validation_observed_s": d_val,
                    "validation_source_ids": source_ids[val_idx],
                }
            )
            if args.deep_reparam_receiver_statics:
                payload["validation_receiver_ids"] = receiver_ids[val_idx]

        start_count = 1 if n_val else args.deep_reparam_starts
        deep_results = []
        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix="wavect_dnr_") as temporary:
            temporary_dir = Path(temporary)
            problem_path = temporary_dir / "problem.npz"
            np.savez_compressed(problem_path, **payload)
            for start_index in range(start_count):
                result_path = temporary_dir / f"result_{start_index}.npz"
                random_seed = (
                    20260724 + args.random_seed
                    if n_val
                    else 20260724 + start_index
                )
                command = [
                    sys.executable,
                    "-m",
                    "wave_ct.tools.deep_reparam_worker",
                    "--problem-npz",
                    str(problem_path),
                    "--output-npz",
                    str(result_path),
                    "--fixed-epochs",
                    str(0 if n_val else args.deep_reparam_full_epochs),
                    "--max-epochs",
                    str(args.deep_reparam_max_epochs),
                    "--random-seed",
                    str(random_seed),
                    "--network-width",
                    str(args.deep_reparam_width),
                    "--network-depth",
                    str(args.deep_reparam_depth),
                    "--learning-rate",
                    str(args.deep_reparam_learning_rate),
                    "--huber-beta-ms",
                    str(args.deep_reparam_huber_ms),
                    "--event-static-shrinkage",
                    str(args.deep_reparam_event_static_shrinkage),
                    "--receiver-static-shrinkage",
                    str(args.deep_reparam_receiver_static_shrinkage),
                    "--receiver-static-max-ms",
                    str(args.deep_reparam_receiver_static_max_ms),
                    "--static-profile-iterations",
                    str(args.deep_reparam_static_profile_iterations),
                    "--output-l2",
                    str(args.deep_reparam_output_l2),
                    "--fourier-bands",
                    str(args.deep_reparam_fourier_bands),
                    "--differential-loss-fraction",
                    str(args.deep_reparam_differential_loss_fraction),
                    "--total-variation",
                    str(args.deep_reparam_tv),
                    "--total-variation-epsilon",
                    str(args.deep_reparam_tv_epsilon),
                    "--device",
                    args.deep_reparam_device,
                ]
                completed = subprocess.run(
                    command,
                    cwd=project_root,
                    check=False,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                if completed.returncode != 0 or not result_path.is_file():
                    raise RuntimeError(
                        "deep reparameterization worker failed: "
                        + completed.stderr.strip()
                    )
                with np.load(result_path, allow_pickle=False) as result:
                    deep_results.append(
                        {
                            key: np.asarray(result[key]).copy()
                            for key in result.files
                        }
                    )

        deep_start_train_rms_ms = np.asarray(
            [float(result["train_rms_ms"]) for result in deep_results],
            dtype=np.float64,
        )
        deep_selected_start = int(np.argmin(deep_start_train_rms_ms))
        selected_deep = deep_results[deep_selected_start]
        slowness = np.asarray(selected_deep["slowness"], dtype=np.float64)
        best_slowness = slowness.copy()
        deep_ensemble_velocity = np.stack(
            [
                1.0 / np.asarray(result["slowness"], dtype=np.float64)
                for result in deep_results
            ],
            axis=0,
        )
        deep_velocity_uncertainty = np.std(
            deep_ensemble_velocity, axis=0
        )
        deep_parameter_count = int(selected_deep["parameter_count"])
        initial_model_source = "untrained_coordinate_mlp"

        prediction_train = np.asarray(g_train @ slowness).ravel()
        (
            profiled_global,
            profiled_sources,
            profiled_deviations,
        ) = profile_event_time_corrections(
            d_train,
            prediction_train,
            source_ids[train_idx],
            shrinkage=args.deep_reparam_event_static_shrinkage,
            maximum_correction_s=args.max_time_correction,
        )
        if time_parameters.size:
            time_parameters[:] = 0.0
            time_parameters[0] = profiled_global
            source_column = {
                int(source_id): 1 + index
                for index, source_id in enumerate(unique_sources)
            }
            for source_id, deviation in zip(
                profiled_sources, profiled_deviations
            ):
                time_parameters[source_column[int(source_id)]] = deviation
        best_time_parameters = time_parameters.copy()
        best_rms = float(
            np.sqrt(
                np.mean(
                    (
                        d_train
                        - predict(
                            g_train,
                            time_train,
                            slowness,
                            time_parameters,
                        )
                    )
                    ** 2
                )
            )
        )
        if n_val:
            best_validation_rms = (
                float(selected_deep["validation_rms_ms"]) / 1000.0
            )
            validation_metric_rays = int(
                selected_deep["validation_metric_rays"]
            )
        else:
            best_validation_rms = best_rms
            _, validation_metric_rays = event_centered_rms(
                d_train,
                prediction_train,
                source_ids[train_idx],
            )
        history = np.asarray(selected_deep["history"], dtype=np.float64)
        initial_deep_rms = float(
            np.sqrt(
                np.mean(
                    (
                        d_train
                        - np.asarray(
                            g_train
                            @ np.full(n_model, 1.0 / v0, dtype=np.float64)
                        ).ravel()
                    )
                    ** 2
                )
            )
        )
        previous_rms = initial_deep_rms
        for epoch, _, train_rms_ms, validation_rms_ms in history:
            current_rms = float(train_rms_ms) / 1000.0
            rms_hist.append(
                (int(epoch), previous_rms, current_rms, 0, int(epoch))
            )
            previous_rms = current_rms
            if np.isfinite(validation_rms_ms):
                validation_hist.append(
                    (int(epoch), float(validation_rms_ms) / 1000.0)
                )
        print(
            "deep reparameterization: "
            f"selected start {deep_selected_start + 1}/{start_count}, "
            f"train event-centered RMS="
            f"{deep_start_train_rms_ms[deep_selected_start]:.4f} ms"
        )

    iteration_range = (
        range(1, args.n_outer + 1)
        if not args.deep_reparameterization
        else ()
    )
    for it in iteration_range:
        d_calc = predict(g_train, time_train, slowness, time_parameters)
        residual = d_train - d_calc
        rms0 = float(np.sqrt(np.mean(residual**2)))

        normalized_residual = residual * train_ray_scale
        residual_median = float(np.median(normalized_residual))
        residual_mad = float(np.median(np.abs(normalized_residual - residual_median)))
        robust_sigma = max(1.4826 * residual_mad, 1e-6)
        huber_limit = max(args.huber_delta, 0.5) * robust_sigma
        robust_weight = np.ones_like(residual)
        large = np.abs(normalized_residual) > huber_limit
        robust_weight[large] = huber_limit / np.abs(normalized_residual[large])
        data_row_scale = np.sqrt(robust_weight) * train_ray_scale
        weighted_g = g_train.multiply(data_row_scale[:, None])
        weighted_parameter_g = (
            weighted_g @ model_basis if model_basis is not None else weighted_g
        )
        weighted_time = time_train.multiply(data_row_scale[:, None])
        weighted_system = hstack(
            [weighted_parameter_g, weighted_time], format="csr"
        )
        weighted_residual = residual * data_row_scale

        data_system_parts = [weighted_system]
        data_rhs_parts = [weighted_residual]
        differential_rms0 = float("nan")
        combined_rms0 = rms0
        if n_differential:
            differential_residual = d_differential - np.asarray(
                g_differential @ slowness
            ).ravel()
            differential_rms0 = float(np.sqrt(np.mean(differential_residual**2)))
            differential_median = float(np.median(differential_residual))
            differential_mad = float(
                np.median(np.abs(differential_residual - differential_median))
            )
            differential_sigma = max(1.4826 * differential_mad, 1e-6)
            differential_limit = max(args.huber_delta, 0.5) * differential_sigma
            differential_robust_weight = np.ones_like(differential_residual)
            differential_large = np.abs(differential_residual) > differential_limit
            differential_robust_weight[differential_large] = (
                differential_limit / np.abs(differential_residual[differential_large])
            )
            differential_row_scale = (
                args.differential_weight * np.sqrt(differential_robust_weight)
            )
            weighted_differential_g = g_differential.multiply(
                differential_row_scale[:, None]
            )
            weighted_differential_parameters = (
                weighted_differential_g @ model_basis
                if model_basis is not None
                else weighted_differential_g
            )
            weighted_differential_system = hstack(
                [
                    weighted_differential_parameters,
                    csr_matrix((n_differential, n_time)),
                ],
                format="csr",
            )
            data_system_parts.append(weighted_differential_system)
            data_rhs_parts.append(differential_residual * differential_row_scale)
            differential_downweighted = int(np.count_nonzero(differential_large))
            combined_rms0 = float(np.sqrt(
                (
                    np.sum(residual**2)
                    + args.differential_weight**2 * np.sum(differential_residual**2)
                )
                / (
                    residual.size
                    + args.differential_weight**2 * differential_residual.size
                )
            ))
        data_system = vstack(data_system_parts, format="csr")
        data_rhs = np.concatenate(data_rhs_parts)

        if args.edge_preserving_tv:
            slowness_gradient = np.asarray(lmat @ slowness).ravel()
            abs_gradient = np.abs(slowness_gradient)
            positive_gradient = abs_gradient[abs_gradient > 1e-12]
            if args.tv_epsilon > 0:
                tv_epsilon = args.tv_epsilon
            elif positive_gradient.size:
                tv_epsilon = max(float(np.median(positive_gradient)) * 0.5, 5e-7)
            else:
                tv_epsilon = max(0.002 / v0, 5e-7)
            tv_weight = 1.0 / np.sqrt(slowness_gradient**2 + tv_epsilon**2)
            tv_weight /= max(float(np.median(tv_weight)), 1e-12)
            tv_weight = np.clip(tv_weight, 0.15, 6.0)
            tv_row_scale = np.sqrt(tv_weight)
            weighted_lmat = lmat.multiply(tv_row_scale[:, None])
        else:
            tv_epsilon = 0.0
            weighted_lmat = lmat
        weighted_parameter_lmat = (
            weighted_lmat @ model_basis
            if model_basis is not None
            else weighted_lmat
        )

        parameter_cmat = cmat @ model_basis if model_basis is not None else cmat

        def assemble_augmented_system(alpha_value: float):
            smooth_aug = hstack(
                [alpha_value * weighted_parameter_lmat, zero_time_for_smoothing],
                format="csr",
            )
            if args.regularize_total_model:
                smooth_rhs = -alpha_value * np.asarray(
                    weighted_lmat @ slowness
                ).ravel()
            else:
                smooth_rhs = zero_reg

            regularization_systems = [smooth_aug]
            regularization_rhs = [smooth_rhs]
            if args.model_damping > 0.0:
                regularization_systems.append(model_damping_aug)
                regularization_rhs.append(
                    args.model_damping * ((1.0 / v0) - slowness)
                )
            if args.curvature_reg_factor > 0.0 and cmat.shape[0] > 0:
                curvature_scale = alpha_value * args.curvature_reg_factor
                curvature_aug = hstack(
                    [curvature_scale * parameter_cmat, zero_time_for_curvature],
                    format="csr",
                )
                if args.regularize_total_model:
                    curvature_rhs = -curvature_scale * np.asarray(
                        cmat @ slowness
                    ).ravel()
                else:
                    curvature_rhs = zero_curvature
                regularization_systems.append(curvature_aug)
                regularization_rhs.append(curvature_rhs)

            background_rhs = background_weight * ((1.0 / v0) - slowness)
            time_reg_rhs = -(
                time_reg_aug
                @ np.concatenate([
                    np.zeros(n_model_parameters, dtype=np.float64),
                    time_parameters,
                ])
            )
            amat = vstack(
                [data_system, *regularization_systems, background_aug, time_reg_aug],
                format="csr",
            )
            bvec = np.concatenate([
                data_rhs,
                *regularization_rhs,
                args.background_damping * background_rhs,
                np.asarray(time_reg_rhs).ravel(),
            ])
            return amat, bvec

        amat, bvec = assemble_augmented_system(args.alpha_reg)

        def apply_timing_step(step_dt: np.ndarray) -> np.ndarray:
            candidate_time = time_parameters + step_dt
            if not candidate_time.size:
                return candidate_time
            candidate_time = np.clip(
                candidate_time, -args.max_time_correction, args.max_time_correction
            )
            source_statics = candidate_time[1:]
            source_statics[~train_source_mask] = 0.0
            if np.any(train_source_mask):
                mean_source_static = float(np.mean(source_statics[train_source_mask]))
                candidate_time[0] += mean_source_static
                source_statics[train_source_mask] -= mean_source_static
            return candidate_time

        if (
            args.solver_method == "sirt"
            and args.sirt_auto_tune
            and it == 1
        ):
            original_alpha = float(args.alpha_reg)
            original_omega = float(args.sirt_omega)
            def tuning_objective(parameters: np.ndarray) -> float:
                alpha_value, omega_value = (float(value) for value in parameters)
                pilot_amat, pilot_bvec = assemble_augmented_system(alpha_value)
                pilot = solve_sirt(
                    pilot_amat,
                    pilot_bvec,
                    iterations=args.sirt_tune_iterations or args.sirt_iterations or args.n_lsqr,
                    relaxation=omega_value,
                    tolerance=args.sirt_tolerance,
                )
                pilot_parameter_step = pilot.x[:n_model_parameters]
                pilot_ds = (
                    np.asarray(model_basis @ pilot_parameter_step).ravel()
                    if model_basis is not None
                    else pilot_parameter_step
                )
                pilot_model = np.clip(
                    slowness + pilot_ds,
                    1.0 / args.vmax_model,
                    1.0 / args.vmin_model,
                )
                pilot_time = apply_timing_step(pilot.x[n_model_parameters:])
                if n_val:
                    metric, _ = grouped_validation_rms(pilot_model, pilot_time)
                else:
                    metric = float(
                        np.sqrt(
                            np.mean(
                                (
                                    d_train
                                    - predict(
                                        g_train,
                                        time_train,
                                        pilot_model,
                                        pilot_time,
                                    )
                                )
                                ** 2
                            )
                        )
                    )
                if not np.isfinite(metric):
                    return 1.0e6
                return float(metric)

            # Compare the DE candidate with the exact baseline configuration
            # under the same pilot solve.  Comparing against the unupdated
            # background alone would let a nominally "tuned" setting replace
            # a better fixed setting simply because the pilot is nonlinear.
            baseline_tuning_metric = tuning_objective(
                np.asarray([original_alpha, original_omega], dtype=np.float64)
            )

            tune_result = differential_evolution(
                tuning_objective,
                bounds=((0.1, 30.0), (0.01, 0.7)),
                strategy="best1bin",
                maxiter=args.sirt_tune_maxiter,
                popsize=args.sirt_tune_popsize,
                mutation=(0.5, 1.0),
                recombination=0.7,
                seed=args.random_seed,
                polish=True,
                updating="immediate",
                workers=1,
            )
            sirt_tuned_metric = float(tune_result.fun)
            sirt_tuning_evaluations = int(tune_result.nfev)
            if tune_result.fun + args.early_stop_tol < baseline_tuning_metric:
                args.alpha_reg = float(tune_result.x[0])
                args.sirt_omega = float(tune_result.x[1])
                sirt_tuning_status = "accepted"
            else:
                args.alpha_reg = original_alpha
                args.sirt_omega = original_omega
                sirt_tuning_status = "rejected_no_validation_gain"
            sirt_tuned_alpha = float(args.alpha_reg)
            sirt_tuned_omega = float(args.sirt_omega)
            print(
                "SIRT automatic tuning: "
                f"trial_alpha={float(tune_result.x[0]):.6g}, "
                f"trial_omega={float(tune_result.x[1]):.6g}, "
                f"trial_objective={sirt_tuned_metric:.6f}, "
                f"baseline={baseline_tuning_metric:.6f}, "
                f"selected_alpha={args.alpha_reg:.6g}, "
                f"selected_omega={args.sirt_omega:.6g}, "
                f"status={sirt_tuning_status}, evaluations={sirt_tuning_evaluations}"
            )
            amat, bvec = assemble_augmented_system(args.alpha_reg)
        if args.solver_method == "sirt":
            sirt_result = solve_sirt(
                amat,
                bvec,
                iterations=args.sirt_iterations or args.n_lsqr,
                relaxation=args.sirt_omega,
                tolerance=args.sirt_tolerance,
            )
            solution_vector = sirt_result.x
            flag = sirt_result.flag
            niter_done = sirt_result.iterations
        else:
            out = lsqr(amat, bvec, atol=1e-8, btol=1e-8, iter_lim=args.n_lsqr)
            solution_vector = out[0]
            flag = int(out[1])
            niter_done = int(out[2])
        parameter_step = solution_vector[:n_model_parameters]
        ds = (
            np.asarray(model_basis @ parameter_step).ravel()
            if model_basis is not None
            else parameter_step
        )
        dt = solution_vector[n_model_parameters:]

        accepted = False
        trial_step = (
            args.sirt_step_damp
            if args.solver_method == "sirt"
            else args.step_damp
        )
        trial_slowness = slowness
        trial_time_parameters = time_parameters
        rms1 = rms0
        for _ in range(10):
            candidate = np.clip(
                slowness + trial_step * ds,
                1.0 / args.vmax_model,
                1.0 / args.vmin_model,
            )
            candidate_wavelet_threshold = 0.0
            candidate_wavelet_retained = 1.0
            if args.joint_sparsity:
                candidate, candidate_wavelet_threshold, candidate_wavelet_retained = (
                    apply_joint_sparsity_wavelet(
                        candidate,
                        1.0 / v0,
                        nx,
                        ny,
                        nz,
                        args.wavelet_levels,
                        args.wavelet_threshold_factor,
                    )
                )
                candidate = np.clip(
                    candidate,
                    1.0 / args.vmax_model,
                    1.0 / args.vmin_model,
                )
            if args.spline_projection:
                if spline_design is None or spline_inverse is None:
                    raise RuntimeError("B样条投影矩阵尚未初始化。")
                spline_candidate = apply_bspline_projection(
                    candidate,
                    1.0 / v0,
                    nx,
                    ny,
                    nz,
                    spline_design,
                    spline_inverse,
                )
                candidate = candidate + args.spline_projection_strength * (
                    spline_candidate - candidate
                )
                candidate = np.clip(
                    candidate,
                    1.0 / args.vmax_model,
                    1.0 / args.vmin_model,
                )
            candidate_time = time_parameters + trial_step * dt
            if candidate_time.size:
                candidate_time = np.clip(
                    candidate_time, -args.max_time_correction, args.max_time_correction
                )
                source_statics = candidate_time[1:]
                source_statics[~train_source_mask] = 0.0
                mean_source_static = float(np.mean(source_statics[train_source_mask]))
                candidate_time[0] += mean_source_static
                source_statics[train_source_mask] -= mean_source_static
            candidate_rms = float(np.sqrt(np.mean(
                (d_train - predict(g_train, time_train, candidate, candidate_time)) ** 2
            )))
            candidate_combined_rms = candidate_rms
            if n_differential:
                candidate_differential_residual = d_differential - np.asarray(
                    g_differential @ candidate
                ).ravel()
                candidate_combined_rms = float(np.sqrt(
                    (
                        candidate_rms**2 * residual.size
                        + args.differential_weight**2
                        * np.sum(candidate_differential_residual**2)
                    )
                    / (
                        residual.size
                        + args.differential_weight**2 * candidate_differential_residual.size
                    )
                ))
            if (
                candidate_combined_rms + args.early_stop_tol < combined_rms0
                and candidate_rms <= rms0 + args.early_stop_tol
            ):
                trial_slowness = candidate
                trial_time_parameters = candidate_time
                rms1 = candidate_rms
                wavelet_threshold = candidate_wavelet_threshold
                wavelet_retained_fraction = candidate_wavelet_retained
                accepted = True
                break
            trial_step *= 0.5

        if accepted:
            slowness = trial_slowness
            time_parameters = trial_time_parameters
        rms_hist.append((it, rms0, rms1, flag, niter_done))
        validation_rms, validation_metric_rays = grouped_validation_rms(
            slowness, time_parameters
        ) if n_val else (rms1, train_idx.size)
        validation_hist.append((it, validation_rms))
        differential_log = (
            f", diff={differential_rms0:.6f}, "
            f"diff_downweighted={differential_downweighted}"
            if n_differential else ""
        )
        print(
            f"iter {it:02d}: RMS {rms0:.6f} -> {rms1:.6f}, "
            f"val={validation_rms:.6f}, downweighted={int(np.count_nonzero(large))}"
            f"{differential_log}, step={trial_step:.6g}, accepted={accepted}, "
            f"{args.solver_method}_flag={flag}, "
            f"{args.solver_method}_iter={niter_done}"
        )

        if validation_rms + args.early_stop_tol < best_validation_rms:
            best_rms = rms1
            best_validation_rms = validation_rms
            best_slowness = slowness.copy()
            best_time_parameters = time_parameters.copy()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.early_stop_patience:
                print(
                    f"early stop at iter {it:02d}; best train RMS = {best_rms:.6f}, "
                    f"best validation RMS = {best_validation_rms:.6f}"
                )
                break

    slowness = best_slowness
    time_parameters = best_time_parameters
    # ``velocity`` is the actual selected inverse solution and is therefore
    # the only field that may be used for forward prediction or quantitative
    # comparison.  A separate coverage-stabilized field is retained purely
    # for visualisation; the previous implementation silently published that
    # display field as the model, invalidating the reported validation RMS.
    raw_velocity = 1.0 / slowness
    velocity = raw_velocity.copy()
    display_velocity = v0 + coverage_weight * (raw_velocity - v0)
    vel3d = velocity.reshape((nx, ny, nz), order="F")
    display_vel3d = display_velocity.reshape((nx, ny, nz), order="F")
    vel_show = np.clip(display_vel3d, args.vmin_model, args.vmax_model)
    ray_density3d = ray_density.reshape((nx, ny, nz), order="F")

    solution_train_rms = float(np.sqrt(np.mean(
        (d_train - predict(g_train, time_train, slowness, time_parameters)) ** 2
    )))
    solution_validation_rms, _ = grouped_validation_rms(slowness, time_parameters)
    display_slowness = 1.0 / np.maximum(display_velocity, 100.0)
    display_train_rms = float(np.sqrt(np.mean(
        (d_train - predict(g_train, time_train, display_slowness, time_parameters)) ** 2
    )))
    display_validation_rms, _ = grouped_validation_rms(display_slowness, time_parameters)
    solution_differential_rms = (
        float(np.sqrt(np.mean(
            (d_differential - np.asarray(g_differential @ slowness).ravel()) ** 2
        )))
        if n_differential else float("nan")
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    global_time_correction = float(time_parameters[0]) if time_parameters.size else 0.0
    source_time_deviations = time_parameters[1:] if time_parameters.size else np.zeros(n_sources)
    with (args.output_dir / "source_time_corrections.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "震源编号", "全局时间修正_ms", "震源相对修正_ms", "总时间修正_ms"
        ])
        for source_id, deviation in zip(unique_sources, source_time_deviations):
            writer.writerow([
                int(source_id),
                f"{global_time_correction * 1000.0:.6f}",
                f"{float(deviation) * 1000.0:.6f}",
                f"{(global_time_correction + float(deviation)) * 1000.0:.6f}",
            ])

    targets = parse_slice_targets(args.slice_z, args.z_min, args.z_max)
    if not args.slice_z.strip():
        # A real 3-D slice product should expose every solved cell-centre layer.
        # The former five-percentile default hid valid intermediate elevations.
        targets = [float(value) for value in zc if np.isfinite(value)]
    validate_slice_targets(targets, zc, grid_dz)
    interp_factor = 4
    xf = np.linspace(xc[0], xc[-1], nx * interp_factor)
    yf = np.linspace(yc[0], yc[-1], ny * interp_factor)
    xfine, yfine = np.meshgrid(xf, yf)
    finite_v_app_stats = v_app[np.isfinite(v_app)]
    if finite_v_app_stats.size:
        apparent_velocity_range_text = (
            f"{finite_v_app_stats.min():.2f}-{finite_v_app_stats.max():.2f}"
        )
        apparent_velocity_mean_text = f"{finite_v_app_stats.mean():.2f}"
    else:
        apparent_velocity_range_text = "n/a"
        apparent_velocity_mean_text = "n/a"

    if hierarchical_leaf_table.size:
        hierarchical_leaf_sizes = (
            (hierarchical_leaf_table[:, 2] - hierarchical_leaf_table[:, 1])
            * (hierarchical_leaf_table[:, 4] - hierarchical_leaf_table[:, 3])
        )
    else:
        hierarchical_leaf_sizes = np.ones(n_model, dtype=np.float64)

    stats_lines = [
        "反演切片统计",
        "=" * 60,
        f"mode={args.mode}",
        f"plot_style={args.plot_style}",
        f"input_csv={args.input_csv}",
        f"inversion_method={inversion_method}",
        f"solver_method={args.solver_method}",
        f"sirt_iterations={args.sirt_iterations or args.n_lsqr}",
        f"sirt_omega={args.sirt_omega:.10g}",
        f"sirt_step_damp={args.sirt_step_damp:.10g}",
        f"sirt_tolerance={args.sirt_tolerance:.10g}",
        f"sirt_auto_tune={'enabled' if args.sirt_auto_tune else 'disabled'}",
        f"sirt_tune_maxiter={args.sirt_tune_maxiter}",
        f"sirt_tune_popsize={args.sirt_tune_popsize}",
        f"sirt_tune_iterations={args.sirt_tune_iterations or args.sirt_iterations or args.n_lsqr}",
        f"sirt_tuned_alpha={sirt_tuned_alpha:.10g}",
        f"sirt_tuned_omega={sirt_tuned_omega:.10g}",
        f"sirt_tuned_metric={sirt_tuned_metric:.10g}",
        f"sirt_tuning_evaluations={sirt_tuning_evaluations}",
        f"sirt_tuning_status={sirt_tuning_status}",
        f"deep_reparameterization="
        f"{'enabled' if args.deep_reparameterization else 'disabled'}",
        f"deep_reparam_width={args.deep_reparam_width}",
        f"deep_reparam_depth={args.deep_reparam_depth}",
        f"deep_reparam_full_epochs={args.deep_reparam_full_epochs}",
        f"deep_reparam_max_epochs={args.deep_reparam_max_epochs}",
        f"deep_reparam_starts={args.deep_reparam_starts}",
        f"deep_reparam_output_l2={args.deep_reparam_output_l2:.10g}",
        f"deep_reparam_fourier_bands={args.deep_reparam_fourier_bands}",
        "deep_reparam_differential_loss_fraction="
        f"{args.deep_reparam_differential_loss_fraction:.10g}",
        f"deep_reparam_tv={args.deep_reparam_tv:.10g}",
        f"deep_reparam_tv_epsilon={args.deep_reparam_tv_epsilon:.10g}",
        "deep_reparam_coverage_gate=training_geometry_only",
        f"deep_reparam_coverage_exponent={0.5:.6f}",
        f"deep_reparam_training_reliable_coverage={training_reliable_coverage:.6f}",
        f"deep_reparam_coverage_gate_range="
        f"{deep_coverage_gate_min:.8f}-{deep_coverage_gate_max:.8f}",
        f"deep_reparam_coverage_gate_mean={deep_coverage_gate_mean:.8f}",
        f"deep_reparam_selected_start={deep_selected_start}",
        f"deep_reparam_network_parameters={deep_parameter_count}",
        "deep_reparam_start_train_rms_ms="
        + ",".join(f"{value:.6f}" for value in deep_start_train_rms_ms),
        f"deep_reparam_uncertainty_role="
        "initialization_ensemble_spread_not_posterior_standard_deviation",
        f"n_rays={sx.size}",
        f"n_outer={args.n_outer}",
        f"n_lsqr={args.n_lsqr}",
        f"alpha_reg={args.alpha_reg:.6f}",
        f"step_damp={args.step_damp:.6f}",
        f"dataset_contract_rows={int(contract_stats['row_count'])}",
        f"dataset_contract_sources={int(contract_stats['source_count'])}",
        f"dataset_contract_stations_per_source_range="
        f"{int(contract_stats['stations_per_source_min'])}-"
        f"{int(contract_stats['stations_per_source_max'])}",
        f"max_time_identity_error_ms={contract_stats['max_time_identity_error_ms']:.10f}",
        f"dropped_by_velocity_qc={dropped_by_velocity_qc}",
        f"velocity_qc_mode={'auto' if qc_auto else 'manual'}",
        f"velocity_qc_range={args.vmin_qc:.2f}-{args.vmax_qc:.2f}",
        f"model_velocity_mode={'auto' if model_auto else 'manual'}",
        f"model_velocity_range={args.vmin_model:.2f}-{args.vmax_model:.2f}",
        f"apparent_velocity_range={apparent_velocity_range_text}",
        f"apparent_velocity_mean={apparent_velocity_mean_text}",
        f"nominal_nonpositive_travel_times={int(np.count_nonzero(full_tt_s <= 0))}",
        f"event_centered_qc={'enabled' if args.event_centered_qc else 'disabled'}",
        f"event_centered_qc_reference_velocity={qc_reference_velocity:.2f}",
        f"qc_apparent_velocity_range={v_app_qc.min():.2f}-{v_app_qc.max():.2f}",
        f"qc_event_shift_range_ms={qc_event_shift_s.min() * 1000.0:.4f}-{qc_event_shift_s.max() * 1000.0:.4f}",
        f"grid={nx}x{ny}x{nz}",
        f"grid_nodes={nx + 1}x{ny + 1}x{nz + 1}",
        f"requested_grid_nodes={args.nx_nodes}x{args.ny_nodes}x{args.nz_nodes}",
        f"initial_model_source={initial_model_source}",
        f"bounds=x[{args.x_min:.2f},{args.x_max:.2f}],y[{args.y_min:.2f},{args.y_max:.2f}],z[{args.z_min:.2f},{args.z_max:.2f}]",
        f"cell_size=dx{grid_dx:.2f},dy{grid_dy:.2f},dz{grid_dz:.2f}",
        f"velocity_range={velocity.min():.2f}-{velocity.max():.2f}",
        f"display_velocity_range={display_velocity.min():.2f}-{display_velocity.max():.2f}",
        f"raw_velocity_range={raw_velocity.min():.2f}-{raw_velocity.max():.2f}",
        "velocity_field_role=forward_consistent_inverse_solution",
        "display_velocity_field_role=coverage_stabilized_visualization_only",
        f"background_velocity={v0:.2f}",
        f"outside_ray_mode={'enabled' if args.allow_outside_rays else 'disabled'}",
        f"roi_dropped_rays={roi_dropped_rays}",
        f"inside_path_length_range_m={traced_length.min():.4f}-{traced_length.max():.4f}",
        f"outside_path_length_range_m={outside_length.min():.4f}-{outside_length.max():.4f}",
        f"full_observed_travel_time_range_s={full_tt_s.min():.8f}-{full_tt_s.max():.8f}",
        f"modeled_roi_travel_time_range_s={tt_s.min():.8f}-{tt_s.max():.8f}",
        f"ray_length_normalization={'enabled' if args.ray_length_normalization else 'disabled'}",
        f"reference_ray_length_m={reference_ray_length:.4f}",
        "ray_kernel_type="
        + (
            "gaussian_ray_tube_approximation"
            if args.ray_kernel_sigma_xy_cells > 0.0
            or args.ray_kernel_sigma_z_cells > 0.0
            else "siddon_centerline"
        ),
        f"ray_kernel_sigma_xy_cells={args.ray_kernel_sigma_xy_cells:.6f}",
        f"ray_kernel_sigma_z_cells={args.ray_kernel_sigma_z_cells:.6f}",
        f"ray_kernel_relative_cutoff={args.ray_kernel_relative_cutoff:.8f}",
        f"ray_kernel_centerline_nnz={centerline_nnz}",
        f"ray_kernel_broadened_nnz={gmat.nnz}",
        f"ray_kernel_nnz_ratio={kernel_nnz_ratio:.8f}",
        f"ray_kernel_row_sum_error_max_m={max_kernel_row_sum_error:.12e}",
        f"kernel_supported_cells={int(np.count_nonzero(kernel_support_density))}/{kernel_support_density.size}",
        f"regularize_total_model={'enabled' if args.regularize_total_model else 'disabled'}",
        f"curvature_reg_factor={args.curvature_reg_factor:.6f}",
        f"curvature_z_factor={args.curvature_z_factor:.6f}",
        f"curvature_rows={cmat.shape[0]}",
        f"spline_projection={'enabled' if args.spline_projection else 'disabled'}",
        f"spline_control_grid={args.spline_control_nx}x{args.spline_control_ny}",
        f"spline_projection_strength={args.spline_projection_strength:.6f}",
        f"spline_free_parameters_per_z="
        f"{args.spline_control_nx * args.spline_control_ny if args.spline_projection else nx * ny}",
        f"hierarchical_parameterization="
        f"{'enabled' if args.hierarchical_parameterization else 'disabled'}",
        "hierarchical_basis_source=training_ray_geometry_only",
        f"hierarchical_split_rays={args.hierarchical_split_rays:.6f}",
        f"hierarchical_min_block="
        f"{args.hierarchical_min_block_x}x{args.hierarchical_min_block_y}",
        f"hierarchical_model_parameters={n_model_parameters}",
        f"hierarchical_parameter_reduction_fraction="
        f"{1.0 - n_model_parameters / n_model:.8f}",
        f"hierarchical_leaf_size_cells_range="
        f"{int(np.min(hierarchical_leaf_sizes))}-{int(np.max(hierarchical_leaf_sizes))}",
        f"edge_preserving_tv={'enabled' if args.edge_preserving_tv else 'disabled'}",
        f"tv_epsilon_last_s_per_m={tv_epsilon:.10e}",
        f"joint_sparsity={'enabled' if args.joint_sparsity else 'disabled'}",
        "wavelet_family=haar_2d_per_z_slice",
        f"wavelet_levels={args.wavelet_levels}",
        f"wavelet_threshold_factor={args.wavelet_threshold_factor:.6f}",
        f"wavelet_threshold_last_s_per_m={wavelet_threshold:.10e}",
        f"wavelet_detail_retained_fraction_last={wavelet_retained_fraction:.8f}",
        f"differential_times={'enabled' if n_differential else 'disabled'}",
        f"differential_weight={args.differential_weight:.6f}",
        f"differential_pairs={n_differential}",
        f"solution_differential_rms_s={solution_differential_rms:.10f}",
        f"requested_min_ray_coverage={args.min_ray_coverage:.2f}",
        f"effective_min_ray_coverage={reliable_coverage:.2f}",
        f"coverage_weight_exponent={args.coverage_weight_exponent:.6f}",
        f"background_prior_weight_range={background_weight.min():.6f}-{background_weight.max():.6f}",
        f"coverage_stabilized_cells={int(np.count_nonzero(coverage_weight < 1.0))}",
        f"path_length_error_max_m={max_path_length_error:.10f}",
        f"initial_rms_s={rms_hist[0][1] if rms_hist else best_rms:.10f}",
        f"best_rms_s={best_rms:.10f}",
        f"best_validation_rms_s={best_validation_rms:.10f}",
        f"solution_train_rms_s={solution_train_rms:.10f}",
        f"solution_validation_rms_s={solution_validation_rms:.10f}",
        f"display_train_rms_s={display_train_rms:.10f}",
        f"display_validation_rms_s={display_validation_rms:.10f}",
        "validation_split=grouped_by_source_event",
        "validation_metric=event_median_centered_rms",
        f"training_rays={train_idx.size}",
        f"validation_rays={val_idx.size}",
        f"validation_metric_rays={validation_metric_rays}",
        f"training_sources={int(np.unique(source_ids[train_idx]).size)}",
        f"validation_sources={int(np.unique(source_ids[val_idx]).size) if n_val else 0}",
        "validation_source_ids="
        + ",".join(str(int(item)) for item in np.sort(validation_sources)),
        f"huber_delta={args.huber_delta:.3f}",
        f"background_damping={args.background_damping:.3f}",
        f"model_damping={args.model_damping:.6f}",
        f"source_time_inversion={'enabled' if args.invert_source_statics else 'disabled'}",
        f"global_time_damping={args.global_time_damping:.6f}",
        f"source_static_damping={args.source_static_damping:.6f}",
        f"global_time_correction_ms={global_time_correction * 1000.0:.6f}",
        f"source_relative_correction_range_ms="
        f"{source_time_deviations.min() * 1000.0 if source_time_deviations.size else 0.0:.6f}-"
        f"{source_time_deviations.max() * 1000.0 if source_time_deviations.size else 0.0:.6f}",
        f"max_time_correction_ms={args.max_time_correction * 1000.0:.3f}",
        f"covered_cells={int(np.count_nonzero(ray_density))}/{ray_density.size}",
        f"covered_fraction={covered_fraction:.8f}",
        f"reliable_coverage_fraction={reliable_fraction:.8f}",
        f"max_ray_density={int(ray_density.max()) if ray_density.size else 0}",
        "",
        "RMS history(iter,rms_before,rms_after,solver_flag,solver_iter)",
    ]
    for it, r0, r1, flag, niter_done in rms_hist:
        stats_lines.append(f"{it},{r0:.8f},{r1:.8f},{flag},{niter_done}")
    stats_lines.append("")
    stats_lines.append("validation_history(iter,rms)")
    for it, val_rms in validation_hist:
        stats_lines.append(f"{it},{val_rms:.8f}")

    stats_lines.append("")
    stats_lines.append("slice_stats(z_target,z_used,covered_cells,reliable_cells,max_ray_density,vmin,vmean,vmax)")

    for zt in targets:
        z_used = float(zt)

        # Interpolate to the requested elevation, then transpose for plotting.
        vs = interpolate_z_slice(vel_show, zc, zt).T
        density_slice = interpolate_z_slice(ray_density3d, zc, zt).T
        covered_cells = int(np.count_nonzero(density_slice))
        reliable_cells = int(np.count_nonzero(density_slice >= reliable_coverage))
        max_density = int(density_slice.max()) if density_slice.size else 0
        stats_lines.append(
            f"{zt:.1f},{z_used:.1f},{covered_cells},{reliable_cells},{max_density},"
            f"{vs.min():.2f},{vs.mean():.2f},{vs.max():.2f}"
        )

        vs_smooth = gaussian_filter(vs, sigma=0.9)
        spline = RectBivariateSpline(yc, xc, vs_smooth)
        vf = spline(yf, xf)
        vf = np.clip(vf, args.vmin_model, args.vmax_model)
        fig, ax = plt.subplots(figsize=(8.0, 6.0), dpi=150, facecolor="white")
        levels = np.linspace(args.vmin_model, args.vmax_model, 80)
        cf = ax.contourf(
            xfine, yfine, vf, levels=levels, cmap="jet",
            vmin=args.vmin_model, vmax=args.vmax_model,
        )

        density_spline = RectBivariateSpline(yc, xc, density_slice, kx=1, ky=1)
        density_fine = density_spline(yf, xf)
        if np.nanmax(density_fine) >= reliable_coverage:
            ax.contour(
                xfine,
                yfine,
                density_fine,
                levels=[reliable_coverage],
                colors="#111827",
                linestyles="--",
                linewidths=1.0,
            )
            ax.text(
                0.01,
                0.015,
                f"虚线内：可靠覆盖（≥{reliable_coverage:g}条射线/单元）",
                transform=ax.transAxes,
                fontsize=8.5,
                color="#111827",
                bbox={"facecolor": "white", "edgecolor": "#94A3B8", "alpha": 0.82, "pad": 2.0},
                zorder=6,
            )

        # 模型边界样式参考原成果图的黑白边界线。
        bx = [args.x_min, args.x_max, args.x_max, args.x_min, args.x_min]
        by = [args.y_min, args.y_min, args.y_max, args.y_max, args.y_min]
        ax.plot(bx, by, "k-", linewidth=2.2)
        ax.plot(bx, by, "w--", linewidth=0.9)

        # 台站投影位置，帮助读图时判断射线覆盖。
        station_xy = np.unique(np.column_stack([rx, ry]), axis=0)
        ax.scatter(station_xy[:, 0], station_xy[:, 1], marker="^", s=28, c="white", edgecolors="black", linewidths=0.8, zorder=4)

        cb = fig.colorbar(cf, ax=ax)
        cb.set_label("P波速度 (m/s)", fontsize=12)
        cb.set_ticks(np.linspace(args.vmin_model, args.vmax_model, 6))

        if args.deep_reparameterization:
            solver_label = "DNR-2026"
        elif args.solver_method == "sirt":
            if args.hierarchical_parameterization:
                solver_label = "SIRT-TV+Hierarchical"
            elif args.joint_sparsity and args.spline_projection:
                solver_label = "SIRT-TV+Wavelet+Spline"
            elif args.spline_projection:
                solver_label = "SIRT-TV+Spline"
            elif args.joint_sparsity:
                solver_label = "SIRT-TV+Wavelet"
            else:
                solver_label = "SIRT-TV" if args.edge_preserving_tv else "SIRT"
        elif args.hierarchical_parameterization:
            solver_label = "LSQR-TV+Hierarchical"
        elif args.joint_sparsity and args.spline_projection:
            solver_label = "LSQR-TV+Wavelet+Spline"
        elif args.spline_projection:
            solver_label = "LSQR-TV+Spline"
        elif args.joint_sparsity:
            solver_label = "LSQR-TV+Wavelet"
        else:
            solver_label = "LSQR-TV" if args.edge_preserving_tv else "LSQR"
        ax.set_title(f"标高 {z_used:.0f}m P波速度（{solver_label}）", fontsize=14, fontweight="bold")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_xlim(args.x_min, args.x_max)
        ax.set_ylim(args.y_min, args.y_max)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=11)
        ax.grid(False)

        out_png = args.output_dir / f"velocity_slice_z{int(zt)}.png"
        out_report_png = args.output_dir / f"velocity_z{int(zt)}.png"
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.savefig(out_report_png, dpi=150)
        plt.close(fig)
        print(f"saved {out_png.name}")

        fig_cov, ax_cov = plt.subplots(figsize=(8.0, 6.0), dpi=150, facecolor="white")
        im = ax_cov.imshow(
            density_slice,
            origin="lower",
            extent=[args.x_min, args.x_max, args.y_min, args.y_max],
            cmap="viridis",
            aspect="equal",
        )
        ax_cov.plot(bx, by, "w-", linewidth=1.8)
        ax_cov.scatter(station_xy[:, 0], station_xy[:, 1], marker="^", s=24, c="white", edgecolors="black", linewidths=0.7)
        cb_cov = fig_cov.colorbar(im, ax=ax_cov)
        cb_cov.set_label("穿过网格的射线条数", fontsize=11)
        ax_cov.set_title(f"标高 {z_used:.0f}m 射线覆盖", fontsize=14, fontweight="bold")
        ax_cov.set_xlabel("X (m)")
        ax_cov.set_ylabel("Y (m)")
        ax_cov.set_xlim(args.x_min, args.x_max)
        ax_cov.set_ylim(args.y_min, args.y_max)
        fig_cov.tight_layout()
        fig_cov.savefig(args.output_dir / f"ray_coverage_z{int(zt)}.png", dpi=150)
        plt.close(fig_cov)

    if args.deep_reparameterization:
        uncertainty3d = deep_velocity_uncertainty.reshape(
            (nx, ny, nz), order="F"
        )
        for zt in targets:
            uncertainty_slice = interpolate_z_slice(uncertainty3d, zc, zt)
            uncertainty_coverage = interpolate_z_slice(
                ray_density3d, zc, zt
            )
            fig_uncertainty, ax_uncertainty = plt.subplots(
                figsize=(8.8, 5.8)
            )
            uncertainty_image = ax_uncertainty.pcolormesh(
                xc,
                yc,
                uncertainty_slice.T / 1000.0,
                shading="nearest",
                cmap="magma",
            )
            fig_uncertainty.colorbar(
                uncertainty_image,
                ax=ax_uncertainty,
                label="velocity std (km/s)",
            )
            if float(np.nanmax(uncertainty_coverage)) >= reliable_coverage:
                ax_uncertainty.contour(
                    xc,
                    yc,
                    uncertainty_coverage.T,
                    levels=[reliable_coverage],
                    colors="cyan",
                    linewidths=1.0,
                )
            ax_uncertainty.set_title(
                f"z={zt:.3f} m DNR initialization spread "
                f"(cyan: ≥{reliable_coverage:g} rays)"
            )
            ax_uncertainty.set_xlabel("X (m)")
            ax_uncertainty.set_ylabel("Y (m)")
            ax_uncertainty.set_xlim(args.x_min, args.x_max)
            ax_uncertainty.set_ylim(args.y_min, args.y_max)
            fig_uncertainty.tight_layout()
            fig_uncertainty.savefig(
                args.output_dir / f"velocity_uncertainty_z{int(zt)}.png",
                dpi=150,
            )
            plt.close(fig_uncertainty)

    np.savez_compressed(
        args.output_dir / "velocity_model.npz",
        velocity=vel3d,
        raw_velocity=raw_velocity.reshape((nx, ny, nz), order="F"),
        display_velocity=display_vel3d,
        velocity_uncertainty=deep_velocity_uncertainty.reshape(
            (nx, ny, nz), order="F"
        ),
        deep_ensemble_velocity=np.stack(
            [
                item.reshape((nx, ny, nz), order="F")
                for item in deep_ensemble_velocity
            ],
            axis=0,
        )
        if deep_ensemble_velocity.shape[0]
        else np.empty((0, nx, ny, nz), dtype=np.float64),
        ray_density=ray_density3d,
        kernel_support_density=kernel_support_density.reshape((nx, ny, nz), order="F"),
        coverage_weight=coverage_weight.reshape((nx, ny, nz), order="F"),
        coverage_weight_exponent=np.asarray(args.coverage_weight_exponent),
        xc=xc,
        yc=yc,
        zc=zc,
        xnodes=xnodes,
        ynodes=ynodes,
        znodes=znodes,
        background_velocity_mps=v0,
        reliable_coverage=reliable_coverage,
        model_schema_version=np.asarray(2, dtype=np.int64),
        inversion_method=np.asarray(inversion_method),
        solver_method=np.asarray(args.solver_method),
        sirt_iterations=np.asarray(args.sirt_iterations or args.n_lsqr, dtype=np.int64),
        sirt_omega=np.asarray(args.sirt_omega, dtype=np.float64),
        sirt_step_damp=np.asarray(args.sirt_step_damp, dtype=np.float64),
        sirt_tolerance=np.asarray(args.sirt_tolerance, dtype=np.float64),
        sirt_auto_tune=np.asarray(args.sirt_auto_tune),
        sirt_tuned_alpha=np.asarray(sirt_tuned_alpha, dtype=np.float64),
        sirt_tuned_omega=np.asarray(sirt_tuned_omega, dtype=np.float64),
        sirt_tuned_metric=np.asarray(sirt_tuned_metric, dtype=np.float64),
        sirt_tuning_evaluations=np.asarray(sirt_tuning_evaluations, dtype=np.int64),
        sirt_tuning_status=np.asarray(sirt_tuning_status),
        deep_reparameterization_enabled=np.asarray(
            args.deep_reparameterization
        ),
        deep_reparam_selected_start=np.asarray(
            deep_selected_start, dtype=np.int64
        ),
        deep_reparam_network_parameters=np.asarray(
            deep_parameter_count, dtype=np.int64
        ),
        deep_reparam_start_train_rms_ms=deep_start_train_rms_ms,
        deep_reparam_fourier_bands=np.asarray(
            args.deep_reparam_fourier_bands, dtype=np.int64
        ),
        deep_reparam_differential_loss_fraction=np.asarray(
            args.deep_reparam_differential_loss_fraction, dtype=np.float64
        ),
        velocity_uncertainty_role=np.asarray(
            "initialization_ensemble_spread_not_posterior_standard_deviation"
        ),
        velocity_field_role=np.asarray("forward_consistent_inverse_solution"),
        display_velocity_field_role=np.asarray("coverage_stabilized_visualization_only"),
        global_time_correction_s=global_time_correction,
        source_ids=unique_sources,
        source_time_deviations_s=source_time_deviations,
        used_observation_row_indices=observation_row_indices,
        validation_source_ids=np.asarray(validation_sources, dtype=np.int64),
        initial_model_source=np.asarray(initial_model_source),
        joint_sparsity_enabled=np.asarray(args.joint_sparsity),
        wavelet_family=np.asarray("haar_2d_per_z_slice"),
        wavelet_levels=np.asarray(args.wavelet_levels, dtype=np.int64),
        wavelet_threshold_factor=np.asarray(args.wavelet_threshold_factor),
        differential_times_enabled=np.asarray(bool(n_differential)),
        differential_weight=np.asarray(args.differential_weight),
        regularize_total_model=np.asarray(args.regularize_total_model),
        model_damping=np.asarray(args.model_damping),
        curvature_reg_factor=np.asarray(args.curvature_reg_factor),
        curvature_z_factor=np.asarray(args.curvature_z_factor),
        spline_projection=np.asarray(args.spline_projection),
        spline_control_nx=np.asarray(args.spline_control_nx, dtype=np.int64),
        spline_control_ny=np.asarray(args.spline_control_ny, dtype=np.int64),
        spline_projection_strength=np.asarray(args.spline_projection_strength),
        hierarchical_parameterization=np.asarray(args.hierarchical_parameterization),
        hierarchical_basis_source=np.asarray("training_ray_geometry_only"),
        hierarchical_split_rays=np.asarray(args.hierarchical_split_rays),
        hierarchical_min_block_x=np.asarray(args.hierarchical_min_block_x, dtype=np.int64),
        hierarchical_min_block_y=np.asarray(args.hierarchical_min_block_y, dtype=np.int64),
        hierarchical_model_parameters=np.asarray(n_model_parameters, dtype=np.int64),
        hierarchical_training_ray_density=hierarchical_training_ray_density.reshape(
            (nx, ny, nz), order="F"
        ),
        hierarchical_leaf_index=hierarchical_leaf_index,
        hierarchical_leaf_table=hierarchical_leaf_table,
        ray_kernel_type=np.asarray(
            "gaussian_ray_tube_approximation"
            if args.ray_kernel_sigma_xy_cells > 0.0
            or args.ray_kernel_sigma_z_cells > 0.0
            else "siddon_centerline"
        ),
        ray_kernel_sigma_xy_cells=np.asarray(args.ray_kernel_sigma_xy_cells),
        ray_kernel_sigma_z_cells=np.asarray(args.ray_kernel_sigma_z_cells),
        ray_kernel_relative_cutoff=np.asarray(args.ray_kernel_relative_cutoff),
        ray_kernel_row_sum_error_max_m=np.asarray(max_kernel_row_sum_error),
    )

    stats_path = args.output_dir / "slice_report.txt"
    stats_path.write_text("\n".join(stats_lines), encoding="utf-8")
    save_rms_curve(args.output_dir, rms_hist, validation_hist)
    if args.export_deliverables:
        from wave_ct.deliverables import export_model_text_bundle

        export_model_text_bundle(
            args.output_dir / "velocity_model.npz",
            args.output_dir,
            targets,
            image_source_dir=args.output_dir,
        )

    print(f"Output dir : {args.output_dir}")
    print(f"Report txt : {stats_path}")


if __name__ == "__main__":
    main()
