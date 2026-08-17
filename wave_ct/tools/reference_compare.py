"""Quantitatively compare a Wave CT model with supplied reference TXT grids."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr

from wave_ct.cad import load_segment_cache


matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
matplotlib.rcParams["axes.unicode_minus"] = False


VELOCITY_RE = re.compile(r"^波速分布\s*z\s*([-+]?\d+(?:\.\d+)?)\.txt$", re.IGNORECASE)
POINT_COLUMNS = [
    "z",
    "x",
    "y",
    "reference_velocity_mps",
    "wave_ct_velocity_mps",
    "difference_mps",
    "reference_anomaly",
    "wave_ct_anomaly_reference_baseline",
    "wave_ct_anomaly_param4200",
    "ray_density",
]


def parse_z(path: Path) -> float:
    match = VELOCITY_RE.match(path.name)
    if not match:
        raise ValueError(f"cannot parse elevation from {path.name}")
    return float(match.group(1))


def load_txt(path: Path) -> np.ndarray:
    data = np.loadtxt(path, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError(f"invalid reference grid: {path}")
    return data[:, :3]


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def metric_block(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float | int]:
    valid = np.isfinite(reference) & np.isfinite(candidate)
    reference = reference[valid]
    candidate = candidate[valid]
    if reference.size == 0:
        return {
            "count": 0,
            "pearson": float("nan"),
            "spearman": float("nan"),
            "rmse_mps": float("nan"),
            "mae_mps": float("nan"),
            "bias_mps": float("nan"),
            "reference_mean_mps": float("nan"),
            "candidate_mean_mps": float("nan"),
            "reference_std_mps": float("nan"),
            "candidate_std_mps": float("nan"),
        }
    difference = candidate - reference
    if (
        reference.size >= 2
        and np.std(reference) > 1e-12
        and np.std(candidate) > 1e-12
    ):
        spearman_result = spearmanr(reference, candidate)
        rho = float(
            getattr(
                spearman_result,
                "statistic",
                getattr(spearman_result, "correlation", float("nan")),
            )
        )
    else:
        rho = float("nan")
    return {
        "count": int(reference.size),
        "pearson": pearson(reference, candidate),
        "spearman": float(rho),
        "rmse_mps": float(np.sqrt(np.mean(difference**2))),
        "mae_mps": float(np.mean(np.abs(difference))),
        "bias_mps": float(np.mean(difference)),
        "reference_mean_mps": float(np.mean(reference)),
        "candidate_mean_mps": float(np.mean(candidate)),
        "reference_std_mps": float(np.std(reference)),
        "candidate_std_mps": float(np.std(candidate)),
    }


def load_anomaly(dataset_root: Path, z: float) -> np.ndarray | None:
    candidates = list(dataset_root.glob(f"波速异常值分布*z*{z:.3f}.txt"))
    if not candidates:
        return None
    return load_txt(candidates[0])


def load_observations(path: Path) -> dict[str, np.ndarray]:
    source_ids: list[int] = []
    source_xyz: list[tuple[float, float, float]] = []
    station_xyz: list[tuple[float, float, float]] = []
    travel_sec: list[float] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            source_ids.append(int(float(row["震源编号"])))
            source_xyz.append(
                (
                    float(row["震源坐标-x"]),
                    float(row["震源坐标-y"]),
                    float(row["震源坐标-z"]),
                )
            )
            station_xyz.append(
                (
                    float(row["台站坐标-x"]),
                    float(row["台站坐标-y"]),
                    float(row["台站坐标-z"]),
                )
            )
            travel_sec.append(float(row["震源-台站传播时间"]) / 1000.0)
    return {
        "source_id": np.asarray(source_ids, dtype=np.int64),
        "source_xyz": np.asarray(source_xyz, dtype=np.float64),
        "station_xyz": np.asarray(station_xyz, dtype=np.float64),
        "travel_sec": np.asarray(travel_sec, dtype=np.float64),
    }


def integrate_straight_rays(
    interpolator: RegularGridInterpolator,
    source_xyz: np.ndarray,
    station_xyz: np.ndarray,
    step_m: float = 5.0,
) -> np.ndarray:
    predictions = np.empty(source_xyz.shape[0], dtype=np.float64)
    for index, (source, station) in enumerate(zip(source_xyz, station_xyz)):
        vector = station - source
        distance = float(np.linalg.norm(vector))
        segment_count = max(1, int(np.ceil(distance / step_m)))
        alpha = (np.arange(segment_count, dtype=np.float64) + 0.5) / segment_count
        points = source[None, :] + alpha[:, None] * vector[None, :]
        velocity = np.asarray(interpolator(points), dtype=np.float64)
        velocity = np.maximum(velocity, 100.0)
        predictions[index] = np.sum((distance / segment_count) / velocity)
    return predictions


def centers_to_edges(centers: np.ndarray) -> np.ndarray:
    """Recover uniform/rectilinear cell edges from stored cell centres."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.ndim != 1 or centers.size < 2 or np.any(np.diff(centers) <= 0.0):
        raise ValueError("model cell centres must be one-dimensional and increasing")
    midpoints = 0.5 * (centers[:-1] + centers[1:])
    first = centers[0] - (midpoints[0] - centers[0])
    last = centers[-1] + (centers[-1] - midpoints[-1])
    return np.concatenate([[first], midpoints, [last]])


def predict_piecewise_constant_velocity(
    velocity: np.ndarray,
    xnodes: np.ndarray,
    ynodes: np.ndarray,
    znodes: np.ndarray,
    source_xyz: np.ndarray,
    station_xyz: np.ndarray,
    background_velocity: float,
) -> np.ndarray:
    """Use Wave CT's native Siddon operator for an output-consistent check."""
    from wave_ct.inversion import build_siddon_matrix

    velocity = np.asarray(velocity, dtype=np.float64)
    nx, ny, nz = velocity.shape
    gmat, _ = build_siddon_matrix(
        source_xyz[:, 0],
        source_xyz[:, 1],
        source_xyz[:, 2],
        station_xyz[:, 0],
        station_xyz[:, 1],
        station_xyz[:, 2],
        np.asarray(xnodes, dtype=np.float64),
        np.asarray(ynodes, dtype=np.float64),
        np.asarray(znodes, dtype=np.float64),
        nx,
        ny,
        nz,
    )
    slowness = 1.0 / np.maximum(velocity.reshape(-1, order="F"), 100.0)
    traced_length = np.asarray(gmat.sum(axis=1)).ravel()
    full_length = np.linalg.norm(station_xyz - source_xyz, axis=1)
    outside_length = np.maximum(full_length - traced_length, 0.0)
    return np.asarray(gmat @ slowness).ravel() + outside_length / background_velocity


def forward_fit_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
    source_ids: np.ndarray,
) -> dict[str, float | int]:
    residual = observed - predicted
    centered_parts: list[np.ndarray] = []
    for source_id in np.unique(source_ids):
        part = residual[source_ids == source_id]
        if part.size >= 2:
            centered_parts.append(part - np.median(part))
    centered = np.concatenate(centered_parts) if centered_parts else np.asarray([], dtype=float)
    return {
        "ray_count": int(observed.size),
        "event_count": int(np.unique(source_ids).size),
        "raw_rms_ms": float(np.sqrt(np.mean(residual**2)) * 1000.0),
        "raw_mae_ms": float(np.mean(np.abs(residual)) * 1000.0),
        "event_centered_ray_count": int(centered.size),
        "event_centered_rms_ms": (
            float(np.sqrt(np.mean(centered**2)) * 1000.0) if centered.size else float("nan")
        ),
        "observed_predicted_pearson": pearson(observed, predicted),
        "residual_median_ms": float(np.median(residual) * 1000.0),
    }


def reference_volume(
    loaded_reference: list[tuple[Path, float, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.unique(np.concatenate([item[2][:, 0] for item in loaded_reference]))
    y = np.unique(np.concatenate([item[2][:, 1] for item in loaded_reference]))
    z = np.asarray(sorted(item[1] for item in loaded_reference), dtype=np.float64)
    volume = np.full((x.size, y.size, z.size), np.nan, dtype=np.float64)
    x_index = {float(value): index for index, value in enumerate(x)}
    y_index = {float(value): index for index, value in enumerate(y)}
    z_index = {float(value): index for index, value in enumerate(z)}
    for _, elevation, data in loaded_reference:
        for row in data:
            volume[x_index[float(row[0])], y_index[float(row[1])], z_index[elevation]] = row[2]
    if not np.isfinite(volume).all():
        raise ValueError("reference TXT files do not form one complete Cartesian volume")
    return x, y, z, volume


def plot_comparison(
    output_path: Path,
    z: float,
    x: np.ndarray,
    y: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    density: np.ndarray,
    vmin: float,
    vmax: float,
) -> None:
    difference = candidate - reference
    diff_limit = max(250.0, float(np.percentile(np.abs(difference), 95.0)))
    x_origin = float(np.floor(np.min(x) / 100.0) * 100.0)
    y_origin = float(np.floor(np.min(y) / 100.0) * 100.0)
    x_relative = x - x_origin
    y_relative = y - y_origin
    fig = plt.figure(figsize=(18.0, 5.4), dpi=150, facecolor="white")
    grid = fig.add_gridspec(
        1, 5,
        width_ratios=[1.0, 1.0, 0.045, 1.0, 0.045],
        left=0.05, right=0.96, bottom=0.13, top=0.79, wspace=0.32,
    )
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[0, 3]),
    ]
    velocity_colorbar_ax = fig.add_subplot(grid[0, 2])
    difference_colorbar_ax = fig.add_subplot(grid[0, 4])
    levels = np.linspace(vmin, vmax, 60)
    first = axes[0].tricontourf(
        x_relative, y_relative, reference, levels=levels, cmap="jet", extend="both"
    )
    axes[0].set_title("Reference TXT")
    axes[1].tricontourf(
        x_relative, y_relative, candidate, levels=levels, cmap="jet", extend="both"
    )
    axes[1].set_title("Wave CT (same scale)")
    diff_levels = np.linspace(-diff_limit, diff_limit, 61)
    third = axes[2].tricontourf(
        x_relative, y_relative, difference,
        levels=diff_levels, cmap="coolwarm", extend="both",
    )
    axes[2].set_title("Wave CT - reference")
    for ax in axes:
        ax.set_xlabel("Relative X (m)")
        ax.set_aspect("equal")
    axes[0].set_ylabel("Relative Y (m)")
    covered = density > 0
    axes[1].scatter(
        x_relative[~covered], y_relative[~covered],
        s=4, c="black", alpha=0.18, label="No ray",
    )
    if np.any(~covered):
        axes[1].legend(loc="upper right", fontsize=8)
    fig.colorbar(first, cax=velocity_colorbar_ax, label="P velocity (m/s)")
    fig.colorbar(third, cax=difference_colorbar_ax, label="Difference (m/s)")
    fig.suptitle(
        f"Elevation {z:.3f} m: numeric comparison\n"
        f"Relative-coordinate origin: X={x_origin:.0f} m, Y={y_origin:.0f} m",
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(output_path)
    plt.close(fig)


def load_cad_plot_context(source_file: Path) -> tuple[list[np.ndarray], tuple[float, float] | None, tuple[float, float] | None]:
    if not source_file.is_file():
        return [], None, None
    values: dict[str, str] = {}
    for line in source_file.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    segments: list[np.ndarray] = []
    cache_text = values.get("vector_cache", "")
    if cache_text:
        cache_path = Path(cache_text)
        if cache_path.is_file():
            segments = load_segment_cache(cache_path)
    window = values.get("plot_window", "")
    match = re.search(
        r"x\[([-+\d.eE]+),([-+\d.eE]+)\],y\[([-+\d.eE]+),([-+\d.eE]+)\]",
        window,
    )
    if match:
        xlim = (float(match.group(1)), float(match.group(2)))
        ylim = (float(match.group(3)), float(match.group(4)))
    else:
        xlim = ylim = None
    return segments, xlim, ylim


def smooth_rectangular_field(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    outside_value: float,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique_x = np.unique(x)
    unique_y = np.unique(y)
    grid = np.full((unique_y.size, unique_x.size), np.nan, dtype=np.float64)
    x_index = {float(value): index for index, value in enumerate(unique_x)}
    y_index = {float(value): index for index, value in enumerate(unique_y)}
    for px, py, value in zip(x, y, values):
        grid[y_index[float(py)], x_index[float(px)]] = float(value)
    if not np.all(np.isfinite(grid)):
        raise ValueError("reference comparison points do not form a complete rectangular grid")
    if sigma > 0.0:
        grid = gaussian_filter(grid, sigma=sigma, mode="nearest")
    order_x = min(3, max(1, unique_y.size - 1))
    order_y = min(3, max(1, unique_x.size - 1))
    spline = RectBivariateSpline(unique_y, unique_x, grid, kx=order_x, ky=order_y)
    xf = np.linspace(xlim[0], xlim[1], 720)
    yf = np.linspace(ylim[0], ylim[1], 420)
    field = spline(yf, xf)
    outside = (
        (xf[None, :] < unique_x[0])
        | (xf[None, :] > unique_x[-1])
        | (yf[:, None] < unique_y[0])
        | (yf[:, None] > unique_y[-1])
    )
    field[outside] = outside_value
    return xf, yf, field


def draw_cad(ax: plt.Axes, segments: list[np.ndarray], xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    for segment in segments:
        if (
            segment[:, 0].max() < xlim[0]
            or segment[:, 0].min() > xlim[1]
            or segment[:, 1].max() < ylim[0]
            or segment[:, 1].min() > ylim[1]
        ):
            continue
        ax.plot(segment[:, 0], segment[:, 1], color="#374151", linewidth=0.38, alpha=0.50, zorder=4)


def plot_same_basemap_comparison(
    output_path: Path,
    z: float,
    x: np.ndarray,
    y: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    reference_background: float,
    candidate_background: float,
    segments: list[np.ndarray],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    vmin: float,
    vmax: float,
    sigma: float,
) -> None:
    xf, yf, reference_field = smooth_rectangular_field(
        x, y, reference, xlim, ylim, reference_background, sigma
    )
    _, _, candidate_field = smooth_rectangular_field(
        x, y, candidate, xlim, ylim, candidate_background, sigma
    )
    xmesh, ymesh = np.meshgrid(xf, yf)
    levels = np.linspace(vmin, vmax, 120)
    fig, axes = plt.subplots(2, 1, figsize=(13.2, 9.0), dpi=180, facecolor="white")
    contour = None
    for ax, field, title in zip(
        axes,
        (reference_field, candidate_field),
        ("学生/Surfer参考速度场", "WaveCT反演速度场（同底图、同色标、同平滑）"),
    ):
        contour = ax.contourf(
            xmesh, ymesh, np.clip(field, vmin, vmax),
            levels=levels, cmap="jet", extend="both", antialiased=True, zorder=1,
        )
        draw_cad(ax, segments, xlim, ylim)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(title, fontsize=12)
    assert contour is not None
    colorbar = fig.colorbar(contour, ax=list(axes), fraction=0.024, pad=0.018)
    colorbar.set_label("P波速度 (m/s)")
    colorbar.set_ticks(np.linspace(vmin, vmax, 6))
    fig.suptitle(
        f"标高 {z:.3f}m 同基准工作面底图对比  |  展示平滑 σ={sigma:g} 网格",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5, 0.018,
        "平滑仅用于统一制图；数值评价仍使用未平滑反演模型。",
        ha="center", fontsize=9, color="#334155",
    )
    fig.subplots_adjust(left=0.025, right=0.91, top=0.91, bottom=0.055, hspace=0.15)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Compare Wave CT NPZ with reference TXT grids")
    parser.add_argument("--dataset-root", type=Path, default=project_root.parent / "测试数据集720")
    parser.add_argument("--result-dir", type=Path, default=project_root / "测试数据集720处理结果" / "反演结果")
    parser.add_argument("--output-dir", type=Path, default=project_root / "测试数据集720处理结果" / "参考对比")
    parser.add_argument("--reliable-coverage", type=float, default=5.0)
    parser.add_argument(
        "--cad-source-file", type=Path, default=None,
        help="workface_plot生成的cad_background_source.txt；用于同底图对比",
    )
    parser.add_argument("--presentation-vmin", type=float, default=0.0)
    parser.add_argument("--presentation-vmax", type=float, default=0.0)
    parser.add_argument("--presentation-sigma", type=float, default=1.0)
    parser.add_argument(
        "--velocity-field",
        choices=["auto", "velocity", "raw_velocity", "display_velocity"],
        default="auto",
        help="NPZ field used for quantitative comparison; auto uses the forward-consistent velocity field",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=project_root / "测试数据集720处理结果" / "inversion_input.csv",
    )
    parser.add_argument(
        "--validation-source-ids",
        default="",
        help="Comma/space separated source IDs reserved for validation.",
    )
    args = parser.parse_args()
    validation_source_ids = sorted(
        {int(token) for token in re.findall(r"\d+", args.validation_source_ids)}
    )

    model_path = args.result_dir / "velocity_model.npz"
    if not model_path.is_file():
        raise FileNotFoundError(f"velocity model not found: {model_path}")
    velocity_files = [
        path for path in sorted(args.dataset_root.glob("波速分布*z*.txt"))
        if VELOCITY_RE.match(path.name)
    ]
    if not velocity_files:
        raise RuntimeError(f"no reference velocity TXT files under: {args.dataset_root}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(model_path, allow_pickle=False) as model:
        available_fields = set(model.files)
        velocity_field = "velocity" if args.velocity_field == "auto" else args.velocity_field
        if velocity_field not in available_fields:
            raise KeyError(
                f"velocity field {velocity_field!r} not found in {model_path}; "
                f"available fields: {sorted(available_fields)}"
            )
        velocity = np.asarray(model[velocity_field], dtype=np.float64)
        presentation_velocity = np.asarray(
            model["display_velocity"] if "display_velocity" in available_fields else model[velocity_field],
            dtype=np.float64,
        )
        density = np.asarray(model["ray_density"], dtype=np.float64)
        xc = np.asarray(model["xc"], dtype=np.float64)
        yc = np.asarray(model["yc"], dtype=np.float64)
        zc = np.asarray(model["zc"], dtype=np.float64)
        xnodes = (
            np.asarray(model["xnodes"], dtype=np.float64)
            if "xnodes" in available_fields else centers_to_edges(xc)
        )
        ynodes = (
            np.asarray(model["ynodes"], dtype=np.float64)
            if "ynodes" in available_fields else centers_to_edges(yc)
        )
        znodes = (
            np.asarray(model["znodes"], dtype=np.float64)
            if "znodes" in available_fields else centers_to_edges(zc)
        )
        background_velocity = (
            float(np.asarray(model["background_velocity_mps"]).reshape(()))
            if "background_velocity_mps" in available_fields else 4200.0
        )
        used_observation_rows = (
            np.asarray(model["used_observation_row_indices"], dtype=np.int64)
            if "used_observation_row_indices" in available_fields else np.asarray([], dtype=np.int64)
        )
        velocity_field_role = (
            str(np.asarray(model["velocity_field_role"]).reshape(()))
            if "velocity_field_role" in available_fields and velocity_field == "velocity"
            else (
                str(np.asarray(model["display_velocity_field_role"]).reshape(()))
                if "display_velocity_field_role" in available_fields and velocity_field == "display_velocity"
                else "legacy_or_explicit_field"
            )
        )
    velocity_interp = RegularGridInterpolator(
        (xc, yc, zc), velocity, bounds_error=False, fill_value=background_velocity
    )
    density_interp = RegularGridInterpolator(
        (xc, yc, zc), density, method="linear", bounds_error=False, fill_value=0.0
    )
    presentation_interp = RegularGridInterpolator(
        (xc, yc, zc), presentation_velocity, bounds_error=False, fill_value=background_velocity
    )

    cad_source_file = args.cad_source_file or (args.result_dir / "cad_background_source.txt")
    cad_segments, cad_xlim, cad_ylim = load_cad_plot_context(cad_source_file)

    all_reference_values: list[np.ndarray] = []
    loaded_reference: list[tuple[Path, float, np.ndarray]] = []
    for path in velocity_files:
        z = parse_z(path)
        data = load_txt(path)
        values = data[:, 2] * 1000.0 if float(np.nanmedian(data[:, 2])) < 100.0 else data[:, 2]
        all_reference_values.append(values)
        loaded_reference.append((path, z, np.column_stack([data[:, :2], values])))
    reference_concat = np.concatenate(all_reference_values)
    common_vmin = float(np.nanmin(reference_concat))
    common_vmax = float(np.nanmax(reference_concat))

    point_rows: list[dict[str, object]] = []
    per_slice: list[dict[str, object]] = []
    inferred_baselines: list[float] = []
    for reference_path, z, data in loaded_reference:
        x, y, reference_velocity = data[:, 0], data[:, 1], data[:, 2]
        sample_z = float(np.clip(z, zc[0], zc[-1]))
        points = np.column_stack([x, y, np.full(x.size, sample_z)])
        candidate = np.asarray(velocity_interp(points), dtype=np.float64)
        presentation_candidate = np.asarray(presentation_interp(points), dtype=np.float64)
        ray_density = np.asarray(density_interp(points), dtype=np.float64)

        anomaly_data = load_anomaly(args.dataset_root, z)
        if anomaly_data is not None:
            anomaly_lookup = {
                (float(row[0]), float(row[1])): float(row[2]) for row in anomaly_data
            }
            reference_anomaly = np.asarray(
                [anomaly_lookup.get((float(px), float(py)), np.nan) for px, py in zip(x, y)],
                dtype=np.float64,
            )
            valid_baseline = np.isfinite(reference_anomaly) & (reference_anomaly > -0.99)
            baseline_values = reference_velocity[valid_baseline] / (1.0 + reference_anomaly[valid_baseline])
            inferred_baseline = float(np.median(baseline_values))
            inferred_baselines.append(inferred_baseline)
        else:
            reference_anomaly = np.full(reference_velocity.shape, np.nan)
            inferred_baseline = 4200.0

        candidate_anomaly_reference = (candidate - inferred_baseline) / inferred_baseline
        candidate_anomaly_param = (candidate - 4200.0) / 4200.0
        covered = ray_density > 0.0
        reliable = ray_density >= args.reliable_coverage
        informative = np.abs(reference_velocity - np.median(reference_velocity)) >= 100.0
        slice_metrics = {
            "z": z,
            "reference_file": str(reference_path.resolve()),
            "reference_anomaly_baseline_mps": inferred_baseline,
            "all_points": metric_block(reference_velocity, candidate),
            "covered_points": metric_block(reference_velocity[covered], candidate[covered]),
            "reliable_points": metric_block(reference_velocity[reliable], candidate[reliable]),
            "reference_informative_points": metric_block(
                reference_velocity[informative], candidate[informative]
            ),
            "coverage": {
                "covered": int(np.count_nonzero(covered)),
                "reliable": int(np.count_nonzero(reliable)),
                "total": int(x.size),
            },
        }
        if np.isfinite(reference_anomaly).any():
            valid_anomaly = np.isfinite(reference_anomaly)
            slice_metrics["anomaly_all_points"] = metric_block(
                reference_anomaly[valid_anomaly],
                candidate_anomaly_reference[valid_anomaly],
            )
        per_slice.append(slice_metrics)

        for index in range(x.size):
            point_rows.append(
                {
                    "z": f"{z:.3f}",
                    "x": f"{x[index]:.4f}",
                    "y": f"{y[index]:.4f}",
                    "reference_velocity_mps": f"{reference_velocity[index]:.6f}",
                    "wave_ct_velocity_mps": f"{candidate[index]:.6f}",
                    "difference_mps": f"{candidate[index] - reference_velocity[index]:.6f}",
                    "reference_anomaly": (
                        "" if not np.isfinite(reference_anomaly[index]) else f"{reference_anomaly[index]:.8f}"
                    ),
                    "wave_ct_anomaly_reference_baseline": f"{candidate_anomaly_reference[index]:.8f}",
                    "wave_ct_anomaly_param4200": f"{candidate_anomaly_param[index]:.8f}",
                    "ray_density": f"{ray_density[index]:.6f}",
                }
            )
        plot_comparison(
            args.output_dir / f"comparison_z{z:.3f}.png",
            z,
            x,
            y,
            reference_velocity,
            candidate,
            ray_density,
            common_vmin,
            common_vmax,
        )
        if cad_segments:
            comparison_xlim = cad_xlim or (float(np.min(x)), float(np.max(x)))
            comparison_ylim = cad_ylim or (float(np.min(y)), float(np.max(y)))
            presentation_vmin = (
                args.presentation_vmin if args.presentation_vmin > 0 else common_vmin
            )
            presentation_vmax = (
                args.presentation_vmax
                if args.presentation_vmax > presentation_vmin else common_vmax
            )
            plot_same_basemap_comparison(
                args.output_dir / f"same_basemap_comparison_z{z:.3f}.png",
                z, x, y, reference_velocity, presentation_candidate,
                inferred_baseline, background_velocity,
                cad_segments, comparison_xlim, comparison_ylim,
                float(presentation_vmin), float(presentation_vmax),
                max(0.0, args.presentation_sigma),
            )

    with (args.output_dir / "point_comparison.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=POINT_COLUMNS)
        writer.writeheader()
        writer.writerows(point_rows)

    summary = {
        "model": str(model_path.resolve()),
        "velocity_field_used": velocity_field,
        "velocity_field_role": velocity_field_role,
        "wave_ct_background_velocity_mps": background_velocity,
        "reference_values_used_as_inversion_prior": False,
        "reference_values_used_only_after_inversion_for_evaluation": True,
        "reference_velocity_scale_mps": [common_vmin, common_vmax],
        "inferred_reference_anomaly_baseline_mps": (
            float(np.median(inferred_baselines)) if inferred_baselines else None
        ),
        "per_slice": per_slice,
    }

    if args.input_csv.is_file():
        observations = load_observations(args.input_csv)
        ref_x, ref_y, ref_z, ref_velocity_volume = reference_volume(loaded_reference)
        reference_baseline = float(np.median(inferred_baselines)) if inferred_baselines else 4200.0
        reference_interp = RegularGridInterpolator(
            (ref_x, ref_y, ref_z),
            ref_velocity_volume,
            bounds_error=False,
            fill_value=reference_baseline,
        )
        wave_ct_interp_full = RegularGridInterpolator(
            (xc, yc, zc), velocity, bounds_error=False, fill_value=background_velocity
        )
        reference_prediction = integrate_straight_rays(
            reference_interp,
            observations["source_xyz"],
            observations["station_xyz"],
        )
        wave_ct_interpolated_prediction = integrate_straight_rays(
            wave_ct_interp_full,
            observations["source_xyz"],
            observations["station_xyz"],
        )
        wave_ct_prediction = predict_piecewise_constant_velocity(
            velocity,
            xnodes,
            ynodes,
            znodes,
            observations["source_xyz"],
            observations["station_xyz"],
            background_velocity,
        )
        summary["forward_fit_to_same_observed_travel_times"] = {
            "reference_txt_model": forward_fit_metrics(
                observations["travel_sec"],
                reference_prediction,
                observations["source_id"],
            ),
            "wave_ct_model": forward_fit_metrics(
                observations["travel_sec"],
                wave_ct_prediction,
                observations["source_id"],
            ),
            "wave_ct_interpolated_display_check": forward_fit_metrics(
                observations["travel_sec"],
                wave_ct_interpolated_prediction,
                observations["source_id"],
            ),
            "note": (
                "Both models use straight rays and event-centered residuals. "
                "Wave CT uses its native piecewise-constant Siddon operator; the reference TXT "
                "uses interpolation on its supplied grid. Outside each volume its own background is used."
            ),
        }
        if used_observation_rows.size:
            valid_rows = used_observation_rows[
                (used_observation_rows >= 0)
                & (used_observation_rows < observations["travel_sec"].size)
            ]
            qc_mask = np.zeros(observations["travel_sec"].size, dtype=bool)
            qc_mask[np.unique(valid_rows)] = True
            summary["forward_fit_on_inversion_qc_rows"] = {
                "observation_count": int(np.count_nonzero(qc_mask)),
                "reference_txt_model": forward_fit_metrics(
                    observations["travel_sec"][qc_mask],
                    reference_prediction[qc_mask],
                    observations["source_id"][qc_mask],
                ),
                "wave_ct_model": forward_fit_metrics(
                    observations["travel_sec"][qc_mask],
                    wave_ct_prediction[qc_mask],
                    observations["source_id"][qc_mask],
                ),
                "note": "This subset exactly matches the CSV row indices retained by Wave CT quality control.",
            }
        else:
            qc_mask = np.ones(observations["travel_sec"].size, dtype=bool)
        if validation_source_ids:
            validation_mask = np.isin(observations["source_id"], validation_source_ids) & qc_mask
            if np.any(validation_mask):
                summary["forward_fit_on_requested_validation_sources"] = {
                    "requested_source_ids": validation_source_ids,
                    "matched_event_count": int(
                        np.unique(observations["source_id"][validation_mask]).size
                    ),
                    "observation_count": int(np.count_nonzero(validation_mask)),
                    "reference_txt_model": forward_fit_metrics(
                        observations["travel_sec"][validation_mask],
                        reference_prediction[validation_mask],
                        observations["source_id"][validation_mask],
                    ),
                    "wave_ct_model": forward_fit_metrics(
                        observations["travel_sec"][validation_mask],
                        wave_ct_prediction[validation_mask],
                        observations["source_id"][validation_mask],
                    ),
                    "note": (
                        "Wave CT did not use these source events during inversion. "
                        "The supplied reference TXT has no documented training/validation split, "
                        "so its value here is not an independently verified holdout score."
                    ),
                }
    (args.output_dir / "comparison_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    text_lines = [
        "Wave CT 与参考 TXT 数值对比",
        "=" * 68,
        "参考值未参与反演，仅在反演完成后用于评价。",
        f"参考异常系数反推背景速度: {summary['inferred_reference_anomaly_baseline_mps']:.2f} m/s",
        "",
    ]
    for item in per_slice:
        all_metric = item["all_points"]
        reliable_metric = item["reliable_points"]
        text_lines.append(
            f"z={item['z']:.3f}m | all: r={all_metric['pearson']:.4f}, "
            f"RMSE={all_metric['rmse_mps']:.2f}m/s, bias={all_metric['bias_mps']:.2f}m/s | "
            f"reliable n={reliable_metric['count']}, r={reliable_metric['pearson']:.4f}, "
            f"RMSE={reliable_metric['rmse_mps']:.2f}m/s"
        )
    forward = summary.get("forward_fit_on_inversion_qc_rows")
    forward_label = "反演质控实际采用走时的正演一致性（逐事件去中位时差）:"
    if not isinstance(forward, dict):
        forward = summary.get("forward_fit_to_same_observed_travel_times")
        forward_label = "同一批观测走时的正演一致性（逐事件去中位时差）:"
    if isinstance(forward, dict):
        text_lines.extend(
            [
                "",
                forward_label,
                "  Reference TXT: "
                f"{forward['reference_txt_model']['event_centered_rms_ms']:.3f} ms",
                "  Wave CT:       "
                f"{forward['wave_ct_model']['event_centered_rms_ms']:.3f} ms",
            ]
        )
    validation_forward = summary.get("forward_fit_on_requested_validation_sources")
    if isinstance(validation_forward, dict):
        text_lines.extend(
            [
                "",
                "固定验证震源的正演一致性（Wave CT 反演未使用这些事件）:",
                f"  验证事件: {validation_forward['matched_event_count']}，"
                f"走时: {validation_forward['observation_count']}",
                "  Reference TXT: "
                f"{validation_forward['reference_txt_model']['event_centered_rms_ms']:.3f} ms",
                "  Wave CT:       "
                f"{validation_forward['wave_ct_model']['event_centered_rms_ms']:.3f} ms",
                "  注意: 参考 TXT 的训练/验证划分未知，不能把该数值视为独立验证精度。",
            ]
        )
    (args.output_dir / "comparison_summary.txt").write_text(
        "\n".join(text_lines) + "\n", encoding="utf-8"
    )
    try:
        from wave_ct.deliverables import export_model_text_bundle

        export_model_text_bundle(
            model_path,
            args.result_dir,
            [float(z) for _, z, _ in loaded_reference],
            image_source_dir=args.result_dir,
        )
    except Exception as exc:
        # The comparison itself remains valid even if optional packaging fails.
        print(f"WARNING: final manifest refresh failed: {exc}")
    print("\n".join(text_lines))
    print(f"Comparison output: {args.output_dir}")


if __name__ == "__main__":
    main()
