"""Render workface CT report assets with optional mine-map evidence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
from matplotlib.lines import Line2D
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator
from scipy.ndimage import gaussian_filter

from wave_ct.cad import load_label_cache, load_segment_cache, prepare_cad_segments
from wave_ct.coordinate_audit import coordinate_alignment_audit

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
matplotlib.rcParams["axes.unicode_minus"] = False


def interpolate_z(volume: np.ndarray, zc: np.ndarray, target: float) -> np.ndarray:
    if target <= zc[0]:
        return volume[:, :, 0]
    if target >= zc[-1]:
        return volume[:, :, -1]
    upper = int(np.searchsorted(zc, target))
    lower = upper - 1
    weight = (target - zc[lower]) / (zc[upper] - zc[lower])
    return (1.0 - weight) * volume[:, :, lower] + weight * volume[:, :, upper]


def load_rays(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sources: list[tuple[float, float, float]] = []
    stations: list[tuple[float, float, float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sources.append((float(row["震源坐标-x"]), float(row["震源坐标-y"]), float(row["震源坐标-z"])))
            stations.append((float(row["台站坐标-x"]), float(row["台站坐标-y"]), float(row["台站坐标-z"])))
    source_rows = np.asarray(sources, dtype=float)
    station_rows = np.asarray(stations, dtype=float)
    return source_rows, station_rows, np.unique(source_rows, axis=0), np.unique(station_rows, axis=0)


def load_boundary(path: Path) -> tuple[list[str], np.ndarray]:
    names: list[str] = []
    points: list[tuple[float, float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"name", "x", "y"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError("工作面边界CSV必须包含 name,x,y 三列")
        for row in reader:
            names.append(str(row["name"]).strip())
            points.append((float(row["x"]), float(row["y"])))
    if len(points) < 3:
        raise ValueError("工作面边界至少需要3个顶点")
    boundary = np.asarray(points, dtype=float)
    if not np.allclose(boundary[0], boundary[-1]):
        boundary = np.vstack([boundary, boundary[0]])
    return names, boundary


def load_mapa_segments(path: Path | None) -> list[np.ndarray]:
    """Read line and polyline entities from the SOS ``Mapa.dat`` export."""
    if path is None or not path.exists():
        return []
    lines = path.read_text(encoding="gb18030", errors="ignore").splitlines()
    segments: list[np.ndarray] = []
    index = 0
    while index < len(lines):
        parts = lines[index].strip().split()
        if len(parts) >= 4 and any(part.startswith("$") for part in parts):
            try:
                point_count = int(parts[-1])
            except ValueError:
                point_count = 0
            if 2 <= point_count <= 2000 and index + point_count < len(lines):
                points: list[tuple[float, float]] = []
                for offset in range(1, point_count + 1):
                    coords = lines[index + offset].strip().split()
                    if len(coords) < 2:
                        points = []
                        break
                    try:
                        x, y = float(coords[0]), float(coords[1])
                    except ValueError:
                        points = []
                        break
                    if not (-100000.0 <= x <= 100000.0 and -100000.0 <= y <= 100000.0):
                        points = []
                        break
                    points.append((x, y))
                if len(points) >= 2:
                    segments.append(np.asarray(points, dtype=float))
                    index += point_count
        index += 1
    return segments


def draw_map(ax: plt.Axes, segments: list[np.ndarray], xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    for segment in segments:
        if (
            segment[:, 0].max() < xlim[0] or segment[:, 0].min() > xlim[1]
            or segment[:, 1].max() < ylim[0] or segment[:, 1].min() > ylim[1]
        ):
            continue
        ax.plot(
            segment[:, 0], segment[:, 1],
            color="#16232D", linewidth=0.50, alpha=0.72, zorder=4,
        )


def draw_boundary(
    ax: plt.Axes,
    boundary: np.ndarray,
    vertex_names: list[str],
    annotate: bool = False,
) -> None:
    ax.plot(boundary[:, 0], boundary[:, 1], color="black", linewidth=2.0, zorder=7)
    ax.plot(boundary[:, 0], boundary[:, 1], color="white", linestyle="--", linewidth=0.8, zorder=8)
    if annotate:
        for name, point in zip(vertex_names, boundary[:-1]):
            ax.annotate(
                name, xy=point, xytext=(3, 3), textcoords="offset points", fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.0},
                zorder=9,
            )


def draw_cad_labels(
    ax: plt.Axes,
    labels: list[tuple[float, float, str, float]],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    """Draw a de-duplicated set of the mine labels relevant to interpretation."""
    keywords = (
        "工作面", "运输巷", "回风巷", "切眼", "采空区",
        "斜巷", "联络巷", "泄水巷", "盘区",
    )
    width = max(xlim[1] - xlim[0], 1.0)
    height = max(ylim[1] - ylim[0], 1.0)
    candidates: list[tuple[int, float, float, str, float]] = []
    seen_text: set[str] = set()
    for x, y, raw_text, rotation in labels:
        text = " ".join(str(raw_text).replace("\\P", " ").split())
        if (
            not text
            or text in seen_text
            or not (xlim[0] <= x <= xlim[1] and ylim[0] <= y <= ylim[1])
        ):
            continue
        matches = [keyword for keyword in keywords if keyword in text]
        if not matches:
            continue
        priority = min(keywords.index(keyword) for keyword in matches)
        candidates.append((priority, x, y, text, rotation))
        seen_text.add(text)

    # Preserve the most useful labels and suppress labels whose anchors would
    # visibly collide in the compact student-style panels.
    selected: list[tuple[float, float, str, float]] = []
    for _, x, y, text, rotation in sorted(candidates, key=lambda item: (item[0], -len(item[3]))):
        if any(
            abs(x - px) / width < 0.105 and abs(y - py) / height < 0.055
            for px, py, _, _ in selected
        ):
            continue
        selected.append((x, y, text, rotation))
        if len(selected) >= 12:
            break

    for x, y, text, rotation in selected:
        ax.text(
            x,
            y,
            text,
            fontsize=6.5,
            color="#E11D1D" if "斜巷" in text else "black",
            rotation=rotation,
            rotation_mode="anchor",
            ha="left",
            va="center",
            zorder=18,
        )


def fine_field(
    slice_xy: np.ndarray,
    xc: np.ndarray,
    yc: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    *,
    smoothing_sigma: float = 0.0,
    cubic: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    field = np.asarray(slice_xy, dtype=np.float64)
    if smoothing_sigma > 0.0:
        field = gaussian_filter(field, sigma=smoothing_sigma, mode="nearest")
    # Quantitative/QC figures stay linear.  The optional cubic path is only
    # used by the explicitly labelled presentation rendering.
    order_x = min(3 if cubic else 1, max(1, yc.size - 1))
    order_y = min(3 if cubic else 1, max(1, xc.size - 1))
    spline = RectBivariateSpline(yc, xc, field.T, kx=order_x, ky=order_y)
    xf = np.linspace(xlim[0], xlim[1], 520)
    yf = np.linspace(ylim[0], ylim[1], 420)
    # ``RectBivariateSpline`` extrapolates outside the native model domain.
    # That is numerically convenient, but it turns a modest plot-window
    # extension into a large artificial high/low-velocity polygon after the
    # result is clipped to the colour bar.  Keep the requested display extent
    # while marking out-of-model samples as missing; the renderer then exposes
    # its neutral background colour there instead of inventing geology.
    field = np.asarray(spline(yf, xf), dtype=np.float64)
    outside = (
        (xf[None, :] < float(np.nanmin(xc)))
        | (xf[None, :] > float(np.nanmax(xc)))
        | (yf[:, None] < float(np.nanmin(yc)))
        | (yf[:, None] > float(np.nanmax(yc)))
    )
    field[outside] = np.nan
    return xf, yf, field


def coverage_aware_presentation_slice(
    velocity_slice: np.ndarray,
    density_slice: np.ndarray,
    background_velocity: float,
    reliable_coverage: float,
    smoothing_sigma: float,
    coverage_exponent: float = 1.0,
) -> np.ndarray:
    """Smooth supported anomalies and continuously fade them into the background.

    This is a presentation transform only.  Normalised convolution avoids
    treating unsampled background cells as velocity observations, while the
    support envelope prevents Surfer-like interpolation from implying that the
    whole map is equally resolved.
    """
    velocity = np.asarray(velocity_slice, dtype=np.float64)
    density = np.asarray(density_slice, dtype=np.float64)
    if velocity.shape != density.shape or velocity.ndim != 2:
        raise ValueError("velocity and density slices must be matching 2-D arrays")
    if not np.isfinite(coverage_exponent) or coverage_exponent <= 0.0:
        raise ValueError("coverage exponent must be finite and positive")
    # Treat ray support as a validity mask, not as a contrast multiplier.
    # A 10 m cell crossed by a single ray is less reliable than a multi-ray
    # cell, but it is still an observation.  The former weighted it by 0.5
    # when ``reliable_coverage`` was two, then faded it again downstream;
    # this removed the yellow/blue belts that users need to inspect.  Quality
    # is communicated through the separate coverage product/contour instead.
    weight = (density > 0.0).astype(np.float64)
    anomaly = velocity - float(background_velocity)
    sigma = max(float(smoothing_sigma), 0.0)
    if sigma <= 0.0:
        return float(background_velocity) + weight * anomaly

    # Normalised convolution keeps the inversion anomaly continuous wherever
    # rays provide support.  The broad support envelope then fades smoothly
    # into the neutral background without a hard boundary or visible clipping.
    numerator = gaussian_filter(anomaly * weight, sigma=sigma, mode="nearest")
    denominator = gaussian_filter(weight, sigma=sigma, mode="nearest")
    normalised_anomaly = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 1e-8,
    )
    support = gaussian_filter(weight, sigma=max(0.5, sigma * 1.35), mode="nearest")
    # A small support taper produces smooth Surfer-style boundaries while
    # preserving the complete anomaly amplitude on all sampled cells.
    fade = np.clip(support / 0.12, 0.0, 1.0)
    fade[support < 1e-3] = 0.0
    return float(background_velocity) + fade * normalised_anomaly


def vendor_enhanced_presentation_slice(
    velocity_slice: np.ndarray,
    background_velocity: float,
    smoothing_sigma: float,
    trend_angle_degrees: float,
    anomaly_limit: float,
) -> np.ndarray:
    """Build a local, smooth Surfer-style presentation field.

    The output may improve legibility, but it must not turn a one- or two-cell
    anomaly into a tens-of-metres red block.  Earlier versions propagated each
    sign by roughly 40--55 m along the roadway and then allowed a 24x gain.
    That looked dramatic but changed the apparent footprint substantially.
    Here we use only local Gaussian smoothing and a bounded robust gain.
    """
    velocity = np.asarray(velocity_slice, dtype=np.float64)
    background = float(background_velocity)
    anomaly = velocity - background
    # ``trend_angle_degrees`` is kept in the API for compatibility with stored
    # projects.  Deliberately do not use it to translate the field: an
    # orientation-dependent morphological dilation is not a faithful map
    # operation and was the cause of the oversized red belt.
    del trend_angle_degrees
    sigma = max(0.45, min(float(smoothing_sigma), 0.75))
    positive_envelope = gaussian_filter(np.maximum(anomaly, 0.0), sigma, mode="nearest")
    negative_envelope = gaussian_filter(np.maximum(-anomaly, 0.0), sigma, mode="nearest")

    # Use one fixed target for every elevation.  The delivered 728 Surfer
    # plates use the complete +/-0.30 anomaly range; capping this display
    # transform at 20 percent made the same data look washed out and removed
    # the red/yellow/blue hierarchy that engineers use for visual comparison.
    # This remains presentation-only: the saved quantitative velocity field is
    # never altered and the colourbar states the fixed anomaly range.
    target = background * min(0.30, float(anomaly_limit))

    def enhance(component: np.ndarray) -> np.ndarray:
        values = component[np.isfinite(component) & (component > 0.0)]
        if values.size == 0:
            return np.zeros_like(component)
        robust = float(np.percentile(values, 97.0))
        if robust <= 1.0e-9:
            return np.zeros_like(component)
        gain = float(np.clip(target / robust, 1.0, 6.0))
        return np.clip(component * gain, 0.0, target)

    displayed = background + enhance(positive_envelope) - enhance(negative_envelope)
    limit = background * float(anomaly_limit)
    return np.clip(displayed, background - limit, background + limit)


def render_presentation_slice(
    velocity: np.ndarray,
    ray_density: np.ndarray,
    xc: np.ndarray,
    yc: np.ndarray,
    zc: np.ndarray,
    target_z: float,
    map_segments: list[np.ndarray],
    output_path: Path,
    vmin: float,
    vmax: float,
    center_velocity: float,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    period: str,
    smoothing_sigma: float,
    boundary: np.ndarray | None = None,
    reliable_coverage: float = 1.0,
    coverage_exponent: float = 1.0,
) -> None:
    """Render a Surfer-like smooth map without changing the saved model."""
    slice_xy = coverage_aware_presentation_slice(
        interpolate_z(velocity, zc, target_z),
        interpolate_z(ray_density, zc, target_z),
        center_velocity,
        reliable_coverage,
        smoothing_sigma,
        coverage_exponent,
    )
    xf, yf, field = fine_field(
        slice_xy,
        xc,
        yc,
        xlim,
        ylim,
        smoothing_sigma=0.0,
        cubic=True,
    )
    xmesh, ymesh = np.meshgrid(xf, yf)
    fig, ax = plt.subplots(figsize=(13.0, 6.2), dpi=180, facecolor="white")
    if vmin < center_velocity < vmax:
        levels = np.unique(np.concatenate([
            np.linspace(vmin, center_velocity, 61),
            np.linspace(center_velocity, vmax, 61),
        ])) / 1000.0
        color_norm = TwoSlopeNorm(
            vmin=vmin / 1000.0,
            vcenter=center_velocity / 1000.0,
            vmax=vmax / 1000.0,
        )
    else:
        levels = np.linspace(vmin / 1000.0, vmax / 1000.0, 120)
        color_norm = None
    contour = ax.contourf(
        xmesh,
        ymesh,
        np.clip(field, vmin, vmax) / 1000.0,
        levels=levels,
        cmap="jet",
        norm=color_norm,
        extend="both",
        antialiased=True,
        zorder=1,
    )
    # Retain the familiar Surfer-style colour field outside the high-confidence
    # area, but mark the reliable-coverage boundary so extrapolated display is
    # not mistaken for resolved structure.  This never changes the model.
    density_xy = interpolate_z(ray_density, zc, target_z)
    _, _, density_field = fine_field(
        density_xy, xc, yc, xlim, ylim, smoothing_sigma=0.0, cubic=False
    )
    if (
        np.isfinite(density_field).any()
        and np.nanmin(density_field) <= reliable_coverage <= np.nanmax(density_field)
    ):
        ax.contour(
            xmesh, ymesh, density_field,
            levels=[reliable_coverage], colors="#1F2937",
            linestyles="--", linewidths=0.8, zorder=6,
        )
    draw_map(ax, map_segments, xlim, ylim)
    if not map_segments and boundary is not None:
        draw_boundary(ax, boundary, [], annotate=False)
    colorbar = fig.colorbar(contour, ax=ax, fraction=0.035, pad=0.025)
    colorbar.set_label("P波速度 (km/s)", fontsize=11)
    if vmin < center_velocity < vmax:
        colorbar.set_ticks(np.asarray([
            vmin,
            0.5 * (vmin + center_velocity),
            center_velocity,
            0.5 * (center_velocity + vmax),
            vmax,
        ]) / 1000.0)
    else:
        colorbar.set_ticks(np.linspace(vmin, vmax, 6) / 1000.0)
    ax.set_title(
        f"标高 {target_z:.3f}m 工作面煤层CT反演结果（平滑展示版）  {period}".strip(),
        fontsize=14,
    )
    ax.text(
        0.01,
        0.015,
        f"覆盖加权连续平滑 σ={smoothing_sigma:g} 网格；定量模型与检验指标未被修改",
        transform=ax.transAxes,
        fontsize=8.2,
        color="#111827",
        bbox={"facecolor": "white", "edgecolor": "#94A3B8", "alpha": 0.78, "pad": 2.0},
        zorder=9,
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def render_presentation_anomaly_slice(
    velocity: np.ndarray,
    ray_density: np.ndarray,
    xc: np.ndarray,
    yc: np.ndarray,
    zc: np.ndarray,
    target_z: float,
    background_velocity: float,
    map_segments: list[np.ndarray],
    boundary: np.ndarray,
    output_path: Path,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    period: str,
    smoothing_sigma: float,
    reliable_coverage: float,
    coverage_exponent: float = 1.0,
    anomaly_limit: float = 0.30,
) -> None:
    """Render a Surfer-like anomaly map while retaining a truthful label."""
    slice_xy = coverage_aware_presentation_slice(
        interpolate_z(velocity, zc, target_z),
        interpolate_z(ray_density, zc, target_z),
        background_velocity,
        reliable_coverage,
        smoothing_sigma,
        coverage_exponent,
    )
    anomaly = (slice_xy - background_velocity) / background_velocity * 100.0
    xf, yf, field = fine_field(
        anomaly,
        xc,
        yc,
        xlim,
        ylim,
        smoothing_sigma=0.0,
        cubic=True,
    )
    limit = float(anomaly_limit * 100.0)
    if not np.isfinite(limit) or limit <= 0.0:
        raise ValueError("anomaly limit must be finite and positive")
    levels = np.unique(np.concatenate([
        np.linspace(-limit, 0.0, 61),
        np.linspace(0.0, limit, 61),
    ]))
    fig, ax = plt.subplots(figsize=(13.0, 6.2), dpi=180, facecolor="white")
    contour = ax.contourf(
        xf,
        yf,
        np.clip(field, -limit, limit),
        levels=levels,
        cmap="seismic",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        extend="both",
        antialiased=True,
        zorder=1,
    )
    draw_map(ax, map_segments, xlim, ylim)
    if not map_segments:
        draw_boundary(ax, boundary, [], annotate=False)
    colorbar = fig.colorbar(contour, ax=ax, fraction=0.035, pad=0.025)
    colorbar.set_label("波速异常 An (%)", fontsize=11)
    colorbar.set_ticks(np.linspace(-limit, limit, 5))
    ax.set_title(
        f"标高 {target_z:.3f}m 工作面波速异常（平滑展示版）  {period}".strip(),
        fontsize=14,
    )
    ax.text(
        0.01,
        0.015,
        f"An=(V−V0)/V0×100%，V0={background_velocity / 1000.0:.3f} km/s；"
        f"覆盖加权连续平滑 σ={smoothing_sigma:g} 网格",
        transform=ax.transAxes,
        fontsize=8.2,
        color="#111827",
        bbox={"facecolor": "white", "edgecolor": "#94A3B8", "alpha": 0.80, "pad": 2.0},
        zorder=9,
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def render_slice(
    velocity: np.ndarray,
    ray_density: np.ndarray,
    xc: np.ndarray,
    yc: np.ndarray,
    zc: np.ndarray,
    target_z: float,
    map_segments: list[np.ndarray],
    output_path: Path,
    vmin: float,
    vmax: float,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    period: str,
    reliable_coverage: float,
    boundary: np.ndarray | None = None,
) -> None:
    slice_xy = interpolate_z(velocity, zc, target_z)
    density_xy = interpolate_z(ray_density, zc, target_z)
    xf, yf, field = fine_field(slice_xy, xc, yc, xlim, ylim)
    _, _, density_field = fine_field(density_xy, xc, yc, xlim, ylim)
    density_field = np.maximum(density_field, 0.0)
    xmesh, ymesh = np.meshgrid(xf, yf)

    fig, ax = plt.subplots(figsize=(13.0, 6.2), dpi=180, facecolor="white")
    levels = np.linspace(vmin / 1000.0, vmax / 1000.0, 90)
    contour = ax.contourf(
        xmesh, ymesh, np.clip(field, vmin, vmax) / 1000.0,
        levels=levels, cmap="jet", extend="both", zorder=1,
    )

    if np.nanmin(density_field) <= reliable_coverage <= np.nanmax(density_field):
        ax.contour(
            xmesh,
            ymesh,
            density_field,
            levels=[reliable_coverage],
            colors="#111827",
            linestyles="--",
            linewidths=1.0,
            zorder=3,
        )

    draw_map(ax, map_segments, xlim, ylim)
    if not map_segments and boundary is not None:
        draw_boundary(ax, boundary, [], annotate=False)

    colorbar = fig.colorbar(contour, ax=ax, fraction=0.035, pad=0.025)
    colorbar.set_label("P波速度 (km/s)", fontsize=11)
    colorbar.set_ticks(np.linspace(vmin, vmax, 5) / 1000.0)
    ax.set_title(f"标高 {target_z:.0f}m 工作面煤层CT反演结果  {period}".strip(), fontsize=14)
    ax.text(
        0.01,
        0.015,
        f"虚线内：可靠覆盖（≥{reliable_coverage:g}条射线/单元）；虚线外仅作趋势显示",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#111827",
        bbox={"facecolor": "white", "edgecolor": "#94A3B8", "alpha": 0.82, "pad": 2.0},
        zorder=9,
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def render_coverage_reliability(
    ray_density: np.ndarray,
    xc: np.ndarray,
    yc: np.ndarray,
    zc: np.ndarray,
    target_z: float,
    map_segments: list[np.ndarray],
    boundary: np.ndarray,
    output_path: Path,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    reliable_coverage: float,
) -> None:
    """Show ray density and the exact coverage reliability boundary on the mine map."""
    density_xy = interpolate_z(ray_density, zc, target_z)
    xf, yf, field = fine_field(density_xy, xc, yc, xlim, ylim)
    field = np.maximum(field, 0.0)
    xmesh, ymesh = np.meshgrid(xf, yf)
    upper = max(1.0, float(np.percentile(field, 99.0)))
    fig, ax = plt.subplots(figsize=(13.0, 6.2), dpi=180, facecolor="white")
    contour = ax.contourf(
        xmesh,
        ymesh,
        np.clip(field, 0.0, upper),
        levels=np.linspace(0.0, upper, 70),
        cmap="viridis",
        extend="max",
        zorder=1,
    )
    if np.nanmin(field) <= reliable_coverage <= np.nanmax(field):
        ax.contour(
            xmesh,
            ymesh,
            field,
            levels=[reliable_coverage],
            colors="#EF4444",
            linestyles="-",
            linewidths=1.4,
            zorder=5,
        )
    draw_map(ax, map_segments, xlim, ylim)
    if not map_segments:
        draw_boundary(ax, boundary, [], annotate=False)
    colorbar = fig.colorbar(contour, ax=ax, fraction=0.035, pad=0.025)
    colorbar.set_label("穿过网格的射线条数", fontsize=11)
    ax.set_title(f"标高 {target_z:.3f}m 射线覆盖与可靠区", fontsize=14)
    ax.text(
        0.01,
        0.015,
        f"红线内达到可靠覆盖阈值（≥{reliable_coverage:g}条射线/单元）",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#111827",
        bbox={"facecolor": "white", "edgecolor": "#94A3B8", "alpha": 0.82, "pad": 2.0},
        zorder=9,
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _draw_student_scale_and_north(
    ax: plt.Axes,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    """Add the compact scale bar and north arrow used by the reference figures."""
    width = xlim[1] - xlim[0]
    height = ylim[1] - ylim[0]
    scale_length = min(100.0, width * 0.18)
    x0 = xlim[0] + 0.025 * width
    y0 = ylim[0] + 0.055 * height
    cap = 0.012 * height
    ax.plot([x0, x0 + scale_length], [y0, y0], color="black", linewidth=1.2, zorder=20)
    ax.plot([x0, x0], [y0 - cap, y0 + cap], color="black", linewidth=1.0, zorder=20)
    ax.plot(
        [x0 + scale_length, x0 + scale_length],
        [y0 - cap, y0 + cap],
        color="black",
        linewidth=1.0,
        zorder=20,
    )
    ax.text(
        x0,
        y0 + 0.018 * height,
        f"0    {scale_length:g}m",
        fontsize=7.5,
        ha="left",
        va="bottom",
        color="black",
        zorder=20,
    )

    north_x = xlim[1] - 0.035 * width
    north_y = ylim[1] - 0.055 * height
    ax.text(
        north_x,
        north_y + 0.018 * height,
        "北",
        fontsize=9,
        ha="center",
        va="bottom",
        color="black",
        bbox={"facecolor": "white", "edgecolor": "#64748B", "linewidth": 0.5, "pad": 1.5},
        zorder=21,
    )
    ax.annotate(
        "",
        xy=(north_x, north_y + 0.012 * height),
        xytext=(north_x, north_y - 0.055 * height),
        arrowprops={"arrowstyle": "-|>", "color": "black", "linewidth": 0.8},
        zorder=21,
    )


def _rotate_xy(
    x: np.ndarray | float,
    y: np.ndarray | float,
    center: tuple[float, float],
    angle_degrees: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate map coordinates for a presentation view only."""
    angle = np.deg2rad(float(angle_degrees))
    cosine, sine = np.cos(angle), np.sin(angle)
    xx = np.asarray(x, dtype=float) - center[0]
    yy = np.asarray(y, dtype=float) - center[1]
    return (
        center[0] + cosine * xx - sine * yy,
        center[1] + sine * xx + cosine * yy,
    )


def _rotated_limits(
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    center: tuple[float, float],
    angle_degrees: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    corners = np.asarray([
        [xlim[0], ylim[0]], [xlim[0], ylim[1]],
        [xlim[1], ylim[0]], [xlim[1], ylim[1]],
    ])
    x_rot, y_rot = _rotate_xy(corners[:, 0], corners[:, 1], center, angle_degrees)
    return (float(x_rot.min()), float(x_rot.max())), (float(y_rot.min()), float(y_rot.max()))


def _dominant_map_rotation(map_segments: list[np.ndarray], boundary: np.ndarray) -> tuple[tuple[float, float], float]:
    """Align the dominant undirected CAD trend with the presentation X axis."""
    candidates = [np.asarray(segment, dtype=float) for segment in map_segments if len(segment) >= 2]
    if candidates:
        points = np.vstack(candidates)
        vectors = np.vstack([np.diff(segment, axis=0) for segment in candidates])
    else:
        points = np.asarray(boundary, dtype=float)
        vectors = np.diff(points, axis=0)
    lengths = np.hypot(vectors[:, 0], vectors[:, 1])
    keep = lengths >= max(5.0, float(np.percentile(lengths, 55.0)) if lengths.size else 5.0)
    vectors, lengths = vectors[keep], lengths[keep]
    if not lengths.size:
        return (float(np.mean(points[:, 0])), float(np.mean(points[:, 1]))), 0.0
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    # Double-angle averaging treats theta and theta+180 degrees as the same roadway trend.
    direction = 0.5 * np.arctan2(
        float(np.sum(lengths * np.sin(2.0 * angles))),
        float(np.sum(lengths * np.cos(2.0 * angles))),
    )
    return (float(np.mean(points[:, 0])), float(np.mean(points[:, 1]))), -float(np.rad2deg(direction))


def _draw_student_panel(
    fig: plt.Figure,
    ax: plt.Axes,
    field_xy: np.ndarray,
    xc: np.ndarray,
    yc: np.ndarray,
    map_segments: list[np.ndarray],
    map_labels: list[tuple[float, float, str, float]],
    boundary: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    vmin: float,
    vmax: float,
    center: float,
    ticks: np.ndarray,
    *,
    show_workface: bool = True,
    show_coordinates: bool = False,
    colorbar_label: str = "",
    reliability_xy: np.ndarray | None = None,
    reliable_coverage: float | None = None,
    sources: np.ndarray | None = None,
    stations: np.ndarray | None = None,
    annotate_extrema: bool = False,
    presentation_rotation: tuple[tuple[float, float], float] | None = None,
    centered_norm: bool = True,
    crop_to_basemap: bool = False,
) -> None:
    xf, yf, field = fine_field(field_xy, xc, yc, xlim, ylim, cubic=True)
    panel_xlim, panel_ylim = xlim, ylim
    contour_x, contour_y = np.meshgrid(xf, yf)
    panel_segments = map_segments
    panel_labels = map_labels
    panel_boundary = boundary
    if presentation_rotation is not None:
        center_point, angle = presentation_rotation
        contour_x, contour_y = _rotate_xy(contour_x, contour_y, center_point, angle)
        panel_xlim, panel_ylim = _rotated_limits(xlim, ylim, center_point, angle)
        panel_segments = [
            np.column_stack(_rotate_xy(segment[:, 0], segment[:, 1], center_point, angle))
            for segment in map_segments
        ]
        panel_labels = [
            (*_rotate_xy(x, y, center_point, angle), text, rotation + angle)
            for x, y, text, rotation in map_labels
        ]
        panel_boundary = np.column_stack(
            _rotate_xy(boundary[:, 0], boundary[:, 1], center_point, angle)
        )
        # A rotated rectangular model window has a large diamond-shaped empty
        # margin.  Frame the actual workface CAD in report products instead;
        # the numerical model itself stays in its native coordinate system.
        if crop_to_basemap and panel_segments:
            # CAD files often contain long construction/grid lines far outside
            # the inversion window.  They must not determine the report crop.
            original_points = np.vstack(map_segments)
            inside = (
                (original_points[:, 0] >= xlim[0])
                & (original_points[:, 0] <= xlim[1])
                & (original_points[:, 1] >= ylim[0])
                & (original_points[:, 1] <= ylim[1])
            )
            cad_points = np.vstack(panel_segments)
            if np.any(inside):
                cad_points = np.column_stack(
                    _rotate_xy(
                        original_points[inside, 0], original_points[inside, 1],
                        center_point, angle,
                    )
                )
            cad_xmin, cad_xmax = np.nanmin(cad_points[:, 0]), np.nanmax(cad_points[:, 0])
            cad_ymin, cad_ymax = np.nanmin(cad_points[:, 1]), np.nanmax(cad_points[:, 1])
            pad_x = max(1.0, 0.025 * (cad_xmax - cad_xmin))
            pad_y = max(1.0, 0.050 * (cad_ymax - cad_ymin))
            panel_xlim = (float(cad_xmin - pad_x), float(cad_xmax + pad_x))
            panel_ylim = (float(cad_ymin - pad_y), float(cad_ymax + pad_y))
    levels = np.linspace(vmin, vmax, 128)
    norm = (
        TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)
        if centered_norm and vmin < center < vmax else None
    )
    # Rotating a native rectangular grid produces triangular corners outside
    # its footprint.  Use the neutral model colour there so the CAD report
    # reads as one continuous workface panel, without inventing values in the
    # quantitative grid.
    colour_position = (
        norm(center) if norm is not None else (center - vmin) / max(vmax - vmin, 1.0e-12)
    )
    neutral_colour = plt.get_cmap("jet")(float(np.clip(colour_position, 0.0, 1.0)))
    ax.set_facecolor(neutral_colour)
    ax.fill(
        [panel_xlim[0], panel_xlim[1], panel_xlim[1], panel_xlim[0]],
        [panel_ylim[0], panel_ylim[0], panel_ylim[1], panel_ylim[1]],
        facecolor=neutral_colour,
        edgecolor="none",
        zorder=0,
    )
    contour = ax.contourf(
        contour_x,
        contour_y,
        np.clip(field, vmin, vmax),
        levels=levels,
        cmap="jet",
        norm=norm,
        extend="both",
        antialiased=False,
        zorder=1,
    )
    if show_workface:
        draw_map(ax, panel_segments, panel_xlim, panel_ylim)
        if not panel_segments and panel_boundary.size:
            draw_boundary(ax, panel_boundary, [], annotate=False)
        draw_cad_labels(ax, panel_labels, panel_xlim, panel_ylim)
    if show_workface:
        _draw_student_scale_and_north(ax, panel_xlim, panel_ylim)
        if stations is not None and np.asarray(stations).size:
            station_array = np.asarray(stations, dtype=float)
            if presentation_rotation is not None:
                station_array = station_array.copy()
                station_array[:, 0], station_array[:, 1] = _rotate_xy(
                    station_array[:, 0], station_array[:, 1], *presentation_rotation
                )
            ax.scatter(
                station_array[:, 0], station_array[:, 1], marker="^", s=22,
                facecolor="white", edgecolor="black", linewidth=0.7,
                zorder=24, label="台站",
            )
            for index, (x_value, y_value, *_rest) in enumerate(station_array, 1):
                ax.annotate(
                    f"S{index}", (x_value, y_value), xytext=(3, 3),
                    textcoords="offset points", fontsize=5.5, color="black",
                    zorder=25,
                )
        if sources is not None and np.asarray(sources).size:
            source_array = np.asarray(sources, dtype=float)
            if presentation_rotation is not None:
                source_array = source_array.copy()
                source_array[:, 0], source_array[:, 1] = _rotate_xy(
                    source_array[:, 0], source_array[:, 1], *presentation_rotation
                )
            # Hundreds of event labels make a mine map unreadable.  Plot every
            # event, but label only a deterministic, spatially distributed subset.
            ax.scatter(
                source_array[:, 0], source_array[:, 1], marker="o", s=7,
                facecolor="#ff2d20", edgecolor="white", linewidth=0.25,
                alpha=0.72, zorder=23, label="震源",
            )
            stride = max(1, int(np.ceil(source_array.shape[0] / 12.0)))
            for index in range(0, source_array.shape[0], stride):
                x_value, y_value = source_array[index, :2]
                ax.annotate(
                    f"E{index + 1}", (x_value, y_value), xytext=(2, -5),
                    textcoords="offset points", fontsize=4.8, color="#7f1d1d",
                    zorder=25,
                )
        if annotate_extrema and np.isfinite(field).any():
            finite = np.where(np.isfinite(field), field, np.nan)
            # Extrema outside reliable ray coverage are commonly edge artefacts.
            # Keep the complete colour field visible, but only label extrema that
            # are supported by the configured engineering coverage threshold.
            if (
                reliability_xy is not None
                and reliable_coverage is not None
                and reliable_coverage > 0
            ):
                _, _, fine_reliability = fine_field(
                    np.asarray(reliability_xy, dtype=float),
                    xc,
                    yc,
                    xlim,
                    ylim,
                    cubic=False,
                )
                reliable_mask = fine_reliability >= float(reliable_coverage)
                if np.any(reliable_mask & np.isfinite(finite)):
                    finite = np.where(reliable_mask, finite, np.nan)
            for label, flat_index, color in (
                ("高值中心", int(np.nanargmax(finite)), "#8b0000"),
                ("低值中心", int(np.nanargmin(finite)), "#003f8c"),
            ):
                row, column = np.unravel_index(flat_index, finite.shape)
                peak_x, peak_y = contour_x[row, column], contour_y[row, column]
                ax.annotate(
                    f"{label}\n{finite[row, column]:.3f}",
                    (peak_x, peak_y), xytext=(7, 7),
                    textcoords="offset points", fontsize=6.2, color=color,
                    bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.78, "edgecolor": color},
                    arrowprops={"arrowstyle": "->", "color": color, "linewidth": 0.7},
                    zorder=30,
                )
    ax.set_xlim(*panel_xlim)
    ax.set_ylim(*panel_ylim)
    ax.set_aspect("equal")
    if show_coordinates:
        ax.set_axis_on()
        ax.set_xlabel("X 坐标 (m)", fontsize=9)
        ax.set_ylabel("Y 坐标 (m)", fontsize=9)
        ax.ticklabel_format(style="plain", axis="both", useOffset=False)
        ax.tick_params(labelsize=7, direction="out", length=3)
        ax.grid(False)
    else:
        ax.set_axis_off()
    colorbar = fig.colorbar(contour, ax=ax, fraction=0.026, pad=0.006)
    colorbar.set_ticks(ticks)
    colorbar.ax.tick_params(labelsize=7, length=2.5, pad=2)
    if colorbar_label:
        colorbar.set_label(colorbar_label, fontsize=8)


def render_vendor_style_bundle(
    velocity: np.ndarray,
    ray_density: np.ndarray,
    xc: np.ndarray,
    yc: np.ndarray,
    zc: np.ndarray,
    targets: list[float],
    background_velocity: float,
    reliable_coverage: float,
    smoothing_sigma: float,
    map_segments: list[np.ndarray],
    map_labels: list[tuple[float, float, str, float]],
    boundary: np.ndarray,
    output_dir: Path,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    velocity_vmin: float,
    velocity_vmax: float,
    sources: np.ndarray | None = None,
    stations: np.ndarray | None = None,
    coverage_exponent: float = 1.0,
    anomaly_limit: float = 0.30,
) -> None:
    """Render the compact Surfer layout of the supplied vendor figures.

    The transform is presentation-only: every panel is derived from ``velocity``
    through :func:`coverage_aware_presentation_slice`; the saved quantitative
    model is never replaced.  Individual vendor panels intentionally omit the
    source/station scatter because the supplied result template uses the CAD
    basemap as its only overlay.  Source geometry remains available in the
    dedicated coverage products.
    """
    if not targets:
        raise ValueError(
            "No slice elevations were supplied; configure slice_z or use model zc layers."
        )
    # Keep the quantitative grid and CAD in their native common coordinate
    # system.  Rotating only a rendered grid produced a diamond-shaped
    # background for oblique projects, which is visually and spatially wrong.
    # A map-specific project may supply a pre-rotated CAD, but this renderer
    # must never manufacture that transformation at report time.
    #
    # The vendor-style plate is a presentation product, not the coverage QC
    # product.  Start from the quantitative slice, then apply a display-only
    # support envelope before Gaussian dilation.  This retains the broad
    # red/blue structures seen in the supplied plates while preventing cells
    # with no rays from becoming a false high-velocity background.
    # Orient the display-only dilation along the dominant workface trend.  The
    # same rigid transform is later applied to the field and CAD overlay.
    presentation_rotation = (
        _dominant_map_rotation(map_segments, boundary)
        if map_segments else None
    )
    trend_angle_degrees = (
        -float(presentation_rotation[1])
        if presentation_rotation is not None else 0.0
    )

    fields: list[tuple[float, np.ndarray, np.ndarray, np.ndarray]] = []
    for target_z in targets:
        density_slice = interpolate_z(ray_density, zc, target_z)
        quantitative_slice = interpolate_z(velocity, zc, target_z)
        # Keep the vendor-like smoothing/high contrast, but first anchor the
        # field to the measured ray support.  The inversion deliberately
        # stores a neutral prior in cells without rays; interpolating and
        # dilating those weak cells across the whole panel creates the large
        # false red background that users were seeing.  This is a rendering
        # transform only -- ``velocity`` remains the quantitative model.
        supported_slice = coverage_aware_presentation_slice(
            quantitative_slice,
            density_slice,
            background_velocity,
            reliable_coverage,
            smoothing_sigma,
            coverage_exponent,
        )
        presentation = vendor_enhanced_presentation_slice(
            supported_slice,
            background_velocity=background_velocity,
            smoothing_sigma=smoothing_sigma,
            trend_angle_degrees=trend_angle_degrees,
            anomaly_limit=anomaly_limit,
        )
        # ``coverage_aware_presentation_slice`` has already set cells without
        # any ray support to the neutral background and supplied a smooth
        # support taper.  A second, asymmetric coverage fade here used to
        # suppress the positive (yellow/red) and negative (cyan/blue) belts
        # a second time.  It made the maps look deceptively sparse without
        # improving the numerical inversion, so do not apply it.
        anomaly = (presentation - background_velocity) / background_velocity
        fields.append((target_z, presentation / 1000.0, anomaly, density_slice))

    # Engineering interpretation uses one fixed relative-anomaly scale for all
    # elevations. This keeps layer-to-layer colour meaning identical and avoids
    # promoting a weak layer merely because its own colour bar expands.
    if not np.isfinite(anomaly_limit) or anomaly_limit <= 0.0:
        raise ValueError("anomaly limit must be finite and positive")
    # One symmetric relative-anomaly scale is shared by all elevations.  It
    # prevents weak layers from being visually amplified by per-layer
    # percentile stretching and keeps the three maps comparable.
    vendor_anomaly_low, vendor_anomaly_high = -float(anomaly_limit), float(anomaly_limit)

    velocity_low = velocity_vmin / 1000.0
    velocity_high = velocity_vmax / 1000.0
    velocity_center = background_velocity / 1000.0
    velocity_ticks = np.linspace(velocity_low, velocity_high, 7)

    # All elevations share exactly the same absolute-velocity colour scale.
    # Keeping the range centred on v0 makes the enhanced anomaly footprint
    # legible while preserving identical colour meaning between layers.
    display_velocity_low = max(
        velocity_low, velocity_center * (1.0 - anomaly_limit)
    )
    display_velocity_high = min(
        velocity_high, velocity_center * (1.0 + anomaly_limit)
    )
    if display_velocity_high <= display_velocity_low:
        display_velocity_low, display_velocity_high = velocity_low, velocity_high
    display_anomaly_low, display_anomaly_high = vendor_anomaly_low, vendor_anomaly_high
    for target_z, velocity_field, anomaly, density_slice in fields:
        for kind, field, low, high, center, ticks in (
            (
                "velocity",
                velocity_field,
                display_velocity_low,
                display_velocity_high,
                velocity_center,
                np.linspace(display_velocity_low, display_velocity_high, 7),
            ),
            (
                "anomaly",
                anomaly,
                display_anomaly_low,
                display_anomaly_high,
                0.0,
                np.linspace(display_anomaly_low, display_anomaly_high, 7),
            ),
        ):
            fig, ax = plt.subplots(figsize=(11.2, 5.0), dpi=180, facecolor="white")
            _draw_student_panel(
                fig, ax, field, xc, yc, map_segments, map_labels, boundary, xlim, ylim,
                low, high, center, ticks,
                reliability_xy=density_slice,
                reliable_coverage=reliable_coverage,
                sources=None,
                stations=None,
                centered_norm=(kind == "anomaly"),
                presentation_rotation=presentation_rotation,
                crop_to_basemap=bool(presentation_rotation is not None),
            )
            fig.subplots_adjust(left=0.004, right=0.965, top=0.995, bottom=0.005)
            fig.savefig(
                output_dir / f"vendor_style_{kind}_z{target_z:.3f}.png",
                dpi=220,
                bbox_inches="tight",
                pad_inches=0.02,
            )
            plt.close(fig)

        # Three final products per elevation: coordinate velocity slice,
        # workface anomaly, and workface absolute velocity.
        layer_velocity_low = max(
            velocity_low, velocity_center * (1.0 - anomaly_limit)
        )
        layer_velocity_high = min(
            velocity_high, velocity_center * (1.0 + anomaly_limit)
        )
        if layer_velocity_high <= layer_velocity_low:
            layer_velocity_low, layer_velocity_high = velocity_low, velocity_high
        layer_velocity_ticks = np.linspace(
            layer_velocity_low, layer_velocity_high, 7
        )
        products = (
            (
                "key_velocity_slice",
                velocity_field,
                layer_velocity_low,
                layer_velocity_high,
                velocity_center,
                layer_velocity_ticks,
                False,
                True,
                f"标高 {target_z:.3f} m P波速度模型切片",
                "P波速度 (km/s)",
            ),
            (
                "key_workface_anomaly",
                anomaly,
                vendor_anomaly_low,
                vendor_anomaly_high,
                0.0,
                np.linspace(vendor_anomaly_low, vendor_anomaly_high, 7),
                True,
                False,
                f"标高 {target_z:.3f} m 工作面P波速度异常反演结果",
                "速度异常系数 An",
            ),
            (
                "key_workface_velocity",
                velocity_field,
                layer_velocity_low,
                layer_velocity_high,
                velocity_center,
                layer_velocity_ticks,
                True,
                False,
                f"标高 {target_z:.3f} m 工作面煤层P波速度反演结果",
                "P波速度 (km/s)",
            ),
        )
        for (
            name, final_field, final_low, final_high, final_center, final_ticks,
            show_workface, show_coordinates, final_title, final_colorbar_label,
        ) in products:
            fig, ax = plt.subplots(figsize=(11.2, 5.0), dpi=180, facecolor="white")
            _draw_student_panel(
                fig, ax, final_field, xc, yc, map_segments, map_labels, boundary,
                xlim, ylim, final_low, final_high, final_center, final_ticks,
                show_workface=show_workface,
                show_coordinates=show_coordinates,
                colorbar_label=final_colorbar_label,
                reliability_xy=density_slice if show_workface else None,
                reliable_coverage=reliable_coverage if show_workface else None,
                sources=sources if show_workface else None,
                stations=stations if show_workface else None,
                annotate_extrema=show_workface,
                centered_norm=(name == "key_workface_anomaly"),
                presentation_rotation=presentation_rotation,
                crop_to_basemap=bool(presentation_rotation is not None),
            )
            ax.set_title(final_title, fontsize=11, pad=4)
            fig.subplots_adjust(left=0.004, right=0.965, top=0.95, bottom=0.005)
            fig.savefig(
                output_dir / f"{name}_z{target_z:.3f}.png",
                dpi=220,
                bbox_inches="tight",
                pad_inches=0.02,
            )
            if name == "key_velocity_slice":
                fig.savefig(
                    output_dir / f"velocity_slice_z{int(target_z)}.png",
                    dpi=220,
                    bbox_inches="tight",
                    pad_inches=0.02,
                )
                fig.savefig(
                    output_dir / f"velocity_z{int(target_z)}.png",
                    dpi=220,
                    bbox_inches="tight",
                    pad_inches=0.02,
                )
            plt.close(fig)

    # The supplied template is a three-elevation by two-product plate.  A
    # project may request additional diagnostic elevations, but silently
    # growing this plate to nine or more rows makes it unusable and previously
    # left a large blank centre in the exported image.  Keep the formal plate
    # to the first three configured engineering elevations; individual files
    # above are still emitted for every requested target.
    overview_fields = fields[:3]
    fig, axes = plt.subplots(
        len(overview_fields),
        2,
        # Two equal-aspect mine maps per row need a near-square canvas.  The
        # former 15.6-inch width left a misleading blank band between columns.
        figsize=(10.4, 10.3),
        dpi=180,
        facecolor="white",
        squeeze=False,
    )
    for row, (_, velocity_field, anomaly, density_slice) in enumerate(overview_fields):
        _draw_student_panel(
            fig,
            axes[row, 0],
            anomaly,
            xc,
            yc,
            map_segments,
            map_labels,
            boundary,
            xlim,
            ylim,
            vendor_anomaly_low,
            vendor_anomaly_high,
            0.0,
            np.linspace(vendor_anomaly_low, vendor_anomaly_high, 7),
            reliability_xy=density_slice,
            reliable_coverage=reliable_coverage,
            presentation_rotation=presentation_rotation,
            crop_to_basemap=bool(presentation_rotation is not None),
        )
        _draw_student_panel(
            fig,
            axes[row, 1],
            velocity_field,
            xc,
            yc,
            map_segments,
            map_labels,
            boundary,
            xlim,
            ylim,
            display_velocity_low,
            display_velocity_high,
            velocity_center,
            np.linspace(display_velocity_low, display_velocity_high, 7),
            reliability_xy=density_slice,
            reliable_coverage=reliable_coverage,
            centered_norm=False,
            presentation_rotation=presentation_rotation,
            crop_to_basemap=bool(presentation_rotation is not None),
        )
    fig.subplots_adjust(left=0.004, right=0.985, top=0.996, bottom=0.004, wspace=0.05, hspace=0.11)
    fig.savefig(
        output_dir / "vendor_style_3x2_overview.png",
        dpi=220,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(fig)


# Backward-compatible import for callers written while this renderer was
# prototyped under the ambiguous "student" name.  New artifacts and manifests
# use ``vendor_style``.
render_student_style_bundle = render_vendor_style_bundle


def render_source_distribution(
    sources: np.ndarray,
    stations: np.ndarray,
    boundary: np.ndarray,
    names: list[str],
    segments: list[np.ndarray],
    output_path: Path,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    period: str,
) -> None:
    fig, ax = plt.subplots(figsize=(12.0, 5.4), dpi=180, facecolor="white")
    draw_map(ax, segments, xlim, ylim)
    if not segments:
        draw_boundary(ax, boundary, names)
    source_mask = (
        (sources[:, 0] >= xlim[0]) & (sources[:, 0] <= xlim[1])
        & (sources[:, 1] >= ylim[0]) & (sources[:, 1] <= ylim[1])
    )
    station_mask = (
        (stations[:, 0] >= xlim[0]) & (stations[:, 0] <= xlim[1])
        & (stations[:, 1] >= ylim[0]) & (stations[:, 1] <= ylim[1])
    )
    ax.scatter(sources[source_mask, 0], sources[source_mask, 1], s=16, c="#1239D8", alpha=0.72,
               linewidths=0, label="反演震源", zorder=6)
    ax.scatter(stations[station_mask, 0], stations[station_mask, 1], marker="^", s=38,
               c="#FF4D2E", edgecolors="black", linewidths=0.55, label="台站", zorder=8)
    ax.set_title(f"{period} 震源与台站分布", fontsize=14)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend(loc="lower left", framealpha=0.85)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def render_ray_plan(
    source_rows: np.ndarray,
    station_rows: np.ndarray,
    boundary: np.ndarray,
    names: list[str],
    segments: list[np.ndarray],
    output_path: Path,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    period: str,
) -> None:
    fig, ax = plt.subplots(figsize=(12.0, 5.4), dpi=180, facecolor="white")
    draw_map(ax, segments, xlim, ylim)
    for source, station in zip(source_rows, station_rows):
        if max(source[0], station[0]) < xlim[0] or min(source[0], station[0]) > xlim[1]:
            continue
        if max(source[1], station[1]) < ylim[0] or min(source[1], station[1]) > ylim[1]:
            continue
        ax.plot([source[0], station[0]], [source[1], station[1]], color="#E53935",
                linewidth=0.35, alpha=0.11, zorder=3)
    if not segments:
        draw_boundary(ax, boundary, names)
    ax.set_title(f"{period} 反演射线覆盖", fontsize=14)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def render_ray_3d(
    source_rows: np.ndarray,
    station_rows: np.ndarray,
    sources: np.ndarray,
    stations: np.ndarray,
    output_path: Path,
    period: str,
) -> None:
    """Render the actual source-receiver geometry without a synthetic surface."""
    fig = plt.figure(figsize=(11.0, 7.2), dpi=180, facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    for source, station in zip(source_rows, station_rows):
        ax.plot(
            [source[0], station[0]], [source[1], station[1]], [source[2], station[2]],
            color="#EC4899", linewidth=0.35, alpha=0.10,
        )
    ax.scatter(sources[:, 0], sources[:, 1], sources[:, 2], s=10, c="#1D4ED8",
               alpha=0.72, label="反演震源", depthshade=False)
    ax.scatter(stations[:, 0], stations[:, 1], stations[:, 2], s=30, marker="^",
               c="#EF4444", edgecolors="black", linewidths=0.45,
               label="台站", depthshade=False)
    ax.set_title(f"{period} 三维反演射线覆盖".strip(), fontsize=14, pad=12)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.view_init(elev=22, azim=-56)
    ax.legend(loc="upper right", framealpha=0.88)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def render_derived_maps(
    velocity: np.ndarray,
    ray_density: np.ndarray,
    xc: np.ndarray,
    yc: np.ndarray,
    zc: np.ndarray,
    target_z: float,
    background_velocity: float,
    boundary: np.ndarray,
    names: list[str],
    segments: list[np.ndarray],
    output_dir: Path,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    reliable_coverage: float,
    anomaly_limit: float = 0.30,
) -> None:
    slice_xy = interpolate_z(velocity, zc, target_z)
    density_xy = interpolate_z(ray_density, zc, target_z)
    anomaly = (slice_xy - background_velocity) / background_velocity * 100.0
    gx, gy = np.gradient(slice_xy, float(np.mean(np.diff(xc))), float(np.mean(np.diff(yc))), edge_order=1)
    gradient = np.sqrt(gx**2 + gy**2)
    density_xf, density_yf, density_field = fine_field(density_xy, xc, yc, xlim, ylim)
    density_mesh_x, density_mesh_y = np.meshgrid(density_xf, density_yf)
    density_field = np.maximum(density_field, 0.0)
    for field_xy, title, label, filename, cmap, symmetric in [
        (anomaly, f"{target_z:.0f}m 波速异常系数 An", "An (%)", f"anomaly_z{int(target_z)}.png", "seismic", True),
        (gradient, f"{target_z:.0f}m 波速变化梯度 VG", "VG ((m/s)/m)", f"gradient_z{int(target_z)}.png", "turbo", False),
    ]:
        xf, yf, field = fine_field(field_xy, xc, yc, xlim, ylim)
        if symmetric:
            limit = float(anomaly_limit * 100.0)
            vmin, vmax = -limit, limit
        else:
            vmin, vmax = 0.0, max(0.05, float(np.percentile(field, 98.0)))
        fig, ax = plt.subplots(figsize=(13.0, 6.2), dpi=180, facecolor="white")
        mesh = ax.contourf(xf, yf, np.clip(field, vmin, vmax), levels=70, cmap=cmap, extend="both")
        if np.nanmin(density_field) <= reliable_coverage <= np.nanmax(density_field):
            ax.contour(
                density_mesh_x,
                density_mesh_y,
                density_field,
                levels=[reliable_coverage],
                colors="#111827",
                linestyles="--",
                linewidths=0.9,
                zorder=3,
            )
        draw_map(ax, segments, xlim, ylim)
        colorbar = fig.colorbar(mesh, ax=ax, fraction=0.035, pad=0.025)
        colorbar.set_label(label)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.text(
            0.01,
            0.015,
            f"虚线内：可靠覆盖（≥{reliable_coverage:g}条射线/单元）",
            transform=ax.transAxes,
            fontsize=8.0,
            color="#111827",
            bbox={"facecolor": "white", "edgecolor": "#94A3B8", "alpha": 0.82, "pad": 2.0},
            zorder=9,
        )
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=220, bbox_inches="tight")
        plt.close(fig)


def render_vertical_sections(
    velocity: np.ndarray,
    xc: np.ndarray,
    yc: np.ndarray,
    zc: np.ndarray,
    boundary: np.ndarray,
    names: list[str],
    output_dir: Path,
    vmin: float,
    vmax: float,
) -> None:
    points = {name: point for name, point in zip(names, boundary[:-1])}
    required = {"V1", "V2", "V3", "V4", "V5", "V6"}
    if not required.issubset(points):
        return
    upper_start, upper_end = points["V1"], points["V5"]
    lower_start, lower_end = points["V2"], points["V6"]
    center_start = (upper_start + lower_start) / 2.0
    center_end = (upper_end + lower_end) / 2.0
    cross_start = upper_start * 0.35 + upper_end * 0.65
    cross_end = lower_start * 0.35 + lower_end * 0.65
    sections = [
        ("material_roadway", "材料顺槽垂直切面", upper_start, upper_end),
        ("transport_roadway", "运输顺槽垂直切面", lower_start, lower_end),
        ("workface_center", "工作面中线垂直切面", center_start, center_end),
        ("mining_dip", "回采位置倾向垂直切面", cross_start, cross_end),
    ]
    interpolator = RegularGridInterpolator((xc, yc, zc), velocity, bounds_error=False, fill_value=np.nan)
    for filename, title, start, end in sections:
        fraction = np.linspace(0.0, 1.0, 420)
        xy = start[None, :] + fraction[:, None] * (end - start)[None, :]
        distance = fraction * float(np.linalg.norm(end - start))
        zz = np.linspace(float(zc.min()), float(zc.max()), 220)
        xgrid, zgrid = np.meshgrid(distance, zz)
        xy_rep = np.repeat(xy[None, :, :], zz.size, axis=0)
        points3d = np.column_stack([xy_rep[:, :, 0].ravel(), xy_rep[:, :, 1].ravel(),
                                    np.repeat(zz, distance.size)])
        field = interpolator(points3d).reshape(zz.size, distance.size)
        fig, ax = plt.subplots(figsize=(11.0, 5.5), dpi=180, facecolor="white")
        levels = np.linspace(vmin / 1000.0, vmax / 1000.0, 90)
        contour = ax.contourf(xgrid, zgrid, np.clip(field, vmin, vmax) / 1000.0,
                              levels=levels, cmap="jet", extend="both")
        colorbar = fig.colorbar(contour, ax=ax, fraction=0.028, pad=0.025)
        colorbar.set_label("P波速度 (km/s)")
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("沿剖面距离 (m)")
        ax.set_ylabel("标高 Z (m)")
        ax.grid(False)
        fig.tight_layout()
        fig.savefig(output_dir / f"vertical_{filename}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="生成通用工作面CT报表图件")
    parser.add_argument("model_npz", type=Path)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--boundary-file", type=Path, required=True)
    parser.add_argument("--mapa-file", type=Path)
    parser.add_argument("--cad-file", type=Path, help="可选DWG或DXF矿井底图")
    parser.add_argument("--dwg-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--accoreconsole", type=Path)
    parser.add_argument("--cad-cache-dir", type=Path)
    parser.add_argument("--cad-x-offset", type=float, default=0.0)
    parser.add_argument("--cad-y-offset", type=float, default=0.0)
    parser.add_argument("--period", default="")
    parser.add_argument("--slice-z", default="1030,1050,1090")
    parser.add_argument("--background-velocity", type=float, default=0.0, help="0表示用模型中位数")
    parser.add_argument("--vmin", type=float, default=0.0, help="0表示按有射线单元稳健估计")
    parser.add_argument("--vmax", type=float, default=0.0, help="0表示按有射线单元稳健估计")
    parser.add_argument(
        "--presentation-vmin", type=float, default=0.0,
        help="平滑展示版色标下限；0表示围绕背景速度采用对称色标",
    )
    parser.add_argument(
        "--presentation-vmax", type=float, default=0.0,
        help="平滑展示版色标上限；0表示围绕背景速度采用对称色标",
    )
    parser.add_argument(
        "--presentation-sigma", type=float, default=0.65,
        help="平滑展示版的网格高斯平滑sigma；不修改velocity_model.npz",
    )
    parser.add_argument(
        "--coverage-weight-exponent", type=float, default=1.5,
        help="coverage confidence exponent used by presentation weighting",
    )
    parser.add_argument(
        "--anomaly-limit", type=float, default=0.30,
        help="shared symmetric relative An colour scale, e.g. 0.30",
    )
    parser.add_argument("--x-min", type=float, default=5500.0)
    parser.add_argument("--x-max", type=float, default=7500.0)
    parser.add_argument("--y-min", type=float, default=4700.0)
    parser.add_argument("--y-max", type=float, default=6300.0)
    args = parser.parse_args()

    model = np.load(args.model_npz)
    source_rows, station_rows, sources, stations = load_rays(args.input_csv)
    if "used_observation_row_indices" in model.files:
        used_rows = np.asarray(model["used_observation_row_indices"], dtype=np.int64)
        used_rows = used_rows[(used_rows >= 0) & (used_rows < source_rows.shape[0])]
        source_rows = source_rows[used_rows]
        station_rows = station_rows[used_rows]
        sources = np.unique(source_rows, axis=0)
        stations = np.unique(station_rows, axis=0)
        print(f"workface rays after inversion QC: {source_rows.shape[0]}")
    vertex_names, boundary = load_boundary(args.boundary_file)
    xlim = (args.x_min, args.x_max)
    ylim = (args.y_min, args.y_max)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    map_segments: list[np.ndarray] = []
    map_labels: list[tuple[float, float, str, float]] = []
    map_source = "none"
    segment_cache: Path | None = None
    cad_file = args.cad_file or args.dwg_file
    if cad_file is not None and cad_file.is_file():
        try:
            segment_cache = prepare_cad_segments(
                cad_file, xlim, ylim,
                args.cad_x_offset, args.cad_y_offset,
                args.accoreconsole,
                args.cad_cache_dir,
            )
            map_segments = load_segment_cache(segment_cache)
            map_labels = load_label_cache(segment_cache)
            map_source = f"{cad_file.suffix.upper().lstrip('.')}: {cad_file}"
            print(f"CAD vector segments: {len(map_segments)}")
            print(f"CAD text labels: {len(map_labels)}")
        except Exception as exc:
            print(f"WARNING: CAD processing failed; fallback to Mapa.dat: {exc}")
    if not map_segments:
        map_segments = load_mapa_segments(args.mapa_file)
        if map_segments:
            map_source = f"Mapa.dat: {args.mapa_file}"
    (args.output_dir / "cad_background_source.txt").write_text(
        "\n".join([
            f"source={map_source}",
            f"vector_cache={segment_cache or ''}",
            f"cad_coordinate_offset={args.cad_x_offset:.3f},{args.cad_y_offset:.3f}",
            f"plot_window=x[{xlim[0]:.3f},{xlim[1]:.3f}],y[{ylim[0]:.3f},{ylim[1]:.3f}]",
            f"segments={len(map_segments)}",
        ]),
        encoding="utf-8",
    )

    # Maps start from the quantitative recovered velocity.  Coverage is used
    # only to build a presentation support envelope and a dedicated QC layer;
    # a stored display field must never erase valid one-ray structure before
    # that renderer gets a chance to show it.
    velocity = np.asarray(model["velocity"], dtype=np.float64)
    ray_density = model["ray_density"]
    coverage_exponent = args.coverage_weight_exponent
    if "coverage_weight_exponent" in model.files:
        coverage_exponent = float(
            np.asarray(model["coverage_weight_exponent"]).reshape(())
        )
    if not np.isfinite(coverage_exponent) or coverage_exponent <= 0.0:
        raise ValueError("coverage weight exponent must be finite and positive")
    if not np.isfinite(args.anomaly_limit) or args.anomaly_limit <= 0.0:
        raise ValueError("anomaly limit must be finite and positive")
    reliable_coverage = (
        float(np.asarray(model["reliable_coverage"]).reshape(()))
        if "reliable_coverage" in model.files
        else 5.0
    )
    # The SIRT backend may determine a calibrated initial/background velocity
    # which differs from the GUI's project-default velocity.  Presentation
    # must be centred on that model value; otherwise a neutral field is shifted
    # toward one end of the colour bar and can look like a false red/blue
    # background.  The saved model is therefore authoritative whenever it
    # carries ``background_velocity_mps``.
    model_background = None
    if "background_velocity_mps" in model.files:
        candidate_background = float(
            np.asarray(model["background_velocity_mps"]).reshape(())
        )
        if np.isfinite(candidate_background) and candidate_background > 0.0:
            model_background = candidate_background
    if model_background is not None:
        if (
            args.background_velocity > 0.0
            and abs(float(args.background_velocity) - model_background) > 1.0
        ):
            print(
                "workface display: using calibrated model background "
                f"{model_background:.2f} m/s instead of GUI value "
                f"{args.background_velocity:.2f} m/s"
            )
        background_velocity = model_background
    elif args.background_velocity > 0.0:
        background_velocity = float(args.background_velocity)
    else:
        background_velocity = float(np.nanmedian(velocity))
    covered_values = velocity[ray_density >= 0.5]
    if covered_values.size < 20:
        covered_values = velocity[np.isfinite(velocity)]
    auto_low, auto_high = np.percentile(covered_values, [2.0, 98.0])
    vmin = args.vmin if args.vmin > 0 else float(auto_low)
    vmax = args.vmax if args.vmax > vmin else float(auto_high)
    if vmax <= vmin:
        span = max(background_velocity * 0.1, 100.0)
        vmin, vmax = background_velocity - span, background_velocity + span
    if args.presentation_vmin > 0 and args.presentation_vmax > args.presentation_vmin:
        presentation_vmin = float(args.presentation_vmin)
        presentation_vmax = float(args.presentation_vmax)
    else:
        symmetric_span = max(background_velocity - vmin, vmax - background_velocity, 100.0)
        presentation_vmin = background_velocity - symmetric_span
        presentation_vmax = background_velocity + symmetric_span
    print(f"workface display velocity: {vmin:.2f}-{vmax:.2f} m/s")
    print(
        "presentation velocity: "
        f"{presentation_vmin:.2f}-{presentation_vmax:.2f} m/s, "
        f"requested sigma={args.presentation_sigma:.2f} cells "
        "(local anomaly renderer uses 0.45-0.75 cells)"
    )
    print(f"An reference velocity: {background_velocity:.2f} m/s")

    render_source_distribution(sources, stations, boundary, vertex_names, map_segments,
                               args.output_dir / "source_distribution.png", xlim, ylim, args.period)
    render_ray_plan(source_rows, station_rows, boundary, vertex_names, map_segments,
                    args.output_dir / "ray_coverage_plan.png", xlim, ylim, args.period)
    render_ray_3d(source_rows, station_rows, sources, stations,
                  args.output_dir / "ray_coverage_3d.png", args.period)
    targets = [float(value.strip()) for value in args.slice_z.split(",") if value.strip()]
    if not targets:
        model_zc = np.asarray(model["zc"], dtype=np.float64).reshape(-1)
        targets = sorted({float(value) for value in model_zc if np.isfinite(value)})
        if not targets:
            raise ValueError(
                "No valid slice elevations: --slice-z is empty and model zc has no finite values."
            )
        print(
            "slice elevations not configured; using model zc layers: "
            + ",".join(f"{value:.3f}" for value in targets)
        )

    # Slice products are regenerated as one coherent set.  Keeping images from
    # an earlier elevation selection caused stale maps (for example -768.066 m)
    # to be copied into the final-result folder after a new -750/-780/-810 m
    # run.  Those stale maps may have been made by an old renderer and can have
    # a completely different background appearance, so they must never be
    # presented beside the current inversion.
    for generated_pattern in (
        "anomaly_z*.png",
        "gradient_z*.png",
        "yanbei_velocity_z*.png",
        "key_inversion_z*.png",
        "key_workface_z*.png",
        "key_workface_anomaly_enhanced_z*.png",
        "surfer_style_velocity_z*.png",
        "surfer_style_anomaly_z*.png",
        "coverage_reliability_z*.png",
        "vendor_style_velocity_z*.png",
        "vendor_style_anomaly_z*.png",
        "key_velocity_slice_z*.png",
        "key_workface_anomaly_z*.png",
        "key_workface_velocity_z*.png",
        "velocity_slice_z*.png",
        "velocity_z*.png",
        "velocity_uncertainty_z*.png",
    ):
        for generated_path in args.output_dir.glob(generated_pattern):
            generated_path.unlink()
    audit = coordinate_alignment_audit(
        velocity,
        model["xc"], model["yc"], model["zc"],
        model["xnodes"], model["ynodes"], model["znodes"],
        source_rows, station_rows, map_segments,
        xlim, ylim, targets,
        (args.cad_x_offset, args.cad_y_offset),
        boundary,
    )
    (args.output_dir / "coordinate_alignment_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if audit["status"] == "FAIL":
        raise ValueError(
            "coordinate alignment contract failed: "
            + "; ".join(str(item) for item in audit["failures"])
        )
    print(f"coordinate alignment audit: {audit['status']}")
    for warning in audit["warnings"]:
        print(f"WARNING: coordinate alignment: {warning}")
    registration = audit.get("cad_workface_registration", {})
    if registration.get("status") == "REVIEW_REQUIRED":
        shift = registration.get("centre_shift_m", [])
        print(
            "WARNING: CAD registration review required; extent-based "
            f"translation candidate (cad_x_offset,cad_y_offset)={shift}. "
            "It is recorded for review and is not applied automatically."
        )
    for target_z in targets:
        presentation_path = args.output_dir / f"surfer_style_velocity_z{target_z:.3f}.png"
        render_presentation_slice(
            velocity, ray_density, model["xc"], model["yc"], model["zc"], target_z,
            map_segments, presentation_path,
            presentation_vmin, presentation_vmax, background_velocity,
            xlim, ylim, args.period, max(0.0, args.presentation_sigma), boundary,
            reliable_coverage,
            coverage_exponent,
        )
        presentation_anomaly_path = (
            args.output_dir / f"surfer_style_anomaly_z{target_z:.3f}.png"
        )
        render_presentation_anomaly_slice(
            velocity, ray_density, model["xc"], model["yc"], model["zc"], target_z,
            background_velocity, map_segments, boundary,
            presentation_anomaly_path, xlim, ylim, args.period,
            max(0.0, args.presentation_sigma), reliable_coverage,
            coverage_exponent, args.anomaly_limit,
        )
        render_coverage_reliability(
            ray_density, model["xc"], model["yc"], model["zc"], target_z,
            map_segments, boundary,
            args.output_dir / f"coverage_reliability_z{target_z:.3f}.png",
            xlim, ylim, reliable_coverage,
        )
        print(f"saved {presentation_path}")
    render_vendor_style_bundle(
        velocity,
        ray_density,
        model["xc"],
        model["yc"],
        model["zc"],
        targets,
        background_velocity,
        reliable_coverage,
        max(0.0, args.presentation_sigma),
        map_segments,
        map_labels,
        boundary,
        args.output_dir,
        xlim,
        ylim,
        presentation_vmin,
        presentation_vmax,
        sources,
        stations,
        coverage_exponent=coverage_exponent,
        anomaly_limit=args.anomaly_limit,
    )
    render_vertical_sections(
        velocity, model["xc"], model["yc"], model["zc"],
        boundary, vertex_names, args.output_dir, vmin, vmax,
    )
    from wave_ct.validation_outputs import (
        run_checkerboard_test,
        write_parameter_report,
        write_target_validation_assets,
    )

    write_parameter_report(
        args.model_npz, args.output_dir, max(0.0, args.presentation_sigma)
    )
    run_checkerboard_test(
        args.model_npz, source_rows, station_rows, args.output_dir,
    )
    write_target_validation_assets(
        args.model_npz, args.output_dir, targets,
    )
    from wave_ct.deliverables import export_model_text_bundle

    export_model_text_bundle(
        args.model_npz,
        args.output_dir,
        targets,
        image_source_dir=args.output_dir,
        include_workface=True,
    )


if __name__ == "__main__":
    main()
