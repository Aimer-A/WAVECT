"""Generate publication-oriented validation assets for a completed inversion."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.sparse.linalg import lsqr

from wave_ct.inversion import build_siddon_matrix


def _display_origins(xnodes: np.ndarray, ynodes: np.ndarray) -> tuple[float, float]:
    """Match Matplotlib's compact mine-coordinate offsets used in velocity_z plots."""
    x_origin = float(np.floor(np.nanmin(xnodes) / 10000.0) * 10000.0)
    y_origin = float(np.floor(np.nanmin(ynodes) / 1000.0) * 1000.0)
    return x_origin, y_origin


def _parse_report(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def write_parameter_report(
    model_path: Path,
    output_dir: Path,
    presentation_sigma: float,
) -> Path:
    """Write the parameters needed to reproduce and describe the inversion."""
    report = _parse_report(output_dir / "slice_report.txt")
    with np.load(model_path, allow_pickle=False) as model:
        velocity = np.asarray(model["velocity"], dtype=float)
        xc = np.asarray(model["xc"], dtype=float)
        yc = np.asarray(model["yc"], dtype=float)
        zc = np.asarray(model["zc"], dtype=float)
        background = float(np.asarray(model["background_velocity_mps"]).reshape(()))
        method = str(np.asarray(model["inversion_method"]).reshape(()))

    fields = {
        "反演方法": method,
        "模型参数化": "相对背景慢度扰动（成果以 An=(V-V0)/V0 表示）",
        "初始/背景速度模型": f"均匀模型 {background:.3f} m/s",
        "网格大小": f"{xc.size} × {yc.size} × {zc.size} 单元",
        "单元尺寸": report.get("cell_size", "未记录"),
        "模型范围": report.get("bounds", "未记录"),
        "展示平滑系数 sigma": f"{presentation_sigma:.3f} 单元",
        "正则化系数 alpha": report.get("alpha_reg", "未记录"),
        "步长阻尼": report.get("step_damp", "未记录"),
        "背景阻尼": report.get("background_damping", "未记录"),
        "模型阻尼": report.get("model_damping", "未记录"),
        "初始 RMS": f"{float(report.get('initial_rms_s', 'nan')) * 1000:.3f} ms",
        "最终训练 RMS": f"{float(report.get('solution_train_rms_s', 'nan')) * 1000:.3f} ms",
        "验证 RMS": (
            f"{float(report.get('solution_validation_rms_s', 'nan')) * 1000:.3f} ms"
        ),
        "外层迭代次数": report.get("n_outer", "未记录"),
        "LSQR 最大迭代次数": report.get("n_lsqr", "未记录"),
        "DNR 网络": (
            f"width={report.get('deep_reparam_width', '-')}, "
            f"depth={report.get('deep_reparam_depth', '-')}, "
            f"epochs={report.get('deep_reparam_full_epochs', '-')}, "
            f"starts={report.get('deep_reparam_starts', '-')}"
        ),
        "DNR 覆盖约束": (
            f"{report.get('deep_reparam_coverage_gate', '未记录')}，"
            f"指数={report.get('deep_reparam_coverage_exponent', '-')}，"
            f"门控范围={report.get('deep_reparam_coverage_gate_range', '-')}"
        ),
        "DNR TV 权重": report.get("deep_reparam_tv", "未记录"),
        "速度范围": f"{np.nanmin(velocity):.3f}–{np.nanmax(velocity):.3f} m/s",
    }
    target = output_dir / "反演参数报告.txt"
    target.write_text(
        "WaveCT 反演参数\n" + "=" * 60 + "\n"
        + "\n".join(f"{key}: {value}" for key, value in fields.items())
        + "\n\n说明：validation RMS 为 NaN 时，表示本次正式反演没有留出独立验证事件，"
        "不能将训练 RMS 写成独立预测误差。\n",
        encoding="utf-8",
    )
    return target


def run_checkerboard_test(
    model_path: Path,
    source_rows: np.ndarray,
    station_rows: np.ndarray,
    output_dir: Path,
    *,
    contrast: float = 0.10,
    noise_ms: float = 0.5,
    damping: float = 5.0,
) -> tuple[Path, Path]:
    """Run a deterministic linearized checkerboard recovery on the real geometry."""
    with np.load(model_path, allow_pickle=False) as model:
        xnodes = np.asarray(model["xnodes"], dtype=float)
        ynodes = np.asarray(model["ynodes"], dtype=float)
        znodes = np.asarray(model["znodes"], dtype=float)
        xc = np.asarray(model["xc"], dtype=float)
        yc = np.asarray(model["yc"], dtype=float)
        zc = np.asarray(model["zc"], dtype=float)
        background = float(np.asarray(model["background_velocity_mps"]).reshape(()))

    nx, ny, nz = xc.size, yc.size, zc.size
    gmat, density = build_siddon_matrix(
        source_rows[:, 0], source_rows[:, 1], source_rows[:, 2],
        station_rows[:, 0], station_rows[:, 1], station_rows[:, 2],
        xnodes, ynodes, znodes, nx, ny, nz,
    )
    ix, iy, iz = np.indices((nx, ny, nz))
    sign = np.where(((ix // 4 + iy // 3 + iz) % 2) == 0, 1.0, -1.0)
    truth_velocity = background * (1.0 + contrast * sign)
    truth_slowness = 1.0 / truth_velocity
    reference_slowness = np.full(truth_velocity.size, 1.0 / background)
    truth_vector = truth_slowness.reshape(-1, order="F")
    synthetic_delta_t = gmat @ (truth_vector - reference_slowness)
    rng = np.random.default_rng(20260724)
    synthetic_delta_t = synthetic_delta_t + rng.normal(
        0.0, noise_ms / 1000.0, synthetic_delta_t.size
    )
    recovered_delta = lsqr(
        gmat, synthetic_delta_t, damp=damping, iter_lim=240,
        atol=1e-10, btol=1e-10,
    )[0]
    recovered_slowness = np.clip(
        reference_slowness + recovered_delta,
        1.0 / (background * 1.8),
        1.0 / (background * 0.45),
    )
    recovered_velocity = (1.0 / recovered_slowness).reshape(
        (nx, ny, nz), order="F"
    )
    recovered_velocity = gaussian_filter(
        recovered_velocity, sigma=(0.65, 0.65, 0.0), mode="nearest"
    )
    truth_anomaly = (truth_velocity - background) / background
    recovered_anomaly = (recovered_velocity - background) / background
    supported = density.reshape((nx, ny, nz), order="F") > 0
    truth_supported = truth_anomaly[supported]
    recovered_supported = recovered_anomaly[supported]
    correlation = (
        float(np.corrcoef(truth_supported, recovered_supported)[0, 1])
        if truth_supported.size > 1 else float("nan")
    )
    sign_recovery = float(
        np.mean(np.sign(truth_supported) == np.sign(recovered_supported))
    )
    amplitude_ratio = float(
        np.sqrt(np.mean(recovered_supported**2))
        / max(np.sqrt(np.mean(truth_supported**2)), 1e-12)
    )
    rmse = float(np.sqrt(np.mean((truth_supported - recovered_supported) ** 2)))

    fig, axes = plt.subplots(nz, 2, figsize=(11.5, 3.4 * nz), squeeze=False)
    x_origin, y_origin = _display_origins(xnodes, ynodes)
    xlocal = xc - x_origin
    ylocal = yc - y_origin
    levels = np.linspace(-contrast, contrast, 81)
    norm = TwoSlopeNorm(vmin=-contrast, vcenter=0.0, vmax=contrast)
    last = None
    for layer in range(nz):
        for col, (field, title) in enumerate((
            (truth_anomaly[:, :, layer], "输入棋盘格"),
            (recovered_anomaly[:, :, layer], "恢复结果"),
        )):
            last = axes[layer, col].contourf(
                xlocal, ylocal, field.T, levels=levels, cmap="jet",
                norm=norm, extend="both",
            )
            axes[layer, col].set_title(f"{title}  z={zc[layer]:.3f} m")
            axes[layer, col].set_xlabel("局部 X (m)")
            axes[layer, col].set_ylabel("局部 Y (m)")
            axes[layer, col].set_aspect("equal")
    if last is not None:
        fig.colorbar(last, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02,
                     label="速度异常系数")
    fig.suptitle(
        f"棋盘格恢复测试  corr={correlation:.3f}, "
        f"符号恢复率={sign_recovery * 100:.1f}%",
        fontsize=13,
    )
    fig.subplots_adjust(top=0.93, right=0.90, hspace=0.38, wspace=0.20)
    image_path = output_dir / "棋盘格恢复测试.png"
    fig.savefig(image_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    metrics = {
        "test_type": "linearized_checkerboard_recovery_real_ray_geometry",
        "checkerboard_contrast": contrast,
        "noise_standard_deviation_ms": noise_ms,
        "damping": damping,
        "rays": int(gmat.shape[0]),
        "grid_cells": [nx, ny, nz],
        "ray_supported_cells": int(np.count_nonzero(supported)),
        "correlation_supported_cells": correlation,
        "sign_recovery_fraction_supported_cells": sign_recovery,
        "amplitude_recovery_ratio_supported_cells": amplitude_ratio,
        "anomaly_rmse_supported_cells": rmse,
        "interpretation_limit": (
            "This linearized checkerboard test evaluates spatial resolution for "
            "the real ray geometry; it does not prove that every real-data anomaly "
            "is geological or replace borehole/mining-pressure/gas validation."
        ),
    }
    metrics_path = output_dir / "棋盘格恢复测试指标.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return image_path, metrics_path


def write_target_validation_assets(
    model_path: Path,
    output_dir: Path,
    targets: list[float],
    *,
    local_x: tuple[float, float] = (1400.0, 1600.0),
    local_y: tuple[float, float] = (400.0, 600.0),
) -> tuple[Path, Path]:
    """Mark and summarize the requested field-validation window."""
    if not targets:
        raise ValueError("At least one target elevation is required")
    with np.load(model_path, allow_pickle=False) as model:
        velocity = np.asarray(model["display_velocity"], dtype=float)
        xc = np.asarray(model["xc"], dtype=float)
        yc = np.asarray(model["yc"], dtype=float)
        zc = np.asarray(model["zc"], dtype=float)
        background = float(np.asarray(model["background_velocity_mps"]).reshape(()))
        xnodes = np.asarray(model["xnodes"], dtype=float)
        ynodes = np.asarray(model["ynodes"], dtype=float)
        ray_density = (
            np.asarray(model["ray_density"], dtype=float)
            if "ray_density" in model.files
            else np.zeros_like(velocity, dtype=float)
        )
        reliable_coverage = (
            float(np.asarray(model["reliable_coverage"]).reshape(()))
            if "reliable_coverage" in model.files
            else 1.0
        )
    x_origin, y_origin = _display_origins(xnodes, ynodes)
    xlocal = xc - x_origin
    ylocal = yc - y_origin
    mask = (
        (xlocal[:, None] >= local_x[0]) & (xlocal[:, None] <= local_x[1])
        & (ylocal[None, :] >= local_y[0]) & (ylocal[None, :] <= local_y[1])
    )
    if not np.any(mask):
        # Imported datasets do not necessarily share the historical 720 local
        # coordinate convention.  Pick a compact, coverage-centred reporting
        # window instead of reducing an empty array.  This affects only the
        # validation annotation and never changes the quantitative model.
        layer_indices = sorted(
            {int(np.argmin(np.abs(zc - target))) for target in targets}
        )
        coverage_2d = np.nanmax(ray_density[:, :, layer_indices], axis=2)
        supported = coverage_2d >= max(reliable_coverage, 1.0)
        if not np.any(supported):
            supported = coverage_2d > 0.0
        xx, yy = np.meshgrid(xlocal, ylocal, indexing="ij")
        if np.any(supported):
            weights = np.where(supported, np.maximum(coverage_2d, 1.0), 0.0)
            center_x = float(np.sum(xx * weights) / np.sum(weights))
            center_y = float(np.sum(yy * weights) / np.sum(weights))
        else:
            center_x = float(np.nanmedian(xlocal))
            center_y = float(np.nanmedian(ylocal))
        dx = float(np.nanmedian(np.diff(xlocal))) if xlocal.size > 1 else 1.0
        dy = float(np.nanmedian(np.diff(ylocal))) if ylocal.size > 1 else 1.0
        span_x = float(np.nanmax(xlocal) - np.nanmin(xlocal))
        span_y = float(np.nanmax(ylocal) - np.nanmin(ylocal))
        width = min(span_x, max(0.25 * span_x, 4.0 * abs(dx)))
        height = min(span_y, max(0.25 * span_y, 4.0 * abs(dy)))
        x0 = float(
            np.clip(
                center_x - width / 2.0,
                np.nanmin(xlocal),
                np.nanmax(xlocal) - width,
            )
        )
        y0 = float(
            np.clip(
                center_y - height / 2.0,
                np.nanmin(ylocal),
                np.nanmax(ylocal) - height,
            )
        )
        local_x = (x0, x0 + width)
        local_y = (y0, y0 + height)
        mask = (
            (xlocal[:, None] >= local_x[0]) & (xlocal[:, None] <= local_x[1])
            & (ylocal[None, :] >= local_y[0]) & (ylocal[None, :] <= local_y[1])
        )
        if not np.any(mask):
            ix = int(np.nanargmin(np.abs(xlocal - center_x)))
            iy = int(np.nanargmin(np.abs(ylocal - center_y)))
            mask[ix, iy] = True
            local_x = (
                float(xlocal[ix] - abs(dx) / 2.0),
                float(xlocal[ix] + abs(dx) / 2.0),
            )
            local_y = (
                float(ylocal[iy] - abs(dy) / 2.0),
                float(ylocal[iy] + abs(dy) / 2.0),
            )
    rows: list[str] = []
    fig, axes = plt.subplots(1, len(targets), figsize=(5.0 * len(targets), 4.2),
                             squeeze=False)
    for col, target in enumerate(targets):
        layer = int(np.argmin(np.abs(zc - target)))
        anomaly = (velocity[:, :, layer] - background) / background
        values = anomaly[mask]
        rows.append(
            f"- z={zc[layer]:.3f} m：重点区 An 范围 "
            f"{np.nanmin(values):.4f}～{np.nanmax(values):.4f}，"
            f"均值 {np.nanmean(values):.4f}，绝对异常均值 {np.nanmean(np.abs(values)):.4f}。"
        )
        ax = axes[0, col]
        mesh = ax.contourf(
            xlocal, ylocal, anomaly.T, levels=np.linspace(-0.3, 0.3, 81),
            cmap="jet", norm=TwoSlopeNorm(vmin=-0.3, vcenter=0, vmax=0.3),
            extend="both",
        )
        ax.add_patch(Rectangle(
            (local_x[0], local_y[0]),
            local_x[1] - local_x[0], local_y[1] - local_y[0],
            fill=False, edgecolor="magenta", linewidth=2.2,
        ))
        ax.set_title(f"z={zc[layer]:.3f} m")
        ax.set_xlabel("局部 X (m)")
        ax.set_ylabel("局部 Y (m)")
        ax.set_aspect("equal")
    fig.colorbar(mesh, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02,
                 label="速度异常系数 An")
    fig.suptitle("重点现场验证区（洋红框）")
    fig.subplots_adjust(top=0.86, right=0.92, wspace=0.28)
    image_path = output_dir / "重点异常区三层对比.png"
    fig.savefig(image_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    report_path = output_dir / "重点异常区验证建议.md"
    report_path.write_text(
        "# 重点异常区现场验证建议\n\n"
        f"- 局部坐标范围：X={local_x[0]:.0f}～{local_x[1]:.0f} m，"
        f"Y={local_y[0]:.0f}～{local_y[1]:.0f} m。\n"
        f"- 全局坐标范围：X={x_origin + local_x[0]:.3f}～"
        f"{x_origin + local_x[1]:.3f} m，Y={y_origin + local_y[0]:.3f}～"
        f"{y_origin + local_y[1]:.3f} m。\n\n"
        + "\n".join(rows)
        + "\n\n## 建议核验资料\n\n"
        "1. 对照矿压监测、支架阻力及微震能量/频次，检查高速度异常与应力集中是否同位。\n"
        "2. 对照瓦斯浓度、抽采量和钻孔异常，检查低速度异常与裂隙发育或破碎带是否同位。\n"
        "3. 对照断层、陷落柱、巷道揭露和钻孔柱状图，验证异常边界及NE–SW展布方向。\n"
        "4. 在没有现场证据前，只能表述为“待验证速度异常体”，不能直接判定为高应力区或裂隙区。\n",
        encoding="utf-8",
    )
    return image_path, report_path
