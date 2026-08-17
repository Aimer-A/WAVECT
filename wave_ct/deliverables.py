"""Export an auditable WaveCT result bundle.

The quantitative inverse solution and the coverage-stabilised presentation
field intentionally remain separate.  This prevents a smooth figure from
silently becoming the velocity model used for forward prediction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np


FINAL_RESULT_DIR_NAME = "最终成果图"


def collect_final_result_images(result_dir: Path) -> list[Path]:
    """Return only the PNG files explicitly packaged as final products."""
    final_dir = Path(result_dir) / FINAL_RESULT_DIR_NAME
    if not final_dir.is_dir():
        return []
    return sorted(
        (path for path in final_dir.glob("*.png") if path.is_file()),
        key=lambda path: path.name.casefold(),
    )


def interpolate_z(volume: np.ndarray, zc: np.ndarray, target: float) -> np.ndarray:
    """Linearly interpolate a cell-centred 3-D volume to one elevation."""
    values = np.asarray(volume, dtype=np.float64)
    levels = np.asarray(zc, dtype=np.float64)
    if values.ndim != 3 or values.shape[2] != levels.size:
        raise ValueError("volume and z coordinates do not match")
    if target <= levels[0]:
        return values[:, :, 0].copy()
    if target >= levels[-1]:
        return values[:, :, -1].copy()
    upper = int(np.searchsorted(levels, target))
    lower = upper - 1
    weight = (target - levels[lower]) / (levels[upper] - levels[lower])
    return (1.0 - weight) * values[:, :, lower] + weight * values[:, :, upper]


def _scalar(model: np.lib.npyio.NpzFile, name: str, default: float) -> float:
    if name not in model.files:
        return float(default)
    return float(np.asarray(model[name]).reshape(()))


def _same_grid(
    candidate: np.lib.npyio.NpzFile,
    xc: np.ndarray,
    yc: np.ndarray,
    zc: np.ndarray,
) -> bool:
    return (
        "velocity" in candidate.files
        and np.asarray(candidate["velocity"]).shape == (xc.size, yc.size, zc.size)
        and all(
            name in candidate.files
            and np.allclose(np.asarray(candidate[name]), expected, rtol=0.0, atol=1e-8)
            for name, expected in (("xc", xc), ("yc", yc), ("zc", zc))
        )
    )


def load_grouped_cv_stability(
    result_dir: Path,
    xc: np.ndarray,
    yc: np.ndarray,
    zc: np.ndarray,
) -> tuple[np.ndarray, int, str]:
    """Return split-to-split velocity spread for the selected auto candidate."""
    empty = np.full((xc.size, yc.size, zc.size), np.nan, dtype=np.float64)
    report_path = result_dir / "auto_strategy_report.json"
    if not report_path.is_file():
        return empty, 0, "not_available_manual_or_legacy_run"
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        selected = str(report["selected_candidate"])
        run_root = result_dir / "auto_strategy_runs"
        models: list[np.ndarray] = []
        for path in sorted(run_root.glob(f"{selected}_s*/velocity_model.npz")):
            with np.load(path, allow_pickle=False) as candidate:
                if _same_grid(candidate, xc, yc, zc):
                    models.append(np.asarray(candidate["velocity"], dtype=np.float64))
        if len(models) < 2:
            return empty, len(models), "fewer_than_two_grid_aligned_grouped_cv_models"
        return (
            np.std(np.stack(models, axis=0), axis=0, ddof=1),
            len(models),
            "selected_candidate_grouped_cv_split_standard_deviation",
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return empty, 0, "auto_report_or_pilot_models_unreadable"


def _write_surfer_xyz(
    path: Path,
    xc: np.ndarray,
    yc: np.ndarray,
    field: np.ndarray,
    decimals: int,
) -> None:
    """Write headerless X/Y/value rows compatible with the supplied Surfer TXT."""
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for iy in range(yc.size - 1, -1, -1):
            for ix in range(xc.size):
                handle.write(
                    f"{xc[ix]:20.3f} {yc[iy]:20.3f} "
                    f"{field[ix, iy]:20.{decimals}f}\n"
                )


def _iter_detailed_rows(
    xc: np.ndarray,
    yc: np.ndarray,
    z_values: Iterable[float],
    velocity_slices: Iterable[np.ndarray],
    background_velocity: float,
    density_slices: Iterable[np.ndarray],
    coverage_slices: Iterable[np.ndarray],
    stability_slices: Iterable[np.ndarray],
    reliable_coverage: float,
) -> Iterable[str]:
    for z_value, velocity, density, coverage, stability in zip(
        z_values, velocity_slices, density_slices, coverage_slices, stability_slices
    ):
        for iy in range(yc.size - 1, -1, -1):
            for ix in range(xc.size):
                speed = float(velocity[ix, iy])
                anomaly = (speed - background_velocity) / background_velocity
                spread = float(stability[ix, iy])
                spread_percent = (
                    spread / background_velocity * 100.0
                    if np.isfinite(spread) else float("nan")
                )
                yield (
                    f"{xc[ix]:.3f}\t{yc[iy]:.3f}\t{z_value:.3f}\t"
                    f"{speed:.6f}\t{speed / 1000.0:.9f}\t"
                    f"{anomaly:.9f}\t{anomaly * 100.0:.6f}\t"
                    f"{float(density[ix, iy]):.6f}\t"
                    f"{float(coverage[ix, iy]):.9f}\t"
                    f"{int(density[ix, iy] >= reliable_coverage)}\t"
                    f"{spread:.6f}\t{spread_percent:.6f}\n"
                )


def _write_detailed_table(path: Path, rows: Iterable[str]) -> None:
    header = (
        "X_m\tY_m\tZ_m\tvelocity_mps\tvelocity_kmps\t"
        "anomaly_fraction\tanomaly_percent\tray_density\tcoverage_weight\t"
        "coverage_reliable_flag\tgrouped_cv_split_std_mps\t"
        "grouped_cv_split_std_percent\n"
    )
    with path.open("w", encoding="utf-8-sig", newline="\n") as handle:
        handle.write(header)
        handle.writelines(rows)


def _canonical_image_name(source: Path) -> str | None:
    name = source.name
    fixed = {
        "source_distribution.png": "01_震源台站与工作面底图.png",
        "ray_coverage_plan.png": "02_射线路径平面覆盖.png",
        "ray_coverage_3d.png": "03_三维射线覆盖.png",
        "rms_convergence.png": "30_反演收敛曲线.png",
    }
    if name in fixed:
        return fixed[name]
    patterns = (
        (r"yanbei_velocity_z(.+)\.png", "10_定量波速_带底图_z{}.png"),
        (r"velocity_slice_z(.+)\.png", "10_定量波速_z{}.png"),
        (r"surfer_style_velocity_z(.+)\.png", "11_平滑展示波速_带底图_z{}.png"),
        (r"anomaly_z(.+)\.png", "12_定量波速异常_带底图_z{}.png"),
        (r"surfer_style_anomaly_z(.+)\.png", "13_平滑展示波速异常_带底图_z{}.png"),
        (r"gradient_z(.+)\.png", "14_波速梯度_带底图_z{}.png"),
        (r"coverage_reliability_z(.+)\.png", "15_射线覆盖与可靠区_带底图_z{}.png"),
        (r"ray_coverage_z(.+)\.png", "15_射线覆盖_z{}.png"),
        (r"vertical_(.+)\.png", "20_垂直剖面_{}.png"),
        (r"vendor_style_velocity_z(.+)\.png", "50_波速_z{}.png"),
        (r"vendor_style_anomaly_z(.+)\.png", "51_异常_z{}.png"),
        # Compatibility with results rendered before the template was given a
        # neutral engineering name.
        (r"student_style_velocity_z(.+)\.png", "50_波速_z{}.png"),
        (r"student_style_anomaly_z(.+)\.png", "51_异常_z{}.png"),
    )
    for pattern, template in patterns:
        match = re.fullmatch(pattern, name)
        if match:
            return template.format(match.group(1))
    return None


def package_result_images(
    source_dir: Path,
    result_dir: Path,
    *,
    include_workface: bool = True,
) -> list[Path]:
    """Package the requested coordinate and workface figures for each elevation."""
    destination = result_dir / FINAL_RESULT_DIR_NAME
    destination.mkdir(parents=True, exist_ok=True)
    # This directory is generated output.  Remove products from older verbose
    # packaging runs so the user is not presented with dozens of alternatives.
    for old_image in destination.glob("*.png"):
        old_image.unlink()
    copied: list[Path] = []
    patterns = (
        # Keep the original inversion filename: these figures retain numeric
        # X/Y axes (displayed by Matplotlib with compact coordinate offsets).
        ("velocity_z*.png", "velocity"),
        ("velocity_uncertainty_z*.png", "DNR初始化不确定性"),
        ("coverage_reliability_z*.png", "射线可靠覆盖"),
        ("key_velocity_slice_z*.png", "速度模型切片"),
        ("key_workface_anomaly_z*.png", "带工作面An图"),
        ("key_workface_velocity_z*.png", "带工作面速度反演图"),
        ("vendor_style_velocity_z*.png", "50_波速"),
        ("vendor_style_anomaly_z*.png", "51_异常"),
    )
    workface_patterns = {
        "coverage_reliability_z*.png",
        "key_workface_anomaly_z*.png",
        "key_workface_velocity_z*.png",
        "vendor_style_velocity_z*.png",
        "vendor_style_anomaly_z*.png",
    }
    for pattern, label in patterns:
        if not include_workface and pattern in workface_patterns:
            continue
        for source in sorted(source_dir.glob(pattern)):
            match = re.search(r"_z(.+)\.png$", source.name)
            if not match:
                continue
            target = destination / f"{label}_z{match.group(1)}.png"
            shutil.copy2(source, target)
            copied.append(target)
    for source_name in (
        "coordinate_alignment_audit.json",
        "棋盘格恢复测试.png",
        "棋盘格恢复测试指标.json",
        "反演参数报告.txt",
        "重点异常区三层对比.png",
        "重点异常区验证建议.md",
    ):
        source = source_dir / source_name
        if source.is_file():
            target = destination / source.name
            shutil.copy2(source, target)
            copied.append(target)
    return copied


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_model_text_bundle(
    model_npz: Path,
    output_dir: Path,
    targets: Iterable[float],
    *,
    image_source_dir: Path | None = None,
    include_workface: bool | None = None,
) -> dict:
    """Export quantitative TXT, Surfer presentation TXT, notes and manifest.

    ``None`` is the safe default: preserve workface products whenever they
    already exist in ``image_source_dir``.  This prevents a later TXT export or
    reference-comparison refresh from clearing valid workface figures that
    were produced earlier in the official workflow.  Callers may pass
    ``False`` only for an explicitly generic, non-workface deliverable.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    quantitative_dir = output_dir / "速度模型TXT" / "定量反演解"
    presentation_dir = output_dir / "速度模型TXT" / "Surfer兼容展示网格"
    quantitative_dir.mkdir(parents=True, exist_ok=True)
    presentation_dir.mkdir(parents=True, exist_ok=True)

    with np.load(model_npz, allow_pickle=False) as model:
        velocity = np.asarray(model["velocity"], dtype=np.float64)
        display_velocity = np.asarray(
            model["display_velocity"] if "display_velocity" in model.files else velocity,
            dtype=np.float64,
        )
        ray_density = np.asarray(model["ray_density"], dtype=np.float64)
        coverage_weight = np.asarray(
            model["coverage_weight"]
            if "coverage_weight" in model.files
            else np.minimum(ray_density, 1.0),
            dtype=np.float64,
        )
        xc = np.asarray(model["xc"], dtype=np.float64)
        yc = np.asarray(model["yc"], dtype=np.float64)
        zc = np.asarray(model["zc"], dtype=np.float64)
        background_velocity = _scalar(
            model, "background_velocity_mps", float(np.nanmedian(velocity))
        )
        reliable_coverage = _scalar(model, "reliable_coverage", 1.0)
        inversion_method = (
            str(np.asarray(model["inversion_method"]).item())
            if "inversion_method" in model.files
            else "legacy_lsqr"
        )
        deep_enabled = bool(
            np.asarray(model["deep_reparameterization_enabled"]).item()
        ) if "deep_reparameterization_enabled" in model.files else False
        deep_stability = (
            np.asarray(model["velocity_uncertainty"], dtype=np.float64)
            if deep_enabled and "velocity_uncertainty" in model.files
            else None
        )
        deep_stability_count = (
            int(np.asarray(model["deep_ensemble_velocity"]).shape[0])
            if deep_enabled and "deep_ensemble_velocity" in model.files
            else 0
        )

    if velocity.shape != (xc.size, yc.size, zc.size):
        raise ValueError("velocity_model.npz contains an inconsistent model grid")
    requested_targets = [float(value) for value in targets]
    if not requested_targets:
        requested_targets = [float(value) for value in zc]
    stability, stability_count, stability_method = load_grouped_cv_stability(
        output_dir, xc, yc, zc
    )
    if deep_stability is not None:
        stability = deep_stability
        stability_count = deep_stability_count
        stability_method = (
            "deep_initialization_ensemble_velocity_standard_deviation"
        )

    written: list[Path] = []
    for target in requested_targets:
        label = f"{target:.3f}"
        exact_slice = interpolate_z(velocity, zc, target)
        display_slice = interpolate_z(display_velocity, zc, target)
        density_slice = interpolate_z(ray_density, zc, target)
        coverage_slice = interpolate_z(coverage_weight, zc, target)
        stability_slice = interpolate_z(stability, zc, target)

        exact_velocity_path = quantitative_dir / f"波速分布 z   {label}.txt"
        exact_anomaly_path = quantitative_dir / f"波速异常值分布 z   {label}.txt"
        _write_surfer_xyz(exact_velocity_path, xc, yc, exact_slice / 1000.0, 6)
        _write_surfer_xyz(
            exact_anomaly_path,
            xc,
            yc,
            (exact_slice - background_velocity) / background_velocity,
            9,
        )
        detail_path = quantitative_dir / f"完整字段 z   {label}.tsv"
        _write_detailed_table(
            detail_path,
            _iter_detailed_rows(
                xc, yc, [target], [exact_slice], background_velocity,
                [density_slice], [coverage_slice], [stability_slice],
                reliable_coverage,
            ),
        )

        presentation_velocity_path = presentation_dir / f"波速分布 z   {label}.txt"
        presentation_anomaly_path = presentation_dir / f"波速异常值分布 z   {label}.txt"
        _write_surfer_xyz(
            presentation_velocity_path, xc, yc, display_slice / 1000.0, 6
        )
        _write_surfer_xyz(
            presentation_anomaly_path,
            xc,
            yc,
            (display_slice - background_velocity) / background_velocity,
            9,
        )
        written.extend([
            exact_velocity_path,
            exact_anomaly_path,
            detail_path,
            presentation_velocity_path,
            presentation_anomaly_path,
        ])

    full_path = quantitative_dir / "三维速度模型_完整字段.tsv"
    full_rows = _iter_detailed_rows(
        xc,
        yc,
        zc,
        (velocity[:, :, iz] for iz in range(zc.size)),
        background_velocity,
        (ray_density[:, :, iz] for iz in range(zc.size)),
        (coverage_weight[:, :, iz] for iz in range(zc.size)),
        (stability[:, :, iz] for iz in range(zc.size)),
        reliable_coverage,
    )
    _write_detailed_table(full_path, full_rows)
    written.append(full_path)

    explanation = output_dir / "速度模型TXT" / "速度模型TXT说明.md"
    explanation.write_text(
        "\n".join([
            "# WaveCT 速度模型 TXT 说明",
            "",
            f"- 背景速度：{background_velocity:.6f} m/s。",
            f"- 可靠覆盖阈值：{reliable_coverage:.6f} 条射线/单元。",
            "- `定量反演解` 来自 `velocity_model.npz` 的 `velocity` 字段，可用于统计、复核和正演走时计算。",
            "- `Surfer兼容展示网格` 来自覆盖稳定化的 `display_velocity` 字段，只用于制图；不能替代定量反演解。",
            "- `波速分布` 第三列单位为 km/s；`波速异常值分布` 第三列为无量纲异常系数 `(V-V0)/V0`，均为无表头三列格式。",
            "- `完整字段.tsv` 的 `anomaly_percent` 单位为 %，`coverage_reliable_flag=1` 表示达到射线覆盖阈值。",
            f"- 稳定性样本数：{stability_count}；方法：`{stability_method}`。NaN 表示本次运行没有可比模型，不能伪造不确定度。",
            "- DNR 初始化分散度只表示优化初始化敏感性，不是贝叶斯后验标准差。",
            "- 图件中的高斯/三次插值仅用于平滑展示，不会写回上述定量速度模型。",
            "",
        ]),
        encoding="utf-8",
    )
    written.append(explanation)

    if include_workface is None:
        include_workface = bool(
            image_source_dir is not None
            and any(
                next(image_source_dir.glob(pattern), None) is not None
                for pattern in (
                    "key_workface_anomaly_z*.png",
                    "key_workface_velocity_z*.png",
                    "vendor_style_velocity_z*.png",
                    "vendor_style_anomaly_z*.png",
                )
            )
        )
    copied_images = (
        package_result_images(
            image_source_dir,
            output_dir,
            include_workface=include_workface,
        )
        if image_source_dir is not None else []
    )
    core_files = [
        model_npz,
        output_dir / "slice_report.txt",
        output_dir / "source_time_corrections.csv",
        output_dir / "auto_strategy_report.json",
        output_dir / "auto_strategy_candidates.csv",
        output_dir / "auto_strategy_cv_runs.csv",
        output_dir / "wave_ct_project_snapshot.json",
        output_dir / "反演参数报告.txt",
        output_dir / "棋盘格恢复测试指标.json",
        output_dir / "重点异常区验证建议.md",
        output_dir / "最终版算法与成果说明.md",
    ]
    evidence_files: list[Path] = []
    for evidence_dir_name in ("physics_gate", "参考结果对比"):
        evidence_dir = output_dir / evidence_dir_name
        if evidence_dir.is_dir():
            evidence_files.extend(path for path in evidence_dir.rglob("*") if path.is_file())
    manifest_files = sorted(
        {
            path.resolve()
            for path in [*written, *copied_images, *core_files, *evidence_files]
            if path.is_file()
        },
        key=lambda path: str(path).lower(),
    )
    auto_report_path = output_dir / "auto_strategy_report.json"
    selected_candidate = ""
    selection_status = ""
    if deep_enabled:
        selected_candidate = inversion_method
        selection_status = "deep_reparameterization_run"
    elif auto_report_path.is_file():
        try:
            auto_report = json.loads(auto_report_path.read_text(encoding="utf-8"))
            selected_candidate = str(auto_report.get("selected_candidate", ""))
            selection_status = str(auto_report.get("selection_status", ""))
        except (OSError, json.JSONDecodeError):
            pass
    resolved_output_dir = output_dir.resolve()

    def manifest_name(path: Path) -> str:
        """Use portable relative names only for files inside the bundle."""
        try:
            return str(path.relative_to(resolved_output_dir))
        except ValueError:
            return str(path)

    manifest = {
        "schema_version": 1,
        "bundle_type": "WaveCT_auditable_final_results",
        "model_npz": str(model_npz.resolve()),
        "model_field_for_quantitative_use": "velocity",
        "model_field_for_presentation_only": "display_velocity",
        "background_velocity_mps": background_velocity,
        "reliable_coverage": reliable_coverage,
        "requested_slice_z_m": requested_targets,
        "grid_cells": [int(xc.size), int(yc.size), int(zc.size)],
        "inversion_method": inversion_method,
        "selected_candidate": selected_candidate,
        "selection_status": selection_status,
        "grouped_cv_stability_models": stability_count,
        "grouped_cv_stability_method": stability_method,
        "files": [
            {
                "path": manifest_name(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in manifest_files
        ],
    }
    manifest_path = output_dir / "成果清单.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    final_result_dir = output_dir / FINAL_RESULT_DIR_NAME
    if final_result_dir.is_dir():
        (final_result_dir / manifest_path.name).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(f"Final result manifest: {manifest_path}")
    return manifest


def _parse_targets(text: str) -> list[float]:
    return [float(value.strip()) for value in text.split(",") if value.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="导出WaveCT完整速度模型与成果清单")
    parser.add_argument("model_npz", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--slice-z", default="")
    parser.add_argument("--image-source-dir", type=Path)
    args = parser.parse_args()
    export_model_text_bundle(
        args.model_npz,
        args.output_dir,
        _parse_targets(args.slice_z),
        image_source_dir=args.image_source_dir,
    )


if __name__ == "__main__":
    main()
