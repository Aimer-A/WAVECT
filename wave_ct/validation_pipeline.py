"""Reusable engineering-validation pipeline for a completed WaveCT inversion.

The pipeline deliberately separates numerical evidence, predictive evidence,
waveform support and independent field evidence.  Missing evidence is reported
as SKIPPED instead of being silently treated as a successful validation.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

from wave_ct.inversion import build_siddon_matrix


STATUS_ORDER = {"PASS": 0, "WARN": 1, "SKIPPED": 2, "FAIL": 3}


@dataclass
class StageResult:
    name: str
    status: str
    summary: str
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)


@dataclass
class ValidationConfig:
    input_csv: Path
    model_npz: Path
    output_dir: Path
    detail_csv: Path | None = None
    waveform_root: Path | None = None
    evidence_csv: Path | None = None
    slice_report: Path | None = None
    anomaly_limit: float = 0.30


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_value) + "\n",
        encoding="utf-8",
    )


def _read_rows(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _number(row: dict[str, str], *keys: str) -> float:
    for key in keys:
        raw = row.get(key)
        if raw not in (None, ""):
            return float(raw)
    raise KeyError("/".join(keys))


def _load_observations(path: Path) -> dict[str, np.ndarray]:
    rows = _read_rows(path)
    source, receiver, time_s, source_id = [], [], [], []
    valid_indices = []
    for index, row in enumerate(rows):
        try:
            source.append([
                _number(row, "震源坐标-x", "source_x"),
                _number(row, "震源坐标-y", "source_y"),
                _number(row, "震源坐标-z", "source_z"),
            ])
            receiver.append([
                _number(row, "台站坐标-x", "station_x"),
                _number(row, "台站坐标-y", "station_y"),
                _number(row, "台站坐标-z", "station_z"),
            ])
            if (row.get("travel_time_sec") or "").strip():
                time_s.append(float(row["travel_time_sec"]))
            else:
                # The legacy Chinese WaveCT contract stores this column in ms.
                time_s.append(_number(row, "震源-台站传播时间") / 1000.0)
            source_id.append(row.get("震源编号") or row.get("source_id") or f"row-{index}")
            valid_indices.append(index)
        except (KeyError, TypeError, ValueError):
            continue
    if not time_s:
        raise ValueError(f"走时CSV没有有效观测记录: {path}")
    return {
        "source": np.asarray(source, dtype=float),
        "receiver": np.asarray(receiver, dtype=float),
        "time_s": np.asarray(time_s, dtype=float),
        "source_id": np.asarray(source_id, dtype=str),
        "row_index": np.asarray(valid_indices, dtype=int),
    }


def _model_arrays(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as model:
        result = {key: np.asarray(model[key]) for key in model.files}
    required = {"velocity", "xc", "yc", "zc", "xnodes", "ynodes", "znodes"}
    missing = required.difference(result)
    if missing:
        raise ValueError(f"模型缺少必要字段: {', '.join(sorted(missing))}")
    if "background_velocity_mps" not in result:
        result["background_velocity_mps"] = np.asarray(
            float(np.nanmedian(result["velocity"]))
        )
    return result


def validate_contract(config: ValidationConfig, obs: dict[str, np.ndarray],
                      model: dict[str, Any]) -> StageResult:
    velocity = np.asarray(model["velocity"], dtype=float)
    expected = (len(model["xc"]), len(model["yc"]), len(model["zc"]))
    problems = []
    if velocity.shape != expected:
        problems.append(f"velocity形状{velocity.shape}与坐标{expected}不一致")
    if not np.all(np.isfinite(velocity)):
        problems.append("velocity包含NaN或无穷值")
    metrics = {
        "observations": int(obs["time_s"].size),
        "events": int(np.unique(obs["source_id"]).size),
        "grid_shape": list(expected),
        "velocity_min_mps": float(np.nanmin(velocity)),
        "velocity_max_mps": float(np.nanmax(velocity)),
        "background_velocity_mps": float(model["background_velocity_mps"]),
    }
    return StageResult(
        "数据与模型契约", "FAIL" if problems else "PASS",
        "；".join(problems) if problems else "输入字段、模型维度和数值有限性检查通过。",
        metrics=metrics,
    )


def validate_residuals(config: ValidationConfig, obs: dict[str, np.ndarray],
                       model: dict[str, Any]) -> StageResult:
    used = np.asarray(model.get("used_observation_row_indices", []), dtype=int)
    select = used[(used >= 0) & (used < obs["time_s"].size)]
    if select.size == 0:
        select = np.arange(obs["time_s"].size)
    src, rec, observed = obs["source"][select], obs["receiver"][select], obs["time_s"][select]
    velocity = np.asarray(model["velocity"], dtype=float)
    nx, ny, nz = velocity.shape
    matrix, _ = build_siddon_matrix(
        src[:, 0], src[:, 1], src[:, 2], rec[:, 0], rec[:, 1], rec[:, 2],
        model["xnodes"], model["ynodes"], model["znodes"], nx, ny, nz,
    )
    calculated = matrix @ (1.0 / velocity).reshape(-1, order="F")
    inside_length = np.asarray(matrix.sum(axis=1)).reshape(-1)
    full_length = np.linalg.norm(rec - src, axis=1)
    outside_length = np.maximum(full_length - inside_length, 0.0)
    calculated += outside_length / float(model["background_velocity_mps"])
    calculated += float(np.asarray(model.get("global_time_correction_s", 0.0)))
    correction_ids = np.asarray(model.get("source_ids", []), dtype=str)
    corrections = np.asarray(model.get("source_time_deviations_s", []), dtype=float)
    if correction_ids.size == corrections.size and correction_ids.size:
        correction_map = dict(zip(correction_ids.tolist(), corrections.tolist()))
        calculated += np.asarray(
            [correction_map.get(value, 0.0) for value in obs["source_id"][select]],
            dtype=float,
        )
    residual_ms = (observed - calculated) * 1000.0
    median = float(np.median(residual_ms))
    mad = float(np.median(np.abs(residual_ms - median)))
    robust_sigma = max(1.4826 * mad, 1e-9)
    tail = np.abs(residual_ms - median) > 3.0 * robust_sigma
    distance = np.linalg.norm(rec - src, axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].scatter(distance, residual_ms, s=9, alpha=0.48)
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set(xlabel="震源-台站距离 (m)", ylabel="走时残差 (ms)", title="最终模型走时残差")
    axes[1].hist(residual_ms, bins=40, color="#2878b5", alpha=0.85)
    axes[1].axvline(median, color="black", ls="--", lw=1)
    axes[1].set(xlabel="走时残差 (ms)", ylabel="数量", title="残差分布")
    fig.tight_layout()
    target = config.output_dir / "01_走时残差诊断.png"
    fig.savefig(target, dpi=200, bbox_inches="tight")
    plt.close(fig)

    rms = float(np.sqrt(np.mean(residual_ms ** 2)))
    status = "WARN" if rms > 25.0 or np.mean(tail) > 0.05 else "PASS"
    return StageResult(
        "走时残差诊断", status,
        f"RMS={rms:.3f} ms，3倍稳健尺度尾部占比={np.mean(tail)*100:.1f}%。",
        metrics={
            "rms_ms": rms, "median_ms": median, "robust_sigma_ms": robust_sigma,
            "tail_fraction": float(np.mean(tail)), "observations_used": int(select.size),
        },
        artifacts=[target.name],
        limitations=["残差变小本身不能证明异常具有地质成因。"],
    )


def validate_group_holdout(config: ValidationConfig, model: dict[str, Any]) -> StageResult:
    validation_ids = np.asarray(model.get("validation_source_ids", []))
    candidates = sorted(config.output_dir.glob("*严格*实验*.json"))
    if validation_ids.size == 0 and not candidates:
        return StageResult(
            "按事件独立留出", "SKIPPED",
            "当前正式模型未记录独立留出事件，也未发现可复用的严格留出实验结果。",
            limitations=["训练RMS不能替代独立事件预测RMS。"],
        )
    metrics: dict[str, Any] = {"validation_source_ids": validation_ids.tolist()}
    artifacts = []
    for path in candidates:
        try:
            metrics.setdefault("external_experiments", []).append(
                {"file": str(path), "result": json.loads(path.read_text(encoding="utf-8"))}
            )
            artifacts.append(str(path))
        except (OSError, json.JSONDecodeError):
            continue
    return StageResult(
        "按事件独立留出", "PASS" if validation_ids.size else "WARN",
        f"正式模型记录了{validation_ids.size}个留出事件。"
        if validation_ids.size else "发现既有严格实验文件，但正式模型本身未保留独立事件。",
        metrics=metrics, artifacts=artifacts,
    )


def validate_resolution(config: ValidationConfig) -> StageResult:
    """Import a checkerboard result produced with the same output bundle."""
    search_roots = (config.output_dir, config.model_npz.parent)
    candidates: list[Path] = []
    seen: set[Path] = set()
    for root in search_roots:
        if not root.is_dir():
            continue
        for path in root.glob("*.json"):
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                candidates.append(path)
        for path in root.glob("*/*.json"):
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                candidates.append(path)
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if "checkerboard" not in str(payload.get("test_type", "")).lower():
            continue
        correlation = payload.get("correlation_supported_cells")
        sign_fraction = payload.get("sign_recovery_fraction_supported_cells")
        supported = payload.get("ray_supported_cells")
        status = "PASS" if isinstance(correlation, (int, float)) and correlation >= 0.6 else "WARN"
        return StageResult(
            "棋盘格分辨率", status,
            f"覆盖单元相关系数={correlation}，符号恢复率={sign_fraction}，"
            f"有射线支持单元={supported}。",
            metrics=payload, artifacts=[path.name],
            limitations=["棋盘格只检验当前射线几何的空间恢复能力，不能证明实测异常是地质体。"],
        )
    return StageResult(
        "棋盘格分辨率", "SKIPPED",
        "当前成果目录未发现与本模型绑定的棋盘格指标JSON。",
        limitations=["发布定量解释前应使用相同射线几何和网格执行棋盘格测试。"],
    )


def validate_vertical_continuity(config: ValidationConfig,
                                 model: dict[str, Any]) -> StageResult:
    # Validation must use the quantitative forward model.  Presentation-only
    # smoothing or coverage blending must never improve a validation verdict.
    velocity = np.asarray(model["velocity"], dtype=float)
    background = float(model["background_velocity_mps"])
    anomaly = (velocity - background) / background
    support = np.asarray(model.get("coverage_weight", np.ones_like(velocity)), dtype=float)
    score = np.abs(anomaly) * np.clip(support, 0.0, 1.0)
    ix, iy, _ = np.unravel_index(int(np.nanargmax(score)), score.shape)
    norm = TwoSlopeNorm(vmin=-config.anomaly_limit, vcenter=0, vmax=config.anomaly_limit)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    a = axes[0].pcolormesh(model["xc"], model["zc"], anomaly[:, iy, :].T,
                           cmap="jet", norm=norm, shading="auto")
    axes[0].set(title=f"X-Z剖面（Y={model['yc'][iy]:.2f} m）",
                xlabel="X (m)", ylabel="Z (m)")
    axes[1].pcolormesh(model["yc"], model["zc"], anomaly[ix, :, :].T,
                       cmap="jet", norm=norm, shading="auto")
    axes[1].set(title=f"Y-Z剖面（X={model['xc'][ix]:.2f} m）",
                xlabel="Y (m)", ylabel="Z (m)")
    fig.colorbar(a, ax=axes.ravel().tolist(), label="相对速度异常 An")
    fig.subplots_adjust(wspace=0.28, right=0.90)
    target = config.output_dir / "02_异常体XZ_YZ连续性.png"
    fig.savefig(target, dpi=210, bbox_inches="tight")
    plt.close(fig)

    correlations = []
    for k in range(anomaly.shape[2] - 1):
        mask = (support[:, :, k] > 0.2) & (support[:, :, k + 1] > 0.2)
        if np.count_nonzero(mask) >= 3:
            correlations.append(float(np.corrcoef(
                anomaly[:, :, k][mask], anomaly[:, :, k + 1][mask]
            )[0, 1]))
    finite = [v for v in correlations if np.isfinite(v)]
    mean_corr = float(np.mean(finite)) if finite else float("nan")
    status = "PASS" if finite and mean_corr >= 0.5 else "WARN"
    return StageResult(
        "三维切片连续性", status,
        f"相邻层覆盖区平均相关系数={mean_corr:.3f}。" if finite else "有效相邻层不足，不能量化连续性。",
        metrics={"adjacent_layer_correlations": correlations,
                 "section_x_m": float(model["xc"][ix]),
                 "section_y_m": float(model["yc"][iy])},
        artifacts=[target.name],
        limitations=[f"模型仅有{anomaly.shape[2]}个Z层，剖面是离散层间展示，不等于高分辨率三维成像。"],
    )


def validate_perturbation(config: ValidationConfig, model: dict[str, Any]) -> StageResult:
    ensemble = np.asarray(model.get("deep_ensemble_velocity", []), dtype=float)
    if ensemble.ndim != 4 or ensemble.shape[0] < 2:
        return StageResult(
            "数值扰动稳定性", "SKIPPED",
            "模型未保存多起点集合，无法从正式成果量化初始化扰动稳定性。",
            limitations=["建议另行执行初速度±5%、拾取±2 ms和网格变化的严格重反演实验。"],
        )
    background = float(model["background_velocity_mps"])
    members = (ensemble - background) / background
    reference = np.mean(members, axis=0)
    correlations = []
    for member in members:
        left, right = reference.ravel(), member.ravel()
        if np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
            correlations.append(1.0 if np.allclose(left, right, rtol=0, atol=1e-6) else 0.0)
        else:
            correlations.append(float(np.corrcoef(left, right)[0, 1]))
    mean_corr = float(np.mean(correlations))
    return StageResult(
        "数值扰动稳定性", "PASS" if mean_corr >= 0.8 else "WARN",
        f"{ensemble.shape[0]}个DNR起点与集合均值的平均相关系数={mean_corr:.3f}。",
        metrics={"ensemble_members": int(ensemble.shape[0]),
                 "member_correlations": correlations, "mean_correlation": mean_corr},
        limitations=["多起点稳定性不替代初速度、拾取误差和网格尺寸的独立敏感性实验。"],
    )


def validate_waveforms(config: ValidationConfig) -> StageResult:
    details = _read_rows(config.detail_csv)
    root = config.waveform_root
    files = list(root.rglob("*.semv")) if root and root.is_dir() else []
    if not files:
        proprietary = (
            list(root.rglob("*.W")) + list(root.rglob("*.W2"))
            if root and root.is_dir()
            else []
        )
        if proprietary:
            return StageResult(
                "波形级支持",
                "SKIPPED",
                (
                    f"发现{len(proprietary)}个SOS波形容器，但当前仅验证了走时标记，"
                    "没有可审计的完整波形正演输入/解码链。"
                ),
                metrics={
                    "detail_rows": len(details),
                    "waveform_files": len(proprietary),
                    "amplitude_codec_status": "unsupported_for_full_waveform_validation",
                },
                limitations=[
                    "人工P标记可用于走时CT，但不能替代振幅、相位和后续能量拟合。",
                    "完整波形结论还需要厂商振幅编码、震源子波、介质参数、边界条件和求解器配置。",
                ],
            )
        return StageResult(
            "波形级支持", "SKIPPED", "没有发现可读取的原始波形文件。",
            metrics={"detail_rows": len(details), "waveform_files": 0},
            limitations=["没有波形时只能验证走时，不能验证振幅、相位或后续能量。"],
        )
    linked = 0
    names = {path.name.lower() for path in files}
    for row in details:
        if (row.get("file_name") or "").lower() in names:
            linked += 1
    return StageResult(
        "波形级支持", "WARN",
        f"发现{len(files)}个波形文件，明细表可关联{linked}条；当前完成数据可用性检查。",
        metrics={"detail_rows": len(details), "waveform_files": len(files),
                 "linked_detail_rows": linked},
        limitations=[
            "完整波形正演还需要震源子波、密度/衰减、边界条件和正演求解器配置。",
            "不得把走时窗口对齐称为完整波形拟合。",
        ],
    )


def _write_evidence_template(path: Path) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["evidence_type", "x", "y", "z", "value", "label", "source", "date", "notes"])
        writer.writerow(["borehole_or_pressure_or_gas", "", "", "", "", "", "", "", ""])


def validate_field_evidence(config: ValidationConfig) -> StageResult:
    rows = _read_rows(config.evidence_csv)
    if not rows:
        template = config.output_dir / "现场独立证据登记模板.csv"
        _write_evidence_template(template)
        return StageResult(
            "现场独立证据", "SKIPPED",
            "未提供钻孔、矿压、瓦斯、揭露或微震等独立证据；已生成通用登记模板。",
            artifacts=[template.name],
            limitations=["反演异常目前只能解释为速度异常，不能直接命名为断层、应力集中或裂隙带。"],
        )
    valid = sum(
        1 for row in rows
        if all((row.get(key) or "").strip() for key in ("evidence_type", "x", "y", "z"))
    )
    return StageResult(
        "现场独立证据", "WARN" if valid < 3 else "PASS",
        f"读取{len(rows)}条登记记录，其中{valid}条具备类型和三维坐标。",
        metrics={"records": len(rows), "spatially_valid_records": valid},
        limitations=["空间重合表示关联，不自动证明因果关系。"],
    )


def _overall_status(stages: Iterable[StageResult]) -> str:
    values = list(stages)
    if any(stage.status == "FAIL" for stage in values):
        return "FAIL"
    if any(stage.status in {"WARN", "SKIPPED"} for stage in values):
        return "PARTIAL"
    return "PASS"


def _write_report(config: ValidationConfig, stages: list[StageResult]) -> Path:
    overall = _overall_status(stages)
    lines = [
        "# WaveCT 工程可信度验证报告", "",
        f"- 生成时间：{datetime.now().isoformat(timespec='seconds')}",
        f"- 综合状态：**{overall}**",
        f"- 反演模型：`{config.model_npz}`",
        f"- 走时数据：`{config.input_csv}`", "",
        "## 验证链结论", "",
        "| 阶段 | 状态 | 结论 |", "|---|---|---|",
    ]
    for stage in stages:
        lines.append(f"| {stage.name} | {stage.status} | {stage.summary.replace('|', '／')} |")
    lines.extend(["", "## 证据边界", ""])
    for stage in stages:
        for item in stage.limitations:
            lines.append(f"- {stage.name}：{item}")
    lines.extend([
        "", "## 工程判定原则", "",
        "- PASS 表示该项检查满足当前阈值，不代表异常已被证明为某种地质构造。",
        "- WARN 表示已有支持证据但仍存在明显限制。",
        "- SKIPPED 表示缺少输入或尚未执行，不能按通过处理。",
        "- 只有独立数据复现、波形/走时改善和现场证据共同指向同一区域时，才建议提升解释置信度。",
    ])
    target = config.output_dir / "工程可信度验证报告.md"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def run_pipeline(config: ValidationConfig) -> dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    obs = _load_observations(config.input_csv)
    model = _model_arrays(config.model_npz)
    stages = [
        validate_contract(config, obs, model),
        validate_residuals(config, obs, model),
        validate_group_holdout(config, model),
        validate_resolution(config),
        validate_perturbation(config, model),
        validate_vertical_continuity(config, model),
        validate_waveforms(config),
        validate_field_evidence(config),
    ]
    report = _write_report(config, stages)
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "overall_status": _overall_status(stages),
        "config": asdict(config),
        "stages": [asdict(stage) for stage in stages],
        "report": report.name,
    }
    summary = config.output_dir / "validation_summary.json"
    _write_json(summary, payload)
    print(f"验证链综合状态: {payload['overall_status']}")
    for stage in stages:
        print(f"[{stage.status:7}] {stage.name}: {stage.summary}")
    print(f"验证报告: {report}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="运行通用WaveCT工程可信度验证链")
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--model-npz", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--detail-csv", type=Path)
    parser.add_argument("--waveform-root", type=Path)
    parser.add_argument("--evidence-csv", type=Path)
    parser.add_argument("--slice-report", type=Path)
    parser.add_argument("--anomaly-limit", type=float, default=0.30)
    args = parser.parse_args()
    model = args.model_npz or args.out_dir / "velocity_model.npz"
    run_pipeline(ValidationConfig(
        input_csv=args.input_csv, model_npz=model, output_dir=args.out_dir,
        detail_csv=args.detail_csv, waveform_root=args.waveform_root,
        evidence_csv=args.evidence_csv, slice_report=args.slice_report,
        anomaly_limit=args.anomaly_limit,
    ))


if __name__ == "__main__":
    main()
