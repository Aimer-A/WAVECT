"""Pilot-CV algorithm selection wrapper for WaveCT inversions."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from wave_ct.auto_strategy import (
    add_coverage_features,
    candidate_priorities,
    extract_dataset_features,
    recommend_inversion_controls,
)
from wave_ct.tools.fat_ray_report import model_diagnostics, read_key_values


VALUE_OPTIONS = {
    "--input-csv", "--output-dir", "--n-outer", "--n-lsqr", "--random-seed",
    "--validation-source-ids", "--alpha-reg", "--wavelet-levels",
    "--wavelet-threshold-factor", "--hierarchical-split-rays",
    "--hierarchical-min-block-x", "--hierarchical-min-block-y",
    "--differential-weight",
    "--huber-delta", "--min-ray-coverage",
    "--model-damping",
    "--curvature-reg-factor", "--curvature-z-factor", "--spline-control-nx",
    "--spline-control-ny", "--spline-projection-strength",
}
FLAG_OPTIONS = {
    "--joint-sparsity", "--no-joint-sparsity", "--hierarchical-parameterization",
    "--no-hierarchical-parameterization", "--regularize-total-model",
    "--no-regularize-total-model", "--spline-projection", "--no-spline-projection",
    "--differential-times", "--no-differential-times",
    "--export-deliverables", "--no-export-deliverables",
}


def option_value(arguments: Sequence[str], option: str, default: str) -> str:
    for index, token in enumerate(arguments[:-1]):
        if token == option:
            return arguments[index + 1]
    return default


def strip_controlled_options(arguments: Sequence[str]) -> List[str]:
    cleaned: List[str] = []
    index = 0
    while index < len(arguments):
        token = arguments[index]
        if token in VALUE_OPTIONS:
            index += 2
            continue
        if token in FLAG_OPTIONS:
            index += 1
            continue
        cleaned.append(token)
        index += 1
    return cleaned


def parse_nodes_and_bounds(arguments: Sequence[str]) -> tuple[list[float] | None, list[int] | None]:
    try:
        bounds = [
            float(option_value(arguments, "--x-min", "nan")),
            float(option_value(arguments, "--x-max", "nan")),
            float(option_value(arguments, "--y-min", "nan")),
            float(option_value(arguments, "--y-max", "nan")),
            float(option_value(arguments, "--z-min", "nan")),
            float(option_value(arguments, "--z-max", "nan")),
        ]
        nodes = [
            int(option_value(arguments, "--nx-nodes", "0")),
            int(option_value(arguments, "--ny-nodes", "0")),
            int(option_value(arguments, "--nz-nodes", "0")),
        ]
    except ValueError:
        return None, None
    if not np.all(np.isfinite(bounds)) or any(
        high <= low for low, high in zip(bounds[::2], bounds[1::2])
    ):
        bounds = None
    if any(value < 2 for value in nodes):
        nodes = None
    return bounds, nodes


def candidate_definitions(features: Dict[str, object]) -> Dict[str, List[str]]:
    coverage = float(features.get("covered_cell_fraction", 0.3))
    # Event-centred velocity dispersion is saved as a diagnostic, but the raw
    # dispersion remains the production mapping after the Stage-7 720-data
    # pilot showed no held-out benefit from replacing it.
    heterogeneity = float(features.get("apparent_velocity_robust_cv", 0.1))
    controls = recommend_inversion_controls(features)
    base_alpha = float(controls["alpha_reg"])
    huber_delta_sigma = float(controls["huber_delta_sigma"])
    reliable_coverage = int(controls["min_ray_coverage"])
    # Coarse hierarchical blocks need at least the same smoothing strength as
    # the full grid; weakening it caused boundary-saturated mine-data pilots.
    hierarchical_alpha = float(np.clip(base_alpha, 1.5, 4.0))
    threshold = int(features.get("recommended_hierarchical_split_rays", 5))
    minimum_block = int(features.get("recommended_hierarchical_min_block", 3))
    candidates = {
        "tv": [
            "--edge-preserving-tv", "--no-joint-sparsity",
            "--no-hierarchical-parameterization", "--no-differential-times",
            "--model-damping", "0", "--alpha-reg", f"{base_alpha:.6g}",
            "--huber-delta", f"{huber_delta_sigma:.8g}",
            "--min-ray-coverage", str(reliable_coverage),
        ],
        "tv_haar": [
            "--edge-preserving-tv", "--joint-sparsity", "--wavelet-levels", "2",
            "--wavelet-threshold-factor", "0.8", "--no-hierarchical-parameterization",
            "--no-differential-times",
            "--model-damping", "0", "--alpha-reg", f"{base_alpha:.6g}",
            "--huber-delta", f"{huber_delta_sigma:.8g}",
            "--min-ray-coverage", str(reliable_coverage),
        ],
        "hierarchical_tv": [
            "--edge-preserving-tv", "--no-joint-sparsity", "--hierarchical-parameterization",
            "--hierarchical-split-rays", str(threshold),
            "--hierarchical-min-block-x", str(minimum_block),
            "--hierarchical-min-block-y", str(minimum_block),
            "--no-differential-times",
            "--model-damping", "0", "--alpha-reg", f"{hierarchical_alpha:.6g}",
            "--huber-delta", f"{huber_delta_sigma:.8g}",
            "--min-ray-coverage", str(reliable_coverage),
        ],
    }
    # Large per-event timing offsets are common in mine data.  Common-source
    # differential equations suppress those nuisance offsets while retaining
    # the absolute-time equations and source-static model.
    if bool(controls["enable_differential_times"]):
        candidates.update({
            "differential_tv": [
                "--edge-preserving-tv", "--no-joint-sparsity",
                "--no-hierarchical-parameterization", "--differential-times",
                "--differential-weight", "1.0", "--model-damping", "0",
                "--alpha-reg", f"{base_alpha:.6g}",
                "--huber-delta", f"{huber_delta_sigma:.8g}",
                "--min-ray-coverage", str(reliable_coverage),
            ],
            "differential_hierarchical_tv": [
                "--edge-preserving-tv", "--no-joint-sparsity",
                "--hierarchical-parameterization", "--differential-times",
                "--differential-weight", "1.0",
                "--hierarchical-split-rays", str(threshold),
                "--hierarchical-min-block-x", str(minimum_block),
                "--hierarchical-min-block-y", str(minimum_block),
                "--model-damping", "0", "--alpha-reg", f"{hierarchical_alpha:.6g}",
                "--huber-delta", f"{huber_delta_sigma:.8g}",
                "--min-ray-coverage", str(reliable_coverage),
            ],
        })
    if coverage < 0.35:
        candidates["damped_hierarchical_tv"] = [
            "--edge-preserving-tv", "--no-joint-sparsity",
            "--hierarchical-parameterization", "--no-differential-times",
            "--hierarchical-split-rays", str(threshold),
            "--hierarchical-min-block-x", str(minimum_block),
            "--hierarchical-min-block-y", str(minimum_block),
            "--model-damping", "20", "--alpha-reg", f"{hierarchical_alpha:.6g}",
                "--huber-delta", f"{huber_delta_sigma:.8g}",
            "--min-ray-coverage", str(reliable_coverage),
        ]
    return candidates


def collect_run(directory: Path) -> Dict[str, float]:
    values = read_key_values(directory / "slice_report.txt")
    diagnostics = model_diagnostics(directory / "velocity_model.npz")
    with np.load(directory / "velocity_model.npz", allow_pickle=False) as model:
        velocity = np.asarray(model["velocity"], dtype=np.float64)
        background_velocity = float(model["background_velocity_mps"])
    allowed_range = values.get("model_velocity_range", "nan-nan").split("-", 1)
    try:
        allowed_min, allowed_max = (float(value) for value in allowed_range)
        tolerance = max(allowed_max - allowed_min, 1.0) * 1e-8
        clipped_fraction = float(np.mean(
            (velocity <= allowed_min + tolerance) | (velocity >= allowed_max - tolerance)
        ))
    except (TypeError, ValueError):
        clipped_fraction = float("nan")
    return {
        "ray_count": float(values.get("n_rays", "nan")),
        "train_rms_ms": 1000.0 * float(values["solution_train_rms_s"]),
        "validation_rms_ms": 1000.0 * float(values["solution_validation_rms_s"]),
        "roughness_reliable_mps": float(diagnostics["lateral_roughness_reliable_mps"]),
        "anomaly_std_reliable_mps": float(diagnostics["velocity_std_reliable_mps"]),
        "covered_fraction": float(diagnostics["centerline_covered_fraction"]),
        "model_parameters": float(values.get("hierarchical_model_parameters", "nan")),
        "velocity_clip_fraction": clipped_fraction,
        "velocity_p01_mps": float(np.percentile(velocity, 1.0)),
        "velocity_p99_mps": float(np.percentile(velocity, 99.0)),
        "background_velocity_mps": background_velocity,
        **_anomaly_geometry(directory / "velocity_model.npz"),
    }


def _anomaly_geometry(model_path: Path) -> Dict[str, float]:
    """Measure where the reliably supported anomaly is, independently of plotting."""
    with np.load(model_path, allow_pickle=False) as model:
        velocity = np.asarray(model["velocity"], dtype=np.float64)
        background = float(model["background_velocity_mps"])
        density = np.asarray(model["ray_density"], dtype=np.float64)
        xc = np.asarray(model["xc"], dtype=np.float64)
        yc = np.asarray(model["yc"], dtype=np.float64)
        zc = np.asarray(model["zc"], dtype=np.float64)
    anomaly = np.abs(velocity - background)
    positive = density[density > 0]
    coverage_threshold = max(3.0, float(np.percentile(positive, 50.0))) if positive.size else 3.0
    reliable = density >= coverage_threshold
    supported = anomaly[reliable]
    threshold = max(float(np.percentile(supported, 70.0)), 0.01 * background) if supported.size else np.inf
    weights = np.where(reliable & (anomaly >= threshold), anomaly, 0.0)
    total = float(np.sum(weights))
    if total <= 0.0:
        return {"anomaly_centroid_x": float("nan"), "anomaly_centroid_y": float("nan"),
                "anomaly_centroid_z": float("nan"), "anomaly_support_fraction": 0.0}
    xx, yy, zz = np.meshgrid(xc, yc, zc, indexing="ij")
    return {
        "anomaly_centroid_x": float(np.sum(weights * xx) / total),
        "anomaly_centroid_y": float(np.sum(weights * yy) / total),
        "anomaly_centroid_z": float(np.sum(weights * zz) / total),
        "anomaly_support_fraction": float(np.mean(weights > 0.0)),
    }


def summarize_candidates(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    names = sorted({str(row["candidate"]) for row in rows})
    # Plain TV is the neutral amplitude reference.  Haar is itself a shrinkage
    # operator, so using it as the reference can falsely label an unsuppressed
    # but data-supported anomaly as amplitude inflation.
    baseline_name = (
        "tv" if "tv" in names
        else "tv_haar" if "tv_haar" in names
        else names[0]
    )
    baseline_by_seed = {
        int(row["seed"]): float(row["anomaly_std_reliable_mps"])
        for row in rows if row["candidate"] == baseline_name
    }
    positive_baseline = [value for value in baseline_by_seed.values() if value >= 20.0]
    baseline_floor = 0.25 * float(np.median(positive_baseline)) if positive_baseline else 20.0
    baseline_stable = len(positive_baseline) >= max(2, int(np.ceil(2.0 * len(baseline_by_seed) / 3.0)))
    baseline_rough = float(np.mean([
        float(row["roughness_reliable_mps"])
        for row in rows if row["candidate"] == baseline_name
    ]))
    summaries: List[Dict[str, object]] = []
    for name in names:
        part = [row for row in rows if row["candidate"] == name]
        validation = np.asarray([float(row["validation_rms_ms"]) for row in part])
        roughness = float(np.mean([float(row["roughness_reliable_mps"]) for row in part]))
        anomaly = float(np.mean([float(row["anomaly_std_reliable_mps"]) for row in part]))
        anomaly_values = np.asarray([
            float(row["anomaly_std_reliable_mps"]) for row in part
        ])
        anomaly_cv = float(np.std(anomaly_values) / max(anomaly, 1e-12))
        background_velocity = float(np.mean([
            float(row.get("background_velocity_mps", 1.0)) for row in part
        ]))
        anomaly_to_background = anomaly / max(background_velocity, 1e-12)
        retention_by_seed = [
            100.0 * float(row["anomaly_std_reliable_mps"])
            / max(baseline_by_seed[int(row["seed"])], baseline_floor)
            for row in part
        ]
        retention = float(np.mean(retention_by_seed))
        minimum_retention = float(np.min(retention_by_seed))
        mean_clip_fraction = float(np.nanmean([
            float(row.get("velocity_clip_fraction", 0.0)) for row in part
        ]))
        max_clip_fraction = float(np.nanmax([
            float(row.get("velocity_clip_fraction", 0.0)) for row in part
        ]))
        nontrivial_model_fraction = float(np.mean([
            float(row["anomaly_std_reliable_mps"]) >= 20.0 for row in part
        ]))
        covered_fraction_values = np.asarray([
            float(row.get("covered_fraction", np.nan)) for row in part
        ], dtype=float)
        ray_count_values = np.asarray([
            float(row.get("ray_count", np.nan)) for row in part
        ], dtype=float)
        finite_covered = covered_fraction_values[np.isfinite(covered_fraction_values)]
        finite_ray_count = ray_count_values[np.isfinite(ray_count_values)]
        mean_covered_fraction = (
            float(np.mean(finite_covered)) if finite_covered.size else float("nan")
        )
        mean_ray_count = (
            float(np.mean(finite_ray_count)) if finite_ray_count.size else float("nan")
        )
        mean_model_parameters = float(np.nanmean([
            float(row["model_parameters"]) for row in part
        ]))
        parameter_to_ray_ratio = (
            mean_model_parameters / max(mean_ray_count, 1.0)
            if np.isfinite(mean_ray_count) else float("nan")
        )
        # A low-coverage full grid can still be made algebraically unique by
        # Tikhonov/background terms, but its cell-wise anomalies are then
        # mostly prior-driven.  Do not let a lower training RMS alone select
        # such a candidate.  Missing diagnostics are treated conservatively
        # for backwards-compatible unit-test/imported pilot rows.
        supportability_eligible = True
        if np.isfinite(mean_covered_fraction) and np.isfinite(parameter_to_ray_ratio):
            supportability_eligible = not (
                mean_covered_fraction < 0.10
                and parameter_to_ray_ratio > 2.0
            )
        centroid = np.asarray([[float(row.get(f"anomaly_centroid_{axis}", np.nan))
                                for axis in "xyz"] for row in part], dtype=float)
        finite_centroid = np.all(np.isfinite(centroid), axis=1)
        if np.count_nonzero(finite_centroid) >= 2:
            spans = np.ptp(centroid[finite_centroid], axis=0)
            centroid_spread = float(np.linalg.norm(spans))
            centroid_scale = max(float(np.linalg.norm(np.nanmean(np.abs(centroid[finite_centroid]), axis=0))), 1.0)
            centroid_spread_relative = centroid_spread / centroid_scale
        else:
            # Older/imported pilot rows do not carry geometry fields.  Keep
            # them usable, while all newly collected runs are measured above.
            centroid_spread_relative = 0.0 if not any(
                "anomaly_centroid_x" in row for row in part
            ) else 1.0
        summaries.append(
            {
                "candidate": name,
                "mean_validation_rms_ms": float(np.mean(validation)),
                "std_validation_rms_ms": float(np.std(validation)),
                "worst_validation_rms_ms": float(np.max(validation)),
                "mean_train_rms_ms": float(np.mean([float(row["train_rms_ms"]) for row in part])),
                "mean_roughness_reliable_mps": roughness,
                "roughness_reduction_vs_tv_percent": 100.0 * (baseline_rough - roughness) / max(baseline_rough, 1e-12),
                "mean_anomaly_std_reliable_mps": anomaly,
                "anomaly_std_cv_across_splits": anomaly_cv,
                "anomaly_std_to_background_ratio": anomaly_to_background,
                "anomaly_std_retention_vs_tv_percent": retention,
                "conservative_minimum_retention_percent": minimum_retention,
                "mean_model_parameters": mean_model_parameters,
                "mean_covered_fraction": mean_covered_fraction,
                "mean_ray_count": mean_ray_count,
                "model_parameters_per_ray": parameter_to_ray_ratio,
                "supportability_eligible": supportability_eligible,
                "mean_velocity_clip_fraction": mean_clip_fraction,
                "max_velocity_clip_fraction": max_clip_fraction,
                "nontrivial_model_fraction": nontrivial_model_fraction,
                "anomaly_centroid_spread_relative": centroid_spread_relative,
                "amplitude_reference_stable": baseline_stable,
                "eligible_amplitude": bool(
                    (not baseline_stable or (
                        70.0 <= retention <= 200.0
                        and minimum_retention >= 50.0
                    ))
                    and anomaly_cv <= 0.50
                    and anomaly_to_background <= 0.50
                    and mean_clip_fraction <= 0.10
                    and max_clip_fraction <= 0.25
                    and nontrivial_model_fraction >= (2.0 / 3.0)
                    and centroid_spread_relative <= 0.10
                ),
            }
        )
    best_validation = min(float(row["mean_validation_rms_ms"]) for row in summaries)
    for row in summaries:
        dispersion = float(row["std_validation_rms_ms"]) / max(best_validation, 1e-12)
        validation_ratio = float(row["mean_validation_rms_ms"]) / max(best_validation, 1e-12)
        amplitude_penalty = 0.0 if bool(row["eligible_amplitude"]) else 0.05
        supportability_penalty = (
            0.0 if bool(row["supportability_eligible"]) else 0.25
        )
        complexity_penalty = 0.001 if row["candidate"] == "tv_haar" else 0.0
        row["supportability_penalty"] = supportability_penalty
        row["complexity_penalty"] = complexity_penalty
        row["selection_score"] = (
            validation_ratio + 0.10 * dispersion + 0.15 * min(float(row["anomaly_centroid_spread_relative"]), 1.0)
            + amplitude_penalty + supportability_penalty + complexity_penalty
        )
    return sorted(summaries, key=lambda row: float(row["selection_score"]))


def run_command(command: List[str], log_path: Path, timeout_seconds: int) -> float:
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    duration = time.perf_counter() - started
    if completed.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
        raise RuntimeError(
            f"pilot inversion failed with code {completed.returncode}:\n" + "\n".join(tail)
        )
    return duration


def interpret_eikonal_forward_gate(result: Dict[str, object]) -> Dict[str, object]:
    """Interpret the forward-only curved-ray gate without overclaiming inversion quality."""
    try:
        validation = result["metrics"]["validation"]  # type: ignore[index]
        straight = float(validation["straight_siddon"]["event_centered_rms_ms"])  # type: ignore[index]
        metric_rays = int(validation["straight_siddon"]["metric_ray_count"])  # type: ignore[index]
        alternatives = {
            str(name): float(metrics["event_centered_rms_ms"])
            for name, metrics in validation.items()  # type: ignore[union-attr]
            if str(name).startswith("eikonal_contrast_")
            and str(name).endswith("_corrected")
        }
        if metric_rays < 4 or not np.isfinite(straight) or not alternatives:
            raise ValueError("insufficient held-out metrics")
        best_name = min(alternatives, key=alternatives.get)
        best_rms = alternatives[best_name]
        improvement = 100.0 * (straight - best_rms) / max(straight, 1e-12)
        passed = bool(np.isfinite(best_rms) and improvement >= 3.0)
        return {
            "status": (
                "curved_ray_inversion_experiment_eligible"
                if passed else "straight_ray_production_retained"
            ),
            "passed_forward_gate": passed,
            "minimum_required_improvement_percent": 3.0,
            "validation_metric_rays": metric_rays,
            "straight_event_centered_rms_ms": straight,
            "best_eikonal_prediction": best_name,
            "best_eikonal_event_centered_rms_ms": best_rms,
            "improvement_percent": improvement,
            "production_action": (
                "retain selected straight-ray inversion; forward success only authorizes "
                "a future nonlinear inversion trial"
                if passed else
                "retain selected straight-ray inversion and do not promote bent rays"
            ),
            "guardrail": (
                "A forward gate cannot prove that a nonlinear bent-ray inversion "
                "recovers a better velocity model."
            ),
        }
    except (KeyError, TypeError, ValueError, AttributeError):
        return {
            "status": "gate_result_unusable_safe_fallback",
            "passed_forward_gate": False,
            "production_action": "retain selected straight-ray inversion",
        }


def run_eikonal_forward_gate(
    model_path: Path,
    input_csv: Path,
    output_dir: Path,
    timeout_seconds: int,
) -> Dict[str, object]:
    """Run a non-fatal held-out forward comparison on one selected pilot model."""
    gate_dir = output_dir / "physics_gate"
    gate_dir.mkdir(parents=True, exist_ok=True)
    result_path = gate_dir / "eikonal_forward_gate.json"
    log_path = gate_dir / "eikonal_forward_gate.log"
    command = [
        sys.executable,
        "-m",
        "wave_ct.tools.eikonal_probe",
        str(model_path),
        str(input_csv),
        str(result_path),
        "--contrast-scales",
        "0.25,0.50,0.75,1.0",
    ]
    started = time.perf_counter()
    try:
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        if completed.returncode != 0 or not result_path.is_file():
            return {
                "status": "gate_execution_failed_safe_fallback",
                "passed_forward_gate": False,
                "runtime_seconds": time.perf_counter() - started,
                "log_path": str(log_path),
                "production_action": "retain selected straight-ray inversion",
            }
        result = json.loads(result_path.read_text(encoding="utf-8"))
        interpreted = interpret_eikonal_forward_gate(result)
        interpreted.update({
            "runtime_seconds": time.perf_counter() - started,
            "result_path": str(result_path),
            "log_path": str(log_path),
        })
        return interpreted
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        return {
            "status": "gate_unavailable_safe_fallback",
            "passed_forward_gate": False,
            "runtime_seconds": time.perf_counter() - started,
            "error": str(exc),
            "log_path": str(log_path),
            "production_action": "retain selected straight-ray inversion",
        }


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="WaveCT grouped-CV automatic strategy selection")
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cv-seeds", default="11,23,41")
    parser.add_argument("--pilot-outer", type=int, default=24)
    parser.add_argument("--pilot-lsqr", type=int, default=160)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--no-final-run", action="store_true")
    parser.add_argument("--reuse-pilot", action="store_true")
    parser.add_argument(
        "--curved-ray-gate",
        action="store_true",
        default=True,
        help="在选中候选的留出事件上执行Eikonal弯曲射线前向物理检验",
    )
    parser.add_argument(
        "--no-curved-ray-gate",
        action="store_false",
        dest="curved_ray_gate",
    )
    parser.add_argument("inversion_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    inversion_args = list(args.inversion_args)
    if inversion_args[:1] == ["--"]:
        inversion_args = inversion_args[1:]
    clean_args = strip_controlled_options(inversion_args)
    bounds, nodes = parse_nodes_and_bounds(inversion_args)
    background_velocity = float(
        option_value(inversion_args, "--background-velocity", "0")
    )
    features = extract_dataset_features(
        args.input_csv,
        reference_velocity=(
            background_velocity if background_velocity > 0.0 else None
        ),
    )
    features = add_coverage_features(features, args.input_csv, bounds=bounds, nodes=nodes)
    features["recommended_inversion_controls"] = recommend_inversion_controls(features)
    features["candidate_priors"] = candidate_priorities(features)
    candidates = candidate_definitions(features)
    seeds = [int(value.strip()) for value in args.cv_seeds.split(",") if value.strip()]
    if len(seeds) < 2:
        raise ValueError("automatic selection requires at least two grouped-CV seeds")

    work_root = args.output_dir / "auto_strategy_runs"
    work_root.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    pilot_csv = args.output_dir / "auto_strategy_cv_runs.csv"
    if args.reuse_pilot:
        if not pilot_csv.is_file():
            raise FileNotFoundError(f"pilot CSV does not exist: {pilot_csv}")
        with pilot_csv.open("r", encoding="utf-8-sig", newline="") as handle:
            previous_rows = list(csv.DictReader(handle))
        previous_duration = {
            (str(row["candidate"]), int(row["seed"])): float(row["duration_s"])
            for row in previous_rows
        }
        for candidate in candidates:
            for seed in seeds:
                run_dir = work_root / f"{candidate}_s{seed}"
                rows.append({
                    "candidate": candidate,
                    "seed": seed,
                    "duration_s": previous_duration.get((candidate, seed), float("nan")),
                    **collect_run(run_dir),
                })
    else:
        for candidate, candidate_args in candidates.items():
            for seed in seeds:
                run_dir = work_root / f"{candidate}_s{seed}"
                run_dir.mkdir(parents=True, exist_ok=True)
                command = [
                    sys.executable, "-m", "wave_ct.inversion",
                    "--input-csv", str(args.input_csv),
                    "--output-dir", str(run_dir),
                    *clean_args,
                    "--n-outer", str(args.pilot_outer),
                    "--n-lsqr", str(args.pilot_lsqr),
                    "--random-seed", str(seed),
                    "--no-spline-projection",
                    "--no-export-deliverables",
                    *candidate_args,
                ]
                print(f"pilot {candidate} seed={seed}")
                duration = run_command(command, run_dir / "run.log", args.timeout_seconds)
                rows.append({
                    "candidate": candidate,
                    "seed": seed,
                    "duration_s": duration,
                    **collect_run(run_dir),
                })

    summaries = summarize_candidates(rows)
    eligible = [row for row in summaries if bool(
        row["eligible_amplitude"] and row["supportability_eligible"]
    )]
    if eligible:
        selected = eligible[0]
        selection_status = "qualified"
        selection_reason = "minimum validation-plus-dispersion score among stability-eligible candidates"
    else:
        supportable = [row for row in summaries if bool(row["supportability_eligible"])]
        if supportable and any(
            not bool(row["supportability_eligible"]) for row in summaries
        ):
            selected = supportable[0]
            selection_status = "qualified_supportability_tradeoff"
            selection_reason = (
                "no candidate passed both amplitude and supportability gates; "
                "undercovered-data supportability takes priority over the full-grid fallback"
            )
        else:
            selected = next(
                (row for row in summaries if str(row["candidate"]) == "tv"),
                summaries[0],
            )
            selection_status = "low_confidence_safe_fallback"
            selection_reason = "no candidate passed stability gates; conservative plain-TV fallback"
    selected_name = str(selected["candidate"])
    write_csv(pilot_csv, rows)
    write_csv(args.output_dir / "auto_strategy_candidates.csv", summaries)
    if args.curved_ray_gate:
        gate_model = work_root / f"{selected_name}_s{seeds[0]}" / "velocity_model.npz"
        if gate_model.is_file():
            physics_gate = run_eikonal_forward_gate(
                gate_model,
                args.input_csv,
                args.output_dir,
                min(args.timeout_seconds, 1800),
            )
        else:
            physics_gate = {
                "status": "selected_pilot_model_missing_safe_fallback",
                "passed_forward_gate": False,
                "production_action": "retain selected straight-ray inversion",
            }
    else:
        physics_gate = {
            "status": "disabled",
            "passed_forward_gate": False,
            "production_action": "retain selected straight-ray inversion",
        }
    report = {
        "schema_version": 1,
        "selector": "grouped_event_pilot_cv_v1",
        "features": features,
        "pilot": {
            "seeds": seeds,
            "n_outer": args.pilot_outer,
            "n_lsqr": args.pilot_lsqr,
            "candidates": candidates,
        },
        "candidate_summary": summaries,
        "selected_candidate": selected_name,
        "selection_status": selection_status,
        "selection_reason": selection_reason,
        "physics_gate": physics_gate,
        "warning": "selection CV is for production choice; a separate outer event split is required for an unbiased research claim",
        "final_training": "selected hyperparameters are refit with validation_fraction=0 using all source events",
    }
    (args.output_dir / "auto_strategy_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"selected": selected_name, "summary": selected}, ensure_ascii=False, indent=2))

    if not args.no_final_run:
        final_outer = option_value(inversion_args, "--n-outer", "18")
        final_lsqr = option_value(inversion_args, "--n-lsqr", "160")
        final_command = [
            sys.executable, "-m", "wave_ct.inversion",
            "--input-csv", str(args.input_csv),
            "--output-dir", str(args.output_dir),
            *clean_args,
            "--n-outer", final_outer,
            "--n-lsqr", final_lsqr,
            "--random-seed", str(seeds[0]),
            "--no-spline-projection",
            "--export-deliverables",
            *candidates[selected_name],
            "--validation-fraction", "0",
        ]
        run_command(final_command, args.output_dir / "auto_strategy_final.log", args.timeout_seconds)


if __name__ == "__main__":
    main()
