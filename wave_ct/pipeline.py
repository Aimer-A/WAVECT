"""One-command WaveCT project workflow.

The fixed stage order is:

    standard input -> inversion -> result rendering -> validation -> report

Each run writes ``pipeline_run.json`` with commands, input hashes, statuses and
artifacts.  Quantitative ``velocity_model.npz`` is never replaced by the
coverage-stabilised presentation field produced during rendering.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from wave_ct.config import load_project_config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _value_args(model: dict[str, Any], mapping: tuple[tuple[str, str], ...]) -> list[str]:
    result: list[str] = []
    for key, option in mapping:
        value = model.get(key)
        if value is None or value == "":
            continue
        if option == "--slice-z":
            # A comma-separated string beginning with a negative elevation is
            # otherwise parsed by argparse as another option.
            result.append(f"{option}={value}")
        else:
            result.extend([option, str(value)])
    return result


def _config_bool(value: Any, default: bool = False) -> bool:
    """Interpret project booleans without treating the string ``"0"`` as true."""

    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on", "enabled"}:
        return True
    if normalized in {"0", "false", "no", "off", "disabled"}:
        return False
    raise ValueError(f"invalid boolean project value: {value!r}")


def build_inversion_command(
    project: dict[str, Any],
    *,
    algorithm: str = "project",
) -> list[str]:
    inputs = project["inputs"]
    model = dict(project.get("model") or {})
    output_dir = Path(project["outputs"]["directory"])
    command = [
        sys.executable,
        "-m",
        "wave_ct.inversion",
        "--input-csv",
        str(inputs["travel_time_csv"]),
        "--output-dir",
        str(output_dir),
    ]
    command.extend(
        _value_args(
            model,
            (
                ("mode", "--mode"),
                ("expected_sources", "--expected-sources"),
                ("expected_stations_per_source", "--expected-stations-per-source"),
                ("plot_style", "--plot-style"),
                ("x_min", "--x-min"),
                ("x_max", "--x-max"),
                ("y_min", "--y-min"),
                ("y_max", "--y-max"),
                ("z_min", "--z-min"),
                ("z_max", "--z-max"),
                ("dx", "--dx"),
                ("dy", "--dy"),
                ("dz", "--dz"),
                ("nx_nodes", "--nx-nodes"),
                ("ny_nodes", "--ny-nodes"),
                ("nz_nodes", "--nz-nodes"),
                ("slice_z", "--slice-z"),
                ("vmin_qc", "--vmin-qc"),
                ("vmax_qc", "--vmax-qc"),
                ("vmin_model", "--vmin-model"),
                ("vmax_model", "--vmax-model"),
                ("background_velocity", "--background-velocity"),
                ("min_ray_coverage", "--min-ray-coverage"),
                ("coverage_weight_exponent", "--coverage-weight-exponent"),
                ("n_outer", "--n-outer"),
                ("n_lsqr", "--n-lsqr"),
                ("solver_method", "--solver-method"),
                ("sirt_iterations", "--sirt-iterations"),
                ("sirt_omega", "--sirt-omega"),
                ("sirt_step_damp", "--sirt-step-damp"),
                ("sirt_tolerance", "--sirt-tolerance"),
                ("sirt_tune_maxiter", "--sirt-tune-maxiter"),
                ("sirt_tune_popsize", "--sirt-tune-popsize"),
                ("sirt_tune_iterations", "--sirt-tune-iterations"),
                ("reference_sirt_profile", "--reference-profile"),
                ("min_rays", "--min-rays"),
                ("alpha_reg", "--alpha-reg"),
                ("step_damp", "--step-damp"),
                ("validation_fraction", "--validation-fraction"),
                ("huber_delta", "--huber-delta"),
                ("background_damping", "--background-damping"),
                ("model_damping", "--model-damping"),
                ("curvature_reg_factor", "--curvature-reg-factor"),
                ("curvature_z_factor", "--curvature-z-factor"),
                ("source_static_damping", "--source-static-damping"),
                ("global_time_damping", "--global-time-damping"),
                ("max_time_correction", "--max-time-correction"),
                ("deep_reparam_width", "--deep-reparam-width"),
                ("deep_reparam_depth", "--deep-reparam-depth"),
                ("deep_reparam_learning_rate", "--deep-reparam-learning-rate"),
                ("deep_reparam_full_epochs", "--deep-reparam-full-epochs"),
                ("deep_reparam_max_epochs", "--deep-reparam-max-epochs"),
                ("deep_reparam_starts", "--deep-reparam-starts"),
                ("deep_reparam_huber_ms", "--deep-reparam-huber-ms"),
                ("deep_reparam_output_l2", "--deep-reparam-output-l2"),
                ("deep_reparam_fourier_bands", "--deep-reparam-fourier-bands"),
                (
                    "deep_reparam_differential_loss_fraction",
                    "--deep-reparam-differential-loss-fraction",
                ),
                ("deep_reparam_tv", "--deep-reparam-tv"),
                ("deep_reparam_tv_epsilon", "--deep-reparam-tv-epsilon"),
                (
                    "deep_reparam_event_static_shrinkage",
                    "--deep-reparam-event-static-shrinkage",
                ),
                (
                    "deep_reparam_receiver_static_shrinkage",
                    "--deep-reparam-receiver-static-shrinkage",
                ),
                (
                    "deep_reparam_receiver_static_max_ms",
                    "--deep-reparam-receiver-static-max-ms",
                ),
                (
                    "deep_reparam_static_profile_iterations",
                    "--deep-reparam-static-profile-iterations",
                ),
                ("deep_reparam_device", "--deep-reparam-device"),
            ),
        )
    )
    boolean_options = (
        ("event_centered_qc", "--event-centered-qc", "--no-event-centered-qc"),
        ("allow_outside_rays", "--allow-outside-rays", "--no-outside-rays"),
        ("edge_preserving_tv", "--edge-preserving-tv", "--no-edge-preserving-tv"),
        ("joint_sparsity", "--joint-sparsity", "--no-joint-sparsity"),
        (
            "hierarchical_parameterization",
            "--hierarchical-parameterization",
            "--no-hierarchical-parameterization",
        ),
        ("differential_times", "--differential-times", "--no-differential-times"),
        (
            "ray_length_normalization",
            "--ray-length-normalization",
            "--no-ray-length-normalization",
        ),
        (
            "regularize_total_model",
            "--regularize-total-model",
            "--no-regularize-total-model",
        ),
        ("sirt_auto_tune", "--sirt-auto-tune", "--no-sirt-auto-tune"),
    )
    for key, enabled, disabled in boolean_options:
        default_enabled = key in {"regularize_total_model", "sirt_auto_tune"}
        command.append(
            enabled
            if _config_bool(model.get(key), default=default_enabled)
            else disabled
        )
    use_deep = _config_bool(model.get("deep_reparameterization"), default=False)
    if algorithm == "lsqr":
        use_deep = False
        command.extend(["--solver-method", "lsqr"])
    elif algorithm == "sirt":
        # Production SIRT uses the project's physical grid.  The standalone
        # reference script is still available through the explicit
        # ``reference_sirt`` algorithm below, but it must not silently replace
        # the configured grid (doing so can create very large extrapolated
        # regions in a workface plot).
        use_deep = False
        command.extend(["--solver-method", "sirt"])
    elif algorithm == "reference_sirt":
        # Exact compatibility path for reproducing the supplied external
        # script; WaveCT GUI uses this path when the experimental selectors are
        # disabled, while the CLI keeps it explicit for reproducible runs.
        use_deep = False
        command.extend(["--solver-method", "sirt"])
        command.append("--script-compatible-sirt")
    elif algorithm == "dnr":
        use_deep = True
        command.extend(["--solver-method", "sirt"])
    elif algorithm == "project":
        if not any(token == "--solver-method" for token in command):
            command.extend(["--solver-method", "sirt"])
        if (
            str(model.get("solver_method") or "sirt").lower() == "sirt"
            and not use_deep
        ):
            # Keep the project grid for the production path.  Use
            # algorithm="reference_sirt" when exact 10 m script compatibility
            # is specifically required for a reproducibility benchmark.
            pass
    else:
        raise ValueError(f"unsupported pipeline algorithm: {algorithm}")
    command.append(
        "--deep-reparameterization" if use_deep else "--no-deep-reparameterization"
    )
    if use_deep and _config_bool(
        model.get("deep_reparam_receiver_statics"), default=False
    ):
        command.append("--deep-reparam-receiver-statics")
    command.extend(["--invert-source-statics", "--export-deliverables"])
    return command


def build_auto_inversion_command(project: dict[str, Any]) -> list[str]:
    """Build the dataset-profiled, grouped-event automatic inversion command."""
    input_csv = str(project["inputs"]["travel_time_csv"])
    output_dir = str(project["outputs"]["directory"])
    # auto_select owns regularization/robust controls, but retains geometry,
    # physical bounds, static corrections and presentation metadata.
    production = build_inversion_command(project, algorithm="sirt")
    inversion_args = production[7:]
    model = project.get("model") or {}
    return [
        sys.executable, "-m", "wave_ct.auto_select",
        "--input-csv", input_csv,
        "--output-dir", output_dir,
        "--cv-seeds", str(model.get("auto_cv_seeds") or "11,23,41"),
        "--pilot-outer", str(model.get("auto_pilot_outer") or 24),
        "--pilot-lsqr", str(model.get("auto_pilot_lsqr") or 160),
        "--", *inversion_args,
    ]


def build_render_command(project: dict[str, Any]) -> list[str] | None:
    output_dir = Path(project["outputs"]["directory"])
    model_path = output_dir / "velocity_model.npz"
    workface = project.get("workface") or {}
    boundary = Path(str(workface.get("boundary_file") or ""))
    if not boundary.is_file():
        return None
    model = project.get("model") or {}
    dataset = project.get("dataset") or {}
    # Match the reference-SIRT rendering contract: frame presentation by the
    # declared workface boundary.  This changes only plot framing, never the
    # inversion grid, quantitative velocity, or CAD coordinates.
    try:
        with boundary.open("r", encoding="utf-8-sig", newline="") as handle:
            vertices = list(csv.DictReader(handle))
        bx = [float(row["x"]) for row in vertices]
        by = [float(row["y"]) for row in vertices]
        if len(bx) < 3:
            raise ValueError("boundary has fewer than three vertices")
        view_padding = float(model.get("workface_view_padding", 0.0) or 0.0)
        if not 0.0 <= view_padding <= 0.25:
            raise ValueError("workface_view_padding must be in [0, 0.25]")
        pad_x = (max(bx) - min(bx)) * view_padding
        pad_y = (max(by) - min(by)) * view_padding
        x_min, x_max = min(bx) - pad_x, max(bx) + pad_x
        y_min, y_max = min(by) - pad_y, max(by) + pad_y
    except (OSError, KeyError, TypeError, ValueError):
        # Keep old projects usable when an old boundary cannot be parsed.
        x_min, x_max = model["x_min"], model["x_max"]
        y_min, y_max = model["y_min"], model["y_max"]
    final_dir = output_dir / "最终成果图"
    command = [
        sys.executable,
        "-m",
        "wave_ct.workface_plot",
        str(model_path),
        str(project["inputs"]["travel_time_csv"]),
        # The renderer owns packaging into the one final-result child folder.
        # Passing that child here created final/final and hid workface images
        # from the GUI preview.
        str(output_dir),
        "--boundary-file",
        str(boundary),
        "--period",
        str(dataset.get("dataset_id") or project.get("project_name") or ""),
        f"--slice-z={model.get('slice_z') or ''}",
        "--presentation-sigma",
        str(model.get("presentation_sigma", 0.65)),
        "--presentation-vmin",
        str(model.get("presentation_vmin", 0.0)),
        "--presentation-vmax",
        str(model.get("presentation_vmax", 0.0)),
        "--anomaly-limit",
        str(model.get("anomaly_limit", 0.30)),
        "--x-min",
        str(x_min),
        "--x-max",
        str(x_max),
        "--y-min",
        str(y_min),
        "--y-max",
        str(y_max),
    ]
    basemap = Path(str(workface.get("basemap_file") or ""))
    mapa = Path(str(workface.get("mapa_file") or ""))
    if basemap.is_file():
        command.extend(["--cad-file", str(basemap)])
    elif mapa.is_file():
        command.extend(["--mapa-file", str(mapa)])
    command.extend(
        [
            "--cad-x-offset",
            str(workface.get("cad_x_offset", 0.0)),
            "--cad-y-offset",
            str(workface.get("cad_y_offset", 0.0)),
        ]
    )
    return command


def build_validation_command(project: dict[str, Any]) -> list[str]:
    output_dir = Path(project["outputs"]["directory"])
    project_dir = Path(project["inputs"]["travel_time_csv"]).parent
    command = [
        sys.executable,
        "-m",
        "wave_ct.validation_pipeline",
        "--input-csv",
        str(project["inputs"]["travel_time_csv"]),
        "--model-npz",
        str(output_dir / "velocity_model.npz"),
        "--out-dir",
        str(project_dir / "验证结果"),
        "--slice-report",
        str(output_dir / "slice_report.txt"),
        "--anomaly-limit",
        str((project.get("model") or {}).get("anomaly_limit", 0.30)),
    ]
    detail = Path(str(project["inputs"].get("detail_csv") or ""))
    evidence = Path(str(project["inputs"].get("evidence_csv") or ""))
    waveform_text = str(project["inputs"].get("waveform_root") or "").strip()
    if detail.is_file():
        command.extend(["--detail-csv", str(detail)])
    if evidence.is_file():
        command.extend(["--evidence-csv", str(evidence)])
    if waveform_text and Path(waveform_text).is_dir():
        waveform = Path(waveform_text)
        command.extend(["--waveform-root", str(waveform)])
    return command


def build_report_command(project: dict[str, Any]) -> list[str]:
    project_dir = Path(project["inputs"]["travel_time_csv"]).parent
    output_dir = Path(project["outputs"]["directory"])
    adapter = str((project.get("dataset") or {}).get("adapter") or "standard_csv")
    return [
        sys.executable,
        "-m",
        "wave_ct.report",
        "--dataset-name",
        str(project.get("project_name") or project_dir.name),
        "--input-csv",
        str(project["inputs"]["travel_time_csv"]),
        "--inversion-dir",
        str(output_dir),
        "--validation-dir",
        str(project_dir / "验证结果"),
        "--output-docx",
        str(project_dir / "CT反演工程报告.docx"),
        "--picking-source",
        adapter,
        "--note",
        "定量velocity与展示display_velocity严格分离；缺失验证项按SKIPPED报告。",
    ]


def _run_stage(
    name: str,
    command: list[str] | None,
    *,
    cwd: Path,
) -> dict[str, Any]:
    started = datetime.now()
    started_monotonic = time.monotonic()
    stage: dict[str, Any] = {
        "name": name,
        "started_at": started.isoformat(timespec="seconds"),
        "command": command or [],
    }
    if command is None:
        stage.update(
            {
                "status": "SKIPPED",
                "summary": "required optional input was not available",
                "duration_s": 0.0,
            }
        )
        return stage
    print(f"\n=== {name} ===", flush=True)
    print(subprocess.list2cmdline(command), flush=True)
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert process.stdout is not None
    tail: list[str] = []
    for line in process.stdout:
        output_encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        safe_line = line.encode(output_encoding, errors="replace").decode(
            output_encoding, errors="replace"
        )
        print(safe_line, end="", flush=True)
        tail.append(line.rstrip())
        if len(tail) > 60:
            tail.pop(0)
    return_code = process.wait()
    stage.update(
        {
            "status": "PASS" if return_code == 0 else "FAIL",
            "return_code": return_code,
            "duration_s": round(time.monotonic() - started_monotonic, 3),
            "output_tail": tail,
        }
    )
    return stage


def run_project(
    project_path: Path | str,
    *,
    algorithm: str = "project",
) -> dict[str, Any]:
    project_path = Path(project_path).resolve()
    project = load_project_config(project_path)
    input_csv = Path(project["inputs"]["travel_time_csv"])
    if not input_csv.is_file():
        raise FileNotFoundError(f"project travel-time CSV is missing: {input_csv}")
    output_dir = Path(project["outputs"]["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    run_path = project_path.parent / "pipeline_run.json"
    payload: dict[str, Any] = {
        "schema_version": 1,
        "project": str(project_path),
        "project_name": project.get("project_name"),
        "dataset": project.get("dataset"),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "algorithm": algorithm,
        "input_sha256": _sha256(input_csv),
        "stages": [],
        "status": "RUNNING",
    }
    _write_json(run_path, payload)
    inversion_command = (
        build_auto_inversion_command(project)
        if algorithm == "auto"
        else build_inversion_command(project, algorithm=algorithm)
    )
    commands = (
        ("反演", inversion_command),
        ("成果渲染", build_render_command(project)),
        ("工程可信度验证", build_validation_command(project)),
        ("报告", build_report_command(project)),
    )
    try:
        for name, command in commands:
            stage = _run_stage(name, command, cwd=Path(__file__).resolve().parents[1])
            payload["stages"].append(stage)
            _write_json(run_path, payload)
            if stage["status"] == "FAIL":
                raise RuntimeError(
                    f"pipeline stage failed: {name} "
                    f"(code {stage.get('return_code', 'unknown')})"
                )
    except Exception as exc:
        payload["status"] = "FAIL"
        payload["error"] = str(exc)
        payload["finished_at"] = datetime.now().isoformat(timespec="seconds")
        _write_json(run_path, payload)
        raise
    payload["status"] = "PASS"
    payload["finished_at"] = datetime.now().isoformat(timespec="seconds")
    payload["artifacts"] = {
        "model": str(output_dir / "velocity_model.npz"),
        "final_images": str(output_dir / "最终成果图"),
        "validation": str(project_path.parent / "验证结果" / "validation_summary.json"),
        "report": str(project_path.parent / "CT反演工程报告.docx"),
    }
    _write_json(run_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="执行WaveCT完整链：反演、成果图、验证、报告"
    )
    parser.add_argument("--project", type=Path, action="append", required=True)
    parser.add_argument(
        "--algorithm",
        choices=("auto", "project", "sirt", "reference_sirt", "dnr", "lsqr"),
        default="auto",
    )
    args = parser.parse_args()
    for project in args.project:
        result = run_project(project, algorithm=args.algorithm)
        print(
            f"Pipeline complete: {result['project_name']} -> {result['status']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
