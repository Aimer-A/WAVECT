"""Persistent application and project configuration for Wave CT."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping


APP_SCHEMA_VERSION = 1
PROJECT_SCHEMA_VERSION = 2
PROJECT_FILENAME = "wave_ct_project.json"


MODEL_PARAMETER_KEYS = (
    "expected_sources",
    "expected_stations_per_source",
    "waveform_pattern",
    "source_pattern",
    "source_coord_filename",
    "mode",
    "dx",
    "dy",
    "dz",
    "n_outer",
    "n_lsqr",
    "solver_method",
    "sirt_iterations",
    "sirt_omega",
    "sirt_step_damp",
    "sirt_tolerance",
    "sirt_auto_tune",
    "sirt_tune_maxiter",
    "sirt_tune_popsize",
    "sirt_tune_iterations",
    "reference_sirt_profile",
    "alpha_reg",
    "step_damp",
    "vmin_qc",
    "vmax_qc",
    "vmin_model",
    "vmax_model",
    "background_velocity",
    "min_ray_coverage",
    "coverage_weight_exponent",
    "min_rays",
    "validation_fraction",
    "huber_delta",
    "background_damping",
    "model_damping",
    "regularize_total_model",
    "curvature_reg_factor",
    "curvature_z_factor",
    "source_static_damping",
    "global_time_damping",
    "max_time_correction",
    "edge_preserving_tv",
    "joint_sparsity",
    "wavelet_levels",
    "wavelet_threshold_factor",
    "hierarchical_parameterization",
    "hierarchical_split_rays",
    "hierarchical_min_block_x",
    "hierarchical_min_block_y",
    "differential_times",
    "differential_weight",
    "ray_length_normalization",
    "allow_outside_rays",
    "event_centered_qc",
    "auto_algorithm",
    "auto_cv_seeds",
    "auto_pilot_outer",
    "auto_pilot_lsqr",
    "deep_reparameterization",
    "deep_reparam_width",
    "deep_reparam_depth",
    "deep_reparam_full_epochs",
    "deep_reparam_starts",
    "deep_reparam_device",
    "nx_nodes",
    "ny_nodes",
    "nz_nodes",
    "auto_bounds",
    "x_min",
    "x_max",
    "y_min",
    "y_max",
    "z_min",
    "z_max",
    "slice_z",
    "plot_style",
    "presentation_vmin",
    "presentation_vmax",
    "presentation_sigma",
    "anomaly_limit",
    "workface_view_padding",
)


def app_data_root() -> Path:
    configured = os.environ.get("WAVECT_HOME", "").strip()
    if configured:
        return Path(configured).expanduser()
    local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
    if local_app_data:
        return Path(local_app_data) / "WaveCT"
    return Path.home() / ".wavect"


def app_settings_path() -> Path:
    configured = os.environ.get("WAVECT_SETTINGS_FILE", "").strip()
    return Path(configured).expanduser() if configured else app_data_root() / "settings.json"


def discover_accoreconsole() -> str:
    candidates: list[Path] = []
    configured = os.environ.get("AUTOCAD_CORE_CONSOLE", "").strip()
    if configured:
        candidates.append(Path(configured))
    for drive in "CDEF":
        root = Path(f"{drive}:/")
        candidates.extend(root.glob("autocad/AutoCAD */accoreconsole.exe"))
        candidates.extend(root.glob("Program Files/Autodesk/AutoCAD */accoreconsole.exe"))
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())
    return ""


def default_app_settings() -> dict[str, Any]:
    return {
        "schema_version": APP_SCHEMA_VERSION,
        "autocad_core_console": discover_accoreconsole(),
        "cad_cache_dir": str(app_data_root() / "cad_cache"),
        "default_output_root": "",
    }


def _merge_known(defaults: Mapping[str, Any], loaded: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(defaults))
    for key in defaults:
        if key in loaded:
            result[key] = loaded[key]
    return result


def load_app_settings(path: Path | None = None) -> dict[str, Any]:
    settings_path = path or app_settings_path()
    defaults = default_app_settings()
    if not settings_path.is_file():
        return defaults
    try:
        loaded = json.loads(settings_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return defaults
    if not isinstance(loaded, dict):
        return defaults
    settings = _merge_known(defaults, loaded)
    if not str(settings.get("autocad_core_console", "")).strip():
        settings["autocad_core_console"] = discover_accoreconsole()
    return settings


def save_app_settings(settings: Mapping[str, Any], path: Path | None = None) -> Path:
    settings_path = path or app_settings_path()
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _merge_known(default_app_settings(), settings)
    payload["schema_version"] = APP_SCHEMA_VERSION
    settings_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return settings_path


def portable_path(value: str | Path | None, project_dir: Path) -> str:
    if value is None or not str(value).strip():
        return ""
    path = Path(value).expanduser()
    try:
        resolved = path.resolve()
        relative = resolved.relative_to(project_dir.resolve())
    except (OSError, ValueError):
        return str(path)
    return "${PROJECT_DIR}/" + relative.as_posix()


def resolve_project_path(value: Any, project_dir: Path) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    prefix = "${PROJECT_DIR}/"
    if text.startswith(prefix):
        return str((project_dir / text[len(prefix):]).resolve())
    if text == "${PROJECT_DIR}":
        return str(project_dir.resolve())
    return str(Path(text).expanduser())


def default_project_config() -> dict[str, Any]:
    return {
        "schema_version": PROJECT_SCHEMA_VERSION,
        "project_name": "",
        "report_template": "auto",
        "dataset": {
            "dataset_id": "",
            "data_type": "",
            "independence_group": "",
            "adapter": "",
            "raw_root": "",
        },
        "inputs": {
            "travel_time_csv": "",
            "detail_csv": "",
            "pick_audit_csv": "",
            "waveform_root": "",
            "station_file": "",
            "evidence_csv": "",
        },
        "workface": {
            "boundary_file": "",
            "basemap_file": "",
            "mapa_file": "",
            "cad_x_offset": 0.0,
            "cad_y_offset": 0.0,
        },
        "outputs": {"directory": ""},
        "model": {},
    }


def project_path_for_csv(csv_path: Path) -> Path:
    return csv_path.resolve().parent / PROJECT_FILENAME


def load_project_config(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("Project configuration must be a JSON object")
    defaults = default_project_config()
    result = deepcopy(defaults)
    for key in ("schema_version", "project_name", "report_template"):
        if key in loaded:
            result[key] = loaded[key]
    for section in ("dataset", "inputs", "workface", "outputs", "model"):
        section_value = loaded.get(section)
        if isinstance(section_value, dict):
            result[section].update(section_value)
    project_dir = path.resolve().parent
    for section, keys in (
        ("dataset", ("raw_root",)),
        (
            "inputs",
            (
                "travel_time_csv",
                "detail_csv",
                "pick_audit_csv",
                "waveform_root",
                "station_file",
                "evidence_csv",
            ),
        ),
        ("workface", ("boundary_file", "basemap_file", "mapa_file")),
        ("outputs", ("directory",)),
    ):
        for key in keys:
            result[section][key] = resolve_project_path(result[section].get(key), project_dir)
    return result


def save_project_config(path: Path, config: Mapping[str, Any]) -> Path:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    defaults = default_project_config()
    payload = deepcopy(defaults)
    for key in ("project_name", "report_template"):
        payload[key] = config.get(key, defaults[key])
    for section in ("dataset", "inputs", "workface", "outputs", "model"):
        value = config.get(section, {})
        if isinstance(value, Mapping):
            payload[section].update(dict(value))
    payload["schema_version"] = PROJECT_SCHEMA_VERSION
    project_dir = path.parent
    for section, keys in (
        ("dataset", ("raw_root",)),
        (
            "inputs",
            (
                "travel_time_csv",
                "detail_csv",
                "pick_audit_csv",
                "waveform_root",
                "station_file",
                "evidence_csv",
            ),
        ),
        ("workface", ("boundary_file", "basemap_file", "mapa_file")),
        ("outputs", ("directory",)),
    ):
        for key in keys:
            payload[section][key] = portable_path(payload[section].get(key), project_dir)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
