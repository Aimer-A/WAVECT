"""Adapter-driven import of raw/legacy acquisition data into WaveCT projects.

The importer converts supported inputs to one auditable CSV contract and
records enough lineage for inversion, rendering, validation and reporting.
Historical result grids may provide geometry/slice labels for comparison, but
their velocity values never enter a new inverse solve.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import date, datetime, time
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np

from wave_ct.cmat_io import (
    CMATDataset,
    WAVECT_INVERSION_COLUMNS,
    read_cmat_3dd,
)
from wave_ct.cmat_model_io import CMATModelGeometry, read_cmat_model_geometry
from wave_ct.config import save_project_config
from wave_ct.sos_lok import SOSStation, merge_station_tables
from wave_ct.sos_w_io import (
    SOSWHeader,
    SOSWTraceRecord,
    iter_marked_records,
    read_sos_w_header,
    station_coordinate_index,
)


DETAIL_COLUMNS = (
    "event_id",
    "event_time",
    "file_name",
    "waveform_file",
    "adapter",
    "station_index",
    "station_id",
    "block_index",
    "channel_id",
    "flag",
    "sample_rate_hz",
    "sample_count",
    "p_index",
    "marker_2",
    "marker_3",
    "marker_4",
    "raw_arrival_ms",
    "estimated_origin_ms",
    "travel_time_ms",
    "distance_m",
    "reference_residual_ms",
    "apparent_velocity_mps",
    "source_x",
    "source_y",
    "source_z",
    "station_x",
    "station_y",
    "station_z",
    "status",
    "reason",
)


@dataclass(frozen=True)
class CatalogEvent:
    event_time: datetime
    x: float
    y: float
    z: float
    energy: float | None

    @property
    def xyz(self) -> tuple[float, float, float]:
        return self.x, self.y, self.z


@dataclass(frozen=True)
class ImportCandidate:
    adapter: str
    primary_input: Path
    period_root: Path
    waveform_dir: Path | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _finite_xyz(values: Iterable[float], *, label: str) -> tuple[float, float, float]:
    xyz = tuple(float(value) for value in values)
    if len(xyz) != 3 or not np.isfinite(np.asarray(xyz)).all():
        raise ValueError(f"invalid {label} coordinates: {xyz}")
    return xyz  # type: ignore[return-value]


def _combine_excel_date_time(day_value: object, time_value: object) -> datetime:
    if isinstance(day_value, datetime):
        day = day_value.date()
    elif isinstance(day_value, date):
        day = day_value
    elif isinstance(day_value, str):
        day = datetime.fromisoformat(day_value.strip()).date()
    else:
        raise ValueError(f"unsupported event date value: {day_value!r}")
    if isinstance(time_value, datetime):
        clock = time_value.time()
    elif isinstance(time_value, time):
        clock = time_value
    elif isinstance(time_value, str):
        clock = time.fromisoformat(time_value.strip())
    elif isinstance(time_value, (float, int)):
        seconds = int(round(float(time_value) * 24.0 * 3600.0))
        clock = time(
            hour=(seconds // 3600) % 24,
            minute=(seconds // 60) % 60,
            second=seconds % 60,
        )
    else:
        raise ValueError(f"unsupported event time value: {time_value!r}")
    return datetime.combine(day, clock).replace(microsecond=0)


def load_event_catalog(path: Path | str) -> dict[datetime, CatalogEvent]:
    """Read the SOS event table, including OOXML workbooks named ``.xls``."""

    try:
        import openpyxl
    except ImportError as exc:  # pragma: no cover - dependency is bundled
        raise RuntimeError("openpyxl is required to read SOS event catalogs") from exc
    path = Path(path)
    workbook = openpyxl.load_workbook(
        io.BytesIO(path.read_bytes()),
        read_only=True,
        data_only=True,
    )
    sheet = workbook.active
    rows = sheet.iter_rows(values_only=True)
    try:
        header = next(rows)
    except StopIteration as exc:
        raise ValueError(f"event catalog is empty: {path}") from exc
    names = [str(value or "").strip().casefold() for value in header]

    def index(*options: str) -> int:
        for option in options:
            if option.casefold() in names:
                return names.index(option.casefold())
        raise ValueError(f"event catalog lacks column {options}: {path}")

    date_i = index("Data", "date", "日期")
    time_i = index("CZAS", "time", "时间")
    x_i, y_i, z_i = index("X"), index("Y"), index("Z")
    energy_i = names.index("energia") if "energia" in names else None
    events: dict[datetime, CatalogEvent] = {}
    for row_number, row in enumerate(rows, start=2):
        if not row or row[date_i] in (None, "") or row[time_i] in (None, ""):
            continue
        event_time = _combine_excel_date_time(row[date_i], row[time_i])
        x, y, z = _finite_xyz(
            (row[x_i], row[y_i], row[z_i]),
            label=f"event row {row_number}",
        )
        energy = (
            float(row[energy_i])
            if energy_i is not None and row[energy_i] not in (None, "")
            else None
        )
        if energy is not None and not math.isfinite(energy):
            raise ValueError(f"non-finite event energy at row {row_number}")
        event = CatalogEvent(event_time, x, y, z, energy)
        previous = events.get(event_time)
        if previous is not None and previous.xyz != event.xyz:
            raise ValueError(
                f"ambiguous duplicate event time {event_time.isoformat()}: {path}"
            )
        events[event_time] = event
    if not events:
        raise ValueError(f"event catalog has no usable rows: {path}")
    return events


def read_reference_slice_levels(period_root: Path) -> tuple[float, ...]:
    """Read requested slice labels from reference filenames, not field values."""

    levels: set[float] = set()
    pattern = re.compile(
        r"(?:波速分布|velocity).*?z\s*([-+]?\d+(?:\.\d+)?)",
        flags=re.IGNORECASE,
    )
    for path in period_root.glob("*.txt"):
        match = pattern.search(path.stem)
        if match:
            levels.add(float(match.group(1)))
    return tuple(sorted(levels, reverse=True))


def find_model_geometry(period_root: Path) -> tuple[CMATModelGeometry | None, Path | None]:
    models = sorted(period_root.glob("*.3DM"))
    if not models:
        models = sorted(period_root.glob("*.3dm"))
    if not models:
        return None, None
    return read_cmat_model_geometry(models[0]), models[0]


def _nearest_period_file(
    start: Path,
    *,
    names: tuple[str, ...] = (),
    patterns: tuple[str, ...] = (),
) -> Path | None:
    current = start
    for _ in range(5):
        for name in names:
            candidate = current / name
            if candidate.is_file():
                return candidate
        for pattern in patterns:
            matches = sorted(current.glob(pattern))
            if matches:
                return matches[0]
        if current.parent == current:
            break
        current = current.parent
    return None


def discover_import_candidates(dataset_root: Path | str) -> tuple[ImportCandidate, ...]:
    """Discover direct CMAT rays and selected loose SOS W event folders."""

    root = Path(dataset_root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"dataset root not found: {root}")
    candidates: list[ImportCandidate] = []
    for path in sorted(root.rglob("*.3dd")):
        candidates.append(
            ImportCandidate("cmat_3dd", path.resolve(), path.parent.resolve())
        )
    waveform_groups: dict[Path, list[Path]] = defaultdict(list)
    for path in root.rglob("*.W"):
        waveform_groups[path.parent.resolve()].append(path.resolve())
    for waveform_dir, paths in sorted(
        waveform_groups.items(), key=lambda item: str(item[0]).casefold()
    ):
        # A selected loose directory is preferred over an archive.  Requiring
        # multiple events also avoids treating incidental single files as a cohort.
        if len(paths) < 2:
            continue
        period_root = waveform_dir.parent
        param = _nearest_period_file(
            period_root,
            names=("SOS/PARAM.LOK", "SOS/Param.lok", "PARAM.LOK", "Param.lok"),
        )
        catalog = _nearest_period_file(
            period_root,
            patterns=("*.xls", "*.xlsx"),
        )
        if param is None or catalog is None:
            continue
        candidates.append(
            ImportCandidate(
                "sos_w",
                paths[0],
                period_root.resolve(),
                waveform_dir=waveform_dir,
            )
        )
    unique: dict[tuple[str, str], ImportCandidate] = {}
    for candidate in candidates:
        key = (candidate.adapter, str(candidate.period_root).casefold())
        unique[key] = candidate
    return tuple(unique.values())


def _project_slug(adapter: str, period_root: Path, start: date, end: date) -> str:
    prefix = "xuzhuang_7432"
    return f"{prefix}_{start.isoformat()}_{end.isoformat()}_{adapter}"


def _compact_grid(geometry: CMATModelGeometry) -> tuple[int, int, int]:
    """Choose a conservative grid that does not claim unsupported resolution."""

    nx = min(geometry.nx_nodes, max(12, int(round((geometry.x_max - geometry.x_min) / 60.0)) + 1))
    ny = min(geometry.ny_nodes, max(12, int(round((geometry.y_max - geometry.y_min) / 60.0)) + 1))
    nz = min(geometry.nz_nodes, max(10, int(round((geometry.z_max - geometry.z_min) / 50.0)) + 1))
    return nx, ny, nz


def _model_config(
    geometry: CMATModelGeometry,
    slice_z: tuple[float, ...],
    *,
    adapter: str,
    background_velocity_mps: float,
) -> dict[str, object]:
    nx, ny, nz = _compact_grid(geometry)
    return {
        "mode": "generic",
        "auto_bounds": False,
        "x_min": geometry.x_min,
        "x_max": geometry.x_max,
        "y_min": geometry.y_min,
        "y_max": geometry.y_max,
        "z_min": geometry.z_min,
        "z_max": geometry.z_max,
        "dx": (geometry.x_max - geometry.x_min) / max(nx - 1, 1),
        "dy": (geometry.y_max - geometry.y_min) / max(ny - 1, 1),
        "dz": (geometry.z_max - geometry.z_min) / max(nz - 1, 1),
        "nx_nodes": nx,
        "ny_nodes": ny,
        "nz_nodes": nz,
        "slice_z": ",".join(f"{value:.3f}" for value in slice_z),
        "expected_sources": 0,
        "expected_stations_per_source": 0,
        "n_outer": 18,
        "n_lsqr": 180,
        "solver_method": "sirt",
        "sirt_iterations": 180,
        "sirt_omega": 0.30,
        "sirt_step_damp": 1.0,
        "sirt_tolerance": 1.0e-8,
        "sirt_auto_tune": False,
        "sirt_tune_maxiter": 5,
        "sirt_tune_popsize": 4,
        "sirt_tune_iterations": 0,
        "min_rays": 20,
        "alpha_reg": 4.0,
        "step_damp": 0.15,
        "vmin_qc": 2500.0 if adapter == "sos_w" else 0.0,
        "vmax_qc": 8000.0 if adapter == "sos_w" else 0.0,
        "vmin_model": 2000.0,
        "vmax_model": 6500.0,
        "background_velocity": background_velocity_mps,
        "min_ray_coverage": 0.0,
        "coverage_weight_exponent": 1.5,
        "validation_fraction": 0.2,
        "huber_delta": 1.5,
        "background_damping": 1.5,
        "model_damping": 0.0,
        "regularize_total_model": True,
        "curvature_reg_factor": 0.25,
        "curvature_z_factor": 0.5,
        "source_static_damping": 5.0,
        "global_time_damping": 1.0,
        "max_time_correction": 0.05,
        "event_centered_qc": True,
        "allow_outside_rays": True,
        "edge_preserving_tv": False,
        "joint_sparsity": False,
        "hierarchical_parameterization": False,
        "differential_times": False,
        "ray_length_normalization": False,
        "anomaly_limit": 0.30,
        "plot_style": "rectangular",
        # DNR/coordinate-MLP is an experimental parameterisation.  It must
        # not become the default merely because the importer can expose it:
        # production selection is owned by grouped-event validation.
        "deep_reparameterization": False,
        "deep_reparam_width": 24,
        "deep_reparam_depth": 3,
        "deep_reparam_full_epochs": 350,
        "deep_reparam_starts": 3,
        "deep_reparam_device": "cpu",
        "deep_reparam_fourier_bands": 0,
        "deep_reparam_differential_loss_fraction": 0.0,
        "deep_reparam_tv": 0.0,
        "deep_reparam_receiver_statics": False,
    }


def _write_boundary(path: Path, geometry: CMATModelGeometry) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["name", "x", "y"])
        writer.writerows(
            [
                ("ROI1", geometry.x_min, geometry.y_min),
                ("ROI2", geometry.x_max, geometry.y_min),
                ("ROI3", geometry.x_max, geometry.y_max),
                ("ROI4", geometry.x_min, geometry.y_max),
            ]
        )


def _write_station_file(path: Path, stations: dict[str, tuple[float, float, float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        for station_id, (x, y, z) in sorted(stations.items()):
            handle.write(f"{station_id} NET {y:.6f} {x:.6f} 0 {z:.6f}\n")


def _write_inversion_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=WAVECT_INVERSION_COLUMNS)
        writer.writeheader()
        for row in rows:
            if row["status"] != "KEEP":
                continue
            writer.writerow(
                {
                    "震源编号": row["event_id"],
                    "震源坐标-x": f"{float(row['source_x']):.6f}",
                    "震源坐标-y": f"{float(row['source_y']):.6f}",
                    "震源坐标-z": f"{float(row['source_z']):.6f}",
                    "发震时刻t": f"{float(row['estimated_origin_ms']):.6f}",
                    "台站坐标-x": f"{float(row['station_x']):.6f}",
                    "台站坐标-y": f"{float(row['station_y']):.6f}",
                    "台站坐标-z": f"{float(row['station_z']):.6f}",
                    "台站P波到时": f"{float(row['raw_arrival_ms']):.6f}",
                    "震源-台站传播时间": f"{float(row['travel_time_ms']):.6f}",
                    "震源事件文件名": row["file_name"],
                }
            )


def _write_detail_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DETAIL_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _estimate_centering_velocity(
    rows: list[dict[str, object]],
    initial_velocity_mps: float,
) -> tuple[float, float]:
    """Fit one reference velocity using only within-event arrival differences."""

    valid = [row for row in rows if row["status"] == "CANDIDATE"]
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in valid:
        grouped[int(row["event_id"])].append(row)
    if len(grouped) < 2:
        raise ValueError("too few SOS W events to estimate event-centered travel times")
    low = max(2000.0, initial_velocity_mps * 0.65)
    high = min(8000.0, initial_velocity_mps * 1.35)
    candidates = np.linspace(low, high, 321)
    best_velocity = float(initial_velocity_mps)
    best_score = float("inf")
    for velocity in candidates:
        residuals: list[float] = []
        for event_rows in grouped.values():
            offsets = np.asarray(
                [
                    float(row["raw_arrival_ms"])
                    - float(row["distance_m"]) / velocity * 1000.0
                    for row in event_rows
                ],
                dtype=float,
            )
            center = float(np.median(offsets))
            residuals.extend(float(value - center) for value in offsets)
        score = float(np.median(np.abs(np.asarray(residuals, dtype=float))))
        if score < best_score:
            best_score = score
            best_velocity = float(velocity)
    return best_velocity, best_score


def _find_sos_inputs(period_root: Path) -> tuple[Path, Path | None, Path]:
    param = _nearest_period_file(
        period_root,
        names=("SOS/PARAM.LOK", "SOS/Param.lok", "PARAM.LOK", "Param.lok"),
    )
    dump = _nearest_period_file(
        period_root,
        names=(
            "SOS/PARAM_SOSDUMP.LOK",
            "SOS/Param_SOSDUMP.lok",
            "PARAM_SOSDUMP.LOK",
        ),
    )
    catalog = _nearest_period_file(period_root, patterns=("*.xls", "*.xlsx"))
    if param is None or catalog is None:
        raise FileNotFoundError(
            f"SOS W import requires PARAM.LOK and an event catalog: {period_root}"
        )
    return param, dump, catalog


def import_sos_w_candidate(
    candidate: ImportCandidate,
    output_dir: Path,
) -> Path:
    if candidate.waveform_dir is None:
        raise ValueError("SOS W candidate lacks waveform directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    param_path, dump_path, catalog_path = _find_sos_inputs(candidate.period_root)
    reference_velocity, stations, station_warnings = merge_station_tables(
        param_path, dump_path
    )
    station_by_index = {station.coordinate_index: station for station in stations}
    events = load_event_catalog(catalog_path)
    w_files = sorted(candidate.waveform_dir.glob("*.W"), key=lambda path: path.name)
    if not w_files:
        raise RuntimeError(f"no loose selected .W files: {candidate.waveform_dir}")
    headers = [read_sos_w_header(path) for path in w_files]
    matched_events: list[CatalogEvent] = []
    rows: list[dict[str, object]] = []
    for event_id, header in enumerate(headers, start=1):
        event = events.get(header.event_time)
        if event is None:
            raise ValueError(
                f"W event time is absent from catalog: {header.path.name}"
            )
        matched_events.append(event)
        source = np.asarray(event.xyz, dtype=float)
        marked_blocks = {record.block_index for record in iter_marked_records(header)}
        for record in header.records:
            row: dict[str, object] = {
                "event_id": event_id,
                "event_time": header.event_time.isoformat(sep=" "),
                "file_name": header.path.name,
                "waveform_file": str(header.path),
                "adapter": "sos_w",
                "station_index": "",
                "station_id": "",
                "block_index": record.block_index,
                "channel_id": record.channel_id,
                "flag": record.flag,
                "sample_rate_hz": header.sample_rate_hz,
                "sample_count": header.sample_count,
                "p_index": record.p_index,
                "marker_2": record.marker_2,
                "marker_3": record.marker_3,
                "marker_4": record.marker_4,
                "raw_arrival_ms": "",
                "estimated_origin_ms": "",
                "travel_time_ms": "",
                "distance_m": "",
                "reference_residual_ms": "",
                "apparent_velocity_mps": "",
                "source_x": event.x,
                "source_y": event.y,
                "source_z": event.z,
                "station_x": "",
                "station_y": "",
                "station_z": "",
                "status": "NO_PICK",
                "reason": "marker_1_is_zero",
            }
            if record.block_index not in marked_blocks:
                if record.p_index >= header.sample_count:
                    row["status"] = "REJECT"
                    row["reason"] = "marker_1_out_of_range"
                rows.append(row)
                continue
            station_index = station_coordinate_index(header, record)
            station = station_by_index.get(station_index)
            row["station_index"] = station_index
            if station is None:
                row["status"] = "REJECT"
                row["reason"] = "station_index_out_of_range"
                rows.append(row)
                continue
            row["station_id"] = station.station_id
            row["station_x"], row["station_y"], row["station_z"] = station.xyz
            if not station.enabled:
                row["status"] = "REJECT"
                row["reason"] = "station_disabled_zero_coordinate"
                rows.append(row)
                continue
            distance = float(np.linalg.norm(np.asarray(station.xyz) - source))
            raw_arrival_ms = record.p_index / header.sample_rate_hz * 1000.0
            row["distance_m"] = distance
            row["raw_arrival_ms"] = raw_arrival_ms
            row["status"] = "CANDIDATE"
            row["reason"] = ""
            rows.append(row)

    centering_velocity, centering_mad_ms = _estimate_centering_velocity(
        rows, reference_velocity
    )
    event_rows: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row["status"] == "CANDIDATE":
            event_rows[int(row["event_id"])].append(row)
    for event_id, members in event_rows.items():
        if len(members) < 2:
            for row in members:
                row["status"] = "REJECT"
                row["reason"] = "fewer_than_two_picks_for_event_centering"
            continue
        offsets = np.asarray(
            [
                float(row["raw_arrival_ms"])
                - float(row["distance_m"]) / centering_velocity * 1000.0
                for row in members
            ],
            dtype=float,
        )
        origin_ms = float(np.median(offsets))
        for row in members:
            travel_ms = float(row["raw_arrival_ms"]) - origin_ms
            residual_ms = (
                travel_ms
                - float(row["distance_m"]) / centering_velocity * 1000.0
            )
            row["estimated_origin_ms"] = origin_ms
            row["travel_time_ms"] = travel_ms
            row["reference_residual_ms"] = residual_ms
            row["apparent_velocity_mps"] = (
                float(row["distance_m"]) / (travel_ms / 1000.0)
                if travel_ms > 0
                else float("nan")
            )
            if (
                travel_ms <= 0
                or not np.isfinite(float(row["apparent_velocity_mps"]))
            ):
                row["status"] = "REJECT"
                row["reason"] = "nonpositive_centered_travel_time"
            else:
                row["status"] = "KEEP"

    inversion_csv = output_dir / "inversion_input.csv"
    detail_csv = output_dir / "inversion_input_detail.csv"
    audit_csv = output_dir / "pick_audit.csv"
    _write_inversion_rows(inversion_csv, rows)
    _write_detail_rows(detail_csv, [row for row in rows if row["status"] == "KEEP"])
    _write_detail_rows(audit_csv, rows)
    used_stations = {
        str(row["station_id"]): (
            float(row["station_x"]),
            float(row["station_y"]),
            float(row["station_z"]),
        )
        for row in rows
        if row["status"] == "KEEP"
    }
    station_file = output_dir / "stations_for_picker.txt"
    _write_station_file(station_file, used_stations)

    geometry, model_path = find_model_geometry(candidate.period_root)
    if geometry is None:
        used = [row for row in rows if row["status"] == "KEEP"]
        coordinates = np.asarray(
            [
                [
                    float(row["source_x"]),
                    float(row["source_y"]),
                    float(row["source_z"]),
                ]
                for row in used
            ]
            + [
                [
                    float(row["station_x"]),
                    float(row["station_y"]),
                    float(row["station_z"]),
                ]
                for row in used
            ],
            dtype=float,
        )
        low = coordinates.min(axis=0)
        high = coordinates.max(axis=0)
        pad = np.maximum((high - low) * 0.05, 10.0)
        geometry = CMATModelGeometry(
            x_min=float(low[0] - pad[0]),
            x_max=float(high[0] + pad[0]),
            y_min=float(low[1] - pad[1]),
            y_max=float(high[1] + pad[1]),
            z_min=float(low[2] - pad[2]),
            z_max=float(high[2] + pad[2]),
            nx_nodes=30,
            ny_nodes=24,
            nz_nodes=18,
            initial_velocity_mps=centering_velocity,
        )
    slices = read_reference_slice_levels(candidate.period_root)
    if not slices:
        slices = tuple(
            float(value)
            for value in np.linspace(geometry.z_max, geometry.z_min, 3)[1:-1]
        )
    boundary_file = output_dir / "workface_boundary.csv"
    _write_boundary(boundary_file, geometry)
    cad_files = sorted(candidate.period_root.glob("*.dxf"))
    if not cad_files:
        cad_files = sorted(candidate.period_root.glob("*.dwg"))
    start_date = min(event.event_time.date() for event in matched_events)
    end_date = max(event.event_time.date() for event in matched_events)
    dataset_id = _project_slug("sos_w", candidate.period_root, start_date, end_date)
    project = {
        "project_name": f"测试数据集728 {start_date}至{end_date} SOS W走时CT",
        "report_template": "generic",
        "dataset": {
            "dataset_id": dataset_id,
            "data_type": "real",
            "independence_group": "xuzhuang_7432",
            "adapter": "sos_w",
            "raw_root": str(candidate.period_root),
        },
        "inputs": {
            "travel_time_csv": str(inversion_csv),
            "detail_csv": str(detail_csv),
            "pick_audit_csv": str(audit_csv),
            "waveform_root": str(candidate.waveform_dir),
            "station_file": str(station_file),
            "evidence_csv": "",
        },
        "workface": {
            "boundary_file": str(boundary_file),
            "basemap_file": str(cad_files[0]) if cad_files else "",
            "mapa_file": "",
            "cad_x_offset": 0.0,
            "cad_y_offset": 0.0,
        },
        "outputs": {"directory": str(output_dir / "反演结果")},
        "model": _model_config(
            geometry,
            slices,
            adapter="sos_w",
            background_velocity_mps=centering_velocity,
        ),
    }
    project_path = save_project_config(output_dir / "wave_ct_project.json", project)
    status_counts = Counter(str(row["status"]) for row in rows)
    kept = [row for row in rows if row["status"] == "KEEP"]
    summary = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "adapter": "sos_w",
        "input_kind": "event_centered_manual_P_markers",
        "waveform_amplitude_status": "UNSUPPORTED_PROPRIETARY_CODEC",
        "waveform_validation_status": "SKIPPED",
        "waveform_files": len(headers),
        "catalog_events": len(events),
        "matched_events": len(matched_events),
        "trace_count_layouts": dict(Counter(header.trace_count for header in headers)),
        "manual_p_candidates": sum(
            1 for row in rows if row["status"] not in {"NO_PICK"}
        ),
        "status_counts": dict(status_counts),
        "kept_rays": len(kept),
        "used_station_count": len(used_stations),
        "event_picks_min_median_max": [
            int(min(len(value) for value in event_rows.values())),
            float(np.median([len(value) for value in event_rows.values()])),
            int(max(len(value) for value in event_rows.values())),
        ],
        "param_reference_velocity_mps": reference_velocity,
        "event_centering_velocity_mps": centering_velocity,
        "event_centering_median_absolute_residual_ms": centering_mad_ms,
        "station_warnings": list(station_warnings),
        "reference_geometry_only": geometry.as_dict(),
        "reference_model_file": str(model_path) if model_path else "",
        "reference_velocity_values_used_as_prior": False,
        "requested_slice_z": list(slices),
        "input_sha256": {
            "catalog": _sha256(catalog_path),
            "param_lok": _sha256(param_path),
            "selected_waveforms_manifest": hashlib.sha256(
                "\n".join(
                    f"{path.name}|{path.stat().st_size}|{_sha256(path)}"
                    for path in w_files
                ).encode("utf-8")
            ).hexdigest(),
        },
        "provenance": {
            "source_coordinates": "SOS event catalog joined by event time to second",
            "station_coordinates": "same-period PARAM.LOK; zero coordinates disabled",
            "station_ids": "PARAM_SOSDUMP.LOK only for labels and drift audit",
            "arrival": "SOS W marker_1 / sample_rate",
            "origin": (
                "per-event robust center of arrival - distance / fitted reference velocity"
            ),
            "travel_time": "arrival - estimated event origin",
            "solver_requirement": "source statics and robust residual weighting enabled",
            "historical_result_fields": "evaluation geometry only; never solver input",
        },
        "outputs": {
            "inversion_csv": str(inversion_csv.resolve()),
            "detail_csv": str(detail_csv.resolve()),
            "pick_audit_csv": str(audit_csv.resolve()),
            "project": str(project_path),
        },
    }
    _write_json(output_dir / "dataset_summary.json", summary)
    return project_path


def _cmat_detail_rows(dataset: CMATDataset) -> tuple[list[dict[str, object]], dict[str, tuple[float, float, float]]]:
    source_ids: dict[tuple[float, float, float], int] = {}
    station_ids: dict[tuple[float, float, float], str] = {}
    rows: list[dict[str, object]] = []
    for ray in dataset.rays:
        event_id = source_ids.setdefault(ray.source_xyz, len(source_ids) + 1)
        station_id = station_ids.setdefault(
            ray.receiver_xyz, f"STA{len(station_ids) + 1:03d}"
        )
        rows.append(
            {
                "event_id": event_id,
                "event_time": "",
                "file_name": f"{dataset.path.stem}_source_{event_id:06d}",
                "waveform_file": "",
                "adapter": "cmat_3dd",
                "station_index": "",
                "station_id": station_id,
                "block_index": "",
                "channel_id": "",
                "flag": "",
                "sample_rate_hz": "",
                "sample_count": "",
                "p_index": "",
                "marker_2": "",
                "marker_3": "",
                "marker_4": "",
                "raw_arrival_ms": ray.travel_time_ms,
                "estimated_origin_ms": 0.0,
                "travel_time_ms": ray.travel_time_ms,
                "distance_m": ray.distance_m,
                "reference_residual_ms": "",
                "apparent_velocity_mps": ray.apparent_velocity_mps,
                "source_x": ray.source_x,
                "source_y": ray.source_y,
                "source_z": ray.source_z,
                "station_x": ray.receiver_x,
                "station_y": ray.receiver_y,
                "station_z": ray.receiver_z,
                "status": "KEEP",
                "reason": "",
            }
        )
    return rows, {value: key for key, value in station_ids.items()}


def import_cmat_candidate(candidate: ImportCandidate, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = read_cmat_3dd(candidate.primary_input)
    inversion_csv = dataset.write_wavect_csv(output_dir / "inversion_input.csv")
    rows, station_coordinates = _cmat_detail_rows(dataset)
    detail_csv = output_dir / "inversion_input_detail.csv"
    _write_detail_rows(detail_csv, rows)
    station_file = output_dir / "stations_for_picker.txt"
    _write_station_file(station_file, station_coordinates)
    geometry, model_path = find_model_geometry(candidate.period_root)
    if geometry is None:
        summary = dataset.summary()
        lows = np.minimum(summary.source_min_xyz, summary.receiver_min_xyz)
        highs = np.maximum(summary.source_max_xyz, summary.receiver_max_xyz)
        pad = np.maximum((highs - lows) * 0.05, 10.0)
        geometry = CMATModelGeometry(
            x_min=float(lows[0] - pad[0]),
            x_max=float(highs[0] + pad[0]),
            y_min=float(lows[1] - pad[1]),
            y_max=float(highs[1] + pad[1]),
            z_min=float(lows[2] - pad[2]),
            z_max=float(highs[2] + pad[2]),
            nx_nodes=30,
            ny_nodes=24,
            nz_nodes=18,
            initial_velocity_mps=summary.apparent_velocity_median_mps,
        )
    slices = read_reference_slice_levels(candidate.period_root)
    if not slices:
        slices = tuple(
            float(value)
            for value in np.linspace(geometry.z_max, geometry.z_min, 5)[1:-1]
        )
    boundary_file = output_dir / "workface_boundary.csv"
    _write_boundary(boundary_file, geometry)
    cad_files = sorted(candidate.period_root.glob("*.dxf"))
    if not cad_files:
        cad_files = sorted(candidate.period_root.glob("*.dwg"))
    dataset_id = _project_slug(
        "cmat_3dd", candidate.period_root, dataset.start_date, dataset.end_date
    )
    project = {
        "project_name": (
            f"测试数据集728 {dataset.start_date}至{dataset.end_date} CMAT射线CT"
        ),
        "report_template": "generic",
        "dataset": {
            "dataset_id": dataset_id,
            "data_type": "real",
            "independence_group": "xuzhuang_7432",
            "adapter": "cmat_3dd",
            "raw_root": str(candidate.period_root),
        },
        "inputs": {
            "travel_time_csv": str(inversion_csv),
            "detail_csv": str(detail_csv),
            "pick_audit_csv": "",
            "waveform_root": "",
            "station_file": str(station_file),
            "evidence_csv": "",
        },
        "workface": {
            "boundary_file": str(boundary_file),
            "basemap_file": str(cad_files[0]) if cad_files else "",
            "mapa_file": "",
            "cad_x_offset": 0.0,
            "cad_y_offset": 0.0,
        },
        "outputs": {"directory": str(output_dir / "反演结果")},
        "model": _model_config(
            geometry,
            slices,
            adapter="cmat_3dd",
            background_velocity_mps=0.0,
        ),
    }
    project_path = save_project_config(output_dir / "wave_ct_project.json", project)
    summary = dataset.summary().as_dict()
    summary.update(
        {
            "schema_version": 1,
            "dataset_id": dataset_id,
            "adapter": "cmat_3dd",
            "input_kind": "vendor_relative_travel_time_rays",
            "waveform_validation_status": "SKIPPED_NO_LINKED_RAW_WAVEFORMS",
            "reference_geometry_only": geometry.as_dict(),
            "reference_model_file": str(model_path) if model_path else "",
            "reference_velocity_values_used_as_prior": False,
            "requested_slice_z": list(slices),
            "input_sha256": _sha256(candidate.primary_input),
            "provenance": {
                "source_receiver_travel_time": "binary CMAT .3dd records",
                "event_grouping": "exact source coordinate triple, first occurrence",
                "historical_result_fields": "evaluation geometry only; never solver input",
            },
            "outputs": {
                "inversion_csv": str(inversion_csv),
                "detail_csv": str(detail_csv.resolve()),
                "project": str(project_path),
            },
        }
    )
    _write_json(output_dir / "dataset_summary.json", summary)
    return project_path


def import_dataset(
    dataset_root: Path | str,
    output_root: Path | str,
) -> tuple[Path, ...]:
    """Discover and import every independently runnable period under a dataset."""

    candidates = discover_import_candidates(dataset_root)
    if not candidates:
        raise RuntimeError(f"no supported WaveCT inputs discovered under {dataset_root}")
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    projects: list[Path] = []
    used_names: Counter[str] = Counter()
    for candidate in candidates:
        if candidate.adapter == "cmat_3dd":
            dataset = read_cmat_3dd(candidate.primary_input)
            name = f"{dataset.start_date:%Y%m%d}-{dataset.end_date:%Y%m%d}"
        else:
            headers = [
                read_sos_w_header(path)
                for path in sorted(candidate.waveform_dir.glob("*.W"))  # type: ignore[union-attr]
            ]
            name = (
                f"{min(header.event_time for header in headers):%Y%m%d}-"
                f"{max(header.event_time for header in headers):%Y%m%d}"
            )
        used_names[name] += 1
        if used_names[name] > 1:
            name = f"{name}-{candidate.adapter}"
        destination = output_root / name
        if candidate.adapter == "cmat_3dd":
            projects.append(import_cmat_candidate(candidate, destination))
        elif candidate.adapter == "sos_w":
            projects.append(import_sos_w_candidate(candidate, destination))
        else:  # pragma: no cover - discovery controls this
            raise ValueError(f"unsupported import adapter: {candidate.adapter}")
    manifest = {
        "schema_version": 1,
        "dataset_root": str(Path(dataset_root).resolve()),
        "projects": [str(path.resolve()) for path in projects],
        "independence_groups": ["xuzhuang_7432"],
    }
    _write_json(output_root / "import_manifest.json", manifest)
    return tuple(projects)
