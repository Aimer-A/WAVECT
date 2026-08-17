"""Readers for SOS ``PARAM.LOK`` and ``PARAM_SOSDUMP.LOK`` station tables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class SOSStation:
    coordinate_index: int
    station_id: str
    x: float
    y: float
    z: float
    enabled: bool

    @property
    def xyz(self) -> tuple[float, float, float]:
        return self.x, self.y, self.z


def _float_fields(line: str) -> tuple[float, ...] | None:
    try:
        values = tuple(float(value) for value in line.split())
    except ValueError:
        return None
    return values if values else None


def read_param_lok(path: Path | str) -> tuple[float, tuple[tuple[float, float, float], ...]]:
    """Read P reference velocity and historical station coordinates.

    SOS versions place a variable settings block before one N-row calibration
    table and one N-row coordinate table.  The reader locates those adjacent
    2-column/3-column runs instead of depending on a hard-coded line number.
    """

    path = Path(path)
    lines = path.read_text(encoding="gbk", errors="strict").splitlines()
    if len(lines) < 4:
        raise ValueError(f"PARAM.LOK is too short: {path}")
    try:
        reference_velocity = float(lines[0].strip())
        station_count = int(float(lines[3].strip()))
    except ValueError as exc:
        raise ValueError(f"invalid PARAM.LOK header: {path}") from exc
    if not np.isfinite(reference_velocity) or reference_velocity <= 0:
        raise ValueError(f"invalid PARAM.LOK P velocity: {reference_velocity}")
    if not 1 <= station_count <= 10_000:
        raise ValueError(f"invalid PARAM.LOK station count: {station_count}")

    fields = [_float_fields(line) for line in lines]
    coordinate_start: int | None = None
    for start in range(4, len(fields) - 2 * station_count + 1):
        calibration = fields[start : start + station_count]
        coordinates = fields[
            start + station_count : start + 2 * station_count
        ]
        if (
            all(value is not None and len(value) == 2 for value in calibration)
            and all(value is not None and len(value) == 3 for value in coordinates)
        ):
            coordinate_start = start + station_count
            break
    if coordinate_start is None:
        raise ValueError(
            f"cannot locate {station_count} station coordinates in {path}"
        )
    coordinate_rows = tuple(
        tuple(float(value) for value in fields[index])  # type: ignore[arg-type]
        for index in range(coordinate_start, coordinate_start + station_count)
    )
    coordinate_array = np.asarray(coordinate_rows, dtype=np.float64)
    if coordinate_array.shape != (station_count, 3):
        raise ValueError(f"invalid PARAM.LOK coordinate shape: {coordinate_array.shape}")
    if not np.isfinite(coordinate_array).all():
        raise ValueError(f"non-finite PARAM.LOK station coordinate: {path}")
    return reference_velocity, coordinate_rows


def read_sosdump_lok(path: Path | str) -> tuple[tuple[str, tuple[float, float, float]], ...]:
    """Read the semicolon-delimited station-ID export used for ID auditing."""

    path = Path(path)
    rows: list[tuple[str, tuple[float, float, float]]] = []
    for line_number, raw in enumerate(
        path.read_text(encoding="gbk", errors="strict").splitlines(), start=1
    ):
        if not raw.strip():
            continue
        parts = [value.strip() for value in raw.split(";")]
        if len(parts) < 4 or not parts[0]:
            raise ValueError(f"invalid SOSDUMP row {line_number}: {raw!r}")
        try:
            xyz = tuple(float(value) for value in parts[1:4])
        except ValueError as exc:
            raise ValueError(
                f"invalid SOSDUMP coordinates at row {line_number}: {raw!r}"
            ) from exc
        if not np.isfinite(np.asarray(xyz)).all():
            raise ValueError(f"non-finite SOSDUMP row {line_number}")
        rows.append((parts[0], xyz))  # type: ignore[arg-type]
    if not rows:
        raise ValueError(f"SOSDUMP station table is empty: {path}")
    return tuple(rows)


def merge_station_tables(
    param_path: Path | str,
    sosdump_path: Path | str | None = None,
) -> tuple[float, tuple[SOSStation, ...], tuple[str, ...]]:
    """Use same-period PARAM coordinates and SOSDUMP IDs with drift warnings."""

    reference_velocity, coordinates = read_param_lok(param_path)
    ids: list[str] = [f"STA{index:03d}" for index in range(len(coordinates))]
    dump_rows: tuple[tuple[str, tuple[float, float, float]], ...] = ()
    warnings: list[str] = []
    if sosdump_path is not None and Path(sosdump_path).is_file():
        dump_rows = read_sosdump_lok(sosdump_path)
        if len(dump_rows) != len(coordinates):
            warnings.append(
                "SOSDUMP/PARAM station-count mismatch: "
                f"{len(dump_rows)} != {len(coordinates)}"
            )
        for index, (station_id, dump_xyz) in enumerate(dump_rows[: len(coordinates)]):
            ids[index] = station_id
            distance = float(
                np.linalg.norm(
                    np.asarray(coordinates[index], dtype=float)
                    - np.asarray(dump_xyz, dtype=float)
                )
            )
            if distance > 1.0:
                warnings.append(
                    f"station[{index}] {station_id} coordinate drift "
                    f"{distance:.2f} m; PARAM.LOK retained"
                )
    stations = tuple(
        SOSStation(
            coordinate_index=index,
            station_id=ids[index],
            x=float(xyz[0]),
            y=float(xyz[1]),
            z=float(xyz[2]),
            enabled=bool(np.linalg.norm(np.asarray(xyz, dtype=float)) > 1.0),
        )
        for index, xyz in enumerate(coordinates)
    )
    return reference_velocity, stations, tuple(warnings)
