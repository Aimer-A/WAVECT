"""Strict reader and WaveCT CSV converter for CMAT ``.3dd`` ray files.

The supported binary layout is intentionally small and explicit::

    header  = <4s6HI   # magic, start Y/M/D, end Y/M/D, ray count
    record  = <i7f     # ray id, source xyz, receiver xyz, travel time (ms)

CMAT stores a relative travel time rather than an absolute event clock.  The
WaveCT CSV export therefore writes a zero millisecond event origin and uses the
same positive relative travel time for both the P-arrival and propagation-time
columns.  This preserves the identity required by :mod:`wave_ct.inversion`
without inventing an absolute origin time.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import date
import math
from pathlib import Path
import statistics
import struct
from typing import Dict, Iterable, Tuple


CMAT_MAGIC = b"CMAT"
CMAT_HEADER_STRUCT = struct.Struct("<4s6HI")
CMAT_RAY_STRUCT = struct.Struct("<i7f")

WAVECT_INVERSION_COLUMNS = (
    "震源编号",
    "震源坐标-x",
    "震源坐标-y",
    "震源坐标-z",
    "发震时刻t",
    "台站坐标-x",
    "台站坐标-y",
    "台站坐标-z",
    "台站P波到时",
    "震源-台站传播时间",
    "震源事件文件名",
)


@dataclass(frozen=True)
class CMATRay:
    """One CMAT source-receiver travel-time observation."""

    ray_id: int
    source_x: float
    source_y: float
    source_z: float
    receiver_x: float
    receiver_y: float
    receiver_z: float
    travel_time_ms: float

    @property
    def source_xyz(self) -> Tuple[float, float, float]:
        return self.source_x, self.source_y, self.source_z

    @property
    def receiver_xyz(self) -> Tuple[float, float, float]:
        return self.receiver_x, self.receiver_y, self.receiver_z

    @property
    def distance_m(self) -> float:
        return math.sqrt(
            (self.receiver_x - self.source_x) ** 2
            + (self.receiver_y - self.source_y) ** 2
            + (self.receiver_z - self.source_z) ** 2
        )

    @property
    def apparent_velocity_mps(self) -> float:
        return self.distance_m / (self.travel_time_ms / 1000.0)


@dataclass(frozen=True)
class CMATSummary:
    """Auditable descriptive statistics for one parsed CMAT dataset."""

    path: str
    start_date: str
    end_date: str
    ray_count: int
    source_count: int
    receiver_count: int
    source_min_xyz: Tuple[float, float, float]
    source_max_xyz: Tuple[float, float, float]
    receiver_min_xyz: Tuple[float, float, float]
    receiver_max_xyz: Tuple[float, float, float]
    travel_time_min_ms: float
    travel_time_median_ms: float
    travel_time_max_ms: float
    distance_min_m: float
    distance_median_m: float
    distance_max_m: float
    apparent_velocity_min_mps: float
    apparent_velocity_median_mps: float
    apparent_velocity_max_mps: float

    def as_dict(self) -> Dict[str, object]:
        """Return a JSON-serializable dictionary."""

        return asdict(self)


@dataclass(frozen=True)
class CMATDataset:
    """Validated contents of a CMAT ``.3dd`` file."""

    path: Path
    start_date: date
    end_date: date
    rays: Tuple[CMATRay, ...]

    def summary(self) -> CMATSummary:
        return summarize_cmat(self)

    def write_wavect_csv(self, output_path: Path | str) -> Path:
        return write_wavect_csv(self, output_path)


def _axis_bounds(
    points: Iterable[Tuple[float, float, float]],
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    point_list = list(points)
    return (
        tuple(min(point[axis] for point in point_list) for axis in range(3)),
        tuple(max(point[axis] for point in point_list) for axis in range(3)),
    )


def _format_float(value: float) -> str:
    """Round-trip a binary float without adding presentation-only rounding."""

    return format(float(value), ".9g")


def read_cmat_3dd(path: Path | str) -> CMATDataset:
    """Read a CMAT ray file and reject any structural or numeric corruption."""

    source_path = Path(path)
    file_size = source_path.stat().st_size
    if file_size < CMAT_HEADER_STRUCT.size:
        raise ValueError(
            f"truncated CMAT header: {file_size}/{CMAT_HEADER_STRUCT.size} bytes"
        )

    with source_path.open("rb") as handle:
        header_bytes = handle.read(CMAT_HEADER_STRUCT.size)
        try:
            (
                magic,
                start_year,
                start_month,
                start_day,
                end_year,
                end_month,
                end_day,
                ray_count,
            ) = CMAT_HEADER_STRUCT.unpack(header_bytes)
        except struct.error as exc:
            raise ValueError("invalid CMAT header") from exc

        if magic != CMAT_MAGIC:
            raise ValueError(
                f"not a supported CMAT .3dd file: magic={magic!r}, "
                f"expected={CMAT_MAGIC!r}"
            )
        if ray_count <= 0:
            raise ValueError(f"CMAT file contains no rays: count={ray_count}")

        expected_size = CMAT_HEADER_STRUCT.size + ray_count * CMAT_RAY_STRUCT.size
        if file_size != expected_size:
            condition = "truncated" if file_size < expected_size else "trailing data"
            raise ValueError(
                f"CMAT file size mismatch ({condition}): actual={file_size}, "
                f"expected={expected_size}, ray_count={ray_count}"
            )

        try:
            start = date(start_year, start_month, start_day)
            end = date(end_year, end_month, end_day)
        except ValueError as exc:
            raise ValueError(f"invalid CMAT date range in {source_path}") from exc
        if end < start:
            raise ValueError(f"CMAT end date {end.isoformat()} precedes {start.isoformat()}")

        rays = []
        for record_index in range(ray_count):
            offset = CMAT_HEADER_STRUCT.size + record_index * CMAT_RAY_STRUCT.size
            raw = handle.read(CMAT_RAY_STRUCT.size)
            if len(raw) != CMAT_RAY_STRUCT.size:
                raise ValueError(f"truncated CMAT ray record at byte {offset}")
            ray_id, *numeric = CMAT_RAY_STRUCT.unpack(raw)
            coordinates_and_time = tuple(float(value) for value in numeric)
            if not all(math.isfinite(value) for value in coordinates_and_time):
                raise ValueError(
                    f"non-finite CMAT value in record {record_index + 1} "
                    f"(ray id {ray_id})"
                )
            travel_time_ms = coordinates_and_time[6]
            if travel_time_ms <= 0.0:
                raise ValueError(
                    f"non-positive CMAT travel time in record {record_index + 1} "
                    f"(ray id {ray_id}): {travel_time_ms} ms"
                )
            rays.append(CMATRay(ray_id, *coordinates_and_time))

    return CMATDataset(
        path=source_path.resolve(),
        start_date=start,
        end_date=end,
        rays=tuple(rays),
    )


def summarize_cmat(dataset: CMATDataset) -> CMATSummary:
    """Calculate geometry, timing, and apparent-velocity diagnostics."""

    if not dataset.rays:
        raise ValueError("cannot summarize an empty CMAT dataset")

    sources = tuple(ray.source_xyz for ray in dataset.rays)
    receivers = tuple(ray.receiver_xyz for ray in dataset.rays)
    source_min, source_max = _axis_bounds(sources)
    receiver_min, receiver_max = _axis_bounds(receivers)
    times = tuple(ray.travel_time_ms for ray in dataset.rays)
    distances = tuple(ray.distance_m for ray in dataset.rays)
    apparent_velocities = tuple(ray.apparent_velocity_mps for ray in dataset.rays)

    return CMATSummary(
        path=str(dataset.path),
        start_date=dataset.start_date.isoformat(),
        end_date=dataset.end_date.isoformat(),
        ray_count=len(dataset.rays),
        source_count=len(set(sources)),
        receiver_count=len(set(receivers)),
        source_min_xyz=source_min,
        source_max_xyz=source_max,
        receiver_min_xyz=receiver_min,
        receiver_max_xyz=receiver_max,
        travel_time_min_ms=min(times),
        travel_time_median_ms=float(statistics.median(times)),
        travel_time_max_ms=max(times),
        distance_min_m=min(distances),
        distance_median_m=float(statistics.median(distances)),
        distance_max_m=max(distances),
        apparent_velocity_min_mps=min(apparent_velocities),
        apparent_velocity_median_mps=float(statistics.median(apparent_velocities)),
        apparent_velocity_max_mps=max(apparent_velocities),
    )


def write_wavect_csv(dataset: CMATDataset, output_path: Path | str) -> Path:
    """Write the exact 11-column WaveCT inversion CSV contract.

    Event identifiers are assigned by first occurrence of each exact source
    coordinate triple.  ``ray_id`` is deliberately not used as the event id:
    CMAT uses it to identify individual ray records, while WaveCT's static
    corrections and event-split validation require all receivers from the same
    source event to share an identifier.
    """

    if not dataset.rays:
        raise ValueError("cannot export an empty CMAT dataset")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    event_ids: Dict[Tuple[float, float, float], int] = {}
    with destination.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=WAVECT_INVERSION_COLUMNS)
        writer.writeheader()
        for ray in dataset.rays:
            event_id = event_ids.setdefault(ray.source_xyz, len(event_ids) + 1)
            travel_text = _format_float(ray.travel_time_ms)
            writer.writerow(
                {
                    "震源编号": event_id,
                    "震源坐标-x": _format_float(ray.source_x),
                    "震源坐标-y": _format_float(ray.source_y),
                    "震源坐标-z": _format_float(ray.source_z),
                    "发震时刻t": "0",
                    "台站坐标-x": _format_float(ray.receiver_x),
                    "台站坐标-y": _format_float(ray.receiver_y),
                    "台站坐标-z": _format_float(ray.receiver_z),
                    "台站P波到时": travel_text,
                    "震源-台站传播时间": travel_text,
                    "震源事件文件名": (
                        f"{dataset.path.stem}_source_{event_id:06d}"
                    ),
                }
            )
    return destination.resolve()

