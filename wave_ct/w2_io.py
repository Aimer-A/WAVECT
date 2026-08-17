"""Reader for the SOS ``.W2`` waveform container used by Wave CT.

The layout implemented here was verified against the files in the 720 test
dataset.  The parser deliberately validates every offset and file-size
identity before exposing waveform samples; a format mismatch therefore fails
open instead of drawing a plausible but incorrect trace.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct
from typing import Iterable

import numpy as np


W2_MAGIC = b"SOS\x00"
MAIN_HEADER_SIZE = 128
STATION_RECORD_SIZE = 96
SAMPLE_DTYPE = np.dtype("<f4")


@dataclass(frozen=True)
class W2StationRecord:
    """Metadata for one station record and its sequential waveform channel."""

    record_index: int
    station_id: int
    sample_rate_hz: int
    sample_count: int
    x: float
    y: float
    z: float
    label: str
    p_index: int
    marker_2: int
    marker_3: int
    marker_4: int
    s_velocity_mps: float
    p_velocity_mps: float

    @property
    def station_name(self) -> str:
        return f"STA{self.station_id:02d}"

    @property
    def p_time_sec(self) -> float | None:
        if self.p_index <= 0 or self.sample_rate_hz <= 0:
            return None
        return self.p_index / float(self.sample_rate_hz)


@dataclass(frozen=True)
class W2Header:
    path: Path
    header_size: int
    station_count: int
    data_slot_count: int
    data_offset: int
    origin_ms: int
    source_x: float
    source_y: float
    source_z: float
    s_velocity_mps: float
    p_velocity_mps: float
    records: tuple[W2StationRecord, ...]

    @property
    def source_xyz(self) -> tuple[float, float, float]:
        return self.source_x, self.source_y, self.source_z

    def station(self, station_id: int) -> W2StationRecord:
        for record in self.records:
            if record.station_id == station_id:
                return record
        raise KeyError(f"station {station_id} is not present in {self.path.name}")


def _unpack(fmt: str, data: bytes, offset: int):
    try:
        return struct.unpack_from(fmt, data, offset)
    except struct.error as exc:
        raise ValueError(f"truncated W2 metadata at byte {offset}") from exc


def _decode_label(raw: bytes) -> str:
    return raw.split(b"\x00", 1)[0].decode("ascii", errors="ignore").strip()


def read_w2_header(path: Path | str) -> W2Header:
    """Read and validate W2 event/station metadata without loading amplitudes."""

    path = Path(path)
    file_size = path.stat().st_size
    with path.open("rb") as handle:
        metadata = handle.read(4096)

    if len(metadata) < MAIN_HEADER_SIZE or metadata[:4] != W2_MAGIC:
        raise ValueError(f"not a supported SOS W2 file: {path}")

    header_size = int(_unpack("<H", metadata, 6)[0])
    if header_size != MAIN_HEADER_SIZE:
        raise ValueError(
            f"unsupported W2 header size {header_size}; expected {MAIN_HEADER_SIZE}"
        )

    count_a = int(_unpack("<H", metadata, 16)[0])
    count_b = int(_unpack("<H", metadata, 18)[0])
    station_count = count_a if 0 < count_a <= 32 else count_b
    if not 0 < station_count <= 32:
        raise ValueError(f"invalid W2 station count: {count_a}/{count_b}")

    metadata_end = header_size + station_count * STATION_RECORD_SIZE
    if len(metadata) < metadata_end:
        with path.open("rb") as handle:
            metadata = handle.read(metadata_end)
    if len(metadata) < metadata_end:
        raise ValueError(f"truncated W2 station table: {path}")

    origin_ms = int(_unpack("<I", metadata, 20)[0])
    source_x = float(_unpack("<d", metadata, 24)[0])
    source_y = float(_unpack("<d", metadata, 32)[0])
    source_z = float(_unpack("<d", metadata, 40)[0])
    s_velocity = float(_unpack("<f", metadata, 68)[0])
    p_velocity = float(_unpack("<f", metadata, 72)[0])

    records: list[W2StationRecord] = []
    station_ids: set[int] = set()
    for record_index in range(station_count):
        offset = header_size + record_index * STATION_RECORD_SIZE
        station_id = int(_unpack("<H", metadata, offset + 2)[0])
        sample_rate = int(_unpack("<H", metadata, offset + 4)[0])
        sample_count = int(_unpack("<H", metadata, offset + 6)[0])
        x = float(_unpack("<d", metadata, offset + 24)[0])
        y = float(_unpack("<d", metadata, offset + 32)[0])
        z = float(_unpack("<d", metadata, offset + 40)[0])
        label = _decode_label(metadata[offset + 48 : offset + 64])
        markers = tuple(int(value) for value in _unpack("<4H", metadata, offset + 80))
        station_s_velocity = float(_unpack("<f", metadata, offset + 88)[0])
        station_p_velocity = float(_unpack("<f", metadata, offset + 92)[0])

        if station_id <= 0 or station_id > 9999:
            raise ValueError(f"invalid station id {station_id} in {path.name}")
        if station_id in station_ids:
            raise ValueError(f"duplicate station id {station_id} in {path.name}")
        if sample_rate <= 0 or sample_count <= 0:
            raise ValueError(
                f"invalid sample metadata for station {station_id}: "
                f"rate={sample_rate}, count={sample_count}"
            )
        station_ids.add(station_id)
        records.append(
            W2StationRecord(
                record_index=record_index,
                station_id=station_id,
                sample_rate_hz=sample_rate,
                sample_count=sample_count,
                x=x,
                y=y,
                z=z,
                label=label,
                p_index=markers[0],
                marker_2=markers[1],
                marker_3=markers[2],
                marker_4=markers[3],
                s_velocity_mps=station_s_velocity,
                p_velocity_mps=station_p_velocity,
            )
        )

    sample_counts = {record.sample_count for record in records}
    sample_rates = {record.sample_rate_hz for record in records}
    if len(sample_counts) != 1 or len(sample_rates) != 1:
        raise ValueError(f"mixed sample layout is not supported: {path.name}")
    sample_count = records[0].sample_count

    payload_bytes = file_size - metadata_end
    bytes_per_slot = sample_count * SAMPLE_DTYPE.itemsize
    data_slot_count, remainder = divmod(payload_bytes, bytes_per_slot)
    if remainder or data_slot_count < station_count or data_slot_count > 64:
        raise ValueError(
            "W2 waveform payload size mismatch: "
            f"payload={payload_bytes}, samples={sample_count}, slots={data_slot_count}, "
            f"remainder={remainder}"
        )

    finite_coordinates = np.asarray(
        [[source_x, source_y, source_z]] + [[r.x, r.y, r.z] for r in records],
        dtype=np.float64,
    )
    if not np.isfinite(finite_coordinates).all():
        raise ValueError(f"non-finite W2 coordinates: {path.name}")

    return W2Header(
        path=path.resolve(),
        header_size=header_size,
        station_count=station_count,
        data_slot_count=data_slot_count,
        data_offset=metadata_end,
        origin_ms=origin_ms,
        source_x=source_x,
        source_y=source_y,
        source_z=source_z,
        s_velocity_mps=s_velocity,
        p_velocity_mps=p_velocity,
        records=tuple(records),
    )


def read_w2_trace(
    path: Path | str,
    *,
    station_id: int | None = None,
    record_index: int | None = None,
) -> tuple[np.ndarray, np.ndarray, W2Header, W2StationRecord]:
    """Load one real waveform channel and its physical time axis.

    Waveform channels are stored sequentially in station-record order.  This is
    intentionally not indexed by station number because the station ids in the
    file are sparse (for example 1, 2, 4, ...).
    """

    header = read_w2_header(path)
    if record_index is None:
        if station_id is None:
            raise ValueError("station_id or record_index is required")
        record = header.station(int(station_id))
    else:
        if record_index < 0 or record_index >= len(header.records):
            raise IndexError(f"W2 record index out of range: {record_index}")
        record = header.records[int(record_index)]
        if station_id is not None and record.station_id != int(station_id):
            raise ValueError(
                f"W2 station mismatch: record {record_index} is "
                f"STA{record.station_id:02d}, not STA{int(station_id):02d}"
            )

    offset = header.data_offset + (
        record.record_index * record.sample_count * SAMPLE_DTYPE.itemsize
    )
    amplitude = np.fromfile(
        header.path,
        dtype=SAMPLE_DTYPE,
        count=record.sample_count,
        offset=offset,
    ).astype(np.float64)
    if amplitude.size != record.sample_count:
        raise ValueError(
            f"truncated W2 trace for STA{record.station_id:02d}: "
            f"{amplitude.size}/{record.sample_count} samples"
        )
    time_sec = np.arange(record.sample_count, dtype=np.float64) / float(
        record.sample_rate_hz
    )
    return time_sec, amplitude, header, record


def iter_marked_records(header: W2Header) -> Iterable[W2StationRecord]:
    return (record for record in header.records if record.p_index > 0)

