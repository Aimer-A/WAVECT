"""Strict metadata reader for the legacy SOS ``.W`` event container.

Only the parts independently verified against the Xu-Zhuang 728 files are
decoded here: event time, trace layout and the four integer phase markers.
The payload uses a proprietary four-byte sample encoding and is intentionally
returned as raw bytes.  WaveCT must not claim amplitude or full-waveform
validation until that codec is independently documented.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import struct
from typing import Iterable


W_MAGIC = 0xFD
DATA_OFFSET = 1632
GLOBAL_MARKER_OFFSET = 1024
CHANNEL_TABLE_OFFSET = 1040
CHANNEL_RECORD_SIZE = 19
BYTES_PER_SAMPLE = 4


@dataclass(frozen=True)
class SOSWTraceRecord:
    """One logical trace block and its phase-marker metadata."""

    block_index: int
    channel_id: int
    flag: int
    p_index: int
    marker_2: int
    marker_3: int
    marker_4: int

    @property
    def markers(self) -> tuple[int, int, int, int]:
        return self.p_index, self.marker_2, self.marker_3, self.marker_4


@dataclass(frozen=True)
class SOSWHeader:
    """Validated header of one SOS ``.W`` event file."""

    path: Path
    event_time: datetime
    millisecond: int
    trace_count: int
    sample_rate_hz: int
    sample_count: int
    data_offset: int
    records: tuple[SOSWTraceRecord, ...]

    def record_for_block(self, block_index: int) -> SOSWTraceRecord:
        if block_index < 0 or block_index >= len(self.records):
            raise IndexError(f"SOS W block index out of range: {block_index}")
        return self.records[block_index]


def _unpack(fmt: str, data: bytes, offset: int):
    try:
        return struct.unpack_from(fmt, data, offset)
    except struct.error as exc:
        raise ValueError(f"truncated SOS W metadata at byte {offset}") from exc


def _validate_datetime(values: tuple[int, ...], path: Path) -> tuple[datetime, int]:
    year, month, day, hour, minute, second, millisecond = values
    if not 0 <= millisecond <= 999:
        raise ValueError(f"invalid SOS W millisecond {millisecond}: {path.name}")
    try:
        value = datetime(year, month, day, hour, minute, second)
    except ValueError as exc:
        raise ValueError(f"invalid SOS W event time in {path.name}: {values}") from exc
    return value, millisecond


def read_sos_w_header(path: Path | str) -> SOSWHeader:
    """Read and validate event/marker metadata without decoding amplitudes.

    The first trace uses the global marker quartet at byte 1024 and represents
    channel 0.  Trace blocks 1..N-1 use the first N-1 entries of the 19-byte
    channel table at byte 1040.  A trailing table entry is a cache/helper
    record and has no corresponding payload block, so it is deliberately not
    exposed.
    """

    path = Path(path)
    file_size = path.stat().st_size
    if file_size < DATA_OFFSET:
        raise ValueError(f"SOS W file is shorter than its header: {path}")
    with path.open("rb") as handle:
        metadata = handle.read(DATA_OFFSET)
    if len(metadata) != DATA_OFFSET or metadata[0] != W_MAGIC:
        raise ValueError(f"not a supported SOS W file: {path}")

    event_time, millisecond = _validate_datetime(
        tuple(int(value) for value in _unpack("<7H", metadata, 1)),
        path,
    )
    trace_count = int(_unpack("<H", metadata, 15)[0])
    sample_rate_hz = int(_unpack("<I", metadata, 17)[0])
    sample_count = int(_unpack("<I", metadata, 21)[0])
    if not 1 <= trace_count <= 64:
        raise ValueError(f"invalid SOS W trace count {trace_count}: {path.name}")
    if not 1 <= sample_rate_hz <= 1_000_000:
        raise ValueError(f"invalid SOS W sample rate {sample_rate_hz}: {path.name}")
    if not 1 <= sample_count <= 100_000_000:
        raise ValueError(f"invalid SOS W sample count {sample_count}: {path.name}")

    expected_size = DATA_OFFSET + (
        trace_count * sample_count * BYTES_PER_SAMPLE
    )
    if file_size != expected_size:
        raise ValueError(
            "SOS W payload size mismatch: "
            f"actual={file_size}, expected={expected_size}, "
            f"traces={trace_count}, samples={sample_count}"
        )
    table_end = CHANNEL_TABLE_OFFSET + trace_count * CHANNEL_RECORD_SIZE
    if table_end > DATA_OFFSET:
        raise ValueError(
            f"SOS W channel table exceeds fixed header: {table_end}>{DATA_OFFSET}"
        )

    global_markers = tuple(
        int(value) for value in _unpack("<4I", metadata, GLOBAL_MARKER_OFFSET)
    )
    records: list[SOSWTraceRecord] = [
        SOSWTraceRecord(
            block_index=0,
            channel_id=0,
            flag=1,
            p_index=global_markers[0],
            marker_2=global_markers[1],
            marker_3=global_markers[2],
            marker_4=global_markers[3],
        )
    ]
    seen_channel_ids = {0}
    for block_index in range(1, trace_count):
        offset = CHANNEL_TABLE_OFFSET + (block_index - 1) * CHANNEL_RECORD_SIZE
        channel_id, flag, p_index, marker_2, marker_3, marker_4 = _unpack(
            "<HB4I", metadata, offset
        )
        channel_id = int(channel_id)
        if channel_id <= 0 or channel_id > 65_535:
            raise ValueError(
                f"invalid SOS W channel id {channel_id} at block {block_index}"
            )
        if channel_id in seen_channel_ids:
            raise ValueError(
                f"duplicate SOS W channel id {channel_id}: {path.name}"
            )
        seen_channel_ids.add(channel_id)
        records.append(
            SOSWTraceRecord(
                block_index=block_index,
                channel_id=channel_id,
                flag=int(flag),
                p_index=int(p_index),
                marker_2=int(marker_2),
                marker_3=int(marker_3),
                marker_4=int(marker_4),
            )
        )

    return SOSWHeader(
        path=path.resolve(),
        event_time=event_time,
        millisecond=millisecond,
        trace_count=trace_count,
        sample_rate_hz=sample_rate_hz,
        sample_count=sample_count,
        data_offset=DATA_OFFSET,
        records=tuple(records),
    )


def iter_marked_records(header: SOSWHeader) -> Iterable[SOSWTraceRecord]:
    """Yield records whose P candidate is inside the trace."""

    for record in header.records:
        if 0 < record.p_index < header.sample_count:
            yield record


def read_sos_w_raw_trace(
    path: Path | str,
    *,
    block_index: int,
) -> tuple[bytes, SOSWHeader, SOSWTraceRecord]:
    """Return one proprietary trace block as raw bytes.

    No numeric amplitude conversion is attempted.  Consumers may use this for
    byte-level audits or a future vendor codec without silently interpreting
    the payload as IEEE float or integer amplitudes.
    """

    header = read_sos_w_header(path)
    record = header.record_for_block(block_index)
    block_bytes = header.sample_count * BYTES_PER_SAMPLE
    offset = header.data_offset + block_index * block_bytes
    with header.path.open("rb") as handle:
        handle.seek(offset)
        payload = handle.read(block_bytes)
    if len(payload) != block_bytes:
        raise ValueError(
            f"truncated SOS W trace block {block_index}: "
            f"{len(payload)}/{block_bytes} bytes"
        )
    return payload, header, record


def station_coordinate_index(
    header: SOSWHeader,
    record: SOSWTraceRecord,
    *,
    layout: str = "auto",
) -> int:
    """Map a trace record to the zero-based PARAM.LOK coordinate row.

    In the verified 728 acquisition the payload/table *position* is the stable
    PARAM coordinate index.  Hardware ``channel_id`` values become sparse
    after removals (for example block 23 has id 25 in the 28-trace layout), so
    using the id as an array index gives a large, systematic timing error.
    Unknown trace layouts fail closed unless the caller supplies an explicit
    mapping mode.
    """

    if record.block_index == 0:
        return 0
    mode = layout
    if mode == "auto":
        if header.trace_count in {28, 30}:
            mode = "block"
        else:
            raise ValueError(
                "unknown SOS W channel layout; specify 'block', 'same' or "
                "'minus_one': "
                f"trace_count={header.trace_count}"
            )
    if mode == "block":
        return record.block_index
    if mode == "same":
        return record.channel_id
    if mode == "minus_one":
        return record.channel_id - 1
    raise ValueError(f"unsupported SOS W channel layout: {layout}")
