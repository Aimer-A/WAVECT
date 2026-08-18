"""Convert stage1 simulated travel times to the CSV contract used by WaveCT.

Usage (from the repository root)::

    python test/prepare_wavect_simulation.py

The script never modifies stage1 files.  It converts seconds to milliseconds
because WaveCT's legacy inversion CSV contract stores travel time in ms.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


OUTPUT_COLUMNS = [
    "震源编号", "震源坐标-x", "震源坐标-y", "震源坐标-z", "发震时刻t",
    "台站坐标-x", "台站坐标-y", "台站坐标-z", "台站P波到时",
    "震源-台站传播时间", "震源事件文件名",
]


def finite_float(row: dict[str, str], name: str, line: int) -> float:
    try:
        value = float(row[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"line {line}: invalid {name!r}: {row.get(name)!r}") from exc
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError(f"line {line}: non-finite {name}")
    return value


def convert(input_csv: Path, output_csv: Path) -> dict[str, object]:
    required = {"source_id", "src_x", "src_y", "src_z", "rcv_x", "rcv_y", "rcv_z", "t_obs"}
    rows: list[dict[str, object]] = []
    sources: set[str] = set()
    receivers: set[tuple[float, float, float]] = set()

    with input_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"missing columns: {', '.join(sorted(missing))}")
        for line, raw in enumerate(reader, start=2):
            source = str(raw["source_id"]).strip()
            if not source:
                raise ValueError(f"line {line}: empty source_id")
            sx, sy, sz = (finite_float(raw, key, line) for key in ("src_x", "src_y", "src_z"))
            rx, ry, rz = (finite_float(raw, key, line) for key in ("rcv_x", "rcv_y", "rcv_z"))
            t_sec = finite_float(raw, "t_obs", line)
            if t_sec <= 0:
                raise ValueError(f"line {line}: t_obs must be positive")
            source_number = int(source.split("_")[-1]) if source.split("_")[-1].isdigit() else len(sources) + 1
            travel_ms = t_sec * 1000.0
            rows.append({
                "震源编号": source_number,
                "震源坐标-x": sx, "震源坐标-y": sy, "震源坐标-z": sz,
                "发震时刻t": 0.0,
                "台站坐标-x": rx, "台站坐标-y": ry, "台站坐标-z": rz,
                "台站P波到时": travel_ms,
                "震源-台站传播时间": travel_ms,
                "震源事件文件名": source,
            })
            sources.add(source)
            receivers.add((rx, ry, rz))

    if not rows:
        raise ValueError("input contains no travel-time rows")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "input": str(input_csv.resolve()),
        "output": str(output_csv.resolve()),
        "rows": len(rows),
        "events": len(sources),
        "receivers": len(receivers),
        "travel_time_unit_out": "ms",
        "source_of_truth": "test/output_stage1/travel_times_obs.csv",
    }
    output_csv.with_name("prepare_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Prepare stage1 simulation data for WaveCT")
    parser.add_argument("--input", type=Path, default=root / "output_stage1" / "travel_times_obs.csv")
    parser.add_argument("--output", type=Path, default=root / "wavect_input" / "inversion_input.csv")
    args = parser.parse_args()
    print(json.dumps(convert(args.input, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
