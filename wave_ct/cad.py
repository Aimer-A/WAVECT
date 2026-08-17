"""AutoCAD DWG conversion and coordinate-aware vector cache for Wave CT."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Iterable

import numpy as np


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_accoreconsole(explicit: Path | None = None) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    configured = os.environ.get("AUTOCAD_CORE_CONSOLE", "").strip()
    if configured:
        candidates.append(Path(configured))
    for drive in "CDEF":
        root = Path(f"{drive}:/")
        candidates.extend(root.glob("autocad/AutoCAD */accoreconsole.exe"))
        candidates.extend(root.glob("Program Files/Autodesk/AutoCAD */accoreconsole.exe"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError("未找到 accoreconsole.exe；请确认AutoCAD主体已安装。")


def _ascii_cache_root(core_console: Path | None = None, explicit: Path | None = None) -> Path:
    configured = os.environ.get("WAVECT_CAD_CACHE", "").strip()
    local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
    candidates = [
        explicit,
        Path(configured) if configured else None,
        Path(local_app_data) / "WaveCT" / "cad_cache" if local_app_data else None,
        Path(core_console.anchor) / "WaveCT_Cache" if core_console is not None else None,
    ]
    for candidate in candidates:
        if candidate is None or not str(candidate):
            continue
        try:
            str(candidate).encode("ascii")
        except UnicodeEncodeError:
            continue
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate
    raise OSError("无法创建AutoCAD英文缓存目录。")


def export_dwg_to_dxf(
    dwg_path: Path,
    core_console: Path | None = None,
    cache_dir: Path | None = None,
) -> Path:
    dwg_path = dwg_path.resolve()
    if not dwg_path.is_file():
        raise FileNotFoundError(f"DWG不存在: {dwg_path}")
    console = find_accoreconsole(core_console)
    cache_root = _ascii_cache_root(console, cache_dir)
    key = _file_digest(dwg_path)[:20]
    dxf_path = cache_root / f"{key}.dxf"
    if dxf_path.is_file() and dxf_path.stat().st_size > 1024:
        return dxf_path

    script_path = cache_root / f"{key}.scr"
    output_text = dxf_path.as_posix()
    script_path.write_text(
        "\n".join([
            '(setvar "FILEDIA" 0)',
            '(setvar "CMDDIA" 0)',
            f'(command "_.DXFOUT" "{output_text}" "16")',
            '(command "_.QUIT")',
            "",
        ]),
        encoding="ascii",
    )
    process = subprocess.run(
        [str(console), "/i", str(dwg_path), "/s", str(script_path)],
        cwd=str(cache_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=600,
        check=False,
    )
    if process.returncode != 0 or not dxf_path.is_file() or dxf_path.stat().st_size <= 1024:
        output = process.stdout.decode("utf-16-le", errors="replace")[-3000:]
        raise RuntimeError(f"AutoCAD DWG转DXF失败（代码{process.returncode}）。\n{output}")
    return dxf_path


def _pack_segments(segments: Iterable[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    segment_list = [np.asarray(segment, dtype=np.float64) for segment in segments if len(segment) >= 2]
    if not segment_list:
        return np.empty((0, 2), dtype=np.float64), np.asarray([0], dtype=np.int64)
    lengths = np.asarray([len(segment) for segment in segment_list], dtype=np.int64)
    offsets = np.concatenate([np.asarray([0], dtype=np.int64), np.cumsum(lengths)])
    return np.vstack(segment_list), offsets


def extract_dxf_segments(
    dxf_path: Path,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    x_offset: float,
    y_offset: float,
    flattening_distance: float = 1.0,
    cache_dir: Path | None = None,
) -> Path:
    params = json.dumps(
        ["cad_cache_schema_v2_text_labels", str(dxf_path.resolve()), _file_digest(dxf_path),
         *xlim, *ylim, x_offset, y_offset, flattening_distance],
        ensure_ascii=True,
    )
    key = hashlib.sha256(params.encode("ascii")).hexdigest()[:20]
    cache_root = _ascii_cache_root(explicit=cache_dir)
    cache_path = cache_root / f"{dxf_path.stem}_{key}_segments.npz"
    if cache_path.is_file():
        return cache_path

    # A valid segment cache is self-contained.  Only require ezdxf when a new
    # cache actually has to be built; otherwise machines used for reviewing
    # existing mine projects could incorrectly lose their basemap.
    try:
        import ezdxf
        from ezdxf.disassemble import recursive_decompose
        from ezdxf.path import make_path
    except ImportError as exc:
        raise RuntimeError("当前Python环境缺少ezdxf，无法读取AutoCAD导出的DXF。") from exc

    logging.disable(logging.CRITICAL)
    try:
        document = ezdxf.readfile(dxf_path)
        modelspace = document.modelspace()
        gx_min, gx_max = xlim[0] + x_offset, xlim[1] + x_offset
        gy_min, gy_max = ylim[0] + y_offset, ylim[1] + y_offset
        segments: list[np.ndarray] = []
        label_positions: list[tuple[float, float]] = []
        label_texts: list[str] = []
        label_rotations: list[float] = []
        entity_count = 0
        for entity in recursive_decompose(modelspace):
            entity_count += 1
            if entity.dxftype() in {"TEXT", "MTEXT", "ATTRIB"}:
                try:
                    insert = entity.dxf.insert
                    label_x = float(insert.x) - x_offset
                    label_y = float(insert.y) - y_offset
                    if not (
                        xlim[0] <= label_x <= xlim[1]
                        and ylim[0] <= label_y <= ylim[1]
                    ):
                        continue
                    if entity.dxftype() == "MTEXT":
                        text_value = str(entity.plain_text())
                    else:
                        text_value = str(entity.dxf.text)
                    text_value = " ".join(text_value.replace("\\P", " ").split())
                    if not text_value:
                        continue
                    label_positions.append((label_x, label_y))
                    label_texts.append(text_value[:256])
                    label_rotations.append(float(entity.dxf.get("rotation", 0.0)))
                except (AttributeError, TypeError, ValueError):
                    pass
                continue
            if entity.dxftype() == "POINT":
                continue
            try:
                path = make_path(entity)
            except Exception:
                continue
            subpaths = path.sub_paths() if path.has_sub_paths else [path]
            for subpath in subpaths:
                try:
                    points = np.asarray(
                        [(point.x, point.y) for point in subpath.flattening(
                            distance=flattening_distance, segments=8
                        )],
                        dtype=np.float64,
                    )
                except Exception:
                    continue
                if points.shape[0] < 2 or not np.isfinite(points).all():
                    continue
                if (
                    points[:, 0].max() < gx_min or points[:, 0].min() > gx_max
                    or points[:, 1].max() < gy_min or points[:, 1].min() > gy_max
                ):
                    continue
                points[:, 0] -= x_offset
                points[:, 1] -= y_offset
                segments.append(points)
        vertices, offsets = _pack_segments(segments)
        np.savez_compressed(
            cache_path,
            vertices=vertices,
            offsets=offsets,
            xlim=np.asarray(xlim),
            ylim=np.asarray(ylim),
            coordinate_offset=np.asarray([x_offset, y_offset]),
            source_dxf=np.asarray(str(dxf_path)),
            entity_count=np.asarray(entity_count),
            label_positions=np.asarray(label_positions, dtype=np.float64).reshape((-1, 2)),
            label_texts=np.asarray(label_texts, dtype="U256"),
            label_rotations=np.asarray(label_rotations, dtype=np.float64),
        )
    finally:
        logging.disable(logging.NOTSET)
    return cache_path


def prepare_dwg_segments(
    dwg_path: Path,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    x_offset: float,
    y_offset: float,
    core_console: Path | None = None,
    cache_dir: Path | None = None,
) -> Path:
    return prepare_cad_segments(
        dwg_path, xlim, ylim, x_offset, y_offset,
        core_console=core_console, cache_dir=cache_dir,
    )


def prepare_cad_segments(
    cad_path: Path,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    x_offset: float,
    y_offset: float,
    core_console: Path | None = None,
    cache_dir: Path | None = None,
) -> Path:
    """Prepare clipped vector segments from a DWG or DXF mine map."""
    cad_path = cad_path.resolve()
    suffix = cad_path.suffix.lower()
    if suffix == ".dwg":
        dxf_path = export_dwg_to_dxf(cad_path, core_console, cache_dir)
    elif suffix == ".dxf":
        if not cad_path.is_file():
            raise FileNotFoundError(f"DXF不存在: {cad_path}")
        dxf_path = cad_path
    else:
        raise ValueError(f"CAD底图仅支持DWG或DXF: {cad_path}")
    return extract_dxf_segments(
        dxf_path, xlim, ylim, x_offset, y_offset,
        cache_dir=cache_dir,
    )


def load_segment_cache(path: Path) -> list[np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        vertices = np.asarray(data["vertices"], dtype=float)
        offsets = np.asarray(data["offsets"], dtype=int)
    return [vertices[start:end] for start, end in zip(offsets[:-1], offsets[1:]) if end - start >= 2]


def load_label_cache(path: Path) -> list[tuple[float, float, str, float]]:
    """Load optional CAD text labels from a schema-v2 segment cache."""
    with np.load(path, allow_pickle=False) as data:
        if not {"label_positions", "label_texts", "label_rotations"}.issubset(data.files):
            return []
        positions = np.asarray(data["label_positions"], dtype=float)
        texts = np.asarray(data["label_texts"]).astype(str)
        rotations = np.asarray(data["label_rotations"], dtype=float)
    return [
        (float(position[0]), float(position[1]), str(text), float(rotation))
        for position, text, rotation in zip(positions, texts, rotations)
        if str(text).strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="读取DWG/DXF并提取工作面矢量底图")
    parser.add_argument("cad_file", type=Path)
    parser.add_argument("--x-min", type=float, required=True)
    parser.add_argument("--x-max", type=float, required=True)
    parser.add_argument("--y-min", type=float, required=True)
    parser.add_argument("--y-max", type=float, required=True)
    parser.add_argument("--x-offset", type=float, default=0.0)
    parser.add_argument("--y-offset", type=float, default=0.0)
    parser.add_argument("--accoreconsole", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    cache_path = prepare_cad_segments(
        args.cad_file,
        (args.x_min, args.x_max),
        (args.y_min, args.y_max),
        args.x_offset,
        args.y_offset,
        args.accoreconsole,
        args.cache_dir,
    )
    segments = load_segment_cache(cache_path)
    labels = load_label_cache(cache_path)
    print(f"CAD segment cache: {cache_path}")
    print(f"CAD segments: {len(segments)}")
    print(f"CAD labels: {len(labels)}")


if __name__ == "__main__":
    main()
