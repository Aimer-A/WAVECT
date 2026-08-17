"""Pure helpers for ordering and navigating WaveCT result images.

The desktop UI deliberately delegates gallery state calculations to this
module.  Keeping these functions independent from Tk makes refresh, keyboard
navigation and zoom behaviour deterministic and easy to test.
"""

from __future__ import annotations

import math
import os
import re
from enum import IntEnum
from pathlib import Path
from typing import Iterable, Sequence


MIN_ZOOM = 0.1
MAX_ZOOM = 8.0


class GalleryGroup(IntEnum):
    """Display order of the main result-product families."""

    OVERVIEW = 0
    SOURCE = 1
    RAY = 2
    VELOCITY = 3
    ANOMALY = 4
    COVERAGE = 5
    VALIDATION = 6
    OTHER = 7


_ELEVATION_PATTERNS = (
    # Production names normally use ``..._z-780.000.png``.
    re.compile(r"(?:^|[_\s])z\s*=?\s*([-+]?\d+(?:\.\d+)?)", re.IGNORECASE),
    # A hyphen immediately after the label is the elevation sign, not a field
    # separator (``标高 -810`` must remain negative).
    re.compile(r"(?:标高|高程|层位)\s*[:=_]?\s*([-+]?\d+(?:\.\d+)?)"),
    # Vendor figures can be named simply ``波速-780.png``.
    re.compile(r"([-+]\d+(?:\.\d+)?)\s*$"),
)


def extract_elevation(path: str | Path) -> float | None:
    """Extract a slice elevation from a result filename when one is present."""

    stem = Path(path).stem
    for pattern in _ELEVATION_PATTERNS:
        match = pattern.search(stem)
        if match:
            try:
                value = float(match.group(1))
            except ValueError:
                continue
            if math.isfinite(value):
                return value
    return None


def semantic_group(path: str | Path) -> GalleryGroup:
    """Classify a result image without relying on numeric filename prefixes."""

    name = Path(path).stem.casefold()

    # More specific families must be checked first.  For example,
    # ``ray_coverage_z-780`` is a coverage slice, while the plan and 3-D ray
    # summaries belong near the start of the gallery.
    if any(token in name for token in ("overview", "总览", "概览", "三层对比", "多层对比")):
        return GalleryGroup.OVERVIEW
    if any(token in name for token in ("source_distribution", "震源", "台站")):
        return GalleryGroup.SOURCE
    if any(
        token in name
        for token in (
            "ray_coverage_plan",
            "ray_coverage_3d",
            "ray_path",
            "射线路径",
            "三维射线",
        )
    ):
        return GalleryGroup.RAY
    if any(
        token in name
        for token in (
            "validation",
            "checkerboard",
            "convergence",
            "uncertainty",
            "rms",
            "验证",
            "棋盘格",
            "收敛",
            "不确定",
            "稳定性",
        )
    ):
        return GalleryGroup.VALIDATION
    if any(token in name for token in ("anomaly", "异常", "an图")):
        return GalleryGroup.ANOMALY
    if any(token in name for token in ("coverage", "reliability", "density", "覆盖", "射线密度")):
        return GalleryGroup.COVERAGE
    if any(token in name for token in ("velocity", "波速", "速度")):
        return GalleryGroup.VELOCITY
    return GalleryGroup.OTHER


def sort_gallery_paths(paths: Iterable[str | Path]) -> list[Path]:
    """Return paths in engineering-report order.

    Slice families are ordered from the highest elevation to the lowest (for
    example ``-750, -780, -810``).  Python's stable sort plus the original
    ordinal preserves input order for equal elevations and unknown products.
    """

    indexed = [(index, Path(path)) for index, path in enumerate(paths)]

    def key(item: tuple[int, Path]) -> tuple[int, int, float, int]:
        original_index, path = item
        group = semantic_group(path)
        elevation = extract_elevation(path)
        if group in {
            GalleryGroup.VELOCITY,
            GalleryGroup.ANOMALY,
            GalleryGroup.COVERAGE,
        } and elevation is not None:
            # Descending elevation: -750 m is above and precedes -810 m.
            return int(group), 0, -elevation, original_index
        return int(group), 1, 0.0, original_index

    return [path for _, path in sorted(indexed, key=key)]


def clamp_zoom(value: float, *, default: float = 1.0) -> float:
    """Clamp a zoom multiplier to the supported 10%--800% interval."""

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = float(default)
    if not math.isfinite(numeric):
        numeric = float(default)
    return min(MAX_ZOOM, max(MIN_ZOOM, numeric))


def fit_zoom(
    image_size: tuple[float, float],
    viewport_size: tuple[float, float],
    *,
    padding: float = 0.0,
) -> float:
    """Calculate the clamped zoom needed to fit an image in a viewport."""

    image_width, image_height = map(float, image_size)
    viewport_width, viewport_height = map(float, viewport_size)
    padding = float(padding)
    if image_width <= 0.0 or image_height <= 0.0:
        raise ValueError("image dimensions must be positive")
    if viewport_width <= 0.0 or viewport_height <= 0.0:
        raise ValueError("viewport dimensions must be positive")
    if padding < 0.0:
        raise ValueError("padding must be non-negative")
    available_width = viewport_width - 2.0 * padding
    available_height = viewport_height - 2.0 * padding
    if available_width <= 0.0 or available_height <= 0.0:
        raise ValueError("padding leaves no usable viewport area")
    return clamp_zoom(min(available_width / image_width, available_height / image_height))


def zoom_by(current: float, factor: float) -> float:
    """Apply a multiplicative zoom step and enforce gallery limits."""

    factor = float(factor)
    if not math.isfinite(factor) or factor <= 0.0:
        raise ValueError("zoom factor must be finite and positive")
    return clamp_zoom(clamp_zoom(current) * factor)


def adjacent_index(current: int, delta: int, item_count: int) -> int | None:
    """Move by ``delta`` with clamped ends; return ``None`` for an empty list."""

    if item_count <= 0:
        return None
    current = min(item_count - 1, max(0, int(current)))
    return min(item_count - 1, max(0, current + int(delta)))


def can_move(current: int, delta: int, item_count: int) -> bool:
    """Return whether a previous/next command would change the selection."""

    target = adjacent_index(current, delta, item_count)
    return target is not None and target != min(item_count - 1, max(0, int(current)))


def _path_identity(path: str | Path) -> str:
    # Avoid ``resolve()``: gallery entries can disappear during a refresh and
    # selection retention should not require filesystem access.
    return os.path.normcase(os.path.normpath(os.fspath(path)))


def retained_selection_index(
    paths: Sequence[str | Path],
    selected_path: str | Path | None,
    *,
    previous_index: int = 0,
) -> int | None:
    """Retain the selected file after refresh, falling back to its old index."""

    if not paths:
        return None
    if selected_path is not None:
        selected_identity = _path_identity(selected_path)
        for index, path in enumerate(paths):
            if _path_identity(path) == selected_identity:
                return index
    return min(len(paths) - 1, max(0, int(previous_index)))
