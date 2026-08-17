"""Read geometry metadata from a legacy text CMAT ``.3DM`` model.

Only the grid geometry and the initial constant velocity are exposed.  The
historical inverted velocity values are deliberately not returned by this
module so they cannot accidentally become a prior for a new WaveCT run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CMATModelGeometry:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float
    nx_nodes: int
    ny_nodes: int
    nz_nodes: int
    initial_velocity_mps: float

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def read_cmat_model_geometry(path: Path | str) -> CMATModelGeometry:
    """Read the compact geometry block immediately following the ``@`` line."""

    path = Path(path)
    lines = path.read_text(encoding="gbk", errors="strict").splitlines()
    try:
        marker = next(index for index, line in enumerate(lines) if line.strip() == "@")
    except StopIteration as exc:
        raise ValueError(f"CMAT 3DM geometry marker '@' not found: {path}") from exc
    if marker + 5 >= len(lines):
        raise ValueError(f"truncated CMAT 3DM geometry block: {path}")
    try:
        x_min, x_max = (float(value) for value in lines[marker + 1].split())
        y_min, y_max = (float(value) for value in lines[marker + 2].split())
        z_min, z_max = (float(value) for value in lines[marker + 3].split())
        nx, ny, nz = (int(value) for value in lines[marker + 4].split())
        first_node = tuple(float(value) for value in lines[marker + 5].split())
    except ValueError as exc:
        raise ValueError(f"invalid CMAT 3DM geometry block: {path}") from exc
    values = np.asarray(
        [x_min, x_max, y_min, y_max, z_min, z_max, *first_node],
        dtype=np.float64,
    )
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite CMAT 3DM geometry value: {path}")
    if not (x_min < x_max and y_min < y_max and z_min < z_max):
        raise ValueError(f"invalid CMAT 3DM model bounds: {path}")
    if min(nx, ny, nz) < 2:
        raise ValueError(f"invalid CMAT 3DM grid nodes: {(nx, ny, nz)}")
    if len(first_node) < 4 or first_node[3] <= 0:
        raise ValueError(f"invalid CMAT 3DM initial velocity: {path}")
    return CMATModelGeometry(
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        z_min=z_min,
        z_max=z_max,
        nx_nodes=nx,
        ny_nodes=ny,
        nz_nodes=nz,
        initial_velocity_mps=float(first_node[3]) * 1000.0,
    )
