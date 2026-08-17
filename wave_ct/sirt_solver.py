"""Sparse simultaneous iterative reconstruction for WaveCT.

The historical ``CT_shi_SIRT_Automatic_tuning_global_optimum.py`` script
uses a row/column normalised SIRT update.  This module keeps that update but
exposes it as a small, testable solver for WaveCT's augmented linear system.

SIRT is an iterative numerical method, not a guarantee of a global optimum.
The caller is responsible for physical bounds, line search, validation and
the distinction between quantitative and presentation-only fields.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix, issparse


@dataclass(frozen=True)
class SIRTResult:
    """Result and diagnostics for one augmented-system SIRT solve."""

    x: np.ndarray
    flag: int
    iterations: int
    residual_norm: float
    relative_residual: float


def _row_column_scales(matrix: csr_matrix, floor: float) -> tuple[np.ndarray, np.ndarray]:
    """Return positive row and column sums used by the SIRT preconditioner."""

    row_scale = np.asarray(np.abs(matrix).sum(axis=1)).ravel().astype(np.float64)
    column_scale = np.asarray(np.abs(matrix).sum(axis=0)).ravel().astype(np.float64)
    row_scale = np.maximum(row_scale, floor)
    # A column with no support must not move.  Its scale is set to infinity so
    # that the reciprocal used below is exactly zero.
    column_scale[column_scale < floor] = np.inf
    return row_scale, column_scale


def solve_sirt(
    matrix,
    rhs: np.ndarray,
    *,
    iterations: int = 160,
    relaxation: float = 0.1,
    tolerance: float = 1.0e-8,
    x0: np.ndarray | None = None,
    scale_floor: float = 1.0e-12,
) -> SIRTResult:
    """Solve ``matrix @ x ~= rhs`` with a damped simultaneous SIRT update.

    The update is

    ``x[k+1] = x[k] + omega D_c^-1 A.T D_r^-1 (b - A x[k])``.

    ``matrix`` is normally WaveCT's sparse augmented data/regularisation
    system.  Row normalisation follows the reference SIRT implementation,
    while physical regularisation remains in the matrix and is still checked
    by WaveCT's outer validation and line-search logic.
    """

    if not issparse(matrix):
        matrix = csr_matrix(np.asarray(matrix, dtype=np.float64))
    matrix = matrix.tocsr().astype(np.float64)
    rhs = np.asarray(rhs, dtype=np.float64).ravel()
    if matrix.ndim != 2 or matrix.shape[0] != rhs.size:
        raise ValueError(
            f"SIRT shape mismatch: matrix={matrix.shape}, rhs={rhs.shape}"
        )
    if iterations < 1:
        raise ValueError("SIRT iterations must be positive")
    if not np.isfinite(relaxation) or not 0.0 < relaxation <= 1.0:
        raise ValueError("SIRT relaxation must be in (0, 1]")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("SIRT tolerance must be positive and finite")
    if not np.isfinite(scale_floor) or scale_floor <= 0.0:
        raise ValueError("SIRT scale_floor must be positive and finite")

    if x0 is None:
        x = np.zeros(matrix.shape[1], dtype=np.float64)
    else:
        x = np.asarray(x0, dtype=np.float64).ravel().copy()
        if x.size != matrix.shape[1]:
            raise ValueError(
                f"SIRT initial vector has {x.size} values; expected {matrix.shape[1]}"
            )
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(rhs)):
        raise ValueError("SIRT inputs must be finite")

    row_scale, column_scale = _row_column_scales(matrix, scale_floor)
    reciprocal_columns = np.divide(
        1.0,
        column_scale,
        out=np.zeros_like(column_scale),
        where=np.isfinite(column_scale),
    )
    rhs_norm = max(float(np.linalg.norm(rhs)), scale_floor)
    residual_norm = float(np.linalg.norm(rhs - matrix @ x))
    relative_residual = residual_norm / rhs_norm
    if relative_residual <= tolerance:
        return SIRTResult(x, 0, 0, residual_norm, relative_residual)

    flag = 7  # SciPy LSQR's "iteration limit reached" convention.
    completed = 0
    for completed in range(1, int(iterations) + 1):
        residual = rhs - matrix @ x
        correction = np.asarray(
            matrix.T @ (residual / row_scale), dtype=np.float64
        ).ravel()
        x += relaxation * reciprocal_columns * correction
        if not np.all(np.isfinite(x)):
            raise FloatingPointError("SIRT produced a non-finite model update")
        residual_norm = float(np.linalg.norm(rhs - matrix @ x))
        relative_residual = residual_norm / rhs_norm
        if relative_residual <= tolerance:
            flag = 1
            break

    return SIRTResult(
        x=x,
        flag=flag,
        iterations=completed,
        residual_norm=residual_norm,
        relative_residual=relative_residual,
    )

