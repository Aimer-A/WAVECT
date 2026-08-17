"""Deep neural reparameterization for sparse traveltime tomography.

The network is untrained: it maps normalized cell coordinates to a bounded
slowness field and is optimized only against the current project's traveltime
observations.  This is an implicit spatial regularizer, not a learned prior
from external velocity models.

The implementation deliberately accepts an already-built WaveCT path matrix.
It can therefore be compared with the production LSQR solver using identical
quality control, ray geometry, event splits, and velocity bounds.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Optional, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.sparse import csr_matrix, issparse


@dataclass(frozen=True)
class DeepReparamConfig:
    """Optimization settings frozen for one reproducible candidate."""

    width: int = 24
    depth: int = 3
    learning_rate: float = 4.0e-3
    max_epochs: int = 1600
    evaluation_interval: int = 10
    early_stop_patience: int = 300
    huber_beta_ms: float = 8.0
    output_l2: float = 2.0e-4
    event_static_shrinkage: float = 5.0
    receiver_static_shrinkage: float = 20.0
    receiver_static_max_ms: float = 12.0
    static_profile_iterations: int = 3
    random_seed: int = 20260724
    device: str = "auto"
    fixed_epochs: int = 0
    fourier_bands: int = 0
    neighbor_smoothness: float = 0.0
    total_variation: float = 0.0
    total_variation_epsilon: float = 1.0e-3
    final_learning_rate_fraction: float = 1.0
    differential_loss_fraction: float = 0.0


@dataclass(frozen=True)
class DeepReparamResult:
    """Selected model and auditable optimization diagnostics."""

    slowness: np.ndarray
    best_epoch: int
    train_rms_ms: float
    validation_rms_ms: float
    validation_metric_rays: int
    parameter_count: int
    device: str
    runtime_seconds: float
    history: np.ndarray
    receiver_static_ids: np.ndarray
    receiver_static_corrections_s: np.ndarray


def normalized_cell_coordinates(
    xc: Sequence[float],
    yc: Sequence[float],
    zc: Sequence[float],
) -> np.ndarray:
    """Return cell coordinates in WaveCT's Fortran flattening order."""

    x = np.asarray(xc, dtype=np.float64)
    y = np.asarray(yc, dtype=np.float64)
    z = np.asarray(zc, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or z.ndim != 1:
        raise ValueError("cell-center axes must be one-dimensional")
    if x.size == 0 or y.size == 0 or z.size == 0:
        raise ValueError("cell-center axes must not be empty")
    gx, gy, gz = np.meshgrid(x, y, z, indexing="ij")
    coordinates = np.column_stack(
        [
            gx.ravel(order="F"),
            gy.ravel(order="F"),
            gz.ravel(order="F"),
        ]
    )
    low = np.min(coordinates, axis=0)
    span = np.max(coordinates, axis=0) - low
    normalized = np.zeros_like(coordinates)
    varying = span > 0.0
    normalized[:, varying] = (
        2.0 * (coordinates[:, varying] - low[varying]) / span[varying] - 1.0
    )
    return normalized


def multiscale_coordinate_features(
    coordinates: np.ndarray,
    fourier_bands: int = 0,
) -> np.ndarray:
    """Append fixed dyadic sine/cosine features to normalized coordinates.

    ``fourier_bands=0`` is exactly the historical three-coordinate input.
    Positive values append ``sin(pi * 2**k * x)`` and the matching cosine for
    each coordinate and band.  The encoding is fixed and has no learned
    parameters, so it can be compared under the same outer validation splits.
    """

    points = np.asarray(coordinates, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("coordinates must have shape (model_cells, 3)")
    if isinstance(fourier_bands, bool) or int(fourier_bands) != fourier_bands:
        raise ValueError("Fourier band count must be an integer")
    band_count = int(fourier_bands)
    if band_count < 0:
        raise ValueError("Fourier band count must be non-negative")
    features = [points]
    for band in range(band_count):
        phase = np.pi * (2.0**band) * points
        features.extend([np.sin(phase), np.cos(phase)])
    return np.column_stack(features)


def structured_neighbor_pairs(
    coordinates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return undirected positive-axis neighbor indices for a full 3-D grid."""

    points = np.asarray(coordinates, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("coordinates must have shape (model_cells, 3)")
    axes = [np.unique(points[:, axis]) for axis in range(3)]
    shape = tuple(axis.size for axis in axes)
    if int(np.prod(shape)) != points.shape[0]:
        raise ValueError("coordinates do not form a complete structured grid")
    gx, gy, gz = np.meshgrid(*axes, indexing="ij")
    expected = np.column_stack(
        [
            gx.ravel(order="F"),
            gy.ravel(order="F"),
            gz.ravel(order="F"),
        ]
    )
    if not np.allclose(points, expected, rtol=0.0, atol=1.0e-10):
        raise ValueError(
            "coordinates must follow WaveCT's Fortran structured-grid order"
        )
    index_volume = np.arange(points.shape[0], dtype=np.int64).reshape(
        shape, order="F"
    )
    left_parts = []
    right_parts = []
    for axis in range(3):
        if shape[axis] <= 1:
            continue
        lower = [slice(None)] * 3
        upper = [slice(None)] * 3
        lower[axis] = slice(0, -1)
        upper[axis] = slice(1, None)
        left_parts.append(index_volume[tuple(lower)].ravel())
        right_parts.append(index_volume[tuple(upper)].ravel())
    if not left_parts:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(left_parts), np.concatenate(right_parts)


def cosine_learning_rate(
    initial_learning_rate: float,
    final_fraction: float,
    epoch: int,
    total_epochs: int,
) -> float:
    """Return a deterministic cosine-decayed learning rate for one epoch."""

    if not np.isfinite(initial_learning_rate) or initial_learning_rate <= 0.0:
        raise ValueError("initial learning rate must be positive and finite")
    if (
        not np.isfinite(final_fraction)
        or final_fraction < 0.0
        or final_fraction > 1.0
    ):
        raise ValueError("final learning-rate fraction must be in [0, 1]")
    if epoch < 1 or total_epochs < 1 or epoch > total_epochs:
        raise ValueError("epoch must lie within the positive training range")
    if total_epochs == 1:
        return float(initial_learning_rate)
    progress = (epoch - 1) / (total_epochs - 1)
    multiplier = final_fraction + (1.0 - final_fraction) * 0.5 * (
        1.0 + np.cos(np.pi * progress)
    )
    return float(initial_learning_rate * multiplier)


def event_pair_indices(
    source_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return all within-event row pairs and event-size normalization weights."""

    groups = np.asarray(source_ids)
    if groups.ndim != 1:
        raise ValueError("source IDs must be a one-dimensional vector")
    left_parts = []
    right_parts = []
    weight_parts = []
    for source_id in np.unique(groups):
        rows = np.flatnonzero(groups == source_id)
        if rows.size < 2:
            continue
        left, right = np.triu_indices(rows.size, k=1)
        left_parts.append(rows[left])
        right_parts.append(rows[right])
        weight_parts.append(
            np.full(left.size, 1.0 / rows.size, dtype=np.float64)
        )
    if not left_parts:
        empty_indices = np.empty(0, dtype=np.int64)
        return empty_indices, empty_indices.copy(), np.empty(0)
    return (
        np.concatenate(left_parts).astype(np.int64, copy=False),
        np.concatenate(right_parts).astype(np.int64, copy=False),
        np.concatenate(weight_parts),
    )


def training_coverage_gate(
    path_matrix: csr_matrix,
    nx: int,
    ny: int,
    nz: int,
    reliable_coverage: float = 5.0,
    smoothing_sigma: tuple[float, float, float] = (1.0, 1.0, 0.45),
    coverage_exponent: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a smooth gate using training-ray geometry only.

    The gate is zero in distant unsupported cells and approaches one in
    reliably crossed cells.  It prevents the neural parameterization from
    inventing unconstrained anomalies while avoiding a hard coverage edge.
    """

    if not issparse(path_matrix):
        path_matrix = csr_matrix(path_matrix)
    expected = int(nx) * int(ny) * int(nz)
    if path_matrix.shape[1] != expected:
        raise ValueError("path matrix column count does not match the grid")
    if not np.isfinite(reliable_coverage) or reliable_coverage <= 0.0:
        raise ValueError("reliable coverage must be positive and finite")
    if len(smoothing_sigma) != 3 or np.any(np.asarray(smoothing_sigma) < 0.0):
        raise ValueError("coverage smoothing sigma must contain three non-negative values")
    if not np.isfinite(coverage_exponent) or coverage_exponent <= 0.0:
        raise ValueError("coverage exponent must be positive and finite")

    density = np.asarray((path_matrix > 0).sum(axis=0)).ravel().astype(np.float64)
    volume = np.power(
        np.clip(density / reliable_coverage, 0.0, 1.0),
        coverage_exponent,
    ).reshape(
        (nx, ny, nz), order="F"
    )
    gate = gaussian_filter(volume, sigma=smoothing_sigma, mode="nearest")
    gate = np.clip(gate, 0.0, 1.0).ravel(order="F")
    return gate, density


def event_centered_rms(
    observed_s: np.ndarray,
    predicted_s: np.ndarray,
    source_ids: np.ndarray,
) -> tuple[float, int]:
    """Return median-centered RMS so held-out event origin times do not leak."""

    observed = np.asarray(observed_s, dtype=np.float64)
    predicted = np.asarray(predicted_s, dtype=np.float64)
    groups = np.asarray(source_ids)
    if observed.shape != predicted.shape or observed.shape != groups.shape:
        raise ValueError("observations, predictions, and source IDs must align")
    parts = []
    for source_id in np.unique(groups):
        residual = observed[groups == source_id] - predicted[groups == source_id]
        if residual.size >= 2:
            parts.append(residual - np.median(residual))
    if parts:
        centered = np.concatenate(parts)
    else:
        centered = observed - predicted
    if centered.size == 0:
        return float("nan"), 0
    return float(np.sqrt(np.mean(centered**2))), int(centered.size)


def profile_event_time_corrections(
    observed_s: np.ndarray,
    predicted_s: np.ndarray,
    source_ids: np.ndarray,
    shrinkage: float = 5.0,
    maximum_correction_s: float = 0.12,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Profile a global shift and shrunken per-event origin-time offsets."""

    observed = np.asarray(observed_s, dtype=np.float64)
    predicted = np.asarray(predicted_s, dtype=np.float64)
    groups = np.asarray(source_ids)
    if observed.shape != predicted.shape or observed.shape != groups.shape:
        raise ValueError("observations, predictions, and source IDs must align")
    if observed.ndim != 1 or observed.size == 0:
        raise ValueError("time-correction inputs must be non-empty vectors")
    if not np.isfinite(shrinkage) or shrinkage < 0.0:
        raise ValueError("shrinkage must be non-negative and finite")
    if not np.isfinite(maximum_correction_s) or maximum_correction_s <= 0.0:
        raise ValueError("maximum correction must be positive and finite")

    residual = observed - predicted
    global_shift = float(
        np.clip(np.mean(residual), -maximum_correction_s, maximum_correction_s)
    )
    centered = residual - global_shift
    unique_sources, inverse = np.unique(groups, return_inverse=True)
    sums = np.bincount(inverse, weights=centered)
    counts = np.bincount(inverse).astype(np.float64)
    deviations = sums / (counts + shrinkage)
    deviations = np.clip(
        deviations,
        -maximum_correction_s,
        maximum_correction_s,
    )
    return global_shift, unique_sources, deviations.astype(np.float64)


def profile_event_receiver_time_corrections(
    observed_s: np.ndarray,
    predicted_s: np.ndarray,
    source_ids: np.ndarray,
    receiver_ids: np.ndarray,
    event_shrinkage: float = 5.0,
    receiver_shrinkage: float = 20.0,
    maximum_event_correction_s: float = 0.12,
    maximum_receiver_correction_s: float = 0.012,
    iterations: int = 3,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Profile identifiable event and receiver statics by ridge backfitting."""

    observed = np.asarray(observed_s, dtype=np.float64)
    predicted = np.asarray(predicted_s, dtype=np.float64)
    sources = np.asarray(source_ids)
    receivers = np.asarray(receiver_ids)
    if not (
        observed.shape == predicted.shape == sources.shape == receivers.shape
    ):
        raise ValueError("two-way static-correction inputs must align")
    if observed.ndim != 1 or observed.size == 0:
        raise ValueError("two-way static-correction inputs must be non-empty")
    if (
        not np.isfinite(event_shrinkage)
        or event_shrinkage < 0.0
        or not np.isfinite(receiver_shrinkage)
        or receiver_shrinkage < 0.0
    ):
        raise ValueError("static-correction shrinkage must be non-negative")
    if (
        maximum_event_correction_s <= 0.0
        or maximum_receiver_correction_s <= 0.0
        or iterations < 1
    ):
        raise ValueError("static-correction limits and iterations are invalid")

    residual = observed - predicted
    global_shift = float(
        np.clip(
            np.mean(residual),
            -maximum_event_correction_s,
            maximum_event_correction_s,
        )
    )
    centered = residual - global_shift
    unique_sources, source_inverse = np.unique(sources, return_inverse=True)
    unique_receivers, receiver_inverse = np.unique(
        receivers, return_inverse=True
    )
    source_counts = np.bincount(source_inverse).astype(np.float64)
    receiver_counts = np.bincount(receiver_inverse).astype(np.float64)
    source_offsets = np.zeros(unique_sources.size, dtype=np.float64)
    receiver_offsets = np.zeros(unique_receivers.size, dtype=np.float64)
    for _ in range(int(iterations)):
        source_sums = np.bincount(
            source_inverse,
            weights=centered - receiver_offsets[receiver_inverse],
        )
        source_offsets = source_sums / (source_counts + event_shrinkage)
        source_offsets = np.clip(
            source_offsets,
            -maximum_event_correction_s,
            maximum_event_correction_s,
        )
        receiver_sums = np.bincount(
            receiver_inverse,
            weights=centered - source_offsets[source_inverse],
        )
        receiver_offsets = receiver_sums / (
            receiver_counts + receiver_shrinkage
        )
        # A weighted zero-mean receiver constraint removes the remaining
        # event/receiver gauge freedom before applying the engineering bound.
        receiver_offsets -= float(
            np.average(receiver_offsets, weights=receiver_counts)
        )
        receiver_offsets = np.clip(
            receiver_offsets,
            -maximum_receiver_correction_s,
            maximum_receiver_correction_s,
        )
    return (
        global_shift,
        unique_sources,
        source_offsets.astype(np.float64),
        unique_receivers,
        receiver_offsets.astype(np.float64),
    )


def _validate_config(config: DeepReparamConfig) -> None:
    if config.width < 2 or config.depth < 1:
        raise ValueError("network width and depth are too small")
    if not np.isfinite(config.learning_rate) or config.learning_rate <= 0.0:
        raise ValueError("learning rate must be positive and finite")
    if config.max_epochs < 1 or config.evaluation_interval < 1:
        raise ValueError("epoch and evaluation counts must be positive")
    if config.early_stop_patience < config.evaluation_interval:
        raise ValueError("early-stop patience must cover at least one evaluation interval")
    if not np.isfinite(config.huber_beta_ms) or config.huber_beta_ms <= 0.0:
        raise ValueError("Huber beta must be positive and finite")
    if not np.isfinite(config.output_l2) or config.output_l2 < 0.0:
        raise ValueError("output L2 weight must be non-negative and finite")
    if (
        not np.isfinite(config.event_static_shrinkage)
        or config.event_static_shrinkage < 0.0
    ):
        raise ValueError("event-static shrinkage must be non-negative and finite")
    if (
        not np.isfinite(config.receiver_static_shrinkage)
        or config.receiver_static_shrinkage < 0.0
        or not np.isfinite(config.receiver_static_max_ms)
        or config.receiver_static_max_ms <= 0.0
        or config.static_profile_iterations < 1
    ):
        raise ValueError("receiver-static profiling settings are invalid")
    if config.fixed_epochs < 0:
        raise ValueError("fixed epochs must be non-negative")
    if (
        isinstance(config.fourier_bands, bool)
        or int(config.fourier_bands) != config.fourier_bands
        or config.fourier_bands < 0
    ):
        raise ValueError("Fourier band count must be a non-negative integer")
    if (
        not np.isfinite(config.neighbor_smoothness)
        or config.neighbor_smoothness < 0.0
    ):
        raise ValueError(
            "neighbor smoothness weight must be non-negative and finite"
        )
    if (
        not np.isfinite(config.total_variation)
        or config.total_variation < 0.0
    ):
        raise ValueError(
            "total-variation weight must be non-negative and finite"
        )
    if (
        not np.isfinite(config.total_variation_epsilon)
        or config.total_variation_epsilon <= 0.0
    ):
        raise ValueError(
            "total-variation epsilon must be positive and finite"
        )
    if (
        not np.isfinite(config.final_learning_rate_fraction)
        or config.final_learning_rate_fraction < 0.0
        or config.final_learning_rate_fraction > 1.0
    ):
        raise ValueError(
            "final learning-rate fraction must be in the interval [0, 1]"
        )
    if (
        not np.isfinite(config.differential_loss_fraction)
        or config.differential_loss_fraction < 0.0
        or config.differential_loss_fraction > 1.0
    ):
        raise ValueError(
            "differential loss fraction must be in the interval [0, 1]"
        )


def _torch_sparse_matrix(matrix: csr_matrix, torch_module, device: str):
    coo = matrix.tocoo()
    indices = torch_module.as_tensor(
        np.vstack([coo.row, coo.col]),
        dtype=torch_module.long,
        device=device,
    )
    values = torch_module.as_tensor(
        coo.data,
        dtype=torch_module.float32,
        device=device,
    )
    return torch_module.sparse_coo_tensor(
        indices,
        values,
        size=coo.shape,
        dtype=torch_module.float32,
        device=device,
    ).coalesce()


def fit_deep_reparameterized_slowness(
    path_matrix: csr_matrix,
    observed_s: np.ndarray,
    source_ids: np.ndarray,
    coordinates: np.ndarray,
    coverage_gate: np.ndarray,
    background_velocity: float,
    minimum_velocity: float,
    maximum_velocity: float,
    config: Optional[DeepReparamConfig] = None,
    validation_path_matrix: Optional[csr_matrix] = None,
    validation_observed_s: Optional[np.ndarray] = None,
    validation_source_ids: Optional[np.ndarray] = None,
    receiver_ids: Optional[np.ndarray] = None,
    validation_receiver_ids: Optional[np.ndarray] = None,
) -> DeepReparamResult:
    """Fit an untrained coordinate MLP to one fixed tomography split."""

    settings = config or DeepReparamConfig()
    _validate_config(settings)
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "deep reparameterization requires PyTorch; the LSQR solver remains available"
        ) from exc

    if not issparse(path_matrix):
        path_matrix = csr_matrix(path_matrix)
    train_matrix = path_matrix.tocsr().astype(np.float64)
    observed = np.asarray(observed_s, dtype=np.float64)
    groups = np.asarray(source_ids)
    receiver_groups = (
        None if receiver_ids is None else np.asarray(receiver_ids)
    )
    points = np.asarray(coordinates, dtype=np.float64)
    gate = np.asarray(coverage_gate, dtype=np.float64)
    if observed.ndim != 1 or groups.shape != observed.shape:
        raise ValueError("training observations and source IDs must be aligned vectors")
    if receiver_groups is not None and receiver_groups.shape != observed.shape:
        raise ValueError("training receiver IDs must align with observations")
    if train_matrix.shape[0] != observed.size:
        raise ValueError("training path rows and observations do not align")
    if points.shape != (train_matrix.shape[1], 3):
        raise ValueError("coordinates must have shape (model_cells, 3)")
    if gate.shape != (train_matrix.shape[1],):
        raise ValueError("coverage gate must contain one value per model cell")
    if (
        not np.isfinite(points).all()
        or not np.isfinite(gate).all()
        or np.any(gate < 0.0)
        or np.any(gate > 1.0)
    ):
        raise ValueError("coordinates and coverage gate must be finite and valid")
    if not (0.0 < minimum_velocity < background_velocity < maximum_velocity):
        raise ValueError("velocity bounds must contain the background velocity")

    has_validation = validation_path_matrix is not None
    if has_validation:
        if validation_observed_s is None or validation_source_ids is None:
            raise ValueError("validation matrix requires observations and source IDs")
        validation_matrix = csr_matrix(validation_path_matrix).tocsr().astype(np.float64)
        validation_observed = np.asarray(validation_observed_s, dtype=np.float64)
        validation_groups = np.asarray(validation_source_ids)
        validation_receiver_groups = (
            None
            if validation_receiver_ids is None
            else np.asarray(validation_receiver_ids)
        )
        if (
            validation_matrix.shape[1] != train_matrix.shape[1]
            or validation_matrix.shape[0] != validation_observed.size
            or validation_groups.shape != validation_observed.shape
        ):
            raise ValueError("validation arrays do not align")
        if (
            receiver_groups is not None
            and (
                validation_receiver_groups is None
                or validation_receiver_groups.shape != validation_observed.shape
            )
        ):
            raise ValueError(
                "validation receiver IDs are required for receiver statics"
            )
    else:
        validation_matrix = None
        validation_observed = np.empty(0, dtype=np.float64)
        validation_groups = np.empty(0, dtype=groups.dtype)
        validation_receiver_groups = None

    requested_device = settings.device.lower()
    if requested_device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif requested_device in {"cpu", "cuda"}:
        if requested_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        device = requested_device
    else:
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'")

    torch.manual_seed(settings.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(settings.random_seed)

    coordinate_features = multiscale_coordinate_features(
        points, settings.fourier_bands
    )

    class CoordinateMLP(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers = []
            input_width = coordinate_features.shape[1]
            for _ in range(settings.depth):
                layers.extend(
                    [
                        torch.nn.Linear(input_width, settings.width),
                        torch.nn.Tanh(),
                    ]
                )
                input_width = settings.width
            layers.append(torch.nn.Linear(input_width, 1))
            self.network = torch.nn.Sequential(*layers)
            torch.nn.init.normal_(self.network[-1].weight, std=1.0e-3)
            torch.nn.init.zeros_(self.network[-1].bias)

        def forward(self, value):
            return self.network(value).squeeze(-1)

    model = CoordinateMLP().to(device)
    parameter_count = int(sum(item.numel() for item in model.parameters()))
    coordinate_tensor = torch.as_tensor(
        coordinate_features, dtype=torch.float32, device=device
    )
    if settings.neighbor_smoothness > 0.0 or settings.total_variation > 0.0:
        neighbor_left, neighbor_right = structured_neighbor_pairs(points)
        neighbor_left_tensor = torch.as_tensor(
            neighbor_left, dtype=torch.long, device=device
        )
        neighbor_right_tensor = torch.as_tensor(
            neighbor_right, dtype=torch.long, device=device
        )
    else:
        neighbor_left_tensor = None
        neighbor_right_tensor = None
    gate_tensor = torch.as_tensor(gate, dtype=torch.float32, device=device)
    # Multiplying by 1000 lets the optimizer work directly in milliseconds.
    train_tensor = _torch_sparse_matrix(train_matrix * 1000.0, torch, device)
    observed_tensor = torch.as_tensor(
        observed * 1000.0, dtype=torch.float32, device=device
    )
    _, event_inverse = np.unique(groups, return_inverse=True)
    event_inverse_tensor = torch.as_tensor(
        event_inverse, dtype=torch.long, device=device
    )
    event_counts = torch.bincount(event_inverse_tensor).to(torch.float32)
    eligible_rows = event_counts[event_inverse_tensor] >= 2.0
    if int(torch.count_nonzero(eligible_rows).item()) == 0:
        raise ValueError("deep reparameterization needs events with at least two arrivals")
    if receiver_groups is not None:
        unique_receivers, receiver_inverse = np.unique(
            receiver_groups, return_inverse=True
        )
        receiver_inverse_tensor = torch.as_tensor(
            receiver_inverse, dtype=torch.long, device=device
        )
        receiver_counts = torch.bincount(receiver_inverse_tensor).to(
            torch.float32
        )
        receiver_lookup = {
            value: index for index, value in enumerate(unique_receivers.tolist())
        }
        if validation_receiver_groups is not None:
            validation_receiver_index = np.asarray(
                [
                    receiver_lookup.get(value, -1)
                    for value in validation_receiver_groups.tolist()
                ],
                dtype=np.int64,
            )
        else:
            validation_receiver_index = np.empty(0, dtype=np.int64)
    else:
        unique_receivers = np.empty(0, dtype=np.int64)
        receiver_inverse_tensor = None
        receiver_counts = None
        validation_receiver_index = np.empty(0, dtype=np.int64)
    if settings.differential_loss_fraction > 0.0:
        pair_left, pair_right, pair_weight = event_pair_indices(groups)
        if pair_left.size == 0:
            raise ValueError(
                "differential loss needs events with at least two arrivals"
            )
        pair_left_tensor = torch.as_tensor(
            pair_left, dtype=torch.long, device=device
        )
        pair_right_tensor = torch.as_tensor(
            pair_right, dtype=torch.long, device=device
        )
        pair_weight_tensor = torch.as_tensor(
            pair_weight, dtype=torch.float32, device=device
        )
    else:
        pair_left_tensor = None
        pair_right_tensor = None
        pair_weight_tensor = None

    minimum_slowness = 1.0 / maximum_velocity
    maximum_slowness = 1.0 / minimum_velocity
    background_slowness = 1.0 / background_velocity
    background_fraction = (
        (background_slowness - minimum_slowness)
        / (maximum_slowness - minimum_slowness)
    )
    background_logit = float(
        np.log(background_fraction / (1.0 - background_fraction))
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=settings.learning_rate
    )
    epochs_to_run = settings.fixed_epochs or settings.max_epochs
    history_rows = []
    best_score_ms = float("inf")
    best_epoch = 0
    best_slowness = np.full(train_matrix.shape[1], background_slowness)
    best_train_rms_ms = float("inf")
    best_validation_rms_ms = float("nan")
    best_validation_rays = 0
    best_receiver_offsets_ms = np.zeros(
        unique_receivers.size, dtype=np.float64
    )
    start_time = perf_counter()

    for epoch in range(1, epochs_to_run + 1):
        learning_rate = cosine_learning_rate(
            settings.learning_rate,
            settings.final_learning_rate_fraction,
            epoch,
            epochs_to_run,
        )
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = learning_rate
        optimizer.zero_grad()
        raw_output = model(coordinate_tensor)
        slowness_tensor = minimum_slowness + (
            maximum_slowness - minimum_slowness
        ) * torch.sigmoid(background_logit + gate_tensor * raw_output)
        prediction_ms = torch.sparse.mm(
            train_tensor, slowness_tensor[:, None]
        ).squeeze(1)
        residual_ms = observed_tensor - prediction_ms
        # A free global shift is removed. Per-event offsets are profiled with
        # pseudocount shrinkage so timing errors do not become velocity, while
        # absolute timing information is not discarded completely.
        residual_ms = residual_ms - residual_ms.mean()
        event_offsets = torch.zeros(
            event_counts.numel(), dtype=torch.float32, device=device
        )
        receiver_offsets = (
            torch.zeros(
                receiver_counts.numel(), dtype=torch.float32, device=device
            )
            if receiver_counts is not None
            else None
        )
        for _ in range(settings.static_profile_iterations):
            event_target = residual_ms
            if receiver_offsets is not None:
                event_target = (
                    event_target
                    - receiver_offsets[receiver_inverse_tensor]
                )
            event_sums = torch.zeros(
                event_counts.numel(), dtype=torch.float32, device=device
            )
            event_sums.scatter_add_(
                0, event_inverse_tensor, event_target
            )
            event_offsets = event_sums / (
                event_counts + settings.event_static_shrinkage
            )
            if receiver_offsets is not None:
                receiver_target = (
                    residual_ms - event_offsets[event_inverse_tensor]
                )
                receiver_sums = torch.zeros(
                    receiver_counts.numel(),
                    dtype=torch.float32,
                    device=device,
                )
                receiver_sums.scatter_add_(
                    0, receiver_inverse_tensor, receiver_target
                )
                receiver_offsets = receiver_sums / (
                    receiver_counts + settings.receiver_static_shrinkage
                )
                receiver_offsets = receiver_offsets - torch.sum(
                    receiver_offsets * receiver_counts
                ) / torch.sum(receiver_counts)
                limit = settings.receiver_static_max_ms
                receiver_offsets = limit * torch.tanh(
                    receiver_offsets / limit
                )
        profiled_residual = residual_ms - event_offsets[event_inverse_tensor]
        if receiver_offsets is not None:
            profiled_residual = (
                profiled_residual
                - receiver_offsets[receiver_inverse_tensor]
            )
        selected_residual = profiled_residual[eligible_rows]
        profiled_loss = torch.nn.functional.smooth_l1_loss(
            selected_residual,
            torch.zeros_like(selected_residual),
            beta=settings.huber_beta_ms,
        )
        if pair_left_tensor is None:
            data_loss = profiled_loss
        else:
            differential_residual = (
                residual_ms[pair_left_tensor]
                - residual_ms[pair_right_tensor]
            ) / np.sqrt(2.0)
            differential_terms = torch.nn.functional.smooth_l1_loss(
                differential_residual,
                torch.zeros_like(differential_residual),
                beta=settings.huber_beta_ms,
                reduction="none",
            )
            differential_loss = torch.sum(
                pair_weight_tensor * differential_terms
            ) / torch.sum(pair_weight_tensor)
            fraction = settings.differential_loss_fraction
            data_loss = (
                (1.0 - fraction) * profiled_loss
                + fraction * differential_loss
            )
        loss = data_loss + settings.output_l2 * torch.mean(raw_output**2)
        if neighbor_left_tensor is not None:
            neighbor_difference = (
                raw_output[neighbor_left_tensor]
                - raw_output[neighbor_right_tensor]
            )
            if settings.neighbor_smoothness > 0.0:
                loss = loss + settings.neighbor_smoothness * torch.mean(
                    neighbor_difference**2
                )
            if settings.total_variation > 0.0:
                # Apply edge preservation to the interpretable relative
                # velocity anomaly, rather than to an arbitrary MLP output.
                relative_velocity = (
                    1.0 / (slowness_tensor * background_velocity) - 1.0
                )
                anomaly_difference = (
                    relative_velocity[neighbor_left_tensor]
                    - relative_velocity[neighbor_right_tensor]
                )
                epsilon = settings.total_variation_epsilon
                tv_terms = torch.sqrt(
                    anomaly_difference**2 + epsilon**2
                ) - epsilon
                loss = loss + settings.total_variation * torch.mean(tv_terms)
        loss.backward()
        optimizer.step()

        should_evaluate = (
            epoch == 1
            or epoch % settings.evaluation_interval == 0
            or epoch == epochs_to_run
        )
        if not should_evaluate:
            continue
        slowness_numpy = slowness_tensor.detach().cpu().numpy().astype(np.float64)
        receiver_offsets_s = (
            receiver_offsets.detach().cpu().numpy().astype(np.float64)
            / 1000.0
            if receiver_offsets is not None
            else np.empty(0, dtype=np.float64)
        )
        train_prediction = np.asarray(
            train_matrix @ slowness_numpy
        ).ravel()
        if receiver_offsets_s.size:
            train_prediction = (
                train_prediction + receiver_offsets_s[receiver_inverse]
            )
        train_rms_s, _ = event_centered_rms(
            observed,
            train_prediction,
            groups,
        )
        train_rms_ms = 1000.0 * train_rms_s
        if has_validation:
            validation_prediction = np.asarray(
                validation_matrix @ slowness_numpy
            ).ravel()
            if receiver_offsets_s.size:
                known_receiver = validation_receiver_index >= 0
                validation_prediction[known_receiver] += receiver_offsets_s[
                    validation_receiver_index[known_receiver]
                ]
            validation_rms_s, validation_rays = event_centered_rms(
                validation_observed,
                validation_prediction,
                validation_groups,
            )
            validation_rms_ms = 1000.0 * validation_rms_s
            score_ms = validation_rms_ms
        else:
            validation_rms_ms = float("nan")
            validation_rays = 0
            score_ms = train_rms_ms
        history_rows.append(
            [
                float(epoch),
                float(loss.detach().cpu().item()),
                train_rms_ms,
                validation_rms_ms,
            ]
        )

        # Fixed-epoch refits deliberately return the last epoch selected by
        # outer CV; they do not silently tune on full-data training error.
        improved = score_ms < best_score_ms - 1.0e-4
        if settings.fixed_epochs:
            improved = epoch == epochs_to_run
        if improved:
            best_score_ms = score_ms
            best_epoch = epoch
            best_slowness = slowness_numpy.copy()
            best_train_rms_ms = train_rms_ms
            best_validation_rms_ms = validation_rms_ms
            best_validation_rays = validation_rays
            best_receiver_offsets_ms = receiver_offsets_s * 1000.0
        elif (
            has_validation
            and not settings.fixed_epochs
            and epoch - best_epoch >= settings.early_stop_patience
        ):
            break

    runtime_seconds = perf_counter() - start_time
    return DeepReparamResult(
        slowness=best_slowness,
        best_epoch=best_epoch,
        train_rms_ms=best_train_rms_ms,
        validation_rms_ms=best_validation_rms_ms,
        validation_metric_rays=best_validation_rays,
        parameter_count=parameter_count,
        device=device,
        runtime_seconds=runtime_seconds,
        history=np.asarray(history_rows, dtype=np.float64),
        receiver_static_ids=np.asarray(unique_receivers),
        receiver_static_corrections_s=(
            best_receiver_offsets_ms / 1000.0
        ),
    )
