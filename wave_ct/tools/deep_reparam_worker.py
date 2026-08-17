"""Isolated deep-reparameterization fit worker.

Keeping this worker free of Matplotlib and the legacy inversion module avoids
mixed OpenMP runtimes in the Windows Anaconda build.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from wave_ct.deep_reparam import (
    DeepReparamConfig,
    fit_deep_reparameterized_slowness,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem-npz", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--fixed-epochs", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=1600)
    parser.add_argument("--random-seed", type=int, required=True)
    parser.add_argument("--network-width", type=int, default=24)
    parser.add_argument("--network-depth", type=int, default=3)
    parser.add_argument("--fourier-bands", type=int, default=0)
    parser.add_argument("--neighbor-smoothness", type=float, default=0.0)
    parser.add_argument("--total-variation", type=float, default=0.0)
    parser.add_argument("--total-variation-epsilon", type=float, default=1.0e-3)
    parser.add_argument(
        "--final-learning-rate-fraction", type=float, default=1.0
    )
    parser.add_argument(
        "--differential-loss-fraction", type=float, default=0.0
    )
    parser.add_argument("--learning-rate", type=float, default=0.004)
    parser.add_argument("--huber-beta-ms", type=float, default=8.0)
    parser.add_argument("--event-static-shrinkage", type=float, default=5.0)
    parser.add_argument("--receiver-static-shrinkage", type=float, default=20.0)
    parser.add_argument("--receiver-static-max-ms", type=float, default=12.0)
    parser.add_argument("--static-profile-iterations", type=int, default=3)
    parser.add_argument("--output-l2", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    with np.load(args.problem_npz, allow_pickle=False) as payload:
        path_matrix = csr_matrix(
            (
                np.asarray(payload["matrix_data"], dtype=np.float64),
                np.asarray(payload["matrix_indices"], dtype=np.int32),
                np.asarray(payload["matrix_indptr"], dtype=np.int32),
            ),
            shape=tuple(int(value) for value in payload["matrix_shape"]),
        )
        observed_s = np.asarray(payload["observed_s"], dtype=np.float64)
        source_ids = np.asarray(payload["source_ids"], dtype=np.int64)
        receiver_ids = (
            np.asarray(payload["receiver_ids"], dtype=np.int64)
            if "receiver_ids" in payload.files
            else None
        )
        coordinates = np.asarray(payload["coordinates"], dtype=np.float64)
        background_velocity = float(payload["background_velocity"])
        velocity_min = float(payload["velocity_min"])
        velocity_max = float(payload["velocity_max"])
        coverage_gate = (
            np.asarray(payload["coverage_gate"], dtype=np.float64)
            if "coverage_gate" in payload.files
            else np.ones(path_matrix.shape[1], dtype=np.float64)
        )
        has_validation = "validation_observed_s" in payload.files
        if has_validation:
            validation_matrix = csr_matrix(
                (
                    np.asarray(
                        payload["validation_matrix_data"], dtype=np.float64
                    ),
                    np.asarray(
                        payload["validation_matrix_indices"], dtype=np.int32
                    ),
                    np.asarray(
                        payload["validation_matrix_indptr"], dtype=np.int32
                    ),
                ),
                shape=tuple(
                    int(value)
                    for value in payload["validation_matrix_shape"]
                ),
            )
            validation_observed_s = np.asarray(
                payload["validation_observed_s"], dtype=np.float64
            )
            validation_source_ids = np.asarray(
                payload["validation_source_ids"], dtype=np.int64
            )
            validation_receiver_ids = (
                np.asarray(
                    payload["validation_receiver_ids"], dtype=np.int64
                )
                if "validation_receiver_ids" in payload.files
                else None
            )
        else:
            validation_matrix = None
            validation_observed_s = None
            validation_source_ids = None
            validation_receiver_ids = None

    config = DeepReparamConfig(
        fixed_epochs=args.fixed_epochs,
        max_epochs=(
            args.fixed_epochs if args.fixed_epochs > 0 else args.max_epochs
        ),
        width=args.network_width,
        depth=args.network_depth,
        learning_rate=args.learning_rate,
        huber_beta_ms=args.huber_beta_ms,
        output_l2=args.output_l2,
        random_seed=args.random_seed,
        device=args.device,
        event_static_shrinkage=args.event_static_shrinkage,
        receiver_static_shrinkage=args.receiver_static_shrinkage,
        receiver_static_max_ms=args.receiver_static_max_ms,
        static_profile_iterations=args.static_profile_iterations,
        fourier_bands=args.fourier_bands,
        neighbor_smoothness=args.neighbor_smoothness,
        total_variation=args.total_variation,
        total_variation_epsilon=args.total_variation_epsilon,
        final_learning_rate_fraction=args.final_learning_rate_fraction,
        differential_loss_fraction=args.differential_loss_fraction,
    )
    result = fit_deep_reparameterized_slowness(
        path_matrix,
        observed_s,
        source_ids,
        coordinates,
        coverage_gate,
        background_velocity,
        velocity_min,
        velocity_max,
        config=config,
        validation_path_matrix=validation_matrix,
        validation_observed_s=validation_observed_s,
        validation_source_ids=validation_source_ids,
        receiver_ids=receiver_ids,
        validation_receiver_ids=validation_receiver_ids,
    )
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        slowness=result.slowness,
        history=result.history,
        train_rms_ms=np.asarray(result.train_rms_ms),
        validation_rms_ms=np.asarray(result.validation_rms_ms),
        validation_metric_rays=np.asarray(result.validation_metric_rays),
        best_epoch=np.asarray(result.best_epoch),
        parameter_count=np.asarray(result.parameter_count),
        runtime_seconds=np.asarray(result.runtime_seconds),
        device=np.asarray(result.device),
        random_seed=np.asarray(args.random_seed),
        receiver_static_ids=result.receiver_static_ids,
        receiver_static_corrections_s=result.receiver_static_corrections_s,
        fixed_epochs=np.asarray(args.fixed_epochs),
        fourier_bands=np.asarray(args.fourier_bands),
        neighbor_smoothness=np.asarray(args.neighbor_smoothness),
        total_variation=np.asarray(args.total_variation),
        total_variation_epsilon=np.asarray(args.total_variation_epsilon),
        coverage_gate_min=np.asarray(float(np.min(coverage_gate))),
        coverage_gate_mean=np.asarray(float(np.mean(coverage_gate))),
        coverage_gate_max=np.asarray(float(np.max(coverage_gate))),
        final_learning_rate_fraction=np.asarray(
            args.final_learning_rate_fraction
        ),
        differential_loss_fraction=np.asarray(
            args.differential_loss_fraction
        ),
    )


if __name__ == "__main__":
    main()
