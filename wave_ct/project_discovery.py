"""Discover lineage-aware WaveCT projects below a workspace root.

New projects advertise their cohort identity in ``wave_ct_project.json``.
Older projects predate that metadata, so callers may provide their existing
hard-coded cohorts as a compatibility fallback. Project metadata always wins
when both sources point at the same travel-time CSV.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable

from wave_ct.config import (
    PROJECT_FILENAME,
    load_project_config,
    resolve_project_path,
)


@dataclass(frozen=True)
class ProjectCohort:
    """A standard WaveCT input and its lineage/model locations."""

    dataset_id: str
    data_type: str
    independence_group: str
    input_csv: Path
    model_npz: Path
    project_path: Path | None = None


def _path_key(path: Path) -> str:
    """Return a stable, case-insensitive key without requiring path existence."""

    try:
        value = path.resolve(strict=False)
    except OSError:
        value = path.absolute()
    return os.path.normcase(str(value))


def _configured_model_path(config: dict[str, object], project_dir: Path) -> Path:
    outputs = config.get("outputs")
    model = config.get("model")
    output_values = outputs if isinstance(outputs, dict) else {}
    model_values = model if isinstance(model, dict) else {}
    explicit = (
        output_values.get("model_npz")
        or output_values.get("velocity_model")
        or model_values.get("model_npz")
        or model_values.get("velocity_model")
    )
    if str(explicit or "").strip():
        resolved = resolve_project_path(explicit, project_dir)
        path = Path(resolved).expanduser()
        if not path.is_absolute():
            path = project_dir / path
        return path.resolve(strict=False)
    output_dir = str(output_values.get("directory") or "").strip()
    if output_dir:
        return (Path(output_dir).expanduser() / "velocity_model.npz").resolve(
            strict=False
        )
    return (project_dir / "inversion" / "velocity_model.npz").resolve(strict=False)


def _read_project(project_path: Path) -> ProjectCohort | None:
    """Read one metadata-bearing project, returning ``None`` when incomplete."""

    try:
        config = load_project_config(project_path)
    except (OSError, ValueError, TypeError):
        return None
    dataset = config.get("dataset")
    inputs = config.get("inputs")
    if not isinstance(dataset, dict) or not isinstance(inputs, dict):
        return None
    dataset_id = str(dataset.get("dataset_id") or "").strip()
    data_type = str(dataset.get("data_type") or "").strip()
    independence_group = str(dataset.get("independence_group") or "").strip()
    input_value = str(inputs.get("travel_time_csv") or "").strip()

    # Schema-1 projects have no lineage metadata. Their established identities
    # are supplied by each caller's legacy fallback instead of being guessed.
    if not dataset_id or not data_type or not independence_group or not input_value:
        return None
    input_csv = Path(input_value).expanduser().resolve(strict=False)
    if not input_csv.is_file():
        return None
    return ProjectCohort(
        dataset_id=dataset_id,
        data_type=data_type,
        independence_group=independence_group,
        input_csv=input_csv,
        model_npz=_configured_model_path(config, project_path.resolve().parent),
        project_path=project_path.resolve(),
    )


def _deduplicate(cohorts: Iterable[ProjectCohort]) -> list[ProjectCohort]:
    """Deduplicate copied configs by both dataset identity and input path."""

    result: list[ProjectCohort] = []
    dataset_ids: set[str] = set()
    input_paths: set[str] = set()
    for cohort in cohorts:
        dataset_key = cohort.dataset_id.casefold()
        input_key = _path_key(cohort.input_csv)
        if dataset_key in dataset_ids or input_key in input_paths:
            continue
        dataset_ids.add(dataset_key)
        input_paths.add(input_key)
        result.append(cohort)
    return result


def discover_project_cohorts(
    project_root: Path,
    *,
    legacy_cohorts: Iterable[ProjectCohort] = (),
    require_model: bool = False,
) -> list[ProjectCohort]:
    """Discover standard cohorts recursively and merge caller-owned fallbacks.

    Only the standard travel-time CSV is required for catalog discovery. Set
    ``require_model`` for experiments which need an existing quantitative
    inversion model. Presentation-only fields are never considered models.
    """

    project_root = project_root.expanduser().resolve(strict=False)
    discovered: list[ProjectCohort] = []
    if project_root.is_dir():
        for project_path in sorted(project_root.rglob(PROJECT_FILENAME)):
            cohort = _read_project(project_path)
            if cohort is not None:
                discovered.append(cohort)

    # Dynamic metadata is authoritative. Legacy entries only fill gaps.
    merged = _deduplicate([*discovered, *legacy_cohorts])
    if require_model:
        merged = [cohort for cohort in merged if cohort.model_npz.is_file()]
    return sorted(
        merged,
        key=lambda cohort: (
            cohort.independence_group.casefold(),
            cohort.dataset_id.casefold(),
            _path_key(cohort.input_csv),
        ),
    )
