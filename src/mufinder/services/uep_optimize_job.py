"""JSON-backed execution helpers for MuFinder-native UEP optimization jobs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from mufinder.science.uep import (
    UepError,
    UepField,
    UepOptimizationResult,
    UepProbeResult,
    load_uep_source,
    optimize_muon_position,
    particle_mass_kg_from_amu,
    probe_point,
)
from mufinder.science.uep_clustering import (
    UepClusteringConfig,
    cluster_optimized_positions,
    cluster_uep_optimizations,
)


@dataclass(frozen=True, slots=True)
class UepJobResult:
    """Parsed result for one MuFinder-native UEP optimization job."""

    completed: bool
    optimization: UepOptimizationResult | None = None
    probe: UepProbeResult | None = None
    error_message: str | None = None


def manifest_file_name(seed_name: str) -> str:
    return f"{seed_name}.uep.json"


def result_file_name(seed_name: str) -> str:
    return f"{seed_name}.uep.result.json"


def write_manifest(job_dir: str | Path, seed_name: str, payload: dict[str, Any]) -> Path:
    """Write one UEP optimization manifest into the job directory."""

    job_path = Path(job_dir)
    job_path.mkdir(parents=True, exist_ok=True)
    target = job_path / manifest_file_name(seed_name)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return target


def _parse_classification(payload: dict[str, Any]):
    from mufinder.science.uep import StationaryPointClassification

    return StationaryPointClassification(
        label=str(payload.get("label", "flat")),
        eigenvalues=tuple(float(value) for value in payload.get("eigenvalues", (0.0, 0.0, 0.0))),
        positive_count=int(payload.get("positive_count", 0)),
        negative_count=int(payload.get("negative_count", 0)),
        near_zero_count=int(payload.get("near_zero_count", 0)),
    )


def parse_result_file(job_dir: str | Path, seed_name: str) -> UepJobResult:
    """Parse one completed UEP optimization result JSON file."""

    source = Path(job_dir) / result_file_name(seed_name)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise UepError(f"Failed to read UEP result file '{source.name}': {exc}") from exc

    optimization_payload = payload.get("optimization")
    if not isinstance(optimization_payload, dict):
        return UepJobResult(
            completed=bool(payload.get("completed", False)),
            optimization=None,
            probe=None,
            error_message=str(payload.get("error_message", "Missing optimization payload.")),
        )

    classification_payload = optimization_payload.get("classification", {})
    optimization = UepOptimizationResult(
        start_position_ang=tuple(
            float(value)
            for value in optimization_payload.get("start_position_ang", (0.0, 0.0, 0.0))
        ),
        optimized_position_ang=tuple(
            float(value)
            for value in optimization_payload.get("optimized_position_ang", (0.0, 0.0, 0.0))
        ),
        optimized_fractional_position=tuple(
            float(value)
            for value in optimization_payload.get("optimized_fractional_position", (0.0, 0.0, 0.0))
        ),
        classical_energy_ev=float(optimization_payload.get("classical_energy_ev", 0.0)),
        zero_point_energy_ev=(
            float(optimization_payload["zero_point_energy_ev"])
            if optimization_payload.get("zero_point_energy_ev") is not None
            else None
        ),
        total_energy_ev=(
            float(optimization_payload["total_energy_ev"])
            if optimization_payload.get("total_energy_ev") is not None
            else None
        ),
        success=bool(optimization_payload.get("success", False)),
        message=str(optimization_payload.get("message", "")),
        iterations=int(optimization_payload.get("iterations", 0)),
        classification=_parse_classification(
            classification_payload if isinstance(classification_payload, dict) else {}
        ),
    )
    probe_payload = payload.get("probe")
    return UepJobResult(
        completed=bool(payload.get("completed", False)),
        optimization=optimization,
        probe=payload_to_probe(probe_payload) if isinstance(probe_payload, dict) else None,
        error_message=str(payload.get("error_message")) if payload.get("error_message") else None,
    )


def payload_to_probe(payload: dict[str, Any]) -> UepProbeResult:
    """Deserialize a stored probe payload into a typed result."""

    classification_payload = payload.get("classification", {})
    classification = _parse_classification(
        classification_payload if isinstance(classification_payload, dict) else {}
    )
    return UepProbeResult(
        position_ang=tuple(float(value) for value in payload.get("position_ang", (0.0, 0.0, 0.0))),
        fractional_position=tuple(
            float(value) for value in payload.get("fractional_position", (0.0, 0.0, 0.0))
        ),
        potential_total_ev=float(payload.get("potential_total_ev", 0.0)),
        potential_electronic_ev=float(payload.get("potential_electronic_ev", 0.0)),
        potential_ionic_ev=float(payload.get("potential_ionic_ev", 0.0)),
        gradient_total_ev_per_ang=tuple(
            float(value) for value in payload.get("gradient_total_ev_per_ang", (0.0, 0.0, 0.0))
        ),
        gradient_electronic_ev_per_ang=tuple(
            float(value) for value in payload.get("gradient_electronic_ev_per_ang", (0.0, 0.0, 0.0))
        ),
        gradient_ionic_ev_per_ang=tuple(
            float(value) for value in payload.get("gradient_ionic_ev_per_ang", (0.0, 0.0, 0.0))
        ),
        hessian_total_ev_per_ang2=tuple(
            tuple(float(value) for value in row)
            for row in payload.get("hessian_total_ev_per_ang2", ((0.0, 0.0, 0.0),) * 3)
        ),
        hessian_electronic_ev_per_ang2=tuple(
            tuple(float(value) for value in row)
            for row in payload.get("hessian_electronic_ev_per_ang2", ((0.0, 0.0, 0.0),) * 3)
        ),
        hessian_ionic_ev_per_ang2=tuple(
            tuple(float(value) for value in row)
            for row in payload.get("hessian_ionic_ev_per_ang2", ((0.0, 0.0, 0.0),) * 3)
        ),
        classification=classification,
        zero_point_energy_ev=float(payload["zero_point_energy_ev"])
        if payload.get("zero_point_energy_ev") is not None
        else None,
        total_quantum_energy_ev=float(payload["total_quantum_energy_ev"])
        if payload.get("total_quantum_energy_ev") is not None
        else None,
    )


def run_manifest_file(path: str | Path) -> Path:
    """Execute a UEP optimization manifest and write the result JSON beside it."""

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    seed_name = str(payload.get("seed_name", source.stem))
    job_dir = source.parent
    try:
        source_data = load_uep_source(
            den_fmt_path=job_dir / str(payload["density_file"]),
            castep_path=job_dir / str(payload["castep_file"]),
            cell_path=job_dir / str(payload["cell_file"]) if payload.get("cell_file") else None,
            gw_factor=float(payload.get("gw_factor", 5.0)),
        )
        field = UepField(source_data)
        particle_mass_kg = (
            float(payload["particle_mass_kg"])
            if payload.get("particle_mass_kg") is not None
            else particle_mass_kg_from_amu(float(payload.get("particle_mass_amu", 0.1134289257)))
        )
        optimization = optimize_muon_position(
            field,
            start_position_ang=tuple(
                float(value) for value in payload.get("start_position_ang", (0.0, 0.0, 0.0))
            ),
            particle_mass_kg=particle_mass_kg,
            tolerance_ev=float(payload.get("tolerance_ev", 1e-5)),
            max_steps=int(payload.get("max_steps", 300)),
            method=str(payload.get("method", "trust-exact")),
        )
        optimize_probe = probe_point(field, optimization.optimized_position_ang)
        result_payload = {
            "completed": bool(optimization.success),
            "optimization": asdict(optimization),
            "probe": asdict(optimize_probe),
            "error_message": None if optimization.success else optimization.message,
        }
    except Exception as exc:  # noqa: BLE001
        result_payload = {
            "completed": False,
            "optimization": None,
            "probe": None,
            "error_message": str(exc),
        }

    target = job_dir / result_file_name(seed_name)
    target.write_text(json.dumps(result_payload, indent=2, sort_keys=True), encoding="utf-8")
    return target


def summarize_batch_results(
    optimizations: list[UepOptimizationResult],
    *,
    threshold_ang: float | None = None,
    source_structure: object | None = None,
    clustering_config: UepClusteringConfig | None = None,
) -> dict[str, Any]:
    """Return a JSON-friendly cluster summary for a finished batch."""

    if clustering_config is not None or source_structure is not None:
        if source_structure is None:
            raise UepError("source_structure is required when using soprano-backed clustering.")
        summary = cluster_uep_optimizations(
            optimizations,
            source_structure=source_structure,
            config=clustering_config,
        ).summary
    else:
        summary = cluster_optimized_positions(
            optimizations,
            threshold_ang=float(0.2 if threshold_ang is None else threshold_ang),
        )
    return asdict(summary)
