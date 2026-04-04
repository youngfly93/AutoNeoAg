from __future__ import annotations

import hashlib
from pathlib import Path

from autoneoag.runtime.evidence import validate_evidence_bundle
from autoneoag.runtime.policy import current_gate_stage, strict_confirm_gate_passes, strict_dev_gate_passes


def test_strict_confirm_helpers_gate_rounds() -> None:
    assert strict_dev_gate_passes(0.81, 0.80) is True
    assert strict_dev_gate_passes(0.80005, 0.80) is False
    assert strict_confirm_gate_passes(0.41, 0.40) is True
    assert strict_confirm_gate_passes(0.3998, 0.40) is False
    assert current_gate_stage(
        failure_type=None,
        confirm_gate_required=True,
        confirm_checked=True,
        dev_passed_gate=True,
    ) == "confirm"


def test_validate_evidence_bundle_detects_file_drift(tmp_path: Path) -> None:
    artifact = tmp_path / "dataset.parquet"
    artifact.write_text("version_a")
    stat = artifact.stat()
    bundle = {
        "artifacts": {
            "dataset": {
                "path": str(artifact),
                "exists": True,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            }
        },
        "manifests": {},
    }

    artifact.write_text("version_b")
    mismatches = validate_evidence_bundle(bundle)
    assert mismatches == ["artifacts:dataset: sha256 drift"]
