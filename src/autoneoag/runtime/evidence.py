from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from autoneoag.config import Settings
from autoneoag.manifests import load_manifest_bundle, manifest_summary
from autoneoag.tasks import processed_dataset_path, split_manifest_path, task_interim_dir


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(path: Path) -> dict[str, object]:
    exists = path.exists()
    stat = path.stat() if exists else None
    return {
        "path": str(path),
        "exists": exists,
        "size": stat.st_size if stat is not None else None,
        "mtime_ns": stat.st_mtime_ns if stat is not None else None,
        "sha256": _sha256(path) if exists and path.is_file() else None,
    }


def _manifest_paths(settings: Settings, task_id: str) -> dict[str, Path]:
    bundle = load_manifest_bundle(settings, task_id)
    return {
        "data_card": bundle.data_card_path,
        "source_manifest": bundle.source_manifest_path,
        "lockbox_manifest": bundle.lockbox_manifest_path,
        "split_manifest": bundle.split_manifest_path,
        "task_policy": bundle.task_policy_path,
    }


def _evidence_targets(settings: Settings, task_id: str, mode: str) -> dict[str, Path]:
    interim_dir = task_interim_dir(settings, task_id, mode)
    targets = {
        "dataset": processed_dataset_path(settings, task_id, mode),
        "sample_split_manifest": split_manifest_path(settings, task_id, mode),
    }
    if mode == "full":
        targets.update(
            {
                "manifest_summary": interim_dir / "manifest_summary.json",
                "source_manifest_staged": interim_dir / "source_manifest_staged.tsv",
                "full_prepare_plan": interim_dir / "full_prepare_plan.json",
                "source_index": interim_dir / "source_index.tsv",
            }
        )
    return targets


def evidence_bundle_path(logs_dir: Path) -> Path:
    return logs_dir / "freeze" / "evidence_bundle.json"


def evidence_bundle_id(task_id: str, mode: str, strategy: str, run_policy: str, run_id: int) -> str:
    return f"{task_id}-{mode}-{strategy}-{run_policy}-run_{run_id:02d}"


def create_evidence_bundle(
    settings: Settings,
    task_id: str,
    mode: str,
    strategy: str,
    run_policy: str,
    run_id: int,
    logs_dir: Path,
) -> dict[str, Any]:
    bundle = load_manifest_bundle(settings, task_id)
    targets = _evidence_targets(settings, task_id, mode)
    manifest_paths = _manifest_paths(settings, task_id)
    record = {
        "bundle_id": evidence_bundle_id(task_id, mode, strategy, run_policy, run_id),
        "task_id": task_id,
        "mode": mode,
        "strategy": strategy,
        "run_id": run_id,
        "run_policy": run_policy,
        "manifest_summary": manifest_summary(bundle),
        "artifacts": {name: _fingerprint(path) for name, path in targets.items()},
        "manifests": {name: _fingerprint(path) for name, path in manifest_paths.items()},
        "baseline": {
            "round_id": None,
            "commit": "",
            "checkpoint_path": "",
            "metrics_path": "",
            "dev_score": None,
            "confirm_score": None,
        },
    }
    path = evidence_bundle_path(logs_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True))
    return record


def load_evidence_bundle(logs_dir: Path) -> dict[str, Any] | None:
    path = evidence_bundle_path(logs_dir)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def write_evidence_bundle(logs_dir: Path, bundle: dict[str, Any]) -> Path:
    path = evidence_bundle_path(logs_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bundle, indent=2, sort_keys=True))
    return path


def update_baseline_record(
    logs_dir: Path,
    bundle: dict[str, Any],
    *,
    round_id: int,
    commit: str,
    checkpoint_path: str,
    metrics_path: str,
    dev_score: float,
    confirm_score: float | None,
) -> dict[str, Any]:
    bundle = dict(bundle)
    baseline = dict(bundle.get("baseline", {}))
    baseline.update(
        {
            "round_id": round_id,
            "commit": commit,
            "checkpoint_path": checkpoint_path,
            "metrics_path": metrics_path,
            "dev_score": dev_score,
            "confirm_score": confirm_score,
        }
    )
    bundle["baseline"] = baseline
    write_evidence_bundle(logs_dir, bundle)
    return bundle


def validate_evidence_bundle(bundle: dict[str, Any]) -> list[str]:
    mismatches: list[str] = []
    for group_name in ("artifacts", "manifests"):
        for name, record in bundle.get(group_name, {}).items():
            path = Path(str(record["path"]))
            exists_now = path.exists()
            if exists_now != bool(record.get("exists")):
                mismatches.append(f"{group_name}:{name}: existence changed")
                continue
            if not exists_now:
                continue
            stat = path.stat()
            size_changed = stat.st_size != record.get("size")
            mtime_changed = stat.st_mtime_ns != record.get("mtime_ns")
            if not size_changed and not mtime_changed:
                continue
            sha_now = _sha256(path) if path.is_file() else None
            if sha_now != record.get("sha256"):
                mismatches.append(f"{group_name}:{name}: sha256 drift")
    return mismatches
