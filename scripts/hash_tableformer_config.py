"""Compute canonical JSON and SHA-256 hash for TableFormer tm_config."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import types
from pathlib import Path
from typing import Any, Dict

from huggingface_hub import snapshot_download

if "torch" not in sys.modules:
    sys.modules["torch"] = types.ModuleType("torch")

from docling_ibm_models.tableformer.common import read_config


def _canonicalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _canonicalize(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_canonicalize(item) for item in value]
    return value


def _compute_hash(config: Dict[str, Any]) -> Dict[str, Any]:
    canonical = _canonicalize(config)
    normalized = json.dumps(canonical, separators=(",", ":"), ensure_ascii=False)
    sha256 = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return {"canonical_json": normalized, "sha256": sha256}


def _download_artifacts(repo_id: str, revision: str, variant: str, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        allow_patterns=[f"model_artifacts/tableformer/{variant}/tm_config.json"],
        local_dir=target_dir,
        local_dir_use_symlinks=False,
    )
    return target_dir / "model_artifacts" / "tableformer" / variant


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="ds4sd/docling-models")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--variant", default="fast")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("tableformer-docling/artifacts"))
    parser.add_argument("--output", type=Path, default=Path("results/tableformer_config_hash.json"))
    args = parser.parse_args()

    model_dir = _download_artifacts(args.repo_id, args.revision, args.variant, args.artifacts_dir)
    config_path = model_dir / "tm_config.json"
    config = read_config(config_path)

    result = {
        "repo_id": args.repo_id,
        "revision": args.revision,
        "variant": args.variant,
        **_compute_hash(config),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    summary = {key: result[key] for key in ("repo_id", "revision", "variant", "sha256")}
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
