"""Export TableFormer initialization reference data from Docling Python."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT / "tableformer-docling"))

from tableformer_docling import TableFormerDocling  # noqa: E402
from safetensors.numpy import load_file  # noqa: E402


@dataclass
class ExportOptions:
    repo_id: str
    revision: str
    variant: str
    artifacts_dir: Path
    output_path: Path


def parse_args(argv: list[str] | None = None) -> ExportOptions:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="ds4sd/docling-models")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--variant", default="fast")
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=REPO_ROOT / "tableformer-docling" / "artifacts",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results" / "tableformer_init_fast_reference.json",
    )
    args = parser.parse_args(argv)
    return ExportOptions(
        repo_id=args.repo_id,
        revision=args.revision,
        variant=args.variant,
        artifacts_dir=args.artifacts_dir,
        output_path=args.output,
    )


def canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def export_reference(options: ExportOptions) -> dict[str, Any]:
    runner = TableFormerDocling(
        artifacts_dir=options.artifacts_dir,
        repo_id=options.repo_id,
        revision=options.revision,
        variant=options.variant,
    )

    predictor = runner.predictor
    config = runner.get_config()
    init_data = predictor.get_init_data()

    word_map = init_data["word_map"]
    word_map_canonical = canonical_json(word_map)
    word_map_sha = sha256_bytes(word_map_canonical.encode("utf-8"))

    model_dir = Path(config["model"]["save_dir"])
    weight_paths = sorted(model_dir.glob("*.safetensors"))
    if not weight_paths:
        raise FileNotFoundError(f"No safetensors files found in {model_dir}")

    weight_entries: list[dict[str, Any]] = []
    for weight_path in weight_paths:
        tensors = load_file(str(weight_path))
        tensor_entries: list[dict[str, Any]] = []
        for name in sorted(tensors):
            array = tensors[name]
            tensor_entries.append(
                {
                    "name": name,
                    "dtype": str(array.dtype),
                    "shape": [int(dim) for dim in array.shape],
                    "sha256": sha256_bytes(array.tobytes()),
                }
            )

        weight_entries.append(
            {
                "file_name": weight_path.name,
                "sha256": sha256_file(weight_path),
                "tensors": tensor_entries,
            }
        )

    return {
        "repo_id": options.repo_id,
        "revision": options.revision,
        "variant": options.variant,
        "artifacts_dir": str(options.artifacts_dir),
        "word_map": word_map,
        "word_map_canonical_json": word_map_canonical,
        "word_map_sha256": word_map_sha,
        "weight_files": weight_entries,
    }


def main(argv: list[str] | None = None) -> None:
    options = parse_args(argv)
    reference = export_reference(options)
    options.output_path.parent.mkdir(parents=True, exist_ok=True)
    with options.output_path.open("w", encoding="utf-8") as fp:
        json.dump(reference, fp, indent=2)
        fp.write("\n")
    print(f"Wrote reference to {options.output_path}")


if __name__ == "__main__":
    main()
