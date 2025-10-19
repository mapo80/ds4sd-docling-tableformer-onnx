"""Run TableFormer inference using the lightweight Docling extraction."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent
sys.path.append(str(REPO_ROOT / "tableformer-docling"))

from tableformer_docling import TableFormerDocling  # noqa: E402


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)!r} is not JSON serializable")


def run_inference(
    dataset_dir: Path,
    output_path: Path,
    *,
    artifacts_dir: Path,
    device: str = "cpu",
    num_threads: int = 4,
) -> None:
    runner = TableFormerDocling(
        artifacts_dir=artifacts_dir,
        device=device,
        num_threads=num_threads,
    )

    results: dict[str, Any] = {}
    image_paths = sorted(dataset_dir.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No PNG files found in {dataset_dir}")

    for image_path in image_paths:
        image = np.asarray(Image.open(image_path).convert("RGB"))
        predictions = runner.predict_page(image)
        results[image_path.name] = {
            "num_tables": len(predictions),
            "tables": predictions,
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2, default=_json_default)

    print(f"Saved {len(results)} predictions to {output_path}")


if __name__ == "__main__":
    dataset_dir = REPO_ROOT / "dataset" / "FinTabNet" / "benchmark"
    output_path = REPO_ROOT / "results" / "tableformer_docling_fintabnet.json"
    artifacts_dir = REPO_ROOT / "tableformer-docling" / "artifacts"

    run_inference(
        dataset_dir=dataset_dir,
        output_path=output_path,
        artifacts_dir=artifacts_dir,
    )
