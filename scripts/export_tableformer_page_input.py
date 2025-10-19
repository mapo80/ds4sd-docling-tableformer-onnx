"""Export TableFormer page input reference data from the Python pipeline."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "dataset" / "FinTabNet" / "benchmark"
OUTPUT_PATH = REPO_ROOT / "results" / "tableformer_page_input_reference.json"


def _serialize_numpy(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)!r} is not JSON serializable")


def export_page_inputs(dataset_dir: Path, output_path: Path) -> None:
    image_paths = sorted(p for p in dataset_dir.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No PNG files found in {dataset_dir}")

    samples: list[dict[str, Any]] = []

    for image_path in image_paths:
        image = np.asarray(Image.open(image_path).convert("RGB"))
        height, width = image.shape[:2]
        raw_bytes = image.tobytes()
        sha256 = hashlib.sha256(raw_bytes).hexdigest()
        table_bboxes = [
            [0.0, 0.0, float(width), float(height)],
        ]

        samples.append(
            {
                "image_name": image_path.name,
                "width": float(width),
                "height": float(height),
                "tokens": [],
                "table_bboxes": table_bboxes,
                "image_bytes_base64": base64.b64encode(raw_bytes).decode("ascii"),
                "image_sha256": sha256,
                "byte_length": len(raw_bytes),
                "channels": 3,
                "dtype": "uint8",
                "shape": [int(height), int(width), 3],
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump({"samples": samples}, fp, indent=2, default=_serialize_numpy)

    print(f"Exported {len(samples)} page inputs to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DATASET_DIR,
        help="Directory containing FinTabNet benchmark PNG images.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Path to write the JSON reference.",
    )
    args = parser.parse_args()

    export_page_inputs(args.dataset_dir, args.output)
