"""Export resized page and table crop reference data from the Python pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "dataset" / "FinTabNet" / "benchmark"
PAGE_INPUT_REFERENCE = REPO_ROOT / "results" / "tableformer_page_input_reference.json"
OUTPUT_PATH = REPO_ROOT / "results" / "tableformer_table_crops_reference.json"


def _serialize_numpy(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)!r} is not JSON serializable")


def _load_page_input_bboxes(reference_path: Path) -> dict[str, list[list[float]]]:
    if not reference_path.exists():
        raise FileNotFoundError(
            "Page input reference not found. Run scripts/export_tableformer_page_input.py first."
        )

    with reference_path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)

    bbox_map: dict[str, list[list[float]]] = {}
    for sample in payload.get("samples", []):
        image_name = sample["image_name"]
        table_bboxes = sample.get("table_bboxes", [])
        bbox_map[image_name] = [
            [float(coord) for coord in bbox]
            for bbox in table_bboxes
        ]

    if not bbox_map:
        raise ValueError("No table bounding boxes found in the page input reference.")

    return bbox_map


def _resize_image(image: np.ndarray, *, target_height: int = 1024) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    if height == target_height:
        return image.copy(), 1.0

    scale_factor = target_height / float(height)
    resized_width = int(width * scale_factor)
    resized = _resize_rgb_bilinear(image, resized_width, target_height)
    return resized, scale_factor


def _resize_rgb_bilinear(image: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    image = image.astype(np.float64, copy=False)
    src_height, src_width = image.shape[:2]
    scale_x = target_width / float(src_width)
    scale_y = target_height / float(src_height)

    x_coords = np.arange(target_width, dtype=np.float64) / scale_x
    y_coords = np.arange(target_height, dtype=np.float64) / scale_y

    x0 = np.floor(x_coords).astype(int)
    y0 = np.floor(y_coords).astype(int)
    x_lerp = (x_coords - x0)[None, :, None]
    y_lerp = (y_coords - y0)[:, None, None]

    x0 = np.clip(x0, 0, src_width - 1)
    y0 = np.clip(y0, 0, src_height - 1)
    x1 = np.clip(x0 + 1, 0, src_width - 1)
    y1 = np.clip(y0 + 1, 0, src_height - 1)

    x_lerp = np.where((x0 == x1)[None, :, None], 0.0, x_lerp)
    y_lerp = np.where((y0 == y1)[:, None, None], 0.0, y_lerp)

    image_y0 = image[y0]
    image_y1 = image[y1]

    p00 = image_y0[:, x0]
    p10 = image_y0[:, x1]
    p01 = image_y1[:, x0]
    p11 = image_y1[:, x1]

    top = p00 + (p10 - p00) * x_lerp
    bottom = p01 + (p11 - p01) * x_lerp
    value = top + (bottom - top) * y_lerp

    return np.clip(np.round(value), 0, 255).astype(np.uint8)


def export_table_crops(dataset_dir: Path, reference_path: Path, output_path: Path) -> None:
    bbox_map = _load_page_input_bboxes(reference_path)

    image_paths = sorted(p for p in dataset_dir.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No PNG files found in {dataset_dir}")

    samples: list[dict[str, Any]] = []

    for image_path in image_paths:
        if image_path.name not in bbox_map:
            raise KeyError(
                f"Image '{image_path.name}' missing from page input reference '{reference_path}'."
            )

        image = np.asarray(Image.open(image_path).convert("RGB"))
        height, width = image.shape[:2]

        resized, scale_factor = _resize_image(image)
        resized_height, resized_width = resized.shape[:2]

        table_crops: list[dict[str, Any]] = []
        for index, bbox in enumerate(bbox_map[image_path.name]):
            scaled_bbox = [float(coord * scale_factor) for coord in bbox]
            rounded_bbox = [int(round(coord)) for coord in scaled_bbox]

            left, top, right, bottom = rounded_bbox
            left = max(0, min(left, resized_width))
            top = max(0, min(top, resized_height))
            right = max(left, min(right, resized_width))
            bottom = max(top, min(bottom, resized_height))

            crop = resized[top:bottom, left:right]
            if crop.size == 0:
                raise ValueError(
                    f"Empty crop encountered for image '{image_path.name}' bbox index {index}."
                )

            crop_bytes = crop.tobytes()
            crop_sha256 = hashlib.sha256(crop_bytes).hexdigest()
            crop_mean = float(np.mean(crop))
            crop_std = float(np.std(crop))

            original_width = float(bbox[2] - bbox[0])
            original_height = float(bbox[3] - bbox[1])
            scaled_width = float(scaled_bbox[2] - scaled_bbox[0])
            scaled_height = float(scaled_bbox[3] - scaled_bbox[1])
            rounded_width = int(right - left)
            rounded_height = int(bottom - top)

            table_crops.append(
                {
                    "table_index": index,
                    "original_bbox": bbox,
                    "scaled_bbox": scaled_bbox,
                    "rounded_bbox": [left, top, right, bottom],
                    "original_bbox_pixel_width": original_width,
                    "original_bbox_pixel_height": original_height,
                    "scaled_bbox_pixel_width": scaled_width,
                    "scaled_bbox_pixel_height": scaled_height,
                    "rounded_bbox_pixel_width": rounded_width,
                    "rounded_bbox_pixel_height": rounded_height,
                    "crop_width": int(crop.shape[1]),
                    "crop_height": int(crop.shape[0]),
                    "crop_byte_length": len(crop_bytes),
                    "crop_image_sha256": crop_sha256,
                    "crop_mean_pixel_value": crop_mean,
                    "crop_std_pixel_value": crop_std,
                    "crop_channels": int(crop.shape[2]),
                }
            )

        samples.append(
            {
                "image_name": image_path.name,
                "original_width": float(width),
                "original_height": float(height),
                "scale_factor": float(scale_factor),
                "resized_width": int(resized_width),
                "resized_height": int(resized_height),
                "table_crops": table_crops,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump({"samples": samples}, fp, indent=2, default=_serialize_numpy)

    print(f"Exported {len(samples)} table crop samples to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DATASET_DIR,
        help="Directory containing FinTabNet benchmark PNG images.",
    )
    parser.add_argument(
        "--page-input-reference",
        type=Path,
        default=PAGE_INPUT_REFERENCE,
        help="Path to the page input reference JSON produced by export_tableformer_page_input.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Path to write the JSON reference.",
    )
    args = parser.parse_args()

    export_table_crops(args.dataset_dir, args.page_input_reference, args.output)
