"""Export normalized tensor batches from Docling's TableFormer pipeline."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import requests
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "dataset" / "FinTabNet" / "benchmark"
PAGE_INPUT_REFERENCE = REPO_ROOT / "results" / "tableformer_page_input_reference.json"
OUTPUT_PATH = REPO_ROOT / "results" / "tableformer_image_tensors_reference.json"
CONFIG_URL = (
    "https://huggingface.co/{repo_id}/resolve/{revision}/model_artifacts/"
    "tableformer/{variant}/tm_config.json?download=1"
)


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

    return value.astype(np.float32)


def _extract_crop(resized: np.ndarray, bbox: list[float], scale_factor: float) -> np.ndarray:
    scaled_bbox = [float(coord * scale_factor) for coord in bbox]
    rounded_bbox = [int(round(coord)) for coord in scaled_bbox]

    left, top, right, bottom = rounded_bbox
    left = max(0, min(left, resized.shape[1]))
    top = max(0, min(top, resized.shape[0]))
    right = max(left, min(right, resized.shape[1]))
    bottom = max(top, min(bottom, resized.shape[0]))

    crop = resized[top:bottom, left:right]
    if crop.size == 0:
        raise ValueError("Encountered empty crop during tensor export.")

    return crop


def _resize_crop_to_target(crop: np.ndarray, target_size: int) -> np.ndarray:
    if crop.shape[0] == target_size and crop.shape[1] == target_size:
        return crop.copy()
    return _resize_rgb_bilinear(crop, target_size, target_size)


def _load_config(repo_id: str, revision: str, variant: str) -> dict[str, Any]:
    url = CONFIG_URL.format(repo_id=repo_id, revision=revision, variant=variant)
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.json()


def _tensorize_crop(
    crop: np.ndarray,
    mean: list[float],
    std: list[float],
    target_size: int,
) -> np.ndarray:
    resized = _resize_crop_to_target(crop, target_size)
    resized = np.clip(resized, 0.0, 255.0)

    normalized = np.empty_like(resized, dtype=np.float32)
    for channel, (channel_mean, channel_std) in enumerate(zip(mean, std, strict=False)):
        channel_values = resized[:, :, channel] / 255.0
        normalized[:, :, channel] = (channel_values - channel_mean) / channel_std

    transposed = normalized.transpose(2, 1, 0)
    return transposed[np.newaxis, ...].astype(np.float32, copy=False)


def export_image_tensors(
    dataset_dir: Path,
    page_reference_path: Path,
    output_path: Path,
    *,
    repo_id: str = "ds4sd/docling-models",
    revision: str = "main",
    variant: str = "fast",
) -> None:
    bbox_map = _load_page_input_bboxes(page_reference_path)

    config = _load_config(repo_id, revision, variant)
    dataset_config = config.get("dataset", {})
    normalization = dataset_config.get("image_normalization", {})
    mean = list(map(float, normalization.get("mean", [])))
    std = list(map(float, normalization.get("std", [])))
    target_size = int(dataset_config.get("resized_image", 448))

    if len(mean) != 3 or len(std) != 3:
        raise ValueError("Expected three-channel normalization parameters.")

    image_paths = sorted(p for p in dataset_dir.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No PNG files found in {dataset_dir}")

    samples: list[dict[str, Any]] = []

    for image_path in image_paths:
        if image_path.name not in bbox_map:
            raise KeyError(
                f"Image '{image_path.name}' missing from page input reference '{page_reference_path}'."
            )

        image = np.asarray(Image.open(image_path).convert("RGB"))
        resized, scale_factor = _resize_image(image)

        for table_index, bbox in enumerate(bbox_map[image_path.name]):
            crop = _extract_crop(resized, bbox, scale_factor)
            tensor_np = _tensorize_crop(crop, mean, std, target_size)

            tensor_bytes = tensor_np.tobytes(order="C")
            tensor_sha256 = hashlib.sha256(tensor_bytes).hexdigest()
            compressed = zlib.compress(tensor_bytes, level=9)
            tensor_base64 = base64.b64encode(compressed).decode("ascii")

            samples.append(
                {
                    "image_name": image_path.name,
                    "table_index": table_index,
                    "tensor_shape": list(tensor_np.shape),
                    "tensor_sha256": tensor_sha256,
                    "tensor_zlib_base64": tensor_base64,
                    "tensor_min": float(tensor_np.min()),
                    "tensor_max": float(tensor_np.max()),
                    "tensor_mean": float(tensor_np.mean()),
                    "tensor_std": float(tensor_np.std()),
                }
            )

    payload: dict[str, Any] = {
        "target_size": target_size,
        "channels": 3,
        "normalization": {
            "mean": mean,
            "std": std,
        },
        "samples": samples,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, default=_serialize_numpy)
        fp.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DATASET_DIR,
        help="Path to the FinTabNet benchmark directory.",
    )
    parser.add_argument(
        "--page-reference",
        type=Path,
        default=PAGE_INPUT_REFERENCE,
        help="Path to the page input reference JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Where to write the tensor reference JSON.",
    )

    args = parser.parse_args()
    export_image_tensors(args.dataset_dir, args.page_reference, args.output)


if __name__ == "__main__":
    main()
