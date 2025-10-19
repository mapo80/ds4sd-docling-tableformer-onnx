#!/usr/bin/env python3
"""Export neural inference reference outputs for TableFormer."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import zlib
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from tableformer_docling.predictor import TableFormerDocling

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TENSOR_REFERENCE = REPO_ROOT / "results" / "tableformer_image_tensors_reference.json"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "results" / "tableformer_neural_outputs_reference.json"


def _decode_tensor(sample: dict[str, Any]) -> torch.Tensor:
    compressed = base64.b64decode(sample["tensor_zlib_base64"])
    decompressed = zlib.decompress(compressed)
    array = np.frombuffer(decompressed, dtype=np.float32)
    shape = sample["tensor_shape"]
    tensor = torch.from_numpy(array.reshape(shape))
    return tensor


def _compute_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _tensor_stats(array: np.ndarray) -> tuple[float | None, float | None, float | None, float | None]:
    if array.size == 0:
        return None, None, None, None

    return (
        float(array.min()),
        float(array.max()),
        float(array.mean()),
        float(array.std()),
    )


def _encode_array(array: np.ndarray) -> str:
    compressed = zlib.compress(array.tobytes())
    return base64.b64encode(compressed).decode("ascii")


def _load_tensor_reference(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Tensor reference '{path}' not found. Run scripts/export_tableformer_image_tensors.py first."
        )

    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _sorted_samples(samples: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(samples, key=lambda sample: (sample["image_name"], sample["table_index"]))


def export_neural_outputs(
    tensor_reference_path: Path,
    output_path: Path,
    *,
    repo_id: str = "ds4sd/docling-models",
    revision: str = "main",
    variant: str = "fast",
) -> None:
    tensor_reference = _load_tensor_reference(tensor_reference_path)
    tensor_samples = _sorted_samples(tensor_reference.get("samples", []))
    if not tensor_samples:
        raise ValueError("Tensor reference does not contain any samples.")

    docling = TableFormerDocling(repo_id=repo_id, revision=revision, variant=variant, device="cpu")
    predictor = docling.predictor
    model = predictor.get_model()
    config = docling.get_config()

    max_steps = int(config["predict"]["max_steps"])
    beam_size = int(config["predict"]["beam_size"])

    device = next(model.parameters()).device

    export_samples: list[dict[str, Any]] = []

    with torch.no_grad():
        for sample in tensor_samples:
            tensor = _decode_tensor(sample).to(device)
            seq, outputs_class, outputs_coord = model.predict(tensor, max_steps, beam_size)

            class_array = outputs_class.detach().cpu().numpy().astype(np.float32, copy=False)
            coord_array = outputs_coord.detach().cpu().numpy().astype(np.float32, copy=False)

            class_shape = list(outputs_class.shape)
            coord_shape = list(outputs_coord.shape)

            class_min, class_max, class_mean, class_std = _tensor_stats(class_array)
            coord_min, coord_max, coord_mean, coord_std = _tensor_stats(coord_array)

            class_sha = _compute_sha256(class_array.tobytes())
            coord_sha = _compute_sha256(coord_array.tobytes())

            tag_sequence = list(map(int, seq))
            tag_bytes = np.asarray(tag_sequence, dtype=np.int32).tobytes()
            tag_sha = _compute_sha256(tag_bytes)

            export_samples.append(
                {
                    "image_name": sample["image_name"],
                    "table_index": sample["table_index"],
                    "tensor_sha256": sample["tensor_sha256"],
                    "tag_sequence": tag_sequence,
                    "tag_sequence_sha256": tag_sha,
                    "class_shape": class_shape,
                    "class_zlib_base64": _encode_array(class_array),
                    "class_sha256": class_sha,
                    "class_min": class_min,
                    "class_max": class_max,
                    "class_mean": class_mean,
                    "class_std": class_std,
                    "coord_shape": coord_shape,
                    "coord_zlib_base64": _encode_array(coord_array),
                    "coord_sha256": coord_sha,
                    "coord_min": coord_min,
                    "coord_max": coord_max,
                    "coord_mean": coord_mean,
                    "coord_std": coord_std,
                }
            )

    payload = {
        "repo_id": repo_id,
        "revision": revision,
        "variant": variant,
        "max_steps": max_steps,
        "beam_size": beam_size,
        "samples": export_samples,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)
        fp.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export TableFormer neural inference reference data.")
    parser.add_argument(
        "--tensor-reference",
        type=Path,
        default=DEFAULT_TENSOR_REFERENCE,
        help="Path to tableformer_image_tensors_reference.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Destination JSON file for neural outputs",
    )
    parser.add_argument("--repo-id", type=str, default="ds4sd/docling-models")
    parser.add_argument("--revision", type=str, default="main")
    parser.add_argument("--variant", type=str, default="fast")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_neural_outputs(
        tensor_reference_path=args.tensor_reference,
        output_path=args.output,
        repo_id=args.repo_id,
        revision=args.revision,
        variant=args.variant,
    )


if __name__ == "__main__":
    main()
