#!/usr/bin/env python3
"""Export TableFormer sequence decoding reference outputs."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import zlib
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from tableformer_docling.predictor import TableFormerDocling
from docling_ibm_models.tableformer.data_management.tf_predictor import TFPredictor
from docling_ibm_models.tableformer.otsl import otsl_to_html

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_NEURAL_REFERENCE = REPO_ROOT / "results" / "tableformer_neural_outputs_reference.json"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "results" / "tableformer_sequence_decoding_reference.json"


def _decode_array(encoded: str, shape: Iterable[int]) -> np.ndarray:
    compressed = base64.b64decode(encoded)
    decompressed = zlib.decompress(compressed)
    array = np.frombuffer(decompressed, dtype=np.float32)
    return array.reshape(tuple(shape))


def _encode_array(array: np.ndarray) -> str:
    compressed = zlib.compress(array.tobytes())
    return base64.b64encode(compressed).decode("ascii")


def _compute_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hash_sequence(values: Iterable[str]) -> str:
    canonical = json.dumps(list(values), separators=(",", ":"))
    return _compute_sha256(canonical.encode("utf-8"))


def _tensor_stats(array: np.ndarray) -> tuple[float | None, float | None, float | None, float | None]:
    if array.size == 0:
        return None, None, None, None

    return (
        float(array.min()),
        float(array.max()),
        float(array.mean()),
        float(array.std()),
    )


def _sorted_samples(samples: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(samples, key=lambda sample: (sample["image_name"], sample["table_index"]))


def _load_neural_reference(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Neural reference '{path}' not found. Run scripts/export_tableformer_neural_outputs.py first."
        )
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _cxcywh_to_xyxy(array: np.ndarray) -> np.ndarray:
    if array.size == 0:
        return np.zeros((0, 4), dtype=np.float32)

    if array.shape[-1] != 4:
        raise ValueError(
            f"Expected coordinates with trailing dimension of 4 but received shape {array.shape}."
        )

    x_c = array[..., 0]
    y_c = array[..., 1]
    w = array[..., 2]
    h = array[..., 3]

    x0 = x_c - 0.5 * w
    y0 = y_c - 0.5 * h
    x1 = x_c + 0.5 * w
    y1 = y_c + 0.5 * h

    return np.stack([x0, y0, x1, y1], axis=-1).astype(np.float32, copy=False)


def export_sequence_decoding(
    neural_reference_path: Path,
    output_path: Path,
    *,
    repo_id: str = "ds4sd/docling-models",
    revision: str = "main",
    variant: str = "fast",
) -> None:
    neural_reference = _load_neural_reference(neural_reference_path)
    neural_samples = _sorted_samples(neural_reference.get("samples", []))
    if not neural_samples:
        raise ValueError("Neural reference does not contain any samples.")

    docling = TableFormerDocling(repo_id=repo_id, revision=revision, variant=variant, device="cpu")
    predictor: TFPredictor = docling.predictor

    export_samples: list[dict[str, Any]] = []

    for sample in neural_samples:
        tag_sequence = list(map(int, sample["tag_sequence"]))
        coord_shape = list(map(int, sample["coord_shape"]))
        coord_array = _decode_array(sample["coord_zlib_base64"], coord_shape)

        raw_bboxes = _cxcywh_to_xyxy(coord_array)

        rs_sequence = predictor._get_html_tags(tag_sequence)  # pylint: disable=protected-access
        html_sequence = otsl_to_html(rs_sequence, False)

        prediction: dict[str, Any] = {
            "bboxes": raw_bboxes.tolist(),
            "html_seq": html_sequence,
        }
        sync, corrected_bboxes = predictor._check_bbox_sync(prediction)  # pylint: disable=protected-access

        if sync:
            final_bboxes = raw_bboxes
        else:
            final_bboxes = np.asarray(corrected_bboxes, dtype=np.float32)

        raw_sha = _compute_sha256(raw_bboxes.tobytes())
        raw_min, raw_max, raw_mean, raw_std = _tensor_stats(raw_bboxes)

        final_sha = _compute_sha256(final_bboxes.tobytes())
        final_min, final_max, final_mean, final_std = _tensor_stats(final_bboxes)

        export_samples.append(
            {
                "image_name": sample["image_name"],
                "table_index": int(sample["table_index"]),
                "tensor_sha256": sample["tensor_sha256"],
                "tag_sequence_sha256": sample["tag_sequence_sha256"],
                "tag_sequence": tag_sequence,
                "rs_sequence": rs_sequence,
                "rs_sequence_sha256": _hash_sequence(rs_sequence),
                "html_sequence": html_sequence,
                "html_sequence_sha256": _hash_sequence(html_sequence),
                "raw_bbox_shape": list(raw_bboxes.shape),
                "raw_bbox_zlib_base64": _encode_array(raw_bboxes),
                "raw_bbox_sha256": raw_sha,
                "raw_bbox_min": raw_min,
                "raw_bbox_max": raw_max,
                "raw_bbox_mean": raw_mean,
                "raw_bbox_std": raw_std,
                "final_bbox_shape": list(final_bboxes.shape),
                "final_bbox_zlib_base64": _encode_array(final_bboxes),
                "final_bbox_sha256": final_sha,
                "final_bbox_min": final_min,
                "final_bbox_max": final_max,
                "final_bbox_mean": final_mean,
                "final_bbox_std": final_std,
                "bbox_sync": bool(sync),
            }
        )

    payload = {
        "repo_id": repo_id,
        "revision": revision,
        "variant": variant,
        "samples": export_samples,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)
        fp.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export TableFormer sequence decoding reference data.")
    parser.add_argument(
        "--neural-reference",
        type=Path,
        default=DEFAULT_NEURAL_REFERENCE,
        help="Path to tableformer_neural_outputs_reference.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Destination JSON file for sequence decoding data",
    )
    parser.add_argument("--repo-id", type=str, default="ds4sd/docling-models")
    parser.add_argument("--revision", type=str, default="main")
    parser.add_argument("--variant", type=str, default="fast")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_sequence_decoding(
        neural_reference_path=args.neural_reference,
        output_path=args.output,
        repo_id=args.repo_id,
        revision=args.revision,
        variant=args.variant,
    )


if __name__ == "__main__":
    main()
