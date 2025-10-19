#!/usr/bin/env python3
"""Export TableFormer cell matching reference data."""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import zlib
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from docling_ibm_models.tableformer.data_management.tf_cell_matcher import CellMatcher

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SEQUENCE_REFERENCE = REPO_ROOT / "results" / "tableformer_sequence_decoding_reference.json"
DEFAULT_NEURAL_REFERENCE = REPO_ROOT / "results" / "tableformer_neural_outputs_reference.json"
DEFAULT_TABLE_CROP_REFERENCE = REPO_ROOT / "results" / "tableformer_table_crops_reference.json"
DEFAULT_PAGE_INPUT_REFERENCE = REPO_ROOT / "results" / "tableformer_page_input_reference.json"
DEFAULT_CONFIG_REFERENCE = REPO_ROOT / "results" / "tableformer_config_fast_hash.json"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "results" / "tableformer_cell_matching_reference.json"


def _decode_array(encoded: str, shape: Iterable[int]) -> np.ndarray:
    compressed = base64.b64decode(encoded)
    decompressed = zlib.decompress(compressed)
    array = np.frombuffer(decompressed, dtype=np.float32)
    return array.reshape(tuple(shape))


def _encode_array(array: np.ndarray) -> str:
    compressed = zlib.compress(array.astype(np.float32, copy=False).tobytes())
    return base64.b64encode(compressed).decode("ascii")


def _compute_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def _sorted_samples(samples: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(samples, key=lambda sample: (sample["image_name"], sample["table_index"]))


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _build_sample_index(samples: Iterable[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    index: dict[tuple[str, int], dict[str, Any]] = {}
    for sample in samples:
        key = (sample["image_name"], int(sample["table_index"]))
        index[key] = sample
    return index


def export_cell_matching(
    sequence_reference_path: Path,
    neural_reference_path: Path,
    table_crop_reference_path: Path,
    page_input_reference_path: Path,
    config_reference_path: Path,
    output_path: Path,
    *,
    repo_id: str = "ds4sd/docling-models",
    revision: str = "main",
    variant: str = "fast",
) -> None:
    sequence_reference = _load_json(sequence_reference_path)
    neural_reference = _load_json(neural_reference_path)
    table_crop_reference = _load_json(table_crop_reference_path)
    page_input_reference = _load_json(page_input_reference_path)
    config_reference = _load_json(config_reference_path)

    sequence_index = _build_sample_index(_sorted_samples(sequence_reference.get("samples", [])))
    neural_index = _build_sample_index(_sorted_samples(neural_reference.get("samples", [])))

    table_crops = {
        sample["image_name"]: sample for sample in table_crop_reference.get("samples", [])
    }
    page_inputs = {
        sample["image_name"]: sample for sample in page_input_reference.get("samples", [])
    }

    if not sequence_index:
        raise ValueError("Sequence reference does not contain any samples.")

    canonical_config = config_reference.get("canonical_json")
    if not canonical_config:
        raise ValueError("Configuration reference does not contain 'canonical_json'.")

    config = json.loads(canonical_config)
    cell_matcher = CellMatcher(config)

    export_samples: list[dict[str, Any]] = []

    for key, sequence_sample in sequence_index.items():
        image_name, table_index = key

        if key not in neural_index:
            raise KeyError(f"Neural reference is missing sample for {image_name}#{table_index}.")

        neural_sample = neural_index[key]
        crop_sample = table_crops.get(image_name)
        if crop_sample is None:
            raise KeyError(f"Table crop reference missing entry for image '{image_name}'.")

        crop_entries = {
            entry["table_index"]: entry for entry in crop_sample.get("table_crops", [])
        }
        if table_index not in crop_entries:
            raise KeyError(f"Crop reference missing table index {table_index} for '{image_name}'.")

        page_sample = page_inputs.get(image_name)
        if page_sample is None:
            raise KeyError(f"Page input reference missing sample for '{image_name}'.")

        coord_shape = list(map(int, neural_sample["coord_shape"]))
        coord_array = _decode_array(neural_sample["coord_zlib_base64"], coord_shape)

        class_shape = list(map(int, neural_sample["class_shape"]))
        class_array = _decode_array(neural_sample["class_zlib_base64"], class_shape)
        class_predictions = class_array.argmax(axis=1).astype(np.int32, copy=False).tolist()

        final_shape = list(map(int, sequence_sample["final_bbox_shape"]))
        final_bboxes = _decode_array(sequence_sample["final_bbox_zlib_base64"], final_shape)

        prediction = {
            "bboxes": final_bboxes.tolist(),
            "classes": class_predictions,
            "html_seq": sequence_sample["html_sequence"],
            "rs_seq": sequence_sample["rs_sequence"],
        }

        table_bbox = crop_entries[table_index]["original_bbox"]

        iocr_page = {
            "tokens": copy.deepcopy(page_sample.get("tokens", [])),
            "height": page_sample.get("height"),
            "width": page_sample.get("width"),
        }

        matching_details = cell_matcher.match_cells(iocr_page, table_bbox, prediction)

        prediction_bboxes_page = np.asarray(matching_details["prediction_bboxes_page"], dtype=np.float32)
        prediction_shape = list(prediction_bboxes_page.shape)
        prediction_sha = _compute_sha256(prediction_bboxes_page.tobytes())

        table_cells = matching_details.get("table_cells", [])
        matches = matching_details.get("matches", {})
        pdf_cells = matching_details.get("pdf_cells", [])

        export_samples.append(
            {
                "image_name": image_name,
                "table_index": table_index,
                "tensor_sha256": sequence_sample["tensor_sha256"],
                "tag_sequence_sha256": sequence_sample["tag_sequence_sha256"],
                "table_bbox": table_bbox,
                "page_width": page_sample.get("width"),
                "page_height": page_sample.get("height"),
                "iou_threshold": matching_details.get("iou_threshold"),
                "prediction_bbox_shape": prediction_shape,
                "prediction_bbox_zlib_base64": _encode_array(prediction_bboxes_page),
                "prediction_bbox_sha256": prediction_sha,
                "table_cells": table_cells,
                "table_cells_sha256": _compute_sha256(_canonical_json(table_cells).encode("utf-8")),
                "matches": matches,
                "matches_sha256": _compute_sha256(_canonical_json(matches).encode("utf-8")),
                "pdf_cells": pdf_cells,
                "pdf_cells_sha256": _compute_sha256(_canonical_json(pdf_cells).encode("utf-8")),
            }
        )

    export_samples.sort(key=lambda sample: (sample["image_name"], sample["table_index"]))

    output = {
        "repo_id": repo_id,
        "revision": revision,
        "variant": variant,
        "samples": export_samples,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(output, fp, indent=2)

    print(f"Saved {len(export_samples)} cell-matching samples to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export TableFormer cell matching reference data.")
    parser.add_argument("--sequence-reference", type=Path, default=DEFAULT_SEQUENCE_REFERENCE)
    parser.add_argument("--neural-reference", type=Path, default=DEFAULT_NEURAL_REFERENCE)
    parser.add_argument("--table-crop-reference", type=Path, default=DEFAULT_TABLE_CROP_REFERENCE)
    parser.add_argument("--page-input-reference", type=Path, default=DEFAULT_PAGE_INPUT_REFERENCE)
    parser.add_argument("--config-reference", type=Path, default=DEFAULT_CONFIG_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--repo-id", default="ds4sd/docling-models")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--variant", default="fast")
    args = parser.parse_args()

    export_cell_matching(
        sequence_reference_path=args.sequence_reference,
        neural_reference_path=args.neural_reference,
        table_crop_reference_path=args.table_crop_reference,
        page_input_reference_path=args.page_input_reference,
        config_reference_path=args.config_reference,
        output_path=args.output,
        repo_id=args.repo_id,
        revision=args.revision,
        variant=args.variant,
    )


if __name__ == "__main__":
    main()
