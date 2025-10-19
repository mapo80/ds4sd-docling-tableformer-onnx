import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CELL_REFERENCE = REPO_ROOT / "results" / "tableformer_cell_matching_reference.json"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "results" / "tableformer_post_processing_reference.json"


def _canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_json(data: Any) -> str:
    return _sha256_bytes(_canonical_json(data).encode("utf-8"))


def _median(values: Iterable[float]) -> float:
    array = list(values)
    if not array:
        return 0.0
    sorted_values = sorted(array)
    mid = len(sorted_values) // 2
    if len(sorted_values) % 2 == 0:
        return (sorted_values[mid - 1] + sorted_values[mid]) / 2.0
    return sorted_values[mid]


def _bbox_area(bbox: list[float]) -> float:
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    if width <= 0 or height <= 0:
        return 0.0
    return width * height


def _bbox_intersection(a: list[float], b: list[float]) -> list[float] | None:
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    if right <= left or bottom <= top:
        return None
    return [left, top, right, bottom]


def _clear_pdf_cells(pdf_cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [cell for cell in pdf_cells if cell.get("text")]


def _compute_intersection_matches(
    table_cells: list[dict[str, Any]],
    pdf_cells: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    matches: dict[str, list[dict[str, Any]]] = {}
    if not table_cells or not pdf_cells:
        return matches

    pdf_areas = {cell["id"]: _bbox_area(cell["bbox"]) for cell in pdf_cells}

    for cell in table_cells:
        bbox = cell["bbox"]
        for pdf_cell in pdf_cells:
            intersection = _bbox_intersection(bbox, pdf_cell["bbox"])
            if intersection is None:
                continue
            area = _bbox_area(intersection)
            pdf_area = pdf_areas.get(pdf_cell["id"], 0.0)
            if pdf_area <= 0:
                continue
            score = area / pdf_area
            if score <= 0:
                continue
            matches.setdefault(pdf_cell["id"], []).append(
                {"table_cell_id": cell["cell_id"], "iopdf": score}
            )

    for key, value in matches.items():
        value.sort(key=lambda item: item.get("iopdf", 0.0), reverse=True)
    return matches


def _get_table_dimension(
    table_cells: list[dict[str, Any]]
) -> tuple[int, int, int]:
    columns = 1
    rows = 1
    max_cell_id = 0
    for cell in table_cells:
        columns = max(columns, int(cell["column_id"]))
        rows = max(rows, int(cell["row_id"]))
        max_cell_id = max(max_cell_id, int(cell["cell_id"]))
    return columns + 1, rows + 1, max_cell_id


def _get_good_bad_cells_in_column(
    table_cells: list[dict[str, Any]],
    column: int,
    matches: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    good: list[dict[str, Any]] = []
    bad: list[dict[str, Any]] = []
    for cell in table_cells:
        if int(cell["column_id"]) != column:
            continue
        if int(cell.get("cell_class", 2)) <= 1:
            bad.append(dict(cell))
            continue
        matched = any(
            any(match["table_cell_id"] == cell["cell_id"] for match in match_list)
            for match_list in matches.values()
        )
        (good if matched else bad).append(dict(cell))
    return good, bad


def _alignment_range(cells: list[dict[str, Any]], selector) -> float:
    if not cells:
        return float("inf")
    values = [selector(cell["bbox"]) for cell in cells]
    return max(values) - min(values)


def _find_alignment_in_column(cells: list[dict[str, Any]]) -> str:
    if not cells:
        return "left"
    left = _alignment_range(cells, lambda bbox: bbox[0])
    center = _alignment_range(cells, lambda bbox: (bbox[0] + bbox[2]) / 2.0)
    right = _alignment_range(cells, lambda bbox: bbox[2])
    alignment = "left"
    min_range = left
    if center < min_range:
        alignment = "center"
        min_range = center
    if right < min_range:
        alignment = "right"
    return alignment


def _median_position_and_size(
    cells: list[dict[str, Any]], alignment: str
) -> tuple[float, float, float, float]:
    if not cells:
        return 0.0, 0.0, 0.0, 0.0
    positions = []
    verticals = []
    widths = []
    heights = []
    for cell in cells:
        bbox = cell["bbox"]
        if alignment == "center":
            positions.append((bbox[0] + bbox[2]) / 2.0)
        elif alignment == "right":
            positions.append(bbox[2])
        else:
            positions.append(bbox[0])
        verticals.append((bbox[1] + bbox[3]) / 2.0)
        widths.append(abs(bbox[2] - bbox[0]))
        heights.append(abs(bbox[3] - bbox[1]))
    return _median(positions), _median(verticals), _median(widths), _median(heights)


def _move_cells_to_alignment(
    cells: list[dict[str, Any]],
    position: float,
    alignment: str,
    width: float,
    height: float,
) -> list[dict[str, Any]]:
    adjusted: list[dict[str, Any]] = []
    for cell in cells:
        bbox = cell["bbox"]
        center_y = (bbox[1] + bbox[3]) / 2.0
        if alignment == "center":
            left = position - width / 2.0
            right = position + width / 2.0
        elif alignment == "right":
            right = position
            left = right - width
        else:
            left = position
            right = left + width
        top = center_y - height / 2.0
        bottom = center_y + height / 2.0
        new_cell = dict(cell)
        new_cell["bbox"] = [left, top, right, bottom]
        adjusted.append(new_cell)
    return adjusted


def _deduplicate_columns(
    tab_columns: int,
    table_cells: list[dict[str, Any]],
    matches: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], int]:
    pdf_cells_per_column: list[list[int]] = []
    scores_per_column: list[float] = []

    for column in range(tab_columns):
        cell_ids = {
            int(cell["cell_id"])
            for cell in table_cells
            if int(cell["column_id"]) == column
        }
        pdf_ids: set[int] = set()
        score = 0.0
        for pdf_id, match_list in matches.items():
            for match in match_list:
                if int(match["table_cell_id"]) in cell_ids:
                    score += float(match.get("iopdf", 0.0))
                    try:
                        pdf_ids.add(int(pdf_id))
                    except ValueError:
                        continue
        pdf_cells_per_column.append(list(pdf_ids))
        scores_per_column.append(score)

    columns_to_remove: set[int] = set()
    for column in range(tab_columns - 1):
        current = pdf_cells_per_column[column]
        if not current:
            continue
        nxt = pdf_cells_per_column[column + 1]
        intersection = len(set(current).intersection(nxt))
        overlap = intersection / len(current)
        if overlap <= 0.6:
            continue
        if scores_per_column[column] >= scores_per_column[column + 1]:
            columns_to_remove.add(column + 1)
        else:
            columns_to_remove.add(column)

    filtered_cells = []
    removed_ids: set[int] = set()
    for cell in table_cells:
        if int(cell["column_id"]) in columns_to_remove:
            removed_ids.add(int(cell["cell_id"]))
            continue
        filtered_cells.append(dict(cell))

    filtered_matches: dict[str, list[dict[str, Any]]] = {}
    for pdf_id, match_list in matches.items():
        filtered = [
            dict(match)
            for match in match_list
            if int(match["table_cell_id"]) not in removed_ids
        ]
        if filtered:
            filtered_matches[pdf_id] = filtered

    return filtered_cells, filtered_matches, tab_columns - len(columns_to_remove)


def _do_final_assignment(matches: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for pdf_id, match_list in matches.items():
        if not match_list:
            continue
        best = max(match_list, key=lambda item: float(item.get("iopdf", 0.0)))
        result[pdf_id] = [dict(best)]
    return result


def _align_table_cells_to_pdf(
    table_cells: list[dict[str, Any]],
    pdf_cells: list[dict[str, Any]],
    matches: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    pdf_lookup = {cell["id"]: cell["bbox"] for cell in pdf_cells}
    aligned = []
    for cell in table_cells:
        new_cell = dict(cell)
        for pdf_id, match_list in matches.items():
            if any(match["table_cell_id"] == cell["cell_id"] for match in match_list):
                if pdf_id in pdf_lookup:
                    new_cell["bbox"] = list(pdf_lookup[pdf_id])
                break
        aligned.append(new_cell)
    return aligned


def _assign_orphan_pdf_cells(
    table_cells: list[dict[str, Any]],
    pdf_cells: list[dict[str, Any]],
    matches: dict[str, list[dict[str, Any]]],
    max_cell_id: int,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], int]:
    result_matches = {pdf_id: [dict(match) for match in match_list] for pdf_id, match_list in matches.items()}
    result_cells = [dict(cell) for cell in table_cells]
    matched_ids = set(matches.keys())
    for pdf_cell in pdf_cells:
        if pdf_cell["id"] in matched_ids:
            continue
        max_cell_id += 1
        new_cell = {
            "cell_id": max_cell_id,
            "row_id": 0,
            "column_id": 0,
            "bbox": list(pdf_cell["bbox"]),
            "label": "body",
            "cell_class": 2,
            "multicol_tag": "",
        }
        result_cells.append(new_cell)
        result_matches[pdf_cell["id"]] = [
            {"table_cell_id": max_cell_id, "iopdf": 1.0}
        ]
    return result_matches, result_cells, max_cell_id


def _ensure_non_overlapping(table_cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(cell) for cell in table_cells]


def process_matching_details(details: dict[str, Any]) -> dict[str, Any]:
    table_cells = [dict(cell) for cell in details.get("table_cells", [])]
    pdf_cells = [dict(cell) for cell in details.get("pdf_cells", [])]
    matches = {
        pdf_id: [dict(match) for match in match_list]
        for pdf_id, match_list in details.get("matches", {}).items()
    }

    pdf_cells = _clear_pdf_cells(pdf_cells)
    if not matches and pdf_cells:
        matches = _compute_intersection_matches(table_cells, pdf_cells)

    columns, rows, max_cell_id = _get_table_dimension(table_cells)
    adjusted_cells: list[dict[str, Any]] = []
    for column in range(columns):
        good_cells, bad_cells = _get_good_bad_cells_in_column(table_cells, column, matches)
        alignment = _find_alignment_in_column(good_cells)
        position, _, width, height = _median_position_and_size(good_cells, alignment)
        adjusted_bad_cells = _move_cells_to_alignment(bad_cells, position, alignment, width, height)
        adjusted_cells.extend(good_cells)
        adjusted_cells.extend(adjusted_bad_cells)

    adjusted_cells.sort(key=lambda cell: int(cell["cell_id"]))
    matches = _compute_intersection_matches(adjusted_cells, pdf_cells)

    dedup_cells, dedup_matches, _ = _deduplicate_columns(columns, adjusted_cells, matches)
    final_matches = _do_final_assignment(dedup_matches)
    aligned_cells = _align_table_cells_to_pdf(dedup_cells, pdf_cells, final_matches)
    post_matches, post_cells, max_cell_id = _assign_orphan_pdf_cells(
        aligned_cells, pdf_cells, final_matches, max_cell_id
    )

    post_cells = _ensure_non_overlapping(post_cells)

    return {
        "table_cells": post_cells,
        "pdf_cells": pdf_cells,
        "matches": post_matches,
        "max_cell_id": max_cell_id,
    }


def _synthesize_pdf_cells(table_cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pdf_cells = []
    for index, cell in enumerate(table_cells):
        pdf_cells.append(
            {
                "id": str(index),
                "bbox": list(cell["bbox"]),
                "text": f"cell_{cell['cell_id']}",
            }
        )
    return pdf_cells


def _generate_doc_output(
    table_cells: list[dict[str, Any]],
    matches: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    table_lookup = {cell["cell_id"]: cell for cell in table_cells}
    doc_output: list[dict[str, Any]] = []
    for pdf_id, match_list in matches.items():
        if not match_list:
            continue
        table_cell_id = match_list[0]["table_cell_id"]
        table_cell = table_lookup.get(table_cell_id)
        if table_cell is None:
            continue
        bbox = table_cell.get("bbox", [0.0, 0.0, 0.0, 0.0])
        colspan = int(table_cell.get("colspan_val", table_cell.get("colspan", 1)) or 1)
        rowspan = int(table_cell.get("rowspan_val", table_cell.get("rowspan", 1)) or 1)
        label = table_cell.get("label", "")
        doc_output.append(
            {
                "pdf_cell_id": pdf_id,
                "table_cell_id": table_cell_id,
                "bbox": bbox,
                "col_span": colspan,
                "row_span": rowspan,
                "column_header": label == "ched",
                "row_header": label == "rhed",
                "row_section": label == "srow",
            }
        )
    doc_output.sort(key=lambda item: int(item["pdf_cell_id"]))
    return doc_output


def export_post_processing(
    cell_reference_path: Path,
    output_path: Path,
) -> None:
    with cell_reference_path.open("r", encoding="utf-8") as fp:
        cell_reference = json.load(fp)

    export_samples: list[dict[str, Any]] = []

    for sample in cell_reference.get("samples", []):
        table_cells = [dict(cell) for cell in sample.get("table_cells", [])]
        pdf_cells = [dict(cell) for cell in sample.get("pdf_cells", [])]
        matches = {
            pdf_id: [dict(match) for match in match_list]
            for pdf_id, match_list in sample.get("matches", {}).items()
        }

        if not pdf_cells:
            pdf_cells = _synthesize_pdf_cells(table_cells)
            matches = {
                str(index): [{"table_cell_id": cell["cell_id"], "iopdf": 1.0}]
                for index, cell in enumerate(table_cells)
            }

        input_details = {
            "table_cells": table_cells,
            "pdf_cells": pdf_cells,
            "matches": matches,
        }

        processed = process_matching_details(input_details)
        doc_output = _generate_doc_output(processed["table_cells"], processed["matches"])

        export_samples.append(
            {
                "image_name": sample.get("image_name"),
                "table_index": sample.get("table_index"),
                "input": input_details,
                "output": processed,
                "doc_output": doc_output,
                "input_table_cells_sha256": _sha256_json(input_details["table_cells"]),
                "input_matches_sha256": _sha256_json(input_details["matches"]),
                "output_table_cells_sha256": _sha256_json(processed["table_cells"]),
                "output_matches_sha256": _sha256_json(processed["matches"]),
                "doc_output_sha256": _sha256_json(doc_output),
            }
        )

    export_samples.sort(key=lambda item: (item["image_name"], int(item["table_index"])) )

    output = {
        "samples": export_samples,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(output, fp, indent=2)

    print(f"Saved {len(export_samples)} post-processing samples to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export TableFormer post-processing reference data.")
    parser.add_argument("--cell-reference", type=Path, default=DEFAULT_CELL_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    export_post_processing(
        cell_reference_path=args.cell_reference,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
