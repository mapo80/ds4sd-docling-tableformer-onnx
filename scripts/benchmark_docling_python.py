"""Benchmark Docling's TableFormer pipeline on the FinTabNet dataset."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
TABLEFORMER_PATH = REPO_ROOT / "tableformer-docling"
if str(TABLEFORMER_PATH) not in sys.path:
    sys.path.append(str(TABLEFORMER_PATH))

from tableformer_docling import TableFormerDocling  # noqa: E402


@dataclass
class BenchmarkResult:
    predictions: Dict[str, Dict[str, Any]]
    timings_ms: Dict[str, float]

    @property
    def total_time_ms(self) -> float:
        return float(sum(self.timings_ms.values()))

    @property
    def average_time_ms(self) -> float:
        if not self.timings_ms:
            return 0.0
        return self.total_time_ms / len(self.timings_ms)

    def to_serializable(self) -> Dict[str, Any]:
        return {
            "predictions": self.predictions,
            "timings_ms": self.timings_ms,
            "summary": {
                "num_documents": len(self.timings_ms),
                "total_ms": self.total_time_ms,
                "average_ms": self.average_time_ms,
            },
        }


def load_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def run_benchmark(
    dataset_dir: Path,
    artifacts_dir: Path,
    *,
    device: str,
    num_threads: int,
) -> BenchmarkResult:
    runner = TableFormerDocling(
        artifacts_dir=artifacts_dir,
        device=device,
        num_threads=num_threads,
    )

    predictions: Dict[str, Dict[str, Any]] = {}
    timings_ms: Dict[str, float] = {}

    image_paths = sorted(dataset_dir.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No PNG files found in {dataset_dir}")

    for image_path in image_paths:
        image = load_image(image_path)
        start = time.perf_counter()
        tables = runner.predict_page(image)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        predictions[image_path.name] = {
            "num_tables": len(tables),
            "tables": tables,
        }
        timings_ms[image_path.name] = elapsed_ms

    return BenchmarkResult(predictions=predictions, timings_ms=timings_ms)


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=REPO_ROOT / "dataset" / "FinTabNet" / "benchmark",
        help="Directory containing FinTabNet benchmark PNG files",
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=TABLEFORMER_PATH / "artifacts",
        help="Directory containing Docling artifacts",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results" / "tableformer_docling_fintabnet_python.json",
        help="Path to save the benchmark report",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=REPO_ROOT / "results" / "tableformer_docling_fintabnet.json",
        help="Canonical Docling multi-table output used for parity checks",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Inference device (default: cpu)",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=4,
        help="Number of intra-op threads for Docling",
    )
    parser.add_argument(
        "--skip-reference-check",
        action="store_true",
        help="Skip strict comparison against the canonical reference JSON",
    )
    return parser.parse_args(list(argv))


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)

    benchmark = run_benchmark(
        dataset_dir=args.dataset,
        artifacts_dir=args.artifacts,
        device=args.device,
        num_threads=args.num_threads,
    )

    reference_data = None
    if args.reference.exists():
        reference_data = json.loads(args.reference.read_text(encoding="utf-8"))

    if reference_data is not None:
        actual_canonical = canonical_json(benchmark.predictions)
        reference_canonical = canonical_json(reference_data)
        if actual_canonical != reference_canonical and not args.skip_reference_check:
            raise SystemExit(
                "Python benchmark output diverges from canonical reference. "
                "Use --skip-reference-check to collect timings anyway."
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(benchmark.to_serializable(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Benchmark report saved to {args.output}")
    for name, elapsed in benchmark.timings_ms.items():
        print(f"{name}: {elapsed:.2f} ms")
    print(
        "Average: "
        f"{benchmark.average_time_ms:.2f} ms over {len(benchmark.timings_ms)} documents"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
