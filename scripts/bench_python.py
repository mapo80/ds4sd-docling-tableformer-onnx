#!/usr/bin/env python3
"""Benchmark ONNX Runtime inference for a given model and dataset."""
import argparse
import time
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

import onnxruntime as ort
from transformers import AutoImageProcessor

from pipeline_utils import iter_images, ensure_dir, seed_everything, print_env_info, summarize_times


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark ORT model")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    seed_everything()
    print_env_info()

    processor = AutoImageProcessor.from_pretrained("ds4sd/docling-layout-heron", cache_dir="models/hf-cache")
    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])

    out_dir = Path(args.out)
    ensure_dir(out_dir)
    timings = []

    first = True
    for path, img in iter_images(args.dataset):
        inputs = processor(images=img, return_tensors="np")
        if first:
            sess.run(None, {"pixel_values": inputs["pixel_values"]})
            first = False
        t0 = time.time()
        sess.run(None, {"pixel_values": inputs["pixel_values"]})
        dt = (time.time() - t0) * 1000
        timings.append(dt)
        (out_dir / "timings.csv").open("a").write(f"{path.name},{dt:.3f}\n")
    stats = summarize_times(timings)
    print(f"mean {stats['mean']:.2f} ms  p95 {stats['p95']:.2f} ms")


if __name__ == "__main__":
    main()
