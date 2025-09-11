#!/usr/bin/env python3
"""Convert ONNX models to ORT format and report size and load time."""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import onnxruntime as ort

sys.path.append(str(Path(__file__).resolve().parent.parent))
from pipeline_utils import seed_everything, print_env_info


def convert(src: Path, out_file: Path) -> None:
    out_dir = out_file.parent
    cmd = [sys.executable, "-m", "onnxruntime.tools.convert_onnx_models_to_ort", src.as_posix(), "--output", out_dir.as_posix()]
    subprocess.run(cmd, check=True)
    ort_path = out_dir / (src.stem + ".ort")
    ort_path.rename(out_file)  # rename to desired path
    size_mb = out_file.stat().st_size / 1024 ** 2
    t0 = time.time()
    ort.InferenceSession(out_file.as_posix(), providers=["CPUExecutionProvider"])
    load_ms = (time.time() - t0) * 1000
    print(f"Converted {src} -> {out_file} ({size_mb:.1f} MB, load {load_ms:.1f} ms)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert ONNX to ORT format")
    parser.add_argument("--optimized", default="models/heron-optimized.onnx")
    parser.add_argument("--quantized", default="models/heron-int8-dynamic.onnx")
    args = parser.parse_args()

    seed_everything()
    print_env_info()

    convert(Path(args.optimized), Path("models/heron-optimized.ort"))
    convert(Path(args.quantized), Path("models/heron-int8-dynamic.ort"))


if __name__ == "__main__":
    main()
