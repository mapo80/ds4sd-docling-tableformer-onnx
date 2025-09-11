.PHONY: convert optimize infer-baseline infer-onnx-converted infer-onnx-optimized infer-all compare

convert:
	python convert_to_onnx.py

optimize: convert
	python optimize_onnx.py

infer-baseline:
	python infer_baseline.py

infer-onnx-converted: convert
	python infer_onnx.py --model models/heron-converted.onnx --variant onnx-converted

infer-onnx-optimized: optimize
	python infer_onnx.py --model models/heron-optimized.onnx --variant onnx-optimized

infer-all: infer-baseline infer-onnx-converted infer-onnx-optimized

compare: infer-all
	python compare_kpi.py
