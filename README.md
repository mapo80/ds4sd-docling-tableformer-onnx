# ds4sd-docling-layout-heron-onnx

Pipeline to compare the `ds4sd/docling-layout-heron` object detection model in three variants:

1. **Baseline HuggingFace model**
2. **ONNX converted model**
3. **ONNX optimized model**

The repository provides scripts to convert the model to ONNX, apply ONNX Runtime graph optimizations, run inference on example images and compare KPIs (IoU, timings, model size).

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Usage

The workflow is driven by the `Makefile`:

```bash
make convert    # export to ONNX
make optimize   # graph optimize and optional quantization
make infer-all  # run inference for baseline and ONNX variants
make compare    # compute KPIs and generate reports
```

Results are written to the `results/` directory. Each run creates a timestamped folder containing reports and links to the outputs of each variant.

> The large model and ONNX files are stored in `models/` and ignored by git.
