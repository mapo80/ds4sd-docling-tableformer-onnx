# ds4sd-docling-layout-heron ONNX

Utilities to export the [`ds4sd/docling-layout-heron`](https://huggingface.co/ds4sd/docling-layout-heron) object detection model to ONNX and compare it against the original PyTorch implementation.

## Usage

1. **Convert the model**
   ```bash
   python convert_to_onnx.py
   ```
   This downloads the model and produces four variants:
   - `docling_layout_heron.onnx` – simplified full precision reference
   - `docling_layout_heron_quant.onnx` – MatMul weights dynamically quantized to int8
   - `docling_layout_heron_fp16.onnx` – most weights converted to float16 while keeping MatMul ops in float32
   - `docling_layout_heron_static.onnx` – statically quantized int8 model using random calibration data

2. **Run a quick comparison**
   ```bash
    python compare_models.py
    ```
    The script generates a random 640x640 image, runs it through PyTorch and the ONNX models, and prints the maximum absolute differences together with average inference times.

   A sample CPU run produced the following results on random 640×640 input:

   | model | size (MB) | time (ms) | max rel diff logits | max rel diff boxes |
   |-------|---------:|----------:|--------------------:|-------------------:|
   | PyTorch | n/a | 1445.89 | – | – |
   | ONNX | 164.2 | 807.15 | – | – |
   | Dynamic Quantized ONNX | 141.8 | 753.66 | 36.93% | 94.77% |
   | Float16 ONNX | 97.4 | 1070.32 | 33.95% | 96.04% |
   | Static Quantized ONNX | 42.8 | 659.89 | 72.71% | 96.09% |

   Static quantization yields the smallest file but shows large divergence on the random test image.

> The generated `.onnx` files are ignored by git and are not part of the repository.
