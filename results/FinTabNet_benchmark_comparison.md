# FinTabNet Benchmark Comparison

This report captures the outputs and runtimes observed on the `dataset/FinTabNet/benchmark` subset for both the Python Docling
pipeline and the TorchSharp port.

## Summary

- Canonical Docling reference: `results/tableformer_docling_fintabnet.json`.
- Python benchmark output: `results/tableformer_docling_fintabnet_python.json` (generated with `--skip-reference-check`).
- .NET benchmark output: `results/tableformer_docling_fintabnet_dotnet.json` (generated with `--skip-reference-check`).
- **Parity status:** the serialized predictions differ between Python and .NET for both pages; verification cannot advance until
  the Python artifacts are realigned with the canonical reference.

## Per-document timings

| Document | Python (ms) | .NET (ms) | Δ (.NET − Python) ms |
| --- | ---: | ---: | ---: |
| HAL.2004.page_82.pdf_125315.png | 4164.38 | 5635.75 | 1471.38 |
| HAL.2004.page_82.pdf_125317.png | 2564.04 | 4612.46 | 2048.42 |
| **TOTAL** | **6728.41** | **10248.21** | **3519.79** |

## Output parity

- `HAL.2004.page_82.pdf_125315.png`: Python and .NET `tf_responses`/`predict_details` diverge (different bounding boxes and
  derived metadata).
- `HAL.2004.page_82.pdf_125317.png`: Python and .NET outputs also diverge; JSON parity remains blocked.

Until the Python multi-table dump matches `results/tableformer_docling_fintabnet.json`, subsequent verification steps must stay on
hold.
