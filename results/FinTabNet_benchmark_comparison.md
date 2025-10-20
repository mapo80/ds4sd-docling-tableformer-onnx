# FinTabNet Benchmark Comparison

This report captures the outputs and runtimes observed on the `dataset/FinTabNet/benchmark` subset for both the Python Docling
pipeline and the TorchSharp port. All runs were executed with four intra-op threads.

## Artifacts

- Canonical Docling reference: `results/tableformer_docling_fintabnet.json`.
- Python benchmark output: `results/tableformer_docling_fintabnet_python.json` (generated with `--skip-reference-check`).
- .NET benchmark output (pre-optimization): `results/tableformer_docling_fintabnet_dotnet.json`.
- .NET benchmark output (post-optimization): `results/tableformer_docling_fintabnet_dotnet_postopt.json`.
- **Parity status:** the serialized predictions still diverge between Python and .NET; verification cannot advance until the Python
  artifacts are realigned with the canonical reference.

## Timings vs Python

### Before optimization

| Document | Python (ms) | .NET pre (ms) | Δ (.NET − Python) ms |
| --- | ---: | ---: | ---: |
| HAL.2004.page_82.pdf_125315.png | 4164.38 | 5635.75 | 1471.38 |
| HAL.2004.page_82.pdf_125317.png | 2564.04 | 4612.46 | 2048.42 |
| **TOTAL** | **6728.41** | **10248.21** | **3519.79** |

### After optimization

| Document | Python (ms) | .NET post (ms) | Δ (.NET − Python) ms |
| --- | ---: | ---: | ---: |
| HAL.2004.page_82.pdf_125315.png | 4164.38 | 3998.64 | -165.74 |
| HAL.2004.page_82.pdf_125317.png | 2564.04 | 1830.90 | -733.14 |
| **TOTAL** | **6728.41** | **5829.53** | **-898.88** |

## .NET improvement (pre → post)

| Document | .NET pre (ms) | .NET post (ms) | Δ (post − pre) ms |
| --- | ---: | ---: | ---: |
| HAL.2004.page_82.pdf_125315.png | 5635.75 | 3998.64 | -1637.11 |
| HAL.2004.page_82.pdf_125317.png | 4612.46 | 1830.90 | -2781.56 |
| **TOTAL** | **10248.21** | **5829.53** | **-4418.68** |

## Output parity

- `HAL.2004.page_82.pdf_125315.png`: Python and .NET `tf_responses`/`predict_details` diverge (different bounding boxes and
derived metadata).
- `HAL.2004.page_82.pdf_125317.png`: Python and .NET outputs also diverge; JSON parity remains blocked.

Until the Python multi-table dump matches `results/tableformer_docling_fintabnet.json`, subsequent verification steps must stay on
hold.
