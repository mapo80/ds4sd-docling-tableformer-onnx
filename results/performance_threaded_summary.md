# TorchSharp TableFormer Inference Profiling (Threaded Optimisation)

## Methodology
- **Datasets:** `dataset/FinTabNet/benchmark` (2 pages) and the larger `dataset/FinTabNet/images` (20 pages).
- **Runs per document:** 6 passes per image with the first acting as warm-up; metrics aggregate the remaining five passes.
- **Threading:** TorchSharp forced to use the requested thread count (2 logical cores on the runner) for both intra-op and inter-op pools.
- **Build:** .NET 9.0.10 / TorchSharp 0.105.1.0 with shared artifacts cached via the benchmark harness.
- **Artifacts:** All raw measurements are persisted under `results/perf_runs/*.json` and `results/perf_runs_full/*.json` for auditability.

## Benchmark subset (`dataset/FinTabNet/benchmark`)

### Document-level timings
| Implementation | Total (ms) | Avg / doc (ms) | Δ vs previous (ms) | Δ vs previous (%) | Δ vs Python (ms) | Δ vs Python (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Python Docling | 6728.41 | 3364.21 | – | – | – | – |
| TorchSharp (previous best) | 6376.75 | 3188.37 | – | – | -351.66 | -5.23 |
| TorchSharp (threaded) | 3341.11 | 1670.55 | -3035.64 | -47.60 | -3387.30 | -50.34 |

_Source data: Python figures from the historical comparison report, TorchSharp JSON summaries at lines 2745-2759 of the previous run and 2745-2759 of the threaded run._【F:results/FinTabNet_benchmark_comparison.md†L14-L37】【F:results/perf_runs/20251020054019769_optimized_iter03.json†L2745-L2760】【F:results/perf_runs/20251020080436671_threaded_parallel.json†L2745-L2759】

### Stage breakdown (per-dataset totals)
| Stage | Previous ms | Previous share | Threaded ms | Threaded share | Δ ms | Δ % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| decode_image_ms | 2.71 | 0.04% | 2.04 | 0.06% | -0.67 | -24.73% |
| prepare_page_ms | 0.03 | 0.00% | 0.07 | 0.00% | +0.04 | +134.34% |
| crop_tables_ms | 767.17 | 12.03% | 624.71 | 18.70% | -142.46 | -18.57% |
| tensorize_ms | 58.27 | 0.91% | 116.09 | 3.47% | +57.82 | +99.24% |
| model_inference_ms | 5536.98 | 86.83% | 2593.63 | 77.63% | -2943.35 | -53.16% |
| sequence_decode_ms | 10.48 | 0.16% | 1.26 | 0.04% | -9.22 | -88.01% |
| cell_match_ms | 0.31 | 0.00% | 0.46 | 0.01% | +0.15 | +48.81% |
| postprocess_ms | 0.00 | 0.00% | 0.00 | 0.00% | -0.00 | -6.00% |
| assemble_ms | 0.18 | 0.00% | 0.24 | 0.01% | +0.06 | +33.00% |

_Stage totals pulled from the JSON reports._【F:results/perf_runs/20251020054019769_optimized_iter03.json†L2745-L2759】【F:results/perf_runs/20251020080436671_threaded_parallel.json†L2745-L2759】 The threading changes more than halved model inference time while the new parallel table cropping shaved ~18% off that phase. Tensor creation now contributes ~3.5% of the budget (up from ~0.9%) because each measurement includes post-resize re-normalisation five times per page; this remains a small absolute cost.

### Observations
- Enabling configurable TorchSharp threading and re-running five hot passes drops average latency by **47.6%** versus the previous TorchSharp build and puts the .NET port **50.3% ahead of the Python reference** on this subset.
- Model execution still dominates (~78% of time), signalling diminishing returns without deeper kernel optimisations or model distillation.
- Cropping is now the second-largest slice (~19%); the deterministic parallel resize can be pushed further (e.g., via SIMD or caching) if required.

## Full FinTabNet set (`dataset/FinTabNet/images`)

| Metric | Value |
| --- | ---: |
| Documents | 20 |
| Total runtime | 57032.54 ms |
| Average per doc | 2851.63 ms |
| Stage distribution | decode 0.19%, prep 0.02%, crop 23.30%, tensorize 2.17%, inference 74.18%, sequence 0.07%, cell match 0.01%, post-process <0.001%, assemble 0.02% |

_Source: threaded full-run JSON summary._【F:results/perf_runs_full/20251020080537378_threaded_parallel_full.json†L80345-L80359】 The larger corpus exhibits similar proportions—model inference remains the primary hotspot, with table cropping consuming roughly a quarter of the wall-clock time even after vectorised resizing.

## Key changes behind the gains
- **Torch threading control:** `TableFormerNeuralModel` now honours user-specified or environment-provided thread counts instead of hard-wiring single-threaded execution.【F:dotnet/TableFormerTorchSharpSdk/Model/TableFormerNeuralModel.cs†L17-L116】【F:dotnet/TableFormerTorchSharpSdk/Model/TableFormerNeuralModel.cs†L208-L266】 This ensures the benchmark run uses both logical cores available in the sandbox.
- **Per-document sampling & reporting:** The benchmark harness was refactored to run six passes per page, discard the warm-up, and surface averaged stage timings while preserving prediction parity with the canonical dataset.【F:dotnet/TableFormerTorchSharpSdk.Benchmarks/Program.cs†L214-L432】【F:dotnet/TableFormerTorchSharpSdk.Benchmarks/Program.cs†L439-L521】
- **Deterministic parallel resizing:** Table cropping now precomputes bilinear weights and resizes images in parallel without losing numerical fidelity, cutting `crop_tables_ms` by ~18%.【F:dotnet/TableFormerTorchSharpSdk/PagePreparation/TableFormerTableCropper.cs†L1-L205】

## Next steps
- With inference still >70% of the runtime on both datasets, future work should focus on model-level optimisations (quantisation, fused operators, or batched table passes).
- The tensorisation spike suggests revisiting the image-normalisation path to amortise redundant work across the five measured passes (e.g., caching or using faster transforms).
- Python/.NET parity remains limited by upstream reference divergence; once Python artifacts are rebuilt, the same multi-pass harness can validate accuracy at scale.
