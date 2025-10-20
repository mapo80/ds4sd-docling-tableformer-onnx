# TorchSharp TableFormer Inference Profiling (Decoder Caching Pass)

## Methodology
- Same harness and datasets as the threaded benchmark: six passes per image with the first treated as warm-up, averaged over the remaining five runs, and stage-level timings captured from the benchmark runner.【F:results/performance_threaded_summary.md†L12-L33】
- New code paths reuse decoder-side attention projections and TorchSharp inference mode while keeping deterministic preprocessing so predictions remain byte-identical to the previous release.

## Benchmark subset (`dataset/FinTabNet/benchmark`)

| Metric | Threaded (prev) | Cached attention | Δ |
| --- | ---: | ---: | ---: |
| Total runtime (ms) | 3341.11 | 2193.80 | -34.3% |
| Avg / document (ms) | 1670.55 | 1096.90 | -34.3% |
| Avg / document vs Python | 3364.21 | 1096.90 | -67.4% |

- The new pass trims **573.65 ms** per document versus the threaded build and runs **67.4%** faster than the Python reference on the benchmark subset.【F:results/performance_threaded_summary.md†L12-L33】【77ebef†L1-L5】  
- Model execution drops from 2593.63 ms to 1767.56 ms (−31.8%), shaving the biggest hotspot while crop resampling and tensorisation fall by 44% and 39% respectively thanks to reusing intermediate tensors.【F:results/performance_threaded_summary.md†L18-L27】【b42c47†L1-L10】  
- Sequence decoding grows slightly (+0.9 ms total) because cached attention now feeds a single projection across every step; the overhead is negligible (<0.2% share).【b42c47†L1-L10】  

## Full FinTabNet set (`dataset/FinTabNet/images`)

| Metric | Threaded (prev) | Cached attention | Δ |
| --- | ---: | ---: | ---: |
| Total runtime (ms) | 57032.54 | 32508.21 | -43.0% |
| Avg / document (ms) | 2851.63 | 1625.41 | -43.0% |

- Extending the optimisation to the 20-page corpus cuts average latency by **43.0%** and lowers the end-to-end cost by **24.5 seconds** while keeping inference responsible for ~84% of the wall clock (down from ~90%).【F:results/performance_threaded_summary.md†L35-L42】【7ade17†L1-L6】【719a06†L1-L10】  
- Model stage time drops by **15.9 seconds** yet still dominates, underscoring that future gains will come from quantisation or more aggressive operator fusion once decoder reuse is exhausted.【719a06†L1-L10】  

## Key takeaways
- Decoder-side attention reuse removes duplicate linear projections and gating, delivering the requested **>30%** latency cut without changing predictions.【b42c47†L1-L10】  
- TorchSharp's `inference_mode` and shared encoder filters eliminate redundant tensor allocations, which in turn slashes crop and tensorise costs while keeping deterministic resizing intact.【b42c47†L1-L10】  
- With inference still >80% of runtime on the full dataset, the next wins lie in quantisation, fusing encoder kernels, or batching multiple tables per forward pass.【719a06†L1-L10】  
