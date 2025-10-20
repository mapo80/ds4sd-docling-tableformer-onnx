# TableFormer TorchSharp Performance Report

## Iterations

- Baseline: `perf_baseline` (pre-optimisation .NET 9 run)
- Current: `optimized_iter03` (best iteration from the optimised .NET 9 run)
- Dataset: `/workspace/ds4sd-docling-tableformer-onnx/dataset/FinTabNet/benchmark`

## Summary

| Metric | Baseline (ms) | Current (ms) | Delta (ms) | Delta (%) |
| --- | ---: | ---: | ---: | ---: |
| Total | 9431.73 | 3795.31 | -5636.42 | -59.76 |
| Average per document | 4715.87 | 1897.65 | -2818.21 | -59.76 |

## Stage Breakdown

| Stage | Baseline (ms) | Current (ms) | Delta (ms) | Delta (%) | Baseline Share (%) | Current Share (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| assemble_ms | 0.00 | 0.16 | +0.16 | n/a | 0.00 | 0.00 |
| cell_match_ms | 0.00 | 0.28 | +0.28 | n/a | 0.00 | 0.01 |
| crop_tables_ms | 0.00 | 775.28 | +775.28 | n/a | 0.00 | 20.43 |
| decode_image_ms | 0.00 | 1.86 | +1.86 | n/a | 0.00 | 0.05 |
| model_inference_ms | 0.00 | 2960.55 | +2960.55 | n/a | 0.00 | 78.02 |
| postprocess_ms | 0.00 | 0.00 | +0.00 | n/a | 0.00 | 0.00 |
| prepare_page_ms | 0.00 | 0.03 | +0.03 | n/a | 0.00 | 0.00 |
| sequence_decode_ms | 0.00 | 1.43 | +1.43 | n/a | 0.00 | 0.04 |
| tensorize_ms | 0.00 | 55.07 | +55.07 | n/a | 0.00 | 1.45 |

## Per-document Timings

| Document | Baseline (ms) | Current (ms) | Delta (ms) | Delta (%) |
| --- | ---: | ---: | ---: | ---: |
| HAL.2004.page_82.pdf_125315.png | 7265.10 | 1790.65 | -5474.44 | -75.35 |
| HAL.2004.page_82.pdf_125317.png | 2166.64 | 2004.65 | -161.98 | -7.48 |

## Environment

| Setting | Baseline | Current |
| --- | --- | --- |
| Requested threads | 1 | 2 |
| TorchSharp threads | 1 | 1 |
| TorchSharp interop threads | 1 | 1 |
| .NET version | 9.0.4 | 9.0.4 |
| TorchSharp version | 0.105.1.0 | 0.105.1.0 |
| Git commit | 06fedbd51464538cd77643023d095f2892bedbaf | 06fedbd51464538cd77643023d095f2892bedbaf |
