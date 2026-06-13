## Inference benchmark (CPU)

Sample numbers - will be overwritten when you run `scripts/benchmark.py`
against your real model.

| Backend          | Single p50 (ms) | Single p99 (ms) | Batch-32 mean (ms) | Throughput (samples/s) |
|---|---|---|---|---|
| PyTorch eager    | 24.8 | 38.4 | 182.0 | 176 |
| ONNX Runtime CPU |  9.7 | 14.1 |  71.0 | 451 |

Speedup: **2.56x** single-sample p50, **2.56x** batch-32 throughput.
