# NAT12 CUDA validation evidence

## Status

NAT12 is fully validated at 100%.

The CPU path was validated before merge in pull request 68. The remaining CUDA validation was completed on 26 June 2026 against the merged NAT12 implementation.

NAT12 implementation commit:

78a0b60

NAT12 merge commit:

10b557f

## Isolation

The validation used a dedicated Docker inference container named `nat12-cuda-inference`.

The container used:

- the merged NAT12 code from `origin/main`
- an isolated read only model cache
- a dedicated input directory
- the NVIDIA GeForce RTX 3060
- MinIO only for publishing the generated visual artifacts

The shared model cache and the Feature AE checkpoint stored in MinIO were not modified.

## Model integrity

The ROI checkpoint SHA256 was:

37d1b8d62370f16ecdb5f4bbab66f39bafc39de99f22d6ab286860af57463cf4

The Feature AE checkpoint SHA256 was:

9e4e3cc84fb359408ec5c58020e799816942d6c46ae024b29acb82c1c8ebe146

This Feature AE SHA256 exactly matches the value declared by the official model manifest.

The source image SHA256 was:

ab598e6cd9d00859b7c0bbd0e8e37061b5e61efe6086e793cd5da492c5c054dc

## CUDA runtime validation

The Docker image exposed the following CUDA runtime:

- PyTorch: 2.11.0+cu128
- CUDA runtime: 12.8
- GPU: NVIDIA GeForce RTX 3060
- CUDA device count: 1
- CUDA tensor computation: successful

A real image was submitted to the NAT12 inference service with:

- piece event: `nat12_real_cuda_001`
- scenario: `production_replay_natural`
- lot: `NAT12_CUDA_VALIDATION`
- source class: `Casting_class3`
- dataset version: `hss_iad_runtime_validation`

The service returned HTTP 200 with:

- anomaly score: `14.808084487915039`
- decision: `Vert`
- ROI status: `ok`
- ROI model: `roi_segmenter_v001_fixed`
- Feature AE model: `rd_feature_ae_gated_v001_bootstrap`

The GPU was genuinely used during inference:

- peak GPU utilization: 38%
- peak GPU memory: 541 MiB

The small numerical difference from the CPU score does not change the decision and is consistent with normal CPU and CUDA floating point differences.

## MinIO artifacts

The CUDA inference generated and published two artifacts:

Heatmap:

`s3://iqa-heatmaps/lots/production_replay_natural/NAT12_CUDA_VALIDATION/nat12_real_cuda_001_nat12_source_heatmap.png`

Size:

622621 bytes

ROI mask:

`s3://iqa-roi-masks/lots/production_replay_natural/NAT12_CUDA_VALIDATION/nat12_real_cuda_001_nat12_source_roi.png`

Size:

7188 bytes

## CI correction discovered after the NAT12 merge

The complete GitHub CI suite initially reported seven failures in security contract tests.

The NAT12 test fixture that replaced the HTTP inference service was located under `tests/api/conftest.py`. Its scope therefore covered API tests but not tests under `tests/security`.

The fixture was moved to the global `tests/conftest.py`. Unit and contract tests are now independent from a running inference service, while `test_inference_http_delegation.py` remains excluded from the fixture and continues to validate the HTTP delegation behavior.

Validation after the correction:

- targeted regression perimeter: 23 passed
- complete CI test suite: 684 passed
- skipped: 17
- warnings: 4
- failures: 0

## Raw evidence

The raw evidence is stored under:

`reports/phase3/nat12_cuda_validation/`

Files:

- `inference_logs.txt`
- `gpu_metrics.csv`
- `inference_response.json`
- `sha256.txt`
