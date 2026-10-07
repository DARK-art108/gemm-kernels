# gemm-kernels

CUDA SGEMM kernels from first principles, benchmarked on remote GPUs with
[Modal](https://modal.com). The repo is organized so source kernels, target
benchmark harnesses, and generated logs live in predictable places.

## Layout

```text
gemm-kernels/
├── kernels/
│   ├── common/
│   │   └── utils.cuh
│   └── sgemm/
│       ├── kernel_blocktiling_1d.cuh
│       └── kernel_blocktiling_2d.cuh
├── benchmarks/
│   ├── sm75/
│   │   └── fp32/
│   │       └── t4/
│   │           └── kernel_blocktiling_1d.py
│   └── sm90/
│       └── fp32/
│           └── h100/
│               ├── kernel_blocktiling_2d.py
│               └── logs/
│                   └── kernel_blocktiling_2d/
├── tools/
│   └── modal_check.py
└── README.md
```

The benchmark path encodes the target:

```text
benchmarks/<sm-version>/<precision>/<gpu>/
```

For example, H100 FP32 Hopper runs live in:

```text
benchmarks/sm90/fp32/h100/
```

Run artifacts for that target are written below that benchmark folder:

```text
benchmarks/sm90/fp32/h100/logs/kernel_blocktiling_2d/run_YYYYMMDD_HHMMSS/
```

`logs/` directories and Python bytecode are ignored by git.

## GEMM

SGEMM computes:

```text
C = alpha * (A @ B) + beta * C
```

For FP32 matrices, the benchmark reports:

```text
FLOPs  = 2 * M * N * K
Time   = average CUDA event time over repeated launches
TFLOPS = FLOPs / time / 1e12
```

Correctness is checked against `torch.matmul` with `rtol=1e-3` and `atol=1e-3`.

## Kernels

### `kernels/sgemm/kernel_blocktiling_1d.cuh`

First shared-memory SGEMM kernel. Each thread computes a vertical strip of
`TM` output values.

```text
BM = 64
BN = 64
BK = 8
TM = 8
threads/block = (BM / TM) * BN = 512
```

Previously measured on Modal T4:

| Size | Kernel ms | Kernel TFLOPS | cuBLAS ms | cuBLAS TFLOPS | % cuBLAS |
|---:|---:|---:|---:|---:|---:|
| 512 | 0.375 | 0.72 | 0.132 | 2.03 | 35.3% |
| 1024 | 2.443 | 0.88 | 0.752 | 2.86 | 30.8% |
| 2048 | 10.019 | 1.71 | 4.254 | 4.04 | 42.5% |
| 4096 | 82.786 | 1.66 | 34.473 | 3.99 | 41.6% |

Run it with:

```bash
modal run benchmarks/sm75/fp32/t4/kernel_blocktiling_1d.py
```

### `kernels/sgemm/kernel_blocktiling_2d.cuh`

Second shared-memory SGEMM kernel. Each thread computes a `TM x TN` micro-tile,
so each block covers a `BM x BN` tile of C.

```text
BM = 64
BN = 64
BK = 8
TM = 8
TN = 8
threads/block = (BM / TM) * (BN / TN) = 64
```

The main kernel handles full tiles. A separate edge kernel handles M/N tails.
The current implementation assumes benchmarked K values are divisible by `BK=8`;
K-tail behavior is not patched yet.

Run it with:

```bash
modal run benchmarks/sm90/fp32/h100/kernel_blocktiling_2d.py
```

Use a different single-launch profile size:

```bash
modal run benchmarks/sm90/fp32/h100/kernel_blocktiling_2d.py --profile-size 8192
```

## Latest H100 FP32 Result

Command:

```bash
modal run benchmarks/sm90/fp32/h100/kernel_blocktiling_2d.py
```

Environment:

| Property | Value |
|---|---|
| GPU | NVIDIA H100 80GB HBM3 |
| SMs | 132 |
| Compute capability | 9.0 |
| CUDA | 12.4 |
| PyTorch | 2.4.0 |
| TF32 | Disabled for benchmark comparison |

Correctness passed for:

| Case | Shape | alpha | beta | Max abs err |
|---|---:|---:|---:|---:|
| square_256 | 256x256x256 | 1.0 | 0.0 | 4.96e-05 |
| square_alpha_beta | 512x512x512 | 1.5 | 0.5 | 1.45e-04 |
| rectangular | 512x768x256 | 1.0 | 0.0 | 0 |
| edge_mn_k_multiple_of_bk | 1000x832x504 | 1.0 | 0.0 | 0 |

Benchmark:

| Size | Custom ms | Custom TFLOPS | cuBLAS ms | cuBLAS TFLOPS | % cuBLAS |
|---:|---:|---:|---:|---:|---:|
| 512 | 0.0779 | 3.44 | 0.0163 | 16.46 | 20.9% |
| 1024 | 0.1560 | 13.76 | 0.0562 | 38.19 | 36.0% |
| 2048 | 0.9721 | 17.67 | 0.3412 | 50.36 | 35.1% |
| 4096 | 7.4775 | 18.38 | 2.6448 | 51.97 | 35.4% |
| 8192 | 58.0961 | 18.93 | 21.0749 | 52.17 | 36.3% |

Peak observed: about `18.9 TFLOPS`, or about `36%` of FP32 cuBLAS on H100 with
TF32 disabled.

## Profiling Notes

The H100 2D harness runs three profiling passes after the benchmark:

- `ncu --set full` over one 4096 kernel launch.
- `nsys profile` over one 4096 kernel launch using `cudaProfilerStart/Stop`.
- A PyTorch profiler Chrome trace that can be opened directly in Perfetto UI.

Current Modal behavior:

- NCU fails in this environment with return code `9`:
  `Failed to prepare kernel for profiling`, `Unknown Error on device 0`, and
  `No kernels were profiled`. No `.ncu-rep` is produced.
- NSYS succeeds. For the 4096 launch, it reports one
  `sgemm_blocktiling_2d_kernel<64,64,8,8,8>` instance taking about `7.47 ms`.
- Nsight Systems 2026.5 rejects `nsys export --type=perfetto`; the harness
  writes a Perfetto-compatible `torch_perfetto_trace.json` instead.

Latest local artifacts from the last run:

```text
benchmarks/sm90/fp32/h100/logs/kernel_blocktiling_2d/run_20261008_004951/
├── benchmark.json
├── raw_result.json
├── summary.json
└── torch_perfetto_trace.json
```

Open `torch_perfetto_trace.json` at [ui.perfetto.dev](https://ui.perfetto.dev/).

## Setup

Install and authenticate Modal:

```bash
pip install modal
modal setup
```

Check the Modal CUDA/PyTorch environment:

```bash
modal run tools/modal_check.py
```

## Roadmap

```text
Kernel 01 - 1D block tiling              done
Kernel 02 - 2D block tiling              benchmarked, ~36% cuBLAS on H100 FP32
Kernel 03 - vectorized loads             next
Kernel 04 - double buffering             planned
Kernel 05 - warp tiling                  planned
Kernel 06 - tensor cores / WMMA          planned
```

Immediate next work:

- Add correct K-tail handling to the 2D kernel.
- Add vectorized global memory loads where alignment permits.
- Improve the shared-memory layout to reduce bank conflicts.
- Keep each new benchmark under `benchmarks/<sm>/<precision>/<gpu>/`.

## References

- [Simon Boehm - How to Optimize a CUDA Matmul Kernel](https://siboehm.com/articles/22/CUDA-MMM)
- [NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [NVIDIA cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/)
- [NVIDIA Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/)
