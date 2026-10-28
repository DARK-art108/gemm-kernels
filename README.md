# gemm-kernels

CUDA SGEMM kernels from first principles, benchmarked on remote GPUs with
[Modal](https://modal.com). The repo is organized so source kernels, target
benchmark harnesses, and generated logs live in predictable places.

## Layout

```text
gemm-kernels/
├── kernels/
│   ├── common/
│   │   ├── helpers.cuh
│   │   ├── reduce.cuh
│   │   └── utils.cuh
│   ├── fused/
│   │   ├── bindings.cuh
│   │   └── fused_rmsnorm_swiglu.cuh
│   ├── attention/
│   │   └── turboquant_attention.cu
│   └── sgemm/
│       ├── kernel_blocktiling_2d.cuh
│       └── kernel_vectorized_float4.cuh
├── benchmarks/
│   ├── sm75/
│   │   └── fp32/
│   │       └── t4/
│   │           └── kernel_blocktiling_1d.py
│   └── sm90/
│       ├── fp16/
│       │   └── h100/
│       │       ├── fused_rmsnorm_swiglu.py
│       │       └── logs/
│       │           └── fused_rmsnorm_swiglu/
│       └── fp32/
│           └── h100/
│               ├── kernel_blocktiling_2d.py
│               ├── kernel_vectorized_float4.py
│               ├── turboquant_attention.py
│               └── logs/
│                   ├── kernel_blocktiling_2d/
│                   ├── kernel_vectorized_float4/
│                   └── turboquant_attention/
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

## Fused RMSNorm + SwiGLU

`kernels/fused/fused_rmsnorm_swiglu.cuh` fuses the per-row computation into
one CUDA launch. For input row $x_m \in \mathbb{R}^{N}$, learned weight
$w \in \mathbb{R}^{N}$, and gate row $g_m \in \mathbb{R}^{N}$, the kernel
computes:

$$
s_{m,n}
  = x_{m,n}\,w_n\,g_{m,n}\,\sigma(g_{m,n}),
\qquad
\sigma(z)=\frac{1}{1+e^{-z}}.
$$

It then applies RMS normalization using the input row:

$$
\mathrm{inv\_rms}_m
  = \left(\frac{1}{N}\sum_{n=0}^{N-1}x_{m,n}^{2}
    +\varepsilon\right)^{-1/2},
\qquad \varepsilon=10^{-6},
$$

and writes:

$$
y_{m,n}=s_{m,n}\,\mathrm{inv\_rms}_m.
$$

The implementation loads FP16 values, performs the products, sigmoid, and
reduction in FP32, then converts the output back to FP16. Each CUDA block
processes one row. Threads cooperatively load eight FP16 values at a time
through `float4`, reduce the row sum in shared memory, and write the fused
result.

The supporting `helpers.cuh` dependency uses CUDA's `__forceinline__`
qualifier so the extension compiles with NVCC.

The kernel's vectorized indexing requires `N` to be divisible by `8`. The
benchmark therefore uses valid `N` values and does not claim arbitrary
non-multiple-of-eight support.

Run it with:

```bash
modal run benchmarks/sm90/fp16/h100/fused_rmsnorm_swiglu.py
```

## TurboQuant KV-Cache Attention

`kernels/attention/turboquant_attention.cu` computes raw attention scores
against TurboQuant-compressed keys without materializing a dequantized
`S x D` key matrix.

For query matrix $Q \in \mathbb{R}^{B\times D}$, codebook indices
$K_{\mathrm{idx}}\in\{0,\ldots,C-1\}^{S\times D}$, codebook
$c\in\mathbb{R}^{C}$, rotation $\Pi$, residual projection
$S_{\mathrm{mat}}\in\mathbb{R}^{D\times D}$,
sign sketch $\sigma$, and residual magnitude $\gamma$, the score is:

$$
\widetilde{K}_{s,:}
 =
c[K_{\mathrm{idx}_{s,:}}]\Pi
 +
\frac{\sqrt{\pi/2}}{D}\,
\gamma_s\,\sigma_{s,:}S_{\mathrm{mat}},
$$

$$
\mathrm{scores}_{b,s}
 =
\sum_{d=0}^{D-1}Q_{b,d}\widetilde{K}_{s,d}.
$$

The implementation factors the query-side matrix products:

$$
A_{b,d}=\sum_{j=0}^{D-1}\Pi_{d,j}Q_{b,j},
\qquad
T_{b,d}=\sum_{j=0}^{D-1}S_{\mathrm{mat},d,j}Q_{b,j}.
$$

Each score then uses:

$$
\mathrm{scores}_{b,s}
 =
\sum_d A_{b,d}\,c[K_{\mathrm{idx}_{s,d}}]
 +
\frac{\sqrt{\pi/2}}{D}\gamma_s
\sum_d T_{b,d}\sigma_{s,d}.
$$

This is a raw score kernel only: it does not apply $1/\sqrt{D}$ scaling,
softmax, or value-cache multiplication.

The source is kept unchanged from the submitted solution. The Modal harness
adds only a C++ binding around the required `extern "C" solve` entry point.
Run the benchmark with:

```bash
modal run benchmarks/sm90/fp32/h100/turboquant_attention.py
```

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

### `kernels/sgemm/kernel_vectorized_float4.cuh`

Vectorized-load SGEMM kernel. Each thread computes an `8 x 8` micro-tile
while loading four contiguous FP32 values at a time with `float4`.

```text
BM = 128
BN = 128
BK = 8
TM = 8
TN = 8
threads/block = (BM / TM) * (BN / TN) = 256
```

The launch uses ceiling-divided grids and masks M/N/K edge tiles. Full aligned
groups of four use `float4`; tail groups use zero-filled scalar loads.
Correctness and performance are benchmarked on an H100 with FP32 TF32 disabled.

Run it with:

```bash
modal run benchmarks/sm90/fp32/h100/kernel_vectorized_float4.py
```

## Latest H100 FP32 Vectorized-Load Result

Command:

```bash
modal run benchmarks/sm90/fp32/h100/kernel_vectorized_float4.py
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

Correctness passed for all six cases, including alpha/beta, rectangular, and
unaligned M/N/K tails. Maximum absolute error was `1.45e-4`.

Benchmark:

| Size | Custom ms | Custom TFLOPS | cuBLAS ms | cuBLAS TFLOPS | % cuBLAS |
|---:|---:|---:|---:|---:|---:|
| 512 | 0.1118 | 2.40 | 0.0152 | 17.68 | 13.6% |
| 1024 | 0.2062 | 10.41 | 0.0542 | 39.64 | 26.3% |
| 2048 | 0.5900 | 29.12 | 0.3386 | 50.74 | 57.4% |
| 4096 | 4.7672 | 28.83 | 2.6394 | 52.07 | 55.4% |
| 8192 | 38.1850 | 28.79 | 21.0683 | 52.19 | 55.2% |

Peak observed: about `29.1 TFLOPS`, or about `57%` of FP32 cuBLAS on H100
with TF32 disabled.

Results are stored in:

```text
benchmarks/sm90/fp32/h100/logs/kernel_vectorized_float4/run_20261014_001037/
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

## Latest H100 FP16 Fused RMSNorm + SwiGLU Result

Command:

```bash
modal run benchmarks/sm90/fp16/h100/fused_rmsnorm_swiglu.py
```

Environment:

| Property | Value |
|---|---|
| GPU | NVIDIA H100 80GB HBM3 |
| SMs | 132 |
| Compute capability | 9.0 |
| CUDA | 12.4 |
| PyTorch | 2.4.0 |
| Input/output dtype | FP16 |
| Internal arithmetic | FP32 |
| Reference | PyTorch eager |

Correctness compares the CUDA output with the same operation evaluated in
PyTorch and converted to FP16. All cases passed with `rtol=2e-3` and
`atol=2e-3`:

| Case | M | N | Max abs err | Mean abs err |
|---|---:|---:|---:|---:|
| small_rows | 37 | 64 | 0 | 0 |
| standard | 128 | 4096 | 9.77e-4 | 1.03e-8 |
| many_rows | 513 | 4096 | 1.95e-3 | 7.07e-9 |
| wide | 1024 | 8192 | 7.81e-3 | 9.31e-9 |

The fused kernel is compared against the following per-element operation:

$$
y_{m,n} =
x_{m,n}\,w_n\,g_{m,n}\,\sigma(g_{m,n})
\left(\frac{1}{N}\sum_{k=0}^{N-1}x_{m,k}^{2}+10^{-6}\right)^{-1/2}.
$$

Benchmark latency:

| Shape `(M, N)` | Fused ms | PyTorch eager ms | Speedup | Fused nominal GB/s |
|---|---:|---:|---:|---:|
| `(128, 4096)` | 0.00492 | 0.05572 | 11.32x | 640.7 |
| `(512, 4096)` | 0.00552 | 0.08737 | 15.84x | 2282.7 |
| `(1024, 4096)` | 0.00854 | 0.14960 | 17.52x | 2948.8 |
| `(1024, 8192)` | 0.02577 | 0.29451 | 11.43x | 1954.0 |

For this non-matmul operation, latency and speedup are more useful than
TFLOPS. The benchmark reports a nominal effective bandwidth using:

$$
B_{\mathrm{nominal}} = 2(3MN+N)\ \mathrm{bytes},
\qquad
\mathrm{GB/s} =
\frac{B_{\mathrm{nominal}}}{t_{\mathrm{ms}}\times 10^{6}}.
$$

This estimate counts FP16 input, gate, output, and one FP16 weight vector.
It is not a hardware-counter measurement and does not account for cache
reuse or all internal traffic.

The current kernel requires `N % 8 == 0` because its vectorized path consumes
eight FP16 values per thread iteration. The benchmark intentionally uses
compatible `N` values and does not modify that kernel behavior.

Results are stored in:

```text
benchmarks/sm90/fp16/h100/logs/fused_rmsnorm_swiglu/run_20261021_004658/
```

## Latest H100 TurboQuant Attention Result

Command:

```bash
modal run benchmarks/sm90/fp32/h100/turboquant_attention.py
```

Environment:

| Property | Value |
|---|---|
| GPU | NVIDIA H100 80GB HBM3 |
| SMs | 132 |
| Compute capability | 9.0 |
| CUDA | 12.4 |
| PyTorch | 2.4.0 |
| Workload | `B=32, S=32768, D=128, C=16` |

Correctness passed for:

| Case | `(B, S, D, C)` | Max abs err | Mean abs err |
|---|---:|---:|---:|
| small | `(2, 3, 2, 4)` | 0 | 0 |
| odd_batch_tokens | `(3, 17, 7, 8)` | `7.15e-7` | `1.69e-7` |
| medium | `(8, 257, 32, 16)` | `3.81e-6` | `5.66e-7` |

All cases passed with `rtol=2e-4` and `atol=2e-4`. The reference computes:

$$
\widetilde{K}_{s,:}
 =
\mathrm{codebook}[K_{\mathrm{idx}_{s,:}}]\Pi
 +
\frac{\sqrt{\pi/2}}{D}\,
\gamma_s\,\sigma_{s,:}S_{\mathrm{mat}},
\qquad
\mathrm{scores}_{b,s}
 =
\sum_{d=0}^{D-1}Q_{b,d}\widetilde{K}_{s,d}.
$$

Benchmark latency:

| Shape `(B, S, D, C)` | Custom solve ms | PyTorch reference ms | Custom / reference |
|---|---:|---:|---:|
| `(32, 32768, 128, 16)` | 1.2102 | 0.2414 | `0.20x` |

The custom implementation is correct but is approximately `5.0x` slower than
the PyTorch reference for this workload. This is an implementation finding,
not a correctness failure. The current kernel:

1. Allocates and frees the transformed-query buffers `A` and `T` on every
   `solve` call.
2. Computes each transformed query coordinate with a serial loop over `D`.
3. Assigns one thread to each `(batch, token)` score and performs a serial
   `D`-term accumulation.

The PyTorch reference uses optimized matrix-multiplication and reduction
kernels. It also materializes the dequantized keys, so the comparison is
useful as a baseline but is not a like-for-like kernel implementation.

No scaling by $1/\sqrt{D}$, softmax, or value-cache operation is included.
Results are stored in:

```text
benchmarks/sm90/fp32/h100/logs/turboquant_attention/run_20261029_003409/
```

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
Kernel 03 - vectorized loads             benchmarked, ~61% cuBLAS on H100 FP32
Kernel 04 - double buffering             planned
Kernel 05 - warp tiling                  planned
Kernel 06 - tensor cores / WMMA          planned
```

Immediate next work:

- Add correct K-tail handling to the 2D kernel.
- Improve the shared-memory layout to reduce bank conflicts.
- Keep each new benchmark under `benchmarks/<sm>/<precision>/<gpu>/`.

## References

- [Simon Boehm - How to Optimize a CUDA Matmul Kernel](https://siboehm.com/articles/22/CUDA-MMM)
- [NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [NVIDIA cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/)
- [NVIDIA Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/)
