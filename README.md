# gemm-kernels

A progressive journey of optimizing CUDA GEMM (General Matrix Multiply) kernels from scratch to peak performance — written in CUDA C++ and benchmarked on real GPU hardware using [Modal](https://modal.com).

> **Goal:** Start from a naive implementation and step by step optimize it until we get as close to cuBLAS as possible.

---

## What is GEMM?

GEMM computes: **C = α × (A @ B) + β × C**

Where A, B, C are matrices and α, β are scalar values. This is the most fundamental operation in deep learning — every linear layer, attention mechanism, and convolution eventually becomes a GEMM.

Getting GEMM fast = getting your model fast.

---

## Hardware

All kernels are benchmarked on:

| Property | Value |
|----------|-------|
| GPU | NVIDIA Tesla T4 |
| Architecture | Turing |
| SM Version | SM 7.5 |
| FP32 Peak | 8.1 TFLOPS |
| VRAM | 16 GB |
| Precision | FP32 (float32) |

Runs are executed remotely via [Modal](https://modal.com) — no local GPU required.

---

## Kernels

### `01_kernel_blocktiling_1d.cuh` — 1D Block Tiling

The first real optimization step. Instead of every thread reading from slow global memory independently, we divide the matrices into tiles and load them into fast **shared memory** (the GPU's on-chip scratchpad).

**Key idea:**
- Output matrix C is divided into `BM × BN` tiles
- Each thread block computes one tile of C
- Each thread computes `TM` rows of output (1D tiling)
- Tiles of A and B are loaded into shared memory and reused

**Parameters:**
```
BM = 64   (tile height — rows per block)
BN = 64   (tile width  — cols per block)
BK = 8    (tile depth  — K-dimension per iteration)
TM = 8    (rows each thread computes)
Threads per block = (BM / TM) × BN = 512
```

**Benchmark on T4 (SM 7.5, FP32):**

| Size (M=N=K) | Kernel (ms) | Kernel TFLOPS | cuBLAS (ms) | cuBLAS TFLOPS | % of cuBLAS |
|---|---|---|---|---|---|
| 512 | 0.375 | 0.72 | 0.132 | 2.03 | 35.3% |
| 1024 | 2.443 | 0.88 | 0.752 | 2.86 | 30.8% |
| 2048 | 10.019 | 1.71 | 4.254 | 4.04 | 42.5% |
| 4096 | 82.786 | 1.66 | 34.473 | 3.99 | 41.6% |

**Peak reached: ~1.7 TFLOPS (~42% of cuBLAS)**

**Why it plateaus:**
- `BK=8` is too small — not enough data reused per shared memory load
- Single-element loads (no `float4` vectorization)
- No double buffering or prefetching

---

## Optimization Roadmap

```
Kernel 01 — 1D Block Tiling              ✅ done   ~42% cuBLAS
Kernel 02 — 2D Block Tiling              🔜 next   ~60% cuBLAS (expected)
Kernel 03 — Vectorized Loads (float4)    🔜         ~70%
Kernel 04 — Double Buffering             🔜         ~80%
Kernel 05 — Warp Tiling                  🔜         ~85%
Kernel 06 — Tensor Core (WMMA)           🔜         ~95%
```

---

## How to Run

Kernels are run on Modal (free GPU cloud). No local GPU needed.

**1. Install Modal:**
```bash
pip install modal
modal setup
```

**2. Run a kernel:**
```bash
modal run run_kernel.py
```

This will:
- Spin up a T4 GPU on Modal
- Compile the CUDA kernel via PyTorch's `load_inline`
- Run correctness checks against PyTorch's `torch.matmul`
- Benchmark against cuBLAS across multiple matrix sizes
- Print the full performance report

---

## How Benchmarking Works

```
FLOPs  = 2 × M × N × K        (multiply + add per element)
Time   = average of 50 runs    (measured with CUDA events)
TFLOPS = FLOPs / time / 10¹²
```

Correctness is verified with `torch.allclose(rtol=1e-3, atol=1e-3)` against `torch.matmul`.

---

## File Structure

```
gemm-kernels/
├── 01_kernel_blocktiling_1d.cuh   # Kernel 01: 1D block tiling
├── utils.cuh                       # Shared utilities (ceil_div etc.)
├── run_kernel.py                   # Modal runner + benchmark harness
└── README.md
```

---

## References

- [Simon Boehm — How to Optimize a CUDA Matmul Kernel](https://siboehm.com/articles/22/CUDA-MMM)
- [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/)
