from __future__ import annotations

import base64
import json
import time
from pathlib import Path

import modal


app = modal.App("sgemm-vectorized-float4")
REPO_ROOT = Path(__file__).resolve().parents[4] if len(Path(__file__).resolve().parents) > 4 else Path.cwd()
LOG_DIR = Path(__file__).resolve().parent / "logs" / "kernel_vectorized_float4"
IMAGE = modal.Image.from_registry("pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel")


@app.function(image=IMAGE, gpu="H100", timeout=60 * 20)
def benchmark_kernel(kernel_b64: str, utils_b64: str) -> dict:
    import os
    import tempfile

    import torch
    from torch.utils.cpp_extension import load_inline

    include_dir = Path(tempfile.mkdtemp(prefix="sgemm_vectorized_include_"))
    (include_dir / "kernel_vectorized_float4.cuh").write_bytes(base64.b64decode(kernel_b64))
    (include_dir / "utils.cuh").write_bytes(base64.b64decode(utils_b64))
    cpp_source = r'''
#include <torch/extension.h>
void sgemm_vectorize(const torch::Tensor&, const torch::Tensor&, torch::Tensor&, float, float);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sgemm_vectorize", &sgemm_vectorize, "Vectorized float4 SGEMM");
}
'''
    cuda_source = '#include "kernel_vectorized_float4.cuh"\n'
    build_dir = Path(tempfile.mkdtemp(prefix="sgemm_vectorized_build_"))
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
    module = load_inline(
        name=f"sgemm_vectorized_float4_ext_{os.getpid()}",
        cpp_sources=[cpp_source],
        cuda_sources=[cuda_source],
        extra_include_paths=[str(include_dir)],
        extra_cuda_cflags=["-O3", "-lineinfo"],
        build_directory=str(build_dir),
        verbose=False,
    )

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    prop = torch.cuda.get_device_properties(0)
    device = {
        "name": prop.name,
        "sm_count": prop.multi_processor_count,
        "capability": f"{prop.major}.{prop.minor}",
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
    }

    correctness = []
    cases = [
        ("square_256", 256, 256, 256, 1.0, 0.0),
        ("square_alpha_beta", 512, 512, 512, 1.5, 0.5),
        ("rectangular", 512, 768, 256, 1.0, 0.0),
        ("large_square", 1024, 1024, 1024, 1.0, 0.0),
        ("irregular_m_n_k", 1000, 832, 504, 1.0, 0.0),
        ("unaligned_tails", 1003, 835, 507, 1.0, 0.0),
    ]
    for name, m, n, k, alpha, beta in cases:
        torch.manual_seed(123)
        a = torch.randn((m, k), device="cuda", dtype=torch.float32)
        b = torch.randn((k, n), device="cuda", dtype=torch.float32)
        c0 = torch.randn((m, n), device="cuda", dtype=torch.float32) if beta else torch.zeros((m, n), device="cuda")
        c = c0.clone()
        ref = alpha * torch.matmul(a, b) + beta * c0
        module.sgemm_vectorize(a, b, c, alpha, beta)
        torch.cuda.synchronize()
        diff = (c - ref).abs()
        correctness.append({
            "name": name,
            "shape": [m, n, k],
            "passed": bool(torch.allclose(c, ref, rtol=1e-3, atol=1e-3)),
            "max_abs_err": float(diff.max().item()),
            "mean_abs_err": float(diff.mean().item()),
        })

    benchmark = []
    for size in [512, 1024, 2048, 4096, 8192]:
        a = torch.randn((size, size), device="cuda", dtype=torch.float32)
        b = torch.randn((size, size), device="cuda", dtype=torch.float32)
        c = torch.zeros((size, size), device="cuda", dtype=torch.float32)
        ref = torch.zeros_like(c)
        for _ in range(20):
            module.sgemm_vectorize(a, b, c, 1.0, 0.0)
            torch.matmul(a, b, out=ref)
        torch.cuda.synchronize()
        iters = 100 if size <= 1024 else 50 if size <= 2048 else 20 if size <= 4096 else 8

        def measure(fn):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                fn()
            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end) / iters

        custom_ms = measure(lambda: module.sgemm_vectorize(a, b, c, 1.0, 0.0))
        cublas_ms = measure(lambda: torch.matmul(a, b, out=ref))
        flops = 2.0 * size**3
        custom_tflops = flops / (custom_ms * 1e9)
        cublas_tflops = flops / (cublas_ms * 1e9)
        benchmark.append({
            "size": size,
            "custom_ms": custom_ms,
            "custom_tflops": custom_tflops,
            "cublas_ms": cublas_ms,
            "cublas_tflops": cublas_tflops,
            "pct_cublas": 100.0 * custom_tflops / cublas_tflops,
        })
    return {"device": device, "correctness": correctness, "benchmark": benchmark}


@app.local_entrypoint()
def main():
    kernel = REPO_ROOT / "kernels/sgemm/kernel_vectorized_float4.cuh"
    utils = REPO_ROOT / "kernels/common/utils.cuh"
    result = benchmark_kernel.remote(
        base64.b64encode(kernel.read_bytes()).decode(),
        base64.b64encode(utils.read_bytes()).decode(),
    )
    timestamp = time.strftime("run_%Y%m%d_%H%M%S")
    out_dir = LOG_DIR / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "benchmark.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"Saved results to {out_dir}")
