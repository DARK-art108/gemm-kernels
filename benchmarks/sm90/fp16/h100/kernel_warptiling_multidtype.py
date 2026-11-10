from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path

import modal


app = modal.App("sgemm-warptiling-multidtype")
REPO_ROOT = (
    Path(__file__).resolve().parents[4]
    if len(Path(__file__).resolve().parents) > 4
    else Path.cwd()
)
LOG_DIR = Path(__file__).resolve().parent / "logs" / "kernel_warptiling_multidtype"
IMAGE = modal.Image.from_registry("pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel")


@app.function(image=IMAGE, gpu="H100", timeout=60 * 35)
def benchmark_kernel(kernel_b64: str, utils_b64: str) -> dict:
    import tempfile

    import torch
    from torch.utils.cpp_extension import load_inline

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    include_dir = Path(tempfile.mkdtemp(prefix="sgemm_warptiling_multidtype_include_"))
    (include_dir / "kernel_warptiling_multidtype.cuh").write_bytes(
        base64.b64decode(kernel_b64)
    )
    (include_dir / "utils.cuh").write_bytes(base64.b64decode(utils_b64))

    cpp_source = r'''
#include <torch/extension.h>

void sgemm_warptiling_fp16(const torch::Tensor&, const torch::Tensor&,
                           torch::Tensor&, float, float);
void sgemm_warptiling_bf16(const torch::Tensor&, const torch::Tensor&,
                           torch::Tensor&, float, float);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sgemm_warptiling_fp16", &sgemm_warptiling_fp16,
          "Warp-tiled FP16 SGEMM with FP32 accumulation");
    m.def("sgemm_warptiling_bf16", &sgemm_warptiling_bf16,
          "Warp-tiled BF16 SGEMM with FP32 accumulation");
}
'''
    cuda_source = '#include "kernel_warptiling_multidtype.cuh"\n'
    build_dir = Path(tempfile.mkdtemp(prefix="sgemm_warptiling_multidtype_build_"))
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
    module = load_inline(
        name=f"sgemm_warptiling_multidtype_ext_{os.getpid()}",
        cpp_sources=[cpp_source],
        cuda_sources=[cuda_source],
        extra_include_paths=[str(include_dir)],
        extra_cuda_cflags=["-O3", "-lineinfo"],
        build_directory=str(build_dir),
        verbose=False,
    )

    prop = torch.cuda.get_device_properties(0)
    device = {
        "name": prop.name,
        "sm_count": prop.multi_processor_count,
        "capability": f"{prop.major}.{prop.minor}",
        "total_memory_gb": prop.total_memory / (1024 ** 3),
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
        "tf32_allowed": bool(torch.backends.cuda.matmul.allow_tf32),
    }

    def run(dtype_name, a, b, c, alpha=1.0, beta=0.0):
        if dtype_name == "float16":
            module.sgemm_warptiling_fp16(a, b, c, float(alpha), float(beta))
        else:
            module.sgemm_warptiling_bf16(a, b, c, float(alpha), float(beta))

    def correctness(dtype_name, dtype) -> list[dict]:
        cases = [
            ("single_block", 128, 128, 128, 1.0, 0.0),
            ("square_alpha_beta", 256, 256, 256, 1.5, 0.5),
            ("rectangular_aligned", 512, 768, 256, 1.0, 0.0),
            ("multi_block_aligned", 1024, 640, 512, 0.75, 0.0),
        ]
        rows = []
        for name, m, n, k, alpha, beta in cases:
            torch.manual_seed(123)
            a = torch.randn((m, k), device="cuda", dtype=dtype)
            b = torch.randn((k, n), device="cuda", dtype=dtype)
            c0 = (
                torch.randn((m, n), device="cuda", dtype=dtype)
                if beta != 0.0
                else torch.zeros((m, n), device="cuda", dtype=dtype)
            )
            c = c0.clone()

            # The kernel accumulates in FP32 and rounds only at the output.
            ref = (alpha * torch.matmul(a.float(), b.float()) + beta * c0.float()).to(dtype)
            run(dtype_name, a, b, c, alpha, beta)
            torch.cuda.synchronize()

            diff = (c.float() - ref.float()).abs()
            rows.append({
                "name": name,
                "shape": [m, n, k],
                "alpha": alpha,
                "beta": beta,
                "passed": bool(torch.allclose(c, ref, rtol=2e-2, atol=2e-2)),
                "max_abs_err": float(diff.max().item()),
                "mean_abs_err": float(diff.mean().item()),
            })
        return rows

    def measure(fn, iters: int) -> float:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters

    def benchmark(dtype_name, dtype) -> list[dict]:
        rows = []
        for size in [512, 1024, 2048, 4096, 8192]:
            torch.manual_seed(456)
            a = torch.randn((size, size), device="cuda", dtype=dtype)
            b = torch.randn((size, size), device="cuda", dtype=dtype)
            c_custom = torch.zeros((size, size), device="cuda", dtype=dtype)
            c_cublas = torch.zeros_like(c_custom)

            for _ in range(20):
                run(dtype_name, a, b, c_custom)
                torch.matmul(a, b, out=c_cublas)
            torch.cuda.synchronize()

            iters = 100 if size <= 1024 else 50 if size <= 2048 else 20 if size <= 4096 else 8
            custom_ms = measure(lambda: run(dtype_name, a, b, c_custom), iters)
            cublas_ms = measure(lambda: torch.matmul(a, b, out=c_cublas), iters)
            flops = 2.0 * size**3
            custom_tflops = flops / (custom_ms * 1e9)
            cublas_tflops = flops / (cublas_ms * 1e9)
            rows.append({
                "size": size,
                "iters": iters,
                "custom_ms": custom_ms,
                "custom_tflops": custom_tflops,
                "cublas_ms": cublas_ms,
                "cublas_tflops": cublas_tflops,
                "pct_cublas": 100.0 * custom_tflops / cublas_tflops,
            })
        return rows

    results = {}
    for dtype_name, dtype in (("float16", torch.float16), ("bfloat16", torch.bfloat16)):
        results[dtype_name] = {
            "correctness": correctness(dtype_name, dtype),
            "benchmark": benchmark(dtype_name, dtype),
        }

    return {
        "device": device,
        "configuration": {
            "BM": 128,
            "BN": 128,
            "BK": 16,
            "WM": 64,
            "WN": 64,
            "WNITER": 4,
            "TM": 8,
            "TN": 4,
            "threads_per_block": 128,
        },
        "results": results,
        "notes": [
            "Inputs and outputs use FP16 or BF16; accumulation is FP32.",
            "The reference computes in FP32 and casts to the tested output dtype.",
            "The cuBLAS comparison uses torch.matmul with the same input dtype.",
            "The kernel requires M and N divisible by 128 and K divisible by 16.",
            "All benchmark shapes are aligned to the kernel tile dimensions.",
        ],
    }


@app.local_entrypoint()
def main():
    kernel = REPO_ROOT / "kernels/sgemm/kernel_warptiling_multidtype.cuh"
    utils = REPO_ROOT / "kernels/common/utils.cuh"
    result = benchmark_kernel.remote(
        base64.b64encode(kernel.read_bytes()).decode("ascii"),
        base64.b64encode(utils.read_bytes()).decode("ascii"),
    )
    timestamp = time.strftime("run_%Y%m%d_%H%M%S")
    out_dir = LOG_DIR / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "benchmark.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"Saved results to {out_dir}")
