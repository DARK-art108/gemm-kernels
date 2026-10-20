from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path

import modal


app = modal.App("fused-rmsnorm-swiglu")
_this_file = Path(__file__).resolve()
REPO_ROOT = _this_file.parents[4] if len(_this_file.parents) > 4 else Path.cwd()
LOG_DIR = _this_file.parent / "logs" / "fused_rmsnorm_swiglu"
IMAGE = modal.Image.from_registry("pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel")


@app.function(image=IMAGE, gpu="H100", timeout=60 * 20)
def benchmark_kernel(kernel_b64: str, helpers_b64: str, reduce_b64: str) -> dict:
    import tempfile

    import torch
    from torch.utils.cpp_extension import load_inline

    include_dir = Path(tempfile.mkdtemp(prefix="fused_rmsnorm_include_"))
    (include_dir / "fused_rmsnorm_swiglu.cuh").write_bytes(base64.b64decode(kernel_b64))
    (include_dir / "helpers.cuh").write_bytes(base64.b64decode(helpers_b64))
    (include_dir / "reduce.cuh").write_bytes(base64.b64decode(reduce_b64))

    cpp_source = r'''
#include <torch/extension.h>
torch::Tensor fused_rmsnorm_swiglu_cuda(torch::Tensor input,
                                        torch::Tensor weight,
                                        torch::Tensor gate);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_rmsnorm_swiglu", &fused_rmsnorm_swiglu_cuda,
          "Fused RMSNorm + SwiGLU (CUDA)");
}
'''
    cuda_source = '#include "fused_rmsnorm_swiglu.cuh"\n'
    build_dir = Path(tempfile.mkdtemp(prefix="fused_rmsnorm_build_"))
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
    module = load_inline(
        name=f"fused_rmsnorm_swiglu_ext_{os.getpid()}",
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
        "dtype": "float16 inputs/output, float32 internal math",
    }

    def reference(input_tensor, weight_tensor, gate_tensor):
        input_f = input_tensor.float()
        gate_f = gate_tensor.float()
        weight_f = weight_tensor.float()
        inv_rms = torch.rsqrt(input_f.square().mean(dim=1, keepdim=True) + 1.0e-6)
        result = input_f * weight_f.unsqueeze(0) * gate_f * torch.sigmoid(gate_f)
        return (result * inv_rms).to(torch.float16)

    correctness = []
    correctness_cases = [
        ("small_rows", 37, 64),
        ("standard", 128, 4096),
        ("many_rows", 513, 4096),
        ("wide", 1024, 8192),
    ]
    for name, m, n in correctness_cases:
        torch.manual_seed(123)
        input_tensor = torch.randn((m, n), device="cuda", dtype=torch.float16)
        weight_tensor = torch.randn((n,), device="cuda", dtype=torch.float16)
        gate_tensor = torch.randn((m, n), device="cuda", dtype=torch.float16)
        expected = reference(input_tensor, weight_tensor, gate_tensor)
        actual = module.fused_rmsnorm_swiglu(input_tensor, weight_tensor, gate_tensor)
        torch.cuda.synchronize()
        diff = (actual.float() - expected.float()).abs()
        correctness.append({
            "name": name,
            "M": m,
            "N": n,
            "passed": bool(torch.allclose(actual, expected, rtol=2e-3, atol=2e-3)),
            "max_abs_err": float(diff.max().item()),
            "mean_abs_err": float(diff.mean().item()),
        })

    def measure(fn, iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters

    benchmark = []
    benchmark_cases = [
        (128, 4096),
        (512, 4096),
        (1024, 4096),
        (1024, 8192),
    ]
    for m, n in benchmark_cases:
        torch.manual_seed(456)
        input_tensor = torch.randn((m, n), device="cuda", dtype=torch.float16)
        weight_tensor = torch.randn((n,), device="cuda", dtype=torch.float16)
        gate_tensor = torch.randn((m, n), device="cuda", dtype=torch.float16)
        for _ in range(20):
            module.fused_rmsnorm_swiglu(input_tensor, weight_tensor, gate_tensor)
            reference(input_tensor, weight_tensor, gate_tensor)
        torch.cuda.synchronize()
        iters = 100 if m * n <= 512 * 4096 else 50
        custom_ms = measure(
            lambda: module.fused_rmsnorm_swiglu(input_tensor, weight_tensor, gate_tensor),
            iters,
        )
        reference_ms = measure(
            lambda: reference(input_tensor, weight_tensor, gate_tensor),
            iters,
        )
        nominal_bytes = 2.0 * (3.0 * m * n + n)
        benchmark.append({
            "M": m,
            "N": n,
            "iters": iters,
            "custom_ms": custom_ms,
            "reference_ms": reference_ms,
            "speedup": reference_ms / custom_ms,
            "custom_effective_gb_s": nominal_bytes / (custom_ms * 1.0e6),
            "reference_effective_gb_s": nominal_bytes / (reference_ms * 1.0e6),
        })

    return {
        "device": device,
        "operation": "input * weight * gate * sigmoid(gate) * rsqrt(mean(input^2) + 1e-6)",
        "correctness": correctness,
        "benchmark": benchmark,
        "notes": [
            "The benchmark uses N divisible by 8 because the kernel loads float4 groups of eight FP16 values.",
            "Reference timing is PyTorch eager execution of the same mathematical operation.",
            "Effective bandwidth is a nominal bytes-per-element estimate, not a hardware-counter measurement.",
        ],
    }


@app.local_entrypoint()
def main():
    kernel = REPO_ROOT / "kernels/fused/fused_rmsnorm_swiglu.cuh"
    helpers = REPO_ROOT / "kernels/common/helpers.cuh"
    reduce = REPO_ROOT / "kernels/common/reduce.cuh"
    result = benchmark_kernel.remote(
        base64.b64encode(kernel.read_bytes()).decode(),
        base64.b64encode(helpers.read_bytes()).decode(),
        base64.b64encode(reduce.read_bytes()).decode(),
    )
    timestamp = time.strftime("run_%Y%m%d_%H%M%S")
    out_dir = LOG_DIR / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "benchmark.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"Saved results to {out_dir}")
