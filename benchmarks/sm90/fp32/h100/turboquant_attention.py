from __future__ import annotations

import base64
import json
import time
from pathlib import Path

import modal


app = modal.App("turboquant-attention")
_this_file = Path(__file__).resolve()
REPO_ROOT = _this_file.parents[4] if len(_this_file.parents) > 4 else Path.cwd()
LOG_DIR = _this_file.parent / "logs" / "turboquant_attention"
IMAGE = modal.Image.from_registry("pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel")


@app.function(image=IMAGE, gpu="H100", timeout=60 * 20)
def benchmark_kernel(kernel_b64: str) -> dict:
    import os
    import tempfile

    import torch
    from torch.utils.cpp_extension import load_inline

    kernel_path = Path(tempfile.mkdtemp(prefix="turboquant_attention_")) / "turboquant_attention.cu"
    kernel_path.write_bytes(base64.b64decode(kernel_b64))
    cpp_source = r'''
#include <torch/extension.h>

extern "C" void solve(
    const float*, const unsigned char*, const signed char*, const float*,
    const float*, const float*, const float*, float*, int, int, int, int);

torch::Tensor run_solve(
    torch::Tensor Q,
    torch::Tensor K_idx,
    torch::Tensor qjl_signs,
    torch::Tensor gamma,
    torch::Tensor Pi,
    torch::Tensor S_mat,
    torch::Tensor codebook) {
    auto scores = torch::empty({Q.size(0), K_idx.size(0)}, Q.options());
    solve(
        Q.data_ptr<float>(),
        K_idx.data_ptr<unsigned char>(),
        qjl_signs.data_ptr<signed char>(),
        gamma.data_ptr<float>(),
        Pi.data_ptr<float>(),
        S_mat.data_ptr<float>(),
        codebook.data_ptr<float>(),
        scores.data_ptr<float>(),
        static_cast<int>(Q.size(0)),
        static_cast<int>(K_idx.size(0)),
        static_cast<int>(Q.size(1)),
        static_cast<int>(codebook.size(0)));
    return scores;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run_solve", &run_solve, "TurboQuant attention scores");
}
'''
    build_dir = Path(tempfile.mkdtemp(prefix="turboquant_attention_build_"))
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
    module = load_inline(
        name=f"turboquant_attention_ext_{os.getpid()}",
        cpp_sources=[cpp_source],
        cuda_sources=[kernel_path.read_text(encoding="utf-8")],
        extra_cuda_cflags=["-O3", "-lineinfo"],
        build_directory=str(build_dir),
        verbose=False,
    )

    prop = torch.cuda.get_device_properties(0)
    device = {
        "name": prop.name,
        "sm_count": prop.multi_processor_count,
        "capability": f"{prop.major}.{prop.minor}",
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
    }
    torch.manual_seed(123)

    def make_inputs(B: int, S: int, D: int, C: int):
        q = torch.randn((B, D), device="cuda", dtype=torch.float32)
        k_idx = torch.randint(0, C, (S, D), device="cuda", dtype=torch.uint8)
        signs = torch.where(
            torch.randint(0, 2, (S, D), device="cuda") == 0,
            torch.tensor(-1, device="cuda", dtype=torch.int8),
            torch.tensor(1, device="cuda", dtype=torch.int8),
        )
        gamma = torch.rand((S,), device="cuda", dtype=torch.float32)
        codebook = torch.linspace(-1.0, 1.0, C, device="cuda", dtype=torch.float32)
        random_matrix = torch.randn((D, D), device="cuda", dtype=torch.float32)
        Pi, _ = torch.linalg.qr(random_matrix)
        s_mat = torch.randn((D, D), device="cuda", dtype=torch.float32)
        return q, k_idx, signs, gamma, Pi.contiguous(), s_mat, codebook

    def reference(q, k_idx, signs, gamma, pi, s_mat, codebook):
        centroids = codebook[k_idx.long()]
        mse_keys = centroids @ pi
        residual_keys = (
            (torch.sqrt(torch.tensor(torch.pi / 2.0, device="cuda")) / q.shape[1])
            * gamma[:, None]
            * (signs.float() @ s_mat)
        )
        return q @ (mse_keys + residual_keys).transpose(0, 1)

    correctness = []
    for name, b, s, d, c in [
        ("small", 2, 3, 2, 4),
        ("odd_batch_tokens", 3, 17, 7, 8),
        ("medium", 8, 257, 32, 16),
    ]:
        inputs = make_inputs(b, s, d, c)
        expected = reference(*inputs)
        actual = module.run_solve(*inputs)
        torch.cuda.synchronize()
        diff = (actual - expected).abs()
        correctness.append({
            "name": name,
            "shape": [b, s, d, c],
            "passed": bool(torch.allclose(actual, expected, rtol=2e-4, atol=2e-4)),
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
    for b, s, d, c in [(32, 32768, 128, 16)]:
        inputs = make_inputs(b, s, d, c)
        for _ in range(10):
            module.run_solve(*inputs)
            reference(*inputs)
        torch.cuda.synchronize()
        custom_ms = measure(lambda: module.run_solve(*inputs), 50)
        reference_ms = measure(lambda: reference(*inputs), 20)
        benchmark.append({
            "B": b,
            "S": s,
            "D": d,
            "C": c,
            "custom_ms": custom_ms,
            "reference_ms": reference_ms,
            "speedup": reference_ms / custom_ms,
        })

    return {
        "device": device,
        "correctness": correctness,
        "benchmark": benchmark,
        "notes": [
            "Scores are raw Q K_tilde^T values; no scaling or softmax is applied.",
            "Reference materializes dequantized keys; the custom kernel contracts transformed queries directly with compressed keys.",
        ],
    }


@app.local_entrypoint()
def main():
    kernel = REPO_ROOT / "kernels/attention/turboquant_attention.cu"
    result = benchmark_kernel.remote(base64.b64encode(kernel.read_bytes()).decode())
    timestamp = time.strftime("run_%Y%m%d_%H%M%S")
    out_dir = LOG_DIR / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "benchmark.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"Saved results to {out_dir}")
