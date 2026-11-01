from __future__ import annotations

import base64
import json
import time
from pathlib import Path

import modal


app = modal.App("triton-flash-attention")
_this_file = Path(__file__).resolve()
REPO_ROOT = _this_file.parents[4] if len(_this_file.parents) > 4 else Path.cwd()
LOG_DIR = _this_file.parent / "logs" / "flash_attention_triton"
IMAGE = modal.Image.from_registry("pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel")


@app.function(image=IMAGE, gpu="H100", timeout=60 * 30)
def benchmark_kernel(kernel_b64: str) -> dict:
    import importlib.util
    import sys
    import tempfile

    import torch
    import torch.nn.functional as F

    source_path = Path(tempfile.mkdtemp(prefix="flash_attention_kernel_")) / "flash_attention.py"
    source_path.write_bytes(base64.b64decode(kernel_b64))
    spec = importlib.util.spec_from_file_location("flash_attention_kernel", source_path)
    kernel_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = kernel_module
    spec.loader.exec_module(kernel_module)
    solve = kernel_module.solve

    prop = torch.cuda.get_device_properties(0)
    device = {
        "name": prop.name,
        "sm_count": prop.multi_processor_count,
        "capability": f"{prop.major}.{prop.minor}",
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
        "triton": kernel_module.triton.__version__,
        "dtype": "float32",
    }
    torch.manual_seed(123)

    def make_inputs(m: int, n: int, d: int):
        q = torch.randn((m, d), device="cuda", dtype=torch.float32)
        k = torch.randn((n, d), device="cuda", dtype=torch.float32)
        v = torch.randn((n, d), device="cuda", dtype=torch.float32)
        return q, k, v

    def explicit_reference(q, k, v):
        scale = 1.0 / (q.shape[1] ** 0.5)
        probabilities = torch.softmax((q @ k.transpose(0, 1)) * scale, dim=-1)
        return probabilities @ v

    def sdpa_reference(q, k, v):
        return F.scaled_dot_product_attention(
            q[None, None, :, :],
            k[None, None, :, :],
            v[None, None, :, :],
            dropout_p=0.0,
        )[0, 0]

    correctness = []
    for name, m, n, d in [
        ("small_non_multiple", 7, 13, 8),
        ("odd_rows", 35, 67, 32),
        ("wide_head", 64, 129, 128),
    ]:
        q, k, v = make_inputs(m, n, d)
        out = torch.empty_like(q)
        solve(q, k, v, out)
        expected = explicit_reference(q, k, v)
        torch.cuda.synchronize()
        diff = (out - expected).abs()
        correctness.append({
            "name": name,
            "shape": [m, n, d],
            "passed": bool(torch.allclose(out, expected, rtol=2e-3, atol=2e-3)),
            "max_abs_err": float(diff.max().item()),
            "mean_abs_err": float(diff.mean().item()),
        })

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

    benchmark = []
    for m, n, d, iters in [
        (128, 512, 64, 100),
        (512, 2048, 128, 50),
        (1024, 4096, 128, 20),
    ]:
        q, k, v = make_inputs(m, n, d)
        out = torch.empty_like(q)
        for _ in range(10):
            solve(q, k, v, out)
            sdpa_reference(q, k, v)
        torch.cuda.synchronize()
        custom_ms = measure(lambda: solve(q, k, v, out), iters)
        sdpa_ms = measure(lambda: sdpa_reference(q, k, v), iters)
        benchmark.append({
            "M": m,
            "N": n,
            "D": d,
            "iters": iters,
            "custom_ms": custom_ms,
            "sdpa_ms": sdpa_ms,
            "speedup_vs_sdpa": sdpa_ms / custom_ms,
        })

    return {
        "device": device,
        "correctness": correctness,
        "benchmark": benchmark,
        "notes": [
            "The custom kernel computes scaled dot-product attention with online softmax and does not materialize the M x N score matrix.",
            "The performance reference is PyTorch scaled_dot_product_attention on the same FP32 inputs.",
            "The correctness reference explicitly evaluates softmax(Q @ K.T / sqrt(D)) @ V.",
        ],
    }


@app.local_entrypoint()
def main():
    kernel = REPO_ROOT / "kernels/attention/triton/flash_attention.py"
    result = benchmark_kernel.remote(base64.b64encode(kernel.read_bytes()).decode())
    timestamp = time.strftime("run_%Y%m%d_%H%M%S")
    out_dir = LOG_DIR / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "benchmark.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"Saved results to {out_dir}")
