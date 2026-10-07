from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path

import modal


app = modal.App("sgemm-blocktiling-2d-profile")

CUDA_IMAGE = "pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel"
REPO_ROOT = Path(__file__).resolve().parents[4]
LOG_DIR = Path(__file__).resolve().parent / "logs" / "kernel_blocktiling_2d"

_NSYS_INSTALL = (
    "bash -lc 'set -e; export DEBIAN_FRONTEND=noninteractive; "
    "apt-get update -qq; "
    "apt-get install -y -qq curl gnupg ca-certificates; "
    "curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub "
    "| gpg --dearmor > /usr/share/keyrings/nvidia-devtools-keyring.gpg; "
    "DIST=ubuntu$(. /etc/lsb-release && echo \"$DISTRIB_RELEASE\" | tr -d .); "
    "ARCH=$(dpkg --print-architecture); "
    "echo \"deb [signed-by=/usr/share/keyrings/nvidia-devtools-keyring.gpg] https://developer.download.nvidia.com/devtools/repos/${DIST}/${ARCH}/ /\" "
    "> /etc/apt/sources.list.d/nvidia-devtools.list; "
    "apt-get update -qq; "
    "apt-get install -y -qq nsight-systems-cli'"
)

image = (
    modal.Image.from_registry(CUDA_IMAGE)
    .run_commands(_NSYS_INSTALL)
    .env({
        "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu",
        "TORCH_CUDA_ARCH_LIST": "9.0a",
    })
)


def _encode_file(path: Path) -> dict | None:
    if not path.exists():
        return None
    data = path.read_bytes()
    return {
        "name": path.name,
        "size_bytes": len(data),
        "content_b64": base64.b64encode(data).decode("ascii"),
    }


BENCH_SCRIPT = r'''
from __future__ import annotations

import argparse
import ctypes
import json
import os
import tempfile
import time
from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


CPP_SOURCE = r"""
#include <torch/extension.h>

void sgemm_blocktiling_2d(const torch::Tensor &matrix_a,
                          const torch::Tensor &matrix_b,
                          torch::Tensor &output_matrix,
                          float alpha,
                          float beta);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sgemm_blocktiling_2d", &sgemm_blocktiling_2d, "SGEMM 2D Block Tiling");
}
"""

CUDA_SOURCE = r"""
#include "kernel_blocktiling_2d.cuh"
"""


def cuda_profiler_start() -> None:
    for lib in ("libcudart.so", "libcudart.so.12"):
        try:
            cudart = ctypes.CDLL(lib)
            err = cudart.cudaProfilerStart()
            if err != 0:
                raise RuntimeError(f"cudaProfilerStart returned {err}")
            return
        except OSError:
            continue
    raise RuntimeError("Unable to load libcudart for cudaProfilerStart")


def cuda_profiler_stop() -> None:
    for lib in ("libcudart.so", "libcudart.so.12"):
        try:
            cudart = ctypes.CDLL(lib)
            err = cudart.cudaProfilerStop()
            if err != 0:
                raise RuntimeError(f"cudaProfilerStop returned {err}")
            return
        except OSError:
            continue
    raise RuntimeError("Unable to load libcudart for cudaProfilerStop")


def load_module():
    include_dir = Path(os.environ["SGEMM_INCLUDE_DIR"])
    build_dir = Path(tempfile.mkdtemp(prefix="sgemm_2d_build_"))
    return load_inline(
        name=f"sgemm_blocktiling_2d_ext_{os.getpid()}",
        cpp_sources=[CPP_SOURCE],
        cuda_sources=[CUDA_SOURCE],
        extra_include_paths=[str(include_dir)],
        extra_cuda_cflags=["-O3", "-lineinfo"],
        build_directory=str(build_dir),
        verbose=False,
    )


def run_kernel(module, a, b, c, alpha=1.0, beta=0.0):
    module.sgemm_blocktiling_2d(a, b, c, float(alpha), float(beta))


def measure_ms(fn, iters: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def correctness(module) -> list[dict]:
    tests = [
        ("square_256", 256, 256, 256, 1.0, 0.0),
        ("square_alpha_beta", 512, 512, 512, 1.5, 0.5),
        ("rectangular", 512, 768, 256, 1.0, 0.0),
        ("edge_mn_k_multiple_of_bk", 1000, 832, 504, 1.0, 0.0),
    ]
    out = []
    for name, m, n, k, alpha, beta in tests:
        torch.manual_seed(123)
        a = torch.randn((m, k), device="cuda", dtype=torch.float32)
        b = torch.randn((k, n), device="cuda", dtype=torch.float32)
        c0 = torch.randn((m, n), device="cuda", dtype=torch.float32) if beta != 0.0 else torch.zeros((m, n), device="cuda", dtype=torch.float32)
        c = c0.clone()
        ref = alpha * torch.matmul(a, b) + beta * c0
        run_kernel(module, a, b, c, alpha, beta)
        torch.cuda.synchronize()
        diff = (c - ref).abs()
        max_err = float(diff.max().item())
        mean_err = float(diff.mean().item())
        passed = bool(torch.allclose(c, ref, rtol=1e-3, atol=1e-3))
        out.append({
            "name": name,
            "M": m,
            "N": n,
            "K": k,
            "alpha": alpha,
            "beta": beta,
            "passed": passed,
            "max_abs_err": max_err,
            "mean_abs_err": mean_err,
        })
    return out


def benchmark(module) -> list[dict]:
    sizes = [512, 1024, 2048, 4096, 8192]
    rows = []
    for size in sizes:
        m = n = k = size
        torch.manual_seed(456)
        a = torch.randn((m, k), device="cuda", dtype=torch.float32)
        b = torch.randn((k, n), device="cuda", dtype=torch.float32)
        c_custom = torch.zeros((m, n), device="cuda", dtype=torch.float32)
        c_cublas = torch.zeros((m, n), device="cuda", dtype=torch.float32)
        for _ in range(20):
            run_kernel(module, a, b, c_custom)
            torch.matmul(a, b, out=c_cublas)
        torch.cuda.synchronize()

        iters = 100 if size <= 1024 else 50 if size <= 2048 else 20 if size <= 4096 else 8
        custom_ms = measure_ms(lambda: run_kernel(module, a, b, c_custom), iters)
        cublas_ms = measure_ms(lambda: torch.matmul(a, b, out=c_cublas), iters)
        flops = 2.0 * m * n * k
        custom_tflops = flops / (custom_ms * 1e-3) / 1e12
        cublas_tflops = flops / (cublas_ms * 1e-3) / 1e12
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


def profile_once(module, size: int, label: str) -> None:
    m = n = k = size
    torch.manual_seed(789)
    a = torch.randn((m, k), device="cuda", dtype=torch.float32)
    b = torch.randn((k, n), device="cuda", dtype=torch.float32)
    c = torch.zeros((m, n), device="cuda", dtype=torch.float32)
    for _ in range(10):
        run_kernel(module, a, b, c)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push(label)
    cuda_profiler_start()
    run_kernel(module, a, b, c)
    torch.cuda.synchronize()
    cuda_profiler_stop()
    torch.cuda.nvtx.range_pop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["benchmark", "ncu", "nsys", "torch_perfetto"], required=True)
    parser.add_argument("--profile-size", type=int, default=4096)
    args = parser.parse_args()

    module = load_module()
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

    if args.mode == "benchmark":
        result = {
            "device": device,
            "correctness": correctness(module),
            "benchmark": benchmark(module),
            "notes": [
                "cuBLAS comparison is FP32 with torch.backends.cuda.matmul.allow_tf32=False.",
                "K-tail correctness is intentionally not patched in the kernel; profiled benchmark sizes use K divisible by BK=8.",
            ],
        }
        print("RESULT_JSON_START")
        print(json.dumps(result, indent=2, sort_keys=True))
        print("RESULT_JSON_END")
    elif args.mode in {"ncu", "nsys"}:
        profile_once(module, args.profile_size, f"sgemm_blocktiling_2d_{args.profile_size}")
        print(json.dumps({"device": device, "profile_size": args.profile_size, "mode": args.mode}))
    else:
        trace_path = Path(os.environ["SGEMM_TRACE_PATH"])
        m = n = k = args.profile_size
        torch.manual_seed(987)
        a = torch.randn((m, k), device="cuda", dtype=torch.float32)
        b = torch.randn((k, n), device="cuda", dtype=torch.float32)
        c = torch.zeros((m, n), device="cuda", dtype=torch.float32)
        for _ in range(10):
            run_kernel(module, a, b, c)
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        ) as prof:
            torch.cuda.nvtx.range_push(f"sgemm_blocktiling_2d_{args.profile_size}")
            run_kernel(module, a, b, c)
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()
        prof.export_chrome_trace(str(trace_path))
        print(json.dumps({
            "device": device,
            "profile_size": args.profile_size,
            "mode": args.mode,
            "trace_path": str(trace_path),
            "trace_size_bytes": trace_path.stat().st_size,
        }))


if __name__ == "__main__":
    main()
'''


@app.function(image=image, gpu="H100", timeout=60 * 40)
def run_remote(kernel_b64: str, utils_b64: str, profile_size: int) -> dict:
    import shutil
    import subprocess
    import sys
    import tempfile

    root = Path(tempfile.mkdtemp(prefix="sgemm_2d_modal_"))
    inc = root / "include"
    inc.mkdir()
    (inc / "kernel_blocktiling_2d.cuh").write_bytes(base64.b64decode(kernel_b64))
    (inc / "utils.cuh").write_bytes(base64.b64decode(utils_b64))
    bench_py = root / "bench_2d.py"
    bench_py.write_text(BENCH_SCRIPT, encoding="utf-8")

    env = dict(os.environ)
    env["SGEMM_INCLUDE_DIR"] = str(inc)
    env["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:" + env.get("LD_LIBRARY_PATH", "")
    env["TORCH_CUDA_ARCH_LIST"] = "9.0a"

    result = {"benchmark": {}, "ncu": {}, "nsys": {}, "artifacts": {}}

    bench_proc = subprocess.run(
        [sys.executable, str(bench_py), "--mode", "benchmark", "--profile-size", str(profile_size)],
        cwd=str(root), env=env, capture_output=True, text=True, timeout=60 * 25,
    )
    result["benchmark"] = {
        "returncode": bench_proc.returncode,
        "stdout": bench_proc.stdout,
        "stderr": bench_proc.stderr,
    }

    ncu = shutil.which("ncu") or "/usr/local/cuda/bin/ncu"
    if Path(ncu).exists():
        ncu_report = root / "profile.ncu-rep"
        ncu_cmd = [
            ncu,
            "--target-processes", "all",
            "--profile-from-start", "off",
            "--clock-control", "none",
            "--kernel-name-base", "demangled",
            "--set", "full",
            "--force-overwrite",
            "--export", str(ncu_report),
            sys.executable, str(bench_py), "--mode", "ncu", "--profile-size", str(profile_size),
        ]
        ncu_proc = subprocess.run(
            ncu_cmd, cwd=str(root), env=env, capture_output=True, text=True, timeout=60 * 20,
        )
        result["ncu"] = {
            "returncode": ncu_proc.returncode,
            "stdout": ncu_proc.stdout,
            "stderr": ncu_proc.stderr,
            "report": _encode_file(ncu_report),
        }
    else:
        result["ncu"] = {"returncode": None, "stdout": "", "stderr": "ncu not found", "report": None}

    nsys = shutil.which("nsys")
    if nsys is None:
        for p in sorted(Path("/opt/nvidia").rglob("bin/nsys"), reverse=True):
            if "nsight-compute" not in str(p):
                nsys = str(p)
                break

    if nsys:
        nsys_base = root / "profile_nsys"
        perfetto_proto = root / "profile_perfetto.proto"
        nsys_cmd = [
            nsys,
            "profile",
            "--trace=cuda,nvtx,osrt",
            "--capture-range=cudaProfilerApi",
            "--capture-range-end=stop",
            "--force-overwrite=true",
            "--output", str(nsys_base),
            sys.executable, str(bench_py), "--mode", "nsys", "--profile-size", str(profile_size),
        ]
        nsys_proc = subprocess.run(
            nsys_cmd, cwd=str(root), env=env, capture_output=True, text=True, timeout=60 * 20,
        )
        nsys_rep = next(iter(root.glob("profile_nsys*.nsys-rep")), None)
        stats_stdout = ""
        stats_stderr = ""
        export_stdout = ""
        export_stderr = ""
        if nsys_rep and nsys_rep.exists():
            stats_proc = subprocess.run(
                [nsys, "stats", "--force-export=true", "--report", "cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,nvtx_sum", str(nsys_rep)],
                cwd=str(root), env=env, capture_output=True, text=True, timeout=60 * 5,
            )
            stats_stdout = stats_proc.stdout
            stats_stderr = stats_proc.stderr
            export_proc = subprocess.run(
                [nsys, "export", "--type=perfetto", "--force-overwrite=true", "--output", str(perfetto_proto), str(nsys_rep)],
                cwd=str(root), env=env, capture_output=True, text=True, timeout=60 * 5,
            )
            export_stdout = export_proc.stdout
            export_stderr = export_proc.stderr
        result["nsys"] = {
            "returncode": nsys_proc.returncode,
            "stdout": nsys_proc.stdout,
            "stderr": nsys_proc.stderr,
            "stats_stdout": stats_stdout,
            "stats_stderr": stats_stderr,
            "export_stdout": export_stdout,
            "export_stderr": export_stderr,
            "rep_size_bytes": nsys_rep.stat().st_size if nsys_rep and nsys_rep.exists() else 0,
            "perfetto": _encode_file(perfetto_proto),
        }
    else:
        result["nsys"] = {"returncode": None, "stdout": "", "stderr": "nsys not found"}

    trace_path = root / "torch_perfetto_trace.json"
    trace_env = dict(env)
    trace_env["SGEMM_TRACE_PATH"] = str(trace_path)
    trace_proc = subprocess.run(
        [sys.executable, str(bench_py), "--mode", "torch_perfetto", "--profile-size", str(profile_size)],
        cwd=str(root), env=trace_env, capture_output=True, text=True, timeout=60 * 10,
    )
    result["torch_perfetto"] = {
        "returncode": trace_proc.returncode,
        "stdout": trace_proc.stdout,
        "stderr": trace_proc.stderr,
        "trace": _encode_file(trace_path),
    }

    return result


def _save_artifact(encoded: dict | None, out_dir: Path) -> str | None:
    if not encoded:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / encoded["name"]
    path.write_bytes(base64.b64decode(encoded["content_b64"]))
    return str(path)


def _extract_benchmark_json(stdout: str) -> dict | None:
    start = stdout.find("RESULT_JSON_START")
    end = stdout.find("RESULT_JSON_END")
    if start < 0 or end < 0:
        return None
    payload = stdout[start + len("RESULT_JSON_START"):end].strip()
    return json.loads(payload)


@app.local_entrypoint()
def main(profile_size: int = 4096):
    kernel_path = REPO_ROOT / "kernels" / "sgemm" / "kernel_blocktiling_2d.cuh"
    utils_path = REPO_ROOT / "kernels" / "common" / "utils.cuh"

    print(f"Running {kernel_path.name} on Modal H100; profiler size={profile_size}")
    result = run_remote.remote(
        base64.b64encode(kernel_path.read_bytes()).decode("ascii"),
        base64.b64encode(utils_path.read_bytes()).decode("ascii"),
        profile_size,
    )

    timestamp = time.strftime("run_%Y%m%d_%H%M%S")
    out_dir = LOG_DIR / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "raw_result.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    parsed = _extract_benchmark_json(result.get("benchmark", {}).get("stdout", ""))
    if parsed:
        (out_dir / "benchmark.json").write_text(json.dumps(parsed, indent=2, sort_keys=True), encoding="utf-8")

    ncu_path = _save_artifact(result.get("ncu", {}).get("report"), out_dir)
    perfetto_path = _save_artifact(result.get("nsys", {}).get("perfetto"), out_dir)
    torch_perfetto_path = _save_artifact(result.get("torch_perfetto", {}).get("trace"), out_dir)

    summary = {
        "out_dir": str(out_dir),
        "benchmark_returncode": result.get("benchmark", {}).get("returncode"),
        "ncu_returncode": result.get("ncu", {}).get("returncode"),
        "nsys_returncode": result.get("nsys", {}).get("returncode"),
        "ncu_report": ncu_path,
        "perfetto_proto": perfetto_path,
        "torch_perfetto_trace": torch_perfetto_path,
        "nsys_rep_size_bytes_remote": result.get("nsys", {}).get("rep_size_bytes"),
        "benchmark": parsed,
        "ncu_stdout_tail": result.get("ncu", {}).get("stdout", "")[-4000:],
        "ncu_stderr_tail": result.get("ncu", {}).get("stderr", "")[-4000:],
        "nsys_stdout_tail": result.get("nsys", {}).get("stdout", "")[-4000:],
        "nsys_stderr_tail": result.get("nsys", {}).get("stderr", "")[-4000:],
        "nsys_stats": result.get("nsys", {}).get("stats_stdout", "")[-8000:],
        "nsys_stats_stderr_tail": result.get("nsys", {}).get("stats_stderr", "")[-4000:],
        "torch_perfetto_returncode": result.get("torch_perfetto", {}).get("returncode"),
        "torch_perfetto_stdout_tail": result.get("torch_perfetto", {}).get("stdout", "")[-4000:],
        "torch_perfetto_stderr_tail": result.get("torch_perfetto", {}).get("stderr", "")[-4000:],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"\nSaved detailed results under: {out_dir}")
