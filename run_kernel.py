import os
import modal

app = modal.App("sgemm-blocktiling-1d")

# Use official PyTorch devel image with CUDA 12.4 and cuDNN 9 (already cached)
image = modal.Image.from_registry("pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel")

@app.function(gpu="T4", image=image, timeout=600)
def benchmark_kernel(kernel_code: str, utils_code: str):
    import time
    import os
    import torch
    from torch.utils.cpp_extension import load_inline

    print("=" * 60)
    print("DEVICE INFORMATION")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    print("=" * 60)

    # Write headers to include directory
    inc_dir = "/root/include"
    os.makedirs(inc_dir, exist_ok=True)
    with open(f"{inc_dir}/utils.cuh", "w") as f:
        f.write(utils_code)
    with open(f"{inc_dir}/01_kernel_blocktiling_1d.cuh", "w") as f:
        f.write(kernel_code)

    cpp_source = """
    #include <torch/extension.h>

    void sgemm_blocktiling_1d(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                              torch::Tensor &output_matrix, float alpha, float beta);

    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        m.def("sgemm_blocktiling_1d", &sgemm_blocktiling_1d, "SGEMM 1D Block Tiling");
    }
    """

    cuda_source = """
    #include "01_kernel_blocktiling_1d.cuh"
    """

    print("Compiling CUDA kernel extension...")
    t0 = time.time()
    sgemm_module = load_inline(
        name="sgemm_blocktiling_ext",
        cpp_sources=[cpp_source],
        cuda_sources=[cuda_source],
        extra_include_paths=[inc_dir],
        extra_cuda_cflags=["-O3"],
        verbose=True
    )
    print(f"Compilation succeeded in {time.time() - t0:.2f}s!\n")

    results = []

    # 1. Correctness Tests
    print("=" * 60)
    print("CORRECTNESS VERIFICATION")
    print("=" * 60)
    test_configs = [
        {"name": "Square (Power of 2)", "M": 1024, "N": 1024, "K": 1024, "alpha": 1.0, "beta": 0.0},
        {"name": "Square with alpha/beta", "M": 1024, "N": 1024, "K": 1024, "alpha": 1.5, "beta": 0.5},
        {"name": "Rectangular (M != N != K)", "M": 512, "N": 256, "K": 1024, "alpha": 1.0, "beta": 0.0},
        {"name": "Arbitrary Non-Multiple of Tile", "M": 1000, "N": 800, "K": 500, "alpha": 1.0, "beta": 0.0},
    ]

    for cfg in test_configs:
        M, N, K = cfg["M"], cfg["N"], cfg["K"]
        alpha, beta = cfg["alpha"], cfg["beta"]

        torch.manual_seed(42)
        A = torch.randn((M, K), device="cuda", dtype=torch.float32)
        B = torch.randn((K, N), device="cuda", dtype=torch.float32)
        C_ref = torch.randn((M, N), device="cuda", dtype=torch.float32) if beta != 0.0 else torch.zeros((M, N), device="cuda", dtype=torch.float32)
        C_custom = C_ref.clone()

        # Compute Reference: C = alpha * (A @ B) + beta * C
        ref = alpha * torch.matmul(A, B) + beta * C_ref

        # Custom Kernel
        sgemm_module.sgemm_blocktiling_1d(A, B, C_custom, alpha, beta)
        torch.cuda.synchronize()

        max_err = (C_custom - ref).abs().max().item()
        mean_err = (C_custom - ref).abs().mean().item()
        allclose = torch.allclose(C_custom, ref, rtol=1e-3, atol=1e-3)
        status = "PASSED" if allclose else "FAILED"

        print(f"[{status}] {cfg['name']} (M={M}, N={N}, K={K}, alpha={alpha}, beta={beta})")
        print(f"         Max Abs Error: {max_err:.6e} | Mean Abs Error: {mean_err:.6e}")
        if not allclose:
            print("         WARNING: Output mismatch!")

    # 2. Performance Benchmarks
    print("\n" + "=" * 60)
    print("PERFORMANCE BENCHMARK (Square Matrices, alpha=1.0, beta=0.0)")
    print("=" * 60)
    print(f"{'Size (M=N=K)':<15} | {'Custom (ms)':<12} | {'Custom TFLOPS':<14} | {'cuBLAS (ms)':<12} | {'cuBLAS TFLOPS':<14} | {'% cuBLAS':<10}")
    print("-" * 88)

    benchmark_sizes = [512, 1024, 2048, 4096]
    for size in benchmark_sizes:
        M = N = K = size
        A = torch.randn((M, K), device="cuda", dtype=torch.float32)
        B = torch.randn((K, N), device="cuda", dtype=torch.float32)
        C_custom = torch.zeros((M, N), device="cuda", dtype=torch.float32)
        C_cublas = torch.zeros((M, N), device="cuda", dtype=torch.float32)

        # Warmup
        for _ in range(10):
            sgemm_module.sgemm_blocktiling_1d(A, B, C_custom, 1.0, 0.0)
            torch.matmul(A, B, out=C_cublas)
        torch.cuda.synchronize()

        # Timing custom kernel
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        num_iters = 50 if size <= 2048 else 20
        start_event.record()
        for _ in range(num_iters):
            sgemm_module.sgemm_blocktiling_1d(A, B, C_custom, 1.0, 0.0)
        end_event.record()
        torch.cuda.synchronize()
        custom_time_ms = start_event.elapsed_time(end_event) / num_iters

        # Timing cuBLAS
        start_event.record()
        for _ in range(num_iters):
            torch.matmul(A, B, out=C_cublas)
        end_event.record()
        torch.cuda.synchronize()
        cublas_time_ms = start_event.elapsed_time(end_event) / num_iters

        # FLOPs = 2 * M * N * K
        flops = 2.0 * M * N * K
        custom_tflops = (flops / (custom_time_ms * 1e-3)) / 1e12
        cublas_tflops = (flops / (cublas_time_ms * 1e-3)) / 1e12
        ratio = (custom_tflops / cublas_tflops) * 100.0

        print(f"{size:<15} | {custom_time_ms:<12.3f} | {custom_tflops:<14.2f} | {cublas_time_ms:<12.3f} | {cublas_tflops:<14.2f} | {ratio:<9.1f}%")
        results.append({
            "size": size,
            "custom_ms": custom_time_ms,
            "custom_tflops": custom_tflops,
            "cublas_ms": cublas_time_ms,
            "cublas_tflops": cublas_tflops,
            "efficiency": ratio
        })

    print("=" * 60)
    print("Benchmark complete!")
    return results

@app.local_entrypoint()
def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base_dir, "01_kernel_blocktiling_1d.cuh")) as f:
        kernel_code = f.read()
    with open(os.path.join(base_dir, "utils.cuh")) as f:
        utils_code = f.read()

    print("Starting Modal job for 01_kernel_blocktiling_1d.cuh...")
    benchmark_kernel.remote(kernel_code, utils_code)
