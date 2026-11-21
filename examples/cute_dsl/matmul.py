import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


@cute.kernel
def matmul_kernel(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor):

    M, K = A.shape
    _, N = B.shape

    tx, ty, _ = cute.arch.thread_idx()
    bx, by, _ = cute.arch.block_idx()

    row = bx * 16 + tx
    col = by * 16 + ty

    if row < M and col < N:
        acc = cutlass.Float32(0.0)

        for k in range(K):
            acc += A[row, k] * B[k, col]

        C[row, col] = acc


@cute.jit
def matmul(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor):

    M = A.shape[0]
    N = B.shape[1]

    grid_x = cute.ceil_div(M, 16)
    grid_y = cute.ceil_div(N, 16)

    matmul_kernel(A, B, C).launch(
        grid=(grid_x, grid_y, 1),
        block=(16, 16, 1)
    )


A = torch.randn(64, 32, device="cuda", dtype=torch.float32)
B = torch.randn(32, 64, device="cuda", dtype=torch.float32)
C = torch.empty((64, 64), device="cuda", dtype=torch.float32)

a = from_dlpack(A)
b = from_dlpack(B)
c = from_dlpack(C)

compiled = cute.compile(matmul, a, b, c)
compiled(a, b, c)

torch.cuda.synchronize()

print(torch.allclose(C, A @ B))
