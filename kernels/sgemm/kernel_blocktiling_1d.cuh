#include "utils.cuh"
#include <cassert>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
/*
This is a 1D kernel with block tiling.
import custom utils.cuh
*/

/*

============================================================
PERFORMANCE BENCHMARK (Square Matrices, alpha=1.0, beta=0.0)
============================================================
Size (M=N=K)    | Custom (ms)  | Custom TFLOPS  | cuBLAS (ms)  | cuBLAS TFLOPS
| % cuBLAS
----------------------------------------------------------------------------------------
512             | 0.375        | 0.72           | 0.132        | 2.03 | 35.3 %
1024            | 2.443        | 0.88           | 0.752        | 2.86 | 30.8 %
2048            | 10.019       | 1.71           | 4.254        | 4.04 | 42.5 %
4096            | 82.786       | 1.66           | 34.473       | 3.99 | 41.6 %
============================================================
Benchmark complete!



*/

template <const int BM, const int BN, const int BK, const int TM>

// C = α*(A@B)+β*C
__global__ void sgemm_blocktiling_1d_kernel(int num_rows_a, int num_cols_b,
                                            int num_cols_a, float alpha,
                                            const float *matrix_a,
                                            const float *matrix_b, float beta,
                                            float *matrix_c) {
  const uint block_row = blockIdx.x;
  const uint block_col = blockIdx.y;

  __shared__ float tile_a[BM * BK];
  __shared__ float tile_b[BK * BN];
  const uint thread_row = threadIdx.x / BN;
  const uint thread_col = threadIdx.x % BN;
  const int global_row = block_row * BM + thread_row * TM;
  const int global_col = block_col * BN + thread_col;
  matrix_a += block_row * BM * num_cols_a;
  matrix_b += block_col * BN;
  matrix_c += block_row * BM * num_cols_b + block_col * BN;
  float thread_results[TM] = {0.0f}; // multiple accumulator's per thread

  for (int tile_idx = 0; tile_idx < num_cols_a; tile_idx += BK) {

    // so we will load each tile from mat a into SRAM with bound checking
    //  one thread loads one element from A
    const uint a_row = threadIdx.x / BK;
    const uint a_col = threadIdx.x % BK;
    if ((block_row * BM + a_row) < num_rows_a &&
        (tile_idx + a_col) < num_cols_a) {
      tile_a[a_row * BK + a_col] = matrix_a[a_row * num_cols_a + a_col];

    } else {
      tile_a[a_row * BK + a_col] = 0.0f;
    }

    // same for Bmat
    const uint b_row = threadIdx.x / BN;
    const uint b_col = threadIdx.x % BN;
    if ((tile_idx + b_row) < num_cols_a &&
        (block_col * BN + b_col) < num_cols_b) {
      tile_b[b_row * BN + b_col] = matrix_b[b_row * num_cols_b + b_col];
    } else {
      tile_b[b_row * BN + b_col] = 0.0f;
    }

    __syncthreads();

    matrix_a += BK;
    matrix_b += BK * num_cols_b;

    for (uint dot_idx = 0; dot_idx < BK; ++dot_idx) {
      float b_tmp = tile_b[dot_idx * BN + thread_col];
      for (uint res_idx = 0; res_idx < TM; ++res_idx) {
        thread_results[res_idx] +=
            tile_a[(thread_row * TM + res_idx) * BK + dot_idx] * b_tmp;
      }
    }
    __syncthreads();
  }

  for (uint res_idx = 0; res_idx < TM; ++res_idx) {
    int row = global_row + res_idx;
    if (row < num_rows_a && global_col < num_cols_b) {
      matrix_c[(thread_row * TM + res_idx) * num_cols_b + thread_col] =
          alpha * thread_results[res_idx] +
          beta *
              matrix_c[(thread_row * TM + res_idx) * num_cols_b + thread_col];
    }
  }
}

void sgemm_blocktiling_1d(const torch::Tensor &matrix_a,
                          const torch::Tensor &matrix_b,
                          torch::Tensor &output_matrix, float alpha,
                          float beta) {
  // Validate inputs
  TORCH_CHECK(matrix_a.device().is_cuda(), "Matrix A must be on CUDA device");
  TORCH_CHECK(matrix_b.device().is_cuda(), "Matrix B must be on CUDA device");
  TORCH_CHECK(matrix_a.dtype() == torch::kFloat32, "Matrix A must be float32");
  TORCH_CHECK(matrix_b.dtype() == torch::kFloat32, "Matrix B must be float32");
  TORCH_CHECK(matrix_a.dim() == 2, "Matrix A must be 2D");
  TORCH_CHECK(matrix_b.dim() == 2, "Matrix B must be 2D");

  const int num_rows_a = static_cast<int>(matrix_a.size(0));
  const int num_cols_a = static_cast<int>(matrix_a.size(1));
  const int num_cols_b = static_cast<int>(matrix_b.size(1));

  TORCH_CHECK(matrix_b.size(0) == num_cols_a,
              "Matrix dimensions must match: A is MxK, B must be KxN");

  TORCH_CHECK(output_matrix.device().is_cuda(),
              "Matrix C must be on CUDA device");
  TORCH_CHECK(output_matrix.dtype() == torch::kFloat32,
              "Matrix C must be float32");
  TORCH_CHECK(output_matrix.size(0) == num_rows_a &&
                  output_matrix.size(1) == num_cols_b,
              "Matrix C must be MxN");

  // Get raw device pointers
  const float *d_matrix_a = matrix_a.data_ptr<float>();
  const float *d_matrix_b = matrix_b.data_ptr<float>();
  auto *d_output_matrix = output_matrix.data_ptr<float>();

  // Template parameters for kernel
  constexpr int BM = 64;
  constexpr int BN = 64;
  constexpr int BK = 8;
  constexpr int TM = 8;

  // Number of threads = (BM / TM) * BN = (64 / 8) * 64 = 512 threads per block
  dim3 block_dim((BM / TM) * BN);
  dim3 grid_dim(ceil_div(num_rows_a, BM), ceil_div(num_cols_b, BN));

  sgemm_blocktiling_1d_kernel<BM, BN, BK, TM>
      <<<grid_dim, block_dim>>>(num_rows_a, num_cols_b, num_cols_a, alpha,
                                d_matrix_a, d_matrix_b, beta, d_output_matrix);
}
