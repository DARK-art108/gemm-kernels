#include <cassert>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <torch/torch.h>
#include "utils.cuh"

/*
Warp-level tiling GEMM kernel.
Hierarchy: Block (BM x BN) → Warps (WM x WN) → Warp Subtiles (WSUBM x WSUBN) → Thread Tiles (TM x TN)
*/
template <const int BM, const int BN, const int BK, const int row_stride_a, const int row_stride_b>
__device__ void load_from_gmem(int num_cols_b, int num_cols_a,
                               const float *matrix_a, const float *matrix_b,
                               float *tile_a, float *tile_b,
                               int inner_row_a, int inner_col_a,
                               int inner_row_b, int inner_col_b)
{
    for (uint offset = 0; offset + row_stride_a <= BM; offset += row_stride_a)
    {
        const float4 tmp_a = reinterpret_cast<const float4 *>(
            &matrix_a[(inner_row_a + offset) * num_cols_a + inner_col_a * 4])[0];
        tile_a[(inner_col_a * 4 + 0) * BM + inner_row_a + offset] = tmp_a.x;
        tile_a[(inner_col_a * 4 + 1) * BM + inner_row_a + offset] = tmp_a.y;
        tile_a[(inner_col_a * 4 + 2) * BM + inner_row_a + offset] = tmp_a.z;
        tile_a[(inner_col_a * 4 + 3) * BM + inner_row_a + offset] = tmp_a.w;
    }

    for (uint offset = 0; offset + row_stride_b <= BK; offset += row_stride_b)
    {
        reinterpret_cast<float4 *>(
            &tile_b[(inner_row_b + offset) * BN + inner_col_b * 4])[0] =
            reinterpret_cast<const float4 *>(
                &matrix_b[(inner_row_b + offset) * num_cols_b + inner_col_b * 4])[0];
    }
}

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WMITER, const int WNITER, const int WSUBM, const int WSUBN,
          const int TM, const int TN>
__device__ void process_warp_tile(float *register_m, float *register_n, float *thread_results,
                                  const float *tile_a, const float *tile_b,
                                  const uint warp_row, const uint warp_col,
                                  const uint thread_row_in_warp, const uint thread_col_in_warp)
{
    for (uint dot_idx = 0; dot_idx < BK; ++dot_idx)
    {
        for (uint wsub_row_idx = 0; wsub_row_idx < WMITER; ++wsub_row_idx)
        {
            for (uint i = 0; i < TM; ++i)
            {
                register_m[wsub_row_idx * TM + i] =
                    tile_a[(dot_idx * BM) + warp_row * WM + wsub_row_idx * WSUBM +
                           thread_row_in_warp * TM + i];
            }
        }

        for (uint wsub_col_idx = 0; wsub_col_idx < WNITER; ++wsub_col_idx)
        {
            for (uint i = 0; i < TN; ++i)
            {
                register_n[wsub_col_idx * TN + i] =
                    tile_b[(dot_idx * BN) + warp_col * WN + wsub_col_idx * WSUBN +
                           thread_col_in_warp * TN + i];
            }
        }

        for (uint wsub_row_idx = 0; wsub_row_idx < WMITER; ++wsub_row_idx)
        {
            for (uint wsub_col_idx = 0; wsub_col_idx < WNITER; ++wsub_col_idx)
            {
                for (uint res_idx_m = 0; res_idx_m < TM; ++res_idx_m)
                {
                    for (uint res_idx_n = 0; res_idx_n < TN; ++res_idx_n)
                    {
                        thread_results[(wsub_row_idx * TM + res_idx_m) * (WNITER * TN) +
                                       (wsub_col_idx * TN) + res_idx_n] +=
                            register_m[wsub_row_idx * TM + res_idx_m] *
                            register_n[wsub_col_idx * TN + res_idx_n];
                    }
                }
            }
        }
    }
}

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    sgemm_warptiling_kernel(int num_rows_a, int num_cols_b, int num_cols_a,
                            float alpha, const float *matrix_a, const float *matrix_b,
                            float beta, float *matrix_c)
{
    const uint block_row = blockIdx.y;
    const uint block_col = blockIdx.x;

    const uint warp_idx = threadIdx.x / WARP_SIZE;
    const uint warp_col = warp_idx % (BN / WN);
    const uint warp_row = warp_idx / (BN / WN);

    constexpr uint WMITER = (WM * WN) / (WARP_SIZE * TM * TN * WNITER);
    constexpr uint WSUBM = WM / WMITER;
    constexpr uint WSUBN = WN / WNITER;

    const uint thread_idx_in_warp = threadIdx.x % WARP_SIZE;
    const uint thread_col_in_warp = thread_idx_in_warp % (WSUBN / TN);
    const uint thread_row_in_warp = thread_idx_in_warp / (WSUBN / TN);

    __shared__ float tile_a[BM * BK];
    __shared__ float tile_b[BK * BN];

    matrix_a += block_row * BM * num_cols_a;
    matrix_b += block_col * BN;
    matrix_c += (block_row * BM + warp_row * WM) * num_cols_b + block_col * BN + warp_col * WN;

    const uint inner_row_a = threadIdx.x / (BK / 4);
    const uint inner_col_a = threadIdx.x % (BK / 4);
    constexpr uint row_stride_a = (NUM_THREADS * 4) / BK;

    const uint inner_row_b = threadIdx.x / (BN / 4);
    const uint inner_col_b = threadIdx.x % (BN / 4);
    constexpr uint row_stride_b = NUM_THREADS / (BN / 4);

    float thread_results[WMITER * TM * WNITER * TN] = {0.0f};
    float register_m[WMITER * TM] = {0.0f};
    float register_n[WNITER * TN] = {0.0f};

    for (uint block_k_idx = 0; block_k_idx < num_cols_a; block_k_idx += BK)
    {
        load_from_gmem<BM, BN, BK, row_stride_a, row_stride_b>(
            num_cols_b, num_cols_a, matrix_a, matrix_b, tile_a, tile_b,
            inner_row_a, inner_col_a, inner_row_b, inner_col_b);

        __syncthreads();

        process_warp_tile<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM, TN>(
            register_m, register_n, thread_results, tile_a, tile_b,
            warp_row, warp_col, thread_row_in_warp, thread_col_in_warp);

        matrix_a += BK;
        matrix_b += BK * num_cols_b;

        __syncthreads();
    }

    for (uint wsub_row_idx = 0; wsub_row_idx < WMITER; ++wsub_row_idx)
    {
        for (uint wsub_col_idx = 0; wsub_col_idx < WNITER; ++wsub_col_idx)
        {
            float *matrix_c_interim = matrix_c + (wsub_row_idx * WSUBM) * num_cols_b +
                                      wsub_col_idx * WSUBN;

            for (uint res_idx_m = 0; res_idx_m < TM; res_idx_m += 1)
            {
                for (uint res_idx_n = 0; res_idx_n < TN; res_idx_n += 4)
                {
                    float4 tmp_c = reinterpret_cast<float4 *>(
                        &matrix_c_interim[(thread_row_in_warp * TM + res_idx_m) * num_cols_b +
                                          thread_col_in_warp * TN + res_idx_n])[0];

                    const int res_idx = (wsub_row_idx * TM + res_idx_m) * (WNITER * TN) +
                                        wsub_col_idx * TN + res_idx_n;
                    tmp_c.x = alpha * thread_results[res_idx + 0] + beta * tmp_c.x;
                    tmp_c.y = alpha * thread_results[res_idx + 1] + beta * tmp_c.y;
                    tmp_c.z = alpha * thread_results[res_idx + 2] + beta * tmp_c.z;
                    tmp_c.w = alpha * thread_results[res_idx + 3] + beta * tmp_c.w;

                    reinterpret_cast<float4 *>(
                        &matrix_c_interim[(thread_row_in_warp * TM + res_idx_m) * num_cols_b +
                                          thread_col_in_warp * TN + res_idx_n])[0] = tmp_c;
                }
            }
        }
    }
}

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int NUM_THREADS>
void sgemm_warptiling(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                      torch::Tensor &output_matrix, float alpha, float beta)
{
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
    TORCH_CHECK(output_matrix.device().is_cuda(), "Matrix C must be on CUDA device");
    TORCH_CHECK(output_matrix.dtype() == torch::kFloat32, "Matrix C must be float32");
    TORCH_CHECK(output_matrix.size(0) == num_rows_a && output_matrix.size(1) == num_cols_b,
                "Matrix C must be MxN");

    TORCH_CHECK(num_rows_a % BM == 0, "Matrix A rows must be multiple of ", BM);
    TORCH_CHECK(num_cols_a % BK == 0, "Matrix A cols must be multiple of ", BK);
    TORCH_CHECK(num_cols_b % BN == 0, "Matrix B cols must be multiple of ", BN);

    constexpr int WMITER = (WM * WN) / (WARP_SIZE * TM * TN * WNITER);
    constexpr int WSUBM = WM / WMITER;
    constexpr int WSUBN = WN / WNITER;
    static_assert(WMITER * WSUBM == WM, "WMITER * WSUBM must equal WM");
    static_assert(WNITER * WSUBN == WN, "WNITER * WSUBN must equal WN");
    static_assert((BM % WM == 0) && (BN % WN == 0), "Block tile must be divisible by warp tile");
    static_assert((WSUBM % TM == 0) && (WSUBN % TN == 0), "Warp subtile must be divisible by thread tile");

    const float *d_matrix_a = matrix_a.data_ptr<float>();
    const float *d_matrix_b = matrix_b.data_ptr<float>();
    float *d_output_matrix = output_matrix.data_ptr<float>();

    dim3 block_dim(NUM_THREADS);
    dim3 grid_dim(ceil_div(num_cols_b, BN), ceil_div(num_rows_a, BM));

    sgemm_warptiling_kernel<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
        <<<grid_dim, block_dim>>>(
            num_rows_a, num_cols_b, num_cols_a,
            alpha, d_matrix_a, d_matrix_b, beta, d_output_matrix);
}

void sgemm_warptiling_default(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                              torch::Tensor &output_matrix, float alpha, float beta)
{
    sgemm_warptiling<128, 128, 16, 64, 64, 4, 8, 4, 128>(
        matrix_a, matrix_b, output_matrix, alpha, beta);
}

template void sgemm_warptiling<128, 128, 16, 64, 64, 4, 8, 4, 128>(
    const torch::Tensor &, const torch::Tensor &, torch::Tensor &, float, float);

template void sgemm_warptiling<128, 128, 16, 64, 32, 2, 8, 4, 256>(
    const torch::Tensor &, const torch::Tensor &, torch::Tensor &, float, float);

template void sgemm_warptiling<64, 64, 16, 32, 32, 2, 4, 4, 64>(
    const torch::Tensor &, const torch::Tensor &, torch::Tensor &, float, float);