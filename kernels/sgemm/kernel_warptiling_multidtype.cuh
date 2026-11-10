#pragma once

#include <cassert>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <type_traits>
#include <torch/torch.h>

#include "utils.cuh"

// The arithmetic stays in FP32 even when the inputs and output use 16-bit
// values. Keeping the conversion in one place makes the compute loop easier
// to read and gives FP16 and BF16 the same accumulation behavior.
__device__ __forceinline__ float to_float(half value) {
    return __half2float(value);
}

__device__ __forceinline__ float to_float(nv_bfloat16 value) {
    return __bfloat162float(value);
}

__device__ __forceinline__ half from_float(float value, half) {
    return __float2half(value);
}

__device__ __forceinline__ nv_bfloat16 from_float(float value, nv_bfloat16) {
    return __float2bfloat16(value);
}

// Each vector contains two 16-bit values. The explicit helpers avoid relying
// on a type-punning cast when unpacking or storing a vector.
__device__ __forceinline__ void unpack_vector(half2 value, half &first, half &second) {
    first = __low2half(value);
    second = __high2half(value);
}

__device__ __forceinline__ void unpack_vector(nv_bfloat162 value,
                                             nv_bfloat16 &first,
                                             nv_bfloat16 &second) {
    first = value.x;
    second = value.y;
}

__device__ __forceinline__ half2 pack_vector(half first, half second) {
    return __halves2half2(first, second);
}

__device__ __forceinline__ nv_bfloat162 pack_vector(nv_bfloat16 first,
                                                   nv_bfloat16 second) {
    nv_bfloat162 value;
    value.x = first;
    value.y = second;
    return value;
}

template <typename InputType, const int BM, const int BN, const int BK,
          const int row_stride_a, const int row_stride_b>
__device__ void load_from_gmem(
    int num_cols_b,
    int num_cols_a,
    const InputType *matrix_a,
    const InputType *matrix_b,
    InputType *tile_a,
    InputType *tile_b,
    int inner_row_a,
    int inner_col_a,
    int inner_row_b,
    int inner_col_b) {
    using VecT = typename std::conditional<
        std::is_same<InputType, half>::value, half2, nv_bfloat162>::type;

    // A is transposed while entering shared memory. That layout gives the
    // compute loop contiguous access for each reduction step.
    for (uint offset = 0; offset < BM; offset += row_stride_a) {
        const VecT *src = reinterpret_cast<const VecT *>(
            &matrix_a[(inner_row_a + offset) * num_cols_a + inner_col_a * 2]);
        InputType first;
        InputType second;
        unpack_vector(*src, first, second);
        tile_a[(inner_col_a * 2 + 0) * BM + inner_row_a + offset] = first;
        tile_a[(inner_col_a * 2 + 1) * BM + inner_row_a + offset] = second;
    }

    // B keeps its row-major layout, so each vector can be copied directly
    // into the matching position in shared memory.
    for (uint offset = 0; offset < BK; offset += row_stride_b) {
        const VecT *src = reinterpret_cast<const VecT *>(
            &matrix_b[(inner_row_b + offset) * num_cols_b + inner_col_b * 2]);
        VecT *dst = reinterpret_cast<VecT *>(
            &tile_b[(inner_row_b + offset) * BN + inner_col_b * 2]);
        *dst = *src;
    }
}

template <typename InputType, const int BM, const int BN, const int BK,
          const int WM, const int WN, const int WMITER, const int WNITER,
          const int WSUBM, const int WSUBN, const int TM, const int TN>
__device__ void process_warp_tile(
    InputType *register_m,
    InputType *register_n,
    float *thread_results,
    const InputType *tile_a,
    const InputType *tile_b,
    uint warp_row,
    uint warp_col,
    uint thread_row_in_warp,
    uint thread_col_in_warp) {
    for (uint dot_idx = 0; dot_idx < BK; ++dot_idx) {
        for (uint wsub_row_idx = 0; wsub_row_idx < WMITER; ++wsub_row_idx) {
            for (uint i = 0; i < TM; ++i) {
                register_m[wsub_row_idx * TM + i] =
                    tile_a[dot_idx * BM + warp_row * WM + wsub_row_idx * WSUBM +
                           thread_row_in_warp * TM + i];
            }
        }

        for (uint wsub_col_idx = 0; wsub_col_idx < WNITER; ++wsub_col_idx) {
            for (uint i = 0; i < TN; ++i) {
                register_n[wsub_col_idx * TN + i] =
                    tile_b[dot_idx * BN + warp_col * WN + wsub_col_idx * WSUBN +
                           thread_col_in_warp * TN + i];
            }
        }

        for (uint wsub_row_idx = 0; wsub_row_idx < WMITER; ++wsub_row_idx) {
            for (uint wsub_col_idx = 0; wsub_col_idx < WNITER; ++wsub_col_idx) {
                for (uint res_idx_m = 0; res_idx_m < TM; ++res_idx_m) {
                    for (uint res_idx_n = 0; res_idx_n < TN; ++res_idx_n) {
                        const int result_idx =
                            (wsub_row_idx * TM + res_idx_m) * (WNITER * TN) +
                            wsub_col_idx * TN + res_idx_n;
                        thread_results[result_idx] +=
                            to_float(register_m[wsub_row_idx * TM + res_idx_m]) *
                            to_float(register_n[wsub_col_idx * TN + res_idx_n]);
                    }
                }
            }
        }
    }
}

template <typename InputType, const int BM, const int BN, const int BK,
          const int WM, const int WN, const int WNITER, const int TM,
          const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    sgemm_warptiling_multidtype_kernel(
        int num_rows_a,
        int num_cols_b,
        int num_cols_a,
        float alpha,
        const InputType *matrix_a,
        const InputType *matrix_b,
        float beta,
        InputType *matrix_c) {
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

    __shared__ InputType tile_a[BM * BK];
    __shared__ InputType tile_b[BK * BN];

    matrix_a += block_row * BM * num_cols_a;
    matrix_b += block_col * BN;
    matrix_c += (block_row * BM + warp_row * WM) * num_cols_b +
                block_col * BN + warp_col * WN;

    // Two 16-bit values are moved by each thread during the cooperative load.
    constexpr int VEC_SIZE = 2;
    const uint inner_row_a = threadIdx.x / (BK / VEC_SIZE);
    const uint inner_col_a = threadIdx.x % (BK / VEC_SIZE);
    constexpr uint row_stride_a = (NUM_THREADS * VEC_SIZE) / BK;

    const uint inner_row_b = threadIdx.x / (BN / VEC_SIZE);
    const uint inner_col_b = threadIdx.x % (BN / VEC_SIZE);
    constexpr uint row_stride_b = NUM_THREADS / (BN / VEC_SIZE);

    // Results are deliberately FP32. Only the final store rounds back to the
    // input type, which is more accurate than accumulating in FP16 or BF16.
    float thread_results[WMITER * TM * WNITER * TN] = {0.0f};
    InputType register_m[WMITER * TM] = {};
    InputType register_n[WNITER * TN] = {};

    for (uint block_k_idx = 0; block_k_idx < num_cols_a; block_k_idx += BK) {
        load_from_gmem<InputType, BM, BN, BK, row_stride_a, row_stride_b>(
            num_cols_b, num_cols_a, matrix_a, matrix_b, tile_a, tile_b,
            inner_row_a, inner_col_a, inner_row_b, inner_col_b);

        __syncthreads();

        process_warp_tile<InputType, BM, BN, BK, WM, WN, WMITER, WNITER,
                          WSUBM, WSUBN, TM, TN>(
            register_m, register_n, thread_results, tile_a, tile_b,
            warp_row, warp_col, thread_row_in_warp, thread_col_in_warp);

        matrix_a += BK;
        matrix_b += BK * num_cols_b;

        __syncthreads();
    }

    using VecT = typename std::conditional<
        std::is_same<InputType, half>::value, half2, nv_bfloat162>::type;
    constexpr int STORE_VEC_SIZE = 2;

    for (uint wsub_row_idx = 0; wsub_row_idx < WMITER; ++wsub_row_idx) {
        for (uint wsub_col_idx = 0; wsub_col_idx < WNITER; ++wsub_col_idx) {
            InputType *matrix_c_interim =
                matrix_c + (wsub_row_idx * WSUBM) * num_cols_b +
                wsub_col_idx * WSUBN;

            for (uint res_idx_m = 0; res_idx_m < TM; ++res_idx_m) {
                for (uint res_idx_n = 0; res_idx_n < TN;
                     res_idx_n += STORE_VEC_SIZE) {
                    const int result_idx =
                        (wsub_row_idx * TM + res_idx_m) * (WNITER * TN) +
                        wsub_col_idx * TN + res_idx_n;
                    const int output_idx =
                        (thread_row_in_warp * TM + res_idx_m) * num_cols_b +
                        thread_col_in_warp * TN + res_idx_n;

                    InputType first = matrix_c_interim[output_idx + 0];
                    InputType second = matrix_c_interim[output_idx + 1];
                    first = from_float(alpha * thread_results[result_idx + 0] +
                                           beta * to_float(first),
                                       InputType{});
                    second = from_float(alpha * thread_results[result_idx + 1] +
                                            beta * to_float(second),
                                        InputType{});

                    VecT *dst = reinterpret_cast<VecT *>(&matrix_c_interim[output_idx]);
                    *dst = pack_vector(first, second);
                }
            }
        }
    }
}

template <typename InputType, const int BM, const int BN, const int BK,
          const int WM, const int WN, const int WNITER, const int TM,
          const int TN, const int NUM_THREADS>
void sgemm_warptiling_multidtype(
    const torch::Tensor &matrix_a,
    const torch::Tensor &matrix_b,
    torch::Tensor &output_matrix,
    float alpha,
    float beta) {
    TORCH_CHECK(matrix_a.device().is_cuda(), "Matrix A must be on CUDA device");
    TORCH_CHECK(matrix_b.device().is_cuda(), "Matrix B must be on CUDA device");
    TORCH_CHECK(output_matrix.device().is_cuda(), "Matrix C must be on CUDA device");
    TORCH_CHECK(matrix_a.is_contiguous() && matrix_b.is_contiguous() &&
                    output_matrix.is_contiguous(),
                "A, B, and C must be contiguous");
    TORCH_CHECK(output_matrix.dtype() == matrix_a.dtype(),
                "Matrix C must have the same dtype as matrix A");
    TORCH_CHECK(matrix_a.dim() == 2 && matrix_b.dim() == 2,
                "A and B must be 2D");

    const int num_rows_a = static_cast<int>(matrix_a.size(0));
    const int num_cols_a = static_cast<int>(matrix_a.size(1));
    const int num_cols_b = static_cast<int>(matrix_b.size(1));

    TORCH_CHECK(matrix_b.size(0) == num_cols_a,
                "Matrix dimensions must match: A is MxK, B must be KxN");
    TORCH_CHECK(output_matrix.size(0) == num_rows_a &&
                    output_matrix.size(1) == num_cols_b,
                "Matrix C must be MxN");
    TORCH_CHECK(num_rows_a % BM == 0,
                "Matrix A rows must be a multiple of ", BM);
    TORCH_CHECK(num_cols_a % BK == 0,
                "Matrix A columns must be a multiple of ", BK);
    TORCH_CHECK(num_cols_b % BN == 0,
                "Matrix B columns must be a multiple of ", BN);

    constexpr int WMITER = (WM * WN) / (WARP_SIZE * TM * TN * WNITER);
    constexpr int WSUBM = WM / WMITER;
    constexpr int WSUBN = WN / WNITER;
    static_assert(WMITER * WSUBM == WM, "Invalid warp-row tiling");
    static_assert(WNITER * WSUBN == WN, "Invalid warp-column tiling");
    static_assert(BM % WM == 0 && BN % WN == 0,
                  "Block tile must be divisible by warp tile");
    static_assert(WSUBM % TM == 0 && WSUBN % TN == 0,
                  "Warp subtile must be divisible by thread tile");

    const auto *d_matrix_a = static_cast<const InputType *>(matrix_a.data_ptr());
    const auto *d_matrix_b = static_cast<const InputType *>(matrix_b.data_ptr());
    auto *d_output_matrix = static_cast<InputType *>(output_matrix.data_ptr());

    const dim3 block_dim(NUM_THREADS);
    const dim3 grid_dim(ceil_div(num_cols_b, BN), ceil_div(num_rows_a, BM));
    sgemm_warptiling_multidtype_kernel<
        InputType, BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
        <<<grid_dim, block_dim>>>(
            num_rows_a, num_cols_b, num_cols_a,
            alpha, d_matrix_a, d_matrix_b, beta, d_output_matrix);
}

void sgemm_warptiling_fp16(
    const torch::Tensor &matrix_a,
    const torch::Tensor &matrix_b,
    torch::Tensor &output_matrix,
    float alpha,
    float beta) {
    TORCH_CHECK(matrix_a.dtype() == torch::kFloat16 &&
                    matrix_b.dtype() == torch::kFloat16,
                "A and B must both be float16");
    sgemm_warptiling_multidtype<half, 128, 128, 16, 64, 64, 4, 8, 4, 128>(
        matrix_a, matrix_b, output_matrix, alpha, beta);
}

void sgemm_warptiling_bf16(
    const torch::Tensor &matrix_a,
    const torch::Tensor &matrix_b,
    torch::Tensor &output_matrix,
    float alpha,
    float beta) {
    TORCH_CHECK(matrix_a.dtype() == torch::kBFloat16 &&
                    matrix_b.dtype() == torch::kBFloat16,
                "A and B must both be bfloat16");
    sgemm_warptiling_multidtype<nv_bfloat16, 128, 128, 16, 64, 64, 4, 8, 4, 128>(
        matrix_a, matrix_b, output_matrix, alpha, beta);
}
