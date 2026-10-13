#include "utils.cuh"
#include <cassert>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <torch/extension.h>



//this kernel usez 4 float technicques to sqeez more tflops from matmuls
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__
void sgemm_vectorize_kernel(int num_rows_a, int num_cols_b, int num_cols_a, float alpha, const float *matrix_a, const float *matrix_b, float beta, float *matrix_c)
{
	const uint block_row = blockIdx.y;
	const uint block_col = blockIdx.x;

	//now we need indices for C tiles compution

	const uint thread_col = threadIdx.x % (BN / TN);
	const uint thread_row = threadIdx.x / (BN / TN);

	__shared__ float tile_a[BM * BK];
	__shared__ float tile_b[BK * BN];

	const float *block_a = matrix_a + block_row * BM * num_cols_a;
	const float *block_b = matrix_b + block_col * BN;

	const uint inner_row_a = threadIdx.x / (BK / 4);
	const uint inner_col_a = threadIdx.x % (BK / 4);
	const uint inner_row_b = threadIdx.x / (BN / 4);
	const uint inner_col_b = threadIdx.x % (BN / 4);

	//register storages
	float thread_results[TM * TN] = {0.0f};
	float register_m[TM] = {0.0f};
	float register_n[TN] = {0.0f};


	for (uint block_k_idx = 0; block_k_idx < static_cast<uint>(num_cols_a); block_k_idx += BK) {
		const uint a_col = block_k_idx + inner_col_a * 4;
		const uint b_row = block_k_idx + inner_row_b;
		const uint b_col = inner_col_b * 4;

		if (num_cols_a % 4 == 0 && inner_row_a < BM && a_col + 3 < static_cast<uint>(num_cols_a)) {
			const float4 tmp_a = reinterpret_cast<const float4*>(
				&block_a[inner_row_a * num_cols_a + a_col])[0];
			tile_a[(inner_col_a * 4 + 0) * BM + inner_row_a] = tmp_a.x;
			tile_a[(inner_col_a * 4 + 1) * BM + inner_row_a] = tmp_a.y;
			tile_a[(inner_col_a * 4 + 2) * BM + inner_row_a] = tmp_a.z;
			tile_a[(inner_col_a * 4 + 3) * BM + inner_row_a] = tmp_a.w;
		} else {
			#pragma unroll
			for (uint i = 0; i < 4; ++i) {
				tile_a[(inner_col_a * 4 + i) * BM + inner_row_a] =
					(a_col + i < static_cast<uint>(num_cols_a))
						? block_a[inner_row_a * num_cols_a + a_col + i]
						: 0.0f;
			}
		}

		if (num_cols_b % 4 == 0 && b_row < static_cast<uint>(num_cols_a) &&
		    b_col + 3 < static_cast<uint>(num_cols_b)) {
			const float4 tmp_b = reinterpret_cast<const float4*>(
				&block_b[b_row * num_cols_b + b_col])[0];
			tile_b[inner_row_b * BN + b_col + 0] = tmp_b.x;
			tile_b[inner_row_b * BN + b_col + 1] = tmp_b.y;
			tile_b[inner_row_b * BN + b_col + 2] = tmp_b.z;
			tile_b[inner_row_b * BN + b_col + 3] = tmp_b.w;
		} else {
			#pragma unroll
			for (uint i = 0; i < 4; ++i) {
				tile_b[inner_row_b * BN + b_col + i] =
					(b_row < static_cast<uint>(num_cols_a) &&
					 b_col + i < static_cast<uint>(num_cols_b))
						? block_b[b_row * num_cols_b + b_col + i]
						: 0.0f;
			}
		}

		__syncthreads();

		for (uint dot_idx = 0; dot_idx < BK; ++dot_idx) {
			#pragma unroll
			for (uint i = 0; i < TM; ++i) {
				register_m[i] = tile_a[dot_idx * BM + thread_row * TM + i];
			}

			#pragma unroll
			for (uint i = 0; i < TN; ++i) {
				register_n[i] = tile_b[dot_idx * BN + thread_col * TN + i];
			}

			#pragma unroll
			for (uint res_idx_m = 0; res_idx_m < TM; ++res_idx_m) {
				#pragma unroll
				for (uint res_idx_n = 0; res_idx_n < TN; ++res_idx_n) {
					thread_results[res_idx_m * TN + res_idx_n] +=
						register_m[res_idx_m] * register_n[res_idx_n];
				}
			}
		}
		__syncthreads();
	}
    #pragma unroll
    for (uint res_idx_m = 0; res_idx_m < TM; ++res_idx_m) {
        #pragma unroll
        for (uint res_idx_n = 0; res_idx_n < TN; ++res_idx_n) {
            const uint row = block_row * BM + thread_row * TM + res_idx_m;
            const uint col = block_col * BN + thread_col * TN + res_idx_n;
            if (row < static_cast<uint>(num_rows_a) &&
                col < static_cast<uint>(num_cols_b)) {
                const uint c_idx = row * num_cols_b + col;
                matrix_c[c_idx] = alpha * thread_results[res_idx_m * TN + res_idx_n] +
                                  beta * matrix_c[c_idx];
            }
        }
    }


}

void sgemm_vectorize(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                     torch::Tensor &output_matrix, float alpha, float beta)
{
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

    TORCH_CHECK(matrix_b.size(0) == num_cols_a, "Matrix dimensions must match: A is MxK, B must be KxN");

    TORCH_CHECK(output_matrix.device().is_cuda(), "Matrix C must be on CUDA device");
    TORCH_CHECK(output_matrix.dtype() == torch::kFloat32, "Matrix C must be float32");
    TORCH_CHECK(output_matrix.size(0) == num_rows_a && output_matrix.size(1) == num_cols_b, "Matrix C must be MxN");

    // Get raw device pointers
    const float *d_matrix_a = matrix_a.data_ptr<float>();
    const float *d_matrix_b = matrix_b.data_ptr<float>();
    float *d_output_matrix = output_matrix.data_ptr<float>();

    // Template parameters for kernel
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 8;
    constexpr int TM = 8;
    constexpr int TN = 8;

    // Edge tiles are masked; aligned groups of four still use float4 loads.

    // Kernel launch.
    dim3 block_dim((BM / TM) * (BN / TN));
    dim3 grid_dim((num_cols_b + BN - 1) / BN,
                  (num_rows_a + BM - 1) / BM);

    // Launch kernel
    sgemm_vectorize_kernel<BM, BN, BK, TM, TN><<<grid_dim, block_dim>>>(
        num_rows_a, num_cols_b, num_cols_a,
        alpha, d_matrix_a, d_matrix_b, beta, d_output_matrix);
}