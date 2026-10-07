#include "utils.cuh"
#include <cassert>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <torch/extension.h>



// this one is more general case it cant handle edge cases of uneven mats

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__
void sgemm_blocktiling_2d_kernel(int num_rows_a, int num_cols_b, int num_cols_a,
	float alpha, const float *matrix_a, const float *matrix_b, float beta, float *matrix_c)

{
	const uint block_row = blockIdx.x;
	const uint block_col = blockIdx.y;

	__shared__ float tile_a[BM * BK];
	__shared__ float tile_b[BK * BN];


	const uint thread_row = threadIdx.x / (BN/TN);
	const uint thread_col = threadIdx.x % (BN/TN);
	const uint num_threads = (BN / TM) * (BN / TN);

	matrix_a += block_row * BM * num_cols_a;
	matrix_b += block_col * BN;
	matrix_c += block_row * BM * num_cols_b + block_col * BN;

	float thread_results[TM * TN] = {0.0f};
	float register_m[TM] = {0.0f};
	float register_n[TN] = {0.0f};

	for(uint block_k_idx = 0; block_k_idx <num_cols_a; block_k_idx += BK)
	{
#pragma unroll
		for(uint load_offset = 0; load_offset < BM * BK; load_offset += num_threads){
			uint load_idx = threadIdx.x + load_offset;
			uint a_row = load_idx / BK;
			uint a_col = load_idx % BK;
			tile_a[load_idx] = matrix_a[a_row * num_cols_a + a_col];
		}
#pragma unroll 
		for(uint load_offset = 0; load_offset < BK * BN; load_offset += num_threads){
			uint load_idx = threadIdx.x + load_offset;
			uint b_row = load_idx / BN;
			uint b_col = load_idx % BN;
			tile_b[load_idx] = matrix_b[b_row * num_cols_b + b_col];
		}

		__syncthreads();

		matrix_a += BK;
		matrix_b += BK * num_cols_b;


		for (uint dot_idx = 0; dot_idx< BK; ++dot_idx)
		{
			for(uint i=0; i< TM; ++i){
				register_m[i] = tile_a[(thread_row*TM+i)*BK+dot_idx];
			}

			for (uint i = 0; i<TN;++i)
			{
				register_n[i] = tile_b[dot_idx * BN + thread_col * TN + i];
			}
			for(uint res_idx_m = 0; res_idx_m<TM;++res_idx_m)
			{
				for(uint res_idx_n = 0; res_idx_n<TN; ++res_idx_n)
				{
					thread_results[res_idx_m*TN+res_idx_n] += register_m[res_idx_m]*register_n[res_idx_n];

				}

			}
		}

		__syncthreads();
	}

#pragma unroll
	for (uint res_idx_m=0; res_idx_m<TM;++res_idx_m)
	{
#pragma unroll
     for(uint res_idx_n=0; res_idx_n<TN;++res_idx_n)
     {

            const uint c_idx = (thread_row * TM + res_idx_m) * num_cols_b +
                               (thread_col * TN + res_idx_n);
            matrix_c[c_idx] = alpha * thread_results[res_idx_m * TN + res_idx_n] +
                              beta * matrix_c[c_idx];
     }		
	}


}


// Now writing the second kernel which is goign to handle the edge case in 2d tiling the whole story is lets say we have 1000x1000 this cant be fully 
//diveded so we do padding means filling with zero in right botton corner edges eg 1000/64 -- 15 reminder 40 now 15x15 = 225 64x64 mats we are having 
// the left are 40 rows 40 cols 64-40 24 rows and cols left we do padding for them

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemm_blocktiling_2d_edge_kernel(int num_rows_a, int num_cols_b, int num_cols_a,
                                                 float alpha, const float *matrix_a,
                                                 const float *matrix_b, float beta,
                                                 float *matrix_c,
                                                 int block_row_offset, int block_col_offset)

{
	const uint block_row = blockIdx.x + block_row_offset;
	const uint block_col = blockIdx.y + block_col_offset;

	__shared__ float tile_a[BM * BK];
	__shared__ float tile_b[BK * BN];
	const uint thread_row = threadIdx.x / (BN / TN);
	const uint thread_col = threadIdx.x % (BN /TN);
	const uint num_threads = (BM / TM) * (BN / TN);

	matrix_a += block_row * BM * num_cols_a;
	matrix_b += block_col * BN;
	matrix_c += block_row * BM * num_cols_b + block_col * BN;
	float thread_results[TM * TN] = {0.0f};
	float register_m[TM] = {0.0f};
	float register_n[TN] = {0.0f};

    for (uint block_k_idx = 0; block_k_idx < num_cols_a; block_k_idx += BK)
    {
#pragma unroll
        for (uint load_offset = 0; load_offset < BM * BK; load_offset += num_threads)
        {
            uint load_idx = threadIdx.x + load_offset;
            uint a_row = load_idx / BK;
            uint a_col = load_idx % BK;
            uint global_row_a = block_row * BM + a_row;
            uint global_col_a = block_k_idx + a_col;

            //padding logic first check weither the element is their or not if not then pad/insert zero for A
            tile_a[load_idx] = (global_row_a < num_rows_a && global_col_a < num_cols_a)
                                   ? matrix_a[a_row * num_cols_a + a_col]
                                   : 0.0f;
        }
#pragma unroll
        for (uint load_offset = 0; load_offset < BK * BN; load_offset += num_threads)
        {
            uint load_idx = threadIdx.x + load_offset;
            uint b_row = load_idx / BN;
            uint b_col = load_idx % BN;
            uint global_row_b = block_k_idx + b_row;
            uint global_col_b = block_col * BN + b_col;


            // same done for b mat 
            tile_b[load_idx] = (global_row_b < num_cols_a && global_col_b < num_cols_b)
                                   ? matrix_b[b_row * num_cols_b + b_col]
                                   : 0.0f;
        }

        __syncthreads();

        matrix_a += BK;
        matrix_b += BK * num_cols_b;

        for (uint dot_idx = 0; dot_idx < BK; ++dot_idx)
        {
            for (uint i = 0; i < TM; ++i)
            {
                register_m[i] = tile_a[(thread_row * TM + i) * BK + dot_idx];
            }

            for (uint i = 0; i < TN; ++i)
            {
                register_n[i] = tile_b[dot_idx * BN + thread_col * TN + i];
            }

            for (uint res_idx_m = 0; res_idx_m < TM; ++res_idx_m)
            {
                for (uint res_idx_n = 0; res_idx_n < TN; ++res_idx_n)
                {
                    thread_results[res_idx_m * TN + res_idx_n] +=
                        register_m[res_idx_m] * register_n[res_idx_n];
                }
            }
        }

        __syncthreads();
    }

    for (uint res_idx_m = 0; res_idx_m < TM; ++res_idx_m)
    {
        for (uint res_idx_n = 0; res_idx_n < TN; ++res_idx_n)
        {
            const uint global_row = block_row * BM + thread_row * TM + res_idx_m;
            const uint global_col = block_col * BN + thread_col * TN + res_idx_n;

            //here we will check weither c row < 1000 and c col is < 1000 if that is their we insert the result if not then exit.

            if (global_row < num_rows_a && global_col < num_cols_b)
            {
                const uint c_idx = (thread_row * TM + res_idx_m) * num_cols_b +
                                   (thread_col * TN + res_idx_n);
                matrix_c[c_idx] = alpha * thread_results[res_idx_m * TN + res_idx_n] +
                                  beta * matrix_c[c_idx];
            }
        }
    }
}

void sgemm_blocktiling_2d(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
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

    TORCH_CHECK(matrix_b.size(0) == num_cols_a, "Matrix dimensions must match: A is MxK, B must be KxN");
    TORCH_CHECK(output_matrix.device().is_cuda(), "Matrix C must be on CUDA device");
    TORCH_CHECK(output_matrix.dtype() == torch::kFloat32, "Matrix C must be float32");
    TORCH_CHECK(output_matrix.size(0) == num_rows_a && output_matrix.size(1) == num_cols_b, "Matrix C must be MxN");

    const float *d_matrix_a = matrix_a.data_ptr<float>();
    const float *d_matrix_b = matrix_b.data_ptr<float>();
    float *d_output_matrix = output_matrix.data_ptr<float>();

    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 8;
    constexpr int TM = 8;
    constexpr int TN = 8;

    dim3 block_dim((BM / TM) * (BN / TN));

    const int num_blocks_m = ceil_div(num_rows_a, BM);
    const int num_blocks_n = ceil_div(num_cols_b, BN);
    const int main_blocks_m = num_rows_a / BM;
    const int main_blocks_n = num_cols_b / BN;

    if (main_blocks_m > 0 && main_blocks_n > 0)
    {
        dim3 main_grid(main_blocks_m, main_blocks_n);
        sgemm_blocktiling_2d_kernel<BM, BN, BK, TM, TN><<<main_grid, block_dim>>>(
            num_rows_a, num_cols_b, num_cols_a,
            alpha, d_matrix_a, d_matrix_b, beta, d_output_matrix);
    }

    if (main_blocks_m > 0 && num_blocks_n > main_blocks_n)
    {
        dim3 edge_right_grid(main_blocks_m, 1);
        sgemm_blocktiling_2d_edge_kernel<BM, BN, BK, TM, TN><<<edge_right_grid, block_dim>>>(
            num_rows_a, num_cols_b, num_cols_a,
            alpha, d_matrix_a, d_matrix_b, beta, d_output_matrix,
            0, main_blocks_n);
    }

    if (num_blocks_m > main_blocks_m && main_blocks_n > 0)
    {
        dim3 edge_bottom_grid(1, main_blocks_n);
        sgemm_blocktiling_2d_edge_kernel<BM, BN, BK, TM, TN><<<edge_bottom_grid, block_dim>>>(
            num_rows_a, num_cols_b, num_cols_a,
            alpha, d_matrix_a, d_matrix_b, beta, d_output_matrix,
            main_blocks_m, 0);
    }

    if (num_blocks_m > main_blocks_m && num_blocks_n > main_blocks_n)
    {
        dim3 edge_corner_grid(1, 1);
        sgemm_blocktiling_2d_edge_kernel<BM, BN, BK, TM, TN><<<edge_corner_grid, block_dim>>>(
            num_rows_a, num_cols_b, num_cols_a,
            alpha, d_matrix_a, d_matrix_b, beta, d_output_matrix,
            main_blocks_m, main_blocks_n);
    }
}




























