#include <cuda_runtime.h>
#include <math.h>

// ------------------------------------------------------------
// Kernel 1: Precompute transformed queries.
//
// A[b, d] = sum_j Pi[d, j]     * Q[b, j]
// T[b, d] = sum_j S_mat[d, j]  * Q[b, j]
//
// These transformations are reused for every cached token.
// ------------------------------------------------------------
__global__ void preprocess_queries(
    const float* Q,
    const float* Pi,
    const float* S_mat,
    float* A,
    float* T,
    int B,
    int D
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * D;

    if (idx >= total) return;

    int b = idx / D;
    int d = idx % D;

    float a_sum = 0.0f;
    float t_sum = 0.0f;

    for (int j = 0; j < D; ++j) {
        float q = Q[b * D + j];

        a_sum += Pi[d * D + j] * q;
        t_sum += S_mat[d * D + j] * q;
    }

    A[b * D + d] = a_sum;
    T[b * D + d] = t_sum;
}


// ------------------------------------------------------------
// Kernel 2: Compute one attention score per CUDA thread.
//
// score[b, s] =
//     sum_d A[b,d] * codebook[K_idx[s,d]]
//   + sqrt(pi/2) / D * gamma[s]
//       * sum_d T[b,d] * qjl_signs[s,d]
//
// No softmax and no 1/sqrt(D) scaling.
// ------------------------------------------------------------
__global__ void attention_scores_kernel(
    const float* A,
    const float* T,
    const unsigned char* K_idx,
    const signed char* qjl_signs,
    const float* gamma,
    const float* codebook,
    float* scores,
    int B,
    int S,
    int D
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * S;

    if (idx >= total) return;

    // Convert the flat output index into batch and token indices.
    int b = idx / S;
    int s = idx % S;

    float mse_sum = 0.0f;
    float residual_sum = 0.0f;

    // Reconstruct the score without creating a full key vector.
    for (int d = 0; d < D; ++d) {
        // Dequantize one rotated key coordinate.
        unsigned char code = K_idx[s * D + d];
        float centroid = codebook[code];

        // MSE / centroid contribution.
        mse_sum += A[b * D + d] * centroid;

        // QJL residual contribution.
        float sign = (float)qjl_signs[s * D + d];
        residual_sum += T[b * D + d] * sign;
    }

    const float correction_scale = 1.2533141373155001f / (float)D;

    float residual =
        correction_scale * gamma[s] * residual_sum;

    scores[idx] = mse_sum + residual;
}


// ------------------------------------------------------------
// Required entry point.
// All pointers refer to device memory.
// ------------------------------------------------------------
extern "C" void solve(
    const float* Q,
    const unsigned char* K_idx,
    const signed char* qjl_signs,
    const float* gamma,
    const float* Pi,
    const float* S_mat,
    const float* codebook,
    float* scores,
    int B,
    int S,
    int D,
    int C
) {
    (void)C;  // The codebook indices already identify centroids.

    // Temporary arrays store the two transformed queries.
    float* A = nullptr;
    float* T = nullptr;

    size_t bytes = (size_t)B * D * sizeof(float);

    cudaMalloc(&A, bytes);
    cudaMalloc(&T, bytes);

    constexpr int THREADS = 256;

    // Stage 1: transform each query coordinate.
    int preprocess_blocks = (B * D + THREADS - 1) / THREADS;

    preprocess_queries<<<preprocess_blocks, THREADS>>>(
        Q, Pi, S_mat, A, T, B, D
    );

    // Stage 2: one thread calculates each output score.
    int total_scores = B * S;
    int score_blocks = (total_scores + THREADS - 1) / THREADS;

    attention_scores_kernel<<<score_blocks, THREADS>>>(
        A, T, K_idx, qjl_signs, gamma,
        codebook, scores, B, S, D
    );

    // Free temporary device allocations.
    cudaFree(A);
    cudaFree(T);
}
