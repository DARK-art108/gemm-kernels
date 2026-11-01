import torch
import triton
import triton.language as tl


@triton.jit
def flash_attention_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, out_ptr,
    M, N, D, scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    # Each program handles a group of query rows.
    pid_m = tl.program_id(0)

    # Indices of the query rows handled by this program.
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    # Feature indices. Extra positions are padded with zeros.
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # Indices used to load K and V in chunks.
    offs_n = tl.arange(0, BLOCK_N)

    # Load Q once and reuse it while processing all K/V blocks.
    q = tl.load(
        Q_ptr + offs_m[:, None] * D + offs_d[None, :],
        mask=(offs_m[:, None] < M) &
             (offs_d[None, :] < D),
        other=0.0
    )

    # Online softmax state:
    # m_i = maximum attention score seen so far.
    # l_i = running sum of exponentials.
    # acc = running weighted sum of V.
    m_i = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_DMODEL), dtype=tl.float32)

    # Process all key/value rows in smaller blocks.
    for start_n in range(tl.cdiv(N, BLOCK_N)):

        cols_n = start_n * BLOCK_N + offs_n

        # Load the current key block.
        k = tl.load(
            K_ptr + cols_n[:, None] * D + offs_d[None, :],
            mask=(cols_n[:, None] < N) &
                 (offs_d[None, :] < D),
            other=0.0
        )

        # Load the corresponding value block.
        v = tl.load(
            V_ptr + cols_n[:, None] * D + offs_d[None, :],
            mask=(cols_n[:, None] < N) &
                 (offs_d[None, :] < D),
            other=0.0
        )

        # Calculate attention scores for the current Q/K block.
        qk = tl.dot(q, tl.trans(k), input_precision="ieee")
        qk = qk * scale

        # Ignore padded key columns and padded query rows.
        qk = tl.where(cols_n[None, :] < N, qk, float("-inf"))
        qk = tl.where(offs_m[:, None] < M, qk, float("-inf"))

        # Find the maximum score in the current block.
        m_ij = tl.max(qk, axis=1)

        # Update the maximum across all blocks processed so far.
        m_new = tl.maximum(m_i, m_ij)

        # Rescale previous results if the maximum has changed.
        alpha = tl.exp(m_i - m_new)

        # Calculate exponentials for the current block.
        p = tl.exp(qk - m_new[:, None])

        # Update the running softmax denominator.
        l_new = alpha * l_i + tl.sum(p, axis=1)

        # Rescale the previous weighted output.
        acc = acc * alpha[:, None]

        # Add weighted values from the current block.
        acc = tl.dot(p, v, acc, input_precision="ieee")

        # Save the updated online softmax state.
        m_i = m_new
        l_i = l_new

    # Normalize the accumulated weighted values.
    out = acc / l_i[:, None]

    # Store only the actual query rows and feature dimensions.
    tl.store(
        out_ptr + offs_m[:, None] * D + offs_d[None, :],
        out,
        mask=(offs_m[:, None] < M) &
             (offs_d[None, :] < D)
    )


def solve(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    out: torch.Tensor
) -> None:

    # M = number of query rows.
    # N = number of key/value rows.
    # D = feature dimension.
    M, D = Q.shape
    N = K.shape[0]

    scale = 1.0 / (D ** 0.5)

    BLOCK_M = 16
    BLOCK_N = 16

    # tl.dot requires a reduction dimension of at least 16.
    BLOCK_DMODEL = max(16, triton.next_power_of_2(D))

    # One program handles BLOCK_M query rows.
    grid = (triton.cdiv(M, BLOCK_M),)

    flash_attention_fwd_kernel[grid](
        Q, K, V, out,
        M, N, D, scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL,
        num_warps=4
    )
