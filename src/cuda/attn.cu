#include <ATen/native/cuda/KernelUtils.cuh>
#include <c10/util/Half.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

#include "utils.h"

#define MAX_HEAD_DIM 32

__global__ void flash_attention_forward_kernel(
    const float* q,  // (B, Nh, Nq, C)
    const float* k,  // (B, Nh, Nk, C)
    const float* v,  // (B, Nh, Nk, C)
    float* out,      // (B, Nh, Nq, C)
    int B, 
    int Nh, 
    int Nq, 
    int Nk, 
    int C
) {
    int total_queries = B * Nh * Nq;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_queries) return;

    // Decompose index into batch, head, and query indices
    int b = idx / (Nh * Nq);
    int h = (idx % (Nh * Nq)) / Nq;
    int i = idx % Nq;

    const float scale = rsqrtf(static_cast<float>(C));

    // Calculate offsets for the current query and key/value heads
    int q_offset = b * (Nh * Nq * C) + h * (Nq * C) + i * C;
    int k_offset = b * (Nh * Nk * C) + h * (Nk * C);
    int v_offset = k_offset; // Same stride as K

    float m_i = -INFINITY;
    float l_i = 0.0f;
    float acc[MAX_HEAD_DIM] = {0.0f};

    for (int j = 0; j < Nk; ++j) {
        float s_ij = 0.0f;
        // Compute Q[i] dot K[j]
        for (int c = 0; c < C; ++c) {
            s_ij += q[q_offset + c] * k[k_offset + j * C + c];
        }
        s_ij *= scale;

        // Online softmax update
        float m_prev = m_i;
        m_i = fmaxf(m_i, s_ij);
        float p_ij = expf(s_ij - m_i);
        float rescale_factor = expf(m_prev - m_i);
        l_i = l_i * rescale_factor + p_ij;

        // Update accumulator
        for (int c = 0; c < C; ++c) {
            acc[c] = acc[c] * rescale_factor + p_ij * v[v_offset + j * C + c];
        }
    }

    // Normalize and write output
    int out_offset = b * (Nh * Nq * C) + h * (Nq * C) + i * C;
    float inv_l_i = 1.0f / l_i;
    for (int c = 0; c < C; ++c) {
        out[out_offset + c] = acc[c] * inv_l_i;
    }
}

torch::Tensor attn_forward(
    const torch::Tensor &Q, // (B, Nh, Nq, C)
    const torch::Tensor &K, // (B, Nh, Nk, C)
    const torch::Tensor &V // (B, Nh, Nk, Cv)
) {
    const int B = Q.size(0);
    const int Nh = Q.size(1);
    const int Nq = Q.size(2);
    const int C = Q.size(3);
    const int Nk = K.size(2);

    auto O = torch::zeros_like(Q);
    int total_queries = B * Nh * Nq;
    int threadsPerBlock = 256;
    int blocks = (total_queries + threadsPerBlock - 1) / threadsPerBlock;

    flash_attention_forward_kernel<<<blocks, threadsPerBlock>>>(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        O.data_ptr<float>(),
        B, Nh, Nq, Nk, C
    );

    CHECK_CUDA_ERROR(cudaPeekAtLastError());

    return O;
}