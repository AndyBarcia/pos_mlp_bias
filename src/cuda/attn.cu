#include <ATen/native/cuda/KernelUtils.cuh>
#include <c10/util/Half.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

#include "utils.h"

// --- Tunable constants for tiling ---
// A thread block will process a tile of BLOCK_M queries.
constexpr int BLOCK_M = 128;
// The inner loop will process keys/values in tiles of size BLOCK_N.
constexpr int BLOCK_N = 64;
// The original accumulator size constraint.
#define MAX_HEAD_DIM 32

__global__ void flash_attention_forward_kernel_v3(
    const float* q,
    const float* k,
    const float* v,
    float* out,
    int B, int Nh, int Nq, int Nk, int C
) {
    // Shared memory for K and V tiles
    __shared__ float k_s[BLOCK_N * MAX_HEAD_DIM];
    __shared__ float v_s[BLOCK_N * MAX_HEAD_DIM];

    // --- V3: Per-thread register array for the Q vector ---
    float q_reg[MAX_HEAD_DIM];

    // 1D Grid Indexing
    const int query_row_in_block = threadIdx.x;
    const int batch_head_query_block_idx = blockIdx.x;
    
    const int b = batch_head_query_block_idx / (Nh * ((Nq + BLOCK_M - 1) / BLOCK_M));
    const int h_and_query_block = batch_head_query_block_idx % (Nh * ((Nq + BLOCK_M - 1) / BLOCK_M));
    const int h = h_and_query_block / ((Nq + BLOCK_M - 1) / BLOCK_M);
    const int query_block_idx = h_and_query_block % ((Nq + BLOCK_M - 1) / BLOCK_M);
    
    const int i = query_block_idx * BLOCK_M + query_row_in_block;
    if (i >= Nq) return;

    // Pointers to Global Memory
    const float* q_ptr = q + b * (Nh * Nq * C) + h * (Nq * C) + i * C;
    float* out_ptr = out + b * (Nh * Nq * C) + h * (Nq * C) + i * C;
    const float* k_batch_head_ptr = k + b * (Nh * Nk * C) + h * (Nk * C);
    const float* v_batch_head_ptr = v + b * (Nh * Nk * C) + h * (Nk * C);

    // --- V3: Pre-load Q into registers once before the main loop ---
    // This is a single, coalesced read that replaces thousands of uncoalesced reads inside the loop.
    const int num_float4_per_q_vector = C / 4;
    for (int idx = 0; idx < num_float4_per_q_vector; ++idx) {
        reinterpret_cast<float4*>(q_reg)[idx] = reinterpret_cast<const float4*>(q_ptr)[idx];
    }
    
    const float scale = rsqrtf(static_cast<float>(C));
    float m_i = -INFINITY;
    float l_i = 0.0f;
    float acc[MAX_HEAD_DIM] = {0.0f};

    // Main loop over K and V tiles
    for (int j_start = 0; j_start < Nk; j_start += BLOCK_N) {
        
        // Optimized Shared Memory Loading (same as V2)
        int num_float4_per_tile_row = C / 4;
        int total_loads_for_tile = (BLOCK_N * num_float4_per_tile_row);
        for (int load_idx = threadIdx.x; load_idx < total_loads_for_tile; load_idx += blockDim.x) {
            int row = load_idx / num_float4_per_tile_row;
            int col_float4 = load_idx % num_float4_per_tile_row;
            int j = j_start + row;

            if (j < Nk) {
                reinterpret_cast<float4*>(k_s)[row * num_float4_per_tile_row + col_float4] = 
                    reinterpret_cast<const float4*>(k_batch_head_ptr)[j * num_float4_per_tile_row + col_float4];
                reinterpret_cast<float4*>(v_s)[row * num_float4_per_tile_row + col_float4] = 
                    reinterpret_cast<const float4*>(v_batch_head_ptr)[j * num_float4_per_tile_row + col_float4];
            }
        }
        __syncthreads();

        // Compute attention for the current block
        for (int j_in_block = 0; j_in_block < BLOCK_N; ++j_in_block) {
            int j = j_start + j_in_block;
            if (j >= Nk) break;

            float s_ij = 0.0f;
            // --- V3: Use the Q vector from fast registers in the dot product ---
            for (int c = 0; c < C; ++c) {
                s_ij += q_reg[c] * k_s[j_in_block * C + c];
            }
            s_ij *= scale;

            float m_prev = m_i;
            m_i = fmaxf(m_i, s_ij);
            float p_ij = expf(s_ij - m_i);
            float rescale_factor = expf(m_prev - m_i);
            l_i = l_i * rescale_factor + p_ij;

            for (int c = 0; c < C; ++c) {
                acc[c] = acc[c] * rescale_factor + p_ij * v_s[j_in_block * C + c];
            }
        }
        __syncthreads();
    }

    // Normalize and write output
    const float inv_l_i = 1.0f / l_i;
    for (int c = 0; c < C; ++c) {
        out_ptr[c] = acc[c] * inv_l_i;
    }
}


torch::Tensor attn_forward(
    const torch::Tensor &Q,
    const torch::Tensor &K,
    const torch::Tensor &V
) {
    const int B = Q.size(0);
    const int Nh = Q.size(1);
    const int Nq = Q.size(2);
    const int C = Q.size(3);
    const int Nk = K.size(2);

    TORCH_CHECK(C <= MAX_HEAD_DIM, "Head dimension C exceeds MAX_HEAD_DIM");
    TORCH_CHECK(C % 4 == 0, "Head dimension C must be a multiple of 4 for float4 coalesced loading");

    auto O = torch::zeros_like(Q);

    const int query_blocks = (Nq + BLOCK_M - 1) / BLOCK_M;
    const int total_blocks = B * Nh * query_blocks;
    
    dim3 threads(BLOCK_M);
    dim3 blocks(total_blocks);

    flash_attention_forward_kernel_v3<<<blocks, threads>>>(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        O.data_ptr<float>(),
        B, Nh, Nq, Nk, C
    );

    CHECK_CUDA_ERROR(cudaPeekAtLastError());

    return O;
}