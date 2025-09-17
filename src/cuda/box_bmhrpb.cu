#include <ATen/native/cuda/KernelUtils.cuh>
#include <c10/util/Half.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

#include "utils.h"

const int THREADS_PER_BLOCK_FORWARD = 512;
//const int THREADS_PER_BLOCK = 256;

template <int HEIGHT, int WIDTH, int C_HIDDEN, int N_HEADS>
__global__ void __launch_bounds__(THREADS_PER_BLOCK_FORWARD, 2) pos_mlp_multi_head_bias_forward_kernel(
    const float* __restrict__ mlp_weights,
    const float* __restrict__ pos,
    int B,
    float* __restrict__ output
) {
    const int total_elements = B * N_HEADS * HEIGHT * WIDTH;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_elements) return;

    const int h = (tid / (HEIGHT * WIDTH)) % N_HEADS;
    const int b = tid / (N_HEADS * HEIGHT * WIDTH);
    const int spatial_idx = tid % (HEIGHT * WIDTH);

    const float i = ((float) (spatial_idx / WIDTH)) / (float) (HEIGHT-1);
    const float j = ((float) (spatial_idx % WIDTH)) / (float) (WIDTH-1);

    const float cx = pos[b * 4 + 0];
    const float cy = pos[b * 4 + 1];
    const float half_w = fmaxf(pos[b * 4 + 2] * 0.5f, 1e-6f);
    const float half_h = fmaxf(pos[b * 4 + 3] * 0.5f, 1e-6f);

    const float rel_x = (j - cx) / half_w;
    const float rel_y = (i - cy) / half_h;

    const int weights_per_batch = 3 * C_HIDDEN + C_HIDDEN * N_HEADS + N_HEADS;
    const int weights_offset = b * weights_per_batch;
    const float* w = &mlp_weights[weights_offset];

    // First layer and softmax are independent of the head, so computation is similar
    float temp[C_HIDDEN];
    float max_val = -FLT_MAX;
    for (int k = 0; k < C_HIDDEN; k++) {
        const float w_x = w[2 * k];
        const float w_y = w[2 * k + 1];
        const float b1 = w[2 * C_HIDDEN + k];
        temp[k] = rel_x * w_x + rel_y * w_y + b1;
        if (temp[k] > max_val) max_val = temp[k];
    }

    float sum_exp = 0.0f;
    for (int k = 0; k < C_HIDDEN; k++) {
        temp[k] = __expf(temp[k] - max_val);
        sum_exp += temp[k];
    }

    const float inv_sum_exp = 1.0f / sum_exp;
    for (int k = 0; k < C_HIDDEN; k++) {
        temp[k] *= inv_sum_exp;
    }

    float out_val = 0.0f;
    const float* w2 = &w[3 * C_HIDDEN];
    const float* b2 = &w[3 * C_HIDDEN + C_HIDDEN * N_HEADS];

    for (int k = 0; k < C_HIDDEN; k++) {
        // Access weight for hidden node k and output head h
        out_val += temp[k] * w2[k * N_HEADS + h];
    }
    // Add bias for the current head h
    out_val += b2[h];

    output[tid] = out_val;
}

template <int MAX_C_HIDDEN>
__global__ void __launch_bounds__(THREADS_PER_BLOCK_FORWARD, 2) pos_mlp_multi_head_bias_forward_dynamic_kernel(
    const float* __restrict__ mlp_weights,
    const float* __restrict__ pos,
    int B,
    int N_HEADS,
    int HEIGHT,
    int WIDTH,
    int C_HIDDEN,
    float* __restrict__ output
) {
    const int total_elements = B * N_HEADS * HEIGHT * WIDTH;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_elements) return;

    const int h = (tid / (HEIGHT * WIDTH)) % N_HEADS;
    const int b = tid / (N_HEADS * HEIGHT * WIDTH);
    const int spatial_idx = tid % (HEIGHT * WIDTH);

    const float i = ((float) (spatial_idx / WIDTH)) / (float) (HEIGHT-1);
    const float j = ((float) (spatial_idx % WIDTH)) / (float) (WIDTH-1);

    const float cx = pos[b * 4 + 0];
    const float cy = pos[b * 4 + 1];
    const float half_w = fmaxf(pos[b * 4 + 2] * 0.5f, 1e-6f);
    const float half_h = fmaxf(pos[b * 4 + 3] * 0.5f, 1e-6f);

    const float rel_x = (j - cx) / half_w;
    const float rel_y = (i - cy) / half_h;

    const int weights_per_batch = 3 * C_HIDDEN + C_HIDDEN * N_HEADS + N_HEADS;
    const int weights_offset = b * weights_per_batch;
    const float* w = &mlp_weights[weights_offset];

    float temp[MAX_C_HIDDEN];
    float max_val = -FLT_MAX;
    for (int k = 0; k < C_HIDDEN; k++) {
        const float w_x = w[2 * k];
        const float w_y = w[2 * k + 1];
        const float b1 = w[2 * C_HIDDEN + k];
        temp[k] = rel_x * w_x + rel_y * w_y + b1;
        if (temp[k] > max_val) max_val = temp[k];
    }

    float sum_exp = 0.0f;
    for (int k = 0; k < C_HIDDEN; k++) {
        temp[k] = __expf(temp[k] - max_val);
        sum_exp += temp[k];
    }

    const float inv_sum_exp = 1.0f / sum_exp;
    for (int k = 0; k < C_HIDDEN; k++) {
        temp[k] *= inv_sum_exp;
    }

    float out_val = 0.0f;
    const float* w2 = &w[3 * C_HIDDEN];
    const float* b2 = &w[3 * C_HIDDEN + C_HIDDEN * N_HEADS];

    for (int k = 0; k < C_HIDDEN; k++) {
        out_val += temp[k] * w2[k * N_HEADS + h];
    }
    out_val += b2[h];

    output[tid] = out_val;
}

torch::Tensor fused_box_bmhrbp_forward(
    const torch::Tensor& mlp_weights, // (B, [3*C' + C'*Nh + Nh])
    const torch::Tensor& pos,    // (B,[x,y,w,h])
    const int c_hidden,
    const int n_heads,
    const int height,
    const int width
) {
    CHECK_INPUT(mlp_weights);
    CHECK_INPUT(pos);

    const int B = mlp_weights.size(0);

    auto output = torch::empty({B, n_heads, height, width}, mlp_weights.options());
    
    const int total_elements = B * n_heads * height * width;
    const int blocks = (total_elements + THREADS_PER_BLOCK_FORWARD - 1) / THREADS_PER_BLOCK_FORWARD;

    // Templated kernel launcher
    auto launch_kernel = [&](auto... Dims) {
        pos_mlp_multi_head_bias_forward_kernel<decltype(Dims)::value...><<<blocks, THREADS_PER_BLOCK_FORWARD>>>(
            mlp_weights.data_ptr<float>(),
            pos.data_ptr<float>(),
            B,
            output.data_ptr<float>()
        );
    };

    // Dynamic kernel launcher (fallback)
    auto launch_dynamic_kernel = [&]() {
        pos_mlp_multi_head_bias_forward_dynamic_kernel<16><<<blocks, THREADS_PER_BLOCK_FORWARD>>>(
            mlp_weights.data_ptr<float>(),
            pos.data_ptr<float>(),
            B,
            n_heads,
            height, 
            width, 
            c_hidden,
            output.data_ptr<float>()
        );
    };

    const auto supported_dims = std::make_tuple(
        std::make_tuple(
            std::integral_constant<int, 64>{}
        ), // height
        std::make_tuple(
            std::integral_constant<int, 64>{}
        ), // width
        std::make_tuple(
            std::integral_constant<int, 16>{}
        ),  // c_hidden
        std::make_tuple(
            std::integral_constant<int, 8>{}
        ) // n_heads
    );

    const auto runtime_dims = std::make_tuple(height, width, c_hidden, n_heads);

    // Call the generalized dispatcher.
    dispatch_kernel_with_fallback(launch_kernel, launch_dynamic_kernel, runtime_dims, supported_dims);

    CHECK_CUDA_ERROR(cudaPeekAtLastError());

    return output;
}

// A robust, reusable warp-level reduction helper for sums
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val; // The result is in the first lane (0, 32, 64...)
}

// A warp-level reduction helper for max
__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

template <int HEIGHT, int WIDTH, int C_HIDDEN, int N_HEADS>
__global__ void __launch_bounds__(256, 4) pos_mlp_multi_head_bias_backward_kernel(
    const float* __restrict__ mlp_weights,
    const float* __restrict__ pos,
    const float* __restrict__ grad_output,
    int B,
    float* __restrict__ grad_weights
) {
    int tid = threadIdx.x;
    int b = blockIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    // Use a warp-based stride for the main loop to ensure warp coherence.
    int warps_per_block = blockDim.x / 32;

    constexpr int grad_size = 3 * C_HIDDEN + C_HIDDEN * N_HEADS + N_HEADS;

    // Shared memory will be used for two things:
    // 1. Caching the weights.
    // 2. A final, block-wide reduction of gradients.
    extern __shared__ float s_mem[];
    float* s_weights = s_mem;
    float* s_grad_reduction = s_mem + grad_size;

    // --- STAGE 1: Load weights into shared memory ---
    for (int idx = tid; idx < grad_size; idx += blockDim.x) {
        s_weights[idx] = mlp_weights[b * grad_size + idx];
    }
    __syncthreads(); // Ensure weights are loaded before proceeding.

    const float* s_w1 = s_weights;
    const float* s_b1 = s_weights + 2 * C_HIDDEN;
    const float* s_w2 = s_weights + 3 * C_HIDDEN;
    
    // --- STAGE 2: Main computation loop with private register accumulation ---
    const float cx = pos[b * 4 + 0];
    const float cy = pos[b * 4 + 1];
    const float half_w = fmaxf(pos[b * 4 + 2] * 0.5f, 1e-6f);
    const float half_h = fmaxf(pos[b * 4 + 3] * 0.5f, 1e-6f);
    const float* grad_out_b = &grad_output[b * N_HEADS * HEIGHT * WIDTH];

    // Each thread accumulates its own portion of gradients into private registers. NO SPILLING.
    float p_grad_w1_x = 0.0f;
    float p_grad_w1_y = 0.0f;
    float p_grad_b1 = 0.0f;
    float p_grad_w2[N_HEADS] = {0.0f};
    float p_grad_b2[N_HEADS] = {0.0f};

    // warp-based loop: An entire warp processes one 'idx' at a time.
    for (int idx = warp_id; idx < HEIGHT * WIDTH; idx += warps_per_block) {
        // Since C_HIDDEN (16) < warp_size (32), we only need a subset of lanes.
        // But the reduction math is simpler if we have all threads participate and zero-out unused lanes.
        
        int k = lane_id;

        float rel_x = ((((float)(idx % WIDTH)) / (float)(WIDTH - 1)) - cx) / half_w;
        float rel_y = ((((float)(idx / WIDTH)) / (float)(HEIGHT - 1)) - cy) / half_h;

        // --- Recompute activations using warp-level primitives ---
        // Load x_k only if the lane is active for the computation.
        float x_k = (k < C_HIDDEN) ? (rel_x * s_w1[2 * k] + rel_y * s_w1[2 * k + 1] + s_b1[k]) : -FLT_MAX;
        
        float max_x_val = warp_reduce_max(x_k);
        float max_x = __shfl_sync(0xFFFFFFFF, max_x_val, 0);

        float s_k_unnormalized = (k < C_HIDDEN) ? __expf(x_k - max_x) : 0.0f;
        float sum_exp_val = warp_reduce_sum(s_k_unnormalized);
        float sum_exp = __shfl_sync(0xFFFFFFFF, sum_exp_val, 0);
        float s_k = s_k_unnormalized / sum_exp;
        
        float dL_dx_k = 0.0f;
        float dL_doutput_hij = grad_out_b[idx];

        for (int h = 0; h < N_HEADS; ++h) {
            float dL_doutput_hij_h = grad_out_b[h * HEIGHT * WIDTH + idx];
            
            if (lane_id == 0) {
                p_grad_b2[h] += dL_doutput_hij_h;
            }

            float s_k_times_w2_kh = (k < C_HIDDEN) ? (s_k * s_w2[k * N_HEADS + h]) : 0.0f;
            float output_h_val = warp_reduce_sum(s_k_times_w2_kh);
            float output_h = __shfl_sync(0xFFFFFFFF, output_h_val, 0);

            if (k < C_HIDDEN) {
                p_grad_w2[h] += dL_doutput_hij_h * s_k;
                dL_dx_k += dL_doutput_hij_h * s_k * (s_w2[k * N_HEADS + h] - output_h);
            }
        }
        
        if (k < C_HIDDEN) {
            p_grad_w1_x += dL_dx_k * rel_x;
            p_grad_w1_y += dL_dx_k * rel_y;
            p_grad_b1 += dL_dx_k;
        }
    }

    // --- STAGE 3: THE CORRECTED Block-wide Reduction ---
    for (int i = tid; i < grad_size; i += blockDim.x) { s_grad_reduction[i] = 0.0f; }
    __syncthreads();

    int k = lane_id;
    if (k < C_HIDDEN) {
        atomicAdd(&s_grad_reduction[2 * k], p_grad_w1_x);
        atomicAdd(&s_grad_reduction[2 * k + 1], p_grad_w1_y);
        atomicAdd(&s_grad_reduction[2 * C_HIDDEN + k], p_grad_b1);
        for (int h = 0; h < N_HEADS; ++h) {
            atomicAdd(&s_grad_reduction[3 * C_HIDDEN + k * N_HEADS + h], p_grad_w2[h]);
        }
    }
    
    // Now reduce b2: each thread has a partial sum that needs to be fully reduced.
    for (int h = 0; h < N_HEADS; ++h) {
        atomicAdd(&s_grad_reduction[3 * C_HIDDEN + C_HIDDEN * N_HEADS + h], p_grad_b2[h]);
    }
    __syncthreads();

    // --- STAGE 4: Final write to Global Memory (Unchanged) ---
    for (int idx = tid; idx < grad_size; idx += blockDim.x) {
        grad_weights[b * grad_size + idx] = s_grad_reduction[idx];
    }
}

torch::Tensor fused_box_bmhrbp_backward(
    const torch::Tensor& grad_out, // (B, Nh, H, W)
    const torch::Tensor& mlp_weights, // (B, [3*C' + C'*Nh + Nh])
    const torch::Tensor& pos,    // (B,[x,y,w,h])
    const int c_hidden,
    const int n_heads
) {
    CHECK_INPUT(grad_out);
    CHECK_INPUT(mlp_weights);
    CHECK_INPUT(pos);

    const int B = mlp_weights.size(0);
    const int height = grad_out.size(2);
    const int width = grad_out.size(3);

    auto grad_weights = torch::zeros_like(mlp_weights);
    int grad_size = 3 * c_hidden + c_hidden * n_heads + n_heads;

    const int THREADS_PER_BLOCK = 256;
    size_t shared_mem_size = (grad_size + grad_size) * sizeof(float);
    
    // Templated kernel launcher
    auto launch_kernel = [&](auto... Dims) {
        // Make sure to call the new kernel!
        pos_mlp_multi_head_bias_backward_kernel<decltype(Dims)::value...><<<B, THREADS_PER_BLOCK, shared_mem_size>>>(
            mlp_weights.data_ptr<float>(),
            pos.data_ptr<float>(),
            grad_out.data_ptr<float>(),
            B,
            grad_weights.data_ptr<float>()
        );
    };

    const auto supported_dims = std::make_tuple(
        std::make_tuple(
            std::integral_constant<int, 64>{}
        ), // height
        std::make_tuple(
            std::integral_constant<int, 64>{}
        ), // width
        std::make_tuple(
            std::integral_constant<int, 16>{}
        ),  // c_hidden
        std::make_tuple(
            std::integral_constant<int, 8>{}
        ) // n_heads
    );

    const auto runtime_dims = std::make_tuple(height, width, c_hidden, n_heads);

    // Call the generalized dispatcher.
    dispatch_kernel(launch_kernel, runtime_dims, supported_dims);

    CHECK_CUDA_ERROR(cudaPeekAtLastError());

    return grad_weights;
}