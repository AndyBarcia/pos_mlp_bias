#include <ATen/native/cuda/KernelUtils.cuh>
#include <c10/util/Half.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

#include "utils.h"

const int THREADS_PER_BLOCK_FORWARD = 512;
const int THREADS_PER_BLOCK_BACKWARD = 1024;

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

// TODO runs in 95 ms. Needs to be even faster.
// Needs a x5 speed up.

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
__global__ void __launch_bounds__(THREADS_PER_BLOCK_BACKWARD, 1) tiled_parallel_grad_kernel_atomic(
    const float* __restrict__ mlp_weights,
    const float* __restrict__ pos,
    const float* __restrict__ grad_output,
    int B,
    int num_blocks_per_batch,
    float* __restrict__ intermediate_grad_weights
) {
    // --- 1. Shared Memory and Indexing ---
    constexpr int grad_size = 3 * C_HIDDEN + C_HIDDEN * N_HEADS + N_HEADS;
    extern __shared__ float s_mem[];
    float* s_weights = s_mem;
    float* s_final_grad = s_mem + grad_size;

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    // --- 2. Cooperative Loading & Initialization ---
    const int b = blockIdx.x / num_blocks_per_batch;
    for (int i = tid; i < grad_size; i += blockDim.x) {
        s_weights[i] = mlp_weights[b * grad_size + i];
        s_final_grad[i] = 0.0f; // Initialize shared gradient array to zero
    }
    __syncthreads();

    const float* s_w1 = s_weights;
    const float* s_b1 = s_weights + 2 * C_HIDDEN;
    const float* s_w2 = s_weights + 3 * C_HIDDEN;

    // --- 3. COALESCED Gradient Computation ---
    constexpr int pixels_per_tile = THREADS_PER_BLOCK_BACKWARD / N_HEADS; // Each block processes a tile of 128 pixels
    const int h = tid / pixels_per_tile;
    const int pixel_in_tile_idx = tid % pixels_per_tile;

    const int block_group_id = blockIdx.x % num_blocks_per_batch;
    const int spatial_idx = block_group_id * pixels_per_tile + pixel_in_tile_idx;

    if (spatial_idx < HEIGHT * WIDTH) {
        // --- Recompute forward pass for this thread's (pixel, head) pair ---
        const int i_coord = spatial_idx / WIDTH;
        const int j_coord = spatial_idx % WIDTH;
        
        const float i_norm = ((float) i_coord) / (float) (HEIGHT - 1);
        const float j_norm = ((float) j_coord) / (float) (WIDTH - 1);

        const float cx = pos[b * 4 + 0];
        const float cy = pos[b * 4 + 1];
        const float half_w = fmaxf(pos[b * 4 + 2] * 0.5f, 1e-6f);
        const float half_h = fmaxf(pos[b * 4 + 3] * 0.5f, 1e-6f);

        const float rel_x = (j_norm - cx) / half_w;
        const float rel_y = (i_norm - cy) / half_h;
        
        float s[C_HIDDEN];
        // Recompute softmax output 's' for this pixel (same as before)
        {
            //float x[C_HIDDEN];
            float max_x = -FLT_MAX;
            for (int k = 0; k < C_HIDDEN; k++) {
                s[k] = rel_x * s_w1[2 * k] + rel_y * s_w1[2 * k + 1] + s_b1[k];
                if (s[k] > max_x) max_x = s[k];
            }
            float sum_exp = 0.0f;
            for (int k = 0; k < C_HIDDEN; k++) {
                s[k] = __expf(s[k] - max_x);
                sum_exp += s[k];
            }
            const float inv_sum = 1.0f / sum_exp;
            for (int k = 0; k < C_HIDDEN; k++) {
                s[k] *= inv_sum;
            }
        }

        const float dL_doutput = grad_output[b * N_HEADS * HEIGHT * WIDTH + h * HEIGHT * WIDTH + spatial_idx];
        
        float output_h_minus_b2_h = 0.0f;
        for (int k = 0; k < C_HIDDEN; k++) {
            output_h_minus_b2_h += s[k] * s_w2[k * N_HEADS + h];
        }

        // --- 4. IMMEDIATE REDUCTION to remove register pressure ---
        // For each component k, calculate its gradient contribution, reduce, and add to shared memory.
        for (int k = 0; k < C_HIDDEN; k++) {
            // --- Grads for w1 and b1 ---
            const float dL_dx_k_contrib = dL_doutput * s[k] * (s_w2[k * N_HEADS + h] - output_h_minus_b2_h);
            
            float grad_w1x = dL_dx_k_contrib * rel_x;
            float reduced_val = warp_reduce_sum(grad_w1x);
            if (lane_id == 0) atomicAdd(&s_final_grad[2 * k], reduced_val);

            float grad_w1y = dL_dx_k_contrib * rel_y;
            reduced_val = warp_reduce_sum(grad_w1y);
            if (lane_id == 0) atomicAdd(&s_final_grad[2 * k + 1], reduced_val);

            float grad_b1k = dL_dx_k_contrib;
            reduced_val = warp_reduce_sum(grad_b1k);
            if (lane_id == 0) atomicAdd(&s_final_grad[2 * C_HIDDEN + k], reduced_val);

            // --- Grad for w2 ---
            float grad_w2kh = dL_doutput * s[k];
            reduced_val = warp_reduce_sum(grad_w2kh);
            if (lane_id == 0) atomicAdd(&s_final_grad[3 * C_HIDDEN + k * N_HEADS + h], reduced_val);
        }

        // --- Grad for b2 (after the main loop) ---
        float reduced_val = warp_reduce_sum(dL_doutput);
        if (lane_id == 0) atomicAdd(&s_final_grad[3 * C_HIDDEN + C_HIDDEN * N_HEADS + h], reduced_val);
    }
    
    __syncthreads();

    // --- 5. Final Coalesced Write to Global Memory ---
    const int out_base_idx = b * num_blocks_per_batch * grad_size + block_group_id * grad_size;
    for (int i = tid; i < grad_size; i += blockDim.x) {
        intermediate_grad_weights[out_base_idx + i] = s_final_grad[i];
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

    const int grad_size = 3 * c_hidden + c_hidden * n_heads + n_heads;
    const int pixels_per_block_tile = THREADS_PER_BLOCK_BACKWARD / n_heads; // Each block processes a tile of 128 pixels
    const int num_blocks_per_batch = (height * width + pixels_per_block_tile - 1) / pixels_per_block_tile;

    // --- STEP 1: Allocate intermediate tensor for block-level results ---
    auto intermediate_grad = torch::zeros({B, num_blocks_per_batch, grad_size}, mlp_weights.options());
    
    // --- STEP 2: Launch the optimized backward kernel ---
    const int grid_size = B * num_blocks_per_batch;
    // Shared memory for weights AND the final reduced gradient for the block
    const size_t shared_mem_size = (grad_size + grad_size) * sizeof(float);

    auto launch_kernel = [&](auto... Dims) {
        tiled_parallel_grad_kernel_atomic<decltype(Dims)::value...><<<grid_size, THREADS_PER_BLOCK_BACKWARD, shared_mem_size>>>(
            mlp_weights.data_ptr<float>(),
            pos.data_ptr<float>(),
            grad_out.data_ptr<float>(),
            B,
            num_blocks_per_batch,
            intermediate_grad.data_ptr<float>()
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

    // --- STEP 3: Use native PyTorch to perform the final reduction ---
    auto grad_weights = intermediate_grad.sum({1});

    return grad_weights;
}