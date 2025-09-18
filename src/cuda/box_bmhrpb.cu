#include <ATen/native/cuda/KernelUtils.cuh>
#include <c10/util/Half.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

#include "utils.h"

const int THREADS_PER_BLOCK_FORWARD = 512;
const int THREADS_PER_BLOCK_BACKWARD = 512; 

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
__global__ void __launch_bounds__(THREADS_PER_BLOCK_BACKWARD, 1) tiled_gemm_backward_kernel(
    const float* __restrict__ mlp_weights,
    const float* __restrict__ pos,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_weights // Note: No intermediate tensor
) {
    // --- 1. KERNEL CONFIG & SHARED MEMORY ---
    constexpr int grad_size = 3 * C_HIDDEN + C_HIDDEN * N_HEADS + N_HEADS;
    constexpr int SPATIAL_TILE_DIM = THREADS_PER_BLOCK_BACKWARD/N_HEADS; // Each tile processes 128 pixels

    extern __shared__ float s_mem[];
    float* s_weights = s_mem;
    float* s_grad = s_mem + grad_size;
    float* s_tile_grad_out = s_grad + grad_size; // N_HEADS x TILE_DIM
    float* s_tile_s = s_tile_grad_out + N_HEADS * SPATIAL_TILE_DIM; // C_HIDDEN x TILE_DIM

    // --- 2. INITIALIZATION ---
    // Each block computes the full gradient for one batch item.
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane_id = tid % 32;

    // Cooperatively load weights and zero out the shared gradient accumulator
    for (int i = tid; i < grad_size; i += blockDim.x) {
        s_weights[i] = mlp_weights[b * grad_size + i];
        s_grad[i] = 0.0f;
    }
    __syncthreads();

    const float* s_w1 = s_weights;
    const float* s_b1 = s_weights + 2 * C_HIDDEN;
    const float* s_w2 = s_weights + 3 * C_HIDDEN;
    
    const float cx = pos[b * 4 + 0];
    const float cy = pos[b * 4 + 1];
    const float half_w = fmaxf(pos[b * 4 + 2] * 0.5f, 1e-6f);
    const float half_h = fmaxf(pos[b * 4 + 3] * 0.5f, 1e-6f);

    // --- 3. MAIN LOOP OVER SPATIAL TILES ---
    for (int spatial_base = 0; spatial_base < HEIGHT * WIDTH; spatial_base += SPATIAL_TILE_DIM) {
        
        // --- 3a. PHASE 1: Load Tile Data into Shared Memory ---
        // 1024 threads cooperatively load a (N_HEADS x 128) tile of grad_output
        // and recompute the corresponding (C_HIDDEN x 128) tile of 's' activations.
        for (int i = tid; i < SPATIAL_TILE_DIM; i += blockDim.x) {
            const int spatial_idx = spatial_base + i;
            if (spatial_idx < HEIGHT * WIDTH) {
                // Load one column of grad_output tile
                for(int h=0; h < N_HEADS; ++h){
                    s_tile_grad_out[h * SPATIAL_TILE_DIM + i] = grad_output[b * N_HEADS * HEIGHT * WIDTH + h * HEIGHT * WIDTH + spatial_idx];
                }
                
                // Recompute one column of 's' activation tile
                const int i_coord = spatial_idx / WIDTH;
                const int j_coord = spatial_idx % WIDTH;
                const float rel_x = ((((float)j_coord) / (float)(WIDTH - 1)) - cx) / half_w;
                const float rel_y = ((((float)i_coord) / (float)(HEIGHT - 1)) - cy) / half_h;
                
                float s[C_HIDDEN];
                {
                    float x[C_HIDDEN];
                    float max_x = -FLT_MAX;
                    for (int k = 0; k < C_HIDDEN; k++) {
                        x[k] = rel_x * s_w1[2 * k] + rel_y * s_w1[2 * k + 1] + s_b1[k];
                        if (x[k] > max_x) max_x = x[k];
                    }
                    float sum_exp = 0.0f;
                    for (int k = 0; k < C_HIDDEN; k++) {
                        s[k] = __expf(x[k] - max_x);
                        sum_exp += s[k];
                    }
                    const float inv_sum = 1.0f / sum_exp;
                    for (int k = 0; k < C_HIDDEN; k++) {
                        s[k] *= inv_sum;
                        s_tile_s[k * SPATIAL_TILE_DIM + i] = s[k];
                    }
                }
            }
        }
        __syncthreads();

        // --- 3b. PHASE 2: Compute on Tile and Accumulate Gradient ---
        // Each thread processes its assigned (head, pixel_in_tile) pair
        const int h = tid / SPATIAL_TILE_DIM;
        const int p_idx = tid % SPATIAL_TILE_DIM;
        const int spatial_idx = spatial_base + p_idx;

        if (spatial_idx < HEIGHT * WIDTH) {
            const float dL_doutput = s_tile_grad_out[h * SPATIAL_TILE_DIM + p_idx];

            float output_h_minus_b2_h = 0.0f;
            for (int k = 0; k < C_HIDDEN; k++) {
                output_h_minus_b2_h += s_tile_s[k * SPATIAL_TILE_DIM + p_idx] * s_w2[k * N_HEADS + h];
            }

            for (int k = 0; k < C_HIDDEN; k++) {
                const float s_val = s_tile_s[k * SPATIAL_TILE_DIM + p_idx];
                const float dL_dx_k_contrib = dL_doutput * s_val * (s_w2[k * N_HEADS + h] - output_h_minus_b2_h);
                
                const int i_coord = spatial_idx / WIDTH;
                const int j_coord = spatial_idx % WIDTH;
                const float rel_x = ((((float)j_coord) / (float)(WIDTH - 1)) - cx) / half_w;
                const float rel_y = ((((float)i_coord) / (float)(HEIGHT - 1)) - cy) / half_h;
                
                float reduced_val = warp_reduce_sum(dL_dx_k_contrib * rel_x);
                if (lane_id == 0) atomicAdd(&s_grad[2 * k], reduced_val);

                reduced_val = warp_reduce_sum(dL_dx_k_contrib * rel_y);
                if (lane_id == 0) atomicAdd(&s_grad[2 * k + 1], reduced_val);
                
                reduced_val = warp_reduce_sum(dL_dx_k_contrib);
                if (lane_id == 0) atomicAdd(&s_grad[2 * C_HIDDEN + k], reduced_val);

                reduced_val = warp_reduce_sum(dL_doutput * s_val);
                if (lane_id == 0) atomicAdd(&s_grad[3 * C_HIDDEN + k * N_HEADS + h], reduced_val);
            }
            float reduced_val = warp_reduce_sum(dL_doutput);
            if (lane_id == 0) atomicAdd(&s_grad[3 * C_HIDDEN + C_HIDDEN * N_HEADS + h], reduced_val);
        }
        __syncthreads();
    }

    // --- 4. FINAL WRITE ---
    // All threads in the block help write the final gradient to global memory.
    for (int i = tid; i < grad_size; i += blockDim.x) {
        grad_weights[b * grad_size + i] = s_grad[i];
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
    
    // Allocate final output tensor directly. No intermediate tensor needed.
    auto grad_weights = torch::empty({B, grad_size}, mlp_weights.options());

    // Grid is just B blocks, one for each batch item.
    const int grid_size = B;
    
    constexpr int SPATIAL_TILE_DIM = THREADS_PER_BLOCK_BACKWARD/8; // Each tile processes 128 pixels
    const size_t shared_mem_size = (
        grad_size + // s_grad
        grad_size + // s_weights
        n_heads * SPATIAL_TILE_DIM + // s_tile_grad_out
        c_hidden * SPATIAL_TILE_DIM  // s_tile_s
    ) * sizeof(float);

    auto launch_kernel = [&](auto... Dims) {
        tiled_gemm_backward_kernel<decltype(Dims)::value...><<<grid_size, THREADS_PER_BLOCK_BACKWARD, shared_mem_size>>>(
            mlp_weights.data_ptr<float>(),
            pos.data_ptr<float>(),
            grad_out.data_ptr<float>(),
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

    dispatch_kernel(launch_kernel, runtime_dims, supported_dims);
    CHECK_CUDA_ERROR(cudaPeekAtLastError());

    // The final reduction is no longer needed, the kernel does it all.
    return grad_weights;
}