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

template <int HEIGHT, int WIDTH, int C_HIDDEN, int N_HEADS, int TILE_H, int TILE_W>
__global__ void tiled_parallel_grad_kernel(
    const float* __restrict__ mlp_weights,
    const float* __restrict__ pos,
    const float* __restrict__ grad_output,
    int B,
    float* __restrict__ intermediate_grad_weights
) {
    // Each block computes a tile. Grid is over (B, H/TILE_H, W/TILE_W)
    int b = blockIdx.x;
    
    // 1D threadIdx mapped to 2D for tile processing
    int tx = threadIdx.x % TILE_W;
    int ty = threadIdx.x / TILE_W;

    // Global (h, w) coordinates for this thread
    int w = blockIdx.z * TILE_W + tx;
    int h = blockIdx.y * TILE_H + ty;

    constexpr int grad_size = 3 * C_HIDDEN + C_HIDDEN * N_HEADS + N_HEADS;
    extern __shared__ float s_weights[];

    // --- STAGE 1: Load weights into shared memory (once per block) ---
    // All threads in the block cooperate to load the weights.
    for (int i = threadIdx.x; i < grad_size; i += blockDim.x) {
        s_weights[i] = mlp_weights[b * grad_size + i];
    }
    __syncthreads();

    // --- STAGE 2: Main computation per-thread ---
    // Boundary check: ensure the thread is within the valid HxW grid
    if (h < HEIGHT && w < WIDTH) {
        const float* s_w1 = s_weights;
        const float* s_b1 = s_weights + 2 * C_HIDDEN;
        const float* s_w2 = s_weights + 3 * C_HIDDEN;

        const float cx = pos[b * 4 + 0];
        const float cy = pos[b * 4 + 1];
        const float half_w = fmaxf(pos[b * 4 + 2] * 0.5f, 1e-6f);
        const float half_h = fmaxf(pos[b * 4 + 3] * 0.5f, 1e-6f);

        int idx = h * WIDTH + w;
        float rel_x = ((((float)w) / (float)(WIDTH - 1)) - cx) / half_w;
        float rel_y = ((((float)h) / (float)(HEIGHT - 1)) - cy) / half_h;
        
        // --- Activation recomputation (warp-parallel) ---
        // This entire section is now executed by each thread for its assigned (h,w)
        int k = threadIdx.x % 32; // lane_id
        
        // Note: Using a single warp's lanes for this part means we need __shfl_sync,
        // but the calculation is independent for each thread's (h,w).
        // This logic remains fundamentally the same, just executed by many more threads.
        float x_k = (k < C_HIDDEN) ? (rel_x * s_w1[2 * k] + rel_y * s_w1[2 * k + 1] + s_b1[k]) : -FLT_MAX;
        float max_x_val = warp_reduce_max(x_k);
        float max_x = __shfl_sync(0xFFFFFFFF, max_x_val, 0);

        float s_k_unnormalized = (k < C_HIDDEN) ? __expf(x_k - max_x) : 0.0f;
        float sum_exp_val = warp_reduce_sum(s_k_unnormalized);
        float sum_exp = __shfl_sync(0xFFFFFFFF, sum_exp_val, 0);
        float s_k = s_k_unnormalized / (sum_exp + 1e-6f);

        float dL_dx_k = 0.0f;
        for (int head = 0; head < N_HEADS; ++head) {
            float dL_doutput_hij_h = grad_output[(b * N_HEADS + head) * HEIGHT * WIDTH + idx];
            float s_k_times_w2_kh = (k < C_HIDDEN) ? (s_k * s_w2[k * N_HEADS + head]) : 0.0f;
            float output_h_val = warp_reduce_sum(s_k_times_w2_kh);
            float output_h = __shfl_sync(0xFFFFFFFF, output_h_val, 0);
            if (k < C_HIDDEN) {
                dL_dx_k += dL_doutput_hij_h * s_k * (s_w2[k * N_HEADS + head] - output_h);
            }
        }
    
        // --- STAGE 3: Write per-location gradients to intermediate buffer ---
        // Each thread writes to its own unique location, no race conditions.
        // We only need one thread to write the final gradient for each of the C_HIDDEN components.
        // We can do this with a warp shuffle.
        if (k == 0) { // Only lane 0 of each warp does the writing
            size_t offset = (size_t)b * HEIGHT * WIDTH * grad_size + (size_t)idx * grad_size;
            float* grad_out_ptr = intermediate_grad_weights + offset;

            for (int ck = 0; ck < C_HIDDEN; ++ck) {
                float dL_dx_k_at_ck = __shfl_sync(0xFFFFFFFF, dL_dx_k, ck);
                grad_out_ptr[2 * ck] = dL_dx_k_at_ck * rel_x;
                grad_out_ptr[2 * ck + 1] = dL_dx_k_at_ck * rel_y;
                grad_out_ptr[2 * C_HIDDEN + ck] = dL_dx_k_at_ck;
            }

            for (int head = 0; head < N_HEADS; ++head) {
                float dL_doutput_hij_h = grad_output[(b * N_HEADS + head) * HEIGHT * WIDTH + idx];
                grad_out_ptr[3 * C_HIDDEN + C_HIDDEN * N_HEADS + head] = dL_doutput_hij_h;
                for (int ck = 0; ck < C_HIDDEN; ++ck) {
                    float s_k_at_ck = __shfl_sync(0xFFFFFFFF, s_k, ck);
                    grad_out_ptr[3 * C_HIDDEN + ck * N_HEADS + head] = dL_doutput_hij_h * s_k_at_ck;
                }
            }
        }
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

    int grad_size = 3 * c_hidden + c_hidden * n_heads + n_heads;

    // Allocate the intermediate buffer for the map-reduce operation
    auto intermediate_opts = grad_out.options().dtype(torch::kFloat32);
    auto intermediate_grad = torch::empty({B, height, width, grad_size}, intermediate_opts);
    
    // --- Define Tile Dimensions ---
    // A 16x16 tile gives a block size of 256, a good starting point for tuning. 94ms
    constexpr int TILE_H = 32;
    constexpr int TILE_W = 32;
    constexpr int THREADS_PER_BLOCK = TILE_H * TILE_W;

    // Templated kernel launcher
    auto launch_kernel = [&](auto... Dims) {
        // Calculate grid dimensions to cover the entire HxW space with tiles
        dim3 gridDim(
            B,
            (height + TILE_H - 1) / TILE_H,
            (width + TILE_W - 1) / TILE_W
        );
        dim3 blockDim(THREADS_PER_BLOCK);
        size_t shared_mem_size = grad_size * sizeof(float);

        tiled_parallel_grad_kernel<decltype(Dims)::value..., TILE_H, TILE_W><<<gridDim, blockDim, shared_mem_size>>>(
            mlp_weights.data_ptr<float>(),
            pos.data_ptr<float>(),
            grad_out.data_ptr<float>(),
            B,
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
    // Sum over the H (dim 1) and W (dim 2) dimensions.
    // The result will have the shape (B, grad_size), which matches mlp_weights.
    auto grad_weights = intermediate_grad.sum({1, 2});

    return grad_weights;
}