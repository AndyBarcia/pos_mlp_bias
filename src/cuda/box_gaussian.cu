#include <ATen/native/cuda/KernelUtils.cuh>
#include <c10/util/Half.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

#include "utils.h"

const int THREADS_PER_BLOCK_FORWARD = 1024;
const int THREADS_PER_BLOCK_BACKWARD = 256;

template <int H, int W, int N_HEADS>
__global__ void __launch_bounds__(THREADS_PER_BLOCK_FORWARD) box_gaussian_forward_kernel(
    const float* __restrict__ boxes,
    const float* __restrict__ offset,
    const float* __restrict__ sigma,
    int B,
    float* __restrict__ output
) {
    const int total_elements = B * N_HEADS * H * W;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_elements) return;

    // Decompose the global thread index to get batch, head, and spatial indices
    const int b = tid / (N_HEADS * H * W);
    const int h_idx = (tid / (H * W)) % N_HEADS;
    const int spatial_idx = tid % (H * W);
    const int i = spatial_idx / W; // y-coordinate (row)
    const int j = spatial_idx % W; // x-coordinate (col)

    // Generate a normalized coordinate grid (from 0 to 1)
    const float grid_x = ((float)j) / (float)(W - 1);
    const float grid_y = ((float)i) / (float)(H - 1);

    // Fetch box centers and half-sizes
    const int box_offset = b * 4;
    const float center_x = boxes[box_offset + 0];
    const float center_y = boxes[box_offset + 1];
    const float half_w = boxes[box_offset + 2] * 0.5f;
    const float half_h = boxes[box_offset + 3] * 0.5f;

    // Fetch head-specific relative offsets and sigmas
    const int head_params_offset = (b * N_HEADS + h_idx) * 2;
    const float offset_x = offset[head_params_offset + 0];
    const float offset_y = offset[head_params_offset + 1];
    const float sigma_x = sigma[head_params_offset + 0];
    const float sigma_y = sigma[head_params_offset + 1];

    // Calculate the effective offset, standard deviation, and center, scaled by box size
    const float effective_offset_x = offset_x * half_w;
    const float effective_offset_y = offset_y * half_h;

    // A sigma of 1 corresponds to a standard deviation of half the box size.
    // Add a small epsilon to avoid division by zero.
    const float effective_sigma_x = fmaxf(sigma_x * half_w, 1e-6f);
    const float effective_sigma_y = fmaxf(sigma_y * half_h, 1e-6f);
    
    const float effective_center_x = center_x + effective_offset_x;
    const float effective_center_y = center_y + effective_offset_y;

    // Calculate the exponent of the Gaussian function
    const float delta_x = grid_x - effective_center_x;
    const float delta_y = grid_y - effective_center_y;

    const float term_x = delta_x / effective_sigma_x;
    const float term_y = delta_y / effective_sigma_y;

    const float exponent = -0.5f * (term_x * term_x + term_y * term_y);

    // Apply the exponential function and write the final value to output
    output[tid] = __expf(exponent);
}

__global__ void __launch_bounds__(THREADS_PER_BLOCK_FORWARD) box_gaussian_forward_dynamic_kernel(
    const float* __restrict__ boxes,
    const float* __restrict__ offset,
    const float* __restrict__ sigma,
    int B,
    int H,
    int W,
    int N_HEADS,
    float* __restrict__ output
) {
    const int total_elements = B * N_HEADS * H * W;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_elements) return;

    // Decompose the global thread index to get batch, head, and spatial indices
    const int b = tid / (N_HEADS * H * W);
    const int h_idx = (tid / (H * W)) % N_HEADS;
    const int spatial_idx = tid % (H * W);
    const int i = spatial_idx / W; // y-coordinate (row)
    const int j = spatial_idx % W; // x-coordinate (col)

    // Generate a normalized coordinate grid (from 0 to 1)
    const float grid_x = ((float)j) / (float)(W - 1);
    const float grid_y = ((float)i) / (float)(H - 1);

    // Fetch box centers and half-sizes
    const int box_offset = b * 4;
    const float center_x = boxes[box_offset + 0];
    const float center_y = boxes[box_offset + 1];
    const float half_w = boxes[box_offset + 2] * 0.5f;
    const float half_h = boxes[box_offset + 3] * 0.5f;

    // Fetch head-specific relative offsets and sigmas
    const int head_params_offset = (b * N_HEADS + h_idx) * 2;
    const float offset_x = offset[head_params_offset + 0];
    const float offset_y = offset[head_params_offset + 1];
    const float sigma_x = sigma[head_params_offset + 0];
    const float sigma_y = sigma[head_params_offset + 1];

    // Calculate the effective offset, standard deviation, and center, scaled by box size
    const float effective_offset_x = offset_x * half_w;
    const float effective_offset_y = offset_y * half_h;

    // A sigma of 1 corresponds to a standard deviation of half the box size.
    // Add a small epsilon to avoid division by zero.
    const float effective_sigma_x = fmaxf(sigma_x * half_w, 1e-6f);
    const float effective_sigma_y = fmaxf(sigma_y * half_h, 1e-6f);
    
    const float effective_center_x = center_x + effective_offset_x;
    const float effective_center_y = center_y + effective_offset_y;

    // Calculate the exponent of the Gaussian function
    const float delta_x = grid_x - effective_center_x;
    const float delta_y = grid_y - effective_center_y;

    const float term_x = delta_x / effective_sigma_x;
    const float term_y = delta_y / effective_sigma_y;

    const float exponent = -0.5f * (term_x * term_x + term_y * term_y);

    // Apply the exponential function and write the final value to output
    output[tid] = __expf(exponent);
}

torch::Tensor fused_box_gaussian_forward(
    const torch::Tensor& boxes,
    const torch::Tensor& offset,
    const torch::Tensor& sigma,
    const int H,
    const int W
) {
    CHECK_INPUT(boxes);
    CHECK_INPUT(offset);
    CHECK_INPUT(sigma);

    const int B = boxes.size(0);
    const int N_HEADS = offset.size(1);

    auto output = torch::empty({B, N_HEADS, H, W}, boxes.options());
    const int total_elements = B * N_HEADS * H * W;
    const int blocks = (total_elements + THREADS_PER_BLOCK_FORWARD - 1) / THREADS_PER_BLOCK_FORWARD;

    // Templated kernel launcher
    auto launch_kernel = [&](auto... Dims) {
        box_gaussian_forward_kernel<decltype(Dims)::value...><<<blocks, THREADS_PER_BLOCK_FORWARD>>>(
            boxes.data_ptr<float>(),
            offset.data_ptr<float>(),
            sigma.data_ptr<float>(),
            B,
            output.data_ptr<float>()
        );
    };

    // Dynamic kernel launcher (fallback)
    auto launch_dynamic_kernel = [&]() {
        box_gaussian_forward_dynamic_kernel<<<blocks, THREADS_PER_BLOCK_FORWARD>>>(
            boxes.data_ptr<float>(),
            offset.data_ptr<float>(),
            sigma.data_ptr<float>(),
            B, H, W, N_HEADS,
            output.data_ptr<float>()
        );
    };

    // Define the supported dimensions for fast, templated kernels.
    const auto supported_dims = std::make_tuple(
        std::make_tuple( // Heights
            std::integral_constant<int, 16>{}, 
            std::integral_constant<int, 32>{}, 
            std::integral_constant<int, 64>{}
        ),
        std::make_tuple( // Widths
            std::integral_constant<int, 16>{}, 
            std::integral_constant<int, 32>{}, 
            std::integral_constant<int, 64>{}
        ),
        std::make_tuple( // Number of Heads
            std::integral_constant<int, 4>{},
            std::integral_constant<int, 8>{},
            std::integral_constant<int, 16>{}
        )
    );

    // The runtime values that need to be dispatched.
    const auto runtime_dims = std::make_tuple(H, W, N_HEADS);

    // Call the generalized dispatcher.
    dispatch_kernel_with_fallback(launch_kernel, launch_dynamic_kernel, runtime_dims, supported_dims);

    CHECK_CUDA_ERROR(cudaPeekAtLastError());

    return output;
}

// A warp-level reduction utility. Sums a float value across all 32 threads in a warp.
__device__ __forceinline__ float warp_reduce_sum(float val) {
    // Each thread adds the value from the thread 16 lanes away
    val += __shfl_down_sync(0xFFFFFFFF, val, 16);
    // Threads 0-15 now have the partial sum
    val += __shfl_down_sync(0xFFFFFFFF, val, 8);
    // Threads 0-7 have the partial sum
    val += __shfl_down_sync(0xFFFFFFFF, val, 4);
    // etc.
    val += __shfl_down_sync(0xFFFFFFFF, val, 2);
    val += __shfl_down_sync(0xFFFFFFFF, val, 1);
    // The final sum is broadcast to all threads in the warp, but we only need it from lane 0.
    return __shfl_sync(0xFFFFFFFF, val, 0);
}


__global__ void __launch_bounds__(THREADS_PER_BLOCK_BACKWARD) box_gaussian_backward_optimized_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ boxes,
    const float* __restrict__ offset,
    const float* __restrict__ sigma,
    int B, int H, int W, int N_HEADS,
    float* __restrict__ d_offset,
    float* __restrict__ d_sigma
) {
    // This kernel uses a grid-stride loop and block-level reductions.
    // Each block is responsible for one head.
    const int head_id = blockIdx.x;
    const int b = head_id / N_HEADS;
    const int h_idx = head_id % N_HEADS;

    // Shared memory to cache input data for the current head
    __shared__ float s_data[8]; // 4 for box, 2 for offset, 2 for sigma
    
    // Thread 0 of each block loads the data for its assigned head into shared memory.
    if (threadIdx.x == 0) {
        const int box_offset = b * 4;
        s_data[0] = boxes[box_offset + 0]; // center_x
        s_data[1] = boxes[box_offset + 1]; // center_y
        s_data[2] = boxes[box_offset + 2] * 0.5f; // half_w
        s_data[3] = boxes[box_offset + 3] * 0.5f; // half_h

        const int head_params_offset = (b * N_HEADS + h_idx) * 2;
        s_data[4] = offset[head_params_offset + 0];
        s_data[5] = offset[head_params_offset + 1];
        s_data[6] = sigma[head_params_offset + 0];
        s_data[7] = sigma[head_params_offset + 1];
    }
    __syncthreads(); // Ensure all threads see the loaded data

    // All threads read the cached data from fast shared memory
    const float center_x = s_data[0];
    const float center_y = s_data[1];
    const float half_w = s_data[2];
    const float half_h = s_data[3];
    const float offset_x = s_data[4];
    const float offset_y = s_data[5];
    const float sigma_x = s_data[6];
    const float sigma_y = s_data[7];

    const float effective_offset_x = offset_x * half_w;
    const float effective_offset_y = offset_y * half_h;
    const float effective_sigma_x = fmaxf(sigma_x * half_w, 1e-6f);
    const float effective_sigma_y = fmaxf(sigma_y * half_h, 1e-6f);
    const float effective_center_x = center_x + effective_offset_x;
    const float effective_center_y = center_y + effective_offset_y;

    // Each thread accumulates gradients for multiple pixels in its private registers.
    float local_d_offset_x = 0.0f;
    float local_d_offset_y = 0.0f;
    float local_d_sigma_x = 0.0f;
    float local_d_sigma_y = 0.0f;
    
    const int total_pixels = H * W;
    const int grad_output_base_idx = head_id * total_pixels;

    // Use a grid-stride loop for each thread to iterate over the spatial dimensions
    for (int spatial_idx = threadIdx.x; spatial_idx < total_pixels; spatial_idx += blockDim.x) {
        const int i = spatial_idx / W;
        const int j = spatial_idx % W;

        const float grid_x = ((float)j) / (float)(W - 1);
        const float grid_y = ((float)i) / (float)(H - 1);

        const float delta_x = grid_x - effective_center_x;
        const float delta_y = grid_y - effective_center_y;

        const float term_x = delta_x / effective_sigma_x;
        const float term_y = delta_y / effective_sigma_y;
        const float exponent = -0.5f * (term_x * term_x + term_y * term_y);
        const float mask_val = __expf(exponent);
        
        const float common_grad = grad_output[grad_output_base_idx + spatial_idx] * mask_val;

        local_d_offset_x += common_grad * (delta_x / (effective_sigma_x * effective_sigma_x)) * half_w;
        local_d_offset_y += common_grad * (delta_y / (effective_sigma_y * effective_sigma_y)) * half_h;
        local_d_sigma_x += common_grad * (term_x * term_x / sigma_x);
        local_d_sigma_y += common_grad * (term_y * term_y / sigma_y);
    }

    // --- Block-Level Reduction ---
    // 1. Reduce sums within each warp
    float warp_d_offset_x = warp_reduce_sum(local_d_offset_x);
    float warp_d_offset_y = warp_reduce_sum(local_d_offset_y);
    float warp_d_sigma_x = warp_reduce_sum(local_d_sigma_x);
    float warp_d_sigma_y = warp_reduce_sum(local_d_sigma_y);

    // Shared memory for the final reduction across warps (one entry per warp)
    __shared__ float s_reduction_memory[4 * (THREADS_PER_BLOCK_BACKWARD / 32)];

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int num_warps = blockDim.x / 32;

    // 2. The first thread of each warp writes its warp's sum to shared memory
    if (lane_id == 0) {
        s_reduction_memory[warp_id] = warp_d_offset_x;
        s_reduction_memory[warp_id + num_warps] = warp_d_offset_y;
        s_reduction_memory[warp_id + 2 * num_warps] = warp_d_sigma_x;
        s_reduction_memory[warp_id + 3 * num_warps] = warp_d_sigma_y;
    }
    __syncthreads();

    // 3. The first warp reduces the final sums from shared memory
    if (warp_id == 0) {
        float final_d_offset_x = (lane_id < num_warps) ? s_reduction_memory[lane_id] : 0.0f;
        float final_d_offset_y = (lane_id < num_warps) ? s_reduction_memory[lane_id + num_warps] : 0.0f;
        float final_d_sigma_x = (lane_id < num_warps) ? s_reduction_memory[lane_id + 2 * num_warps] : 0.0f;
        float final_d_sigma_y = (lane_id < num_warps) ? s_reduction_memory[lane_id + 3 * num_warps] : 0.0f;

        final_d_offset_x = warp_reduce_sum(final_d_offset_x);
        final_d_offset_y = warp_reduce_sum(final_d_offset_y);
        final_d_sigma_x = warp_reduce_sum(final_d_sigma_x);
        final_d_sigma_y = warp_reduce_sum(final_d_sigma_y);

        // 4. Thread 0 performs the single atomic write for the entire block
        if (lane_id == 0) {
            const int head_params_offset = (b * N_HEADS + h_idx) * 2;
            atomicAdd(&d_offset[head_params_offset + 0], final_d_offset_x);
            atomicAdd(&d_offset[head_params_offset + 1], final_d_offset_y);
            atomicAdd(&d_sigma[head_params_offset + 0], final_d_sigma_x);
            atomicAdd(&d_sigma[head_params_offset + 1], final_d_sigma_y);
        }
    }
}


std::vector<torch::Tensor> fused_box_gaussian_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& boxes,
    const torch::Tensor& offset,
    const torch::Tensor& sigma
) {
    CHECK_INPUT(grad_output);
    CHECK_INPUT(boxes);
    CHECK_INPUT(offset);
    CHECK_INPUT(sigma);

    const int B = boxes.size(0);
    const int N_HEADS = offset.size(1);
    const int H = grad_output.size(2);
    const int W = grad_output.size(3);

    auto d_offset = torch::zeros_like(offset);
    auto d_sigma = torch::zeros_like(sigma);

    // Each block processes one head
    const int grid_dim = B * N_HEADS;
    const int block_dim = THREADS_PER_BLOCK_BACKWARD;

    box_gaussian_backward_optimized_kernel<<<grid_dim, block_dim>>>(
        grad_output.data_ptr<float>(),
        boxes.data_ptr<float>(),
        offset.data_ptr<float>(),
        sigma.data_ptr<float>(),
        B, H, W, N_HEADS,
        d_offset.data_ptr<float>(),
        d_sigma.data_ptr<float>()
    );

    CHECK_CUDA_ERROR(cudaPeekAtLastError());

    return {d_offset, d_sigma};
}