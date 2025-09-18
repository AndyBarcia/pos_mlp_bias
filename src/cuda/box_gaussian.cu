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

template <int H, int W, int N_HEADS>
__global__ void __launch_bounds__(THREADS_PER_BLOCK_BACKWARD) box_gaussian_backward_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ boxes,
    const float* __restrict__ offset,
    const float* __restrict__ sigma,
    int B,
    float* __restrict__ d_offset, // gradient w.r.t. offset
    float* __restrict__ d_sigma   // gradient w.r.t. sigma
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

    // Re-compute intermediate values from the forward pass
    const float grid_x = ((float)j) / (float)(W - 1);
    const float grid_y = ((float)i) / (float)(H - 1);

    const int box_offset = b * 4;
    const float center_x = boxes[box_offset + 0];
    const float center_y = boxes[box_offset + 1];
    const float half_w = boxes[box_offset + 2] * 0.5f;
    const float half_h = boxes[box_offset + 3] * 0.5f;

    const int head_params_offset = (b * N_HEADS + h_idx) * 2;
    const float offset_x = offset[head_params_offset + 0];
    const float offset_y = offset[head_params_offset + 1];
    const float sigma_x = sigma[head_params_offset + 0];
    const float sigma_y = sigma[head_params_offset + 1];

    const float effective_offset_x = offset_x * half_w;
    const float effective_offset_y = offset_y * half_h;
    
    const float effective_sigma_x = fmaxf(sigma_x * half_w, 1e-6f);
    const float effective_sigma_y = fmaxf(sigma_y * half_h, 1e-6f);

    const float effective_center_x = center_x + effective_offset_x;
    const float effective_center_y = center_y + effective_offset_y;

    const float delta_x = grid_x - effective_center_x;
    const float delta_y = grid_y - effective_center_y;

    const float term_x = delta_x / effective_sigma_x;
    const float term_y = delta_y / effective_sigma_y;
    const float exponent = -0.5f * (term_x * term_x + term_y * term_y);
    const float mask_val = __expf(exponent);

    // Common term for chain rule: (dL/d_mask) * (d_mask/d_exponent)
    const float common_grad = grad_output[tid] * mask_val;

    // ===== Gradient for offset =====
    const float grad_offset_x = common_grad * (delta_x / (effective_sigma_x * effective_sigma_x)) * half_w;
    const float grad_offset_y = common_grad * (delta_y / (effective_sigma_y * effective_sigma_y)) * half_h;

    // ===== Gradient for sigma =====
    const float grad_sigma_x = common_grad * (term_x * term_x / sigma_x);
    const float grad_sigma_y = common_grad * (term_y * term_y / sigma_y);
    
    // Atomically add the computed gradients to the output tensors
    atomicAdd(&d_offset[head_params_offset + 0], grad_offset_x);
    atomicAdd(&d_offset[head_params_offset + 1], grad_offset_y);
    atomicAdd(&d_sigma[head_params_offset + 0], grad_sigma_x);
    atomicAdd(&d_sigma[head_params_offset + 1], grad_sigma_y);
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

    // Initialize gradient tensors to zero for atomic accumulation
    auto d_offset = torch::zeros_like(offset);
    auto d_sigma = torch::zeros_like(sigma);

    const int total_elements = B * N_HEADS * H * W;
    const int blocks = (total_elements + THREADS_PER_BLOCK_BACKWARD - 1) / THREADS_PER_BLOCK_BACKWARD;

    // Templated kernel launcher
    auto launch_kernel = [&](auto... Dims) {
        box_gaussian_backward_kernel<decltype(Dims)::value...><<<blocks, THREADS_PER_BLOCK_BACKWARD>>>(
            grad_output.data_ptr<float>(),
            boxes.data_ptr<float>(),
            offset.data_ptr<float>(),
            sigma.data_ptr<float>(),
            B,
            d_offset.data_ptr<float>(),
            d_sigma.data_ptr<float>()
        );
    };

    // Define the supported dimensions for fast, templated kernels.
    const auto supported_dims = std::make_tuple(
        std::make_tuple( // Heights
            std::integral_constant<int, 64>{}
        ),
        std::make_tuple( // Widths
            std::integral_constant<int, 64>{}
        ),
        std::make_tuple( // Number of Heads
            std::integral_constant<int, 8>{}
        )
    );

    const auto runtime_dims = std::make_tuple(H, W, N_HEADS);
    dispatch_kernel(launch_kernel, runtime_dims, supported_dims);

    CHECK_CUDA_ERROR(cudaPeekAtLastError());

    return {d_offset, d_sigma};
}