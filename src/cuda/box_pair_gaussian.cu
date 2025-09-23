#include <ATen/native/cuda/KernelUtils.cuh>
#include <c10/util/Half.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

#include "utils.h"

const int THREADS_PER_BLOCK_FORWARD = 1024;
const int THREADS_PER_BLOCK_BACKWARD = 64;

__global__ void __launch_bounds__(THREADS_PER_BLOCK_FORWARD) box_pair_gaussian_forward_kernel(
    const float* __restrict__ boxes1,
    const float* __restrict__ offset1,
    const float* __restrict__ sigma1,
    const float* __restrict__ boxes2,
    const float* __restrict__ offset2,
    const float* __restrict__ sigma2,
    int B,
    int N_HEADS,
    int M_HEADS,
    float* __restrict__ output
) {
    const int total_elements = B * N_HEADS * M_HEADS;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_elements) return;

    // Decompose the global thread index to get batch, head1, and head2 indices
    const int b = tid / (N_HEADS * M_HEADS);
    const int n_idx = (tid / M_HEADS) % N_HEADS;
    const int m_idx = tid % M_HEADS;

    // --- Process the first set of Gaussians ---

    // Fetch box centers and half-sizes for the first set
    // MODIFIED: Indexing for boxes1 changed to support (B, N, 4) shape
    const int box1_offset = (b * N_HEADS + n_idx) * 4;
    const float center1_x = boxes1[box1_offset + 0];
    const float center1_y = boxes1[box1_offset + 1];
    const float half_w1 = boxes1[box1_offset + 2] * 0.5f;
    const float half_h1 = boxes1[box1_offset + 3] * 0.5f;

    // Fetch head-specific relative offsets and sigmas for the first set
    const int head1_params_offset = (b * N_HEADS + n_idx) * 2;
    const float offset1_x = offset1[head1_params_offset + 0];
    const float offset1_y = offset1[head1_params_offset + 1];
    const float sigma1_x = sigma1[head1_params_offset + 0];
    const float sigma1_y = sigma1[head1_params_offset + 1];

    // Calculate effective center and sigma for the first Gaussian
    const float effective_center1_x = center1_x + offset1_x * half_w1;
    const float effective_center1_y = center1_y + offset1_y * half_h1;
    const float effective_sigma1_x = fmaxf(sigma1_x * half_w1, 1e-6f);
    const float effective_sigma1_y = fmaxf(sigma1_y * half_h1, 1e-6f);

    // --- Process the second set of Gaussians ---

    // Fetch box centers and half-sizes for the second set
    // MODIFIED: Indexing for boxes2 changed to support (B, M, 4) shape
    const int box2_offset = (b * M_HEADS + m_idx) * 4;
    const float center2_x = boxes2[box2_offset + 0];
    const float center2_y = boxes2[box2_offset + 1];
    const float half_w2 = boxes2[box2_offset + 2] * 0.5f;
    const float half_h2 = boxes2[box2_offset + 3] * 0.5f;

    // Fetch head-specific relative offsets and sigmas for the second set
    const int head2_params_offset = (b * M_HEADS + m_idx) * 2;
    const float offset2_x = offset2[head2_params_offset + 0];
    const float offset2_y = offset2[head2_params_offset + 1];
    const float sigma2_x = sigma2[head2_params_offset + 0];
    const float sigma2_y = sigma2[head2_params_offset + 1];

    // Calculate effective center and sigma for the second Gaussian
    const float effective_center2_x = center2_x + offset2_x * half_w2;
    const float effective_center2_y = center2_y + offset2_y * half_h2;
    const float effective_sigma2_x = fmaxf(sigma2_x * half_w2, 1e-6f);
    const float effective_sigma2_y = fmaxf(sigma2_y * half_h2, 1e-6f);

    // --- Compute the Bhattacharyya coefficient ---

    // X-dimension
    const float mu_diff_x_sq = (effective_center1_x - effective_center2_x) * (effective_center1_x - effective_center2_x);
    const float var_sum_x = effective_sigma1_x * effective_sigma1_x + effective_sigma2_x * effective_sigma2_x;
    const float exp_term_x = __expf(-0.25f * mu_diff_x_sq / var_sum_x);
    const float sqrt_term_x = sqrtf(2.0f * effective_sigma1_x * effective_sigma2_x / var_sum_x);
    const float bc_x = exp_term_x * sqrt_term_x;

    // Y-dimension
    const float mu_diff_y_sq = (effective_center1_y - effective_center2_y) * (effective_center1_y - effective_center2_y);
    const float var_sum_y = effective_sigma1_y * effective_sigma1_y + effective_sigma2_y * effective_sigma2_y;
    const float exp_term_y = __expf(-0.25f * mu_diff_y_sq / var_sum_y);
    const float sqrt_term_y = sqrtf(2.0f * effective_sigma1_y * effective_sigma2_y / var_sum_y);
    const float bc_y = exp_term_y * sqrt_term_y;

    // Combine dimensions
    output[tid] = bc_x * bc_y;
}

torch::Tensor fused_box_pair_gaussian_forward(
    const torch::Tensor& boxes1,
    const torch::Tensor& offset1,
    const torch::Tensor& sigma1,
    const torch::Tensor& boxes2,
    const torch::Tensor& offset2,
    const torch::Tensor& sigma2
) {
    CHECK_INPUT(boxes1);
    CHECK_INPUT(offset1);
    CHECK_INPUT(sigma1);
    CHECK_INPUT(boxes2);
    CHECK_INPUT(offset2);
    CHECK_INPUT(sigma2);

    const int B = boxes1.size(0);
    const int N_HEADS = offset1.size(1);
    const int M_HEADS = offset2.size(1);

    auto output = torch::empty({B, N_HEADS, M_HEADS}, boxes1.options());
    const int total_elements = B * N_HEADS * M_HEADS;
    const int blocks = (total_elements + THREADS_PER_BLOCK_FORWARD - 1) / THREADS_PER_BLOCK_FORWARD;

    box_pair_gaussian_forward_kernel<<<blocks, THREADS_PER_BLOCK_FORWARD>>>(
        boxes1.data_ptr<float>(),
        offset1.data_ptr<float>(),
        sigma1.data_ptr<float>(),
        boxes2.data_ptr<float>(),
        offset2.data_ptr<float>(),
        sigma2.data_ptr<float>(),
        B,
        N_HEADS,
        M_HEADS,
        output.data_ptr<float>()
    );

    CHECK_CUDA_ERROR(cudaPeekAtLastError());

    return output;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    val += __shfl_down_sync(0xFFFFFFFF, val, 16);
    val += __shfl_down_sync(0xFFFFFFFF, val, 8);
    val += __shfl_down_sync(0xFFFFFFFF, val, 4);
    val += __shfl_down_sync(0xFFFFFFFF, val, 2);
    val += __shfl_down_sync(0xFFFFFFFF, val, 1);
    return __shfl_sync(0xFFFFFFFF, val, 0);
}

// Kernel to compute gradients for the first set of inputs (offset1, sigma1)
__global__ void __launch_bounds__(THREADS_PER_BLOCK_BACKWARD) box_pair_gaussian_backward_grad1_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ boxes1,
    const float* __restrict__ offset1,
    const float* __restrict__ sigma1,
    const float* __restrict__ boxes2,
    const float* __restrict__ offset2,
    const float* __restrict__ sigma2,
    int B, int N_HEADS, int M_HEADS,
    float* __restrict__ d_offset1,
    float* __restrict__ d_sigma1
) {
    const int head1_id = blockIdx.x;
    const int b = head1_id / N_HEADS;
    const int n_idx = head1_id % N_HEADS;

    __shared__ float s_data1[8]; // box1(4), offset1(2), sigma1(2)

    if (threadIdx.x == 0) {
        // MODIFIED: Indexing for boxes1 changed to support (B, N, 4) shape
        const int box1_offset = head1_id * 4;
        s_data1[0] = boxes1[box1_offset + 0];
        s_data1[1] = boxes1[box1_offset + 1];
        s_data1[2] = boxes1[box1_offset + 2] * 0.5f; // half_w
        s_data1[3] = boxes1[box1_offset + 3] * 0.5f; // half_h

        const int head1_params_offset = (b * N_HEADS + n_idx) * 2;
        s_data1[4] = offset1[head1_params_offset + 0];
        s_data1[5] = offset1[head1_params_offset + 1];
        s_data1[6] = sigma1[head1_params_offset + 0];
        s_data1[7] = sigma1[head1_params_offset + 1];
    }
    __syncthreads();

    const float half_w1 = s_data1[2];
    const float half_h1 = s_data1[3];
    const float mu1_x = s_data1[0] + s_data1[4] * half_w1;
    const float mu1_y = s_data1[1] + s_data1[5] * half_h1;
    const float sig1_x = fmaxf(s_data1[6] * half_w1, 1e-6f);
    const float sig1_y = fmaxf(s_data1[7] * half_h1, 1e-6f);

    float local_d_offset_x = 0.0f, local_d_offset_y = 0.0f;
    float local_d_sigma_x = 0.0f, local_d_sigma_y = 0.0f;

    for (int m_idx = threadIdx.x; m_idx < M_HEADS; m_idx += blockDim.x) {
        // MODIFIED: Indexing for boxes2 changed to support (B, M, 4) shape
        const int box2_offset = (b * M_HEADS + m_idx) * 4;
        const float center2_x = boxes2[box2_offset + 0];
        const float center2_y = boxes2[box2_offset + 1];
        const float half_w2 = boxes2[box2_offset + 2] * 0.5f;
        const float half_h2 = boxes2[box2_offset + 3] * 0.5f;

        const int head2_params_offset = (b * M_HEADS + m_idx) * 2;
        const float mu2_x = center2_x + offset2[head2_params_offset + 0] * half_w2;
        const float mu2_y = center2_y + offset2[head2_params_offset + 1] * half_h2;
        const float sig2_x = fmaxf(sigma2[head2_params_offset + 0] * half_w2, 1e-6f);
        const float sig2_y = fmaxf(sigma2[head2_params_offset + 1] * half_h2, 1e-6f);
        
        const float mu_diff_x = mu1_x - mu2_x;
        const float var_sum_x = sig1_x * sig1_x + sig2_x * sig2_x;
        const float bc_x = sqrtf(2.0f * sig1_x * sig2_x / var_sum_x) * __expf(-0.25f * mu_diff_x * mu_diff_x / var_sum_x);

        const float mu_diff_y = mu1_y - mu2_y;
        const float var_sum_y = sig1_y * sig1_y + sig2_y * sig2_y;
        const float bc_y = sqrtf(2.0f * sig1_y * sig2_y / var_sum_y) * __expf(-0.25f * mu_diff_y * mu_diff_y / var_sum_y);

        const int grad_idx = b * N_HEADS * M_HEADS + n_idx * M_HEADS + m_idx;
        const float g_out = grad_output[grad_idx];
        const float g_bc_x = g_out * bc_y;
        const float g_bc_y = g_out * bc_x;

        const float dL_dmu1_x = g_bc_x * bc_x * (-0.5f * mu_diff_x / var_sum_x);
        const float dL_dmu1_y = g_bc_y * bc_y * (-0.5f * mu_diff_y / var_sum_y);

        const float term1_sig1_x = (sig2_x * sig2_x - sig1_x * sig1_x) / sig1_x;
        const float term2_sig1_x = (mu_diff_x * mu_diff_x * sig1_x) / var_sum_x;
        const float dL_dsig1_x = g_bc_x * bc_x * (term1_sig1_x + term2_sig1_x) / (2.0f * var_sum_x);

        const float term1_sig1_y = (sig2_y * sig2_y - sig1_y * sig1_y) / sig1_y;
        const float term2_sig1_y = (mu_diff_y * mu_diff_y * sig1_y) / var_sum_y;
        const float dL_dsig1_y = g_bc_y * bc_y * (term1_sig1_y + term2_sig1_y) / (2.0f * var_sum_y);

        local_d_offset_x += dL_dmu1_x * half_w1;
        local_d_offset_y += dL_dmu1_y * half_h1;
        local_d_sigma_x += dL_dsig1_x * half_w1;
        local_d_sigma_y += dL_dsig1_y * half_h1;
    }

    float warp_d_offset_x = warp_reduce_sum(local_d_offset_x);
    float warp_d_offset_y = warp_reduce_sum(local_d_offset_y);
    float warp_d_sigma_x = warp_reduce_sum(local_d_sigma_x);
    float warp_d_sigma_y = warp_reduce_sum(local_d_sigma_y);

    __shared__ float s_reduction_memory[4 * (THREADS_PER_BLOCK_BACKWARD / 32)];
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int num_warps = blockDim.x / 32;

    if (lane_id == 0) {
        s_reduction_memory[warp_id] = warp_d_offset_x;
        s_reduction_memory[warp_id + num_warps] = warp_d_offset_y;
        s_reduction_memory[warp_id + 2 * num_warps] = warp_d_sigma_x;
        s_reduction_memory[warp_id + 3 * num_warps] = warp_d_sigma_y;
    }
    __syncthreads();

    if (warp_id == 0) {
        float final_d_offset_x = (lane_id < num_warps) ? s_reduction_memory[lane_id] : 0.0f;
        float final_d_offset_y = (lane_id < num_warps) ? s_reduction_memory[lane_id + num_warps] : 0.0f;
        float final_d_sigma_x = (lane_id < num_warps) ? s_reduction_memory[lane_id + 2 * num_warps] : 0.0f;
        float final_d_sigma_y = (lane_id < num_warps) ? s_reduction_memory[lane_id + 3 * num_warps] : 0.0f;

        final_d_offset_x = warp_reduce_sum(final_d_offset_x);
        final_d_offset_y = warp_reduce_sum(final_d_offset_y);
        final_d_sigma_x = warp_reduce_sum(final_d_sigma_x);
        final_d_sigma_y = warp_reduce_sum(final_d_sigma_y);

        if (lane_id == 0) {
            const int head1_params_offset = head1_id * 2;
            d_offset1[head1_params_offset + 0] = final_d_offset_x;
            d_offset1[head1_params_offset + 1] = final_d_offset_y;
            d_sigma1[head1_params_offset + 0] = final_d_sigma_x;
            d_sigma1[head1_params_offset + 1] = final_d_sigma_y;
        }
    }
}

// Kernel to compute gradients for the second set of inputs (offset2, sigma2)
__global__ void __launch_bounds__(THREADS_PER_BLOCK_BACKWARD) box_pair_gaussian_backward_grad2_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ boxes1,
    const float* __restrict__ offset1,
    const float* __restrict__ sigma1,
    const float* __restrict__ boxes2,
    const float* __restrict__ offset2,
    const float* __restrict__ sigma2,
    int B, int N_HEADS, int M_HEADS,
    float* __restrict__ d_offset2,
    float* __restrict__ d_sigma2
) {
    const int head2_id = blockIdx.x;
    const int b = head2_id / M_HEADS;
    const int m_idx = head2_id % M_HEADS;

    __shared__ float s_data2[8]; // box2(4), offset2(2), sigma2(2)

    if (threadIdx.x == 0) {
        // MODIFIED: Indexing for boxes2 changed to support (B, M, 4) shape
        const int box2_offset = head2_id * 4;
        s_data2[0] = boxes2[box2_offset + 0];
        s_data2[1] = boxes2[box2_offset + 1];
        s_data2[2] = boxes2[box2_offset + 2] * 0.5f; // half_w
        s_data2[3] = boxes2[box2_offset + 3] * 0.5f; // half_h

        const int head2_params_offset = (b * M_HEADS + m_idx) * 2;
        s_data2[4] = offset2[head2_params_offset + 0];
        s_data2[5] = offset2[head2_params_offset + 1];
        s_data2[6] = sigma2[head2_params_offset + 0];
        s_data2[7] = sigma2[head2_params_offset + 1];
    }
    __syncthreads();

    const float half_w2 = s_data2[2];
    const float half_h2 = s_data2[3];
    const float mu2_x = s_data2[0] + s_data2[4] * half_w2;
    const float mu2_y = s_data2[1] + s_data2[5] * half_h2;
    const float sig2_x = fmaxf(s_data2[6] * half_w2, 1e-6f);
    const float sig2_y = fmaxf(s_data2[7] * half_h2, 1e-6f);

    float local_d_offset_x = 0.0f, local_d_offset_y = 0.0f;
    float local_d_sigma_x = 0.0f, local_d_sigma_y = 0.0f;

    for (int n_idx = threadIdx.x; n_idx < N_HEADS; n_idx += blockDim.x) {
        // MODIFIED: Indexing for boxes1 changed to support (B, N, 4) shape
        const int box1_offset = (b * N_HEADS + n_idx) * 4;
        const float center1_x = boxes1[box1_offset + 0];
        const float center1_y = boxes1[box1_offset + 1];
        const float half_w1 = boxes1[box1_offset + 2] * 0.5f;
        const float half_h1 = boxes1[box1_offset + 3] * 0.5f;

        const int head1_params_offset = (b * N_HEADS + n_idx) * 2;
        const float mu1_x = center1_x + offset1[head1_params_offset + 0] * half_w1;
        const float mu1_y = center1_y + offset1[head1_params_offset + 1] * half_h1;
        const float sig1_x = fmaxf(sigma1[head1_params_offset + 0] * half_w1, 1e-6f);
        const float sig1_y = fmaxf(sigma1[head1_params_offset + 1] * half_h1, 1e-6f);

        const float mu_diff_x = mu1_x - mu2_x;
        const float var_sum_x = sig1_x * sig1_x + sig2_x * sig2_x;
        const float bc_x = sqrtf(2.0f * sig1_x * sig2_x / var_sum_x) * __expf(-0.25f * mu_diff_x * mu_diff_x / var_sum_x);

        const float mu_diff_y = mu1_y - mu2_y;
        const float var_sum_y = sig1_y * sig1_y + sig2_y * sig2_y;
        const float bc_y = sqrtf(2.0f * sig1_y * sig2_y / var_sum_y) * __expf(-0.25f * mu_diff_y * mu_diff_y / var_sum_y);

        const int grad_idx = b * N_HEADS * M_HEADS + n_idx * M_HEADS + m_idx;
        const float g_out = grad_output[grad_idx];
        const float g_bc_x = g_out * bc_y;
        const float g_bc_y = g_out * bc_x;

        const float dL_dmu2_x = g_bc_x * bc_x * (0.5f * mu_diff_x / var_sum_x);
        const float dL_dmu2_y = g_bc_y * bc_y * (0.5f * mu_diff_y / var_sum_y);

        const float term1_sig2_x = (sig1_x * sig1_x - sig2_x * sig2_x) / sig2_x;
        const float term2_sig2_x = (mu_diff_x * mu_diff_x * sig2_x) / var_sum_x;
        const float dL_dsig2_x = g_bc_x * bc_x * (term1_sig2_x + term2_sig2_x) / (2.0f * var_sum_x);

        const float term1_sig2_y = (sig1_y * sig1_y - sig2_y * sig2_y) / sig2_y;
        const float term2_sig2_y = (mu_diff_y * mu_diff_y * sig2_y) / var_sum_y;
        const float dL_dsig2_y = g_bc_y * bc_y * (term1_sig2_y + term2_sig2_y) / (2.0f * var_sum_y);

        local_d_offset_x += dL_dmu2_x * half_w2;
        local_d_offset_y += dL_dmu2_y * half_h2;
        local_d_sigma_x += dL_dsig2_x * half_w2;
        local_d_sigma_y += dL_dsig2_y * half_h2;
    }

    float warp_d_offset_x = warp_reduce_sum(local_d_offset_x);
    float warp_d_offset_y = warp_reduce_sum(local_d_offset_y);
    float warp_d_sigma_x = warp_reduce_sum(local_d_sigma_x);
    float warp_d_sigma_y = warp_reduce_sum(local_d_sigma_y);

    __shared__ float s_reduction_memory[4 * (THREADS_PER_BLOCK_BACKWARD / 32)];
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int num_warps = blockDim.x / 32;

    if (lane_id == 0) {
        s_reduction_memory[warp_id] = warp_d_offset_x;
        s_reduction_memory[warp_id + num_warps] = warp_d_offset_y;
        s_reduction_memory[warp_id + 2 * num_warps] = warp_d_sigma_x;
        s_reduction_memory[warp_id + 3 * num_warps] = warp_d_sigma_y;
    }
    __syncthreads();

    if (warp_id == 0) {
        float final_d_offset_x = (lane_id < num_warps) ? s_reduction_memory[lane_id] : 0.0f;
        float final_d_offset_y = (lane_id < num_warps) ? s_reduction_memory[lane_id + num_warps] : 0.0f;
        float final_d_sigma_x = (lane_id < num_warps) ? s_reduction_memory[lane_id + 2 * num_warps] : 0.0f;
        float final_d_sigma_y = (lane_id < num_warps) ? s_reduction_memory[lane_id + 3 * num_warps] : 0.0f;

        final_d_offset_x = warp_reduce_sum(final_d_offset_x);
        final_d_offset_y = warp_reduce_sum(final_d_offset_y);
        final_d_sigma_x = warp_reduce_sum(final_d_sigma_x);
        final_d_sigma_y = warp_reduce_sum(final_d_sigma_y);

        if (lane_id == 0) {
            const int head2_params_offset = head2_id * 2;
            d_offset2[head2_params_offset + 0] = final_d_offset_x;
            d_offset2[head2_params_offset + 1] = final_d_offset_y;
            d_sigma2[head2_params_offset + 0] = final_d_sigma_x;
            d_sigma2[head2_params_offset + 1] = final_d_sigma_y;
        }
    }
}

std::vector<torch::Tensor> fused_box_pair_gaussian_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& boxes1,
    const torch::Tensor& offset1,
    const torch::Tensor& sigma1,
    const torch::Tensor& boxes2,
    const torch::Tensor& offset2,
    const torch::Tensor& sigma2
) {
    CHECK_INPUT(grad_output); CHECK_INPUT(boxes1); CHECK_INPUT(offset1); CHECK_INPUT(sigma1);
    CHECK_INPUT(boxes2); CHECK_INPUT(offset2); CHECK_INPUT(sigma2);

    const int B = boxes1.size(0);
    const int N_HEADS = offset1.size(1);
    const int M_HEADS = offset2.size(1);

    auto d_offset1 = torch::zeros_like(offset1);
    auto d_sigma1 = torch::zeros_like(sigma1);
    auto d_offset2 = torch::zeros_like(offset2);
    auto d_sigma2 = torch::zeros_like(sigma2);

    const int block_dim = THREADS_PER_BLOCK_BACKWARD;

    if (N_HEADS > 0) {
        const int grid_dim1 = B * N_HEADS;
        box_pair_gaussian_backward_grad1_kernel<<<grid_dim1, block_dim>>>(
            grad_output.data_ptr<float>(), boxes1.data_ptr<float>(), offset1.data_ptr<float>(), sigma1.data_ptr<float>(),
            boxes2.data_ptr<float>(), offset2.data_ptr<float>(), sigma2.data_ptr<float>(),
            B, N_HEADS, M_HEADS,
            d_offset1.data_ptr<float>(), d_sigma1.data_ptr<float>()
        );
    }
    
    if (M_HEADS > 0) {
        const int grid_dim2 = B * M_HEADS;
        box_pair_gaussian_backward_grad2_kernel<<<grid_dim2, block_dim>>>(
            grad_output.data_ptr<float>(), boxes1.data_ptr<float>(), offset1.data_ptr<float>(), sigma1.data_ptr<float>(),
            boxes2.data_ptr<float>(), offset2.data_ptr<float>(), sigma2.data_ptr<float>(),
            B, N_HEADS, M_HEADS,
            d_offset2.data_ptr<float>(), d_sigma2.data_ptr<float>()
        );
    }

    CHECK_CUDA_ERROR(cudaPeekAtLastError());

    return {d_offset1, d_sigma1, d_offset2, d_sigma2};
}