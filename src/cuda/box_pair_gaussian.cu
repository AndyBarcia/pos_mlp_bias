#include <ATen/native/cuda/KernelUtils.cuh>
#include <c10/util/Half.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

#include "utils.h"

const int THREADS_PER_BLOCK_FORWARD = 1024;
const int THREADS_PER_BLOCK_BACKWARD = 1024;

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
    const int box1_offset = b * 4;
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
    const int box2_offset = b * 4;
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

    // A dynamic kernel is sufficient here as the number of heads is less critical for performance
    // than spatial dimensions, and it simplifies the implementation significantly.
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

__global__ void box_pair_gaussian_backward_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ boxes1,
    const float* __restrict__ offset1,
    const float* __restrict__ sigma1,
    const float* __restrict__ boxes2,
    const float* __restrict__ offset2,
    const float* __restrict__ sigma2,
    int B,
    int N_HEADS,
    int M_HEADS,
    float* __restrict__ d_offset1,
    float* __restrict__ d_sigma1,
    float* __restrict__ d_offset2,
    float* __restrict__ d_sigma2
) {
    const int total_elements = B * N_HEADS * M_HEADS;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_elements) return;

    // Decompose the global thread index to get batch and head indices
    const int b = tid / (N_HEADS * M_HEADS);
    const int n_idx = (tid / M_HEADS) % N_HEADS;
    const int m_idx = tid % M_HEADS;

    // --- 1. Re-compute the forward pass for the current pair ---

    // Gaussian 1 (from set 1, head n)
    const int box1_offset = b * 4;
    const float half_w1 = boxes1[box1_offset + 2] * 0.5f;
    const float half_h1 = boxes1[box1_offset + 3] * 0.5f;
    
    const int head1_params_offset = (b * N_HEADS + n_idx) * 2;
    const float mu1_x = boxes1[box1_offset + 0] + offset1[head1_params_offset + 0] * half_w1;
    const float mu1_y = boxes1[box1_offset + 1] + offset1[head1_params_offset + 1] * half_h1;
    const float sig1_x = fmaxf(sigma1[head1_params_offset + 0] * half_w1, 1e-6f);
    const float sig1_y = fmaxf(sigma1[head1_params_offset + 1] * half_h1, 1e-6f);

    // Gaussian 2 (from set 2, head m)
    const int box2_offset = b * 4;
    const float half_w2 = boxes2[box2_offset + 2] * 0.5f;
    const float half_h2 = boxes2[box2_offset + 3] * 0.5f;

    const int head2_params_offset = (b * M_HEADS + m_idx) * 2;
    const float mu2_x = boxes2[box2_offset + 0] + offset2[head2_params_offset + 0] * half_w2;
    const float mu2_y = boxes2[box2_offset + 1] + offset2[head2_params_offset + 1] * half_h2;
    const float sig2_x = fmaxf(sigma2[head2_params_offset + 0] * half_w2, 1e-6f);
    const float sig2_y = fmaxf(sigma2[head2_params_offset + 1] * half_h2, 1e-6f);

    // Bhattacharyya coefficients per dimension
    const float mu_diff_x = mu1_x - mu2_x;
    const float var_sum_x = sig1_x * sig1_x + sig2_x * sig2_x;
    const float bc_x = sqrtf(2.0f * sig1_x * sig2_x / var_sum_x) * __expf(-0.25f * mu_diff_x * mu_diff_x / var_sum_x);

    const float mu_diff_y = mu1_y - mu2_y;
    const float var_sum_y = sig1_y * sig1_y + sig2_y * sig2_y;
    const float bc_y = sqrtf(2.0f * sig1_y * sig2_y / var_sum_y) * __expf(-0.25f * mu_diff_y * mu_diff_y / var_sum_y);

    // --- 2. Calculate gradients using the chain rule ---

    const float g_out = grad_output[tid];

    // Incoming gradients for each 1D BC
    const float g_bc_x = g_out * bc_y;
    const float g_bc_y = g_out * bc_x;

    // Gradient w.r.t. effective centers (mu)
    const float dL_dmu1_x = g_bc_x * bc_x * (-0.5f * mu_diff_x / var_sum_x);
    const float dL_dmu1_y = g_bc_y * bc_y * (-0.5f * mu_diff_y / var_sum_y);
    const float dL_dmu2_x = -dL_dmu1_x;
    const float dL_dmu2_y = -dL_dmu1_y;

    // Gradient w.r.t. effective sigmas
    const float term1_sig1_x = (sig2_x * sig2_x - sig1_x * sig1_x) / sig1_x;
    const float term2_sig1_x = (mu_diff_x * mu_diff_x * sig1_x) / var_sum_x;
    const float dL_dsig1_x = g_bc_x * bc_x * (term1_sig1_x + term2_sig1_x) / (2.0f * var_sum_x);

    const float term1_sig1_y = (sig2_y * sig2_y - sig1_y * sig1_y) / sig1_y;
    const float term2_sig1_y = (mu_diff_y * mu_diff_y * sig1_y) / var_sum_y;
    const float dL_dsig1_y = g_bc_y * bc_y * (term1_sig1_y + term2_sig1_y) / (2.0f * var_sum_y);

    const float term1_sig2_x = (sig1_x * sig1_x - sig2_x * sig2_x) / sig2_x;
    const float term2_sig2_x = (mu_diff_x * mu_diff_x * sig2_x) / var_sum_x;
    const float dL_dsig2_x = g_bc_x * bc_x * (term1_sig2_x + term2_sig2_x) / (2.0f * var_sum_x);

    const float term1_sig2_y = (sig1_y * sig1_y - sig2_y * sig2_y) / sig2_y;
    const float term2_sig2_y = (mu_diff_y * mu_diff_y * sig2_y) / var_sum_y;
    const float dL_dsig2_y = g_bc_y * bc_y * (term1_sig2_y + term2_sig2_y) / (2.0f * var_sum_y);

    // --- 3. Chain gradients back to original inputs ---

    const float d_offset1_x = dL_dmu1_x * half_w1;
    const float d_offset1_y = dL_dmu1_y * half_h1;
    const float d_sigma1_x = dL_dsig1_x * half_w1;
    const float d_sigma1_y = dL_dsig1_y * half_h1;

    const float d_offset2_x = dL_dmu2_x * half_w2;
    const float d_offset2_y = dL_dmu2_y * half_h2;
    const float d_sigma2_x = dL_dsig2_x * half_w2;
    const float d_sigma2_y = dL_dsig2_y * half_h2;

    // --- 4. Atomically add gradients to output tensors ---

    atomicAdd(&d_offset1[head1_params_offset + 0], d_offset1_x);
    atomicAdd(&d_offset1[head1_params_offset + 1], d_offset1_y);
    atomicAdd(&d_sigma1[head1_params_offset + 0], d_sigma1_x);
    atomicAdd(&d_sigma1[head1_params_offset + 1], d_sigma1_y);

    atomicAdd(&d_offset2[head2_params_offset + 0], d_offset2_x);
    atomicAdd(&d_offset2[head2_params_offset + 1], d_offset2_y);
    atomicAdd(&d_sigma2[head2_params_offset + 0], d_sigma2_x);
    atomicAdd(&d_sigma2[head2_params_offset + 1], d_sigma2_y);
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
    CHECK_INPUT(grad_output);
    CHECK_INPUT(boxes1);
    CHECK_INPUT(offset1);
    CHECK_INPUT(sigma1);
    CHECK_INPUT(boxes2);
    CHECK_INPUT(offset2);
    CHECK_INPUT(sigma2);

    const int B = boxes1.size(0);
    const int N_HEADS = offset1.size(1);
    const int M_HEADS = offset2.size(1);

    auto d_offset1 = torch::zeros_like(offset1);
    auto d_sigma1 = torch::zeros_like(sigma1);
    auto d_offset2 = torch::zeros_like(offset2);
    auto d_sigma2 = torch::zeros_like(sigma2);

    const int total_elements = B * N_HEADS * M_HEADS;
    if (total_elements == 0) {
        return {d_offset1, d_sigma1, d_offset2, d_sigma2};
    }
    
    const int blocks = (total_elements + THREADS_PER_BLOCK_BACKWARD - 1) / THREADS_PER_BLOCK_BACKWARD;

    box_pair_gaussian_backward_kernel<<<blocks, THREADS_PER_BLOCK_BACKWARD>>>(
        grad_output.data_ptr<float>(),
        boxes1.data_ptr<float>(),
        offset1.data_ptr<float>(),
        sigma1.data_ptr<float>(),
        boxes2.data_ptr<float>(),
        offset2.data_ptr<float>(),
        sigma2.data_ptr<float>(),
        B,
        N_HEADS,
        M_HEADS,
        d_offset1.data_ptr<float>(),
        d_sigma1.data_ptr<float>(),
        d_offset2.data_ptr<float>(),
        d_sigma2.data_ptr<float>()
    );

    CHECK_CUDA_ERROR(cudaPeekAtLastError());

    // Note: Gradients for boxes1 and boxes2 are not computed and implicitly returned as None/zero.
    return {d_offset1, d_sigma1, d_offset2, d_sigma2};
}