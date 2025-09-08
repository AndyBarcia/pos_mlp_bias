#include <ATen/native/cuda/KernelUtils.cuh>
#include <c10/util/Half.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

#include "utils.h"

const int THREADS_PER_BLOCK_FORWARD = 512;
const int THREADS_PER_BLOCK_BACKWARD = 256;

template <int C_HIDDEN>
__global__ void box_pair_brbp_kernel(
    const float* __restrict__ mlp_weights, // Shape: (B, N1, 6*C'+1)
    const float* __restrict__ boxes1,
    const float* __restrict__ boxes2,
    int B,
    int N1,
    int N2,
    float* __restrict__ output
) {
    const int total_elements = B * N1 * N2;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= total_elements) return;

    const int b = tid / (N1 * N2);
    const int n1n2_idx = tid % (N1 * N2);
    const int n1 = n1n2_idx / N2;
    const int n2 = n1n2_idx % N2;

    const float* p_box1 = boxes1 + (b * N1 + n1) * 4;
    const float* p_box2 = boxes2 + (b * N2 + n2) * 4;

    const float cx1 = p_box1[0];
    const float cy1 = p_box1[1];
    const float w1 = p_box1[2];
    const float h1 = p_box1[3];

    const float cx2 = p_box2[0];
    const float cy2 = p_box2[1];
    const float w2 = p_box2[2];
    const float h2 = p_box2[3];

    const float epsilon = 1e-6f;
    const float dx = (cx2 - cx1) / (w1 + epsilon);
    const float dy = (cy2 - cy1) / (h1 + epsilon);
    const float dw = logf(w2 / (w1 + epsilon));
    const float dh = logf(h2 / (h1 + epsilon));
    
    // Select the correct MLP weights for the current (b, n1) pair
    const int grad_size = 6 * C_HIDDEN + 1;
    const float* p_mlp_weights = mlp_weights + (b * N1 + n1) * grad_size;

    const float* w1_ptr = p_mlp_weights;
    const float* b1_ptr = p_mlp_weights + 4 * C_HIDDEN;
    const float* w2_ptr = p_mlp_weights + 5 * C_HIDDEN;
    const float b2_val = p_mlp_weights[6 * C_HIDDEN];

    float temp[C_HIDDEN];
    float max_val = -FLT_MAX;

    // First MLP layer
    for (int k = 0; k < C_HIDDEN; k++) {
        const float w_dx = w1_ptr[k];
        const float w_dy = w1_ptr[C_HIDDEN + k];
        const float w_dw = w1_ptr[2 * C_HIDDEN + k];
        const float w_dh = w1_ptr[3 * C_HIDDEN + k];
        const float b1 = b1_ptr[k];
        
        temp[k] = dx * w_dx + dy * w_dy + dw * w_dw + dh * w_dh + b1;
        if (temp[k] > max_val) max_val = temp[k];
    }

    // Softmax
    float sum_exp = 0.0f;
    for (int k = 0; k < C_HIDDEN; k++) {
        temp[k] = __expf(temp[k] - max_val);
        sum_exp += temp[k];
    }

    const float inv_sum_exp = 1.0f / sum_exp;
    for (int k = 0; k < C_HIDDEN; k++) {
        temp[k] *= inv_sum_exp;
    }

    // Second MLP layer
    float out_val = 0.0f;
    for (int k = 0; k < C_HIDDEN; k++) {
        out_val += temp[k] * w2_ptr[k];
    }
    out_val += b2_val;

    output[tid] = out_val;
}

template <int MAX_C_HIDDEN>
__global__ void box_pair_brbp_dynamic_kernel(
    const float* __restrict__ mlp_weights, // Shape: (B, N1, 6*C'+1)
    const float* __restrict__ boxes1,
    const float* __restrict__ boxes2,
    int B,
    int N1,
    int N2,
    int C_HIDDEN,
    float* __restrict__ output
) {
    const int total_elements = B * N1 * N2;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= total_elements) return;

    const int b = tid / (N1 * N2);
    const int n1n2_idx = tid % (N1 * N2);
    const int n1 = n1n2_idx / N2;
    const int n2 = n1n2_idx % N2;

    const float* p_box1 = boxes1 + (b * N1 + n1) * 4;
    const float* p_box2 = boxes2 + (b * N2 + n2) * 4;

    const float cx1 = p_box1[0];
    const float cy1 = p_box1[1];
    const float w1 = p_box1[2];
    const float h1 = p_box1[3];

    const float cx2 = p_box2[0];
    const float cy2 = p_box2[1];
    const float w2 = p_box2[2];
    const float h2 = p_box2[3];

    const float epsilon = 1e-6f;
    const float dx = (cx2 - cx1) / (w1 + epsilon);
    const float dy = (cy2 - cy1) / (h1 + epsilon);
    const float dw = logf(w2 / (w1 + epsilon));
    const float dh = logf(h2 / (h1 + epsilon));
    
    // Select the correct MLP weights for the current (b, n1) pair
    const int grad_size = 6 * C_HIDDEN + 1;
    const float* p_mlp_weights = mlp_weights + (b * N1 + n1) * grad_size;

    const float* w1_ptr = p_mlp_weights;
    const float* b1_ptr = p_mlp_weights + 4 * C_HIDDEN;
    const float* w2_ptr = p_mlp_weights + 5 * C_HIDDEN;
    const float b2_val = p_mlp_weights[6 * C_HIDDEN];

    float temp[MAX_C_HIDDEN];
    float max_val = -FLT_MAX;

    // First MLP layer
    for (int k = 0; k < C_HIDDEN; k++) {
        const float w_dx = w1_ptr[k];
        const float w_dy = w1_ptr[C_HIDDEN + k];
        const float w_dw = w1_ptr[2 * C_HIDDEN + k];
        const float w_dh = w1_ptr[3 * C_HIDDEN + k];
        const float b1 = b1_ptr[k];
        
        temp[k] = dx * w_dx + dy * w_dy + dw * w_dw + dh * w_dh + b1;
        if (temp[k] > max_val) max_val = temp[k];
    }

    // Softmax
    float sum_exp = 0.0f;
    for (int k = 0; k < C_HIDDEN; k++) {
        temp[k] = __expf(temp[k] - max_val);
        sum_exp += temp[k];
    }

    const float inv_sum_exp = 1.0f / sum_exp;
    for (int k = 0; k < C_HIDDEN; k++) {
        temp[k] *= inv_sum_exp;
    }

    // Second MLP layer
    float out_val = 0.0f;
    for (int k = 0; k < C_HIDDEN; k++) {
        out_val += temp[k] * w2_ptr[k];
    }
    out_val += b2_val;

    output[tid] = out_val;
}

torch::Tensor fused_box_pair_brpb_forward(
    const torch::Tensor& mlp_weights, // (B, N1, [4*C' + C' + 1*C' + 1])
    const torch::Tensor& pos1,        // (B, N1, [x,y,w,h])
    const torch::Tensor& pos2,        // (B, N2, [x,y,w,h])
    const int c_hidden
) {
    CHECK_INPUT(mlp_weights);
    CHECK_INPUT(pos1);
    CHECK_INPUT(pos2);
    TORCH_CHECK(c_hidden <= 16, "c_hidden must be <= 16");

    const int B = pos1.size(0);
    const int N1 = pos1.size(1);
    const int N2 = pos2.size(1);

    auto output = torch::empty({B, N1, N2}, mlp_weights.options());
    const int total_elements = B * N1 * N2;
    const int blocks = (total_elements + THREADS_PER_BLOCK_FORWARD - 1) / THREADS_PER_BLOCK_FORWARD;

    if (c_hidden == 16) {
        box_pair_brbp_kernel<16><<<blocks, THREADS_PER_BLOCK_FORWARD>>>(
            mlp_weights.data_ptr<float>(),
            pos1.data_ptr<float>(),
            pos2.data_ptr<float>(),
            B,
            N1,
            N2,
            output.data_ptr<float>()
        );
    } else {
        box_pair_brbp_dynamic_kernel<16><<<blocks, THREADS_PER_BLOCK_FORWARD>>>(
            mlp_weights.data_ptr<float>(),
            pos1.data_ptr<float>(),
            pos2.data_ptr<float>(),
            B,
            N1,
            N2,
            c_hidden,
            output.data_ptr<float>()
        );
    }

    CHECK_CUDA_ERROR(cudaPeekAtLastError());

    return output;
}

template <int C_HIDDEN>
__global__ void box_pair_brbp_backward_kernel(
    const float* __restrict__ mlp_weights,
    const float* __restrict__ boxes1,
    const float* __restrict__ boxes2,
    const float* __restrict__ grad_output,
    int B,
    int N1,
    int N2,
    float* __restrict__ grad_weights
) {
    // Each block processes one (b, n1) instance.
    const int b = blockIdx.x;
    const int n1 = blockIdx.y;
    const int tid = threadIdx.x;
    const int grad_size = 6 * C_HIDDEN + 1;

    extern __shared__ float s_mem[];

    // Use shared memory for weights and for gradient reduction
    float* s_weights = s_mem;
    float* s_grad = s_mem + grad_size;

    // Use private, register-based array for local gradient accumulation.
    float p_grad[6 * C_HIDDEN + 1];
    for (int i = 0; i < grad_size; ++i) {
        p_grad[i] = 0.0f;
    }

    // Cooperatively load the shared mlp_weights into shared memory.
    if (tid < grad_size) {
        s_weights[tid] = mlp_weights[(b * N1 + n1) * grad_size + tid];
    }

    // 0-initialize shared memory for gradients
    if (tid < grad_size) {
        s_grad[tid] = 0.0f;
    }
    __syncthreads();

    // Loop over all boxes in boxes2 for the current (b, n1) pair.
    // Each thread computes gradients for a subset of n2.
    for (int n2 = tid; n2 < N2; n2 += blockDim.x) {
        const float* p_box1 = boxes1 + (b * N1 + n1) * 4;
        const float* p_box2 = boxes2 + (b * N2 + n2) * 4;
        
        const float cx1 = p_box1[0];
        const float cy1 = p_box1[1];
        const float w1 = p_box1[2];
        const float h1 = p_box1[3];

        const float cx2 = p_box2[0];
        const float cy2 = p_box2[1];
        const float w2 = p_box2[2];
        const float h2 = p_box2[3];

        const float epsilon = 1e-6f;
        const float dx = (cx2 - cx1) / (w1 + epsilon);
        const float dy = (cy2 - cy1) / (h1 + epsilon);
        const float dw = logf(w2 / (w1 + epsilon));
        const float dh = logf(h2 / (h1 + epsilon));

        const float rel_features[4] = {dx, dy, dw, dh};
        
        const float* w1_ptr = s_weights;
        const float* b1_ptr = s_weights + 4 * C_HIDDEN;
        const float* w2_ptr = s_weights + 5 * C_HIDDEN;

        float hidden_activations[C_HIDDEN];
        float softmax_output[C_HIDDEN];
        float max_val = -FLT_MAX;

        for (int k = 0; k < C_HIDDEN; k++) {
            hidden_activations[k] = rel_features[0] * w1_ptr[k]
                                + rel_features[1] * w1_ptr[C_HIDDEN + k]
                                + rel_features[2] * w1_ptr[2 * C_HIDDEN + k]
                                + rel_features[3] * w1_ptr[3 * C_HIDDEN + k]
                                + b1_ptr[k];
            if (hidden_activations[k] > max_val) {
                max_val = hidden_activations[k];
            }
        }
        
        float sum_exp = 0.0f;
        for (int k = 0; k < C_HIDDEN; k++) {
            softmax_output[k] = __expf(hidden_activations[k] - max_val);
            sum_exp += softmax_output[k];
        }

        const float inv_sum_exp = 1.0f / sum_exp;
        for (int k = 0; k < C_HIDDEN; k++) {
            softmax_output[k] *= inv_sum_exp;
        }

        const float dL_doutput = grad_output[b * N1 * N2 + n1 * N2 + n2];

        // Gradient for b2
        p_grad[6 * C_HIDDEN] += dL_doutput;

        float output_minus_b2 = 0.0f;
        for(int k=0; k<C_HIDDEN; ++k){
            output_minus_b2 += softmax_output[k] * w2_ptr[k];
        }

        // Gradients for w2, w1, and b1
        for (int k = 0; k < C_HIDDEN; k++) {
            p_grad[5 * C_HIDDEN + k] += dL_doutput * softmax_output[k];
            
            float dL_dx_k = dL_doutput * softmax_output[k] * (w2_ptr[k] - output_minus_b2);

            p_grad[k] += dL_dx_k * rel_features[0];
            p_grad[C_HIDDEN + k] += dL_dx_k * rel_features[1];
            p_grad[2 * C_HIDDEN + k] += dL_dx_k * rel_features[2];
            p_grad[3 * C_HIDDEN + k] += dL_dx_k * rel_features[3];
            p_grad[4 * C_HIDDEN + k] += dL_dx_k;
        }
    }
    
    __syncthreads();

    // Each thread contributes its p_grad values.
    for (int i = 0; i < grad_size; ++i) {
        float val = p_grad[i];

        // Perform a warp-level reduction using shuffle instructions.
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }

        // The first thread in each warp (the "warp leader") writes the
        // partial sum for its warp to shared memory. This greatly reduces contention.
        if ((tid % 32) == 0) {
            atomicAdd(&s_grad[i], val);
        }
    }
    
    __syncthreads();

    // Write final result from shared memory to global memory.
    if (tid < grad_size) {
        grad_weights[(b * N1 + n1) * grad_size + tid] = s_grad[tid];
    }
}


torch::Tensor fused_box_pair_brpb_backward(
    const torch::Tensor& grad_out,    // (B, N1, N2)
    const torch::Tensor& mlp_weights, // (B, N1, [4*C' + C' + 1*C' + 1])
    const torch::Tensor& pos1,        // (B, N1, [x,y,w,h])
    const torch::Tensor& pos2,        // (B, N2, [x,y,w,h])
    const int c_hidden
) {
    CHECK_INPUT(grad_out);
    CHECK_INPUT(mlp_weights);
    CHECK_INPUT(pos1);
    CHECK_INPUT(pos2);
    TORCH_CHECK(c_hidden != 16, "c_hidden must be 16 for the backward pass");

    const int B = pos1.size(0);
    const int N1 = pos1.size(1);
    const int N2 = pos2.size(1);

    auto grad_weights = torch::zeros_like(mlp_weights);
    const int grad_size = 6 * c_hidden + 1;
    const size_t shared_mem_size = 2 * grad_size * sizeof(float);

    // Launch a 2D grid of blocks: (B, N1)
    const dim3 grid(B, N1);

    box_pair_brbp_backward_kernel<16><<<grid, THREADS_PER_BLOCK_BACKWARD, shared_mem_size>>>(
        mlp_weights.data_ptr<float>(),
        pos1.data_ptr<float>(),
        pos2.data_ptr<float>(),
        grad_out.data_ptr<float>(),
        B,
        N1,
        N2,
        grad_weights.data_ptr<float>()
    );

    CHECK_CUDA_ERROR(cudaPeekAtLastError());

    return grad_weights;
}