#include <ATen/native/cuda/KernelUtils.cuh>
#include <c10/util/Half.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

#include "utils.h"

const int THREADS_PER_BLOCK_FORWARD = 512;
const int THREADS_PER_BLOCK = 256;

template <int HEIGHT, int WIDTH, int C_HIDDEN>
__global__ void __launch_bounds__(THREADS_PER_BLOCK_FORWARD, 2) pos_mlp_bias_forward_kernel(
    const float* __restrict__ mlp_weights,
    const float* __restrict__ pos,
    int B,
    float* __restrict__ output
) {
    const int total_elements = B * HEIGHT * WIDTH;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_elements) return;

    const int b = tid / (HEIGHT * WIDTH);
    const int spatial_idx = tid % (HEIGHT * WIDTH);

    const float i = ((float) (spatial_idx / WIDTH)) / (float) (HEIGHT-1);
    const float j = ((float) (spatial_idx % WIDTH)) / (float) (WIDTH-1);

    const float cx = pos[b * 4 + 0];
    const float cy = pos[b * 4 + 1];
    const float half_w = fmaxf(pos[b * 4 + 2] * 0.5f, 1e-6f);
    const float half_h = fmaxf(pos[b * 4 + 3] * 0.5f, 1e-6f);

    const float rel_x = (j - cx) / half_w;
    const float rel_y = (i - cy) / half_h;

    const int weights_offset = b * (4 * C_HIDDEN + 1);
    const float* w = &mlp_weights[weights_offset];

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
    for (int k = 0; k < C_HIDDEN; k++) {
        out_val += temp[k] * w[3 * C_HIDDEN + k];
    }
    out_val += w[4 * C_HIDDEN];

    output[tid] = out_val;
}

template <int MAX_C_HIDDEN>
__global__ void __launch_bounds__(THREADS_PER_BLOCK_FORWARD, 2) pos_mlp_bias_forward_dynamic_kernel(
    const float* __restrict__ mlp_weights,
    const float* __restrict__ pos,
    int B,
    int HEIGHT,
    int WIDTH,
    int C_HIDDEN,
    float* __restrict__ output
) {
    const int total_elements = B * HEIGHT * WIDTH;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_elements) return;

    const int b = tid / (HEIGHT * WIDTH);
    const int spatial_idx = tid % (HEIGHT * WIDTH);

    const float i = ((float) (spatial_idx / WIDTH)) / (float) (HEIGHT-1);
    const float j = ((float) (spatial_idx % WIDTH)) / (float) (WIDTH-1);

    const float cx = pos[b * 4 + 0];
    const float cy = pos[b * 4 + 1];
    const float half_w = fmaxf(pos[b * 4 + 2] * 0.5f, 1e-6f);
    const float half_h = fmaxf(pos[b * 4 + 3] * 0.5f, 1e-6f);

    const float rel_x = (j - cx) / half_w;
    const float rel_y = (i - cy) / half_h;

    const int weights_offset = b * (4 * C_HIDDEN + 1);
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
    for (int k = 0; k < C_HIDDEN; k++) {
        out_val += temp[k] * w[3 * C_HIDDEN + k];
    }
    out_val += w[4 * C_HIDDEN];

    output[tid] = out_val;
}

torch::Tensor fused_box_brpb_forward(
    const torch::Tensor& mlp_weights, // (B, [2*C' + C' + 1*C' + 1])
    const torch::Tensor& pos,    // (B,[x,y,w,h])
    const int c_hidden,
    const int height,
    const int width
) {
    CHECK_INPUT(mlp_weights);
    CHECK_INPUT(pos);

    const int B = mlp_weights.size(0);

    auto output = torch::empty({B, height, width}, mlp_weights.options());
    const int total_elements = B * height * width;
    const int blocks = (total_elements + THREADS_PER_BLOCK_FORWARD - 1) / THREADS_PER_BLOCK_FORWARD;

    // Templated kernel launcher
    auto launch_kernel = [&](auto... Dims) {
        pos_mlp_bias_forward_kernel<decltype(Dims)::value...><<<blocks, THREADS_PER_BLOCK_FORWARD>>>(
            mlp_weights.data_ptr<float>(),
            pos.data_ptr<float>(),
            B,
            output.data_ptr<float>()
        );
    };

    // Dynamic kernel launcher (fallback)
    auto launch_dynamic_kernel = [&]() {
        pos_mlp_bias_forward_dynamic_kernel<16><<<blocks, THREADS_PER_BLOCK_FORWARD>>>(
            mlp_weights.data_ptr<float>(),
            pos.data_ptr<float>(),
            B,
            height, 
            width, 
            c_hidden,
            output.data_ptr<float>()
        );
    };

    // Define the supported dimensions in a clear and centralized way.
    const auto supported_dims = std::make_tuple(
        std::make_tuple(
            std::integral_constant<int, 8>{}, 
            std::integral_constant<int, 16>{}, 
            std::integral_constant<int, 32>{}, 
            std::integral_constant<int, 64>{},
            std::integral_constant<int, 128>{}
        ), // height
        std::make_tuple(
            std::integral_constant<int, 8>{}, 
            std::integral_constant<int, 16>{}, 
            std::integral_constant<int, 32>{}, 
            std::integral_constant<int, 64>{},
            std::integral_constant<int, 128>{}
        ), // width
        std::make_tuple(std::integral_constant<int, 16>{})  // c_hidden
    );

    // The runtime values that need to be dispatched.
    const auto runtime_dims = std::make_tuple(height, width, c_hidden);

    // Call the generalized dispatcher.
    dispatch_kernel_with_fallback(launch_kernel, launch_dynamic_kernel, runtime_dims, supported_dims);

    CHECK_CUDA_ERROR(cudaPeekAtLastError());

    return output;
}

template <int HEIGHT, int WIDTH, int C_HIDDEN>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, 4) pos_mlp_bias_backward_kernel(
    const float* __restrict__ mlp_weights,
    const float* __restrict__ pos,
    const float* __restrict__ grad_output,
    int B,
    float* __restrict__ grad_weights
) {

    int b = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ float s_mem[];
    int grad_size = 4 * C_HIDDEN + 1;

    // Use shared memory for weights and for gradient reduction
    float* s_weights = s_mem;
    float* s_grad = s_mem + grad_size;

    // Use a private, register-based array for gradient accumulation.
    // Initialize it to zero.
    float p_grad[4 * C_HIDDEN + 1]; // Use the compile-time constant
    for (int i = 0; i < grad_size; ++i) {
        p_grad[i] = 0.0f;
    }

    // Cooperatively load mlp_weights into shared memory.
    // This access is fully coalesced.
    if (tid < grad_size) {
        s_weights[tid] = mlp_weights[b * grad_size + tid];
    }

    // 0-initialize shared memory for gradients
    if (tid < grad_size) {
        s_grad[tid] = 0.0f;
    }
    __syncthreads();

    const float cx = pos[b * 4 + 0];
    const float cy = pos[b * 4 + 1];
    const float half_w = fmaxf(pos[b * 4 + 2] * 0.5f, 1e-6f);
    const float half_h = fmaxf(pos[b * 4 + 3] * 0.5f, 1e-6f);

    const float* grad_out_b = &grad_output[b * HEIGHT * WIDTH];

    for (int idx = tid; idx < HEIGHT * WIDTH; idx += blockDim.x) {
        int i = idx / WIDTH;
        int j = idx % WIDTH;

        float rel_x = ((((float)j) / (float)(WIDTH - 1)) - cx) / half_w;
        float rel_y = ((((float)i) / (float)(HEIGHT - 1)) - cy) / half_h;

        float x[C_HIDDEN];
        float s[C_HIDDEN];

        float max_x = -FLT_MAX;
        for (int k = 0; k < C_HIDDEN; k++) {
            x[k] = rel_x * s_weights[2 * k] + rel_y * s_weights[2 * k + 1] + s_weights[2 * C_HIDDEN + k];
            if (x[k] > max_x) max_x = x[k];
        }

        float sum_exp = 0.0f;
        for (int k = 0; k < C_HIDDEN; k++) {
            s[k] = __expf(x[k] - max_x);
            sum_exp += s[k];
        }

        float inv_sum = 1.0f / sum_exp;
        for (int k = 0; k < C_HIDDEN; k++) {
            s[k] *= inv_sum;
        }

        float dL_doutput_bij = grad_out_b[i * WIDTH + j];

        float output_minus_b2 = 0.0f;
        for (int k = 0; k < C_HIDDEN; k++) {
            output_minus_b2 += s[k] * s_weights[3 * C_HIDDEN + k];
        }

        p_grad[4 * C_HIDDEN] += dL_doutput_bij;
        for (int k = 0; k < C_HIDDEN; k++) {
            p_grad[3 * C_HIDDEN + k] += dL_doutput_bij * s[k];
            float dL_dx_k = s[k] * (dL_doutput_bij * s_weights[3 * C_HIDDEN + k] - dL_doutput_bij * output_minus_b2);
            p_grad[2 * k] += dL_dx_k * rel_x;
            p_grad[2 * k + 1] += dL_dx_k * rel_y;
            p_grad[2 * C_HIDDEN + k] += dL_dx_k;
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
        grad_weights[b * grad_size + tid] = s_grad[tid];
    }
}

torch::Tensor fused_box_brpb_backward(
    const torch::Tensor& grad_out, // (B, HEIGHT, WIDTH)
    const torch::Tensor& mlp_weights, // (B, [2*C' + C' + 1*C' + 1])
    const torch::Tensor& pos,    // (B,[x,y,w,h])
    const int c_hidden
) {
    CHECK_INPUT(grad_out);
    CHECK_INPUT(mlp_weights);
    CHECK_INPUT(pos);

    const int B = mlp_weights.size(0);
    const int height = grad_out.size(1);
    const int width = grad_out.size(2);

    auto grad_weights = torch::zeros_like(mlp_weights);
    int grad_size = 4 * c_hidden + 1;
    size_t shared_mem_size = 2 * grad_size * sizeof(float);

    // Templated kernel launcher
    auto launch_kernel = [&](auto... Dims) {
        pos_mlp_bias_backward_kernel<decltype(Dims)::value...><<<B, THREADS_PER_BLOCK, shared_mem_size>>>(
            mlp_weights.data_ptr<float>(),
            pos.data_ptr<float>(),
            grad_out.data_ptr<float>(),
            B,
            grad_weights.data_ptr<float>()
        );
    };

    // Define the supported dimensions in a clear and centralized way.
    const auto supported_dims = std::make_tuple(
        std::make_tuple(
            std::integral_constant<int, 8>{}, 
            std::integral_constant<int, 16>{}, 
            std::integral_constant<int, 32>{}, 
            std::integral_constant<int, 64>{},
            std::integral_constant<int, 128>{}
        ), // height
        std::make_tuple(
            std::integral_constant<int, 8>{}, 
            std::integral_constant<int, 16>{}, 
            std::integral_constant<int, 32>{}, 
            std::integral_constant<int, 64>{},
            std::integral_constant<int, 128>{}
        ), // width
        std::make_tuple(std::integral_constant<int, 16>{})  // c_hidden
    );

    // The runtime values that need to be dispatched.
    const auto runtime_dims = std::make_tuple(height, width, c_hidden);

    // Call the generalized dispatcher.
    dispatch_kernel(launch_kernel, runtime_dims, supported_dims);

    CHECK_CUDA_ERROR(cudaPeekAtLastError());

    return grad_weights;
}