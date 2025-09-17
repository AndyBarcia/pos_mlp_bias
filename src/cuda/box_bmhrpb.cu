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

template <int HEIGHT, int WIDTH, int C_HIDDEN, int N_HEADS>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, 4) pos_mlp_multi_head_bias_backward_kernel(
    const float* __restrict__ mlp_weights,
    const float* __restrict__ pos,
    const float* __restrict__ grad_output,
    int B,
    float* __restrict__ grad_weights
) {

    int b = blockIdx.x;
    int tid = threadIdx.x;

    constexpr int grad_size = 3 * C_HIDDEN + C_HIDDEN * N_HEADS + N_HEADS;

    extern __shared__ float s_mem[];
    // Use shared memory for weights and for gradient reduction
    float* s_weights = s_mem;
    float* s_grad = s_mem + grad_size;

    // Private, register-based array for gradient accumulation is larger.
    float p_grad[grad_size];
    for (int i = 0; i < grad_size; ++i) {
        p_grad[i] = 0.0f;
    }

    // Cooperatively load mlp_weights into shared memory and zero out the shared gradient accumulator
    for (int idx = tid; idx < grad_size; idx += blockDim.x) {
        s_weights[idx] = mlp_weights[b * grad_size + idx];
        s_grad[idx] = 0.0f;
    }
    __syncthreads();

    // Define pointers for easier access to weights in shared memory
    const float* s_w1 = s_weights;
    const float* s_b1 = s_weights + 2 * C_HIDDEN;
    const float* s_w2 = s_weights + 3 * C_HIDDEN;

    const float cx = pos[b * 4 + 0];
    const float cy = pos[b * 4 + 1];
    const float half_w = fmaxf(pos[b * 4 + 2] * 0.5f, 1e-6f);
    const float half_h = fmaxf(pos[b * 4 + 3] * 0.5f, 1e-6f);

    const float* grad_out_b = &grad_output[b * N_HEADS * HEIGHT * WIDTH];

    // Each thread block processes all spatial positions for one batch item
    for (int idx = tid; idx < HEIGHT * WIDTH; idx += blockDim.x) {
        int i = idx / WIDTH;
        int j = idx % WIDTH;

        float rel_x = ((((float)j) / (float)(WIDTH - 1)) - cx) / half_w;
        float rel_y = ((((float)i) / (float)(HEIGHT - 1)) - cy) / half_h;

        float x[C_HIDDEN]; // Input to softmax
        float s[C_HIDDEN]; // Output of softmax

        // First MLP layer
        float max_x = -FLT_MAX;
        for (int k = 0; k < C_HIDDEN; k++) {
            x[k] = rel_x * s_w1[2 * k] + rel_y * s_w1[2 * k + 1] + s_b1[k];
            if (x[k] > max_x) max_x = x[k];
        }

        // Softmax activation
        float sum_exp = 0.0f;
        for (int k = 0; k < C_HIDDEN; k++) {
            s[k] = __expf(x[k] - max_x);
            sum_exp += s[k];
        }
        float inv_sum = 1.0f / sum_exp;
        for (int k = 0; k < C_HIDDEN; k++) {
            s[k] *= inv_sum;
        }

        // Gradient w.r.t. softmax input (dL/dx), must be accumulated over all heads
        float dL_dx[C_HIDDEN];
        for(int k=0; k<C_HIDDEN; ++k) dL_dx[k] = 0.0f;

        // Loop over heads to compute gradients
        for (int h = 0; h < N_HEADS; ++h) {
            float dL_doutput_hij = grad_out_b[h * HEIGHT * WIDTH + idx];

            // Gradient for bias b2
            p_grad[3 * C_HIDDEN + C_HIDDEN * N_HEADS + h] += dL_doutput_hij;

            // Recompute (output_h - b2_h) for the dL/dx calculation
            float output_h_minus_b2_h = 0.0f;
            for (int k = 0; k < C_HIDDEN; k++) {
                output_h_minus_b2_h += s[k] * s_w2[k * N_HEADS + h];
            }

            for (int k = 0; k < C_HIDDEN; k++) {
                // Gradient for weights w2
                p_grad[3 * C_HIDDEN + k * N_HEADS + h] += dL_doutput_hij * s[k];

                // Accumulate gradient w.r.t. softmax input for head h
                dL_dx[k] += dL_doutput_hij * s[k] * (s_w2[k * N_HEADS + h] - output_h_minus_b2_h);
            }
        }

        // Use the final dL/dx (summed over all heads) to calculate gradients for the first layer
        for (int k = 0; k < C_HIDDEN; k++) {
            p_grad[2 * k]               += dL_dx[k] * rel_x; // grad w1_x
            p_grad[2 * k + 1]           += dL_dx[k] * rel_y; // grad w1_y
            p_grad[2 * C_HIDDEN + k]    += dL_dx[k];       // grad b1
        }
    }

    __syncthreads();

    // Reduction of private gradients into shared memory, same logic as before
    for (int i = 0; i < grad_size; ++i) {
        float val = p_grad[i];

        // Perform a warp-level reduction using shuffle instructions.
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
        
        if ((tid % 32) == 0) {
            atomicAdd(&s_grad[i], val);
        }
    }
    
    __syncthreads();

    // Write final reduced gradients from shared memory to global memory
    for (int idx = tid; idx < grad_size; idx += blockDim.x) {
        grad_weights[b * grad_size + idx] = s_grad[idx];
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
    size_t shared_mem_size = 2 * grad_size * sizeof(float);

    // Templated kernel launcher
    auto launch_kernel = [&](auto... Dims) {
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