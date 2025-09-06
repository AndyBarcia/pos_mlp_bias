#include <ATen/native/cuda/KernelUtils.cuh>
#include <c10/util/Half.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define CHECK_CUDA_ERROR(err) do { cuda_check((err), __FILE__, __LINE__); } while(false)
inline void cuda_check(cudaError_t error_code, const char *file, int line)
{
    if (error_code != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error %d: %s. In file '%s' on line %d\n", error_code, cudaGetErrorString(error_code), file, line);
        fflush(stderr);
        exit(error_code);
    }
}

const int MAX_C = 16;
const int THREADS_PER_BLOCK_FORWARD = 512;
const int THREADS_PER_BLOCK = 256;

__global__ void __launch_bounds__(THREADS_PER_BLOCK_FORWARD, 2) box_rbp_forward_kernel(
    const float* __restrict__ mlp_weights,
    const float* __restrict__ pos,
    int c_hidden,
    int W,
    int H,
    int B,
    float* __restrict__ output
) {
    const int total_elements = B * H * W;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_elements) return;

    const int b = tid / (H * W);
    const int spatial_idx = tid % (H * W);

    const float i = ((float) (spatial_idx / W)) / (float) (H-1);
    const float j = ((float) (spatial_idx % W)) / (float) (W-1);

    const float cx = pos[b * 4 + 0];
    const float cy = pos[b * 4 + 1];
    const float half_w = fmaxf(pos[b * 4 + 2] * 0.5f, 1e-6f);
    const float half_h = fmaxf(pos[b * 4 + 3] * 0.5f, 1e-6f);

    const float rel_x = (j - cx) / half_w;
    const float rel_y = (i - cy) / half_h;

    // Weights are now shared across the batch, so no batch offset is needed.
    const float* w = mlp_weights;

    float temp[MAX_C];
    float max_val = -FLT_MAX;
    for (int k = 0; k < c_hidden; k++) {
        const float w_x = w[2 * k];
        const float w_y = w[2 * k + 1];
        const float b1 = w[2 * c_hidden + k];
        temp[k] = rel_x * w_x + rel_y * w_y + b1;
        if (temp[k] > max_val) max_val = temp[k];
    }

    float sum_exp = 0.0f;
    for (int k = 0; k < c_hidden; k++) {
        temp[k] = __expf(temp[k] - max_val);
        sum_exp += temp[k];
    }

    const float inv_sum_exp = 1.0f / sum_exp;
    for (int k = 0; k < c_hidden; k++) {
        temp[k] *= inv_sum_exp;
    }

    float out_val = 0.0f;
    for (int k = 0; k < c_hidden; k++) {
        out_val += temp[k] * w[3 * c_hidden + k];
    }
    out_val += w[4 * c_hidden];

    output[tid] = out_val;
}

torch::Tensor fused_box_rpb_forward(
    const torch::Tensor& mlp_weights, // ([2*C' + C' + 1*C' + 1])
    const torch::Tensor& pos,    // (B,[x,y,w,h])
    const int c_hidden,
    const int H,
    const int W
) {
    CHECK_INPUT(mlp_weights);
    CHECK_INPUT(pos);

    // B is now derived from the 'pos' tensor.
    const int B = pos.size(0);

    auto output = torch::empty({B, H, W}, mlp_weights.options());
    const int total_elements = B * H * W;
    const int blocks = (total_elements + THREADS_PER_BLOCK_FORWARD - 1) / THREADS_PER_BLOCK_FORWARD;

    box_rbp_forward_kernel<<<blocks, THREADS_PER_BLOCK_FORWARD>>>(
        mlp_weights.data_ptr<float>(),
        pos.data_ptr<float>(),
        c_hidden,
        W,
        H,
        B,
        output.data_ptr<float>()
    );

    CHECK_CUDA_ERROR(cudaPeekAtLastError());

    return output;
}

__global__ void __launch_bounds__(THREADS_PER_BLOCK, 4) box_rbp_backward_kernel(
    const float* __restrict__ mlp_weights,
    const float* __restrict__ pos,
    const float* __restrict__ grad_output,
    int c_hidden,
    int W,
    int H,
    int B,
    float* __restrict__ grad_weights
) {

    int b = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ float s_mem[];
    int grad_size = 4 * c_hidden + 1;

    // s_mem will be used for both weights and gradients
    float* s_weights = s_mem;
    float* s_grad = s_mem;

    // Use a private, register-based array for gradient accumulation.
    float p_grad[4 * MAX_C + 1];
    for (int i = 0; i < grad_size; ++i) {
        p_grad[i] = 0.0f;
    }

    // Cooperatively load the shared mlp_weights into shared memory.
    if (tid < grad_size) {
        s_weights[tid] = mlp_weights[tid];
    }
    __syncthreads();

    const float cx = pos[b * 4 + 0];
    const float cy = pos[b * 4 + 1];
    const float half_w = fmaxf(pos[b * 4 + 2] * 0.5f, 1e-6f);
    const float half_h = fmaxf(pos[b * 4 + 3] * 0.5f, 1e-6f);

    const float* w = s_weights;
    const float* grad_out_b = &grad_output[b * H * W];

    for (int idx = tid; idx < H * W; idx += blockDim.x) {
        int i = idx / W;
        int j = idx % W;

        float rel_x = ((((float)j) / (float)(W - 1)) - cx) / half_w;
        float rel_y = ((((float)i) / (float)(H - 1)) - cy) / half_h;

        float x[MAX_C];
        float s[MAX_C];

        float max_x = -FLT_MAX;
        for (int k = 0; k < c_hidden; k++) {
            x[k] = rel_x * w[2 * k] + rel_y * w[2 * k + 1] + w[2 * c_hidden + k];
            if (x[k] > max_x) max_x = x[k];
        }

        float sum_exp = 0.0f;
        for (int k = 0; k < c_hidden; k++) {
            s[k] = __expf(x[k] - max_x);
            sum_exp += s[k];
        }

        float inv_sum = 1.0f / sum_exp;
        for (int k = 0; k < c_hidden; k++) {
            s[k] *= inv_sum;
        }

        float dL_doutput_bij = grad_out_b[i * W + j];

        float output_minus_b2 = 0.0f;
        for (int k = 0; k < c_hidden; k++) {
            output_minus_b2 += s[k] * w[3 * c_hidden + k];
        }

        p_grad[4 * c_hidden] += dL_doutput_bij;
        for (int k = 0; k < c_hidden; k++) {
            p_grad[3 * c_hidden + k] += dL_doutput_bij * s[k];
            float dL_dx_k = s[k] * (dL_doutput_bij * w[3 * c_hidden + k] - dL_doutput_bij * output_minus_b2);
            p_grad[2 * k] += dL_dx_k * rel_x;
            p_grad[2 * k + 1] += dL_dx_k * rel_y;
            p_grad[2 * c_hidden + k] += dL_dx_k;
        }
    }

    __syncthreads();

    // Re-Initialize shared memory for reduction.
    if (tid < grad_size) {
        s_grad[tid] = 0.0f;
    }

    __syncthreads();

    // Reduce private gradients (p_grad) into shared memory (s_grad).
    for (int i = 0; i < grad_size; ++i) {
        float val = p_grad[i];
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
        if ((tid % 32) == 0) {
            atomicAdd(&s_grad[i], val);
        }
    }
    
    __syncthreads();

    // Atomically add the reduced gradients from this block to the global output tensor.
    if (tid < grad_size) {
        atomicAdd(&grad_weights[tid], s_grad[tid]);
    }
}

torch::Tensor fused_box_rpb_backward(
    const torch::Tensor& grad_out, // (B, H, W)
    const torch::Tensor& mlp_weights, // ([2*C' + C' + 1*C' + 1])
    const torch::Tensor& pos,    // (B,[x,y,w,h])
    const int c_hidden
) {
    CHECK_INPUT(grad_out);
    CHECK_INPUT(mlp_weights);
    CHECK_INPUT(pos);

    // B is now derived from the 'pos' or 'grad_out' tensor.
    const int B = pos.size(0);
    const int H = grad_out.size(1);
    const int W = grad_out.size(2);

    // zeros_like correctly creates a tensor with the same shape as mlp_weights (no batch dim).
    auto grad_weights = torch::zeros_like(mlp_weights);
    int grad_size = 4 * c_hidden + 1;
    size_t shared_mem_size = grad_size * sizeof(float);

    box_rbp_backward_kernel<<<B, THREADS_PER_BLOCK, shared_mem_size>>>(
        mlp_weights.data_ptr<float>(),
        pos.data_ptr<float>(),
        grad_out.data_ptr<float>(),
        c_hidden,
        W,
        H,
        B,
        grad_weights.data_ptr<float>()
    );

    CHECK_CUDA_ERROR(cudaPeekAtLastError());

    return grad_weights;
}