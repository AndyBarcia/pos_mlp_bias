#include <torch/extension.h>

torch::Tensor fused_box_brpb_forward(
    const torch::Tensor& mlp_weights, // (B, [2*C' + C' + 1*C' + 1])
    const torch::Tensor& pos,    // (B,[x,y,w,h])
    const int c_hidden,
    const int H,
    const int W
);

torch::Tensor fused_box_brpb_backward(
    const torch::Tensor& grad_out, // (B, H, W)
    const torch::Tensor& mlp_weights, // (B, [2*C' + C' + 1*C' + 1])
    const torch::Tensor& pos,    // (B,[x,y,w,h])
    const int c_hidden
);

torch::Tensor fused_box_rpb_forward(
    const torch::Tensor& mlp_weights, // ([2*C' + C' + 1*C' + 1])
    const torch::Tensor& pos,    // (B,[x,y,w,h])
    const int c_hidden,
    const int H,
    const int W
);

torch::Tensor fused_box_rpb_backward(
    const torch::Tensor& grad_out, // (B, H, W)
    const torch::Tensor& mlp_weights, // ([2*C' + C' + 1*C' + 1])
    const torch::Tensor& pos,    // (B,[x,y,w,h])
    const int c_hidden
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_brpb", &fused_box_brpb_forward, "Fused Attention Forward (Multi-Head)");
    m.def("backward_brpb", &fused_box_brpb_backward, "Fused Attention Backward (Multi-Head)");
    m.def("forward_rpb", &fused_box_rpb_forward, "Fused Attention Forward (Multi-Head)");
    m.def("backward_rpb", &fused_box_rpb_backward, "Fused Attention Backward (Multi-Head)");
}