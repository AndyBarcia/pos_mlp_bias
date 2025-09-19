#include <torch/extension.h>

torch::Tensor attn_forward(
    const torch::Tensor &Q, // (B, Nh, Nq, C)
    const torch::Tensor &K, // (B, Nh, Nk, C)
    const torch::Tensor &V // (B, Nh, Nk, Cv)
);

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

torch::Tensor fused_box_bmhrbp_forward(
    const torch::Tensor& mlp_weights, // (B, [3*C' + C'*Nh + Nh])
    const torch::Tensor& pos,    // (B,[x,y,w,h])
    const int c_hidden,
    const int n_heads,
    const int height,
    const int width
);

torch::Tensor fused_box_bmhrbp_backward(
    const torch::Tensor& grad_out, // (B, Nh, H, W)
    const torch::Tensor& mlp_weights, // (B, [3*C' + C'*Nh + Nh])
    const torch::Tensor& pos,    // (B,[x,y,w,h])
    const int c_hidden,
    const int n_heads
);

torch::Tensor fused_box_gaussian_forward(
    const torch::Tensor& boxes,
    const torch::Tensor& offset,
    const torch::Tensor& sigma,
    const int H,
    const int W
);

std::vector<torch::Tensor> fused_box_gaussian_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& boxes,
    const torch::Tensor& offset,
    const torch::Tensor& sigma
);

torch::Tensor fused_box_pair_rpb_forward(
    const torch::Tensor& mlp_weights, // ([4*C' + C' + 1*C' + 1])
    const torch::Tensor& pos1,    // (B,N1,[x,y,w,h])
    const torch::Tensor& pos2,    // (B,N2,[x,y,w,h])
    const int c_hidden
);

torch::Tensor fused_box_pair_rpb_backward(
    const torch::Tensor& grad_out, // (B,N1,N2)
    const torch::Tensor& mlp_weights, // ([4*C' + C' + 1*C' + 1])
    const torch::Tensor& pos1,    // (B,N1,[x,y,w,h])
    const torch::Tensor& pos2,    // (B,N2,[x,y,w,h])
    const int c_hidden
);

torch::Tensor fused_box_pair_brpb_forward(
    const torch::Tensor& mlp_weights, // (B, N1, [4*C' + C' + 1*C' + 1])
    const torch::Tensor& pos1,        // (B, N1, [x,y,w,h])
    const torch::Tensor& pos2,        // (B, N2, [x,y,w,h])
    const int c_hidden
);

torch::Tensor fused_box_pair_brpb_backward(
    const torch::Tensor& grad_out,    // (B, N1, N2)
    const torch::Tensor& mlp_weights, // (B, N1, [4*C' + C' + 1*C' + 1])
    const torch::Tensor& pos1,        // (B, N1, [x,y,w,h])
    const torch::Tensor& pos2,        // (B, N2, [x,y,w,h])
    const int c_hidden
);

torch::Tensor fused_box_pair_gaussian_forward(
    const torch::Tensor& boxes1,
    const torch::Tensor& offset1,
    const torch::Tensor& sigma1,
    const torch::Tensor& boxes2,
    const torch::Tensor& offset2,
    const torch::Tensor& sigma2
);

std::vector<torch::Tensor> fused_box_pair_gaussian_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& boxes1,
    const torch::Tensor& offset1,
    const torch::Tensor& sigma1,
    const torch::Tensor& boxes2,
    const torch::Tensor& offset2,
    const torch::Tensor& sigma2
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_attn", &attn_forward, "Fused Attention Forward (Multi-Head)");
    m.def("forward_brpb", &fused_box_brpb_forward, "Fused Attention Forward (Multi-Head)");
    m.def("backward_brpb", &fused_box_brpb_backward, "Fused Attention Backward (Multi-Head)");
    m.def("forward_rpb", &fused_box_rpb_forward, "Fused Attention Forward (Multi-Head)");
    m.def("backward_rpb", &fused_box_rpb_backward, "Fused Attention Backward (Multi-Head)");
    m.def("forward_bmhrpb", &fused_box_bmhrbp_forward, "Fused Attention Forward (Multi-Head)");
    m.def("backward_bmhrpb", &fused_box_bmhrbp_backward, "Fused Attention Forward (Multi-Head)");
    m.def("forward_gaussian", &fused_box_gaussian_forward, "Fused Attention Forward (Multi-Head)");
    m.def("backward_gaussian", &fused_box_gaussian_backward, "Fused Attention Backward (Multi-Head)");
    m.def("forward_pair_rpb", &fused_box_pair_rpb_forward, "Fused Attention Forward (Multi-Head)");
    m.def("backward_pair_rpb", &fused_box_pair_rpb_backward, "Fused Attention Backward (Multi-Head)");
    m.def("forward_pair_brpb", &fused_box_pair_brpb_forward, "Fused Attention Forward (Multi-Head)");
    m.def("backward_pair_brpb", &fused_box_pair_brpb_backward, "Fused Attention Backward (Multi-Head)");
    m.def("forward_pair_gaussian", &fused_box_pair_gaussian_forward, "Fused Attention Forward (Multi-Head)");
    m.def("backward_pair_gaussian", &fused_box_pair_gaussian_backward, "Fused Attention Backward (Multi-Head)");
}