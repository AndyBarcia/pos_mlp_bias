import math
import torch
import torch.nn.functional as F
from torch.autograd import Function
from einops import rearrange, einsum
from torch.autograd import Function
from torch.utils.checkpoint import checkpoint

try:
    import pos_mlp_bias
except ImportError:
    print("CUDA extension pos_mlp_bias not found. Please compile it first.")
    print("Run: pip3 install --no-build-isolation .")


class BoxPairBRPBCUDAFunction(Function):
    @staticmethod
    def forward(ctx, mlp_weights, pos1, pos2, c_hidden):
        mlp_weights = mlp_weights.contiguous()
        pos1 = pos1.contiguous()
        pos2 = pos2.contiguous()
        ctx.save_for_backward(mlp_weights, pos1, pos2)
        ctx.c_hidden = c_hidden
        output = pos_mlp_bias.forward_pair_brpb(mlp_weights, pos1, pos2, c_hidden)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        mlp_weights, pos1, pos2 = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_weights = pos_mlp_bias.backward_pair_brpb(grad_output, mlp_weights, pos1, pos2, ctx.c_hidden)
        return grad_weights, None, None, None


def box_pair_brbp_python(mlp_weights, boxes1, boxes2, c_hidden) -> torch.Tensor:
    """
    Python reference implementation for the box-pair RBP bias kernel.
    mlp_weights: (B,N1,[4*C' + C' + 1*C' + 1])
    boxes1: (B,N1,4) tensor of boxes [cx, cy, w, h]
    boxes2: (B,N2,4) tensor of boxes [cx, cy, w, h]
    c_hidden: C, number of hidden units
    output: (B,N1,N2) tensor of biases for each box pair
    """
    B,N1 = boxes1.shape[0], boxes1.shape[1]
    _,N2 = boxes2.shape[0], boxes2.shape[1]
    C = c_hidden
    assert mlp_weights.shape[-1] == 6 * C + 1
    
    # Extract box parameters
    cx1, cy1, w1, h1 = boxes1[:, :, 0], boxes1[:, :, 1], boxes1[:, :, 2], boxes1[:, :, 3]
    cx2, cy2, w2, h2 = boxes2[:, :, 0], boxes2[:, :, 1], boxes2[:, :, 2], boxes2[:, :, 3]
    
    # Compute relative features
    epsilon = 1e-6
    # Compute relative differences of centers relative to the sizes of the first box
    dx = (cx2.unsqueeze(1) - cx1.unsqueeze(2)) / (w1.unsqueeze(2) + epsilon)
    dy = (cy2.unsqueeze(1) - cy1.unsqueeze(2)) / (h1.unsqueeze(2) + epsilon)
    # Compute log-scale differences of widths and heights
    dw = torch.log(w2.unsqueeze(1) / (w1.unsqueeze(2) + epsilon))
    dh = torch.log(h2.unsqueeze(1) / (h1.unsqueeze(2) + epsilon))
    
    # Combine features into a (B1, B2, 4) tensor
    rel_features = torch.stack([dx, dy, dw, dh], dim=-1)
    
    # Reshape MLP weights
    w1 = mlp_weights[:,:,:4 * C].view(B,N1,4,C)  # (B,N1,4, C)
    b1 = mlp_weights[:,:,4 * C:5 * C].view(B,N1,1,C).expand(-1,-1,N2,-1) # (B,N1,C,)
    w2 = mlp_weights[:,:,5 * C:6 * C].view(B,N1,C, 1) # (B,N1,C, 1)
    b2 = mlp_weights[:,:,6 * C].view(B,N1,1,1).expand(-1,-1,N2,-1) # (B,N1)
    
    # Compute MLP output
    x = torch.einsum('bijk,bikl->bijl', rel_features, w1) + b1  # (B1, B2, C)
    x = torch.softmax(x, dim=-1)
    x = torch.einsum('bijk,bikl->bijl', x, w2) + b2  # (B1, B2, 1)
    
    return x.squeeze(-1)