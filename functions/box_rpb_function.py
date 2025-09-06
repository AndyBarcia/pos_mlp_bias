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


class BoxRPBCUDAFunction(Function):
    @staticmethod
    def forward(ctx, mlp_weights, pos, c_hidden, H, W):
        ctx.save_for_backward(mlp_weights, pos)
        ctx.c_hidden = c_hidden
        mlp_weights = mlp_weights.contiguous()
        pos = pos.contiguous()
        output = pos_mlp_bias.forward_rpb(mlp_weights, pos, c_hidden, H, W)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        mlp_weights, pos = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_weights = pos_mlp_bias.backward_rpb(grad_output, mlp_weights, pos, ctx.c_hidden)
        return grad_weights, None, None, None, None


def box_rbp_python(mlp_weights, pos, c_hidden, W, H) -> torch.Tensor:
    """
    Python reference implementation for the RBP bias kernel.
    mlp_weights: ([2*C' + C' + 1*C' + 1])
    pos: (B,[x,y,w,h])
    c_hidden: C'
    w: W
    h: H
    output: (B,H,W)
    """
    B = pos.shape[0]
    C2 = mlp_weights.shape[0]
    C = c_hidden
    assert C2 == 4 * C + 1

    # Extract boxes
    cx, cy, w, h = pos[:, 0], pos[:, 1], pos[:, 2], pos[:, 3]
    half_w = w / 2
    half_h = h / 2

    # Avoid 0 division
    epsilon = 1e-6
    half_w = half_w.clamp(min=epsilon)
    half_h = half_h.clamp(min=epsilon)

    # Stack together x and y axis
    centers = torch.stack([cx, cy], dim=1).unsqueeze(1) # Shape: (B, 1, 2)
    half_sizes = torch.stack([half_w, half_h], dim=1).unsqueeze(1) # Shape: (B, 1, 2)

    # Generate relative grid.
    x, y = torch.meshgrid(
        torch.linspace(0, 1, W, dtype=torch.float32, device=mlp_weights.device),
        torch.linspace(0, 1, H, dtype=torch.float32, device=mlp_weights.device),
        indexing='xy',
    )
    # Transformar el grid absoluto a relativo para cada caja
    relative_grid = (torch.stack([x, y], dim=-1).view(-1, 2).unsqueeze(0) - centers) / half_sizes # (B,H*W,[x,y])

    # Obtain weights.
    w1 = mlp_weights[0:2*C].view(-1,C,2).expand(B,-1,-1) # (C',[dx,dy])
    b1 = mlp_weights[2*C : 2*C+C].view(-1,1,C).expand(B,-1,-1) # (1,C') 
    w2 = mlp_weights[2*C+C : 2*C+C+C].view(-1,C,1).expand(B,-1,-1) # (C',1)
    b2 = mlp_weights[2*C+C+C : ].view(-1,1,1).expand(B,-1,-1) # (1,1)

    # Use the generated weights to compute the bias for each position.
    # Use softmax because it generates smoother results than relu.
    x = torch.bmm(relative_grid, w1.transpose(1,2)) + b1
    x = torch.softmax(x, dim=-1)
    x = torch.bmm(x, w2) + b2

    # Return the final bias map
    return x.view(B, H, W)
