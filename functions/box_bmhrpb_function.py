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


class BoxBMHRPBCUDAFunction(Function):
    @staticmethod
    def forward(ctx, mlp_weights, pos, c_hidden, n_heads, H, W):
        mlp_weights = mlp_weights.contiguous().float()
        pos = pos.contiguous().float()
        ctx.save_for_backward(mlp_weights, pos)
        ctx.c_hidden = c_hidden
        ctx.n_heads = n_heads
        output = pos_mlp_bias.forward_bmhrpb(mlp_weights, pos, c_hidden, n_heads, H, W)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        mlp_weights, pos = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_weights = pos_mlp_bias.backward_bmhrpb(grad_output, mlp_weights, pos, ctx.c_hidden, ctx.n_heads)
        return grad_weights, None, None, None, None, None


def box_bmhrbp_python(mlp_weights, pos, c_hidden, n_heads, W, H) -> torch.Tensor:
    """
    Python reference implementation for the MLP position bias kernel.
    mlp_weights: (B, [2*C' + C' + C'*Nh + Nh])
    pos: (B,[x,y,w,h])
    c_hidden: C'
    n_heads: Nh
    w: W
    h: H
    output: (B,Nh,H,W)
    """
    B, C2 = mlp_weights.shape
    C = c_hidden
    Nh = n_heads
    assert C2 == 3*C + C*Nh + Nh

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
    w1 = mlp_weights[:, 0:2*C].view(-1,C,2) # (B,C',[dx,dy])
    b1 = mlp_weights[:, 2*C : 2*C+C].view(-1,1,C) # (B,1,C') 
    w2 = mlp_weights[:, 2*C+C : 2*C+C+C*Nh].view(-1,C,Nh) # (B,C',Nh)
    b2 = mlp_weights[:, 2*C+C+C*Nh : ].view(-1,1,Nh) # (B,1,Nh)

    # Use the generated weights to compute the bias for each position.
    # Use softmax because it generates smoother results than relu.
    x = torch.bmm(relative_grid, w1.transpose(1,2)) + b1
    x = torch.softmax(x, dim=-1)
    x = torch.bmm(x, w2) + b2

    # Move head dimension.
    x = x.view(B, H, W, Nh).flatten(1,2) # (B,H*W,Nh)
    x = x.transpose(1,2) # (B,Nh,H*W)

    # Return the final bias map
    return x.view(B, Nh, H, W)
