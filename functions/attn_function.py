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


class AttentionCUDAFunction(Function):
    @staticmethod
    def forward(ctx, q, k, v):
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        ctx.save_for_backward(q, k, v)
        output = pos_mlp_bias.forward_attn(q, k ,v)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        q, k ,v = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        #grad_weights = pos_mlp_bias.backward_rpb(grad_output, mlp_weights, pos, ctx.c_hidden)
        grad_q = torch.zeros_like(q)
        grad_k = torch.zeros_like(k)
        grad_v = torch.zeros_like(v)
        return grad_q, grad_k, grad_v


def attn_python(q, k, v) -> torch.Tensor:
    """
    Python reference implementation for the attention mechanism.
    q: (B, Nh, Nq, C)
    k: (B, Nh, Nk, C)
    v: (B, Nh, Nk, Cv)
    output: (B, Nh, Nq, Cv)
    """
    att = (q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))))
    att = F.softmax(att, dim=-1)
    y = att @ v
    return y
