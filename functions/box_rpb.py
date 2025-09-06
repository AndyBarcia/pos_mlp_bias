import torch
import torch.nn.functional as F

from typing import Union, Tuple, Optional
from einops import pack

from .box_rpb_function import BoxRPBCUDAFunction, box_rbp_python
from .box_brpb_function import BoxBRPBCUDAFunction, box_brbp_python


class PosMLP(torch.nn.Module):
    """
    nn.Module wrapper for the fused_attention operation.
    """
    def __init__(
        self, 
        dim,
        hidden_dim=16,
        batched: bool = True,
        implementation: str = "cuda"
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        assert implementation in ["python", "cuda"]
        self.implementation = implementation
        self.batched = batched

        if self.batched:
            self.weight_generator = torch.nn.Linear(dim, 4 * hidden_dim + 1)
        else:
            self.weights = torch.nn.Parameter(torch.randn(4 * hidden_dim + 1))

    def forward(
        self,
        pos: torch.Tensor, # (...,[x,y,w,h])
        size: Union[int, Tuple[int, int]], # (H,W) or int for square
        queries: Optional[torch.Tensor] = None, # (...,C)
        implementation: str = "cuda"
    ) -> torch.Tensor: # (...,H,W)
        queries, queries_ps = pack([queries], "* c")
        pos, _ = pack([pos], "* xywh")

        implementation_to_use = self.implementation if implementation is None else implementation
        size = (size, size) if isinstance(size, int) else size

        if self.batched:
            assert queries is not None, "Queries must be provided for batched mode."
            weights = self.weight_generator(queries) # (..., 4*hidden_dim + 1)
            if implementation_to_use == "cuda":
                output = BoxBRPBCUDAFunction.apply(weights, pos, self.hidden_dim, size[0], size[1])
            elif implementation_to_use == "python":
                output = box_brbp_python(weights, pos, self.hidden_dim, size[0], size[1])
        else:
            weights = self.weights # (4*hidden_dim + 1)
            if implementation_to_use == "cuda":
                output = BoxRPBCUDAFunction.apply(weights, pos, self.hidden_dim, size[0], size[1])
            elif implementation_to_use == "python":
                output = box_rbp_python(weights, pos, self.hidden_dim, size[0], size[1])
        
        final_shape = queries_ps[0] + (size[0], size[1])
        return output.view(final_shape)


class PosMLPAttention(torch.nn.Module):
    """
    nn.Module wrapper for the fused_attention operation.
    """
    def __init__(
        self, 
        dim,
        k_dim=None,
        hidden_dim=16,
        n_heads=8,
        batched_rpb: bool = True,
        implementation: str = "cuda"
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        assert dim % n_heads == 0, f"dim ({dim}) must be divisible by n_heads ({n_heads})"

        self.pos_mlp = PosMLP(dim, hidden_dim, batched_rpb, implementation)

        # Linear projections for Q, K, V
        self.q_proj = torch.nn.Linear(dim, dim, bias=False)
        self.kv_proj = torch.nn.Linear(k_dim if k_dim is not None else k_dim, dim*2, bias=False)
        self.out_proj = torch.nn.Linear(dim, dim, bias=False)
        
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        queries: torch.Tensor, # (..., C)
        memory: torch.Tensor, # (..., H, W, C)
        pos: torch.Tensor, # (..., [x,y,w,h])
    ) -> torch.Tensor: # (..., C)
        queries, queries_ps = pack([queries], "* c")
        memory, memory_ps = pack([memory], "* h w c")
        pos, _ = pack([pos], "* xywh")
        
        batch_size = queries.shape[0]
        H, W = memory.shape[1], memory.shape[2]
        
        # Flatten spatial dimensions of memory for attention computation
        memory_flat = memory.view(batch_size, H * W, -1)  # (B, H*W, C)
        
        # Compute Q, K, V
        Q = self.q_proj(queries)  # (B, C)
        KV = self.kv_proj(memory_flat)  # (B, H*W, C*2)
        K, V = KV.chunk(2, dim=-1)  # Each of shape (B, H*W, C)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, 1, self.n_heads, self.head_dim)  # (B, 1, n_heads, head_dim)
        K = K.view(batch_size, H * W, self.n_heads, self.head_dim)  # (B, H*W, n_heads, head_dim)
        V = V.view(batch_size, H * W, self.n_heads, self.head_dim)  # (B, H*W, n_heads, head_dim)
        
        # Transpose for attention computation: (B, n_heads, seq_len, head_dim)
        Q = Q.transpose(1, 2)  # (B, n_heads, 1, head_dim)
        K = K.transpose(1, 2)  # (B, n_heads, H*W, head_dim)
        V = V.transpose(1, 2)  # (B, n_heads, H*W, head_dim)
        
        # Compute attention scores
        attention = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, n_heads, 1, H*W)
        
        # Reshape attention to match positional bias shape
        attention = attention.view(batch_size, self.n_heads, H, W)  # (B, n_heads, H, W)
        
        # Add positional bias
        pos_bias = self.pos_mlp(pos, (H, W), queries)  # (B, H, W)
        pos_bias = pos_bias.unsqueeze(1).expand(-1, self.n_heads, -1, -1)  # (B, n_heads, H, W)
        attention = attention + pos_bias
        
        # Flatten back for softmax
        attention = attention.view(batch_size, self.n_heads, 1, H * W)  # (B, n_heads, 1, H*W)
        
        # Apply softmax
        attention_weights = F.softmax(attention, dim=-1)  # (B, n_heads, 1, H*W)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)  # (B, n_heads, 1, head_dim)
        
        # Reshape and concatenate heads
        output = output.transpose(1, 2)  # (B, 1, n_heads, head_dim)
        output = output.contiguous().view(batch_size, self.dim)  # (B, C)
        
        # Final linear projection
        output = self.out_proj(output)  # (B, C)
        
        final_shape = queries_ps[0] + (-1,)
        return output.view(final_shape)