import torch
import torch.nn.functional as F

from typing import Union, Tuple, Optional
from einops import pack, unpack

from .box_rpb_function import BoxRPBCUDAFunction, box_rbp_python
from .box_brpb_function import BoxBRPBCUDAFunction, box_brbp_python

from .box_pair_rpb_function import BoxPairRPBCUDAFunction, box_pair_rbp_python
from .box_pair_brpb_function import BoxPairBRPBCUDAFunction, box_pair_brbp_python


class PosMLP(torch.nn.Module):
    """
    Generates a positional bias map from a bounding box coordinate.

    The bias is calculated relative to a 2D grid of a given size.
    This module can operate in two modes:
    - Batched: Dynamically generates MLP weights based on input query features.
    - Non-batched: Uses a single set of learned MLP weights.
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
        """
        Forward pass for PosMLP.

        Args:
            pos: Bounding box coordinates.
            size: The height and width of the grid.
            queries: Optional query features for batched mode.
            implementation: Override the default implementation ('cuda' or 'python').

        Returns:
            A tensor representing the positional bias map.
        """
        pos, pos_ps = pack([pos], "* xywh")

        implementation_to_use = self.implementation if implementation is None else implementation
        size = (size, size) if isinstance(size, int) else size

        if self.batched:
            assert queries is not None, "Queries must be provided for batched mode."
            queries, _ = pack([queries], "* c")
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
        
        final_shape = pos_ps[0] + (size[0], size[1])
        return output.view(final_shape)


class PairPosMLP(torch.nn.Module):
    """
    Generates a pairwise positional bias map between two sets of bounding boxes.
    
    This module can operate in two modes:
    - Batched: Dynamically generates MLP weights based on input query features.
    - Non-batched: Uses a single set of learned MLP weights.
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
            self.weight_generator = torch.nn.Linear(dim, 6 * hidden_dim + 1)
        else:
            self.weights = torch.nn.Parameter(torch.randn(6 * hidden_dim + 1))

    def forward(
        self,
        pos1: torch.Tensor, # (...,N1,[x,y,w,h])
        pos2: torch.Tensor, # (...,N2,[x,y,w,h])
        queries: Optional[torch.Tensor] = None, # (...,N1,C)
        implementation: str = "cuda"
    ) -> torch.Tensor: # (...,N1,N2)
        """
        Forward pass for PairPosMLP.

        Args:
            pos1: First set of bounding box coordinates.
            pos2: Second set of bounding box coordinates.
            queries: Optional query features for batched mode.
            implementation: Override the default implementation ('cuda' or 'python').

        Returns:
            A tensor representing the pairwise positional bias map.
        """
        N1,N2 = pos1.shape[-2], pos2.shape[-2]
        pos1, pos_ps = pack([pos1], "* n1 xywh")
        pos2, _ = pack([pos2], "* n2 xywh")

        implementation_to_use = self.implementation if implementation is None else implementation

        if self.batched:
            assert queries is not None, "Queries must be provided for batched mode."
            queries, _ = pack([queries], "* n1 c")
            weights = self.weight_generator(queries) # (..., N1, 6*hidden_dim + 1)
            if implementation_to_use == "cuda":
                output = BoxPairBRPBCUDAFunction.apply(weights, pos1, pos2, self.hidden_dim)
            elif implementation_to_use == "python":
                output = box_pair_brbp_python(weights, pos1, pos2, self.hidden_dim)
        else:
            weights = self.weights # (6*hidden_dim + 1)
            if implementation_to_use == "cuda":
                output = BoxPairRPBCUDAFunction.apply(weights, pos1, pos2, self.hidden_dim)
            elif implementation_to_use == "python":
                output = box_pair_rbp_python(weights, pos1, pos2, self.hidden_dim)
        
        final_shape = pos_ps[0] + (N1, N2)
        return output.view(final_shape)


class PosMLPAttention(torch.nn.Module):
    """
    Cross-attention module with a positional bias generated by PosMLP.
    A query attends to a 2D feature map (memory).
    """
    def __init__(
        self,
        dim: int,
        k_dim: Optional[int] = None,
        hidden_dim: int = 16,
        n_heads: int = 8,
        batched_rpb: bool = True,
        implementation: str = "cuda"
    ):
        super().__init__()
        self.dim = dim
        self.k_dim = k_dim if k_dim is not None else dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        assert dim % n_heads == 0, f"dim ({dim}) must be divisible by n_heads ({n_heads})"

        self.pos_mlp = PosMLP(dim, hidden_dim, batched_rpb, implementation)

        # Linear projections for Q, K, V
        self.q_proj = torch.nn.Linear(dim, dim, bias=False)
        self.kv_proj = torch.nn.Linear(self.k_dim, dim * 2, bias=False)
        self.out_proj = torch.nn.Linear(dim, dim, bias=False)

        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        queries: torch.Tensor, # (..., C)
        memory: torch.Tensor, # (..., H, W, k_dim)
        pos: torch.Tensor, # (..., [x, y, w, h])
    ) -> torch.Tensor: # (..., C)
        """
        Forward pass for PosMLPAttention.

        Args:
            queries: The query tensor.
            memory: The key/value source tensor (a 2D feature map).
            pos: The positional information corresponding to the queries.

        Returns:
            The output tensor after attention.
        """
        assert memory.shape[-1] == self.k_dim, \
            f"Memory feature dimension ({memory.shape[-1]}) must match k_dim ({self.k_dim})"

        queries, queries_ps = pack([queries], "* c")
        memory, _ = pack([memory], "* h w c")
        pos, _ = pack([pos], "* xywh")

        batch_size = queries.shape[0]
        H, W = memory.shape[1], memory.shape[2]

        # Flatten spatial dimensions of memory
        memory_flat = memory.view(batch_size, H * W, -1)  # (B, H*W, k_dim)

        # Compute Q, K, V
        Q = self.q_proj(queries)  # (B, dim)
        K, V = self.kv_proj(memory_flat).chunk(2, dim=-1)  # Each: (B, H*W, dim)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, 1, self.n_heads, self.head_dim)
        K = K.view(batch_size, H * W, self.n_heads, self.head_dim)
        V = V.view(batch_size, H * W, self.n_heads, self.head_dim)

        # Transpose for attention computation
        Q = Q.transpose(1, 2)  # (B, n_heads, 1, head_dim)
        K = K.transpose(1, 2)  # (B, n_heads, H*W, head_dim)
        V = V.transpose(1, 2)  # (B, n_heads, H*W, head_dim)

        # Compute attention scores
        attention = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, n_heads, 1, H*W)

        # Reshape attention to add positional bias
        attention = attention.view(batch_size, self.n_heads, H, W)

        # Add positional bias
        pos_bias = self.pos_mlp(pos, (H, W), queries)  # (B, H, W)
        attention = attention + pos_bias.unsqueeze(1) # (B, n_heads, H, W)

        # Flatten back for softmax
        attention = attention.view(batch_size, self.n_heads, 1, H * W)

        # Apply softmax
        attention_weights = F.softmax(attention, dim=-1)

        # Apply attention to values
        output = torch.matmul(attention_weights, V)  # (B, n_heads, 1, head_dim)

        # Reshape and concatenate heads
        output = output.transpose(1, 2).contiguous().view(batch_size, self.dim)

        # Final linear projection
        output = self.out_proj(output)

        # Unpack to original shape
        output, = unpack(output, queries_ps, "* c")
        return output


class PosMLPSelfAttention(torch.nn.Module):
    """
    Self-attention module with a pairwise positional bias generated by PairPosMLP.
    """
    def __init__(
        self,
        dim: int,
        hidden_dim: int = 16,
        n_heads: int = 8,
        batched_rpb: bool = True,
        implementation: str = "cuda"
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        assert dim % n_heads == 0, f"dim ({dim}) must be divisible by n_heads ({n_heads})"

        self.pos_mlp = PairPosMLP(dim, hidden_dim, batched_rpb, implementation)

        # Linear projections for Q, K, V
        self.qkv_proj = torch.nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = torch.nn.Linear(dim, dim, bias=False)

        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        x: torch.Tensor, # (..., Q, C)
        pos: torch.Tensor, # (..., Q, [x,y,w,h])
    ) -> torch.Tensor: # (..., Q, C)
        """
        Forward pass for PosMLPSelfAttention.

        Args:
            x: The input sequence tensor.
            pos: The positional information for each element in the sequence.

        Returns:
            The output tensor after self-attention.
        """
        x, x_ps = pack([x], "* q c")
        pos, _ = pack([pos], "* q xywh")

        batch_size, q_len, _ = x.shape

        # Compute Q, K, V from a single projection for efficiency
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)

        # Reshape for multi-head attention
        q = q.view(batch_size, q_len, self.n_heads, self.head_dim)
        k = k.view(batch_size, q_len, self.n_heads, self.head_dim)
        v = v.view(batch_size, q_len, self.n_heads, self.head_dim)

        # Transpose for attention computation
        q = q.transpose(1, 2) # (B, n_heads, Q, head_dim)
        k = k.transpose(1, 2) # (B, n_heads, Q, head_dim)
        v = v.transpose(1, 2) # (B, n_heads, Q, head_dim)

        # Compute attention scores
        attention = torch.matmul(q, k.transpose(-2, -1)) * self.scale # (B, n_heads, Q, Q)

        # Add pairwise positional bias
        # The bias is generated based on the original features `x` for batched mode
        pos_bias = self.pos_mlp(pos, pos, x) # (B, Q, Q)
        attention = attention + pos_bias.unsqueeze(1) # Broadcast bias across heads

        # Apply softmax
        attention_weights = F.softmax(attention, dim=-1)

        # Apply attention to values
        output = torch.matmul(attention_weights, v)

        # Reshape and concatenate heads
        output = output.transpose(1, 2).contiguous().view(batch_size, q_len, self.dim)

        # Final linear projection
        output = self.out_proj(output)

        # Unpack to original shape
        output, = unpack(output, x_ps, "* q c")
        return output