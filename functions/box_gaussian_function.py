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


class BoxGaussianCUDAFunction(Function):
    @staticmethod
    def forward(ctx, boxes, offset, sigma, H, W):
        boxes = boxes.contiguous().float()
        offset = offset.contiguous().float()
        sigma = sigma.contiguous().float()
        ctx.save_for_backward(boxes, offset, sigma)
        output = pos_mlp_bias.forward_gaussian(boxes, offset, sigma, H, W)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        boxes, offset, sigma = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_offset, grad_sigma = pos_mlp_bias.backward_gaussian(grad_output, boxes, offset, sigma)
        return None, grad_offset, grad_sigma, None, None


def box_gaussian_python(
    boxes: torch.Tensor,
    offset: torch.Tensor,
    sigma: torch.Tensor,
    H: int,
    W: int,
) -> torch.Tensor:
    """
    Generates a batch of 2D Gaussian masks from bounding boxes.
    The standard deviation (sigma) of the Gaussian is relative to the size of each
    bounding box. A sigma of 1.0 means the standard deviation will be half the
    width and height of the box. The computation is fully differentiable with
    respect to both `boxes` and `sigma`.
    
    Args:
        boxes (torch.Tensor): A tensor of shape (B, 4) containing the normalized
            bounding box coordinates [x, y, w, h] for each mask.
            All values are expected to be in the range [0, 1].
        offset (torch.Tensor): A tensor of shape (B, Nheads, 2) containing the relative
            offsets [offset_x, offset_y] for each mask.
        sigma (torch.Tensor): A tensor of shape (B, Nheads, 2) containing the relative
            standard deviations [sigma_x, sigma_y] for each mask.
        H (int): The height of the output mask.
        W (int): The width of the output mask.
    
    Returns:
        torch.Tensor: A tensor of shape (B, NHeads, H, W) containing the generated
            Gaussian masks.
    """
    # Get batch size, number of heads, and device from input tensors
    B = boxes.shape[0]
    Nheads = offset.shape[1]
    device = boxes.device
    dtype = boxes.dtype
    
    # Extract box centers and sizes
    centers = boxes[:, :2]  # Shape: (B, 2)
    sizes = boxes[:, 2:]    # Shape: (B, 2)
    half_sizes = sizes / 2.0
    
    # Generate a normalized coordinate grid (from 0 to 1)
    x, y = torch.meshgrid(
        torch.linspace(0, 1, W, device=device, dtype=dtype),
        torch.linspace(0, 1, H, device=device, dtype=dtype),
        indexing='xy',
    )
    grid = torch.stack([x, y], dim=-1)  # Shape: (H, W, 2)
    
    # Reshape tensors for broadcasting
    # grid: (H, W, 2) -> (1, 1, H, W, 2)
    # centers: (B, 2) -> (B, 1, 1, 1, 2)
    # half_sizes: (B, 2) -> (B, 1, 1, 1, 2)
    # offset: (B, Nheads, 2) -> (B, Nheads, 1, 1, 2)
    # sigma: (B, Nheads, 2) -> (B, Nheads, 1, 1, 2)
    grid = grid.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W, 2)
    centers = centers.unsqueeze(1).unsqueeze(2).unsqueeze(3)  # Shape: (B, 1, 1, 1, 2)
    half_sizes = half_sizes.unsqueeze(1).unsqueeze(2).unsqueeze(3)  # Shape: (B, 1, 1, 1, 2)
    offset = offset.unsqueeze(2).unsqueeze(3)  # Shape: (B, Nheads, 1, 1, 2)
    sigma = sigma.unsqueeze(2).unsqueeze(3)    # Shape: (B, Nheads, 1, 1, 2)
    
    # Calculate the effective offset and standard deviation, scaled by box size.
    # A sigma of 1 corresponds to a standard deviation of half the box size.
    effective_offset = offset * half_sizes
    effective_sigma = sigma * half_sizes
    
    # Add a small epsilon to avoid division by zero, ensuring stability and
    # differentiability.
    epsilon = 1e-6
    effective_sigma = effective_sigma.clamp(min=epsilon)
    
    # Calculate the center position with offset
    effective_center = centers + effective_offset
    
    # Calculate the exponent of the Gaussian function:
    # -0.5 * [ ((x - mu_x) / sigma_eff_x)^2 + ((y - mu_y) / sigma_eff_y)^2 ]
    delta = grid - effective_center
    exponent = -0.5 * torch.sum((delta / effective_sigma) ** 2, dim=-1)
    
    # Apply the exponential function to compute the final Gaussian mask
    mask = torch.exp(exponent)
    
    return mask