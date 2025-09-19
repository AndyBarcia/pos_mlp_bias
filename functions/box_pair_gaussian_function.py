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


class BoxPairGaussianCUDAFunction(Function):
    @staticmethod
    def forward(ctx, boxes1, offset1, sigma1, boxes2, offset2, sigma2):
        boxes1 = boxes1.contiguous().float()
        offset1 = offset1.contiguous().float()
        sigma1 = sigma1.contiguous().float()
        boxes2 = boxes2.contiguous().float()
        offset2 = offset2.contiguous().float()
        sigma2 = sigma2.contiguous().float()
        ctx.save_for_backward(boxes1, offset1, sigma1, boxes2, offset2, sigma2)
        output = pos_mlp_bias.forward_pair_gaussian(boxes1, offset1, sigma1, boxes2, offset2, sigma2)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        boxes1, offset1, sigma1, boxes2, offset2, sigma2 = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_offset1, grad_sigma1, grad_offset2, grad_sigma2 = pos_mlp_bias.backward_pair_gaussian(
            grad_output, 
            boxes1, offset1, sigma1, 
            boxes2, offset2, sigma2
        )
        return None, grad_offset1, grad_sigma1, None, grad_offset2, grad_sigma2


def box_pair_gaussian_python(
    boxes1: torch.Tensor,
    offset1: torch.Tensor,
    sigma1: torch.Tensor,
    boxes2: torch.Tensor,
    offset2: torch.Tensor,
    sigma2: torch.Tensor,
) -> torch.Tensor:
    """
    Computes pairwise Bhattacharyya coefficients between two sets of 2D Gaussians
    derived from bounding boxes with offsets and sigmas.
    
    Args:
        boxes1 (torch.Tensor): A tensor of shape (B, 4) containing the normalized
            bounding box coordinates [x, y, w, h] for the first set.
        offset1 (torch.Tensor): A tensor of shape (B, N, 2) containing the relative
            offsets [offset_x, offset_y] for the first set.
        sigma1 (torch.Tensor): A tensor of shape (B, N, 2) containing the relative
            standard deviations [sigma_x, sigma_y] for the first set.
        boxes2 (torch.Tensor): A tensor of shape (B, 4) containing the normalized
            bounding box coordinates [x, y, w, h] for the second set.
        offset2 (torch.Tensor): A tensor of shape (B, M, 2) containing the relative
            offsets [offset_x, offset_y] for the second set.
        sigma2 (torch.Tensor): A tensor of shape (B, M, 2) containing the relative
            standard deviations [sigma_x, sigma_y] for the second set.
    
    Returns:
        torch.Tensor: A tensor of shape (B, N, M) containing pairwise Bhattacharyya
            coefficients between all pairs of Gaussians.
    """    
    # Extract box centers and sizes for both sets
    centers1 = boxes1[:, :2]  # Shape: (B, 2)
    sizes1 = boxes1[:, 2:]    # Shape: (B, 2)
    half_sizes1 = sizes1 / 2.0
    
    centers2 = boxes2[:, :2]  # Shape: (B, 2)
    sizes2 = boxes2[:, 2:]    # Shape: (B, 2)
    half_sizes2 = sizes2 / 2.0
    
    # Calculate effective centers and sigmas for both sets
    # For set 1: expand to (B, N, 2)
    effective_centers1 = centers1.unsqueeze(1) + offset1 * half_sizes1.unsqueeze(1)
    effective_sigmas1 = sigma1 * half_sizes1.unsqueeze(1)
    
    # For set 2: expand to (B, M, 2)
    effective_centers2 = centers2.unsqueeze(1) + offset2 * half_sizes2.unsqueeze(1)
    effective_sigmas2 = sigma2 * half_sizes2.unsqueeze(1)
    
    # Add epsilon for numerical stability
    epsilon = 1e-6
    effective_sigmas1 = effective_sigmas1.clamp(min=epsilon)
    effective_sigmas2 = effective_sigmas2.clamp(min=epsilon)
    
    # Expand dimensions for pairwise computation
    # Set 1: (B, N, 2) -> (B, N, 1, 2)
    mu1 = effective_centers1.unsqueeze(2)  # (B, N, 1, 2)
    sig1 = effective_sigmas1.unsqueeze(2)  # (B, N, 1, 2)
    
    # Set 2: (B, M, 2) -> (B, 1, M, 2)
    mu2 = effective_centers2.unsqueeze(1)  # (B, 1, M, 2)
    sig2 = effective_sigmas2.unsqueeze(1)  # (B, 1, M, 2)
    
    # Compute Bhattacharyya coefficient for each dimension
    # Shape after broadcasting: (B, N, M, 2)
    
    # Distance term: (mu1 - mu2)^2
    mu_diff_sq = (mu1 - mu2) ** 2  # (B, N, M, 2)
    
    # Variance sum: sigma1^2 + sigma2^2
    var_sum = sig1 ** 2 + sig2 ** 2  # (B, N, M, 2)
    
    # Exponential term: exp(-0.25 * (mu1 - mu2)^2 / (sigma1^2 + sigma2^2))
    exp_term = torch.exp(-0.25 * mu_diff_sq / var_sum)  # (B, N, M, 2)
    
    # Square root term: sqrt(2 * sigma1 * sigma2 / (sigma1^2 + sigma2^2))
    sqrt_term = torch.sqrt(2 * sig1 * sig2 / var_sum)  # (B, N, M, 2)
    
    # Bhattacharyya coefficient for each dimension
    bc_per_dim = exp_term * sqrt_term  # (B, N, M, 2)
    
    # For 2D Gaussians with diagonal covariance, multiply coefficients across dimensions
    bc_2d = bc_per_dim[:, :, :, 0] * bc_per_dim[:, :, :, 1]  # (B, N, M)
    
    return bc_2d