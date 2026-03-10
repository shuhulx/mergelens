"""Tensor operations for metric computation."""

from __future__ import annotations

import numpy as np
import torch


def flatten_to_2d(tensor: torch.Tensor) -> torch.Tensor:
    """Flatten tensor to 2D matrix for SVD and other operations.

    For 1D tensors, reshapes to (1, N).
    For 3D+, reshapes to (first_dim, product_of_rest).
    """
    if tensor.ndim == 1:
        return tensor.unsqueeze(0)
    if tensor.ndim == 2:
        return tensor
    return tensor.reshape(tensor.shape[0], -1)


MAX_ELEMENTS_FOR_SVD: int = 50_000_000  # 50M elements; skip spectral metrics above this


def truncated_svd(
    matrix: torch.Tensor, k: int = 64
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute truncated SVD, returning top-k singular vectors/values.

    Returns (U[:, :k], S[:k], Vh[:k, :]) where k = min(k, min(matrix.shape)).
    Raises ValueError if the matrix exceeds MAX_ELEMENTS_FOR_SVD.
    """
    matrix = flatten_to_2d(matrix).float()
    if matrix.numel() > MAX_ELEMENTS_FOR_SVD:
        raise ValueError(
            f"Tensor too large for SVD ({matrix.numel():,} elements > "
            f"{MAX_ELEMENTS_FOR_SVD:,} limit). Skipping spectral metrics."
        )
    k = min(k, min(matrix.shape))
    U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
    return U[:, :k], S[:k], Vh[:k, :]


def effective_rank(matrix: torch.Tensor) -> float:
    """Compute effective rank via Shannon entropy of normalized singular values.

    erank = exp(-sum(p_i * log(p_i))) where p_i = s_i / sum(s_j)

    Higher effective rank = more distributed information.
    Returns float >= 1.0.
    """
    matrix = flatten_to_2d(matrix).float()
    S = torch.linalg.svdvals(matrix)
    S = S[S > 1e-10]  # Filter near-zero
    if len(S) == 0:
        return 1.0
    # Normalize to probability distribution
    p = S / S.sum()
    entropy = -(p * torch.log(p)).sum().item()
    return float(np.exp(entropy))


def grassmann_distance(U1: torch.Tensor, U2: torch.Tensor) -> float:
    """Compute Grassmann distance between two subspaces.

    Uses principal angles: distance = sqrt(sum(theta_i^2))
    Returns value in [0, pi/2 * sqrt(k)], normalized to [0, 1].
    """
    if U1.shape[1] == 0 or U2.shape[1] == 0:
        return float('nan')
    # Compute cosines of principal angles
    M = U1.T @ U2
    sigmas = torch.linalg.svdvals(M)
    sigmas = torch.clamp(sigmas, -1.0, 1.0)
    # Principal angles
    angles = torch.acos(sigmas)
    distance = torch.sqrt((angles**2).sum()).item()
    # Normalize by max possible distance
    max_distance = (np.pi / 2) * np.sqrt(min(U1.shape[1], U2.shape[1]))
    if max_distance == 0:
        return 0.0
    return float(min(distance / max_distance, 1.0))


def compute_task_vector(model_weights: torch.Tensor, base_weights: torch.Tensor) -> torch.Tensor:
    """Compute task vector: model_weights - base_weights."""
    return model_weights.float() - base_weights.float()
