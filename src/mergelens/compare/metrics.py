"""Comparison metrics for model weight analysis.

Implements 10 metrics:
  cosine_similarity, l2_distance, kl_divergence, spectral_subspace_overlap,
  effective_rank_ratio, tsv_interference_score, sign_disagreement_rate,
  centered_task_vector_energy, cka_similarity, merge_compatibility_index

References:
  - Zhou et al. 2026 "Demystifying Mergeability" (arXiv:2601.22285) — subspace overlap
  - Gargiulo et al. CVPR 2025 "Task Singular Vectors" (arXiv:2412.00081) — TSV interference
  - Yadav et al. NeurIPS 2023 "TIES-Merging" (arXiv:2306.01708) — sign disagreement
  - Choi et al. 2024 "Revisiting Weight Averaging" (arXiv:2412.12153) — task vector energy
  - Kornblith et al. 2019 — CKA similarity
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from mergelens.models import MergeCompatibilityIndex
from mergelens.utils.tensor_ops import (
    effective_rank,
    flatten_to_2d,
    grassmann_distance,
    truncated_svd,
)

# ── Metric Registry ───────────────────────────────────────────────

METRIC_REGISTRY: dict[str, Callable] = {}


def register_metric(name: str):
    """Decorator to register a metric function."""

    def decorator(func: Callable) -> Callable:
        METRIC_REGISTRY[name] = func
        return func

    return decorator


# ── Standard Metrics ──────────────────────────────────────────────


@register_metric("cosine_similarity")
def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between flattened weight vectors.

    Returns value in [-1, 1]. Higher = more similar.
    """
    if a.numel() != b.numel():
        raise ValueError(f"Tensor size mismatch: {a.numel()} vs {b.numel()}")
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    norm_a = torch.norm(a_flat)
    norm_b = torch.norm(b_flat)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    result = float(torch.dot(a_flat, b_flat) / (norm_a * norm_b))
    return max(-1.0, min(1.0, result))  # Clamp for float precision


@register_metric("l2_distance")
def l2_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """Normalized L2 distance between weight tensors.

    Normalized by the average L2 norm to make comparable across layers.
    Returns value >= 0. Lower = more similar.
    """
    if a.numel() != b.numel():
        raise ValueError(f"Tensor size mismatch: {a.numel()} vs {b.numel()}")
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    dist = torch.norm(a_flat - b_flat).item()
    avg_norm = (torch.norm(a_flat).item() + torch.norm(b_flat).item()) / 2
    if avg_norm < 1e-10:
        return 0.0
    return dist / avg_norm


@register_metric("kl_divergence")
def kl_divergence(a: torch.Tensor, b: torch.Tensor) -> float:
    """Approximate KL divergence between weight distributions.

    Treats weights as unnormalized distributions, applies softmax.
    Returns value >= 0. Lower = more similar.
    """
    if a.numel() != b.numel():
        raise ValueError(f"Tensor size mismatch: {a.numel()} vs {b.numel()}")
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    # Use temperature scaling for numerical stability
    temp = max(a_flat.std().item(), b_flat.std().item(), 1e-6)
    p = torch.softmax(a_flat / temp, dim=0)
    q = torch.softmax(b_flat / temp, dim=0)
    # KL(P || Q) with epsilon for numerical safety
    eps = 1e-10
    kl = (p * torch.log((p + eps) / (q + eps))).sum().item()
    return max(0.0, kl)


# ── Research-Based Metrics ────────────────────────────────────────


@register_metric("spectral_subspace_overlap")
def spectral_subspace_overlap(a: torch.Tensor, b: torch.Tensor, k: int = 64) -> float:
    """Top-k SVD subspace overlap via Grassmann distance.

    Measures how similar the principal directions of two weight matrices are.
    Returns value in [0, 1]. Higher = more overlap = more compatible.

    Based on: Zhou et al. 2026 "Demystifying Mergeability" (arXiv:2601.22285)
    """
    if a.numel() != b.numel():
        raise ValueError(f"Tensor size mismatch: {a.numel()} vs {b.numel()}")
    U_a, _, _ = truncated_svd(a, k=k)
    U_b, _, _ = truncated_svd(b, k=k)
    min_cols = min(U_a.shape[1], U_b.shape[1])
    U_a = U_a[:, :min_cols]
    U_b = U_b[:, :min_cols]
    # Grassmann distance returns [0, 1] where 0 = identical subspaces
    distance = grassmann_distance(U_a, U_b)
    return 1.0 - distance  # Convert to overlap


@register_metric("effective_rank_ratio")
def effective_rank_ratio(a: torch.Tensor, b: torch.Tensor) -> float:
    """Ratio of effective ranks — measures dimensionality compatibility.

    Returns min(erank_a, erank_b) / max(erank_a, erank_b).
    Value in [0, 1]. Higher = more compatible dimensionality.
    """
    if a.numel() != b.numel():
        raise ValueError(f"Tensor size mismatch: {a.numel()} vs {b.numel()}")
    er_a = effective_rank(a)
    er_b = effective_rank(b)
    if max(er_a, er_b) < 1e-10:
        return 1.0
    return min(er_a, er_b) / max(er_a, er_b)


@register_metric("sign_disagreement_rate")
def sign_disagreement_rate(
    task_vectors: list[torch.Tensor],
) -> float:
    """Per-parameter sign conflict rate across task vectors.

    Measures what fraction of parameters have conflicting signs across models.
    Returns value in [0, 1]. Lower = less conflict = more compatible.

    Based on: Yadav et al. NeurIPS 2023 "TIES-Merging" (arXiv:2306.01708)
    """
    if len(task_vectors) < 2:
        return 0.0

    signs = [torch.sign(tv.flatten().float()) for tv in task_vectors]
    total_disagreements = 0
    total_pairs = 0

    for i in range(len(signs)):
        for j in range(i + 1, len(signs)):
            # Count all sign mismatches, including zero vs non-zero
            sign_mismatch = (signs[i] != signs[j]).float().mean().item()
            total_disagreements += sign_mismatch
            total_pairs += 1

    if total_pairs == 0:
        return 0.0
    return total_disagreements / total_pairs


@register_metric("tsv_interference_score")
def tsv_interference_score(
    task_vectors: list[torch.Tensor],
    k: int = 64,
) -> float:
    """Cross-task singular vector interference score.

    Measures how much the principal singular vectors of different task vectors
    interfere with each other. High interference -> merging will cause conflicts.
    Returns value >= 0. Lower = less interference.

    Based on: Gargiulo et al. CVPR 2025 "Task Singular Vectors" (arXiv:2412.00081)
    """
    if len(task_vectors) < 2:
        return 0.0

    # Get top-k right singular vectors for each task vector
    Vs = []
    for tv in task_vectors:
        _, _, Vh = truncated_svd(tv, k=k)
        Vs.append(Vh)  # shape: (k, d)

    min_k = min(V.shape[0] for V in Vs)
    Vs = [V[:min_k, :] for V in Vs]

    # Compute pairwise interference: how aligned are the singular vectors?
    total_interference = 0.0
    n_pairs = 0
    for i in range(len(Vs)):
        for j in range(i + 1, len(Vs)):
            # Interference = Frobenius norm of V_i @ V_j^T
            # Normalized by k to give per-direction interference
            overlap = Vs[i] @ Vs[j].T
            interference = torch.norm(overlap, p="fro").item() / k
            total_interference += interference
            n_pairs += 1

    if n_pairs == 0:
        return float('nan')
    return total_interference / n_pairs


@register_metric("centered_task_vector_energy")
def centered_task_vector_energy(
    task_vector: torch.Tensor,
    k: int = 64,
) -> float:
    """Knowledge concentration in top-k singular vectors.

    Measures what fraction of the task vector's energy is in the top-k
    singular values. High concentration = knowledge is localized.
    Returns value in [0, 1].

    Based on: Choi et al. 2024 "Revisiting Weight Averaging" (arXiv:2412.12153)
    """
    matrix = flatten_to_2d(task_vector).float()
    S = torch.linalg.svdvals(matrix)
    total_energy = (S**2).sum().item()
    if total_energy < 1e-10:
        return 0.0
    k = min(k, len(S))
    top_k_energy = (S[:k] ** 2).sum().item()
    return top_k_energy / total_energy


@register_metric("cka_similarity")
def cka_similarity(
    activations_a: torch.Tensor,
    activations_b: torch.Tensor,
) -> float:
    """Centered Kernel Alignment between activation representations.

    Compares the representations two models produce, not just their weights.
    Requires activation matrices of shape (n_samples, hidden_dim).
    Returns value in [0, 1]. Higher = more similar representations.

    Based on: Kornblith et al. 2019 "Similarity of Neural Network Representations Revisited"
    """
    if activations_a.numel() != activations_b.numel():
        raise ValueError(f"Tensor size mismatch: {activations_a.numel()} vs {activations_b.numel()}")
    X = activations_a.float()
    Y = activations_b.float()

    # Center the activations
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    # Linear CKA: HSIC(K, L) / sqrt(HSIC(K, K) * HSIC(L, L))
    # where K = X @ X^T, L = Y @ Y^T
    XtX = X @ X.T
    YtY = Y @ Y.T
    YtX = Y @ X.T

    hsic_xy = torch.norm(YtX, p="fro").item() ** 2
    hsic_xx = torch.norm(XtX, p="fro").item() ** 2
    hsic_yy = torch.norm(YtY, p="fro").item() ** 2

    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-10:
        return 0.0
    result = hsic_xy / denom
    return float(max(0.0, min(1.0, result)))


# ── Composite Score ───────────────────────────────────────────────


@register_metric("merge_compatibility_index")
def merge_compatibility_index(
    cosine_sims: list[float],
    spectral_overlaps: list[float] | None = None,
    rank_ratios: list[float] | None = None,
    sign_disagreements: list[float] | None = None,
    tsv_scores: list[float] | None = None,
    energy_scores: list[float] | None = None,
    cka_scores: list[float] | None = None,
) -> MergeCompatibilityIndex:
    """Compute Merge Compatibility Index — a composite 0-100 score.

    Aggregates all available metrics with learned weights.
    Returns MCI with score, confidence interval, and go/no-go verdict.
    """
    if not cosine_sims:
        return MergeCompatibilityIndex(
            score=0.0,
            confidence=0.0,
            ci_lower=0.0,
            ci_upper=0.0,
            verdict="insufficient data",
            components={},
        )

    components = {}
    weights = {}

    # Cosine similarity (always available, most reliable)
    avg_cos = np.mean(cosine_sims)
    components["cosine_similarity"] = float(avg_cos)
    weights["cosine_similarity"] = 0.30

    # Spectral overlap
    if spectral_overlaps:
        avg_spec = np.mean(spectral_overlaps)
        components["spectral_overlap"] = float(avg_spec)
        weights["spectral_overlap"] = 0.20

    # Rank ratio
    if rank_ratios:
        avg_rank = np.mean(rank_ratios)
        components["rank_ratio"] = float(avg_rank)
        weights["rank_ratio"] = 0.10

    # Sign disagreement (inverted — lower is better)
    if sign_disagreements:
        avg_sign = 1.0 - np.mean(sign_disagreements)
        components["sign_agreement"] = float(avg_sign)
        weights["sign_agreement"] = 0.15

    # TSV interference (inverted and clamped)
    if tsv_scores:
        avg_tsv = max(0.0, 1.0 - np.mean(tsv_scores))
        components["tsv_compatibility"] = float(avg_tsv)
        weights["tsv_compatibility"] = 0.10

    # Task vector energy (moderate is best)
    if energy_scores:
        avg_energy = np.mean(energy_scores)
        # Bell curve: 0.3-0.7 is ideal
        energy_score = 1.0 - 2.0 * abs(avg_energy - 0.5)
        components["energy_balance"] = float(max(0.0, energy_score))
        weights["energy_balance"] = 0.05

    # CKA (activation-based)
    if cka_scores:
        avg_cka = np.mean(cka_scores)
        components["cka_similarity"] = float(avg_cka)
        weights["cka_similarity"] = 0.10

    # Normalize weights to sum to 1.0
    total_weight = sum(weights.values())
    if total_weight == 0:
        return MergeCompatibilityIndex(
            score=0.0,
            confidence=0.0,
            ci_lower=0.0,
            ci_upper=0.0,
            verdict="insufficient data",
            components=components,
        )

    # Weighted average
    raw_score = sum(components[k] * weights[k] / total_weight for k in components)
    score = float(np.clip(raw_score * 100, 0, 100))

    # Confidence weighted by metric importance
    metric_weights = {
        "cosine": 0.25, "spectral": 0.15, "rank": 0.10,
        "sign": 0.15, "tsv": 0.10, "energy": 0.10, "kl": 0.15,
    }
    confidence = sum(metric_weights.get(m, 0.1) for m in components)
    confidence = min(confidence, 1.0)

    # Confidence interval (wider with fewer metrics)
    margin = (1.0 - confidence) * 15 + 5  # 5-20 point margin
    ci_lower = float(np.clip(score - margin, 0, 100))
    ci_upper = float(np.clip(score + margin, 0, 100))

    # Verdict
    if score >= 75:
        verdict = "highly compatible"
    elif score >= 55:
        verdict = "compatible"
    elif score >= 35:
        verdict = "risky"
    else:
        verdict = "incompatible"

    return MergeCompatibilityIndex(
        score=round(score, 1),
        confidence=round(confidence, 2),
        ci_lower=round(ci_lower, 1),
        ci_upper=round(ci_upper, 1),
        verdict=verdict,
        components=components,
    )
