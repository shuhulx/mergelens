"""Merge strategy recommender — maps diagnostic profiles to optimal merge methods.

Zhou et al. 2026 found only 46.7% metric overlap between what predicts
success for different merge methods. Different diagnostic profiles
map to different optimal methods.
"""

from __future__ import annotations

import logging

import yaml

from mergelens.models import (
    CompareResult,
    MergeMethod,
    StrategyRecommendation,
)


def recommend_strategy(result: CompareResult) -> StrategyRecommendation:
    """Recommend a merge strategy based on diagnostic profile.

    Rule-based decision tree (interpretable, no training data needed):
    - High cosine sim everywhere → SLERP
    - Isolated conflict zones → SLERP with per-layer t overrides
    - High sign disagreement (>30%) → TIES
    - Concentrated task vector energy → DARE
    - Low spectral overlap + high rank divergence → Linear with small alpha
    - MCI < 30 → Warning: may not be compatible
    """
    mci = result.mci
    metrics = result.layer_metrics
    conflicts = result.conflict_zones

    # Aggregate diagnostic signals
    avg_cos = sum(m.cosine_similarity for m in metrics) / max(len(metrics), 1)

    sign_rates = [m.sign_disagreement_rate for m in metrics if m.sign_disagreement_rate is not None]
    avg_sign_disagree = sum(sign_rates) / len(sign_rates) if sign_rates else 0.0

    energy_scores = [m.task_vector_energy for m in metrics if m.task_vector_energy is not None]
    avg_energy = sum(energy_scores) / len(energy_scores) if energy_scores else 0.5

    spectral_scores = [m.spectral_overlap for m in metrics if m.spectral_overlap is not None]
    avg_spectral = sum(spectral_scores) / len(spectral_scores) if spectral_scores else 0.5

    rank_scores = [m.effective_rank_ratio for m in metrics if m.effective_rank_ratio is not None]
    avg_rank_ratio = sum(rank_scores) / len(rank_scores) if rank_scores else 0.5

    warnings: list[str] = []
    per_layer_overrides: dict = {}

    # Decision tree
    if mci.score < 30:
        return StrategyRecommendation(
            method=MergeMethod.LINEAR,
            confidence=0.3,
            reasoning=(
                f"MCI score is {mci.score}/100 ({mci.verdict}). "
                "These models may not be compatible for merging. "
                "If you proceed, use linear merge with very low weight (alpha=0.1-0.2)."
            ),
            mergekit_yaml=_generate_yaml(MergeMethod.LINEAR, result, alpha=0.1),
            warnings=["Low compatibility detected. Merge quality likely to be poor."],
        )

    if avg_sign_disagree > 0.30:
        # High sign disagreement → TIES resolves by design
        confidence = 0.8 if avg_sign_disagree < 0.5 else 0.6
        return StrategyRecommendation(
            method=MergeMethod.TIES,
            confidence=confidence,
            reasoning=(
                f"Sign disagreement rate is {avg_sign_disagree:.1%}, above 30% threshold. "
                "TIES merging resolves sign conflicts by trimming, resetting, and electing dominant signs."
            ),
            mergekit_yaml=_generate_yaml(MergeMethod.TIES, result, density=0.5),
            warnings=warnings,
            per_layer_overrides=per_layer_overrides,
        )

    if avg_energy > 0.8:
        # Concentrated energy → DARE (prune-then-merge preserves concentrated knowledge)
        return StrategyRecommendation(
            method=MergeMethod.DARE_TIES,
            confidence=0.75,
            reasoning=(
                f"Task vector energy is highly concentrated (avg {avg_energy:.2f}). "
                "DARE randomly prunes task vectors before merging, which works well "
                "when knowledge is concentrated in few directions."
            ),
            mergekit_yaml=_generate_yaml(MergeMethod.DARE_TIES, result, density=0.5),
            warnings=warnings,
        )

    if avg_spectral < 0.4 or avg_rank_ratio < 0.5:
        # Low spectral overlap or high rank divergence → conservative linear
        alpha = max(0.2, min(0.5, avg_cos - 0.3))
        return StrategyRecommendation(
            method=MergeMethod.LINEAR,
            confidence=0.6,
            reasoning=(
                f"Low spectral overlap ({avg_spectral:.2f}) or rank divergence "
                f"(ratio {avg_rank_ratio:.2f}). Models have different weight structure. "
                f"Linear merge with conservative alpha={alpha:.2f} recommended."
            ),
            mergekit_yaml=_generate_yaml(MergeMethod.LINEAR, result, alpha=alpha),
            warnings=warnings,
        )

    # Default: SLERP (good general-purpose method)
    t = min(0.5, max(0.3, 1.0 - avg_cos))  # Lower t when models are more similar

    # Add per-layer overrides for conflict zones, scaled by severity
    severity_factor = {"LOW": 0.9, "MEDIUM": 0.7, "HIGH": 0.5, "CRITICAL": 0.3}
    if conflicts:
        for zone in conflicts:
            factor = severity_factor.get(
                zone.severity.value if hasattr(zone.severity, "value") else str(zone.severity), 0.5
            )
            for layer_name in zone.layer_names:
                per_layer_overrides[layer_name] = {"t": t * factor}
        warnings.append(f"Found {len(conflicts)} conflict zone(s). Per-layer overrides applied.")

    base_conf = 0.85 if avg_cos > 0.9 else 0.7
    slerp_confidence = round(base_conf * min(mci.score / 75.0, 1.0), 2)

    return StrategyRecommendation(
        method=MergeMethod.SLERP,
        confidence=slerp_confidence,
        reasoning=(
            f"Models are generally compatible (cosine sim {avg_cos:.3f}, "
            f"spectral overlap {avg_spectral:.2f}). "
            f"SLERP with t={t:.2f} provides smooth interpolation."
        ),
        mergekit_yaml=_generate_yaml(MergeMethod.SLERP, result, t=t),
        warnings=warnings,
        per_layer_overrides=per_layer_overrides,
    )


def _generate_yaml(
    method: MergeMethod,
    result: CompareResult,
    **params,
) -> str:
    """Generate a ready-to-use MergeKit YAML config."""
    models = result.models

    logger = logging.getLogger(__name__)

    if method == MergeMethod.SLERP:
        if len(models) > 2:
            logger.warning("SLERP only supports 2 models; using first two, ignoring the rest.")
        config = {
            "merge_method": "slerp",
            "slices": [
                {
                    "sources": [
                        {
                            "model": models[0].path_or_repo,
                            "layer_range": [0, models[0].num_layers or 32],
                        },
                        {
                            "model": models[1].path_or_repo,
                            "layer_range": [0, models[1].num_layers or 32],
                        },
                    ],
                }
            ],
            "parameters": {"t": [params.get("t", 0.5)]},
            "dtype": "bfloat16",
        }
    elif method == MergeMethod.TIES:
        config = {
            "merge_method": "ties",
            "slices": [
                {
                    "sources": [{"model": m.path_or_repo} for m in models],
                }
            ],
            "base_model": models[0].path_or_repo,
            "parameters": {
                "density": params.get("density", 0.5),
                "weight": [1.0 / len(models)] * len(models),
            },
            "dtype": "bfloat16",
        }
    elif method in (MergeMethod.DARE_TIES, MergeMethod.DARE_LINEAR):
        config = {
            "merge_method": method.value,
            "slices": [
                {
                    "sources": [{"model": m.path_or_repo} for m in models],
                }
            ],
            "base_model": models[0].path_or_repo,
            "parameters": {
                "density": params.get("density", 0.5),
                "weight": [1.0 / len(models)] * len(models),
            },
            "dtype": "bfloat16",
        }
    else:  # LINEAR
        config = {
            "merge_method": "linear",
            "slices": [
                {
                    "sources": [{"model": m.path_or_repo} for m in models],
                }
            ],
            "parameters": {
                "weight": (
                    [params.get("alpha", 0.5), 1.0 - params.get("alpha", 0.5)]
                    if len(models) == 2
                    else [params.get("alpha", 0.5)]
                    + [(1.0 - params.get("alpha", 0.5)) / (len(models) - 1)] * (len(models) - 1)
                ),
            },
            "dtype": "bfloat16",
        }

    return yaml.dump(config, default_flow_style=False, sort_keys=False)
