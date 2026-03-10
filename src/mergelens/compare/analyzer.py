"""Layer-by-layer model comparison orchestrator."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)
from rich.progress import track

from mergelens.compare.loader import ModelHandle, find_common_tensors, iter_aligned_tensors
from mergelens.compare.metrics import (
    centered_task_vector_energy,
    cosine_similarity,
    effective_rank_ratio,
    l2_distance,
    merge_compatibility_index,
    sign_disagreement_rate,
    spectral_subspace_overlap,
    tsv_interference_score,
)
from mergelens.compare.strategy import recommend_strategy
from mergelens.models import (
    CompareResult,
    ConflictZone,
    LayerMetrics,
    Severity,
)
from mergelens.utils.cache import MetricCache
from mergelens.utils.tensor_ops import compute_task_vector


def compare_models(
    model_paths: list[str],
    base_model: str | None = None,
    device: str = "cpu",
    metrics: list[str] | None = None,
    svd_rank: int = 64,
    cache: MetricCache | None = None,
    show_progress: bool = True,
    include_strategy: bool = True,
) -> CompareResult:
    """Compare two or more models layer-by-layer.

    Args:
        model_paths: Paths or HF repo IDs for models to compare.
        base_model: Optional base model for task vector computation.
            If not provided, first model is used as base.
        device: Torch device for computation ("cpu" or "cuda").
        metrics: List of metric names to compute (None = all available).
        svd_rank: Number of singular vectors for spectral metrics.
        cache: Optional metric cache instance.
        show_progress: Show rich progress bar.
        include_strategy: Whether to include merge strategy recommendation.

    Returns:
        CompareResult with all metrics, conflict zones, MCI, and optional strategy.
    """
    if len(model_paths) < 2:
        raise ValueError("Need at least 2 models to compare.")

    handles = [ModelHandle(p, device=device) for p in model_paths]
    base_handle = ModelHandle(base_model, device=device) if base_model else handles[0]

    all_handles = handles if base_model is None else [base_handle, *handles]
    common_names = find_common_tensors(all_handles)

    if not common_names:
        raise ValueError("No common tensor names found between models.")

    all_layer_metrics: list[LayerMetrics] = []
    all_cosines: list[float] = []
    all_spectral: list[float] = []
    all_rank_ratios: list[float] = []
    all_sign_disagree: list[float] = []
    all_tsv: list[float] = []
    all_energy: list[float] = []

    iterator = iter_aligned_tensors(all_handles, common_names)
    if show_progress:
        iterator = track(list(iterator), description="Comparing layers...")

    for name, layer_type, tensors in iterator:
        base_tensor = tensors[0]
        model_tensors = tensors[1:]

        for _i, model_tensor in enumerate(model_tensors):
            cos_sim = cosine_similarity(base_tensor, model_tensor)
            l2_dist = l2_distance(base_tensor, model_tensor)

            lm = LayerMetrics(
                layer_name=name,
                layer_type=layer_type,
                shape=tuple(base_tensor.shape),
                cosine_similarity=cos_sim,
                l2_distance=l2_dist,
            )

            all_cosines.append(cos_sim)

            # Spectral overlap (only for 2D+ tensors with enough elements)
            if base_tensor.numel() > 128:
                try:
                    spec_overlap = spectral_subspace_overlap(base_tensor, model_tensor, k=svd_rank)
                    lm.spectral_overlap = spec_overlap
                    all_spectral.append(spec_overlap)
                except Exception:
                    logger.debug(
                        "spectral_subspace_overlap computation failed for %s", name, exc_info=True
                    )

                try:
                    rank_r = effective_rank_ratio(base_tensor, model_tensor)
                    lm.effective_rank_ratio = rank_r
                    all_rank_ratios.append(rank_r)
                except Exception:
                    logger.debug(
                        "effective_rank_ratio computation failed for %s", name, exc_info=True
                    )

            task_vec = compute_task_vector(model_tensor, base_tensor)

            if task_vec.numel() > 64:
                try:
                    energy = centered_task_vector_energy(task_vec, k=svd_rank)
                    lm.task_vector_energy = energy
                    all_energy.append(energy)
                except Exception:
                    logger.debug(
                        "centered_task_vector_energy computation failed for %s", name, exc_info=True
                    )

            all_layer_metrics.append(lm)

        # Multi-model task vector metrics (need 2+ task vectors)
        if len(model_tensors) >= 2:
            task_vecs = [compute_task_vector(mt, base_tensor) for mt in model_tensors]

            try:
                sign_dis = sign_disagreement_rate(task_vecs)
                all_sign_disagree.append(sign_dis)
                for lm in all_layer_metrics[-len(model_tensors) :]:
                    lm.sign_disagreement_rate = sign_dis
            except Exception:
                logger.debug(
                    "sign_disagreement_rate computation failed for %s", name, exc_info=True
                )

            try:
                tsv = tsv_interference_score(task_vecs, k=svd_rank)
                all_tsv.append(tsv)
                for lm in all_layer_metrics[-len(model_tensors) :]:
                    lm.tsv_interference = tsv
            except Exception:
                logger.debug(
                    "tsv_interference_score computation failed for %s", name, exc_info=True
                )

    mci = merge_compatibility_index(
        cosine_sims=all_cosines,
        spectral_overlaps=all_spectral or None,
        rank_ratios=all_rank_ratios or None,
        sign_disagreements=all_sign_disagree or None,
        tsv_scores=all_tsv or None,
        energy_scores=all_energy or None,
    )

    conflict_zones = _detect_conflict_zones(all_layer_metrics)

    result = CompareResult(
        models=[h.info for h in handles],
        layer_metrics=all_layer_metrics,
        conflict_zones=conflict_zones,
        mci=mci,
    )

    if include_strategy:
        result.strategy = recommend_strategy(result)

    return result


def _detect_conflict_zones(
    layer_metrics: list[LayerMetrics],
    cos_threshold: float = 0.80,
    min_zone_size: int = 2,
) -> list[ConflictZone]:
    """Detect contiguous groups of layers with high disagreement."""
    zones: list[ConflictZone] = []
    current_zone_layers: list[tuple[int, LayerMetrics]] = []

    for i, lm in enumerate(layer_metrics):
        if lm.cosine_similarity < cos_threshold:
            current_zone_layers.append((i, lm))
        else:
            if len(current_zone_layers) >= min_zone_size:
                zones.append(_build_zone(current_zone_layers))
            current_zone_layers = []

    # Don't forget the last zone
    if len(current_zone_layers) >= min_zone_size:
        zones.append(_build_zone(current_zone_layers))

    return zones


def _build_zone(layers: list[tuple[int, LayerMetrics]]) -> ConflictZone:
    """Build a ConflictZone from a list of (index, LayerMetrics)."""
    indices = [i for i, _ in layers]
    metrics = [lm for _, lm in layers]
    avg_cos = sum(lm.cosine_similarity for lm in metrics) / len(metrics)

    sign_disagrees = [
        lm.sign_disagreement_rate for lm in metrics if lm.sign_disagreement_rate is not None
    ]
    avg_sign = sum(sign_disagrees) / len(sign_disagrees) if sign_disagrees else None

    if avg_cos < 0.5:
        severity = Severity.CRITICAL
    elif avg_cos < 0.7:
        severity = Severity.HIGH
    elif avg_cos < 0.8:
        severity = Severity.MEDIUM
    else:
        severity = Severity.LOW

    if severity == Severity.CRITICAL:
        rec = "Critical conflict zone. Consider excluding these layers from merge or using passthrough."
    elif severity == Severity.HIGH:
        rec = "High conflict. Use TIES merging with sign resolution or reduce merge weight for these layers."
    elif severity == Severity.MEDIUM:
        rec = "Moderate conflict. SLERP with reduced t value recommended for these layers."
    else:
        rec = "Minor conflict. Standard merge parameters should work."

    return ConflictZone(
        start_layer=indices[0],
        end_layer=indices[-1],
        layer_names=[lm.layer_name for lm in metrics],
        severity=severity,
        avg_cosine_sim=round(avg_cos, 4),
        avg_sign_disagreement=round(avg_sign, 4) if avg_sign is not None else None,
        recommendation=rec,
    )
