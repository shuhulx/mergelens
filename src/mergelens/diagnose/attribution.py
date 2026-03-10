"""Source model contribution decomposition — which model 'won' each layer."""

from __future__ import annotations

import logging

from mergelens.compare.loader import ModelHandle, find_common_tensors
from mergelens.compare.metrics import cosine_similarity

logger = logging.getLogger(__name__)


def compute_attribution(
    merged_handle: ModelHandle,
    source_handles: list[ModelHandle],
) -> dict[str, dict[str, float]]:
    """Compute per-layer attribution: how much each source contributed to the merged result.

    Uses cosine similarity between merged weights and each source's weights.
    Returns {layer_name: {source_name: contribution_score}}.
    """
    all_handles = [merged_handle, *source_handles]
    common = find_common_tensors(all_handles)

    attribution = {}

    for name in common:
        merged_tensor = merged_handle.get_tensor(name)

        similarities = {}
        for sh in source_handles:
            source_tensor = sh.get_tensor(name)
            sim = cosine_similarity(merged_tensor, source_tensor)
            similarities[sh.info.name] = sim

        # Normalize to contributions that sum to ~1
        total = sum(max(0.0, v) for v in similarities.values())
        if total > 0:
            contributions = {k: round(max(0.0, v) / total, 4) for k, v in similarities.items()}
        else:
            logger.warning(
                "All source contributions are negative for layer '%s'; "
                "falling back to uniform distribution.",
                name,
            )
            contributions = {k: round(1.0 / len(similarities), 4) for k in similarities}

        attribution[name] = contributions

    return attribution
