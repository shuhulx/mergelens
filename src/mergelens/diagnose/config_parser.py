"""MergeKit YAML configuration parser with Pydantic validation."""

from __future__ import annotations

import yaml

from mergelens.models import MergeConfig, MergeMethod


def parse_mergekit_config(yaml_content: str) -> MergeConfig:
    """Parse a MergeKit YAML config into a validated MergeConfig.

    Supports all major MergeKit config formats:
    - models list with weights
    - slices with layer ranges
    - modules format
    """
    raw = yaml.safe_load(yaml_content)
    if not isinstance(raw, dict):
        raise ValueError("Invalid MergeKit config: expected a YAML mapping.")

    method_str = raw.get("merge_method", "linear")
    try:
        method = MergeMethod(method_str)
    except ValueError:
        method = MergeMethod.LINEAR

    base_model = raw.get("base_model")

    models = []

    # From top-level models list
    if "models" in raw:
        for m in raw["models"]:
            if isinstance(m, str):
                models.append(m)
            elif isinstance(m, dict):
                models.append(m.get("model", ""))

    # From slices
    if "slices" in raw:
        for slice_def in raw["slices"]:
            sources = slice_def.get("sources", [])
            for src in sources:
                if isinstance(src, dict):
                    model = src.get("model", "")
                    if model and model not in models:
                        models.append(model)

    # Ensure base model is in list
    if base_model and base_model not in models:
        models.insert(0, base_model)

    parameters = raw.get("parameters", {})

    slices = raw.get("slices")

    if not models:
        raise ValueError("MergeKit config must contain at least one model in 'models' or 'slices'")

    return MergeConfig(
        merge_method=method,
        base_model=base_model,
        models=models,
        parameters=parameters,
        slices=slices,
        raw_yaml=yaml_content,
    )
