"""HuggingFace Hub utilities for model resolution and metadata."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import get_safetensors_metadata, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError


@dataclass
class ModelMetadata:
    """Metadata about a HuggingFace model."""

    repo_id: str
    architecture: str | None = None
    num_parameters: int | None = None
    safetensors_files: list[str] | None = None
    config: dict | None = None

    def __post_init__(self):
        if self.safetensors_files is None:
            self.safetensors_files = []
        if self.config is None:
            self.config = {}


def resolve_model_path(path_or_repo: str) -> tuple[str, bool]:
    """Resolve a model path to either local path or HF repo ID.

    Returns (resolved_path, is_local).
    """
    local_path = Path(path_or_repo)
    if local_path.exists() and local_path.is_dir():
        # Check for safetensors files
        st_files = list(local_path.glob("*.safetensors"))
        if st_files:
            return str(local_path.resolve()), True
    # Treat as HF repo ID
    return path_or_repo, False


def get_model_metadata(path_or_repo: str) -> ModelMetadata:
    """Get metadata about a model from local path or HF Hub."""
    resolved, is_local = resolve_model_path(path_or_repo)

    if is_local:
        return _get_local_metadata(resolved)
    return _get_hub_metadata(resolved)


def _get_local_metadata(local_path: str) -> ModelMetadata:
    """Get metadata from a local model directory."""
    import json

    path = Path(local_path)

    meta = ModelMetadata(repo_id=path.name)
    meta.safetensors_files = [f.name for f in path.glob("*.safetensors")]

    # Try to read config.json
    config_path = path / "config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                meta.config = json.load(f)
            meta.architecture = (
                meta.config.get("model_type") or meta.config.get("architectures", [None])[0]
            )
            # Estimate parameters from config
            meta.num_parameters = _estimate_params_from_config(meta.config)
        except (json.JSONDecodeError, IOError):
            pass  # Config file is optional/malformed

    return meta


def _get_hub_metadata(repo_id: str) -> ModelMetadata:
    """Get metadata from HuggingFace Hub (header-only, no download)."""
    meta = ModelMetadata(repo_id=repo_id)

    try:
        # Get safetensors metadata (header-only — no model download)
        st_meta = get_safetensors_metadata(repo_id)
        if st_meta and hasattr(st_meta, "parameter_count"):
            total = (
                sum(st_meta.parameter_count.values())
                if isinstance(st_meta.parameter_count, dict)
                else 0
            )
            if total > 0:
                meta.num_parameters = total
        if st_meta and hasattr(st_meta, "sharded") and st_meta.sharded:
            meta.safetensors_files = (
                list(st_meta.files_metadata.keys()) if hasattr(st_meta, "files_metadata") else []
            )
        else:
            meta.safetensors_files = ["model.safetensors"]
    except Exception:
        pass

    try:
        # Download just config.json
        config_path = hf_hub_download(repo_id, "config.json")
        import json

        with open(config_path) as f:
            meta.config = json.load(f)
        meta.architecture = (
            meta.config.get("model_type") or meta.config.get("architectures", [None])[0]
        )
        if meta.num_parameters is None:
            meta.num_parameters = _estimate_params_from_config(meta.config)
    except (EntryNotFoundError, RepositoryNotFoundError, json.JSONDecodeError, IOError):
        pass

    return meta


def check_architecture_compatibility(models: list[str]) -> tuple[bool, str]:
    """Check if models have compatible architectures for merging.

    Returns (is_compatible, message).
    """
    metas = [get_model_metadata(m) for m in models]
    architectures = {m.architecture for m in metas if m.architecture}

    if len(architectures) <= 1:
        return True, "Models have compatible architectures."
    return False, f"Architecture mismatch: {architectures}. Models may not be mergeable."


def _estimate_params_from_config(config: dict) -> int | None:
    """Rough parameter estimate from config.json fields."""
    h = config.get("hidden_size")
    n_layers = config.get("num_hidden_layers")
    v = config.get("vocab_size")
    inter = config.get("intermediate_size")
    if h and n_layers and v:
        # Very rough: embeddings + n_layers * (4*h*h + 2*h*inter) + lm_head
        inter = inter or 4 * h
        return v * h + n_layers * (4 * h * h + 2 * h * inter) + v * h
    return None
