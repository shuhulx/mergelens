"""MergeLens MCP server — 8 tools for AI assistant integration.

Tools:
    compare_models — Layer-by-layer weight comparison + MCI score
    diagnose_merge — Analyze a MergeKit config
    get_conflict_zones — Quick conflict identification
    suggest_strategy — Recommend merge method with generated YAML
    generate_report — Create HTML report
    explain_layer — Explain a layer's role in merging
    get_compatibility_score — Quick MCI score
    audit_model — Run capability probes (requires mergelens[audit])
"""

from __future__ import annotations

from typing import Any


def create_server():
    """Create and configure the MergeLens MCP server."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        raise ImportError(
            "MCP server requires the 'mcp' package. Install with: pip install mergelens[mcp]"
        )

    try:
        mcp = FastMCP("mergelens", description="Pre-merge diagnostics for LLM model merging")
    except TypeError:
        # Older mcp versions don't support description kwarg
        mcp = FastMCP("mergelens")

    @mcp.tool()
    def compare_models(
        models: list[str],
        base_model: str | None = None,
        device: str = "cpu",
        svd_rank: int = 64,
    ) -> dict[str, Any]:
        """Compare two or more models layer-by-layer with rich diagnostics.

        Returns MCI score, layer metrics, conflict zones, and strategy recommendation.
        """
        from mergelens.compare.analyzer import compare_models as _compare

        result = _compare(
            model_paths=models,
            base_model=base_model,
            device=device,
            svd_rank=svd_rank,
            show_progress=False,
        )
        return result.model_dump()

    @mcp.tool()
    def diagnose_merge(config_yaml: str, device: str = "cpu") -> dict[str, Any]:
        """Diagnose a MergeKit YAML config for potential issues before merging.

        Parses the config, loads referenced models, and identifies interference.
        """
        import tempfile
        from pathlib import Path

        from mergelens.diagnose import diagnose_config

        # Write YAML to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            config_path = f.name

        try:
            result = diagnose_config(config_path, device=device)
            return result.model_dump()
        finally:
            Path(config_path).unlink(missing_ok=True)

    @mcp.tool()
    def get_conflict_zones(
        models: list[str],
        device: str = "cpu",
    ) -> list[dict[str, Any]]:
        """Quickly identify conflict zones between models.

        Returns a list of layer ranges where models disagree significantly.
        """
        from mergelens.compare.analyzer import compare_models as _compare

        result = _compare(
            model_paths=models,
            device=device,
            show_progress=False,
            include_strategy=False,
        )
        return [z.model_dump() for z in result.conflict_zones]

    @mcp.tool()
    def suggest_strategy(
        models: list[str],
        base_model: str | None = None,
        device: str = "cpu",
    ) -> dict[str, Any]:
        """Recommend the best merge method with a ready-to-use MergeKit YAML config.

        Maps diagnostic profiles to optimal merge strategies.
        """
        from mergelens.compare.analyzer import compare_models as _compare

        result = _compare(
            model_paths=models,
            base_model=base_model,
            device=device,
            show_progress=False,
        )
        if result.strategy:
            return result.strategy.model_dump()
        return {"error": "Could not generate strategy recommendation."}

    @mcp.tool()
    def generate_report(
        models: list[str],
        output_path: str = "mergelens_report.html",
        device: str = "cpu",
    ) -> str:
        """Generate an interactive HTML diagnostic report.

        Creates a self-contained HTML file with Plotly charts.
        """
        from pathlib import Path

        from mergelens.compare.analyzer import compare_models as _compare
        from mergelens.report.generator import generate_report as _report

        # Security: prevent path traversal from MCP clients
        resolved = Path(output_path).resolve()
        cwd = Path.cwd().resolve()
        if not resolved.is_relative_to(cwd):
            raise ValueError(
                f"output_path must be within the current working directory. Resolved to: {resolved}"
            )

        result = _compare(model_paths=models, device=device, show_progress=False)
        path = _report(compare_result=result, output_path=output_path)
        return f"Report saved to {path}"

    @mcp.tool()
    def explain_layer(layer_name: str) -> str:
        """Explain what a transformer layer does and its role in merging.

        Helps users understand which layers matter most for merge quality.
        """
        from mergelens.compare.loader import classify_layer
        from mergelens.models import LayerType

        layer_type = classify_layer(layer_name)

        explanations = {
            LayerType.ATTENTION_Q: "Query projection in self-attention. Controls what the model 'looks for' in context. Conflicts here affect how the model attends to different parts of input.",
            LayerType.ATTENTION_K: "Key projection in self-attention. Defines what information each token 'advertises'. Critical for maintaining coherent attention patterns.",
            LayerType.ATTENTION_V: "Value projection in self-attention. Carries the actual information retrieved by attention. Changes here directly affect output quality.",
            LayerType.ATTENTION_O: "Output projection in self-attention. Mixes attention head outputs. Sensitive to merging — conflicts can break multi-head coordination.",
            LayerType.MLP_GATE: "Gate projection in MLP. Controls information flow through the FFN. In SwiGLU architectures, this is the gating mechanism.",
            LayerType.MLP_UP: "Up projection in MLP. Expands hidden dimension. Changes here affect the model's internal knowledge representations.",
            LayerType.MLP_DOWN: "Down projection in MLP. Compresses back to hidden dim. Often where factual knowledge is stored.",
            LayerType.NORM: "Layer normalization. Relatively safe to merge — small parameters that mainly affect scale. Low conflict risk.",
            LayerType.EMBEDDING: "Token embedding layer. Very sensitive — conflicts here affect ALL tokens. Consider using one model's embeddings entirely.",
            LayerType.LM_HEAD: "Language model head. Maps hidden states to vocabulary. Must match embedding layer for consistency. Usually tied to embeddings.",
            LayerType.OTHER: "Unrecognized layer type. Check the full layer name for hints about its function.",
        }

        explanation = explanations.get(layer_type, explanations[LayerType.OTHER])
        return f"**{layer_name}** (type: {layer_type.value})\n\n{explanation}"

    @mcp.tool()
    def get_compatibility_score(
        models: list[str],
        device: str = "cpu",
    ) -> dict[str, Any]:
        """Get a quick Merge Compatibility Index (0-100) for two or more models.

        The MCI is a composite score indicating how well models will merge.
        75+: Highly compatible, 55-75: Compatible, 35-55: Risky, <35: Incompatible.
        """
        from mergelens.compare.analyzer import compare_models as _compare

        result = _compare(
            model_paths=models,
            device=device,
            show_progress=False,
            include_strategy=False,
        )
        return result.mci.model_dump()

    @mcp.tool()
    def audit_model(
        base_model: str,
        merged_model: str,
        categories: list[str] | None = None,
        device: str = "cpu",
    ) -> dict[str, Any]:
        """Audit a merged model's capabilities against its base.

        Requires mergelens[audit] extra. Runs probes across categories
        like reasoning, code, chat, math, safety, instruction_following.
        """
        return {
            "status": "not_implemented",
            "message": "Audit module is not yet available. Install mergelens[audit] for future support.",
            "categories_available": [
                "reasoning",
                "code",
                "chat",
                "math",
                "safety",
                "instruction_following",
            ],
        }

    return mcp
