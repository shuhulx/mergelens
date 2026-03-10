<p align="center">
  <h1 align="center">MergeLens</h1>
  <p align="center"><strong>Pre-merge diagnostics for LLM model merging</strong></p>
  <p align="center">
    <a href="https://pypi.org/project/mergelens/"><img src="https://img.shields.io/pypi/v/mergelens" alt="PyPI"></a>
    <a href="https://pypi.org/project/mergelens/"><img src="https://img.shields.io/pypi/pyversions/mergelens" alt="Python"></a>
    <a href="https://github.com/shuhulx/mergelens/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue" alt="License"></a>
    <a href="https://pypi.org/project/mergelens/"><img src="https://img.shields.io/pypi/dm/mergelens" alt="Downloads"></a>
    <img src="https://img.shields.io/badge/tests-192%20passed-brightgreen.svg" alt="Tests">
  </p>
</p>

---

34% of top Open LLM Leaderboard models are merges, yet merging is blind trial-and-error. MergeLens tells you **before you merge** whether it will work — and which method to use.

## Features

- **Single compatibility score** — Merge Compatibility Index (MCI): 0-100, go/no-go verdict
- **10 diagnostic metrics** — cosine similarity, spectral overlap, sign disagreement, TSV interference, CKA, and more
- **Strategy recommender** — optimal merge method + ready-to-paste MergeKit YAML
- **Conflict zone detection** — pinpoints problematic layers
- **Interactive HTML reports** — self-contained Plotly dashboards
- **MCP server** — AI assistants can diagnose merges natively
- **Memory efficient** — lazy safetensors loading, peak memory = 2× largest layer

## Install

```bash
pip install mergelens
```

Optional extras:

```bash
pip install mergelens[report]  # Interactive HTML report generation
pip install mergelens[mcp]    # MCP server for AI assistants
pip install mergelens[audit]  # Capability probing (requires transformers)
pip install mergelens[all]    # Everything
```

## Quick Start

### CLI

Compare two models (local paths or HuggingFace Hub IDs):

```bash
mergelens compare model_a/ model_b/
mergelens compare meta-llama/Llama-3-8B mistralai/Mistral-7B-v0.1
```

Add a base model for task vector metrics:

```bash
mergelens compare model_a/ model_b/ --base base_model/
```

Generate an HTML report:

```bash
mergelens compare model_a/ model_b/ --report report.html
```

Diagnose a MergeKit config before running it:

```bash
mergelens diagnose merge.yaml
```

### Python API

```python
from mergelens import compare_models

result = compare_models(["model_a/", "model_b/"])

print(f"MCI: {result.mci.score} — {result.mci.verdict}")
# MCI: 72.3 — compatible
```

Inspect conflicts and get a strategy recommendation:

```python
for zone in result.conflict_zones:
    print(f"Layers {zone.start_layer}-{zone.end_layer}: {zone.severity.value}")

if result.strategy:
    print(f"Recommended: {result.strategy.method.value}")
    print(result.strategy.mergekit_yaml)  # copy-paste into MergeKit
```

Diagnose a MergeKit config:

```python
from mergelens import diagnose_config

result = diagnose_config("merge.yaml")
print(f"Overall interference: {result.overall_interference:.4f}")
```

## Metrics

| Metric | What It Measures | Range | Source |
|--------|-----------------|-------|--------|
| Cosine Similarity | Weight vector alignment | [-1, 1] | Standard |
| L2 Distance | Normalized weight divergence | [0, +inf) | Standard |
| KL Divergence | Weight distribution difference | [0, +inf) | Standard |
| Spectral Subspace Overlap | Top-k SVD direction alignment | [0, 1] | Zhou et al. 2026 |
| Effective Rank Ratio | Dimensionality compatibility | [0, 1] | Shannon entropy |
| Sign Disagreement Rate | Parameter sign conflicts | [0, 1] | TIES-Merging (Yadav et al. 2023) |
| TSV Interference | Cross-task singular vector conflict | [0, +inf) | Gargiulo et al. 2025 |
| Task Vector Energy | Knowledge concentration in top SVs | [0, 1] | Choi et al. 2024 |
| CKA Similarity | Activation representation similarity | [0, 1] | Kornblith et al. 2019 |
| **Merge Compatibility Index** | **Composite go/no-go score** | **[0, 100]** | **Ours** |

<details>
<summary><strong>MCI Verdicts</strong></summary>

| Score | Verdict | Meaning |
|-------|---------|---------|
| 75-100 | Highly Compatible | Merge with confidence |
| 55-74 | Compatible | Should work, monitor quality |
| 35-54 | Risky | Expect degradation, use targeted methods |
| 0-34 | Incompatible | These models likely shouldn't be merged |

</details>

<details>
<summary><strong>Strategy Recommendations</strong></summary>

MergeLens maps diagnostic profiles to merge methods. Different metrics predict success for different methods ([Zhou et al. 2026](https://arxiv.org/abs/2601.22285) found only 46.7% metric overlap between methods):

| Diagnostic Profile | Recommended Method |
|--------------------|--------------------|
| High cosine similarity everywhere | SLERP |
| High sign disagreement (>30%) | TIES |
| Concentrated task vector energy | DARE |
| Low spectral overlap | Linear (small alpha) |

Each recommendation includes a ready-to-paste MergeKit YAML config.

</details>

## MCP Integration

```json
{
  "mcpServers": {
    "mergelens": {
      "command": "mergelens",
      "args": ["serve"]
    }
  }
}
```

Tools: `compare_models`, `diagnose_merge`, `get_conflict_zones`, `suggest_strategy`, `generate_report`, `explain_layer`, `get_compatibility_score`, `audit_model`

## How It Works

MergeLens loads model weights lazily via memory-mapped safetensors (peak memory: 2× largest layer, not 2× full model). It computes metrics layer-by-layer, detects conflict zones, and aggregates everything into the MCI score.

**Security:** No pickle/torch.load (safetensors only), `yaml.safe_load()`, tensor size limits, no credential leakage.

## Development

```bash
git clone https://github.com/shuhulx/mergelens.git
cd mergelens
pip install -e ".[dev,all]"
pytest
```

## References

- Zhou et al. 2026, "Demystifying Mergeability of Homologous LLMs" ([arXiv:2601.22285](https://arxiv.org/abs/2601.22285))
- Gargiulo et al. CVPR 2025, "Task Singular Vectors" ([arXiv:2412.00081](https://arxiv.org/abs/2412.00081))
- Yadav et al. NeurIPS 2023, "TIES-Merging" ([arXiv:2306.01708](https://arxiv.org/abs/2306.01708))
- Choi et al. 2024, "Revisiting Weight Averaging for Model Merging" ([arXiv:2412.12153](https://arxiv.org/abs/2412.12153))
- Kornblith et al. 2019, "Similarity of Neural Network Representations Revisited" ([arXiv:1905.00414](https://arxiv.org/abs/1905.00414))
- Rahamim et al. 2026, "Will it Merge?" ([arXiv:2601.06672](https://arxiv.org/abs/2601.06672))

## Changelog

### 0.1.5

- **Fixed:** Sign disagreement rate now includes zero-vs-nonzero mismatches
- **Fixed:** SLERP strategy confidence now factors in MCI score
- **Fixed:** Conflict zone detection threshold aligned with severity bands (0.85 -> 0.80)
- **Fixed:** Per-layer override factors now scale by zone severity (LOW=0.9 to CRITICAL=0.3)
- **Fixed:** TSV interference returns NaN instead of 0.0 when no pairs available
- **Fixed:** Grassmann distance returns NaN for degenerate zero subspaces
- **Fixed:** Empty models list in MergeKit config parser now raises ValueError
- **Fixed:** Path traversal hardening in MCP report generation
- **Fixed:** MCP temp file cleanup in proper try/finally
- **Fixed:** Report title XSS protection via html.escape()
- **Fixed:** JSON parse errors in hf_utils now caught gracefully
- **Improved:** MCI confidence uses weighted metric importance instead of simple count
- **Improved:** Tensor shape validation in all pairwise metric functions
- **Improved:** Cache hash uses full tensor for <1M elements, 1000 samples for larger
- **Added:** Cache key versioning (v2) to prevent stale cache hits
- **Added:** Attribution normalization fallback warning

## License

Apache 2.0
