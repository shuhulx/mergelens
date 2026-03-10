"""HTML report generator using Plotly and Jinja2."""

from __future__ import annotations

import html as html_mod
import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from mergelens.models import AuditResult, CompareResult, DiagnoseResult

_TEMPLATE_DIR = Path(__file__).parent / "templates"


def generate_report(
    compare_result: CompareResult | None = None,
    diagnose_result: DiagnoseResult | None = None,
    audit_result: AuditResult | None = None,
    output_path: str = "mergelens_report.html",
    title: str = "MergeLens Report",
) -> str:
    """Generate a self-contained interactive HTML report.

    Embeds Plotly.js for interactive charts. Single file, no external dependencies.

    Args:
        compare_result: Results from compare.models()
        diagnose_result: Results from diagnose.from_config()
        audit_result: Results from audit.run()
        output_path: Where to save the HTML file
        title: Report title

    Returns:
        Path to generated report.
    """
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template("base.html.j2")

    # Build plotly chart data
    charts = {}

    if compare_result:
        charts["mci"] = _build_mci_gauge(compare_result)
        charts["heatmap"] = _build_similarity_heatmap(compare_result)
        charts["spectral"] = _build_spectral_chart(compare_result)
        charts["conflicts"] = _build_conflict_chart(compare_result)

    title = html_mod.escape(title)

    html = template.render(
        title=title,
        compare=compare_result,
        diagnose=diagnose_result,
        audit=audit_result,
        charts=charts,
        charts_json={k: json.dumps(v) for k, v in charts.items()},
    )

    Path(output_path).write_text(html)
    return output_path


def _build_mci_gauge(result: CompareResult) -> dict:
    """Build Plotly gauge chart for MCI score."""
    mci = result.mci
    return {
        "data": [
            {
                "type": "indicator",
                "mode": "gauge+number+delta",
                "value": mci.score,
                "title": {"text": "Merge Compatibility Index"},
                "gauge": {
                    "axis": {"range": [0, 100]},
                    "bar": {"color": _score_color(mci.score)},
                    "steps": [
                        {"range": [0, 35], "color": "#ffebee"},
                        {"range": [35, 55], "color": "#fff3e0"},
                        {"range": [55, 75], "color": "#e8f5e9"},
                        {"range": [75, 100], "color": "#c8e6c9"},
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 2},
                        "thickness": 0.75,
                        "value": mci.score,
                    },
                },
            }
        ],
        "layout": {
            "height": 300,
            "margin": {"t": 50, "b": 20, "l": 30, "r": 30},
            "annotations": [
                {
                    "text": f"{mci.verdict} (CI: {mci.ci_lower:.0f}-{mci.ci_upper:.0f})",
                    "x": 0.5,
                    "y": 0,
                    "showarrow": False,
                    "font": {"size": 14},
                }
            ],
        },
    }


def _build_similarity_heatmap(result: CompareResult) -> dict:
    """Build Plotly heatmap of layer-by-layer cosine similarity."""
    layers = [m.layer_name.split(".")[-1][:30] for m in result.layer_metrics]
    values = [m.cosine_similarity for m in result.layer_metrics]

    # Build y-axis labels from model pairs
    model_names = [m.name for m in result.models]
    if len(model_names) >= 2:
        from itertools import combinations

        pair_labels = [f"{a} vs {b}" for a, b in combinations(model_names, 2)]
    else:
        pair_labels = ["Cosine Similarity"]

    # Organize into rows per model pair. If layer_metrics is a flat list for a
    # single pair (the common case), wrap as one row. If there are N*(N-1)/2
    # pairs worth of data, split into rows.
    n_pairs = len(pair_labels)
    n_layers = len(layers) // n_pairs if n_pairs > 1 and len(layers) % n_pairs == 0 else len(layers)

    if n_pairs > 1 and len(values) == n_layers * n_pairs:
        z = [values[i * n_layers : (i + 1) * n_layers] for i in range(n_pairs)]
        x_labels = [m.layer_name.split(".")[-1][:30] for m in result.layer_metrics[:n_layers]]
    else:
        z = [values]
        x_labels = layers
        pair_labels = pair_labels[:1] if pair_labels else ["Cosine Similarity"]

    height = max(200, 60 * len(pair_labels) + 80)

    return {
        "data": [
            {
                "type": "heatmap",
                "z": z,
                "x": x_labels,
                "y": pair_labels,
                "colorscale": "RdYlGn",
                "zmin": 0,
                "zmax": 1,
                "colorbar": {"title": "Cosine Sim"},
            }
        ],
        "layout": {
            "title": "Layer-by-Layer Cosine Similarity",
            "height": height,
            "margin": {"t": 40, "b": 80, "l": 150, "r": 30},
            "xaxis": {"tickangle": -45, "tickfont": {"size": 8}},
        },
    }


def _build_spectral_chart(result: CompareResult) -> dict:
    """Build Plotly line chart for spectral metrics across layers."""
    layers = list(range(len(result.layer_metrics)))

    traces = []

    spec = [m.spectral_overlap for m in result.layer_metrics]
    if any(v is not None for v in spec):
        traces.append(
            {
                "type": "scatter",
                "mode": "lines+markers",
                "x": layers,
                "y": [v if v is not None else None for v in spec],
                "name": "Spectral Overlap",
                "connectgaps": True,
            }
        )

    rank = [m.effective_rank_ratio for m in result.layer_metrics]
    if any(v is not None for v in rank):
        traces.append(
            {
                "type": "scatter",
                "mode": "lines+markers",
                "x": layers,
                "y": [v if v is not None else None for v in rank],
                "name": "Rank Ratio",
                "connectgaps": True,
            }
        )

    energy = [m.task_vector_energy for m in result.layer_metrics]
    if any(v is not None for v in energy):
        traces.append(
            {
                "type": "scatter",
                "mode": "lines+markers",
                "x": layers,
                "y": [v if v is not None else None for v in energy],
                "name": "Task Vector Energy",
                "connectgaps": True,
            }
        )

    return {
        "data": traces,
        "layout": {
            "title": "Spectral Analysis Dashboard",
            "xaxis": {"title": "Layer Index"},
            "yaxis": {"title": "Score", "range": [0, 1.1]},
            "height": 400,
            "margin": {"t": 40, "b": 40, "l": 60, "r": 30},
        },
    }


def _build_conflict_chart(result: CompareResult) -> dict:
    """Build Plotly bar chart for conflict zones."""
    if not result.conflict_zones:
        return {"data": [], "layout": {"title": "No conflict zones detected"}}

    zones = result.conflict_zones
    return {
        "data": [
            {
                "type": "bar",
                "x": [f"Zone {i + 1}" for i in range(len(zones))],
                "y": [z.avg_cosine_sim for z in zones],
                "marker": {
                    "color": [_severity_color_hex(z.severity.value) for z in zones],
                },
                "text": [f"Layers {z.start_layer}-{z.end_layer}" for z in zones],
                "textposition": "auto",
            }
        ],
        "layout": {
            "title": "Conflict Zone Analysis",
            "yaxis": {"title": "Avg Cosine Similarity", "range": [0, 1]},
            "height": 300,
            "margin": {"t": 40, "b": 40, "l": 60, "r": 30},
        },
    }


def _score_color(score: float) -> str:
    if score >= 75:
        return "#4caf50"
    if score >= 55:
        return "#ff9800"
    if score >= 35:
        return "#f44336"
    return "#b71c1c"


def _severity_color_hex(severity: str) -> str:
    return {"low": "#4caf50", "medium": "#ff9800", "high": "#f44336", "critical": "#b71c1c"}.get(
        severity, "#9e9e9e"
    )
