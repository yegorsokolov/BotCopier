"""Utilities for generating simple Markdown model cards."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

try:  # pragma: no cover - optional dependency
    from jinja2 import Environment, select_autoescape

    _HAS_JINJA = True
except Exception:  # pragma: no cover - fallback when jinja2 is unavailable
    Environment = None  # type: ignore[assignment]
    select_autoescape = None  # type: ignore[assignment]
    _HAS_JINJA = False

from botcopier.models.schema import ModelParams

_TEMPLATE = """
# Model Card

## Model Parameters
- **Version:** {{ params.version }}
- **Features:** {{ params.feature_names | join(', ') }}
{% if params.metadata %}
## Metadata
{% for key, value in params.metadata.items() %}
- **{{ key }}:** {{ value }}
{% endfor %}
{% endif %}

## Metrics
{% for key, value in metrics.items() %}- **{{ key }}:** {{ '%.4f' | format(value) if value is not none else 'N/A' }}
{% endfor %}
{% if dependencies_path %}

## Environment
See [dependencies]({{ dependencies_path }}) for the exact package versions.
{% endif %}
"""


def generate_model_card(
    model_params: ModelParams,
    metrics: Mapping[str, object],
    output_path: Path,
    *,
    dependencies_path: Path | None = None,
) -> None:
    """Render and write a simple model card.

    Parameters
    ----------
    model_params:
        Parameters describing the trained model.
    metrics:
        Aggregated evaluation metrics.
    output_path:
        Location to write the rendered markdown file.
    """
    output_path = Path(output_path)
    dep_ref = dependencies_path.name if dependencies_path else None

    if _HAS_JINJA:
        env = Environment(autoescape=select_autoescape())
        template = env.from_string(_TEMPLATE)
        content = template.render(
            params=model_params, metrics=metrics, dependencies_path=dep_ref
        )
    else:  # pragma: no cover - exercised when optional dependency missing
        feature_list = ", ".join(model_params.feature_names)
        sections = ["# Model Card", "", "## Model Parameters"]
        sections.append(f"- **Version:** {model_params.version}")
        sections.append(f"- **Features:** {feature_list}")
        metadata = getattr(model_params, "metadata", None) or {}
        if metadata:
            sections.append("")
            sections.append("## Metadata")
            for key, value in metadata.items():
                sections.append(f"- **{key}:** {value}")
        if metrics:
            sections.append("")
            sections.append("## Metrics")
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    sections.append(f"- **{key}:** {value:.4f}")
                else:
                    sections.append(f"- **{key}:** {value}")
        if dep_ref:
            sections.extend(
                [
                    "",
                    "## Environment",
                    f"See [dependencies]({dep_ref}) for the exact package versions.",
                ]
            )
        content = "\n".join(sections) + "\n"

    output_path.write_text(content)
