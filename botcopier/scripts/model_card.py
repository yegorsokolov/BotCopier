"""Utilities for generating simple Markdown model cards."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

from jinja2 import Environment, select_autoescape

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
    env = Environment(autoescape=select_autoescape())
    template = env.from_string(_TEMPLATE)
    dep_ref = dependencies_path.name if dependencies_path else None
    content = template.render(
        params=model_params, metrics=metrics, dependencies_path=dep_ref
    )
    Path(output_path).write_text(content)
