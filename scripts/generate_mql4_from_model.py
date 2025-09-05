"""Generate MQL4 expert advisor snippets from model.json.

This script reads ``feature_names`` from a model JSON file and renders a
``GetFeature`` function containing a ``switch`` statement mapping each feature
index to a runtime MQL4 expression.  The generated function is inserted into a
strategy template file in place, allowing the resulting ``.mq4`` file to be
compiled directly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

# Mapping from feature name to MQL4 runtime expression.
FEATURE_MAP: dict[str, str] = {
    "spread": "MarketInfo(Symbol(), MODE_SPREAD)",
    "ask": "MarketInfo(Symbol(), MODE_ASK)",
    "bid": "MarketInfo(Symbol(), MODE_BID)",
    "hour": "TimeHour(TimeCurrent())",
}

GET_FEATURE_TEMPLATE = """double GetFeature(int idx)\n{{\n    switch(idx)\n    {{\n{cases}\n    }}\n    return 0.0;\n}}\n"""
CASE_TEMPLATE = "    case {idx}: return {expr}; // {name}"


def build_switch(names: Sequence[str]) -> str:
    """Render switch cases for each feature name."""
    cases = []
    for i, name in enumerate(names):
        expr = FEATURE_MAP.get(name, "0.0")
        cases.append(CASE_TEMPLATE.format(idx=i, expr=expr, name=name))
    return GET_FEATURE_TEMPLATE.format(cases="\n".join(cases))


def _build_session_models(data: dict) -> str:
    sessions = data.get("session_models", {})
    lines: list[str] = []
    for name, params in sessions.items():
        coeffs = [params.get("intercept", 0.0)] + params.get("coefficients", [])
        coeff_str = ", ".join(f"{c}" for c in coeffs)
        lines.append(f"double g_coeffs_{name}[] = {{{coeff_str}}};")
        lines.append(
            f"double g_threshold_{name} = {params.get('threshold', 0.5)};"
        )
    return "\n".join(lines)


def insert_get_feature(model: Path, template: Path) -> None:
    """Insert generated GetFeature and session models into ``template``."""
    data = json.loads(model.read_text())
    feature_names = data.get("feature_names", [])
    get_feature = build_switch(feature_names)
    session_models = _build_session_models(data)
    content = template.read_text()
    output = content.replace("// __GET_FEATURE__", get_feature)
    output = output.replace("// __SESSION_MODELS__", session_models)
    template.write_text(output)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=Path("model.json"), help="Path to model.json")
    p.add_argument(
        "--template", type=Path, default=Path("StrategyTemplate.mq4"), help="Template .mq4 file"
    )
    args = p.parse_args()
    insert_get_feature(args.model, args.template)


if __name__ == "__main__":
    main()
