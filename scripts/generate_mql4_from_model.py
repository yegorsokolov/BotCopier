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
# Add new feature mappings here as additional model features appear.
FEATURE_MAP: dict[str, str] = {
    "spread": "MarketInfo(Symbol(), MODE_SPREAD)",
    "ask": "MarketInfo(Symbol(), MODE_ASK)",
    "bid": "MarketInfo(Symbol(), MODE_BID)",
    "hour_sin": "MathSin(TimeHour(TimeCurrent())*2*MathPi()/24)",
    "hour_cos": "MathCos(TimeHour(TimeCurrent())*2*MathPi()/24)",
    "dow_sin": "MathSin(TimeDayOfWeek(TimeCurrent())*2*MathPi()/7)",
    "dow_cos": "MathCos(TimeDayOfWeek(TimeCurrent())*2*MathPi()/7)",
    "volume": "iVolume(Symbol(), PERIOD_CURRENT, 0)",
    "slippage": "OrderSlippage()",
    "equity": "AccountEquity()",
    "margin_level": "AccountMarginLevel()",
}

GET_FEATURE_TEMPLATE = """double GetFeature(int idx)\n{{\n    switch(idx)\n    {{\n{cases}\n    }}\n    return 0.0;\n}}\n"""
CASE_TEMPLATE = "    case {idx}: return {expr}; // {name}"


def build_switch(names: Sequence[str]) -> str:
    """Render switch cases for each feature name.

    Raises:
        KeyError: If a feature name is missing from ``FEATURE_MAP``.
    """

    cases = []
    for i, name in enumerate(names):
        if name in FEATURE_MAP:
            expr = FEATURE_MAP[name]
        elif name.startswith("ratio_"):
            try:
                _, a, b = name.split("_", 2)
            except ValueError:
                raise KeyError(f"Invalid ratio feature name '{name}'") from None
            expr = f'iClose("{a}", PERIOD_CURRENT, 0) / iClose("{b}", PERIOD_CURRENT, 0)'
        elif name.startswith("corr_"):
            try:
                _, a, b = name.split("_", 2)
            except ValueError:
                raise KeyError(f"Invalid corr feature name '{name}'") from None
            expr = f'RollingCorrelation("{a}", "{b}", 5)'
        elif name.startswith("graph_emb"):
            try:
                idx = int(name[len("graph_emb"):])
            except ValueError:
                raise KeyError(f"Invalid graph embedding feature name '{name}'") from None
            expr = f"GraphEmbedding({idx})"
        else:
            raise KeyError(
                "No runtime expression for feature "
                f"'{name}'. Update StrategyTemplate.mq4 or FEATURE_MAP to add it."
            )
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
        mean = params.get("feature_mean", [])
        std = params.get("feature_std", [])
        mean_str = ", ".join(f"{m}" for m in mean)
        std_str = ", ".join(f"{s}" for s in std)
        lines.append(f"double g_feature_mean_{name}[] = {{{mean_str}}};")
        lines.append(f"double g_feature_std_{name}[] = {{{std_str}}};")

        def _coeff_line(prefix: str, model: dict | None) -> None:
            if model:
                arr = [model.get("intercept", 0.0)] + model.get("coefficients", [])
            else:
                arr = [0.0]
            arr_str = ", ".join(f"{c}" for c in arr)
            lines.append(f"double g_{prefix}_{name}[] = {{{arr_str}}};")

        _coeff_line("lot_coeffs", params.get("lot_model"))
        _coeff_line("sl_coeffs", params.get("sl_model"))
        _coeff_line("tp_coeffs", params.get("tp_model"))
        lines.append(
            f"double g_conformal_lower_{name} = {params.get('conformal_lower', 0.0)};"
        )
        lines.append(
            f"double g_conformal_upper_{name} = {params.get('conformal_upper', 1.0)};"
        )
    return "\n".join(lines)


def _build_symbol_embeddings(emb_map: dict) -> str:
    """Generate MQL4 arrays and lookup function for symbol embeddings."""

    if not emb_map:
        return ""
    lines: list[str] = []
    for sym, vec in emb_map.items():
        arr = ", ".join(str(v) for v in vec)
        lines.append(f'double g_emb_{sym}[] = {{{arr}}};')
    lines.append("double GraphEmbedding(int idx)\n{")
    lines.append("    string s = Symbol();")
    for sym in emb_map.keys():
        lines.append(f'    if(s == "{sym}") return g_emb_{sym}[idx];')
    lines.append("    return 0.0;")
    lines.append("}")
    return "\n".join(lines)


def insert_get_feature(model: Path, template: Path) -> None:
    """Insert generated GetFeature and session models into ``template``."""
    data = json.loads(model.read_text())
    feature_names = data.get("feature_names", [])
    get_feature = build_switch(feature_names)
    session_models = _build_session_models(data)
    symbol_emb = _build_symbol_embeddings(data.get("symbol_embeddings", {}))
    content = template.read_text()
    output = content.replace("// __GET_FEATURE__", get_feature)
    output = output.replace("// __SESSION_MODELS__", session_models)
    output = output.replace("// __SYMBOL_EMBEDDINGS__", symbol_emb)
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
