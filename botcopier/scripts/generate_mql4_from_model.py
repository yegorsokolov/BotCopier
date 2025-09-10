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
import re
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
    "month_sin": "MathSin((TimeMonth(TimeCurrent())-1)*2*MathPi()/12)",
    "month_cos": "MathCos((TimeMonth(TimeCurrent())-1)*2*MathPi()/12)",
    "dom_sin": "MathSin((TimeDay(TimeCurrent())-1)*2*MathPi()/31)",
    "dom_cos": "MathCos((TimeDay(TimeCurrent())-1)*2*MathPi()/31)",
    "volume": "iVolume(Symbol(), PERIOD_CURRENT, 0)",
    "slippage": "OrderSlippage()",
    "equity": "AccountEquity()",
    "margin_level": "AccountMarginLevel()",
    "atr": "iATR(Symbol(), PERIOD_CURRENT, 14, 0)",
    "event_flag": "CalendarFlag()",
    "event_impact": "CalendarImpact()",
    "news_sentiment": "NewsSentiment()",
    "spread_lag_1": "MarketInfo(Symbol(), MODE_SPREAD)",
    "spread*hour_sin": "MarketInfo(Symbol(), MODE_SPREAD) * MathSin(TimeHour(TimeCurrent())*2*MathPi()/24)",
    "spread*hour_cos": "MarketInfo(Symbol(), MODE_SPREAD) * MathCos(TimeHour(TimeCurrent())*2*MathPi()/24)",
    "spread*spread_lag_1": "MarketInfo(Symbol(), MODE_SPREAD) * MarketInfo(Symbol(), MODE_SPREAD)",
    "spread*spread_lag_5": "MarketInfo(Symbol(), MODE_SPREAD) * MarketInfo(Symbol(), MODE_SPREAD)",
    "spread*spread_diff": "MarketInfo(Symbol(), MODE_SPREAD) * MarketInfo(Symbol(), MODE_SPREAD)",
    "hour_sin*hour_cos": "MathSin(TimeHour(TimeCurrent())*2*MathPi()/24) * MathCos(TimeHour(TimeCurrent())*2*MathPi()/24)",
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
        elif name.startswith("spread_"):
            expr = FEATURE_MAP.get("spread", "0")
        else:
            raise KeyError(
                "No runtime expression for feature "
                f"'{name}'. Update StrategyTemplate.mq4 or FEATURE_MAP to add it."
            )
        cases.append(CASE_TEMPLATE.format(idx=i, expr=expr, name=name))
    return GET_FEATURE_TEMPLATE.format(cases="\n".join(cases))


def _build_session_models(data: dict) -> str:
    """Generate arrays for individual models and the ensemble router."""

    lines: list[str] = []

    # ``model.json`` produced by training historically stored per-algorithm
    # parameters under ``models``.  Newer pipelines may instead emit
    # ``session_models`` keyed by trading session.  Accept either key so the
    # generator remains backward compatible.
    models = data.get("models") or data.get("session_models") or {}
    for name, params in models.items():
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

    router = data.get("ensemble_router") or {}
    if router:
        intercept = ", ".join(str(v) for v in router.get("intercept", []))
        coeffs = ", ".join(
            str(v) for row in router.get("coefficients", []) for v in row
        )
        mean = ", ".join(str(v) for v in router.get("feature_mean", []))
        std = ", ".join(str(v) for v in router.get("feature_std", []))
        lines.append(f"double g_router_intercept[] = {{{intercept}}};")
        lines.append(f"double g_router_coeffs[] = {{{coeffs}}};")
        lines.append(f"double g_router_feature_mean[] = {{{mean}}};")
        lines.append(f"double g_router_feature_std[] = {{{std}}};")
    else:
        # Provide default zeroed router arrays so the template compiles even
        # without trained gating weights.
        lines.append("double g_router_intercept[1] = {0.0};")
        lines.append("double g_router_coeffs[2] = {0.0, 0.0};")
        lines.append("double g_router_feature_mean[2] = {0.0, 0.0};")
        lines.append("double g_router_feature_std[2] = {1.0, 1.0};")

    return "\n".join(lines)


def _build_symbol_embeddings(emb_map: dict) -> str:
    """Generate MQL4 arrays and lookup function for symbol embeddings."""

    lines: list[str] = []
    if emb_map:
        for sym, vec in emb_map.items():
            arr = ", ".join(str(v) for v in vec)
            lines.append(f'double g_emb_{sym}[] = {{{arr}}};')
        lines.append("double GraphEmbedding(int idx)\n{")
        lines.append("    string s = Symbol();")
        for sym in emb_map.keys():
            lines.append(
                f'    if(s == "{sym}" && idx < ArraySize(g_emb_{sym})) return g_emb_{sym}[idx];'
            )
        lines.append("    return 0.0;")
        lines.append("}")
    else:
        lines.append(
            "double GraphEmbedding(int idx)\n{\n    return 0.0;\n}"
        )
    return "\n".join(lines)


def _build_symbol_thresholds(thresh_map: dict) -> str:
    """Generate function for per-symbol probability thresholds."""

    lines: list[str] = []
    lines.append("double SymbolThreshold()\n{")
    lines.append("    string s = Symbol();")
    for sym, thresh in thresh_map.items():
        lines.append(f'    if(s == "{sym}") return {thresh};')
    lines.append("    return g_threshold;")
    lines.append("}")
    return "\n".join(lines)

def _build_transformer_params(data: dict) -> str:
    block_start = "// __TRANSFORMER_PARAMS_START__"
    block_end = "// __TRANSFORMER_PARAMS_END__"
    if data.get("model_type") != "transformer":
        return (
            f"{block_start}\n"
            "int g_transformer_window = 0;\n"
            "int g_transformer_dim = 0;\n"
            "int g_transformer_feat_dim = 0;\n"
            "double g_tfeature_mean[1] = {0.0};\n"
            "double g_tfeature_std[1] = {1.0};\n"
            "double g_embed_weight[1] = {0.0};\n"
            "double g_embed_bias[1] = {0.0};\n"
            "double g_q_weight[1] = {0.0};\n"
            "double g_q_bias[1] = {0.0};\n"
            "double g_k_weight[1] = {0.0};\n"
            "double g_k_bias[1] = {0.0};\n"
            "double g_v_weight[1] = {0.0};\n"
            "double g_v_bias[1] = {0.0};\n"
            "double g_out_weight[1] = {0.0};\n"
            "double g_out_bias = 0.0;\n"
            f"{block_end}"
        )

    w = data.get("weights", {})

    def _flt(val: float) -> str:
        return f"{val}"

    def _flat(mat):
        return [v for row in mat for v in row]

    lines = [block_start]
    lines.append(f"int g_transformer_window = {data.get('window_size', 0)};")
    dim = len(w.get("embed_bias", []))
    feat_dim = len(w.get("embed_weight", [[]])[0]) if w.get("embed_weight") else 0
    lines.append(f"int g_transformer_dim = {dim};")
    lines.append(f"int g_transformer_feat_dim = {feat_dim};")
    lines.append(
        "double g_tfeature_mean[] = {"
        + ", ".join(_flt(x) for x in data.get("feature_mean", []))
        + "};"
    )
    lines.append(
        "double g_tfeature_std[] = {"
        + ", ".join(_flt(x) for x in data.get("feature_std", []))
        + "};"
    )
    lines.append(
        "double g_embed_weight[] = {"
        + ", ".join(_flt(x) for x in _flat(w.get("embed_weight", [])))
        + "};"
    )
    lines.append(
        "double g_embed_bias[] = {" + ", ".join(_flt(x) for x in w.get("embed_bias", [])) + "};"
    )
    lines.append(
        "double g_q_weight[] = {"
        + ", ".join(_flt(x) for x in _flat(w.get("q_weight", [])))
        + "};"
    )
    lines.append(
        "double g_q_bias[] = {" + ", ".join(_flt(x) for x in w.get("q_bias", [])) + "};"
    )
    lines.append(
        "double g_k_weight[] = {"
        + ", ".join(_flt(x) for x in _flat(w.get("k_weight", [])))
        + "};"
    )
    lines.append(
        "double g_k_bias[] = {" + ", ".join(_flt(x) for x in w.get("k_bias", [])) + "};"
    )
    lines.append(
        "double g_v_weight[] = {"
        + ", ".join(_flt(x) for x in _flat(w.get("v_weight", [])))
        + "};"
    )
    lines.append(
        "double g_v_bias[] = {" + ", ".join(_flt(x) for x in w.get("v_bias", [])) + "};"
    )
    lines.append(
        "double g_out_weight[] = {"
        + ", ".join(_flt(x) for x in w.get("out_weight", []))
        + "};"
    )
    out_bias = w.get("out_bias", [0.0])
    if isinstance(out_bias, list):
        out_bias = out_bias[0]
    lines.append(f"double g_out_bias = {out_bias};")
    lines.append(block_end)
    return "\n".join(lines)


def insert_get_feature(
    model: Path, template: Path, calendar_file: Path | None = None
) -> None:
    """Insert generated GetFeature and session models into ``template``."""
    data = json.loads(model.read_text())
    # If a distilled student exists and no explicit models are provided, expose
    # it under the ``models`` key so the template receives logistic coefficients.
    if "distilled" in data and not data.get("models"):
        data["models"] = {"logreg": data["distilled"]}

    feature_names = data.get("retained_features") or data.get("feature_names", [])
    get_feature = build_switch(feature_names)
    session_models = _build_session_models(data)
    symbol_emb = _build_symbol_embeddings(data.get("symbol_embeddings", {}))
    symbol_thresh = _build_symbol_thresholds(data.get("symbol_thresholds", {}))
    transformer_block = _build_transformer_params(data)
    content = template.read_text()
    output = content.replace("// __GET_FEATURE__", get_feature)
    output = output.replace("// __SESSION_MODELS__", session_models)
    pattern_emb = re.compile(
        r"// __SYMBOL_EMBEDDINGS_START__.*// __SYMBOL_EMBEDDINGS_END__",
        re.DOTALL,
    )
    output = re.sub(pattern_emb, symbol_emb, output)
    pattern_thresh = re.compile(
        r"// __SYMBOL_THRESHOLDS_START__.*// __SYMBOL_THRESHOLDS_END__",
        re.DOTALL,
    )
    output = re.sub(pattern_thresh, symbol_thresh, output)
    pattern = re.compile(
        r"// __TRANSFORMER_PARAMS_START__.*// __TRANSFORMER_PARAMS_END__",
        re.DOTALL,
    )
    output = re.sub(pattern, transformer_block, output)
    if data.get("model_type") == "transformer" and not data.get("distilled"):
        output = output.replace("bool g_use_transformer = false;", "bool g_use_transformer = true;")
    cal_path = str(calendar_file) if calendar_file else "calendar.csv"
    output = output.replace("__CALENDAR_FILE__", cal_path)
    template.write_text(output)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=Path("model.json"), help="Path to model.json")
    p.add_argument(
        "--template", type=Path, default=Path("StrategyTemplate.mq4"), help="Template .mq4 file"
    )
    p.add_argument("--calendar-file", type=Path, help="CSV file with calendar events")
    args = p.parse_args()
    insert_get_feature(args.model, args.template, calendar_file=args.calendar_file)


if __name__ == "__main__":
    main()
