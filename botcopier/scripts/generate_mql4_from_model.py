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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    from botcopier.models.schema import ModelParams  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    ModelParams = None  # type: ignore[assignment]

@dataclass(frozen=True)
class FeatureRuntime:
    """MQL4 runtime expression with optional helper dependencies."""

    expr: str
    helpers: tuple[str, ...] = ()


def _market_feature(mode: str, *, tracked: bool = False) -> FeatureRuntime:
    expr = f"MarketInfo(Symbol(), {mode})"
    helpers: tuple[str, ...] = ()
    if tracked:
        expr = f"MarketSeries({mode}, 0)"
        helpers = ("market_series",)
    return FeatureRuntime(expr, helpers)


def _time_feature(func: str) -> FeatureRuntime:
    return FeatureRuntime(f"{func}(TimeCurrent())")


def _time_phase(expr: str, period: int, trig: str, *, wrap: bool = False) -> FeatureRuntime:
    base = f"({expr})" if wrap else expr
    return FeatureRuntime(f"Math{trig}({base}*2*MathPi()/{period})")


# Mapping from feature name to MQL4 runtime expression.
# Add new feature mappings here as additional model features appear.
FEATURE_MAP: dict[str, FeatureRuntime] = {
    "spread": _market_feature("MODE_SPREAD", tracked=True),
    "ask": _market_feature("MODE_ASK", tracked=True),
    "bid": _market_feature("MODE_BID", tracked=True),
    "price": FeatureRuntime("iClose(Symbol(), PERIOD_CURRENT, 0)"),
    "volume": FeatureRuntime("iVolume(Symbol(), PERIOD_CURRENT, 0)"),
    "hour": _time_feature("TimeHour"),
    "dayofweek": _time_feature("TimeDayOfWeek"),
    "month": _time_feature("TimeMonth"),
    "hour_sin": _time_phase("TimeHour(TimeCurrent())", 24, "Sin"),
    "hour_cos": _time_phase("TimeHour(TimeCurrent())", 24, "Cos"),
    "dow_sin": _time_phase("TimeDayOfWeek(TimeCurrent())", 7, "Sin"),
    "dow_cos": _time_phase("TimeDayOfWeek(TimeCurrent())", 7, "Cos"),
    "month_sin": _time_phase("TimeMonth(TimeCurrent())-1", 12, "Sin", wrap=True),
    "month_cos": _time_phase("TimeMonth(TimeCurrent())-1", 12, "Cos", wrap=True),
    "dom_sin": _time_phase("TimeDay(TimeCurrent())-1", 31, "Sin", wrap=True),
    "dom_cos": _time_phase("TimeDay(TimeCurrent())-1", 31, "Cos", wrap=True),
    "sma": FeatureRuntime(
        "iMA(Symbol(), PERIOD_CURRENT, 5, 0, MODE_SMA, PRICE_CLOSE, 0)"
    ),
    "rsi": FeatureRuntime("iRSI(Symbol(), PERIOD_CURRENT, 14, PRICE_CLOSE, 0)"),
    "macd": FeatureRuntime(
        "iMACD(Symbol(), PERIOD_CURRENT, 12, 26, 9, PRICE_CLOSE, MODE_MAIN, 0)"
    ),
    "macd_signal": FeatureRuntime(
        "iMACD(Symbol(), PERIOD_CURRENT, 12, 26, 9, PRICE_CLOSE, MODE_SIGNAL, 0)"
    ),
    "bollinger_upper": FeatureRuntime(
        "iBands(Symbol(), PERIOD_CURRENT, 20, 2.0, 0, PRICE_CLOSE, MODE_UPPER, 0)"
    ),
    "bollinger_middle": FeatureRuntime(
        "iBands(Symbol(), PERIOD_CURRENT, 20, 2.0, 0, PRICE_CLOSE, MODE_MAIN, 0)"
    ),
    "bollinger_lower": FeatureRuntime(
        "iBands(Symbol(), PERIOD_CURRENT, 20, 2.0, 0, PRICE_CLOSE, MODE_LOWER, 0)"
    ),
    "atr": FeatureRuntime("iATR(Symbol(), PERIOD_CURRENT, 14, 0)"),
    "slippage": FeatureRuntime("OrderSlippage()"),
    "equity": FeatureRuntime("AccountEquity()"),
    "margin_level": FeatureRuntime("AccountMarginLevel()"),
    "event_flag": FeatureRuntime("CalendarFlag()"),
    "event_impact": FeatureRuntime("CalendarImpact()"),
    "news_sentiment": FeatureRuntime("NewsSentiment()"),
    "depth_microprice": FeatureRuntime("DepthMicroprice()", ("orderbook",)),
    "depth_vol_imbalance": FeatureRuntime(
        "DepthVolumeImbalance()", ("orderbook",)
    ),
    "depth_order_flow_imbalance": FeatureRuntime(
        "DepthOrderFlowImbalance()", ("orderbook",)
    ),
    "spread*hour_sin": FeatureRuntime(
        "MarketSeries(MODE_SPREAD, 0) * MathSin(TimeHour(TimeCurrent())*2*MathPi()/24)",
        ("market_series",),
    ),
    "spread*hour_cos": FeatureRuntime(
        "MarketSeries(MODE_SPREAD, 0) * MathCos(TimeHour(TimeCurrent())*2*MathPi()/24)",
        ("market_series",),
    ),
    "spread*spread_lag_1": FeatureRuntime(
        "MarketSeries(MODE_SPREAD, 0) * MarketSeries(MODE_SPREAD, 1)",
        ("market_series",),
    ),
    "spread*spread_lag_5": FeatureRuntime(
        "MarketSeries(MODE_SPREAD, 0) * MarketSeries(MODE_SPREAD, 5)",
        ("market_series",),
    ),
    "spread*spread_diff": FeatureRuntime(
        "MarketSeries(MODE_SPREAD, 0) * (MarketSeries(MODE_SPREAD, 0) - MarketSeries(MODE_SPREAD, 1))",
        ("market_series",),
    ),
    "hour_sin*hour_cos": FeatureRuntime(
        "MathSin(TimeHour(TimeCurrent())*2*MathPi()/24) * MathCos(TimeHour(TimeCurrent())*2*MathPi()/24)"
    ),
}


SERIES_TEMPLATES: dict[str, tuple[str, tuple[str, ...]]] = {
    "price": ("iClose(Symbol(), PERIOD_CURRENT, {shift})", ()),
    "volume": ("iVolume(Symbol(), PERIOD_CURRENT, {shift})", ()),
    "spread": ("MarketSeries(MODE_SPREAD, {shift})", ("market_series",)),
    "ask": ("MarketSeries(MODE_ASK, {shift})", ("market_series",)),
    "bid": ("MarketSeries(MODE_BID, {shift})", ("market_series",)),
}


def _series_expr(name: str, shift: int, helpers: set[str]) -> str:
    template = SERIES_TEMPLATES.get(name)
    if template is None:
        raise KeyError(name)
    expr, needed = template
    _register_helpers(helpers, needed)
    return expr.format(shift=shift)


def _series_diff(name: str, helpers: set[str]) -> str:
    template = SERIES_TEMPLATES.get(name)
    if template is None:
        raise KeyError(name)
    expr, needed = template
    _register_helpers(helpers, needed)
    return f"{expr.format(shift=0)} - {expr.format(shift=1)}"


def _load_params(path: Path) -> dict[str, Any]:
    raw = path.read_text()
    data = json.loads(raw)
    if ModelParams is not None:
        try:
            params = ModelParams.model_validate(data)
            return params.model_dump()
        except Exception:  # pragma: no cover - fallback when validation fails
            return data
    return data


HELPER_SNIPPETS: dict[str, str] = {
    "fft": """
double FftMagnitude(int freq)
{
    int bars = iBars(Symbol(), PERIOD_CURRENT);
    int N = 4;
    if(freq < 0 || freq >= N || bars <= 0)
        return 0.0;
    double real = 0.0;
    double imag = 0.0;
    for(int k = 0; k < N && k < bars; k++)
    {
        double price = iClose(Symbol(), PERIOD_CURRENT, k);
        double angle = 2.0 * MathPi() * freq * k / N;
        real += price * MathCos(angle);
        imag -= price * MathSin(angle);
    }
    return MathSqrt(real * real + imag * imag);
}

double FftPhase(int freq)
{
    int bars = iBars(Symbol(), PERIOD_CURRENT);
    int N = 4;
    if(freq < 0 || freq >= N || bars <= 0)
        return 0.0;
    double real = 0.0;
    double imag = 0.0;
    for(int k = 0; k < N && k < bars; k++)
    {
        double price = iClose(Symbol(), PERIOD_CURRENT, k);
        double angle = 2.0 * MathPi() * freq * k / N;
        real += price * MathCos(angle);
        imag -= price * MathSin(angle);
    }
    return MathArctan2(imag, real);
}
""".strip(),
    "orderbook": """
bool CollectDepthStats(double &bid_vol, double &ask_vol, double &best_bid, double &best_ask)
{
    bid_vol = 0.0;
    ask_vol = 0.0;
    best_bid = MarketInfo(Symbol(), MODE_BID);
    best_ask = MarketInfo(Symbol(), MODE_ASK);
    MqlBookInfo book[];
    if(!MarketBookGet(book))
        return false;
    double top_bid = 0.0;
    double top_ask = 0.0;
    int count = ArraySize(book);
    for(int i = 0; i < count; i++)
    {
        MqlBookInfo entry = book[i];
        if(entry.type == BOOK_TYPE_BUY)
        {
            bid_vol += entry.volume;
            if(top_bid == 0.0 || entry.price > top_bid)
                top_bid = entry.price;
        }
        else if(entry.type == BOOK_TYPE_SELL)
        {
            ask_vol += entry.volume;
            if(top_ask == 0.0 || entry.price < top_ask)
                top_ask = entry.price;
        }
    }
    if(top_bid > 0.0)
        best_bid = top_bid;
    if(top_ask > 0.0)
        best_ask = top_ask;
    return true;
}

double DepthMicroprice()
{
    double bid_vol = 0.0;
    double ask_vol = 0.0;
    double best_bid = MarketInfo(Symbol(), MODE_BID);
    double best_ask = MarketInfo(Symbol(), MODE_ASK);
    CollectDepthStats(bid_vol, ask_vol, best_bid, best_ask);
    double denom = bid_vol + ask_vol;
    if(denom <= 1e-9)
        return (best_bid + best_ask) / 2.0;
    return (best_ask * bid_vol + best_bid * ask_vol) / denom;
}

double DepthVolumeImbalance()
{
    double bid_vol = 0.0;
    double ask_vol = 0.0;
    double best_bid = 0.0;
    double best_ask = 0.0;
    CollectDepthStats(bid_vol, ask_vol, best_bid, best_ask);
    double denom = bid_vol + ask_vol;
    if(denom <= 1e-9)
        return 0.0;
    return (bid_vol - ask_vol) / denom;
}

double DepthOrderFlowImbalance()
{
    static double prev_bid_vol = 0.0;
    static double prev_ask_vol = 0.0;
    double bid_vol = 0.0;
    double ask_vol = 0.0;
    double best_bid = 0.0;
    double best_ask = 0.0;
    CollectDepthStats(bid_vol, ask_vol, best_bid, best_ask);
    double value = (bid_vol - prev_bid_vol) - (ask_vol - prev_ask_vol);
    prev_bid_vol = bid_vol;
    prev_ask_vol = ask_vol;
    return value;
}

double DepthCnnEmbedding(int idx)
{
    double bid_vol = 0.0;
    double ask_vol = 0.0;
    double best_bid = 0.0;
    double best_ask = 0.0;
    CollectDepthStats(bid_vol, ask_vol, best_bid, best_ask);
    double total = bid_vol + ask_vol;
    double imbalance = 0.0;
    if(total > 1e-9)
        imbalance = (bid_vol - ask_vol) / total;
    double features[4];
    features[0] = best_bid;
    features[1] = best_ask;
    features[2] = total;
    features[3] = imbalance;
    if(idx >= 0 && idx < 4)
        return features[idx];
    return 0.0;
}
""".strip(),
    "market_series": """
double MarketSeries(int mode, int shift)
{
    const int MAX_LAG = 32;
    static double history[3][MAX_LAG];
    static int count = 0;
    static datetime last_time = 0;
    datetime now = TimeCurrent();
    if(last_time != now || count == 0)
    {
        for(int row = 0; row < 3; row++)
        {
            for(int i = MAX_LAG - 1; i > 0; i--)
                history[row][i] = history[row][i - 1];
        }
        history[0][0] = MarketInfo(Symbol(), MODE_SPREAD);
        history[1][0] = MarketInfo(Symbol(), MODE_ASK);
        history[2][0] = MarketInfo(Symbol(), MODE_BID);
        if(count < MAX_LAG)
            count++;
        last_time = now;
    }
    if(count <= 0)
        return MarketInfo(Symbol(), mode);
    if(shift < 0)
        shift = 0;
    if(shift >= count)
        shift = count - 1;
    int idx = 0;
    if(mode == MODE_SPREAD)
        idx = 0;
    else if(mode == MODE_ASK)
        idx = 1;
    else if(mode == MODE_BID)
        idx = 2;
    else
        return MarketInfo(Symbol(), mode);
    return history[idx][shift];
}
""".strip(),
}

GET_FEATURE_TEMPLATE = """double GetFeature(int idx)\n{{\n    switch(idx)\n    {{\n{cases}\n    }}\n    return 0.0;\n}}\n"""
CASE_TEMPLATE = "    case {idx}: return {expr}; // {name}"
GET_REGIME_TEMPLATE = """double GetRegimeFeature(int idx)\n{{\n    switch(idx)\n    {{\n{cases}\n    }}\n    return 0.0;\n}}\n"""


def _register_helpers(target: set[str], helpers: Iterable[str]) -> None:
    for name in helpers:
        if name not in HELPER_SNIPPETS:
            raise KeyError(f"No helper implementation registered for '{name}'")
        target.add(name)


def _resolve_feature(name: str, helpers: set[str]) -> str:
    runtime = FEATURE_MAP.get(name)
    if runtime is not None:
        _register_helpers(helpers, runtime.helpers)
        return runtime.expr

    if "_lag_" in name:
        base, _, lag_str = name.rpartition("_lag_")
        if lag_str.isdigit():
            lag = int(lag_str)
            if lag < 0:
                raise KeyError(name)
            return _series_expr(base, lag, helpers)

    if name.endswith("_diff"):
        base = name[: -len("_diff")]
        if not base:
            raise KeyError(name)
        return _series_diff(base, helpers)

    if name.startswith("ratio_"):
        try:
            _, a, b = name.split("_", 2)
        except ValueError:
            raise KeyError(name) from None
        return (
            f'iClose("{a}", PERIOD_CURRENT, 0) / iClose("{b}", PERIOD_CURRENT, 0)'
        )

    if name.startswith("corr_"):
        try:
            _, a, b = name.split("_", 2)
        except ValueError:
            raise KeyError(name) from None
        return f'RollingCorrelation("{a}", "{b}", 5)'

    if name.startswith("graph_emb"):
        try:
            idx = int(name[len("graph_emb") :])
        except ValueError:
            raise KeyError(name) from None
        return f"GraphEmbedding({idx})"

    fft_map = {
        "fft_0_mag": "FftMagnitude(0)",
        "fft_0_phase": "FftPhase(0)",
        "fft_1_mag": "FftMagnitude(1)",
        "fft_1_phase": "FftPhase(1)",
    }
    if name in fft_map:
        _register_helpers(helpers, ["fft"])
        return fft_map[name]

    if name.startswith("depth_cnn_"):
        try:
            idx = int(name.split("_")[-1])
        except ValueError:
            raise KeyError(name) from None
        _register_helpers(helpers, ["orderbook"])
        return f"DepthCnnEmbedding({idx})"

    if name.startswith("spread_"):
        runtime = FEATURE_MAP.get("spread")
        if runtime is not None:
            _register_helpers(helpers, runtime.helpers)
            return runtime.expr

    raise KeyError(name)


def build_switch(names: Sequence[str], helpers: set[str]) -> str:
    """Render switch cases for each feature name."""

    cases: list[str] = []
    missing: list[str] = []
    for i, name in enumerate(names):
        try:
            expr = _resolve_feature(name, helpers)
        except KeyError:
            missing.append(name)
            continue
        cases.append(CASE_TEMPLATE.format(idx=i, expr=expr, name=name))
    if missing:
        names_str = ", ".join(sorted(missing))
        raise KeyError(
            "No runtime expressions for features: "
            f"{names_str}. Update StrategyTemplate.mq4 or FEATURE_MAP to add them."
        )
    return GET_FEATURE_TEMPLATE.format(cases="\n".join(cases))


def build_regime_switch(
    names: Sequence[str], feature_names: Sequence[str], helpers: set[str]
) -> str:
    """Render switch cases for regime features."""

    if not names:
        return "double GetRegimeFeature(int idx)\n{\n    return 0.0;\n}\n"

    feature_index = {name: idx for idx, name in enumerate(feature_names)}
    cases: list[str] = []
    missing: list[str] = []
    for i, name in enumerate(names):
        if name in feature_index:
            expr = f"GetFeature({feature_index[name]})"
        else:
            try:
                expr = _resolve_feature(name, helpers)
            except KeyError:
                missing.append(name)
                continue
        cases.append(CASE_TEMPLATE.format(idx=i, expr=expr, name=name))
    if missing:
        names_str = ", ".join(sorted(missing))
        raise KeyError(
            "No runtime expressions for regime features: "
            f"{names_str}. Update StrategyTemplate.mq4 or FEATURE_MAP to add them."
        )
    return GET_REGIME_TEMPLATE.format(cases="\n".join(cases))


def build_indicator_helpers(helpers: set[str]) -> str:
    if not helpers:
        return "// indicator helpers not required\n"
    parts = [HELPER_SNIPPETS[name] for name in sorted(helpers)]
    return "\n\n".join(parts) + "\n"


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
        lines.append(f"double g_threshold_{name} = {params.get('threshold', 0.5)};")
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
            lines.append(f"double g_emb_{sym}[] = {{{arr}}};")
        lines.append("double GraphEmbedding(int idx)\n{")
        lines.append("    string s = Symbol();")
        for sym in emb_map.keys():
            lines.append(
                f'    if(s == "{sym}" && idx < ArraySize(g_emb_{sym})) return g_emb_{sym}[idx];'
            )
        lines.append("    return 0.0;")
        lines.append("}")
    else:
        lines.append("double GraphEmbedding(int idx)\n{\n    return 0.0;\n}")
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
        "double g_embed_bias[] = {"
        + ", ".join(_flt(x) for x in w.get("embed_bias", []))
        + "};"
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
    data = _load_params(model)
    # If a distilled student exists and no explicit models are provided, expose
    # it under the ``models`` key so the template receives logistic coefficients.
    if "distilled" in data and not data.get("models"):
        data["models"] = {"logreg": data["distilled"]}

    feature_names = data.get("retained_features") or data.get("feature_names", [])
    helpers: set[str] = set()
    get_feature = build_switch(feature_names, helpers)
    gating = data.get("regime_gating") or {}
    regime_features = (
        gating.get("feature_names")
        or data.get("regime_features")
        or []
    )
    get_regime = build_regime_switch(regime_features, feature_names, helpers)
    session_models = _build_session_models(data)
    symbol_emb = _build_symbol_embeddings(data.get("symbol_embeddings", {}))
    symbol_thresh = _build_symbol_thresholds(data.get("symbol_thresholds", {}))
    transformer_block = _build_transformer_params(data)
    indicator_block = build_indicator_helpers(helpers)
    content = template.read_text()
    output = content.replace("// __GET_FEATURE__", get_feature)
    output = output.replace("// __GET_REGIME_FEATURE__", get_regime)
    output = output.replace("// __SESSION_MODELS__", session_models)
    output = output.replace("// __INDICATOR_FUNCTIONS__", indicator_block)
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
        output = output.replace(
            "bool g_use_transformer = false;", "bool g_use_transformer = true;"
        )
    if (
        data.get("model_type") == "moe"
        and regime_features
        and data.get("experts")
        and gating.get("weights")
    ):
        experts = data.get("experts") or []
        n_experts = len(experts)
        expert_dim = len(experts[0].get("weights", [])) if experts else 0
        regime_dim = len(regime_features)

        def _fmt(values: Sequence[float]) -> str:
            return ", ".join(str(float(v)) for v in values)

        expert_weights = [
            float(w)
            for exp in experts
            for w in exp.get("weights", [])
        ]
        expert_bias = [float(exp.get("bias", 0.0)) for exp in experts]
        gate_weights = gating.get("weights", [])
        if gate_weights and isinstance(gate_weights[0], list):
            gate_flat = [float(v) for row in gate_weights for v in row]
        else:
            gate_flat = [float(v) for v in gate_weights]
        gate_bias = [float(v) for v in gating.get("bias", [])]
        output = output.replace("bool g_use_moe = false;", "bool g_use_moe = true;")
        output = output.replace(
            "int g_moe_num_experts = 0;",
            f"int g_moe_num_experts = {n_experts};",
        )
        output = output.replace(
            "int g_moe_feature_dim = 0;",
            f"int g_moe_feature_dim = {expert_dim};",
        )
        output = output.replace(
            "int g_moe_regime_dim = 0;",
            f"int g_moe_regime_dim = {regime_dim};",
        )
        weight_len = max(len(expert_weights), 1)
        gate_len = max(len(gate_flat), 1)
        bias_len = max(len(expert_bias), 1)
        gate_bias_len = max(len(gate_bias), 1)
        output = output.replace(
            "double g_moe_expert_weights[1] = {0.0};",
            f"double g_moe_expert_weights[{weight_len}] = {{{_fmt(expert_weights) if expert_weights else '0.0'}}};",
        )
        output = output.replace(
            "double g_moe_expert_bias[1] = {0.0};",
            f"double g_moe_expert_bias[{bias_len}] = {{{_fmt(expert_bias) if expert_bias else '0.0'}}};",
        )
        output = output.replace(
            "double g_moe_gate_weights[1] = {0.0};",
            f"double g_moe_gate_weights[{gate_len}] = {{{_fmt(gate_flat) if gate_flat else '0.0'}}};",
        )
        output = output.replace(
            "double g_moe_gate_bias[1] = {0.0};",
            f"double g_moe_gate_bias[{gate_bias_len}] = {{{_fmt(gate_bias) if gate_bias else '0.0'}}};",
        )
    cal_path = str(calendar_file) if calendar_file else "calendar.csv"
    output = output.replace("__CALENDAR_FILE__", cal_path)
    template.write_text(output)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model", type=Path, default=Path("model.json"), help="Path to model.json"
    )
    p.add_argument(
        "--template",
        type=Path,
        default=Path("StrategyTemplate.mq4"),
        help="Template .mq4 file",
    )
    p.add_argument("--calendar-file", type=Path, help="CSV file with calendar events")
    args = p.parse_args()
    insert_get_feature(args.model, args.template, calendar_file=args.calendar_file)


if __name__ == "__main__":
    main()
