import argparse
import json
from pathlib import Path

FEATURE_MAP = {
    "spread": "MarketInfo(Symbol(), MODE_SPREAD)",
    "slippage": "OrderSlippage()",
    "equity": "AccountEquity()",
    "margin_level": "AccountMarginLevel()",
    "volume": "iVolume(Symbol(), PERIOD_CURRENT, 0)",
    "hour_sin": "MathSin(TimeHour(TimeCurrent())*2*MathPi()/24)",
    "hour_cos": "MathCos(TimeHour(TimeCurrent())*2*MathPi()/24)",
    "month_sin": "MathSin((TimeMonth(TimeCurrent())-1)*2*MathPi()/12)",
    "month_cos": "MathCos((TimeMonth(TimeCurrent())-1)*2*MathPi()/12)",
    "dom_sin": "MathSin((TimeDay(TimeCurrent())-1)*2*MathPi()/31)",
    "dom_cos": "MathCos((TimeDay(TimeCurrent())-1)*2*MathPi()/31)",
    "atr": "iATR(Symbol(), PERIOD_CURRENT, 14, 0)",
    "event_flag": "CalendarFlag()",
    "event_impact": "CalendarImpact()",
}


def _emit_features(model: dict) -> tuple[str, list[str]]:
    features = model.get("retained_features") or model.get("feature_names") or []
    lines = ["double GetFeature(int idx)", "{", "    switch(idx)", "    {"]
    emitted = []
    for idx, name in enumerate(features):
        if name not in FEATURE_MAP:
            raise ValueError(f"Update StrategyTemplate.mq4 to map feature {name}")
        lines.append(
            f"        case {idx}: return {FEATURE_MAP[name]}; // {name}"
        )
        emitted.append(name)
    lines.extend([
        "    }",
        "    return 0.0;",
        "}",
    ])
    return "\n".join(lines), emitted


def _emit_models(model: dict) -> str:
    models = model.get("models", {})
    out_lines = []
    for name, params in models.items():
        coeffs = [params.get("intercept", 0.0)] + params.get("coefficients", [])
        fmt = ", ".join(f"{c}" for c in coeffs)
        out_lines.append(f"double g_coeffs_{name}[] = {{{fmt}}};")
        lot = ", ".join(str(c) for c in params.get("lot_coefficients", [0.0]))
        out_lines.append(f"double g_lot_coeffs_{name}[] = {{{lot}}};")
        sl = ", ".join(str(c) for c in params.get("sl_coefficients", [0.0]))
        out_lines.append(f"double g_sl_coeffs_{name}[] = {{{sl}}};")
        tp = ", ".join(str(c) for c in params.get("tp_coefficients", [0.0]))
        out_lines.append(f"double g_tp_coeffs_{name}[] = {{{tp}}};")
        out_lines.append(
            f"double g_threshold_{name} = {params.get('threshold', 0.5)};"
        )
        fm = ", ".join(str(c) for c in params.get("feature_mean", []))
        out_lines.append(f"double g_feature_mean_{name}[] = {{{fm}}};")
        fs = ", ".join(str(c) for c in params.get("feature_std", []))
        out_lines.append(f"double g_feature_std_{name}[] = {{{fs}}};")
        out_lines.append(
            f"double g_conformal_lower_{name} = {params.get('conformal_lower', 0.0)};"
        )
        out_lines.append(
            f"double g_conformal_upper_{name} = {params.get('conformal_upper', 1.0)};"
        )
    router = model.get("ensemble_router")
    if router:
        inter = ", ".join(str(x) for x in router.get("intercept", []))
        out_lines.append(f"double g_router_intercept[] = {{{inter}}};")
        coeffs = router.get("coefficients", [])
        if coeffs:
            cols = len(coeffs[0])
            rows = ["{ " + ", ".join(str(v) for v in row) + " }" for row in coeffs]
            out_lines.append(
                f"double g_router_coeffs[][ {cols} ] = {{{', '.join(rows)}}};"
            )
        fm = ", ".join(str(x) for x in router.get("feature_mean", []))
        out_lines.append(f"double g_router_feature_mean[] = {{{fm}}};")
        fs = ", ".join(str(x) for x in router.get("feature_std", []))
        out_lines.append(f"double g_router_feature_std[] = {{{fs}}};")
    return "\n".join(out_lines)


def _emit_symbol_thresholds(model: dict) -> str:
    thrs = model.get("symbol_thresholds")
    if not thrs:
        return "double SymbolThreshold()\n{\n    return g_threshold;\n}"
    lines = ["double SymbolThreshold()", "{", "    string s = Symbol();"]
    for sym, thr in thrs.items():
        lines.append(f'    if(s == "{sym}") return {thr};')
    lines.append("    return g_threshold;")
    lines.append("}")
    return "\n".join(lines)


def generate(model_path: Path, template_path: Path, out_path: Path, calendar_file: str | None = None) -> None:
    model = json.loads(model_path.read_text())
    features_block, emitted = _emit_features(model)
    model["feature_names"] = emitted
    if "retained_features" in model:
        del model["retained_features"]
    model_path.write_text(json.dumps(model))

    models_block = _emit_models(model)
    thr_block = _emit_symbol_thresholds(model)

    content = template_path.read_text()
    content = content.replace("// __GET_FEATURE__", features_block)
    content = content.replace("// __SESSION_MODELS__", models_block)
    content = content.replace(
        "// __SYMBOL_THRESHOLDS_START__\ndouble SymbolThreshold()\n{\n    return g_threshold;\n}\n// __SYMBOL_THRESHOLDS_END__",
        f"// __SYMBOL_THRESHOLDS_START__\n{thr_block}\n// __SYMBOL_THRESHOLDS_END__",
    )
    if calendar_file:
        content = content.replace("__CALENDAR_FILE__", str(calendar_file))
    out_path.write_text(content)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--calendar-file")
    args = parser.parse_args()
    out = args.out or args.template
    generate(args.model, args.template, out, args.calendar_file)


if __name__ == "__main__":
    main()
