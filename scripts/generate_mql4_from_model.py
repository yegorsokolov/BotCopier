#!/usr/bin/env python3
"""Render MQL4 strategy file from model description."""
import argparse
import json
from pathlib import Path


def _fmt(value: float) -> str:
    """Format floats for insertion into MQL4 code."""
    return f"{value:.6g}"

template_path = (
    Path(__file__).resolve().parent.parent / 'experts' / 'StrategyTemplate.mq4'
)


def generate(model_json: Path, out_dir: Path):
    with open(model_json) as f:
        model = json.load(f)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(template_path) as f:
        template = f.read()

    output = template.replace(
        'MagicNumber = 1234',
        f"MagicNumber = {model.get('magic', 9999)}",
    )

    coeffs = model.get('coefficients') or model.get('coef_vector', [])
    coeff_str = ', '.join(_fmt(c) for c in coeffs)
    output = output.replace('__COEFFICIENTS__', coeff_str)

    prob_table = model.get('probability_table', [])
    prob_str = ', '.join(_fmt(p) for p in prob_table)
    output = output.replace('__PROBABILITY_TABLE__', prob_str)

    feature_names = model.get('feature_names', [])

    feature_map = {
        'hour': 'TimeHour(TimeCurrent())',
        'spread': 'MarketInfo(SymbolToTrade, MODE_SPREAD)',
        'lots': 'Lots',
        'sl_dist': 'GetSLDistance()',
        'tp_dist': 'GetTPDistance()',
        'sma': 'iMA(SymbolToTrade, 0, 5, 0, MODE_SMA, PRICE_CLOSE, 0)',
        'rsi': 'iRSI(SymbolToTrade, 0, 14, PRICE_CLOSE, 0)',
        'macd': 'iMACD(SymbolToTrade, 0, 12, 26, 9, PRICE_CLOSE, MODE_MAIN, 0)',
        'macd_signal': 'iMACD(SymbolToTrade, 0, 12, 26, 9, PRICE_CLOSE, MODE_SIGNAL, 0)',
        'volatility': 'iStdDev(SymbolToTrade, 0, 20, 0, MODE_SMA, PRICE_CLOSE, 0)',
    }

    cases = []
    for idx, name in enumerate(feature_names):
        expr = feature_map.get(name, '0.0')
        cases.append(f"      case {idx}:\n         return({expr});")
    case_block = "\n".join(cases)
    if case_block:
        case_block += "\n"
    output = output.replace('__FEATURE_CASES__', case_block)

    intercept = model.get('intercept', 0.0)
    output = output.replace('__INTERCEPT__', _fmt(intercept))

    threshold = model.get('threshold', 0.5)
    output = output.replace('__THRESHOLD__', _fmt(threshold))
    out_file = out_dir / f"Generated_{model.get('model_id', 'model')}.mq4"
    with open(out_file, 'w') as f:
        f.write(output)
    print(f"Strategy written to {out_file}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('model_json')
    p.add_argument('out_dir')
    args = p.parse_args()
    generate(Path(args.model_json), Path(args.out_dir))


if __name__ == '__main__':
    main()
