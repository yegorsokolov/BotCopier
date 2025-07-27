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

    nn_weights = model.get('nn_weights', [])
    if nn_weights:
        l1_w = ', '.join(_fmt(v) for row in nn_weights[0] for v in row)
        l1_b = ', '.join(_fmt(v) for v in nn_weights[1])
        l2_w = ', '.join(_fmt(v) for row in nn_weights[2] for v in (row if isinstance(row, list) else [row]))
        l2_b = _fmt(nn_weights[3][0] if isinstance(nn_weights[3], list) else nn_weights[3])
        hidden_size = len(nn_weights[1])
    else:
        l1_w = l1_b = l2_w = ''
        l2_b = '0'
        hidden_size = 0
    output = output.replace('__NN_L1_WEIGHTS__', l1_w)
    output = output.replace('__NN_L1_BIAS__', l1_b)
    output = output.replace('__NN_L2_WEIGHTS__', l2_w)
    output = output.replace('__NN_L2_BIAS__', l2_b)
    output = output.replace('__NN_HIDDEN_SIZE__', str(hidden_size))

    lstm_weights = model.get('lstm_weights', [])
    if lstm_weights:
        kernel = ', '.join(_fmt(v) for row in lstm_weights[0] for v in row)
        recurrent = ', '.join(_fmt(v) for row in lstm_weights[1] for v in row)
        bias = ', '.join(_fmt(v) for v in lstm_weights[2])
        dense_w = ', '.join(_fmt(v) for row in lstm_weights[3] for v in (row if isinstance(row, list) else [row]))
        dense_b = _fmt(lstm_weights[4][0] if isinstance(lstm_weights[4], list) else lstm_weights[4])
        lstm_hidden = len(lstm_weights[1])
    else:
        kernel = recurrent = bias = dense_w = ''
        dense_b = '0'
        lstm_hidden = 0
    seq_len = model.get('sequence_length', 0)
    output = output.replace('__LSTM_KERNEL__', kernel)
    output = output.replace('__LSTM_RECURRENT__', recurrent)
    output = output.replace('__LSTM_BIAS__', bias)
    output = output.replace('__LSTM_DENSE_W__', dense_w)
    output = output.replace('__LSTM_DENSE_B__', dense_b)
    output = output.replace('__LSTM_HIDDEN_SIZE__', str(lstm_hidden))
    output = output.replace('__LSTM_SEQ_LEN__', str(seq_len))

    feature_names = model.get('feature_names', [])
    feature_count = len(feature_names)

    feature_map = {
        'hour': 'TimeHour(TimeCurrent())',
        'spread': 'MarketInfo(SymbolToTrade, MODE_SPREAD)',
        'lots': 'Lots',
        'sl_dist': 'GetSLDistance()',
        'tp_dist': 'GetTPDistance()',
        'day_of_week': 'TimeDayOfWeek(TimeCurrent())',
        'sma': 'iMA(SymbolToTrade, 0, 5, 0, MODE_SMA, PRICE_CLOSE, 0)',
        'rsi': 'iRSI(SymbolToTrade, 0, 14, PRICE_CLOSE, 0)',
        'macd': 'iMACD(SymbolToTrade, 0, 12, 26, 9, PRICE_CLOSE, MODE_MAIN, 0)',
        'macd_signal': 'iMACD(SymbolToTrade, 0, 12, 26, 9, PRICE_CLOSE, MODE_SIGNAL, 0)',
        'volatility': 'StdDevRecentTicks()',
        'atr': 'iATR(SymbolToTrade, 0, 14, 0)',
        'bollinger_upper': 'iBands(SymbolToTrade, 0, 20, 2, 0, PRICE_CLOSE, MODE_UPPER, 0)',
        'bollinger_middle': 'iBands(SymbolToTrade, 0, 20, 2, 0, PRICE_CLOSE, MODE_MAIN, 0)',
        'bollinger_lower': 'iBands(SymbolToTrade, 0, 20, 2, 0, PRICE_CLOSE, MODE_LOWER, 0)',
    }

    cases = []
    for idx, name in enumerate(feature_names):
        expr = feature_map.get(name, '0.0')
        cases.append(f"      case {idx}:\n         return({expr});")
    case_block = "\n".join(cases)
    if case_block:
        case_block += "\n"
    output = output.replace('__FEATURE_CASES__', case_block)
    output = output.replace('__FEATURE_COUNT__', str(feature_count))

    intercept = model.get('intercept', 0.0)
    output = output.replace('__INTERCEPT__', _fmt(intercept))

    threshold = model.get('threshold', 0.5)
    output = output.replace('__THRESHOLD__', _fmt(threshold))
    ts = model.get('trained_at')
    if ts:
        try:
            from datetime import datetime
            ts = datetime.fromisoformat(ts).strftime('%Y%m%d_%H%M%S')
        except Exception:
            ts = None
    if not ts:
        from datetime import datetime
        ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    out_file = out_dir / f"Generated_{model.get('model_id', 'model')}_{ts}.mq4"
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
