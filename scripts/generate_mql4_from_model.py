#!/usr/bin/env python3
"""Render MQL4 strategy file from model description."""
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Iterable, List, Union


def _fmt(value: float) -> str:
    """Format floats for insertion into MQL4 code."""
    return f"{value:.6g}"

template_path = (
    Path(__file__).resolve().parent.parent / 'experts' / 'StrategyTemplate.mq4'
)


def generate(model_jsons: Union[Path, Iterable[Path]], out_dir: Path):
    if isinstance(model_jsons, (str, Path)):
        model_jsons = [model_jsons]
    models: List[dict] = []
    for mj in model_jsons:
        with open(mj) as f:
            models.append(json.load(f))
    base = models[0]
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(template_path) as f:
        template = f.read()

    output = template.replace(
        'MagicNumber = 1234',
        f"MagicNumber = {base.get('magic', 9999)}",
    )

    # merge feature names preserving order
    feature_names: List[str] = []
    for m in models:
        for name in m.get('feature_names', []):
            if name not in feature_names:
                feature_names.append(name)
    feature_count = len(feature_names)

    coeff_rows = []
    intercepts = []
    for m in models:
        coeffs = m.get('coefficients') or m.get('coef_vector', [])
        if not coeffs:
            q_w = m.get('q_weights')
            if isinstance(q_w, list) and len(q_w) >= 2:
                try:
                    import numpy as np
                    coeffs = (np.array(q_w[0]) - np.array(q_w[1])).tolist()
                except Exception:
                    coeffs = []
        fmap = {f: c for f, c in zip(m.get('feature_names', []), coeffs)}
        vec = [_fmt(fmap.get(f, 0.0)) for f in feature_names]
        coeff_rows.append('{'+', '.join(vec)+'}')
        intercept = m.get('intercept')
        if intercept is None:
            q_int = m.get('q_intercepts')
            if isinstance(q_int, list) and len(q_int) >= 2:
                intercept = float(q_int[0]) - float(q_int[1])
            else:
                intercept = 0.0
        intercepts.append(_fmt(intercept))

    coeff_str = ', '.join(coeff_rows)
    output = output.replace('__COEFFICIENTS__', coeff_str)
    output = output.replace('__INTERCEPTS__', ', '.join(intercepts))
    output = output.replace('__MODEL_COUNT__', str(len(models)))

    prob_table = base.get('probability_table', [])
    prob_str = ', '.join(_fmt(p) for p in prob_table)
    output = output.replace('__PROBABILITY_TABLE__', prob_str)

    hourly_thr = base.get('hourly_thresholds', [])
    thr_str = ', '.join(_fmt(t) for t in hourly_thr)
    output = output.replace('__HOURLY_THRESHOLDS__', thr_str)

    sl_coeff = base.get('sl_coefficients', [])
    sl_str = ', '.join(_fmt(c) for c in sl_coeff)
    output = output.replace('__SL_COEFFICIENTS__', sl_str)
    output = output.replace('__SL_INTERCEPT__', _fmt(base.get('sl_intercept', 0.0)))

    tp_coeff = base.get('tp_coefficients', [])
    tp_str = ', '.join(_fmt(c) for c in tp_coeff)
    output = output.replace('__TP_COEFFICIENTS__', tp_str)
    output = output.replace('__TP_INTERCEPT__', _fmt(base.get('tp_intercept', 0.0)))

    nn_weights = base.get('nn_weights', [])
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

    lstm_weights = base.get('lstm_weights', [])
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
    seq_len = base.get('sequence_length', 0)
    output = output.replace('__LSTM_KERNEL__', kernel)
    output = output.replace('__LSTM_RECURRENT__', recurrent)
    output = output.replace('__LSTM_BIAS__', bias)
    output = output.replace('__LSTM_DENSE_W__', dense_w)
    output = output.replace('__LSTM_DENSE_B__', dense_b)
    output = output.replace('__LSTM_HIDDEN_SIZE__', str(lstm_hidden))
    output = output.replace('__LSTM_SEQ_LEN__', str(seq_len))

    trans_weights = base.get('transformer_weights', [])
    def _flat(a):
        if isinstance(a, list):
            r = []
            for x in a:
                r.extend(_flat(x))
            return r
        return [a]
    if trans_weights:
        qk = ', '.join(_fmt(v) for v in _flat(trans_weights[0]))
        qb = ', '.join(_fmt(v) for v in _flat(trans_weights[1]))
        kk = ', '.join(_fmt(v) for v in _flat(trans_weights[2]))
        kb = ', '.join(_fmt(v) for v in _flat(trans_weights[3]))
        vk = ', '.join(_fmt(v) for v in _flat(trans_weights[4]))
        vb = ', '.join(_fmt(v) for v in _flat(trans_weights[5]))
        ok = ', '.join(_fmt(v) for v in _flat(trans_weights[6]))
        ob = ', '.join(_fmt(v) for v in _flat(trans_weights[7]))
        dw = ', '.join(_fmt(v) for v in _flat(trans_weights[8]))
        db = _fmt(trans_weights[9][0] if isinstance(trans_weights[9], list) else trans_weights[9])
    else:
        qk = qb = kk = kb = vk = vb = ok = ob = dw = ''
        db = '0'
    output = output.replace('__TRANS_QK__', qk)
    output = output.replace('__TRANS_QB__', qb)
    output = output.replace('__TRANS_KK__', kk)
    output = output.replace('__TRANS_KB__', kb)
    output = output.replace('__TRANS_VK__', vk)
    output = output.replace('__TRANS_VB__', vb)
    output = output.replace('__TRANS_OK__', ok)
    output = output.replace('__TRANS_OB__', ob)
    output = output.replace('__TRANS_DENSE_W__', dw)
    output = output.replace('__TRANS_DENSE_B__', db)

    enc_weights = base.get('encoder_weights', [])
    enc_window = int(base.get('encoder_window', 0))
    enc_dim = len(enc_weights[0]) if enc_weights else 0
    enc_flat = ', '.join(_fmt(v) for row in enc_weights for v in row)
    output = output.replace('__ENCODER_WEIGHTS__', enc_flat)
    output = output.replace('__ENCODER_WINDOW__', str(enc_window))
    output = output.replace('__ENCODER_DIM__', str(enc_dim))

    centers = base.get('encoder_centers', [])
    center_flat = ', '.join(_fmt(v) for row in centers for v in row)
    output = output.replace('__ENCODER_CENTERS__', center_flat)
    output = output.replace('__ENCODER_CENTER_COUNT__', str(len(centers)))

    feature_mean = base.get('feature_mean', [])
    mean_str = ', '.join(_fmt(v) for v in feature_mean)
    output = output.replace('__FEATURE_MEAN__', mean_str)
    feature_std = base.get('feature_std', [])
    std_str = ', '.join(_fmt(v) for v in feature_std)
    output = output.replace('__FEATURE_STD__', std_str)

    cal_events = base.get('calendar_events', [])
    if cal_events:
        time_vals = ', '.join(
            datetime.fromisoformat(t).strftime("D'%Y.%m.%d %H:%M'")
            for t, _ in cal_events
        )
        impact_vals = ', '.join(_fmt(float(imp)) for _, imp in cal_events)
    else:
        time_vals = ''
        impact_vals = ''
    event_window = _fmt(base.get('event_window', 60.0))
    output = output.replace('__CALENDAR_TIMES__', time_vals)
    output = output.replace('__CALENDAR_IMPACTS__', impact_vals)
    output = output.replace('__EVENT_WINDOW__', event_window)

    feature_map = {
        'hour': 'TimeHour(TimeCurrent())',
        'hour_sin': 'MathSin(2*M_PI*TimeHour(TimeCurrent())/24)',
        'hour_cos': 'MathCos(2*M_PI*TimeHour(TimeCurrent())/24)',
        'dow_sin': 'MathSin(2*M_PI*TimeDayOfWeek(TimeCurrent())/7)',
        'dow_cos': 'MathCos(2*M_PI*TimeDayOfWeek(TimeCurrent())/7)',
        'spread': 'MarketInfo(SymbolToTrade, MODE_SPREAD)',
        'lots': 'Lots',
        'sl_dist': 'GetSLDistance()',
        'tp_dist': 'GetTPDistance()',
        'equity': 'AccountEquity()',
        'margin_level': 'AccountMarginLevel()',
        'day_of_week': 'TimeDayOfWeek(TimeCurrent())',
        'sma': 'CachedSMA[TFIdx(0)]',
        'rsi': 'CachedRSI[TFIdx(0)]',
        'macd': 'CachedMACD[TFIdx(0)]',
        'macd_signal': 'CachedMACDSignal[TFIdx(0)]',
        'volatility': 'StdDevRecentTicks()',
        'atr': 'iATR(SymbolToTrade, 0, 14, 0)',
        'bollinger_upper': 'iBands(SymbolToTrade, 0, 20, 2, 0, PRICE_CLOSE, MODE_UPPER, 0)',
        'bollinger_middle': 'iBands(SymbolToTrade, 0, 20, 2, 0, PRICE_CLOSE, MODE_MAIN, 0)',
        'bollinger_lower': 'iBands(SymbolToTrade, 0, 20, 2, 0, PRICE_CLOSE, MODE_LOWER, 0)',
        'stochastic_k': 'iStochastic(SymbolToTrade, 0, 14, 3, 3, MODE_SMA, 0, MODE_MAIN, 0)',
        'stochastic_d': 'iStochastic(SymbolToTrade, 0, 14, 3, 3, MODE_SMA, 0, MODE_SIGNAL, 0)',
        'adx': 'iADX(SymbolToTrade, 0, 14, PRICE_CLOSE, MODE_MAIN, 0)',
        'regime': 'GetRegime()',
        'volume': 'iVolume(SymbolToTrade, 0, 0)',
        'event_flag': 'GetCalendarFlag()',
        'event_impact': 'GetCalendarImpact()',
    }

    tf_const = {
        'M1': 'PERIOD_M1',
        'M5': 'PERIOD_M5',
        'M15': 'PERIOD_M15',
        'M30': 'PERIOD_M30',
        'H1': 'PERIOD_H1',
        'H4': 'PERIOD_H4',
        'D1': 'PERIOD_D1',
        'W1': 'PERIOD_W1',
        'MN1': 'PERIOD_MN1',
    }

    import re
    cases = []
    used_tfs = {'0'}
    tf_pattern = re.compile(r'^(sma|rsi|macd|macd_signal)_([A-Za-z0-9]+)$')
    indicator_expr = {
        'sma': 'CachedSMA',
        'rsi': 'CachedRSI',
        'macd': 'CachedMACD',
        'macd_signal': 'CachedMACDSignal',
    }
    for idx, name in enumerate(feature_names):
        expr = feature_map.get(name)
        if expr is None:
            m = tf_pattern.match(name)
            if m:
                ind, tf = m.groups()
                tf = tf.upper()
                tf_val = tf_const.get(tf, '0')
                used_tfs.add(tf_val)
                expr = f"{indicator_expr[ind]}[TFIdx({tf_val})]"
            elif name.startswith('ratio_'):
                parts = name[6:].split('_')
                if len(parts) == 2:
                    expr = f'iClose("{parts[0]}", 0, 0) / iClose("{parts[1]}", 0, 0)'
            elif name.startswith('corr_'):
                parts = name[5:].split('_')
                if len(parts) == 2:
                    expr = (
                        f'iMA("{parts[0]}", 0, 5, 0, MODE_SMA, PRICE_CLOSE, 0) - '
                        f'iMA("{parts[1]}", 0, 5, 0, MODE_SMA, PRICE_CLOSE, 0)'
                    )
            elif name.startswith('ae') and name[2:].isdigit():
                idx_ae = int(name[2:])
                expr = f'GetEncodedFeature({idx_ae})'
        if expr is None:
            expr = '0.0'
        cases.append(f"      case {idx}:\n         val = ({expr});\n         break;")
    case_block = "\n".join(cases)
    if case_block:
        case_block += "\n"
    output = output.replace('__FEATURE_CASES__', case_block)
    output = output.replace('__FEATURE_COUNT__', str(feature_count))

    tf_order = [
        '0',
        'PERIOD_M1',
        'PERIOD_M5',
        'PERIOD_M15',
        'PERIOD_M30',
        'PERIOD_H1',
        'PERIOD_H4',
        'PERIOD_D1',
        'PERIOD_W1',
        'PERIOD_MN1',
    ]
    tf_list = [t for t in tf_order if t in used_tfs]
    output = output.replace('__CACHE_TIMEFRAMES__', ', '.join(tf_list))
    output = output.replace('__CACHE_TF_COUNT__', str(len(tf_list)))

    threshold = base.get('threshold', 0.5)
    output = output.replace('__THRESHOLD__', _fmt(threshold))
    ts = base.get('trained_at')
    if ts:
        try:
            ts = datetime.fromisoformat(ts).strftime('%Y%m%d_%H%M%S')
        except Exception:
            ts = None
    if not ts:
        ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    out_file = out_dir / f"Generated_{base.get('model_id', 'model')}_{ts}.mq4"
    with open(out_file, 'w') as f:
        f.write(output)
    print(f"Strategy written to {out_file}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('model_json', nargs='+')
    p.add_argument('out_dir')
    args = p.parse_args()
    generate([Path(m) for m in args.model_json], Path(args.out_dir))


if __name__ == '__main__':
    main()
