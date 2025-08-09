#!/usr/bin/env python3
"""Render MQL4 strategy file from model description.

This utility now supports embedding weights for decision transformer models
trained via :mod:`scripts.train_rl_agent` using the ``--algo decision_transformer``
option. When such weights are present in the model JSON they are injected into
the generated MQL4 source so that inference can be performed on-platform.
"""
import argparse
import json
import gzip
from pathlib import Path
from datetime import datetime
from typing import Iterable, List, Union, Optional


def _fmt(value: float) -> str:
    """Format floats for insertion into MQL4 code."""
    return f"{value:.6g}"

template_path = (
    Path(__file__).resolve().parent.parent / 'experts' / 'StrategyTemplate.mq4'
)


def generate(
    model_jsons: Union[Path, Iterable[Path]],
    out_dir: Path,
    lite_mode: bool = False,
    gating_json: Optional[Path] = None,
):
    """Render an MQL4 strategy using one or more base models.

    Parameters
    ----------
    model_jsons : list of Path
        Paths to JSON files describing base models.
    out_dir : Path
        Directory where the generated MQL4 file will be written.
    lite_mode : bool, optional
        If True, exclude order book features.
    gating_json : Path, optional
        Optional JSON file produced by ``meta_strategy.py`` containing
        ``gating_coefficients`` and ``gating_intercepts``. When provided,
        these parameters will be embedded to allow runtime selection of the
        best-performing base model.
    """
    if isinstance(model_jsons, (str, Path)):
        model_jsons = [model_jsons]
    models: List[dict] = []
    gating_data = None
    for mj in model_jsons:
        open_func = gzip.open if str(mj).endswith('.gz') else open
        with open_func(mj, 'rt') as f:
            data = json.load(f)
        if data.get('regime_models') and data.get('meta_model'):
            gating_data = data.get('meta_model')
            base_info = {k: v for k, v in data.items() if k not in ('regime_models', 'meta_model')}
            for sm in data.get('regime_models', []):
                m = base_info.copy()
                m.update(sm)
                models.append(m)
        else:
            sessions = data.get('session_models')
            if sessions:
                base_info = {k: v for k, v in data.items() if k != 'session_models'}
                for sm in sessions:
                    m = base_info.copy()
                    m.update(sm)
                    models.append(m)
            else:
                models.append(data)
    base = models[0]
    if gating_json:
        with open(gating_json, 'rt') as f:
            gating_data = json.load(f)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(template_path) as f:
        template = f.read()

    output = template.replace(
        'MagicNumber = 1234',
        f"MagicNumber = {base.get('magic', 9999)}",
    )

    # Embed hierarchy metadata if present.  The data is JSON encoded and
    # escaped so it can be stored in the generated MQL4 source as a string
    # literal.  Consumers may parse this at runtime for additional context
    # about the trained meta-controller and sub-policies.
    hier_json = json.dumps(base.get('hierarchy', {})).replace('"', '\\"')
    output = output.replace('__HIERARCHY_JSON__', hier_json)

    # merge feature names preserving order
    feature_names: List[str] = []
    for m in models:
        for name in m.get('feature_names', []):
            if name not in feature_names:
                feature_names.append(name)
    if gating_data:
        for name in gating_data.get('feature_names', []):
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

    # Gating coefficients / intercepts
    if gating_data:
        g_feat = gating_data.get('feature_names', [])
        g_coeff = gating_data.get('coefficients') or gating_data.get('gating_coefficients', [])
        coeff_list: List[List[float]]
        if g_coeff and isinstance(g_coeff[0], list):
            coeff_list = g_coeff
        else:
            coeff_list = []
            for i in range(len(models)):
                start = i * len(g_feat)
                coeff_list.append(g_coeff[start : start + len(g_feat)])
        g_rows: List[str] = []
        for row in coeff_list:
            fmap = {f: c for f, c in zip(g_feat, row)}
            vec = [_fmt(fmap.get(f, 0.0)) for f in feature_names]
            g_rows.append('{' + ', '.join(vec) + '}')
        output = output.replace('__GATING_COEFFICIENTS__', ', '.join(g_rows))
        g_inter = gating_data.get('intercepts') or gating_data.get('gating_intercepts', [])
        g_inter_str = ', '.join(_fmt(x) for x in g_inter) if g_inter else ''
        output = output.replace('__GATING_INTERCEPTS__', g_inter_str)
    else:
        g_coeff = base.get('gating_coefficients', [])
        if isinstance(g_coeff, list) and g_coeff and isinstance(g_coeff[0], list):
            flat = [c for row in g_coeff for c in row]
        else:
            flat = g_coeff
        g_rows: List[str] = []
        if flat:
            for i in range(len(models)):
                row = flat[i * feature_count : (i + 1) * feature_count]
                g_rows.append('{' + ', '.join(_fmt(c) for c in row) + '}')
        output = output.replace('__GATING_COEFFICIENTS__', ', '.join(g_rows))
        g_inter = base.get('gating_intercepts', [])
        g_inter_str = ', '.join(_fmt(x) for x in g_inter) if g_inter else ''
        output = output.replace('__GATING_INTERCEPTS__', g_inter_str)

    cal_coef = _fmt(base.get('calibration_coef', 1.0))
    cal_inter = _fmt(base.get('calibration_intercept', 0.0))
    output = output.replace('__CAL_COEF__', cal_coef)
    output = output.replace('__CAL_INTERCEPT__', cal_inter)
    session_starts: List[str] = []
    session_ends: List[str] = []
    prob_rows: List[str] = []
    for m in models:
        rng = m.get('session_range', [0, 24])
        session_starts.append(str(int(rng[0])))
        session_ends.append(str(int(rng[1])))
        pt = m.get('probability_table', [0.0] * 24)
        if len(pt) < 24:
            pt = list(pt) + [0.0] * (24 - len(pt))
        prob_rows.append('{' + ', '.join(_fmt(p) for p in pt) + '}')
    output = output.replace('__SESSION_STARTS__', ', '.join(session_starts))
    output = output.replace('__SESSION_ENDS__', ', '.join(session_ends))
    output = output.replace('__PROBABILITY_TABLE__', ', '.join(prob_rows))

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

    # Decision Transformer weights exported by train_rl_agent.py
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

    reg_centers = base.get('regime_centers', [])
    reg_feat_names = base.get('regime_feature_names', [])
    reg_feat_idx = [feature_names.index(f) for f in reg_feat_names if f in feature_names]
    reg_center_flat = ', '.join('{' + ', '.join(_fmt(v) for v in c) + '}' for c in reg_centers) if reg_centers else '{0}'
    reg_feat_idx_str = ', '.join(str(i) for i in reg_feat_idx) if reg_feat_idx else '0'
    output = output.replace('__REGIME_CENTERS__', reg_center_flat)
    output = output.replace('__REGIME_COUNT__', str(len(reg_centers) or 1))
    output = output.replace('__REGIME_FEATURE_COUNT__', str(len(reg_feat_idx) or 1))
    output = output.replace('__REGIME_FEATURE_IDX__', reg_feat_idx_str)

    mean_vals = base.get('mean', base.get('feature_mean', []))
    std_vals = base.get('std', base.get('feature_std', []))

    mean_map = {f: m for f, m in zip(base.get('feature_names', []), mean_vals)}
    mean_vec = [_fmt(mean_map.get(f, 0.0)) for f in feature_names]
    output = output.replace('__FEATURE_MEAN__', ', '.join(mean_vec))
    std_map = {f: s for f, s in zip(base.get('feature_names', []), std_vals)}
    std_vec = [_fmt(std_map.get(f, 1.0)) for f in feature_names]
    output = output.replace('__FEATURE_STD__', ', '.join(std_vec))

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
        'hour_sin': 'HourSin()',
        'hour_cos': 'HourCos()',
        'dow_sin': 'DowSin()',
        'dow_cos': 'DowCos()',
        'spread': 'MarketInfo(SymbolToTrade, MODE_SPREAD)',
        'lots': 'Lots',
        'sl_dist': 'GetSLDistance()',
        'tp_dist': 'GetTPDistance()',
        'equity': 'AccountEquity()',
        'margin_level': 'AccountMarginLevel()',
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
        'volume': 'iVolume(SymbolToTrade, 0, 0)',
        'event_flag': 'GetCalendarFlag()',
        'event_impact': 'GetCalendarImpact()',
        'book_bid_vol': 'BookBidVol()',
        'book_ask_vol': 'BookAskVol()',
        'book_imbalance': 'BookImbalance()',
    }

    if lite_mode:
        feature_map['book_bid_vol'] = '0.0'
        feature_map['book_ask_vol'] = '0.0'
        feature_map['book_imbalance'] = '0.0'

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
            elif name.startswith('regime_') and name[7:].isdigit():
                rid = int(name.split('_')[1])
                expr = f'(GetRegime() == {rid} ? 1.0 : 0.0)'
            elif name == 'regime':
                expr = 'GetRegime()'
            elif name.startswith('ratio_'):
                parts = name[6:].split('_')
                if len(parts) == 2:
                    expr = f'iClose("{parts[0]}", 0, 0) / iClose("{parts[1]}", 0, 0)'
            elif name.startswith('corr_'):
                parts = name[5:].split('_')
                if len(parts) == 2:
                    expr = f'PairCorrelation("{parts[0]}", "{parts[1]}")'
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
    p.add_argument('--lite-mode', action='store_true')
    p.add_argument('--gating-json')
    args = p.parse_args()
    generate(
        [Path(m) for m in args.model_json],
        Path(args.out_dir),
        lite_mode=args.lite_mode,
        gating_json=Path(args.gating_json) if args.gating_json else None,
    )


if __name__ == '__main__':
    main()
