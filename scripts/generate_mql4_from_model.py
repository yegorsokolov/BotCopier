#!/usr/bin/env python3
"""Render MQL4 strategy file from model description.

This utility now supports embedding weights for decision transformer models
trained via :mod:`scripts.train_rl_agent` using the ``--algo decision_transformer``
option. When such weights are present in the model JSON they are injected into
the generated MQL4 source so that inference can be performed on-platform.

The generated Expert Advisor also supports a ``ReplayDecisions`` flag which
causes it to reprocess an existing decision log at start-up and print any
divergences between past and current model outputs.  This mirrors the
behaviour of :mod:`scripts.replay_decisions` directly inside MetaTrader.
"""
import argparse
import json
import gzip
import logging
import numpy as np
from sklearn.feature_extraction import FeatureHasher
from pathlib import Path
from datetime import datetime
from typing import Iterable, List, Union, Optional


def _fmt(value: float) -> str:
    """Format floats for insertion into MQL4 code."""
    return f"{value:.6g}"


def _prune_model_features(model: dict) -> dict:
    """Remove features with zero importance from ``model``.

    Training scripts persist SHAP importance scores under
    ``feature_importance``.  When present, we drop any feature whose
    importance is zero so the generated MQL4 code only contains cases for
    the active subset.
    """

    fi = model.get("feature_importance")
    names = model.get("feature_names")
    if isinstance(fi, dict) and isinstance(names, list):
        keep = [i for i, n in enumerate(names) if fi.get(n, 0.0) > 0.0]
        if len(keep) != len(names):
            model["feature_names"] = [names[i] for i in keep]
            # Keep related arrays in sync when their lengths match the feature list
            for key in (
                "coefficients",
                "student_coefficients",
                "coef_variances",
                "mean",
                "feature_mean",
                "std",
                "feature_std",
                "sl_coefficients",
                "tp_coefficients",
                "lot_coefficients",
            ):
                if key in model and isinstance(model[key], list):
                    arr = model[key]
                    model[key] = [arr[i] for i in keep if i < len(arr)]
    return model

template_path = (
    Path(__file__).resolve().parent.parent / 'experts' / 'StrategyTemplate.mq4'
)


def generate(
    model_jsons: Union[Path, Iterable[Path]],
    out_dir: Path,
    lite_mode: bool | None = None,
    gating_json: Optional[Path] = None,
    symbol_graph: Optional[Path] = None,
):
    """Render an MQL4 strategy using one or more base models.

    Parameters
    ----------
    model_jsons : list of Path
        Paths to JSON files describing base models.
    out_dir : Path
        Directory where the generated MQL4 file will be written.
    lite_mode : bool, optional
        If True, exclude order book features. If ``None`` the mode is inferred
        from ``model.json`` metadata.
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
        data = _prune_model_features(data)
        if data.get('regime_models') and data.get('meta_model'):
            gating_data = _prune_model_features(data.get('meta_model'))
            base_info = {k: v for k, v in data.items() if k not in ('regime_models', 'meta_model')}
            for sm in data.get('regime_models', []):
                m = base_info.copy()
                m.update(sm)
                models.append(_prune_model_features(m))
        else:
            sessions = data.get('session_models')
            if sessions:
                base_info = {k: v for k, v in data.items() if k != 'session_models'}
                for sm in sessions:
                    m = base_info.copy()
                    m.update(sm)
                    models.append(_prune_model_features(m))
            else:
                models.append(data)
    base = models[0]
    if lite_mode is None:
        mode = base.get('mode') or base.get('training_mode')
        lite_mode = not (mode and mode in ('heavy', 'deep', 'rl'))
        ff = base.get('feature_flags', {})
        if 'order_book' in ff:
            lite_mode = not ff.get('order_book', False)
    if gating_json:
        with open(gating_json, 'rt') as f:
            gating_data = _prune_model_features(json.load(f))
    if gating_data and 'feature_names' not in gating_data:
        gating_data['feature_names'] = base.get('feature_names', [])
    hash_size = base.get('hash_size')
    if hash_size and gating_data:
        gating_data = None

    has_gpu_weights = any(
        m.get('transformer_weights')
        or m.get('lstm_weights')
        or m.get('nn_weights')
        for m in models
    )
    print(
        'GPU-trained weights detected'
        if has_gpu_weights
        else 'No GPU-trained weights detected'
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(template_path) as f:
        template = f.read()

    enabled_feats: List[str] = []
    for m in models:
        for k, v in (m.get('feature_flags') or {}).items():
            if v and k not in enabled_feats:
                enabled_feats.append(k)
    if 'month' not in enabled_feats:
        enabled_feats.append('month')
    feats_comment = (
        "// features: " + ", ".join(sorted(enabled_feats)) + "\n" if enabled_feats else ""
    )
    output = (
        f"// GPU-trained weights: {'yes' if has_gpu_weights else 'no'}\n" + feats_comment + template
    )

    output = output.replace(
        'MagicNumber = 1234',
        f"MagicNumber = {base.get('magic', 9999)}",
    )

    # Embed hierarchy metadata if present.  The data is JSON encoded and
    # escaped so it can be stored in the generated MQL4 source as a string
    # literal.  Consumers may parse this at runtime for additional context
    # about the trained meta-controller and sub-policies.
    hier_json = json.dumps(base.get('hierarchy', {})).replace('"', '\\"')
    output = output.replace('__HIERARCHY_JSON__', hier_json)

    # merge feature names preserving order or gather for hashing
    feature_names: List[str] = []
    if hash_size:
        seen = set()
        for m in models:
            for name in m.get('feature_names', []):
                if name not in seen:
                    seen.add(name)
                    feature_names.append(name)
        feature_count = int(hash_size)
    else:
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
    var_rows = []
    noise_vars = []
    for m in models:
        coeffs = (
            m.get('student_coefficients')
            or m.get('coefficients')
            or m.get('coef_vector', [])
        )
        if not coeffs:
            q_w = m.get('q_weights')
            if isinstance(q_w, list) and len(q_w) >= 2:
                try:
                    coeffs = (np.array(q_w[0]) - np.array(q_w[1])).tolist()
                except Exception:
                    coeffs = []
        if hash_size:
            vec = [_fmt(c) for c in coeffs]
        else:
            fmap = {f: c for f, c in zip(m.get('feature_names', []), coeffs)}
            vec = [_fmt(fmap.get(f, 0.0)) for f in feature_names]
        coeff_rows.append('{' + ', '.join(vec) + '}')
        intercept = m.get('student_intercept')
        if intercept is None:
            intercept = m.get('intercept')
        if intercept is None:
            q_int = m.get('q_intercepts')
            if isinstance(q_int, list) and len(q_int) >= 2:
                intercept = float(q_int[0]) - float(q_int[1])
            else:
                intercept = 0.0
        intercepts.append(_fmt(intercept))
        coef_var = m.get('coef_variances', [])
        if hash_size:
            vec_v = [_fmt(v) for v in coef_var]
        else:
            fmap_v = {f: v for f, v in zip(m.get('feature_names', []), coef_var)}
            vec_v = [_fmt(fmap_v.get(f, 0.0)) for f in feature_names]
        var_rows.append('{' + ', '.join(vec_v) + '}')
        noise_vars.append(_fmt(m.get('noise_variance', 0.0)))

    coeff_str = ', '.join(coeff_rows)
    output = output.replace('__COEFFICIENTS__', coeff_str)
    output = output.replace('__INTERCEPTS__', ', '.join(intercepts))
    output = output.replace('__MODEL_COUNT__', str(len(models)))
    output = output.replace('__COEF_VARIANCES__', ', '.join(var_rows))
    output = output.replace('__NOISE_VARIANCES__', ', '.join(noise_vars))

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

    # Entry and exit models
    entry_coeff = base.get('entry_coefficients', [])
    entry_str = ', '.join(_fmt(c) for c in entry_coeff)
    output = output.replace('__ENTRY_COEFFICIENTS__', entry_str)
    output = output.replace('__ENTRY_INTERCEPT__', _fmt(base.get('entry_intercept', 0.0)))
    output = output.replace('__ENTRY_THRESHOLD__', _fmt(base.get('entry_threshold', 0.5)))

    exit_coeff = base.get('exit_coefficients', [])
    exit_str = ', '.join(_fmt(c) for c in exit_coeff)
    output = output.replace('__EXIT_COEFFICIENTS__', exit_str)
    output = output.replace('__EXIT_INTERCEPT__', _fmt(base.get('exit_intercept', 0.0)))
    output = output.replace('__EXIT_THRESHOLD__', _fmt(base.get('exit_threshold', 0.5)))

    # Lot size model
    lot_coeff = base.get('lot_coefficients', [])
    lot_str = ', '.join(_fmt(c) for c in lot_coeff)
    output = output.replace('__LOT_COEFFICIENTS__', lot_str)
    output = output.replace('__LOT_INTERCEPT__', _fmt(base.get('lot_intercept', 0.0)))

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
    trans_call = (
        '   if(LSTMSequenceLength > 0 && ArraySize(TransformerDenseWeights) > 0)\n'
        '      return(ComputeDecisionTransformerScore());\n'
    )
    if not trans_weights:
        output = output.replace(trans_call, '')

    enc_weights = base.get('encoder_weights', [])
    enc_window = int(base.get('encoder_window', 0))
    enc_dim = len(enc_weights[0]) if enc_weights else 0
    enc_flat = ', '.join(_fmt(v) for row in enc_weights for v in row)
    output = output.replace('__ENCODER_WEIGHTS__', enc_flat)
    output = output.replace('__ENCODER_WINDOW__', str(enc_window))
    output = output.replace('__ENCODER_DIM__', str(enc_dim))
    output = output.replace('__ENCODER_ONNX__', base.get('encoder_onnx', 'encoder.onnx'))
    onnx_file = base.get('onnx_file')
    if not onnx_file and base.get('rl_algo') == 'decision_transformer':
        onnx_file = 'decision_transformer.onnx'
    output = output.replace('__MODEL_ONNX__', onnx_file or '')

    centers = base.get('encoder_centers', [])
    center_flat = ', '.join(_fmt(v) for row in centers for v in row)
    output = output.replace('__ENCODER_CENTERS__', center_flat)
    output = output.replace('__ENCODER_CENTER_COUNT__', str(len(centers)))

    reg_centers = base.get('regime_centers', [])
    if not reg_centers:
        reg_centers = [[0.0]] * len(models)
    reg_feat_names = base.get('regime_feature_names', [])
    reg_feat_idx = [feature_names.index(f) for f in reg_feat_names if f in feature_names]
    if not reg_feat_idx:
        reg_feat_idx = [0]
    reg_center_flat = ', '.join('{' + ', '.join(_fmt(v) for v in c) + '}' for c in reg_centers)
    reg_feat_idx_str = ', '.join(str(i) for i in reg_feat_idx)
    output = output.replace('__REGIME_CENTERS__', reg_center_flat)
    output = output.replace('__REGIME_COUNT__', str(len(reg_centers)))
    output = output.replace('__REGIME_FEATURE_COUNT__', str(len(reg_feat_idx)))
    output = output.replace('__REGIME_FEATURE_IDX__', reg_feat_idx_str)
    reg_model_idx = base.get('regime_model_idx')
    if not reg_model_idx:
        reg_model_idx = list(range(len(models)))
    reg_model_idx_str = ', '.join(str(int(i)) for i in reg_model_idx)
    output = output.replace('__REGIME_MODEL_IDX__', reg_model_idx_str)
    reg_thr = base.get('regime_thresholds') or []
    reg_thr_str = ', '.join(_fmt(t) for t in reg_thr) if reg_thr else ''
    output = output.replace('__REGIME_THRESHOLDS__', reg_thr_str)

    mean_vals = base.get('feature_mean', base.get('mean', []))
    std_vals = base.get('feature_std', base.get('std', []))

    if hash_size:
        mean_vec = [_fmt(m) for m in mean_vals]
        std_vec = [_fmt(s) for s in std_vals]
    else:
        mean_map = {f: m for f, m in zip(base.get('feature_names', []), mean_vals)}
        mean_vec = [_fmt(mean_map.get(f, 0.0)) for f in feature_names]
        std_map = {f: s for f, s in zip(base.get('feature_names', []), std_vals)}
        std_vec = [_fmt(std_map.get(f, 1.0)) for f in feature_names]
    output = output.replace('__FEATURE_MEAN__', ', '.join(mean_vec))
    output = output.replace('__FEATURE_STD__', ', '.join(std_vec))

    emb_data = base.get('symbol_embeddings', {})
    if emb_data:
        emb_symbols = list(emb_data.keys())
        emb_dim = len(next(iter(emb_data.values()), []))
        sym_list = ', '.join(f'"{s}"' for s in emb_symbols)
        emb_rows = ', '.join(
            '{' + ', '.join(_fmt(v) for v in emb_data[s]) + '}' for s in emb_symbols
        )
        output = output.replace('__SYM_EMB_DIM__', str(emb_dim))
        output = output.replace('__SYM_EMB_COUNT__', str(len(emb_symbols)))
        output = output.replace('__SYM_EMB_SYMBOLS__', sym_list)
        output = output.replace('__SYM_EMB_VALUES__', emb_rows)
    else:
        output = output.replace('__SYM_EMB_DIM__', '0')
        output = output.replace('__SYM_EMB_COUNT__', '0')
        output = output.replace('__SYM_EMB_SYMBOLS__', '')
        output = output.replace('__SYM_EMB_VALUES__', '')

    graph_data = base.get('graph', {})
    if (not graph_data or not graph_data.get('symbols')) and symbol_graph:
        try:
            with open(symbol_graph) as f:
                graph_data = json.load(f)
        except Exception:
            graph_data = {}
    g_symbols = graph_data.get('symbols', [])
    if g_symbols:
        sym_list = ', '.join(f'"{s}"' for s in g_symbols)
        output = output.replace('__GRAPH_SYMBOLS__', sym_list)
    else:
        output = output.replace('__GRAPH_SYMBOLS__', '')
    metrics = graph_data.get('metrics') or {}
    if metrics and g_symbols:
        deg_vals = metrics.get('degree', [0.0] * len(g_symbols))
        pr_vals = metrics.get('pagerank', [0.0] * len(g_symbols))
        output = output.replace('__GRAPH_DEGREE__', ', '.join(_fmt(v) for v in deg_vals))
        output = output.replace('__GRAPH_PAGERANK__', ', '.join(_fmt(v) for v in pr_vals))
    else:
        output = output.replace('__GRAPH_DEGREE__', '')
        output = output.replace('__GRAPH_PAGERANK__', '')
    emb_map = graph_data.get('embeddings') or {}
    emb_dim = int(graph_data.get('embedding_dim') or 0)
    if emb_map and emb_dim > 0 and g_symbols:
        emb_rows = []
        for s in g_symbols:
            vec = emb_map.get(s, [0.0] * emb_dim)
            emb_rows.append('{' + ', '.join(_fmt(float(v)) for v in vec) + '}')
        output = output.replace('__GRAPH_EMB_DIM__', str(emb_dim))
        output = output.replace('__GRAPH_EMB_COUNT__', str(len(g_symbols)))
        output = output.replace('__GRAPH_EMB__', ', '.join(emb_rows))
    else:
        output = output.replace('__GRAPH_EMB_DIM__', '0')
        output = output.replace('__GRAPH_EMB_COUNT__', '0')
        output = output.replace('__GRAPH_EMB__', '')

    coint = graph_data.get('cointegration') or {}
    if coint:
        bases: List[str] = []
        peers: List[str] = []
        betas: List[str] = []
        for a, pmap in coint.items():
            for b, stats in pmap.items():
                if isinstance(stats, dict):
                    beta_val = stats.get('beta', 0.0)
                else:
                    beta_val = stats
                bases.append(f'"{a}"')
                peers.append(f'"{b}"')
                betas.append(_fmt(float(beta_val)))
        output = output.replace('__COINT_BASE__', ', '.join(bases))
        output = output.replace('__COINT_PEER__', ', '.join(peers))
        output = output.replace('__COINT_BETA__', ', '.join(betas))
    else:
        output = output.replace('__COINT_BASE__', '')
        output = output.replace('__COINT_PEER__', '')
        output = output.replace('__COINT_BETA__', '')

    rps = base.get('risk_parity_symbols', [])
    rpw = base.get('risk_parity_weights', [])
    if rps and rpw:
        sym_str = ', '.join(f'"{s}"' for s in rps)
        weight_str = ', '.join(_fmt(float(w)) for w in rpw)
    else:
        sym_str = ''
        weight_str = ''
    output = output.replace('__RISK_PARITY_SYMBOLS__', sym_str)
    output = output.replace('__RISK_PARITY_WEIGHTS__', weight_str)

    cal_events = base.get('calendar_events', [])
    if cal_events:
        time_vals = ', '.join(
            datetime.fromisoformat(t).strftime("D'%Y.%m.%d %H:%M'")
            for t, _, _ in cal_events
        )
        impact_vals = ', '.join(_fmt(float(imp)) for _, imp, _ in cal_events)
        id_vals = ', '.join(str(int(eid)) for _, _, eid in cal_events)
    else:
        time_vals = ''
        impact_vals = ''
        id_vals = ''
    event_window = _fmt(base.get('event_window', 60.0))
    output = output.replace('__CALENDAR_TIMES__', time_vals)
    output = output.replace('__CALENDAR_IMPACTS__', impact_vals)
    output = output.replace('__CALENDAR_IDS__', id_vals)
    output = output.replace('__EVENT_WINDOW__', event_window)

    # time-derived features
    feature_map = {
        'hour': 'TimeHour(TimeCurrent())',
        'hour_sin': 'HourSin()',
        'hour_cos': 'HourCos()',
        'dow_sin': 'DowSin()',
        'dow_cos': 'DowCos()',
        'month_sin': 'MonthSin()',
        'month_cos': 'MonthCos()',
        'dom_sin': 'DomSin()',
        'dom_cos': 'DomCos()',
        'spread': 'MarketInfo(SymbolToTrade, MODE_SPREAD)',
        'slippage': 'GetSlippage()',
        'lots': 'Lots',
        'sl_dist': 'SLDistance()',
        'tp_dist': 'TPDistance()',
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
        'event_flag': 'CalendarFlag()',
        'event_impact': 'CalendarImpact()',
        'calendar_event_id': 'CalendarEventId()',
        'book_bid_vol': 'BookBidVol()',
        'book_ask_vol': 'BookAskVol()',
        'book_imbalance': 'BookImbalance()',
        'book_spread': 'BookSpread()',
        'bid_ask_ratio': 'BidAskRatio()',
        'book_imbalance_roll': 'BookImbalanceRoll()',
        'trend_estimate': 'TrendEstimate',
        'trend_variance': 'TrendVariance',
        'duration_sec': 'TradeDuration()',
    }

    if lite_mode:
        feature_map['book_bid_vol'] = '0.0'
        feature_map['book_ask_vol'] = '0.0'
        feature_map['book_imbalance'] = '0.0'
        feature_map['book_spread'] = '0.0'
        feature_map['bid_ask_ratio'] = '0.0'
        feature_map['book_imbalance_roll'] = '0.0'

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
    def resolve_expr(fname: str) -> str | None:
        expr = feature_map.get(fname)
        if expr is not None:
            return expr
        m = tf_pattern.match(fname)
        if m:
            ind, tf = m.groups()
            tf = tf.upper()
            tf_val = tf_const.get(tf, '0')
            used_tfs.add(tf_val)
            return f"{indicator_expr[ind]}[TFIdx({tf_val})]"
        if '*' in fname or '^' in fname:
            parts = fname.split('*')
            sub = []
            for p in parts:
                if '^' in p:
                    base, pow_str = p.split('^', 1)
                    base_expr = resolve_expr(base)
                    if base_expr is None:
                        return None
                    try:
                        pw = int(pow_str)
                    except ValueError:
                        return None
                    if pw == 2:
                        sub.append(f"({base_expr} * {base_expr})")
                    else:
                        sub.append(f"MathPow({base_expr}, {pw})")
                else:
                    base_expr = resolve_expr(p)
                    if base_expr is None:
                        return None
                    sub.append(base_expr)
            return '(' + ' * '.join(sub) + ')'
        if fname.startswith('regime_') and fname[7:].isdigit():
            rid = int(fname.split('_')[1])
            return f'(GetRegime() == {rid} ? 1.0 : 0.0)'
        if fname == 'regime':
            return 'GetRegime()'
        if fname.startswith('ratio_'):
            parts = fname[6:].split('_')
            if len(parts) == 2:
                return f'iClose("{parts[0]}", 0, 0) / iClose("{parts[1]}", 0, 0)'
            if len(parts) == 1:
                return f'iClose(SymbolToTrade, 0, 0) / iClose("{parts[0]}", 0, 0)'
        if fname.startswith('corr_'):
            parts = fname[5:].split('_')
            if len(parts) >= 2:
                sym1 = parts[0]
                sym2 = '_'.join(parts[1:])
                return f'PairCorrelation("{sym1}", "{sym2}")'
            if len(parts) == 1:
                return f'PairCorrelation("{parts[0]}")'
        if fname.startswith('coint_residual_'):
            parts = fname[16:].split('_')
            if len(parts) >= 2:
                sym1 = parts[0]
                sym2 = '_'.join(parts[1:])
                return f'CointegrationResidual("{sym1}", "{sym2}")'
            if len(parts) == 1:
                return f'CointegrationResidual("{parts[0]}")'
        if fname.startswith('exit_reason='):
            reason = fname.split('=', 1)[1]
            return f'ExitReasonFlag("{reason}")'
        if fname == 'graph_degree':
            return 'GraphDegree()'
        if fname == 'graph_pagerank':
            return 'GraphPagerank()'
        if fname.startswith('graph_emb') and fname[9:].isdigit():
            idx_g = int(fname[9:])
            return f'GraphEmbedding({idx_g})'
        if fname.startswith('ae') and fname[2:].isdigit():
            idx_ae = int(fname[2:])
            return f'GetEncodedFeature({idx_ae})'
        if fname == 'news_sentiment':
            return 'GetNewsSentiment()'
        return None

    if hash_size:
        hasher = FeatureHasher(n_features=hash_size, input_type="dict")
        idx_map: dict[int, list[tuple[str, float]]] = {}
        for name in feature_names:
            vec = hasher.transform([{name: 1.0}]).toarray()[0]
            hidx = int(np.flatnonzero(vec)[0])
            sign = 1.0 if vec[hidx] >= 0 else -1.0
            idx_map.setdefault(hidx, []).append((name, sign))
        for idx, items in sorted(idx_map.items()):
            expr_parts = []
            names = []
            for name, sign in items:
                expr = resolve_expr(name)
                if expr is None:
                    logging.error(
                        "Unknown feature '%s'. Please add a matching GetFeature() case to StrategyTemplate.mq4.",
                        name,
                    )
                    raise ValueError(
                        f"Unknown feature '{name}'. Update StrategyTemplate.mq4 with a matching GetFeature() case."
                    )
                names.append(name)
                part = f"({expr})"
                if sign < 0:
                    part = f"-({expr})"
                expr_parts.append(part)
            expr_sum = ' + '.join(expr_parts)
            cases.append(
                f"      case {idx}: // {', '.join(names)}\\n         raw = {expr_sum};\\n         break;",
            )
    else:
        for idx, name in enumerate(feature_names):
            expr = resolve_expr(name)
            if expr is None:
                logging.error(
                    "Unknown feature '%s'. Please add a matching GetFeature() case to StrategyTemplate.mq4.",
                    name,
                )
                raise ValueError(
                    f"Unknown feature '{name}'. Update StrategyTemplate.mq4 with a matching GetFeature() case."
                )
            cases.append(
                f"      case {idx}: // {name}\\n         raw = ({expr});\\n         break;",
            )
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

    # Ensure generated experts reload model parameters when the underlying
    # model file is updated.  We track the last modification timestamp and
    # trigger ``LoadModel`` whenever it changes.
    output = output.replace(
        'int ModelCount = __MODEL_COUNT__;',
        'int ModelCount = __MODEL_COUNT__;'\
        '\nint ModelTimestamp = 0;',
    )
    output = output.replace(
        'bool ok = LoadModel();',
        'bool ok = LoadModel();\n   ModelTimestamp = FileGetInteger(ModelFileName, FILE_MODIFY_DATE);',
    )
    output = output.replace(
        'if(ReloadModelInterval > 0 && TimeCurrent() - LastModelLoad >= ReloadModelInterval)',
        'datetime ts = FileGetInteger(ModelFileName, FILE_MODIFY_DATE);\n   if(ts != ModelTimestamp)',
    )
    output = output.replace(
        'LastModelLoad = TimeCurrent();',
        'ModelTimestamp = ts;',
    )

    threshold = base.get('threshold', 0.5)
    output = output.replace('__THRESHOLD__', _fmt(threshold))
    metrics = base.get('teacher_metrics') or (
        {"accuracy": base.get('teacher_accuracy')} if base.get('teacher_accuracy') is not None else {}
    )
    if metrics:
        for key, val in metrics.items():
            if val is not None:
                output += f"\n// Teacher {key}: {_fmt(val)}"
        output += "\n"
    if base.get('student_coefficients'):
        coeffs_comment = ', '.join(_fmt(c) for c in base['student_coefficients'])
        output += f"// Student coefficients: {coeffs_comment}\n"
        output += f"// Student intercept: {_fmt(base.get('student_intercept', 0.0))}\n"
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
    p.add_argument('--lite-mode', action='store_true', default=None,
                   help='Force lite mode; otherwise inferred from model.json')
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
