import json
import math
import time
import logging
import numbers
from datetime import datetime
from itertools import combinations_with_replacement
from pathlib import Path
from typing import Iterable

import numpy as np
import psutil

try:  # Numba is optional
    from numba import jit
    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - fallback when numba isn't installed
    NUMBA_AVAILABLE = False

    def jit(*args, **kwargs):  # type: ignore
        """Fallback ``jit`` decorator when Numba is unavailable."""
        def wrapper(func):
            return func

        return wrapper


def _sma(values, window):
    """Simple moving average for the last ``window`` values."""
    if not values:
        return 0.0
    w = min(window, len(values))
    return float(sum(values[-w:]) / w)


def _atr(values, period):
    """Average true range over ``period`` price changes."""
    if len(values) < 2:
        return 0.0
    diffs = np.abs(np.diff(values[-(period + 1) :]))
    if len(diffs) == 0:
        return 0.0
    w = min(period, len(diffs))
    return float(diffs[-w:].mean())


def _bollinger(values, window, dev=2.0):
    """Return Bollinger Bands for the last ``window`` values."""
    if not values:
        return 0.0, 0.0, 0.0
    w = min(window, len(values))
    arr = np.array(values[-w:])
    sma = arr.mean()
    std = arr.std(ddof=0)
    upper = sma + dev * std
    lower = sma - dev * std
    return float(upper), float(sma), float(lower)


def _safe_float(val, default=0.0):
    """Convert ``val`` to float, treating ``None``/NaN as ``default``."""
    try:
        f = float(val)
        if math.isnan(f):
            return default
        return f
    except Exception:
        return default


def _rsi(values, period):
    """Very small RSI implementation on ``values``."""
    if len(values) < 2:
        return 50.0
    deltas = np.diff(values[-(period + 1) :])
    gains = deltas[deltas > 0].sum()
    losses = -deltas[deltas < 0].sum()
    if losses == 0:
        return 100.0
    rs = gains / losses
    return float(100 - (100 / (1 + rs)))


def _macd_update(state, price, short=12, long=26, signal=9):
    """Update MACD EMA state with ``price`` and return (macd, signal)."""
    alpha_short = 2 / (short + 1)
    alpha_long = 2 / (long + 1)
    alpha_signal = 2 / (signal + 1)

    ema_short = state.get("ema_short")
    ema_long = state.get("ema_long")
    ema_signal = state.get("ema_signal")

    ema_short = price if ema_short is None else alpha_short * price + (1 - alpha_short) * ema_short
    ema_long = price if ema_long is None else alpha_long * price + (1 - alpha_long) * ema_long
    macd = ema_short - ema_long
    ema_signal = macd if ema_signal is None else alpha_signal * macd + (1 - alpha_signal) * ema_signal

    state["ema_short"] = ema_short
    state["ema_long"] = ema_long
    state["ema_signal"] = ema_signal

    return macd, ema_signal


@jit(nopython=True)
def _stochastic_k_nb(price: float, prices: np.ndarray) -> float:
    low = prices.min()
    high = prices.max()
    if high == low:
        return 0.0
    return (price - low) / (high - low) * 100.0


def _stochastic_update(state, price, k_period=14, d_period=3):
    """Update and return stochastic %K and %D values."""
    prices = state.setdefault("prices", [])
    prices.append(price)
    if len(prices) > k_period:
        del prices[0]
    arr = np.array(prices, dtype=np.float64)
    if NUMBA_AVAILABLE:
        k = float(_stochastic_k_nb(price, arr))
    else:  # fallback to numpy implementation
        low = float(arr.min())
        high = float(arr.max())
        k = 0.0 if high == low else (price - low) / (high - low) * 100.0

    k_history = state.setdefault("k_values", [])
    k_history.append(k)
    if len(k_history) > d_period:
        del k_history[0]

    if NUMBA_AVAILABLE:
        d = float(np.mean(np.array(k_history, dtype=np.float64)))
    else:
        d = float(sum(k_history) / len(k_history))
    return float(k), d


@jit(nopython=True)
def _adx_dx_nb(plus_dm: np.ndarray, minus_dm: np.ndarray, tr: np.ndarray) -> float:
    atr = tr.mean()
    if atr == 0:
        di_plus = 0.0
        di_minus = 0.0
    else:
        di_plus = 100.0 * plus_dm.mean() / atr
        di_minus = 100.0 * minus_dm.mean() / atr
    denom = di_plus + di_minus
    return 0.0 if denom == 0 else 100.0 * abs(di_plus - di_minus) / denom


def _adx_update(state, price, period=14):
    """Update ADX state with ``price`` and return current ADX."""
    prev = state.get("prev_price")
    state["prev_price"] = price
    if prev is None:
        return 0.0

    up_move = price - prev if price > prev else 0.0
    down_move = prev - price if price < prev else 0.0
    tr = abs(price - prev)

    plus_dm = state.setdefault("plus_dm", [])
    minus_dm = state.setdefault("minus_dm", [])
    tr_list = state.setdefault("tr", [])
    dx_list = state.setdefault("dx", [])

    plus_dm.append(up_move)
    minus_dm.append(down_move)
    tr_list.append(tr)
    if len(plus_dm) > period:
        del plus_dm[0]
    if len(minus_dm) > period:
        del minus_dm[0]
    if len(tr_list) > period:
        del tr_list[0]

    arr_plus = np.array(plus_dm, dtype=np.float64)
    arr_minus = np.array(minus_dm, dtype=np.float64)
    arr_tr = np.array(tr_list, dtype=np.float64)

    if NUMBA_AVAILABLE:
        dx = float(_adx_dx_nb(arr_plus, arr_minus, arr_tr))
    else:
        atr = arr_tr.mean()
        if atr == 0:
            di_plus = di_minus = 0.0
        else:
            di_plus = 100.0 * (arr_plus.mean()) / atr
            di_minus = 100.0 * (arr_minus.mean()) / atr
        denom = di_plus + di_minus
        dx = 0.0 if denom == 0 else 100.0 * abs(di_plus - di_minus) / denom

    dx_list.append(dx)
    if len(dx_list) > period:
        del dx_list[0]

    if NUMBA_AVAILABLE:
        adx = float(np.mean(np.array(dx_list, dtype=np.float64)))
    else:
        adx = float(sum(dx_list) / len(dx_list))
    return adx


@jit(nopython=True)
def _rolling_corr_nb(arr1: np.ndarray, arr2: np.ndarray) -> float:
    n = arr1.size
    mean1 = 0.0
    mean2 = 0.0
    for i in range(n):
        mean1 += arr1[i]
        mean2 += arr2[i]
    mean1 /= n
    mean2 /= n
    num = 0.0
    denom1 = 0.0
    denom2 = 0.0
    for i in range(n):
        d1 = arr1[i] - mean1
        d2 = arr2[i] - mean2
        num += d1 * d2
        denom1 += d1 * d1
        denom2 += d2 * d2
    if denom1 == 0.0 or denom2 == 0.0:
        return 0.0
    return num / math.sqrt(denom1 * denom2)


def _rolling_corr(a, b, window=5):
    """Return correlation of the last ``window`` points of ``a`` and ``b``."""
    if not a or not b:
        return 0.0
    w = min(len(a), len(b), window)
    if w < 2:
        return 0.0
    arr1 = np.array(a[-w:], dtype=np.float64)
    arr2 = np.array(b[-w:], dtype=np.float64)
    if arr1.std(ddof=0) == 0 or arr2.std(ddof=0) == 0:
        return 0.0
    if NUMBA_AVAILABLE:
        return float(_rolling_corr_nb(arr1, arr2))
    return float(np.corrcoef(arr1, arr2)[0, 1])


def _extract_features(
    rows,
    use_sma=False,
    sma_window=5,
    use_rsi=False,
    rsi_period=14,
    use_macd=False,
    use_atr=False,
    atr_period=14,
    use_bollinger=False,
    boll_window=20,
    use_stochastic=False,
    use_adx=False,
    use_volume=False,
    use_orderbook=False,
    use_dom=False,
    volatility=None,
    higher_timeframes=None,
    poly_degree: int = 2,
    *,
    corr_map=None,
    extra_price_series=None,
    symbol_graph: dict | str | Path | None = None,
    corr_window: int = 5,
    encoder: dict | None = None,
    calendar_events: list[tuple[datetime, float, int]] | None = None,
    event_window: float = 60.0,
    perf_budget: float | None = None,
    news_sentiment: dict[str, list[tuple[datetime, float]]] | None = None,
):
    feature_dicts = []
    labels = []
    sl_targets = []
    tp_targets = []
    lot_targets = []
    prices = []
    hours = []
    times = []
    imbalance_history: list[float] = []
    corr_map = corr_map or {}
    extra_series = extra_price_series or {}
    calendar_events = calendar_events or []
    price_map = {sym: [] for sym in extra_series.keys()}
    for base, peers in corr_map.items():
        price_map.setdefault(base, [])
        for p in peers:
            price_map.setdefault(p, [])
    macd_state = {}
    higher_timeframes = [str(tf).upper() for tf in (higher_timeframes or [])]
    tf_map = {
        "M1": 1,
        "M5": 5,
        "M15": 15,
        "M30": 30,
        "H1": 60,
        "H4": 240,
        "D1": 1440,
        "W1": 10080,
        "MN1": 43200,
    }
    tf_secs = {tf: tf_map.get(tf, 60) * 60 for tf in higher_timeframes}
    tf_prices = {tf: [] for tf in higher_timeframes}
    tf_macd_state = {tf: {} for tf in higher_timeframes}
    tf_macd = {tf: 0.0 for tf in higher_timeframes}
    tf_macd_sig = {tf: 0.0 for tf in higher_timeframes}
    tf_last_bin = {tf: None for tf in higher_timeframes}
    tf_prev_price = {tf: None for tf in higher_timeframes}
    stoch_state = {}
    adx_state = {}

    pair_weights: dict[tuple[str, str], float] = {}
    sym_metrics: dict[str, dict[str, float]] = {}
    sym_embeddings: dict[str, list[float]] = {}
    coint_stats: dict[tuple[str, str], dict[str, float]] = {}
    if symbol_graph:
        try:
            if not isinstance(symbol_graph, dict):
                with open(symbol_graph) as f_g:
                    graph_params = json.load(f_g)
            else:
                graph_params = symbol_graph
            symbols = graph_params.get("symbols", [])
            edge_index = graph_params.get("edge_index", [])
            weights = graph_params.get("edge_weight", [])
            for (i, j), w in zip(edge_index, weights):
                if i < len(symbols) and j < len(symbols):
                    a = symbols[i]
                    b = symbols[j]
                    pair_weights[(a, b)] = float(w)

            # Prefer consolidated per-symbol node data when available.
            nodes = graph_params.get("nodes") or {}
            if nodes:
                for sym, vals in nodes.items():
                    for m_name, m_val in vals.items():
                        if isinstance(m_val, list):
                            try:
                                sym_embeddings[sym] = [float(v) for v in m_val]
                            except Exception:
                                continue
                        else:
                            sym_metrics.setdefault(sym, {})[m_name] = float(m_val)
            else:
                metrics = graph_params.get("metrics", {})
                for m_name, vals in metrics.items():
                    for i, sym in enumerate(symbols):
                        sym_metrics.setdefault(sym, {})[m_name] = float(vals[i])
                emb_map = graph_params.get("embeddings", {})
                for sym, emb in emb_map.items():
                    try:
                        sym_embeddings[sym] = [float(v) for v in emb]
                    except Exception:
                        continue
            coint = graph_params.get("cointegration", {})
            for base, peers in coint.items():
                for peer, stats in peers.items():
                    if isinstance(stats, dict):
                        beta = float(stats.get("beta", 0.0))
                        pval = float(stats.get("pvalue", 1.0))
                    else:
                        beta = float(stats)
                        pval = float("nan")
                    if not np.isnan(pval) and pval > 0.05:
                        continue
                    coint_stats[(base, peer)] = {"beta": beta, "pvalue": pval}
            if coint_stats:
                valid_pairs = set(coint_stats.keys())
                pair_weights = {
                    (a, b): w for (a, b), w in pair_weights.items() if (a, b) in valid_pairs
                }
        except Exception:
            pair_weights = {}
            sym_metrics = {}
            sym_embeddings = {}
            coint_stats = {}
    enc_window = int(encoder.get("window")) if encoder else 0
    enc_weights = (
        np.array(encoder.get("weights", []), dtype=float) if encoder else np.empty((0, 0))
    )
    enc_centers = (
        np.array(encoder.get("centers", []), dtype=float) if encoder else np.empty((0, 0))
    )
    news_indices = {sym: 0 for sym in (news_sentiment or {}).keys()}
    row_idx = 0

    start_time = time.perf_counter()
    psutil.cpu_percent(interval=None)
    heavy_order = [
        "multi_tf",
        "order_book",
        "use_adx",
        "use_stochastic",
        "use_bollinger",
        "use_atr",
    ]

    for r in rows:
        if r.get("action", "").upper() != "OPEN":
            continue
        if str(r.get("is_anomaly", "0")).lower() in ("1", "true", "yes"):
            continue

        t = r["event_time"]
        if not isinstance(t, datetime):
            parsed = None
            for fmt in ("%Y.%m.%d %H:%M:%S", "%Y.%m.%d %H:%M"):
                try:
                    parsed = datetime.strptime(str(t), fmt)
                    break
                except Exception:
                    continue
            if parsed is None:
                continue
            t = parsed

        times.append(t)
        order_type = int(float(r.get("order_type", 0)))
        label = 1 if order_type == 0 else 0  # buy=1, sell=0

        price = _safe_float(r.get("price", 0))
        sl = _safe_float(r.get("sl", 0))
        tp = _safe_float(r.get("tp", 0))
        lots = _safe_float(r.get("lots", 0))
        profit = _safe_float(r.get("profit", 0))

        for tf in higher_timeframes:
            tf_bin = int(t.timestamp() // tf_secs[tf])
            if tf_last_bin[tf] is None:
                tf_last_bin[tf] = tf_bin
            elif tf_bin != tf_last_bin[tf]:
                if tf_prev_price[tf] is not None:
                    tf_prices[tf].append(tf_prev_price[tf])
                    if use_macd:
                        tf_macd[tf], tf_macd_sig[tf] = _macd_update(
                            tf_macd_state[tf], tf_prev_price[tf]
                        )
                tf_last_bin[tf] = tf_bin
            tf_prev_price[tf] = price

        symbol = r.get("symbol", "")
        sym_prices = price_map.setdefault(symbol, [])

        spread = _safe_float(r.get("spread", 0))
        slippage = _safe_float(r.get("slippage", 0))
        account_equity = _safe_float(r.get("equity", 0))
        margin_level = _safe_float(r.get("margin_level", 0))
        commission = _safe_float(r.get("commission", 0))
        swap = _safe_float(r.get("swap", 0))
        net_profit = profit - commission - swap
        trend_est = _safe_float(r.get("trend_estimate", 0))
        trend_var = _safe_float(r.get("trend_variance", 0))

        hour_angle = 2 * math.pi * t.hour / 24
        hour_sin = math.sin(hour_angle)
        hour_cos = math.cos(hour_angle)
        dow_angle = 2 * math.pi * t.weekday() / 7
        dow_sin = math.sin(dow_angle)
        dow_cos = math.cos(dow_angle)

        month = t.month
        month_angle = 2 * math.pi * (month - 1) / 12
        month_sin = math.sin(month_angle)
        month_cos = math.cos(month_angle)

        dom = t.day
        if use_dom:
            dom_angle = 2 * math.pi * (dom - 1) / 31
            dom_sin = math.sin(dom_angle)
            dom_cos = math.cos(dom_angle)

        sl_dist = _safe_float(r.get("sl_dist", sl - price))
        tp_dist = _safe_float(r.get("tp_dist", tp - price))
        sl_hit = _safe_float(r.get("sl_hit_dist", 0.0))
        tp_hit = _safe_float(r.get("tp_hit_dist", 0.0))
        exit_reason = str(r.get("exit_reason", "") or "").upper()
        duration_sec = int(float(r.get("duration_sec", 0) or 0))

        feat = {
            "symbol": symbol,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "dow_sin": dow_sin,
            "dow_cos": dow_cos,
            "month_sin": month_sin,
            "month_cos": month_cos,
            "lots": lots,
            "profit": profit,
            "net_profit": net_profit,
            "sl_dist": sl_dist,
            "tp_dist": tp_dist,
            "sl_hit_dist": sl_hit,
            "tp_hit_dist": tp_hit,
            "spread": spread,
            "slippage": slippage,
            "equity": account_equity,
            "margin_level": margin_level,
            "commission": commission,
            "swap": swap,
            "trend_estimate": trend_est,
            "trend_variance": trend_var,
            "exit_reason": exit_reason,
            "duration_sec": duration_sec,
            "event_id": int(float(r.get("event_id", 0) or 0)),
        }

        if use_dom:
            feat["dom_sin"] = dom_sin
            feat["dom_cos"] = dom_cos

        bid_vol_raw = float(r.get("book_bid_vol", 0) or 0)
        ask_vol_raw = float(r.get("book_ask_vol", 0) or 0)
        total_vol = bid_vol_raw + ask_vol_raw
        if total_vol > 0:
            bid_vol = bid_vol_raw / total_vol
            ask_vol = ask_vol_raw / total_vol
            imbalance = (bid_vol - ask_vol)
        else:
            bid_vol = 0.0
            ask_vol = 0.0
            imbalance = 0.0
        feat.update(
            {
                "book_bid_vol": bid_vol,
                "book_ask_vol": ask_vol,
                "book_imbalance": imbalance,
            }
        )
        if use_orderbook:
            imbalance_history.append(imbalance)
            window = 5
            roll = sum(imbalance_history[-window:]) / min(len(imbalance_history), window)
            spread_vol = ask_vol - bid_vol
            ratio = bid_vol / (ask_vol + 1e-9)
            feat.update(
                {
                    "book_spread": spread_vol,
                    "bid_ask_ratio": ratio,
                    "book_imbalance_roll": roll,
                }
            )

        flag = 0.0
        impact_val = 0.0
        event_id_val = 0
        for ev_time, ev_imp, ev_id in calendar_events:
            if abs((t - ev_time).total_seconds()) <= event_window * 60.0:
                flag = 1.0
                if ev_imp > impact_val:
                    impact_val = ev_imp
                    event_id_val = int(ev_id)
        feat["event_flag"] = flag
        feat["event_impact"] = impact_val
        feat["calendar_event_id"] = event_id_val
        if event_id_val:
            feat[f"event_id_{event_id_val}"] = 1.0

        if use_volume:
            feat["volume"] = float(r.get("volume", 0) or 0)

        if volatility is not None:
            key = t.strftime("%Y-%m-%d %H")
            vol = volatility.get(key)
            if vol is None:
                key = t.strftime("%Y-%m-%d")
                vol = volatility.get(key, 0.0)
            feat["volatility"] = float(vol)

        if use_sma:
            feat["sma"] = _sma(prices, sma_window)

        if use_rsi:
            feat["rsi"] = _rsi(prices, rsi_period)

        if use_macd:
            macd, signal = _macd_update(macd_state, price)
            feat["macd"] = macd
            feat["macd_signal"] = signal

        if use_atr:
            feat["atr"] = _atr(prices, atr_period)

        if use_bollinger:
            upper, mid, lower = _bollinger(prices, boll_window)
            feat["bollinger_upper"] = upper
            feat["bollinger_middle"] = mid
            feat["bollinger_lower"] = lower

        if use_stochastic:
            k, d_val = _stochastic_update(stoch_state, price)
            feat["stochastic_k"] = k
            feat["stochastic_d"] = d_val

        if use_adx:
            feat["adx"] = _adx_update(adx_state, price)

        for tf in higher_timeframes:
            prices_tf = tf_prices.get(tf, [])
            if use_sma:
                feat[f"sma_{tf}"] = _sma(prices_tf, sma_window)
            if use_rsi:
                feat[f"rsi_{tf}"] = _rsi(prices_tf, rsi_period)
            if use_macd:
                feat[f"macd_{tf}"] = tf_macd.get(tf, 0.0)
                feat[f"macd_signal_{tf}"] = tf_macd_sig.get(tf, 0.0)

        base_prices = price_map.get(symbol, [])
        base_seq = base_prices + [price]
        for peer in corr_map.get(symbol, []):
            peer_prices = price_map.get(peer, [])
            peer_seq = list(peer_prices)
            series = extra_series.get(peer)
            if series is not None and row_idx < len(series):
                peer_seq.append(float(series[row_idx]))
            corr = _rolling_corr(base_seq, peer_seq, corr_window)
            ratio = 0.0
            if peer_seq and peer_seq[-1] != 0:
                ratio = base_seq[-1] / peer_seq[-1]
            feat[f"corr_{symbol}_{peer}"] = corr
            feat[f"ratio_{symbol}_{peer}"] = ratio
            stats = coint_stats.get((symbol, peer))
            if stats is not None and peer_seq:
                beta = stats.get("beta", 0.0)
                resid = base_seq[-1] - beta * peer_seq[-1]
                feat[f"coint_residual_{symbol}_{peer}"] = resid

        if pair_weights:
            for (a, b), w in pair_weights.items():
                if a == symbol:
                    feat[f"corr_{symbol}_{b}"] = w
        if sym_metrics:
            mvals = sym_metrics.get(symbol)
            if mvals:
                for m_name, m_val in mvals.items():
                    feat[f"graph_{m_name}"] = m_val
        if sym_embeddings:
            emb = sym_embeddings.get(symbol)
            if emb:
                for i, v in enumerate(emb):
                    feat[f"graph_emb{i}"] = v

        if enc_window > 0 and enc_weights.size > 0:
            seq = (prices + [price])[-(enc_window + 1) :]
            if len(seq) < enc_window + 1:
                seq = [seq[0]] * (enc_window + 1 - len(seq)) + seq
            deltas = np.diff(seq)
            vals = deltas.dot(enc_weights)
            for i, v in enumerate(vals):
                feat[f"ae{i}"] = float(v)
        if enc_centers.size > 0:
            d = ((enc_centers - vals) ** 2).sum(axis=1)
            feat["regime"] = float(int(np.argmin(d)))

        if news_sentiment is not None:
            sent_list = news_sentiment.get(symbol)
            score = 0.0
            if sent_list:
                idx = news_indices.get(symbol, 0)
                while idx + 1 < len(sent_list) and sent_list[idx + 1][0] <= t:
                    idx += 1
                news_indices[symbol] = idx
            if sent_list[idx][0] <= t:
                score = float(sent_list[idx][1])
            feat["news_sentiment"] = score

        if poly_degree and poly_degree > 1:
            poly_exclude = {
                "profit",
                "net_profit",
                "commission",
                "swap",
                "sl_hit_dist",
                "tp_hit_dist",
                "symbol",
                "event_id",
                "calendar_event_id",
            }
            numeric_keys = [
                k
                for k, v in feat.items()
                if isinstance(v, numbers.Real) and k not in poly_exclude
            ]
            for deg in range(2, poly_degree + 1):
                for combo in combinations_with_replacement(sorted(numeric_keys), deg):
                    val = 1.0
                    for k in combo:
                        val *= feat[k]
                    counts = {}
                    for k in combo:
                        counts[k] = counts.get(k, 0) + 1
                    name_parts = []
                    for k in sorted(counts.keys()):
                        c = counts[k]
                        if c == 1:
                            name_parts.append(k)
                        else:
                            name_parts.append(f"{k}^{c}")
                    name = "*".join(name_parts)
                    feat[name] = val

        prices.append(price)
        sym_prices.append(price)
        for sym, series in extra_series.items():
            if sym == symbol:
                continue
            if row_idx < len(series):
                price_map.setdefault(sym, []).append(float(series[row_idx]))
        # Propagate synthetic flag for downstream weighting if present
        if str(r.get("synthetic", "0")).lower() in ("1", "true", "yes"):
            feat["synthetic"] = 1.0
        feature_dicts.append(feat)
        labels.append(label)
        sl_targets.append(sl_dist)
        tp_targets.append(tp_dist)
        lot_targets.append(lots)
        hours.append(t.hour)
        row_idx += 1

        if perf_budget is not None:
            elapsed = time.perf_counter() - start_time
            load = psutil.cpu_percent(interval=None)
            while heavy_order and (
                elapsed > perf_budget * row_idx or load > 90.0
            ):
                feat_name = heavy_order.pop(0)
                if feat_name == "use_atr":
                    use_atr = False
                elif feat_name == "use_bollinger":
                    use_bollinger = False
                elif feat_name == "use_stochastic":
                    use_stochastic = False
                elif feat_name == "use_adx":
                    use_adx = False
                elif feat_name == "order_book":
                    use_orderbook = False
                elif feat_name == "multi_tf":
                    higher_timeframes = []
                    tf_prices.clear()
                    tf_macd_state.clear()
                    tf_macd.clear()
                    tf_macd_sig.clear()
                    tf_last_bin.clear()
                    tf_prev_price.clear()
                logging.info("Disabling %s due to performance budget", feat_name)
                elapsed = time.perf_counter() - start_time
                load = psutil.cpu_percent(interval=None)

    enabled_feats = []
    enabled_feats.append("month")
    if use_sma:
        enabled_feats.append("sma")
    if use_rsi:
        enabled_feats.append("rsi")
    if use_macd:
        enabled_feats.append("macd")
    if use_atr:
        enabled_feats.append("atr")
    if use_bollinger:
        enabled_feats.append("bollinger")
    if use_stochastic:
        enabled_feats.append("stochastic")
    if use_adx:
        enabled_feats.append("adx")
    if use_dom:
        enabled_feats.append("dom")
    if higher_timeframes:
        enabled_feats.extend(f"tf_{tf}" for tf in higher_timeframes)
    if news_sentiment is not None:
        enabled_feats.append("news_sentiment")
    if sym_metrics:
        enabled_feats.append("graph_metrics")
    if sym_embeddings:
        enabled_feats.append("graph_embeddings")
    logging.info("Enabled features: %s", sorted(enabled_feats))

    return (
        feature_dicts,
        np.array(labels),
        np.array(sl_targets),
        np.array(tp_targets),
        np.array(hours, dtype=int),
        np.array(lot_targets),
        np.array(times, dtype="datetime64[s]"),
        price_map,
    )


