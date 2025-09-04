import json
import io
import gzip
import struct
from pathlib import Path
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.generate_mql4_from_model import generate


def _parse_model_bin(path: Path):
    with gzip.open(path, 'rb') as f:
        data = f.read()
    buf = io.BytesIO(data)
    def rint():
        return struct.unpack('<i', buf.read(4))[0]
    mc = rint()
    fc = rint()
    coeff = np.frombuffer(buf.read(4 * mc * fc), dtype='<f4').reshape(mc, fc)
    sc = rint()
    sd = rint()
    buf.seek(4 * sc * sd, 1)
    gc = rint()
    gd = rint()
    buf.seek(4 * gc * gd, 1)
    tl = rint()
    thr = np.frombuffer(buf.read(4 * tl), dtype='<f4') if tl else np.empty(0)
    rl = rint()
    buf.seek(4 * rl, 1)
    sl = rint()
    buf.seek(4 * sl, 1)
    return coeff, thr


def test_quantized_roundtrip(tmp_path: Path):
    model = {
        "model_id": "qt",
        "magic": 1,
        "coefficients": [0.1, -0.2],
        "intercept": 0.05,
        "hourly_thresholds": [0.4] * 24,
        "feature_names": ["hour", "spread"],
        "symbol_embeddings": {"EURUSD": [0.1, 0.2]},
        "regime_thresholds": [0.1, 0.2],
        "symbol_thresholds": {"EURUSD": 0.7},
    }
    model_file = tmp_path / "model.json"
    model_file.write_text(json.dumps(model))
    out_dir = tmp_path / "out"
    generate(model_file, out_dir)
    bin_path = out_dir / "model.bin"
    assert bin_path.exists()
    coeff_q, thr_q = _parse_model_bin(bin_path)
    x = np.array([1.5, -0.7])
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))
    pred_orig = sigmoid(np.dot(model["coefficients"], x) + model["intercept"])
    pred_q = sigmoid(np.dot(coeff_q[0], x) + model["intercept"])
    assert abs(pred_orig - pred_q) < 0.01
    assert np.allclose(thr_q[:24], 0.4, atol=1e-6)
