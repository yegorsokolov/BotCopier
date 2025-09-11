#!/usr/bin/env python3
"""Run inference on an exported ONNX model."""

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort


def main() -> None:
    p = argparse.ArgumentParser(description="Predict using ONNX model")
    p.add_argument("model", type=Path, help="Path to model.onnx")
    p.add_argument("features", help="Comma separated feature values")
    args = p.parse_args()

    sess = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    feats = np.array([float(x) for x in args.features.split(",")], dtype=np.float32)
    if feats.ndim == 1:
        feats = feats.reshape(1, -1)
    preds = sess.run(None, {input_name: feats})[0]
    print(",".join(str(float(x)) for x in np.asarray(preds).ravel()))


if __name__ == "__main__":
    main()
