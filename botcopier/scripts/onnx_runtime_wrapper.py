#!/usr/bin/env python3
"""Run ONNX models with optional GPU acceleration."""

import argparse
from pathlib import Path
import numpy as np

try:
    import onnxruntime as ort
except Exception as exc:  # pragma: no cover - import error
    ort = None
    _IMPORT_ERROR = exc


class OnnxModel:
    """Lightweight wrapper around onnxruntime.InferenceSession."""

    def __init__(self, model_path: Path, use_gpu: bool = False):
        if ort is None:
            raise RuntimeError(f"onnxruntime is required: {_IMPORT_ERROR}")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, features: np.ndarray) -> np.ndarray:
        arr = features.astype(np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return self.session.run(None, {self.input_name: arr})[0]


def main() -> None:
    p = argparse.ArgumentParser(description="Execute ONNX model")
    p.add_argument("model", type=Path, help="path to model.onnx")
    p.add_argument(
        "features",
        help="comma separated feature values, e.g. '0.1,0.2,0.3'",
    )
    p.add_argument("--gpu", action="store_true", help="use GPU execution provider if available")
    args = p.parse_args()
    model = OnnxModel(args.model, use_gpu=args.gpu)
    feats = np.array([float(x) for x in args.features.split(",")], dtype=np.float32)
    preds = model.predict(feats)
    print(",".join(str(float(x)) for x in preds.reshape(-1)))


if __name__ == "__main__":
    main()
