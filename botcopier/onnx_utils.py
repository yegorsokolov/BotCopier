"""Utility helpers for exporting models to ONNX and validating them.

This module provides convenience functions to convert scikit-learn and
PyTorch models to the ONNX format.  The resulting graphs are validated using
``onnxruntime`` to ensure operator and shape compatibility.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:  # Optional dependencies
    import onnx
    import onnxruntime as ort
except Exception:  # pragma: no cover - optional import failure
    onnx = None  # type: ignore
    ort = None  # type: ignore

try:  # Scikit-learn ONNX conversion
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    from sklearn.base import BaseEstimator
except Exception:  # pragma: no cover - optional import failure
    convert_sklearn = None  # type: ignore
    FloatTensorType = None  # type: ignore
    BaseEstimator = object

try:  # PyTorch ONNX export
    import torch
except Exception:  # pragma: no cover - optional import failure
    torch = None  # type: ignore


def _runtime_check(model_path: Path, sample: np.ndarray) -> None:
    """Run ``onnx.checker`` and an ``onnxruntime`` session.

    Parameters
    ----------
    model_path:
        Path to the ONNX graph.
    sample:
        Sample feature array used to execute a dummy forward pass.
    """

    if onnx is None or ort is None:  # pragma: no cover - optional
        return
    onnx_model = onnx.load(str(model_path))
    onnx.checker.check_model(onnx_model)
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    arr = sample.astype(np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    sess.run(None, {inp: arr[:1]})


def convert_sklearn_model(model: BaseEstimator, sample: np.ndarray, out: Path, *, opset: int = 17) -> None:
    """Convert a scikit-learn model to ONNX format and validate it."""

    if convert_sklearn is None or FloatTensorType is None:  # pragma: no cover - optional
        raise RuntimeError("skl2onnx is required to convert scikit-learn models")
    initial_types = [("input", FloatTensorType([None, sample.shape[1]]))]
    onx = convert_sklearn(model, initial_types=initial_types, target_opset=opset)
    out.write_bytes(onx.SerializeToString())
    _runtime_check(out, sample)


def convert_torch_model(model: "torch.nn.Module", sample: np.ndarray, out: Path, *, opset: int = 17) -> None:
    """Export a PyTorch model to ONNX format and validate it."""

    if torch is None:  # pragma: no cover - optional
        raise RuntimeError("PyTorch is required to export models")
    model.eval()
    dummy = torch.as_tensor(sample[:1], dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        str(out),
        opset_version=opset,
        input_names=["input"],
        output_names=["output"],
    )
    _runtime_check(out, sample)


def export_model(model: Any, sample: np.ndarray, out: Path, *, opset: int = 17) -> Path:
    """Export ``model`` to ONNX format.

    The function dispatches to the appropriate converter based on the model
    type (scikit-learn or PyTorch).  ``sample`` is a representative input used
    both for schema generation and for a dummy runtime check.

    Returns
    -------
    Path
        The path to the written ONNX file.
    """

    if torch is not None and isinstance(model, torch.nn.Module):
        convert_torch_model(model, sample, out, opset=opset)
    elif isinstance(model, BaseEstimator):
        convert_sklearn_model(model, sample, out, opset=opset)
    else:
        raise TypeError(f"Unsupported model type for ONNX export: {type(model)!r}")
    return out


__all__ = [
    "convert_sklearn_model",
    "convert_torch_model",
    "export_model",
]
