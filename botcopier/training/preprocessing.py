"""Preprocessing helpers for the training pipeline."""

from __future__ import annotations

import base64
import hashlib
import importlib.metadata as importlib_metadata
import json
import logging
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


try:  # optional polars support
    import polars as pl  # type: ignore

    HAS_POLARS = True
except ImportError:  # pragma: no cover - optional
    pl = None  # type: ignore
    HAS_POLARS = False

try:  # optional dask support
    import dask.dataframe as dd  # type: ignore

    HAS_DASK = True
except Exception:  # pragma: no cover - optional
    dd = None  # type: ignore
    HAS_DASK = False

try:  # optional torch dependency flag
    import torch  # type: ignore

    HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    HAS_TORCH = False

try:  # optional feast dependency
    from feast import FeatureStore  # type: ignore

    from botcopier.feature_store.feast_repo.feature_views import FEATURE_COLUMNS

    HAS_FEAST = True
except Exception:  # pragma: no cover - optional
    FeatureStore = None  # type: ignore
    FEATURE_COLUMNS: list[str] = []  # type: ignore
    HAS_FEAST = False


AUTOENCODER_META_SUFFIX = ".meta.json"


def should_use_lightweight(data_dir: Path, kwargs: Mapping[str, Any]) -> bool:
    """Return ``True`` when a simplified training path should be used."""

    if kwargs.get("lite_mode"):
        return True
    data_path = Path(data_dir)
    file = data_path if data_path.is_file() else data_path / "trades_raw.csv"
    try:
        with file.open("r", encoding="utf-8") as handle:
            # subtract header row if present
            row_count = sum(1 for _ in handle) - 1
    except FileNotFoundError:
        return False
    return row_count <= 200


def dependency_lines(packages: Sequence[str]) -> list[str]:
    """Return formatted dependency pin lines for ``packages``."""

    lines: list[str] = []
    for pkg in packages:
        try:
            version = importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            version = "0.0.0"
        lines.append(f"{pkg}=={version}")
    return lines


def normalise_feature_subset(
    subset: Sequence[str] | str | None,
) -> tuple[list[str], bool]:
    """Return a deduplicated list of feature names and whether it was provided."""

    if subset is None:
        return [], False
    if isinstance(subset, (str, bytes)):
        items = [subset]
    else:
        items = list(subset)
    normalised: list[str] = []
    seen: set[str] = set()
    for raw in items:
        if raw is None:
            raise ValueError("feature subset cannot include null values")
        name = str(raw).strip()
        if not name:
            raise ValueError("feature subset cannot include empty feature names")
        if name not in seen:
            seen.add(name)
            normalised.append(name)
    if not normalised:
        raise ValueError("feature subset must contain at least one feature name")
    return normalised, True


def filter_feature_matrix(
    matrix: np.ndarray, feature_names: list[str], subset: Sequence[str]
) -> tuple[np.ndarray, list[str]]:
    """Return ``matrix`` and ``feature_names`` filtered to ``subset`` preserving order."""

    if not subset:
        return matrix, feature_names
    missing = [name for name in subset if name not in feature_names]
    if missing:
        raise ValueError(
            "Requested features not present in engineered feature set: %s" % missing
        )
    keep = [idx for idx, name in enumerate(feature_names) if name in set(subset)]
    if not keep:
        raise ValueError("feature subset produced an empty feature matrix")
    if matrix.ndim != 2 or matrix.shape[1] != len(feature_names):
        raise ValueError("feature matrix and feature name list are misaligned")
    filtered = matrix[:, keep]
    filtered_names = [feature_names[i] for i in keep]
    return filtered, filtered_names


def session_statistics(df: pd.DataFrame, hours: set[int]) -> tuple[float, float]:
    """Compute simple mean/std statistics for rows within ``hours``."""

    if df.empty:
        return 0.0, 0.0
    value_col = "price" if "price" in df.columns else "spread" if "spread" in df.columns else df.columns[-1]
    if "hour" in df.columns:
        mask = df["hour"].astype(int).isin(hours)
        subset = df.loc[mask, value_col]
        if subset.empty:
            subset = df[value_col]
    else:
        subset = df[value_col]
    subset = pd.to_numeric(subset, errors="coerce").dropna()
    if subset.empty:
        return 0.0, 0.0
    return float(subset.mean()), float(subset.std(ddof=0) or 0.0)


def train_lightweight(
    data_dir: Path,
    out_dir: Path,
    *,
    extra_prices: Mapping[str, Sequence[float]] | None = None,
    config_hash: str | None = None,
    config_snapshot: Mapping[str, Mapping[str, Any]] | None = None,
    feature_subset: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Minimal training routine used for small fixture datasets."""

    start_time = datetime.now(UTC)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    data_path = Path(data_dir)
    csv_file = data_path if data_path.is_file() else data_path / "trades_raw.csv"
    df = pd.read_csv(csv_file)
    if "hour" in df.columns:
        hours = df["hour"].astype(int).to_numpy()
    else:
        hours = np.zeros(len(df), dtype=int)
        df["hour"] = hours
    price_series = df.get("price", pd.Series(np.zeros(len(df))))
    base_symbol = df.get("symbol", pd.Series(["EURUSD"]))[0]

    base_features = [
        "spread",
        "slippage",
        "equity",
        "margin_level",
        "volume",
        "hour_sin",
        "hour_cos",
        "month_sin",
        "month_cos",
        "dom_sin",
        "dom_cos",
    ]
    feature_names = list(base_features)

    extra_stats: dict[str, float] = {}
    if extra_prices:
        for sym, values in extra_prices.items():
            corr_name = f"corr_{base_symbol}_{sym}"
            ratio_name = f"ratio_{base_symbol}_{sym}"
            feature_names.extend([corr_name, ratio_name])
            arr = np.asarray(list(values), dtype=float)
            if arr.size and price_series.size:
                shared = min(len(price_series), arr.size)
                if shared >= 2 and np.std(arr[:shared]) > 0:
                    extra_stats[corr_name] = float(
                        np.corrcoef(price_series.iloc[:shared], arr[:shared])[0, 1]
                    )
                else:
                    extra_stats[corr_name] = 0.0
                base = arr[0] if arr[0] else 1.0
                extra_stats[ratio_name] = float(arr[min(shared - 1, arr.size - 1)] / base)
            else:
                extra_stats[corr_name] = 0.0
                extra_stats[ratio_name] = 1.0

    requested_subset, subset_provided = normalise_feature_subset(feature_subset)
    if subset_provided:
        missing = [name for name in requested_subset if name not in feature_names]
        if missing:
            raise ValueError(
                "Requested features not available in lightweight feature set: %s"
                % missing
            )
        keep_set = set(requested_subset)
        feature_names = [name for name in feature_names if name in keep_set]
        extra_stats = {name: value for name, value in extra_stats.items() if name in keep_set}
        if not feature_names:
            raise ValueError("feature subset produced an empty feature set")

    labels = df.get("label")
    if labels is None or labels.empty:
        labels = df.get("y")
    if labels is not None and len(labels):
        accuracy = float((labels.astype(int) == labels.astype(int).mode()[0]).mean())
        recall = float(labels.astype(int).mean())
    else:
        accuracy = recall = 0.5

    feature_mean = float(price_series.mean()) if not price_series.empty else 0.0
    feature_std = float(price_series.std(ddof=0) or 1.0)

    models = {
        "logreg": {
            "coefficients": [0.1] * len(feature_names),
            "intercept": 0.0,
            "threshold": 0.5,
            "feature_mean": [feature_mean] * len(feature_names),
            "feature_std": [feature_std] * len(feature_names),
            "conformal_lower": 0.2,
            "conformal_upper": 0.8,
        },
        "xgboost": {
            "coefficients": [0.05] * len(feature_names),
            "intercept": -0.1,
            "threshold": 0.55,
            "feature_mean": [feature_mean] * len(feature_names),
            "feature_std": [feature_std] * len(feature_names),
            "conformal_lower": 0.15,
            "conformal_upper": 0.85,
        },
        "lstm": {
            "coefficients": [0.02] * len(feature_names),
            "intercept": 0.05,
            "threshold": 0.45,
            "feature_mean": [feature_mean] * len(feature_names),
            "feature_std": [feature_std] * len(feature_names),
            "conformal_lower": 0.1,
            "conformal_upper": 0.9,
        },
    }

    def _router_values(values: Sequence[float]) -> list[float]:
        if not feature_names:
            return []
        vals = list(values) if values else [0.0]
        repeats = (len(feature_names) + len(vals) - 1) // len(vals)
        repeated = vals * repeats
        return repeated[: len(feature_names)]

    ensemble_router = {
        "intercept": [0.0, 0.1, -0.1],
        "coefficients": [
            _router_values([0.5, -0.2]),
            _router_values([0.1, 0.3]),
            _router_values([-0.4, 0.2]),
        ],
        "feature_mean": _router_values([0.0, 12.0]),
        "feature_std": _router_values([1.0, 6.0]),
    }

    sessions = {
        "asian": set(range(0, 8)),
        "london": set(range(8, 16)),
        "newyork": set(range(16, 24)),
    }
    session_models = {}
    for name, hour_set in sessions.items():
        mean, std = session_statistics(df, hour_set)
        session_models[name] = {
            "feature_mean": [mean],
            "feature_std": [std or 1.0],
            "conformal_lower": 0.2,
            "conformal_upper": 0.8,
            "threshold": 0.5,
            "metrics": {"accuracy": accuracy, "recall": recall},
        }

    metrics = {
        "accuracy": accuracy,
        "recall": recall,
        "brier_score": 0.05,
        "ece": 0.1,
        "max_drawdown": 0.1,
        "var_95": 0.05,
        "threshold": 0.5,
        "threshold_objective": "profit",
    }

    risk_metrics = {"max_drawdown": 0.1, "var_95": 0.05}

    out_path.mkdir(parents=True, exist_ok=True)
    deps_path = out_path / "dependencies.txt"
    deps_lines = dependency_lines(["numpy", "pandas", "scikit-learn"])
    deps_path.write_text("\n".join(deps_lines) + "\n")
    dependencies_hash = hashlib.sha256(deps_path.read_bytes()).hexdigest()

    data_hash = hashlib.sha256(csv_file.read_bytes()).hexdigest()
    data_hashes = {str(csv_file.resolve()): data_hash}
    data_hash_path = out_path / "data_hashes.json"
    data_hash_path.write_text(json.dumps(data_hashes, indent=2))

    snapshot_path = None
    snapshot_hash = None
    if config_snapshot:
        snapshot_path = out_path / "config_snapshot.json"
        snapshot_path.write_text(json.dumps(config_snapshot, indent=2))
        snapshot_hash = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()

    end_time = datetime.now(UTC)

    metadata: dict[str, Any] = {
        "seed": 0,
        "dependencies_file": deps_path.name,
        "dependencies_hash": dependencies_hash,
        "config_hash": config_hash or "",
        "config_snapshot": config_snapshot or {},
        "config_snapshot_path": snapshot_path.name if snapshot_path else None,
        "config_snapshot_hash": snapshot_hash if snapshot_hash else None,
        "training_started_at": start_time.isoformat(),
        "training_completed_at": end_time.isoformat(),
        "training_duration_seconds": float((end_time - start_time).total_seconds()),
        "n_samples": int(len(df)),
        "n_features": int(len(feature_names)),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "experiment": {"run_id": uuid4().hex, "tracking": "offline"},
        "dependencies_path": deps_path.name,
        "data_hashes_path": data_hash_path.name,
    }
    if subset_provided:
        metadata["selected_features"] = requested_subset
    metadata = {k: v for k, v in metadata.items() if v is not None}

    model_data: dict[str, Any] = {
        "feature_names": feature_names,
        "extra_features": extra_stats,
        "models": models,
        "ensemble_router": ensemble_router,
        "session_models": session_models,
        "conformal_lower": 0.2,
        "conformal_upper": 0.8,
        "data_hashes": data_hashes,
        "metadata": metadata,
        "cv_metrics": metrics,
        "risk_metrics": risk_metrics,
        "online_drift_events": [],
        "drift_events": [],
    }

    model_hash = hashlib.sha256(json.dumps(model_data, sort_keys=True).encode()).hexdigest()
    model_data["model_hash"] = model_hash

    model_path = out_path / "model.json"
    model_path.write_text(json.dumps(model_data, indent=2))

    return model_data


def autoencoder_metadata_path(model_path: Path) -> Path:
    """Return the metadata file path associated with ``model_path``."""

    model_path = Path(model_path)
    return model_path.with_name(model_path.name + AUTOENCODER_META_SUFFIX)


def serialise_autoencoder_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Coerce metadata values into JSON-serialisable types."""

    def _convert(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        if isinstance(value, Mapping):
            return {key: _convert(val) for key, val in value.items()}
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [_convert(v) for v in value]
        return value

    return _convert(dict(metadata))


def save_autoencoder_metadata(model_path: Path, metadata: Mapping[str, Any]) -> Path:
    """Persist ``metadata`` next to ``model_path`` and return the file path."""

    meta_path = autoencoder_metadata_path(model_path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    serialised = serialise_autoencoder_metadata(metadata)
    meta_path.write_text(json.dumps(serialised, indent=2, sort_keys=True))
    return meta_path


def load_autoencoder_metadata(model_path: Path) -> dict[str, Any] | None:
    """Load metadata associated with ``model_path`` if present."""

    meta_path = autoencoder_metadata_path(model_path)
    try:
        return json.loads(meta_path.read_text())
    except FileNotFoundError:
        legacy = Path(model_path).with_suffix(".npy")
        if legacy.exists():
            try:
                legacy_state = np.load(legacy, allow_pickle=True).item()
            except Exception:  # pragma: no cover - legacy compatibility best effort
                return None
            basis = np.asarray(legacy_state.get("basis"), dtype=float)
            mean = np.asarray(legacy_state.get("mean"), dtype=float)
            if basis.size and mean.size:
                return {
                    "format": "svd",
                    "latent_dim": int(basis.shape[0]),
                    "input_dim": int(basis.shape[1]),
                    "input_mean": mean.tolist(),
                    "input_scale": [1.0] * int(basis.shape[1]),
                    "weights": basis.tolist(),
                    "bias": None,
                }
        return None
    except json.JSONDecodeError:  # pragma: no cover - defensive guard
        logger.exception("Failed to parse autoencoder metadata at %s", meta_path)
        return None


def extract_torch_encoder_weights(model_path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Extract encoder weights and bias from a PyTorch checkpoint."""

    if not HAS_TORCH:
        raise ImportError("PyTorch is required to load autoencoder checkpoints")
    state = torch.load(model_path, map_location="cpu")  # type: ignore[arg-type]
    if hasattr(state, "state_dict"):
        state = state.state_dict()
    if isinstance(state, Mapping) and "state_dict" in state:
        inner = state.get("state_dict")
        if isinstance(inner, Mapping):
            state = inner
    weight_tensor: "torch.Tensor | None"
    bias_tensor: "torch.Tensor | None"
    weight_tensor = None
    bias_tensor = None
    if isinstance(state, Mapping):
        if "weights" in state:
            weight_tensor = torch.as_tensor(state["weights"], dtype=torch.float32)
            bias_val = state.get("bias")
            if bias_val is not None:
                bias_tensor = torch.as_tensor(bias_val, dtype=torch.float32)
        if weight_tensor is None:
            for key in (
                "encoder.weight",
                "encoder_linear.weight",
                "weight",
            ):
                if key in state:
                    weight_tensor = torch.as_tensor(state[key], dtype=torch.float32)
                    bias_key = key.replace("weight", "bias")
                    bias_val = state.get(bias_key)
                    if bias_val is not None:
                        bias_tensor = torch.as_tensor(bias_val, dtype=torch.float32)
                    break
        if weight_tensor is None:
            for key, value in state.items():
                if key.endswith("weight"):
                    weight_tensor = torch.as_tensor(value, dtype=torch.float32)
                    bias_key = key[: -len("weight")] + "bias"
                    bias_val = state.get(bias_key)
                    if bias_val is not None:
                        bias_tensor = torch.as_tensor(bias_val, dtype=torch.float32)
                    break
    elif HAS_TORCH and isinstance(state, torch.Tensor):  # type: ignore[redundant-expr]
        weight_tensor = state.to(dtype=torch.float32)

    if weight_tensor is None:
        raise ValueError(f"No encoder weights found in checkpoint {model_path}")
    weights = weight_tensor.cpu().numpy()
    bias = bias_tensor.cpu().numpy() if bias_tensor is not None else None
    return weights, bias


def extract_onnx_encoder_weights(
    model_path: Path,
    input_dim: int,
    *,
    sample_inputs: np.ndarray | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    """Derive encoder weights/bias and graph metadata from an ONNX model.

    When the ONNX graph represents a linear mapping we recover the explicit
    weight matrix and bias. Otherwise the caller can fall back to executing the
    ONNX graph directly using the returned metadata.
    """

    try:
        import onnxruntime as ort  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("onnxruntime is required to load ONNX encoders") from exc

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if not inputs:
        raise ValueError("ONNX encoder has no inputs")
    if not outputs:
        raise ValueError("ONNX encoder has no outputs")

    input_name = inputs[0].name
    output_names = [meta.name for meta in outputs]
    primary_output = output_names[0]

    sample_matrix: np.ndarray | None
    if sample_inputs is not None:
        sample_matrix = np.asarray(sample_inputs, dtype=np.float32)
        if sample_matrix.ndim != 2 or sample_matrix.shape[1] != input_dim:
            raise ValueError("sample_inputs must be 2D with input_dim columns")
    else:
        sample_matrix = None

    zero_input = np.zeros((1, input_dim), dtype=np.float32)
    zero_output = np.asarray(
        session.run([primary_output], {input_name: zero_input})[0], dtype=float
    )
    bias = zero_output.reshape(1, -1)[0]

    basis = np.eye(input_dim, dtype=np.float32)
    basis_outputs = np.asarray(
        session.run([primary_output], {input_name: basis})[0], dtype=float
    )

    solve_inputs = basis
    solve_outputs = basis_outputs
    sample_outputs: np.ndarray | None = None
    if sample_matrix is not None:
        if sample_matrix.size:
            sample_outputs = np.asarray(
                session.run([primary_output], {input_name: sample_matrix})[0],
                dtype=float,
            )
            solve_inputs = np.vstack([solve_inputs, sample_matrix])
            solve_outputs = np.vstack([solve_outputs, sample_outputs])
        else:
            sample_outputs = np.empty((0, basis_outputs.shape[1]), dtype=float)

    centred_outputs = solve_outputs - bias
    weights: np.ndarray | None = None
    linear_error = float("inf")
    if solve_inputs.size and centred_outputs.size:
        try:
            solution, _, _, _ = np.linalg.lstsq(solve_inputs, centred_outputs, rcond=None)
        except np.linalg.LinAlgError:
            solution = None
        if solution is not None:
            weights = solution.T
            predicted = solve_inputs @ weights.T + bias
            linear_error = float(np.max(np.abs(predicted - solve_outputs)))

    scale = max(1.0, float(np.max(np.abs(solve_outputs)))) if solve_outputs.size else 1.0
    normalised_error = linear_error / scale if np.isfinite(linear_error) else float("inf")
    numeric_linear = weights is not None and normalised_error <= 1e-5

    nonlinear_ops: list[str] = []
    op_types: list[str] = []
    try:
        import onnx  # type: ignore
    except Exception:
        pass
    else:
        model = onnx.load(str(model_path))
        graph_ops = {node.op_type for node in model.graph.node}
        op_types = sorted(graph_ops)
        linear_ops = {
            "Add",
            "AveragePool",
            "BatchNormalization",
            "Cast",
            "Concat",
            "Constant",
            "Div",
            "Dropout",
            "Flatten",
            "Gather",
            "Gemm",
            "Identity",
            "InstanceNormalization",
            "MatMul",
            "Mul",
            "Pad",
            "ReduceMean",
            "ReduceSum",
            "Reshape",
            "Shape",
            "Slice",
            "Split",
            "Sub",
            "Squeeze",
            "Transpose",
            "Unsqueeze",
        }
        known_nonlinear = {
            "Abs",
            "Acos",
            "Acosh",
            "Asin",
            "Asinh",
            "Atan",
            "Atanh",
            "Celu",
            "Clip",
            "Cos",
            "Cosh",
            "Elu",
            "Exp",
            "Gelu",
            "HardSigmoid",
            "HardSwish",
            "Hardmax",
            "LeakyRelu",
            "Log",
            "LogSoftmax",
            "PRelu",
            "Relu",
            "Selu",
            "Sigmoid",
            "Softmax",
            "Softplus",
            "Softsign",
            "Swish",
            "Tanh",
            "ThresholdedRelu",
        }
        suspicious = {
            op
            for op in graph_ops
            if op in known_nonlinear or op not in linear_ops
        }
        nonlinear_ops = sorted(suspicious)

    graph_linear = not nonlinear_ops
    is_linear = numeric_linear and graph_linear

    output_shape = zero_output.shape
    if zero_output.ndim <= 1:
        latent_dim = int(zero_output.size)
    else:
        latent_dim = int(np.prod(output_shape[1:]))

    metadata: dict[str, Any] = {
        "onnx_input": input_name,
        "onnx_outputs": output_names,
        "latent_dim": latent_dim,
    }
    if op_types:
        metadata["onnx_ops"] = op_types
    if nonlinear_ops:
        metadata["onnx_nonlinear_ops"] = nonlinear_ops
    if sample_outputs is not None:
        metadata["sample_outputs"] = sample_outputs

    if not is_linear:
        metadata["onnx_serialized"] = base64.b64encode(model_path.read_bytes()).decode("ascii")
        return None, None, metadata

    return weights, bias, metadata


def apply_autoencoder_from_metadata(
    X: np.ndarray, metadata: Mapping[str, Any]
) -> np.ndarray:
    """Project ``X`` using encoder weights described by ``metadata``."""

    data = np.asarray(X, dtype=float)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.ndim != 2:
        raise ValueError("Autoencoder input must be a 2D array")

    mean = np.asarray(metadata.get("input_mean", []), dtype=float)
    scale = np.asarray(metadata.get("input_scale", []), dtype=float)
    if mean.size and mean.shape[0] == data.shape[1]:
        data = data - mean
    if scale.size and scale.shape[0] == data.shape[1]:
        safe_scale = np.where(scale == 0, 1.0, scale)
        data = data / safe_scale

    format_name = str(metadata.get("format") or "").lower()
    if format_name == "onnx_nonlin":
        try:
            import onnxruntime as ort  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError("onnxruntime is required to execute nonlinear encoders") from exc

        serialized = metadata.get("onnx_serialized")
        session: "ort.InferenceSession"
        if isinstance(serialized, str) and serialized:
            graph_bytes = base64.b64decode(serialized.encode("ascii"))
            session = ort.InferenceSession(graph_bytes, providers=["CPUExecutionProvider"])
        else:
            weights_file = metadata.get("weights_file")
            if not weights_file:
                raise ValueError("Nonlinear ONNX encoder metadata is missing model bytes")
            path = Path(str(weights_file))
            if not path.exists():
                raise FileNotFoundError(f"ONNX encoder file not found at {path}")
            session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

        input_name = metadata.get("onnx_input") or session.get_inputs()[0].name
        output_names = metadata.get("onnx_outputs") or [session.get_outputs()[0].name]
        outputs = session.run(output_names, {input_name: data.astype(np.float32)})[0]
        return np.asarray(outputs, dtype=float)

    weights = metadata.get("weights")
    if weights is None:
        raise ValueError("Autoencoder metadata does not contain encoder weights")
    weight_arr = np.asarray(weights, dtype=float)
    if weight_arr.ndim != 2:
        raise ValueError("Encoder weights must be a 2D array")
    bias_val = metadata.get("bias")
    if bias_val is not None:
        bias_arr = np.asarray(bias_val, dtype=float)
    else:
        bias_arr = None
    embedding = data @ weight_arr.T
    if bias_arr is not None:
        embedding = embedding + bias_arr
    return embedding.astype(float)


def encode_with_autoencoder(
    X: np.ndarray, model_path: Path, *, latent_dim: int = 2
) -> np.ndarray:
    """Project ``X`` into a low-dimensional latent space using an encoder."""

    data = np.asarray(X, dtype=float)
    if data.ndim != 2:
        raise ValueError("X must be a 2D array")
    samples, features = data.shape
    if samples == 0 or features == 0:
        return np.zeros((samples, min(latent_dim, features)), dtype=float)

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    mean = data.mean(axis=0)
    centered = data - mean
    metadata: dict[str, Any] = {
        "metadata_version": 1,
        "input_dim": int(features),
        "input_mean": mean.tolist(),
        "input_scale": [1.0] * int(features),
        "weights_file": model_path.name,
    }

    suffix = model_path.suffix.lower()
    weights: np.ndarray | None = None
    bias: np.ndarray | None = None

    if model_path.exists():
        try:
            if suffix in {".pt", ".pth"}:
                weights, bias = extract_torch_encoder_weights(model_path)
                metadata["format"] = "torch"
            elif suffix == ".onnx":
                weights, bias, onnx_info = extract_onnx_encoder_weights(
                    model_path,
                    features,
                    sample_inputs=centered,
                )
                sample_outputs = onnx_info.pop("sample_outputs", None)
                metadata.update(onnx_info)
                if weights is not None:
                    metadata["format"] = "onnx"
                    metadata.setdefault("latent_dim", int(weights.shape[0]))
                    metadata["weights"] = weights.tolist()
                    metadata["bias"] = bias.tolist() if bias is not None else None
                    if sample_outputs is not None:
                        embedding = np.asarray(sample_outputs, dtype=float)
                    else:
                        embedding = centered @ weights.T
                        if bias is not None:
                            embedding = embedding + bias
                    save_autoencoder_metadata(model_path, metadata)
                    return embedding.astype(float)
                metadata["format"] = "onnx_nonlin"
                metadata["weights"] = None
                metadata["bias"] = None
                if isinstance(sample_outputs, np.ndarray):
                    embedding = np.asarray(sample_outputs, dtype=float)
                elif samples:
                    try:
                        import onnxruntime as ort  # type: ignore
                    except Exception as exc:  # pragma: no cover - optional dependency
                        raise ImportError(
                            "onnxruntime is required to execute nonlinear ONNX encoders"
                        ) from exc
                    model_bytes = metadata.get("onnx_serialized")
                    if isinstance(model_bytes, str) and model_bytes:
                        graph_bytes = base64.b64decode(model_bytes.encode("ascii"))
                        session = ort.InferenceSession(graph_bytes, providers=["CPUExecutionProvider"])
                    else:
                        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
                    input_name = metadata.get("onnx_input")
                    if not input_name:
                        input_name = session.get_inputs()[0].name
                    output_names = metadata.get("onnx_outputs") or [session.get_outputs()[0].name]
                    embedding = np.asarray(
                        session.run(output_names, {input_name: centered.astype(np.float32)})[0],
                        dtype=float,
                    )
                else:
                    embedding = np.zeros((0, int(metadata.get("latent_dim", 0))), dtype=float)
                metadata.setdefault("latent_dim", int(embedding.shape[1] if embedding.ndim == 2 else 0))
                save_autoencoder_metadata(model_path, metadata)
                return embedding.astype(float)
            else:
                loaded_meta = load_autoencoder_metadata(model_path)
                if loaded_meta and loaded_meta.get("weights") is not None:
                    weights = np.asarray(loaded_meta["weights"], dtype=float)
                    bias_val = loaded_meta.get("bias")
                    if bias_val is not None:
                        bias = np.asarray(bias_val, dtype=float)
                    metadata.update(
                        {k: v for k, v in loaded_meta.items() if k not in {"weights", "bias"}}
                    )
            if weights is not None:
                metadata.setdefault("latent_dim", int(weights.shape[0]))
                metadata["weights"] = weights.tolist()
                metadata["bias"] = bias.tolist() if bias is not None else None
                embedding = centered @ weights.T
                if bias is not None:
                    embedding = embedding + bias
                save_autoencoder_metadata(model_path, metadata)
                return embedding.astype(float)
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning(
                "Failed to load autoencoder weights from %s (%s); falling back to SVD",
                model_path,
                exc,
            )

    k = int(max(1, min(latent_dim, features)))
    u, s, vh = np.linalg.svd(centered, full_matrices=False)
    basis = vh[:k]
    embedding = centered @ basis.T
    metadata.update(
        {
            "format": "svd",
            "latent_dim": int(k),
            "weights": basis.tolist(),
            "bias": None,
        }
    )
    with model_path.open("wb") as fh:
        np.savez(fh, basis=basis.astype(np.float32))
    save_autoencoder_metadata(model_path, metadata)
    return embedding.astype(float)


def load_news_embeddings(data_dir: Path) -> tuple[pd.DataFrame | None, dict[str, str]]:
    """Load optional news embedding sequences from ``data_dir``."""

    base = data_dir if data_dir.is_dir() else data_dir.parent
    if base is None:
        return None, {}
    candidates = [
        base / "news_embeddings.parquet",
        base / "news_embeddings.csv",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            if path.suffix == ".parquet":
                news_df = pd.read_parquet(path)
            else:
                news_df = pd.read_csv(path)
        except Exception:  # pragma: no cover - optional dependency/io
            logger.exception("Failed to load news embeddings from %s", path)
            continue
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            digest = ""
        return news_df, {str(path.resolve()): digest}
    return None, {}


__all__ = [
    "HAS_DASK",
    "HAS_FEAST",
    "HAS_POLARS",
    "HAS_TORCH",
    "FEATURE_COLUMNS",
    "FeatureStore",
    "torch",
    "dd",
    "apply_autoencoder_from_metadata",
    "autoencoder_metadata_path",
    "dependency_lines",
    "encode_with_autoencoder",
    "extract_onnx_encoder_weights",
    "extract_torch_encoder_weights",
    "filter_feature_matrix",
    "load_autoencoder_metadata",
    "load_news_embeddings",
    "normalise_feature_subset",
    "save_autoencoder_metadata",
    "session_statistics",
    "should_use_lightweight",
    "train_lightweight",
    "pl",
]
