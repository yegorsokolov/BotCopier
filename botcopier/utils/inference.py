from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.preprocessing import PowerTransformer

from botcopier.models.schema import FeatureMetadata
from botcopier.training.preprocessing import (
    apply_autoencoder_from_metadata,
    load_autoencoder_metadata,
)


logger = logging.getLogger(__name__)


def _normalise_feature_list(values: Sequence[Any]) -> list[str]:
    return [str(v) for v in values if v is not None]


def resolve_autoencoder_metadata(
    meta: Mapping[str, Any] | None, model_dir: Path | None
) -> dict[str, Any] | None:
    """Return encoder metadata ensuring weights are populated when available."""

    if not isinstance(meta, Mapping):
        return None

    resolved: dict[str, Any] = {k: v for k, v in meta.items()}

    def _merge(source: Mapping[str, Any] | None) -> None:
        if not isinstance(source, Mapping):
            return
        for key, value in source.items():
            if key not in resolved or resolved[key] in (None, []):
                resolved[key] = value

    meta_file = resolved.get("metadata_file")
    if meta_file and model_dir is not None:
        path = Path(str(meta_file))
        if not path.is_absolute():
            path = (model_dir / path).resolve()
        try:
            data = json.loads(path.read_text())
        except FileNotFoundError:
            logger.warning("Autoencoder metadata file not found at %s", path)
        except json.JSONDecodeError:
            logger.warning("Failed to parse autoencoder metadata at %s", path)
        else:
            _merge(data)

    weights_file = resolved.get("weights_file")
    if (resolved.get("weights") in (None, [])) and weights_file and model_dir is not None:
        path = Path(str(weights_file))
        if not path.is_absolute():
            path = (model_dir / path).resolve()
        loaded = load_autoencoder_metadata(path)
        if loaded:
            _merge(loaded)

    format_name = str(resolved.get("format") or "").lower()
    if resolved.get("weights") in (None, []) and format_name != "onnx_nonlin":
        return None

    return resolved


@dataclass
class FeaturePipeline:
    """Utility encapsulating feature transformations required for inference."""

    feature_names: list[str]
    feature_metadata: list[FeatureMetadata]
    input_columns: list[str]
    schema_columns: list[str]
    autoencoder_meta: dict[str, Any] | None = None
    autoencoder_inputs: list[str] = field(default_factory=list)
    autoencoder_outputs: list[str] = field(default_factory=list)
    power_transformer: PowerTransformer | None = None
    power_indices: list[int] = field(default_factory=list)
    feature_name_to_original: dict[str, str] = field(default_factory=dict)
    _input_indices: dict[str, int] = field(init=False, repr=False)
    _feature_indices: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._ensure_index_maps()

    def _ensure_index_maps(self) -> None:
        self._input_indices = {
            name: idx for idx, name in enumerate(self.input_columns)
        }
        self._feature_indices = {
            name: idx for idx, name in enumerate(self.feature_names)
        }

    @classmethod
    def from_model(
        cls,
        model: Mapping[str, Any],
        *,
        model_dir: Path | None = None,
        session_key: str | None = None,
    ) -> "FeaturePipeline":
        session_models = model.get("session_models")
        if session_models:
            if session_key is None:
                params = next(iter(session_models.values()))
            else:
                if session_key not in session_models:
                    available = ", ".join(sorted(map(str, session_models))) or "none"
                    raise ValueError(
                        "model does not contain session "
                        f"{session_key!r}; available sessions: {available}"
                    )
                params = session_models[session_key]
        else:
            if session_key is not None:
                raise ValueError(
                    "model does not define session_models but session "
                    f"{session_key!r} was requested"
                )
            params = model

        feature_names = [str(f) for f in params.get("feature_names") or model.get("feature_names", [])]
        metadata_raw = params.get("feature_metadata") or model.get("feature_metadata") or []
        feature_metadata: list[FeatureMetadata] = []
        for entry in metadata_raw:
            if isinstance(entry, FeatureMetadata):
                feature_metadata.append(entry)
            else:
                try:
                    feature_metadata.append(FeatureMetadata(**entry))
                except Exception:
                    continue

        feature_name_to_original: dict[str, str] = {}
        if feature_metadata and len(feature_metadata) == len(feature_names):
            for name, meta in zip(feature_names, feature_metadata):
                feature_name_to_original[name] = meta.original_column

        expected_cols = [meta.original_column for meta in feature_metadata] if feature_metadata else []
        input_columns = _normalise_feature_list(expected_cols or feature_names)

        auto_meta = (
            params.get("autoencoder")
            or model.get("autoencoder")
            or params.get("masked_encoder")
            or model.get("masked_encoder")
        )
        autoencoder_meta = resolve_autoencoder_metadata(auto_meta, model_dir)

        auto_inputs: list[str] = []
        auto_outputs: list[str] = []
        schema_columns = list(input_columns)

        if autoencoder_meta:
            sources = (
                autoencoder_meta.get("input_features")
                or autoencoder_meta.get("original_features")
            )
            if sources:
                auto_inputs = _normalise_feature_list(sources)
            elif input_columns:
                auto_inputs = list(input_columns)

            outputs_raw = (
                autoencoder_meta.get("feature_names")
                or autoencoder_meta.get("autoencoder_outputs")
                or []
            )
            auto_outputs = _normalise_feature_list(outputs_raw)
            if not auto_outputs:
                raise ValueError(
                    "autoencoder metadata must define output feature names via "
                    "'feature_names' or 'autoencoder_outputs'"
                )
            missing_outputs = [
                name for name in auto_outputs if name not in feature_names
            ]
            if missing_outputs:
                missing = ", ".join(sorted(missing_outputs))
                available = ", ".join(feature_names) or "none"
                raise ValueError(
                    "autoencoder outputs %s are not present in feature_names (%s)"
                    % (missing, available)
                )
            schema_columns = list(auto_inputs)

        combined_inputs: list[str] = []
        for name in (*input_columns, *auto_inputs):
            if name and name not in combined_inputs:
                combined_inputs.append(name)
        input_columns = combined_inputs

        pt_meta = params.get("power_transformer") or model.get("power_transformer")
        power_transformer: PowerTransformer | None = None
        power_indices: list[int] = []
        if pt_meta:
            pt = PowerTransformer(method="yeo-johnson")
            pt.lambdas_ = np.asarray(pt_meta.get("lambdas", []), dtype=float)
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            scaler.mean_ = np.asarray(pt_meta.get("mean", []), dtype=float)
            scaler.scale_ = np.asarray(pt_meta.get("scale", []), dtype=float)
            pt._scaler = scaler
            pt.n_features_in_ = pt.lambdas_.shape[0]
            power_transformer = pt
            selected = [str(f) for f in pt_meta.get("features", [])]
            for name in selected:
                if name in feature_names:
                    power_indices.append(feature_names.index(name))

        return cls(
            feature_names=list(feature_names),
            feature_metadata=feature_metadata,
            input_columns=input_columns,
            schema_columns=schema_columns,
            autoencoder_meta=autoencoder_meta,
            autoencoder_inputs=auto_inputs,
            autoencoder_outputs=auto_outputs,
            power_transformer=power_transformer,
            power_indices=power_indices,
            feature_name_to_original=feature_name_to_original,
        )

    def transform_array(self, values: Sequence[float]) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        if arr.ndim != 1:
            arr = arr.ravel()
        length = arr.shape[0]
        if length == len(self.input_columns):
            features = self._transform_from_input_matrix(arr.reshape(1, -1))[0]
        elif length == len(self.feature_names):
            features = arr.astype(float)
        else:
            raise ValueError("feature length mismatch")
        return self._apply_power_transform(features)

    def transform_matrix(self, matrix: np.ndarray) -> np.ndarray:
        data = np.asarray(matrix, dtype=float)
        if data.ndim != 2:
            raise ValueError("feature matrix must be 2-dimensional")
        if data.shape[1] == len(self.input_columns):
            features = self._transform_from_input_matrix(data)
            return self._apply_power_transform(features)
        if data.shape[1] == len(self.feature_names):
            return self._apply_power_transform(data.astype(float))
        raise ValueError("feature matrix column mismatch")

    def transform_dict(self, values: Mapping[str, float]) -> np.ndarray:
        mapping = {str(k): float(v) for k, v in values.items()}
        features = self._transform_from_mapping(mapping)
        return self._apply_power_transform(features)

    def _transform_from_mapping(self, mapping: Mapping[str, float]) -> np.ndarray:
        resolved = {str(k): float(v) for k, v in mapping.items()}
        for name, original in self.feature_name_to_original.items():
            if original in mapping and name not in resolved:
                resolved[name] = float(mapping[original])

        if self.autoencoder_meta and self.autoencoder_inputs:
            missing = [name for name in self.autoencoder_inputs if name not in resolved]
            if missing:
                raise ValueError(
                    f"missing autoencoder inputs: {', '.join(sorted(missing))}"
                )
            arr = np.asarray([resolved[name] for name in self.autoencoder_inputs], dtype=float)
            embedding = apply_autoencoder_from_metadata(arr.reshape(1, -1), self.autoencoder_meta)
            latent = embedding.ravel()
            outputs = self.autoencoder_outputs
            if not outputs:
                raise ValueError(
                    "autoencoder_outputs are not configured; ensure metadata "
                    "includes output feature names"
                )
            if len(outputs) != len(latent):
                limit = min(len(outputs), len(latent))
                outputs = outputs[:limit]
                latent = latent[:limit]
            for name, value in zip(outputs, latent):
                resolved[name] = float(value)

        final: list[float] = []
        missing_final: list[str] = []
        for name in self.feature_names:
            if name in resolved:
                final.append(float(resolved[name]))
            else:
                missing_final.append(name)
        if missing_final:
            raise ValueError(
                "missing features: " + ", ".join(sorted(missing_final))
            )
        return np.asarray(final, dtype=float)

    def _transform_from_input_matrix(self, data: np.ndarray) -> np.ndarray:
        matrix = np.asarray(data, dtype=float)
        self._ensure_index_maps()
        if matrix.ndim != 2:
            raise ValueError("input matrix must be 2-dimensional")
        samples, columns = matrix.shape
        if columns != len(self.input_columns):
            raise ValueError("input matrix column mismatch")
        if samples == 0:
            return np.zeros((0, len(self.feature_names)), dtype=float)

        result = np.full((samples, len(self.feature_names)), np.nan, dtype=float)
        filled = np.zeros(len(self.feature_names), dtype=bool)

        target_indices: list[int] = []
        source_indices: list[int] = []
        for idx, name in enumerate(self.feature_names):
            src_idx = self._input_indices.get(name)
            if src_idx is not None:
                target_indices.append(idx)
                source_indices.append(src_idx)
                continue
            original = self.feature_name_to_original.get(name)
            if original:
                src_idx = self._input_indices.get(original)
                if src_idx is not None:
                    target_indices.append(idx)
                    source_indices.append(src_idx)

        if source_indices:
            gathered = np.take(matrix, source_indices, axis=1)
            result[:, target_indices] = gathered
            filled_indices = np.asarray(target_indices, dtype=int)
            filled[filled_indices] = True

        if self.autoencoder_meta and self.autoencoder_inputs:
            missing_inputs = [
                name for name in self.autoencoder_inputs if name not in self._input_indices
            ]
            if missing_inputs:
                raise ValueError(
                    "missing autoencoder inputs: " + ", ".join(sorted(missing_inputs))
                )
            ae_indices = [self._input_indices[name] for name in self.autoencoder_inputs]
            latent = apply_autoencoder_from_metadata(
                np.take(matrix, ae_indices, axis=1), self.autoencoder_meta
            )
            outputs = self.autoencoder_outputs
            if not outputs:
                raise ValueError(
                    "autoencoder_outputs are not configured; ensure metadata "
                    "includes output feature names"
                )
            limit = min(len(outputs), latent.shape[1])
            if limit:
                latent_slice = latent[:, :limit]
                for pos, name in enumerate(outputs[:limit]):
                    target_idx = self._feature_indices.get(name)
                    if target_idx is not None:
                        result[:, target_idx] = latent_slice[:, pos]
                        filled[target_idx] = True

        missing_features = [
            self.feature_names[idx] for idx, is_filled in enumerate(filled) if not is_filled
        ]
        if missing_features:
            raise ValueError(
                "missing features: " + ", ".join(sorted(missing_features))
            )
        return result

    def _apply_power_transform(self, features: np.ndarray) -> np.ndarray:
        arr = np.asarray(features, dtype=float)
        if self.power_transformer is None or not self.power_indices:
            return arr
        if arr.ndim == 1:
            subset = arr[self.power_indices].reshape(1, -1)
            transformed = self.power_transformer.transform(subset).ravel()
            result = arr.copy()
            result[self.power_indices] = transformed
            return result
        if arr.ndim == 2:
            if arr.shape[0] == 0:
                return arr
            transformed = self.power_transformer.transform(arr[:, self.power_indices])
            result = arr.copy()
            result[:, self.power_indices] = transformed
            return result
        raise ValueError("power transform input must be 1D or 2D array")

