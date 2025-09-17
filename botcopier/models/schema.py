from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class FeatureMetadata(BaseModel):
    """Metadata describing how a feature was generated."""

    original_column: str = Field(
        ..., description="Source column name before any transformations"
    )
    transformations: list[str] = Field(
        default_factory=list, description="Applied transformation names"
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Parameters for transformations"
    )


class ModelParams(BaseModel):
    """Flexible model parameter schema persisted as JSON.

    A ``version`` field allows backward compatibility checks.  All other
    parameters are accepted and round-tripped without explicit declaration
    so older and newer model files can coexist.
    """

    feature_names: list[str] = Field(default_factory=list)
    feature_metadata: list[FeatureMetadata] = Field(default_factory=list)
    data_hashes: dict[str, str] = Field(default_factory=dict)
    config_hash: str | None = Field(
        default=None, description="SHA256 hash of the configuration used"
    )
    regime_features: list[str] = Field(
        default_factory=list,
        description="Feature names used by the Mixture-of-Experts gating network",
    )
    experts: list[dict[str, Any]] = Field(
        default_factory=list, description="Serialised expert layer parameters"
    )
    regime_gating: dict[str, Any] | None = Field(
        default=None, description="Serialised gating network parameters"
    )
    version: Literal[1] = 1

    model_config = ConfigDict(extra="allow")


__all__ = ["ModelParams", "FeatureMetadata"]
