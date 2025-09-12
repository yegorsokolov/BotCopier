from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelParams(BaseModel):
    """Flexible model parameter schema persisted as JSON.

    A ``version`` field allows backward compatibility checks.  All other
    parameters are accepted and round-tripped without explicit declaration
    so older and newer model files can coexist.
    """

    feature_names: list[str] = Field(default_factory=list)
    data_hashes: dict[str, str] = Field(default_factory=dict)
    version: Literal[1] = 1

    model_config = ConfigDict(extra="allow")


__all__ = ["ModelParams"]
