"""Adapters for the self-play environment within the legacy ``scripts`` namespace."""
from self_play.env import (
    SelfPlayEnv,
    train_trader_only,
    train_self_play,
    evaluate,
)

__all__ = [
    "SelfPlayEnv",
    "train_trader_only",
    "train_self_play",
    "evaluate",
]
