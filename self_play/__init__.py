"""Self-play training environment for adversarial trading."""

from .env import SelfPlayEnv, evaluate, train_self_play, train_trader_only

__all__ = ["SelfPlayEnv", "train_self_play", "train_trader_only", "evaluate"]
