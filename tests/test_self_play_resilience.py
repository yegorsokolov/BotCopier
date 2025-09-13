from pathlib import Path

from scripts.self_play_env import (
    SelfPlayEnv,
    evaluate,
    train_self_play,
    train_trader_only,
)


def test_self_play_yields_resilient_strategy(tmp_path: Path) -> None:
    """Self-play training should outperform historical-only training under attack."""

    env_hist = SelfPlayEnv(
        drift=0.01, perturb_scale=0.05, steps=1, volatility=0.0, seed=0
    )
    hist_q = train_trader_only(env_hist, episodes=200, lr=0.5, epsilon=0.1)
    pnl_hist = evaluate(env_hist, hist_q, perturb_action=-1, episodes=50)

    env_sp = SelfPlayEnv(
        drift=0.01, perturb_scale=0.05, steps=1, volatility=0.0, seed=0
    )
    sp_q, _ = train_self_play(env_sp, episodes=200, lr=0.5, epsilon=0.1)
    pnl_sp = evaluate(env_sp, sp_q, perturb_action=-1, episodes=50)

    assert pnl_sp >= pnl_hist
