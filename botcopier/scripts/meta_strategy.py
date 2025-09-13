#!/usr/bin/env python3
"""Train a meta-classifier that maps regime features to best base model."""
import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, SGDClassifier

try:  # pragma: no cover - optional dependency
    import stable_baselines3 as sb3  # type: ignore
    try:  # gym is optional, gymnasium fallback
        from gym import Env, spaces  # type: ignore
    except Exception:  # pragma: no cover - gymnasium fallback
        from gymnasium import Env, spaces  # type: ignore
    HAS_SB3 = True
except Exception:  # pragma: no cover - RL deps optional
    sb3 = None  # type: ignore
    Env = object  # type: ignore
    spaces = None  # type: ignore
    HAS_SB3 = False


def train_meta_model(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str = "best_model",
    params: Optional[Dict[str, object]] = None,
    use_partial_fit: bool = False,
) -> Dict[str, object]:
    """Train multinomial logistic regression to select best model.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing regime features and labels of best model.
    feature_cols : List[str]
        Columns to use as features.
    label_col : str, default "best_model"
        Column containing the index of the best-performing base model.
    params : dict, optional
        Previously trained parameters to warm-start incremental training.
    use_partial_fit : bool, default False
        When ``True``, use :class:`~sklearn.linear_model.SGDClassifier` with
        ``partial_fit`` to update parameters incrementally.

    Returns
    -------
    Dict[str, object]
        Dictionary with feature names, gating coefficients and intercepts.
    """
    X = df[feature_cols].values
    y = df[label_col].values

    if use_partial_fit:
        clf = SGDClassifier(loss="log_loss", random_state=0)
        if params:
            try:
                coef = np.array(params.get("gating_coefficients", []))
                intercept = np.array(params.get("gating_intercepts", []))
                classes = np.array(params.get("classes", []))
                if coef.shape[0] == 2 and classes.size == 2:
                    coef = coef[1:2]
                    intercept = intercept[1:2]
                clf.coef_ = coef
                clf.intercept_ = intercept
                if classes.size:
                    clf.classes_ = classes
            except Exception:
                pass
        if not hasattr(clf, "classes_") or len(getattr(clf, "classes_", [])) == 0:
            classes = np.unique(y)
        else:
            classes = clf.classes_
        clf.partial_fit(X, y, classes=classes)
    else:
        clf = LogisticRegression(
            multi_class="multinomial", solver="lbfgs", max_iter=200
        )
        clf.fit(X, y)

    coeffs = clf.coef_
    intercepts = clf.intercept_
    # For binary problems, sklearn returns a single row; create symmetric rows
    if coeffs.shape[0] == 1 and len(clf.classes_) == 2:
        coeffs = np.vstack([np.zeros_like(coeffs[0]), coeffs[0]])
        intercepts = np.array([0.0, intercepts[0]])
    return {
        "feature_names": feature_cols,
        "gating_coefficients": coeffs.tolist(),
        "gating_intercepts": intercepts.tolist(),
        "classes": clf.classes_.tolist(),
    }


def select_model(params: Dict[str, object], features: Dict[str, float]) -> int:
    """Select the best model given regime features.

    Parameters
    ----------
    params : dict
        Output from :func:`train_meta_model` containing coefficients.
    features : Dict[str, float]
        Mapping from feature name to value.

    Returns
    -------
    int
        Index of chosen base model.
    """
    names = params.get("feature_names", [])
    coeffs = np.array(params.get("gating_coefficients", []))
    intercepts = np.array(params.get("gating_intercepts", []))
    if coeffs.size == 0:
        return 0
    x = np.array([features.get(n, 0.0) for n in names])
    scores = intercepts + coeffs.dot(x)
    return int(np.argmax(scores))


class RollingMetrics:
    """Maintain rolling profit and accuracy for base models.

    The metrics are updated using exponential moving averages so that recent
    performance has greater influence.  Weights for combining model outputs are
    computed from these metrics using an exponential transformation and
    normalisation.
    """

    def __init__(self, n_models: int, alpha: float = 0.1) -> None:
        self.alpha = float(alpha)
        self.profit = np.zeros(n_models, dtype=float)
        self.accuracy = np.zeros(n_models, dtype=float)

    def update(self, model_idx: int, profit: float, correct: bool) -> None:
        """Update rolling statistics for ``model_idx``."""
        self.profit[model_idx] = (1 - self.alpha) * self.profit[
            model_idx
        ] + self.alpha * float(profit)
        self.accuracy[model_idx] = (1 - self.alpha) * self.accuracy[
            model_idx
        ] + self.alpha * (1.0 if correct else 0.0)

    def weights(self) -> np.ndarray:
        """Return exponential weights derived from recent performance."""
        score = self.profit + self.accuracy
        score -= score.max()
        exp_score = np.exp(score)
        if exp_score.sum() == 0:
            return np.ones_like(exp_score) / len(exp_score)
        return exp_score / exp_score.sum()

    def combine(self, outputs: Iterable[float]) -> float:
        """Combine ``outputs`` from base models using current weights."""
        preds = np.array(list(outputs), dtype=float)
        w = self.weights()[: len(preds)]
        return float(np.dot(w, preds))


class ThresholdEnv(Env):  # pragma: no cover - exercised via tests
    """Simple environment that yields thresholds for trading decisions.

    The observation at each step consists of regime features followed by the
    current model probability.  The agent outputs a threshold in ``[0, 1]``. If
    the model probability exceeds this threshold a trade is taken and the
    realised profit at that timestep is used as the reward; otherwise the reward
    is zero.
    """

    def __init__(self, features: np.ndarray, probs: np.ndarray, profits: np.ndarray):
        self.features = np.asarray(features, dtype=np.float32)
        self.probs = np.asarray(probs, dtype=np.float32)
        self.profits = np.asarray(profits, dtype=np.float32)
        self.n_steps = len(self.profits)
        obs_dim = self.features.shape[1] + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self._t = 0

    def reset(self):  # type: ignore[override]
        self._t = 0
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        feat = self.features[self._t]
        prob = self.probs[self._t]
        return np.concatenate([feat, [prob]]).astype(np.float32)

    def step(self, action):  # type: ignore[override]
        threshold = float(np.clip(action[0], 0.0, 1.0))
        prob = self.probs[self._t]
        reward = self.profits[self._t] if prob >= threshold else 0.0
        self._t += 1
        done = self._t >= self.n_steps
        if not done:
            obs = self._get_obs()
        else:
            obs = np.zeros_like(self._get_obs())
        return obs, float(reward), done, {}


class ThresholdAgent:
    """Policy-gradient agent that outputs trade thresholds.

    The agent observes regime features and current model probability and learns
    a policy that maps these observations to a threshold.  A
    :class:`stable_baselines3.PPO` model is used under the hood.  During
    inference the :meth:`act` method returns the threshold and whether a trade
    should be taken while logging the action.
    """

    def __init__(self) -> None:
        self.model: Optional["sb3.PPO"] = None
        self._logs: List[Dict[str, float]] = []

    def train(
        self,
        features: np.ndarray,
        probs: np.ndarray,
        profits: np.ndarray,
        training_steps: int = 1000,
    ) -> None:
        if not HAS_SB3:  # pragma: no cover - dependency missing
            raise RuntimeError("stable-baselines3 not installed")
        env = ThresholdEnv(features, probs, profits)
        self.model = sb3.PPO("MlpPolicy", env, verbose=0)
        self.model.learn(total_timesteps=training_steps)
        self.env = env

    def act(self, features: np.ndarray, prob: float) -> Tuple[float, bool]:
        if not self.model:
            raise RuntimeError("Agent has not been trained")
        obs = np.concatenate([np.asarray(features, dtype=np.float32), [float(prob)]])
        threshold, _ = self.model.predict(obs, deterministic=True)
        thr = float(np.clip(threshold[0], 0.0, 1.0))
        trade = float(prob) >= thr
        self._logs.append({"threshold": thr, "prob": float(prob), "trade": float(trade)})
        return thr, trade

    def logs(self) -> List[Dict[str, float]]:
        """Return logged actions."""
        return list(self._logs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train meta gating model")
    parser.add_argument(
        "--data", help="CSV file with regime features and best_model column"
    )
    parser.add_argument("--out", help="Output JSON file for gating parameters")
    parser.add_argument("--features", nargs="+", help="Feature column names")
    parser.add_argument("--label", help="Label column name")
    args = parser.parse_args()

    from botcopier.config.settings import DataConfig, TrainingConfig, save_params

    data_cfg = DataConfig(
        **{k: getattr(args, k) for k in ["data", "out"] if getattr(args, k) is not None}
    )
    train_cfg = TrainingConfig(
        **{
            k: getattr(args, k)
            for k in ["features", "label"]
            if getattr(args, k) is not None
        }
    )
    save_params(data_cfg, train_cfg)

    df = pd.read_csv(data_cfg.data)
    params = train_meta_model(df, train_cfg.features, label_col=train_cfg.label)
    out_path = Path(data_cfg.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(params, f)


if __name__ == "__main__":
    main()
