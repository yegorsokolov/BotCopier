import json
from pathlib import Path

from automl.controller import AutoMLController


def test_automl_controller_converges(tmp_path):
    features = ["f1", "f2"]
    models = {"linear": 1, "tree": 2}

    model_file = tmp_path / "model.json"
    controller = AutoMLController(features, models, model_path=model_file)

    profit_map = {
        (("f1",), "linear"): 1.0,
        (("f2",), "linear"): 1.2,
        (("f1", "f2"), "linear"): 1.5,
        (("f1",), "tree"): 1.1,
        (("f2",), "tree"): 1.3,
        (("f1", "f2"), "tree"): 1.7,
    }

    def env(action):
        subset, model = action
        return profit_map[(tuple(subset), model)]

    controller.train(env, episodes=200, alpha=0.2, penalty=0.1)

    best = controller.select_best()
    assert best == (("f1", "f2"), "tree")

    data = json.loads(model_file.read_text())
    assert data["automl_controller"]["best_action"] == {
        "features": ["f1", "f2"],
        "model": "tree",
    }
