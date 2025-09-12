import math

from botcopier.scripts.router import TrafficRouter


def test_routing_proportions_approximate_config():
    router = TrafficRouter([0.7, 0.3], seed=123)
    counts = [0, 0]
    trials = 10000
    for _ in range(trials):
        idx = router.choose()
        counts[idx] += 1
    frac0 = counts[0] / trials
    frac1 = counts[1] / trials
    assert math.isclose(frac0, 0.7, rel_tol=0.1)
    assert math.isclose(frac1, 0.3, rel_tol=0.1)


def test_switch_to_best_model():
    router = TrafficRouter([0.5, 0.5], seed=42)
    # model 0 gets losses, model 1 wins
    for _ in range(50):
        router.update(0, 0)
        router.update(1, 1)
    assert router.best_model() == 1
    router.switch_to_best()
    assert router.weights[1] == 1.0
    assert sum(router.weights) == 1.0
