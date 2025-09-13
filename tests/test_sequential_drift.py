from scripts.sequential_drift import PageHinkley


def test_page_hinkley_detects_expected_change():
    detector = PageHinkley(delta=0.1, threshold=1.0, min_samples=5)
    data = [0.0] * 10 + [1.0] * 10
    alarm = None
    for i, v in enumerate(data):
        if detector.update(v):
            alarm = i
            break
    assert alarm == 11

