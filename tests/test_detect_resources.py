import importlib
import types
import scripts.train_target_clone as tc


class DummyVM:
    available = 16 * 1024**3


class DummyCPUFreq:
    def __init__(self, m):
        self.max = m


def _make_torch(mem):
    class Props:
        def __init__(self, m):
            self.total_memory = m

    class Cuda:
        def __init__(self, m):
            self._m = m

        def is_available(self):
            return True

        def get_device_properties(self, idx):
            return Props(self._m)

    return types.SimpleNamespace(cuda=Cuda(mem))


def _fake_find_spec(name):
    if name in {"transformers", "pytorch_forecasting", "torch", "sklearn"}:
        return types.SimpleNamespace()
    return _orig_find_spec(name)


def test_detect_resources_gpu_threshold(monkeypatch):
    monkeypatch.setattr(tc.psutil, "virtual_memory", lambda: DummyVM)
    monkeypatch.setattr(tc.psutil, "cpu_count", lambda logical=False: 8)
    monkeypatch.setattr(tc.psutil, "cpu_freq", lambda: DummyCPUFreq(3000))
    monkeypatch.setattr(
        tc.shutil, "disk_usage", lambda path: types.SimpleNamespace(free=10 * 1024**3)
    )
    global _orig_find_spec
    _orig_find_spec = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", _fake_find_spec)
    monkeypatch.setattr(tc, "_HAS_TORCH", True)

    monkeypatch.setattr(tc, "torch", _make_torch(4 * 1024**3))
    res = tc.detect_resources()
    assert res["gpu_mem_gb"] == 4
    assert res["model_type"] == "logreg"
    assert res["cpu_mhz"] == 3000

    monkeypatch.setattr(tc, "torch", _make_torch(12 * 1024**3))
    res = tc.detect_resources()
    assert res["gpu_mem_gb"] == 12
    assert res["model_type"] == "transformer"


def test_detect_resources_low_disk(monkeypatch):
    monkeypatch.setattr(tc.psutil, "virtual_memory", lambda: DummyVM)
    monkeypatch.setattr(tc.psutil, "cpu_count", lambda logical=False: 8)
    monkeypatch.setattr(tc.psutil, "cpu_freq", lambda: DummyCPUFreq(3000))
    monkeypatch.setattr(
        tc.shutil, "disk_usage", lambda path: types.SimpleNamespace(free=4 * 1024**3)
    )
    res = tc.detect_resources()
    assert res["disk_gb"] == 4
    assert res["lite_mode"]


def test_detect_resources_cpu_threshold(monkeypatch):
    monkeypatch.setattr(tc.psutil, "virtual_memory", lambda: DummyVM)
    monkeypatch.setattr(tc.psutil, "cpu_count", lambda logical=False: 8)
    monkeypatch.setattr(tc.psutil, "cpu_freq", lambda: DummyCPUFreq(2000))
    monkeypatch.setattr(
        tc.shutil, "disk_usage", lambda path: types.SimpleNamespace(free=10 * 1024**3)
    )
    global _orig_find_spec
    _orig_find_spec = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", _fake_find_spec)
    monkeypatch.setattr(tc, "_HAS_TORCH", True)
    monkeypatch.setattr(tc, "torch", _make_torch(12 * 1024**3))
    res = tc.detect_resources()
    assert res["cpu_mhz"] == 2000
    assert res["model_type"] == "logreg"
