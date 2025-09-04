import pathlib

ALLOWED = {"test_generate.py"}


def pytest_ignore_collect(path, config):
    if pathlib.Path(str(path)).name not in ALLOWED:
        return True
