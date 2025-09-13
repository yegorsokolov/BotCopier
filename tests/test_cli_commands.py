import sys
import types

import pytest
from click.testing import CliRunner
from typer.main import get_command


@pytest.fixture
def cli_app(monkeypatch):
    """Provide the BotCopier CLI command with heavy dependencies stubbed."""
    pipeline_stub = types.ModuleType("botcopier.training.pipeline")
    pipeline_stub.train = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "botcopier.training.pipeline", pipeline_stub)

    evaluation_stub = types.ModuleType("botcopier.scripts.evaluation")
    evaluation_stub.evaluate = lambda *a, **k: {}
    monkeypatch.setitem(sys.modules, "botcopier.scripts.evaluation", evaluation_stub)

    online_stub = types.ModuleType("botcopier.scripts.online_trainer")
    async def run_stub(*a, **k):
        return None
    online_stub.run = run_stub
    monkeypatch.setitem(sys.modules, "botcopier.scripts.online_trainer", online_stub)

    drift_stub = types.ModuleType("botcopier.scripts.drift_monitor")
    drift_stub.run = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "botcopier.scripts.drift_monitor", drift_stub)

    import botcopier.cli as cli_module
    cli_command = get_command(cli_module.app)

    yield cli_command, cli_module

    sys.modules.pop("botcopier.cli", None)


def test_train_requires_args(cli_app):
    cli, _ = cli_app
    runner = CliRunner()
    result = runner.invoke(cli, ["train"], prog_name="botcopier")
    assert result.exit_code != 0


def test_train_help(cli_app):
    cli, _ = cli_app
    runner = CliRunner()
    result = runner.invoke(cli, ["train", "--help"], prog_name="botcopier")
    assert result.exit_code == 0
    assert "--model-type" in result.output


def test_evaluate_requires_args(cli_app):
    cli, _ = cli_app
    runner = CliRunner()
    result = runner.invoke(cli, ["evaluate"], prog_name="botcopier")
    assert result.exit_code != 0


def test_evaluate_help(cli_app):
    cli, _ = cli_app
    runner = CliRunner()
    result = runner.invoke(cli, ["evaluate", "--help"], prog_name="botcopier")
    assert result.exit_code == 0
    assert "--model-json" in result.output


def test_online_train_runs(monkeypatch, cli_app):
    cli, cli_module = cli_app
    called = {}

    async def fake_run(data_cfg, train_cfg):
        called["called"] = True

    monkeypatch.setattr(cli_module, "run_online_trainer", fake_run)
    runner = CliRunner()
    result = runner.invoke(cli, ["online-train"], prog_name="botcopier")
    assert result.exit_code == 0
    assert called.get("called")


def test_online_train_help(cli_app):
    cli, _ = cli_app
    runner = CliRunner()
    result = runner.invoke(cli, ["online-train", "--help"], prog_name="botcopier")
    assert result.exit_code == 0
    assert "--batch-size" in result.output


def test_drift_monitor_requires_files(cli_app):
    cli, _ = cli_app
    runner = CliRunner()
    result = runner.invoke(cli, ["drift-monitor"], prog_name="botcopier")
    assert result.exit_code != 0


def test_drift_monitor_help(cli_app):
    cli, _ = cli_app
    runner = CliRunner()
    result = runner.invoke(cli, ["drift-monitor", "--help"], prog_name="botcopier")
    assert result.exit_code == 0
    assert "--baseline-file" in result.output
