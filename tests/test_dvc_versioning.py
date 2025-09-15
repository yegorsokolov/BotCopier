from pathlib import Path
import sys
import types

import pytest

if "gplearn.genetic" not in sys.modules:
    sys.modules.setdefault("gplearn", types.ModuleType("gplearn"))
    genetic = types.ModuleType("gplearn.genetic")
    genetic.SymbolicTransformer = object  # type: ignore[attr-defined]
    sys.modules["gplearn.genetic"] = genetic

from botcopier.training.pipeline import train


def _write_sample_dataset(path: Path) -> None:
    lines = ["label,spread,hour"]
    for i in range(5):
        lines.append(f"0,{1.0 + i * 0.1},{i % 24}")
    for i in range(5):
        lines.append(f"1,{2.0 + i * 0.1},{i % 24}")
    path.write_text("\n".join(lines))


def test_train_versions_artifacts_with_dvc(tmp_path):
    pytest.importorskip("dvc")
    from dvc.repo import Repo as DvcRepo  # type: ignore

    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    repo = DvcRepo.init(str(repo_dir), no_scm=True)
    repo.close()

    data = repo_dir / "trades.csv"
    _write_sample_dataset(data)

    out_dir = repo_dir / "models"

    train(
        data,
        out_dir,
        dvc_repo=repo_dir,
    )

    assert (repo_dir / "trades.csv.dvc").exists()
    assert (out_dir / "model.json.dvc").exists()
    assert (out_dir / "data_hashes.json.dvc").exists()
