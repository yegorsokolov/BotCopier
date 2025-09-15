import sys
import types

if "gplearn.genetic" not in sys.modules:
    sys.modules.setdefault("gplearn", types.ModuleType("gplearn"))
    genetic = types.ModuleType("gplearn.genetic")
    genetic.SymbolicTransformer = object  # type: ignore[attr-defined]
    sys.modules["gplearn.genetic"] = genetic

from botcopier.training.pipeline import train


def test_mlflow_creates_artifacts(tmp_path):
    data = tmp_path / "trades_raw.csv"
    lines = ["label,spread,hour"]
    for i in range(5):
        lines.append(f"0,{1.0 + i * 0.1},{i % 24}")
    for i in range(5):
        lines.append(f"1,{2.0 + i * 0.1},{i % 24}")
    data.write_text("\n".join(lines))

    tracking_dir = tmp_path / "mlruns"
    out_dir = tmp_path / "out"

    train(
        data,
        out_dir,
        tracking_uri=tracking_dir.as_uri(),
        experiment_name="test-exp",
    )

    artifact = next(tracking_dir.glob("**/artifacts/model/model.json"))
    run_dir = artifact.parents[2]
    assert (run_dir / "params" / "model_type").exists()
    assert (run_dir / "params" / "n_features").exists()
    assert (run_dir / "params" / "random_seed").exists()
    assert (run_dir / "params" / "model_uri").exists()
    assert (run_dir / "params" / "data_hashes_uri").exists()
    assert (run_dir / "metrics" / "train_accuracy").exists()
    assert (run_dir / "metrics" / "cv_accuracy").exists()
