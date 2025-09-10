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
    assert (run_dir / "metrics" / "train_accuracy").exists()
