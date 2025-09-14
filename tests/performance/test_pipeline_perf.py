from pathlib import Path
import shutil

from botcopier.training.pipeline import train


def test_pipeline_training_benchmark(benchmark, tmp_path):
    """Benchmark the end-to-end training pipeline on a small dataset."""
    rows = [
        "label,price,volume,spread,hour,symbol\n",
        "0,1.0,100,1.5,0,EURUSD\n",
        "1,1.1,110,1.6,1,EURUSD\n",
        "0,1.2,120,1.7,2,EURUSD\n",
        "1,1.3,130,1.8,3,EURUSD\n",
    ]
    data_file = tmp_path / "trades.csv"
    data_file.write_text("".join(rows))

    def run() -> None:
        out_dir = tmp_path / "out"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        train(data_file, out_dir)

    benchmark(run)
