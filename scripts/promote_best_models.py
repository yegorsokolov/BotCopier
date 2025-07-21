#!/usr/bin/env python3
"""Select best models and copy to best folder."""
import argparse
import shutil
from pathlib import Path


def promote(models_dir: Path, best_dir: Path, max_models: int):
    best_dir.mkdir(parents=True, exist_ok=True)
    models = sorted(models_dir.glob('model_*.json'))[:max_models]
    for m in models:
        dest = best_dir / m.name
        shutil.copy(m, dest)
        print(f"Promoted {m} to {dest}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('models_dir')
    p.add_argument('best_dir')
    p.add_argument('--max-models', type=int, default=3)
    args = p.parse_args()
    promote(Path(args.models_dir), Path(args.best_dir), args.max_models)

if __name__ == '__main__':
    main()
