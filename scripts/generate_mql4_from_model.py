#!/usr/bin/env python3
"""Render MQL4 strategy file from model description."""
import argparse
import json
from pathlib import Path

template_path = Path(__file__).resolve().parent.parent / 'experts' / 'StrategyTemplate.mq4'


def generate(model_json: Path, out_dir: Path):
    with open(model_json) as f:
        model = json.load(f)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(template_path) as f:
        template = f.read()
    # Placeholder: simply copy template
    output = template.replace('MagicNumber = 1234', f'MagicNumber = {model.get("magic", 9999)}')
    out_file = out_dir / f"Generated_{model.get('model_id','model')}.mq4"
    with open(out_file, 'w') as f:
        f.write(output)
    print(f"Strategy written to {out_file}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('model_json')
    p.add_argument('out_dir')
    args = p.parse_args()
    generate(Path(args.model_json), Path(args.out_dir))

if __name__ == '__main__':
    main()
