import json
from pathlib import Path

TEMPLATE = """// Auto-generated strategy
// Computes logistic regression probability
// Inputs array should align with feature_names order

double predict(double &inputs[])
{{
    double z = {intercept};
{body}
    return 1.0 / (1.0 + MathExp(-z));
}}
"""

def generate(model_file, out_dir):
    """Generate a tiny MQL4 strategy from ``model.json``."""
    model_file = Path(model_file)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(model_file) as f:
        model = json.load(f)
    coeffs = model.get("coefficients", [])
    names = model.get("feature_names", [f"f{i}" for i in range(len(coeffs))])
    intercept = float(model.get("intercept", 0.0))
    body_lines = []
    for idx, (c, name) in enumerate(zip(coeffs, names)):
        body_lines.append(f"    z += {float(c)} * inputs[{idx}]; // {name}")
    body = "\n".join(body_lines)
    code = TEMPLATE.format(intercept=intercept, body=body)
    out_path = out_dir / f"Strategy_{model.get('model_id', 'model')}.mq4"
    out_path.write_text(code)
    return out_path

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Generate MQL4 strategy from model")
    p.add_argument("model", type=Path, help="model.json")
    p.add_argument("out_dir", type=Path, help="output directory")
    args = p.parse_args()
    generate(args.model, args.out_dir)
