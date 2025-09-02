"""CLI wrapper for evaluating predictions against actual trades."""
import argparse
import json
from pathlib import Path

try:
    from .model_fitting import load_logs
except ImportError:
    from model_fitting import load_logs

try:
    from .evaluation import evaluate
except ImportError:  # when run as a script
    from evaluation import evaluate


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("predicted_log", help="CSV file of predicted trades")
    p.add_argument("actual_log", help="CSV file of actual trades")
    p.add_argument("--window", type=int, default=60, help="Match window in seconds")
    p.add_argument("--model-json", type=Path, default=None, help="Optional model JSON for conformal bounds")
    p.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Path to write JSON summary (default: evaluation.json next to prediction file)",
    )
    args = p.parse_args()

    # Ensure actual log can be parsed using shared loader
    load_logs(Path(args.actual_log))

    stats = evaluate(
        Path(args.predicted_log),
        Path(args.actual_log),
        args.window,
        args.model_json,
    )

    out_path = args.json_out or Path(args.predicted_log).parent / "evaluation.json"
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Wrote report to {out_path}")

    print("--- Evaluation Summary ---")
    print(f"Predicted events : {stats['predicted_events']}")
    print(
        f"Matched events   : {stats['matched_events']} ({stats['precision']*100:.1f}% precision)"
    )
    print(f"Recall           : {stats['recall']*100:.1f}% of actual trades")
    print(f"Accuracy         : {stats['accuracy']*100:.1f}%")
    print(
        f"Profit Factor    : {stats['profit_factor']:.2f}"
        f" (gross P/L: {stats['gross_profit']-stats['gross_loss']:.2f})"
    )
    print(f"Sharpe Ratio     : {stats['sharpe_ratio']:.2f}")
    print(f"Sortino Ratio    : {stats['sortino_ratio']:.2f}")
    print(f"Expectancy       : {stats['expectancy']:.2f}")
    if args.model_json:
        print(
            f"Probability coverage : {stats['conformal_coverage']*100:.1f}%"
        )
        if "model_value_mean" in stats and "model_value_std" in stats:
            print(
                f"Model value mean : {stats['model_value_mean']:.2f}"
                f" (std: {stats['model_value_std']:.2f})"
            )


if __name__ == "__main__":
    main()

