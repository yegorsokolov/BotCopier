"""Write model coefficients and thresholds for the MQL4 strategy.

This helper saves comma separated ``coeffs`` and ``thresholds`` values to a
file inside the ``Files`` directory.  ``StrategyTemplate.mq4`` reloads the file
on a timer and therefore can pick up new parameters after each training run
without requiring recompilation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


def write_params(coeffs: Sequence[float], thresholds: Sequence[float], path: Path) -> None:
    """Write ``coeffs`` and ``thresholds`` to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(",".join(map(str, coeffs)) + "\n")
        f.write(",".join(map(str, thresholds)) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--coeffs", type=float, nargs="+", required=True, help="Model coefficients")
    p.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        required=True,
        help="Decision thresholds",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("Files/model_params.csv"),
        help="Destination parameters file",
    )
    args = p.parse_args()
    write_params(args.coeffs, args.thresholds, args.output)


if __name__ == "__main__":
    main()
