"""Small JSON front door: python3 -m gaussian_orthant examples/orthant.json."""
import argparse
import json
from pathlib import Path

from . import (gaussian_probability, orthant_probability,
               exchangeable_precision_probability, equicorrelated_probability)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a Gaussian probability with BootLoops")
    parser.add_argument("input", type=Path, help="JSON problem description")
    parser.add_argument("--digits", type=int, help="Override the input's requested precision")
    parser.add_argument("--backend", choices=("auto", "mpmath", "flint"),
                        help="Override the arithmetic backend")
    args = parser.parse_args()
    problem = json.loads(args.input.read_text())
    kind = problem.pop("kind", "box")
    if args.digits is not None:
        problem["digits"] = args.digits
    if args.backend is not None:
        problem["backend"] = args.backend
    evaluators = {"box": gaussian_probability, "orthant": orthant_probability,
                  "exchangeable_precision": exchangeable_precision_probability,
                  "equicorrelated": equicorrelated_probability}
    if kind not in evaluators:
        parser.error(f"Unknown kind {kind!r}; choose from {', '.join(evaluators)}")
    value = evaluators[kind](**problem)
    print(json.dumps(value.as_dict(), indent=2))


if __name__ == "__main__":
    main()
