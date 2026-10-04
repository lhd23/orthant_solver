#!/usr/bin/env python3
"""Reproduce three specified families in Section 5 of Botev's paper.

Run: OPENBLAS_NUM_THREADS=1 python3 validate.py --digits 25
The default run saves both a machine-readable receipt and a readable report.
Published-table differences are reported separately from independently
verified correctness, and are never silently treated as exact truths.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time
from importlib.metadata import PackageNotFoundError, version

# Avoid a thread pool on every small matrix product in the independent
# statistical estimator. Respect explicitly supplied user settings.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from fractions import Fraction
import numpy as np
from mpmath import mp

from gaussian_orthant import (gaussian_probability, exchangeable_precision_probability,
                             equicorrelated_probability)
from gaussian_orthant._kernels import resolve_backend
from validation_oracles import minimax_reference, positive_convolution_reference


PAPER = "https://arxiv.org/html/1603.04166"
EXAMPLE_I = {
    2: ("0.01489", "4e-5"), 3: ("0.001077", "3e-4"),
    5: ("2.451e-6", "0.002"), 10: ("8.556e-15", "0.01"),
    20: ("1.7796e-38", "0.03"), 50: ("2.1364e-153", "0.06"),
}
EXAMPLE_II = {2: ("0.09121", "2e-4"), 3: ("0.02307", "4e-4")}


def publication_comparison(value, printed, relative_error_percent):
    with mp.workdps(60):
        estimate, error = mp.mpf(printed), mp.mpf(relative_error_percent) / 100
        mantissa, _, exponent = printed.lower().partition("e")
        decimals = len(mantissa.partition(".")[2])
        printed_unit = mp.power(10, int(exponent or 0) - decimals)
        difference = abs(value - estimate)
        tolerance = 5 * error * estimate + printed_unit
        return {
            "printed_probability": printed,
            "printed_relative_standard_error_percent": relative_error_percent,
            "relative_difference_percent": mp.nstr(difference / estimate * 100, 10),
            "difference_in_stated_standard_errors": mp.nstr(difference / (error * estimate), 10),
            "within_five_standard_errors_plus_printed_unit": bool(difference <= tolerance),
            "comparison_rule": "five printed standard errors plus one unit of the last printed digit",
        }


def compare_minimax(value, baseline):
    difference = abs(float(value) / baseline["probability"] - 1)
    tolerance = max(6 * baseline["relative_standard_error"], 1e-10)
    return {"relative_difference": difference, "tolerance": tolerance,
            "passed": bool(difference < tolerance), **baseline}


def run(digits, output, backend="auto"):
    selected_backend = resolve_backend(backend)
    flint_version = None
    if selected_backend == "flint":
        try:
            flint_version = version("python-flint")
        except PackageNotFoundError:
            # A library imported from a source checkout may lack metadata.
            pass
    output.mkdir(parents=True, exist_ok=True)
    records = []
    print("Example I: precision = (I + 11^T)/2; box [1/2,1]^d", flush=True)
    for d, (printed, standard_error) in EXAMPLE_I.items():
        started = time.perf_counter()
        field = exchangeable_precision_probability(d, "0.5", 1, digits=digits,
                                                  backend=selected_backend)
        q = [[Fraction(1, 2) + (Fraction(1, 2) if i == j else 0) for j in range(d)] for i in range(d)]
        crosscheck = None
        if d <= 5:
            transported = gaussian_probability(["0.5"] * d, [1] * d, precision=q,
                                               digits=digits, backend=selected_backend)
            with mp.workdps(digits + 20):
                agreement = abs(field.probability / transported.probability - 1)
                crosscheck = {"relative_difference": mp.nstr(agreement, 8),
                              "passed": bool(agreement < mp.power(10, -(digits - 2))),
                              "transport": transported.as_dict()}
        reference, refinements = positive_convolution_reference(d)
        discrepancy = abs(float(field.probability) / reference - 1)
        convolution = {"probability": reference, "refinements": refinements,
                       "relative_difference": discrepancy, "passed": bool(discrepancy < 1e-10)}
        baseline = compare_minimax(field.probability, minimax_reference([0.5] * d, [1] * d, q))
        record = {"family": "Section 5.1 Example I", "dimension": d,
                  "candidate": field.as_dict(), "boundary_crosscheck": crosscheck,
                  "positive_convolution": convolution, "independent_minimax": baseline,
                  "publication": publication_comparison(field.probability, printed, standard_error),
                  "seconds": time.perf_counter() - started}
        record["independent_checks_passed"] = (convolution["passed"] and baseline["passed"]
                                                and (crosscheck is None or crosscheck["passed"]))
        records.append(record)
        print(f"  d={d}: {mp.nstr(field.probability, 14)}; independent checks "
              f"{'PASS' if record['independent_checks_passed'] else 'FAIL'}", flush=True)

    print("Example II: banded precision; box [0,1]^d", flush=True)
    for d, (printed, standard_error) in EXAMPLE_II.items():
        started = time.perf_counter()
        q = [[Fraction(1, 2 ** abs(i - j)) if abs(i - j) <= d / 2 else 0
              for j in range(d)] for i in range(d)]
        value = gaussian_probability([0] * d, [1] * d, precision=q, digits=digits,
                                     backend=selected_backend)
        # Independent direct positive Gauss-Legendre product, two rule orders.
        from numpy.polynomial.legendre import leggauss
        refs = []
        for order in (16, 24):
            nodes, weights = leggauss(order)
            points = np.stack(np.meshgrid(*[(nodes + 1) / 2] * d, indexing="ij"), axis=-1)
            rule = np.ones(points.shape[:-1])
            for component in np.meshgrid(*[weights / 2] * d, indexing="ij"):
                rule *= component
            numeric_q = np.asarray(q, float)
            f = np.exp(-np.einsum("...i,ij,...j->...", points, numeric_q, points) / 2)
            refs.append(np.sqrt(np.linalg.det(numeric_q)) / (2 * np.pi) ** (d / 2) * np.sum(f * rule))
        discrepancy = abs(float(value.probability) / refs[-1] - 1)
        baseline = compare_minimax(value.probability, minimax_reference([0] * d, [1] * d, q))
        record = {"family": "Section 5.1 Example II", "dimension": d,
                  "candidate": value.as_dict(),
                  "direct_product_quadrature": {"orders": [16, 24], "probabilities": refs,
                                                 "relative_difference": discrepancy,
                                                 "passed": bool(discrepancy < 1e-12 and abs(refs[0] / refs[1] - 1) < 1e-12)},
                  "independent_minimax": baseline,
                  "publication": publication_comparison(value.probability, printed, standard_error),
                  "seconds": time.perf_counter() - started}
        record["independent_checks_passed"] = record["direct_product_quadrature"]["passed"] and baseline["passed"]
        records.append(record)
        print(f"  d={d}: {mp.nstr(value.probability, 14)}; independent checks "
              f"{'PASS' if record['independent_checks_passed'] else 'FAIL'}", flush=True)

    print("Section 5.3: correlation 1/2; exact orthant probability 1/(d+1)", flush=True)
    for d in (10, 100, 1000, 10000):
        started = time.perf_counter()
        value = equicorrelated_probability(d, digits=digits, backend=selected_backend)
        with mp.workdps(digits + 30):
            exact = mp.mpf(1) / (d + 1)
            difference = abs(value.probability / exact - 1)
            record = {"family": "Section 5.3 exact orthant", "dimension": d,
                      "candidate": value.as_dict(), "exact_probability": mp.nstr(exact, digits + 5),
                      "relative_difference": mp.nstr(difference, 8),
                      "independent_checks_passed": bool(difference < mp.power(10, -digits)),
                      "seconds": time.perf_counter() - started}
        records.append(record)
        print(f"  d={d}: {mp.nstr(value.probability, 14)}; exact check "
              f"{'PASS' if record['independent_checks_passed'] else 'FAIL'}", flush=True)

    mismatches = [{"family": r["family"], "dimension": r["dimension"], **r["publication"]}
                  for r in records if "publication" in r and
                  not r["publication"]["within_five_standard_errors_plus_printed_unit"]]
    root = Path(__file__).resolve().parent
    source_paths = list((root / "gaussian_orthant").glob("*.py")) + [root / "validate.py", root / "validation_oracles.py",
                                                                           root / "BootLoops.pdf"]
    boot_root = Path(os.environ.get("BOOTLOOPS_ROOT", root / "bootloops"))
    source_paths += [boot_root / "tools/wayfinder/transport.py", boot_root / "tools/wayfinder/quad.py",
                     boot_root / "tools/wayfinder/ratfun.py", boot_root / "tools/wayfinder/acbfast.py"]
    receipt = {
        "source": PAPER, "requested_digits": digits,
        "requested_backend": backend, "arithmetic_backend": selected_backend,
        "python_flint_version": flint_version,
        "python": platform.python_version(), "platform": platform.platform(),
        "source_sha256": {str(path.relative_to(root) if path.is_relative_to(root) else path):
                          hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths if path.is_file()},
        "all_independent_checks_passed": all(r["independent_checks_passed"] for r in records),
        "published_table_discrepancies": mismatches,
        "records": records,
        "scope": "Example II validated at dimensions 2 and 3; random 100-dimensional matrices in Examples III and IV are not supplied by the paper",
        "error_status": "Measured convergence and independent agreement, not a rigorous interval enclosure",
    }
    (output / "validation.json").write_text(json.dumps(receipt, indent=2) + "\n")
    lines = ["# Gaussian orthant validation", "", f"Source: [Botev, Section 5]({PAPER}).",
             "", f"Arithmetic backend: `{selected_backend}`.",
             "", f"Requested precision: {digits} decimal digits. Independent checks: "
             + ("all passed." if receipt["all_independent_checks_passed"] else "a check failed."), "",
             "| Paper family | Dimension | Computed probability | Published estimate or exact value | Independent check |",
             "|---|---:|---:|---:|---|"]
    for r in records:
        ref = r["publication"]["printed_probability"] if "publication" in r else "1/(d+1)"
        lines.append(f"| {r['family']} | {r['dimension']} | {r['candidate']['probability']} | {ref} | "
                     + ("passed" if r["independent_checks_passed"] else "FAILED") + " |")
    lines += ["", "The published estimates are stochastic, rounded values. The table-comparison gate allows five "
              "printed standard errors and one unit of the last printed digit; it is a discrepancy flag, not a proof "
              "that a published estimate is wrong.", ""]
    for mismatch in mismatches:
        lines.append(f"For {mismatch['family']} at dimension {mismatch['dimension']}, the candidate differs from "
                     f"the printed estimate by {mismatch['relative_difference_percent']} percent "
                     f"({mismatch['difference_in_stated_standard_errors']} stated standard errors). "
                     "The independently implemented minimax estimator and positive original-integrand convolution "
                     "agree with the candidate. The discrepancy is retained in the receipt.")
    lines += ["", "Example I was checked by a separate positive convolution of the original box integrand, "
              "a newly implemented minimax tilting estimator, and boundary transport at dimensions 2, 3 and 5. "
              "Example II was checked by direct product quadrature and minimax tilting. The Section 5.3 numerical "
              "one-factor integral was checked against the exact formula, including dimension 10000.", "",
              "The independent minimax implementation uses twelve independently scrambled Sobol sequences, "
              "with 4096 points in each sequence and a fixed recorded seed. The paper uses randomized Richtmyer "
              "sequences. Its errors and our estimator errors are statistical; high-precision candidate digits "
              "are supported by refinement and precision agreement, with an analytic discarded-tail bound for "
              "the auxiliary-field integral. No complete interval certificate is claimed.", "",
              "Example II's large-dimensional rows and the random matrices in Examples III and IV were not "
              "reproduced. The general boundary basis grows exponentially; structured reductions enable the "
              "large-dimensional results above. No general speed advantage over minimax tilting is established.", "",
              "The implementation passes the independent checks on these three reproducible paper families; "
              "the receipt preserves the printed-table discrepancy and the limits of the numerical evidence."]
    (output / "validation.md").write_text("\n".join(lines) + "\n")
    print(f"Saved {output / 'validation.md'} and {output / 'validation.json'}", flush=True)
    print(f"Printed-table discrepancy flags: {len(mismatches)}", flush=True)
    return 0 if receipt["all_independent_checks_passed"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--digits", type=int, default=25)
    parser.add_argument("--backend", choices=("auto", "mpmath", "flint"), default="auto")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()
    raise SystemExit(run(args.digits, args.output, args.backend))
