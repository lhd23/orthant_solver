"""Direct Gaussian products, with the same precision gates as transport."""
from mpmath import mp

from .common import (ConvergenceError, complement_guard_digits, interval_guard_digits,
                     normal_interval, number, precision_errors, result)
from ._tolerance import exact_rtol, finish_rtol, goal_digits, magnitude_guard


def independent_probability(lower, upper, precision, *, digits, backend, repeat=1, rtol=None):
    """Inputs are exact, centered endpoints and positive diagonal precisions.

    repeat represents identical independent copies without building a list
    proportional to the requested dimension.
    """
    coordinates = tuple(zip(lower, upper, precision))
    dimension = len(coordinates) * repeat
    if all(a.is_infinite and b.is_infinite for a, b, _ in coordinates):
        if rtol is not None:
            return exact_rtol(mp.mpf(1), digits, "full Gaussian space",
                              {"dimension": dimension}, rtol)
        return result(mp.mpf(1), digits, "full Gaussian space", {"dimension": dimension})
    cancellation = max(interval_guard_digits(a, b) for a, b, _ in coordinates)
    if rtol is not None:
        base = max(18, goal_digits(rtol) + 10) + cancellation + len(str(dimension))
        base += magnitude_guard(*(x for coordinate in coordinates for x in coordinate))
        previous, runs = None, []
        for extra in (0, 6, 14, 30, 62):
            with mp.workdps(base + extra):
                factors = [normal_interval(number(a) * mp.sqrt(number(q)),
                                           number(b) * mp.sqrt(number(q)), backend=backend)
                           for a, b, q in coordinates]
                value = mp.fprod(factors) ** repeat
                runs.append({"work_digits": base + extra})
                if previous is not None and mp.isfinite(value) and 0 < value <= 1:
                    estimate = 4 * abs(value - previous) / value
                    estimate += mp.power(10, -(base + extra - 6)) * dimension
                    if estimate < rtol / 2:
                        method = ("univariate Gaussian formula" if dimension == 1
                                  else "independent Gaussian product")
                        return finish_rtol(value, digits, method,
                                           {"dimension": dimension, "backend": backend,
                                            "runs": runs}, rtol, estimate)
                previous = value
        raise ConvergenceError("Direct Gaussian product did not meet rtol")
    with mp.workdps(digits + 25 + cancellation):
        guards = []
        for a, b, q in coordinates:
            if a.is_infinite and b.is_infinite:
                continue
            root = mp.sqrt(number(q))
            guards.append(complement_guard_digits(number(a) * root, number(b) * root))
        log_guard = min(guards, default=0)
    previous, runs = None, []
    for extra in (25, 45, 85, 155):
        work_digits = digits + extra + cancellation + log_guard + len(str(dimension))
        with mp.workdps(work_digits):
            factors = []
            for a, b, q in coordinates:
                root = mp.sqrt(number(q))
                factors.append(normal_interval(number(a) * root, number(b) * root, backend=backend))
            value = mp.fprod(factors) ** repeat
            runs.append({"work_digits": work_digits})
            if previous is not None and value > 0:
                relative, log_relative = precision_errors(value, previous, bool(log_guard))
                if max(relative, log_relative) < mp.power(10, -digits):
                    method = "univariate Gaussian formula" if dimension == 1 else "independent Gaussian product"
                    return result(value, digits, method, {
                        "dimension": dimension, "backend": backend, "runs": runs,
                        "relative_precision_agreement": mp.nstr(relative, 8),
                        "relative_log_precision_agreement": mp.nstr(log_relative, 8),
                        "error_status": "direct Gaussian formula and precision agreement; not an interval certificate",
                    })
            previous = value
    raise ConvergenceError("Direct Gaussian product did not pass relative precision agreement")
