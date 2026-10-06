"""Precision-safe inputs and result objects."""
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any, Dict

from mpmath import mp
import sympy as sp


class ConvergenceError(RuntimeError):
    """Requested agreement was not obtained; no probability is returned."""


def rational(x):
    """Retain exact supplied values before testing bounds or matrix signs."""
    if str(x).lower() in ("inf", "+inf", "infinity", "+infinity", "oo", "+oo"):
        return sp.oo
    if str(x).lower() in ("-inf", "-infinity", "-oo"):
        return -sp.oo
    if str(x).lower() == "nan":
        return sp.nan
    if isinstance(x, mp.mpc):
        if mp.im(x):
            raise ValueError("Inputs must be real")
        x = mp.re(x)
    if hasattr(x, "_mpf_"):
        return sp.Rational(x)
    return sp.Rational(str(x))


def interval_guard_digits(lower, upper):
    """Budget for endpoint subtraction using the exact centered interval."""
    if not (lower.is_finite and upper.is_finite) or lower == upper:
        return 0
    ratio = max(1, abs(lower), abs(upper)) / (upper - lower)
    # An upper bound for ceil(log10(ratio)), without rounding its inputs.
    numerator, denominator = ratio.as_numer_denom()
    return max(0, len(str(numerator)) - len(str(denominator)) + 1)


def number(x):
    if str(x).lower() in ("oo", "+oo"):
        return mp.inf
    if str(x).lower() == "-oo":
        return mp.ninf
    # Decimal strings retain caller-specified precision. Floats retain only
    # their supplied decimal representation; more output digits do not
    # restore precision lost before this call.
    if isinstance(x, Integral):
        return mp.mpf(int(x))
    if hasattr(x, "numerator") and hasattr(x, "denominator"):
        return mp.mpf(int(x.numerator)) / int(x.denominator)
    if isinstance(x, (mp.mpf, mp.mpc)):
        if mp.im(x):
            raise ValueError("Inputs must be real")
        return mp.mpf(mp.re(x))
    return mp.mpf(str(x))


def check_digits(digits):
    if not isinstance(digits, Integral) or isinstance(digits, bool) or digits < 8:
        raise ValueError("digits must be an integer of at least 8")


def check_dimension(dimension):
    if not isinstance(dimension, Integral) or isinstance(dimension, bool) or dimension < 1:
        raise ValueError("dimension must be a positive integer")


def centered_interval_moments(a, s, lower, upper):
    """Stable local series for a short interval; None outside its regime.

    Expand in t=(x-midpoint)/half_width, rather than subtracting endpoint
    antiderivatives. The same series supplies the mass and centered moments.
    The recurrence is valid for complex s as well as a real tilt.
    """
    if not (mp.isfinite(lower) and mp.isfinite(upper)) or lower >= upper:
        return None
    midpoint, half_width = (lower + upper) / 2, (upper - lower) / 2
    linear = (a * midpoint + s) * half_width
    quadratic = a * half_width * half_width / 2
    if abs(linear) + abs(quadratic) > mp.mpf("0.5"):
        return None
    sums = [mp.mpf(0), mp.mpf(0), mp.mpf(0)]
    previous, coefficient = mp.mpf(0), mp.mpf(1)
    for n in range(10 * mp.dps + 20):
        for k in range(3):
            if (n + k) % 2 == 0:
                sums[k] += 2 * coefficient / (n + k + 1)
        if n >= 2 and abs(coefficient) + abs(previous) < mp.eps * abs(sums[0]):
            mass = half_width * mp.exp(-a * midpoint * midpoint / 2 - s * midpoint) * sums[0]
            centered_mean = sums[1] / sums[0]
            mean = midpoint + half_width * centered_mean
            variance = half_width ** 2 * (sums[2] / sums[0] - centered_mean ** 2)
            return mass, mean, variance
        following = (-linear * coefficient - 2 * quadratic * previous) / (n + 1)
        previous, coefficient = coefficient, following
    raise ConvergenceError("Local Gaussian interval series did not converge")


def normal_interval(lower, upper, *, backend="mpmath"):
    """Standard normal mass, avoiding subtraction of nearly equal ones."""
    local = centered_interval_moments(1, 0, lower, upper)
    if local is not None:
        return local[0] / mp.sqrt(2 * mp.pi)
    if lower == upper:
        return mp.mpf(0)
    if upper <= 0:
        return normal_interval(-upper, -lower, backend=backend)
    if backend == "flint":
        from ._kernels import normal_interval as compiled_interval
        return compiled_interval(lower, upper)
    if lower >= 0:
        return (mp.erfc(lower / mp.sqrt(2)) - mp.erfc(upper / mp.sqrt(2))) / 2
    return (mp.erf(upper / mp.sqrt(2)) - mp.erf(lower / mp.sqrt(2))) / 2


def complement_guard_digits(lower, upper):
    """Extra precision needed for log(P) when a marginal mass is near one.

    A box's complement contains each marginal complement. Computing P
    with this many extra digits therefore also resolves its small log(P).
    Evaluate the two outside tails directly, without subtracting from one.
    """
    if lower >= 0 or upper <= 0:
        return 0
    outside = normal_interval(mp.ninf, lower) + normal_interval(upper, mp.inf)
    if outside == 0 or outside >= mp.mpf("0.01"):
        return 0
    return max(0, int(mp.ceil(-mp.log10(outside))))


def precision_errors(value, previous, check_log=False):
    relative = abs(value - previous) / value
    log_relative = mp.mpf(0)
    if 0 < value < 1 and 0 < previous < 1:
        log_relative = abs(mp.log(value) - mp.log(previous)) / abs(mp.log(value))
    elif check_log:
        log_relative = mp.inf
    return relative, log_relative


@dataclass
class ProbabilityResult:
    probability: Any
    log_probability: Any
    digits: int
    method: str
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    rtol: Any = None
    estimated_relative_error: Any = None

    def as_dict(self):
        with mp.workdps(self.digits + 10):
            output = {
                "probability": mp.nstr(self.probability, self.digits),
                "log_probability": mp.nstr(self.log_probability, self.digits),
                "digits": self.digits,
                "method": self.method,
                "diagnostics": self.diagnostics,
            }
            if self.rtol is not None:
                output.update(rtol=mp.nstr(self.rtol, self.digits),
                              estimated_relative_error=mp.nstr(self.estimated_relative_error, 8))
            return output


def result(value, digits, method, diagnostics):
    # Taking the logarithm after rounding a probability close to one can
    # erase it altogether. Retain the incoming arithmetic precision here.
    logarithm = mp.log(value) if 0 <= value <= 1 else None
    with mp.workdps(digits + 10):
        value = +value
        if not mp.isfinite(value) or value < 0 or value > 1:
            raise ConvergenceError("Computed value lies outside the probability range")
        # Permit an overshoot only when it rounds back into the valid range.
        if logarithm is None:
            logarithm = mp.log(value)
        return ProbabilityResult(value, +logarithm, digits, method, diagnostics)
