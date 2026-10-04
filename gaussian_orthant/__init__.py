"""Deterministic Gaussian orthant and box probabilities using BootLoops.

The general interface is gaussian_probability(lower, upper, covariance=...).
Use orthant_probability for one-sided bounds. Structured families have
explicit dimension-only interfaces that avoid exponential boundary bases.
"""
from .common import ProbabilityResult, ConvergenceError
from .transport import transport_probability
from .structured import exchangeable_precision_probability, equicorrelated_probability


def gaussian_probability(lower, upper, *, covariance=None, precision=None,
                         mean=None, digits=25, max_masters=256, backend="auto"):
    """Evaluate P(lower <= X <= upper), X Gaussian with the supplied mean.

    Exactly one of covariance and precision is required. Infinite endpoints
    are accepted. Correlated problems use boundary-master transport;
    independent coordinates use direct Gaussian formulas. The backend is
    "auto" (compiled when available), "mpmath", or "flint".
    """
    return transport_probability(lower, upper, covariance=covariance,
                                 precision=precision, mean=mean, digits=digits,
                                 max_masters=max_masters, backend=backend)


def orthant_probability(covariance=None, *, precision=None, lower=None,
                        upper=None, mean=None, digits=25, max_masters=256,
                        backend="auto"):
    """Positive or negative one-sided Gaussian orthant probability.

    Default event is X_i >= 0. Set lower for X_i >= lower_i, or set upper
    for X_i <= upper_i. Use gaussian_probability for two finite bounds.
    """
    if lower is not None and upper is not None:
        raise ValueError("Use gaussian_probability when supplying both lower and upper")
    matrix = covariance if covariance is not None else precision
    if matrix is None:
        raise ValueError("Supply covariance or precision")
    d = len(matrix)
    if upper is not None:
        lo, hi = ["-inf"] * d, upper
    else:
        lo, hi = ([0] * d if lower is None else lower), ["inf"] * d
    return gaussian_probability(lo, hi, covariance=covariance, precision=precision,
                                mean=mean, digits=digits, max_masters=max_masters,
                                backend=backend)


__all__ = ["ProbabilityResult", "ConvergenceError", "gaussian_probability",
           "orthant_probability", "exchangeable_precision_probability",
           "equicorrelated_probability"]
