"""Optional compiled arithmetic; no loss of precision through machine floats."""
from functools import lru_cache
from importlib import import_module

from mpmath import mp


@lru_cache(maxsize=1)
def _flint():
    try:
        module = import_module("flint")
    except ImportError:
        return None
    required = ("arb", "acb", "acb_poly", "acb_series", "ctx")
    if not all(hasattr(module, name) for name in required):
        return None
    return module


def resolve_backend(backend):
    if backend not in ("auto", "mpmath", "flint"):
        raise ValueError("backend must be 'auto', 'mpmath', or 'flint'")
    if backend == "mpmath":
        return backend
    if _flint() is not None:
        return "flint"
    if backend == "flint":
        raise ImportError("backend='flint' requires python-flint with complex power-series support")
    return "mpmath"


def _dyadic_ball(components, module):
    # An mpmath value is an exact dyadic mantissa times a power of two.
    # Rebuilding that value avoids conversion through float or decimal text.
    sign, mantissa, exponent, _ = components
    if not mantissa:
        if exponent:
            raise ValueError("Compiled kernel requires finite arguments")
        return module.arb(0)
    return module.arb(-int(mantissa) if sign else int(mantissa)) * module.arb(2) ** int(exponent)


def _real_ball(value, module):
    return _dyadic_ball(mp.mpf(value)._mpf_, module)


def _complex_ball(value, module):
    return module.acb(_real_ball(mp.re(value), module), _real_ball(mp.im(value), module))


def _complex_key(value):
    """Exact binary components, without precision-dependent text rounding."""
    return mp.re(value)._mpf_, mp.im(value)._mpf_


@lru_cache(maxsize=32)
def _normal_root(module, precision):
    """Reuse sqrt(2), retaining its ball radius at the requested precision."""
    if module.ctx.prec != precision:
        raise ValueError("Compiled kernel constant precision mismatch")
    return module.acb(2).sqrt()


@lru_cache(maxsize=64)
def _exponential_constants(module, precision, a_key, lower_key, upper_key):
    """Fixed ball quantities, cached separately at each compiled precision.

    Called inside _evaluate after it sets the compiled arithmetic context.
    Cached balls retain their radii and are only read by the evaluator.
    The bounded cache does not retain quadrature nodes or changing shifts.
    """
    if module.ctx.prec != precision:
        raise ValueError("Compiled kernel constant precision mismatch")

    def convert(key):
        return module.acb(_dyadic_ball(key[0], module), _dyadic_ball(key[1], module))

    aa, lo, hi = convert(a_key), convert(lower_key), convert(upper_key)
    quadratic = 2 * aa
    root = quadratic.sqrt()
    prefactor = (module.acb.pi() / quadratic).sqrt()
    return aa * lo, aa * hi, quadratic, root, prefactor


def _midpoint(value):
    components = []
    for part in (value.real.mid(), value.imag.mid()):
        mantissa, exponent = part.man_exp()
        components.append(mp.ldexp(mp.mpf(int(mantissa)), int(exponent)))
    return mp.mpc(*components)


def _evaluate(evaluator):
    """Refine the ball evaluation until its midpoint resolves mp.prec bits.

    Radii are checked before discarding them. This local check does not
    turn the complete solver into an interval-certified calculation.
    Both precision and series-cap context settings are restored on failure.
    """
    from .common import ConvergenceError
    module = _flint()
    if module is None:
        raise ImportError("Compiled Gaussian kernels require python-flint")
    target = mp.prec
    old_precision, old_cap = module.ctx.prec, module.ctx.cap
    try:
        for multiplier in (1, 2, 4, 8, 16):
            module.ctx.prec = multiplier * (target + 32)
            value = evaluator(module)
            if value.is_finite() and (value.is_exact() or value.rel_accuracy_bits() >= target):
                return _midpoint(value)
    finally:
        module.ctx.prec, module.ctx.cap = old_precision, old_cap
    raise ConvergenceError("Compiled Gaussian kernel could not resolve the requested precision")


def normal_interval(lower, upper):
    def evaluate(module):
        root = _normal_root(module, module.ctx.prec)
        left = module.acb(2) if lower == mp.ninf else (_complex_ball(lower, module) / root).erfc()
        right = module.acb(0) if upper == mp.inf else (_complex_ball(upper, module) / root).erfc()
        return (left - right) / 2
    return mp.re(_evaluate(evaluate))


def exponential_interval(a, s, lower, upper):
    a_key, lower_key, upper_key = map(_complex_key, (a, lower, upper))

    def evaluate(module):
        left, right, quadratic, root, prefactor = _exponential_constants(
            module, module.ctx.prec, a_key, lower_key, upper_key)
        ss = _complex_ball(s, module)
        difference = ((left + ss) / root).erfc() - ((right + ss) / root).erfc()
        return prefactor * (ss * ss / quadratic).exp() * difference
    return _evaluate(evaluate)
