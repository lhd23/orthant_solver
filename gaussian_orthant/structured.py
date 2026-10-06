"""One-auxiliary-field reductions for exchangeable Gaussian families."""
from mpmath import mp

from ._bootloops import engines
from ._independent import independent_probability
from ._kernels import exponential_interval as compiled_exponential_interval, resolve_backend
from ._tolerance import (exact_rtol, finish_rtol, goal_digits, magnitude_guard,
                         resolve_accuracy)
from .common import (ConvergenceError, check_dimension,
                     centered_interval_moments, complement_guard_digits,
                     interval_guard_digits, normal_interval, number,
                     precision_errors, rational, result)


def _exponential_interval(a, s, lower, upper, backend="mpmath"):
    """Integral exp(-a*x*x/2-s*x) dx, also for complex s."""
    local = centered_interval_moments(a, s, lower, upper)
    if local is not None:
        return local[0]
    # Reflection avoids subtracting two erfc values close to two.
    if mp.re(a * (lower + upper) / 2 + s) < 0:
        return _exponential_interval(a, -s, -upper, -lower, backend)
    if backend == "flint":
        value = compiled_exponential_interval(a, s, lower, upper)
        return value if mp.im(s) else mp.re(value)
    root = mp.sqrt(2 * a)
    return (mp.sqrt(mp.pi / (2 * a)) * mp.exp(s * s / (2 * a)) *
            (mp.erfc((a * lower + s) / root) - mp.erfc((a * upper + s) / root)))


def _tilted_moments(a, lam, lower, upper, backend="mpmath"):
    local = centered_interval_moments(a, lam, lower, upper)
    if local is not None:
        return local
    mass = _exponential_interval(a, lam, lower, upper, backend)
    at_lower = mp.exp(-a * lower * lower / 2 - lam * lower) / mass
    at_upper = mp.exp(-a * upper * upper / 2 - lam * upper) / mass
    mean = (at_lower - at_upper - lam) / a
    second = (1 - lam * mean - upper * at_upper + lower * at_lower) / a
    variance = max(mp.mpf(0), second - mean * mean)
    return mass, mean, variance


def _saddle(a, b, dimension, lower, upper, digits, backend="mpmath"):
    left, right = b * dimension * lower, b * dimension * upper
    lam = (left + right) / 2
    for _ in range(120):
        mass, mean, variance = _tilted_moments(a, lam, lower, upper, backend)
        residual = lam / b - dimension * mean
        if abs(residual) < mp.power(10, -(digits + 10)) * max(1, abs(lam / b)):
            return lam, mass, variance, mean
        if residual > 0:
            right = lam
        else:
            left = lam
        candidate = lam - residual / (1 / b + dimension * variance)
        lam = candidate if left < candidate < right else (left + right) / 2
    raise ConvergenceError("Auxiliary-field saddle did not converge")


def _normal_moments(lower, upper, backend="mpmath"):
    local = centered_interval_moments(1, 0, lower, upper)
    if local is not None:
        return local[0] / mp.sqrt(2 * mp.pi), local[1], local[2]
    mass = normal_interval(lower, upper, backend=backend)
    at_lower = (mp.exp(-lower * lower / 2) / (mp.sqrt(2 * mp.pi) * mass)
                if mp.isfinite(lower) else mp.mpf(0))
    at_upper = (mp.exp(-upper * upper / 2) / (mp.sqrt(2 * mp.pi) * mass)
                if mp.isfinite(upper) else mp.mpf(0))
    mean = at_lower - at_upper
    second = (1 + (lower * at_lower if mp.isfinite(lower) else 0)
              - (upper * at_upper if mp.isfinite(upper) else 0))
    return mass, mean, min(mp.mpf(1), max(mp.mpf(0), second - mean * mean))


def _factor_peak(dimension, lower, upper, common, independent, digits, backend="mpmath"):
    """Locate the unique mode of the log-concave common-factor integrand."""
    ratio = common / independent

    def evaluate(z):
        mass, mean, variance = _normal_moments((lower - common * z) / independent,
                                             (upper - common * z) / independent, backend)
        score = -z + dimension * ratio * mean
        curvature = -1 - dimension * ratio * ratio * (1 - variance)
        return score, curvature, mass

    score, curvature, mass = evaluate(mp.mpf(0))
    if score == 0:
        return mp.mpf(0), 1 / mp.sqrt(-curvature), mass
    left, right = mp.mpf(0), mp.mpf(0)
    if score > 0:
        right = mp.mpf(1)
        while evaluate(right)[0] > 0:
            right *= 2
    else:
        left = mp.mpf(-1)
        while evaluate(left)[0] < 0:
            left *= 2
    z = (left + right) / 2
    for _ in range(120):
        score, curvature, mass = evaluate(z)
        width = 1 / mp.sqrt(-curvature)
        if abs(score) * width < mp.power(10, -(digits + 10)):
            return z, width, mass
        if score > 0:
            left = z
        else:
            right = z
        candidate = z - score / curvature
        z = candidate if left < candidate < right else (left + right) / 2
    raise ConvergenceError("Common-factor mode did not converge")


def _factor_scale_bound(dimension, common, independent, width):
    """Lower bound on the integral after centering and scaling.

    For the original log integrand h, h'' >= -K, where
    K = 1 + dimension*correlation/(1-correlation), because the conditional
    normal variance is nonnegative. Thus h(z0+width*w)-h(z0) is bounded
    below by h'(z0)*width*w - K*width**2*w**2/2. Integrating its exponential
    gives at least sqrt(2*pi/K)/width, even if z0 is not the exact mode.
    The factor 1/2 leaves numerical headroom when forming this scale.
    """
    curvature_bound = 1 + dimension * (common / independent) ** 2
    return mp.sqrt(2 * mp.pi / curvature_bound) / (2 * width)


def exchangeable_precision_probability(dimension, lower, upper, *,
                                       diagonal="0.5", coupling="0.5", mean=0,
                                       digits=None, rtol=None, backend="auto"):
    """P(lower <= X_i <= upper), Q = diagonal*I + coupling*11^T.

    Finite equal bounds, diagonal > 0, coupling >= 0. A contour shift of
    the Hubbard-Stratonovich auxiliary field centers its phase at a real
    saddle. The Gaussian tail has an analytic relative bound. Rule and
    precision agreement still give empirical quadrature errors, not a
    rigorous enclosure of the complete probability.
    Defaults to digits=30; alternatively rtol requests an estimated relative
    probability error with adaptive quadrature, without a logarithm target.
    """
    digits, rtol = resolve_accuracy(digits, rtol, 30)
    check_dimension(dimension)
    dimension = int(dimension)
    backend = resolve_backend(backend)
    exact_a, exact_b, exact_mu = map(rational, (diagonal, coupling, mean))
    exact_lo, exact_hi = map(rational, (lower, upper))
    if not all(x.is_finite for x in (exact_a, exact_b, exact_lo, exact_hi, exact_mu)):
        raise ValueError("Exchangeable precision reduction requires finite inputs")
    if exact_a <= 0 or exact_b < 0 or exact_lo > exact_hi:
        raise ValueError("Require diagonal > 0, coupling >= 0, and lower <= upper")
    if exact_lo == exact_hi:
        if rtol is not None:
            return exact_rtol(mp.mpf(0), digits, "zero-width box", {}, rtol)
        return result(mp.mpf(0), digits, "zero-width box", {})
    exact_lo, exact_hi = exact_lo - exact_mu, exact_hi - exact_mu
    if dimension == 1 or exact_b == 0:
        diagonal_precision = exact_a + exact_b if dimension == 1 else exact_a
        return independent_probability((exact_lo,), (exact_hi,), (diagonal_precision,),
                                       digits=digits, backend=backend, repeat=dimension, rtol=rtol)
    cancellation_digits = interval_guard_digits(exact_lo, exact_hi)
    if rtol is not None:
        return _exchangeable_rtol(dimension, exact_lo, exact_hi, exact_a, exact_b,
                                  digits, rtol, cancellation_digits, backend)
    with mp.workdps(digits + 25 + cancellation_digits):
        a, b = number(exact_a), number(exact_b)
        marginal_sigma = mp.sqrt((a + b * (dimension - 1)) / (a * (a + b * dimension)))
        log_guard = complement_guard_digits(number(exact_lo) / marginal_sigma,
                                            number(exact_hi) / marginal_sigma)
    _, quadrature, _ = engines()
    previous = None
    records = []
    for extra in (25, 45, 85):
        work_digits = digits + extra + cancellation_digits + log_guard
        with mp.workdps(work_digits):
            a, b = number(exact_a), number(exact_b)
            lo, hi = number(exact_lo), number(exact_hi)
            lam, mass, variance, tilted_mean = _saddle(a, b, dimension, lo, hi, digits, backend)
            log_prefactor = (((dimension - 1) * mp.log(a) + mp.log(a + b * dimension)) / 2
                             - dimension * mp.log(2 * mp.pi) / 2
                             + lam * lam / (2 * b) + dimension * mp.log(mass))
            width = mp.sqrt(b / (1 + b * dimension * variance))
            # Popoviciu's bound Var(S) <= d*(upper-lower)^2/4 and
            # Jensen give K >= exp(-b*Var(S)/2). This lower bound makes
            # the discarded Fourier-Gaussian tail relative, even when
            # the final probability is extremely small.
            centering_residual = dimension * tilted_mean - lam / b
            variance_bound = dimension * (hi - lo) ** 2 / 4 + centering_residual ** 2
            tail_target = mp.power(10, -(digits + log_guard + 12))
            cutoff = mp.sqrt(2 * b * ((digits + log_guard + 14) * mp.log(10) + b * variance_bound / 2))
            relative_tail_bound = mp.erfc(cutoff / mp.sqrt(2 * b)) * mp.exp(b * variance_bound / 2)
            while relative_tail_bound > tail_target:
                cutoff *= mp.mpf("1.1")
                relative_tail_bound = mp.erfc(cutoff / mp.sqrt(2 * b)) * mp.exp(b * variance_bound / 2)

            def integrand(z):
                y = width * z
                ratio = _exponential_interval(a, lam - 1j * y, lo, hi, backend) / mass
                phase = mp.exp(-1j * lam * y / b) * ratio ** dimension
                return mp.exp(-y * y / (2 * b)) * mp.re(phase)

            end = cutoff / width
            # Separate the saddle peak from the rigorously bounded tails.
            points = sorted(set([mp.mpf(0), min(end, mp.mpf(1)), min(end, mp.mpf(4)),
                                 min(end, mp.mpf(10)), end]))
            quadrature_digits = digits + (extra - 15) // 2 + log_guard
            certificate = quadrature(integrand, points, quadrature_digits, guard=5,
                                     wp_extra=(extra + 5) // 2 + cancellation_digits, depth0=2,
                                     full_output=True)
            reduced = 2 * width * certificate.value / mp.sqrt(2 * mp.pi * b)
            value = mp.exp(log_prefactor) * reduced
            records.append({
                "work_digits": work_digits,
                "centering_residual": mp.nstr(centering_residual, 8),
                "quadrature_depths": list(certificate.depths),
                "relative_analytic_tail_bound": mp.nstr(relative_tail_bound, 8),
                "quadrature_agreement_estimate": mp.nstr(certificate.agreement, 8),
            })
            if previous is not None and value > 0:
                agreement, log_agreement = precision_errors(value, previous, bool(log_guard))
                if max(agreement, log_agreement) < mp.power(10, -digits):
                    return result(value, digits, "one auxiliary field / BootLoops quadrature", {
                        "dimension": int(dimension), "saddle": mp.nstr(lam, 12),
                        "backend": backend,
                        "relative_precision_agreement": mp.nstr(agreement, 8),
                        "relative_log_precision_agreement": mp.nstr(log_agreement, 8),
                        "runs": records,
                        "error_status": "analytic Fourier tail bound; empirical quadrature and precision agreement",
                    })
            previous = value
    raise ConvergenceError("Auxiliary-field reduction did not pass relative precision agreement")


def equicorrelated_probability(dimension, lower=0, upper="inf", *,
                              correlation="0.5", standard_deviation=1,
                              mean=0, digits=None, rtol=None, backend="auto"):
    """One-factor Gaussian reduction, with equal bounds and nonnegative correlation.

    X_i = mean + standard_deviation*(sqrt(correlation)*Z
          + sqrt(1-correlation)*Z_i), with all Z independent standard normals.
    Defaults to digits=30. Specify rtol instead for faster estimated relative
    probability accuracy; no separate log_probability accuracy is imposed.
    """
    digits, rtol = resolve_accuracy(digits, rtol, 30)
    check_dimension(dimension)
    dimension = int(dimension)
    backend = resolve_backend(backend)
    exact_rho, exact_sigma, exact_mu = map(rational, (correlation, standard_deviation, mean))
    exact_lo, exact_hi = map(rational, (lower, upper))
    if not all(x.is_finite for x in (exact_rho, exact_sigma, exact_mu)):
        raise ValueError("Correlation, standard deviation and mean must be finite")
    if (not 0 <= exact_rho < 1 or exact_sigma <= 0
            or not (exact_lo.is_real or exact_lo.is_infinite)
            or not (exact_hi.is_real or exact_hi.is_infinite) or exact_lo > exact_hi):
        raise ValueError("Require 0 <= correlation < 1, standard_deviation > 0 and valid bounds")
    if exact_lo == exact_hi:
        if rtol is not None:
            return exact_rtol(mp.mpf(0), digits, "zero-width box", {}, rtol)
        return result(mp.mpf(0), digits, "zero-width box", {})
    exact_lo, exact_hi = (exact_lo - exact_mu) / exact_sigma, (exact_hi - exact_mu) / exact_sigma
    if dimension == 1 or exact_rho == 0 or (exact_lo.is_infinite and exact_hi.is_infinite):
        return independent_probability((exact_lo,), (exact_hi,), (rational(1),),
                                       digits=digits, backend=backend, repeat=dimension, rtol=rtol)
    cancellation_digits = interval_guard_digits(exact_lo, exact_hi)
    if rtol is not None:
        return _equicorrelated_rtol(dimension, exact_lo, exact_hi, exact_rho,
                                    digits, rtol, cancellation_digits, backend)
    with mp.workdps(digits + 25 + cancellation_digits):
        log_guard = complement_guard_digits(number(exact_lo), number(exact_hi))
    _, quadrature, _ = engines()
    previous = None
    runs = []
    for extra in (25, 45, 85):
        work_digits = digits + extra + cancellation_digits + log_guard
        with mp.workdps(work_digits):
            lo, hi = number(exact_lo), number(exact_hi)
            common, independent = mp.sqrt(number(exact_rho)), mp.sqrt(number(1 - exact_rho))
            mode, width, peak_mass = _factor_peak(dimension, lo, hi, common, independent, digits, backend)
            log_peak_mass = mp.log(peak_mass)
            log_peak = -mode * mode / 2 - mp.log(2 * mp.pi) / 2 + dimension * log_peak_mass

            def integrand(w):
                z = mode + width * w
                mass = normal_interval((lo - common * z) / independent,
                                       (hi - common * z) / independent, backend=backend)
                return mp.exp(-(z - mode) * (z + mode) / 2
                              + dimension * (mp.log(mass) - log_peak_mass))

            # The curvature bound provides a conservative tolerance scale,
            # avoiding a preliminary numerical integration of the same function.
            points = [mp.ninf, -10, -4, -1, 0, 1, 4, 10, mp.inf]
            scale = _factor_scale_bound(dimension, common, independent, width)
            if not mp.isfinite(scale) or scale <= 0:
                raise ConvergenceError("One-factor quadrature could not resolve the probability scale")
            quadrature_digits = digits + (extra - 15) // 2 + log_guard
            certificate = quadrature(integrand, points, quadrature_digits,
                                     guard=5, wp_extra=(extra + 5) // 2 + cancellation_digits,
                                     depth0=2, scale=scale,
                                     max_depth=12, full_output=True)
            value = mp.exp(log_peak) * width * certificate.value
            runs.append({"work_digits": work_digits,
                         "mode": mp.nstr(mode, 12), "width": mp.nstr(width, 12),
                         "scale_source": "analytic curvature lower bound",
                         "quadrature_scale_lower_bound": mp.nstr(scale, 12),
                         "quadrature_depths": list(certificate.depths),
                         "relative_quadrature_agreement_estimate": mp.nstr(certificate.agreement / certificate.value, 8)})
            if previous is not None:
                relative, log_relative = precision_errors(value, previous, bool(log_guard))
                if max(relative, log_relative) < mp.power(10, -digits):
                    return result(value, digits, "one Gaussian factor / BootLoops quadrature", {
                        "dimension": int(dimension), "backend": backend, "runs": runs,
                        "relative_precision_agreement": mp.nstr(relative, 8),
                        "relative_log_precision_agreement": mp.nstr(log_relative, 8),
                        "error_status": "quadrature and precision agreement; not an interval certificate",
                    })
            previous = value
    raise ConvergenceError("One-factor quadrature did not pass relative precision agreement")


def _quadrature_rtol(build, dimension, digits, rtol, guard, backend, method):
    """Let nested BootLoops rules resolve integration before raising precision.

    Agreement, representation rounding and roundoff are estimated separately
    from the analytic tail. A single successful nested refinement suffices;
    the legacy second full high-precision integration is not required.
    """
    _, quadrature, _ = engines()
    goal, runs = goal_digits(rtol), []
    for extra in (0, 4, 12, 28, 60):
        quadrature_digits = goal + 1 + extra
        work_digits = quadrature_digits + 17 + guard
        with mp.workdps(work_digits):
            integrand, points, scale, log_prefactor, tail, metadata = build(goal, rtol)
            try:
                certificate = quadrature(integrand, points, quadrature_digits,
                                         guard=1, wp_extra=16 + guard, depth0=2,
                                         max_depth=12, scale=scale, full_output=True)
            except RuntimeError as error:
                if type(error).__name__ != "QuadNonConvergence":
                    raise
                runs.append({"work_digits": work_digits, "failure": str(error)})
                continue
            integral = mp.re(certificate.value)
            if not mp.isfinite(integral) or integral <= 0:
                runs.append({"work_digits": work_digits, "failure": "nonpositive integral"})
                continue
            value = mp.exp(log_prefactor) * integral
            arithmetic = mp.power(10, -(work_digits - 6)) * max(
                1, dimension, abs(log_prefactor))
            estimate = 4 * certificate.err_bound / integral + tail + arithmetic
            runs.append({"work_digits": work_digits,
                         "quadrature_digits": quadrature_digits,
                         "quadrature_depths": list(certificate.depths),
                         "integrand_evaluations": certificate.evals,
                         "relative_quadrature_error_estimate": mp.nstr(
                             4 * certificate.err_bound / integral, 8),
                         "relative_analytic_tail_bound": mp.nstr(tail, 8),
                         **metadata})
            if mp.isfinite(value) and 0 < value <= 1 and estimate < rtol / 2:
                return finish_rtol(value, digits, method,
                                   {"dimension": dimension, "backend": backend,
                                    "runs": runs}, rtol, estimate)
    raise ConvergenceError("Structured quadrature did not meet rtol")


def _equicorrelated_rtol(dimension, lower, upper, correlation, digits, rtol,
                         cancellation, backend):
    guard = cancellation + magnitude_guard(lower, upper) + len(str(dimension))

    def build(goal, tolerance):
        lo, hi = number(lower), number(upper)
        common, independent = mp.sqrt(number(correlation)), mp.sqrt(number(1 - correlation))
        mode, width, peak_mass = _factor_peak(dimension, lo, hi, common, independent,
                                             goal, backend)
        log_peak_mass = mp.log(peak_mass)
        log_peak = -mode * mode / 2 - mp.log(2 * mp.pi) / 2 + dimension * log_peak_mass

        def integrand(w):
            z = mode + width * w
            mass = normal_interval((lo - common * z) / independent,
                                   (hi - common * z) / independent, backend=backend)
            return mp.exp(-(z - mode) * (z + mode) / 2
                          + dimension * (mp.log(mass) - log_peak_mass))

        scale = _factor_scale_bound(dimension, common, independent, width)
        # Splitting at the centred peak resolves both tails without eight
        # separately refined segments. No finite tail is discarded here.
        return (integrand, [mp.ninf, 0, mp.inf], scale, log_peak + mp.log(width),
                mp.mpf(0), {"mode": mp.nstr(mode, 12), "width": mp.nstr(width, 12),
                            "scale_source": "analytic curvature lower bound"})

    return _quadrature_rtol(build, dimension, digits, rtol, guard, backend,
                            "one Gaussian factor / BootLoops quadrature")


def _exchangeable_rtol(dimension, lower, upper, diagonal, coupling, digits, rtol,
                       cancellation, backend):
    guard = cancellation + magnitude_guard(lower, upper, diagonal, coupling)
    guard += len(str(dimension))

    def build(goal, tolerance):
        a, b = number(diagonal), number(coupling)
        lo, hi = number(lower), number(upper)
        lam, mass, variance, mean = _saddle(a, b, dimension, lo, hi, goal, backend)
        width = mp.sqrt(b / (1 + b * dimension * variance))
        log_prefactor = (((dimension - 1) * mp.log(a) + mp.log(a + b * dimension)) / 2
                         - dimension * mp.log(2 * mp.pi) / 2
                         + lam * lam / (2 * b) + dimension * mp.log(mass)
                         + mp.log(2 * width / mp.sqrt(2 * mp.pi * b)))
        residual = dimension * mean - lam / b
        variance_bound = dimension * (hi - lo) ** 2 / 4 + residual ** 2
        tail_target = tolerance / 16
        cutoff = mp.sqrt(2 * b * (mp.log(32 / tolerance) + b * variance_bound / 2))
        tail = mp.erfc(cutoff / mp.sqrt(2 * b)) * mp.exp(b * variance_bound / 2)
        while tail > tail_target:
            cutoff *= mp.mpf("1.1")
            tail = mp.erfc(cutoff / mp.sqrt(2 * b)) * mp.exp(b * variance_bound / 2)

        def integrand(z):
            y = width * z
            ratio = _exponential_interval(a, lam - 1j * y, lo, hi, backend) / mass
            return mp.exp(-y * y / (2 * b)) * mp.re(mp.exp(-1j * lam * y / b)
                                                  * ratio ** dimension)

        end = cutoff / width
        points = sorted(set([mp.mpf(0), min(end, mp.mpf(1)), min(end, mp.mpf(4)), end]))
        # Jensen's lower bound on the reduced integral supplies a relative
        # scale even for oscillatory integrands and extremely rare boxes.
        scale = mp.sqrt(2 * mp.pi * b) / (2 * width) * mp.exp(-b * variance_bound / 2)
        return integrand, points, scale, log_prefactor, tail, {
            "saddle": mp.nstr(lam, 12), "centering_residual": mp.nstr(residual, 8)}

    return _quadrature_rtol(build, dimension, digits, rtol, guard, backend,
                            "one auxiliary field / BootLoops quadrature")
