"""Independent truths, invariances and refusal paths for the new solver."""
from fractions import Fraction

import numpy as np
import pytest
from mpmath import mp
from numpy.polynomial.legendre import leggauss

from gaussian_orthant import (gaussian_probability, orthant_probability,
                             equicorrelated_probability,
                             exchangeable_precision_probability)
from gaussian_orthant.transport import BoundarySystem, prepare


def close(actual, expected, digits=20):
    with mp.workdps(digits + 20):
        assert abs(actual - expected) < mp.power(10, -digits) * abs(expected)


@pytest.mark.parametrize("rho", ["-0.8", "0.3", "0.75"])
def test_bivariate_arcsine_truth(rho):
    with mp.workdps(50):
        expected = mp.mpf(1) / 4 + mp.asin(mp.mpf(rho)) / (2 * mp.pi)
    value = orthant_probability([[1, rho], [rho, 1]], digits=22)
    close(value.probability, expected)


def test_trivariate_arcsine_truth():
    covariance = [[1, "0.2", "-0.15"], ["0.2", 1, "0.35"], ["-0.15", "0.35", 1]]
    with mp.workdps(50):
        expected = mp.mpf(1) / 8 + sum(mp.asin(mp.mpf(x)) for x in ["0.2", "-0.15", "0.35"]) / (4 * mp.pi)
    value = orthant_probability(covariance, digits=22)
    close(value.probability, expected)


def test_noncentral_independent_and_infinite_bounds():
    value = gaussian_probability(["-inf", 2], ["0.25", "inf"], covariance=[[4, 0], [0, 9]],
                                 mean=[1, -1], digits=22)
    with mp.workdps(50):
        expected = mp.erfc(mp.mpf("0.375") / mp.sqrt(2)) / 2 * mp.erfc(1 / mp.sqrt(2)) / 2
    close(value.probability, expected)


def tensor_reference(lower, upper, precision, mean=None, order=24):
    """Independent positive Gauss-Legendre product integration."""
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    q = np.asarray(precision, float)
    d = len(lower)
    nodes, weights = leggauss(order)
    coordinates = [lower[i] + (nodes + 1) * (upper[i] - lower[i]) / 2 for i in range(d)]
    rules = [weights * (upper[i] - lower[i]) / 2 for i in range(d)]
    x = np.stack(np.meshgrid(*coordinates, indexing="ij"), axis=-1)
    if mean is not None:
        x = x - np.asarray(mean, float)
    weight = np.ones(x.shape[:-1])
    for w in np.meshgrid(*rules, indexing="ij"):
        weight *= w
    exponent = np.einsum("...i,ij,...j->...", x, q, x)
    return np.sqrt(np.linalg.det(q)) / (2 * np.pi) ** (d / 2) * np.sum(weight * np.exp(-exponent / 2))


def test_general_noncentral_box_against_direct_integrand():
    q = [["1.7", "0.2", "-0.1"], ["0.2", "1.3", "0.25"], ["-0.1", "0.25", "1.1"]]
    lower, upper, mean = [-1, "0.1", "-0.7"], ["0.6", "0.9", "0.2"], ["0.2", "-0.1", "0.15"]
    value = gaussian_probability(lower, upper, precision=q, mean=mean, digits=20)
    references = [tensor_reference(lower, upper, q, mean, order=n) for n in (16, 24)]
    assert abs(references[0] / references[1] - 1) < 2e-14
    assert abs(float(value.probability) / references[1] - 1) < 2e-13


def test_permutation_covariance_precision_and_upper_orthants():
    covariance = [[Fraction(3, 2), Fraction(1, 4)], [Fraction(1, 4), 1]]
    precision = [[Fraction(16, 23), Fraction(-4, 23)], [Fraction(-4, 23), Fraction(24, 23)]]
    first = orthant_probability(covariance, lower=["0.2", "-0.3"], mean=["0.1", "0.4"], digits=22)
    second = orthant_probability(precision=precision, lower=["0.2", "-0.3"], mean=["0.1", "0.4"], digits=22)
    third = orthant_probability([[1, Fraction(1, 4)], [Fraction(1, 4), Fraction(3, 2)]],
                               upper=["0.3", "-0.2"], mean=["-0.4", "-0.1"], digits=22)
    close(first.probability, second.probability)
    close(first.probability, third.probability)


def test_empty_and_full_space():
    assert gaussian_probability([0], [0], covariance=[[1]]).probability == 0
    value = gaussian_probability(["-inf"] * 2, ["inf"] * 2,
                                 covariance=[[1, "0.3"], ["0.3", 1]], digits=20)
    close(value.probability, mp.mpf(1), 18)


def test_exchangeable_face_reduction_matches_auxiliary_field():
    q = [[1, "0.5"], ["0.5", 1]]
    transport = gaussian_probability(["0.5"] * 2, [1] * 2, precision=q, digits=22)
    field = exchangeable_precision_probability(2, "0.5", 1, digits=22)
    close(transport.probability, field.probability)
    assert transport.diagnostics["masters"] == 6


def test_equicorrelated_exact_truth_and_nonzero_bounds():
    value = equicorrelated_probability(100, digits=22)
    with mp.workdps(50):
        expected = mp.mpf(1) / 101
    close(value.probability, expected)
    value = equicorrelated_probability(2, "-0.2", "0.8", correlation="0.3", mean="0.1", digits=22)
    transported = gaussian_probability(["-0.2"] * 2, ["0.8"] * 2,
                                      covariance=[[1, "0.3"], ["0.3", 1]], mean=["0.1"] * 2, digits=22)
    close(value.probability, transported.probability)


def test_log_probability_retains_rare_event_scale():
    value = exchangeable_precision_probability(50, "0.5", 1, digits=22)
    assert mp.mpf("2e-153") < value.probability < mp.mpf("2.3e-153")
    with mp.workdps(45):
        close(mp.exp(value.log_probability), value.probability)


@pytest.mark.parametrize("kwargs", [
    {"covariance": [[1, 2], [2, 1]]},
    {"covariance": [[1, 0], [1, 1]]},
    {"covariance": [[1, "nan"], ["nan", 1]]},
    {"covariance": [[1]]},
    {"covariance": [[1, 0], [0, 1]], "precision": [[1, 0], [0, 1]]},
    {"covariance": [[1, 0], [0, 1]], "digits": 2},
])
def test_invalid_inputs_are_refused(kwargs):
    with pytest.raises(ValueError):
        gaussian_probability([0, 0], [1, 1], **kwargs)


def test_dimension_and_bound_refusals():
    with pytest.raises(ValueError):
        gaussian_probability([1], [0], covariance=[[1]])
    with pytest.raises(ValueError):
        exchangeable_precision_probability(2, 0, "inf")
    with pytest.raises(ValueError):
        equicorrelated_probability(0)
    with pytest.raises(ValueError):
        orthant_probability([[1]], lower=[0], upper=[1])
    with pytest.raises(ValueError, match="masters"):
        gaussian_probability([0] * 6, [1] * 6,
                             precision=[[int(i == j) + Fraction(1, (i + j + 2) * 10)
                                         for j in range(6)] for i in range(6)])


def test_ambient_precision_is_restored_and_connection_derivative_is_independent():
    initial = mp.dps
    orthant_probability([[1, "0.2"], ["0.2", 1]], digits=20)
    assert mp.dps == initial
    # A held-out point on the homotopy: integrate each master directly,
    # then compare A(t)J with finite differences of the original integrand.
    from scipy.integrate import quad
    q, lo, hi = prepare(["-0.3", "0.1"], ["0.6", "0.9"], None,
                         [[1, "0.25"], ["0.25", "1.2"]], None, 20)
    system = BoundarySystem(q, lo, hi)
    with mp.workdps(60):
        connection, _ = system.normalized(60)
        qt = np.array(q.tolist(), float)
        e = qt - np.diag(np.diag(qt))
        diag = np.diag(np.diag(qt))

        def direct(face, t):
            free = [i for i, flag in enumerate(face) if flag == 0]
            x = np.array([0 if not flag else float(lo[i] if flag == 1 else hi[i])
                          for i, flag in enumerate(face)], dtype=float)
            def integrate(k):
                if k == len(free):
                    return np.exp(-x @ (diag + t * e) @ x / 2)
                i = free[k]
                def f(z):
                    x[i] = z
                    return integrate(k + 1)
                return quad(f, float(lo[i]), float(hi[i]), epsabs=1e-13, epsrel=1e-13)[0]
            return integrate(0)

        anchors = np.array([direct(face, 0) for face in system.faces])
        t, h = 0.37, 1e-5
        y = np.array([direct(face, t) for face in system.faces]) / anchors
        derivative = np.array([(direct(face, t + h) - direct(face, t - h)) / (2 * h)
                               for face in system.faces]) / anchors
        matrix = np.array([[complex(z).real for z in row] for row in connection.A(mp.mpf(str(t)), 0, 50)])
        assert np.max(abs(matrix @ y - derivative)) < 2e-10


def test_comparison_gate_detects_perturbed_truth():
    actual = orthant_probability([[1, "0.5"], ["0.5", 1]], digits=20)
    with mp.workdps(40):
        truth = mp.mpf(1) / 3
        close(actual.probability, truth, 18)
        with pytest.raises(AssertionError):
            close(actual.probability, truth * mp.mpf("1.001"), 18)


def test_independent_minimax_implementation_recovers_bivariate_box():
    from validation_oracles import minimax_reference
    expected = 0.014896313886064507
    baseline = minimax_reference([0.5] * 2, [1] * 2, [[1, 0.5], [0.5, 1]], power=10)
    assert abs(baseline["probability"] / expected - 1) < max(8 * baseline["relative_standard_error"], 1e-10)


@pytest.mark.parametrize("solver", [exchangeable_precision_probability, equicorrelated_probability])
@pytest.mark.parametrize("dimension", [1, 2])
def test_structured_tiny_nonempty_interval(solver, dimension):
    upper = "1." + "0" * 59 + "1"  # Width 10^-60, below the old input precision.
    value = solver(dimension, 1, upper, digits=25)
    with mp.workdps(110):
        d, width = dimension, mp.mpf("1e-60")
        if solver is exchangeable_precision_probability:
            determinant = mp.mpf("0.5") ** (d - 1) * (mp.mpf("0.5") + mp.mpf("0.5") * d)
            exponent = -d * (mp.mpf("0.5") + mp.mpf("0.5") * d) / 2
        else:
            determinant = 1 / (mp.mpf("0.5") ** (d - 1) * (1 + mp.mpf("0.5") * (d - 1)))
            exponent = -d / (2 * (1 + mp.mpf("0.5") * (d - 1)))
        # The density varies by order 10^-60 over this box. Its value at
        # the lower corner times volume is an independent 25-digit truth.
        expected = mp.sqrt(determinant) * mp.exp(exponent) * width ** d / (2 * mp.pi) ** (mp.mpf(d) / 2)
        close(value.probability, expected, 25)
        assert mp.isfinite(value.log_probability)


@pytest.mark.parametrize("solver", [exchangeable_precision_probability, equicorrelated_probability])
def test_structured_exact_order_and_zero_width(solver):
    upper = "1." + "0" * 59 + "1"
    with pytest.raises(ValueError):
        solver(1, upper, 1, digits=25)
    assert solver(1, upper, upper, digits=25).probability == 0


@pytest.mark.parametrize("solver,kwargs", [
    (exchangeable_precision_probability, {"diagonal": 1, "coupling": 0}),
    (equicorrelated_probability, {"correlation": 0}),
])
def test_independent_structured_branch_retains_narrow_interval_digits(solver, kwargs):
    upper = "2." + "0" * 39 + "1"
    value = solver(2, 2, upper, mean=1, digits=25, **kwargs)
    with mp.workdps(100):
        expected = (mp.exp(-mp.mpf("0.5")) / mp.sqrt(2 * mp.pi) * mp.mpf("1e-40")) ** 2
        close(value.probability, expected, 25)
    assert len(value.diagnostics["runs"]) >= 2


@pytest.mark.parametrize("threshold", [12, 13])
def test_near_one_probability_logarithm_retains_requested_digits(threshold):
    value = orthant_probability([[1]], lower=[-threshold], digits=25)
    with mp.workdps(100):
        expected = mp.log1p(-mp.erfc(mp.mpf(threshold) / mp.sqrt(2)) / 2)
        close(value.log_probability, expected, 25)


def test_result_takes_logarithm_before_rounding_probability():
    from gaussian_orthant.common import result
    with mp.workdps(100):
        probability = 1 - mp.mpf("1e-40")
        value = result(probability, 25, "test", {})
        assert value.probability == 1
        close(value.log_probability, mp.log1p(-mp.mpf("1e-40")), 25)


@pytest.mark.parametrize("solver,kwargs,upper,tail_multiplier", [
    (equicorrelated_probability, {}, "inf", 1),
    (equicorrelated_probability, {"correlation": 0}, "inf", 1),
    (exchangeable_precision_probability, {}, 13, 2),
    (exchangeable_precision_probability, {"diagonal": 1, "coupling": 0}, 13, 2),
])
def test_structured_near_one_logarithm_requires_tighter_integration(solver, kwargs, upper, tail_multiplier):
    value = solver(1, -13, upper, digits=25, **kwargs)
    with mp.workdps(110):
        tail = mp.erfc(13 / mp.sqrt(2)) / 2
        close(value.log_probability, mp.log1p(-tail_multiplier * tail), 25)
    assert mp.mpf(value.diagnostics["relative_log_precision_agreement"]) < mp.mpf("1e-25")


@pytest.mark.parametrize("structured", [False, True])
def test_multivariate_near_one_logarithm(structured):
    if structured:
        value = equicorrelated_probability(2, -13, correlation="0.1", digits=25)
    else:
        value = orthant_probability([[1, "0.1"], ["0.1", 1]], lower=[-13, -13], digits=25)
    with mp.workdps(110):
        tail = mp.erfc(13 / mp.sqrt(2)) / 2
        # The omitted joint tail is bounded by P(X1+X2 >= 26), less
        # than 10^-29 of the marginal complement for correlation 0.1.
        close(value.log_probability, mp.log1p(-2 * tail), 25)


@pytest.mark.parametrize("lower,upper", [(10, 11), (-11, -10)])
def test_univariate_tail_uses_direct_formula(lower, upper):
    value = equicorrelated_probability(1, lower, upper, correlation="0.99", digits=20)
    with mp.workdps(70):
        expected = (mp.erfc(10 / mp.sqrt(2)) - mp.erfc(11 / mp.sqrt(2))) / 2
        close(value.probability, expected, 20)
    assert value.method == "univariate Gaussian formula"


def test_common_factor_displaced_peak_in_multiple_dimensions():
    value = equicorrelated_probability(2, 10, 11, correlation="0.99", digits=20)
    assert abs(mp.mpf(value.diagnostics["runs"][-1]["mode"])) > 9
    # Independent direct integration in the original, unscaled coordinate
    # with explicit tail breakpoints, rather than the solver's mode search.
    with mp.workdps(65):
        root_rho, root_residual = mp.sqrt(mp.mpf("0.99")), mp.mpf("0.1")
        def original(z):
            lower = (10 - root_rho * z) / root_residual
            upper = (11 - root_rho * z) / root_residual
            mass = (mp.erfc(lower / mp.sqrt(2)) - mp.erfc(upper / mp.sqrt(2))) / 2
            return mp.exp(-z * z / 2) / mp.sqrt(2 * mp.pi) * mass ** 2
        expected = mp.quad(original, [mp.ninf, 0, 5, 9, 10, 11, 12, mp.inf])
        close(value.probability, expected, 20)


def test_exact_nearly_singular_matrix_validation():
    from gaussian_orthant.transport import _matrix
    rho = "0." + "9" * 60
    matrix = _matrix([[1, rho], [rho, 1]], 25)
    assert matrix.det() > 0
    # Singular and indefinite matrices must still fail the exact fallback.
    for invalid_rho in (1, "1." + "0" * 59 + "1"):
        with pytest.raises(ValueError, match="positive definite"):
            _matrix([[1, invalid_rho], [invalid_rho, 1]], 25)


def test_exchangeable_master_limit_precedes_face_allocation(monkeypatch):
    import sympy as sp
    import gaussian_orthant.transport as transport
    def forbidden_enumeration(*args, **kwargs):
        raise AssertionError("Faces were enumerated before checking the master limit")
    monkeypatch.setattr(transport, "combinations_with_replacement", forbidden_enumeration)
    with pytest.raises(ValueError, match="276 masters"):
        BoundarySystem(sp.eye(22), (sp.S.Zero,) * 22, (sp.S.One,) * 22, max_masters=256)
