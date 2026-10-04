"""Regression coverage for the optional kernels and transport shortcuts.

Written with the performance changes; not executed during implementation.
Compiled cases skip when the optional python-flint dependency is absent.
"""
import pytest
from mpmath import mp

from gaussian_orthant import (equicorrelated_probability,
                             exchangeable_precision_probability,
                             gaussian_probability, orthant_probability)
from gaussian_orthant import _kernels
from gaussian_orthant._bootloops import engines
from gaussian_orthant.transport import BoundarySystem, prepare


def assert_relative(actual, expected, digits=20):
    with mp.workdps(digits + 25):
        assert abs(actual / expected - 1) < mp.power(10, -digits)


def forbidden(*args, **kwargs):
    raise AssertionError("An unnecessary computation was invoked")


def test_missing_optional_dependency_falls_back(monkeypatch):
    monkeypatch.setattr(_kernels, "_flint", lambda: None)
    value = orthant_probability([[1]], backend="auto", digits=20)
    assert value.probability == mp.mpf("0.5")
    assert value.diagnostics["backend"] == "mpmath"
    with pytest.raises(ImportError, match="python-flint"):
        orthant_probability([[1]], backend="flint")
    with pytest.raises(ValueError, match="backend"):
        equicorrelated_probability(2, backend="unknown")


def test_independent_coordinates_bypass_master_allocation(monkeypatch):
    import gaussian_orthant.transport as transport
    monkeypatch.setattr(transport, "BoundarySystem", forbidden)
    monkeypatch.setattr(transport, "engines", forbidden)
    monkeypatch.setattr(mp, "cholesky", forbidden)
    dimension = 12
    covariance = [[(i + 1) ** 2 if i == j else 0
                   for j in range(dimension)] for i in range(dimension)]
    value = gaussian_probability([0] * dimension, list(range(1, dimension + 1)),
                                 covariance=covariance, max_masters=1,
                                 digits=22, backend="mpmath")
    with mp.workdps(60):
        expected = (mp.erf(1 / mp.sqrt(2)) / 2) ** dimension
        assert_relative(value.probability, expected)
    assert value.method == "independent Gaussian product"


@pytest.mark.parametrize("diagonal", [0, -1])
def test_diagonal_shortcut_rejects_nonpositive_variance(diagonal):
    with pytest.raises(ValueError, match="positive definite"):
        orthant_probability([[1, 0], [0, diagonal]], backend="mpmath")


def test_univariate_coupling_uses_total_precision(monkeypatch):
    import gaussian_orthant.structured as structured
    monkeypatch.setattr(structured, "engines", forbidden)
    value = exchangeable_precision_probability(1, -1, 1, diagonal=2,
                                              coupling=3, digits=22,
                                              backend="mpmath")
    with mp.workdps(60):
        assert_relative(value.probability, mp.erf(mp.sqrt(mp.mpf("2.5"))))


@pytest.mark.parametrize("detour", [False, True])
def test_sparse_transport_against_arcsine_without_dense_series(monkeypatch, detour):
    q, lo, hi = prepare([0, 0], ["inf", "inf"],
                        [[1, "0.3"], ["0.3", 1]], None, None, 22)
    system = BoundarySystem(q, lo, hi)
    transport, _, _ = engines()
    with mp.workdps(80):
        connection, anchor = system.normalized(80)
        monkeypatch.setattr(connection, "A_series", forbidden)
        options = {"path": [mp.mpc("0.4", "0.1")]} if detour else {}
        values = transport(connection, 0, 0, 1, [1] * system.n, 30,
                           backend="mpmath", **options)
        determinant = mp.mpf(int(q.det().p)) / int(q.det().q)
        probability = values[0] * anchor * mp.sqrt(determinant) / (2 * mp.pi)
        expected = mp.mpf(1) / 4 + mp.asin(mp.mpf("0.3")) / (2 * mp.pi)
        assert_relative(probability, expected, 27)


def test_short_sparse_series_is_rejected():
    engines()
    from wayfinder.transport import _local_a_series

    class Truncated:
        n = 1
        real_transport = True

        def A_series_sparse(self, *args):
            return {(0, 0): [mp.mpf(1)]}

    with pytest.raises(ValueError, match="fewer than M\\+1"):
        _local_a_series(Truncated(), mp.zero, mp.zero, mp.mpf(1), 10, 50, False)


def test_common_factor_does_not_integrate_twice(monkeypatch):
    monkeypatch.setattr(mp, "quad", forbidden)
    value = equicorrelated_probability(10, digits=22, backend="mpmath")
    with mp.workdps(60):
        assert_relative(value.probability, mp.mpf(1) / 11)
    assert all(run["scale_source"] == "analytic curvature lower bound"
               for run in value.diagnostics["runs"])


@pytest.mark.parametrize("center", ["0", "1.7"])
def test_curvature_scale_bound_with_inexact_center(center):
    from gaussian_orthant.structured import _factor_scale_bound
    # rho=1/2, d=3, lower=0: the complete probability is exactly 1/4.
    # The proposed lower bound must hold away from the mode as well.
    with mp.workdps(60):
        center, width = mp.mpf(center), mp.mpf("0.7")
        common = independent = mp.sqrt(mp.mpf("0.5"))
        mass = mp.erfc(-center / mp.sqrt(2)) / 2
        peak = mp.exp(-center ** 2 / 2) / mp.sqrt(2 * mp.pi) * mass ** 3
        actual_scaled_integral = 1 / (4 * width * peak)
        bound = _factor_scale_bound(3, common, independent, width)
        assert 0 < bound < actual_scaled_integral


@pytest.fixture
def compiled_backend():
    module = pytest.importorskip("flint")
    _kernels.resolve_backend("flint")
    return module


def test_compiled_kernels_against_direct_integrals(compiled_backend):
    with mp.workdps(85):
        a, s = mp.mpf("1.7"), mp.mpc("1.2", "2.3")
        lower, upper = mp.mpf("-0.3"), mp.mpf("0.7")
        expected = mp.quad(lambda x: mp.exp(-a * x * x / 2 - s * x), [lower, upper])
        tail = (mp.erfc(10 / mp.sqrt(2)) - mp.erfc(11 / mp.sqrt(2))) / 2
    old_context = compiled_backend.ctx.prec, compiled_backend.ctx.cap
    with mp.workdps(55):
        actual = _kernels.exponential_interval(a, s, lower, upper)
        actual_tail = _kernels.normal_interval(mp.mpf(10), mp.mpf(11))
    assert_relative(actual, expected, 50)
    assert_relative(actual_tail, tail, 50)
    assert (compiled_backend.ctx.prec, compiled_backend.ctx.cap) == old_context


@pytest.mark.parametrize("changed", ["a", "lower", "upper"])
def test_compiled_constants_distinguish_inputs_and_precision(compiled_backend, changed):
    # Differences beyond the initial working precision must not collide in
    # the cache, and a later high-precision call must rebuild its constants.
    _kernels._exponential_constants.cache_clear()
    old_context = compiled_backend.ctx.prec, compiled_backend.ctx.cap
    try:
        with mp.workdps(110):
            a, s = mp.mpf("1.7"), mp.mpc("1.2", "2.3")
            lower, upper = mp.mpf("-0.3"), mp.mpf("0.7")
            base = {"a": a, "lower": lower, "upper": upper}
            perturbed = dict(base)
            perturbed[changed] += mp.mpf("1e-45")

            def reference(parameters):
                return mp.quad(lambda x: mp.exp(-parameters["a"] * x * x / 2 - s * x),
                               [parameters["lower"], parameters["upper"]])

            expected, expected_perturbed = reference(base), reference(perturbed)
        with mp.workdps(25):
            _kernels.exponential_interval(a, s, lower, upper)
        with mp.workdps(70):
            actual = _kernels.exponential_interval(a, s, lower, upper)
            different = _kernels.exponential_interval(perturbed["a"], s,
                                                     perturbed["lower"], perturbed["upper"])
        assert_relative(actual, expected, 65)
        assert_relative(different, expected_perturbed, 65)
        assert (compiled_backend.ctx.prec, compiled_backend.ctx.cap) == old_context
    finally:
        _kernels._exponential_constants.cache_clear()


def test_compiled_constants_reused_across_shifts(compiled_backend):
    _kernels._exponential_constants.cache_clear()
    old_context = compiled_backend.ctx.prec, compiled_backend.ctx.cap
    try:
        with mp.workdps(110):
            a, lower, upper = mp.mpf("1.7"), mp.mpf("-0.3"), mp.mpf("0.7")
            shifts = [mp.mpc("1.2", "2.3"), mp.mpc("1.2", "-2.3")]
            expected = [mp.quad(lambda x: mp.exp(-a * x * x / 2 - shift * x),
                                [lower, upper]) for shift in shifts]
        with mp.workdps(70):
            actual = [_kernels.exponential_interval(a, shift, lower, upper) for shift in shifts]
        for value, reference in zip(actual, expected):
            assert_relative(value, reference, 65)
        assert _kernels._exponential_constants.cache_info().hits > 0
        assert (compiled_backend.ctx.prec, compiled_backend.ctx.cap) == old_context
    finally:
        _kernels._exponential_constants.cache_clear()


def test_compiled_context_restored_on_failure(compiled_backend):
    old_context = compiled_backend.ctx.prec, compiled_backend.ctx.cap

    def fail(module):
        module.ctx.cap = 17
        raise ValueError("deliberate failure")

    with pytest.raises(ValueError, match="deliberate failure"):
        _kernels._evaluate(fail)
    assert (compiled_backend.ctx.prec, compiled_backend.ctx.cap) == old_context


def test_compiled_transport_against_arcsine(compiled_backend):
    value = orthant_probability([[1, "-0.3"], ["-0.3", 1]],
                                digits=22, backend="flint")
    with mp.workdps(60):
        expected = mp.mpf(1) / 4 - mp.asin(mp.mpf("0.3")) / (2 * mp.pi)
        assert_relative(value.probability, expected)
    assert value.diagnostics["backend"] == "flint"


def test_compiled_structured_reductions(compiled_backend):
    compiled = exchangeable_precision_probability(2, "0.5", 1, digits=22,
                                                  backend="flint")
    reference = gaussian_probability(["0.5"] * 2, [1] * 2,
                                     precision=[[1, "0.5"], ["0.5", 1]],
                                     digits=22, backend="mpmath")
    assert_relative(compiled.probability, reference.probability)
    common = equicorrelated_probability(10, digits=22, backend="flint")
    with mp.workdps(60):
        assert_relative(common.probability, mp.mpf(1) / 11)


def test_compiled_near_one_logarithm_and_narrow_interval(compiled_backend):
    value = equicorrelated_probability(2, -13, "inf", correlation=0,
                                      digits=22, backend="flint")
    with mp.workdps(110):
        tail = mp.erfc(13 / mp.sqrt(2)) / 2
        assert_relative(value.log_probability, 2 * mp.log1p(-tail))
    upper = "1." + "0" * 59 + "1"
    # A narrow box exercises the retained centered interval series.
    box = gaussian_probability([1], [upper], covariance=[[1]],
                               digits=22, backend="flint")
    with mp.workdps(110):
        expected = mp.exp(-mp.mpf("0.5")) / mp.sqrt(2 * mp.pi) * mp.mpf("1e-60")
        assert_relative(box.probability, expected)
