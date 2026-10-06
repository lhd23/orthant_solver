"""Gaussian integration-by-parts reduction to boundary master integrals.

The deformation is Q(t) = diag(Q) + t (Q - diag(Q)). At t=0 all
masters factor into univariate Gaussian integrals. At t=1 their transport
gives the requested correlated probability. Infinite endpoints contribute
zero boundary terms and are never represented by infinity times zero.
"""
from itertools import product, combinations_with_replacement
from functools import lru_cache
from fractions import Fraction
from math import comb

import sympy as sp
from mpmath import mp

from ._bootloops import engines
from ._independent import independent_probability
from ._kernels import resolve_backend
from ._tolerance import exact_rtol, finish_rtol, goal_digits, resolve_accuracy
from ._series import polynomial_value, rational_series
from ._symbolic import reduced_inverse, use_reduced_inverse
from .common import (ConvergenceError, complement_guard_digits,
                     interval_guard_digits, normal_interval, number,
                     precision_errors, rational, result)


def _matrix(data, digits):
    try:
        q = sp.Matrix([[rational(v) for v in row] for row in data])
    except (TypeError, ValueError) as error:
        raise ValueError("Matrix must be a finite, square real matrix") from error
    if q.rows == 0 or q.rows != q.cols or any(not x.is_finite for x in q):
        raise ValueError("Matrix must be a finite, nonempty square matrix")
    if q.is_diagonal():
        if any(value <= 0 for value in q.diagonal()):
            raise ValueError("Matrix must be positive definite")
        return q
    with mp.workdps(digits + 25):
        # Accept roundoff from floating-point matrix inversion, but reject
        # materially asymmetric inputs. Average only the accepted pairs.
        scale = max(abs(number(v)) for v in q)
        for i in range(q.rows):
            for j in range(i):
                if abs(number(q[i, j] - q[j, i])) > scale * mp.mpf("1e-14"):
                    raise ValueError("Matrix must be symmetric")
                q[i, j] = q[j, i] = (q[i, j] + q[j, i]) / 2
        numeric = mp.matrix([[number(q[i, j]) for j in range(q.cols)] for i in range(q.rows)])
        try:
            mp.cholesky(numeric)
        except ValueError as error:
            # A valid exact pivot can disappear when the matrix is rounded.
            # SymPy checks the signs of the exact rational principal minors.
            if q.is_positive_definite is not True:
                raise ValueError("Matrix must be positive definite") from error
    return q


def prepare(lower, upper, covariance, precision, mean, digits):
    if (covariance is None) == (precision is None):
        raise ValueError("Supply exactly one of covariance or precision")
    q = _matrix(precision if precision is not None else covariance, digits)
    if precision is None:
        q = sp.diag(*(1 / value for value in q.diagonal())) if q.is_diagonal() else q.inv()
    d = q.rows
    if len(lower) != d or len(upper) != d:
        raise ValueError("Bounds must have one entry per coordinate")
    mean = [0] * d if mean is None else list(mean)
    if len(mean) != d:
        raise ValueError("mean must have one entry per coordinate")
    lo, hi = [], []
    for a, b, m in zip(lower, upper, mean):
        m = rational(m)
        if not m.is_finite:
            raise ValueError("mean must be finite")
        a, b = rational(a), rational(b)
        if a is sp.nan or b is sp.nan or a is sp.oo or b is -sp.oo:
            raise ValueError("Invalid bounds")
        if a > b:
            raise ValueError("lower exceeds upper")
        lo.append(a - m)
        hi.append(b - m)
    return q, tuple(lo), tuple(hi)


def _add(out, key, value):
    out[key] = out.get(key, 0) + value


class BoundarySystem:
    """Exact rational connection, before constant diagonal normalization."""

    def __init__(self, q, lower, upper, max_masters=256, *, polynomial_backend=None):
        self.q, self.lower, self.upper = q, lower, upper
        self.d = q.rows
        self.var = "t"
        choices = [[0] + ([1] if a != -sp.oo else []) + ([2] if b != sp.oo else [])
                   for a, b in zip(lower, upper)]
        self.exchangeable = (
            len(set(lower)) == len(set(upper)) == 1
            and len(set(q.diagonal())) == 1
            and len({q[i, j] for i in range(self.d) for j in range(self.d) if i != j}) <= 1
        )
        count = 1
        for choice in choices:
            count *= len(choice)
        if self.exchangeable:
            count = comb(self.d + len(choices[0]) - 1, len(choices[0]) - 1)
        if count > max_masters:
            raise ValueError(f"Boundary system needs {count} masters (limit {max_masters}); "
                             "use a structured reduction or explicitly raise max_masters")
        self.faces = (list(combinations_with_replacement(choices[0], self.d))
                      if self.exchangeable else list(product(*choices)))
        self.index = {face: i for i, face in enumerate(self.faces)}
        def index(face):
            return self.index[tuple(sorted(face)) if self.exchangeable else face]
        self.n = len(self.faces)
        self.t = sp.Symbol("t")
        self.diagonal = sp.diag(*q.diagonal())
        self.e = q - self.diagonal
        self.qt = self.diagonal + self.t * self.e
        self.entries = {}
        self.free_sets = set()
        self.polynomial_statistics = {}
        if polynomial_backend == "flint":
            from ._polynomial import build_connection, compiled_polynomials
            module = compiled_polynomials()
            if module is not None:
                self.entries, self.free_sets, self.polynomial_statistics = build_connection(
                    q, lower, upper, self.faces, self.index, self.exchangeable, module)
                self.symbolic_strategy = "compiled polynomial connection"
                return
        accelerated = use_reduced_inverse(self.exchangeable, self.d)
        self.symbolic_strategy = "controlled elimination" if accelerated else "SymPy inverse"
        # Reuse identical reductions only while this one system is built.
        # No prepared systems or problem inputs are retained between calls.
        cancel = lru_cache(None)(sp.cancel) if accelerated else sp.cancel

        @lru_cache(None)
        def face_data(face):
            free = tuple(i for i, flag in enumerate(face) if flag == 0)
            fixed = tuple(i for i, flag in enumerate(face) if flag)
            x = sp.Matrix([lower[i] if face[i] == 1 else upper[i] for i in fixed])
            self.free_sets.add(free)
            c = self.qt.extract(free, fixed) * x if fixed else sp.zeros(len(free), 1)
            return free, fixed, x, c

        @lru_cache(None)
        def inverse(free):
            matrix = self.qt.extract(free, free)
            return reduced_inverse(matrix, cancel) if accelerated else matrix.inv()

        def children(face, j):
            for flag, sign, endpoint in [(1, -1, lower[j]), (2, 1, upper[j])]:
                if endpoint in (sp.oo, -sp.oo):
                    continue
                child = list(face)
                child[j] = flag
                yield tuple(child), sign, endpoint

        @lru_cache(None)
        def moments(face):
            free, fixed, x, c = face_data(face)
            if not free:
                return {}
            r = inverse(free)
            # This symbolic product is shared by every coordinate moment.
            rc = r * c
            answer = {}
            for k, coordinate in enumerate(free):
                row = {}
                _add(row, index(face), -rc[k])
                for j, other in enumerate(free):
                    for child, sign, _ in children(face, other):
                        _add(row, index(child), -sign * r[k, j])
                answer[coordinate] = {key: cancel(value) for key, value in row.items() if value != 0}
            return answer

        for face in self.faces:
            row = {}
            row_index = index(face)
            free, fixed, x, c = face_data(face)
            constant = (x.T * self.e.extract(fixed, fixed) * x)[0] if fixed else 0
            _add(row, row_index, -constant / 2)
            if free:
                r = inverse(free)
                er = self.e.extract(free, free) * r
                _add(row, row_index, -sp.trace(er) / 2)
                dc = self.e.extract(free, fixed) * x if fixed else sp.zeros(len(free), 1)
                v = er * c / 2 - dc
                for k, i in enumerate(free):
                    for key, coefficient in moments(face)[i].items():
                        _add(row, key, v[k] * coefficient)
                # The boundary matrix B_{j,i} integrates x_i on face j.
                for j, coordinate in enumerate(free):
                    for child, sign, endpoint in children(face, coordinate):
                        for i, other in enumerate(free):
                            factor = sign * er[i, j] / 2
                            boundary = ({index(child): endpoint} if other == coordinate
                                        else moments(child)[other])
                            for key, coefficient in boundary.items():
                                _add(row, key, factor * coefficient)
            for key, expression in row.items():
                expression = sp.cancel(expression)
                if expression == 0:
                    continue
                num, den = sp.fraction(expression)
                coeffs = lambda poly: [Fraction(v) for v in reversed(sp.Poly(poly, self.t).all_coeffs())]
                self.entries[row_index, key] = (coeffs(num), coeffs(den))

    def normalized(self, work_digits, backend="mpmath"):
        """Create a Wayfinder system for W_F = J_F / J_F(0), seeded by ones."""
        _, _, RF = engines()
        with mp.workdps(work_digits):
            coordinate_anchors = []
            for diagonal, lower, upper in zip(self.q.diagonal(), self.lower, self.upper):
                a, lo, hi = number(diagonal), number(lower), number(upper)
                root = mp.sqrt(a)
                mass = normal_interval(lo * root, hi * root, backend=backend)
                coordinate_anchors.append((mp.sqrt(2 * mp.pi / a) * mass,
                                           mp.exp(-a * lo * lo / 2) if lower.is_finite else mp.zero,
                                           mp.exp(-a * hi * hi / 2) if upper.is_finite else mp.zero))
            anchors = []
            for face in self.faces:
                anchor = mp.fprod(coordinate_anchors[i][flag] for i, flag in enumerate(face))
                if anchor <= 0:
                    raise ConvergenceError("Diagonal boundary seed lost precision; increase digits")
                anchors.append(anchor)
            entries = {}
            for (i, j), (num, den) in self.entries.items():
                entries[i, j] = (tuple(number(v) for v in num),
                                 tuple(number(v) for v in den), anchors[j] / anchors[i])
            # All possible poles are roots of principal minors of Q(t).
            # Q(t) is positive definite throughout the real path [0,1].
            poles = []
            for free in self.free_sets:
                if not free:
                    continue
                matrix = mp.matrix([[number(self.e[i, j]) / mp.sqrt(number(self.q[i, i] * self.q[j, j]))
                                     for j in free] for i in free])
                eigenvalues = mp.eigsy(matrix, eigvals_only=True)
                for eigenvalue in eigenvalues:
                    if abs(eigenvalue) > mp.power(10, -(work_digits - 10)):
                        poles.append(-1 / eigenvalue)
            owner = self

            class NormalizedConnection:
                n = owner.n
                var = "t"
                meta = {"singular_points": poles}
                real_transport = True

                def _check(self, eps, dps):
                    if eps != 0 or dps > work_digits:
                        raise ValueError("Connection precision or deformation mismatch")

                def A(self, t, eps, dps):
                    self._check(eps, dps)
                    if not mp.im(t):
                        t = mp.re(t)
                    answer = [[mp.zero] * self.n for _ in range(self.n)]
                    for (i, j), (num, den, scale) in entries.items():
                        answer[i][j] = polynomial_value(num, t) / polynomial_value(den, t) * scale
                    return answer

                def A_series_sparse(self, t, eps, dps, order):
                    self._check(eps, dps)
                    return rational_series(entries, t, order)

                def A_series(self, t, eps, dps, order):
                    # Compatibility for consumers of the original dense interface.
                    answer = [[[mp.zero] * self.n for _ in range(self.n)]
                              for _ in range(order + 1)]
                    for (i, j), coefficients in self.A_series_sparse(t, eps, dps, order).items():
                        for k, coefficient in enumerate(coefficients):
                            answer[k][i][j] = coefficient
                    return answer

            connection = NormalizedConnection()
            if backend == "flint":
                from wayfinder.acbfast import AcbFastTable
                table_entries = {key: RF([coefficient * scale for coefficient in num], list(den))
                                 for key, (num, den, scale) in entries.items()}
                connection._acb_fast = AcbFastTable(table_entries, work_digits)
                connection._acb_check_eps = connection._check
            return connection, anchors[0]


def transport_probability(lower, upper, *, covariance=None, precision=None,
                          mean=None, digits=None, rtol=None, max_masters=256, backend="auto"):
    """Evaluate a Gaussian box or orthant using the actual BootLoops engine.

    Default resource cap: at most 256 boundary masters (five fully bounded
    coordinates or eight one-sided coordinates). Independent coordinates
    and full-space events bypass master allocation. Decimal strings and
    fractions are recommended for high-precision input. backend="auto"
    selects compiled arithmetic when available; "mpmath" forces Python
    arithmetic and "flint" requires the optional python-flint package.
    A compatible compiled backend also constructs the exact connection using
    rational polynomial arithmetic; otherwise construction remains in SymPy.
    """
    digits, rtol = resolve_accuracy(digits, rtol, 25)
    backend = resolve_backend(backend)
    q, lo, hi = prepare(lower, upper, covariance, precision, mean, digits)
    if any(a == b for a, b in zip(lo, hi)):
        if rtol is not None:
            return exact_rtol(mp.mpf(0), digits, "zero-width box", {}, rtol)
        return result(mp.mpf(0), digits, "zero-width box", {})
    if all(a == -sp.oo and b == sp.oo for a, b in zip(lo, hi)):
        if rtol is not None:
            return exact_rtol(mp.mpf(1), digits, "full Gaussian space",
                              {"dimension": q.rows}, rtol)
        return result(mp.mpf(1), digits, "full Gaussian space", {"dimension": q.rows})
    if q.is_diagonal():
        return independent_probability(lo, hi, tuple(q.diagonal()), digits=digits,
                                       backend=backend, rtol=rtol)
    system = BoundarySystem(q, lo, hi, max_masters, polynomial_backend=backend)
    determinant = q.det()
    if rtol is not None:
        return _transport_rtol(system, determinant, digits, rtol, backend)
    log_guard = 0
    if all(a < 0 < b for a, b in zip(lo, hi)):
        with mp.workdps(digits + 25):
            variances = q.inv().diagonal()
            guards = [complement_guard_digits(number(a) / mp.sqrt(number(variance)),
                                               number(b) / mp.sqrt(number(variance)))
                      for a, b, variance in zip(lo, hi, variances)
                      if a != -sp.oo or b != sp.oo]
            log_guard = min(guards, default=0)
    transport, _, _ = engines()
    previous = None
    histories = []
    # Relative agreement on the root master matters: a small local vector
    # tail does not by itself bound a tiny probability relative to itself.
    for extra in (15, 35, 75, 155):
        requested = digits + extra + log_guard
        with mp.workdps(requested + 40):
            connection, anchor = system.normalized(requested + 40, backend=backend)
            values, diagnostics = transport(connection, 0, 0, 1, [1] * system.n,
                                            requested, return_diag=True,
                                            backend="acb" if backend == "flint" else "mpmath")
            normalizer = mp.sqrt(number(determinant)) / mp.power(2 * mp.pi, q.rows / 2)
            value = mp.re(values[0]) * anchor * normalizer
            imaginary = abs(mp.im(values[0]))
            histories.append({"transport_digits": requested, **diagnostics})
            if previous is not None and value > 0:
                relative, log_relative = precision_errors(value, previous, bool(log_guard))
                if (max(relative, log_relative) < mp.power(10, -digits)
                        and imaginary < mp.power(10, -requested)):
                    return result(value, digits, "boundary masters / BootLoops Wayfinder", {
                        "masters": system.n,
                        "backend": backend,
                        "exchangeable_faces_merged": system.exchangeable,
                        "symbolic_strategy": system.symbolic_strategy,
                        "polynomial_statistics": system.polynomial_statistics,
                        "connection_nonzero_entries": len(system.entries),
                        "relative_precision_agreement": mp.nstr(relative, 8),
                        "relative_log_precision_agreement": mp.nstr(log_relative, 8),
                        "runs": histories,
                        "error_status": "precision agreement and local series estimates; not an interval certificate",
                    })
            previous = value
    raise ConvergenceError("Boundary transport did not pass the relative precision agreement gate")


def _transport_rtol(system, determinant, digits, rtol, backend):
    """Check the root probability after distinct, inexpensive transports.

    A vector-norm local Taylor estimate alone cannot establish relative
    accuracy of a small root component. Change both precision and the
    step fraction/order before accepting the root's agreement. Reserve
    precision for exact narrow endpoints, and retain higher-precision
    fallbacks for rare events whose root is small relative to other masters.
    """
    transport, _, _ = engines()
    goal, previous, histories = goal_digits(rtol), None, []
    cancellation = max(interval_guard_digits(a, b)
                       for a, b in zip(system.lower, system.upper))
    order_floor, last_failure = 0, None
    for attempt, extra in enumerate((0, 1, 3, 7, 15, 31, 63, 127, 255)):
        # Wayfinder controls its own arithmetic from requested precision.
        # The interval guard must therefore reach the engine as well as the
        # seed/connection construction, so endpoint differences survive both.
        requested = goal + 1 + cancellation + extra
        work_digits = requested + 30
        initial_order = max(24, int(0.75 * requested) + 18) + (4 if attempt else 0)
        initial_order = max(initial_order, order_floor)
        ratio = 0.5 if attempt % 2 == 0 else 0.4
        with mp.workdps(work_digits):
            connection, anchor = system.normalized(work_digits, backend=backend)
            # Cheap Taylor caps are speed seeds, not reasons to reject a
            # valid Gaussian problem. Retry recognized convergence failures
            # at the same precision before moving to the next refinement.
            for order in (initial_order, max(60, 2 * initial_order),
                          max(120, 4 * initial_order)):
                try:
                    values, diagnostics = transport(
                        connection, 0, 0, 1, [1] * system.n, requested,
                        mtay=order, ratio=ratio, guard_extra=2, return_diag=True,
                        backend="acb" if backend == "flint" else "mpmath")
                except (AssertionError, RuntimeError) as error:
                    message = str(error)
                    retryable = ((isinstance(error, AssertionError)
                                 and "cannot converge even at step fraction" in message)
                                 or (isinstance(error, RuntimeError)
                                     and message.startswith("too many steps (")
                                     and " on leg " in message))
                    if not retryable:
                        raise
                    last_failure = error
                    histories.append({"transport_digits": requested,
                                      "taylor_order": order, "step_fraction": ratio,
                                      "failure": message})
                    continue
                order_floor = order
                break
            else:
                continue
            normalizer = mp.sqrt(number(determinant)) / mp.power(2 * mp.pi, system.d / 2)
            value = mp.re(values[0]) * anchor * normalizer
            histories.append({"transport_digits": requested, "taylor_order": order,
                              "step_fraction": ratio, **diagnostics})
            if previous is not None and mp.isfinite(value) and 0 < value <= 1:
                estimate = 4 * abs(value - previous) / value
                # Includes the returned root's significant-digit rounding;
                # local vector estimates are diagnostics, not root bounds.
                estimate += 4 * mp.power(10, -requested)
                estimate += abs(mp.im(values[0])) / abs(mp.re(values[0]))
                if estimate < rtol / 2:
                    return finish_rtol(value, digits, "boundary masters / BootLoops Wayfinder", {
                        "masters": system.n, "backend": backend,
                        "interval_guard_digits": cancellation,
                        "exchangeable_faces_merged": system.exchangeable,
                        "symbolic_strategy": system.symbolic_strategy,
                        "polynomial_statistics": system.polynomial_statistics,
                        "connection_nonzero_entries": len(system.entries),
                        "relative_precision_agreement": mp.nstr(abs(value - previous) / value, 8),
                        "runs": histories}, rtol, estimate)
            previous = value
    raise ConvergenceError("Boundary transport did not meet rtol") from last_failure
