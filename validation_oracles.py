"""Independent numerical routes used only by the validation campaign.

These routines do not construct or use the boundary connection or Fourier
integrand. They require the optional NumPy and SciPy dependencies.
"""
import numpy as np
from scipy.optimize import root
from scipy.special import log_ndtr, logsumexp
from scipy.stats import qmc, truncnorm


def positive_convolution_reference(dimension, *, first_level=4, last_level=10):
    """Example I: positive product-trapezoid integration, then Richardson.

    On an equally spaced grid, sums of coordinates lie on an equally spaced
    grid too. Repeated positive polynomial convolution integrates the full
    original box integrand without visiting all grid tuples. No contour
    shift, differential equation, or reference probability enters this route.
    Richardson eliminates successive even powers of the mesh width.
    """
    table, history = [], []
    d = dimension
    normalizer = np.sqrt((0.5 ** d) * (d + 1)) / (2 * np.pi) ** (d / 2)
    for level in range(first_level, last_level + 1):
        n = 2 ** level
        nodes = np.linspace(0.5, 1, n + 1)
        weights = np.exp(-nodes * nodes / 4) * (0.5 / n)
        weights[0] *= 0.5
        weights[-1] *= 0.5
        distribution = np.array([1.0])
        for _ in range(d):
            distribution = np.convolve(distribution, weights)
        sums = 0.5 * d + np.arange(d * n + 1) * (0.5 / n)
        row = [normalizer * np.dot(distribution, np.exp(-sums * sums / 4))]
        for j in range(1, len(table) + 1):
            row.append(row[j - 1] + (row[j - 1] - table[-1][j - 1]) / (4 ** j - 1))
        table.append(row)
        history.append({"intervals_per_coordinate": n, "probability": float(row[-1])})
    return float(table[-1][-1]), history


def _log_interval(lower, upper):
    answer = np.empty_like(lower)
    positive = lower >= 0
    other = ~positive
    answer[positive] = log_ndtr(-lower[positive]) + np.log(-np.expm1(
        log_ndtr(-upper[positive]) - log_ndtr(-lower[positive])))
    answer[other] = log_ndtr(upper[other]) + np.log(-np.expm1(
        log_ndtr(lower[other]) - log_ndtr(upper[other])))
    return answer


def minimax_reference(lower, upper, precision, *, power=12, replications=12, seed=160304166):
    """Botev's minimax exponential tilting with scrambled Sobol replications.

    Derived from Sections 2 and 3 of the paper; this is an independent
    implementation, not a wrapper around the candidate probability solver.
    No variable permutation is used. Error estimates are statistical.
    """
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    covariance = np.linalg.inv(np.asarray(precision, float))
    cholesky = np.linalg.cholesky(covariance)
    diagonal = np.diag(cholesky)
    c = cholesky / diagonal[:, None] - np.eye(len(lower))
    lo, hi = lower / diagonal, upper / diagonal
    d = len(lo)

    def saddle(variables, jacobian=False):
        x, mu = np.r_[variables[:d - 1], 0], np.r_[variables[d - 1:], 0]
        a, b = lo - c @ x - mu, hi - c @ x - mu
        mass = _log_interval(a, b)
        at_a = np.exp(-a * a / 2 - np.log(2 * np.pi) / 2 - mass)
        at_b = np.exp(-b * b / 2 - np.log(2 * np.pi) / 2 - mass)
        p = at_a - at_b
        if not jacobian:
            return np.r_[(-mu + c.T @ p)[:d - 1], (mu - x + p)[:d - 1]]
        a_pdf, b_pdf = np.zeros(d), np.zeros(d)
        np.multiply(a, at_a, out=a_pdf, where=np.isfinite(a))
        np.multiply(b, at_b, out=b_pdf, where=np.isfinite(b))
        derivative = a_pdf - b_pdf - p * p
        xx = c.T @ (derivative[:, None] * c)
        xm = c.T * derivative[None, :] - np.eye(d)
        mm = np.diag(1 + derivative)
        return np.block([[xx[:d - 1, :d - 1], xm[:d - 1, :d - 1]],
                         [xm.T[:d - 1, :d - 1], mm[:d - 1, :d - 1]]])

    solution = root(saddle, np.zeros(2 * (d - 1)), jac=lambda x: saddle(x, True), tol=1e-10)
    residual = np.max(np.abs(saddle(solution.x))) if d > 1 else 0
    if residual > 1e-7:
        raise RuntimeError(f"Independent minimax saddle failed: residual {residual:g}")
    mu = np.r_[solution.x[d - 1:], 0]
    logs = []
    for k in range(replications):
        points = qmc.Sobol(d=max(1, d - 1), scramble=True, seed=seed + k).random_base2(power)
        z = np.zeros((len(points), d))
        weights = np.zeros(len(points))
        for i in range(d):
            shift = z[:, :i] @ c[i, :i]
            a, b = lo[i] - shift - mu[i], hi[i] - shift - mu[i]
            logmass = _log_interval(a, b)
            if i < d - 1:
                z[:, i] = mu[i] + truncnorm.ppf(points[:, i], a, b)
                weights += logmass + mu[i] * mu[i] / 2 - mu[i] * z[:, i]
            else:
                weights += logmass
        logs.append(logsumexp(weights) - np.log(len(weights)))
    log_average = logsumexp(logs) - np.log(replications)
    ratios = np.exp(np.asarray(logs) - log_average)
    relative_error = float(np.std(ratios, ddof=1) / np.sqrt(replications))
    return {
        "probability": float(np.exp(log_average)),
        "log_probability": float(log_average),
        "relative_standard_error": relative_error,
        "saddle_residual": float(residual),
        "replications": replications,
        "points_per_replication": 2 ** power,
        "seed": seed,
        "sequence": "independently scrambled Sobol; paper uses randomized Richtmyer",
    }

