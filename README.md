# Gaussian orthant probabilities using BootLoops

This package evaluates Gaussian orthants and boxes by integration-by-parts
reduction and differential-equation transport. It calls the actual
`wayfinder.transport.transport_fixed_eps` and `wayfinder.quad.quad_refine`
engines in the supplied local BootLoops checkout. It implements a new Gaussian
integral family; BootLoops does not supply a ready-made orthant solver.

**Execution status, 4 October 2026:** the performance changes below were
implemented by editing and inspecting source files only. The solver, tests,
benchmarks, and dependency installation were not run, as requested. Saved
validation and timing results describe the earlier implementation. Correctness
and speed gains for these changes have not yet been measured.

The mathematical approach follows the **integration leg** of Section 3.2 of
[BootLoops.pdf](BootLoops.pdf): reduce to master integrals, start from a known
boundary value, and transport to the desired parameters. Gaussian exponential
integrands require their own boundary identities; the rational Euler-integral
and scattering-alphabet machinery cannot be applied unchanged. This code does
not claim a new closed form, a full scattering-amplitude bootstrap, or the
paper's thirty-digit independent closed-form acceptance standard.

The [mathematical documentation](output/pdf/gaussian_orthant_methods.pdf) is a
one-column REVTeX document explaining the boundary differential system, the
structured one-dimensional reductions, the roles of BootLoops and SymPy,
and the numerical acceptance checks, with references to the underlying
literature. Its [standalone LaTeX source](output/pdf/gaussian_orthant_methods.tex)
is included. The document describes the current source; its historical
validation results do not validate the unexecuted performance changes.

## Run

From the workspace directory, enter the project with `cd orthant_solver`.
The commands below run from that directory, which also contains the local
BootLoops checkout, reference paper, tests, examples, and validation results.

The existing `python3` environment on this machine has the required packages.
For a fresh environment, from this directory:

```sh
python3 -m pip install -e '.[test]'
python3 -m gaussian_orthant examples/orthant.json
python3 -m gaussian_orthant examples/botev_example_i.json
python3 -m pytest -q
python3 validate.py --digits 25
```

## Arithmetic and performance

The Python interfaces accept `backend="auto"`, `backend="mpmath"`, or
`backend="flint"`. The default selects the compiled backend when a compatible
`python-flint` installation is available and otherwise uses `mpmath`.
An explicit `flint` request raises an import error if the dependency is absent.
The optional dependency has not been installed during this update. For future
installation and validation, from the project directory:

```sh
python3 -m pip install -e '.[test,fast]'
python3 -m gaussian_orthant examples/orthant.json --backend flint
python3 -m pytest -q
python3 validate.py --digits 25 --backend mpmath --output results/mpmath
python3 validate.py --digits 25 --backend flint --output results/flint
```

The compiled path uses [python-flint](https://python-flint.readthedocs.io/)
for arbitrary-precision real and complex arithmetic. General boundary
transport calls BootLoops' existing compiled Taylor-series engine. Gaussian
intervals and complex auxiliary-field intervals evaluate their complementary
error functions in compiled arithmetic. Conversions preserve the stored binary
mantissa and exponent, avoiding machine-float conversion. Compiled interval
kernels refine their precision and check the local ball radius before returning
a midpoint. The full solver retains its numerical error status; these local
checks do not constitute a complete interval certificate.

The compiled auxiliary-field interval kernel caches its fixed input
conversions, endpoint products, square root, and normalization factor.
The bounded cache uses exact binary input components and the actual compiled
working precision, including any precision-refinement retry. Changing the
shift at a quadrature node reuses these constants; changing an endpoint,
the Gaussian coefficient, or the compiled precision selects a separate entry.
The cached quantities retain their ball radii. This change has been inspected
but has not been executed or timed.

The compiled standard-normal interval kernel also caches its square root of
two separately at each working precision. Common-factor quadrature can reuse
this constant across nodes while retaining its ball radius and the existing
precision checks. This optimization has not been executed or timed.

The fallback transport supplies sparse connection coefficients directly,
inverts each distinct polynomial denominator once per expansion point, and
reuses the unscaled Taylor coefficients for entries with identical numerator
and denominator coefficient tuples. Each entry applies its own normalization.
This reuse lasts only for the current expansion call, so a different point,
order, or precision rebuilds the coefficients. The additional reuse has not
been executed or timed. Real paths use real scalars; complex detours retain
complex arithmetic.
Initial coordinate integrals are computed once and reused across faces;
the determinant is computed once and reused across precision runs.
During symbolic boundary-system construction, each face's inverse-matrix
product with its fixed-coordinate contribution is computed once and reused
for all coordinate moments. This setup optimization has not been executed
or timed.

Independent coordinates and single-variable cases use direct Gaussian
formulas with precision agreement checks. They avoid master construction and
quadrature. Diagonal matrices are checked exactly and inverted component by
component. The common-factor reduction replaces its preliminary integration
with a conservative analytic bound for the quadrature tolerance scale, and
reuses the logarithm of the mass at its center.

The local [BootLoops transport module](bootloops/tools/wayfinder/transport.py)
has a small adapter change to accept `A_series_sparse` and an explicit
`real_transport` flag. Existing dense connections keep their original
behavior. The Gaussian connection also supplies the dense interface for
compatibility with another BootLoops checkout, although that checkout needs
the same adapter changes to obtain the sparse and real transport improvements.
The [new regression tests](tests/test_acceleration.py) cover exact orthant
references, independent products, complex detours, tail logarithms, local
context restoration, and the optional compiled kernels. They have been
written but have not been executed.

## Python interface

The checkout defaults to `./bootloops`. If it is elsewhere, set
`BOOTLOOPS_ROOT` to the repository directory. No external scattering-integral
engine, Julia installation, or compiled reduction program is needed for the
`mpmath` backend.

```python
from gaussian_orthant import (
    orthant_probability,
    gaussian_probability,
    exchangeable_precision_probability,
    equicorrelated_probability,
)

# A positive orthant: P(X_1 >= 0, X_2 >= 0).
r = orthant_probability([[1, "0.5"], ["0.5", 1]], digits=25)
print(r.as_dict())  # probability = 1/3, numerically transported

# Nonzero means and mixed finite/infinite bounds.
r = gaussian_probability(
    ["-inf", "0.2"], ["0.7", "inf"],
    covariance=[[1, "0.3"], ["0.3", 2]],
    mean=["0.1", "-0.2"], digits=25,
)

# Section 5.1 Example I: Q = (I + 11^T)/2 and box [1/2,1]^50.
r = exchangeable_precision_probability(50, "0.5", 1, digits=25)
print(r.as_dict()["probability"])

# Section 5.3: correlation 1/2 and exact probability 1/(10001).
r = equicorrelated_probability(10000, digits=25)
print(r.as_dict()["probability"])
```

Use decimal strings or `fractions.Fraction` for exact high-precision inputs.
Floating-point inputs specify only the precision already present in those
numbers. Printing an arbitrary-precision value at the ambient precision can
hide its digits: use `result.as_dict()` or an explicit `mpmath` precision
context. Both probability and natural logarithm of probability are retained.
The library restores the caller's arithmetic precision after every call.
Its engines use shared `mpmath` and, when selected, `python-flint` contexts;
concurrent calls in separate threads are unsupported. Use separate processes
for concurrent evaluation.

Bounds are compared and centered exactly before numerical conversion. Short
intervals receive additional working precision and a local Gaussian series
to avoid subtracting nearly equal endpoint values. Independent cases also
require agreement between two precision runs. The logarithm is
computed before the final rounding of the probability, preserving small
negative logarithms when the displayed probability rounds to one.
When a probability is close to one, marginal tail bounds also increase the
integration precision, and convergence is checked for the logarithm itself.

## Gaussian master system

Write (Q=\Sigma^{-1}), center the bounds by subtracting the mean, and let

\[
Q(t)=D+tE,\qquad D=\operatorname{diag}(Q),\quad E=Q-D.
\]

The path is positive definite for (0\leq t\leq1\) because it is a convex
combination of two positive definite matrices. At the start, all coordinates
are independent. At the endpoint the precision matrix is the requested one.

For each face (F), fix a subset of coordinates to their finite lower or upper
bounds, and integrate (\exp(-x^TQ(t)x/2)) over the remaining coordinates.
Call that integral (J_F(t)). Differentiating produces Gaussian second
moments. Integration by parts reduces these to the same face masters.
Boundary contributions at infinite endpoints vanish.

For free coordinates (A), fixed coordinates (B), fixed values (b), and
(R=Q_{AA}^{-1}\), set (c=Q_{AB}b\). Let (\Delta_j\) be the upper-face
integral minus the lower-face integral. The first moments satisfy
(m=-R(\Delta+cJ_F)\). If (T_{j,i}\) is the corresponding boundary integral
of (x_i\), the second-moment matrix satisfies
(M=R(J_F I-cm^T-T)\). A fixed coordinate contributes its endpoint times a
child master; other boundary moments use the child's first-moment identity.
Substitution into the parameter derivative gives

\[
J'_F=-\tfrac12\operatorname{tr}(E_{AA}M)
     -(E_{AB}b)^T m-\tfrac12 b^TE_{BB}bJ_F.
\]

The resulting first-order connection is rational in (t\) and is assembled
exactly with SymPy. Its possible poles lie at zeros of the principal minors
of (Q(t)\). Their locations are supplied to Wayfinder, which advances local
Taylor series using rational coefficient recurrences and step refinement.
Known diagonal Gaussian integrals seed every face. Dividing each master by
its own initial value improves scaling; the final probability is
(\sqrt{\det Q}\,J_{\varnothing}(1)/(2\pi)^{d/2}\).

Permutation-equivalent faces are merged for exchangeable matrices with
identical bounds. This reduces a fully bounded exchangeable family from
(3^d\) masters to (\binom{d+2}{2}\).

## Structured reductions

For (Q=aI+b\mathbf1\mathbf1^T\), with (a>0\), (b\geq0\), and finite equal
bounds, `exchangeable_precision_probability` introduces one
Hubbard-Stratonovich auxiliary field. The identity
(e^{-bS^2/2}=(2\pi b)^{-1/2}\int_{\mathbb R}e^{-y^2/(2b)+iyS}\,dy\)
factorizes the coordinate integrations. Moving the contour to
(y+i\lambda\), where (\lambda/b=d\,\mathbb E_\lambda[X]\), centers the
phase and avoids the severe cancellation of the unshifted representation.
The real one-dimensional integral is evaluated with BootLoops quadrature.

For a cutoff (T\), the relative discarded-tail bound is at most
(\operatorname{erfc}(T/\sqrt{2b})\exp(bd(u-l)^2/8)\). This follows from
the triangle inequality for characteristic functions, the finite-interval
variance bound, and Jensen's inequality applied to the positive tilted
expectation. The bound covers the omitted tail; it does not certify all
quadrature and arithmetic errors.

For nonnegative equal correlations, `equicorrelated_probability` conditions
on a single common Gaussian factor and evaluates its one-dimensional
integral. It supports equal finite or infinite bounds and a common nonzero
mean and standard deviation. The function numerically integrates the
Section 5.3 example; it does not return the known reference formula directly.
Before quadrature, it locates the integrand's mode and scales the integration
coordinate and amplitude there. This keeps rare tail peaks visible to the
refined quadrature. If the log integrand is (h), its curvature is bounded
below by (-K), where (K=1+d\rho/(1-\rho)). For a coordinate width (w),
the centered integral is at least (\sqrt{2\pi/K}/w), including when the
center has a small residual slope. Half this lower bound supplies a
conservative scale for quadrature tolerance without preliminary integration.

## Validation and limits

[validate.py](validate.py) recreates three families from
[Botev's Section 5](https://arxiv.org/html/1603.04166):

- Example I at dimensions 2, 3, 5, 10, 20 and 50. The candidate is checked
  against positive convolution of the original integrand, a separate
  minimax tilting implementation, and boundary transport at dimensions
  2, 3 and 5.
- Example II at dimensions 2 and 3, using the stated **precision** matrix,
  including the cutoff ( |i-j|\leq d/2\). Direct positive product
  quadrature and minimax tilting independently check the values.
- Section 5.3 at dimensions 10, 100, 1000 and 10000, compared with the exact
  probability (1/(d+1)\).

The separate minimax implementation in [validation_oracles.py](validation_oracles.py)
solves the paper's saddle equations and uses twelve independently scrambled
Sobol sequences, each with 4096 points. The paper uses randomized Richtmyer
sequences. Neither published stochastic estimates nor their error bars are
treated as exact references. Every comparison, seed, refinement history,
engine diagnostic and source-file digest is recorded in
[results/validation.json](results/validation.json); the readable report is
[results/validation.md](results/validation.md).
These saved results predate the performance changes and do not validate the
current source. Their numerical values and recorded source digests have been
preserved; [the results note](results/README.md) explains their status.

The dimension-10 Example I value is about (8.56248967736\times10^{-15}\),
whereas the paper prints (8.556\times10^{-15}\) with a relative standard
error of 0.01 percent. Independent positive convolution, boundary transport,
and the newly implemented minimax estimator agree with the candidate.
The difference from the printed estimate is retained as a discrepancy,
without asserting its cause.

General problems require (2^d\) masters for one finite endpoint per
coordinate, or (3^d\) for fully bounded boxes, before symmetry reduction.
The default limit is 256 masters: eight one-sided coordinates or five fully
bounded coordinates. Exact connection construction and arbitrary-precision
transport can become expensive before that limit. Raising `max_masters` is
an explicit resource choice. Full-space and diagonal cases return through
direct formulas before master allocation and are not subject to the master
limit. Correlated problems retain the exponential resource limit.
The master count is checked before face enumeration. If rounded matrix
validation loses a positive pivot, exact rational arithmetic checks positive
definiteness; subsequent transport must still pass its convergence checks.

Example II's large-dimensional rows and the random 100-dimensional matrices
in Examples III and IV are not reproduced. The latter are not supplied as
reproducible matrices in the paper. No general speed advantage over minimax
tilting is established.

The requested digits are checked by rerunning with higher arithmetic
precision and stricter quadrature tolerances. Diagnostics report measured
agreement and local Taylor estimates. These are numerical validation,
**not a complete rigorous interval enclosure**. Failures to converge raise
an exception instead of returning an unchecked value. A rigorous arbitrary
matrix solver would additionally need interval arithmetic through the
entire reduction, seeds, transport and normalization.

The earlier implementation has independently checked Gaussian examples.
The current performance changes preserve the convergence gates in source,
but require fresh execution to establish correctness and any speed advantage.
