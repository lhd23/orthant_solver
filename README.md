![Examples of Gaussian integration domains: orthants, finite boxes, semi-infinite boxes, and unbounded strips in two and three dimensions.](output/images/gaussian_domains.png)

# Gaussian orthant probabilities

Compute multivariate Gaussian orthant and box probabilities with configurable
arbitrary precision. The package uses [BootLoops](https://github.com/BootLoops-ai/bootloops)
for numerical integration and transport, and SymPy for symbolic preparation.
Use this code if you need high precision and runtime is not a major concern.

- Accepts a covariance or precision matrix, nonzero means, and finite or infinite bounds.
- Returns both the probability and its natural logarithm, with convergence diagnostics.
- Provides specialized routines for exchangeable precision matrices and equal-correlation models.
- Includes examples, independent validation, and comparisons with SciPy.

See the [mathematical documentation](output/pdf/gaussian_orthant_methods.pdf)
for the methods and their derivation.

## Installation

Requires Python 3.9 or later and a local BootLoops checkout. From the project directory:

```sh
git clone https://github.com/BootLoops-ai/bootloops.git bootloops
python3 -m pip install -e '.[test]'
```

The `test` extra installs NumPy, SciPy, and pytest for the examples and validation tools.

For optional compiled arithmetic, install `python3 -m pip install -e '.[test,fast]'`.
This requires a Python version supported by `python-flint>=0.8`.

## Usage

To evaluate the eight domains labelled (a)-(h) in the banner:

```python
from gaussian_orthant import gaussian_probability

# (a) Positive orthant
gaussian_probability(
    lower=[0, 0], upper=["inf", "inf"],
    covariance=[[1, 0.5], [0.5, 1]], mean=[0, 0], digits=25,
)

# (b) Shifted mixed orthant
gaussian_probability(
    lower=["0.6", "-inf"], upper=["inf", "0.8"],
    covariance=[[1, 0.5], [0.5, 1]], mean=[0, 0], digits=25,
)

# (c) Finite box
gaussian_probability(
    lower=["-0.9", "-0.7"], upper=["1.2", "1.3"],
    covariance=[[1, 0.5], [0.5, 1]], mean=[0, 0], digits=25,
)

# (d) Semi-infinite box
gaussian_probability(
    lower=[-1, "-0.6"], upper=[1, "inf"],
    covariance=[[1, 0.5], [0.5, 1]], mean=[0, 0], digits=25,
)

# (e) Unbounded strip
gaussian_probability(
    lower=["-inf", "-0.7"], upper=["inf", 1],
    covariance=[[1, 0.5], [0.5, 1]], mean=[0, 0], digits=25,
)

# (f) Positive orthant (3 dimensions)
gaussian_probability(
    lower=[0, 0, 0], upper=["inf", "inf", "inf"],
    covariance=[[1, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1]],
    mean=[0, 0, 0], digits=25,
)

# (g) Finite box (3 dimensions)
gaussian_probability(
    lower=[0, 0, 0], upper=["1.7", "1.6", "1.6"],
    covariance=[[1, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1]],
    mean=[0, 0, 0], digits=25,
)

# (h) Semi-infinite box (3 dimensions)
gaussian_probability(
    lower=[0, 0, 0], upper=["1.7", "1.6", "inf"],
    covariance=[[1, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1]],
    mean=[0, 0, 0], digits=25,
)
```
Note that for a nonzero mean include an argument, e.g. `mean=[0.2, -0.3]`.
Evaluation of higher-dimensional Gaussians and domains are also possible.

In addition to `gaussian_probability` (for boxes with arbitrary lower and upper bounds)
one can also call:

- `orthant_probability`: positive or negative orthants, with optional thresholds.
- `exchangeable_precision_probability`: equal finite bounds with an exchangeable precision matrix.
- `equicorrelated_probability`: equal bounds and nonnegative equal correlations.

For faster evaluation, specify `rtol` instead of `digits`, for example
`equicorrelated_probability(10, correlation=0.5, rtol=1e-6)`. This targets an
estimated relative probability error of one part in a million. The result
includes `rtol` and `estimated_relative_error`; these are convergence estimates,
not certified error bounds. This mode does not separately target relative
accuracy in `log_probability`; `digits` in the result controls display precision.
Without `rtol`, the existing digit mode is used.

Use decimal strings or `fractions.Fraction` when inputs need more precision
than ordinary floating-point numbers. Use `result.as_dict()` to display the
requested digits.

Examples can also be run from the command line:

```sh
python3 -m gaussian_orthant examples/orthant.json
python3 -m gaussian_orthant examples/botev_example_i.json
python3 compare_solvers.py
```

The comparison script calls all four functions and prints probabilities,
independently measured relative errors, and runtimes alongside SciPy.
Use `--digits 8 10` to compare multiple accuracy targets, or
`--rtol 1e-4 1e-6` to compare the faster tolerance mode, and
`--match-achieved` to require comparable achieved errors. A shared requested
accuracy does not guarantee that both solvers achieve the same error.

## Arithmetic and performance

Choose `backend="mpmath"` for Python arbitrary-precision arithmetic or
`backend="flint"` for compiled arithmetic through python-flint. The default,
`backend="auto"`, selects a compatible compiled backend when available and
otherwise uses mpmath. An explicit compiled request requires a compatible
python-flint installation.

Runtime depends on dimension, matrix structure, requested precision, and
backend. Specialized routines can handle much larger problems than the
general solver. Precision checks may produce substantially more accuracy
than requested, so compare measured errors as well as runtimes when assessing
performance against SciPy or other methods.

## Tests and validation

```sh
python3 -m pytest -q -rs
```

- The regression campaign covers examples from [Botev, Section 5](https://arxiv.org/html/1603.04166), exact reference cases, and SciPy comparisons.
- Independent integration and minimax tilting provide additional checks.
- Compiled-backend tests run when python-flint is available; set `ORTHANT_REQUIRE_FLINT=1` to require it.
- See the [test guide](tests/README.md) for acceptance criteria and the [test report](results/regression_tests.md) for recorded results.

## Limits

- General problems have a default limit of 256 boundary masters: up to eight one-sided or five fully bounded coordinates before symmetry reductions. Structured and independent cases can exceed these dimensions.
- Covariance and precision matrices must be positive definite.
- Convergence checks provide numerical evidence of accuracy, rather than rigorous bounds on the complete calculation. A convergence failure raises an exception.
- Use separate processes for concurrent evaluations; shared arithmetic contexts do not support concurrent threads.
