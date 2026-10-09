![Examples of Gaussian integration domains: orthants, finite boxes, semi-infinite boxes, and unbounded strips in two and three dimensions.](output/images/gaussian_domains.png)

# Gaussian orthant probabilities

Compute multivariate Gaussian orthant and box probabilities with configurable
precision. The package makes heavy use of [BootLoops](https://github.com/BootLoops-ai/bootloops)
for numerical integration and transport, and SymPy for symbolic preparation.
Use this code if your number one concern is having a high-precision estimate, rather
than having something that is fast and reasonably precise.
This code yields estimates to arbitrary precision.

See the [mathematical documentation](output/pdf/gaussian_orthant_methods.pdf)
for the methods and their derivation.

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

- Accepts a covariance or precision matrix (inverse covariance), nonzero means, e.g.  e.g. `mean=[0.2, -0.3]`
- Returns both the probability and its natural logarithm, with convergence diagnostics.
- Provides specialized routines for exchangeable precision matrices and equal-correlation models.
- Evaluation of higher-dimensional Gaussians and domains also possible.

In addition to `gaussian_probability` (for boxes with arbitrary lower and upper bounds)
one can also call:

- `orthant_probability`: positive or negative orthants, with optional thresholds.
- `exchangeable_precision_probability`: equal finite bounds with an exchangeable precision matrix.
- `equicorrelated_probability`: equal bounds and nonnegative equal correlations.

For faster evaluation, specify `rtol` instead of `digits`, for example
`equicorrelated_probability(10, correlation=0.5, rtol=1e-6)`. The result
includes `rtol` and `estimated_relative_error`.
(Note: use decimal strings or `fractions.Fraction`, as in the code above, when inputs
need more precision
than ordinary floating-point numbers. Use `result.as_dict()` to display the
requested digits.)


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

<img src="output/images/flopscope_solver_error_comparison_1x3.png"
     alt="Floating-point operation count versus relative error for three-, five-, and ten-dimensional orthant integrals"
     width="100%">


## Installation

Requires Python 3.9 or later and a local BootLoops checkout. From the project directory:

```sh
git clone https://github.com/BootLoops-ai/bootloops.git
python3 -m pip install -e '.[test]'
```

The `test` extra installs NumPy, SciPy, and pytest for the examples and validation tools.

For optional compiled arithmetic, install `python3 -m pip install -e '.[test,fast]'`.
This requires a Python version supported by `python-flint>=0.8`.

## Arithmetic and performance

Choose `backend="mpmath"` for Python arbitrary-precision arithmetic or
`backend="flint"` for compiled numerical and exact polynomial arithmetic
through python-flint. The default,
`backend="auto"`, selects a compatible compiled backend when available and
otherwise uses mpmath. An explicit compiled request requires a compatible
python-flint installation.

The compiled backend builds the general solver's boundary equations directly
from rational polynomial coefficients, sharing factors within each construction.
Runtime depends on dimension, matrix structure, requested precision, and
backend. Specialized routines can handle much larger problems than the
general solver. Precision checks may produce substantially more accuracy
than requested, so compare measured errors as well as runtimes when assessing
performance against SciPy or other methods.

## Limits

- General problems have a default limit of 256 boundary masters: up to eight one-sided or five fully bounded coordinates before symmetry reductions. Structured and independent cases can exceed these dimensions.
- Covariance and precision matrices must be positive definite.
- Convergence checks provide numerical evidence of accuracy, rather than rigorous bounds on the complete calculation. A convergence failure raises an exception.
- Use separate processes for concurrent evaluations; shared arithmetic contexts do not support concurrent threads.
