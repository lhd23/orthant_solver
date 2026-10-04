# Gaussian orthant probabilities using BootLoops

Compute multivariate Gaussian orthant and box probabilities with configurable
arbitrary precision. The package uses [BootLoops](https://github.com/BootLoops-ai/bootloops)
for numerical integration and transport, and SymPy for symbolic preparation.

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

If BootLoops is already installed elsewhere, set `BOOTLOOPS_ROOT` to its
repository directory instead of cloning it again. The `test` extra installs
NumPy, SciPy, and pytest for the examples and validation tools.

For optional compiled arithmetic, install `python3 -m pip install -e '.[test,fast]'`.
This requires a Python version supported by `python-flint>=0.8`.

## Usage

```python
from gaussian_orthant import orthant_probability, gaussian_probability

# Probability that both coordinates are positive.
result = orthant_probability([[1, "0.5"], ["0.5", 1]], digits=25)
print(result.as_dict())

# Probability of a finite box.
result = gaussian_probability(
    [0, 0], [1, 1], covariance=[[1, "0.5"], ["0.5", 1]], digits=25
)
print(result.as_dict())
```

The four public functions are:

- `orthant_probability`: positive or negative orthants, with optional thresholds.
- `gaussian_probability`: boxes with arbitrary lower and upper bounds.
- `exchangeable_precision_probability`: equal finite bounds with an exchangeable precision matrix.
- `equicorrelated_probability`: equal bounds and nonnegative equal correlations.

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

Run the full suite after major changes:

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
