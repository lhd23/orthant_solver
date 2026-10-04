# Status of saved results

The files `validation.json`, `validation.md`, and `performance.json` describe
runs performed before the performance changes edited on 4 October 2026.
Their probabilities, comparison thresholds, timing measurements, and source
digests are historical evidence for the earlier source.

The updated solver has now passed the full test suite, including the Botev and
SciPy comparisons on both arithmetic backends. See
[regression_tests.md](regression_tests.md) for the executed checks and the
compatibility and tail-evaluation fixes they required. The numerical data and
digests in the older receipts have not been regenerated or changed. The
readable validation report retains its notice identifying it as historical.

Future validation can select `--backend mpmath` or `--backend flint` and use
separate output directories. The validation program records the selected
backend, optional compiled-library version, and source digests, including the
BootLoops compiled transport module. Compiled regression tests require the
optional `python-flint` dependency.

The fresh regression checks support their stated cases and tolerances.
Timing benchmarks remain required before claiming a performance improvement.
