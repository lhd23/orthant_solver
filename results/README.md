# Status of saved results

The files `validation.json`, `validation.md`, and `performance.json` describe
runs performed before the performance changes edited on 4 October 2026.
Their probabilities, comparison thresholds, timing measurements, and source
digests are historical evidence for the earlier source.

The updated solver, tests, benchmarks, and dependency installation have not
been executed, following the user's instruction. No new passing result or
speedup measurement is claimed. The numerical data and digests in the saved
receipts have not been regenerated or changed. The readable validation report
has an added status notice identifying it as historical.

Future validation can select `--backend mpmath` or `--backend flint` and use
separate output directories. The validation program records the selected
backend, optional compiled-library version, and source digests, including the
BootLoops compiled transport module. Compiled regression tests require the
optional `python-flint` dependency.

Fresh accuracy checks and timing measurements are required before relying
on the modified implementation or claiming a performance improvement.
