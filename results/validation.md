# Gaussian orthant validation

**Historical report:** these results predate the performance changes edited
on 4 October 2026. The modified solver and tests have not been executed.
The passing checks below apply to the earlier source recorded in the receipt;
they do not validate the current implementation. Numerical results are
preserved. See [the results note](README.md).

Source: [Botev, Section 5](https://arxiv.org/html/1603.04166).

Requested precision: 25 decimal digits. Independent checks: all passed.

| Paper family | Dimension | Computed probability | Published estimate or exact value | Independent check |
|---|---:|---:|---:|---|
| Section 5.1 Example I | 2 | 0.01489631388606450747510203 | 0.01489 | passed |
| Section 5.1 Example I | 3 | 0.001077321645861562743078148 | 0.001077 | passed |
| Section 5.1 Example I | 5 | 0.000002451691596984394242873498 | 2.451e-6 | passed |
| Section 5.1 Example I | 10 | 8.56248967736346261889415e-15 | 8.556e-15 | passed |
| Section 5.1 Example I | 20 | 1.779997766416905936339532e-38 | 1.7796e-38 | passed |
| Section 5.1 Example I | 50 | 2.137302826361027523140585e-153 | 2.1364e-153 | passed |
| Section 5.1 Example II | 2 | 0.09121878147964991547559589 | 0.09121 | passed |
| Section 5.1 Example II | 3 | 0.02307072735939417709903235 | 0.02307 | passed |
| Section 5.3 exact orthant | 10 | 0.09090909090909090909090909 | 1/(d+1) | passed |
| Section 5.3 exact orthant | 100 | 0.00990099009900990099009901 | 1/(d+1) | passed |
| Section 5.3 exact orthant | 1000 | 0.000999000999000999000999001 | 1/(d+1) | passed |
| Section 5.3 exact orthant | 10000 | 0.0000999900009999000099990001 | 1/(d+1) | passed |

The published estimates are stochastic, rounded values. The table-comparison gate allows five printed standard errors and one unit of the last printed digit; it is a discrepancy flag, not a proof that a published estimate is wrong.

For Section 5.1 Example I at dimension 10, the candidate differs from the printed estimate by 0.07584943155 percent (7.584943155 stated standard errors). The independently implemented minimax estimator and positive original-integrand convolution agree with the candidate. The discrepancy is retained in the receipt.

Example I was checked by a separate positive convolution of the original box integrand, a newly implemented minimax tilting estimator, and boundary transport at dimensions 2, 3 and 5. Example II was checked by direct product quadrature and minimax tilting. The Section 5.3 numerical one-factor integral was checked against the exact formula, including dimension 10000.

The independent minimax implementation uses twelve independently scrambled Sobol sequences, with 4096 points in each sequence and a fixed recorded seed. The paper uses randomized Richtmyer sequences. Its errors and our estimator errors are statistical; high-precision candidate digits are supported by refinement and precision agreement, with an analytic discarded-tail bound for the auxiliary-field integral. No complete interval certificate is claimed.

Example II's large-dimensional rows and the random matrices in Examples III and IV were not reproduced. The general boundary basis grows exponentially; structured reductions enable the large-dimensional results above. No general speed advantage over minimax tilting is established.

The implementation passes the independent checks on these three reproducible paper families; the receipt preserves the printed-table discrepancy and the limits of the numerical evidence.
