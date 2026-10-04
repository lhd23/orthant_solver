# Plan for evaluating one activation-pattern contribution to the final-layer mean

Revised 4 October 2026. This plan concerns numerical integration for a supplied activation pattern. It does not include activation-pattern enumeration or summation. No solver, test, benchmark, or attached LaTeX document was executed or compiled during this revision.

## 1. Scope and numerical target

The input is $X\sim\mathcal N(0,I_N)$. The network has width $N$, depth $L$, zero biases, and fixed He-initialized weights $W_1,\ldots,W_L$, each of size $N\times N$. Each layer applies a linear map followed by the rectified linear activation $\phi(z)=\max(0,z)$. The expectation is over inputs with these weights held fixed.

The target is the first moment of a selected final-layer postactivation, $x_L^i$. The numerical solver evaluates one term that contributes to that mean. It is not a probability-only solver and does not divide its answer by the probability of the pattern.

The supplied derivation, [euler_moments.tex](/Users/lhd/Documents/Codex/2026-10-02/https-bootloops-ai-bootloops-pdf-https/outputs/euler_moments.tex), fixes the notation and normalization. The Euler formulas below assume $N\geq2$; width one has a separate analytic case. For a supplied pattern $s$, chart sign $\varepsilon\in\{-1,+1\}$, and final neuron $i$, the principal task is

$$
I_{s,\varepsilon,i}
=\int_{P_{s,\varepsilon}^{+}}
\frac{q_{s,\varepsilon,i}(t)}
{(1+t\cdot t)^{(N+1)/2}}\,d^{N-1}t.
$$

The corresponding unnormalized Gaussian first-moment contribution is $M_{s,\varepsilon,i}=c_N I_{s,\varepsilon,i}$, where $c_N=\Gamma((N+1)/2)/(\sqrt2\,\pi^{N/2})$.

The complete mean obeys $\mathbb E[x_L^i]=\sum_s\sum_{\varepsilon=\pm1}M_{s,\varepsilon,i}$. That identity describes how an external caller uses the results; carrying out the sum is outside this solver's responsibility.

### Responsibility boundary

The caller chooses the activation pattern, neuron, chart, and tolerances. The solver constructs or accepts that single domain, evaluates its integral, and reports its error evidence and resource usage.

The solver does not discover other patterns, traverse an activation tree, estimate the total mass of omitted patterns, return the complete network mean, or decide how to distribute error across a sum over patterns. A batch interface may evaluate a caller-supplied list independently, without generating extra patterns.

Subdivision of one domain into numerical integration pieces is permitted. Adding their contributions is part of evaluating the same supplied integral, not summation over activation patterns.

Second moments, Jacobian averages, and general moment orders are outside the initial interface. Probability integrals may appear internally as auxiliary quantities, but the returned target remains the first-moment contribution above.

## 2. Build exactly the domain in the attachment

### 2.1 Patterns stop before the final layer

A pattern is $s=(s_1,\ldots,s_{L-1})$, with one binary mask for each preceding layer. Active means strictly positive preactivation; inactive means nonpositive preactivation. No complete final-layer mask is needed. Only the chosen final neuron's positive-output inequality is added.

Put $D_{k,s}=\operatorname{diag}(s_k)$ and build $A_{1,s}=W_1$, then $A_{k+1,s}=W_{k+1}D_{k,s}A_{k,s}$. The chosen final preactivation is $a_{s,i}^{\mathsf T}X$, where $a_{s,i}^{\mathsf T}$ is row $i$ of $A_{L,s}$.

Stack the rows of $(2D_{k,s}-I_N)A_{k,s}$ for $k=1,\ldots,L-1$ into $H_s$. The pattern cell has the closed cone representation $C_s=\{x:H_sx\geq0\}$, after handling the strict active constraints and degenerate rows correctly. For $L=1$, the sole pattern is empty and $C_s=\mathbb R^N$.

A requested pattern with an identically zero active preactivation is impossible. An identically zero inactive preactivation imposes no constraint. Check these cases before replacing active inequalities by their closures. Nonzero constraint hyperplanes have zero input Gaussian measure. Lower-dimensional cells contribute zero.

### 2.2 The two coordinate charts

Use $r=|X_N|>0$, $\varepsilon=\operatorname{sign}(X_N)$, and $y_\varepsilon(t)=(t_1,\ldots,t_{N-1},\varepsilon)$. Then $X=r y_\varepsilon(t)$.

Here $r$ is the absolute value of one input coordinate. It is not the Euclidean Gaussian radius. Its Jacobian is $r^{N-1}$, and the degree-one observable supplies another factor of $r$. Integrating that scale gives the stated exponent $(N+1)/2$ and prefactor $c_N$.

The slice domain is $P_{s,\varepsilon}=\{t:H_s y_\varepsilon(t)\geq0\}$. The affine numerator is $q_{s,\varepsilon,i}(t)=a_{s,i}^{\mathsf T}y_\varepsilon(t)=\ell^{\mathsf T}t+c$, with $\ell=(a_{s,i})_{1:N-1}$ and $c=\varepsilon(a_{s,i})_N$.

For postactivation, clip the slice once by $q_{s,\varepsilon,i}(t)\geq0$ to obtain $P_{s,\varepsilon}^{+}$. The integrand is then nonnegative and smooth in the domain interior. Using the positive part on the unclipped slice is mathematically equivalent but introduces a kink that can hinder quadrature.

Both chart signs are needed for the complete contribution of a pattern. The primary call handles one sign; a convenience call may return the two results separately. Returning their sum should be an explicitly requested same-pattern operation, never an implicit sum over different patterns.

### 2.3 Equivalent clipped cone

Define $K_{s,\varepsilon,i}=C_s\cap\{x:a_{s,i}^{\mathsf T}x\geq0\}\cap\{x:\varepsilon x_N\geq0\}$. Then $M_{s,\varepsilon,i}=\mathbb E[(a_{s,i}^{\mathsf T}X)\mathbf1_{K_{s,\varepsilon,i}}(X)]$.

This equivalent representation permits Gaussian boundary transport, angular integration, or simplex integration to evaluate the exact Euler integral. An implementation must restore the original $c_N$ when reporting $I_{s,\varepsilon,i}$.

The covariance of the stacked forms $H_sX$ is generally singular at depth, because there can be more constraints than input dimensions. Work with the cone in input space. Do not add diagonal noise and treat the result as the same integral.

## 3. Numerical interface and result contract

Provide two entry points:

- **evaluate_pattern_moment:** accept weights, a supplied pattern through layer $L-1$, a selected final neuron, a chart sign, tolerances, precision settings, and resource limits.
- **evaluate_euler_integral:** accept $N$, affine inequalities defining a slice, the affine numerator, and an indication of whether the positive-output clipping is already included. This lower-level interface allows an external geometry or pattern-summation program to supply the integral directly.

The second interface evaluates the stated mathematical integral without inventing network metadata. The first additionally checks network mask conventions and constructs its geometry.

Return at least:

| Field | Meaning |
|---|---|
| integral | $I_{s,\varepsilon,i}$, without $c_N$ |
| contribution | $M_{s,\varepsilon,i}=c_N I_{s,\varepsilon,i}$ |
| integral_error | Absolute integration error estimate or bound |
| contribution_error | The error propagated through $c_N$, including its numerical evaluation |
| error_kind | Numerical convergence, statistical uncertainty, or rigorous enclosure |
| status | Accepted, exact zero, resource limit, or unresolved accuracy |
| method | Euler quadrature, simplex transport, facet reduction, or angular estimation |
| diagnostics | Geometry size, conditioning, precision, refinement, time, and memory |
| provenance | Weight and pattern identifiers, neuron, chart sign, and normalization convention |

Use log values when a nonzero contribution is too small to display reliably. An exact-zero status requires a justified geometric or algebraic reason. A small numerical value is not proof of zero.

Set tolerances in the Gaussian contribution units by default, because those are the quantities used by the external summation program. Translate them to raw Euler-integral units using $c_N$. Allow raw-integral tolerances explicitly for the lower-level interface. Always report which convention was used.

An absolute tolerance is essential for tiny pattern contributions. A relative tolerance may also be requested. If both are required, say so explicitly and enforce both; do not silently accept an easy absolute check while claiming accurate relative digits.

## 4. Shared computations that can be hardcoded

The denominator polynomial $1+t\cdot t$, its exponent, and the degree-one numerator class are identical for every network at fixed width. Weight dependence enters through the inequalities and affine numerator.

| Computation | Reusable across all weights | Still specific to a supplied integral |
|---|---|---|
| Scale integration | Formula for $c_N$ and exponent $(N+1)/2$ | Requested numerical precision |
| Euler integrand | Stable denominator evaluation and affine-numerator kernels | $\ell$, $c$, and domain inequalities |
| Network geometry | Effective-map recurrence and mask conventions | Actual weights and supplied masks |
| Integration by parts | Degree-one derivative identities and moment reconstruction | Facets, matrix values, and coefficients |
| Coordinate-orthant transport | Subset indexing, parity rules, sparse recurrence, and diagonal seeds | Gram matrix, conditioning, step sizes, and solution |
| Simplex integration | Parameter maps, polynomial-power formula, and quadrature templates | Ray matrix, Gram matrix, and numerator |
| Boundary integration | Surface-normal factors and intrinsic Gaussian normalization | Actual facets and their restricted domains |
| Error management | Precision escalation, tail formulas, and result contract | Magnitude, cancellation, and observed convergence |

Compute $c_N$ once per width and precision, using log-gamma arithmetic when needed. No runtime symbolic integration of the Gaussian scale is necessary.

A supplied pattern only needs its own effective maps. Row selection replaces multiplication by a mask matrix; if $k$ preceding neurons are active, $W_l[:,\mathrm{active}]A_{l-1,s}[\mathrm{active},:]$ reduces arithmetic relative to a full dense product. Shared-prefix data may be accepted from an external caller, but the solver does not build a prefix tree itself.

If an intermediate postactivation map vanishes identically, check the remaining supplied masks: they must be inactive. The final contribution is zero. If the selected final effective row is zero, return zero immediately.

Positive rescaling of completed inequality rows preserves their geometry. Independent rescaling of network weight rows generally changes deeper geometry unless compensated downstream; do not use it as a silent simplification.

Different fixed networks usually have different facets and coefficients. Hardcoding the universal identities does not provide one universal numerical transport matrix or one universal cone decomposition for all He draws.

## 5. First-moment reductions worth implementing

### 5.1 Direct Euler integration by parts

Let $d=N-1$, $Q(t)=1+t\cdot t$, and $q(t)=\ell^{\mathsf T}t+c$. The shared identity is $\nabla Q^{-d/2}=-d\,t\,Q^{-(d+2)/2}$. On a full-dimensional clipped polyhedron $P$ it gives

$$
\int_P q\,Q^{-(d+2)/2}\,dt
=c\int_P Q^{-(d+2)/2}\,dt
-\frac1d\sum_{F\subset\partial P}
(\ell\cdot n_F)\int_F Q^{-d/2}\,dS.
$$

Here $n_F$ is the outward unit normal and the sum is over actual facets. The flux at infinity vanishes; the scalar kernel decreases sufficiently quickly. Lower-dimensional or empty domains are handled separately.

This formula reduces the linear numerator to one scalar bulk integral and lower-dimensional facet integrals. The clipping plane $q=0$ is a genuine facet when it bounds the domain; include its surface term rather than assuming its contribution vanishes.

Derive and verify this identity once with SymPy, then hardcode its evaluation rules. The remaining facet geometry is weight dependent. This is a candidate direct Euler reduction, not a claim that a small closed master system for every possible polyhedron has already been established.

### 5.2 A Gaussian boundary identity tailored to the first moment

For the equivalent clipped cone $K$, Gaussian integration by parts gives an especially useful alternative:

$$
M_{s,\varepsilon,i}
=-\frac1{\sqrt{2\pi}}
\sum_{F\subset\partial K}(a_{s,i}\cdot n_F)\,p_F.
$$

Each $p_F$ is the probability of the facet cone under a standard Gaussian in its $(N-1)$-dimensional supporting hyperplane, expressed in orthonormal coordinates. The identity follows from $\nabla e^{-\|x\|^2/2}=-x e^{-\|x\|^2/2}$ and the surface Gaussian normalization. All facets of a cone lie in hyperplanes through the origin.

This representation directly evaluates the first moment through lower-dimensional Gaussian integrals. It does not require a probability for the complete activation pattern and does not produce a conditional mean.

Use actual nonredundant facets with outward normals. For an inequality $h^{\mathsf T}x\geq0$, the outward normal on its facet is $-h/\|h\|$. Duplicate facet counting or an inward normal changes the answer. Include the output-clipping and chart facets when they are actual boundaries.

Internal face integrals use the closed geometry of the original cell. A parent active inequality that becomes equality on a boundary is allowed there. Do not reapply the network's strict active-label validation to an auxiliary boundary integral and incorrectly delete that face.

Individual surface terms can have mixed signs even though the final integral is nonnegative. Track cancellation and use absolute error propagation and precision refinement.

For $N=3$, facet Gaussian probabilities are two-dimensional angular measures, giving a useful analytic geometry route. At greater widths, facet probabilities can themselves be difficult. Compare this method with whole-cone simplex transport rather than assuming that reducing the dimension always reduces total cost.

### 5.3 Simplicial pieces and odd-parity Gaussian transport

For a full-dimensional simplicial piece $K_V=V\mathbb R_+^N$ inside the clipped cone, set $G=V^{\mathsf T}V$ and $b=V^{\mathsf T}a_{s,i}$. Its Gaussian contribution is $|\det V|(2\pi)^{-N/2}b^{\mathsf T}m(G)$, where $m(G)=\int_{y\geq0}y\,e^{-y^{\mathsf T}Gy/2}\,dy$.

The numerator is nonnegative on this piece. In exact geometry its ray coefficients $b_j$ are nonnegative; an unexpectedly negative value deserves a geometry or accuracy check.

Let $v_j(G)$ be the Gaussian integral on the coordinate face $y_j=0$. Integration by parts gives the exact reconstruction $m(G)=G^{-1}v(G)$. Compute the required contraction by solving $G w=b$ and forming $w^{\mathsf T}v$, rather than forming an inverse solely for that final contraction.

The required $v_j$ have one coordinate fixed at zero. Their centered differential system closes on faces with an odd number of fixed coordinates: derivatives connect a face to itself and to faces with two additional fixed coordinates. Thus first-moment transport can use the **odd parity block**, with $2^{N-1}$ transported scalar masters for $N\geq1$.

The first-moment evaluator should transport this odd block and reconstruct the selected scalar moment. If probabilities are not requested, there is no reason to transport the even block as well.

Use $G(\lambda)=D+\lambda(G-D)$, $D=\operatorname{diag}(G)$, on $0\leq\lambda\leq1$. The symbol $\lambda$ is a transport parameter, distinct from the Euler coordinates $t$. Every intermediate Gram matrix is positive definite.

At $\lambda=0$, face integrals factor into $\sqrt{\pi/(2D_{jj})}$ for each free coordinate. The root first-moment seeds are $m_j(D)=D_{jj}^{-1}\prod_{k\ne j}\sqrt{\pi/(2D_{kk})}$. These elementary formulas can be hardcoded and evaluated without error functions or repeated SymPy work.

Build the universal centered face identities once. Assemble weight-dependent coefficient series numerically, for example using $R_0=G_A(\lambda_0)^{-1}$ and $R_{k+1}=-R_0 E_A R_k$ for each free principal block, with $E=G-D$. Reuse stable factorizations and sparse subset indexing. Matrix coefficients, convergence radii, and accepted steps still depend on the actual Gram matrix.

### 5.4 Direct simplex Euler representation

The attachment also gives the equivalent integral over the standard simplex $\Delta_{N-1}$:

$I_V=|\det V|\int_{\Delta_{N-1}}(b^{\mathsf T}\alpha)(\alpha^{\mathsf T}G\alpha)^{-(N+1)/2}\,d^{N-1}\alpha$.

For a piece confined to the selected chart, this is its contribution to the raw Euler integral. Multiplying by $c_N$ gives the same Gaussian contribution as the preceding moment reconstruction.

The affine numerator expands into terms $b_j\alpha_j$. The supplied derivation maps each term to a formal one-loop Feynman parameter integral with one power raised to two, the others equal to one, and formal dimension $N+1$. This is a useful algebraic connection to scattering-integral methods. A direct evaluator must match normalization, analytic branches, and any supported kinematic conventions.

Prototype direct Euler differential reduction only after the Gaussian moment route supplies a validated comparison. Do not assume that the correspondence alone provides an efficient evaluator or that generic scattering reduction tools already support every required instance.

## 6. Geometry for one supplied integral

Construct only the requested pattern and chart. Clip by the selected final-output inequality, then:

1. Reject inconsistent masks and handle zero rows before taking closures.
2. Establish whether the domain is empty, lower dimensional, or full dimensional. Numerical uncertainty in feasibility should trigger refinement or an unresolved status.
3. Normalize nonzero constraint rows by positive factors and remove duplicates.
4. Remove other inequalities only when redundancy is established, for example by a linear optimization check or a nonnegative-combination certificate.
5. Obtain the facets needed by boundary reduction, or a budgeted simplicial subdivision for moment transport. These are numerical integration pieces belonging to the same pattern.
6. Record conditioning, geometry approximation, and the number of pieces.

A linear dependence between normals does not by itself justify removing an inequality. The number of actual facets is bounded by the number of supplied inequalities, but generating rays, face incidences, or simplicial pieces can still be expensive.

For lineality or a genuine rank reduction, project onto the span of all constraints of $K$, including the numerator-clipping normal and chart normal. The projected Gaussian remains standard. Integrate unrestricted orthogonal directions analytically. If the integration dimension becomes $r<N$, compute the Gaussian contribution with the reduced-dimensional normalization and then divide by the **original** $c_N$ to recover the requested Euler integral. Do not simply replace $N$ in the original slice integrand without changing variables and normalization.

An external caller may provide validated geometry to avoid repeated preparation. Cache keys must include weights, masks, selected neuron, chart, geometry conventions, precision, and engine version where numerical data depend on them. Cached base-pattern geometry may be shared across neurons; their positive-output clipping generally differs.

## 7. Numerical methods and their proper roles

### Low-dimensional exact and quadrature methods

For $N=2$, the slice has one dimension. Its linear inequalities give an interval, possibly empty or unbounded, and the output clipping adds one more interval constraint. For $q(t)=\ell t+c$, the antiderivative is $(-\ell+c t)/\sqrt{1+t^2}$. Hardcode stable finite-endpoint evaluation and infinite limits. This works for any depth once the supplied pattern has been compiled.

For $N=3$, compare two-dimensional Euler quadrature with the analytic facet-angle route. These are valuable independent checks.

At higher dimensions, low-dimensional subintegrals may use BootLoops' adaptive quadrature helper. That helper is a one-dimensional integrator; nesting it is not an automatic efficient polyhedral integration algorithm. Propagate errors from nested integrations and include domain subdivision costs.

### BootLoops differential transport

Use the actual BootLoops transport engine for the odd Gaussian face block on simplicial pieces, or for a separately derived and verified direct Euler system. Supply specialized sparse Taylor coefficients through an isolated adapter. Use real arithmetic on positive-definite real paths and retain adaptive stepping and precision agreement.

Hardcode the recurrence and indexing, not the numerical solution or accepted step schedule for every weight realization. No auxiliary mass parameter is needed for the Gram interpolation because its diagonal boundary values are known.

Local arithmetic enclosures, Taylor tails, and successive refinement provide different kinds of evidence. A complete rigorous enclosure is a separate result; do not call ordinary precision agreement an interval certificate.

### Independent angular estimation of the same contribution

For Euclidean radius $R_0=\|X\|$ and uniform sphere direction $U$, use the identity

$M_{s,\varepsilon,i}=\mathbb E[R_0]\mathbb E_U[(a_{s,i}^{\mathsf T}U)_+\mathbf1_{C_s}(U)\mathbf1_{\{\varepsilon U_N>0\}}]$,

with $\mathbb E[R_0]=\sqrt2\,\Gamma((N+1)/2)/\Gamma(N/2)$. This estimator checks the selected integral directly; it does not estimate the complete mean or enumerate other patterns.

Angular sampling removes radial noise, but a rare selected cone can have very few hits. For such cases, investigate a valid importance proposal restricted to a larger tractable cone and include all proposal-density factors. Independent Gaussian minimax tilting is useful only for supported nonsingular constraint systems; it is not directly applicable to the complete dependent stack.

Use independent repetitions or independent randomizations for uncertainty assessment. A no-hit sample is not proof that the integral vanishes. This method is a reference or an explicitly statistical result when deterministic evaluation exceeds resources, not an unannounced replacement for a requested deterministic accuracy guarantee.

## 8. Tail control, normalization, and acceptance

The integral is absolutely convergent even for unbounded slices. Since $|q(t)|\leq\|a_{s,i}\|\sqrt{1+t\cdot t}$, its absolute integrand is bounded by $\|a_{s,i}\|(1+\|t\|^2)^{-N/2}$.

For $d=N-1\geq1$ and truncation radius $T>0$, a shared conservative tail bound is

$\int_{\|t\|>T}|q(t)|(1+\|t\|^2)^{-(N+1)/2}\,dt\leq\|a_{s,i}\|\,S_{d-1}/T$,

where $S_{d-1}=2\pi^{d/2}/\Gamma(d/2)$. This follows by bounding the radial integrand by $r^{-2}$. The clipped-domain tail is no larger than this whole-space tail.

The bound is reusable across all weights apart from the numerator norm. It can be too conservative for tiny contributions. Prefer compact simplex or angular representations, or sharper domain-specific bounds, when a large chart truncation radius would make the computation impractical.

For any pattern and chart, $0\leq M_{s,\varepsilon,i}\leq\|a_{s,i}\|/\sqrt{2\pi}$, because its domain is contained in the full positive halfspace for that linear observable. This supplies a scale check, not a relative-accuracy certificate.

Propagate error separately from geometry, quadrature or transport, any chart tail truncation, numerical normalization, and internal piece summation. For facet formulas or moment reconstructions, include cancellation in the precision requirement. Do not clip a negative approximate result to zero to conceal a failed computation.

Rerun deterministic evaluations with higher precision and stricter integration controls. Use independent representations where practical. Preserve a reliable logarithm for a nonzero tiny contribution. Accuracy refers to the supplied weight representation; requesting more digits does not recover digits absent from machine-precision weights.

The external summation program chooses per-integral budgets. If it has a finite set of rigorous contribution bounds, it can add those bounds to control its total. This solver reports the local evidence and does not certify an unspecified sum over patterns.

## 9. Implementation stages

### Stage 1: Fix the single-integral contract

Implement the two entry points, pattern-through-$L-1$ convention, chart signs, final-neuron selection, positive-output clipping, and both raw and Gaussian-normalized result fields. Construct only the requested geometry.

Completion criterion: the domain and numerator match the supplied derivation, and no call generates unrelated activation patterns.

### Stage 2: Deliver exact small-width checks

Implement the width-two interval formula, exact-zero handling, and stable $c_N$. Build independent direct-network and Euler-domain checks for supplied patterns. Use low-dimensional angular geometry where appropriate.

Completion criterion: the worked example below and chart normalization checks agree analytically.

### Stage 3: Build first-moment boundary reduction

Implement the Gaussian facet identity and projected facet domains, initially at small widths. Verify orientation, duplicate handling, surface normalization, and strict-cell versus closed-boundary conventions. Compare direct Euler quadrature with facet results.

Completion criterion: independent formulations agree for selected patterns and charts, including extinction and clipping cases.

### Stage 4: Implement specialized odd-parity transport

Use SymPy during development to verify the centered face identities and odd-block closure. Generate common subset indexing and sparse numerical recurrences. Evaluate Gram-dependent coefficients numerically and reuse BootLoops transport.

Reconstruct only the selected scalar first moment. Extend to a vector of numerators only where the clipped integration domain is genuinely shared.

Completion criterion: simplicial Euler, Gaussian moment, and independent numerical results agree at the declared accuracy. Report setup and integration costs separately.

### Stage 5: Support more difficult single domains

Add budgeted geometry preparation, simplex subdivision, sharper tail control, and optional angular importance estimation. Return a resource or accuracy status when a requested integral exceeds the supported range.

Completion criterion: geometry and numerical limits are explicit and reproducible for one supplied pattern.

### Stage 6: Compile measured bottlenecks

Profile pattern preparation, facet extraction or subdivision, matrix-series assembly, transport, and scalar reconstruction. Compile expensive shared loops and reuse the existing optional compiled arithmetic backend where beneficial. Keep the external interface in Python.

A C++ rewrite is justified only by measurements of the selected integral workload. It does not remove exponential boundary-state growth or difficult cone geometry.

Completion criterion: measured gains preserve independent first-moment checks; distinguish cold setup from repeated calls using prepared geometry. Do not claim speedups from source changes alone.

## 10. Validation focused on individual first-moment integrals

The validation set must not require enumeration of all network patterns.

For the attachment's worked example, use $N=2$, $L=2$, $W_1=\left(\begin{smallmatrix}1&0\\1&1\end{smallmatrix}\right)$, selected output row $(W_2)_{1,:}=(1,0)$, supplied mask $s_1=(1,0)$, and chart $\varepsilon=-1$. Only this supplied integral is tested.

| Case | Required check |
|---|---|
| Attachment's width-two example | For the supplied pattern, negative chart, $q=t$, and $P^+=[0,1]$, raw $I=1-1/\sqrt2$ and contribution $(1-1/\sqrt2)/(2\sqrt{2\pi})$ |
| Width-two general interval | Endpoint formula agrees with independently refined quadrature |
| Constant positive numerator $q=c$ on a full slice | Raw $I=c\,\pi^{(N-1)/2}/\Gamma((N+1)/2)$ and contribution $c/\sqrt{2\pi}$ |
| Empty slice or impossible active zero row | Exactly zero, with a justified status |
| Selected final effective row zero | Exactly zero for that pattern and chart |
| Output clipping intersects a supplied slice | Boundary reduction includes the new clipping facet |
| Unbounded slice | Chart evaluation with controlled tail agrees with compact or Gaussian representation |
| One supplied simplicial piece | Euler-simplex integral agrees with odd-block Gaussian moment transport |
| One supplied cone and chart | Direct Euler evaluation agrees with Gaussian facet reduction |
| Width one edge case | Explicit calculation on the two input directions; no positive-dimensional Euler integration |
| Positive scalar rescaling of all weights in one layer | Pattern and clipping geometry unchanged; the first-moment contribution scales by that factor |
| Closed auxiliary face of an active parent constraint | Surface integral is retained even though the active preactivation is zero on that boundary |
| Nearly degenerate geometry | Refinement preserves accuracy or returns unresolved status; no hidden covariance perturbation |
| Repeated prepared-domain call | Matches a fresh call at the same accuracy |

For $L=1$, the empty pattern's two chart contributions add to $\|(W_1)_{i,:}\|/\sqrt{2\pi}$. This is a simple same-pattern normalization check. It is not a requirement to sum over activation patterns.

A low-dimensional benchmark should use several fixed weight realizations, supplied feasible and infeasible patterns, both chart signs, and numerator choices that produce small or strongly clipped contributions. Record how queries were selected: choosing a pattern from a sampled input favors regions with larger probability.

Deterministic comparison tolerances must apply to the first-moment contribution, not reuse a probability-validation threshold without justification. Statistical comparisons should use their reported uncertainty. Both routes should distinguish local numerical evidence from a rigorous enclosure.

The current project's historical orthant receipts do not validate this new first-moment engine. Implementation will require fresh checks; this plan does not run them.

## 11. Expected efficiencies and remaining bottlenecks

The main reusable gains are the exact scale integral, a fixed degree-one kernel family, analytic width-two evaluation, first-moment boundary identities, odd-parity transport, sparse compiled arithmetic, and reuse of validated geometry supplied by the caller. The solver also avoids the much larger task of enumerating activation patterns.

Per-integral cost can nevertheless be substantial. For an $n$-dimensional simplicial Gaussian moment problem, the odd state contains $2^{n-1}$ masters:

| Integration dimension | Odd-parity scalar master count |
|---:|---:|
| 8 | 128 |
| 12 | 2,048 |
| 16 | 32,768 |
| 20 | 524,288 |
| 32 | 2,147,483,648 |

These counts exclude coefficient storage and temporary matrix data. Facet reduction can lower integration dimension but introduces multiple facet integrals. A nonsimplicial cone can require many subdivisions. Depth affects the number and conditioning of inequalities even when the number of input variables is fixed.

He initialization does not supply an exact low-rank or equal-correlation reduction for a fixed network. Specialization improves repeated algebra and arithmetic; practical high-width accuracy still needs to be established for the actual single-integral workload.

Measure geometry preparation, integration, precision refinement, and memory independently. Report whether one numerical contribution is difficult because of many facets, many simplicial pieces, poor conditioning, tiny magnitude, or transport state size. None of these diagnostics requires examining the full family of activation patterns.

## 12. Proposed module boundaries and references

A future specialized package can separate:

- **pattern_integral:** public contract for one supplied pattern, neuron, and chart.
- **pattern_geometry:** effective maps, affine slice constraints, clipping, and feasibility.
- **euler_kernels:** stable scalar kernels, normalization, and low-dimensional formulas.
- **facet_moments:** Gaussian first-moment surface reduction and restricted domains.
- **simplex_moments:** ray and Gram conversion, and scalar moment reconstruction.
- **odd_transport:** universal centered face templates and the BootLoops adapter.
- **local_accuracy:** precision, error propagation, resource limits, and provenance.
- **validation:** independent checks of individual integrals.

The pattern-enumeration and pattern-summation programs are external consumers of this interface, not modules that this implementation must build.

The mathematical source is the supplied [Euler-moment derivation](/Users/lhd/Documents/Codex/2026-10-02/https-bootloops-ai-bootloops-pdf-https/outputs/euler_moments.tex), especially its postactivation formula, simplex conversion, and scale normalization. The Gaussian surface and odd-parity reductions proposed here should be independently verified during implementation.

The existing [Gaussian transport source](/Users/lhd/Projects/orthant_probs/orthant_solver/gaussian_orthant/transport.py) and [mathematical documentation](/Users/lhd/Projects/orthant_probs/orthant_solver/output/pdf/gaussian_orthant_methods.pdf) provide reusable numerical infrastructure. [BootLoops](https://bootloops.ai/bootloops.html) supplies actual adaptive transport and quadrature machinery. [Koyama and Takemura](https://arxiv.org/abs/1211.6822) provide established Gaussian differential-system context, and [Botev](https://arxiv.org/html/1603.04166) supplies an independent method for supported nonsingular Gaussian restriction subproblems.

The practical conclusion is to build an evaluator for the exact clipped Euler integral of one supplied pattern, chart, and final neuron. Hardcode the shared first-moment identities and numerical recurrences, prepare the weight-dependent geometry locally, and return its normalized contribution with explicit accuracy evidence. Pattern selection and the complete mean-output sum remain separate tasks.
