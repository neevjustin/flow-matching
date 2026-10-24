# Learned Flow-Matching Velocity Fields Are Nearly Curl-Free

### Measuring spurious circulation and its distributionally invisible transport cost

**Status of this draft.** Every number below traces to a row in
`paper/claims_and_evidence.md` and a file under `results/raw/`. Claims are tagged
**[established]**, **[derived]**, **[measured]**, or **[conjecture]** on first use. The
central empirical result is a **falsification of our own pre-registered hypothesis**.
Scope limits are in `paper/limitations.md` and summarised in §8; the most important is
that all experiments are two-dimensional.

---

## Abstract

A path of probability densities does not determine the velocity field that transports it:
marginal-preserving ("flux-null") perturbations form a subspace as large as the space of
gradient fields **[established]**. It is therefore natural to suppose that learned
flow-matching models, scored by training loss and sample quality, might differ
consequentially along that subspace. We show this supposition is false in two steps, and
then measure what is actually true.

First, the conditional-flow-matching objective is a least-squares regression with a unique
minimiser, so there is no gauge freedom at the optimum **[established]**; and for a
Gaussian source with independent coupling and a linear interpolant that minimiser is
itself a *gradient field* **[established]**. Hence the flux-null subspace can contain
*only error*, with known correct value zero. This turns a question about model
identifiability into a measurable question about the geometry of a trained network's
residual.

Second, we measure it. Using closed-form ground truth on 2-D Gaussian-mixture targets, an
exactly-orthogonal weighted-Helmholtz estimator, and a grid-free Jacobian-asymmetry
diagnostic, we find that trained CFM networks place only `f = 0.140 ± 0.032` of their
velocity-error energy in the flux-null subspace, against `0.720 ± 0.037` for a matched null
that preserves the residual's magnitude field and spectrum and randomises only its
direction (`p = 1.3 × 10⁻¹⁴`). Local rotationality is `κ = 0.030 ± 0.004` versus
`0.803 ± 0.336` at random initialisation. **Networks suppress the marginal-preserving
freedom rather than exploiting it** — our pre-registered hypothesis is falsified in the
opposite direction.

The circulation that remains is nonetheless exactly invisible to distributional metrics.
Injecting the *measured* flux-null component of a trained network's error into the true
field leaves the endpoint distribution indistinguishable from ground truth
(`|KL| ≤ 0.007`, with an instrument validated to `7 × 10⁻⁸` nats that reads exactly `0` on
the true field) while displacing the transport map by 0.27 % of the data radius, moving the
worst particles by 42 % of it, and re-routing `0.31 % ± 0.04 %` of samples to a *different
mode* at essentially unchanged mode occupancy. A curl-free (potential) parameterisation
removes the circulation by construction and reduces map error by 39 %, though we show most
of that gain follows from its lower velocity error rather than from the removal itself.

We contribute the measurement, a dimension-scalable diagnostic with a known correct value
of zero, and an explicit negative result. We do **not** contribute the underlying theory or
the architecture, both of which are prior work.

---

## 1. Introduction

Flow matching trains a velocity field `v_θ(t,x)` so that the ODE `ẋ = v_θ` transports a
simple base distribution to data. The field is fitted by regression onto a conditional
target, and models are judged by held-out loss and by distributional metrics on the
generated samples.

There is a well-known gap in this picture. The link between a field and the densities it
moves is the continuity equation

$$\partial_t p_t + \nabla\cdot(p_t v_t) = 0,$$

which is one scalar constraint on a `d`-component field. Adding any `w_t` with
`∇·(p_t w_t) = 0` leaves every marginal `p_t` untouched while changing every trajectory.
These **flux-null** perturbations form the orthogonal complement of the gradient fields in
`L²(p_t)` and are parameterised, in two dimensions, by one free scalar function per time
slice — as many degrees of freedom as the gradient part **[established]**
(Lim, 2025, Prop. 5.4–5.5; classically Benamou–Brenier, Ambrosio–Gigli–Savaré). Recent
work independently notes that standard metrics such as FID are "trajectory-blind"
**[established]** (Li et al., 2026).

Put together, these invite a hypothesis: *learned flow-matching models may differ
consequentially in their trajectories while matching on loss and on endpoint
distributions.* We set out to test it. Two corrections were needed before it could even be
stated precisely, and both matter enough to go in the introduction.

**Correction 1: the objective has no gauge freedom.** The CFM loss
`E‖v_θ(t,x_t) − (x_1−x_0)‖²` is pointwise strictly convex in the field's value, so its
minimiser over an unrestricted class is the unique conditional expectation
`v*(t,x) = E[x_1−x_0 | x_t=x]`, determined `dt⊗p_t`-a.e. **[established]** (Lim, 2025,
Prop. 4.3). Non-uniqueness of the *continuity equation* is not non-uniqueness of the
*loss*. A model at the optimum cannot select a different flux-equivalent field. Whatever
freedom there is must live in the residual `e = v_θ − v*`.

**Correction 2: the target is itself a gradient field.** For `p_0 = N(0,I)` with
independent coupling and the linear interpolant, `x_t | x_1 ∼ N(t x_1, (1−t)²I)`, so `p_t`
is a Gaussian convolution and Tweedie's formula gives
`E[x_0|x_t=x] = −(1−t)∇\log p_t(x)`. Hence

$$v^*(t,x) \;=\; \frac{x + (1-t)\,\nabla \log p_t(x)}{t} \;=\; \nabla_x\Big[\tfrac{\|x\|^2}{2t} + \tfrac{1-t}{t}\log p_t(x)\Big].$$

So `v*` is **curl-free**: its flux-null component is exactly zero. This is
**[established]** — it is Example 5.1 of Lim (2025), which gives the equivalent
velocity–score relation for independent coupling, and the relation itself is standard
(Lipman et al., 2022; Albergo & Vanden-Eijnden, 2022). We rediscovered it mid-project and
verified it numerically to `10⁻¹⁴`; we flag the attribution prominently because the
temptation to claim it is real.

The consequence reframes the whole question. Since `f(v^*) = 0`, the flux-null part of the
*error* is the flux-null part of the *learned field itself*:

$$r(e) \;=\; r(v_\theta).$$

A rotational component in a learned field is therefore **not** a legitimate alternative
choice. It is **spurious circulation**: pure error, with a known correct value of zero,
exactly invisible to every functional of the generated distribution, and fully active on
the transport map. The well-posed question becomes:

> How much spurious circulation do trained flow-matching networks carry, is it more or
> less than an unstructured field of the same smoothness would have, and what does it cost?

This is precisely the question the closest prior work declines to answer. Lim (2025) builds
the entire flux-null theory but states: *"All experiments evaluate the closed-form
empirical velocity directly; no neural network is trained,"* listing neural-network
approximation and solver effects as future work.

**Contributions.**

1. **Two instruments with known correct value zero** (§4). A finite-volume weighted
   Helmholtz decomposition whose orthogonality holds to machine precision, so the
   flux-null energy fraction `f ∈ [0,1]` by construction; and a grid-free, reference-free
   Jacobian-asymmetry diagnostic `κ = ‖J−Jᵀ‖_F/‖J+Jᵀ‖_F` estimable at cost independent of
   dimension.
2. **The measurement, and a negative result** (§6.2–6.3). `f = 0.140 ± 0.032` and
   `κ = 0.030 ± 0.004`, far *below* matched nulls. Our pre-registered hypothesis — that
   networks preferentially err in marginal-preserving directions — is **falsified in the
   opposite direction**. We also report, and do not excuse, the degeneracy of our own
   pre-registered null model.
3. **The consequence** (§6.4–6.5). The residual circulation is distributionally invisible
   yet re-routes samples between modes, with the damage concentrated in a heavy tail and
   localised to inter-modal separatrices.
4. **A controlled intervention, reported with its confound** (§6.6).

---

## 2. Setup

Data `p_1`, base `p_0 = N(0,I_d)`, independent coupling `q = p_0 ⊗ p_1`, linear interpolant
`x_t = (1−t)x_0 + t x_1`, regression target `u = x_1 − x_0`. We write
`⟨a,b⟩_{p} = ∫ p\,a\!\cdot\!b`, `‖a‖²_{dt⊗p} = ∫_0^1‖a_t‖²_{p_t}dt`.

**Flux-null space** **[established]**.
`R_p = {w ∈ L²(p;ℝ^d) : ∫ p\,w\!\cdot\!\nabla\psi = 0\ \forall\psi\in C_c^\infty\}`.
Since this says `w ⊥ \overline{\{\nabla\psi\}}`, we have the orthogonal splitting
`L²(p;ℝ^d) = G_p ⊕ R_p`, so every `e` decomposes uniquely as `e = ∇φ + r` with
`‖e‖²_p = ‖∇φ‖²_p + ‖r‖²_p`, where `φ` solves the weighted Poisson problem
`∇\cdot(p∇φ) = ∇\cdot(p e)`. We define the **flux-null energy fraction**

$$f \;=\; \frac{\int_0^1\|r_t\|^2_{p_t}\,dt}{\int_0^1\|e_t\|^2_{p_t}\,dt}\;\in[0,1].$$

**Regularity** (used throughout, see `research/theory.md` §1). For `t ∈ (0,1)` and `p_1`
of finite second moment, `p_t` has a strictly positive `C^∞` density on all of `ℝ^d`, so
there are no support-boundary subtleties; both `v*` and a finite SiLU MLP are locally
Lipschitz with linear growth, giving well-posed flows (cf. Wan et al., 2025, Thm 4.1). All
measurements are restricted to `t ∈ [0.02, 0.98]`.

**Two identities we use** **[derived]**. Because `E[u|x_t] = v^*`, the cross term vanishes
and the held-out excess CFM loss is exactly the squared velocity error,
`L(θ) − L_{\text{floor}} = η² := ‖e‖²_{dt⊗p}`. And because the law of `X^*(s;x_0)` is
`p_s`, that same quantity is the *along-trajectory* error energy,
`E_{x_0}\int_0^1\|e(s,X^*(s))\|^2ds = η²`.

---

## 3. What each error component does

**Proposition 1 (asymmetry)** **[derived]**, `research/theory.md` §3.2. Let
`v_θ = v^* + e` and let `p̂_t` be the law of the flow of `v_θ`.

*(a) Marginals.* Linearising the continuity equation,
`∂_tε_t + ∇\cdot(ε_t v^*_t) = -∇\cdot(p_t e_t)` with `ε_0 = 0`. Only `∇\cdot(p_t e_t)`
enters, hence only `∇φ`. (This error PDE is Prop. 3.1 of Huang et al., 2025; we re-derive
it to expose that the forcing term is exactly the weighted divergence.) Non-perturbatively:
if `e_t ∈ R_{p_t}` for a.e. `t`, then `p̂_t = p_t` **exactly**, so *every* functional of
`p̂_1` — FID, `W_2`, KL, TV, precision/recall, mode coverage — is exactly blind to it.

*(b) Trajectories.* With `δ = X_θ − X^*`,
`δ(1) = \int_0^1 J(1,s)\,e(s,X^*(s))\,ds`, where `J` solves the linearised flow. This
involves the pointwise values of *all* of `e`, with no divergence taken and no cancellation
of that kind. Hence for flux-null `e ≠ 0`, `ε_1 = 0` while `δ(1) ≠ 0` generically. ∎

With Cauchy–Schwarz this gives `M := ‖δ(1)‖_{L²(p_0)} ≤ C_G\,η`, a Gronwall bound whose
amplification factor `C_G = \sup_s\|J(1,s)\|` we measure directly rather than bound
(§6.1), so that "the error is amplified" and "the error is large" stay separable.

---

## 4. Two instruments

**(i) `f`, by an exactly-orthogonal discretisation** **[derived]**. We sample the field at
the *midpoints of grid faces* (no interpolation: `p_t`, `v^*` and `v_θ` are all evaluable
there) and solve the Galerkin problem `⟨∇_hφ,∇_hψ⟩_h = ⟨e,∇_hψ⟩_h` for all grid functions
`ψ`, where `⟨a,b⟩_h = h²Σ_{\text{faces}} p_f a_f b_f`. Taking `ψ = φ` gives
`⟨∇_hφ, r⟩_h = 0` **identically**, so `f ∈ [0,1]` by construction and
`‖e‖²_h = ‖∇_hφ‖²_h + ‖r‖²_h` to linear-solver accuracy. The measured orthogonality
residual is therefore a *solver-convergence* diagnostic, reported with every
decomposition; the closure residual is `~10⁻¹⁶`. Cells with `p_t < 10⁻⁸\max p_t` are
dropped, which is the correct discretisation of the `L²(p_t)` integral and — unlike
flooring the weights — does not promote the network's large extrapolation error in the far
tails. Because the discrete gradient space is a *subspace* of the continuum one, `f` is
biased slightly **upward**; refining `192 → 256` changes it by `3×10⁻⁵`.

**(ii) `κ`, grid-free and dimension-scalable** **[derived]**. Since `v^*` is a gradient
field, its Jacobian is symmetric, so

$$\kappa \;=\; \big\langle \|J - J^\top\|_F \big/ \|J + J^\top\|_F \big\rangle_{t,\,p_t},
\qquad J = \nabla_x v_\theta,$$

measures spurious circulation with known correct value `0`, needing **no grid and no
reference field**. `‖J−Jᵀ‖_F²` is estimable by Hutchinson probes
(`E_u\|(J−J^\top)u\|² = \|J−J^\top\|_F²`), i.e. one JVP and one VJP per probe, at cost
independent of `d`. In 2-D the exact and Hutchinson values agree to 4 % (0.0272 vs 0.0282;
0.0352 vs 0.0360), and `κ(v^*) = 0.0` to machine precision. This is the instrument we would
use at image scale.

**Why standard instruments fail here** **[measured]**. `W_2` in 2-D converges like
`n^{-1/4}`; at `n = 4000` the two-sample noise floor is `0.3668` while the **ground-truth
map itself** reads `W_2(Φ^*\#p_0, p_1) = 0.3871`. `W_2` cannot resolve the error of a
perfect model, let alone a good one, and no claim here rests on it. The energy distance is
within one standard error of zero for every field we tested. We therefore use
`KL(p̂_1\|p_1)` computed by the exact continuous change of variables against the
closed-form `\log p_1`, with analytic divergences and fixed-step RK4. Validation: on the
**true** field, whose KL is exactly `0`, the instrument returns `9.2×10⁻⁸` (50 steps),
`7.0×10⁻⁸` (100), `6.6×10⁻⁹` (200).

---

## 5. Testbed, and the validation gate

**Closed-form ground truth** **[derived]**. For `p_1 = Σ_k π_k N(μ_k,Σ_k)` under a Gaussian
base, `(x_t,u)` is conditionally Gaussian per component, giving closed forms for `p_t`,
`v^*`, and the CFM loss floor (`research/theory.md` §4). The primary target is `ring8`:
eight equal-weight isotropic Gaussians (`σ = 0.3`) on a radius-4 ring. Because `σ > 0`,
`v^*` is smooth up to `t = 1` and the `1/(1−t)` terminal singularity of the empirical
minimiser is *absent by construction* rather than merely bounded.

**The most important design choice.** We train on a **fresh i.i.d. sample at every step**.
The exact minimiser of the *empirical* CFM loss memorises the training set, so using it as
ground truth would conflate error with desirable generalisation. With a fresh-sample
stream the regression target genuinely is the population `v^*`, and `e = v_θ − v^*` is pure
approximation + optimisation error. The price is that all results describe the
infinite-data regime (§8.7).

**Models.** MLP, width 128, depth 3, SiLU, 16-frequency fixed Fourier time embedding,
37 890 parameters; Adam `2×10⁻³` cosine-decayed, batch 2048, 20 000 steps; trained in
float32 and cast to float64 for every measurement. Deliberately standard — anything
unusual would be a confound.

**Validation gate (E0): 25 checks, all pass.** Nothing downstream was believed until it
did.

| check | result |
|---|---|
| `∫ p_t` on the grid | `1.00000000` at `t ∈ \{0,¼,½,¾,1\}` |
| `v^*` is `E[u|x_t]` (residual ⟂ `\{1,x,x²,\sin\}`) | worst z-score 3.16 over `4×10⁵` samples |
| closed-form loss floor vs Monte Carlo | rel. `6×10⁻⁶` … `3.5×10⁻³` |
| continuity equation `∂_tp + ∇\cdot(p v^*)` | L² rel. residual `2.3×10⁻²→5.8×10⁻³→1.5×10⁻³`, **observed order 1.99–2.00** |
| `v^*` transports `p_0→p_1` | `W_2 = 0.269` vs same-`n` floor `0.329` |
| decomposition of a pure gradient field | `f = 0.00000` |
| decomposition of a pure flux-null field | `f = 1.00000` |
| equal-energy mixture | `f = 0.50000` |
| grid refinement / density-threshold sweep | `f` stable to `10⁻⁷` |

Two bugs were found by this gate, both in the *tests* rather than the measured objects: a
degenerate z-score scale for the constant test function, and reversed face orientations in
the finite-volume divergence (which produced a relative residual of exactly `1.000` — the
signature of computing `∂_tp − \nabla\cdot` instead of `∂_tp + \nabla\cdot`).

**Mechanism check.** A *constructed* analytic flux-null perturbation of amplitude 56 % of
`‖v^*‖` leaves `W_2` at the sampling-noise floor while moving the transport map by 26.6 %
of the data radius — the phenomenon exists in this testbed before any network is involved.

---

## 6. Results

### 6.1 First, the control: not a solver artefact **[measured]**

Trajectory differences could be integrator error rather than field error. They are not.
Across `rtol ∈ \{10^{-5}, 10^{-7}, 10^{-9}\}` (DOP853), the map error and the between-seed
map disagreement each change by **less than 0.01 %** (measured 0.0068 %) (`M`: `0.01966 → 0.01966`). Our pre-declared
refutation threshold was 10 %. Separately, the Gronwall amplification along the true flow
is modest — `C_G` median `1.22`, mean `3.47`, p90 `5.84` — so map error is not
predominantly an amplification effect.

### 6.2 The measurement **[measured]**

Over 8 seeds: `f = 0.1402 ± 0.0319`, with `η_rel = ‖e‖/‖v^*‖ = 0.0164 ± 0.0008` and
`M = 0.0217 ± 0.0023`. Grid stability `|f(192) − f(256)| = 3.0×10⁻⁵`. Independently and
grid-free, `κ = 0.0304 ± 0.0041`. So roughly 14 % of velocity-error energy, and ~3 % of
local Jacobian asymmetry, is spurious circulation — non-zero, but small.

### 6.3 The natural hypothesis is falsified, in the opposite direction **[measured, negative]**

We pre-registered H1: that `f` would exceed a matched null by more than `0.05`. The matched
nulls take the *measured* residual and rotate its vector values (globally, by a random
angle or by 90°), which preserves the magnitude field `|e|(x)`, its spatial spectrum, and
the total energy exactly, and randomises only the alignment between the error's direction
and the local weighted-gradient geometry — exactly the quantity H1 is about.

| | `f` | vs trained |
|---|---|---|
| trained CFM net | **0.1402 ± 0.0319** | — |
| null: residual rotated 90° | 0.7201 ± 0.0367 | `p = 1.3×10⁻¹⁴` |
| null: residual rotated randomly | 0.2737 ± 0.1937 | `p = 0.094` |
| null: random-init net (**degenerate**) | 0.0050 ± 0.0087 | `p = 2.7×10⁻⁶` |
| `v^*` (the target) | **0** exactly | — |

`f` is far **below** chance. Equivalently, `κ = 0.030` for trained nets against
`0.803 ± 0.336` at random initialisation — a ~26× suppression of local rotationality.
**Trained flow-matching networks do not exploit the marginal-preserving freedom; they
substantially inherit the exact curl-free structure of the target.** H1 is falsified.

**We report a failure of our own pre-registration.** Against the *pre-registered* null
(`null_init`) the criterion is formally met. We do not count it, because that null is
degenerate: for a randomly-initialised net of this architecture the *constant* (bias)
component carries **89–99.5 %** of the field's energy (`|E[v]| ≈ 0.04–0.13` while
`std_x(v) ≈ 0.004–0.011`), and a constant field is a trivial gradient. It measures "what
fraction of a near-constant field is rotational" ≈ 0, not "what an unstructured field of
matched smoothness would give". The lesson — validate the *null* for non-degeneracy before
the decisive run, as we did for the instruments — is one we learned here, late.

**Why the suppression happens is open** **[conjecture]**. The CFM loss penalises `‖∇φ‖²`
and `‖r‖²` *equally*, since they are orthogonal in `L²(p_t)`. Any asymmetry must therefore
come from the function class or the optimiser's implicit bias, not from the objective. We
have no proof and offer none.

### 6.4 The remaining circulation is invisible, and it re-routes samples **[measured]**

We extract the measured components of a trained network's residual and inject them, one at
a time, into the *true* field. Only the scalar potential is interpolated (bicubic spline,
analytic derivatives); the residual is then `r = e − ∇φ_{\text{spline}}` with `e` evaluated
**exactly**, so `∇φ + r = e` identically. (An earlier version interpolated both components
on faces; its reconstruction check failed at 28–35 % of `M_θ`, comparable to the quantity
of interest, and it is recorded as superseded and unused.) Validation: the Monte-Carlo
flux-null fraction of the spline split matches the grid value to `≤ 0.003` on all seeds.

| field injected into `v^*` | `M` | p99 | max | mode-flip | occupancy TV | `KL(p̂_1\|p_1)` |
|---|---|---|---|---|---|---|
| — (`Φ^*`, ground truth) | 0 | — | — | 0 | 0 | **0.00000 ± 0.00000** |
| **flux-null part `r` only** | **0.0106 ± 0.0004** | 0.060–0.077 | 1.39–1.70 | **0.31 % ± 0.04 %** | 0.0013–0.0017 | **−0.0045 ± 0.0021** |
| full error `e` (i.e. `v_θ`) | 0.0205 ± 0.0024 | 0.155–0.326 | 1.51–1.62 | 0.53 % | 0.0010–0.0037 | 0.0183 ± 0.0035 |
| gradient part `∇φ` only | 0.0188 ± 0.0021 | 0.135–0.302 | 1.48–1.53 | 0.50 % | 0.0020–0.0033 | 0.1479 ± 0.0247 |

The measured flux-null component leaves the endpoint distribution indistinguishable from
ground truth (`|KL| ≤ 0.0069`, within ~2σ of zero and 4× below the full model's own
`0.0183`, on an instrument that resolves `10⁻⁷`) while displacing the map by `0.0106`
(0.27 % of the data radius), moving the worst particle by `1.70` (**42 % of the data
radius**), and re-routing `0.31 %` of samples to a different mode with mode occupancy
essentially unchanged. This is "same endpoint distribution, different transport map" made
concrete at the level of individual samples, with a *measured* component of a *trained*
network's error.

**Two caveats we do not hide.** (i) `KL_{\text{null}}` is slightly *negative* in all three
seeds; true KL is `≥ 0`, so there is a small systematic bias we cannot fully attribute
(the spline split is exactly flux-null only in the discrete face-weighted sense; `t` is
interpolated linearly). We therefore claim `|KL_{\text{null}}| ≤ 0.007`, not
`KL_{\text{null}} = 0`. (ii) The components do **not** act additively on the marginals:
`KL(Φ_{∇φ}) = 0.148` is 8× the full model's, because `r` is flux-null with respect to
`p_t`, which `v^*` generates — not with respect to the different path that `v^* + ∇φ`
generates. **Consequently we may not say "a fraction `f` of the model's map error is
invisible."** The defensible claim is the injection claim above, and nothing stronger.

### 6.5 Heavy tail, and localisation to the separatrices **[measured]**

The mean map error hides its own structure: against the true map, seed 0 gives mean
`0.0198` but median `0.0090`, p90 `0.025`, p99 `0.171`, max `1.94`; between seeds
RMS/mean `= 4.4`. Spatially (Fig. 2), the local flux-null share of error energy forms
radial spokes in the angular *gaps between* mode centres and is near zero on the modes
themselves. The re-routed particles (Fig. 5) cluster near the origin in the source and in
the inter-modal gaps at the endpoints. This is the low-density separatrix region, where the
component posteriors `γ_k(t,x)` vary fastest and where the `p_t`-weighted loss supplies the
least gradient signal — the mechanism we had hypothesised for H1, visible in the spatial
structure even though H1's aggregate prediction failed.

### 6.6 A curl-free parameterisation, and its confound **[measured]**

Because `v^*` is exactly a gradient, parameterising `v_θ = ∇_x Ψ_θ(t,x)` is a **zero-bias**
restriction that removes the flux-null subspace by construction. The architecture is
**not** ours — Action Matching learns `v = ∇s_t`, and Energy Matching uses a
time-independent potential motivated by optimal-transport heuristics rather than by the
exact curl-freeness above; neither validates irrotationality empirically or measures
transport-map fidelity. We treat it purely as a controlled intervention. Matched width,
depth, steps, batch, lr and seeds; 37 761 vs 37 890 parameters.

| | standard `v_θ` | curl-free `∇Ψ_θ` | Welch `p` |
|---|---|---|---|
| `f` | 0.1334 ± 0.0347 | `5.5×10⁻⁵ ± 3.8×10⁻⁶` | `4.6×10⁻³` |
| `η_rel` | 0.01682 ± 0.00087 | **0.01226 ± 0.00068** | `2.2×10⁻⁴` |
| `M` | 0.02351 ± 0.00254 | **0.01432 ± 0.00080** | `3.4×10⁻³` |
| `KL` | 0.0214 ± 0.0099 | **0.0082 ± 0.0018** | 0.073 |
| `M/η_rel` (amplification) | 1.400 ± 0.167 | **1.170 ± 0.070** | 0.064 |
| wall-clock / run | 188 s | 392 s (**2.09×**) | — |

The restriction is better on every axis. But the comparison is **confounded**, and the
confound is the interesting part: the potential network is also a **better velocity
regressor** (`η_rel` lower by 27 %), so its 39 % map-error reduction is not a clean
isolation of "removing circulation". Normalising by velocity error, the amplification ratio
`M/η` falls only from `1.400` to `1.170` — a 16 % reduction, marginal at four seeds
(`p = 0.064`). **About three-fifths of the map-error gain is explained by the better
velocity fit; roughly two-fifths is specific to the geometry.** A clean test would hold `η`
fixed across architectures, which we did not do.

Two of our four pre-registered predictions for this experiment were wrong, both
informatively. Q2 (that `η` would change by `< 25 %`) **failed** at 27.1 %, in the
favourable direction — it was written to guard against the restriction costing accuracy,
and that concern was simply mistaken. Q4 (that the distributional discrepancy would *not*
improve, so that the gain would be map-specific) was **refuted**: `KL` improves by 62 %.
The intervention is therefore better than we predicted and less diagnostic than we hoped.

### 6.7 A ground-truth-free surrogate **[measured]**

`v^*` is unavailable outside synthetic settings, but `Δ = v_A − v_B` for two independently
trained models is not. Decomposing `Δ` gives `f_Δ = 0.2069 ± 0.0270` over 28 pairs, against
the true `f = 0.1402` — biased **up by ~48 %**. This is the direction predicted by the
failure of the independence assumption behind the surrogate: the *shared, systematic* part
of the error is more gradient-like than the seed-idiosyncratic part, and differencing
removes the shared part. The surrogate is usable as an order-of-magnitude proxy, with that
bias stated. (`κ` needs no surrogate — it requires no reference field at all.)

---

## 7. Related work

**Lim (2025), *On the Hidden Biases of Flow Matching Samplers*** is the closest work and
contains essentially all of the theory we use: the flux-null space and its orthogonality
to gradient fields, flux equivalence, the explicit Gaussian flux-null family, the
closed-form empirical minimiser, and — in Example 5.1 — the gradient structure of the
population minimiser under independent coupling. It trains **no networks** and measures
neither the flux-null content of a learned field nor transport-map error, listing both as
future work. Our paper is the empirical complement, not an extension of the theory. One
clarification: that paper's Prop. 5.3 (the empirical minimiser is generically *not* a
gradient field) is stated for general affine conditional flows with a latent that is *"not
necessarily Gaussian"*; for the Gaussian-source linear interpolant the same Tweedie
argument applies to the empirical `p̂_t`, so `v̂^*` **is** a gradient field and the
genericity remark does not bite in the standard CFM setting (verified numerically on a
48-component mixture, `f = 10⁻⁶`).

**Huang et al. (ICML 2025), *Improving Flow Matching by Aligning Flow Divergence*** proves
that the density error is driven by the weighted divergence of the velocity error and adds
a divergence-matching loss. Their own error PDE therefore implies that a flux-null velocity
error produces *zero* density error — the mechanism we exploit — but they neither name it
nor explore it, and all their metrics are distributional. Their method cannot address
spurious circulation: `L_{CDM}` annihilates flux-null perturbations by construction.

**Li et al. (ICML 2026), *A Kinetic Energy Perspective of Flow Matching*** states that
"standard metrics like FID are fundamentally trajectory-blind" and introduces a per-sample
kinetic-energy diagnostic. It does not discuss non-uniqueness or any decomposition. Our
`f` and `κ` are complementary: kinetic energy measures how far a trajectory works, `κ`
measures whether the field has structure the target provably lacks.

**Wan et al. (ICML 2025)** give the regularity and terminal-convergence theory we cite, and
note that "behavior with finite-capacity networks [is] uncharacterized". **Vaitl & Klein
(2025)** differentiate through the flow **map** for path gradients, making them a downstream
consumer that spurious circulation would bias; they observe, without framing it as a
pathology, that FM loss stays "largely unchanged" while sampling efficiency improves.
**Action Matching** and **Energy Matching (NeurIPS 2025)** supply the potential
parameterisation of §6.6. Metric Flow Matching and AuxPath-FM change the probability *path*
itself, which is a different construction from changing the field at fixed path.

**Two entries we could not verify.** Our starting reading list included "FlowGuard:
Guarding Flow Matching via Conformal Sampling (2025)" and "High-Order Flow Matching:
Unified Framework and Sharp Statistical Rates (NeurIPS 2025)". Neither could be located;
`abs:"conformal" AND abs:"flow matching" AND abs:"guard"` returns zero results and
`ti:"High-Order Flow Matching"` returns zero. We flag them as unverified rather than
substitute a plausible neighbour.

---

## 8. Limitations

Full list in `paper/limitations.md`. The ones that should most change a reader's confidence:

1. **Two dimensions only, and the dimension dependence is the whole question.** A
   dimension-counting heuristic suggests the flux-null subspace occupies a fraction
   `(d−1)/d` of the field's degrees of freedom, so the *room* for invisible error grows
   with `d` and is ≈ 1 for image models **[conjecture]**. We did not test it. Our 2-D
   finding could strengthen, weaken, or reverse. `κ` was built to make that test cheap;
   running it on an image model is the single most important follow-up.
2. **The mean effect is small.** `M_{\text{null}} = 0.27 %` of the data radius; `C_G` median
   1.22. What looks consequential is the tail and the discreteness of mode flips, not the
   average.
3. **Narrow scope for curl-freeness.** Gaussian source, independent coupling, linear
   interpolant. Minibatch-OT couplings, heavy-tailed sources and non-linear interpolants
   all break it — and in those regimes the flux-null subspace is *no longer pure error*,
   so our central reframing does not apply.
4. **Our pre-registered null was degenerate** and the null we interpret was chosen post hoc
   (§6.3).
5. **Non-additivity** forbids any error-budget reading of our numbers (§6.4).
6. **The E3 comparison is confounded** and matched on steps, not compute (§6.6).
7. **Infinite-data regime only**; the memorisation regime is untouched.
8. Deterministic ODE sampling only; one data family for the trained-network measurements.
9. **No mechanism.** We show suppression; we cannot explain it.

---

## 9. Reproducibility

CPU only — 2 cores, no GPU; total compute under 4 hours. Python 3.13, torch 2.14.1+cpu.

```bash
python3.13 src/e0_validate.py                                        # 25-check gate, ~6 min
python3.13 src/e1_main.py     --tag main --n-seeds 8 --steps 20000    # ~55 min
python3.13 src/e2b_reinject.py --src e1_main --tag main --seeds 0 1 2 --kl-steps 100
python3.13 src/e3_curlfree.py --tag main --n-seeds 4 --steps 20000    # ~40 min
python3.13 src/figures.py
```

Raw outputs (`results/raw/`) are kept separate from generated summaries
(`results/figures/`) and are never overwritten; the superseded E2a run is retained.
Per-experiment configs, seeds, code revisions and result paths are in
`research/experiment_registry.csv`; the full decision trail, including the hypotheses we
abandoned and why, is in `research/hypothesis_ledger.md` and `research/lab_notebook.md`.

---

## 10. Conclusion

The continuity equation leaves a large marginal-preserving subspace of velocity fields, and
it is natural to suspect learned flow-matching models of drifting into it. For the standard
setting this suspicion is doubly wrong: the training objective has a unique minimiser, and
that minimiser is curl-free, so the subspace holds only error. Measuring that error's
geometry, we find trained networks suppress circulation about 5× in energy and 26× in local
rotationality relative to chance — a negative result for the hypothesis, and mild good news
for practice. What remains is small on average but exactly invisible to distributional
metrics and concentrated where it is hardest to see: a heavy tail of badly-routed
particles, on the low-density separatrices between modes, changing which sample lands in
which mode without changing how many land there. For anyone using a flow-matching model as
a transport *map* rather than a sampler, that is the error that sample-quality metrics
cannot report, and `κ` is a cheap way to look for it.
