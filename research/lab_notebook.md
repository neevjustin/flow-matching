# Lab Notebook

Append-only. One entry per experiment. Raw outputs live under `results/raw/<exp_id>/`
and are never overwritten; summaries and figures are generated separately.

Environment (fixed for all entries unless stated): Linux aarch64, 2 CPU cores, 7 GB RAM,
**no GPU**. Python 3.13.0, torch 2.14.1+cpu, numpy 2.5.4, scipy 1.18.1, POT 0.9.7.
All ground-truth quantities in float64; networks trained in float32 and cast to float64
for every measurement.

---

## E0 — Validation gate for the whole testbed

**Date** 2026-10-24
**Code** `src/gmm_path.py`, `src/helmholtz.py`, `src/ode.py`, `src/e0_validate.py`
**Command** `python3.13 src/e0_validate.py`
**Raw** `results/raw/e0/e0_report.json`
**Hypothesis** None — this is infrastructure verification. Nothing downstream may be
believed until E0 passes.

**Prediction (written first).** All closed forms in `theory.md` §4 should match Monte
Carlo; the Helmholtz solver should return `f = 0` on a pure gradient field, `f = 1` on a
pure flux-null field, and `f = 0.5` on an equal-energy mixture; an analytic flux-null
perturbation of `v*` should preserve the endpoint distribution while moving the map.

**Result — 25/25 checks pass.**

| Check | Outcome |
|---|---|
| V1 grid quadrature of `p_t` | `∫p_t = 1.00000000` at `t ∈ {0,¼,½,¾,1}` |
| V2 `v*` is `E[u|x_t]` | residual ⟂ test functions {1, x, x², sin}; worst z-score 3.16 over 4×10⁵ MC samples |
| V3 closed-form loss floor | matches MC to rel. 6×10⁻⁶ … 3.5×10⁻³ |
| V4 continuity equation | L² rel. residual converges at **observed order 1.99–2.00** (design 2): 2.3×10⁻² → 5.8×10⁻³ → 1.5×10⁻³ |
| V5 `v*` transports `p_0→p_1` | `W₂ = 0.269` vs same-`n` noise floor `0.329` |
| V6 pure gradient input | `f = 0.00000`, orthogonality residual `4×10⁻¹⁰` |
| V7 pure flux-null input | `f = 1.00000`, orthogonality residual `0` |
| V8 equal-energy mixture | `f = 0.50000` (expected 0.500) |
| V9 grid refinement | `f(192)` vs `f(256)` differ by `1.0×10⁻⁷`; `tol_density` 10⁻⁶→10⁻¹⁰ spread `6.6×10⁻⁷` |
| V10 **mechanism existence** | analytic flux-null perturbation of amplitude **56 % of ‖v*‖**: `W₂` stays at the noise floor (0.319 vs floor 0.329) while the map moves **1.066 = 26.6 % of the data radius** |

**Two bugs found and fixed, both in the *tests*, not the measured objects.**
1. V2's z-score used `std(g)` as a scale, which is identically 0 for the constant test
   function. Replaced with the per-entry standard error of the product.
2. V4's finite-volume divergence had the face-orientation signs reversed (`A` and `B`
   swapped), which produced a relative residual of exactly 1.000 — the signature of
   computing `∂_t p − div` instead of `∂_t p + div`. Fixed; residual then converged at
   second order.
   V3 passing at 10⁻⁶ while V2/V4 "failed" was the clue that `v*` itself was correct.

**Interpretation.** The testbed is sound. V10 is the empirical existence proof, inside our
own testbed, of the phenomenon the project is about: a large velocity perturbation that is
exactly invisible to the endpoint distribution and highly visible in the transport map.
Note this is a **constructed analytic** perturbation — it says nothing yet about whether
trained networks produce such components. That is E1.

**Next action.** E1.

---

## E1 — Pre-registered decisive experiment (H1 primary, H2 secondary, H4 gate)

**Date** 2026-10-24
**Code rev** `3b7110e` (+ `src/e1_main.py`, `src/evaluate.py`, `src/model.py`)
**Command** `python3.13 src/e1_main.py --tag main --n-seeds 8 --steps 20000 --width 128`
**Raw** `results/raw/e1_main/`
**Hypothesis** H1 (primary), H2 (secondary), H4 (gate).

**Prediction (written before the run, see `hypothesis_ledger.md` §3).** H1: `f − f₀ > 0.05`
with Welch `p < 0.01`. Stated prior that H1 survives: ~40 %; the modal alternative is
`f ≈ f₀`.

**Configuration.** `ring8` (8 equal Gaussians, radius 4, σ = 0.3), `p_0 = N(0,I)`,
independent coupling, linear interpolant. MLP width 128, depth 3, SiLU, 16-frequency fixed
Fourier time embedding, 37 890 parameters. Adam, lr 2e-3 cosine-decayed, batch 2048,
20 000 steps, **fresh i.i.d. sample every step** (no finite training set — so no
memorisation confound). 8 seeds. Decomposition on a 192² grid over `[−8,8]²`, 25 time
slices on `[0.02, 0.98]`, `tol_density = 10⁻⁸`.

**Status at time of writing:** running. Partial results and the verdict are recorded in the
entry below once complete.

**Measured before training (ground-truth properties):**
* Gronwall amplification along the true flow: `C_G` median **1.22**, mean 3.47, p90 5.84.
  So the field-to-map amplification in this testbed is modest — map error is not
  predominantly an amplification artefact.
* `W₂` noise floor at `n = 4000` between two independent `p_1` samples: **0.3668**;
  `W₂(Φ*#p_0, p_1) = 0.3871`. **The ground-truth map's own measured `W₂` exceeds the
  two-sample floor, i.e. `W₂` at this `n` in 2-D cannot resolve the distributional error
  we care about.** Consequence: `W₂` is unusable as H2's denominator. Sharper instruments
  (energy distance, exact CNF KL against the closed-form `log p_1`) were implemented in
  `src/sharp_metrics.py` for E2. **This is a change of instrument, not of the H1 criterion
  or threshold** — H1's primary metric and criterion are untouched.

**RESULT — E1 complete (8 seeds).**

| quantity | value |
|---|---|
| `f` (flux-null fraction of velocity error) | **0.1402 ± 0.0319** |
| `η_rel = ‖e‖/‖v*‖` | 0.0164 ± 0.0008 |
| `M` (map error vs `Φ*`) | 0.0217 ± 0.0023 |
| `W₂` to data | 0.3838 ± 0.0049 (floor 0.3668, `Φ*` itself 0.3871) |
| grid stability `|f(192)−f(256)|` | 3.0×10⁻⁵ (threshold 0.01) ✓ |

Nulls: `null_init` 0.0050 ± 0.0087 · `null_rot` 0.2737 ± 0.1937 · `null_swap` **0.7201 ± 0.0367**.

**H4 GATE: PASSED decisively.** `M` and the between-seed map disagreement each change by
**0.01 %** between `rtol = 10⁻⁵` and `10⁻⁹` (`M` = 0.01966 → 0.01966). The effect is
categorically *not* solver-induced.

**H1 VERDICT: FALSIFIED, in the direction opposite to the hypothesis.**
By the letter of the pre-registration, H1 "passes" against the pre-registered null
(`f − f₀ = +0.135`, `p = 2.7×10⁻⁶`). **But the pre-registered null is degenerate, and we
report it as a pre-registration failure rather than a success.** Post-hoc diagnosis,
quantified: for a randomly-initialised net of this architecture the *constant* (bias)
component carries **89–99.5 %** of the field's energy (`|E[v]| ≈ 0.04–0.13` while
`std_x(v) ≈ 0.004–0.011`). A constant field is a trivial gradient, so `null_init` measures
"what fraction of a near-constant field is rotational" ≈ 0. It does not instantiate the
"unstructured field of matched smoothness" the null was meant to supply.
Against the two **properly matched** nulls — which preserve the measured residual's own
magnitude field, spectrum and energy exactly and randomise only its direction relative to
the local weighted-gradient geometry — `f` is far **below** chance:
`0.140` vs `0.720 ± 0.037` (`p = 1.3×10⁻¹⁴`).

**Independent, grid-free confirmation.** Since `v*` is a gradient, its Jacobian is
symmetric, so `κ = ‖J−Jᵀ‖_F/‖J+Jᵀ‖_F` measures spurious circulation with known correct
value 0. Measured: `κ(v*) = 0.0` (machine zero), **trained `κ = 0.0304 ± 0.0041`**,
random-init `κ = 0.803 ± 0.336`. Trained nets are ~26× less rotational than a random
field. A Hutchinson estimator of the same quantity (one JVP + one VJP per probe, cost
independent of dimension) agrees with the exact value to 4 % (0.0272 vs 0.0282;
0.0352 vs 0.0360) — so the diagnostic is dimension-scalable.

**Per-particle structure (heavy tail).** `M` is a mean and hides the tail: seed 0 vs truth
gives mean 0.0198, **median 0.0090, p90 0.025, p99 0.171, max 1.94** (= 48 % of the data
radius); RMS/mean = 4.4 between seeds.

**Mode-assignment flips.** 0.40–0.65 % of particles land in a *different* mode than the
true map, while the mode-*occupancy* TV is only 0.0010–0.0037 — i.e. individual samples
are misrouted 2–4× more than the distribution over modes changes.

**Ground-truth-free surrogate.** `f_Δ` from `v_A − v_B` = 0.2069 ± 0.0270, vs the true
`f = 0.140`: biased **up by ~48 %**. Consistent with the independence assumption of
`theory.md` §7 being false in the expected direction — the *shared, systematic* error is
more gradient-like than the seed-idiosyncratic error, and differencing removes the shared
part.

**H2 from E1: inconclusive, by design failure of the instrument.** `R = M/(W₂−floor) = 1.28`,
not ≥ 3. But `W₂(Φ*#p₀,p₁) = 0.3871 > ` the two-sample floor 0.3668, so `W₂` at `n = 4000`
in 2-D cannot resolve even the ground-truth map. The pre-registered H2 threshold is
therefore **not evaluable with this instrument** and is carried to E2b with a sharp one.
We do not claim H2 either way from E1.

**Next action.** E2b with the KL instrument.

---

## E2a — re-injecting measured components (SUPERSEDED, kept on record)

**Command** `python3.13 src/e2_reinject.py --src e1_main --tag main --seeds 0 1 2`
**Raw** `results/raw/e2_main/`
**Outcome: P3 FAILED.** Reconstructing `v* + interp(∇φ) + interp(r)` reproduced `Φ_θ` only
to 28–35 % of `M_θ` (pointwise error 0.005–0.007 vs `M_θ ≈ 0.019`), i.e. the bilinear
interpolation of the face-sampled components was as large as the quantity of interest
(`M_null ≈ 0.009`). **Not used for any claim.** Recorded because the failure motivated the
redesign and because the directional result it gave (`M_null ≈ 0.009`, distributionally
null) was later confirmed by the corrected experiment.

Also found here: the **energy distance is not a usable instrument** at `n = 3000` —
`ED(floor) = −1.5×10⁻³ ± 3.1×10⁻³`, `ED(Φ*) = −2.1×10⁻³ ± 3.0×10⁻³`. All values are within
one standard error of zero and of each other; the negative signs are estimator noise
(the estimator is unbiased for a quantity that is ≈ 0 here). Replaced by CNF KL.

---

## E2b — re-injecting measured components, exactly consistent split

**Command** `python3.13 src/e2b_reinject.py --src e1_main --tag main --seeds 0 1 2 --kl-steps 100`
**Raw** `results/raw/e2b_main/`  **Hypothesis** H2 (and the P1/P2/P4 predictions).

**Redesign.** Only the *scalar potential* `φ` is interpolated (bicubic spline, analytic
derivatives); the residual is then `r = e − ∇φ_spline` with `e = v_θ − v*` evaluated
**exactly** at every query point. Hence `∇φ + r = e` identically and the E2a
reconstruction failure is removed *by construction*. `N = 256`, 33 time slices.

**Instrument validation (both pre-conditions met).**
* **P4** — MC flux-null fraction of the spline split vs the grid value:
  0.1326/0.1296, 0.1796/0.1778, 0.1013/0.1000. All within 0.003 (threshold 0.03). ✓
* **KL instrument** — integrating the *true* field must give KL = 0 exactly. Measured with
  fixed-step RK4 and analytic divergences: **9.2×10⁻⁸ (50 steps), 7.0×10⁻⁸ (100),
  6.6×10⁻⁹ (200)**. So the instrument resolves ~10⁻⁷ nats — five orders of magnitude
  sharper than the energy distance. (Adaptive solvers had to be abandoned: with a
  finite-difference divergence the RHS is non-smooth and `nfev` blew up / the run stalled.
  Divergences are now analytic: `div(v*+∇φ) = div v* + Δφ`, `div(v*+e−∇φ) = div v_θ − Δφ`.)

**RESULT (3 seeds).**

| field injected into `v*` | `M` | p99 | max | mode-flip | occupancy TV | `KL(p̂₁‖p₁)` |
|---|---|---|---|---|---|---|
| — (`Φ*`, ground truth) | 0 | — | — | 0 | 0 | **0.00000 ± 0.00000** |
| **flux-null part `r` only** | **0.0106 ± 0.0004** | 0.060–0.077 | 1.39–1.70 | **0.31 % ± 0.04 %** | 0.0013–0.0017 | **−0.0046 ± 0.0021** |
| full error `e` (= `v_θ`) | 0.0205 ± 0.0024 | 0.155–0.326 | 1.51–1.62 | 0.53 % | 0.0010–0.0037 | 0.0183 ± 0.0035 |
| gradient part `∇φ` only | 0.0188 ± 0.0021 | 0.135–0.302 | 1.48–1.53 | 0.50 % | 0.0020–0.0033 | 0.1479 ± 0.0247 |

**P1 CONFIRMED, and this is the project's central positive result.** Injecting the
*measured* flux-null component of a *trained* network's error into the true field leaves
the endpoint distribution indistinguishable from ground truth (`|KL| ≤ 0.0069`, within
~2σ of 0, and 4× smaller than the full model's own `KL = 0.018`) while displacing the
transport map by `M = 0.0106`, moving the worst particles by up to **1.70 ≈ 42 % of the
data radius**, and re-routing **0.31 % of samples to a different mode** with essentially
no change in mode occupancy.

**Caveat recorded honestly.** `KL_null` is negative in all three seeds (−0.0031, −0.0036,
−0.0069). True KL is ≥ 0, so this indicates a small systematic negative bias we cannot
fully attribute (candidates: the spline split is exactly flux-null only in the discrete
face-weighted sense, and linear interpolation in `t`). It does not affect the claim, which
is `|KL_null| ≪ KL_θ`, but it means we may not claim `KL_null = 0` exactly — only that it
is at or below our resolution of the distributional effect.

**P2 REFUTED, and the reason matters.** `Φ_∇φ` has `KL = 0.148`, **8× LARGER** than the
full model's 0.018. So the gradient part alone is distributionally *much worse* than the
full error, and adding `r` back repairs most of it. This is **not** a contradiction:
`r` is flux-null with respect to `p_t`, which `v*` generates — it is *not* flux-null with
respect to the different path that `v* + ∇φ` generates. The components therefore do **not**
act additively on the marginals at finite amplitude. **Consequence for interpretation: we
may NOT say "half of the model's map error is invisible."** The defensible claim is the
one P1 states — that this measured component, injected into the true field, is
distributionally invisible and geometrically active.

**Localisation (fig 5).** The re-routed particles cluster near the origin in the source and
in the **inter-modal gaps** at the endpoints — the separatrix/low-density region predicted
by the H1 mechanism, even though H1's aggregate prediction failed.

**Next action.** E3 (curl-free parameterisation), the only remaining test of whether the
measured circulation is actionable.
