# Outline

**Working title:** *Learned Flow-Matching Velocity Fields Are Nearly Curl-Free:
Measuring Spurious Circulation and Its Distributionally Invisible Transport Cost*

**One-sentence thesis.** The flow-matching target field is exactly curl-free, so any
rotational component of a learned field is spurious and exactly invisible to every
distributional metric; we measure how much trained networks have (far less than chance —
falsifying the natural hypothesis), show that the part they do have re-routes samples
without changing the sample distribution, and give a dimension-scalable diagnostic for it.

**Intended venue/length.** Short empirical paper (~8 pages) or workshop paper. The
contribution is a measurement plus a diagnostic, not a new method or new theory. If the
reviewer asks "what is new?", the answer must be: *the numbers and the instrument*.

---

## 1. Introduction
* The continuity equation under-determines the velocity field — a large
  marginal-preserving subspace exists [A1]. It is natural to infer that learned
  flow-matching models might differ consequentially along it while scoring equally.
* **Two corrections that gate the whole question** (and that we state up front because
  they are easy to get wrong):
  1. The CFM objective is a least-squares regression with a *unique* minimiser [A3], so
     there is no gauge freedom to exploit at the optimum. Freedom can only live in the
     residual.
  2. For a Gaussian source, independent coupling and a linear interpolant, that minimiser
     is itself a **gradient field** [A4, B1]. So the entire flux-null subspace contains
     only error, with known correct value zero.
* Therefore the well-posed question is empirical: *how much spurious circulation do
  trained networks carry, and what does it cost?*
* Contributions: (i) an exactly-orthogonal estimator of the flux-null energy fraction `f`
  and a dimension-scalable Jacobian-asymmetry diagnostic `κ`, both with known correct
  value 0 on the true field; (ii) the measurement — `f = 0.140`, `κ = 0.030`, far **below**
  a matched null, falsifying the natural hypothesis; (iii) the consequence — the measured
  component re-routes 0.31 % of samples with no detectable distributional signature;
  (iv) a controlled curl-free-parameterisation comparison, reported with the confound.

## 2. Setup and background
* CFM, linear interpolant, independent coupling. Notation.
* Flux-null space and the weighted Helmholtz decomposition [A1]. Fig: schematic.
* The five distinctions that must not be collapsed (`theory.md` §2.5) — short table.
* Excess-loss identity `L − L_floor = η²` [B2] and the along-trajectory identity [B3].

## 3. The target is curl-free, so the null space is pure error
* Lemma 1 (**attributed**, [A4]/[B1]): `v* = ∇[‖x‖²/2t + ((1−t)/t)log p_t]`, `f(v*) = 0`.
  Verified numerically to `1e-14`.
* Scope: Gaussian source is required; clarifies where Lim 2025 Prop. 5.3's genericity
  remark applies (general latents) and where it does not (standard CFM).
* Corollary [B4]: `r(e) = r(v_θ)`. Spurious circulation, known correct value 0.
* Proposition 2 [B5]: exact blindness of all distributional functionals; non-blindness of
  the flow map.

## 4. Two instruments
* `f`: finite-volume face-weighted decomposition, orthogonal to machine precision, so
  `f ∈ [0,1]` by construction [B7]. Validation table (E0/V6–V9).
* `κ = ‖J−Jᵀ‖_F/‖J+Jᵀ‖_F` [B8]: grid-free, reference-free, `O(1)` per probe in any
  dimension via one JVP + one VJP. `κ(v*) = 0`; Hutchinson agrees with exact to 4 %.
* Why the usual instruments fail here: `W₂` at `n = 4000` in 2-D cannot resolve the
  ground-truth map [C13]; energy distance is within one SE of zero. CNF KL against the
  closed-form `log p_1` resolves `1e-7` and reads exactly 0 on the true field.

## 5. Testbed and validation
* Gaussian-mixture data + Gaussian base ⟹ closed-form `p_t`, `v*`, loss floor [B6], and a
  tight ground-truth flow map. **Fresh-sample training stream**, so the target really is
  the population `v*` and memorisation is not a confound.
* E0 validation gate: 25 checks, all pass. Continuity residual converges at observed order
  1.99–2.00. Report this table — the paper's claims rest on it.

## 6. Results
* **6.1 H4 first: not a solver artefact.** 0.01 % change over four orders of `rtol` [C6].
* **6.2 The measurement.** `f = 0.1402 ± 0.0319`; `κ = 0.0304 ± 0.0041`. Fig 1.
* **6.3 Falsification of the natural hypothesis.** Matched null `f₀ = 0.720 ± 0.037`,
  `p = 1.3e-14`. Networks *suppress* circulation ~5× in energy, ~26× in `κ`. Includes the
  **pre-registration failure** of our own `null_init` and why we do not count it [C3].
* **6.4 The consequence.** Injecting the measured component: `|KL| ≤ 0.007` vs the model's
  own 0.018, `M = 0.0106`, 0.31 % mode flips at unchanged occupancy, worst particle 42 % of
  the data radius. Figs 4, 5. Includes the non-additivity caveat [C11].
* **6.5 Tail and localisation.** Fig 3 (heavy tail), Fig 2 + Fig 5 (separatrix).
* **6.6 Curl-free parameterisation.** E3, reported *with* the confound that the potential
  net is also a better regressor; the `M/η` control.
* **6.7 A ground-truth-free surrogate** and its measured bias [C12].

## 7. Related work
Lim 2025 (closest; theory only, trains no networks, states this as future work);
Huang et al. 2025 (divergence alignment — distributional metrics only); Li et al. 2026
(trajectory-blindness of FID, no decomposition); Wan et al. 2025; Action/Energy Matching
(the architecture); Vaitl & Klein 2025 (path gradients differentiate through the map, so
a downstream consumer this would damage).

## 8. Limitations
See `paper/limitations.md`. Dimension is the headline limitation and the headline open
question.

## 9. Reproducibility
Exact commands, seeds, code revisions, wall-clock, and the raw/summary separation.

---

## Figures
| # | Content | File | Status |
|---|---|---|---|
| 1 | `f` per seed vs three nulls, with `f(v*) = 0` reference | `fig1_fluxnull_fraction.png` | done |
| 2 | Spatial map of the local flux-null share at `t = 0.2/0.45/0.7` | `fig2_spatial.png` | pending |
| 3 | CCDF of per-particle map displacement (theta/grad/null) | `fig3_tail.png` | done |
| 4 | Map error vs KL for ground truth / `r` only / full / `∇φ` only | `fig4_blindness.png` | done |
| 5 | Which samples get re-routed, in source and endpoint space | `fig5_modeflip.png` | done |
| 6 | E3: `f`, `η`, `M`, `KL` by architecture | pending | pending |

## Decision rule for whether to publish at all
If E3 shows no map-fidelity benefit **and** the E2b effect is judged too small to matter,
the honest output is a negative-result note, not a paper. As of E2b the positive content
is: a falsification of a natural and literature-invited hypothesis, a validated
dimension-scalable diagnostic, and a small-but-precisely-measured invisible transport
cost. That is a short paper, not a strong one, and the draft must not oversell it.
