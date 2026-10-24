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
