# STATUS

Last updated: 2026-10-24, after E0 / E0b / E1 / E2a / E2b / E3. **All planned experiments
complete.** Paper drafted.

## Current research question (final form)

> For a Gaussian base, independent coupling and a linear interpolant the CFM target field
> `v*` is **exactly curl-free**, so the entire marginal-preserving (flux-null) subspace can
> contain only error. **How much spurious circulation do trained CFM networks carry, is it
> more or less than an unstructured field of the same smoothness, and what does it cost in
> transport-map fidelity given that it is exactly invisible to every distributional metric?**

The question moved twice from the brief's framing; both moves are documented in
`hypothesis_ledger.md` §0 and §5. In short: (1) the CFM objective has a *unique* minimiser,
so there is no gauge freedom to exploit — the freedom can only live in the residual; (2)
that minimiser is itself a gradient field (**prior work**: Lim 2025 Example 5.1), so the
null space is pure error with known correct value zero.

## Strongest evidence

1. **H1 FALSIFIED, in the direction opposite to the brief's hypothesis.**
   `f = 0.1402 ± 0.0319` (8 seeds) against a matched null `f₀ = 0.7201 ± 0.0367`
   (`p = 1.3×10⁻¹⁴`). Grid-free confirmation: `κ = 0.0304 ± 0.0041` vs `0.803 ± 0.336` at
   random init, with `κ(v*) = 0` exactly. **Trained networks suppress the
   marginal-preserving freedom ~5× in energy and ~26× in local rotationality, rather than
   exploiting it.**
2. **H4 gate passed decisively.** `M` changes by **0.01 %** between `rtol = 10⁻⁵` and
   `10⁻⁹`. Not a solver artefact.
3. **The residual circulation is invisible yet consequential (E2b, 3 seeds).** Injecting
   the *measured* flux-null component into the true field: `|KL| ≤ 0.007` (instrument
   validated to `7×10⁻⁸`, reads exactly 0 on the true field; full model's own KL = 0.018),
   while `M = 0.0106 ± 0.0004` (0.27 % of the data radius), worst particle `1.70`
   (42 % of it), and **0.31 % ± 0.04 % of samples re-routed to a different mode** at
   essentially unchanged mode occupancy.
4. **Localisation.** Spurious circulation forms radial spokes in the gaps *between* modes
   and is near zero on them (fig 2); re-routed particles sit on the separatrices (fig 5).
5. **Heavy tail.** Per-particle map error: median 0.009, p99 0.17, max 1.94; RMS/mean 4.4.
6. **E3 curl-free parameterisation** is better on every axis (`η` −27 %, `M` −39 %,
   `KL` −62 %) at 2.09× wall-clock — **but confounded**: controlling by `M/η`, the
   map-specific gain is only −16 % (`p = 0.064`), so ~58 % of it follows from the better
   velocity fit.
7. **E0 validation gate: 25/25.** Continuity residual converges at observed order
   1.99–2.00; decomposition orthogonal to `~10⁻¹⁶`; `f` recovers 0 / 1 / 0.5 exactly on
   constructed fields.

## Main unresolved issues

1. **Dimension — the headline open question.** The heuristic `f₀ ≈ (d−1)/d` suggests the
   room for invisible error grows with `d` and is ≈ 1 for image models, but this is
   untested and our 2-D finding could reverse. `κ` was built to make the test cheap
   (`O(1)` per probe in any `d`, validated against the exact decomposition); we lacked the
   compute to run it on an image model.
2. **No mechanism for the suppression.** The CFM loss penalises both components *equally*
   (orthogonality), so the asymmetry must come from the function class or optimiser.
   Unproved either way.
3. **Non-additivity** (`KL(Φ_∇φ) = 0.148`, 8× the full model's) forbids any
   error-budget attribution of map error to the two components.
4. **E3 is confounded** (see above); a clean test would match `η` across architectures.
5. **Small mean effect.** 0.27 % of the data radius. The tail and the discrete mode-flip
   failure are what look consequential, not the average.

## Experiments completed

| ID | Purpose | Status |
|---|---|---|
| E0 | validation gate, 25 checks | complete, all pass |
| E0b | `f(v*) = 0` across 4 datasets; `v* = ∇Ψ*` to `1e-14` | complete |
| E1 | pre-registered decisive run (H1, H2, H4 gate) | complete — **H1 falsified**, H4 passed |
| E2a | re-injection, bilinear interpolation | **superseded** (P3 failed), retained on record |
| E2b | re-injection, exactly-consistent spline split | complete — H2 supported in narrowed form |
| E3 | curl-free parameterisation (post hoc) | complete — works, confounded |

## Next experiment and exact command

No further experiment is warranted at this scale. The one that matters next requires a GPU:

```bash
# NOT RUN -- needs a GPU. Measure kappa on an image-scale flow-matching model
# using the dimension-scalable Hutchinson estimator in src/evaluate.py
# (curl_ratio_hutchinson), whose correct value is exactly 0.
```

Reproduce everything here on CPU:

```bash
python3.13 src/e0_validate.py
python3.13 src/e1_main.py      --tag main --n-seeds 8 --steps 20000
python3.13 src/e2b_reinject.py --src e1_main --tag main --seeds 0 1 2 --kl-steps 100
python3.13 src/e3_curlfree.py  --tag main --n-seeds 4 --steps 20000
python3.13 src/figures.py
```

## Budget

* **Spent:** ~3.5 h wall-clock on 2 CPU cores. E0 ≈ 6 min, E1 ≈ 50 min, E2a ≈ 8 min,
  E2b ≈ 20 min, E3 ≈ 40 min, figures ≈ 2 min; the rest is literature (10 papers read in
  full text) and analysis.
* **No GPU used.** Peak memory ~300 MB (the 256² sparse solve).
* **Remaining:** the project is at a natural stopping point. Scaling to image models is the
  next step and needs resources we do not have here.

## Honest assessment

The brief's hypothesis is **false** in the standard CFM setting, and we can say so with a
pre-registered test, a passed solver gate, a matched null, and two independent instruments.
Two of the brief's three conceptual ingredients turned out to be published already (the
flux-null subspace and the curl-freeness of `v*`, both in Lim 2025); the third (potential
parameterisation) exists as an architecture. **What is ours is the measurement and the
instrument**, plus the negative result.

That supports a short, honest paper (`paper/draft.md`) whose headline is a falsification
plus a small, precisely-quantified invisible transport cost and a cheap diagnostic. It is
not a strong paper, and the draft is written not to oversell it: the abstract states the
falsification first, the limitations lead with "two dimensions only", and
`paper/claims_and_evidence.md` lists every claim we explicitly do *not* make. Whether this
clears a conference bar is a judgement call for the reader; we would submit it to a
workshop rather than a main track.
