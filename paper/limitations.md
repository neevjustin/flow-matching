# Limitations

Ordered by how much they should change a reader's confidence. The first three are, in our
judgement, severe enough that the paper must not be read as establishing anything about
practical generative modelling.

---

## 1. Two dimensions only. The dimension dependence is untested and is the whole question.

Every experiment is 2-D. The dimension-counting heuristic (`theory.md` §2.3) suggests the
flux-null subspace occupies a fraction `(d−1)/d` of the velocity-field degrees of freedom,
so the *room* for distributionally invisible error grows with dimension and is ≈ 1 for
image models. **We did not test this.** It is a heuristic about an unweighted decomposition
and says nothing about where a trained network's error actually lands — which is precisely
the quantity we found to be far from chance in 2-D. The 2-D finding (networks suppress
circulation) could become stronger, weaker, or reverse in high dimension. Anyone citing
this work as evidence about image models would be over-reading it.

The `κ` diagnostic was built to make that test cheap (one JVP + one VJP per probe,
cost independent of `d`), and we validated it against the exact decomposition in 2-D. We
simply did not have the compute to run it on an image model. This is the single most
important follow-up.

## 2. The effect size is small in the mean.

Injecting the measured flux-null component displaces the transport map by `M = 0.0106`,
which is **0.27 % of the data radius**, and re-routes 0.31 % of samples. The Gronwall
amplification in this testbed is modest (`C_G` median 1.22). What looks consequential is
the *tail* (worst particle 42 % of the data radius; per-particle p99/median ≈ 19) and the
discreteness of the mode-flip failure — not the average. A reader who cares about average
sample quality should conclude this effect is negligible in 2-D.

## 3. Scope of the curl-freeness result, and of the whole framing.

Lemma 1 requires a **Gaussian source**, **independent coupling**, and a **linear
interpolant**. All three are standard but all three are often violated in practice:
* minibatch-OT / rectified couplings are *not* independent, and Tweedie's argument fails;
* non-Gaussian sources (e.g. the Student-`t` sources studied by Lim 2025) break it, which
  is exactly the regime where that paper's generic-non-gradient result bites;
* non-linear interpolants and general `(m_t, σ_t)` schedules are untested here.

Outside this scope, `f(v*) ≠ 0`, the flux-null subspace is **no longer pure error**, and
our central reframing does not apply. We have not checked how large `f(v*)` is in any of
those regimes.

## 4. Our pre-registered null model was degenerate, and we only discovered it afterwards.

The pre-registered null (randomly-initialised networks of the same architecture) turned out
to measure the wrong thing: the constant/bias component carries 89–99.5 % of such a field's
energy, and a constant field is a trivial gradient, so `f₀ ≈ 0.005`. By the letter of our
pre-registration H1 therefore "passed"; we report it as **falsified** on the basis of two
better-matched nulls. That is the right scientific call but it is a *post-hoc* choice of
null, and a sceptical reader should discount accordingly. The lesson — validate the null
for non-degeneracy before the decisive run, as we did for the measurement instruments — is
one we learned the hard way here.

## 5. The two error components do not act additively, which limits the attribution.

`KL(Φ_∇φ) = 0.148` is 8× the full model's `KL = 0.018`: removing `r` from the model's error
makes the marginals *worse*, because `r` is flux-null with respect to `p_t`, not with
respect to the path that `v* + ∇φ` generates. Consequently we can say "injecting the
measured `r` into the true field is distributionally invisible and geometrically active",
but we **cannot** say "a fraction `f` of the model's map error is invisible". Any
decomposition-of-error-budget reading of our numbers is unsupported.

## 6. The E3 curl-free comparison is confounded.

The potential-parameterised network does not merely have `f ≈ 0`; it is also a **better
velocity regressor** (`η_rel` lower by ~27 %). So its map-error advantage is not a clean
isolation of "removing circulation": most of it is explained by lower velocity error. We
report the `M/η` ratio as a partial control, but a clean test would hold `η` fixed across
architectures (e.g. by capacity or early-stopping matching), which we did not do. The
potential arm also costs ~2× the wall-clock per step; the comparison is matched on steps,
not on compute.

## 7. Ground truth is the *population* field, which is the right choice but a restrictive one.

We train on a fresh i.i.d. sample every step so that `v*` really is the regression target.
This deliberately removes the memorisation/generalisation confound, but it also means our
numbers describe the **infinite-data** regime. In the finite-data regime the exact
minimiser memorises (`theory.md` §6), `v̂*` has a `1/(1−t)` singularity, and `f` would have
to be interpreted against a different reference. We did not run that regime.

## 8. Discretisation and estimator caveats.

* `f` is computed on a grid; the decomposition is orthogonal to machine precision by
  construction, but the discrete gradient space is a *subspace* of the continuum one, so
  `f` is biased slightly **upward**. Refinement 192 → 256 changes it by `3e-5`, so the bias
  has converged at our resolution; we did not prove a bound.
* `KL_null` came out slightly **negative** in all three seeds (−0.003 to −0.007). True KL
  is ≥ 0, so there is a small systematic negative bias we could not fully attribute
  (candidates: the spline split is exactly flux-null only in the discrete face-weighted
  sense; linear interpolation in `t`). We therefore claim only `|KL_null| ≤ 0.007`, not
  `KL_null = 0`.
* `W₂` and the energy distance are reported only to document that they are **unusable** at
  these sample sizes; no claim rests on them.
* All results are for one data family (Gaussian mixtures with varying geometry). We built
  but did not fully exploit the imbalanced / anisotropic / moons variants; `f(v*) = 0` was
  checked on all four, but the trained-network measurement is `ring8` only.

## 9. No high-order or stochastic samplers; no conditional generation.

Deterministic ODE sampling only. SDE samplers, guidance, distillation, few-step samplers,
and conditional models are all out of scope, and the flux-null framing may not transfer to
them (for an SDE the relevant null space is different).

## 10. Mechanism unexplained.

We show that networks suppress circulation but cannot say *why*. The CFM loss penalises
`‖∇φ‖²` and `‖r‖²` **equally** (they are orthogonal in `L²(p_t)`), so the asymmetry must
come from the function class or the optimiser's implicit bias — neither of which we
characterise. This is stated as an open problem, not an explanation.
