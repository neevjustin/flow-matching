# STATUS

Last updated: 2026-10-24, during E1.

## Current research question

Original framing (from the brief) was **corrected twice**, both times because of what the
literature and the mathematics actually say. Current question:

> For a Gaussian base, independent coupling and a linear interpolant, the CFM target field
> `v*` is **exactly curl-free** (`f(v*) = 0`), so the entire marginal-preserving (flux-null)
> subspace can contain only error. **How much spurious circulation do trained CFM networks
> contain, is it more or less than an unstructured field of the same smoothness, and what
> does it cost in transport-map fidelity — given that it is exactly invisible to every
> distributional metric?**

### Why the question moved (recorded so the trail is auditable)

1. **Correction 1 (literature + algebra).** The brief's premise was that learned models
   might "exploit" the velocity-field freedom. But the CFM loss is a least-squares
   regression whose minimiser is the unique conditional expectation, so **the objective has
   no gauge freedom**. Non-uniqueness of the continuity equation is not non-uniqueness of
   the loss. The freedom can only live in the residual `e = v_θ − v*`.
2. **Correction 2 (our measurement, then traced to prior work).** We measured
   `f(v*) = 0` to 8–29 significant figures on four datasets and derived
   `v* = ∇[‖x‖²/(2t) + ((1−t)/t) log p_t]` via Tweedie, verified to `10⁻¹⁴`. **This is
   already in the closest prior work** (Lim 2025, Example 5.1) and must not be claimed.
   Its consequence, however, sharpens the question: a learned flux-null component is not an
   alternative valid field, it is *spurious circulation with a known correct value of zero*.

## Strongest evidence so far

* **E0 (25/25 checks pass).** Testbed fully validated: closed-form `p_t`, `v*`, and loss
  floor verified against Monte Carlo; continuity equation satisfied at observed order
  1.99–2.00; Helmholtz solver exact (returns `f = 0`, `1`, `0.5` on constructed fields,
  closure residual `~10⁻¹⁶`).
* **E0/V10 — mechanism exists.** A constructed flux-null perturbation of amplitude **56 % of
  ‖v*‖** leaves `W₂` at the sampling-noise floor while moving the transport map by
  **26.6 % of the data radius**. Proof-of-concept that the blindness is real, *for
  constructed fields*.
* **E1 (5 of 8 seeds).** Trained CFM networks do contain spurious circulation:
  `f = 0.133 ± 0.031` (seeds: 0.127, 0.176, 0.092, 0.139, 0.130), i.e. **~13 % of the
  held-out velocity-error energy is in the flux-null subspace and is structurally
  undetectable by any distributional metric**. Highly reproducible:
  `η_rel = 0.0167 ± 0.0009`, `M = 0.0210 ± 0.0017`.
* Grid stability of the primary metric is excellent: `f(192)` and `f(256)` agree to
  6 decimal places.
* Gronwall amplification along the true flow is modest: `C_G` median 1.22, p90 5.84 — so
  map error is **not** predominantly an amplification artefact.

## Main unresolved issues

1. **The null level `f₀` is not yet measured** (running). Until it is, we cannot say whether
   `f ≈ 0.13` is *more* or *less* circulation than chance. Current expectation: `f < f₀`,
   which would mean networks **partially inherit** the target's curl-free structure — the
   opposite of the brief's hypothesis, and a clean falsification of pre-registered H1.
2. **Effect size may be practically small.** `f ≈ 0.13` and `C_G ≈ 1.2` imply the invisible
   part of the map error is `~0.007` in absolute units, i.e. `~0.2 %` of the data radius.
   Honest reporting requires saying so. Two things could still make it matter and are
   being measured: the **per-particle tail** (the mean hides outliers) and the
   **mode-assignment flip rate** (a discrete, interpretable sample-level failure that is
   invisible to mode-occupancy statistics by construction).
3. **`W₂` is unusable as H2's denominator.** At `n = 4000` in 2-D its two-sample noise floor
   is 0.367 while `W₂(Φ*#p₀, p₁) = 0.387` — the ground-truth map itself cannot be
   distinguished from a finite sample. Replaced by energy distance and exact CNF KL
   (`src/sharp_metrics.py`). H1's metric and threshold are untouched.
4. **Dimension.** The heuristic `f₀ ≈ (d−1)/d` predicts the blind fraction grows with
   dimension, which would be the reason to care beyond 2-D. Untestable on this budget;
   recorded as the headline open question.

## Experiments completed

| ID | Purpose | Status |
|---|---|---|
| E0 | validation gate (25 checks) | **complete, all pass** |
| E1 | pre-registered decisive run: `f` vs `f₀`, H2, H4 gate | running (5/8 seeds) |
| E2 | re-inject *measured* Helmholtz components into the ODE | coded, queued |
| E3 | curl-free potential parameterisation vs standard MLP | coded, queued |

## Next experiment and exact command

```bash
# 1. finish E1 (in flight)
python3.13 src/e1_main.py --tag main --n-seeds 8 --steps 20000 --width 128

# 2. E2 -- does the MEASURED flux-null component move the map while leaving the
#    endpoint distribution at the noise floor?  (includes a Phi_recon validation
#    of the interpolation, without which E2 means nothing)
python3.13 src/e2_reinject.py --src e1_main --tag main --seeds 0 1 2

# 3. E3 -- does removing the circulation by construction buy map fidelity?
python3.13 src/e3_curlfree.py --tag main --n-seeds 4 --steps 20000
```

## Budget

* **Spent:** ~1.6 h wall-clock of the 2-core CPU. E0 ≈ 6 min; E1 ≈ 55 min (in flight);
  literature review ≈ 10 full-text paper reads.
* **Remaining plan:** E2 ≈ 15 min, E3 ≈ 50 min (potential arm costs ~3× per step),
  multi-dataset replication ≈ 40 min. Total ≈ 2 h more.
* No GPU used or needed. Largest memory footprint is the 192² sparse solve (~100 MB).

## Honest assessment of whether to continue

Continue, but with the contribution explicitly scoped **down** from the brief's framing.
Two of the brief's three conceptual ingredients turned out to be published already (the
flux-null subspace and the curl-freeness of `v*`, both in Lim 2025), and a third (the
potential parameterisation) exists as an architecture (Action Matching, Energy Matching).
What remains genuinely unmeasured — and is the only thing we may claim — is the
**empirical geometry of trained networks' velocity error and its transport-map
consequence**. Whether that clears a publication bar depends on E2/E3 effect sizes, which
are not yet known. If E2 shows the invisible map error is negligible and E3 shows no
improvement, the correct output is a **negative result**, and `paper/` should not be
started.
