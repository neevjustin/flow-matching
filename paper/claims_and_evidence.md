# Claims and Evidence

Every claim the paper makes, with its status and the exact artefact that supports it.
Status codes:

* **[E]** established by prior work — cited, not claimed by us
* **[D]** derived here (elementary; proof in `research/theory.md`)
* **[M]** our measurement, with the file that produced it
* **[C]** conjecture — explicitly labelled as such in the paper
* **[N]** negative result / falsification of one of our own hypotheses

No claim in the paper may lack a row here.

---

## A. Background that is NOT ours

| # | Claim | Status | Source |
|---|---|---|---|
| A1 | The continuity equation does not determine the velocity field; marginal-preserving perturbations form the flux-null space `R_{p_t} = G_{p_t}^⊥` in `L²(p_t)`, and `v = ∇φ + r` is an orthogonal decomposition | **[E]** | Lim 2025 (arXiv:2512.16768) Prop. 5.4 and §5; classically Benamou–Brenier, Ambrosio–Gigli–Savaré |
| A2 | For Gaussian mixtures the flux-null family can be written explicitly via antisymmetric matrices | **[E]** | Lim 2025 Prop. 5.5 |
| A3 | The exact empirical CFM minimiser has a closed form and is unique `dt⊗p̂_t`-a.e. — the CFM objective has **no** gauge freedom | **[E]** | Lim 2025 Prop. 4.3 |
| A4 | For independent coupling the population minimiser is a **gradient field**, via `∇log p_t = (t/(1−t))v − z/(1−t)` | **[E]** | Lim 2025 **Example 5.1**; velocity–score relation standard in Lipman et al. 2022, Albergo & Vanden-Eijnden 2022 |
| A5 | The distributional error induced by a velocity error is driven by the weighted divergence `∇·(p_t e_t)` alone | **[E]** | Huang et al. ICML 2025 (arXiv:2602.00869) Prop. 3.1 |
| A6 | Distributional metrics such as FID are "trajectory-blind" | **[E]** | Li et al. ICML 2026 (arXiv:2602.07928) |
| A7 | Flux-equivalent fields can have very different kinetic-energy tails | **[E]** | Lim 2025 Remark 5.1 |
| A8 | Parameterising a generative velocity field as the gradient of a learned scalar is an existing architecture | **[E]** | Action Matching; Energy Matching (arXiv:2504.10612) |
| A9 | FM ODE well-posedness on `[0,1)` for finite second moment, incl. manifold support | **[E]** | Wan et al. ICML 2025 (arXiv:2412.18730) Thm 4.1 |

**A4 is the single most important attribution in the paper.** We independently rediscovered
it mid-project and must not present it as new. See `research/STATUS.md` "Correction 2".

## B. Derivations we present (elementary, for self-containedness)

| # | Claim | Status | Where |
|---|---|---|---|
| B1 | `v*(t,x) = (x + (1−t)∇log p_t(x))/t = ∇[‖x‖²/(2t) + ((1−t)/t)log p_t]`, hence `f(v*) = 0`; same argument applies to the **empirical** `v̂*` for a Gaussian source, so Lim 2025 Prop. 5.3's genericity remark (stated for general non-Gaussian latents) does not bite in the standard CFM setting | **[D]** + scope clarification of **[E]** | `theory.md` §2.4; verified numerically to `1e-14` (autograd of `Ψ*` vs closed-form `v*`, 4 datasets × 5 times) |
| B2 | Held-out excess CFM loss equals `η² = ‖e‖²_{dt⊗p_t}` exactly (cross term vanishes) | **[D]** | `theory.md` §3.1 |
| B3 | `E_{x_0}∫‖e(s,X*(s))‖²ds = η²`: the excess loss *is* the along-trajectory error energy | **[D]** | `theory.md` §3.3 |
| B4 | Since `f(v*) = 0`, the flux-null part of the **error** is the flux-null part of the **learned field**; it is spurious, with known correct value zero | **[D]** | `theory.md` §2.4 eq. (6'') |
| B5 | A flux-null `e` gives `p̂_t = p_t` **exactly**, so every functional of `p̂_1` is exactly blind to it, while `δ(1) = ∫J(1,s)e(s,X*(s))ds` is generically non-zero | **[D]** | `theory.md` §3.2 |
| B6 | Closed forms for `p_t`, `v*`, and the CFM loss floor for Gaussian-mixture data under a Gaussian base | **[D]** | `theory.md` §4; validated in E0/V1–V3 |
| B7 | A finite-volume face-weighted discretisation for which the decomposition is orthogonal to machine precision, so `f ∈ [0,1]` by construction | **[D]** | `theory.md` §5; validated E0/V6–V9, closure residual `~1e-16` |
| B8 | `κ = ‖J−Jᵀ‖_F/‖J+Jᵀ‖_F` measures spurious circulation with known correct value 0, needs no grid and no reference field, and is estimable in `O(1)` cost per probe in any dimension (one JVP + one VJP) | **[D]** | `src/evaluate.py`; `κ(v*) = 0.0` exactly; Hutchinson vs exact agree to 4 % |

## C. Our measurements

| # | Claim | Evidence | Artefact |
|---|---|---|---|
| C1 | Trained CFM nets have `f = 0.1402 ± 0.0319` (8 seeds) | 8 seeds, grid 192, 25 time slices; grid stability `3e-5` | `results/raw/e1_main/metrics.json` |
| C2 | This is far **below** chance: matched null `f₀ = 0.7201 ± 0.0367`, `p = 1.3e-14` | null preserves the residual's magnitude field, spectrum and energy; randomises direction only | same |
| C3 | **[N]** Our pre-registered null (`null_init`) was degenerate: the constant component of a random-init net carries 89–99.5 % of its energy, and a constant field is a trivial gradient | measured `\|E[v]\|` vs `std_x(v)` | lab notebook E1 |
| C4 | Grid-free confirmation: trained `κ = 0.0304 ± 0.0041` vs random-init `0.803 ± 0.336`; `κ(v*) = 0` | 8 seeds | `src/evaluate.py` run logged in notebook |
| C5 | The dimension-scalable Hutchinson estimator of `κ` agrees with the exact value to 4 % | 2 seeds | same |
| C6 | **H4 gate:** `M` and between-seed map disagreement change by 0.01 % between `rtol 1e-5` and `1e-9` — the effect is not solver-induced | 3 seeds × 3 tolerances | `results/raw/e1_main/metrics.json` → `gate` |
| C7 | Injecting the **measured** flux-null component into the true field: `\|KL\| ≤ 0.0069` (vs the model's own `KL = 0.018`; true field reads `0.00000`) while `M = 0.0106 ± 0.0004` | 3 seeds; KL instrument validated to `7e-8` | `results/raw/e2b_main/metrics.json` |
| C8 | …re-routing `0.31 % ± 0.04 %` of samples to a different mode at occupancy TV `0.0013–0.0017`, with worst-particle displacement up to `1.70` = 42 % of the data radius | 3 seeds, 3000 shared `x_0` | same; fig 5 |
| C9 | The per-particle map error is heavy-tailed: median 0.009, p99 0.17, max 1.94; RMS/mean = 4.4 | seed-level | fig 3 |
| C10 | The re-routed particles localise to the inter-modal separatrix / low-density region | fig 5, fig 2 | `results/figures/` |
| C11 | **[N]** The two components do **not** act additively on the marginals: `KL(Φ_∇φ) = 0.148`, 8× the full model's. Hence we may not attribute a fraction of the model's map error to `r` | 3 seeds | `results/raw/e2b_main/metrics.json` |
| C12 | A ground-truth-free surrogate `f_Δ` (from `v_A − v_B`) reads `0.207 ± 0.027` vs the true `0.140` — biased up ~48 %, consistent with the shared systematic error being more gradient-like than the idiosyncratic error | 28 pairs | `results/raw/e1_main/metrics.json` → `pairwise` |
| C13 | `W₂` at `n = 4000` in 2-D cannot resolve even the ground-truth map (`W₂(Φ*#p₀,p₁) = 0.387` > two-sample floor `0.367`), so it is unusable as a denominator here | — | same; `misc` |
| C14 | Gronwall amplification along the true flow is modest: `C_G` median 1.22, p90 5.84 — map error is not predominantly an amplification artefact | 200 particles, Jacobian ODE | same; `misc.C_G` |
| C15 | E3 curl-free arm results | **PENDING** | `results/raw/e3_main/metrics.json` |

## D. Conjectures (labelled as such in the paper; NOT claimed)

| # | Claim | Why unresolved |
|---|---|---|
| D1 | The blind fraction should grow with dimension, `f₀ ≈ (d−1)/d` | dimension-counting heuristic only; `theory.md` §2.3 eq. (5). **Untested** — our budget is 2-D. This is the paper's headline open question and the main reason the result might matter at scale. |
| D2 | A matching lower bound `M ≥ c‖r‖` would turn the blindness claim into a theorem | we have only the Gronwall upper bound; `theory.md` §8.2 |
| D3 | The set of velocity errors invisible to endpoint metrics may be strictly larger than `R_{p_t}` (if the transport semigroup can annihilate a non-flux-null `∇·(p e)`) | `theory.md` §8.3 |
| D4 | Mechanism for why networks suppress circulation: the CFM loss penalises `‖∇φ‖²` and `‖r‖²` **equally**, so the asymmetry must come from the function class or optimiser, not the objective. We have no proof. | `theory.md` §8.1 — stated as an open problem, not an explanation |

## E. Things we explicitly do NOT claim

1. That the flux-null subspace, its orthogonal decomposition, or the curl-freeness of `v*`
   are our discoveries. (A1–A4.)
2. That gradient/potential parameterisation of a velocity field is novel. (A8.)
3. That learned models "exploit" velocity-field non-uniqueness — **we measured the
   opposite** (C2).
4. That a fixed fraction of a model's transport-map error is distributionally invisible
   (C11 forbids it).
5. Any statement about image-scale models, higher dimensions, non-Gaussian sources,
   non-linear interpolants, or stochastic samplers.
6. That the measured effect is *practically large*: the mean map displacement from the
   measured flux-null component is 0.0106 ≈ 0.27 % of the data radius. The tail and the
   mode-flip rate are the parts that look consequential; the mean does not.
