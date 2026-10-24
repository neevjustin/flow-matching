# Literature Matrix — Flow Matching: velocity-field non-uniqueness and trajectory geometry

Date of review: 2026-10-24.
Reading protocol: for every paper marked **FULL** below, the arXiv HTML full text was
fetched and queried directly. Papers marked **ABS/SEARCH-ONLY** were seen only through
search metadata and are explicitly *not* treated as read. Nothing in this file is
inferred from a title alone.

Search tooling: `src/arxiv_search.py`, `src/arxiv_meta.py` (arXiv API). Queries run:
`"flow matching" AND "continuity equation"`, `... AND "kinetic energy"`,
`... AND "non-uniqueness"`, `... AND "divergence" AND "velocity"`,
`... AND "identifiability"`, `... AND ("divergence-free" OR "Helmholtz")`,
`"aligning flow divergence"`, `"flow matching" AND "transport map" AND "error"`,
plus title lookups.

---

## 0. Two seed-list entries could not be verified — flagged, not cited as real

The prompt's seed list contained two entries we **could not find**:

| Seed-list entry | Outcome |
|---|---|
| "FlowGuard: Guarding Flow Matching via Conformal Sampling (2025)" | **Not found.** Three distinct arXiv papers are titled "FlowGuard" (MCP security detection 2607.14754; in-generation safety for diffusion 2604.07879; model-stealing detection 2606.03430). None is about conformal sampling for flow matching. The query `abs:"conformal" AND abs:"flow matching" AND abs:"guard"` returned **zero** results. |
| "High-Order Flow Matching: Unified Framework and Sharp Statistical Rates (NeurIPS 2025)" | **Not found.** `ti:"High-Order Flow Matching"` returned zero results. Closest existing work: Cao et al., *Towards High-Order Mean Flow Generative Models* (2508.07102, 2025) — feasibility/expressivity of high-order mean flows, not "sharp statistical rates". |

These two are treated as **non-existent / misremembered** and are **not** used to support
any claim. Reported rather than silently substituted.

---

## 1. THE CLOSEST PRIOR WORK — read in full, twice

### Lim (2025), *On The Hidden Biases of Flow Matching Samplers* — arXiv:2512.16768v3 — **FULL**
Preprint, 41 pp. Venue not stated by the author. URL: https://arxiv.org/abs/2512.16768

**This paper independently contains essentially the entire mathematical content of our
starting hypothesis.** It must be read before any theory work is attempted.

- **Problem.** Characterises four finite-sample biases of empirical flow matching (EFM):
  statistical, geometric, **non-uniqueness**, and energetic.
- **Main contributions relevant to us.**
  - Defines the **flux-null remainder space** `R_{p_t} = { r : ∫ p_t(z) r(z)·∇φ(z) dz = 0 ∀φ ∈ C_c^∞ }`.
  - **Prop. 5.4 (flux equivalence):** `r_t ∈ R_{p̂_t}` ⟹ `u_t = v̂_t + r_t` satisfies the same
    continuity equation. Same marginals, different trajectories.
  - **Orthogonal decomposition** `v_t = ∇φ_t + r_t` with `R_{p_t} = G_{p_t}^⊥` in `L²(p_t)`.
  - **Prop. 5.5:** explicit flux-null family for Gaussian mixtures, parameterised by
    antisymmetric matrices: `r_t^A(z) = Σ_i w_i(t,z) Σ_i(t) A_i(t) (z − m_i(t))`.
  - **Prop. 4.3:** closed form for the exact empirical CFM minimiser,
    `v̂*(t,z) = Σ_i w_i(t,z)(a_t(x^i) z + b_t(x^i))`, unique `dt ⊗ p̂_t`-a.e.
  - **Prop. 5.3:** `v̂*` is a gradient field iff
    `Σ_i (v_i ∇_z w_i^T − ∇_z w_i v_i^T) = 0`; "not expected to hold except in special
    symmetric or degenerate configurations". So the empirical minimiser is *generically
    non-gradient*, hence generically not an OT map.
  - Thms 5.7/5.8/Prop 5.9: source-distribution tails control kinetic-energy tails.
- **Assumptions.** Exact minimisers over *unrestricted* `L²(dt p̂_t)` function classes;
  affine conditional flows; unique ODE solutions on `[0,T]`, `T < 1`; finite dataset with
  bounded norms; antisymmetric-matrix flux-nulls require `L²(p̂_t)` membership.
- **Experiments.** 2D toy data (two moons, 8 Gaussians, checkerboard); Gaussian vs
  Student-t sources; metric = integrated kinetic energy `E_T`; survival curves and tail
  exponents. Forward Euler. Explicitly: *"the goal is not to benchmark generative
  quality, but to test the qualitative mechanism."*
- **CRITICAL for us — what it does NOT do.** Verified by a second targeted full-text read:
  - *"All experiments evaluate the closed-form empirical velocity directly; **no neural
    network is trained**."* (Appendix C)
  - Does **not** quantify the flux-null fraction of any *learned* field's error.
  - Does **not** measure transport-map or per-particle endpoint error against ground truth.
  - Explicitly leaves open: *"error bounds for neural network approximations to exact
    minimizers"*, *"effects of numerical ODE solvers or timestep schedules"*,
    *"understanding how these effects persist under model approximation, discretization,
    stochastic sampling... is an important direction for future work."*
  - One-sided (upper-tail) bounds only.
- **Overlap with our direction.** **Near-total on theory, zero on empirics.** Any
  "discovery" we might claim about the *existence* or *structure* of marginal-preserving
  velocity perturbations is already published here. Our contribution can only live in the
  empirical regime the author explicitly excludes: **trained neural networks**.

---

## 2. Papers read in full

### Li et al. (2026), *A Kinetic Energy Perspective of Flow Matching* — arXiv:2602.07928v2 — **FULL**
ICML 2026 Spotlight (authors' comment). Shares an author (Soon Hoe Lim) with §1.

- **Contribution.** Kinetic Path Energy `E = ½∫₀¹‖v_θ(x(t),t)‖²dt` as a *per-sample*
  diagnostic; Kinetic Trajectory Shaping (KTS), a training-free inference-time fix.
- **Findings for us.** (i) Thm 4.2 ties instantaneous squared speed to `−log p̂_t` under a
  "posterior dominance" regime ⟹ low-density traversal costs energy. (ii) Explicitly:
  *"Standard metrics like FID are fundamentally trajectory-blind; they aggregate global
  statistics but overlook the dynamics of individual paths."* (iii) 98% near-copies
  coexisting with low FID — direct evidence that distributional metrics miss
  trajectory/sample-level pathology.
- **Assumptions.** Posterior-dominance regime (regime-specific, acknowledged); ODE flows
  only; local support estimates are representation-dependent proxies, *"should not be
  interpreted as calibrated estimates of data-manifold density."*
- **Experiments.** ImageNet-256, CIFAR-10, CelebA, synthetic 2D (`dense_sparse`,
  `multiscale_clusters`, `sandwich`). Baselines: vanilla FM, empirical FM, SiT-XL/2,
  OT-CFM. Metrics: FID, CLIP, Spearman ρ, Cliff's δ, memorisation fraction, kNN/KDE.
- **Does NOT establish.** No non-uniqueness, flux-null, or Helmholtz discussion at all.
  Treats the learned field as canonical. No decomposition of velocity *error*.
- **Overlap.** Supplies our *motivation* (trajectory-blindness of distributional metrics)
  and a baseline family of trajectory diagnostics we must beat/distinguish ourselves from.
  Does not pre-empt the identifiability question.

### Huang et al. (ICML 2025), *Improving Flow Matching by Aligning Flow Divergence* — arXiv:2602.00869v1 — **FULL**
- **Contribution.** CFM loss alone does not control the probability path. Prop 3.1: the
  density error `ε_t = p_t − p̂_t` obeys `∂_t ε_t + ∇·(ε_t v_t) = L_t` with forcing
  `L_t = −p_t[∇·(u_t−v_t) + (u_t−v_t)·∇log p_t]`. Thm 3.3: `TV(p_t,p̂_t) ≤ ½ L_DM(θ)`.
  Thm 4.1: tractable conditional surrogate `L_CDM ≥ L_DM`. Objective `λ₁L_CFM + λ₂L_CDM`.
- **Why this matters enormously for us.** The forcing term `L_t` is *exactly*
  `−∇·(p_t (u_t − v_t))/1` in disguise. So this paper's own error PDE says: **the
  distributional error is driven only by the weighted divergence of the velocity error.**
  Velocity error with zero weighted divergence (= flux-null) produces `L_t ≡ 0` and hence
  zero density error. They state the mechanism without naming it or exploring it.
- **Assumptions.** "Mild assumptions adopted in (Lu 2022; Lipman 2023; Albergo 2023)",
  relegated to appendix; smoothness/boundedness/regularity.
- **Experiments.** Checkerboard 2D, 1D GMM, CIFAR-10, DNA promoters, Lorenz/FitzHugh-
  Nagumo, KTH, BAIR. Baselines CFM, DM, DDM. Metrics NLL, TV, KL, FID, FVD, PSNR, MSE.
  Checkerboard TV 0.0945 → 0.0587; CIFAR FID 6.35 → 5.62, NLL 2.99 → 2.85.
- **Limitations (authors').** λ chosen by grid search; TV bound does not bound KL;
  ~50% compute overhead; video results behind Chen et al. 2024.
- **Does NOT establish.** No non-uniqueness or flux-null decomposition. **All metrics are
  distributional**; verified — no pointwise trajectory deviation, no path-measure
  Wasserstein, no coupling/transport-map error.
- **Overlap.** This is the paper whose *theory* most directly implies our mechanism, and
  whose *method* is the natural baseline: if flux-null error is the problem, FDM should
  **not** fix it (it only penalises the divergence-carrying part). That is a sharp,
  pre-registrable prediction distinguishing us from them.

### Yang et al. (ICML 2026), *Stable Velocity: A Variance Perspective* — arXiv:2602.05435v3 — **FULL**
- **Contribution.** Two-regime variance structure of the CFM target; StableVM
  (multi-sample variance-reduced, unbiased, Thms 3.1–3.3 giving `O(1/n)` variance decay),
  VA-REPA, StableVS.
- **Assumptions.** Affine conditional velocity; uniform integrability.
- **Experiments.** ImageNet-256 latents, CIFAR-10, GMMs, GenEval, T2V-CompBench;
  SiT-B/L/XL, SD3.5, Flux-dev, Qwen-Image, Wan2.2. Baselines REPA, iREPA, REG, REPA-E,
  MaskDiT, DiT-XL/2. FID 1.80 @80 ep vs REPA 1.98.
- **Does NOT establish.** No non-uniqueness / flux-null / Helmholtz. No decomposition of
  trained-network velocity error. No transport-map error. No learned-vs-true velocity
  comparison. Split point ξ unprincipled (acknowledged).
- **Overlap.** Orthogonal. It reduces *target variance*; our question is about the
  *geometry of the residual error*. A relevant confounder control: variance reduction
  changes optimisation noise, which could change the error's geometry — worth an ablation.

### Li et al. (ICML 2026), *The Velocity Deficit* — arXiv:2605.14819v1 — **FULL**
- **Contribution.** `‖v*‖² = ‖E[u|x_t]‖² < E[‖u‖²|x_t]` (Jensen) ⟹ systematic speed
  underestimation ⟹ "integration lag"; fixed by initial energy injection (SSC).
- **Assumptions.** Asm 3.1 independent coupling; high-dimensional near-orthogonality
  (cross term ~1.24% on ImageNet).
- **Experiments.** ImageNet-1k 256/512, MS-COCO; SiT S/2–XL/2, REPA-SiT, MMDiT, U-Net.
  Introduces Fréchet Latent Distance tracked across timesteps. SiT-XL/2 FID 13.68 → 7.58
  @NFE 50.
- **Does NOT establish.** No non-uniqueness/flux-null/Helmholtz. Trajectory error and
  transport-map error are *conflated* as "integration lag" (their own framing).
  Acknowledges the fix may overshoot on low-dimensional toy data — relevant: our testbed
  is 2D, so SSC is not a fair baseline there.
- **Overlap.** A *magnitude* effect (speed too small), ours is a *direction* effect
  (error points along marginal-preserving directions). Must be measured separately so the
  two are not confused — an ablation we owe the reader.

### Wan et al. (ICML 2025), *Elucidating Flow Matching ODE Dynamics* — arXiv:2412.18730v4 — **FULL**
- **Contribution.** Three-stage trajectory picture (mean attraction → cluster absorption →
  terminal convergence). Thm 4.1 well-posedness on `[0,1)` for finite second moment,
  including manifold support. Thm 5.3 flow map exists a.e. under Asm 5.2 (positive reach,
  `p(B_r(x)) ≥ C_R r^k`); rates `O(σ_t^{ζ/2})`, `O(√σ_t)` manifolds, `O(σ_t)` discrete.
- **Does NOT establish.** No non-uniqueness/flux-null/Helmholtz — treats the conditional
  field as canonical. No measurement of learned-vs-true velocity error, learned-vs-true
  flow map, or coupling error. Appendix J is qualitative visualisation only; no FID.
- **Limitations (authors').** Constants are loose worst-case; Local Cluster Assumption
  rarely exactly holds; independence of `X`,`Z` assumed; *"memorization analysis assumes
  asymptotically optimal denoisers; behavior with finite-capacity networks
  uncharacterized."*
- **Overlap.** Gives us the regularity results we need to cite for well-posedness of the
  ground-truth flow map. The finite-capacity gap it names is adjacent to our question.

### Vaitl & Klein (2025), *Path Gradients after Flow Matching* — arXiv:2505.10139v3 — **FULL**
- **Contribution.** FM pretraining + path-gradient fine-tuning; up to 3× ESS. Augmented
  adjoint for constant-memory path gradients.
- **Assumptions.** Unbiased training samples; differentiable energy function (so: no
  images).
- **Experiments.** 2D GMM, LJ-13/55, alanine dipeptide, transferable BGs. Metrics ESS_q,
  ESS_p, NLL, trajectory length, energy histograms. Baselines: pure FM, pure PG, MLE.
- **Does NOT establish.** No non-uniqueness/flux-null/Helmholtz; velocity field treated as
  a black box. Does not separate flow-map fidelity from marginal fidelity. Notably they
  observe *"PG fine-tunes q_θ without significantly altering the dynamics v_θ"* and that
  the FM MSE stays *"largely unchanged"* while sampling efficiency improves — a hint of
  the decoupling we want to measure, but **not framed as a pathology and not measured**.
- **Overlap.** Strong *motivation*: path gradients differentiate through the flow **map**,
  so flux-null error is a live source of bias for this use case. Cite as a downstream
  consumer that our effect would damage. 275× per-iteration cost means we cannot use it
  as a baseline under our compute budget.

### Milanfar (2026), *A Lagrangian View of Flow Matching* — arXiv:2609.00198v2 — **FULL**
- **Contribution.** Derives FM straight-line paths from an invariance principle
  (`d/dt f(x(t),t)=0`) rather than OT, via the advection PDE
  `∂_t f + J_f v = 0` and a residual ansatz `v = c(t)(x − f(x,t))`.
- **Does NOT establish.** Only a 1D two-mode toy example; no networks trained; no learned
  velocity analysis; no coupling error. Author explicitly concedes: *"We have not proven
  that the residual ansatz is the unique velocity field solving the invariance PDE—other
  choices of v could in principle also be made self-consistent."*
- **Overlap.** That concession is literally our non-uniqueness question in another
  language, but left entirely open and uncharacterised. Useful framing citation.

### Peng & Gao (2026), *Flow Matching with Arbitrary Auxiliary Paths* — arXiv:2605.06364v1 — **FULL**
- **Contribution.** `X_t = a(t)X_1 + b(t)X_0 + c(t)η` for arbitrary auxiliary `η`.
- **Critical distinction.** This changes the **probability path itself**, not the velocity
  field at fixed path. It is *not* a flux-null construction. No Helmholtz discussion.
- **Experiments.** MNIST, CIFAR-10, ImageNet-1k; FID/sFID/IS/accuracy; η ∈ {Gaussian,
  uniform, Laplace, Rademacher, label-guided}. Improvements described as marginal.
  No trajectory/transport-map comparison at matched quality. No limitations section.
- **Overlap.** Low. Important as a *disambiguation*: "different velocity field, same
  marginals" (us) ≠ "different marginals by design" (them).

---

## 3. Papers identified and classified but NOT read in full — **ABS/SEARCH-ONLY**

Listed for completeness and as the candidate queue. **These are not evidence.** None is
relied upon for any claim in `hypothesis_ledger.md`.

| Paper | arXiv | Why queued / provisional relevance |
|---|---|---|
| Isobe et al., *Extended Flow Matching: Generalized Continuity Equation* | 2402.18839 | Generalised continuity equation — could bear on non-uniqueness. **Must read before any theory claim.** |
| Kapuśniak et al., *Metric Flow Matching* (NeurIPS 2024) | 2405.14780 | Data-manifold geodesic interpolants; changes the path, like AuxPath. Relevant to path-geometry baselines. |
| Cao et al., *Towards High-Order Mean Flow Generative Models* | 2508.07102 | Closest match to an unverifiable seed entry. |
| Fukumizu et al., *Flow matching achieves almost minimax optimal convergence* | 2405.20879 | Statistical rates — bears on how fast velocity error shrinks, i.e. the denominator of our ratio. |
| Zeng & Yan, *FM in the Low-Noise Regime: Pathologies* | 2509.20952 | `t→1` pathologies; a confounder for our terminal-time measurements. |
| Yin et al., *Background Gradients Shape Memorization in FM* | 2609.33210 | Memorisation mechanism; relevant to our empirical-minimiser secondary study. |
| Lim?/—, *What Do Flow-Based Inverse Solvers Approximate? A Posterior-Transport View* | 2606.24516 | Transport-vs-posterior framing. |
| —, *What Does a Stream Model Buy You in Flow Matching?* | 2609.34123 | Possibly about path/velocity representation choice. |
| —, *On Flow Matching KL Divergence* | 2511.05480 | Divergence bounds. |
| —, *Lipschitz regularity in FM and Diffusion: sharp sampling rates* | 2604.06065 | Gronwall/Lipschitz amplification constants we need. |
| —, *Optimal Transport Flow Matching by Design* | 2606.04092 | Enforcing gradient/OT structure — a potential *remedy* baseline. |
| —, *Diversified Flow Matching with Translation Identifiability* | 2511.05558 | Uses "identifiability" but for translation tasks. |
| —, *Lagrangian Flow Matching: A Least-Action Framework* | 2605.15419 | Least action ⟹ gradient fields; potential remedy baseline. |
| —, *Prediction–Loss Alignment for Sampler-Robust FM Training* | 2602.10420 | Solver-robustness confounder. |
| —, *HCFM: Helmholtz Component Flow Matching* (ICDE 2027) | 2610.25059 | **Name collision only** — Helmholtz decomposition of *time-series signals* for anomaly detection, not of velocity error. Checked via search metadata; title/abstract confirm different subject. |

**Negative search result worth recording:** the query
`all:"flow matching" AND (all:"divergence-free" OR all:"Helmholtz")` returned **6** hits,
*none* of which decomposes a learned flow-matching velocity field (or its error) into
gradient and divergence-free parts. The only Helmholtz+FM paper is the unrelated HCFM.

---

## 4. Synthesis: where the gap actually is

**Established by prior work (we must not claim any of it):**
1. The continuity equation does not determine the velocity field. Marginal-preserving
   perturbations form the flux-null space `R_{p_t} = G_{p_t}^⊥` in `L²(p_t)`. [lim2025hiddenbiases; classical: benamou2000computational, ambrosio2008gradientflows, villani2009ot]
2. The exact empirical CFM minimiser has a closed form and is unique `dt⊗p̂_t`-a.e.
   [lim2025hiddenbiases, Prop 4.3] — **so the CFM objective is *not* degenerate**; the
   non-uniqueness of the continuity equation is *not* a non-uniqueness of the loss.
3. That minimiser is generically *not* a gradient field. [lim2025hiddenbiases, Prop 5.3]
4. Flux-equivalent fields can have radically different kinetic-energy tails. [lim2025hiddenbiases, Rmk 5.1]
5. Distributional error is driven by the *weighted divergence* of velocity error.
   [huang2025divergence, Prop 3.1]
6. Distributional metrics are trajectory-blind. [li2026kinetic]

**Not established by any paper we read — the gap:**
> Prior work proves that a large marginal-preserving subspace *exists*, and separately
> that distributional metrics are *blind* to trajectory structure. **No paper measures
> how a trained neural flow-matching model's velocity error is actually distributed
> between the marginal-preserving (flux-null) and marginal-changing (gradient) subspaces.**
> Without that measurement, the mathematical non-uniqueness has no demonstrated
> empirical consequence for learned models — which is exactly the inference the starting
> hypothesis makes and which the closest prior work (lim2025hiddenbiases) explicitly
> declines to make, having trained no networks.

**Corollary that reshapes our starting hypothesis.** The starting idea said learned models
might "exploit" the velocity-field freedom. Point 2 above shows the CFM population/empirical
objective has a **unique** minimiser, so there is no gauge freedom for a model to exploit
*at the optimum*. The freedom can only enter through the **residual** `e = v_θ − v*`. The
right question is therefore not "which field did the model pick?" but **"where in the
error space does the model's residual live, and is that location consequential?"**
This correction is recorded in `hypothesis_ledger.md` and drives the experiment design.
