# Theory

Status labels used throughout:
**[ESTABLISHED]** = standard or published elsewhere, cited.
**[DERIVED]** = derived here, with proof sketch; elementary but written out because the
experiment depends on it.
**[CONJECTURE]** = not proved; to be probed numerically only.
**[OPEN]** = unresolved question, recorded deliberately.

Throughout: `d`-dimensional state, `p_0 = N(0, I_d)`, data `p_1`, independent coupling
`q(x_0,x_1) = p_0(x_0)p_1(x_1)`, linear interpolant `x_t = (1−t)x_0 + t x_1`, target
`u = x_1 − x_0`. `p_t` = law of `x_t`. `⟨a,b⟩_{p} = ∫ p(x) a(x)·b(x) dx`,
`‖a‖²_p = ⟨a,a⟩_p`, and `‖a‖²_{dt⊗p} = ∫₀¹ ‖a_t‖²_{p_t} dt`.

---

## 1. Regularity and support assumptions (stated once, used everywhere)

**(A1)** `p_1` has finite second moment. With `p_0 = N(0,I)` and `t ∈ (0,1)`, `p_t` is then
absolutely continuous with a strictly positive, `C^∞` density on all of `R^d` (it is a
convolution with the non-degenerate Gaussian `N(0,(1−t)²I)`). **Consequence:** on
`t ∈ (0,1)` there are no support-boundary subtleties and `supp p_t = R^d`. This is why we
restrict all measurements to `t ∈ [0, 1−δ]`.

**(A2)** `v_t` is locally Lipschitz in `x`, uniformly on compact `t`-intervals, with at most
linear growth: `‖v_t(x)‖ ≤ C(1 + ‖x‖)`. Then the ODE `ẋ = v_t(x)` has a unique global
solution and a well-defined flow map `Φ_{t←s}`. **Both the neural `v_θ` (finite MLP, SiLU)
and the closed-form `v*` of §4 satisfy (A2) on `t ∈ [0,1−δ]`.**

**(A3)** `∫₀^{1−δ} ‖v_t‖_{L²(p_t)} dt < ∞`, so the flux `p_t v_t` is locally integrable and
the continuity equation makes sense distributionally.

**Why (A2) is not optional.** Without a Lipschitz-type condition the ODE flow may be
non-unique and the marginals of the (non-unique) flow need not equal `p_t`; the link between
the Eulerian continuity equation and the Lagrangian ODE is the superposition / uniqueness
theory of Ambrosio–Gigli–Savaré [ambrosio2008gradientflows] and DiPerna–Lions. Our testbed
is chosen so that (A1)–(A3) hold by construction and this is never the binding issue.
Well-posedness on `[0,1)` for merely finite-second-moment `p_1`, including manifold-supported
data, is Theorem 4.1 of [wan2024elucidating]. **[ESTABLISHED]**

**(A4)** For the terminal time: if `p_1` is a *discrete* measure (the empirical case) then
`v*` has a `1/(1−t)` singularity as `t→1` [li2026kinetic, Eq. 10] and `p_1` is not
absolutely continuous. **Our primary testbed deliberately uses a Gaussian mixture with
`σ = 0.3 > 0`, so `S_k(1) = σ²I ≻ 0` and `v*` is smooth up to and including `t = 1`.** This
removes the terminal-singularity confounder rather than merely bounding it.

---

## 2. Non-uniqueness: the flux-null space

### 2.1 The condition

Let `∂_t p_t + ∇·(p_t v_t) = 0` and set `ṽ_t = v_t + w_t`. Then

    ∂_t p_t + ∇·(p_t ṽ_t) = ∇·(p_t w_t) ,

so `ṽ` generates the same path of marginals **iff**

    ∇·(p_t w_t) = 0   for a.e. t,  distributionally,
    i.e.  ∫ p_t(x) w_t(x)·∇ψ(x) dx = 0   ∀ψ ∈ C_c^∞(R^d).      (★)

Define the **flux-null space** `R_{p} = { w ∈ L²(p;R^d) : (★) holds }`. **[ESTABLISHED]** —
this is Prop. 5.4 of [lim2025hiddenbiases]; also classical.

### 2.2 Orthogonal decomposition

Let `G_p = closure_{L²(p)} { ∇ψ : ψ ∈ C_c^∞ }`. Equation (★) says precisely
`w ⊥_{L²(p)} G_p`. Hence

    L²(p; R^d) = G_p ⊕ R_p ,   R_p = G_p^⊥ ,  orthogonally.      (1)

Every `e ∈ L²(p;R^d)` therefore has a unique decomposition `e = ∇φ + r` with `r ∈ R_p`, and

    ‖e‖²_p = ‖∇φ‖²_p + ‖r‖²_p .                                   (2)

The potential solves the **weighted (degenerate) Poisson equation**

    ∇·(p ∇φ) = ∇·(p e)   with natural (zero-flux) boundary behaviour.   (3)

`φ` is determined up to an additive constant; `∇φ` is unique. **[ESTABLISHED]** — the
`v = ∇φ + r`, `R_p = G_p^⊥` statement is in [lim2025hiddenbiases] §5; the weighted Hodge /
Helmholtz decomposition itself is standard.

**Existence/uniqueness of `∇φ`.** `G_p` is a closed subspace of the Hilbert space
`L²(p;R^d)`, so orthogonal projection onto it exists and is unique; `∇φ := Proj_{G_p} e`.
No regularity of `p` beyond measurability and `e ∈ L²(p)` is needed for *this*. Pointwise
solvability of (3) as a PDE needs more (e.g. `p` locally bounded above and below on compacts
— satisfied by (A1) for `t ∈ (0,1)`). **[DERIVED, elementary]**

### 2.3 Explicit form in `d = 2` and the size of the null space

For `p > 0` smooth on a simply connected domain, `∇·(p w) = 0` ⟺ `p w = ∇^⊥ψ := (−∂_2ψ, ∂_1ψ)`
for a stream function `ψ`, i.e.

    w = ∇^⊥ψ / p .                                                 (4)

So in 2-D the null space is parameterised by **one free scalar function per time slice** —
exactly as many degrees of freedom as the gradient space. The non-uniqueness is not a thin
exceptional set; it is half the space. **[DERIVED]** (The Gaussian-mixture special case
`r^A_t(z) = Σ_i w_i Σ_i(t) A_i(t)(z − m_i(t))` with `A_i` antisymmetric is Prop. 5.5 of
[lim2025hiddenbiases]. **[ESTABLISHED]**)

**Dimension-counting heuristic for a chance-level null fraction.** Condition (★) is one
scalar constraint on a `d`-vector field. Locally (e.g. per Fourier mode, where `(★)` reads
`k·ŵ(k) = 0` in the unweighted case) it removes one of `d` directions, leaving `d−1`. A
field with no preferred orientation relative to the local gradient direction therefore has
expected null-energy fraction

    f_0^heuristic = (d−1)/d ,   = 1/2  in  d = 2 .                 (5)

**[CONJECTURE — heuristic only.]** Equation (5) ignores the weight `p`, the non-stationarity
of the error field, and its spectrum. **We therefore do not assume `f_0 = 1/2`; we estimate
`f_0` from a matched null model** (same architecture at random init, rescaled to the same
norm, same decomposition operator). (5) is used only as a sanity reference.

### 2.4 The CFM target is curl-free — so the entire flux-null subspace is error

**This is the single most consequential fact we found, and it is NOT ours.**

For a Gaussian base `p_0 = N(0,I)`, independent coupling and the linear interpolant,
`x_t | x_1 ~ N(t x_1, (1−t)² I)`, so `p_t` is a Gaussian convolution and Tweedie's formula
gives `E[x_0 | x_t = x] = −(1−t) ∇log p_t(x)`. Then
`E[x_1 | x_t = x] = (x + (1−t)² ∇log p_t)/t`, and

    v*(t,x) = E[x_1 − x_0 | x_t = x] = (1/t) [ x + (1−t) ∇log p_t(x) ]
            = ∇_x Ψ*_t(x) ,    Ψ*_t(x) = ‖x‖²/(2t) + ((1−t)/t) log p_t(x) .   (6')

**So the CFM marginal velocity field is exactly a gradient field: `f(v*) = 0`.**

**[ESTABLISHED — prior work.]** The velocity–score relation is standard
[lipman2022flowmatching; albergo2022stochastic]. Crucially, **the closest prior work already
states the gradient conclusion**: [lim2025hiddenbiases] Example 5.1 gives, for independent
coupling, "the population regression minimizer … `= ∇_z Φ(t,z)`", together with
`∇_z log p_t(z) = (t/(1−t)) v(t,z) − z/(1−t)`, which rearranges to exactly (6'). **We must
not claim this.** We verified it numerically to `10⁻¹⁴` (autograd of `Ψ*` vs closed-form
`v*`, four datasets, five times each) and measured `f(v*) = 4×10⁻⁸ … 1×10⁻²⁹` with our
decomposition operator — which doubles as an independent validation of the solver against a
published theoretical prediction.

**Scope and a clarification of prior work.** Tweedie's formula requires the *source* to be
Gaussian. [lim2025hiddenbiases] Prop. 5.3 states that the empirical minimiser is a gradient
field iff `Σ_i (v_i ∇w_iᵀ − ∇w_i v_iᵀ) = 0`, and remarks that this "is not expected to hold
except in special symmetric or degenerate configurations". We confirmed (targeted re-read)
that Prop. 5.3 is stated for **general affine conditional flows with a latent `Z ~ Q`,
"not necessarily Gaussian"**. For the Gaussian-source linear interpolant — the standard CFM
setting — the same Tweedie argument applies to the *empirical* `p̂_t` (still a Gaussian
convolution), so `v̂*` **is** a gradient field, and the genericity remark does not bite.
**[DERIVED]** We verified this numerically on a 48-component mixture with `σ = 0.12`
(`f = 1×10⁻⁶`). This is a scope clarification of prior work, not a contribution.

**Why this reframes the project.** Writing `v_θ = v* + e` and decomposing
`e = ∇φ + r`, the fact `f(v*) = 0` means

    r(e) = r(v_θ) :  the flux-null component of the ERROR is the flux-null
                     component of the LEARNED FIELD ITSELF.                 (6'')

So a learned flux-null component is not a legitimate alternative choice of field — it is
**spurious circulation that the true field does not possess**. It is:
* exactly invisible to every distributional metric (§3.2),
* fully active on trajectories and the transport map (§3.2),
* pure error, with a known correct value of zero.

The project's question therefore sharpens from *"which flux-equivalent field did the model
pick?"* (ill-posed: the minimiser is unique **and** is a gradient) to:

> **How much spurious circulation do trained CFM networks contain, is it more or less than
> an unstructured field would have, and what does it cost in transport-map fidelity?**

The reference level is now **exact (`f = 0`)** rather than a heuristic null. The matched
null `f_0` retains a different and still useful role: it says how much circulation an
*unstructured* field of the same smoothness would show, so `f < f_0` would mean the network
partially inherits the target's curl-free structure, and `f > f_0` would mean it does worse
than chance.

**Immediate methodological consequence (tested in E3).** Since `v*` is exactly a gradient,
parameterising `v_θ = ∇_x Ψ_θ(t,x)` is a **zero-bias** restriction that removes the
flux-null subspace by construction. Note this architecture is **not new**: Action Matching
learns `v = ∇s_t`, and Energy Matching [see literature_matrix.md] uses a time-independent
potential motivated by OT heuristics ("any rotational component would add unnecessary
distance"), explicitly *not* by (6'). Neither validates irrotationality empirically nor
measures transport-map fidelity. Our contribution can only be the **measurement and its
consequence**, never the architecture.

### 2.5 Five distinctions that must not be collapsed

| Equal … | does **not** imply equal … | why |
|---|---|---|
| marginals `p_t` | velocity field | (★): add any `r ∈ R_{p_t}` |
| marginals `p_t` | flow map / coupling | §3.2: map error sees all of `e` |
| endpoint law `p_1` | transport map | Brenier's map is the unique *gradient* one; a flux-equivalent field need not be a gradient [lim2025hiddenbiases, Prop. 5.3] |
| training loss | generalisation | finite-sample `v̂*` memorises (§6) |
| mathematical non-uniqueness | networks exploiting it | **the whole empirical question**; §3.3 shows the CFM loss is *not* degenerate, so this inference is invalid without measurement |

Also: **solver-induced** trajectory differences are not **field-induced** ones. Separated by
the H4 gate (tolerance sweep), not by assumption.

---

## 3. What the residual error does

Write a trained model as `v_θ = v* + e`.

### 3.1 The CFM loss is exactly the squared velocity error, up to a constant

Because `E[u | x_t] = v*(t,x_t)`, the cross term vanishes:

    L(θ) = E‖v_θ(t,x_t) − u‖²
         = E‖v_θ − v*‖² + 2E⟨v_θ−v*, v*−u⟩ + E‖v*−u‖²
         = ‖e‖²_{dt⊗p} + L_floor ,   L_floor = E[Var(u | t, x_t)] .   (6)

**[DERIVED]** (standard bias–variance identity for conditional-mean regression; the
uniqueness of the minimiser it implies is Prop. 4.3 of [lim2025hiddenbiases] in the
empirical case.)

Two consequences. (i) **Held-out excess CFM loss *is* `η² := ‖e‖²_{dt⊗p}`** — measurable in
any setting where `L_floor` can be estimated, without knowing `v*`. (ii) The minimiser is
unique `dt⊗p_t`-a.e.: **there is no gauge freedom in the objective.** The freedom of §2 can
only manifest in *where `e` lies*, never in *which optimum was selected*.

### 3.2 The structural asymmetry (core of the project)

**Marginals.** Let `p̂_t` be the law of the flow of `v_θ`. With `ε_t = p̂_t − p_t`,
linearising the continuity equation:

    ∂_t ε_t + ∇·(ε_t v*_t) = −∇·(p_t e_t) ,   ε_0 = 0 .              (7)

So, writing `T_{1,s}` for the solution operator of the homogeneous linear transport
`∂_tε + ∇·(εv*) = 0`,

    ε_1 = − ∫₀¹ T_{1,s}[ ∇·(p_s e_s) ] ds + O(‖e‖²) .                (8)

**[DERIVED]**; (7) is Prop. 3.1 of [huang2025divergence] (their forcing term
`−p[∇·(u−v) + (u−v)·∇log p]` is literally `−∇·(p(u−v))`), re-derived here to expose that
**only `∇·(p e)` enters**, hence by (3) **only the `∇φ` component of `e` enters**.

Non-perturbative version: if `e_t ∈ R_{p_t}` for a.e. `t` and (A2) holds for `v_θ`, then
`p̂_t = p_t` **exactly** for all `t` — not just to first order — because `v_θ` generates the
same path. Hence **every** functional of `p̂_1` (FID, `W_2`, KL, TV, precision/recall, mode
coverage) is *exactly* blind to the flux-null component. **[DERIVED]**

**Trajectories.** Let `X*(t;x_0)` solve `ẋ = v*`, `X_θ` solve `ẋ = v_θ`, same `x_0`, and
`δ = X_θ − X*`. Then `δ̇ = ∇v*(t,X*)δ + e(t,X*) + O(‖δ‖²)`, `δ(0)=0`, so

    δ(1) = ∫₀¹ J(1,s) e(s, X*(s;x_0)) ds ,   ∂_t J(t,s) = ∇v*(t,X*) J(t,s), J(s,s)=I.  (9)

**[DERIVED]** This depends on the **pointwise values of all of `e`**, with no divergence
taken — both components contribute.

**Proposition (blindness).** For `e` flux-null and non-zero, `ε_1 = 0` exactly while
`δ(1) = ∫ J(1,s) e(s,X*(s)) ds`, which is generically non-zero. Therefore no distributional
metric bounds the map error. **[DERIVED]** — this is the precise, corrected form of the
project's starting idea.

### 3.3 An identity that makes `η` the natural denominator

Since the law of `X*(s; x_0)` under `x_0 ~ p_0` is exactly `p_s`,

    E_{x_0~p_0} ∫₀¹ ‖e(s, X*(s;x_0))‖² ds = ∫₀¹ ‖e_s‖²_{p_s} ds = η² .   (10)

**[DERIVED]** So the excess CFM loss is precisely the *along-trajectory* error energy. With
(9) and Cauchy–Schwarz,

    M := ‖δ(1)‖_{L²(p_0)} ≤ C_G · η ,   C_G = sup_{s}‖J(1,s)‖ ≤ exp∫‖∇v*‖_∞ ,   (11)

a Gronwall bound. `C_G` is the **amplification factor**; it is measurable in our testbed by
integrating the Jacobian ODE along `v*`. Reporting `C_G` separates "the field error is
amplified" from "the field error is large".

### 3.4 Why `R = M / W₂_excess` can be large even at `f = 1/2`

From (8), the distributional error involves a *divergence* of `p e` — a derivative, followed
by time-integration against a transport semigroup, both of which permit cancellation. From
(9) the map error involves no derivative and no cancellation of this kind. So the ratio `R`
is governed by `C_G`, by the fraction `f`, and by how much cancellation (8) achieves. There
is no reason for `R ≈ 1`. **[CONJECTURE]** — to be measured, not assumed.

---

## 4. Closed-form ground truth for the testbed **[DERIVED]**

Let `p_1 = Σ_k π_k N(μ_k, Σ_k)`, `p_0 = N(0,I)`, independent coupling, linear interpolant.
Conditional on component `k`, `(x_0, x_1) ~ N(0,I) ⊗ N(μ_k,Σ_k)` and `(x_t, u)` is jointly
Gaussian with

    m_k(t) = E[x_t|k] = t μ_k ,                 E[u|k] = μ_k ,
    S_k(t) = Cov(x_t|k) = (1−t)² I + t² Σ_k ,
    Cov(u|k) = I + Σ_k ,
    C_k(t)  = Cov(x_t, u|k) = −(1−t) I + t Σ_k      (symmetric).

Hence, with `b_k(t,x) := μ_k + C_k(t) S_k(t)^{-1} (x − m_k(t))` and posterior
`γ_k(t,x) := π_k N(x; m_k, S_k) / Σ_j π_j N(x; m_j, S_j)`:

    p_t(x) = Σ_k π_k N(x; m_k(t), S_k(t)) ,                          (12)
    v*(t,x) = Σ_k γ_k(t,x) b_k(t,x) ,                                (13)
    L_floor(t,x) = Σ_k γ_k [ tr(Cov(u|k) − C_k S_k^{-1} C_k) + ‖b_k − v*‖² ] .  (14)

**Analytic sanity checks (verified by hand, and numerically in E0):**
* `t = 0`: `m_k = 0, S_k = I, C_k = −I` ⟹ `γ_k = π_k`, `b_k = μ_k − x`, so
  `v*(0,x) = E[x_1] − x`. Matches `E[x_1 − x_0 | x_0 = x]`. ✓
* `t = 1`: `m_k = μ_k, S_k = Σ_k, C_k = Σ_k` ⟹ `b_k = μ_k + (x−μ_k) = x`, so `v*(1,x) = x`.
  Matches `E[x_1 − x_0 | x_1 = x] = x − 0`. ✓

**Relevance to H1.** `γ_k` varies fastest where two or more components have comparable
responsibility — the inter-modal separatrices — and these are exactly the regions where
`p_t` is *small*. By Prop. 5.3 of [lim2025hiddenbiases] the non-gradient structure of `v*`
is generated by `∇γ_k`. So the rotational structure of the target is co-located with the
smallest loss weight. H1 is the hypothesis that this co-location biases the *residual*.

---

## 5. Discretisation with **exact** discrete orthogonality **[DERIVED]**

This matters: if the decomposition is only approximately orthogonal, `f` is confounded by
discretisation error. We choose a finite-volume scheme for which (1)–(2) hold **to machine
precision**, so the only remaining numerical question is consistency (a grid-refinement
study), not orthogonality.

Uniform grid, spacing `h`, cell centres `x_{ij}` on `[−L,L]²`. Unknowns `φ_{ij}` at cell
centres. Define the discrete inner product on **faces**:

    ⟨a,b⟩_h := h² Σ_{faces f} p_f a_f b_f ,

where `p_f` and `a_f` are evaluated at the face midpoint (both are available in closed form
for `p_t`, and `v_θ`, `v*` can be evaluated exactly at face midpoints — no interpolation).
Discrete gradient on a face between cells `A,B`: `(∇_h φ)_f = (φ_B − φ_A)/h`.

Solve the discrete weighted Poisson system: find `φ` with

    ⟨∇_h φ, ∇_h ψ⟩_h = ⟨e, ∇_h ψ⟩_h    for all grid functions ψ.     (15)

This is a symmetric positive-semidefinite 5-point system (`p_f ≥ 0` weights), singular
exactly on constants; we pin one DOF. Zero-flux boundary: drop boundary faces, which is
consistent because `p_t` is `< 10⁻¹²` at `|x| = 8` in our testbed. Set
`r := e − ∇_h φ` on faces.

**Lemma (exact discrete orthogonality).** Taking `ψ = φ` in (15) gives
`⟨∇_hφ, ∇_hφ⟩_h = ⟨e, ∇_hφ⟩_h`, hence

    ⟨∇_hφ, r⟩_h = ⟨e,∇_hφ⟩_h − ⟨∇_hφ,∇_hφ⟩_h = 0 ,
    ‖e‖²_h = ‖∇_hφ‖²_h + ‖r‖²_h   exactly (to linear-solver tolerance).   (16)

**[DERIVED]** So `f_h := ‖r‖²_h / ‖e‖²_h ∈ [0,1]` by construction, and the Galerkin property
(15) means `∇_hφ` is the *exact* `⟨·,·⟩_h`-orthogonal projection of `e` onto discrete
gradients. Consequences: `f_h` is a genuine energy fraction, never negative, never `>1`; and
because discrete gradients are a *subspace* of the continuum `G_{p}`, `f_h` is an
**upper bound-biased** estimate of the continuum `f` (projecting onto a smaller gradient
space can only leave more residual). The refinement study `f_h(192)` vs `f_h(256)` tests
whether this bias has converged. **[DERIVED]**

Time integration of `f`: `f = Σ_t w_t ‖r_t‖²_h / Σ_t w_t ‖e_t‖²_h` with quadrature weights
`w_t` on `t ∈ [0, 1−δ]`.

---

## 6. The empirical/memorisation regime (secondary; kept separate on purpose)

For a finite dataset `D = {x_1^{(i)}}_{i=1}^n` with `p_1` replaced by the empirical measure,
the same algebra with `Σ_k → 0`, `π_i = 1/n`, `μ_i = x_1^{(i)}` gives
`S_i(t) = (1−t)²I`, `C_i(t) = −(1−t)I`, `b_i = x_1^{(i)} − (x − t x_1^{(i)})/(1−t)`, and

    v̂*(t,x) = Σ_i w_i(t,x) [ x_1^{(i)} − (x − t x_1^{(i)})/(1−t) ] ,
    w_i ∝ exp( −‖x − t x_1^{(i)}‖² / (2(1−t)²) ) .                   (17)

**[DERIVED]**, and it agrees with Prop. 4.3 of [lim2025hiddenbiases] and Eq. 10 of
[li2026kinetic]. This field memorises `D` and has a `1/(1−t)` singularity at `t→1`.

**Design decision.** Using (17) as the "ground truth" would conflate *error* with
*desirable generalisation*: a good network should **not** converge to `v̂*`. Our primary
experiment therefore trains on a **fresh i.i.d. sample at every step**, so the regression
target is genuinely the population `v*` of (13) and `e = v_θ − v*` is pure
approximation + optimisation error. The finite-`n` regime is a separate, later study where
`f` must be interpreted differently. This is the single most important confounder control in
the design.

---

## 7. A ground-truth-free estimator of the error geometry **[DERIVED / to validate]**

`f` as defined needs `v*`, which exists in closed form only in synthetic settings. But for
two independently trained models, `Δ := v_A − v_B = e_A − e_B`, and `Δ` is observable with
no ground truth at all. Decomposing `Δ` by the same operator gives `f_Δ`.

If `e_A, e_B` were independent and identically distributed with null fraction `f`, then
`E‖r_A − r_B‖² = 2E‖r‖²` and `E‖∇φ_A − ∇φ_B‖² = 2E‖∇φ‖²`, so `f_Δ = f` in expectation.
**[DERIVED under an independence assumption that is certainly false in detail]** — seeds
share architecture, data distribution and optimiser, so `e_A` and `e_B` have a common
systematic part. Measuring `f` and `f_Δ` side by side in the synthetic testbed *tests* this
assumption and, if it holds, yields a diagnostic usable at scale (where `v*` is unavailable).
This is a secondary deliverable, not a claim.

---

## 8. Unresolved mathematical questions **[OPEN]**

1. **Is there any mechanism by which gradient descent on `L_CFM` prefers one component of
   `e` over the other?** By (6) and (2) the loss is exactly `‖∇φ‖² + ‖r‖² + const`, so the
   *loss* is symmetric. Any asymmetry must come from the architecture's function class or
   the optimiser's implicit bias interacting with the *spatial co-location* of §4. We have
   no proof either way. This is precisely why H1 is an experiment and not a theorem.
2. **Sharpness of (11).** Is there a matching lower bound `M ≥ c·‖r‖` under
   non-degeneracy of `J`? A lower bound would turn H2 from a measurement into a theorem.
   Note [lim2025hiddenbiases] also reports only one-sided (upper-tail) bounds.
3. **Does (8) ever cancel the `∇φ` contribution?** If `T_{1,s}` can annihilate
   `∇·(p_s e_s)` for a non-flux-null `e`, then distributional metrics are blind to *more*
   than `R_{p_t}` — a strictly larger blind subspace. Characterising
   `ker( e ↦ ∫ T_{1,s}∇·(p_s e_s) ds )` is open and would strengthen H2.
4. **Does `f` have a limit as capacity → ∞?** If `e → 0` in `L²`, `f` is a ratio of two
   vanishing quantities and need not converge. Behaviour of `f` along the capacity sweep is
   an empirical question with no theory.
5. **Dimension dependence.** (5) suggests `f_0 → 1` as `d → ∞`, so the blind fraction should
   *grow* with dimension. If true this is the reason to care about images — but it is a
   heuristic, untested, and our budget is 2-D. Recorded as the main scaling question.
6. Whether the weighted Poisson problem (3) is well-posed in `L²(p_t)` without the
   positivity afforded by (A1) — relevant only for manifold-supported `p_1`, which we avoid.
