"""Measurement layer: all quantities reported in E1.

Primary metric (pre-registered): f, the time-integrated flux-null energy
fraction of the held-out velocity error e = v_theta - v*.

Secondary: eta^2 (velocity error energy), endpoint W2 and its sampling-noise
floor, transport-map error M against the ground-truth map, pairwise map
disagreement between models, and Gronwall amplification C_G.
"""

import numpy as np
import ot
import torch

from helmholtz import FaceGrid, active_masks, decompose
from ode import integrate_adaptive

DEFAULT_TS = np.linspace(0.02, 0.98, 25)


# ------------------------------------------------------------ velocity error

def eta2_mc(field_fn, gmm, ts=DEFAULT_TS, n=200_000, seed=0):
    """Monte-Carlo estimate of eta^2 = int ||e_t||^2_{L2(p_t)} dt and of
    ||v*||^2 over the same measure (for a relative scale)."""
    gen = torch.Generator().manual_seed(seed)
    num, den, per_t = 0.0, 0.0, []
    for t in ts:
        xt, _, _ = gmm.sample_x_t(n, float(t), generator=gen)
        with torch.no_grad():
            e = field_fn(float(t), xt) - gmm.v_star(float(t), xt)
            vs = gmm.v_star(float(t), xt)
        a = float((e**2).sum(1).mean())
        b = float((vs**2).sum(1).mean())
        per_t.append(a)
        num += a
        den += b
    w = (ts[-1] - ts[0]) / (len(ts) - 1)
    return dict(eta2=num * w, vstar2=den * w, rel=np.sqrt(num / den),
                per_t=np.array(per_t), ts=np.asarray(ts))


# ------------------------------------------------------- Helmholtz over time

def flux_null_fraction(field_fn, gmm, grid=None, ts=DEFAULT_TS, N=192, L=8.0,
                       tol_density=1e-8, return_maps=False):
    """Decompose e = field_fn - v* at each t and integrate the energies.

    ``field_fn(t, x_tensor) -> (n,d) tensor``. For a null-model field this may
    be any field; ``v*`` is subtracted only if ``subtract_vstar`` style use is
    wanted -- here we always pass in the already-formed residual via field_fn
    semantics: field_fn must return the RESIDUAL e directly.
    """
    grid = grid or FaceGrid(N, L)
    cp = torch.as_tensor(grid.cell_positions())
    fp_all = torch.as_tensor(grid.face_pos)
    tot_e2 = tot_g2 = tot_r2 = 0.0
    rows, maps = [], []
    for t in ts:
        t = float(t)
        p_cell = gmm.p_t(t, cp).numpy()
        _, act = active_masks(grid, p_cell, tol_density)
        idx = np.flatnonzero(act)
        p_face = np.zeros(grid.nface)
        p_face[idx] = gmm.p_t(t, fp_all[idx]).numpy()
        with torch.no_grad():
            vec = field_fn(t, fp_all[idx])                     # (m, d) residual
        e_face = np.zeros(grid.nface)
        comp = grid.face_comp[idx]
        v = vec.numpy()
        e_face[idx] = np.where(comp == 0, v[:, 0], v[:, 1])
        r = decompose(grid, p_face, e_face, p_cell, tol_density=tol_density)
        tot_e2 += r["e2"]; tot_g2 += r["g2"]; tot_r2 += r["r2"]
        rows.append(dict(t=t, f=r["f"], e2=r["e2"], g2=r["g2"], r2=r["r2"],
                         ortho=r["ortho"], closure=r["closure"],
                         n_active_cell=r["n_active_cell"]))
        if return_maps:
            maps.append((t, r["r_face"].copy(), r["g_face"].copy(), act.copy()))
    out = dict(f=tot_r2 / tot_e2, e2=tot_e2, g2=tot_g2, r2=tot_r2,
               per_t=rows,
               max_ortho=max(x["ortho"] for x in rows),
               max_closure=max(x["closure"] for x in rows))
    if return_maps:
        out["maps"] = maps
    return out


def residual_fn(model, gmm, dtype=torch.float64):
    """Build field_fn returning the residual e = v_theta - v*."""
    def fn(t, x):
        with torch.no_grad():
            return model(t, x.to(dtype)) - gmm.v_star(t, x)
    return fn


def difference_fn(model_a, model_b, dtype=torch.float64):
    """Ground-truth-free surrogate: Delta = v_A - v_B (theory.md Sec. 7)."""
    def fn(t, x):
        with torch.no_grad():
            return model_a(t, x.to(dtype)) - model_b(t, x.to(dtype))
    return fn


# ------------------------------------------------------------- maps and W2

def w2(a, b):
    """W2 between two equal-size empirical samples (exact EMD)."""
    n = a.shape[0]
    M = ot.dist(np.ascontiguousarray(a, dtype=np.float64),
                np.ascontiguousarray(b, dtype=np.float64))
    w = np.full(n, 1.0 / n)
    return float(np.sqrt(ot.emd2(w, w, M, numItermax=2_000_000)))


def flow_map(field_fn_np, x0, rtol=1e-9, atol=1e-11, t1=1.0, method="DOP853"):
    return integrate_adaptive(field_fn_np, x0, 0.0, t1, rtol=rtol, atol=atol,
                              method=method)


def np_field(model, dtype=torch.float64, clamp_t=(0.0, 1.0)):
    def f(t, x):
        tt = min(max(float(t), clamp_t[0]), clamp_t[1])
        with torch.no_grad():
            return model(tt, torch.as_tensor(x, dtype=dtype)).numpy()
    return f


def np_vstar(gmm):
    def f(t, x):
        tt = min(max(float(t), 0.0), 1.0)
        return gmm.v_star(tt, torch.as_tensor(x, dtype=gmm.dtype)).numpy()
    return f


def gronwall_factor(gmm, x0, ts=None):
    """Measure C_G = sup_s ||J(1,s)|| along the ground-truth flow by
    integrating the Jacobian ODE jointly with the state.

    The state is (x, J) with J in R^{d x d}; dJ/dt = grad_x v*(t,x) J.
    grad_x v* is obtained by autograd on the closed-form field.
    """
    d = gmm.d
    n = x0.shape[0]

    def rhs(t, y):
        y = y.reshape(n, d + d * d)
        x = torch.tensor(y[:, :d], dtype=gmm.dtype, requires_grad=True)
        J = y[:, d:].reshape(n, d, d)
        v = gmm.v_star(float(t), x)
        rows = []
        for i in range(d):
            gi = torch.autograd.grad(v[:, i].sum(), x, retain_graph=(i < d - 1))[0]
            rows.append(gi.detach().numpy())
        Dv = np.stack(rows, axis=1)                    # (n, d, d): Dv[:,i,j]=dv_i/dx_j
        dJ = np.einsum("nij,njk->nik", Dv, J)
        return np.concatenate([v.detach().numpy(), dJ.reshape(n, -1)], axis=1).reshape(-1)

    y0 = np.concatenate([x0, np.tile(np.eye(d).reshape(-1), (n, 1))], axis=1)
    from scipy.integrate import solve_ivp
    ts = ts if ts is not None else np.linspace(0, 1, 21)
    sol = solve_ivp(rhs, (0.0, 1.0), y0.reshape(-1), method="RK45",
                    rtol=1e-7, atol=1e-9, t_eval=ts)
    if not sol.success:
        raise RuntimeError(sol.message)
    # J(1, s) = J(1,0) J(s,0)^{-1}
    Y = sol.y.reshape(n, d + d * d, -1)
    J0 = Y[:, d:, :].reshape(n, d, d, -1)
    J1 = J0[..., -1]
    out = []
    for k in range(len(ts)):
        Js = J0[..., k]
        try:
            Jinv = np.linalg.inv(Js)
        except np.linalg.LinAlgError:
            continue
        Jrel = np.einsum("nij,njk->nik", J1, Jinv)
        out.append(np.linalg.norm(Jrel, ord=2, axis=(1, 2)))
    A = np.stack(out, axis=1)                          # (n, len(ts))
    return dict(C_G_mean=float(A.max(axis=1).mean()),
                C_G_median=float(np.median(A.max(axis=1))),
                C_G_p90=float(np.quantile(A.max(axis=1), 0.9)),
                per_s_mean=A.mean(axis=0), ts=np.asarray(ts))


# ------------------------------------------- scalable spurious-curl diagnostic

def curl_ratio_exact(model, gmm, ts=DEFAULT_TS, n=2000, seed=0):
    """Exact (d=2) local rotationality of a learned field:

        kappa = ||J - J^T||_F / ||J + J^T||_F      averaged over p_t and t

    where J = grad_x v_theta. Since the CFM target is exactly a gradient field
    (theory.md Sec. 2.4), its Jacobian is symmetric and kappa(v*) = 0 exactly.
    kappa therefore measures spurious circulation with a KNOWN correct value of
    zero, needs no grid and no reference field, and is grid-free -- a useful
    cross-check on the weighted-Helmholtz fraction f.
    """
    gen = torch.Generator().manual_seed(seed)
    num, den, rows = 0.0, 0.0, []
    d = gmm.d
    for t in ts:
        xt, _, _ = gmm.sample_x_t(n, float(t), generator=gen)
        xg = xt.clone().requires_grad_(True)
        v = model(float(t), xg)
        J = torch.stack([torch.autograd.grad(v[:, i].sum(), xg, retain_graph=(i < d - 1))[0]
                         for i in range(d)], dim=1)          # J[:,i,j] = dv_i/dx_j
        a = (J - J.transpose(1, 2)).pow(2).sum((1, 2))
        s = (J + J.transpose(1, 2)).pow(2).sum((1, 2))
        num += float(a.sum()); den += float(s.sum())
        rows.append((float(t), float((a.sum() / s.sum()).sqrt())))
    return dict(kappa=float(np.sqrt(num / den)), per_t=rows)


def curl_ratio_hutchinson(model, gmm, ts=DEFAULT_TS, n=2000, n_probe=4, seed=0):
    """Dimension-scalable estimator of the same kappa.

    For v ~ N(0, I),  E||(J - J^T) v||^2 = ||J - J^T||_F^2, and likewise for
    (J + J^T). Each probe costs one JVP and one VJP, so the cost is independent
    of d -- this is the version that would be used on an image model. Validated
    against ``curl_ratio_exact`` in 2-D.
    """
    gen = torch.Generator().manual_seed(seed)
    num, den = 0.0, 0.0
    for t in ts:
        xt, _, _ = gmm.sample_x_t(n, float(t), generator=gen)
        for _ in range(n_probe):
            u = torch.randn(xt.shape, dtype=xt.dtype, generator=gen)
            xg = xt.clone().requires_grad_(True)
            # JVP via double-backward trick; VJP directly
            v = model(float(t), xg)
            (vjp,) = torch.autograd.grad(v, xg, grad_outputs=u, create_graph=True)
            jvp = torch.autograd.grad(vjp, u, grad_outputs=u, retain_graph=True,
                                      allow_unused=True)[0] if u.requires_grad else None
            if jvp is None:
                # fall back: compute JVP with a second forward-over-reverse pass
                u2 = u.clone().requires_grad_(True)
                xg2 = xt.clone().requires_grad_(True)
                v2 = model(float(t), xg2)
                (vjp2,) = torch.autograd.grad(v2, xg2, grad_outputs=u2, create_graph=True)
                (jvp,) = torch.autograd.grad(vjp2, u2, grad_outputs=u, retain_graph=False)
            num += float(((jvp - vjp.detach()) ** 2).sum())
            den += float(((jvp + vjp.detach()) ** 2).sum())
    return dict(kappa=float(np.sqrt(num / den)))
