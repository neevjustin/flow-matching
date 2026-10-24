"""Low-variance distributional discrepancies, and grid-field interpolation.

Why this module exists
----------------------
Sample-based W2 in 2-D converges like n^{-1/4}; at n = 4000 its noise floor
(~0.37 in our testbed) is far larger than the distributional error of a
well-trained model. Using it as the denominator of H2's ratio R would measure
estimator inefficiency, not geometry. We therefore add:

  * ``energy_distance``  -- O(n^{-1/2}), a proper metric on distributions
  * ``cnf_kl``           -- KL(p_hat_1 || p_1) via the exact continuous
                            change-of-variables, using the closed-form log p_1.
                            This is an average of a pointwise quantity, so it
                            also converges at O(n^{-1/2}).

``GridFieldInterp`` lifts the per-time-slice Helmholtz components (which live
on grid faces) to a callable velocity field, so that a measured component can
be re-injected into the ODE (experiment E2).
"""

import numpy as np
import torch
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator


# ----------------------------------------------------------- energy distance

def energy_distance(a, b, rng=None, n_sub=None):
    """Energy distance E = 2E|X-Y| - E|X-X'| - E|Y-Y'| (>= 0, zero iff equal)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if n_sub is not None and a.shape[0] > n_sub:
        rng = rng or np.random.default_rng(0)
        a = a[rng.choice(a.shape[0], n_sub, replace=False)]
        b = b[rng.choice(b.shape[0], n_sub, replace=False)]

    def md(x, y):
        d = np.linalg.norm(x[:, None, :] - y[None, :, :], axis=2)
        return d

    ab = md(a, b).mean()
    aa = md(a, a)
    bb = md(b, b)
    na, nb = a.shape[0], b.shape[0]
    # unbiased within-sample terms (exclude the diagonal)
    aa = aa.sum() / (na * (na - 1))
    bb = bb.sum() / (nb * (nb - 1))
    return float(2 * ab - aa - bb)


# ---------------------------------------------------------------- CNF log-lik

def cnf_logdet_and_endpoint(model, x0, rtol=1e-7, atol=1e-9, dtype=torch.float64):
    """Integrate (x, a) with da/dt = div v_theta(t, x), returning the endpoint
    and the accumulated log-volume change.

    log p_hat_1(x(1)) = log p_0(x(0)) - a(1).
    Divergence is computed exactly (not with Hutchinson) since d = 2.
    """
    x0 = np.asarray(x0, dtype=np.float64)
    n, d = x0.shape

    def rhs(t, y):
        y = y.reshape(n, d + 1)
        x = torch.tensor(y[:, :d], dtype=dtype, requires_grad=True)
        v = model(float(t), x)
        div = torch.zeros(n, dtype=dtype)
        for i in range(d):
            gi = torch.autograd.grad(v[:, i].sum(), x, retain_graph=(i < d - 1))[0]
            div = div + gi[:, i]
        out = np.concatenate([v.detach().numpy(), div.detach().numpy()[:, None]], axis=1)
        return out.reshape(-1)

    y0 = np.concatenate([x0, np.zeros((n, 1))], axis=1).reshape(-1)
    sol = solve_ivp(rhs, (0.0, 1.0), y0, method="RK45", rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(sol.message)
    Y = sol.y[:, -1].reshape(n, d + 1)
    return Y[:, :d], Y[:, d], dict(nfev=int(sol.nfev))


def cnf_kl(model, gmm, x0, rtol=1e-7, atol=1e-9):
    """Monte-Carlo KL(p_hat_1 || p_1) with its standard error.

    KL = E_{x0~p0}[ log p_hat_1(Phi(x0)) - log p_1(Phi(x0)) ]
       = E_{x0~p0}[ log p_0(x0) - a(1) - log p_1(Phi(x0)) ]
    """
    xe, a, info = cnf_logdet_and_endpoint(model, x0, rtol=rtol, atol=atol)
    lp0 = (-0.5 * (x0**2).sum(1) - np.log(2 * np.pi))
    lp1 = gmm.log_p_t(1.0, torch.as_tensor(xe)).numpy()
    vals = lp0 - a - lp1
    return dict(kl=float(vals.mean()), se=float(vals.std(ddof=1) / np.sqrt(len(vals))),
                endpoint=xe, nfev=info["nfev"])


# ------------------------------------------------- grid component -> field

class GridFieldInterp:
    """Interpolate a face-sampled, per-time-slice vector field.

    Built from the ``maps`` output of ``flux_null_fraction(..., return_maps=True)``.
    x-component lives on x-faces (an (N-1) x N grid), y-component on y-faces
    (an N x (N-1) grid). Bilinear in space, linear in time, zero outside.
    """

    def __init__(self, grid, slices, which="r"):
        """slices: list of (t, r_face, g_face, active_mask) as returned by
        flux_null_fraction(return_maps=True)."""
        self.ts = np.array([s[0] for s in slices])
        N, L, h = grid.N, grid.L, grid.h
        xf_x = -L + (np.arange(N - 1) + 1) * h
        xf_y = -L + (np.arange(N) + 0.5) * h
        yf_x = -L + (np.arange(N) + 0.5) * h
        yf_y = -L + (np.arange(N - 1) + 1) * h
        self.ix, self.iy = [], []
        k = 1 if which == "r" else 2
        for s in slices:
            vals = s[k]
            vx = vals[: grid.nxf].reshape(N - 1, N)
            vy = vals[grid.nxf:].reshape(N, N - 1)
            self.ix.append(RegularGridInterpolator((xf_x, xf_y), vx,
                                                   bounds_error=False, fill_value=0.0))
            self.iy.append(RegularGridInterpolator((yf_x, yf_y), vy,
                                                   bounds_error=False, fill_value=0.0))

    def __call__(self, t, x):
        x = np.asarray(x, dtype=np.float64)
        t = float(np.clip(t, self.ts[0], self.ts[-1]))
        j = int(np.searchsorted(self.ts, t, side="right") - 1)
        j = max(0, min(j, len(self.ts) - 2))
        t0, t1 = self.ts[j], self.ts[j + 1]
        w = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
        out = np.empty_like(x)
        out[:, 0] = (1 - w) * self.ix[j](x) + w * self.ix[j + 1](x)
        out[:, 1] = (1 - w) * self.iy[j](x) + w * self.iy[j + 1](x)
        return out
