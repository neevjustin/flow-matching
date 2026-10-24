"""ODE integration of velocity fields over batches of particles.

Two integrators are provided deliberately:

  * ``integrate_adaptive`` -- scipy DOP853/RK45 with user-set rtol/atol. Used
    for the ground-truth flow map and for the solver-tolerance gate (H4).
  * ``integrate_fixed``    -- Euler / Heun / RK4 with a fixed number of steps.
    Used only to quantify how much of any observed effect is solver-induced.

The distinction matters: H4 requires showing that trajectory differences do
NOT shrink as the integrator is tightened, which is only meaningful if the
integrator error can be driven far below the effect size.
"""

import numpy as np
from scipy.integrate import solve_ivp


def integrate_adaptive(field, x0, t0=0.0, t1=1.0, rtol=1e-10, atol=1e-12,
                       method="DOP853", t_eval=None):
    """Integrate dx/dt = field(t, x) for a batch.

    field : callable (t: float, x: (n,d) float64 ndarray) -> (n,d) ndarray
    x0    : (n, d)
    Returns (n,d) endpoint, or (n,d,len(t_eval)) if t_eval is given, plus info.
    """
    x0 = np.asarray(x0, dtype=np.float64)
    n, d = x0.shape

    def rhs(t, y):
        return field(float(t), y.reshape(n, d)).reshape(-1)

    sol = solve_ivp(rhs, (t0, t1), x0.reshape(-1), method=method,
                    rtol=rtol, atol=atol, t_eval=t_eval, dense_output=False)
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")
    info = dict(nfev=int(sol.nfev), nsteps=int(sol.t.size), method=method,
                rtol=rtol, atol=atol)
    if t_eval is None:
        return sol.y[:, -1].reshape(n, d), info
    return sol.y.reshape(n, d, -1), info


def integrate_fixed(field, x0, t0=0.0, t1=1.0, n_steps=100, scheme="rk4"):
    """Fixed-step integration. Returns (n,d) endpoint and info."""
    x = np.asarray(x0, dtype=np.float64).copy()
    h = (t1 - t0) / n_steps
    nfev = 0
    for k in range(n_steps):
        t = t0 + k * h
        if scheme == "euler":
            x = x + h * field(t, x)
            nfev += 1
        elif scheme == "heun":
            k1 = field(t, x)
            k2 = field(t + h, x + h * k1)
            x = x + 0.5 * h * (k1 + k2)
            nfev += 2
        elif scheme == "rk4":
            k1 = field(t, x)
            k2 = field(t + 0.5 * h, x + 0.5 * h * k1)
            k3 = field(t + 0.5 * h, x + 0.5 * h * k2)
            k4 = field(t + h, x + h * k3)
            x = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            nfev += 4
        else:
            raise ValueError(scheme)
    return x, dict(nfev=nfev, n_steps=n_steps, scheme=scheme)


def torch_field(fn, dtype=None):
    """Wrap a torch velocity field as a numpy callable for the integrators."""
    import torch

    def f(t, x):
        with torch.no_grad():
            xt = torch.as_tensor(x, dtype=dtype) if dtype is not None else torch.as_tensor(x)
            out = fn(t, xt)
        return out.detach().cpu().numpy().astype(np.float64)

    return f
