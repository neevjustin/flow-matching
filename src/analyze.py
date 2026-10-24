"""Analysis layer: per-particle diagnostics, mode-assignment flips, figures.

The mean transport-map error M can be small while a minority of particles are
badly misrouted. Since a flux-null perturbation preserves every marginal, the
FRACTION of samples landing in each mode is unchanged while WHICH noise sample
goes to WHICH mode can change. The mode-assignment flip rate therefore makes
"same endpoint distribution, different transport map" concrete and
interpretable at the level of individual samples.
"""

import numpy as np
import torch


def mode_assign(x, mu):
    """Nearest-mode index for each row of x."""
    d = np.linalg.norm(np.asarray(x)[:, None, :] - np.asarray(mu)[None, :, :], axis=2)
    return d.argmin(1), d.min(1)


def mode_flip_stats(phi_a, phi_b, mu):
    """Compare two transport maps on the same x0."""
    ka, da = mode_assign(phi_a, mu)
    kb, db = mode_assign(phi_b, mu)
    flip = (ka != kb)
    # per-mode occupancy: the marginal-level quantity (should match closely)
    K = len(mu)
    occ_a = np.bincount(ka, minlength=K) / len(ka)
    occ_b = np.bincount(kb, minlength=K) / len(kb)
    return dict(
        flip_rate=float(flip.mean()),
        n_flip=int(flip.sum()),
        occupancy_a=occ_a.tolist(),
        occupancy_b=occ_b.tolist(),
        occupancy_tv=float(0.5 * np.abs(occ_a - occ_b).sum()),
        dist_to_mode_a=float(da.mean()),
        dist_to_mode_b=float(db.mean()),
    )


def per_particle_error(phi_a, phi_b):
    e = np.linalg.norm(np.asarray(phi_a) - np.asarray(phi_b), axis=1)
    return dict(mean=float(e.mean()), median=float(np.median(e)),
                p90=float(np.quantile(e, 0.90)), p99=float(np.quantile(e, 0.99)),
                max=float(e.max()), err=e)


def radial_profile(x0, err, nbin=12):
    """Mean error vs |x0|, to see whether misrouting is concentrated in the
    low-density core / separatrix region of the base distribution."""
    r = np.linalg.norm(np.asarray(x0), axis=1)
    qs = np.quantile(r, np.linspace(0, 1, nbin + 1))
    out = []
    for i in range(nbin):
        m = (r >= qs[i]) & (r <= qs[i + 1])
        if m.sum() > 0:
            out.append((float(0.5 * (qs[i] + qs[i + 1])), float(err[m].mean()), int(m.sum())))
    return out


def density_at_start(gmm, x0, t=0.0):
    return gmm.p_t(t, torch.as_tensor(x0)).numpy()
