"""Closed-form probability path and marginal velocity field for
Gaussian-mixture data under a Gaussian base with independent coupling and a
linear interpolant.

Implements theory.md Sec. 4, Eqs. (12)-(14).

Setup
-----
    x_0 ~ N(0, I_d)
    x_1 ~ sum_k pi_k N(mu_k, Sigma_k)     (independent of x_0)
    x_t = (1 - t) x_0 + t x_1
    u   = x_1 - x_0                        (CFM regression target)

Conditional on component k, (x_t, u) is jointly Gaussian with

    m_k(t) = t mu_k
    S_k(t) = (1-t)^2 I + t^2 Sigma_k
    C_k(t) = -(1-t) I + t Sigma_k          (= Cov(x_t, u | k), symmetric)
    Cov(u | k) = I + Sigma_k

giving  v*(t,x) = sum_k gamma_k(t,x) b_k(t,x)  with
        b_k = mu_k + C_k S_k^{-1} (x - m_k)
        gamma_k = posterior responsibility under p_t.

Everything is done in float64. These are the reference quantities against
which trained networks are measured, so they must not carry float32 error.
"""

import math
import torch

LOG2PI = math.log(2.0 * math.pi)


class GMMPath:
    def __init__(self, mu, Sigma, pi=None, dtype=torch.float64):
        """mu: (K, d); Sigma: (K, d, d) SPD; pi: (K,) mixing weights."""
        self.dtype = dtype
        self.mu = torch.as_tensor(mu, dtype=dtype)
        self.Sigma = torch.as_tensor(Sigma, dtype=dtype)
        K, d = self.mu.shape
        assert self.Sigma.shape == (K, d, d)
        if pi is None:
            pi = torch.full((K,), 1.0 / K, dtype=dtype)
        self.pi = torch.as_tensor(pi, dtype=dtype)
        self.pi = self.pi / self.pi.sum()
        self.K, self.d = K, d
        self.logpi = torch.log(self.pi)
        self.I = torch.eye(d, dtype=dtype)

    # ---------------- per-component path quantities ----------------

    def _mSC(self, t):
        """Return m_k(t) (K,d), S_k(t) (K,d,d), C_k(t) (K,d,d) for scalar t."""
        t = float(t)
        m = t * self.mu
        S = (1.0 - t) ** 2 * self.I + t**2 * self.Sigma
        C = -(1.0 - t) * self.I + t * self.Sigma
        return m, S, C

    # ---------------- density ----------------

    def log_component_densities(self, t, x):
        """(n, K) log N(x; m_k(t), S_k(t))."""
        m, S, _ = self._mSC(t)
        L = torch.linalg.cholesky(S)                      # (K,d,d)
        diff = x[:, None, :] - m[None, :, :]              # (n,K,d)
        # solve L z = diff  ->  quad = |z|^2
        z = torch.linalg.solve_triangular(
            L[None].expand(x.shape[0], -1, -1, -1).reshape(-1, self.d, self.d),
            diff.reshape(-1, self.d, 1),
            upper=False,
        ).reshape(x.shape[0], self.K, self.d)
        quad = (z**2).sum(-1)                             # (n,K)
        logdet = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)  # (K,)
        return -0.5 * (quad + logdet[None, :] + self.d * LOG2PI)

    def log_p_t(self, t, x):
        return torch.logsumexp(self.logpi[None, :] + self.log_component_densities(t, x), dim=1)

    def p_t(self, t, x):
        return torch.exp(self.log_p_t(t, x))

    def posterior(self, t, x):
        """(n, K) responsibilities gamma_k(t, x)."""
        lj = self.logpi[None, :] + self.log_component_densities(t, x)
        return torch.softmax(lj, dim=1)

    # ---------------- velocity ----------------

    def b_components(self, t, x):
        """(n, K, d) component conditional means b_k(t,x) = mu_k + C_k S_k^{-1}(x - m_k)."""
        m, S, C = self._mSC(t)
        A = torch.linalg.solve(S, C)                      # (K,d,d) = S^{-1} C
        diff = x[:, None, :] - m[None, :, :]              # (n,K,d)
        # b = mu + (C S^{-1}) diff = mu + (S^{-1} C)^T diff  (C, S symmetric)
        corr = torch.einsum("kij,nkj->nki", A.transpose(-1, -2), diff)
        return self.mu[None, :, :] + corr

    def v_star(self, t, x):
        """(n, d) marginal velocity field v*(t, x). theory.md Eq. (13)."""
        g = self.posterior(t, x)                          # (n,K)
        b = self.b_components(t, x)                       # (n,K,d)
        return (g[:, :, None] * b).sum(1)

    def grad_log_p_t(self, t, x):
        """(n, d) score  grad_x log p_t(x) = -sum_k gamma_k S_k^{-1}(x - m_k)."""
        m, S, _ = self._mSC(t)
        g = self.posterior(t, x)                          # (n,K)
        diff = x[:, None, :] - m[None, :, :]              # (n,K,d)
        Sinv = torch.linalg.inv(S)                        # (K,d,d)
        sc = -torch.einsum("kij,nkj->nki", Sinv, diff)    # (n,K,d)
        return (g[:, :, None] * sc).sum(1)

    def loss_floor(self, t, x):
        """(n,) E[||u - v*||^2 | x_t = x]. theory.md Eq. (14)."""
        m, S, C = self._mSC(t)
        g = self.posterior(t, x)
        b = self.b_components(t, x)
        v = (g[:, :, None] * b).sum(1)
        # within-component conditional covariance trace
        cond_cov = (self.I + self.Sigma) - C @ torch.linalg.solve(S, C)   # (K,d,d)
        tr = torch.diagonal(cond_cov, dim1=-2, dim2=-1).sum(-1)           # (K,)
        between = ((b - v[:, None, :]) ** 2).sum(-1)                      # (n,K)
        return (g * (tr[None, :] + between)).sum(1)

    # ---------------- sampling ----------------

    def sample_x1(self, n, generator=None):
        k = torch.multinomial(self.pi, n, replacement=True, generator=generator)
        L = torch.linalg.cholesky(self.Sigma)             # (K,d,d)
        z = torch.randn(n, self.d, dtype=self.dtype, generator=generator)
        return self.mu[k] + torch.einsum("nij,nj->ni", L[k], z)

    def sample_x_t(self, n, t, generator=None):
        x0 = torch.randn(n, self.d, dtype=self.dtype, generator=generator)
        x1 = self.sample_x1(n, generator=generator)
        return (1.0 - t) * x0 + t * x1, x0, x1


# ---------------- standard testbed configurations ----------------

def ring8(sigma=0.3, radius=4.0, dtype=torch.float64):
    """8 equal-weight isotropic Gaussians on a ring. Primary testbed."""
    ang = torch.arange(8, dtype=dtype) * (2 * math.pi / 8)
    mu = torch.stack([radius * torch.cos(ang), radius * torch.sin(ang)], dim=1)
    Sigma = (sigma**2) * torch.eye(2, dtype=dtype)[None].repeat(8, 1, 1)
    return GMMPath(mu, Sigma, dtype=dtype)


def two_modes_imbalanced(sep=5.0, sigma=0.35, w=0.9, dtype=torch.float64):
    """Two modes with a 9:1 weight imbalance and a wide low-density gap."""
    mu = torch.tensor([[-sep / 2, 0.0], [sep / 2, 0.0]], dtype=dtype)
    Sigma = (sigma**2) * torch.eye(2, dtype=dtype)[None].repeat(2, 1, 1)
    return GMMPath(mu, Sigma, pi=torch.tensor([w, 1 - w], dtype=dtype), dtype=dtype)


def anisotropic_pair(sep=4.5, dtype=torch.float64):
    """Two strongly anisotropic, mutually rotated modes: v* has strong shear."""
    mu = torch.tensor([[-sep / 2, 0.0], [sep / 2, 0.0]], dtype=dtype)
    def rot(a):
        c, s = math.cos(a), math.sin(a)
        return torch.tensor([[c, -s], [s, c]], dtype=dtype)
    D = torch.diag(torch.tensor([0.8**2, 0.15**2], dtype=dtype))
    S0 = rot(0.6) @ D @ rot(0.6).T
    S1 = rot(-0.6) @ D @ rot(-0.6).T
    return GMMPath(mu, torch.stack([S0, S1]), dtype=dtype)


def moons_gmm(n_per=24, sigma=0.12, dtype=torch.float64):
    """Two-moons approximated as a Gaussian mixture along two arcs.

    Using a GMM (rather than the usual sampler) keeps p_t and v* in closed
    form while retaining the two-moons geometry: curved, close-approaching,
    non-convex supports with a thin low-density channel between them.
    """
    th = torch.linspace(0, math.pi, n_per, dtype=dtype)
    upper = torch.stack([torch.cos(th), torch.sin(th)], dim=1)
    lower = torch.stack([1.0 - torch.cos(th), -torch.sin(th) + 0.3], dim=1)
    mu = 2.0 * torch.cat([upper, lower], dim=0)
    Sigma = (sigma**2) * torch.eye(2, dtype=dtype)[None].repeat(mu.shape[0], 1, 1)
    return GMMPath(mu, Sigma, dtype=dtype)


REGISTRY = {
    "ring8": ring8,
    "imbalanced2": two_modes_imbalanced,
    "aniso2": anisotropic_pair,
    "moons": moons_gmm,
}
