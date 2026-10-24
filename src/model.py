"""Velocity-field network and CFM training.

Deliberately small and standard: the point of the experiment is to measure the
geometry of a *typical* CFM model's residual error, not to propose an
architecture. Anything unusual here would be a confounder.
"""

import math

import torch
import torch.nn as nn


class FourierTime(nn.Module):
    """Sinusoidal time embedding with fixed (non-learned) frequencies."""

    def __init__(self, n_freq=16, max_freq=32.0):
        super().__init__()
        f = torch.exp(torch.linspace(0.0, math.log(max_freq), n_freq))
        self.register_buffer("freq", f)

    def forward(self, t):
        a = t[:, None] * self.freq[None, :] * 2 * math.pi
        return torch.cat([torch.sin(a), torch.cos(a), t[:, None]], dim=1)

    @property
    def dim(self):
        return 2 * self.freq.numel() + 1


class VelocityMLP(nn.Module):
    def __init__(self, d=2, width=128, depth=3, n_freq=16, act="silu"):
        super().__init__()
        self.temb = FourierTime(n_freq)
        A = {"silu": nn.SiLU, "gelu": nn.GELU, "tanh": nn.Tanh, "relu": nn.ReLU}[act]
        layers, din = [], d + self.temb.dim
        for _ in range(depth):
            layers += [nn.Linear(din, width), A()]
            din = width
        layers += [nn.Linear(din, d)]
        self.net = nn.Sequential(*layers)
        self.d = d

    def forward(self, t, x):
        """t: (n,) or scalar-expandable; x: (n,d)."""
        if not torch.is_tensor(t):
            t = torch.full((x.shape[0],), float(t), dtype=x.dtype, device=x.device)
        elif t.dim() == 0:
            t = t.expand(x.shape[0])
        return self.net(torch.cat([x, self.temb(t)], dim=1))

    def field_at(self, t, x):
        with torch.no_grad():
            return self(t, x)


class PotentialMLP(nn.Module):
    """Curl-free velocity field: v_theta(t,x) = grad_x Psi_theta(t,x).

    Motivation (theory.md Sec. 2.5): for a Gaussian base, independent coupling
    and a linear interpolant, the CFM target satisfies
        v*(t,x) = (x + (1-t) grad log p_t(x)) / t = grad_x Psi*_t(x),
    i.e. the target is EXACTLY a gradient field. Parameterising v as a gradient
    is therefore a zero-bias restriction, not an approximation -- it removes the
    entire flux-null subspace, which can only contain error.

    Architecture is matched to VelocityMLP except for a scalar head, so the
    parameter counts agree to within ~130 parameters at width 128.
    NOTE: each forward pass costs an extra autograd pass, and training needs a
    double backward. The wall-clock cost is reported honestly in E3 alongside
    the matched-step comparison.
    """

    def __init__(self, d=2, width=128, depth=3, n_freq=16, act="silu"):
        super().__init__()
        self.temb = FourierTime(n_freq)
        A = {"silu": nn.SiLU, "gelu": nn.GELU, "tanh": nn.Tanh, "relu": nn.ReLU}[act]
        layers, din = [], d + self.temb.dim
        for _ in range(depth):
            layers += [nn.Linear(din, width), A()]
            din = width
        layers += [nn.Linear(din, 1)]
        self.net = nn.Sequential(*layers)
        self.d = d

    def potential(self, t, x):
        if not torch.is_tensor(t):
            t = torch.full((x.shape[0],), float(t), dtype=x.dtype, device=x.device)
        elif t.dim() == 0:
            t = t.expand(x.shape[0])
        return self.net(torch.cat([x, self.temb(t)], dim=1)).squeeze(-1)

    def forward(self, t, x):
        need_graph = torch.is_grad_enabled()
        with torch.enable_grad():
            xg = x if (x.requires_grad and need_graph) else x.detach().requires_grad_(True)
            psi = self.potential(t, xg)
            (g,) = torch.autograd.grad(psi.sum(), xg, create_graph=need_graph)
        return g if need_graph else g.detach()

    def field_at(self, t, x):
        return self(t, x).detach()


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def train_cfm(gmm, seed=0, width=128, depth=3, steps=20000, batch=4096,
              lr=2e-3, lam_div=0.0, dtype=torch.float64, log_every=4000,
              act="silu", weight_decay=0.0, t_eps=0.0, arch="velocity"):
    """Train a CFM velocity field on a FRESH i.i.d. sample at every step.

    The fresh-sample stream is the key design choice (theory.md Sec. 6): it
    makes the population v* the true regression target, so the residual
    e = v_theta - v* is pure approximation+optimisation error with no
    memorisation/generalisation confound.

    lam_div > 0 adds the conditional divergence-matching penalty of
    Huang et al. (ICML 2025) -- used only for hypothesis H3.
    """
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(100_000 + seed)
    Cls = {"velocity": VelocityMLP, "potential": PotentialMLP}[arch]
    model = Cls(d=gmm.d, width=width, depth=depth, act=act).to(dtype)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    hist = []

    for step in range(steps):
        # sample in the GMM's own (float64) dtype, then cast to the training
        # dtype -- keeps the data stream identical across training precisions
        t64 = torch.rand(batch, dtype=gmm.dtype, generator=gen)
        if t_eps > 0:
            t64 = t_eps + (1.0 - 2 * t_eps) * t64
        x0 = torch.randn(batch, gmm.d, dtype=gmm.dtype, generator=gen)
        x1 = gmm.sample_x1(batch, generator=gen)
        t = t64.to(dtype)
        xt = ((1.0 - t64)[:, None] * x0 + t64[:, None] * x1).to(dtype)
        u = (x1 - x0).to(dtype)

        v = model(t, xt)
        loss = ((v - u) ** 2).sum(1).mean()

        if lam_div > 0.0:
            # L_CDM of Huang et al.: |div(u_c - v) + (u_c - v).grad log p_t(x|x1)|
            # For the linear interpolant the conditional path is
            # x_t | x_1 ~ N(t x_1, (1-t)^2 I), so grad log p_t(.|x1) =
            # -(x_t - t x_1)/(1-t)^2, and div(u_c) = 0 since u_c = x_1 - x_0
            # is constant in x along the conditional path.
            xt_g = xt.detach().requires_grad_(True)
            vg = model(t, xt_g)
            div_v = torch.zeros(batch, dtype=dtype)
            for i in range(gmm.d):
                gi = torch.autograd.grad(vg[:, i].sum(), xt_g, create_graph=True)[0]
                div_v = div_v + gi[:, i]
            score_c = -(xt_g - t[:, None] * x1) / ((1.0 - t)[:, None] ** 2).clamp(min=1e-8)
            diff = u - vg
            cdm = (-div_v + (diff * score_c).sum(1)).abs().mean()
            loss = loss + lam_div * cdm

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()

        if log_every and (step % log_every == 0 or step == steps - 1):
            hist.append((step, float(loss.detach())))

    return model, hist
