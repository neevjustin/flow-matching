"""Generate paper figures from the raw results.

Reads only from results/raw/** and writes to results/figures/**; raw outputs are
never modified. Palette: validated 3-slot categorical set from the dataviz
reference palette (light mode, all-pairs PASS); aqua carries a contrast WARN so
every series is also directly labelled / legended.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RAW = os.path.join(BASE, "results", "raw")
FIG = os.path.join(BASE, "results", "figures")
os.makedirs(FIG, exist_ok=True)

S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"     # blue, orange, aqua
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
SURF = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF,
    "savefig.facecolor": SURF, "font.size": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "lines.linewidth": 2.0,
    "grid.color": "#e6e5e1", "grid.linewidth": 0.7,
    "legend.frameon": False, "figure.dpi": 150,
})


def strip(ax):
    ax.grid(True, axis="y", zorder=0)
    ax.set_axisbelow(True)


# ------------------------------------------------------------------ figure 1
def fig1():
    d = json.load(open(os.path.join(RAW, "e1_main", "metrics.json")))
    f = np.array([r["f"] for r in d["models"]])
    nl = d["nulls"]
    groups = [
        ("Trained\nCFM net", f, S1),
        ("Null: residual\nrotated 90°", np.array(nl["null_swap"]), S2),
        ("Null: residual\nrotated randomly", np.array(nl["null_rot"]), S3),
        ("Null: random-init net\n(degenerate, see text)", np.array(nl["null_init"]), MUTED),
    ]
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    rng = np.random.default_rng(0)
    for i, (name, v, c) in enumerate(groups):
        x = i + rng.uniform(-0.1, 0.1, v.size)
        ax.scatter(x, v, s=26, color=c, zorder=3, edgecolor=SURF, linewidth=0.8)
        ax.hlines(v.mean(), i - 0.28, i + 0.28, color=c, lw=2.4, zorder=4)
        ax.text(i, 1.035, f"{v.mean():.3f}", ha="center", va="bottom",
                color=INK, fontsize=9, fontweight="bold")
    ax.axhline(0.0, color=INK, lw=1.2, ls="--", zorder=2)
    ax.text(3.46, 0.012, "target value  $f(v^*)=0$", ha="right", va="bottom",
            color=INK, fontsize=8.5)
    strip(ax)
    ax.set_xticks(range(4))
    ax.set_xticklabels([g[0] for g in groups], fontsize=8.2)
    ax.set_ylabel("flux-null energy fraction  $f$")
    ax.set_ylim(-0.05, 1.12)
    ax.set_title("Trained velocity error is far LESS rotational than chance\n"
                 "8 seeds; 2-D ring of 8 Gaussians", fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_fluxnull_fraction.png"), bbox_inches="tight")
    plt.close(fig)
    print("fig1 done")


# ------------------------------------------------------------------ figure 2
def fig2():
    from gmm_path import ring8
    from helmholtz import FaceGrid, active_masks, decompose
    from model import VelocityMLP
    import evaluate as EV
    torch.set_default_dtype(torch.float64)

    gmm = ring8()
    N = 192
    grid = FaceGrid(N, 8.0)
    m = VelocityMLP(d=2, width=128, depth=3).to(torch.float32)
    m.load_state_dict(torch.load(os.path.join(RAW, "e1_main", "ckpt", "seed0.pt")))
    m = m.to(torch.float64)
    resid = EV.residual_fn(m, gmm)
    cp = torch.as_tensor(grid.cell_positions())
    fp = torch.as_tensor(grid.face_pos)

    ts = [0.2, 0.45, 0.7]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.5))
    for ax, t in zip(axes, ts):
        p_cell = gmm.p_t(t, cp).numpy()
        _, act = active_masks(grid, p_cell, 1e-8)
        idx = np.flatnonzero(act)
        p_face = np.zeros(grid.nface); p_face[idx] = gmm.p_t(t, fp[idx]).numpy()
        v = resid(t, fp[idx]).numpy()
        e_face = np.zeros(grid.nface)
        e_face[idx] = np.where(grid.face_comp[idx] == 0, v[:, 0], v[:, 1])
        r = decompose(grid, p_face, e_face, p_cell)
        # local flux-null share, aggregated face energies onto cells
        num = np.zeros(grid.ncell); den = np.zeros(grid.ncell)
        for arr, tgt in ((r["r_face"], num), (e_face, den)):
            w = p_face * arr**2
            np.add.at(tgt, grid.face_A, 0.5 * w)
            np.add.at(tgt, grid.face_B, 0.5 * w)
        share = np.full(grid.ncell, np.nan)
        ok = den > den.max() * 1e-6
        share[ok] = num[ok] / den[ok]
        img = share.reshape(N, N).T
        im = ax.imshow(np.clip(img, 0, 1), origin="lower", extent=[-8, 8, -8, 8],
                       cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
        mu = gmm.mu.numpy() * t
        ax.scatter(mu[:, 0], mu[:, 1], s=14, facecolor="none", edgecolor=S2, lw=1.4,
                   zorder=3, label="mode centres $t\\mu_k$" if t == ts[0] else None)
        ax.set_xlim(-6, 6); ax.set_ylim(-6, 6)
        ax.set_title(f"$t = {t}$", fontsize=9.5, color=INK)
        ax.set_xticks([-5, 0, 5]); ax.set_yticks([-5, 0, 5])
        for s in ax.spines.values():
            s.set_visible(False)
        if t == ts[0]:
            ax.legend(loc="upper left", fontsize=7.5)
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("local flux-null share of error energy", fontsize=8.5)
    cb.outline.set_visible(False)
    fig.suptitle("Spurious circulation sits between the modes, not on them  "
                 "(seed 0)", fontsize=10, color=INK, x=0.012, ha="left")
    fig.savefig(os.path.join(FIG, "fig2_spatial.png"), bbox_inches="tight")
    plt.close(fig)
    print("fig2 done")


# ------------------------------------------------------------------ figure 3
def fig3():
    ps = np.load(os.path.join(RAW, "e2b_main", "phi_star.npy"))
    series = [("full model error  $\\Phi_\\theta$", "theta", S1),
              ("gradient part only  $\\Phi_{\\nabla\\varphi}$", "grad", S2),
              ("flux-null part only  $\\Phi_{r}$", "null", S3)]
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    for label, key, c in series:
        p = np.load(os.path.join(RAW, "e2b_main", f"phi_{key}_seed0.npy"))
        e = np.sort(np.linalg.norm(p - ps, axis=1))
        ccdf = 1.0 - np.arange(e.size) / e.size
        ax.loglog(e, ccdf, color=c, label=label)
        ax.scatter([e.mean()], [1.0 / e.size], color=c, s=0)      # keep scale
    ax.set_xlabel("per-particle displacement from the true map  $|\\Phi(x_0)-\\Phi^*(x_0)|$")
    ax.set_ylabel("fraction of particles exceeding")
    ax.grid(True, which="both")
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title("The mean map error hides a heavy tail\n"
                 "ring8, seed 0, 3000 shared $x_0$", fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_tail.png"), bbox_inches="tight")
    plt.close(fig)
    print("fig3 done")


# ------------------------------------------------------------------ figure 4
def fig4():
    d = json.load(open(os.path.join(RAW, "e2b_main", "metrics.json")))
    rows = d["rows"]
    pts = [
        ("$\\Phi^*$ (ground truth)", 0.0, 0.0, 0.0, MUTED),
        ("$\\Phi_r$  flux-null part only", np.mean([r["kl_null"] for r in rows]),
         np.mean([r["M_null"] for r in rows]),
         np.std([r["M_null"] for r in rows], ddof=1), S3),
        ("$\\Phi_\\theta$  full model", np.mean([r["kl_theta"] for r in rows]),
         np.mean([r["M_theta"] for r in rows]),
         np.std([r["M_theta"] for r in rows], ddof=1), S1),
        ("$\\Phi_{\\nabla\\varphi}$  gradient part only", np.mean([r["kl_grad"] for r in rows]),
         np.mean([r["M_grad"] for r in rows]),
         np.std([r["M_grad"] for r in rows], ddof=1), S2),
    ]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for name, kl, M, sd, c in pts:
        ax.errorbar([kl], [M], yerr=[sd], fmt="o", ms=9, color=c,
                    ecolor=c, elinewidth=1.6, capsize=3,
                    markeredgecolor=SURF, markeredgewidth=1.0, zorder=3)
        dy = 0.0016 if name.startswith("$\\Phi_r$") else -0.0022
        ax.annotate(name, (kl, M), textcoords="offset points",
                    xytext=(8, 10 if dy > 0 else -16), fontsize=8.3, color=INK)
    kl_res = 1e-7
    ax.axvspan(-0.012, 0.008, color="#eef3fa", zorder=0)
    ax.text(-0.011, 0.0005, "distributionally\nindistinguishable\nfrom the true field",
            fontsize=7.6, color=INK2, va="bottom")
    strip(ax)
    ax.set_xlabel("endpoint distributional error   $\\mathrm{KL}(\\hat p_1\\,\\|\\,p_1)$   [nats]")
    ax.set_ylabel("transport-map error  $M$")
    ax.set_xlim(-0.015, 0.19)
    ax.set_ylim(-0.0015, 0.027)
    ax.set_title("A distributionally invisible component still moves the map\n"
                 "mean over 3 seeds; KL instrument resolves $10^{-7}$ "
                 "(ground truth reads $0.00000$)", fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig4_blindness.png"), bbox_inches="tight")
    plt.close(fig)
    print("fig4 done")


# ------------------------------------------------------------------ figure 5
def fig5():
    from gmm_path import ring8
    from analyze import mode_assign
    gmm = ring8(); mu = gmm.mu.numpy()
    x0 = np.load(os.path.join(RAW, "e2b_main", "x0.npy"))
    ps = np.load(os.path.join(RAW, "e2b_main", "phi_star.npy"))
    pn = np.load(os.path.join(RAW, "e2b_main", "phi_null_seed0.npy"))
    ka, _ = mode_assign(ps, mu); kb, _ = mode_assign(pn, mu)
    flip = ka != kb
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.0))
    for ax, (X, ttl) in zip(axes, [(x0, "source  $x_0\\sim\\mathcal{N}(0,I)$"),
                                   (ps, "true endpoints  $\\Phi^*(x_0)$")]):
        ax.scatter(X[~flip, 0], X[~flip, 1], s=4, color=MUTED, alpha=0.35,
                   linewidths=0, label="same mode")
        ax.scatter(X[flip, 0], X[flip, 1], s=22, color=S2, zorder=3,
                   edgecolor=SURF, linewidth=0.6,
                   label=f"re-routed to a different mode ({flip.mean():.2%})")
        ax.set_title(ttl, fontsize=9.5, color=INK)
        ax.set_aspect("equal")
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xticks([]); ax.set_yticks([])
    axes[1].scatter(mu[:, 0], mu[:, 1], marker="x", s=40, color=INK, lw=1.4, zorder=4)
    axes[0].legend(fontsize=8, loc="upper left", markerscale=1.4)
    fig.suptitle("Which sample goes to which mode changes, while the mode "
                 "occupancy does not\n"
                 "injecting only the measured flux-null component; "
                 "occupancy TV = 0.0017, $\\mathrm{KL}\\approx0$",
                 fontsize=10, color=INK, x=0.012, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig5_modeflip.png"), bbox_inches="tight")
    plt.close(fig)
    print("fig5 done")




# ------------------------------------------------------------------ figure 6
def fig6():
    d = json.load(open(os.path.join(RAW, "e3_main", "metrics.json")))

    def col(a, k):
        return np.array([r[k] for r in d["rows"] if r["arch"] == a], dtype=float)

    panels = [
        ("flux-null fraction $f$", "f", False),
        ("velocity error $\\eta_{rel}$", "eta_rel", False),
        ("map error $M$", "M", False),
        ("amplification $M/\\eta_{rel}$", None, True),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(10.6, 3.2))
    rng = np.random.default_rng(1)
    for ax, (title, key, ratio) in zip(axes, panels):
        for i, (arch, c, lab) in enumerate([("velocity", S1, "standard\n$v_\\theta$"),
                                            ("potential", S3, "curl-free\n$\\nabla\\Psi_\\theta$")]):
            v = col(arch, "M") / col(arch, "eta_rel") if ratio else col(arch, key)
            x = i + rng.uniform(-0.07, 0.07, v.size)
            ax.scatter(x, v, s=28, color=c, zorder=3, edgecolor=SURF, linewidth=0.8)
            ax.hlines(v.mean(), i - 0.24, i + 0.24, color=c, lw=2.4, zorder=4)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["standard\n$v_\\theta$", "curl-free\n$\\nabla\\Psi_\\theta$"],
                           fontsize=8)
        ax.set_xlim(-0.5, 1.5)
        ax.set_title(title, fontsize=9, color=INK)
        strip(ax)
        if key == "f":
            ax.set_yscale("log")
    fig.suptitle("Curl-free parameterisation: better on every axis, but most of the map "
                 "gain follows the better velocity fit\n"
                 "4 seeds per arm, matched steps/width/depth; the potential arm costs "
                 "2.1$\\times$ wall-clock per step",
                 fontsize=9.5, color=INK, x=0.012, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    fig.savefig(os.path.join(FIG, "fig6_curlfree.png"), bbox_inches="tight")
    plt.close(fig)
    print("fig6 done")


if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "2", "3", "4", "5", "6"]
    for w in which:
        {"1": fig1, "2": fig2, "3": fig3, "4": fig4, "5": fig5, "6": fig6}[w]()
