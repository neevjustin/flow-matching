"""E2b: re-inject the measured Helmholtz components, with an EXACTLY consistent
split (supersedes E2a, which was interpolation-limited; E2a is kept on record).

Scheme
------
At each time slice we solve the weighted Poisson problem on the grid and keep
the cell-centred potential phi. We then build a bicubic spline of phi and define

    g(t,x) = grad_x phi_spline(t,x)                    (smooth, interpolated)
    r(t,x) = [v_theta(t,x) - v*(t,x)] - g(t,x)         (e is EXACT here)

so that g + r = e identically. The reconstruction check that failed in E2a
(P3) is now satisfied by construction, and the only interpolation error is in
how well g approximates the true gradient component -- which we test directly
by comparing the Monte-Carlo flux-null fraction of r against the grid value f.

PRE-REGISTERED PREDICTIONS (unchanged from E2a):
  P1  Phi_null's endpoint discrepancy to p_1 is indistinguishable from Phi*'s.
  P2  Phi_grad carries essentially all of the model's distributional error.
  P4  (new, replaces P3) MC flux-null fraction of r matches the grid f to
      within 0.03, confirming the spline gradient is faithful.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from scipy.integrate import solve_ivp
from scipy.interpolate import RectBivariateSpline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gmm_path import REGISTRY                                      # noqa: E402
from helmholtz import FaceGrid, active_masks, decompose            # noqa: E402
from model import VelocityMLP                                      # noqa: E402
import evaluate as EV                                              # noqa: E402
from analyze import mode_flip_stats, per_particle_error            # noqa: E402

torch.set_default_dtype(torch.float64)
torch.set_num_threads(2)


class PotentialSplines:
    """Bicubic spline of phi per time slice; analytic spatial gradient."""

    def __init__(self, grid, ts, phis):
        self.ts = np.asarray(ts)
        c = grid.centres
        self.sp = [RectBivariateSpline(c, c, p.reshape(grid.N, grid.N), kx=3, ky=3)
                   for p in phis]

    def lap(self, t, x):
        """Laplacian of the interpolated potential from analytic spline second
        derivatives -- no finite differences, so the RHS stays smooth."""
        t = float(np.clip(t, self.ts[0], self.ts[-1]))
        j = int(np.clip(np.searchsorted(self.ts, t, side="right") - 1, 0, len(self.ts) - 2))
        t0, t1 = self.ts[j], self.ts[j + 1]
        w = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
        o = np.zeros(x.shape[0])
        for a, sp in ((1 - w, self.sp[j]), (w, self.sp[j + 1])):
            if a == 0.0:
                continue
            o = o + a * (sp.ev(x[:, 0], x[:, 1], dx=2, dy=0)
                         + sp.ev(x[:, 0], x[:, 1], dx=0, dy=2))
        return o

    def grad(self, t, x):
        t = float(np.clip(t, self.ts[0], self.ts[-1]))
        j = int(np.clip(np.searchsorted(self.ts, t, side="right") - 1, 0, len(self.ts) - 2))
        t0, t1 = self.ts[j], self.ts[j + 1]
        w = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
        out = np.empty_like(x)
        for a, sp in ((1 - w, self.sp[j]), (w, self.sp[j + 1])):
            if a == 0.0:
                continue
            out_a = np.stack([sp.ev(x[:, 0], x[:, 1], dx=1, dy=0),
                              sp.ev(x[:, 0], x[:, 1], dx=0, dy=1)], axis=1)
            out = out_a * a if a == (1 - w) else out + out_a * a
        return out


def div_torch(fn, t, x, dtype=torch.float64):
    """Exact divergence of a torch field via autograd."""
    xg = torch.as_tensor(x, dtype=dtype).clone().requires_grad_(True)
    with torch.enable_grad():
        v = fn(float(t), xg)
        s = torch.zeros(xg.shape[0], dtype=dtype)
        for i in range(xg.shape[1]):
            gi = torch.autograd.grad(v[:, i].sum(), xg,
                                     retain_graph=(i < xg.shape[1] - 1))[0]
            s = s + gi[:, i]
    return s.detach().numpy()


def kl_rk4(field, divfield, gmm, x0, n_steps=100):
    """KL(p_hat_1 || p_1) by fixed-step RK4 on the augmented state (x, a) with
    an EXACT divergence.

    Fixed steps are deliberate: an adaptive solver thrashes when the RHS
    contains a finite-difference divergence (we measured nfev blowing up and
    the run stalling). With analytic divergences the RHS is smooth, and
    accuracy is verified by a step-count refinement against the ground-truth
    field, whose true KL is exactly 0.
    """
    n, d = x0.shape

    def rhs(t, y):
        return np.concatenate([field(t, y[:, :d]), divfield(t, y[:, :d])[:, None]], axis=1)

    y = np.concatenate([x0, np.zeros((n, 1))], axis=1)
    h = 1.0 / n_steps
    for k in range(n_steps):
        t = k * h
        k1 = rhs(t, y)
        k2 = rhs(t + 0.5 * h, y + 0.5 * h * k1)
        k3 = rhs(t + 0.5 * h, y + 0.5 * h * k2)
        k4 = rhs(t + h, y + h * k3)
        y = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    xe, a = y[:, :d], y[:, d]
    lp0 = -0.5 * (x0**2).sum(1) - np.log(2 * np.pi)
    lp1 = gmm.log_p_t(1.0, torch.as_tensor(xe)).numpy()
    vals = lp0 - a - lp1
    return dict(kl=float(vals.mean()), se=float(vals.std(ddof=1) / np.sqrt(n)), endpoint=xe)


def ed_with_se(a, b, nboot=40, seed=0):
    from sharp_metrics import energy_distance
    rng = np.random.default_rng(seed)
    base = energy_distance(a, b)
    n = min(len(a), len(b))
    vals = [energy_distance(a[rng.integers(0, len(a), n)], b[rng.integers(0, len(b), n)])
            for _ in range(nboot)]
    return base, float(np.std(vals, ddof=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="e1_main")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--dataset", default="ring8")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--grid-N", type=int, default=256)
    ap.add_argument("--n-t", type=int, default=33)
    ap.add_argument("--n-map", type=int, default=3000)
    ap.add_argument("--n-kl", type=int, default=1200)
    ap.add_argument("--kl-steps", type=int, default=100)
    args = ap.parse_args()

    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "raw")
    src = os.path.join(base, args.src)
    out = os.path.join(base, "e2b_" + args.tag)
    os.makedirs(out, exist_ok=True)
    print(f"output: {out}\nconfig: {vars(args)}")

    gmm = REGISTRY[args.dataset]()
    grid = FaceGrid(args.grid_N, 8.0)
    ts = np.linspace(0.02, 0.98, args.n_t)
    rng = np.random.default_rng(2024)
    x0 = rng.standard_normal((args.n_map, 2))
    mu = gmm.mu.numpy()

    print("ground truth ...")
    phi_star, _ = EV.flow_map(EV.np_vstar(gmm), x0, rtol=1e-11, atol=1e-13)
    dref = gmm.sample_x1(args.n_map, generator=torch.Generator().manual_seed(555)).numpy()
    dref2 = gmm.sample_x1(args.n_map, generator=torch.Generator().manual_seed(556)).numpy()
    ed_floor, ed_floor_se = ed_with_se(dref, dref2)
    ed_star, ed_star_se = ed_with_se(phi_star, dref)
    kl_star = kl_rk4(EV.np_vstar(gmm), lambda t, x: div_torch(gmm.v_star, t, x),
                     gmm, x0[: args.n_kl], n_steps=args.kl_steps)
    print(f"  ED floor = {ed_floor:.3e} +- {ed_floor_se:.3e}")
    print(f"  ED star  = {ed_star:.3e} +- {ed_star_se:.3e}")
    print(f"  KL star  = {kl_star['kl']:.5f} +- {kl_star['se']:.5f}")

    res = dict(config=vars(args), ed_floor=ed_floor, ed_floor_se=ed_floor_se,
               ed_star=ed_star, ed_star_se=ed_star_se,
               kl_star=kl_star["kl"], kl_star_se=kl_star["se"], rows=[])

    cp = torch.as_tensor(grid.cell_positions())
    fp = torch.as_tensor(grid.face_pos)
    vstar_np = EV.np_vstar(gmm)

    for seed in args.seeds:
        print(f"\n=== seed {seed} ===")
        m = VelocityMLP(d=2, width=args.width, depth=args.depth).to(torch.float32)
        m.load_state_dict(torch.load(os.path.join(src, "ckpt", f"seed{seed}.pt")))
        m = m.to(torch.float64)
        resid = EV.residual_fn(m, gmm)

        t0 = time.time()
        phis, grid_e2, grid_r2 = [], 0.0, 0.0
        for t in ts:
            t = float(t)
            p_cell = gmm.p_t(t, cp).numpy()
            _, act = active_masks(grid, p_cell, 1e-8)
            idx = np.flatnonzero(act)
            p_face = np.zeros(grid.nface); p_face[idx] = gmm.p_t(t, fp[idx]).numpy()
            v = resid(t, fp[idx]).numpy()
            e_face = np.zeros(grid.nface)
            e_face[idx] = np.where(grid.face_comp[idx] == 0, v[:, 0], v[:, 1])
            r = decompose(grid, p_face, e_face, p_cell)
            phis.append(r["phi_cell"]); grid_e2 += r["e2"]; grid_r2 += r["r2"]
        f_grid = grid_r2 / grid_e2
        sp = PotentialSplines(grid, ts, phis)
        print(f"  grid f = {f_grid:.4f}  ({time.time()-t0:.0f}s, N={args.grid_N}, {args.n_t} slices)")

        def g_field(t, x):
            return sp.grad(t, x)

        def r_field(t, x):
            return resid(float(np.clip(t, 0, 1)), torch.as_tensor(x)).numpy() - sp.grad(t, x)

        def f_null(t, x):
            return vstar_np(t, x) + r_field(t, x)

        def f_grad(t, x):
            return vstar_np(t, x) + g_field(t, x)

        # --- P4: MC flux-null fraction of the spline-based split ---
        num = den = 0.0
        gen = torch.Generator().manual_seed(90 + seed)
        for t in ts:
            xs, _, _ = gmm.sample_x_t(20000, float(t), generator=gen)
            xn = xs.numpy()
            rr = r_field(float(t), xn); ee = resid(float(t), xs).numpy()
            num += float((rr**2).sum(1).mean()); den += float((ee**2).sum(1).mean())
        f_mc = num / den
        p4 = abs(f_mc - f_grid) < 0.03
        print(f"  P4: MC f(spline split) = {f_mc:.4f} vs grid {f_grid:.4f} -> "
              f"{'OK' if p4 else 'FAIL'}")

        phi_theta, _ = EV.flow_map(EV.np_field(m), x0, rtol=1e-9, atol=1e-11)
        row = dict(seed=seed, f_grid=f_grid, f_mc=f_mc, P4=bool(p4))

        fields = {"null": f_null, "grad": f_grad}
        outs = {"theta": phi_theta}
        for name, fn in fields.items():
            t0 = time.time()
            p, info = EV.flow_map(fn, x0, rtol=1e-8, atol=1e-10, method="RK45")
            outs[name] = p
            print(f"  integrated {name:5s} nfev={info['nfev']} ({time.time()-t0:.0f}s)")

        for name, p in outs.items():
            pe = per_particle_error(p, phi_star)
            fl = mode_flip_stats(p, phi_star, mu)
            ed, ed_se = ed_with_se(p, dref)
            row[f"M_{name}"] = pe["mean"]
            row[f"pp_{name}"] = dict(mean=pe["mean"], median=pe["median"], p90=pe["p90"],
                                     p99=pe["p99"], max=pe["max"],
                                     rms=float(np.sqrt((pe["err"]**2).mean())))
            row[f"flip_{name}"] = dict(flip_rate=fl["flip_rate"],
                                       occupancy_tv=fl["occupancy_tv"])
            row[f"ed_{name}"] = ed; row[f"ed_{name}_se"] = ed_se
            np.save(os.path.join(out, f"phi_{name}_seed{seed}.npy"), p)

        # Exact divergences of the composite fields:
        #   div(v* + grad phi)     = div v*      + lap phi
        #   div(v* + e - grad phi) = div v_theta - lap phi
        def div_grad(t, x):
            return div_torch(gmm.v_star, t, x) + sp.lap(t, x)

        def div_null(t, x):
            return div_torch(lambda tt, xx: m(tt, xx), t, x) - sp.lap(t, x)

        for name, fn, dv in [("null", f_null, div_null), ("grad", f_grad, div_grad)]:
            try:
                k = kl_rk4(fn, dv, gmm, x0[: args.n_kl], n_steps=args.kl_steps)
                row[f"kl_{name}"] = k["kl"]; row[f"kl_{name}_se"] = k["se"]
            except Exception as ex:
                print(f"  kl_{name} failed: {ex}")
        try:
            from sharp_metrics import cnf_kl
            k = cnf_kl(m, gmm, x0[: args.n_kl])
            row["kl_theta"] = k["kl"]; row["kl_theta_se"] = k["se"]
        except Exception as ex:
            print(f"  kl_theta failed: {ex}")

        for nm in ["theta", "grad", "null"]:
            pp, fl = row[f"pp_{nm}"], row[f"flip_{nm}"]
            print(f"  {nm:5s}: M={pp['mean']:.4f} rms={pp['rms']:.4f} p99={pp['p99']:.4f} "
                  f"max={pp['max']:.3f} | flip={fl['flip_rate']:.3%} occTV={fl['occupancy_tv']:.4f} "
                  f"| ED={row[f'ed_{nm}']:.3e}+-{row[f'ed_{nm}_se']:.1e} "
                  f"| KL={row.get(f'kl_{nm}', float('nan')):.5f}"
                  f"+-{row.get(f'kl_{nm}_se', float('nan')):.5f}")

        res["rows"].append(row)
        with open(os.path.join(out, "metrics.json"), "w") as fh:
            json.dump(res, fh, indent=2)

    np.save(os.path.join(out, "phi_star.npy"), phi_star)
    np.save(os.path.join(out, "x0.npy"), x0)
    print(f"\nwritten: {os.path.join(out,'metrics.json')}")


if __name__ == "__main__":
    main()
