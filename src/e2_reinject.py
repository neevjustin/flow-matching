"""E2: re-inject the MEASURED Helmholtz components of a trained model's error.

E0/V10 showed that an *analytic* flux-null perturbation preserves the marginals
while moving the transport map. E2 asks the empirical version: do the components
actually extracted from a trained network's residual behave the same way?

For each model we build four fields and integrate them from a common x0:

    v*                      -> Phi_star            (ground truth)
    v* + interp(grad phi)   -> Phi_grad            (the "visible" half)
    v* + interp(r)          -> Phi_null            (the "invisible" half)
    v* + interp(grad phi+r) -> Phi_recon           (VALIDATION: should ~ Phi_theta)

The Phi_recon check is essential: it bounds the interpolation error of the
whole procedure. If Phi_recon does not track Phi_theta, nothing else in E2 is
trustworthy.

PRE-REGISTERED PREDICTIONS (written before running):
  P1  Phi_null has a distributional discrepancy to p_1 statistically
      indistinguishable from Phi_star's, while its map shift is > 0.
  P2  Phi_grad carries essentially all of the model's distributional error.
  P3  ||Phi_recon - Phi_theta|| << M, validating the interpolation.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gmm_path import REGISTRY                                      # noqa: E402
from helmholtz import FaceGrid                                     # noqa: E402
from model import VelocityMLP                                      # noqa: E402
import evaluate as EV                                              # noqa: E402
from sharp_metrics import GridFieldInterp, energy_distance         # noqa: E402
from analyze import mode_flip_stats, per_particle_error            # noqa: E402

torch.set_default_dtype(torch.float64)
torch.set_num_threads(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="e1_main")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--dataset", default="ring8")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--grid-N", type=int, default=192)
    ap.add_argument("--L", type=float, default=8.0)
    ap.add_argument("--n-map", type=int, default=3000)
    args = ap.parse_args()

    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "raw")
    src = os.path.join(base, args.src)
    out = os.path.join(base, "e2_" + args.tag)
    os.makedirs(out, exist_ok=True)

    gmm = REGISTRY[args.dataset]()
    grid = FaceGrid(args.grid_N, args.L)
    rng = np.random.default_rng(2024)
    x0 = rng.standard_normal((args.n_map, 2))

    print("integrating ground truth ...")
    phi_star, _ = EV.flow_map(EV.np_vstar(gmm), x0, rtol=1e-11, atol=1e-13)
    dref = gmm.sample_x1(args.n_map, generator=torch.Generator().manual_seed(555)).numpy()
    dref2 = gmm.sample_x1(args.n_map, generator=torch.Generator().manual_seed(556)).numpy()

    ed_floor = energy_distance(dref, dref2)
    ed_star = energy_distance(phi_star, dref)
    w2_floor = EV.w2(dref, dref2)
    print(f"  energy-distance floor (two p1 samples) = {ed_floor:.3e}")
    print(f"  energy distance Phi*#p0 vs p1          = {ed_star:.3e}")

    res = dict(config=vars(args), ed_floor=ed_floor, ed_star=ed_star,
               w2_floor=w2_floor, rows=[])

    vstar_np = EV.np_vstar(gmm)

    for seed in args.seeds:
        print(f"\n=== seed {seed} ===")
        m = VelocityMLP(d=2, width=args.width, depth=args.depth).to(torch.float32)
        m.load_state_dict(torch.load(os.path.join(src, "ckpt", f"seed{seed}.pt")))
        m = m.to(torch.float64)

        t0 = time.time()
        fr = EV.flux_null_fraction(EV.residual_fn(m, gmm), gmm, grid=grid,
                                   return_maps=True)
        print(f"  f = {fr['f']:.4f}  (decompose {time.time()-t0:.0f}s)")
        ir = GridFieldInterp(grid, fr["maps"], which="r")
        ig = GridFieldInterp(grid, fr["maps"], which="g")

        def f_null(t, x):
            return vstar_np(t, x) + ir(t, x)

        def f_grad(t, x):
            return vstar_np(t, x) + ig(t, x)

        def f_recon(t, x):
            return vstar_np(t, x) + ir(t, x) + ig(t, x)

        phi_theta, _ = EV.flow_map(EV.np_field(m), x0, rtol=1e-9, atol=1e-11)
        out_fields = {}
        for name, fn in [("null", f_null), ("grad", f_grad), ("recon", f_recon)]:
            t0 = time.time()
            p, info = EV.flow_map(fn, x0, rtol=1e-8, atol=1e-10, method="RK45")
            out_fields[name] = p
            print(f"  integrated {name:6s} nfev={info['nfev']} ({time.time()-t0:.0f}s)")

        row = dict(seed=seed, f=fr["f"],
                   M_theta=float(np.linalg.norm(phi_theta - phi_star, axis=1).mean()),
                   ed_theta=energy_distance(phi_theta, dref),
                   w2_theta=EV.w2(phi_theta, dref))
        mu = gmm.mu.numpy()
        pe = per_particle_error(phi_theta, phi_star)
        row["pp_theta"] = {k: v for k, v in pe.items() if k != "err"}
        row["pp_theta"]["rms"] = float(np.sqrt((pe["err"] ** 2).mean()))
        row["flip_theta"] = {k: v for k, v in mode_flip_stats(phi_theta, phi_star, mu).items()
                             if k not in ("occupancy_a", "occupancy_b")}
        for name, p in out_fields.items():
            row[f"M_{name}"] = float(np.linalg.norm(p - phi_star, axis=1).mean())
            row[f"ed_{name}"] = energy_distance(p, dref)
            row[f"w2_{name}"] = EV.w2(p, dref)
            pe = per_particle_error(p, phi_star)
            row[f"pp_{name}"] = {k: v for k, v in pe.items() if k != "err"}
            row[f"pp_{name}"]["rms"] = float(np.sqrt((pe["err"] ** 2).mean()))
            row[f"flip_{name}"] = {k: v for k, v in mode_flip_stats(p, phi_star, mu).items()
                                   if k not in ("occupancy_a", "occupancy_b")}
        row["recon_err"] = float(np.linalg.norm(out_fields["recon"] - phi_theta, axis=1).mean())
        row["recon_rel"] = row["recon_err"] / row["M_theta"]

        print(f"  M_theta={row['M_theta']:.4f}  M_grad={row['M_grad']:.4f}  "
              f"M_null={row['M_null']:.4f}  M_recon={row['M_recon']:.4f}")
        print(f"  recon validation: |Phi_recon - Phi_theta| = {row['recon_err']:.5f} "
              f"= {row['recon_rel']:.1%} of M_theta  "
              f"{'[P3 OK]' if row['recon_rel'] < 0.25 else '[P3 FAIL - interpolation untrustworthy]'}")
        for nm in ["theta", "grad", "null"]:
            pp, fl = row[f"pp_{nm}"], row[f"flip_{nm}"]
            print(f"  {nm:6s}: mean={pp['mean']:.4f} rms={pp['rms']:.4f} "
                  f"p99={pp['p99']:.4f} max={pp['max']:.4f} | "
                  f"mode-flip={fl['flip_rate']:.3%} occTV={fl['occupancy_tv']:.4f}")
        print(f"  energy distance: floor={ed_floor:.3e} star={ed_star:.3e} "
              f"null={row['ed_null']:.3e} grad={row['ed_grad']:.3e} theta={row['ed_theta']:.3e}")

        for name, p in out_fields.items():
            np.save(os.path.join(out, f"phi_{name}_seed{seed}.npy"), p)
        np.save(os.path.join(out, f"phi_theta_seed{seed}.npy"), phi_theta)
        res["rows"].append(row)
        with open(os.path.join(out, "metrics.json"), "w") as fh:
            json.dump(res, fh, indent=2)

    np.save(os.path.join(out, "phi_star.npy"), phi_star)
    np.save(os.path.join(out, "x0.npy"), x0)
    print(f"\nwritten: {os.path.join(out, 'metrics.json')}")


if __name__ == "__main__":
    main()
