"""E3: curl-free (potential) parameterisation as a controlled intervention.

Since v* is exactly a gradient field (theory.md Sec. 2.4), parameterising
v_theta = grad_x Psi_theta(t,x) is a ZERO-BIAS restriction that eliminates the
flux-null subspace by construction. This experiment asks whether removing the
spurious circulation measured in E1 actually buys transport-map fidelity.

PRE-REGISTERED PREDICTIONS (written before running):
  Q1 (sanity)  f_pot ~ 0 (< 0.01). If not, the implementation is wrong.
  Q2           eta_rel(pot) is within 25% of eta_rel(base) -- the restriction
               costs little or nothing in velocity-regression accuracy, since
               it is zero-bias.
  Q3 (primary) M(pot) < M(base), i.e. transport-map error improves.
  Q4           the distributional discrepancy (energy distance, CNF KL) is
               NOT better for pot by more than its standard error -- i.e. the
               improvement is specifically in the map, which is the whole point.

FALSIFICATION of the practical claim: M(pot) >= M(base). That would mean the
spurious circulation is not what limits map fidelity, and the measurement in E1
has no actionable consequence.

Compute fairness: both arms get identical steps/batch/lr/seeds/architecture
width and depth. The potential arm costs more wall-clock per step (double
backward); both the matched-step result and the wall-clock ratio are reported.
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
from model import count_params, train_cfm                          # noqa: E402
import evaluate as EV                                              # noqa: E402
from sharp_metrics import cnf_kl, energy_distance                  # noqa: E402

torch.set_default_dtype(torch.float64)
torch.set_num_threads(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    ap.add_argument("--dataset", default="ring8")
    ap.add_argument("--n-seeds", type=int, default=4)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--grid-N", type=int, default=192)
    ap.add_argument("--n-map", type=int, default=3000)
    ap.add_argument("--n-kl", type=int, default=2000)
    ap.add_argument("--arms", nargs="+", default=["velocity", "potential"])
    args = ap.parse_args()

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                       "results", "raw", "e3_" + args.tag)
    os.makedirs(os.path.join(out, "ckpt"), exist_ok=True)
    print(f"output: {out}\nconfig: {vars(args)}")

    gmm = REGISTRY[args.dataset]()
    grid = FaceGrid(args.grid_N, 8.0)
    rng = np.random.default_rng(31337)
    x0 = rng.standard_normal((args.n_map, 2))

    print("ground truth map ...")
    phi_star, _ = EV.flow_map(EV.np_vstar(gmm), x0, rtol=1e-11, atol=1e-13)
    dref = gmm.sample_x1(args.n_map, generator=torch.Generator().manual_seed(4242)).numpy()
    dref2 = gmm.sample_x1(args.n_map, generator=torch.Generator().manual_seed(4243)).numpy()
    ed_floor = energy_distance(dref, dref2)
    ed_star = energy_distance(phi_star, dref)
    print(f"  energy-distance floor = {ed_floor:.4e};  Phi* = {ed_star:.4e}")

    res = dict(config=vars(args), ed_floor=ed_floor, ed_star=ed_star, rows=[])

    for arch in args.arms:
        for seed in range(args.n_seeds):
            print(f"\n=== arch={arch} seed={seed} ===")
            t0 = time.time()
            m32, hist = train_cfm(gmm, seed=seed, width=args.width, depth=args.depth,
                                  steps=args.steps, batch=args.batch, lr=args.lr,
                                  dtype=torch.float32, arch=arch,
                                  log_every=max(1, args.steps // 4))
            t_train = time.time() - t0
            m = m32.to(torch.float64)
            torch.save(m32.state_dict(), os.path.join(out, "ckpt", f"{arch}_seed{seed}.pt"))

            r = EV.eta2_mc(lambda t, x: m(t, x), gmm, n=150_000, seed=2000 + seed)
            fr = EV.flux_null_fraction(EV.residual_fn(m, gmm), gmm, grid=grid)
            phi, info = EV.flow_map(EV.np_field(m), x0, rtol=1e-9, atol=1e-11)
            M = float(np.linalg.norm(phi - phi_star, axis=1).mean())
            ed = energy_distance(phi, dref)
            w2v = EV.w2(phi, dref)
            try:
                kl = cnf_kl(m, gmm, x0[: args.n_kl], rtol=1e-7, atol=1e-9)
                klv, klse = kl["kl"], kl["se"]
            except Exception as ex:
                print(f"  cnf_kl failed: {ex}")
                klv, klse = None, None

            row = dict(arch=arch, seed=seed, t_train=t_train,
                       n_params=count_params(m), loss_hist=hist,
                       eta2=r["eta2"], eta_rel=float(r["rel"]),
                       f=fr["f"], max_closure=fr["max_closure"],
                       M=M, ed=ed, w2=w2v, kl=klv, kl_se=klse,
                       map_nfev=info["nfev"])
            print(f"  params={row['n_params']}  train={t_train:.0f}s  "
                  f"eta_rel={row['eta_rel']:.4f}  f={row['f']:.5f}  M={M:.5f}")
            print(f"  ed={ed:.4e} (floor {ed_floor:.4e}, star {ed_star:.4e})  "
                  f"w2={w2v:.4f}  KL={klv if klv is None else round(klv,5)}"
                  f"{'' if klse is None else ' +- %.5f' % klse}")
            np.save(os.path.join(out, f"phi_{arch}_seed{seed}.npy"), phi)
            res["rows"].append(row)
            with open(os.path.join(out, "metrics.json"), "w") as fh:
                json.dump(res, fh, indent=2)

    # ---------------- comparison ----------------
    from scipy import stats
    def col(arch, k):
        return np.array([r[k] for r in res["rows"] if r["arch"] == arch and r[k] is not None],
                        dtype=float)

    print("\n================ E3 COMPARISON ================")
    summary = {}
    for k in ["eta_rel", "f", "M", "ed", "w2", "kl", "t_train"]:
        a, b = col("velocity", k), col("potential", k)
        if a.size == 0 or b.size == 0:
            continue
        tt = stats.ttest_ind(a, b, equal_var=False)
        summary[k] = dict(velocity_mean=float(a.mean()), velocity_sd=float(a.std(ddof=1)),
                          potential_mean=float(b.mean()), potential_sd=float(b.std(ddof=1)),
                          p=float(tt.pvalue))
        print(f"{k:10s} velocity {a.mean():.5g} +- {a.std(ddof=1):.3g} | "
              f"potential {b.mean():.5g} +- {b.std(ddof=1):.3g} | Welch p={tt.pvalue:.3e}")
    res["summary"] = summary

    if "M" in summary and "ed" in summary:
        s = summary
        print("\nVerdict against pre-registered predictions:")
        print(f"  Q1 f_pot ~ 0            : f_pot={s['f']['potential_mean']:.2e}  "
              f"{'OK' if s['f']['potential_mean'] < 0.01 else 'FAIL'}")
        rel = abs(s['eta_rel']['potential_mean'] - s['eta_rel']['velocity_mean']) / s['eta_rel']['velocity_mean']
        print(f"  Q2 eta within 25%       : rel diff {rel:.1%}  {'OK' if rel < 0.25 else 'FAIL'}")
        print(f"  Q3 M(pot) < M(base)     : {s['M']['potential_mean']:.5f} vs "
              f"{s['M']['velocity_mean']:.5f}  "
              f"{'OK' if s['M']['potential_mean'] < s['M']['velocity_mean'] else 'FAIL'}"
              f"  (p={s['M']['p']:.3e})")

    with open(os.path.join(out, "metrics.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"\nwritten: {os.path.join(out,'metrics.json')}")


if __name__ == "__main__":
    main()
