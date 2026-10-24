"""E1: the pre-registered decisive experiment (hypothesis_ledger.md Sec. 3).

PRIMARY METRIC      f  = time-integrated flux-null energy fraction of the
                         held-out velocity error e = v_theta - v*
PRIMARY CRITERION   H1 supported iff  f_bar - f0_bar > 0.05, Welch p < 0.01,
                         and |f(192) - f(256)| < 0.01
SECONDARY (H2)      R = M / (W2 - W2_floor) >= 3 for the median seed
GATE (H4)           M and between-seed map disagreement must change < 10%
                         between rtol 1e-5 and 1e-9

Nulls for f0:
  null_init   8 randomly-initialised nets of the same architecture  <- PRE-REGISTERED
  null_rot    the measured residual with its vector values globally rotated by a
              random angle (preserves |e|(x), its spectrum and ||e||^2 exactly;
              destroys only the alignment between e's direction and the local
              weighted-gradient geometry).  Strictly better matched than
              null_init, reported alongside it.
  null_swap   components of the measured residual swapped and one negated
              (a deterministic 90-degree rotation)

Usage:
  python3.13 src/e1_main.py --tag main --n-seeds 8 --steps 20000 --width 128
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
from model import VelocityMLP, count_params, train_cfm             # noqa: E402
import evaluate as EV                                              # noqa: E402

torch.set_default_dtype(torch.float64)
torch.set_num_threads(2)


def rotated_residual_fn(model, gmm, angle):
    c, s = np.cos(angle), np.sin(angle)
    R = torch.tensor([[c, -s], [s, c]], dtype=torch.float64)

    def fn(t, x):
        with torch.no_grad():
            e = model(t, x) - gmm.v_star(t, x)
            return e @ R.T
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    ap.add_argument("--dataset", default="ring8")
    ap.add_argument("--n-seeds", type=int, default=8)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--act", default="silu")
    ap.add_argument("--lam-div", type=float, default=0.0)
    ap.add_argument("--grid-N", type=int, default=192)
    ap.add_argument("--n-map", type=int, default=4000)
    ap.add_argument("--L", type=float, default=8.0)
    ap.add_argument("--skip-gate", action="store_true")
    args = ap.parse_args()

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                       "results", "raw", "e1_" + args.tag)
    os.makedirs(out, exist_ok=True)
    os.makedirs(os.path.join(out, "ckpt"), exist_ok=True)
    print(f"output dir: {out}\nconfig: {vars(args)}")

    gmm = REGISTRY[args.dataset]()
    grid = FaceGrid(args.grid_N, args.L)
    grid256 = FaceGrid(256, args.L)
    res = dict(config=vars(args), models=[], nulls={}, gate={}, misc={})

    # ---------- ground truth map, W2 floor ----------
    rng = np.random.default_rng(12345)
    x0 = rng.standard_normal((args.n_map, 2))
    print("\n[gt] integrating ground-truth flow map ...")
    t0 = time.time()
    phi_star, gtinfo = EV.flow_map(EV.np_vstar(gmm), x0, rtol=1e-11, atol=1e-13)
    print(f"  done in {time.time()-t0:.0f}s, nfev={gtinfo['nfev']}")
    d_ref = gmm.sample_x1(args.n_map, generator=torch.Generator().manual_seed(777)).numpy()
    d_ref2 = gmm.sample_x1(args.n_map, generator=torch.Generator().manual_seed(778)).numpy()
    w2_floor = EV.w2(d_ref, d_ref2)
    w2_star = EV.w2(phi_star, d_ref)
    print(f"  W2 floor (same-n, two p1 samples) = {w2_floor:.4f}")
    print(f"  W2(Phi*#p0, p1)                   = {w2_star:.4f}")
    np.save(os.path.join(out, "phi_star.npy"), phi_star)
    np.save(os.path.join(out, "x0.npy"), x0)
    res["misc"].update(w2_floor=w2_floor, w2_star=w2_star, gt_nfev=gtinfo["nfev"])

    # ---------- Gronwall amplification along the true flow ----------
    print("[gt] measuring Gronwall amplification C_G ...")
    t0 = time.time()
    try:
        cg = EV.gronwall_factor(gmm, x0[:200])
        res["misc"]["C_G"] = {k: (v if np.isscalar(v) else np.asarray(v).tolist())
                              for k, v in cg.items()}
        print(f"  C_G mean={cg['C_G_mean']:.2f} median={cg['C_G_median']:.2f} "
              f"p90={cg['C_G_p90']:.2f}  ({time.time()-t0:.0f}s)")
    except Exception as ex:                                     # pragma: no cover
        print(f"  C_G failed: {ex}")
        res["misc"]["C_G"] = None

    # ---------- train and measure ----------
    models = []
    for seed in range(args.n_seeds):
        print(f"\n=== seed {seed} ===")
        t0 = time.time()
        m32, hist = train_cfm(gmm, seed=seed, width=args.width, depth=args.depth,
                              steps=args.steps, batch=args.batch, lr=args.lr,
                              act=args.act, lam_div=args.lam_div,
                              dtype=torch.float32, log_every=max(1, args.steps // 6))
        t_train = time.time() - t0
        m = m32.to(torch.float64)
        models.append(m)
        torch.save(m32.state_dict(), os.path.join(out, "ckpt", f"seed{seed}.pt"))

        rec = dict(seed=seed, t_train=t_train, loss_hist=hist,
                   n_params=count_params(m))

        r = EV.eta2_mc(lambda t, x: m(t, x), gmm, n=200_000, seed=1000 + seed)
        rec.update(eta2=r["eta2"], vstar2=r["vstar2"], eta_rel=float(r["rel"]),
                   eta2_per_t=r["per_t"].tolist())

        fr = EV.flux_null_fraction(EV.residual_fn(m, gmm), gmm, grid=grid)
        rec.update(f=fr["f"], e2_grid=fr["e2"], g2_grid=fr["g2"], r2_grid=fr["r2"],
                   max_ortho=fr["max_ortho"], max_closure=fr["max_closure"],
                   f_per_t=[(x["t"], x["f"], x["e2"]) for x in fr["per_t"]])

        # grid-refinement check of the primary metric (first 3 seeds only)
        if seed < 3:
            fr2 = EV.flux_null_fraction(EV.residual_fn(m, gmm), gmm, grid=grid256)
            rec["f_grid256"] = fr2["f"]

        phi, info = EV.flow_map(EV.np_field(m), x0, rtol=1e-9, atol=1e-11)
        M = float(np.linalg.norm(phi - phi_star, axis=1).mean())
        w2_m = EV.w2(phi, d_ref)
        rec.update(M=M, w2=w2_m, map_nfev=info["nfev"],
                   w2_excess=w2_m - w2_floor)
        np.save(os.path.join(out, f"phi_seed{seed}.npy"), phi)

        print(f"  train {t_train:.0f}s | eta_rel={rec['eta_rel']:.4f} | "
              f"f={rec['f']:.4f} | M={M:.4f} | W2={w2_m:.4f} "
              f"(excess {rec['w2_excess']:+.4f}) | ortho={fr['max_ortho']:.1e}")
        res["models"].append(rec)
        with open(os.path.join(out, "metrics.json"), "w") as fh:
            json.dump(res, fh, indent=2)

    # ---------- nulls ----------
    print("\n=== nulls ===")
    # (a) pre-registered: randomly-initialised nets of the same architecture
    f_init = []
    for seed in range(args.n_seeds):
        torch.manual_seed(50_000 + seed)
        mn = VelocityMLP(d=2, width=args.width, depth=args.depth,
                         act=args.act).to(torch.float64)
        # f is scale-invariant, so no rescaling is needed; the field itself is
        # the surrogate residual
        fr = EV.flux_null_fraction(lambda t, x, mn=mn: mn(t, x), gmm, grid=grid)
        f_init.append(fr["f"])
        print(f"  null_init seed{seed}: f0={fr['f']:.4f}")
    res["nulls"]["null_init"] = f_init

    # (b) random global rotation of each measured residual
    f_rot = []
    rr = np.random.default_rng(99)
    for i, m in enumerate(models):
        ang = float(rr.uniform(0, 2 * np.pi))
        fr = EV.flux_null_fraction(rotated_residual_fn(m, gmm, ang), gmm, grid=grid)
        f_rot.append(fr["f"])
        print(f"  null_rot  seed{i}: angle={ang:.2f} f0={fr['f']:.4f}")
    res["nulls"]["null_rot"] = f_rot

    # (c) deterministic 90-degree rotation
    f_swap = []
    for i, m in enumerate(models):
        fr = EV.flux_null_fraction(rotated_residual_fn(m, gmm, np.pi / 2), gmm, grid=grid)
        f_swap.append(fr["f"])
    res["nulls"]["null_swap"] = f_swap
    print(f"  null_swap: {np.mean(f_swap):.4f} +- {np.std(f_swap, ddof=1):.4f}")

    # ---------- ground-truth-free surrogate f_Delta (theory.md Sec. 7) ----------
    print("\n=== pairwise model differences (ground-truth-free surrogate) ===")
    f_delta, map_disagree, w2_disagree = [], [], []
    phis = [np.load(os.path.join(out, f"phi_seed{s}.npy")) for s in range(args.n_seeds)]
    for i in range(args.n_seeds):
        for j in range(i + 1, args.n_seeds):
            fr = EV.flux_null_fraction(EV.difference_fn(models[i], models[j]), gmm, grid=grid)
            f_delta.append(fr["f"])
            map_disagree.append(float(np.linalg.norm(phis[i] - phis[j], axis=1).mean()))
            w2_disagree.append(EV.w2(phis[i], phis[j]))
    res["pairwise"] = dict(f_delta=f_delta, map_disagree=map_disagree,
                           w2_disagree=w2_disagree)
    print(f"  f_Delta      = {np.mean(f_delta):.4f} +- {np.std(f_delta, ddof=1):.4f}")
    print(f"  map disagree = {np.mean(map_disagree):.4f}")
    print(f"  W2  disagree = {np.mean(w2_disagree):.4f}")

    # ---------- H4 gate: solver tolerance ----------
    if not args.skip_gate:
        print("\n=== H4 gate: solver tolerance sweep ===")
        gate = {}
        for rtol in [1e-5, 1e-7, 1e-9]:
            Ms, ps = [], []
            for s in range(min(3, args.n_seeds)):
                p, _ = EV.flow_map(EV.np_field(models[s]), x0[:1500], rtol=rtol,
                                   atol=rtol * 1e-2)
                ps.append(p)
                Ms.append(float(np.linalg.norm(p - phi_star[:1500], axis=1).mean()))
            dis = [float(np.linalg.norm(ps[a] - ps[b], axis=1).mean())
                   for a in range(len(ps)) for b in range(a + 1, len(ps))]
            gate[f"rtol{rtol:.0e}"] = dict(M=Ms, disagree=dis)
            print(f"  rtol={rtol:.0e}  M={['%.5f' % x for x in Ms]}  "
                  f"disagree={['%.5f' % x for x in dis]}")
        a = np.array(gate["rtol1e-05"]["M"]); b = np.array(gate["rtol1e-09"]["M"])
        da = np.array(gate["rtol1e-05"]["disagree"]); db = np.array(gate["rtol1e-09"]["disagree"])
        gate["M_rel_change"] = float(np.max(np.abs(a - b) / b))
        gate["disagree_rel_change"] = float(np.max(np.abs(da - db) / db))
        gate["passed"] = bool(gate["M_rel_change"] < 0.10 and gate["disagree_rel_change"] < 0.10)
        print(f"  max rel change: M {gate['M_rel_change']:.4f}, "
              f"disagree {gate['disagree_rel_change']:.4f}  -> "
              f"{'GATE PASS (effect is not solver-induced)' if gate['passed'] else 'GATE FAIL'}")
        res["gate"] = gate

    # ---------- verdict ----------
    from scipy import stats
    fs = np.array([r["f"] for r in res["models"]])
    verdict = {}
    for name, f0 in res["nulls"].items():
        f0 = np.array(f0)
        tt = stats.ttest_ind(fs, f0, equal_var=False)
        verdict[name] = dict(f_mean=float(fs.mean()), f_sd=float(fs.std(ddof=1)),
                             f0_mean=float(f0.mean()), f0_sd=float(f0.std(ddof=1)),
                             diff=float(fs.mean() - f0.mean()),
                             t=float(tt.statistic), p=float(tt.pvalue))
    g_ref = [r.get("f_grid256") for r in res["models"] if r.get("f_grid256") is not None]
    g_192 = [r["f"] for r in res["models"] if r.get("f_grid256") is not None]
    verdict["grid_stability"] = float(np.max(np.abs(np.array(g_ref) - np.array(g_192)))) if g_ref else None

    Ms = np.array([r["M"] for r in res["models"]])
    ex = np.array([r["w2_excess"] for r in res["models"]])
    with np.errstate(divide="ignore", invalid="ignore"):
        R = Ms / ex
    verdict["H2"] = dict(M_median=float(np.median(Ms)), w2_excess_median=float(np.median(ex)),
                         R_median=float(np.median(R)), R_all=R.tolist(),
                         note="R is meaningless where w2_excess <= 0; see analysis")
    res["verdict"] = verdict

    print("\n================ VERDICT ================")
    print(f"f (trained)  = {fs.mean():.4f} +- {fs.std(ddof=1):.4f}   (n={len(fs)})")
    for name in res["nulls"]:
        v = verdict[name]
        print(f"{name:10s}  f0 = {v['f0_mean']:.4f} +- {v['f0_sd']:.4f}  "
              f"diff = {v['diff']:+.4f}  Welch p = {v['p']:.3e}")
    print(f"grid stability |f(192)-f(256)| max = {verdict['grid_stability']}")
    print(f"H2: M median = {verdict['H2']['M_median']:.4f}, "
          f"W2 excess median = {verdict['H2']['w2_excess_median']:+.4f}, "
          f"R median = {verdict['H2']['R_median']:.2f}")

    with open(os.path.join(out, "metrics.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"\nwritten: {os.path.join(out, 'metrics.json')}")


if __name__ == "__main__":
    main()
