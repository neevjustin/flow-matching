"""E0: validation gate. Nothing downstream is trustworthy until all of this passes.

Checks, in order:

  V1  grid quadrature of the closed-form p_t integrates to 1
  V2  v* really is E[u | x_t] -- Monte-Carlo orthogonality of the residual
      against several test functions
  V3  the closed-form loss floor matches the Monte-Carlo conditional variance
  V4  v* satisfies the continuity equation numerically on the grid
  V5  v* transports p_0 to p_1 (endpoint W2 at the sampling-noise floor)
  V6  Helmholtz solver: pure gradient input          -> f ~ 0
  V7  Helmholtz solver: pure flux-null input         -> f ~ 1
  V8  Helmholtz solver: known mixture                -> f ~ prescribed
  V9  grid-refinement and tol_density sensitivity of f
  V10 MECHANISM EXISTENCE: adding an exact flux-null field to v* leaves the
      endpoint distribution unchanged while changing the transport map a lot

Run:  python3.13 -I src/e0_validate.py
"""

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gmm_path import ring8                                  # noqa: E402
from helmholtz import FaceGrid, decompose                   # noqa: E402
from ode import integrate_adaptive                          # noqa: E402

torch.set_default_dtype(torch.float64)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                   "results", "raw", "e0")
os.makedirs(OUT, exist_ok=True)

L = 8.0
report = {}


def ok(name, cond, detail):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}: {detail}")
    report[name] = dict(status=status, detail=detail)
    return cond


# ----------------------------------------------------------------- helpers

def analytic_stream_fluxnull(gmm, t, pos):
    """An EXACTLY flux-null field at face positions.

    Take psi = p_t * s  with s smooth. Then  grad_perp(psi)/p_t
      = grad_perp(s) + s * grad_perp(log p_t),
    and div(p_t * that) = div(grad_perp(p_t s)) = 0 identically.
    Here s(x) = sin(a x1) cos(b x2), so grad_perp(s) is analytic and
    grad_perp(log p_t) comes from the closed-form score.
    """
    a, b = 0.45, 0.37
    x = torch.as_tensor(pos)
    x1, x2 = x[:, 0], x[:, 1]
    s = torch.sin(a * x1) * torch.cos(b * x2)
    ds1 = a * torch.cos(a * x1) * torch.cos(b * x2)
    ds2 = -b * torch.sin(a * x1) * torch.sin(b * x2)
    # grad_perp(h) = (-d2 h, d1 h)
    gp_s = torch.stack([-ds2, ds1], dim=1)
    sc = gmm.grad_log_p_t(t, x)                       # (n,2) grad log p
    gp_logp = torch.stack([-sc[:, 1], sc[:, 0]], dim=1)
    return gp_s + s[:, None] * gp_logp


def analytic_gradient_field(pos):
    """A pure gradient field: e = grad(chi), chi = exp(-|x|^2/18) * x1 * x2."""
    x = torch.as_tensor(pos)
    x1, x2 = x[:, 0], x[:, 1]
    g = torch.exp(-(x1**2 + x2**2) / 18.0)
    # chi = g * x1 * x2
    d1 = g * x2 * (1.0 - 2.0 * x1 * x1 / 18.0)
    d2 = g * x1 * (1.0 - 2.0 * x2 * x2 / 18.0)
    return torch.stack([d1, d2], dim=1)


def face_normal(vec, grid):
    """Take the (nface,2) field and keep the component normal to each face."""
    out = np.where(grid.face_comp == 0, vec[:, 0], vec[:, 1])
    return out


def grid_fields(gmm, grid, t):
    fp = torch.as_tensor(grid.face_pos)
    cp = torch.as_tensor(grid.cell_positions())
    p_face = gmm.p_t(t, fp).numpy()
    p_cell = gmm.p_t(t, cp).numpy()
    return p_face, p_cell


# ----------------------------------------------------------------- V1

print("\n=== V1: grid quadrature of p_t ===")
gmm = ring8()
g256 = FaceGrid(256, L)
cp = torch.as_tensor(g256.cell_positions())
for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
    mass = float(gmm.p_t(t, cp).sum()) * g256.h**2
    ok(f"V1_mass_t{t}", abs(mass - 1.0) < 2e-4, f"integral(p_t) = {mass:.8f}")

# ----------------------------------------------------------------- V2, V3

print("\n=== V2/V3: v* is the conditional mean; loss floor is correct ===")
gen = torch.Generator().manual_seed(0)
n_mc = 400_000
for t in [0.1, 0.35, 0.6, 0.85]:
    xt, x0, x1 = gmm.sample_x_t(n_mc, t, generator=gen)
    u = x1 - x0
    res = u - gmm.v_star(t, xt)
    # E[res * g(x_t)] must be 0 for any test function g
    tests = {
        "const": torch.ones(n_mc, 1),
        "lin": xt,
        "quad": xt**2,
        "sin": torch.sin(0.7 * xt),
    }
    worst, worst_name = 0.0, ""
    for name, gfun in tests.items():
        # products res_i * g_j; test each mean against its own standard error
        prod = res[:, :, None] * gfun[:, None, :]          # (n, d, m)
        m = prod.mean(0)
        se = prod.std(0) / np.sqrt(n_mc)
        z = float((m.abs() / se.clamp(min=1e-300)).max())
        if z > worst:
            worst, worst_name = z, name
    ok(f"V2_t{t}", worst < 5.0, f"max |E[res*g]| z-score = {worst:.2f} ({worst_name})")

    mc_floor = float((res**2).sum(1).mean())
    cf_floor = float(gmm.loss_floor(t, xt).mean())
    rel = abs(mc_floor - cf_floor) / cf_floor
    ok(f"V3_t{t}", rel < 5e-3, f"MC floor {mc_floor:.6f} vs closed form {cf_floor:.6f} (rel {rel:.2e})")

# ----------------------------------------------------------------- V4

print("\n=== V4: continuity equation for v* on the grid ===")
# d_t p + div(p v*) = 0, central differences in t, finite volume in x
# The residual of a 2nd-order finite-volume scheme cannot be made exactly zero
# at fixed h. The correct test is therefore that the residual CONVERGES at the
# scheme's design order, which falsifies "v* is the wrong field" (that would
# leave an O(1) residual that does not shrink). A fixed threshold would only be
# testing the grid resolution.
def continuity_residual(t, N):
    gN = FaceGrid(N, L)
    cpos = torch.as_tensor(gN.cell_positions())
    fpos = torch.as_tensor(gN.face_pos)
    dt = 1e-5
    dpdt = (gmm.p_t(t + dt, cpos) - gmm.p_t(t - dt, cpos)).numpy() / (2 * dt)
    pf = gmm.p_t(t, fpos).numpy()
    vf = face_normal(gmm.v_star(t, fpos).numpy(), gN)
    flux = pf * vf
    # div(F)|_cell = (1/h^2) * sum_faces F.n * h.  For a face between A (lower
    # index) and B, the outward normal is +axis for A and -axis for B.
    div = np.zeros(gN.ncell)
    np.add.at(div, gN.face_A, flux / gN.h)
    np.add.at(div, gN.face_B, -flux / gN.h)
    resid = dpdt + div
    num = np.sqrt((resid**2 * gN.h**2).sum())
    den = np.sqrt((dpdt**2 * gN.h**2).sum())
    return float(num / den)

for t in [0.2, 0.5, 0.8]:
    rs = [continuity_residual(t, N) for N in (96, 192, 384)]
    orders = [np.log2(rs[0] / rs[1]), np.log2(rs[1] / rs[2])]
    ok(f"V4_t{t}", min(orders) > 1.7 and rs[-1] < 5e-3,
       f"L2 rel residual {rs[0]:.2e} -> {rs[1]:.2e} -> {rs[2]:.2e}; "
       f"observed order {orders[0]:.2f}, {orders[1]:.2f} (design 2)")

# ----------------------------------------------------------------- V5

print("\n=== V5: v* transports p_0 to p_1 ===")
import ot  # noqa: E402

def w2(a, b):
    """Exact squared-W2 -> W2 between two equal-size empirical samples."""
    n = a.shape[0]
    M = ot.dist(np.ascontiguousarray(a), np.ascontiguousarray(b))
    w = np.full(n, 1.0 / n)
    return float(np.sqrt(ot.emd2(w, w, M, numItermax=2_000_000)))


def vstar_np(t, x):
    return gmm.v_star(min(max(t, 0.0), 1.0), torch.as_tensor(x)).numpy()


n_s = 4000
rng = np.random.default_rng(0)
x0s = rng.standard_normal((n_s, 2))
t0 = time.time()
phi_star, info = integrate_adaptive(vstar_np, x0s, 0.0, 1.0, rtol=1e-11, atol=1e-13)
t_gt = time.time() - t0
gen2 = torch.Generator().manual_seed(123)
d1 = gmm.sample_x1(n_s, generator=gen2).numpy()
d2 = gmm.sample_x1(n_s, generator=torch.Generator().manual_seed(456)).numpy()
floor = w2(d1, d2)
got = w2(phi_star, d1)
ok("V5_transport", got < 1.6 * floor,
   f"W2(Phi*#p0, p1) = {got:.4f} vs same-n noise floor {floor:.4f} "
   f"(ratio {got/floor:.3f}); nfev={info['nfev']}, {t_gt:.1f}s")

# ----------------------------------------------------------------- V6/V7/V8

print("\n=== V6/V7/V8: Helmholtz solver on fields with known decomposition ===")
t_test = 0.5
for N in [128, 192]:
    gg = FaceGrid(N, L)
    p_face, p_cell = grid_fields(gmm, gg, t_test)
    fp = torch.as_tensor(gg.face_pos)

    e_grad = face_normal(analytic_gradient_field(fp).numpy(), gg)
    r1 = decompose(gg, p_face, e_grad, p_cell)
    ok(f"V6_N{N}", r1["f"] < 0.02,
       f"pure gradient -> f = {r1['f']:.5f} (ortho {r1['ortho']:.2e}, closure {r1['closure']:.2e})")

    e_null = face_normal(analytic_stream_fluxnull(gmm, t_test, fp).numpy(), gg)
    r2 = decompose(gg, p_face, e_null, p_cell)
    ok(f"V7_N{N}", r2["f"] > 0.98,
       f"pure flux-null -> f = {r2['f']:.5f} (ortho {r2['ortho']:.2e}, closure {r2['closure']:.2e})")

    # V8: prescribed mixture. Scale each part to equal weighted energy, then
    # the true f should be 1/2 by orthogonality.
    sc = np.sqrt(r1["e2"] / r2["e2"])
    e_mix = e_grad + sc * e_null
    r3 = decompose(gg, p_face, e_mix, p_cell)
    ok(f"V8_N{N}", abs(r3["f"] - 0.5) < 0.03,
       f"equal-energy mixture -> f = {r3['f']:.5f} (expect 0.500)")

# ----------------------------------------------------------------- V9

print("\n=== V9: grid refinement and tol_density sensitivity ===")
fs = {}
for N in [128, 192, 256]:
    gg = FaceGrid(N, L)
    p_face, p_cell = grid_fields(gmm, gg, t_test)
    fp = torch.as_tensor(gg.face_pos)
    sc_g = analytic_gradient_field(fp).numpy()
    sc_n = analytic_stream_fluxnull(gmm, t_test, fp).numpy()
    e = face_normal(sc_g + 0.5 * sc_n, gg)
    t1 = time.time()
    r = decompose(gg, p_face, e, p_cell)
    fs[N] = r["f"]
    print(f"    N={N:4d}  f={r['f']:.6f}  ortho={r['ortho']:.2e}  "
          f"active={r['n_active_cell']}  {time.time()-t1:.1f}s")
ok("V9_refine", abs(fs[192] - fs[256]) < 0.01,
   f"f(192)={fs[192]:.5f} f(256)={fs[256]:.5f} |diff|={abs(fs[192]-fs[256]):.2e}")

gg = FaceGrid(192, L)
p_face, p_cell = grid_fields(gmm, gg, t_test)
fp = torch.as_tensor(gg.face_pos)
e = face_normal((analytic_gradient_field(fp) + 0.5 * analytic_stream_fluxnull(gmm, t_test, fp)).numpy(), gg)
ft = {}
for tol in [1e-6, 1e-8, 1e-10]:
    r = decompose(gg, p_face, e, p_cell, tol_density=tol)
    ft[tol] = r["f"]
    print(f"    tol_density={tol:.0e}  f={r['f']:.6f}  active_cells={r['n_active_cell']}")
spread = max(ft.values()) - min(ft.values())
ok("V9_tol", spread < 0.01, f"f spread over tol_density = {spread:.2e}")

# ----------------------------------------------------------------- V10

print("\n=== V10: MECHANISM EXISTENCE -- exact flux-null perturbation of v* ===")
# Build an exactly flux-null field and add it to v*. Theory says: identical
# marginals for all t, different transport map.
amp = 1.0

def v_perturbed(t, x):
    tt = min(max(t, 0.0), 1.0)
    xt_ = torch.as_tensor(x)
    base = gmm.v_star(tt, xt_)
    pert = analytic_stream_fluxnull(gmm, tt, xt_)
    return (base + amp * pert).numpy()

phi_pert, info_p = integrate_adaptive(v_perturbed, x0s, 0.0, 1.0, rtol=1e-11, atol=1e-13)
w2_pert = w2(phi_pert, d1)
map_shift = float(np.linalg.norm(phi_pert - phi_star, axis=1).mean())
data_scale = float(np.linalg.norm(d1 - d1.mean(0), axis=1).mean())

# relative energy of the perturbation against v* in L2(dt x p_t)
num, den = 0.0, 0.0
ts = np.linspace(0.02, 0.98, 25)
for t in ts:
    xs, _, _ = gmm.sample_x_t(20000, float(t), generator=torch.Generator().manual_seed(7))
    num += float((analytic_stream_fluxnull(gmm, float(t), xs) ** 2).sum(1).mean())
    den += float((gmm.v_star(float(t), xs) ** 2).sum(1).mean())
rel_amp = amp * np.sqrt(num / den)

ok("V10_marginal_preserved", w2_pert < 1.6 * floor,
   f"W2(Phi_pert#p0, p1) = {w2_pert:.4f} vs floor {floor:.4f} "
   f"(unperturbed {got:.4f})")
ok("V10_map_changed", map_shift > 0.25 * data_scale,
   f"mean |Phi_pert - Phi*| = {map_shift:.4f} = {map_shift/data_scale:.1%} of data radius; "
   f"perturbation is {rel_amp:.1%} of ||v*||")

report["V10_numbers"] = dict(w2_unperturbed=got, w2_perturbed=w2_pert, w2_floor=floor,
                             map_shift=map_shift, data_scale=data_scale,
                             rel_amplitude=rel_amp)
report["V9_refine_values"] = {str(k): v for k, v in fs.items()}
report["V9_tol_values"] = {f"{k:.0e}": v for k, v in ft.items()}

np.save(os.path.join(OUT, "v10_phi_star.npy"), phi_star)
np.save(os.path.join(OUT, "v10_phi_pert.npy"), phi_pert)
np.save(os.path.join(OUT, "v10_x0.npy"), x0s)

n_fail = sum(1 for v in report.values() if isinstance(v, dict) and v.get("status") == "FAIL")
print(f"\n==== E0 SUMMARY: {n_fail} failure(s) ====")
with open(os.path.join(OUT, "e0_report.json"), "w") as fh:
    json.dump(report, fh, indent=2)
print(f"written: {os.path.join(OUT, 'e0_report.json')}")
sys.exit(1 if n_fail else 0)
