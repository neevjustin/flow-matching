"""Weighted Helmholtz / Hodge decomposition of a vector field in L^2(p).

Implements theory.md Sec. 5. Given a vector field e and a density p on a 2-D
box, split

    e = grad(phi) + r,     div(p r) = 0,     <grad(phi), r>_{L^2(p)} = 0

and report the flux-null energy fraction f = ||r||^2 / ||e||^2.

Discretisation
--------------
Finite volume on a uniform cell-centred grid with the field sampled at FACE
MIDPOINTS (no interpolation: both p and e are evaluated there exactly).

    cell centres   x_i = -L + (i + 1/2) h,            i = 0..N-1,  h = 2L/N
    x-faces        between (i,j) and (i+1,j),         i = 0..N-2
    y-faces        between (i,j) and (i,j+1),         j = 0..N-2

Only interior faces are used, which imposes the natural zero-flux boundary
condition. Weighted inner product on faces:

    <a, b>_h = h^2 * sum_f p_f a_f b_f

With G the face-cell incidence matrix ((G phi)_f = phi_B - phi_A) and
W = diag(p_f), the Galerkin condition <grad_h phi, grad_h psi>_h =
<e, grad_h psi>_h for all psi becomes

    (G^T W G) phi = h * G^T W e                                       (*)

and then grad_h phi = (G phi)/h, r = e - grad_h phi.

Exact discrete orthogonality
----------------------------
Taking psi = phi in the Galerkin condition gives <grad_h phi, r>_h = 0
IDENTICALLY, so f in [0, 1] by construction and
||e||^2 = ||grad phi||^2 + ||r||^2 to linear-solver accuracy. The measured
normalised orthogonality residual is therefore a direct diagnostic of how well
(*) was solved -- it is reported with every decomposition and must be checked.

Active set
----------
Cells where p < tol_density * max(p) are dropped entirely from both the solve
and the norms. This is the correct discretisation of the continuum L^2(p)
integral: those regions carry negligible true weight. It is NOT the same as
flooring the weights, which would wrongly promote the (potentially huge)
network extrapolation error in the far tails. Sensitivity to tol_density is
swept in the E0 validation.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class FaceGrid:
    """Geometry and index bookkeeping for a cell-centred grid with face fields."""

    def __init__(self, N, L):
        self.N, self.L = N, L
        self.h = 2.0 * L / N
        c = -L + (np.arange(N) + 0.5) * self.h          # cell-centre coords
        self.centres = c
        # cell index map
        self.ncell = N * N
        cid = np.arange(self.ncell).reshape(N, N)       # [i, j] -> flat
        self.cid = cid

        # --- x-faces: (i,j)-(i+1,j), i=0..N-2 ---
        ii, jj = np.meshgrid(np.arange(N - 1), np.arange(N), indexing="ij")
        self.xf_A = cid[ii, jj].ravel()
        self.xf_B = cid[ii + 1, jj].ravel()
        xf_x = -L + (ii.ravel() + 1) * self.h
        xf_y = c[jj.ravel()]
        self.xf_pos = np.stack([xf_x, xf_y], axis=1)

        # --- y-faces: (i,j)-(i,j+1), j=0..N-2 ---
        ii, jj = np.meshgrid(np.arange(N), np.arange(N - 1), indexing="ij")
        self.yf_A = cid[ii, jj].ravel()
        self.yf_B = cid[ii, jj + 1].ravel()
        yf_x = c[ii.ravel()]
        yf_y = -L + (jj.ravel() + 1) * self.h
        self.yf_pos = np.stack([yf_x, yf_y], axis=1)

        self.nxf = self.xf_A.size
        self.nyf = self.yf_A.size
        self.nface = self.nxf + self.nyf

        self.face_pos = np.concatenate([self.xf_pos, self.yf_pos], axis=0)
        self.face_A = np.concatenate([self.xf_A, self.yf_A])
        self.face_B = np.concatenate([self.xf_B, self.yf_B])
        # which component of the vector field lives on each face
        self.face_comp = np.concatenate(
            [np.zeros(self.nxf, dtype=np.int64), np.ones(self.nyf, dtype=np.int64)]
        )

    def cell_positions(self):
        ii, jj = np.meshgrid(np.arange(self.N), np.arange(self.N), indexing="ij")
        return np.stack([self.centres[ii.ravel()], self.centres[jj.ravel()]], axis=1)


def active_masks(grid, p_cell, tol_density=1e-8):
    """Return (active_cell, active_face) using the same rule as ``decompose``.

    Exposed so callers can evaluate an expensive field (e.g. a network) only on
    the faces that will actually be used, instead of all of them.
    """
    thr = tol_density * float(p_cell.max())
    act_cell = p_cell >= thr
    return act_cell, act_cell[grid.face_A] & act_cell[grid.face_B]


def decompose(grid, p_face, e_face, p_cell, tol_density=1e-8, solver="direct"):
    """Decompose a face-sampled field.

    Parameters
    ----------
    grid      : FaceGrid
    p_face    : (nface,)  density at face midpoints
    e_face    : (nface,)  normal component of the field at each face
                          (x-component on x-faces, y-component on y-faces)
    p_cell    : (ncell,)  density at cell centres, used to build the active set
    tol_density : cells with p_cell < tol_density * p_cell.max() are dropped

    Returns
    -------
    dict with keys:
      f          flux-null energy fraction ||r||^2 / ||e||^2
      e2, g2, r2 squared weighted norms of e, grad(phi), r
      ortho      normalised |<grad phi, r>| / (||grad phi|| ||r||)  (solver check)
      closure    |e2 - (g2 + r2)| / e2                             (identity check)
      r_face, g_face  the two components on faces (for spatial maps)
      active_face     boolean mask of faces retained
    """
    N, h = grid.N, grid.h

    # ---- active set of cells, then faces whose BOTH neighbours are active ----
    thr = tol_density * float(p_cell.max())
    act_cell = p_cell >= thr
    act_face = act_cell[grid.face_A] & act_cell[grid.face_B]

    # local (compressed) cell numbering over active cells
    cell_loc = -np.ones(grid.ncell, dtype=np.int64)
    act_idx = np.flatnonzero(act_cell)
    cell_loc[act_idx] = np.arange(act_idx.size)
    nloc = act_idx.size

    fA = cell_loc[grid.face_A[act_face]]
    fB = cell_loc[grid.face_B[act_face]]
    w = p_face[act_face]
    ef = e_face[act_face]
    nf = fA.size

    # ---- incidence matrix G: (nf, nloc), row f = +1 at B, -1 at A ----
    rows = np.repeat(np.arange(nf), 2)
    cols = np.empty(2 * nf, dtype=np.int64)
    cols[0::2], cols[1::2] = fB, fA
    vals = np.empty(2 * nf)
    vals[0::2], vals[1::2] = 1.0, -1.0
    G = sp.csr_matrix((vals, (rows, cols)), shape=(nf, nloc))

    W = sp.diags(w)
    A = (G.T @ W @ G).tocsr()
    rhs = h * (G.T @ (w * ef))

    # ---- pin one DOF (A is singular on constants within each component) ----
    # pin the highest-density active cell, which is certainly in the main
    # connected component
    pin = int(np.argmax(p_cell[act_idx]))
    A = A.tolil()
    A[pin, :] = 0.0
    A[:, pin] = 0.0
    A[pin, pin] = 1.0
    A = A.tocsc()
    rhs = rhs.copy()
    rhs[pin] = 0.0

    # isolated active cells (all incident faces dropped) give empty rows ->
    # regularise them so the factorisation does not fail
    deg = np.asarray(abs(A).sum(axis=1)).ravel()
    dead = np.flatnonzero(deg == 0.0)
    if dead.size:
        A = A.tolil()
        for k in dead:
            A[k, k] = 1.0
        A = A.tocsc()
        rhs[dead] = 0.0

    if solver == "direct":
        phi = spla.spsolve(A, rhs)
    else:
        phi, info = spla.cg(A, rhs, rtol=1e-12, maxiter=20000)
        if info != 0:
            raise RuntimeError(f"CG did not converge: info={info}")

    gf = (G @ phi) / h                      # grad_h phi on active faces
    rf = ef - gf

    m = h * h
    e2 = m * float(np.sum(w * ef * ef))
    g2 = m * float(np.sum(w * gf * gf))
    r2 = m * float(np.sum(w * rf * rf))
    cross = m * float(np.sum(w * gf * rf))

    denom = np.sqrt(g2 * r2)
    ortho = abs(cross) / denom if denom > 0 else 0.0
    closure = abs(e2 - (g2 + r2)) / e2 if e2 > 0 else 0.0

    g_full = np.zeros(grid.nface)
    r_full = np.zeros(grid.nface)
    g_full[act_face] = gf
    r_full[act_face] = rf

    return dict(
        f=r2 / e2 if e2 > 0 else float("nan"),
        e2=e2, g2=g2, r2=r2, ortho=ortho, closure=closure,
        g_face=g_full, r_face=r_full, active_face=act_face,
        n_active_cell=int(nloc), n_active_face=int(nf),
    )
