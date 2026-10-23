#!/usr/bin/env python3
"""
Post-processing diagnostics for the 3D MHD model (mhd3d.cxx).

Everything is computed from the dumped state (density, pressure, v_x, v_y,
v_z, Bx, By, Bz) plus the metric, so the physics model carries no
diagnostic code.

Conventions matching mhd3d.cxx:
    v  is stored COVARIANT      -> v_x, v_y, v_z
    B  is stored CONTRAVARIANT  -> Bx,  By,  Bz

Curvilinear identities used (general, works for the flared metric):
    div V     = (1/J) d_i ( J V^i )                 [V contravariant]
    (curl A)^i = (1/J) eps^{ijk} d_j A_k            [A covariant]
    |V|^2     = g_ij V^i V^j     = g^ij V_i V_j
    v . B     = v_i B^i                             [no metric needed]
    integral  = sum f * J * dx * dy * dz

Derivatives use 2nd-order centred differences, matching first = C2 in
BOUT.inp. Periodic directions use np.roll and are exact to round-off;
non-periodic directions are masked one cell in from each end (no guard
cells are read), so derivative-based integrals are over a slightly
reduced volume -- reported explicitly.

Usage:
    python boutmhd.py -p data -o plots                 # Orszag-Tang (y periodic)
    python boutmhd.py -p data -o plots --y-bndry       # flared tube (y bounded)
    python boutmhd.py -p data --only budget health
"""

import argparse
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from boutdata.collect import collect


# ====================================================================== mesh
class Mesh:
    """Metric and geometry, broadcast to (nx, ny, nz)."""

    NAMES = ["J", "dx", "dy", "dz",
             "g11", "g22", "g33", "g12", "g13", "g23",
             "g_11", "g_22", "g_33", "g_12", "g_13", "g_23"]

    def __init__(self, path, shape, y_periodic=True):
        self.path = path
        self.shape = shape
        self.y_periodic = y_periodic
        self.m = {}
        for nm in self.NAMES:
            self.m[nm] = self._load(nm)
        self.dV = self.m["J"] * self.m["dx"] * self.m["dy"] * self.m["dz"]
        self.total_volume = float(np.sum(self.dV))

        self.diagonal = all(
            np.max(np.abs(self.m[k])) < 1e-13
            for k in ("g12", "g13", "g23", "g_12", "g_13", "g_23")
        )

    def _load(self, name):
        """Try 'm'-prefixed (written by mhd3d.cxx), then the plain name,
        then fall back to a sensible default."""
        for candidate in ("m" + name, name):
            try:
                a = np.asarray(collect(candidate, path=self.path,
                                       xguards=False, yguards=False, info=False))
            except Exception:
                continue
            if a.size == 0:
                continue
            a = np.squeeze(a)
            if a.ndim == 4:          # (t,x,y,z) -- metric is static
                a = a[0]
            return np.broadcast_to(np.atleast_1d(a).astype(float),
                                   self.shape).copy() if a.ndim == 3 \
                else np.broadcast_to(a.reshape(a.shape + (1,)), self.shape).copy() \
                if a.ndim == 2 else np.full(self.shape, float(a))
        # defaults: identity metric, unit spacing
        default = 0.0 if name in ("g12", "g13", "g23",
                                  "g_12", "g_13", "g_23") else 1.0
        print(f"  ! metric '{name}' not found; using {default}")
        return np.full(self.shape, default)

    def __getitem__(self, k):
        return self.m[k]

    # ------------------------------------------------------- derivatives
    def ddx(self, f):
        """x is periodic (periodicX = true)."""
        return (np.roll(f, -1, 0) - np.roll(f, 1, 0)) / (2.0 * self.m["dx"])

    def ddz(self, f):
        """z is always periodic in BOUT++."""
        return (np.roll(f, -1, 2) - np.roll(f, 1, 2)) / (2.0 * self.m["dz"])

    def ddy(self, f):
        if self.y_periodic:
            return (np.roll(f, -1, 1) - np.roll(f, 1, 1)) / (2.0 * self.m["dy"])
        d = np.full_like(f, np.nan)
        d[:, 1:-1, :] = (f[:, 2:, :] - f[:, :-2, :]) / (2.0 * self.m["dy"][:, 1:-1, :])
        return d

    # ------------------------------------------------ vector operators
    def div_contra(self, Vx, Vy, Vz):
        """Divergence of a contravariant vector."""
        J = self.m["J"]
        return (self.ddx(J * Vx) + self.ddy(J * Vy) + self.ddz(J * Vz)) / J

    def curl_cov(self, Ax, Ay, Az):
        """Curl of a covariant vector -> contravariant components."""
        J = self.m["J"]
        cx = (self.ddy(Az) - self.ddz(Ay)) / J
        cy = (self.ddz(Ax) - self.ddx(Az)) / J
        cz = (self.ddx(Ay) - self.ddy(Ax)) / J
        return cx, cy, cz

    def lower(self, Vx, Vy, Vz):
        """Contravariant -> covariant:  V_i = g_ij V^j."""
        g = self.m
        return (g["g_11"] * Vx + g["g_12"] * Vy + g["g_13"] * Vz,
                g["g_12"] * Vx + g["g_22"] * Vy + g["g_23"] * Vz,
                g["g_13"] * Vx + g["g_23"] * Vy + g["g_33"] * Vz)

    def raise_(self, Vx, Vy, Vz):
        """Covariant -> contravariant:  V^i = g^ij V_j."""
        g = self.m
        return (g["g11"] * Vx + g["g12"] * Vy + g["g13"] * Vz,
                g["g12"] * Vx + g["g22"] * Vy + g["g23"] * Vz,
                g["g13"] * Vx + g["g23"] * Vy + g["g33"] * Vz)

    def mag2_contra(self, Vx, Vy, Vz):
        """|V|^2 = g_ij V^i V^j."""
        lx, ly, lz = self.lower(Vx, Vy, Vz)
        return Vx * lx + Vy * ly + Vz * lz

    def mag2_cov(self, Vx, Vy, Vz):
        """|V|^2 = g^ij V_i V_j."""
        ux, uy, uz = self.raise_(Vx, Vy, Vz)
        return Vx * ux + Vy * uy + Vz * uz

    # ------------------------------------------------------- integrals
    def vol_int(self, f):
        """Volume integral, NaN-safe (masked cells excluded)."""
        w = f * self.dV
        return float(np.nansum(w))

    def valid_volume(self, f):
        """Volume over which f is finite -- so integrals can be normalised."""
        return float(np.sum(self.dV[np.isfinite(f)]))

    def vol_mean(self, f):
        vv = self.valid_volume(f)
        return self.vol_int(f) / vv if vv > 0 else np.nan


# ====================================================================== state
class State:
    """Derived quantities at one time index."""

    def __init__(self, run, tind):
        m = run.mesh
        self.mesh = m
        self.t = float(run.t[tind])

        self.rho = run.snap("density", tind)
        self.p = run.snap("pressure", tind)

        # v: covariant
        self.vx_l = run.snap("v_x", tind)
        self.vy_l = run.snap("v_y", tind)
        self.vz_l = run.snap("v_z", tind)
        # B: contravariant
        self.Bx_u = run.snap("Bx", tind)
        self.By_u = run.snap("By", tind)
        self.Bz_u = run.snap("Bz", tind)

        self.v2 = m.mag2_cov(self.vx_l, self.vy_l, self.vz_l)
        self.B2 = m.mag2_contra(self.Bx_u, self.By_u, self.Bz_u)
        self.Bmag = np.sqrt(np.maximum(self.B2, 0.0))

        # v . B = v_i B^i, no metric required
        self.vdotB = (self.vx_l * self.Bx_u + self.vy_l * self.By_u
                      + self.vz_l * self.Bz_u)

        # div B, from contravariant B
        self.divB = m.div_contra(self.Bx_u, self.By_u, self.Bz_u)

        # div v: raise v first
        vxu, vyu, vzu = m.raise_(self.vx_l, self.vy_l, self.vz_l)
        self.divV = m.div_contra(vxu, vyu, vzu)

        # current: lower B, take curl -> contravariant J^i
        Bxl, Byl, Bzl = m.lower(self.Bx_u, self.By_u, self.Bz_u)
        Jx, Jy, Jz = m.curl_cov(Bxl, Byl, Bzl)
        self.J2 = m.mag2_contra(Jx, Jy, Jz)

        # vorticity: v already covariant
        wx, wy, wz = m.curl_cov(self.vx_l, self.vy_l, self.vz_l)
        self.w2 = m.mag2_contra(wx, wy, wz)

        self.cs2 = run.gamma * self.p / self.rho

    # --------------------------------------------------------- integrals
    def scalars(self, gamma):
        m = self.mesh
        Ekin = m.vol_int(0.5 * self.rho * self.v2)
        Emag = m.vol_int(0.5 * self.B2)
        Eth = m.vol_int(self.p / (gamma - 1.0))
        Brms = np.sqrt(m.vol_mean(self.B2))
        dl = np.cbrt(m.dV)

        out = {
            "t": self.t,
            "mass": m.vol_int(self.rho),
            "Ekin": Ekin, "Emag": Emag, "Eth": Eth,
            "Etot": Ekin + Emag + Eth,
            "Px": m.vol_int(self.rho * self.vx_l),
            "Py": m.vol_int(self.rho * self.vy_l),
            "Pz": m.vol_int(self.rho * self.vz_l),
            "Hcross": m.vol_int(self.vdotB),
            "intJ2": m.vol_int(self.J2),
            "intW2": m.vol_int(self.w2),
            "intDivV2": m.vol_int(self.divV**2),
            "p_min": float(np.min(self.p)),
            "rho_min": float(np.min(self.rho)),
            "J_max": float(np.sqrt(np.nanmax(self.J2))),
            "Ms_max": float(np.sqrt(np.max(self.v2 / self.cs2))),
            "divV_min": float(np.nanmin(self.divV)),
            "divB_max_norm": float(np.nanmax(np.abs(self.divB) * dl)) / (Brms + 1e-30),
            "divB_L2_norm": (np.sqrt(m.vol_int(self.divB**2 * dl**2))
                             / (np.sqrt(m.vol_int(self.B2)) + 1e-30)),
        }
        return out


# ======================================================================== run
class Run:
    def __init__(self, path, gamma=5.0 / 3.0, y_periodic=True):
        self.path = path
        self.gamma = gamma
        self._cache = {}

        self.t = np.asarray(collect("t_array", path=path, info=False)).ravel()
        probe = self._raw("density", 0)
        self.shape = probe.shape
        self.mesh = Mesh(path, self.shape, y_periodic=y_periodic)

    def _raw(self, name, tind):
        a = np.asarray(collect(name, path=self.path, tind=[tind, tind],
                               xguards=False, yguards=False, info=False))
        return np.squeeze(a, axis=0) if a.ndim == 4 else a

    def snap(self, name, tind):
        key = (name, tind)
        if key not in self._cache:
            self._cache[key] = self._raw(name, tind)
        return self._cache[key]

    def state(self, tind):
        return State(self, tind)

    def traces(self, stride=1):
        """Scalar time series, recomputed from 3D snapshots."""
        inds = range(0, len(self.t), stride)
        rows = []
        for i in inds:
            s = self.state(i)
            rows.append(s.scalars(self.gamma))
            self._cache.clear()          # keep memory bounded
            print(f"    t = {s.t:9.4f}   Etot = {rows[-1]['Etot']:.6e}")
        return {k: np.array([r[k] for r in rows]) for k in rows[0]}


# =================================================================== analytic
def ot_reference():
    """t=0 values for the cyclic 3D Orszag-Tang IC on [0,2pi]^3."""
    rho, p, V = 25 / (36 * np.pi), 5 / (12 * np.pi), (2 * np.pi) ** 3
    return {
        "volume": V,
        "mass": rho * V,
        "Ekin": 0.5 * rho * 1.5 * V,
        "Emag": 0.5 * (1.5 / (4 * np.pi)) * V,
        "Eth": p / (5 / 3 - 1) * V,
        "Hcross": 0.5 * V / np.sqrt(4 * np.pi),
        "intW2": 1.5 * V,
        "intJ2": (4.5 / (4 * np.pi)) * V,
    }


# ====================================================================== plots
def budget(tr, outdir, ref=None):
    print("\n[budget]")
    t, Et = tr["t"], tr["Etot"]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    for k, lab in (("Ekin", "kinetic"), ("Emag", "magnetic"), ("Eth", "thermal")):
        ax[0].plot(t, tr[k], label=lab)
    ax[0].plot(t, Et, "k--", lw=2, label="total")
    ax[0].set_xlabel("t"); ax[0].set_ylabel("energy"); ax[0].legend()
    ax[0].set_title("Energy budget")

    ax[1].plot(t, (Et - Et[0]) / Et[0]); ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set_xlabel("t"); ax[1].set_ylabel(r"$\Delta E_{tot}/E_{tot}(0)$")
    ax[1].set_title("Total energy drift")

    for k, lab in (("Ekin", "kinetic"), ("Emag", "magnetic"), ("Eth", "thermal")):
        ax[2].plot(t, tr[k] - tr[k][0], label=lab)
    ax[2].axhline(0, color="k", lw=0.5); ax[2].legend()
    ax[2].set_xlabel("t"); ax[2].set_ylabel(r"$E-E(0)$")
    ax[2].set_title("Change from t=0")
    fig.tight_layout(); fig.savefig(f"{outdir}/budget.png", dpi=140); plt.close(fig)

    print(f"    Etot(0)      = {Et[0]:.6f}")
    print(f"    energy loss  = {100*(Et[0]-Et[-1])/Et[0]:.3f} %  by t={t[-1]:.3f}")
    print(f"    max |dM/M|   = {np.abs(tr['mass']/tr['mass'][0]-1).max():.3e}")
    print(f"    max |P|      = {max(np.abs(tr[k]).max() for k in ('Px','Py','Pz')):.3e}")
    if ref:
        for k in ("mass", "Ekin", "Emag", "Eth", "Hcross", "intW2", "intJ2"):
            if k in tr and k in ref:
                got, want = tr[k][0], ref[k]
                print(f"    {k:<8s} {got:14.6e}  analytic {want:14.6e}"
                      f"   rel {abs(got-want)/abs(want):.2e}")


def closure(tr, outdir, rho0, viscos, eta):
    """dE/dt vs the predicted explicit-dissipation sink."""
    print("\n[closure]")
    t, Et = tr["t"], tr["Etot"]
    if len(t) < 3:
        print("    need >=3 outputs"); return
    dEdt = np.gradient(Et, t)
    pred = -(rho0 * viscos * tr["intW2"] + eta * tr["intJ2"])

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
    ax[0].plot(t, dEdt, label=r"measured $dE_{tot}/dt$")
    ax[0].plot(t, pred, "--", label=r"$-(\rho\nu\int\omega^2+\eta\int J^2)$")
    ax[0].axhline(0, color="k", lw=0.5); ax[0].legend()
    ax[0].set_xlabel("t"); ax[0].set_title("Energy closure")

    excess = dEdt - pred
    ax[1].plot(t, -excess / Et[0])
    ax[1].set_xlabel("t"); ax[1].set_ylabel("numerical loss rate / Etot(0)")
    ax[1].set_title("Unaccounted dissipation")
    fig.tight_layout(); fig.savefig(f"{outdir}/closure.png", dpi=140); plt.close(fig)

    n = max(1, len(t) // 4)
    print(f"    early measured dE/dt = {dEdt[:n].mean():+.4e}")
    print(f"    early predicted      = {pred[:n].mean():+.4e}")
    print(f"    early numerical      = {(dEdt-pred)[:n].mean():+.4e}"
          f"  ({100*abs((dEdt-pred)[:n].mean())/Et[0]:.4f} %/unit t)")


def health(tr, outdir):
    print("\n[health]")
    t = tr["t"]
    fig, ax = plt.subplots(2, 3, figsize=(16, 8))
    a = ax[0, 0]
    a.semilogy(t, np.abs(tr["divB_max_norm"]) + 1e-300, label="max")
    a.semilogy(t, np.abs(tr["divB_L2_norm"]) + 1e-300, label="L2")
    a.axhline(1e-2, color="r", ls=":"); a.legend()
    a.set_xlabel("t"); a.set_title(r"$|\nabla\cdot B|\,\Delta l/B_{rms}$")

    a = ax[0, 1]
    a.plot(t, tr["p_min"], label="min p"); a.plot(t, tr["rho_min"], label=r"min $\rho$")
    a.axhline(0, color="r", ls=":"); a.legend()
    a.set_xlabel("t"); a.set_title("Positivity")

    a = ax[0, 2]
    a.plot(t, tr["intJ2"]); ipk = int(np.argmax(tr["intJ2"]))
    a.axvline(t[ipk], color="r", ls=":")
    a.set_xlabel("t"); a.set_ylabel(r"$\int J^2 dV$")
    a.set_title(f"peak t={t[ipk]:.3f}")
    print(f"    int J^2 peak: t={t[ipk]:.4f}  value={tr['intJ2'][ipk]:.5e}")

    ax[1, 0].plot(t, tr["intDivV2"]); ax[1, 0].set_xlabel("t")
    ax[1, 0].set_title(r"$\int(\nabla\cdot v)^2 dV$")
    ax[1, 1].plot(t, tr["divV_min"]); ax[1, 1].set_xlabel("t")
    ax[1, 1].set_title(r"$\min\nabla\cdot v$")
    ax[1, 2].plot(t, tr["J_max"], label=r"max$|J|$")
    ax[1, 2].plot(t, tr["Ms_max"], label=r"max$M_s$")
    ax[1, 2].legend(); ax[1, 2].set_xlabel("t"); ax[1, 2].set_title("Extrema")
    fig.tight_layout(); fig.savefig(f"{outdir}/health.png", dpi=140); plt.close(fig)


def spectra(run, outdir, tind):
    print("\n[spectra]")
    s = run.state(tind)
    nx, ny, nz = run.shape
    kx, ky, kz = (np.fft.fftfreq(n, d=1.0 / n) for n in (nx, ny, nz))
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    kmag = np.sqrt(KX**2 + KY**2 + KZ**2)
    edges = np.arange(0.5, min(nx, ny, nz) // 2 + 0.5, 1.0)
    kc = 0.5 * (edges[1:] + edges[:-1])
    idx = np.digitize(kmag.ravel(), edges)

    def shell(fields):
        E = np.zeros(kmag.shape)
        for f in fields:
            E += 0.5 * np.abs(np.fft.fftn(f) / f.size) ** 2
        return np.bincount(idx, weights=E.ravel(), minlength=len(edges) + 1)[1:len(edges)]

    sq = np.sqrt(np.maximum(s.rho, 0))
    Ev = shell([sq * s.vx_l, sq * s.vy_l, sq * s.vz_l])
    Eb = shell([s.Bx_u, s.By_u, s.Bz_u])

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.loglog(kc, Ev + 1e-300, label="kinetic")
    ax.loglog(kc, Eb + 1e-300, label="magnetic")
    r = kc >= 2
    nrm = (Ev + Eb)[r][0] * kc[r][0] ** 2
    ax.loglog(kc[r], nrm * kc[r] ** -2.0, "k:", label=r"$k^{-2}$")
    ax.loglog(kc[r], nrm * kc[r] ** (-5 / 3), "k--", label=r"$k^{-5/3}$")
    ax.set_xlabel("k"); ax.set_ylabel("E(k)"); ax.legend()
    ax.set_title(f"t = {s.t:.3f}")
    fig.tight_layout(); fig.savefig(f"{outdir}/spectra.png", dpi=140); plt.close(fig)

    tot = Ev + Eb
    print(f"    E(k_max)/E(k_peak) = {tot[-1]/tot.max():.3e}")


def slices(run, outdir, tind):
    print("\n[slices]")
    s = run.state(tind)
    nx, ny, nz = run.shape
    k = nz // 2
    panels = [(np.sqrt(np.maximum(s.J2, 0))[:, :, k], r"$|J|$", "magma"),
              (s.rho[:, :, k], r"$\rho$", "viridis"),
              (s.p[:, :, k], r"$p$", "inferno"),
              (s.divV[:, :, k], r"$\nabla\cdot v$", "RdBu_r"),
              (s.divB[:, :, k], r"$\nabla\cdot B$", "RdBu_r"),
              (s.Bmag[:, :, k], r"$|B|$", "cividis")]
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    for a, (f, lab, cm) in zip(ax.ravel(), panels):
        kw = {}
        if cm == "RdBu_r":
            mx = np.nanmax(np.abs(f)); kw = dict(vmin=-mx, vmax=mx)
        im = a.imshow(f.T, origin="lower", cmap=cm, **kw)
        a.set_title(f"{lab}  (mid-z, t={s.t:.3f})")
        a.set_xlabel("x index"); a.set_ylabel("y index")
        fig.colorbar(im, ax=a)
    fig.tight_layout(); fig.savefig(f"{outdir}/slices.png", dpi=140); plt.close(fig)


# ======================================================================= main
ALL = ["budget", "closure", "health", "spectra", "slices"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", default="data")
    ap.add_argument("-o", "--outdir", default="plots")
    ap.add_argument("--tind", type=int, default=-1)
    ap.add_argument("--stride", type=int, default=1,
                    help="time stride for the scalar traces")
    ap.add_argument("--gamma", type=float, default=5.0 / 3.0)
    ap.add_argument("--y-bndry", action="store_true",
                    help="y has boundaries (flared tube), not periodic")
    ap.add_argument("--rho0", type=float, default=1.0, help="for the closure test")
    ap.add_argument("--viscos", type=float, default=0.0)
    ap.add_argument("--eta", type=float, default=0.0)
    ap.add_argument("--ot-ref", action="store_true",
                    help="compare t=0 to analytic 3D Orszag-Tang values")
    ap.add_argument("--only", nargs="+", choices=ALL, default=ALL)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    run = Run(args.path, gamma=args.gamma, y_periodic=not args.y_bndry)
    nx, ny, nz = run.shape
    tind = args.tind if args.tind >= 0 else len(run.t) + args.tind

    print("=" * 66)
    print(f"  path     : {args.path}")
    print(f"  grid     : {nx} x {ny} x {nz}   (interior)")
    print(f"  outputs  : {len(run.t)}  t = {run.t[0]:.4f} .. {run.t[-1]:.4f}")
    print(f"  volume   : {run.mesh.total_volume:.6e}")
    print(f"  metric   : {'diagonal' if run.mesh.diagonal else 'NON-diagonal'}")
    print(f"  y        : {'periodic' if run.mesh.y_periodic else 'bounded (masked)'}")
    print("=" * 66)

    tr = None
    if any(k in args.only for k in ("budget", "closure", "health")):
        print("\ncomputing scalar traces from 3D snapshots ...")
        tr = run.traces(stride=args.stride)

    jobs = {
        "budget": lambda: budget(tr, args.outdir,
                                 ot_reference() if args.ot_ref else None),
        "closure": lambda: closure(tr, args.outdir, args.rho0,
                                   args.viscos, args.eta),
        "health": lambda: health(tr, args.outdir),
        "spectra": lambda: spectra(run, args.outdir, tind),
        "slices": lambda: slices(run, args.outdir, tind),
    }
    for nm in args.only:
        try:
            jobs[nm]()
        except Exception as exc:
            print(f"  !! {nm} failed: {type(exc).__name__}: {exc}", file=sys.stderr)

    print(f"\nplots -> {args.outdir}/")


if __name__ == "__main__":
    main()
