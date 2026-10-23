#!/usr/bin/env python3
"""
Diagnostics for the 3D Orszag-Tang vortex run with mhd3d.cxx.

Variable names match the outputVars() implementation in mhd3d.cxx:

  0-D traces : mass Ekin Emag Eth Etot Px Py Pz Hcross intJ2 intW2
               intDivV2 p_min rho_min divV_min J_max Ms_max
               divB_max_norm divB_L2_norm volume
  3-D fields : density pressure v_x v_y v_z Bx By Bz
               divB J2 divV beta Ms

Usage:
    python otdiag.py -p build/data -o plots
    python otdiag.py -p build/data --only budget health
"""

import argparse
import os
import re
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless-safe; drop for interactive use
import matplotlib.pyplot as plt

from boutdata.collect import collect

GAMMA = 5.0 / 3.0

# ---------------------------------------------------------------- analytic
# Cyclic 3D Orszag-Tang on [0, 2pi]^3 with mu_0 = 1:
#   rho = 25/(36 pi), p = 5/(12 pi)
#   v = (-sin Z,  sin X,  sin Y)
#   B = (-sin Z, sin 2X, sin 2Y) / sqrt(4 pi)
RHO_A = 25.0 / (36.0 * np.pi)
P_A = 5.0 / (12.0 * np.pi)
VOL_A = (2.0 * np.pi) ** 3
EKIN_A = 0.5 * RHO_A * 1.5 * VOL_A
EMAG_A = 0.5 * (1.5 / (4.0 * np.pi)) * VOL_A
ETH_A = P_A / (GAMMA - 1.0) * VOL_A
ETOT_A = EKIN_A + EMAG_A + ETH_A
HC_A = 0.5 * VOL_A / np.sqrt(4.0 * np.pi)


# ================================================================ readers
class Run:
    """Lazy, cached access to one BOUT++ output directory."""

    def __init__(self, path):
        self.path = path
        self._cache = {}
        self._shape = None
        self._log = None

    # ---------------------------------------------------------- 3D fields
    def field(self, name, **kw):
        kw.setdefault("xguards", False)
        kw.setdefault("yguards", False)
        kw.setdefault("info", False)
        return np.asarray(collect(name, path=self.path, **kw))

    def snapshot(self, name, tind):
        """Single time slice, shape (nx, ny, nz)."""
        key = (name, tind)
        if key not in self._cache:
            arr = self.field(name, tind=[tind, tind])
            self._cache[key] = np.squeeze(arr, axis=0) if arr.ndim == 4 else arr
        return self._cache[key]

    @property
    def shape(self):
        if self._shape is None:
            self._shape = self.snapshot("density", 0).shape
        return self._shape

    # ---------------------------------------------------------- time axis
    @property
    def t(self):
        if "t" not in self._cache:
            self._cache["t"] = np.asarray(
                collect("t_array", path=self.path, info=False)
            ).ravel()
        return self._cache["t"]

    # ------------------------------------------------------- 0-D traces
    def trace(self, name):
        """
        Scalar time trace. Tries the dump files first; falls back to
        parsing the [diag] lines that mhd3d.cxx writes to BOUT.log.0
        (useful if assignRepeat<BoutReal> is not supported by your
        BOUT++ version, in which case the scalars are absent from the
        NetCDF output but still present in the log).
        """
        if name in self._cache:
            return self._cache[name]

        val = None
        try:
            val = np.asarray(collect(name, path=self.path, info=False)).ravel()
            if val.size == 0:
                val = None
        except Exception:
            val = None

        if val is None:
            val = self._from_log(name)

        if val is None:
            raise KeyError(
                f"could not find '{name}' in dump files or BOUT.log.0 "
                f"under {self.path}"
            )

        self._cache[name] = val
        return val

    def _from_log(self, name):
        """Parse '[diag] Etot=... Ekin=... ' lines from BOUT.log.0."""
        if self._log is None:
            self._log = {}
            logfile = os.path.join(self.path, "BOUT.log.0")
            if not os.path.exists(logfile):
                return None
            pat = re.compile(r"([A-Za-z_]\w*)=([-+0-9.eE]+)")
            with open(logfile, errors="ignore") as fh:
                for line in fh:
                    if not line.startswith("[diag]"):
                        continue
                    for k, v in pat.findall(line):
                        self._log.setdefault(k, []).append(float(v))
            self._log = {k: np.array(v) for k, v in self._log.items()}
            if self._log:
                print(f"  (recovered {sorted(self._log)} from BOUT.log.0)")
        # log uses 'divBn' for divB_max_norm, 'pmin' for p_min
        alias = {"divB_max_norm": "divBn", "p_min": "pmin"}
        return self._log.get(name, self._log.get(alias.get(name, name)))

    def time_for(self, y):
        """Match the time axis length to a trace (log may have an extra t=0)."""
        t = self.t
        if len(y) == len(t):
            return t
        n = min(len(y), len(t))
        return t[:n]


def _trim(t, y):
    n = min(len(t), len(y))
    return t[:n], y[:n]


# ================================================================ 1. budget
def budget(run, outdir):
    print("\n[1] energy budget")
    t = run.t
    Ek, Em, Eth = (run.trace(k) for k in ("Ekin", "Emag", "Eth"))
    try:
        Et = run.trace("Etot")
    except KeyError:
        Et = Ek + Em + Eth

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))

    for arr, lab in ((Ek, "kinetic"), (Em, "magnetic"), (Eth, "thermal")):
        ax[0].plot(*_trim(t, arr), label=lab)
    ax[0].plot(*_trim(t, Et), "k--", lw=2, label="total")
    ax[0].set_xlabel("t")
    ax[0].set_ylabel("energy")
    ax[0].set_title("Energy budget")
    ax[0].legend()

    tt, EE = _trim(t, Et)
    ax[1].plot(tt, (EE - EE[0]) / EE[0])
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set_xlabel("t")
    ax[1].set_ylabel(r"$\Delta E_{tot}/E_{tot}(0)$")
    ax[1].set_title("Total energy drift")

    # changes relative to t=0, which is where the physics shows up
    for arr, lab in ((Ek, "kinetic"), (Em, "magnetic"), (Eth, "thermal")):
        ts, a = _trim(t, arr)
        ax[2].plot(ts, a - a[0], label=lab)
    ax[2].axhline(0, color="k", lw=0.5)
    ax[2].set_xlabel("t")
    ax[2].set_ylabel(r"$E - E(0)$")
    ax[2].set_title("Change from t=0")
    ax[2].legend()

    fig.tight_layout()
    fig.savefig(f"{outdir}/budget.png", dpi=140)
    plt.close(fig)

    print(f"    E_tot(0)    = {EE[0]:12.5f}   analytic {ETOT_A:12.5f}"
          f"   rel {abs(EE[0]-ETOT_A)/ETOT_A:.2e}")
    print(f"    E_kin(0)    = {Ek[0]:12.5f}   analytic {EKIN_A:12.5f}")
    print(f"    E_mag(0)    = {Em[0]:12.5f}   analytic {EMAG_A:12.5f}")
    print(f"    E_th(0)     = {Eth[0]:12.5f}   analytic {ETH_A:12.5f}")
    print(f"    energy loss = {100*(EE[0]-EE[-1])/EE[0]:8.3f} %  by t={tt[-1]:.3f}")

    try:
        m = run.trace("mass")
        print(f"    max |dM/M|  = {np.abs(m/m[0]-1).max():.3e}  (want < 1e-12)")
    except KeyError:
        pass

    try:
        P = np.array([run.trace(k) for k in ("Px", "Py", "Pz")])
        print(f"    max |P|     = {np.abs(P).max():.3e}  (want ~ 0)")
    except KeyError:
        pass


# ================================================================ 2. health
def health(run, outdir):
    print("\n[2] numerical health")
    t = run.t
    fig, ax = plt.subplots(2, 3, figsize=(16, 8))

    # div B
    a = ax[0, 0]
    for nm, lab in (("divB_max_norm", "max"), ("divB_L2_norm", "L2")):
        try:
            a.semilogy(*_trim(t, np.abs(run.trace(nm)) + 1e-300), label=lab)
        except KeyError:
            pass
    a.axhline(1e-2, color="r", ls=":", label="tolerance")
    a.set_xlabel("t")
    a.set_ylabel(r"$|\nabla\cdot B|\,\Delta l / B_{rms}$")
    a.set_title("Divergence error")
    a.legend()

    # positivity
    a = ax[0, 1]
    a.plot(*_trim(t, run.trace("p_min")), label=r"min $p$")
    try:
        a.plot(*_trim(t, run.trace("rho_min")), label=r"min $\rho$")
    except KeyError:
        pass
    a.axhline(0, color="r", ls=":")
    a.set_xlabel("t")
    a.set_title("Positivity")
    a.legend()

    # int J^2  -- the classic OT diagnostic
    a = ax[0, 2]
    J2 = run.trace("intJ2")
    ts, y = _trim(t, J2)
    a.plot(ts, y)
    ipk = int(np.argmax(y))
    a.axvline(ts[ipk], color="r", ls=":")
    a.set_xlabel("t")
    a.set_ylabel(r"$\int J^2\,dV$")
    a.set_title(f"Current sheets (peak t={ts[ipk]:.3f})")
    print(f"    int J^2 peak: t = {ts[ipk]:.4f},  value = {y[ipk]:.5e}")

    # compressive activity -> shock formation
    a = ax[1, 0]
    try:
        a.plot(*_trim(t, run.trace("intDivV2")))
        a.set_ylabel(r"$\int (\nabla\cdot v)^2 dV$")
    except KeyError:
        a.set_visible(True)
    a.set_xlabel("t")
    a.set_title("Compressive activity")

    a = ax[1, 1]
    try:
        ts, dv = _trim(t, run.trace("divV_min"))
        a.plot(ts, dv)
        a.set_ylabel(r"$\min\,\nabla\cdot v$")
        # first strongly compressive time = shock onset
        thresh = 0.2 * dv.min()
        idx = np.where(dv < thresh)[0]
        if idx.size:
            a.axvline(ts[idx[0]], color="r", ls=":")
            print(f"    shock onset (min div v < 20% of extremum): "
                  f"t ~ {ts[idx[0]]:.3f}")
    except KeyError:
        pass
    a.set_xlabel("t")
    a.set_title("Strongest compression")

    a = ax[1, 2]
    for nm, lab in (("J_max", r"max $|J|$"), ("Ms_max", r"max $M_s$")):
        try:
            a.plot(*_trim(t, run.trace(nm)), label=lab)
        except KeyError:
            pass
    try:
        a.plot(*_trim(t, run.trace("Hcross") / HC_A), label=r"$H_c/H_c(0)_{an}$")
    except KeyError:
        pass
    a.set_xlabel("t")
    a.set_title("Extrema / helicity")
    a.legend()

    fig.tight_layout()
    fig.savefig(f"{outdir}/health.png", dpi=140)
    plt.close(fig)


# ================================================================ 3. spectra
def spectra(run, outdir, tind):
    print("\n[3] shell-averaged spectra")
    nx, ny, nz = run.shape
    rho = run.snapshot("density", tind)

    kx = np.fft.fftfreq(nx, d=1.0 / nx)
    ky = np.fft.fftfreq(ny, d=1.0 / ny)
    kz = np.fft.fftfreq(nz, d=1.0 / nz)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    kmag = np.sqrt(KX**2 + KY**2 + KZ**2)

    nbin = min(nx, ny, nz) // 2
    edges = np.arange(0.5, nbin + 0.5, 1.0)
    kcen = 0.5 * (edges[1:] + edges[:-1])
    idx = np.digitize(kmag.ravel(), edges)

    def shell(fields):
        E = np.zeros(kmag.shape)
        for f in fields:
            F = np.fft.fftn(f) / f.size
            E += 0.5 * np.abs(F) ** 2
        s = np.bincount(idx, weights=E.ravel(), minlength=len(edges) + 1)
        return s[1:len(edges)]

    # sqrt(rho) weighting preserves the kinetic-energy interpretation
    Ev = shell([np.sqrt(np.maximum(rho, 0)) * run.snapshot(c, tind)
                for c in ("v_x", "v_y", "v_z")])
    Eb = shell([run.snapshot(c, tind) for c in ("Bx", "By", "Bz")])

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.loglog(kcen, Ev + 1e-300, label="kinetic")
    ax.loglog(kcen, Eb + 1e-300, label="magnetic")
    ref = kcen >= 2
    norm = (Ev + Eb)[ref][0] * kcen[ref][0] ** 2
    ax.loglog(kcen[ref], norm * kcen[ref] ** -2.0, "k:", label=r"$k^{-2}$")
    ax.loglog(kcen[ref], norm * kcen[ref] ** (-5 / 3), "k--", label=r"$k^{-5/3}$")
    ax.set_xlabel("k")
    ax.set_ylabel("E(k)")
    ax.set_title(f"Spectra at t = {run.t[tind]:.3f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{outdir}/spectra.png", dpi=140)
    plt.close(fig)

    tot = Ev + Eb
    tail = tot[-1] / tot.max()
    print(f"    E(k_max)/E(k_peak) = {tail:.3e}")
    if tail > 1e-4:
        print("    -> energy piling up at the grid scale: under-resolved,")
        print("       or centred-difference ringing. Raise n or viscos/eta.")


# ============================================================ 4. 3D fraction
def threed(run, outdir, tind):
    print("\n[4] is it genuinely 3D?")
    tot = kz0 = 0.0
    for nm in ("v_x", "v_y", "v_z"):
        f = run.snapshot(nm, tind)
        P = np.abs(np.fft.fft(f, axis=2) / f.shape[2]) ** 2
        tot += P.sum()
        kz0 += P[:, :, 0].sum()
    frac = (tot - kz0) / tot
    print(f"    kinetic energy fraction in kz != 0 : {frac:.4f}")
    print("    (cyclic 3D OT starts near 0.33; ~0 means an extruded 2D IC)")


# ====================================================== 5. total variation
def total_variation(run, outdir):
    print("\n[5] total variation of p (ringing metric)")
    nx, ny, nz = run.shape
    # single x-line: cheap, avoids reading the whole 4D array
    p = run.field("pressure",
                  yind=[ny // 2, ny // 2], zind=[nz // 2, nz // 2])
    line = p.reshape(p.shape[0], -1)
    tv = np.abs(np.diff(line, axis=1)).sum(axis=1)
    t, tv = _trim(run.t, tv)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(t, tv / tv[0])
    ax.set_xlabel("t")
    ax.set_ylabel(r"TV$(p)$ / TV$(p)_{t=0}$")
    ax.set_title("Total variation along an x-line")
    fig.tight_layout()
    fig.savefig(f"{outdir}/tv.png", dpi=140)
    plt.close(fig)
    print(f"    TV growth factor = {tv[-1]/tv[0]:.3f}")
    print("    (bounded for a monotone scheme; growth quantifies the")
    print("     oscillations the FV/limiter port would remove)")


# =============================================================== 6. slices
def slices(run, outdir, tind):
    print("\n[6] slices")
    nx, ny, nz = run.shape
    k = nz // 2
    L = 2.0 * np.pi
    t = run.t[tind]

    panels = []
    for nm, lab, cmap in (
        ("J2", r"$|J|$", "magma"),
        ("density", r"$\rho$", "viridis"),
        ("pressure", r"$p$", "inferno"),
        ("divV", r"$\nabla\cdot v$", "RdBu_r"),
        ("divB", r"$\nabla\cdot B$", "RdBu_r"),
        ("Ms", r"$M_s$", "plasma"),
    ):
        try:
            f = run.snapshot(nm, tind)
            if nm == "J2":
                f = np.sqrt(np.maximum(f, 0))
            panels.append((f[:, :, k], lab, cmap))
        except Exception:
            print(f"    (skipping {nm}: not in output)")

    ncol = 3
    nrow = int(np.ceil(len(panels) / ncol))
    fig, ax = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 4.6 * nrow))
    ax = np.atleast_1d(ax).ravel()
    for a, (f, lab, cmap) in zip(ax, panels):
        kw = {}
        if cmap == "RdBu_r":
            m = np.abs(f).max()
            kw = dict(vmin=-m, vmax=m)
        im = a.imshow(f.T, origin="lower", extent=[0, L, 0, L], cmap=cmap, **kw)
        a.set_title(f"{lab}   (z = $\\pi$, t = {t:.3f})")
        a.set_xlabel("x")
        a.set_ylabel("y")
        fig.colorbar(im, ax=a)
    for a in ax[len(panels):]:
        a.set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{outdir}/slices.png", dpi=140)
    plt.close(fig)

    # orthogonal cuts of |J| to confirm 3D structure
    J = np.sqrt(np.maximum(run.snapshot("J2", tind), 0))
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    for a, (f, lab) in zip(
        ax,
        ((J[:, :, nz // 2], "x-y  (z=$\\pi$)"),
         (J[:, ny // 2, :], "x-z  (y=$\\pi$)"),
         (J[nx // 2, :, :], "y-z  (x=$\\pi$)")),
    ):
        im = a.imshow(f.T, origin="lower", extent=[0, L, 0, L], cmap="magma")
        a.set_title(f"$|J|$  {lab}")
        fig.colorbar(im, ax=a)
    fig.tight_layout()
    fig.savefig(f"{outdir}/J_cuts.png", dpi=140)
    plt.close(fig)


# ============================================================ 7. time montage
def montage(run, outdir, nframe=6):
    print("\n[7] |J| time montage")
    nt = len(run.t)
    inds = np.unique(np.linspace(0, nt - 1, nframe).astype(int))
    nx, ny, nz = run.shape
    L = 2.0 * np.pi
    ncol = 3
    nrow = int(np.ceil(len(inds) / ncol))
    fig, ax = plt.subplots(nrow, ncol, figsize=(4.8 * ncol, 4.3 * nrow))
    ax = np.atleast_1d(ax).ravel()
    for a, i in zip(ax, inds):
        J = np.sqrt(np.maximum(run.snapshot("J2", i), 0))[:, :, nz // 2]
        im = a.imshow(J.T, origin="lower", extent=[0, L, 0, L], cmap="magma")
        a.set_title(f"$|J|$  t = {run.t[i]:.3f}")
        fig.colorbar(im, ax=a)
    for a in ax[len(inds):]:
        a.set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{outdir}/J_montage.png", dpi=140)
    plt.close(fig)


# ================================================================== main
ALL = ["budget", "health", "spectra", "threed", "tv", "slices", "montage"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-p", "--path", default="data", help="BOUT++ output directory")
    ap.add_argument("-o", "--outdir", default="plots", help="where to write PNGs")
    ap.add_argument("--tind", type=int, default=-1, help="time index for 3D plots")
    ap.add_argument("--only", nargs="+", choices=ALL, default=ALL)
    ap.add_argument("--nframe", type=int, default=6, help="frames in the montage")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    run = Run(args.path)

    t = run.t
    tind = args.tind if args.tind >= 0 else len(t) + args.tind
    nx, ny, nz = run.shape

    print("=" * 66)
    print(f"  path      : {args.path}")
    print(f"  grid      : nx={nx}  ny={ny}  nz={nz}   (interior)")
    print(f"  outputs   : {len(t)}   t = {t[0]:.4f} .. {t[-1]:.4f}")
    print(f"  3D plots  : index {tind}  (t = {t[tind]:.4f})")
    print("=" * 66)

    jobs = {
        "budget": lambda: budget(run, args.outdir),
        "health": lambda: health(run, args.outdir),
        "spectra": lambda: spectra(run, args.outdir, tind),
        "threed": lambda: threed(run, args.outdir, tind),
        "tv": lambda: total_variation(run, args.outdir),
        "slices": lambda: slices(run, args.outdir, tind),
        "montage": lambda: montage(run, args.outdir, args.nframe),
    }
    for name in args.only:
        try:
            jobs[name]()
        except Exception as exc:
            print(f"  !! {name} failed: {type(exc).__name__}: {exc}",
                  file=sys.stderr)

    print(f"\nplots written to {args.outdir}/")


if __name__ == "__main__":
    main()
