# Optional conservative MHD

The default remains `mhd:shock_capturing=false`: the primitive equations and
physical dissipation described in [dissipation.md](dissipation.md). Enable the
finite-volume ideal terms with ordinary BOUT++ input sections:

```ini
[mhd]
shock_capturing = true
apply_floors = false
divb_diff = 0

[mhd:shock]
limiter = mc
cfl = 0.3
glm_ch = 2
glm_damping = 1
rho_min = 1e-12
p_min = 1e-12

[solver]
type = rk3ssp
```

`glm_ch` is a fixed **physical speed in simulation units**, not a fraction of
another speed. Choose it comparable to the largest relevant wave speed, check
cleaning effectiveness, and include it in convergence studies. `glm_damping`
is an inverse code time. The displayed values are examples, not a calibration
for the driven wind. In particular, a wind with U=4 has characteristic speeds
larger than the example cleaning speed. Larger cleaning speeds increase the
Rusanov diffusion and reduce the timestep.

When changing from CVODE, remove its solver-specific tolerances (`atol`,
`rtol`) from the input; they are not controls for this explicit integrator.
The `[mhd:shock]` settings can remain in the file when the shock switch is off.

Use `solver:max_timestep` to impose a tighter internal step cap if needed.
The root `timestep` is still output spacing. The model reduces the initial
internal step to at most 0.8 times its multidimensional CFL estimate. This cap
stays fixed: the installed RK3SSP implementation cannot roll back a stage and
its `setMaxTimestep` affects subsequent steps. If later states violate the cap,
the model stops with a diagnostic. CFL estimates include cleaning, geometry,
and physical diffusion; they are not a positivity proof for arbitrary shocks
or highly distorted cells. Only `rk3ssp` is currently accepted by this path.

## Equations and discretization

The solver evolves density, **covariant momentum** `momentum_x/y/z`, total
energy `total_energy`, **contravariant magnetic field** `Bx/By/Bz`, and the
auxiliary scalar `glm_psi`. Total energy is

    E = p/(gamma-1) + rho |u|^2/2 + |B|^2/2.

The finite-volume part uses one coupled local Lax-Friedrichs (Rusanov) flux
for mass, all momentum components, energy, induction, and GLM cleaning.
Reconstruction is componentwise in physical primitive variables, with BOUT++'s
`FV::MC` or `FV::MinMod` limiters. `limiter=first_order` disables reconstruction.
An inadmissible reconstructed face falls back to its entire cell state;
negative evolved cell density/pressure causes a collective error, never clipping.
`rho_min` and `p_min` are rejection thresholds, not imposed floors.

In a local orthonormal frame, the physical flux along n is

    F_rho = rho u_n
    F_mi  = rho u_n u_i + (p + B^2/2) delta_ni - B_n B_i
    F_E   = (E+p+B^2/2) u_n - (u.B) B_n
    F_Bi  = u_n B_i - u_i B_n + psi delta_ni
    F_psi = ch^2 B_n.

The Rusanov speed is the maximum of `|u_n|+c_fast` on both sides and `ch`.
This is a deliberately diffusive baseline MHD scheme, not HLLD, constrained
transport, or a drop-in Hermes-3 MHD solver. Hermes-3 motivates evolving
conserved momentum and energy through shocks; the coupled 3D magnetic flux is
implemented here in `mhd_fv.hxx`. See the
[Hermes-3 shock discussion](https://hermes3.readthedocs.io/en/latest/developer.html)
and the [conservative GLM-MHD formulation](https://arxiv.org/abs/0911.3410).

The GLM system has induction flux `F_Bi` above and
`dpsi/dt + ch^2 div(B) = -glm_damping psi`. E excludes psi and has the usual MHD flux. Consequently cleaning
changes magnetic energy and the pressure recovered from E; it is **not physical
Ohmic heating**. Monitor `divB`, `glm_psi`, pressure, and cleaning sensitivity.
The existing `cleaning_magnetic_work` diagnostic describes only the default
path's parabolic cleaner, and remains zero in GLM mode.

## Geometry

Supported: Cartesian identity metric and the existing static diagonal model

    ds^2 = a(y)^2 dx^2 + dy^2 + a(y)^2 dz^2,  J=a^2,

with constant logical dx,dy,dz, identity parallel transform, at least two x/y
guards, and periodic z. Unsupported metrics fail during initialization.
No evolving metric or expanding-box source terms are introduced.

For scale factors h=(a,1,a), an orthonormal face flux is multiplied by J/h_n;
momentum flux components additionally by h_i, induction components by 1/h_i.
The face scale factor is the average of its neighboring cell values. Define

    H_j = (a_(j+1/2)^2 - a_(j-1/2)^2)/(2 a_j^2 dy).

The radial covariant momentum source is
`H [rho (u_x^2+u_z^2) + 2p + B_y^2]`. The radial induction source `2H psi`
corrects the densitized GLM flux to a contravariant gradient. This geometry
preserves uniform static gas pressure exactly in the discrete equations.
It is not a well-balanced discretization of every accelerating wind background.
The coordinate metric remains the local approximation of the original model;
shock capturing does not make it an exact global spherical wind geometry.

## Physical dissipation

The existing `include_viscos`, `viscos`, `bulk_viscos`, and `eta` switches also
work in this mode. Momentum receives the covariant stress force per volume,
induction receives `-Curl(eta Curl(B))`, and energy receives

    u.force + B.induction + Q_visc + Q_ohm.

This is the thermodynamic chain rule for the existing resolved physical
operators. Their centered discretization and auxiliary boundary limitations
still apply; they are not replaced by a face-centered viscous Riemann solver.
Numerical Rusanov dissipation is distinct from the reported physical heating.
`divb_diff` must be zero because GLM supplies divergence cleaning in this mode.

## Inputs, boundaries, and restarts

Keep the same `[density]`, `[pressure]`, `[v_x/y/z]`, and `[Bx/By/Bz]`
initial functions and boundary expressions. As before, input velocity is
covariant and magnetic field contravariant. Internally momentum and E are
initialized from these primitive fields. Do not supply initial functions or
boundary conditions for `momentum_*` or `total_energy`.

Before each RHS, conserved fields are communicated, primitives are recovered,
and primitive boundary expressions are applied at the current stage time.
Flux reconstruction uses those primitive ghosts. `glm_psi` defaults to zero
with Neumann physical boundaries; it has its own optional input section.
Interior output fields include both conserved state and recovered primitives.
`shock_cfl_dt` reports the current estimated CFL bound, not the actual step.
`shock_reconstruction_fallbacks` counts flattened face reconstructions in the
last RHS; shared faces are evaluated twice. Conservative physical guard cells
are not the boundary interface; use the
primitive fields for boundary diagnostics. `free_o2` remains an extrapolation
boundary, not a characteristic MHD outflow.

Restart files carry a formulation flag. Restarts across the two formulations
are rejected; old unmarked restarts are treated as primitive mode. Restart
using the same geometry, equation of state, and formulation. Automatic
conversion of old primitive restarts to conserved states is not implemented.

## Verification

Build with BOUT++ and run the compiled-model tests in a scratch directory:

```sh
python3 tests/test_shock.py --exe build/mhd3d --workdir /tmp/mhd-checks
python3 tests/test_dissipation.py --exe build/mhd3d --workdir /tmp/mhd-checks
```

These require NumPy, netCDF4, and an MPI launcher. The shock suite includes
an exact Sod Riemann solution comparison, Brio-Wu admissibility/conservation,
smooth circular Alfven-wave convergence, GLM propagation/damping, geometric
pressure balance, a time-dependent inlet, periodic 3D conservation, MPI
comparison, and restart checks. These are implementation checks, not evidence
that a particular switchback run is converged or publication-ready. Strong
multidimensional shocks, long driven runs, turbulence spectra, reversal
statistics, outflow sensitivity, and divergence errors need problem-specific
validation. Rusanov diffusion can suppress small-scale switchbacks.

### Local verification record

Verified with the installed BOUT++ development build (source revision
`61ca9f191`, 2D metric build) on 2026-11-13:

| Check | Result |
| --- | --- |
| Sod density mean absolute error, 64 / 128 cells | 0.01125 / 0.005706 |
| Circular Alfven Bx mean absolute error, 32 / 64 cells | 1.219e-4 / 3.213e-5 |
| Brio-Wu minimum pressure through t=0.1 | 0.08616 |
| Brio-Wu relative total-energy drift | 4.1e-15 or less |
| Warped-metric static-pressure evolution, all three limiters | No change detected |
| GLM magnetic amplitude at t=0.3, measured / linear analytic | 0.841156 / 0.841468 |
| Periodic 3D relative mass / energy drift | Both below 2e-15 |
| Curved 3D case with physical dissipation, relative energy drift | Below 2e-15 |
| Serial versus two ranks, x and y decomposition | No field difference detected |
| Split restart versus uninterrupted continuation | No field difference detected |

The suite also verifies the initial automatic CFL cap, rejection of invalid
states/geometry/solver selection, rejection of cross-formulation restarts,
and switching back to the default formulation while leaving shock options in
the input. The separate default-path dissipation suite passed after integration.
The Brio-Wu check establishes positivity and conservation for this test; it is
not a comparison of every MHD wave family against a reference profile.
