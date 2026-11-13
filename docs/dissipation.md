# Physical dissipation in MHD3D

The model now includes covariant Newtonian viscosity and resistive induction with matching thermal heating. It still evolves density, pressure, covariant velocity and contravariant magnetic field. The ideal terms, static metric, initial profiles and primitive-variable output names are unchanged. This is a resolved viscous/resistive MHD extension, not a conservative Godunov shock solver.

## Equations and conventions

In units mu0=1, with current j = curl B and a fixed coordinate metric:

- Induction contribution: dB/dt = -curl(eta j).
- Momentum contribution: dv_i/dt = (div tau)_i / rho.
- Pressure contribution: dp/dt = (gamma-1) [eta j_i j^i + tau^ij S_ij].

Here mu=rho*viscos is the dynamic shear viscosity and zeta=rho*bulk_viscos is the dynamic bulk viscosity. Both input coefficients are constant kinematic viscosities (length squared per time). Eta is a constant magnetic diffusivity in these normalized units. All coefficients must be finite and nonnegative; gamma must exceed one.

The covariant rate of strain is computed without explicitly storing Christoffel symbols:

    S_ij = 1/2 [g_jk partial_i(v^k) + g_ik partial_j(v^k)
                + v^k partial_k(g_ij)]
    theta = g^ij S_ij
    tau^ij = 2 mu S^ij + (zeta - 2 mu/3) theta g^ij
    (div tau)_i = (1/J) partial_j(J tau^j_i)
                  - 1/2 tau^jk partial_i(g_jk)

Repeated indices are summed; S is raised with both inverse metrics. Metric derivatives are cached at initialization. This supports fixed coordinate metrics, including nonorthogonal ones, with the identity parallel transform. Shifted/FCI transforms are explicitly rejected when physical dissipation is enabled; extending the tensor discretization to their parallel slices/torsion needs separate work.

Viscous heating is tau:S = 2 mu S_dev:S_dev + zeta theta^2, nonnegative for positive rho and a positive-definite metric. It includes bulk compression, shear and geometry. Using theta from the same strain tensor avoids mixing incompatible discrete traces. The momentum flux includes gradients of mu and zeta; multiplying a scalar velocity Laplacian by a coefficient would omit them.

The pressure heating pairs with magnetic or kinetic energy loss. The full primitive ideal-MHD discretization is NOT thereby made exactly energy conservative. Shocks that are not resolved by explicit physical dissipation still need a conservative shock-capturing treatment.

## Options

Existing `mhd:include_viscos`, `mhd:viscos` and `mhd:eta` names are retained. Their interpretation is now physical: viscosity is a stress divergence, and resistivity is a curl of current. Even in Cartesian geometry, viscosity is not simply nu Laplace(v) for compressible or varying-density flow. Resistivity is eta [Laplacian(B)-grad div B] in Cartesian coordinates with constant eta, not a component Laplacian for a non-solenoidal field.

Example syntax only; these coefficients have NOT been selected or validated for switchback production:

```ini
[mhd]
include_viscos = true
viscos = 0.01
bulk_viscos = 0.0
eta = 0.01
divb_diff = 0.0
apply_floors = false
diagnose = true
current_boundary = neumann
stress_boundary = neumann
cleaning_boundary = neumann
```

`include_viscos=false` disables both shear and bulk viscosity. Eta is independent of that flag. Coefficients default to zero, so existing ideal configurations remain ideal. The research baseline BOUT.inp still has them disabled: changing the source alone does not select a dissipative regime or guarantee stability.

`apply_floors=true` is now rejected with a clear error. The previous implementation overwrote the integrator's evolving state inside an RHS evaluation; this is not a valid positivity method. This change does not introduce a positivity-preserving solver. No evolved fields are clipped by the new implementation. Legacy rho_floor/p_floor options are not used.

The auxiliary boundaries operate on newly differentiated fields before the second derivative. `current_boundary` applies to covariant current components. `stress_boundary` applies to each mixed stress component tau^j_i before its flux divergence. `cleaning_boundary` applies to Div(B). Their default Neumann condition extrapolates these quantities; it is not a characteristic outflow, a stress-free boundary, or a zero-energy-flux condition. The strings are static BOUT applyBoundary(condition) specifications; time-dependent auxiliary boundary expressions are not supported by this interface. MPI/periodic halos are communicated separately. Physical boundary sensitivity remains part of the model validation.

## Discrete limitations

The stress uses C2 coordinate derivatives. Resistive curls use BOUT's configured first-derivative methods. The supplied and tested cases use C2 in all directions. On periodic uniform logical grids the central derivative adjoints give matching integrated viscous/Ohmic work and heating, including the tested varying metric. This property is not asserted for arbitrary nonuniform meshes or physical boundary closures.

Composing cell-centered C2 first derivatives leaves a Nyquist checkerboard null mode. These operators resolve physical diffusion of smooth modes but are not a grid-scale shock stabilization method. Face-based compatible diffusion or a conservative MHD numerical flux is needed if unresolved grid-scale modes/shocks are central to the study. High viscosity alone does not remove this limitation.

Parabolic `divb_diff` remains optional numerical cleaning. Its magnetic work is reported separately and is not silently converted to physical heat. It does not make the induction scheme constrained transport or establish that divergence is acceptably small.

## Diagnostics

With `mhd:diagnose=true`, output includes:

- `Q_visc`, `Q_ohm`: heating per physical volume per unit code time, before multiplying by gamma-1.
- `viscous_force_x/y/z`: covariant components of force per volume (not acceleration).
- `viscous_work`: v dot div(tau).
- `resistive_magnetic_work`: B dot [-curl(eta j)].
- `cleaning_magnetic_work`: work of the optional numerical cleaning term.
- `energy_density`: p/(gamma-1) + rho |v|^2/2 + |B|^2/2.
- `divB`: BOUT's discrete divergence.

These are recomputed at the output state, rather than copied from the last CVODE trial RHS. Metric dumping remains independently controlled. Volume integrals require J*dx*dy*dz weights. In periodic domains the dissipation-only integrals of `viscous_work+Q_visc` and `resistive_magnetic_work+Q_ohm` should approach zero; for open domains include boundary energy fluxes. These diagnostics add full 3D fields at the ordinary output cadence; they do not implement the proposed frequent scalar/probe monitor.

## Verification

Build with the existing CMake configuration, then:

```sh
cmake --build build -j 2
python3 tests/test_dissipation.py --exe build/mhd3d --workdir /path/to/scratch
```

The test script requires numpy, netCDF4 and mpirun and creates fresh temporary directories. Tests cover:

1. Periodic force-free magnetic decay and Ohmic heating against an analytic solution.
2. Periodic circular shear decay and viscous heating at density 2, testing the kinematic coefficient convention.
3. Second-order continuum convergence of these decays at 32 and 64 cells.
4. A Killing flow in a varying diagonal metric: strain, stress and heating vanish despite varying covariant velocity.
5. Curved-metric shear force/heating against analytic expressions, second-order force convergence, and integrated work balance.
6. Bulk heating with spatially varying density/viscosity.
7. Curved-metric force-free current and resistive work balance.
8. Constant nonorthogonal metric, testing both tensor index raising and energy conversion.
9. Two-rank Y decomposition, including a short open-boundary tube exercise.

Verified against the installed BOUT++ development build with 2D metrics and CVODE. Maximum total-energy error in the analytic decay cases was below 8e-9 in code energy-density units; refinement reduced spatial errors by approximately four. Curved-metric dissipation work balances were below 1e-17 in the tested averages. Open-boundary one/two-rank field differences were below 1e-20 in this short test. This does not validate long-duration switchback evolution, shock jump conditions, 3D metric builds, arbitrary transforms, or arbitrary boundary conditions.

## Relationship to Hermes-3 shock capturing

BOUT++ provides finite-volume reconstruction/limiter and dissipative flux operators (`FV::Div_par_mod`, `FV::Div_par_fvv`, `FV::Div_par_fvv_heating` and related routines). Hermes-3 uses these for fluid transport and includes Euler shock tests. Its documentation shows that evolving total fluid energy improves the Sod shock location compared with evolving pressure:
https://hermes3.readthedocs.io/en/latest/developer.html#sod-shock

These are useful building blocks, not a drop-in full 3D MHD Riemann solver found in the inspected installation. A conservative MHD extension must evolve compatible mass, momentum, total energy and induction fluxes, use magnetosonic signal speeds, and maintain the magnetic constraint. Replacing V_dot_Grad with a scalar FV operator while retaining the present magnetic forces and primitive pressure equation would not accomplish that. The current change addresses the physical dissipation terms only; it does not implement that larger flux-system change or alter the geometric approximation.
