/**************************************************************************
 * Viscous, resistive MHD physics module for BOUT++ -- 3D, physics only.
 *
 * Evolves the entire quantity (initial + perturbed). Initial conditions
 * and geometry are set entirely from BOUT.inp; no grid file required.
 *
 * Sections read for initial conditions:
 *   [density] [pressure]     scalars
 *   [v_x] [v_y] [v_z]        covariant velocity components
 *   [Bx]  [By]  [Bz]         contravariant magnetic field components
 *
 * Units: mu_0 = 1.
 *
 * Fixed metric; smooth/resolved solutions only (not a Godunov shock solver).
 * Physical dissipation includes Newtonian stress and Ohmic/viscous heating.
 * Optional diagnostics expose energy conversion and numerical cleaning work.
 **************************************************************************/

#include <bout/physicsmodel.hxx>
#include <bout/derivs.hxx>
#include <array>
#include <cmath>

/*========================================================================
 *  Metric access. BOUT-dev 'next' exposes these as accessor FUNCTIONS
 *  (coord->J()); the v5.0/v5.1 releases as DATA MEMBERS (coord->J).
 *  Flip to 0 if you get "'...' cannot be used as a function".
 *======================================================================*/
#define COORD_METRIC_IS_FUNCTION 1

#if COORD_METRIC_IS_FUNCTION
#define MET(c, name) ((c)->name())
#else
#define MET(c, name) ((c)->name)
#endif

class MHD3D : public PhysicsModel {
private:
  Field3D rho, p; // density, pressure
  Vector3D v, B;  // velocity (covariant), magnetic field (contravariant)

  BoutReal g;          // adiabatic index
  bool include_viscos; // Newtonian stress, mu = rho * viscos
  BoutReal viscos, bulk_viscos;
  BoutReal eta;        // resistivity: E_res = eta * Curl(B), mu0 = 1
  BoutReal divb_diff;  // parabolic (Marder) div(B) cleaning


  bool dump_metric, diagnose;
  std::string current_boundary, stress_boundary, cleaning_boundary;
  using Metric = Coordinates::FieldMetric;
  using Tensor = std::array<std::array<Field3D, 3>, 3>;
  std::array<std::array<Metric, 3>, 3> gcov, gcon;
  std::array<std::array<std::array<Metric, 3>, 3>, 3> dg;

  static Field3D derivative(const Field3D& f, int i) {
    if (i == 0) return DDX(f, CELL_DEFAULT, "C2");
    if (i == 1) return DDY(f, CELL_DEFAULT, "C2");
    return DDZ(f, CELL_DEFAULT, "C2");
  }

  struct Dissipation {
    Vector3D force, induction; // covariant force per volume; contravariant dB/dt
    Field3D viscous_heat, ohmic_heat;
    Dissipation() {
      force.covariant = true;
      induction.covariant = false;
      force = 0.; induction = 0.;
      viscous_heat = 0.; ohmic_heat = 0.;
    }
  };

  void initialiseViscousMetric() {
    auto* c = mesh->getCoordinates();
    gcov = {{{MET(c,g_11), MET(c,g_12), MET(c,g_13)},
             {MET(c,g_12), MET(c,g_22), MET(c,g_23)},
             {MET(c,g_13), MET(c,g_23), MET(c,g_33)}}};
    gcon = {{{MET(c,g11), MET(c,g12), MET(c,g13)},
             {MET(c,g12), MET(c,g22), MET(c,g23)},
             {MET(c,g13), MET(c,g23), MET(c,g33)}}};
    for (int i=0; i<3; ++i) {
      for (int j=i; j<3; ++j) {
        auto grad = Grad(gcov[i][j], CELL_DEFAULT, "C2");
        dg[0][i][j] = dg[0][j][i] = grad.x;
        dg[1][i][j] = dg[1][j][i] = grad.y;
        dg[2][i][j] = dg[2][j][i] = grad.z;
      }
    }
  }

  Dissipation dissipativeTerms() {
    Dissipation d;
    if (eta > 0.) {
      Vector3D current = Curl(B);
      current.toCovariant();
      // Derivatives create new fields: their MPI halos and physical boundary
      // closures must be supplied before taking a second curl.
      mesh->communicate(current);
      current.applyBoundary(current_boundary);
      d.induction = -Curl(eta * current);
      d.ohmic_heat = eta * (current * current);
    }
    if (include_viscos && (viscos > 0. || bulk_viscos > 0.)) {
      Vector3D u = v;
      u.toContravariant();
      std::array<Field3D,3> uc = {u.x,u.y,u.z};
      Tensor du, strain, stress;
      for (int i=0; i<3; ++i)
        for (int j=0; j<3; ++j) du[i][j] = derivative(uc[j],i);

      // S_ij = (Lie_v g)_ij / 2. This form includes all connection
      // terms without storing Christoffel symbols, for any fixed metric.
      Field3D theta = 0.;
      for (int i=0; i<3; ++i) {
        for (int j=i; j<3; ++j) {
          Field3D sij = 0.;
          for (int k=0; k<3; ++k)
            sij += 0.5*(gcov[j][k]*du[i][k] + gcov[i][k]*du[j][k]
                        + uc[k]*dg[k][i][j]);
          strain[i][j] = strain[j][i] = sij;
          theta += (i==j ? 1. : 2.) * gcon[i][j] * sij;
        }
      }
      // tau^ij = 2 mu S^ij + (zeta - 2 mu/3) theta g^ij.
      // viscos and bulk_viscos are kinematic coefficients, so mu,zeta vary
      // with density. Their gradients are retained inside the stress flux.
      Field3D mu = viscos * rho;
      Field3D zeta = bulk_viscos * rho;
      for (int i=0; i<3; ++i) {
        for (int j=i; j<3; ++j) {
          Field3D raised = 0.;
          for (int k=0; k<3; ++k)
            for (int l=0; l<3; ++l)
              raised += gcon[i][k]*gcon[j][l]*strain[k][l];
          stress[i][j] = stress[j][i] =
              2.*mu*raised + (zeta - (2./3.)*mu)*theta*gcon[i][j];
          d.viscous_heat += (i==j ? 1. : 2.)*stress[i][j]*strain[i][j];
        }
      }
      const auto& jac = MET(mesh->getCoordinates(), J);
      std::array<Field3D,3> force;
      for (int i=0; i<3; ++i) {
        force[i] = 0.;
        // (div tau)_i = (1/J) d_j(J tau^j_i)
        //               - (1/2) tau^jk d_i(g_jk).
        for (int j=0; j<3; ++j) {
          Field3D mixed = 0.;
          for (int k=0; k<3; ++k) mixed += stress[j][k]*gcov[k][i];
          mesh->communicate(mixed);
          mixed.applyBoundary(stress_boundary);
          force[i] += derivative(jac*mixed,j)/jac;
          for (int k=0; k<3; ++k)
            force[i] -= 0.5*stress[j][k]*dg[i][j][k];
        }
      }
      d.force.x = force[0]; d.force.y = force[1]; d.force.z = force[2];
    }
    return d;
  }

  int init(bool UNUSED(restarting)) override {
    auto& options = Options::root()["mhd"];

    g = options["g"].doc("Adiabatic index").withDefault(5.0 / 3.0);

    include_viscos =
        options["include_viscos"].doc("Enable Newtonian viscosity and heating?").withDefault(false);
    viscos = options["viscos"].doc("Kinematic shear viscosity, mu=rho*viscos").withDefault(0.0);
    bulk_viscos = options["bulk_viscos"].doc("Kinematic bulk viscosity, zeta=rho*bulk_viscos").withDefault(0.0);

    eta = options["eta"].doc("Resistivity coefficient").withDefault(0.0);

    divb_diff = options["divb_diff"]
                    .doc("Parabolic div(B) cleaning coefficient")
                    .withDefault(0.0);

    // Never overwrite the integrator's state inside an RHS evaluation.
    if (options["apply_floors"].withDefault(false)) {
      throw BoutException("mhd:apply_floors is no longer supported: RHS clipping changes the solver state. Use a positivity-preserving discretization.");
    }
    if (!std::isfinite(g) || g <= 1. || !std::isfinite(viscos) || viscos < 0.
        || !std::isfinite(bulk_viscos) || bulk_viscos < 0.
        || !std::isfinite(eta) || eta < 0. || !std::isfinite(divb_diff) || divb_diff < 0.) {
      throw BoutException("Require finite gamma>1 and nonnegative finite dissipation coefficients");
    }
    if ((eta > 0. || (include_viscos && (viscos > 0. || bulk_viscos > 0.)))
        && Options::root()["mesh"]["paralleltransform"]["type"].withDefault(std::string("identity")) != "identity") {
      throw BoutException("Physical dissipation currently requires mesh:paralleltransform:type=identity");
    }
    diagnose = options["diagnose"].doc("Dump dissipation and energy diagnostics").withDefault(false);
    current_boundary = options["current_boundary"].doc("Physical boundary closure for covariant current before resistive curl").withDefault(std::string("neumann"));
    stress_boundary = options["stress_boundary"].doc("Physical boundary closure for mixed viscous stress before divergence").withDefault(std::string("neumann"));
    cleaning_boundary = options["cleaning_boundary"].doc("Physical boundary closure for Div(B) before cleaning gradient").withDefault(std::string("neumann"));

    dump_metric = options["dump_metric"]
                      .doc("Write J and g_ij to dump files for post-processing")
                      .withDefault(true);

    // ---------------------------------------------------------- evolve
    solver->add(rho, "density");
    solver->add(p, "pressure");
    v.covariant = true;  // -> [v_x] [v_y] [v_z]
    solver->add(v, "v");
    B.covariant = false; // -> [Bx]  [By]  [Bz]
    solver->add(B, "B");

    if (include_viscos && (viscos > 0. || bulk_viscos > 0.)) initialiseViscousMetric();

    // ------------------------------------------------- geometry summary
    Coordinates* coord = mesh->getCoordinates();
    output.write("\n--- MHD3D ---\n");
    output.write("  local nx={:d} ny={:d} nz={:d}  (nx incl. guards)\n",
                 mesh->LocalNx, mesh->LocalNy, mesh->LocalNz);
    output.write("  dx(0,0,0) = {:e}, dy(0,0,0) = {:e}\n", coord->dx(0, 0, 0),
                 coord->dy(0, 0, 0));
    output.write("  gamma = {:f}, viscos = {:e}, eta = {:e}, divb_diff = {:e}\n", g,
                 viscos, eta, divb_diff);
    output.write("  bulk_viscos = {:e}; physical dissipation includes heating\n", bulk_viscos);
    output.write("-------------\n\n");

    return 0;
  }

  /// Write the metric ONCE (assign, not assignRepeat) so post-processing
  /// has everything it needs. BOUT++ may already write these; duplicate
  /// names under an 'm' prefix avoid any clash.
  void outputVars(Options& state) override {
    if (diagnose) {
      // Recompute at the state being output, not at a cached CVODE trial stage.
      mesh->communicate(v, B, p, rho);
      auto d = dissipativeTerms();
      state["Q_visc"].assignRepeat(d.viscous_heat);
      state["Q_ohm"].assignRepeat(d.ohmic_heat);
      state["viscous_force_x"].assignRepeat(d.force.x);
      state["viscous_force_y"].assignRepeat(d.force.y);
      state["viscous_force_z"].assignRepeat(d.force.z);
      state["viscous_work"].assignRepeat(v*d.force);
      state["resistive_magnetic_work"].assignRepeat(B*d.induction);
      Field3D energy = p/(g-1.) + 0.5*rho*(v*v) + 0.5*(B*B);
      state["energy_density"].assignRepeat(energy);
      state["divB"].assignRepeat(Div(B));
      Field3D cleaning_work = 0.;
      if (divb_diff > 0.) {
        Field3D divB = Div(B);
        mesh->communicate(divB);
        divB.applyBoundary(cleaning_boundary);
        cleaning_work = B*(divb_diff*Grad(divB));
      }
      state["cleaning_magnetic_work"].assignRepeat(cleaning_work);
    }
    if (!dump_metric) {
      return;
    }
    Coordinates* coord = mesh->getCoordinates();

    // Promote to Field3D so this works whether FieldMetric is 2D or 3D.
    auto as3d = [](auto&& q) {
      Field3D f = 0.0;
      f += q;
      return f;
    };

    state["mJ"].assign(as3d(MET(coord, J)));
    state["mdx"].assign(as3d(MET(coord, dx)));
    state["mdy"].assign(as3d(MET(coord, dy)));
    state["mdz"].assign(as3d(MET(coord, dz)));

    state["mg11"].assign(as3d(MET(coord, g11)));
    state["mg22"].assign(as3d(MET(coord, g22)));
    state["mg33"].assign(as3d(MET(coord, g33)));
    state["mg12"].assign(as3d(MET(coord, g12)));
    state["mg13"].assign(as3d(MET(coord, g13)));
    state["mg23"].assign(as3d(MET(coord, g23)));

    state["mg_11"].assign(as3d(MET(coord, g_11)));
    state["mg_22"].assign(as3d(MET(coord, g_22)));
    state["mg_33"].assign(as3d(MET(coord, g_33)));
    state["mg_12"].assign(as3d(MET(coord, g_12)));
    state["mg_13"].assign(as3d(MET(coord, g_13)));
    state["mg_23"].assign(as3d(MET(coord, g_23)));
  }

  int rhs(BoutReal UNUSED(time)) override {
    mesh->communicate(v, B, p, rho);

    auto dissip = dissipativeTerms();

    {
      TRACE("ddt(rho)");
      ddt(rho) = -V_dot_Grad(v, rho) - rho * Div(v);
    }
    {
      TRACE("ddt(p)");
      ddt(p) = -V_dot_Grad(v, p) - g * p * Div(v)
               + (g - 1.)*(dissip.viscous_heat + dissip.ohmic_heat);
    }
    {
      TRACE("ddt(v)");
      ddt(v) = -V_dot_Grad(v, v) + (cross(Curl(B), B) - Grad(p)) / rho;

      ddt(v) += dissip.force / rho;
    }
    {
      TRACE("ddt(B)");
      ddt(B) = Curl(cross(v, B));

      ddt(B) += dissip.induction;

      if (divb_diff > 0.0) {
        Field3D divB = Div(B);
        mesh->communicate(divB);
        divB.applyBoundary(cleaning_boundary);
        ddt(B) += divb_diff * Grad(divB);
      }
    }

    return 0;
  }
};

BOUTMAIN(MHD3D);
