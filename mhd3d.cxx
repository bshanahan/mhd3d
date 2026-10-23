/**************************************************************************
 * Ideal MHD physics module for BOUT++ -- 3D, physics only.
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
 * All diagnostics are done in post-processing (see boutmhd.py). This file
 * dumps only the evolved state plus, optionally, the metric.
 **************************************************************************/

#include <bout/physicsmodel.hxx>

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
  bool include_viscos; // ad-hoc damping on v
  BoutReal viscos;
  BoutReal eta;        // ad-hoc resistivity on B
  BoutReal divb_diff;  // parabolic (Marder) div(B) cleaning

  bool apply_floors;   // clamp rho, p -- breaks energy conservation
  BoutReal rho_floor, p_floor;

  bool dump_metric;    // write metric to the dump files

  int init(bool UNUSED(restarting)) override {
    auto& options = Options::root()["mhd"];

    g = options["g"].doc("Adiabatic index").withDefault(5.0 / 3.0);

    include_viscos =
        options["include_viscos"].doc("Add damping to v?").withDefault(false);
    viscos = options["viscos"].doc("Viscosity coefficient").withDefault(0.0);

    eta = options["eta"].doc("Resistivity coefficient").withDefault(0.0);

    divb_diff = options["divb_diff"]
                    .doc("Parabolic div(B) cleaning coefficient")
                    .withDefault(0.0);

    apply_floors = options["apply_floors"]
                       .doc("Clamp rho and p? Breaks conservation")
                       .withDefault(false);
    rho_floor = options["rho_floor"].withDefault(1e-6);
    p_floor = options["p_floor"].withDefault(1e-6);

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

    // ------------------------------------------------- geometry summary
    Coordinates* coord = mesh->getCoordinates();
    output.write("\n--- MHD3D ---\n");
    output.write("  local nx={:d} ny={:d} nz={:d}  (nx incl. guards)\n",
                 mesh->LocalNx, mesh->LocalNy, mesh->LocalNz);
    output.write("  dx(0,0,0) = {:e}, dy(0,0,0) = {:e}\n", coord->dx(0, 0, 0),
                 coord->dy(0, 0, 0));
    output.write("  gamma = {:f}, viscos = {:e}, eta = {:e}, divb_diff = {:e}\n", g,
                 viscos, eta, divb_diff);
    output.write("-------------\n\n");

    return 0;
  }

  /// Write the metric ONCE (assign, not assignRepeat) so post-processing
  /// has everything it needs. BOUT++ may already write these; duplicate
  /// names under an 'm' prefix avoid any clash.
  void outputVars(Options& state) override {
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

    if (apply_floors) {
      rho = floor(rho, rho_floor);
      p = floor(p, p_floor);
    }

    {
      TRACE("ddt(rho)");
      ddt(rho) = -V_dot_Grad(v, rho) - rho * Div(v);
    }
    {
      TRACE("ddt(p)");
      ddt(p) = -V_dot_Grad(v, p) - g * p * Div(v);
    }
    {
      TRACE("ddt(v)");
      ddt(v) = -V_dot_Grad(v, v) + (cross(Curl(B), B) - Grad(p)) / rho;

      if (include_viscos) {
        // Component-wise scalar Laplacian on covariant components: only
        // tensor-correct in a Cartesian metric. Ad-hoc damping otherwise.
        ddt(v).x += viscos * Laplace(v.x);
        ddt(v).y += viscos * Laplace(v.y);
        ddt(v).z += viscos * Laplace(v.z);
      }
    }
    {
      TRACE("ddt(B)");
      ddt(B) = Curl(cross(v, B));

      if (eta > 0.0) {
        ddt(B).x += eta * Laplace(B.x);
        ddt(B).y += eta * Laplace(B.y);
        ddt(B).z += eta * Laplace(B.z);
      }

      if (divb_diff > 0.0) {
        Field3D divB = Div(B);
        mesh->communicate(divB);
        ddt(B) += divb_diff * Grad(divB);
      }
    }

    return 0;
  }
};

BOUTMAIN(MHD3D);

// /**************************************************************************
//  * Ideal MHD physics module for BOUT++
//  * This version evolves the entire quantity (initial + perturbed)
//  **************************************************************************/

// #include <bout/physicsmodel.hxx>

// class MHD : public PhysicsModel {
// private:
//   // 3D evolving variables
//   Field3D rho, p; // density, pressure
//   Vector3D v, B;  // velocity, magnetic field

//   // parameters
//   BoutReal g;
//   bool include_viscos;
//   BoutReal viscos;

//   int init(bool UNUSED(restarting)) override {
//     // read options
//     auto& options = Options::root()["mhd"];
//     g = options["g"].withDefault(5.0 / 3.0);
//     include_viscos = options["include_viscos"].withDefault(false);
//     viscos = options["viscos"].withDefault(0.1);

//     // Tell BOUT which variables to evolve.
//     // Initial conditions come entirely from BOUT.inp:
//     //   [density] [pressure] [v_x] [v_y] [v_z] [Bx] [By] [Bz]
//     solver->add(rho, "density");
//     solver->add(p, "pressure");
//     v.covariant = true;    // evolve covariant components  -> v_x, v_y, v_z
//     solver->add(v, "v");
//     B.covariant = false;   // evolve contravariant components -> Bx, By, Bz
//     solver->add(B, "B");

//     Coordinates* coord = mesh->getCoordinates();
//     output.write("dx(0,0,0) = {:e}, dy(0,0,0) = {:e}, dz(0,0,0) = {:e}\n",
//                  coord->dx(0, 0, 0), coord->dy(0, 0, 0), coord->dz(0, 0, 0));

//     return 0;
//   }
//   /// Add variables to the output. This can be used to calculate
//   /// diagnostics
//   ///
//   /// @param[inout] state  A nested dictionary that can be added to
//   void outputVars(Options& state) override {
//     state["divB"].assignRepeat(Div(B)).setAttributes({{"long_name", "Divergence of B"}});
//   }

//   int rhs(BoutReal UNUSED(time)) override {
//     // Communicate variables

//     mesh->communicate(v, B, p, rho);

//     {
//       TRACE("ddt(rho)");
//       ddt(rho) = -V_dot_Grad(v, rho) - rho * Div(v);
//     }
//     {
//       TRACE("ddt(p)");
//       ddt(p) = -V_dot_Grad(v, p) - g * p * Div(v);
//     }
//     {
//       TRACE("ddt(v)");
//       ddt(v) = -V_dot_Grad(v, v) + (cross(Curl(B), B) - Grad(p)) / rho;

//       if (include_viscos) {
//         ddt(v).x += viscos * Laplace(v.x);
//         ddt(v).y += viscos * Laplace(v.y);
//         ddt(v).z += viscos * Laplace(v.z);
//       }
//     }
//     {
//       TRACE("ddt(B)");

//       ddt(B) = Curl(cross(v, B));
//     }

//     return 0;
//   }
// };

// BOUTMAIN(MHD);
