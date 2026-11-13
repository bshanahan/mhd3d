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
 * Fixed metric; primitive equations by default, optional finite-volume GLM-MHD.
 * Physical dissipation includes Newtonian stress and Ohmic/viscous heating.
 * Optional diagnostics expose energy conversion and numerical cleaning work.
 **************************************************************************/

#include <bout/physicsmodel.hxx>
#include <bout/derivs.hxx>
#include <bout/boutcomm.hxx>
#include <bout/initialprofiles.hxx>
#include <bout/fv_ops_impl.hxx>
#include "mhd_fv.hxx"
#include <algorithm>
#include <limits>
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

  // Optional conservative finite-volume state. Primitive names remain the
  // public initialization/boundary/output interface in either formulation.
  bool shock_capturing{false}, shock_first_rhs{true};
  Vector3D momentum;
  Field3D total_energy, glm_psi, shock_a;
  std::string shock_limiter;
  BoutReal glm_ch, glm_damping, shock_cfl, shock_rho_min, shock_p_min;
  BoutReal shock_dt{0.}, shock_time{0.}, shock_step_limit{0.};
  std::array<BoutReal,3> shock_dx{};
  int reconstruction_fallbacks{0};

  static void collectiveCheck(bool bad, const std::string& message) {
    int local = bad ? 1 : 0, global = 0;
    MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MAX, BoutComm::get());
    if (global) throw BoutException(message);
  }

  void initialiseShockGeometry() {
    auto* c = mesh->getCoordinates();
    if (mesh->xstart < 2 || mesh->ystart < 2) {
      throw BoutException("Shock reconstruction requires MXG>=2 and MYG>=2");
    }
    if (Options::root()["mesh"]["paralleltransform"]["type"]
            .withDefault(std::string("identity")) != "identity") {
      throw BoutException("Shock capturing requires the identity parallel transform");
    }
    shock_a = sqrt(MET(c,g_11));
    mesh->communicate(shock_a);
    const int x0=mesh->xstart, y0=mesh->ystart, z0=mesh->zstart;
    shock_dx = {MET(c,dx)(x0,y0,z0), MET(c,dy)(x0,y0,z0), MET(c,dz)(x0,y0,z0)};
    bool bad = false;
    auto close = [](BoutReal x, BoutReal y) {
      return std::isfinite(x) && std::isfinite(y)
             && std::abs(x-y) <= 1e-10*std::max({1.,std::abs(x),std::abs(y)});
    };
    for (int x=0; x<mesh->LocalNx; ++x) {
      for (int y=0; y<mesh->LocalNy; ++y) {
        if ((x<mesh->xstart || x>mesh->xend)
            && (y<mesh->ystart || y>mesh->yend)) continue; // unused corners
        for (int z=0; z<mesh->LocalNz; ++z) {
          const BoutReal a=shock_a(x,y,z), a2=a*a;
          bad = bad || !std::isfinite(a) || a<=0. || !close(a,shock_a(x0,y,z0))
              || !close(MET(c,g_11)(x,y,z),a2) || !close(MET(c,g_22)(x,y,z),1.)
              || !close(MET(c,g_33)(x,y,z),a2) || !close(MET(c,J)(x,y,z),a2)
              || !close(MET(c,g11)(x,y,z),1./a2) || !close(MET(c,g22)(x,y,z),1.)
              || !close(MET(c,g33)(x,y,z),1./a2)
              || !close(MET(c,g12)(x,y,z),0.) || !close(MET(c,g13)(x,y,z),0.)
              || !close(MET(c,g23)(x,y,z),0.) || !close(MET(c,g_12)(x,y,z),0.)
              || !close(MET(c,g_13)(x,y,z),0.) || !close(MET(c,g_23)(x,y,z),0.)
              || !close(MET(c,ShiftTorsion)(x,y,z),0.)
              || !close(MET(c,dx)(x,y,z),shock_dx[0])
              || !close(MET(c,dy)(x,y,z),shock_dx[1])
              || !close(MET(c,dz)(x,y,z),shock_dx[2]);
        }
      }
    }
    std::array<BoutReal,3> smallest{},largest{};
    MPI_Allreduce(shock_dx.data(),smallest.data(),3,MPI_DOUBLE,MPI_MIN,BoutComm::get());
    MPI_Allreduce(shock_dx.data(),largest.data(),3,MPI_DOUBLE,MPI_MAX,BoutComm::get());
    for (int i=0; i<3; ++i) bad = bad || smallest[i]<=0. || !close(smallest[i],largest[i]);
    collectiveCheck(bad,"Shock capturing supports uniform logical spacings and only ds^2=a(y)^2(dx^2+dz^2)+dy^2, J=a^2 (Cartesian a=1 included)");
  }

  void initialiseShockState() {
    auto& root = Options::root();
    auto& opt = root["mhd"]["shock"];
    shock_limiter = opt["limiter"].withDefault(std::string("mc"));
    glm_ch = opt["glm_ch"].doc("Fixed physical GLM cleaning speed").withDefault(1.0);
    glm_damping = opt["glm_damping"].doc("GLM psi damping rate, inverse code time").withDefault(1.0);
    shock_cfl = opt["cfl"].withDefault(0.3);
    shock_rho_min = opt["rho_min"].doc("Admissibility threshold, not a state floor").withDefault(1e-12);
    shock_p_min = opt["p_min"].doc("Admissibility threshold, not a state floor").withDefault(1e-12);
    if ((shock_limiter!="mc" && shock_limiter!="minmod" && shock_limiter!="first_order")
        || !std::isfinite(glm_ch) || glm_ch<=0. || !std::isfinite(glm_damping) || glm_damping<0.
        || !std::isfinite(shock_cfl) || shock_cfl<=0. || shock_cfl>0.5
        || !std::isfinite(shock_rho_min) || shock_rho_min<=0.
        || !std::isfinite(shock_p_min) || shock_p_min<=0.) {
      throw BoutException("Invalid mhd:shock options: limiter=mc|minmod|first_order, glm_ch>0, damping>=0, 0<cfl<=0.5, positive admissibility thresholds");
    }
    // The installed rk3ssp implements model-supplied timestep limits. Other
    // integrators require separate CFL/admissibility handling and validation.
    if (root["solver"]["type"].withDefault(std::string("cvode")) != "rk3ssp") {
      throw BoutException("mhd:shock_capturing=true currently requires [solver] type=rk3ssp");
    }
    if (divb_diff != 0.) {
      throw BoutException("Use mhd:shock GLM cleaning, not mhd:divb_diff, with shock_capturing=true");
    }
    initialiseShockGeometry();
    v.covariant = true;
    initial_profile("pressure",p);
    initial_profile("v",v);
    p.setBoundary("pressure");
    v.x.setBoundary("v_x"); v.y.setBoundary("v_y"); v.z.setBoundary("v_z");
    p.applyBoundary(0.);
    v.x.applyBoundary(0.); v.y.applyBoundary(0.); v.z.applyBoundary(0.);

    for (const auto& name : {"momentum_x","momentum_y","momentum_z","total_energy"}) {
      auto& section=root[name];
      if (section.isSet("function")) {
        throw BoutException("Initialize pressure and v, not internal conservative variable "+std::string(name));
      }
      for (const auto& kv : section.getChildren()) {
        if (kv.first.find("bndry_")==0 || kv.first=="evolve_bndry") {
          throw BoutException("Specify primitive-field boundaries, not boundaries for "+std::string(name));
        }
      }
      section["function"].withDefault(std::string("0"));
      section["bndry_all"].withDefault(std::string("none"));
      section["evolve_bndry"].withDefault(false);
    }
    root["glm_psi"]["function"].withDefault(std::string("0"));
    root["glm_psi"]["bndry_all"].withDefault(std::string("neumann"));
    momentum.covariant = true;
    solver->add(momentum,"momentum");
    solver->add(total_energy,"total_energy");
    solver->add(glm_psi,"glm_psi");
    momentum = rho*v;
    total_energy = p/(g-1.) + 0.5*rho*(v*v) + 0.5*(B*B);
    output.write("  Conservative GLM-MHD enabled; limiter={}, ch={}, damping={}\n",
                 shock_limiter,glm_ch,glm_damping);
  }

  void recoverShockPrimitives(BoutReal time) {
    mesh->communicate(rho,momentum,total_energy,B,glm_psi);
    v.covariant = true;
    bool bad = false;
    for (int x=0; x<mesh->LocalNx; ++x) {
      for (int y=0; y<mesh->LocalNy; ++y) {
        for (int z=0; z<mesh->LocalNz; ++z) {
          const BoutReal r=rho(x,y,z), a=shock_a(x,y,z), a2=a*a;
          const bool interior=x>=mesh->xstart && x<=mesh->xend && y>=mesh->ystart && y<=mesh->yend;
          // Unevolved physical conservative guards may not be admissible.
          // They are never used as states: primitive BCs fill those guards below.
          if (!std::isfinite(r) || r<=shock_rho_min) {
            bad = bad || interior;
            v.x(x,y,z)=v.y(x,y,z)=v.z(x,y,z)=0.; p(x,y,z)=1.;
            continue;
          }
          v.x(x,y,z)=momentum.x(x,y,z)/r;
          v.y(x,y,z)=momentum.y(x,y,z)/r;
          v.z(x,y,z)=momentum.z(x,y,z)/r;
          const BoutReal u2=(v.x(x,y,z)*v.x(x,y,z)+v.z(x,y,z)*v.z(x,y,z))/a2+v.y(x,y,z)*v.y(x,y,z);
          const BoutReal b2=a2*(B.x(x,y,z)*B.x(x,y,z)+B.z(x,y,z)*B.z(x,y,z))+B.y(x,y,z)*B.y(x,y,z);
          p(x,y,z)=(g-1.)*(total_energy(x,y,z)-0.5*r*u2-0.5*b2);
          bad = bad || (interior && (!std::isfinite(p(x,y,z)) || p(x,y,z)<=shock_p_min));
        }
      }
    }
    collectiveCheck(bad,"Nonpositive/nonfinite conservative MHD cell state; reduce timestep or use more dissipative reconstruction. No state clipping was applied.");
    rho.applyBoundary(time); p.applyBoundary(time);
    v.x.applyBoundary(time); v.y.applyBoundary(time); v.z.applyBoundary(time);
    B.x.applyBoundary(time); B.y.applyBoundary(time); B.z.applyBoundary(time);
    glm_psi.applyBoundary(time);
    bad=false;
    // Check stencil cells, including physical ghosts, before any flux is used.
    for (int x=0; x<mesh->LocalNx; ++x)
      for (int y=0; y<mesh->LocalNy; ++y)
        for (int z=0; z<mesh->LocalNz; ++z)
          if ((x>=mesh->xstart && x<=mesh->xend)
              || (y>=mesh->ystart && y<=mesh->yend)) {
            bad = bad || !mhd_fv::admissible(shockPrimitive(x,y,z),shock_rho_min,shock_p_min);
          }
    collectiveCheck(bad,"Shock primitive boundary/stencil state is nonpositive or nonfinite; review the primitive boundary conditions");
  }

  mhd_fv::State shockPrimitive(int x,int y,int z) const {
    z=(z%mesh->LocalNz+mesh->LocalNz)%mesh->LocalNz;
    const BoutReal a=shock_a(x,y,z);
    return {rho(x,y,z),v.x(x,y,z)/a,v.y(x,y,z),v.z(x,y,z)/a,
            p(x,y,z),a*B.x(x,y,z),B.y(x,y,z),a*B.z(x,y,z),glm_psi(x,y,z)};
  }

  mhd_fv::State shockEdge(int x,int y,int z,int direction,bool upper) {
    const int dx=direction==0,dy=direction==1,dz=direction==2;
    const auto q=shockPrimitive(x,y,z);
    if (shock_limiter=="first_order") return q;
    const auto qm=shockPrimitive(x-dx,y-dy,z-dz),qp=shockPrimitive(x+dx,y+dy,z+dz);
    mhd_fv::State edge{};
    for (int k=0; k<9; ++k) {
      FV::Stencil1D stencil;
      stencil.m=qm[k]; stencil.c=q[k]; stencil.p=qp[k];
      if (shock_limiter=="mc") FV::MC{}(stencil);
      else FV::MinMod{}(stencil);
      edge[k]=upper ? stencil.R : stencil.L;
    }
    if (!mhd_fv::admissible(edge,shock_rho_min,shock_p_min)) {
      ++reconstruction_fallbacks;
      return q; // Flatten the entire face state; never clip evolved cells.
    }
    return edge;
  }

  mhd_fv::State shockFace(int x,int y,int z,int direction,BoutReal& rate) {
    const int dx=direction==0,dy=direction==1,dz=direction==2;
    const auto left=shockEdge(x,y,z,direction,true);
    const auto right=shockEdge(x+dx,y+dy,z+dz,direction,false);
    const int zl=(z%mesh->LocalNz+mesh->LocalNz)%mesh->LocalNz;
    const int zr=((z+dz)%mesh->LocalNz+mesh->LocalNz)%mesh->LocalNz;
    const BoutReal af=0.5*(shock_a(x,y,zl)+shock_a(x+dx,y+dy,zr));
    BoutReal signal;
    auto flux=mhd_fv::rusanov(left,right,direction,g,glm_ch,&signal);
    rate=signal*af*af/(direction==1 ? 1. : af);
    return mhd_fv::coordinate_flux(flux,direction,af);
  }

  int shockRHS(BoutReal time) {
    shock_time=time;
    recoverShockPrimitives(time);
    ddt(rho)=0.; ddt(momentum)=0.; ddt(total_energy)=0.; ddt(B)=0.; ddt(glm_psi)=0.;
    reconstruction_fallbacks=0;
    BoutReal local_dt=std::numeric_limits<BoutReal>::max();
    for (int x=mesh->xstart; x<=mesh->xend; ++x) {
      for (int y=mesh->ystart; y<=mesh->yend; ++y) {
        for (int z=mesh->zstart; z<=mesh->zend; ++z) {
          const auto q=shockPrimitive(x,y,z);
          const BoutReal a=shock_a(x,y,z), jac=a*a;
          mhd_fv::State rhs{};
          BoutReal rate=glm_damping;
          for (int n=0; n<3; ++n) {
            BoutReal sp,sm;
            const auto plus=shockFace(x,y,z,n,sp);
            const auto minus=shockFace(x-(n==0),y-(n==1),z-(n==2),n,sm);
            for (int k=0; k<9; ++k) rhs[k]-=(plus[k]-minus[k])/(jac*shock_dx[n]);
            rate+=std::max(sp,sm)/(jac*shock_dx[n]);
            const BoutReal length=shock_dx[n]*(n==1 ? 1. : a);
            const BoutReal visc=include_viscos ? (4.*viscos/3.+bulk_viscos) : 0.;
            rate+=2.*(visc+eta)/(length*length);
          }
          const BoutReal ap=0.5*(a+shock_a(x,y+1,z)),am=0.5*(a+shock_a(x,y-1,z));
          const BoutReal H=(ap*ap-am*am)/(2.*jac*shock_dx[1]);
          // Mixed stress geometric source, and connection correction that
          // makes the conservative GLM flux a contravariant gradient of psi.
          rhs[2]+=H*(q[0]*(q[1]*q[1]+q[3]*q[3])+2.*q[4]+q[6]*q[6]);
          rhs[6]+=2.*H*q[8];
          rhs[8]-=glm_damping*q[8];
          rate+=4.*std::abs(H)*(std::abs(q[2])+mhd_fv::fast_speed(q,1,g)+glm_ch);
          local_dt=std::min(local_dt,shock_cfl/rate);
          ddt(rho)(x,y,z)=rhs[0];
          ddt(momentum).x(x,y,z)=rhs[1]; ddt(momentum).y(x,y,z)=rhs[2]; ddt(momentum).z(x,y,z)=rhs[3];
          ddt(total_energy)(x,y,z)=rhs[4];
          ddt(B).x(x,y,z)=rhs[5]; ddt(B).y(x,y,z)=rhs[6]; ddt(B).z(x,y,z)=rhs[7];
          ddt(glm_psi)(x,y,z)=rhs[8];
        }
      }
    }
    MPI_Allreduce(&local_dt,&shock_dt,1,MPI_DOUBLE,MPI_MIN,BoutComm::get());
    if (!std::isfinite(shock_dt) || shock_dt<=0.) throw BoutException("Invalid shock CFL bound");
    if (shock_first_rhs) {
      shock_step_limit=std::min(solver->getCurrentTimestep(),0.8*shock_dt);
      solver->setMaxTimestep(shock_step_limit);
      shock_first_rhs=false;
    } else if (shock_step_limit>shock_dt*(1.+1e-10)) {
      // RK3SSP changes its stored timestep for the NEXT step, not the stage
      // in progress. Keep a fixed safe cap rather than claiming stage rollback.
      throw BoutException("Shock CFL bound exceeded; choose a smaller solver:max_timestep or mhd:shock:cfl. The timestepper cannot roll back an SSP stage.");
    }
    int fallbacks=0;
    MPI_Allreduce(&reconstruction_fallbacks,&fallbacks,1,MPI_INT,MPI_SUM,BoutComm::get());
    reconstruction_fallbacks=fallbacks;
    auto d=dissipativeTerms();
    ddt(momentum)+=d.force;
    ddt(B)+=d.induction;
    // Exact thermodynamic chain rule for resolved physical dissipation.
    // With matching central operators, its periodic volume sum is zero.
    ddt(total_energy)+=v*d.force+B*d.induction+d.viscous_heat+d.ohmic_heat;
    return 0;
  }

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
    shock_capturing = options["shock_capturing"].doc("Use conservative finite-volume GLM-MHD instead of primitive ideal terms").withDefault(false);
    if (!shock_capturing) {
      options["shock"].setConditionallyUsed();
      Options::root()["glm_psi"].setConditionallyUsed();
    }

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
    B.covariant = false;
    if (shock_capturing) {
      solver->add(B, "B");
      initialiseShockState();
    } else {
      solver->add(p, "pressure");
      v.covariant = true;
      solver->add(v, "v");
      solver->add(B, "B");
    }

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

  // Store an integer: NetCDF round-trips it without a bool conversion.
  void restartVars(Options& state) override {
    state["mhd_shock_capturing"].assign(static_cast<int>(shock_capturing));
  }

  int postInit(bool restarting) override {
    if (restarting && readFromRestartFile("mhd_shock_capturing").withDefault(0)!=static_cast<int>(shock_capturing)) {
      throw BoutException("Restart formulation mismatch: use a restart from the same mhd:shock_capturing mode");
    }
    return PhysicsModel::postInit(restarting);
  }

  int outputMonitor(BoutReal time, int, int) override {
    shock_time = time;
    return 0;
  }

  void outputVars(Options& state) override {
    state["mhd_shock_capturing"].assign(static_cast<int>(shock_capturing));
    if (shock_capturing) {
      recoverShockPrimitives(shock_time);
      state["pressure"].assignRepeat(p);
      state["v_x"].assignRepeat(v.x);
      state["v_y"].assignRepeat(v.y);
      state["v_z"].assignRepeat(v.z);
      state["shock_cfl_dt"].assignRepeat(shock_dt);
      state["shock_reconstruction_fallbacks"].assignRepeat(reconstruction_fallbacks);
    }
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
    // Write the metric once. An 'm' prefix avoids BOUT output name clashes.
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

  int rhs(BoutReal time) override {
    if (shock_capturing) return shockRHS(time);
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
