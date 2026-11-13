#pragma once

#include <algorithm>
#include <array>
#include <cmath>

// Conservative GLM-MHD fluxes in a local orthonormal frame, mu0=1.
// Primitive: rho, u[3], p, B[3], psi. Conserved: rho, momentum[3], E, B[3], psi.
// E excludes the auxiliary GLM scalar, as in the conservative Dedner/GLM system.
// The time integrator and metric conversion are supplied by the BOUT++ model.
namespace mhd_fv {
using State = std::array<double, 9>;
constexpr int density = 0, energy = 4, magnetic = 5, cleaning = 8;

inline bool admissible(const State& q, double rho_min, double p_min) {
  for (double x : q) if (!std::isfinite(x)) return false;
  return q[0] > rho_min && q[4] > p_min;
}

inline State conserved(const State& q, double gamma) {
  State u = q;
  double kinetic = 0., magnetic_energy = 0.;
  for (int i = 0; i < 3; ++i) {
    u[1+i] = q[0]*q[1+i];
    kinetic += 0.5*q[0]*q[1+i]*q[1+i];
    magnetic_energy += 0.5*q[5+i]*q[5+i];
  }
  u[4] = q[4]/(gamma-1.) + kinetic + magnetic_energy;
  return u;
}

inline double fast_speed(const State& q, int n, double gamma) {
  double b2 = 0.;
  for (int i = 0; i < 3; ++i) b2 += q[5+i]*q[5+i];
  const double cs2 = gamma*q[4]/q[0];
  const double sum = cs2 + b2/q[0];
  const double disc = std::max(0., sum*sum - 4.*cs2*q[5+n]*q[5+n]/q[0]);
  return std::sqrt(0.5*(sum+std::sqrt(disc)));
}

inline State physical_flux(const State& q, int n, double gamma, double ch) {
  const State u = conserved(q, gamma);
  double b2 = 0., ub = 0.;
  for (int i = 0; i < 3; ++i) {
    b2 += q[5+i]*q[5+i];
    ub += q[1+i]*q[5+i];
  }
  const double pt = q[4]+0.5*b2;
  State f{};
  f[0] = q[0]*q[1+n];
  for (int i = 0; i < 3; ++i) {
    f[1+i] = q[0]*q[1+n]*q[1+i] - q[5+n]*q[5+i] + (i==n ? pt : 0.);
    f[5+i] = q[1+n]*q[5+i] - q[1+i]*q[5+n] + (i==n ? q[8] : 0.);
  }
  f[4] = (u[4]+pt)*q[1+n] - ub*q[5+n];
  f[8] = ch*ch*q[5+n];
  return f;
}

inline State rusanov(const State& left, const State& right, int n,
                     double gamma, double ch, double* signal_speed = nullptr) {
  const double speed = std::max({std::abs(left[1+n])+fast_speed(left,n,gamma),
                                std::abs(right[1+n])+fast_speed(right,n,gamma),ch});
  if (signal_speed) *signal_speed = speed;
  const State ul = conserved(left,gamma), ur = conserved(right,gamma);
  const State fl = physical_flux(left,n,gamma,ch), fr = physical_flux(right,n,gamma,ch);
  State f{};
  for (int k = 0; k < 9; ++k) f[k] = 0.5*(fl[k]+fr[k]-speed*(ur[k]-ul[k]));
  return f;
}

// J F^n for ds^2=a(y)^2 dx^2 + dy^2 + a(y)^2 dz^2.
inline State coordinate_flux(State f, int n, double a) {
  const std::array<double,3> h{a,1.,a};
  const double area = a*a/h[n];
  for (int k = 0; k < 9; ++k) f[k] *= area;
  for (int i = 0; i < 3; ++i) {
    f[1+i] *= h[i];
    f[5+i] /= h[i];
  }
  return f;
}
} // namespace mhd_fv
