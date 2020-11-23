////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by:
// Jeongnim Kim, jeongnim.kim@intel.com,
//    Intel Corp.
// Amrita Mathuriya, amrita.mathuriya@intel.com,
//    Intel Corp.
//
// File created by:
// Jeongnim Kim, jeongnim.kim@intel.com,
//    Intel Corp.
////////////////////////////////////////////////////////////////////////////////
// -*- C++ -*-
/** @file check_spo.cpp
 * @brief Miniapp to check 3D spline implementation against the reference.
 */
#include <iostream>
#include <QMCHamiltonian/BareKineticEnergy.hpp>
#include <QMCWaveFunctions/WaveFunction.h>
#include <Particle/ParticleSet.h>

using namespace qmcplusplus;
using namespace std;

int main(int argc, char** argv)
{
  ParticleSet electrons;
  WaveFunction wfn;
  std::cout<<"Hello world\n";
  BareKineticEnergy bk;
  bk.evaluate(electrons,wfn);
  return 0;
}
