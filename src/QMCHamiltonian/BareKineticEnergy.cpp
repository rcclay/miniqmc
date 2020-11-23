////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by:
// Raymond Clay, rclay@sandia.gov,
//    Sandia National Laboratories
//
// File created by:
// Raymond Clay, rclay@sandia.gov,
//    Sandia National Laboratories
////////////////////////////////////////////////////////////////////////////////
// -*- C++ -*-

#include <Utilities/Configuration.h>
#include <Utilities/RandomGenerator.h>
#include <Particle/ParticleSet.h>
#include "QMCWaveFunctions/SPOSet.h"
#include <QMCWaveFunctions/WaveFunction.h>
#include <Particle/ParticleSet_builder.hpp>
#include <Drivers/NonLocalPP.hpp>
#include <QMCHamiltonian/BareKineticEnergy.hpp>
#include <vector>

namespace qmcplusplus
{

double BareKineticEnergy::evaluate(ParticleSet& P, WaveFunction & wfn)
{
  std::cout<<"Hello from BareKineticEnergy\n";
  return 0.0;
}

}
