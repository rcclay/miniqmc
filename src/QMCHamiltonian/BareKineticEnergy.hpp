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

/**
 * @file BareKineticEnergy.h
 * @brief Declaration of BareKineticEnergy class
 *
 */

#ifndef QMCPLUSPLUS_BAREKINETIC_HPP
#define QMCPLUSPLUS_BAREKINETIC_HPP

#include <Utilities/Configuration.h>
#include <Utilities/RandomGenerator.h>
#include <Particle/ParticleSet.h>
#include "QMCWaveFunctions/SPOSet.h"
#include <QMCWaveFunctions/WaveFunction.h>
#include <Particle/ParticleSet_builder.hpp>
#include <QMCHamiltonian/OperatorBase.hpp>
#include <Drivers/NonLocalPP.hpp>

namespace qmcplusplus
{
/**
   * @brief This class evaluates the local kinetic energy given a wavefunction and particle set
   *
   */
class BareKineticEnergy : public OperatorBase 
{
  public:
    BareKineticEnergy(){};
    ~BareKineticEnergy(){};

    double evaluate(ParticleSet& P, WaveFunction& wfn) override;
  private:
   
};

} // namespace qmcplusplus

#endif
