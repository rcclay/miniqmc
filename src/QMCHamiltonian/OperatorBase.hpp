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
 * @file OperatorBase.hpp
 * @brief Declaration of OperatorBase base class
 *
 */

#ifndef QMCPLUSPLUS_OPERATORBASE_HPP
#define QMCPLUSPLUS_OPERATORBASE_HPP

#include <Utilities/Configuration.h>
#include <Utilities/RandomGenerator.h>
#include <Particle/ParticleSet.h>
#include "QMCWaveFunctions/SPOSet.h"
#include <QMCWaveFunctions/WaveFunction.h>
#include <Particle/ParticleSet_builder.hpp>
#include <Drivers/NonLocalPP.hpp>
#include <vector>
namespace qmcplusplus
{
/**
   * @brief This class evaluates the local kinetic energy given a wavefunction and particle set
   *
   */

struct OperatorBase
{

  //constructor
  OperatorBase(){};
  //virtual destructor
  virtual ~OperatorBase() {} 

  virtual double evaluate(ParticleSet& P, WaveFunction & wf) = 0; //I'll make the return type generic later
  /** Evaluate the contribution of this component of multiple walkers */
  virtual void mw_evaluate(const std::vector<OperatorBase>& O_list, const std::vector<ParticleSet>& P_list);
};
} // namespace qmcplusplus

#endif
