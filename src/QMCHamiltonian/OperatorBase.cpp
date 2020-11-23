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

#include <Utilities/Configuration.h>
#include <Utilities/RandomGenerator.h>
#include <Particle/ParticleSet.h>
#include "QMCWaveFunctions/SPOSet.h"
#include <QMCWaveFunctions/WaveFunction.h>
#include <Particle/ParticleSet_builder.hpp>
#include <Drivers/NonLocalPP.hpp>
#include <QMCHamiltonian/OperatorBase.hpp>
#include <vector>

namespace qmcplusplus
{

/** Take o_list and p_list update evaluation result variables in o_list?
 *
 * really should reduce vector of local_energies. matching the ordering and size of o list
 * the this can be call for 1 or more QMCHamiltonians
 */
void OperatorBase::mw_evaluate(const std::vector<OperatorBase>& O_list, const std::vector<ParticleSet>& P_list)
{
  /**  Temporary raw omp pragma for simple thread parallelism
   *   ignoring the driver level concurrency
   *   
   *  \todo replace this with a proper abstraction. It should adequately describe the behavior
   *  and strictly limit the activation of this level concurrency to when it is intended.
   *  It is unlikely to belong in this function.
   *  
   *  This implicitly depends on openmp work division logic. Essentially adhoc runtime
   *  crowds over which we have given up control of thread/global scope.
   *  How many walkers per thread? How to handle their data movement if any of these
   *  hamiltonians should be accelerated? We can neither reason about or describe it in C++
   *
   *  As I understand it it should only be required for as long as the AMD openmp offload 
   *  compliler is incapable of running multiple threads. They should/must fix their compiler
   *  before delivery of frontier and it should be removed at that point at latest
   *
   *  If you want 16 threads of 1 walker that should be 16 crowds of 1
   *  not one crowd of 16 with openmp thrown in at hamiltonian level.
   *  If this must be different from the other crowd batching. Make this a reasoned about
   *  and controlled level of concurency blocking at the driver level.
   *
   *  This is only thread safe only if each walker has a complete
   *  set of anything involved in an Operator.evaluate.
   */
 // #pragma omp parallel for
//  for (int iw = 0; iw < O_list.size(); iw++)
//    O_list[iw].get().evaluate(P_list[iw]);
}

}
