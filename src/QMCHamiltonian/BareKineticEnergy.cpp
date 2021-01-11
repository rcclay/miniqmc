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
  //This function assumes that M^-1 already exists and is ready for fast computations.  
  
  //TrialWavefunction takes care of this automatically, but the way things are computed right now, 
  //Laplacian is built up from the following formula:
  //
  // $\Psi_T^{-1} \nabla^2 \Psi_T = \nabla^2 \ln(\Psi_T) + \left( \nabla \ln(\Psi_T) \right) \cdot \left( \nabla \ln(\Psi_T) \right)$
  // $\ln(\Psi_T)$ is nice because since we have a product wavefunction, each needed quantity is computed for each WaveFunctionComponent separately, then added together.  
  //
  //


  //preliminary temporary storage initialization. 
  int numParticles = P.getTotalNum(); 

  // gradients of the particles
  ParticleSet::ParticleGradient_t G;
  // laplacians of the particles
  ParticleSet::ParticleLaplacian_t L;
  
  G.resize(numParticles);
  L.resize(numParticles); 

  double lapl=0.0;
 // wfn.evaluateLog(P);
  for (int i=0; i<numParticles; i++)
  {
   lapl+=P.L[i]+dot(P.G[i],P.G[i]); 
  }
  
  return -0.5*lapl;
}

}
