//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//                    Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//
// File created by: Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//////////////////////////////////////////////////////////////////////////////////////


/** @file VirtualParticleSet.h
 * A proxy class to the quantum ParticleSet
 */
#ifndef QMCPLUSPLUS_VIRTUAL_PARTICLESET_H
#define QMCPLUSPLUS_VIRTUAL_PARTICLESET_H

#include <Utilities/Configuration.h>
#include <Particle/ParticleSet.h>

namespace qmcplusplus
{
/** Introduced to handle virtual moves and ratio computations, e.g. for non-local PP evaluations.
   */
class VirtualParticleSet : public ParticleSet
{
private:
  /// true, if virtual particles are on a sphere for NLPP
  bool onSphere;

public:
  /// Reference particle
  int refPtcl;
  /// Reference source particle, used when onSphere=true
  int refSourcePtcl;

  /// ParticleSet this object refers to
  const ParticleSet& refPS;

  inline bool isOnSphere() const { return onSphere; }

  /** constructor 
     * @param p ParticleSet whose virtual moves are handled by this object
     * @param nptcl number of virtual particles
     */
  VirtualParticleSet(const ParticleSet& p, int nptcl);

  /** move virtual particles to new postions and update distance tables
     * @param jel reference particle that all the VP moves from
     * @param vitualPos new positions
     * @param sphere set true if VP are on a sphere around the reference source particle
     * @param iat reference source particle
     */
  void makeMoves(int jel, const ParticlePos_t& vitualPos, bool sphere = false, int iat = -1);
};
} // namespace qmcplusplus
#endif
