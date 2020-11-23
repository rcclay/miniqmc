#//////////////////////////////////////////////////////////////////////////////////////
#// This file is distributed under the University of Illinois/NCSA Open Source License.
#// See LICENSE file in top directory for details.
#//
#// Copyright (c) 2019 QMCPACK developers.
#//
#// File developed by: Raymond Clay, rclay@sandia.gov, Sandia National Laboratories 
#//
#// File created by: Raymond Clay, rclay@sandia.gov, Sandia National Laboratories 
#//////////////////////////////////////////////////////////////////////////////////////

ADD_LIBRARY(qmcham
             ../QMCHamiltonian/OperatorBase.cpp
             ../QMCHamiltonian/BareKineticEnergy.cpp) 

TARGET_LINK_LIBRARIES(qmcham PRIVATE Math::BLAS_LAPACK)

#SUBDIRS(tests)
