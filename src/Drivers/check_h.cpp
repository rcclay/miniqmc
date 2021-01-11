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

#include <Utilities/Configuration.h>
#include <Utilities/Communicate.h>
#include <Particle/ParticleSet.h>
#include <Particle/DistanceTable.h>
#include <Utilities/PrimeNumberSet.h>
#include <Utilities/NewTimer.h>
#include <Utilities/XMLWriter.h>
#include <Utilities/RandomGenerator.h>
#include <Utilities/qmcpack_version.h>
#include <Input/Input.hpp>
#include <QMCWaveFunctions/SPOSet.h>
#include <QMCWaveFunctions/SPOSet_builder.h>
#include <QMCWaveFunctions/WaveFunction.h>
#include <Drivers/Mover.hpp>
#include <getopt.h>


using namespace qmcplusplus;
using namespace std;

enum HTimers
{
  Timer_Total,
  Timer_Init,
  Timer_Eval,
};

TimerNameList_t<HTimers> HTimerNames = {
    {Timer_Total, "Total"},
    {Timer_Init, "Initialization"},
    {Timer_Eval, "Evaluation"},
};

void print_help()
{
  // clang-format off
  app_summary() << "usage:" << '\n';
  app_summary() << "  miniqmc   [-bhjvV] [-g \"n0 n1 n2\"] [-m meshfactor]"      << '\n';
  app_summary() << "            [-n steps] [-N substeps] [-x rmax]"              << '\n';
  app_summary() << "            [-r AcceptanceRatio] [-s seed] [-w walkers]"     << '\n';
  app_summary() << "            [-a tile_size] [-t timer_level] [-k delay_rank]" << '\n';
  app_summary() << "options:"                                                    << '\n';
  app_summary() << "  -a  size of each spline tile       default: num of orbs"   << '\n';
  app_summary() << "  -b  use reference implementations  default: off"           << '\n';
  app_summary() << "  -g  set the 3D tiling.             default: 1 1 1"         << '\n';
  app_summary() << "  -h  print help and exit"                                   << '\n';
  app_summary() << "  -j  enable three body Jastrow      default: off"           << '\n';
  app_summary() << "  -m  meshfactor                     default: 1.0"           << '\n';
  app_summary() << "  -s  set the random seed.           default: 11"            << '\n';
  app_summary() << "  -t  timer level: coarse or fine    default: fine"          << '\n';
  app_summary() << "  -k  matrix delayed update rank     default: 32"            << '\n';
  app_summary() << "  -v  verbose output"                                        << '\n';
  app_summary() << "  -V  print version information and exit"                    << '\n';
  app_summary() << "  -w  number of walker(movers)       default: num of threads"<< '\n';
  app_summary() << "  -x  set the Rmax.                  default: 1.7"           << '\n';
  // clang-format on
}

int main(int argc, char** argv)
{
  
  // clang-format off
  typedef QMCTraits::RealType           RealType;
  typedef ParticleSet::ParticlePos_t    ParticlePos_t;
  typedef ParticleSet::PosType          PosType;
  // clang-format on

  int na     = 1;
  int nb     = 1;
  int nc     = 1;
  int nx = 37, ny = 37, nz = 37;
  int tileSize  = -1;
  int delay_rank=0;

  RealType Rmax(1.7);

  bool useRef   = false;
  bool enableJ3 = false;

  bool verbose                 = false;
  std::string timer_level_name = "fine";
  
  int opt;
  while (optind < argc)
  {
    if ((opt = getopt(argc, argv, "bhjvVa:c:g:m:n:N:r:s:t:k:w:x:")) != -1)
    {
      switch (opt)
      {
      case 'a':
        tileSize = atoi(optarg);
        break;
      case 'b':
        useRef = true;
        break;
      case 'g': // tiling1 tiling2 tiling3
        sscanf(optarg, "%d %d %d", &na, &nb, &nc);
        break;
      case 'h':
        print_help();
        return 1;
        break;
      case 'j':
        enableJ3 = true;
        break;
      case 'm':
      {
        const RealType meshfactor = atof(optarg);
        nx *= meshfactor;
        ny *= meshfactor;
        nz *= meshfactor;
      }
      break;
      case 't':
        timer_level_name = std::string(optarg);
        break;
      case 'v':
        verbose = true;
        break;
      case 'V':
        print_version(true);
        return 1;
        break;
      case 'x': // rmax
        Rmax = atof(optarg);
        break;
      default:
        print_help();
        return 1;
      }
    }
    else // disallow non-option arguments
    {
      app_error() << "Non-option arguments not allowed" << endl;
      print_help();
    }
  }

  int number_of_electrons = 0;

  Tensor<int, 3> tmat(na, 0, 0, 0, nb, 0, 0, 0, nc);

  timer_levels timer_level = timer_level_fine;
  if (timer_level_name == "coarse")
  {
    timer_level = timer_level_coarse;
  }
  else if (timer_level_name != "fine")
  {
    app_error() << "Timer level should be 'coarse' or 'fine', name given: " << timer_level_name
                << endl;
    return 1;
  }

  TimerManager.set_timer_threshold(timer_level);
  TimerList_t Timers;
  setup_timers(Timers, HTimerNames, timer_level_coarse);

  print_version(verbose);

  SPOSet* spo_main;
  int nTiles = 1;

  PrimeNumberSet<uint32_t> myPrimes;
  RandomGenerator<RealType> rng(myPrimes[0]); 


  ParticleSet ions;
  // initialize ions and splines which are shared by all threads later
  {
    Tensor<OHMMS_PRECISION, 3> lattice_b;
    build_ions(ions, tmat, lattice_b);
    const int nels = count_electrons(ions, 1);
    const int norb = nels / 2;
    tileSize       = (tileSize > 0) ? tileSize : norb;
    nTiles         = norb / tileSize;

    number_of_electrons = nels;

    const size_t SPO_coeff_size =
        static_cast<size_t>(norb) * (nx + 3) * (ny + 3) * (nz + 3) * sizeof(RealType);
    const double SPO_coeff_size_MB = SPO_coeff_size * 1.0 / 1024 / 1024;

    app_summary() << "Number of orbitals/splines = " << norb << endl
                  << "Tile size = " << tileSize << endl
                  << "Number of tiles = " << nTiles << endl
                  << "Number of electrons = " << nels << endl
                  << "Rmax = " << Rmax << endl;
#ifdef HAVE_MPI
    app_summary() << "MPI processes = " << comm.size() << endl;
#endif
    app_summary() << "OpenMP threads = " << omp_get_max_threads() << endl;

    app_summary() << "\nSPO coefficients size = " << SPO_coeff_size << " bytes ("
                  << SPO_coeff_size_MB << " MB)" << endl;
    app_summary() << "delayed update rank = " << delay_rank << endl;


    spo_main = build_SPOSet(useRef, nx, ny, nz, norb, nTiles, lattice_b);
  }

  if (!useRef)
    app_summary() << "Using SoA distance table, Jastrow + einspline, " << endl
                  << "and determinant update." << endl;
  else
    app_summary() << "Using the reference implementation for Jastrow, " << endl
                  << "determinant update, and distance table + einspline of the " << endl
                  << "reference implementation " << endl;

  Timers[Timer_Total]->start();

  ParticleSet els;
  build_els(els, ions, rng);
  
  WaveFunction wfn;
  build_WaveFunction(useRef, spo_main, wfn, ions, els, rng, delay_rank, enableJ3);  
  
  BareKineticEnergy bk;
 


  Timers[Timer_Eval]->start();
  els.update();
  wfn.evaluateLog(els);


  double ke_val(0.0);
  ke_val=bk.evaluate(els,wfn);
  Timers[Timer_Eval]->stop();
  std:cout<<"KE = "<<ke_val<<std::endl;

  TimerManager.print(); 
  return 0;
}
