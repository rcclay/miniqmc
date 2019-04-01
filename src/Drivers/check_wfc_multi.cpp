////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by:
// Jeongnim Kim, jeongnim.kim@intel.com,
//    Intel Corp.
// Ye Luo, yeluo@anl.gov,
//    Argonne National Laboratory
//
// File created by:
// Ye Luo, yeluo@anl.gov,
//    Argonne National Laboratory
////////////////////////////////////////////////////////////////////////////////
// -*- C++ -*-
/** @file check_wfc_multi.cpp
 * @brief Miniapp to check individual wave function component against its
 * reference.
 */

#include <Utilities/Configuration.h>
#include <Particle/ParticleSet.h>
#include <Particle/ParticleSet_builder.hpp>
#include <Particle/DistanceTable.h>
#include <Numerics/Containers.h>
#include <Utilities/PrimeNumberSet.h>
#include <Utilities/RandomGenerator.h>
#include <Input/Input.hpp>
#include <QMCWaveFunctions/Jastrow/PolynomialFunctor3D.h>
#include <QMCWaveFunctions/Jastrow/ThreeBodyJastrowRef.h>
#include <QMCWaveFunctions/Jastrow/ThreeBodyJastrow.h>
#include <QMCWaveFunctions/Jastrow/BsplineFunctor.h>
#include <QMCWaveFunctions/Jastrow/BsplineFunctorRef.h>
#include <QMCWaveFunctions/Jastrow/OneBodyJastrowRef.h>
#include <QMCWaveFunctions/Jastrow/OneBodyJastrow.h>
#include <QMCWaveFunctions/Jastrow/TwoBodyJastrowRef.h>
#include <QMCWaveFunctions/Jastrow/TwoBodyJastrow.h>
#include <Utilities/qmcpack_version.h>
#include <getopt.h>

using namespace std;
using namespace qmcplusplus;

void print_help()
{
  // clang-format off
  cout << "usage:" << '\n';
  cout << "  check_wfc [-hvV] [-f wfc_component] [-g \"n0 n1 n2\"]" << '\n';
  cout << "            [-r rmax] [-s seed]" << '\n';
  cout << "options:" << '\n';
  cout << "  -f  specify wavefunction component to check" << '\n';
  cout << "      one of: J1, J2, J3.            default: J2" << '\n';
  cout << "  -g  set the 3D tiling.             default: 1 1 1" << '\n';
  cout << "  -w  set the number of walkers.     default: 1" << '\n';
  cout << "  -h  print help and exit" << '\n';
  cout << "  -r  set the Rmax.                  default: 1.7" << '\n';
  cout << "  -s  set the random seed.           default: 11" << '\n';
  cout << "  -v  verbose output" << '\n';
  cout << "  -V  print version information and exit" << '\n';
  //clang-format on

  exit(1); // print help and exit
}

int main(int argc, char** argv)
{
  Kokkos::initialize(argc, argv);
  { //Begin kokkos block.


    // clang-format off
    typedef QMCTraits::RealType           RealType;
    typedef ParticleSet::ParticlePos_t    ParticlePos_t;
    typedef ParticleSet::PosType          PosType;
    // clang-format on

    // use the global generator

    int na    = 1;
    int nb    = 1;
    int nc    = 1;

    //Number of walkers
    int nW    = 1;

    int iseed = 11;
    RealType Rmax(1.7);
    string wfc_name("J2");

    bool verbose = false;

    int opt;
    while (optind < argc)
    {
      if ((opt = getopt(argc, argv, "hvVf:g:w:r:s:")) != -1)
      {
        switch (opt)
        {
        case 'f': // Wave function component
          wfc_name = optarg;
          break;
        case 'g': // tiling1 tiling2 tiling3
          sscanf(optarg, "%d %d %d", &na, &nb, &nc);
          break;
        case 'h':
          print_help();
          break;
        case 'w':
          nW = atoi(optarg);
          break;
        case 'r': // rmax
          Rmax = atof(optarg);
          break;
        case 's':
          iseed = atoi(optarg);
          break;
        case 'v':
          verbose = true;
          break;
        case 'V':
          print_version(true);
          return 1;
          break;
        default:
          print_help();
        }
      }
      else // disallow non-option arguments
      {
        cerr << "Non-option arguments not allowed" << endl;
        print_help();
      }
    }

    print_version(verbose);

    if (verbose)
      outputManager.setVerbosity(Verbosity::HIGH);
    else
      outputManager.setVerbosity(Verbosity::LOW);

    if (wfc_name != "J1" && wfc_name != "J2" && wfc_name != "J3" && wfc_name != "JeeI")
    {
      cerr << "Unknown wave function component:  " << wfc_name << endl << endl;
      print_help();
    }

    Tensor<int, 3> tmat(na, 0, 0, 0, nb, 0, 0, 0, nc);

    // setup ions
    ParticleSet ions;
    Tensor<OHMMS_PRECISION, 3> lattice_b;
    build_ions(ions, tmat, lattice_b);

    // list of accumulated errors
    double evaluateLog_v_err = 0.0;
    double evaluateLog_g_err = 0.0;
    double evaluateLog_l_err = 0.0;
    double evalGrad_g_err    = 0.0;
    double ratioGrad_r_err   = 0.0;
    double ratioGrad_g_err   = 0.0;
    double evaluateGL_g_err  = 0.0;
    double evaluateGL_l_err  = 0.0;
    double ratio_err         = 0.0;

    PrimeNumberSet<uint32_t> myPrimes;

    // clang-format off
//  #pragma omp parallel reduction(+:evaluateLog_v_err,evaluateLog_g_err,evaluateLog_l_err,evalGrad_g_err) \
   reduction(+:ratioGrad_r_err,ratioGrad_g_err,evaluateGL_g_err,evaluateGL_l_err,ratio_err)
    // clang-format on
    {
      int ip = omp_get_thread_num();

      // create generator within the thread
      RandomGenerator<RealType> random_th(myPrimes[ip]);
      
      //We are working with batches of walkers.  Here are the walker
      //"coordinates".
      std::vector<ParticleSet> el_list(nW);
      std::vector<ParticleSet> el_list_ref(nW);
      std::vector<int> ei_TableIDs(nW);
      for(int nw=0; nw<nW; nw++)
      {
        ParticleSet els;
        build_els(el_list[nw], ions, random_th);
        el_list[nw].update();
        el_list_ref[nw] = el_list[nw];
        el_list_ref[nw].RSoA = el_list_ref[nw].R;
        
        el_list[nw].addTable(el_list[nw],DT_SOA);
        el_list_ref[nw].addTable(el_list_ref[nw],DT_SOA);
        ei_TableIDs[nw] = el_list_ref[nw].addTable(ions,DT_SOA);
      }
   //   ParticleSet els;
   //   build_els(els, ions, random_th);
   //   els.update();
      int nw=0; //Dummy index for now 
      const int nions = ions.getTotalNum();
      const int nels  = el_list[nw].getTotalNum();
      const int nels3 = 3 * nels;

  //    ParticleSet els_ref(els);
 //     els_ref.RSoA = els_ref.R;

      // create tables
//      els.addTable(els, DT_SOA);
//      els_ref.addTable(els_ref, DT_SOA);
///      const int ei_TableID = els_ref.addTable(ions, DT_SOA);

      ParticlePos_t delta(nels);

      RealType sqrttau = 2.0;

      vector<RealType> ur(nels);
      random_th.generate_uniform(ur.data(), nels);

      WaveFunctionComponentPtr wfc     = nullptr;
      WaveFunctionComponentPtr wfc_ref = nullptr;
      if (wfc_name == "J2")
      {
        TwoBodyJastrow<BsplineFunctor<RealType>>* J =
            new TwoBodyJastrow<BsplineFunctor<RealType>>(el_list[nw]);
        buildJ2(*J, el_list[nw].Lattice.WignerSeitzRadius);
        wfc = dynamic_cast<WaveFunctionComponentPtr>(J);
        cout << "Built J2" << endl;
        miniqmcreference::TwoBodyJastrowRef<BsplineFunctorRef<RealType>>* J_ref =
            new miniqmcreference::TwoBodyJastrowRef<BsplineFunctorRef<RealType>>(el_list_ref[nw]);
        buildJ2(*J_ref, el_list[nw].Lattice.WignerSeitzRadius);
        wfc_ref = dynamic_cast<WaveFunctionComponentPtr>(J_ref);
        cout << "Built J2_ref" << endl;
      }
      else if (wfc_name == "J1")
      {
        OneBodyJastrow<BsplineFunctor<RealType>>* J =
            new OneBodyJastrow<BsplineFunctor<RealType>>(ions, el_list[nw]);
        buildJ1(*J, el_list[nw].Lattice.WignerSeitzRadius);
        wfc = dynamic_cast<WaveFunctionComponentPtr>(J);
        cout << "Built J1" << endl;
        miniqmcreference::OneBodyJastrowRef<BsplineFunctorRef<RealType>>* J_ref =
            new miniqmcreference::OneBodyJastrowRef<BsplineFunctorRef<RealType>>(ions, el_list_ref[nw]);
        buildJ1(*J_ref, el_list[nw].Lattice.WignerSeitzRadius);
        wfc_ref = dynamic_cast<WaveFunctionComponentPtr>(J_ref);
        cout << "Built J1_ref" << endl;
      }
      else if (wfc_name == "JeeI" || wfc_name == "J3")
      {
        ThreeBodyJastrow<PolynomialFunctor3D>* J =
            new ThreeBodyJastrow<PolynomialFunctor3D>(ions, el_list[nw]);
        buildJeeI(*J, el_list[nw].Lattice.WignerSeitzRadius);
        wfc = dynamic_cast<WaveFunctionComponentPtr>(J);
        cout << "Built JeeI" << endl;
        miniqmcreference::ThreeBodyJastrowRef<PolynomialFunctor3D>* J_ref =
            new miniqmcreference::ThreeBodyJastrowRef<PolynomialFunctor3D>(ions, el_list_ref[nw]);
        buildJeeI(*J_ref, el_list[nw].Lattice.WignerSeitzRadius);
        wfc_ref = dynamic_cast<WaveFunctionComponentPtr>(J_ref);
        cout << "Built JeeI_ref" << endl;
      }

      constexpr RealType czero(0);

      // compute distance tables
      el_list[nw].update();
      el_list_ref[nw].update();

      {
        el_list[nw].G = czero;
        el_list[nw].L = czero;
        wfc->evaluateLog(el_list[nw], el_list[nw].G, el_list[nw].L);

        el_list_ref[nw].G = czero;
        el_list_ref[nw].L = czero;
        wfc_ref->evaluateLog(el_list_ref[nw], el_list_ref[nw].G, el_list_ref[nw].L);

        cout << "Check values " << wfc->LogValue << " " << el_list[nw].G[12] << " " << el_list[nw].L[12] << endl;
        cout << "Check values ref " << wfc_ref->LogValue << " " << el_list_ref[nw].G[12] << " "
             << el_list_ref[nw].L[12] << endl
             << endl;
        cout << "evaluateLog::V Error = " << (wfc->LogValue - wfc_ref->LogValue) / nels << endl;
        evaluateLog_v_err += std::fabs((wfc->LogValue - wfc_ref->LogValue) / nels);
        {
          double g_err = 0.0;
          for (int iel = 0; iel < nels; ++iel)
          {
            PosType dr = (el_list[nw].G[iel] - el_list_ref[nw].G[iel]);
            RealType d = sqrt(dot(dr, dr));
            g_err += d;
          }
          cout << "evaluateLog::G Error = " << g_err / nels << endl;
          evaluateLog_g_err += std::fabs(g_err / nels);
        }
        {
          double l_err = 0.0;
          for (int iel = 0; iel < nels; ++iel)
          {
            l_err += abs(el_list[nw].L[iel] - el_list_ref[nw].L[iel]);
          }
          cout << "evaluateLog::L Error = " << l_err / nels << endl;
          evaluateLog_l_err += std::fabs(l_err / nels);
        }

        random_th.generate_normal(&delta[0][0], nels3);
        double g_eval  = 0.0;
        double r_ratio = 0.0;
        double g_ratio = 0.0;

        int naccepted = 0;

        for (int iel = 0; iel < nels; ++iel)
        {
          el_list[nw].setActive(iel);
          PosType grad_soa = wfc->evalGrad(el_list[nw], iel);

          el_list_ref[nw].setActive(iel);
          PosType grad_ref = wfc_ref->evalGrad(el_list_ref[nw], iel) - grad_soa;
          g_eval += sqrt(dot(grad_ref, grad_ref));

          PosType dr = sqrttau * delta[iel];
          el_list[nw].makeMoveAndCheck(iel, dr);
          bool good_ref = el_list_ref[nw].makeMoveAndCheck(iel, dr);

          if (!good_ref)
            continue;

          grad_soa       = 0;
          RealType r_soa = wfc->ratioGrad(el_list[nw], iel, grad_soa);
          grad_ref       = 0;
          RealType r_ref = wfc_ref->ratioGrad(el_list_ref[nw], iel, grad_ref);

          grad_ref -= grad_soa;
          g_ratio += sqrt(dot(grad_ref, grad_ref));
          r_ratio += abs(r_soa / r_ref - 1);

          if (ur[iel] < r_ref)
          {
            wfc->acceptMove(el_list[nw], iel);
            el_list[nw].acceptMove(iel);

            wfc_ref->acceptMove(el_list_ref[nw], iel);
            el_list_ref[nw].acceptMove(iel);
            naccepted++;
          }
          else
          {
            el_list[nw].rejectMove(iel);
            el_list_ref[nw].rejectMove(iel);
          }
        }
        cout << "Accepted " << naccepted << "/" << nels << endl;
        cout << "evalGrad::G      Error = " << g_eval / nels << endl;
        cout << "ratioGrad::G     Error = " << g_ratio / nels << endl;
        cout << "ratioGrad::Ratio Error = " << r_ratio / nels << endl;
        evalGrad_g_err += std::fabs(g_eval / nels);
        ratioGrad_g_err += std::fabs(g_ratio / nels);
        ratioGrad_r_err += std::fabs(r_ratio / nels);

        // nothing to do with J2 but needs for general cases
        el_list[nw].donePbyP();
        el_list_ref[nw].donePbyP();

        el_list[nw].G = czero;
        el_list[nw].L = czero;
        wfc->evaluateGL(el_list[nw], el_list[nw].G, el_list[nw].L);

        el_list_ref[nw].G = czero;
        el_list_ref[nw].L = czero;
        wfc_ref->evaluateGL(el_list_ref[nw], el_list_ref[nw].G, el_list_ref[nw].L);

        {
          double g_err = 0.0;
          for (int iel = 0; iel < nels; ++iel)
          {
            PosType dr = (el_list[nw].G[iel] - el_list_ref[nw].G[iel]);
            RealType d = sqrt(dot(dr, dr));
            g_err += d;
          }
          cout << "evaluteGL::G Error = " << g_err / nels << endl;
          evaluateGL_g_err += std::fabs(g_err / nels);
        }
        {
          double l_err = 0.0;
          for (int iel = 0; iel < nels; ++iel)
          {
            l_err += abs(el_list[nw].L[iel] - el_list_ref[nw].L[iel]);
          }
          cout << "evaluteGL::L Error = " << l_err / nels << endl;
          evaluateGL_l_err += std::fabs(l_err / nels);
        }

        // now ratio only
        r_ratio              = 0.0;
        constexpr int nknots = 12;
        int nsphere          = 0;
        for (int jel = 0; jel < el_list_ref[nw].getTotalNum(); ++jel)
        {
          const auto& dist = el_list_ref[nw].DistTables[ei_TableIDs[nw]]->Distances[jel];
          for (int iat = 0; iat < nions; ++iat)
            if (dist[iat] < Rmax)
            {
              nsphere++;
              random_th.generate_uniform(&delta[0][0], nknots * 3);
              for (int k = 0; k < nknots; ++k)
              {
                el_list[nw].makeMoveOnSphere(jel, delta[k]);
                RealType r_soa = wfc->ratio(el_list[nw], jel);
                el_list[nw].rejectMove(jel);

                el_list_ref[nw].makeMoveOnSphere(jel, delta[k]);
                RealType r_ref = wfc_ref->ratio(el_list_ref[nw], jel);
                el_list_ref[nw].rejectMove(jel);
                r_ratio += abs(r_soa / r_ref - 1);
              }
            }
        }
        cout << "ratio with SphereMove  Error = " << r_ratio / nsphere << " # of moves =" << nsphere
             << endl;
        ratio_err += std::fabs(r_ratio / (nels * nknots));
      }
    } // end of omp parallel

    int np                   = omp_get_max_threads();
    constexpr RealType small = std::numeric_limits<RealType>::epsilon() * 1e4;
    bool fail                = false;
    cout << std::endl;
    if (evaluateLog_v_err / np > small)
    {
      cout << "Fail in evaluateLog, V error =" << evaluateLog_v_err / np << " for " << wfc_name
           << std::endl;
      fail = true;
    }
    if (evaluateLog_g_err / np > small)
    {
      cout << "Fail in evaluateLog, G error =" << evaluateLog_g_err / np << " for " << wfc_name
           << std::endl;
      fail = true;
    }
    if (evaluateLog_l_err / np > small)
    {
      cout << "Fail in evaluateLog, L error =" << evaluateLog_l_err / np << " for " << wfc_name
           << std::endl;
      fail = true;
    }
    if (evalGrad_g_err / np > small)
    {
      cout << "Fail in evalGrad, G error =" << evalGrad_g_err / np << " for " << wfc_name
           << std::endl;
      fail = true;
    }
    if (ratioGrad_r_err / np > small)
    {
      cout << "Fail in ratioGrad, ratio error =" << ratioGrad_r_err / np << " for " << wfc_name
           << std::endl;
      fail = true;
    }
    if (ratioGrad_g_err / np > small)
    {
      cout << "Fail in ratioGrad, G error =" << ratioGrad_g_err / np << " for " << wfc_name
           << std::endl;
      fail = true;
    }
    if (evaluateGL_g_err / np > small)
    {
      cout << "Fail in evaluateGL, G error =" << evaluateGL_g_err / np << " for " << wfc_name
           << std::endl;
      fail = true;
    }
    if (evaluateGL_l_err / np > small)
    {
      cout << "Fail in evaluateGL, L error =" << evaluateGL_l_err / np << " for " << wfc_name
           << std::endl;
      fail = true;
    }
    if (ratio_err / np > small)
    {
      cout << "Fail in ratio, ratio error =" << ratio_err / np << " for " << wfc_name << std::endl;
      fail = true;
    }
    if (!fail)
      cout << "All checks passed for " << wfc_name << std::endl;

  } //end kokkos block
  Kokkos::finalize();
  return 0;
}
