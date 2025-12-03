/*
   Copyright (c) 2012 The Regents of the University of California,
   through Lawrence Berkeley National Laboratory.  

Author: Lin Lin, Wei Hu and Amartya Banerjee

This file is part of DGDFT. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

(1) Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.
(2) Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.
(3) Neither the name of the University of California, Lawrence Berkeley
National Laboratory, U.S. Dept. of Energy nor the names of its contributors may
be used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

You are under no obligation whatsoever to provide any bug fixes, patches, or
upgrades to the features, functionality or performance of the source code
("Enhancements") to anyone; however, if you choose to make your Enhancements
available either publicly, or directly to Lawrence Berkeley National
Laboratory, without imposing a separate written license agreement for such
Enhancements, then you hereby grant the following license: a non-exclusive,
royalty-free perpetual license to install, use, modify, prepare derivative
works, incorporate into other computer software, distribute, and sublicense
such enhancements or derivative works thereof, in binary and source code form.
 */
/// @file scf.cpp
/// @brief SCF class for the global domain or extended element.
/// @date 2012-10-25 Initial version
/// @date 2014-02-01 Dual grid implementation
/// @date 2014-08-07 Parallelization for PWDFT
/// @date 2016-01-19 Add hybrid functional
/// @date 2016-04-08 Update mixing
#include  "scf.hpp"
#include  "blas.hpp"
#include  "lapack.hpp"
#include  "scalapack.hpp"
#include  "mpi_interf.hpp"
#include  "utility.hpp"
#include  "spinor.hpp"
#include  "periodtable.hpp"
#ifdef _GPUGPU
#include "cuda_utils.h"
//#include  "magma.hpp"
#include  "cublas.hpp"
#endif

namespace  dgdft{

using namespace dgdft::DensityComponent;
using namespace dgdft::esdf;
using namespace dgdft::scalapack;

SCF::SCF    (  )
{
  eigSolPtr_ = NULL;
  ptablePtr_ = NULL;
}         // -----  end of method SCF::SCF  ----- 

SCF::~SCF    (  )
{

}         // -----  end of method SCF::~SCF  ----- 

void
SCF::Setup    ( EigenSolver& eigSol, PeriodTable& ptable )
{
  int mpirank;  MPI_Comm_rank(esdfParam.domain.comm, &mpirank);
  int mpisize;  MPI_Comm_size(esdfParam.domain.comm, &mpisize);
  Real timeSta, timeEnd;

  // esdf parameters
  {
    mixMaxDim_     = esdfParam.mixMaxDim;
    mixType_       = esdfParam.mixType;
    mixVariable_   = esdfParam.mixVariable;
    mixStepLength_ = esdfParam.mixStepLength;
    // Note: for PW SCF there is no inner loop. Use the parameter value
    // for the outer SCF loop only.
    eigTolerance_  = esdfParam.eigTolerance;
    eigMinTolerance_  = esdfParam.eigMinTolerance;
    eigMaxIter_    = esdfParam.eigMaxIter;
    scfTolerance_  = esdfParam.scfOuterTolerance;
    scfMaxIter_    = esdfParam.scfOuterMaxIter;
    scfPhiMaxIter_ = esdfParam.scfPhiMaxIter;
    scfPhiTolerance_ = esdfParam.scfPhiTolerance;
    isEigToleranceDynamic_ = esdfParam.isEigToleranceDynamic;
    BlockSizeScaLAPACK_      = esdfParam.BlockSizeScaLAPACK;

    EIonSR_ = 0.0;
    Efock_ = 0.0;


    // Choice of smearing scheme : Fermi-Dirac (FD) or Gaussian_Broadening (GB) or Methfessel-Paxton (MP)
    // Currently PEXSI only supports FD smearing, so GB or MP have to be used with diag type methods
    SmearingScheme_ = esdfParam.smearing_scheme;
    if(SmearingScheme_ == "GB")
      MP_smearing_order_ = 0;
    else if(SmearingScheme_ == "MP")
      MP_smearing_order_ = 2;
    else
      MP_smearing_order_ = -1; // For safety

    Tbeta_            = esdfParam.Tbeta;
    Tsigma_           = 1.0 / Tbeta_;

    // Chebyshev Filtering related parameters
    if(esdfParam.PWSolver == "CheFSI")
      Diag_SCF_PWDFT_by_Cheby_ = 1;
    else
      Diag_SCF_PWDFT_by_Cheby_ = 0;

    First_SCF_PWDFT_ChebyFilterOrder_ = esdfParam.First_SCF_PWDFT_ChebyFilterOrder;
    First_SCF_PWDFT_ChebyCycleNum_ = esdfParam.First_SCF_PWDFT_ChebyCycleNum;
    General_SCF_PWDFT_ChebyFilterOrder_ = esdfParam.General_SCF_PWDFT_ChebyFilterOrder;
    PWDFT_Cheby_use_scala_ = esdfParam.PWDFT_Cheby_use_scala;
    PWDFT_Cheby_apply_wfn_ecut_filt_ =  esdfParam.PWDFT_Cheby_apply_wfn_ecut_filt;
    Cheby_iondynamics_schedule_flag_ = 0;


  }

  // other SCF parameters
  {
    eigSolPtr_ = &eigSol;
    ptablePtr_ = &ptable;

    //        Int ntot = eigSolPtr_->Psi().NumGridTotal();
//    Int ntot = esdfParam.domain.NumGridTotal();
    Int ntotFine = esdfParam.domain.NumGridTotalFine();
    Int ntot = esdfParam.domain.NumGridTotalFine();

    vtotNew_.Resize(ntotFine); SetValue(vtotNew_, 0.0);
    if( mixVariable_ == "density" ){
      densityOld_.Resize(ntotFine); SetValue(densityOld_, 0.0);
    }

 // Anderson mixing is performed in real space, while Broyden mixing
    // is performed in reciprocal space
    if( mixType_ == "anderson"|| mixType_ == "kerker+anderson" ){
      dfMat_.Resize( ntotFine, mixMaxDim_ ); SetValue( dfMat_, D_ZERO );
      dvMat_.Resize( ntotFine, mixMaxDim_ ); SetValue( dvMat_, D_ZERO );
    }
    else if( mixType_ == "broyden"){
      GdfMat_.Resize( ntot, mixMaxDim_ ); SetValue( GdfMat_, Z_ZERO );
      GdvMat_.Resize( ntot, mixMaxDim_ ); SetValue( GdvMat_, Z_ZERO );
      GcdfMat_.Resize( ntot, 2 ); SetValue( GcdfMat_, Z_ZERO );
    }
    else{
       ErrorHandling("Invalid mixing type.");
    }
    
    restartDensityFileName_ = "DEN";
    restartPotentialFileName_ = "POT";
    restartWfnFileName_     = "WFN";
  }

  // Density
  {
    Hamiltonian& ham = eigSolPtr_->Ham();
    DblNumMat&  density = ham.Density();

    if( esdfParam.isRestartDensity ) {
      std::istringstream rhoStream;      
      SharedRead( restartDensityFileName_, rhoStream);
      // TODO Error checking
      // Read the grid
      std::vector<DblNumVec> gridpos(DIM);
      for( Int d = 0; d < DIM; d++ ){
        deserialize( gridpos[d], rhoStream, NO_MASK );
      }
      DblNumVec densityVec;
      // only for restricted spin case
      deserialize( densityVec, rhoStream, NO_MASK );    
      blas::Copy( densityVec.m(), densityVec.Data(), 1, 
          density.VecData(RHO), 1 );
      statusOFS << "Density restarted from file " 
        << restartDensityFileName_ << std::endl;
    } // else using the zero initial guess
    else{
      if( esdfParam.isUseAtomDensity ){
#if ( _DEBUGlevel_ >= 0 )
        statusOFS 
          << "Use superposition of atomic density as initial "
          << "guess for electron density." << std::endl;
#endif
        
        GetTime( timeSta );
        ham.CalculateAtomDensity( *ptablePtr_, eigSolPtr_->FFT() );
        GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
        statusOFS << "Time for calculating the atomic density = " 
          << timeEnd - timeSta << " [s]" << std::endl;
#endif

        // Use the superposition of atomic density as the initial guess for density
        const Domain& dm = esdfParam.domain;
        Int ntotFine = dm.NumGridTotalFine();

        SetValue( density, 0.0 );
        blas::Copy( ntotFine, ham.AtomDensity().Data(), 1, 
            density.VecData(0), 1 );

      }
      else{
        // Start from pseudocharge, usually this is not a very good idea
        // make sure the pseudocharge is initialized
#if ( _DEBUGlevel_ >= 0 )
        statusOFS << "Generating initial density through linear combination of pseudocharges." 
          << std::endl;
#endif
        DblNumVec&  pseudoCharge = ham.PseudoCharge();
        const Domain& dm = esdfParam.domain;

        SetValue( density, 0.0 );

        Int ntotFine = dm.NumGridTotalFine();

        Real sum0 = 0.0, sum1 = 0.0;
        Real EPS = 1e-14;

        // make sure that the electron density is positive
        for (Int i=0; i<ntotFine; i++){
          density(i, RHO) = ( pseudoCharge(i) > EPS ) ? pseudoCharge(i) : EPS;
          //                density(i, RHO) = pseudoCharge(i);
          sum0 += density(i, RHO);
          sum1 += pseudoCharge(i);
        }

        Print( statusOFS, "Initial density. Sum of density      = ", 
            sum0 * dm.Volume() / dm.NumGridTotalFine() );

        // Rescale the density
        for (int i=0; i <ntotFine; i++){
          density(i, RHO) *= sum1 / sum0;
        } 

        Print( statusOFS, "Rescaled density. Sum of density      = ", 
            sum1 * dm.Volume() / dm.NumGridTotalFine() );
      }
    }
  }

  if( !esdfParam.isRestartWfn ) {
    // Randomized input from outside
    // Setup the occupation rate by aufbau principle (needed for hybrid functional calculation)
    DblNumVec& occ = eigSolPtr_->Ham().OccupationRate();
    Int npsi = eigSolPtr_->Psi().NumStateTotal();
    Int nocc = eigSolPtr_->Ham().NumOccupiedState();
    occ.Resize( npsi );
    SetValue( occ, 0.0 );
    for( Int k = 0; k < nocc; k++ ){
      occ[k] = 1.0;
    }
  }
  else {
    std::istringstream wfnStream;
    SeparateRead( restartWfnFileName_, wfnStream, mpirank );
    deserialize( eigSolPtr_->Psi().Wavefun(), wfnStream, NO_MASK );
    deserialize( eigSolPtr_->Ham().OccupationRate(), wfnStream, NO_MASK );
    statusOFS << "Wavefunction restarted from file " 
      << restartWfnFileName_ << std::endl;
  }

  // XC functional
  {
    isCalculateGradRho_ = false;
    if( esdfParam.XCType == "XC_GGA_XC_PBE" || 
        esdfParam.XCType == "XC_HYB_GGA_XC_HSE06" ||
        esdfParam.XCType == "XC_HYB_GGA_XC_PBEH" ) {
      isCalculateGradRho_ = true;
    }
  }


  // FIXME external force when needed

  return ;
}         // -----  end of method SCF::Setup  ----- 

void
SCF::Update    ( )
{
  {
    Int ntotFine  = eigSolPtr_->FFT().domain.NumGridTotalFine();

   ///FIXME :: real space mixing
    Int ntot = esdfParam.domain.NumGridTotalFine();

    vtotNew_.Resize(ntotFine); SetValue(vtotNew_, 0.0);
    if( mixVariable_ == "density" ){
      densityOld_.Resize(ntotFine); SetValue(densityOld_, 0.0);
    }

    if( mixType_ == "anderson"|| mixType_ == "kerker+anderson" ){
      dfMat_.Resize( ntotFine, mixMaxDim_ ); SetValue( dfMat_, D_ZERO );
      dvMat_.Resize( ntotFine, mixMaxDim_ ); SetValue( dvMat_, D_ZERO );
    }
    else if( mixType_ == "broyden"){
      GdfMat_.Resize( ntot, mixMaxDim_ ); SetValue( GdfMat_, Z_ZERO );
      GdvMat_.Resize( ntot, mixMaxDim_ ); SetValue( GdvMat_, Z_ZERO );
      GcdfMat_.Resize( ntot, 2 ); SetValue( GcdfMat_, Z_ZERO );
    }
    else{
       ErrorHandling("Invalid mixing type.");
    }
  }


  return ;
}         // -----  end of method SCF::Update  ----- 



void
SCF::Iterate (  )
{
  int mpirank;  MPI_Comm_rank(eigSolPtr_->FFT().domain.comm, &mpirank);
  int mpisize;  MPI_Comm_size(eigSolPtr_->FFT().domain.comm, &mpisize);

  Real timeSta, timeEnd;
  Real timeSta1, timeEnd1;
  Real timeIterStart(0), timeIterEnd(0);

  // Only works for KohnSham class
  Hamiltonian& ham = eigSolPtr_->Ham();
  Fourier&     fft = eigSolPtr_->FFT();
  Spinor&      psi = eigSolPtr_->Psi();

  // EXX: Only allow hybrid functional here

  // Compute the exchange-correlation potential and energy
  if( isCalculateGradRho_ ){
    ham.CalculateGradDensity( fft );
  }

  // Compute the Hartree energy
  ham.CalculateXC( Exc_, fft ); 
  ham.CalculateHartree( fft );

  // Compute the total potential
  ham.CalculateVtot( ham.Vtot() );

  // FIXME The following treatment of the initial density is not
  // compatible with the density extrapolation step in MD
  if(0){
    if( esdfParam.isRestartDensity ){ 
      ham.CalculateXC( Exc_, fft ); 
      // Compute the Hartree energy
      ham.CalculateHartree( fft );
      // No external potential

      // Compute the total potential
      ham.CalculateVtot( ham.Vtot() );
    }
    else{
      // Technically needed, otherwise the initial Vtot will be zero 
      // (density = sum of pseudocharge). 
      // Note that the treatment will be different if the initial
      // density is taken from linear superposition of atomic orbitals
      // 
      // In the future this might need to be changed to something else
      // (see more from QE, VASP and QBox)?
      SetValue(ham.Vtot(), 1.0 );
      statusOFS << "Density may be negative, " << 
        "Skip the calculation of XC for the initial setup. " << std::endl;
    }
  }

#ifdef GPU
    cuda_init_vtot();
    cublas::Init();
#endif
  // Perform non-hybrid functional calculation first
  if( !ham.IsHybrid() || !ham.IsEXXActive()){
    std::ostringstream msg;
    msg << "Starting regular SCF iteration.";
    PrintBlock( statusOFS, msg.str() );
    bool isSCFConverged = false;

    if( !ham.IsEXXActive() && ham.IsHybrid() ) {
      ham.Setup_XC( "XC_GGA_XC_PBE");

      statusOFS << "Re-calculate XC " << std::endl;
      if( isCalculateGradRho_ ){
        ham.CalculateGradDensity( fft );
      }
      ham.CalculateXC( Exc_, fft ); 
      ham.CalculateHartree( fft );
      // Compute the total potential
      ham.CalculateVtot( ham.Vtot() );
    }

    for (Int iter=1; iter <= scfMaxIter_; iter++) {
      if ( isSCFConverged ) break;
      // *********************************************************************
      // Performing each iteartion
      // *********************************************************************
      {
        std::ostringstream msg;
        msg << "SCF iteration # " << iter;
        PrintBlock( statusOFS, msg.str() );
      }

      GetTime( timeIterStart );

      // Solve eigenvalue problem
      // Update density, gradDensity, potential (stored in vtotNew_)
      InnerSolve( iter );

      if( mixVariable_ == "potential" ){
        Real normVtotDif = 0.0, normVtotOld = 0.0;
        DblNumVec& vtotOld_ = ham.Vtot();
        Int ntot = vtotOld_.m();

        for( Int i = 0; i < ntot; i++ ){
          normVtotDif += pow( vtotOld_(i) - vtotNew_(i), 2.0 );
          normVtotOld += pow( vtotOld_(i), 2.0 );
        }

        normVtotDif = sqrt( normVtotDif );
        normVtotOld = sqrt( normVtotOld );
        scfNorm_    = normVtotDif / normVtotOld;
      }


      GetTime( timeSta );
      CalculateEnergy();
      GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
      statusOFS << "Time for computing energy in PWDFT is " <<
        timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

      PrintState( iter );
      
      Int numAtom = ham.AtomList().size();
      efreeDifPerAtom_ = std::abs(Efree_ - EfreeHarris_) / numAtom;

      Print(statusOFS, "norm(out-in)/norm(in) = ", scfNorm_ );
      Print(statusOFS, "Efree diff per atom   = ", efreeDifPerAtom_ ); 

      if( scfNorm_ < scfTolerance_ ){
        /* converged */
        statusOFS << "SCF is converged in " << iter << " steps !" << std::endl;
        isSCFConverged = true;
      }

      if ( mixVariable_ == "density" ){
        ham.Vtot() = vtotNew_;
      }

      // Potential mixing
      GetTime( timeSta );
    if( !isSCFConverged && mixVariable_ == "potential" ){
        DblNumVec& vtotOld_ = ham.Vtot();
      if( mixType_ == "anderson" || mixType_ == "kerker+anderson" ){
        AndersonMix(
            iter,
            mixStepLength_,
            mixType_,
            ham.Vtot(),
            vtotOld_,
            vtotNew_,
            dfMat_,
            dvMat_);
      }
      else if( mixType_ == "broyden"){
        BroydenMix(
            iter,
            mixStepLength_,
            mixType_,
            ham.Vtot(),
            vtotOld_,
            vtotNew_,
            GdfMat_,
            GdvMat_,
            GcdfMat_);
      }   
      else{
        ErrorHandling("Invalid mixing type.");
      }
    }  // ---- end of if( !isSCFConverged && mixVariable_ == "potential" ) ----
    GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
      statusOFS << "Time for computing potential mixing in PWDFT is " <<
        timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

      GetTime( timeIterEnd );

      statusOFS << "Total wall clock time for this SCF iteration = " << timeIterEnd - timeIterStart
        << " [s]" << std::endl;

    } // for (iter)
  }

  // NOTE: The different mixing mode of hybrid functional calculations
  // are not compatible with each other. So each requires its own code
  if( ham.IsHybrid() ){
  {

    if( ham.IsEXXActive() == false ) ham.SetEXXActive(true);

    if(esdfParam.XCType == "XC_HYB_GGA_XC_HSE06")
    {
      ham.Setup_XC( "XC_HYB_GGA_XC_HSE06");
    }
    else if (esdfParam.XCType == "XC_HYB_GGA_XC_PBEH")
    {
      ham.Setup_XC( "XC_HYB_GGA_XC_PBEH");
    }

    statusOFS << "Re-calculate XC " << std::endl;

    if( isCalculateGradRho_ ){
      ham.CalculateGradDensity( fft );
    }

    ham.CalculateXC( Exc_, fft );
    ham.CalculateHartree( fft );
      // Compute the total potential
    ham.CalculateVtot( ham.Vtot() );
  }

  // Fock energies
  Real fock0 = 0.0, fock1 = 0.0, fock2 = 0.0;

  // EXX: Run SCF::Iterate here
  bool isPhiIterConverged = false;

  // FIXME
  bool isFixColumnDF = false;
  Real timePhiIterStart(0), timePhiIterEnd(0);
  Real dExx;
  
  // Evaluate the Fock energy
  // Update Phi <- Psi
  GetTime( timeSta );
  ham.SetPhiEXX( psi, fft );

  // Update the ACE if needed
  if( esdfParam.isHybridACE ){
    if( esdfParam.isHybridDF ){

#ifdef GPU
      ham.CalculateVexxACEDF_GPU( psi, fft, isFixColumnDF );
#else
      ham.CalculateVexxACEDF( psi, fft, isFixColumnDF );
#endif
      // Fix the column after the first iteraiton
      isFixColumnDF = true;
    }
    else{
#ifdef GPU
      ham.CalculateVexxACE_GPU ( psi, fft );
#else
      ham.CalculateVexxACE ( psi, fft );
#endif
    }
  }

  GetTime( timeEnd );
  statusOFS << "Time for updating Phi related variable is " <<
    timeEnd - timeSta << " [s]" << std::endl << std::endl;

  GetTime( timeSta );
  fock2 = ham.CalculateEXXEnergy( psi, fft );
  GetTime( timeEnd );
  statusOFS << "Time for computing the EXX energy is " <<
    timeEnd - timeSta << " [s]" << std::endl << std::endl;

  Efock_ = fock2;
  fock1  = fock2;
  
  if( esdfParam.hybridMixType == "nested" ){
    GetTime( timeSta1 );

    for( Int phiIter = 1; phiIter <= scfPhiMaxIter_; phiIter++ ){

      GetTime( timePhiIterStart );

      std::ostringstream msg;
      msg << "Phi iteration # " << phiIter;
      PrintBlock( statusOFS, msg.str() );

      // Nested SCF iteration
      bool isSCFConverged = false;
      for (Int iter=1; iter <= scfMaxIter_; iter++) {
        if ( isSCFConverged ) break;
        // *********************************************************************
        // Performing each iteartion
        // *********************************************************************
        {
          std::ostringstream msg;
          msg << "SCF iteration # " << iter;
          PrintBlock( statusOFS, msg.str() );
        }

        GetTime( timeIterStart );

        // Solve eigenvalue problem
        // Update density, gradDensity, potential (stored in vtotNew_)
        InnerSolve( iter );

         // Use potential as convergence criteria when potential is mixed
        if( mixVariable_ == "potential" ){
          Real normVtotDif = 0.0, normVtotOld = 0.0;
          DblNumVec& vtotOld_ = ham.Vtot();
          Int ntot = vtotOld_.m();
          for( Int i = 0; i < ntot; i++ ){
            normVtotDif += pow( vtotOld_(i) - vtotNew_(i), 2.0 );
            normVtotOld += pow( vtotOld_(i), 2.0 );
          }
          normVtotDif = sqrt( normVtotDif );
          normVtotOld = sqrt( normVtotOld );
          scfNorm_    = normVtotDif / normVtotOld;
        }

        GetTime( timeSta );
        CalculateEnergy();
        GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
        statusOFS << "Time for computing energy in PWDFT is " <<
          timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

        PrintState( iter );
          
        Int numAtom = ham.AtomList().size();
        efreeDifPerAtom_ = std::abs(Efree_ - EfreeHarris_) / numAtom;

        Print(statusOFS, "norm(out-in)/norm(in) = ", scfNorm_ );
        Print(statusOFS, "Efree diff per atom   = ", efreeDifPerAtom_ ); 


        if( scfNorm_ < scfTolerance_ ){
            /* converged */
          statusOFS << "SCF is converged in " << iter << " steps !" << std::endl;
          isSCFConverged = true;
        }

        if ( mixVariable_ == "density" ){
          ham.Vtot() = vtotNew_;
        }

        // Potential mixing
        GetTime( timeSta );
        if( !isSCFConverged && mixVariable_ == "potential" ){
            DblNumVec& vtotOld_ = ham.Vtot();
          if( mixType_ == "anderson" || mixType_ == "kerker+anderson" ){
            AndersonMix(
                iter,
                mixStepLength_,
                mixType_,
                ham.Vtot(),
                vtotOld_,
                vtotNew_,
                dfMat_,
                dvMat_);
          }
          else if( mixType_ == "broyden"){
            BroydenMix(
            	iter,
            	mixStepLength_,
            	mixType_,
            	ham.Vtot(),
            	vtotOld_,
            	vtotNew_,
            	GdfMat_,
            	GdvMat_,
            	GcdfMat_);
      	  }
          else{
            ErrorHandling("Invalid mixing type.");
          }
        } // ---- end of if( !isSCFConverged && mixVariable_ == "potential" ) ---- 
          GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
          statusOFS << "Time for computing potential mixing in PWDFT is " <<
            timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

          GetTime( timeIterEnd );

          statusOFS << "Total wall clock time for this SCF iteration = " << timeIterEnd - timeIterStart
            << " [s]" << std::endl;

        } // for (iter)

        GetTime( timePhiIterEnd );

        statusOFS << "Total wall clock time for this Phi iteration = " << 
          timePhiIterEnd - timePhiIterStart << " [s]" << std::endl;
      
        fock1 = ham.CalculateEXXEnergy( psi, fft ); 
        // Update Phi <- Psi
        GetTime( timeSta );
        ham.SetPhiEXX( psi, fft ); 

        // Update the ACE if needed
        if( esdfParam.isHybridACE ){
          if( esdfParam.isHybridDF ){
            ham.CalculateVexxACEDF( psi, fft, isFixColumnDF );
            // Fix the column after the first iteraiton
            isFixColumnDF = true;
          }
          else{
            ham.CalculateVexxACE ( psi, fft );
          }
        }

        GetTime( timeEnd );
        statusOFS << "Time for updating Phi related variable is " <<
          timeEnd - timeSta << " [s]" << std::endl << std::endl;

        GetTime( timeSta );
        fock2 = ham.CalculateEXXEnergy( psi, fft ); 

        // Note: initially fock1 = 0.0. So it should at least run for 1 iteration.
        dExx = 4.0 * std::abs( fock1 - 0.5 * (fock0 + fock2) );

        fock0 = fock2;
        Efock_ = fock2;

        Etot_ = Etot_ - 2.0 * fock1 + Efock_;  
        Efree_ = Efree_ - 2.0 * fock1 + Efock_;  

        statusOFS << std::endl;
        Print(statusOFS, "Fock energy       = ",  Efock_, "[au]");
        Print(statusOFS, "Etot(with fock)   = ",  Etot_, "[au]");
        Print(statusOFS, "Efree(with fock)  = ",  Efree_, "[au]");
        Print(statusOFS, "dExx              = ",  dExx, "[au]");
        if( dExx < scfPhiTolerance_ ){
          statusOFS << "SCF for hybrid functional is converged in " 
            << phiIter << " steps !" << std::endl;
          isPhiIterConverged = true;
        }
        if ( isPhiIterConverged ) break;
      } // for(phiIter)

      GetTime( timeEnd1 );
      statusOFS << "Time for using nested method is " <<
        timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
  }  // hybridMixType == "nested"

#ifndef _COMPLEX_ 

    // New method for the commutator-DIIS with column selection strategy
    if( esdfParam.hybridMixType == "scdiis" ){
      GetTime( timeSta1 );
      // Computing selected columns of the density matrix. 
      // This requires a good initial guess of wavefunctions, 
      // from one of the following
      // 1) a regular SCF calculation
      // 2) restarting wavefunction with Hybrid_Active_Init = true

      if( mpisize > 1 )
        ErrorHandling("scdiis only works for mpisize == 1.");

      Int ntot      = fft.domain.NumGridTotal();
      Int ntotFine  = fft.domain.NumGridTotalFine();
      Int numStateTotal = psi.NumStateTotal();
      Int numOcc = ham.NumOccupiedState();
      
      IntNumVec permPsiOcc(numOcc);

      // For residual
      dfMat_.Resize( ntot * numOcc, mixMaxDim_ ); SetValue( dfMat_, 0.0 );
      // For Pc
      dvMat_.Resize( ntot * numOcc, mixMaxDim_ ); SetValue( dvMat_, 0.0 );

      // Selected columns of density matrix
      DblNumMat Pc(ntot, numOcc);
      DblNumMat Res(ntot, numOcc);
      
      // Compute all Hpsi for simplicity
      DblNumMat psiPc(ntot, numStateTotal);
      DblNumMat Hpsi(ntot, numStateTotal);
      DblNumMat psiMuT(numOcc, numOcc);
      DblNumMat HpsiMuT(numOcc, numOcc);

      {
        // This currently only works for insulating system
        IntNumVec permPsi(ntot);
        DblNumVec tau(ntot);

        // Quick and dirty way to generate the transpose matrix
        DblNumMat psiOccT;
        Transpose( DblNumMat(ntot, numOcc, false, psi.Wavefun().Data()), psiOccT );

        // Important since if permPsi is not zero, it will try to use the number as 
        // fixed columns
        SetValue( permPsi, 0 ); 
        lapack::QRCP( numOcc, ntot, psiOccT.Data(), numOcc, 
            permPsi.Data(), tau.Data() );

        for( Int mu = 0; mu < numOcc; mu++ ){
          permPsiOcc[mu] = permPsi[mu];
        }
      }

      for( Int phiIter = 1; phiIter <= scfPhiMaxIter_; phiIter++ ){

        GetTime( timePhiIterStart );

        for( Int mu = 0; mu < numOcc; mu++ ){
          for( Int k = 0; k < numOcc; k++ ){
            psiMuT(k,mu) = psi.Wavefun()(permPsiOcc[mu],0,k);
          }
        }

#if ( _DEBUGlevel_ >= 1 )
        {
          // Monitor the singular values as a measure as the quality of
          // the selected columns
          DblNumMat tmp = psiMuT;
          DblNumVec s(numOcc);
          lapack::SingularValues( numOcc, numOcc, tmp.Data(), numOcc, s.Data() );
          statusOFS << "Spsi = " << s << std::endl;
        }
#endif

        blas::Gemm( 'N', 'N', ntot, numOcc, numOcc, 1.0, 
            psi.Wavefun().Data(), ntot, psiMuT.Data(), numOcc, 
            0.0, Pc.Data(), ntot );

        std::ostringstream msg;
        msg << "Phi iteration # " << phiIter;
        PrintBlock( statusOFS, msg.str() );

        // Compute the residual
        {
          // Compute Hpsi for all psi 
          NumTns<Real> tnsTemp(ntot, 1, numStateTotal, false, 
              Hpsi.Data());
          ham.MultSpinor( psi, tnsTemp, fft );
          for( Int mu = 0; mu < numOcc; mu++ ){
            for( Int k = 0; k < numOcc; k++ ){
              HpsiMuT(k,mu) = Hpsi(permPsiOcc[mu],k);
            }
          }
          blas::Gemm( 'N', 'N', ntot, numOcc, numOcc, 1.0, 
              Hpsi.Data(), ntot, psiMuT.Data(), numOcc, 
              0.0, Res.Data(), ntot );
          blas::Gemm( 'N', 'N', ntot, numOcc, numOcc, -1.0, 
              psi.Wavefun().Data(), ntot, HpsiMuT.Data(), numOcc, 
              1.0, Res.Data(), ntot );
        }

        // Anderson mixing. Use the same mixMaxDim_ for Phi mixing
        {
          // Optimal input potential in Anderon mixing.
          DblNumVec vOpt( ntot * numOcc ); 

          // Number of iterations used, iter should start from 1
          Int iterused = std::min( phiIter-1, mixMaxDim_ ); 
          // The current position of dfMat, dvMat
          Int ipos = phiIter - 1 - ((phiIter-2)/ mixMaxDim_ ) * mixMaxDim_;
          // The next position of dfMat, dvMat
          Int inext = phiIter - ((phiIter-1)/ mixMaxDim_) * mixMaxDim_;

          blas::Copy( ntot * numOcc, Pc.Data(), 1, vOpt.Data(), 1 );

          if( phiIter > 1 ){
            // dfMat(:, ipos-1) = res(:) - dfMat(:, ipos-1);
            // dvMat(:, ipos-1) = vOld(:) - dvMat(:, ipos-1);
            blas::Scal( ntot * numOcc, -1.0, dfMat_.VecData(ipos-1), 1 );
            blas::Axpy( ntot * numOcc, 1.0, Res.Data(), 1, dfMat_.VecData(ipos-1), 1 );
            blas::Scal( ntot * numOcc, -1.0, dvMat_.VecData(ipos-1), 1 );
            blas::Axpy( ntot * numOcc, 1.0, Pc.Data(), 1, dvMat_.VecData(ipos-1), 1 );

            // Calculating pseudoinverse
            DblNumMat dfMatTemp;
            DblNumVec gammas(ntot * numOcc), S(iterused);

            Int rank;
            // FIXME Magic number
            Real rcond = 1e-6;

            // gammas    = res;
            blas::Copy( ntot * numOcc, Res.Data(), 1, gammas.Data(), 1 );
            dfMatTemp = dfMat_;

            // May need different strategy in a parallel setup
            lapack::SVDLeastSquare( ntot * numOcc, iterused, 1, 
                dfMatTemp.Data(), ntot * numOcc,
                gammas.Data(), ntot * numOcc,
                S.Data(), rcond, &rank );

            Print( statusOFS, "  Rank of dfmat = ", rank );

            // Update vOpt only, which is the mixed Pc

            blas::Gemv('N', ntot * numOcc, iterused, -1.0, dvMat_.Data(),
                ntot * numOcc, gammas.Data(), 1, 1.0, vOpt.Data(), 1 );
          }

          // Update dfMat, dvMat, vMix 
          // dfMat(:, inext-1) = Res(:)
          // dvMat(:, inext-1) = Pc(:)
          blas::Copy( ntot * numOcc, Res.Data(), 1, 
              dfMat_.VecData(inext-1), 1 );
          blas::Copy( ntot * numOcc, Pc.Data(),  1, 
              dvMat_.VecData(inext-1), 1 );

          // Orthogonalize vOpt to obtain psiPc. 
          // psiPc has the same size
          SetValue( psiPc, 0.0 );
          blas::Copy( ntot * numOcc, vOpt.Data(), 1, psiPc.Data(), 1 );
          lapack::Orth( ntot, numOcc, psiPc.Data(), ntot );
        }

        // Construct the new Hamiltonian operator
        Spinor spnPsiPc(fft.domain, 1, numStateTotal,
            numStateTotal, false, psiPc.Data());

        // Compute the electron density
        GetTime( timeSta );
#ifdef GPU //-----------------modified by lijl 20200521
        ham.CalculateDensity_GPU(
            spnPsiPc,
            ham.OccupationRate(),
            totalCharge_,
            fft );
#else
        ham.CalculateDensity(
            spnPsiPc,
            ham.OccupationRate(),
            totalCharge_, 
            fft );
#endif
        GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
        statusOFS << "Time for computing density in PWDFT is " <<
          timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

        // Compute the exchange-correlation potential and energy
        if( isCalculateGradRho_ ){
          GetTime( timeSta );
          ham.CalculateGradDensity( fft );
          GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
          statusOFS << "Time for computing gradient density in PWDFT is " <<
            timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
        }

        GetTime( timeSta );
        ham.CalculateXC( Exc_, fft ); 
        GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
        statusOFS << "Time for computing XC potential in PWDFT is " <<
          timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

        // Compute the Hartree energy
        GetTime( timeSta );
        ham.CalculateHartree( fft );
        GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
        statusOFS << "Time for computing Hartree potential in PWDFT is " <<
          timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
        // No external potential

        // Compute the total potential
        GetTime( timeSta );
        ham.CalculateVtot( vtotNew_ );
        blas::Copy( ntotFine, vtotNew_.Data(), 1, ham.Vtot().Data(), 1 );

        GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
        statusOFS << "Time for computing total potential in PWDFT is " <<
          timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

        // Update Phi <- Psi
        GetTime( timeSta );
        ham.SetPhiEXX( spnPsiPc, fft ); 

        // Update the ACE if needed
        // Still use psi but phi has changed
        if( esdfParam.isHybridACE ){
          if( esdfParam.isHybridDF ){
            ham.CalculateVexxACEDF( psi, fft, isFixColumnDF );
            // Fix the column after the first iteraiton
            isFixColumnDF = true;
          }
          else{
            ham.CalculateVexxACE ( psi, fft );
          }
        }

        GetTime( timeEnd );
        statusOFS << "Time for updating Phi related variable is " <<
          timeEnd - timeSta << " [s]" << std::endl << std::endl;

        InnerSolve( phiIter );
        blas::Copy( ntotFine, vtotNew_.Data(), 1, ham.Vtot().Data(), 1 );

        CalculateEnergy();

        PrintState( phiIter );

        GetTime( timePhiIterEnd );

        statusOFS << "Total wall clock time for this Phi iteration = " << 
          timePhiIterEnd - timePhiIterStart << " [s]" << std::endl;


        // Update Phi <- Psi
        GetTime( timeSta );
        ham.SetPhiEXX( psi, fft ); 

        // In principle there is no need to construct ACE operator here
        // However, this makes the code more readable by directly calling 
        // the MultSpinor function later
        if( esdfParam.isHybridACE ){
          if( esdfParam.isHybridDF ){
            ham.CalculateVexxACEDF( psi, fft, isFixColumnDF );
            // Fix the column after the first iteraiton
            isFixColumnDF = true;
          }
          else{
            ham.CalculateVexxACE ( psi, fft );
          }
        }

        GetTime( timeEnd );
        statusOFS << "Time for updating Phi related variable is " <<
          timeEnd - timeSta << " [s]" << std::endl << std::endl;

        GetTime( timeSta );
        fock2 = ham.CalculateEXXEnergy( psi, fft ); 
        GetTime( timeEnd );
        statusOFS << "Time for computing the EXX energy is " <<
          timeEnd - timeSta << " [s]" << std::endl << std::endl;

        // Note: initially fock1 = 0.0. So it should at least run for 1 iteration.
        dExx = std::abs(fock2 - fock1) / std::abs(fock2);
        // use scfNorm to reflect dExx
        scfNorm_ = dExx;
        fock1 = fock2;
        Efock_ = fock2;

        Etot_ = Etot_ - Efock_;
        Efree_ = Efree_ - Efock_;

        statusOFS << std::endl;
        Print(statusOFS, "Fock energy       = ",  Efock_, "[au]");
        Print(statusOFS, "Etot(with fock)   = ",  Etot_, "[au]");
        Print(statusOFS, "Efree(with fock)  = ",  Efree_, "[au]");
        Print(statusOFS, "dExx              = ",  dExx, "[au]");

        if( dExx < scfPhiTolerance_ ){
          statusOFS << "SCF for hybrid functional is converged in " 
            << phiIter << " steps !" << std::endl;
          isPhiIterConverged = true;
        }
        if ( isPhiIterConverged ) break;
      } // for(phiIter)

      GetTime( timeEnd1 );
      statusOFS << "Time for using scdiis method is " <<
        timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
    } // hybridMixType == "scdiis"
    
#ifdef GPU
    // GPU version of pc-diis. 
    // note, ACE nested and scdiis can not work on GPU.
    if( esdfParam.hybridMixType == "pcdiis" ){

     GetTime( timeSta1 );
      // This requires a good initial guess of wavefunctions, 
      // from one of the following
      // 1) a regular SCF calculation
      // 2) restarting wavefunction with Hybrid_Active_Init = true
        
      Int ntot      = fft.domain.NumGridTotal();
      Int ntotFine  = fft.domain.NumGridTotalFine();
      Int numStateTotal = psi.NumStateTotal();
      Int numStateLocal = psi.NumState();
      Int numOccTotal = ham.NumOccupiedState();

      MPI_Comm mpi_comm = eigSolPtr_->FFT().domain.comm;

      Int I_ONE = 1, I_ZERO = 0;
      double D_ONE = 1.0;
      double D_ZERO = 0.0;
      double D_MinusONE = -1.0;

      Real timeSta, timeEnd, timeSta1, timeEnd1;
      
      Int contxt0D, contxt1DCol, contxt1DRow,  contxt2D;
      Int nprow0D, npcol0D, myrow0D, mycol0D, info0D;
      Int nprow1DCol, npcol1DCol, myrow1DCol, mycol1DCol, info1DCol;
      Int nprow1DRow, npcol1DRow, myrow1DRow, mycol1DRow, info1DRow;
      Int nprow2D, npcol2D, myrow2D, mycol2D, info2D;

      Int ncolsNgNe1DCol, nrowsNgNe1DCol, lldNgNe1DCol; 
      Int ncolsNgNe1DRow, nrowsNgNe1DRow, lldNgNe1DRow; 
      Int ncolsNgNo1DCol, nrowsNgNo1DCol, lldNgNo1DCol; 
      Int ncolsNgNo1DRow, nrowsNgNo1DRow, lldNgNo1DRow; 

      Int desc_NgNe1DCol[9];
      Int desc_NgNe1DRow[9];
      Int desc_NgNo1DCol[9];
      Int desc_NgNo1DRow[9];

      Int Ne = numStateTotal; 
      Int No = numOccTotal; 
      Int Ng = ntot;

      // 1D col MPI
      nprow1DCol = 1;
      npcol1DCol = mpisize;

      Cblacs_get(0, 0, &contxt1DCol);
      Cblacs_gridinit(&contxt1DCol, "C", nprow1DCol, npcol1DCol);
      Cblacs_gridinfo(contxt1DCol, &nprow1DCol, &npcol1DCol, &myrow1DCol, &mycol1DCol);

      // 1D row MPI
      nprow1DRow = mpisize;
      npcol1DRow = 1;

      Cblacs_get(0, 0, &contxt1DRow);
      Cblacs_gridinit(&contxt1DRow, "C", nprow1DRow, npcol1DRow);
      Cblacs_gridinfo(contxt1DRow, &nprow1DRow, &npcol1DRow, &myrow1DRow, &mycol1DRow);


      //desc_NgNe1DCol
      if(contxt1DCol >= 0){
        nrowsNgNe1DCol = SCALAPACK(numroc)(&Ng, &Ng, &myrow1DCol, &I_ZERO, &nprow1DCol);
        ncolsNgNe1DCol = SCALAPACK(numroc)(&Ne, &I_ONE, &mycol1DCol, &I_ZERO, &npcol1DCol);
        lldNgNe1DCol = std::max( nrowsNgNe1DCol, 1 );
      }    

      SCALAPACK(descinit)(desc_NgNe1DCol, &Ng, &Ne, &Ng, &I_ONE, &I_ZERO, 
          &I_ZERO, &contxt1DCol, &lldNgNe1DCol, &info1DCol);

      //desc_NgNe1DRow
      if(contxt1DRow >= 0){
        nrowsNgNe1DRow = SCALAPACK(numroc)(&Ng, &BlockSizeScaLAPACK_, &myrow1DRow, &I_ZERO, &nprow1DRow);
        ncolsNgNe1DRow = SCALAPACK(numroc)(&Ne, &Ne, &mycol1DRow, &I_ZERO, &npcol1DRow);
        lldNgNe1DRow = std::max( nrowsNgNe1DRow, 1 );
      }    

      SCALAPACK(descinit)(desc_NgNe1DRow, &Ng, &Ne, &BlockSizeScaLAPACK_, &Ne, &I_ZERO, 
          &I_ZERO, &contxt1DRow, &lldNgNe1DRow, &info1DRow);

      //desc_NgNo1DCol
      if(contxt1DCol >= 0){
        nrowsNgNo1DCol = SCALAPACK(numroc)(&Ng, &Ng, &myrow1DCol, &I_ZERO, &nprow1DCol);
        ncolsNgNo1DCol = SCALAPACK(numroc)(&No, &I_ONE, &mycol1DCol, &I_ZERO, &npcol1DCol);
        lldNgNo1DCol = std::max( nrowsNgNo1DCol, 1 );
      }    

      SCALAPACK(descinit)(desc_NgNo1DCol, &Ng, &No, &Ng, &I_ONE, &I_ZERO, 
          &I_ZERO, &contxt1DCol, &lldNgNo1DCol, &info1DCol);

      //desc_NgNo1DRow
      if(contxt1DRow >= 0){
        nrowsNgNo1DRow = SCALAPACK(numroc)(&Ng, &BlockSizeScaLAPACK_, &myrow1DRow, &I_ZERO, &nprow1DRow);
        ncolsNgNo1DRow = SCALAPACK(numroc)(&No, &No, &mycol1DRow, &I_ZERO, &npcol1DRow);
        lldNgNo1DRow = std::max( nrowsNgNo1DRow, 1 );
      }    

      SCALAPACK(descinit)(desc_NgNo1DRow, &Ng, &No, &BlockSizeScaLAPACK_, &No, &I_ZERO, 
          &I_ZERO, &contxt1DRow, &lldNgNo1DRow, &info1DRow);
      
      if(numStateLocal !=  ncolsNgNe1DCol){
        statusOFS << "numStateLocal = " << numStateLocal << " ncolsNgNe1DCol = " << ncolsNgNe1DCol << std::endl;
        ErrorHandling("The size of numState is not right!");
      }
      
      if(nrowsNgNe1DRow !=  nrowsNgNo1DRow){
        statusOFS << "nrowsNgNe1DRow = " << nrowsNgNe1DRow << " ncolsNgNo1DRow = " << ncolsNgNo1DRow << std::endl;
        ErrorHandling("The size of nrowsNgNe1DRow and ncolsNgNo1DRow is not right!");
      }


      Int numOccLocal = ncolsNgNo1DCol;
      Int ntotLocal = nrowsNgNe1DRow;

      DblNumMat psiPcCol(ntot, numStateLocal);
      DblNumMat psiPcRow(ntotLocal, numStateTotal);
      DblNumMat HpsiCol(ntot, numStateLocal);
      DblNumMat HpsiRow(ntotLocal, numStateTotal);

      dfMat_.Resize( ntot * numOccLocal, mixMaxDim_ ); SetValue( dfMat_, 0.0 );
      dvMat_.Resize( ntot * numOccLocal, mixMaxDim_ ); SetValue( dvMat_, 0.0 );

      DblNumMat PcCol(ntot, numOccLocal);
      DblNumMat PcRow(ntotLocal, numOccTotal);
      DblNumMat ResCol(ntot, numOccLocal);
      DblNumMat ResRow(ntotLocal, numOccTotal);

      DblNumMat psiMuT(numOccTotal, numOccTotal);
      DblNumMat psiMuTLocal(numOccLocal, numOccTotal);
      DblNumMat HpsiMuT(numOccTotal, numOccTotal);
      DblNumMat HpsiMuTLocal(numOccLocal, numOccTotal);

      DblNumMat psiCol( ntot, numStateLocal );
      SetValue( psiCol, 0.0 );
      DblNumMat psiRow( ntotLocal, numStateTotal );
      SetValue( psiRow, 0.0 );

      lapack::Lacpy( 'A', ntot, numStateLocal, psi.Wavefun().Data(), ntot, psiCol.Data(), ntot );
      //AlltoallForward (psiCol, psiRow, mpi_comm);
      SCALAPACK(pdgemr2d)(&Ng, &Ne, psiCol.Data(), &I_ONE, &I_ONE, desc_NgNe1DCol, 
          psiRow.Data(), &I_ONE, &I_ONE, desc_NgNe1DRow, &contxt1DCol );

      DblNumMat psiTemp(ntotLocal, numOccTotal);
      SetValue( psiTemp, 0.0 );

      lapack::Lacpy( 'A', ntotLocal, numOccTotal, psiRow.Data(), ntotLocal, psiTemp.Data(), ntotLocal );
      Real one = 1.0;
      Real minus_one = -1.0;
      Real zero = 0.0;

     
#ifdef GPU
    //cuda_init_vtot();
#endif
      // Phi loop
      for( Int phiIter = 1; phiIter <= scfPhiMaxIter_; phiIter++ ){
      {
      cuDblNumMat cu_psiMuT(numOccTotal, numOccTotal);
      cuDblNumMat cu_HpsiMuT(numOccTotal, numOccTotal);
      cuDblNumMat cu_psiRow( ntotLocal, numStateTotal );
      cuDblNumMat cu_psiTemp(ntotLocal, numOccTotal);
      cuDblNumMat cu_PcRow(ntotLocal, numOccTotal);
      cuDblNumMat cu_HpsiCol(ntot, numStateLocal);
      cuDblNumMat cu_psi(ntot, numStateLocal);
      cuDblNumMat cu_HpsiRow(ntotLocal, numStateTotal);
      cuDblNumMat cu_ResRow(ntotLocal, numOccTotal);
      cuDblNumMat cu_psiPcRow(ntotLocal, numStateTotal);
      cu_psiTemp.CopyFrom( psiTemp);


        GetTime( timePhiIterStart );
          
        SetValue( psiCol, 0.0 );
        lapack::Lacpy( 'A', ntot, numStateLocal, psi.Wavefun().Data(), ntot, psiCol.Data(), ntot );
        SetValue( psiRow, 0.0 );
        //AlltoallForward (psiCol, psiRow, mpi_comm);
        SCALAPACK(pdgemr2d)(&Ng, &Ne, psiCol.Data(), &I_ONE, &I_ONE, desc_NgNe1DCol, 
            psiRow.Data(), &I_ONE, &I_ONE, desc_NgNe1DRow, &contxt1DCol );

        if(1){

          DblNumMat psiMuTTemp(numOccTotal, numOccTotal);
          SetValue( psiMuTTemp, 0.0 );

          cuDblNumMat cu_psiMuTTemp(numOccTotal, numOccTotal);
          cu_psiRow.CopyFrom(psiRow);
          cublas::Gemm( HIPBLAS_OP_T, HIPBLAS_OP_N, numOccTotal, numOccTotal, ntotLocal, 
              &one, cu_psiRow.Data(), ntotLocal, cu_psiTemp.Data(), ntotLocal, 
              &zero, cu_psiMuTTemp.Data(), numOccTotal );
          cu_psiMuTTemp.CopyTo( psiMuTTemp );

          /*
          blas::Gemm( 'T', 'N', numOccTotal, numOccTotal, ntotLocal, 1.0, 
              psiRow.Data(), ntotLocal, psiTemp.Data(), ntotLocal, 
              0.0, psiMuTTemp.Data(), numOccTotal );
          */
          SetValue( psiMuT, 0.0 );
          MPI_Allreduce( psiMuTTemp.Data(), psiMuT.Data(), 
              numOccTotal * numOccTotal, MPI_DOUBLE, MPI_SUM, mpi_comm );

        }//if

#if ( _DEBUGlevel_ >= 1 )
        {
          // Monitor the singular values as a measure as the quality of
          // the selected columns
          DblNumMat tmp = psiMuT;
          DblNumVec s(numOccTotal);
          lapack::SingularValues( numOccTotal, numOccTotal, tmp.Data(), numOccTotal, s.Data() );
          statusOFS << "Spsi = " << s << std::endl;
        }
#endif
        cu_psiMuT.CopyFrom( psiMuT);
        cublas::Gemm( HIPBLAS_OP_N, HIPBLAS_OP_N, ntotLocal, numOccTotal, numOccTotal, &one, 
            cu_psiRow.Data(), ntotLocal, cu_psiMuT.Data(), numOccTotal, 
            &zero, cu_PcRow.Data(), ntotLocal );
         cu_PcRow.CopyTo( PcRow );
        
        /*
        blas::Gemm( 'N', 'N', ntotLocal, numOccTotal, numOccTotal, 1.0, 
            psiRow.Data(), ntotLocal, psiMuT.Data(), numOccTotal, 
            0.0, PcRow.Data(), ntotLocal );
        */
        SetValue( PcCol, 0.0 );
        //AlltoallBackward (PcRow, PcCol, mpi_comm);
        SCALAPACK(pdgemr2d)(&Ng, &No, PcRow.Data(), &I_ONE, &I_ONE, desc_NgNo1DRow, 
            PcCol.Data(), &I_ONE, &I_ONE, desc_NgNo1DCol, &contxt1DCol );
        
        std::ostringstream msg;
        msg << "Phi iteration # " << phiIter;
        PrintBlock( statusOFS, msg.str() );

        // Compute the residual
        {
          // Compute Hpsi for all psi 
          Int ncom = psi.NumComponent();
          Int noccTotal = psi.NumStateTotal();
          Int noccLocal = psi.NumState();

          cuda_memcpy_CPU2GPU(cu_psi.Data(), psi.Wavefun().Data(), ntot*numStateLocal*sizeof(Real) );
          cuNumTns<Real> tnsTemp(ntot, 1, numStateLocal, false, cu_HpsiCol.Data());
          // there are two sets of grid for Row parallelization. 
          // the old fashioned split and the Scalapack split.
          // note that they turns out to be different ones in that: 
          // Scalapack divide the Psi by blocks, and old way is continuous.  
          // this is error prone.
          Spinor spnTemp(fft.domain, ncom, noccTotal, noccLocal, false, cu_psi.Data(), true);
          ham.MultSpinor_old( spnTemp, tnsTemp, fft );
          cu_HpsiCol.CopyTo(HpsiCol);

          // remember to reset the vtot
          cuda_set_vtot_flag();
          //ham.MultSpinor( psi, tnsTemp, fft );
        
          //SetValue( HpsiRow, 0.0 );
          //AlltoallForward (HpsiCol, HpsiRow, mpi_comm);
          SCALAPACK(pdgemr2d)(&Ng, &Ne, HpsiCol.Data(), &I_ONE, &I_ONE, desc_NgNe1DCol, 
            HpsiRow.Data(), &I_ONE, &I_ONE, desc_NgNe1DRow, &contxt1DCol );

          cu_HpsiRow.CopyFrom(HpsiRow);

          // perform the ACE operator here
          //ham.ACEOperator( cu_psiRow, fft, cu_HpsiRow );
          //cu_HpsiRow.CopyTo(HpsiRow);

          if(1){

            DblNumMat HpsiMuTTemp(numOccTotal,numOccTotal);
            cuDblNumMat cu_HpsiMuTTemp(numOccTotal,numOccTotal);
            //SetValue( HpsiMuTTemp, 0.0 );

            cublas::Gemm( HIPBLAS_OP_T, HIPBLAS_OP_N, numOccTotal, numOccTotal, ntotLocal, &one, 
                cu_HpsiRow.Data(), ntotLocal, cu_psiTemp.Data(), ntotLocal, 
                &zero, cu_HpsiMuTTemp.Data(), numOccTotal );
            cu_HpsiMuTTemp.CopyTo(HpsiMuTTemp);
/*
            blas::Gemm( 'T', 'N', numOccTotal, numOccTotal, ntotLocal, 1.0, 
                HpsiRow.Data(), ntotLocal, psiTemp.Data(), ntotLocal, 
                0.0, HpsiMuTTemp.Data(), numOccTotal );
*/
            //SetValue( HpsiMuT, 0.0 );

            MPI_Allreduce( HpsiMuTTemp.Data(), HpsiMuT.Data(), 
                numOccTotal * numOccTotal, MPI_DOUBLE, MPI_SUM, mpi_comm );

          }//if
          
          cublas::Gemm( HIPBLAS_OP_N, HIPBLAS_OP_N, ntotLocal, numOccTotal, numOccTotal, &one, 
              cu_HpsiRow.Data(), ntotLocal, cu_psiMuT.Data(), numOccTotal, 
              &zero, cu_ResRow.Data(), ntotLocal );

          cu_HpsiMuT.CopyFrom( HpsiMuT );
          cublas::Gemm( HIPBLAS_OP_N, HIPBLAS_OP_N, ntotLocal, numOccTotal, numOccTotal, &minus_one, 
              cu_psiRow.Data(), ntotLocal, cu_HpsiMuT.Data(), numOccTotal, 
              &one, cu_ResRow.Data(), ntotLocal );
          cu_ResRow.CopyTo(ResRow);

          /*
          blas::Gemm( 'N', 'N', ntotLocal, numOccTotal, numOccTotal, 1.0, 
              HpsiRow.Data(), ntotLocal, psiMuT.Data(), numOccTotal, 
              0.0, ResRow.Data(), ntotLocal );
          blas::Gemm( 'N', 'N', ntotLocal, numOccTotal, numOccTotal, -1.0, 
              psiRow.Data(), ntotLocal, HpsiMuT.Data(), numOccTotal, 
              1.0, ResRow.Data(), ntotLocal );
          SetValue( ResCol, 0.0 );
          */
          //AlltoallBackward (ResRow, ResCol, mpi_comm);
          SCALAPACK(pdgemr2d)(&Ng, &No, ResRow.Data(), &I_ONE, &I_ONE, desc_NgNo1DRow, 
              ResCol.Data(), &I_ONE, &I_ONE, desc_NgNo1DCol, &contxt1DCol );
        }
        
        GetTime( timeSta );
        // Anderson mixing. Use the same mixMaxDim_ for Phi mixing
        {
          // Optimal input potential in Anderon mixing.
          DblNumVec vOpt( ntot * numOccLocal ); 

          // Number of iterations used, iter should start from 1
          Int iterused = std::min( phiIter-1, mixMaxDim_ ); 
          // The current position of dfMat, dvMat
          Int ipos = phiIter - 1 - ((phiIter-2)/ mixMaxDim_ ) * mixMaxDim_;
          // The next position of dfMat, dvMat
          Int inext = phiIter - ((phiIter-1)/ mixMaxDim_) * mixMaxDim_;
        
          blas::Copy( ntot * numOccLocal, PcCol.Data(), 1, vOpt.Data(), 1 );

          if( phiIter > 1 ){
            // dfMat(:, ipos-1) = res(:) - dfMat(:, ipos-1);
            // dvMat(:, ipos-1) = vOld(:) - dvMat(:, ipos-1);
            blas::Scal( ntot * numOccLocal, -1.0, dfMat_.VecData(ipos-1), 1 );
            blas::Axpy( ntot * numOccLocal, 1.0, ResCol.Data(), 1, dfMat_.VecData(ipos-1), 1 );
            blas::Scal( ntot * numOccLocal, -1.0, dvMat_.VecData(ipos-1), 1 );
            blas::Axpy( ntot * numOccLocal, 1.0, PcCol.Data(), 1, dvMat_.VecData(ipos-1), 1 );

            // Calculating pseudoinverse
            DblNumMat dfMatTemp(ntot * numOccLocal, mixMaxDim_);
            DblNumVec gammas(ntot * numOccLocal), S(iterused);
            
            SetValue( dfMatTemp, 0.0 );
            SetValue( gammas, 0.0 );

            Int rank;
            // FIXME Magic number
            Real rcond = 1e-9;

            // gammas    = res;
            blas::Copy( ntot * numOccLocal, ResCol.Data(), 1, gammas.Data(), 1 );
            lapack::Lacpy( 'A', ntot * numOccLocal, mixMaxDim_, dfMat_.Data(), ntot * numOccLocal, 
                dfMatTemp.Data(), ntot * numOccLocal );
        
            // May need different strategy in a parallel setup
            if(0){  
              
              lapack::SVDLeastSquare( ntot * numOccLocal, iterused, 1, 
                  dfMatTemp.Data(), ntot * numOccLocal,
                  gammas.Data(), ntot * numOccLocal,
                  S.Data(), rcond, &rank );
            
              blas::Gemv('N', ntot * numOccLocal, iterused, -1.0, dvMat_.Data(),
                  ntot * numOccLocal, gammas.Data(), 1, 1.0, vOpt.Data(), 1 );
            
            }
        
            if(1){

              DblNumMat XTX(iterused, iterused);
              DblNumMat XTXTemp(iterused, iterused);
              
              SetValue( XTXTemp, 0.0 );
              blas::Gemm( 'T', 'N', iterused, iterused, ntot * numOccLocal, 1.0, 
              dfMatTemp.Data(), ntot * numOccLocal, dfMatTemp.Data(), ntot * numOccLocal, 
              0.0, XTXTemp.Data(), iterused );
        
              SetValue( XTX, 0.0 );
              MPI_Allreduce( XTXTemp.Data(), XTX.Data(), 
                  iterused * iterused, MPI_DOUBLE, MPI_SUM, mpi_comm );
            
              DblNumVec gammasTemp1(iterused);
              SetValue( gammasTemp1, 0.0 );
              //blas::Gemv('T', ntot * numOccLocal, iterused, 1.0, dfMatTemp.Data(),
              //    ntot * numOccLocal, gammas.Data(), 1, 0.0, gammasTemp1.Data(), 1 );

              blas::Gemm( 'T', 'N', iterused, I_ONE, ntot * numOccLocal, 1.0, 
                  dfMatTemp.Data(), ntot * numOccLocal, gammas.Data(), ntot * numOccLocal, 
                  0.0, gammasTemp1.Data(), iterused );

              DblNumVec gammasTemp2(iterused);
              SetValue( gammasTemp2, 0.0 );
              MPI_Allreduce( gammasTemp1.Data(), gammasTemp2.Data(), 
                  iterused, MPI_DOUBLE, MPI_SUM, mpi_comm );
              
              lapack::SVDLeastSquare( iterused, iterused, 1, 
                  XTX.Data(), iterused,
                  gammasTemp2.Data(), iterused,
                  S.Data(), rcond, &rank );
            
              //blas::Gemv('N', ntot * numOccLocal, iterused, -1.0, dvMat_.Data(),
              //    ntot * numOccLocal, gammasTemp2.Data(), 1, 1.0, vOpt.Data(), 1 );
              
              blas::Gemm( 'N', 'N', ntot * numOccLocal, I_ONE, iterused, -1.0, 
                  dvMat_.Data(), ntot * numOccLocal, gammasTemp2.Data(), iterused, 
                  1.0, vOpt.Data(), ntot * numOccLocal );
            
            }

            Print( statusOFS, "  Rank of dfmat = ", rank );

          }

          // Update dfMat, dvMat, vMix 
          // dfMat(:, inext-1) = Res(:)
          // dvMat(:, inext-1) = Pc(:)
          blas::Copy( ntot * numOccLocal, ResCol.Data(), 1, 
              dfMat_.VecData(inext-1), 1 );
          blas::Copy( ntot * numOccLocal, PcCol.Data(),  1, 
              dvMat_.VecData(inext-1), 1 );
            
          // Orthogonalize vOpt to obtain psiPc. 
          // psiPc has the same size
          SetValue( psiPcCol, 0.0 );
          blas::Copy( ntot * numOccLocal, vOpt.Data(), 1, psiPcCol.Data(), 1 );
          //lapack::Orth( ntot, numOccLocal, psiPcCol.Data(), ntotLocal );
          
          // Orthogonalization through Cholesky factorization
         if(1){ 
            SetValue( psiPcRow, 0.0 );
            //AlltoallForward (psiPcCol, psiPcRow, mpi_comm);
            SCALAPACK(pdgemr2d)(&Ng, &No, psiPcCol.Data(), &I_ONE, &I_ONE, desc_NgNo1DCol, 
                psiPcRow.Data(), &I_ONE, &I_ONE, desc_NgNo1DRow, &contxt1DCol );

            cu_psiPcRow.CopyFrom( psiPcRow );

            DblNumMat XTX(numOccTotal, numOccTotal);
            DblNumMat XTXTemp(numOccTotal, numOccTotal);
            cuDblNumMat cu_XTXTemp(numOccTotal, numOccTotal);
            
            cublas::Gemm( HIPBLAS_OP_T, HIPBLAS_OP_N, numOccTotal, numOccTotal, ntotLocal, &one, cu_psiPcRow.Data(), 
                ntotLocal, cu_psiPcRow.Data(), ntotLocal, &zero, cu_XTXTemp.Data(), numOccTotal );
            cu_XTXTemp.CopyTo(XTXTemp);

            /*
            blas::Gemm( 'T', 'N', numOccTotal, numOccTotal, ntotLocal, 1.0, psiPcRow.Data(), 
                ntotLocal, psiPcRow.Data(), ntotLocal, 0.0, XTXTemp.Data(), numOccTotal );
            SetValue( XTX, 0.0 );
            */
            MPI_Allreduce(XTXTemp.Data(), XTX.Data(), numOccTotal*numOccTotal, MPI_DOUBLE, MPI_SUM, mpi_comm);

            //if ( mpirank == 0) {
            //  lapack::Potrf( 'U', numOccTotal, XTX.Data(), numOccTotal );
            //}
            //MPI_Bcast(XTX.Data(), numOccTotal*numOccTotal, MPI_DOUBLE, 0, mpi_comm);

 //           cu_XTXTemp.CopyFrom(XTX);
//            MAGMA::Potrf('U', numOccTotal, cu_XTXTemp.Data(), numOccTotal);
//modified by xmqin 20191202
//-------------------------------------------------------------------------------
            lapack::Potrf( 'U', numOccTotal, XTX.Data(), numOccTotal ); 
            cu_XTXTemp.CopyFrom(XTX);
//-------------------------------------------------------------------------------
//
            cublas::Trsm( HIPBLAS_SIDE_RIGHT, HIPBLAS_FILL_MODE_UPPER, HIPBLAS_OP_N, HIPBLAS_DIAG_NON_UNIT, 
                          ntotLocal, numOccTotal, &one, cu_XTXTemp.Data(), numOccTotal, cu_psiPcRow.Data(),
                          ntotLocal);
            cu_psiPcRow.CopyTo( psiPcRow );

            // X <- X * U^{-1} is orthogonal
            /*
            blas::Trsm( 'R', 'U', 'N', 'N', ntotLocal, numOccTotal, 1.0, XTX.Data(), numOccTotal, 
                psiPcRow.Data(), ntotLocal );
            SetValue( psiPcCol, 0.0 );
            */

            //AlltoallBackward (psiPcRow, psiPcCol, mpi_comm);
            SCALAPACK(pdgemr2d)(&Ng, &No, psiPcRow.Data(), &I_ONE, &I_ONE, desc_NgNo1DRow, 
                psiPcCol.Data(), &I_ONE, &I_ONE, desc_NgNo1DCol, &contxt1DCol );
          } 
        
        }//Anderson mixing
        } // GPU malloc and free.

        GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
        statusOFS << "Time for GPU  Anderson mixing in PWDFT is " <<
          timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

        // Construct the new Hamiltonian operator
        Spinor spnPsiPc(fft.domain, 1, numStateTotal,
            numStateLocal, false, psiPcCol.Data());

        // Compute the electron density
        GetTime( timeSta );
#ifdef GPU //---------------BY lijl 20200521
        ham.CalculateDensity_GPU(
            spnPsiPc,
            ham.OccupationRate(),
            totalCharge_,
            fft );
#else
        ham.CalculateDensity(
            spnPsiPc,
            ham.OccupationRate(),
            totalCharge_, 
            fft );
#endif
        GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
        statusOFS << "Time for computing density in PWDFT is " <<
          timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

        // Compute the exchange-correlation potential and energy
        if( isCalculateGradRho_ ){
          GetTime( timeSta );
          ham.CalculateGradDensity( fft );
          GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
          statusOFS << "Time for computing gradient density in PWDFT is " <<
            timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
        }

        GetTime( timeSta );
        ham.CalculateXC( Exc_, fft ); 
        GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
        statusOFS << "Time for computing XC potential in PWDFT is " <<
          timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

        // Compute the Hartree energy
        GetTime( timeSta );
        ham.CalculateHartree( fft );
        GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
        statusOFS << "Time for computing Hartree potential in PWDFT is " <<
          timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
        // No external potential

        // Compute the total potential
        GetTime( timeSta );
        ham.CalculateVtot( vtotNew_ );
        blas::Copy( ntotFine, vtotNew_.Data(), 1, ham.Vtot().Data(), 1 );

        GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
        statusOFS << "Time for computing total potential in PWDFT is " <<
          timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

        // Update Phi <- Psi
        GetTime( timeSta );
        ham.SetPhiEXX( spnPsiPc, fft ); 

        // Update the ACE if needed
        // Still use psi but phi has changed
        if( esdfParam.isHybridACE ){
          if( esdfParam.isHybridDF ){
            ham.CalculateVexxACEDF_GPU( psi, fft, isFixColumnDF );
            // Fix the column after the first iteraiton
            isFixColumnDF = true;
          }
          else{
            ham.CalculateVexxACE_GPU ( psi, fft );
          }
        }

        GetTime( timeEnd );
        statusOFS << "GPU Time for updating Phi related variable is " <<
          timeEnd - timeSta << " [s]" << std::endl << std::endl;

        InnerSolve( phiIter );
        blas::Copy( ntotFine, vtotNew_.Data(), 1, ham.Vtot().Data(), 1 );


        EVdw_ = 0.0;

        CalculateEnergy();

        PrintState( phiIter );

        GetTime( timePhiIterEnd );

        statusOFS << "Total wall clock time for this Phi iteration = " << 
          timePhiIterEnd - timePhiIterStart << " [s]" << std::endl;

        if(1){

          // Update Phi <- Psi
          GetTime( timeSta );
          ham.SetPhiEXX( psi, fft ); 

          // In principle there is no need to construct ACE operator here
          // However, this makes the code more readable by directly calling 
          // the MultSpinor function later
          if( esdfParam.isHybridACE ){
            if( esdfParam.isHybridDF ){
              ham.CalculateVexxACEDF_GPU( psi, fft, isFixColumnDF );
              // Fix the column after the first iteraiton
              isFixColumnDF = true;
            }
            else{
              // GPU needs to be done
              ham.CalculateVexxACE_GPU ( psi, fft );
            }
          }

          GetTime( timeEnd );
          statusOFS << " GPU Time for updating Phi related variable is " <<
            timeEnd - timeSta << " [s]" << std::endl << std::endl;

        }//if

        GetTime( timeSta );
        fock2 = ham.CalculateEXXEnergy( psi, fft ); 
        GetTime( timeEnd );
        statusOFS << "Time for computing the EXX energy is " <<
          timeEnd - timeSta << " [s]" << std::endl << std::endl;

        // Note: initially fock1 = 0.0. So it should at least run for 1 iteration.
        dExx = std::abs(fock2 - fock1) / std::abs(fock2);
        // use scfNorm to reflect dExx
        scfNorm_ = dExx;
        fock1 = fock2;
        Efock_ = fock2;

        Etot_ = Etot_ - Efock_;
        Efree_ = Efree_ - Efock_;

        statusOFS << std::endl;
        Print(statusOFS, "Fock energy       = ",  Efock_, "[au]");
        Print(statusOFS, "Etot(with fock)   = ",  Etot_, "[au]");
        Print(statusOFS, "Efree(with fock)  = ",  Efree_, "[au]");
        Print(statusOFS, "dExx              = ",  dExx, "[au]");
        
        if( dExx < scfPhiTolerance_ ){
          statusOFS << "SCF for hybrid functional is converged in " 
            << phiIter << " steps !" << std::endl;
          isPhiIterConverged = true;
        }
        if ( isPhiIterConverged ) break;
      } // for(phiIter)

      GetTime( timeEnd1 );
      statusOFS << "Time for using pcdiis method is " <<
        timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;

    } // hybridMixType == "pcdiis"

#else
    if( esdfParam.hybridMixType == "pcdiis" ){
     GetTime( timeSta1 );
      // This requires a good initial guess of wavefunctions, 
      // from one of the following
      // 1) a regular SCF calculation
      // 2) restarting wavefunction with Hybrid_Active_Init = true
        
      Int ntot      = fft.domain.NumGridTotal();
      Int ntotFine  = fft.domain.NumGridTotalFine();
      Int numStateTotal = psi.NumStateTotal();
      Int numStateLocal = psi.NumState();
      Int numOccTotal = ham.NumOccupiedState();

      MPI_Comm mpi_comm = eigSolPtr_->FFT().domain.comm;

      Int I_ONE = 1, I_ZERO = 0;
      double D_ONE = 1.0;
      double D_ZERO = 0.0;
      double D_MinusONE = -1.0;

      Real timeSta, timeEnd, timeSta1, timeEnd1;
      
      //Int contxt0D, contxt2D;
      Int contxt1DCol, contxt1DRow;
      Int nprow0D, npcol0D, myrow0D, mycol0D, info0D;
      Int nprow1DCol, npcol1DCol, myrow1DCol, mycol1DCol, info1DCol;
      Int nprow1DRow, npcol1DRow, myrow1DRow, mycol1DRow, info1DRow;
      Int nprow2D, npcol2D, myrow2D, mycol2D, info2D;

      Int ncolsNgNe1DCol, nrowsNgNe1DCol, lldNgNe1DCol; 
      Int ncolsNgNe1DRow, nrowsNgNe1DRow, lldNgNe1DRow; 
      Int ncolsNgNo1DCol, nrowsNgNo1DCol, lldNgNo1DCol; 
      Int ncolsNgNo1DRow, nrowsNgNo1DRow, lldNgNo1DRow; 

      Int desc_NgNe1DCol[9];
      Int desc_NgNe1DRow[9];
      Int desc_NgNo1DCol[9];
      Int desc_NgNo1DRow[9];

      Int Ne = numStateTotal; 
      Int No = numOccTotal; 
      Int Ng = ntot;

      // 1D col MPI
      nprow1DCol = 1;
      npcol1DCol = mpisize;

      Cblacs_get(0, 0, &contxt1DCol);
      Cblacs_gridinit(&contxt1DCol, "C", nprow1DCol, npcol1DCol);
      Cblacs_gridinfo(contxt1DCol, &nprow1DCol, &npcol1DCol, &myrow1DCol, &mycol1DCol);

      // 1D row MPI
      nprow1DRow = mpisize;
      npcol1DRow = 1;

      Cblacs_get(0, 0, &contxt1DRow);
      Cblacs_gridinit(&contxt1DRow, "C", nprow1DRow, npcol1DRow);
      Cblacs_gridinfo(contxt1DRow, &nprow1DRow, &npcol1DRow, &myrow1DRow, &mycol1DRow);


      //desc_NgNe1DCol
      if(contxt1DCol >= 0){
        nrowsNgNe1DCol = SCALAPACK(numroc)(&Ng, &Ng, &myrow1DCol, &I_ZERO, &nprow1DCol);
        ncolsNgNe1DCol = SCALAPACK(numroc)(&Ne, &I_ONE, &mycol1DCol, &I_ZERO, &npcol1DCol);
        lldNgNe1DCol = std::max( nrowsNgNe1DCol, 1 );
      }    

      SCALAPACK(descinit)(desc_NgNe1DCol, &Ng, &Ne, &Ng, &I_ONE, &I_ZERO, 
          &I_ZERO, &contxt1DCol, &lldNgNe1DCol, &info1DCol);

      //desc_NgNe1DRow
      if(contxt1DRow >= 0){
        nrowsNgNe1DRow = SCALAPACK(numroc)(&Ng, &BlockSizeScaLAPACK_, &myrow1DRow, &I_ZERO, &nprow1DRow);
        ncolsNgNe1DRow = SCALAPACK(numroc)(&Ne, &Ne, &mycol1DRow, &I_ZERO, &npcol1DRow);
        lldNgNe1DRow = std::max( nrowsNgNe1DRow, 1 );
      }    

      SCALAPACK(descinit)(desc_NgNe1DRow, &Ng, &Ne, &BlockSizeScaLAPACK_, &Ne, &I_ZERO, 
          &I_ZERO, &contxt1DRow, &lldNgNe1DRow, &info1DRow);

      //desc_NgNo1DCol
      if(contxt1DCol >= 0){
        nrowsNgNo1DCol = SCALAPACK(numroc)(&Ng, &Ng, &myrow1DCol, &I_ZERO, &nprow1DCol);
        ncolsNgNo1DCol = SCALAPACK(numroc)(&No, &I_ONE, &mycol1DCol, &I_ZERO, &npcol1DCol);
        lldNgNo1DCol = std::max( nrowsNgNo1DCol, 1 );
      }    

      SCALAPACK(descinit)(desc_NgNo1DCol, &Ng, &No, &Ng, &I_ONE, &I_ZERO, 
          &I_ZERO, &contxt1DCol, &lldNgNo1DCol, &info1DCol);

      //desc_NgNo1DRow
      if(contxt1DRow >= 0){
        nrowsNgNo1DRow = SCALAPACK(numroc)(&Ng, &BlockSizeScaLAPACK_, &myrow1DRow, &I_ZERO, &nprow1DRow);
        ncolsNgNo1DRow = SCALAPACK(numroc)(&No, &No, &mycol1DRow, &I_ZERO, &npcol1DRow);
        lldNgNo1DRow = std::max( nrowsNgNo1DRow, 1 );
      }    

      SCALAPACK(descinit)(desc_NgNo1DRow, &Ng, &No, &BlockSizeScaLAPACK_, &No, &I_ZERO, 
          &I_ZERO, &contxt1DRow, &lldNgNo1DRow, &info1DRow);
      
      if(numStateLocal !=  ncolsNgNe1DCol){
        statusOFS << "numStateLocal = " << numStateLocal << " ncolsNgNe1DCol = " << ncolsNgNe1DCol << std::endl;
        ErrorHandling("The size of numState is not right!");
      }
      
      if(nrowsNgNe1DRow !=  nrowsNgNo1DRow){
        statusOFS << "nrowsNgNe1DRow = " << nrowsNgNe1DRow << " ncolsNgNo1DRow = " << ncolsNgNo1DRow << std::endl;
        ErrorHandling("The size of nrowsNgNe1DRow and ncolsNgNo1DRow is not right!");
      }


      Int numOccLocal = ncolsNgNo1DCol;
      Int ntotLocal = nrowsNgNe1DRow;

      //DblNumMat psiPcCol(ntot, numStateLocal);
      //DblNumMat psiPcRow(ntotLocal, numStateTotal);
      DblNumMat HpsiCol(ntot, numStateLocal);
      DblNumMat HpsiRow(ntotLocal, numStateTotal);

      dfMat_.Resize( ntot * numOccLocal, mixMaxDim_ ); SetValue( dfMat_, 0.0 );
      dvMat_.Resize( ntot * numOccLocal, mixMaxDim_ ); SetValue( dvMat_, 0.0 );

      DblNumMat psiPcCol(ntot, numOccLocal);
      DblNumMat psiPcRow(ntotLocal, numOccTotal);
      DblNumMat PcCol(ntot, numOccLocal);
      DblNumMat PcRow(ntotLocal, numOccTotal);
      DblNumMat ResCol(ntot, numOccLocal);
      DblNumMat ResRow(ntotLocal, numOccTotal);

      DblNumMat psiMuT(numOccTotal, numOccTotal);
      DblNumMat psiMuTLocal(numOccLocal, numOccTotal);
      DblNumMat HpsiMuT(numOccTotal, numOccTotal);
      DblNumMat HpsiMuTLocal(numOccLocal, numOccTotal);

      DblNumMat psiCol( ntot, numStateLocal );
      SetValue( psiCol, 0.0 );
      DblNumMat psiRow( ntotLocal, numStateTotal );
      SetValue( psiRow, 0.0 );

      lapack::Lacpy( 'A', ntot, numStateLocal, psi.Wavefun().Data(), ntot, psiCol.Data(), ntot );
      //AlltoallForward (psiCol, psiRow, mpi_comm);
      SCALAPACK(pdgemr2d)(&Ng, &Ne, psiCol.Data(), &I_ONE, &I_ONE, desc_NgNe1DCol, 
          psiRow.Data(), &I_ONE, &I_ONE, desc_NgNe1DRow, &contxt1DCol );

      DblNumMat psiTemp(ntotLocal, numOccTotal);
      SetValue( psiTemp, 0.0 );

      lapack::Lacpy( 'A', ntotLocal, numOccTotal, psiRow.Data(), ntotLocal, psiTemp.Data(), ntotLocal );
      
      // Phi loop
      for( Int phiIter = 1; phiIter <= scfPhiMaxIter_; phiIter++ ){

        GetTime( timePhiIterStart );
          
        SetValue( psiCol, 0.0 );
        lapack::Lacpy( 'A', ntot, numStateLocal, psi.Wavefun().Data(), ntot, psiCol.Data(), ntot );
        SetValue( psiRow, 0.0 );
        //AlltoallForward (psiCol, psiRow, mpi_comm);
        SCALAPACK(pdgemr2d)(&Ng, &Ne, psiCol.Data(), &I_ONE, &I_ONE, desc_NgNe1DCol, 
            psiRow.Data(), &I_ONE, &I_ONE, desc_NgNe1DRow, &contxt1DCol );

        if(1){

          DblNumMat psiMuTTemp(numOccTotal, numOccTotal);
          SetValue( psiMuTTemp, 0.0 );
          blas::Gemm( 'T', 'N', numOccTotal, numOccTotal, ntotLocal, 1.0, 
              psiRow.Data(), ntotLocal, psiTemp.Data(), ntotLocal, 
              0.0, psiMuTTemp.Data(), numOccTotal );

          SetValue( psiMuT, 0.0 );
          MPI_Allreduce( psiMuTTemp.Data(), psiMuT.Data(), 
              numOccTotal * numOccTotal, MPI_DOUBLE, MPI_SUM, mpi_comm );

        }//if

#if ( _DEBUGlevel_ >= 1 )
        {
          // Monitor the singular values as a measure as the quality of
          // the selected columns
          DblNumMat tmp = psiMuT;
          DblNumVec s(numOccTotal);
          lapack::SingularValues( numOccTotal, numOccTotal, tmp.Data(), numOccTotal, s.Data() );
          statusOFS << "Spsi = " << s << std::endl;
        }
#endif

        blas::Gemm( 'N', 'N', ntotLocal, numOccTotal, numOccTotal, 1.0, 
            psiRow.Data(), ntotLocal, psiMuT.Data(), numOccTotal, 
            0.0, PcRow.Data(), ntotLocal );
        
        SetValue( PcCol, 0.0 );
        //AlltoallBackward (PcRow, PcCol, mpi_comm);
        SCALAPACK(pdgemr2d)(&Ng, &No, PcRow.Data(), &I_ONE, &I_ONE, desc_NgNo1DRow, 
            PcCol.Data(), &I_ONE, &I_ONE, desc_NgNo1DCol, &contxt1DCol );
        
        std::ostringstream msg;
        msg << "Phi iteration # " << phiIter;
        PrintBlock( statusOFS, msg.str() );

        // Compute the residual
        {
          // Compute Hpsi for all psi 
          NumTns<Real> tnsTemp(ntot, 1, numStateLocal, false, 
              HpsiCol.Data());
          ham.MultSpinor( psi, tnsTemp, fft );
        
          SetValue( HpsiRow, 0.0 );
          //AlltoallForward (HpsiCol, HpsiRow, mpi_comm);
          SCALAPACK(pdgemr2d)(&Ng, &Ne, HpsiCol.Data(), &I_ONE, &I_ONE, desc_NgNe1DCol, 
            HpsiRow.Data(), &I_ONE, &I_ONE, desc_NgNe1DRow, &contxt1DCol );

          if(1){

            DblNumMat HpsiMuTTemp(numOccTotal,numOccTotal);
            SetValue( HpsiMuTTemp, 0.0 );

            blas::Gemm( 'T', 'N', numOccTotal, numOccTotal, ntotLocal, 1.0, 
                HpsiRow.Data(), ntotLocal, psiTemp.Data(), ntotLocal, 
                0.0, HpsiMuTTemp.Data(), numOccTotal );

            SetValue( HpsiMuT, 0.0 );

            MPI_Allreduce( HpsiMuTTemp.Data(), HpsiMuT.Data(), 
                numOccTotal * numOccTotal, MPI_DOUBLE, MPI_SUM, mpi_comm );

          }//if

          blas::Gemm( 'N', 'N', ntotLocal, numOccTotal, numOccTotal, 1.0, 
              HpsiRow.Data(), ntotLocal, psiMuT.Data(), numOccTotal, 
              0.0, ResRow.Data(), ntotLocal );
          blas::Gemm( 'N', 'N', ntotLocal, numOccTotal, numOccTotal, -1.0, 
              psiRow.Data(), ntotLocal, HpsiMuT.Data(), numOccTotal, 
              1.0, ResRow.Data(), ntotLocal );
        
          SetValue( ResCol, 0.0 );
          //AlltoallBackward (ResRow, ResCol, mpi_comm);
          SCALAPACK(pdgemr2d)(&Ng, &No, ResRow.Data(), &I_ONE, &I_ONE, desc_NgNo1DRow, 
              ResCol.Data(), &I_ONE, &I_ONE, desc_NgNo1DCol, &contxt1DCol );
        }
        
        // Anderson mixing. Use the same mixMaxDim_ for Phi mixing
        {
          // Optimal input potential in Anderon mixing.
          DblNumVec vOpt( ntot * numOccLocal ); 

          // Number of iterations used, iter should start from 1
          Int iterused = std::min( phiIter-1, mixMaxDim_ ); 
          // The current position of dfMat, dvMat
          Int ipos = phiIter - 1 - ((phiIter-2)/ mixMaxDim_ ) * mixMaxDim_;
          // The next position of dfMat, dvMat
          Int inext = phiIter - ((phiIter-1)/ mixMaxDim_) * mixMaxDim_;
        
          blas::Copy( ntot * numOccLocal, PcCol.Data(), 1, vOpt.Data(), 1 );

          if( phiIter > 1 ){
            // dfMat(:, ipos-1) = res(:) - dfMat(:, ipos-1);
            // dvMat(:, ipos-1) = vOld(:) - dvMat(:, ipos-1);
            blas::Scal( ntot * numOccLocal, -1.0, dfMat_.VecData(ipos-1), 1 );
            blas::Axpy( ntot * numOccLocal, 1.0, ResCol.Data(), 1, dfMat_.VecData(ipos-1), 1 );
            blas::Scal( ntot * numOccLocal, -1.0, dvMat_.VecData(ipos-1), 1 );
            blas::Axpy( ntot * numOccLocal, 1.0, PcCol.Data(), 1, dvMat_.VecData(ipos-1), 1 );

            // Calculating pseudoinverse
            DblNumMat dfMatTemp(ntot * numOccLocal, mixMaxDim_);
            DblNumVec gammas(ntot * numOccLocal), S(iterused);
            
            SetValue( dfMatTemp, 0.0 );
            SetValue( gammas, 0.0 );

            Int rank;
            // FIXME Magic number
            Real rcond = 1e-9;

            // gammas    = res;
            blas::Copy( ntot * numOccLocal, ResCol.Data(), 1, gammas.Data(), 1 );
            lapack::Lacpy( 'A', ntot * numOccLocal, mixMaxDim_, dfMat_.Data(), ntot * numOccLocal, 
                dfMatTemp.Data(), ntot * numOccLocal );
        
            // May need different strategy in a parallel setup
            if(0){  
              
              lapack::SVDLeastSquare( ntot * numOccLocal, iterused, 1, 
                  dfMatTemp.Data(), ntot * numOccLocal,
                  gammas.Data(), ntot * numOccLocal,
                  S.Data(), rcond, &rank );
            
              blas::Gemv('N', ntot * numOccLocal, iterused, -1.0, dvMat_.Data(),
                  ntot * numOccLocal, gammas.Data(), 1, 1.0, vOpt.Data(), 1 );
            
            }
        
            if(1){

              DblNumMat XTX(iterused, iterused);
              DblNumMat XTXTemp(iterused, iterused);
              
              Int lld_ntotnumOccLocal = std::max( ntot * numOccLocal, 1 );
              
              SetValue( XTXTemp, 0.0 );
              blas::Gemm( 'T', 'N', iterused, iterused, ntot * numOccLocal, 1.0, 
              dfMatTemp.Data(), lld_ntotnumOccLocal, dfMatTemp.Data(), 
              lld_ntotnumOccLocal, 0.0, XTXTemp.Data(), iterused );
        
              SetValue( XTX, 0.0 );
              MPI_Allreduce( XTXTemp.Data(), XTX.Data(), 
                  iterused * iterused, MPI_DOUBLE, MPI_SUM, mpi_comm );
            
              DblNumVec gammasTemp1(iterused);
              SetValue( gammasTemp1, 0.0 );
              //blas::Gemv('T', ntot * numOccLocal, iterused, 1.0, dfMatTemp.Data(),
              //    ntot * numOccLocal, gammas.Data(), 1, 0.0, gammasTemp1.Data(), 1 );

              blas::Gemm( 'T', 'N', iterused, I_ONE, ntot * numOccLocal, 1.0, 
                  dfMatTemp.Data(), lld_ntotnumOccLocal, gammas.Data(), 
                  lld_ntotnumOccLocal, 0.0, gammasTemp1.Data(), iterused );

              DblNumVec gammasTemp2(iterused);
              SetValue( gammasTemp2, 0.0 );
              MPI_Allreduce( gammasTemp1.Data(), gammasTemp2.Data(), 
                  iterused, MPI_DOUBLE, MPI_SUM, mpi_comm );
              
              lapack::SVDLeastSquare( iterused, iterused, 1, 
                  XTX.Data(), iterused,
                  gammasTemp2.Data(), iterused,
                  S.Data(), rcond, &rank );
            
              //blas::Gemv('N', ntot * numOccLocal, iterused, -1.0, dvMat_.Data(),
              //    ntot * numOccLocal, gammasTemp2.Data(), 1, 1.0, vOpt.Data(), 1 );
              
              blas::Gemm( 'N', 'N', ntot * numOccLocal, I_ONE, iterused, -1.0, 
                  dvMat_.Data(), lld_ntotnumOccLocal, gammasTemp2.Data(), iterused, 
                  1.0, vOpt.Data(), lld_ntotnumOccLocal );
            
            }

            Print( statusOFS, "  Rank of dfmat = ", rank );

          }

          // Update dfMat, dvMat, vMix 
          // dfMat(:, inext-1) = Res(:)
          // dvMat(:, inext-1) = Pc(:)
          blas::Copy( ntot * numOccLocal, ResCol.Data(), 1, 
              dfMat_.VecData(inext-1), 1 );
          blas::Copy( ntot * numOccLocal, PcCol.Data(),  1, 
              dvMat_.VecData(inext-1), 1 );
            
          // Orthogonalize vOpt to obtain psiPc. 
          // psiPc has the same size
          SetValue( psiPcCol, 0.0 );
          blas::Copy( ntot * numOccLocal, vOpt.Data(), 1, psiPcCol.Data(), 1 );
          //lapack::Orth( ntot, numOccLocal, psiPcCol.Data(), ntotLocal );
          
          // Orthogonalization through Cholesky factorization
         if(1){ 
            SetValue( psiPcRow, 0.0 );
            //AlltoallForward (psiPcCol, psiPcRow, mpi_comm);
            SCALAPACK(pdgemr2d)(&Ng, &No, psiPcCol.Data(), &I_ONE, &I_ONE, desc_NgNo1DCol, 
                psiPcRow.Data(), &I_ONE, &I_ONE, desc_NgNo1DRow, &contxt1DCol );

            DblNumMat XTX(numOccTotal, numOccTotal);
            DblNumMat XTXTemp(numOccTotal, numOccTotal);

            blas::Gemm( 'T', 'N', numOccTotal, numOccTotal, ntotLocal, 1.0, psiPcRow.Data(), 
                ntotLocal, psiPcRow.Data(), ntotLocal, 0.0, XTXTemp.Data(), numOccTotal );
            SetValue( XTX, 0.0 );
            MPI_Allreduce(XTXTemp.Data(), XTX.Data(), numOccTotal*numOccTotal, MPI_DOUBLE, MPI_SUM, mpi_comm);

            if ( mpirank == 0) {
              lapack::Potrf( 'U', numOccTotal, XTX.Data(), numOccTotal );
            }
            MPI_Bcast(XTX.Data(), numOccTotal*numOccTotal, MPI_DOUBLE, 0, mpi_comm);

            // X <- X * U^{-1} is orthogonal
            blas::Trsm( 'R', 'U', 'N', 'N', ntotLocal, numOccTotal, 1.0, XTX.Data(), numOccTotal, 
                psiPcRow.Data(), ntotLocal );

            SetValue( psiPcCol, 0.0 );
            //AlltoallBackward (psiPcRow, psiPcCol, mpi_comm);
            SCALAPACK(pdgemr2d)(&Ng, &No, psiPcRow.Data(), &I_ONE, &I_ONE, desc_NgNo1DRow, 
                psiPcCol.Data(), &I_ONE, &I_ONE, desc_NgNo1DCol, &contxt1DCol );
          } 
        
        }//Anderson mixing

        DblNumMat psiPcColTemp(ntot, numStateLocal);
        SetValue( psiPcColTemp, 0.0 );

        lapack::Lacpy( 'A', ntot, numOccLocal, psiPcCol.Data(), ntot, psiPcColTemp.Data(), ntot );

        // Construct the new Hamiltonian operator
        Spinor spnPsiPc(fft.domain, 1, numStateTotal,
            numStateLocal, false, psiPcColTemp.Data());

        // Compute the electron density
        GetTime( timeSta );
#ifdef GPU//----------------by lijl 20200521
        ham.CalculateDensity_GPU(
            spnPsiPc,
            ham.OccupationRate(),
            totalCharge_,
            fft);
#else
        ham.CalculateDensity(
            spnPsiPc,
            ham.OccupationRate(),
            totalCharge_, 
            fft );
#endif
        GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
        statusOFS << "Time for computing density in PWDFT is " <<
          timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

        // Compute the exchange-correlation potential and energy
        if( isCalculateGradRho_ ){
          GetTime( timeSta );
          ham.CalculateGradDensity( fft );
          GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
          statusOFS << "Time for computing gradient density in PWDFT is " <<
            timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
        }

        GetTime( timeSta );
        ham.CalculateXC( Exc_, fft ); 
        GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
        statusOFS << "Time for computing XC potential in PWDFT is " <<
          timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

        // Compute the Hartree energy
        GetTime( timeSta );
        ham.CalculateHartree( fft );
        GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
        statusOFS << "Time for computing Hartree potential in PWDFT is " <<
          timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
        // No external potential

        // Compute the total potential
        GetTime( timeSta );
        ham.CalculateVtot( vtotNew_ );
        blas::Copy( ntotFine, vtotNew_.Data(), 1, ham.Vtot().Data(), 1 );

        GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
        statusOFS << "Time for computing total potential in PWDFT is " <<
          timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

        // Update Phi <- Psi
        GetTime( timeSta );
        ham.SetPhiEXX( spnPsiPc, fft ); 

        // Update the ACE if needed
        // Still use psi but phi has changed
        if( esdfParam.isHybridACE ){
          if( esdfParam.isHybridDF ){
            ham.CalculateVexxACEDF( psi, fft, isFixColumnDF );
            // Fix the column after the first iteraiton
            isFixColumnDF = true;
          }
          else{
            ham.CalculateVexxACE ( psi, fft );
          }
        }

        GetTime( timeEnd );
        statusOFS << "Time for updating Phi related variable is " <<
          timeEnd - timeSta << " [s]" << std::endl << std::endl;

        InnerSolve( phiIter );
        blas::Copy( ntotFine, vtotNew_.Data(), 1, ham.Vtot().Data(), 1 );


        CalculateEnergy();

        PrintState( phiIter );

        GetTime( timePhiIterEnd );

        statusOFS << "Total wall clock time for this Phi iteration = " << 
          timePhiIterEnd - timePhiIterStart << " [s]" << std::endl;

        if(esdfParam.isHybridACETwicePCDIIS == 1){

          // Update Phi <- Psi
          GetTime( timeSta );
          ham.SetPhiEXX( psi, fft ); 

          // In principle there is no need to construct ACE operator here
          // However, this makes the code more readable by directly calling 
          // the MultSpinor function later
          if( esdfParam.isHybridACE ){
            if( esdfParam.isHybridDF ){
              ham.CalculateVexxACEDF( psi, fft, isFixColumnDF );
              // Fix the column after the first iteraiton
              isFixColumnDF = true;
            }
            else{
              ham.CalculateVexxACE ( psi, fft );
            }
          }

          GetTime( timeEnd );
          statusOFS << "Time for updating Phi related variable is " <<
            timeEnd - timeSta << " [s]" << std::endl << std::endl;

        }//if

        GetTime( timeSta );
        fock2 = ham.CalculateEXXEnergy( psi, fft ); 
        GetTime( timeEnd );
        statusOFS << "Time for computing the EXX energy is " <<
          timeEnd - timeSta << " [s]" << std::endl << std::endl;

        // Note: initially fock1 = 0.0. So it should at least run for 1 iteration.
        dExx = std::abs(fock2 - fock1) / std::abs(fock2);
        // use scfNorm to reflect dExx
        scfNorm_ = dExx;
        fock1 = fock2;
        Efock_ = fock2;

        Etot_ = Etot_ - Efock_;
        Efree_ = Efree_ - Efock_;

        statusOFS << std::endl;
        Print(statusOFS, "Fock energy       = ",  Efock_, "[au]");
        Print(statusOFS, "Etot(with fock)   = ",  Etot_, "[au]");
        Print(statusOFS, "Efree(with fock)  = ",  Efree_, "[au]");
        Print(statusOFS, "dExx              = ",  dExx, "[au]");
        
        if( dExx < scfPhiTolerance_ ){
          statusOFS << "SCF for hybrid functional is converged in " 
            << phiIter << " steps !" << std::endl;
          isPhiIterConverged = true;
        }
        if ( isPhiIterConverged ) break;
      } // for(phiIter)

      if(contxt1DCol >= 0) {
        Cblacs_gridexit( contxt1DCol );
      }

      if(contxt1DRow >= 0) {
        Cblacs_gridexit( contxt1DRow );
      }

      GetTime( timeEnd1 );
      statusOFS << "Time for using pcdiis method is " <<
        timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
    } // hybridMixType == "pcdiis"
#endif

  } // isHybrid == true
#ifdef GPU
    cublas::Destroy();
    //MAGMA::Destroy();
    cuda_clean_vtot();
#endif
#endif

  // Calculate the Force. This includes contribution from ionic repulsion, VDW etc
  ham.CalculateForce( psi, fft );

  // Output the information after SCF
  {
    // Energy
    Real HOMO, LUMO, EG;
    HOMO = eigSolPtr_->EigVal()(eigSolPtr_->Ham().NumOccupiedState()-1);
    if( eigSolPtr_->Ham().NumExtraState() > 0 ) {
      LUMO = eigSolPtr_->EigVal()(eigSolPtr_->Ham().NumOccupiedState());
      EG = LUMO - HOMO;
    }
    // Print out the energy
    PrintBlock( statusOFS, "Energy" );
//    statusOFS 
//      << "NOTE:  Ecor  = Exc - EVxc - Ehart - Eself + EIonSR + EVdw + Eext" << std::endl
//      << "       Etot  = Ekin + Ecor" << std::endl
//      << "       Efree = Etot    + Entropy" << std::endl << std::endl;
    Print(statusOFS, "! EfreeHarris     = ",  EfreeHarris_, "[au]");
    Print(statusOFS, "! Etot            = ",  Etot_, "[au]");
    Print(statusOFS, "! Exc             = ",  Exc_, "[au]");
    Print(statusOFS, "! Exx             = ",  Efock_, "[au]");
    Print(statusOFS, "! Efree           = ",  Efree_, "[au]");
    Print(statusOFS, "! Evdw            = ",  EVdw_, "[au]");
    Print(statusOFS, "! Eext              = ",  Eext_, "[au]");
    Print(statusOFS, "! Fermi           = ",  fermi_, "[au]");
    Print(statusOFS, "! HOMO            = ",  HOMO*au2ev, "[eV]");
    if( ham.NumExtraState() > 0 ){
      Print(statusOFS, "! LUMO            = ",  LUMO*au2ev, "[eV]");
      Print(statusOFS, "! Band Gap        = ",  EG*au2ev, "[eV]");
    }
  }

  {
    // Print out the force
    PrintBlock( statusOFS, "Atomic Force" );

    Point3 forceCM(0.0, 0.0, 0.0);
    std::vector<Atom>& atomList = ham.AtomList();
    Int numAtom = atomList.size();

    for( Int a = 0; a < numAtom; a++ ){
      Print( statusOFS, "atom", a, "force", atomList[a].force );
      forceCM += atomList[a].force;
    }
    statusOFS << std::endl;
    Print( statusOFS, "force for centroid  : ", forceCM );
    Print( statusOFS, "Max force magnitude : ", MaxForce(atomList) );
    statusOFS << std::endl;
  }


  // Output the structure information
  if(1){
    if( mpirank == 0 ){
      std::ostringstream structStream;
#if ( _DEBUGlevel_ >= 0 )
      statusOFS << std::endl 
        << "Output the structure information" 
        << std::endl;
#endif
      // Domain
      const Domain& dm =  eigSolPtr_->FFT().domain;
      serialize( dm.length, structStream, NO_MASK );
      serialize( dm.numGrid, structStream, NO_MASK );
      serialize( dm.numGridFine, structStream, NO_MASK );
      serialize( dm.posStart, structStream, NO_MASK );

      // Atomic information
      serialize( ham.AtomList(), structStream, NO_MASK );
      std::string structFileName = "STRUCTURE";

      std::ofstream fout(structFileName.c_str());
      if( !fout.good() ){
        std::ostringstream msg;
        msg 
          << "File " << structFileName.c_str() << " cannot be open." 
          << std::endl;
        ErrorHandling( msg.str().c_str() );
      }
      fout << structStream.str();
      fout.close();
    }
  }


  // Output restarting information
  if( esdfParam.isOutputDensity ){
    if( mpirank == 0 ){
      std::ofstream rhoStream(restartDensityFileName_.c_str());
      if( !rhoStream.good() ){
        ErrorHandling( "Density file cannot be opened." );
      }

      const Domain& dm =  eigSolPtr_->FFT().domain;
      std::vector<DblNumVec>   gridpos(DIM);
      UniformMeshFine ( dm, gridpos );
      for( Int d = 0; d < DIM; d++ ){
        serialize( gridpos[d], rhoStream, NO_MASK );
      }

      // Only work for the restricted spin case
      DblNumMat& densityMat = eigSolPtr_->Ham().Density();
      DblNumVec densityVec(densityMat.m(), false, densityMat.Data());
      serialize( densityVec, rhoStream, NO_MASK );
      rhoStream.close();
    }
  }    

  // Output the total potential
  if( esdfParam.isOutputPotential ){
    if( mpirank == 0 ){
      std::ofstream vtotStream(restartPotentialFileName_.c_str());
      if( !vtotStream.good() ){
        ErrorHandling( "Potential file cannot be opened." );
      }

      const Domain& dm =  eigSolPtr_->FFT().domain;
      std::vector<DblNumVec>   gridpos(DIM);
      UniformMeshFine ( dm, gridpos );
      for( Int d = 0; d < DIM; d++ ){
        serialize( gridpos[d], vtotStream, NO_MASK );
      }

      serialize( eigSolPtr_->Ham().Vtot(), vtotStream, NO_MASK );
      vtotStream.close();
    }
  }

  if( esdfParam.isOutputWfn ){
    std::ostringstream wfnStream;
    serialize( eigSolPtr_->Psi().Wavefun(), wfnStream, NO_MASK );
    serialize( eigSolPtr_->Ham().OccupationRate(), wfnStream, NO_MASK );
    SeparateWrite( restartWfnFileName_, wfnStream, mpirank );
  }   


  return ;
}         // -----  end of method SCF::Iterate  ----- 


void
SCF::InnerSolve	( Int iter )
{
  int mpirank;  MPI_Comm_rank(eigSolPtr_->FFT().domain.comm, &mpirank);
  int mpisize;  MPI_Comm_size(eigSolPtr_->FFT().domain.comm, &mpisize);

  Real timeSta, timeEnd;
  // Only works for KohnSham class
  Hamiltonian& ham = eigSolPtr_->Ham();
  Fourier&     fft = eigSolPtr_->FFT();
  Spinor&      psi = eigSolPtr_->Psi();

  // Solve the eigenvalue problem

  Real eigTolNow;
  if( isEigToleranceDynamic_ ){
    // Dynamic strategy to control the tolerance
    if( iter == 1 )
      eigTolNow = 1e-2;
    else
      eigTolNow = std::max( std::min( scfNorm_*1e-4, 1e-2 ) , eigTolerance_);
  }
  else{
    // Static strategy to control the tolerance
    eigTolNow = eigTolerance_;
  }

  Int numEig = (psi.NumStateTotal());

//  if(Diag_SCF_PWDFT_by_Cheby_ == 0)
//  {  
//
//    statusOFS << "The current tolerance used by the eigensolver is " 
//      << eigTolNow << std::endl;
//    statusOFS << "The target number of converged eigenvectors is " 
//      << numEig << std::endl;
//  }

  GetTime( timeSta );

  if(Diag_SCF_PWDFT_by_Cheby_ == 1)
  {
    if(Cheby_iondynamics_schedule_flag_ == 0)
    {
      // Use static schedule
      statusOFS << std::endl << " CheFSI in PWDFT working on static schedule." << std::endl;
      // Use CheFSI or LOBPCG on first step 
#ifndef _COMPLEX_
      if(iter <= 1){
        if(First_SCF_PWDFT_ChebyCycleNum_ <= 0)
          eigSolPtr_->LOBPCGSolveReal(numEig, iter, eigMaxIter_, eigMinTolerance_, eigTolNow );    
        else
          eigSolPtr_->FirstChebyStep(numEig, First_SCF_PWDFT_ChebyCycleNum_, First_SCF_PWDFT_ChebyFilterOrder_);
      }
      else{
        eigSolPtr_->GeneralChebyStep(numEig, General_SCF_PWDFT_ChebyFilterOrder_);
      }
#endif
    }
    else
    {
      // Use ion-dynamics schedule
#ifndef _COMPLEX_
      statusOFS << std::endl << " CheFSI in PWDFT working on ion-dynamics schedule." << std::endl;
      if( iter <= 1)
      {
        for (int cheby_iter = 1; cheby_iter <= eigMaxIter_; cheby_iter ++)
          eigSolPtr_->GeneralChebyStep(numEig, General_SCF_PWDFT_ChebyFilterOrder_);
      }
      else
      {
        eigSolPtr_->GeneralChebyStep(numEig, General_SCF_PWDFT_ChebyFilterOrder_);
      }
#endif
    }
  }
  else
  {
    // Use LOBPCG
#ifdef _COMPLEX_
    if( esdfParam.PWSolver == "PPCG" || esdfParam.PWSolver == "PPCGScaLAPACK" ){
      eigSolPtr_->PPCGSolveComplex(numEig, iter, eigMaxIter_, eigMinTolerance_, eigTolNow );    
    }
    else{
      // FIXME Merge the Chebyshev into an option of PWSolver
      ErrorHandling("Not supported PWSolver for complex type.");
    }
#else
    if( esdfParam.PWSolver == "LOBPCG" || esdfParam.PWSolver == "LOBPCGScaLAPACK"){
      eigSolPtr_->LOBPCGSolveReal(numEig, iter, eigMaxIter_, eigMinTolerance_, eigTolNow );    
    } // Use PPCG
    else if( esdfParam.PWSolver == "PPCG" || esdfParam.PWSolver == "PPCGScaLAPACK" ){
#ifdef GPU
     if(mpisize == 1)
      {
        statusOFS << " >>>>>>>>> Calling serial  PPCG  ..." << std::endl;
        eigSolPtr_->PPCGSolveReal_GPU(numEig, iter, eigMaxIter_, eigMinTolerance_, eigTolNow, true );
      }
      else
      {
        statusOFS << " >>>>>>>>> Calling parallel PPCG  ..." << std::endl;
        eigSolPtr_->PPCGSolveReal_GPU(numEig, iter, eigMaxIter_, eigMinTolerance_, eigTolNow );
   }

      
#else
     if(mpisize == 1)
      {
        statusOFS << " >>>>>>>>> Calling serial  PPCG  ..." << std::endl;
        eigSolPtr_->PPCGSolveReal(numEig, iter, eigMaxIter_, eigMinTolerance_, eigTolNow, true );
      }
      else{
        statusOFS << " >>>>>>>>> Calling parallel PPCG  ..." << std::endl;
        eigSolPtr_->PPCGSolveReal(numEig, iter, eigMaxIter_, eigMinTolerance_, eigTolNow );    
      }
#endif
    }
    else{
      // FIXME Merge the Chebyshev into an option of PWSolver
      ErrorHandling("Not supported PWSolver type.");
    }
#endif
  }

  GetTime( timeEnd );

#if ( _DEBUGlevel_ >= 0 )
  statusOFS << std::endl << "Time for the eigensolver is " <<
    timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

  GetTime( timeSta );
  ham.EigVal() = eigSolPtr_->EigVal();
  GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
  statusOFS << "Time for ham.EigVal() in PWDFT is " <<
    timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

  // No need for normalization using LOBPCG

  // Compute the occupation rate
  GetTime( timeSta );
  CalculateOccupationRate( ham.EigVal(), ham.OccupationRate() );
  GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
  statusOFS << "Time for computing occupation rate in PWDFT is " <<
    timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

  // Calculate the Harris energy before updating the density
  CalculateHarrisEnergy ();

  // Compute the electron density

  // Store the old density in densityOld_ before updating it
  Int ntotFine = fft.domain.NumGridTotalFine();
  DblNumMat & density = ham.Density();

    blas::Copy( ntotFine, density.VecData(RHO), 1, densityOld_.Data(), 1 );

  GetTime( timeSta );
#ifdef GPU 
  ham.CalculateDensity_GPU(
      psi,
      ham.OccupationRate(),
      totalCharge_,
      fft );
#else
  ham.CalculateDensity(
      psi,
      ham.OccupationRate(),
      totalCharge_, 
      fft );
#endif
  GetTime( timeEnd );

#if ( _DEBUGlevel_ >= 0 )
  statusOFS << "Time for computing density in PWDFT is " <<
    timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

 // Mixing density if the convergence does not reach
  if( mixVariable_ == "density" ){
    // Use density as convergence criteria when density is mixed
    Real normDenDif = 0.0, normDenOld = 0.0;

    DblNumVec densityNew_;
    densityNew_.Resize(ntotFine); SetValue(densityNew_, 0.0);    

    blas::Copy( ntotFine, density.VecData(RHO), 1, densityNew_.Data(), 1 );

    for( Int i = 0; i < ntotFine; i++ ){
        normDenDif += pow( densityOld_(i) - densityNew_(i), 2.0 );
        normDenOld += pow( densityOld_(i), 2.0 );
    }
    
    normDenDif = sqrt( normDenDif );
    normDenOld = sqrt( normDenOld );
    scfNorm_    = normDenDif / normDenOld;

    if( scfNorm_ >= scfTolerance_ ){

      DblNumVec  densityVec(ntotFine) ;
      SetValue(densityVec, 0.0);

      if( mixType_ == "anderson" || mixType_ == "kerker+anderson" ){
        AndersonMix(
            iter,
            mixStepLength_,
            mixType_,
            densityVec,
            densityOld_,
            densityNew_,
            dfMat_,
            dvMat_);

      }
      else if( mixType_ == "broyden"){
        BroydenMix(
            iter,
            mixStepLength_,
            mixType_,
            densityVec,
            densityOld_,
            densityNew_,
            GdfMat_,
            GdvMat_,
            GcdfMat_);   
      }
      else{
        ErrorHandling("Invalid mixing type.");
      }

      blas::Copy( ntotFine, densityVec.Data(), 1, ham.Density().VecData(RHO), 1 );

    }
  } // if mixVariable_ == "density"

  // Compute the exchange-correlation potential and energy
  if( isCalculateGradRho_ ){
    GetTime( timeSta );
    ham.CalculateGradDensity( fft );
    GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for computing gradient density in PWDFT is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
  }

  GetTime( timeSta );
  ham.CalculateXC( Exc_, fft ); 
  GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
  statusOFS << "Time for computing XC potential in PWDFT is " <<
    timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

  // Compute the Hartree energy
  GetTime( timeSta );
  ham.CalculateHartree( fft );
  GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
  statusOFS << "Time for computing Hartree potential in PWDFT is " <<
    timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
  // No external potential

  // Compute the total potential
  GetTime( timeSta );
  ham.CalculateVtot( vtotNew_ );
  GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
  statusOFS << "Time for computing total potential in PWDFT is " <<
    timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif


  return ;
} 		// -----  end of method SCF::InnerSolve  ----- 

void
SCF::CalculateOccupationRate    ( DblNumVec& eigVal, DblNumVec& occupationRate )
{
  Int npsi       = eigSolPtr_->Ham().NumStateTotal();
  Int nOccStates = eigSolPtr_->Ham().NumOccupiedState();

  std::string smearing_scheme = esdfParam.smearing_scheme;

  if( eigVal.m() != npsi ){
    std::ostringstream msg;
    msg 
      << "The number of eigenstates do not match."  << std::endl
      << "eigVal         ~ " << eigVal.m() << std::endl
      << "numStateTotal  ~ " << npsi << std::endl;
    ErrorHandling( msg.str().c_str() );
  }

  if( occupationRate.m() != npsi ) occupationRate.Resize( npsi );
  
  DblNumVec eigValTotal( npsi );
  blas::Copy( npsi, eigVal.Data(), 1, eigValTotal.Data(), 1 );

  Sort( eigValTotal );
  
  if( npsi == nOccStates ){
    for( Int j = 0; j < npsi; j++ ){
      occupationRate(j) = 1.0;
    }
    fermi_ = eigValTotal(npsi-1);
  }  
  else if( npsi > nOccStates ){
    if( esdfParam.temperature == 0.0 ){
      fermi_ = eigValTotal(nOccStates-1);
      for( Int j = 0; j < npsi; j++ ){
        if( eigValTotal[j] <= fermi_ ){
          occupationRate(j) = 1.0;
        }
        else{
          occupationRate(j) = 0.0;
        }
      }
    }
    else{
      Real tol = 1e-16;
      Int maxiter = 200;

      Real lb, ub, flb, fub, occsum;
      Int ilb, iub, iter;

      ilb = 1;
      iub = npsi;
      lb = eigValTotal(ilb-1);
      ub = eigValTotal(iub-1);

      fermi_ = (lb+ub)*0.5;
      occsum = 0.0;
      for(Int j = 0; j < npsi; j++){
        occsum += wgauss( eigValTotal(j), fermi_, Tbeta_, smearing_scheme );
      }

      iter = 1;
      while( (fabs(occsum - nOccStates) > tol) && (iter < maxiter) ) {
        if( occsum < nOccStates ) {lb = fermi_;}
        else {ub = fermi_;}

        fermi_ = (lb+ub)*0.5;
        occsum = 0.0;
        for(Int j = 0; j < npsi; j++){
          occsum += wgauss( eigValTotal(j), fermi_, Tbeta_, smearing_scheme );
        }
        iter++;
      }

      for(Int j = 0; j < npsi; j++){
        occupationRate(j) = wgauss( eigValTotal(j), fermi_, Tbeta_, smearing_scheme );
      }
    }
  }
  else{
    ErrorHandling( "The number of eigenvalues in ev should be larger than nocc" );
  }

  return;
}         // -----  end of method SCF::CalculateOccupationRate  ----- 


void
SCF::CalculateEnergy    (  )
{

  std::string smearing_scheme = esdfParam.smearing_scheme;  

  Ekin_ = 0.0;
  DblNumVec&  eigVal         = eigSolPtr_->Ham().EigVal();
  DblNumVec&  occupationRate = eigSolPtr_->Ham().OccupationRate();

  // Kinetic energy
  Int numSpin = eigSolPtr_->Ham().NumSpin();
  for (Int i=0; i < eigVal.m(); i++) {
    Ekin_  += numSpin * eigVal(i) * occupationRate(i);
  }

  // Hartree and xc part
  Int  ntot = eigSolPtr_->FFT().domain.NumGridTotalFine();
  Real vol  = eigSolPtr_->FFT().domain.Volume();
  DblNumMat&  density      = eigSolPtr_->Ham().Density();
//  DblNumMat&  vxc          = eigSolPtr_->Ham().Vxc();
  DblNumVec&  pseudoCharge = eigSolPtr_->Ham().PseudoCharge();
  DblNumVec&  vhart        = eigSolPtr_->Ham().Vhart();
  DblNumVec&  vtotOld      = eigSolPtr_->Ham().Vtot();
  DblNumVec&  vLocalSR     = eigSolPtr_->Ham().VLocalSR();
  DblNumVec&  vext         = eigSolPtr_->Ham().Vext();

  Ecor_ = 0.0;
  for( Int i = 0; i < ntot; i++ ){
    Ecor_ += ( vLocalSR(i) + vext(i) - vtotOld(i) ) * density(i,RHO);
  }
  Ecor_ *= vol/Real(ntot);
 
  Ehart_ = 0.0;

  for (Int i=0; i<ntot; i++) {
    Ehart_ += 0.5 * vhart(i) * ( density(i,RHO) - pseudoCharge(i) );
  }

  Ehart_ *= vol/Real(ntot);
  // Ionic repulsion related energy
  Eself_ = eigSolPtr_->Ham().Eself();

  Ecor_ = Ecor_ + Exc_ + Ehart_ - Eself_;
  if( esdfParam.isUseVLocal == true ){
    EIonSR_ = eigSolPtr_->Ham().EIonSR();
    Ecor_ += EIonSR_;
  }

  // Van der Waals energy
  EVdw_ = eigSolPtr_->Ham().EVdw();
  Ecor_ += EVdw_;

  // External energy
  Eext_ = eigSolPtr_->Ham().Eext();
  Ecor_ += Eext_;

  // Total energy
  Etot_ = Ekin_ + Ecor_;

  // Helmholtz fre energy
  if( eigSolPtr_->Ham().NumOccupiedState() == eigSolPtr_->Ham().NumStateTotal()
      || esdfParam.temperature == 0.0 ){
    // Zero temperature
    Efree_ = Etot_;
  }
  else{
    // Finite temperature
    Efree_ = 0.0;
    Real fermi = fermi_;
    Real Tbeta = Tbeta_;

    for(Int l=0; l< eigVal.m(); l++) {
      Real eig = eigVal(l);
      Efree_ += numSpin * getEntropy( eig, fermi, Tbeta, smearing_scheme );
    }
    Efree_ += Etot_;
  }

  return ;
}         // -----  end of method SCF::CalculateEnergy  ----- 

void
SCF::CalculateHarrisEnergy ( )
{
  // These variables are temporary variables only used in this routine

  std::string smearing_scheme = esdfParam.smearing_scheme; 
  Real Ekin, Eself, Ehart, EVxc, Exc, Ecor, Efree, EIonSR, EVdw, Eext;
  
  Ekin = 0.0;
  DblNumVec&  eigVal         = eigSolPtr_->Ham().EigVal();
  DblNumVec&  occupationRate = eigSolPtr_->Ham().OccupationRate();

  // Kinetic energy
  Int numSpin = eigSolPtr_->Ham().NumSpin();
  for (Int i=0; i < eigVal.m(); i++) {
    Ekin  += numSpin * eigVal(i) * occupationRate(i);
  }

  // Self energy part
  Eself = 0;
  std::vector<Atom>&  atomList = eigSolPtr_->Ham().AtomList();
  for(Int a=0; a< atomList.size() ; a++) {
    Int type = atomList[a].type;
    Eself +=  ptablePtr_->SelfIonInteraction(type);
  }


  // Ionic repulsion related energy
  Eself = eigSolPtr_->Ham().Eself();

  EIonSR = eigSolPtr_->Ham().EIonSR();

  // Van der Waals energy
  EVdw = eigSolPtr_->Ham().EVdw();

  // External energy
  Eext = eigSolPtr_->Ham().Eext();


  // Nonlinear correction part.  This part uses the Hartree energy and
  // XC correlation energy from the old electron density.
  Int  ntot = eigSolPtr_->FFT().domain.NumGridTotalFine();
  Real vol  = eigSolPtr_->FFT().domain.Volume();
  DblNumMat&  density      = eigSolPtr_->Ham().Density();
  DblNumMat&  vxc          = eigSolPtr_->Ham().Vxc();
  DblNumVec&  pseudoCharge = eigSolPtr_->Ham().PseudoCharge();
  DblNumVec&  vhart        = eigSolPtr_->Ham().Vhart();
  Ehart = 0.0;
  EVxc  = 0.0;
  for (Int i=0; i<ntot; i++) {
    EVxc  += vxc(i,RHO) * density(i,RHO);
    Ehart += 0.5 * vhart(i) * ( density(i,RHO) + pseudoCharge(i) );
  }
  Ehart *= vol/Real(ntot);
  EVxc  *= vol/Real(ntot);
  Exc    = Exc_;


  // Correction energy
  Ecor = (Exc - EVxc) - Ehart - Eself + EIonSR + EVdw + Eext;

  // Helmholtz free energy
  
  if( eigSolPtr_->Ham().NumOccupiedState() == eigSolPtr_->Ham().NumStateTotal() 
      || esdfParam.temperature == 0.0 ){
    // Zero temperature
    Efree = Ekin + Ecor;
  }
  else{
    // Finite temperature
    Efree = 0.0;
    Real fermi = fermi_;
    Real Tbeta = Tbeta_;
    for(Int l=0; l< eigVal.m(); l++) {
      Real eig = eigVal(l);
      if( eig - fermi >= 0){
        Efree += -numSpin / Tbeta*log(1.0+exp(-Tbeta*(eig - fermi))); 
      }
      else{
        Efree += numSpin * (eig - fermi) - numSpin / Tbeta*log(1.0+exp(Tbeta*(eig-fermi)));
      }
    }
    Efree += Ecor + fermi * eigSolPtr_->Ham().NumOccupiedState() * numSpin; 
  }

  EfreeHarris_ = Efree;

  return ;
}         // -----  end of method SCF::CalculateHarrisEnergy  ----- 

void
SCF::AndersonMix    ( 
    Int iter,
    Real            mixStepLength,
    std::string     mixType,
    DblNumVec&      vMix,
    DblNumVec&      vOld,
    DblNumVec&      vNew,
    DblNumMat&      dfMat,
    DblNumMat&      dvMat ) {
  Int ntot  = eigSolPtr_->FFT().domain.NumGridTotalFine();
  if(0){  // old version
    // Residual 
    DblNumVec res;
    // Optimal input potential in Anderon mixing.
    DblNumVec vOpt; 
    // Optimal residual in Anderson mixing
    DblNumVec resOpt; 
    // Preconditioned optimal residual in Anderson mixing
    DblNumVec precResOpt;

    res.Resize(ntot);
    vOpt.Resize(ntot);
    resOpt.Resize(ntot);
    precResOpt.Resize(ntot);

    // Number of iterations used, iter should start from 1
    Int iterused = std::min( iter-1, mixMaxDim_ ); 
    // The current position of dfMat, dvMat
    Int ipos = iter - 1 - ((iter-2)/ mixMaxDim_ ) * mixMaxDim_;
    // The next position of dfMat, dvMat
    Int inext = iter - ((iter-1)/ mixMaxDim_) * mixMaxDim_;

    res = vOld;
    // res(:) = vOld(:) - vNew(:) is the residual
    blas::Axpy( ntot, -1.0, vNew.Data(), 1, res.Data(), 1 );

    vOpt = vOld;
    resOpt = res;

    if( iter > 1 ){
      // dfMat(:, ipos-1) = res(:) - dfMat(:, ipos-1);
      // dvMat(:, ipos-1) = vOld(:) - dvMat(:, ipos-1);
      blas::Scal( ntot, -1.0, dfMat.VecData(ipos-1), 1 );
      blas::Axpy( ntot, 1.0, res.Data(), 1, dfMat.VecData(ipos-1), 1 );
      blas::Scal( ntot, -1.0, dvMat.VecData(ipos-1), 1 );
      blas::Axpy( ntot, 1.0, vOld.Data(), 1, dvMat.VecData(ipos-1), 1 );


      // Calculating pseudoinverse
      Int nrow = iterused;
      DblNumMat dfMatTemp;
      DblNumVec gammas, S;

      Int rank;
      // FIXME Magic number
      Real rcond = 1e-12;
  
      S.Resize(nrow);
  
      gammas    = res;
      dfMatTemp = dfMat;
  
      lapack::SVDLeastSquare( ntot, iterused, 1, 
          dfMatTemp.Data(), ntot, gammas.Data(), ntot,
          S.Data(), rcond, &rank );
  
      Print( statusOFS, "  Rank of dfmat = ", rank );
      Print( statusOFS, "  Rcond = ", rcond );
      // Update vOpt, resOpt. 
  
      blas::Gemv('N', ntot, nrow, -1.0, dvMat.Data(),
          ntot, gammas.Data(), 1, 1.0, vOpt.Data(), 1 );
  
      blas::Gemv('N', ntot, iterused, -1.0, dfMat.Data(),
          ntot, gammas.Data(), 1, 1.0, resOpt.Data(), 1 );
    }

    if( mixType == "kerker+anderson" ){
      KerkerPrecond( precResOpt, resOpt );
    }
    else if( mixType == "anderson" ){
      precResOpt = resOpt;
    }
    else{
      ErrorHandling("Invalid mixing type.");
    }

    // Update dfMat, dvMat, vMix 
    // dfMat(:, inext-1) = res(:)
    // dvMat(:, inext-1) = vOld(:)
    blas::Copy( ntot, res.Data(), 1, 
        dfMat.VecData(inext-1), 1 );
    blas::Copy( ntot, vOld.Data(),  1, 
        dvMat.VecData(inext-1), 1 );

    // vMix(:) = vOpt(:) - mixStepLength * precRes(:)
    vMix = vOpt;
    blas::Axpy( ntot, -mixStepLength, precResOpt.Data(), 1, vMix.Data(), 1 );
  }
  else{
    int pos=((iter-1)%mixMaxDim_-1+mixMaxDim_)%mixMaxDim_;
    int next=(pos+1)%mixMaxDim_;
    statusOFS<< " pos " << pos << std::endl;
    statusOFS<< " next " << next << std::endl;

    blas::Axpy(ntot,-1.0,vOld.Data(),1,vNew.Data(),1); // F = Xout - Xin
    DblNumMat tempMat(ntot,2); // store this iteration's Xin and F
    blas::Copy(ntot,vOld.Data(),1,tempMat.VecData(0),1);
    blas::Copy(ntot,vNew.Data(),1,tempMat.VecData(1),1);
    if(iter>1){
      int dim=std::min(iter-1,mixMaxDim_);
      blas::Axpy(ntot,-1.0,vOld.Data(),1,dvMat.VecData(pos),1); // -\Delta X = Xin(i-1) - Xin(i)
      blas::Axpy(ntot,-1.0,vNew.Data(),1,dfMat.VecData(pos),1); // -\Delta F = F(i-1) - F(i)
      DblNumMat overlap(dim,dim);
      DblNumVec work(dim);
      for(int i=0;i<dim;i++){
        for(int j=i;j<dim;j++){
          overlap(i,j)=blas::Dot(ntot,dfMat.VecData(i),1,dfMat.VecData(j),1); // <\Delta F(i)|\Delta F(j)>
          overlap(j,i)=overlap(i,j);
        }
        work(i)=blas::Dot(ntot,dfMat.VecData(i),1,vNew.Data(),1); // -<\Delta F(i)|F>
      }
      // <\Delta F(i)|\Delta F(j)> * \gamma = <\Delta F(i)|F>
      // \gamma = - work
      lapack::Potrf('L',dim,overlap.Data(),dim);
      lapack::Potrs('L',dim,overlap.Data(),dim,I_ONE,work.Data(),dim);
      for(int i=0;i<dim;i++){
        blas::Axpy(ntot,-work(i),dvMat.VecData(i),1,vOld.Data(),1); // Xopt = Xin - \sum\gamma\Delta X
        blas::Axpy(ntot,-work(i),dfMat.VecData(i),1,vNew.Data(),1); // Fopt = F - \sum\gamma\Delta F
      }
    }
    for(int i=0;i<ntot;i++){
      vMix(i)=vOld(i)+mixStepLength*vNew(i); // Xin(i+1) = Xopt + \beta*Fopt
    }
    blas::Copy(ntot,tempMat.VecData(0),1,dvMat.VecData(next),1);
    blas::Copy(ntot,tempMat.VecData(1),1,dfMat.VecData(next),1);
  }

  return ;

}         // -----  end of method SCF::AndersonMix  ----- 

void
SCF::KerkerPrecond (
    DblNumVec&  precResidual,
    const DblNumVec&  residual )
{
  Fourier& fft = eigSolPtr_->FFT();
  Int ntot  = fft.domain.NumGridTotalFine();

  // NOTE Fixed KerkerB parameter
  //
  // From the point of view of the elliptic preconditioner
  //
  // (-\Delta + 4 * pi * b) r_p = -Delta r
  //
  // The Kerker preconditioner in the Fourier space is
  //
  // k^2 / (k^2 + 4 * pi * b)
  //
  // or using gkk = k^2 /2 
  //
  // gkk / ( gkk + 2 * pi * b )
  //
  // Here we choose KerkerB to be a fixed number.

  // FIXME hard coded
  Real KerkerB = 0.08; 
  Real Amin = 0.4;

  for (Int i=0; i<ntot; i++) {
    fft.inputComplexVecFine(i) = Complex(residual(i), 0.0);
  }

  FFTWExecute( fft, fft.forwardPlanFine );

  DblNumVec&  gkkFine = fft.gkkFine;

  for(Int i=0; i<ntot; i++) {
    // Procedure taken from VASP
    if( gkkFine(i) != 0 ){
      fft.outputComplexVecFine(i) *= gkkFine(i) / 
        ( gkkFine(i) + 2.0 * PI * KerkerB );
      //            fft.outputComplexVecFine(i) *= std::min(gkkFine(i) / 
      //                    ( gkkFine(i) + 2.0 * PI * KerkerB ), Amin);
    }
  }
  FFTWExecute ( fft, fft.backwardPlanFine );

  for (Int i=0; i<ntot; i++){
    precResidual(i) = fft.inputComplexVecFine(i).real();    
  }


  return ;
}         // -----  end of method SCF::KerkerPrecond  ----- 

void
SCF::BroydenMix    (
    Int iter,
    Real            mixStepLength,
    std::string     mixType,
    DblNumVec&      vMix,
    DblNumVec&      vOld,
    DblNumVec&      vNew,
    CpxNumMat&      dfMat,
    CpxNumMat&      dvMat,
    CpxNumMat&      cdfMat ) {
  Fourier& fft = eigSolPtr_->FFT();

  Int ntotFine = fft.domain.NumGridTotalFine();
  Int ntot = fft.domain.NumGridTotalFine();
  //Int ntot = fft.domain.NumGridTotal();
  // Transform vOld and vNew from real space to reciprocal space
  CpxNumVec GvOld, GvNew, GvMix;
  GvOld.Resize(ntot); GvNew.Resize(ntot); GvMix.Resize(ntot);
  SetValue( GvOld, Z_ZERO ); 
  SetValue( GvNew, Z_ZERO );
  SetValue( GvMix, Z_ZERO );

  for (Int i=0; i<ntotFine; i++) {
    fft.inputComplexVecFine(i) = Complex(vOld(i), D_ZERO); 
  }
  FFTWExecute( fft, fft.forwardPlanFine );

  for( Int i = 0; i < ntot; i++ ){
    GvOld(i) = fft.outputComplexVecFine(i);
  }
  
  for (Int i=0; i<ntotFine; i++) {
    fft.inputComplexVecFine(i) = Complex(vNew(i), D_ZERO);
  }
  FFTWExecute( fft, fft.forwardPlanFine );

  for( Int i = 0; i < ntot; i++ ){
    GvNew(i) = fft.outputComplexVecFine(i);
  }

  blas::Axpy( ntot, -1.0, GvOld.Data(), 1, GvNew.Data(), 1);

  Int iter_used = std::min( iter-1, mixMaxDim_ );
  Int ipos = iter - 1 - ((iter-2)/ mixMaxDim_ ) * mixMaxDim_;   

  if( iter > 1 ){
    blas::Copy( ntot, cdfMat.VecData(0), 1, dfMat.VecData(ipos-1), 1 );  
    blas::Axpy( ntot, -1.0, GvNew.Data(), 1, dfMat.VecData(ipos-1), 1);
    blas::Copy( ntot, cdfMat.VecData(1), 1, dvMat.VecData(ipos-1), 1 ); 
    blas::Axpy( ntot, -1.0, GvOld.Data(), 1, dvMat.VecData(ipos-1), 1);
  }

  blas::Copy( ntot, GvNew.Data(), 1, cdfMat.VecData(0), 1 );
  blas::Copy( ntot, GvOld.Data(), 1, cdfMat.VecData(1), 1 );
 
  if( iter_used > 0 ){
    DblNumMat betamix;
    betamix.Resize( iter_used, iter_used ); SetValue( betamix, D_ZERO );
    // Calculate matrix betamix   
    for (Int i=0; i<iter_used; i++) {
      for (Int j=i; j<iter_used; j++) {
        betamix(i,j) = this->RhoDdot( dfMat.VecData(j), dfMat.VecData(i) );
        betamix(j,i) = betamix(i,j);
      }
    }
    // Inverse betamix using the Bunch-Kaufman diagonal pivoting method
    IntNumVec iwork;
    iwork.Resize( iter_used ); SetValue( iwork, I_ZERO );
    
    lapack::Sytrf( 'U', iter_used, betamix.Data(), iter_used, iwork.Data() );
    lapack::Sytri( 'U', iter_used, betamix.Data(), iter_used, iwork.Data() );
    for (Int i=0; i<iter_used; i++) {
      for (Int j=i+1; j<iter_used; j++) {
        betamix(j,i) = betamix(i,j);
      }
    }
    
    DblNumVec work;
    Real gamma0 = D_ZERO;
    work.Resize( iter_used ); SetValue( work, D_ZERO );

    for (Int i=0; i<iter_used; i++){
      work(i) = this->RhoDdot( dfMat.VecData(i), GvNew.Data() );
    }

    for (Int i=0; i<iter_used; i++){
      gamma0 = blas::Dot( iter_used, betamix.VecData(i), 1, work.Data(), 1 );
      blas::Axpy( ntot, -gamma0, dvMat.VecData(i), 1, GvOld.Data(), 1);     
      blas::Axpy( ntot, -gamma0, dfMat.VecData(i), 1, GvNew.Data(), 1);
    }
  } // end of if( iter_used > 0 )
  
  // Transform vMix to real space
  GvMix = GvOld;
  blas::Axpy( ntot, mixStepLength, GvNew.Data(), 1, GvMix.Data(), 1 );
  SetValue( fft.outputComplexVecFine, Z_ZERO);
  for( Int i = 0; i < ntot; i++ ){
    fft.outputComplexVecFine(i) = GvMix(i);
  }

  FFTWExecute( fft, fft.backwardPlanFine );

  for( Int i = 0; i < ntotFine; i++ ){
    vMix(i) = fft.inputComplexVecFine(i).real();
  }
} // -----  end of method SCF::BroydenMix  -----
  
Real
SCF::RhoDdot    ( Complex* rho1, Complex* rho2 )
{
  Fourier& fft = eigSolPtr_->FFT();  
  Int ntot = fft.domain.NumGridTotalFine();

  Real rhodot = D_ZERO;
  Real fac = PI * fft.domain.Volume();
  
  for( Int i = 0; i < ntot; i++ ){ 
    if( fft.gkkFine(i) != 0 ){
      rhodot += (std::conj(rho1[i]) * rho2[i]).real() / fft.gkkFine(i);
    }
  }
  rhodot *= fac;
  return rhodot;
}     

void
SCF::PrintState    ( const Int iter  )
{
  Real HOMO, LUMO, EG;
  HOMO = eigSolPtr_->EigVal()(eigSolPtr_->Ham().NumOccupiedState()-1);
  if( eigSolPtr_->Ham().NumExtraState() > 0 ){
    LUMO = eigSolPtr_->EigVal()(eigSolPtr_->Ham().NumOccupiedState());
    EG = LUMO - HOMO;
  }
#if 0
  for(Int i = 0; i < eigSolPtr_->EigVal().m(); i++){
    Print(statusOFS, 
        "band#    = ", i, 
        "eigval   = ", eigSolPtr_->EigVal()(i),
        "resval   = ", eigSolPtr_->ResVal()(i),
        "occrate  = ", eigSolPtr_->Ham().OccupationRate()(i));
  }
  statusOFS << std::endl;
#endif

//  statusOFS 
//    << "NOTE:  Ecor  = Exc - EVxc - Ehart - Eself + EIonSR + EVdw + Eext" << std::endl
//    << "       Etot  = Ekin + Ecor" << std::endl
//    << "       Efree = Etot    + Entropy" << std::endl << std::endl;
  Print(statusOFS, "EfreeHarris       = ",  EfreeHarris_, "[au]");    
  Print(statusOFS, "Etot              = ",  Etot_, "[au]");
  Print(statusOFS, "Efree             = ",  Efree_, "[au]");
  Print(statusOFS, "Ekin              = ",  Ekin_, "[au]");
  Print(statusOFS, "Ehart             = ",  Ehart_, "[au]");
  Print(statusOFS, "Exc               = ",  Exc_, "[au]"); 
  Print(statusOFS, "EVdw              = ",  EVdw_, "[au]"); 
  Print(statusOFS, "Eself             = ",  Eself_, "[au]");
  Print(statusOFS, "EIonSR            = ",  EIonSR_, "[au]");
  Print(statusOFS, "Ecor              = ",  Ecor_, "[au]");
  Print(statusOFS, "Fermi             = ",  fermi_, "[au]");
  Print(statusOFS, "Total charge      = ",  totalCharge_, "[au]");
  Print(statusOFS, "HOMO              = ",  HOMO*au2ev, "[eV]");
  if( eigSolPtr_->Ham().NumExtraState() > 0 ){
    Print(statusOFS, "LUMO              = ",  LUMO*au2ev, "[eV]");
    Print(statusOFS, "Band Gap        = ",  EG*au2ev, "[eV]"); 
  }


  return ;
}         // -----  end of method SCF::PrintState  ----- 


void
SCF::UpdateMDParameters    ( )
{
  scfMaxIter_ = esdfParam.MDscfOuterMaxIter;
  scfPhiMaxIter_ = esdfParam.MDscfPhiMaxIter;
  return ;
}         // -----  end of method SCF::UpdateMDParameters  ----- 

void
SCF::UpdateTDDFTParameters    ( )
{
  //scfMaxIter_    = esdfParam.TDDFTscfOuterMaxIter;
  //scfPhiMaxIter_ = esdfParam.TDDFTscfPhiMaxIter;
  return ;
}         // -----  end of method SCF::UpdateTDDFTParameters  ----- 

} // namespace dgdft
