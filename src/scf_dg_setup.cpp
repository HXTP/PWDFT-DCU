/*
   Copyright (c) 2012 The Regents of the University of California,
   through Lawrence Berkeley National Laboratory.  

Authors: Lin Lin, Wei Hu, Amartya Banerjee, and Xinming Qin

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
/// @file scf_dg.cpp
/// @brief Self consistent iteration using the DG method.
/// @date 2013-02-05
/// @date 2014-08-06 Add intra-element parallelization.
//
/// Xinming Qin xmqin03@gmail.com
/// @date 2023-10-17 Add Hybrid DFT ALBs
/// @date 2023-11-05 Add HFX hamiltonian matrix
/// @date 2024-01-29 Add different mixing schemes 
// 
#include  "scf_dg.hpp"
#include  "blas.hpp"
#include  "lapack.hpp"
#include  "utility.hpp"
#include  "environment.hpp"
#ifdef ELSI
#include  "elsi.h"
#endif

namespace  dgdft{

using namespace dgdft::DensityComponent;
using namespace dgdft::esdf;
using namespace dgdft::scalapack;

// FIXME Leave the smoother function to somewhere more appropriate
Real Smoother ( Real x )
{
  Real t, z;
  if( x <= 0 )
    t = 1.0;
  else if( x >= 1 )
    t = 0.0;
  else{
    z = -1.0 / x + 1.0 / (1.0 - x );
    if( z < 0 )
      t = 1.0 / ( std::exp(z) + 1.0 );
    else
      t = std::exp(-z) / ( std::exp(-z) + 1.0 );
  }
  return t;
}        // -----  end of function Smoother  ----- 

SCFDG::SCFDG    (  )
{
  isPEXSIInitialized_ = false;
}        // -----  end of method SCFDG::SCFDG  ----- 

SCFDG::~SCFDG    (  )
{
  Int mpirank, mpisize;
  MPI_Comm_rank( domain_.comm, &mpirank );
  MPI_Comm_size( domain_.comm, &mpisize );
#ifdef _USE_PEXSI_
  if( isPEXSIInitialized_ == true ){
    Int info;
    PPEXSIPlanFinalize(plan_, &info);
    if( info != 0 ){
      std::ostringstream msg;
      msg 
        << "PEXSI finalization returns info " << info << std::endl;
      ErrorHandling( msg.str().c_str() );
    }

    MPI_Comm_free( &pexsiComm_ );
  }
#endif
}         // -----  end of method SCFDG::~SCFDG  ----- 

void
SCFDG::Setup    ( 
    HamiltonianDG&              hamDG,
    DistVec<Index3, EigenSolver, ElemPrtn>&  distEigSol,
    DistFourier&                distfft,
    PeriodTable&                ptable,
    Int                         contxt    )
{
  Real timeSta, timeEnd;

  // *********************************************************************
  // Read parameters from ESDFParam
  // *********************************************************************
  // Control parameters
  {
    domain_             = esdfParam.domain;
    mixMaxDim_          = esdfParam.mixMaxDim;
    HybridmixMaxDim_    = esdfParam.DGHybridmixMaxDim;
    OutermixVariable_   = esdfParam.OutermixVariable;
    InnermixVariable_   = esdfParam.InnermixVariable;
    HybridInnermixVariable_   = esdfParam.HybridInnermixVariable;
    mixType_            = esdfParam.mixType;
    HybridmixType_      = esdfParam.DGHybridmixType;
    DGHFXNestedLoop_    = esdfParam.DGHFXNestedLoop;
    mixStepLength_      = esdfParam.mixStepLength;
    HybridmixStepLength_   = esdfParam.DGHybridmixStepLength;
    eigMinTolerance_    = esdfParam.eigMinTolerance;
    eigTolerance_       = esdfParam.eigTolerance;
    eigMinIter_         = esdfParam.eigMinIter;
    eigMaxIter_         = esdfParam.eigMaxIter;

    DFTscfInnerTolerance_  = esdfParam.DFTscfInnerTolerance;
    DFTscfInnerMinIter_    = esdfParam.DFTscfInnerMinIter;
    DFTscfInnerMaxIter_    = esdfParam.DFTscfInnerMaxIter;
    DFTscfOuterTolerance_  = esdfParam.DFTscfOuterTolerance;
    DFTscfOuterMinIter_    = esdfParam.DFTscfOuterMinIter;
    DFTscfOuterMaxIter_    = esdfParam.DFTscfOuterMaxIter;
    DFTscfOuterEnergyTolerance_    = esdfParam.DFTscfOuterEnergyTolerance;
    HybridscfInnerTolerance_       = esdfParam.HybridscfInnerTolerance;

    scfHFXEnergyTolerance_          = esdfParam.DGHFXEnergyTolerance;
 
    HybridscfInnerMinIter_         = esdfParam.HybridscfInnerMinIter;
    HybridscfInnerMaxIter_         = esdfParam.HybridscfInnerMaxIter;
    HybridscfOuterTolerance_       = esdfParam.HybridscfOuterTolerance;
    HybridscfOuterMinIter_         = esdfParam.HybridscfOuterMinIter;
    HybridscfOuterMaxIter_         = esdfParam.HybridscfOuterMaxIter;
    HybridscfOuterEnergyTolerance_ = esdfParam.HybridscfOuterEnergyTolerance;

    scfInnerDMTolerance_          = esdfParam.scfInnerDMTolerance;   // yaml_double( "SCF_Inner_DM_Tolerance", 1e-6 );
    scfInnerHamTolerance_         = esdfParam.scfInnerHamTolerance;   //= yaml_double( "SCF_Inner_Ham_Tolerance", 1e-6 );



    Ehfx_               = 0.0; 

    PhiMaxIter_         = esdfParam.HybridPhiMaxIter;
    PhiTolerance_       = esdfParam.HybridPhiTolerance;

   
    numUnusedState_     = esdfParam.numUnusedState;
    SVDBasisTolerance_  = esdfParam.SVDBasisTolerance;
    solutionMethod_     = esdfParam.solutionMethod;
    diagSolutionMethod_ = esdfParam.diagSolutionMethod;

    // Choice of smearing scheme : Fermi-Dirac (FD) or Gaussian_Broadening (GB) or Methfessel-Paxton (MP)
    // Currently PEXSI only supports FD smearing, so GB or MP have to be used with diag type methods
    SmearingScheme_ = esdfParam.smearing_scheme;
    if(solutionMethod_ == "pexsi")
      SmearingScheme_ = "FD";

    if(SmearingScheme_ == "GB")
      MP_smearing_order_ = 0;
    else if(SmearingScheme_ == "MP")
      MP_smearing_order_ = 2;
    else
      MP_smearing_order_ = -1; // For safety

    PWSolver_           = esdfParam.PWSolver;

    // Chebyshev Filtering related parameters for PWDFT on extended element
    if(PWSolver_ == "CheFSI")
      Diag_SCF_PWDFT_by_Cheby_ = 1;
    else
      Diag_SCF_PWDFT_by_Cheby_ = 0;

    First_SCF_PWDFT_ChebyFilterOrder_ = esdfParam.First_SCF_PWDFT_ChebyFilterOrder;
    First_SCF_PWDFT_ChebyCycleNum_ = esdfParam.First_SCF_PWDFT_ChebyCycleNum;
    General_SCF_PWDFT_ChebyFilterOrder_ = esdfParam.General_SCF_PWDFT_ChebyFilterOrder;
    PWDFT_Cheby_use_scala_ = esdfParam.PWDFT_Cheby_use_scala;
    PWDFT_Cheby_apply_wfn_ecut_filt_ =  esdfParam.PWDFT_Cheby_apply_wfn_ecut_filt;

    // Using PPCG for PWDFT on extended element
    if(PWSolver_ == "PPCG" || PWSolver_ == "PPCGScaLAPACK")
      Diag_SCF_PWDFT_by_PPCG_ = 1;
    else
      Diag_SCF_PWDFT_by_PPCG_ = 0;

    Tbeta_            = esdfParam.Tbeta;
    Tsigma_           = 1.0 / Tbeta_;
    scaBlockSize_     = esdfParam.scaBlockSize;
    numElem_          = esdfParam.numElem;
    ecutWavefunction_ = esdfParam.ecutWavefunction;
    densityGridFactor_= esdfParam.densityGridFactor;
    LGLGridFactor_    = esdfParam.LGLGridFactor;
    distancePeriodize_= esdfParam.distancePeriodize;

    potentialBarrierW_  = esdfParam.potentialBarrierW;
    potentialBarrierS_  = esdfParam.potentialBarrierS;
    potentialBarrierR_  = esdfParam.potentialBarrierR;

    XCType_             = esdfParam.XCType;
    VDWType_            = esdfParam.VDWType;
  }

  // Variables related to Chebyshev Filtered SCF iterations for DG  
  {
    Diag_SCFDG_by_Cheby_ = esdfParam.Diag_SCFDG_by_Cheby; // Default: 0
    SCFDG_Cheby_use_ScaLAPACK_ = esdfParam.SCFDG_Cheby_use_ScaLAPACK; // Default: 0

    First_SCFDG_ChebyFilterOrder_ = esdfParam.First_SCFDG_ChebyFilterOrder; // Default 60
    First_SCFDG_ChebyCycleNum_ = esdfParam.First_SCFDG_ChebyCycleNum; // Default 5

    Second_SCFDG_ChebyOuterIter_ = esdfParam.Second_SCFDG_ChebyOuterIter; // Default = 3
    Second_SCFDG_ChebyFilterOrder_ = esdfParam.Second_SCFDG_ChebyFilterOrder; // Default = 60
    Second_SCFDG_ChebyCycleNum_ = esdfParam.Second_SCFDG_ChebyCycleNum; // Default 3 

    General_SCFDG_ChebyFilterOrder_ = esdfParam.General_SCFDG_ChebyFilterOrder; // Default = 60
    General_SCFDG_ChebyCycleNum_ = esdfParam.General_SCFDG_ChebyCycleNum; // Default 1

    Cheby_iondynamics_schedule_flag_ = 0;
    scfdg_ion_dyn_iter_ = 0;
  }

  // Variables related to Chebyshev polynomial filtered 
  // complementary subspace iteration strategy in DGDFT
  // Only accessed if CheFSI is in use 

  if(Diag_SCFDG_by_Cheby_ == 1)
  {
    SCFDG_use_comp_subspace_ = esdfParam.scfdg_use_chefsi_complementary_subspace;  // Default: 0

    SCFDG_comp_subspace_parallel_ = SCFDG_Cheby_use_ScaLAPACK_; // Use serial or parallel routine depending on early CheFSI steps

    // Syrk and Syr2k based updates, available in parallel routine only
    SCFDG_comp_subspace_syrk_ = esdfParam.scfdg_chefsi_complementary_subspace_syrk; 
    SCFDG_comp_subspace_syr2k_ = esdfParam.scfdg_chefsi_complementary_subspace_syr2k;

    // Safeguard to ensure that CS strategy is called only after atleast one general CheFSI cycle has been called
    // This allows the initial guess vectors to be copied
    if(  SCFDG_use_comp_subspace_ == 1 && Second_SCFDG_ChebyOuterIter_ < 2)
      Second_SCFDG_ChebyOuterIter_ = 2;

    SCFDG_comp_subspace_nstates_ = esdfParam.scfdg_complementary_subspace_nstates; // Defaults to a fraction of extra states

    SCFDG_CS_ioniter_regular_cheby_freq_ = esdfParam.scfdg_cs_ioniter_regular_cheby_freq; // Defaults to 20

    SCFDG_CS_bigger_grid_dim_fac_ = esdfParam.scfdg_cs_bigger_grid_dim_fac; // Defaults to 1;

    // LOBPCG for top states option
    SCFDG_comp_subspace_LOBPCG_iter_ = esdfParam.scfdg_complementary_subspace_lobpcg_iter; // Default = 15
    SCFDG_comp_subspace_LOBPCG_tol_ = esdfParam.scfdg_complementary_subspace_lobpcg_tol; // Default = 1e-8

    // CheFSI for top states option
    Hmat_top_states_use_Cheby_ = esdfParam.Hmat_top_states_use_Cheby;
    Hmat_top_states_ChebyFilterOrder_ = esdfParam.Hmat_top_states_ChebyFilterOrder; 
    Hmat_top_states_ChebyCycleNum_ = esdfParam.Hmat_top_states_ChebyCycleNum; 
    Hmat_top_states_Cheby_delta_fudge_ = 0.0;

    // Extra precaution : Inner LOBPCG only available in serial mode and syrk type updates only available in paralle mode
    if(SCFDG_comp_subspace_parallel_ == 1){
      Hmat_top_states_use_Cheby_ = 1; 
    }
    else
    { 
      SCFDG_comp_subspace_syrk_ = 0;
      SCFDG_comp_subspace_syr2k_ = 0;
    }

    SCFDG_comp_subspace_N_solve_ = hamDG.NumExtraState() + SCFDG_comp_subspace_nstates_;     
    SCFDG_comp_subspace_engaged_ = false;
  }
  else
  {
    SCFDG_use_comp_subspace_ = false;
    SCFDG_comp_subspace_engaged_ = false;
  }

  // Ionic iteration related parameters
  scfdg_ion_dyn_iter_ = 0; // Ionic iteration number
  useEnergySCFconvergence_ = 0; // Whether to use energy based SCF convergence
  md_scf_etot_diff_tol_ = esdfParam.MDscfEtotdiff; // Tolerance for SCF total energy for energy based SCF convergence
  md_scf_eband_diff_tol_ = esdfParam.MDscfEbanddiff; // Tolerance for SCF band energy for energy based SCF convergence

  md_scf_etot_ = 0.0;
  md_scf_etot_old_ = 0.0;
  md_scf_etot_diff_ = 0.0;
  md_scf_eband_ = 0.0;
  md_scf_eband_old_ = 0.0; 
  md_scf_eband_diff_ = 0.0;

  Int mpirank; MPI_Comm_rank( domain_.comm, &mpirank );
  Int mpisize; MPI_Comm_size( domain_.comm, &mpisize );
  Int mpirankRow;  MPI_Comm_rank(domain_.rowComm, &mpirankRow);
  Int mpisizeRow;  MPI_Comm_size(domain_.rowComm, &mpisizeRow);
  Int mpirankCol;  MPI_Comm_rank(domain_.colComm, &mpirankCol);
  Int mpisizeCol;  MPI_Comm_size(domain_.colComm, &mpisizeCol);

  dmCol_ = numElem_[0] * numElem_[1] * numElem_[2];
  dmRow_ = mpisize / dmCol_;

  numProcScaLAPACK_ = esdfParam.numProcScaLAPACK;

  // Initialize PEXSI
#ifdef _USE_PEXSI_
  if( solutionMethod_ == "pexsi" )
  {
    Int info;
    // Initialize the PEXSI options
    PPEXSISetDefaultOptions( &pexsiOptions_ );

    pexsiOptions_.temperature      = 1.0 / Tbeta_;
    pexsiOptions_.gap              = esdfParam.energyGap;
    pexsiOptions_.deltaE           = esdfParam.spectralRadius;
    pexsiOptions_.numPole          = esdfParam.numPole;
    pexsiOptions_.isInertiaCount   = 1; 
    pexsiOptions_.maxPEXSIIter     = esdfParam.maxPEXSIIter;
    pexsiOptions_.muMin0           = esdfParam.muMin;
    pexsiOptions_.muMax0           = esdfParam.muMax;
    pexsiOptions_.muInertiaTolerance = 
      esdfParam.muInertiaTolerance;
    pexsiOptions_.muInertiaExpansion = 
      esdfParam.muInertiaExpansion;
    pexsiOptions_.muPEXSISafeGuard   = 
      esdfParam.muPEXSISafeGuard;
    pexsiOptions_.numElectronPEXSITolerance = 
      esdfParam.numElectronPEXSITolerance;

    muInertiaToleranceTarget_ = esdfParam.muInertiaTolerance;
    numElectronPEXSIToleranceTarget_ = esdfParam.numElectronPEXSITolerance;

    pexsiOptions_.ordering           = esdfParam.matrixOrdering;
    pexsiOptions_.npSymbFact         = esdfParam.npSymbFact;
    pexsiOptions_.verbosity          = 1; // FIXME

    numProcRowPEXSI_     = esdfParam.numProcRowPEXSI;
    numProcColPEXSI_     = esdfParam.numProcColPEXSI;
    inertiaCountSteps_   = esdfParam.inertiaCountSteps;

    // Provide a communicator for PEXSI
    numProcPEXSICommCol_ = numProcRowPEXSI_ * numProcColPEXSI_;

    if( numProcPEXSICommCol_ > dmCol_ ){
      std::ostringstream msg;
      msg 
        << "In the current implementation, "
        << "the number of processors per pole = " << numProcPEXSICommCol_ 
        << ", and cannot exceed the number of elements = " << dmCol_ 
        << std::endl;
      ErrorHandling( msg.str().c_str() );
    }

    Int numProcPEXSICommRow_ = mpisize / numProcPEXSICommCol_;
    Int mpisizePEXSI = numProcPEXSICommRow_ * numProcPEXSICommCol_;
    numProcTotalPEXSI_ = numProcPEXSICommRow_ * numProcPEXSICommCol_;
    Int mpirank_tanspose = mpirankCol + mpirankRow * mpisizeCol;
    Int outputFileIndex;

    if(mpirank_tanspose == 0 ){
      outputFileIndex = 0;		  
    }
    else{
      outputFileIndex = -1;
    }

    Int isProcPEXSI = 1;
    Int mpirankPEXSI = mpirank_tanspose;

#if ( _DEBUGlevel_ >= 2 )
    statusOFS 
      << "mpirank = " << mpirank << std::endl
      << "mpisize = " << mpisize << std::endl
      << "mpirankRow = " << mpirankRow << std::endl
      << "mpirankCol = " << mpirankCol << std::endl
      << "mpisizeRow = " << mpisizeRow << std::endl
      << "mpisizeCol = " << mpisizeCol << std::endl
      << "outputFileIndex = " << outputFileIndex << std::endl
      << "mpirankPEXSI = " << mpirankPEXSI << std::endl
      << "numProcPEXSICommRow_ = " << numProcPEXSICommRow_ << std::endl
      << "numProcPEXSICommCol_ = " << numProcPEXSICommCol_ << std::endl
      << "isProcPEXSI = " << isProcPEXSI << std::endl
      << "mpisizePEXSI = " << numProcTotalPEXSI_ << std::endl
      << std::endl;
#endif
    MPI_Comm_split( domain_.comm, isProcPEXSI, mpirankPEXSI, &planComm_);

    plan_ = 
       PPEXSIPlanInitialize(
       planComm_,
       numProcRowPEXSI_,
       numProcColPEXSI_,
       outputFileIndex,
       &info );

    if( info != 0 ){
        std::ostringstream msg;
        msg 
          << "PEXSI initialization returns info " << info << std::endl;
        ErrorHandling( msg.str().c_str() );
    }
  }
#endif // _USE_PEXSI_

  // other SCFDG parameters
  {
    hamDGPtr_      = &hamDG;
    distEigSolPtr_ = &distEigSol;
    distfftPtr_    = &distfft;
    ptablePtr_     = &ptable;
    elemPrtn_      = distEigSol.Prtn();
    contxt_        = contxt;

    vtotLGLSave_.SetComm(domain_.colComm);
    vtotLGLSave_.Prtn()   = elemPrtn_;

    if( hamDG.IsHybrid() ){
      distDMMat_.SetComm(domain_.colComm);
      distHFXMat_.SetComm(domain_.colComm);
      distDMMat_.Prtn()     = hamDG.HMat().Prtn();
      distHFXMat_.Prtn()     = hamDG.HMat().Prtn();
     
      if( !DGHFXNestedLoop_ ){
        distDMMatSave_.SetComm(domain_.colComm);
        distHMatSave_.SetComm(domain_.colComm);
        distDMMatSave_.Prtn()     = hamDG.HMat().Prtn();
        distHMatSave_.Prtn()     = hamDG.HMat().Prtn();
      }
    }


    // For outer iteration mixing
    mixOuterSave_.SetComm(domain_.colComm);
    dfOuterMat_.SetComm(domain_.colComm);
    dvOuterMat_.SetComm(domain_.colComm);
    mixOuterSave_.Prtn()  = elemPrtn_;
    dfOuterMat_.Prtn()    = elemPrtn_;
    dvOuterMat_.Prtn()    = elemPrtn_;


    // Inner Mixing
    if( InnermixVariable_ == "potential" || InnermixVariable_ == "density" ){
      mixInnerSave_.SetComm(domain_.colComm);
      dfInnerMat_.SetComm(domain_.colComm);
      dvInnerMat_.SetComm(domain_.colComm);

      mixInnerSave_.Prtn()  = elemPrtn_;
      dfInnerMat_.Prtn()    = elemPrtn_;
      dvInnerMat_.Prtn()    = elemPrtn_;
     
      if( mixType_ == "broyden") {
        cdfInnerMat_.SetComm(domain_.colComm);
        cdfInnerMat_.Prtn()   = elemPrtn_;
      }

    }

#ifdef _USE_PEXSI_
//    distDMMat_.Prtn()      = hamDG.HMat().Prtn();
    distEDMMat_.Prtn()     = hamDG.HMat().Prtn();
    distFDMMat_.Prtn()     = hamDG.HMat().Prtn(); 
#endif
    if(SCFDG_use_comp_subspace_ == 1)
      distDMMat_.Prtn() = hamDG.HMat().Prtn();

    // The number of processors in the column communicator must be the
    // number of elements, and mpisize should be a multiple of the
    // number of elements.
    if( (mpisize % dmCol_) != 0 ){
      statusOFS << "mpisize = " << mpisize << " mpirank = " << mpirank << std::endl;
      statusOFS << "dmCol_ = " << dmCol_ << " dmRow_ = " << dmRow_ << std::endl;
      std::ostringstream msg;
      msg << "Total number of processors do not fit to the number processors per element." << std::endl;
      ErrorHandling( msg.str().c_str() );
    }


//  for(typename std::map<ElemMatKey, DblNumMat >::iterator
//    My_iterator = hamDG.HMat().LocalMap().begin();
//    My_iterator != hamDG.HMat().LocalMap().end();
//    ++ My_iterator )
//  {
//    ElemMatKey matkey = (*My_iterator).first;
//
//    std::map<ElemMatKey, DblNumMat>::iterator mi =
//        distHMatSave_.LocalMap().find( matkey );
//
//    if( mi == distHFXMat_.LocalMap().end() ){
//      distHFXMat_.LocalMap()[matkey] = emptyMat;
//    }
//    else{
//      DblNumMat&  mat = (*mi).second;
//      blas::Copy( emptyMat.Size(), emptyMat.Data(), 1,
//        mat.Data(), 1);
//    }


    // FIXME fixed ratio between the size of the extended element and
    // the element
    for( Int d = 0; d < DIM; d++ ){
      extElemRatio_[d] = ( numElem_[d]>1 ) ? 3 : 1;
    }

    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
        for( Int i = 0; i < numElem_[0]; i++ ){
          Index3 key( i, j, k );
          if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
            DblNumVec emptyLGLVec( hamDG.NumLGLGridElem().prod() );
            SetValue( emptyLGLVec, 0.0 );
            vtotLGLSave_.LocalMap()[key] = emptyLGLVec;
          }
    } // for (i)

    if( InnermixVariable_ == "potential" ||  InnermixVariable_ == "density"){
      for( Int k = 0; k < numElem_[2]; k++ )
        for( Int j = 0; j < numElem_[1]; j++ )
          for( Int i = 0; i < numElem_[0]; i++ ){
            Index3 key( i, j, k );
            if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
              DblNumVec  emptyVec( hamDG.NumUniformGridElemFine().prod() );
              SetValue( emptyVec, 0.0 );
              mixOuterSave_.LocalMap()[key] = emptyVec;
              mixInnerSave_.LocalMap()[key] = emptyVec;
              DblNumMat  emptyMat( hamDG.NumUniformGridElemFine().prod(), mixMaxDim_ );
              SetValue( emptyMat, 0.0 );
              DblNumMat  emptyMat2( hamDG.NumUniformGridElemFine().prod(), 2 );
              SetValue( emptyMat2, 0.0 );
              dfOuterMat_.LocalMap()[key]   = emptyMat;
              dvOuterMat_.LocalMap()[key]   = emptyMat;
              dfInnerMat_.LocalMap()[key]   = emptyMat;
              dvInnerMat_.LocalMap()[key]   = emptyMat;
              if( mixType_ == "broyden" ){
                cdfInnerMat_.LocalMap()[key] = emptyMat2;
              }
            } // own this element
      }  // for (i)
    } // mixVariable

    // Restart the density in the global domain
    restartDensityFileName_ = "DEN";
    // Restart the wavefunctions in the extended element
    restartWfnFileName_     = "WFNEXT";
  }

  // *********************************************************************
  // Initialization
  // *********************************************************************

  // Density
  DistDblNumVec&  density = hamDGPtr_->Density();

  density.SetComm(domain_.colComm);

  if( esdfParam.isRestartDensity ) {
    // Only the first processor column reads the matrix

#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Restarting density from DEN_ files." << std::endl;
#endif

    if( mpirankRow == 0 ){
      std::istringstream rhoStream;      
      SeparateRead( restartDensityFileName_, rhoStream, mpirankCol );

      Real sumDensityLocal = 0.0, sumDensity = 0.0;

      for( Int k = 0; k < numElem_[2]; k++ )
        for( Int j = 0; j < numElem_[1]; j++ )
          for( Int i = 0; i < numElem_[0]; i++ ){
            Index3 key( i, j, k );
            if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
              std::vector<DblNumVec> gridpos(DIM);

              // Dummy variables and not used
              for( Int d = 0; d < DIM; d++ ){
                deserialize( gridpos[d], rhoStream, NO_MASK );
              }

              Index3 keyRead;
              deserialize( keyRead, rhoStream, NO_MASK );
              if( keyRead[0] != key[0] ||
                  keyRead[1] != key[1] ||
                  keyRead[2] != key[2] ){
                std::ostringstream msg;
                msg 
                  << "Mpirank " << mpirank << " is reading the wrong file."
                  << std::endl
                  << "key     ~ " << key << std::endl
                  << "keyRead ~ " << keyRead << std::endl;
                ErrorHandling( msg.str().c_str() );
              }

              DblNumVec   denVecRead;
              DblNumVec&  denVec = density.LocalMap()[key];
              deserialize( denVecRead, rhoStream, NO_MASK );
              if( denVecRead.Size() != denVec.Size() ){
                std::ostringstream msg;
                msg 
                  << "The size of restarting density does not match with the current setup."  
                  << std::endl
                  << "input density size   ~ " << denVecRead.Size() << std::endl
                  << "current density size ~ " << denVec.Size()     << std::endl;
                ErrorHandling( msg.str().c_str() );
              }
              denVec = denVecRead;
              for( Int p = 0; p < denVec.Size(); p++ ){
                sumDensityLocal += denVec(p);
              }
            }
      } // for (i)

      // Rescale the density
      mpi::Allreduce( &sumDensityLocal, &sumDensity, 1, MPI_SUM,
          domain_.colComm );

      Print( statusOFS, "Restart density. Sum of density      = ", 
          sumDensity * domain_.Volume() / domain_.NumGridTotalFine() );
    } // mpirank == 0

    // Broadcast the density to the column
    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
        for( Int i = 0; i < numElem_[0]; i++ ){
          Index3 key( i, j, k );
          if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
            DblNumVec&  denVec = density.LocalMap()[key];
            MPI_Bcast( denVec.Data(), denVec.Size(), MPI_DOUBLE, 0, domain_.rowComm );
          }
    }

  } // else using the zero initial guess
  else {
    if( esdfParam.isUseAtomDensity ){
//      statusOFS << "Use superposition of atomic density as initial "
//        << "guess for electron density." << std::endl;

      GetTime( timeSta );
      hamDGPtr_->CalculateAtomDensity( *ptablePtr_, *distfftPtr_ );
      GetTime( timeEnd );
//      statusOFS << "Time for calculating the atomic density = " 
//        << timeEnd - timeSta << " [s]" << std::endl;

      for( Int k = 0; k < numElem_[2]; k++ )
        for( Int j = 0; j < numElem_[1]; j++ )
          for( Int i = 0; i < numElem_[0]; i++ ){
            Index3 key( i, j, k );
            if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
              DblNumVec&  denVec = density.LocalMap()[key];
              DblNumVec&  atomdenVec  = hamDGPtr_->AtomDensity().LocalMap()[key];
              blas::Copy( denVec.Size(), atomdenVec.Data(), 1, denVec.Data(), 1 );
            }
          } // for (i)
    }
    else{
      statusOFS << "Generating initial density through linear combination of pseudocharges." 
        << std::endl;
      // Initialize the electron density using the pseudocharge
      // make sure the pseudocharge is initialized
      DistDblNumVec& pseudoCharge = hamDGPtr_->PseudoCharge();

      pseudoCharge.SetComm(domain_.colComm);

      Real sumDensityLocal = 0.0, sumPseudoChargeLocal = 0.0;
      Real sumDensity, sumPseudoCharge;
      Real EPS = 1e-14;

      // make sure that the electron density is positive
      for( Int k = 0; k < numElem_[2]; k++ )
        for( Int j = 0; j < numElem_[1]; j++ )
          for( Int i = 0; i < numElem_[0]; i++ ){
            Index3 key( i, j, k );
            if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
              DblNumVec&  denVec = density.LocalMap()[key];
              DblNumVec&  ppVec  = pseudoCharge.LocalMap()[key];
              for( Int p = 0; p < denVec.Size(); p++ ){
                //                            denVec(p) = ppVec(p);
                denVec(p) = ( ppVec(p) > EPS ) ? ppVec(p) : D_ZERO;
                sumDensityLocal += denVec(p);
                sumPseudoChargeLocal += ppVec(p);
              }
            }
      } // for (i)

      // Rescale the density
      mpi::Allreduce( &sumDensityLocal, &sumDensity, 1, MPI_SUM,
          domain_.colComm );
      mpi::Allreduce( &sumPseudoChargeLocal, &sumPseudoCharge, 
          1, MPI_SUM, domain_.colComm );

      Print( statusOFS, "Initial density. Sum of density      = ", 
          sumDensity * domain_.Volume() / domain_.NumGridTotalFine() );
#if ( _DEBUGlevel_ >= 1 )
      Print( statusOFS, "Sum of pseudo charge        = ", 
          sumPseudoCharge * domain_.Volume() / domain_.NumGridTotalFine() );
#endif
      for( Int k = 0; k < numElem_[2]; k++ )
        for( Int j = 0; j < numElem_[1]; j++ )
          for( Int i = 0; i < numElem_[0]; i++ ){
            Index3 key( i, j, k );
            if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
              DblNumVec&  denVec = density.LocalMap()[key];
              blas::Scal( denVec.Size(), sumPseudoCharge / sumDensity, 
                  denVec.Data(), 1 );
            }
      } // for (i)
    }  // esdfParam.isUseAtomDensity
  } // Restart the density

  // Wavefunctions in the extended element
  if( esdfParam.isRestartWfn ){
    statusOFS << "Restarting basis functions from WFNEXT_ files"
      << std::endl;
    std::istringstream wfnStream;      
    SeparateRead( restartWfnFileName_, wfnStream, mpirank );

    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
        for( Int i = 0; i < numElem_[0]; i++ ){
          Index3 key( i, j, k );
          if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
            EigenSolver&  eigSol = distEigSolPtr_->LocalMap()[key];
            Spinor& psi = eigSol.Psi();
            DblNumTns& wavefun = psi.Wavefun();
            DblNumTns  wavefunRead;
            std::vector<DblNumVec> gridpos(DIM);
            for( Int d = 0; d < DIM; d++ ){
              deserialize( gridpos[d], wfnStream, NO_MASK );
            }

            Index3 keyRead;
            deserialize( keyRead, wfnStream, NO_MASK );
            if( keyRead[0] != key[0] ||
                keyRead[1] != key[1] ||
                keyRead[2] != key[2] ){
              std::ostringstream msg;
              msg 
                << "Mpirank " << mpirank << " is reading the wrong file."
                << std::endl
                << "key     ~ " << key << std::endl
                << "keyRead ~ " << keyRead << std::endl;
              ErrorHandling( msg.str().c_str() );
            }
            deserialize( wavefunRead, wfnStream, NO_MASK );

            if( wavefunRead.Size() != wavefun.Size() ){
              std::ostringstream msg;
              msg 
                << "The size of restarting basis function does not match with the current setup."  
                << std::endl
                << "input basis size   ~ " << wavefunRead.Size() << std::endl
                << "current basis size ~ " << wavefun.Size()     << std::endl;
              ErrorHandling( msg.str().c_str() );
            }

            wavefun = wavefunRead;
          }
    } // for (i)

  }  //esdfParam.isRestartWfn 
  else{ 
//    statusOFS << "Initial random basis functions in the extended element."
//      << std::endl;

    // Use random initial guess for basis functions in the extended element.
    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
        for( Int i = 0; i < numElem_[0]; i++ ){
          Index3 key( i, j, k );
          if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
            EigenSolver&  eigSol = distEigSolPtr_->LocalMap()[key];
            Spinor& psi = eigSol.Psi();

         // Wavefuns have been set in dgdft init, why reInit them in extended element here ?  
         // FIXME   xmqin
         //   UniformRandom( psi.Wavefun() );
            // For debugging purpose
            // Make sure that the initial wavefunctions in each element
            // are the same, when different number of processors are
            // used for intra-element parallelization.
            if(0){ 
              Spinor  psiTemp;
              psiTemp.Setup( eigSol.FFT().domain, 1, psi.NumStateTotal(), psi.NumStateTotal(), 0.0 );

              Int mpirankp, mpisizep;
              MPI_Comm_rank( domain_.rowComm, &mpirankp );
              MPI_Comm_size( domain_.rowComm, &mpisizep );

              if (mpirankp == 0){
                SetRandomSeed(1);
                UniformRandom( psiTemp.Wavefun() );
              }
              MPI_Bcast(psiTemp.Wavefun().Data(), psiTemp.Wavefun().m()*psiTemp.Wavefun().n()*psiTemp.Wavefun().p(), MPI_DOUBLE, 0, domain_.rowComm);

              Int size = psi.Wavefun().m() * psi.Wavefun().n();
              Int nocc = psi.Wavefun().p();

              IntNumVec& wavefunIdx = psi.WavefunIdx();
              NumTns<Real>& wavefun = psi.Wavefun();
             
              for (Int k=0; k<nocc; k++) {
                Real *ptr = psi.Wavefun().MatData(k);
                Real *ptr1 = psiTemp.Wavefun().MatData(wavefunIdx(k));
                for (Int i=0; i<size; i++) {
                  *ptr = *ptr1;
                  ptr = ptr + 1;
                  ptr1 = ptr1 + 1;
                }
              }
            } // random

           // xmqin add for occ HSE  //Check
            DblNumVec& occ = eigSol.Ham().OccupationRate();
            Int npsi = psi.NumStateTotal();
            Int nocc = eigSol.Ham().NumOccupiedState();

            if(nocc <= npsi) {
              occ.Resize( npsi );
              SetValue( occ, 0.0 );
              for( Int k = 0; k < nocc; k++ ){
                occ[k] = 1.0;
              }
            }
            else {
              std::ostringstream msg;
              msg
              << "number of ALBs is " << npsi << " is less than number of occupied orbitals " << nocc << " in this extended element "  
              << std::endl;
              ErrorHandling( msg.str().c_str() );
            }
          }
        } // for (i)
    Print( statusOFS, "Initial basis functions with random guess." );
  } // if (isRestartWfn_)

  // Generate the transfer matrix from the periodic uniform grid on each
  // extended element to LGL grid.  
  // 05/06/2015:
  // Based on the new understanding of the dual grid treatment, the
  // interpolation must be performed through a fine Fourier grid
  // (uniform grid) and then interpolate to the LGL grid.
  {
    PeriodicUniformToLGLMat_.resize(DIM);
    PeriodicUniformFineToLGLMat_.resize(DIM);
    PeriodicGridExtElemToGridElemMat_.resize(DIM);

    EigenSolver& eigSol = (*distEigSol.LocalMap().begin()).second;
    Domain dmExtElem = eigSol.FFT().domain;
    Domain dmElem;
    for( Int d = 0; d < DIM; d++ ){
      dmElem.length[d]   = domain_.length[d] / numElem_[d];
      dmElem.numGrid[d]  = domain_.numGrid[d] / numElem_[d];
      dmElem.numGridFine[d]  = domain_.numGridFine[d] / numElem_[d];
      // PosStart relative to the extended element 
      dmExtElem.posStart[d] = 0.0;
      dmElem.posStart[d] = ( numElem_[d] > 1 ) ? dmElem.length[d] : 0;
    }

    Index3 numLGL        = hamDG.NumLGLGridElem();
    Index3 numUniform    = dmExtElem.numGrid;
    Index3 numUniformFine    = dmExtElem.numGridFine;
    Index3 numUniformFineElem    = dmElem.numGridFine;
    Point3 lengthUniform = dmExtElem.length;

    std::vector<DblNumVec>  LGLGrid(DIM);
    LGLMesh( dmElem, numLGL, LGLGrid ); 
    std::vector<DblNumVec>  UniformGrid(DIM);
    UniformMesh( dmExtElem, UniformGrid );
    std::vector<DblNumVec>  UniformGridFine(DIM);
    UniformMeshFine( dmExtElem, UniformGridFine );
    std::vector<DblNumVec>  UniformGridFineElem(DIM);
    UniformMeshFine( dmElem, UniformGridFineElem );

    for( Int d = 0; d < DIM; d++ ){
      DblNumMat&  localMat = PeriodicUniformToLGLMat_[d];
      DblNumMat&  localMatFineElem = PeriodicGridExtElemToGridElemMat_[d];
      localMat.Resize( numLGL[d], numUniform[d] );
      localMatFineElem.Resize( numUniformFineElem[d], numUniform[d] );
      SetValue( localMat, 0.0 );
      SetValue( localMatFineElem, 0.0 );
      DblNumVec KGrid( numUniform[d] );
      for( Int i = 0; i <= numUniform[d] / 2; i++ ){
        KGrid(i) = i * 2.0 * PI / lengthUniform[d];
      }
      for( Int i = numUniform[d] / 2 + 1; i < numUniform[d]; i++ ){
        KGrid(i) = ( i - numUniform[d] ) * 2.0 * PI / lengthUniform[d];
      }

      for( Int j = 0; j < numUniform[d]; j++ ){

        for( Int i = 0; i < numLGL[d]; i++ ){
          localMat(i, j) = 0.0;
          for( Int k = 0; k < numUniform[d]; k++ ){
            localMat(i,j) += std::cos( KGrid(k) * ( LGLGrid[d](i) -
                  UniformGrid[d](j) ) ) / numUniform[d];
          } // for (k)
        } // for (i)

        for( Int i = 0; i < numUniformFineElem[d]; i++ ){
          localMatFineElem(i, j) = 0.0;
          for( Int k = 0; k < numUniform[d]; k++ ){
            localMatFineElem(i,j) += std::cos( KGrid(k) * ( UniformGridFineElem[d](i) -
                  UniformGrid[d](j) ) ) / numUniform[d];
          } // for (k)
        } // for (i)
      } // for (j)
    } // for (d)


    for( Int d = 0; d < DIM; d++ ){
      DblNumMat&  localMatFine = PeriodicUniformFineToLGLMat_[d];
      localMatFine.Resize( numLGL[d], numUniformFine[d] );
      SetValue( localMatFine, 0.0 );
      DblNumVec KGridFine( numUniformFine[d] );
      for( Int i = 0; i <= numUniformFine[d] / 2; i++ ){
        KGridFine(i) = i * 2.0 * PI / lengthUniform[d];
      }
      for( Int i = numUniformFine[d] / 2 + 1; i < numUniformFine[d]; i++ ){
        KGridFine(i) = ( i - numUniformFine[d] ) * 2.0 * PI / lengthUniform[d];
      }

      for( Int j = 0; j < numUniformFine[d]; j++ ){

        for( Int i = 0; i < numLGL[d]; i++ ){
          localMatFine(i, j) = 0.0;
          for( Int k = 0; k < numUniformFine[d]; k++ ){
            localMatFine(i,j) += std::cos( KGridFine(k) * ( LGLGrid[d](i) -
                  UniformGridFine[d](j) ) ) / numUniformFine[d];
          } // for (k)
        } // for (i)

      } // for (j)
    } // for (d)

    // Assume the initial error is O(1)
    scfOuterNorm_ = 1.0;
    scfInnerNorm_ = 1.0;
    scfInnerDMMaxDif_ = 1.0;
    scfInnerHamMaxDif_ = 1.0 ;
    

#if ( _DEBUGlevel_ >= 2 )
    statusOFS << "PeriodicUniformToLGLMat[0] = "
      << PeriodicUniformToLGLMat_[0] << std::endl;
    statusOFS << "PeriodicUniformToLGLMat[1] = " 
      << PeriodicUniformToLGLMat_[1] << std::endl;
    statusOFS << "PeriodicUniformToLGLMat[2] = "
      << PeriodicUniformToLGLMat_[2] << std::endl;
    statusOFS << "PeriodicUniformFineToLGLMat[0] = "
      << PeriodicUniformFineToLGLMat_[0] << std::endl;
    statusOFS << "PeriodicUniformFineToLGLMat[1] = " 
      << PeriodicUniformFineToLGLMat_[1] << std::endl;
    statusOFS << "PeriodicUniformFineToLGLMat[2] = "
      << PeriodicUniformFineToLGLMat_[2] << std::endl;
    statusOFS << "PeriodicGridExtElemToGridElemMat[0] = "
      << PeriodicGridExtElemToGridElemMat_[0] << std::endl;
    statusOFS << "PeriodicGridExtElemToGridElemMat[1] = "
      << PeriodicGridExtElemToGridElemMat_[1] << std::endl;
    statusOFS << "PeriodicGridExtElemToGridElemMat[2] = "
      << PeriodicGridExtElemToGridElemMat_[2] << std::endl;
#endif
  }

  // Whether to apply potential barrier in the extended element. CANNOT
  // be used together with periodization option
  if( esdfParam.isPotentialBarrier ) {
    vBarrier_.resize(DIM);
    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
        for( Int i = 0; i < numElem_[0]; i++ ){
          Index3 key( i, j, k );
          if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
            Domain& dmExtElem = distEigSolPtr_->LocalMap()[key].FFT().domain;
            std::vector<DblNumVec> gridpos(DIM);
            UniformMeshFine ( dmExtElem, gridpos );

            for( Int d = 0; d < DIM; d++ ){
              Real length   = dmExtElem.length[d];
              Int numGridFine   = dmExtElem.numGridFine[d];
              Real posStart = dmExtElem.posStart[d]; 
              Real center   = posStart + length / 2.0;

              // FIXME
              Real EPS      = 1.0;           // For stability reason
              Real dist;

              vBarrier_[d].Resize( numGridFine );
              SetValue( vBarrier_[d], 0.0 );
              for( Int p = 0; p < numGridFine; p++ ){
                dist = std::abs( gridpos[d][p] - center );
                // Only apply the barrier for region outside barrierR
                if( dist > potentialBarrierR_){
                  vBarrier_[d][p] = potentialBarrierS_* std::exp( - potentialBarrierW_ / 
                      ( dist - potentialBarrierR_ ) ) / std::pow( dist - length / 2.0 - EPS, 2.0 );
                }
              }
            } // for (d)

#if ( _DEBUGlevel_ >= 2  )
            statusOFS << "gridpos[0] = " << std::endl << gridpos[0] << std::endl;
            statusOFS << "gridpos[1] = " << std::endl << gridpos[1] << std::endl;
            statusOFS << "gridpos[2] = " << std::endl << gridpos[2] << std::endl;
            statusOFS << "vBarrier[0] = " << std::endl << vBarrier_[0] << std::endl;
            statusOFS << "vBarrier[1] = " << std::endl << vBarrier_[1] << std::endl;
            statusOFS << "vBarrier[2] = " << std::endl << vBarrier_[2] << std::endl;
#endif
          } // own this element
    } // for (k)
  }

  // Whether to periodize the potential in the extended element. CANNOT
  // be used together with barrier option.
  if( esdfParam.isPeriodizePotential ){
    vBubble_.resize(DIM);
    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
        for( Int i = 0; i < numElem_[0]; i++ ){
          Index3 key( i, j, k );
          if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
            Domain& dmExtElem = distEigSolPtr_->LocalMap()[key].FFT().domain;
            std::vector<DblNumVec> gridpos(DIM);
            UniformMeshFine ( dmExtElem, gridpos );

            for( Int d = 0; d < DIM; d++ ){
              Real length   = dmExtElem.length[d];
              Int numGridFine   = dmExtElem.numGridFine[d];
              Real posStart = dmExtElem.posStart[d]; 
              // FIXME
              Real EPS = 0.2; // Criterion for distancePeriodize_
              vBubble_[d].Resize( numGridFine );
              SetValue( vBubble_[d], 1.0 );

              if( distancePeriodize_[d] > EPS ){
                Real lb = posStart + distancePeriodize_[d];
                Real rb = posStart + length - distancePeriodize_[d];
                for( Int p = 0; p < numGridFine; p++ ){
                  if( gridpos[d][p] > rb ){
                    vBubble_[d][p] = Smoother( (gridpos[d][p] - rb ) / 
                        (distancePeriodize_[d] - EPS) );
                  }

                  if( gridpos[d][p] < lb ){
                    vBubble_[d][p] = Smoother( (lb - gridpos[d][p] ) / 
                        (distancePeriodize_[d] - EPS) );
                  }
                }
              }
            } // for (d)

#if ( _DEBUGlevel_ >= 2  )
            statusOFS << "gridpos[0] = " << std::endl << gridpos[0] << std::endl;
            statusOFS << "gridpos[1] = " << std::endl << gridpos[1] << std::endl;
            statusOFS << "gridpos[2] = " << std::endl << gridpos[2] << std::endl;
            statusOFS << "vBubble[0] = " << std::endl << vBubble_[0] << std::endl;
            statusOFS << "vBubble[1] = " << std::endl << vBubble_[1] << std::endl;
            statusOFS << "vBubble[2] = " << std::endl << vBubble_[2] << std::endl;
#endif
          } // own this element
    } // for (k)
  }

  // Initial value
  efreeDifPerAtom_ = 100.0;

#ifdef ELSI
  // ELSI interface initilization for ELPA
  if((diagSolutionMethod_ == "elpa") && ( solutionMethod_ == "diag" ))
  {
    // Step 1. init the ELSI interface 
    Int Solver = 1;      // 1: ELPA, 2: LibSOMM 3: PEXSI for dense matrix, default to use ELPA
    Int parallelism = 1; // 1 for multi-MPIs 
    Int storage = 0;     // ELSI only support DENSE(0) 
    Int sizeH = hamDG.NumBasisTotal(); 
    Int n_states = hamDG.NumOccupiedState();

    Int n_electrons = 2.0* n_states;
    statusOFS << std::endl<<" Done Setting up ELSI iterface " 
              << Solver << " " << sizeH << " " << n_states
              << std::endl<<std::endl;

    c_elsi_init(Solver, parallelism, storage, sizeH, n_electrons, n_states);

    // Step 2.  setup MPI Domain
    MPI_Comm newComm;
    MPI_Comm_split(domain_.comm, contxt, mpirank, &newComm);
    Int comm = MPI_Comm_c2f(newComm);
    c_elsi_set_mpi(comm); 

    // step 3: setup blacs for elsi. 

    if(contxt >= 0)
      c_elsi_set_blacs(contxt, scaBlockSize_);   

    //  customize the ELSI interface to use identity matrix S
    c_elsi_customize(0, 1, 1.0E-8, 1, 0, 0); 

    // use ELPA 2 stage solver
    c_elsi_customize_elpa(2); 
  }

  if( solutionMethod_ == "pexsi" ){
    Int Solver = 3;      // 1: ELPA, 2: LibSOMM 3: PEXSI for dense matrix, default to use ELPA
    Int parallelism = 1; // 1 for multi-MPIs 
    Int storage = 1;     // PEXSI only support sparse(1) 
    Int sizeH = hamDG.NumBasisTotal(); 
    Int n_states = hamDG.NumOccupiedState();
    Int n_electrons = 2.0* n_states;

    statusOFS << std::endl<<" Done Setting up ELSI iterface " 
              << std::endl << " sizeH " << sizeH 
              << std::endl << " n_electron " << n_electrons
              << std::endl << " n_states "  << n_states
              << std::endl<<std::endl;

    c_elsi_init(Solver, parallelism, storage, sizeH, n_electrons, n_states);

    Int comm = MPI_Comm_c2f(pexsiComm_);
    c_elsi_set_mpi(comm); 

    c_elsi_customize(1, 1, 1.0E-8, 1, 0, 0); 
  }
#endif

  // Need density gradient for semilocal XC functionals,  xmqin
  {
    isCalculateGradRho_ = false;
    if( esdfParam.XCType == "XC_GGA_XC_PBE" ||
      esdfParam.XCType == "XC_HYB_GGA_XC_HSE06" ||
      esdfParam.XCType == "XC_HYB_GGA_XC_PBEH" ) {
      isCalculateGradRho_ = true;
    }
  }

  return ;
}         // -----  end of method SCFDG::Setup  ----- 

void
SCFDG::Update    ( )
{
  Int mpirank, mpisize;
  MPI_Comm_rank( domain_.comm, &mpirank );
  MPI_Comm_size( domain_.comm, &mpisize );

  HamiltonianDG& hamDG = *hamDGPtr_;

  {
    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
        for( Int i = 0; i < numElem_[0]; i++ ){
          Index3 key( i, j, k );
          if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
            DblNumVec  emptyVec( hamDG.NumUniformGridElemFine().prod() );
            SetValue( emptyVec, 0.0 );
            mixOuterSave_.LocalMap()[key] = emptyVec;
            mixInnerSave_.LocalMap()[key] = emptyVec;
            DblNumMat  emptyMat( hamDG.NumUniformGridElemFine().prod(), mixMaxDim_ );
            SetValue( emptyMat, 0.0 );
            dfOuterMat_.LocalMap()[key]   = emptyMat;
            dvOuterMat_.LocalMap()[key]   = emptyMat;
            dfInnerMat_.LocalMap()[key]   = emptyMat;
            dvInnerMat_.LocalMap()[key]   = emptyMat;

            DblNumVec  emptyLGLVec( hamDG.NumLGLGridElem().prod() );
            SetValue( emptyLGLVec, 0.0 );
            vtotLGLSave_.LocalMap()[key] = emptyLGLVec;
          } // own this element
    }  // for (i)
  }

  return ;
}         // -----  end of method SCFDG::Update  ----- 

// This routine calculates the full density matrix
void SCFDG::scfdg_compute_fullDM()
{
  Int mpirank, mpisize;
  MPI_Comm_rank( domain_.comm, &mpirank );
  MPI_Comm_size( domain_.comm, &mpisize );

  HamiltonianDG&  hamDG = *hamDGPtr_;
  std::vector<Index3>  getKeys_list;

  DistDblNumMat& my_dist_mat = hamDG.EigvecCoef();

  // Check that vectors provided only contain one entry in the local map
  // This is a safeguard to ensure that we are really dealing with distributed matrices
  if((my_dist_mat.LocalMap().size() != 1))
  {
    statusOFS << std::endl << " Eigenvector not formatted correctly !!"
      << std::endl << " Aborting ... " << std::endl;
    exit(1);
  }

  // Copy eigenvectors to temp bufer
  DblNumMat &eigvecs_local = (hamDG.EigvecCoef().LocalMap().begin())->second;

  DblNumMat scal_local_eig_vec;
  scal_local_eig_vec.Resize(eigvecs_local.m(), eigvecs_local.n());
  blas::Copy((eigvecs_local.m() * eigvecs_local.n()), eigvecs_local.Data(), 1, scal_local_eig_vec.Data(), 1);

  // Scale temp buffer by occupation * numspin
  for(int iter_scale = 0; iter_scale < eigvecs_local.n(); iter_scale ++)
  {
    blas::Scal(  scal_local_eig_vec.m(),  
        (hamDG.NumSpin()* hamDG.OccupationRate()[iter_scale]), 
        scal_local_eig_vec.Data() + iter_scale * scal_local_eig_vec.m(), 1 );
  }

  // Obtain key based on my_dist_mat : This assumes that my_dist_mat is formatted correctly
  // based on processor number, etc.
  Index3 key = (my_dist_mat.LocalMap().begin())->first;

  // Obtain keys of neighbors using the Hamiltonian matrix
  for(typename std::map<ElemMatKey, DblNumMat >::iterator 
      get_neighbors_from_Ham_iterator = hamDG.HMat().LocalMap().begin();
      get_neighbors_from_Ham_iterator != hamDG.HMat().LocalMap().end();
      get_neighbors_from_Ham_iterator ++)
  {
    Index3 neighbor_key = (get_neighbors_from_Ham_iterator->first).second;

    if(neighbor_key == key)
      continue;
    else
      getKeys_list.push_back(neighbor_key);
  }

  // Do the communication necessary to get the information from
  // procs holding the neighbors
  my_dist_mat.GetBegin( getKeys_list, NO_MASK ); 
  my_dist_mat.GetEnd( NO_MASK );

  // First compute the diagonal block
  {
    DblNumMat &mat_local = my_dist_mat.LocalMap()[key];
    ElemMatKey diag_block_key = std::make_pair(key, key);

    // Compute the X*X^T portion
    distDMMat_.LocalMap()[diag_block_key].Resize( mat_local.m(),  mat_local.m());

    blas::Gemm( 'N', 'T', mat_local.m(), mat_local.m(), mat_local.n(),
        1.0, 
        scal_local_eig_vec.Data(), scal_local_eig_vec.m(), 
        mat_local.Data(), mat_local.m(),
        0.0, 
        distDMMat_.LocalMap()[diag_block_key].Data(),  mat_local.m());
  }

  // Now handle the off-diagonal blocks
  {
    DblNumMat &mat_local = my_dist_mat.LocalMap()[key];
    for(Int off_diag_iter = 0; off_diag_iter < getKeys_list.size(); off_diag_iter ++)
    {
      DblNumMat &mat_neighbor = my_dist_mat.LocalMap()[getKeys_list[off_diag_iter]];
      ElemMatKey off_diag_key = std::make_pair(key, getKeys_list[off_diag_iter]);

      // First compute the Xi * Xj^T portion
      distDMMat_.LocalMap()[off_diag_key].Resize( scal_local_eig_vec.m(),  mat_neighbor.m());

      blas::Gemm( 'N', 'T', scal_local_eig_vec.m(), mat_neighbor.m(), scal_local_eig_vec.n(),
          1.0, 
          scal_local_eig_vec.Data(), scal_local_eig_vec.m(), 
          mat_neighbor.Data(), mat_neighbor.m(),
          0.0, 
          distDMMat_.LocalMap()[off_diag_key].Data(),  mat_local.m());
    }
  }

  // Need to clean up extra entries in my_dist_mat
  typename std::map<Index3, DblNumMat >::iterator it;
  for(Int delete_iter = 0; delete_iter <  getKeys_list.size(); delete_iter ++)
  {
    it = my_dist_mat.LocalMap().find(getKeys_list[delete_iter]);
    (my_dist_mat.LocalMap()).erase(it);
  }

  return;
} // calculate density matrix

void
SCFDG::UpdateElemLocalPotential    (  )
{
  Int mpirank, mpisize;
  MPI_Comm_rank( domain_.comm, &mpirank );
  MPI_Comm_size( domain_.comm, &mpisize );

  HamiltonianDG&  hamDG = *hamDGPtr_;
  // vtot gather the neighborhood
  DistDblNumVec&  vtot = hamDG.Vtot();

  std::set<Index3> neighborSet;
  for( Int k = 0; k < numElem_[2]; k++ )
    for( Int j = 0; j < numElem_[1]; j++ )
      for( Int i = 0; i < numElem_[0]; i++ ){
        Index3 key( i, j, k );
        if( elemPrtn_.Owner(key) == (mpirank / dmRow_) ){
          std::vector<Index3>   idx(3);

          for( Int d = 0; d < DIM; d++ ){
            // Previous
            if( key[d] == 0 ) 
              idx[0][d] = numElem_[d]-1; 
            else 
              idx[0][d] = key[d]-1;

            // Current
            idx[1][d] = key[d];

            // Next
            if( key[d] == numElem_[d]-1) 
              idx[2][d] = 0;
            else
              idx[2][d] = key[d] + 1;
          } // for (d)

          // Tensor product 
          for( Int c = 0; c < 3; c++ )
            for( Int b = 0; b < 3; b++ )
              for( Int a = 0; a < 3; a++ ){
                // Not the element key itself
                if( idx[a][0] != i || idx[b][1] != j || idx[c][2] != k ){
                  neighborSet.insert( Index3( idx[a][0], idx[b][1], idx[c][2] ) );
                }
              } // for (a)
        } // own this element
      } // for (i)
  std::vector<Index3>  neighborIdx;
  neighborIdx.insert( neighborIdx.begin(), neighborSet.begin(), neighborSet.end() );

  // communicate
  vtot.Prtn()   = elemPrtn_;
  vtot.SetComm(domain_.colComm);
  vtot.GetBegin( neighborIdx, NO_MASK );
  vtot.GetEnd( NO_MASK );

  // Update of the local potential in each extended element locally.
  // The nonlocal potential does not need to be updated
  //
  // Also update the local potential on the LGL grid in hamDG.
  //
  // NOTE:
  //
  // 1. It is hard coded that the extended element is 1 or 3
  // times the size of the element
  for( Int k = 0; k < numElem_[2]; k++ )
    for( Int j = 0; j < numElem_[1]; j++ )
      for( Int i = 0; i < numElem_[0]; i++ ){
        Index3 key( i, j, k );
        if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
          EigenSolver&  eigSol = distEigSolPtr_->LocalMap()[key];
          // Skip the calculation if there is no adaptive local
          // basis function.  
          if( eigSol.Psi().NumState() == 0 )
            continue;

          Hamiltonian&  hamExtElem  = eigSol.Ham();
          DblNumVec&    vtotExtElem = hamExtElem.Vtot();
          SetValue( vtotExtElem, 0.0 );

          Index3 numGridElem = hamDG.NumUniformGridElemFine();
          Index3 numGridExtElem = eigSol.FFT().domain.numGridFine;

          // Update the potential in the extended element
          for(std::map<Index3, DblNumVec>::iterator 
              mi = vtot.LocalMap().begin();
              mi != vtot.LocalMap().end(); ++mi ){
            Index3      keyElem = (*mi).first;
            DblNumVec&  vtotElem = (*mi).second;
            // Determine the shiftIdx which maps the position of vtotElem to 
            // vtotExtElem
            Index3 shiftIdx;
            for( Int d = 0; d < DIM; d++ ){
              shiftIdx[d] = keyElem[d] - key[d];
              shiftIdx[d] = shiftIdx[d] - IRound( Real(shiftIdx[d]) / 
                  numElem_[d] ) * numElem_[d];
              // FIXME Adjustment  
              if( numElem_[d] > 1 ) shiftIdx[d] ++;
              shiftIdx[d] *= numGridElem[d];
            }

            Int ptrExtElem, ptrElem;
            for( Int k = 0; k < numGridElem[2]; k++ )
              for( Int j = 0; j < numGridElem[1]; j++ )
                for( Int i = 0; i < numGridElem[0]; i++ ){
                  ptrExtElem = (shiftIdx[0] + i) + 
                    ( shiftIdx[1] + j ) * numGridExtElem[0] +
                    ( shiftIdx[2] + k ) * numGridExtElem[0] * numGridExtElem[1];
                  ptrElem    = i + j * numGridElem[0] + k * numGridElem[0] * numGridElem[1];
                  vtotExtElem( ptrExtElem ) = vtotElem( ptrElem );
                } // for (i)
          } // for (mi)

          // Loop over the neighborhood

        } // own this element
      } // for (i)

  // Clean up vtot not owned by this element
  std::vector<Index3>  eraseKey;
  for( std::map<Index3, DblNumVec>::iterator 
      mi  = vtot.LocalMap().begin();
      mi != vtot.LocalMap().end(); ++mi ){
    Index3 key = (*mi).first;
    if( vtot.Prtn().Owner(key) != (mpirank / dmRow_) ){
      eraseKey.push_back( key );
    }
  }

  for( std::vector<Index3>::iterator vi = eraseKey.begin();
      vi != eraseKey.end(); ++vi ){
    vtot.LocalMap().erase( *vi );
  }

  // Modify the potential in the extended element.  Current options are
  //
  // 1. Add barrier
  // 2. Periodize the potential
  //
  // Numerical results indicate that option 2 seems to be better.
  for( Int k = 0; k < numElem_[2]; k++ )
    for( Int j = 0; j < numElem_[1]; j++ )
      for( Int i = 0; i < numElem_[0]; i++ ){
        Index3 key( i, j, k );
        if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
          EigenSolver&  eigSol = distEigSolPtr_->LocalMap()[key];
          Index3 numGridExtElemFine = eigSol.FFT().domain.numGridFine;

          // Add the external barrier potential. CANNOT be used
          // together with periodization option
          if( esdfParam.isPotentialBarrier ){
            Domain& dmExtElem = eigSol.FFT().domain;
            DblNumVec& vext = eigSol.Ham().Vext();
            SetValue( vext, 0.0 );
            for( Int gk = 0; gk < dmExtElem.numGridFine[2]; gk++)
              for( Int gj = 0; gj < dmExtElem.numGridFine[1]; gj++ )
                for( Int gi = 0; gi < dmExtElem.numGridFine[0]; gi++ ){
                  Int idx = gi + gj * dmExtElem.numGridFine[0] + 
                    gk * dmExtElem.numGridFine[0] * dmExtElem.numGridFine[1];
                  vext[idx] = vBarrier_[0][gi] + vBarrier_[1][gj] + vBarrier_[2][gk];
                } // for (gi)
            // NOTE:
            // Directly modify the vtot.  vext is not used in the
            // matrix-vector multiplication in the eigensolver.
            blas::Axpy( numGridExtElemFine.prod(), 1.0, eigSol.Ham().Vext().Data(), 1,
                eigSol.Ham().Vtot().Data(), 1 );
          }

          // Periodize the external potential. CANNOT be used together
          // with the barrier potential option
          if( esdfParam.isPeriodizePotential ){
            Domain& dmExtElem = eigSol.FFT().domain;
            // Get the potential
            DblNumVec& vext = eigSol.Ham().Vext();
            DblNumVec& vtot = eigSol.Ham().Vtot();

            // Find the max of the potential in the extended element
            Real vtotMax = *std::max_element( &vtot[0], &vtot[0] + vtot.Size() );
            Real vtotAvg = 0.0;
            for(Int i = 0; i < vtot.Size(); i++){
              vtotAvg += vtot[i];
            }
            vtotAvg /= Real(vtot.Size());
            Real vtotMin = *std::min_element( &vtot[0], &vtot[0] + vtot.Size() );

            SetValue( vext, 0.0 );
            for( Int gk = 0; gk < dmExtElem.numGridFine[2]; gk++)
              for( Int gj = 0; gj < dmExtElem.numGridFine[1]; gj++ )
                for( Int gi = 0; gi < dmExtElem.numGridFine[0]; gi++ ){
                  Int idx = gi + gj * dmExtElem.numGridFine[0] + 
                    gk * dmExtElem.numGridFine[0] * dmExtElem.numGridFine[1];
                  // Bring the potential to the vacuum level
                  vext[idx] = ( vtot[idx] - 0.0 ) * 
                    ( vBubble_[0][gi] * vBubble_[1][gj] * vBubble_[2][gk] - 1.0 );
                } // for (gi)
            // NOTE:
            // Directly modify the vtot.  vext is not used in the
            // matrix-vector multiplication in the eigensolver.
            blas::Axpy( numGridExtElemFine.prod(), 1.0, eigSol.Ham().Vext().Data(), 1,
                eigSol.Ham().Vtot().Data(), 1 );
          } // if ( isPeriodizePotential_ ) 
        } // own this element
      } // for (i)

  // Update the potential in element on LGL grid
  //
  // The local potential on the LGL grid is done by using Fourier
  // interpolation from the extended element to the element. Gibbs
  // phenomena MAY be there but at least this is better than
  // Lagrange interpolation on a uniform grid.
  //
  // NOTE: The interpolated potential on the LGL grid is taken to be the
  // MODIFIED potential with vext on the extended element. Therefore it
  // is important that the artificial vext vanishes inside the element.
  // When periodization option is used, it can potentially reduce the
  // effect of Gibbs phenomena.

  for( Int k = 0; k < numElem_[2]; k++ )
    for( Int j = 0; j < numElem_[1]; j++ )
      for( Int i = 0; i < numElem_[0]; i++ ){
        Index3 key( i, j, k );
        if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
          EigenSolver&  eigSol = distEigSolPtr_->LocalMap()[key];
          Index3 numGridExtElemFine = eigSol.FFT().domain.numGridFine;

          DblNumVec&  vtotLGLElem = hamDG.VtotLGL().LocalMap()[key];
          Index3 numLGLGrid       = hamDG.NumLGLGridElem();

          DblNumVec&    vtotExtElem = eigSol.Ham().Vtot();

          InterpPeriodicUniformFineToLGL( 
              numGridExtElemFine,
              numLGLGrid,
              vtotExtElem.Data(),
              vtotLGLElem.Data() );
        } // own this element
      } // for (i)

  return ;
}         // -----  end of method SCFDG::UpdateElemLocalPotential  ----- 

void
SCFDG::CalculateOccupationRate    ( DblNumVec& eigVal, DblNumVec& occupationRate )
{
  // For a given finite temperature, update the occupation number */
  Int npsi       = hamDGPtr_->NumStateTotal();
  Int nOccStates = hamDGPtr_->NumOccupiedState();

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
}         // -----  end of method SCFDG::CalculateOccupationRate  ----- 

void
SCFDG::CalculateOccupationRateExtElem    ( DblNumVec& eigVal, DblNumVec& occupationRate, Int npsi, Int nOccStates )
{

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

  return ;
}         // -----  end of method SCFDG::CalculateOccupationRateExtElm  -----

void
SCFDG::InterpPeriodicUniformToLGL    ( 
    const Index3& numUniformGrid, 
    const Index3& numLGLGrid, 
    const Real*   psiUniform, 
    Real*         psiLGL )
{

  Index3 Ns1 = numUniformGrid;
  Index3 Ns2 = numLGLGrid;

  DblNumVec  tmp1( Ns2[0] * Ns1[1] * Ns1[2] );
  DblNumVec  tmp2( Ns2[0] * Ns2[1] * Ns1[2] );
  SetValue( tmp1, 0.0 );
  SetValue( tmp2, 0.0 );

  // x-direction, use Gemm
  {
    Int m = Ns2[0], n = Ns1[1] * Ns1[2], k = Ns1[0];
    blas::Gemm( 'N', 'N', m, n, k, 1.0, PeriodicUniformToLGLMat_[0].Data(),
        m, psiUniform, k, 0.0, tmp1.Data(), m );
  }

  // y-direction, use Gemv
  {
    Int   m = Ns2[1], n = Ns1[1];
    Int   ptrShift1, ptrShift2;
    Int   inc = Ns2[0];
    for( Int k = 0; k < Ns1[2]; k++ ){
      for( Int i = 0; i < Ns2[0]; i++ ){
        ptrShift1 = i + k * Ns2[0] * Ns1[1];
        ptrShift2 = i + k * Ns2[0] * Ns2[1];
        blas::Gemv( 'N', m, n, 1.0, 
            PeriodicUniformToLGLMat_[1].Data(), m, 
            tmp1.Data() + ptrShift1, inc, 0.0, 
            tmp2.Data() + ptrShift2, inc );
      } // for (i)
    } // for (k)
  }


  // z-direction, use Gemm
  {
    Int m = Ns2[0] * Ns2[1], n = Ns2[2], k = Ns1[2]; 
    blas::Gemm( 'N', 'T', m, n, k, 1.0, 
        tmp2.Data(), m, 
        PeriodicUniformToLGLMat_[2].Data(), n, 0.0, psiLGL, m );
  }


  return ;
}         // -----  end of method SCFDG::InterpPeriodicUniformToLGL  ----- 

void
SCFDG::InterpPeriodicUniformFineToLGL    ( 
    const Index3& numUniformGridFine, 
    const Index3& numLGLGrid, 
    const Real*   rhoUniform, 
    Real*         rhoLGL )
{

  Index3 Ns1 = numUniformGridFine;
  Index3 Ns2 = numLGLGrid;

  DblNumVec  tmp1( Ns2[0] * Ns1[1] * Ns1[2] );
  DblNumVec  tmp2( Ns2[0] * Ns2[1] * Ns1[2] );
  SetValue( tmp1, 0.0 );
  SetValue( tmp2, 0.0 );

  // x-direction, use Gemm
  {
    Int m = Ns2[0], n = Ns1[1] * Ns1[2], k = Ns1[0];
    blas::Gemm( 'N', 'N', m, n, k, 1.0, PeriodicUniformFineToLGLMat_[0].Data(),
        m, rhoUniform, k, 0.0, tmp1.Data(), m );
  }

  // y-direction, use Gemv
  {
    Int   m = Ns2[1], n = Ns1[1];
    Int   rhoShift1, rhoShift2;
    Int   inc = Ns2[0];
    for( Int k = 0; k < Ns1[2]; k++ ){
      for( Int i = 0; i < Ns2[0]; i++ ){
        rhoShift1 = i + k * Ns2[0] * Ns1[1];
        rhoShift2 = i + k * Ns2[0] * Ns2[1];
        blas::Gemv( 'N', m, n, 1.0, 
            PeriodicUniformFineToLGLMat_[1].Data(), m, 
            tmp1.Data() + rhoShift1, inc, 0.0, 
            tmp2.Data() + rhoShift2, inc );
      } // for (i)
    } // for (k)
  }


  // z-direction, use Gemm
  {
    Int m = Ns2[0] * Ns2[1], n = Ns2[2], k = Ns1[2]; 
    blas::Gemm( 'N', 'T', m, n, k, 1.0, 
        tmp2.Data(), m, 
        PeriodicUniformFineToLGLMat_[2].Data(), n, 0.0, rhoLGL, m );
  }


  return ;
}         // -----  end of method SCFDG::InterpPeriodicUniformFineToLGL  ----- 

void
SCFDG::InterpPeriodicGridExtElemToGridElem ( 
    const Index3& numUniformGridFineExtElem, 
    const Index3& numUniformGridFineElem, 
    const Real*   rhoUniformExtElem, 
    Real*         rhoUniformElem )
{

  Index3 Ns1 = numUniformGridFineExtElem;
  Index3 Ns2 = numUniformGridFineElem;

  DblNumVec  tmp1( Ns2[0] * Ns1[1] * Ns1[2] );
  DblNumVec  tmp2( Ns2[0] * Ns2[1] * Ns1[2] );
  SetValue( tmp1, 0.0 );
  SetValue( tmp2, 0.0 );

  // x-direction, use Gemm
  {
    Int m = Ns2[0], n = Ns1[1] * Ns1[2], k = Ns1[0];
    blas::Gemm( 'N', 'N', m, n, k, 1.0, PeriodicGridExtElemToGridElemMat_[0].Data(),
        m, rhoUniformExtElem, k, 0.0, tmp1.Data(), m );
  }

  // y-direction, use Gemv
  {
    Int   m = Ns2[1], n = Ns1[1];
    Int   rhoShift1, rhoShift2;
    Int   inc = Ns2[0];
    for( Int k = 0; k < Ns1[2]; k++ ){
      for( Int i = 0; i < Ns2[0]; i++ ){
        rhoShift1 = i + k * Ns2[0] * Ns1[1];
        rhoShift2 = i + k * Ns2[0] * Ns2[1];
        blas::Gemv( 'N', m, n, 1.0, 
            PeriodicGridExtElemToGridElemMat_[1].Data(), m, 
            tmp1.Data() + rhoShift1, inc, 0.0, 
            tmp2.Data() + rhoShift2, inc );
      } // for (i)
    } // for (k)
  }


  // z-direction, use Gemm
  {
    Int m = Ns2[0] * Ns2[1], n = Ns2[2], k = Ns1[2]; 
    blas::Gemm( 'N', 'T', m, n, k, 1.0, 
        tmp2.Data(), m, 
        PeriodicGridExtElemToGridElemMat_[2].Data(), n, 0.0, rhoUniformElem, m );
  }

  return ;
}         // -----  end of method SCFDG::InterpPeriodicGridExtElemToGridElem  ----- 

void
SCFDG::CalculateKSEnergy    (  )
{
  Int mpirank, mpisize;
  MPI_Comm_rank( domain_.comm, &mpirank );
  MPI_Comm_size( domain_.comm, &mpisize );

  HamiltonianDG&  hamDG = *hamDGPtr_;

  DblNumVec&  eigVal         = hamDG.EigVal();
  DblNumVec&  occupationRate = hamDG.OccupationRate();

  std::string smearing_scheme = esdfParam.smearing_scheme;

  // Band energy
  Int numSpin = hamDG.NumSpin();

  if(SCFDG_comp_subspace_engaged_ == 1)
  {
    double HC_part = 0.0;

    for(Int sum_iter = 0; sum_iter < SCFDG_comp_subspace_N_solve_; sum_iter ++)
      HC_part += (1.0 - SCFDG_comp_subspace_top_occupations_(sum_iter)) * SCFDG_comp_subspace_top_eigvals_(sum_iter);

    Ekin_ = numSpin * (SCFDG_comp_subspace_trace_Hmat_ - HC_part) ;
  }
  else
  {  
    Ekin_ = 0.0;
    for (Int i=0; i < eigVal.m(); i++) {
      Ekin_  += numSpin * eigVal(i) * occupationRate(i);
//      statusOFS << " i " << i << " eigVal " << eigVal(i) << "  occupationRate " << occupationRate(i) << std::endl; 
    }
  }

  // Self energy part
  Eself_ = 0.0;
  std::vector<Atom>&  atomList = hamDG.AtomList();
  for(Int a=0; a< atomList.size() ; a++) {
    Int type = atomList[a].type;
    Eself_ += ptablePtr_->SelfIonInteraction(type);
  }

  // Hartree and XC part
  Ehart_ = 0.0;
  EVxc_  = 0.0;

//  Real EhartLocal = 0.0, EVxcLocal = 0.0;
  Real EhartLocal = 0.0, EcorLocal = 0.0;

  for( Int k = 0; k < numElem_[2]; k++ )
    for( Int j = 0; j < numElem_[1]; j++ )
      for( Int i = 0; i < numElem_[0]; i++ ){
        Index3 key( i, j, k );
        if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
          DblNumVec&  density      = hamDG.Density().LocalMap()[key];
//          DblNumVec&  vxc          = hamDG.Vxc().LocalMap()[key];
          DblNumVec&  pseudoCharge = hamDG.PseudoCharge().LocalMap()[key];
          DblNumVec&  vhart        = hamDG.Vhart().LocalMap()[key];

          // Correct the incorrect Ecoul and Exc included in Ekin
          DblNumVec&  vtotOld      = hamDG.Vtot().LocalMap()[key];
          DblNumVec&  vLocalSR     = hamDG.VLocalSR().LocalMap()[key];
          DblNumVec&  vext         = hamDG.Vext().LocalMap()[key];

          for (Int p=0; p < density.Size(); p++) {
            EhartLocal += 0.5 * vhart(p) * ( density(p) - pseudoCharge(p) );
          }
        
        for (Int p=0; p < density.Size(); p++) {
            EcorLocal += ( vLocalSR(p) + vext(p) - vtotOld(p) ) * density(p);
//            EVxcLocal  += vxc(p) * density(p);
//            EhartLocal += 0.5 * vhart(p) * ( density(p) + pseudoCharge(p) );
          }

        } // own this element
      } // for (i)

//  mpi::Allreduce( &EVxcLocal, &EVxc_, 1, MPI_SUM, domain_.colComm );
  mpi::Allreduce( &EhartLocal, &Ehart_, 1, MPI_SUM, domain_.colComm );
  mpi::Allreduce( &EcorLocal, &Ecor_, 1, MPI_SUM, domain_.colComm );

  Ehart_ *= domain_.Volume() / domain_.NumGridTotalFine();
//  EVxc_  *= domain_.Volume() / domain_.NumGridTotalFine();
  Ecor_  *= domain_.Volume() / domain_.NumGridTotalFine();

  // Correction energy
//  Ecor_   = (Exc_ - Ehfx_ - EVxc_) - Ehart_ - Eself_;
  Ecor_ = Ecor_ + Exc_ + Ehart_ - Eself_;
  if( esdfParam.isUseVLocal == true ){
    Ecor_ += hamDG.EIonSR();
  }

  // Total energy
//  Etot_ = Ekin_ + Ecor_;
  Etot_ = Ekin_ + Ecor_;

  if( !DGHFXNestedLoop_ && hamDG.IsHybrid() && hamDG.IsEXXActive() ) {
      Etot_ -= Ehfx_;
  }

  // Helmholtz free energy
  if( hamDG.NumOccupiedState() == 
      hamDG.NumStateTotal() ){
    // Zero temperature
    Efree_ = Etot_;
  }
  else{
    // Finite temperature
    Efree_ = 0.0;
    Real fermi = fermi_;
    Real Tbeta = Tbeta_;

    if(SCFDG_comp_subspace_engaged_ == 1)
    {
      double occup_energy_part = 0.0;
      double occup_tol = 1e-12;
      double fl;
      for(Int l=0; l < SCFDG_comp_subspace_top_occupations_.m(); l++)
      {
        fl = SCFDG_comp_subspace_top_occupations_(l);
        if((fl > occup_tol) && ((1.0 - fl) > occup_tol))
          occup_energy_part += fl * log(fl) + (1.0 - fl) * log(1 - fl);
      }

      Efree_ = Ekin_ + Ecor_ + (numSpin / Tbeta) * occup_energy_part;
    }
    else
    {
      for( Int k = 0; k < eigVal.m(); k++){
        Real eig = eigVal(k);

        Efree_ += numSpin * getEntropy( eig, fermi, Tbeta, smearing_scheme );
      }
    }
    Efree_ += Etot_;
  }

  return ;
}         // -----  end of method SCFDG::CalculateKSEnergy  ----- 

void
  SCFDG::CalculateKSEnergyDM (
      Real totalEnergyH,
      DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn>& distEDMMat,
      DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn>& distFDMMat )
  {
    Int mpirank, mpisize;
    MPI_Comm_rank( domain_.comm, &mpirank );
    MPI_Comm_size( domain_.comm, &mpisize );

    HamiltonianDG&  hamDG = *hamDGPtr_;

    DblNumVec&  eigVal         = hamDG.EigVal();
    DblNumVec&  occupationRate = hamDG.OccupationRate();

    // Kinetic energy
    Int numSpin = hamDG.NumSpin();

    // Self energy part
    Eself_ = 0.0;
    std::vector<Atom>&  atomList = hamDG.AtomList();
    for(Int a=0; a< atomList.size() ; a++) {
      Int type = atomList[a].type;
      Eself_ += ptablePtr_->SelfIonInteraction(type);
    }

    // Hartree and XC part
    Ehart_ = 0.0;
    EVxc_  = 0.0;

    Real EhartLocal = 0.0, EVxcLocal = 0.0;

    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
        for( Int i = 0; i < numElem_[0]; i++ ){
          Index3 key( i, j, k );
          if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
            DblNumVec&  density      = hamDG.Density().LocalMap()[key];
            DblNumVec&  vxc          = hamDG.Vxc().LocalMap()[key];
            DblNumVec&  pseudoCharge = hamDG.PseudoCharge().LocalMap()[key];
            DblNumVec&  vhart        = hamDG.Vhart().LocalMap()[key];

            for (Int p=0; p < density.Size(); p++) {
              EVxcLocal  += vxc(p) * density(p);
              EhartLocal += 0.5 * vhart(p) * ( density(p) + pseudoCharge(p) );
            }

          } // own this element
        } // for (i)

    mpi::Allreduce( &EVxcLocal, &EVxc_, 1, MPI_SUM, domain_.colComm );
    mpi::Allreduce( &EhartLocal, &Ehart_, 1, MPI_SUM, domain_.colComm );

    Ehart_ *= domain_.Volume() / domain_.NumGridTotalFine();
    EVxc_  *= domain_.Volume() / domain_.NumGridTotalFine();

    // Correction energy
    Ecor_   = (Exc_  - EVxc_) - Ehart_ - Eself_;

    if( !DGHFXNestedLoop_ && hamDG.IsHybrid() && hamDG.IsEXXActive() ) {
      Ecor_ -= Ehfx_;
    }

    if( esdfParam.isUseVLocal == true ){
      Ecor_ += hamDG.EIonSR();
    }

    // Kinetic energy and helmholtz free energy, calculated from the
    // energy and free energy density matrices.
    // Here 
    // 
    //   Ekin = Tr[H 2/(1+exp(beta(H-mu)))] 
    // and
    //   Ehelm = -2/beta Tr[log(1+exp(mu-H))] + mu*N_e
    // FIXME Put the above documentation to the proper place like the hpp
    // file

    Real Ehelm = 0.0, EhelmLocal = 0.0, EkinLocal = 0.0;

    // FIXME Ekin is not used later.
    if( 1 ) {
      // Compute the trace of the energy density matrix in each element
      for( Int k = 0; k < numElem_[2]; k++ )
        for( Int j = 0; j < numElem_[1]; j++ )
          for( Int i = 0; i < numElem_[0]; i++ ){
            Index3 key( i, j, k );
            if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){

              DblNumMat& localEDM = distEDMMat.LocalMap()[
                ElemMatKey(key, key)];
              DblNumMat& localFDM = distFDMMat.LocalMap()[
                ElemMatKey(key, key)];

              for( Int a = 0; a < localEDM.m(); a++ ){
                EkinLocal  += localEDM(a,a);
                EhelmLocal += localFDM(a,a);
              }
            } // own this element
          } // for (i)

      // Reduce the results 
      mpi::Allreduce( &EkinLocal, &Ekin_, 
          1, MPI_SUM, domain_.colComm );

      mpi::Allreduce( &EhelmLocal, &Ehelm, 
          1, MPI_SUM, domain_.colComm );

      // Add the mu*N term for the free energy
      Ehelm += fermi_ * hamDG.NumOccupiedState() * numSpin;

    }

//    statusOFS << " Ekin_ " << Ekin_ << std::endl;
//    statusOFS << " totalEnergyH " << totalEnergyH << std::endl;

    // FIXME In order to be compatible with PPEXSIDFTDriver3, the
    // Tr[H*DM] part is directly read from totalEnergyH
    Ekin_ = totalEnergyH;

    // Total energy
    Etot_ = Ekin_ + Ecor_;

    // Free energy at finite temperature
    // FIXME PPEXSIDFTDriver3 does not have free energy
    Ehelm = totalEnergyH;
    Efree_ = Ehelm + Ecor_;

    return ;
  }         // -----  end of method SCFDG::CalculateKSEnergyDM  ----- 

void
SCFDG::CalculateHarrisEnergy    (  )
{
  Int mpirank, mpisize;
  MPI_Comm_rank( domain_.comm, &mpirank );
  MPI_Comm_size( domain_.comm, &mpisize );

  HamiltonianDG&  hamDG = *hamDGPtr_;

  DblNumVec&  eigVal         = hamDG.EigVal();
  DblNumVec&  occupationRate = hamDG.OccupationRate();

  std::string smearing_scheme = esdfParam.smearing_scheme;

  // NOTE: To avoid confusion, all energies in this routine are
  // temporary variables other than EfreeHarris_.
  //
  // The related energies will be computed again in the routine
  //
  // CalculateKSEnergy()

  Real Ekin, Eself, Ehart, EVxc, Exc, Exx, Ecor;

  // Kinetic energy from the new density matrix.
  Int numSpin = hamDG.NumSpin();
  Ekin = 0.0;

  if(SCFDG_comp_subspace_engaged_ == 1)
  {
    // This part is the same irrespective of smearing type
    double HC_part = 0.0;

    for(Int sum_iter = 0; sum_iter < SCFDG_comp_subspace_N_solve_; sum_iter ++)
      HC_part += (1.0 - SCFDG_comp_subspace_top_occupations_(sum_iter)) * SCFDG_comp_subspace_top_eigvals_(sum_iter);

    Ekin = numSpin * (SCFDG_comp_subspace_trace_Hmat_ - HC_part) ;
  }
  else
  {  
    for (Int i=0; i < eigVal.m(); i++) {
      Ekin  += numSpin * eigVal(i) * occupationRate(i);
    }
  }
  // Self energy part
  Eself = 0.0;
  std::vector<Atom>&  atomList = hamDG.AtomList();
  for(Int a=0; a< atomList.size() ; a++) {
    Int type = atomList[a].type;
    Eself += ptablePtr_->SelfIonInteraction(type);
  }

  // Nonlinear correction part.  This part uses the Hartree energy and
  // XC correlation energy from the old electron density.

  Real EhartLocal = 0.0, EVxcLocal = 0.0;

  for( Int k = 0; k < numElem_[2]; k++ )
    for( Int j = 0; j < numElem_[1]; j++ )
      for( Int i = 0; i < numElem_[0]; i++ ){
        Index3 key( i, j, k );
        if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
          DblNumVec&  density      = hamDG.Density().LocalMap()[key];
          DblNumVec&  vxc          = hamDG.Vxc().LocalMap()[key];
          DblNumVec&  pseudoCharge = hamDG.PseudoCharge().LocalMap()[key];
          DblNumVec&  vhart        = hamDG.Vhart().LocalMap()[key];

          for (Int p=0; p < density.Size(); p++) {
            EVxcLocal  += vxc(p) * density(p);
            EhartLocal += 0.5 * vhart(p) * ( density(p) + pseudoCharge(p) );
          }

        } // own this element
      } // for (i)

  mpi::Allreduce( &EVxcLocal, &EVxc, 1, MPI_SUM, domain_.colComm );
  mpi::Allreduce( &EhartLocal, &Ehart, 1, MPI_SUM, domain_.colComm );

  Ehart *= domain_.Volume() / domain_.NumGridTotalFine();
  EVxc  *= domain_.Volume() / domain_.NumGridTotalFine();
  // Use the previous exchange-correlation energy
  Exc    = Exc_;

  // Correction energy.  
  Ecor   = (Exc - EVxc) - Ehart - Eself;
 
  if( !DGHFXNestedLoop_ && hamDG.IsHybrid() && hamDG.IsEXXActive() ) {
      Ecor -= Ehfx_;
  }

  if( esdfParam.isUseVLocal == true ){
    Ecor  += hamDG.EIonSR();
  }

  // Harris free energy functional
  if( hamDG.NumOccupiedState() == 
      hamDG.NumStateTotal() ){
    // Zero temperature
    EfreeHarris_ = Ekin + Ecor;
  }
  else{
    // Finite temperature
    EfreeHarris_ = 0.0;
    Real fermi = fermi_;
    Real Tbeta = Tbeta_;

    if(SCFDG_comp_subspace_engaged_ == 1)
    {
      // Complementary subspace technique in use

      double occup_energy_part = 0.0;
      double occup_tol = 1e-12;
      double fl, x;

      if(SmearingScheme_ == "FD")
      {
        for(Int l=0; l < SCFDG_comp_subspace_top_occupations_.m(); l++)
        {
          fl = SCFDG_comp_subspace_top_occupations_(l);
          if((fl > occup_tol) && ((1.0 - fl) > occup_tol))
            occup_energy_part += fl * log(fl) + (1.0 - fl) * log(1 - fl);
        }

        EfreeHarris_ = Ekin + Ecor + (numSpin / Tbeta) * occup_energy_part;
      }
      else
      {
        // Other kinds of smearing

        for(Int l=0; l < SCFDG_comp_subspace_top_occupations_.m(); l++)
        {
          fl = SCFDG_comp_subspace_top_occupations_(l);
          if((fl > occup_tol) && ((1.0 - fl) > occup_tol))
          {
            x = (SCFDG_comp_subspace_top_eigvals_(l) - fermi_) / Tsigma_ ;
            occup_energy_part += mp_entropy(x, MP_smearing_order_);
          }
        }

        EfreeHarris_ = Ekin + Ecor + (numSpin / Tbeta) * occup_energy_part;

      }
    }  
    else
    { 
      // Complementary subspace technique not in use : full spectrum available
//      if(SmearingScheme_ == "FD")
//      {
        for(Int k =0; k < eigVal.m(); k++) {
          Real eig = eigVal(k);
          EfreeHarris_ += numSpin * getEntropy( eig, fermi, Tbeta, smearing_scheme );
        }
//        EfreeHarris_ += Etot_;
          EfreeHarris_ += Ekin + Ecor;

//      }
//      else
//      {
//        // GB or MP schemes in use
//        double occup_energy_part = 0.0;
//        double occup_tol = 1e-12;
//        double fl, x;
//
//        for(Int l=0; l < eigVal.m(); l++)
//        {
//          fl = occupationRate(l);
//          if((fl > occup_tol) && ((1.0 - fl) > occup_tol))
//          { 
//            x = (eigVal(l) - fermi_) / Tsigma_ ;
//            occup_energy_part += mp_entropy(x, MP_smearing_order_) ;
//          }
//        }
//
//        EfreeHarris_ = Ekin + Ecor + (numSpin * Tsigma_) * occup_energy_part;

//      }

    } // end of full spectrum available calculation
  } // end of finite temperature calculation

  return ;
}         // -----  end of method SCFDG::CalculateHarrisEnergy  ----- 

void
SCFDG::CalculateHarrisEnergyDM(
    Real totalFreeEnergy,
    DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn>& distFDMMat )
{
  Int mpirank, mpisize;
  MPI_Comm_rank( domain_.comm, &mpirank );
  MPI_Comm_size( domain_.comm, &mpisize );

  HamiltonianDG&  hamDG = *hamDGPtr_;

  // NOTE: To avoid confusion, all energies in this routine are
  // temporary variables other than EfreeHarris_.
  //
  // The related energies will be computed again in the routine
  //
  // CalculateKSEnergy()

  Real Ehelm, Eself, Ehart, EVxc, Exc, Exx, Ecor;

  Int numSpin = hamDG.NumSpin();

  // Self energy part
  Eself = 0.0;
  std::vector<Atom>&  atomList = hamDG.AtomList();
  for(Int a=0; a< atomList.size() ; a++) {
    Int type = atomList[a].type;
    Eself += ptablePtr_->SelfIonInteraction(type);
  }

  // Nonlinear correction part.  This part uses the Hartree energy and
  // XC correlation energy from the old electron density.

  Real EhartLocal = 0.0, EVxcLocal = 0.0;

  for( Int k = 0; k < numElem_[2]; k++ )
    for( Int j = 0; j < numElem_[1]; j++ )
      for( Int i = 0; i < numElem_[0]; i++ ){
        Index3 key( i, j, k );
        if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
          DblNumVec&  density      = hamDG.Density().LocalMap()[key];
          DblNumVec&  vxc          = hamDG.Vxc().LocalMap()[key];
          DblNumVec&  pseudoCharge = hamDG.PseudoCharge().LocalMap()[key];
          DblNumVec&  vhart        = hamDG.Vhart().LocalMap()[key];

          for (Int p=0; p < density.Size(); p++) {
            EVxcLocal  += vxc(p) * density(p);
            EhartLocal += 0.5 * vhart(p) * ( density(p) + pseudoCharge(p) );
          }
        } // own this element
      } // for (i)

  mpi::Allreduce( &EVxcLocal, &EVxc, 1, MPI_SUM, domain_.colComm );
  mpi::Allreduce( &EhartLocal, &Ehart, 1, MPI_SUM, domain_.colComm );

  Ehart *= domain_.Volume() / domain_.NumGridTotalFine();
  EVxc  *= domain_.Volume() / domain_.NumGridTotalFine();
  // Use the previous exchange-correlation energy
  Exc    = Exc_;

  // Correction energy.  
  Ecor   = (Exc - EVxc) - Ehart - Eself;
  if( esdfParam.isUseVLocal == true ){
    Ecor  += hamDG.EIonSR();
  }

  if( !DGHFXNestedLoop_ && hamDG.IsHybrid() && hamDG.IsEXXActive() ) {
      Ecor -= Ehfx_;
  }

  // The Helmholtz part of the free energy
  // Ehelm = -2/beta Tr[log(1+exp(mu-H))] + mu*N_e
  // FIXME Put the above documentation to the proper place like the hpp
  // file
  Real EhelmLocal = 0.0;
  Ehelm = 0.0;

  // Compute the trace of the energy density matrix in each element
  for( Int k = 0; k < numElem_[2]; k++ )
    for( Int j = 0; j < numElem_[1]; j++ )
      for( Int i = 0; i < numElem_[0]; i++ ){
        Index3 key( i, j, k );
        if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
          DblNumMat& localFDM = distFDMMat.LocalMap()[
          ElemMatKey(key, key)];

          for( Int a = 0; a < localFDM.m(); a++ ){
            EhelmLocal += localFDM(a,a);
          }
        } // own this element
  } // for (i)

  mpi::Allreduce( &EhelmLocal, &Ehelm, 1, MPI_SUM, domain_.colComm );

  // Add the mu*N term
  Ehelm += fermi_ * hamDG.NumOccupiedState() * numSpin;

  // Harris free energy functional. This has to be the finite
  // temperature formulation

  // FIXME
  Ehelm = totalFreeEnergy;
  EfreeHarris_ = Ehelm + Ecor;

  return ;
}         // -----  end of method SCFDG::CalculateHarrisEnergyDM  ----- 

void
SCFDG::CalculateSecondOrderEnergy  (  )
{
  Int mpirank, mpisize;
  MPI_Comm_rank( domain_.comm, &mpirank );
  MPI_Comm_size( domain_.comm, &mpisize );

  HamiltonianDG&  hamDG = *hamDGPtr_;

  DblNumVec&  eigVal         = hamDG.EigVal();
  DblNumVec&  occupationRate = hamDG.OccupationRate();

  // NOTE: To avoid confusion, all energies in this routine are
  // temporary variables other than EfreeSecondOrder_.
  // 
  // This is similar to the situation in 
  //
  // CalculateHarrisEnergy()

  Real Ekin, Eself, Ehart, EVtot, Exc, Exx, Ecor;

  // Kinetic energy from the new density matrix.
  Int numSpin = hamDG.NumSpin();
  Ekin = 0.0;

  if(SCFDG_comp_subspace_engaged_ == 1)
  {
    // This part is the same, irrespective of smearing type
    double HC_part = 0.0;

    for(Int sum_iter = 0; sum_iter < SCFDG_comp_subspace_N_solve_; sum_iter ++)
      HC_part += (1.0 - SCFDG_comp_subspace_top_occupations_(sum_iter)) * SCFDG_comp_subspace_top_eigvals_(sum_iter);

    Ekin = numSpin * (SCFDG_comp_subspace_trace_Hmat_ - HC_part) ;
  }
  else
  {  
    for (Int i=0; i < eigVal.m(); i++) {
      Ekin  += numSpin * eigVal(i) * occupationRate(i);
    }
  }

  // Self energy part
  Eself = 0.0;
  std::vector<Atom>&  atomList = hamDG.AtomList();
  for(Int a=0; a< atomList.size() ; a++) {
    Int type = atomList[a].type;
    Eself_ += ptablePtr_->SelfIonInteraction(type);
  }

  // Nonlinear correction part.  This part uses the Hartree energy and
  // XC correlation energy from the OUTPUT electron density, but the total
  // potential is the INPUT one used in the diagonalization process.
  // The density is also the OUTPUT density.
  //
  // NOTE the sign flip in Ehart, which is different from those in KS
  // energy functional and Harris energy functional.

  Real EhartLocal = 0.0, EVtotLocal = 0.0;

  for( Int k = 0; k < numElem_[2]; k++ )
    for( Int j = 0; j < numElem_[1]; j++ )
      for( Int i = 0; i < numElem_[0]; i++ ){
        Index3 key( i, j, k );
        if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
          DblNumVec&  density      = hamDG.Density().LocalMap()[key];
          DblNumVec&  vext         = hamDG.Vext().LocalMap()[key];
          DblNumVec&  vtot         = hamDG.Vtot().LocalMap()[key];
          DblNumVec&  pseudoCharge = hamDG.PseudoCharge().LocalMap()[key];
          DblNumVec&  vhart        = hamDG.Vhart().LocalMap()[key];

          for (Int p=0; p < density.Size(); p++) {
            EVtotLocal  += (vtot(p) - vext(p)) * density(p);
            // NOTE the sign flip
            EhartLocal  += 0.5 * vhart(p) * ( density(p) - pseudoCharge(p) );
          }
        } // own this element
      } // for (i)

  mpi::Allreduce( &EVtotLocal, &EVtot, 1, MPI_SUM, domain_.colComm );
  mpi::Allreduce( &EhartLocal, &Ehart, 1, MPI_SUM, domain_.colComm );

  Ehart *= domain_.Volume() / domain_.NumGridTotalFine();
  EVtot *= domain_.Volume() / domain_.NumGridTotalFine();

  // Use the exchange-correlation energy with respect to the new
  // electron density
  Exc = Exc_;
  // Correction energy.  
  // NOTE The correction energy in the second order method means
  // differently from that in Harris energy functional or the KS energy
  // functional.
  Ecor   = (Exc + Ehart - Eself) - EVtot;

  if( !DGHFXNestedLoop_ && hamDG.IsHybrid() && hamDG.IsEXXActive() ) {
      Ecor -= Ehfx_;
  }

  // FIXME
  //    statusOFS
  //        << "Component energy for second order correction formula = " << std::endl
  //        << "Exc     = " << Exc      << std::endl
  //        << "Ehart   = " << Ehart    << std::endl
  //        << "Eself   = " << Eself    << std::endl
  //        << "EVtot   = " << EVtot    << std::endl
  //        << "Ecor    = " << Ecor     << std::endl;
  //    

  // Second order accurate free energy functional
  if( hamDG.NumOccupiedState() == 
      hamDG.NumStateTotal() ){
    // Zero temperature
    EfreeSecondOrder_ = Ekin + Ecor;
  }
  else{
    // Finite temperature
    EfreeSecondOrder_ = 0.0;
    Real fermi = fermi_;
    Real Tbeta = Tbeta_;

    if(SCFDG_comp_subspace_engaged_ == 1)
    {
      double occup_energy_part = 0.0;
      double occup_tol = 1e-12;
      double fl, x;

      if(SmearingScheme_ == "FD")
      {
        for(Int l = 0; l < SCFDG_comp_subspace_top_occupations_.m(); l++)
        {
          fl = SCFDG_comp_subspace_top_occupations_(l);
          if((fl > occup_tol) && ((1.0 - fl) > occup_tol))
            occup_energy_part += fl * log(fl) + (1.0 - fl) * log(1 - fl);
        }

        EfreeSecondOrder_ = Ekin + Ecor + (numSpin / Tbeta) * occup_energy_part;
      }
      else
      {
        // MP and GB smearing    
        for(Int l = 0; l < SCFDG_comp_subspace_top_occupations_.m(); l++)
        {
          fl = SCFDG_comp_subspace_top_occupations_(l);
          if((fl > occup_tol) && ((1.0 - fl) > occup_tol))
          {
            x = (SCFDG_comp_subspace_top_eigvals_(l) - fermi_) / Tsigma_ ;
            occup_energy_part += mp_entropy(x, MP_smearing_order_);
          }
        }

        EfreeSecondOrder_ = Ekin + Ecor + (numSpin / Tbeta) * occup_energy_part;            
      }
    }
    else
    {
      // Complementary subspace technique not in use : full spectrum available
      if(SmearingScheme_ == "FD")
      {
        for(Int l=0; l< eigVal.m(); l++) {
          Real eig = eigVal(l);
          if( eig - fermi >= 0){
            EfreeSecondOrder_ += -numSpin /Tbeta*log(1.0+exp(-Tbeta*(eig - fermi))); 
          }
          else{
            EfreeSecondOrder_ += numSpin * (eig - fermi) - numSpin / Tbeta*log(1.0+exp(Tbeta*(eig-fermi)));
          }
        }

        EfreeSecondOrder_ += Ecor + fermi * hamDG.NumOccupiedState() * numSpin; 
      }
      else
      {
        // GB or MP schemes in use
        double occup_energy_part = 0.0;
        double occup_tol = 1e-12;
        double fl, x;

        for(Int l=0; l < eigVal.m(); l++)
        {
          fl = occupationRate(l);
          if((fl > occup_tol) && ((1.0 - fl) > occup_tol))
          { 
            x = (eigVal(l) - fermi_) / Tsigma_ ;
            occup_energy_part += mp_entropy(x, MP_smearing_order_) ;
          }
        }

        EfreeSecondOrder_ = Ekin + Ecor + (numSpin * Tsigma_) * occup_energy_part;
      }  // end of full spectrum available calculation
    }
  } // end of finite temperature calculation

  return ;
}         // -----  end of method SCFDG::CalculateSecondOrderEnergy  ----- 

void
SCFDG::CalculateVDW    ( Real& VDWEnergy, DblNumMat& VDWForce )
{
  HamiltonianDG&  hamDG = *hamDGPtr_;
  std::vector<Atom>& atomList = hamDG.AtomList();
  Evdw_ = 0.0;
  forceVdw_.Resize( atomList.size(), DIM );
  SetValue( forceVdw_, 0.0 );

  Int numAtom = atomList.size();

  Domain& dm = domain_;

  if( VDWType_ == "DFT-D2"){

    const Int vdw_nspecies = 55;
    Int ia,is1,is2,is3,itypat,ja,jtypat,npairs,nshell;
    bool need_gradient,newshell;
    const Real vdw_d = 20.0;
    const Real vdw_tol_default = 1e-10;
    const Real vdw_s_pbe = 0.75;
    Real c6,c6r6,ex,fr,fred1,fred2,fred3,gr,grad,r0,r1,r2,r3,rcart1,rcart2,rcart3;

    double vdw_c6_dftd2[vdw_nspecies] = 
    {  0.14, 0.08, 1.61, 1.61, 3.13, 1.75, 1.23, 0.70, 0.75, 0.63,
      5.71, 5.71,10.79, 9.23, 7.84, 5.57, 5.07, 4.61,10.80,10.80,
      10.80,10.80,10.80,10.80,10.80,10.80,10.80,10.80,10.80,10.80,
      16.99,17.10,16.37,12.64,12.47,12.01,24.67,24.67,24.67,24.67,
      24.67,24.67,24.67,24.67,24.67,24.67,24.67,24.67,37.32,38.71,
      38.44,31.74,31.50,29.99, 0.00 };

    double vdw_r0_dftd2[vdw_nspecies] =
    { 1.001,1.012,0.825,1.408,1.485,1.452,1.397,1.342,1.287,1.243,
      1.144,1.364,1.639,1.716,1.705,1.683,1.639,1.595,1.485,1.474,
      1.562,1.562,1.562,1.562,1.562,1.562,1.562,1.562,1.562,1.562,
      1.650,1.727,1.760,1.771,1.749,1.727,1.628,1.606,1.639,1.639,
      1.639,1.639,1.639,1.639,1.639,1.639,1.639,1.639,1.672,1.804,
      1.881,1.892,1.892,1.881,1.000 };

    for(Int i=0; i<vdw_nspecies; i++) {
      vdw_c6_dftd2[i] = vdw_c6_dftd2[i] / 2625499.62 * pow(10/0.52917706, 6);
      vdw_r0_dftd2[i] = vdw_r0_dftd2[i] / 0.52917706;
    }

    DblNumMat vdw_c6(vdw_nspecies, vdw_nspecies);
    DblNumMat vdw_r0(vdw_nspecies, vdw_nspecies);
    SetValue( vdw_c6, 0.0 );
    SetValue( vdw_r0, 0.0 );

    for(Int i=0; i<vdw_nspecies; i++) {
      for(Int j=0; j<vdw_nspecies; j++) {
        vdw_c6(i,j) = std::sqrt( vdw_c6_dftd2[i] * vdw_c6_dftd2[j] );
        vdw_r0(i,j) = vdw_r0_dftd2[i] + vdw_r0_dftd2[j];
      }
    }

    Real vdw_s;
    if (XCType_ == "XC_GGA_XC_PBE") {
      vdw_s=vdw_s_pbe;
    }
    else {
      ErrorHandling( "Van der Waals DFT-D2 correction in only compatible with GGA-PBE!" );
    }

    for(Int ii=-1; ii<2; ii++) {
      for(Int jj=-1; jj<2; jj++) {
        for(Int kk=-1; kk<2; kk++) {

          for(Int i=0; i<atomList.size(); i++) {
            Int iType = atomList[i].type;
            for(Int j=0; j<(i+1); j++) {
              Int jType = atomList[j].type;

              Real rx = atomList[i].pos[0] - atomList[j].pos[0] + ii * dm.length[0];
              Real ry = atomList[i].pos[1] - atomList[j].pos[1] + jj * dm.length[1];
              Real rz = atomList[i].pos[2] - atomList[j].pos[2] + kk * dm.length[2];
              Real rr = std::sqrt( rx * rx + ry * ry + rz * rz );

              if ( ( rr > 0.0001 ) && ( rr < 75.0 ) ) {

                Real sfact = vdw_s;
                if ( i == j ) sfact = sfact * 0.5;

                Real c6 = vdw_c6(iType-1, jType-1);
                Real r0 = vdw_r0(iType-1, jType-1);

                Real ex = exp( -vdw_d * ( rr / r0 - 1 ));
                Real fr = 1.0 / ( 1.0 + ex );
                Real c6r6 = c6 / pow(rr, 6.0);

                // Contribution to energy
                Evdw_ = Evdw_ - sfact * fr * c6r6;

                // Contribution to force
                if( i != j ) {

                  Real gr = ( vdw_d / r0 ) * ( fr * fr ) * ex;
                  Real grad = sfact * ( gr - 6.0 * fr / rr ) * c6r6 / rr; 

                  Real fx = grad * rx;
                  Real fy = grad * ry;
                  Real fz = grad * rz;

                  forceVdw_( i, 0 ) = forceVdw_( i, 0 ) + fx; 
                  forceVdw_( i, 1 ) = forceVdw_( i, 1 ) + fy; 
                  forceVdw_( i, 2 ) = forceVdw_( i, 2 ) + fz; 
                  forceVdw_( j, 0 ) = forceVdw_( j, 0 ) - fx; 
                  forceVdw_( j, 1 ) = forceVdw_( j, 1 ) - fy; 
                  forceVdw_( j, 2 ) = forceVdw_( j, 2 ) - fz; 

                } // end for i != j

              } // end if

            } // end for j
          } // end for i

        } // end for ii
      } // end for jj
    } // end for kk
  } // If DFT-D2

  VDWEnergy = Evdw_;
  VDWForce = forceVdw_;

  return ;
}         // -----  end of method SCFDG::CalculateVDW  ----- 

void
SCFDG::PrintState    ( ) 
{
  HamiltonianDG&  hamDG = *hamDGPtr_;
  Real HOMO, LUMO, EG;

  if(solutionMethod_ != "pexsi")
  {
    HOMO = hamDG.EigVal()( hamDG.NumOccupiedState()-1 );
    if( hamDG.NumExtraState() > 0 ){
      LUMO = hamDG.EigVal()( hamDG.NumOccupiedState());
      EG = LUMO -HOMO;
    }
  }

#if 0
  if(SCFDG_comp_subspace_engaged_ == false)
  {  
    statusOFS << std::endl << "Eigenvalues in the global domain." << std::endl;
    for(Int i = 0; i < hamDG.EigVal().m(); i++){
      Print(statusOFS, 
          "band#    = ", i, 
          "eigval   = ", hamDG.EigVal()(i),
          "occrate  = ", hamDG.OccupationRate()(i));
    }
  }
#endif

  statusOFS << std::endl;
  Print(statusOFS, "EfreeHarris       = ",  EfreeHarris_, "[au]");
  //    Print(statusOFS, "EfreeSecondOrder  = ",  EfreeSecondOrder_, "[au]");
  Print(statusOFS, "Etot              = ",  Etot_, "[au]");
  Print(statusOFS, "Efree             = ",  Efree_, "[au]");
  Print(statusOFS, "Ekin              = ",  Ekin_, "[au]");
  Print(statusOFS, "Ehart             = ",  Ehart_, "[au]");
  Print(statusOFS, "EVxc              = ",  EVxc_, "[au]");
  Print(statusOFS, "Exc               = ",  Exc_, "[au]"); 
//  Print(statusOFS, "Exx               = ",  Ehfx_, "[au]");
  Print(statusOFS, "EVdw              = ",  Evdw_, "[au]"); 
  Print(statusOFS, "Eself             = ",  Eself_, "[au]");
  Print(statusOFS, "EIonSR            = ",  hamDG.EIonSR(), "[au]");
  Print(statusOFS, "Ecor              = ",  Ecor_, "[au]");
  Print(statusOFS, "Fermi             = ",  fermi_, "[au]");

  if(solutionMethod_ != "pexsi")
  {
    Print(statusOFS, "HOMO#            = ",  HOMO*au2ev, "[eV]");
    if( hamDG.NumExtraState() > 0 ){
      Print(statusOFS, "LUMO#            = ",  LUMO*au2ev, "[eV]");
      Print(statusOFS, "Bandgap#         = ",  EG*au2ev, "[eV]");
    }
  }

  return ;
}         // -----  end of method SCFDG::PrintState  ----- 

void
SCFDG::UpdateMDParameters    ( )
{
  scfOuterMaxIter_ = esdfParam.MDscfOuterMaxIter;
  useEnergySCFconvergence_ = 1;

  return ;
}         // -----  end of method SCFDG::UpdateMDParameters  ----- 

// xmqin
void
SCFDG:: ProjectDM ( DistDblNumMat&  Oldbasis,
                    DistDblNumMat&  Newbasis,
                    DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn>&  distDMMat )
{
  MPI_Barrier(domain_.comm);
  MPI_Barrier(domain_.colComm);
  MPI_Barrier(domain_.rowComm);
  Int mpirank, mpisize;
  MPI_Comm_rank( domain_.comm, &mpirank );
  MPI_Comm_size( domain_.comm, &mpisize );

  Int mpirankRow;  MPI_Comm_rank(domain_.rowComm, &mpirankRow);
  Int mpisizeRow;  MPI_Comm_size(domain_.rowComm, &mpisizeRow);

  HamiltonianDG& hamDG = *hamDGPtr_;

  for( Int k = 0; k < numElem_[2]; k++ )
    for( Int j = 0; j < numElem_[1]; j++ )
      for( Int i = 0; i < numElem_[0]; i++ ){
        Index3 key( i, j, k );
        if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
          DblNumMat& basisNew = Newbasis.LocalMap()[key];
          DblNumMat& basisOld = Oldbasis.LocalMap()[key];
          DblNumMat& SMat = hamDG.distSMat().LocalMap()[key];

          Int height = basisNew.m();
          Int numBasis = basisNew.n();
          Int numBasisTotal = 0;

          MPI_Allreduce( &numBasis, &numBasisTotal, 1, MPI_INT,
            MPI_SUM, domain_.rowComm );

          for( Int g = 0; g < numBasis; g++ ){
            Real *ptr1 = hamDG.LGLWeight3D().Data();
            Real *ptr2 = basisOld.VecData(g);
            for( Int l = 0; l < height; l++ ){
              *(ptr2++) *= *(ptr1++) ;
            }
          }

          Int width = numBasisTotal;

          Int widthBlocksize = width / mpisizeRow;
          Int heightBlocksize = height / mpisizeRow;

          Int widthLocal = widthBlocksize;
          Int heightLocal = heightBlocksize;

          if(mpirankRow < (width % mpisizeRow)){
            widthLocal = widthBlocksize + 1;
          }

          if(mpirankRow < (height % mpisizeRow)){
            heightLocal = heightBlocksize + 1;
          }

          DblNumMat basisOldRow( heightLocal, width );
          DblNumMat basisNewRow( heightLocal, width );

          AlltoallForward (basisOld, basisOldRow, domain_.rowComm);
          AlltoallForward (basisNew, basisNewRow, domain_.rowComm);

          DblNumMat localMatSTemp( width, width );
          SetValue( localMatSTemp, 0.0 );
          blas::Gemm( 'T', 'N', width, width, heightLocal,
              1.0, basisNewRow.Data(), heightLocal,
              basisOldRow.Data(), heightLocal, 0.0,
              localMatSTemp.Data(), width );

          SMat.Resize(width, width);
          SetValue( SMat, 0.0 );
          MPI_Allreduce( localMatSTemp.Data(),
              SMat.Data(), width*width, 
              MPI_DOUBLE, MPI_SUM, domain_.rowComm );

       if(0){
          DblNumMat Mat2( width, width );
          SetValue( Mat2, 0.0 );

          blas::Gemm( 'T', 'N', width, width, width,
              1.0, SMat.Data(), width,
              SMat.Data(), width, 0.0,
              Mat2.Data(), width );

          for(Int  p = 0; p < Mat2.m(); p++){
              statusOFS <<" Mat2 " << Mat2(p, p)  << std::endl;
          }
        }
        } //Owner element
  }

  std::vector<Index3>  getKeys_list;
  for(typename std::map<ElemMatKey, DblNumMat >::iterator
    get_neighbors_from_DM_iterator = distDMMat.LocalMap().begin();
    get_neighbors_from_DM_iterator != distDMMat.LocalMap().end();
    ++get_neighbors_from_DM_iterator)
  {
    Index3 key =  (get_neighbors_from_DM_iterator->first).first;
    Index3 neighbor_key = (get_neighbors_from_DM_iterator->first).second;

    if(neighbor_key == key)
      continue;
    else
    getKeys_list.push_back(neighbor_key);
  }

  hamDG.distSMat().GetBegin( getKeys_list, NO_MASK );
  hamDG.distSMat().GetEnd( NO_MASK );

  for( Int k = 0; k < numElem_[2]; k++ )
    for( Int j = 0; j < numElem_[1]; j++ )
      for( Int i = 0; i < numElem_[0]; i++ ){
        Index3 key( i, j, k );
        if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
          Int numBasisElem =  esdfParam.numALBElem(i,j,k);

          for(typename  std::map<Index3, DblNumMat>::iterator 
            get_neighbors_iterator = hamDG.distSMat().LocalMap().begin();
            get_neighbors_iterator != hamDG.distSMat().LocalMap().end();
            ++get_neighbors_iterator)
          {
            Index3 key2 =  (*get_neighbors_iterator).first;
            
            DblNumMat& localDM = distDMMat.LocalMap()[ElemMatKey(key, key2)];
            DblNumMat& MatS = hamDG.distSMat().LocalMap()[key];
            DblNumMat localMatTemp( numBasisElem, numBasisElem );
            SetValue( localMatTemp, 0.0 );
                    
            blas::Gemm( 'N', 'N', numBasisElem, numBasisElem, numBasisElem,
              1.0, MatS.Data(), numBasisElem,
              localDM.Data(), numBasisElem, 0.0,
              localMatTemp.Data(), numBasisElem );

            DblNumMat& MatST = hamDG.distSMat().LocalMap()[key2];
            SetValue ( localDM, 0.0 ); 

            blas::Gemm( 'N', 'T', numBasisElem, numBasisElem, numBasisElem,
              1.0, localMatTemp.Data(), numBasisElem,
              MatST.Data(), numBasisElem, 0.0,
              localDM.Data(), numBasisElem );
           } // iterator

         } //Owner element
  }  //for i

  return;
}

void 
SCFDG:: SVDLocalizeBasis ( Int iter, 
                   Index3 numGridExtElem,
//                   Index3 numGridExtElemFine,
                   Index3 numGridElemFine,
                   Index3 numLGLGrid,
                   Spinor& psi,
                   DblNumMat& basis ) 
{
  MPI_Barrier(domain_.rowComm);
  Int mpirank, mpisize;
  MPI_Comm_rank( domain_.comm, &mpirank );
  MPI_Comm_size( domain_.comm, &mpisize );

  Int mpirankRow;  MPI_Comm_rank(domain_.rowComm, &mpirankRow);
  Int mpisizeRow;  MPI_Comm_size(domain_.rowComm, &mpisizeRow);

  HamiltonianDG& hamDG = *hamDGPtr_;
  //Real timeSta, timeEnd;

//  GetTime( timeSta );
  // LL: 11/11/2019.
  // For debugging purposes, let all eigenfunctions have non-negative averages. 
  // This should fix the sign flips due to LAPACK (but this would not fix the problem
  // due to degenerate eigenvectors
  {
    for( Int l = 0; l < psi.NumState(); l++ ){
      Real sum_psi = 0.0;
      for( Int i = 0; i < psi.NumGridTotal(); i++ ){
        sum_psi += psi.Wavefun(i,0,l);
//      for( Int i = 0; i < psi.NumGridTotalFine(); i++ ){
//        sum_psi += psi.WavefunFine(i,0,l);
      }
      Real sgn = (sum_psi >= 0.0) ? 1.0 : -1.0;
      blas::Scal( psi.NumGridTotal(), sgn, psi.Wavefun().VecData(0,l), 1 );
//      blas::Scal( psi.NumGridTotalFine(), sgn, psi.WavefunFine().VecData(0,l), 1 );
    }
  }

  DblNumTns&   LGLWeight3D = hamDG.LGLWeight3D();
  DblNumTns    sqrtLGLWeight3D( numLGLGrid[0], numLGLGrid[1], numLGLGrid[2] );

  Real *ptr1 = LGLWeight3D.Data(), *ptr2 = sqrtLGLWeight3D.Data();
  for( Int i = 0; i < numLGLGrid.prod(); i++ ){
    *(ptr2++) = std::sqrt( *(ptr1++) );
  }

  // Int numBasis = psi.NumState() + 1;
  // Compute numBasis in the presence of numUnusedState
  Int numBasisTotal = psi.NumStateTotal() - numUnusedState_;

  Int numBasis; // local number of basis functions
  numBasis = numBasisTotal / mpisizeRow;
  if( mpirankRow < (numBasisTotal % mpisizeRow) )
    numBasis++;

  Int numBasisTotalTest = 0;
  mpi::Allreduce( &numBasis, &numBasisTotalTest, 1, MPI_SUM, domain_.rowComm );
  if( numBasisTotalTest != numBasisTotal ){
    statusOFS << "numBasisTotal = " << numBasisTotal << std::endl;
    statusOFS << "numBasisTotalTest = " << numBasisTotalTest << std::endl;
    ErrorHandling("Sum{numBasis} = numBasisTotal does not match on local element.");
  }

  // FIXME The constant mode is now not used.
  DblNumMat localBasis( numLGLGrid.prod(), numBasis );
  SetValue( localBasis, 0.0 );

  //FIXME  xmqin This transform remove the normalization
  for( Int l = 0; l < numBasis; l++ ){
    InterpPeriodicUniformToLGL( 
//    InterpPeriodicUniformFineToLGL( 
        numGridExtElem,
//        numGridExtElemFine,
        numLGLGrid,
	psi.Wavefun().VecData(0, l), 
//        psi.WavefunFine().VecData(0, l), 
        localBasis.VecData(l) );
  }

//  GetTime( timeEnd );
//  statusOFS << "Time for interpolating basis = "     << timeEnd - timeSta
//      << " [s]" << std::endl;
  // Post processing for the basis functions on the LGL grid.
  // Perform GEMM and threshold the basis functions for the
  // small matrix.
  //
  // This method might have lower numerical accuracy, but is
  // much more scalable than other known options.

 // GetTime( timeSta );

  // Scale the basis functions by sqrt(weight).  This
  // allows the consequent SVD decomposition of the form
  //
  // X' * W * X
  for( Int g = 0; g < localBasis.n(); g++ ){
    Real *ptr1 = localBasis.VecData(g);
    Real *ptr2 = sqrtLGLWeight3D.Data();
    for( Int l = 0; l < localBasis.m(); l++ ){
      *(ptr1++)  *= *(ptr2++);
    }
  }

  // Convert the column partition to row partition
//  Int height = psi.NumGridTotal() * psi.NumComponent();
  Int heightLGL = numLGLGrid.prod();
//  Int heightElem = numGridElemFine.prod();
  Int width = numBasisTotal;

  Int widthBlocksize = width / mpisizeRow;
//  Int heightBlocksize = height / mpisizeRow;
  Int heightLGLBlocksize = heightLGL / mpisizeRow;
//  Int heightElemBlocksize = heightElem / mpisizeRow;

  Int widthLocal = widthBlocksize;
//  Int heightLocal = heightBlocksize;
  Int heightLGLLocal = heightLGLBlocksize;
//  Int heightElemLocal = heightElemBlocksize;

  if(mpirankRow < (width % mpisizeRow)){
    widthLocal = widthBlocksize + 1;
  }

//  if(mpirankRow < (height % mpisizeRow)){
//    heightLocal = heightBlocksize + 1;
//  }

  if(mpirankRow < (heightLGL % mpisizeRow)){
    heightLGLLocal = heightLGLBlocksize + 1;
  }

//  if(mpirankRow == (heightElem % mpisizeRow)){
//    heightElemLocal = heightElemBlocksize + 1;
//  }

  // FIXME Use AlltoallForward and AlltoallBackward
  // functions to replace below

  DblNumMat MMat( numBasisTotal, numBasisTotal );
  DblNumMat MMatTemp( numBasisTotal, numBasisTotal );
  SetValue( MMat, 0.0 );
  SetValue( MMatTemp, 0.0 );
  Int numLGLGridTotal = numLGLGrid.prod();
  Int numLGLGridLocal = heightLGLLocal;

  DblNumMat localBasisRow(heightLGLLocal, numBasisTotal );
  SetValue( localBasisRow, 0.0 );

  AlltoallForward (localBasis, localBasisRow, domain_.rowComm);

  SetValue( MMatTemp, 0.0 );
  blas::Gemm( 'T', 'N', numBasisTotal, numBasisTotal, numLGLGridLocal,
      1.0, localBasisRow.Data(), numLGLGridLocal, 
      localBasisRow.Data(), numLGLGridLocal, 0.0,
      MMatTemp.Data(), numBasisTotal );

  SetValue( MMat, 0.0 );
  MPI_Allreduce( MMatTemp.Data(), MMat.Data(), numBasisTotal * numBasisTotal, MPI_DOUBLE, MPI_SUM, domain_.rowComm );

  // The following operation is only performed on the
  // master processor in the row communicator

  DblNumMat    U( numBasisTotal, numBasisTotal );
  DblNumMat   VT( numBasisTotal, numBasisTotal );
  DblNumVec    S( numBasisTotal );
  SetValue(U, 0.0);
  SetValue(VT, 0.0);
  SetValue(S, 0.0);

  MPI_Barrier( domain_.rowComm );

  if ( mpirankRow == 0) {
    lapack::QRSVD( numBasisTotal, numBasisTotal, 
        MMat.Data(), numBasisTotal,
        S.Data(), U.Data(), U.m(), VT.Data(), VT.m() );
  } 
  // Broadcast U and S
  MPI_Bcast(S.Data(), numBasisTotal, MPI_DOUBLE, 0, domain_.rowComm);
  MPI_Bcast(U.Data(), numBasisTotal * numBasisTotal, MPI_DOUBLE, 0, domain_.rowComm);
  MPI_Bcast(VT.Data(), numBasisTotal * numBasisTotal, MPI_DOUBLE, 0, domain_.rowComm);

  MPI_Barrier( domain_.rowComm );

  for( Int g = 0; g < numBasisTotal; g++ ){
    S[g] = std::sqrt( S[g] );
  }
  // Total number of SVD basis functions. NOTE: Determined at the first
  // outer SCF and is not changed later. This facilitates the reuse of
  // symbolic factorization
  if( iter == 1 ){
    numSVDBasisTotal_ = 0;    
    for( Int g = 0; g < numBasisTotal; g++ ){
      if( S[g] / S[0] > SVDBasisTolerance_ )
        numSVDBasisTotal_++;
    }
  }
//    else{
//     // Reuse the value saved in numSVDBasisTotal
//      statusOFS 
//        << "NOTE: The number of basis functions (after SVD) " 
//        << "is the same as the number in the first SCF iteration." << std::endl
//        << "This facilitates the reuse of symbolic factorization in PEXSI." 
//        << std::endl;
//    }

  Int numSVDBasisBlocksize = numSVDBasisTotal_ / mpisizeRow;

  Int numSVDBasisLocal = numSVDBasisBlocksize;    

  if(mpirankRow < (numSVDBasisTotal_ % mpisizeRow)){
    numSVDBasisLocal = numSVDBasisBlocksize + 1;
  }

  Int numSVDBasisTotalTest = 0;

  mpi::Allreduce( &numSVDBasisLocal, &numSVDBasisTotalTest, 1, MPI_SUM, domain_.rowComm );

  if( numSVDBasisTotal_ != numSVDBasisTotalTest ){
    statusOFS << "numSVDBasisLocal = " << numSVDBasisLocal << std::endl;
    statusOFS << "numSVDBasisTotal = " << numSVDBasisTotal_ << std::endl;
    statusOFS << "numSVDBasisTotalTest = " << numSVDBasisTotalTest << std::endl;
    ErrorHandling("numSVDBasisTotal != numSVDBasisTotalTest");
  }

  // Multiply X <- X*U in the row-partitioned format
  // Get the first numSVDBasis which are significant.

  basis.Resize( numLGLGridTotal, numSVDBasisLocal );
  DblNumMat basisRow( numLGLGridLocal, numSVDBasisTotal_ );

  SetValue( basis, 0.0 );
  SetValue( basisRow, 0.0 );

  for( Int g = 0; g < numSVDBasisTotal_; g++ ){
    blas::Scal( numBasisTotal, 1.0 / S[g], U.VecData(g), 1 );
  }

  // FIXME
  blas::Gemm( 'N', 'N', numLGLGridLocal, numSVDBasisTotal_,
      numBasisTotal, 1.0, localBasisRow.Data(), numLGLGridLocal,
      U.Data(), numBasisTotal, 0.0, basisRow.Data(), numLGLGridLocal );

  AlltoallBackward (basisRow, basis, domain_.rowComm);

  // FIXME
  // row-partition to column partition via MPI_Alltoallv

  // Unscale the orthogonal basis functions by sqrt of
  // integration weight
  // FIXME

  for( Int g = 0; g < basis.n(); g++ ){
    Real *ptr1 = basis.VecData(g);
    Real *ptr2 = sqrtLGLWeight3D.Data();
    for( Int l = 0; l < basis.m(); l++ ){
      *(ptr1++)  /= *(ptr2++);
    }
  }

// xmqin add for fix the phase of ALBs
/*                    for( Int g = 0; g < basis.n(); g++ ){
                      Real *ptr = basis.VecData(g);
                      Real sum = 0.0;
                      for( Int l = 0; l < basis.m(); l++ ){
                       sum += *ptr++ ;
                      }
                    if (sum <= 0.0) {
                      ptr = basis.VecData(g);
                      for( Int l = 0; l < basis.m(); l++ ){
                          *ptr = - (*ptr);
                           ptr++;
                    }
                   }
                   }
*/

#if ( _DEBUGlevel_ >= 1 )
  statusOFS << " Singular values of the basis = " 
      << S << std::endl;
#endif

#if ( _DEBUGlevel_ >= 1 )
  statusOFS << "Number of significant SVD basis = " 
      << numSVDBasisTotal_ << std::endl;
#endif

//  MPI_Barrier( domain_.rowComm );

//  GetTime( timeEnd );
//#if ( _DEBUGlevel_ >= 1 )
//  statusOFS << "Time for SVD of basis = "     << timeEnd - timeSta
//      << " [s]" << std::endl;
//#endif

  return;
}

} // namespace dgdft
