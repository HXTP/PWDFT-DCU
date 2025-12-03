/*
   Copyright (c) 2012 The Regents of the University of California,
   through Lawrence Berkeley National Laboratory.  

Authors: Lin Lin, Wei Hu and Amartya Banerjee

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
#include  "scf_dg.hpp"
#include  "blas.hpp"
#include  "lapack.hpp"
#include  "utility.hpp"
#include  "environment.hpp"
#ifdef ELSI
#include  "elsi.h"
#endif
#include  "scfdg_upper_end_of_spectrum.hpp"

#ifdef GPU
 #include "cuda_utils.h"
 #include  "cublas.hpp"
// #include "magma.hpp"
 #include "cusolver.hpp"
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
  }         // -----  end of method SCFDG::SCFDG  ----- 


  SCFDG::~SCFDG    (  )
  {
    Int mpirank, mpisize;
    MPI_Comm_rank( domain_.comm, &mpirank );
    MPI_Comm_size( domain_.comm, &mpisize );
#ifdef _USE_PEXSI_

    if( isPEXSIInitialized_ == true ){
      Int mpirankRow;  MPI_Comm_rank(domain_.rowComm, &mpirankRow);
      Int mpisizeRow;  MPI_Comm_size(domain_.rowComm, &mpisizeRow);
      Int mpirankCol;  MPI_Comm_rank(domain_.colComm, &mpirankCol);
      Int mpisizeCol;  MPI_Comm_size(domain_.colComm, &mpisizeCol);

      if( mpirankRow < numProcPEXSICommRow_ & mpirankCol < numProcPEXSICommCol_ ){
        Int info;
        PPEXSIPlanFinalize(
            pexsiPlan_,
            &info );
        if( info != 0 ){
          std::ostringstream msg;
          msg 
            << "PEXSI finalization returns info " << info << std::endl;
          ErrorHandling( msg.str().c_str() );
        }
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
        domain_        = esdfParam.domain;
        mixMaxDim_     = esdfParam.mixMaxDim;
        mixVariable_   = esdfParam.mixVariable;
        mixType_       = esdfParam.mixType;
        mixStepLength_ = esdfParam.mixStepLength;
        eigMinTolerance_  = esdfParam.eigMinTolerance;
        eigTolerance_  = esdfParam.eigTolerance;
        eigMinIter_    = esdfParam.eigMinIter;
        eigMaxIter_    = esdfParam.eigMaxIter;
        scfInnerTolerance_  = esdfParam.scfInnerTolerance;
        scfInnerMinIter_    = esdfParam.scfInnerMinIter;
        scfInnerMaxIter_    = esdfParam.scfInnerMaxIter;
        scfOuterTolerance_  = esdfParam.scfOuterTolerance;
        scfOuterMinIter_    = esdfParam.scfOuterMinIter;
        scfOuterMaxIter_    = esdfParam.scfOuterMaxIter;
        scfOuterEnergyTolerance_ = esdfParam.scfOuterEnergyTolerance;
        numUnusedState_ = esdfParam.numUnusedState;
        SVDBasisTolerance_  = esdfParam.SVDBasisTolerance;
        solutionMethod_   = esdfParam.solutionMethod;
        diagSolutionMethod_   = esdfParam.diagSolutionMethod;

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




        PWSolver_                = esdfParam.PWSolver;

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

      // ~~**~~
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


      // **###**
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
        if(SCFDG_comp_subspace_parallel_ == 1)
          Hmat_top_states_use_Cheby_ = 1; 
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


      MPI_Barrier(domain_.comm);
      MPI_Barrier(domain_.colComm);
      MPI_Barrier(domain_.rowComm);
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

        //dmCol_ = numElem_[0] * numElem_[1] * numElem_[2];
        //dmRow_ = mpisize / dmCol_;

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

	// check check
	Int numPoints =  mpisize / (numProcPEXSICommCol_ * esdfParam.numPole);
	if(numPoints == 0) numPoints +=1;
	//std::cout << " numPoints in PEXSI : " << numPoints << std::endl << std::endl;

        numProcPEXSICommRow_ = std::min( esdfParam.numPole * numPoints, dmRow_ );
	if(numProcPEXSICommRow_ > esdfParam.numPole)
		numProcPEXSICommRow_ = (numProcPEXSICommRow_ / esdfParam.numPole ) * esdfParam.numPole;
	

        numProcTotalPEXSI_   = numProcPEXSICommRow_ * numProcPEXSICommCol_;

        Int isProcPEXSI = 0;
        if( (mpirankRow < numProcPEXSICommRow_) && (mpirankCol < numProcPEXSICommCol_) )
          isProcPEXSI = 1;

        // Transpose way of ranking MPI processors to be consistent with PEXSI.
        Int mpirankPEXSI = mpirankCol + mpirankRow * ( numProcPEXSICommCol_ );

        MPI_Comm_split( domain_.comm, isProcPEXSI, mpirankPEXSI, &pexsiComm_ );

       // Initialize PEXSI
        if( isProcPEXSI ){

#if ( _DEBUGlevel_ >= 0 )
          statusOFS << "mpirank = " << mpirank << ", mpirankPEXSI = " << mpirankPEXSI << std::endl;
#endif

          // FIXME More versatile control of the output of the PEXSI module
          Int outputFileIndex = mpirank;
          if(mpirank > 0)
            outputFileIndex = -1;
#ifndef ELSI
          pexsiPlan_        = PPEXSIPlanInitialize(
              pexsiComm_,
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
#endif
        }
      }
#endif


      // other SCFDG parameters
      {
        hamDGPtr_      = &hamDG;
        distEigSolPtr_ = &distEigSol;
        distfftPtr_    = &distfft;
        ptablePtr_     = &ptable;
        elemPrtn_      = distEigSol.Prtn();
        contxt_        = contxt;


        distDMMat_.SetComm(domain_.colComm);
        distEDMMat_.SetComm(domain_.colComm);
        distFDMMat_.SetComm(domain_.colComm);

        //distEigSolPtr_.SetComm(domain_.colComm);
        //distfftPtr_.SetComm(domain_.colComm);


        mixOuterSave_.SetComm(domain_.colComm);
        mixInnerSave_.SetComm(domain_.colComm);
        dfOuterMat_.SetComm(domain_.colComm);
        dvOuterMat_.SetComm(domain_.colComm);
        dfInnerMat_.SetComm(domain_.colComm);
        dvInnerMat_.SetComm(domain_.colComm);
        vtotLGLSave_.SetComm(domain_.colComm);


        mixOuterSave_.Prtn()  = elemPrtn_;
        mixInnerSave_.Prtn()  = elemPrtn_;
        dfOuterMat_.Prtn()    = elemPrtn_;
        dvOuterMat_.Prtn()    = elemPrtn_;
        dfInnerMat_.Prtn()    = elemPrtn_;
        dvInnerMat_.Prtn()    = elemPrtn_;
        vtotLGLSave_.Prtn()   = elemPrtn_;

#ifdef _USE_PEXSI_
        distDMMat_.Prtn()     = hamDG.HMat().Prtn();
        distEDMMat_.Prtn()     = hamDG.HMat().Prtn();
        distFDMMat_.Prtn()     = hamDG.HMat().Prtn(); 
#endif

        // **###**
        if(SCFDG_use_comp_subspace_ == 1)
          distDMMat_.Prtn()     = hamDG.HMat().Prtn();

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
        }

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
#if ( _DEBUGlevel_ >= 0 )
          statusOFS << " Use superposition of atomic density as initial "
            << "guess for electron density." << std::endl;
#endif
          GetTime( timeSta );
          hamDGPtr_->CalculateAtomDensity( *ptablePtr_, *distfftPtr_ );
          GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
          statusOFS << " Time for calculating the atomic density = " 
            << timeEnd - timeSta << " [s]" << std::endl;
#endif

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
#if ( _DEBUGlevel_ >= 0 )
          statusOFS << " Generating initial density through linear combination of pseudocharges." 
            << std::endl;
#endif

          // Initialize the electron density using the pseudocharge
          // make sure the pseudocharge is initialized
          DistDblNumVec& pseudoCharge = hamDGPtr_->PseudoCharge();

          pseudoCharge.SetComm(domain_.colComm);

          Real sumDensityLocal = 0.0, sumPseudoChargeLocal = 0.0;
          Real sumDensity, sumPseudoCharge;
          Real EPS = 1e-6;

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
                    denVec(p) = ( ppVec(p) > EPS ) ? ppVec(p) : EPS;
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
        }

      } // Restart the density


      // Wavefunctions in the extended element
      if( esdfParam.isRestartWfn ){
#if ( _DEBUGlevel_ >= 0 )
        statusOFS << "Restarting basis functions from WFNEXT_ files"
          << std::endl;
#endif
        std::istringstream wfnStream;      
        SeparateRead( restartWfnFileName_, wfnStream, mpirank );

        for( Int k = 0; k < numElem_[2]; k++ )
          for( Int j = 0; j < numElem_[1]; j++ )
            for( Int i = 0; i < numElem_[0]; i++ ){
              Index3 key( i, j, k );
              if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
                EigenSolver&  eigSol = distEigSolPtr_->LocalMap()[key];
                Spinor& psi = eigSol.Psi();
#ifdef _COMPLEX_
                // FIXME placeholder
                DblNumTns  wavefun;
                DblNumTns  wavefunRead;
#else
                DblNumTns& wavefun = psi.Wavefun();
                DblNumTns  wavefunRead;
#endif
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

      } 
      else{ 
#if ( _DEBUGlevel_ >= 0 )
        statusOFS << "Initial random basis functions in the extended element."
          << std::endl;
#endif

        // Use random initial guess for basis functions in the extended element.
        for( Int k = 0; k < numElem_[2]; k++ )
          for( Int j = 0; j < numElem_[1]; j++ )
            for( Int i = 0; i < numElem_[0]; i++ ){
              Index3 key( i, j, k );
              if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
                EigenSolver&  eigSol = distEigSolPtr_->LocalMap()[key];
                Spinor& psi = eigSol.Psi();

//                UniformRandom( psi.Wavefun() );


                // For debugging purpose
                // Make sure that the initial wavefunctions in each element
                // are the same, when different number of processors are
                // used for intra-element parallelization.
                if(1){ 
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
#ifdef _COMPLEX_
                  // FIXME placeholder
                  NumTns<Real> wavefun ();
#else
                  NumTns<Real>& wavefun = psi.Wavefun();
#endif
                  for (Int k=0; k<nocc; k++) {
#ifdef _COMPLEX_
                    // FIXME placeholder
                    Real *ptr; 
                    Real *ptr1;
#else
                    Real *ptr = psi.Wavefun().MatData(k);
                    Real *ptr1 = psiTemp.Wavefun().MatData(wavefunIdx(k));
#endif
                    for (Int i=0; i<size; i++) {
                      *ptr = *ptr1;
                      ptr = ptr + 1;
                      ptr1 = ptr1 + 1;
                    }
                  }
                }//if(0)
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

#if ( _DEBUGlevel_ >= 0  )
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

#if ( _DEBUGlevel_ >= 0  )
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
       int comm = MPI_Comm_c2f(newComm);
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

       int comm = MPI_Comm_c2f(pexsiComm_);
       c_elsi_set_mpi(comm); 

       c_elsi_customize(1, 1, 1.0E-8, 1, 0, 0); 
     }
#endif

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


  void
    SCFDG::Iterate    (  )
    {

      MPI_Barrier(domain_.comm);
      MPI_Barrier(domain_.colComm);
      MPI_Barrier(domain_.rowComm);

      Int mpirank; MPI_Comm_rank( domain_.comm, &mpirank );
      Int mpisize; MPI_Comm_size( domain_.comm, &mpisize );

      Int mpirankRow;  MPI_Comm_rank(domain_.rowComm, &mpirankRow);
      Int mpisizeRow;  MPI_Comm_size(domain_.rowComm, &mpisizeRow);

      Int mpirankCol;  MPI_Comm_rank(domain_.colComm, &mpirankCol);
      Int mpisizeCol;  MPI_Comm_size(domain_.colComm, &mpisizeCol);

      Real timeSta, timeEnd;
      Real one = 1.0;
      Real zero = 0.0;
      Domain dmElem;
      for( Int d = 0; d < DIM; d++ ){
        dmElem.length[d]   = domain_.length[d] / numElem_[d];
        dmElem.numGrid[d]  = domain_.numGrid[d] / numElem_[d];
        dmElem.numGridFine[d]  = domain_.numGridFine[d] / numElem_[d];
        dmElem.posStart[d] = ( numElem_[d] > 1 ) ? dmElem.length[d] : 0;
      }

      HamiltonianDG&  hamDG = *hamDGPtr_;

      if( XCType_ == "XC_GGA_XC_PBE" ){
        GetTime( timeSta );
        hamDG.CalculateGradDensity(  *distfftPtr_ );
        GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
        statusOFS << " Time for calculating gradient of density is " <<
          timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
      }

      GetTime( timeSta );
      hamDG.CalculateXC( Exc_, hamDG.Epsxc(), hamDG.Vxc(), *distfftPtr_ );
      GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
      statusOFS << " Time for calculating XC is " <<
        timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
      // Compute the Hartree potential
      GetTime( timeSta );
      hamDG.CalculateHartree( hamDG.Vhart(), *distfftPtr_ );
      GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
      statusOFS << " Time for calculating Hartree is " <<
        timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
      // No external potential

      // Compute the total potential
      GetTime( timeSta );
      hamDG.CalculateVtot( hamDG.Vtot() );
      GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
      statusOFS << " Time for calculating Vtot is " <<
        timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

      // The following treatment is not suitable for MD
      if(0){
        // Compute the exchange-correlation potential and energy
        // Only compute the XC if restarting the density, since the initial
        // density can contain some negative contribution
        if( esdfParam.isRestartDensity ){ 
          GetTime( timeSta );
          hamDG.CalculateXC( Exc_, hamDG.Epsxc(), hamDG.Vxc(), *distfftPtr_ );
          GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
          statusOFS << " Time for calculating XC is " <<
            timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
          // Compute the Hartree potential
          GetTime( timeSta );
          hamDG.CalculateHartree( hamDG.Vhart(), *distfftPtr_ );
          GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
          statusOFS << " Time for calculating Hartree is " <<
            timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
          // No external potential

          // Compute the total potential
          GetTime( timeSta );
          hamDG.CalculateVtot( hamDG.Vtot() );
          GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
          statusOFS << " Time for calculating Vtot is " <<
            timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
        }else {
          // Technically needed, otherwise the initial Vtot will be zero 
          // (density = sum of pseudocharge). 
          // Note that the treatment will be different if the initial
          // density is taken from linear superposition of atomic orbitals
          // 
          // In the future this might need to be changed to something else
          // (see more from QE, VASP and QBox)?
          for( Int k = 0; k < numElem_[2]; k++ )
            for( Int j = 0; j < numElem_[1]; j++ )
              for( Int i = 0; i < numElem_[0]; i++ ){
                Index3 key = Index3( i, j, k );
                if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
                  SetValue( hamDG.Vtot().LocalMap()[key], 1.0 );
                }
              } // for (i)
          statusOFS << " Density may be negative, " << 
            "Skip the calculation of XC for the initial setup. " << std::endl;
        }
      }


#ifdef GPU
     cuda_init_vtot();
     cublas::Init();
     cusolver::Init();
     ////MAGMA::Init();
#endif

      Real timeIterStart(0), timeIterEnd(0);
      Real timeTotalStart(0), timeTotalEnd(0);

      bool isSCFConverged = false;

      scfTotalInnerIter_  = 0;

      GetTime( timeTotalStart );

      Int iter;

      // Total number of SVD basis functions. Determined at the first
      // outer SCF and is not changed later. This facilitates the reuse of
      // symbolic factorization
      Int numSVDBasisTotal;    

      // ~~**~~
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      // FIXME: Assuming spinor has only one component here    

      for (iter=1; iter <= scfOuterMaxIter_; iter++) {
        if ( isSCFConverged && (iter >= scfOuterMinIter_ ) ) break;


        // Performing each iteartion
        {
          std::ostringstream msg;
          msg << " Outer SCF iteration # " << iter;
          PrintBlock( statusOFS, msg.str() );
        }

        GetTime( timeIterStart );

        // *********************************************************************
        // Update the local potential in the extended element and the element.
        //
        // NOTE: The modification of the potential on the extended element
        // to reduce the Gibbs phenomena is now in UpdateElemLocalPotential
        // *********************************************************************
        {
          GetTime(timeSta);

          UpdateElemLocalPotential();

          GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
          statusOFS << " Time for updating the local potential in the extended element and the element is " <<
            timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
        }



        // *********************************************************************
        // Solve the basis functions in the extended element
        // *********************************************************************

        Real timeBasisSta, timeBasisEnd;
        GetTime(timeBasisSta);
        // FIXME  magic numbers to fixe the basis
        //        if( (iter <= 5) || (efreeDifPerAtom_ >= 1e-3) ){
        if(1){
          for( Int k = 0; k < numElem_[2]; k++ )
            for( Int j = 0; j < numElem_[1]; j++ )
              for( Int i = 0; i < numElem_[0]; i++ ){
                Index3 key( i, j, k );
                if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
                  EigenSolver&  eigSol = distEigSolPtr_->LocalMap()[key];
                  DblNumVec&    vtotExtElem = eigSol.Ham().Vtot();
                  Index3 numGridExtElem = eigSol.FFT().domain.numGrid;
                  Index3 numGridExtElemFine = eigSol.FFT().domain.numGridFine;
                  Index3 numLGLGrid     = hamDG.NumLGLGridElem();

                  Index3 numGridElemFine    = dmElem.numGridFine;

                  // Skip the interpoation if there is no adaptive local
                  // basis function.  
                  if( eigSol.Psi().NumState() == 0 ){
                    hamDG.BasisLGL().LocalMap()[key].Resize( numLGLGrid.prod(), 0 );  
                    // FIXME
                    hamDG.BasisUniformFine().LocalMap()[key].Resize( numGridElemFine.prod(), 0 );  
                    continue;
                  }

                  // Solve the basis functions in the extended element

                  Real eigTolNow;
                  if( esdfParam.isEigToleranceDynamic ){
                    // Dynamic strategy to control the tolerance
                    if( iter == 1 )
                      // FIXME magic number
                      eigTolNow = 1e-2;
                    else
                      eigTolNow = eigTolerance_;
                  }
                  else{
                    // Static strategy to control the tolerance
                    eigTolNow = eigTolerance_;
                  }

                  Int numEig = (eigSol.Psi().NumStateTotal())-numUnusedState_;
#if ( _DEBUGlevel_ >= 0 ) 
                  statusOFS << " The current tolerance used by the eigensolver is " 
                    << eigTolNow << std::endl;
                  statusOFS << " The target number of converged eigenvectors is " 
                    << numEig << std::endl << std::endl;
#endif

                  GetTime( timeSta );
                  // FIXME multiple choices of solvers for the extended
                  // element should be given in the input file
                  if(Diag_SCF_PWDFT_by_Cheby_ == 1)
                  {
                    // Use CheFSI or LOBPCG on first step 
                    if(iter <= 1)
                    {
                      if(First_SCF_PWDFT_ChebyCycleNum_ <= 0)
                      { 
                        statusOFS << " >>>> Calling LOBPCG for ALB generation on extended element ..." << std::endl;
#ifndef _COMPLEX_
                        eigSol.LOBPCGSolveReal(numEig, eigMaxIter_, eigMinTolerance_, eigTolNow );    
#endif
                      }
                      else
                      {
                        statusOFS << " >>>> Calling CheFSI with random guess for ALB generation on extended element ..." << std::endl;
                        eigSol.FirstChebyStep(numEig, First_SCF_PWDFT_ChebyCycleNum_, First_SCF_PWDFT_ChebyFilterOrder_);
                      }
                      statusOFS << std::endl;

                    }
                    else
                    {
                      statusOFS << " >>>> Calling CheFSI with previous ALBs for generation of new ALBs ..." << std::endl;
                      statusOFS << " >>>> Will carry out " << eigMaxIter_ << " CheFSI cycles." << std::endl;

                      for (int cheby_iter = 1; cheby_iter <= eigMaxIter_; cheby_iter ++)
                      {
                        statusOFS << std::endl << " >>>> CheFSI for ALBs : Cycle " << cheby_iter << " of " << eigMaxIter_ << " ..." << std::endl;
                        eigSol.GeneralChebyStep(numEig, General_SCF_PWDFT_ChebyFilterOrder_);
                      }
                      statusOFS << std::endl;
                    }
                  }
                  else if(Diag_SCF_PWDFT_by_PPCG_ == 1)
                  {
                    // Use LOBPCG on very first step, i.e., while starting from random guess
                    if(iter <= 0)
                    {
                      statusOFS << " >>>> Calling LOBPCG for ALB generation on extended element ..." << std::endl;
#ifndef _COMPLEX_
                      eigSol.LOBPCGSolveReal(numEig, eigMaxIter_, eigMinTolerance_, eigTolNow );    
#endif
                    }
                    else
                    {
                      statusOFS << " >>>> Calling PPCG with previous ALBs for generation of new ALBs ..." << std::endl;
#ifndef _COMPLEX_
                     statusOFS << " mpisizeRow " << mpisizeRow << std::endl;
                     statusOFS << " mpisizeCol " << mpisizeCol << std::endl;

//                    if(mpisizeRow == 1) 
//                         {
//                          statusOFS << " >>>>>>>>> Calling serial  PPCG  ..." << std::endl;
//                          eigSol.PPCGSolveReal(numEig, eigMaxIter_, eigMinTolerance_, eigTolNow, iter, true );
//                         }
//                     else
//                        {
                         statusOFS << " >>>>>>>>> Calling parallel PPCG  ..." << std::endl;
                         eigSol.PPCGSolveReal(numEig, eigMaxIter_, eigMinTolerance_, eigTolNow, iter  );
//                          }



#endif
                    }

                  }             
                  else 
                  {
                    Int eigDynMaxIter = eigMaxIter_;
                    //                if( iter <= 2 ){
                    //                    eigDynMaxIter = 15;
                    //                }
                    //                else{
                    //                    eigDynMaxIter = eigMaxIter_;
                    //                }
#ifndef _COMPLEX_
                    eigSol.LOBPCGSolveReal(numEig, eigDynMaxIter, eigMinTolerance_, eigTolNow );
#endif
                  }

                  GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
                  statusOFS << std::endl << " Eigensolver time = "     << timeEnd - timeSta
                    << " [s]" << std::endl;
#endif


                  // Print out the information
                  statusOFS << std::endl 
                    << "ALB calculation in extended element " << key << std::endl;
                  Real maxRes = 0.0, avgRes = 0.0;
                  for(Int ii = 0; ii < eigSol.EigVal().m(); ii++){
                    if( maxRes < eigSol.ResVal()(ii) ){
                      maxRes = eigSol.ResVal()(ii);
                    }
                    avgRes = avgRes + eigSol.ResVal()(ii);
#if ( _DEBUGlevel_ >= 0 )
                    Print(statusOFS, 
                        "basis#   = ", ii, 
                        "eigval   = ", eigSol.EigVal()(ii),
                        "resval   = ", eigSol.ResVal()(ii));
#endif
                  }
                  avgRes = avgRes / eigSol.EigVal().m();
#if ( _DEBUGlevel_ >= 0 )
                  statusOFS << std::endl;
                  Print(statusOFS, " Max residual of basis = ", maxRes );
                  Print(statusOFS, " Avg residual of basis = ", avgRes );
                  statusOFS << std::endl;
#endif

                  GetTime( timeSta );
                  Spinor& psi = eigSol.Psi();

                  // Assuming that wavefun has only 1 component.  This should
                  // be changed when spin-polarization is added.
#ifdef _COMPLEX_
                  // FIXME placeholder
                  DblNumTns& wavefun() ;
#else
                  DblNumTns& wavefun = psi.Wavefun();
#endif

                  // LL: 11/11/2019.
                  // For debugging purposes, let all eigenfunctions have non-negative averages. 
                  // This should fix the sign flips due to LAPACK (but this would not fix the problem
                  // due to degenerate eigenvectors
                  {
                    for( Int l = 0; l < psi.NumState(); l++ ){
                      Real sum_psi = 0.0;
                      for( Int i = 0; i < psi.NumGridTotal(); i++ ){
                        sum_psi += psi.Wavefun(i,0,l);
                      }
                      Real sgn = (sum_psi >= 0.0) ? 1.0 : -1.0;
                      blas::Scal( psi.NumGridTotal(), sgn, psi.Wavefun().VecData(0,l), 1 );
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
                  DblNumMat localBasis( 
                      numLGLGrid.prod(), 
                      numBasis );

                  SetValue( localBasis, 0.0 );

#ifdef _USE_OPENMP_
//#pragma omp parallel
//                  {
#endif
#ifdef _USE_OPENMP_
//#pragma omp for schedule (dynamic,1) nowait
#endif
#ifdef _COMPLEX_
                  // FIXME placeholder
#else

                 Real timeSta_xm, timeEnd_xm;
                 GetTime( timeSta_xm );

#ifdef GPU
                 Real timeSta2, timeEnd2;
                 Real timeCPU2DCUCopy = 0.0;
                 Int  iterCPU2DCUCopy = 0;
                 Real timeUniformToLGL = 0.0;
                 Int  iterUniformToLGL = 0;

                 cu_PeriodicUniformToLGLMat_x.Resize( numLGLGrid[0], numGridExtElem[0] );
                 cu_PeriodicUniformToLGLMat_y.Resize( numLGLGrid[1], numGridExtElem[1] );
                 cu_PeriodicUniformToLGLMat_z.Resize( numLGLGrid[2], numGridExtElem[2] );
                
                 GetTime( timeSta2 );
                 cuda_memcpy_CPU2GPU( cu_PeriodicUniformToLGLMat_x.Data(), PeriodicUniformToLGLMat_[0].Data(), sizeof(Real)*numLGLGrid[0]*numGridExtElem[0]);
                 cuda_memcpy_CPU2GPU( cu_PeriodicUniformToLGLMat_y.Data(), PeriodicUniformToLGLMat_[1].Data(), sizeof(Real)*numLGLGrid[1]*numGridExtElem[1]);
                 cuda_memcpy_CPU2GPU( cu_PeriodicUniformToLGLMat_z.Data(), PeriodicUniformToLGLMat_[2].Data(), sizeof(Real)*numLGLGrid[2]*numGridExtElem[2]);
                 GetTime( timeEnd2 );
                 iterCPU2DCUCopy = iterCPU2DCUCopy + 3;
                 timeCPU2DCUCopy = timeCPU2DCUCopy +(timeEnd2-timeSta2);              

                 cuDblNumVec  cu_tmp1( numLGLGrid[0] * numGridExtElem[1] * numGridExtElem[2] );
                 cuDblNumVec  cu_tmp2( numLGLGrid[0] * numLGLGrid[1] * numGridExtElem[2] );

                 cuDblNumMat  cu_psiUniform( numGridExtElem[0], numGridExtElem[1] * numGridExtElem[2] );
                 cuDblNumMat  cu_psiLGL( numLGLGrid[0] * numLGLGrid[1], numLGLGrid[2] );

                 for( Int l = 0; l < numBasis; l++ ){
                      GetTime( timeSta2 );
                      cuda_memcpy_CPU2GPU( cu_psiUniform.Data(), wavefun.VecData(0,l), sizeof(Real)*numGridExtElem[0]*numGridExtElem[1]*numGridExtElem[2]);
                      GetTime( timeEnd2 );
                      iterCPU2DCUCopy = iterCPU2DCUCopy + 1;
                      timeCPU2DCUCopy = timeCPU2DCUCopy +(timeEnd2-timeSta2);

                      GetTime( timeSta2 );
                      InterpPeriodicUniformToLGL( 
                          numGridExtElem,
                          numLGLGrid,
                          cu_psiUniform,
                          cu_psiLGL, cu_tmp1, cu_tmp2);
                      GetTime( timeEnd2 );
                      iterUniformToLGL  = iterUniformToLGL + 1;
                      timeUniformToLGL  = timeUniformToLGL  +(timeEnd2-timeSta2);

                      GetTime( timeSta2 );
                      cuda_memcpy_GPU2CPU( localBasis.VecData(l), cu_psiLGL.Data(), sizeof(Real)* numLGLGrid[0] * numLGLGrid[1] * numLGLGrid[2]);
                      GetTime( timeEnd2 );
                      iterCPU2DCUCopy = iterCPU2DCUCopy + 1;
                      timeCPU2DCUCopy = timeCPU2DCUCopy +(timeEnd2-timeSta2);

                    }
#else
                 for( Int l = 0; l < numBasis; l++ ){
                      InterpPeriodicUniformToLGL(
                          numGridExtElem,
                          numLGLGrid,
                          wavefun.VecData(0, l),
                          localBasis.VecData(l) );
                    }
#endif

#endif

                    //#ifdef _USE_OPENMP_
                    //#pragma omp for schedule (dynamic,1) nowait
                    //#endif
                    //              for( Int l = 0; l < psi.NumState(); l++ ){
                    //                // FIXME Temporarily removing the mean value from each
                    //                // basis function and add the constant mode later
                    //                Real avg = blas::Dot( numLGLGrid.prod(),
                    //                    localBasis.VecData(l), 1,
                    //                    LGLWeight3D.Data(), 1 );
                    //                avg /= ( domain_.Volume() / numElem_.prod() );
                    //                for( Int p = 0; p < numLGLGrid.prod(); p++ ){
                    //                  localBasis(p, l) -= avg;
                    //                }
                    //              }
                    //
                    //              // FIXME Temporary adding the constant mode. Should be done more systematically later.
                    //              for( Int p = 0; p < numLGLGrid.prod(); p++ ){
                    //                localBasis(p,psi.NumState()) = 1.0 / std::sqrt( domain_.Volume() / numElem_.prod() );
                    //              }

#ifdef _USE_OPENMP_
//                  }
#endif
                  GetTime( timeEnd_xm );
#if ( _DEBUGlevel_ >= 0 )
                  statusOFS << "Time for iterCPU2DCUCopy  = " << iterCPU2DCUCopy     << "  timeCPU2DCUCopy  = " << timeCPU2DCUCopy << std::endl;
                  statusOFS << "Time for iterUniformToLGL  = " << iterUniformToLGL   << "  timeUniformToLGL  = " << timeUniformToLGL << std::endl;

                  statusOFS << " Time for interpolating basis = "     << timeEnd_xm - timeSta_xm
                    << " [s]" << std::endl;
#endif


                  // Post processing for the basis functions on the LGL grid.
                  // Perform GEMM and threshold the basis functions for the
                  // small matrix.
                  //
                  // This method might have lower numerical accuracy, but is
                  // much more scalable than other known options.
                  if(1){

                   Int  iterCPU2DCUCopy = 0;
                   Int  iterSVD = 0 ;
                   Int  iterGemmT = 0;
                   Int  iterGemmN = 0;
                   Int  iterAllreduce = 0;
                   Int  iterCycle = 0;
                   Int  iterCPUCycle = 0;
                   Int  iterBcast  = 0;
                   Int  iterAlltoallv  = 0;
                   
                   Real timeCPU2DCUCopy = 0.0;
                   Real timeGemmN = 0.0;
                   Real timeGemmT = 0.0;
                   Real timeCPUCycle = 0.0;
                   Real timeBcast = 0.0;
                   Real timeAlltoallv = 0.0;
                   Real timeSVD = 0.0;
                   Real timeAllreduce = 0.0; 
                   Real timeCycle = 0.0;

                   GetTime( timeSta );

                    // Scale the basis functions by sqrt(weight).  This
                    // allows the consequent SVD decomposition of the form
                    //
                    // X' * W * X
                    cuDblNumTns cu_sqrtLGLWeight3D( numLGLGrid[0], numLGLGrid[1], numLGLGrid[2] );
                    cuDblNumMat cu_localBasis(numLGLGrid.prod(),numBasis);
                    GetTime( timeSta2 );
                    cuda_memcpy_CPU2GPU(cu_sqrtLGLWeight3D.Data(), sqrtLGLWeight3D.Data(), sizeof(Real)*numLGLGrid[0]*numLGLGrid[1]*numLGLGrid[2]);
                    cuda_memcpy_CPU2GPU(cu_localBasis.Data(), localBasis.Data(), sizeof(Real)*numLGLGrid.prod()*numBasis);
                    GetTime( timeEnd2 );
                    iterCPU2DCUCopy = iterCPU2DCUCopy + 2;
                    timeCPU2DCUCopy = timeCPU2DCUCopy + ( timeEnd2 - timeSta2 );


                    #if 1
                    GetTime( timeSta2 ); 
                    cuda_X_Equal_XP(cu_localBasis.Data(), cu_sqrtLGLWeight3D.Data(), localBasis.m(), localBasis.n());
                    GetTime( timeEnd2 );
                    iterCycle = iterCycle + 1;
                    timeCycle = timeCycle + ( timeEnd2 - timeSta2 );

                    #else                    
                    for( Int g = 0; g < localBasis.n(); g++ ){
                      Real *ptr1 = localBasis.VecData(g);
                      Real *ptr2 = sqrtLGLWeight3D.Data();
                      for( Int l = 0; l < localBasis.m(); l++ ){
                        *(ptr1++)  *= *(ptr2++);
                      }
                    }
                    #endif
                    // Convert the column partition to row partition

                    Int height = psi.NumGridTotal() * psi.NumComponent();
                    Int heightLGL = numLGLGrid.prod();
                    Int heightElem = numGridElemFine.prod();
                    Int width = numBasisTotal;

                    Int widthBlocksize = width / mpisizeRow;
                    Int heightBlocksize = height / mpisizeRow;
                    Int heightLGLBlocksize = heightLGL / mpisizeRow;
                    Int heightElemBlocksize = heightElem / mpisizeRow;

                    Int widthLocal = widthBlocksize;
                    Int heightLocal = heightBlocksize;
                    Int heightLGLLocal = heightLGLBlocksize;
                    Int heightElemLocal = heightElemBlocksize;

                    if(mpirankRow < (width % mpisizeRow)){
                      widthLocal = widthBlocksize + 1;
                    }

                    if(mpirankRow < (height % mpisizeRow)){
                      heightLocal = heightBlocksize + 1;
                    }

                    if(mpirankRow < (heightLGL % mpisizeRow)){
                      heightLGLLocal = heightLGLBlocksize + 1;
                    }

                    if(mpirankRow == (heightElem % mpisizeRow)){
                      heightElemLocal = heightElemBlocksize + 1;
                    }

                    // FIXME Use AlltoallForward and AlltoallBackward
                    // functions to replace below

                    DblNumMat MMat( numBasisTotal, numBasisTotal );
                    cuDblNumMat cu_MMat( numBasisTotal, numBasisTotal );
                    cuDblNumMat cu_MMatTemp( numBasisTotal, numBasisTotal );
                    DblNumMat MMatTemp( numBasisTotal, numBasisTotal );
//                    SetValue( MMat, 0.0 );
//                    SetValue( MMatTemp, 0.0 );
                    Int numLGLGridTotal = numLGLGrid.prod();
                    Int numLGLGridLocal = heightLGLLocal;
                      cuDblNumMat cu_localBasisRow(heightLGLLocal, numBasisTotal );
 //                   DblNumMat localBasisRow(heightLGLLocal, numBasisTotal );
//                    SetValue( localBasisRow, 0.0 );

                    //DblNumMat localBasisElemRow(heightElemLocal, numBasisTotal );
                    //SetValue( localBasisElemRow, 0.0 );

//                    AlltoallForward (localBasis, localBasisRow, domain_.rowComm);
                     GetTime( timeSta2 );
                     GPU_AlltoallForward (cu_localBasis, cu_localBasisRow, domain_.rowComm);
                     GetTime( timeEnd2 );
                     iterAlltoallv = iterAlltoallv + 1;
                     timeAlltoallv = timeAlltoallv + ( timeEnd2 - timeSta2 );

                    //DblNumMat& localBasisUniformGrid = hamDG.BasisUniformFine().LocalMap()[key];

                    //AlltoallForward (localBasisUniformGrid, localBasisElemRow, domain_.rowComm);

//                    SetValue( MMatTemp, 0.0 );
 //                   blas::Gemm( 'T', 'N', numBasisTotal, numBasisTotal, numLGLGridLocal,
//                        1.0, localBasisRow.Data(), numLGLGridLocal, 
//                        localBasisRow.Data(), numLGLGridLocal, 0.0,
//                        MMatTemp.Data(), numBasisTotal );
                    GetTime( timeSta2 );                         
                    cublas::Gemm( HIPBLAS_OP_T, HIPBLAS_OP_N, numBasisTotal, numBasisTotal, numLGLGridLocal,
                        &one, cu_localBasisRow.Data(), numLGLGridLocal,
                        cu_localBasisRow.Data(), numLGLGridLocal, &zero,
                        cu_MMatTemp.Data(), numBasisTotal );
                    GetTime( timeEnd2 );
                    iterGemmT = iterGemmT + 1;
                    timeGemmT = timeGemmT + ( timeEnd2 - timeSta2 );


                    SetValue( MMat, 0.0 );
//                    MPI_Allreduce( MMatTemp.Data(), MMat.Data(), numBasisTotal * numBasisTotal, MPI_DOUBLE, MPI_SUM, domain_.rowComm );
                    GetTime( timeSta2 );
                    cu_MMatTemp.CopyTo(MMatTemp);
                    GetTime( timeEnd2 );
                    iterCPU2DCUCopy = iterCPU2DCUCopy + 1;
                    timeCPU2DCUCopy = timeCPU2DCUCopy + ( timeEnd2 - timeSta2 );

                    GetTime( timeSta2 );
                    MPI_Allreduce( MMatTemp.Data(), MMat.Data(), numBasisTotal * numBasisTotal, MPI_DOUBLE, MPI_SUM, domain_.rowComm );
                    GetTime( timeEnd2 );
                    iterAllreduce = iterAllreduce + 1;
                    timeAllreduce = timeAllreduce + ( timeEnd2 - timeSta2 );


                    // The following operation is only performed on the
                    // master processor in the row communicator

                    DblNumMat    U( numBasisTotal, numBasisTotal );
                    DblNumMat   VT( numBasisTotal, numBasisTotal );
                    DblNumVec    S( numBasisTotal );
                    SetValue(U, 0.0);
                    SetValue(VT, 0.0);
                    SetValue(S, 0.0);

                    MPI_Barrier( domain_.rowComm );

                    GetTime( timeSta2 );
                    if ( mpirankRow == 0) {
                      lapack::QRSVD( numBasisTotal, numBasisTotal, 
                          MMat.Data(), numBasisTotal,
                          S.Data(), U.Data(), U.m(), VT.Data(), VT.m() );
                    } 
                    GetTime( timeEnd2 );
                    iterSVD = iterSVD + 1;
                    timeSVD = timeSVD + ( timeEnd2 - timeSta2 );

                    GetTime( timeSta2 );
                    // Broadcast U and S
                    MPI_Bcast(S.Data(), numBasisTotal, MPI_DOUBLE, 0, domain_.rowComm);
                    MPI_Bcast(U.Data(), numBasisTotal * numBasisTotal, MPI_DOUBLE, 0, domain_.rowComm);
                    MPI_Bcast(VT.Data(), numBasisTotal * numBasisTotal, MPI_DOUBLE, 0, domain_.rowComm);
                    GetTime( timeEnd2 );
                    iterBcast = iterBcast + 3;
                    timeBcast = timeBcast + ( timeEnd2 - timeSta2 );

                    cuDblNumMat   cu_U( numBasisTotal, numBasisTotal );
                    //cuDblNumMat   cu_VT( numBasisTotal, numBasisTotal );
                    cuDblNumVec    cu_S( numBasisTotal );

                    GetTime( timeSta2 );
                    cu_S.CopyFrom(S);
                    GetTime( timeEnd2 );
                    iterCPU2DCUCopy = iterCPU2DCUCopy + 1;
                    timeCPU2DCUCopy = timeCPU2DCUCopy + ( timeEnd2 - timeSta2 );

                    GetTime( timeSta2 );
                    cuda_sqrt(cu_S.Data(), numBasisTotal);
                    GetTime( timeEnd2 );
                    iterCycle = iterCycle + 1;
                    timeCycle = timeCycle + ( timeEnd2 - timeSta2 );

                    GetTime( timeSta2 );
                    cu_S.CopyTo(S);
                    GetTime( timeEnd2 );
                    iterCPU2DCUCopy = iterCPU2DCUCopy + 1;
                    timeCPU2DCUCopy = timeCPU2DCUCopy + ( timeEnd2 - timeSta2 );


//                    for( Int g = 0; g < numBasisTotal; g++ ){
 //                     S[g] = std::sqrt( S[g] );
 //                   }

                    // Total number of SVD basis functions. NOTE: Determined at the first
                    // outer SCF and is not changed later. This facilitates the reuse of
                    // symbolic factorization
                    GetTime( timeSta2 );
                    if( iter == 1 ){
                      numSVDBasisTotal = 0;    
                      for( Int g = 0; g < numBasisTotal; g++ ){
                        if( S[g] / S[0] > SVDBasisTolerance_ )
                          numSVDBasisTotal++;
                      }
                    }
                    else{
                      // Reuse the value saved in numSVDBasisTotal
                      statusOFS 
                        << "NOTE: The number of basis functions (after SVD) " 
                        << "is the same as the number in the first SCF iteration." << std::endl
                        << "This facilitates the reuse of symbolic factorization in PEXSI." 
                        << std::endl;
                    }
                    GetTime( timeEnd2 );
                    iterCPUCycle = iterCPUCycle + 1;
                    timeCPUCycle = timeCPUCycle + ( timeEnd2 - timeSta2 );


                    Int numSVDBasisBlocksize = numSVDBasisTotal / mpisizeRow;

                    Int numSVDBasisLocal = numSVDBasisBlocksize;    

                    if(mpirankRow < (numSVDBasisTotal % mpisizeRow)){
                      numSVDBasisLocal = numSVDBasisBlocksize + 1;
                    }

                    Int numSVDBasisTotalTest = 0;

                    GetTime( timeSta2 );
                    mpi::Allreduce( &numSVDBasisLocal, &numSVDBasisTotalTest, 1, MPI_SUM, domain_.rowComm );
                    GetTime( timeEnd2 );
                    iterAllreduce = iterAllreduce + 1;
                    timeAllreduce = timeAllreduce + ( timeEnd2 - timeSta2 );

                    if( numSVDBasisTotal != numSVDBasisTotalTest ){
                      statusOFS << "numSVDBasisLocal = " << numSVDBasisLocal << std::endl;
                      statusOFS << "numSVDBasisTotal = " << numSVDBasisTotal << std::endl;
                      statusOFS << "numSVDBasisTotalTest = " << numSVDBasisTotalTest << std::endl;
                      ErrorHandling("numSVDBasisTotal != numSVDBasisTotalTest");
                    }

                    // Multiply X <- X*U in the row-partitioned format
                    // Get the first numSVDBasis which are significant.

                    DblNumMat& basis = hamDG.BasisLGL().LocalMap()[key];

                    basis.Resize( numLGLGridTotal, numSVDBasisLocal );
//                    DblNumMat basisRow( numLGLGridLocal, numSVDBasisTotal );
//                    SetValue( basis, 0.0 );
//                    SetValue( basisRow, 0.0 );
                    cuDblNumMat cu_basisRow( numLGLGridLocal, numSVDBasisTotal );
                    cuDblNumMat cu_basis( numLGLGridTotal, numSVDBasisLocal );

                    GetTime( timeSta2 );
                    for( Int g = 0; g < numSVDBasisTotal; g++ ){
                      blas::Scal( numBasisTotal, 1.0 / S[g], U.VecData(g), 1 );
                    }
                    GetTime( timeEnd2 );
                    iterCycle = iterCycle + 1;
                    timeCycle = timeCycle + ( timeEnd2 - timeSta2 );

                    GetTime( timeSta2 );
                    cu_U.CopyFrom(U);
                    GetTime( timeEnd2 );
                    iterCPU2DCUCopy = iterCPU2DCUCopy + 1;
                    timeCPU2DCUCopy = timeCPU2DCUCopy + ( timeEnd2 - timeSta2 );

                    // FIXME
//                    blas::Gemm( 'N', 'N', numLGLGridLocal, numSVDBasisTotal,
//                        numBasisTotal, 1.0, localBasisRow.Data(), numLGLGridLocal,
//                        U.Data(), numBasisTotal, 0.0, basisRow.Data(), numLGLGridLocal );
//
//                    AlltoallBackward (basisRow, basis, domain_.rowComm);

                    GetTime( timeSta2 );
                    cublas::Gemm( HIPBLAS_OP_N, HIPBLAS_OP_N, numLGLGridLocal, numSVDBasisTotal,
                        numBasisTotal, &one, cu_localBasisRow.Data(), numLGLGridLocal,
                        cu_U.Data(), numBasisTotal, &zero, cu_basisRow.Data(), numLGLGridLocal );
                    GetTime( timeEnd2 );
                    iterGemmN = iterGemmN + 1;
                    timeGemmN = timeGemmN + ( timeEnd2 - timeSta2 );

                    GetTime( timeSta2 );
                    GPU_AlltoallBackward (cu_basisRow, cu_basis, domain_.rowComm);
                    GetTime( timeEnd2 );
                    iterAlltoallv = iterAlltoallv + 1;
                    timeAlltoallv = timeAlltoallv + ( timeEnd2 - timeSta2 );

                    GetTime( timeSta2 );
                    cuda_division( cu_basis.Data(), cu_sqrtLGLWeight3D.Data(), basis.m(), basis.n());
                    GetTime( timeEnd2 );
                    iterCycle = iterCycle + 1;
                    timeCycle = timeCycle + ( timeEnd2 - timeSta2 );

                    GetTime( timeSta2 );
                    cu_basis.CopyTo(basis);
                    GetTime( timeEnd2 );
                    iterCPU2DCUCopy = iterCPU2DCUCopy + 1;
                    timeCPU2DCUCopy = timeCPU2DCUCopy + ( timeEnd2 - timeSta2 );

                    // FIXME
                    // row-partition to column partition via MPI_Alltoallv

                    // Unscale the orthogonal basis functions by sqrt of
                    // integration weight
                    // FIXME



// xmqin add for fix the phase of ALBs
                    GetTime( timeSta2 );
                    for( Int g = 0; g < basis.n(); g++ ){
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
                   GetTime( timeEnd2 );
                   iterCPUCycle = iterCPUCycle + 1;
                   timeCPUCycle = timeCPUCycle + ( timeEnd2 - timeSta2 );

                    // FIXME
                    // FIXME
                    //blas::Gemm( 'N', 'N', heightElemLocal, numSVDBasisTotal,
                    //   numBasisTotal, 1.0, localBasisElemRow.Data(), heightElemLocal,
                    //   U.Data(), numBasisTotal, 0.0, localBasisElemRow.Data(), heightElemLocal );

                    //AlltoallBackward (localBasisElemRow, localBasisUniformGrid, domain_.rowComm);


#if ( _DEBUGlevel_ >= 1 )
                    statusOFS << " Singular values of the basis = " 
                      << S << std::endl;
#endif

#if ( _DEBUGlevel_ >= 0 )
                    statusOFS << " Number of significant SVD basis = " 
                      << numSVDBasisTotal << std::endl;
#endif

                    MPI_Barrier( domain_.rowComm );


                    GetTime( timeEnd );
                    statusOFS << " Time for SVD of basis = "     << timeEnd - timeSta
                      << " [s]" << std::endl;

    statusOFS << "Time for iterCPU2DCUCopy  = " << iterCPU2DCUCopy     << "  timeCPU2DCUCopy  = " << timeCPU2DCUCopy << std::endl;
    statusOFS << "Time for iterCycle        = " << iterCycle           << "  timeCycle        = " << timeCycle << std::endl;
    statusOFS << "Time for iterGemmN        = " << iterGemmN           << "  timeGemmN        = " << timeGemmN << std::endl;
    statusOFS << "Time for iterGemmT        = " << iterGemmT           << "  timeGemmT        = " << timeGemmT << std::endl;
    statusOFS << "Time for iterBcast        = " << iterBcast           << "  timeBcast        = " << timeBcast << std::endl;
    statusOFS << "Time for iterAlltoallv    = " << iterAlltoallv       << "  timeAlltoallv    = " << timeAlltoallv << std::endl;
    statusOFS << "Time for iterAllreduce    = " << iterAllreduce       << "  timeAllreduce    = " << timeAllreduce << std::endl;
    statusOFS << "Time for iterCPUCycle     = " << iterCPUCycle        << "  timeCPUCycle     = " << timeCPUCycle << std::endl;
    statusOFS << "Time for iterSVD          = " << iterSVD             << "  timeSVD          = " << timeSVD << std::endl;



                    //if(1){
                    //  statusOFS << std::endl<< "All processors exit with abort in scf_dg.cpp." << std::endl;
                    //  abort();
                    // }


                    // Transfer psi from coarse grid to fine grid with FFT
                    // FIXME Maybe remove this
                    if(0){ 

                      Int ntot  = psi.NumGridTotal();
                      Int ntotFine  = numGridExtElemFine.prod ();
                      Int ncom  = psi.NumComponent();
                      Int nocc  = psi.NumState();

                      //DblNumMat psiUniformGridFine( 
                      //ntotFine, 
                      //numBasis );

                      DblNumMat localBasisUniformGrid;

                      localBasisUniformGrid.Resize( numGridElemFine.prod(), numBasis );

                      SetValue( localBasisUniformGrid, 0.0 );

                      //Fourier& fft = eigSol.FFT();

                      for (Int k=0; k<nocc; k++) {

                        //                    for( Int i = 0; i < ntot; i++ ){
                        //                      fft.inputComplexVec(i) = Complex( psi.Wavefun(i,0,k), 0.0 );
                        //                    }
                        //
                        //                    fftw_execute( fft.forwardPlan );
                        //
                        //                    // fft Coarse to Fine 
                        //
                        //                    SetValue( fft.outputComplexVecFine, Z_ZERO );
                        //                    for( Int i = 0; i < ntot; i++ ){
                        //                      fft.outputComplexVecFine(fft.idxFineGrid(i)) = fft.outputComplexVec(i);
                        //                    }
                        //
                        //                    fftw_execute( fft.backwardPlanFine );
                        //
                        //                    DblNumVec psiTemp (ntotFine);
                        //                    SetValue( psiTemp, 0.0 ); 
                        //
                        //                    for( Int i = 0; i < ntotFine; i++ ){
                        //
                        //                      psiTemp(i) = fft.inputComplexVecFine(i).real() / (double(ntot) * double(ntotFine));
                        //
                        //                    }
                        //
#ifdef _COMPLEX_
                        // FIXME placeholder
#else
                        InterpPeriodicGridExtElemToGridElem( 
                            numGridExtElem,
                            numGridElemFine,
                            wavefun.VecData(0, k), 
                            localBasisUniformGrid.VecData(k) );
#endif

                      } // for k


                      DblNumMat localBasisElemRow( heightElemLocal, numBasisTotal );
                      SetValue( localBasisElemRow, 0.0 );

                      DblNumMat  basisUniformGridRow( heightElemLocal, numSVDBasisTotal );
                      SetValue( basisUniformGridRow, 0.0 );

                      DblNumMat& basisUniformGrid = hamDG.BasisUniformFine().LocalMap()[key];
                      basisUniformGrid.Resize( numGridElemFine.prod(), numSVDBasisLocal );
                      SetValue( basisUniformGrid, 0.0 );

                      AlltoallForward (localBasisUniformGrid, localBasisElemRow, domain_.rowComm);

                      // FIXME
                      blas::Gemm( 'N', 'N', heightElemLocal, numSVDBasisTotal,
                          numBasisTotal, 1.0, localBasisElemRow.Data(), heightElemLocal,
                          U.Data(), numBasisTotal, 0.0, basisUniformGridRow.Data(), heightElemLocal );

                      AlltoallBackward (  basisUniformGridRow, basisUniformGrid, domain_.rowComm);

                    } // if (1)


                    MPI_Barrier( domain_.rowComm );

                  } // if(1)


                } // own this element
              } // for (i)
        } // if( perform ALB calculation )


        GetTime( timeBasisEnd );

        statusOFS << std::endl << " Time for generating ALB function is " <<
          timeBasisEnd - timeBasisSta << " [s]" << std::endl << std::endl;


        //    MPI_Barrier( domain_.comm );

        // ~~**~~
        // ~~~~~~~~~~~~~~~~~~~~~~~~    
        // Routine for re-orienting eigenvectors based on current basis set
        if(Diag_SCFDG_by_Cheby_ == 1)
        {
          Real timeSta2, timeEnd2;
          Real extra_timeSta, extra_timeEnd;
          Real timeInit = 0.0;
          Int  iterInit = 0;
          Real timeCPU2DCUCopy =0.0;
          Int  iterCPU2DCUCopy = 0;
          Real timeGemmT = 0.0;
          Int  iterGemmT = 0;
          Real timeAlltoallv =0.0;
          Int  iterAlltoallv = 0;
          Real timeGemmN = 0.0;
          Int  iterGemmN = 0;
          Real timeAllreduce = 0.0;
          Int  iterAllreduce = 0;


          if(  ALB_LGL_deque_.size() > 0)
          {  
            statusOFS << std::endl << " Rotating the eigenvectors from the previous step ... ";
            GetTime(timeSta);
          }
          // Figure out the element that we own using the standard loop
          for( Int k = 0; k < numElem_[2]; k++ )
            for( Int j = 0; j < numElem_[1]; j++ )
              for( Int i = 0; i < numElem_[0]; i++ )
              {
                Index3 key( i, j, k );

                // If we own this element
                if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) )
                {
                  GetTime(timeSta2);
                  EigenSolver&  eigSol = distEigSolPtr_->LocalMap()[key];
                  Index3 numLGLGrid    = hamDG.NumLGLGridElem();
                  Spinor& psi = eigSol.Psi();

                  // Assuming that wavefun has only 1 component, i.e., spin-unpolarized
                  // These are element specific quantities

                  // This is the band distributed local basis
                  DblNumMat& ref_band_distrib_local_basis = hamDG.BasisLGL().LocalMap()[key];
                  DblNumMat band_distrib_local_basis(ref_band_distrib_local_basis.m(),ref_band_distrib_local_basis.n());

                  blas::Copy(ref_band_distrib_local_basis.m() * ref_band_distrib_local_basis.n(), 
                      ref_band_distrib_local_basis.Data(), 1,
                      band_distrib_local_basis.Data(), 1);   

                  // LGL weights and sqrt weights
                  DblNumTns&   LGLWeight3D = hamDG.LGLWeight3D();
                  DblNumTns    sqrtLGLWeight3D( numLGLGrid[0], numLGLGrid[1], numLGLGrid[2] );

                  Real *ptr1 = LGLWeight3D.Data(), *ptr2 = sqrtLGLWeight3D.Data();
                  for( Int i = 0; i < numLGLGrid.prod(); i++ ){
                    *(ptr2++) = std::sqrt( *(ptr1++) );
                  }


                  // Scale band_distrib_local_basis using sqrt(weights)
                  for( Int g = 0; g < band_distrib_local_basis.n(); g++ ){
                    Real *ptr1 = band_distrib_local_basis.VecData(g);
                    Real *ptr2 = sqrtLGLWeight3D.Data();
                    for( Int l = 0; l < band_distrib_local_basis.m(); l++ ){
                      *(ptr1++)  *= *(ptr2++);
                    }
                  }

                  // Figure out a few dimensions for the row-distribution
                  Int heightLGL = numLGLGrid.prod();
                  // FIXME! This assumes that SVD does not get rid of basis
                  // In the future there should be a parameter to return the
                  // number of basis functions on the local DG element
                  Int width = psi.NumStateTotal() - numUnusedState_;

                  Int widthBlocksize = width / mpisizeRow;
                  Int widthLocal = widthBlocksize;

                  Int heightLGLBlocksize = heightLGL / mpisizeRow;
                  Int heightLGLLocal = heightLGLBlocksize;

                  if(mpirankRow < (width % mpisizeRow)){
                    widthLocal = widthBlocksize + 1;
                  }

                  if(mpirankRow < (heightLGL % mpisizeRow)){
                    heightLGLLocal = heightLGLBlocksize + 1;
                  }


                  // Convert from band distribution to row distribution
                  DblNumMat row_distrib_local_basis(heightLGLLocal, width);
                  SetValue(row_distrib_local_basis, 0.0);  
                  GetTime(timeEnd2);
//                  statusOFS << " CheFSI init " << timeEnd2 - timeSta2 << " [s] " << std::endl;
                  iterInit = iterInit + 1;
                  timeInit = timeInit + timeEnd2 - timeSta2;


                  statusOFS << std::endl << " AlltoallForward: Changing distribution of local basis functions ... ";
                  GetTime(extra_timeSta);

#ifdef GPU // xmqin 2022/6/2
                  cuDblNumMat cu_row_distrib_local_basis(heightLGLLocal, width);
                  cuDblNumMat cu_band_distrib_local_basis(ref_band_distrib_local_basis.m(),ref_band_distrib_local_basis.n());
                  
                  GetTime(timeSta2);
                  cu_band_distrib_local_basis.CopyFrom(band_distrib_local_basis);
                  GetTime(timeEnd2);
                  iterCPU2DCUCopy = iterCPU2DCUCopy + 1;
                  timeCPU2DCUCopy = timeCPU2DCUCopy +(timeEnd2-timeSta2);

                  GetTime(timeSta2);
                  GPU_AlltoallForward (cu_band_distrib_local_basis, cu_row_distrib_local_basis, domain_.rowComm);
                  GetTime(timeEnd2);
                  iterAlltoallv = iterAlltoallv + 1;
                  timeAlltoallv = timeAlltoallv +(timeEnd2-timeSta2);

                  GetTime(timeSta2);
                  cu_row_distrib_local_basis.CopyTo(row_distrib_local_basis);
                  GetTime(timeEnd2);
                  iterCPU2DCUCopy = iterCPU2DCUCopy + 1;
                  timeCPU2DCUCopy = timeCPU2DCUCopy +(timeEnd2-timeSta2);

#else
                  AlltoallForward(band_distrib_local_basis, row_distrib_local_basis, domain_.rowComm);
#endif

                  GetTime(extra_timeEnd);
                  statusOFS << " Done. ( " << (extra_timeEnd - extra_timeSta) << " s)";

                  // Push the row-distributed matrix into the deque
                  ALB_LGL_deque_.push_back( row_distrib_local_basis );


                  // If the deque has 2 elements, compute the overlap and perform a rotation of the eigenvectors
                  if( ALB_LGL_deque_.size() == 2)
                  {

                    GetTime(extra_timeSta);
                    statusOFS << std::endl << " Computing the overlap matrix using basis sets on LGL grid ... ";

                    // Compute the local overlap matrix V2^T * V1            
                    DblNumMat Overlap_Mat( width, width );
                    DblNumMat Overlap_Mat_Temp( width, width );
                    SetValue( Overlap_Mat, 0.0 );
                    SetValue( Overlap_Mat_Temp, 0.0 );

#ifdef GPU // xmqin 2022/6/2
//                    cuDblNumMat cu_Overlap_Mat( width, width );
                    cuDblNumMat cu_Overlap_Mat_Temp( width, width );
                    cuDblNumMat cu_ALB_LGL_deque_0( heightLGLLocal, width);
                    cuDblNumMat cu_ALB_LGL_deque_1( heightLGLLocal, width);
                    
                    GetTime(timeSta2);
                    cuda_memcpy_CPU2GPU( cu_ALB_LGL_deque_0.Data(), ALB_LGL_deque_[0].Data(), sizeof(Real)*heightLGLLocal*width);     
                    cuda_memcpy_CPU2GPU( cu_ALB_LGL_deque_1.Data(), ALB_LGL_deque_[1].Data(), sizeof(Real)*heightLGLLocal*width);     
                    GetTime( timeEnd2 );
                    iterCPU2DCUCopy = iterCPU2DCUCopy + 2;
                    timeCPU2DCUCopy = timeCPU2DCUCopy +(timeEnd2-timeSta2);

                    GetTime(timeSta2);
                    cublas::Gemm( HIPBLAS_OP_T, HIPBLAS_OP_N, width, width, heightLGLLocal,
                        &one, cu_ALB_LGL_deque_1.Data(), heightLGLLocal,
                        cu_ALB_LGL_deque_0.Data(), heightLGLLocal,
                        &zero, cu_Overlap_Mat_Temp.Data(), width );
                    GetTime( timeEnd2 );
                    iterGemmT = iterGemmT + 1;
                    timeGemmT = timeGemmT +(timeEnd2-timeSta2);

                    GetTime(timeSta2);
                    cu_Overlap_Mat_Temp.CopyTo(Overlap_Mat_Temp);
                    GetTime( timeEnd2 );
                    iterCPU2DCUCopy = iterCPU2DCUCopy + 1;
                    timeCPU2DCUCopy = timeCPU2DCUCopy +(timeEnd2-timeSta2);

#else
                    double *ptr_0 = ALB_LGL_deque_[0].Data();
                    double *ptr_1 = ALB_LGL_deque_[1].Data();

                    blas::Gemm( 'T', 'N', width, width, heightLGLLocal,
                        1.0, ptr_1, heightLGLLocal, 
                        ptr_0, heightLGLLocal, 
                        0.0, Overlap_Mat_Temp.Data(), width );
#endif

                    // Reduce along rowComm (i.e., along the intra-element direction)
                    // to compute the actual overlap matrix
                    GetTime(timeSta2);
                    MPI_Allreduce( Overlap_Mat_Temp.Data(), 
                        Overlap_Mat.Data(), 
                        width * width, 
                        MPI_DOUBLE, 
                        MPI_SUM, 
                        domain_.rowComm );
                    GetTime( timeEnd2 );
                    iterAllreduce = iterAllreduce + 1;
                    timeAllreduce = timeAllreduce +(timeEnd2-timeSta2);


                    GetTime(extra_timeEnd);
                    statusOFS << " Done. ( " << (extra_timeEnd - extra_timeSta) << " s)";

                    // Rotate the current eigenvectors : This can also be done in parallel
                    // at the expense of an AllReduce along rowComm

                    statusOFS << std::endl << " Rotating the eigenvectors using overlap matrix ... ";
                    GetTime(extra_timeSta);

                    DblNumMat &eigvecs_local = (hamDG.EigvecCoef().LocalMap().begin())->second;               

                    DblNumMat temp_loc_eigvecs_buffer;
                    temp_loc_eigvecs_buffer.Resize(eigvecs_local.m(), eigvecs_local.n());

                    blas::Copy( eigvecs_local.m() * eigvecs_local.n(),
                        eigvecs_local.Data(), 1,
                        temp_loc_eigvecs_buffer.Data(), 1 );


#ifdef GPU // xmqin 2022/6/2
                    cuDblNumMat cu_Overlap_Mat( width, width );
                    cu_Overlap_Mat.CopyFrom(Overlap_Mat);

                    Int eigvecs_m = eigvecs_local.m();
                    Int eigvecs_n = eigvecs_local.n();
                    
                    cuDblNumMat cu_eigvecs_local(eigvecs_m,eigvecs_n);
//                    cu_eigvecs_local.CopyFrom(eigvecs_local);
 
                    cuDblNumMat cu_temp_loc_eigvecs_buffer(eigvecs_m, eigvecs_n);
                   
                    GetTime( timeSta2 );
                    cu_temp_loc_eigvecs_buffer.CopyFrom(temp_loc_eigvecs_buffer);
                    GetTime( timeEnd2 );
                    iterCPU2DCUCopy = iterCPU2DCUCopy + 1;
                    timeCPU2DCUCopy = timeCPU2DCUCopy +(timeEnd2-timeSta2);

                   GetTime( timeSta2 );
                   cublas::Gemm( HIPBLAS_OP_N, HIPBLAS_OP_N, width, eigvecs_n, width,
                        &one, cu_Overlap_Mat.Data(), width,
                        cu_temp_loc_eigvecs_buffer.Data(),  eigvecs_m ,
                        &zero, cu_eigvecs_local.Data(), eigvecs_m);
                   GetTime( timeEnd2 );
                   iterGemmN = iterGemmN + 1;
                   timeGemmN = timeGemmN +(timeEnd2-timeSta2);

                    GetTime( timeSta2 );
                    cu_eigvecs_local.CopyTo(eigvecs_local);
                    GetTime( timeEnd2 );
                    iterCPU2DCUCopy = iterCPU2DCUCopy + 1;
                    timeCPU2DCUCopy = timeCPU2DCUCopy +(timeEnd2-timeSta2);

#else

                    blas::Gemm( 'N', 'N', Overlap_Mat.m(), eigvecs_local.n(), Overlap_Mat.n(), 
                        1.0, Overlap_Mat.Data(), Overlap_Mat.m(), 
                        temp_loc_eigvecs_buffer.Data(), temp_loc_eigvecs_buffer.m(), 
                        0.0, eigvecs_local.Data(), eigvecs_local.m());
#endif

                    GetTime(extra_timeEnd);
                    statusOFS << " Done. ( " << (extra_timeEnd - extra_timeSta) << " s)" << std::endl;


                    ALB_LGL_deque_.pop_front();
                  }        

                } // End of if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) )

              } // End of loop over key indices i.e., for( Int i = 0; i < numElem_[0]; i++ )

          if( iter > 1)
          {
            GetTime(timeEnd);
            statusOFS << std::endl << " All steps of basis rotation completed. ( " << (timeEnd - timeSta) << " s )" << std::endl;

          }

    statusOFS << "Time for iterInit         = " << iterInit            << "  timeInit         = " << timeInit << std::endl;
    statusOFS << "Time for iterGemmN        = " << iterGemmN           << "  timeGemmN        = " << timeGemmN << std::endl;
    statusOFS << "Time for iterGemmT        = " << iterGemmT           << "  timeGemmT        = " << timeGemmT << std::endl;
    statusOFS << "Time for iterCPU2DCUCopy  = " << iterCPU2DCUCopy     << "  timeCPU2DCUCopy  = " << timeCPU2DCUCopy << std::endl;
    statusOFS << "Time for iterAlltoallv    = " << iterAlltoallv       << "  timeAlltoallv    = " << timeAlltoallv << std::endl;
    statusOFS << "Time for iterAllreduce    = " << iterAllreduce       << "  timeAllreduce    = " << timeAllreduce << std::endl;


        } // End of if(Diag_SCFDG_by_Cheby_ == 1)



        // *********************************************************************
        // Inner SCF iteration 
        //
        // Assemble and diagonalize the DG matrix until convergence is
        // reached for updating the basis functions in the next step.
        // *********************************************************************

        GetTime(timeSta);

        // Save the mixing variable in the outer SCF iteration 
        for( Int k = 0; k < numElem_[2]; k++ )
          for( Int j = 0; j < numElem_[1]; j++ )
            for( Int i = 0; i < numElem_[0]; i++ ){
              Index3 key( i, j, k );
              if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
                if( mixVariable_ == "density" ){
                  DblNumVec& oldVec = hamDG.Density().LocalMap()[key];
                  mixOuterSave_.LocalMap()[key] = oldVec;
                }
                else if( mixVariable_ == "potential" ){
                  DblNumVec& oldVec = hamDG.Vtot().LocalMap()[key];
                  mixOuterSave_.LocalMap()[key] = oldVec;
                }
              } // own this element
            } // for (i)

        // Main function here
        InnerIterate( iter );

        MPI_Barrier( domain_.comm );
        GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
        statusOFS << " Time for all inner SCF iterations is " <<
          timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

        // *********************************************************************
        // Post processing 
        // *********************************************************************

        Int numAtom = hamDG.AtomList().size();
        efreeDifPerAtom_ = std::abs(Efree_ - EfreeHarris_) / numAtom;

        // Energy based convergence parameters
        if(iter > 1)
        {        
          md_scf_eband_old_ = md_scf_eband_;
          md_scf_etot_old_ = md_scf_etot_;      
        }
        else
        {
          md_scf_eband_old_ = 0.0;                
          md_scf_etot_old_ = 0.0;
        } 

        md_scf_eband_ = Ekin_;
        md_scf_eband_diff_ = std::abs(md_scf_eband_old_ - md_scf_eband_) / double(numAtom);
        md_scf_etot_ = Etot_;
        //md_scf_etot_ = EfreeHarris_;
        md_scf_etot_diff_ = std::abs(md_scf_etot_old_ - md_scf_etot_) / double(numAtom);
        //Int numAtom = hamDG.AtomList().size();;



        // Compute the error of the mixing variable 
        {
          Real normMixDifLocal = 0.0, normMixOldLocal = 0.0;
          Real normMixDif, normMixOld;
          for( Int k = 0; k < numElem_[2]; k++ )
            for( Int j = 0; j < numElem_[1]; j++ )
              for( Int i = 0; i < numElem_[0]; i++ ){
                Index3 key( i, j, k );
                if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
                  if( mixVariable_ == "density" ){
                    DblNumVec& oldVec = mixOuterSave_.LocalMap()[key];
                    DblNumVec& newVec = hamDG.Density().LocalMap()[key];

                    for( Int p = 0; p < oldVec.m(); p++ ){
                      normMixDifLocal += pow( oldVec(p) - newVec(p), 2.0 );
                      normMixOldLocal += pow( oldVec(p), 2.0 );
                    }
                  }
                  else if( mixVariable_ == "potential" ){
                    DblNumVec& oldVec = mixOuterSave_.LocalMap()[key];
                    DblNumVec& newVec = hamDG.Vtot().LocalMap()[key];

                    for( Int p = 0; p < oldVec.m(); p++ ){
                      normMixDifLocal += pow( oldVec(p) - newVec(p), 2.0 );
                      normMixOldLocal += pow( oldVec(p), 2.0 );
                    }
                  }
                } // own this element
              } // for (i)


          mpi::Allreduce( &normMixDifLocal, &normMixDif, 1, MPI_SUM, 
              domain_.colComm );
          mpi::Allreduce( &normMixOldLocal, &normMixOld, 1, MPI_SUM,
              domain_.colComm );

          normMixDif = std::sqrt( normMixDif );
          normMixOld = std::sqrt( normMixOld );

          scfOuterNorm_    = normMixDif / normMixOld;



          Print(statusOFS, "OUTERSCF: EfreeHarris                 = ", EfreeHarris_ ); 
          //            FIXME
          //            Print(statusOFS, "OUTERSCF: EfreeSecondOrder            = ", EfreeSecondOrder_ ); 
          Print(statusOFS, "OUTERSCF: Efree                       = ", Efree_ ); 
          Print(statusOFS, "OUTERSCF: norm(out-in)/norm(in) = ", scfOuterNorm_ ); 
          Print(statusOFS, "OUTERSCF: Efree diff per atom   = ", efreeDifPerAtom_ ); 

          if(useEnergySCFconvergence_ == 1)
          {
            Print(statusOFS, "OUTERSCF: MD SCF Etot diff (per atom)           = ", md_scf_etot_diff_); 
            Print(statusOFS, "OUTERSCF: MD SCF Eband diff (per atom)          = ", md_scf_eband_diff_); 
          }
          statusOFS << std::endl;
        }

        //        // Print out the state variables of the current iteration
        //    PrintState( );


        // Check for convergence
        if(useEnergySCFconvergence_ == 0)
        {  
          if( (iter >= 2) && 
              ( (scfOuterNorm_ < scfOuterTolerance_) && 
                (efreeDifPerAtom_ < scfOuterEnergyTolerance_) ) ){
            /* converged */
            statusOFS << " Outer SCF is converged in " << iter << " steps !" << std::endl;
            isSCFConverged = true;
          }
        }
        else
        {
          if( (iter >= 2) && 
              (md_scf_etot_diff_ < md_scf_etot_diff_tol_) &&
              (md_scf_eband_diff_ < md_scf_eband_diff_tol_) )
          {
            // converged via energy criterion
            statusOFS << " Outer SCF is converged via energy condition in " << iter << " steps !" << std::endl;
            isSCFConverged = true;

          }

        }
        // Potential mixing for the outer SCF iteration. or no mixing at all anymore?
        // It seems that no mixing is the best.


        GetTime( timeIterEnd );
        statusOFS << " Time for this SCF iteration = " << timeIterEnd - timeIterStart
          << " [s]" << std::endl;
      } // for( iter )

      GetTime( timeTotalEnd );

      statusOFS << std::endl;
      statusOFS << "Total time for all SCF iterations = " << 
        timeTotalEnd - timeTotalStart << " [s]" << std::endl;
      if(scfdg_ion_dyn_iter_ >= 1)
      {
        statusOFS << " Ion dynamics iteration " << scfdg_ion_dyn_iter_ << " : ";
      }

      if( isSCFConverged == true ){
        statusOFS << " Total number of outer SCF steps for SCF convergence = " <<
          iter - 1 << std::endl;
      }
      else{
        statusOFS << " Total number of outer SCF steps (SCF not converged) = " <<
          scfOuterMaxIter_ << std::endl;

      }

      //    if(0)
      //    {
      //      // Output the electron density on the LGL grid in each element
      //      std::ostringstream rhoStream;      
      //
      //      NumTns<std::vector<DblNumVec> >& LGLGridElem =
      //        hamDG.LGLGridElem();
      //
      //      for( Int k = 0; k < numElem_[2]; k++ )
      //        for( Int j = 0; j < numElem_[1]; j++ )
      //          for( Int i = 0; i < numElem_[0]; i++ ){
      //            Index3 key( i, j, k );
      //            if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
      //              DblNumVec&  denVec = hamDG.DensityLGL().LocalMap()[key];
      //              std::vector<DblNumVec>& grid = LGLGridElem(i, j, k);
      //              for( Int d = 0; d < DIM; d++ ){
      //                serialize( grid[d], rhoStream, NO_MASK );
      //              }
      //              serialize( denVec, rhoStream, NO_MASK );
      //            }
      //          } // for (i)
      //      SeparateWrite( "DENLGL", rhoStream );
      //    }



      // *********************************************************************
      // Calculate the VDW contribution and the force
      // *********************************************************************
      Real timeForceSta, timeForceEnd;
      GetTime( timeForceSta );
      if( solutionMethod_ == "diag" ){

        if(SCFDG_comp_subspace_engaged_ == false)
        {
//          statusOFS << std::endl << " Computing forces using eigenvectors ..." << std::endl;
//          hamDG.CalculateForce( *distfftPtr_ );
        }
        else
        {
          double extra_timeSta, extra_timeEnd;

          statusOFS << std::endl << " Computing forces using Density Matrix ...";
          statusOFS << std::endl << " Computing full Density Matrix for Complementary Subspace method ...";
          GetTime(extra_timeSta);

          // Compute the full DM in the complementary subspace method
          scfdg_complementary_subspace_compute_fullDM();

          GetTime(extra_timeEnd);

          statusOFS << std::endl << " DM Computation took " << (extra_timeEnd - extra_timeSta) << " s." << std::endl;

          // Call the PEXSI force evaluator
          hamDG.CalculateForceDM( *distfftPtr_, distDMMat_ );        
        }
      }
      else if( solutionMethod_ == "pexsi" ){
        hamDG.CalculateForceDM( *distfftPtr_, distDMMat_ );
      }
      GetTime( timeForceEnd );
//      statusOFS << " Time for computing the force is " <<
//        timeForceEnd - timeForceSta << " [s]" << std::endl << std::endl;

      // Calculate the VDW energy
      if( VDWType_ == "DFT-D2"){
        CalculateVDW ( Evdw_, forceVdw_ );
        // Update energy
        Etot_  += Evdw_;
        Efree_ += Evdw_;
        EfreeHarris_ += Evdw_;
        Ecor_  += Evdw_;

        // Update force
        std::vector<Atom>& atomList = hamDG.AtomList();
        for( Int a = 0; a < atomList.size(); a++ ){
          atomList[a].force += Point3( forceVdw_(a,0), forceVdw_(a,1), forceVdw_(a,2) );
        }
      } 


      // Output the information after SCF
      {

        // Print out the energy
        PrintBlock( statusOFS, "Energy" );
        statusOFS 
          << "NOTE:  Ecor  = Exc - EVxc - Ehart - Eself + Evdw" << std::endl
          << "       Etot  = Ekin + Ecor" << std::endl
          << "       Efree = Etot    + Entropy" << std::endl << std::endl;
        Print(statusOFS, "! EfreeHarris     = ",  EfreeHarris_, "[au]");
        Print(statusOFS, "! Etot            = ",  Etot_, "[au]");
        Print(statusOFS, "! Efree           = ",  Efree_, "[au]");
        Print(statusOFS, "! Evdw            = ",  Evdw_, "[au]"); 
        Print(statusOFS, "! Fermi           = ",  fermi_, "[au]");

        statusOFS << std::endl << "  Convergence information : " << std::endl;
        Print(statusOFS, "! norm(out-in)/norm(in) = ",  scfOuterNorm_ ); 
        Print(statusOFS, "! Efree diff per atom   = ",  efreeDifPerAtom_, "[au]"); 

        if(useEnergySCFconvergence_ == 1)
        {
          Print(statusOFS, "! MD SCF Etot diff (per atom)  = ",  md_scf_etot_diff_, "[au]"); 
          Print(statusOFS, "! MD SCF Eband diff (per atom) = ",  md_scf_eband_diff_, "[au]"); 
        }
      }

/*      {
        // Print out the force
        PrintBlock( statusOFS, "Atomic Force" );

        Point3 forceCM(0.0, 0.0, 0.0);
        std::vector<Atom>& atomList = hamDG.AtomList();
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
*/


      // *********************************************************************
      // Output information
      // *********************************************************************

      // Output the atomic structure, and other information for describing
      // density, basis functions etc.
      // 
      // Only mpirank == 0 works on this

      Real timeOutputSta, timeOutputEnd;
      GetTime( timeOutputSta );

      if(1){
        if( mpirank == 0 ){
          std::ostringstream structStream;
#if ( _DEBUGlevel_ >= 0 )
          statusOFS << std::endl 
            << "Output the structure information" 
            << std::endl;
#endif
          // Domain
          serialize( domain_.length, structStream, NO_MASK );
          serialize( domain_.numGrid, structStream, NO_MASK );
          serialize( domain_.numGridFine, structStream, NO_MASK );
          serialize( domain_.posStart, structStream, NO_MASK );
          serialize( numElem_, structStream, NO_MASK );

          // Atomic information
          serialize( hamDG.AtomList(), structStream, NO_MASK );
          std::string structFileName = "STRUCTURE";

          std::ofstream fout(structFileName.c_str());
          if( !fout.good() ){
            std::ostringstream msg;
            msg 
              << "File " << structFileName.c_str() << " cannot be opened." 
              << std::endl;
            ErrorHandling( msg.str().c_str() );
          }
          fout << structStream.str();
          fout.close();
        }
      }


      for( Int k = 0; k < numElem_[2]; k++ )
        for( Int j = 0; j < numElem_[1]; j++ )
          for( Int i = 0; i < numElem_[0]; i++ ){
            Index3 key( i, j, k );
            if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
              // Output density, and only mpirankRow == 0 does the job of
              // for each element.
              if( esdfParam.isOutputDensity ){
                if( mpirankRow == 0 ){
#if ( _DEBUGlevel_ >= 0 )
                  statusOFS << std::endl 
                    << "Output the electron density on the global grid" 
                    << std::endl;
#endif
                  // Output the wavefunctions on the uniform grid
                  {
                    std::ostringstream rhoStream;      

                    NumTns<std::vector<DblNumVec> >& uniformGridElem =
                      hamDG.UniformGridElemFine();
                    std::vector<DblNumVec>& grid = hamDG.UniformGridElemFine()(i, j, k);
                    for( Int d = 0; d < DIM; d++ ){
                      serialize( grid[d], rhoStream, NO_MASK );
                    }

                    serialize( key, rhoStream, NO_MASK );
                    serialize( hamDG.Density().LocalMap()[key], rhoStream, NO_MASK );

                    SeparateWrite( restartDensityFileName_, rhoStream, mpirankCol );
                  }

                  // Output the wavefunctions on the LGL grid
                  if(0)
                  {
                    std::ostringstream rhoStream;      

                    // Generate the uniform mesh on the extended element.
                    std::vector<DblNumVec>& gridpos = hamDG.LGLGridElem()(i,j,k);
                    for( Int d = 0; d < DIM; d++ ){
                      serialize( gridpos[d], rhoStream, NO_MASK );
                    }
                    serialize( key, rhoStream, NO_MASK );
                    serialize( hamDG.DensityLGL().LocalMap()[key], rhoStream, NO_MASK );
                    SeparateWrite( "DENLGL", rhoStream, mpirankCol );
                  }

                } // if( mpirankRow == 0 )
              }

              // Output potential in extended element, and only mpirankRow
              // == 0 does the job of for each element.
              if( esdfParam.isOutputPotExtElem ) {
                if( mpirankRow == 0 ){
#if ( _DEBUGlevel_ >= 0 )
                  statusOFS 
                    << std::endl 
                    << "Output the total potential and external potential in the extended element."
                    << std::endl;
#endif
                  std::ostringstream potStream;      
                  EigenSolver&  eigSol = distEigSolPtr_->LocalMap()[key];

                  // Generate the uniform mesh on the extended element.
                  //              std::vector<DblNumVec> gridpos;
                  //              UniformMeshFine ( eigSol.FFT().domain, gridpos );
                  //              for( Int d = 0; d < DIM; d++ ){
                  //                serialize( gridpos[d], potStream, NO_MASK );
                  //              }


                  serialize( key, potStream, NO_MASK );
                  serialize( eigSol.Ham().Vtot(), potStream, NO_MASK );
                  serialize( eigSol.Ham().Vext(), potStream, NO_MASK );
                  SeparateWrite( "POTEXT", potStream, mpirankCol );
                } // if( mpirankRow == 0 )
              }

              // Output wavefunction in the extended element.  All processors participate
              if( esdfParam.isOutputWfnExtElem )
              {
#if ( _DEBUGlevel_ >= 0 )
                statusOFS 
                  << std::endl 
                  << "Output the wavefunctions in the extended element."
                  << std::endl;
#endif

                EigenSolver&  eigSol = distEigSolPtr_->LocalMap()[key];
                std::ostringstream wavefunStream;      

                // Generate the uniform mesh on the extended element.
                // NOTE 05/06/2015: THIS IS NOT COMPATIBLE WITH THAT OF THE ALB2DEN!!
                std::vector<DblNumVec> gridpos;
                UniformMesh ( eigSol.FFT().domain, gridpos );
                for( Int d = 0; d < DIM; d++ ){
                  serialize( gridpos[d], wavefunStream, NO_MASK );
                }

                serialize( key, wavefunStream, NO_MASK );
                serialize( eigSol.Psi().Wavefun(), wavefunStream, NO_MASK );
                SeparateWrite( restartWfnFileName_, wavefunStream, mpirank);
              }

              // Output wavefunction in the element on LGL grid. All processors participate.
              if( esdfParam.isOutputALBElemLGL )
              {
#if ( _DEBUGlevel_ >= 0 )
                statusOFS 
                  << std::endl 
                  << "Output the wavefunctions in the element on a LGL grid."
                  << std::endl;
#endif
                // Output the wavefunctions in the extended element.
                std::ostringstream wavefunStream;      

                // Generate the uniform mesh on the extended element.
                std::vector<DblNumVec>& gridpos = hamDG.LGLGridElem()(i,j,k);
                for( Int d = 0; d < DIM; d++ ){
                  serialize( gridpos[d], wavefunStream, NO_MASK );
                }
                serialize( key, wavefunStream, NO_MASK );
                serialize( hamDG.BasisLGL().LocalMap()[key], wavefunStream, NO_MASK );
                serialize( hamDG.LGLWeight3D(), wavefunStream, NO_MASK );
                SeparateWrite( "ALBLGL", wavefunStream, mpirank );
              }

              // Output wavefunction in the element on uniform fine grid.
              // All processors participate
              // NOTE: 
              // Since interpolation needs to be performed, this functionality can be slow.
              if( esdfParam.isOutputALBElemUniform )
              {
#if ( _DEBUGlevel_ >= 0 )
                statusOFS 
                  << std::endl 
                  << "Output the wavefunctions in the element on a fine LGL grid."
                  << std::endl;
#endif
                // Output the wavefunctions in the extended element.
                std::ostringstream wavefunStream;      

                // Generate the uniform mesh on the extended element.
                serialize( key, wavefunStream, NO_MASK );
                DblNumMat& basisLGL = hamDG.BasisLGL().LocalMap()[key];
                DblNumMat basisUniformFine( 
                    hamDG.NumUniformGridElemFine().prod(), 
                    basisLGL.n() );
                SetValue( basisUniformFine, 0.0 );

                for( Int g = 0; g < basisLGL.n(); g++ ){
                  hamDG.InterpLGLToUniform(
                      hamDG.NumLGLGridElem(),
                      hamDG.NumUniformGridElemFine(),
                      basisLGL.VecData(g),
                      basisUniformFine.VecData(g) );
                }
                // Generate the uniform mesh on the extended element.
                // NOTE 05/06/2015: THIS IS NOT COMPATIBLE WITH THAT OF THE ALB2DEN!!
                std::vector<DblNumVec> gridpos;
                EigenSolver&  eigSol = distEigSolPtr_->LocalMap()[key];
                // UniformMeshFine ( eigSol.FFT().domain, gridpos );
                // for( Int d = 0; d < DIM; d++ ){
                //   serialize( gridpos[d], wavefunStream, NO_MASK );
                // }

                serialize( key, wavefunStream, NO_MASK );
                serialize( basisUniformFine, wavefunStream, NO_MASK );
                SeparateWrite( "ALBUNIFORM", wavefunStream, mpirank );
              }

              // Output the eigenvector coefficients and only
              // mpirankRow == 0 does the job of for each element.
              // This option is only valid for diagonalization
              // methods
              if( esdfParam.isOutputEigvecCoef && solutionMethod_ == "diag" ) {
                if( mpirankRow == 0 ){
#if ( _DEBUGlevel_ >= 0 )
                  statusOFS << std::endl 
                    << "Output the eigenvector coefficients after diagonalization."
                    << std::endl;
#endif
                  std::ostringstream eigvecStream;      
                  DblNumMat& eigvecCoef = hamDG.EigvecCoef().LocalMap()[key];


                  serialize( key, eigvecStream, NO_MASK );
                  serialize( eigvecCoef, eigvecStream, NO_MASK );
                  SeparateWrite( "EIGVEC", eigvecStream, mpirankCol );
                } // if( mpirankRow == 0 )
              }

            } // (own this element)
          } // for (i)

      GetTime( timeOutputEnd );
#if ( _DEBUGlevel_ >= 0 )
      statusOFS << std::endl 
        << " Time for outputing data is = " << timeOutputEnd - timeOutputSta
        << " [s]" << std::endl;
#endif


      return ;
      }         // -----  end of method SCFDG::Iterate  -----


      // ~~**~~
      // ---------------------------------------------------------
      // ------------ Routines for Chebyshev Filtering -----------
      // ---------------------------------------------------------

      // Dot product for conforming distributed vectors
      // Requires that a distributed vector (and not a distributed matrix) has been sent
      double 
        SCFDG::scfdg_distvec_dot(DistVec<Index3, DblNumMat, ElemPrtn>  &dist_vec_a,
            DistVec<Index3, DblNumMat, ElemPrtn>  &dist_vec_b)
        {

          double temp_result = 0.0, final_result = 0.0;
          DblNumMat& vec_a= (dist_vec_a.LocalMap().begin())->second;
          DblNumMat& vec_b= (dist_vec_b.LocalMap().begin())->second;

          // Conformity check
          if( (dist_vec_a.LocalMap().size() != 1) ||
              (dist_vec_b.LocalMap().size() != 1) ||
              (vec_a.m() != vec_b.m()) ||
              (vec_a.n() != 1) ||
              (vec_b.n() != 1) )
          {
            statusOFS << std::endl << " Non-conforming vectors in dot product !! Aborting ... " << std::endl;
            exit(1);
          }

          // Local result
          temp_result = blas::Dot(vec_a.m(),vec_a.Data(), 1, vec_b.Data(), 1);
          // Global reduce  across colComm since the partion is by DG elements
          mpi::Allreduce( &temp_result, &final_result, 1, MPI_SUM, domain_.colComm );



          return final_result;
        } // End of routine scfdg_distvec_dot

      // L2 norm : Requires that a distributed vector (and not a distributed matrix) has been sent
      double 
        SCFDG::scfdg_distvec_nrm2(DistVec<Index3, DblNumMat, ElemPrtn>  &dist_vec_a)
        {

          double temp_result = 0.0, final_result = 0.0;
          DblNumMat& vec_a= (dist_vec_a.LocalMap().begin())->second;
          double *ptr = vec_a.Data();

          if((dist_vec_a.LocalMap().size() != 1) ||
              (vec_a.n() != 1))
          {
            statusOFS << std::endl << " Unacceptable vector in norm routine !! Aborting ... " << std::endl;
            exit(1);
          }

          // Local result 
          for(Int iter = 0; iter < vec_a.m(); iter ++)
            temp_result += (ptr[iter] * ptr[iter]);

          // Global reduce  across colComm since the partion is by DG elements
          mpi::Allreduce( &temp_result, &final_result, 1, MPI_SUM, domain_.colComm );
          final_result = sqrt(final_result);

          return  final_result;

        } // End of routine scfdg_distvec_nrm2

      // Computes b = scal_a * a + scal_b * b for conforming distributed vectors / matrices
      // Set scal_a = 0.0 and use any vector / matrix a to obtain blas::scal on b
      // Set scal_b = 1.0 for blas::axpy with b denoting y, i.e., b = scal_a * a + b;
      void 
        SCFDG::scfdg_distmat_update(DistVec<Index3, DblNumMat, ElemPrtn>  &dist_mat_a, 
            double scal_a,
            DistVec<Index3, DblNumMat, ElemPrtn>  &dist_mat_b,
            double scal_b)
        {

          DblNumMat& mat_a= (dist_mat_a.LocalMap().begin())->second;
          DblNumMat& mat_b= (dist_mat_b.LocalMap().begin())->second;  

          double *ptr_a = mat_a.Data();
          double *ptr_b = mat_b.Data();

          // Conformity check
          if( (dist_mat_a.LocalMap().size() != 1) ||
              (dist_mat_b.LocalMap().size() != 1) ||
              (mat_a.m() != mat_b.m()) ||
              (mat_a.n() != mat_b.n()) )
          {
            statusOFS << std::endl << " Non-conforming distributed vectors / matrices in update routine !!" 
              << std::endl << " Aborting ... " << std::endl;
            exit(1);
          }

         for(Int iter = 0; iter < (mat_a.m() * mat_a.n()) ; iter ++)
            ptr_b[iter] = scal_a * ptr_a[iter] + scal_b * ptr_b[iter];
 //          daxpby(mat_a.m() * mat_a.n(), scal_a, ptr_a, scal_b, ptr_b);



        } // End of routine scfdg_distvec_update


      // This routine computes Hmat_times_my_dist_mat = Hmat_times_my_dist_mat + Hmat * my_dist_mat
      void SCFDG::scfdg_hamiltonian_times_distmat(DistVec<Index3, DblNumMat, ElemPrtn>  &my_dist_mat, 
          DistVec<Index3, DblNumMat, ElemPrtn>  &Hmat_times_my_dist_mat)
      {
       
        Real one = 1.0 ;
        Int mpirank, mpisize;
        MPI_Comm_rank( domain_.comm, &mpirank );
        MPI_Comm_size( domain_.comm, &mpisize );

        HamiltonianDG&  hamDG = *hamDGPtr_;
        std::vector<Index3>  getKeys_list;

        // Check that vectors provided only contain one entry in the local map
        // This is a safeguard to ensure that we are really dealing with distributed matrices
        if((my_dist_mat.LocalMap().size() != 1) ||
            (Hmat_times_my_dist_mat.LocalMap().size() != 1) ||
            ((my_dist_mat.LocalMap().begin())->first != (Hmat_times_my_dist_mat.LocalMap().begin())->first))
        {
          statusOFS << std::endl << " Vectors in Hmat * vector_block product not formatted correctly !!"
            << std::endl << " Aborting ... " << std::endl;
          exit(1);
        }


        // Obtain key based on my_dist_mat : This assumes that my_dist_mat is formatted correctly
        // based on processor number, etc.
        Index3 key = (my_dist_mat.LocalMap().begin())->first;

        // Obtain keys of neighbors using the Hamiltonian matrix
        // We only use these keys to minimize communication in GetBegin since other parts of the vector
        // block, even if they are non-zero will get multiplied by the zero block of the Hamiltonian
        // anyway.
        for(typename std::map<ElemMatKey, DblNumMat >::iterator 
            get_neighbors_from_Ham_iterator = hamDG.HMat().LocalMap().begin();
            get_neighbors_from_Ham_iterator != hamDG.HMat().LocalMap().end();
//            get_neighbors_from_Ham_iterator ++)
            ++get_neighbors_from_Ham_iterator)
        {
          Index3 neighbor_key = (get_neighbors_from_Ham_iterator->first).second;

          if(neighbor_key == key)
            continue;
          else
            getKeys_list.push_back(neighbor_key);
        }


        // Do the communication necessary to get the information from
        // procs holding the neighbors
        // Supposedly, independent row communicators (i.e. colComm)
        //  are being used for this
        my_dist_mat.GetBegin( getKeys_list, NO_MASK ); 
        my_dist_mat.GetEnd( NO_MASK );


        // Obtain a reference to the chunk where we want to store
        DblNumMat& mat_Y_local = Hmat_times_my_dist_mat.LocalMap()[key];

        cuDblNumMat cu_mat_Y_local(mat_Y_local.m(), mat_Y_local.n());
        cu_mat_Y_local.CopyFrom(mat_Y_local);

        // Now pluck out relevant chunks of the Hamiltonian and the vector and multiply
        for(typename std::map<Index3, DblNumMat >::iterator 
            mat_X_iterator = my_dist_mat.LocalMap().begin();
            mat_X_iterator != my_dist_mat.LocalMap().end(); ++mat_X_iterator ){

          Index3 iter_key = mat_X_iterator->first;       
          DblNumMat& mat_X_local = mat_X_iterator->second; // Chunk from input block of vectors

          // Create key for looking up Hamiltonian chunk 
          ElemMatKey myelemmatkey = std::make_pair(key, iter_key);

          std::map<ElemMatKey, DblNumMat >::iterator ham_iterator = hamDG.HMat().LocalMap().find(myelemmatkey);

          //statusOFS << std::endl << " Working on key " << key << "   " << iter_key << std::endl;

          // Now do the actual multiplication
          DblNumMat& mat_H_local = ham_iterator->second; // Chunk from Hamiltonian

          Int m = mat_H_local.m(), n = mat_X_local.n(), k = mat_H_local.n();

#ifdef GPU
           cuDblNumMat cu_mat_H_local(mat_H_local.m(), mat_H_local.n());
           cuDblNumMat cu_mat_X_local(mat_X_local.m(), mat_X_local.n());
           cu_mat_H_local.CopyFrom(mat_H_local);
           cu_mat_X_local.CopyFrom(mat_X_local);
           cublas::Gemm( HIPBLAS_OP_N, HIPBLAS_OP_N, m, n, k,
                         &one, cu_mat_H_local.Data(), m,
                         cu_mat_X_local.Data(), k,
                         &one, cu_mat_Y_local.Data(), m);
#else
          blas::Gemm( 'N', 'N', m, n, k,
              1.0, mat_H_local.Data(), m,
              mat_X_local.Data(), k,
              1.0, mat_Y_local.Data(), m);
#endif
        } // End of loop using mat_X_iterator
        cu_mat_Y_local.CopyTo(mat_Y_local);


        // Matrix * vector_block product is ready now ... 
        // Need to clean up extra entries in my_dist_mat
        typename std::map<Index3, DblNumMat >::iterator it;
        for(Int delete_iter = 0; delete_iter <  getKeys_list.size(); delete_iter ++)
        {
          it = my_dist_mat.LocalMap().find(getKeys_list[delete_iter]);
          (my_dist_mat.LocalMap()).erase(it);
        }


      }



      // This routine estimates the spectral bounds using the Lanczos method
      double 
        SCFDG::scfdg_Cheby_Upper_bound_estimator(DblNumVec& ritz_values, 
            int Num_Lanczos_Steps
            )
        {

          Int mpirank, mpisize;
          MPI_Comm_rank( domain_.comm, &mpirank );
          MPI_Comm_size( domain_.comm, &mpisize );


          Real timeSta, timeEnd;
          Real timeIterStart, timeIterEnd;


          HamiltonianDG&  hamDG = *hamDGPtr_;

          // Declare vectors partioned according to DG elements
          // These will be used for Lanczos

          // Input vector
          DistVec<Index3, DblNumMat, ElemPrtn>  dist_vec_v; 
          dist_vec_v.Prtn() = elemPrtn_;
          dist_vec_v.SetComm(domain_.colComm);

          // vector to hold f = H*v
          DistVec<Index3, DblNumMat, ElemPrtn>  dist_vec_f; 
          dist_vec_f.Prtn() = elemPrtn_;
          dist_vec_f.SetComm(domain_.colComm);

          // vector v0
          DistVec<Index3, DblNumMat, ElemPrtn>  dist_vec_v0; 
          dist_vec_v0.Prtn() = elemPrtn_;
          dist_vec_v0.SetComm(domain_.colComm);

          // Step 1 : Generate a random vector v
          // Fill up the vector v using random entries
          // Also, initialize the vectors f and v0 to 0

          // We use this loop for setting up the keys. 
          for( Int k = 0; k < numElem_[2]; k++ )
            for( Int j = 0; j < numElem_[1]; j++ )
              for( Int i = 0; i < numElem_[0]; i++ ){
                Index3 key( i, j, k );

                // If the current processor owns this element
                if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){ 

                  // Associate the current key with a vector that contains the stuff 
                  // that should reside on this process.
                  const std::vector<Int>&  idx = hamDG.ElemBasisIdx()(i, j, k); 

                  dist_vec_v.LocalMap()[key].Resize(idx.size(), 1); // Because of this, the LocalMap is of size 1 now        
                  dist_vec_f.LocalMap()[key].Resize(idx.size(), 1); // Because of this, the LocalMap is of size 1 now
                  dist_vec_v0.LocalMap()[key].Resize(idx.size(), 1); // Because of this, the LocalMap is of size 1 now

                  // Initialize the local maps    
                  // Vector v is initialized randomly
                  UniformRandom(dist_vec_v.LocalMap()[key]);

                  // Vector f and v0 are filled with zeros
                  SetValue(dist_vec_f.LocalMap()[key], 0.0);
                  SetValue(dist_vec_v0.LocalMap()[key], 0.0);

                }
              } // End of vector initializations


          // Normalize the vector v
          double norm_v = scfdg_distvec_nrm2(dist_vec_v);     
          scfdg_distmat_update(dist_vec_v, 0.0, dist_vec_v, (1.0 / norm_v));

          // Step 2a : f = H * v 
          //scfdg_hamiltonian_times_distvec(dist_vec_v,  dist_vec_f); // f has already been set to 0
          scfdg_hamiltonian_times_distmat(dist_vec_v,  dist_vec_f); // f has already been set to 0

          // Step 2b : alpha = f^T * v
          double alpha, beta;
          alpha = scfdg_distvec_dot(dist_vec_f, dist_vec_v);

          statusOFS << " alpha " << alpha  << std::endl;

          // Step 2c : f = f - alpha * v
          scfdg_distmat_update(dist_vec_v, (-alpha), dist_vec_f, 1.0);


          // Step 2d: T(1,1) = alpha
          DblNumMat matT( Num_Lanczos_Steps, Num_Lanczos_Steps );
          SetValue(matT, 0.0);    
          matT(0,0) = alpha;

          // Step 3: Lanczos iteration
          for(Int j = 1; j < Num_Lanczos_Steps; j ++)
          {
            // Step 4 : beta = norm(f)
            beta = scfdg_distvec_nrm2(dist_vec_f);

            // Step 5a : v0 = v
            scfdg_distmat_update(dist_vec_v, 1.0, dist_vec_v0,  0.0);

            // Step 5b : v = f / beta
            scfdg_distmat_update(dist_vec_f, (1.0 / beta), dist_vec_v, 0.0);

            // Step 6a : f = H * v
            SetValue((dist_vec_f.LocalMap().begin())->second, 0.0);// Set f to zero first !
            scfdg_hamiltonian_times_distmat(dist_vec_v,  dist_vec_f);

            // Step 6b : f = f - beta * v0
            scfdg_distmat_update(dist_vec_v0, (-beta) , dist_vec_f,  1.0);

            // Step 7a : alpha = f^T * v
            alpha = scfdg_distvec_dot(dist_vec_f, dist_vec_v);

            // Step 7b : f = f - alpha * v
            scfdg_distmat_update(dist_vec_v, (-alpha) , dist_vec_f,  1.0);

            // Step 8 : Fill up the matrix
            matT(j, j - 1) = beta;
            matT(j - 1, j) = beta;
            matT(j, j) = alpha;
          } // End of loop over Lanczos steps 

          // Step 9 : Compute the Lanczos-Ritz values
          ritz_values.Resize(Num_Lanczos_Steps);
          SetValue( ritz_values, 0.0 );

          // Solve the eigenvalue problem for the Ritz values
          lapack::Syevd( 'N', 'U', Num_Lanczos_Steps, matT.Data(), Num_Lanczos_Steps, ritz_values.Data() );

          // Step 10 : Compute the upper bound on each process
          double b_up = ritz_values(Num_Lanczos_Steps - 1) + scfdg_distvec_nrm2(dist_vec_f);

//          statusOFS << std::endl << " Ritz values in estimator here : " << ritz_values ;
//          statusOFS << std::endl << " Upper bound of spectrum = " << b_up << std::endl;

          // Need to synchronize the Ritz values and the upper bound across the processes
          MPI_Bcast(&b_up, 1, MPI_DOUBLE, 0, domain_.comm);
          MPI_Bcast(ritz_values.Data(), Num_Lanczos_Steps, MPI_DOUBLE, 0, domain_.comm);



          return b_up;

        } // End of scfdg_Cheby_Upper_bound_estimator

      // Apply the scaled Chebyshev Filter on the Eigenvectors
      // Use a distributor to work on selected bands based on
      // number of processors per element (i.e., no. of rows in process grid).
      void 
        SCFDG::scfdg_Chebyshev_filter_scaled(int m, 
            double a, 
            double b, 
            double a_L)
        {
          Int mpirank, mpisize;
          MPI_Comm_rank( domain_.comm, &mpirank );
          MPI_Comm_size( domain_.comm, &mpisize );
          Int mpirankRow;  MPI_Comm_rank(domain_.rowComm, &mpirankRow);
          Int mpisizeRow;  MPI_Comm_size(domain_.rowComm, &mpisizeRow);
          Int mpirankCol;  MPI_Comm_rank(domain_.colComm, &mpirankCol);
          Int mpisizeCol;  MPI_Comm_size(domain_.colComm, &mpisizeCol);

          Real timeSta, timeEnd;
          Real extra_timeSta, extra_timeEnd;
          Real filter_total_time = 0.0;

          HamiltonianDG&  hamDG = *hamDGPtr_;

          // We need to distribute bands according to rowComm since colComm is
          // used for element-wise partition.
          if(mpisizeRow > hamDG.NumStateTotal())
          {
            statusOFS << std::endl << " Number of processors per element exceeds number of bands !!"
              << std::endl << " Cannot continue with band-parallelization. "
              << std::endl << " Aborting ... " << std::endl;
            exit(1);

          }
          simple_distributor band_distributor(hamDG.NumStateTotal(), mpisizeRow, mpirankRow);


          // Create distributed matrices pluck_X, pluck_Y, pluck_Yt for filtering 
          const Index3 key = (hamDG.EigvecCoef().LocalMap().begin())->first; // Will use same key as eigenvectors
          DblNumMat &eigvecs_local = (hamDG.EigvecCoef().LocalMap().begin())->second;

          Int local_width = band_distributor.current_proc_size;
          Int local_height = eigvecs_local.m();
          Int local_pluck_sz = local_height * local_width;


          DistVec<Index3, DblNumMat, ElemPrtn>  pluck_X; 
          pluck_X.Prtn() = elemPrtn_;
          pluck_X.SetComm(domain_.colComm);
          pluck_X.LocalMap()[key].Resize(local_height, local_width);

          DistVec<Index3, DblNumMat, ElemPrtn>  pluck_Y; 
          pluck_Y.Prtn() = elemPrtn_;
          pluck_Y.SetComm(domain_.colComm);
          pluck_Y.LocalMap()[key].Resize(local_height, local_width);

          DistVec<Index3, DblNumMat, ElemPrtn>  pluck_Yt; 
          pluck_Yt.Prtn() = elemPrtn_;
          pluck_Yt.SetComm(domain_.colComm);
          pluck_Yt.LocalMap()[key].Resize(local_height, local_width);


          // Initialize the distributed matrices
          blas::Copy(local_pluck_sz, 
              eigvecs_local.Data() + local_height * band_distributor.current_proc_start, 
              1,
              pluck_X.LocalMap()[key].Data(),
              1);

          SetValue(pluck_Y.LocalMap()[key], 0.0);
          SetValue(pluck_Yt.LocalMap()[key], 0.0);

          // Filtering scalars
          double e = (b - a) / 2.0;
          double c = (a + b) / 2.0;
          double sigma = e / (c - a_L);
          double tau = 2.0 / sigma;
          double sigma_new;

          // Begin the filtering process
          // Y = (H * X - c * X) * (sigma / e)
          // pluck_Y has been initialized to 0 already

          statusOFS << std::endl << " Chebyshev filtering : Process " << mpirank << " working on " 
            << local_width << " of " << eigvecs_local.n() << " bands.";

          statusOFS << std::endl << " Chebyshev filtering : Lower bound = " << a
            << std::endl << "                     : Upper bound = " << b
            << std::endl << "                     : a_L = " << a_L;

          //statusOFS << std::endl << " Chebyshev filtering step 1 of " << m << " ... ";
          GetTime( extra_timeSta );

          scfdg_hamiltonian_times_distmat(pluck_X, pluck_Y); // Y = H * X
          scfdg_distmat_update(pluck_X, (-c) , pluck_Y,  1.0); // Y = -c * X + 1.0 * Y
          scfdg_distmat_update(pluck_Y, 0.0 , pluck_Y,  (sigma / e)); // Y = 0.0 * Y + (sigma / e) * Y

          GetTime( extra_timeEnd );
          //statusOFS << " Done. ( " << (extra_timeEnd - extra_timeSta ) << " s.)";
          filter_total_time += (extra_timeEnd - extra_timeSta );
          Real xmqin_time1 = 0.0;
          Real xmqin_time2 = 0.0 ; 
          Real xmqin_time3 = 0.0  ;
          Real xmqin_time4 = 0.0  ;
          Real  filter_time1;
          Real  filter_time2;
          for(Int filter_iter = 2; filter_iter < m; filter_iter ++)
          {   
            //statusOFS << std::endl << " Chebyshev filtering step " << filter_iter << " of " << m << " ... ";
            GetTime( extra_timeSta );

            sigma_new = 1.0 / (tau - sigma);

            //Compute Yt = (H * Y - c * Y) * (2 * sigma_new / e) - (sigma * sigma_new) * X
            // Set Yt to 0
            SetValue(pluck_Yt.LocalMap()[key], 0.0);

            GetTime( filter_time1);  
            scfdg_hamiltonian_times_distmat(pluck_Y, pluck_Yt); // Yt = H * Y
            GetTime( filter_time2);
            xmqin_time1 = xmqin_time1 + filter_time2-filter_time1 ;

            GetTime( filter_time1);  
            scfdg_distmat_update(pluck_Y, (-c) , pluck_Yt,  1.0); // Yt = -c * Y + 1.0 * Yt
            GetTime( filter_time2); 
            xmqin_time2 = xmqin_time2 + filter_time2-filter_time1 ;
 
            GetTime( filter_time1);
            scfdg_distmat_update(pluck_Yt, 0.0 , pluck_Yt,  (2.0 * sigma_new / e)); // Yt = 0.0 * Yt + (2.0 * sigma_new / e) * Yt

            scfdg_distmat_update(pluck_X, (-sigma * sigma_new) , pluck_Yt,  1.0 ); // Yt = (-sigma * sigma_new) * X + 1.0 * Yt
            GetTime( filter_time2);  
            xmqin_time3 = xmqin_time3 + filter_time2-filter_time1;

            GetTime( filter_time1); 
            // Update / Re-assign : X = Y, Y = Yt, sigma = sigma_new
            scfdg_distmat_update(pluck_Y, 1.0 , pluck_X,  0.0 ); // X = 0.0 * X + 1.0 * Y
            scfdg_distmat_update(pluck_Yt, 1.0 , pluck_Y,  0.0 ); // Y = 0.0 * Y + 1.0 * Yt
            GetTime( filter_time2);
            xmqin_time4 = xmqin_time4 + filter_time2-filter_time1;

            sigma = sigma_new;   

            GetTime( extra_timeEnd );
            //statusOFS << " Done. ( " << (extra_timeEnd - extra_timeSta ) << " s.)";
            filter_total_time += (extra_timeEnd - extra_timeSta );

          }

          statusOFS << std::endl <<  " filter1 time for 1 = "  << xmqin_time1 << " s.";
          statusOFS << std::endl <<  " filter1 time for 2 = "  << xmqin_time2 << " s.";
          statusOFS << std::endl <<  " filter1 time for 3 = "  << xmqin_time3 << " s.";
          statusOFS << std::endl <<  " filter1 time for 4 = "  << xmqin_time4 << " s.";

          statusOFS << std::endl <<  " Total filtering time for " 
            << m << " filter steps = " << filter_total_time << " s."
            << std::endl <<  " Average per filter step = " << ( filter_total_time / double(m) ) << " s.";


          // pluck_Y contains the results of filtering.
          // Copy back pluck_Y to the eigenvector
          // SetValue(eigvecs_local, 0.0); // All entries set to zero for All-Reduce
          GetTime( extra_timeSta );

          DblNumMat temp_buffer;
          temp_buffer.Resize(eigvecs_local.m(), eigvecs_local.n());
          SetValue(temp_buffer, 0.0);

          blas::Copy(local_pluck_sz, 
              pluck_Y.LocalMap()[key].Data(),
              1,
              temp_buffer.Data() + local_height * band_distributor.current_proc_start,
              1);

          MPI_Allreduce(temp_buffer.Data(),
              eigvecs_local.Data(),
              (eigvecs_local.m() * eigvecs_local.n()),
              MPI_DOUBLE,
              MPI_SUM,
              domain_.rowComm);

          GetTime( extra_timeEnd );
          statusOFS << std::endl << " Eigenvector block rebuild time = " 
            << (extra_timeEnd - extra_timeSta ) << " s.";





        } // End of scfdg_Chebyshev_filter


      // Apply the Hamiltonian to the Eigenvectors and place result in result_mat
      // This routine is actually not used by the Chebyshev filter routine since all 
      // the filtering can be done block-wise to reduce communication 
      // among processor rows (i.e., rowComm), i.e., we do the full filter and 
      // communicate only in the end.
      // This routine is used for the Raleigh-Ritz step.
      // Uses a distributor to work on selected bands based on
      // number of processors per element (i.e., no. of rows in process grid).  
      void 
        SCFDG::scfdg_Hamiltonian_times_eigenvectors(DistVec<Index3, DblNumMat, ElemPrtn>  &result_mat)
        {
          Int mpirank, mpisize;
          MPI_Comm_rank( domain_.comm, &mpirank );
          MPI_Comm_size( domain_.comm, &mpisize );
          Int mpirankRow;  MPI_Comm_rank(domain_.rowComm, &mpirankRow);
          Int mpisizeRow;  MPI_Comm_size(domain_.rowComm, &mpisizeRow);
          Int mpirankCol;  MPI_Comm_rank(domain_.colComm, &mpirankCol);
          Int mpisizeCol;  MPI_Comm_size(domain_.colComm, &mpisizeCol);

          Real timeSta, timeEnd;
          Real extra_timeSta, extra_timeEnd;
          Real filter_total_time = 0.0;

          HamiltonianDG&  hamDG = *hamDGPtr_;

          // We need to distribute bands according to rowComm since colComm is
          // used for element-wise partition.
          if(mpisizeRow > hamDG.NumStateTotal())
          {
            statusOFS << std::endl << " Number of processors per element exceeds number of bands !!"
              << std::endl << " Cannot continue with band-parallelization. "
              << std::endl << " Aborting ... " << std::endl;
            exit(1);

          }
          simple_distributor band_distributor(hamDG.NumStateTotal(), mpisizeRow, mpirankRow);

          // Create distributed matrices pluck_X, pluck_Y, pluck_Yt for filtering 
          const Index3 key = (hamDG.EigvecCoef().LocalMap().begin())->first; // Will use same key as eigenvectors
          DblNumMat &eigvecs_local = (hamDG.EigvecCoef().LocalMap().begin())->second;

          Int local_width = band_distributor.current_proc_size;
          Int local_height = eigvecs_local.m();
          Int local_pluck_sz = local_height * local_width;


          DistVec<Index3, DblNumMat, ElemPrtn>  pluck_X; 
          pluck_X.Prtn() = elemPrtn_;
          pluck_X.SetComm(domain_.colComm);
          pluck_X.LocalMap()[key].Resize(local_height, local_width);

          DistVec<Index3, DblNumMat, ElemPrtn>  pluck_Y; 
          pluck_Y.Prtn() = elemPrtn_;
          pluck_Y.SetComm(domain_.colComm);
          pluck_Y.LocalMap()[key].Resize(local_height, local_width);

          // Initialize the distributed matrices
          blas::Copy(local_pluck_sz, 
              eigvecs_local.Data() + local_height * band_distributor.current_proc_start, 
              1,
              pluck_X.LocalMap()[key].Data(),
              1);

          SetValue(pluck_Y.LocalMap()[key], 0.0); // pluck_Y is initialized to 0 


          GetTime( extra_timeSta );

          scfdg_hamiltonian_times_distmat(pluck_X, pluck_Y); // Y = H * X

          GetTime( extra_timeEnd );

          statusOFS << std::endl << " Hamiltonian times eigenvectors calculation time = " 
            << (extra_timeEnd - extra_timeSta ) << " s.";

          // Copy pluck_Y to result_mat after preparing it
          GetTime( extra_timeSta );

          DblNumMat temp_buffer;
          temp_buffer.Resize(eigvecs_local.m(), eigvecs_local.n());
          SetValue(temp_buffer, 0.0);

          blas::Copy(local_pluck_sz, 
              pluck_Y.LocalMap()[key].Data(),
              1,
              temp_buffer.Data() + local_height * band_distributor.current_proc_start,
              1);

          // Empty out the results and prepare the distributed matrix
          result_mat.LocalMap().clear();
          result_mat.Prtn() = elemPrtn_;
          result_mat.SetComm(domain_.colComm);
          result_mat.LocalMap()[key].Resize(eigvecs_local.m(), eigvecs_local.n());

          MPI_Allreduce(temp_buffer.Data(),
              result_mat.LocalMap()[key].Data(),
              (eigvecs_local.m() * eigvecs_local.n()),
              MPI_DOUBLE,
              MPI_SUM,
              domain_.rowComm);

          GetTime( extra_timeEnd );
          statusOFS << std::endl << " Eigenvector block rebuild time = " 
            << (extra_timeEnd - extra_timeSta ) << " s.";


        } // End of scfdg_Hamiltonian_times_eigenvectors


      //   // Given a block of eigenvectors (size:  hamDG.NumBasisTotal() * hamDG.NumStateTotal()),
      //   // convert this to ScaLAPACK format for subsequent use with ScaLAPACK routines
      //   // Should only be called by processors for which context > 0
      //   // This is the older version which has a higher communcation load.
      //   void 
      //   SCFDG::scfdg_Cheby_convert_eigvec_distmat_to_ScaLAPACK_old(DistVec<Index3, DblNumMat, ElemPrtn>  &my_dist_vec, 
      //                              std::vector<int> &my_cheby_scala_info,
      //                              dgdft::scalapack::Descriptor &my_scala_descriptor,
      //                              dgdft::scalapack::ScaLAPACKMatrix<Real>  &my_scala_vec)
      //   {
      //    
      //     HamiltonianDG&  hamDG = *hamDGPtr_;
      //     
      //     // Load up the important ScaLAPACK info
      //     int cheby_scala_num_rows = my_cheby_scala_info[0];
      //     int cheby_scala_num_cols = my_cheby_scala_info[1];
      //     int my_cheby_scala_proc_row = my_cheby_scala_info[2];
      //     int my_cheby_scala_proc_col = my_cheby_scala_info[3];
      // 
      //     // Set the descriptor for the ScaLAPACK matrix
      //     my_scala_vec.SetDescriptor( my_scala_descriptor );
      //     
      //     // Get the original key for the distributed vector
      //     Index3 my_original_key = (my_dist_vec.LocalMap().begin())->first;
      //         
      //     // Form the list of unique keys that we will be requiring
      //     // Use the usual map trick for this
      //     std::map<Index3, int> unique_keys_list;
      //         
      //     // Use the row index for figuring out the key         
      //     for(int iter = 0; iter < hamDG.ElemBasisInvIdx().size(); iter ++)
      //       {
      //     int pr = 0 + int(iter / scaBlockSize_) % cheby_scala_num_rows;
      //     if(pr == my_cheby_scala_proc_row)
      //       unique_keys_list[hamDG.ElemBasisInvIdx()[iter]] = 0;
      //           
      //       }
      //         
      //     
      //     std::vector<Index3>  getKeys_list;
      //         
      //     // Form the list for Get-Begin and Get-End
      //     for(typename std::map<Index3, int >::iterator 
      //       it = unique_keys_list.begin(); 
      //     it != unique_keys_list.end(); 
      //     it ++)
      //       { 
      //     getKeys_list.push_back(it->first);
      //       }    
      //         
      //     // Communication
      //     my_dist_vec.GetBegin( getKeys_list, NO_MASK ); 
      //     my_dist_vec.GetEnd( NO_MASK ); 
      //         
      //         
      //     // Get the offset value for each of the matrices pointed to by the keys
      //     std::map<Index3, int> offset_map;
      //     for(typename std::map<Index3, DblNumMat >::iterator 
      //       test_iterator = my_dist_vec.LocalMap().begin();
      //         test_iterator != my_dist_vec.LocalMap().end();
      //     test_iterator ++)
      //       {          
      //     Index3 key = test_iterator->first;
      //     const std::vector<Int>&  my_idx = hamDG.ElemBasisIdx()(key[0],key[1],key[2]); 
      //     offset_map[key] = my_idx[0];
      //       }
      //      
      //     // All the data is now available: simply use this to fill up the local
      //     // part of the ScaLAPACK matrix. We do this by looping over global indices
      //     // and computing local indices from the globals      
      //             
      //     double *local_scala_mat_ptr = my_scala_vec.Data();
      //     int local_scala_mat_height = my_scala_vec.LocalHeight();
      //         
      //     for(int global_col_iter = 0; global_col_iter < hamDG.NumStateTotal(); global_col_iter ++)
      //       {
      //     int m = int(global_col_iter / (cheby_scala_num_cols * scaBlockSize_));
      //     int y = global_col_iter  % scaBlockSize_;
      //     int pc = int(global_col_iter / scaBlockSize_) % cheby_scala_num_cols;
      //         
      //     int local_col_iter = m * scaBlockSize_ + y;
      //           
      //     for(int global_row_iter = 0; global_row_iter <  hamDG.NumBasisTotal(); global_row_iter ++)
      //       {
      //         int l = int(global_row_iter / (cheby_scala_num_rows * scaBlockSize_));
      //         int x = global_row_iter % scaBlockSize_;
      //         int pr = int(global_row_iter / scaBlockSize_) % cheby_scala_num_rows;
      //                         
      //         int local_row_iter = l * scaBlockSize_ + x;
      //         
      //         // Check if this entry resides on the current process
      //         if((pr == my_cheby_scala_proc_row) && (pc == my_cheby_scala_proc_col))
      //           {  
      //           
      //         // Figure out where to read entry from
      //         Index3 key = hamDG.ElemBasisInvIdx()[global_row_iter];
      //         DblNumMat &mat_chunk = my_dist_vec.LocalMap()[key]; 
      //           
      //         // Assignment to local part of ScaLAPACK matrix
      //         local_scala_mat_ptr[local_col_iter * local_scala_mat_height + local_row_iter] 
      //           = mat_chunk(global_row_iter - offset_map[key], global_col_iter);          
      //           }
      //       }
      //       }
      //         
      //         
      // 
      //     // Clean up extra entries from Get-Begin / Get-End
      //     typename std::map<Index3, DblNumMat >::iterator delete_it;
      //     for(Int delete_iter = 0; delete_iter <  getKeys_list.size(); delete_iter ++)
      //       {
      //     if(getKeys_list[delete_iter] != my_original_key) // Be careful about original key
      //       {  
      //         delete_it = my_dist_vec.LocalMap().find(getKeys_list[delete_iter]);
      //         (my_dist_vec.LocalMap()).erase(delete_it);
      //       }
      //       }
      //       
      //       
      //       
      //   
      //     
      //   } // End of scfdg_Cheby_convert_eigvec_distmat_to_ScaLAPACK_old
      //   


      // Given a block of eigenvectors (size:  hamDG.NumBasisTotal() * hamDG.NumStateTotal()),
      // convert this to ScaLAPACK format for subsequent use with ScaLAPACK routines
      // Should only be called by processors for which context > 0
      // This routine is largely based on the Hamiltonian to ScaLAPACK conversion routine which is similar
      void 
        SCFDG::scfdg_Cheby_convert_eigvec_distmat_to_ScaLAPACK(DistVec<Index3, DblNumMat, ElemPrtn>  &my_dist_mat, 
            MPI_Comm comm,
            dgdft::scalapack::Descriptor &my_scala_descriptor,
            dgdft::scalapack::ScaLAPACKMatrix<Real>  &my_scala_mat)
        {

          HamiltonianDG&  hamDG = *hamDGPtr_;

          // Load up the important ScaLAPACK info
          int nprow = my_scala_descriptor.NpRow();
          int npcol = my_scala_descriptor.NpCol();
          int myprow = my_scala_descriptor.MypRow();
          int mypcol = my_scala_descriptor.MypCol();

          // Set the descriptor for the ScaLAPACK matrix
          my_scala_mat.SetDescriptor( my_scala_descriptor );

          // Save the original key for the distributed vector
          const Index3 my_original_key = (my_dist_mat.LocalMap().begin())->first;

          // Get some basic information set up
          Int mpirank, mpisize;
          MPI_Comm_rank( comm, &mpirank );
          MPI_Comm_size( comm, &mpisize );

          Int MB = my_scala_mat.MB(); 
          Int NB = my_scala_mat.NB();


          if( MB != NB )
          {
            ErrorHandling("MB must be equal to NB.");
          }

          // ScaLAPACK block information
          Int numRowBlock = my_scala_mat.NumRowBlocks();
          Int numColBlock = my_scala_mat.NumColBlocks();

          // Get the processor map
          IntNumMat  procGrid( nprow, npcol );
          SetValue( procGrid, 0 );
          {
            IntNumMat  procTmp( nprow, npcol );
            SetValue( procTmp, 0 );
            procTmp( myprow, mypcol ) = mpirank;
            mpi::Allreduce( procTmp.Data(), procGrid.Data(), nprow * npcol,
                MPI_SUM, comm );
          }


          // ScaLAPACK block partition 
          BlockMatPrtn  blockPrtn;

          // Fill up the owner information
          IntNumMat&    blockOwner = blockPrtn.ownerInfo;
          blockOwner.Resize( numRowBlock, numColBlock );   
          for( Int jb = 0; jb < numColBlock; jb++ ){
            for( Int ib = 0; ib < numRowBlock; ib++ ){
              blockOwner( ib, jb ) = procGrid( ib % nprow, jb % npcol );
            }
          }

          // Distributed matrix in ScaLAPACK format
          DistVec<Index2, DblNumMat, BlockMatPrtn> distScaMat;
          distScaMat.Prtn() = blockPrtn;
          distScaMat.SetComm(comm);


          // Zero out and initialize
          DblNumMat empty_mat( MB, MB ); // As MB = NB
          SetValue( empty_mat, 0.0 );
          // We need this loop here since we are initializing the distributed 
          // ScaLAPACK matrix. Subsequently, the LocalMap().begin()->first technique can be used
          for( Int jb = 0; jb < numColBlock; jb++ )
            for( Int ib = 0; ib < numRowBlock; ib++ )
            {
              Index2 key( ib, jb );
              if( distScaMat.Prtn().Owner( key ) == mpirank )
              {
                distScaMat.LocalMap()[key] = empty_mat;
              }
            }



          // Copy data from DG distributed matrix to ScaLAPACK distributed matrix
          DblNumMat& localMat = (my_dist_mat.LocalMap().begin())->second;
          const std::vector<Int>& my_idx = hamDG.ElemBasisIdx()( my_original_key(0), my_original_key(1), my_original_key(2) );

          {
            Int ib, jb, io, jo;
            for( Int b = 0; b < localMat.n(); b++ )
            {
              for( Int a = 0; a < localMat.m(); a++ )
              {
                ib = my_idx[a] / MB;
                io = my_idx[a] % MB;

                jb = b / MB;            
                jo = b % MB;

                typename std::map<Index2, DblNumMat >::iterator 
                  ni = distScaMat.LocalMap().find( Index2(ib, jb) );
                if( ni == distScaMat.LocalMap().end() )
                {
                  distScaMat.LocalMap()[Index2(ib, jb)] = empty_mat;
                  ni = distScaMat.LocalMap().find( Index2(ib, jb) );
                }

                DblNumMat&  localScaMat = ni->second;
                localScaMat(io, jo) += localMat(a, b);
              } // for (a)
            } // for (b)

          }     

          // Communication step
          {
            // Prepare
            std::vector<Index2>  keyIdx;
            for( typename std::map<Index2, DblNumMat >::iterator 
                mi  = distScaMat.LocalMap().begin();
                mi != distScaMat.LocalMap().end(); ++mi )
            {
              Index2 key = mi->first;

              // Include all keys which do not reside on current processor
              if( distScaMat.Prtn().Owner( key ) != mpirank )
              {
                keyIdx.push_back( key );
              }
            } // for (mi)

            // Communication
            distScaMat.PutBegin( keyIdx, NO_MASK );
            distScaMat.PutEnd( NO_MASK, PutMode::COMBINE );

            // Clean up
            std::vector<Index2>  eraseKey;
            for( typename std::map<Index2, DblNumMat >::iterator 
                mi  = distScaMat.LocalMap().begin();
                mi != distScaMat.LocalMap().end(); ++mi)
            {
              Index2 key = mi->first;
              if( distScaMat.Prtn().Owner( key ) != mpirank )
              {
                eraseKey.push_back( key );
              }
            } // for (mi)

            for( std::vector<Index2>::iterator vi = eraseKey.begin();
                vi != eraseKey.end(); ++vi )
            {
              distScaMat.LocalMap().erase( *vi );
            }    


          } // End of communication step


          // Final step: Copy from distributed ScaLAPACK matrix to local part of actual
          // ScaLAPACK matrix
          {
            for( typename std::map<Index2, DblNumMat>::iterator 
                mi  = distScaMat.LocalMap().begin();
                mi != distScaMat.LocalMap().end(); ++mi )
            {
              Index2 key = mi->first;
              if( distScaMat.Prtn().Owner( key ) == mpirank )
              {
                Int ib = key(0), jb = key(1);
                Int offset = ( jb / npcol ) * MB * my_scala_mat.LocalLDim() + 
                  ( ib / nprow ) * MB;
                lapack::Lacpy( 'A', MB, MB, mi->second.Data(),
                    MB, my_scala_mat.Data() + offset, my_scala_mat.LocalLDim() );
              } // own this block
            } // for (mi)

          }


        } // End of scfdg_Cheby_convert_eigvec_distmat_to_ScaLAPACK




      void 
        SCFDG::scfdg_FirstChebyStep(Int MaxIter,
            Int filter_order )
        {
          Int mpirank, mpisize;
          MPI_Comm_rank( domain_.comm, &mpirank );
          MPI_Comm_size( domain_.comm, &mpisize );
          Int mpirankRow;  MPI_Comm_rank(domain_.rowComm, &mpirankRow);
          Int mpisizeRow;  MPI_Comm_size(domain_.rowComm, &mpisizeRow);
          Int mpirankCol;  MPI_Comm_rank(domain_.colComm, &mpirankCol);
          Int mpisizeCol;  MPI_Comm_size(domain_.colComm, &mpisizeCol);

          Real timeSta, timeEnd;
          Real extra_timeSta, extra_timeEnd;
          Real cheby_timeSta, cheby_timeEnd;

          HamiltonianDG&  hamDG = *hamDGPtr_;


          // Step 1: Obtain the upper bound and the Ritz values (for the lower bound)
          // using the Lanczos estimator

          DblNumVec Lanczos_Ritz_values;


          statusOFS << std::endl << " Estimating the spectral bounds ... "; 
          GetTime( extra_timeSta );
          const int Num_Lanczos_Steps = 6;
          double b_up = scfdg_Cheby_Upper_bound_estimator(Lanczos_Ritz_values, Num_Lanczos_Steps);  
          GetTime( extra_timeEnd );
          statusOFS << " Done. ( " << (extra_timeEnd - extra_timeSta ) << " s.)";

          double a_L = Lanczos_Ritz_values(0);
          double beta = 0.5; // 0.5 <= beta < 1.0
          double b_low = beta * Lanczos_Ritz_values(0) + (1.0 - beta) * Lanczos_Ritz_values(Num_Lanczos_Steps - 1);
           //b_low = 0.0; // This is probably better than the above estimate based on Lanczos

//            statusOFS << std::endl << "a_L  = " << a_L << "b_low = " << b_low <<  std::endl ;

          // Step 2: Initialize the eigenvectors to random numbers
          statusOFS << std::endl << " Initializing eigenvectors randomly on first SCF step ... "; 

          hamDG.EigvecCoef().Prtn() = elemPrtn_;
          hamDG.EigvecCoef().SetComm(domain_.colComm);

          GetTime( extra_timeSta );

          // This is one of the very few places where we use this loop 
          // for setting up the keys. In other cases, we can simply access
          // the first and second elements of the LocalMap of the distributed vector
          // once it has been set up.
          for( Int k = 0; k < numElem_[2]; k++ )
            for( Int j = 0; j < numElem_[1]; j++ )
              for( Int i = 0; i < numElem_[0]; i++ ){
                Index3 key( i, j, k );

                // If the current processor owns this element
                if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){ 

                  // Save the key 
                  my_cheby_eig_vec_key_ = key;

                  // Associate the current key with a vector that contains the stuff 
                  // that should reside on this process.
                  const std::vector<Int>&  idx = hamDG.ElemBasisIdx()(i, j, k); 
                  hamDG.EigvecCoef().LocalMap()[key].Resize(idx.size(), hamDG.NumStateTotal()); 

                  DblNumMat &ref_mat =  hamDG.EigvecCoef().LocalMap()[key];

                  // Only the first processor on every column does this
                  if(mpirankRow == 0)          
                    UniformRandom(ref_mat);
                  // Now broadcast this
                  MPI_Bcast(ref_mat.Data(), (ref_mat.m() * ref_mat.n()), MPI_DOUBLE, 0, domain_.rowComm);

                }
              }// End of eigenvector initialization
          GetTime( extra_timeEnd );
          statusOFS << " Done. ( " << (extra_timeEnd - extra_timeSta ) << " s.)";


          // Step 3: Main loop
          const Int Iter_Max = MaxIter;
          const Int Filter_Order = filter_order ;

          DblNumVec eig_vals_Raleigh_Ritz;
          Real one = 1.0;
          Real zero = 0.0;

          for(Int i = 1; i <= Iter_Max; i ++)
          {
            GetTime( cheby_timeSta );
            statusOFS << std::endl << std::endl << " ------------------------------- ";
            statusOFS << std::endl << " First CheFSI step for DGDFT cycle " << i << " of " << Iter_Max << " . ";
            // Filter the eigenvectors
            statusOFS << std::endl << std::endl << " Filtering the eigenvectors ... (Filter order = " << Filter_Order << ")";
            GetTime( timeSta );
            scfdg_Chebyshev_filter_scaled(Filter_Order, b_low, b_up, a_L);
            GetTime( timeEnd );
            statusOFS << std::endl << " Filtering completed. ( " << (timeEnd - timeSta ) << " s.)";

            // Subspace projected problems: Orthonormalization, Raleigh-Ritz and subspace rotation steps
            // This can be done serially or using ScaLAPACK    
            if(SCFDG_Cheby_use_ScaLAPACK_ == 0)
            {
              // Do the subspace problem serially 
              statusOFS << std::endl << std::endl << " Solving subspace problems serially ...";

              // Orthonormalize using Cholesky factorization  
              statusOFS << std::endl << " Orthonormalizing filtered vectors ... ";
              GetTime( timeSta );



              DblNumMat &local_eigvec_mat = (hamDG.EigvecCoef().LocalMap().begin())->second;
              DblNumMat square_mat;
              DblNumMat temp_square_mat;

              Int width = local_eigvec_mat.n();
              Int height_local = local_eigvec_mat.m();

              square_mat.Resize(width, width);
              temp_square_mat.Resize(width, width);

              SetValue(temp_square_mat, 0.0);

              // Compute square_mat = X^T * X for Cholesky    
#ifdef GPU //--------by lijl 20220526
              Real sta_jl, end_jl;
              cuDblNumMat cu_local_eigvec_mat(height_local, width);
              cu_local_eigvec_mat.CopyFrom(local_eigvec_mat);
              cuDblNumMat cu_temp_square_mat(width, width);
              cublas::Gemm( HIPBLAS_OP_T, HIPBLAS_OP_N, width, width, height_local,
                  &one, cu_local_eigvec_mat.Data(), height_local,
                  cu_local_eigvec_mat.Data(), height_local,
                  &zero, cu_temp_square_mat.Data(), width );
              cu_temp_square_mat.CopyTo(temp_square_mat);
#else
              blas::Gemm( 'T', 'N', width, width, height_local, 
                  1.0, local_eigvec_mat.Data(), height_local,
                  local_eigvec_mat.Data(), height_local, 
                  0.0, temp_square_mat.Data(), width );
#endif
              SetValue( square_mat, 0.0 );
              MPI_Allreduce( temp_square_mat.Data(), square_mat.Data(), width*width, MPI_DOUBLE, MPI_SUM, domain_.colComm );

              // In the following, reduction happens on colComm but the result is broadcast to everyone
              // This can probably be band-parallelized

              // Make the Cholesky factorization call on proc 0
              if ( mpirank == 0) {
#if 0 //def GPU
              cuDblNumMat cu_square_mat(width, width);
              cu_square_mat.CopyFrom(square_mat);
              //MAGMA::Potrf( 'U', width, cu_square_mat.Data(), width );
              cusolver::Potrf( 'U', width, cu_square_mat.Data(), width );
              cu_square_mat.CopyTo(square_mat);
#else

//             for(Int k=0; k<width*width; k++){
//               statusOFS << k << " square "<< square_mat.data_[k] <<  std::endl;
//              }

               lapack::Potrf( 'U', width, square_mat.Data(), width );
#endif
              }
              // Send the Cholesky factor to every process
              MPI_Bcast(square_mat.Data(), width*width, MPI_DOUBLE, 0, domain_.comm);

              // Do a solve with the Cholesky factor : Band parallelization ??
              // X = X * U^{-1} is orthogonal, where U is the Cholesky factor
#ifdef GPU
              cuDblNumMat cu_square_mat(width, width);              
              cu_square_mat.CopyFrom(square_mat);
              cublas::Trsm( HIPBLAS_SIDE_RIGHT, HIPBLAS_FILL_MODE_UPPER, HIPBLAS_OP_N, HIPBLAS_DIAG_NON_UNIT, height_local, width, &one, cu_square_mat.Data(), width,
                  cu_local_eigvec_mat.Data(), height_local );
              cu_local_eigvec_mat.CopyTo(local_eigvec_mat);


#else
              blas::Trsm( 'R', 'U', 'N', 'N', height_local, width, 1.0, square_mat.Data(), width, 
                  local_eigvec_mat.Data(), height_local );
#endif


              GetTime( timeEnd );
              statusOFS << std::endl << " Orthonormalization completed ( " << (timeEnd - timeSta ) << " s.)";

              // Raleigh-Ritz step: This part is non-scalable and needs to be fixed
              // !!!!!!!!!!!!!!!!!!!!!!!!!!!!
              statusOFS << std::endl << std::endl << " Raleigh-Ritz step ... ";
              GetTime( timeSta );

              // Compute H * X
              DistVec<Index3, DblNumMat, ElemPrtn>  result_mat;
              scfdg_Hamiltonian_times_eigenvectors(result_mat);
              DblNumMat &local_result_mat = (result_mat.LocalMap().begin())->second;

              SetValue(temp_square_mat, 0.0);

              // Compute square_mat = X^T * HX 
#ifdef GPU

              cuDblNumMat cu_local_result_mat(height_local, width);
              cu_local_result_mat.CopyFrom(local_result_mat);
              cublas::Gemm( HIPBLAS_OP_T, HIPBLAS_OP_N, width, width, height_local,
                  &one, cu_local_eigvec_mat.Data(), height_local,
                  cu_local_result_mat.Data(), height_local,
                  &zero, cu_temp_square_mat.Data(), width );
              cu_temp_square_mat.CopyTo(temp_square_mat);


#else
              blas::Gemm( 'T', 'N', width, width, height_local, 
                  1.0, local_eigvec_mat.Data(), height_local,
                  local_result_mat.Data(), height_local, 
                  0.0, temp_square_mat.Data(), width );
#endif

              SetValue( square_mat, 0.0 );
              MPI_Allreduce( temp_square_mat.Data(), square_mat.Data(), width*width, MPI_DOUBLE, MPI_SUM, domain_.colComm );


              eig_vals_Raleigh_Ritz.Resize(width);
              SetValue(eig_vals_Raleigh_Ritz, 0.0);

              if ( mpirank == 0 ) {
#if 0 //def GPU
              cu_square_mat.CopyFrom(square_mat);
              cuDblNumVec cu_eig_vals_Raleigh_Ritz(width);
              cu_eig_vals_Raleigh_Ritz.CopyFrom(eig_vals_Raleigh_Ritz);
              //MAGMA::Syevd( 'V', 'U', width, cu_square_mat.Data(), width, eig_vals_Raleigh_Ritz.Data() );
              cusolver::Syev( 'U', width, cu_square_mat.Data(), width, eig_vals_Raleigh_Ritz.Data() );
              cu_square_mat.CopyTo(square_mat);
              //cu_eig_vals_Raleigh_Ritz.CopyTo(eig_vals_Raleigh_Ritz);
#else
                lapack::Syevd( 'V', 'U', width, square_mat.Data(), width, eig_vals_Raleigh_Ritz.Data() );
#endif
              }

              // ~~ Send the results to every process
              MPI_Bcast(square_mat.Data(), width*width, MPI_DOUBLE, 0, domain_.comm); // Eigen-vectors
              MPI_Bcast(eig_vals_Raleigh_Ritz.Data(), width, MPI_DOUBLE, 0,  domain_.comm); // Eigen-values

              GetTime( timeEnd );
              statusOFS << std::endl << " Raleigh-Ritz step completed ( " << (timeEnd - timeSta ) << " s.)";


              // Subspace rotation step X <- X * Q: This part is non-scalable and needs to be fixed
              // !!!!!!!!!!!!!!!!!!!!!!!!!!!!
              statusOFS << std::endl << std::endl << " Subspace rotation step ... ";
              GetTime( timeSta );

              // ~~ So copy X to HX 
              lapack::Lacpy( 'A', height_local, width, local_eigvec_mat.Data(),  height_local, local_result_mat.Data(), height_local );

              // ~~ Gemm: X <-- HX (= X) * Q
#ifdef GPU

             cu_local_result_mat.CopyFrom(local_result_mat);
             cu_square_mat.CopyFrom(square_mat);
             cublas::Gemm( HIPBLAS_OP_N, HIPBLAS_OP_N, height_local, width, width, &one, cu_local_result_mat.Data(),
                           height_local, cu_square_mat.Data(), width, &zero, cu_local_eigvec_mat.Data(), height_local );
             cu_local_eigvec_mat.CopyTo(local_eigvec_mat);


#else
              blas::Gemm( 'N', 'N', height_local, width, width, 1.0, local_result_mat.Data(),
                  height_local, square_mat.Data(), width, 0.0, local_eigvec_mat.Data(), height_local );    
#endif
              GetTime( timeEnd );
              statusOFS << std::endl << " Subspace rotation step completed ( " << (timeEnd - timeSta ) << " s.)";

              // Reset the filtering bounds using results of the Raleigh-Ritz step
              b_low = eig_vals_Raleigh_Ritz(width - 1);
              a_L = eig_vals_Raleigh_Ritz(0);

              // Fill up the eigen-values
              DblNumVec& eigval = hamDG.EigVal(); 
              eigval.Resize( hamDG.NumStateTotal() );    

              for(Int i = 0; i < hamDG.NumStateTotal(); i ++)
                eigval[i] =  eig_vals_Raleigh_Ritz[i];



            } // End of if(SCFDG_Cheby_use_ScaLAPACK_ == 0)
            else
            {
              // Do the subspace problems using ScaLAPACK
              statusOFS << std::endl << std::endl << " Solving subspace problems using ScaLAPACK ...";

              statusOFS << std::endl << " Setting up BLACS and ScaLAPACK Process Grid ...";
              GetTime( timeSta );

              double detail_timeSta, detail_timeEnd;

              // Basic ScaLAPACK setup steps
              //Number of ScaLAPACK processors equal to number of DG elements
              const int num_cheby_scala_procs = mpisizeCol; 

              // Figure out the process grid dimensions
              int temp_factor = int(sqrt(double(num_cheby_scala_procs)));
              while(num_cheby_scala_procs % temp_factor != 0 )
                ++temp_factor;

              // temp_factor now contains the process grid height
              const int cheby_scala_num_rows = temp_factor;      
              const int cheby_scala_num_cols = num_cheby_scala_procs / temp_factor;



              // Set up the ScaLAPACK context
              IntNumVec cheby_scala_pmap(num_cheby_scala_procs);

              // Use the first processor from every DG-element 
              for ( Int pmap_iter = 0; pmap_iter < num_cheby_scala_procs; pmap_iter++ )
                cheby_scala_pmap[pmap_iter] = pmap_iter * mpisizeRow; 

              // Set up BLACS for subsequent ScaLAPACK operations
              Int cheby_scala_context = -2;
              dgdft::scalapack::Cblacs_get( 0, 0, &cheby_scala_context );
              dgdft::scalapack::Cblacs_gridmap(&cheby_scala_context, &cheby_scala_pmap[0], cheby_scala_num_rows, cheby_scala_num_rows, cheby_scala_num_cols);

              // Figure out my ScaLAPACK information
              int dummy_np_row, dummy_np_col;
              int my_cheby_scala_proc_row, my_cheby_scala_proc_col;

              dgdft::scalapack::Cblacs_gridinfo(cheby_scala_context, &dummy_np_row, &dummy_np_col, &my_cheby_scala_proc_row, &my_cheby_scala_proc_col);


              GetTime( timeEnd );
              statusOFS << " Done. ( " << (timeEnd - timeSta ) << " s.)";

              statusOFS << std::endl << " ScaLAPACK will use " << num_cheby_scala_procs << " processes.";
              statusOFS << std::endl << " ScaLAPACK process grid = " << cheby_scala_num_rows << " * " << cheby_scala_num_cols << " ."  << std::endl;

              // Eigenvcetors in ScaLAPACK format : this will be used multiple times
              dgdft::scalapack::ScaLAPACKMatrix<Real>  cheby_scala_eigvecs_X; // Declared here for scope, empty constructor invoked

              if(cheby_scala_context >= 0)
              { 
                // Now setup the ScaLAPACK matrix
                statusOFS << std::endl << " Orthonormalization step : ";
                statusOFS << std::endl << " Distributed vector to ScaLAPACK format conversion ... ";
                GetTime( timeSta );

                // The dimensions should be  hamDG.NumBasisTotal() * hamDG.NumStateTotal()
                // But this is not verified here as far as the distributed vector is concerned
                dgdft::scalapack::Descriptor cheby_eigvec_desc( hamDG.NumBasisTotal(), hamDG.NumStateTotal(), 
                    scaBlockSize_, scaBlockSize_, 
                    0, 0, 
                    cheby_scala_context);



                // Make the conversion call         
                scfdg_Cheby_convert_eigvec_distmat_to_ScaLAPACK(hamDG.EigvecCoef(),
                    domain_.colComm,
                    cheby_eigvec_desc,
                    cheby_scala_eigvecs_X);


                //         //Older version of conversion call
                //         // Store the important ScaLAPACK information
                //         std::vector<int> my_cheby_scala_info;
                //         my_cheby_scala_info.resize(4,0);
                //         my_cheby_scala_info[0] = cheby_scala_num_rows;
                //         my_cheby_scala_info[1] = cheby_scala_num_cols;
                //         my_cheby_scala_info[2] = my_cheby_scala_proc_row;
                //         my_cheby_scala_info[3] = my_cheby_scala_proc_col;
                //       

                //         scfdg_Cheby_convert_eigvec_distmat_to_ScaLAPACK_old(hamDG.EigvecCoef(),
                //                                     my_cheby_scala_info,
                //                                     cheby_eigvec_desc,
                //                                     cheby_scala_eigvecs_X);
                //         





                GetTime( timeEnd );
                statusOFS << " Done. ( " << (timeEnd - timeSta ) << " s.)";


                statusOFS << std::endl << " Orthonormalizing filtered vectors ... ";
                GetTime( timeSta );

                GetTime( detail_timeSta);

                // Compute C = X^T * X
                dgdft::scalapack::Descriptor cheby_chol_desc( hamDG.NumStateTotal(), hamDG.NumStateTotal(), 
                    scaBlockSize_, scaBlockSize_, 
                    0, 0, 
                    cheby_scala_context);

                dgdft::scalapack::ScaLAPACKMatrix<Real>  cheby_scala_chol_mat;
                cheby_scala_chol_mat.SetDescriptor(cheby_chol_desc);

                dgdft::scalapack::Gemm( 'T', 'N',
                    hamDG.NumStateTotal(), hamDG.NumStateTotal(), hamDG.NumBasisTotal(),
                    1.0,
                    cheby_scala_eigvecs_X.Data(), I_ONE, I_ONE, cheby_scala_eigvecs_X.Desc().Values(), 
                    cheby_scala_eigvecs_X.Data(), I_ONE, I_ONE, cheby_scala_eigvecs_X.Desc().Values(),
                    0.0,
                    cheby_scala_chol_mat.Data(), I_ONE, I_ONE, cheby_scala_chol_mat.Desc().Values(),
                    cheby_scala_context);

                GetTime( detail_timeEnd);
                statusOFS << std::endl << " Overlap matrix computed in : " << (detail_timeEnd - detail_timeSta) << " s.";


                GetTime( detail_timeSta);
                // Compute V = Chol(C)
                dgdft::scalapack::Potrf( 'U', cheby_scala_chol_mat);

                GetTime( detail_timeEnd);
                statusOFS << std::endl << " Cholesky factorization computed in : " << (detail_timeEnd - detail_timeSta) << " s.";


                GetTime( detail_timeSta);
                // Compute  X = X * V^{-1}
                dgdft::scalapack::Trsm( 'R', 'U', 'N', 'N', 1.0,
                    cheby_scala_chol_mat, 
                    cheby_scala_eigvecs_X );

                GetTime( detail_timeEnd);
                statusOFS << std::endl << " TRSM computed in : " << (detail_timeEnd - detail_timeSta) << " s.";

                GetTime( timeEnd );
                statusOFS << std::endl << " Orthonormalization steps completed ( " << (timeEnd - timeSta ) << " s.)" << std::endl;

                statusOFS << std::endl << " ScaLAPACK to Distributed vector format conversion ... ";
                GetTime( timeSta );

                // Now convert this back to DG-distributed matrix format
                ScaMatToDistNumMat(cheby_scala_eigvecs_X ,
                    elemPrtn_,
                    hamDG.EigvecCoef(),
                    hamDG.ElemBasisIdx(), 
                    domain_.colComm, 
                    hamDG.NumStateTotal() );

                GetTime( timeEnd );
                statusOFS << " Done. ( " << (timeEnd - timeSta ) << " s.)";


              } // End of  if(cheby_scala_context >= 0)
              else
              {
                statusOFS << std::endl << " Waiting for ScaLAPACK solution of subspace problems ...";
              }        

              // Communicate the orthonormalized eigenvectors (to other intra-element processors)
              statusOFS << std::endl << " Communicating orthonormalized filtered vectors ... ";
              GetTime( timeSta );

              DblNumMat &ref_mat_1 =  (hamDG.EigvecCoef().LocalMap().begin())->second;
              MPI_Bcast(ref_mat_1.Data(), (ref_mat_1.m() * ref_mat_1.n()), MPI_DOUBLE, 0, domain_.rowComm);

              GetTime( timeEnd );
              statusOFS << " Done. ( " << (timeEnd - timeSta ) << " s.)" << std::endl;

              // Set up space for eigenvalues in the Hamiltonian object for results of Raleigh-Ritz step
              DblNumVec& eigval = hamDG.EigVal(); 
              eigval.Resize( hamDG.NumStateTotal() );    


              // Compute H * X
              statusOFS << std::endl << " Computing H * X for filtered orthonormal vectors : ";
              GetTime( timeSta );

              DistVec<Index3, DblNumMat, ElemPrtn>  result_mat;
              scfdg_Hamiltonian_times_eigenvectors(result_mat);

              GetTime( timeEnd );
              statusOFS << std::endl << " H * X for filtered orthonormal vectors computed . ( " << (timeEnd - timeSta ) << " s.)";

              // Raleigh-Ritz step
              if(cheby_scala_context >= 0)
              { 
                statusOFS << std::endl << std::endl << " Raleigh - Ritz step : ";


                // Convert HX to ScaLAPACK format
                dgdft::scalapack::ScaLAPACKMatrix<Real>  cheby_scala_HX;
                dgdft::scalapack::Descriptor cheby_HX_desc( hamDG.NumBasisTotal(), hamDG.NumStateTotal(), 
                    scaBlockSize_, scaBlockSize_, 
                    0, 0, 
                    cheby_scala_context);


                statusOFS << std::endl << " Distributed vector to ScaLAPACK format conversion ... ";
                GetTime( timeSta );


                scfdg_Cheby_convert_eigvec_distmat_to_ScaLAPACK(result_mat,
                    domain_.colComm,
                    cheby_HX_desc,
                    cheby_scala_HX);




                GetTime( timeEnd );
                statusOFS << " Done. ( " << (timeEnd - timeSta ) << " s.)";

                statusOFS << std::endl << " Solving the subspace problem ... ";
                GetTime( timeSta );

                dgdft::scalapack::Descriptor cheby_XTHX_desc( hamDG.NumStateTotal(), hamDG.NumStateTotal(), 
                    scaBlockSize_, scaBlockSize_, 
                    0, 0, 
                    cheby_scala_context);

                dgdft::scalapack::ScaLAPACKMatrix<Real>  cheby_scala_XTHX_mat;
                cheby_scala_XTHX_mat.SetDescriptor(cheby_XTHX_desc);

                GetTime( detail_timeSta);

                dgdft::scalapack::Gemm( 'T', 'N',
                    hamDG.NumStateTotal(), hamDG.NumStateTotal(), hamDG.NumBasisTotal(),
                    1.0,
                    cheby_scala_eigvecs_X.Data(), I_ONE, I_ONE, cheby_scala_eigvecs_X.Desc().Values(), 
                    cheby_scala_HX.Data(), I_ONE, I_ONE, cheby_scala_HX.Desc().Values(),
                    0.0,
                    cheby_scala_XTHX_mat.Data(), I_ONE, I_ONE, cheby_scala_XTHX_mat.Desc().Values(),
                    cheby_scala_context);


                GetTime( detail_timeEnd);
                statusOFS << std::endl << " X^T(HX) computed in : " << (detail_timeEnd - detail_timeSta) << " s.";


                scalapack::ScaLAPACKMatrix<Real>  scaZ;

                std::vector<Real> eigen_values;

                GetTime( detail_timeSta);

                // Eigenvalue probem solution call
                scalapack::Syevd('U', cheby_scala_XTHX_mat, eigen_values, scaZ);

                // Copy the eigenvalues to the Hamiltonian object          
                for( Int i = 0; i < hamDG.NumStateTotal(); i++ )
                  eigval[i] = eigen_values[i];


                GetTime( detail_timeEnd);
                statusOFS << std::endl << " Eigenvalue problem solved in : " << (detail_timeEnd - detail_timeSta) << " s.";



                // Subspace rotation step : X <- X * Q
                GetTime( detail_timeSta);


                // To save memory, copy X to HX
                blas::Copy((cheby_scala_eigvecs_X.LocalHeight() * cheby_scala_eigvecs_X.LocalWidth()), 
                    cheby_scala_eigvecs_X.Data(),
                    1,
                    cheby_scala_HX.Data(),
                    1);

                // Now perform X <- HX (=X) * Q (=scaZ)
                dgdft::scalapack::Gemm( 'N', 'N',
                    hamDG.NumBasisTotal(), hamDG.NumStateTotal(), hamDG.NumStateTotal(),
                    1.0,
                    cheby_scala_HX.Data(), I_ONE, I_ONE, cheby_scala_HX.Desc().Values(), 
                    scaZ.Data(), I_ONE, I_ONE, scaZ.Desc().Values(),
                    0.0,
                    cheby_scala_eigvecs_X.Data(), I_ONE, I_ONE, cheby_scala_eigvecs_X.Desc().Values(),
                    cheby_scala_context);

                GetTime( detail_timeEnd);
                statusOFS << std::endl << " Subspace rotation step completed in : " << (detail_timeEnd - detail_timeSta) << " s.";



                GetTime( timeEnd );
                statusOFS << std::endl << " All subspace problem steps completed . ( " << (timeEnd - timeSta ) << " s.)" << std::endl ;


                // Convert the eigenvectors back to distributed vector format
                statusOFS << std::endl << " ScaLAPACK to Distributed vector format conversion ... ";
                GetTime( timeSta );

                // Now convert this back to DG-distributed matrix format
                ScaMatToDistNumMat(cheby_scala_eigvecs_X ,
                    elemPrtn_,
                    hamDG.EigvecCoef(),
                    hamDG.ElemBasisIdx(), 
                    domain_.colComm, 
                    hamDG.NumStateTotal() );

                GetTime( timeEnd );
                statusOFS << " Done. ( " << (timeEnd - timeSta ) << " s.)";


              }  // End of if(cheby_scala_context >= 0)

              // Communicate the final eigenvectors (to other intra-element processors)
              statusOFS << std::endl << " Communicating eigenvalues and eigenvectors ... ";
              GetTime( timeSta );

              DblNumMat &ref_mat_2 =  (hamDG.EigvecCoef().LocalMap().begin())->second;
              MPI_Bcast(ref_mat_2.Data(), (ref_mat_2.m() * ref_mat_2.n()), MPI_DOUBLE, 0, domain_.rowComm);

              // Communicate the final eigenvalues (to other intra-element processors)
              MPI_Bcast(eigval.Data(), ref_mat_2.n(), MPI_DOUBLE, 0,  domain_.rowComm); // Eigen-values


              GetTime( timeEnd );
              statusOFS << " Done. ( " << (timeEnd - timeSta ) << " s.)";




              // Reset the filtering bounds using results of the Raleigh-Ritz step    
              b_low = eigval(ref_mat_2.n() - 1);
              a_L = eigval(0);

              MPI_Barrier(domain_.rowComm);
              MPI_Barrier(domain_.colComm);
              MPI_Barrier(domain_.comm);

              // Clean up BLACS
              if(cheby_scala_context >= 0) {
                dgdft::scalapack::Cblacs_gridexit( cheby_scala_context );
              }




            } // End of if(SCFDG_Cheby_use_ScaLAPACK_ == 0) ... else 



            statusOFS << std::endl << " ------------------------------- ";
            GetTime( cheby_timeEnd );

            //statusOFS << std::endl << " Eigenvalues via Chebyshev filtering : " << std::endl;
            //statusOFS << eig_vals_Raleigh_Ritz << std::endl;

            statusOFS << std::endl << " This Chebyshev cycle took a total of " << (cheby_timeEnd - cheby_timeSta ) << " s.";
            statusOFS << std::endl << " ------------------------------- " << std::endl;

          } // End of loop over inner iteration repeats





        }

 #ifdef GPU
      void 
        SCFDG::scfdg_GeneralChebyStep_GPU(Int MaxIter, 
            Int filter_order )
        {
          statusOFS << std::endl << " SCFDG GeneralCheby GPU begin "  << std::endl;

          Int mpirank, mpisize;
          MPI_Comm_rank( domain_.comm, &mpirank );
          MPI_Comm_size( domain_.comm, &mpisize );
          Int mpirankRow;  MPI_Comm_rank(domain_.rowComm, &mpirankRow);
          Int mpisizeRow;  MPI_Comm_size(domain_.rowComm, &mpisizeRow);
          Int mpirankCol;  MPI_Comm_rank(domain_.colComm, &mpirankCol);
          Int mpisizeCol;  MPI_Comm_size(domain_.colComm, &mpisizeCol);

          Real timeSta, timeEnd;
          Real extra_timeSta, extra_timeEnd;
          Real cheby_timeSta, cheby_timeEnd;

          HamiltonianDG&  hamDG = *hamDGPtr_;

          DblNumVec& eigval = hamDG.EigVal(); 

          // Step 0: Safeguard against the eigenvector containing extra keys
          {
            std::map<Index3, DblNumMat>::iterator cleaner_itr = hamDG.EigvecCoef().LocalMap().begin();
            while (cleaner_itr != hamDG.EigvecCoef().LocalMap().end()) 
            {
              if (cleaner_itr->first != my_cheby_eig_vec_key_) 
              {
                std::map<Index3, DblNumMat>::iterator toErase = cleaner_itr;
                ++ cleaner_itr;
                hamDG.EigvecCoef().LocalMap().erase(toErase);
              } 
              else 
              {
                ++ cleaner_itr;
              }
            }
          }


          // Step 1: Obtain the upper bound and the Ritz values (for the lower bound)
          // using the Lanczos estimator

          DblNumVec Lanczos_Ritz_values;


          statusOFS << std::endl << " Estimating the spectral bounds ... "; 
          GetTime( extra_timeSta );
          const int Num_Lanczos_Steps = 6;
          double b_up = scfdg_Cheby_Upper_bound_estimator(Lanczos_Ritz_values, Num_Lanczos_Steps);  
          GetTime( extra_timeEnd );
          statusOFS << " Done. ( " << (extra_timeEnd - extra_timeSta ) << " s.)";

          // statusOFS << std::endl << "Lanczos-Ritz values : " << Lanczos_Ritz_values << std::endl ;

          double a_L = eigval[0];
          double b_low = eigval[hamDG.NumStateTotal() - 1];

//          //statusOFS << std::endl << "a_L " << a_L << "B_low" << b_low <<  std::endl ;

          // Step 2: Main loop
          const Int Iter_Max = MaxIter;
          const Int Filter_Order = filter_order ;

          DblNumVec eig_vals_Raleigh_Ritz;

          for(Int i = 1; i <= Iter_Max; i ++)
          {
            GetTime( cheby_timeSta );
            statusOFS << std::endl << std::endl << " ------------------------------- ";
            statusOFS << std::endl << " General CheFSI step for DGDFT cycle " << i << " of " << Iter_Max << " . ";

            // Filter the eigenvectors
            statusOFS << std::endl << std::endl << " Filtering the eigenvectors ... (Filter order = " << Filter_Order << ")";
            GetTime( timeSta );
            scfdg_Chebyshev_filter_scaled(Filter_Order, b_low, b_up, a_L);
            GetTime( timeEnd );
            statusOFS << std::endl << " Filtering completed. ( " << (timeEnd - timeSta ) << " s.)";


            // Subspace projected problems: Orthonormalization, Raleigh-Ritz and subspace rotation steps
            // This can be done serially or using ScaLAPACK    
            if(SCFDG_Cheby_use_ScaLAPACK_ == 0)
            {
              // Do the subspace problem serially 
              statusOFS << std::endl << std::endl << " Solving subspace problems serially ...";

              // Orthonormalize using Cholesky factorization  
              statusOFS << std::endl << " Orthonormalizing filtered vectors ... ";
              GetTime( timeSta );



              DblNumMat &local_eigvec_mat = (hamDG.EigvecCoef().LocalMap().begin())->second;
              DblNumMat square_mat;
              DblNumMat temp_square_mat;

              Int width = local_eigvec_mat.n();
              Int height_local = local_eigvec_mat.m();

              square_mat.Resize(width, width);
              temp_square_mat.Resize(width, width);

              SetValue(temp_square_mat, 0.0);

              // Compute square_mat = X^T * X for Cholesky    
              blas::Gemm( 'T', 'N', width, width, height_local, 
                  1.0, local_eigvec_mat.Data(), height_local,
                  local_eigvec_mat.Data(), height_local, 
                  0.0, temp_square_mat.Data(), width );

              SetValue( square_mat, 0.0 );
              MPI_Allreduce( temp_square_mat.Data(), square_mat.Data(), width*width, MPI_DOUBLE, MPI_SUM, domain_.colComm );

              // In the following, reduction happens on colComm but the result is broadcast to everyone
              // This can probably be band-parallelized

              // Make the Cholesky factorization call on proc 0
              if ( mpirank == 0) {
                lapack::Potrf( 'U', width, square_mat.Data(), width );
              }
              // Send the Cholesky factor to every process
              MPI_Bcast(square_mat.Data(), width*width, MPI_DOUBLE, 0, domain_.comm);

              // Do a solve with the Cholesky factor : Band parallelization ??
              // X = X * U^{-1} is orthogonal, where U is the Cholesky factor
              blas::Trsm( 'R', 'U', 'N', 'N', height_local, width, 1.0, square_mat.Data(), width, 
                  local_eigvec_mat.Data(), height_local );


              GetTime( timeEnd );
              statusOFS << std::endl << " Orthonormalization completed ( " << (timeEnd - timeSta ) << " s.)";

              // Raleigh-Ritz step: This part is non-scalable and needs to be fixed
              // !!!!!!!!!!!!!!!!!!!!!!!!!!!!
              statusOFS << std::endl << std::endl << " Raleigh-Ritz step ... ";
              GetTime( timeSta );

              // Compute H * X
              DistVec<Index3, DblNumMat, ElemPrtn>  result_mat;
              scfdg_Hamiltonian_times_eigenvectors(result_mat);
              DblNumMat &local_result_mat = (result_mat.LocalMap().begin())->second;

              SetValue(temp_square_mat, 0.0);

              // Compute square_mat = X^T * HX 
              blas::Gemm( 'T', 'N', width, width, height_local, 
                  1.0, local_eigvec_mat.Data(), height_local,
                  local_result_mat.Data(), height_local, 
                  0.0, temp_square_mat.Data(), width );

              SetValue( square_mat, 0.0 );
              MPI_Allreduce( temp_square_mat.Data(), square_mat.Data(), width*width, MPI_DOUBLE, MPI_SUM, domain_.colComm );


              eig_vals_Raleigh_Ritz.Resize(width);
              SetValue(eig_vals_Raleigh_Ritz, 0.0);

              if ( mpirank == 0 ) {
                lapack::Syevd( 'V', 'U', width, square_mat.Data(), width, eig_vals_Raleigh_Ritz.Data() );

              }

              // ~~ Send the results to every process
              MPI_Bcast(square_mat.Data(), width*width, MPI_DOUBLE, 0, domain_.comm); // Eigen-vectors
              MPI_Bcast(eig_vals_Raleigh_Ritz.Data(), width, MPI_DOUBLE, 0,  domain_.comm); // Eigen-values

              GetTime( timeEnd );
              statusOFS << std::endl << " Raleigh-Ritz step completed ( " << (timeEnd - timeSta ) << " s.)";

              // This is for use with the Complementary Subspace strategy in subsequent steps
              if(SCFDG_use_comp_subspace_ == 1)
              {
                GetTime( timeSta );

                statusOFS << std::endl << std::endl << " Copying top states for serial CS strategy ... ";  
                SCFDG_comp_subspace_start_guess_.Resize(width, SCFDG_comp_subspace_N_solve_);

                for(Int copy_iter = 0; copy_iter < SCFDG_comp_subspace_N_solve_; copy_iter ++)
                {

                  blas::Copy( width, square_mat.VecData(width - 1 - copy_iter), 1, 
                      SCFDG_comp_subspace_start_guess_.VecData(copy_iter), 1 );

                  // lapack::Lacpy( 'A', width, 1, square_mat.VecData(width - 1 - copy_iter), width, 
                  //      SCFDG_comp_subspace_start_guess_.VecData(copy_iter), width );
                }

                GetTime( timeEnd );
                statusOFS << " Done. ( " << (timeEnd - timeSta ) << " s.)";
              }


              // Subspace rotation step X <- X * Q: This part is non-scalable and needs to be fixed
              // !!!!!!!!!!!!!!!!!!!!!!!!!!!!
              statusOFS << std::endl << std::endl << " Subspace rotation step ... ";
              GetTime( timeSta );

              // ~~ So copy X to HX 
              lapack::Lacpy( 'A', height_local, width, local_eigvec_mat.Data(),  height_local, local_result_mat.Data(), height_local );

              // ~~ Gemm: X <-- HX (= X) * Q
              blas::Gemm( 'N', 'N', height_local, width, width, 1.0, local_result_mat.Data(),
                  height_local, square_mat.Data(), width, 0.0, local_eigvec_mat.Data(), height_local );    

              GetTime( timeEnd );
              statusOFS << std::endl << " Subspace rotation step completed ( " << (timeEnd - timeSta ) << " s.)";

              // Reset the filtering bounds using results of the Raleigh-Ritz step
              b_low = eig_vals_Raleigh_Ritz(width - 1);
              a_L = eig_vals_Raleigh_Ritz(0);

              // Fill up the eigen-values
              DblNumVec& eigval = hamDG.EigVal(); 
              eigval.Resize( hamDG.NumStateTotal() );    

              for(Int i = 0; i < hamDG.NumStateTotal(); i ++)
                eigval[i] =  eig_vals_Raleigh_Ritz[i];



            } // End of if(SCFDG_Cheby_use_ScaLAPACK_ == 0)
            else
            {
              // Do the subspace problems using ScaLAPACK
              statusOFS << std::endl << std::endl << " Solving subspace problems using ScaLAPACK ...";

              statusOFS << std::endl << " Setting up BLACS and ScaLAPACK Process Grid ...";
              GetTime( timeSta );

              double detail_timeSta, detail_timeEnd;

              // Basic ScaLAPACK setup steps
              //Number of ScaLAPACK processors equal to number of DG elements
              const int num_cheby_scala_procs = mpisizeCol; 

              // Figure out the process grid dimensions
              int temp_factor = int(sqrt(double(num_cheby_scala_procs)));
              while(num_cheby_scala_procs % temp_factor != 0 )
                ++temp_factor;

              // temp_factor now contains the process grid height
              const int cheby_scala_num_rows = temp_factor;      
              const int cheby_scala_num_cols = num_cheby_scala_procs / temp_factor;



              // Set up the ScaLAPACK context
              IntNumVec cheby_scala_pmap(num_cheby_scala_procs);

              // Use the first processor from every DG-element 
              for ( Int pmap_iter = 0; pmap_iter < num_cheby_scala_procs; pmap_iter++ )
                cheby_scala_pmap[pmap_iter] = pmap_iter * mpisizeRow; 

              // Set up BLACS for subsequent ScaLAPACK operations
              Int cheby_scala_context = -2;
              dgdft::scalapack::Cblacs_get( 0, 0, &cheby_scala_context );
              dgdft::scalapack::Cblacs_gridmap(&cheby_scala_context, &cheby_scala_pmap[0], cheby_scala_num_rows, cheby_scala_num_rows, cheby_scala_num_cols);

              // Figure out my ScaLAPACK information
              int dummy_np_row, dummy_np_col;
              int my_cheby_scala_proc_row, my_cheby_scala_proc_col;

              dgdft::scalapack::Cblacs_gridinfo(cheby_scala_context, &dummy_np_row, &dummy_np_col, &my_cheby_scala_proc_row, &my_cheby_scala_proc_col);


              GetTime( timeEnd );
              statusOFS << " Done. ( " << (timeEnd - timeSta ) << " s.)";

              statusOFS << std::endl << " ScaLAPACK will use " << num_cheby_scala_procs << " processes.";
              statusOFS << std::endl << " ScaLAPACK process grid = " << cheby_scala_num_rows << " * " << cheby_scala_num_cols << " ."  << std::endl;

              // Eigenvcetors in ScaLAPACK format : this will be used multiple times
              dgdft::scalapack::ScaLAPACKMatrix<Real>  cheby_scala_eigvecs_X; // Declared here for scope, empty constructor invoked

              if(cheby_scala_context >= 0)
              { 
                // Now setup the ScaLAPACK matrix
                statusOFS << std::endl << " Orthonormalization step : ";
                statusOFS << std::endl << " Distributed vector to ScaLAPACK format conversion ... ";
                GetTime( timeSta );

                // The dimensions should be  hamDG.NumBasisTotal() * hamDG.NumStateTotal()
                // But this is not verified here as far as the distributed vector is concerned
                dgdft::scalapack::Descriptor cheby_eigvec_desc( hamDG.NumBasisTotal(), hamDG.NumStateTotal(), 
                    scaBlockSize_, scaBlockSize_, 
                    0, 0, 
                    cheby_scala_context);



                // Make the conversion call         
                scfdg_Cheby_convert_eigvec_distmat_to_ScaLAPACK(hamDG.EigvecCoef(),
                    domain_.colComm,
                    cheby_eigvec_desc,
                    cheby_scala_eigvecs_X);


                //         //Older version of conversion call
                //         // Store the important ScaLAPACK information
                //         std::vector<int> my_cheby_scala_info;
                //         my_cheby_scala_info.resize(4,0);
                //         my_cheby_scala_info[0] = cheby_scala_num_rows;
                //         my_cheby_scala_info[1] = cheby_scala_num_cols;
                //         my_cheby_scala_info[2] = my_cheby_scala_proc_row;
                //         my_cheby_scala_info[3] = my_cheby_scala_proc_col;
                //       

                //         scfdg_Cheby_convert_eigvec_distmat_to_ScaLAPACK_old(hamDG.EigvecCoef(),
                //                                     my_cheby_scala_info,
                //                                     cheby_eigvec_desc,
                //                                     cheby_scala_eigvecs_X);
                //         


                GetTime( timeEnd );
                statusOFS << " Done. ( " << (timeEnd - timeSta ) << " s.)";


                statusOFS << std::endl << " Orthonormalizing filtered vectors ... ";
                GetTime( timeSta );

                GetTime( detail_timeSta);

                // Compute C = X^T * X
                dgdft::scalapack::Descriptor cheby_chol_desc( hamDG.NumStateTotal(), hamDG.NumStateTotal(), 
                    scaBlockSize_, scaBlockSize_, 
                    0, 0, 
                    cheby_scala_context);

                dgdft::scalapack::ScaLAPACKMatrix<Real>  cheby_scala_chol_mat;
                cheby_scala_chol_mat.SetDescriptor(cheby_chol_desc);

                dgdft::scalapack::Gemm( 'T', 'N',
                    hamDG.NumStateTotal(), hamDG.NumStateTotal(), hamDG.NumBasisTotal(),
                    1.0,
                    cheby_scala_eigvecs_X.Data(), I_ONE, I_ONE, cheby_scala_eigvecs_X.Desc().Values(), 
                    cheby_scala_eigvecs_X.Data(), I_ONE, I_ONE, cheby_scala_eigvecs_X.Desc().Values(),
                    0.0,
                    cheby_scala_chol_mat.Data(), I_ONE, I_ONE, cheby_scala_chol_mat.Desc().Values(),
                    cheby_scala_context);

                GetTime( detail_timeEnd);
                statusOFS << std::endl << " Overlap matrix computed in : " << (detail_timeEnd - detail_timeSta) << " s.";


                GetTime( detail_timeSta);
                // Compute V = Chol(C)
                dgdft::scalapack::Potrf( 'U', cheby_scala_chol_mat);

                GetTime( detail_timeEnd);
                statusOFS << std::endl << " Cholesky factorization computed in : " << (detail_timeEnd - detail_timeSta) << " s.";


                GetTime( detail_timeSta);
                // Compute  X = X * V^{-1}
                dgdft::scalapack::Trsm( 'R', 'U', 'N', 'N', 1.0,
                    cheby_scala_chol_mat, 
                    cheby_scala_eigvecs_X );

                GetTime( detail_timeEnd);
                statusOFS << std::endl << " TRSM computed in : " << (detail_timeEnd - detail_timeSta) << " s.";


                GetTime( timeEnd );
                statusOFS << std::endl << " Orthonormalization steps completed ( " << (timeEnd - timeSta ) << " s.)" << std::endl;

                statusOFS << std::endl << " ScaLAPACK to Distributed vector format conversion ... ";
                GetTime( timeSta );

                // Now convert this back to DG-distributed matrix format
                ScaMatToDistNumMat(cheby_scala_eigvecs_X ,
                    elemPrtn_,
                    hamDG.EigvecCoef(),
                    hamDG.ElemBasisIdx(), 
                    domain_.colComm, 
                    hamDG.NumStateTotal() );

                GetTime( timeEnd );
                statusOFS << " Done. ( " << (timeEnd - timeSta ) << " s.)";


              } // End of  if(cheby_scala_context >= 0)
              else
              {
                statusOFS << std::endl << " Waiting for ScaLAPACK solution of subspace problems ...";
              }    

              // Communicate the orthonormalized eigenvectors (to other intra-element processors)
              statusOFS << std::endl << " Communicating orthonormalized filtered vectors ... ";
              GetTime( timeSta );

              DblNumMat &ref_mat_1 =  (hamDG.EigvecCoef().LocalMap().begin())->second;
              MPI_Bcast(ref_mat_1.Data(), (ref_mat_1.m() * ref_mat_1.n()), MPI_DOUBLE, 0, domain_.rowComm);

              GetTime( timeEnd );
              statusOFS << " Done. ( " << (timeEnd - timeSta ) << " s.)" << std::endl;

              // Set up space for eigenvalues in the Hamiltonian object for results of Raleigh-Ritz step
              DblNumVec& eigval = hamDG.EigVal(); 
              eigval.Resize( hamDG.NumStateTotal() );    

              // Compute H * X
              statusOFS << std::endl << " Computing H * X for filtered orthonormal vectors : ";
              GetTime( timeSta );

              DistVec<Index3, DblNumMat, ElemPrtn>  result_mat;
              scfdg_Hamiltonian_times_eigenvectors(result_mat);

              GetTime( timeEnd );
              statusOFS << std::endl << " H * X for filtered orthonormal vectors computed . ( " << (timeEnd - timeSta ) << " s.)";


              // Set up a single process ScaLAPACK matrix for use with parallel CS strategy          
              // This is for copying the relevant portion of scaZ                        
              Int single_proc_context = -1;
              scalapack::Descriptor single_proc_desc;
              scalapack::ScaLAPACKMatrix<Real>  single_proc_scala_mat;


              if(SCFDG_use_comp_subspace_ == 1 && SCFDG_comp_subspace_N_solve_ != 0)
              {  

                // Reserve the serial space : this will actually contain the vectors in the reversed order
                SCFDG_comp_subspace_start_guess_.Resize( hamDG.NumStateTotal(), SCFDG_comp_subspace_N_solve_);

                Int single_proc_pmap[1];   
                single_proc_pmap[0] = 0; // Just using proc. 0 for the job.

                // Set up BLACS for for the single proc context

                dgdft::scalapack::Cblacs_get( 0, 0, &single_proc_context );
                dgdft::scalapack::Cblacs_gridmap(&single_proc_context, &single_proc_pmap[0], 1, 1, 1);

                if( single_proc_context >= 0)
                {
                  // For safety, make sure this is MPI Rank zero : throw an error otherwise
                  // Fix this in the future ?
                  if(mpirank != 0)
                  {
                    statusOFS << std::endl << std::endl << "  Error !! BLACS rank 0 does not match MPI rank 0 "
                      << " Aborting ... " << std::endl << std::endl;
                    MPI_Abort(domain_.comm, 0);
                  }

                  single_proc_desc.Init( hamDG.NumStateTotal(), SCFDG_comp_subspace_N_solve_,
                      scaBlockSize_, scaBlockSize_, 
                      0, 0,  single_proc_context );              

                  single_proc_scala_mat.SetDescriptor( single_proc_desc );

                }
              }


              // Raleigh-Ritz step
              if(cheby_scala_context >= 0)
              { 
                statusOFS << std::endl << std::endl << " Raleigh - Ritz step : ";

                // Convert HX to ScaLAPACK format
                dgdft::scalapack::ScaLAPACKMatrix<Real>  cheby_scala_HX;
                dgdft::scalapack::Descriptor cheby_HX_desc( hamDG.NumBasisTotal(), hamDG.NumStateTotal(), 
                    scaBlockSize_, scaBlockSize_, 
                    0, 0, 
                    cheby_scala_context);


                statusOFS << std::endl << " Distributed vector to ScaLAPACK format conversion ... ";
                GetTime( timeSta );

                // Make the conversion call                 
                scfdg_Cheby_convert_eigvec_distmat_to_ScaLAPACK(result_mat,
                    domain_.colComm,
                    cheby_HX_desc,
                    cheby_scala_HX);


                GetTime( timeEnd );
                statusOFS << " Done. ( " << (timeEnd - timeSta ) << " s.)";

                statusOFS << std::endl << " Solving the subspace problem ... ";
                GetTime( timeSta );

                dgdft::scalapack::Descriptor cheby_XTHX_desc( hamDG.NumStateTotal(), hamDG.NumStateTotal(), 
                    scaBlockSize_, scaBlockSize_, 
                    0, 0, 
                    cheby_scala_context);

                dgdft::scalapack::ScaLAPACKMatrix<Real>  cheby_scala_XTHX_mat;
                cheby_scala_XTHX_mat.SetDescriptor(cheby_XTHX_desc);


                GetTime( detail_timeSta);

                dgdft::scalapack::Gemm( 'T', 'N',
                    hamDG.NumStateTotal(), hamDG.NumStateTotal(), hamDG.NumBasisTotal(),
                    1.0,
                    cheby_scala_eigvecs_X.Data(), I_ONE, I_ONE, cheby_scala_eigvecs_X.Desc().Values(), 
                    cheby_scala_HX.Data(), I_ONE, I_ONE, cheby_scala_HX.Desc().Values(),
                    0.0,
                    cheby_scala_XTHX_mat.Data(), I_ONE, I_ONE, cheby_scala_XTHX_mat.Desc().Values(),
                    cheby_scala_context);


                GetTime( detail_timeEnd);
                statusOFS << std::endl << " X^T(HX) computed in : " << (detail_timeEnd - detail_timeSta) << " s.";

                scalapack::ScaLAPACKMatrix<Real>  scaZ;

                std::vector<Real> eigen_values;

                GetTime( detail_timeSta);

                // Eigenvalue probem solution call
                scalapack::Syevd('U', cheby_scala_XTHX_mat, eigen_values, scaZ);

                // Copy the eigenvalues to the Hamiltonian object          
                for( Int i = 0; i < hamDG.NumStateTotal(); i++ )
                  eigval[i] = eigen_values[i];

                GetTime( detail_timeEnd);
                statusOFS << std::endl << " Eigenvalue problem solved in : " << (detail_timeEnd - detail_timeSta) << " s.";


                // This is for use with the Complementary Subspace strategy in subsequent steps
                if(SCFDG_use_comp_subspace_ == 1 && SCFDG_comp_subspace_N_solve_ != 0)
                {
                  GetTime( detail_timeSta);
                  statusOFS << std::endl << std::endl << " Distributing and copying top states for parallel CS strategy ... ";


                  const Int M_copy_ = hamDG.NumStateTotal();
                  const Int N_copy_ = SCFDG_comp_subspace_N_solve_;
                  const Int copy_src_col = M_copy_ - N_copy_ + 1;

                  // Note that this is being called inside cheby_scala_context >= 0
                  SCALAPACK(pdgemr2d)(&M_copy_, &N_copy_, 
                      scaZ.Data(), &I_ONE, &copy_src_col,
                      scaZ.Desc().Values(), 
                      single_proc_scala_mat.Data(), &I_ONE, &I_ONE, 
                      single_proc_scala_mat.Desc().Values(), 
                      &cheby_scala_context);    


                  // Copy the data from single_proc_scala_mat to SCFDG_comp_subspace_start_guess_
                  if( single_proc_context >= 0)
                  {
                    double *src_ptr, *dest_ptr; 

                    //                 statusOFS << std::endl << std::endl 
                    //                            << " Ht = " << single_proc_scala_mat.Height()
                    //                            << " Width = " << single_proc_scala_mat.Width()
                    //                            << " loc. Ht = " << single_proc_scala_mat.LocalHeight()
                    //                            << " loc. Width = " << single_proc_scala_mat.LocalWidth()
                    //                            << " loc. LD = " << single_proc_scala_mat.LocalLDim();           
                    //                 statusOFS << std::endl << std::endl;                        
                    //                
                    // Do this in the reverse order      
                    for(Int copy_iter = 0; copy_iter < SCFDG_comp_subspace_N_solve_; copy_iter ++)
                    {
                      src_ptr = single_proc_scala_mat.Data() + (SCFDG_comp_subspace_N_solve_ - copy_iter - 1) * single_proc_scala_mat.LocalLDim();
                      dest_ptr =  SCFDG_comp_subspace_start_guess_.VecData(copy_iter);

                      blas::Copy( M_copy_, src_ptr, 1, dest_ptr, 1 );                          
                    }


                  }


                  GetTime( detail_timeEnd);
                  statusOFS << "Done. (" << (detail_timeEnd - detail_timeSta) << " s.)";

                } // end of if(SCFDG_use_comp_subspace_ == 1)



                // Subspace rotation step : X <- X * Q        
                statusOFS << std::endl << std::endl << " Subspace rotation step ...  ";
                GetTime( detail_timeSta);

                // To save memory, copy X to HX
                blas::Copy((cheby_scala_eigvecs_X.LocalHeight() * cheby_scala_eigvecs_X.LocalWidth()), 
                    cheby_scala_eigvecs_X.Data(),
                    1,
                    cheby_scala_HX.Data(),
                    1);

                // Now perform X <- HX (=X) * Q (=scaZ)
                dgdft::scalapack::Gemm( 'N', 'N',
                    hamDG.NumBasisTotal(), hamDG.NumStateTotal(), hamDG.NumStateTotal(),
                    1.0,
                    cheby_scala_HX.Data(), I_ONE, I_ONE, cheby_scala_HX.Desc().Values(), 
                    scaZ.Data(), I_ONE, I_ONE, scaZ.Desc().Values(),
                    0.0,
                    cheby_scala_eigvecs_X.Data(), I_ONE, I_ONE, cheby_scala_eigvecs_X.Desc().Values(),
                    cheby_scala_context);

                GetTime( detail_timeEnd);
                statusOFS << " Done. (" << (detail_timeEnd - detail_timeSta) << " s.)";





                GetTime( timeEnd );
                statusOFS << std::endl << " All subspace problem steps completed . ( " << (timeEnd - timeSta ) << " s.)" << std::endl ;

                // Convert the eigenvectors back to distributed vector format
                statusOFS << std::endl << " ScaLAPACK to Distributed vector format conversion ... ";
                GetTime( timeSta );

                // Now convert this back to DG-distributed matrix format
                ScaMatToDistNumMat(cheby_scala_eigvecs_X ,
                    elemPrtn_,
                    hamDG.EigvecCoef(),
                    hamDG.ElemBasisIdx(), 
                    domain_.colComm, 
                    hamDG.NumStateTotal() );

                GetTime( timeEnd );
                statusOFS << " Done. ( " << (timeEnd - timeSta ) << " s.)";


              } // End of if(cheby_scala_context >= 0)
              else
              {
                statusOFS << std::endl << std::endl << " Waiting for ScaLAPACK solution of subspace problems ...";
              }





              // Communicate the final eigenvectors (to other intra-element processors)
              statusOFS << std::endl << " Communicating eigenvalues and eigenvectors ... ";
              GetTime( timeSta );

              DblNumMat &ref_mat_2 =  (hamDG.EigvecCoef().LocalMap().begin())->second;
              MPI_Bcast(ref_mat_2.Data(), (ref_mat_2.m() * ref_mat_2.n()), MPI_DOUBLE, 0, domain_.rowComm);

              // Communicate the final eigenvalues (to other intra-element processors)
              MPI_Bcast(eigval.Data(), ref_mat_2.n(), MPI_DOUBLE, 0,  domain_.rowComm); // Eigen-values


              GetTime( timeEnd );
              statusOFS << " Done. ( " << (timeEnd - timeSta ) << " s.)";


              // Broadcast the guess vectors for the CS strategy from proc 0
              if(SCFDG_use_comp_subspace_ == 1 && SCFDG_comp_subspace_N_solve_ != 0)
              {
                statusOFS << std::endl << std::endl << " Broadcasting guess vectors for parallel CS strategy ... ";
                GetTime( timeSta );

                MPI_Bcast(SCFDG_comp_subspace_start_guess_.Data(),hamDG.NumStateTotal() * SCFDG_comp_subspace_N_solve_, 
                    MPI_DOUBLE, 0,  domain_.comm); 

                GetTime( timeEnd );
                statusOFS << " Done. ( " << (timeEnd - timeSta ) << " s.)";


              }    

              // Reset the filtering bounds using results of the Raleigh-Ritz step    
              b_low = eigval(ref_mat_2.n() - 1);
              a_L = eigval(0);

              MPI_Barrier(domain_.rowComm);
              MPI_Barrier(domain_.colComm);
              MPI_Barrier(domain_.comm);

              // Clean up BLACS
              if(cheby_scala_context >= 0) 
              {
                dgdft::scalapack::Cblacs_gridexit( cheby_scala_context );
              }

              if( single_proc_context >= 0)
              {
                dgdft::scalapack::Cblacs_gridexit( single_proc_context );
              }


            } // End of if(SCFDG_Cheby_use_ScaLAPACK_ == 0) ... else 

            // Display the eigenvalues 
            statusOFS << std::endl << " ------------------------------- ";
            GetTime( cheby_timeEnd );

            //statusOFS << std::endl << " Eigenvalues via Chebyshev filtering : " << std::endl;
            //statusOFS << eig_vals_Raleigh_Ritz << std::endl;

            statusOFS << std::endl << " This Chebyshev cycle took a total of " << (cheby_timeEnd - cheby_timeSta ) << " s.";
            statusOFS << std::endl << " ------------------------------- " << std::endl;

          } // End of loop over inner iteration repeats


        }
#endif

     void
        SCFDG::scfdg_GeneralChebyStep(Int MaxIter,
            Int filter_order )
        {
          Int mpirank, mpisize;
          MPI_Comm_rank( domain_.comm, &mpirank );
          MPI_Comm_size( domain_.comm, &mpisize );
          Int mpirankRow;  MPI_Comm_rank(domain_.rowComm, &mpirankRow);
          Int mpisizeRow;  MPI_Comm_size(domain_.rowComm, &mpisizeRow);
          Int mpirankCol;  MPI_Comm_rank(domain_.colComm, &mpirankCol);
          Int mpisizeCol;  MPI_Comm_size(domain_.colComm, &mpisizeCol);

          Real timeSta, timeEnd;
          Real extra_timeSta, extra_timeEnd;
          Real cheby_timeSta, cheby_timeEnd;

          HamiltonianDG&  hamDG = *hamDGPtr_;

          DblNumVec& eigval = hamDG.EigVal();

         // Step 0: Safeguard against the eigenvector containing extra keys
          {
            std::map<Index3, DblNumMat>::iterator cleaner_itr = hamDG.EigvecCoef().LocalMap().begin();
            while (cleaner_itr != hamDG.EigvecCoef().LocalMap().end())
            {
              if (cleaner_itr->first != my_cheby_eig_vec_key_)
              {
                std::map<Index3, DblNumMat>::iterator toErase = cleaner_itr;
                ++ cleaner_itr;
                hamDG.EigvecCoef().LocalMap().erase(toErase);
              }
              else
              {
                ++ cleaner_itr;
              }
            }
          }


          // Step 1: Obtain the upper bound and the Ritz values (for the lower bound)
          // using the Lanczos estimator

          DblNumVec Lanczos_Ritz_values;


          statusOFS << std::endl << " Estimating the spectral bounds ... ";
          GetTime( extra_timeSta );
          const int Num_Lanczos_Steps = 6;
          double b_up = scfdg_Cheby_Upper_bound_estimator(Lanczos_Ritz_values, Num_Lanczos_Steps);
          GetTime( extra_timeEnd );
          statusOFS << " Done. ( " << (extra_timeEnd - extra_timeSta ) << " s.)";

          // statusOFS << std::endl << "Lanczos-Ritz values : " << Lanczos_Ritz_values << std::endl ;

          double a_L = eigval[0];
          double b_low = eigval[hamDG.NumStateTotal() - 1];

         // Step 2: Main loop
          const Int Iter_Max = MaxIter;
          const Int Filter_Order = filter_order ;

          DblNumVec eig_vals_Raleigh_Ritz;
          Real one = 1.0;
          Real zero = 0.0;
          for(Int i = 1; i <= Iter_Max; i ++)
          {
            GetTime( cheby_timeSta );
            statusOFS << std::endl << std::endl << " ------------------------------- ";
            statusOFS << std::endl << " General CheFSI step for DGDFT cycle " << i << " of " << Iter_Max << " . ";

            // Filter the eigenvectors
            statusOFS << std::endl << std::endl << " Filtering the eigenvectors ... (Filter order = " << Filter_Order << ")";
            GetTime( timeSta );
            scfdg_Chebyshev_filter_scaled(Filter_Order, b_low, b_up, a_L);
            GetTime( timeEnd );
            statusOFS << std::endl << " Filtering completed. ( " << (timeEnd - timeSta ) << " s.)";


            // Subspace projected problems: Orthonormalization, Raleigh-Ritz and subspace rotation steps
            // This can be done serially or using ScaLAPACK
            if(SCFDG_Cheby_use_ScaLAPACK_ == 0)
            {
              // Do the subspace problem serially
              statusOFS << std::endl << std::endl << " Solving subspace problems serially ...";

              // Orthonormalize using Cholesky factorization
              statusOFS << std::endl << " Orthonormalizing filtered vectors ... ";
              GetTime( timeSta );



              DblNumMat &local_eigvec_mat = (hamDG.EigvecCoef().LocalMap().begin())->second;
              DblNumMat square_mat;
              DblNumMat temp_square_mat;

              Int width = local_eigvec_mat.n();
              Int height_local = local_eigvec_mat.m();

             square_mat.Resize(width, width);
              temp_square_mat.Resize(width, width);

              SetValue(temp_square_mat, 0.0);

              // Compute square_mat = X^T * X for Cholesky
#ifdef GPU //--------by lijl 20220526
              Real sta_jl, end_jl;
              cuDblNumMat cu_local_eigvec_mat(height_local, width);
              cu_local_eigvec_mat.CopyFrom(local_eigvec_mat);
              cuDblNumMat cu_temp_square_mat(width, width);
              cublas::Gemm( HIPBLAS_OP_T, HIPBLAS_OP_N, width, width, height_local,
                  &one, cu_local_eigvec_mat.Data(), height_local,
                  cu_local_eigvec_mat.Data(), height_local,
                  &zero, cu_temp_square_mat.Data(), width );
              cu_temp_square_mat.CopyTo(temp_square_mat);
#else
              blas::Gemm( 'T', 'N', width, width, height_local,
                  1.0, local_eigvec_mat.Data(), height_local,
                  local_eigvec_mat.Data(), height_local,
                  0.0, temp_square_mat.Data(), width );
#endif
              SetValue( square_mat, 0.0 );
              MPI_Allreduce( temp_square_mat.Data(), square_mat.Data(), width*width, MPI_DOUBLE, MPI_SUM, domain_.colComm );

#ifdef GPU
              cuDblNumMat cu_square_mat(width, width);
              cu_square_mat.CopyFrom(square_mat);
#endif
              // In the following, reduction happens on colComm but the result is broadcast to everyone
              // This can probably be band-parallelized

              // Make the Cholesky factorization call on proc 0
              if ( mpirank == 0) {
#if 0
                //MAGMA::Potrf( 'U', width, cu_square_mat.Data(), width );
                cusolver::Potrf( 'U', width, cu_square_mat.Data(), width );
                cu_square_mat.CopyTo(square_mat);
#else
                lapack::Potrf( 'U', width, square_mat.Data(), width );
#endif
              }
              // Send the Cholesky factor to every process
              MPI_Bcast(square_mat.Data(), width*width, MPI_DOUBLE, 0, domain_.comm);

              // Do a solve with the Cholesky factor : Band parallelization ??
              // X = X * U^{-1} is orthogonal, where U is the Cholesky factor
#ifdef GPU

              cu_square_mat.CopyFrom(square_mat);
              cublas::Trsm( HIPBLAS_SIDE_RIGHT, HIPBLAS_FILL_MODE_UPPER, HIPBLAS_OP_N, HIPBLAS_DIAG_NON_UNIT, height_local, width, &one, cu_square_mat.Data(), width,
                  cu_local_eigvec_mat.Data(), height_local );
              cu_local_eigvec_mat.CopyTo(local_eigvec_mat);
#else
              blas::Trsm( 'R', 'U', 'N', 'N', height_local, width, 1.0, square_mat.Data(), width,
                  local_eigvec_mat.Data(), height_local );
#endif


              GetTime( timeEnd );
              statusOFS << std::endl << " Orthonormalization completed ( " << (timeEnd - timeSta ) << " s.)";

              // Raleigh-Ritz step: This part is non-scalable and needs to be fixed
              // !!!!!!!!!!!!!!!!!!!!!!!!!!!!
              statusOFS << std::endl << std::endl << " Raleigh-Ritz step ... ";
              GetTime( timeSta );

              // Compute H * X
              DistVec<Index3, DblNumMat, ElemPrtn>  result_mat;
              scfdg_Hamiltonian_times_eigenvectors(result_mat);
              DblNumMat &local_result_mat = (result_mat.LocalMap().begin())->second;

              SetValue(temp_square_mat, 0.0);

              // Compute square_mat = X^T * HX
#ifdef GPU

              cuDblNumMat cu_local_result_mat(height_local, width);
              cu_local_result_mat.CopyFrom(local_result_mat);
              cublas::Gemm( HIPBLAS_OP_T, HIPBLAS_OP_N, width, width, height_local,
                  &one, cu_local_eigvec_mat.Data(), height_local,
                  cu_local_result_mat.Data(), height_local,
                  &zero, cu_temp_square_mat.Data(), width );
              cu_temp_square_mat.CopyTo(temp_square_mat);


#else
              blas::Gemm( 'T', 'N', width, width, height_local,
                  1.0, local_eigvec_mat.Data(), height_local,
                  local_result_mat.Data(), height_local,
                  0.0, temp_square_mat.Data(), width );
#endif

              SetValue( square_mat, 0.0 );
              MPI_Allreduce( temp_square_mat.Data(), square_mat.Data(), width*width, MPI_DOUBLE, MPI_SUM, domain_.colComm );


              eig_vals_Raleigh_Ritz.Resize(width);
              SetValue(eig_vals_Raleigh_Ritz, 0.0);

              if ( mpirank == 0 ) {
#if 0 //def GPU
              cu_square_mat.CopyFrom(square_mat);
              cuDblNumVec cu_eig_vals_Raleigh_Ritz(width);
              cu_eig_vals_Raleigh_Ritz.CopyFrom(eig_vals_Raleigh_Ritz);
              MAGMA::Syevd( 'V', 'U', width, cu_square_mat.Data(), width, eig_vals_Raleigh_Ritz.Data() );
              cu_square_mat.CopyTo(square_mat);
              //cu_eig_vals_Raleigh_Ritz.CopyTo(eig_vals_Raleigh_Ritz);
#else
                lapack::Syevd( 'V', 'U', width, square_mat.Data(), width, eig_vals_Raleigh_Ritz.Data() );
#endif
              }

              // ~~ Send the results to every process
              MPI_Bcast(square_mat.Data(), width*width, MPI_DOUBLE, 0, domain_.comm); // Eigen-vectors
              MPI_Bcast(eig_vals_Raleigh_Ritz.Data(), width, MPI_DOUBLE, 0,  domain_.comm); // Eigen-values

              GetTime( timeEnd );
              statusOFS << std::endl << " Raleigh-Ritz step completed ( " << (timeEnd - timeSta ) << " s.)";

              // This is for use with the Complementary Subspace strategy in subsequent steps
              if(SCFDG_use_comp_subspace_ == 1)
              {
                GetTime( timeSta );

                statusOFS << std::endl << std::endl << " Copying top states for serial CS strategy ... ";
                SCFDG_comp_subspace_start_guess_.Resize(width, SCFDG_comp_subspace_N_solve_);

                for(Int copy_iter = 0; copy_iter < SCFDG_comp_subspace_N_solve_; copy_iter ++)
                {

                  blas::Copy( width, square_mat.VecData(width - 1 - copy_iter), 1,
                      SCFDG_comp_subspace_start_guess_.VecData(copy_iter), 1 );

                  // lapack::Lacpy( 'A', width, 1, square_mat.VecData(width - 1 - copy_iter), width,
                  //              SCFDG_comp_subspace_start_guess_.VecData(copy_iter), width );
                }

                GetTime( timeEnd );
                statusOFS << " Done. ( " << (timeEnd - timeSta ) << " s.)";
              }


              // Subspace rotation step X <- X * Q: This part is non-scalable and needs to be fixed
              // !!!!!!!!!!!!!!!!!!!!!!!!!!!!
              statusOFS << std::endl << std::endl << " Subspace rotation step ... ";
              GetTime( timeSta );

              // ~~ So copy X to HX
              lapack::Lacpy( 'A', height_local, width, local_eigvec_mat.Data(),  height_local, local_result_mat.Data(), height_local );

              // ~~ Gemm: X <-- HX (= X) * Q
#ifdef GPU

             cu_local_result_mat.CopyFrom(local_result_mat);
             cu_square_mat.CopyFrom(square_mat);
             cublas::Gemm( HIPBLAS_OP_N, HIPBLAS_OP_N, height_local, width, width, &one, cu_local_result_mat.Data(),
                           height_local, cu_square_mat.Data(), width, &zero, cu_local_eigvec_mat.Data(), height_local );
             cu_local_eigvec_mat.CopyTo(local_eigvec_mat);
#else
              blas::Gemm( 'N', 'N', height_local, width, width, 1.0, local_result_mat.Data(),
                  height_local, square_mat.Data(), width, 0.0, local_eigvec_mat.Data(), height_local );
#endif
              GetTime( timeEnd );
              statusOFS << std::endl << " Subspace rotation step completed ( " << (timeEnd - timeSta ) << " s.)";

              // Reset the filtering bounds using results of the Raleigh-Ritz step
              b_low = eig_vals_Raleigh_Ritz(width - 1);
              a_L = eig_vals_Raleigh_Ritz(0);

              // Fill up the eigen-values
              DblNumVec& eigval = hamDG.EigVal();
              eigval.Resize( hamDG.NumStateTotal() );

              for(Int i = 0; i < hamDG.NumStateTotal(); i ++)
                eigval[i] =  eig_vals_Raleigh_Ritz[i];



            } // End of if(SCFDG_Cheby_use_ScaLAPACK_ == 0)
            else
            {
             // Do the subspace problems using ScaLAPACK
              statusOFS << std::endl << std::endl << " Solving subspace problems using ScaLAPACK ...";

              statusOFS << std::endl << " Setting up BLACS and ScaLAPACK Process Grid ...";
              GetTime( timeSta );

              double detail_timeSta, detail_timeEnd;

              // Basic ScaLAPACK setup steps
              //Number of ScaLAPACK processors equal to number of DG elements
              const int num_cheby_scala_procs = mpisizeCol;

              // Figure out the process grid dimensions
              int temp_factor = int(sqrt(double(num_cheby_scala_procs)));
              while(num_cheby_scala_procs % temp_factor != 0 )
                ++temp_factor;

              // temp_factor now contains the process grid height
              const int cheby_scala_num_rows = temp_factor;
              const int cheby_scala_num_cols = num_cheby_scala_procs / temp_factor;



              // Set up the ScaLAPACK context
              IntNumVec cheby_scala_pmap(num_cheby_scala_procs);

              // Use the first processor from every DG-element
              for ( Int pmap_iter = 0; pmap_iter < num_cheby_scala_procs; pmap_iter++ )
                cheby_scala_pmap[pmap_iter] = pmap_iter * mpisizeRow;

              // Set up BLACS for subsequent ScaLAPACK operations
              Int cheby_scala_context = -2;
              dgdft::scalapack::Cblacs_get( 0, 0, &cheby_scala_context );
              dgdft::scalapack::Cblacs_gridmap(&cheby_scala_context, &cheby_scala_pmap[0], cheby_scala_num_rows, cheby_scala_num_rows, cheby_scala_num_cols);

              // Figure out my ScaLAPACK information
              int dummy_np_row, dummy_np_col;
              int my_cheby_scala_proc_row, my_cheby_scala_proc_col;

              dgdft::scalapack::Cblacs_gridinfo(cheby_scala_context, &dummy_np_row, &dummy_np_col, &my_cheby_scala_proc_row, &my_cheby_scala_proc_col);


              GetTime( timeEnd );
              statusOFS << " Done. ( " << (timeEnd - timeSta ) << " s.)";

              statusOFS << std::endl << " ScaLAPACK will use " << num_cheby_scala_procs << " processes.";
              statusOFS << std::endl << " ScaLAPACK process grid = " << cheby_scala_num_rows << " * " << cheby_scala_num_cols << " ."  << std::endl;

              // Eigenvcetors in ScaLAPACK format : this will be used multiple times
              dgdft::scalapack::ScaLAPACKMatrix<Real>  cheby_scala_eigvecs_X; // Declared here for scope, empty constructor invoked

              if(cheby_scala_context >= 0)
              {
                // Now setup the ScaLAPACK matrix
                statusOFS << std::endl << " Orthonormalization step : ";
                statusOFS << std::endl << " Distributed vector to ScaLAPACK format conversion ... ";
                GetTime( timeSta );

                // The dimensions should be  hamDG.NumBasisTotal() * hamDG.NumStateTotal()
                // But this is not verified here as far as the distributed vector is concerned
                dgdft::scalapack::Descriptor cheby_eigvec_desc( hamDG.NumBasisTotal(), hamDG.NumStateTotal(),
                    scaBlockSize_, scaBlockSize_,
                    0, 0,
                    cheby_scala_context);



                // Make the conversion call
                scfdg_Cheby_convert_eigvec_distmat_to_ScaLAPACK(hamDG.EigvecCoef(),
                    domain_.colComm,
                    cheby_eigvec_desc,
                    cheby_scala_eigvecs_X);


                //         //Older version of conversion call
                //         // Store the important ScaLAPACK information
                //         std::vector<int> my_cheby_scala_info;
                //         my_cheby_scala_info.resize(4,0);
                //         my_cheby_scala_info[0] = cheby_scala_num_rows;
                //         my_cheby_scala_info[1] = cheby_scala_num_cols;
                //         my_cheby_scala_info[2] = my_cheby_scala_proc_row;
                //         my_cheby_scala_info[3] = my_cheby_scala_proc_col;
                //

                //         scfdg_Cheby_convert_eigvec_distmat_to_ScaLAPACK_old(hamDG.EigvecCoef(),
                //                                     my_cheby_scala_info,
                //                                     cheby_eigvec_desc,
                //                                     cheby_scala_eigvecs_X);
                //


                GetTime( timeEnd );
                statusOFS << " Done. ( " << (timeEnd - timeSta ) << " s.)";


                statusOFS << std::endl << " Orthonormalizing filtered vectors ... ";
                GetTime( timeSta );

                GetTime( detail_timeSta);

                // Compute C = X^T * X
                dgdft::scalapack::Descriptor cheby_chol_desc( hamDG.NumStateTotal(), hamDG.NumStateTotal(),
                    scaBlockSize_, scaBlockSize_,
                    0, 0,
                    cheby_scala_context);

                dgdft::scalapack::ScaLAPACKMatrix<Real>  cheby_scala_chol_mat;
                cheby_scala_chol_mat.SetDescriptor(cheby_chol_desc);

                dgdft::scalapack::Gemm( 'T', 'N',
                    hamDG.NumStateTotal(), hamDG.NumStateTotal(), hamDG.NumBasisTotal(),
                    1.0,
                    cheby_scala_eigvecs_X.Data(), I_ONE, I_ONE, cheby_scala_eigvecs_X.Desc().Values(),
                    cheby_scala_eigvecs_X.Data(), I_ONE, I_ONE, cheby_scala_eigvecs_X.Desc().Values(),
                    0.0,
                    cheby_scala_chol_mat.Data(), I_ONE, I_ONE, cheby_scala_chol_mat.Desc().Values(),
                    cheby_scala_context);

                GetTime( detail_timeEnd);
                statusOFS << std::endl << " Overlap matrix computed in : " << (detail_timeEnd - detail_timeSta) << " s.";


                GetTime( detail_timeSta);
                // Compute V = Chol(C)
                dgdft::scalapack::Potrf( 'U', cheby_scala_chol_mat);

                GetTime( detail_timeEnd);
                statusOFS << std::endl << " Cholesky factorization computed in : " << (detail_timeEnd - detail_timeSta) << " s.";


                GetTime( detail_timeSta);
                // Compute  X = X * V^{-1}
                dgdft::scalapack::Trsm( 'R', 'U', 'N', 'N', 1.0,
                    cheby_scala_chol_mat,
                    cheby_scala_eigvecs_X );

                GetTime( detail_timeEnd);
                statusOFS << std::endl << " TRSM computed in : " << (detail_timeEnd - detail_timeSta) << " s.";


                GetTime( timeEnd );
                statusOFS << std::endl << " Orthonormalization steps completed ( " << (timeEnd - timeSta ) << " s.)" << std::endl;

                statusOFS << std::endl << " ScaLAPACK to Distributed vector format conversion ... ";
                GetTime( timeSta );

                // Now convert this back to DG-distributed matrix format
                ScaMatToDistNumMat(cheby_scala_eigvecs_X ,
                    elemPrtn_,
                    hamDG.EigvecCoef(),
                    hamDG.ElemBasisIdx(),
                    domain_.colComm,
                    hamDG.NumStateTotal() );

                GetTime( timeEnd );
                statusOFS << " Done. ( " << (timeEnd - timeSta ) << " s.)";


              } // End of  if(cheby_scala_context >= 0)
              else
              {
                statusOFS << std::endl << " Waiting for ScaLAPACK solution of subspace problems ...";
              }

              // Communicate the orthonormalized eigenvectors (to other intra-element processors)
              statusOFS << std::endl << " Communicating orthonormalized filtered vectors ... ";
              GetTime( timeSta );

              DblNumMat &ref_mat_1 =  (hamDG.EigvecCoef().LocalMap().begin())->second;
              MPI_Bcast(ref_mat_1.Data(), (ref_mat_1.m() * ref_mat_1.n()), MPI_DOUBLE, 0, domain_.rowComm);

              GetTime( timeEnd );
              statusOFS << " Done. ( " << (timeEnd - timeSta ) << " s.)" << std::endl;

              // Set up space for eigenvalues in the Hamiltonian object for results of Raleigh-Ritz step
              DblNumVec& eigval = hamDG.EigVal();
              eigval.Resize( hamDG.NumStateTotal() );

              // Compute H * X
              statusOFS << std::endl << " Computing H * X for filtered orthonormal vectors : ";
              GetTime( timeSta );

              DistVec<Index3, DblNumMat, ElemPrtn>  result_mat;
              scfdg_Hamiltonian_times_eigenvectors(result_mat);

              GetTime( timeEnd );
              statusOFS << std::endl << " H * X for filtered orthonormal vectors computed . ( " << (timeEnd - timeSta ) << " s.)";


              // Set up a single process ScaLAPACK matrix for use with parallel CS strategy
              // This is for copying the relevant portion of scaZ
              Int single_proc_context = -1;
              scalapack::Descriptor single_proc_desc;
              scalapack::ScaLAPACKMatrix<Real>  single_proc_scala_mat;


              if(SCFDG_use_comp_subspace_ == 1 && SCFDG_comp_subspace_N_solve_ != 0)
              {

                // Reserve the serial space : this will actually contain the vectors in the reversed order
                SCFDG_comp_subspace_start_guess_.Resize( hamDG.NumStateTotal(), SCFDG_comp_subspace_N_solve_);

                Int single_proc_pmap[1];
                single_proc_pmap[0] = 0; // Just using proc. 0 for the job.

                // Set up BLACS for for the single proc context

                dgdft::scalapack::Cblacs_get( 0, 0, &single_proc_context );
                dgdft::scalapack::Cblacs_gridmap(&single_proc_context, &single_proc_pmap[0], 1, 1, 1);

                if( single_proc_context >= 0)
                {
                  // For safety, make sure this is MPI Rank zero : throw an error otherwise
                  // Fix this in the future ?
                  if(mpirank != 0)
                  {
                    statusOFS << std::endl << std::endl << "  Error !! BLACS rank 0 does not match MPI rank 0 "
                      << " Aborting ... " << std::endl << std::endl;
                    MPI_Abort(domain_.comm, 0);
                  }

                  single_proc_desc.Init( hamDG.NumStateTotal(), SCFDG_comp_subspace_N_solve_,
                      scaBlockSize_, scaBlockSize_,
                      0, 0,  single_proc_context );

                  single_proc_scala_mat.SetDescriptor( single_proc_desc );

                }
              }


              // Raleigh-Ritz step
              if(cheby_scala_context >= 0)
              {
                statusOFS << std::endl << std::endl << " Raleigh - Ritz step : ";

                // Convert HX to ScaLAPACK format
                dgdft::scalapack::ScaLAPACKMatrix<Real>  cheby_scala_HX;
                dgdft::scalapack::Descriptor cheby_HX_desc( hamDG.NumBasisTotal(), hamDG.NumStateTotal(),
                    scaBlockSize_, scaBlockSize_,
                    0, 0,
                    cheby_scala_context);


                statusOFS << std::endl << " Distributed vector to ScaLAPACK format conversion ... ";
                GetTime( timeSta );

                // Make the conversion call
                scfdg_Cheby_convert_eigvec_distmat_to_ScaLAPACK(result_mat,
                    domain_.colComm,
                    cheby_HX_desc,
                    cheby_scala_HX);


                GetTime( timeEnd );
                statusOFS << " Done. ( " << (timeEnd - timeSta ) << " s.)";

                statusOFS << std::endl << " Solving the subspace problem ... ";
                GetTime( timeSta );

                dgdft::scalapack::Descriptor cheby_XTHX_desc( hamDG.NumStateTotal(), hamDG.NumStateTotal(),
                    scaBlockSize_, scaBlockSize_,
                    0, 0,
                    cheby_scala_context);
                dgdft::scalapack::ScaLAPACKMatrix<Real>  cheby_scala_XTHX_mat;
                cheby_scala_XTHX_mat.SetDescriptor(cheby_XTHX_desc);


                GetTime( detail_timeSta);

                dgdft::scalapack::Gemm( 'T', 'N',
                    hamDG.NumStateTotal(), hamDG.NumStateTotal(), hamDG.NumBasisTotal(),
                    1.0,
                    cheby_scala_eigvecs_X.Data(), I_ONE, I_ONE, cheby_scala_eigvecs_X.Desc().Values(),
                    cheby_scala_HX.Data(), I_ONE, I_ONE, cheby_scala_HX.Desc().Values(),
                    0.0,
                    cheby_scala_XTHX_mat.Data(), I_ONE, I_ONE, cheby_scala_XTHX_mat.Desc().Values(),
                    cheby_scala_context);


                GetTime( detail_timeEnd);
                statusOFS << std::endl << " X^T(HX) computed in : " << (detail_timeEnd - detail_timeSta) << " s.";

                scalapack::ScaLAPACKMatrix<Real>  scaZ;

                std::vector<Real> eigen_values;

                GetTime( detail_timeSta);

                // Eigenvalue probem solution call
                scalapack::Syevd('U', cheby_scala_XTHX_mat, eigen_values, scaZ);

                // Copy the eigenvalues to the Hamiltonian object
                for( Int i = 0; i < hamDG.NumStateTotal(); i++ )
                  eigval[i] = eigen_values[i];

                GetTime( detail_timeEnd);
                statusOFS << std::endl << " Eigenvalue problem solved in : " << (detail_timeEnd - detail_timeSta) << " s.";


                // This is for use with the Complementary Subspace strategy in subsequent steps
                if(SCFDG_use_comp_subspace_ == 1 && SCFDG_comp_subspace_N_solve_ != 0)
                {
                  GetTime( detail_timeSta);
                  statusOFS << std::endl << std::endl << " Distributing and copying top states for parallel CS strategy ... ";


                 const Int M_copy_ = hamDG.NumStateTotal();
                  const Int N_copy_ = SCFDG_comp_subspace_N_solve_;
                  const Int copy_src_col = M_copy_ - N_copy_ + 1;

                  // Note that this is being called inside cheby_scala_context >= 0
                  SCALAPACK(pdgemr2d)(&M_copy_, &N_copy_,
                      scaZ.Data(), &I_ONE, &copy_src_col,
                      scaZ.Desc().Values(),
                      single_proc_scala_mat.Data(), &I_ONE, &I_ONE,
                      single_proc_scala_mat.Desc().Values(),
                      &cheby_scala_context);


                  // Copy the data from single_proc_scala_mat to SCFDG_comp_subspace_start_guess_
                  if( single_proc_context >= 0)
                  {
                    double *src_ptr, *dest_ptr;

                    //                 statusOFS << std::endl << std::endl
                    //                            << " Ht = " << single_proc_scala_mat.Height()
                    //                            << " Width = " << single_proc_scala_mat.Width()
                    //                            << " loc. Ht = " << single_proc_scala_mat.LocalHeight()
                    //                            << " loc. Width = " << single_proc_scala_mat.LocalWidth()
                    //                            << " loc. LD = " << single_proc_scala_mat.LocalLDim();
                    //                 statusOFS << std::endl << std::endl;
                    //
                    // Do this in the reverse order
                    for(Int copy_iter = 0; copy_iter < SCFDG_comp_subspace_N_solve_; copy_iter ++)
                    {
                      src_ptr = single_proc_scala_mat.Data() + (SCFDG_comp_subspace_N_solve_ - copy_iter - 1) * single_proc_scala_mat.LocalLDim();
                      dest_ptr =  SCFDG_comp_subspace_start_guess_.VecData(copy_iter);

                      blas::Copy( M_copy_, src_ptr, 1, dest_ptr, 1 );
                    }


                  }


                  GetTime( detail_timeEnd);
                  statusOFS << "Done. (" << (detail_timeEnd - detail_timeSta) << " s.)";

                } // end of if(SCFDG_use_comp_subspace_ == 1)



                // Subspace rotation step : X <- X * Q
                statusOFS << std::endl << std::endl << " Subspace rotation step ...  ";
                GetTime( detail_timeSta);

                // To save memory, copy X to HX
                blas::Copy((cheby_scala_eigvecs_X.LocalHeight() * cheby_scala_eigvecs_X.LocalWidth()),
                    cheby_scala_eigvecs_X.Data(),
                    1,
                    cheby_scala_HX.Data(),
                    1);

                // Now perform X <- HX (=X) * Q (=scaZ)
                dgdft::scalapack::Gemm( 'N', 'N',
                    hamDG.NumBasisTotal(), hamDG.NumStateTotal(), hamDG.NumStateTotal(),
                    1.0,
                    cheby_scala_HX.Data(), I_ONE, I_ONE, cheby_scala_HX.Desc().Values(),
                    scaZ.Data(), I_ONE, I_ONE, scaZ.Desc().Values(),
                    0.0,
                    cheby_scala_eigvecs_X.Data(), I_ONE, I_ONE, cheby_scala_eigvecs_X.Desc().Values(),
                    cheby_scala_context);

                GetTime( detail_timeEnd);
                statusOFS << " Done. (" << (detail_timeEnd - detail_timeSta) << " s.)";





                GetTime( timeEnd );
                statusOFS << std::endl << " All subspace problem steps completed . ( " << (timeEnd - timeSta ) << " s.)" << std::endl ;

                // Convert the eigenvectors back to distributed vector format
                statusOFS << std::endl << " ScaLAPACK to Distributed vector format conversion ... ";
                GetTime( timeSta );

                // Now convert this back to DG-distributed matrix format
                ScaMatToDistNumMat(cheby_scala_eigvecs_X ,
                    elemPrtn_,
                    hamDG.EigvecCoef(),
                    hamDG.ElemBasisIdx(),
                    domain_.colComm,
                    hamDG.NumStateTotal() );

                GetTime( timeEnd );
                statusOFS << " Done. ( " << (timeEnd - timeSta ) << " s.)";


              } // End of if(cheby_scala_context >= 0)
              else
              {
                statusOFS << std::endl << std::endl << " Waiting for ScaLAPACK solution of subspace problems ...";
              }





              // Communicate the final eigenvectors (to other intra-element processors)
              statusOFS << std::endl << " Communicating eigenvalues and eigenvectors ... ";
              GetTime( timeSta );

              DblNumMat &ref_mat_2 =  (hamDG.EigvecCoef().LocalMap().begin())->second;
              MPI_Bcast(ref_mat_2.Data(), (ref_mat_2.m() * ref_mat_2.n()), MPI_DOUBLE, 0, domain_.rowComm);

              // Communicate the final eigenvalues (to other intra-element processors)
              MPI_Bcast(eigval.Data(), ref_mat_2.n(), MPI_DOUBLE, 0,  domain_.rowComm); // Eigen-values


              GetTime( timeEnd );
              statusOFS << " Done. ( " << (timeEnd - timeSta ) << " s.)";


              // Broadcast the guess vectors for the CS strategy from proc 0
              if(SCFDG_use_comp_subspace_ == 1 && SCFDG_comp_subspace_N_solve_ != 0)
              {
                statusOFS << std::endl << std::endl << " Broadcasting guess vectors for parallel CS strategy ... ";
                GetTime( timeSta );

                MPI_Bcast(SCFDG_comp_subspace_start_guess_.Data(),hamDG.NumStateTotal() * SCFDG_comp_subspace_N_solve_,
                    MPI_DOUBLE, 0,  domain_.comm);

                GetTime( timeEnd );
                statusOFS << " Done. ( " << (timeEnd - timeSta ) << " s.)";


              }

              // Reset the filtering bounds using results of the Raleigh-Ritz step
              b_low = eigval(ref_mat_2.n() - 1);
              a_L = eigval(0);

              MPI_Barrier(domain_.rowComm);
              MPI_Barrier(domain_.colComm);
              MPI_Barrier(domain_.comm);

              // Clean up BLACS
              if(cheby_scala_context >= 0)
              {
                dgdft::scalapack::Cblacs_gridexit( cheby_scala_context );
              }

              if( single_proc_context >= 0)
              {
                dgdft::scalapack::Cblacs_gridexit( single_proc_context );
              }


            } // End of if(SCFDG_Cheby_use_ScaLAPACK_ == 0) ... else

            // Display the eigenvalues
            statusOFS << std::endl << " ------------------------------- ";
            GetTime( cheby_timeEnd );

            //statusOFS << std::endl << " Eigenvalues via Chebyshev filtering : " << std::endl;
            //statusOFS << eig_vals_Raleigh_Ritz << std::endl;

            statusOFS << std::endl << " This Chebyshev cycle took a total of " << (cheby_timeEnd - cheby_timeSta ) << " s.";
            statusOFS << std::endl << " ------------------------------- " << std::endl;

          } // End of loop over inner iteration repeats


        }

      // **###**    
      /// @brief Routines related to Chebyshev polynomial filtered 
      /// complementary subspace iteration strategy in DGDFT
      void SCFDG::scfdg_complementary_subspace_serial(Int filter_order )
      {
        statusOFS << std::endl << " ----------------------------------------------------------" ; 
        statusOFS << std::endl << " Complementary Subspace Strategy (Serial Subspace version): " << std::endl ; 


        Int mpirank, mpisize;
        MPI_Comm_rank( domain_.comm, &mpirank );
        MPI_Comm_size( domain_.comm, &mpisize );
        Int mpirankRow;  MPI_Comm_rank(domain_.rowComm, &mpirankRow);
        Int mpisizeRow;  MPI_Comm_size(domain_.rowComm, &mpisizeRow);
        Int mpirankCol;  MPI_Comm_rank(domain_.colComm, &mpirankCol);
        Int mpisizeCol;  MPI_Comm_size(domain_.colComm, &mpisizeCol);

        Real timeSta, timeEnd;
        Real extra_timeSta, extra_timeEnd;
        Real cheby_timeSta, cheby_timeEnd;

        HamiltonianDG&  hamDG = *hamDGPtr_;

        DblNumVec& eigval = hamDG.EigVal(); 

        // Step 0: Safeguard against the eigenvector containing extra keys
        {
          std::map<Index3, DblNumMat>::iterator cleaner_itr = hamDG.EigvecCoef().LocalMap().begin();
          while (cleaner_itr != hamDG.EigvecCoef().LocalMap().end()) 
          {
            if (cleaner_itr->first != my_cheby_eig_vec_key_) 
            {
              std::map<Index3, DblNumMat>::iterator toErase = cleaner_itr;
              ++ cleaner_itr;
              hamDG.EigvecCoef().LocalMap().erase(toErase);
            } 
            else 
            {
              ++ cleaner_itr;
            }
          }
        }
        
        

        // Step 1: Obtain the upper bound and the Ritz values (for the lower bound)
        // using the Lanczos estimator

        DblNumVec Lanczos_Ritz_values;


        statusOFS << std::endl << " Estimating the spectral bounds ... "; 
        GetTime( extra_timeSta );
        const int Num_Lanczos_Steps = 6;
        double b_up = scfdg_Cheby_Upper_bound_estimator(Lanczos_Ritz_values, Num_Lanczos_Steps);  
        GetTime( extra_timeEnd );
        statusOFS << " Done. ( " << (extra_timeEnd - extra_timeSta ) << " s.)";

        double a_L;
        double b_low;

        if(SCFDG_comp_subspace_engaged_ == 1)
        {
          a_L = SCFDG_comp_subspace_saved_a_L_;
          b_low = SCFDG_comp_subspace_top_eigvals_[0];
        }
        else
        {
          // First time we are doing this : use earlier ScaLAPACK results 
          a_L = eigval[0];
          SCFDG_comp_subspace_saved_a_L_ = a_L;

          b_low = eigval[hamDG.NumStateTotal() - 1];
        }


        // Step 2: Perform filtering
        const Int Filter_Order = filter_order ;




        GetTime( cheby_timeSta );
        // Filter the subspace
        statusOFS << std::endl << std::endl << " Filtering the subspace ... (Filter order = " << Filter_Order << ")";
        GetTime( timeSta );
        scfdg_Chebyshev_filter_scaled(Filter_Order, b_low, b_up, a_L);
        GetTime( timeEnd );
        statusOFS << std::endl << " Filtering completed. ( " << (timeEnd - timeSta ) << " s.)";


        // Step 3: Perform subspace projected problems
        // Subspace problems serially done here

        statusOFS << std::endl << std::endl << " Solving subspace problems serially ...";
	
	
	// Some diagnostic info
	statusOFS << std::endl;
        statusOFS << std::endl << " Note : Outer subspace problem dimension = " << hamDG.NumStateTotal() << " * " << hamDG.NumStateTotal();
        statusOFS << std::endl << "        Inner subspace problem dimension = " << SCFDG_comp_subspace_N_solve_ << " * " << SCFDG_comp_subspace_N_solve_;
	statusOFS << std::endl << "        Matrix C has dimension = " << hamDG.NumStateTotal() << " * " << SCFDG_comp_subspace_N_solve_;
	statusOFS << std::endl << "        Matrix C occupies " << (double(hamDG.NumStateTotal() *  SCFDG_comp_subspace_N_solve_ * 8) / double(1048576)) << " MBs per process.";

        statusOFS << std::endl ;

        // Orthonormalize using Cholesky factorization  
        statusOFS << std::endl << " Orthonormalizing filtered vectors ... ";
        GetTime( timeSta );

        DblNumMat &local_eigvec_mat = (hamDG.EigvecCoef().LocalMap().begin())->second;
        DblNumMat square_mat;
        DblNumMat temp_square_mat;

        Int width = local_eigvec_mat.n();
        Int height_local = local_eigvec_mat.m();

        square_mat.Resize(width, width);
        temp_square_mat.Resize(width, width);

        SetValue(temp_square_mat, 0.0);

        // Compute square_mat = X^T * X for Cholesky    
        blas::Gemm( 'T', 'N', width, width, height_local, 
            1.0, local_eigvec_mat.Data(), height_local,
            local_eigvec_mat.Data(), height_local, 
            0.0, temp_square_mat.Data(), width );

        SetValue( square_mat, 0.0 );
        MPI_Allreduce( temp_square_mat.Data(), square_mat.Data(), width*width, MPI_DOUBLE, MPI_SUM, domain_.colComm );

        // In the following, reduction happens on colComm but the result is broadcast to everyone

        // Make the Cholesky factorization call on proc 0
        if ( mpirank == 0) {
          lapack::Potrf( 'U', width, square_mat.Data(), width );
        }
        // Send the Cholesky factor to every process
        MPI_Bcast(square_mat.Data(), width*width, MPI_DOUBLE, 0, domain_.comm);

        // Do a solve with the Cholesky factor : Band parallelization ??
        // X = X * U^{-1} is orthogonal, where U is the Cholesky factor
        blas::Trsm( 'R', 'U', 'N', 'N', height_local, width, 1.0, square_mat.Data(), width, 
            local_eigvec_mat.Data(), height_local );


        GetTime( timeEnd );
        statusOFS << std::endl << " Orthonormalization completed ( " << (timeEnd - timeSta ) << " s.)";

        // Alternate to Raleigh-Ritz step
        // !!!!!!!!!!!!!!!!!!!!!!!!!!!!
        statusOFS << std::endl << std::endl << " Performing alternate to Raleigh-Ritz step ... ";
        GetTime( timeSta );

        // Compute H * X
        DistVec<Index3, DblNumMat, ElemPrtn>  result_mat;
        scfdg_Hamiltonian_times_eigenvectors(result_mat);
        DblNumMat &local_result_mat = (result_mat.LocalMap().begin())->second;

        SetValue(temp_square_mat, 0.0);

        // Compute square_mat = X^T * HX 
        blas::Gemm( 'T', 'N', width, width, height_local, 
            1.0, local_eigvec_mat.Data(), height_local,
            local_result_mat.Data(), height_local, 
            0.0, temp_square_mat.Data(), width );

        SetValue( square_mat, 0.0 );
        MPI_Allreduce( temp_square_mat.Data(), square_mat.Data(), width*width, MPI_DOUBLE, MPI_SUM, domain_.colComm );

        DblNumMat temp_Hmat(square_mat);

        // Space for top few eigenpairs of projected Hamiltonian
        // Note that SCFDG_comp_subspace_start_guess_ should contain the starting guess already
        // This is either from the earlier ScaLAPACK results (when SCFDG_comp_subspace_engaged_ = 0)
        // or from the previous LOBPCG results which are copied

        DblNumMat temp_Xmat;
        temp_Xmat.Resize(width, SCFDG_comp_subspace_N_solve_);
        lapack::Lacpy( 'A', width, SCFDG_comp_subspace_N_solve_, SCFDG_comp_subspace_start_guess_.Data(), width, 
            temp_Xmat.Data(), width );


        // Space for top few eigenvalues of projected Hamiltonian
        DblNumVec temp_eig_vals_Xmat(SCFDG_comp_subspace_N_solve_);
        SetValue( temp_eig_vals_Xmat, 0.0 );


        if(Hmat_top_states_use_Cheby_ == 0)
        {  

          // Use serial LOBPCG to get the top states  
          GetTime(extra_timeSta);


          LOBPCG_Hmat_top_serial(temp_Hmat,
              temp_Xmat,
              temp_eig_vals_Xmat,
              SCFDG_comp_subspace_LOBPCG_iter_, SCFDG_comp_subspace_LOBPCG_tol_); // The tolerance should be dynamic probably


          GetTime(extra_timeEnd);

          statusOFS << std::endl << " Serial LOBPCG completed on " <<  SCFDG_comp_subspace_N_solve_ 
            << " top states ( " << (extra_timeEnd - extra_timeSta ) << " s.)";

        }
        else
        {

          // XXXXXXXXXXXXXXXXXXXXXX
          // Use CheFSI for top states here : Use -H for the matrix

          // Fix the filter bounds
          if(SCFDG_comp_subspace_engaged_ != 1)
          {
            SCFDG_comp_subspace_inner_CheFSI_a_L_ = - eigval[hamDG.NumStateTotal() - 1];

            int state_ind = hamDG.NumStateTotal() - SCFDG_comp_subspace_N_solve_;        
            SCFDG_comp_subspace_inner_CheFSI_lower_bound_ = -0.5 * (eigval[state_ind] + eigval[state_ind - 1]);

            SCFDG_comp_subspace_inner_CheFSI_upper_bound_ = find_comp_subspace_UB_serial(temp_Hmat);

            Hmat_top_states_Cheby_delta_fudge_ = 0.5 * (eigval[state_ind] - eigval[state_ind - 1]);

            statusOFS << std::endl << " Going into inner CheFSI routine for top states ... ";
            statusOFS << std::endl << "   Lower bound = -(average of eigenvalues  " << eigval[state_ind] 
              << " and " <<  eigval[state_ind - 1] << ") = " << SCFDG_comp_subspace_inner_CheFSI_lower_bound_
              << std::endl << "   Lanczos upper bound = " << SCFDG_comp_subspace_inner_CheFSI_upper_bound_ ;

          }
          else
          {
            SCFDG_comp_subspace_inner_CheFSI_lower_bound_ = - (SCFDG_comp_subspace_top_eigvals_[SCFDG_comp_subspace_N_solve_ - 1] - Hmat_top_states_Cheby_delta_fudge_);

            SCFDG_comp_subspace_inner_CheFSI_upper_bound_ = find_comp_subspace_UB_serial(temp_Hmat);

            statusOFS << std::endl << " Going into inner CheFSI routine for top states ... ";
            statusOFS << std::endl << "   Lower bound eigenvalue = " << SCFDG_comp_subspace_top_eigvals_[SCFDG_comp_subspace_N_solve_ - 1] 
              << std::endl << "   delta_fudge = " << Hmat_top_states_Cheby_delta_fudge_
              << std::endl << "   Lanczos upper bound = " << SCFDG_comp_subspace_inner_CheFSI_upper_bound_ ;


          }


          GetTime(extra_timeSta);

          CheFSI_Hmat_top_serial(temp_Hmat,
              temp_Xmat,
              temp_eig_vals_Xmat,
              Hmat_top_states_ChebyFilterOrder_,
              Hmat_top_states_ChebyCycleNum_,
              SCFDG_comp_subspace_inner_CheFSI_lower_bound_,SCFDG_comp_subspace_inner_CheFSI_upper_bound_, SCFDG_comp_subspace_inner_CheFSI_a_L_);


          GetTime(extra_timeEnd);



          statusOFS << std::endl << " Serial CheFSI completed on " <<  SCFDG_comp_subspace_N_solve_ 
            << " top states ( " << (extra_timeEnd - extra_timeSta ) << " s.)";

          //exit(1);                

        }

        // Broadcast the results from proc 0 to ensure all procs are using the same eigenstates
        MPI_Bcast(temp_Xmat.Data(), SCFDG_comp_subspace_N_solve_ * width, MPI_DOUBLE, 0, domain_.comm); // Eigenvectors
        MPI_Bcast(temp_eig_vals_Xmat.Data(), SCFDG_comp_subspace_N_solve_ , MPI_DOUBLE, 0, domain_.comm); // Eigenvalues


        // Copy back the eigenstates to the guess for the next step

        // Eigenvectors
        lapack::Lacpy( 'A', width, SCFDG_comp_subspace_N_solve_, temp_Xmat.Data(), width, 
            SCFDG_comp_subspace_start_guess_.Data(), width );


        // Eigenvalues  
        SCFDG_comp_subspace_top_eigvals_.Resize(SCFDG_comp_subspace_N_solve_);
        for(Int copy_iter = 0; copy_iter < SCFDG_comp_subspace_N_solve_; copy_iter ++)
          SCFDG_comp_subspace_top_eigvals_[copy_iter] = temp_eig_vals_Xmat[copy_iter];

        // Also update the top eigenvalues in hamDG in case we need them
        // For example, they are required if we switch back to regular CheFSI at some stage 
        Int n_top = hamDG.NumStateTotal() - 1;
        for(Int copy_iter = 0; copy_iter < SCFDG_comp_subspace_N_solve_; copy_iter ++)
          eigval[n_top - copy_iter] = SCFDG_comp_subspace_top_eigvals_[copy_iter];

        // Compute the occupations    
        SCFDG_comp_subspace_top_occupations_.Resize(SCFDG_comp_subspace_N_solve_);

        Int howmany_to_calc = (hamDGPtr_->NumOccupiedState() + SCFDG_comp_subspace_N_solve_) - hamDGPtr_->NumStateTotal(); 
        scfdg_calc_occ_rate_comp_subspc(SCFDG_comp_subspace_top_eigvals_,SCFDG_comp_subspace_top_occupations_, howmany_to_calc);


        statusOFS << std::endl << " npsi = " << hamDGPtr_->NumStateTotal();
        statusOFS << std::endl << " nOccStates = " << hamDGPtr_->NumOccupiedState();
        statusOFS << std::endl << " howmany_to_calc = " << howmany_to_calc;
	statusOFS << std::endl << " SCFDG_comp_subspace_N_solve_ = " << SCFDG_comp_subspace_N_solve_ << std::endl;


        statusOFS << std::endl << " Fermi level = " << fermi_ << std::endl;
	statusOFS << std::endl << " Top Eigenvalues and occupations (in reverse order) : ";
	
	for(int print_iter = 0; print_iter < SCFDG_comp_subspace_N_solve_; print_iter ++)
	  statusOFS << std::endl <<  " " << std::setw(8) << print_iter << std::setw(20) << SCFDG_comp_subspace_top_eigvals_[print_iter] 
	                         << '\t' << SCFDG_comp_subspace_top_occupations_[print_iter];


        // Form the matrix C by scaling the eigenvectors with the appropriate occupation related weights
        SCFDG_comp_subspace_matC_.Resize(width, SCFDG_comp_subspace_N_solve_);
        lapack::Lacpy( 'A', width, SCFDG_comp_subspace_N_solve_, temp_Xmat.Data(), width, 
            SCFDG_comp_subspace_matC_.Data(), width );

        double scale_fac;
        for(Int scal_iter = 0; scal_iter < SCFDG_comp_subspace_N_solve_; scal_iter ++)
        {
          scale_fac = sqrt(1.0 - SCFDG_comp_subspace_top_occupations_(scal_iter));
          blas::Scal(width, scale_fac, SCFDG_comp_subspace_matC_.VecData(scal_iter), 1);
        }



        // This calculation is done for computing the band energy later
        SCFDG_comp_subspace_trace_Hmat_ = 0.0;
        for(Int trace_calc = 0; trace_calc < width; trace_calc ++)
          SCFDG_comp_subspace_trace_Hmat_ += temp_Hmat(trace_calc, trace_calc);


        statusOFS << std::endl << std::endl << " ------------------------------- ";


        // // This is for debugging purposes 
        //     DblNumVec eig_vals_Raleigh_Ritz; 
        //     DblNumVec occup_Raleigh_Ritz; 
        //     
        //     eig_vals_Raleigh_Ritz.Resize(width);
        //     occup_Raleigh_Ritz.Resize(width);
        //     
        //     
        //     SetValue(eig_vals_Raleigh_Ritz, 0.0);
        //     SetValue(occup_Raleigh_Ritz, 0.0);
        //     
        //     lapack::Syevd( 'V', 'U', width, square_mat.Data(), width, eig_vals_Raleigh_Ritz.Data() );
        //     CalculateOccupationRate(eig_vals_Raleigh_Ritz , occup_Raleigh_Ritz );
        //     
        //     statusOFS << std::endl << " LAPACK eigenvalues = " << std::endl << eig_vals_Raleigh_Ritz << std::endl;
        //     statusOFS << std::endl << " LAPACK occupations = " << std::endl << occup_Raleigh_Ritz << std::endl;
        //     statusOFS << std::endl << " LAPACK Fermi level = " <<  fermi_ << std::endl;
        //     
        //statusOFS << std::endl << " LOBPCG eigenvalues = " << std::endl << temp_eig_vals_Xmat << std::endl;
        //statusOFS << std::endl << " LAPACK eigenvalues = " << std::endl << eig_vals_Raleigh_Ritz << std::endl;

        //statusOFS << std::endl << " LOBPCG Eigenvectors = " << std::endl << temp_Xmat;
        //statusOFS << std::endl << " LAPACK Eigenvectors = " << std::endl << square_mat;
      }



      /// @brief Routines related to Chebyshev polynomial filtered 
      /// complementary subspace iteration strategy in DGDFT in parallel
      void SCFDG::scfdg_complementary_subspace_parallel(Int filter_order )
      {

        statusOFS << std::endl << " ----------------------------------------------------------" ; 
        statusOFS << std::endl << " Complementary Subspace Strategy (Parallel Subspace version): " << std::endl ; 

        Int mpirank, mpisize;
        MPI_Comm_rank( domain_.comm, &mpirank );
        MPI_Comm_size( domain_.comm, &mpisize );
        Int mpirankRow;  MPI_Comm_rank(domain_.rowComm, &mpirankRow);
        Int mpisizeRow;  MPI_Comm_size(domain_.rowComm, &mpisizeRow);
        Int mpirankCol;  MPI_Comm_rank(domain_.colComm, &mpirankCol);
        Int mpisizeCol;  MPI_Comm_size(domain_.colComm, &mpisizeCol);

        Real timeSta, timeEnd;
        Real extra_timeSta, extra_timeEnd;
        Real cheby_timeSta, cheby_timeEnd;

        HamiltonianDG&  hamDG = *hamDGPtr_;

        DblNumVec& eigval = hamDG.EigVal(); 

        // Step 0: Safeguard against the eigenvector containing extra keys
        {
          std::map<Index3, DblNumMat>::iterator cleaner_itr = hamDG.EigvecCoef().LocalMap().begin();
          while (cleaner_itr != hamDG.EigvecCoef().LocalMap().end()) 
          {
            if (cleaner_itr->first != my_cheby_eig_vec_key_) 
            {
              std::map<Index3, DblNumMat>::iterator toErase = cleaner_itr;
              ++ cleaner_itr;
              hamDG.EigvecCoef().LocalMap().erase(toErase);
            } 
            else 
            {
              ++ cleaner_itr;
            }
          }
        }


        // Step 1: Obtain the upper bound and the Ritz values (for the lower bound)
        // using the Lanczos estimator
        DblNumVec Lanczos_Ritz_values;
        statusOFS << std::endl << " Estimating the spectral bounds ... "; 
        GetTime( extra_timeSta );
        const int Num_Lanczos_Steps = 6;
        double b_up = scfdg_Cheby_Upper_bound_estimator(Lanczos_Ritz_values, Num_Lanczos_Steps);  
        GetTime( extra_timeEnd );
        statusOFS << " Done. ( " << (extra_timeEnd - extra_timeSta ) << " s.)";

        double a_L;
        double b_low;

        if(SCFDG_comp_subspace_engaged_ == 1)
        {
          a_L = SCFDG_comp_subspace_saved_a_L_;
          b_low = SCFDG_comp_subspace_top_eigvals_[0];
        }
        else
        {
          // First time we are doing this : use earlier ScaLAPACK results 
          a_L = eigval[0];
          SCFDG_comp_subspace_saved_a_L_ = a_L;

          b_low = eigval[hamDG.NumStateTotal() - 1];
        }


        // Step 2: Perform filtering
        const Int Filter_Order = filter_order ;

        GetTime( cheby_timeSta );
        // Filter the subspace
        statusOFS << std::endl << std::endl << " Filtering the subspace ... (Filter order = " << Filter_Order << ")";
        GetTime( timeSta );
        scfdg_Chebyshev_filter_scaled(Filter_Order, b_low, b_up, a_L);
        GetTime( timeEnd );
        statusOFS << std::endl << " Filtering completed. ( " << (timeEnd - timeSta ) << " s.)";


        // Step 3: Perform subspace projected problems
        // Subspace problems solved in parallel here

        statusOFS << std::endl << std::endl << " Solving subspace problems in parallel :" << std::endl;

        // YYYY
        // Step a : Convert to ScaLAPACK format
        // Setup BLACS / ScaLAPACK    
        statusOFS << std::endl << " Setting up BLACS Process Grids ...";
        GetTime( timeSta );

        // Set up 3 independent ScaLAPACK contexts 
        // First one is just the regular one that arises in CheFSI.
        // It involves the first row of processes (i.e., 1 processor for every DG element)

        int num_cheby_scala_procs = mpisizeCol; 

        // Figure out the process grid dimensions for the cheby_scala_context
        int temp_factor = int(sqrt(double(num_cheby_scala_procs)));
        while(num_cheby_scala_procs % temp_factor != 0 )
          ++temp_factor;

        // temp_factor now contains the process grid height
        int cheby_scala_num_rows = temp_factor;      
        int cheby_scala_num_cols = num_cheby_scala_procs / temp_factor;


        // We favor process grids which are taller instead of being wider
        if(cheby_scala_num_cols > cheby_scala_num_rows)
        {
          int exchg_temp = cheby_scala_num_cols;
          cheby_scala_num_cols = cheby_scala_num_rows;
          cheby_scala_num_rows = exchg_temp;
        } 

        // Set up the ScaLAPACK context
        IntNumVec cheby_scala_pmap(num_cheby_scala_procs);

        // Use the first processor from every DG-element 
        for ( Int pmap_iter = 0; pmap_iter < num_cheby_scala_procs; pmap_iter++ )
          cheby_scala_pmap[pmap_iter] = pmap_iter * mpisizeRow; 

        // Set up BLACS for subsequent ScaLAPACK operations
        Int cheby_scala_context = -1;
        dgdft::scalapack::Cblacs_get( 0, 0, &cheby_scala_context );
        dgdft::scalapack::Cblacs_gridmap(&cheby_scala_context, &cheby_scala_pmap[0], cheby_scala_num_rows, cheby_scala_num_rows, cheby_scala_num_cols);

        statusOFS << std::endl << " Cheby-Scala context will use " << num_cheby_scala_procs << " processes.";
        statusOFS << std::endl << " Cheby-Scala process grid dim. = " << cheby_scala_num_rows << " * " << cheby_scala_num_cols << " .";


        // Next one is the "bigger grid" for doing some linear algebra operations
        int bigger_grid_num_procs = num_cheby_scala_procs * SCFDG_CS_bigger_grid_dim_fac_;

        if(bigger_grid_num_procs > mpisize)
        {
          SCFDG_CS_bigger_grid_dim_fac_ = mpisize / num_cheby_scala_procs;
          bigger_grid_num_procs = mpisize;

          statusOFS << std::endl << std::endl << " Warning !! Check input parameter SCFDG_CS_bigger_grid_dim_fac .";
          statusOFS << std::endl << " Requested process grid is bigger than total no. of processes.";
          statusOFS << std::endl << " Using " << bigger_grid_num_procs << " processes instead. ";
          statusOFS << std::endl << " Calculation will now continue without throwing exception.";

          statusOFS << std::endl;      
        }

        int bigger_grid_num_rows = cheby_scala_num_rows * SCFDG_CS_bigger_grid_dim_fac_;
        int bigger_grid_num_cols = cheby_scala_num_cols;

        IntNumVec bigger_grid_pmap(bigger_grid_num_procs);
        int pmap_ctr = 0;
        for (Int pmap_iter_1 = 0; pmap_iter_1 < num_cheby_scala_procs; pmap_iter_1 ++)
        {
          for (Int pmap_iter_2 = 0; pmap_iter_2 < SCFDG_CS_bigger_grid_dim_fac_; pmap_iter_2 ++)
          {
            bigger_grid_pmap[pmap_ctr] =  pmap_iter_1 * mpisizeRow + pmap_iter_2;
            pmap_ctr ++;
          }
        }

        Int bigger_grid_context = -1;
        dgdft::scalapack::Cblacs_get( 0, 0, &bigger_grid_context );
        dgdft::scalapack::Cblacs_gridmap(&bigger_grid_context, &bigger_grid_pmap[0], bigger_grid_num_rows, bigger_grid_num_rows, bigger_grid_num_cols);   

        statusOFS << std::endl << " Bigger grid context will use " << bigger_grid_num_procs << " processes.";
        statusOFS << std::endl << " Bigger process grid dim. = " << bigger_grid_num_rows << " * " << bigger_grid_num_cols << " .";

        // Finally, there is the single process case
        Int single_proc_context = -1;
        Int single_proc_pmap[1];  
        single_proc_pmap[0] = 0; // Just using proc. 0 for the job.

        // Set up BLACS for for the single proc context
        dgdft::scalapack::Cblacs_get( 0, 0, &single_proc_context );
        dgdft::scalapack::Cblacs_gridmap(&single_proc_context, &single_proc_pmap[0], 1, 1, 1);

        // For safety, make sure this is MPI Rank zero : throw an error otherwise
        // Fix this in the future ?
        if(single_proc_context >= 0)
        {  
          if(mpirank != 0)
          {
            statusOFS << std::endl << std::endl << "  Error !! BLACS rank 0 does not match MPI rank 0 "
              << " Aborting ... " << std::endl << std::endl;

            MPI_Abort(domain_.comm, 0);
          }
        }

        statusOFS << std::endl << " Single process context works with process 0 .";
        statusOFS << std::endl << " Single process dimension = 1 * 1 ." << std::endl;

        GetTime( timeEnd );
        statusOFS << " BLACS setup done. ( " << (timeEnd - timeSta ) << " s.)";

        // Some diagnostic info
	statusOFS << std::endl;
	statusOFS << std::endl << " Note : On Cheby-Scala grid, scala_block_size * cheby_scala_num_rows = " << (scaBlockSize_ * cheby_scala_num_rows);
        statusOFS << std::endl << "        On Cheby-Scala grid, scala_block_size * cheby_scala_num_cols = " << (scaBlockSize_ * cheby_scala_num_cols);
        statusOFS << std::endl << "        On bigger grid, scala_block_size * bigger_grid_num_rows = " << (scaBlockSize_ * bigger_grid_num_rows);
	statusOFS << std::endl << "        On bigger grid, scala_block_size * bigger_grid_num_cols = " << (scaBlockSize_ * bigger_grid_num_cols);
        statusOFS << std::endl << "        Outer subspace problem dimension = " << hamDG.NumStateTotal() << " * " << hamDG.NumStateTotal();
        statusOFS << std::endl << "        Inner subspace problem dimension = " << SCFDG_comp_subspace_N_solve_ << " * " << SCFDG_comp_subspace_N_solve_;
	statusOFS << std::endl << "        Matrix C has dimension = " << hamDG.NumStateTotal() << " * " << SCFDG_comp_subspace_N_solve_;
	statusOFS << std::endl << "        Matrix C occupies " << (double(hamDG.NumStateTotal() *  SCFDG_comp_subspace_N_solve_ * 8) / double(1048576)) << " MBs per process.";

        statusOFS << std::endl ;
	
        // Step b: Orthonormalize using "bigger grid"
        dgdft::scalapack::ScaLAPACKMatrix<Real>  cheby_scala_eigvecs_X;
        dgdft::scalapack::ScaLAPACKMatrix<Real>  bigger_grid_eigvecs_X;

        scalapack::Descriptor cheby_eigvec_desc;
        scalapack::Descriptor bigger_grid_eigvec_desc;

        statusOFS << std::endl << std::endl << " Orthonormalization step : " << std::endl;

        // DG to Cheby-Scala format conversion
        if(cheby_scala_context >= 0)
        { 

          statusOFS << std::endl << " Distributed vector X to ScaLAPACK (Cheby-Scala grid) conversion ... ";
          GetTime( timeSta );

          cheby_eigvec_desc.Init( hamDG.NumBasisTotal(), hamDG.NumStateTotal(), 
              scaBlockSize_, scaBlockSize_, 
              0, 0, 
              cheby_scala_context);

          cheby_scala_eigvecs_X.SetDescriptor(cheby_eigvec_desc);


          // Make the conversion call         
          scfdg_Cheby_convert_eigvec_distmat_to_ScaLAPACK(hamDG.EigvecCoef(),
              domain_.colComm,
              cheby_eigvec_desc,
              cheby_scala_eigvecs_X);

          GetTime( timeEnd );
          statusOFS << " Done. ( " << (timeEnd - timeSta ) << " s.)";
        }

        // Cheby-Scala to Big grid format conversion

        Int M_copy_ =  hamDG.NumBasisTotal();
        Int N_copy_ =  hamDG.NumStateTotal();


        if(bigger_grid_context >= 0)
        {

          statusOFS << std::endl << " Cheby-Scala grid to bigger grid pdgemr2d for X... ";
          GetTime( timeSta );

          bigger_grid_eigvec_desc.Init( M_copy_, N_copy_, 
              scaBlockSize_, scaBlockSize_, 
              0, 0, 
              bigger_grid_context);  

          bigger_grid_eigvecs_X.SetDescriptor(bigger_grid_eigvec_desc);

          SCALAPACK(pdgemr2d)(&M_copy_, &N_copy_, 
              cheby_scala_eigvecs_X.Data(), &I_ONE, &I_ONE,
              cheby_scala_eigvecs_X.Desc().Values(), 
              bigger_grid_eigvecs_X.Data(), &I_ONE, &I_ONE, 
              bigger_grid_eigvecs_X.Desc().Values(), 
              &bigger_grid_context);      

          GetTime( timeEnd );
          statusOFS << " Done. ( " << (timeEnd - timeSta ) << " s.)" << std::endl;

        }


        // Make the ScaLAPACK calls for orthonormalization   
        GetTime( timeSta );
        if(bigger_grid_context >= 0)
        {
          GetTime( extra_timeSta );

          // Compute C = X^T * X
          dgdft::scalapack::Descriptor bigger_grid_chol_desc( hamDG.NumStateTotal(), hamDG.NumStateTotal(), 
              scaBlockSize_, scaBlockSize_, 
              0, 0, 
              bigger_grid_context);

          dgdft::scalapack::ScaLAPACKMatrix<Real>  bigger_grid_chol_mat;
          bigger_grid_chol_mat.SetDescriptor(bigger_grid_chol_desc);

          if(SCFDG_comp_subspace_syrk_ == 0)
          {
            dgdft::scalapack::Gemm( 'T', 'N',
                hamDG.NumStateTotal(), hamDG.NumStateTotal(), hamDG.NumBasisTotal(),
                1.0,
                bigger_grid_eigvecs_X.Data(), I_ONE, I_ONE, bigger_grid_eigvecs_X.Desc().Values(), 
                bigger_grid_eigvecs_X.Data(), I_ONE, I_ONE, bigger_grid_eigvecs_X.Desc().Values(),
                0.0,
                bigger_grid_chol_mat.Data(), I_ONE, I_ONE, bigger_grid_chol_mat.Desc().Values(),
                bigger_grid_context);
          }
          else
          {  

            dgdft::scalapack::Syrk('U', 'T',
                hamDG.NumStateTotal(), hamDG.NumBasisTotal(),
                1.0, bigger_grid_eigvecs_X.Data(),
                I_ONE, I_ONE,bigger_grid_eigvecs_X.Desc().Values(),
                0.0, bigger_grid_chol_mat.Data(),
                I_ONE, I_ONE,bigger_grid_chol_mat.Desc().Values());
          }


          GetTime( extra_timeEnd);            
          if(SCFDG_comp_subspace_syrk_ == 0)
            statusOFS << std::endl << " Overlap matrix computed using GEMM in : " << (extra_timeEnd - extra_timeSta) << " s.";
          else
            statusOFS << std::endl << " Overlap matrix computed using SYRK in : " << (extra_timeEnd - extra_timeSta) << " s.";

          GetTime( extra_timeSta);

          // Compute V = Chol(C)
          dgdft::scalapack::Potrf( 'U', bigger_grid_chol_mat);

          GetTime( extra_timeEnd);
          statusOFS << std::endl << " Cholesky factorization computed in : " << (extra_timeEnd - extra_timeSta) << " s.";


          GetTime( extra_timeSta);

          // Compute  X = X * V^{-1}
          dgdft::scalapack::Trsm( 'R', 'U', 'N', 'N', 1.0,
              bigger_grid_chol_mat, 
              bigger_grid_eigvecs_X );

          GetTime( extra_timeEnd);
          statusOFS << std::endl << " TRSM computed in : " << (extra_timeEnd - extra_timeSta) << " s." << std::endl;    
        }

        GetTime( timeEnd );
        statusOFS << " Total time for ScaLAPACK calls during Orthonormalization  = " << (timeEnd - timeSta ) << " s." << std::endl;


        if(bigger_grid_context >= 0)
        {
          // Convert back to Cheby-Scala grid
          statusOFS << std::endl << " Bigger grid to Cheby-Scala grid pdgemr2d ... ";
          GetTime( timeSta );


          SCALAPACK(pdgemr2d)(&M_copy_, &N_copy_, 
              bigger_grid_eigvecs_X.Data(), &I_ONE, &I_ONE,
              bigger_grid_eigvecs_X.Desc().Values(), 
              cheby_scala_eigvecs_X.Data(), &I_ONE, &I_ONE, 
              cheby_scala_eigvecs_X.Desc().Values(), 
              &bigger_grid_context);  

          GetTime( timeEnd );
          statusOFS << " Done. ( " << (timeEnd - timeSta ) << " s.)";
        }    



        if(cheby_scala_context >= 0)
        {
          statusOFS << std::endl << " ScaLAPACK (Cheby-Scala grid) to Distributed vector conversion ... ";
          GetTime( timeSta );

          // Convert to DG-distributed matrix format
          ScaMatToDistNumMat(cheby_scala_eigvecs_X ,
              elemPrtn_,
              hamDG.EigvecCoef(),
              hamDG.ElemBasisIdx(), 
              domain_.colComm, 
              hamDG.NumStateTotal() );

          GetTime( timeEnd );
          statusOFS << " Done. ( " << (timeEnd - timeSta ) << " s.)" << std::endl;     

        }

        // Communicate the orthonormalized eigenvectors (to other intra-element processors)
        statusOFS << std::endl << " Communicating orthonormalized filtered vectors ... ";
        GetTime( timeSta );

        DblNumMat &ref_mat_1 =  (hamDG.EigvecCoef().LocalMap().begin())->second;
        MPI_Bcast(ref_mat_1.Data(), (ref_mat_1.m() * ref_mat_1.n()), MPI_DOUBLE, 0, domain_.rowComm);

        GetTime( timeEnd );
        statusOFS << " Done. ( " << (timeEnd - timeSta ) << " s.)" << std::endl;


        // Step c : Perform alternate to Raleigh-Ritz step
        // Compute H * X
        statusOFS << std::endl << " Computing H * X for filtered orthonormal vectors : ";
        GetTime( timeSta );

        DistVec<Index3, DblNumMat, ElemPrtn>  result_mat_HX;
        scfdg_Hamiltonian_times_eigenvectors(result_mat_HX);

        GetTime( timeEnd );
        statusOFS << std::endl << " H * X for filtered orthonormal vectors computed . ( " << (timeEnd - timeSta ) << " s.)";


        statusOFS << std::endl << std::endl << " Alternate to Raleigh-Ritz step : " << std::endl;

        GetTime(timeSta);
        // Convert HX to ScaLAPACK format on Cheby-Scala grid
        dgdft::scalapack::ScaLAPACKMatrix<Real>  cheby_scala_HX;
        dgdft::scalapack::ScaLAPACKMatrix<Real>  bigger_grid_HX;

        dgdft::scalapack::Descriptor cheby_scala_HX_desc;
        dgdft::scalapack::Descriptor bigger_grid_HX_desc;


        // Convert HX to ScaLAPACK format on Cheby-Scala grid
        if(cheby_scala_context >= 0)
        {
          statusOFS << std::endl << " Distributed vector HX to ScaLAPACK (Cheby-Scala grid) conversion ... ";
          GetTime( extra_timeSta );

          cheby_scala_HX_desc.Init(hamDG.NumBasisTotal(), hamDG.NumStateTotal(), 
              scaBlockSize_, scaBlockSize_, 
              0, 0, 
              cheby_scala_context);

          cheby_scala_HX.SetDescriptor(cheby_scala_HX_desc);   



          scfdg_Cheby_convert_eigvec_distmat_to_ScaLAPACK(result_mat_HX,
              domain_.colComm,
              cheby_scala_HX_desc,
              cheby_scala_HX);

          GetTime( extra_timeEnd );
          statusOFS << " Done. ( " << (extra_timeEnd - extra_timeSta ) << " s.)";            

        }

        // Move to bigger grid from Cheby-Scala grid 
        if(bigger_grid_context >= 0)
        {
          statusOFS << std::endl << " Cheby-Scala grid to bigger grid pdgemr2d for HX ... ";
          GetTime( extra_timeSta );


          bigger_grid_HX_desc.Init(hamDG.NumBasisTotal(), hamDG.NumStateTotal(), 
              scaBlockSize_, scaBlockSize_, 
              0, 0, 
              bigger_grid_context);

          bigger_grid_HX.SetDescriptor(bigger_grid_HX_desc);   



          SCALAPACK(pdgemr2d)(&M_copy_, &N_copy_, 
              cheby_scala_HX.Data(), &I_ONE, &I_ONE,
              cheby_scala_HX.Desc().Values(), 
              bigger_grid_HX.Data(), &I_ONE, &I_ONE, 
              bigger_grid_HX.Desc().Values(), 
              &bigger_grid_context);      

          GetTime( extra_timeEnd );
          statusOFS << " Done. ( " << (extra_timeEnd - extra_timeSta ) << " s.)";

        }
        // Compute X^T * HX on bigger grid
        dgdft::scalapack::Descriptor bigger_grid_square_mat_desc;
        dgdft::scalapack::ScaLAPACKMatrix<Real>  bigger_grid_square_mat;

        if(SCFDG_comp_subspace_syr2k_ == 0)
          statusOFS << std::endl << " Computing X^T * HX on bigger grid using GEMM ... ";
        else
          statusOFS << std::endl << " Computing X^T * HX on bigger grid using SYR2K + TRADD ... ";

        GetTime( extra_timeSta );
        if(bigger_grid_context >= 0)
        {
          bigger_grid_square_mat_desc.Init( hamDG.NumStateTotal(), hamDG.NumStateTotal(), 
              scaBlockSize_, scaBlockSize_, 
              0, 0, 
              bigger_grid_context);


          bigger_grid_square_mat.SetDescriptor(bigger_grid_square_mat_desc);


          if(SCFDG_comp_subspace_syr2k_ == 0)
          {        
            dgdft::scalapack::Gemm('T', 'N',
                hamDG.NumStateTotal(), hamDG.NumStateTotal(), hamDG.NumBasisTotal(),
                1.0,
                bigger_grid_eigvecs_X.Data(), I_ONE, I_ONE, bigger_grid_eigvecs_X.Desc().Values(), 
                bigger_grid_HX.Data(), I_ONE, I_ONE, bigger_grid_HX.Desc().Values(),
                0.0,
                bigger_grid_square_mat.Data(), I_ONE, I_ONE, bigger_grid_square_mat.Desc().Values(),
                bigger_grid_context);
          }
          else
          {

            dgdft::scalapack::Syr2k ('U', 'T',
                hamDG.NumStateTotal(), hamDG.NumBasisTotal(),
                0.5, 
                bigger_grid_eigvecs_X.Data(), I_ONE, I_ONE, bigger_grid_eigvecs_X.Desc().Values(),
                bigger_grid_HX.Data(), I_ONE, I_ONE, bigger_grid_HX.Desc().Values(),
                0.0,
                bigger_grid_square_mat.Data(), I_ONE, I_ONE, bigger_grid_square_mat.Desc().Values());


            // Copy the upper triangle to a temporary location
            dgdft::scalapack::ScaLAPACKMatrix<Real>  bigger_grid_square_mat_copy;
            bigger_grid_square_mat_copy.SetDescriptor(bigger_grid_square_mat_desc);

            char uplo = 'A';
            int ht = hamDG.NumStateTotal();
            dgdft::scalapack::SCALAPACK(pdlacpy)(&uplo, &ht, &ht,
                bigger_grid_square_mat.Data(), &I_ONE, &I_ONE, bigger_grid_square_mat.Desc().Values(), 
                bigger_grid_square_mat_copy.Data(), &I_ONE, &I_ONE, bigger_grid_square_mat_copy.Desc().Values() );

            uplo = 'L';
            char trans = 'T';
            double scalar_one = 1.0, scalar_zero = 0.0;
            dgdft::scalapack::SCALAPACK(pdtradd)(&uplo, &trans, &ht, &ht,
                &scalar_one,
                bigger_grid_square_mat_copy.Data(), &I_ONE, &I_ONE, bigger_grid_square_mat_copy.Desc().Values(), 
                &scalar_zero,
                bigger_grid_square_mat.Data(), &I_ONE, &I_ONE, bigger_grid_square_mat.Desc().Values());

          }        

        } 

        GetTime( extra_timeEnd );
        statusOFS << " Done. ( " << (extra_timeEnd - extra_timeSta ) << " s.)" ;

        // Move square matrix to Cheby-Scala grid for working with top states
        dgdft::scalapack::Descriptor cheby_scala_square_mat_desc;
        dgdft::scalapack::ScaLAPACKMatrix<Real>  cheby_scala_square_mat;

        statusOFS << std::endl << " Moving X^T * HX to Cheby-Scala grid using pdgemr2d ... ";
        GetTime( extra_timeSta );

        if(cheby_scala_context >= 0)
        {
          cheby_scala_square_mat_desc.Init( hamDG.NumStateTotal(), hamDG.NumStateTotal(), 
              scaBlockSize_, scaBlockSize_, 
              0, 0, 
              cheby_scala_context);

          cheby_scala_square_mat.SetDescriptor(cheby_scala_square_mat_desc);     
        }

        if(bigger_grid_context >= 0)
        {
          SCALAPACK(pdgemr2d)(&N_copy_, &N_copy_, 
              bigger_grid_square_mat.Data(), &I_ONE, &I_ONE,
              bigger_grid_square_mat.Desc().Values(), 
              cheby_scala_square_mat.Data(), &I_ONE, &I_ONE, 
              cheby_scala_square_mat.Desc().Values(), 
              &bigger_grid_context);      
        }

        GetTime( extra_timeEnd );
        statusOFS << " Done. ( " << (extra_timeEnd - extra_timeSta ) << " s.)" << std::endl;



        // All ready for doing the inner CheFSI
        if(cheby_scala_context >= 0)
        {
          // Obtain the spectral bounds    

          // Fix the filter bounds using full ScaLAPACK results if inner Cheby has not been engaged
          if(SCFDG_comp_subspace_engaged_ != 1)
          {
            SCFDG_comp_subspace_inner_CheFSI_a_L_ = - eigval[hamDG.NumStateTotal() - 1];

            int state_ind = hamDG.NumStateTotal() - SCFDG_comp_subspace_N_solve_;        
            SCFDG_comp_subspace_inner_CheFSI_lower_bound_ = -0.5 * (eigval[state_ind] + eigval[state_ind - 1]);

            statusOFS << std::endl << " Computing upper bound of projected Hamiltonian (parallel) ... ";
            GetTime( extra_timeSta );
            SCFDG_comp_subspace_inner_CheFSI_upper_bound_ = find_comp_subspace_UB_parallel(cheby_scala_square_mat);
            GetTime( extra_timeEnd );
            statusOFS << " Done. ( " << (extra_timeEnd - extra_timeSta ) << " s.)" << std::endl;

            Hmat_top_states_Cheby_delta_fudge_ = 0.5 * (eigval[state_ind] - eigval[state_ind - 1]);

            statusOFS << std::endl << " Going into inner CheFSI routine (parallel) for top states ... ";
            statusOFS << std::endl << "   Lower bound = -(average of prev. eigenvalues " << eigval[state_ind] 
              << " and " <<  eigval[state_ind - 1] << ") = " << SCFDG_comp_subspace_inner_CheFSI_lower_bound_;

          }
          else
          {

            // Fix the filter bounds using earlier inner CheFSI results
            SCFDG_comp_subspace_inner_CheFSI_lower_bound_ = - (SCFDG_comp_subspace_top_eigvals_[SCFDG_comp_subspace_N_solve_ - 1] - Hmat_top_states_Cheby_delta_fudge_);

            statusOFS << std::endl << " Computing upper bound of projected Hamiltonian (parallel) ... ";
            GetTime( extra_timeSta );
            SCFDG_comp_subspace_inner_CheFSI_upper_bound_ = find_comp_subspace_UB_parallel(cheby_scala_square_mat);
            GetTime( extra_timeEnd );
            statusOFS << " Done. ( " << (extra_timeEnd - extra_timeSta ) << " s.)" << std::endl;

            statusOFS << std::endl << " Going into inner CheFSI routine (parallel) for top states ... ";
            statusOFS << std::endl << "   Lower bound eigenvalue = " << SCFDG_comp_subspace_top_eigvals_[SCFDG_comp_subspace_N_solve_ - 1] 
              << std::endl << "   delta_fudge = " << Hmat_top_states_Cheby_delta_fudge_;


          }    
        }


        // Broadcast the inner filter bounds, etc. to every process. 
        // This is definitely required by the procs participating in cheby_scala_context
        // Some of the info is redundant
        double bounds_array[4];
        bounds_array[0] = SCFDG_comp_subspace_inner_CheFSI_lower_bound_;
        bounds_array[1] = SCFDG_comp_subspace_inner_CheFSI_upper_bound_;
        bounds_array[2] = SCFDG_comp_subspace_inner_CheFSI_a_L_;
        bounds_array[3] = Hmat_top_states_Cheby_delta_fudge_;

        MPI_Bcast(bounds_array, 4, MPI_DOUBLE, 0, domain_.comm); 

        SCFDG_comp_subspace_inner_CheFSI_lower_bound_ = bounds_array[0];
        SCFDG_comp_subspace_inner_CheFSI_upper_bound_ = bounds_array[1];
        SCFDG_comp_subspace_inner_CheFSI_a_L_ = bounds_array[2];
        Hmat_top_states_Cheby_delta_fudge_ = bounds_array[3];

        if(cheby_scala_context >= 0)
          statusOFS << std::endl << "   Lanczos upper bound = " << SCFDG_comp_subspace_inner_CheFSI_upper_bound_ << std::endl;


        // Load up and distibute top eigenvectors from serial storage
        scalapack::Descriptor temp_single_proc_desc;
        scalapack::ScaLAPACKMatrix<Real>  temp_single_proc_scala_mat;

        GetTime( extra_timeSta );
        statusOFS << std::endl << " Loading up and distributing initial guess vectors ... ";

        int M_temp_ =  hamDG.NumStateTotal();
        int N_temp_ =  SCFDG_comp_subspace_N_solve_;

        if(single_proc_context >= 0)
        {
          temp_single_proc_desc.Init(M_temp_, N_temp_,
              scaBlockSize_, scaBlockSize_, 
              0, 0,  single_proc_context );              

          temp_single_proc_scala_mat.SetDescriptor( temp_single_proc_desc );

          // Copy from the serial storage to the single process ScaLAPACK matrix
          double *src_ptr, *dest_ptr; 

          // Copy in the regular order      
          for(Int copy_iter = 0; copy_iter < SCFDG_comp_subspace_N_solve_; copy_iter ++)
          {
            src_ptr = SCFDG_comp_subspace_start_guess_.VecData(copy_iter);
            dest_ptr = temp_single_proc_scala_mat.Data() + copy_iter * temp_single_proc_scala_mat.LocalLDim();

            blas::Copy( M_temp_, src_ptr, 1, dest_ptr, 1 );                                                 
          }

        }

        // Distribute onto Cheby-Scala grid    
        scalapack::Descriptor Xmat_desc;
        dgdft::scalapack::ScaLAPACKMatrix<Real> Xmat;

        if(cheby_scala_context >= 0)
        {
          Xmat_desc.Init(M_temp_, N_temp_,
              scaBlockSize_, scaBlockSize_, 
              0, 0,  cheby_scala_context );          

          Xmat.SetDescriptor(Xmat_desc);     

          SCALAPACK(pdgemr2d)(&M_temp_, &N_temp_, 
              temp_single_proc_scala_mat.Data(), &I_ONE, &I_ONE,
              temp_single_proc_scala_mat.Desc().Values(), 
              Xmat.Data(), &I_ONE, &I_ONE, 
              Xmat.Desc().Values(), 
              &cheby_scala_context);    
        }


        GetTime( extra_timeEnd );
        statusOFS << " Done. ( " << (extra_timeEnd - extra_timeSta ) << " s.)" << std::endl;

        // Call the inner CheFSI routine 
        DblNumVec eig_vals_Xmat;
        eig_vals_Xmat.Resize(SCFDG_comp_subspace_N_solve_);

        if(cheby_scala_context >= 0)
        {
          GetTime(extra_timeSta);

          CheFSI_Hmat_top_parallel(cheby_scala_square_mat,
              Xmat,
              eig_vals_Xmat,
              Hmat_top_states_ChebyFilterOrder_,
              Hmat_top_states_ChebyCycleNum_,
              SCFDG_comp_subspace_inner_CheFSI_lower_bound_,SCFDG_comp_subspace_inner_CheFSI_upper_bound_, SCFDG_comp_subspace_inner_CheFSI_a_L_);

          GetTime(extra_timeEnd);


          statusOFS << std::endl << " Parallel CheFSI completed on " <<  SCFDG_comp_subspace_N_solve_ 
            << " top states ( " << (extra_timeEnd - extra_timeSta ) << " s.)";


        }

        // Redistribute and broadcast top eigenvectors to serial storage
        GetTime( extra_timeSta );
        statusOFS << std::endl << " Distributing back and broadcasting inner CheFSI vectors ... ";

        // Redistribute to single process ScaLAPACK matrix
        if(cheby_scala_context >= 0)
        {    
          SCALAPACK(pdgemr2d)(&M_temp_, &N_temp_, 
              Xmat.Data(), &I_ONE, &I_ONE,
              Xmat.Desc().Values(), 
              temp_single_proc_scala_mat.Data(), &I_ONE, &I_ONE, 
              temp_single_proc_scala_mat.Desc().Values(), 
              &cheby_scala_context);    

        }

        if(single_proc_context >= 0)
        {

          // Copy from the single process ScaLAPACK matrix to serial storage
          double *src_ptr, *dest_ptr; 

          // Copy in the regular order      
          for(Int copy_iter = 0; copy_iter < SCFDG_comp_subspace_N_solve_; copy_iter ++)
          {
            src_ptr = temp_single_proc_scala_mat.Data() + copy_iter * temp_single_proc_scala_mat.LocalLDim();
            dest_ptr = SCFDG_comp_subspace_start_guess_.VecData(copy_iter);

            blas::Copy( M_temp_, src_ptr, 1, dest_ptr, 1 );                                                 
          }

        }

        // Broadcast top eigenvectors to all processes
        MPI_Bcast(SCFDG_comp_subspace_start_guess_.Data(),hamDG.NumStateTotal() * SCFDG_comp_subspace_N_solve_, 
            MPI_DOUBLE, 0,  domain_.comm); 

        GetTime( extra_timeEnd );
        statusOFS << " Done. ( " << (extra_timeEnd - extra_timeSta ) << " s.)" << std::endl;

        GetTime(timeEnd);
        statusOFS << std::endl << std::endl << " Alternate to Raleigh-Ritz step performed in " << (timeEnd - timeSta ) << " s.";


        GetTime( extra_timeSta );
        statusOFS << std::endl << " Adjusting top eigenvalues ... ";

        // Broadcast the top eigenvalues to every processor
        MPI_Bcast(eig_vals_Xmat.Data(), SCFDG_comp_subspace_N_solve_ , MPI_DOUBLE, 0, domain_.comm); 


        // Copy these to native storage
        SCFDG_comp_subspace_top_eigvals_.Resize(SCFDG_comp_subspace_N_solve_);
        for(Int copy_iter = 0; copy_iter < SCFDG_comp_subspace_N_solve_; copy_iter ++)
          SCFDG_comp_subspace_top_eigvals_[copy_iter] = eig_vals_Xmat[copy_iter];

        // Also update the top eigenvalues in hamDG in case we need them
        // For example, they are required if we switch back to regular CheFSI at some stage 
        Int n_top = hamDG.NumStateTotal() - 1;
        for(Int copy_iter = 0; copy_iter < SCFDG_comp_subspace_N_solve_; copy_iter ++)
          eigval[n_top - copy_iter] = SCFDG_comp_subspace_top_eigvals_[copy_iter];

        GetTime( extra_timeEnd );
        statusOFS << " Done. ( " << (extra_timeEnd - extra_timeSta ) << " s.)" << std::endl;


        // Compute the occupations    
        GetTime( extra_timeSta );
        statusOFS << std::endl << " Computing occupation numbers : ";   
        SCFDG_comp_subspace_top_occupations_.Resize(SCFDG_comp_subspace_N_solve_);
	

        Int howmany_to_calc = (hamDGPtr_->NumOccupiedState() + SCFDG_comp_subspace_N_solve_) - hamDGPtr_->NumStateTotal(); 
        scfdg_calc_occ_rate_comp_subspc(SCFDG_comp_subspace_top_eigvals_,SCFDG_comp_subspace_top_occupations_, howmany_to_calc);

	statusOFS << std::endl << " npsi = " << hamDGPtr_->NumStateTotal();
        statusOFS << std::endl << " nOccStates = " << hamDGPtr_->NumOccupiedState();
        statusOFS << std::endl << " howmany_to_calc = " << howmany_to_calc;
	statusOFS << std::endl << " SCFDG_comp_subspace_N_solve_ = " << SCFDG_comp_subspace_N_solve_ << std::endl;
	
	GetTime( extra_timeEnd );
        statusOFS << std::endl << " Completed computing occupations. ( " << (extra_timeEnd - extra_timeSta ) << " s.)" << std::endl;

        
	statusOFS << std::endl << " Fermi level = " << fermi_ << std::endl;
	statusOFS << std::endl << " Top Eigenvalues and occupations (in reverse order) : ";
	
	for(int print_iter = 0; print_iter < SCFDG_comp_subspace_N_solve_; print_iter ++)
	   statusOFS << std::endl <<  " " << std::setw(8) << print_iter << std::setw(20) << SCFDG_comp_subspace_top_eigvals_[print_iter] 
	                         << '\t' << SCFDG_comp_subspace_top_occupations_[print_iter];
        
        // Form the matrix C by scaling the eigenvectors with the appropriate occupation related weights

        GetTime( extra_timeSta );
        statusOFS << std::endl << " Forming the occupation number weighted matrix C : ";   

        int wd = hamDGPtr_->NumStateTotal();

        SCFDG_comp_subspace_matC_.Resize(wd, SCFDG_comp_subspace_N_solve_);
        lapack::Lacpy( 'A', wd, SCFDG_comp_subspace_N_solve_, SCFDG_comp_subspace_start_guess_.Data(), wd, 
            SCFDG_comp_subspace_matC_.Data(), wd );

	statusOFS << std::endl << "  Space for matrix C locally allocated. ( " << (double(hamDG.NumStateTotal() *  SCFDG_comp_subspace_N_solve_ * 8) / double(1048576)) << " MBs per process. )";;   

        double scale_fac;
        for(Int scal_iter = 0; scal_iter < SCFDG_comp_subspace_N_solve_; scal_iter ++)
        {
          scale_fac = sqrt(1.0 - SCFDG_comp_subspace_top_occupations_(scal_iter));
          blas::Scal(wd, scale_fac, SCFDG_comp_subspace_matC_.VecData(scal_iter), 1);
        }
        statusOFS << std::endl << "  BLAS Scal operations completed. ";   

        
        GetTime( extra_timeEnd );
        statusOFS << std::endl  << " Occupation number weighted matrix C formed. ( " << (extra_timeEnd - extra_timeSta ) << " s.)" << std::endl;


        // This calculation is done for computing the band energy later
        GetTime( extra_timeSta );
        statusOFS << std::endl << " Computing the trace of the projected Hamiltonian ... ";   

        SCFDG_comp_subspace_trace_Hmat_ = 0.0;

        if(cheby_scala_context >= 0)
        {  
          SCFDG_comp_subspace_trace_Hmat_ = dgdft::scalapack::SCALAPACK(pdlatra)(&M_temp_ , 
              cheby_scala_square_mat.Data() , &I_ONE , &I_ONE , 
              cheby_scala_square_mat.Desc().Values());

        }

        // Broadcast the trace
        MPI_Bcast(&SCFDG_comp_subspace_trace_Hmat_, 1 , MPI_DOUBLE, 0, domain_.comm); 


        GetTime( extra_timeEnd );
        statusOFS << " Done. ( " << (extra_timeEnd - extra_timeSta ) << " s.)" << std::endl;

        statusOFS << std::endl << std::endl << " ------------------------------- ";






        // Adjust other things... trace, etc. Broadcast necessary parts of these results




        statusOFS << std::endl << std::endl << " ------------------------------- " << std::endl;



        // Clean up BLACS
        if(cheby_scala_context >= 0) 
        {
          dgdft::scalapack::Cblacs_gridexit( cheby_scala_context );
        }

        if(bigger_grid_context >= 0)
        {
          dgdft::scalapack::Cblacs_gridexit( bigger_grid_context );

        }

        if( single_proc_context >= 0)
        {
          dgdft::scalapack::Cblacs_gridexit( single_proc_context );             
        }



      } // end of scfdg_complementary_subspace_parallel


      // **###**  
      void SCFDG::scfdg_complementary_subspace_compute_fullDM()
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

        DblNumMat XC_mat;
        // First compute the diagonal block
        {
          DblNumMat &mat_local = my_dist_mat.LocalMap()[key];
          ElemMatKey diag_block_key = std::make_pair(key, key);

          //statusOFS << std::endl << " Diag key = " << diag_block_key.first << "  " << diag_block_key.second << std::endl;

          // First compute the X*X^T portion : adjust for numspin
          distDMMat_.LocalMap()[diag_block_key].Resize( mat_local.m(),  mat_local.m());

          blas::Gemm( 'N', 'T', mat_local.m(), mat_local.m(), mat_local.n(),
              hamDG.NumSpin(), 
              mat_local.Data(), mat_local.m(), 
              mat_local.Data(), mat_local.m(),
              0.0, 
              distDMMat_.LocalMap()[diag_block_key].Data(),  mat_local.m());


          // Now compute the X * C portion
          XC_mat.Resize(mat_local.m(), SCFDG_comp_subspace_N_solve_);        
          blas::Gemm( 'N', 'N', mat_local.m(), SCFDG_comp_subspace_N_solve_, mat_local.n(),
              1.0, 
              mat_local.Data(), mat_local.m(), 
              SCFDG_comp_subspace_matC_.Data(), SCFDG_comp_subspace_matC_.m(),
              0.0, 
              XC_mat.Data(),  XC_mat.m());

          // Subtract XC*XC^T from DM : adjust for numspin
          blas::Gemm( 'N', 'T', XC_mat.m(), XC_mat.m(), XC_mat.n(),
              -hamDG.NumSpin(), 
              XC_mat.Data(), XC_mat.m(), 
              XC_mat.Data(), XC_mat.m(),
              1.0, 
              distDMMat_.LocalMap()[diag_block_key].Data(),  mat_local.m());
        }

        // Now handle the off-diagonal blocks
        {

          DblNumMat &mat_local = my_dist_mat.LocalMap()[key];
          for(Int off_diag_iter = 0; off_diag_iter < getKeys_list.size(); off_diag_iter ++)
          {
            DblNumMat &mat_neighbor = my_dist_mat.LocalMap()[getKeys_list[off_diag_iter]];
            ElemMatKey off_diag_key = std::make_pair(key, getKeys_list[off_diag_iter]);

            //statusOFS << std::endl << " Off Diag key = " << off_diag_key.first << "  " << off_diag_key.second << std::endl;

            // First compute the Xi * Xj^T portion : adjust for numspin
            distDMMat_.LocalMap()[off_diag_key].Resize( mat_local.m(),  mat_neighbor.m());

            blas::Gemm( 'N', 'T', mat_local.m(), mat_neighbor.m(), mat_local.n(),
                hamDG.NumSpin(), 
                mat_local.Data(), mat_local.m(), 
                mat_neighbor.Data(), mat_neighbor.m(),
                0.0, 
                distDMMat_.LocalMap()[off_diag_key].Data(),  mat_local.m());


            // Now compute the XC portion for the off-diagonal block
            DblNumMat XC_neighbor_mat;
            XC_neighbor_mat.Resize(mat_neighbor.m(), SCFDG_comp_subspace_N_solve_);

            blas::Gemm( 'N', 'N', mat_neighbor.m(), SCFDG_comp_subspace_N_solve_, mat_neighbor.n(),
                1.0, 
                mat_neighbor.Data(), mat_neighbor.m(), 
                SCFDG_comp_subspace_matC_.Data(), SCFDG_comp_subspace_matC_.m(),
                0.0, 
                XC_neighbor_mat.Data(),  XC_neighbor_mat.m());


            // Subtract (Xi C)* (Xj C)^T from off diagonal block : adjust for numspin
            blas::Gemm( 'N', 'T', XC_mat.m(), XC_neighbor_mat.m(), XC_mat.n(),
                -hamDG.NumSpin(), 
                XC_mat.Data(), XC_mat.m(), 
                XC_neighbor_mat.Data(), XC_neighbor_mat.m(),
                1.0, 
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

      }


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

          //statusOFS << std::endl << " Diag key = " << diag_block_key.first << "  " << diag_block_key.second << std::endl;

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

            //statusOFS << std::endl << " Off Diag key = " << off_diag_key.first << "  " << off_diag_key.second << std::endl;

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

      }


      void 
        SCFDG::scfdg_calc_occ_rate_comp_subspc( DblNumVec& top_eigVals, DblNumVec& top_occ, Int num_solve)
        {
          // For a given finite temperature, update the occupation number
          // FIXME Magic number here
          Real tol = 1e-10; 
          Int maxiter = 100;  

          Real lb, ub, flb, fub, fx;
          Int  iter;

          Int npsi       = top_eigVals.m();
          Int nOccStates = num_solve;

          top_occ.Resize(npsi);

          if( npsi > nOccStates )  
          { 
            if(SmearingScheme_ == "FD")
            {  
              // The reverse order for the bounds needs to be used because the eigenvalues appear in decreasing order
              lb = top_eigVals(npsi - 1);
              ub = top_eigVals(0);

              flb = scfdg_fermi_func_comp_subspc(top_eigVals, top_occ, num_solve, lb);
              fub = scfdg_fermi_func_comp_subspc(top_eigVals, top_occ, num_solve, ub);

              if(flb * fub > 0.0)
                ErrorHandling( "Bisection method for finding Fermi level cannot proceed !!" );

              fermi_ = (lb+ub)*0.5;

              /* Start bisection iteration */
              iter = 1;
              fx = scfdg_fermi_func_comp_subspc(top_eigVals, top_occ, num_solve, fermi_);


              while( (fabs(fx) > tol) && (iter < maxiter) ) 
              {
                flb = scfdg_fermi_func_comp_subspc(top_eigVals, top_occ, num_solve, lb);
                fub = scfdg_fermi_func_comp_subspc(top_eigVals, top_occ, num_solve, ub);

                if( (flb * fx) < 0.0 )
                  ub = fermi_;
                else
                  lb = fermi_;

                fermi_ = (lb+ub)*0.5;
                fx = scfdg_fermi_func_comp_subspc(top_eigVals, top_occ, num_solve, fermi_);

                iter++;
              }
            } // end of if (SmearingScheme_ == "fd")
            else
            {
              // GB and MP smearing schemes

              // The reverse order for the bounds needs to be used because the eigenvalues appear in decreasing order
              lb = top_eigVals(npsi - 1);
              ub = top_eigVals(0);

              // Set up the function bounds
              flb = mp_occupations_residual(top_eigVals, lb, num_solve);
              fub = mp_occupations_residual(top_eigVals, ub, num_solve);

              if(flb * fub > 0.0)
                ErrorHandling( "Bisection method for finding Fermi level cannot proceed !!" );

              fermi_ = ( lb + ub ) * 0.5;

              /* Start bisection iteration */
              iter = 1;
              fx = mp_occupations_residual(top_eigVals, fermi_, num_solve);

              while( (fabs(fx) > tol) && (iter < maxiter) ) 
              {
                flb = mp_occupations_residual(top_eigVals, lb, num_solve);
                fub = mp_occupations_residual(top_eigVals, ub, num_solve);

                if( (flb * fx) < 0.0 )
                  ub = fermi_;
                else
                  lb = fermi_;

                fermi_ = ( lb + ub ) * 0.5;
                fx = mp_occupations_residual(top_eigVals, fermi_, num_solve);

                iter++;
              }

              if(iter >= maxiter)
                ErrorHandling( "Bisection method for finding Fermi level does not appear to converge !!" );
              else
              {
                // Bisection method seems to have converged
                // Fill up the occupations
                populate_mp_occupations(top_eigVals, top_occ, fermi_);
              }

            } // end of GB and MP smearing cases

          } // End of finite temperature case
          else 
          {
            if (npsi == nOccStates ) 
            {
              for(Int j = 0; j < npsi; j++) 
                top_occ(j) = 1.0;
              fermi_ = top_eigVals(npsi-1);
            }
            else 
            {
              ErrorHandling( "The number of top eigenvalues should be larger than number of top occupied states" );
            }
          }




          return ;
        }


      double 
        SCFDG::scfdg_fermi_func_comp_subspc( DblNumVec& top_eigVals, DblNumVec& top_occ, Int num_solve, Real x)
        {
          double occsum = 0.0, retval;
          Int npsi = top_eigVals.m();


          for(Int j = 0; j < npsi; j++) 
          {
            top_occ(j) = 1.0 / (1.0 + exp(Tbeta_*(top_eigVals(j) - x)));
            occsum += top_occ(j);     
          }

          retval = occsum - Real(num_solve);

          return retval;
        }



      // Internal routines for MP (and GB) type smearing
      double SCFDG::low_order_hermite_poly(double x, int order)
      {
        double y; 
        switch (order)
        {
          case 0: y = 1; break;
          case 1: y = 2.0 * x; break;
          case 2: y = 4.0 * x * x - 2.0; break;
          case 3: y = 8.0 * x * x * x - 12.0 * x; break;
          case 4: y = 16.0 * x * x * x * x - 48.0 * x * x + 12.0; break;
          case 5: y = 32.0 * x * x * x * x * x - 160.0 * x * x * x + 120.0 * x; break;
          case 6: y = 64.0 * x * x * x * x * x * x - 480.0 * x * x * x * x + 720.0 * x * x - 120.0; 
        }

        return y;
      }

      double SCFDG::mp_occupations(double x, int order)
      {
        const double sqrt_pi = sqrt(M_PI);
        double A_vec[4] = { 1.0 / sqrt_pi, -1.0 / (4.0 * sqrt_pi), 1.0 / (32.0 * sqrt_pi), -1.0 / (384 * sqrt_pi) };
        double y = 0.5 *(1.0 - erf(x));

        for (int m = 1; m <= order; m++)
          y = y + A_vec[m] * low_order_hermite_poly(x, 2 * order - 1) * exp(- x * x);

        return y;

      }


      double SCFDG::mp_entropy(double x, int order)
      {
        const double sqrt_pi = sqrt(M_PI);
        double A_vec[4] = { 1.0 / sqrt_pi, -1.0 / (4.0 * sqrt_pi), 1.0 / (32.0 * sqrt_pi), -1.0 / (384 * sqrt_pi) };

        double y = 0.5 * A_vec[order] * low_order_hermite_poly(x, 2 * order) * exp(- x * x);

        return y;
      }

      // This fills up the the output_occ occupations using the input eigvals, according to the Methfessel-Paxton recipe 
      void SCFDG::populate_mp_occupations(DblNumVec& input_eigvals, DblNumVec& output_occ, double fermi_mu)
      {
        double x, t;

        for(int ii = 0; ii < input_eigvals.m(); ii ++)
        {
          x = (input_eigvals(ii) - fermi_mu) / Tsigma_ ;
          t = mp_occupations(x, MP_smearing_order_); 

          if(t < 0.0)
            t = 0.0;
          if(t > 1.0)
            t = 1.0;

          output_occ(ii) = t;

        }

      }


      // This computes the residual of (\sum_i f_i ) - n_e used for computing the Fermi level
      double  SCFDG::mp_occupations_residual(DblNumVec& input_eigvals, double fermi_mu, int num_solve)
      {
        double x;
        double y = 0.0, t;

        for(int ii = 0; ii < input_eigvals.m(); ii ++)
        {
          x = (input_eigvals(ii) - fermi_mu) / Tsigma_ ;
          t = mp_occupations(x, MP_smearing_order_);

          if(t < 0.0)
            t = 0.0;
          if(t > 1.0)
            t = 1.0;


          y += t;    
        }

        return (y - double(num_solve));

      }


      void
        SCFDG::InnerIterate    ( Int outerIter )
        {
          Int mpirank, mpisize;
          MPI_Comm_rank( domain_.comm, &mpirank );
          MPI_Comm_size( domain_.comm, &mpisize );
          Int mpirankRow;  MPI_Comm_rank(domain_.rowComm, &mpirankRow);
          Int mpisizeRow;  MPI_Comm_size(domain_.rowComm, &mpisizeRow);
          Int mpirankCol;  MPI_Comm_rank(domain_.colComm, &mpirankCol);
          Int mpisizeCol;  MPI_Comm_size(domain_.colComm, &mpisizeCol);

          Real timeSta, timeEnd;
          Real timeIterStart, timeIterEnd;

          HamiltonianDG&  hamDG = *hamDGPtr_;

          bool isInnerSCFConverged = false;

          for( Int innerIter = 1; innerIter <= scfInnerMaxIter_; innerIter++ ){
            if ( isInnerSCFConverged ) break;
            scfTotalInnerIter_++;

            GetTime( timeIterStart );

            statusOFS << std::endl << " Inner SCF iteration #"  
              << innerIter << " starts." << std::endl << std::endl;


            // *********************************************************************
            // Update potential and construct/update the DG matrix
            // *********************************************************************

            if( innerIter == 1 ){
              // The first inner iteration does not update the potential, and
              // construct the global Hamiltonian matrix from scratch
              GetTime(timeSta);


//              MPI_Barrier( domain_.comm );
//             MPI_Barrier( domain_.rowComm );
//              MPI_Barrier( domain_.colComm );


              hamDG.CalculateDGMatrix( );

              MPI_Barrier( domain_.comm );
              MPI_Barrier( domain_.rowComm );
              MPI_Barrier( domain_.colComm );
              GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
              statusOFS << " Time for constructing the DG matrix is " <<
                timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
            }
            else{
              // The consequent inner iterations update the potential in the
              // element, and only update the global Hamiltonian matrix

              // Update the potential in the element (and the extended element)


              GetTime(timeSta);

              // Save the old potential on the LGL grid
              for( Int k = 0; k < numElem_[2]; k++ )
                for( Int j = 0; j < numElem_[1]; j++ )
                  for( Int i = 0; i < numElem_[0]; i++ ){
                    Index3 key( i, j, k );
                    if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
                      Index3 numLGLGrid     = hamDG.NumLGLGridElem();
                      blas::Copy( numLGLGrid.prod(),
                          hamDG.VtotLGL().LocalMap()[key].Data(), 1,
                          vtotLGLSave_.LocalMap()[key].Data(), 1 );
                    } // if (own this element)
                  } // for (i)


              // Update the local potential on the extended element and on the
              // element.
              UpdateElemLocalPotential();


              // Save the difference of the potential on the LGL grid into vtotLGLSave_
              for( Int k = 0; k < numElem_[2]; k++ )
                for( Int j = 0; j < numElem_[1]; j++ )
                  for( Int i = 0; i < numElem_[0]; i++ ){
                    Index3 key( i, j, k );
                    if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
                      Index3 numLGLGrid     = hamDG.NumLGLGridElem();
                      Real *ptrNew = hamDG.VtotLGL().LocalMap()[key].Data();
                      Real *ptrDif = vtotLGLSave_.LocalMap()[key].Data();
                      for( Int p = 0; p < numLGLGrid.prod(); p++ ){
                        (*ptrDif) = (*ptrNew) - (*ptrDif);
                        ptrNew++;
                        ptrDif++;
                      } 
                    } // if (own this element)
                  } // for (i)



              MPI_Barrier( domain_.comm );
              GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
              statusOFS << " Time for updating the local potential in the extended element and the element is " <<
                timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif


              // Update the DG Matrix
              GetTime(timeSta);
              hamDG.UpdateDGMatrix( vtotLGLSave_ );
              MPI_Barrier( domain_.comm );
              GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
              statusOFS << " Time for updating the DG matrix is " <<
                timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

            } // if ( innerIter == 1 )

#if ( _DEBUGlevel_ >= 2 )
            {
              statusOFS << "Owned H matrix blocks on this processor" << std::endl;
              for( std::map<ElemMatKey, DblNumMat>::iterator 
                  mi  = hamDG.HMat().LocalMap().begin();
                  mi != hamDG.HMat().LocalMap().end(); ++mi ){
                ElemMatKey key = (*mi).first;
                statusOFS << key.first << " -- " << key.second << std::endl;
              }
            }
#endif


            // *********************************************************************
            // Write the Hamiltonian matrix to a file (if needed) 
            // *********************************************************************

            if( esdfParam.isOutputHMatrix ){
              // Only the first processor column participates in the conversion
              if( mpirankRow == 0 ){
                DistSparseMatrix<Real>  HSparseMat;

                GetTime(timeSta);
                DistElemMatToDistSparseMat( 
                    hamDG.HMat(),
                    hamDG.NumBasisTotal(),
                    HSparseMat,
                    hamDG.ElemBasisIdx(),
                    domain_.colComm );
                GetTime(timeEnd);
#if ( _DEBUGlevel_ >= 0 )
                statusOFS << " Time for converting the DG matrix to DistSparseMatrix format is " <<
                  timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

                GetTime(timeSta);
                ParaWriteDistSparseMatrix( "H.csc", HSparseMat );
                //            WriteDistSparseMatrixFormatted( "H.matrix", HSparseMat );
                GetTime(timeEnd);
#if ( _DEBUGlevel_ >= 0 )
                statusOFS << " Time for writing the matrix in parallel is " <<
                  timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
              }

              MPI_Barrier( domain_.comm );

            }


            // *********************************************************************
            // Evaluate the density matrix
            // 
            // This can be done either using diagonalization method or using PEXSI
            // *********************************************************************

            // Save the mixing variable first
            {
              for( Int k = 0; k < numElem_[2]; k++ )
                for( Int j = 0; j < numElem_[1]; j++ )
                  for( Int i = 0; i < numElem_[0]; i++ ){
                    Index3 key( i, j, k );
                    if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
                      if( mixVariable_ == "density" ){
                        DblNumVec& oldVec = hamDG.Density().LocalMap()[key];
                        DblNumVec& newVec = mixInnerSave_.LocalMap()[key];
                        blas::Copy( oldVec.Size(), oldVec.Data(), 1,
                            newVec.Data(), 1 );
                      }
                      else if( mixVariable_ == "potential" ){
                        DblNumVec& oldVec = hamDG.Vtot().LocalMap()[key];
                        DblNumVec& newVec = mixInnerSave_.LocalMap()[key];
                        blas::Copy( oldVec.Size(), oldVec.Data(), 1,
                            newVec.Data(), 1 );
                      }

                    } // own this element
                  } // for (i)
            }



            // Method 1: Using diagonalization method
            // With a versatile choice of processors for using ScaLAPACK.
            // Or using Chebyshev filtering

            if( solutionMethod_ == "diag" ){
              {
                // ~~**~~
                if(Diag_SCFDG_by_Cheby_ == 1 )
                {
                  // Chebyshev filtering based diagonalization
                  GetTime(timeSta);

                  if(scfdg_ion_dyn_iter_ != 0)
                  {
                      statusOFS << std::endl << " Calling CheFSI for MD or relaxition calculations  " << std::endl;
                    if(SCFDG_use_comp_subspace_ == 1)
                    {

                      if((scfdg_ion_dyn_iter_ % SCFDG_CS_ioniter_regular_cheby_freq_ == 0) && (outerIter <= Second_SCFDG_ChebyOuterIter_ / 2)) // Just some adhoc criterion used here
                      {
                        // Usual CheFSI to help corrrect drift / SCF convergence
                        statusOFS << std::endl << " Calling Second stage Chebyshev Iter in iondynamics step to improve drift / SCF convergence ..." << std::endl;    

                        scfdg_GeneralChebyStep(Second_SCFDG_ChebyCycleNum_, Second_SCFDG_ChebyFilterOrder_);

                        SCFDG_comp_subspace_engaged_ = 0;
                      }
                      else
                      {  
                        // Decide serial or parallel version here
                        if(SCFDG_comp_subspace_parallel_ == 0)
                        {  
                          statusOFS << std::endl << " Calling Complementary Subspace Strategy (serial subspace version) ...  " << std::endl;
                          scfdg_complementary_subspace_serial(General_SCFDG_ChebyFilterOrder_);
                        }
                        else
                        {
                          statusOFS << std::endl << " Calling Complementary Subspace Strategy (parallel subspace version) ...  " << std::endl;
                          scfdg_complementary_subspace_parallel(General_SCFDG_ChebyFilterOrder_);                     
                        }
                        // Set the engaged flag 
                        SCFDG_comp_subspace_engaged_ = 1;
                      }

                    }
                    else
                    {
                      if(outerIter <= Second_SCFDG_ChebyOuterIter_ / 2) // Just some adhoc criterion used here
                      {
                        // Need to re-use current guess, so do not call the first Cheby step
                        statusOFS << std::endl << " Calling Second stage Chebyshev Iter in iondynamics step " << std::endl;         
                        scfdg_GeneralChebyStep(Second_SCFDG_ChebyCycleNum_, Second_SCFDG_ChebyFilterOrder_);
                      }
                      else
                      {     
                        // Subsequent MD Steps
                        statusOFS << std::endl << " Calling General Chebyshev Iter in iondynamics step " << std::endl;
                        scfdg_GeneralChebyStep(General_SCFDG_ChebyCycleNum_, General_SCFDG_ChebyFilterOrder_); 
                      }

                    }
                  } // if (scfdg_ion_dyn_iter_ != 0)
                  else
                  {    
                      statusOFS << std::endl << " Calling CheFSI for static calculations  " << std::endl;
                    // 0th MD / Geometry Optimization step (or static calculation)        
                    if(outerIter == 1)
                    {
                      statusOFS << std::endl << " Calling First Chebyshev Iter  " << std::endl;
                      scfdg_FirstChebyStep(First_SCFDG_ChebyCycleNum_, First_SCFDG_ChebyFilterOrder_);
                    }
                    else if(outerIter > 1 &&     outerIter <= Second_SCFDG_ChebyOuterIter_)
                    {
                      statusOFS << std::endl << " Calling Second Stage Chebyshev Iter  " << std::endl;
                      scfdg_GeneralChebyStep(Second_SCFDG_ChebyCycleNum_, Second_SCFDG_ChebyFilterOrder_);
                    }
                    else
                    {  
                      if(SCFDG_use_comp_subspace_ == 1)
                      {
                        // Decide serial or parallel version here
                        if(SCFDG_comp_subspace_parallel_ == 0)
                        {  
                          statusOFS << std::endl << " Calling Complementary Subspace Strategy (serial subspace version)  " << std::endl;
                          scfdg_complementary_subspace_serial(General_SCFDG_ChebyFilterOrder_);
                        }
                        else
                        {
                          statusOFS << std::endl << " Calling Complementary Subspace Strategy (parallel subspace version)  " << std::endl;
                          scfdg_complementary_subspace_parallel(General_SCFDG_ChebyFilterOrder_);                     
                        }

                        // Now set the engaged flag 
                        SCFDG_comp_subspace_engaged_ = 1;

                      }
                      else
                      {
                        statusOFS << std::endl << " Calling General Chebyshev Iter  " << std::endl;
                        scfdg_GeneralChebyStep(General_SCFDG_ChebyCycleNum_, General_SCFDG_ChebyFilterOrder_); 
                      }


                    }
                  } // end of if(scfdg_ion_dyn_iter_ != 0)


                  MPI_Barrier( domain_.comm );
                  MPI_Barrier( domain_.rowComm );
                  MPI_Barrier( domain_.colComm );

                  GetTime( timeEnd );

                  if(SCFDG_comp_subspace_engaged_ == 1)
                    statusOFS << std::endl << " Total time for Complementary Subspace Method is " << timeEnd - timeSta << " [s]" << std::endl << std::endl;
                  else
                    statusOFS << std::endl << " Total time for diag DG matrix via Chebyshev filtering is " << timeEnd - timeSta << " [s]" << std::endl << std::endl;



                  DblNumVec& eigval = hamDG.EigVal();          
                  //for(Int i = 0; i < hamDG.NumStateTotal(); i ++)
                  //  statusOFS << setw(8) << i << setw(20) << '\t' << eigval[i] << std::endl;

                }
               else // call the ELSI interface and old Scalapack interface
                {
                  GetTime(timeSta);

                   Int sizeH = hamDG.NumBasisTotal(); // used for the size of Hamitonian. 
                  DblNumVec& eigval = hamDG.EigVal(); 
                  eigval.Resize( hamDG.NumStateTotal() );        

                  for( Int k = 0; k < numElem_[2]; k++ )
                    for( Int j = 0; j < numElem_[1]; j++ )
                      for( Int i = 0; i < numElem_[0]; i++ ){
                        Index3 key( i, j, k );
                        if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
                          const std::vector<Int>&  idx = hamDG.ElemBasisIdx()(i, j, k); 
                          DblNumMat& localCoef  = hamDG.EigvecCoef().LocalMap()[key];
                          localCoef.Resize( idx.size(), hamDG.NumStateTotal() );        
                        }
                      } 

                  scalapack::Descriptor descH;
                  if( contxt_ >= 0 ){
                    descH.Init( sizeH, sizeH, scaBlockSize_, scaBlockSize_, 
                        0, 0, contxt_ );
                  }

                  scalapack::ScaLAPACKMatrix<Real>  scaH, scaZ;

                  std::vector<Int> mpirankElemVec(dmCol_);
                  std::vector<Int> mpirankScaVec( numProcScaLAPACK_ );

                  // The processors in the first column are the source
                  for( Int i = 0; i < dmCol_; i++ ){
                    mpirankElemVec[i] = i * dmRow_;
                  }
                  // The first numProcScaLAPACK processors are the target
                  for( Int i = 0; i < numProcScaLAPACK_; i++ ){
                    mpirankScaVec[i] = i;
                  }

#if ( _DEBUGlevel_ >= 2 )
                  statusOFS << "mpirankElemVec = " << mpirankElemVec << std::endl;
                  statusOFS << "mpirankScaVec = " << mpirankScaVec << std::endl;
#endif

                  Real timeConversionSta, timeConversionEnd;

                  GetTime( timeConversionSta );
                  DistElemMatToScaMat2( hamDG.HMat(),     descH,
                      scaH, hamDG.ElemBasisIdx(), domain_.comm,
                      domain_.colComm, mpirankElemVec,
                      mpirankScaVec );
                  GetTime( timeConversionEnd );


#if ( _DEBUGlevel_ >= 1 )
                  statusOFS << " Time for converting from DistElemMat to ScaMat is " <<
                    timeConversionEnd - timeConversionSta << " [s]" << std::endl << std::endl;
#endif
                  if(contxt_ >= 0){

                  std::vector<Real> eigs(sizeH);
                  double * Smatrix = NULL;
                  GetTime( timeConversionSta );

                  // allocate memory for the scaZ. and call ELSI: ELPA

                  if( diagSolutionMethod_ == "scalapack"){
                     scalapack::Syevd('U', scaH, eigs, scaZ);
                  }
                  else // by default to use ELPA
                  {
#ifdef ELSI
                     scaZ.SetDescriptor(scaH.Desc());
                     c_elsi_ev_real(scaH.Data(), Smatrix, &eigs[0], scaZ.Data()); 
#endif
                  }
                  
                  GetTime( timeConversionEnd );
#if ( _DEBUGlevel_ >= 1 )
                  if( diagSolutionMethod_ == "scalapack"){
                      statusOFS << " Time for Scalapack::diag " <<
                          timeConversionEnd - timeConversionSta << " [s]" 
                          << std::endl << std::endl;
                  }
                  else
                  {
                      statusOFS << " Time for ELSI::ELPA  Diag " <<
                          timeConversionEnd - timeConversionSta << " [s]" 
                          << std::endl << std::endl;
                  }
#endif
                  for( Int i = 0; i < hamDG.NumStateTotal(); i++ )
                    eigval[i] = eigs[i];

                  } //if(contxt_ >= -1)

                  GetTime( timeConversionSta );
                  ScaMatToDistNumMat2( scaZ, hamDG.Density().Prtn(), 
                      hamDG.EigvecCoef(), hamDG.ElemBasisIdx(), domain_.comm,
                      domain_.colComm, mpirankElemVec, mpirankScaVec, 
                      hamDG.NumStateTotal() );
                  GetTime( timeConversionEnd );
#if ( _DEBUGlevel_ >= 1 )
                  statusOFS << " Time for converting from ScaMat to DistNumMat is " <<
                    timeConversionEnd - timeConversionSta << " [s]" << std::endl << std::endl;
#endif

                  GetTime( timeConversionSta );

                  for( Int k = 0; k < numElem_[2]; k++ )
                    for( Int j = 0; j < numElem_[1]; j++ )
                      for( Int i = 0; i < numElem_[0]; i++ ){
                        Index3 key( i, j, k );
                        if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
                          DblNumMat& localCoef  = hamDG.EigvecCoef().LocalMap()[key];
                          MPI_Bcast(localCoef.Data(), localCoef.m() * localCoef.n(), MPI_DOUBLE, 0, domain_.rowComm);
                        }
                      } 
                  GetTime( timeConversionEnd );
#if ( _DEBUGlevel_ >= 1 )
                  statusOFS << " Time for MPI_Bcast eigval and localCoef is " <<
                    timeConversionEnd - timeConversionSta << " [s]" << std::endl << std::endl;
#endif

                  MPI_Barrier( domain_.comm );
                  MPI_Barrier( domain_.rowComm );
                  MPI_Barrier( domain_.colComm );

                  GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
                  if( diagSolutionMethod_ == "scalapack"){
                  statusOFS << " Time for diag DG matrix via ScaLAPACK is " <<
                    timeEnd - timeSta << " [s]" << std::endl << std::endl;
                  }
                  else{
                  statusOFS << " Time for diag DG matrix via ELSI:ELPA is " <<
                    timeEnd - timeSta << " [s]" << std::endl << std::endl;
                  }
#endif

                  // Communicate the eigenvalues
                  Int mpirankScaSta = mpirankScaVec[0];
                  MPI_Bcast(eigval.Data(), hamDG.NumStateTotal(), MPI_DOUBLE, 
                      mpirankScaVec[0], domain_.comm);


                } // End of ELSI

              }// End of diagonalization routines

              // Post processing

              Evdw_ = 0.0;

              if(SCFDG_comp_subspace_engaged_ == 1)
              {
                // Calculate Harris energy without computing the occupations
                CalculateHarrisEnergy();

              }        
              else
              {        


                // Compute the occupation rate - specific smearing types dealt with within this function
                CalculateOccupationRate( hamDG.EigVal(), hamDG.OccupationRate() );

                // Compute the Harris energy functional.  
                // NOTE: In computing the Harris energy, the density and the
                // potential must be the INPUT density and potential without ANY
                // update.
                CalculateHarrisEnergy();
              }

              MPI_Barrier( domain_.comm );
              MPI_Barrier( domain_.rowComm );
              MPI_Barrier( domain_.colComm );



              // Calculate the new electron density

              // ~~**~~
              GetTime( timeSta );

              if(SCFDG_comp_subspace_engaged_ == 1)
              {
                // Density calculation for complementary subspace method
                statusOFS << std::endl << " Using complementary subspace method for electron density ... " << std::endl;

                Real GetTime_extra_sta, GetTime_extra_end;          
		Real GetTime_fine_sta, GetTime_fine_end;
		
                GetTime(GetTime_extra_sta);
                statusOFS << std::endl << " Forming diagonal blocks of density matrix : ";
                GetTime(GetTime_fine_sta);
		
                // Compute the diagonal blocks of the density matrix
                DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn> cheby_diag_dmat;  
                cheby_diag_dmat.Prtn()     = hamDG.HMat().Prtn();
                cheby_diag_dmat.SetComm(domain_.colComm);

                // Copy eigenvectors to temp bufer
                DblNumMat &eigvecs_local = (hamDG.EigvecCoef().LocalMap().begin())->second;

                DblNumMat temp_local_eig_vec;
                temp_local_eig_vec.Resize(eigvecs_local.m(), eigvecs_local.n());
                blas::Copy((eigvecs_local.m() * eigvecs_local.n()), eigvecs_local.Data(), 1, temp_local_eig_vec.Data(), 1);

                // First compute the X*X^T portion
                // Multiply out to obtain diagonal block of density matrix
                ElemMatKey diag_block_key = std::make_pair(my_cheby_eig_vec_key_, my_cheby_eig_vec_key_);
                cheby_diag_dmat.LocalMap()[diag_block_key].Resize( temp_local_eig_vec.m(),  temp_local_eig_vec.m());

                blas::Gemm( 'N', 'T', temp_local_eig_vec.m(), temp_local_eig_vec.m(), temp_local_eig_vec.n(),
                    1.0, 
                    temp_local_eig_vec.Data(), temp_local_eig_vec.m(), 
                    temp_local_eig_vec.Data(), temp_local_eig_vec.m(),
                    0.0, 
                    cheby_diag_dmat.LocalMap()[diag_block_key].Data(),  temp_local_eig_vec.m());

                GetTime(GetTime_fine_end);
		statusOFS << std::endl << " X * X^T computed in " << (GetTime_fine_end - GetTime_fine_sta) << " s.";
		
		GetTime(GetTime_fine_sta);
                if(SCFDG_comp_subspace_N_solve_ != 0)
		{
		  // Now compute the X * C portion
                 DblNumMat XC_mat;
                 XC_mat.Resize(eigvecs_local.m(), SCFDG_comp_subspace_N_solve_);

                 blas::Gemm( 'N', 'N', temp_local_eig_vec.m(), SCFDG_comp_subspace_N_solve_, temp_local_eig_vec.n(),
                             1.0, 
                             temp_local_eig_vec.Data(), temp_local_eig_vec.m(), 
                             SCFDG_comp_subspace_matC_.Data(), SCFDG_comp_subspace_matC_.m(),
                             0.0, 
                             XC_mat.Data(),  XC_mat.m());

                 // Subtract XC*XC^T from DM
                 blas::Gemm( 'N', 'T', XC_mat.m(), XC_mat.m(), XC_mat.n(),
                            -1.0, 
                             XC_mat.Data(), XC_mat.m(), 
                             XC_mat.Data(), XC_mat.m(),
                             1.0, 
                             cheby_diag_dmat.LocalMap()[diag_block_key].Data(),  temp_local_eig_vec.m());
		}
                GetTime(GetTime_fine_end);
		statusOFS << std::endl << " X*C and XC * (XC)^T computed in " << (GetTime_fine_end - GetTime_fine_sta) << " s.";
                
                
                GetTime(GetTime_extra_end);
                statusOFS << std::endl << " Total time for computing diagonal blocks of DM = " << (GetTime_extra_end - GetTime_extra_sta)  << " s." << std::endl ;
                statusOFS << std::endl;

                // Make the call evaluate this on the real space grid 
#ifdef GPU
                hamDG.CalculateDensityDM2(hamDG.Density(), hamDG.DensityLGL(), cheby_diag_dmat, true );
#else
                hamDG.CalculateDensityDM2(hamDG.Density(), hamDG.DensityLGL(), cheby_diag_dmat );
#endif

              }        
              else
              {        

                int temp_m = hamDG.NumBasisTotal() / (numElem_[0] * numElem_[1] * numElem_[2]); // Average no. of ALBs per element
                int temp_n = hamDG.NumStateTotal();
                if((Diag_SCFDG_by_Cheby_ == 1) && (temp_m < temp_n))
                {  
                  statusOFS << std::endl << " Using alternate routine for electron density: " << std::endl;

                  Real GetTime_extra_sta, GetTime_extra_end;                
                  GetTime(GetTime_extra_sta);
                  statusOFS << std::endl << " Forming diagonal blocks of density matrix ... ";

                  // Compute the diagonal blocks of the density matrix
                  DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn> cheby_diag_dmat;  
                  cheby_diag_dmat.Prtn()     = hamDG.HMat().Prtn();
                  cheby_diag_dmat.SetComm(domain_.colComm);

                  // Copy eigenvectors to temp bufer
                  DblNumMat &eigvecs_local = (hamDG.EigvecCoef().LocalMap().begin())->second;

                  DblNumMat scal_local_eig_vec;
                  scal_local_eig_vec.Resize(eigvecs_local.m(), eigvecs_local.n());
                  blas::Copy((eigvecs_local.m() * eigvecs_local.n()), eigvecs_local.Data(), 1, scal_local_eig_vec.Data(), 1);

                  // Scale temp buffer by occupation square root
                  for(int iter_scale = 0; iter_scale < eigvecs_local.n(); iter_scale ++)
                  {
                    blas::Scal(  scal_local_eig_vec.m(),  sqrt(hamDG.OccupationRate()[iter_scale]), scal_local_eig_vec.Data() + iter_scale * scal_local_eig_vec.m(), 1 );
                  }

                  // Multiply out to obtain diagonal block of density matrix
                  ElemMatKey diag_block_key = std::make_pair(my_cheby_eig_vec_key_, my_cheby_eig_vec_key_);
                  cheby_diag_dmat.LocalMap()[diag_block_key].Resize( scal_local_eig_vec.m(),  scal_local_eig_vec.m());

                  blas::Gemm( 'N', 'T', scal_local_eig_vec.m(), scal_local_eig_vec.m(), scal_local_eig_vec.n(),
                      1.0, 
                      scal_local_eig_vec.Data(), scal_local_eig_vec.m(), 
                      scal_local_eig_vec.Data(), scal_local_eig_vec.m(),
                      0.0, 
                      cheby_diag_dmat.LocalMap()[diag_block_key].Data(),  scal_local_eig_vec.m());

                  GetTime(GetTime_extra_end);
                  statusOFS << " Done. ( " << (GetTime_extra_end - GetTime_extra_sta)  << " s) " << std::endl ;

                  // Make the call evaluate this on the real space grid 
#ifdef GPU
                 hamDG.CalculateDensityDM2(hamDG.Density(), hamDG.DensityLGL(), cheby_diag_dmat, true );
#else
                 hamDG.CalculateDensityDM2(hamDG.Density(), hamDG.DensityLGL(), cheby_diag_dmat );
#endif

                }
                else
                {  

                  // FIXME 
                  // Do not need the conversion from column to row partition as well
                  hamDG.CalculateDensity( hamDG.Density(), hamDG.DensityLGL() );

                  // 2016/11/20: Add filtering of the density. Impacts
                  // convergence at the order of 1e-5 for the LiH dimer
                  // example and therefore is not activated
                  if(0){
                    DistFourier& fft = *distfftPtr_;
                    Int ntot      = fft.numGridTotal;
                    Int ntotLocal = fft.numGridLocal;

                    DblNumVec  tempVecLocal;
                    DistNumVecToDistRowVec(
                        hamDG.Density(),
                        tempVecLocal,
                        domain_.numGridFine,
                        numElem_,
                        fft.localNzStart,
                        fft.localNz,
                        fft.isInGrid,
                        domain_.colComm );

                    if( fft.isInGrid ){
                      for( Int i = 0; i < ntotLocal; i++ ){
                        fft.inputComplexVecLocal(i) = Complex( 
                            tempVecLocal(i), 0.0 );
                      }

                      fftw_execute( fft.forwardPlan );

                      // Filter out high frequency modes
                      for( Int i = 0; i < ntotLocal; i++ ){
                        if( fft.gkkLocal(i) > std::pow(densityGridFactor_,2.0) * ecutWavefunction_ ){
                          fft.outputComplexVecLocal(i) = Z_ZERO;
                        }
                      }

                      fftw_execute( fft.backwardPlan );


                      for( Int i = 0; i < ntotLocal; i++ ){
                        tempVecLocal(i) = fft.inputComplexVecLocal(i).real() / ntot;
                      }
                    }

                    DistRowVecToDistNumVec( 
                        tempVecLocal,
                        hamDG.Density(),
                        domain_.numGridFine,
                        numElem_,
                        fft.localNzStart,
                        fft.localNz,
                        fft.isInGrid,
                        domain_.colComm );


                    // Compute the sum of density and normalize again.
                    Real sumRhoLocal = 0.0, sumRho = 0.0;
                    for( Int k = 0; k < numElem_[2]; k++ )
                      for( Int j = 0; j < numElem_[1]; j++ )
                        for( Int i = 0; i < numElem_[0]; i++ ){
                          Index3 key( i, j, k );
                          if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
                            DblNumVec& localRho = hamDG.Density().LocalMap()[key];

                            Real* ptrRho = localRho.Data();
                            for( Int p = 0; p < localRho.Size(); p++ ){
                              sumRhoLocal += ptrRho[p];
                            }
                          }
                        }

                    sumRhoLocal *= domain_.Volume() / domain_.NumGridTotalFine(); 
                    mpi::Allreduce( &sumRhoLocal, &sumRho, 1, MPI_SUM, domain_.colComm );

#if ( _DEBUGlevel_ >= 0 )
                    statusOFS << std::endl;
                    Print( statusOFS, "Sum Rho on uniform grid (after Fourier filtering) = ", sumRho );
                    statusOFS << std::endl;
#endif
                    Real fac = hamDG.NumSpin() * hamDG.NumOccupiedState() / sumRho;
                    sumRhoLocal = 0.0, sumRho = 0.0;
                    for( Int k = 0; k < numElem_[2]; k++ )
                      for( Int j = 0; j < numElem_[1]; j++ )
                        for( Int i = 0; i < numElem_[0]; i++ ){
                          Index3 key( i, j, k );
                          if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
                            DblNumVec& localRho = hamDG.Density().LocalMap()[key];
                            blas::Scal(  localRho.Size(),  fac, localRho.Data(), 1 );

                            Real* ptrRho = localRho.Data();
                            for( Int p = 0; p < localRho.Size(); p++ ){
                              sumRhoLocal += ptrRho[p];
                            }
                          }
                        }

                    sumRhoLocal *= domain_.Volume() / domain_.NumGridTotalFine(); 
                    mpi::Allreduce( &sumRhoLocal, &sumRho, 1, MPI_SUM, domain_.colComm );

#if ( _DEBUGlevel_ >= 0 )
                    statusOFS << std::endl;
                    Print( statusOFS, "Sum Rho on uniform grid (after normalization again) = ", sumRho );
                    statusOFS << std::endl;
#endif
                  }
                }

              }        

              MPI_Barrier( domain_.comm );
              MPI_Barrier( domain_.rowComm );
              MPI_Barrier( domain_.colComm );

              GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
              statusOFS << " Time for computing density in the global domain is " <<
                timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

              // Update the output potential, and the KS and second order accurate
              // energy
              {
                // Update the Hartree energy and the exchange correlation energy and
                // potential for computing the KS energy and the second order
                // energy.
                // NOTE Vtot should not be updated until finishing the computation
                // of the energies.

                if( XCType_ == "XC_GGA_XC_PBE" ){
                  GetTime( timeSta );
                  hamDG.CalculateGradDensity(  *distfftPtr_ );
                  GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
                  statusOFS << " Time for calculating gradient of density is " <<
                    timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
                }

                GetTime( timeSta );
                hamDG.CalculateXC( Exc_, hamDG.Epsxc(), hamDG.Vxc(), *distfftPtr_ );
                GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
                statusOFS << " Time for computing Exc in the global domain is " <<
                  timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

                GetTime( timeSta );

                hamDG.CalculateHartree( hamDG.Vhart(), *distfftPtr_ );

                GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
                statusOFS << " Time for computing Vhart in the global domain is " <<
                  timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

                // Compute the second order accurate energy functional.

                // Compute the second order accurate energy functional.
                // NOTE: In computing the second order energy, the density and the
                // potential must be the OUTPUT density and potential without ANY
                // MIXING.
                CalculateSecondOrderEnergy();

                // Compute the KS energy 

                GetTime( timeSta );

                CalculateKSEnergy();

                GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
                statusOFS << " Time for computing KSEnergy in the global domain is " <<
                  timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

                // Update the total potential AFTER updating the energy

                // No external potential

                // Compute the new total potential

                GetTime( timeSta );

                hamDG.CalculateVtot( hamDG.Vtot() );

                GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
                statusOFS << " Time for computing Vtot in the global domain is " <<
                  timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
              }

              // Compute the force at every step
              if( esdfParam.isCalculateForceEachSCF ){

                // Compute force
                GetTime( timeSta );

                if(SCFDG_comp_subspace_engaged_ == false)
                {
                  if(1)
                  {
                    statusOFS << std::endl << " Computing forces using eigenvectors ... " << std::endl;
                    hamDG.CalculateForce( *distfftPtr_ );
                  }
                  else
                  {         
                    // Alternate (highly unusual) routine for debugging purposes
                    // Compute the Full DM (from eigenvectors) and call the PEXSI force evaluator

                    double extra_timeSta, extra_timeEnd;

                    statusOFS << std::endl << " Computing forces using Density Matrix ... ";
                    statusOFS << std::endl << " Computing full Density Matrix from eigenvectors ...";
                    GetTime(extra_timeSta);

                    distDMMat_.Prtn()     = hamDG.HMat().Prtn();

                    // Compute the full DM 
                    scfdg_compute_fullDM();

                    GetTime(extra_timeEnd);

                    statusOFS << std::endl << " Computation took " << (extra_timeEnd - extra_timeSta) << " s." << std::endl;

                    // Call the PEXSI force evaluator
                    hamDG.CalculateForceDM( *distfftPtr_, distDMMat_ );        
                  }



                }
                else
                {
                  double extra_timeSta, extra_timeEnd;

                  statusOFS << std::endl << " Computing forces using Density Matrix ... ";

                  statusOFS << std::endl << " Computing full Density Matrix for Complementary Subspace method ...";
                  GetTime(extra_timeSta);

                  // Compute the full DM in the complementary subspace method
                  scfdg_complementary_subspace_compute_fullDM();

                  GetTime(extra_timeEnd);

                  statusOFS << std::endl << " DM Computation took " << (extra_timeEnd - extra_timeSta) << " s." << std::endl;

                  // Call the PEXSI force evaluator
                  hamDG.CalculateForceDM( *distfftPtr_, distDMMat_ );        
                }



                GetTime( timeEnd );
                statusOFS << " Time for computing the force is " <<
                  timeEnd - timeSta << " [s]" << std::endl << std::endl;

                // Print out the force
                // Only master processor output information containing all atoms
                if( mpirank == 0 ){
                  PrintBlock( statusOFS, "Atomic Force" );
                  {
                    Point3 forceCM(0.0, 0.0, 0.0);
                    std::vector<Atom>& atomList = hamDG.AtomList();
                    Int numAtom = atomList.size();
                    for( Int a = 0; a < numAtom; a++ ){
                      Print( statusOFS, "atom", a, "force", atomList[a].force );
                      forceCM += atomList[a].force;
                    }
                    statusOFS << std::endl;
                    Print( statusOFS, "force for centroid: ", forceCM );
                    statusOFS << std::endl;
                  }
                }
              }

              // Compute the a posteriori error estimator at every step
              // FIXME This is not used when intra-element parallelization is
              // used.
              if( esdfParam.isCalculateAPosterioriEachSCF && 0 )
              {
                GetTime( timeSta );
                DblNumTns  eta2Total, eta2Residual, eta2GradJump, eta2Jump;
                hamDG.CalculateAPosterioriError( 
                    eta2Total, eta2Residual, eta2GradJump, eta2Jump );
                GetTime( timeEnd );
                statusOFS << " Time for computing the a posteriori error is " <<
                  timeEnd - timeSta << " [s]" << std::endl << std::endl;

                // Only master processor output information containing all atoms
                if( mpirank == 0 ){
                  PrintBlock( statusOFS, "A Posteriori error" );
                  {
                    statusOFS << std::endl << "Total a posteriori error:" << std::endl;
                    statusOFS << eta2Total << std::endl;
                    statusOFS << std::endl << "Residual term:" << std::endl;
                    statusOFS << eta2Residual << std::endl;
                    statusOFS << std::endl << "Jump of gradient term:" << std::endl;
                    statusOFS << eta2GradJump << std::endl;
                    statusOFS << std::endl << "Jump of function value term:" << std::endl;
                    statusOFS << eta2Jump << std::endl;
                  }
                }
              }
            }


            // Method 2: Using the pole expansion and selected inversion (PEXSI) method
            // FIXME Currently it is assumed that all processors used by DG will be used by PEXSI.
#ifdef _USE_PEXSI_
/*
            // The following version is with intra-element parallelization
            if( solutionMethod_ == "pexsi" ){
              Real timePEXSISta, timePEXSIEnd;
              GetTime( timePEXSISta );

              Real numElectronExact = hamDG.NumOccupiedState() * hamDG.NumSpin();
              Real muMinInertia, muMaxInertia;
              Real muPEXSI, numElectronPEXSI;
              Int numTotalInertiaIter = 0, numTotalPEXSIIter = 0;

              std::vector<Int> mpirankSparseVec( numProcPEXSICommCol_ );

              // FIXME 
              // Currently, only the first processor column participate in the
              // communication between PEXSI and DGDFT For the first processor
              // column involved in PEXSI, the first numProcPEXSICommCol_
              // processors are involved in the data communication between PEXSI
              // and DGDFT

              for( Int i = 0; i < numProcPEXSICommCol_; i++ ){
                mpirankSparseVec[i] = i;
              }

#if ( _DEBUGlevel_ >= 1 )
              statusOFS << "mpirankSparseVec = " << mpirankSparseVec << std::endl;
#endif

              Int info;

              // Temporary matrices 
              DistSparseMatrix<Real>  HSparseMat;
              DistSparseMatrix<Real>  DMSparseMat;
              DistSparseMatrix<Real>  EDMSparseMat;
              DistSparseMatrix<Real>  FDMSparseMat;

              if( mpirankRow == 0 ){

                // Convert the DG matrix into the distributed CSC format

                GetTime(timeSta);
                DistElemMatToDistSparseMat3( 
                    hamDG.HMat(),
                    hamDG.NumBasisTotal(),
                    HSparseMat,
                    hamDG.ElemBasisIdx(),
                    domain_.colComm,
                    mpirankSparseVec );
                GetTime(timeEnd);

#if ( _DEBUGlevel_ >= 0 )
                statusOFS << "Time for converting the DG matrix to DistSparseMatrix format is " <<
                  timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif


#if ( _DEBUGlevel_ >= 0 )
                if( mpirankCol < numProcPEXSICommCol_ ){
                  statusOFS << "H.size = " << HSparseMat.size << std::endl;
                  statusOFS << "H.nnz  = " << HSparseMat.nnz << std::endl;
                  statusOFS << "H.nnzLocal  = " << HSparseMat.nnzLocal << std::endl;
                  statusOFS << "H.colptrLocal.m() = " << HSparseMat.colptrLocal.m() << std::endl;
                  statusOFS << "H.rowindLocal.m() = " << HSparseMat.rowindLocal.m() << std::endl;
                  statusOFS << "H.nzvalLocal.m() = " << HSparseMat.nzvalLocal.m() << std::endl;
                }
#endif
              }


              if( (mpirankRow < numProcPEXSICommRow_) && (mpirankCol < numProcPEXSICommCol_) ){
                // Load the matrices into PEXSI.  
                // Only the processors with mpirankCol == 0 need to carry the
                // nonzero values of HSparseMat

#if ( _DEBUGlevel_ >= 0 )
                statusOFS << "numProcPEXSICommRow_ = " << numProcPEXSICommRow_ << std::endl;
                statusOFS << "numProcPEXSICommCol_ = " << numProcPEXSICommCol_ << std::endl;
                statusOFS << "mpirankRow = " << mpirankRow << std::endl;
                statusOFS << "mpirankCol = " << mpirankCol << std::endl;
#endif


                GetTime( timeSta );
                PPEXSILoadRealHSMatrix(
                    pexsiPlan_,
                    pexsiOptions_,
                    HSparseMat.size,
                    HSparseMat.nnz,
                    HSparseMat.nnzLocal,
                    HSparseMat.colptrLocal.m() - 1,
                    HSparseMat.colptrLocal.Data(),
                    HSparseMat.rowindLocal.Data(),
                    HSparseMat.nzvalLocal.Data(),
                    1,  // isSIdentity
                    NULL,
                    &info );
                GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
                statusOFS << "Time for loading the matrix into PEXSI is " <<
                  timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

                if( info != 0 ){
                  std::ostringstream msg;
                  msg 
                    << "PEXSI loading H matrix returns info " << info << std::endl;
                  ErrorHandling( msg.str().c_str() );
                }

                // PEXSI solver

                {
                  if( outerIter >= inertiaCountSteps_ ){
                    pexsiOptions_.isInertiaCount = 0;
                  }
                  // Note: Heuristics strategy for dynamically adjusting the
                  // tolerance
                  pexsiOptions_.muInertiaTolerance = 
                    std::min( std::max( muInertiaToleranceTarget_, 0.1 * scfOuterNorm_ ), 0.01 );
                  pexsiOptions_.numElectronPEXSITolerance = 
                    std::min( std::max( numElectronPEXSIToleranceTarget_, 1.0 * scfOuterNorm_ ), 0.5 );

                  // Only perform symbolic factorization for the first outer SCF. 
                  // Reuse the previous Fermi energy as the initial guess for mu.
                  if( outerIter == 1 ){
                    pexsiOptions_.isSymbolicFactorize = 1;
                    pexsiOptions_.mu0 = 0.5 * (pexsiOptions_.muMin0 + pexsiOptions_.muMax0);
                  }
                  else{
                    pexsiOptions_.isSymbolicFactorize = 0;
                    pexsiOptions_.mu0 = fermi_;
                  }

                  statusOFS << std::endl 
                    << "muInertiaTolerance        = " << pexsiOptions_.muInertiaTolerance << std::endl
                    << "numElectronPEXSITolerance = " << pexsiOptions_.numElectronPEXSITolerance << std::endl
                    << "Symbolic factorization    =  " << pexsiOptions_.isSymbolicFactorize << std::endl;
                }


                GetTime( timeSta );
                // Old version of PEXSI driver, uses inertia counting + Newton's iteration
                if(0){
                  PPEXSIDFTDriver(
                      pexsiPlan_,
                      pexsiOptions_,
                      numElectronExact,
                      &muPEXSI,
                      &numElectronPEXSI,         
                      &muMinInertia,              
                      &muMaxInertia,             
                      &numTotalInertiaIter,
                      &numTotalPEXSIIter,
                      &info );
                }

                // New version of PEXSI driver, uses inertia count + pole update
                // strategy. No Newton's iteration
                if(1){
                  PPEXSIDFTDriver2(
                      pexsiPlan_,
                      pexsiOptions_,
                      numElectronExact,
                      &muPEXSI,
                      &numElectronPEXSI,         
                      &muMinInertia,              
                      &muMaxInertia,             
                      &numTotalInertiaIter,
                      &info );
                }
                GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
                statusOFS << "Time for the main PEXSI Driver is " <<
                  timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

                if( info != 0 ){
                  std::ostringstream msg;
                  msg 
                    << "PEXSI main driver returns info " << info << std::endl;
                  ErrorHandling( msg.str().c_str() );
                }

                // Update the fermi level 
                fermi_ = muPEXSI;

                // Heuristics for the next step
                pexsiOptions_.muMin0 = muMinInertia - 5.0 * pexsiOptions_.temperature;
                pexsiOptions_.muMax0 = muMaxInertia + 5.0 * pexsiOptions_.temperature;

                // Retrieve the PEXSI data

                if( mpirankRow == 0 ){
                  Real totalEnergyH, totalEnergyS, totalFreeEnergy;

                  GetTime( timeSta );

                  CopyPattern( HSparseMat, DMSparseMat );
                  CopyPattern( HSparseMat, EDMSparseMat );
                  CopyPattern( HSparseMat, FDMSparseMat );

                  PPEXSIRetrieveRealDFTMatrix(
                      pexsiPlan_,
                      DMSparseMat.nzvalLocal.Data(),
                      EDMSparseMat.nzvalLocal.Data(),
                      FDMSparseMat.nzvalLocal.Data(),
                      &totalEnergyH,
                      &totalEnergyS,
                      &totalFreeEnergy,
                      &info );
                  GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
                  statusOFS << "Time for retrieving PEXSI data is " <<
                    timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

                  statusOFS << std::endl
                    << "Results obtained from PEXSI:" << std::endl
                    << "Total energy (H*DM)         = " << totalEnergyH << std::endl
                    << "Total energy (S*EDM)        = " << totalEnergyS << std::endl
                    << "Total free energy           = " << totalFreeEnergy << std::endl 
                    << "InertiaIter                 = " << numTotalInertiaIter << std::endl
                    << "PEXSIIter                   = " <<  numTotalPEXSIIter << std::endl
                    << "mu                          = " << muPEXSI << std::endl
                    << "numElectron                 = " << numElectronPEXSI << std::endl 
                    << std::endl;

                  if( info != 0 ){
                    std::ostringstream msg;
                    msg 
                      << "PEXSI data retrieval returns info " << info << std::endl;
                    ErrorHandling( msg.str().c_str() );
                  }
                }
              } // if( mpirank < numProcTotalPEXSI_ )

              // Broadcast the Fermi level
              MPI_Bcast( &fermi_, 1, MPI_DOUBLE, 0, domain_.comm );

              if( mpirankRow == 0 )
              {
                GetTime(timeSta);
                // Convert the density matrix from DistSparseMatrix format to the
                // DistElemMat format
                DistSparseMatToDistElemMat3(
                    DMSparseMat,
                    hamDG.NumBasisTotal(),
                    hamDG.HMat().Prtn(),
                    distDMMat_,
                    hamDG.ElemBasisIdx(),
                    hamDG.ElemBasisInvIdx(),
                    domain_.colComm,
                    mpirankSparseVec );


                // Convert the energy density matrix from DistSparseMatrix
                // format to the DistElemMat format
                DistSparseMatToDistElemMat3(
                    EDMSparseMat,
                    hamDG.NumBasisTotal(),
                    hamDG.HMat().Prtn(),
                    distEDMMat_,
                    hamDG.ElemBasisIdx(),
                    hamDG.ElemBasisInvIdx(),
                    domain_.colComm,
                    mpirankSparseVec );


                // Convert the free energy density matrix from DistSparseMatrix
                // format to the DistElemMat format
                DistSparseMatToDistElemMat3(
                    FDMSparseMat,
                    hamDG.NumBasisTotal(),
                    hamDG.HMat().Prtn(),
                    distFDMMat_,
                    hamDG.ElemBasisIdx(),
                    hamDG.ElemBasisInvIdx(),
                    domain_.colComm,
                    mpirankSparseVec );
                GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
                statusOFS << "Time for converting the DistSparseMatrices to DistElemMat " << 
                  "for post-processing is " <<
                  timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
              }

              // Broadcast the distElemMat matrices
              // FIXME this is not a memory efficient implementation
              GetTime(timeSta);
              {
                Int sstrSize;
                std::vector<char> sstr;
                if( mpirankRow == 0 ){
                  std::stringstream distElemMatStream;
                  Int cnt = 0;
                  for( typename std::map<ElemMatKey, DblNumMat >::iterator mi  = distDMMat_.LocalMap().begin();
                      mi != distDMMat_.LocalMap().end(); mi++ ){
                    cnt++;
                  } // for (mi)
                  serialize( cnt, distElemMatStream, NO_MASK );
                  for( typename std::map<ElemMatKey, DblNumMat >::iterator mi  = distDMMat_.LocalMap().begin();
                      mi != distDMMat_.LocalMap().end(); mi++ ){
                    ElemMatKey key = (*mi).first;
                    serialize( key, distElemMatStream, NO_MASK );
                    serialize( distDMMat_.LocalMap()[key], distElemMatStream, NO_MASK );
                    serialize( distEDMMat_.LocalMap()[key], distElemMatStream, NO_MASK ); 
                    serialize( distFDMMat_.LocalMap()[key], distElemMatStream, NO_MASK ); 
                  } // for (mi)
                  sstr.resize( Size( distElemMatStream ) );
                  distElemMatStream.read( &sstr[0], sstr.size() );
                  sstrSize = sstr.size();
                }

                MPI_Bcast( &sstrSize, 1, MPI_INT, 0, domain_.rowComm );
                sstr.resize( sstrSize );
                MPI_Bcast( &sstr[0], sstrSize, MPI_BYTE, 0, domain_.rowComm );

                if( mpirankRow != 0 ){
                  std::stringstream distElemMatStream;
                  distElemMatStream.write( &sstr[0], sstrSize );
                  Int cnt;
                  deserialize( cnt, distElemMatStream, NO_MASK );
                  for( Int i = 0; i < cnt; i++ ){
                    ElemMatKey key;
                    DblNumMat mat;
                    deserialize( key, distElemMatStream, NO_MASK );
                    deserialize( mat, distElemMatStream, NO_MASK );
                    distDMMat_.LocalMap()[key] = mat;
                    deserialize( mat, distElemMatStream, NO_MASK );
                    distEDMMat_.LocalMap()[key] = mat;
                    deserialize( mat, distElemMatStream, NO_MASK );
                    distFDMMat_.LocalMap()[key] = mat;
                  } // for (mi)
                }
              }
              GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
              statusOFS << "Time for broadcasting the density matrix for post-processing is " <<
                timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

              Evdw_ = 0.0;

              // Compute the Harris energy functional.  
              // NOTE: In computing the Harris energy, the density and the
              // potential must be the INPUT density and potential without ANY
              // update.
              GetTime( timeSta );
              CalculateHarrisEnergyDM( distFDMMat_ );
              GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
              statusOFS << "Time for calculating the Harris energy is " <<
                timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

              // Evaluate the electron density

              for( Int k = 0; k < numElem_[2]; k++ )
                for( Int j = 0; j < numElem_[1]; j++ )
                  for( Int i = 0; i < numElem_[0]; i++ ){
                    Index3 key( i, j, k );
                    if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
                      DblNumVec&  density      = hamDG.Density().LocalMap()[key];
                    } // own this element
                  } // for (i)


              GetTime( timeSta );
              hamDG.CalculateDensityDM2( 
                  hamDG.Density(), hamDG.DensityLGL(), distDMMat_ );
              MPI_Barrier( domain_.comm );
              GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
              statusOFS << "Time for computing density in the global domain is " <<
                timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

              for( Int k = 0; k < numElem_[2]; k++ )
                for( Int j = 0; j < numElem_[1]; j++ )
                  for( Int i = 0; i < numElem_[0]; i++ ){
                    Index3 key( i, j, k );
                    if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
                      DblNumVec&  density      = hamDG.Density().LocalMap()[key];
                    } // own this element
                  } // for (i)


              // Update the output potential, and the KS and second order accurate
              // energy
              GetTime(timeSta);
              {
                // Update the Hartree energy and the exchange correlation energy and
                // potential for computing the KS energy and the second order
                // energy.
                // NOTE Vtot should not be updated until finishing the computation
                // of the energies.

                if( XCType_ == "XC_GGA_XC_PBE" ){
                  GetTime( timeSta );
                  hamDG.CalculateGradDensity(  *distfftPtr_ );
                  GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
                  statusOFS << "Time for calculating gradient of density is " <<
                    timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
                }

                hamDG.CalculateXC( Exc_, hamDG.Epsxc(), hamDG.Vxc(), *distfftPtr_ );

                hamDG.CalculateHartree( hamDG.Vhart(), *distfftPtr_ );

                // Compute the second order accurate energy functional.
                // NOTE: In computing the second order energy, the density and the
                // potential must be the OUTPUT density and potential without ANY
                // MIXING.
                //        CalculateSecondOrderEnergy();

                // Compute the KS energy 
                CalculateKSEnergyDM( 
                    distEDMMat_, distFDMMat_ );

                // Update the total potential AFTER updating the energy

                // No external potential

                // Compute the new total potential

                hamDG.CalculateVtot( hamDG.Vtot() );

              }
              GetTime(timeEnd);
#if ( _DEBUGlevel_ >= 0 )
              statusOFS << "Time for computing the potential is " <<
                timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

              // Compute the force at every step
              //      if( esdfParam.isCalculateForceEachSCF ){
              //        // Compute force
              //        GetTime( timeSta );
              //        hamDG.CalculateForceDM( *distfftPtr_, distDMMat_ );
              //        GetTime( timeEnd );
              //        statusOFS << "Time for computing the force is " <<
              //          timeEnd - timeSta << " [s]" << std::endl << std::endl;
              //
              //        // Print out the force
              //        // Only master processor output information containing all atoms
              //        if( mpirank == 0 ){
              //          PrintBlock( statusOFS, "Atomic Force" );
              //          {
              //            Point3 forceCM(0.0, 0.0, 0.0);
              //            std::vector<Atom>& atomList = hamDG.AtomList();
              //            Int numAtom = atomList.size();
              //            for( Int a = 0; a < numAtom; a++ ){
              //              Print( statusOFS, "atom", a, "force", atomList[a].force );
              //              forceCM += atomList[a].force;
              //            }
              //            statusOFS << std::endl;
              //            Print( statusOFS, "force for centroid: ", forceCM );
              //            statusOFS << std::endl;
              //          }
              //        }
              //      }

              // TODO Evaluate the a posteriori error estimator

              GetTime( timePEXSIEnd );
#if ( _DEBUGlevel_ >= 0 )
              statusOFS << "Time for PEXSI evaluation is " <<
                timePEXSIEnd - timePEXSISta << " [s]" << std::endl << std::endl;
#endif
            } //if( solutionMethod_ == "pexsi" )
*/

            // The following version is with intra-element parallelization
            DistDblNumVec VtotHist; // check check
	    // check check
	    Real difNumElectron = 0.0;
            if( solutionMethod_ == "pexsi" ){

            // Initialize the history of vtot , check check
            for( Int k=0; k< numElem_[2]; k++ )
              for( Int j=0; j< numElem_[1]; j++ )
                for( Int i=0; i< numElem_[0]; i++ ) {
                  Index3 key = Index3(i,j,k);
                  if( distEigSolPtr_->Prtn().Owner(key) == (mpirank / dmRow_) ){
                    DistDblNumVec& vtotCur = hamDG.Vtot();
                    VtotHist.LocalMap()[key] = vtotCur.LocalMap()[key];
                    //VtotHist.LocalMap()[key] = mixInnerSave_.LocalMap()[key];
                  } // owns this element
                } // for (i)



              Real timePEXSISta, timePEXSIEnd;
              GetTime( timePEXSISta );

              Real numElectronExact = hamDG.NumOccupiedState() * hamDG.NumSpin();
              Real muMinInertia, muMaxInertia;
              Real muPEXSI, numElectronPEXSI;
              Int numTotalInertiaIter = 0, numTotalPEXSIIter = 0;

              std::vector<Int> mpirankSparseVec( numProcPEXSICommCol_ );

              // FIXME 
              // Currently, only the first processor column participate in the
              // communication between PEXSI and DGDFT For the first processor
              // column involved in PEXSI, the first numProcPEXSICommCol_
              // processors are involved in the data communication between PEXSI
              // and DGDFT

              for( Int i = 0; i < numProcPEXSICommCol_; i++ ){
                mpirankSparseVec[i] = i;
              }

#if ( _DEBUGlevel_ >= 1 )
              statusOFS << "mpirankSparseVec = " << mpirankSparseVec << std::endl;
#endif

              Int info;

              // Temporary matrices 
              DistSparseMatrix<Real>  HSparseMat;
              DistSparseMatrix<Real>  DMSparseMat;
              DistSparseMatrix<Real>  EDMSparseMat;
              DistSparseMatrix<Real>  FDMSparseMat;

              if( mpirankRow == 0 ){

                // Convert the DG matrix into the distributed CSC format

                GetTime(timeSta);
                DistElemMatToDistSparseMat3( 
                    hamDG.HMat(),
                    hamDG.NumBasisTotal(),
                    HSparseMat,
                    hamDG.ElemBasisIdx(),
                    domain_.colComm,
                    mpirankSparseVec );
                GetTime(timeEnd);

#if ( _DEBUGlevel_ >= 0 )
                statusOFS << " Time for converting the DG matrix to DistSparseMatrix format is " <<
                  timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif


#if ( _DEBUGlevel_ >= 0 )
                if( mpirankCol < numProcPEXSICommCol_ ){
                  statusOFS << "H.size = " << HSparseMat.size << std::endl;
                  statusOFS << "H.nnz  = " << HSparseMat.nnz << std::endl;
                  statusOFS << "H.nnzLocal  = " << HSparseMat.nnzLocal << std::endl;
                  statusOFS << "H.colptrLocal.m() = " << HSparseMat.colptrLocal.m() << std::endl;
                  statusOFS << "H.rowindLocal.m() = " << HSparseMat.rowindLocal.m() << std::endl;
                  statusOFS << "H.nzvalLocal.m() = " << HSparseMat.nzvalLocal.m() << std::endl;
                }
#endif
              }

              // So energy must be obtained from DM as in totalEnergyH
              // and free energy is nothing but energy..
              Real totalEnergyH, totalFreeEnergy;
              if( (mpirankRow < numProcPEXSICommRow_) && (mpirankCol < numProcPEXSICommCol_) )
              {

                // Load the matrices into PEXSI.  
                // Only the processors with mpirankCol == 0 need to carry the
                // nonzero values of HSparseMat

#if ( _DEBUGlevel_ >= 0 )
                statusOFS << "numProcPEXSICommRow_ = " << numProcPEXSICommRow_ << std::endl;
                statusOFS << "numProcPEXSICommCol_ = " << numProcPEXSICommCol_ << std::endl;
                statusOFS << "mpirankRow = " << mpirankRow << std::endl;
                statusOFS << "mpirankCol = " << mpirankCol << std::endl;
#endif


                GetTime( timeSta );

#ifndef ELSI                
                PPEXSILoadRealHSMatrix(
                    pexsiPlan_,
                    pexsiOptions_,
                    HSparseMat.size,
                    HSparseMat.nnz,
                    HSparseMat.nnzLocal,
                    HSparseMat.colptrLocal.m() - 1,
                    HSparseMat.colptrLocal.Data(),
                    HSparseMat.rowindLocal.Data(),
                    HSparseMat.nzvalLocal.Data(),
                    1,  // isSIdentity
                    NULL,
                    &info );
                GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
                statusOFS << " Time for loading the matrix into PEXSI is " <<
                  timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

                if( info != 0 ){
                  std::ostringstream msg;
                  msg 
                    << "PEXSI loading H matrix returns info " << info << std::endl;
                  ErrorHandling( msg.str().c_str() );
                }
#endif           

                // PEXSI solver

                {
                  if( outerIter >= inertiaCountSteps_ ){
                    pexsiOptions_.isInertiaCount = 0;
                  }
                  // Note: Heuristics strategy for dynamically adjusting the
                  // tolerance
                  pexsiOptions_.muInertiaTolerance = 
                    std::min( std::max( muInertiaToleranceTarget_, 0.1 * scfOuterNorm_ ), 0.01 );
                  pexsiOptions_.numElectronPEXSITolerance = 
                    std::min( std::max( numElectronPEXSIToleranceTarget_, 1.0 * scfOuterNorm_ ), 0.5 );

                  // Only perform symbolic factorization for the first outer SCF. 
                  // Reuse the previous Fermi energy as the initial guess for mu.
                  if( outerIter == 1 ){
                    pexsiOptions_.isSymbolicFactorize = 1;
                    pexsiOptions_.mu0 = 0.5 * (pexsiOptions_.muMin0 + pexsiOptions_.muMax0);
                  }
                  else{
                    pexsiOptions_.isSymbolicFactorize = 0;
                    pexsiOptions_.mu0 = fermi_;
                  }

                  statusOFS << std::endl 
                    << "muInertiaTolerance        = " << pexsiOptions_.muInertiaTolerance << std::endl
                    << "numElectronPEXSITolerance = " << pexsiOptions_.numElectronPEXSITolerance << std::endl
                    << "Symbolic factorization    =  " << pexsiOptions_.isSymbolicFactorize << std::endl;
                }
#ifdef ELSI
#if ( _DEBUGlevel_ >= 0 )
                statusOFS << std::endl << "ELSI PEXSI set sparsity start" << std::endl<< std::flush;
#endif
#endif

                // ///////////////////////////////////////////////////////////////////////
                // ///////////////////////////////////////////////////////////////////////
#ifdef ELSI
                c_elsi_set_sparsity( HSparseMat.nnz,
                                   HSparseMat.nnzLocal,
                                   HSparseMat.colptrLocal.m() - 1,
                                   HSparseMat.rowindLocal.Data(),
                                   HSparseMat.colptrLocal.Data() );

                c_elsi_customize_pexsi(pexsiOptions_.temperature,
                                       pexsiOptions_.gap,
                                       pexsiOptions_.deltaE,
                                       pexsiOptions_.numPole,
                                       numProcPEXSICommCol_,  // # n_procs_per_pole
                                       pexsiOptions_.maxPEXSIIter,
                                       pexsiOptions_.muMin0,
                                       pexsiOptions_.muMax0,
                                       pexsiOptions_.mu0,
                                       pexsiOptions_.muInertiaTolerance,
                                       pexsiOptions_.muInertiaExpansion,
                                       pexsiOptions_.muPEXSISafeGuard,
                                       pexsiOptions_.numElectronPEXSITolerance,
                                       pexsiOptions_.matrixType,
                                       pexsiOptions_.isSymbolicFactorize,
                                       pexsiOptions_.ordering,
                                       pexsiOptions_.npSymbFact,
                                       pexsiOptions_.verbosity);

#if ( _DEBUGlevel_ >= 0 )
                statusOFS << std::endl << "ELSI PEXSI Customize Done " << std::endl;
#endif

                if( mpirankRow == 0 )
                   CopyPattern( HSparseMat, DMSparseMat );
                statusOFS << std::endl << "ELSI PEXSI Copy pattern done" << std::endl;
                c_elsi_dm_real_sparse(HSparseMat.nzvalLocal.Data(), NULL, DMSparseMat.nzvalLocal.Data());

                GetTime( timeEnd );
                statusOFS << std::endl << "ELSI PEXSI real sparse done" << std::endl;

                if( mpirankRow == 0 ){
                  CopyPattern( HSparseMat, EDMSparseMat );
                  CopyPattern( HSparseMat, FDMSparseMat );
                  c_elsi_collect_pexsi(&fermi_,EDMSparseMat.nzvalLocal.Data(),FDMSparseMat.nzvalLocal.Data());
                  statusOFS << std::endl << "ELSI PEXSI collecte done " << std::endl;
                }
                statusOFS << std::endl << "Time for ELSI PEXSI = " << 
                       timeEnd - timeSta << " [s]" << std::endl << std::endl<<std::flush;

#endif

#ifndef ELSI
                GetTime( timeSta );
                // Old version of PEXSI driver, uses inertia counting + Newton's iteration
                if(1){
                  PPEXSIDFTDriver(
                      pexsiPlan_,
                      pexsiOptions_,
                      numElectronExact,
                      &muPEXSI,
                      &numElectronPEXSI,         
                      &muMinInertia,              
                      &muMaxInertia,             
                      &numTotalInertiaIter,
                      &numTotalPEXSIIter,
                      &info );
                }

                // New version of PEXSI driver, uses inertia count + pole update
                // strategy. No Newton's iteration. But this is not very stable.
//                if(0){
//                  PPEXSIDFTDriver2(
//                      pexsiPlan_,
//                      pexsiOptions_,
//                      numElectronExact,
//                      &muPEXSI,
//                      &numElectronPEXSI,         
//                      &muMinInertia,              
//                      &muMaxInertia,             
//                      &numTotalInertiaIter,
//                      &info );
//                }

                // New version of PEXSI driver, use inertia count + pole update.
                // two method of pole expansion. default is 2
                int method = esdfParam.pexsiMethod;
                int npoint = esdfParam.pexsiNpoint;

//                if(0){
//                  PPEXSIDFTDriver3(
//                      pexsiPlan_,
//                      pexsiOptions_,
//                      numElectronExact,
//                      method,
//                      npoint,
//                      &muPEXSI,
//                      &numElectronPEXSI,         
//                      &pexsiOptions_.muMin0,              
//                      &pexsiOptions_.muMax0,             
//                      &numTotalInertiaIter,
//                      &info );
//                }
 
                GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
                statusOFS << " Time for the main PEXSI Driver is " <<
                  timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

                if( info != 0 ){
                  std::ostringstream msg;
                  msg 
                    << "PEXSI main driver returns info " << info << std::endl;
                  ErrorHandling( msg.str().c_str() );
                }

                // Update the fermi level 
                fermi_ = muPEXSI;
                difNumElectron = std::abs(numElectronPEXSI - numElectronExact);

                // Heuristics for the next step
                //pexsiOptions_.muMin0 = muMinInertia - 5.0 * pexsiOptions_.temperature;
                //pexsiOptions_.muMax0 = muMaxInertia + 5.0 * pexsiOptions_.temperature;

                // Retrieve the PEXSI data

                // FIXME: Hack: in PEXSIDriver3, only DM is available.

                if( mpirankRow == 0 ){
                  Real totalEnergyS;

                  GetTime( timeSta );

                  CopyPattern( HSparseMat, DMSparseMat );
                  CopyPattern( HSparseMat, EDMSparseMat );
                  CopyPattern( HSparseMat, FDMSparseMat );

                  statusOFS << "Before retrieve" << std::endl;
                  PPEXSIRetrieveRealDFTMatrix(
                      pexsiPlan_,
                      DMSparseMat.nzvalLocal.Data(),
                      EDMSparseMat.nzvalLocal.Data(),
                      FDMSparseMat.nzvalLocal.Data(),
                      &totalEnergyH,
                      &totalEnergyS,
                      &totalFreeEnergy,
                      &info );
                  GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
                  statusOFS << " Time for retrieving PEXSI data is " <<
                    timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

                  // FIXME: Hack: there is no free energy really. totalEnergyS is to be added later
                  statusOFS << "NOTE: Free energy = Energy in PPEXSIDFTDriver3!" << std::endl;

                  statusOFS << std::endl
                    << "Results obtained from PEXSI:" << std::endl
                    << "Total energy (H*DM)         = " << totalEnergyH << std::endl
                    << "Total energy (S*EDM)        = " << totalEnergyS << std::endl
                    << "Total free energy           = " << totalFreeEnergy << std::endl 
                    << "InertiaIter                 = " << numTotalInertiaIter << std::endl
//                    << "PEXSIIter                   = " <<  numTotalPEXSIIter << std::endl
                    << "mu                          = " << muPEXSI << std::endl
                    << "numElectron                 = " << numElectronPEXSI << std::endl 
                    << std::endl;

                  if( info != 0 ){
                    std::ostringstream msg;
                    msg 
                      << "PEXSI data retrieval returns info " << info << std::endl;
                    ErrorHandling( msg.str().c_str() );
                  }
                }
#endif
              } // if( mpirank < numProcTotalPEXSI_ )

              // Broadcast the total energy Tr[H*DM] and free energy (which is energy)
              MPI_Bcast( &totalEnergyH, 1, MPI_DOUBLE, 0, domain_.comm );
              MPI_Bcast( &totalFreeEnergy, 1, MPI_DOUBLE, 0, domain_.comm );
              // Broadcast the Fermi level
              MPI_Bcast( &fermi_, 1, MPI_DOUBLE, 0, domain_.comm );
              MPI_Bcast( &difNumElectron, 1, MPI_DOUBLE, 0, domain_.comm );

              if( mpirankRow == 0 )
              {
                GetTime(timeSta);
                // Convert the density matrix from DistSparseMatrix format to the
                // DistElemMat format
                DistSparseMatToDistElemMat3(
                    DMSparseMat,
                    hamDG.NumBasisTotal(),
                    hamDG.HMat().Prtn(),
                    distDMMat_,
                    hamDG.ElemBasisIdx(),
                    hamDG.ElemBasisInvIdx(),
                    domain_.colComm,
                    mpirankSparseVec );


                // Convert the energy density matrix from DistSparseMatrix
                // format to the DistElemMat format

                DistSparseMatToDistElemMat3(
                    EDMSparseMat,
                    hamDG.NumBasisTotal(),
                    hamDG.HMat().Prtn(),
                    distEDMMat_,
                    hamDG.ElemBasisIdx(),
                    hamDG.ElemBasisInvIdx(),
                    domain_.colComm,
                    mpirankSparseVec );

                // Convert the free energy density matrix from DistSparseMatrix
                // format to the DistElemMat format
                DistSparseMatToDistElemMat3(
                    FDMSparseMat,
                    hamDG.NumBasisTotal(),
                    hamDG.HMat().Prtn(),
                    distFDMMat_,
                    hamDG.ElemBasisIdx(),
                    hamDG.ElemBasisInvIdx(),
                    domain_.colComm,
                    mpirankSparseVec );

                GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
                statusOFS << " Time for converting the DistSparseMatrices to DistElemMat " << 
                  "for post-processing is " <<
                  timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
              }

              // Broadcast the distElemMat matrices
              // FIXME this is not a memory efficient implementation
              GetTime(timeSta);
              {
                Int sstrSize;
                std::vector<char> sstr;
                if( mpirankRow == 0 ){
                  std::stringstream distElemMatStream;
                  Int cnt = 0;
                  for( typename std::map<ElemMatKey, DblNumMat >::iterator mi  = distDMMat_.LocalMap().begin();
                      mi != distDMMat_.LocalMap().end(); ++mi ){
                    cnt++;
                  } // for (mi)
                  serialize( cnt, distElemMatStream, NO_MASK );
                  for( typename std::map<ElemMatKey, DblNumMat >::iterator mi  = distDMMat_.LocalMap().begin();
                      mi != distDMMat_.LocalMap().end(); ++mi ){
                    ElemMatKey key = (*mi).first;
                    serialize( key, distElemMatStream, NO_MASK );
                    serialize( distDMMat_.LocalMap()[key], distElemMatStream, NO_MASK );

                    serialize( distEDMMat_.LocalMap()[key], distElemMatStream, NO_MASK ); 
                    serialize( distFDMMat_.LocalMap()[key], distElemMatStream, NO_MASK ); 

                  } // for (mi)
                  sstr.resize( Size( distElemMatStream ) );
                  distElemMatStream.read( &sstr[0], sstr.size() );
                  sstrSize = sstr.size();
                }

                MPI_Bcast( &sstrSize, 1, MPI_INT, 0, domain_.rowComm );
                sstr.resize( sstrSize );
                MPI_Bcast( &sstr[0], sstrSize, MPI_BYTE, 0, domain_.rowComm );

                if( mpirankRow != 0 ){
                  std::stringstream distElemMatStream;
                  distElemMatStream.write( &sstr[0], sstrSize );
                  Int cnt;
                  deserialize( cnt, distElemMatStream, NO_MASK );
                  for( Int i = 0; i < cnt; i++ ){
                    ElemMatKey key;
                    DblNumMat mat;
                    deserialize( key, distElemMatStream, NO_MASK );
                    deserialize( mat, distElemMatStream, NO_MASK );
                    distDMMat_.LocalMap()[key] = mat;

                    deserialize( mat, distElemMatStream, NO_MASK );
                    distEDMMat_.LocalMap()[key] = mat;
                    deserialize( mat, distElemMatStream, NO_MASK );
                    distFDMMat_.LocalMap()[key] = mat;

                  } // for (mi)
                }
              }

              GetTime(timeSta);
              GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
              statusOFS << " Time for broadcasting the density matrix for post-processing is " <<
                timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

              Evdw_ = 0.0;

              // Compute the Harris energy functional.  
              // NOTE: In computing the Harris energy, the density and the
              // potential must be the INPUT density and potential without ANY
              // update.
              GetTime( timeSta );
              CalculateHarrisEnergyDM( totalFreeEnergy, distFDMMat_ );
              GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
              statusOFS << " Time for calculating the Harris energy is " <<
                timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

              // Evaluate the electron density

              for( Int k = 0; k < numElem_[2]; k++ )
                for( Int j = 0; j < numElem_[1]; j++ )
                  for( Int i = 0; i < numElem_[0]; i++ ){
                    Index3 key( i, j, k );
                    if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
                      DblNumVec&  density      = hamDG.Density().LocalMap()[key];
                    } // own this element
                  } // for (i)


              GetTime( timeSta );
              hamDG.CalculateDensityDM2( 
                  hamDG.Density(), hamDG.DensityLGL(), distDMMat_ );
              MPI_Barrier( domain_.comm );
              GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
              statusOFS << " Time for computing density in the global domain is " <<
                timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

              for( Int k = 0; k < numElem_[2]; k++ )
                for( Int j = 0; j < numElem_[1]; j++ )
                  for( Int i = 0; i < numElem_[0]; i++ ){
                    Index3 key( i, j, k );
                    if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
                      DblNumVec&  density      = hamDG.Density().LocalMap()[key];
                    } // own this element
                  } // for (i)


              // Update the output potential, and the KS and second order accurate
              // energy
              GetTime(timeSta);
              {
                // Update the Hartree energy and the exchange correlation energy and
                // potential for computing the KS energy and the second order
                // energy.
                // NOTE Vtot should not be updated until finishing the computation
                // of the energies.

                if( XCType_ == "XC_GGA_XC_PBE" ){
                  GetTime( timeSta );
                  hamDG.CalculateGradDensity(  *distfftPtr_ );
                  GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
                  statusOFS << " Time for calculating gradient of density is " <<
                    timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
                }

                hamDG.CalculateXC( Exc_, hamDG.Epsxc(), hamDG.Vxc(), *distfftPtr_ );

                hamDG.CalculateHartree( hamDG.Vhart(), *distfftPtr_ );

                // Compute the second order accurate energy functional.
                // NOTE: In computing the second order energy, the density and the
                // potential must be the OUTPUT density and potential without ANY
                // MIXING.
                //        CalculateSecondOrderEnergy();

                // Compute the KS energy 
                CalculateKSEnergyDM( totalEnergyH, distEDMMat_, distFDMMat_ );

                // Update the total potential AFTER updating the energy

                // No external potential

                // Compute the new total potential

                hamDG.CalculateVtot( hamDG.Vtot() );

             }
              GetTime(timeEnd);
#if ( _DEBUGlevel_ >= 0 )
              statusOFS << " Time for computing the potential is " <<
                timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

              // Compute the force at every step
              //      if( esdfParam.isCalculateForceEachSCF ){
              //        // Compute force
              //        GetTime( timeSta );
              //        hamDG.CalculateForceDM( *distfftPtr_, distDMMat_ );
              //        GetTime( timeEnd );
              //        statusOFS << "Time for computing the force is " <<
              //          timeEnd - timeSta << " [s]" << std::endl << std::endl;
              //
              //        // Print out the force
              //        // Only master processor output information containing all atoms
              //        if( mpirank == 0 ){
              //          PrintBlock( statusOFS, "Atomic Force" );
              //          {
              //            Point3 forceCM(0.0, 0.0, 0.0);
              //            std::vector<Atom>& atomList = hamDG.AtomList();
              //            Int numAtom = atomList.size();
              //            for( Int a = 0; a < numAtom; a++ ){
              //              Print( statusOFS, "atom", a, "force", atomList[a].force );
              //              forceCM += atomList[a].force;
              //            }
              //            statusOFS << std::endl;
              //            Print( statusOFS, "force for centroid: ", forceCM );
              //            statusOFS << std::endl;
              //          }
              //        }
              //      }

              // TODO Evaluate the a posteriori error estimator

              GetTime( timePEXSIEnd );
#if ( _DEBUGlevel_ >= 0 )
              statusOFS << " Time for PEXSI evaluation is " <<
                timePEXSIEnd - timePEXSISta << " [s]" << std::endl << std::endl;
#endif
            } //if( solutionMethod_ == "pexsi" )


#endif


            // Compute the error of the mixing variable

            GetTime(timeSta);
            {
              Real normMixDifLocal = 0.0, normMixOldLocal = 0.0;
              Real normMixDif, normMixOld;
              for( Int k = 0; k < numElem_[2]; k++ )
                for( Int j = 0; j < numElem_[1]; j++ )
                  for( Int i = 0; i < numElem_[0]; i++ ){
                    Index3 key( i, j, k );
                    if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
                      if( mixVariable_ == "density" ){
                        DblNumVec& oldVec = mixInnerSave_.LocalMap()[key];
                        DblNumVec& newVec = hamDG.Density().LocalMap()[key];

                        for( Int p = 0; p < oldVec.m(); p++ ){
                          normMixDifLocal += pow( oldVec(p) - newVec(p), 2.0 );
                          normMixOldLocal += pow( oldVec(p), 2.0 );
                        }
                      }
                      else if( mixVariable_ == "potential" ){
                        DblNumVec& oldVec = mixInnerSave_.LocalMap()[key];
                        DblNumVec& newVec = hamDG.Vtot().LocalMap()[key];

                        for( Int p = 0; p < oldVec.m(); p++ ){
                          normMixDifLocal += pow( oldVec(p) - newVec(p), 2.0 );
                          normMixOldLocal += pow( oldVec(p), 2.0 );
                        }
                      }
                    } // own this element
                  } // for (i)




              mpi::Allreduce( &normMixDifLocal, &normMixDif, 1, MPI_SUM, 
                  domain_.colComm );
              mpi::Allreduce( &normMixOldLocal, &normMixOld, 1, MPI_SUM,
                  domain_.colComm );

              normMixDif = std::sqrt( normMixDif );
              normMixOld = std::sqrt( normMixOld );

              scfInnerNorm_    = normMixDif / normMixOld;
#if ( _DEBUGlevel_ >= 1 )
              Print(statusOFS, "norm(MixDif)          = ", normMixDif );
              Print(statusOFS, "norm(MixOld)          = ", normMixOld );
              Print(statusOFS, "norm(out-in)/norm(in) = ", scfInnerNorm_ );
#endif
            }

            if( scfInnerNorm_ < scfInnerTolerance_ ){
              /* converged */
              Print( statusOFS, "Inner SCF is converged!\n" );
              isInnerSCFConverged = true;
            }

            MPI_Barrier( domain_.colComm );
            GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
            statusOFS << " Time for computing the SCF residual is " <<
              timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

            // Mixing for the inner SCF iteration.
            GetTime( timeSta );

            // The number of iterations used for Anderson mixing
            Int numAndersonIter;

            if( scfInnerMaxIter_ == 1 ){
              // Maximum inner iteration = 1 means there is no distinction of
              // inner/outer SCF.  Anderson mixing uses the global history
              numAndersonIter = scfTotalInnerIter_;
            }
            else{
              // If more than one inner iterations is used, then Anderson only
              // uses local history.  For explanation see 
              //
              // Note 04/11/2013:  
              // "Problem of Anderson mixing in inner/outer SCF loop"
              numAndersonIter = innerIter;
            }

            if( mixVariable_ == "density" ){
              if( mixType_ == "anderson" ||
                  mixType_ == "kerker+anderson"    ){
                AndersonMix(
                    numAndersonIter, 
                    mixStepLength_,
                    mixType_,
                    hamDG.Density(),
                    mixInnerSave_,
                    hamDG.Density(),
                    dfInnerMat_,
                    dvInnerMat_);
              } else{
                ErrorHandling("Invalid mixing type.");
              }
            }
            else if( mixVariable_ == "potential" ){
              if( mixType_ == "anderson" ||
                  mixType_ == "kerker+anderson"    ){
                AndersonMix(
                    numAndersonIter, 
                    mixStepLength_,
                    mixType_,
                    hamDG.Vtot(),
                    mixInnerSave_,
                    hamDG.Vtot(),
                    dfInnerMat_,
                    dvInnerMat_);
              } else{
                ErrorHandling("Invalid mixing type.");
              }
            }



            MPI_Barrier( domain_.comm );
            GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
            statusOFS << " Time for mixing is " <<
              timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

            // Post processing for the density mixing. Make sure that the
            // density is positive, and compute the potential again. 
            // This is only used for density mixing.
            if( mixVariable_ == "density" )
            {
              Real sumRhoLocal = 0.0;
              Real sumRho;
              for( Int k = 0; k < numElem_[2]; k++ )
                for( Int j = 0; j < numElem_[1]; j++ )
                  for( Int i = 0; i < numElem_[0]; i++ ){
                    Index3 key( i, j, k );
                    if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
                      DblNumVec&  density      = hamDG.Density().LocalMap()[key];

                      for (Int p=0; p < density.Size(); p++) {
                        density(p) = std::max( density(p), 0.0 );
                        sumRhoLocal += density(p);
                      }
                    } // own this element
                  } // for (i)
              mpi::Allreduce( &sumRhoLocal, &sumRho, 1, MPI_SUM, domain_.colComm );
              sumRho *= domain_.Volume() / domain_.NumGridTotalFine();

              Real rhofac = hamDG.NumSpin() * hamDG.NumOccupiedState() / sumRho;

#if ( _DEBUGlevel_ >= 1 )
              statusOFS << std::endl;
              Print( statusOFS, "Sum Rho after mixing (raw data) = ", sumRho );
              statusOFS << std::endl;
#endif


              // Normalize the electron density in the global domain
/*              for( Int k = 0; k < numElem_[2]; k++ )
                for( Int j = 0; j < numElem_[1]; j++ )
                  for( Int i = 0; i < numElem_[0]; i++ ){
                    Index3 key( i, j, k );
                    if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
                      DblNumVec& localRho = hamDG.Density().LocalMap()[key];
                      blas::Scal( localRho.Size(), rhofac, localRho.Data(), 1 );
                    } // own this element
                  } // for (i)
*/

              // Update the potential after mixing for the next iteration.  
              // This is only used for potential mixing

              // Compute the exchange-correlation potential and energy from the
              // new density

              if( XCType_ == "XC_GGA_XC_PBE" ){
                GetTime( timeSta );
                hamDG.CalculateGradDensity(  *distfftPtr_ );
                GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
                statusOFS << " Time for calculating gradient of density is " <<
                  timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
              }

              hamDG.CalculateXC( Exc_, hamDG.Epsxc(), hamDG.Vxc(), *distfftPtr_ );

              hamDG.CalculateHartree( hamDG.Vhart(), *distfftPtr_ );

              // No external potential

              // Compute the new total potential

              hamDG.CalculateVtot( hamDG.Vtot() );
            }
            // check check 
#ifdef _USE_PEXSI_
            if( solutionMethod_ == "pexsi" )
            {
            Real deltaVmin = 0.0;
            Real deltaVmax = 0.0;

            for( Int k=0; k< numElem_[2]; k++ )
              for( Int j=0; j< numElem_[1]; j++ )
                for( Int i=0; i< numElem_[0]; i++ ) {
                  Index3 key = Index3(i,j,k);
                  if( distEigSolPtr_->Prtn().Owner(key) == (mpirank / dmRow_) ){
                    DblNumVec vtotCur;
                    vtotCur = hamDG.Vtot().LocalMap()[key];
                    DblNumVec& oldVtot = VtotHist.LocalMap()[key];
                    blas::Axpy( vtotCur.m(), -1.0, oldVtot.Data(),
                                    1, vtotCur.Data(), 1);
                    deltaVmin = std::min( deltaVmin, findMin(vtotCur) );
                    deltaVmax = std::max( deltaVmax, findMax(vtotCur) );
                  }
                }

              {
                int color = mpirank % dmRow_;
                MPI_Comm elemComm;
                std::vector<Real> vlist(mpisize/dmRow_);
  
                MPI_Comm_split( domain_.comm, color, mpirank, &elemComm );
                MPI_Allgather( &deltaVmin, 1, MPI_DOUBLE, &vlist[0], 1, MPI_DOUBLE, elemComm);
                deltaVmin = 0.0;
                for(int i =0; i < mpisize/dmRow_; i++)
                  if(deltaVmin > vlist[i])
                     deltaVmin = vlist[i];

                MPI_Allgather( &deltaVmax, 1, MPI_DOUBLE, &vlist[0], 1, MPI_DOUBLE, elemComm);
                deltaVmax = 0.0;
                for(int i =0; i < mpisize/dmRow_; i++)
                  if(deltaVmax < vlist[i])
                     deltaVmax = vlist[i];
 
                pexsiOptions_.muMin0 += deltaVmin;
                pexsiOptions_.muMax0 += deltaVmax;
                MPI_Comm_free( &elemComm);
              }
            }
#endif 
            // Print out the state variables of the current iteration

            // Only master processor output information containing all atoms
            if( mpirank == 0 ){
              PrintState( );
            }

            GetTime( timeIterEnd );

            statusOFS << " Time for this inner SCF iteration = " << timeIterEnd - timeIterStart
              << " [s]" << std::endl << std::endl;
          } // for (innerIter)


          return ;
        }         // -----  end of method SCFDG::InnerIterate  ----- 

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

#if ( _DEBUGlevel_ >= 1 )
          statusOFS << "neighborIdx = " << neighborIdx << std::endl;
#endif

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

#if ( _DEBUGlevel_ >= 1 )
                    statusOFS << "keyExtElem         = " << key << std::endl;
                    statusOFS << "numGridExtElemFine = " << numGridExtElem << std::endl;
                    statusOFS << "numGridElemFine    = " << numGridElem << std::endl;
                    statusOFS << "keyElem            = " << keyElem << ", shiftIdx = " << shiftIdx << std::endl;
#endif

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

#if ( _DEBUGlevel_ >= 0 ) 
                    Print( statusOFS, "vtotMax  = ", vtotMax );
                    Print( statusOFS, "vtotAvg  = ", vtotAvg );
                    Print( statusOFS, "vtotMin  = ", vtotMin );
#endif

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
          // FIXME Magic number here
          Real tol = 1e-10; 
          Int maxiter = 100;  

          if (SmearingScheme_ == "FD")
          {

            Real lb, ub, flb, fub, occsum;
            Int ilb, iub, iter;

            Int npsi       = hamDGPtr_->NumStateTotal();
            Int nOccStates = hamDGPtr_->NumOccupiedState();

            if( eigVal.m() != npsi ){
              std::ostringstream msg;
              msg 
                << "The number of eigenstates do not match."  << std::endl
                << "eigVal         ~ " << eigVal.m() << std::endl
                << "numStateTotal  ~ " << npsi << std::endl;
              ErrorHandling( msg.str().c_str() );
            }


            if( occupationRate.m() != npsi ) occupationRate.Resize( npsi );

            if( npsi > nOccStates )  {
              /* use bisection to find efermi such that 
               * sum_i fermidirac(ev(i)) = nocc
               */
              ilb = nOccStates-1;
              iub = nOccStates+1;

              lb = eigVal(ilb-1);
              ub = eigVal(iub-1);

              /* Calculate Fermi-Dirac function and make sure that
               * flb < nocc and fub > nocc
               */

              flb = 0.0;
              fub = 0.0;
              for(Int j = 0; j < npsi; j++) {
                flb += 1.0 / (1.0 + exp(Tbeta_*(eigVal(j)-lb)));
                fub += 1.0 / (1.0 + exp(Tbeta_*(eigVal(j)-ub))); 
              }

              while( (nOccStates-flb)*(fub-nOccStates) < 0 ) {
                if( flb > nOccStates ) {
                  if(ilb > 0){
                    ilb--;
                    lb = eigVal(ilb-1);
                    flb = 0.0;
                    for(Int j = 0; j < npsi; j++) flb += 1.0 / (1.0 + exp(Tbeta_*(eigVal(j)-lb)));
                  }
                  else {
                    ErrorHandling( "Cannot find a lower bound for efermi" );
                  }
                }

                if( fub < nOccStates ) {
                  if( iub < npsi ) {
                    iub++;
                    ub = eigVal(iub-1);
                    fub = 0.0;
                    for(Int j = 0; j < npsi; j++) fub += 1.0 / (1.0 + exp(Tbeta_*(eigVal(j)-ub)));
                  }
                  else {
                    ErrorHandling( "Cannot find a lower bound for efermi, try to increase the number of wavefunctions" );
                  }
                }
              }  /* end while */

              fermi_ = (lb+ub)*0.5;
              occsum = 0.0;
              for(Int j = 0; j < npsi; j++) {
                occupationRate(j) = 1.0 / (1.0 + exp(Tbeta_*(eigVal(j) - fermi_)));
                occsum += occupationRate(j);
              }

              /* Start bisection iteration */
              iter = 1;
              while( (fabs(occsum - nOccStates) > tol) && (iter < maxiter) ) {
                if( occsum < nOccStates ) {lb = fermi_;}
                else {ub = fermi_;}

                fermi_ = (lb+ub)*0.5;
                occsum = 0.0;
                for(Int j = 0; j < npsi; j++) {
                  occupationRate(j) = 1.0 / (1.0 + exp(Tbeta_*(eigVal(j) - fermi_)));
                  occsum += occupationRate(j);
                }
                iter++;
              }
            }
            else {
              if (npsi == nOccStates ) {
                for(Int j = 0; j < npsi; j++) 
                  occupationRate(j) = 1.0;
                fermi_ = eigVal(npsi-1);
              }
              else {
                ErrorHandling( "The number of eigenvalues in ev should be larger than nocc" );
              }
            }
          } // end of if (SmearingScheme_ == "FD")        
          else
          {
            // MP and GB type smearing

            Int npsi       = hamDGPtr_->NumStateTotal();
            Int nOccStates = hamDGPtr_->NumOccupiedState();

            if( eigVal.m() != npsi ){
              std::ostringstream msg;
              msg 
                << "The number of eigenstates do not match."  << std::endl
                << "eigVal         ~ " << eigVal.m() << std::endl
                << "numStateTotal  ~ " << npsi << std::endl;
              ErrorHandling( msg.str().c_str() );
            }


            if( occupationRate.m() != npsi ) 
              occupationRate.Resize( npsi );

            Real lb, ub, flb, fub, fx;
            Int  iter;


            if( npsi > nOccStates )  
            { 
              // Set up the bounds
              lb = eigVal(0);
              ub = eigVal(npsi - 1);

              // Set up the function bounds
              flb = mp_occupations_residual(eigVal, lb, nOccStates);
              fub = mp_occupations_residual(eigVal, ub, nOccStates);

              if(flb * fub > 0.0)
                ErrorHandling( "Bisection method for finding Fermi level cannot proceed !!" );

              fermi_ = (lb + ub) * 0.5;

              /* Start bisection iteration */
              iter = 1;
              fx = mp_occupations_residual(eigVal, fermi_, nOccStates);


              // Iterate using the bisection method
              while( (fabs(fx) > tol) && (iter < maxiter) ) 
              {
                flb = mp_occupations_residual(eigVal, lb, nOccStates);
                fub = mp_occupations_residual(eigVal, ub, nOccStates);

                if( (flb * fx) < 0.0 )
                  ub = fermi_;
                else
                  lb = fermi_;

                fermi_ = (lb + ub) * 0.5;
                fx = mp_occupations_residual(eigVal, fermi_, nOccStates);

                iter++;
              }

              if(iter >= maxiter)
                ErrorHandling( "Bisection method for finding Fermi level does not appear to converge !!" );
              else
              {
                // Bisection method seems to have converged
                // Fill up the occupations
                populate_mp_occupations(eigVal, occupationRate, fermi_);

              }        
            } // end of if(npsi > nOccStates)
            else 
            {
              if (npsi == nOccStates ) 
              {
                for(Int j = 0; j < npsi; j++) 
                  occupationRate(j) = 1.0;

                fermi_ = eigVal(npsi-1);
              }
              else 
              {
                // npsi < nOccStates
                ErrorHandling( "The number of top eigenvalues should be larger than number of occupied states !! " );
              }
            } // end of if(npsi > nOccStates) ... else

          } // end of if(SmearingScheme_ == "FD") ... else

          return ;
        }         // -----  end of method SCFDG::CalculateOccupationRate  ----- 


#ifdef GPU
      void
        SCFDG::InterpPeriodicUniformToLGL    (
            const Index3& numUniformGrid,
            const Index3& numLGLGrid,
            cuDblNumMat&  cu_psiUniform,
            cuDblNumMat&  cu_psiLGL,
            cuDblNumVec& cu_tmp1, cuDblNumVec& cu_tmp2 )
        {
//          Real time1, time2;
          
          Index3 Ns1 = numUniformGrid;
          Index3 Ns2 = numLGLGrid;
          Real one = 1.0;
          Real zero = 0.0;
          DblNumVec  tmp1( Ns2[0] * Ns1[1] * Ns1[2] );
          DblNumVec  tmp1_trans( Ns2[0] * Ns1[1] * Ns1[2] );
          DblNumVec  tmp2( Ns2[0] * Ns2[1] * Ns1[2] );
          SetValue( tmp1, 0.0 );
          SetValue( tmp2, 0.0 );

          {
            Int m = Ns2[0], n = Ns1[1] * Ns1[2], k = Ns1[0];
            cublas::Gemm( HIPBLAS_OP_N, HIPBLAS_OP_N, m, n, k, &one, cu_PeriodicUniformToLGLMat_x.Data(), m, cu_psiUniform.Data(), k, &zero, cu_tmp1.Data(), m );
          }
  //        cuda_sync();

          {
          //-----add by lijl 20220708
          // xmqin 20220714
          #if 1
            cuDblNumVec cu_tmp1_trans( Ns2[0] * Ns1[1] * Ns1[2] );
            cuDblNumVec cu_tmp2_trans( Ns2[0] * Ns2[1] * Ns1[2] );
            transpose3D_revised_xyz2yxz_device( cu_tmp1_trans.Data(), cu_tmp1.Data(), Ns2[0], Ns1[1], Ns1[2]);
            Int m = Ns2[1], n = Ns1[2]*Ns2[0], k = Ns1[1];
            cublas::Gemm( HIPBLAS_OP_N, HIPBLAS_OP_N, m, n, k, &one, cu_PeriodicUniformToLGLMat_y.Data(), m, cu_tmp1_trans.Data(), k, &zero, cu_tmp2_trans.Data(), m);
            transpose3D_revised_xyz2yxz_device(cu_tmp2.Data(), cu_tmp2_trans.Data(), Ns2[1], Ns2[0], Ns1[2]);
          #else
            Int   m = Ns2[1], n = Ns1[1];
            Int   ptrShift1, ptrShift2;
            Int   inc = Ns2[0];
           cu_tmp1.CopyTo(tmp1);
 //           cuda_sync();
            for( Int k = 0; k < Ns1[2]; k++ ){
              for( Int i = 0; i < Ns2[0]; i++ ){
                ptrShift1 = i + k * Ns2[0] * Ns1[1];
                ptrShift2 = i + k * Ns2[0] * Ns2[1];
             #if 0
                cublas::Gemv( HIPBLAS_OP_N, m, n, &one,  cu_PeriodicUniformToLGLMat_y.Data(), m,
                      cu_tmp1.Data() + ptrShift1, inc, &zero, cu_tmp2.Data() + ptrShift2, inc );
             #else
                 blas::Gemv( 'N', m, n, 1.0,
                    PeriodicUniformToLGLMat_[1].Data(), m,
                    tmp1.Data() + ptrShift1, inc, 0.0,
                    tmp2.Data() + ptrShift2, inc );        
             #endif
              } // for (i)
            } // for (k)
           cu_tmp2.CopyFrom(tmp2);
//           cuda_sync();
          #endif
          }

          {
            Int m = Ns2[0] * Ns2[1], n = Ns2[2], k = Ns1[2];
            cublas::Gemm( HIPBLAS_OP_N, HIPBLAS_OP_T, m, n, k, &one,
                cu_tmp2.Data(), m,
                cu_PeriodicUniformToLGLMat_z.Data(), n, &zero, cu_psiLGL.Data(), m );
          }

        return ;
        }           
#else
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

#endif
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
          Ecor_   = (Exc_ - EVxc_) - Ehart_ - Eself_;
          if( esdfParam.isUseVLocal == true ){
            Ecor_ += hamDG.EIonSR();
          }



          // Total energy
          Etot_ = Ekin_ + Ecor_;

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

              //           for(Int l=0; l< SCFDG_comp_subspace_top_eigvals_.m(); l++) 
              //             {
              //               Real eig = SCFDG_comp_subspace_top_eigvals_(l);
              //               if( eig - fermi >= 0){
              //                 Efree_ += -numSpin /Tbeta*log(1.0+exp(-Tbeta*(eig - fermi))); 
              //               }
              //               else
              //                 {
              //                   Efree_ += numSpin * (eig - fermi) - numSpin / Tbeta*log(1.0+exp(Tbeta*(eig-fermi)));
              //                 } 
              //             }
              //           
              //           Int howmany_to_calc = (hamDG.NumOccupiedState() + SCFDG_comp_subspace_N_solve_) - hamDG.NumStateTotal();
              //           Efree_ += Ecor_ + fermi * (howmany_to_calc) * numSpin;


            }
            else
            {  
              for(Int l=0; l< eigVal.m(); l++) {
                Real eig = eigVal(l);
                if( eig - fermi >= 0){
                  Efree_ += -numSpin /Tbeta*log(1.0+exp(-Tbeta*(eig - fermi))); 
                }
                else{
                  Efree_ += numSpin * (eig - fermi) - numSpin / Tbeta*log(1.0+exp(Tbeta*(eig-fermi)));
                } 
              }

              Efree_ += Ecor_ + fermi * hamDG.NumOccupiedState() * numSpin; 
            }


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
          Ecor_   = (Exc_ - EVxc_) - Ehart_ - Eself_;
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

          // NOTE: To avoid confusion, all energies in this routine are
          // temporary variables other than EfreeHarris_.
          //
          // The related energies will be computed again in the routine
          //
          // CalculateKSEnergy()

          Real Ekin, Eself, Ehart, EVxc, Exc, Ecor;

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
              if(SmearingScheme_ == "FD")
              {
                for(Int l=0; l< eigVal.m(); l++) {
                  Real eig = eigVal(l);
                  if( eig - fermi >= 0){
                    EfreeHarris_ += -numSpin /Tbeta*log(1.0+exp(-Tbeta*(eig - fermi))); 
                  }
                  else{
                    EfreeHarris_ += numSpin * (eig - fermi) - numSpin / Tbeta*log(1.0+exp(Tbeta*(eig-fermi)));
                  }
                }
                EfreeHarris_ += Ecor + fermi * hamDG.NumOccupiedState() * numSpin; 
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

                EfreeHarris_ = Ekin + Ecor + (numSpin * Tsigma_) * occup_energy_part;

              }

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

          Real Ehelm, Eself, Ehart, EVxc, Exc, Ecor;

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


          // The Helmholtz part of the free energy
          //   Ehelm = -2/beta Tr[log(1+exp(mu-H))] + mu*N_e
          // FIXME Put the above documentation to the proper place like the hpp
          // file
          if( 1 ) {
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

            mpi::Allreduce( &EhelmLocal, &Ehelm, 
                1, MPI_SUM, domain_.colComm );

            // Add the mu*N term
            Ehelm += fermi_ * hamDG.NumOccupiedState() * numSpin;

          }


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


          Real Ekin, Eself, Ehart, EVtot, Exc, Ecor;

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

              }

            }  // end of full spectrum available calculation


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

            // Calculate the number of atom types.
            //    Real numAtomType = 0;   
            //    for(Int a=0; a< atomList.size() ; a++) {
            //      Int type1 = atomList[a].type;
            //      Int a1 = 0;
            //      Int a2 = 0;
            //      for(Int b=0; b<a ; b++) {
            //        a1 = a1 + 1;
            //        Int type2 = atomList[b].type;
            //        if ( type1 != type2 ) {
            //          a2 = a2 + 1;
            //        }
            //      }
            //
            //      if ( a1 == a2 ) {
            //        numAtomType = numAtomType + 1;
            //      }
            //
            //    }
            //

            //    IntNumVec  atomType ( numAtomType );
            //    SetValue( atomType, 0 );

            //    Real numAtomType1 = 0;
            //    atomType(0) = atomList[0].type;


            //    for(Int a=0; a< atomList.size() ; a++) {
            //      Int type1 = atomList[a].type;
            //      Int a1 = 0;
            //      Int a2 = 0;
            //      for(Int b=0; b<a ; b++) {
            //        a1 = a1 + 1;
            //        Int type2 = atomList[b].type;
            //        if ( type1 != type2 ) {
            //          a2 = a2 + 1;
            //        }
            //      }
            //      if ( a1 == a2 ) {
            //        numAtomType1 = numAtomType1 + 1;
            //        atomType(numAtomType1-1) = atomList[a].type;
            //      }
            //    }


            //    DblNumMat  vdw_c6 ( numAtomType, numAtomType );
            //    DblNumMat  vdw_r0 ( numAtomType, numAtomType );
            //    SetValue( vdw_c6, 0.0 );
            //    SetValue( vdw_r0, 0.0 );
            //
            //    for(Int i=0; i< numAtomType; i++) {
            //      for(Int j=0; j< numAtomType; j++) {
            //        vdw_c6(i,j)=std::sqrt(vdw_c6_dftd2[atomType(i)-1]*vdw_c6_dftd2[atomType(j)-1]);
            //        //vdw_r0(i,j)=(vdw_r0_dftd2(atomType(i))+vdw_r0_dftd2(atomType(j)))/Bohr_Ang;
            //        vdw_r0(i,j)=(vdw_r0_dftd2[atomType(i)-1]+vdw_r0_dftd2[atomType(j)-1]);
            //      }
            //    }

            //    statusOFS << "vdw_c6 = " << vdw_c6 << std::endl;
            //    statusOFS << "vdw_r0 = " << vdw_r0 << std::endl;


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
                        //Real c6 = std::sqrt( vdw_c6_dftd2[iType-1] * vdw_c6_dftd2[jType-1] );
                        //Real r0 = vdw_r0_dftd2[iType-1] + vdw_r0_dftd2[jType-1];

                        Real ex = exp( -vdw_d * ( rr / r0 - 1 ));
                        Real fr = 1.0 / ( 1.0 + ex );
                        Real c6r6 = c6 / pow(rr, 6.0);

                        // Contribution to energy
                        Evdw_ = Evdw_ - sfact * fr * c6r6;

                        // Contribution to force
                        if( i != j ) {

                          Real gr = ( vdw_d / r0 ) * ( fr * fr ) * ex;
                          Real grad = sfact * ( gr - 6.0 * fr / rr ) * c6r6 / rr; 

                          //Real fx = grad * rx * dm.length[0];
                          //Real fy = grad * ry * dm.length[1];
                          //Real fz = grad * rz * dm.length[2];
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


            //#endif 

          } // If DFT-D2


          VDWEnergy = Evdw_;
          VDWForce = forceVdw_;



          return ;
        }         // -----  end of method SCFDG::CalculateVDW  ----- 



      void
        SCFDG::AndersonMix    ( 
            Int             iter, 
            Real            mixStepLength,
            std::string     mixType,
            DistDblNumVec&  distvMix,
            DistDblNumVec&  distvOld,
            DistDblNumVec&  distvNew,
            DistDblNumMat&  dfMat,
            DistDblNumMat&  dvMat )
        {
          Int mpirank, mpisize;
          MPI_Comm_rank( domain_.comm, &mpirank );
          MPI_Comm_size( domain_.comm, &mpisize );

          Int mpirankRow;  MPI_Comm_rank(domain_.rowComm, &mpirankRow);
          Int mpisizeRow;  MPI_Comm_size(domain_.rowComm, &mpisizeRow);
          Int mpirankCol;  MPI_Comm_rank(domain_.colComm, &mpirankCol);
          Int mpisizeCol;  MPI_Comm_size(domain_.colComm, &mpisizeCol);

          distvMix.SetComm(domain_.colComm);
          distvOld.SetComm(domain_.colComm);
          distvNew.SetComm(domain_.colComm);
          dfMat.SetComm(domain_.colComm);
          dvMat.SetComm(domain_.colComm);

          // Residual 
          DistDblNumVec distRes;
          // Optimal input potential in Anderon mixing.
          DistDblNumVec distvOpt; 
          // Optimal residual in Anderson mixing
          DistDblNumVec distResOpt; 
          // Preconditioned optimal residual in Anderson mixing
          DistDblNumVec distPrecResOpt;

          distRes.SetComm(domain_.colComm);
          distvOpt.SetComm(domain_.colComm);
          distResOpt.SetComm(domain_.colComm);
          distPrecResOpt.SetComm(domain_.colComm);

          // *********************************************************************
          // Initialize
          // *********************************************************************
          Int ntot  = hamDGPtr_->NumUniformGridElemFine().prod();

          // Number of iterations used, iter should start from 1
          Int iterused = std::min( iter-1, mixMaxDim_ ); 
          // The current position of dfMat, dvMat
          Int ipos = iter - 1 - ((iter-2)/ mixMaxDim_ ) * mixMaxDim_;
          // The next position of dfMat, dvMat
          Int inext = iter - ((iter-1)/ mixMaxDim_) * mixMaxDim_;

          distRes.Prtn()          = elemPrtn_;
          distvOpt.Prtn()         = elemPrtn_;
          distResOpt.Prtn()       = elemPrtn_;
          distPrecResOpt.Prtn()   = elemPrtn_;



          for( Int k = 0; k < numElem_[2]; k++ )
            for( Int j = 0; j < numElem_[1]; j++ )
              for( Int i = 0; i < numElem_[0]; i++ ){
                Index3 key( i, j, k );
                if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
                  DblNumVec  emptyVec( ntot );
                  SetValue( emptyVec, 0.0 );
                  distRes.LocalMap()[key]        = emptyVec;
                  distvOpt.LocalMap()[key]       = emptyVec;
                  distResOpt.LocalMap()[key]     = emptyVec;
                  distPrecResOpt.LocalMap()[key] = emptyVec;
                } // if ( own this element )
              } // for (i)



          // *********************************************************************
          // Anderson mixing
          // *********************************************************************

          for( Int k = 0; k < numElem_[2]; k++ )
            for( Int j = 0; j < numElem_[1]; j++ )
              for( Int i = 0; i < numElem_[0]; i++ ){
                Index3 key( i, j, k );
                if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
                  // res(:) = vOld(:) - vNew(:) is the residual
                  distRes.LocalMap()[key] = distvOld.LocalMap()[key];
                  blas::Axpy( ntot, -1.0, distvNew.LocalMap()[key].Data(), 1, 
                      distRes.LocalMap()[key].Data(), 1 );

                  distvOpt.LocalMap()[key]   = distvOld.LocalMap()[key];
                  distResOpt.LocalMap()[key] = distRes.LocalMap()[key];


                  // dfMat(:, ipos-1) = res(:) - dfMat(:, ipos-1);
                  // dvMat(:, ipos-1) = vOld(:) - dvMat(:, ipos-1);
                  if( iter > 1 ){
                    blas::Scal( ntot, -1.0, dfMat.LocalMap()[key].VecData(ipos-1), 1 );
                    blas::Axpy( ntot, 1.0,  distRes.LocalMap()[key].Data(), 1, 
                        dfMat.LocalMap()[key].VecData(ipos-1), 1 );
                    blas::Scal( ntot, -1.0, dvMat.LocalMap()[key].VecData(ipos-1), 1 );
                    blas::Axpy( ntot, 1.0,  distvOld.LocalMap()[key].Data(),  1, 
                        dvMat.LocalMap()[key].VecData(ipos-1), 1 );
                  }
                } // own this element
              } // for (i)



          // For iter == 1, Anderson mixing is the same as simple mixing. 
          if( iter > 1){

            Int nrow = iterused;

            // Normal matrix FTF = F^T * F
            DblNumMat FTFLocal( nrow, nrow ), FTF( nrow, nrow );
            SetValue( FTFLocal, 0.0 );
            SetValue( FTF, 0.0 );

            // Right hand side FTv = F^T * vout
            DblNumVec FTvLocal( nrow ), FTv( nrow );
            SetValue( FTvLocal, 0.0 );
            SetValue( FTv, 0.0 );

            // Local construction of FTF and FTv
            for( Int k = 0; k < numElem_[2]; k++ )
              for( Int j = 0; j < numElem_[1]; j++ )
                for( Int i = 0; i < numElem_[0]; i++ ){
                  Index3 key( i, j, k );
                  if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
                    DblNumMat& df     = dfMat.LocalMap()[key];
                    DblNumVec& res    = distRes.LocalMap()[key];
                    for( Int q = 0; q < nrow; q++ ){
                      FTvLocal(q) += blas::Dot( ntot, df.VecData(q), 1,
                          res.Data(), 1 );

                      for( Int p = q; p < nrow; p++ ){
                        FTFLocal(p, q) += blas::Dot( ntot, df.VecData(p), 1, 
                            df.VecData(q), 1 );
                        if( p > q )
                          FTFLocal(q,p) = FTFLocal(p,q);
                      } // for (p)
                    } // for (q)

                  } // own this element
                } // for (i)

            // Reduce the data
            mpi::Allreduce( FTFLocal.Data(), FTF.Data(), nrow * nrow, 
                MPI_SUM, domain_.colComm );
            mpi::Allreduce( FTvLocal.Data(), FTv.Data(), nrow, 
                MPI_SUM, domain_.colComm );

            // All processors solve the least square problem

            // FIXME Magic number for pseudo-inverse
            Real rcond = 1e-12;
            Int rank;

            DblNumVec  S( nrow );

            // FTv = pinv( FTF ) * res
            lapack::SVDLeastSquare( nrow, nrow, 1, 
                FTF.Data(), nrow, FTv.Data(), nrow,
                S.Data(), rcond, &rank );

            statusOFS << "Rank of dfmat = " << rank <<
              ", rcond = " << rcond << std::endl;

            // Update vOpt, resOpt. 
            // FTv = Y^{\dagger} r as in the usual notation.
            // 
            for( Int k = 0; k < numElem_[2]; k++ )
              for( Int j = 0; j < numElem_[1]; j++ )
                for( Int i = 0; i < numElem_[0]; i++ ){
                  Index3 key( i, j, k );
                  if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
                    // vOpt   -= dv * FTv
                    blas::Gemv('N', ntot, nrow, -1.0, dvMat.LocalMap()[key].Data(),
                        ntot, FTv.Data(), 1, 1.0, 
                        distvOpt.LocalMap()[key].Data(), 1 );

                    // resOpt -= df * FTv
                    blas::Gemv('N', ntot, nrow, -1.0, dfMat.LocalMap()[key].Data(),
                        ntot, FTv.Data(), 1, 1.0, 
                        distResOpt.LocalMap()[key].Data(), 1 );
                  } // own this element
                } // for (i)
          } // (iter > 1)




          if( mixType == "kerker+anderson" ){
            KerkerPrecond( distPrecResOpt, distResOpt );
          }
          else if( mixType == "anderson" ){
            for( Int k = 0; k < numElem_[2]; k++ )
              for( Int j = 0; j < numElem_[1]; j++ )
                for( Int i = 0; i < numElem_[0]; i++ ){
                  Index3 key( i, j, k );
                  if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
                    distPrecResOpt.LocalMap()[key] = 
                      distResOpt.LocalMap()[key];
                  } // own this element
                } // for (i)
          }
          else{
            ErrorHandling("Invalid mixing type.");
          }




          // Update dfMat, dvMat, vMix 
          for( Int k = 0; k < numElem_[2]; k++ )
            for( Int j = 0; j < numElem_[1]; j++ )
              for( Int i = 0; i < numElem_[0]; i++ ){
                Index3 key( i, j, k );
                if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
                  // dfMat(:, inext-1) = res(:)
                  // dvMat(:, inext-1) = vOld(:)
                  blas::Copy( ntot, distRes.LocalMap()[key].Data(), 1, 
                      dfMat.LocalMap()[key].VecData(inext-1), 1 );
                  blas::Copy( ntot, distvOld.LocalMap()[key].Data(),  1, 
                      dvMat.LocalMap()[key].VecData(inext-1), 1 );

                  // vMix(:) = vOpt(:) - mixStepLength * precRes(:)
                  distvMix.LocalMap()[key] = distvOpt.LocalMap()[key];
                  blas::Axpy( ntot, -mixStepLength, 
                      distPrecResOpt.LocalMap()[key].Data(), 1, 
                      distvMix.LocalMap()[key].Data(), 1 );
                } // own this element
              } // for (i)




          return ;
        }         // -----  end of method SCFDG::AndersonMix  ----- 

      void
        SCFDG::KerkerPrecond ( 
            DistDblNumVec&  distPrecResidual,
            const DistDblNumVec&  distResidual )
        {
          Int mpirank, mpisize;
          MPI_Comm_rank( domain_.comm, &mpirank );
          MPI_Comm_size( domain_.comm, &mpisize );

          Int mpirankRow;  MPI_Comm_rank(domain_.rowComm, &mpirankRow);
          Int mpisizeRow;  MPI_Comm_size(domain_.rowComm, &mpisizeRow);
          Int mpirankCol;  MPI_Comm_rank(domain_.colComm, &mpirankCol);
          Int mpisizeCol;  MPI_Comm_size(domain_.colComm, &mpisizeCol);

          DistFourier& fft = *distfftPtr_;
          //DistFourier.SetComm(domain_.colComm);

          Int ntot      = fft.numGridTotal;
          Int ntotLocal = fft.numGridLocal;

          Index3 numUniformGridElem = hamDGPtr_->NumUniformGridElem();

          // Convert distResidual to tempVecLocal in distributed row vector format
          DblNumVec  tempVecLocal;

          DistNumVecToDistRowVec(
              distResidual,
              tempVecLocal,
              domain_.numGridFine,
              numElem_,
              fft.localNzStart,
              fft.localNz,
              fft.isInGrid,
              domain_.colComm );

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

          if( fft.isInGrid ){

            for( Int i = 0; i < ntotLocal; i++ ){
              fft.inputComplexVecLocal(i) = Complex( 
                  tempVecLocal(i), 0.0 );
            }
            fftw_execute( fft.forwardPlan );

            for( Int i = 0; i < ntotLocal; i++ ){
              // Do not touch the zero frequency
              // Procedure taken from VASP
              if( fft.gkkLocal(i) != 0 ){
                fft.outputComplexVecLocal(i) *= fft.gkkLocal(i) / 
                  ( fft.gkkLocal(i) + 2.0 * PI * KerkerB );
                //                fft.outputComplexVecLocal(i) *= std::min(fft.gkkLocal(i) / 
                //                        ( fft.gkkLocal(i) + 2.0 * PI * KerkerB ), Amin);
              }
            }
            fftw_execute( fft.backwardPlan );

            for( Int i = 0; i < ntotLocal; i++ ){
              tempVecLocal(i) = fft.inputComplexVecLocal(i).real() / ntot;
            }
          } // if (fft.isInGrid)

          // Convert tempVecLocal to distPrecResidual in the DistNumVec format 

          DistRowVecToDistNumVec(
              tempVecLocal,
              distPrecResidual,
              domain_.numGridFine,
              numElem_,
              fft.localNzStart,
              fft.localNz,
              fft.isInGrid,
              domain_.colComm );



          return ;
        }         // -----  end of method SCFDG::KerkerPrecond  ----- 


      void SCFDG::PrintState    ( ) {

        HamiltonianDG&  hamDG = *hamDGPtr_;

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
        statusOFS << std::endl;
        // FIXME
        //    statusOFS 
        //        << "NOTE:  Ecor  = Exc - EVxc - Ehart - Eself" << std::endl
        //      << "       Etot  = Ekin + Ecor" << std::endl
        //      << "       Efree = Etot    + Entropy" << std::endl << std::endl;
        Print(statusOFS, "EfreeHarris       = ",  EfreeHarris_, "[au]");
        //            FIXME
        //    Print(statusOFS, "EfreeSecondOrder  = ",  EfreeSecondOrder_, "[au]");
        Print(statusOFS, "Etot              = ",  Etot_, "[au]");
        Print(statusOFS, "Efree             = ",  Efree_, "[au]");
        Print(statusOFS, "Ekin              = ",  Ekin_, "[au]");
        Print(statusOFS, "Ehart             = ",  Ehart_, "[au]");
        Print(statusOFS, "EVxc              = ",  EVxc_, "[au]");
        Print(statusOFS, "Exc               = ",  Exc_, "[au]"); 
        Print(statusOFS, "Evdw              = ",  Evdw_, "[au]"); 
        Print(statusOFS, "Eself             = ",  Eself_, "[au]");
        Print(statusOFS, "EionSR            = ",  hamDG.EIonSR(), "[au]");
        Print(statusOFS, "Ecor              = ",  Ecor_, "[au]");
        Print(statusOFS, "Fermi             = ",  fermi_, "[au]");


        return ;
      }         // -----  end of method SCFDG::PrintState  ----- 

      void
        SCFDG::UpdateMDParameters    ( )
        {
          scfOuterMaxIter_ = esdfParam.MDscfOuterMaxIter;
          useEnergySCFconvergence_ = 1;

          return ;
        }         // -----  end of method SCFDG::UpdateMDParameters  ----- 



    } // namespace dgdft
