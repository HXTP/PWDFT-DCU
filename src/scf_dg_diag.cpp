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

void
SCFDG::InnerSolver ( Int outerIter )
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

  // The first inner iteration does not update the potential, and
  // construct the global Hamiltonian matrix from scratch
  GetTime(timeSta);

  // Method 1: Using diagonalization method
  // With a versatile choice of processors for using ScaLAPACK.
  // Or using Chebyshev filtering

  if( solutionMethod_ == "diag" ){

//    statusOFS << " InnerDG SCF " << innerIter  << " : DIAG DG Hamiltonian Matrix" << std::endl;

    if(Diag_SCFDG_by_Cheby_ == 1 ){
      // Chebyshev filtering based diagonalization
      GetTime(timeSta);
      if(scfdg_ion_dyn_iter_ != 0){
        if(SCFDG_use_comp_subspace_ == 1){
          if((scfdg_ion_dyn_iter_ % SCFDG_CS_ioniter_regular_cheby_freq_ == 0)
             && (outerIter <= Second_SCFDG_ChebyOuterIter_ / 2)){
            // Just some adhoc criterion used here
            // Usual CheFSI to help corrrect drift / SCF convergence
#if ( _DEBUGlevel_ >= 1 )                
            statusOFS << std::endl << " Calling Second stage Chebyshev Iter in iondynamics step to improve drift / SCF convergence ..." << std::endl;    
#endif
            scfdg_GeneralChebyStep(Second_SCFDG_ChebyCycleNum_, Second_SCFDG_ChebyFilterOrder_);

            SCFDG_comp_subspace_engaged_ = 0;
          }
          else{  
          // Decide serial or parallel version here
            if(SCFDG_comp_subspace_parallel_ == 0){  
#if ( _DEBUGlevel_ >= 1 )
              statusOFS << std::endl << " Calling Complementary Subspace Strategy (serial subspace version) ...  " << std::endl;
#endif
              scfdg_complementary_subspace_serial(General_SCFDG_ChebyFilterOrder_);
            }
            else{
#if ( _DEBUGlevel_ >= 1 )
              statusOFS << std::endl << " Calling Complementary Subspace Strategy (parallel subspace version) ...  " << std::endl;
#endif
              scfdg_complementary_subspace_parallel(General_SCFDG_ChebyFilterOrder_);                     
            }
            // Set the engaged flag 
            SCFDG_comp_subspace_engaged_ = 1;
          }
        } // if(SCFDG_use_comp_subspace_ == 1)
        else{
            // Just some adhoc criterion used here
          if(outerIter <= Second_SCFDG_ChebyOuterIter_ / 2){
          // Need to re-use current guess, so do not call the first Cheby step
#if ( _DEBUGlevel_ >= 1 )              
            statusOFS << std::endl << " Calling Second stage Chebyshev Iter in iondynamics step " << std::endl;         
#endif
            scfdg_GeneralChebyStep(Second_SCFDG_ChebyCycleNum_, Second_SCFDG_ChebyFilterOrder_);
          }
          else{     
          // Subsequent MD Steps
#if ( _DEBUGlevel_ >= 1 )              
            statusOFS << std::endl << " Calling General Chebyshev Iter in iondynamics step " << std::endl;
#endif
            scfdg_GeneralChebyStep(General_SCFDG_ChebyCycleNum_, General_SCFDG_ChebyFilterOrder_); 
          }
        } //if(SCFDG_use_comp_subspace_ != 1)
      } // if (scfdg_ion_dyn_iter_ != 0)
      else{    
        // 0th MD / Geometry Optimization step (or static calculation)     
        if(outerIter == 1){
#if ( _DEBUGlevel_ >= 1 )
          statusOFS << std::endl << " Calling First Chebyshev Iter  " << std::endl;
#endif
          scfdg_FirstChebyStep(First_SCFDG_ChebyCycleNum_, First_SCFDG_ChebyFilterOrder_);
        }
        else if(outerIter > 1 && outerIter <= Second_SCFDG_ChebyOuterIter_){
#if ( _DEBUGlevel_ >= 1 )
          statusOFS << std::endl << " Calling Second Stage Chebyshev Iter  " << std::endl;
#endif
          scfdg_GeneralChebyStep(Second_SCFDG_ChebyCycleNum_, Second_SCFDG_ChebyFilterOrder_);
        }
        else{  
          if(SCFDG_use_comp_subspace_ == 1){
            // Decide serial or parallel version here
            if(SCFDG_comp_subspace_parallel_ == 0){  
#if ( _DEBUGlevel_ >= 1 )
              statusOFS << std::endl << " Calling Complementary Subspace Strategy (serial subspace version)  " << std::endl;
#endif
              scfdg_complementary_subspace_serial(General_SCFDG_ChebyFilterOrder_);
            }
            else{
#if ( _DEBUGlevel_ >= 1 )
              statusOFS << std::endl << " Calling Complementary Subspace Strategy (parallel subspace version)  " << std::endl;
#endif
              scfdg_complementary_subspace_parallel(General_SCFDG_ChebyFilterOrder_);       
            }
            // Now set the engaged flag 
            SCFDG_comp_subspace_engaged_ = 1;
          }
          else{
#if ( _DEBUGlevel_ >= 1 )                  
            statusOFS << std::endl << " Calling General Chebyshev Iter  " << std::endl;
#endif
            scfdg_GeneralChebyStep(General_SCFDG_ChebyCycleNum_, General_SCFDG_ChebyFilterOrder_); 
          }
        }
      } // end of if(scfdg_ion_dyn_iter_ != 0)

      GetTime( timeEnd );

      if(SCFDG_comp_subspace_engaged_ == 1){
        statusOFS << std::endl << " Total time for Complementary Subspace Method is " << timeEnd - timeSta << " [s]" << std::endl << std::endl;
      }
      else{
          statusOFS << std::endl << " Total time for diag DG matrix via Chebyshev filtering is " << timeEnd - timeSta << " [s]" << std::endl << std::endl;
      }

      DblNumVec& eigval = hamDG.EigVal();          

    } // if(Diag_SCFDG_by_Cheby_ == 1 )
    else // DIAG :: call the ELSI interface and old Scalapack interface 
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
      DistElemMatToScaMat2( hamDG.HMat(), descH,
          scaH, hamDG.ElemBasisIdx(), domain_.comm,
          domain_.colComm, mpirankElemVec,
          mpirankScaVec );
      GetTime( timeConversionEnd );

#if ( _DEBUGlevel_ >= 1 )
      statusOFS << "Time for converting from DistElemMat to ScaMat is " <<
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
#if ( _DEBUGlevel_ >= 2 )
        if( diagSolutionMethod_ == "scalapack"){
           statusOFS << "InnerSCF: Time for Scalapack::diag " <<
             timeConversionEnd - timeConversionSta << " [s]" 
             << std::endl << std::endl;
        }
        else{
          statusOFS << "InnerSCF: Time for ELSI::ELPA  Diag " <<
            timeConversionEnd - timeConversionSta << " [s]" 
            << std::endl << std::endl;
        }
#endif
        for( Int i = 0; i < hamDG.NumStateTotal(); i++ ){
          eigval[i] = eigs[i];
        }
      } //if(contxt_ >= -1)

      GetTime( timeConversionSta );
      ScaMatToDistNumMat2( scaZ, hamDG.Density().Prtn(), 
          hamDG.EigvecCoef(), hamDG.ElemBasisIdx(), domain_.comm,
          domain_.colComm, mpirankElemVec, mpirankScaVec, 
          hamDG.NumStateTotal() );
      GetTime( timeConversionEnd );
#if ( _DEBUGlevel_ >= 1 )
      statusOFS << "Time for converting from ScaMat to DistNumMat is " <<
        timeConversionEnd - timeConversionSta << " [s]" << std::endl << std::endl;
#endif
      GetTime( timeConversionSta );
      for( Int k = 0; k < numElem_[2]; k++ )
        for( Int j = 0; j < numElem_[1]; j++ )
          for( Int i = 0; i < numElem_[0]; i++ ){
            Index3 key( i, j, k );
            if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
              DblNumMat& localCoef  = hamDG.EigvecCoef().LocalMap()[key];
              MPI_Bcast(localCoef.Data(), localCoef.m() * localCoef.n(), 
                  MPI_DOUBLE, 0, domain_.rowComm);
            }
      } 
      GetTime( timeConversionEnd );
#if ( _DEBUGlevel_ >= 1 )
      statusOFS << "Time for MPI_Bcast eigval and localCoef is " <<
        timeConversionEnd - timeConversionSta << " [s]" << std::endl << std::endl;
#endif

      MPI_Barrier( domain_.comm );
      MPI_Barrier( domain_.rowComm );
      MPI_Barrier( domain_.colComm );

      GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 1 )
      if( diagSolutionMethod_ == "scalapack"){
        statusOFS << "InnerSCF: Time for diag DG matrix via ScaLAPACK is " <<
          timeEnd - timeSta << " [s]" << std::endl << std::endl;
      }
      else{
        statusOFS << "InnerSCF: Time for diag DG matrix via ELSI:ELPA is " <<
          timeEnd - timeSta << " [s]" << std::endl << std::endl;
      }
#endif

      // Communicate the eigenvalues
      Int mpirankScaSta = mpirankScaVec[0];
      MPI_Bcast(eigval.Data(), hamDG.NumStateTotal(), MPI_DOUBLE, 
         mpirankScaVec[0], domain_.comm);

    } // End of ELSI

    // Post processing

    Evdw_ = 0.0;

    if(SCFDG_comp_subspace_engaged_ == 1)
    {
      // Calculate Harris energy without computing the occupations
      CalculateHarrisEnergy();
    }        
    else{
//      statusOFS << " InnerDG SCF " << innerIter  << " : Calculate New Density Matrix" << std::endl;
     // Compute the occupation rate - specific smearing types dealt with within this function
      CalculateOccupationRate( hamDG.EigVal(), hamDG.OccupationRate() );

      if( hamDG.IsEXXActive() ) {
        GetTime(timeSta);
        scfdg_compute_fullDM();
        GetTime( timeEnd );
//        statusOFS << "InnerSCF: Recalculate density matrix for diag method " << std::endl;
      }

#if ( _DEBUGlevel_ >= 1 )
      statusOFS << "InnerSCF: DIAG Time for computing full density matrix " <<
        timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

      // Compute the Harris energy functional.  
      // NOTE: In computing the Harris energy, the density and the
      // potential must be the INPUT density and potential without ANY
      // update.
      GetTime(timeSta);
      CalculateHarrisEnergy();
      GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 1 )
      statusOFS << "InnerSCF: Time for computing harris energy " <<
        timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
    }

    // Calculate the new electron density
    GetTime( timeSta );

    if(SCFDG_comp_subspace_engaged_ == 1){
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
      hamDG.CalculateDensityDM2(hamDG.Density(), hamDG.DensityLGL(), cheby_diag_dmat );
    } // (SCFDG_comp_subspace_engaged_ == 1)
    else {
      Int temp_m = hamDG.NumBasisTotal() / (numElem_[0] * numElem_[1] * numElem_[2]); // Average no. of ALBs per element
      Int temp_n = hamDG.NumStateTotal();
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
        for(Int iter_scale = 0; iter_scale < eigvecs_local.n(); iter_scale ++)
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
        hamDG.CalculateDensityDM2(hamDG.Density(), hamDG.DensityLGL(), cheby_diag_dmat );
      }
      else
      {  
        // FIXME 
        // Do not need the conversion from column to row partition as well
//        statusOFS << " InnerDG SCF " << innerIter  << " : Calculate New Density After DIAG" << std::endl;

      // xmqin 20240218 calculate density using different methods
//
//        if( InnermixVariable_ == "densitymatrix" || InnermixVariable_ == "hamiltonian"  ) { //||  hamDG.IsEXXActive() ) {
//          hamDG.CalculateDensityDM2(hamDG.Density(), hamDG.DensityLGL(), distDMMat_ );
//          statusOFS << "InnerSCF: Recalculate density from density matrix" << std::endl;
//        }
//        else{
          hamDG.CalculateDensity( hamDG.Density(), hamDG.DensityLGL() );
//        }

        // 2016/11/20: Add filtering of the density. Impacts
        // convergence at the order of 1e-5 for the LiH dimer
        // example and therefore is not activated
#if 0
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
          statusOFS << " fac " << fac << std::endl;
          Print( statusOFS, "Sum Rho on uniform grid (after normalization again) = ", sumRho );
          statusOFS << std::endl;
#endif
#endif

      }// ((Diag_SCFDG_by_Cheby_ == 1) && (temp_m < temp_n))

    } // if(SCFDG_comp_subspace_engaged_ == 1)

    GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 1 )
    statusOFS << "InnerSCF: Time for computing density in the global domain is " <<
        timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

    // *******************************************************************************
    // Update Step :: Potential, KS and 2rd Energy
    //
    // Update the output potential, and the KS and second order accurate
    // energy

    // Update the Hartree energy and the exchange correlation energy and
    // potential for computing the KS energy and the second order
    // energy.
    // NOTE Vtot should not be updated until finishing the computation
    // of the energies.

//    statusOFS << " InnerDG SCF " << innerIter  << " : Update potential ?" << std::endl;
//    statusOFS << "InnerSCF: Recalculate potential after DIAG " << std::endl ;
    if( isCalculateGradRho_  ){
      GetTime( timeSta );
      hamDG.CalculateGradDensity(  *distfftPtr_ );
      GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 1 )
      statusOFS << "Time for calculating gradient of density is " <<
        timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
    }

    GetTime( timeSta );
    hamDG.CalculateXC( Exc_, hamDG.Epsxc(), hamDG.Vxc(), *distfftPtr_ );

    GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 1 )
    statusOFS << "Time for computing Exc in the global domain is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

    GetTime( timeSta );

    hamDG.CalculateHartree( hamDG.Vhart(), *distfftPtr_ );

    GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 1 )
    statusOFS << "InnerSCF: Time for computing Vhart in the global domain is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
    // Compute the second order accurate energy functional.
    // NOTE: In computing the second order energy, the density and the
    // potential must be the OUTPUT density and potential without ANY
    // MIXING.
    GetTime( timeSta );

    CalculateSecondOrderEnergy();
      // Compute the KS energy 

    CalculateKSEnergy();

    GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 1 )
    statusOFS << "InnerSCF: Time for computing KSEnergy in the global domain is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

    // Update the total potential AFTER updating the energy

    // No external potential

    // Compute the new total potential

    GetTime( timeSta );

    hamDG.CalculateVtot( hamDG.Vtot() );

    GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 1 )
    statusOFS << "InnerSCF: Time for computing Vtot in the global domain is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

    // **********************************************************************************
    // Atomic Forces
    //
    // Compute the force at every step
    if( esdfParam.isCalculateForceEachSCF ){

      // Compute force
      GetTime( timeSta );

      if(SCFDG_comp_subspace_engaged_ == false)
      {
        if(1)
        {
          statusOFS << std::endl << "InnerSCF: Computing forces using eigenvectors ... " << std::endl;
          hamDG.CalculateForce( *distfftPtr_ );
        }
        else
        {         
          // Alternate (highly unusual) routine for debugging purposes
          // Compute the Full DM (from eigenvectors) and call the PEXSI force evaluator

          double extra_timeSta, extra_timeEnd;

          statusOFS << std::endl << "InnerSCF: Computing forces using Density Matrix ... ";
          statusOFS << std::endl << "InnerSCF: Computing full Density Matrix from eigenvectors ...";
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

      } //  if(SCFDG_comp_subspace_engaged_ == false)

      GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 1 )
      statusOFS << "Time for computing the force is " <<
        timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
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
      } // output

    } //  if( esdfParam.isCalculateForceEachSCF )

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
      statusOFS << "Time for computing the a posteriori error is " <<
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
    // Atomic Forces
    //**********************************************************************************
  } // IF (DIAG) 

  // Method 2: Using the pole expansion and selected inversion (PEXSI) method
  // FIXME Currently it is assumed that all processors used by DG will be used by PEXSI.
#ifdef _USE_PEXSI_
  // The following version is with intra-element parallelization
//  DistDblNumVec VtotHist; // check check
  // check check
  Real difNumElectron = 0.0;
  Real totalEnergyH, totalFreeEnergy;

  if( solutionMethod_ == "pexsi" ){
    // Initialize the history of vtot , check check
    for( Int k=0; k< numElem_[2]; k++ )
      for( Int j=0; j< numElem_[1]; j++ )
        for( Int i=0; i< numElem_[0]; i++ ) {
          Index3 key = Index3(i,j,k);
          if( distEigSolPtr_->Prtn().Owner(key) == (mpirank / dmRow_) ){
            DistDblNumVec& vtotCur = hamDG.Vtot();
            VtotHist_.LocalMap()[key] = vtotCur.LocalMap()[key];
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

#if ( _DEBUGlevel_ >= 1 )
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
    }// if( mpirankRow == 0)

    // So energy must be obtained from DM as in totalEnergyH
    // and free energy is nothing but energy..
    Real totalEnergyH, totalFreeEnergy;
    //if( (mpirankRow < numProcPEXSICommRow_) && (mpirankCol < numProcPEXSICommCol_) )
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
          plan_,
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

#ifdef ELSI
      c_elsi_set_sparsity( HSparseMat.nnz,
                           HSparseMat.nnzLocal,
                           HSparseMat.colptrLocal.m() - 1,
                           HSparseMat.rowindLocal.Data(),
                           HSparseMat.colptrLocal.Data() );

      c_elsi_customize_pexsi( pexsiOptions_.temperature,
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

      // New version of PEXSI driver, uses inertia count + pole update
      // strategy. No Newton's iteration. But this is not very stable.
      pexsiOptions_.method = esdfParam.pexsiMethod;
      pexsiOptions_.nPoints = esdfParam.pexsiNpoint;

      PPEXSIDFTDriver2(
          plan_,
          &pexsiOptions_,
          numElectronExact,
          &muPEXSI,
          &numElectronPEXSI,         
          &numTotalInertiaIter,
          &info );

      // New version of PEXSI driver, use inertia count + pole update.
      // two method of pole expansion. default is 2

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
      difNumElectron = std::abs(numElectronPEXSI - numElectronExact);

      // Heuristics for the next step
      //pexsiOptions_.muMin0 = muMinInertia - 5.0 * pexsiOptions_.temperature;
      //pexsiOptions_.muMax0 = muMaxInertia + 5.0 * pexsiOptions_.temperature;

      // Retrieve the PEXSI data

      // FIXME: Hack: in PEXSIDriver3, only DM is available.

      if( ( mpirankRow == 0 ) && (mpirankCol < numProcPEXSICommCol_) ){
      // if( mpirankRow == 0 ){
        Real totalEnergyS;

        GetTime( timeSta );

        CopyPattern( HSparseMat, DMSparseMat );
        CopyPattern( HSparseMat, EDMSparseMat );
        CopyPattern( HSparseMat, FDMSparseMat );

        statusOFS << "Before retrieve" << std::endl;
        PPEXSIRetrieveRealDFTMatrix(
            // pexsiPlan_,
            plan_,
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
        // FIXME: Hack: there is no free energy really. totalEnergyS is to be added later
        statusOFS << "NOTE: Free energy = Energy in PPEXSIDFTDriver3!" << std::endl;

        statusOFS << std::endl
          << "Results obtained from PEXSI:" << std::endl
          << "Total energy (H*DM)         = " << totalEnergyH << std::endl
          << "Total energy (S*EDM)        = " << totalEnergyS << std::endl
          << "Total free energy           = " << totalFreeEnergy << std::endl 
          << "InertiaIter                 = " << numTotalInertiaIter << std::endl
          << "mu                          = " << muPEXSI << std::endl
          << "numElectron                 = " << numElectronPEXSI << std::endl 
          << std::endl;

        if( info != 0 ){
          std::ostringstream msg;
          msg 
            << "PEXSI data retrieval returns info " << info << std::endl;
          ErrorHandling( msg.str().c_str() );
        }
      }  //  if( ( mpirankRow == 0 ) && (mpirankCol < numProcPEXSICommCol_) )
#endif
    } // if( (mpirankRow < numProcPEXSICommRow_) && (mpirankCol < numProcPEXSICommCol_) )

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
    CalculateHarrisEnergyDM( totalFreeEnergy, distFDMMat_ );
    GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for calculating the Harris energy is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

    // Evaluate the electron density
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

      if( isCalculateGradRho_  ){
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

//      if( hamDG.IsHybrid() && hamDG.IsEXXActive()){
//        GetTime( timeSta );
//        if(esdfParam.isDGHFISDF){
//          hamDG.CalculateDGHFXMatrix_ISDF( Ehfx_, distDMMat_ );
//        }
//        else{
//          hamDG.CalculateDGHFXMatrix( Ehfx_, distDMMat_ );
//          hamDG.CalculateDGHFXEnergy( Ehfx_, distDMMat_, hamDG.HFXMat() );
//
//        }
//        GetTime( timeEnd );
//#if ( _DEBUGlevel_ >= 0 )
//        statusOFS << "InnerSCF: Time for computing HFX in the global domain is " <<
//          timeEnd - timeSta << " [s]" << std::endl << std::endl;
//#endif
//      }

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
    statusOFS << "Time for computing the potential is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

    // Compute the force at every step
    if( esdfParam.isCalculateForceEachSCF ){
      // Compute force
      GetTime( timeSta );
      hamDG.CalculateForceDM( *distfftPtr_, distDMMat_ );
      GetTime( timeEnd );
      statusOFS << "Time for computing the force is " <<
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

    // TODO Evaluate the a posteriori error estimator

    GetTime( timePEXSIEnd );
#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for PEXSI evaluation is " <<
      timePEXSIEnd - timePEXSISta << " [s]" << std::endl << std::endl;
#endif
  } //if( solutionMethod_ == "pexsi" )

#endif

  return ;

}         // -----  end of method SCFDG::InnerIterate  ----- 

}
