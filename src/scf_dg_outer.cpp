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
//#include  "scfdg_upper_end_of_spectrum.hpp"

namespace  dgdft{

using namespace dgdft::DensityComponent;
using namespace dgdft::esdf;
using namespace dgdft::scalapack;

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

  Real timeSta0, timeEnd0;
  Real timeSta, timeEnd;

  Domain dmElem;
  for( Int d = 0; d < DIM; d++ ){
    dmElem.length[d]   = domain_.length[d] / numElem_[d];
    dmElem.numGrid[d]  = domain_.numGrid[d] / numElem_[d];
    dmElem.numGridFine[d]  = domain_.numGridFine[d] / numElem_[d];
    dmElem.posStart[d] = ( numElem_[d] > 1 ) ? dmElem.length[d] : 0;
  }

  HamiltonianDG&  hamDG = *hamDGPtr_;
  DistFourier&    fftDG = *distfftPtr_;

  // xmqin add for HSE06
  //  FixMe:  First outer SCF for hybrids using PBE
  if( !hamDG.IsHybrid() || ( hamDG.IsHybrid() && !hamDG.IsEXXActive() ) ){
    std::ostringstream msg;
    msg << "Starting Regular DGDFT SCF Iteration.";
    PrintBlock( statusOFS, msg.str() );
    bool isSCFConverged = false;

    GetTime( timeSta0 );

    if( hamDG.IsHybrid() && !hamDG.IsEXXActive() ){
      hamDG.Setup_XC( "XC_GGA_XC_PBE");
    }

    if( isCalculateGradRho_ ) {
      GetTime( timeSta );
      hamDG.CalculateGradDensity( fftDG );
      GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 1 )
      statusOFS << "Time for calculating gradient of density is " <<
        timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
    }

    GetTime( timeSta );
    hamDG.CalculateXC( Exc_, hamDG.Epsxc(), hamDG.Vxc(), fftDG );
    GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 1 )
    statusOFS << "Time for calculating XC is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
    // Compute the Hartree potential
    GetTime( timeSta );
    hamDG.CalculateHartree( hamDG.Vhart(), fftDG );
    GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 1 )
    statusOFS << "Time for calculating Hartree is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
    // No external potential
    // Compute the total potential
    GetTime( timeSta );
    hamDG.CalculateVtot( hamDG.Vtot() );
    GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 1 )
    statusOFS << "Time for calculating Vtot is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

    GetTime( timeEnd0 );

    statusOFS << "OuterSCF: Time for initial DG effective potential is " <<
      timeEnd0 - timeSta0 << " [s]" << std::endl << std::endl;

    Real timeIterStart(0), timeIterEnd(0);
    Real timeTotalStart(0), timeTotalEnd(0);

    scfTotalInnerIter_  = 0;

    GetTime( timeTotalStart );

    Int iter;

    // Total number of SVD basis functions. Determined at the first
    // outer SCF and is not changed later. This facilitates the reuse of
    // symbolic factorization

    for (iter=1; iter <= DFTscfOuterMaxIter_; iter++) {
      if ( isSCFConverged && (iter >= DFTscfOuterMinIter_ ) ) break;
      // *********************************************************************
      // Performing each iteartion
      // *********************************************************************
      {
        std::ostringstream msg;
        msg << "Outer SCF iteration # " << iter;
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
        statusOFS << "OuterSCF:: Time for updating local DG potentials is " <<
            timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
      }

      // *********************************************************************
      // Solve the basis functions in the extended element
      // *********************************************************************

      Real timeBasisSta, timeBasisEnd;
      GetTime(timeBasisSta);

      for( Int k = 0; k < numElem_[2]; k++ )
        for( Int j = 0; j < numElem_[1]; j++ )
          for( Int i = 0; i < numElem_[0]; i++ ){
            Index3 key( i, j, k );
            if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){

              EigenSolver&  eigSol = distEigSolPtr_->LocalMap()[key];
              DblNumVec&    vtotExtElem = eigSol.Ham().Vtot();
              Index3 numGridExtElem = eigSol.FFT().domain.numGrid;
              Index3 numGridExtElemFine = eigSol.FFT().domain.numGridFine;
              Index3 numLGLGrid = hamDG.NumLGLGridElem();

              Index3 numGridElemFine = dmElem.numGridFine;

              // Skip the interpoation if there is no adaptive local basis function.  
              if( eigSol.Psi().NumState() == 0 ){
                hamDG.BasisLGL().LocalMap()[key].Resize( numLGLGrid.prod(), 0 );  
                hamDG.BasisUniformFine().LocalMap()[key].Resize( numGridElemFine.prod(), 0 );  
                continue;
              }

              // Solve the basis functions in the extended element
              Real eigTolNow;
              if( esdfParam.isEigToleranceDynamic ){
                // Dynamic strategy to control the tolerance
                if( iter == 1 )
                  eigTolNow = 1e-2;
                else
                  eigTolNow = eigTolerance_;
              }
              else{
                // Static strategy to control the tolerance
                eigTolNow = eigTolerance_;
              }

              Int numEig = (eigSol.Psi().NumStateTotal())-numUnusedState_;
#if ( _DEBUGlevel_ >= 2 ) 
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
//                    statusOFS << " >>>> Calling LOBPCG for ALB generation on extended element ..." << std::endl;
#ifndef _COMPLEX_
                    eigSol.LOBPCGSolveReal(numEig, iter, eigMaxIter_, eigMinTolerance_, eigTolNow );    
#endif
                  }
                  else
                  {
//                    statusOFS << " >>>> Calling CheFSI with random guess for ALB generation on extended element ..." << std::endl;
                    eigSol.FirstChebyStep(numEig, First_SCF_PWDFT_ChebyCycleNum_, First_SCF_PWDFT_ChebyFilterOrder_);
                  }
//                  statusOFS << std::endl;

                }
                else
                {
//                  statusOFS << " >>>> Calling CheFSI with previous ALBs for generation of new ALBs ..." << std::endl;
//                  statusOFS << " >>>> Will carry out " << eigMaxIter_ << " CheFSI cycles." << std::endl;
                  for (Int cheby_iter = 1; cheby_iter <= eigMaxIter_; cheby_iter ++)
                  {
//                    statusOFS << std::endl << " >>>> CheFSI for ALBs : Cycle " << cheby_iter << " of " << eigMaxIter_ << " ..." << std::endl;
                    eigSol.GeneralChebyStep(numEig, General_SCF_PWDFT_ChebyFilterOrder_);
                  }
//                  statusOFS << std::endl;
                }
              }
              else if(Diag_SCF_PWDFT_by_PPCG_ == 1)
              {
                // Use LOBPCG on very first step, i.e., while starting from random guess
                if(iter <= 1)
                {
//                  statusOFS << " >>>> Calling LOBPCG for ALB generation on extended element ..." << std::endl;
                  eigSol.LOBPCGSolveReal(numEig, iter, eigMaxIter_, eigMinTolerance_, eigTolNow );    
                }
                else
                {
//                  statusOFS << " >>>> Calling PPCG with previous ALBs for generation of new ALBs ..." << std::endl;
                  eigSol.PPCGSolveReal(numEig, iter, eigMaxIter_, eigMinTolerance_, eigTolNow );
                }
              }             
              else 
              {
                Int eigDynMaxIter = eigMaxIter_;
                eigSol.LOBPCGSolveReal(numEig, iter, eigDynMaxIter, eigMinTolerance_, eigTolNow );
              }

              GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
              statusOFS << "ALBs: Time for calculating extended Phi via PWDFT is " <<
                timeEnd - timeSta << " [s]" << std::endl;
#endif

              // Print out the information
#if ( _DEBUGlevel_ >= 2 )
              Real maxRes = 0.0, avgRes = 0.0;
              for(Int ii = 0; ii < eigSol.EigVal().m(); ii++){
                if( maxRes < eigSol.ResVal()(ii) ){
                  maxRes = eigSol.ResVal()(ii);
                }
              avgRes = avgRes + eigSol.ResVal()(ii);
              Print( statusOFS, 
                     "basis#   = ", ii, 
                     "eigval   = ", eigSol.EigVal()(ii),
                     "resval   = ", eigSol.ResVal()(ii));
              }
              avgRes = avgRes / eigSol.EigVal().m();
              statusOFS << std::endl;
              Print(statusOFS, " Max residual of basis = ", maxRes );
              Print(statusOFS, " Avg residual of basis = ", avgRes );
              statusOFS << std::endl;
#endif

              GetTime( timeSta );
              Spinor& psi = eigSol.Psi();
              DblNumMat& basis = hamDG.BasisLGL().LocalMap()[key];
              SVDLocalizeBasis ( iter, numGridExtElem,
                    numGridElemFine, numLGLGrid, psi, basis );
              GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
              statusOFS << "ALBs: Time for ALBs from SVD and localizatoin is " <<
                timeEnd - timeSta << " [s]" << std::endl;
#endif
            } // own this element
      } // for (i)

      GetTime( timeBasisEnd );
#if ( _DEBUGlevel_ >= 0 )
      statusOFS << std::endl << "OuterSCF: Time for generating ALBs is " <<
        timeBasisEnd - timeBasisSta << " [s]" << std::endl << std::endl;
#endif

// =================================================================================
      // Routine for re-orienting eigenvectors based on current basis set
      if(Diag_SCFDG_by_Cheby_ == 1)
      {
//        Real timeSta, timeEnd;
        Real extra_timeSta, extra_timeEnd;

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

                statusOFS << std::endl << " AlltoallForward: Changing distribution of local basis functions ... ";
                GetTime(extra_timeSta);
                AlltoallForward(band_distrib_local_basis, row_distrib_local_basis, domain_.rowComm);
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

                  Real *ptr_0 = ALB_LGL_deque_[0].Data();
                  Real *ptr_1 = ALB_LGL_deque_[1].Data();

                  blas::Gemm( 'T', 'N', width, width, heightLGLLocal,
                      1.0, ptr_1, heightLGLLocal, 
                      ptr_0, heightLGLLocal, 
                      0.0, Overlap_Mat_Temp.Data(), width );

                  // Reduce along rowComm (i.e., along the intra-element direction)
                  // to compute the actual overlap matrix
                  MPI_Allreduce( Overlap_Mat_Temp.Data(), 
                      Overlap_Mat.Data(), 
                      width * width, 
                      MPI_DOUBLE, 
                      MPI_SUM, 
                      domain_.rowComm );

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

                  blas::Gemm( 'N', 'N', Overlap_Mat.m(), eigvecs_local.n(), Overlap_Mat.n(), 
                      1.0, Overlap_Mat.Data(), Overlap_Mat.m(), 
                      temp_loc_eigvecs_buffer.Data(), temp_loc_eigvecs_buffer.m(), 
                      0.0, eigvecs_local.Data(), eigvecs_local.m());

                  GetTime(extra_timeEnd);
                  statusOFS << " Done. ( " << (extra_timeEnd - extra_timeSta) << " s)" << std::endl;

                  ALB_LGL_deque_.pop_front();
                }        

              } // End of if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) )

        } // End of loop over key indices i.e., for( Int i = 0; i < numElem_[0]; i++ )

        if( iter > 1)
        {
          GetTime(timeEnd);
          statusOFS << std::endl << " All steps of basis rotation completed. ( " << (timeEnd - timeSta) << " s )"<< std::endl;
        }
      } // End of if(Diag_SCFDG_by_Cheby_ == 1)

      // *********************************************************************
      // Inner SCF iteration 
      //
      // Assemble and diagonalize the DG matrix until convergence is
      // reached for updating the basis functions in the next step.
      // *********************************************************************

      GetTime(timeSta);

      // Save the mixing variable in the outer SCF iteration 
      if( OutermixVariable_ == "density" || OutermixVariable_ == "potential" ){
        for( Int k = 0; k < numElem_[2]; k++ )
          for( Int j = 0; j < numElem_[1]; j++ )
            for( Int i = 0; i < numElem_[0]; i++ ){
              Index3 key( i, j, k );
              if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
                if( OutermixVariable_ == "density" ){
                  DblNumVec& oldVec = hamDG.Density().LocalMap()[key];
                  mixOuterSave_.LocalMap()[key] = oldVec;
                }
                else if( OutermixVariable_ == "potential" ){
                  DblNumVec& oldVec = hamDG.Vtot().LocalMap()[key];
                  mixOuterSave_.LocalMap()[key] = oldVec;
                }
              } // own this element
        } // for (i)
      }

      // Main function here
      InnerIterate( iter );

      if( mpirank == 0 ){
        PrintState( );
      }

      MPI_Barrier( domain_.comm );
      GetTime( timeEnd );
      statusOFS << "OuterSCF: Time for all inner SCF iterations is " <<
        timeEnd - timeSta << " [s]" << std::endl << std::endl;

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
      md_scf_etot_diff_ = std::abs(md_scf_etot_old_ - md_scf_etot_) / double(numAtom);

      //--------------------------------------------------------------------------------------
      // Compute the error of the mixing variable 
      {
        Real normMixDifLocal = 0.0, normMixOldLocal = 0.0;
        Real normMixDif, normMixOld;
        for( Int k = 0; k < numElem_[2]; k++ )
          for( Int j = 0; j < numElem_[1]; j++ )
            for( Int i = 0; i < numElem_[0]; i++ ){
              Index3 key( i, j, k );
              if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
                if( OutermixVariable_ == "density" ){
                  DblNumVec& oldVec = mixOuterSave_.LocalMap()[key];
                  DblNumVec& newVec = hamDG.Density().LocalMap()[key];

                  for( Int p = 0; p < oldVec.m(); p++ ){
                    normMixDifLocal += pow( oldVec(p) - newVec(p), 2.0 );
                    normMixOldLocal += pow( oldVec(p), 2.0 );
                  }
                }
                else if( OutermixVariable_ == "potential" ){
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

        scfOuterNorm_  = normMixDif / normMixOld;

        Print(statusOFS, "OUTERSCF: EfreeHarris                 = ", EfreeHarris_ ); 
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
      } // Compute the error of the mixing variable

//-------------------------------------------------------------------------
      // Print out the state variables of the current iteration
      //    PrintState( );
      // Check for convergence
      if(useEnergySCFconvergence_ == 0)
      { 
        if( (iter > 1) && 
          ( (scfOuterNorm_ < DFTscfOuterTolerance_) && 
            (efreeDifPerAtom_ < DFTscfOuterEnergyTolerance_) ) ){
            /* converged */
          statusOFS << "Outer SCF is converged in " << iter << " steps !" << std::endl;
          isSCFConverged = true;
        }
      }  
      else
      {
        if( (iter > 1) && 
            (md_scf_etot_diff_ < md_scf_etot_diff_tol_) &&
            (md_scf_eband_diff_ < md_scf_eband_diff_tol_) )
        {
          // converged via energy criterion
          statusOFS << " Outer SCF is converged via energy condition in " << iter << " steps !" << std::endl;
          isSCFConverged = true;
        }
      } // if(useEnergySCFconvergence_ == 0)
      // Potential mixing for the outer SCF iteration. or no mixing at all anymore?
      // It seems that no mixing is the best.

      GetTime( timeIterEnd );
      statusOFS << "OuterSCF: Time for this outer SCF iteration = " << timeIterEnd - timeIterStart
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
      statusOFS << "Total number of outer SCF steps for SCF convergence = " <<
        iter - 1 << std::endl;
    }
    else{
      statusOFS << "Total number of outer SCF steps (SCF not converged) = " <<
        DFTscfOuterMaxIter_ << std::endl;
    } // if(scfdg_ion_dyn_iter_ >= 1)


    if( hamDG.IsHybrid() && !hamDG.IsEXXActive() ){
      Real HOMO, LUMO, EG;

      HOMO = hamDG.EigVal()( hamDG.NumOccupiedState()-1 );
      if( hamDG.NumExtraState() > 0 ) {
        LUMO = hamDG.EigVal()( hamDG.NumOccupiedState());
        EG = LUMO - HOMO;
      }

#if ( _DEBUGlevel_ >= 1 )
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

      // Print out the energy
      PrintBlock( statusOFS, "Energy" );
      //statusOFS
      //  << "NOTE:  Ecor  = Exc + Exx - EVxc - Ehart - Eself + Evdw" << std::endl
      //  << "       Etot  = Ekin + Ecor" << std::endl
      //  << "       Efree = Etot + Entropy" << std::endl << std::endl;
      Print(statusOFS, "! EfreeHarris     = ",  EfreeHarris_, "[au]");
      Print(statusOFS, "! Etot            = ",  Etot_, "[au]");
      Print(statusOFS, "! Exc             = ",  Exc_, "[au]");
      Print(statusOFS, "! Exx             = ",  Ehfx_, "[au]");
      Print(statusOFS, "! Efree           = ",  Efree_, "[au]");
      Print(statusOFS, "! Evdw            = ",  Evdw_, "[au]");
      Print(statusOFS, "! Fermi           = ",  fermi_, "[au]");
      Print(statusOFS, "! HOMO            = ",  HOMO*au2ev, "[eV]");
      if( hamDG.NumExtraState() > 0 ){
        Print(statusOFS, "! LUMO            = ",  LUMO*au2ev, "[eV]");
        Print(statusOFS, "! Band Gap        = ",  EG*au2ev, "[eV]");
      }
    }
  } // if( !hamDG.IsHybrid() || !hamDG.IsEXXActive())

  // xmqin add for HSE06
  //  FixMe:  Next outer SCF for hybrids

  if( hamDG.IsHybrid() ){

    std::ostringstream msg;
    msg << "Starting Hybrid DFT SCF iteration.";
    PrintBlock( statusOFS, msg.str() );
    bool isSCFConverged = false;

    GetTime( timeSta0 );
 
    if( hamDG.IsEXXActive() == false )
      hamDG.SetEXXActive(true);

    if(esdfParam.XCType == "XC_HYB_GGA_XC_HSE06")
    {
      hamDG.Setup_XC( "XC_HYB_GGA_XC_HSE06");
    }
    else if (esdfParam.XCType == "XC_HYB_GGA_XC_PBEH")
    {
      hamDG.Setup_XC( "XC_HYB_GGA_XC_PBEH");
    }

//    statusOFS << "Re-calculate XC for Hybrid DFT " << std::endl;

    if( isCalculateGradRho_ ) {
      GetTime( timeSta );
      hamDG.CalculateGradDensity( fftDG );
      GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 1 )
      statusOFS << "Time for calculating gradient of density is " <<
         timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
    }

    GetTime( timeSta );
    hamDG.CalculateXC( Exc_, hamDG.Epsxc(), hamDG.Vxc(), fftDG );
    GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 1 )
    statusOFS << "Time for calculating XC is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
    // Compute the Hartree potential
    GetTime( timeSta );
    hamDG.CalculateHartree( hamDG.Vhart(), fftDG );
    GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 1 )
    statusOFS << "Time for calculating Hartree is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
    // No external potential
    // Compute the total potential
    GetTime( timeSta );
    hamDG.CalculateVtot( hamDG.Vtot() );
    GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 1 )
    statusOFS << "Time for calculating Vtot is " <<
      timeEnd - timeSta << " [s]" << std::endl;
#endif
    GetTime( timeEnd0 );

    statusOFS << " OuterSCF: Time for initial DG local effective potential is " <<
      timeEnd0 - timeSta0 << " [s]" << std::endl << std::endl;


    Real timeIterStart(0), timeIterEnd(0);
    Real timeTotalStart(0), timeTotalEnd(0);

    scfTotalInnerIter_  = 0;

    GetTime( timeTotalStart );

    Int iter;

    // Total number of SVD basis functions. Determined at the first
    // outer SCF and is not changed later. This facilitates the reuse of
    // symbolic factorization

    if( !DGHFXNestedLoop_ ){
      SetupDMMix (); 
    }

    for (iter=1; iter <= HybridscfOuterMaxIter_; iter++) {
      if ( isSCFConverged && (iter >= HybridscfOuterMinIter_ ) ) break;

      // Performing each iteartion
      {
        std::ostringstream msg;
        msg << "Hybrid DFT Outer SCF iteration # " << iter;
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
        statusOFS << "OuterSCF:: Time for updating local potentials in each element and its extended element is " <<
            timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

      }

      if( iter == 1 || solutionMethod_ == "diag" ) {
        GetTime(timeSta);
        scfdg_compute_fullDM();
        GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
        statusOFS << "OuterSCF: Time for initializing density matrix from DIAG method is " <<
           timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
      }

      GetTime( timeSta );
      // Generate new basis for HSE 
      for( Int k = 0; k < numElem_[2]; k++ )
        for( Int j = 0; j < numElem_[1]; j++ )
          for( Int i = 0; i < numElem_[0]; i++ ){
            Index3 key( i, j, k );
            if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){

              DblNumMat& basisOld = hamDG.BasisLGL().LocalMap()[key];
              DblNumMat& basisSave = hamDG.BasisLGLSave().LocalMap()[key];

              basisSave.Resize( basisOld.m(), basisOld.n() );

              SetValue( basisSave, 0.0 );

              blas::Copy( basisOld.m()*basisOld.n(), basisOld.Data(), 1, basisSave.Data(), 1 );
            }
      }
      GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 1 )
      statusOFS << " OuterSCF: Time for saving previous ALBs is " <<
        timeEnd0 - timeSta0 << " [s]" << std::endl << std::endl;
#endif


      // *********************************************************************
      // Solve the basis functions in the extended element
      // *********************************************************************

      Real timeBasisSta, timeBasisEnd;
      GetTime(timeBasisSta);
      // FIXME  magic numbers to fixe the basis
//    if( (iter <= 5) || (efreeDifPerAtom_ >= 1e-3) ){
      for( Int k = 0; k < numElem_[2]; k++ )
        for( Int j = 0; j < numElem_[1]; j++ )
          for( Int i = 0; i < numElem_[0]; i++ ){
            Index3 key( i, j, k );
            if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
              EigenSolver&  eigSol = distEigSolPtr_->LocalMap()[key];
      
             // Need ham for extended elements  xmqin
             // hamDG don't include hamKS for extended element
              Hamiltonian&  ham = eigSol.Ham();
              Spinor&       psi = eigSol.Psi();
              Fourier& fft =  eigSol.FFT();
             
              DblNumVec&    vtotExtElem = eigSol.Ham().Vtot();
              Index3 numGridExtElem = eigSol.FFT().domain.numGrid;
              Index3 numGridExtElemFine = eigSol.FFT().domain.numGridFine;
              Index3 numLGLGrid     = hamDG.NumLGLGridElem();

              Index3 numGridElemFine    = dmElem.numGridFine;

              // Skip the interpoation if there is no adaptive local
              // basis function.  
              if( eigSol.Psi().NumState() == 0 ){
                hamDG.BasisLGL().LocalMap()[key].Resize( numLGLGrid.prod(), 0 );  
                hamDG.BasisUniformFine().LocalMap()[key].Resize( numGridElemFine.prod(), 0 );  
                continue;
              }

              // Solve the basis functions in the extended element
              Real eigTolNow;
              if( esdfParam.isEigToleranceDynamic ){
                // Dynamic strategy to control the tolerance
                if( iter == 1 )
                  eigTolNow = 1e-2;
                else
                  eigTolNow = eigTolerance_;
              }
              else{
                // Static strategy to control the tolerance
                eigTolNow = eigTolerance_;
              }

              Int numEig = (eigSol.Psi().NumStateTotal())-numUnusedState_;
//              statusOFS << "The current tolerance used by the eigensolver is " 
//                << eigTolNow << std::endl;
//              statusOFS << "The target number of converged eigenvectors is " 
//                << numEig << std::endl << std::endl;
              //------------------------------------------------------------------------------------
              //Hybrid calculation in PWDFT to obtain hybrid-functional basis
              bool isFixColumnDF = false;
              if( ham.IsEXXActive() == false )
                 ham.SetEXXActive(true);
              // psi for EXX
              // Each psi includes a factor fac=sqrt(Volume/ntot) for DFT calculations
              // That is \int |psi(r)|^2 dr =  sum_g |psi(r_g)|^2 * dv = 1
              // dv = Volume/ntot , fac = \sqrt (dv)
              // For simply, psi(r_g) is multiplied by fac
              //
              // However, the Denisty \rho(r) and Potential V(r) do not include this fac
              // Because the density has been be scaled in hamiltonian.cpp
              // blas::Scal( ntotFine, (numSpin_ * Real(numOccupiedState_) * Real(ntotFine)) / ( vol * val ),
              //        density_.VecData(RHO), 1 );

              // Then,  H * psi_i = \epsilon psi_i aslo has a fac (not in H(r) but in psi(r) )
              //
              // For EXX calculation: phi is update in inner iteration step
              //
              // Vx(r,r')*phi_i(r') = \sum_j psi_j(r) *v(r,r')* psi_j(r') * phi_i(r')
              //
              // In order to make Vx * phi shares the same fac as H * psi
              // we have to use SetPhiEXX transforms psi(r) from psi(r_g) * fac to psi(r_g), that is remove the fac
              // for Vx(r, r')

              // Fock energies
              Real fock0 = 0.0, fock1 = 0.0, fock2 = 0.0;

              // EXX: Run SCF::Iterate here
              bool isPhiIterConverged = false;

              Real dExtElemExx;

              GetTime( timeSta );

              ham.SetPhiEXX( psi, fft );

#if ( _DEBUGlevel_ >= 2 )
              statusOFS << " psi.NumStateTotal " << psi.NumStateTotal() << std::endl;
              statusOFS << " ham.NumOccupiedState " << ham.NumOccupiedState() << std::endl;
              statusOFS << " ham.OccupationRate " << ham.OccupationRate() << std::endl;
#endif

              if( esdfParam.isHybridACE ){
               if( esdfParam.isHybridDF ){
                  ham.CalculateVexxACEDF( psi, fft, isFixColumnDF );
                  // Fix the column after the first iteraiton
                  isFixColumnDF = true;
               }
                else
                {
                  ham.CalculateVexxACE ( psi, fft );
                }
              }

              GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 1 )
              statusOFS << "Time for updating Phi related variable is " <<
              timeEnd - timeSta << " [s]" << std::endl;
#endif
              GetTime( timeSta );
              fock0 = ham.CalculateEXXEnergy( psi, fft );
              GetTime( timeEnd );

//              Print(statusOFS, "ExtElem Fock energy 1    = ",  fock0, "[au]");
#if ( _DEBUGlevel_ >= 1 )
              statusOFS << "Time for computing the EXX energy is " <<
                timeEnd - timeSta << " [s]" << std::endl;
#endif
//              Efock_ = fock2;
//              fock1  = fock2; 

              Real timeStaPhi, timeEndPhi;

              GetTime( timeStaPhi );

              for( Int phiIter = 1; phiIter <= PhiMaxIter_; phiIter++ ){
                std::ostringstream msg;
                msg << "Phi iteration # " << phiIter;
                PrintBlock( statusOFS, msg.str() );

                if(Diag_SCF_PWDFT_by_Cheby_ == 1)
                {
                // Use CheFSI or LOBPCG on first step 
                  if(iter <= 1)
                  {
                    if(First_SCF_PWDFT_ChebyCycleNum_ <= 0)
                    { 
//                      statusOFS << " >>>> Calling LOBPCG for ALB generation on extended element ..." << std::endl;
                      eigSol.LOBPCGSolveReal(numEig, iter, eigMaxIter_, eigMinTolerance_, eigTolNow );    
                    }
                    else
                    {
//                      statusOFS << " >>>> Calling CheFSI with random guess for ALB generation on extended element ..." << std::endl;
                      eigSol.FirstChebyStep(numEig, First_SCF_PWDFT_ChebyCycleNum_, First_SCF_PWDFT_ChebyFilterOrder_);
                    }
//                    statusOFS << std::endl;
                  }
                  else
                  {
//                    statusOFS << " >>>> Calling CheFSI with previous ALBs for generation of new ALBs ..." << std::endl;
//                    statusOFS << " >>>> Will carry out " << eigMaxIter_ << " CheFSI cycles." << std::endl;

                    for (int cheby_iter = 1; cheby_iter <= eigMaxIter_; cheby_iter ++)
                    {
//                      statusOFS << std::endl << " >>>> CheFSI for ALBs : Cycle " << cheby_iter << " of " << eigMaxIter_ << " ..." << std::endl;
                      eigSol.GeneralChebyStep(numEig, General_SCF_PWDFT_ChebyFilterOrder_);
                    }
//                    statusOFS << std::endl;
                  }
                }
                else if(Diag_SCF_PWDFT_by_PPCG_ == 1)
                {
                  // Use LOBPCG on very first step, i.e., while starting from random guess
                  if(iter <= 1)
                  {
//                    statusOFS << " >>>> Calling LOBPCG for ALB generation on extended element ..." << std::endl;
                    eigSol.LOBPCGSolveReal(numEig, iter, eigMaxIter_, eigMinTolerance_, eigTolNow );    
                  }
                  else
                  {
//                    statusOFS << " >>>> Calling PPCG with previous ALBs for generation of new ALBs ..." << std::endl;
                    eigSol.PPCGSolveReal(numEig, iter, eigMaxIter_, eigMinTolerance_, eigTolNow );
                  }
                }             
                else 
                {
                  Int eigDynMaxIter = eigMaxIter_;
                  eigSol.LOBPCGSolveReal(numEig, iter, eigDynMaxIter, eigMinTolerance_, eigTolNow );
                }

//                GetTime( timeEnd );
//                statusOFS << std::endl << "Eigensolver time = "     << timeEnd - timeSta
//                  << " [s]" << std::endl;

                GetTime( timeSta );
#if ( _DEBUGlevel_ >= 2 )
                statusOFS << " psi.NumStateTotal " << psi.NumStateTotal() << std::endl;
                statusOFS << " ham.NumOccupiedState " << ham.NumOccupiedState() << std::endl;
                statusOFS << " ham.OccupationRate " << ham.OccupationRate() << std::endl;
#endif

                CalculateOccupationRateExtElem( eigSol.EigVal(), ham.OccupationRate(),
                    psi.NumStateTotal(), ham.NumOccupiedState() );
                GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 1 )
                statusOFS << "Time for computing occupation rate in PWDFT is " <<
                  timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
 
                fock1 = ham.CalculateEXXEnergy( psi, fft );

                // Update Phi <- Psi
                GetTime( timeSta );
                ham.SetPhiEXX( psi, fft );
  
                if( esdfParam.isHybridACE ){
                  if( esdfParam.isHybridDF ){
                    ham.CalculateVexxACEDF( psi, fft, isFixColumnDF );
                    // Fix the column after the first iteraiton
                    isFixColumnDF = true;
                  }
                  else
                  {
                    ham.CalculateVexxACE ( psi, fft );
                  }
                }
  
                GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 1 )
                statusOFS << "Time for updating Phi related variable is " <<
                timeEnd - timeSta << " [s]" << std::endl;
#endif
  
                GetTime( timeSta );
                fock2 = ham.CalculateEXXEnergy( psi, fft );
                GetTime( timeEnd );
  
                // Note: initially fock1 = 0.0. So it should at least run for 1 iteration.
//                dExx = std::abs(fock2 - fock1) / std::abs(fock2);
                dExtElemExx = 4.0 * std::abs( fock1 - 0.5 * (fock0 + fock2) );

                fock0 = fock2;
                Efock_ = fock2;
            
//                statusOFS << std::endl;
//                Print(statusOFS, "ExtElem Fock energy       = ",  Efock_, "[au]");
//                Print(statusOFS, "dExx for PWDFT            = ",  dExx, "[au]");
                if( dExtElemExx < PhiTolerance_ ){
                  statusOFS << "ALBs: Hybrid functional SCF in extended element is converged in "
                    << phiIter << " steps !" << std::endl;
                  isPhiIterConverged = true;

                Print(statusOFS, "ALBs: ExtElem Fock energy       = ",  Efock_, "[au]");
                Print(statusOFS, "ALBs: dExtElemExx               = ",  dExtElemExx, "[au]");

                }
  
                if ( isPhiIterConverged ) break;

              } // for(phiIter)
 
              GetTime( timeEndPhi );
 
              // Print out the information
#if ( _DEBUGlevel_ >= 0 )
              statusOFS << "ALBs: Time for calculating extended Phi via PWDFT is " <<
                timeEndPhi - timeStaPhi << " [s]" << std::endl;
#endif

#if ( _DEBUGlevel_ >= 2 )
              Real maxRes = 0.0, avgRes = 0.0;
              for(Int ii = 0; ii < eigSol.EigVal().m(); ii++){
                if( maxRes < eigSol.ResVal()(ii) ){
                  maxRes = eigSol.ResVal()(ii);
                }
                avgRes = avgRes + eigSol.ResVal()(ii);
                Print(statusOFS, 
                  "basis#   = ", ii, 
                  "eigval   = ", eigSol.EigVal()(ii),
                  "resval   = ", eigSol.ResVal()(ii));
              }
              avgRes = avgRes / eigSol.EigVal().m();
              statusOFS << std::endl;
              Print(statusOFS, "Max residual of basis = ", maxRes );
              Print(statusOFS, "Avg residual of basis = ", avgRes );
              statusOFS << std::endl;
#endif

              GetTime( timeSta );
              DblNumMat& basis = hamDG.BasisLGL().LocalMap()[key];
              SVDLocalizeBasis ( iter, numGridExtElem,
                numGridElemFine, numLGLGrid, psi, basis );
//              SVDLocalizeBasis ( iter, numGridExtElemFine,
//                numGridElemFine, numLGLGrid, psi, basis );
              GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
              statusOFS << "ALBs: Time for ALBs from SVD and localizatoin is " <<
                timeEndPhi - timeStaPhi << " [s]" << std::endl;
#endif

            } // own this element
      } // for (i)

      GetTime( timeBasisEnd );

      statusOFS << std::endl << "OuterSCF: Time for generating ALB function is " <<
        timeBasisEnd - timeBasisSta << " [s]" << std::endl;

      MPI_Barrier( domain_.comm );

        // New DM  xmqin
      GetTime( timeSta );
      ProjectDM ( hamDG.BasisLGLSave(), hamDG.BasisLGL(), distDMMat_);
      GetTime (timeEnd);
#if ( _DEBUGlevel_ >= 0 )
      statusOFS << "OuterSCF: Time for projecting density metrix to currrent ALBs is " <<
        timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
//Check ! new deneity is required ?
//        GetTime( timeSta );
//        hamDG.CalculateDensityDM2( hamDG.Density(), hamDG.DensityLGL(), distDMMat_ );
//        GetTime( timeEnd );
//
//#if ( _DEBUGlevel_ >= 1 )
//        statusOFS << " Time for computing density in the global domain is " <<
//          timeEnd - timeSta << " [s]" << std::endl << std::endl;
//#endif

      if(esdfParam.isDGHFISDF){
        GetTime( timeSta );
        hamDG.DGHFX_ISDF( );
        GetTime (timeEnd);
#if ( _DEBUGlevel_ >= 0 )
        statusOFS << "OuterSCF: Time for performing ISDF in element is " <<
          timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
      }
      else{
        GetTime( timeSta );
        hamDG.CollectNeighborBasis( );
        GetTime (timeEnd);
#if ( _DEBUGlevel_ >= 0 )
        statusOFS << "OuterSCF: Time for collecting ALBs from neighboring element is " <<
          timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
      }

      // Routine for re-orienting eigenvectors based on current basis set
      if(Diag_SCFDG_by_Cheby_ == 1)
      {
//        Real timeSta, timeEnd;
        Real extra_timeSta, extra_timeEnd;

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

                statusOFS << std::endl << " AlltoallForward: Changing distribution of local basis functions ... ";
                GetTime(extra_timeSta);
                AlltoallForward(band_distrib_local_basis, row_distrib_local_basis, domain_.rowComm);
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

                  double *ptr_0 = ALB_LGL_deque_[0].Data();
                  double *ptr_1 = ALB_LGL_deque_[1].Data();

                  blas::Gemm( 'T', 'N', width, width, heightLGLLocal,
                      1.0, ptr_1, heightLGLLocal, 
                      ptr_0, heightLGLLocal, 
                      0.0, Overlap_Mat_Temp.Data(), width );

                  // Reduce along rowComm (i.e., along the intra-element direction)
                  // to compute the actual overlap matrix
                  MPI_Allreduce( Overlap_Mat_Temp.Data(), 
                      Overlap_Mat.Data(), 
                      width * width, 
                      MPI_DOUBLE, 
                      MPI_SUM, 
                      domain_.rowComm );

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

                  blas::Gemm( 'N', 'N', Overlap_Mat.m(), eigvecs_local.n(), Overlap_Mat.n(), 
                      1.0, Overlap_Mat.Data(), Overlap_Mat.m(), 
                      temp_loc_eigvecs_buffer.Data(), temp_loc_eigvecs_buffer.m(), 
                      0.0, eigvecs_local.Data(), eigvecs_local.m());

                  GetTime(extra_timeEnd);
                  statusOFS << " Done. ( " << (extra_timeEnd - extra_timeSta) << " s)" << std::endl;

                  ALB_LGL_deque_.pop_front();
                }        
              } // End of if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) )
          } // End of loop over key indices i.e., for( Int i = 0; i < numElem_[0]; i++ )
  
          if( iter > 1)
          {
            GetTime(timeEnd);
            statusOFS << std::endl << " All steps of basis rotation completed. ( " << (timeEnd - timeSta) << " s )"<< std::endl;
          }
      } // End of if(Diag_SCFDG_by_Cheby_ == 1)
  
      // *********************************************************************
      // Inner SCF iteration 
      //
      // Assemble and diagonalize the DG matrix until convergence is
      // reached for updating the basis functions in the next step.
      // *********************************************************************
  
      GetTime(timeSta);
  
      // Save the mixing variable in the outer SCF iteration 
      if( OutermixVariable_ == "density" || OutermixVariable_ == "potential" ){
        for( Int k = 0; k < numElem_[2]; k++ )
          for( Int j = 0; j < numElem_[1]; j++ )
            for( Int i = 0; i < numElem_[0]; i++ ){
              Index3 key( i, j, k );
              if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
                if( OutermixVariable_ == "density" ){
                  DblNumVec& oldVec = hamDG.Density().LocalMap()[key];
                  mixOuterSave_.LocalMap()[key] = oldVec;
                }
                else if( OutermixVariable_ == "potential" ){
                  DblNumVec& oldVec = hamDG.Vtot().LocalMap()[key];
                  mixOuterSave_.LocalMap()[key] = oldVec;
                }
              } // own this element
            } // for (i)
      }
      // Main function here
      InnerIterate( iter );
  
      MPI_Barrier( domain_.comm );
      GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
      statusOFS << "OuterSCF: Time for all inner SCF iterations is " <<
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
                if( OutermixVariable_ == "density" ){
                  DblNumVec& oldVec = mixOuterSave_.LocalMap()[key];
                  DblNumVec& newVec = hamDG.Density().LocalMap()[key];

                  for( Int p = 0; p < oldVec.m(); p++ ){
                    normMixDifLocal += pow( oldVec(p) - newVec(p), 2.0 );
                    normMixOldLocal += pow( oldVec(p), 2.0 );
                  }
                }
                else if( OutermixVariable_ == "potential" ){
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
      //    PrintState( );

    // Check for convergence
      if(useEnergySCFconvergence_ == 0)
      {  
        if( (iter > 1) && 
            ( (scfOuterNorm_ < HybridscfOuterTolerance_) && 
              (efreeDifPerAtom_ < HybridscfOuterEnergyTolerance_) ) ){
        /* converged */
          statusOFS << " Outer SCF is converged in " << iter << " steps !" << std::endl;
          isSCFConverged = true;
        }
      }
      else
      {
        if( (iter > 1 ) && 
          (md_scf_etot_diff_ < md_scf_etot_diff_tol_) &&
          (md_scf_eband_diff_ < md_scf_eband_diff_tol_) )
        {
        // converged via energy criterion
          statusOFS << " Outer SCF is converged via energy condition in " << iter << " steps !" << std::endl;
          isSCFConverged = true;

        }
      } // if(useEnergySCFconvergence_ == 0)

      // Potential mixing for the outer SCF iteration. or no mixing at all anymore?
      // It seems that no mixing is the best.

      GetTime( timeIterEnd );
      statusOFS << "OuterSCF: Time for this outer SCF iteration = " << timeIterEnd - timeIterStart
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
        HybridscfOuterMaxIter_ << std::endl;
    } // if(scfdg_ion_dyn_iter_ >= 1)

  } //       if( hamDG.IsHybrid()  )


  //    if(0)
  //    {
  //      // Output the electron density on the LGL grid in each element
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
      statusOFS << std::endl << " Computing forces using eigenvectors ..." << std::endl;
      hamDG.CalculateForce( fftDG );
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
      hamDG.CalculateForceDM( fftDG, distDMMat_ );        
    }
  }
  else if( solutionMethod_ == "pexsi" ){
    hamDG.CalculateForceDM( fftDG, distDMMat_ );
  }
  GetTime( timeForceEnd );
  statusOFS << "Time for computing the force is " <<
    timeForceEnd - timeForceSta << " [s]" << std::endl << std::endl;

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
    Real HOMO, LUMO, EG;

    if(solutionMethod_ != "pexsi")
    {
      HOMO = hamDG.EigVal()( hamDG.NumOccupiedState()-1 );
      if( hamDG.NumExtraState() > 0 ) {
        LUMO = hamDG.EigVal()( hamDG.NumOccupiedState());
        EG = LUMO - HOMO;
      }
    }

#if ( _DEBUGlevel_ >= 1 )
  if(SCFDG_comp_subspace_engaged_ == false && (solutionMethod_ != "pexsi") )
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

   
    // Print out the energy
    PrintBlock( statusOFS, "Energy" );
    //statusOFS 
    //  << "NOTE:  Ecor  = Exc + Exx - EVxc - Ehart - Eself + Evdw" << std::endl
    //  << "       Etot  = Ekin + Ecor" << std::endl
    //  << "       Efree = Etot + Entropy" << std::endl << std::endl;
    Print(statusOFS, "! EfreeHarris     = ",  EfreeHarris_, "[au]");
    Print(statusOFS, "! Etot            = ",  Etot_, "[au]");
    Print(statusOFS, "! Exc             = ",  Exc_, "[au]");
    Print(statusOFS, "! Exx             = ",  Ehfx_, "[au]");
    Print(statusOFS, "! Efree           = ",  Efree_, "[au]");
    Print(statusOFS, "! Evdw            = ",  Evdw_, "[au]"); 
    Print(statusOFS, "! Fermi           = ",  fermi_, "[au]");

    if(solutionMethod_ != "pexsi")
    {
      Print(statusOFS, "! HOMO            = ",  HOMO*au2ev, "[eV]");

      if( hamDG.NumExtraState() > 0 ){
        Print(statusOFS, "! LUMO            = ",  LUMO*au2ev, "[eV]");
        Print(statusOFS, "! Band Gap        = ",  EG*au2ev, "[eV]");
      }
    }

    statusOFS << std::endl << "  Convergence information : " << std::endl;
    Print(statusOFS, "! norm(out-in)/norm(in) = ",  scfOuterNorm_ ); 
    Print(statusOFS, "! Efree diff per atom   = ",  efreeDifPerAtom_, "[au]"); 

    if(useEnergySCFconvergence_ == 1)
    {
      Print(statusOFS, "! MD SCF Etot diff (per atom)  = ",  md_scf_etot_diff_, "[au]"); 
      Print(statusOFS, "! MD SCF Eband diff (per atom) = ",  md_scf_eband_diff_, "[au]"); 
    }
  }

  // Print out the force
  PrintBlock( statusOFS, "Atomic Force" );

  Point3 forceCM(0.0, 0.0, 0.0);
  std::vector<Atom>& atomList = hamDG.AtomList();
  Int numAtom = atomList.size();

  for( Int a = 0; a < numAtom; a++ ){

//    Print( statusOFS, "atom", a, "force", atomList[a].force );

    forceCM += atomList[a].force;
  }
  statusOFS << std::endl;
  Print( statusOFS, "force for centroid  : ", forceCM );
  Print( statusOFS, "Max force magnitude : ", MaxForce(atomList) );
  statusOFS << std::endl;

  // *********************************************************************
  // Output information
  // *********************************************************************

  // Output the atomic structure, and other information for describing
  // density, basis functions etc.
  // 
  // Only mpirank == 0 works on this

  Real timeOutputSta, timeOutputEnd;
  GetTime( timeOutputSta );

  if( mpirank == 0 ){
    std::ostringstream structStream;
    statusOFS << std::endl 
      << "Output the structure information" 
      << std::endl;
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

  for( Int k = 0; k < numElem_[2]; k++ )
    for( Int j = 0; j < numElem_[1]; j++ )
      for( Int i = 0; i < numElem_[0]; i++ ){
        Index3 key( i, j, k );
        if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
          if( esdfParam.isOutputDensity ){
            if( mpirankRow == 0 ){
              statusOFS << std::endl 
                << "Output the electron density on the global grid" 
                << std::endl;
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
              statusOFS 
                << std::endl 
                << "Output the total potential and external potential in the extended element."
                << std::endl;
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
            statusOFS 
              << std::endl 
              << "Output the wavefunctions in the extended element."
              << std::endl;

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
            statusOFS 
              << std::endl 
              << "Output the wavefunctions in the element on a LGL grid."
              << std::endl;
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
            statusOFS 
              << std::endl 
              << "Output the wavefunctions in the element on a fine LGL grid."
              << std::endl;
            // Output the wavefunctions in the extended element.
            std::ostringstream wavefunStream;      

            // Generate the uniform mesh on the extended element.
            serialize( key, wavefunStream, NO_MASK );
            DblNumMat& basisLGL = hamDG.BasisLGL().LocalMap()[key];
            DblNumMat basisUniformFine( 
                hamDG.NumUniformGridElemFine().prod(), 
                basisLGL.n() );
            SetValue( basisUniformFine, 0.0 );

            DblNumMat basisUniform(
                hamDG.NumUniformGridElem().prod(),
                basisLGL.n() );
            SetValue( basisUniform, 0.0 );

            for( Int g = 0; g < basisLGL.n(); g++ ){
              hamDG.InterpLGLToUniform(
                  hamDG.NumLGLGridElem(),
                  hamDG.NumUniformGridElemFine(),
                  basisLGL.VecData(g),
                  basisUniformFine.VecData(g) );

              hamDG.InterpLGLToUniform2(
                  hamDG.NumLGLGridElem(),
                  hamDG.NumUniformGridElem(),
                  basisLGL.VecData(g),
                  basisUniform.VecData(g) );
            }

            DblNumTns&   LGLWeight3D = hamDG.LGLWeight3D();
            DblNumMat basisTemp (basisLGL.m(), basisLGL.n());
            SetValue( basisTemp, 0.0 );
            Real factor = domain_.Volume() / domain_.NumGridTotalFine();
            Real factor2 = domain_.Volume() / domain_.NumGridTotal();

            // This is the same as the FourDotProduct process.
            for( Int g = 0; g < basisLGL.n(); g++ ){
              Real *ptr1 = LGLWeight3D.Data();
              Real *ptr2 = basisLGL.VecData(g);
              Real *ptr3 = basisTemp.VecData(g);
              for( Int l = 0; l < basisLGL.m(); l++ ){
                *(ptr3++) = (*(ptr1++)) * (*(ptr2++)) ;
              }
            }

            DblNumMat Smat(basisLGL.n(), basisLGL.n() );
            SetValue( Smat, 0.0 );
            DblNumMat Smat2(basisLGL.n(), basisLGL.n() );
            SetValue( Smat2, 0.0 );
            DblNumMat Smat3(basisLGL.n(), basisLGL.n() );
            SetValue( Smat3, 0.0 );

            blas::Gemm( 'T', 'N',basisLGL.n() , basisLGL.n(), basisLGL.m(),
                    1.0, basisLGL.Data(), basisLGL.m(),
                    basisTemp.Data(), basisLGL.m(), 0.0,
                    Smat.Data(), basisLGL.n() );

            blas::Gemm( 'T', 'N',basisUniformFine.n() , basisUniformFine.n(), basisUniformFine.m(),
                    factor, basisUniformFine.Data(), basisUniformFine.m(),
                    basisUniformFine.Data(), basisUniformFine.m(), 0.0,
                    Smat2.Data(), basisUniformFine.n() );

            blas::Gemm( 'T', 'N',basisUniform.n() , basisUniform.n(), basisUniform.m(),
                    factor2, basisUniform.Data(), basisUniform.m(),
                    basisUniform.Data(), basisUniform.m(), 0.0,
                    Smat3.Data(), basisUniform.n() );

            for( Int p = 0; p < basisLGL.n(); p++ ){
                statusOFS << " p " << p << " q " << p << std::endl;
                statusOFS << " SmatLGL " << Smat(p,p) << " SmatUniformFine " <<  Smat2(p,p) << " SmatUniform " <<  Smat3(p,p)  << std::endl;
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
              statusOFS << std::endl 
                << "Output the eigenvector coefficients after diagonalization."
                << std::endl;
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
  statusOFS << std::endl 
    << "Time for outputing data is = " << timeOutputEnd - timeOutputSta
    << " [s]" << std::endl;

  return;
}         // -----  end of method SCFDG::Iterate  -----

}
