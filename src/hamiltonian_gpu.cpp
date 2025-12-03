/*
   Copyright (c) 2012 The Regents of the University of California,
   through Lawrence Berkeley National Laboratory.  

Author: Lin Lin, Wei Hu, Weile Jia

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
/// @file hamiltonian.cpp
/// @brief Hamiltonian class for planewave basis diagonalization method.
/// @date 2012-09-16
#include  "hamiltonian.hpp"
#include  "blas.hpp"
#include  "lapack.hpp"

namespace dgdft{

using namespace dgdft::PseudoComponent;
using namespace dgdft::DensityComponent;
using namespace dgdft::esdf;

void
KohnSham::CalculateDensity_GPU ( const Spinor &psi, const DblNumVec &occrate, Real &val, Fourier &fft )
{

  Int ntot  = psi.NumGridTotal();
  Int ncom  = psi.NumComponent();
  Int nocc  = psi.NumState();

  Real vol  = domain_.Volume();
//  statusOFS << ntot << " " << ncom << " " << nocc << " " << vol << std::endl << std::flush;
  
  Int ntotFine  = fft.domain.NumGridTotalFine();

  MPI_Barrier(domain_.comm);
  int mpirank;  MPI_Comm_rank(domain_.comm, &mpirank);
  int mpisize;  MPI_Comm_size(domain_.comm, &mpisize);

  DblNumMat   densityLocal;
  densityLocal.Resize( ntotFine, ncom );   
  //SetValue( densityLocal, 0.0 );

  Real fac;
  //cuda_free(dev_idxFineGridR2C);
  int * dev_idxFineGrid;
  //dev_idxFineGrid = NULL;
  //SetValue( density_, 0.0 );

  /* psi wavefunc.Data is the GPU wavefunction */
  CpxNumVec psi_temp(ntot);
  cuCpxNumVec cu_psi(ntot);
  cuCpxNumVec cu_psi_out(ntot);
  cuCpxNumVec cu_psi_fine_out(ntotFine);
  cuCpxNumVec cu_psi_fine(ntotFine);
  cuDblNumVec cu_density(ntotFine);
  cuDblNumVec cu_den(ntotFine);

  cuda_setValue( cu_density.Data(), 0.0, ntotFine);
  cuDoubleComplex zero; zero.x = 0.0; zero.y = 0.0;
  //very important----
  dev_idxFineGrid = ( int*) cuda_malloc ( sizeof(int   ) * ntot);
  cuda_memcpy_CPU2GPU(dev_idxFineGrid, fft.idxFineGrid.Data(), sizeof(Int) *ntot);

#ifdef _PROFILING_
  Real timeSta1, timeEnd1;
  MPI_Barrier(MPI_COMM_WORLD);
  cuda_sync();
  GetTime( timeSta1 );
#endif
  for (Int k=0; k<nocc; k++) {
    for (Int j=0; j<ncom; j++) {
      SetValue( psi_temp, Z_ZERO );
      for(Int i=0; i < ntot; i++){
	psi_temp(i) = Complex( psi.Wavefun(i,j,k), 0.0 );
      }
      cuda_memcpy_CPU2GPU(cu_psi.Data(), psi_temp.Data(), sizeof(cuDoubleComplex)*ntot);
      cuFFTExecuteForward2( fft, fft.cuPlanC2C[0], 0, cu_psi, cu_psi_out );
      cuDoubleComplex zero_hipcpx(0.0,0.0);
      cuda_setValue(reinterpret_cast<cuDoubleComplex*>(cu_psi_fine_out.Data()), zero_hipcpx , ntotFine);
      //cuda_setValue(cu_psi_fine_out.Data(), (cuDoubleComplex)zero , ntotFine);
      
      Real fac = sqrt( double(ntot) / double(ntotFine) );
      cuda_interpolate_wf_C2F( reinterpret_cast<cuDoubleComplex*>(cu_psi_out.Data()), 
                               reinterpret_cast<cuDoubleComplex*>(cu_psi_fine_out.Data()), 
                               dev_idxFineGrid,
                               ntot, 
                               fac);
      cuFFTExecuteInverse(fft, fft.cuPlanC2CFine[0], 1, cu_psi_fine_out, cu_psi_fine);
      fac = numSpin_ * occrate(psi.WavefunIdx(k));
      cuda_XTX( cu_psi_fine.Data(), cu_den.Data(), ntotFine);
      cublas::Axpy( ntotFine, &fac, cu_den.Data(), 1, cu_density.Data(), 1);
    }
  }
  cuda_free(dev_idxFineGrid);
#ifdef _PROFILING_
  MPI_Barrier(MPI_COMM_WORLD);
  cuda_sync();
  GetTime( timeEnd1 );
  statusOFS << " Evaluate Density time " << timeEnd1 - timeSta1 << " [s] " << std::endl;
  Real a1 = mpi::allreduceTime;
#endif

  #ifdef GPUDIRECT
  mpi::Allreduce( cu_density.Data(), cu_den.Data(), ntotFine, MPI_SUM, domain_.comm );
  #else
  cuda_memcpy_GPU2CPU( densityLocal.Data(), cu_density.Data(), ntotFine *sizeof(double));
  mpi::Allreduce( densityLocal.Data(), density_.Data(), ntotFine, MPI_SUM, domain_.comm );
  cuda_memcpy_CPU2GPU( cu_den.Data(), density_.Data(), ntotFine *sizeof(double));
  #endif 

#ifdef _PROFILING_
  statusOFS << " Evaluate Density reduce " << mpi::allreduceTime - a1 << " [s] " << std::endl;
#endif

  #ifdef GPU
  double * val_dev = (double*) cuda_malloc( sizeof(double));
  val = 0.0; // sum of density
  cuda_reduce( cu_den.Data(), val_dev, 1, ntotFine);
  cuda_memcpy_GPU2CPU( &val, val_dev, sizeof(double));
  Real val1 = val;
  Real temp = (numSpin_ * Real(numOccupiedState_) * Real(ntotFine)) / ( vol * val );
  cublas::Scal( ntotFine, &temp, cu_den.Data(), 1 );
  cuda_memcpy_GPU2CPU( density_.Data(), cu_den.Data(), ntotFine *sizeof(double));

  //cuda_memcpy_GPU2GPU( cu_density.Data(), cu_den.Data(), ntotFine*sizeof(double) );
  cuda_set_vector( cu_density.Data(), cu_den.Data(), ntotFine);
  temp = vol / ntotFine;
  cublas::Scal( ntotFine, &temp, cu_density.Data(), 1 );

  cuda_reduce( cu_density.Data(), val_dev, 1, ntotFine);
  cuda_memcpy_GPU2CPU( &val, val_dev, sizeof(double));
  Real val2 = val;
  
  cuda_free(val_dev);
  #else

  val = 0.0; // sum of density
  for (Int i=0; i<ntotFine; i++) {
    val  += density_(i, RHO);
  }

  Real val1 = val;

  // Scale the density
  blas::Scal( ntotFine, (numSpin_ * Real(numOccupiedState_) * Real(ntotFine)) / ( vol * val ), 
      density_.VecData(RHO), 1 );

  // Double check (can be neglected)
  val = 0.0; // sum of density
  for (Int i=0; i<ntotFine; i++) {
    val  += density_(i, RHO) * vol / ntotFine;
  }

  Real val2 = val;
  #endif

#if ( _DEBUGlevel_ >= 0 )
  statusOFS << "Raw data, sum of density          = " << val1 << std::endl;
  statusOFS << "Expected sum of density           = " << numSpin_ * numOccupiedState_ << std::endl;
  statusOFS << "Raw data, sum of adjusted density = " << val2 << std::endl;
#endif

  return ;
}         // -----  end of method KohnSham::CalculateDensity GPU ----- 

void
KohnSham::ACEOperator_GPU ( cuDblNumMat& cu_psi, Fourier& fft, cuDblNumMat& cu_Hpsi)
{
  if( isHybrid_ && isEXXActive_ ){
    if( esdfParam.isHybridACE ){ 
    int mpirank;  MPI_Comm_rank(domain_.comm, &mpirank);
    int mpisize;  MPI_Comm_size(domain_.comm, &mpisize);

     Int ntot      = fft.domain.NumGridTotal();
     Int ntotFine  = fft.domain.NumGridTotalFine();
     Int numStateTotal = cu_psi.n();

     Int ntotBlocksize = ntot / mpisize;
     Int ntotLocal = ntotBlocksize;
     if(mpirank < (ntot % mpisize)){
       ntotLocal = ntotBlocksize + 1;
     }

     Real one = 1.0;
     Real minus_one = -1.0;
     Real zero = 0.0;

     DblNumMat MTemp( numStateTotal, numStateTotal );
     cuDblNumMat cu_MTemp( numStateTotal, numStateTotal );

     cublas::Gemm( HIPBLAS_OP_T, HIPBLAS_OP_N, numStateTotal, numStateTotal, ntotLocal, &one, cu_vexxProj_.Data(), ntotLocal, 
                   cu_psi.Data(), ntotLocal, &zero,
                   cu_MTemp.Data(), numStateTotal );
     cuda_memcpy_GPU2CPU( MTemp.Data(), cu_MTemp.Data(), numStateTotal*numStateTotal*sizeof(Real) );

     DblNumMat M(numStateTotal, numStateTotal);
     MPI_Allreduce( MTemp.Data(), M.Data(), numStateTotal * numStateTotal, MPI_DOUBLE, MPI_SUM, domain_.comm );
     cuda_memcpy_CPU2GPU(  cu_MTemp.Data(), M.Data(), numStateTotal*numStateTotal*sizeof(Real) );
     cublas::Gemm( HIPBLAS_OP_N, HIPBLAS_OP_N, ntotLocal, numStateTotal, numStateTotal, 
                   &minus_one, cu_vexxProj_.Data(), ntotLocal, 
                   cu_MTemp.Data(), numStateTotal, &one, 
                   cu_Hpsi.Data(), ntotLocal );
    }
  }
}

void
KohnSham::MultSpinor_old    ( Spinor& psi, cuNumTns<Real>& a3, Fourier& fft )
{

  int mpirank;  MPI_Comm_rank(domain_.comm, &mpirank);
  int mpisize;  MPI_Comm_size(domain_.comm, &mpisize);

  Int ntot      = fft.domain.NumGridTotal();
  Int ntotFine  = fft.domain.NumGridTotalFine();
  Int numStateTotal = psi.NumStateTotal();
  Int numStateLocal = psi.NumState();
  NumTns<Real>& wavefun = psi.Wavefun();
  Int ncom = wavefun.n();

  Real timeSta, timeEnd;
  Real timeSta1, timeEnd1;
  
  //SetValue( a3, 0.0 );
  GetTime( timeSta );
  psi.AddMultSpinorFineR2C_GPU( fft, vtot_, pseudo_, a3 );
  GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 1 )
  statusOFS << "Time for psi.AddMultSpinorFineR2C is " <<
    timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

  // adding up the Hybrid part in the GPU
  // CHECK CHECK
  // Note now, the psi.data is the GPU data. and a3.data is also in GPU. 
  // also, a3 constains the Hpsi
  // need to do this in another subroutine.
  if(1)  
  if( isHybrid_ && isEXXActive_ ){

    GetTime( timeSta );

    if( esdfParam.isHybridACE ){ 

      if(1){ // for MPI
        // Convert the column partition to row partition

        Int numStateBlocksize = numStateTotal / mpisize;
        Int ntotBlocksize = ntot / mpisize;

        Int numStateLocal = numStateBlocksize;
        Int ntotLocal = ntotBlocksize;

        if(mpirank < (numStateTotal % mpisize)){
          numStateLocal = numStateBlocksize + 1;
        }

        if(mpirank < (ntot % mpisize)){
          ntotLocal = ntotBlocksize + 1;
        }

        // copy the GPU data to CPU.
        DblNumMat psiCol( ntot, numStateLocal );
        cuda_memcpy_GPU2CPU( psiCol.Data(), psi.cuWavefun().Data(), ntot*numStateLocal*sizeof(Real) );

        // for the Project VexxProj 
        DblNumMat vexxProjCol( ntot, numStateLocal );
        DblNumMat vexxProjRow( ntotLocal, numStateTotal );
        lapack::Lacpy( 'A', ntot, numStateLocal, vexxProj_.Data(), ntot, vexxProjCol.Data(), ntot );

        // MPI_Alltoall for the data redistribution.
        DblNumMat psiRow( ntotLocal, numStateTotal );
        AlltoallForward (psiCol, psiRow, domain_.comm);

        // MPI_Alltoall for data redistribution.
        AlltoallForward (vexxProjCol, vexxProjRow, domain_.comm);

        // GPU data for the G-para
        cuDblNumMat cu_vexxProjRow ( ntotLocal, numStateTotal );
        cuDblNumMat cu_psiRow ( ntotLocal, numStateTotal );
        cuDblNumMat cu_MTemp( numStateTotal, numStateTotal );
        DblNumMat MTemp( numStateTotal, numStateTotal );

        // Copy data from CPU to GPU.
        cuda_memcpy_CPU2GPU( cu_psiRow.Data(), psiRow.Data(), numStateTotal*ntotLocal*sizeof(Real) );
        cuda_memcpy_CPU2GPU( cu_vexxProjRow.Data(), vexxProjRow.Data(), numStateTotal*ntotLocal*sizeof(Real) );

	Real one = 1.0;
	Real minus_one = -1.0;
	Real zero = 0.0;
        // GPU DGEMM calculation
        cublas::Gemm( HIPBLAS_OP_T, HIPBLAS_OP_N, numStateTotal, numStateTotal, ntotLocal,
                    &one, cu_vexxProjRow.Data(), ntotLocal, 
                    cu_psiRow.Data(), ntotLocal, &zero,
                    cu_MTemp.Data(), numStateTotal );

        cuda_memcpy_GPU2CPU( MTemp.Data(), cu_MTemp.Data(), numStateTotal*numStateTotal*sizeof(Real) );
        DblNumMat M(numStateTotal, numStateTotal);
        MPI_Allreduce( MTemp.Data(), M.Data(), numStateTotal * numStateTotal, MPI_DOUBLE, MPI_SUM, domain_.comm );

	// copy from CPU to GPU
        cuda_memcpy_CPU2GPU(  cu_MTemp.Data(), M.Data(), numStateTotal*numStateTotal*sizeof(Real) );
        
        cuDblNumMat cu_a3Row( ntotLocal, numStateTotal );
        DblNumMat a3Row( ntotLocal, numStateTotal );

        cublas::Gemm( HIPBLAS_OP_N, HIPBLAS_OP_N, ntotLocal, numStateTotal, numStateTotal, 
                     &minus_one, cu_vexxProjRow.Data(), ntotLocal, 
                     cu_MTemp.Data(), numStateTotal, &zero, 
                     cu_a3Row.Data(), ntotLocal );

        cuda_memcpy_GPU2CPU( a3Row.Data(), cu_a3Row.Data(), numStateTotal*ntotLocal*sizeof(Real) );

        // a3Row to a3Col
        DblNumMat a3Col( ntot, numStateLocal );
        cuDblNumMat cu_a3Col( ntot, numStateLocal );
        AlltoallBackward (a3Row, a3Col, domain_.comm);

	//Copy a3Col to GPU.
        cuda_memcpy_CPU2GPU( cu_a3Col.Data(), a3Col.Data(), numStateLocal*ntot*sizeof(Real) );

        // do the matrix addition.
	cuda_DMatrix_Add( a3.Data(), cu_a3Col.Data(), ntot, numStateLocal);

      } //if(1)

    }
    else{

      ErrorHandling(" GPU does not support normal HSE, try ACE");
      
    }

    GetTime( timeEnd );
//#if ( _DEBUGlevel_ >= 0 )
//    statusOFS << "Time for updating hybrid Spinor is " <<
//      timeEnd - timeSta << " [s]" << std::endl << std::endl;
//    statusOFS << "Time for Gemm is " <<
//      timeGemm << " [s]" << std::endl << std::endl;
//    statusOFS << "Time for Alltoallv is " <<
//      timeAlltoallv << " [s]" << std::endl << std::endl;
//    statusOFS << "Time for Allreduce is " <<
//      timeAllreduce << " [s]" << std::endl << std::endl;
//#endif


  }


  return ;
}         // -----  end of method KohnSham::MultSpinor  ----- 

void
KohnSham::MultSpinor_GPU  ( Spinor& psi, cuNumTns<Real>& a3, Fourier& fft )
{

  int mpirank;  MPI_Comm_rank(domain_.comm, &mpirank);
  int mpisize;  MPI_Comm_size(domain_.comm, &mpisize);

  Int ntot      = fft.domain.NumGridTotal();
  Int ntotFine  = fft.domain.NumGridTotalFine();
  Int numStateTotal = psi.NumStateTotal();
  Int numStateLocal = psi.NumState();
  NumTns<Real>& wavefun = psi.Wavefun();
  Int ncom = wavefun.n();

  Real timeSta, timeEnd;
  Real timeSta1, timeEnd1;
  
  //SetValue( a3, 0.0 );
  GetTime( timeSta );
  psi.AddMultSpinorFineR2C_GPU( fft, vtot_, pseudo_, a3 );
  GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 1 )
  statusOFS << "Time for psi.AddMultSpinorFineR2C is " <<
    timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

  // adding up the Hybrid part in the GPU
  // CHECK CHECK
  // Note now, the psi.data is the GPU data. and a3.data is also in GPU. 
  // also, a3 constains the Hpsi
  // need to do this in another subroutine.
//  if(0)  // comment out the following parts.
  if( isHybrid_ && isEXXActive_ ){

    GetTime( timeSta );

    if( esdfParam.isHybridACE ){ 

      if(1){ // for MPI
        // Convert the column partition to row partition

        Int numStateBlocksize = numStateTotal / mpisize;
        Int ntotBlocksize = ntot / mpisize;

        Int numStateLocal = numStateBlocksize;
        Int ntotLocal = ntotBlocksize;

        if(mpirank < (numStateTotal % mpisize)){
          numStateLocal = numStateBlocksize + 1;
        }

        if(mpirank < (ntot % mpisize)){
          ntotLocal = ntotBlocksize + 1;
        }

        // copy the GPU data to CPU.
        DblNumMat psiCol( ntot, numStateLocal );
        cuda_memcpy_GPU2CPU( psiCol.Data(), psi.cuWavefun().Data(), ntot*numStateLocal*sizeof(Real) );

        // for the Project VexxProj 
        DblNumMat vexxProjCol( ntot, numStateLocal );
        DblNumMat vexxProjRow( ntotLocal, numStateTotal );
        lapack::Lacpy( 'A', ntot, numStateLocal, vexxProj_.Data(), ntot, vexxProjCol.Data(), ntot );

        // MPI_Alltoall for the data redistribution.
        DblNumMat psiRow( ntotLocal, numStateTotal );
        AlltoallForward (psiCol, psiRow, domain_.comm);

        // MPI_Alltoall for data redistribution.
        AlltoallForward (vexxProjCol, vexxProjRow, domain_.comm);

        // GPU data for the G-para
        cuDblNumMat cu_vexxProjRow ( ntotLocal, numStateTotal );
        cuDblNumMat cu_psiRow ( ntotLocal, numStateTotal );
        cuDblNumMat cu_MTemp( numStateTotal, numStateTotal );
        DblNumMat MTemp( numStateTotal, numStateTotal );

        // Copy data from CPU to GPU.
        cuda_memcpy_CPU2GPU( cu_psiRow.Data(), psiRow.Data(), numStateTotal*ntotLocal*sizeof(Real) );
        cuda_memcpy_CPU2GPU( cu_vexxProjRow.Data(), vexxProjRow.Data(), numStateTotal*ntotLocal*sizeof(Real) );

	Real one = 1.0;
	Real minus_one = -1.0;
	Real zero = 0.0;
        // GPU DGEMM calculation
        cublas::Gemm( HIPBLAS_OP_T, HIPBLAS_OP_N, numStateTotal, numStateTotal, ntotLocal,
                    &one, cu_vexxProjRow.Data(), ntotLocal, 
                    cu_psiRow.Data(), ntotLocal, &zero,
                    cu_MTemp.Data(), numStateTotal );

        cuda_memcpy_GPU2CPU( MTemp.Data(), cu_MTemp.Data(), numStateTotal*numStateTotal*sizeof(Real) );
        DblNumMat M(numStateTotal, numStateTotal);
        MPI_Allreduce( MTemp.Data(), M.Data(), numStateTotal * numStateTotal, MPI_DOUBLE, MPI_SUM, domain_.comm );

	// copy from CPU to GPU
        cuda_memcpy_CPU2GPU(  cu_MTemp.Data(), M.Data(), numStateTotal*numStateTotal*sizeof(Real) );
        
        cuDblNumMat cu_a3Row( ntotLocal, numStateTotal );
        DblNumMat a3Row( ntotLocal, numStateTotal );

        cublas::Gemm( HIPBLAS_OP_N, HIPBLAS_OP_N, ntotLocal, numStateTotal, numStateTotal, 
                     &minus_one, cu_vexxProjRow.Data(), ntotLocal, 
                     cu_MTemp.Data(), numStateTotal, &zero, 
                     cu_a3Row.Data(), ntotLocal );

        cuda_memcpy_GPU2CPU( a3Row.Data(), cu_a3Row.Data(), numStateTotal*ntotLocal*sizeof(Real) );

        // a3Row to a3Col
        DblNumMat a3Col( ntot, numStateLocal );
        cuDblNumMat cu_a3Col( ntot, numStateLocal );
        AlltoallBackward (a3Row, a3Col, domain_.comm);

	//Copy a3Col to GPU.
        cuda_memcpy_CPU2GPU( cu_a3Col.Data(), a3Col.Data(), numStateLocal*ntot*sizeof(Real) );

        // do the matrix addition.
	cuda_DMatrix_Add( a3.Data(), cu_a3Col.Data(), ntot, numStateLocal);

      } //if(1)

    }
    else{

      ErrorHandling(" GPU does not support normal HSE, try ACE");
      
    }

    GetTime( timeEnd );
//#if ( _DEBUGlevel_ >= 0 )
//    statusOFS << "Time for updating hybrid Spinor is " <<
//      timeEnd - timeSta << " [s]" << std::endl << std::endl;
//    statusOFS << "Time for Gemm is " <<
//      timeGemm << " [s]" << std::endl << std::endl;
//    statusOFS << "Time for Alltoallv is " <<
//      timeAlltoallv << " [s]" << std::endl << std::endl;
//    statusOFS << "Time for Allreduce is " <<
//      timeAllreduce << " [s]" << std::endl << std::endl;
//#endif


  }


  return ;
}         // -----  end of method KohnSham::MultSpinor  ----- 

void
KohnSham::CalculateVexxACE_GPU ( Spinor& psi, Fourier& fft )
{
  // This assumes SetPhiEXX has been called so that phiEXX and psi
  // contain the same information. 

  // Since this is a projector, it should be done on the COARSE grid,
  // i.e. to the wavefunction directly

  //MPI_Barrier(domain_.comm);
  Real timeSta, timeEnd;
  GetTime( timeSta );

  int mpirank;  MPI_Comm_rank(domain_.comm, &mpirank);
  int mpisize;  MPI_Comm_size(domain_.comm, &mpisize);

  // Only works for single processor
  Int ntot      = fft.domain.NumGridTotal();
  Int ntotFine  = fft.domain.NumGridTotalFine();
  Int numStateTotal = psi.NumStateTotal();
  Int numStateLocal = psi.NumState();
  cuNumTns<Real>  cu_vexxPsi( ntot, 1, numStateLocal );
  NumTns<Real>  vexxPsi( ntot, 1, numStateLocal );

  // VexxPsi = V_{exx}*Phi.
  //SetValue( vexxPsi, 0.0 );
  cuda_setValue( cu_vexxPsi.Data(), 0.0, ntot*numStateLocal);
  psi.AddMultSpinorEXX_GPU( fft, phiEXX_, exxgkkR2C_,
      exxFraction_,  numSpin_, occupationRate_, cu_vexxPsi );

  
  //cuda_memcpy_GPU2CPU(vexxPsi.Data(),cu_vexxPsi.Data(), sizeof(Real)*ntot*numStateLocal);
  // Implementation based on SVD
  DblNumMat  M(numStateTotal, numStateTotal);
  
  GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for AddMulSpinorEXX with GPU  is " <<
         timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

  GetTime( timeSta );
  if(0){
    // FIXME
    Real SVDTolerance = 1e-4;
    // M = Phi'*vexxPsi
    blas::Gemm( 'T', 'N', numStateTotal, numStateTotal, ntot, 
        1.0, psi.Wavefun().Data(), ntot, vexxPsi.Data(), ntot,
        0.0, M.Data(), numStateTotal );

    DblNumMat  U( numStateTotal, numStateTotal );
    DblNumMat VT( numStateTotal, numStateTotal );
    DblNumVec  S( numStateTotal );
    SetValue( S, 0.0 );

    lapack::QRSVD( numStateTotal, numStateTotal, M.Data(), numStateTotal,
        S.Data(), U.Data(), U.m(), VT.Data(), VT.m() );


    for( Int g = 0; g < numStateTotal; g++ ){
      S[g] = std::sqrt( S[g] );
    }

    Int rankM = 0;
    for( Int g = 0; g < numStateTotal; g++ ){
      if( S[g] / S[0] > SVDTolerance ){
        rankM++;
      }
    }
    statusOFS << "rank of Phi'*VPhi matrix = " << rankM << std::endl;
    for( Int g = 0; g < rankM; g++ ){
      blas::Scal( numStateTotal, 1.0 / S[g], U.VecData(g), 1 );
    }

    vexxProj_.Resize( ntot, rankM );
    blas::Gemm( 'N', 'N', ntot, rankM, numStateTotal, 1.0, 
        vexxPsi.Data(), ntot, U.Data(), numStateTotal, 0.0,
        vexxProj_.Data(), ntot );
  }

  // Implementation based on Cholesky
  if(0){
    // M = -Phi'*vexxPsi. The minus sign comes from vexx is a negative
    // semi-definite matrix.
    blas::Gemm( 'T', 'N', numStateTotal, numStateTotal, ntot, 
        -1.0, psi.Wavefun().Data(), ntot, vexxPsi.Data(), ntot,
        0.0, M.Data(), numStateTotal );

    lapack::Potrf('L', numStateTotal, M.Data(), numStateTotal);

    blas::Trsm( 'R', 'L', 'T', 'N', ntot, numStateTotal, 1.0, 
        M.Data(), numStateTotal, vexxPsi.Data(), ntot );

    vexxProj_.Resize( ntot, numStateTotal );
    blas::Copy( ntot * numStateTotal, vexxPsi.Data(), 1, vexxProj_.Data(), 1 );
  }

  if(1){ //For MPI

    // Convert the column partition to row partition
    Int numStateBlocksize = numStateTotal / mpisize;
    Int ntotBlocksize = ntot / mpisize;

    Int numStateLocal = numStateBlocksize;
    Int ntotLocal = ntotBlocksize;

    if(mpirank < (numStateTotal % mpisize)){
      numStateLocal = numStateBlocksize + 1;
    }

    if(mpirank < (ntot % mpisize)){
      ntotLocal = ntotBlocksize + 1;
    }

    /*
    DblNumMat localPsiCol( ntot, numStateLocal );
    SetValue( localPsiCol, 0.0 );

    DblNumMat localVexxPsiCol( ntot, numStateLocal );
    SetValue( localVexxPsiCol, 0.0 );

    DblNumMat localPsiRow( ntotLocal, numStateTotal );
    SetValue( localPsiRow, 0.0 );

    DblNumMat localVexxPsiRow( ntotLocal, numStateTotal );
    SetValue( localVexxPsiRow, 0.0 );
    */
    DblNumMat localPsiRow( ntotLocal, numStateTotal );
    DblNumMat localVexxPsiRow( ntotLocal, numStateTotal );
    DblNumMat localPsiCol( ntot, numStateLocal );
    //DblNumMat localVexxPsiCol( ntot, numStateLocal );

    // Initialize
    //lapack::Lacpy( 'A', ntot, numStateLocal, vexxPsi.Data(), ntot, localVexxPsiCol.Data(), ntot );
    cuDblNumMat cu_temp( ntot, numStateLocal, false, cu_vexxPsi.Data() );
    cu_vexxProj_.Resize( ntotLocal, numStateTotal );
    GPU_AlltoallForward (cu_temp, cu_vexxProj_, domain_.comm);

    //lapack::Lacpy( 'A', ntot, numStateLocal, psi.Wavefun().Data(), ntot, localPsiCol.Data(), ntot );
    //AlltoallForward (localPsiCol, localPsiRow, domain_.comm);
    cuda_memcpy_CPU2GPU( cu_temp.Data(), psi.Wavefun().Data(), ntot*numStateLocal*sizeof(Real));
    cuDblNumMat cu_localPsiRow( ntotLocal, numStateTotal);
    GPU_AlltoallForward (cu_temp, cu_localPsiRow, domain_.comm);
    //cu_localPsiRow.CopyFrom(localPsiRow);
    //AlltoallForward (localVexxPsiCol, localVexxPsiRow, domain_.comm);

    DblNumMat MTemp( numStateTotal, numStateTotal );
    //SetValue( MTemp, 0.0 );
    cuDblNumMat cu_MTemp( numStateTotal, numStateTotal );
    //cuDblNumMat cu_vexxProj_( ntotLocal, numStateTotal );

    //cu_vexxProj_.CopyFrom(localVexxPsiRow);

    Real minus_one = -1.0;
    Real zero =  0.0;
    Real one  =  1.0;

    cublas::Gemm( HIPBLAS_OP_T, HIPBLAS_OP_N, numStateTotal, numStateTotal, ntotLocal,
                  &minus_one, cu_localPsiRow.Data(), ntotLocal, 
                  cu_vexxProj_.Data(), ntotLocal, &zero,
                  cu_MTemp.Data(), numStateTotal );
    cu_MTemp.CopyTo(MTemp);

    MPI_Allreduce( MTemp.Data(), M.Data(), numStateTotal * numStateTotal, MPI_DOUBLE, MPI_SUM, domain_.comm );
    /*
    blas::Gemm( 'T', 'N', numStateTotal, numStateTotal, ntotLocal,
        -1.0, localPsiRow.Data(), ntotLocal, 
        localVexxPsiRow.Data(), ntotLocal, 0.0,
        MTemp.Data(), numStateTotal );
    */
    //SetValue( M, 0.0 );
 
    //if ( mpirank == 0) {
    //  lapack::Potrf('L', numStateTotal, M.Data(), numStateTotal);
    //}
    //MPI_Bcast(M.Data(), numStateTotal * numStateTotal, MPI_DOUBLE, 0, domain_.comm);
    /*
    blas::Trsm( 'R', 'L', 'T', 'N', ntotLocal, numStateTotal, 1.0, 
        M.Data(), numStateTotal, localVexxPsiRow.Data(), ntotLocal );
    */

//    cu_MTemp.CopyFrom(M);
//    MAGMA::Potrf('L', numStateTotal, cu_MTemp.Data(), numStateTotal);

// Add by xmqin 20191202-----------------------------------------------
    lapack::Potrf( 'L', numStateTotal, M.Data(), numStateTotal );
    cu_MTemp.CopyFrom(M);
//---------------------------------------------------------------------
//
    cublas::Trsm( HIPBLAS_SIDE_RIGHT, HIPBLAS_FILL_MODE_LOWER, HIPBLAS_OP_T, HIPBLAS_DIAG_NON_UNIT, 
                  ntotLocal, numStateTotal, &one, cu_MTemp.Data(), numStateTotal, cu_vexxProj_.Data(),
                  ntotLocal);
    //cu_vexxProj_.CopyTo(localVexxPsiRow);
    vexxProj_.Resize( ntot, numStateLocal );
    cu_localPsiRow.Resize( ntot, numStateLocal ); // use this as a column distribution data.

    //AlltoallBackward (localVexxPsiRow, vexxProj_, domain_.comm);
    GPU_AlltoallBackward (cu_vexxProj_, cu_localPsiRow, domain_.comm);
    cu_localPsiRow.CopyTo( vexxProj_ );
  } //if(1)
  GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for GPU calculate vexxProjector  " <<
         timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

  // Sanity check. For debugging only
  //  if(0){
  //  // Make sure U and VT are the same. Should be an identity matrix
  //    blas::Gemm( 'N', 'N', numStateTotal, numStateTotal, numStateTotal, 1.0, 
  //        VT.Data(), numStateTotal, U.Data(), numStateTotal, 0.0,
  //        M.Data(), numStateTotal );
  //    statusOFS << "M = " << M << std::endl;
  //
  //    NumTns<Real> vpsit = psi.Wavefun();
  //    Int numProj = rankM;
  //    DblNumMat Mt(numProj, numStateTotal);
  //    
  //    blas::Gemm( 'T', 'N', numProj, numStateTotal, ntot, 1.0,
  //        vexxProj_.Data(), ntot, psi.Wavefun().Data(), ntot, 
  //        0.0, Mt.Data(), Mt.m() );
  //    // Minus sign comes from that all eigenvalues are negative
  //    blas::Gemm( 'N', 'N', ntot, numStateTotal, numProj, -1.0,
  //        vexxProj_.Data(), ntot, Mt.Data(), numProj,
  //        0.0, vpsit.Data(), ntot );
  //
  //    for( Int k = 0; k < numStateTotal; k++ ){
  //      Real norm = 0.0;
  //      for( Int ir = 0; ir < ntot; ir++ ){
  //        norm = norm + std::pow(vexxPsi(ir,0,k) - vpsit(ir,0,k), 2.0);
  //      }
  //      statusOFS << "Diff of vexxPsi " << std::sqrt(norm) << std::endl;
  //    }
  //  }


  return ;
}         // -----  end of method KohnSham::CalculateVexxACEGPU  ----- 


void
 KohnSham::CalculateVexxACEDF_GPU ( Spinor& psi, Fourier& fft, bool isFixColumnDF )
{
  // This assumes SetPhiEXX has been called so that phiEXX and psi
  // contain the same information. 

  // Since this is a projector, it should be done on the COARSE grid,
  // i.e. to the wavefunction directly

  MPI_Barrier(domain_.comm);
  int mpirank;  MPI_Comm_rank(domain_.comm, &mpirank);
  int mpisize;  MPI_Comm_size(domain_.comm, &mpisize);
  Real timeSta, timeEnd;

  GetTime( timeSta );
  // Only works for single processor
  Int ntot      = fft.domain.NumGridTotal();
  Int ntotFine  = fft.domain.NumGridTotalFine();
  Int numStateTotal = psi.NumStateTotal();
  Int numStateLocal = psi.NumState();
  //NumTns<Real>  vexxPsi( ntot, 1, numStateLocal );

  Int ntotBlocksize = ntot / mpisize;
  Int ntotLocal = ntotBlocksize;
  if(mpirank < (ntot % mpisize)){
    ntotLocal = ntotBlocksize + 1;
  }

  cuDblNumMat cu_vexxPsi( ntotLocal, numStateTotal );

  // VexxPsi = V_{exx}*Phi.
  DblNumMat  M(numStateTotal, numStateTotal);
  //SetValue( vexxPsi, 0.0 );
  //SetValue( M, 0.0 );

  // M = -Phi'*vexxPsi. The minus sign comes from vexx is a negative
  // semi-definite matrix.

  // why keep so many MPI_Alltoalls? while this can be easily avoided. 
  psi.AddMultSpinorEXXDF3_GPU( fft, phiEXX_, exxgkkR2C_, exxFraction_,  numSpin_, 
      occupationRate_, hybridDFNumMu_, hybridDFNumGaussianRandom_,
      hybridDFNumProcScaLAPACK_, BlockSizeScaLAPACK_,
      cu_vexxPsi, M, isFixColumnDF );

  GetTime( timeEnd );
  statusOFS << "GPU Time for AddMulSpinorEXXDF3_GPU  is " <<
          timeEnd - timeSta << " [s]" << std::endl << std::endl;

  GetTime( timeSta );
  // Implementation based on Cholesky
  /*
  if(0){
    lapack::Potrf('L', numStateTotal, M.Data(), numStateTotal);

    blas::Trsm( 'R', 'L', 'T', 'N', ntot, numStateTotal, 1.0, 
        M.Data(), numStateTotal, vexxPsi.Data(), ntot );

    vexxProj_.Resize( ntot, numStateTotal );
    blas::Copy( ntot * numStateTotal, vexxPsi.Data(), 1, vexxProj_.Data(), 1 );
  }
  */
  if(1){ //For MPI

    // Convert the column partition to row partition
    Int numStateBlocksize = numStateTotal / mpisize;
    Int ntotBlocksize = ntot / mpisize;

    Int numStateLocal = numStateBlocksize;
    Int ntotLocal = ntotBlocksize;

    if(mpirank < (numStateTotal % mpisize)){
      numStateLocal = numStateBlocksize + 1;
    }

    if(mpirank < (ntot % mpisize)){
      ntotLocal = ntotBlocksize + 1;
    }

    DblNumMat localVexxPsiCol( ntot, numStateLocal );
    //SetValue( localVexxPsiCol, 0.0 );

    DblNumMat localVexxPsiRow( ntotLocal, numStateTotal );
    //SetValue( localVexxPsiRow, 0.0 );

    // Initialize
    //lapack::Lacpy( 'A', ntot, numStateLocal, vexxPsi.Data(), ntot, localVexxPsiCol.Data(), ntot );

    //AlltoallForward (localVexxPsiCol, localVexxPsiRow, domain_.comm);
    
    /*
    if ( mpirank == 0) {
      lapack::Potrf('L', numStateTotal, M.Data(), numStateTotal);
    }

    MPI_Bcast(M.Data(), numStateTotal * numStateTotal, MPI_DOUBLE, 0, domain_.comm);
    blas::Trsm( 'R', 'L', 'T', 'N', ntotLocal, numStateTotal, 1.0, 
        M.Data(), numStateTotal, localVexxPsiRow.Data(), ntotLocal );
    */

    Real minus_one = -1.0;
    Real zero =  0.0;
    Real one  =  1.0;

    cu_vexxProj_.Resize( ntotLocal, numStateTotal );
    cu_vexxPsi.CopyTo( cu_vexxProj_);
    //cu_vexxProj_.CopyFrom(localVexxPsiRow);

    cuDblNumMat cu_M( numStateTotal, numStateTotal );
    //cu_M.CopyFrom(M);

    //MAGMA::Potrf('L', numStateTotal, cu_M.Data(), numStateTotal);
    //-------------------add by lijl 20191202
    lapack::Potrf( 'L', numStateTotal, M.Data(), numStateTotal );
    cu_M.CopyFrom(M);
//----------------------------
    cublas::Trsm( HIPBLAS_SIDE_RIGHT, HIPBLAS_FILL_MODE_LOWER, HIPBLAS_OP_T, HIPBLAS_DIAG_NON_UNIT, 
                  ntotLocal, numStateTotal, &one, cu_M.Data(), numStateTotal, cu_vexxProj_.Data(),
                  ntotLocal);

    cu_vexxProj_.CopyTo(localVexxPsiRow);

    vexxProj_.Resize( ntot, numStateLocal );

    AlltoallBackward (localVexxPsiRow, vexxProj_, domain_.comm);
  } //if(1)
  GetTime( timeEnd );
  statusOFS << "GPU Time for Vexx calculation is " <<
          timeEnd - timeSta << " [s]" << std::endl << std::endl;
  return ;
}

} // namespace dgdft
