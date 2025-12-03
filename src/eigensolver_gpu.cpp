/*
   Copyright (c) 2012 The Regents of the University of California,
   through Lawrence Berkeley National Laboratory.  

Author: Lin Lin, Wei Hu, Amartya Banerjee, Weile Jia

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
/// @file eigensolver.cpp
/// @brief Eigensolver in the global domain or extended element.
/// @date 2014-04-25 First version of parallelized version. This does
/// not scale well.
/// @date 2014-08-07 Intra-element parallelization.  This has much
/// improved scalability.
/// @date 2016-04-04 Adjust some parameters for controlling the number
/// of iterations dynamically.
/// @date 2016-04-07 Add Chebyshev filtering.
#include  "eigensolver.hpp"
#include  "utility.hpp"
#include  "blas.hpp"
//#include  "magma.hpp"
#include  "cublas.hpp"
#include  "cuda_utils.h"
#include  "cu_nummat_impl.hpp"
#include  "lapack.hpp"
#include  "scalapack.hpp"
#include  "mpi_interf.hpp"

using namespace dgdft::scalapack;
using namespace dgdft::esdf;


namespace dgdft{

void
EigenSolver::PPCGSolveReal_GPU (
    Int          numEig,
    Int          scfIter,
    Int          eigMaxIter,
    Real         eigMinTolerance,
    Real         eigTolerance )
{
  // *********************************************************************
  // Initialization
  // *********************************************************************
  MPI_Comm mpi_comm = fftPtr_->domain.comm;
  MPI_Barrier(mpi_comm);
  Int mpirank;  MPI_Comm_rank(mpi_comm, &mpirank);
  Int mpisize;  MPI_Comm_size(mpi_comm, &mpisize);

  Int ntot = psiPtr_->NumGridTotal();
  Int ncom = psiPtr_->NumComponent();
  Int noccLocal = psiPtr_->NumState();
  Int noccTotal = psiPtr_->NumStateTotal();

  /* init the CUDA Device */
  hipblasStatus_t status;
  hipblasSideMode_t right  = HIPBLAS_SIDE_RIGHT;
  hipblasFillMode_t up     = HIPBLAS_FILL_MODE_UPPER;
  hipblasDiagType_t nondiag   = HIPBLAS_DIAG_NON_UNIT;
  hipblasOperation_t cu_transT = HIPBLAS_OP_T;
  hipblasOperation_t cu_transN = HIPBLAS_OP_N;
  hipblasOperation_t cu_transC = HIPBLAS_OP_C;
  //cuda_init_vtot();

/*  if(mpirank == 0)
  {
    std::cout << " GPU PPCG ........... " << std::endl;
    cuda_memory();
  }
*/

  Int height = ntot * ncom;
  Int width = noccTotal;
  Int lda = 3 * width;

  Int widthBlocksize = width / mpisize;
  Int heightBlocksize = height / mpisize;
  Int widthLocal = widthBlocksize;
  Int heightLocal = heightBlocksize;

  if(mpirank < (width % mpisize)){
    widthLocal = widthBlocksize + 1;
  }

  if(mpirank < (height % mpisize)){
    heightLocal = heightBlocksize + 1;
  }

  if( widthLocal != noccLocal ){
    throw std::logic_error("widthLocal != noccLocal.");
  }

  Int notconv = numEig;
  eigTolerance = std::sqrt( eigTolerance );

  statusOFS << "eigMaxIter = " << eigMaxIter << " eigTolerance = " << eigTolerance << std::endl;

  Real timeSta, timeEnd;
  Real timeSta1, timeEnd1;
  Real timeSta2, timeEnd2;
//  Real timeHpsi = 0.0;
//  Real timeStart = 0.0;
  Real timeGemmT = 0.0;
  Real timeGemmN = 0.0;
  Real timeBcast = 0.0;
  Real timeAllreduce = 0.0;
  Real timeAlltoallv = 0.0;
  Real timeAlltoallvMap = 0.0;
  Real timeSpinor = 0.0;
  Real timeTrsm = 0.0;
  Real timePotrf = 0.0;
  Real timeSyevd = 0.0;
  Real timeSygvd = 0.0;
  Real timeMpirank0 = 0.0;
  Real timeScaLAPACKFactor = 0.0;
  Real timeScaLAPACK = 0.0;
  Real timeSweepT = 0.0;
  Real timeCPU2DCUCopy = 0.0;
  Real timeDCU2DCUCopy = 0.0;
  Real timeOther = 0.0;
  Int  iterGemmT = 0;
  Int  iterGemmN = 0;
  Int  iterBcast = 0;
  Int  iterAllreduce = 0;
  Int  iterAlltoallv = 0;
  Int  iterAlltoallvMap = 0;
  Int  iterSpinor = 0;
  Int  iterTrsm = 0;
  Int  iterPotrf = 0;
  Int  iterSyevd = 0;
  Int  iterSygvd = 0;
  Int  iterMpirank0 = 0;
  Int  iterScaLAPACKFactor = 0;
  Int  iterScaLAPACK = 0;
  Int  iterSweepT = 0;
  Int  iterCPU2DCUCopy = 0;
  Int  iterDCU2DCUCopy = 0;
  Int  iterOther = 0;

  if( numEig > width ){
    std::ostringstream msg;
    msg
      << "Number of eigenvalues requested  = " << numEig << std::endl
      << "which is larger than the number of columns in psi = " << width << std::endl;
    throw std::runtime_error( msg.str().c_str() );
  }

  GetTime( timeSta2 );

  // S = ( X | W | P ) is a triplet used for LOBPCG.
  // W is the preconditioned residual
  DblNumMat       S( heightLocal, 3*width ),    AS( heightLocal, 3*width );
  cuDblNumMat  cu_S( heightLocal, 3*width ), cu_AS( heightLocal, 3*width );
  // AMat = S' * (AS),  BMat = S' * S
  //
  // AMat = (X'*AX   X'*AW   X'*AP)
  //      = (  *     W'*AW   W'*AP)
  //      = (  *       *     P'*AP)
  //
  // BMat = (X'*X   X'*W   X'*P)
  //      = (  *    W'*W   W'*P)
  //      = (  *      *    P'*P)
  //


  //    DblNumMat  AMat( 3*width, 3*width ), BMat( 3*width, 3*width );
  //    DblNumMat  AMatT1( 3*width, 3*width );

  // Temporary buffer array.
  // The unpreconditioned residual will also be saved in Xtemp
  DblNumMat  XTX( width, width );
  DblNumMat  XTXtemp1( width, width );

  DblNumMat  Xtemp( heightLocal, width );

  Real  resBlockNormLocal, resBlockNorm; // Frobenius norm of the residual block
  Real  resMax, resMin;
  DblNumVec resNormLocal( width );
  DblNumVec resNorm( width );

#if ( _DEBUGlevel_ >= 1 )
  if(mpirank  == 0)  { std::cout << " after malloc cuS, cu_AS " << std::endl; cuda_memory(); }
#endif
  // For convenience
  DblNumMat  X( heightLocal, width, false, S.VecData(0) );
  DblNumMat  W( heightLocal, width, false, S.VecData(width) );
  DblNumMat  P( heightLocal, width, false, S.VecData(2*width) );
  DblNumMat AX( heightLocal, width, false, AS.VecData(0) );
  DblNumMat AW( heightLocal, width, false, AS.VecData(width) );
  DblNumMat AP( heightLocal, width, false, AS.VecData(2*width) );

  DblNumMat  Xcol( height, widthLocal );
  DblNumMat  Wcol( height, widthLocal );
  DblNumMat AXcol( height, widthLocal );
  DblNumMat AWcol( height, widthLocal );

  // for GPU. please note we need to use copyTo adn copyFrom in the GPU matrix
  cuDblNumMat cu_XTX(width, width);
  cuDblNumMat cu_XTXtemp1(width, width);
  cuDblNumMat cu_Xtemp(heightLocal, width);

  cuDblNumMat cu_X ( heightLocal, width, false, cu_S.VecData(0)        );
  cuDblNumMat cu_W ( heightLocal, width, false, cu_S.VecData(width)    );
  cuDblNumMat cu_P ( heightLocal, width, false, cu_S.VecData(2*width)  );
  cuDblNumMat cu_AX( heightLocal, width, false, cu_AS.VecData(0)       );
  cuDblNumMat cu_AW( heightLocal, width, false, cu_AS.VecData(width)   );
  cuDblNumMat cu_AP( heightLocal, width, false, cu_AS.VecData(2*width) );

  cuDblNumMat cu_Xcol ( height, widthLocal );
  cuDblNumMat cu_Wcol ( height, widthLocal );
  cuDblNumMat cu_AXcol( height, widthLocal );
  cuDblNumMat cu_AWcol( height, widthLocal );

//  if(mpirank == 0)
//  {
//    std::cout << " GPU PPCG begins alloc partially done" << std::endl;
//    std::cout << " Each G parallel WF takes: " << heightLocal * width/1024/128 << " MB" << std::endl;
//    std::cout << " Each band paralelel WF s: " << height* widthLocal/1024/128 << " MB" << std::endl;
//    std::cout << " Each S  takes GPU memory: " << width* width/1024/128 << " MB" << std::endl;
//    cuda_memory();
//  }
  //Int info;
  bool isRestart = false;
  // numSet = 2    : Steepest descent (Davidson), only use (X | W)
  //        = 3    : Conjugate gradient, use all the triplet (X | W | P)
  Int numSet = 2;

  // numLocked is the number of converged vectors
  Int numLockedLocal = 0, numLockedSaveLocal = 0;
  Int numLockedTotal = 0, numLockedSaveTotal = 0;
  Int numLockedSave = 0;
  Int numActiveLocal = 0;
  Int numActiveTotal = 0;

  const Int numLocked = 0;  // Never perform locking in this version
  const Int numActive = width;

  bool isConverged = false;

  // Initialization
  SetValue( S, 0.0 );
  SetValue( AS, 0.0 );

  DblNumVec  eigValS(lda);
  SetValue( eigValS, 0.0 );

  // Initialize X by the data in psi
  Real one = 1.0;
  Real minus_one = -1.0;
  Real zero = 0.0;


  GetTime( timeSta );
  cuda_memcpy_CPU2GPU(cu_Xcol.Data(), psiPtr_->Wavefun().Data(), sizeof(Real)*height*widthLocal);
  GetTime( timeEnd );
  iterCPU2DCUCopy = iterCPU2DCUCopy + 1;
  timeCPU2DCUCopy = timeCPU2DCUCopy + ( timeEnd - timeSta );

  GetTime( timeSta );
  GPU_AlltoallForward( cu_Xcol, cu_X, mpi_comm );
  GetTime( timeEnd );
  iterAlltoallv = iterAlltoallv + 1;
  timeAlltoallv = timeAlltoallv + ( timeEnd - timeSta );

//  cu_Xcol.CopyTo(Xcol);
//  statusOFS << "Xcol " << Xcol << std::endl;

  // *********************************************************************
  // Main loop
  // *********************************************************************

  if(scfIter == 1)
  {
    // Orthogonalization through Cholesky factorization

    GetTime( timeSta );
#if 1
    cublas::Gemm( cu_transT, cu_transN, width, width, heightLocal, &one, cu_X.Data(),
        heightLocal, cu_X.Data(), heightLocal, &zero, cu_XTXtemp1.Data(), width );
#else
    cublas::GemmEx( cu_transT, cu_transN, width, width, heightLocal, &one, cu_X.Data(), HIPBLAS_R_64F, heightLocal, cu_X.Data(), HIPBLAS_R_64F, heightLocal, &zero, cu_XTXtemp1.Data(), HIPBLAS_R_64F, width, HIPBLAS_R_64F, HIPBLAS_GEMM_DEFAULT );
#endif
    GetTime( timeEnd );
    iterGemmT = iterGemmT + 1;
    timeGemmT = timeGemmT + ( timeEnd - timeSta );

    GetTime( timeSta );
    cu_XTXtemp1.CopyTo(XTXtemp1);
    GetTime( timeEnd );
    iterCPU2DCUCopy = iterCPU2DCUCopy + 1;
    timeCPU2DCUCopy = timeCPU2DCUCopy + ( timeEnd - timeSta );

    GetTime( timeSta );
    SetValue( XTX, 0.0 );
    MPI_Allreduce( XTXtemp1.Data(), XTX.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm );
    GetTime( timeEnd );
    iterAllreduce = iterAllreduce + 1;
    timeAllreduce = timeAllreduce + ( timeEnd - timeSta );

    if ( mpirank == 0) {
      GetTime( timeSta );
      lapack::Potrf( 'U', width, XTX.Data(), width );
      GetTime( timeEnd );
      iterPotrf = iterPotrf + 1;
      timePotrf = timePotrf + ( timeEnd - timeSta );
    }
    GetTime( timeSta );
    MPI_Bcast(XTX.Data(), width*width, MPI_DOUBLE, 0, mpi_comm);
    GetTime( timeEnd );
    iterBcast = iterBcast + 1;
    timeBcast = timeBcast + ( timeEnd - timeSta );

    // X <- X * U^{-1} is orthogonal
    GetTime( timeSta );
    cu_XTX.CopyFrom( XTX );
    GetTime( timeEnd );
    iterCPU2DCUCopy = iterCPU2DCUCopy + 1;
    timeCPU2DCUCopy = timeCPU2DCUCopy + ( timeEnd - timeSta );

    GetTime( timeSta );
    cublas::Trsm( right, up, cu_transN, nondiag, heightLocal, width, &one, cu_XTX.Data(), width, cu_X.Data(), heightLocal );
    GetTime( timeEnd );
    iterTrsm = iterTrsm + 1;
    timeTrsm = timeTrsm + ( timeEnd - timeSta );

    GetTime( timeSta );
    cu_XTX.CopyTo( XTX );
    GetTime( timeEnd );
    iterCPU2DCUCopy = iterCPU2DCUCopy + 1;
    timeCPU2DCUCopy = timeCPU2DCUCopy + ( timeEnd - timeSta );

    GetTime( timeSta );
    GPU_AlltoallBackward (cu_X, cu_Xcol, mpi_comm);
    GetTime( timeEnd );
    iterAlltoallv = iterAlltoallv + 1;
    timeAlltoallv = timeAlltoallv + ( timeEnd - timeSta );

   // cu_Xcol.CopyTo(Xcol);
  //  statusOFS << "Xcol " << Xcol << std::endl;

  }

  // Applying the Hamiltonian matrix
  {
    GetTime( timeSta );
    Spinor spnTemp(fftPtr_->domain, ncom, noccTotal, noccLocal, false, cu_Xcol.Data(), true);
    cuNumTns<Real> tnsTemp(ntot, ncom, noccLocal, false, cu_AXcol.Data());

    hamPtr_->MultSpinor_GPU( spnTemp, tnsTemp, *fftPtr_ );
    GetTime( timeEnd );
    iterSpinor = iterSpinor + 1;
    timeSpinor = timeSpinor + ( timeEnd - timeSta );
  }

  GetTime( timeSta );
  GPU_AlltoallForward (cu_AXcol, cu_AX, mpi_comm);
  GetTime( timeEnd );
  iterAlltoallv = iterAlltoallv + 1;
  timeAlltoallv = timeAlltoallv + ( timeEnd - timeSta );

//  cu_AXcol.CopyTo(AXcol);
//  statusOFS << "AXcol " << AXcol << std::endl;

  // Start the main loop
  Int iter = 0;
//  statusOFS << "Minimum tolerance is " << eigMinTolerance << std::endl;
//  statusOFS << " eigMaxIter  = " << eigMaxIter << std::endl;

  do{
    iter++;
#if ( _DEBUGlevel_ >= 1 )
    statusOFS << "iter = " << iter << std::endl;
#endif

    if( iter == 1 || isRestart == true )
      numSet = 2;
    else
      numSet = 3;

    // XTX <- X' * (AX)
    GetTime( timeSta );
#if 1
    cublas::Gemm( cu_transT, cu_transN, width, width, heightLocal, &one, cu_X.Data(),
                  heightLocal, cu_AX.Data(), heightLocal, &zero, cu_XTXtemp1.Data(), width );
#else
    cublas::GemmEx( cu_transT, cu_transN, width, width, heightLocal, &one, cu_X.Data(), HIPBLAS_R_64F, heightLocal, cu_X.Data(), HIPBLAS_R_64F, heightLocal, &zero, cu_XTXtemp1.Data(), HIPBLAS_R_64F, width, HIPBLAS_R_64F, HIPBLAS_GEMM_DEFAULT);
#endif
    GetTime( timeEnd );
    iterGemmT = iterGemmT + 1;
    timeGemmT = timeGemmT + ( timeEnd - timeSta );

    GetTime( timeSta );
    cu_XTXtemp1.CopyTo(XTXtemp1);
    GetTime( timeEnd );
    iterCPU2DCUCopy = iterCPU2DCUCopy + 1;
    timeCPU2DCUCopy = timeCPU2DCUCopy + ( timeEnd - timeSta );

    GetTime( timeSta );
    SetValue( XTX, 0.0 );
    MPI_Allreduce( XTXtemp1.Data(), XTX.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm );
    GetTime( timeEnd );
    iterAllreduce = iterAllreduce + 1;
    timeAllreduce = timeAllreduce + ( timeEnd - timeSta );

    // Compute the residual.
    // R <- AX - X*(X'*AX)
    GetTime( timeSta );
    cu_Xtemp.CopyFrom ( cu_AX );
    GetTime( timeEnd );
    iterDCU2DCUCopy = iterDCU2DCUCopy + 1;
    timeDCU2DCUCopy = timeDCU2DCUCopy + ( timeEnd - timeSta );

//#if 0
//    lapack::Lacpy( 'A', heightLocal, width, AX.Data(), heightLocal, Xtemp.Data(), heightLocal );
//#endif
    GetTime( timeSta );
    cu_XTX.CopyFrom(XTX);
    GetTime( timeEnd );
    iterCPU2DCUCopy = iterCPU2DCUCopy + 1;
    timeCPU2DCUCopy = timeCPU2DCUCopy + ( timeEnd - timeSta );

    GetTime( timeSta );
    cublas::Gemm( cu_transN, cu_transN, heightLocal, width, width, &minus_one, cu_X.Data(),
                  heightLocal, cu_XTX.Data(), width, &one, cu_Xtemp.Data(), heightLocal );
    GetTime( timeEnd );
    iterGemmN = iterGemmN + 1;
    timeGemmN = timeGemmN + ( timeEnd - timeSta );

    // Compute the Frobenius norm of the residual block

    cu_Xtemp.CopyTo(Xtemp);
    cu_XTX.CopyTo(XTX);


//    statusOFS << "Xtemp " << Xtemp << std::endl;
//    statusOFS << "XTX " << XTX << std::endl;

    SetValue( resNormLocal, 0.0 );
    GetTime( timeSta );
    
    for( Int k = 0; k < width; k++ ){
      resNormLocal(k) = Energy(DblNumVec(heightLocal, false, Xtemp.VecData(k)));
    }
    GetTime( timeEnd );
    iterOther = iterOther + 1;
    timeOther = timeOther + ( timeEnd - timeSta );

    SetValue( resNorm, 0.0 );
    GetTime( timeSta );
    MPI_Allreduce( resNormLocal.Data(), resNorm.Data(), width, MPI_DOUBLE,
        MPI_SUM, mpi_comm );
    GetTime( timeEnd );
    iterAllreduce = iterAllreduce + 1;
    timeAllreduce = timeAllreduce + ( timeEnd - timeSta );

    if ( mpirank == 0 ){
      GetTime( timeSta );
      for( Int k = 0; k < width; k++ ){
        resNorm(k) = std::sqrt( resNorm(k) ) / std::max( 1.0, std::abs( XTX(k,k) ) );
      }
      GetTime( timeEnd );
      iterMpirank0 = iterMpirank0 + 1;
      timeMpirank0 = timeMpirank0 + ( timeEnd - timeSta );

      iterOther = iterOther + 1;
      timeOther = timeOther + ( timeEnd - timeSta );
    }

    GetTime( timeSta );
    MPI_Bcast(resNorm.Data(), width, MPI_DOUBLE, 0, mpi_comm);
    GetTime( timeEnd );
    iterBcast = iterBcast + 1;
    timeBcast = timeBcast + ( timeEnd - timeSta );

    GetTime( timeSta );
    resMax = *(std::max_element( resNorm.Data(), resNorm.Data() + numEig ) );
    resMin = *(std::min_element( resNorm.Data(), resNorm.Data() + numEig ) );

//    statusOFS << "resNorm " << resNorm << std::endl;

    notconv = 0;
    for( Int i = 0; i < numEig; i++ ){
      if( resNorm[i] > eigTolerance ){
        notconv ++;
      }
    }
    GetTime( timeEnd );
    iterOther = iterOther + 3;
    timeOther = timeOther + ( timeEnd - timeSta );

#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Iter " << iter << " :resMax = " << resMax << " resMin = "
        << resMin << " notconv = " << notconv << std::endl;
#endif

    // LOCKING not supported, PPCG needs Rayleigh--Ritz to lock
    //        numActiveTotal = width - numLockedTotal;
    //        numActiveLocal = widthLocal - numLockedLocal;

    // Compute the preconditioned residual W = T*R.
    // The residual is saved in Xtemp

    // Convert from row format to column format.
    // MPI_Alltoallv
    // Only convert Xtemp here

    GetTime( timeSta );
    GPU_AlltoallBackward (cu_Xtemp, cu_Xcol, mpi_comm);
    GetTime( timeEnd );
    iterAlltoallv = iterAlltoallv + 1;
    timeAlltoallv = timeAlltoallv + ( timeEnd - timeSta );

    // Compute W = TW
    {
      GetTime( timeSta );
      Spinor spnTemp(fftPtr_->domain, ncom, noccTotal, widthLocal-numLockedLocal, false, cu_Xcol.Data(),true);
      cuNumTns<Real> tnsTemp(ntot, ncom, widthLocal-numLockedLocal, false, cu_Wcol.Data());

      spnTemp.AddTeterPrecond_GPU( fftPtr_, tnsTemp );
      GetTime( timeEnd );
      iterSpinor = iterSpinor + 1;
      timeSpinor = timeSpinor + ( timeEnd - timeSta );
    }

    // Compute AW = A*W
    {
      GetTime( timeSta );
      Spinor spnTemp(fftPtr_->domain, ncom, noccTotal, widthLocal-numLockedLocal, false, cu_Wcol.Data(), true);
      cuNumTns<Real> tnsTemp(ntot, ncom, widthLocal-numLockedLocal, false, cu_AWcol.Data());

      hamPtr_->MultSpinor_GPU( spnTemp, tnsTemp, *fftPtr_ );
      GetTime( timeEnd );
      iterSpinor = iterSpinor + 1;
      timeSpinor = timeSpinor + ( timeEnd - timeSta );
    }

    // Convert from column format to row format
    // MPI_Alltoallv
    // Only convert W and AW
    GetTime( timeSta );
    GPU_AlltoallForward (cu_Wcol, cu_W, mpi_comm);
    GPU_AlltoallForward (cu_AWcol, cu_AW, mpi_comm);
    GetTime( timeEnd );
    iterAlltoallv = iterAlltoallv + 2;
    timeAlltoallv = timeAlltoallv + ( timeEnd - timeSta );


    // W = W - X(X'W), AW = AW - AX(X'W)
    GetTime( timeSta );
#if 1
    cublas::Gemm( cu_transT, cu_transN, width, width, heightLocal, &one, cu_X.Data(),
                  heightLocal, cu_W.Data(), heightLocal, &zero, cu_XTXtemp1.Data(), width );
    //cublas::Gemm( cu_transT, cu_transN, width, width, heightLocal, &one, cu_X.Data(), heightLocal, cu_W.Data(), heightLocal, &zero, cu_XTXtemp1.Data(), width );
#else
    cublas::GemmEx( cu_transT, cu_transN, width, width, heightLocal, &one, cu_X.Data(), HIPBLAS_R_64F, heightLocal, cu_X.Data(), HIPBLAS_R_64F, heightLocal, &zero, cu_XTXtemp1.Data(), HIPBLAS_R_64F, width, HIPBLAS_R_64F, HIPBLAS_GEMM_DEFAULT );
#endif
    GetTime( timeEnd );
    iterGemmT = iterGemmT + 1;
    timeGemmT = timeGemmT + ( timeEnd - timeSta );

    GetTime( timeSta );
    cu_XTXtemp1.CopyTo(XTXtemp1);
    GetTime( timeEnd );
    iterCPU2DCUCopy = iterCPU2DCUCopy + 1;
    timeCPU2DCUCopy = timeCPU2DCUCopy + ( timeEnd - timeSta );

    GetTime( timeSta );
    SetValue( XTX, 0.0 );
    MPI_Allreduce( XTXtemp1.Data(), XTX.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm );
    GetTime( timeEnd );
    iterAllreduce = iterAllreduce + 1;
    timeAllreduce = timeAllreduce + ( timeEnd - timeSta );

    GetTime( timeSta );
    cu_XTX.CopyFrom(XTX);
    GetTime( timeEnd );
    iterCPU2DCUCopy = iterCPU2DCUCopy + 1;
    timeCPU2DCUCopy = timeCPU2DCUCopy + ( timeEnd - timeSta );

    GetTime( timeSta );
    cublas::Gemm( cu_transN, cu_transN, heightLocal, width, width, &minus_one, cu_X.Data(),
                  heightLocal, cu_XTX.Data(), width, &one, cu_W.Data(), heightLocal );
    cublas::Gemm( cu_transN, cu_transN, heightLocal, width, width, &minus_one,
                  cu_AX.Data(), heightLocal, cu_XTX.Data(), width, &one, cu_AW.Data(), heightLocal );
    GetTime( timeEnd );
    iterGemmN = iterGemmN + 2;
    timeGemmN = timeGemmN + ( timeEnd - timeSta );

    // Normalize columns of W
    Real normLocal[width];
    Real normGlobal[width];

    GetTime( timeSta );
    cuDblNumVec cu_normLocal(width);
    cuda_calculate_Energy( cu_W.Data(), cu_normLocal.Data(), width-numLockedLocal, heightLocal ); // note, numLockedLocal == 0
    cuda_memcpy_GPU2CPU( normLocal, cu_normLocal.Data(), sizeof(Real)*width);
    GetTime( timeEnd );
    iterOther = iterOther + 1;
    timeOther = timeOther + ( timeEnd - timeSta );

    GetTime( timeSta );
    MPI_Allreduce( &normLocal[0], &normGlobal[0], width, MPI_DOUBLE, MPI_SUM, mpi_comm );
    GetTime( timeEnd );
    iterAllreduce = iterAllreduce + 1;
    timeAllreduce = timeAllreduce + ( timeEnd - timeSta );

    GetTime( timeSta );
    cuda_memcpy_CPU2GPU(cu_normLocal.Data(), normGlobal, sizeof(Real)*width);
    cuda_batch_Scal( cu_W.Data(),  cu_normLocal.Data(), width, heightLocal);
    cuda_batch_Scal( cu_AW.Data(), cu_normLocal.Data(), width, heightLocal);
    GetTime( timeEnd );
    iterOther = iterOther + 2;
    timeOther = timeOther + ( timeEnd - timeSta );

    // P = P - X(X'P), AP = AP - AX(X'P)
    if( numSet == 3 ){
      GetTime( timeSta );
#if 1
      cublas::Gemm( cu_transT, cu_transN, width, width, heightLocal, &one, cu_X.Data(),
                  heightLocal, cu_P.Data(), heightLocal, &zero, cu_XTXtemp1.Data(), width );
#else
      cublas::GemmEx( cu_transT, cu_transN, width, width, heightLocal, &one, cu_X.Data(), HIPBLAS_R_64F, heightLocal, cu_P.Data(), HIPBLAS_R_64F, heightLocal, &zero, cu_XTXtemp1.Data(), HIPBLAS_R_64F, width, HIPBLAS_R_64F, HIPBLAS_GEMM_DEFAULT);
#endif
      GetTime( timeEnd );
      iterGemmT = iterGemmT + 1;
      timeGemmT = timeGemmT + ( timeEnd - timeSta );

      GetTime( timeSta );
      cu_XTXtemp1.CopyTo(XTXtemp1);
      GetTime( timeEnd ); 
      iterCPU2DCUCopy = iterCPU2DCUCopy + 1;     
      timeCPU2DCUCopy = timeCPU2DCUCopy + ( timeEnd - timeSta );

      GetTime( timeSta );
      SetValue( XTX, 0.0 );
      MPI_Allreduce( XTXtemp1.Data(), XTX.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm );
      GetTime( timeEnd );
      iterAllreduce = iterAllreduce + 1;
      timeAllreduce = timeAllreduce + ( timeEnd - timeSta );

      GetTime( timeSta );
      cu_XTX.CopyFrom( XTX );
      GetTime( timeEnd );
      iterCPU2DCUCopy = iterCPU2DCUCopy + 1;
      timeCPU2DCUCopy = timeCPU2DCUCopy + ( timeEnd - timeSta );

      GetTime( timeSta );
      cublas::Gemm( cu_transN, cu_transN, heightLocal, width, width, &minus_one,
                    cu_X.Data(), heightLocal, cu_XTX.Data(), width, &one, cu_P.Data(), heightLocal );

      cublas::Gemm( cu_transN, cu_transN, heightLocal, width, width, &minus_one,
                    cu_AX.Data(), heightLocal, cu_XTX.Data(), width, &one, cu_AP.Data(), heightLocal );
      GetTime( timeEnd );
      iterGemmN = iterGemmN + 2;
      timeGemmN = timeGemmN + ( timeEnd - timeSta );

      // Normalize the conjugate direction
      GetTime( timeSta );
      cuda_calculate_Energy( cu_P.Data(), cu_normLocal.Data(), width-numLockedLocal, heightLocal ); // note, numLockedLocal == 0
      cuda_memcpy_GPU2CPU( normLocal, cu_normLocal.Data(), sizeof(Real)*width);

      GetTime( timeEnd );
      iterOther = iterOther + 1;
      timeOther = timeOther + ( timeEnd - timeSta );

      GetTime( timeSta );
      MPI_Allreduce( &normLocal[0], &normGlobal[0], width, MPI_DOUBLE, MPI_SUM, mpi_comm );
      GetTime( timeEnd );
      iterAllreduce = iterAllreduce + 1;
      timeAllreduce = timeAllreduce + ( timeEnd - timeSta );

      GetTime( timeSta );
      cuda_memcpy_CPU2GPU(cu_normLocal.Data(), normGlobal, sizeof(Real)*width);
      cuda_batch_Scal( cu_P.Data(),  cu_normLocal.Data(), width, heightLocal);
      cuda_batch_Scal( cu_AP.Data(), cu_normLocal.Data(), width, heightLocal);
      GetTime( timeEnd );
      iterOther = iterOther + 2;
      timeOther = timeOther + ( timeEnd - timeSta );

    }


    // Perform the sweep
    Int sbSize = esdfParam.PPCGsbSize, nsb = width/sbSize; // this should be generalized to subblocks
    DblNumMat AMat( 3*sbSize, 3*sbSize ), BMat( 3*sbSize, 3*sbSize );
    DblNumMat AMatAll( 3*sbSize, 3*sbSize*nsb ), BMatAll( 3*sbSize, 3*sbSize*nsb ); // contains all nsb 3-by-3 matrices
    DblNumMat AMatAllLocal( 3*sbSize, 3*sbSize*nsb ), BMatAllLocal( 3*sbSize, 3*sbSize*nsb ); // contains local parts of all nsb 3-by-3 matrices

    // gpu
    cuDblNumMat cu_AMatAllLocal( 3*sbSize, 3*sbSize*nsb );
    cuDblNumMat cu_BMatAllLocal( 3*sbSize, 3*sbSize*nsb );

    // LOCKING NOT SUPPORTED, loop over all columns
    cuda_setValue( cu_AMatAllLocal.Data(), 0.0, 9*sbSize*sbSize*nsb);
    cuda_setValue( cu_BMatAllLocal.Data(), 0.0, 9*sbSize*sbSize*nsb);

    for( Int k = 0; k < nsb; k++ ){

      // fetch indiviual columns
      DblNumMat  x( heightLocal, sbSize, false, X.VecData(sbSize*k) );
      DblNumMat  w( heightLocal, sbSize, false, W.VecData(sbSize*k) );
      DblNumMat ax( heightLocal, sbSize, false, AX.VecData(sbSize*k) );
      DblNumMat aw( heightLocal, sbSize, false, AW.VecData(sbSize*k) );

      // gpu data structure.
      cuDblNumMat cu_ax( heightLocal, sbSize, false, cu_AX.VecData(sbSize*k)  );
      cuDblNumMat cu_x ( heightLocal, sbSize, false, cu_X.VecData(sbSize*k)  );
      cuDblNumMat cu_w ( heightLocal, sbSize, false, cu_W.VecData(sbSize*k) );
      cuDblNumMat cu_aw( heightLocal, sbSize, false, cu_AW.VecData(sbSize*k) );

      // Compute AMatAllLoc and BMatAllLoc
      // AMatAllLoc
      GetTime( timeSta );
      cublas::Gemm( cu_transT, cu_transN, sbSize, sbSize, heightLocal, &one, cu_x.Data(),
                  heightLocal, cu_ax.Data(), heightLocal, &zero, &cu_AMatAllLocal(0,3*sbSize*k), 3*sbSize );
      cublas::Gemm( cu_transT, cu_transN, sbSize, sbSize, heightLocal, &one, cu_w.Data(),
                   heightLocal, cu_aw.Data(), heightLocal, &zero, &cu_AMatAllLocal(sbSize,3*sbSize*k+sbSize), 3*sbSize);

      cublas::Gemm( cu_transT, cu_transN, sbSize, sbSize, heightLocal, &one, cu_x.Data(),
                   heightLocal, cu_aw.Data(), heightLocal, &zero, &cu_AMatAllLocal(0,3*sbSize*k+sbSize), 3*sbSize);

      // BMatAllLoc
      cublas::Gemm( cu_transT, cu_transN, sbSize, sbSize, heightLocal, &one, cu_x.Data(),
                   heightLocal, cu_x.Data(), heightLocal, &zero, &cu_BMatAllLocal(0,3*sbSize*k), 3*sbSize);

      cublas::Gemm( cu_transT, cu_transN, sbSize, sbSize, heightLocal, &one, cu_w.Data(),
                   heightLocal, cu_w.Data(), heightLocal, &zero, &cu_BMatAllLocal(sbSize,3*sbSize*k+sbSize), 3*sbSize);

      cublas::Gemm( cu_transT, cu_transN, sbSize, sbSize, heightLocal, &one, cu_x.Data(),
                   heightLocal, cu_w.Data(), heightLocal, &zero, &cu_BMatAllLocal(0,3*sbSize*k+sbSize), 3*sbSize);


      GetTime( timeEnd );
      iterGemmT = iterGemmT + 6;
      timeGemmT = timeGemmT + ( timeEnd - timeSta );

      if ( numSet == 3 ){

        DblNumMat  p( heightLocal, sbSize, false, P.VecData(k) );
        DblNumMat ap( heightLocal, sbSize, false, AP.VecData(k) );

        // GPU numMat
        cuDblNumMat  cu_p (heightLocal, sbSize, false, cu_P.VecData(k)  );
        cuDblNumMat cu_ap (heightLocal, sbSize, false, cu_AP.VecData(k) );

        // AMatAllLoc
        GetTime( timeSta );
        cublas::Gemm( cu_transT, cu_transN, sbSize, sbSize, heightLocal, &one, cu_p.Data(),
                     heightLocal, cu_ap.Data(), heightLocal, &zero, &cu_AMatAllLocal(2*sbSize,3*sbSize*k+2*sbSize), 3*sbSize);

        cublas::Gemm( cu_transT, cu_transN, sbSize, sbSize, heightLocal, &one, cu_x.Data(),
                     heightLocal, cu_ap.Data(), heightLocal, &zero, &cu_AMatAllLocal(0,3*sbSize*k+2*sbSize), 3*sbSize );


        cublas::Gemm( cu_transT, cu_transN, sbSize, sbSize, heightLocal, &one, cu_w.Data(),
                     heightLocal, cu_ap.Data(), heightLocal, &zero, &cu_AMatAllLocal(sbSize,3*sbSize*k+2*sbSize), 3*sbSize );


        // BMatAllLoc
        cublas::Gemm( cu_transT, cu_transN, sbSize, sbSize, heightLocal, &one, cu_p.Data(),
                     heightLocal, cu_p.Data(), heightLocal, &zero, &cu_BMatAllLocal(2*sbSize,3*sbSize*k+2*sbSize), 3*sbSize );


        cublas::Gemm( cu_transT, cu_transN, sbSize, sbSize, heightLocal, &one, cu_x.Data(),
                     heightLocal, cu_p.Data(), heightLocal, &zero, &cu_BMatAllLocal(0,3*sbSize*k+2*sbSize), 3*sbSize );


        cublas::Gemm( cu_transT, cu_transN, sbSize, sbSize, heightLocal, &one, cu_w.Data(),
                     heightLocal, cu_p.Data(), heightLocal, &zero, &cu_BMatAllLocal(sbSize,3*sbSize*k+2*sbSize), 3*sbSize );

        GetTime( timeEnd );
        iterGemmT = iterGemmT + 6;
        timeGemmT = timeGemmT + ( timeEnd - timeSta );

      }

    }

    GetTime( timeSta );
    cu_AMatAllLocal.CopyTo( AMatAllLocal );
    cu_BMatAllLocal.CopyTo( BMatAllLocal );
    GetTime( timeEnd );
    iterCPU2DCUCopy = iterCPU2DCUCopy + 2;
    timeCPU2DCUCopy = timeCPU2DCUCopy + ( timeEnd - timeSta );

    GetTime( timeSta );
    MPI_Allreduce( AMatAllLocal.Data(), AMatAll.Data(), 9*sbSize*sbSize*nsb, MPI_DOUBLE, MPI_SUM, mpi_comm );
    GetTime( timeEnd );
    iterAllreduce = iterAllreduce + 1;
    timeAllreduce = timeAllreduce + ( timeEnd - timeSta );

    GetTime( timeSta );
    MPI_Allreduce( BMatAllLocal.Data(), BMatAll.Data(), 9*sbSize*sbSize*nsb, MPI_DOUBLE, MPI_SUM, mpi_comm );
    GetTime( timeEnd );
    iterAllreduce = iterAllreduce + 1;
    timeAllreduce = timeAllreduce + ( timeEnd - timeSta );

    // Solve nsb small eigenproblems and update columns of X
    for( Int k = 0; k < nsb; k++ ){

      Real eigs[3*sbSize];
      DblNumMat  cx( sbSize, sbSize ), cw( sbSize, sbSize ), cp( sbSize, sbSize);
      DblNumMat tmp( heightLocal, sbSize );

      // gpu
      cuDblNumMat  cu_cx( sbSize, sbSize ), cu_cw( sbSize, sbSize ), cu_cp( sbSize, sbSize);
      cuDblNumMat cu_tmp( heightLocal, sbSize );

      // small eigensolve
      GetTime( timeSta );
      lapack::Lacpy( 'A', 3*sbSize, 3*sbSize, &AMatAll(0,3*sbSize*k), 3*sbSize, AMat.Data(), 3*sbSize );
      lapack::Lacpy( 'A', 3*sbSize, 3*sbSize, &BMatAll(0,3*sbSize*k), 3*sbSize, BMat.Data(), 3*sbSize );
      GetTime( timeEnd );
      iterOther = iterOther + 2;
      timeOther = timeOther + ( timeEnd - timeSta );

      Int dim = (numSet == 3) ? 3*sbSize : 2*sbSize;
      GetTime( timeSta );
      lapack::Sygvd(1, 'V', 'U', dim, AMat.Data(), 3*sbSize, BMat.Data(), 3*sbSize, eigs);
      GetTime( timeEnd );
      iterSygvd = iterSygvd + 1;
      timeSygvd = timeSygvd + ( timeEnd - timeSta );

      // fetch indiviual columns
      DblNumMat  x( heightLocal, sbSize, false, X.VecData(sbSize*k) );
      DblNumMat  w( heightLocal, sbSize, false, W.VecData(sbSize*k) );
      DblNumMat  p( heightLocal, sbSize, false, P.VecData(sbSize*k) );
      DblNumMat ax( heightLocal, sbSize, false, AX.VecData(sbSize*k) );
      DblNumMat aw( heightLocal, sbSize, false, AW.VecData(sbSize*k) );
      DblNumMat ap( heightLocal, sbSize, false, AP.VecData(sbSize*k) );

      // cuda parts.
      cuDblNumMat  cu_x( heightLocal, sbSize, false, cu_X.VecData(sbSize*k) );
      cuDblNumMat  cu_w( heightLocal, sbSize, false, cu_W.VecData(sbSize*k) );
      cuDblNumMat  cu_p( heightLocal, sbSize, false, cu_P.VecData(sbSize*k) );
      cuDblNumMat cu_ax( heightLocal, sbSize, false, cu_AX.VecData(sbSize*k) );
      cuDblNumMat cu_aw( heightLocal, sbSize, false, cu_AW.VecData(sbSize*k) );
      cuDblNumMat cu_ap( heightLocal, sbSize, false, cu_AP.VecData(sbSize*k) );


      //cuda_memcpy_CPU2GPU( cu_cx.Data(), &AMat(0,0), sbSize *sbSize*sizeof(Real));
      //cuda_memcpy_CPU2GPU( cu_cw.Data(), &AMat(sbSize,0), sbSize *sbSize*sizeof(Real));
      GetTime( timeSta );
      lapack::Lacpy( 'A', sbSize, sbSize, &AMat(0,0), 3*sbSize, cx.Data(), sbSize );
      lapack::Lacpy( 'A', sbSize, sbSize, &AMat(sbSize,0), 3*sbSize, cw.Data(), sbSize );
      GetTime( timeEnd );
      iterOther = iterOther + 2;
      timeOther = timeOther + ( timeEnd - timeSta );

      GetTime( timeSta );
      cuda_memcpy_CPU2GPU( cu_cx.Data(), cx.Data(), sbSize *sbSize*sizeof(Real));
      cuda_memcpy_CPU2GPU( cu_cw.Data(), cw.Data(), sbSize *sbSize*sizeof(Real));
      GetTime( timeEnd );
      iterCPU2DCUCopy = iterCPU2DCUCopy + 2;
      timeCPU2DCUCopy = timeCPU2DCUCopy + ( timeEnd - timeSta );

      //  p = w*cw + p*cp; x = x*cx + p; ap = aw*cw + ap*cp; ax = ax*cx + ap;
      if( numSet == 3 ){

        GetTime( timeSta );
        lapack::Lacpy( 'A', sbSize, sbSize, &AMat(2*sbSize,0), 3*sbSize, cp.Data(), sbSize );
        GetTime( timeEnd );
        iterOther = iterOther + 1;
        timeOther = timeOther + ( timeEnd - timeSta );  

        GetTime( timeSta );
        cuda_memcpy_CPU2GPU( cu_cp.Data(), cp.Data(), sbSize *sbSize*sizeof(Real));
        GetTime( timeEnd );
        iterCPU2DCUCopy = iterCPU2DCUCopy + 1;
        timeCPU2DCUCopy = timeCPU2DCUCopy + ( timeEnd - timeSta );

        // tmp <- p*cp
        GetTime( timeSta );
        cublas::Gemm( cu_transN, cu_transN, heightLocal, sbSize, sbSize, &one,
                cu_p.Data(), heightLocal, cu_cp.Data(), sbSize, &zero, cu_tmp.Data(),heightLocal);

        // p <- w*cw + tmp
        cublas::Gemm( cu_transN, cu_transN, heightLocal, sbSize, sbSize, &one,
                cu_w.Data(), heightLocal, cu_cw.Data(), sbSize, &one, cu_tmp.Data(),heightLocal);
        GetTime( timeEnd );
        iterGemmN = iterGemmN + 2;
        timeGemmN = timeGemmN + ( timeEnd - timeSta );

        GetTime( timeSta );
        cuda_memcpy_GPU2GPU( cu_p.Data(), cu_tmp.Data(), heightLocal*sbSize*sizeof(Real));
        GetTime( timeEnd );
        iterDCU2DCUCopy = iterDCU2DCUCopy + 1;
        timeDCU2DCUCopy = timeDCU2DCUCopy + ( timeEnd - timeSta );

        // tmp <- ap*cp
        GetTime( timeSta );
        cublas::Gemm( cu_transN, cu_transN, heightLocal, sbSize, sbSize, &one,
                cu_ap.Data(), heightLocal, cu_cp.Data(), sbSize, &zero, cu_tmp.Data(),heightLocal);
        // ap <- aw*cw + tmp
        cublas::Gemm( cu_transN, cu_transN, heightLocal, sbSize, sbSize, &one,
                cu_aw.Data(), heightLocal, cu_cw.Data(), sbSize, &one, cu_tmp.Data(),heightLocal);
        GetTime( timeEnd );
        iterGemmN = iterGemmN + 2;
        timeGemmN = timeGemmN + ( timeEnd - timeSta );
        GetTime( timeSta );
        cuda_memcpy_GPU2GPU( cu_ap.Data(), cu_tmp.Data(), heightLocal*sbSize*sizeof(Real));
        GetTime( timeEnd );
        iterDCU2DCUCopy = iterDCU2DCUCopy + 1;
        timeDCU2DCUCopy = timeDCU2DCUCopy + ( timeEnd - timeSta );

      }else{
        // p <- w*cw
        GetTime( timeSta );

        cublas::Gemm( cu_transN, cu_transN, heightLocal, sbSize, sbSize, &one,
                cu_w.Data(), heightLocal, cu_cw.Data(), sbSize, &zero, cu_p.Data(),heightLocal);
        // ap <- aw*cw
        //
        cublas::Gemm( cu_transN, cu_transN, heightLocal, sbSize, sbSize, &one,
                cu_aw.Data(), heightLocal, cu_cw.Data(), sbSize, &zero, cu_ap.Data(),heightLocal);
        GetTime( timeEnd );
        iterGemmN = iterGemmN + 2;
        timeGemmN = timeGemmN + ( timeEnd - timeSta );
      }

      // x <- x*cx + p
      GetTime( timeSta );
      cuda_memcpy_GPU2GPU( cu_tmp.Data(), cu_p.Data(), heightLocal*sbSize*sizeof(Real));
      GetTime( timeEnd );
      iterDCU2DCUCopy = iterDCU2DCUCopy + 1;
      timeDCU2DCUCopy = timeDCU2DCUCopy + ( timeEnd - timeSta );

      GetTime( timeSta );
      cublas::Gemm( cu_transN, cu_transN, heightLocal, sbSize, sbSize, &one,
              cu_x.Data(), heightLocal, cu_cx.Data(), sbSize, &one, cu_tmp.Data(),heightLocal);
      GetTime( timeEnd );
      iterGemmN = iterGemmN + 1;
      timeGemmN = timeGemmN + ( timeEnd - timeSta );

      GetTime( timeSta );
      cuda_memcpy_GPU2GPU( cu_x.Data(), cu_tmp.Data(), heightLocal*sbSize*sizeof(Real));

      // ax <- ax*cx + ap
      cuda_memcpy_GPU2GPU( cu_tmp.Data(), cu_ap.Data(), heightLocal*sbSize*sizeof(Real));
      GetTime( timeEnd );
      iterDCU2DCUCopy = iterDCU2DCUCopy + 2;
      timeDCU2DCUCopy = timeDCU2DCUCopy + ( timeEnd - timeSta );

      GetTime( timeSta );
      cublas::Gemm( cu_transN, cu_transN, heightLocal, sbSize, sbSize, &one,
              cu_ax.Data(), heightLocal, cu_cx.Data(), sbSize, &one, cu_tmp.Data(),heightLocal);
      GetTime( timeEnd );
      iterGemmN = iterGemmN + 1;
      timeGemmN = timeGemmN + ( timeEnd - timeSta );

      GetTime( timeSta );
      cuda_memcpy_GPU2GPU( cu_ax.Data(), cu_tmp.Data(), heightLocal*sbSize*sizeof(Real));
      GetTime( timeEnd );
      iterDCU2DCUCopy = iterDCU2DCUCopy + 1;
      timeDCU2DCUCopy = timeDCU2DCUCopy + ( timeEnd - timeSta );

    }

    // CholeskyQR of the updated block X
    GetTime( timeSta );
#if 1
    cublas::Gemm( cu_transT, cu_transN, width, width, heightLocal, &one, cu_X.Data(),
              heightLocal, cu_X.Data(), heightLocal, &zero, cu_XTXtemp1.Data(), width );
#else
    cublas::GemmEx( cu_transT, cu_transN, width, width, heightLocal, &one, cu_X.Data(), HIPBLAS_R_64F, heightLocal, cu_X.Data(), HIPBLAS_R_64F, heightLocal, &zero, cu_XTXtemp1.Data(), HIPBLAS_R_64F, width, HIPBLAS_R_64F, HIPBLAS_GEMM_DEFAULT );
#endif
    GetTime( timeEnd );
    iterGemmT = iterGemmT + 1;
    timeGemmT = timeGemmT + ( timeEnd - timeSta );

    GetTime( timeSta );
    cu_XTXtemp1.CopyTo(XTXtemp1);
    GetTime( timeEnd );
    iterCPU2DCUCopy = iterCPU2DCUCopy + 1; 
    timeCPU2DCUCopy = timeCPU2DCUCopy + ( timeEnd - timeSta );


    GetTime( timeSta );
    SetValue( XTX, 0.0 );
    MPI_Allreduce( XTXtemp1.Data(), XTX.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm );
    GetTime( timeEnd );
    iterAllreduce = iterAllreduce + 1;
    timeAllreduce = timeAllreduce + ( timeEnd - timeSta );

    if ( mpirank == 0) {
      GetTime( timeSta );
      lapack::Potrf( 'U',width, XTX.Data(), width );
      GetTime( timeEnd );
      iterPotrf = iterPotrf + 1;
      timePotrf = timePotrf + ( timeEnd - timeSta );
    }

    GetTime( timeSta );
    MPI_Bcast(XTX.Data(), width*width, MPI_DOUBLE, 0, mpi_comm);
    GetTime( timeEnd );
    iterBcast = iterBcast + 1;
    timeBcast = timeBcast + ( timeEnd - timeSta );

    GetTime( timeSta );
    cu_XTX.CopyFrom(XTX);
    GetTime( timeEnd );  
    iterCPU2DCUCopy = iterCPU2DCUCopy + 1;
    timeCPU2DCUCopy = timeCPU2DCUCopy + ( timeEnd - timeSta );

    // X <- X * U^{-1} is orthogonal
    GetTime( timeSta );
    cublas::Trsm( right, up, cu_transN, nondiag, heightLocal, width, &one, cu_XTX.Data(), width, cu_X.Data(), heightLocal );
    cublas::Trsm( right, up, cu_transN, nondiag, heightLocal, width, &one, cu_XTX.Data(), width, cu_AX.Data(), heightLocal );
    GetTime( timeEnd );
    iterTrsm = iterTrsm + 2;
    timeTrsm = timeTrsm + ( timeEnd - timeSta );

    GetTime( timeSta );
    cu_XTX.CopyTo( XTX);
    GetTime( timeEnd );
    iterCPU2DCUCopy = iterCPU2DCUCopy + 1;
    timeCPU2DCUCopy = timeCPU2DCUCopy + ( timeEnd - timeSta ); 

//   statusOFS << " iter " << iter << " resMin " << resMin << std::endl;
  } while( (iter < eigMaxIter) && (resMax > eigTolerance) );



  // *********************************************************************
  // Post processing
  // *********************************************************************

  // Obtain the eigenvalues and eigenvectors
  // if isConverged==true then XTX should contain the matrix X' * (AX); and X is an
  // orthonormal set

  if (!isConverged){
    GetTime( timeSta );
#if 1
    cublas::Gemm( cu_transT, cu_transN, width, width, heightLocal, &one, cu_X.Data(),
                heightLocal, cu_AX.Data(), heightLocal, &zero, cu_XTXtemp1.Data(), width);
#else
    cublas::GemmEx( cu_transT, cu_transN, width, width, heightLocal, &one, cu_X.Data(), HIPBLAS_R_64F, heightLocal, cu_X.Data(), HIPBLAS_R_64F, heightLocal, &zero, cu_XTXtemp1.Data(), HIPBLAS_R_64F, width, HIPBLAS_R_64F, HIPBLAS_GEMM_DEFAULT );
#endif
    GetTime( timeSta );
    cu_XTXtemp1.CopyTo(XTXtemp1);
    GetTime( timeEnd );
    iterCPU2DCUCopy = iterCPU2DCUCopy + 1;
    timeCPU2DCUCopy = timeCPU2DCUCopy + ( timeEnd - timeSta );

    GetTime( timeEnd );
    iterGemmT = iterGemmT + 1;
    timeGemmT = timeGemmT + ( timeEnd - timeSta );
    GetTime( timeSta );
    SetValue( XTX, 0.0 );
    MPI_Allreduce( XTXtemp1.Data(), XTX.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm );
    GetTime( timeEnd );
    iterAllreduce = iterAllreduce + 1;
    timeAllreduce = timeAllreduce + ( timeEnd - timeSta );
  }

  GetTime( timeSta1 );

  
  if ( mpirank == 0 ){
    GetTime( timeSta );
    lapack::Syevd( 'V','U',width, XTX.Data(), width, eigValS.Data() );
    GetTime( timeEnd );
    iterSyevd = iterSyevd + 1;
    timeSyevd = timeSyevd + ( timeEnd - timeSta );
  }
  
  GetTime( timeSta );
  MPI_Bcast(XTX.Data(), width*width, MPI_DOUBLE, 0, mpi_comm);
  MPI_Bcast(eigValS.Data(), width, MPI_DOUBLE, 0, mpi_comm);
  GetTime( timeEnd );
  iterBcast = iterBcast + 2;
  timeBcast = timeBcast + ( timeEnd - timeSta );

  GetTime( timeSta );
  cu_XTX.CopyFrom( XTX );
  GetTime( timeEnd );
  iterCPU2DCUCopy = iterCPU2DCUCopy + 1;
  timeCPU2DCUCopy = timeCPU2DCUCopy + ( timeEnd - timeSta );


  GetTime( timeSta );
  // X <- X*C
  cublas::Gemm( cu_transN, cu_transN, heightLocal, width, width, &one, cu_X.Data(),
                heightLocal, cu_XTX.Data(), width, &zero, cu_Xtemp.Data(), heightLocal);
  GetTime( timeEnd );
  iterGemmN = iterGemmN + 1;
  timeGemmN = timeGemmN + ( timeEnd - timeSta );

  GetTime( timeSta );
  cu_Xtemp.CopyTo( cu_X );
  GetTime( timeEnd );
  iterDCU2DCUCopy = iterDCU2DCUCopy + 1;
  timeDCU2DCUCopy = timeDCU2DCUCopy + ( timeEnd - timeSta );

  GetTime( timeSta );
  // AX <- AX*C
  cublas::Gemm( cu_transN, cu_transN, heightLocal, width, width, &one, cu_AX.Data(),
                heightLocal, cu_XTX.Data(), width, &zero, cu_Xtemp.Data(), heightLocal);
  GetTime( timeEnd );
  iterGemmN = iterGemmN + 1;
  timeGemmN = timeGemmN + ( timeEnd - timeSta );

  GetTime( timeSta );
  cu_Xtemp.CopyTo( cu_AX );
  GetTime( timeEnd );
  iterDCU2DCUCopy = iterDCU2DCUCopy + 1;
  timeDCU2DCUCopy = timeDCU2DCUCopy + ( timeEnd - timeSta );

  // Compute norms of individual eigenpairs
  cuDblNumVec cu_eigValS(lda);

  GetTime( timeSta );
  cu_eigValS.CopyFrom(eigValS);
  cu_X_Equal_AX_minus_X_eigVal(cu_Xtemp.Data(), cu_AX.Data(), cu_X.Data(),
                               cu_eigValS.Data(), width, heightLocal);
  //cu_Xtemp.CopyTo( Xtemp );
  GetTime( timeEnd );
  iterOther = iterOther + 1;
  timeOther = timeOther + ( timeEnd - timeSta );

  SetValue( resNormLocal, 0.0 );
  GetTime( timeSta );

  cuDblNumVec  cu_resNormLocal ( width );
  cuda_calculate_Energy( cu_Xtemp.Data(), cu_resNormLocal.Data(), width, heightLocal);
  cu_resNormLocal.CopyTo(resNormLocal);

  GetTime( timeEnd );
  iterOther = iterOther + 1;
  timeOther = timeOther + ( timeEnd - timeSta );

  GetTime( timeSta );
  SetValue( resNorm, 0.0 );
  MPI_Allreduce( resNormLocal.Data(), resNorm.Data(), width, MPI_DOUBLE,
      MPI_SUM, mpi_comm );
  iterAllreduce = iterAllreduce + 1;
  timeAllreduce = timeAllreduce + ( timeEnd - timeSta );

  if ( mpirank == 0 ){
    GetTime( timeSta );
    for( Int k = 0; k < width; k++ ){
      resNorm(k) = std::sqrt( resNorm(k) ) / std::max( 1.0, std::abs( eigValS(k) ) );
    }
    GetTime( timeEnd );
    iterOther = iterOther + 1;
    timeOther = timeOther + ( timeEnd - timeSta );

  }
  GetTime( timeSta );
  MPI_Bcast(resNorm.Data(), width, MPI_DOUBLE, 0, mpi_comm);
  GetTime( timeEnd );
  iterBcast = iterBcast + 1;
  timeBcast = timeBcast + ( timeEnd - timeSta );

  GetTime( timeSta );
  resMax = *(std::max_element( resNorm.Data(), resNorm.Data() + numEig ) );
  resMin = *(std::min_element( resNorm.Data(), resNorm.Data() + numEig ) );
  GetTime( timeEnd );
  iterOther = iterOther + 2;
  timeOther = timeOther + ( timeEnd - timeSta );


#if ( _DEBUGlevel_ >= 1 )
  statusOFS << "resNorm = " << resNorm << std::endl;
  statusOFS << "eigValS = " << eigValS << std::endl;
  statusOFS << "maxRes  = " << resMax  << std::endl;
  statusOFS << "minRes  = " << resMin  << std::endl;
#endif


#if ( _DEBUGlevel_ >= 2 )

  GetTime( timeSta );
  //cu_X.CopyFrom( X );
#if 1
  cublas::Gemm( cu_transT, cu_transN, width, width, heightLocal, &one, cu_X.Data(),
      heightLocal, cu_X.Data(), heightLocal, &zero, cu_XTXtemp1.Data(), width );
#else
  cublas::GemmEx( cu_transT, cu_transN, width, width, heightLocal, &one, cu_X.Data(), rocblas_datatype_f64_r, heightLocal, cu_X.Data(), rocblas_datatype_f64_r, heightLocal, &zero, cu_XTXtemp1.Data(), rocblas_datatype_f64_r, width, rocblas_datatype_f64_r, rocblas_gemm_algo_standard );
#endif

  GetTime( timeEnd );
  iterGemmT = iterGemmT + 1;
  timeGemmT = timeGemmT + ( timeEnd - timeSta );
  GetTime( timeSta );
  cu_XTXtemp1.CopyTo(XTXtemp1);
  GetTime( timeEnd );
  iterCPU2DCUCopy = iterCPU2DCUCopy + 1;
  timeCPU2DCUCopy = timeCPU2DCUCopy + ( timeEnd - timeSta );

  GetTime( timeSta );
  SetValue( XTX, 0.0 );
  MPI_Allreduce( XTXtemp1.Data(), XTX.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm );
  GetTime( timeEnd );
  iterAllreduce = iterAllreduce + 1;
  timeAllreduce = timeAllreduce + ( timeEnd - timeSta );

  statusOFS << "After the PPCG, XTX = " << XTX << std::endl;

#endif

  // Save the eigenvalues and eigenvectors back to the eigensolver data
  // structure

  eigVal_ = DblNumVec( width, true, eigValS.Data() );
  resVal_ = resNorm;

  GetTime( timeSta );
  GPU_AlltoallBackward (cu_X, cu_Xcol, mpi_comm);
  GetTime( timeEnd );
  iterAlltoallv = iterAlltoallv + 1;
  timeAlltoallv = timeAlltoallv + ( timeEnd - timeSta );

  GetTime( timeSta ); 
  cuda_memcpy_GPU2CPU( psiPtr_->Wavefun().Data(), cu_Xcol.Data(), sizeof(Real)*height*widthLocal);
  GetTime( timeEnd );
  iterCPU2DCUCopy = iterCPU2DCUCopy + 1;
  timeCPU2DCUCopy = timeCPU2DCUCopy + ( timeEnd - timeSta );

  // REPORT ACTUAL EIGENRESIDUAL NORMS?
  statusOFS << std::endl << "After " << iter
    << " PPCG iterations the min res norm is "
    << resMin << ". The max res norm is " << resMax << std::endl << std::endl;

  GetTime( timeEnd2 );

#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for iterSpinor       = " << iterSpinor          << "  timeSpinor       = " << timeSpinor << std::endl;
    statusOFS << "Time for iterGemmT        = " << iterGemmT           << "  timeGemmT        = " << timeGemmT << std::endl;
    statusOFS << "Time for iterGemmN        = " << iterGemmN           << "  timeGemmN        = " << timeGemmN << std::endl;
    statusOFS << "Time for iterAllreduce    = " << iterAllreduce       << "  timeAllreduce    = " << timeAllreduce << std::endl;
    statusOFS << "Time for iterAlltoallvMap = " << iterAlltoallvMap    << "  timeAlltoallvMap = " << timeAlltoallvMap   << std::endl;
    statusOFS << "Time for iterAlltoallv    = " << iterAlltoallv       << "  timeAlltoallv    = " << timeAlltoallv << std::endl;
    statusOFS << "Time for iterTrsm         = " << iterTrsm            << "  timeTrsm         = " << timeTrsm << std::endl;
    statusOFS << "Time for iterPotrf        = " << iterPotrf           << "  timePotrf        = " << timePotrf << std::endl;
    statusOFS << "Time for iterSyevd        = " << iterSyevd           << "  timeSyevd        = " << timeSyevd << std::endl;
    statusOFS << "Time for iterSygvd        = " << iterSygvd           << "  timeSygvd        = " << timeSygvd << std::endl;
//    statusOFS << "Time for iterSweepT       = " << iterSweepT          << "  timeSweepT       = " << timeSweepT << std::endl;
    statusOFS << "Time for iterCPU2DCUCopy  = " << iterCPU2DCUCopy     << "  timeCPU2DCUCopy  = " << timeCPU2DCUCopy << std::endl;
    statusOFS << "Time for iterDCU2DCUCopy  = " << iterDCU2DCUCopy     << "  timeDCU2DCUCopy  = " << timeDCU2DCUCopy << std::endl;
    statusOFS << "Time for iterOther        = " << iterOther           << "  timeOther        = " << timeOther << std::endl;
//    statusOFS << "Time for start overhead   = " << iterOther           << "  overheadTime     = " << timeStart << std::endl;
//    statusOFS << "Time for calTime          = " << iterOther           << "  calTime          = " << calTime   << std::endl;
//    statusOFS << "Time for FIRST            = " << iterOther           << "  firstTime        = " << firstTime << std::endl;
//    statusOFS << "Time for SECOND           = " << iterOther           << "  secondTime       = " << secondTime<< std::endl;
//    statusOFS << "Time for Third            = " << iterOther           << "  thirdTime        = " << thirdTime << std::endl;
//    statusOFS << "Time for overhead + first + second + third      = " << timeStart + calTime + firstTime + secondTime + thirdTime << std::endl;

    statusOFS << "Time for PPCG in PWDFT is " <<  timeEnd2 - timeSta2  << std::endl << std::endl;
#endif

    cuda_set_vtot_flag();   // set the vtot_flag to false.
    //cuda_clean_vtot();
    return ;
}         // -----  end of method EigenSolver::PPCGSolveReal  -----

void
EigenSolver::PPCGSolveReal_GPU (
    Int          numEig,
    Int          scfIter,
    Int          eigMaxIter,
    Real         eigMinTolerance,
    Real         eigTolerance,
    bool         isSerial)
{
  // *********************************************************************
  // Initialization
  // *********************************************************************
  MPI_Comm mpi_comm = fftPtr_->domain.comm;
  MPI_Barrier(mpi_comm);
  Int mpirank;  MPI_Comm_rank(mpi_comm, &mpirank);
  Int mpisize;  MPI_Comm_size(mpi_comm, &mpisize);

  Int ntot = psiPtr_->NumGridTotal();
  Int ncom = psiPtr_->NumComponent();
  Int noccTotal = psiPtr_->NumStateTotal();

  /* init the CUDA Device */
  hipblasStatus_t status;
  hipblasSideMode_t right  = HIPBLAS_SIDE_RIGHT;
  hipblasFillMode_t up     = HIPBLAS_FILL_MODE_UPPER;
  hipblasDiagType_t nondiag   = HIPBLAS_DIAG_NON_UNIT;
  hipblasOperation_t cu_transT = HIPBLAS_OP_T;
  hipblasOperation_t cu_transN = HIPBLAS_OP_N;
  hipblasOperation_t cu_transC = HIPBLAS_OP_C;

  Int height = ntot * ncom;
  Int width = noccTotal;
  Int lda = 3 * width;

  Int notconv = numEig;
  eigTolerance = std::sqrt( eigTolerance );

  statusOFS << "eigMaxIter = " << eigMaxIter << " eigTolerance = " << eigTolerance << std::endl;

  Real time1, time2;
  Real timeSta, timeEnd;
  Real timeSta1, timeEnd1;
  Real timeSta2, timeEnd2;
  Real firstTime = 0.0;
  Real secondTime= 0.0;
  Real thirdTime= 0.0;
  Real timeHpsi = 0.0;
  Real timeStart = 0.0;
  Real timeGemmT = 0.0;
  Real timeGemmN = 0.0;
  Real timeSpinor = 0.0;
  Real timeTrsm = 0.0;
  Real timePotrf = 0.0;
  Real timeSyevd = 0.0;
  Real timeSygvd = 0.0;
  Real timeSweepT = 0.0;
  Real timeCPU2DCUCopy = 0.0;
  Real timeDCU2DCUCopy = 0.0;
  Real timeOther = 0.0;
  Int  iterGemmT = 0;
  Int  iterGemmN = 0;
  Int  iterSpinor = 0;
  Int  iterTrsm = 0;
  Int  iterPotrf = 0;
  Int  iterSyevd = 0;
  Int  iterSygvd = 0;
  Int  iterSweepT = 0;
  Int  iterCPU2DCUCopy = 0;
  Int  iterDCU2DCUCopy = 0;
  Int  iterOther = 0;

  if( numEig > width ){
    std::ostringstream msg;
    msg 
      << "Number of eigenvalues requested  = " << numEig << std::endl
      << "which is larger than the number of columns in psi = " << width << std::endl;
    throw std::runtime_error( msg.str().c_str() );
  }

  GetTime( timeSta2 );
  // S = ( X | W | P ) is a triplet used for LOBPCG.  
  // W is the preconditioned residual
  DblNumMat       S( height, 3*width ),    AS( height, 3*width ); 
  cuDblNumMat  cu_S( height, 3*width ), cu_AS( height, 3*width ); 
  // AMat = S' * (AS),  BMat = S' * S
  // 
  // AMat = (X'*AX   X'*AW   X'*AP)
  //      = (  *     W'*AW   W'*AP)
  //      = (  *       *     P'*AP)
  //
  // BMat = (X'*X   X'*W   X'*P)
  //      = (  *    W'*W   W'*P)
  //      = (  *      *    P'*P)
  //

  // Temporary buffer array.
  // The unpreconditioned residual will also be saved in Xtemp
  DblNumMat  XTX( width, width );
  DblNumMat  XTXtemp1( width, width );

  DblNumMat  Xtemp( height, width );

  Real  resBlockNorm; // Frobenius norm of the residual block  
  Real  resMax, resMin;
  DblNumVec resNorm( width );

  // For convenience
  DblNumMat  X( height, width, false, S.VecData(0) );
  DblNumMat  W( height, width, false, S.VecData(width) );
  DblNumMat  P( height, width, false, S.VecData(2*width) );
  DblNumMat AX( height, width, false, AS.VecData(0) );
  DblNumMat AW( height, width, false, AS.VecData(width) );
  DblNumMat AP( height, width, false, AS.VecData(2*width) );

  // for GPU. please note we need to use copyTo adn copyFrom in the GPU matrix 
  cuDblNumMat cu_XTX(width, width);
  cuDblNumMat cu_XTXtemp1(width, width);
  cuDblNumMat cu_Xtemp(height, width);

  cuDblNumMat cu_X ( height, width, false, cu_S.VecData(0)        );
  cuDblNumMat cu_W ( height, width, false, cu_S.VecData(width)    );
  cuDblNumMat cu_P ( height, width, false, cu_S.VecData(2*width)  );
  cuDblNumMat cu_AX( height, width, false, cu_AS.VecData(0)       );
  cuDblNumMat cu_AW( height, width, false, cu_AS.VecData(width)   );
  cuDblNumMat cu_AP( height, width, false, cu_AS.VecData(2*width) );

  //Int info;
  bool isRestart = false;
  // numSet = 2    : Steepest descent (Davidson), only use (X | W)
  //        = 3    : Conjugate gradient, use all the triplet (X | W | P)
  Int numSet = 2;

  // numLocked is the number of converged vectors
  Int numLockedTotal = 0, numLockedSaveTotal = 0; 
  Int numLockedSave = 0;
  Int numActiveTotal = 0;

  const Int numLocked = 0;  // Never perform locking in this version
  const Int numActive = width;

  bool isConverged = false;

  // Initialization
  SetValue( S, 0.0 );
  SetValue( AS, 0.0 );

  DblNumVec  eigValS(lda);
  SetValue( eigValS, 0.0 );

  GetTime( timeSta );
  cuda_memcpy_CPU2GPU(cu_X.Data(), psiPtr_->Wavefun().Data(), sizeof(Real)*height*width);
  GetTime( timeEnd );
  iterCPU2DCUCopy = iterCPU2DCUCopy + 1;
  timeCPU2DCUCopy = timeCPU2DCUCopy + ( timeEnd - timeSta );

  Real one = 1.0;
  Real minus_one = -1.0;
  Real zero = 0.0;

  // *********************************************************************
  // Main loop
  // *********************************************************************
 
  // Orthogonalization through Cholesky factorization
  // needed only in first SCF cycle where psi is generated randomly
  if(scfIter == 1) 
  {
    GetTime( timeSta );
    cublas::Gemm( cu_transT, cu_transN, width, width, height, &one, cu_X.Data(), 
        height, cu_X.Data(), height, &zero, cu_XTX.Data(), width );
    GetTime( timeEnd );
    iterGemmT = iterGemmT + 1;
    timeGemmT = timeGemmT + ( timeEnd - timeSta );

    GetTime( timeSta );
    cu_XTX.CopyTo( XTX );
    GetTime( timeEnd );
    iterCPU2DCUCopy = iterCPU2DCUCopy + 1;
    timeCPU2DCUCopy = timeCPU2DCUCopy + ( timeEnd - timeSta );

  // each node do the Potrf, without the MPI_Bcast.
    GetTime( timeSta );
    lapack::Potrf( 'U', width, XTX.Data(), width );
    GetTime( timeEnd );
    iterPotrf = iterPotrf + 1;
    timePotrf = timePotrf + ( timeEnd - timeSta );

    // X <- X * U^{-1} is orthogonal
    GetTime( timeSta );
    cu_XTX.CopyFrom( XTX );
    GetTime( timeEnd );
    iterCPU2DCUCopy = iterCPU2DCUCopy + 1;
    timeCPU2DCUCopy = timeCPU2DCUCopy + ( timeEnd - timeSta );

    GetTime( timeSta );
    cublas::Trsm( right, up, cu_transN, nondiag, height, width, &one, cu_XTX.Data(), width, cu_X.Data(), height );
    GetTime( timeEnd );
    iterTrsm = iterTrsm + 1;
    timeTrsm = timeTrsm + ( timeEnd - timeSta );
  } // Orthogonalization step for first cycle


  // Applying the Hamiltonian matrix
  {
    GetTime( timeSta );
    Spinor spnTemp(fftPtr_->domain, ncom, noccTotal, noccTotal, false, cu_X.Data(), true);
    cuNumTns<Real> tnsTemp(ntot, ncom, noccTotal, false, cu_AX.Data());
  
    hamPtr_->MultSpinor_GPU( spnTemp, tnsTemp, *fftPtr_ );
    GetTime( timeEnd );
    iterSpinor = iterSpinor + 1;
    timeSpinor = timeSpinor + ( timeEnd - timeSta );
  }

  // Start the main loop
  Int iter = 0;

  do{
    iter++;
#if ( _DEBUGlevel_ >= 1 )
    statusOFS << "iter = " << iter << std::endl;
#endif

    if( iter == 1 || isRestart == true )
      numSet = 2;
    else
      numSet = 3;

    // XTX <- X' * (AX)
    GetTime( timeSta );
    cublas::Gemm( cu_transT, cu_transN, width, width, height, &one, cu_X.Data(),
                  height, cu_AX.Data(), height, &zero, cu_XTX.Data(), width );

    GetTime( timeEnd );
    iterGemmT = iterGemmT + 1;
    timeGemmT = timeGemmT + ( timeEnd - timeSta );

    // Compute the residual.
    // R <- AX - X*(X'*AX)
    GetTime( timeSta );
    cu_Xtemp.CopyFrom ( cu_AX );
    GetTime( timeEnd );
    iterDCU2DCUCopy = iterDCU2DCUCopy + 1;
    timeDCU2DCUCopy = timeDCU2DCUCopy + ( timeEnd - timeSta );

    GetTime( timeSta );
    cublas::Gemm( cu_transN, cu_transN, height, width, width, &minus_one, cu_X.Data(),
                  height, cu_XTX.Data(), width, &one, cu_Xtemp.Data(), height );
    GetTime( timeEnd );
    iterGemmN = iterGemmN + 1;
    timeGemmN = timeGemmN + ( timeEnd - timeSta );

    // Compute the Frobenius norm of the residual block
    cu_Xtemp.CopyTo(Xtemp);
    cu_XTX.CopyTo(XTX);

    SetValue( resNorm, 0.0 );
    GetTime( timeSta );
    for( Int k = 0; k < width; k++ ){
      resNorm(k) = Energy(DblNumVec(height, false, Xtemp.VecData(k)));
    }

    for( Int k = 0; k < width; k++ ){
      resNorm(k) = std::sqrt( resNorm(k) ) / std::max( 1.0, std::abs( XTX(k,k) ) );
    }

    resMax = *(std::max_element( resNorm.Data(), resNorm.Data() + numEig ) );
    resMin = *(std::min_element( resNorm.Data(), resNorm.Data() + numEig ) );

    notconv = 0;
    for( Int i = 0; i < numEig; i++ ){
      if( resNorm[i] > eigTolerance ){
        notconv ++;
      }
    }
    GetTime( timeEnd );
    iterOther = iterOther + 3;
    timeOther = timeOther + ( timeEnd - timeSta );

#if ( _DEBUGlevel_ >= 1 )
    statusOFS << "Iter " << iter << " :resMax = " << resMax << " resMin = "
        << resMin << " notconv = " << notconv << std::endl;
#endif

    // Compute W = TW
    {
      GetTime( timeSta );
      // PS:: In parallel version, Xcol is changed to Xtemp, but Xcol is  Xtemp in serial version!!
      // Note:  Xcol is not X but Xtemp, It's easy to make mistakes here!!!!!!! 
      Spinor spnTemp(fftPtr_->domain, ncom, noccTotal, width-numLocked, false, cu_Xtemp.Data(),true);
      cuNumTns<Real> tnsTemp(ntot, ncom, width-numLocked, false, cu_W.Data());

      //SetValue( tnsTemp, 0.0 );
      spnTemp.AddTeterPrecond_GPU( fftPtr_, tnsTemp );
      GetTime( timeEnd );
      iterSpinor = iterSpinor + 1;
      timeSpinor = timeSpinor + ( timeEnd - timeSta );
    }

    // Compute AW = A*W
    {
      GetTime( timeSta );
      Spinor spnTemp(fftPtr_->domain, ncom, noccTotal, width-numLocked, false, cu_W.Data(), true);
      cuNumTns<Real> tnsTemp(ntot, ncom, width-numLocked, false, cu_AW.Data());

      hamPtr_->MultSpinor_GPU( spnTemp, tnsTemp, *fftPtr_ );
      GetTime( timeEnd );
      iterSpinor = iterSpinor + 1;
      timeSpinor = timeSpinor + ( timeEnd - timeSta );
    }

    // W = W - X(X'W), AW = AW - AX(X'W)
    GetTime( timeSta );
    cublas::Gemm( cu_transT, cu_transN, width, width, height, &one, cu_X.Data(),
                  height, cu_W.Data(), height, &zero, cu_XTX.Data(), width );

    GetTime( timeEnd );
    iterGemmT = iterGemmT + 1;
    timeGemmT = timeGemmT + ( timeEnd - timeSta );

    GetTime( timeSta );
    cublas::Gemm( cu_transN, cu_transN, height, width, width, &minus_one, cu_X.Data(),
                  height, cu_XTX.Data(), width, &one, cu_W.Data(), height );
    cublas::Gemm( cu_transN, cu_transN, height, width, width, &minus_one,
                  cu_AX.Data(), height, cu_XTX.Data(), width, &one, cu_AW.Data(), height );
    GetTime( timeEnd );
    iterGemmN = iterGemmN + 2;
    timeGemmN = timeGemmN + ( timeEnd - timeSta );

    // Normalize columns of W
    Real norm[width]; 
//    Real normGlobal[width];

    GetTime( timeSta );
    cuDblNumVec cu_norm(width);
    cuda_calculate_Energy( cu_W.Data(), cu_norm.Data(), width-numLocked, height ); // note, numLockedLocal == 0
    GetTime( timeEnd );    
    iterOther = iterOther + 1;  
    timeOther = timeOther + ( timeEnd - timeSta );    

    GetTime( timeSta );
    cuda_batch_Scal( cu_W.Data(),  cu_norm.Data(), width, height);
    cuda_batch_Scal( cu_AW.Data(), cu_norm.Data(), width, height);
    GetTime( timeEnd );
    iterOther = iterOther + 2;
    timeOther = timeOther + ( timeEnd - timeSta );

    // P = P - X(X'P), AP = AP - AX(X'P)
    if( numSet == 3 ){
      
      GetTime( timeSta );
      cublas::Gemm( cu_transT, cu_transN, width, width, height, &one, cu_X.Data(),
                  height, cu_P.Data(), height, &zero, cu_XTX.Data(), width );
      GetTime( timeEnd );
      iterGemmT = iterGemmT + 1;
      timeGemmT = timeGemmT + ( timeEnd - timeSta );

      GetTime( timeSta );
      cublas::Gemm( cu_transN, cu_transN, height, width, width, &minus_one,
                    cu_X.Data(), height, cu_XTX.Data(), width, &one, cu_P.Data(), height );

      cublas::Gemm( cu_transN, cu_transN, height, width, width, &minus_one,
                    cu_AX.Data(), height, cu_XTX.Data(), width, &one, cu_AP.Data(), height );
      GetTime( timeEnd );
      iterGemmN = iterGemmN + 2;
      timeGemmN = timeGemmN + ( timeEnd - timeSta );

      // Normalize the conjugate direction
      GetTime( timeSta );
      cuda_calculate_Energy( cu_P.Data(), cu_norm.Data(), width-numLocked, height ); // note, numLockedLocal == 0

      GetTime( timeEnd );   
      iterOther = iterOther + 1;
      timeOther = timeOther + ( timeEnd - timeSta );   

      GetTime( timeSta );
      cuda_batch_Scal( cu_P.Data(),  cu_norm.Data(), width, height);
      cuda_batch_Scal( cu_AP.Data(), cu_norm.Data(), width, height);
      GetTime( timeEnd );
      iterOther = iterOther + 2;
      timeOther = timeOther + ( timeEnd - timeSta );
   
    }

    // Perform the sweep
    Int sbSize = esdfParam.PPCGsbSize, nsb = width/sbSize; // this should be generalized to subblocks 
    DblNumMat AMat( 3*sbSize, 3*sbSize ), BMat( 3*sbSize, 3*sbSize );
    DblNumMat AMatAll( 3*sbSize, 3*sbSize*nsb ), BMatAll( 3*sbSize, 3*sbSize*nsb ); // contains all nsb 3-by-3 matrices
   
    cuDblNumMat cu_AMatAll( 3*sbSize, 3*sbSize*nsb );
    cuDblNumMat cu_BMatAll( 3*sbSize, 3*sbSize*nsb );

    // LOCKING NOT SUPPORTED, loop over all columns 
    GetTime( time1);
    cuda_setValue( cu_AMatAll.Data(), 0.0, 9*sbSize*sbSize*nsb);
    cuda_setValue( cu_BMatAll.Data(), 0.0, 9*sbSize*sbSize*nsb);

    for( Int k = 0; k < nsb; k++ ){

      // fetch indiviual columns
      DblNumMat  x( height, sbSize, false, X.VecData(sbSize*k) );
      DblNumMat  w( height, sbSize, false, W.VecData(sbSize*k) );
      DblNumMat ax( height, sbSize, false, AX.VecData(sbSize*k) );
      DblNumMat aw( height, sbSize, false, AW.VecData(sbSize*k) );

      // gpu data structure. 
      cuDblNumMat cu_ax( height, sbSize, false, cu_AX.VecData(sbSize*k)  );
      cuDblNumMat cu_x ( height, sbSize, false, cu_X.VecData(sbSize*k)  );
      cuDblNumMat cu_w ( height, sbSize, false, cu_W.VecData(sbSize*k) );
      cuDblNumMat cu_aw( height, sbSize, false, cu_AW.VecData(sbSize*k) );

      // Compute AMatAllLoc and BMatAllLoc            
      // AMatAllLoc
      GetTime( timeSta );

      cublas::Gemm( cu_transT, cu_transN, sbSize, sbSize, height, &one, cu_x.Data(),
                    height, cu_ax.Data(), height, &zero, &cu_AMatAll(0,3*sbSize*k), 3*sbSize );
      cublas::Gemm( cu_transT, cu_transN, sbSize, sbSize, height, &one, cu_w.Data(),
                    height, cu_aw.Data(), height, &zero, &cu_AMatAll(sbSize,3*sbSize*k+sbSize), 3*sbSize);

      cublas::Gemm( cu_transT, cu_transN, sbSize, sbSize, height, &one, cu_x.Data(),
                   height, cu_aw.Data(), height, &zero, &cu_AMatAll(0,3*sbSize*k+sbSize), 3*sbSize);
      // BMatAllLoc            
      cublas::Gemm( cu_transT, cu_transN, sbSize, sbSize, height, &one, cu_x.Data(),
                   height, cu_x.Data(), height, &zero, &cu_BMatAll(0,3*sbSize*k), 3*sbSize);

      cublas::Gemm( cu_transT, cu_transN, sbSize, sbSize, height, &one, cu_w.Data(),
                   height, cu_w.Data(), height, &zero, &cu_BMatAll(sbSize,3*sbSize*k+sbSize), 3*sbSize);

      cublas::Gemm( cu_transT, cu_transN, sbSize, sbSize, height, &one, cu_x.Data(),
                   height, cu_w.Data(), height, &zero, &cu_BMatAll(0,3*sbSize*k+sbSize), 3*sbSize);


      GetTime( timeEnd );
      iterGemmT = iterGemmT + 6;
      timeGemmT = timeGemmT + ( timeEnd - timeSta );

      if ( numSet == 3 ){

        DblNumMat  p( height, sbSize, false, P.VecData(k) );
        DblNumMat ap( height, sbSize, false, AP.VecData(k) );
        
        // GPU numMat
        cuDblNumMat  cu_p (height, sbSize, false, cu_P.VecData(k)  );
        cuDblNumMat cu_ap (height, sbSize, false, cu_AP.VecData(k) );

        // AMatAllLoc
        GetTime( timeSta );
        cublas::Gemm( cu_transT, cu_transN, sbSize, sbSize, height, &one, cu_p.Data(),
                     height, cu_ap.Data(), height, &zero, &cu_AMatAll(2*sbSize,3*sbSize*k+2*sbSize), 3*sbSize);

        cublas::Gemm( cu_transT, cu_transN, sbSize, sbSize, height, &one, cu_x.Data(),
                     height, cu_ap.Data(), height, &zero, &cu_AMatAll(0,3*sbSize*k+2*sbSize), 3*sbSize );

        cublas::Gemm( cu_transT, cu_transN, sbSize, sbSize, height, &one, cu_w.Data(),
                     height, cu_ap.Data(), height, &zero, &cu_AMatAll(sbSize,3*sbSize*k+2*sbSize), 3*sbSize );

        // BMatAllLoc
        cublas::Gemm( cu_transT, cu_transN, sbSize, sbSize, height, &one, cu_p.Data(),
                     height, cu_p.Data(), height, &zero, &cu_BMatAll(2*sbSize,3*sbSize*k+2*sbSize), 3*sbSize );

        cublas::Gemm( cu_transT, cu_transN, sbSize, sbSize, height, &one, cu_x.Data(),
                     height, cu_p.Data(), height, &zero, &cu_BMatAll(0,3*sbSize*k+2*sbSize), 3*sbSize );

        cublas::Gemm( cu_transT, cu_transN, sbSize, sbSize, height, &one, cu_w.Data(),
                     height, cu_p.Data(), height, &zero, &cu_BMatAll(sbSize,3*sbSize*k+2*sbSize), 3*sbSize );

        GetTime( timeEnd );
        iterGemmT = iterGemmT + 6;
        timeGemmT = timeGemmT + ( timeEnd - timeSta );

      }             

    }

    GetTime( time1);
    cu_AMatAll.CopyTo( AMatAll );
    cu_BMatAll.CopyTo( BMatAll );
    GetTime( timeEnd );
    iterCPU2DCUCopy = iterCPU2DCUCopy + 2;
    timeCPU2DCUCopy = timeCPU2DCUCopy + ( timeEnd - timeSta );

    GetTime( time1);
    // Solve nsb small eigenproblems and update columns of X 

    for( Int k = 0; k < nsb; k++ ){

      Real eigs[3*sbSize];
      DblNumMat  cx( sbSize, sbSize ), cw( sbSize, sbSize ), cp( sbSize, sbSize);
      DblNumMat tmp( height, sbSize );      
       
      // gpu     
      cuDblNumMat  cu_cx( sbSize, sbSize ), cu_cw( sbSize, sbSize ), cu_cp( sbSize, sbSize);
      cuDblNumMat cu_tmp( height, sbSize );      

      // small eigensolve
      GetTime( timeSta );
      lapack::Lacpy( 'A', 3*sbSize, 3*sbSize, &AMatAll(0,3*sbSize*k), 3*sbSize, AMat.Data(), 3*sbSize );
      lapack::Lacpy( 'A', 3*sbSize, 3*sbSize, &BMatAll(0,3*sbSize*k), 3*sbSize, BMat.Data(), 3*sbSize );
      GetTime( timeEnd );
      iterOther = iterOther + 2;
      timeOther = timeOther + ( timeEnd - timeSta );

      Int dim = (numSet == 3) ? 3*sbSize : 2*sbSize;
      GetTime( timeSta );
      lapack::Sygvd(1, 'V', 'U', dim, AMat.Data(), 3*sbSize, BMat.Data(), 3*sbSize, eigs);
      GetTime( timeEnd );
      iterSygvd = iterSygvd + 1;
      timeSygvd = timeSygvd + ( timeEnd - timeSta );

      // fetch indiviual columns
      DblNumMat  x( height, sbSize, false, X.VecData(sbSize*k) );
      DblNumMat  w( height, sbSize, false, W.VecData(sbSize*k) );
      DblNumMat  p( height, sbSize, false, P.VecData(sbSize*k) );
      DblNumMat ax( height, sbSize, false, AX.VecData(sbSize*k) );
      DblNumMat aw( height, sbSize, false, AW.VecData(sbSize*k) );
      DblNumMat ap( height, sbSize, false, AP.VecData(sbSize*k) );

      // cuda parts. 
      cuDblNumMat  cu_x( height, sbSize, false, cu_X.VecData(sbSize*k) );
      cuDblNumMat  cu_w( height, sbSize, false, cu_W.VecData(sbSize*k) );
      cuDblNumMat  cu_p( height, sbSize, false, cu_P.VecData(sbSize*k) );
      cuDblNumMat cu_ax( height, sbSize, false, cu_AX.VecData(sbSize*k) );
      cuDblNumMat cu_aw( height, sbSize, false, cu_AW.VecData(sbSize*k) );
      cuDblNumMat cu_ap( height, sbSize, false, cu_AP.VecData(sbSize*k) );

      GetTime( timeSta );

      //cuda_memcpy_CPU2GPU( cu_cx.Data(), &AMat(0,0), sbSize *sbSize*sizeof(Real));
      //cuda_memcpy_CPU2GPU( cu_cw.Data(), &AMat(sbSize,0), sbSize *sbSize*sizeof(Real));
      lapack::Lacpy( 'A', sbSize, sbSize, &AMat(0,0), 3*sbSize, cx.Data(), sbSize );
      lapack::Lacpy( 'A', sbSize, sbSize, &AMat(sbSize,0), 3*sbSize, cw.Data(), sbSize );
      GetTime( timeEnd );
      iterOther = iterOther + 2;
      timeOther = timeOther + ( timeEnd - timeSta );

      GetTime( timeSta );
      cuda_memcpy_CPU2GPU( cu_cx.Data(), cx.Data(), sbSize *sbSize*sizeof(Real));
      cuda_memcpy_CPU2GPU( cu_cw.Data(), cw.Data(), sbSize *sbSize*sizeof(Real));
      GetTime( timeEnd );
      iterCPU2DCUCopy = iterCPU2DCUCopy + 2;
      timeCPU2DCUCopy = timeCPU2DCUCopy + ( timeEnd - timeSta );

      //  p = w*cw + p*cp; x = x*cx + p; ap = aw*cw + ap*cp; ax = ax*cx + ap;
      if( numSet == 3 ){

        GetTime( timeSta );
        lapack::Lacpy( 'A', sbSize, sbSize, &AMat(2*sbSize,0), 3*sbSize, cp.Data(), sbSize );
        GetTime( timeEnd );
        iterOther = iterOther + 1;
        timeOther = timeOther + ( timeEnd - timeSta );

        GetTime( timeSta );
        cuda_memcpy_CPU2GPU( cu_cp.Data(), cp.Data(), sbSize *sbSize*sizeof(Real));
        GetTime( timeEnd );
        iterCPU2DCUCopy = iterCPU2DCUCopy + 1;
        timeCPU2DCUCopy = timeCPU2DCUCopy + ( timeEnd - timeSta );
       
        // tmp <- p*cp 
        GetTime( timeSta );
        cublas::Gemm( cu_transN, cu_transN, height, sbSize, sbSize, &one,
                cu_p.Data(), height, cu_cp.Data(), sbSize, &zero, cu_tmp.Data(),height);

        // p <- w*cw + tmp
        cublas::Gemm( cu_transN, cu_transN, height, sbSize, sbSize, &one,
                cu_w.Data(), height, cu_cw.Data(), sbSize, &one, cu_tmp.Data(),height);

        GetTime( timeEnd );
        iterGemmN = iterGemmN + 2;
        timeGemmN = timeGemmN + ( timeEnd - timeSta );

        GetTime( timeSta );
        cuda_memcpy_GPU2GPU( cu_p.Data(), cu_tmp.Data(), height*sbSize*sizeof(Real));
        GetTime( timeEnd );
        iterDCU2DCUCopy = iterDCU2DCUCopy + 1;
        timeDCU2DCUCopy = timeDCU2DCUCopy + ( timeEnd - timeSta );

        // tmp <- ap*cp 
        GetTime( timeSta );
        cublas::Gemm( cu_transN, cu_transN, height, sbSize, sbSize, &one,
                cu_ap.Data(), height, cu_cp.Data(), sbSize, &zero, cu_tmp.Data(),height);
        // ap <- aw*cw + tmp
        cublas::Gemm( cu_transN, cu_transN, height, sbSize, sbSize, &one,
                cu_aw.Data(), height, cu_cw.Data(), sbSize, &one, cu_tmp.Data(),height);
        GetTime( timeEnd );
        iterGemmN = iterGemmN + 2;
        timeGemmN = timeGemmN + ( timeEnd - timeSta );
        GetTime( timeSta );
        cuda_memcpy_GPU2GPU( cu_ap.Data(), cu_tmp.Data(), height*sbSize*sizeof(Real));
        GetTime( timeEnd );
        iterDCU2DCUCopy = iterDCU2DCUCopy + 1;
        timeDCU2DCUCopy = timeDCU2DCUCopy + ( timeEnd - timeSta );

      }else{
        // p <- w*cw
        GetTime( timeSta );
       
        cublas::Gemm( cu_transN, cu_transN, height, sbSize, sbSize, &one,
                cu_w.Data(), height, cu_cw.Data(), sbSize, &zero, cu_p.Data(),height);
        // ap <- aw*cw
        //
        cublas::Gemm( cu_transN, cu_transN, height, sbSize, sbSize, &one,
                cu_aw.Data(), height, cu_cw.Data(), sbSize, &zero, cu_ap.Data(),height);
        GetTime( timeEnd );
        iterGemmN = iterGemmN + 2;
        timeGemmN = timeGemmN + ( timeEnd - timeSta );
      }

      // x <- x*cx + p
      GetTime( timeSta );
      cuda_memcpy_GPU2GPU( cu_tmp.Data(), cu_p.Data(), height*sbSize*sizeof(Real));
      GetTime( timeEnd );
      iterDCU2DCUCopy = iterDCU2DCUCopy + 1;
      timeDCU2DCUCopy = timeDCU2DCUCopy + ( timeEnd - timeSta );
     
      GetTime( timeSta );
      cublas::Gemm( cu_transN, cu_transN, height, sbSize, sbSize, &one,
              cu_x.Data(), height, cu_cx.Data(), sbSize, &one, cu_tmp.Data(),height);
      GetTime( timeEnd );
      iterGemmN = iterGemmN + 1;
      timeGemmN = timeGemmN + ( timeEnd - timeSta );
      
      GetTime( timeSta );
      cuda_memcpy_GPU2GPU( cu_x.Data(), cu_tmp.Data(), height*sbSize*sizeof(Real));

      // ax <- ax*cx + ap
      cuda_memcpy_GPU2GPU( cu_tmp.Data(), cu_ap.Data(), height*sbSize*sizeof(Real));
      GetTime( timeEnd );
      iterDCU2DCUCopy = iterDCU2DCUCopy + 2;
      timeDCU2DCUCopy = timeDCU2DCUCopy + ( timeEnd - timeSta );
      
      GetTime( timeSta );
      cublas::Gemm( cu_transN, cu_transN, height, sbSize, sbSize, &one,
              cu_ax.Data(), height, cu_cx.Data(), sbSize, &one, cu_tmp.Data(),height);
      GetTime( timeEnd );
      iterGemmN = iterGemmN + 1;
      timeGemmN = timeGemmN + ( timeEnd - timeSta );

      GetTime( timeSta );
      cuda_memcpy_GPU2GPU( cu_ax.Data(), cu_tmp.Data(), height*sbSize*sizeof(Real));
      GetTime( timeEnd );
      iterDCU2DCUCopy = iterDCU2DCUCopy + 1;
      timeDCU2DCUCopy = timeDCU2DCUCopy + ( timeEnd - timeSta );

    }

    // CholeskyQR of the updated block X
    GetTime( timeSta );
    cublas::Gemm( cu_transT, cu_transN, width, width, height, &one, cu_X.Data(), 
              height, cu_X.Data(), height, &zero, cu_XTX.Data(), width );
    GetTime( timeEnd );
    iterGemmT = iterGemmT + 1;
    timeGemmT = timeGemmT + ( timeEnd - timeSta );

    GetTime( timeSta );
    cu_XTX.CopyTo(XTX);
    GetTime( timeEnd );
    iterCPU2DCUCopy = iterCPU2DCUCopy + 1;
    timeCPU2DCUCopy = timeCPU2DCUCopy + ( timeEnd - timeSta );

    GetTime( timeSta );
    lapack::Potrf( 'U',width, XTX.Data(), width );
    GetTime( timeEnd );
    iterPotrf = iterPotrf + 1;
    timePotrf = timePotrf + ( timeEnd - timeSta );

    GetTime( timeSta );
    cu_XTX.CopyFrom(XTX);
    GetTime( timeEnd );
    iterCPU2DCUCopy = iterCPU2DCUCopy + 1;
    timeCPU2DCUCopy = timeCPU2DCUCopy + ( timeEnd - timeSta );

    // X <- X * U^{-1} is orthogonal
    GetTime( timeSta );
    cublas::Trsm( right, up, cu_transN, nondiag, height, width, &one, cu_XTX.Data(), width, cu_X.Data(), height );
    cublas::Trsm( right, up, cu_transN, nondiag, height, width, &one, cu_XTX.Data(), width, cu_AX.Data(), height );
    GetTime( timeEnd );
    iterTrsm = iterTrsm + 2;
    timeTrsm = timeTrsm + ( timeEnd - timeSta );

    GetTime( timeSta );
    cu_XTX.CopyTo( XTX);
    GetTime( timeEnd );
    iterCPU2DCUCopy = iterCPU2DCUCopy + 1;
    timeCPU2DCUCopy = timeCPU2DCUCopy + ( timeEnd - timeSta );

  } while( (iter < eigMaxIter) && (resMax > eigTolerance) );

  // *********************************************************************
  // Post processing
  // *********************************************************************

  // Obtain the eigenvalues and eigenvectors
  // if isConverged==true then XTX should contain the matrix X' * (AX); and X is an
  // orthonormal set

  if (!isConverged){
    GetTime( timeSta );
    cublas::Gemm( cu_transT, cu_transN, width, width, height, &one, cu_X.Data(),
                height, cu_AX.Data(), height, &zero, cu_XTX.Data(), width);
    GetTime( timeEnd );
    iterGemmT = iterGemmT + 1;
    timeGemmT = timeGemmT + ( timeEnd - timeSta );
  }

  GetTime( timeSta );
  cu_XTX.CopyTo(XTX);
  GetTime( timeEnd );
  iterCPU2DCUCopy = iterCPU2DCUCopy + 1;
  timeCPU2DCUCopy = timeCPU2DCUCopy + ( timeEnd - timeSta );

  GetTime( timeSta1 );

  GetTime( timeSta );
  lapack::Syevd( 'V','U',width, XTX.Data(), width, eigValS.Data() );
  GetTime( timeEnd );
  iterSyevd = iterSyevd + 1 ;
  timeSyevd  = timeSyevd  + (timeEnd-timeSta);

  GetTime( timeSta );
  cu_XTX.CopyFrom( XTX );
  GetTime( timeEnd );
  iterCPU2DCUCopy = iterCPU2DCUCopy + 1;
  timeCPU2DCUCopy = timeCPU2DCUCopy + ( timeEnd - timeSta );

  GetTime( timeSta );
  // X <- X*C
  cublas::Gemm( cu_transN, cu_transN, height, width, width, &one, cu_X.Data(),
                height, cu_XTX.Data(), width, &zero, cu_Xtemp.Data(), height);
  GetTime( timeEnd );
  iterGemmN = iterGemmN + 1;
  timeGemmN = timeGemmN + ( timeEnd - timeSta );

  GetTime( timeSta );
  cu_Xtemp.CopyTo( cu_X );
  GetTime( timeEnd );
  iterDCU2DCUCopy = iterDCU2DCUCopy + 1;
  timeDCU2DCUCopy = timeDCU2DCUCopy + ( timeEnd - timeSta );


  GetTime( timeSta );
  // AX <- AX*C
  cublas::Gemm( cu_transN, cu_transN, height, width, width, &one, cu_AX.Data(),
                height, cu_XTX.Data(), width, &zero, cu_Xtemp.Data(), height);
  GetTime( timeEnd );
  iterGemmN = iterGemmN + 1;
  timeGemmN = timeGemmN + ( timeEnd - timeSta );

  GetTime( timeSta );
  cu_Xtemp.CopyTo( cu_AX );
  GetTime( timeEnd );
  iterDCU2DCUCopy = iterDCU2DCUCopy + 1;
  timeDCU2DCUCopy = timeDCU2DCUCopy + ( timeEnd - timeSta );


  // Compute norms of individual eigenpairs 
  cuDblNumVec cu_eigValS(lda);

  GetTime( timeSta );
  cu_eigValS.CopyFrom(eigValS);
  cu_X_Equal_AX_minus_X_eigVal(cu_Xtemp.Data(), cu_AX.Data(), cu_X.Data(), 
                               cu_eigValS.Data(), width, height);
  //cu_Xtemp.CopyTo( Xtemp );
  GetTime( timeEnd );
  iterOther = iterOther + 1;
  timeOther = timeOther + ( timeEnd - timeSta );
      
  GetTime( timeSta );
  SetValue( resNorm, 0.0 );
  cuDblNumVec  cu_resNorm( width ); 
  cuda_calculate_Energy( cu_Xtemp.Data(), cu_resNorm.Data(), width, height);
  cu_resNorm.CopyTo(resNorm);

  for( Int k = 0; k < width; k++ ){
    resNorm(k) = std::sqrt( resNorm(k) ) / std::max( 1.0, std::abs( eigValS(k) ) );
  }

  resMax = *(std::max_element( resNorm.Data(), resNorm.Data() + numEig ) );
  resMin = *(std::min_element( resNorm.Data(), resNorm.Data() + numEig ) );
  GetTime( timeEnd );
  iterOther = iterOther + 4;
  timeOther = timeOther + ( timeEnd - timeSta );

#if ( _DEBUGlevel_ >= 1 )
  statusOFS << "resNorm = " << resNorm << std::endl;
  statusOFS << "eigValS = " << eigValS << std::endl;
  statusOFS << "maxRes  = " << resMax  << std::endl;
  statusOFS << "minRes  = " << resMin  << std::endl;
#endif



#if ( _DEBUGlevel_ >= 2 )

  GetTime( timeSta );
  cublas::Gemm( cu_transT, cu_transN, width, width, height, &one, cu_X.Data(),
                height, cu_X.Data(), height, &zero, cu_XTX.Data(), width );
  GetTime( timeEnd );
  iterGemmT = iterGemmT + 1;
  timeGemmT = timeGemmT + ( timeEnd - timeSta );

  GetTime( timeSta );
  cu_XTX.CopyTo(XTX);
  GetTime( timeEnd );
  iterCPU2DCUCopy = iterCPU2DCUCopy + 1;
  timeCPU2DCUCopy = timeCPU2DCUCopy + ( timeEnd - timeSta );

  statusOFS << "After the PPCG, XTX = " << XTX << std::endl;

#endif

  // Save the eigenvalues and eigenvectors back to the eigensolver data
  // structure

  eigVal_ = DblNumVec( width, true, eigValS.Data() );
  resVal_ = resNorm;

  GetTime( timeSta );
  cuda_memcpy_GPU2CPU( psiPtr_->Wavefun().Data(), cu_X.Data(), sizeof(Real)*height*width);
  GetTime( timeEnd );
  iterCPU2DCUCopy = iterCPU2DCUCopy + 1;
  timeCPU2DCUCopy = timeCPU2DCUCopy + ( timeEnd - timeSta );

  // REPORT ACTUAL EIGENRESIDUAL NORMS?
  statusOFS << std::endl << "After " << iter 
    << " PPCG iterations the min res norm is " 
    << resMin << ". The max res norm is " << resMax << std::endl << std::endl;

  GetTime( timeEnd2 );

#if ( _DEBUGlevel_ >= 0 )
  statusOFS << "Time for iterSpinor       = " << iterSpinor          << "  timeSpinor       = " << timeSpinor << std::endl;
  statusOFS << "Time for iterGemmT        = " << iterGemmT           << "  timeGemmT        = " << timeGemmT << std::endl;
  statusOFS << "Time for iterGemmN        = " << iterGemmN           << "  timeGemmN        = " << timeGemmN << std::endl;
//  statusOFS << "Time for iterAllreduce    = " << iterAllreduce       << "  timeAllreduce    = " << timeAllreduce << std::endl;
//  statusOFS << "Time for iterAlltoallvMap = " << iterAlltoallvMap    << "  timeAlltoallvMap = " << timeAlltoallvMap   << std::endl;
//  statusOFS << "Time for iterAlltoallv    = " << iterAlltoallv       << "  timeAlltoallv    = " << timeAlltoallv << std::endl;
  statusOFS << "Time for iterTrsm         = " << iterTrsm            << "  timeTrsm         = " << timeTrsm << std::endl;
  statusOFS << "Time for iterPotrf        = " << iterPotrf           << "  timePotrf        = " << timePotrf << std::endl;
  statusOFS << "Time for iterSyevd        = " << iterSyevd           << "  timeSyevd        = " << timeSyevd << std::endl;
  statusOFS << "Time for iterSygvd        = " << iterSygvd           << "  timeSygvd        = " << timeSygvd << std::endl;
//  statusOFS << "Time for iterSweepT       = " << iterSweepT          << "  timeSweepT       = " << timeSweepT << std::endl;
  statusOFS << "Time for iterCPU2DCUCopy  = " << iterCPU2DCUCopy     << "  timeCPU2DCUCopy  = " << timeCPU2DCUCopy << std::endl;
  statusOFS << "Time for iterDCU2DCUCopy  = " << iterDCU2DCUCopy     << "  timeDCU2DCUCopy  = " << timeDCU2DCUCopy << std::endl;
  statusOFS << "Time for iterOther        = " << iterOther           << "  timeOther        = " << timeOther << std::endl;
//  statusOFS << "Time for start overhead   = " << iterOther           << "  overheadTime     = " << timeStart << std::endl;
//  statusOFS << "Time for calTime          = " << iterOther           << "  calTime          = " << calTime   << std::endl;
//  statusOFS << "Time for FIRST            = " << iterOther           << "  firstTime        = " << firstTime << std::endl;
//  statusOFS << "Time for SECOND           = " << iterOther           << "  secondTime       = " << secondTime<< std::endl;
//  statusOFS << "Time for Third            = " << iterOther           << "  thirdTime        = " << thirdTime << std::endl;
//  statusOFS << "Time for overhead + first + second + third      = " << timeStart + calTime + firstTime + secondTime + thirdTime << std::endl;

  statusOFS << "Time for PPCG in PWDFT is " <<  timeEnd2 - timeSta2  << std::endl << std::endl;
#endif

    cuda_set_vtot_flag();   // set the vtot_flag to false.
    //cuda_clean_vtot();
    return ;
}         // -----  end of method EigenSolver::PPCGSolveReal  ----- 


} // namespace dgdft
