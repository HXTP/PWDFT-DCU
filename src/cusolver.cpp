/*
   Copyright (c) 2012 The Regents of the University of California,
   through Lawrence Berkeley National Laboratory.  

Authors: Weile Jia

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
/// @file device_solver.cpp
/// @brief Thin interface to device_solver
/// @date 2020-08-21
#ifdef  GPU
#include "cusolver.hpp"

rocsolver_handle cusolverH;


namespace dgdft {
namespace cusolver {

void Init(void)
{
  cusolverH = NULL;
  rocsolver_status status = rocblas_status_success;

  status = rocblas_create_handle(&cusolverH);
  assert(rocblas_status_success == status);


  if( status != rocblas_status_success ) {
    std::ostringstream msg;
    msg << " CU_SOLVER init Error... " << std::endl;
  }
  else{
    statusOFS << " cuSparse Solver is initialized successfully " << std::endl;
  }
   
}

void Destroy(void)
{
  if (cusolverH) rocsolver_destroy_handle(cusolverH);
}

// *********************************************************************
// Cholesky factorization
// *********************************************************************
void Potrf( char uplo_host, Int n, double* A, Int lda ) { 


  //int lwork;
  int *info;
  //double * work_array;
  rocblas_fill uplo; 
  //rocsolver_status cusolver_status;

  if(uplo_host == 'u' || uplo_host == 'U')
    uplo  = rocblas_fill_upper;
  else 
    uplo = rocblas_fill_lower;

  //assert(hipDeviceSynchronize() == hipSuccess);

  rocsolver_dpotrf( cusolverH, 
                               uplo, 
                               n, 
                               A, 
                               lda, 
                               info );
  
  //assert (cusolver_status == rocblas_status_success);
  //assert(cudaThreadSynchronize() == hipSuccess);
  //assert(hipMalloc( (void**) & work_array, sizeof(double) * lwork) == hipSuccess); 
  //assert(hipMalloc( (void**) & info, sizeof(int)  ) == hipSuccess); 

/* for debugging purposes. 
  double * A_host = new double[n*n];
  assert(cudaMemcpy( A_host, A, sizeof(double) * n * n , cudaMemcpyDeviceToHost) == hipSuccess);
  for(int i =0; i < n; i ++)
  for(int j =0; j < n; j ++)
   std::cout << " A ["<< i << "][" << j << "] = " << A_host[i*lda + j] << std::endl << std::flush;
*/
   
 // cusolver_status =
 // rocsolver_dpotrf(cusolverH,
 //                  uplo,
 //                  n,
 //                  A,
 //                  lda,
 //                  //work_array,
 //                  //lwork,
 //                  info);

  int info1;
  hipMemcpy( &info1, info, sizeof(int) , hipMemcpyDeviceToHost);
  if(info1 != 0) {
    std::ostringstream msg;
    msg << "cu_solver potrf returned with info = " << info1;
    ErrorHandling( msg.str().c_str() );
  }
/*
  if( cusolver_status ==  CUSOLVER_STATUS_NOT_INITIALIZED) {
    std::cout << " CUSOLVER_STATUS_NOT_INITIALIZED " << std::endl << std::flush;
  }

  if( cusolver_status ==  CUSOLVER_STATUS_INTERNAL_ERROR) {
    std::cout << " CUSOLVER_STATUS_INTERNAL_ERROR " << std::endl << std::flush;
  }
  if( cusolver_status ==  CUSOLVER_STATUS_ALLOC_FAILED) 
    std::cout << " CUSOLVER_STATUS_ALLOC_FAILED " << std::endl << std::flush;

  if( cusolver_status ==  CUSOLVER_STATUS_INVALID_VALUE ) 
    std::cout << "CUSOLVER_STATUS_INVALID_VALUE  " << std::endl << std::flush;

  if( cusolver_status ==  CUSOLVER_STATUS_ARCH_MISMATCH) 
    std::cout << " CUSOLVER_STATUS_ARCH_MISMATCH " << std::endl << std::flush;

  if( cusolver_status ==  CUSOLVER_STATUS_EXECUTION_FAILED) 
    std::cout << " CUSOLVER_STATUS_EXECUTION_FAILED " << std::endl << std::flush;
  std::cout << "status " <<cusolver_status <<std::flush;
  std::cout << " cuPotrf execute done" << lwork << std::endl;
*/
//  assert( hipSuccess == cudaDeviceSynchronize());
//  assert( hipSuccess == cudaFree(work_array));
//  assert( hipSuccess == cudaFree(info));
//  assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);

}
#if 0
void Potrf( char uplo_host, Int n, cuDoubleComplex * A, Int lda ) { 


  int lwork;
  int *info;
  cuDoubleComplex * work_array;
  cublasFillMode_t uplo; 
  cusolverStatus_t cusolver_status;

  if(uplo_host == 'u' || uplo_host == 'U')
    uplo  = CUBLAS_FILL_MODE_UPPER;
  else 
    uplo = CUBLAS_FILL_MODE_LOWER;

  //assert(cudaThreadSynchronize() == hipSuccess);

  cusolver_status = 
  cusolverDnZpotrf_bufferSize( cusolverH, 
                               uplo, 
                               n, 
                               A, 
                               lda, 
                               &lwork );
  
  assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);
  assert(cudaThreadSynchronize() == hipSuccess);
  assert(cudaMalloc( (void**) & work_array, sizeof(cuDoubleComplex) * lwork) == hipSuccess); 
  assert(cudaMalloc( (void**) & info, sizeof(int)  ) == hipSuccess); 

// for debugging purposes. 
/*
  double * A_host = new double[n*n];
  assert(cudaMemcpy( A_host, A, sizeof(double) * n * n , cudaMemcpyDeviceToHost) == hipSuccess);
  for(int i =0; i < n; i ++)
  for(int j =0; j < n; j ++)
   std::cout << " A ["<< i << "][" << j << "] = " << A_host[i*lda + j] << std::endl << std::flush;
*/
   
  cusolver_status =
  cusolverDnZpotrf(cusolverH,
                   uplo,
                   n,
                   A,
                   lda,
                   work_array,
                   lwork,
                   info);

  int info1;
  assert(cudaMemcpy( &info1, info, sizeof(int) , cudaMemcpyDeviceToHost) == hipSuccess);
  if(info1 != 0) {
    std::ostringstream msg;
    msg << "cu_solver potrf returned with info = " << info1;
    ErrorHandling( msg.str().c_str() );
  }
/*
  if( cusolver_status ==  CUSOLVER_STATUS_NOT_INITIALIZED) {
    std::cout << " CUSOLVER_STATUS_NOT_INITIALIZED " << std::endl << std::flush;
  }

  if( cusolver_status ==  CUSOLVER_STATUS_INTERNAL_ERROR) {
    std::cout << " CUSOLVER_STATUS_INTERNAL_ERROR " << std::endl << std::flush;
  }
  if( cusolver_status ==  CUSOLVER_STATUS_ALLOC_FAILED) 
    std::cout << " CUSOLVER_STATUS_ALLOC_FAILED " << std::endl << std::flush;

  if( cusolver_status ==  CUSOLVER_STATUS_INVALID_VALUE ) 
    std::cout << "CUSOLVER_STATUS_INVALID_VALUE  " << std::endl << std::flush;

  if( cusolver_status ==  CUSOLVER_STATUS_ARCH_MISMATCH) 
    std::cout << " CUSOLVER_STATUS_ARCH_MISMATCH " << std::endl << std::flush;

  if( cusolver_status ==  CUSOLVER_STATUS_EXECUTION_FAILED) 
    std::cout << " CUSOLVER_STATUS_EXECUTION_FAILED " << std::endl << std::flush;
*/
  assert( hipSuccess == cudaDeviceSynchronize());
  assert( hipSuccess == cudaFree(work_array));
  assert( hipSuccess == cudaFree(info));
  assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);

}
#endif
// *********************************************************************
// LU factorization (with partial pivoting)
// *********************************************************************

// *********************************************************************
// For solving the standard eigenvalue problem using the divide and
// conquer algorithm
// *********************************************************************
void Syev
( char uplo_c, Int n, double* A, Int lda, double* eigs ){
  // first assume we have everything on the GPU now. 
  // The perform the eigen value problem.
  double * d_work;
  rocblas_evect evect = rocblas_evect_original; 
  rocblas_fill uplo; 
  rocblas_status cusolver_status;

  if(uplo_c == 'U' || uplo_c == 'u' ) uplo = rocblas_fill_upper; 
                                 else uplo = rocblas_fill_lower;

  double * eigs_dev;
  double * E;
  int *devInfo;
  hipMalloc((void**)&devInfo, sizeof(int));
  hipMalloc( (void**) &eigs_dev, sizeof(double) * n );

  hipMalloc((void**)&E, sizeof(double)*n);

  cusolver_status = rocsolver_dsyev(
                    cusolverH,
                    evect,
                    uplo,
                    n,
                    A,
                    lda,
                    eigs_dev,
                    E,
                    devInfo);

  //cudaError_t cudaStat1;
  //cudaStat1 = cudaDeviceSynchronize();
  //assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
  //assert(hipSuccess == cudaStat1);

  int info;
  hipMemcpy( &info, devInfo, sizeof(int) , hipMemcpyDeviceToHost);
  hipMemcpy( eigs, eigs_dev, sizeof(double) * n , hipMemcpyDeviceToHost);
 
  //assert(cudaFree(d_work) == hipSuccess);
  //assert(cudaFree(devInfo) == hipSuccess);
  //assert(cudaFree(eigs_dev) == hipSuccess);

  if(info != 0) {
    std::ostringstream msg;
    msg << "cu_solver syev returned with info = " << info;
    ErrorHandling( msg.str().c_str() );
  }

}


#if 0
// *********************************************************************
// For solving the standard eigenvalue problem using the divide and
// conquer algorithm
// *********************************************************************

void Syevd
( char zjob, char uplo_c, Int n, double* A, Int lda, double* eigs ){


  // first assume we have everything on the GPU now. 
  // The perform the eigen value problem.
  // step 1: get the buffer.
  double * d_work;
  cusolverEigMode_t jobz; 
  cublasFillMode_t uplo; 
  cusolverStatus_t cusolver_status;

  if (zjob == 'v' || zjob == 'V') jobz = CUSOLVER_EIG_MODE_VECTOR; 
                            else  jobz = CUSOLVER_EIG_MODE_NOVECTOR;

  if(uplo_c == 'U' || uplo_c == 'u' ) uplo = CUBLAS_FILL_MODE_UPPER; 
                                 else uplo = CUBLAS_FILL_MODE_LOWER;

  double * eigs_dev;
  int *devInfo;
  assert(cudaMalloc((void**)&devInfo, sizeof(int)) == hipSuccess);
  assert(cudaMalloc( (void**) & eigs_dev, sizeof(double) * n ) == hipSuccess);

  int lwork;
  cusolver_status = cusolverDnDsyevd_bufferSize(
      cusolverH,
      jobz,
      uplo,
      n,
      A,
      lda,
      eigs_dev,
      &lwork);
  assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);

  assert(cudaMalloc((void**)&d_work, sizeof(double)*lwork) == hipSuccess);

  // step 2: compute spectrum
  cusolver_status = cusolverDnDsyevd(
                    cusolverH,
                    jobz,
                    uplo,
                    n,
                    A,
                    lda,
                    eigs_dev,
                    d_work,
                    lwork,
                    devInfo);

  cudaError_t cudaStat1;
  cudaStat1 = cudaDeviceSynchronize();
  assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
  assert(hipSuccess == cudaStat1);

  int info;
  assert(cudaMemcpy( &info, devInfo, sizeof(int) , cudaMemcpyDeviceToHost) == hipSuccess);
  assert(cudaMemcpy( eigs, eigs_dev, sizeof(double) * n , cudaMemcpyDeviceToHost) == hipSuccess);
 
  assert(cudaFree(d_work) == hipSuccess);
  assert(cudaFree(devInfo) == hipSuccess);
  assert(cudaFree(eigs_dev) == hipSuccess);

  if(info != 0) {
    std::ostringstream msg;
    msg << "cu_solver syevd returned with info = " << info;
    ErrorHandling( msg.str().c_str() );
  }

}


// *********************************************************************
// For solving the standard eigenvalue problem using the divide and
// conquer algorithm
// *********************************************************************

void Syevd
( char zjob, char uplo_c, Int n, cuDoubleComplex * A, Int lda, double* eigs ){


  // first assume we have everything on the GPU now. 
  // The perform the eigen value problem.
  // step 1: get the buffer.
  cuDoubleComplex * d_work;
  cusolverEigMode_t jobz; 
  cublasFillMode_t uplo; 
  cusolverStatus_t cusolver_status;

  if (zjob == 'v' || zjob == 'V') jobz = CUSOLVER_EIG_MODE_VECTOR; 
                            else  jobz = CUSOLVER_EIG_MODE_NOVECTOR;

  if(uplo_c == 'U' || uplo_c == 'u' ) uplo = CUBLAS_FILL_MODE_UPPER; 
                                 else uplo = CUBLAS_FILL_MODE_LOWER;

  double * eigs_dev;
  int *devInfo;
  assert(cudaMalloc((void**)&devInfo, sizeof(int)) == hipSuccess);
  assert(cudaMalloc( (void**) & eigs_dev, sizeof(double) * n ) == hipSuccess);

  int lwork;
  cusolver_status = cusolverDnZheevd_bufferSize(
      cusolverH,
      jobz,
      uplo,
      n,
      A,
      lda,
      eigs_dev,
      &lwork);
  assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);

  assert(cudaMalloc((void**)&d_work, sizeof(cuDoubleComplex)*lwork) == hipSuccess);

  // step 2: compute spectrum
  cusolver_status = cusolverDnZheevd(
                    cusolverH,
                    jobz,
                    uplo,
                    n,
                    A,
                    lda,
                    eigs_dev,
                    d_work,
                    lwork,
                    devInfo);

  cudaError_t cudaStat1;
  cudaStat1 = cudaDeviceSynchronize();
  assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
  assert(hipSuccess == cudaStat1);

  int info;
  assert(cudaMemcpy( &info, devInfo, sizeof(int) , cudaMemcpyDeviceToHost) == hipSuccess);
  assert(cudaMemcpy( eigs, eigs_dev, sizeof(double) * n , cudaMemcpyDeviceToHost) == hipSuccess);
 
  assert(cudaFree(d_work) == hipSuccess);
  assert(cudaFree(devInfo) == hipSuccess);
  assert(cudaFree(eigs_dev) == hipSuccess);

  if(info != 0) {
    std::ostringstream msg;
    msg << "cu_solver syevd returned with info = " << info;
    ErrorHandling( msg.str().c_str() );
  }

}
// *********************************************************************
// Copy
// *********************************************************************

void Lacpy( char uplo, Int m, Int n, const double* A, Int lda, double* B, Int ldb )
{
  printf( " the cuSolver Lacpy is not implemented yet! \n" ); 
}
#endif
} // namespace cuSolver
} // namespace dgdft
#endif
