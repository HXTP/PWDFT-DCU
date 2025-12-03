/*
   Copyright (c) 2012 The Regents of the University of California,
   through Lawrence Berkeley National Laboratory.  

Authors: Weile Jia and Lin Lin

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
/// @file cublas.hpp
/// @brief Thin interface to CUBLAS
/// @date 2016-10-21
#ifdef GPU  // only used for the GPU version of the PWDFT code. 
#ifndef __HIP_PLATFORM_HCC__
#define __HIP_PLATFORM_HCC__
#endif
#include  "environment.hpp"
#include <hip/hip_runtime.h>
#include "cublas.hpp"
#include "utility.hpp"
hipblasHandle_t hcublas;

/*
inline void __cublas_error(rocblas_status status, const char *file, int line, const char *msg)
{
    if(status != rocblas_status_success)
    {
      float* foo = NULL;
      float bar = foo[0];
      printf("Tried to segfault! %f\n", bar);

        printf("\nCUBLAS Error in %s, line %d: %s\n %s\n", file, line, hipGetErrorString(status), msg);
        hipDeviceReset();
        exit(-1);
    }
}

#define CUBLAS_ERROR(status, msg) __cublas_error( status, __FILE__, __LINE__, msg )
*/

namespace dgdft {

/// @namespace cublas
///
/// @brief Thin interface to CUBLAS
namespace cublas {

typedef  int               Int;

void Init(void)
{
    hipblasCreate(&hcublas);
    hipblasSetAtomicsMode(hcublas, HIPBLAS_ATOMICS_ALLOWED);
}

void Destroy(void)
{
    hipblasDestroy(hcublas);
}
// *********************************************************************
// Level 3 BLAS GEMM 
// *********************************************************************
 void Gemm 
            ( hipblasOperation_t transA, hipblasOperation_t transB, Int m, Int n, Int k,
            const float *alpha, const float* A, Int lda, const float* B, Int ldb,
            const float *beta,        float* C, Int ldc )
{
    hipblasSgemm(hcublas, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    return;
}

 void Gemm 
           ( hipblasOperation_t transA, hipblasOperation_t transB, Int m, Int n, Int k,
            const double *alpha, const double* A, Int lda, const double* B, Int ldb,
            double *beta,        double* C, Int ldc )
{
    hipblasDgemm(hcublas, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    return;
}
 void Gemv 
           ( hipblasOperation_t transA, Int m, Int n,
            const double *alpha, const double* A, Int lda, const double* x, Int incx,
            double *beta,        double* y, Int incy )
{
    hipblasDgemv(hcublas, transA, m, n, alpha, A, lda, x, incx, beta, y, incy);
    return;
}
 void Geam 
           ( hipblasOperation_t transA, hipblasOperation_t transB, Int m, Int n,
            const double *alpha, const double* A, Int lda, double *beta, const double* B, Int ldb,
            double* C, Int ldc )
{
    hipblasDgeam(hcublas, transA, transB, m, n, alpha, A, lda, beta,  B, ldb, C, ldc);
    return;
}
#ifdef _COMPLEX_
 void Gemm 
          ( hipblasOperation_t transA, hipblasOperation_t transB, Int m, Int n, Int k,
            const scomplex *alpha, const scomplex* A, Int lda, const scomplex* B, Int ldb,
            const scomplex *beta,        scomplex* C, Int ldc )
{
    cublasCgemm_v2(hcublas, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    return;
}
 void Gemm 
          ( cublasOperation_t transA, cublasOperation_t transB, Int m, Int n, Int k,
            const dcomplex *alpha, const dcomplex* A, Int lda, const dcomplex* B, Int ldb,
            const dcomplex *beta,        dcomplex* C, Int ldc )
{
    CUBLAS_ERROR(cublasZgemm_v2(hcublas, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc), " cublasZgemm failed !");
    return;
}
#endif

 void Gemm
           ( hipblasOperation_t transA, hipblasOperation_t transB, Int m, Int n, Int k,
            const double *alpha, double* A[], Int lda, double* B[], Int ldb,
            const double *beta,   double* C[], Int ldc ,Int batchCount)
{
    
    hipblasDgemmBatched(hcublas, transA, transB, m, n, k, alpha, const_cast<double**>(A), lda, const_cast<double**>(B), ldb, beta, C, ldc, batchCount);
    return;
}
 void GemmEx
           ( hipblasOperation_t transA, hipblasOperation_t transB, int m, int n, int k, const void *alpha, const void *A, hipblasDatatype_t Atype, int lda, const void *B, hipblasDatatype_t Btype, int ldb, const void *beta, void *C, hipblasDatatype_t Ctype, int ldc, hipblasDatatype_t computeType, hipblasGemmAlgo_t algo)
{  
    hipblasGemmEx(hcublas, transA, transB, m, n, k, alpha, A,  HIPBLAS_R_64F, lda, B,  HIPBLAS_R_64F, ldb, beta, C,  HIPBLAS_R_64F, ldc,  HIPBLAS_R_64F, HIPBLAS_GEMM_DEFAULT);
    return;
}
void Axpy( int n, const float * alpha, const float * x, int incx, float * y, int incy)
  {
      hipblasSaxpy(hcublas, n, alpha, x, incx, y, incy);
  }
  
void Axpy( int n, const double * alpha, const double * x, int incx, double * y, int incy)
{
     hipblasDaxpy(hcublas, n, alpha, x, incx, y, incy);
}

void Axpy( int n, const scomplex * alpha, const scomplex * x, int incx, scomplex * y, int incy)
{
    hipblasCaxpy(hcublas, n, alpha, x, incx, y, incy);
}

void Axpy( int n, const dcomplex * alpha, const dcomplex * x, int incx, dcomplex * y, int incy)
{
    hipblasZaxpy(hcublas, n, alpha, x, incx, y, incy);
}

 void Scal (int n, const float *alpha, float *x, int incx)
{
    hipblasSscal(hcublas, n, alpha, x, incx);
    return;
}
 void Scal (int n, const double *alpha, double *x, int incx)
{
    hipblasDscal(hcublas, n, alpha, x, incx);
    return;
}

 void Scal (int n, const scomplex *alpha, scomplex *x, int incx)
{
    hipblasCscal(hcublas, n, alpha, x, incx);
    return;
}
 void Scal (int n, const float *alpha, scomplex *x, int incx)
{
    hipblasCsscal(hcublas, n, alpha, x, incx);
    return;
}
 void Scal (int n, const dcomplex *alpha, dcomplex *x, int incx)
{
    hipblasZscal(hcublas, n, alpha, x, incx);
    return;
}
 void Scal (int n, const double *alpha, dcomplex *x, int incx)
{
    hipblasZdscal(hcublas, n, alpha, x, incx);
    return;
}

 void ScalP (hipblasHandle_t &plan, int n, const float *alpha, float *x, int incx)
{
    hipblasSscal(plan, n, alpha, x, incx);
    return;
}
 void ScalP (hipblasHandle_t &plan, int n, const double *alpha, double *x, int incx)
{
    hipblasDscal(plan, n, alpha, x, incx);
    return;
}
 void ScalP (hipblasHandle_t &plan, int n, const scomplex *alpha, scomplex *x, int incx)
{
    hipblasCscal(plan, n, alpha, x, incx);
    return;
}
 void ScalP (hipblasHandle_t &plan, int n, const float *alpha, scomplex *x, int incx)
{
    hipblasCsscal(plan, n, alpha, x, incx);
    return;
}
 void ScalP (hipblasHandle_t &plan, int n, const dcomplex *alpha, dcomplex *x, int incx)
{
    hipblasZscal(plan, n, alpha, x, incx);
    return;
}
 void ScalP (hipblasHandle_t &plan, int n, const double *alpha, dcomplex *x, int incx)
{
    hipblasZdscal(plan, n, alpha, x, incx);
    return;
}

 void Trsm ( hipblasSideMode_t side, hipblasFillMode_t uplo, hipblasOperation_t trans, 
             hipblasDiagType_t diag, int m, int n, const float *alpha,  const float *A,  
             int lda, float  *B, int ldb )
{
    hipblasStrsm(hcublas, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb );
    return;
}
 void Trsm ( hipblasSideMode_t side, hipblasFillMode_t uplo, hipblasOperation_t trans, 
             hipblasDiagType_t diag, int m, int n, const double *alpha, const double *A, 
             int lda, double *B, int ldb )
{
    hipblasDtrsm(hcublas, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb );
    return;
}

#ifdef _COMPLEX_
 void Trsm ( cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, 
             cublasDiagType_t diag, int m, int n, const cuComplex *alpha, 
             const cuComplex *A, int lda, cuComplex *B, int ldb )
{
    CUBLAS_ERROR( cublasCtrsm(hcublas, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb ), 
                  " cublas Ctrsm failed! ");
    return;
}
 void Trsm ( cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, 
             cublasDiagType_t diag, int m, int n, const cuDoubleComplex *alpha, 
             const cuDoubleComplex *A, int lda, cuDoubleComplex *B, int ldb )
{
    CUBLAS_ERROR( cublasZtrsm(hcublas, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb ), 
                  " cublas Ztrsm failed! ");
    return;
}
#endif

void batched_Gemm( hipblasOperation_t transA, hipblasOperation_t transB, int m, int n, int k, const double *alpha, double *A, int lda, double *B, int ldb, const double *beta, double *C, int ldc, int batchCount, int x, int y, int z)
{
	double ** h_A = (double **) malloc( sizeof(double*) *batchCount);
	double ** h_B = (double **) malloc( sizeof(double*) *batchCount);
	double ** h_C = (double **) malloc( sizeof(double*) *batchCount);
	for(int i = 0; i < batchCount; i++ )
 	{
		h_A[i] = A + i* lda;
		h_B[i] = B + i* ldb;
		h_C[i] = C + i*3*x + 3*y + z;
	}
	
	double **d_A, **d_B, **d_C;
	hipMalloc((void**)&d_A, sizeof(double*) * batchCount);
	hipMalloc((void**)&d_B, sizeof(double*) * batchCount);
	hipMalloc((void**)&d_C, sizeof(double*) * batchCount);
	
	hipMemcpy( d_A, h_A, batchCount *sizeof(double*), hipMemcpyHostToDevice);
	hipMemcpy( d_B, h_B, batchCount *sizeof(double*), hipMemcpyHostToDevice);
	hipMemcpy( d_C, h_C, batchCount *sizeof(double*), hipMemcpyHostToDevice);
	
	hipblasDgemmBatched(hcublas, transA, transB, m, n, k, alpha, (const double**)(d_A), lda, (const double**)(d_B), ldb, beta, d_C, ldc, batchCount);

	free(h_A);
	free(h_B);
	free(h_C);

	hipFree(d_A);
	hipFree(d_B);
	hipFree(d_C);
}

void batched_Gemm6( hipblasOperation_t transA, hipblasOperation_t transB, int m, int n, int k, const double *alpha, double *A, int lda, double *B, int ldb, const double *beta, double *C, int ldc, int batchCount, int x, int y, int z, 
 double *A2, double *B2, double *C2, int x2, int y2, int z2,
 double *A3, double *B3, double *C3, int x3, int y3, int z3,
 double *A4, double *B4, double *C4, int x4, int y4, int z4,
 double *A5, double *B5, double *C5, int x5, int y5, int z5,
 double *A6, double *B6, double *C6, int x6, int y6, int z6)
{
	double ** h_A = (double **) malloc( sizeof(double*) *6*batchCount);
	double ** h_B = (double **) malloc( sizeof(double*) *6*batchCount);
	double ** h_C = (double **) malloc( sizeof(double*) *6*batchCount);

	for(int i = 0; i < batchCount; i++ )
	{
		h_A[i] = A + i* lda;
		h_B[i] = B + i* ldb;
		h_C[i] = C + i*3*x + 3*y + z;
	}
	for(int i = 0; i < batchCount; i++ )
	{
		h_A[i+batchCount] = A2 + i* lda;
		h_B[i+batchCount] = B2 + i* ldb;
		h_C[i+batchCount] = C2 + i*3*x2 + 3*y2 + z2;
	}
	for(int i = 0; i < batchCount; i++ )
	{
		h_A[i+2*batchCount] = A3 + i* lda;
		h_B[i+2*batchCount] = B3 + i* ldb;
		h_C[i+2*batchCount] = C3 + i*3*x3 + 3*y3 + z3;
	}
	for(int i = 0; i < batchCount; i++ )
	{
		h_A[i+3*batchCount] = A4 + i* lda;
		h_B[i+3*batchCount] = B4 + i* ldb;
		h_C[i+3*batchCount] = C4 + i*3*x4 + 3*y4 + z4;
	}
	for(int i = 0; i < batchCount; i++ )
	{
		h_A[i+4*batchCount] = A5 + i* lda;
		h_B[i+4*batchCount] = B5 + i* ldb;
		h_C[i+4*batchCount] = C5 + i*3*x5 + 3*y5 + z5;
	}
	for(int i = 0; i < batchCount; i++ )
	{
		h_A[i+5*batchCount] = A6 + i* lda;
		h_B[i+5*batchCount] = B6 + i* ldb;
		h_C[i+5*batchCount] = C6 + i*3*x6 + 3*y6 + z6;
	}

	double **d_A, **d_B, **d_C;
	hipMalloc((void**)&d_A, sizeof(double*) * 6*batchCount);
	hipMalloc((void**)&d_B, sizeof(double*) * 6*batchCount);
	hipMalloc((void**)&d_C, sizeof(double*) * 6*batchCount);

	hipMemcpy( d_A, h_A, batchCount *6*sizeof(double*), hipMemcpyHostToDevice);
	hipMemcpy( d_B, h_B, batchCount *6*sizeof(double*), hipMemcpyHostToDevice);
	hipMemcpy( d_C, h_C, batchCount *6*sizeof(double*), hipMemcpyHostToDevice);
        hipblasDgemmBatched(hcublas, transA, transB, m, n, k, alpha, (const double**)(d_A), lda, (const double**)(d_B), ldb, beta, d_C, ldc, 6*batchCount);
	free(h_A);
	free(h_B);
	free(h_C);
	hipFree(d_A);
	hipFree(d_B);
	hipFree(d_C);
}


} // namespace cublas
} // namespace dgdft

#endif
