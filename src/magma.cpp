/*
   Copyright (c) 2012 The Regents of the University of California,
   through Lawrence Berkeley National Laboratory.  

Authors: Jack Poulson and Lin Lin

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
/// @file magma.cpp
/// @brief Thin interface to magma
/// @date 2012-09-12
#include "magma.hpp"

namespace dgdft {
namespace MAGMA {
#ifdef GPU
void Init(void)
{
  magma_int_t st = magma_init();
  if( st != MAGMA_SUCCESS)
  {
    std::ostringstream msg;
    msg << " MAGMA init wrong " << std::endl;
  }
}

void Destroy(void)
{
  magma_int_t st = magma_finalize();
  if( st != MAGMA_SUCCESS)
  {
    std::ostringstream msg;
    msg << " MAGMA destroy wrong " << std::endl;
  }
}

// *********************************************************************
// Cholesky factorization
// *********************************************************************

void Potrf( char uplo, Int n, const double* A, Int lda )
{
  magma_int_t info;
  magma_uplo_t job; 

  if (uplo == 'u' || uplo == 'U')
       job = MagmaUpper;
  else 
       job = MagmaLower;

  magmaDouble_ptr Aptr = (magmaDouble_ptr) A;
  magma_dpotrf_gpu( job, n, Aptr, lda, &info );

  if( info < 0 )
  {
    std::ostringstream msg;
    msg << "dpotrf returned with info = " << info;
    ErrorHandling( msg.str().c_str() );
  }
  else if( info > 0 ){
    std::ostringstream msg;
    msg << "dpotrf returned with info = " << info << std::endl;
    msg << "A(info,info) = " << A[info-1+(info-1)*lda] << std::endl;
    ErrorHandling( msg.str().c_str() );
  }
}
/*
*/
// *********************************************************************
// LU factorization (with partial pivoting)
// *********************************************************************

/*
void Getrf( Int m, Int n, double* A, Int lda, Int* p )
{
  Int info;
  MAGMA(dgetrf)( &m, &n, A, &lda, p, &info );
  if( info < 0 )
  {
    std::ostringstream msg;
    msg << "dgetrf returned with info = " << info;
    ErrorHandling( msg.str().c_str() );
  }
  else if( info > 0 )
    ErrorHandling("Matrix is singular.");
}
*/

// *********************************************************************
// For solving the standard eigenvalue problem using the divide and
// conquer algorithm
// *********************************************************************

void Syevd
( char jobz, char uplo, Int n, double* A, Int lda, double* eigs ){

////
    magma_int_t *iwork, *isuppz, *ifail, aux_iwork[1];
    magma_int_t N, n2, info, lwork, liwork, ldda;
    N = n;
    ldda = lda;
    double aux_work[1];
#ifdef COMPLEX
    double *rwork, aux_rwork[1];
    magma_int_t lrwork;
#endif
////

  magma_vec_t zjob;
  if (jobz == 'v' || jobz == 'V') zjob = MagmaVec; else  zjob = MagmaNoVec;
  
  magma_uplo_t m_uplo;
  if(uplo == 'U') m_uplo = MagmaUpper; else m_uplo = MagmaLower;
  magmaDouble_ptr dA = (magmaDouble_ptr) A;  

  /*
  Int lwork = -1, info;
  Int liwork = -1;
  std::vector<double> work(1);
  std::vector<int>    iwork(1);
  */
  magma_dsyevd_gpu( zjob, m_uplo,
                    N, NULL, lda, NULL,  // A, w
                    NULL, lda,            // host A
                    aux_work,  -1,
                    #ifdef COMPLEX
                    aux_rwork, -1,
                    #endif
                    aux_iwork, -1,
                    &info );

  lwork  = (magma_int_t) MAGMA_D_REAL( aux_work[0] );
  liwork = aux_iwork[0];
  double * h_R, *h_work;

  magma_dmalloc_pinned( &h_R,    N*lda  );
  magma_dmalloc_pinned( &h_work, lwork  );
  magma_imalloc_cpu( &iwork,  liwork );

#ifdef COMPLEX
  lrwork = (magma_int_t) aux_rwork[0];
  double * rwork;
  magma_dmalloc_pinned( &rwork,    lrwork );
#endif



  magma_dsyevd_gpu( zjob, m_uplo,
                    N, dA, ldda, eigs,
                    h_R, lda,          // h_R
                    h_work, lwork,
                    #ifdef COMPLEX
                    rwork, lrwork,
                    #endif
                    iwork, liwork,
                    &info );
  if( info != 0 )
  {
    std::ostringstream msg;
    msg << "magma_dsyevd_gpu returned with info = " << info;
    ErrorHandling( msg.str().c_str() );
  }

  magma_free_pinned( h_R    );
  magma_free_pinned( h_work );
  magma_free_cpu( iwork );
#ifdef COMPLEX
  magma_free_pinned( rwork );
#endif
}
/*
*/

// *********************************************************************
// For computing the inverse of a triangular matrix
// *********************************************************************


/*
void Trtri( char uplo, char diag, Int n, const double* A, Int lda )
{
  Int info;
  MAGMA(dtrtri)( &uplo, &diag, &n, A, &lda, &info );
  if( info < 0 )
  {
    std::ostringstream msg;
    msg << "dtrtri returned with info = " << info;
    throw std::logic_error( msg.str().c_str() );
  }
  else if( info > 0 )
    throw std::runtime_error("Matrix is singular.");
}
*/
// *********************************************************************
// Copy
// *********************************************************************

void Lacpy( char uplo, Int m, Int n, const double* A, Int lda, double* B, Int ldb )
{
  //MAGMA(dlacpy)( &uplo, &m, &n, A, &lda, B, &ldb );
 /*
  magma_uplo_t magmaUplo;
  magmaDouble_const_ptr dA = (magmaDouble_const_ptr ) A;
  magmaDouble_const_ptr dB = (magmaDouble_const_ptr ) B;
  magma_queue_t queue ;
  
  magmablas_dlacpy(magmaUplo, m, n, dA, lda, dB, ldb, queue);
  */
}
#endif

} // namespace MAGMA
} // namespace dgdft
