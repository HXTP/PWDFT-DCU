/*
   Copyright (c) 2012 The Regents of the University of California,
   through Lawrence Berkeley National Laboratory.  

Author: Lin Lin

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
/// @file scalapack.cpp
/// @brief Thin interface to ScaLAPACK
/// @date 2012-06-05
#include "scalapack.hpp"

namespace dgdft {
namespace scalapack {


// *********************************************************************
// Descriptor
// *********************************************************************

void
  Descriptor::Init(Int m, Int n, Int mb,
      Int nb, Int irsrc, Int icsrc,
      Int contxt )
  {
    values_.resize(DLEN);
    Cblacs_gridinfo(contxt, &nprow_, &npcol_, &myprow_, &mypcol_);

    // Compute the leading dimension.  Use the upper bound directly,
    // which costs a bit memory but saves the coding effort for book-keeping.
    Int lld = ( ( ( (m + mb - 1 ) / mb ) + nprow_ - 1 ) / nprow_ ) * mb;

    Int info;
    SCALAPACK(descinit)(&values_[0], &m, &n, &mb, &nb, &irsrc, &icsrc,
        &contxt, &lld, &info);
    if( info )
    {
      std::ostringstream msg;
      msg << "Descriptor:: descinit returned with info = " << info;
      ErrorHandling( msg.str().c_str() );
    }


    return ;
  }         // -----  end of method Descriptor::Init  ----- 


void
  Descriptor::Init(Int m, Int n, Int mb,
      Int nb, Int irsrc, Int icsrc,
      Int contxt, Int lld )
  {
    values_.resize(DLEN);
    Cblacs_gridinfo(contxt, &nprow_, &npcol_, &myprow_, &mypcol_);

    Int info;
    SCALAPACK(descinit)(&values_[0], &m, &n, &mb, &nb, &irsrc, &icsrc,
        &contxt, &lld, &info);
    if( info )
    {
      std::ostringstream msg;
      msg << "Descriptor:: descinit returned with info = " << info;
      ErrorHandling( msg.str().c_str() );
    }


    return ;
  }         // -----  end of method Descriptor::Init  ----- 


Descriptor& Descriptor::operator =    ( const Descriptor& desc  )
{
  if( this == &desc ) return *this;
  values_ = desc.values_;
  Cblacs_gridinfo(values_[CTXT], &nprow_, &npcol_, &myprow_, &mypcol_);
  if( nprow_ != desc.nprow_ ||
      npcol_ != desc.npcol_ ||
      myprow_ != desc.myprow_ ||
      mypcol_ != desc.mypcol_ ){
    std::ostringstream msg;
    msg << "Descriptor:: the context information does not match" << std::endl;
    ErrorHandling( msg.str().c_str() );
  }

  return *this;
}         // -----  end of method Descriptor::operator=  ----- 

// *********************************************************************
// ScaLAPACK routines
// *********************************************************************


template<class F>
inline F ScaLAPACKMatrix<F>::GetLocal    ( Int iLocal, Int jLocal ) const
{
  if( iLocal < 0 || iLocal > this->LocalHeight() ||
      jLocal < 0 || jLocal > this->LocalWidth() ){
    std::ostringstream msg;
    msg << "ScaLAPACK::GetLocal index is out of range" << std::endl;
    ErrorHandling( msg.str().c_str() );
  }

  return localMatrix_[iLocal + jLocal * (this->LocalHeight())];
}         // -----  end of method ScaLAPACKMatrix::GetLocal  ----- 


template<class F>
inline void ScaLAPACKMatrix<F>::SetLocal    ( Int iLocal, Int jLocal, F val )
{
  if( iLocal < 0 || iLocal > this->LocalHeight() ||
      jLocal < 0 || jLocal > this->LocalWidth() ){
    std::ostringstream msg;
    msg << "ScaLAPACK::SetLocal index is out of range" << std::endl;
    ErrorHandling( msg.str().c_str() );
  }
  localMatrix_[iLocal+jLocal*this->LocalHeight()] = val;

  return ;
}         // -----  end of method ScaLAPACKMatrix::SetLocal  ----- 


template<class F>
inline ScaLAPACKMatrix<F>& 
ScaLAPACKMatrix<F>::operator=    ( const ScaLAPACKMatrix<F>& A )
{
  if( this == &A ) return *this;
  desc_ = A.desc_;
  localMatrix_ = A.localMatrix_;
  return *this;
}         // -----  end of method ScaLAPACKMatrix::operator=  ----- 


// huwei
//void
//Gemm(char transA, char transB, const double alpha,
//    const ScaLAPACKMatrix<double>& A, ScaLAPACKMatrix<double>& B, 
//    const double beta, ScaLAPACKMatrix<double>& C){
//  if( A.Height() != B.Height() ){
//    std::ostringstream msg;
//    msg 
//      << "Gemm:: Global matrix dimension does not match\n" 
//      << "The dimension of A is " << A.Height() << " x " << A.Width() << std::endl
//      << "The dimension of B is " << B.Height() << " x " << B.Width() << std::endl;
//    ErrorHandling( msg.str().c_str() );
//  }

//  if( A.Context() != B.Context() ){
//    std::ostringstream msg;
//    msg << "Gemm:: A and B are not sharing the same context." << std::endl; 
//    ErrorHandling( msg.str().c_str() );
//  }

//  const Int M = A.Height();
//  const Int N = A.Width();
//  const Int K = B.Width();
//  const Int contxt = A.Context();

//  SCALAPACK(pdgemm)(&transA, &transB,
//      &M, &N, &K, 
//      &alpha,
//      A.Data(), &I_ONE, &I_ONE, A.Desc().Values(), 
//      B.Data(), &I_ONE, &I_ONE, B.Desc().Values(), 
//      &beta,
//      C.Data(), &I_ONE, &I_ONE, C.Desc().Values());    
//  return;
//}   // -----  end of function Gemm  ----- 


// huwei
void
Lacpy( char uplo, 
    Int m, Int n, 
    double* A, Int ia, Int ja, Int* desca, 
    double* B, Int ib, Int jb, Int* descb){


  SCALAPACK(pdlacpy)( &uplo,
      &m, &n,
      A, &ia, &ja, desca, 
      B, &ib, &jb, descb );

  return;
}   // -----  end of function Lacpy  ----- 

// huwei
void
Gemm( char transA, char transB,
    Int m, Int n, Int k,
    double alpha,
    double* A, Int ia, Int ja, Int* desca, 
    double* B, Int ib, Int jb, Int* descb,
    double beta,
    double* C, Int ic, Int jc, Int* descc,
    Int contxt){


  SCALAPACK(pdgemm)( &transA, &transB,
      &m, &n, &k,
      &alpha,
      A, &ia, &ja, desca, 
      B, &ib, &jb, descb,
      &beta,
      C, &ic, &jc, descc);

  return;
}   // -----  end of function Gemm  ----- 

// Amartya Banerjee
void 
Syrk ( char uplo, char trans,
       Int n, int k,
       double alpha, 
       double *A, Int ia, Int ja, Int *desca,
       double beta, 
       double *C, Int ic, Int jc, Int *descc)
{
   SCALAPACK(pdsyrk)( &uplo, &trans, &n, &k,
		      &alpha, 
		      A, &ia, &ja, desca,
		      &beta, 
		      C, &ic, &jc, descc);
  
  return; 
  
}
       
void
Syr2k (char uplo, char trans,
       Int n, int k,
       double alpha, 
       double *A, Int ia, Int ja, Int *desca,
       double *B, Int ib, Int jb, Int *descb,
       double beta,
       double *C, Int ic, Int jc, Int *descc)
{
  SCALAPACK(pdsyr2k)(&uplo , &trans , 
		     &n , &k , 
		     &alpha , 
		     A , &ia , &ja , desca , 
		     B , &ib , &jb , descb , 
		     &beta , 
		     C , &ic , &jc , descc );

 
 return;    
}
       


void
Gemr2d(const ScaLAPACKMatrix<double>& A, ScaLAPACKMatrix<double>& B){
  if( A.Height() != B.Height() || A.Width() != B.Width() ){
    std::ostringstream msg;
    msg 
      << "Gemr2d:: Global matrix dimension does not match\n" 
      << "The dimension of A is " << A.Height() << " x " << A.Width() << std::endl
      << "The dimension of B is " << B.Height() << " x " << B.Width() << std::endl;
    ErrorHandling( msg.str().c_str() );
  }

  if( A.Context() != B.Context() ){
    std::ostringstream msg;
    msg << "Gemr2d:: A and B are not sharing the same context." << std::endl; 
    ErrorHandling( msg.str().c_str() );
  }

  const Int M = A.Height();
  const Int N = A.Width();
  const Int contxt = A.Context();

  SCALAPACK(pdgemr2d)(&M, &N, A.Data(), &I_ONE, &I_ONE,
      A.Desc().Values(), 
      B.Data(), &I_ONE, &I_ONE, 
      B.Desc().Values(), &contxt );    
  return;
}   // -----  end of function Gemr2d  ----- 



void
Trsm( char side, char uplo, char trans, char diag, double alpha,
    const ScaLAPACKMatrix<double>& A, ScaLAPACKMatrix<double>& B )
{
  const Int M = B.Height(); // const Int M = A.Height();
  const Int N = A.Width();

  SCALAPACK(pdtrsm)(&side, &uplo, &trans, &diag, &M, &N, &alpha,
      A.Data(), &I_ONE, &I_ONE, A.Desc().Values(), 
      B.Data(), &I_ONE, &I_ONE, B.Desc().Values());


  return ;
}        // -----  end of function Trsm  ----- 


void
Syev(char uplo, ScaLAPACKMatrix<double>& A, 
    std::vector<double>& eigs){

  if( A.Height() != A.Width() ){
    std::ostringstream msg;
    msg 
      << "Syev: A must be a square matrix. \n" 
      << "The dimension of A is " << A.Height() << " x " << A.Width() << std::endl;
    ErrorHandling( msg.str().c_str() );
  }

  char jobz = 'N';
  Int lwork = -1, info;
  std::vector<double> work(1);
  Int N = A.Height();

  eigs.resize(N);
  ScaLAPACKMatrix<double> dummyZ;

  SCALAPACK(pdsyev)(&jobz, &uplo, &N, A.Data(), &I_ONE, &I_ONE,
      A.Desc().Values(), &eigs[0], dummyZ.Data(),
      &I_ONE, &I_ONE, dummyZ.Desc().Values(), &work[0],
      &lwork, &info);
  lwork = (Int)work[0];
  work.resize(lwork);

  SCALAPACK(pdsyev)(&jobz, &uplo, &N, A.Data(), &I_ONE, &I_ONE,
      A.Desc().Values(), &eigs[0], dummyZ.Data(),
      &I_ONE, &I_ONE, dummyZ.Desc().Values(), &work[0],
      &lwork, &info);

  if( info < 0 )
  {
    std::ostringstream msg;
    msg << "pdsyev: logic error. Info = " << info << std::endl;
    ErrorHandling( msg.str().c_str() );
  }
  else if( info > 0 )
  {
    std::ostringstream msg;
    msg << "pdsyev: runtime error. Info = " << info << std::endl;
    ErrorHandling( msg.str().c_str() );
  }
  return;
}   // -----  end of function Syev ----- 



void
Syev(char uplo, ScaLAPACKMatrix<double>& A, 
    std::vector<double>& eigs,
    ScaLAPACKMatrix<double>& Z){
  if( A.Height() != A.Width() ){
    std::ostringstream msg;
    msg 
      << "Syev: A must be a square matrix. \n" 
      << "The dimension of A is " << A.Height() << " x " << A.Width() << std::endl;
    ErrorHandling( msg.str().c_str() );
  }

  char jobz = 'V';
  Int lwork = -1, info;
  std::vector<double> work(1);
  Int N = A.Height();

  eigs.resize(N);
  Z.SetDescriptor(A.Desc());

  SCALAPACK(pdsyev)(&jobz, &uplo, &N, A.Data(), &I_ONE, &I_ONE,
      A.Desc().Values(), &eigs[0], Z.Data(),
      &I_ONE, &I_ONE, Z.Desc().Values(), &work[0],
      &lwork, &info);
  lwork = (Int)work[0];
  work.resize(lwork);

  SCALAPACK(pdsyev)(&jobz, &uplo, &N, A.Data(), &I_ONE, &I_ONE,
      A.Desc().Values(), &eigs[0], Z.Data(),
      &I_ONE, &I_ONE, Z.Desc().Values(), &work[0],
      &lwork, &info);

  if( info < 0 )
  {
    std::ostringstream msg;
    msg << "pdsyev: logic error. Info = " << info << std::endl;
    ErrorHandling( msg.str().c_str() );
  }
  else if( info > 0 )
  {
    std::ostringstream msg;
    msg << "pdsyev: runtime error. Info = " << info << std::endl;
    ErrorHandling( msg.str().c_str() );
  }
  return;
}   // -----  end of function Syev ----- 

// FIXME here is memory issue in Syevd (lwork and liwork)
void
Syevd(char uplo, ScaLAPACKMatrix<double>& A, 
    std::vector<double>& eigs,
    ScaLAPACKMatrix<double>& Z){

  if( A.Height() != A.Width() ){
    std::ostringstream msg;
    msg 
      << "Syevd: A must be a square matrix. \n" 
      << "The dimension of A is " << A.Height() << " x " << A.Width() << std::endl;
    ErrorHandling( msg.str().c_str() );
  }

  char jobz = 'V';
  Int  liwork = -1, lwork = -1, info;
  std::vector<double> work(1);
  std::vector<Int>    iwork(1);
  Int N = A.Height();

  eigs.resize(N);
  Z.SetDescriptor(A.Desc());

  SCALAPACK(pdsyevd)(&jobz, &uplo, &N, A.Data(), &I_ONE, &I_ONE,
      A.Desc().Values(), &eigs[0], Z.Data(),
      &I_ONE, &I_ONE, Z.Desc().Values(), 
      &work[0], &lwork,&iwork[0], &liwork, &info);
  lwork = (Int)work[0];
  // NOTE: Buggy memory allocation in pdsyevd?
  lwork = lwork+2048;
  work.resize(lwork);
  liwork = iwork[0];
  // NOTE: Buggy memory allocation in pdsyevd?
  liwork = liwork+2048;
  iwork.resize(liwork);

  SCALAPACK(pdsyevd)(&jobz, &uplo, &N, A.Data(), &I_ONE, &I_ONE,
      A.Desc().Values(), &eigs[0], Z.Data(),
      &I_ONE, &I_ONE, Z.Desc().Values(), 
      &work[0], &lwork,&iwork[0], &liwork, &info);

  if( info < 0 )
  {
    std::ostringstream msg;
    msg << "pdsyevd: logic error. Info = " << info << std::endl;
    ErrorHandling( msg.str().c_str() );
  }
  else if( info > 0 )
  {
    std::ostringstream msg;
    msg << "pdsyevd: runtime error. Info = " << info << std::endl;
    ErrorHandling( msg.str().c_str() );
  }
  return;
}   // -----  end of function Syevd ----- 

void
Syevd(char uplo, ScaLAPACKMatrix<dcomplex>& A, 
    std::vector<double>& eigs,
    ScaLAPACKMatrix<dcomplex>& Z){

  if( A.Height() != A.Width() ){
    std::ostringstream msg;
    msg 
      << "Syevd: A must be a square matrix. \n" 
      << "The dimension of A is " << A.Height() << " x " << A.Width() << std::endl;
    ErrorHandling( msg.str().c_str() );
  }

  char jobz = 'V';
  Int  liwork = -1, lwork = -1, lrwork = -1, info;
  std::vector<dcomplex> work(1);
  std::vector<Int>    iwork(1);
  std::vector<double> rwork(1);
  Int N = A.Height();

  eigs.resize(N);
  Z.SetDescriptor(A.Desc());

  SCALAPACK(pzheevd)(&jobz, &uplo, &N, A.Data(), &I_ONE, &I_ONE,
      A.Desc().Values(), &eigs[0], Z.Data(),
      &I_ONE, &I_ONE, Z.Desc().Values(), 
      &work[0], &lwork,&rwork[0], &lrwork, &iwork[0], &liwork, &info);
  lwork = (Int)work[0].real();
  // NOTE: Buggy memory allocation in pdsyevd?
  lwork = lwork+2048;
  work.resize(lwork);
  liwork = iwork[0];
  // NOTE: Buggy memory allocation in pdsyevd?
  liwork = liwork+2048;
  iwork.resize(liwork);
  lrwork = (Int)rwork[0];
  rwork.resize(lrwork);

  SCALAPACK(pzheevd)(&jobz, &uplo, &N, A.Data(), &I_ONE, &I_ONE,
      A.Desc().Values(), &eigs[0], Z.Data(),
      &I_ONE, &I_ONE, Z.Desc().Values(), 
      &work[0], &lwork, &rwork[0], &lrwork, &iwork[0], &liwork, &info);

  if( info < 0 )
  {
    std::ostringstream msg;
    msg << "pzheevd: logic error. Info = " << info << std::endl;
    ErrorHandling( msg.str().c_str() );
  }
  else if( info > 0 )
  {
    std::ostringstream msg;
    msg << "pzheevd: runtime error. Info = " << info << std::endl;
    ErrorHandling( msg.str().c_str() );
  }
  return;
}   // -----  end of function Syevd ----- 


void
Syevr(char uplo, ScaLAPACKMatrix<double>& A, 
    std::vector<double>& eigs,
    ScaLAPACKMatrix<double>& Z){

  if( A.Height() != A.Width() ){
    std::ostringstream msg;
    msg 
      << "Syevr: A must be a square matrix. \n" 
      << "The dimension of A is " << A.Height() << " x " << A.Width() << std::endl;
    ErrorHandling( msg.str().c_str() );
  }

  char jobz = 'V';
  char range = 'A'; // Compute all eigenvalues

  Int  liwork = -1, lwork = -1, info;
  std::vector<double> work(1);
  std::vector<Int>    iwork(1);
  Int N = A.Height();

  eigs.resize(N);
  Z.SetDescriptor(A.Desc());
  double dummyV = 0.0;
  Int dummyI = 0;
  Int numEigValueFound, numEigVectorFound;


  SCALAPACK(pdsyevr)(&jobz, &range, &uplo, &N, A.Data(), &I_ONE, &I_ONE,
      A.Desc().Values(), &dummyV, &dummyV, 
      &dummyI, &dummyI, &numEigValueFound, &numEigVectorFound,
      &eigs[0], Z.Data(), &I_ONE, &I_ONE, Z.Desc().Values(), 
      &work[0], &lwork, &iwork[0], &liwork, &info);
  lwork = (Int)work[0];
  work.resize(lwork);
  liwork = iwork[0];
  iwork.resize(liwork);

  SCALAPACK(pdsyevr)(&jobz, &range, &uplo, &N, A.Data(), &I_ONE, &I_ONE,
      A.Desc().Values(), &dummyV, &dummyV, 
      &dummyI, &dummyI, &numEigValueFound, &numEigVectorFound,
      &eigs[0], Z.Data(), &I_ONE, &I_ONE, Z.Desc().Values(), 
      &work[0], &lwork, &iwork[0], &liwork, &info);

  if( info < 0 )
  {
    std::ostringstream msg;
    msg << "pdsyevr: logic error. Info = " << info << std::endl;
    ErrorHandling( msg.str().c_str() );
  }
  else if( info > 0 )
  {
    std::ostringstream msg;
    msg << "pdsyevr: runtime error. Info = " << info << std::endl;
    ErrorHandling( msg.str().c_str() );
  }

  if( numEigValueFound != N ){
    std::ostringstream msg;
    msg 
      << "pdsyevr: Not all eigenvalues are found.\n " 
      << "Found " << numEigValueFound << " eigenvalues, " << 
      N << " eigenvalues in total." << std::endl;
    ErrorHandling( msg.str().c_str() );
  }
  if( numEigVectorFound != N ){
    std::ostringstream msg;
    msg 
      << "pdsyevr: Not all eigenvectors are found.\n " 
      << "Found " << numEigVectorFound << " eigenvectors, " << 
      N << " eigenvectors in total." << std::endl;
    ErrorHandling( msg.str().c_str() );
  }
  return;
}   // -----  end of function Syevr ----- 

void
Syevr(char uplo, ScaLAPACKMatrix<double>& A, 
    std::vector<double>& eigs,
    ScaLAPACKMatrix<double>& Z,
    Int il,
    Int iu){

  if( A.Height() != A.Width() ){
    std::ostringstream msg;
    msg 
      << "Syevr: A must be a square matrix. \n" 
      << "The dimension of A is " << A.Height() << " x " << A.Width() << std::endl;
    ErrorHandling( msg.str().c_str() );
  }

  char jobz = 'V';
  char range = 'I'; // Compute selected range of eigenvalues

  Int  liwork = -1, lwork = -1, info;
  std::vector<double> work(1);
  std::vector<Int>    iwork(1);
  Int N = A.Height();
  Int numEigValue = std::min( N, iu - il + 1 );

  eigs.resize( N );
  Z.SetDescriptor(A.Desc());
  double dummyV = 0.0;
  Int dummyI = 0;
  Int numEigValueFound, numEigVectorFound;


  SCALAPACK(pdsyevr)(&jobz, &range, &uplo, &N, A.Data(), &I_ONE, &I_ONE,
      A.Desc().Values(), &dummyV, &dummyV, 
      &il, &iu, &numEigValueFound, &numEigVectorFound,
      &eigs[0], Z.Data(), &I_ONE, &I_ONE, Z.Desc().Values(), 
      &work[0], &lwork,&iwork[0], &liwork, &info);
  lwork = (Int)work[0];
  work.resize(lwork);
  liwork = iwork[0];
  iwork.resize(liwork);

  SCALAPACK(pdsyevr)(&jobz, &range, &uplo, &N, A.Data(), &I_ONE, &I_ONE,
      A.Desc().Values(), &dummyV, &dummyV, 
      &il, &iu, &numEigValueFound, &numEigVectorFound,
      &eigs[0], Z.Data(), &I_ONE, &I_ONE, Z.Desc().Values(), 
      &work[0], &lwork,&iwork[0], &liwork, &info);

  if( info < 0 )
  {
    std::ostringstream msg;
    msg << "pdsyevr: logic error. Info = " << info << std::endl;
    ErrorHandling( msg.str().c_str() );
  }
  else if( info > 0 )
  {
    std::ostringstream msg;
    msg << "pdsyevr: runtime error. Info = " << info << std::endl;
    ErrorHandling( msg.str().c_str() );
  }

  if( numEigValueFound != numEigValue ){
    std::ostringstream msg;
    msg 
      << "pdsyevr: Not all eigenvalues are found.\n " 
      << "Found " << numEigValueFound << " eigenvalues, " << 
      N << " eigenvalues in total." << std::endl;
    ErrorHandling( msg.str().c_str() );
  }
  if( numEigVectorFound != numEigValue ){
    std::ostringstream msg;
    msg 
      << "pdsyevr: Not all eigenvectors are found.\n " 
      << "Found " << numEigVectorFound << " eigenvectors, " << 
      N << " eigenvectors in total." << std::endl;
    ErrorHandling( msg.str().c_str() );
  }

  // Post processing of eigs by resize (not destroying the computed
  // eigenvalues) 
  eigs.resize( numEigValue );


  return;
}   // -----  end of function Syevr ----- 


void
Potrf( char uplo, ScaLAPACKMatrix<double>& A )
{
  Int info;

  Int N = A.Height();

  SCALAPACK(pdpotrf)(&uplo, &N, A.Data(), &I_ONE,
      &I_ONE, A.Desc().Values(), &info);

  if( info < 0 )
  {
    std::ostringstream msg;
    msg << "pdpotrf: logic error. Info = " << info << std::endl;
    ErrorHandling( msg.str().c_str() );
  }
  else if( info > 0 )
  {
    std::ostringstream msg;
    msg << "pdpotrf: runtime error. Info = " << info << std::endl;
    ErrorHandling( msg.str().c_str() );
  }


  return ;
}     // -----  end of function Potrf  ----- 


void 
Sygst( Int ibtype, char uplo, ScaLAPACKMatrix<double>& A,
    ScaLAPACKMatrix<double>& B )
{

  Int info;
  double scale;
  Int N = A.Height();

  if( A.Height() != B.Height() ){
    ErrorHandling("A and B are not of the same size.");
  }

  SCALAPACK(pdsygst)(&ibtype, &uplo, &N, A.Data(), 
      &I_ONE, &I_ONE, A.Desc().Values(),
      B.Data(), &I_ONE, &I_ONE, B.Desc().Values(), &scale, &info);

  if( info < 0 )
  {
    std::ostringstream msg;
    msg << "pdsygst: logic error. Info = " << info << std::endl;
    ErrorHandling( msg.str().c_str() );
  }


  return ;
}        // -----  end of function Sygst  ----- 

#ifndef _COMPLEX_
void QRCPF( Int m, Int n, double* A, Int* desca, Int* piv, double* tau) 
{
  if( m==0 || n==0 )
  {
    return;
  }

  Int lwork=-1, info;
  double dummyWork;
  int I_ONE = 1;

  SCALAPACK(pdgeqpf)(&m, &n, A, &I_ONE, &I_ONE, &desca[0],
      piv, tau, &dummyWork, &lwork, &info);

  lwork = dummyWork;
  std::vector<double> work(lwork);
  SCALAPACK(pdgeqpf)(&m, &n, A, &I_ONE, &I_ONE, &desca[0],
      piv, tau, &work[0], &lwork, &info);

  // Important: fortran index is 1-based. Change to 0-based
  for( Int i = 0; i < n; i++ ){
    piv[i]--;
  }

  if( info < 0 )
  {
    std::ostringstream msg;
    msg << "Argument " << -info << " had illegal value";
    ErrorHandling( msg.str().c_str() );
  }

  return;
}
#else
//------------add QRCPF for complex version by lijl
void QRCPF( Int m, Int n, dcomplex* A, Int* desca, Int* piv, dcomplex* tau)
{
  if( m==0 || n==0 )
  {
    return;
  }

  Int lwork=-1, lrwork=-1, info;
  std::vector<dcomplex> dummyWork(1);
  std::vector<double> dummyrWork(1);
  int I_ONE = 1;
  //dcomplex *work;
  //double *rwork;

  SCALAPACK(pzgeqpf)(&m, &n, A, &I_ONE, &I_ONE, &desca[0],
      piv, tau, &dummyWork[0], &lwork, &dummyrWork[0], &lrwork, &info);

  lwork = (int)dummyWork[0].real();
  lrwork = (int)dummyrWork[0];
  std::vector<dcomplex> work(lwork);
  std::vector<double> rwork(lrwork);
  SCALAPACK(pzgeqpf)(&m, &n, A, &I_ONE, &I_ONE, &desca[0],
      piv, tau, &work[0], &lwork, &rwork[0], &lrwork, &info);

  // Important: fortran index is 1-based. Change to 0-based
  for( Int i = 0; i < n; i++ ){
    piv[i]--;
  }

  if( info < 0 )
  {
    std::ostringstream msg;
    msg << "Argument " << -info << " had illegal value";
    ErrorHandling( msg.str().c_str() );
  }

  return;
}
#endif
//--------------------by lijl

void QRCPR( Int m, Int n, Int k, double* A, Int* desca, Int* piv, double* tau, Int nb_dist, Int nb_alg ) 
{

  int nprow, npcol, myrow, mycol;
  int ictxt = desca[1]; // context

	Cblacs_gridinfo( ictxt, &nprow, &npcol, &myrow, &mycol ); 

  if( m==0 || n==0 )
  {
    return;
  }

  int I_ONE = 1, I_ZERO = 0;

  // Assume that Omega matrix is only stored on the first matrix
  int m_omega = nb_dist + 10;
  int mp_omega = SCALAPACK(numroc)( &m_omega, &m_omega, &myrow, &I_ZERO, &nprow );
  int mq       = SCALAPACK(numroc)( &m, &m, &mycol, &I_ZERO, &npcol );
  int nq = desca[5]; // NB

  std::vector<double> OMEGA( mp_omega * mq );
  int ii = 0;
  for (int i = 0; i < mp_omega; i++) {
    for (int j = 0; j < mq; j++) {
      OMEGA[ii] = drand48() - 0.5 ;
      ii++;	
    }
  }
  std::vector<double> B( mp_omega * nq );

  int descb[9], desc_OMEGA[9];

  // Estimate lwork
  Int lwork=-1, info;
  double dummyWork;
  // Is this necessary?
  SCALAPACK(partial_pdgeqrf)( &m, &n, &k, A, &I_ONE, &I_ONE, 
      desca, tau, &dummyWork, &lwork, &info );
  lwork = (int) dummyWork;
  std::vector<double> work(lwork);

  int itemp;
  itemp = std::max( 1, mp_omega ); 
  SCALAPACK(descinit)( descb, &m_omega, &n, &m_omega, &nq, &I_ZERO, 
      &I_ZERO, &ictxt, &itemp, &info );
  itemp = std::max( 1, mp_omega ); 
  SCALAPACK(descinit)( desc_OMEGA, &m_omega, &m, &m_omega, &m, &I_ZERO, 
      &I_ZERO, &ictxt, &itemp, &info );

  std::vector<int> pivB(n);
  std::vector<double> tauB(n);
  SCALAPACK(rqrcp)( &m, &n, &k, A, desca, &m_omega, &n, &B[0], descb, 
      &OMEGA[0], desc_OMEGA, piv, tau, &nb_alg, &pivB[0], &tauB[0], 
      &work[0], &lwork );


  // Important: fortran index is 1-based. Change to 0-based
  for( Int i = 0; i < n; i++ ){
    piv[i]--;
  }

  if( info < 0 )
  {
    std::ostringstream msg;
    msg << "Argument " << -info << " had illegal value";
    ErrorHandling( msg.str().c_str() );
  }

  return;
}

} // namespace scalapack
} // namespace dgdft
