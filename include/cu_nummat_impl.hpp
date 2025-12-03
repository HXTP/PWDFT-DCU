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
/// @file cu_nummat_impl.hpp
/// @brief Implementation of numerical matrix.
/// @date 2016-10-22
#ifdef GPU
#ifndef _CU_NUMMAT_IMPL_HPP_
#define _CU_NUMMAT_IMPL_HPP_

#include "cu_nummat.hpp"

namespace  dgdft{

template <class F> inline cuNumMat<F>::cuNumMat(Int m, Int n): m_(m), n_(n), owndata_(true) {
  if(m_>0 && n_>0) 
     data_ = (F*)cuda_malloc( sizeof(F) * m_ * n_ ); 
  else 
     data_=NULL;
}

template <class F> inline cuNumMat<F>::cuNumMat(Int m, Int n, bool owndata, F* data): m_(m), n_(n), owndata_(owndata) {
  if(owndata_) {
    if(m_>0 && n_>0) { data_ = (F*)cuda_malloc( sizeof(F) * m_ * n_ ); }  else data_=NULL;
    if(m_>0 && n_>0) { cuda_memcpy_GPU2GPU (data_, data, sizeof(F)*m_*n_); }
  } else {
    data_ = data;
  }
}

template <class F> inline cuNumMat<F>::cuNumMat(const cuNumMat& C): m_(C.m_), n_(C.n_), owndata_(C.owndata_) {
  if(owndata_) {
    if(m_>0 && n_>0) { data_ = (F*)cuda_malloc( sizeof(F) * m_ * n_ ); } else data_=NULL;
    if(m_>0 && n_>0) { cuda_memcpy_GPU2GPU (data_, C.data_, sizeof(F)*m_*n_); }
  } else {
    data_ = C.data_;
  }
}

template <class F> inline cuNumMat<F>::~cuNumMat() {
  if(owndata_) {
    if(m_>0 && n_>0) { cuda_free(data_); data_ = NULL; }
  }
}

template <class F> inline cuNumMat<F>& cuNumMat<F>::operator=(const cuNumMat& C) {
  // Do not copy if it is the same matrix.
  if(C.data_ == data_) return *this;
  if(owndata_) {
    if(m_>0 && n_>0) { cuda_free(data_); data_ = NULL; }
  }
  m_ = C.m_; n_=C.n_; owndata_=C.owndata_;
  if(owndata_) {
    if(m_>0 && n_>0) { data_ = (F*)cuda_malloc( sizeof(F) * m_ * n_ ); } else data_=NULL;
    if(m_>0 && n_>0) { for(Int i=0; i<m_*n_; i++) data_[i] = C.data_[i]; }
  } else {
    data_ = C.data_;
  }
  return *this;
}

template <class F> inline void cuNumMat<F>::Resize(Int m, Int n)  {
  if( owndata_ == false ){
    ErrorHandling("Matrix being resized must own data.");
  }
  if(m_!=m || n_!=n) {
    if(m_>0 && n_>0) { cuda_free(data_); data_ = NULL; }
    m_ = m; n_ = n;
    if(m_>0 && n_>0) { data_ = (F*)cuda_malloc( sizeof(F) * m_ * n_ ); } else data_=NULL;
  }
}

template <class F> inline void cuNumMat<F>::CopyTo(cuNumMat<F> &C) {
  // copy from the GPU NumMat to the CPU NumMat.
  if( C.m_*C.n_ < m_ * n_) 
  { 
    C.Resize(m_, n_);
  }
  if(C.m_*C.n_ >= m_*n_) {
    if(m_>0 && n_>0) { cuda_memcpy_GPU2GPU(C.data_, data_, sizeof(F)*m_*n_);}
  }
}

template <class F> inline void cuNumMat<F>::CopyTo(NumMat<F> &C) {
  // copy from the GPU NumMat to the CPU NumMat.
  if( C.m_*C.n_ < m_ * n_) 
  { 
    C.Resize(m_, n_);
  }
  if(C.m_*C.n_ >= m_*n_) {
    if(m_>0 && n_>0) { cuda_memcpy_GPU2CPU(C.data_, data_, sizeof(F)*m_*n_);}
  }
}

template <class F> inline void cuNumMat<F>::CopyFrom(const cuNumMat<F> &C) {
  // copy from the GPU NumMat to the CPU NumMat.
  if( C.m_*C.n_ > m_ * n_) 
  { 
    std:: cout << " GPU memory not big enough. " << m_*n_ <<" "<< C.m_ * C.n_ << std:: endl;
    cuda_free(data_);
    m_ = C.m_; n_=C.n_; 
    if(m_>0 && n_>0) { data_ = (F*)cuda_malloc( sizeof(F) * m_ * n_ ); } else data_=NULL;
   }
  if(C.m_*C.n_ <= m_*n_) {
    //std::cout << " m n: "<< m_ <<" " <<n_ << std::endl;
    //std::flush(std::cout);
    if(m_>0 && n_>0) { cuda_memcpy_GPU2GPU(data_, C.data_, sizeof(F)*C.m_*C.n_);}
  }
}

template <class F> inline void cuNumMat<F>::CopyFrom(const NumMat<F> &C) {
  // copy from the GPU NumMat to the CPU NumMat.
  if( C.m_*C.n_ > m_ * n_) 
  { 
    std:: cout << " GPU memory not big enough. " << m_*n_ <<" "<< C.m_ * C.n_ << std:: endl;
    cuda_free(data_);
    m_ = C.m_; n_=C.n_; 
    if(m_>0 && n_>0) { data_ = (F*)cuda_malloc( sizeof(F) * m_ * n_ ); } else data_=NULL;
   }
  if(C.m_*C.n_ <= m_*n_) {
    //std::cout << " m n: "<< m_ <<" " <<n_ << std::endl;
    //std::flush(std::cout);
    if(m_>0 && n_>0) { cuda_memcpy_CPU2GPU(data_, C.data_, sizeof(F)*C.m_*C.n_);}
  }
}



template <class F> 
inline const F& cuNumMat<F>::operator()(Int i, Int j) const  { 
  if( i < 0 || i >= m_ ||
      j < 0 || j >= n_ ) {
    std::ostringstream msg;
    msg 
      << "Index is out of bound."  << std::endl
      << "Index bound    ~ (" << m_ << ", " << n_ << ")" << std::endl
      << "This index     ~ (" << i  << ", " << j  << ")" << std::endl;
    ErrorHandling( msg.str().c_str() ); 
  }
  return data_[i+j*m_];
}
template <class F>
inline F& cuNumMat<F>::operator()(Int i, Int j)  { 
  if( i < 0 || i >= m_ ||
      j < 0 || j >= n_ ) {
    std::ostringstream msg;
    msg 
      << "Index is out of bound."  << std::endl
      << "Index bound    ~ (" << m_ << ", " << n_ << ")" << std::endl
      << "This index     ~ (" << i  << ", " << j  << ")" << std::endl;
    ErrorHandling( msg.str().c_str() ); 
  }
  return data_[i+j*m_];
}

template <class F>
inline F* cuNumMat<F>::VecData(Int j)  const 
{ 
  if( j < 0 || j >= n_ ) {
    std::ostringstream msg;
    msg 
      << "Index is out of bound."  << std::endl
      << "Index bound    ~ (" << n_ << ")" << std::endl
      << "This index     ~ (" << j  << ")" << std::endl;
    ErrorHandling( msg.str().c_str() ); 
  }
  return &(data_[j*m_]); 
}


// *********************************************************************
// Utilities
// *********************************************************************
#if 0
template <class F> inline void SetValue(cuNumMat<F>& M, F val)
{
  F *ptr = M.data_;
  for (Int i=0; i < M.m()*M.n(); i++) *(ptr++) = val;
}

template <class F> inline Real Energy(const cuNumMat<F>& M)
{
  Real sum = 0;
  F *ptr = M.data_;
  for (Int i=0; i < M.m()*M.n(); i++) 
    sum += std::abs(ptr[i]) * std::abs(ptr[i]);
  return sum;
}


template <class F> inline void
Transpose ( const cuNumMat<F>& A, cuNumMat<F>& B )
{
  if( A.m() != B.n() || A.n() != B.m() ){
    B.Resize( A.n(), A.m() );
  }

  F* Adata = A.Data();
  F* Bdata = B.Data();
  Int m = A.m(), n = A.n();

  for( Int i = 0; i < m; i++ ){
    for( Int j = 0; j < n; j++ ){
      Bdata[ j + n*i ] = Adata[ i + j*m ];
    }
  }


  return ;
}        // -----  end of function Transpose  ----- 

template <class F> inline void
Symmetrize( cuNumMat<F>& A )
{
  if( A.m() != A.n() ){
    ErrorHandling( "The matrix to be symmetrized should be a square matrix." );
  }

  cuNumMat<F> B;
  Transpose( A, B );

  F* Adata = A.Data();
  F* Bdata = B.Data();

  F  half = (F) 0.5;

  for( Int i = 0; i < A.m() * A.n(); i++ ){
    *Adata = half * (*Adata + *Bdata);
    Adata++; Bdata++;
  }


  return ;
}        // -----  end of function Symmetrize ----- 
#endif

} // namespace dgdft

#endif // _CU_NUMMAT_IMPL_HPP_
#endif // GPU
