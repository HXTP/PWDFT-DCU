/*
   Copyright (c) 2012 The Regents of the University of California,
   through Lawrence Berkeley National Laboratory.  

Authors: Lexing Ying and Lin Lin

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
/// @file cu_numtns_impl.hpp
/// @brief Implementation of numerical tensor.
/// @date 2016-11-21
#ifndef _CU_NUMTNS_IMPL_HPP_
#define _CU_NUMTNS_IMPL_HPP_

#include  "cu_numtns_decl.hpp"

namespace  dgdft{

template <class F> 
  inline cuNumTns<F>::cuNumTns(Int m, Int n, Int p): m_(m), n_(n), p_(p), owndata_(true) {
    if(m_>0 && n_>0 && p_>0) 
      data_ = (F*)cuda_malloc( sizeof(F) * m_ * n_ *p_); 
    else 
      data_=NULL;
  }

template <class F> 
  inline cuNumTns<F>::cuNumTns(Int m, Int n, Int p, bool owndata, F* data): m_(m), n_(n), p_(p), owndata_(owndata) {
    if(owndata_) {
      if(m_>0 && n_>0 && p_>0) {
        data_ = (F*) cuda_malloc( sizeof(F) * m_ * n_ * p_);
        cuda_memcpy_GPU2GPU(data_, data, sizeof(F) * m_ * n_ * p_ );
      }
      else 
        data_=NULL;
    }
    else{ data_ = data; } 
  } 

template <class F> 
  inline cuNumTns<F>::cuNumTns(const cuNumTns& C): m_(C.m_), n_(C.n_), p_(C.p_), owndata_(C.owndata_) {
    if(owndata_) {
      if(m_>0 && n_>0 && p_>0) {
        data_ = (F*) cuda_malloc( sizeof(F) * m_ * n_ * p_);
        cuda_memcpy_GPU2GPU(data_, C.data_, sizeof(F) * m_ * n_ * p_ );
      }
      else 
        data_=NULL;
    }
    else{ data_ = C.data_; } 
  } 


template <class F> 
  inline cuNumTns<F>::~cuNumTns() { 
    if(owndata_) { 
      if(m_>0 && n_>0 && p_>0) { cuda_free(data_) ; data_ = NULL; } 
    }
  }

template <class F> 
  inline cuNumTns<F>& cuNumTns<F>::operator=(const cuNumTns& C) {
    // Do not copy if it is the same matrix.
    if(C.data_ == data_) return *this;

    if(owndata_) { 
      if(m_>0 && n_>0 && p_>0) { cuda_free(data_) ; data_ = NULL; } 
    }
    m_ = C.m_; n_=C.n_; p_=C.p_; owndata_=C.owndata_;
    if(owndata_) {
      if(m_>0 && n_>0 && p_>0) {
        data_ = (F*) cuda_malloc( sizeof(F) * m_ * n_ * p_);
        cuda_memcpy_GPU2GPU(data_, C.data_, sizeof(F) * m_ * n_ * p_ );
      }
      else data_ = NULL;
    } 
    else data_ = C.data_;
    return *this;
  }


template <class F> 
  inline void cuNumTns<F>::Resize(Int m, Int n, Int p)  {
    if( owndata_ == false ){
      ErrorHandling("Tensor being resized must own data.");
    }
    if(m_!=m || n_!=n || p_!=p) {
      if(m_>0 && n_>0 && p_>0) { cuda_free(data_); data_ = NULL; } 
      m_ = m; n_ = n; p_=p;
      if(m_>0 && n_>0 && p_>0) { data_ = (F*) cuda_malloc( sizeof(F) * m_ * n_ * p_); } else data_=NULL;
    }
  }

template <class F> 
  inline const F& cuNumTns<F>::operator()(Int i, Int j, Int k) const  {
    if( i < 0 || i >= m_ ||
        j < 0 || j >= n_ ||
        k < 0 || k >= p_ ) {
      std::ostringstream msg;
      msg 
        << "Index is out of bound."  << std::endl
        << "Index bound    ~ (" << m_ << ", " << n_ << ", " << p_ << ")" << std::endl
        << "This index     ~ (" << i  << ", " << j  << ", " << k  << ")" << std::endl;
      ErrorHandling( msg.str().c_str() );
    }
    return data_[i+j*m_+k*m_*n_];
  }

template <class F> 
  inline F& cuNumTns<F>:: operator()(Int i, Int j, Int k)  {
    if( i < 0 || i >= m_ ||
        j < 0 || j >= n_ ||
        k < 0 || k >= p_ ) {
      std::ostringstream msg;
      msg 
        << "Index is out of bound."  << std::endl
        << "Index bound    ~ (" << m_ << ", " << n_ << ", " << p_ << ")" << std::endl
        << "This index     ~ (" << i  << ", " << j  << ", " << k  << ")" << std::endl;
      ErrorHandling( msg.str().c_str() );
    }
    return data_[i+j*m_+k*m_*n_];
  }

template <class F> 
  inline F* cuNumTns<F>::MatData (Int k) const {
    if( k < 0 || k >= p_ ) {
      std::ostringstream msg;
      msg 
        << "Index is out of bound."  << std::endl
        << "Index bound    ~ (" << p_ << ")" << std::endl
        << "This index     ~ (" << k  << ")" << std::endl;
      ErrorHandling( msg.str().c_str() );
    }
    return &(data_[k*m_*n_]);
  }

template <class F> 
  inline F* cuNumTns<F>::VecData (Int j, Int k) const {
    if( j < 0 || j >= n_ ||
        k < 0 || k >= p_ ) {
      std::ostringstream msg;
      msg 
        << "Index is out of bound."  << std::endl
        << "Index bound    ~ (" << n_ << ", " << p_ << ")" << std::endl
        << "This index     ~ (" << j  << ", " << k  << ")" << std::endl;
      ErrorHandling( msg.str().c_str() );
    }
    return &(data_[k*m_*n_+j*m_]);
  }


// *********************************************************************
// Utilities
// *********************************************************************
/*
template <class F> inline void SetValue(cuNumTns<F>& T, F val)
{
  F *ptr = T.data_;
  for(Int i=0; i < T.m() * T.n() * T.p(); i++) *(ptr++) = val; 

  return;
}

template <class F> inline Real Energy(const cuNumTns<F>& T)
{
  Real sum = 0;

  F *ptr = T.Data();
  for(Int i=0; i < T.m() * T.n() * T.p(); i++) 
    sum += std::abs(ptr[i]) * std::abs(ptr[i]);

  return sum;
}
*/
} // namespace dgdft

#endif // _CU_NUMTNS_IMPL_HPP_
