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
/// @file cu_numtns_decl.hpp
/// @brief Numerical tensor
/// @date 2016-11-19
#ifndef _CU_NUMTNS_DECL_HPP_
#define _CU_NUMTNS_DECL_HPP_

#include "environment.hpp"
#include "cu_nummat_impl.hpp"

namespace  dgdft{

// Templated form of numerical tensor
//
// The main advantage of this portable NumVec structure is that it can
// either own (owndata == true) or view (owndata == false) a piece of
// data.

template <class F>
  class cuNumTns
  {
  public:
    Int m_, n_, p_;
    bool owndata_;
    F* data_;
  public:
    cuNumTns(Int m=0, Int n=0, Int p=0);

    cuNumTns(Int m, Int n, Int p, bool owndata, F* data);

    cuNumTns(const cuNumTns& C);

    ~cuNumTns();

    cuNumTns& operator=(const cuNumTns& C);

    void Resize(Int m, Int n, Int p);

    const F& operator()(Int i, Int j, Int k) const;

    F& operator()(Int i, Int j, Int k);

    bool IsOwnData() const { return owndata_; }

    F* Data() const { return data_; }

    F* MatData (Int k) const; 

    F* VecData (Int j, Int k) const; 

    Int m() const { return m_; }

    Int n() const { return n_; }

    Int p() const { return p_; }

    Int Size() const { return m_ * n_ * p_; }
  };


// Commonly used
typedef cuNumTns<bool>       cuBolNumTns;
typedef cuNumTns<Int>        cuIntNumTns;
typedef cuNumTns<Real>       cuDblNumTns;
typedef cuNumTns<double2>    cuCpxNumTns;

// Utilities
//template <class F> inline void SetValue(cuNumTns<F>& T, F val);

//template <class F> inline Real Energy(const cuNumTns<F>& T);



} // namespace dgdft

#endif // _CU_NUMTNS_DECL_HPP_
