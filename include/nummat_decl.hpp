/*
   Copyright (c) 2012 The Regents of the University of California,
   through Lawrence Berkeley National Laboratory.  

Author: Lexing Ying and Lin Lin

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
/// @file nummat_decl.hpp
/// @brief Numerical matrix.
/// @date 2010-09-27
#ifndef _NUMMAT_DECL_HPP_
#define _NUMMAT_DECL_HPP_

#include "environment.hpp"

namespace  dgdft{

// Templated form of numerical matrix
//
// The main advantage of this portable NumVec structure is that it can
// either own (owndata == true) or view (owndata == false) a piece of
// data.

template <class F>
  class NumMat
  {
  public:
    Int m_, n_;
    bool owndata_;
    F* data_;
  public:
    NumMat(Int m=0, Int n=0);

    NumMat(Int m, Int n, bool owndata, F* data);

    NumMat(const NumMat& C);

    ~NumMat();

    NumMat& operator=(const NumMat& C);

    void Resize(Int m, Int n);

    const F& operator()(Int i, Int j) const;  

    F& operator()(Int i, Int j);  

    bool IsOwnData() const { return owndata_; }

    F* Data() const { return data_; }

    F* VecData(Int j)  const; 

    Int m() const { return m_; }

    Int n() const { return n_; }

    Int Size() const { return m_ * n_; }

    void ExchangeCols(IntNumVec& colMap);

    void Rearrange(Int blockNum);

    void RevertRearrange(Int blockNum);


  };

// Commonly used
typedef NumMat<bool>     BolNumMat;
typedef NumMat<Int>      IntNumMat;
typedef NumMat<Real>     DblNumMat;
typedef NumMat<Complex>  CpxNumMat;
typedef NumMat<float>    FloNumMat;

// Utilities
template <class F> inline void SetValue(NumMat<F>& M, F val);
template <class F> inline Real Energy(const NumMat<F>& M);
template <class F> inline void Transpose ( const NumMat<F>& A, NumMat<F>& B );
template <class F> inline void Symmetrize( NumMat<F>& A );

} // namespace dgdft

#endif // _NUMMAT_DECL_HPP_
