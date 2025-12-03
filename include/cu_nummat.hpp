/*
   Copyright (c) 2012 The Regents of the University of California,
   through Lawrence Berkeley National Laboratory.  

Author: Weile Jia and Lin Lin

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
/// @file cu_nummat.hpp
/// @brief Numerical matrix.
/// @date 2016-10-24
#ifdef GPU
#ifndef _CU_NUMMAT_HPP_
#define _CU_NUMMAT_HPP_

#include "environment.hpp"
#include "nummat_decl.hpp"
#include "cuda_utils.h"

namespace  dgdft{

// Templated form of numerical matrix
//
// The main advantage of this portable cuNumMat structure is that it can
// either own (owndata == true) or view (owndata == false) a piece of
// data.

template <class F>
  class cuNumMat
  {
  public:
    Int m_, n_;
    bool owndata_;
    F* data_;

  public:

    cuNumMat(Int m=0, Int n=0);

    cuNumMat(Int m, Int n, bool owndata, F* data);

    cuNumMat(const cuNumMat& C); // the C must be a cuMat

    ~cuNumMat();

    cuNumMat& operator=(const cuNumMat& C);

    void Resize(Int m, Int n);

    const F& operator()(Int i, Int j) const;   // must be a device function, currently do not implement.

    F& operator()(Int i, Int j);   // must be device function.

    bool IsOwnData() const { return owndata_; }

    F* Data() const { return data_; }

    F* VecData(Int j)  const;  // current do no implement this func.

    Int m() const { return m_; }

    Int n() const { return n_; }

    Int Size() const { return m_ * n_; }

    void CopyTo(NumMat<F> & C); 

    void CopyFrom(const NumMat<F> &C);

    void CopyTo(cuNumMat<F> & C); 

    void CopyFrom(const cuNumMat<F> &C);
  };

// Commonly used
typedef cuNumMat<bool>     cuBolNumMat;
typedef cuNumMat<Int>      cuIntNumMat;
typedef cuNumMat<Real>     cuDblNumMat;
typedef cuNumMat<float>    cuFltNumMat;
typedef cuNumMat<double2>  cuCpxNumMat;

// Utilities
/*
template <class F> inline void SetValue(cuNumMat<F>& M, F val);
template <class F> inline Real Energy(const cuNumMat<F>& M);
template <class F> inline void Transpose ( const cuNumMat<F>& A, cuNumMat<F>& B );
template <class F> inline void Symmetrize( cuNumMat<F>& A );
*/
} // namespace dgdft

#endif // _CU_NUMMAT_HPP_
#endif // GPU

