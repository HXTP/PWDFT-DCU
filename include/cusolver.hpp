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
/// @file device_solver.hpp
/// @brief Thin interface to device_solver
/// @date 2020-08-21

#ifdef GPU // only used for the GPU version of the PWDFT code 

#ifndef __HIP_PLATFORM_HCC__
#define __HIP_PLATFORM_HCC__
#endif

#ifndef _DEVICE_SOLVER_HPP_
#define _DEVICE_SOLVER_HPP_
#include  "environment.hpp"

#include <hip/hip_runtime.h>
#include <rocsolver/rocsolver.h>

extern rocsolver_handle cusolverH;

namespace dgdft {

/// @namespace device_solver
///
/// @brief Thin interface to device_solver
namespace cusolver{

typedef  int               Int;
//typedef  cuComplex         scomplex;
//typedef  cuDoubleComplex   dcomplex;

void Init(void);

void Destroy(void);

void Potrf( char uplo, Int n, double * A, Int lda );

void Syev( char uplo, Int n, double *A, Int lda, double *eigs);

//void Potrf( char uplo, Int n, cuDoubleComplex * A, Int lda );

//void Syevd( char jobz, char uplo, Int n, double *A, Int lda, double *eigs);

//void Syevd( char jobz, char uplo, Int n, cuDoubleComplex *A, Int lda, double *eigs);

//void Lacpy(char uplo, Int m, Int n, const double * A , Int lda, double *B, Int ldb);


} // namespace cusolver
} // namespace dgdft

#endif
#endif
