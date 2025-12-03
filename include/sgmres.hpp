/*
   Copyright (c) 2012 The Regents of the University of California,
   through Lawrence Berkeley National Laboratory.  

Author: Dong An, Weile Jia, Lin Lin

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
/// @file sgmres.hpp
/// @brief sgmres in the global domain or extended element.
/// @date 2017-10-17
#ifndef _SGMRES_HPP_
#define _SGMRES_HPP_

#include  "environment.hpp"
#include  "numvec_impl.hpp"
#include  "domain.hpp"
#include  "fourier.hpp"
#include  "hamiltonian.hpp"
#include  "spinor.hpp"
#include  "esdf.hpp"

#ifdef _COMPLEX_

namespace dgdft{

  // Solving linear system Ax = b using GMRES with preconditioning matrix M
  class Sgmres
  {
    public:

      Hamiltonian*        hamPtr_;
      Fourier*            fftPtr_;
      Spinor*             psiPtr_; // please note, the spinor here must be a 1 column psi.

      // input
      Complex* AMat_; // matrix A
      Complex* rhs_;  // vector b 
      double tol_;  // tolerance of GMRES iteration, default 1e-6
      int max_it_;  // maximum iteration of GMRES, default 20
      Complex* Precond_; // predondioning matrix, only supporting diagonal matrix whose only diagonal entries are stored
      int size_;  // dimension of the linear system

      // input & output
      Complex* xVec_; // initial value needed to be given. After GMRES, storing the solution

      // output
      double relres_;  // 2-norm of residue r = Ax - b after GMRES
      int iter_;   // total number of iterations
      bool flag_;  // ''true'' if GMRES converges, ''false'' otherwise

    public:
      Sgmres();
      ~Sgmres(){};

      void Setup(
        Hamiltonian& ham,
        Spinor& psi,
        Fourier& fft,
        int size);
 
      void Setup(
        Hamiltonian& ham,
        Spinor& psi,
        Fourier& fft,
        Complex* AMat, 
        Complex* rhs, 
        Complex* xVec, 
        Complex* Precond, 
        int size);
      
      void Setup(
        Hamiltonian& ham,
        Spinor& psi,
        Fourier& fft,
        Complex* AMat, 
        Complex* rhs, 
        Complex* xVec, 
        Complex* Precond, 
        int size, 
        double tol, 
        int max_it);

      //  operation: computing A * yVec = Vout
      //  suggest this function only used in sgmres::Solve, not manually
      void AMatdotVec(
        Complex* yVec, 
        Complex* Vout);

      void AMatdotVec(
        Complex omega,
        Complex* yVec,
        Complex* Vout);

      // operation: solving H * solu = b, where H is an upper triangular matrix
      // suggest this function only used in sgmres::Solve, not manually
      void HSqr(
          Complex* H, 
          Complex* b, 
          Complex* solu, 
          int size);

      // Main action: implementing GMRES
      void Solve( Complex * rhs, Complex *xVec, Complex omega);
  };



} // namespace dgdft



#endif


#endif
