/*
   Copyright (c) 2012 The Regents of the University of California,
   through Lawrence Berkeley National Laboratory.  

Author: Lin Lin, Wei Hu, Amartya Banerjee, Weile Jia

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
/// @file eigensolver.hpp
/// @brief Eigensolver in the global domain or extended element.
/// @date 2012-11-20
#ifndef _EIGENSOLVER_HPP_
#define _EIGENSOLVER_HPP_

#include  "environment.hpp"
#include  "numvec_impl.hpp"
#include  "domain.hpp"
#include  "fourier.hpp"
#include  "hamiltonian.hpp"
#include  "spinor.hpp"
#include  "esdf.hpp"

namespace dgdft{

class EigenSolver
{
private:

  Hamiltonian*        hamPtr_;
  Fourier*            fftPtr_;
  Spinor*             psiPtr_;


  DblNumVec           eigVal_;
  DblNumVec           resVal_;


  // Routines for Chebyshev Filtering
  double Cheby_Upper_bound_estimator(DblNumVec& ritz_values, int Num_Lanczos_Steps);
  void Chebyshev_filter(int m, double a, double b);
  void Chebyshev_filter_scaled(int m, double a, double b, double a_L);


  // ScaLAPACK parameters
  Int           scaBlockSize_;
  Int           numProcScaLAPACK_;
  Int           contxt_;
  Int           nprow_, npcol_;



  Int PWDFT_PPCG_use_scala_ ;
  Int PWDFT_Cheby_use_scala_ ;

public:

  // ********************  LIFECYCLE   *******************************

  EigenSolver ();

  ~EigenSolver();

  // ********************  OPERATORS   *******************************

  void Setup(
      Hamiltonian& ham,
      Spinor& psi,
      Fourier& fft );

  // ********************  OPERATIONS  *******************************


#ifdef _COMPLEX_
  void PPCGSolveComplex(
      Int          numEig,
      Int          scfIter,
      Int          eigMaxIter,
      Real         eigMinTolerance,
      Real         eigTolerance );
#else

  /// @brief Parallel LOBPCG solver with intra-element
  /// parallelization.
  ///
  /// The dense eigenvalue problem can be solved with LAPACK or
  /// ScaLAPACK depending on PWSolver
  ///
  /// @param[in] numEig  Number of eigenvalues to be counted in the
  /// convergence criterion.  numEig must be less than or equal to
  /// width.
  /// @param[in] eigMaxIter    Maximum number of iterations
  /// @param[in] eigMinTolerance Minimum tolerance must be reached
  /// during the LOBPCG iteration
  /// @param[in] eigTolerance  Residual tolerance.
  void LOBPCGSolveReal(
      Int          numEig,
      Int          scfIter,
      Int          eigMaxIter,
      Real         eigMinTolerance,
      Real         eigTolerance );

#endif

  /// @brief Routines for Chebyshev filtering
  void FirstChebyStep(
      Int      numEig,
      Int      eigMaxIter,
      Int        filter_order );

  void GeneralChebyStep(
      Int      numEig,
      Int        filter_order );


  /// @brief Parallel PPCG solver
  /// by ScaLAPACK.
#ifdef GPU
  void PPCGSolveReal_GPU(
      Int          numEig,
      Int          scfIter,
      Int          eigMaxIter,
      Real         eigMinTolerance,
      Real         eigTolerance );

  void PPCGSolveReal_GPU(
      Int          numEig,
      Int          scfIter, 
      Int          eigMaxIter,
      Real         eigMinTolerance,
      Real         eigTolerance,
      bool         isSerial );
#endif

  void PPCGSolveReal(
      Int          numEig,
      Int          scfIter,
      Int          eigMaxIter,
      Real         eigMinTolerance,
      Real         eigTolerance );

  void PPCGSolveReal(
      Int          numEig,
      Int          scfIter,
      Int          eigMaxIter,
      Real         eigMinTolerance,
      Real         eigTolerance,
      bool         isSerial );



  // ********************  ACCESS      *******************************
  DblNumVec& EigVal() { return eigVal_; }
  DblNumVec& ResVal() { return resVal_; }


  Hamiltonian& Ham()  {return *hamPtr_;}
  Spinor&      Psi()  {return *psiPtr_;}
  Fourier&     FFT()  {return *fftPtr_;}

}; // -----  end of class  EigenSolver  ----- 

} // namespace dgdft
#endif // _EIGENSOLVER_HPP_
