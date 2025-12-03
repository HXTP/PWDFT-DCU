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
/// @file fourier.cpp
/// @brief Sequential and Distributed Fourier wrapper.
/// @date 2011-11-01
/// @date 2015-05-02 Add some dual grid functions
#ifndef _FOURIER_HPP_
#define _FOURIER_HPP_

#include  "environment.hpp"
#include  "domain.hpp"
#include  "numvec_impl.hpp"
#ifdef GPU
#include "cu_numvec_impl.hpp"
#include <assert.h>
#include <hipfft/hipfft.h>
#endif
namespace dgdft{

// *********************************************************************
// Sequential FFTW interface
// *********************************************************************

/// @struct Fourier
/// @brief Sequential FFTW interface.
struct Fourier {
  Domain           domain;
  bool             isInitialized;
  Int              numGridTotal;
  Int              numGridTotalFine;
  Int              numGridTotalHFX;

  // plans
  fftw_plan backwardPlan;
  fftw_plan forwardPlan;
  fftw_plan backwardPlanFine;
  fftw_plan forwardPlanFine;
  fftw_plan backwardPlanHFX;
  fftw_plan forwardPlanHFX;

  fftw_plan backwardPlanR2C;
  fftw_plan forwardPlanR2C;
  fftw_plan backwardPlanR2CFine;
  fftw_plan forwardPlanR2CFine;
  fftw_plan backwardPlanR2CHFX;
  fftw_plan forwardPlanR2CHFX;
#ifdef GPU
  hipfftHandle cuPlanR2C[NSTREAM];
  hipfftHandle cuPlanR2CFine[NSTREAM];
  hipfftHandle cuPlanC2R[NSTREAM];
  hipfftHandle cuPlanC2RFine[NSTREAM];
  hipfftHandle cuPlanC2CFine[NSTREAM];
  hipfftHandle cuPlanC2C[NSTREAM];

  hipblasHandle_t cuBlasPlanR2C[NSTREAM];
  hipblasHandle_t cuBlasPlanR2CFine[NSTREAM];
  hipblasHandle_t cuBlasPlanC2R[NSTREAM];
  hipblasHandle_t cuBlasPlanC2RFine[NSTREAM];
  hipStream_t cuStream[NSTREAM];
#endif
  unsigned  plannerFlag;

  // Laplacian operator related
  DblNumVec                gkk;
  std::vector<CpxNumVec>   ik;
  DblNumVec                TeterPrecond;

  DblNumVec                gkkFine;
  std::vector<CpxNumVec>   ikFine;

  //
  DblNumVec                gkkHFX;
  std::vector<CpxNumVec>   ikHFX;
  // FIXME Teter should be moved to Hamiltonian
  DblNumVec                TeterPrecondFine;

  // Temporary vectors that can also be used globally
  CpxNumVec                inputComplexVec;     
  CpxNumVec                outputComplexVec;     

  CpxNumVec                inputComplexVecFine;     
  CpxNumVec                outputComplexVecFine;     

  // HFX
  CpxNumVec                inputComplexVecHFX;
  CpxNumVec                outputComplexVecHFX;
  // Real data Fourier transform
  Int       numGridTotalR2C;
  Int       numGridTotalR2CFine;
  Int       numGridTotalR2CHFX;

  DblNumVec                gkkR2C;
  std::vector<CpxNumVec>   ikR2C;
  DblNumVec                TeterPrecondR2C;

  DblNumVec                gkkR2CFine;
  std::vector<CpxNumVec>   ikR2CFine;
  DblNumVec                TeterPrecondR2CFine;

  DblNumVec                gkkR2CHFX;
  std::vector<CpxNumVec>   ikR2CHFX;

  // Temporary vectors that can also be used globally
  DblNumVec                inputVecR2C;     
  CpxNumVec                outputVecR2C;     

  DblNumVec                inputVecR2CFine;     
  CpxNumVec                outputVecR2CFine;     

  DblNumVec                inputVecR2CHFX;
  CpxNumVec                outputVecR2CHFX;

  /// @brief index array for mapping a coarse grid to a fine grid
  IntNumVec                idxFineGrid;
  IntNumVec                idxFineGridR2C;

  IntNumVec                idxHFXGrid;
  IntNumVec                idxHFXGridR2C;

  Fourier();
  ~Fourier();

  void Initialize( const Domain& dm );
  void InitializeFine( const Domain& dm );
  /// HFX 
  void InitializeHFX( const Domain& dm );

};
#ifdef GPU
void cuFFTExecuteInverse( Fourier& fft, hipfftHandle &plan, int fft_type, cuCpxNumVec &cu_psi_in, cuCpxNumVec &cu_psi_out );
void cuFFTExecuteInverse( Fourier& fft, hipfftHandle &plan, int fft_type, cuCpxNumVec &cu_psi_in, cuCpxNumVec &cu_psi_out , int nbands);
void cuFFTExecuteInverse2( Fourier& fft, hipfftHandle &plan, int fft_type, cuCpxNumVec &cu_psi_in, cuCpxNumVec &cu_psi_out );
void cuFFTExecuteForward( Fourier& fft, hipfftHandle &plan, int fft_type, cuCpxNumVec &cu_psi_in, cuCpxNumVec &cu_psi_out );
void cuFFTExecuteForward2( Fourier& fft, hipfftHandle &plan, int fft_type, cuCpxNumVec &cu_psi_in, cuCpxNumVec &cu_psi_out );
void cuFFTExecuteInverse( Fourier& fft, hipfftHandle &plan, int fft_type, cuDblNumVec &cu_psi_in, cuDblNumVec &cu_psi_out );
void cuFFTExecuteForward( Fourier& fft, hipfftHandle &plan, int fft_type, cuDblNumVec &cu_psi_in, cuDblNumVec &cu_psi_out );

void cuFFTExecuteInverse_stream( Fourier& fft, hipfftHandle &plan, hipblasHandle_t  &planBlas, int fft_type, cuDblNumVec &cu_psi_in, cuDblNumVec &cu_psi_out );
void cuFFTExecuteForward_stream( Fourier& fft, hipfftHandle &plan, hipblasHandle_t  &planBlas, int fft_type, cuDblNumVec &cu_psi_in, cuDblNumVec &cu_psi_out );

#endif
void FFTWExecute( Fourier& fft, fftw_plan& plan );

// *********************************************************************
// Parallel FFTW interface
// *********************************************************************

/// @struct DistFourier
/// @brief Distributed memory (MPI only) parallel FFTW interface.
struct DistFourier {
  Domain           domain;                      
  bool             isInitialized;
  Int              numGridTotal;
  Int              numGridLocal;    
  ptrdiff_t        localNz;
  ptrdiff_t        localNzStart;
  // numAllocLocal is the size for the FFTW vectors.
  // numAllocLocal is close but may not be exactly the same as numGridLocal, which is
  // localNz * Ny * Nx.  This is because FFTW may need some intermediate
  // space.  For more information see FFTW's manual.
  ptrdiff_t        numAllocLocal;

  /// @brief Whether the processor according to the rank in domain
  /// participate in the FFTW calculation.
  bool             isInGrid;
  /// @brief The communicator used by parallel FFTW, should be
  /// consistent with inGrid.
  MPI_Comm         comm;

  // plans
  fftw_plan        backwardPlan;
  fftw_plan        forwardPlan;

  unsigned         plannerFlag;

  // Laplacian operator related
  DblNumVec                gkkLocal;
  std::vector<CpxNumVec>   ikLocal;
  DblNumVec                TeterPrecondLocal;

  // Temporary vectors that can also be used globally
  CpxNumVec                inputComplexVecLocal;     
  CpxNumVec                outputComplexVecLocal;     

  DistFourier();
  ~DistFourier();

  /// @brief Initialize the FFTW variables.
  /// 
  /// @param[in] dm Domain for the FFTW calculation. 
  /// @param[in] numProc The number of processors actually participate
  /// in the FFTW calculation.  A processor participates in the FFTW
  /// calculation if mpirank(dm.comm) < numProc.
  ///
  /// FIXME: The initialization routine ASSUMES that a fine grid is
  /// generated from the domain. This can be confusing and need to be
  /// fixed later.
  void Initialize( const Domain& dm, Int numProc );
};

} // namespace dgdft


#endif // _FOURIER_HPP_
