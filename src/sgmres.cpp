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
/// @file sgmres.cpp
/// @brief sgmres in the global domain or extended element.
/// @date 2017-10-17

#include "sgmres.hpp"
#include  "utility.hpp"
#include  "blas.hpp"
#include  "lapack.hpp"
#include  "scalapack.hpp"
#include  "mpi_interf.hpp"


#ifdef _COMPLEX_
using namespace dgdft::scalapack;
using namespace dgdft::esdf;

namespace dgdft{

// Solving linear system Ax = b using GMRES with preconditioning matrix M
Sgmres::Sgmres(){
  hamPtr_ = NULL;
  psiPtr_ = NULL;
  fftPtr_ = NULL;
  AMat_ = NULL;  rhs_ = NULL;  xVec_ = NULL;  Precond_ = NULL;
  size_ = 0;  tol_ = 1;  max_it_ = 0;
  relres_ = 1.0; iter_ = 0; flag_ = true;
};

// currently, use this contructor. 
void Sgmres::Setup(
    Hamiltonian& ham,
    Spinor& psi,
    Fourier& fft,
    Int size){

  hamPtr_ = &ham;
  psiPtr_ = &psi;
  fftPtr_ = &fft;
  Precond_ = NULL;
  size_ = size;  tol_ = 1e-6;  max_it_ = 100;
  relres_ = 1.0; iter_ = 0; flag_ = true;
}


void Sgmres::Setup(
    Hamiltonian& ham,
    Spinor& psi,
    Fourier& fft,
    Complex* AMat, 
    Complex* rhs, 
    Complex* xVec, 
    Complex* Precond, 
    int size){

  hamPtr_ = &ham;
  psiPtr_ = &psi;
  fftPtr_ = &fft;
  AMat_ = AMat;  rhs_ = rhs;  xVec_ = xVec;  Precond_ = Precond;
  size_ = size;  tol_ = 1e-6;  max_it_ = 20;
  relres_ = 1.0; iter_ = 0; flag_ = true;
}


void Sgmres::Setup(
    Hamiltonian& ham,
    Spinor& psi,
    Fourier& fft,
    Complex* AMat, 
    Complex* rhs, 
    Complex* xVec, 
    Complex* Precond, 
    int size, 
    double tol, 
    int max_it){

  hamPtr_ = &ham;
  psiPtr_ = &psi;
  fftPtr_ = &fft;
  AMat_ = AMat;  rhs_ = rhs;  xVec_ = xVec;  Precond_ = Precond;
  size_ = size;  tol_ = tol;  max_it_ = max_it;
  relres_ = 1.0; iter_ = 0; flag_ = true;
}


//  operation: computing A * yVec = Vout
//  suggest this function only used in sgmres::Solve, not manually
void Sgmres::AMatdotVec(
    Complex* yVec, 
    Complex* Vout){

  int i, j;
  for(i = 0; i < size_; ++i)
    Vout[i] = 0;

  for(i = 0; i < size_; ++i)
  {
    for(j = 0; j < size_; ++j)
    {
      Vout[i] = Vout[i] + AMat_[j*size_+i]*yVec[j];
    }
  }
}
void Sgmres::AMatdotVec(
    Complex omega,
    Complex* yVec,
    Complex* Vout){

  Hamiltonian& ham = *hamPtr_;
  Fourier&     fft = *fftPtr_;
  //Spinor&      psi = *psiPtr_;
  int i, j;

  // build up a spinor. 
  Spinor psi_temp (fft.domain, 1, 1, 1, false, yVec );
  Int ntot  = fft.domain.NumGridTotal();
  NumTns<Complex> tnsTemp(ntot, 1, 1, false, Vout);

  // Vout <-- H * yVec
  ham.MultSpinor( psi_temp, tnsTemp, fft );

  // Vout <-- H*yVec - omega * yVec
  for(i = 0; i < size_; ++i)
    Vout[i] = Vout[i] - omega * yVec [i];
}

// operation: solving H * solu = b, where H is an upper triangular matrix
// suggest this function only used in sgmres::Solve, not manually
void Sgmres::HSqr(
    Complex* H, 
    Complex* b, 
    Complex* solu, 
    int size){

  int i, j; 
  for(i = 0; i < size; ++i)
    solu[i] = b[i];

  solu[size-1] = solu[size-1] / H[size-1 + (size-1)*size];
  for (i = size-2; i >=0; --i)
  {
    for (j = i+1; j < size; ++j)
      solu[i] = solu[i] - H[i+j*size]*solu[j];
    solu[i] = solu[i] / H[i+i*size];
  }


}

// Main action: implementing GMRES
void Sgmres::Solve( Complex * rhs, Complex* xVec, Complex omega) {

  Hamiltonian& ham = *hamPtr_;
  Fourier&     fft = *fftPtr_;
  Spinor&      psi = *psiPtr_;
  rhs_ = rhs; // point to the same address
  xVec_ = xVec; // point to the same address

  // part of Setup here.
  tol_ = esdfParam.TDDFTKrylovTol;
  max_it_ = esdfParam.TDDFTKrylovMax;
  relres_ = 1.0; iter_ = 0; flag_ = true;

  // get the precondition matrix
  Precond_ = new Complex[size_];

  for( int i =0; i < size_; i++)
    Precond_[i] = fft.gkk[i] - omega;

  int iter = 0;
  bool flag = true;
  double bnrm2, nrm2; 
  int i, j, k;
  double relres;

  int n = size_;
  int m = max_it_;
  Complex temp;
  Complex ar, br, cr, sr;

  for(i = 0, bnrm2 = 0; i < size_; ++i)
    bnrm2 = bnrm2 + std::norm(rhs_[i]);
  bnrm2 = sqrt(bnrm2);

  if (bnrm2 == 0)
  {
    for(i = 0; i < size_; ++i)
      xVec_[i] = 0;
    bnrm2 = 1.0;
  }


  MPI_Comm mpi_comm = fftPtr_->domain.comm;
  int mpirank;  MPI_Comm_rank(mpi_comm, &mpirank);
  int mpisize;  MPI_Comm_size(mpi_comm, &mpisize);

  double * relres_list = new double[mpisize];

  Complex* r = new Complex[size_];
  // I think the preconditioner only works in the G-space, 
  // In the current code, what we have is in the Real Space. 
  // I do not assume they are the same, same thing in the next part 
  // of calling the Precond_
  // CHECK CHECK
  AMatdotVec(omega, xVec_, r);
  for(i = 0; i < size_; ++i)
    r[i] = (rhs_[i] - r[i]);

  /// Preconditioning here.
  {
    // do the FFT here. 
    blas::Copy( size_, r, 1, fft.inputComplexVec.Data(), 1 );
    // G-space r is in the fft.outputComplexVec.Data()
    fftw_execute( fft.forwardPlan );
    Complex * tempPtr = fft.outputComplexVec.Data();
    for(i = 0; i < size_; ++i)
      tempPtr[i] = tempPtr[i] / Precond_[i];
    fftw_execute( fft.backwardPlan );
    blas::Copy( size_,  fft.inputComplexVec.Data(), 1, r, 1 );
    blas::Scal( size_,  1.0/Real(size_), r, 1 );
    //blas::Axpy( size_, 1.0 / Real(size_), fft.inputComplexVec.Data(), 1, r, 1 );
  }

  /*
     for(i = 0; i < size_; ++i)
     r[i] = (rhs_[i] - r[i]) / Precond_[i];
   */

  for(i = 0, relres = 0; i < size_; ++i)
    relres = relres + std::norm(r[i]);
  relres = sqrt(relres) / bnrm2;

  {
    MPI_Allgather(&relres, 1, MPI_DOUBLE, relres_list, 1, MPI_DOUBLE, mpi_comm);

    bool stopFlag = true;
    for( int ii = 0; ii < mpisize ; ii++){
      if ( relres_list[ii] > tol_ )
        stopFlag = false;
    }

    if (stopFlag)
    {
      flag_ = true; iter_ = 0;
      delete r; delete relres_list;
      return;
    }
  }

  Complex* V = new Complex[n*(m+1)];
  Complex* H = new Complex[(m+1)*m];
  for(i = 0; i < (m+1)*m; ++i)
    H[i] = 0;

  Complex* cs = new Complex[m];
  for(i = 0; i < m; ++i)
    cs[i] = 0;
  Complex* sn = new Complex[m];
  for(i = 0; i < m; ++i)
    sn[i] = 0;

  Complex* s = new Complex[m+1];
  for(i = 0; i < m+1; ++i)
    s[i] = 0;
  s[0] = relres * bnrm2; 

  for(i = 0; i < n; ++i)
    V[i] = r[i] / s[0];

  Complex* w = new Complex[n];

  for(i = 0; i < m; ++i)
  {

    AMatdotVec(omega, V+i*n, w);

    /// Preconditioning here.
    {
      blas::Copy( size_, w, 1, fft.inputComplexVec.Data(), 1 );
      fftw_execute( fft.forwardPlan );
      Complex * tempPtr = fft.outputComplexVec.Data();
      for(j = 0; j < size_; ++j)
        tempPtr[j] = tempPtr[j] / Precond_[j];
      fftw_execute( fft.backwardPlan );
      //blas::Axpy( size_, 1.0 / Real(size_), fft.inputComplexVec.Data(), 1, w, 1 );
      blas::Copy( size_,  fft.inputComplexVec.Data(), 1, w, 1 );
      blas::Scal( size_,  1.0/Real(size_), w, 1 );
    }

    /*
       for(j = 0; j < n; ++j)
       w[j] = w[j] / Precond_[j];
     */
    for(k = 0; k <= i; ++k)
    {
      for(j = 0, H[k+i*(m+1)] = 0; j < n; ++j)
        H[k+i*(m+1)] = H[k+i*(m+1)] + std::conj(w[j])*V[j+k*n];
      for(j = 0; j < n; ++j)
        w[j] = w[j] - H[k+i*(m+1)] * V[j+k*n];
    }

    for(j = 0, nrm2 = 0; j < n; ++j)
      nrm2 = nrm2 + std::norm(w[j]);
    H[i+1+i*(m+1)] = sqrt(nrm2);

    for(j = 0; j < n; ++j)
      V[j+(i+1)*n] = w[j] / H[i+1+i*(m+1)];

    for(k = 0; k <= i-1; ++k)
    {
      temp = cs[k]*H[k+i*(m+1)] + sn[k]*H[k+1+i*(m+1)];
      H[k+1+i*(m+1)] = -sn[k]*H[k+i*(m+1)] + cs[k]*H[k+1+i*(m+1)];
      H[k+i*(m+1)] = temp;
    }

    // find rotmat
    ar = H[i+i*(m+1)];  br = H[i+1+i*(m+1)];
    if (std::norm(br) == 0.0)
    {
      cr = 1.0;  sr = 0.0;
    }
    else if (std::abs(br) > std::abs(ar))
    {
      temp = ar / br;
      sr = 1.0 / std::sqrt(1.0 + temp*temp);
      cr = temp * sr;
    }
    else
    {
      temp = br / ar;
      cr = 1.0 / std::sqrt(1.0 + temp*temp);
      sr = temp * cr;
    }
    cs[i] = cr;  sn[i] = sr;

    // end finding rotmat
    temp   = cs[i]*s[i];
    s[i+1] = -sn[i]*s[i];
    s[i]   = temp;
    H[i+i*(m+1)] = cs[i]*H[i+i*(m+1)] + sn[i]*H[i+1+i*(m+1)];
    H[i+1+i*(m+1)] = 0.0;

    relres  = std::abs(s[i+1]) / bnrm2;

    MPI_Allgather(&relres, 1, MPI_DOUBLE, relres_list, 1, MPI_DOUBLE, mpi_comm);

    bool stopFlag = true;
    for( int ii = 0; ii < mpisize ; ii++){
      if ( relres_list[ii] > tol_ )
        stopFlag = false;
    }

    if(stopFlag) {
      break;
    }
  }


  // bug, fixed .. 
  //iter = i + 1; 
  iter = std::min(i+1, m);

  Complex* Hp = new Complex[iter*iter];
  for(i = 0; i < iter; ++i)
  {
    for(j = 0; j < iter; ++j)
      Hp[i+j*iter] = H[i+j*(m+1)];
  }

  Complex* y = new Complex[iter];
  HSqr(Hp, s, y, iter);

  for(i = 0; i < n; ++i)
  {
    for(j = 0; j < iter; ++j)
      xVec_[i] = xVec_[i] + V[i+j*n]*y[j];
  }
  relres = std::abs(s[iter]) / bnrm2;

  if (relres > tol_)
    flag = false;

  delete r; delete V; delete H; delete cs; delete sn; delete s;
  delete w; delete Hp; delete y; delete Precond_; delete relres_list;

  relres_ = relres;
  iter_ = iter;
  flag_ = flag;
  if(flag_)  statusOFS << " GMRES used " <<  iter << " iterations, reached convergence, Residual is: " << relres<< std::endl;
  else       statusOFS << " GMRES used " <<  iter << " iterations, did not reach convergence, Residual is: "<< relres << std::endl;
}    

} // namespace dgdft

#endif




