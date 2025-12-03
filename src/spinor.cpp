/*
   Copyright (c) 2012 The Regents of the University of California,
   through Lawrence Berkeley National Laboratory.  

Author: Lin Lin and Wei Hu

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
/// @file spinor.cpp
/// @brief Spinor (wavefunction) for the global domain or extended
/// element.
/// @date 2012-10-06
#include "spinor.hpp"
#include  "utility.hpp"
#include  "blas.hpp"
#include  "lapack.hpp"
#include  "scalapack.hpp"
#include  "mpi_interf.hpp"

#define _NON_LOCAL_  1
#define _LOCAL_ 1
#define _EKIN_ 1

namespace dgdft{

using namespace dgdft::scalapack;

using namespace dgdft::PseudoComponent;
//xmqin
using namespace dgdft::esdf;

Spinor::Spinor () { }         
Spinor::~Spinor    () {}

#ifdef _COMPLEX_
Spinor::Spinor ( 
    const Domain &dm, 
    const Int     numComponent,
    const Int     numStateTotal,
    Int     numStateLocal,
    const Complex  val ) {
  this->Setup( dm, numComponent, numStateTotal, numStateLocal, val );

}         // -----  end of method Spinor::Spinor  ----- 

Spinor::Spinor ( const Domain &dm, 
    const Int numComponent, 
    const Int numStateTotal,
    Int numStateLocal,
    const bool owndata, 
    Complex* data )
{
  this->Setup( dm, numComponent, numStateTotal, numStateLocal, owndata, data );

}         // -----  end of method Spinor::Spinor  ----- 

void Spinor::Setup ( 
    const Domain &dm, 
    const Int     numComponent,
    const Int     numStateTotal,
    Int     numStateLocal,
    const Complex  val ) {

  domain_       = dm;
  MPI_Barrier(domain_.comm);
  int mpirank;  MPI_Comm_rank(domain_.comm, &mpirank);
  int mpisize;  MPI_Comm_size(domain_.comm, &mpisize);

  Int blocksize;

  if ( numStateTotal <=  mpisize ) {
    blocksize = 1;
  }
  else {  // numStateTotal >  mpisize
    if ( numStateTotal % mpisize == 0 ){
      blocksize = numStateTotal / mpisize;
    }
    else {
      blocksize = ((numStateTotal - 1) / mpisize) + 1;
    }    
  }

  numStateTotal_ = numStateTotal;
  blocksize_ = blocksize;

  wavefunIdx_.Resize( numStateLocal );
  SetValue( wavefunIdx_, 0 );
  for (Int i = 0; i < numStateLocal; i++){
    wavefunIdx_[i] = i * mpisize + mpirank ;
  }

  wavefun_.Resize( dm.NumGridTotal(), numComponent, numStateLocal );
  SetValue( wavefun_, val );

}         // -----  end of method Spinor::Setup  ----- 

void Spinor::Setup ( const Domain &dm, 
    const Int numComponent, 
    const Int numStateTotal,
    Int numStateLocal,
    const bool owndata, 
    Complex* data )
{

  domain_       = dm;
  MPI_Barrier(domain_.comm);
  int mpirank;  MPI_Comm_rank(domain_.comm, &mpirank);
  int mpisize;  MPI_Comm_size(domain_.comm, &mpisize);

  wavefun_      = NumTns<Complex>( dm.NumGridTotal(), numComponent, numStateLocal,
      owndata, data );

  Int blocksize;

  if ( numStateTotal <=  mpisize ) {
    blocksize = 1;
  }
  else {  // numStateTotal >  mpisize
    if ( numStateTotal % mpisize == 0 ){
      blocksize = numStateTotal / mpisize;
    }
    else {
      blocksize = ((numStateTotal - 1) / mpisize) + 1;
    }    
  }

  numStateTotal_ = numStateTotal;
  blocksize_ = blocksize;

  wavefunIdx_.Resize( numStateLocal );
  SetValue( wavefunIdx_, 0 );
  for (Int i = 0; i < numStateLocal; i++){
    wavefunIdx_[i] = i * mpisize + mpirank ;
  }

}         // -----  end of method Spinor::Setup  ----- 

void
Spinor::Normalize    ( )
{
  Int size = wavefun_.m() * wavefun_.n();
  Int nocc = wavefun_.p();

  for (Int k=0; k<nocc; k++) {
    Complex *ptr = wavefun_.MatData(k);
    Real sum = 0.0;
    for (Int i=0; i<size; i++) {
      sum += pow(abs(*ptr++), 2.0);
    }
    sum = sqrt(sum);
    if (sum != 0.0) {
      ptr = wavefun_.MatData(k);
      for (Int i=0; i<size; i++) *(ptr++) /= sum;
    }
  }
  return ;
}         // -----  end of method Spinor::Normalize  ----- 
void
Spinor::AddTeterPrecond (Fourier* fftPtr, NumTns<Complex>& a3)
{
  Fourier& fft = *fftPtr;
  if( !fftPtr->isInitialized ){
    ErrorHandling("Fourier is not prepared.");
  }
  Int ntot = wavefun_.m();
  Int ncom = wavefun_.n();
  Int nocc = wavefun_.p();

  if( fftPtr->domain.NumGridTotal() != ntot ){
    ErrorHandling("Domain size does not match.");
  }

  Int numFFTGrid = fftPtr->numGridTotal;
  // These two are private variables in the OpenMP context

  for (Int k=0; k<nocc; k++) {
    for (Int j=0; j<ncom; j++) {
      // For c2r and r2c transforms, the default is to DESTROY the
      // input, therefore a copy of the original matrix is necessary. 
      blas::Copy( ntot, wavefun_.VecData(j,k), 1, 
          fft.inputComplexVec.Data(), 1 );

      FFTWExecute ( fft, fft.forwardPlan );

      Real* ptr1d      = fftPtr->TeterPrecond.Data();
      Complex* ptr2    = fft.outputComplexVec.Data();
      for (Int i=0; i<numFFTGrid; i++) 
        *(ptr2++) *= *(ptr1d++);

      FFTWExecute ( fft, fft.backwardPlan);

      blas::Axpy( ntot, 1.0, fft.inputComplexVec.Data(), 1, a3.VecData(j,k), 1 );
    }
  }


  return ;
}         // -----  end of method Spinor::AddTeterPrecond ----- 

void
Spinor::AddMultSpinorFine ( Fourier& fft, const DblNumVec& vtot, 
    const std::vector<PseudoPot>& pseudo, NumTns<Complex>& a3 )
{
  // Complex case -- just temporary for TDDFT

  if( !fft.isInitialized ){
    ErrorHandling("Fourier is not prepared.");
  }
  Int ntot = wavefun_.m();
  Int ncom = wavefun_.n();
  Int numStateLocal = wavefun_.p();
  Int ntotFine = domain_.NumGridTotalFine();
  Real vol = domain_.Volume();

  if( fft.domain.NumGridTotal() != ntot ){
    ErrorHandling("Domain size does not match.");
  }

  // Temporary variable for saving wavefunction on a fine grid
  CpxNumVec psiFine(ntotFine);
  CpxNumVec psiUpdateFine(ntotFine);

  for (Int k=0; k<numStateLocal; k++) {
    for (Int j=0; j<ncom; j++) {

      SetValue( psiFine, Complex(0.0,0.0) );
      SetValue( psiUpdateFine, Complex(0.0,0.0) );

      // Fourier transform
      SetValue( fft.inputComplexVec, Z_ZERO ); // no need to set to zero
      blas::Copy( ntot, wavefun_.VecData(j,k), 1, fft.inputComplexVec.Data(), 1 );

      // Fourier transform of wavefunction saved in fft.outputComplexVec
      fftw_execute( fft.forwardPlan );

      // Interpolate wavefunction from coarse to fine grid
      {
        SetValue( fft.outputComplexVecFine, Z_ZERO ); 
        Int *idxPtr = fft.idxFineGrid.Data();
        Complex *fftOutFinePtr = fft.outputComplexVecFine.Data();
        Complex *fftOutPtr = fft.outputComplexVec.Data();
        for( Int i = 0; i < ntot; i++ ){
          //fft.outputComplexVecFine(fft.idxFineGrid(i)) = fft.outputComplexVec(i);
          fftOutFinePtr[*(idxPtr++)] = *(fftOutPtr++);
        }
      }
      fftw_execute( fft.backwardPlanFine );
      Real fac = 1.0 / std::sqrt( double(domain_.NumGridTotal())  *
          double(domain_.NumGridTotalFine()) ); 
      //      for( Int i = 0; i < ntotFine; i++ ){
      //        psiFine(i) = fft.inputComplexVecFine(i).real() * fac; 
      //      }
      blas::Copy( ntotFine, fft.inputComplexVecFine.Data(),
          1, psiFine.Data(), 1 );
      blas::Scal( ntotFine, fac, psiFine.Data(), 1 );

      // Add the contribution from local pseudopotential
      //      for( Int i = 0; i < ntotFine; i++ ){
      //        psiUpdateFine(i) += psiFine(i) * vtot(i);
      //      }
      {
        Complex *psiUpdateFinePtr = psiUpdateFine.Data();
        Complex *psiFinePtr = psiFine.Data();
        Real *vtotPtr = vtot.Data();
        for( Int i = 0; i < ntotFine; i++ ){
          *(psiUpdateFinePtr++) += *(psiFinePtr++) * *(vtotPtr++);
        }
      }

      // Add the contribution from nonlocal pseudopotential
      if(1){
        Int natm = pseudo.size();
        for (Int iatm=0; iatm<natm; iatm++) {
          Int nobt = pseudo[iatm].vnlList.size();
          for (Int iobt=0; iobt<nobt; iobt++) {
            const Real       vnlwgt = pseudo[iatm].vnlList[iobt].second;
            const SparseVec &vnlvec = pseudo[iatm].vnlList[iobt].first;
            const IntNumVec &iv = vnlvec.first;
            const DblNumMat &dv = vnlvec.second;

            Complex weight = (0.0,0.0); 
            const Int    *ivptr = iv.Data();
            const Real   *dvptr = dv.VecData(VAL);
            for (Int i=0; i<iv.m(); i++) {
              weight += (*(dvptr++)) * psiFine[*(ivptr++)];
            }
            weight *= vol/Real(ntotFine)*vnlwgt;

            ivptr = iv.Data();
            dvptr = dv.VecData(VAL);
            for (Int i=0; i<iv.m(); i++) {
              psiUpdateFine[*(ivptr++)] += (*(dvptr++)) * weight;
            }
          } // for (iobt)
        } // for (iatm)
      }


      // Laplacian operator. Perform inverse Fourier transform in the end
      {
        for (Int i=0; i<ntot; i++) 
          fft.outputComplexVec(i) *= fft.gkk(i);
      }

      // Restrict psiUpdateFine from fine grid in the real space to
      // coarse grid in the Fourier space. Combine with the Laplacian contribution
      //      for( Int i = 0; i < ntotFine; i++ ){
      //        fft.inputComplexVecFine(i) = Complex( psiUpdateFine(i), 0.0 ); 
      //      }
      SetValue( fft.inputComplexVecFine, Z_ZERO );
      blas::Copy( ntotFine, psiUpdateFine.Data(), 1,
          fft.inputComplexVecFine.Data(), 1 );

      // Fine to coarse grid
      // Note the update is important since the Laplacian contribution is already taken into account.
      // The computation order is also important
      fftw_execute( fft.forwardPlanFine );
      {
        Real fac = std::sqrt(Real(ntot) / (Real(ntotFine)));
        Int* idxPtr = fft.idxFineGrid.Data();
        Complex *fftOutFinePtr = fft.outputComplexVecFine.Data();
        Complex *fftOutPtr = fft.outputComplexVec.Data();

        for( Int i = 0; i < ntot; i++ ){
          //          fft.outputComplexVec(i) += fft.outputComplexVecFine(fft.idxFineGrid(i)) * fac;
          *(fftOutPtr++) += fftOutFinePtr[*(idxPtr++)] * fac;
        }
      }

      // Inverse Fourier transform to save back to the output vector
      fftw_execute( fft.backwardPlan );

      //      Real    *ptr1 = a3.VecData(j,k);
      //      for( Int i = 0; i < ntot; i++ ){
      //        ptr1[i] += fft.inputComplexVec(i).real() / Real(ntot);
      //      }
      blas::Axpy( ntot, 1.0 / Real(ntot), 
          fft.inputComplexVec.Data(), 1, a3.VecData(j,k), 1 );
    }
  }



  return ;
}        // -----  end of method Spinor::AddMultSpinorFine  ----- 

void Spinor::AddMultSpinorEXX ( Fourier& fft, 
    const NumTns<Complex>& phi,
    const DblNumVec& exxgkkR2C,
    Real  exxFraction,
    Real  numSpin,
    const DblNumVec& occupationRate,
    NumTns<Complex>& a3 )
{
  if( !fft.isInitialized ){
    ErrorHandling("Fourier is not prepared.");
  }
  MPI_Barrier(domain_.comm);
  int mpirank;  MPI_Comm_rank(domain_.comm, &mpirank);
  int mpisize;  MPI_Comm_size(domain_.comm, &mpisize);

  Index3& numGrid = domain_.numGrid;
  Index3& numGridFine = domain_.numGridFine;

  Int ntot     = domain_.NumGridTotal();
  Int ntotFine = domain_.NumGridTotalFine();
  Int ntotR2C = fft.numGridTotalR2C;
  Int ntotR2CFine = fft.numGridTotalR2CFine;
  Int ncom = wavefun_.n();
  Int numStateLocal = wavefun_.p();
  Int numStateTotal = numStateTotal_;

  Int ncomPhi = phi.n();

  Real vol = domain_.Volume();

  if( ncomPhi != 1 || ncom != 1 ){
    ErrorHandling("Spin polarized case not implemented.");
  }

  if( fft.domain.NumGridTotal() != ntot ){
    ErrorHandling("Domain size does not match.");
  }

  // Temporary variable for saving wavefunction on a fine grid
  CpxNumVec phiTemp(ntot);

  Int numStateLocalTemp;

  MPI_Barrier(domain_.comm);

  for( Int iproc = 0; iproc < mpisize; iproc++ ){

    if( iproc == mpirank )
      numStateLocalTemp = numStateLocal;

    MPI_Bcast( &numStateLocalTemp, 1, MPI_INT, iproc, domain_.comm );

    IntNumVec wavefunIdxTemp(numStateLocalTemp);
    if( iproc == mpirank ){
      wavefunIdxTemp = wavefunIdx_;
    }

    MPI_Bcast( wavefunIdxTemp.Data(), numStateLocalTemp, MPI_INT, iproc, domain_.comm );

    // FIXME OpenMP does not work since all variables are shared
    for( Int kphi = 0; kphi < numStateLocalTemp; kphi++ ){
      for( Int jphi = 0; jphi < ncomPhi; jphi++ ){

        SetValue( phiTemp, Z_ZERO );

        if( iproc == mpirank )
        { 
          Complex* phiPtr = phi.VecData(jphi, kphi);
          for( Int ir = 0; ir < ntot; ir++ ){
            phiTemp(ir) = phiPtr[ir];
          }
        }

        MPI_Bcast( phiTemp.Data(), 2*ntot, MPI_DOUBLE, iproc, domain_.comm );

        for (Int k=0; k<numStateLocal; k++) {
          for (Int j=0; j<ncom; j++) {

            Complex* psiPtr = wavefun_.VecData(j,k);
            for( Int ir = 0; ir < ntot; ir++ ){
              fft.inputComplexVec(ir) = psiPtr[ir] * std::conj(phiTemp(ir));
            }

            FFTWExecute ( fft, fft.forwardPlan );

            // Solve the Poisson-like problem for exchange
            for( Int ig = 0; ig < ntot; ig++ ){
              fft.outputComplexVec(ig) *= exxgkkR2C(ig);
            }

            FFTWExecute ( fft, fft.backwardPlan );

            Complex* a3Ptr = a3.VecData(j,k);
            Real fac = -exxFraction * occupationRate[wavefunIdxTemp(kphi)];  
            for( Int ir = 0; ir < ntot; ir++ ) {
              a3Ptr[ir] += fft.inputComplexVec(ir) * phiTemp(ir) * fac;
            }

          } // for (j)
        } // for (k)

        MPI_Barrier(domain_.comm);

      } // for (jphi)
    } // for (kphi)

  } //iproc

  MPI_Barrier(domain_.comm);


  return ;
}        // -----  end of method Spinor::AddMultSpinorEXX  ----- 


// Kmeans for ISDF
// Update: 10/15/2019
void Spinor::AddMultSpinorEXXDF7 ( Fourier& fft, 
    const NumTns<Complex>& phi,
    const DblNumVec& exxgkkR2C,
    Real  exxFraction,
    Real  numSpin,
    const DblNumVec& occupationRate,
    std::string hybridDFType,
    std::string hybridDFKmeansWFType,
    const Real hybridDFKmeansWFAlpha,
    Real  hybridDFKmeansTolerance, 
    Int   hybridDFKmeansMaxIter, 
    const Real numMuFac,
    const Real numGaussianRandomFac,
    const Int numProcScaLAPACK,  
    const Real hybridDFTolerance,
    const Int BlockSizeScaLAPACK,  
    NumTns<Complex>& a3, 
    NumMat<Complex>& VxMat,
    bool isFixColumnDF )
{
  Real timeSta, timeEnd;
  Real timeSta1, timeEnd1;

  if( !fft.isInitialized ){
    ErrorHandling("Fourier is not prepared.");
  }
  statusOFS << "lijl test for AddMultSpinorEXXDF7" << std::endl;
  MPI_Barrier(domain_.comm);
  int mpirank;  MPI_Comm_rank(domain_.comm, &mpirank);
  int mpisize;  MPI_Comm_size(domain_.comm, &mpisize);

  Index3& numGrid = domain_.numGrid;
  Index3& numGridFine = domain_.numGridFine;

  Int ntot     = domain_.NumGridTotal();
  Int ntotFine = domain_.NumGridTotalFine();
  Int ntotR2C = fft.numGridTotalR2C;
  Int ntotR2CFine = fft.numGridTotalR2CFine;
  Int ncom = wavefun_.n();
  Int numStateLocal = wavefun_.p();
  Int numStateTotal = numStateTotal_;

  Int ncomPhi = phi.n();

  Real vol = domain_.Volume();

  if( ncomPhi != 1 || ncom != 1 ){
    ErrorHandling("Spin polarized case not implemented.");
  }

  if( fft.domain.NumGridTotal() != ntot ){
    ErrorHandling("Domain size does not match.");
  }


  // *********************************************************************
  // Perform interpolative separable density fitting
  // *********************************************************************

  //numMu_ = std::min(IRound(numStateTotal*numMuFac), ntot);
  numMu_ = IRound(numStateTotal*numMuFac);

  Int numPre = IRound(std::sqrt(numMu_*numGaussianRandomFac));
  if( numPre > numStateTotal ){
    ErrorHandling("numMu is too large for interpolative separable density fitting!");
  }

  statusOFS << "ntot          = " << ntot << std::endl;
  statusOFS << "numMu         = " << numMu_ << std::endl;
  statusOFS << "numPre        = " << numPre << std::endl;
  statusOFS << "numPre*numPre = " << numPre * numPre << std::endl;

  // Convert the column partition to row partition
  Int numStateBlocksize = numStateTotal / mpisize;
  //Int ntotBlocksize = ntot / mpisize;

  Int numMuBlocksize = numMu_ / mpisize;

  Int numStateLocal1 = numStateBlocksize;
  //Int ntotLocal = ntotBlocksize;

  Int numMuLocal = numMuBlocksize;

  if(mpirank < (numStateTotal % mpisize)){
    numStateLocal1 = numStateBlocksize + 1;
  }

  if(numStateLocal !=  numStateLocal1){
    statusOFS << "numStateLocal = " << numStateLocal << " numStateLocal1 = " << numStateLocal1 << std::endl;
    ErrorHandling("The size is not right in interpolative separable density fitting!");
  }

  if(mpirank < (numMu_ % mpisize)){
    numMuLocal = numMuBlocksize + 1;
  }

  //if(mpirank < (ntot % mpisize)){
  //  ntotLocal = ntotBlocksize + 1;
  //}

  //huwei 
  //2D MPI commucation for all the matrix

  Int I_ONE = 1, I_ZERO = 0;
  //Complex Z_ONE  = Complex(1.0, 0.0);
  //Complex Z_ZERO = Complex(0.0, 0.0);
  //double D_MinusONE = -1.0;
  Complex Z_MinusONE = Complex(-1.0, 0.0);

  Int contxt0, contxt1, contxt11, contxt2;
  Int nprow0, npcol0, myrow0, mycol0, info0;
  Int nprow1, npcol1, myrow1, mycol1, info1;
  Int nprow11, npcol11, myrow11, mycol11, info11;
  Int nprow2, npcol2, myrow2, mycol2, info2;

  Int ncolsNgNe1DCol, nrowsNgNe1DCol, lldNgNe1DCol; 
  Int ncolsNgNe1DRow, nrowsNgNe1DRow, lldNgNe1DRow; 
  Int ncolsNgNe2D, nrowsNgNe2D, lldNgNe2D; 
  Int ncolsNgNu1D, nrowsNgNu1D, lldNgNu1D; 
  Int ncolsNgNu2D, nrowsNgNu2D, lldNgNu2D; 
  Int ncolsNuNg2D, nrowsNuNg2D, lldNuNg2D; 
  Int ncolsNeNe0D, nrowsNeNe0D, lldNeNe0D; 
  Int ncolsNeNe2D, nrowsNeNe2D, lldNeNe2D; 
  Int ncolsNuNu1D, nrowsNuNu1D, lldNuNu1D; 
  Int ncolsNuNu2D, nrowsNuNu2D, lldNuNu2D; 
  Int ncolsNeNu1D, nrowsNeNu1D, lldNeNu1D; 
  Int ncolsNeNu2D, nrowsNeNu2D, lldNeNu2D; 
  Int ncolsNuNe2D, nrowsNuNe2D, lldNuNe2D; 

  Int desc_NgNe1DCol[9];
  Int desc_NgNe1DRow[9];
  Int desc_NgNe2D[9];
  Int desc_NgNu1D[9];
  Int desc_NgNu2D[9];
  Int desc_NuNg2D[9];
  Int desc_NeNe0D[9];
  Int desc_NeNe2D[9];
  Int desc_NuNu1D[9];
  Int desc_NuNu2D[9];
  Int desc_NeNu1D[9];
  Int desc_NeNu2D[9];
  Int desc_NuNe2D[9];

  Int Ng = ntot;
  Int Ne = numStateTotal_; 
  Int Nu = numMu_; 

  // 0D MPI
  nprow0 = 1;
  npcol0 = mpisize;

  Cblacs_get(0, 0, &contxt0);
  Cblacs_gridinit(&contxt0, "C", nprow0, npcol0);
  Cblacs_gridinfo(contxt0, &nprow0, &npcol0, &myrow0, &mycol0);

  SCALAPACK(descinit)(desc_NeNe0D, &Ne, &Ne, &Ne, &Ne, &I_ZERO, &I_ZERO, &contxt0, &Ne, &info0);

  // 1D MPI
  nprow1 = 1;
  npcol1 = mpisize;

  Cblacs_get(0, 0, &contxt1);
  Cblacs_gridinit(&contxt1, "C", nprow1, npcol1);
  Cblacs_gridinfo(contxt1, &nprow1, &npcol1, &myrow1, &mycol1);

  //desc_NgNe1DCol
  if(contxt1 >= 0){
    nrowsNgNe1DCol = SCALAPACK(numroc)(&Ng, &Ng, &myrow1, &I_ZERO, &nprow1);
    ncolsNgNe1DCol = SCALAPACK(numroc)(&Ne, &I_ONE, &mycol1, &I_ZERO, &npcol1);
    lldNgNe1DCol = std::max( nrowsNgNe1DCol, 1 );
  }    

  SCALAPACK(descinit)(desc_NgNe1DCol, &Ng, &Ne, &Ng, &I_ONE, &I_ZERO, 
      &I_ZERO, &contxt1, &lldNgNe1DCol, &info1);

  nprow11 = mpisize;
  npcol11 = 1;

  Cblacs_get(0, 0, &contxt11);
  Cblacs_gridinit(&contxt11, "C", nprow11, npcol11);
  Cblacs_gridinfo(contxt11, &nprow11, &npcol11, &myrow11, &mycol11);

  Int BlockSizeScaLAPACKTemp = BlockSizeScaLAPACK; 
  //desc_NgNe1DRow
  if(contxt11 >= 0){
    nrowsNgNe1DRow = SCALAPACK(numroc)(&Ng, &BlockSizeScaLAPACKTemp, &myrow11, &I_ZERO, &nprow11);
    ncolsNgNe1DRow = SCALAPACK(numroc)(&Ne, &Ne, &mycol11, &I_ZERO, &npcol11);
    lldNgNe1DRow = std::max( nrowsNgNe1DRow, 1 );
  }    

  SCALAPACK(descinit)(desc_NgNe1DRow, &Ng, &Ne, &BlockSizeScaLAPACKTemp, &Ne, &I_ZERO, 
      &I_ZERO, &contxt11, &lldNgNe1DRow, &info11);

  //desc_NeNu1D
  if(contxt11 >= 0){
    nrowsNeNu1D = SCALAPACK(numroc)(&Ne, &I_ONE, &myrow11, &I_ZERO, &nprow11);
    ncolsNeNu1D = SCALAPACK(numroc)(&Nu, &Nu, &mycol11, &I_ZERO, &npcol11);
    lldNeNu1D = std::max( nrowsNeNu1D, 1 );
  }    

  SCALAPACK(descinit)(desc_NeNu1D, &Ne, &Nu, &I_ONE, &Nu, &I_ZERO, 
      &I_ZERO, &contxt11, &lldNeNu1D, &info11);

  //desc_NgNu1D
  if(contxt1 >= 0){
    nrowsNgNu1D = SCALAPACK(numroc)(&Ng, &Ng, &myrow1, &I_ZERO, &nprow1);
    ncolsNgNu1D = SCALAPACK(numroc)(&Nu, &I_ONE, &mycol1, &I_ZERO, &npcol1);
    lldNgNu1D = std::max( nrowsNgNu1D, 1 );
  }    

  SCALAPACK(descinit)(desc_NgNu1D, &Ng, &Nu, &Ng, &I_ONE, &I_ZERO, 
      &I_ZERO, &contxt1, &lldNgNu1D, &info1);

  //desc_NuNu1D
  if(contxt1 >= 0){
    nrowsNuNu1D = SCALAPACK(numroc)(&Nu, &Nu, &myrow1, &I_ZERO, &nprow1);
    ncolsNuNu1D = SCALAPACK(numroc)(&Nu, &I_ONE, &mycol1, &I_ZERO, &npcol1);
    lldNuNu1D = std::max( nrowsNuNu1D, 1 );
  }    

  SCALAPACK(descinit)(desc_NuNu1D, &Nu, &Nu, &Nu, &I_ONE, &I_ZERO, 
      &I_ZERO, &contxt1, &lldNuNu1D, &info1);


  // 2D MPI
  for( Int i = IRound(sqrt(double(mpisize))); i <= mpisize; i++){
    nprow2 = i; npcol2 = mpisize / nprow2;
    if( (nprow2 >= npcol2) && (nprow2 * npcol2 == mpisize) ) break;
  }

  Cblacs_get(0, 0, &contxt2);
  //Cblacs_gridinit(&contxt2, "C", nprow2, npcol2);

  IntNumVec pmap2(mpisize);
  for ( Int i = 0; i < mpisize; i++ ){
    pmap2[i] = i;
  }
  Cblacs_gridmap(&contxt2, &pmap2[0], nprow2, nprow2, npcol2);

  Int mb2 = BlockSizeScaLAPACK;
  Int nb2 = BlockSizeScaLAPACK;

  //desc_NgNe2D
  if(contxt2 >= 0){
    Cblacs_gridinfo(contxt2, &nprow2, &npcol2, &myrow2, &mycol2);
    nrowsNgNe2D = SCALAPACK(numroc)(&Ng, &mb2, &myrow2, &I_ZERO, &nprow2);
    ncolsNgNe2D = SCALAPACK(numroc)(&Ne, &nb2, &mycol2, &I_ZERO, &npcol2);
    lldNgNe2D = std::max( nrowsNgNe2D, 1 );
  }

  SCALAPACK(descinit)(desc_NgNe2D, &Ng, &Ne, &mb2, &nb2, &I_ZERO, 
      &I_ZERO, &contxt2, &lldNgNe2D, &info2);

  //desc_NgNu2D
  if(contxt2 >= 0){
    Cblacs_gridinfo(contxt2, &nprow2, &npcol2, &myrow2, &mycol2);
    nrowsNgNu2D = SCALAPACK(numroc)(&Ng, &mb2, &myrow2, &I_ZERO, &nprow2);
    ncolsNgNu2D = SCALAPACK(numroc)(&Nu, &nb2, &mycol2, &I_ZERO, &npcol2);
    lldNgNu2D = std::max( nrowsNgNu2D, 1 );
  }

  SCALAPACK(descinit)(desc_NgNu2D, &Ng, &Nu, &mb2, &nb2, &I_ZERO, 
      &I_ZERO, &contxt2, &lldNgNu2D, &info2);

  //desc_NuNg2D
  if(contxt2 >= 0){
    Cblacs_gridinfo(contxt2, &nprow2, &npcol2, &myrow2, &mycol2);
    nrowsNuNg2D = SCALAPACK(numroc)(&Nu, &mb2, &myrow2, &I_ZERO, &nprow2);
    ncolsNuNg2D = SCALAPACK(numroc)(&Ng, &nb2, &mycol2, &I_ZERO, &npcol2);
    lldNuNg2D = std::max( nrowsNuNg2D, 1 );
  }

  SCALAPACK(descinit)(desc_NuNg2D, &Nu, &Ng, &mb2, &nb2, &I_ZERO, 
      &I_ZERO, &contxt2, &lldNuNg2D, &info2);

  //desc_NeNe2D
  if(contxt2 >= 0){
    Cblacs_gridinfo(contxt2, &nprow2, &npcol2, &myrow2, &mycol2);
    nrowsNeNe2D = SCALAPACK(numroc)(&Ne, &mb2, &myrow2, &I_ZERO, &nprow2);
    ncolsNeNe2D = SCALAPACK(numroc)(&Ne, &nb2, &mycol2, &I_ZERO, &npcol2);
    lldNeNe2D = std::max( nrowsNeNe2D, 1 );
  }

  SCALAPACK(descinit)(desc_NeNe2D, &Ne, &Ne, &mb2, &nb2, &I_ZERO, 
      &I_ZERO, &contxt2, &lldNeNe2D, &info2);

  //desc_NuNu2D
  if(contxt2 >= 0){
    Cblacs_gridinfo(contxt2, &nprow2, &npcol2, &myrow2, &mycol2);
    nrowsNuNu2D = SCALAPACK(numroc)(&Nu, &mb2, &myrow2, &I_ZERO, &nprow2);
    ncolsNuNu2D = SCALAPACK(numroc)(&Nu, &nb2, &mycol2, &I_ZERO, &npcol2);
    lldNuNu2D = std::max( nrowsNuNu2D, 1 );
  }

  SCALAPACK(descinit)(desc_NuNu2D, &Nu, &Nu, &mb2, &nb2, &I_ZERO, 
      &I_ZERO, &contxt2, &lldNuNu2D, &info2);

  //desc_NeNu2D
  if(contxt2 >= 0){
    Cblacs_gridinfo(contxt2, &nprow2, &npcol2, &myrow2, &mycol2);
    nrowsNeNu2D = SCALAPACK(numroc)(&Ne, &mb2, &myrow2, &I_ZERO, &nprow2);
    ncolsNeNu2D = SCALAPACK(numroc)(&Nu, &nb2, &mycol2, &I_ZERO, &npcol2);
    lldNeNu2D = std::max( nrowsNeNu2D, 1 );
  }

  SCALAPACK(descinit)(desc_NeNu2D, &Ne, &Nu, &mb2, &nb2, &I_ZERO, 
      &I_ZERO, &contxt2, &lldNeNu2D, &info2);

  //desc_NuNe2D
  if(contxt2 >= 0){
    Cblacs_gridinfo(contxt2, &nprow2, &npcol2, &myrow2, &mycol2);
    nrowsNuNe2D = SCALAPACK(numroc)(&Nu, &mb2, &myrow2, &I_ZERO, &nprow2);
    ncolsNuNe2D = SCALAPACK(numroc)(&Ne, &nb2, &mycol2, &I_ZERO, &npcol2);
    lldNuNe2D = std::max( nrowsNuNe2D, 1 );
  }

  SCALAPACK(descinit)(desc_NuNe2D, &Nu, &Ne, &mb2, &nb2, &I_ZERO, 
      &I_ZERO, &contxt2, &lldNuNe2D, &info2);

  CpxNumMat phiCol( ntot, numStateLocal );
  SetValue( phiCol, Z_ZERO );

  CpxNumMat psiCol( ntot, numStateLocal );
  SetValue( psiCol, Z_ZERO );

  lapack::Lacpy( 'A', ntot, numStateLocal, phi.Data(), ntot, phiCol.Data(), ntot );
  lapack::Lacpy( 'A', ntot, numStateLocal, wavefun_.Data(), ntot, psiCol.Data(), ntot );

  //-------add by lijl 20211101
  CpxNumMat phiColconj( ntot, numStateLocal );
  SetValue( phiColconj, Z_ZERO );
  Complex* phiColconjPtr = phiColconj.Data();
  Complex* phiColPtr = phiCol.Data();
  for( int i = 0; i < numStateLocal; i++)
  {    
      for ( int j = 0; j < ntot; j++)
      {
          phiColconjPtr[i*ntot+j] = std::conj(phiColPtr[i*ntot+j]);
      }
  } 
  // Computing the indices is optional


  Int ntotLocal = nrowsNgNe1DRow; 
  //Int ntotLocalMG = nrowsNgNe1DRow;
  //Int ntotMG = ntot;
  Int ntotLocalMG, ntotMG;

  //if( (ntot % mpisize) == 0 ){
  //  ntotLocalMG = ntotBlocksize;
  //}
  //else{
  //  ntotLocalMG = ntotBlocksize + 1;
  // }
  // --------------by lijl 20200526
  /*  
  if ((pivQR_.m_ != ntot) || (hybridDFType == "QRCP")){
    pivQR_.Resize(ntot);
    SetValue( pivQR_, 0 ); // Important. Otherwise QRCP uses piv as initial guess
    // Q factor does not need to be used
  }
  */
  pivQR_.Resize(ntot);

  if( (isFixColumnDF == false) && ((hybridDFType == "QRCP") || (hybridDFType == "Kmeans+QRCP"))){
    GetTime( timeSta );

    CpxNumMat localphiGRow( ntotLocal, numPre );
    SetValue( localphiGRow, Z_ZERO );

    CpxNumMat localpsiGRow( ntotLocal, numPre );
    SetValue( localpsiGRow, Z_ZERO );

    CpxNumMat G(numStateTotal, numPre);
    SetValue( G, Z_ZERO );

    CpxNumMat phiRow( ntotLocal, numStateTotal );
    SetValue( phiRow, Z_ZERO );

    CpxNumMat psiRow( ntotLocal, numStateTotal );
    SetValue( psiRow, Z_ZERO );

    SCALAPACK(pzgemr2d)(&Ng, &Ne, psiCol.Data(), &I_ONE, &I_ONE, desc_NgNe1DCol, 
        psiRow.Data(), &I_ONE, &I_ONE, desc_NgNe1DRow, &contxt1 );

    SCALAPACK(pzgemr2d)(&Ng, &Ne, phiColconj.Data(), &I_ONE, &I_ONE, desc_NgNe1DCol, 
        phiRow.Data(), &I_ONE, &I_ONE, desc_NgNe1DRow, &contxt1 );

    // Step 1: Pre-compression of the wavefunctions. This uses
    // multiplication with orthonormalized random Gaussian matrices

    GetTime( timeSta1 );
   /*
    if (G_.m_!=numStateTotal){
      CpxNumMat G(numStateTotal, numPre);
      if ( mpirank == 0 ) {
        GaussianRandom(G);
        lapack::Orth( numStateTotal, numPre, G.Data(), numStateTotal );
        statusOFS << "Random projection initialzied." << std::endl << std::endl;
      }
      MPI_Bcast(G.Data(), 2*numStateTotal * numPre, MPI_DOUBLE, 0, domain_.comm);
      G_ = G;
    } else {
      statusOFS << "Random projection reused." << std::endl;
    }
    statusOFS << "G(0,0) = " << G_(0,0) << std::endl << std::endl;
    */
    if ( mpirank == 0 ) {
      GaussianRandom(G);
      lapack::Orth( numStateTotal, numPre, G.Data(), numStateTotal );
      statusOFS << "Random projection initialzied." << std::endl << std::endl;
      }
    MPI_Bcast(G.Data(), 2*numStateTotal * numPre, MPI_DOUBLE, 0, domain_.comm);
    GetTime( timeEnd1 );
#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for Gaussian MPI_Bcast is " <<
      timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
#endif

    GetTime( timeSta1 );

    blas::Gemm( 'N', 'N', ntotLocal, numPre, numStateTotal, Z_ONE, 
        phiRow.Data(), ntotLocal, G.Data(), numStateTotal, Z_ZERO,
        localphiGRow.Data(), ntotLocal );

    blas::Gemm( 'N', 'N', ntotLocal, numPre, numStateTotal, Z_ONE, 
        psiRow.Data(), ntotLocal, G.Data(), numStateTotal, Z_ZERO,
        localpsiGRow.Data(), ntotLocal );

    GetTime( timeEnd1 );
#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for localphiGRow and localpsiGRow Gemm is " <<
      timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
#endif

    // Step 2: Pivoted QR decomposition  for the Hadamard product of
    // the compressed matrix. Transpose format for QRCP

    // NOTE: All processors should have the same ntotLocalMG


    int m_MGTemp = numPre*numPre;
    int n_MGTemp = ntot;

    CpxNumMat MGCol( m_MGTemp, ntotLocal );

    DblNumVec MGNorm(ntot);
    for( Int k = 0; k < ntot; k++ ){
      MGNorm(k) = 0;
    }
    DblNumVec MGNormLocal(ntotLocal);
    for( Int k = 0; k < ntotLocal; k++ ){
      MGNormLocal(k) = 0;
    }

    GetTime( timeSta1 );
//add t_temp ----by lijl
    //Complex t_temp;
//------by lijl
    for( Int j = 0; j < numPre; j++ ){
      for( Int i = 0; i < numPre; i++ ){
        for( Int ir = 0; ir < ntotLocal; ir++ ){
          //t_temp.real(localphiGRow(ir,i).real()*localpsiGRow(ir,j).real() + localphiGRow(ir,i).imag()*localpsiGRow(ir,j).imag());
	  //t_temp.imag(-localphiGRow(ir,i).real()*localpsiGRow(ir,j).imag() + localphiGRow(ir,i).imag()*localpsiGRow(ir,j).real()); 
          MGCol(i+j*numPre,ir) = localphiGRow(ir,i) * std::conj(localphiGRow(ir,j));
          MGNormLocal(ir) += std::norm(MGCol(i+j*numPre,ir));   
        }
      }
    }

    GetTime( timeEnd1 );
#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for computing MG from localphiGRow and localpsiGRow is " <<
      timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
#endif

    //int ntotMGTemp, ntotLocalMGTemp;

    IntNumVec MGIdx(ntot);

    {
      Int ncols0D, nrows0D, lld0D; 
      Int ncols1D, nrows1D, lld1D; 

      Int desc_0D[9];
      Int desc_1D[9];

      Cblacs_get(0, 0, &contxt0);
      Cblacs_gridinit(&contxt0, "C", nprow0, npcol0);
      Cblacs_gridinfo(contxt0, &nprow0, &npcol0, &myrow0, &mycol0);

      SCALAPACK(descinit)(desc_0D, &Ng, &I_ONE, &Ng, &I_ONE, &I_ZERO, &I_ZERO, &contxt0, &Ng, &info0);

      if(contxt11 >= 0){
        nrows1D = SCALAPACK(numroc)(&Ng, &BlockSizeScaLAPACKTemp, &myrow11, &I_ZERO, &nprow11);
        ncols1D = SCALAPACK(numroc)(&I_ONE, &I_ONE, &mycol11, &I_ZERO, &npcol11);
        lld1D = std::max( nrows1D, 1 );
      }    

      SCALAPACK(descinit)(desc_1D, &Ng, &I_ONE, &BlockSizeScaLAPACKTemp, &I_ONE, &I_ZERO, 
          &I_ZERO, &contxt11, &lld1D, &info11);


      SCALAPACK(pdgemr2d)(&Ng, &I_ONE, MGNormLocal.Data(), &I_ONE, &I_ONE, desc_1D, 
          MGNorm.Data(), &I_ONE, &I_ONE, desc_0D, &contxt11 );

      MPI_Bcast( MGNorm.Data(), ntot, MPI_DOUBLE, 0, domain_.comm );


      double MGNormMax = *(std::max_element( MGNorm.Data(), MGNorm.Data() + ntot ) );
      double MGNormMin = *(std::min_element( MGNorm.Data(), MGNorm.Data() + ntot ) );

      ntotMG = 0;
      SetValue( MGIdx, 0 );

      for( Int k = 0; k < ntot; k++ ){
        if(MGNorm(k) > hybridDFTolerance){
          MGIdx(ntotMG) = k; 
          ntotMG = ntotMG + 1;
        }
      }

#if ( _DEBUGlevel_ >= 0 )
      statusOFS << "The col size for MG: " << " ntotMG = " << ntotMG << " ntotMG/ntot = " << Real(ntotMG)/Real(ntot) << std::endl << std::endl;
      statusOFS << "The norm range for MG: " <<  " MGNormMax = " << MGNormMax << " MGNormMin = " << MGNormMin << std::endl << std::endl;
#endif

    }

    Int ncols1DCol, nrows1DCol, lld1DCol; 
    Int ncols1DRow, nrows1DRow, lld1DRow; 

    Int desc_1DCol[9];
    Int desc_1DRow[9];

    if(contxt1 >= 0){
      nrows1DCol = SCALAPACK(numroc)(&m_MGTemp, &m_MGTemp, &myrow1, &I_ZERO, &nprow1);
      ncols1DCol = SCALAPACK(numroc)(&n_MGTemp, &BlockSizeScaLAPACKTemp, &mycol1, &I_ZERO, &npcol1);
      lld1DCol = std::max( nrows1DCol, 1 );
    }    

    SCALAPACK(descinit)(desc_1DCol, &m_MGTemp, &n_MGTemp, &m_MGTemp, &BlockSizeScaLAPACKTemp, &I_ZERO, 
        &I_ZERO, &contxt1, &lld1DCol, &info1);

    if(contxt11 >= 0){
      nrows1DRow = SCALAPACK(numroc)(&m_MGTemp, &I_ONE, &myrow11, &I_ZERO, &nprow11);
      ncols1DRow = SCALAPACK(numroc)(&n_MGTemp, &n_MGTemp, &mycol11, &I_ZERO, &npcol11);
      lld1DRow = std::max( nrows1DRow, 1 );
    }    

    SCALAPACK(descinit)(desc_1DRow, &m_MGTemp, &n_MGTemp, &I_ONE, &n_MGTemp, &I_ZERO, 
        &I_ZERO, &contxt11, &lld1DRow, &info11);

    CpxNumMat MGRow( nrows1DRow, ntot );
    SetValue( MGRow, Z_ZERO );


    GetTime( timeSta1 );

    SCALAPACK(pzgemr2d)(&m_MGTemp, &n_MGTemp, MGCol.Data(), &I_ONE, &I_ONE, desc_1DCol, 
        MGRow.Data(), &I_ONE, &I_ONE, desc_1DRow, &contxt1 );

    GetTime( timeEnd1 );
#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for computing MG from Col and Row is " <<
      timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
#endif

    CpxNumMat MG( nrows1DRow, ntotMG );

    for( Int i = 0; i < nrows1DRow; i++ ){
      for( Int j = 0; j < ntotMG; j++ ){
        MG(i,j) = MGRow(i,MGIdx(j));
      }
    }

    CpxNumVec tau(ntotMG);
    //pivQR_.Resize(ntot);
    SetValue( pivQR_, 0 ); // Important. Otherwise QRCP uses piv as initial guess
    // Q factor does not need to be used

    for( Int k = 0; k < ntotMG; k++ ){
      tau[k] = Z_ZERO;
    }

    Real timeQRCPSta, timeQRCPEnd;
    GetTime( timeQRCPSta );
#if 0
    if(0){  
      lapack::QRCP( numPre*numPre, ntotMG, MG.Data(), numPre*numPre, 
          pivQR_.Data(), tau.Data() );
    }//
#endif

    if(0){ // ScaLAPACL QRCP
      Int contxt;
      Int nprow, npcol, myrow, mycol, info;
      Cblacs_get(0, 0, &contxt);
      nprow = 1;
      npcol = mpisize;

      Cblacs_gridinit(&contxt, "C", nprow, npcol);
      Cblacs_gridinfo(contxt, &nprow, &npcol, &myrow, &mycol);
      Int desc_MG[9];

      Int irsrc = 0;
      Int icsrc = 0;

      Int mb_MG = numPre*numPre;
      Int nb_MG = ntotLocalMG;

      // FIXME The current routine does not actually allow ntotLocal to be different on different processors.
      // This must be fixed.
      SCALAPACK(descinit)(&desc_MG[0], &mb_MG, &ntotMG, &mb_MG, &nb_MG, &irsrc, 
          &icsrc, &contxt, &mb_MG, &info);

      IntNumVec pivQRTmp(ntotMG), pivQRLocal(ntotMG);
      if( mb_MG > ntot ){
        std::ostringstream msg;
        msg << "numPre*numPre > ntot. The number of grid points is perhaps too small!" << std::endl;
        ErrorHandling( msg.str().c_str() );
      }
      // DiagR is only for debugging purpose
      //        DblNumVec diagRLocal( mb_MG );
      //        DblNumVec diagR( mb_MG );

      SetValue( pivQRTmp, 0 );
      SetValue( pivQRLocal, 0 );
      SetValue( pivQR_, 0 );


      //        SetValue( diagRLocal, 0.0 );
      //        SetValue( diagR, 0.0 );
#if 1
      //-----lijl
      if(1) {
        scalapack::QRCPF( mb_MG, ntotMG, MG.Data(), &desc_MG[0], 
            pivQRTmp.Data(), tau.Data() );
      }
#endif
#if 0
      if(1) {
        scalapack::QRCPR( mb_MG, ntotMG, numMu_, MG.Data(), &desc_MG[0], 
            pivQRTmp.Data(), tau.Data(), 80, 40 );
      }
#endif

      // Combine the local pivQRTmp to global pivQR_
      for( Int j = 0; j < ntotLocalMG; j++ ){
        pivQRLocal[j + mpirank * ntotLocalMG] = pivQRTmp[j];
      }

      //        std::cout << "diag of MG = " << std::endl;
      //        if(mpirank == 0){
      //          std::cout << pivQRLocal << std::endl;
      //          for( Int j = 0; j < mb_MG; j++ ){
      //            std::cout << MG(j,j) << std::endl;
      //          }
      //        }
      MPI_Allreduce( pivQRLocal.Data(), pivQR_.Data(), 
          ntotMG, MPI_INT, MPI_SUM, domain_.comm );


      if(contxt >= 0) {
        Cblacs_gridexit( contxt );
      }

    } //ScaLAPACL QRCP


    if(1){ //ScaLAPACL QRCP 2D

      Int contxt1D, contxt2D;
      Int nprow1D, npcol1D, myrow1D, mycol1D, info1D;
      Int nprow2D, npcol2D, myrow2D, mycol2D, info2D;

      Int ncols1D, nrows1D, lld1D; 
      Int ncols2D, nrows2D, lld2D; 

      Int desc_MG1D[9];
      Int desc_MG2D[9];

      Int m_MG = numPre*numPre;
      Int n_MG = ntotMG;

      Int mb_MG1D = 1;
      Int nb_MG1D = ntotMG;

      nprow1D = mpisize;
      npcol1D = 1;

      Cblacs_get(0, 0, &contxt1D);
      Cblacs_gridinit(&contxt1D, "C", nprow1D, npcol1D);
      Cblacs_gridinfo(contxt1D, &nprow1D, &npcol1D, &myrow1D, &mycol1D);

      nrows1D = SCALAPACK(numroc)(&m_MG, &mb_MG1D, &myrow1D, &I_ZERO, &nprow1D);
      ncols1D = SCALAPACK(numroc)(&n_MG, &nb_MG1D, &mycol1D, &I_ZERO, &npcol1D);

      lld1D = std::max( nrows1D, 1 );

      SCALAPACK(descinit)(desc_MG1D, &m_MG, &n_MG, &mb_MG1D, &nb_MG1D, &I_ZERO, 
          &I_ZERO, &contxt1D, &lld1D, &info1D);

      for( Int i = std::min(mpisize, IRound(sqrt(double(mpisize*(n_MG/m_MG))))); 
          i <= mpisize; i++){
        npcol2D = i; nprow2D = mpisize / npcol2D;
        if( (npcol2D >= nprow2D) && (nprow2D * npcol2D == mpisize) ) break;
      }

      Cblacs_get(0, 0, &contxt2D);
      //Cblacs_gridinit(&contxt2D, "C", nprow2D, npcol2D);

      IntNumVec pmap(mpisize);
      for ( Int i = 0; i < mpisize; i++ ){
        pmap[i] = i;
      }
      Cblacs_gridmap(&contxt2D, &pmap[0], nprow2D, nprow2D, npcol2D);

      Int m_MG2DBlocksize = BlockSizeScaLAPACK;
      Int n_MG2DBlocksize = BlockSizeScaLAPACK;

      //Int m_MG2Local = m_MG/(m_MG2DBlocksize*nprow2D)*m_MG2DBlocksize;
      //Int n_MG2Local = n_MG/(n_MG2DBlocksize*npcol2D)*n_MG2DBlocksize;
      Int m_MG2Local, n_MG2Local;

      //MPI_Comm rowComm = MPI_COMM_NULL;
      MPI_Comm colComm = MPI_COMM_NULL;

      Int mpirankRow, mpisizeRow, mpirankCol, mpisizeCol;

      //MPI_Comm_split( domain_.comm, mpirank / nprow2D, mpirank, &rowComm );
      MPI_Comm_split( domain_.comm, mpirank % nprow2D, mpirank, &colComm );

      //MPI_Comm_rank(rowComm, &mpirankRow);
      //MPI_Comm_size(rowComm, &mpisizeRow);

      MPI_Comm_rank(colComm, &mpirankCol);
      MPI_Comm_size(colComm, &mpisizeCol);

      if(0){

        if((m_MG % (m_MG2DBlocksize * nprow2D))!= 0){ 
          if(mpirankRow < ((m_MG % (m_MG2DBlocksize*nprow2D)) / m_MG2DBlocksize)){
            m_MG2Local = m_MG2Local + m_MG2DBlocksize;
          }
          if(((m_MG % (m_MG2DBlocksize*nprow2D)) % m_MG2DBlocksize) != 0){
            if(mpirankRow == ((m_MG % (m_MG2DBlocksize*nprow2D)) / m_MG2DBlocksize)){
              m_MG2Local = m_MG2Local + ((m_MG % (m_MG2DBlocksize*nprow2D)) % m_MG2DBlocksize);
            }
          }
        }

        if((n_MG % (n_MG2DBlocksize * npcol2D))!= 0){ 
          if(mpirankCol < ((n_MG % (n_MG2DBlocksize*npcol2D)) / n_MG2DBlocksize)){
            n_MG2Local = n_MG2Local + n_MG2DBlocksize;
          }
          if(((n_MG % (n_MG2DBlocksize*nprow2D)) % n_MG2DBlocksize) != 0){
            if(mpirankCol == ((n_MG % (n_MG2DBlocksize*npcol2D)) / n_MG2DBlocksize)){
              n_MG2Local = n_MG2Local + ((n_MG % (n_MG2DBlocksize*nprow2D)) % n_MG2DBlocksize);
            }
          }
        }

      } // if(0)

      if(contxt2D >= 0){
        Cblacs_gridinfo(contxt2D, &nprow2D, &npcol2D, &myrow2D, &mycol2D);
        nrows2D = SCALAPACK(numroc)(&m_MG, &m_MG2DBlocksize, &myrow2D, &I_ZERO, &nprow2D);
        ncols2D = SCALAPACK(numroc)(&n_MG, &n_MG2DBlocksize, &mycol2D, &I_ZERO, &npcol2D);
        lld2D = std::max( nrows2D, 1 );
      }

      SCALAPACK(descinit)(desc_MG2D, &m_MG, &n_MG, &m_MG2DBlocksize, &n_MG2DBlocksize, &I_ZERO, 
          &I_ZERO, &contxt2D, &lld2D, &info2D);

      m_MG2Local = nrows2D;
      n_MG2Local = ncols2D;

      IntNumVec pivQRTmp1(ntotMG), pivQRTmp2(ntotMG), pivQRLocal(ntotMG);
      if( m_MG > ntot ){
        std::ostringstream msg;
        msg << "numPre*numPre > ntot. The number of grid points is perhaps too small!" << std::endl;
        ErrorHandling( msg.str().c_str() );
      }
      // DiagR is only for debugging purpose
      //        DblNumVec diagRLocal( mb_MG );
      //        DblNumVec diagR( mb_MG );

      CpxNumMat&  MG1D = MG;
      CpxNumMat  MG2D (m_MG2Local, n_MG2Local);

      SCALAPACK(pzgemr2d)(&m_MG, &n_MG, MG1D.Data(), &I_ONE, &I_ONE, desc_MG1D, 
          MG2D.Data(), &I_ONE, &I_ONE, desc_MG2D, &contxt1D );

      if(contxt2D >= 0){

        Real timeQRCP1, timeQRCP2;
        GetTime( timeQRCP1 );

        SetValue( pivQRTmp1, 0 );
        scalapack::QRCPF( m_MG, n_MG, MG2D.Data(), desc_MG2D, pivQRTmp1.Data(), tau.Data() );

        //scalapack::QRCPR( m_MG, n_MG, numMu_, MG2D.Data(), desc_MG2D, pivQRTmp1.Data(), tau.Data(), BlockSizeScaLAPACK, 32);

        GetTime( timeQRCP2 );

#if ( _DEBUGlevel_ >= 0 )
        statusOFS << "Time only for QRCP is " << timeQRCP2 - timeQRCP1 << " [s]" << std::endl << std::endl;
#endif

      }

      // Redistribute back eigenvectors
      //SetValue(MG1D, 0.0 );

      //SCALAPACK(pdgemr2d)( &m_MG, &n_MG, MG2D.Data(), &I_ONE, &I_ONE, desc_MG2D,
      //    MG1D.Data(), &I_ONE, &I_ONE, desc_MG1D, &contxt1D );

      // Combine the local pivQRTmp to global pivQR_
      SetValue( pivQRLocal, 0 );
      for( Int j = 0; j < n_MG2Local; j++ ){
        pivQRLocal[ (j / n_MG2DBlocksize) * n_MG2DBlocksize * npcol2D + mpirankCol * n_MG2DBlocksize + j % n_MG2DBlocksize] = pivQRTmp1[j];
      }

      SetValue( pivQRTmp2, 0 );
      MPI_Allreduce( pivQRLocal.Data(), pivQRTmp2.Data(), 
          ntotMG, MPI_INT, MPI_SUM, colComm );

      SetValue( pivQR_, 0 );
      for( Int j = 0; j < ntotMG; j++ ){
        pivQR_(j) = MGIdx(pivQRTmp2(j));
      }

      //if( rowComm != MPI_COMM_NULL ) MPI_Comm_free( & rowComm );
      if( colComm != MPI_COMM_NULL ) MPI_Comm_free( & colComm );

      if(contxt2D >= 0) {
        Cblacs_gridexit( contxt2D );
      }

      if(contxt1D >= 0) {
        Cblacs_gridexit( contxt1D );
      }

    } // if(1) ScaLAPACL QRCP

    GetTime( timeQRCPEnd );

    //statusOFS << std::endl<< "All processors exit with abort in spinor.cpp." << std::endl;

#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for QRCP alone is " <<
      timeQRCPEnd - timeQRCPSta << " [s]" << std::endl << std::endl;
#endif
    
    if(0){
      Real tolR = std::abs(MG(numMu_-1,numMu_-1)/MG(0,0));
      statusOFS << "numMu_ = " << numMu_ << std::endl;
      statusOFS << "|R(numMu-1,numMu-1)/R(0,0)| = " << tolR << std::endl;
    }

    GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for density fitting with QRCP is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

    // Dump out pivQR_
    if(0){
      std::ostringstream muStream;
      serialize( pivQR_, muStream, NO_MASK );
      SharedWrite( "pivQR", muStream );
    }

    if (hybridDFType == "Kmeans+QRCP"){
      hybridDFType = "Kmeans";
    }
  } //  if( (isFixColumnDF == false) && (hybridDFType == "QRCP") )

  if( (isFixColumnDF == false) && (hybridDFType == "Kmeans") ){
   
    GetTime( timeSta );

    DblNumVec weight(ntot);
    Real* wp = weight.Data();
    Real timeW1,timeW2;
    Real timeKMEANSta, timeKMEANEnd;

    GetTime(timeW1); 
    DblNumVec phiW(ntot);
    SetValue(phiW,0.0);
    Real* phW = phiW.Data();
    Complex* ph = phiColconj.Data();
    //add psi part
    Complex* ps = psiCol.Data();
    // only consider single wavefunction phi
 #if 0
    if(0){
    for (int j = 0; j < numStateLocal; j++){
      for(int i = 0; i < ntot; i++){
        phW[i] += ph[i+j*ntot]*ph[i+j*ntot];
      }
    }
    }
 #endif
    //consider two wavefunction phi and psi
    if(1){
    for (int j = 0; j < numStateLocal; j++){
      for(int i = 0; i < ntot; i++){
        //phW[i] += (ph[i+j*ntot]*std::conj(ph[i+j*ntot])).real() + (ps[i+j*ntot]*std::conj(ps[i+j*ntot])).real();
        if (hybridDFKmeansWFType == "Add"){
            if (hybridDFKmeansWFAlpha == 2.0){
                phW[i] += std::norm(ph[i+j*ntot]) + std::norm(ps[i+j*ntot]);}  // ph^2 + ps^2
            else if (hybridDFKmeansWFAlpha == 1.0){
	        phW[i] += std::sqrt(std::norm(ph[i+j*ntot])) + std::sqrt(std::norm(ps[i+j*ntot]));}  // ph^1 + ps^1
            else if (hybridDFKmeansWFAlpha == 0.5){
	        phW[i] += std::sqrt(std::sqrt(std::norm(ph[i+j*ntot]))) + std::sqrt(std::sqrt(std::norm(ps[i+j*ntot])));}  // ph^0.5 + ps^0.5
            else if (hybridDFKmeansWFAlpha == 1.5){
	        phW[i] += (std::sqrt(std::norm(ph[i+j*ntot])))*(std::sqrt(std::sqrt(std::norm(ph[i+j*ntot])))) + (std::sqrt(std::norm(ps[i+j*ntot])))*(std::sqrt(std::sqrt(std::norm(ps[i+j*ntot]))));}  // ph^1.5 + ps^1.5
            else if (hybridDFKmeansWFAlpha == 2.5){
	        phW[i] += (std::norm(ph[i+j*ntot]))*(std::sqrt(std::sqrt(std::norm(ph[i+j*ntot])))) + (std::norm(ps[i+j*ntot]))*(std::sqrt(std::sqrt(std::norm(ps[i+j*ntot]))));} // ph^2.5 + ps^2.5
	    else if (hybridDFKmeansWFAlpha == 3.0){
                phW[i] += (std::norm(ph[i+j*ntot]))*(std::sqrt(std::norm(ph[i+j*ntot]))) + (std::norm(ps[i+j*ntot]))*(std::sqrt(std::norm(ps[i+j*ntot])));} // ph^3 + ps^3
	    else if (hybridDFKmeansWFAlpha == 4.0){
                phW[i] += (std::norm(ph[i+j*ntot]))*(std::norm(ph[i+j*ntot])) + (std::norm(ps[i+j*ntot]))*(std::norm(ps[i+j*ntot]));} // ph^4 + ps^4
        }
        if (hybridDFKmeansWFType == "Multi"){
            phW[i] += std::norm(ph[i+j*ntot]) * std::norm(ps[i+j*ntot]);
        }
      }
    }
    }
    MPI_Barrier(domain_.comm);
    MPI_Reduce(phW, wp, ntot, MPI_DOUBLE, MPI_SUM, 0, domain_.comm);
    MPI_Bcast(wp, ntot, MPI_DOUBLE, 0, domain_.comm);
    GetTime(timeW2);
    statusOFS << "Time for computing weight in Kmeans: " << timeW2-timeW1 << "[s]" << std::endl << std::endl;

    int rk = numMu_;
    SetValue( pivQR_, 0 ); // Important. Otherwise QRCP uses piv as initial guess
    GetTime(timeKMEANSta);
    KMEAN(ntot, weight, rk, hybridDFKmeansTolerance, hybridDFKmeansMaxIter, hybridDFTolerance, domain_, pivQR_.Data());
    GetTime(timeKMEANEnd);
    statusOFS << "Time for Kmeans alone is lijl complex" << timeKMEANEnd-timeKMEANSta << "[s]" << std::endl << std::endl;
 
    GetTime(timeEnd);
    
    statusOFS << "Time for density fitting with Kmeans is " << timeEnd-timeSta << "[s]" << std::endl << std::endl;

    // Dump out pivQR_
    if(0){
      std::ostringstream muStream;
      serialize( pivQR_, muStream, NO_MASK );
      SharedWrite( "pivQR", muStream );
    }

  } //  if( (isFixColumnDF == false) && (hybridDFType == "Kmeans") )

  // Load pivQR_ file
  if(0){
    statusOFS << "Loading pivQR file.." << std::endl;
    std::istringstream muStream;
    SharedRead( "pivQR", muStream );
    deserialize( pivQR_, muStream, NO_MASK );
  }

  // *********************************************************************
  // Compute the interpolation matrix via the density matrix formulation
  // *********************************************************************

  GetTime( timeSta );

  // PhiMu is scaled by the occupation number to reflect the "true" density matrix
  //IntNumVec pivMu(numMu_);
  IntNumVec pivMu1(numMu_);

  for( Int mu = 0; mu < numMu_; mu++ ){
    pivMu1(mu) = pivQR_(mu);
  }


  //if(ntot % mpisize != 0){
  //  for( Int mu = 0; mu < numMu_; mu++ ){
  //    Int k1 = (pivMu1(mu) / ntotLocalMG) - (ntot % mpisize);
  //    if(k1 > 0){
  //      pivMu1(mu) = pivQR_(mu) - k1;
  //    }
  //  }
  //}

  GetTime( timeSta1 );

  CpxNumMat psiMuCol(numStateLocal, numMu_);
  CpxNumMat phiMuCol(numStateLocal, numMu_);
  SetValue( psiMuCol, Z_ZERO );
  SetValue( phiMuCol, Z_ZERO );

  for (Int k=0; k<numStateLocal; k++) {
    for (Int mu=0; mu<numMu_; mu++) {
      psiMuCol(k, mu) = std::conj(psiCol(pivMu1(mu),k));
      phiMuCol(k, mu) = std::conj(phiCol(pivMu1(mu),k)) * occupationRate[(k * mpisize + mpirank)];
    }
  }

  CpxNumMat psiMu2D(nrowsNeNu2D, ncolsNeNu2D);
  CpxNumMat phiMu2D(nrowsNeNu2D, ncolsNeNu2D);
  SetValue( psiMu2D, Z_ZERO );
  SetValue( phiMu2D, Z_ZERO );

  SCALAPACK(pzgemr2d)(&Ne, &Nu, psiMuCol.Data(), &I_ONE, &I_ONE, desc_NeNu1D, 
      psiMu2D.Data(), &I_ONE, &I_ONE, desc_NeNu2D, &contxt11 );

  SCALAPACK(pzgemr2d)(&Ne, &Nu, phiMuCol.Data(), &I_ONE, &I_ONE, desc_NeNu1D, 
      phiMu2D.Data(), &I_ONE, &I_ONE, desc_NeNu2D, &contxt11 );

  GetTime( timeEnd1 );

#if ( _DEBUGlevel_ >= 0 )
  statusOFS << "Time for computing psiMuRow and phiMuRow is " <<
    timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
#endif

  GetTime( timeSta1 );

  CpxNumMat psi2D(nrowsNgNe2D, ncolsNgNe2D);
  CpxNumMat phi2D(nrowsNgNe2D, ncolsNgNe2D);
  SetValue( psi2D, Z_ZERO );
  SetValue( phi2D, Z_ZERO );

  SCALAPACK(pzgemr2d)(&Ng, &Ne, psiCol.Data(), &I_ONE, &I_ONE, desc_NgNe1DCol, 
      psi2D.Data(), &I_ONE, &I_ONE, desc_NgNe2D, &contxt1 );
  SCALAPACK(pzgemr2d)(&Ng, &Ne, phiCol.Data(), &I_ONE, &I_ONE, desc_NgNe1DCol, 
      phi2D.Data(), &I_ONE, &I_ONE, desc_NgNe2D, &contxt1 );

  CpxNumMat PpsiMu2D(nrowsNgNu2D, ncolsNgNu2D);
  CpxNumMat PphiMu2D(nrowsNgNu2D, ncolsNgNu2D);
  SetValue( PpsiMu2D, Z_ZERO );
  SetValue( PphiMu2D, Z_ZERO );

  SCALAPACK(pzgemm)("N", "N", &Ng, &Nu, &Ne, 
      &Z_ONE,
      psi2D.Data(), &I_ONE, &I_ONE, desc_NgNe2D,
      psiMu2D.Data(), &I_ONE, &I_ONE, desc_NeNu2D, 
      &Z_ZERO,
      PpsiMu2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D);

  SCALAPACK(pzgemm)("N", "N", &Ng, &Nu, &Ne, 
      &Z_ONE,
      phi2D.Data(), &I_ONE, &I_ONE, desc_NgNe2D,
      phiMu2D.Data(), &I_ONE, &I_ONE, desc_NeNu2D, 
      &Z_ZERO,
      PphiMu2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D);

  GetTime( timeEnd1 );

#if ( _DEBUGlevel_ >= 0 )
  statusOFS << "Time for PpsiMu and PphiMu GEMM is " <<
    timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
#endif

  GetTime( timeSta1 );

  CpxNumMat Xi2D(nrowsNgNu2D, ncolsNgNu2D);
  SetValue( Xi2D, Z_ZERO );

  Complex* Xi2DPtr = Xi2D.Data();
  Complex* PpsiMu2DPtr = PpsiMu2D.Data();
  Complex* PphiMu2DPtr = PphiMu2D.Data();
  
  //Complex t_temp2;
  for( Int g = 0; g < nrowsNgNu2D * ncolsNgNu2D; g++ ){
    //t_temp2.real(PpsiMu2DPtr[g].real() * PphiMu2DPtr[g].real() - PpsiMu2DPtr[g].imag()*PphiMu2DPtr[g].imag());
    //t_temp2.imag(PpsiMu2DPtr[g].real() * PphiMu2DPtr[g].imag() + PpsiMu2DPtr[g].imag()* PphiMu2DPtr[g].real());
    //Xi2DPtr[g] = t_temp2;
    //Xi2DPtr[g] = PpsiMu2DPtr[g] * PphiMu2DPtr[g];
    Xi2DPtr[g] = PpsiMu2DPtr[g] * std::conj(PphiMu2DPtr[g]);
  }

  CpxNumMat Xi1D(nrowsNgNu1D, ncolsNuNu1D);
  SetValue( Xi1D, Z_ZERO );

  SCALAPACK(pzgemr2d)(&Ng, &Nu, Xi2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D, 
      Xi1D.Data(), &I_ONE, &I_ONE, desc_NgNu1D, &contxt2 );

  CpxNumMat PMuNu1D(nrowsNuNu1D, ncolsNuNu1D);
  SetValue( PMuNu1D, Z_ZERO );

  for (Int mu=0; mu<nrowsNuNu1D; mu++) {
    for (Int nu=0; nu<ncolsNuNu1D; nu++) {
      PMuNu1D(mu, nu) = Xi1D(pivMu1(mu),nu);
    }
  }

  CpxNumMat PMuNu2D(nrowsNuNu2D, ncolsNuNu2D);
  SetValue( PMuNu2D, Z_ZERO );

  SCALAPACK(pzgemr2d)(&Nu, &Nu, PMuNu1D.Data(), &I_ONE, &I_ONE, desc_NuNu1D, 
      PMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D, &contxt1 );

  GetTime( timeEnd1 );

#if ( _DEBUGlevel_ >= 0 )
  statusOFS << "Time for computing PMuNu is " <<
    timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
#endif


  //Method 1
  if(1){

    GetTime( timeSta1 );

    SCALAPACK(pzpotrf)("L", &Nu, PMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D, &info2);

    GetTime( timeEnd1 );

#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for PMuNu Potrf is " <<
      timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
#endif

    GetTime( timeSta1 );

    SCALAPACK(pztrsm)("R", "L", "C", "N", &Ng, &Nu, &Z_ONE,
        PMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D,
        Xi2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D);

    SCALAPACK(pztrsm)("R", "L", "N", "N", &Ng, &Nu, &Z_ONE,
        PMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D,
        Xi2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D);

    GetTime( timeEnd1 );

#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for PMuNu and Xi pdtrsm is " <<
      timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
#endif

  }


  //Method 2
  if(0){

    GetTime( timeSta1 );

    SCALAPACK(pzpotrf)("L", &Nu, PMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D, &info2);
    SCALAPACK(pzpotri)("L", &Nu, PMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D, &info2);

    GetTime( timeEnd1 );

#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for PMuNu Potrf is " <<
      timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
#endif

    GetTime( timeSta1 );

    CpxNumMat PMuNu2DTemp(nrowsNuNu2D, ncolsNuNu2D);
    SetValue( PMuNu2DTemp, Z_ZERO );

    lapack::Lacpy( 'A', nrowsNuNu2D, ncolsNuNu2D, PMuNu2D.Data(), nrowsNuNu2D, PMuNu2DTemp.Data(), nrowsNuNu2D );
//--------"T" TO "C" by lijl
    SCALAPACK(pztradd)("U", "C", &Nu, &Nu, 
        &Z_ONE,
        PMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D, 
        &Z_ZERO,
        PMuNu2DTemp.Data(), &I_ONE, &I_ONE, desc_NuNu2D);

    CpxNumMat Xi2DTemp(nrowsNgNu2D, ncolsNgNu2D);
    SetValue( Xi2DTemp, Z_ZERO );

    SCALAPACK(pzgemm)("N", "N", &Ng, &Nu, &Nu, 
        &Z_ONE,
        Xi2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D,
        PMuNu2DTemp.Data(), &I_ONE, &I_ONE, desc_NuNu2D, 
        &Z_ZERO,
        Xi2DTemp.Data(), &I_ONE, &I_ONE, desc_NgNu2D);

    SetValue( Xi2D, Z_ZERO );
    lapack::Lacpy( 'A', nrowsNgNu2D, ncolsNgNu2D, Xi2DTemp.Data(), nrowsNgNu2D, Xi2D.Data(), nrowsNgNu2D );

    GetTime( timeEnd1 );

#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for PMuNu and Xi pdgemm is " <<
      timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
#endif

  }


  //Method 3
  if(0){

    GetTime( timeSta1 );

    SCALAPACK(pzpotrf)("L", &Nu, PMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D, &info2);
    SCALAPACK(pzpotri)("L", &Nu, PMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D, &info2);

    GetTime( timeEnd1 );

#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for PMuNu Potrf is " <<
      timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
#endif

    GetTime( timeSta1 );

    CpxNumMat Xi2DTemp(nrowsNgNu2D, ncolsNgNu2D);
    SetValue( Xi2DTemp, Z_ZERO );

    SCALAPACK(pzsymm)("R", "L", &Ng, &Nu, 
        &Z_ONE,
        PMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D,
        Xi2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D,
        &Z_ZERO,
        Xi2DTemp.Data(), &I_ONE, &I_ONE, desc_NgNu2D);

    SetValue( Xi2D, Z_ZERO );
    lapack::Lacpy( 'A', nrowsNgNu2D, ncolsNgNu2D, Xi2DTemp.Data(), nrowsNgNu2D, Xi2D.Data(), nrowsNgNu2D );

    GetTime( timeEnd1 );

#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for PMuNu and Xi pdsymm is " <<
      timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
#endif

  }


  //Method 4
  if(0){

    GetTime( timeSta1 );

    CpxNumMat Xi2DTemp(nrowsNuNg2D, ncolsNuNg2D);
    SetValue( Xi2DTemp, Z_ZERO );

    CpxNumMat PMuNu2DTemp(ncolsNuNu2D, nrowsNuNu2D);
    SetValue( PMuNu2DTemp, Z_ZERO );

    SCALAPACK(pzgeadd)("T", &Nu, &Ng,
        &Z_ONE,
        Xi2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D,
        &Z_ZERO,
        Xi2DTemp.Data(), &I_ONE, &I_ONE, desc_NuNg2D);

    SCALAPACK(pzgeadd)("T", &Nu, &Nu,
        &Z_ONE,
        PMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D,
        &Z_ZERO,
        PMuNu2DTemp.Data(), &I_ONE, &I_ONE, desc_NuNu2D);

    Int lwork=-1, info;
    double dummyWork;

    SCALAPACK(pzgels)("N", &Nu, &Nu, &Ng, 
        PMuNu2DTemp.Data(), &I_ONE, &I_ONE, desc_NuNu2D,
        Xi2DTemp.Data(), &I_ONE, &I_ONE, desc_NuNg2D,
        &dummyWork, &lwork, &info);

    lwork = dummyWork;
    std::vector<double> work(lwork);

    SCALAPACK(pzgels)("N", &Nu, &Nu, &Ng, 
        PMuNu2DTemp.Data(), &I_ONE, &I_ONE, desc_NuNu2D,
        Xi2DTemp.Data(), &I_ONE, &I_ONE, desc_NuNg2D,
        &work[0], &lwork, &info);

    SetValue( Xi2D, Z_ZERO );
    SCALAPACK(pzgeadd)("T", &Ng, &Nu,
        &Z_ONE,
        Xi2DTemp.Data(), &I_ONE, &I_ONE, desc_NuNg2D,
        &Z_ZERO,
        Xi2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D);

    GetTime( timeEnd1 );

#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for PMuNu and Xi pdgels is " <<
      timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
#endif

  }


  GetTime( timeEnd );

#if ( _DEBUGlevel_ >= 0 )
  statusOFS << "Time for computing the interpolation vectors is " <<
    timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif



  // *********************************************************************
  // Solve the Poisson equations.
  // Store VXi separately. This is not the most memory efficient
  // implementation
  // *********************************************************************

  CpxNumMat VXi2D(nrowsNgNu2D, ncolsNgNu2D);

  {
    GetTime( timeSta );
    // XiCol used for both input and output
    CpxNumMat XiCol(ntot, numMuLocal);
    SetValue(XiCol, Z_ZERO );

    SCALAPACK(pzgemr2d)(&Ng, &Nu, Xi2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D, 
        XiCol.Data(), &I_ONE, &I_ONE, desc_NgNu1D, &contxt2 );

    GetTime( timeSta );
    for( Int mu = 0; mu < numMuLocal; mu++ ){
//----------fft.inputVecR2C to fft.inputComplexVec
      blas::Copy( ntot,  XiCol.VecData(mu), 1, fft.inputComplexVec.Data(), 1 );

      FFTWExecute ( fft, fft.forwardPlan );

      for( Int ig = 0; ig < ntot; ig++ ){
        fft.outputComplexVec(ig) *= -exxFraction * exxgkkR2C(ig);
      }

      FFTWExecute ( fft, fft.backwardPlan );

      blas::Copy( ntot, fft.inputComplexVec.Data(), 1, XiCol.VecData(mu), 1 );
//-------------------by lijl
    } // for (mu)

    SetValue(VXi2D, Z_ZERO );
    SCALAPACK(pzgemr2d)(&Ng, &Nu, XiCol.Data(), &I_ONE, &I_ONE, desc_NgNu1D, 
        VXi2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D, &contxt1 );

    GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for solving Poisson-like equations is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
  }

  // Prepare for the computation of the M matrix
  GetTime( timeSta );

  CpxNumMat MMatMuNu2D(nrowsNuNu2D, ncolsNuNu2D);
  SetValue(MMatMuNu2D, Z_ZERO );
//-----------"T" change to "C" by lijjl 20200422
  SCALAPACK(pzgemm)("C", "N", &Nu, &Nu, &Ng, 
      &Z_MinusONE,
      Xi2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D,
      VXi2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D, 
      &Z_ZERO,
      MMatMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D);

  CpxNumMat phiMuNu2D(nrowsNuNu2D, ncolsNuNu2D);
  SCALAPACK(pzgemm)("C", "N", &Nu, &Nu, &Ne, 
      &Z_ONE,
      phiMu2D.Data(), &I_ONE, &I_ONE, desc_NeNu2D,
      phiMu2D.Data(), &I_ONE, &I_ONE, desc_NeNu2D, 
      &Z_ZERO,
      phiMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D);
//-----------------by lijl
  Complex* MMat2DPtr = MMatMuNu2D.Data();
  Complex* phi2DPtr  = phiMuNu2D.Data();

  for( Int g = 0; g < nrowsNuNu2D * ncolsNuNu2D; g++ ){
     MMat2DPtr[g] *= phi2DPtr[g];
  }

  GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
  statusOFS << "Time for preparing the M matrix is " <<
    timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

  // *********************************************************************
  // Compute the exchange potential and the symmetrized inner product
  // *********************************************************************

  GetTime( timeSta );

  // Rewrite VXi by VXi.*PcolPhi
  Complex* VXi2DPtr = VXi2D.Data();
  Complex* PphiMu2DPtr1 = PphiMu2D.Data();
  for( Int g = 0; g < nrowsNgNu2D * ncolsNgNu2D; g++ ){
    VXi2DPtr[g] *= PphiMu2DPtr1[g];
  }

  // NOTE: a3 must be zero in order to compute the M matrix later
  CpxNumMat a32D( nrowsNgNe2D, ncolsNgNe2D );
  SetValue( a32D, Z_ZERO );

  SCALAPACK(pzgemm)("N", "C", &Ng, &Ne, &Nu, 
      &Z_ONE,
      VXi2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D,
      psiMu2D.Data(), &I_ONE, &I_ONE, desc_NeNu2D, 
      &Z_ZERO,
      a32D.Data(), &I_ONE, &I_ONE, desc_NgNe2D);

  CpxNumMat a3Col( ntot, numStateLocal );
  SetValue(a3Col, Z_ZERO );

  SCALAPACK(pzgemr2d)(&Ng, &Ne, a32D.Data(), &I_ONE, &I_ONE, desc_NgNe2D, 
      a3Col.Data(), &I_ONE, &I_ONE, desc_NgNe1DCol, &contxt2 );

  lapack::Lacpy( 'A', ntot, numStateLocal, a3Col.Data(), ntot, a3.Data(), ntot );

  GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
  statusOFS << "Time for computing the exchange potential is " <<
    timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

  // Compute the matrix VxMat = -Psi'* vexxPsi and symmetrize
  // vexxPsi (a3) must be zero before entering this routine
  VxMat.Resize( numStateTotal, numStateTotal );

  GetTime( timeSta );

  if(1){

    CpxNumMat VxMat2D( nrowsNeNe2D, ncolsNeNe2D );
    CpxNumMat VxMatTemp2D( nrowsNuNe2D, ncolsNuNe2D );

    SCALAPACK(pzgemm)("N", "C", &Nu, &Ne, &Nu, 
        &Z_ONE,
        MMatMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D,
        psiMu2D.Data(), &I_ONE, &I_ONE, desc_NeNu2D, 
        &Z_ZERO,
        VxMatTemp2D.Data(), &I_ONE, &I_ONE, desc_NuNe2D);

    SCALAPACK(pzgemm)("N", "N", &Ne, &Ne, &Nu, 
        &Z_ONE,
        psiMu2D.Data(), &I_ONE, &I_ONE, desc_NeNu2D, 
        VxMatTemp2D.Data(), &I_ONE, &I_ONE, desc_NuNe2D, 
        &Z_ZERO,
        VxMat2D.Data(), &I_ONE, &I_ONE, desc_NeNe2D);

    SCALAPACK(pzgemr2d)(&Ne, &Ne, VxMat2D.Data(), &I_ONE, &I_ONE, desc_NeNe2D, 
        VxMat.Data(), &I_ONE, &I_ONE, desc_NeNe0D, &contxt2 );

    //if(mpirank == 0){
    //  MPI_Bcast( VxMat.Data(), Ne * Ne, MPI_DOUBLE, 0, domain_.comm );
    //}

  }
  GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
  statusOFS << "Time for computing VxMat in the sym format is " <<
    timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

  if(contxt0 >= 0) {
    Cblacs_gridexit( contxt0 );
  }

  if(contxt1 >= 0) {
    Cblacs_gridexit( contxt1 );
  }

  if(contxt11 >= 0) {
    Cblacs_gridexit( contxt11 );
  }

  if(contxt2 >= 0) {
    Cblacs_gridexit( contxt2 );
  }

  MPI_Barrier(domain_.comm);

  return ;
}        // -----  end of method Spinor::AddMultSpinorEXXDF7  ----- 

#else

Spinor::Spinor ( 
    const Domain &dm, 
    const Int     numComponent,
    const Int     numStateTotal,
    Int     numStateLocal,
    const Real  val ) {
  this->Setup( dm, numComponent, numStateTotal, numStateLocal, val );

}         // -----  end of method Spinor::Spinor  ----- 

Spinor::Spinor ( const Domain &dm, 
    const Int numComponent, 
    const Int numStateTotal,
    Int numStateLocal,
    const bool owndata, 
    Real* data )
{
  this->Setup( dm, numComponent, numStateTotal, numStateLocal, owndata, data );

}         // -----  end of method Spinor::Spinor  ----- 

void Spinor::Setup ( 
    const Domain &dm, 
    const Int     numComponent,
    const Int     numStateTotal,
    Int     numStateLocal,
    const Real  val ) {

  domain_       = dm;
  MPI_Barrier(domain_.comm);
  int mpirank;  MPI_Comm_rank(domain_.comm, &mpirank);
  int mpisize;  MPI_Comm_size(domain_.comm, &mpisize);

  Int blocksize;

  if ( numStateTotal <=  mpisize ) {
    blocksize = 1;
  }
  else {  // numStateTotal >  mpisize
    if ( numStateTotal % mpisize == 0 ){
      blocksize = numStateTotal / mpisize;
    }
    else {
      blocksize = ((numStateTotal - 1) / mpisize) + 1;
    }    
  }

  numStateTotal_ = numStateTotal;
  blocksize_ = blocksize;

  wavefunIdx_.Resize( numStateLocal );
  SetValue( wavefunIdx_, 0 );
  for (Int i = 0; i < numStateLocal; i++){
    wavefunIdx_[i] = i * mpisize + mpirank ;
  }

  wavefun_.Resize( dm.NumGridTotal(), numComponent, numStateLocal );
  SetValue( wavefun_, val );

}         // -----  end of method Spinor::Setup  ----- 

void Spinor::Setup ( const Domain &dm, 
    const Int numComponent, 
    const Int numStateTotal,
    Int numStateLocal,
    const bool owndata, 
    Real* data )
{

  domain_       = dm;
  MPI_Barrier(domain_.comm);
  int mpirank;  MPI_Comm_rank(domain_.comm, &mpirank);
  int mpisize;  MPI_Comm_size(domain_.comm, &mpisize);

  wavefun_      = NumTns<Real>( dm.NumGridTotal(), numComponent, numStateLocal,
      owndata, data );

  Int blocksize;

  if ( numStateTotal <=  mpisize ) {
    blocksize = 1;
  }
  else {  // numStateTotal >  mpisize
    if ( numStateTotal % mpisize == 0 ){
      blocksize = numStateTotal / mpisize;
    }
    else {
      blocksize = ((numStateTotal - 1) / mpisize) + 1;
    }    
  }

  numStateTotal_ = numStateTotal;
  blocksize_ = blocksize;

  wavefunIdx_.Resize( numStateLocal );
  SetValue( wavefunIdx_, 0 );
  for (Int i = 0; i < numStateLocal; i++){
    wavefunIdx_[i] = i * mpisize + mpirank ;
  }

}         // -----  end of method Spinor::Setup  ----- 

void
Spinor::Normalize    ( )
{
  Int size = wavefun_.m() * wavefun_.n();
  Int nocc = wavefun_.p();

  for (Int k=0; k<nocc; k++) {
    Real *ptr = wavefun_.MatData(k);
    Real   sum = 0.0;
    for (Int i=0; i<size; i++) {
      sum += pow(abs(*ptr++), 2.0);
    }
    sum = sqrt(sum);
    if (sum != 0.0) {
      ptr = wavefun_.MatData(k);
      for (Int i=0; i<size; i++) *(ptr++) /= sum;
    }
  }
  return ;
}         // -----  end of method Spinor::Normalize  ----- 

void
Spinor::AddTeterPrecond (Fourier* fftPtr, NumTns<Real>& a3)
{
  Fourier& fft = *fftPtr;
  if( !fftPtr->isInitialized ){
    ErrorHandling("Fourier is not prepared.");
  }
  Int ntot = wavefun_.m();
  Int ncom = wavefun_.n();
  Int nocc = wavefun_.p();

  if( fftPtr->domain.NumGridTotal() != ntot ){
    ErrorHandling("Domain size does not match1.");
  }

  //#ifdef _USE_OPENMP_
  //#pragma omp parallel
  //  {
  //#endif
  Int ntothalf = fftPtr->numGridTotalR2C;
  // These two are private variables in the OpenMP context

  //#ifdef _USE_OPENMP_
  //#pragma omp for schedule (dynamic,1) nowait
  //#endif
  for (Int k=0; k<nocc; k++) {
    for (Int j=0; j<ncom; j++) {
      // For c2r and r2c transforms, the default is to DESTROY the
      // input, therefore a copy of the original matrix is necessary. 
      blas::Copy( ntot, wavefun_.VecData(j,k), 1, 
          reinterpret_cast<Real*>(fft.inputVecR2C.Data()), 1 );

      FFTWExecute ( fft, fft.forwardPlanR2C );
      //          fftw_execute_dft_r2c(
      //                  fftPtr->forwardPlanR2C, 
      //                  realInVec.Data(),
      //                  reinterpret_cast<fftw_complex*>(cpxOutVec.Data() ));

      Real*    ptr1d   = fftPtr->TeterPrecondR2C.Data();
      Complex* ptr2    = fft.outputVecR2C.Data();
      for (Int i=0; i<ntothalf; i++) 
        *(ptr2++) *= *(ptr1d++);

      FFTWExecute ( fft, fft.backwardPlanR2C);
      //          fftw_execute_dft_c2r(
      //                  fftPtr->backwardPlanR2C,
      //                  reinterpret_cast<fftw_complex*>(cpxOutVec.Data() ),
      //                  realInVec.Data() );
      //          blas::Axpy( ntot, 1.0 / Real(ntot), realInVec.Data(), 1, 
      //                  a3o.VecData(j, k), 1 );

      blas::Axpy( ntot, 1.0, fft.inputVecR2C.Data(), 1, a3.VecData(j,k), 1 );
    }
  }
  //#ifdef _USE_OPENMP_
  //  }
  //#endif


  return ;
}         // -----  end of method Spinor::AddTeterPrecond ----- 

void
Spinor::AddMultSpinorFine ( Fourier& fft, const DblNumVec& vtot, 
    const std::vector<PseudoPot>& pseudo, NumTns<Real>& a3 )
{
  // TODO Complex case

  if( !fft.isInitialized ){
    ErrorHandling("Fourier is not prepared.");
  }
  Int ntot = wavefun_.m();
  Int ncom = wavefun_.n();
  Int numStateLocal = wavefun_.p();
  Int ntotFine = domain_.NumGridTotalFine();
  Real vol = domain_.Volume();

  if( fft.domain.NumGridTotal() != ntot ){
    ErrorHandling("Domain size does not match2.");
  }

  // Temporary variable for saving wavefunction on a fine grid
  DblNumVec psiFine(ntotFine);
  DblNumVec psiUpdateFine(ntotFine);

  for (Int k=0; k<numStateLocal; k++) {
    for (Int j=0; j<ncom; j++) {

      SetValue( psiFine, 0.0 );
      SetValue( psiUpdateFine, 0.0 );

      // Fourier transform

      //      for( Int i = 0; i < ntot; i++ ){
      //        fft.inputComplexVec(i) = Complex( wavefun_(i,j,k), 0.0 ); 
      //      }
      SetValue( fft.inputComplexVec, Z_ZERO );
      blas::Copy( ntot, wavefun_.VecData(j,k), 1,
          reinterpret_cast<Real*>(fft.inputComplexVec.Data()), 2 );

      // Fourier transform of wavefunction saved in fft.outputComplexVec
      fftw_execute( fft.forwardPlan );

      // Interpolate wavefunction from coarse to fine grid
      {
        SetValue( fft.outputComplexVecFine, Z_ZERO ); 
        Int *idxPtr = fft.idxFineGrid.Data();
        Complex *fftOutFinePtr = fft.outputComplexVecFine.Data();
        Complex *fftOutPtr = fft.outputComplexVec.Data();
        for( Int i = 0; i < ntot; i++ ){
          //          fft.outputComplexVecFine(fft.idxFineGrid(i)) = fft.outputComplexVec(i);
          fftOutFinePtr[*(idxPtr++)] = *(fftOutPtr++);
        }
      }
      fftw_execute( fft.backwardPlanFine );
      Real fac = 1.0 / std::sqrt( double(domain_.NumGridTotal())  *
          double(domain_.NumGridTotalFine()) ); 
      //      for( Int i = 0; i < ntotFine; i++ ){
      //        psiFine(i) = fft.inputComplexVecFine(i).real() * fac; 
      //      }
      blas::Copy( ntotFine, reinterpret_cast<Real*>(fft.inputComplexVecFine.Data()),
          2, psiFine.Data(), 1 );
      blas::Scal( ntotFine, fac, psiFine.Data(), 1 );

      // Add the contribution from local pseudopotential
      //      for( Int i = 0; i < ntotFine; i++ ){
      //        psiUpdateFine(i) += psiFine(i) * vtot(i);
      //      }
      {
        Real *psiUpdateFinePtr = psiUpdateFine.Data();
        Real *psiFinePtr = psiFine.Data();
        Real *vtotPtr = vtot.Data();
        for( Int i = 0; i < ntotFine; i++ ){
          *(psiUpdateFinePtr++) += *(psiFinePtr++) * *(vtotPtr++);
        }
      }

      // Add the contribution from nonlocal pseudopotential
      if(1){
        Int natm = pseudo.size();
        for (Int iatm=0; iatm<natm; iatm++) {
          Int nobt = pseudo[iatm].vnlList.size();
          for (Int iobt=0; iobt<nobt; iobt++) {
            const Real       vnlwgt = pseudo[iatm].vnlList[iobt].second;
            const SparseVec &vnlvec = pseudo[iatm].vnlList[iobt].first;
            const IntNumVec &iv = vnlvec.first;
            const DblNumMat &dv = vnlvec.second;

            Real    weight = 0.0; 
            const Int    *ivptr = iv.Data();
            const Real   *dvptr = dv.VecData(VAL);
            for (Int i=0; i<iv.m(); i++) {
              weight += (*(dvptr++)) * psiFine[*(ivptr++)];
            }
            weight *= vol/Real(ntotFine)*vnlwgt;

            ivptr = iv.Data();
            dvptr = dv.VecData(VAL);
            for (Int i=0; i<iv.m(); i++) {
              psiUpdateFine[*(ivptr++)] += (*(dvptr++)) * weight;
            }
          } // for (iobt)
        } // for (iatm)
      }


      // Laplacian operator. Perform inverse Fourier transform in the end
      {
        for (Int i=0; i<ntot; i++) 
          fft.outputComplexVec(i) *= fft.gkk(i);
      }

      // Restrict psiUpdateFine from fine grid in the real space to
      // coarse grid in the Fourier space. Combine with the Laplacian contribution
      //      for( Int i = 0; i < ntotFine; i++ ){
      //        fft.inputComplexVecFine(i) = Complex( psiUpdateFine(i), 0.0 ); 
      //      }
      SetValue( fft.inputComplexVecFine, Z_ZERO );
      blas::Copy( ntotFine, psiUpdateFine.Data(), 1,
          reinterpret_cast<Real*>(fft.inputComplexVecFine.Data()), 2 );

      // Fine to coarse grid
      // Note the update is important since the Laplacian contribution is already taken into account.
      // The computation order is also important
      fftw_execute( fft.forwardPlanFine );
      {
        Real fac = std::sqrt(Real(ntot) / (Real(ntotFine)));
        Int* idxPtr = fft.idxFineGrid.Data();
        Complex *fftOutFinePtr = fft.outputComplexVecFine.Data();
        Complex *fftOutPtr = fft.outputComplexVec.Data();

        for( Int i = 0; i < ntot; i++ ){
          //          fft.outputComplexVec(i) += fft.outputComplexVecFine(fft.idxFineGrid(i)) * fac;
          *(fftOutPtr++) += fftOutFinePtr[*(idxPtr++)] * fac;
        }
      }

      // Inverse Fourier transform to save back to the output vector
      fftw_execute( fft.backwardPlan );

      //      Real    *ptr1 = a3.VecData(j,k);
      //      for( Int i = 0; i < ntot; i++ ){
      //        ptr1[i] += fft.inputComplexVec(i).real() / Real(ntot);
      //      }
      blas::Axpy( ntot, 1.0 / Real(ntot), 
          reinterpret_cast<Real*>(fft.inputComplexVec.Data()), 2,
          a3.VecData(j,k), 1 );
    }
  }



  return ;
}        // -----  end of method Spinor::AddMultSpinorFine  ----- 

void
Spinor::AddMultSpinorFineR2C ( Fourier& fft, const DblNumVec& vtot, 
    const std::vector<PseudoPot>& pseudo, NumTns<Real>& a3 )
{

  if( !fft.isInitialized ){
    ErrorHandling("Fourier is not prepared.");
  }
  Index3& numGrid = domain_.numGrid;
  Index3& numGridFine = domain_.numGridFine;
  Int ntot = wavefun_.m();
  Int ncom = wavefun_.n();
  Int numStateLocal = wavefun_.p();
  Int ntotFine = domain_.NumGridTotalFine();
  Real vol = domain_.Volume();

  Int ntotR2C = fft.numGridTotalR2C;
  Int ntotR2CFine = fft.numGridTotalR2CFine;

  if( fft.domain.NumGridTotal() != ntot ){
    ErrorHandling("Domain size does not match3.");
  }

  // Temporary variable for saving wavefunction on a fine grid
  DblNumVec psiFine(ntotFine);
  DblNumVec psiUpdateFine(ntotFine);

  Real timeSta, timeEnd;
  Real timeSta1, timeEnd1;
  
  Real timeFFTCoarse = 0.0;
  Real timeFFTFine = 0.0;
  Real timeNonlocal = 0.0;
  Real timeOther = 0.0;
  Int  iterFFTCoarse = 0;
  Int  iterFFTFine = 0;
  Int  iterNonlocal = 0;
  Int  iterOther = 0;

  GetTime( timeSta1 );
 
  if(0)
  {
    for (Int k=0; k<numStateLocal; k++) {
      for (Int j=0; j<ncom; j++) {

        SetValue( fft.inputVecR2C, 0.0 );
        SetValue( fft.outputVecR2C, Z_ZERO );

        blas::Copy( ntot, wavefun_.VecData(j,k), 1,
            fft.inputVecR2C.Data(), 1 );
        FFTWExecute ( fft, fft.forwardPlanR2C ); // So outputVecR2C contains the FFT result now


        for (Int i=0; i<ntotR2C; i++)
        {
          if(fft.gkkR2C(i) > 5.0)
            fft.outputVecR2C(i) = Z_ZERO;
        }

        FFTWExecute ( fft, fft.backwardPlanR2C );
        blas::Copy( ntot,  fft.inputVecR2C.Data(), 1,
            wavefun_.VecData(j,k), 1 );

      }
    }
  }



  //#ifdef _USE_OPENMP_
  //#pragma omp parallel
  //    {
  //#endif
  for (Int k=0; k<numStateLocal; k++) {
    for (Int j=0; j<ncom; j++) {

      SetValue( psiFine, 0.0 );
      SetValue( psiUpdateFine, 0.0 );

      // R2C version
      if(1)
      {
        SetValue( fft.inputVecR2C, 0.0 ); 
        SetValue( fft.inputVecR2CFine, 0.0 ); 
        SetValue( fft.outputVecR2C, Z_ZERO ); 
        SetValue( fft.outputVecR2CFine, Z_ZERO ); 


        // For c2r and r2c transforms, the default is to DESTROY the
        // input, therefore a copy of the original matrix is necessary. 
        blas::Copy( ntot, wavefun_.VecData(j,k), 1, 
            fft.inputVecR2C.Data(), 1 );

        GetTime( timeSta );
        FFTWExecute ( fft, fft.forwardPlanR2C );
        GetTime( timeEnd );
        iterFFTCoarse = iterFFTCoarse + 1;
        timeFFTCoarse = timeFFTCoarse + ( timeEnd - timeSta );

        // statusOFS << std::endl << " Input vec = " << fft.inputVecR2C << std::endl;
        // statusOFS << std::endl << " Output vec = " << fft.outputVecR2C << std::endl;


        // Interpolate wavefunction from coarse to fine grid
//1. coarse to fine
// The FFT number is even(esdf.cpp now), change as follows

        if(  esdfParam.FFTtype == "even" || esdfParam.FFTtype == "power")
        {
            Complex *fftOutFinePtr = fft.outputVecR2CFine.Data();
            Complex *fftOutPtr = fft.outputVecR2C.Data();
            IP_c2f(numGrid.Data(),numGridFine.Data(),fftOutPtr,fftOutFinePtr);
        }
        else if( esdfParam.FFTtype == "odd" )
// odd version
        {                          
            Real fac = sqrt( double(ntot) / double(ntotFine) );    
            Int *idxPtr = fft.idxFineGridR2C.Data();               
            Complex *fftOutFinePtr = fft.outputVecR2CFine.Data();  
            Complex *fftOutPtr = fft.outputVecR2C.Data();          
            for( Int i = 0; i < ntotR2C; i++ ){                    
              fftOutFinePtr[*(idxPtr++)] = *(fftOutPtr++) * fac; 
            }                                                   
        }                             
      
        GetTime( timeSta );
        FFTWExecute ( fft, fft.backwardPlanR2CFine );
        GetTime( timeEnd );
        iterFFTFine = iterFFTFine + 1;
        timeFFTFine = timeFFTFine + ( timeEnd - timeSta );

        blas::Copy( ntotFine, fft.inputVecR2CFine.Data(), 1, psiFine.Data(), 1 );

      }  // if (1)

      // Add the contribution from local pseudopotential
      //      for( Int i = 0; i < ntotFine; i++ ){
      //        psiUpdateFine(i) += psiFine(i) * vtot(i);
      //      }
      if(_LOCAL_){
        Real *psiUpdateFinePtr = psiUpdateFine.Data();
        Real *psiFinePtr = psiFine.Data();
        Real *vtotPtr = vtot.Data();
        for( Int i = 0; i < ntotFine; i++ ){
          *(psiUpdateFinePtr++) += *(psiFinePtr++) * *(vtotPtr++);
        }
      }

      // Add the contribution from nonlocal pseudopotential
      GetTime( timeSta );
      if(_NON_LOCAL_){
        Int natm = pseudo.size();
        for (Int iatm=0; iatm<natm; iatm++) {
          Int nobt = pseudo[iatm].vnlList.size();
          for (Int iobt=0; iobt<nobt; iobt++) {
            const Real       vnlwgt = pseudo[iatm].vnlList[iobt].second;
            const SparseVec &vnlvec = pseudo[iatm].vnlList[iobt].first;
            const IntNumVec &iv = vnlvec.first;
            const DblNumMat &dv = vnlvec.second;

            Real    weight = 0.0; 
            const Int    *ivptr = iv.Data();
            const Real   *dvptr = dv.VecData(VAL);
            for (Int i=0; i<iv.m(); i++) {
              weight += (*(dvptr++)) * psiFine[*(ivptr++)];
            }
            weight *= vol/Real(ntotFine)*vnlwgt;

            ivptr = iv.Data();
            dvptr = dv.VecData(VAL);
            for (Int i=0; i<iv.m(); i++) {
              psiUpdateFine[*(ivptr++)] += (*(dvptr++)) * weight;
            }
          } // for (iobt)
        } // for (iatm)
      }
      GetTime( timeEnd );
      iterNonlocal = iterNonlocal + 1;
      timeNonlocal = timeNonlocal + ( timeEnd - timeSta );

      // Laplacian operator. Perform inverse Fourier transform in the end
      if(_EKIN_){
        for (Int i=0; i<ntotR2C; i++) 
          fft.outputVecR2C(i) *= fft.gkkR2C(i);
      }

      // Restrict psiUpdateFine from fine grid in the real space to
      // coarse grid in the Fourier space. Combine with the Laplacian contribution
      //      for( Int i = 0; i < ntotFine; i++ ){
      //        fft.inputComplexVecFine(i) = Complex( psiUpdateFine(i), 0.0 ); 
      //      }
      //SetValue( fft.inputComplexVecFine, Z_ZERO );
      SetValue( fft.inputVecR2CFine, 0.0 );
      blas::Copy( ntotFine, psiUpdateFine.Data(), 1, fft.inputVecR2CFine.Data(), 1 );

      // Fine to coarse grid
      // Note the update is important since the Laplacian contribution is already taken into account.
      // The computation order is also important
      // fftw_execute( fft.forwardPlanFine );
      GetTime( timeSta );
      FFTWExecute ( fft, fft.forwardPlanR2CFine );
      GetTime( timeEnd );
      iterFFTFine = iterFFTFine + 1;
      timeFFTFine = timeFFTFine + ( timeEnd - timeSta );
      //      {
      //        Real fac = std::sqrt(Real(ntot) / (Real(ntotFine)));
      //        Int* idxPtr = fft.idxFineGrid.Data();
      //        Complex *fftOutFinePtr = fft.outputComplexVecFine.Data();
      //        Complex *fftOutPtr = fft.outputComplexVec.Data();
      //
      //        for( Int i = 0; i < ntot; i++ ){
      //          //          fft.outputComplexVec(i) += fft.outputComplexVecFine(fft.idxFineGrid(i)) * fac;
      //          *(fftOutPtr++) += fftOutFinePtr[*(idxPtr++)] * fac;
      //        }
      //      }


//2. fine to coarse
     if(  esdfParam.FFTtype == "even" || esdfParam.FFTtype == "power")
     {
        Complex *fftOutFinePtr = fft.outputVecR2CFine.Data();
        Complex *fftOutPtr = fft.outputVecR2C.Data();
        IP_f2c(numGrid.Data(),numGridFine.Data(),fftOutPtr,fftOutFinePtr);
     }
     else if(  esdfParam.FFTtype == "odd")
     {
        Real fac = sqrt( double(ntotFine) / double(ntot) );
        Int *idxPtr = fft.idxFineGridR2C.Data();
        Complex *fftOutFinePtr = fft.outputVecR2CFine.Data();
        Complex *fftOutPtr = fft.outputVecR2C.Data();
        for( Int i = 0; i < ntotR2C; i++ ){
          *(fftOutPtr++) += fftOutFinePtr[*(idxPtr++)] * fac;
        }
      }
      GetTime( timeSta );
      FFTWExecute ( fft, fft.backwardPlanR2C );
      GetTime( timeEnd );
      iterFFTCoarse = iterFFTCoarse + 1;
      timeFFTCoarse = timeFFTCoarse + ( timeEnd - timeSta );

      // Inverse Fourier transform to save back to the output vector
      //fftw_execute( fft.backwardPlan );

      blas::Axpy( ntot, 1.0, fft.inputVecR2C.Data(), 1, a3.VecData(j,k), 1 );

    } // j++
  } // k++
  //#ifdef _USE_OPENMP_
  //    }
  //#endif

  if(0)
  {
    for (Int k=0; k<numStateLocal; k++) {
      for (Int j=0; j<ncom; j++) {

        SetValue( fft.inputVecR2C, 0.0 );
        SetValue( fft.outputVecR2C, Z_ZERO );

        blas::Copy( ntot, a3.VecData(j,k), 1,
            fft.inputVecR2C.Data(), 1 );
      
        GetTime( timeSta );
        FFTWExecute ( fft, fft.forwardPlanR2C ); // So outputVecR2C contains the FFT result now
        GetTime( timeEnd );
        iterFFTCoarse = iterFFTCoarse + 1;
        timeFFTCoarse = timeFFTCoarse + ( timeEnd - timeSta );

        for (Int i=0; i<ntotR2C; i++)
        {
          if(fft.gkkR2C(i) > 5.0)
            fft.outputVecR2C(i) = Z_ZERO;
        }

        GetTime( timeSta );
        FFTWExecute ( fft, fft.backwardPlanR2C );
        GetTime( timeEnd );
        iterFFTCoarse = iterFFTCoarse + 1;
        timeFFTCoarse = timeFFTCoarse + ( timeEnd - timeSta );
        
        blas::Copy( ntot,  fft.inputVecR2C.Data(), 1,
            a3.VecData(j,k), 1 );

      }
    }
  }

  GetTime( timeEnd1 );
  iterOther = iterOther + 1;
  timeOther = timeOther + ( timeEnd1 - timeSta1 ) - timeFFTCoarse - timeFFTFine - timeNonlocal;

#if ( _DEBUGlevel_ >= 1 )
    statusOFS << "Time for iterFFTCoarse    = " << iterFFTCoarse       << "  timeFFTCoarse    = " << timeFFTCoarse << std::endl;
    statusOFS << "Time for iterFFTFine      = " << iterFFTFine         << "  timeFFTFine      = " << timeFFTFine << std::endl;
    statusOFS << "Time for iterNonlocal     = " << iterNonlocal        << "  timeNonlocal     = " << timeNonlocal << std::endl;
    statusOFS << "Time for iterOther        = " << iterOther           << "  timeOther        = " << timeOther << std::endl;
#endif

  return ;
}        // -----  end of method Spinor::AddMultSpinorFineR2C  ----- 

void Spinor::TransformCoarseToFine( Fourier& fft ){

  if( !fft.isInitialized ){
    ErrorHandling("Fourier is not prepared.");
  }
  Index3& numGrid = domain_.numGrid;
  Index3& numGridFine = domain_.numGridFine;
  Int ntot = wavefun_.m();
  Int ncom = wavefun_.n();
  Int numStateLocal = wavefun_.p();
  Int ntotFine = domain_.NumGridTotalFine();
  Real vol = domain_.Volume();

  Int ntotR2C = fft.numGridTotalR2C;
  Int ntotR2CFine = fft.numGridTotalR2CFine;

  if( fft.domain.NumGridTotal() != ntot ){
    ErrorHandling("Domain size does not match3.");
  }

  wavefunFine_.Resize( numGridFine.prod(), ncom, numStateLocal );

  for (Int k=0; k<numStateLocal; k++) {
    for (Int j=0; j<ncom; j++) {
      
      DblNumVec psiFine = DblNumVec( ntotFine, false, wavefunFine_.VecData(j,k) );

      SetValue( fft.inputVecR2C, 0.0 );
      SetValue( fft.inputVecR2CFine, 0.0 );
      SetValue( fft.outputVecR2C, Z_ZERO );
      SetValue( fft.outputVecR2CFine, Z_ZERO );

      blas::Copy( ntot, wavefun_.VecData(j,k), 1,
            fft.inputVecR2C.Data(), 1 );

      FFTWExecute ( fft, fft.forwardPlanR2C );

      if(  esdfParam.FFTtype == "even" || esdfParam.FFTtype == "power")
      {
        Complex *fftOutFinePtr = fft.outputVecR2CFine.Data();
        Complex *fftOutPtr = fft.outputVecR2C.Data();
        IP_c2f(numGrid.Data(),numGridFine.Data(),fftOutPtr,fftOutFinePtr);
      }
      else if( esdfParam.FFTtype == "odd" ){
        Real fac = sqrt( double(ntot) / double(ntotFine) );
        Int *idxPtr = fft.idxFineGridR2C.Data();
        Complex *fftOutFinePtr = fft.outputVecR2CFine.Data();
        Complex *fftOutPtr = fft.outputVecR2C.Data();
        for( Int i = 0; i < ntotR2C; i++ ){
          fftOutFinePtr[*(idxPtr++)] = *(fftOutPtr++) * fac;
        }
      }

      FFTWExecute ( fft, fft.backwardPlanR2CFine );

      blas::Copy( ntotFine, fft.inputVecR2CFine.Data(), 1, psiFine.Data(), 1 );
    }
  }
}

void Spinor::AddMultSpinorEXX ( Fourier& fft, 
    const NumTns<Real>& phi,
    const DblNumVec& exxgkkR2C,
    Real  exxFraction,
    Real  numSpin,
    const DblNumVec& occupationRate,
    NumTns<Real>& a3 )
{
  if( !fft.isInitialized ){
    ErrorHandling("Fourier is not prepared.");
  }

  MPI_Barrier(domain_.comm);
  int mpirank;  MPI_Comm_rank(domain_.comm, &mpirank);
  int mpisize;  MPI_Comm_size(domain_.comm, &mpisize);

  Index3& numGrid = domain_.numGrid;
  Index3& numGridFine = domain_.numGridFine;

  Int ntot     = domain_.NumGridTotal();
  Int ntotFine = domain_.NumGridTotalFine();
  Int ntotR2C = fft.numGridTotalR2C;
  Int ntotR2CFine = fft.numGridTotalR2CFine;
  Int ncom = wavefun_.n();
  Int numStateLocal = wavefun_.p();
  Int numStateTotal = numStateTotal_;

  Int ncomPhi = phi.n();

  Real vol = domain_.Volume();

  if( ncomPhi != 1 || ncom != 1 ){
    ErrorHandling("Spin polarized case not implemented.");
  }

  if( fft.domain.NumGridTotal() != ntot ){
    ErrorHandling("Domain size does not match4.");
  }

  // Temporary variable for saving wavefunction on a fine grid
  DblNumVec phiTemp(ntot);

  Int numStateLocalTemp;

  MPI_Barrier(domain_.comm);

  for( Int iproc = 0; iproc < mpisize; iproc++ ){

    if( iproc == mpirank )
      numStateLocalTemp = numStateLocal;

    MPI_Bcast( &numStateLocalTemp, 1, MPI_INT, iproc, domain_.comm );

    IntNumVec wavefunIdxTemp(numStateLocalTemp);
    if( iproc == mpirank ){
      wavefunIdxTemp = wavefunIdx_;
    }

    MPI_Bcast( wavefunIdxTemp.Data(), numStateLocalTemp, MPI_INT, iproc, domain_.comm );

    // FIXME OpenMP does not work since all variables are shared
    for( Int kphi = 0; kphi < numStateLocalTemp; kphi++ ){
      for( Int jphi = 0; jphi < ncomPhi; jphi++ ){

        SetValue( phiTemp, 0.0 );

        if( iproc == mpirank )
        { 
          Real* phiPtr = phi.VecData(jphi, kphi);
          for( Int ir = 0; ir < ntot; ir++ ){
            phiTemp(ir) = phiPtr[ir];
          }
        }

        MPI_Bcast( phiTemp.Data(), ntot, MPI_DOUBLE, iproc, domain_.comm );

        for (Int k=0; k<numStateLocal; k++) {
          for (Int j=0; j<ncom; j++) {

            Real* psiPtr = wavefun_.VecData(j,k);
            for( Int ir = 0; ir < ntot; ir++ ){
              fft.inputVecR2C(ir) = psiPtr[ir] * phiTemp(ir);
            }

            FFTWExecute ( fft, fft.forwardPlanR2C );

            // Solve the Poisson-like problem for exchange
            for( Int ig = 0; ig < ntotR2C; ig++ ){
              fft.outputVecR2C(ig) *= exxgkkR2C(ig);
            }

            FFTWExecute ( fft, fft.backwardPlanR2C );

            Real* a3Ptr = a3.VecData(j,k);
            Real fac = -exxFraction * occupationRate[wavefunIdxTemp(kphi)];  
            for( Int ir = 0; ir < ntot; ir++ ){
              a3Ptr[ir] += fft.inputVecR2C(ir) * phiTemp(ir) * fac;
            }

          } // for (j)
        } // for (k)

        MPI_Barrier(domain_.comm);


      } // for (jphi)
    } // for (kphi)

  } //iproc

  MPI_Barrier(domain_.comm);


  return ;
}        // -----  end of method Spinor::AddMultSpinorEXX  ----- 


// This is the new density matrix based algorithm for compressing the Coulomb integrals
void Spinor::AddMultSpinorEXXDF ( Fourier& fft, 
    const NumTns<Real>& phi,
    const DblNumVec& exxgkkR2C,
    Real  exxFraction,
    Real  numSpin,
    const DblNumVec& occupationRate,
    const Real numMuFac,
    const Real numGaussianRandomFac,
    const Int numProcScaLAPACKPotrf,  
    const Int scaPotrfBlockSize,  
    NumTns<Real>& a3, 
    NumMat<Real>& VxMat,
    bool isFixColumnDF )
{
  Real timeSta, timeEnd;
  Real timeSta1, timeEnd1;

  if( !fft.isInitialized ){
    ErrorHandling("Fourier is not prepared.");
  }

  MPI_Barrier(domain_.comm);
  int mpirank;  MPI_Comm_rank(domain_.comm, &mpirank);
  int mpisize;  MPI_Comm_size(domain_.comm, &mpisize);

  Index3& numGrid = domain_.numGrid;
  Index3& numGridFine = domain_.numGridFine;

  Int ntot     = domain_.NumGridTotal();
  Int ntotFine = domain_.NumGridTotalFine();
  Int ntotR2C = fft.numGridTotalR2C;
  Int ntotR2CFine = fft.numGridTotalR2CFine;
  Int ncom = wavefun_.n();
  Int numStateLocal = wavefun_.p();
  Int numStateTotal = numStateTotal_;

  Int ncomPhi = phi.n();

  Real vol = domain_.Volume();

  if( ncomPhi != 1 || ncom != 1 ){
    ErrorHandling("Spin polarized case not implemented.");
  }

  if( fft.domain.NumGridTotal() != ntot ){
    ErrorHandling("Domain size does not match5.");
  }


//  if(0){
//
//    // *********************************************************************
//    // Perform interpolative separable density fitting
//    // *********************************************************************
//
//    // Computing the indices is optional
//    if( isFixColumnDF == false ){
//      GetTime( timeSta );
//      numMu_ = std::min(IRound(numStateTotal*numMuFac), ntot);
//
//      // Step 1: Pre-compression of the wavefunctions. This uses
//      // multiplication with orthonormalized random Gaussian matrices
//      //
//      /// @todo The factor 2.0 is hard coded.  The PhiG etc should in
//      /// principle be a tensor, but only treated as matrix.
//      Int numPre = std::min(IRound(std::sqrt(numMu_*2.0)), numStateTotal);
//      //    Int numPre = std::min(IRound(std::sqrt(numMu_))+5, numStateTotal);
//      DblNumMat phiG(ntot, numPre), psiG(ntot, numPre);
//      {
//        DblNumMat G(numStateTotal, numPre);
//        // Generate orthonormal Gaussian random matrix 
//        GaussianRandom(G);
//        lapack::Orth( numStateTotal, numPre, G.Data(), numStateTotal );
//
//        blas::Gemm( 'N', 'N', ntot, numPre, numStateTotal, 1.0, 
//            phi.Data(), ntot, G.Data(), numStateTotal, 0.0,
//            phiG.Data(), ntot );
//
//        GaussianRandom(G);
//        lapack::Orth( numStateTotal, numPre, G.Data(), numStateTotal );
//
//        blas::Gemm( 'N', 'N', ntot, numPre, numStateTotal, 1.0, 
//            wavefun_.Data(), ntot, G.Data(), numStateTotal, 0.0,
//            psiG.Data(), ntot );
//      }
//
//      // Step 2: Pivoted QR decomposition  for the Hadamard product of
//      // the compressed matrix. Transpose format for QRCP
//      DblNumMat MG( numPre*numPre, ntot );
//      for( Int j = 0; j < numPre; j++ ){
//        for( Int i = 0; i < numPre; i++ ){
//          for( Int ir = 0; ir < ntot; ir++ ){
//            MG(i+j*numPre,ir) = phiG(ir,i) * psiG(ir,j);
//          }
//        }
//      }
//
//      // IntNumVec pivQR_(ntot);
//
//      DblNumVec tau(ntot);
//      pivQR_.Resize(ntot);
//      SetValue( pivQR_, 0 ); // Important. Otherwise QRCP uses piv as initial guess
//      // Q factor does not need to be used
//      Real timeQRCPSta, timeQRCPEnd;
//      GetTime( timeQRCPSta );
//      lapack::QRCP( numPre*numPre, ntot, MG.Data(), numPre*numPre, 
//          pivQR_.Data(), tau.Data() );
//      GetTime( timeQRCPEnd );
//#if ( _DEBUGlevel_ >= 0 )
//      statusOFS << "Time for QRCP alone is " <<
//        timeQRCPEnd - timeQRCPSta << " [s]" << std::endl << std::endl;
//#endif
//
//
//      if(1){
//        Real tolR = std::abs(MG(numMu_-1,numMu_-1)/MG(0,0));
//        statusOFS << "numMu_ = " << numMu_ << std::endl;
//        statusOFS << "|R(numMu-1,numMu-1)/R(0,0)| = " << tolR << std::endl;
//      }
//
//      GetTime( timeEnd );
//#if ( _DEBUGlevel_ >= 0 )
//      statusOFS << "Time for density fitting with QRCP is " <<
//        timeEnd - timeSta << " [s]" << std::endl << std::endl;
//#endif
//    }
//
//    // *********************************************************************
//    // Compute the interpolation matrix via the density matrix formulation
//    // *********************************************************************
//
//    GetTime( timeSta );
//    DblNumMat Xi(ntot, numMu_);
//    DblNumMat psiMu(numStateTotal, numMu_);
//    // PhiMu is scaled by the occupation number to reflect the "true" density matrix
//    DblNumMat PcolPhiMu(ntot, numMu_);
//    IntNumVec pivMu(numMu_);
//
//    {
//      GetTime( timeSta );
//      for( Int mu = 0; mu < numMu_; mu++ ){
//        pivMu(mu) = pivQR_(mu);
//      }
//
//      // These three matrices are used only once
//      DblNumMat phiMu(numStateTotal, numMu_);
//      DblNumMat PcolMuNu(numMu_, numMu_);
//      DblNumMat PcolPsiMu(ntot, numMu_);
//
//      for( Int mu = 0; mu < numMu_; mu++ ){
//        Int muInd = pivMu(mu);
//        for (Int k=0; k<numStateTotal; k++) {
//          psiMu(k, mu) = wavefun_(muInd,0,k);
//          phiMu(k, mu) = phi(muInd,0,k) * occupationRate[k];
//        }
//      }
//
//      blas::Gemm( 'N', 'N', ntot, numMu_, numStateTotal, 1.0, 
//          wavefun_.Data(), ntot, psiMu.Data(), numStateTotal, 0.0,
//          PcolPsiMu.Data(), ntot );
//      blas::Gemm( 'N', 'N', ntot, numMu_, numStateTotal, 1.0, 
//          phi.Data(), ntot, phiMu.Data(), numStateTotal, 0.0,
//          PcolPhiMu.Data(), ntot );
//
//      Real* xiPtr = Xi.Data();
//      Real* PcolPsiMuPtr = PcolPsiMu.Data();
//      Real* PcolPhiMuPtr = PcolPhiMu.Data();
//
//      for( Int g = 0; g < ntot * numMu_; g++ ){
//        xiPtr[g] = PcolPsiMuPtr[g] * PcolPhiMuPtr[g];
//      }
//
//      for( Int mu = 0; mu < numMu_; mu++ ){
//        Int muInd = pivMu(mu);
//        for (Int nu=0; nu < numMu_; nu++) {
//          PcolMuNu( mu, nu ) = Xi( muInd, nu );
//        }
//      }
//
//      //        statusOFS << "PcolMuNu = " << PcolMuNu << std::endl;
//
//      // Inversion based on Cholesky factorization
//      // Xi <- Xi * L^{-T} L^{-1}
//      // If overflow / underflow, reduce numMu_
//      lapack::Potrf( 'L', numMu_, PcolMuNu.Data(), numMu_ );
//
//      blas::Trsm( 'R', 'L', 'T', 'N', ntot, numMu_, 1.0, 
//          PcolMuNu.Data(), numMu_, Xi.Data(), ntot );
//
//      blas::Trsm( 'R', 'L', 'N', 'N', ntot, numMu_, 1.0, 
//          PcolMuNu.Data(), numMu_, Xi.Data(), ntot );
//
//      GetTime( timeEnd );
//#if ( _DEBUGlevel_ >= 0 )
//      statusOFS << "Time for computing the interpolation vectors is " <<
//        timeEnd - timeSta << " [s]" << std::endl << std::endl;
//#endif
//    }
//
//
//    // *********************************************************************
//    // Solve the Poisson equations
//    // Rewrite Xi by the potential of Xi
//    // *********************************************************************
//
//    {
//      GetTime( timeSta );
//      for( Int mu = 0; mu < numMu_; mu++ ){
//        blas::Copy( ntot,  Xi.VecData(mu), 1, fft.inputVecR2C.Data(), 1 );
//
//        FFTWExecute ( fft, fft.forwardPlanR2C );
//
//        for( Int ig = 0; ig < ntotR2C; ig++ ){
//          fft.outputVecR2C(ig) *= -exxFraction * exxgkkR2C(ig);
//        }
//
//        FFTWExecute ( fft, fft.backwardPlanR2C );
//
//        blas::Copy( ntot, fft.inputVecR2C.Data(), 1, Xi.VecData(mu), 1 );
//      } // for (mu)
//
//      GetTime( timeEnd );
//#if ( _DEBUGlevel_ >= 0 )
//      statusOFS << "Time for solving Poisson-like equations is " <<
//        timeEnd - timeSta << " [s]" << std::endl << std::endl;
//#endif
//    }
//
//    // *********************************************************************
//    // Compute the exchange potential and the symmetrized inner product
//    // *********************************************************************
//
//    {
//      GetTime( timeSta );
//      // Rewrite Xi by Xi.*PcolPhi
//      Real* xiPtr = Xi.Data();
//      Real* PcolPhiMuPtr = PcolPhiMu.Data();
//      for( Int g = 0; g < ntot * numMu_; g++ ){
//        xiPtr[g] *= PcolPhiMuPtr[g];
//      }
//
//      // NOTE: a3 must be zero in order to compute the M matrix later
//      blas::Gemm( 'N', 'T', ntot, numStateTotal, numMu_, 1.0, 
//          Xi.Data(), ntot, psiMu.Data(), numStateTotal, 1.0,
//          a3.Data(), ntot ); 
//
//      GetTime( timeEnd );
//#if ( _DEBUGlevel_ >= 0 )
//      statusOFS << "Time for computing the exchange potential is " <<
//        timeEnd - timeSta << " [s]" << std::endl << std::endl;
//#endif
//    }
//
//    // Compute the matrix VxMat = -Psi'* vexxPsi and symmetrize
//    // vexxPsi (a3) must be zero before entering this routine
//    VxMat.Resize( numStateTotal, numStateTotal );
//    {
//      // Minus sign so that VxMat is positive semidefinite
//      // NOTE: No measure factor vol / ntot due to the normalization
//      // factor of psi
//      GetTime( timeSta );
//      blas::Gemm( 'T', 'N', numStateTotal, numStateTotal, ntot, -1.0,
//          wavefun_.Data(), ntot, a3.Data(), ntot, 0.0, 
//          VxMat.Data(), numStateTotal );
//
//      //        statusOFS << "VxMat = " << VxMat << std::endl;
//
//      Symmetrize( VxMat );
//
//      GetTime( timeEnd );
//#if ( _DEBUGlevel_ >= 0 )
//      statusOFS << "Time for computing VxMat in the sym format is " <<
//        timeEnd - timeSta << " [s]" << std::endl << std::endl;
//#endif
//    }
//
//  }//if(0)



  // Pre-processing. Perform SCDM to align the orbitals into localized orbitals
  // This assumes that the phi and psi orbitals are the same
  IntNumVec permPhi(ntot);
  DblNumMat Q(numStateTotal, numStateTotal);
  DblNumMat phiSave(ntot, numStateTotal);
  DblNumMat psiSave(ntot, numStateTotal);
  lapack::Lacpy( 'A', ntot, numStateTotal, phi.Data(), ntot, phiSave.Data(), ntot );
  lapack::Lacpy( 'A', ntot, numStateTotal, wavefun_.Data(), ntot, psiSave.Data(), ntot );

  if(1){
    if( mpisize > 1 )
      ErrorHandling("Only works for mpisize == 1.");

    DblNumMat phiT;
    Transpose( DblNumMat(ntot, numStateTotal, false, phi.Data()), phiT );

    // SCDM using sequential QRCP
    DblNumMat R(numStateTotal, ntot);
    lapack::QRCP( numStateTotal, ntot, phiT.Data(), Q.Data(), R.Data(), numStateTotal, 
        permPhi.Data() );

    // Make a copy before GEMM

    // Rotate phi
    blas::Gemm( 'N', 'N', ntot, numStateTotal, numStateTotal, 1.0, 
        phiSave.Data(), ntot, Q.Data(), numStateTotal, 0.0,
        phi.Data(), ntot );
    // Rotate psi
    blas::Gemm( 'N', 'N', ntot, numStateTotal, numStateTotal, 1.0, 
        psiSave.Data(), ntot, Q.Data(), numStateTotal, 0.0,
        wavefun_.Data(), ntot );
  }



  if(1){ //For MPI

    // *********************************************************************
    // Perform interpolative separable density fitting
    // *********************************************************************

    //numMu_ = std::min(IRound(numStateTotal*numMuFac), ntot);
    numMu_ = IRound(numStateTotal*numMuFac);

    /// @todo The factor 2.0 is hard coded.  The PhiG etc should in
    /// principle be a tensor, but only treated as matrix.
    //Int numPre = std::min(IRound(std::sqrt(numMu_*2.0)), numStateTotal);
    //Int numPre = std::min(IRound(std::sqrt(numMu_))+5, numStateTotal);
    //Int numPre = IRound(std::sqrt(numMu_*numGaussianRandomFac));
    if( IRound(std::sqrt(numMu_*numGaussianRandomFac)) > numStateTotal ){
      ErrorHandling("numMu is too large for interpolative separable density fitting!");
    }

    Int numPre = numMuFac*numGaussianRandomFac;

    statusOFS << "numMu  = " << numMu_ << std::endl;
    statusOFS << "numPre*numStateTotal = " << numPre*numStateTotal << std::endl;

    // Convert the column partition to row partition
    Int numStateBlocksize = numStateTotal / mpisize;
    Int ntotBlocksize = ntot / mpisize;

    Int numMuBlocksize = numMu_ / mpisize;

    Int numStateLocal = numStateBlocksize;
    Int ntotLocal = ntotBlocksize;

    Int numMuLocal = numMuBlocksize;

    if(mpirank < (numStateTotal % mpisize)){
      numStateLocal = numStateBlocksize + 1;
    }

    if(mpirank < (numMu_ % mpisize)){
      numMuLocal = numMuBlocksize + 1;
    }

    if(mpirank < (ntot % mpisize)){
      ntotLocal = ntotBlocksize + 1;
    }

    DblNumMat localVexxPsiCol( ntot, numStateLocal );
    SetValue( localVexxPsiCol, 0.0 );

    DblNumMat localVexxPsiRow( ntotLocal, numStateTotal );
    SetValue( localVexxPsiRow, 0.0 );

    //DblNumMat localphiGRow( ntotLocal, numPre );
    //SetValue( localphiGRow, 0.0 );

    DblNumMat localpsiGRow( ntotLocal, numPre );
    SetValue( localpsiGRow, 0.0 );

    DblNumMat G( numStateTotal, numPre );
    SetValue( G, 0.0 );

    DblNumMat phiCol( ntot, numStateLocal );
    SetValue( phiCol, 0.0 );
    DblNumMat phiRow( ntotLocal, numStateTotal );
    SetValue( phiRow, 0.0 );

    DblNumMat psiCol( ntot, numStateLocal );
    SetValue( psiCol, 0.0 );
    DblNumMat psiRow( ntotLocal, numStateTotal );
    SetValue( psiRow, 0.0 );

    lapack::Lacpy( 'A', ntot, numStateLocal, phi.Data(), ntot, phiCol.Data(), ntot );
    lapack::Lacpy( 'A', ntot, numStateLocal, wavefun_.Data(), ntot, psiCol.Data(), ntot );

    AlltoallForward (phiCol, phiRow, domain_.comm);
    AlltoallForward (psiCol, psiRow, domain_.comm);

    // Computing the indices is optional

    Int ntotLocalMG, ntotMG;

    if( (ntot % mpisize) == 0 ){
      ntotLocalMG = ntotBlocksize;
    }
    else{
      ntotLocalMG = ntotBlocksize + 1;
    }


    if( isFixColumnDF == false ){
      GetTime( timeSta );

      // Step 1: Pre-compression of the wavefunctions. This uses
      // multiplication with orthonormalized random Gaussian matrices
      if ( mpirank == 0) {
        GaussianRandom(G);
        lapack::Orth( numStateTotal, numPre, G.Data(), numStateTotal );
      }

      MPI_Bcast(G.Data(), numStateTotal * numPre, MPI_DOUBLE, 0, domain_.comm);

      //blas::Gemm( 'N', 'N', ntotLocal, numPre, numStateTotal, 1.0, 
      //    phiRow.Data(), ntotLocal, G.Data(), numStateTotal, 0.0,
      //    localphiGRow.Data(), ntotLocal );

      blas::Gemm( 'N', 'N', ntotLocal, numPre, numStateTotal, 1.0, 
          psiRow.Data(), ntotLocal, G.Data(), numStateTotal, 0.0,
          localpsiGRow.Data(), ntotLocal );

      // Step 2: Pivoted QR decomposition  for the Hadamard product of
      // the compressed matrix. Transpose format for QRCP

      // NOTE: All processors should have the same ntotLocalMG
      ntotMG = ntotLocalMG * mpisize;

      if(0){

        //  DblNumMat MG( numPre*numPre, ntotLocalMG );
        //  SetValue( MG, 0.0 );
        //  for( Int j = 0; j < numPre; j++ ){
        //    for( Int i = 0; i < numPre; i++ ){
        //      for( Int ir = 0; ir < ntotLocal; ir++ ){
        //        MG(i+j*numPre,ir) = localphiGRow(ir,i) * localpsiGRow(ir,j);
        //      }
        //    }
        //  }

      }//if(0)


      DblNumMat MG( numStateTotal*numPre, ntotLocalMG );
      SetValue( MG, 0.0 );

      if(1){

        for( Int j = 0; j < numPre; j++ ){
          for( Int i = 0; i < numStateTotal; i++ ){
            for( Int ir = 0; ir < ntotLocal; ir++ ){
              MG(i+j*numStateTotal,ir) = phiRow(ir,i) * ( localpsiGRow(ir,j) - psiRow(ir,i) * G(i, j) );
            }
          }
        }

      } // if(1)

      DblNumVec tau(ntotMG);
      pivQR_.Resize(ntotMG);
      SetValue( pivQR_, 0 ); // Important. Otherwise QRCP uses piv as initial guess
      // Q factor does not need to be used

      Real timeQRCPSta, timeQRCPEnd;
      GetTime( timeQRCPSta );

      if(0){  
        lapack::QRCP( numStateTotal*numPre, ntotMG, MG.Data(), numStateTotal*numPre, 
            pivQR_.Data(), tau.Data() );
      }//

      if(1){ // ScaLAPACL QRCP
        Int contxt;
        Int nprow, npcol, myrow, mycol, info;
        Cblacs_get(0, 0, &contxt);
        nprow = 1;
        npcol = mpisize;

        Cblacs_gridinit(&contxt, "C", nprow, npcol);
        Cblacs_gridinfo(contxt, &nprow, &npcol, &myrow, &mycol);
        Int desc_MG[9];
        Int desc_QR[9];

        Int irsrc = 0;
        Int icsrc = 0;

        Int mb_MG = numStateTotal*numPre;
        Int nb_MG = ntotLocalMG;

        // FIXME The current routine does not actually allow ntotLocal to be different on different processors.
        // This must be fixed.
        SCALAPACK(descinit)(&desc_MG[0], &mb_MG, &ntotMG, &mb_MG, &nb_MG, &irsrc, 
            &icsrc, &contxt, &mb_MG, &info);

        IntNumVec pivQRTmp(ntotMG), pivQRLocal(ntotMG);
        if( mb_MG > ntot ){
          std::ostringstream msg;
          msg << "numStateTotal*numPre > ntot. The number of grid points is perhaps too small!" << std::endl;
          ErrorHandling( msg.str().c_str() );
        }
        // DiagR is only for debugging purpose
        //        DblNumVec diagRLocal( mb_MG );
        //        DblNumVec diagR( mb_MG );

        SetValue( pivQRTmp, 0 );
        SetValue( pivQRLocal, 0 );
        SetValue( pivQR_, 0 );

        //        SetValue( diagRLocal, 0.0 );
        //        SetValue( diagR, 0.0 );

        scalapack::QRCPF( mb_MG, ntotMG, MG.Data(), &desc_MG[0], 
            pivQRTmp.Data(), tau.Data() );

        //        scalapack::QRCPR( mb_MG, ntotMG, numMu_, MG.Data(), &desc_MG[0], 
        //            pivQRTmp.Data(), tau.Data(), 80, 40 );

        // Combine the local pivQRTmp to global pivQR_
        for( Int j = 0; j < ntotLocalMG; j++ ){
          pivQRLocal[j + mpirank * ntotLocalMG] = pivQRTmp[j];
        }

        //        std::cout << "diag of MG = " << std::endl;
        //        if(mpirank == 0){
        //          std::cout << pivQRLocal << std::endl;
        //          for( Int j = 0; j < mb_MG; j++ ){
        //            std::cout << MG(j,j) << std::endl;
        //          }
        //        }
        MPI_Allreduce( pivQRLocal.Data(), pivQR_.Data(), 
            ntotMG, MPI_INT, MPI_SUM, domain_.comm );

        if(contxt >= 0) {
          Cblacs_gridexit( contxt );
        }
      } //

      GetTime( timeQRCPEnd );

      //statusOFS << std::endl<< "All processors exit with abort in spinor.cpp." << std::endl;

#if ( _DEBUGlevel_ >= 0 )
      statusOFS << "Time for QRCP alone is " <<
        timeQRCPEnd - timeQRCPSta << " [s]" << std::endl << std::endl;
#endif

      if(0){
        Real tolR = std::abs(MG(numMu_-1,numMu_-1)/MG(0,0));
        statusOFS << "numMu_ = " << numMu_ << std::endl;
        statusOFS << "|R(numMu-1,numMu-1)/R(0,0)| = " << tolR << std::endl;
      }

      GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
      statusOFS << "Time for density fitting with QRCP is " <<
        timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

      // Dump out pivQR_
      if(0){
        std::ostringstream muStream;
        serialize( pivQR_, muStream, NO_MASK );
        SharedWrite( "pivQR", muStream );
      }
    }

    // Load pivQR_ file
    if(0){
      statusOFS << "Loading pivQR file.." << std::endl;
      std::istringstream muStream;
      SharedRead( "pivQR", muStream );
      deserialize( pivQR_, muStream, NO_MASK );
    }

    // *********************************************************************
    // Compute the interpolation matrix via the density matrix formulation
    // *********************************************************************

    GetTime( timeSta );

    DblNumMat XiRow(ntotLocal, numMu_);
    DblNumMat psiMu(numStateTotal, numMu_);
    // PhiMu is scaled by the occupation number to reflect the "true" density matrix
    DblNumMat PcolPhiMu(ntotLocal, numMu_);
    IntNumVec pivMu(numMu_);


    for( Int mu = 0; mu < numMu_; mu++ ){
      pivMu(mu) = pivQR_(mu);
    }

    // These three matrices are used only once. 
    // Used before reduce
    DblNumMat psiMuRow(numStateTotal, numMu_);
    DblNumMat phiMuRow(numStateTotal, numMu_);
    //DblNumMat PcolMuNuRow(numMu_, numMu_);
    DblNumMat PcolPsiMuRow(ntotLocal, numMu_);

    // Collecting the matrices obtained from row partition
    DblNumMat phiMu(numStateTotal, numMu_);
    DblNumMat PcolMuNu(numMu_, numMu_);
    DblNumMat PcolPsiMu(ntotLocal, numMu_);

    SetValue( psiMuRow, 0.0 );
    SetValue( phiMuRow, 0.0 );
    //SetValue( PcolMuNuRow, 0.0 );
    SetValue( PcolPsiMuRow, 0.0 );

    SetValue( phiMu, 0.0 );
    SetValue( PcolMuNu, 0.0 );
    SetValue( PcolPsiMu, 0.0 );

    GetTime( timeSta1 );

    for( Int mu = 0; mu < numMu_; mu++ ){
      Int muInd = pivMu(mu);
      // NOTE Hard coded here with the row partition strategy
      if( muInd <  mpirank * ntotLocalMG ||
          muInd >= (mpirank + 1) * ntotLocalMG )
        continue;

      Int muIndRow = muInd - mpirank * ntotLocalMG;

      for (Int k=0; k<numStateTotal; k++) {
        psiMuRow(k, mu) = psiRow(muIndRow,k);
        phiMuRow(k, mu) = phiRow(muIndRow,k) * occupationRate[k];
      }
    }

    GetTime( timeEnd1 );

#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for computing the MuRow is " <<
      timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
#endif

    GetTime( timeSta1 );

    MPI_Allreduce( psiMuRow.Data(), psiMu.Data(), 
        numStateTotal * numMu_, MPI_DOUBLE, MPI_SUM, domain_.comm );
    MPI_Allreduce( phiMuRow.Data(), phiMu.Data(), 
        numStateTotal * numMu_, MPI_DOUBLE, MPI_SUM, domain_.comm );

    GetTime( timeEnd1 );

#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for MPI_Allreduce is " <<
      timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
#endif

    GetTime( timeSta1 );

    blas::Gemm( 'N', 'N', ntotLocal, numMu_, numStateTotal, 1.0, 
        psiRow.Data(), ntotLocal, psiMu.Data(), numStateTotal, 0.0,
        PcolPsiMu.Data(), ntotLocal );
    blas::Gemm( 'N', 'N', ntotLocal, numMu_, numStateTotal, 1.0, 
        phiRow.Data(), ntotLocal, phiMu.Data(), numStateTotal, 0.0,
        PcolPhiMu.Data(), ntotLocal );

    GetTime( timeEnd1 );

#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for GEMM is " <<
      timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
#endif

    DblNumMat psiphiRow(ntotLocal, numStateTotal);
    SetValue( psiphiRow, 0.0 );

    for( Int ir = 0; ir < ntotLocal; ir++ ){
      for (Int i=0; i<numStateTotal; i++) {
        psiphiRow(ir,i) = psiRow(ir,i) * phiRow(ir, i); 
      }
    }       

    DblNumMat psiphiMu(numStateTotal, numMu_);
    SetValue( psiphiMu, 0.0 );

    for( Int i = 0; i < numStateTotal; i++ ){
      for (Int j = 0; j < numMu_; j++) {
        psiphiMu(i,j) = psiMu(i,j) * phiMu(i,j); 
      }
    }       

    DblNumMat PcolPsiPhiMu(ntotLocal, numMu_);
    SetValue( PcolPsiPhiMu, 0.0 );

    blas::Gemm( 'N', 'N', ntotLocal, numMu_, numStateTotal, 1.0, 
        psiphiRow.Data(), ntotLocal, psiphiMu.Data(), numStateTotal, 0.0,
        PcolPsiPhiMu.Data(), ntotLocal );

    Real* xiPtr = XiRow.Data();
    Real* PcolPsiMuPtr = PcolPsiMu.Data();
    Real* PcolPhiMuPtr = PcolPhiMu.Data();
    Real* PcolPsiPhiMuPtr = PcolPsiPhiMu.Data();

    GetTime( timeSta1 );

    for( Int g = 0; g < ntotLocal * numMu_; g++ ){
      xiPtr[g] = PcolPsiMuPtr[g] * PcolPhiMuPtr[g] - PcolPsiPhiMuPtr[g];
    }

    GetTime( timeEnd1 );

#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for xiPtr is " <<
      timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
#endif
    {
      GetTime( timeSta1 );

      DblNumMat PcolMuNuRow(numMu_, numMu_);
      SetValue( PcolMuNuRow, 0.0 );

      for( Int mu = 0; mu < numMu_; mu++ ){

        Int muInd = pivMu(mu);
        // NOTE Hard coded here with the row partition strategy
        if( muInd <  mpirank * ntotLocalMG ||
            muInd >= (mpirank + 1) * ntotLocalMG )
          continue;
        Int muIndRow = muInd - mpirank * ntotLocalMG;

        for (Int nu=0; nu < numMu_; nu++) {
          PcolMuNuRow( mu, nu ) = XiRow( muIndRow, nu );
        }

      }//for mu

      GetTime( timeEnd1 );

#if ( _DEBUGlevel_ >= 0 )
      statusOFS << "Time for PcolMuNuRow is " <<
        timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
#endif

      GetTime( timeSta1 );

      MPI_Allreduce( PcolMuNuRow.Data(), PcolMuNu.Data(), 
          numMu_* numMu_, MPI_DOUBLE, MPI_SUM, domain_.comm );

      GetTime( timeEnd1 );

#if ( _DEBUGlevel_ >= 0 )
      statusOFS << "Time for MPI_Allreduce is " <<
        timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
#endif

    }

    GetTime( timeSta1 );

    if(0){
      if ( mpirank == 0) {
        lapack::Potrf( 'L', numMu_, PcolMuNu.Data(), numMu_ );
      }
    } // if(0)

    if(1){ // Parallel Portf

      Int contxt;
      Int nprow, npcol, myrow, mycol, info;
      Cblacs_get(0, 0, &contxt);

      for( Int i = IRound(sqrt(double(numProcScaLAPACKPotrf))); 
          i <= numProcScaLAPACKPotrf; i++){
        nprow = i; npcol = numProcScaLAPACKPotrf / nprow;
        if( nprow * npcol == numProcScaLAPACKPotrf ) break;
      }

      IntNumVec pmap(numProcScaLAPACKPotrf);
      // Take the first numProcScaLAPACK processors for diagonalization
      for ( Int i = 0; i < numProcScaLAPACKPotrf; i++ ){
        pmap[i] = i;
      }

      Cblacs_gridmap(&contxt, &pmap[0], nprow, nprow, npcol);

      if( contxt >= 0 ){

        Int numKeep = numMu_; 
        Int lda = numMu_;

        scalapack::ScaLAPACKMatrix<Real> square_mat_scala;

        scalapack::Descriptor descReduceSeq, descReducePar;

        // Leading dimension provided
        descReduceSeq.Init( numKeep, numKeep, numKeep, numKeep, I_ZERO, I_ZERO, contxt, lda );

        // Automatically comptued Leading Dimension
        descReducePar.Init( numKeep, numKeep, scaPotrfBlockSize, scaPotrfBlockSize, I_ZERO, I_ZERO, contxt );

        square_mat_scala.SetDescriptor( descReducePar );

        DblNumMat&  square_mat = PcolMuNu;
        // Redistribute the input matrix over the process grid
        SCALAPACK(pdgemr2d)(&numKeep, &numKeep, square_mat.Data(), &I_ONE, &I_ONE, descReduceSeq.Values(), 
            &square_mat_scala.LocalMatrix()[0], &I_ONE, &I_ONE, square_mat_scala.Desc().Values(), &contxt );

        // Make the ScaLAPACK call
        char LL = 'L';
        //SCALAPACK(pdpotrf)(&LL, &numMu_, square_mat_scala.Data(), &I_ONE, &I_ONE, square_mat_scala.Desc().Values(), &info);
        scalapack::Potrf(LL, square_mat_scala);

        // Redistribute back eigenvectors
        SetValue(square_mat, 0.0 );
        SCALAPACK(pdgemr2d)( &numKeep, &numKeep, square_mat_scala.Data(), &I_ONE, &I_ONE, square_mat_scala.Desc().Values(),
            square_mat.Data(), &I_ONE, &I_ONE, descReduceSeq.Values(), &contxt );

      } // if(contxt >= 0)

      if(contxt >= 0) {
        Cblacs_gridexit( contxt );
      }

    } // if(1) for Parallel Portf

    GetTime( timeEnd1 );

#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for Potrf is " <<
      timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
#endif

    { 

      GetTime( timeSta1 );

      MPI_Bcast(PcolMuNu.Data(), numMu_ * numMu_, MPI_DOUBLE, 0, domain_.comm);

      GetTime( timeEnd1 );

#if ( _DEBUGlevel_ >= 0 )
      statusOFS << "Time for MPI_Bcast is " <<
        timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
#endif

      GetTime( timeSta1 );

    }

    blas::Trsm( 'R', 'L', 'T', 'N', ntotLocal, numMu_, 1.0, 
        PcolMuNu.Data(), numMu_, XiRow.Data(), ntotLocal );

    blas::Trsm( 'R', 'L', 'N', 'N', ntotLocal, numMu_, 1.0, 
        PcolMuNu.Data(), numMu_, XiRow.Data(), ntotLocal );

    GetTime( timeEnd1 );

#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for Trsm is " <<
      timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
#endif


    GetTime( timeEnd );

#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for computing the interpolation vectors is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

    // *********************************************************************
    // Solve the Poisson equations
    // Rewrite Xi by the potential of Xi
    // *********************************************************************

    DblNumMat XiCol(ntot, numMuLocal);

    AlltoallBackward (XiRow, XiCol, domain_.comm);

    {
      GetTime( timeSta );
      for( Int mu = 0; mu < numMuLocal; mu++ ){
        blas::Copy( ntot,  XiCol.VecData(mu), 1, fft.inputVecR2C.Data(), 1 );

        FFTWExecute ( fft, fft.forwardPlanR2C );

        for( Int ig = 0; ig < ntotR2C; ig++ ){
          fft.outputVecR2C(ig) *= -exxFraction * exxgkkR2C(ig);
        }

        FFTWExecute ( fft, fft.backwardPlanR2C );

        blas::Copy( ntot, fft.inputVecR2C.Data(), 1, XiCol.VecData(mu), 1 );

      } // for (mu)

      AlltoallForward (XiCol, XiRow, domain_.comm);

      GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
      statusOFS << "Time for solving Poisson-like equations is " <<
        timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
    }

    // *********************************************************************
    // Compute the exchange potential and the symmetrized inner product
    // *********************************************************************

    GetTime( timeSta );
    // Rewrite Xi by Xi.*PcolPhi
    DblNumMat a3Row( ntotLocal, numStateTotal );
    SetValue( a3Row, 0.0 );

    if(0){

      Real* xiPtr = XiRow.Data();
      Real* PcolPhiMuPtr = PcolPhiMu.Data();
      for( Int g = 0; g < ntotLocal * numMu_; g++ ){
        xiPtr[g] *= PcolPhiMuPtr[g];
      }

      // NOTE: a3 must be zero in order to compute the M matrix later
      blas::Gemm( 'N', 'T', ntotLocal, numStateTotal, numMu_, 1.0, 
          XiRow.Data(), ntotLocal, psiMu.Data(), numStateTotal, 1.0,
          a3Row.Data(), ntotLocal ); 

    } //if(0)


    if(1){

      for (Int i = 0; i < numStateTotal; i++) {

        DblNumMat PcolPhiMui = PcolPhiMu;

        // Remove the self-contribution
        for (Int mu = 0; mu < numMu_; mu++) {
          blas::Axpy( ntotLocal, -phiMu(i,mu), phiRow.VecData(i),
              1, PcolPhiMui.VecData(mu), 1 );
        }

        DblNumMat XiRowTemp(ntotLocal, numMu_);
        SetValue( XiRowTemp, 0.0 );
        lapack::Lacpy( 'A', ntotLocal, numMu_, XiRow.Data(), ntotLocal, XiRowTemp.Data(), ntotLocal );

        Real* xiPtr = XiRowTemp.Data();
        Real* PcolPhiMuiPtr = PcolPhiMui.Data();
        for( Int g = 0; g < ntotLocal * numMu_; g++ ){
          xiPtr[g] *= PcolPhiMuiPtr[g];
        }


        for ( Int mu = 0; mu < numMu_; mu++ ){
          blas::Axpy( ntotLocal, psiMu(i,mu), XiRowTemp.VecData(mu),
              1, a3Row.VecData(i), 1 );
        }


      } //end for i

    } //if(1)

    DblNumMat a3Col( ntot, numStateLocal );
    SetValue( a3Col, 0.0 );

    AlltoallBackward (a3Row, a3Col, domain_.comm);

    if(1){

      for (Int i=0; i<numStateLocal; i++) {

        for( Int ir = 0; ir < ntot; ir++ ){
          fft.inputVecR2C(ir) = psiCol(ir, i) * phiCol(ir, i);
        }

        FFTWExecute ( fft, fft.forwardPlanR2C );

        // Solve the Poisson-like problem for exchange
        for( Int ig = 0; ig < ntotR2C; ig++ ){
          fft.outputVecR2C(ig) *= exxgkkR2C(ig);
        }

        FFTWExecute ( fft, fft.backwardPlanR2C );

        Real fac = -exxFraction * occupationRate[wavefunIdx_(i)];  
        for( Int ir = 0; ir < ntot; ir++ ){
          a3Col(ir,i) += fft.inputVecR2C(ir) * phiCol(ir,i) * fac;
        }

      } // for i

    } //if(1)

    lapack::Lacpy( 'A', ntot, numStateLocal, a3Col.Data(), ntot, a3.Data(), ntot );

    GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for computing the exchange potential is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

    // Unrotate if SCDM is used
    if(1){
      lapack::Lacpy( 'A', ntot, numStateTotal, phiSave.Data(), ntot, 
          phi.Data(), ntot );
      lapack::Lacpy( 'A', ntot, numStateTotal, psiSave.Data(), ntot, 
          wavefun_.Data(), ntot );
    }





    // Compute the matrix VxMat = -Psi'* vexxPsi and symmetrize
    // vexxPsi (a3) must be zero before entering this routine
    VxMat.Resize( numStateTotal, numStateTotal );
    {
      // Minus sign so that VxMat is positive semidefinite
      // NOTE: No measure factor vol / ntot due to the normalization
      // factor of psi
      DblNumMat VxMatTemp( numStateTotal, numStateTotal );
      SetValue( VxMatTemp, 0.0 );
      GetTime( timeSta );

      SetValue( a3Row, 0.0 );
      AlltoallForward (a3Col, a3Row, domain_.comm);

      blas::Gemm( 'T', 'N', numStateTotal, numStateTotal, ntotLocal, -1.0,
          psiRow.Data(), ntotLocal, a3Row.Data(), ntotLocal, 0.0, 
          VxMatTemp.Data(), numStateTotal );

      SetValue( VxMat, 0.0 );
      MPI_Allreduce( VxMatTemp.Data(), VxMat.Data(), numStateTotal * numStateTotal, 
          MPI_DOUBLE, MPI_SUM, domain_.comm );

      Symmetrize( VxMat );

      GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
      statusOFS << "Time for computing VxMat in the sym format is " <<
        timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
    }

  } //if(1) for For MPI

  MPI_Barrier(domain_.comm);

  return ;
}        // -----  end of method Spinor::AddMultSpinorEXXDF  ----- 


// This is the latest density fitting algorithm using a weighted method
// for implementing the least square procedure
// Update: 1/2/2017
//void Spinor::AddMultSpinorEXXDF2 ( Fourier& fft, 
//    const NumTns<Real>& phi,
//    const DblNumVec& exxgkkR2C,
//    Real  exxFraction,
//    Real  numSpin,
//    const DblNumVec& occupationRate,
//    const Real numMuFac,
//    const Real numGaussianRandomFac,
//    const Int numProcScaLAPACKPotrf,  
//    const Int scaPotrfBlockSize,  
//    NumTns<Real>& a3, 
//    NumMat<Real>& VxMat,
//    bool isFixColumnDF )
//{
//  Real timeSta, timeEnd;
//  Real timeSta1, timeEnd1;
//
//  if( !fft.isInitialized ){
//    ErrorHandling("Fourier is not prepared.");
//  }
//
//  MPI_Barrier(domain_.comm);
//  int mpirank;  MPI_Comm_rank(domain_.comm, &mpirank);
//  int mpisize;  MPI_Comm_size(domain_.comm, &mpisize);
//
//  Index3& numGrid = domain_.numGrid;
//  Index3& numGridFine = domain_.numGridFine;
//
//  Int ntot     = domain_.NumGridTotal();
//  Int ntotFine = domain_.NumGridTotalFine();
//  Int ntotR2C = fft.numGridTotalR2C;
//  Int ntotR2CFine = fft.numGridTotalR2CFine;
//  Int ncom = wavefun_.n();
//  Int numStateLocal = wavefun_.p();
//  Int numStateTotal = numStateTotal_;
//
//  Int ncomPhi = phi.n();
//
//  Real vol = domain_.Volume();
//
//  if( ncomPhi != 1 || ncom != 1 ){
//    ErrorHandling("Spin polarized case not implemented.");
//  }
//
//  if( fft.domain.NumGridTotal() != ntot ){
//    ErrorHandling("Domain size does not match.");
//  }
//
//
//  if(1){ //For MPI
//
//    // *********************************************************************
//    // Perform interpolative separable density fitting
//    // *********************************************************************
//
//    //numMu_ = std::min(IRound(numStateTotal*numMuFac), ntot);
//    numMu_ = IRound(numStateTotal*numMuFac);
//
//    Int numPre = IRound(std::sqrt(numMu_*numGaussianRandomFac));
//    if( numPre > numStateTotal ){
//      ErrorHandling("numMu is too large for interpolative separable density fitting!");
//    }
//    
//    statusOFS << "numMu  = " << numMu_ << std::endl;
//    statusOFS << "numPre*numPre = " << numPre * numPre << std::endl;
//
//    // Convert the column partition to row partition
//    Int numStateBlocksize = numStateTotal / mpisize;
//    Int ntotBlocksize = ntot / mpisize;
//
//    Int numMuBlocksize = numMu_ / mpisize;
//
//    Int numStateLocal = numStateBlocksize;
//    Int ntotLocal = ntotBlocksize;
//
//    Int numMuLocal = numMuBlocksize;
//
//    if(mpirank < (numStateTotal % mpisize)){
//      numStateLocal = numStateBlocksize + 1;
//    }
//
//    if(mpirank < (numMu_ % mpisize)){
//      numMuLocal = numMuBlocksize + 1;
//    }
//
//    if(mpirank < (ntot % mpisize)){
//      ntotLocal = ntotBlocksize + 1;
//    }
//
//    DblNumMat localVexxPsiCol( ntot, numStateLocal );
//    SetValue( localVexxPsiCol, 0.0 );
//
//    DblNumMat localVexxPsiRow( ntotLocal, numStateTotal );
//    SetValue( localVexxPsiRow, 0.0 );
//
//    DblNumMat localphiGRow( ntotLocal, numPre );
//    SetValue( localphiGRow, 0.0 );
//
//    DblNumMat localpsiGRow( ntotLocal, numPre );
//    SetValue( localpsiGRow, 0.0 );
//
//    DblNumMat G(numStateTotal, numPre);
//    SetValue( G, 0.0 );
//
//    DblNumMat phiCol( ntot, numStateLocal );
//    SetValue( phiCol, 0.0 );
//    DblNumMat phiRow( ntotLocal, numStateTotal );
//    SetValue( phiRow, 0.0 );
//
//    DblNumMat psiCol( ntot, numStateLocal );
//    SetValue( psiCol, 0.0 );
//    DblNumMat psiRow( ntotLocal, numStateTotal );
//    SetValue( psiRow, 0.0 );
//
//    lapack::Lacpy( 'A', ntot, numStateLocal, phi.Data(), ntot, phiCol.Data(), ntot );
//    lapack::Lacpy( 'A', ntot, numStateLocal, wavefun_.Data(), ntot, psiCol.Data(), ntot );
//
//    AlltoallForward (phiCol, phiRow, domain_.comm);
//    AlltoallForward (psiCol, psiRow, domain_.comm);
//
//    // Computing the indices is optional
//
//    Int ntotLocalMG, ntotMG;
//
//    if( (ntot % mpisize) == 0 ){
//      ntotLocalMG = ntotBlocksize;
//    }
//    else{
//      ntotLocalMG = ntotBlocksize + 1;
//    }
//
//
//    if( isFixColumnDF == false ){
//      GetTime( timeSta );
//
//      // Step 1: Pre-compression of the wavefunctions. This uses
//      // multiplication with orthonormalized random Gaussian matrices
//      if ( mpirank == 0) {
//        GaussianRandom(G);
//        lapack::Orth( numStateTotal, numPre, G.Data(), numStateTotal );
//      }
//
//      MPI_Bcast(G.Data(), numStateTotal * numPre, MPI_DOUBLE, 0, domain_.comm);
//
//      blas::Gemm( 'N', 'N', ntotLocal, numPre, numStateTotal, 1.0, 
//          phiRow.Data(), ntotLocal, G.Data(), numStateTotal, 0.0,
//          localphiGRow.Data(), ntotLocal );
//
//      blas::Gemm( 'N', 'N', ntotLocal, numPre, numStateTotal, 1.0, 
//          psiRow.Data(), ntotLocal, G.Data(), numStateTotal, 0.0,
//          localpsiGRow.Data(), ntotLocal );
//
//      // Step 2: Pivoted QR decomposition  for the Hadamard product of
//      // the compressed matrix. Transpose format for QRCP
//
//      // NOTE: All processors should have the same ntotLocalMG
//      ntotMG = ntotLocalMG * mpisize;
//
//      DblNumMat MG( numPre*numPre, ntotLocalMG );
//      SetValue( MG, 0.0 );
//      for( Int j = 0; j < numPre; j++ ){
//        for( Int i = 0; i < numPre; i++ ){
//          for( Int ir = 0; ir < ntotLocal; ir++ ){
//            MG(i+j*numPre,ir) = localphiGRow(ir,i) * localpsiGRow(ir,j);
//          }
//        }
//      }
//
//      DblNumVec tau(ntotMG);
//      pivQR_.Resize(ntotMG);
//      SetValue( pivQR_, 0 ); // Important. Otherwise QRCP uses piv as initial guess
//      // Q factor does not need to be used
//
//      Real timeQRCPSta, timeQRCPEnd;
//      GetTime( timeQRCPSta );
//
//
//
//      if(0){  
//        lapack::QRCP( numPre*numPre, ntotMG, MG.Data(), numPre*numPre, 
//            pivQR_.Data(), tau.Data() );
//      }//
//
//
//      if(1){ // ScaLAPACL QRCP
//        Int contxt;
//        Int nprow, npcol, myrow, mycol, info;
//        Cblacs_get(0, 0, &contxt);
//        nprow = 1;
//        npcol = mpisize;
//
//        Cblacs_gridinit(&contxt, "C", nprow, npcol);
//        Cblacs_gridinfo(contxt, &nprow, &npcol, &myrow, &mycol);
//        Int desc_MG[9];
//        Int desc_QR[9];
//
//        Int irsrc = 0;
//        Int icsrc = 0;
//
//        Int mb_MG = numPre*numPre;
//        Int nb_MG = ntotLocalMG;
//
//        // FIXME The current routine does not actually allow ntotLocal to be different on different processors.
//        // This must be fixed.
//        SCALAPACK(descinit)(&desc_MG[0], &mb_MG, &ntotMG, &mb_MG, &nb_MG, &irsrc, 
//            &icsrc, &contxt, &mb_MG, &info);
//
//        IntNumVec pivQRTmp(ntotMG), pivQRLocal(ntotMG);
//        if( mb_MG > ntot ){
//          std::ostringstream msg;
//          msg << "numPre*numPre > ntot. The number of grid points is perhaps too small!" << std::endl;
//          ErrorHandling( msg.str().c_str() );
//        }
//        // DiagR is only for debugging purpose
////        DblNumVec diagRLocal( mb_MG );
////        DblNumVec diagR( mb_MG );
//
//        SetValue( pivQRTmp, 0 );
//        SetValue( pivQRLocal, 0 );
//        SetValue( pivQR_, 0 );
//
////        SetValue( diagRLocal, 0.0 );
////        SetValue( diagR, 0.0 );
//
//        scalapack::QRCPF( mb_MG, ntotMG, MG.Data(), &desc_MG[0], 
//            pivQRTmp.Data(), tau.Data() );
//
////        scalapack::QRCPR( mb_MG, ntotMG, numMu_, MG.Data(), &desc_MG[0], 
////            pivQRTmp.Data(), tau.Data(), 80, 40 );
//
//        // Combine the local pivQRTmp to global pivQR_
//        for( Int j = 0; j < ntotLocalMG; j++ ){
//          pivQRLocal[j + mpirank * ntotLocalMG] = pivQRTmp[j];
//        }
//
//        //        std::cout << "diag of MG = " << std::endl;
//        //        if(mpirank == 0){
//        //          std::cout << pivQRLocal << std::endl;
//        //          for( Int j = 0; j < mb_MG; j++ ){
//        //            std::cout << MG(j,j) << std::endl;
//        //          }
//        //        }
//        MPI_Allreduce( pivQRLocal.Data(), pivQR_.Data(), 
//            ntotMG, MPI_INT, MPI_SUM, domain_.comm );
//
//        if(contxt >= 0) {
//          Cblacs_gridexit( contxt );
//        }
//      } //
//
//
//      GetTime( timeQRCPEnd );
//
//      //statusOFS << std::endl<< "All processors exit with abort in spinor.cpp." << std::endl;
//
//#if ( _DEBUGlevel_ >= 0 )
//      statusOFS << "Time for QRCP alone is " <<
//        timeQRCPEnd - timeQRCPSta << " [s]" << std::endl << std::endl;
//#endif
//
//      if(0){
//        Real tolR = std::abs(MG(numMu_-1,numMu_-1)/MG(0,0));
//        statusOFS << "numMu_ = " << numMu_ << std::endl;
//        statusOFS << "|R(numMu-1,numMu-1)/R(0,0)| = " << tolR << std::endl;
//      }
//
//      GetTime( timeEnd );
//#if ( _DEBUGlevel_ >= 0 )
//      statusOFS << "Time for density fitting with QRCP is " <<
//        timeEnd - timeSta << " [s]" << std::endl << std::endl;
//#endif
//
//      // Dump out pivQR_
//      if(0){
//        std::ostringstream muStream;
//        serialize( pivQR_, muStream, NO_MASK );
//        SharedWrite( "pivQR", muStream );
//      }
//    }
//
//    // Load pivQR_ file
//    if(0){
//      statusOFS << "Loading pivQR file.." << std::endl;
//      std::istringstream muStream;
//      SharedRead( "pivQR", muStream );
//      deserialize( pivQR_, muStream, NO_MASK );
//    }
//
//    // *********************************************************************
//    // Compute the interpolation matrix via the density matrix formulation
//    // *********************************************************************
//
//    GetTime( timeSta );
//    
//    DblNumMat XiRow(ntotLocal, numMu_);
//    DblNumMat psiMu(numStateTotal, numMu_);
//    // PhiMu is scaled by the occupation number to reflect the "true" density matrix
//    DblNumMat PcolPhiMu(ntotLocal, numMu_);
//    IntNumVec pivMu(numMu_);
//
//    {
//    
//      for( Int mu = 0; mu < numMu_; mu++ ){
//        pivMu(mu) = pivQR_(mu);
//      }
//
//      // These three matrices are used only once. 
//      // Used before reduce
//      DblNumMat psiMuRow(numStateTotal, numMu_);
//      DblNumMat phiMuRow(numStateTotal, numMu_);
//      //DblNumMat PcolMuNuRow(numMu_, numMu_);
//      DblNumMat PcolPsiMuRow(ntotLocal, numMu_);
//
//      // Collecting the matrices obtained from row partition
//      DblNumMat phiMu(numStateTotal, numMu_);
//      DblNumMat PcolMuNu(numMu_, numMu_);
//      DblNumMat PcolPsiMu(ntotLocal, numMu_);
//
//      SetValue( psiMuRow, 0.0 );
//      SetValue( phiMuRow, 0.0 );
//      //SetValue( PcolMuNuRow, 0.0 );
//      SetValue( PcolPsiMuRow, 0.0 );
//
//      SetValue( phiMu, 0.0 );
//      SetValue( PcolMuNu, 0.0 );
//      SetValue( PcolPsiMu, 0.0 );
//
//      GetTime( timeSta1 );
//
//      for( Int mu = 0; mu < numMu_; mu++ ){
//        Int muInd = pivMu(mu);
//        // NOTE Hard coded here with the row partition strategy
//        if( muInd <  mpirank * ntotLocalMG ||
//            muInd >= (mpirank + 1) * ntotLocalMG )
//          continue;
//
//        Int muIndRow = muInd - mpirank * ntotLocalMG;
//
//        for (Int k=0; k<numStateTotal; k++) {
//          psiMuRow(k, mu) = psiRow(muIndRow,k);
//          phiMuRow(k, mu) = phiRow(muIndRow,k) * occupationRate[k];
//        }
//      }
//
//      GetTime( timeEnd1 );
//
//#if ( _DEBUGlevel_ >= 0 )
//      statusOFS << "Time for computing the MuRow is " <<
//        timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
//#endif
//
//      GetTime( timeSta1 );
//      
//      MPI_Allreduce( psiMuRow.Data(), psiMu.Data(), 
//          numStateTotal * numMu_, MPI_DOUBLE, MPI_SUM, domain_.comm );
//      MPI_Allreduce( phiMuRow.Data(), phiMu.Data(), 
//          numStateTotal * numMu_, MPI_DOUBLE, MPI_SUM, domain_.comm );
//      
//      GetTime( timeEnd1 );
//
//#if ( _DEBUGlevel_ >= 0 )
//      statusOFS << "Time for MPI_Allreduce is " <<
//        timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
//#endif
//      
//      GetTime( timeSta1 );
//
//      blas::Gemm( 'N', 'N', ntotLocal, numMu_, numStateTotal, 1.0, 
//          psiRow.Data(), ntotLocal, psiMu.Data(), numStateTotal, 0.0,
//          PcolPsiMu.Data(), ntotLocal );
//      blas::Gemm( 'N', 'N', ntotLocal, numMu_, numStateTotal, 1.0, 
//          phiRow.Data(), ntotLocal, phiMu.Data(), numStateTotal, 0.0,
//          PcolPhiMu.Data(), ntotLocal );
//      
//      GetTime( timeEnd1 );
//
//#if ( _DEBUGlevel_ >= 0 )
//      statusOFS << "Time for GEMM is " <<
//        timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
//#endif
//      
//      GetTime( timeSta1 );
//
//      Real* xiPtr = XiRow.Data();
//      Real* PcolPsiMuPtr = PcolPsiMu.Data();
//      Real* PcolPhiMuPtr = PcolPhiMu.Data();
//      
//      for( Int g = 0; g < ntotLocal * numMu_; g++ ){
//        xiPtr[g] = PcolPsiMuPtr[g] * PcolPhiMuPtr[g];
//      }
//
//      // 1/2/2017 Add extra weight to certain entries to the XiRow matrix
//      // Currently only works for one processor
//      if(0)
//      {
//        Real wgt = 10.0;
//        // Correction for Diagonal entries
//        for( Int mu = 0; mu < numMu_; mu++ ){
//          xiPtr = XiRow.VecData(mu);
//          for( Int i = 0; i < numStateTotal; i++ ){
//            Real* phiPtr = phi.VecData(0, i);
//            Real* psiPtr = wavefun_.VecData(0,i);
//            Real  fac = phiMu(i,mu) * psiMu(i,mu) * wgt; 
//            for( Int g = 0; g < ntotLocal; g++ ){
//              xiPtr[g] += phiPtr[g] * psiPtr[g] * fac;
//            } 
//          }
//        }
//      }
//
//      if(0)
//      {
//        Real wgt = 10.0;
//        // Correction for HOMO 
//        for( Int mu = 0; mu < numMu_; mu++ ){
//          xiPtr = XiRow.VecData(mu);
//          Real* phiPtr = phi.VecData(0, numStateTotal-1);
//          for( Int i = 0; i < numStateTotal; i++ ){
//            Real* psiPtr = wavefun_.VecData(0,i);
//            Real  fac = phiMu(numStateTotal-1,mu) * psiMu(i,mu) * wgt; 
//            for( Int g = 0; g < ntotLocal; g++ ){
//              xiPtr[g] += phiPtr[g] * psiPtr[g] * fac;
//            } 
//          }
//        }
//      }
//      
//      
//      GetTime( timeEnd1 );
//
//#if ( _DEBUGlevel_ >= 0 )
//      statusOFS << "Time for xiPtr is " <<
//        timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
//#endif
//      {
//
//        GetTime( timeSta1 );
//
//        DblNumMat PcolMuNuRow(numMu_, numMu_);
//        SetValue( PcolMuNuRow, 0.0 );
//
//        for( Int mu = 0; mu < numMu_; mu++ ){
//
//          Int muInd = pivMu(mu);
//          // NOTE Hard coded here with the row partition strategy
//          if( muInd <  mpirank * ntotLocalMG ||
//              muInd >= (mpirank + 1) * ntotLocalMG )
//            continue;
//          Int muIndRow = muInd - mpirank * ntotLocalMG;
//
//          for (Int nu=0; nu < numMu_; nu++) {
//            PcolMuNuRow( mu, nu ) = XiRow( muIndRow, nu );
//          }
//
//        }//for mu
//
//        GetTime( timeEnd1 );
//
//#if ( _DEBUGlevel_ >= 0 )
//        statusOFS << "Time for PcolMuNuRow is " <<
//          timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
//#endif
//
//        GetTime( timeSta1 );
//
//        MPI_Allreduce( PcolMuNuRow.Data(), PcolMuNu.Data(), 
//            numMu_* numMu_, MPI_DOUBLE, MPI_SUM, domain_.comm );
//
//        GetTime( timeEnd1 );
//
//#if ( _DEBUGlevel_ >= 0 )
//        statusOFS << "Time for MPI_Allreduce is " <<
//          timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
//#endif
//
//      }
//        
//      GetTime( timeSta1 );
//
//      if(0){
//        if ( mpirank == 0) {
//          lapack::Potrf( 'L', numMu_, PcolMuNu.Data(), numMu_ );
//        }
//      } // if(0)
//
//      if(1){ // Parallel Portf
//
//        Int contxt;
//        Int nprow, npcol, myrow, mycol, info;
//        Cblacs_get(0, 0, &contxt);
//
//        for( Int i = IRound(sqrt(double(numProcScaLAPACKPotrf))); 
//            i <= numProcScaLAPACKPotrf; i++){
//          nprow = i; npcol = numProcScaLAPACKPotrf / nprow;
//          if( nprow * npcol == numProcScaLAPACKPotrf ) break;
//        }
//
//        IntNumVec pmap(numProcScaLAPACKPotrf);
//        // Take the first numProcScaLAPACK processors for diagonalization
//        for ( Int i = 0; i < numProcScaLAPACKPotrf; i++ ){
//          pmap[i] = i;
//        }
//
//        Cblacs_gridmap(&contxt, &pmap[0], nprow, nprow, npcol);
//
//        if( contxt >= 0 ){
//
//          Int numKeep = numMu_; 
//          Int lda = numMu_;
//
//          scalapack::ScaLAPACKMatrix<Real> square_mat_scala;
//
//          scalapack::Descriptor descReduceSeq, descReducePar;
//
//          // Leading dimension provided
//          descReduceSeq.Init( numKeep, numKeep, numKeep, numKeep, I_ZERO, I_ZERO, contxt, lda );
//
//          // Automatically comptued Leading Dimension
//          descReducePar.Init( numKeep, numKeep, scaPotrfBlockSize, scaPotrfBlockSize, I_ZERO, I_ZERO, contxt );
//
//          square_mat_scala.SetDescriptor( descReducePar );
//
//          DblNumMat&  square_mat = PcolMuNu;
//          // Redistribute the input matrix over the process grid
//          SCALAPACK(pdgemr2d)(&numKeep, &numKeep, square_mat.Data(), &I_ONE, &I_ONE, descReduceSeq.Values(), 
//              &square_mat_scala.LocalMatrix()[0], &I_ONE, &I_ONE, square_mat_scala.Desc().Values(), &contxt );
//
//          // Make the ScaLAPACK call
//          char LL = 'L';
//          //SCALAPACK(pdpotrf)(&LL, &numMu_, square_mat_scala.Data(), &I_ONE, &I_ONE, square_mat_scala.Desc().Values(), &info);
//          scalapack::Potrf(LL, square_mat_scala);
//
//          // Redistribute back eigenvectors
//          SetValue(square_mat, 0.0 );
//          SCALAPACK(pdgemr2d)( &numKeep, &numKeep, square_mat_scala.Data(), &I_ONE, &I_ONE, square_mat_scala.Desc().Values(),
//              square_mat.Data(), &I_ONE, &I_ONE, descReduceSeq.Values(), &contxt );
//        
//        } // if(contxt >= 0)
//
//        if(contxt >= 0) {
//          Cblacs_gridexit( contxt );
//        }
//
//      } // if(1) for Parallel Portf
//
//      GetTime( timeEnd1 );
//
//#if ( _DEBUGlevel_ >= 0 )
//      statusOFS << "Time for Potrf is " <<
//        timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
//#endif
//
//      { 
//
//        GetTime( timeSta1 );
//
//        MPI_Bcast(PcolMuNu.Data(), numMu_ * numMu_, MPI_DOUBLE, 0, domain_.comm);
//
//        GetTime( timeEnd1 );
//
//#if ( _DEBUGlevel_ >= 0 )
//        statusOFS << "Time for MPI_Bcast is " <<
//          timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
//#endif
//
//        GetTime( timeSta1 );
//
//      }
//
//      blas::Trsm( 'R', 'L', 'T', 'N', ntotLocal, numMu_, 1.0, 
//          PcolMuNu.Data(), numMu_, XiRow.Data(), ntotLocal );
//
//      blas::Trsm( 'R', 'L', 'N', 'N', ntotLocal, numMu_, 1.0, 
//          PcolMuNu.Data(), numMu_, XiRow.Data(), ntotLocal );
//
//      GetTime( timeEnd1 );
//
//#if ( _DEBUGlevel_ >= 0 )
//      statusOFS << "Time for Trsm is " <<
//        timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
//#endif
//
//    }
//      
//    GetTime( timeEnd );
//
//#if ( _DEBUGlevel_ >= 0 )
//    statusOFS << "Time for computing the interpolation vectors is " <<
//      timeEnd - timeSta << " [s]" << std::endl << std::endl;
//#endif
//
//    // *********************************************************************
//    // Solve the Poisson equations
//    // Rewrite Xi by the potential of Xi
//    // *********************************************************************
//
//    DblNumMat XiCol(ntot, numMuLocal);
//
//    AlltoallBackward (XiRow, XiCol, domain_.comm);
//       
//    {
//      GetTime( timeSta );
//      for( Int mu = 0; mu < numMuLocal; mu++ ){
//        blas::Copy( ntot,  XiCol.VecData(mu), 1, fft.inputVecR2C.Data(), 1 );
//
//        FFTWExecute ( fft, fft.forwardPlanR2C );
//
//        for( Int ig = 0; ig < ntotR2C; ig++ ){
//          fft.outputVecR2C(ig) *= -exxFraction * exxgkkR2C(ig);
//        }
//
//        FFTWExecute ( fft, fft.backwardPlanR2C );
//
//        blas::Copy( ntot, fft.inputVecR2C.Data(), 1, XiCol.VecData(mu), 1 );
//
//      } // for (mu)
//
//      AlltoallForward (XiCol, XiRow, domain_.comm);
//
//      GetTime( timeEnd );
//#if ( _DEBUGlevel_ >= 0 )
//      statusOFS << "Time for solving Poisson-like equations is " <<
//        timeEnd - timeSta << " [s]" << std::endl << std::endl;
//#endif
//    }
//
//
//
//
//    // *********************************************************************
//    // Compute the exchange potential and the symmetrized inner product
//    // *********************************************************************
//
//    GetTime( timeSta );
//    // Rewrite Xi by Xi.*PcolPhi
//    Real* xiPtr = XiRow.Data();
//    Real* PcolPhiMuPtr = PcolPhiMu.Data();
//    for( Int g = 0; g < ntotLocal * numMu_; g++ ){
//      xiPtr[g] *= PcolPhiMuPtr[g];
//    }
//
//    // NOTE: a3 must be zero in order to compute the M matrix later
//    DblNumMat a3Row( ntotLocal, numStateTotal );
//    SetValue( a3Row, 0.0 );
//    blas::Gemm( 'N', 'T', ntotLocal, numStateTotal, numMu_, 1.0, 
//        XiRow.Data(), ntotLocal, psiMu.Data(), numStateTotal, 1.0,
//        a3Row.Data(), ntotLocal ); 
//
//    DblNumMat a3Col( ntot, numStateLocal );
//    
//    AlltoallBackward (a3Row, a3Col, domain_.comm);
//
//    lapack::Lacpy( 'A', ntot, numStateLocal, a3Col.Data(), ntot, a3.Data(), ntot );
//
//    GetTime( timeEnd );
//#if ( _DEBUGlevel_ >= 0 )
//    statusOFS << "Time for computing the exchange potential is " <<
//      timeEnd - timeSta << " [s]" << std::endl << std::endl;
//#endif
//
//    // Compute the matrix VxMat = -Psi'* vexxPsi and symmetrize
//    // vexxPsi (a3) must be zero before entering this routine
//    VxMat.Resize( numStateTotal, numStateTotal );
//    if(0)
//    {
//      // Minus sign so that VxMat is positive semidefinite
//      // NOTE: No measure factor vol / ntot due to the normalization
//      // factor of psi
//      DblNumMat VxMatTemp( numStateTotal, numStateTotal );
//      SetValue( VxMatTemp, 0.0 );
//      GetTime( timeSta );
//      blas::Gemm( 'T', 'N', numStateTotal, numStateTotal, ntotLocal, -1.0,
//          psiRow.Data(), ntotLocal, a3Row.Data(), ntotLocal, 0.0, 
//          VxMatTemp.Data(), numStateTotal );
//
//      SetValue( VxMat, 0.0 );
//      MPI_Allreduce( VxMatTemp.Data(), VxMat.Data(), numStateTotal * numStateTotal, 
//          MPI_DOUBLE, MPI_SUM, domain_.comm );
//
//      Symmetrize( VxMat );
//     
//      GetTime( timeEnd );
//#if ( _DEBUGlevel_ >= 0 )
//      statusOFS << "Time for computing VxMat in the sym format is " <<
//        timeEnd - timeSta << " [s]" << std::endl << std::endl;
//#endif
//    }
//
//  } //if(1) for For MPI
//
//  MPI_Barrier(domain_.comm);
//
//  return ;
//}        // -----  end of method Spinor::AddMultSpinorEXXDF2  ----- 

// This is density fitting formulation with symmetric implementation of
// the M matrix when combined with ACE, so that the POTRF does not fail as often
// Update: 1/10/2017
//void Spinor::AddMultSpinorEXXDF3 ( Fourier& fft, 
//    const NumTns<Real>& phi,
//    const DblNumVec& exxgkkR2C,
//    Real  exxFraction,
//    Real  numSpin,
//    const DblNumVec& occupationRate,
//    const Real numMuFac,
//    const Real numGaussianRandomFac,
//    const Int numProcScaLAPACKPotrf,  
//    const Int scaPotrfBlockSize,  
//    NumTns<Real>& a3, 
//    NumMat<Real>& VxMat,
//    bool isFixColumnDF )
//{
//  Real timeSta, timeEnd;
//  Real timeSta1, timeEnd1;
//
//  if( !fft.isInitialized ){
//    ErrorHandling("Fourier is not prepared.");
//  }
//
//  MPI_Barrier(domain_.comm);
//  int mpirank;  MPI_Comm_rank(domain_.comm, &mpirank);
//  int mpisize;  MPI_Comm_size(domain_.comm, &mpisize);
//
//  Index3& numGrid = domain_.numGrid;
//  Index3& numGridFine = domain_.numGridFine;
//
//  Int ntot     = domain_.NumGridTotal();
//  Int ntotFine = domain_.NumGridTotalFine();
//  Int ntotR2C = fft.numGridTotalR2C;
//  Int ntotR2CFine = fft.numGridTotalR2CFine;
//  Int ncom = wavefun_.n();
//  Int numStateLocal = wavefun_.p();
//  Int numStateTotal = numStateTotal_;
//
//  Int ncomPhi = phi.n();
//
//  Real vol = domain_.Volume();
//
//  if( ncomPhi != 1 || ncom != 1 ){
//    ErrorHandling("Spin polarized case not implemented.");
//  }
//
//  if( fft.domain.NumGridTotal() != ntot ){
//    ErrorHandling("Domain size does not match.");
//  }
//
//
//  if(1){ //For MPI
//
//    // *********************************************************************
//    // Perform interpolative separable density fitting
//    // *********************************************************************
//
//    //numMu_ = std::min(IRound(numStateTotal*numMuFac), ntot);
//    numMu_ = IRound(numStateTotal*numMuFac);
//
//    Int numPre = IRound(std::sqrt(numMu_*numGaussianRandomFac));
//    if( numPre > numStateTotal ){
//      ErrorHandling("numMu is too large for interpolative separable density fitting!");
//    }
//    
//    statusOFS << "numMu  = " << numMu_ << std::endl;
//    statusOFS << "numPre*numPre = " << numPre * numPre << std::endl;
//
//    // Convert the column partition to row partition
//    Int numStateBlocksize = numStateTotal / mpisize;
//    Int ntotBlocksize = ntot / mpisize;
//
//    Int numMuBlocksize = numMu_ / mpisize;
//
//    Int numStateLocal = numStateBlocksize;
//    Int ntotLocal = ntotBlocksize;
//
//    Int numMuLocal = numMuBlocksize;
//
//    if(mpirank < (numStateTotal % mpisize)){
//      numStateLocal = numStateBlocksize + 1;
//    }
//
//    if(mpirank < (numMu_ % mpisize)){
//      numMuLocal = numMuBlocksize + 1;
//    }
//
//    if(mpirank < (ntot % mpisize)){
//      ntotLocal = ntotBlocksize + 1;
//    }
//
//    DblNumMat localVexxPsiCol( ntot, numStateLocal );
//    SetValue( localVexxPsiCol, 0.0 );
//
//    DblNumMat localVexxPsiRow( ntotLocal, numStateTotal );
//    SetValue( localVexxPsiRow, 0.0 );
//
//    DblNumMat localphiGRow( ntotLocal, numPre );
//    SetValue( localphiGRow, 0.0 );
//
//    DblNumMat localpsiGRow( ntotLocal, numPre );
//    SetValue( localpsiGRow, 0.0 );
//
//    DblNumMat G(numStateTotal, numPre);
//    SetValue( G, 0.0 );
//
//    DblNumMat phiCol( ntot, numStateLocal );
//    SetValue( phiCol, 0.0 );
//    DblNumMat phiRow( ntotLocal, numStateTotal );
//    SetValue( phiRow, 0.0 );
//
//    DblNumMat psiCol( ntot, numStateLocal );
//    SetValue( psiCol, 0.0 );
//    DblNumMat psiRow( ntotLocal, numStateTotal );
//    SetValue( psiRow, 0.0 );
//
//    lapack::Lacpy( 'A', ntot, numStateLocal, phi.Data(), ntot, phiCol.Data(), ntot );
//    lapack::Lacpy( 'A', ntot, numStateLocal, wavefun_.Data(), ntot, psiCol.Data(), ntot );
//
//    // Computing the indices is optional
//
//    Int ntotLocalMG, ntotMG;
//
//    if( (ntot % mpisize) == 0 ){
//      ntotLocalMG = ntotBlocksize;
//    }
//    else{
//      ntotLocalMG = ntotBlocksize + 1;
//    }
//
//
//    if( isFixColumnDF == false ){
//      GetTime( timeSta );
//
//      // Step 1: Pre-compression of the wavefunctions. This uses
//      // multiplication with orthonormalized random Gaussian matrices
//      if ( mpirank == 0) {
//        GaussianRandom(G);
//        lapack::Orth( numStateTotal, numPre, G.Data(), numStateTotal );
//      }
//
//      MPI_Bcast(G.Data(), numStateTotal * numPre, MPI_DOUBLE, 0, domain_.comm);
//    
//      AlltoallForward (phiCol, phiRow, domain_.comm);
//      AlltoallForward (psiCol, psiRow, domain_.comm);
//
//      blas::Gemm( 'N', 'N', ntotLocal, numPre, numStateTotal, 1.0, 
//          phiRow.Data(), ntotLocal, G.Data(), numStateTotal, 0.0,
//          localphiGRow.Data(), ntotLocal );
//
//      blas::Gemm( 'N', 'N', ntotLocal, numPre, numStateTotal, 1.0, 
//          psiRow.Data(), ntotLocal, G.Data(), numStateTotal, 0.0,
//          localpsiGRow.Data(), ntotLocal );
//
//      // Step 2: Pivoted QR decomposition  for the Hadamard product of
//      // the compressed matrix. Transpose format for QRCP
//
//      // NOTE: All processors should have the same ntotLocalMG
//      ntotMG = ntotLocalMG * mpisize;
//
//      DblNumMat MG( numPre*numPre, ntotLocalMG );
//      SetValue( MG, 0.0 );
//      for( Int j = 0; j < numPre; j++ ){
//        for( Int i = 0; i < numPre; i++ ){
//          for( Int ir = 0; ir < ntotLocal; ir++ ){
//            MG(i+j*numPre,ir) = localphiGRow(ir,i) * localpsiGRow(ir,j);
//          }
//        }
//      }
//
//      DblNumVec tau(ntotMG);
//      pivQR_.Resize(ntotMG);
//      SetValue( pivQR_, 0 ); // Important. Otherwise QRCP uses piv as initial guess
//      // Q factor does not need to be used
//
//      Real timeQRCPSta, timeQRCPEnd;
//      GetTime( timeQRCPSta );
//
//
//
//      if(0){  
//        lapack::QRCP( numPre*numPre, ntotMG, MG.Data(), numPre*numPre, 
//            pivQR_.Data(), tau.Data() );
//      }//
//
//
//      if(1){ // ScaLAPACL QRCP
//        Int contxt;
//        Int nprow, npcol, myrow, mycol, info;
//        Cblacs_get(0, 0, &contxt);
//        nprow = 1;
//        npcol = mpisize;
//
//        Cblacs_gridinit(&contxt, "C", nprow, npcol);
//        Cblacs_gridinfo(contxt, &nprow, &npcol, &myrow, &mycol);
//        Int desc_MG[9];
//        Int desc_QR[9];
//
//        Int irsrc = 0;
//        Int icsrc = 0;
//
//        Int mb_MG = numPre*numPre;
//        Int nb_MG = ntotLocalMG;
//
//        // FIXME The current routine does not actually allow ntotLocal to be different on different processors.
//        // This must be fixed.
//        SCALAPACK(descinit)(&desc_MG[0], &mb_MG, &ntotMG, &mb_MG, &nb_MG, &irsrc, 
//            &icsrc, &contxt, &mb_MG, &info);
//
//        IntNumVec pivQRTmp(ntotMG), pivQRLocal(ntotMG);
//        if( mb_MG > ntot ){
//          std::ostringstream msg;
//          msg << "numPre*numPre > ntot. The number of grid points is perhaps too small!" << std::endl;
//          ErrorHandling( msg.str().c_str() );
//        }
//        // DiagR is only for debugging purpose
////        DblNumVec diagRLocal( mb_MG );
////        DblNumVec diagR( mb_MG );
//
//        SetValue( pivQRTmp, 0 );
//        SetValue( pivQRLocal, 0 );
//        SetValue( pivQR_, 0 );
//
////        SetValue( diagRLocal, 0.0 );
////        SetValue( diagR, 0.0 );
//
//        scalapack::QRCPF( mb_MG, ntotMG, MG.Data(), &desc_MG[0], 
//            pivQRTmp.Data(), tau.Data() );
//
////        scalapack::QRCPR( mb_MG, ntotMG, numMu_, MG.Data(), &desc_MG[0], 
////            pivQRTmp.Data(), tau.Data(), 80, 40 );
//
//        // Combine the local pivQRTmp to global pivQR_
//        for( Int j = 0; j < ntotLocalMG; j++ ){
//          pivQRLocal[j + mpirank * ntotLocalMG] = pivQRTmp[j];
//        }
//
//        //        std::cout << "diag of MG = " << std::endl;
//        //        if(mpirank == 0){
//        //          std::cout << pivQRLocal << std::endl;
//        //          for( Int j = 0; j < mb_MG; j++ ){
//        //            std::cout << MG(j,j) << std::endl;
//        //          }
//        //        }
//        MPI_Allreduce( pivQRLocal.Data(), pivQR_.Data(), 
//            ntotMG, MPI_INT, MPI_SUM, domain_.comm );
//
//        if(contxt >= 0) {
//          Cblacs_gridexit( contxt );
//        }
//      } //
//
//
//      GetTime( timeQRCPEnd );
//
//      //statusOFS << std::endl<< "All processors exit with abort in spinor.cpp." << std::endl;
//
//#if ( _DEBUGlevel_ >= 0 )
//      statusOFS << "Time for QRCP alone is " <<
//        timeQRCPEnd - timeQRCPSta << " [s]" << std::endl << std::endl;
//#endif
//
//      if(0){
//        Real tolR = std::abs(MG(numMu_-1,numMu_-1)/MG(0,0));
//        statusOFS << "numMu_ = " << numMu_ << std::endl;
//        statusOFS << "|R(numMu-1,numMu-1)/R(0,0)| = " << tolR << std::endl;
//      }
//
//      GetTime( timeEnd );
//#if ( _DEBUGlevel_ >= 0 )
//      statusOFS << "Time for density fitting with QRCP is " <<
//        timeEnd - timeSta << " [s]" << std::endl << std::endl;
//#endif
//
//      // Dump out pivQR_
//      if(0){
//        std::ostringstream muStream;
//        serialize( pivQR_, muStream, NO_MASK );
//        SharedWrite( "pivQR", muStream );
//      }
//    }
//
//    // Load pivQR_ file
//    if(0){
//      statusOFS << "Loading pivQR file.." << std::endl;
//      std::istringstream muStream;
//      SharedRead( "pivQR", muStream );
//      deserialize( pivQR_, muStream, NO_MASK );
//    }
//
//    // *********************************************************************
//    // Compute the interpolation matrix via the density matrix formulation
//    // *********************************************************************
//
//    GetTime( timeSta );
//    
//    DblNumMat XiRow(ntotLocal, numMu_);
//    DblNumMat psiMu(numStateTotal, numMu_);
//    // PhiMu is scaled by the occupation number to reflect the "true" density matrix
//    DblNumMat PcolPhiMu(ntotLocal, numMu_);
//    IntNumVec pivMu(numMu_);
//    DblNumMat phiMu(numStateTotal, numMu_);
//
//    {
//    
//      for( Int mu = 0; mu < numMu_; mu++ ){
//        pivMu(mu) = pivQR_(mu);
//      }
//
//      // These three matrices are used only once. 
//      // Used before reduce
//      DblNumMat psiMuRow(numStateTotal, numMu_);
//      DblNumMat phiMuRow(numStateTotal, numMu_);
//      //DblNumMat PcolMuNuRow(numMu_, numMu_);
//      DblNumMat PcolPsiMuRow(ntotLocal, numMu_);
//
//      // Collecting the matrices obtained from row partition
//      DblNumMat PcolMuNu(numMu_, numMu_);
//      DblNumMat PcolPsiMu(ntotLocal, numMu_);
//
//      SetValue( psiMuRow, 0.0 );
//      SetValue( phiMuRow, 0.0 );
//      //SetValue( PcolMuNuRow, 0.0 );
//      SetValue( PcolPsiMuRow, 0.0 );
//
//      SetValue( phiMu, 0.0 );
//      SetValue( PcolMuNu, 0.0 );
//      SetValue( PcolPsiMu, 0.0 );
//
//      GetTime( timeSta1 );
//
//      for( Int mu = 0; mu < numMu_; mu++ ){
//        Int muInd = pivMu(mu);
//        // NOTE Hard coded here with the row partition strategy
//        if( muInd <  mpirank * ntotLocalMG ||
//            muInd >= (mpirank + 1) * ntotLocalMG )
//          continue;
//
//        Int muIndRow = muInd - mpirank * ntotLocalMG;
//
//        for (Int k=0; k<numStateTotal; k++) {
//          psiMuRow(k, mu) = psiRow(muIndRow,k);
//          phiMuRow(k, mu) = phiRow(muIndRow,k) * occupationRate[k];
//        }
//      }
//
//      GetTime( timeEnd1 );
//
//#if ( _DEBUGlevel_ >= 0 )
//      statusOFS << "Time for computing the MuRow is " <<
//        timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
//#endif
//
//      GetTime( timeSta1 );
//      
//      MPI_Allreduce( psiMuRow.Data(), psiMu.Data(), 
//          numStateTotal * numMu_, MPI_DOUBLE, MPI_SUM, domain_.comm );
//      MPI_Allreduce( phiMuRow.Data(), phiMu.Data(), 
//          numStateTotal * numMu_, MPI_DOUBLE, MPI_SUM, domain_.comm );
//      
//      GetTime( timeEnd1 );
//
//#if ( _DEBUGlevel_ >= 0 )
//      statusOFS << "Time for MPI_Allreduce is " <<
//        timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
//#endif
//      
//      GetTime( timeSta1 );
//
//      blas::Gemm( 'N', 'N', ntotLocal, numMu_, numStateTotal, 1.0, 
//          psiRow.Data(), ntotLocal, psiMu.Data(), numStateTotal, 0.0,
//          PcolPsiMu.Data(), ntotLocal );
//      blas::Gemm( 'N', 'N', ntotLocal, numMu_, numStateTotal, 1.0, 
//          phiRow.Data(), ntotLocal, phiMu.Data(), numStateTotal, 0.0,
//          PcolPhiMu.Data(), ntotLocal );
//      
//      GetTime( timeEnd1 );
//
//#if ( _DEBUGlevel_ >= 0 )
//      statusOFS << "Time for GEMM is " <<
//        timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
//#endif
//      
//      GetTime( timeSta1 );
//
//      Real* xiPtr = XiRow.Data();
//      Real* PcolPsiMuPtr = PcolPsiMu.Data();
//      Real* PcolPhiMuPtr = PcolPhiMu.Data();
//      
//      for( Int g = 0; g < ntotLocal * numMu_; g++ ){
//        xiPtr[g] = PcolPsiMuPtr[g] * PcolPhiMuPtr[g];
//      }
//
//      // 1/2/2017 Add extra weight to certain entries to the XiRow matrix
//      // Currently only works for one processor
//      if(0)
//      {
//        Real wgt = 10.0;
//        // Correction for Diagonal entries
//        for( Int mu = 0; mu < numMu_; mu++ ){
//          xiPtr = XiRow.VecData(mu);
//          for( Int i = 0; i < numStateTotal; i++ ){
//            Real* phiPtr = phi.VecData(0, i);
//            Real* psiPtr = wavefun_.VecData(0,i);
//            Real  fac = phiMu(i,mu) * psiMu(i,mu) * wgt; 
//            for( Int g = 0; g < ntotLocal; g++ ){
//              xiPtr[g] += phiPtr[g] * psiPtr[g] * fac;
//            } 
//          }
//        }
//      }
//
//      if(0)
//      {
//        Real wgt = 10.0;
//        // Correction for HOMO 
//        for( Int mu = 0; mu < numMu_; mu++ ){
//          xiPtr = XiRow.VecData(mu);
//          Real* phiPtr = phi.VecData(0, numStateTotal-1);
//          for( Int i = 0; i < numStateTotal; i++ ){
//            Real* psiPtr = wavefun_.VecData(0,i);
//            Real  fac = phiMu(numStateTotal-1,mu) * psiMu(i,mu) * wgt; 
//            for( Int g = 0; g < ntotLocal; g++ ){
//              xiPtr[g] += phiPtr[g] * psiPtr[g] * fac;
//            } 
//          }
//        }
//      }
//      
//      
//      GetTime( timeEnd1 );
//
//#if ( _DEBUGlevel_ >= 0 )
//      statusOFS << "Time for xiPtr is " <<
//        timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
//#endif
//      {
//
//        GetTime( timeSta1 );
//
//        DblNumMat PcolMuNuRow(numMu_, numMu_);
//        SetValue( PcolMuNuRow, 0.0 );
//
//        for( Int mu = 0; mu < numMu_; mu++ ){
//
//          Int muInd = pivMu(mu);
//          // NOTE Hard coded here with the row partition strategy
//          if( muInd <  mpirank * ntotLocalMG ||
//              muInd >= (mpirank + 1) * ntotLocalMG )
//            continue;
//          Int muIndRow = muInd - mpirank * ntotLocalMG;
//
//          for (Int nu=0; nu < numMu_; nu++) {
//            PcolMuNuRow( mu, nu ) = XiRow( muIndRow, nu );
//          }
//
//        }//for mu
//
//        GetTime( timeEnd1 );
//
//#if ( _DEBUGlevel_ >= 0 )
//        statusOFS << "Time for PcolMuNuRow is " <<
//          timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
//#endif
//
//        GetTime( timeSta1 );
//
//        MPI_Allreduce( PcolMuNuRow.Data(), PcolMuNu.Data(), 
//            numMu_* numMu_, MPI_DOUBLE, MPI_SUM, domain_.comm );
//
//        GetTime( timeEnd1 );
//
//#if ( _DEBUGlevel_ >= 0 )
//        statusOFS << "Time for MPI_Allreduce is " <<
//          timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
//#endif
//
//      }
//        
//      GetTime( timeSta1 );
//
//      if(0){
//        if ( mpirank == 0) {
//          lapack::Potrf( 'L', numMu_, PcolMuNu.Data(), numMu_ );
//        }
//      } // if(0)
//
//      if(1){ // Parallel Portf
//
//        Int contxt;
//        Int nprow, npcol, myrow, mycol, info;
//        Cblacs_get(0, 0, &contxt);
//
//        for( Int i = IRound(sqrt(double(numProcScaLAPACKPotrf))); 
//            i <= numProcScaLAPACKPotrf; i++){
//          nprow = i; npcol = numProcScaLAPACKPotrf / nprow;
//          if( nprow * npcol == numProcScaLAPACKPotrf ) break;
//        }
//
//        IntNumVec pmap(numProcScaLAPACKPotrf);
//        // Take the first numProcScaLAPACK processors for diagonalization
//        for ( Int i = 0; i < numProcScaLAPACKPotrf; i++ ){
//          pmap[i] = i;
//        }
//
//        Cblacs_gridmap(&contxt, &pmap[0], nprow, nprow, npcol);
//
//        if( contxt >= 0 ){
//
//          Int numKeep = numMu_; 
//          Int lda = numMu_;
//
//          scalapack::ScaLAPACKMatrix<Real> square_mat_scala;
//
//          scalapack::Descriptor descReduceSeq, descReducePar;
//
//          // Leading dimension provided
//          descReduceSeq.Init( numKeep, numKeep, numKeep, numKeep, I_ZERO, I_ZERO, contxt, lda );
//
//          // Automatically comptued Leading Dimension
//          descReducePar.Init( numKeep, numKeep, scaPotrfBlockSize, scaPotrfBlockSize, I_ZERO, I_ZERO, contxt );
//
//          square_mat_scala.SetDescriptor( descReducePar );
//
//          DblNumMat&  square_mat = PcolMuNu;
//          // Redistribute the input matrix over the process grid
//          SCALAPACK(pdgemr2d)(&numKeep, &numKeep, square_mat.Data(), &I_ONE, &I_ONE, descReduceSeq.Values(), 
//              &square_mat_scala.LocalMatrix()[0], &I_ONE, &I_ONE, square_mat_scala.Desc().Values(), &contxt );
//
//          // Make the ScaLAPACK call
//          char LL = 'L';
//          //SCALAPACK(pdpotrf)(&LL, &numMu_, square_mat_scala.Data(), &I_ONE, &I_ONE, square_mat_scala.Desc().Values(), &info);
//          scalapack::Potrf(LL, square_mat_scala);
//
//          // Redistribute back eigenvectors
//          SetValue(square_mat, 0.0 );
//          SCALAPACK(pdgemr2d)( &numKeep, &numKeep, square_mat_scala.Data(), &I_ONE, &I_ONE, square_mat_scala.Desc().Values(),
//              square_mat.Data(), &I_ONE, &I_ONE, descReduceSeq.Values(), &contxt );
//        
//        } // if(contxt >= 0)
//
//        if(contxt >= 0) {
//          Cblacs_gridexit( contxt );
//        }
//
//      } // if(1) for Parallel Portf
//
//      GetTime( timeEnd1 );
//
//#if ( _DEBUGlevel_ >= 0 )
//      statusOFS << "Time for Potrf is " <<
//        timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
//#endif
//
//      { 
//
//        GetTime( timeSta1 );
//
//        MPI_Bcast(PcolMuNu.Data(), numMu_ * numMu_, MPI_DOUBLE, 0, domain_.comm);
//
//        GetTime( timeEnd1 );
//
//#if ( _DEBUGlevel_ >= 0 )
//        statusOFS << "Time for MPI_Bcast is " <<
//          timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
//#endif
//
//        GetTime( timeSta1 );
//
//      }
//
//      blas::Trsm( 'R', 'L', 'T', 'N', ntotLocal, numMu_, 1.0, 
//          PcolMuNu.Data(), numMu_, XiRow.Data(), ntotLocal );
//
//      blas::Trsm( 'R', 'L', 'N', 'N', ntotLocal, numMu_, 1.0, 
//          PcolMuNu.Data(), numMu_, XiRow.Data(), ntotLocal );
//
//      GetTime( timeEnd1 );
//
//#if ( _DEBUGlevel_ >= 0 )
//      statusOFS << "Time for Trsm is " <<
//        timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
//#endif
//
//    }
//      
//    GetTime( timeEnd );
//
//#if ( _DEBUGlevel_ >= 0 )
//    statusOFS << "Time for computing the interpolation vectors is " <<
//      timeEnd - timeSta << " [s]" << std::endl << std::endl;
//#endif
//
//
//
//    // *********************************************************************
//    // Solve the Poisson equations.
//    // Store VXi separately. This is not the most memory efficient
//    // implementation
//    // *********************************************************************
//
//    DblNumMat VXiRow(ntotLocal, numMu_);
//
//    if(1){
//      // XiCol used for both input and output
//      DblNumMat XiCol(ntot, numMuLocal);
//      AlltoallBackward (XiRow, XiCol, domain_.comm);
//
//      {
//        GetTime( timeSta );
//        for( Int mu = 0; mu < numMuLocal; mu++ ){
//          blas::Copy( ntot,  XiCol.VecData(mu), 1, fft.inputVecR2C.Data(), 1 );
//
//          FFTWExecute ( fft, fft.forwardPlanR2C );
//
//          for( Int ig = 0; ig < ntotR2C; ig++ ){
//            fft.outputVecR2C(ig) *= -exxFraction * exxgkkR2C(ig);
//          }
//
//          FFTWExecute ( fft, fft.backwardPlanR2C );
//
//          blas::Copy( ntot, fft.inputVecR2C.Data(), 1, XiCol.VecData(mu), 1 );
//
//        } // for (mu)
//
//        AlltoallForward (XiCol, VXiRow, domain_.comm);
//
//        GetTime( timeEnd );
//#if ( _DEBUGlevel_ >= 0 )
//        statusOFS << "Time for solving Poisson-like equations is " <<
//          timeEnd - timeSta << " [s]" << std::endl << std::endl;
//#endif
//      }
//    }
//
//    // Prepare for the computation of the M matrix
//
//    GetTime( timeSta );
//
//    DblNumMat MMatMuNu(numMu_, numMu_);
//    {
//      DblNumMat MMatMuNuTemp( numMu_, numMu_ );
//      GetTime( timeSta );
//      // Minus sign so that MMat is positive semidefinite
//      blas::Gemm( 'T', 'N', numMu_, numMu_, ntotLocal, -1.0,
//          XiRow.Data(), ntotLocal, VXiRow.Data(), ntotLocal, 0.0, 
//          MMatMuNuTemp.Data(), numMu_ );
//
//      MPI_Allreduce( MMatMuNuTemp.Data(), MMatMuNu.Data(), numMu_ * numMu_, 
//          MPI_DOUBLE, MPI_SUM, domain_.comm );
//    }
//
//    // Element-wise multiply with phiMuNu matrix (i.e. density matrix)
//    {
//
//      DblNumMat phiMuNu(numMu_, numMu_);
//      blas::Gemm( 'T', 'N', numMu_, numMu_, numStateTotal, 1.0,
//          phiMu.Data(), numStateTotal, phiMu.Data(), numStateTotal, 0.0,
//          phiMuNu.Data(), numMu_ );
//
//      Real* MMatPtr = MMatMuNu.Data();
//      Real* phiPtr  = phiMuNu.Data();
//
//      for( Int g = 0; g < numMu_ * numMu_; g++ ){
//        MMatPtr[g] *= phiPtr[g];
//      }
//    }
//    GetTime( timeEnd );
//#if ( _DEBUGlevel_ >= 0 )
//    statusOFS << "Time for preparing the M matrix is " <<
//      timeEnd - timeSta << " [s]" << std::endl << std::endl;
//#endif
//
//
//    // *********************************************************************
//    // Compute the exchange potential and the symmetrized inner product
//    // *********************************************************************
//
//    GetTime( timeSta );
//    // Rewrite VXi by VXi.*PcolPhi
//    Real* VxiPtr = VXiRow.Data();
//    Real* PcolPhiMuPtr = PcolPhiMu.Data();
//    for( Int g = 0; g < ntotLocal * numMu_; g++ ){
//      VxiPtr[g] *= PcolPhiMuPtr[g];
//    }
//
//    // NOTE: a3 must be zero in order to compute the M matrix later
//    DblNumMat a3Row( ntotLocal, numStateTotal );
//    SetValue( a3Row, 0.0 );
//    blas::Gemm( 'N', 'T', ntotLocal, numStateTotal, numMu_, 1.0, 
//        VXiRow.Data(), ntotLocal, psiMu.Data(), numStateTotal, 1.0,
//        a3Row.Data(), ntotLocal ); 
//
//    DblNumMat a3Col( ntot, numStateLocal );
//    
//    AlltoallBackward (a3Row, a3Col, domain_.comm);
//
//    lapack::Lacpy( 'A', ntot, numStateLocal, a3Col.Data(), ntot, a3.Data(), ntot );
//
//    GetTime( timeEnd );
//#if ( _DEBUGlevel_ >= 0 )
//    statusOFS << "Time for computing the exchange potential is " <<
//      timeEnd - timeSta << " [s]" << std::endl << std::endl;
//#endif
//
//    // Compute the matrix VxMat = -Psi'* vexxPsi and symmetrize
//    // vexxPsi (a3) must be zero before entering this routine
//    VxMat.Resize( numStateTotal, numStateTotal );
//    GetTime( timeSta );
//    if(0)
//    {
//      // Minus sign so that VxMat is positive semidefinite
//      // NOTE: No measure factor vol / ntot due to the normalization
//      // factor of psi
//      DblNumMat VxMatTemp( numStateTotal, numStateTotal );
//      SetValue( VxMatTemp, 0.0 );
//      blas::Gemm( 'T', 'N', numStateTotal, numStateTotal, ntotLocal, -1.0,
//          psiRow.Data(), ntotLocal, a3Row.Data(), ntotLocal, 0.0, 
//          VxMatTemp.Data(), numStateTotal );
//
//      SetValue( VxMat, 0.0 );
//      MPI_Allreduce( VxMatTemp.Data(), VxMat.Data(), numStateTotal * numStateTotal, 
//          MPI_DOUBLE, MPI_SUM, domain_.comm );
//
//      Symmetrize( VxMat );
//     
//    }
//    if(1){
//      DblNumMat VxMatTemp( numMu_, numStateTotal );
//      blas::Gemm( 'N', 'T', numMu_, numStateTotal, numMu_, 1.0, 
//          MMatMuNu.Data(), numMu_, psiMu.Data(), numStateTotal, 0.0,
//          VxMatTemp.Data(), numMu_ );
//      blas::Gemm( 'N', 'N', numStateTotal, numStateTotal, numMu_, 1.0,
//          psiMu.Data(), numStateTotal, VxMatTemp.Data(), numMu_, 0.0,
//          VxMat.Data(), numStateTotal );
//    }
//    GetTime( timeEnd );
//#if ( _DEBUGlevel_ >= 0 )
//    statusOFS << "Time for computing VxMat in the sym format is " <<
//      timeEnd - timeSta << " [s]" << std::endl << std::endl;
//#endif
//
//  } //if(1) for For MPI
//
//  MPI_Barrier(domain_.comm);
//
//  return ;
//}        // -----  end of method Spinor::AddMultSpinorEXXDF3  ----- 


// This is density fitting formulation with symmetric implementation of
// the M matrix when combined with ACE, so that the POTRF does not fail as often
// 2D MPI communication for matrix
// Update: 2/26/2017
//void Spinor::AddMultSpinorEXXDF4 ( Fourier& fft, 
//    const NumTns<Real>& phi,
//    const DblNumVec& exxgkkR2C,
//    Real  exxFraction,
//    Real  numSpin,
//    const DblNumVec& occupationRate,
//    const Real numMuFac,
//    const Real numGaussianRandomFac,
//    const Int numProcScaLAPACK,  
//    const Int BlockSizeScaLAPACK,  
//    NumTns<Real>& a3, 
//    NumMat<Real>& VxMat,
//    bool isFixColumnDF )
//{
//  Real timeSta, timeEnd;
//  Real timeSta1, timeEnd1;
//
//  if( !fft.isInitialized ){
//    ErrorHandling("Fourier is not prepared.");
//  }
//
//  MPI_Barrier(domain_.comm);
//  int mpirank;  MPI_Comm_rank(domain_.comm, &mpirank);
//  int mpisize;  MPI_Comm_size(domain_.comm, &mpisize);
//
//  Index3& numGrid = domain_.numGrid;
//  Index3& numGridFine = domain_.numGridFine;
//
//  Int ntot     = domain_.NumGridTotal();
//  Int ntotFine = domain_.NumGridTotalFine();
//  Int ntotR2C = fft.numGridTotalR2C;
//  Int ntotR2CFine = fft.numGridTotalR2CFine;
//  Int ncom = wavefun_.n();
//  Int numStateLocal = wavefun_.p();
//  Int numStateTotal = numStateTotal_;
//
//  Int ncomPhi = phi.n();
//
//  Real vol = domain_.Volume();
//
//  if( ncomPhi != 1 || ncom != 1 ){
//    ErrorHandling("Spin polarized case not implemented.");
//  }
//
//  if( fft.domain.NumGridTotal() != ntot ){
//    ErrorHandling("Domain size does not match.");
//  }
//
//
//  // *********************************************************************
//  // Perform interpolative separable density fitting
//  // *********************************************************************
//
//  //numMu_ = std::min(IRound(numStateTotal*numMuFac), ntot);
//  numMu_ = IRound(numStateTotal*numMuFac);
//
//  Int numPre = IRound(std::sqrt(numMu_*numGaussianRandomFac));
//  if( numPre > numStateTotal ){
//    ErrorHandling("numMu is too large for interpolative separable density fitting!");
//  }
//
//  statusOFS << "ntot          = " << ntot << std::endl;
//  statusOFS << "numMu         = " << numMu_ << std::endl;
//  statusOFS << "numPre        = " << numPre << std::endl;
//  statusOFS << "numPre*numPre = " << numPre * numPre << std::endl;
//
//  // Convert the column partition to row partition
//  Int numStateBlocksize = numStateTotal / mpisize;
//  Int ntotBlocksize = ntot / mpisize;
//
//  Int numMuBlocksize = numMu_ / mpisize;
//
//  Int numStateLocal1 = numStateBlocksize;
//  Int ntotLocal = ntotBlocksize;
//
//  Int numMuLocal = numMuBlocksize;
//
//  if(mpirank < (numStateTotal % mpisize)){
//    numStateLocal1 = numStateBlocksize + 1;
//  }
//
//  if(numStateLocal !=  numStateLocal1){
//    statusOFS << "numStateLocal = " << numStateLocal << " numStateLocal1 = " << numStateLocal1 << std::endl;
//    ErrorHandling("The size is not right in interpolative separable density fitting!");
//  }
//
//  if(mpirank < (numMu_ % mpisize)){
//    numMuLocal = numMuBlocksize + 1;
//  }
//
//  if(mpirank < (ntot % mpisize)){
//    ntotLocal = ntotBlocksize + 1;
//  }
//
//  //huwei 
//  //2D MPI commucation for all the matrix
//
//  Int I_ONE = 1, I_ZERO = 0;
//  double D_ONE = 1.0;
//  double D_ZERO = 0.0;
//  double D_MinusONE = -1.0;
//
//  Int contxt0, contxt1, contxt11, contxt2;
//  Int nprow0, npcol0, myrow0, mycol0, info0;
//  Int nprow1, npcol1, myrow1, mycol1, info1;
//  Int nprow11, npcol11, myrow11, mycol11, info11;
//  Int nprow2, npcol2, myrow2, mycol2, info2;
//
//  Int ncolsNgNe1D, nrowsNgNe1D, lldNgNe1D; 
//  Int ncolsNgNe2D, nrowsNgNe2D, lldNgNe2D; 
//  Int ncolsNgNu1D, nrowsNgNu1D, lldNgNu1D; 
//  Int ncolsNgNu2D, nrowsNgNu2D, lldNgNu2D; 
//  Int ncolsNuNg2D, nrowsNuNg2D, lldNuNg2D; 
//  Int ncolsNeNe0D, nrowsNeNe0D, lldNeNe0D; 
//  Int ncolsNeNe2D, nrowsNeNe2D, lldNeNe2D; 
//  Int ncolsNuNu1D, nrowsNuNu1D, lldNuNu1D; 
//  Int ncolsNuNu2D, nrowsNuNu2D, lldNuNu2D; 
//  Int ncolsNeNu1D, nrowsNeNu1D, lldNeNu1D; 
//  Int ncolsNeNu2D, nrowsNeNu2D, lldNeNu2D; 
//  Int ncolsNuNe2D, nrowsNuNe2D, lldNuNe2D; 
//
//  Int desc_NgNe1D[9];
//  Int desc_NgNe2D[9];
//  Int desc_NgNu1D[9];
//  Int desc_NgNu2D[9];
//  Int desc_NuNg2D[9];
//  Int desc_NeNe0D[9];
//  Int desc_NeNe2D[9];
//  Int desc_NuNu1D[9];
//  Int desc_NuNu2D[9];
//  Int desc_NeNu1D[9];
//  Int desc_NeNu2D[9];
//  Int desc_NuNe2D[9];
//
//  Int Ng = ntot;
//  Int Ne = numStateTotal_; 
//  Int Nu = numMu_; 
//
//  // 0D MPI
//  nprow0 = 1;
//  npcol0 = mpisize;
//
//  Cblacs_get(0, 0, &contxt0);
//  Cblacs_gridinit(&contxt0, "C", nprow0, npcol0);
//  Cblacs_gridinfo(contxt0, &nprow0, &npcol0, &myrow0, &mycol0);
//
//  SCALAPACK(descinit)(desc_NeNe0D, &Ne, &Ne, &Ne, &Ne, &I_ZERO, &I_ZERO, &contxt0, &Ne, &info0);
//
//  // 1D MPI
//  nprow1 = 1;
//  npcol1 = mpisize;
//
//  Cblacs_get(0, 0, &contxt1);
//  Cblacs_gridinit(&contxt1, "C", nprow1, npcol1);
//  Cblacs_gridinfo(contxt1, &nprow1, &npcol1, &myrow1, &mycol1);
//
//  //desc_NgNe1D
//  if(contxt1 >= 0){
//    nrowsNgNe1D = SCALAPACK(numroc)(&Ng, &Ng, &myrow1, &I_ZERO, &nprow1);
//    ncolsNgNe1D = SCALAPACK(numroc)(&Ne, &I_ONE, &mycol1, &I_ZERO, &npcol1);
//    lldNgNe1D = std::max( nrowsNgNe1D, 1 );
//  }    
//
//  SCALAPACK(descinit)(desc_NgNe1D, &Ng, &Ne, &Ng, &I_ONE, &I_ZERO, 
//      &I_ZERO, &contxt1, &lldNgNe1D, &info1);
//
//  nprow11 = mpisize;
//  npcol11 = 1;
//
//  Cblacs_get(0, 0, &contxt11);
//  Cblacs_gridinit(&contxt11, "C", nprow11, npcol11);
//  Cblacs_gridinfo(contxt11, &nprow11, &npcol11, &myrow11, &mycol11);
//
//  //desc_NeNu1D
//  if(contxt11 >= 0){
//    nrowsNeNu1D = SCALAPACK(numroc)(&Ne, &I_ONE, &myrow11, &I_ZERO, &nprow11);
//    ncolsNeNu1D = SCALAPACK(numroc)(&Nu, &Nu, &mycol11, &I_ZERO, &npcol11);
//    lldNeNu1D = std::max( nrowsNeNu1D, 1 );
//  }    
//
//  SCALAPACK(descinit)(desc_NeNu1D, &Ne, &Nu, &I_ONE, &Nu, &I_ZERO, 
//      &I_ZERO, &contxt11, &lldNeNu1D, &info11);
//
//  //desc_NgNu1D
//  if(contxt1 >= 0){
//    nrowsNgNu1D = SCALAPACK(numroc)(&Ng, &Ng, &myrow1, &I_ZERO, &nprow1);
//    ncolsNgNu1D = SCALAPACK(numroc)(&Nu, &I_ONE, &mycol1, &I_ZERO, &npcol1);
//    lldNgNu1D = std::max( nrowsNgNu1D, 1 );
//  }    
//
//  SCALAPACK(descinit)(desc_NgNu1D, &Ng, &Nu, &Ng, &I_ONE, &I_ZERO, 
//      &I_ZERO, &contxt1, &lldNgNu1D, &info1);
//
//  //desc_NuNu1D
//  if(contxt1 >= 0){
//    nrowsNuNu1D = SCALAPACK(numroc)(&Nu, &Nu, &myrow1, &I_ZERO, &nprow1);
//    ncolsNuNu1D = SCALAPACK(numroc)(&Nu, &I_ONE, &mycol1, &I_ZERO, &npcol1);
//    lldNuNu1D = std::max( nrowsNuNu1D, 1 );
//  }    
//
//  SCALAPACK(descinit)(desc_NuNu1D, &Nu, &Nu, &Nu, &I_ONE, &I_ZERO, 
//      &I_ZERO, &contxt1, &lldNuNu1D, &info1);
//
//
//  // 2D MPI
//  for( Int i = IRound(sqrt(double(mpisize))); i <= mpisize; i++){
//    nprow2 = i; npcol2 = mpisize / nprow2;
//    if( (nprow2 >= npcol2) && (nprow2 * npcol2 == mpisize) ) break;
//  }
//
//  Cblacs_get(0, 0, &contxt2);
//  //Cblacs_gridinit(&contxt2, "C", nprow2, npcol2);
//
//  IntNumVec pmap2(mpisize);
//  for ( Int i = 0; i < mpisize; i++ ){
//    pmap2[i] = i;
//  }
//  Cblacs_gridmap(&contxt2, &pmap2[0], nprow2, nprow2, npcol2);
//
//  Int mb2 = BlockSizeScaLAPACK;
//  Int nb2 = BlockSizeScaLAPACK;
//
//  //desc_NgNe2D
//  if(contxt2 >= 0){
//    Cblacs_gridinfo(contxt2, &nprow2, &npcol2, &myrow2, &mycol2);
//    nrowsNgNe2D = SCALAPACK(numroc)(&Ng, &mb2, &myrow2, &I_ZERO, &nprow2);
//    ncolsNgNe2D = SCALAPACK(numroc)(&Ne, &nb2, &mycol2, &I_ZERO, &npcol2);
//    lldNgNe2D = std::max( nrowsNgNe2D, 1 );
//  }
//
//  SCALAPACK(descinit)(desc_NgNe2D, &Ng, &Ne, &mb2, &nb2, &I_ZERO, 
//      &I_ZERO, &contxt2, &lldNgNe2D, &info2);
//
//  //desc_NgNu2D
//  if(contxt2 >= 0){
//    Cblacs_gridinfo(contxt2, &nprow2, &npcol2, &myrow2, &mycol2);
//    nrowsNgNu2D = SCALAPACK(numroc)(&Ng, &mb2, &myrow2, &I_ZERO, &nprow2);
//    ncolsNgNu2D = SCALAPACK(numroc)(&Nu, &nb2, &mycol2, &I_ZERO, &npcol2);
//    lldNgNu2D = std::max( nrowsNgNu2D, 1 );
//  }
//
//  SCALAPACK(descinit)(desc_NgNu2D, &Ng, &Nu, &mb2, &nb2, &I_ZERO, 
//      &I_ZERO, &contxt2, &lldNgNu2D, &info2);
//  
//  //desc_NuNg2D
//  if(contxt2 >= 0){
//    Cblacs_gridinfo(contxt2, &nprow2, &npcol2, &myrow2, &mycol2);
//    nrowsNuNg2D = SCALAPACK(numroc)(&Nu, &mb2, &myrow2, &I_ZERO, &nprow2);
//    ncolsNuNg2D = SCALAPACK(numroc)(&Ng, &nb2, &mycol2, &I_ZERO, &npcol2);
//    lldNuNg2D = std::max( nrowsNuNg2D, 1 );
//  }
//
//  SCALAPACK(descinit)(desc_NuNg2D, &Nu, &Ng, &mb2, &nb2, &I_ZERO, 
//      &I_ZERO, &contxt2, &lldNuNg2D, &info2);
//
//  //desc_NeNe2D
//  if(contxt2 >= 0){
//    Cblacs_gridinfo(contxt2, &nprow2, &npcol2, &myrow2, &mycol2);
//    nrowsNeNe2D = SCALAPACK(numroc)(&Ne, &mb2, &myrow2, &I_ZERO, &nprow2);
//    ncolsNeNe2D = SCALAPACK(numroc)(&Ne, &nb2, &mycol2, &I_ZERO, &npcol2);
//    lldNeNe2D = std::max( nrowsNeNe2D, 1 );
//  }
//
//  SCALAPACK(descinit)(desc_NeNe2D, &Ne, &Ne, &mb2, &nb2, &I_ZERO, 
//      &I_ZERO, &contxt2, &lldNeNe2D, &info2);
//
//  //desc_NuNu2D
//  if(contxt2 >= 0){
//    Cblacs_gridinfo(contxt2, &nprow2, &npcol2, &myrow2, &mycol2);
//    nrowsNuNu2D = SCALAPACK(numroc)(&Nu, &mb2, &myrow2, &I_ZERO, &nprow2);
//    ncolsNuNu2D = SCALAPACK(numroc)(&Nu, &nb2, &mycol2, &I_ZERO, &npcol2);
//    lldNuNu2D = std::max( nrowsNuNu2D, 1 );
//  }
//
//  SCALAPACK(descinit)(desc_NuNu2D, &Nu, &Nu, &mb2, &nb2, &I_ZERO, 
//      &I_ZERO, &contxt2, &lldNuNu2D, &info2);
//
//  //desc_NeNu2D
//  if(contxt2 >= 0){
//    Cblacs_gridinfo(contxt2, &nprow2, &npcol2, &myrow2, &mycol2);
//    nrowsNeNu2D = SCALAPACK(numroc)(&Ne, &mb2, &myrow2, &I_ZERO, &nprow2);
//    ncolsNeNu2D = SCALAPACK(numroc)(&Nu, &nb2, &mycol2, &I_ZERO, &npcol2);
//    lldNeNu2D = std::max( nrowsNeNu2D, 1 );
//  }
//
//  SCALAPACK(descinit)(desc_NeNu2D, &Ne, &Nu, &mb2, &nb2, &I_ZERO, 
//      &I_ZERO, &contxt2, &lldNeNu2D, &info2);
//
//  //desc_NuNe2D
//  if(contxt2 >= 0){
//    Cblacs_gridinfo(contxt2, &nprow2, &npcol2, &myrow2, &mycol2);
//    nrowsNuNe2D = SCALAPACK(numroc)(&Nu, &mb2, &myrow2, &I_ZERO, &nprow2);
//    ncolsNuNe2D = SCALAPACK(numroc)(&Ne, &nb2, &mycol2, &I_ZERO, &npcol2);
//    lldNuNe2D = std::max( nrowsNuNe2D, 1 );
//  }
//
//  SCALAPACK(descinit)(desc_NuNe2D, &Nu, &Ne, &mb2, &nb2, &I_ZERO, 
//      &I_ZERO, &contxt2, &lldNuNe2D, &info2);
//
//  DblNumMat phiCol( ntot, numStateLocal );
//  SetValue( phiCol, 0.0 );
//
//  DblNumMat psiCol( ntot, numStateLocal );
//  SetValue( psiCol, 0.0 );
//
//  lapack::Lacpy( 'A', ntot, numStateLocal, phi.Data(), ntot, phiCol.Data(), ntot );
//  lapack::Lacpy( 'A', ntot, numStateLocal, wavefun_.Data(), ntot, psiCol.Data(), ntot );
//
//  // Computing the indices is optional
//
//  Int ntotLocalMG, ntotMG;
//
//  if( (ntot % mpisize) == 0 ){
//    ntotLocalMG = ntotBlocksize;
//  }
//  else{
//    ntotLocalMG = ntotBlocksize + 1;
//  }
//
//  if( isFixColumnDF == false ){
//
//    GetTime( timeSta );
//
//    DblNumMat localphiGRow( ntotLocal, numPre );
//    SetValue( localphiGRow, 0.0 );
//
//    DblNumMat localpsiGRow( ntotLocal, numPre );
//    SetValue( localpsiGRow, 0.0 );
//
//    DblNumMat G(numStateTotal, numPre);
//    SetValue( G, 0.0 );
//
//    DblNumMat phiRow( ntotLocal, numStateTotal );
//    SetValue( phiRow, 0.0 );
//
//    DblNumMat psiRow( ntotLocal, numStateTotal );
//    SetValue( psiRow, 0.0 );
//
//    AlltoallForward (phiCol, phiRow, domain_.comm);
//    AlltoallForward (psiCol, psiRow, domain_.comm);
//
//    // Step 1: Pre-compression of the wavefunctions. This uses
//    // multiplication with orthonormalized random Gaussian matrices
//    if ( mpirank == 0) {
//      GaussianRandom(G);
//      lapack::Orth( numStateTotal, numPre, G.Data(), numStateTotal );
//    }
//
//    GetTime( timeSta1 );
//
//    MPI_Bcast(G.Data(), numStateTotal * numPre, MPI_DOUBLE, 0, domain_.comm);
//
//    GetTime( timeEnd1 );
//#if ( _DEBUGlevel_ >= 0 )
//    statusOFS << "Time for Gaussian MPI_Bcast is " <<
//      timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
//#endif
//
//    GetTime( timeSta1 );
//
//    blas::Gemm( 'N', 'N', ntotLocal, numPre, numStateTotal, 1.0, 
//        phiRow.Data(), ntotLocal, G.Data(), numStateTotal, 0.0,
//        localphiGRow.Data(), ntotLocal );
//
//    blas::Gemm( 'N', 'N', ntotLocal, numPre, numStateTotal, 1.0, 
//        psiRow.Data(), ntotLocal, G.Data(), numStateTotal, 0.0,
//        localpsiGRow.Data(), ntotLocal );
//
//    GetTime( timeEnd1 );
//#if ( _DEBUGlevel_ >= 0 )
//    statusOFS << "Time for localphiGRow and localpsiGRow Gemm is " <<
//      timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
//#endif
//
//    // Step 2: Pivoted QR decomposition  for the Hadamard product of
//    // the compressed matrix. Transpose format for QRCP
//
//    // NOTE: All processors should have the same ntotLocalMG
//    ntotMG = ntotLocalMG * mpisize;
//
//    DblNumMat MG( numPre*numPre, ntotLocalMG );
//    SetValue( MG, 0.0 );
//
//    GetTime( timeSta1 );
//
//    for( Int j = 0; j < numPre; j++ ){
//      for( Int i = 0; i < numPre; i++ ){
//        for( Int ir = 0; ir < ntotLocal; ir++ ){
//          MG(i+j*numPre,ir) = localphiGRow(ir,i) * localpsiGRow(ir,j);
//        }
//      }
//    }
//
//    GetTime( timeEnd1 );
//#if ( _DEBUGlevel_ >= 0 )
//    statusOFS << "Time for computing MG from localphiGRow and localpsiGRow is " <<
//      timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
//#endif
//
//    DblNumVec tau(ntotMG);
//    pivQR_.Resize(ntotMG);
//    SetValue( pivQR_, 0 ); // Important. Otherwise QRCP uses piv as initial guess
//    // Q factor does not need to be used
//
//    for( Int k = 0; k < ntotMG; k++ ){
//      tau[k] = 0.0;
//    }
//
//
//    Real timeQRCPSta, timeQRCPEnd;
//    GetTime( timeQRCPSta );
//
//    if(0){  
//      lapack::QRCP( numPre*numPre, ntotMG, MG.Data(), numPre*numPre, 
//          pivQR_.Data(), tau.Data() );
//    }//
//
//
//    if(0){ // ScaLAPACL QRCP
//      Int contxt;
//      Int nprow, npcol, myrow, mycol, info;
//      Cblacs_get(0, 0, &contxt);
//      nprow = 1;
//      npcol = mpisize;
//
//      Cblacs_gridinit(&contxt, "C", nprow, npcol);
//      Cblacs_gridinfo(contxt, &nprow, &npcol, &myrow, &mycol);
//      Int desc_MG[9];
//
//      Int irsrc = 0;
//      Int icsrc = 0;
//
//      Int mb_MG = numPre*numPre;
//      Int nb_MG = ntotLocalMG;
//
//      // FIXME The current routine does not actually allow ntotLocal to be different on different processors.
//      // This must be fixed.
//      SCALAPACK(descinit)(&desc_MG[0], &mb_MG, &ntotMG, &mb_MG, &nb_MG, &irsrc, 
//          &icsrc, &contxt, &mb_MG, &info);
//
//      IntNumVec pivQRTmp(ntotMG), pivQRLocal(ntotMG);
//      if( mb_MG > ntot ){
//        std::ostringstream msg;
//        msg << "numPre*numPre > ntot. The number of grid points is perhaps too small!" << std::endl;
//        ErrorHandling( msg.str().c_str() );
//      }
//      // DiagR is only for debugging purpose
//      //        DblNumVec diagRLocal( mb_MG );
//      //        DblNumVec diagR( mb_MG );
//
//      SetValue( pivQRTmp, 0 );
//      SetValue( pivQRLocal, 0 );
//      SetValue( pivQR_, 0 );
//
//
//      //        SetValue( diagRLocal, 0.0 );
//      //        SetValue( diagR, 0.0 );
//
//      if(0) {
//        scalapack::QRCPF( mb_MG, ntotMG, MG.Data(), &desc_MG[0], 
//            pivQRTmp.Data(), tau.Data() );
//      }
//
//      if(1) {
//        scalapack::QRCPR( mb_MG, ntotMG, numMu_, MG.Data(), &desc_MG[0], 
//            pivQRTmp.Data(), tau.Data(), 80, 40 );
//      }
//
//
//      // Combine the local pivQRTmp to global pivQR_
//      for( Int j = 0; j < ntotLocalMG; j++ ){
//        pivQRLocal[j + mpirank * ntotLocalMG] = pivQRTmp[j];
//      }
//
//      //        std::cout << "diag of MG = " << std::endl;
//      //        if(mpirank == 0){
//      //          std::cout << pivQRLocal << std::endl;
//      //          for( Int j = 0; j < mb_MG; j++ ){
//      //            std::cout << MG(j,j) << std::endl;
//      //          }
//      //        }
//      MPI_Allreduce( pivQRLocal.Data(), pivQR_.Data(), 
//          ntotMG, MPI_INT, MPI_SUM, domain_.comm );
//
//
//      if(contxt >= 0) {
//        Cblacs_gridexit( contxt );
//      }
//
//    } //ScaLAPACL QRCP
//
//
//    if(1){ //ScaLAPACL QRCP 2D
//
//      Int contxt1D, contxt2D;
//      Int nprow1D, npcol1D, myrow1D, mycol1D, info1D;
//      Int nprow2D, npcol2D, myrow2D, mycol2D, info2D;
//
//      Int ncols1D, nrows1D, lld1D; 
//      Int ncols2D, nrows2D, lld2D; 
//
//      Int desc_MG1D[9];
//      Int desc_MG2D[9];
//
//      Int m_MG = numPre*numPre;
//      Int n_MG = ntotMG;
//
//      Int mb_MG1D = numPre*numPre;
//      Int nb_MG1D = ntotLocalMG;
//
//      nprow1D = 1;
//      npcol1D = mpisize;
//
//      Cblacs_get(0, 0, &contxt1D);
//      Cblacs_gridinit(&contxt1D, "C", nprow1D, npcol1D);
//      Cblacs_gridinfo(contxt1D, &nprow1D, &npcol1D, &myrow1D, &mycol1D);
//
//      nrows1D = SCALAPACK(numroc)(&m_MG, &mb_MG1D, &myrow1D, &I_ZERO, &nprow1D);
//      ncols1D = SCALAPACK(numroc)(&n_MG, &nb_MG1D, &mycol1D, &I_ZERO, &npcol1D);
//
//      lld1D = std::max( nrows1D, 1 );
//
//      SCALAPACK(descinit)(desc_MG1D, &m_MG, &n_MG, &mb_MG1D, &nb_MG1D, &I_ZERO, 
//          &I_ZERO, &contxt1D, &lld1D, &info1D);
//
//      for( Int i = std::min(mpisize, IRound(sqrt(double(mpisize*(n_MG/m_MG))))); 
//          i <= mpisize; i++){
//        npcol2D = i; nprow2D = mpisize / npcol2D;
//        if( (npcol2D >= nprow2D) && (nprow2D * npcol2D == mpisize) ) break;
//      }
//
//      Cblacs_get(0, 0, &contxt2D);
//      //Cblacs_gridinit(&contxt2D, "C", nprow2D, npcol2D);
//
//      IntNumVec pmap(mpisize);
//      for ( Int i = 0; i < mpisize; i++ ){
//        pmap[i] = i;
//      }
//      Cblacs_gridmap(&contxt2D, &pmap[0], nprow2D, nprow2D, npcol2D);
//
//      Int m_MG2DBlocksize = BlockSizeScaLAPACK;
//      Int n_MG2DBlocksize = BlockSizeScaLAPACK;
//
//      Int m_MG2Local = m_MG/(m_MG2DBlocksize*nprow2D)*m_MG2DBlocksize;
//      Int n_MG2Local = n_MG/(n_MG2DBlocksize*npcol2D)*n_MG2DBlocksize;
//
//      MPI_Comm rowComm = MPI_COMM_NULL;
//      MPI_Comm colComm = MPI_COMM_NULL;
//
//      Int mpirankRow, mpisizeRow, mpirankCol, mpisizeCol;
//
//      MPI_Comm_split( domain_.comm, mpirank / nprow2D, mpirank, &rowComm );
//      MPI_Comm_split( domain_.comm, mpirank % nprow2D, mpirank, &colComm );
//
//      MPI_Comm_rank(rowComm, &mpirankRow);
//      MPI_Comm_size(rowComm, &mpisizeRow);
//
//      MPI_Comm_rank(colComm, &mpirankCol);
//      MPI_Comm_size(colComm, &mpisizeCol);
//
//      if((m_MG % (m_MG2DBlocksize * nprow2D))!= 0){ 
//        if(mpirankRow < ((m_MG % (m_MG2DBlocksize*nprow2D)) / m_MG2DBlocksize)){
//          m_MG2Local = m_MG2Local + m_MG2DBlocksize;
//        }
//        if(((m_MG % (m_MG2DBlocksize*nprow2D)) % m_MG2DBlocksize) != 0){
//          if(mpirankRow == ((m_MG % (m_MG2DBlocksize*nprow2D)) / m_MG2DBlocksize)){
//            m_MG2Local = m_MG2Local + ((m_MG % (m_MG2DBlocksize*nprow2D)) % m_MG2DBlocksize);
//          }
//        }
//      }
//
//      if((n_MG % (n_MG2DBlocksize * npcol2D))!= 0){ 
//        if(mpirankCol < ((n_MG % (n_MG2DBlocksize*npcol2D)) / n_MG2DBlocksize)){
//          n_MG2Local = n_MG2Local + n_MG2DBlocksize;
//        }
//        if(((n_MG % (n_MG2DBlocksize*nprow2D)) % n_MG2DBlocksize) != 0){
//          if(mpirankCol == ((n_MG % (n_MG2DBlocksize*npcol2D)) / n_MG2DBlocksize)){
//            n_MG2Local = n_MG2Local + ((n_MG % (n_MG2DBlocksize*nprow2D)) % n_MG2DBlocksize);
//          }
//        }
//      }
//
//      if(contxt2D >= 0){
//        Cblacs_gridinfo(contxt2D, &nprow2D, &npcol2D, &myrow2D, &mycol2D);
//        nrows2D = SCALAPACK(numroc)(&m_MG, &m_MG2DBlocksize, &myrow2D, &I_ZERO, &nprow2D);
//        ncols2D = SCALAPACK(numroc)(&n_MG, &n_MG2DBlocksize, &mycol2D, &I_ZERO, &npcol2D);
//        lld2D = std::max( nrows2D, 1 );
//      }
//
//      SCALAPACK(descinit)(desc_MG2D, &m_MG, &n_MG, &m_MG2DBlocksize, &n_MG2DBlocksize, &I_ZERO, 
//          &I_ZERO, &contxt2D, &lld2D, &info2D);
//
//      IntNumVec pivQRTmp(ntotMG), pivQRLocal(ntotMG);
//      if( m_MG > ntot ){
//        std::ostringstream msg;
//        msg << "numPre*numPre > ntot. The number of grid points is perhaps too small!" << std::endl;
//        ErrorHandling( msg.str().c_str() );
//      }
//      // DiagR is only for debugging purpose
//      //        DblNumVec diagRLocal( mb_MG );
//      //        DblNumVec diagR( mb_MG );
//
//      SetValue( pivQRTmp, 0 );
//      SetValue( pivQRLocal, 0 );
//      SetValue( pivQR_, 0 );
//
//      DblNumMat&  MG1D = MG;
//      DblNumMat  MG2D (m_MG2Local, n_MG2Local);
//
//      SCALAPACK(pdgemr2d)(&m_MG, &n_MG, MG1D.Data(), &I_ONE, &I_ONE, desc_MG1D, 
//          MG2D.Data(), &I_ONE, &I_ONE, desc_MG2D, &contxt1D );
//
//      if(contxt2D >= 0){
//
//        Real timeQRCP1, timeQRCP2;
//        GetTime( timeQRCP1 );
//
//        scalapack::QRCPF( m_MG, n_MG, MG2D.Data(), desc_MG2D, pivQRTmp.Data(), tau.Data() );
//
//        //scalapack::QRCPR( m_MG, n_MG, numMu_, MG2D.Data(), desc_MG2D, pivQRTmp.Data(), tau.Data(), BlockSizeScaLAPACK, 32);
//
//        GetTime( timeQRCP2 );
//
//#if ( _DEBUGlevel_ >= 0 )
//        statusOFS << "Time for QRCP is " << timeQRCP2 - timeQRCP1 << " [s]" << std::endl;
//#endif
//
//      }
//
//      // Redistribute back eigenvectors
//      SetValue(MG1D, 0.0 );
//
//      SCALAPACK(pdgemr2d)( &m_MG, &n_MG, MG2D.Data(), &I_ONE, &I_ONE, desc_MG2D,
//          MG1D.Data(), &I_ONE, &I_ONE, desc_MG1D, &contxt1D );
//
//      // Combine the local pivQRTmp to global pivQR_
//      for( Int j = 0; j < n_MG2Local; j++ ){
//        pivQRLocal[ (j / n_MG2DBlocksize) * n_MG2DBlocksize * npcol2D + mpirankCol * n_MG2DBlocksize + j % n_MG2DBlocksize] = pivQRTmp[j];
//      }
//
//      MPI_Allreduce( pivQRLocal.Data(), pivQR_.Data(), 
//          ntotMG, MPI_INT, MPI_SUM, colComm );
//
//      if(contxt2D >= 0) {
//        Cblacs_gridexit( contxt2D );
//      }
//
//      if(contxt1D >= 0) {
//        Cblacs_gridexit( contxt1D );
//      }
//
//    } // if(1) ScaLAPACL QRCP
//
//    GetTime( timeQRCPEnd );
//
//    //statusOFS << std::endl<< "All processors exit with abort in spinor.cpp." << std::endl;
//
//#if ( _DEBUGlevel_ >= 0 )
//    statusOFS << "Time for QRCP alone is " <<
//      timeQRCPEnd - timeQRCPSta << " [s]" << std::endl << std::endl;
//#endif
//
//    if(0){
//      Real tolR = std::abs(MG(numMu_-1,numMu_-1)/MG(0,0));
//      statusOFS << "numMu_ = " << numMu_ << std::endl;
//      statusOFS << "|R(numMu-1,numMu-1)/R(0,0)| = " << tolR << std::endl;
//    }
//
//    GetTime( timeEnd );
//#if ( _DEBUGlevel_ >= 0 )
//    statusOFS << "Time for density fitting with QRCP is " <<
//      timeEnd - timeSta << " [s]" << std::endl << std::endl;
//#endif
//
//    // Dump out pivQR_
//    if(0){
//      std::ostringstream muStream;
//      serialize( pivQR_, muStream, NO_MASK );
//      SharedWrite( "pivQR", muStream );
//    }
//  }
//
//  // Load pivQR_ file
//  if(0){
//    statusOFS << "Loading pivQR file.." << std::endl;
//    std::istringstream muStream;
//    SharedRead( "pivQR", muStream );
//    deserialize( pivQR_, muStream, NO_MASK );
//  }
//
//  // *********************************************************************
//  // Compute the interpolation matrix via the density matrix formulation
//  // *********************************************************************
//
//  GetTime( timeSta );
//
//  // PhiMu is scaled by the occupation number to reflect the "true" density matrix
//  //IntNumVec pivMu(numMu_);
//  IntNumVec pivMu1(numMu_);
//
//  for( Int mu = 0; mu < numMu_; mu++ ){
//    //pivMu(mu) = pivQR_(mu);
//    pivMu1(mu) = pivQR_(mu);
//  }
//
//
//  if(ntot % mpisize != 0){
//    for( Int mu = 0; mu < numMu_; mu++ ){
//      Int k1 = (pivMu1(mu) / ntotLocalMG) - (ntot % mpisize);
//      if(k1 > 0){
//        pivMu1(mu) = pivQR_(mu) - k1;
//      }
//    }
//  }
//
//  GetTime( timeSta1 );
//
//  DblNumMat psiMuCol(numStateLocal, numMu_);
//  DblNumMat phiMuCol(numStateLocal, numMu_);
//  SetValue( psiMuCol, 0.0 );
//  SetValue( phiMuCol, 0.0 );
//
//  for (Int k=0; k<numStateLocal; k++) {
//    for (Int mu=0; mu<numMu_; mu++) {
//      psiMuCol(k, mu) = psiCol(pivMu1(mu),k);
//      phiMuCol(k, mu) = phiCol(pivMu1(mu),k) * occupationRate[(k * mpisize + mpirank)];
//    }
//  }
//
//  DblNumMat psiMu2D(nrowsNeNu2D, ncolsNeNu2D);
//  DblNumMat phiMu2D(nrowsNeNu2D, ncolsNeNu2D);
//  SetValue( psiMu2D, 0.0 );
//  SetValue( phiMu2D, 0.0 );
//
//  SCALAPACK(pdgemr2d)(&Ne, &Nu, psiMuCol.Data(), &I_ONE, &I_ONE, desc_NeNu1D, 
//      psiMu2D.Data(), &I_ONE, &I_ONE, desc_NeNu2D, &contxt11 );
//
//  SCALAPACK(pdgemr2d)(&Ne, &Nu, phiMuCol.Data(), &I_ONE, &I_ONE, desc_NeNu1D, 
//      phiMu2D.Data(), &I_ONE, &I_ONE, desc_NeNu2D, &contxt11 );
//
//  GetTime( timeEnd1 );
//
//#if ( _DEBUGlevel_ >= 0 )
//  statusOFS << "Time for computing psiMuRow and phiMuRow is " <<
//    timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
//#endif
//
//  GetTime( timeSta1 );
//
//  DblNumMat psi2D(nrowsNgNe2D, ncolsNgNe2D);
//  DblNumMat phi2D(nrowsNgNe2D, ncolsNgNe2D);
//  SetValue( psi2D, 0.0 );
//  SetValue( phi2D, 0.0 );
//
//  SCALAPACK(pdgemr2d)(&Ng, &Ne, psiCol.Data(), &I_ONE, &I_ONE, desc_NgNe1D, 
//      psi2D.Data(), &I_ONE, &I_ONE, desc_NgNe2D, &contxt1 );
//  SCALAPACK(pdgemr2d)(&Ng, &Ne, phiCol.Data(), &I_ONE, &I_ONE, desc_NgNe1D, 
//      phi2D.Data(), &I_ONE, &I_ONE, desc_NgNe2D, &contxt1 );
//
//  DblNumMat PpsiMu2D(nrowsNgNu2D, ncolsNgNu2D);
//  DblNumMat PphiMu2D(nrowsNgNu2D, ncolsNgNu2D);
//  SetValue( PpsiMu2D, 0.0 );
//  SetValue( PphiMu2D, 0.0 );
//
//  SCALAPACK(pdgemm)("N", "N", &Ng, &Nu, &Ne, 
//      &D_ONE,
//      psi2D.Data(), &I_ONE, &I_ONE, desc_NgNe2D,
//      psiMu2D.Data(), &I_ONE, &I_ONE, desc_NeNu2D, 
//      &D_ZERO,
//      PpsiMu2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D);
//
//  SCALAPACK(pdgemm)("N", "N", &Ng, &Nu, &Ne, 
//      &D_ONE,
//      phi2D.Data(), &I_ONE, &I_ONE, desc_NgNe2D,
//      phiMu2D.Data(), &I_ONE, &I_ONE, desc_NeNu2D, 
//      &D_ZERO,
//      PphiMu2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D);
//
//  GetTime( timeEnd1 );
//
//#if ( _DEBUGlevel_ >= 0 )
//  statusOFS << "Time for PpsiMu and PphiMu GEMM is " <<
//    timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
//#endif
//
//  GetTime( timeSta1 );
//
//  DblNumMat Xi2D(nrowsNgNu2D, ncolsNgNu2D);
//  SetValue( Xi2D, 0.0 );
//
//  Real* Xi2DPtr = Xi2D.Data();
//  Real* PpsiMu2DPtr = PpsiMu2D.Data();
//  Real* PphiMu2DPtr = PphiMu2D.Data();
//
//  for( Int g = 0; g < nrowsNgNu2D * ncolsNgNu2D; g++ ){
//    Xi2DPtr[g] = PpsiMu2DPtr[g] * PphiMu2DPtr[g];
//  }
//
//  DblNumMat Xi1D(nrowsNgNu1D, ncolsNuNu1D);
//  SetValue( Xi1D, 0.0 );
//
//  SCALAPACK(pdgemr2d)(&Ng, &Nu, Xi2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D, 
//      Xi1D.Data(), &I_ONE, &I_ONE, desc_NgNu1D, &contxt2 );
//
//  DblNumMat PMuNu1D(nrowsNuNu1D, ncolsNuNu1D);
//  SetValue( PMuNu1D, 0.0 );
//
//  for (Int mu=0; mu<nrowsNuNu1D; mu++) {
//    for (Int nu=0; nu<ncolsNuNu1D; nu++) {
//      PMuNu1D(mu, nu) = Xi1D(pivMu1(mu),nu);
//    }
//  }
//
//  DblNumMat PMuNu2D(nrowsNuNu2D, ncolsNuNu2D);
//  SetValue( PMuNu2D, 0.0 );
//
//  SCALAPACK(pdgemr2d)(&Nu, &Nu, PMuNu1D.Data(), &I_ONE, &I_ONE, desc_NuNu1D, 
//      PMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D, &contxt1 );
//
//  GetTime( timeEnd1 );
//
//#if ( _DEBUGlevel_ >= 0 )
//  statusOFS << "Time for computing PMuNu is " <<
//    timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
//#endif
//
//
//  //Method 1
//  if(0){
//
//    GetTime( timeSta1 );
//
//    SCALAPACK(pdpotrf)("L", &Nu, PMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D, &info2);
//
//    GetTime( timeEnd1 );
//
//#if ( _DEBUGlevel_ >= 0 )
//    statusOFS << "Time for PMuNu Potrf is " <<
//      timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
//#endif
//
//    GetTime( timeSta1 );
//
//    SCALAPACK(pdtrsm)("R", "L", "T", "N", &Ng, &Nu, &D_ONE,
//        PMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D,
//        Xi2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D);
//
//    SCALAPACK(pdtrsm)("R", "L", "N", "N", &Ng, &Nu, &D_ONE,
//        PMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D,
//        Xi2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D);
//
//    GetTime( timeEnd1 );
//
//#if ( _DEBUGlevel_ >= 0 )
//    statusOFS << "Time for PMuNu and Xi pdtrsm is " <<
//      timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
//#endif
//
//  }
//
//
//  //Method 2
//  if(1){
//
//    GetTime( timeSta1 );
//
//    SCALAPACK(pdpotrf)("L", &Nu, PMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D, &info2);
//    SCALAPACK(pdpotri)("L", &Nu, PMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D, &info2);
//
//    GetTime( timeEnd1 );
//
//#if ( _DEBUGlevel_ >= 0 )
//    statusOFS << "Time for PMuNu Potrf is " <<
//      timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
//#endif
//
//    GetTime( timeSta1 );
//
//    DblNumMat PMuNu2DTemp(nrowsNuNu2D, ncolsNuNu2D);
//    SetValue( PMuNu2DTemp, 0.0 );
//
//    lapack::Lacpy( 'A', nrowsNuNu2D, ncolsNuNu2D, PMuNu2D.Data(), nrowsNuNu2D, PMuNu2DTemp.Data(), nrowsNuNu2D );
//
//    SCALAPACK(pdtradd)("U", "T", &Nu, &Nu, 
//        &D_ONE,
//        PMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D, 
//        &D_ZERO,
//        PMuNu2DTemp.Data(), &I_ONE, &I_ONE, desc_NuNu2D);
//
//    DblNumMat Xi2DTemp(nrowsNgNu2D, ncolsNgNu2D);
//    SetValue( Xi2DTemp, 0.0 );
//
//    SCALAPACK(pdgemm)("N", "N", &Ng, &Nu, &Nu, 
//        &D_ONE,
//        Xi2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D,
//        PMuNu2DTemp.Data(), &I_ONE, &I_ONE, desc_NuNu2D, 
//        &D_ZERO,
//        Xi2DTemp.Data(), &I_ONE, &I_ONE, desc_NgNu2D);
//
//    SetValue( Xi2D, 0.0 );
//    lapack::Lacpy( 'A', nrowsNgNu2D, ncolsNgNu2D, Xi2DTemp.Data(), nrowsNgNu2D, Xi2D.Data(), nrowsNgNu2D );
//
//    GetTime( timeEnd1 );
//
//#if ( _DEBUGlevel_ >= 0 )
//    statusOFS << "Time for PMuNu and Xi pdgemm is " <<
//      timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
//#endif
//
//  }
//
//
//  //Method 3
//  if(0){
//
//    GetTime( timeSta1 );
//
//    SCALAPACK(pdpotrf)("L", &Nu, PMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D, &info2);
//    SCALAPACK(pdpotri)("L", &Nu, PMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D, &info2);
//
//    GetTime( timeEnd1 );
//
//#if ( _DEBUGlevel_ >= 0 )
//    statusOFS << "Time for PMuNu Potrf is " <<
//      timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
//#endif
//
//    GetTime( timeSta1 );
//
//    DblNumMat Xi2DTemp(nrowsNgNu2D, ncolsNgNu2D);
//    SetValue( Xi2DTemp, 0.0 );
//
//    SCALAPACK(pdsymm)("R", "L", &Ng, &Nu, 
//        &D_ONE,
//        PMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D,
//        Xi2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D,
//        &D_ZERO,
//        Xi2DTemp.Data(), &I_ONE, &I_ONE, desc_NgNu2D);
//
//    SetValue( Xi2D, 0.0 );
//    lapack::Lacpy( 'A', nrowsNgNu2D, ncolsNgNu2D, Xi2DTemp.Data(), nrowsNgNu2D, Xi2D.Data(), nrowsNgNu2D );
//
//    GetTime( timeEnd1 );
//
//#if ( _DEBUGlevel_ >= 0 )
//    statusOFS << "Time for PMuNu and Xi pdsymm is " <<
//      timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
//#endif
//
//  }
//
//
//  //Method 4
//  if(0){
//
//    GetTime( timeSta1 );
//
//    DblNumMat Xi2DTemp(nrowsNuNg2D, ncolsNuNg2D);
//    SetValue( Xi2DTemp, 0.0 );
//
//    DblNumMat PMuNu2DTemp(ncolsNuNu2D, nrowsNuNu2D);
//    SetValue( PMuNu2DTemp, 0.0 );
//
//    SCALAPACK(pdgeadd)("T", &Nu, &Ng,
//        &D_ONE,
//        Xi2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D,
//        &D_ZERO,
//        Xi2DTemp.Data(), &I_ONE, &I_ONE, desc_NuNg2D);
//
//    SCALAPACK(pdgeadd)("T", &Nu, &Nu,
//        &D_ONE,
//        PMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D,
//        &D_ZERO,
//        PMuNu2DTemp.Data(), &I_ONE, &I_ONE, desc_NuNu2D);
//
//    Int lwork=-1, info;
//    double dummyWork;
//
//    SCALAPACK(pdgels)("N", &Nu, &Nu, &Ng, 
//        PMuNu2DTemp.Data(), &I_ONE, &I_ONE, desc_NuNu2D,
//        Xi2DTemp.Data(), &I_ONE, &I_ONE, desc_NuNg2D,
//        &dummyWork, &lwork, &info);
//
//    lwork = dummyWork;
//    std::vector<double> work(lwork);
//
//    SCALAPACK(pdgels)("N", &Nu, &Nu, &Ng, 
//        PMuNu2DTemp.Data(), &I_ONE, &I_ONE, desc_NuNu2D,
//        Xi2DTemp.Data(), &I_ONE, &I_ONE, desc_NuNg2D,
//        &work[0], &lwork, &info);
//
//    SetValue( Xi2D, 0.0 );
//    SCALAPACK(pdgeadd)("T", &Ng, &Nu,
//        &D_ONE,
//        Xi2DTemp.Data(), &I_ONE, &I_ONE, desc_NuNg2D,
//        &D_ZERO,
//        Xi2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D);
//
//    GetTime( timeEnd1 );
//
//#if ( _DEBUGlevel_ >= 0 )
//    statusOFS << "Time for PMuNu and Xi pdgels is " <<
//      timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
//#endif
//
//  }
//
//
//  GetTime( timeEnd );
//
//#if ( _DEBUGlevel_ >= 0 )
//  statusOFS << "Time for computing the interpolation vectors is " <<
//    timeEnd - timeSta << " [s]" << std::endl << std::endl;
//#endif
//
//
//
//  // *********************************************************************
//  // Solve the Poisson equations.
//  // Store VXi separately. This is not the most memory efficient
//  // implementation
//  // *********************************************************************
//
//  DblNumMat VXi2D(nrowsNgNu2D, ncolsNgNu2D);
//
//  {
//    GetTime( timeSta );
//    // XiCol used for both input and output
//    DblNumMat XiCol(ntot, numMuLocal);
//    SetValue(XiCol, 0.0 );
//
//    SCALAPACK(pdgemr2d)(&Ng, &Nu, Xi2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D, 
//        XiCol.Data(), &I_ONE, &I_ONE, desc_NgNu1D, &contxt2 );
//
//    GetTime( timeSta );
//    for( Int mu = 0; mu < numMuLocal; mu++ ){
//      blas::Copy( ntot,  XiCol.VecData(mu), 1, fft.inputVecR2C.Data(), 1 );
//
//      FFTWExecute ( fft, fft.forwardPlanR2C );
//
//      for( Int ig = 0; ig < ntotR2C; ig++ ){
//        fft.outputVecR2C(ig) *= -exxFraction * exxgkkR2C(ig);
//      }
//
//      FFTWExecute ( fft, fft.backwardPlanR2C );
//
//      blas::Copy( ntot, fft.inputVecR2C.Data(), 1, XiCol.VecData(mu), 1 );
//
//    } // for (mu)
//
//    SetValue(VXi2D, 0.0 );
//    SCALAPACK(pdgemr2d)(&Ng, &Nu, XiCol.Data(), &I_ONE, &I_ONE, desc_NgNu1D, 
//        VXi2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D, &contxt1 );
//
//    GetTime( timeEnd );
//#if ( _DEBUGlevel_ >= 0 )
//    statusOFS << "Time for solving Poisson-like equations is " <<
//      timeEnd - timeSta << " [s]" << std::endl << std::endl;
//#endif
//  }
//
//  // Prepare for the computation of the M matrix
//  GetTime( timeSta );
//
//  DblNumMat MMatMuNu2D(nrowsNuNu2D, ncolsNuNu2D);
//  SetValue(MMatMuNu2D, 0.0 );
//
//  SCALAPACK(pdgemm)("T", "N", &Nu, &Nu, &Ng, 
//      &D_MinusONE,
//      Xi2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D,
//      VXi2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D, 
//      &D_ZERO,
//      MMatMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D);
//
//  DblNumMat phiMuNu2D(nrowsNuNu2D, ncolsNuNu2D);
//  SCALAPACK(pdgemm)("T", "N", &Nu, &Nu, &Ne, 
//      &D_ONE,
//      phiMu2D.Data(), &I_ONE, &I_ONE, desc_NeNu2D,
//      phiMu2D.Data(), &I_ONE, &I_ONE, desc_NeNu2D, 
//      &D_ZERO,
//      phiMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D);
//
//  Real* MMat2DPtr = MMatMuNu2D.Data();
//  Real* phi2DPtr  = phiMuNu2D.Data();
//
//  for( Int g = 0; g < nrowsNuNu2D * ncolsNuNu2D; g++ ){
//    MMat2DPtr[g] *= phi2DPtr[g];
//  }
//
//  GetTime( timeEnd );
//#if ( _DEBUGlevel_ >= 0 )
//  statusOFS << "Time for preparing the M matrix is " <<
//    timeEnd - timeSta << " [s]" << std::endl << std::endl;
//#endif
//
//  // *********************************************************************
//  // Compute the exchange potential and the symmetrized inner product
//  // *********************************************************************
//
//  GetTime( timeSta );
//
//  // Rewrite VXi by VXi.*PcolPhi
//  Real* VXi2DPtr = VXi2D.Data();
//  Real* PphiMu2DPtr1 = PphiMu2D.Data();
//  for( Int g = 0; g < nrowsNgNu2D * ncolsNgNu2D; g++ ){
//    VXi2DPtr[g] *= PphiMu2DPtr1[g];
//  }
//
//  // NOTE: a3 must be zero in order to compute the M matrix later
//  DblNumMat a32D( nrowsNgNe2D, ncolsNgNe2D );
//  SetValue( a32D, 0.0 );
//
//  SCALAPACK(pdgemm)("N", "T", &Ng, &Ne, &Nu, 
//      &D_ONE,
//      VXi2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D,
//      psiMu2D.Data(), &I_ONE, &I_ONE, desc_NeNu2D, 
//      &D_ZERO,
//      a32D.Data(), &I_ONE, &I_ONE, desc_NgNe2D);
//
//  DblNumMat a3Col( ntot, numStateLocal );
//  SetValue(a3Col, 0.0 );
//
//  SCALAPACK(pdgemr2d)(&Ng, &Ne, a32D.Data(), &I_ONE, &I_ONE, desc_NgNe2D, 
//      a3Col.Data(), &I_ONE, &I_ONE, desc_NgNe1D, &contxt2 );
//
//  lapack::Lacpy( 'A', ntot, numStateLocal, a3Col.Data(), ntot, a3.Data(), ntot );
//
//  GetTime( timeEnd );
//#if ( _DEBUGlevel_ >= 0 )
//  statusOFS << "Time for computing the exchange potential is " <<
//    timeEnd - timeSta << " [s]" << std::endl << std::endl;
//#endif
//
//  // Compute the matrix VxMat = -Psi'* vexxPsi and symmetrize
//  // vexxPsi (a3) must be zero before entering this routine
//  VxMat.Resize( numStateTotal, numStateTotal );
//
//  GetTime( timeSta );
//
//  if(1){
//
//    DblNumMat VxMat2D( nrowsNeNe2D, ncolsNeNe2D );
//    DblNumMat VxMatTemp2D( nrowsNuNe2D, ncolsNuNe2D );
//
//    SCALAPACK(pdgemm)("N", "T", &Nu, &Ne, &Nu, 
//        &D_ONE,
//        MMatMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D,
//        psiMu2D.Data(), &I_ONE, &I_ONE, desc_NeNu2D, 
//        &D_ZERO,
//        VxMatTemp2D.Data(), &I_ONE, &I_ONE, desc_NuNe2D);
//
//    SCALAPACK(pdgemm)("N", "N", &Ne, &Ne, &Nu, 
//        &D_ONE,
//        psiMu2D.Data(), &I_ONE, &I_ONE, desc_NeNu2D, 
//        VxMatTemp2D.Data(), &I_ONE, &I_ONE, desc_NuNe2D, 
//        &D_ZERO,
//        VxMat2D.Data(), &I_ONE, &I_ONE, desc_NeNe2D);
//
//    SCALAPACK(pdgemr2d)(&Ne, &Ne, VxMat2D.Data(), &I_ONE, &I_ONE, desc_NeNe2D, 
//        VxMat.Data(), &I_ONE, &I_ONE, desc_NeNe0D, &contxt2 );
//
//    //if(mpirank == 0){
//    //  MPI_Bcast( VxMat.Data(), Ne * Ne, MPI_DOUBLE, 0, domain_.comm );
//    //}
//
//  }
//  GetTime( timeEnd );
//#if ( _DEBUGlevel_ >= 0 )
//  statusOFS << "Time for computing VxMat in the sym format is " <<
//    timeEnd - timeSta << " [s]" << std::endl << std::endl;
//#endif
//
//  if(contxt0 >= 0) {
//    Cblacs_gridexit( contxt0 );
//  }
//
//  if(contxt1 >= 0) {
//    Cblacs_gridexit( contxt1 );
//  }
//
//  if(contxt11 >= 0) {
//    Cblacs_gridexit( contxt11 );
//  }
//
//  if(contxt2 >= 0) {
//    Cblacs_gridexit( contxt2 );
//  }
//
//  MPI_Barrier(domain_.comm);
//
//  return ;
//}        // -----  end of method Spinor::AddMultSpinorEXXDF4  ----- 


// 2D MPI communication for matrix
// Update: 6/26/2017
//void Spinor::AddMultSpinorEXXDF5 ( Fourier& fft, 
//    const NumTns<Real>& phi,
//    const DblNumVec& exxgkkR2C,
//    Real  exxFraction,
//    Real  numSpin,
//    const DblNumVec& occupationRate,
//    const Real numMuFac,
//    const Real numGaussianRandomFac,
//    const Int numProcScaLAPACK,  
//    const Int BlockSizeScaLAPACK,  
//    NumTns<Real>& a3, 
//    NumMat<Real>& VxMat,
//    bool isFixColumnDF )
//{
//  Real timeSta, timeEnd;
//  Real timeSta1, timeEnd1;
//
//  if( !fft.isInitialized ){
//    ErrorHandling("Fourier is not prepared.");
//  }
//
//  MPI_Barrier(domain_.comm);
//  int mpirank;  MPI_Comm_rank(domain_.comm, &mpirank);
//  int mpisize;  MPI_Comm_size(domain_.comm, &mpisize);
//
//  Index3& numGrid = domain_.numGrid;
//  Index3& numGridFine = domain_.numGridFine;
//
//  Int ntot     = domain_.NumGridTotal();
//  Int ntotFine = domain_.NumGridTotalFine();
//  Int ntotR2C = fft.numGridTotalR2C;
//  Int ntotR2CFine = fft.numGridTotalR2CFine;
//  Int ncom = wavefun_.n();
//  Int numStateLocal = wavefun_.p();
//  Int numStateTotal = numStateTotal_;
//
//  Int ncomPhi = phi.n();
//
//  Real vol = domain_.Volume();
//
//  if( ncomPhi != 1 || ncom != 1 ){
//    ErrorHandling("Spin polarized case not implemented.");
//  }
//
//  if( fft.domain.NumGridTotal() != ntot ){
//    ErrorHandling("Domain size does not match.");
//  }
//
//
//  // *********************************************************************
//  // Perform interpolative separable density fitting
//  // *********************************************************************
//
//  //numMu_ = std::min(IRound(numStateTotal*numMuFac), ntot);
//  numMu_ = IRound(numStateTotal*numMuFac);
//
//  Int numPre = IRound(std::sqrt(numMu_*numGaussianRandomFac));
//  if( numPre > numStateTotal ){
//    ErrorHandling("numMu is too large for interpolative separable density fitting!");
//  }
//
//  statusOFS << "ntot          = " << ntot << std::endl;
//  statusOFS << "numMu         = " << numMu_ << std::endl;
//  statusOFS << "numPre        = " << numPre << std::endl;
//  statusOFS << "numPre*numPre = " << numPre * numPre << std::endl;
//
//  // Convert the column partition to row partition
//  Int numStateBlocksize = numStateTotal / mpisize;
//  //Int ntotBlocksize = ntot / mpisize;
//
//  Int numMuBlocksize = numMu_ / mpisize;
//
//  Int numStateLocal1 = numStateBlocksize;
//  //Int ntotLocal = ntotBlocksize;
//
//  Int numMuLocal = numMuBlocksize;
//
//  if(mpirank < (numStateTotal % mpisize)){
//    numStateLocal1 = numStateBlocksize + 1;
//  }
//
//  if(numStateLocal !=  numStateLocal1){
//    statusOFS << "numStateLocal = " << numStateLocal << " numStateLocal1 = " << numStateLocal1 << std::endl;
//    ErrorHandling("The size is not right in interpolative separable density fitting!");
//  }
//
//  if(mpirank < (numMu_ % mpisize)){
//    numMuLocal = numMuBlocksize + 1;
//  }
//
//  //if(mpirank < (ntot % mpisize)){
//  //  ntotLocal = ntotBlocksize + 1;
//  //}
//
//  //huwei 
//  //2D MPI commucation for all the matrix
//
//  Int I_ONE = 1, I_ZERO = 0;
//  double D_ONE = 1.0;
//  double D_ZERO = 0.0;
//  double D_MinusONE = -1.0;
//
//  Int contxt0, contxt1, contxt11, contxt2;
//  Int nprow0, npcol0, myrow0, mycol0, info0;
//  Int nprow1, npcol1, myrow1, mycol1, info1;
//  Int nprow11, npcol11, myrow11, mycol11, info11;
//  Int nprow2, npcol2, myrow2, mycol2, info2;
//
//  Int ncolsNgNe1DCol, nrowsNgNe1DCol, lldNgNe1DCol; 
//  Int ncolsNgNe1DRow, nrowsNgNe1DRow, lldNgNe1DRow; 
//  Int ncolsNgNe2D, nrowsNgNe2D, lldNgNe2D; 
//  Int ncolsNgNu1D, nrowsNgNu1D, lldNgNu1D; 
//  Int ncolsNgNu2D, nrowsNgNu2D, lldNgNu2D; 
//  Int ncolsNuNg2D, nrowsNuNg2D, lldNuNg2D; 
//  Int ncolsNeNe0D, nrowsNeNe0D, lldNeNe0D; 
//  Int ncolsNeNe2D, nrowsNeNe2D, lldNeNe2D; 
//  Int ncolsNuNu1D, nrowsNuNu1D, lldNuNu1D; 
//  Int ncolsNuNu2D, nrowsNuNu2D, lldNuNu2D; 
//  Int ncolsNeNu1D, nrowsNeNu1D, lldNeNu1D; 
//  Int ncolsNeNu2D, nrowsNeNu2D, lldNeNu2D; 
//  Int ncolsNuNe2D, nrowsNuNe2D, lldNuNe2D; 
//
//  Int desc_NgNe1DCol[9];
//  Int desc_NgNe1DRow[9];
//  Int desc_NgNe2D[9];
//  Int desc_NgNu1D[9];
//  Int desc_NgNu2D[9];
//  Int desc_NuNg2D[9];
//  Int desc_NeNe0D[9];
//  Int desc_NeNe2D[9];
//  Int desc_NuNu1D[9];
//  Int desc_NuNu2D[9];
//  Int desc_NeNu1D[9];
//  Int desc_NeNu2D[9];
//  Int desc_NuNe2D[9];
//
//  Int Ng = ntot;
//  Int Ne = numStateTotal_; 
//  Int Nu = numMu_; 
//
//  // 0D MPI
//  nprow0 = 1;
//  npcol0 = mpisize;
//
//  Cblacs_get(0, 0, &contxt0);
//  Cblacs_gridinit(&contxt0, "C", nprow0, npcol0);
//  Cblacs_gridinfo(contxt0, &nprow0, &npcol0, &myrow0, &mycol0);
//
//  SCALAPACK(descinit)(desc_NeNe0D, &Ne, &Ne, &Ne, &Ne, &I_ZERO, &I_ZERO, &contxt0, &Ne, &info0);
//
//  // 1D MPI
//  nprow1 = 1;
//  npcol1 = mpisize;
//
//  Cblacs_get(0, 0, &contxt1);
//  Cblacs_gridinit(&contxt1, "C", nprow1, npcol1);
//  Cblacs_gridinfo(contxt1, &nprow1, &npcol1, &myrow1, &mycol1);
//
//  //desc_NgNe1DCol
//  if(contxt1 >= 0){
//    nrowsNgNe1DCol = SCALAPACK(numroc)(&Ng, &Ng, &myrow1, &I_ZERO, &nprow1);
//    ncolsNgNe1DCol = SCALAPACK(numroc)(&Ne, &I_ONE, &mycol1, &I_ZERO, &npcol1);
//    lldNgNe1DCol = std::max( nrowsNgNe1DCol, 1 );
//  }    
//
//  SCALAPACK(descinit)(desc_NgNe1DCol, &Ng, &Ne, &Ng, &I_ONE, &I_ZERO, 
//      &I_ZERO, &contxt1, &lldNgNe1DCol, &info1);
//
//  nprow11 = mpisize;
//  npcol11 = 1;
//
//  Cblacs_get(0, 0, &contxt11);
//  Cblacs_gridinit(&contxt11, "C", nprow11, npcol11);
//  Cblacs_gridinfo(contxt11, &nprow11, &npcol11, &myrow11, &mycol11);
// 
//  Int BlockSizeScaLAPACKTemp = BlockSizeScaLAPACK; 
//  //desc_NgNe1DRow
//  if(contxt11 >= 0){
//    nrowsNgNe1DRow = SCALAPACK(numroc)(&Ng, &BlockSizeScaLAPACKTemp, &myrow11, &I_ZERO, &nprow11);
//    ncolsNgNe1DRow = SCALAPACK(numroc)(&Ne, &Ne, &mycol11, &I_ZERO, &npcol11);
//    lldNgNe1DRow = std::max( nrowsNgNe1DRow, 1 );
//  }    
//
//  SCALAPACK(descinit)(desc_NgNe1DRow, &Ng, &Ne, &BlockSizeScaLAPACKTemp, &Ne, &I_ZERO, 
//      &I_ZERO, &contxt11, &lldNgNe1DRow, &info11);
//
//  //desc_NeNu1D
//  if(contxt11 >= 0){
//    nrowsNeNu1D = SCALAPACK(numroc)(&Ne, &I_ONE, &myrow11, &I_ZERO, &nprow11);
//    ncolsNeNu1D = SCALAPACK(numroc)(&Nu, &Nu, &mycol11, &I_ZERO, &npcol11);
//    lldNeNu1D = std::max( nrowsNeNu1D, 1 );
//  }    
//
//  SCALAPACK(descinit)(desc_NeNu1D, &Ne, &Nu, &I_ONE, &Nu, &I_ZERO, 
//      &I_ZERO, &contxt11, &lldNeNu1D, &info11);
//
//  //desc_NgNu1D
//  if(contxt1 >= 0){
//    nrowsNgNu1D = SCALAPACK(numroc)(&Ng, &Ng, &myrow1, &I_ZERO, &nprow1);
//    ncolsNgNu1D = SCALAPACK(numroc)(&Nu, &I_ONE, &mycol1, &I_ZERO, &npcol1);
//    lldNgNu1D = std::max( nrowsNgNu1D, 1 );
//  }    
//
//  SCALAPACK(descinit)(desc_NgNu1D, &Ng, &Nu, &Ng, &I_ONE, &I_ZERO, 
//      &I_ZERO, &contxt1, &lldNgNu1D, &info1);
//
//  //desc_NuNu1D
//  if(contxt1 >= 0){
//    nrowsNuNu1D = SCALAPACK(numroc)(&Nu, &Nu, &myrow1, &I_ZERO, &nprow1);
//    ncolsNuNu1D = SCALAPACK(numroc)(&Nu, &I_ONE, &mycol1, &I_ZERO, &npcol1);
//    lldNuNu1D = std::max( nrowsNuNu1D, 1 );
//  }    
//
//  SCALAPACK(descinit)(desc_NuNu1D, &Nu, &Nu, &Nu, &I_ONE, &I_ZERO, 
//      &I_ZERO, &contxt1, &lldNuNu1D, &info1);
//
//
//  // 2D MPI
//  for( Int i = IRound(sqrt(double(mpisize))); i <= mpisize; i++){
//    nprow2 = i; npcol2 = mpisize / nprow2;
//    if( (nprow2 >= npcol2) && (nprow2 * npcol2 == mpisize) ) break;
//  }
//
//  Cblacs_get(0, 0, &contxt2);
//  //Cblacs_gridinit(&contxt2, "C", nprow2, npcol2);
//
//  IntNumVec pmap2(mpisize);
//  for ( Int i = 0; i < mpisize; i++ ){
//    pmap2[i] = i;
//  }
//  Cblacs_gridmap(&contxt2, &pmap2[0], nprow2, nprow2, npcol2);
//
//  Int mb2 = BlockSizeScaLAPACK;
//  Int nb2 = BlockSizeScaLAPACK;
//
//  //desc_NgNe2D
//  if(contxt2 >= 0){
//    Cblacs_gridinfo(contxt2, &nprow2, &npcol2, &myrow2, &mycol2);
//    nrowsNgNe2D = SCALAPACK(numroc)(&Ng, &mb2, &myrow2, &I_ZERO, &nprow2);
//    ncolsNgNe2D = SCALAPACK(numroc)(&Ne, &nb2, &mycol2, &I_ZERO, &npcol2);
//    lldNgNe2D = std::max( nrowsNgNe2D, 1 );
//  }
//
//  SCALAPACK(descinit)(desc_NgNe2D, &Ng, &Ne, &mb2, &nb2, &I_ZERO, 
//      &I_ZERO, &contxt2, &lldNgNe2D, &info2);
//
//  //desc_NgNu2D
//  if(contxt2 >= 0){
//    Cblacs_gridinfo(contxt2, &nprow2, &npcol2, &myrow2, &mycol2);
//    nrowsNgNu2D = SCALAPACK(numroc)(&Ng, &mb2, &myrow2, &I_ZERO, &nprow2);
//    ncolsNgNu2D = SCALAPACK(numroc)(&Nu, &nb2, &mycol2, &I_ZERO, &npcol2);
//    lldNgNu2D = std::max( nrowsNgNu2D, 1 );
//  }
//
//  SCALAPACK(descinit)(desc_NgNu2D, &Ng, &Nu, &mb2, &nb2, &I_ZERO, 
//      &I_ZERO, &contxt2, &lldNgNu2D, &info2);
//  
//  //desc_NuNg2D
//  if(contxt2 >= 0){
//    Cblacs_gridinfo(contxt2, &nprow2, &npcol2, &myrow2, &mycol2);
//    nrowsNuNg2D = SCALAPACK(numroc)(&Nu, &mb2, &myrow2, &I_ZERO, &nprow2);
//    ncolsNuNg2D = SCALAPACK(numroc)(&Ng, &nb2, &mycol2, &I_ZERO, &npcol2);
//    lldNuNg2D = std::max( nrowsNuNg2D, 1 );
//  }
//
//  SCALAPACK(descinit)(desc_NuNg2D, &Nu, &Ng, &mb2, &nb2, &I_ZERO, 
//      &I_ZERO, &contxt2, &lldNuNg2D, &info2);
//
//  //desc_NeNe2D
//  if(contxt2 >= 0){
//    Cblacs_gridinfo(contxt2, &nprow2, &npcol2, &myrow2, &mycol2);
//    nrowsNeNe2D = SCALAPACK(numroc)(&Ne, &mb2, &myrow2, &I_ZERO, &nprow2);
//    ncolsNeNe2D = SCALAPACK(numroc)(&Ne, &nb2, &mycol2, &I_ZERO, &npcol2);
//    lldNeNe2D = std::max( nrowsNeNe2D, 1 );
//  }
//
//  SCALAPACK(descinit)(desc_NeNe2D, &Ne, &Ne, &mb2, &nb2, &I_ZERO, 
//      &I_ZERO, &contxt2, &lldNeNe2D, &info2);
//
//  //desc_NuNu2D
//  if(contxt2 >= 0){
//    Cblacs_gridinfo(contxt2, &nprow2, &npcol2, &myrow2, &mycol2);
//    nrowsNuNu2D = SCALAPACK(numroc)(&Nu, &mb2, &myrow2, &I_ZERO, &nprow2);
//    ncolsNuNu2D = SCALAPACK(numroc)(&Nu, &nb2, &mycol2, &I_ZERO, &npcol2);
//    lldNuNu2D = std::max( nrowsNuNu2D, 1 );
//  }
//
//  SCALAPACK(descinit)(desc_NuNu2D, &Nu, &Nu, &mb2, &nb2, &I_ZERO, 
//      &I_ZERO, &contxt2, &lldNuNu2D, &info2);
//
//  //desc_NeNu2D
//  if(contxt2 >= 0){
//    Cblacs_gridinfo(contxt2, &nprow2, &npcol2, &myrow2, &mycol2);
//    nrowsNeNu2D = SCALAPACK(numroc)(&Ne, &mb2, &myrow2, &I_ZERO, &nprow2);
//    ncolsNeNu2D = SCALAPACK(numroc)(&Nu, &nb2, &mycol2, &I_ZERO, &npcol2);
//    lldNeNu2D = std::max( nrowsNeNu2D, 1 );
//  }
//
//  SCALAPACK(descinit)(desc_NeNu2D, &Ne, &Nu, &mb2, &nb2, &I_ZERO, 
//      &I_ZERO, &contxt2, &lldNeNu2D, &info2);
//
//  //desc_NuNe2D
//  if(contxt2 >= 0){
//    Cblacs_gridinfo(contxt2, &nprow2, &npcol2, &myrow2, &mycol2);
//    nrowsNuNe2D = SCALAPACK(numroc)(&Nu, &mb2, &myrow2, &I_ZERO, &nprow2);
//    ncolsNuNe2D = SCALAPACK(numroc)(&Ne, &nb2, &mycol2, &I_ZERO, &npcol2);
//    lldNuNe2D = std::max( nrowsNuNe2D, 1 );
//  }
//
//  SCALAPACK(descinit)(desc_NuNe2D, &Nu, &Ne, &mb2, &nb2, &I_ZERO, 
//      &I_ZERO, &contxt2, &lldNuNe2D, &info2);
//
//  DblNumMat phiCol( ntot, numStateLocal );
//  SetValue( phiCol, 0.0 );
//
//  DblNumMat psiCol( ntot, numStateLocal );
//  SetValue( psiCol, 0.0 );
//
//  lapack::Lacpy( 'A', ntot, numStateLocal, phi.Data(), ntot, phiCol.Data(), ntot );
//  lapack::Lacpy( 'A', ntot, numStateLocal, wavefun_.Data(), ntot, psiCol.Data(), ntot );
//
//  // Computing the indices is optional
//
//
//  Int ntotLocal = nrowsNgNe1DRow; 
//  Int ntotLocalMG = nrowsNgNe1DRow;
//  Int ntotMG = ntot;
//    
//  //if( (ntot % mpisize) == 0 ){
//  //  ntotLocalMG = ntotBlocksize;
//  //}
//  //else{
//  //  ntotLocalMG = ntotBlocksize + 1;
// // }
//
//  if( isFixColumnDF == false ){
//
//    GetTime( timeSta );
//
//    DblNumMat localphiGRow( ntotLocal, numPre );
//    SetValue( localphiGRow, 0.0 );
//
//    DblNumMat localpsiGRow( ntotLocal, numPre );
//    SetValue( localpsiGRow, 0.0 );
//
//    DblNumMat G(numStateTotal, numPre);
//    SetValue( G, 0.0 );
//
//    DblNumMat phiRow( ntotLocal, numStateTotal );
//    SetValue( phiRow, 0.0 );
//
//    DblNumMat psiRow( ntotLocal, numStateTotal );
//    SetValue( psiRow, 0.0 );
//
//    SCALAPACK(pdgemr2d)(&Ng, &Ne, psiCol.Data(), &I_ONE, &I_ONE, desc_NgNe1DCol, 
//        psiRow.Data(), &I_ONE, &I_ONE, desc_NgNe1DRow, &contxt1 );
//
//    SCALAPACK(pdgemr2d)(&Ng, &Ne, phiCol.Data(), &I_ONE, &I_ONE, desc_NgNe1DCol, 
//        phiRow.Data(), &I_ONE, &I_ONE, desc_NgNe1DRow, &contxt1 );
//
//    // Step 1: Pre-compression of the wavefunctions. This uses
//    // multiplication with orthonormalized random Gaussian matrices
//    if ( mpirank == 0) {
//      GaussianRandom(G);
//      lapack::Orth( numStateTotal, numPre, G.Data(), numStateTotal );
//    }
//
//    GetTime( timeSta1 );
//
//    MPI_Bcast(G.Data(), numStateTotal * numPre, MPI_DOUBLE, 0, domain_.comm);
//
//    GetTime( timeEnd1 );
//#if ( _DEBUGlevel_ >= 0 )
//    statusOFS << "Time for Gaussian MPI_Bcast is " <<
//      timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
//#endif
//
//    GetTime( timeSta1 );
//
//    blas::Gemm( 'N', 'N', ntotLocal, numPre, numStateTotal, 1.0, 
//        phiRow.Data(), ntotLocal, G.Data(), numStateTotal, 0.0,
//        localphiGRow.Data(), ntotLocal );
//
//    blas::Gemm( 'N', 'N', ntotLocal, numPre, numStateTotal, 1.0, 
//        psiRow.Data(), ntotLocal, G.Data(), numStateTotal, 0.0,
//        localpsiGRow.Data(), ntotLocal );
//
//    GetTime( timeEnd1 );
//#if ( _DEBUGlevel_ >= 0 )
//    statusOFS << "Time for localphiGRow and localpsiGRow Gemm is " <<
//      timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
//#endif
//
//    // Step 2: Pivoted QR decomposition  for the Hadamard product of
//    // the compressed matrix. Transpose format for QRCP
//
//    // NOTE: All processors should have the same ntotLocalMG
//    //ntotMG = ntotLocalMG * mpisize;
//
//    DblNumMat MG( numPre*numPre, ntotLocalMG );
//    //SetValue( MG, 0.0 );
//
//    GetTime( timeSta1 );
//
//    for( Int j = 0; j < numPre; j++ ){
//      for( Int i = 0; i < numPre; i++ ){
//        for( Int ir = 0; ir < ntotLocal; ir++ ){
//          MG(i+j*numPre,ir) = localphiGRow(ir,i) * localpsiGRow(ir,j);
//        }
//      }
//    }
//
//    GetTime( timeEnd1 );
//#if ( _DEBUGlevel_ >= 0 )
//    statusOFS << "Time for computing MG from localphiGRow and localpsiGRow is " <<
//      timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
//#endif
//
//    DblNumVec tau(ntotMG);
//    pivQR_.Resize(ntotMG);
//    SetValue( pivQR_, 0 ); // Important. Otherwise QRCP uses piv as initial guess
//    // Q factor does not need to be used
//
//    for( Int k = 0; k < ntotMG; k++ ){
//      tau[k] = 0.0;
//    }
//
//
//    Real timeQRCPSta, timeQRCPEnd;
//    GetTime( timeQRCPSta );
//
//    if(0){  
//      lapack::QRCP( numPre*numPre, ntotMG, MG.Data(), numPre*numPre, 
//          pivQR_.Data(), tau.Data() );
//    }//
//
//
//    if(0){ // ScaLAPACL QRCP
//      Int contxt;
//      Int nprow, npcol, myrow, mycol, info;
//      Cblacs_get(0, 0, &contxt);
//      nprow = 1;
//      npcol = mpisize;
//
//      Cblacs_gridinit(&contxt, "C", nprow, npcol);
//      Cblacs_gridinfo(contxt, &nprow, &npcol, &myrow, &mycol);
//      Int desc_MG[9];
//
//      Int irsrc = 0;
//      Int icsrc = 0;
//
//      Int mb_MG = numPre*numPre;
//      Int nb_MG = ntotLocalMG;
//
//      // FIXME The current routine does not actually allow ntotLocal to be different on different processors.
//      // This must be fixed.
//      SCALAPACK(descinit)(&desc_MG[0], &mb_MG, &ntotMG, &mb_MG, &nb_MG, &irsrc, 
//          &icsrc, &contxt, &mb_MG, &info);
//
//      IntNumVec pivQRTmp(ntotMG), pivQRLocal(ntotMG);
//      if( mb_MG > ntot ){
//        std::ostringstream msg;
//        msg << "numPre*numPre > ntot. The number of grid points is perhaps too small!" << std::endl;
//        ErrorHandling( msg.str().c_str() );
//      }
//      // DiagR is only for debugging purpose
//      //        DblNumVec diagRLocal( mb_MG );
//      //        DblNumVec diagR( mb_MG );
//
//      SetValue( pivQRTmp, 0 );
//      SetValue( pivQRLocal, 0 );
//      SetValue( pivQR_, 0 );
//
//
//      //        SetValue( diagRLocal, 0.0 );
//      //        SetValue( diagR, 0.0 );
//
//      if(0) {
//        scalapack::QRCPF( mb_MG, ntotMG, MG.Data(), &desc_MG[0], 
//            pivQRTmp.Data(), tau.Data() );
//      }
//
//      if(1) {
//        scalapack::QRCPR( mb_MG, ntotMG, numMu_, MG.Data(), &desc_MG[0], 
//            pivQRTmp.Data(), tau.Data(), 80, 40 );
//      }
//
//
//      // Combine the local pivQRTmp to global pivQR_
//      for( Int j = 0; j < ntotLocalMG; j++ ){
//        pivQRLocal[j + mpirank * ntotLocalMG] = pivQRTmp[j];
//      }
//
//      //        std::cout << "diag of MG = " << std::endl;
//      //        if(mpirank == 0){
//      //          std::cout << pivQRLocal << std::endl;
//      //          for( Int j = 0; j < mb_MG; j++ ){
//      //            std::cout << MG(j,j) << std::endl;
//      //          }
//      //        }
//      MPI_Allreduce( pivQRLocal.Data(), pivQR_.Data(), 
//          ntotMG, MPI_INT, MPI_SUM, domain_.comm );
//
//
//      if(contxt >= 0) {
//        Cblacs_gridexit( contxt );
//      }
//
//    } //ScaLAPACL QRCP
//
//
//    if(1){ //ScaLAPACL QRCP 2D
//
//      Int contxt1D, contxt2D;
//      Int nprow1D, npcol1D, myrow1D, mycol1D, info1D;
//      Int nprow2D, npcol2D, myrow2D, mycol2D, info2D;
//
//      Int ncols1D, nrows1D, lld1D; 
//      Int ncols2D, nrows2D, lld2D; 
//
//      Int desc_MG1D[9];
//      Int desc_MG2D[9];
//
//      Int m_MG = numPre*numPre;
//      Int n_MG = ntotMG;
//
//      Int mb_MG1D = numPre*numPre;
//      Int nb_MG1D = BlockSizeScaLAPACK;
//
//      nprow1D = 1;
//      npcol1D = mpisize;
//
//      Cblacs_get(0, 0, &contxt1D);
//      Cblacs_gridinit(&contxt1D, "C", nprow1D, npcol1D);
//      Cblacs_gridinfo(contxt1D, &nprow1D, &npcol1D, &myrow1D, &mycol1D);
//
//      nrows1D = SCALAPACK(numroc)(&m_MG, &mb_MG1D, &myrow1D, &I_ZERO, &nprow1D);
//      ncols1D = SCALAPACK(numroc)(&n_MG, &nb_MG1D, &mycol1D, &I_ZERO, &npcol1D);
//
//      lld1D = std::max( nrows1D, 1 );
//
//      SCALAPACK(descinit)(desc_MG1D, &m_MG, &n_MG, &mb_MG1D, &nb_MG1D, &I_ZERO, 
//          &I_ZERO, &contxt1D, &lld1D, &info1D);
//
//      for( Int i = std::min(mpisize, IRound(sqrt(double(mpisize*(n_MG/m_MG))))); 
//          i <= mpisize; i++){
//        npcol2D = i; nprow2D = mpisize / npcol2D;
//        if( (npcol2D >= nprow2D) && (nprow2D * npcol2D == mpisize) ) break;
//      }
//
//      Cblacs_get(0, 0, &contxt2D);
//      //Cblacs_gridinit(&contxt2D, "C", nprow2D, npcol2D);
//
//      IntNumVec pmap(mpisize);
//      for ( Int i = 0; i < mpisize; i++ ){
//        pmap[i] = i;
//      }
//      Cblacs_gridmap(&contxt2D, &pmap[0], nprow2D, nprow2D, npcol2D);
//
//      Int m_MG2DBlocksize = BlockSizeScaLAPACK;
//      Int n_MG2DBlocksize = BlockSizeScaLAPACK;
//
//      //Int m_MG2Local = m_MG/(m_MG2DBlocksize*nprow2D)*m_MG2DBlocksize;
//      //Int n_MG2Local = n_MG/(n_MG2DBlocksize*npcol2D)*n_MG2DBlocksize;
//      Int m_MG2Local, n_MG2Local;
//
//      MPI_Comm rowComm = MPI_COMM_NULL;
//      MPI_Comm colComm = MPI_COMM_NULL;
//
//      Int mpirankRow, mpisizeRow, mpirankCol, mpisizeCol;
//
//      MPI_Comm_split( domain_.comm, mpirank / nprow2D, mpirank, &rowComm );
//      MPI_Comm_split( domain_.comm, mpirank % nprow2D, mpirank, &colComm );
//
//      MPI_Comm_rank(rowComm, &mpirankRow);
//      MPI_Comm_size(rowComm, &mpisizeRow);
//
//      MPI_Comm_rank(colComm, &mpirankCol);
//      MPI_Comm_size(colComm, &mpisizeCol);
//
//      if(0){
//
//        if((m_MG % (m_MG2DBlocksize * nprow2D))!= 0){ 
//          if(mpirankRow < ((m_MG % (m_MG2DBlocksize*nprow2D)) / m_MG2DBlocksize)){
//            m_MG2Local = m_MG2Local + m_MG2DBlocksize;
//          }
//          if(((m_MG % (m_MG2DBlocksize*nprow2D)) % m_MG2DBlocksize) != 0){
//            if(mpirankRow == ((m_MG % (m_MG2DBlocksize*nprow2D)) / m_MG2DBlocksize)){
//              m_MG2Local = m_MG2Local + ((m_MG % (m_MG2DBlocksize*nprow2D)) % m_MG2DBlocksize);
//            }
//          }
//        }
//
//        if((n_MG % (n_MG2DBlocksize * npcol2D))!= 0){ 
//          if(mpirankCol < ((n_MG % (n_MG2DBlocksize*npcol2D)) / n_MG2DBlocksize)){
//            n_MG2Local = n_MG2Local + n_MG2DBlocksize;
//          }
//          if(((n_MG % (n_MG2DBlocksize*nprow2D)) % n_MG2DBlocksize) != 0){
//            if(mpirankCol == ((n_MG % (n_MG2DBlocksize*npcol2D)) / n_MG2DBlocksize)){
//              n_MG2Local = n_MG2Local + ((n_MG % (n_MG2DBlocksize*nprow2D)) % n_MG2DBlocksize);
//            }
//          }
//        }
//
//      } // if(0)
//
//
//      if(contxt2D >= 0){
//        Cblacs_gridinfo(contxt2D, &nprow2D, &npcol2D, &myrow2D, &mycol2D);
//        nrows2D = SCALAPACK(numroc)(&m_MG, &m_MG2DBlocksize, &myrow2D, &I_ZERO, &nprow2D);
//        ncols2D = SCALAPACK(numroc)(&n_MG, &n_MG2DBlocksize, &mycol2D, &I_ZERO, &npcol2D);
//        lld2D = std::max( nrows2D, 1 );
//      }
//
//      SCALAPACK(descinit)(desc_MG2D, &m_MG, &n_MG, &m_MG2DBlocksize, &n_MG2DBlocksize, &I_ZERO, 
//          &I_ZERO, &contxt2D, &lld2D, &info2D);
//      
//      m_MG2Local = nrows2D;
//      n_MG2Local = ncols2D;
//
//      IntNumVec pivQRTmp(ntotMG), pivQRLocal(ntotMG);
//      if( m_MG > ntot ){
//        std::ostringstream msg;
//        msg << "numPre*numPre > ntot. The number of grid points is perhaps too small!" << std::endl;
//        ErrorHandling( msg.str().c_str() );
//      }
//      // DiagR is only for debugging purpose
//      //        DblNumVec diagRLocal( mb_MG );
//      //        DblNumVec diagR( mb_MG );
//
//      SetValue( pivQRTmp, 0 );
//      SetValue( pivQRLocal, 0 );
//      SetValue( pivQR_, 0 );
//
//      DblNumMat&  MG1D = MG;
//      DblNumMat  MG2D (m_MG2Local, n_MG2Local);
//
//      SCALAPACK(pdgemr2d)(&m_MG, &n_MG, MG1D.Data(), &I_ONE, &I_ONE, desc_MG1D, 
//          MG2D.Data(), &I_ONE, &I_ONE, desc_MG2D, &contxt1D );
//
//      if(contxt2D >= 0){
//
//        Real timeQRCP1, timeQRCP2;
//        GetTime( timeQRCP1 );
//
//        scalapack::QRCPF( m_MG, n_MG, MG2D.Data(), desc_MG2D, pivQRTmp.Data(), tau.Data() );
//
//        //scalapack::QRCPR( m_MG, n_MG, numMu_, MG2D.Data(), desc_MG2D, pivQRTmp.Data(), tau.Data(), BlockSizeScaLAPACK, 32);
//
//        GetTime( timeQRCP2 );
//
//#if ( _DEBUGlevel_ >= 0 )
//        statusOFS << "Time for QRCP is " << timeQRCP2 - timeQRCP1 << " [s]" << std::endl;
//#endif
//
//      }
//
//      // Redistribute back eigenvectors
//      SetValue(MG1D, 0.0 );
//
//      SCALAPACK(pdgemr2d)( &m_MG, &n_MG, MG2D.Data(), &I_ONE, &I_ONE, desc_MG2D,
//          MG1D.Data(), &I_ONE, &I_ONE, desc_MG1D, &contxt1D );
//
//      // Combine the local pivQRTmp to global pivQR_
//      for( Int j = 0; j < n_MG2Local; j++ ){
//        pivQRLocal[ (j / n_MG2DBlocksize) * n_MG2DBlocksize * npcol2D + mpirankCol * n_MG2DBlocksize + j % n_MG2DBlocksize] = pivQRTmp[j];
//      }
//
//      MPI_Allreduce( pivQRLocal.Data(), pivQR_.Data(), 
//          ntotMG, MPI_INT, MPI_SUM, colComm );
//      
//      if( rowComm != MPI_COMM_NULL ) MPI_Comm_free( & rowComm );
//      if( colComm != MPI_COMM_NULL ) MPI_Comm_free( & colComm );
//
//      if(contxt2D >= 0) {
//        Cblacs_gridexit( contxt2D );
//      }
//
//      if(contxt1D >= 0) {
//        Cblacs_gridexit( contxt1D );
//      }
//
//    } // if(1) ScaLAPACL QRCP
//
//    GetTime( timeQRCPEnd );
//
//    //statusOFS << std::endl<< "All processors exit with abort in spinor.cpp." << std::endl;
//
//#if ( _DEBUGlevel_ >= 0 )
//    statusOFS << "Time for QRCP alone is " <<
//      timeQRCPEnd - timeQRCPSta << " [s]" << std::endl << std::endl;
//#endif
//
//    if(0){
//      Real tolR = std::abs(MG(numMu_-1,numMu_-1)/MG(0,0));
//      statusOFS << "numMu_ = " << numMu_ << std::endl;
//      statusOFS << "|R(numMu-1,numMu-1)/R(0,0)| = " << tolR << std::endl;
//    }
//
//    GetTime( timeEnd );
//#if ( _DEBUGlevel_ >= 0 )
//    statusOFS << "Time for density fitting with QRCP is " <<
//      timeEnd - timeSta << " [s]" << std::endl << std::endl;
//#endif
//
//    // Dump out pivQR_
//    if(0){
//      std::ostringstream muStream;
//      serialize( pivQR_, muStream, NO_MASK );
//      SharedWrite( "pivQR", muStream );
//    }
//  }
//
//  // Load pivQR_ file
//  if(0){
//    statusOFS << "Loading pivQR file.." << std::endl;
//    std::istringstream muStream;
//    SharedRead( "pivQR", muStream );
//    deserialize( pivQR_, muStream, NO_MASK );
//  }
//
//  // *********************************************************************
//  // Compute the interpolation matrix via the density matrix formulation
//  // *********************************************************************
//
//  GetTime( timeSta );
//
//  // PhiMu is scaled by the occupation number to reflect the "true" density matrix
//  //IntNumVec pivMu(numMu_);
//  IntNumVec pivMu1(numMu_);
//
//  for( Int mu = 0; mu < numMu_; mu++ ){
//    pivMu1(mu) = pivQR_(mu);
//  }
//
//
//  //if(ntot % mpisize != 0){
//  //  for( Int mu = 0; mu < numMu_; mu++ ){
//  //    Int k1 = (pivMu1(mu) / ntotLocalMG) - (ntot % mpisize);
//  //    if(k1 > 0){
//  //      pivMu1(mu) = pivQR_(mu) - k1;
//  //    }
//  //  }
//  //}
//
//  GetTime( timeSta1 );
//
//  DblNumMat psiMuCol(numStateLocal, numMu_);
//  DblNumMat phiMuCol(numStateLocal, numMu_);
//  SetValue( psiMuCol, 0.0 );
//  SetValue( phiMuCol, 0.0 );
//
//  for (Int k=0; k<numStateLocal; k++) {
//    for (Int mu=0; mu<numMu_; mu++) {
//      psiMuCol(k, mu) = psiCol(pivMu1(mu),k);
//      phiMuCol(k, mu) = phiCol(pivMu1(mu),k) * occupationRate[(k * mpisize + mpirank)];
//    }
//  }
//
//  DblNumMat psiMu2D(nrowsNeNu2D, ncolsNeNu2D);
//  DblNumMat phiMu2D(nrowsNeNu2D, ncolsNeNu2D);
//  SetValue( psiMu2D, 0.0 );
//  SetValue( phiMu2D, 0.0 );
//
//  SCALAPACK(pdgemr2d)(&Ne, &Nu, psiMuCol.Data(), &I_ONE, &I_ONE, desc_NeNu1D, 
//      psiMu2D.Data(), &I_ONE, &I_ONE, desc_NeNu2D, &contxt11 );
//
//  SCALAPACK(pdgemr2d)(&Ne, &Nu, phiMuCol.Data(), &I_ONE, &I_ONE, desc_NeNu1D, 
//      phiMu2D.Data(), &I_ONE, &I_ONE, desc_NeNu2D, &contxt11 );
//
//  GetTime( timeEnd1 );
//
//#if ( _DEBUGlevel_ >= 0 )
//  statusOFS << "Time for computing psiMuRow and phiMuRow is " <<
//    timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
//#endif
//
//  GetTime( timeSta1 );
//
//  DblNumMat psi2D(nrowsNgNe2D, ncolsNgNe2D);
//  DblNumMat phi2D(nrowsNgNe2D, ncolsNgNe2D);
//  SetValue( psi2D, 0.0 );
//  SetValue( phi2D, 0.0 );
//
//  SCALAPACK(pdgemr2d)(&Ng, &Ne, psiCol.Data(), &I_ONE, &I_ONE, desc_NgNe1DCol, 
//      psi2D.Data(), &I_ONE, &I_ONE, desc_NgNe2D, &contxt1 );
//  SCALAPACK(pdgemr2d)(&Ng, &Ne, phiCol.Data(), &I_ONE, &I_ONE, desc_NgNe1DCol, 
//      phi2D.Data(), &I_ONE, &I_ONE, desc_NgNe2D, &contxt1 );
//
//  DblNumMat PpsiMu2D(nrowsNgNu2D, ncolsNgNu2D);
//  DblNumMat PphiMu2D(nrowsNgNu2D, ncolsNgNu2D);
//  SetValue( PpsiMu2D, 0.0 );
//  SetValue( PphiMu2D, 0.0 );
//
//  SCALAPACK(pdgemm)("N", "N", &Ng, &Nu, &Ne, 
//      &D_ONE,
//      psi2D.Data(), &I_ONE, &I_ONE, desc_NgNe2D,
//      psiMu2D.Data(), &I_ONE, &I_ONE, desc_NeNu2D, 
//      &D_ZERO,
//      PpsiMu2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D);
//
//  SCALAPACK(pdgemm)("N", "N", &Ng, &Nu, &Ne, 
//      &D_ONE,
//      phi2D.Data(), &I_ONE, &I_ONE, desc_NgNe2D,
//      phiMu2D.Data(), &I_ONE, &I_ONE, desc_NeNu2D, 
//      &D_ZERO,
//      PphiMu2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D);
//
//  GetTime( timeEnd1 );
//
//#if ( _DEBUGlevel_ >= 0 )
//  statusOFS << "Time for PpsiMu and PphiMu GEMM is " <<
//    timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
//#endif
//
//  GetTime( timeSta1 );
//
//  DblNumMat Xi2D(nrowsNgNu2D, ncolsNgNu2D);
//  SetValue( Xi2D, 0.0 );
//
//  Real* Xi2DPtr = Xi2D.Data();
//  Real* PpsiMu2DPtr = PpsiMu2D.Data();
//  Real* PphiMu2DPtr = PphiMu2D.Data();
//
//  for( Int g = 0; g < nrowsNgNu2D * ncolsNgNu2D; g++ ){
//    Xi2DPtr[g] = PpsiMu2DPtr[g] * PphiMu2DPtr[g];
//  }
//
//  DblNumMat Xi1D(nrowsNgNu1D, ncolsNuNu1D);
//  SetValue( Xi1D, 0.0 );
//
//  SCALAPACK(pdgemr2d)(&Ng, &Nu, Xi2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D, 
//      Xi1D.Data(), &I_ONE, &I_ONE, desc_NgNu1D, &contxt2 );
//
//  DblNumMat PMuNu1D(nrowsNuNu1D, ncolsNuNu1D);
//  SetValue( PMuNu1D, 0.0 );
//
//  for (Int mu=0; mu<nrowsNuNu1D; mu++) {
//    for (Int nu=0; nu<ncolsNuNu1D; nu++) {
//      PMuNu1D(mu, nu) = Xi1D(pivMu1(mu),nu);
//    }
//  }
//
//  DblNumMat PMuNu2D(nrowsNuNu2D, ncolsNuNu2D);
//  SetValue( PMuNu2D, 0.0 );
//
//  SCALAPACK(pdgemr2d)(&Nu, &Nu, PMuNu1D.Data(), &I_ONE, &I_ONE, desc_NuNu1D, 
//      PMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D, &contxt1 );
//
//  GetTime( timeEnd1 );
//
//#if ( _DEBUGlevel_ >= 0 )
//  statusOFS << "Time for computing PMuNu is " <<
//    timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
//#endif
//
//
//  //Method 1
//  if(0){
//
//    GetTime( timeSta1 );
//
//    SCALAPACK(pdpotrf)("L", &Nu, PMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D, &info2);
//
//    GetTime( timeEnd1 );
//
//#if ( _DEBUGlevel_ >= 0 )
//    statusOFS << "Time for PMuNu Potrf is " <<
//      timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
//#endif
//
//    GetTime( timeSta1 );
//
//    SCALAPACK(pdtrsm)("R", "L", "T", "N", &Ng, &Nu, &D_ONE,
//        PMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D,
//        Xi2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D);
//
//    SCALAPACK(pdtrsm)("R", "L", "N", "N", &Ng, &Nu, &D_ONE,
//        PMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D,
//        Xi2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D);
//
//    GetTime( timeEnd1 );
//
//#if ( _DEBUGlevel_ >= 0 )
//    statusOFS << "Time for PMuNu and Xi pdtrsm is " <<
//      timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
//#endif
//
//  }
//
//
//  //Method 2
//  if(1){
//
//    GetTime( timeSta1 );
//
//    SCALAPACK(pdpotrf)("L", &Nu, PMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D, &info2);
//    SCALAPACK(pdpotri)("L", &Nu, PMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D, &info2);
//
//    GetTime( timeEnd1 );
//
//#if ( _DEBUGlevel_ >= 0 )
//    statusOFS << "Time for PMuNu Potrf is " <<
//      timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
//#endif
//
//    GetTime( timeSta1 );
//
//    DblNumMat PMuNu2DTemp(nrowsNuNu2D, ncolsNuNu2D);
//    SetValue( PMuNu2DTemp, 0.0 );
//
//    lapack::Lacpy( 'A', nrowsNuNu2D, ncolsNuNu2D, PMuNu2D.Data(), nrowsNuNu2D, PMuNu2DTemp.Data(), nrowsNuNu2D );
//
//    SCALAPACK(pdtradd)("U", "T", &Nu, &Nu, 
//        &D_ONE,
//        PMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D, 
//        &D_ZERO,
//        PMuNu2DTemp.Data(), &I_ONE, &I_ONE, desc_NuNu2D);
//
//    DblNumMat Xi2DTemp(nrowsNgNu2D, ncolsNgNu2D);
//    SetValue( Xi2DTemp, 0.0 );
//
//    SCALAPACK(pdgemm)("N", "N", &Ng, &Nu, &Nu, 
//        &D_ONE,
//        Xi2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D,
//        PMuNu2DTemp.Data(), &I_ONE, &I_ONE, desc_NuNu2D, 
//        &D_ZERO,
//        Xi2DTemp.Data(), &I_ONE, &I_ONE, desc_NgNu2D);
//
//    SetValue( Xi2D, 0.0 );
//    lapack::Lacpy( 'A', nrowsNgNu2D, ncolsNgNu2D, Xi2DTemp.Data(), nrowsNgNu2D, Xi2D.Data(), nrowsNgNu2D );
//
//    GetTime( timeEnd1 );
//
//#if ( _DEBUGlevel_ >= 0 )
//    statusOFS << "Time for PMuNu and Xi pdgemm is " <<
//      timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
//#endif
//
//  }
//
//
//  //Method 3
//  if(0){
//
//    GetTime( timeSta1 );
//
//    SCALAPACK(pdpotrf)("L", &Nu, PMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D, &info2);
//    SCALAPACK(pdpotri)("L", &Nu, PMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D, &info2);
//
//    GetTime( timeEnd1 );
//
//#if ( _DEBUGlevel_ >= 0 )
//    statusOFS << "Time for PMuNu Potrf is " <<
//      timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
//#endif
//
//    GetTime( timeSta1 );
//
//    DblNumMat Xi2DTemp(nrowsNgNu2D, ncolsNgNu2D);
//    SetValue( Xi2DTemp, 0.0 );
//
//    SCALAPACK(pdsymm)("R", "L", &Ng, &Nu, 
//        &D_ONE,
//        PMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D,
//        Xi2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D,
//        &D_ZERO,
//        Xi2DTemp.Data(), &I_ONE, &I_ONE, desc_NgNu2D);
//
//    SetValue( Xi2D, 0.0 );
//    lapack::Lacpy( 'A', nrowsNgNu2D, ncolsNgNu2D, Xi2DTemp.Data(), nrowsNgNu2D, Xi2D.Data(), nrowsNgNu2D );
//
//    GetTime( timeEnd1 );
//
//#if ( _DEBUGlevel_ >= 0 )
//    statusOFS << "Time for PMuNu and Xi pdsymm is " <<
//      timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
//#endif
//
//  }
//
//
//  //Method 4
//  if(0){
//
//    GetTime( timeSta1 );
//
//    DblNumMat Xi2DTemp(nrowsNuNg2D, ncolsNuNg2D);
//    SetValue( Xi2DTemp, 0.0 );
//
//    DblNumMat PMuNu2DTemp(ncolsNuNu2D, nrowsNuNu2D);
//    SetValue( PMuNu2DTemp, 0.0 );
//
//    SCALAPACK(pdgeadd)("T", &Nu, &Ng,
//        &D_ONE,
//        Xi2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D,
//        &D_ZERO,
//        Xi2DTemp.Data(), &I_ONE, &I_ONE, desc_NuNg2D);
//
//    SCALAPACK(pdgeadd)("T", &Nu, &Nu,
//        &D_ONE,
//        PMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D,
//        &D_ZERO,
//        PMuNu2DTemp.Data(), &I_ONE, &I_ONE, desc_NuNu2D);
//
//    Int lwork=-1, info;
//    double dummyWork;
//
//    SCALAPACK(pdgels)("N", &Nu, &Nu, &Ng, 
//        PMuNu2DTemp.Data(), &I_ONE, &I_ONE, desc_NuNu2D,
//        Xi2DTemp.Data(), &I_ONE, &I_ONE, desc_NuNg2D,
//        &dummyWork, &lwork, &info);
//
//    lwork = dummyWork;
//    std::vector<double> work(lwork);
//
//    SCALAPACK(pdgels)("N", &Nu, &Nu, &Ng, 
//        PMuNu2DTemp.Data(), &I_ONE, &I_ONE, desc_NuNu2D,
//        Xi2DTemp.Data(), &I_ONE, &I_ONE, desc_NuNg2D,
//        &work[0], &lwork, &info);
//
//    SetValue( Xi2D, 0.0 );
//    SCALAPACK(pdgeadd)("T", &Ng, &Nu,
//        &D_ONE,
//        Xi2DTemp.Data(), &I_ONE, &I_ONE, desc_NuNg2D,
//        &D_ZERO,
//        Xi2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D);
//
//    GetTime( timeEnd1 );
//
//#if ( _DEBUGlevel_ >= 0 )
//    statusOFS << "Time for PMuNu and Xi pdgels is " <<
//      timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
//#endif
//
//  }
//
//
//  GetTime( timeEnd );
//
//#if ( _DEBUGlevel_ >= 0 )
//  statusOFS << "Time for computing the interpolation vectors is " <<
//    timeEnd - timeSta << " [s]" << std::endl << std::endl;
//#endif
//
//
//
//  // *********************************************************************
//  // Solve the Poisson equations.
//  // Store VXi separately. This is not the most memory efficient
//  // implementation
//  // *********************************************************************
//
//  DblNumMat VXi2D(nrowsNgNu2D, ncolsNgNu2D);
//
//  {
//    GetTime( timeSta );
//    // XiCol used for both input and output
//    DblNumMat XiCol(ntot, numMuLocal);
//    SetValue(XiCol, 0.0 );
//
//    SCALAPACK(pdgemr2d)(&Ng, &Nu, Xi2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D, 
//        XiCol.Data(), &I_ONE, &I_ONE, desc_NgNu1D, &contxt2 );
//
//    GetTime( timeSta );
//    for( Int mu = 0; mu < numMuLocal; mu++ ){
//      blas::Copy( ntot,  XiCol.VecData(mu), 1, fft.inputVecR2C.Data(), 1 );
//
//      FFTWExecute ( fft, fft.forwardPlanR2C );
//
//      for( Int ig = 0; ig < ntotR2C; ig++ ){
//        fft.outputVecR2C(ig) *= -exxFraction * exxgkkR2C(ig);
//      }
//
//      FFTWExecute ( fft, fft.backwardPlanR2C );
//
//      blas::Copy( ntot, fft.inputVecR2C.Data(), 1, XiCol.VecData(mu), 1 );
//
//    } // for (mu)
//
//    SetValue(VXi2D, 0.0 );
//    SCALAPACK(pdgemr2d)(&Ng, &Nu, XiCol.Data(), &I_ONE, &I_ONE, desc_NgNu1D, 
//        VXi2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D, &contxt1 );
//
//    GetTime( timeEnd );
//#if ( _DEBUGlevel_ >= 0 )
//    statusOFS << "Time for solving Poisson-like equations is " <<
//      timeEnd - timeSta << " [s]" << std::endl << std::endl;
//#endif
//  }
//
//  // Prepare for the computation of the M matrix
//  GetTime( timeSta );
//
//  DblNumMat MMatMuNu2D(nrowsNuNu2D, ncolsNuNu2D);
//  SetValue(MMatMuNu2D, 0.0 );
//
//  SCALAPACK(pdgemm)("T", "N", &Nu, &Nu, &Ng, 
//      &D_MinusONE,
//      Xi2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D,
//      VXi2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D, 
//      &D_ZERO,
//      MMatMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D);
//
//  DblNumMat phiMuNu2D(nrowsNuNu2D, ncolsNuNu2D);
//  SCALAPACK(pdgemm)("T", "N", &Nu, &Nu, &Ne, 
//      &D_ONE,
//      phiMu2D.Data(), &I_ONE, &I_ONE, desc_NeNu2D,
//      phiMu2D.Data(), &I_ONE, &I_ONE, desc_NeNu2D, 
//      &D_ZERO,
//      phiMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D);
//
//  Real* MMat2DPtr = MMatMuNu2D.Data();
//  Real* phi2DPtr  = phiMuNu2D.Data();
//
//  for( Int g = 0; g < nrowsNuNu2D * ncolsNuNu2D; g++ ){
//    MMat2DPtr[g] *= phi2DPtr[g];
//  }
//
//  GetTime( timeEnd );
//#if ( _DEBUGlevel_ >= 0 )
//  statusOFS << "Time for preparing the M matrix is " <<
//    timeEnd - timeSta << " [s]" << std::endl << std::endl;
//#endif
//
//  // *********************************************************************
//  // Compute the exchange potential and the symmetrized inner product
//  // *********************************************************************
//
//  GetTime( timeSta );
//
//  // Rewrite VXi by VXi.*PcolPhi
//  Real* VXi2DPtr = VXi2D.Data();
//  Real* PphiMu2DPtr1 = PphiMu2D.Data();
//  for( Int g = 0; g < nrowsNgNu2D * ncolsNgNu2D; g++ ){
//    VXi2DPtr[g] *= PphiMu2DPtr1[g];
//  }
//
//  // NOTE: a3 must be zero in order to compute the M matrix later
//  DblNumMat a32D( nrowsNgNe2D, ncolsNgNe2D );
//  SetValue( a32D, 0.0 );
//
//  SCALAPACK(pdgemm)("N", "T", &Ng, &Ne, &Nu, 
//      &D_ONE,
//      VXi2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D,
//      psiMu2D.Data(), &I_ONE, &I_ONE, desc_NeNu2D, 
//      &D_ZERO,
//      a32D.Data(), &I_ONE, &I_ONE, desc_NgNe2D);
//
//  DblNumMat a3Col( ntot, numStateLocal );
//  SetValue(a3Col, 0.0 );
//
//  SCALAPACK(pdgemr2d)(&Ng, &Ne, a32D.Data(), &I_ONE, &I_ONE, desc_NgNe2D, 
//      a3Col.Data(), &I_ONE, &I_ONE, desc_NgNe1DCol, &contxt2 );
//
//  lapack::Lacpy( 'A', ntot, numStateLocal, a3Col.Data(), ntot, a3.Data(), ntot );
//
//  GetTime( timeEnd );
//#if ( _DEBUGlevel_ >= 0 )
//  statusOFS << "Time for computing the exchange potential is " <<
//    timeEnd - timeSta << " [s]" << std::endl << std::endl;
//#endif
//
//  // Compute the matrix VxMat = -Psi'* vexxPsi and symmetrize
//  // vexxPsi (a3) must be zero before entering this routine
//  VxMat.Resize( numStateTotal, numStateTotal );
//
//  GetTime( timeSta );
//
//  if(1){
//
//    DblNumMat VxMat2D( nrowsNeNe2D, ncolsNeNe2D );
//    DblNumMat VxMatTemp2D( nrowsNuNe2D, ncolsNuNe2D );
//
//    SCALAPACK(pdgemm)("N", "T", &Nu, &Ne, &Nu, 
//        &D_ONE,
//        MMatMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D,
//        psiMu2D.Data(), &I_ONE, &I_ONE, desc_NeNu2D, 
//        &D_ZERO,
//        VxMatTemp2D.Data(), &I_ONE, &I_ONE, desc_NuNe2D);
//
//    SCALAPACK(pdgemm)("N", "N", &Ne, &Ne, &Nu, 
//        &D_ONE,
//        psiMu2D.Data(), &I_ONE, &I_ONE, desc_NeNu2D, 
//        VxMatTemp2D.Data(), &I_ONE, &I_ONE, desc_NuNe2D, 
//        &D_ZERO,
//        VxMat2D.Data(), &I_ONE, &I_ONE, desc_NeNe2D);
//
//    SCALAPACK(pdgemr2d)(&Ne, &Ne, VxMat2D.Data(), &I_ONE, &I_ONE, desc_NeNe2D, 
//        VxMat.Data(), &I_ONE, &I_ONE, desc_NeNe0D, &contxt2 );
//
//    //if(mpirank == 0){
//    //  MPI_Bcast( VxMat.Data(), Ne * Ne, MPI_DOUBLE, 0, domain_.comm );
//    //}
//
//  }
//  GetTime( timeEnd );
//#if ( _DEBUGlevel_ >= 0 )
//  statusOFS << "Time for computing VxMat in the sym format is " <<
//    timeEnd - timeSta << " [s]" << std::endl << std::endl;
//#endif
//
//  if(contxt0 >= 0) {
//    Cblacs_gridexit( contxt0 );
//  }
//
//  if(contxt1 >= 0) {
//    Cblacs_gridexit( contxt1 );
//  }
//
//  if(contxt11 >= 0) {
//    Cblacs_gridexit( contxt11 );
//  }
//
//  if(contxt2 >= 0) {
//    Cblacs_gridexit( contxt2 );
//  }
//
//  MPI_Barrier(domain_.comm);
//
//  return ;
//}        // -----  end of method Spinor::AddMultSpinorEXXDF5  ----- 



// 2D MPI communication for matrix
// Update: 6/26/2017
void Spinor::AddMultSpinorEXXDF6 ( Fourier& fft, 
    const NumTns<Real>& phi,
    const DblNumVec& exxgkkR2C,
    Real  exxFraction,
    Real  numSpin,
    const DblNumVec& occupationRate,
    const Real numMuFac,
    const Real numGaussianRandomFac,
    const Int numProcScaLAPACK,  
    const Real hybridDFTolerance,
    const Int BlockSizeScaLAPACK,  
    NumTns<Real>& a3, 
    NumMat<Real>& VxMat,
    bool isFixColumnDF )
{
  Real timeSta, timeEnd;
  Real timeSta1, timeEnd1;

  if( !fft.isInitialized ){
    ErrorHandling("Fourier is not prepared.");
  }

  MPI_Barrier(domain_.comm);
  int mpirank;  MPI_Comm_rank(domain_.comm, &mpirank);
  int mpisize;  MPI_Comm_size(domain_.comm, &mpisize);

  Index3& numGrid = domain_.numGrid;
  Index3& numGridFine = domain_.numGridFine;

  Int ntot     = domain_.NumGridTotal();
  Int ntotFine = domain_.NumGridTotalFine();
  Int ntotR2C = fft.numGridTotalR2C;
  Int ntotR2CFine = fft.numGridTotalR2CFine;
  Int ncom = wavefun_.n();
  Int numStateLocal = wavefun_.p();
  Int numStateTotal = numStateTotal_;

  Int ncomPhi = phi.n();

  Real vol = domain_.Volume();

  if( ncomPhi != 1 || ncom != 1 ){
    ErrorHandling("Spin polarized case not implemented.");
  }

  if( fft.domain.NumGridTotal() != ntot ){
    ErrorHandling("Domain size does not match.");
  }


  // *********************************************************************
  // Perform interpolative separable density fitting
  // *********************************************************************

  //numMu_ = std::min(IRound(numStateTotal*numMuFac), ntot);
  numMu_ = IRound(numStateTotal*numMuFac);

  Int numPre = IRound(std::sqrt(numMu_*numGaussianRandomFac));
  if( numPre > numStateTotal ){
    ErrorHandling("numMu is too large for interpolative separable density fitting!");
  }

  statusOFS << "ntot          = " << ntot << std::endl;
  statusOFS << "numMu         = " << numMu_ << std::endl;
  statusOFS << "numPre        = " << numPre << std::endl;
  statusOFS << "numPre*numPre = " << numPre * numPre << std::endl;

  // Convert the column partition to row partition
  Int numStateBlocksize = numStateTotal / mpisize;
  //Int ntotBlocksize = ntot / mpisize;

  Int numMuBlocksize = numMu_ / mpisize;

  Int numStateLocal1 = numStateBlocksize;
  //Int ntotLocal = ntotBlocksize;

  Int numMuLocal = numMuBlocksize;

  if(mpirank < (numStateTotal % mpisize)){
    numStateLocal1 = numStateBlocksize + 1;
  }

  if(numStateLocal !=  numStateLocal1){
    statusOFS << "numStateLocal = " << numStateLocal << " numStateLocal1 = " << numStateLocal1 << std::endl;
    ErrorHandling("The size is not right in interpolative separable density fitting!");
  }

  if(mpirank < (numMu_ % mpisize)){
    numMuLocal = numMuBlocksize + 1;
  }

  //if(mpirank < (ntot % mpisize)){
  //  ntotLocal = ntotBlocksize + 1;
  //}

  //huwei 
  //2D MPI commucation for all the matrix

  Int I_ONE = 1, I_ZERO = 0;
  double D_ONE = 1.0;
  double D_ZERO = 0.0;
  double D_MinusONE = -1.0;

  Int contxt0, contxt1, contxt11, contxt2;
  Int nprow0, npcol0, myrow0, mycol0, info0;
  Int nprow1, npcol1, myrow1, mycol1, info1;
  Int nprow11, npcol11, myrow11, mycol11, info11;
  Int nprow2, npcol2, myrow2, mycol2, info2;

  Int ncolsNgNe1DCol, nrowsNgNe1DCol, lldNgNe1DCol; 
  Int ncolsNgNe1DRow, nrowsNgNe1DRow, lldNgNe1DRow; 
  Int ncolsNgNe2D, nrowsNgNe2D, lldNgNe2D; 
  Int ncolsNgNu1D, nrowsNgNu1D, lldNgNu1D; 
  Int ncolsNgNu2D, nrowsNgNu2D, lldNgNu2D; 
  Int ncolsNuNg2D, nrowsNuNg2D, lldNuNg2D; 
  Int ncolsNeNe0D, nrowsNeNe0D, lldNeNe0D; 
  Int ncolsNeNe2D, nrowsNeNe2D, lldNeNe2D; 
  Int ncolsNuNu1D, nrowsNuNu1D, lldNuNu1D; 
  Int ncolsNuNu2D, nrowsNuNu2D, lldNuNu2D; 
  Int ncolsNeNu1D, nrowsNeNu1D, lldNeNu1D; 
  Int ncolsNeNu2D, nrowsNeNu2D, lldNeNu2D; 
  Int ncolsNuNe2D, nrowsNuNe2D, lldNuNe2D; 

  Int desc_NgNe1DCol[9];
  Int desc_NgNe1DRow[9];
  Int desc_NgNe2D[9];
  Int desc_NgNu1D[9];
  Int desc_NgNu2D[9];
  Int desc_NuNg2D[9];
  Int desc_NeNe0D[9];
  Int desc_NeNe2D[9];
  Int desc_NuNu1D[9];
  Int desc_NuNu2D[9];
  Int desc_NeNu1D[9];
  Int desc_NeNu2D[9];
  Int desc_NuNe2D[9];

  Int Ng = ntot;
  Int Ne = numStateTotal_; 
  Int Nu = numMu_; 

  // 0D MPI
  nprow0 = 1;
  npcol0 = mpisize;

  Cblacs_get(0, 0, &contxt0);
  Cblacs_gridinit(&contxt0, "C", nprow0, npcol0);
  Cblacs_gridinfo(contxt0, &nprow0, &npcol0, &myrow0, &mycol0);

  SCALAPACK(descinit)(desc_NeNe0D, &Ne, &Ne, &Ne, &Ne, &I_ZERO, &I_ZERO, &contxt0, &Ne, &info0);

  // 1D MPI
  nprow1 = 1;
  npcol1 = mpisize;

  Cblacs_get(0, 0, &contxt1);
  Cblacs_gridinit(&contxt1, "C", nprow1, npcol1);
  Cblacs_gridinfo(contxt1, &nprow1, &npcol1, &myrow1, &mycol1);

  //desc_NgNe1DCol
  if(contxt1 >= 0){
    nrowsNgNe1DCol = SCALAPACK(numroc)(&Ng, &Ng, &myrow1, &I_ZERO, &nprow1);
    ncolsNgNe1DCol = SCALAPACK(numroc)(&Ne, &I_ONE, &mycol1, &I_ZERO, &npcol1);
    lldNgNe1DCol = std::max( nrowsNgNe1DCol, 1 );
  }    

  SCALAPACK(descinit)(desc_NgNe1DCol, &Ng, &Ne, &Ng, &I_ONE, &I_ZERO, 
      &I_ZERO, &contxt1, &lldNgNe1DCol, &info1);

  nprow11 = mpisize;
  npcol11 = 1;

  Cblacs_get(0, 0, &contxt11);
  Cblacs_gridinit(&contxt11, "C", nprow11, npcol11);
  Cblacs_gridinfo(contxt11, &nprow11, &npcol11, &myrow11, &mycol11);

  Int BlockSizeScaLAPACKTemp = BlockSizeScaLAPACK; 
  //desc_NgNe1DRow
  if(contxt11 >= 0){
    nrowsNgNe1DRow = SCALAPACK(numroc)(&Ng, &BlockSizeScaLAPACKTemp, &myrow11, &I_ZERO, &nprow11);
    ncolsNgNe1DRow = SCALAPACK(numroc)(&Ne, &Ne, &mycol11, &I_ZERO, &npcol11);
    lldNgNe1DRow = std::max( nrowsNgNe1DRow, 1 );
  }    

  SCALAPACK(descinit)(desc_NgNe1DRow, &Ng, &Ne, &BlockSizeScaLAPACKTemp, &Ne, &I_ZERO, 
      &I_ZERO, &contxt11, &lldNgNe1DRow, &info11);

  //desc_NeNu1D
  if(contxt11 >= 0){
    nrowsNeNu1D = SCALAPACK(numroc)(&Ne, &I_ONE, &myrow11, &I_ZERO, &nprow11);
    ncolsNeNu1D = SCALAPACK(numroc)(&Nu, &Nu, &mycol11, &I_ZERO, &npcol11);
    lldNeNu1D = std::max( nrowsNeNu1D, 1 );
  }    

  SCALAPACK(descinit)(desc_NeNu1D, &Ne, &Nu, &I_ONE, &Nu, &I_ZERO, 
      &I_ZERO, &contxt11, &lldNeNu1D, &info11);

  //desc_NgNu1D
  if(contxt1 >= 0){
    nrowsNgNu1D = SCALAPACK(numroc)(&Ng, &Ng, &myrow1, &I_ZERO, &nprow1);
    ncolsNgNu1D = SCALAPACK(numroc)(&Nu, &I_ONE, &mycol1, &I_ZERO, &npcol1);
    lldNgNu1D = std::max( nrowsNgNu1D, 1 );
  }    

  SCALAPACK(descinit)(desc_NgNu1D, &Ng, &Nu, &Ng, &I_ONE, &I_ZERO, 
      &I_ZERO, &contxt1, &lldNgNu1D, &info1);

  //desc_NuNu1D
  if(contxt1 >= 0){
    nrowsNuNu1D = SCALAPACK(numroc)(&Nu, &Nu, &myrow1, &I_ZERO, &nprow1);
    ncolsNuNu1D = SCALAPACK(numroc)(&Nu, &I_ONE, &mycol1, &I_ZERO, &npcol1);
    lldNuNu1D = std::max( nrowsNuNu1D, 1 );
  }    

  SCALAPACK(descinit)(desc_NuNu1D, &Nu, &Nu, &Nu, &I_ONE, &I_ZERO, 
      &I_ZERO, &contxt1, &lldNuNu1D, &info1);


  // 2D MPI
  for( Int i = IRound(sqrt(double(mpisize))); i <= mpisize; i++){
    nprow2 = i; npcol2 = mpisize / nprow2;
    if( (nprow2 >= npcol2) && (nprow2 * npcol2 == mpisize) ) break;
  }

  Cblacs_get(0, 0, &contxt2);
  //Cblacs_gridinit(&contxt2, "C", nprow2, npcol2);

  IntNumVec pmap2(mpisize);
  for ( Int i = 0; i < mpisize; i++ ){
    pmap2[i] = i;
  }
  Cblacs_gridmap(&contxt2, &pmap2[0], nprow2, nprow2, npcol2);

  Int mb2 = BlockSizeScaLAPACK;
  Int nb2 = BlockSizeScaLAPACK;

  //desc_NgNe2D
  if(contxt2 >= 0){
    Cblacs_gridinfo(contxt2, &nprow2, &npcol2, &myrow2, &mycol2);
    nrowsNgNe2D = SCALAPACK(numroc)(&Ng, &mb2, &myrow2, &I_ZERO, &nprow2);
    ncolsNgNe2D = SCALAPACK(numroc)(&Ne, &nb2, &mycol2, &I_ZERO, &npcol2);
    lldNgNe2D = std::max( nrowsNgNe2D, 1 );
  }

  SCALAPACK(descinit)(desc_NgNe2D, &Ng, &Ne, &mb2, &nb2, &I_ZERO, 
      &I_ZERO, &contxt2, &lldNgNe2D, &info2);

  //desc_NgNu2D
  if(contxt2 >= 0){
    Cblacs_gridinfo(contxt2, &nprow2, &npcol2, &myrow2, &mycol2);
    nrowsNgNu2D = SCALAPACK(numroc)(&Ng, &mb2, &myrow2, &I_ZERO, &nprow2);
    ncolsNgNu2D = SCALAPACK(numroc)(&Nu, &nb2, &mycol2, &I_ZERO, &npcol2);
    lldNgNu2D = std::max( nrowsNgNu2D, 1 );
  }

  SCALAPACK(descinit)(desc_NgNu2D, &Ng, &Nu, &mb2, &nb2, &I_ZERO, 
      &I_ZERO, &contxt2, &lldNgNu2D, &info2);

  //desc_NuNg2D
  if(contxt2 >= 0){
    Cblacs_gridinfo(contxt2, &nprow2, &npcol2, &myrow2, &mycol2);
    nrowsNuNg2D = SCALAPACK(numroc)(&Nu, &mb2, &myrow2, &I_ZERO, &nprow2);
    ncolsNuNg2D = SCALAPACK(numroc)(&Ng, &nb2, &mycol2, &I_ZERO, &npcol2);
    lldNuNg2D = std::max( nrowsNuNg2D, 1 );
  }

  SCALAPACK(descinit)(desc_NuNg2D, &Nu, &Ng, &mb2, &nb2, &I_ZERO, 
      &I_ZERO, &contxt2, &lldNuNg2D, &info2);

  //desc_NeNe2D
  if(contxt2 >= 0){
    Cblacs_gridinfo(contxt2, &nprow2, &npcol2, &myrow2, &mycol2);
    nrowsNeNe2D = SCALAPACK(numroc)(&Ne, &mb2, &myrow2, &I_ZERO, &nprow2);
    ncolsNeNe2D = SCALAPACK(numroc)(&Ne, &nb2, &mycol2, &I_ZERO, &npcol2);
    lldNeNe2D = std::max( nrowsNeNe2D, 1 );
  }

  SCALAPACK(descinit)(desc_NeNe2D, &Ne, &Ne, &mb2, &nb2, &I_ZERO, 
      &I_ZERO, &contxt2, &lldNeNe2D, &info2);

  //desc_NuNu2D
  if(contxt2 >= 0){
    Cblacs_gridinfo(contxt2, &nprow2, &npcol2, &myrow2, &mycol2);
    nrowsNuNu2D = SCALAPACK(numroc)(&Nu, &mb2, &myrow2, &I_ZERO, &nprow2);
    ncolsNuNu2D = SCALAPACK(numroc)(&Nu, &nb2, &mycol2, &I_ZERO, &npcol2);
    lldNuNu2D = std::max( nrowsNuNu2D, 1 );
  }

  SCALAPACK(descinit)(desc_NuNu2D, &Nu, &Nu, &mb2, &nb2, &I_ZERO, 
      &I_ZERO, &contxt2, &lldNuNu2D, &info2);

  //desc_NeNu2D
  if(contxt2 >= 0){
    Cblacs_gridinfo(contxt2, &nprow2, &npcol2, &myrow2, &mycol2);
    nrowsNeNu2D = SCALAPACK(numroc)(&Ne, &mb2, &myrow2, &I_ZERO, &nprow2);
    ncolsNeNu2D = SCALAPACK(numroc)(&Nu, &nb2, &mycol2, &I_ZERO, &npcol2);
    lldNeNu2D = std::max( nrowsNeNu2D, 1 );
  }

  SCALAPACK(descinit)(desc_NeNu2D, &Ne, &Nu, &mb2, &nb2, &I_ZERO, 
      &I_ZERO, &contxt2, &lldNeNu2D, &info2);

  //desc_NuNe2D
  if(contxt2 >= 0){
    Cblacs_gridinfo(contxt2, &nprow2, &npcol2, &myrow2, &mycol2);
    nrowsNuNe2D = SCALAPACK(numroc)(&Nu, &mb2, &myrow2, &I_ZERO, &nprow2);
    ncolsNuNe2D = SCALAPACK(numroc)(&Ne, &nb2, &mycol2, &I_ZERO, &npcol2);
    lldNuNe2D = std::max( nrowsNuNe2D, 1 );
  }

  SCALAPACK(descinit)(desc_NuNe2D, &Nu, &Ne, &mb2, &nb2, &I_ZERO, 
      &I_ZERO, &contxt2, &lldNuNe2D, &info2);

  DblNumMat phiCol( ntot, numStateLocal );
  SetValue( phiCol, 0.0 );

  DblNumMat psiCol( ntot, numStateLocal );
  SetValue( psiCol, 0.0 );

  lapack::Lacpy( 'A', ntot, numStateLocal, phi.Data(), ntot, phiCol.Data(), ntot );
  lapack::Lacpy( 'A', ntot, numStateLocal, wavefun_.Data(), ntot, psiCol.Data(), ntot );

  // Computing the indices is optional


  Int ntotLocal = nrowsNgNe1DRow; 
  //Int ntotLocalMG = nrowsNgNe1DRow;
  //Int ntotMG = ntot;
  Int ntotLocalMG, ntotMG;

  //if( (ntot % mpisize) == 0 ){
  //  ntotLocalMG = ntotBlocksize;
  //}
  //else{
  //  ntotLocalMG = ntotBlocksize + 1;
  // }

  if( isFixColumnDF == false ){

    GetTime( timeSta );

    DblNumMat localphiGRow( ntotLocal, numPre );
    SetValue( localphiGRow, 0.0 );

    DblNumMat localpsiGRow( ntotLocal, numPre );
    SetValue( localpsiGRow, 0.0 );

    DblNumMat G(numStateTotal, numPre);
    SetValue( G, 0.0 );

    DblNumMat phiRow( ntotLocal, numStateTotal );
    SetValue( phiRow, 0.0 );

    DblNumMat psiRow( ntotLocal, numStateTotal );
    SetValue( psiRow, 0.0 );

    SCALAPACK(pdgemr2d)(&Ng, &Ne, psiCol.Data(), &I_ONE, &I_ONE, desc_NgNe1DCol, 
        psiRow.Data(), &I_ONE, &I_ONE, desc_NgNe1DRow, &contxt1 );

    SCALAPACK(pdgemr2d)(&Ng, &Ne, phiCol.Data(), &I_ONE, &I_ONE, desc_NgNe1DCol, 
        phiRow.Data(), &I_ONE, &I_ONE, desc_NgNe1DRow, &contxt1 );

    // Step 1: Pre-compression of the wavefunctions. This uses
    // multiplication with orthonormalized random Gaussian matrices
    if ( mpirank == 0) {
      GaussianRandom(G);
      lapack::Orth( numStateTotal, numPre, G.Data(), numStateTotal );
    }

    GetTime( timeSta1 );

    MPI_Bcast(G.Data(), numStateTotal * numPre, MPI_DOUBLE, 0, domain_.comm);

    GetTime( timeEnd1 );
#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for Gaussian MPI_Bcast is " <<
      timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
#endif

    GetTime( timeSta1 );

    blas::Gemm( 'N', 'N', ntotLocal, numPre, numStateTotal, 1.0, 
        phiRow.Data(), ntotLocal, G.Data(), numStateTotal, 0.0,
        localphiGRow.Data(), ntotLocal );

    blas::Gemm( 'N', 'N', ntotLocal, numPre, numStateTotal, 1.0, 
        psiRow.Data(), ntotLocal, G.Data(), numStateTotal, 0.0,
        localpsiGRow.Data(), ntotLocal );

    GetTime( timeEnd1 );
#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for localphiGRow and localpsiGRow Gemm is " <<
      timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
#endif

    // Step 2: Pivoted QR decomposition  for the Hadamard product of
    // the compressed matrix. Transpose format for QRCP

    // NOTE: All processors should have the same ntotLocalMG


    int m_MGTemp = numPre*numPre;
    int n_MGTemp = ntot;

    DblNumMat MGCol( m_MGTemp, ntotLocal );

    DblNumVec MGNorm(ntot);
    for( Int k = 0; k < ntot; k++ ){
      MGNorm(k) = 0;
    }
    DblNumVec MGNormLocal(ntotLocal);
    for( Int k = 0; k < ntotLocal; k++ ){
      MGNormLocal(k) = 0;
    }

    GetTime( timeSta1 );

    for( Int j = 0; j < numPre; j++ ){
      for( Int i = 0; i < numPre; i++ ){
        for( Int ir = 0; ir < ntotLocal; ir++ ){
          MGCol(i+j*numPre,ir) = localphiGRow(ir,i) * localpsiGRow(ir,j);
          MGNormLocal(ir) += MGCol(i+j*numPre,ir) * MGCol(i+j*numPre,ir);   
        }
      }
    }

    GetTime( timeEnd1 );
#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for computing MG from localphiGRow and localpsiGRow is " <<
      timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
#endif

    //int ntotMGTemp, ntotLocalMGTemp;

    IntNumVec MGIdx(ntot);

    {
      Int ncols0D, nrows0D, lld0D; 
      Int ncols1D, nrows1D, lld1D; 

      Int desc_0D[9];
      Int desc_1D[9];

      Cblacs_get(0, 0, &contxt0);
      Cblacs_gridinit(&contxt0, "C", nprow0, npcol0);
      Cblacs_gridinfo(contxt0, &nprow0, &npcol0, &myrow0, &mycol0);

      SCALAPACK(descinit)(desc_0D, &Ng, &I_ONE, &Ng, &I_ONE, &I_ZERO, &I_ZERO, &contxt0, &Ng, &info0);

      if(contxt11 >= 0){
        nrows1D = SCALAPACK(numroc)(&Ng, &BlockSizeScaLAPACKTemp, &myrow11, &I_ZERO, &nprow11);
        ncols1D = SCALAPACK(numroc)(&I_ONE, &I_ONE, &mycol11, &I_ZERO, &npcol11);
        lld1D = std::max( nrows1D, 1 );
      }    

      SCALAPACK(descinit)(desc_1D, &Ng, &I_ONE, &BlockSizeScaLAPACKTemp, &I_ONE, &I_ZERO, 
          &I_ZERO, &contxt11, &lld1D, &info11);


      SCALAPACK(pdgemr2d)(&Ng, &I_ONE, MGNormLocal.Data(), &I_ONE, &I_ONE, desc_1D, 
          MGNorm.Data(), &I_ONE, &I_ONE, desc_0D, &contxt11 );

      MPI_Bcast( MGNorm.Data(), ntot, MPI_DOUBLE, 0, domain_.comm );


      double MGNormMax = *(std::max_element( MGNorm.Data(), MGNorm.Data() + ntot ) );
      double MGNormMin = *(std::min_element( MGNorm.Data(), MGNorm.Data() + ntot ) );

      ntotMG = 0;
      SetValue( MGIdx, 0 );

      for( Int k = 0; k < ntot; k++ ){
        if(MGNorm(k) > hybridDFTolerance){
          MGIdx(ntotMG) = k; 
          ntotMG = ntotMG + 1;
        }
      }

#if ( _DEBUGlevel_ >= 0 )
      statusOFS << "The col size for MG: " << " ntotMG = " << ntotMG << " ntotMG/ntot = " << Real(ntotMG)/Real(ntot) << std::endl << std::endl;
      statusOFS << "The norm range for MG: " <<  " MGNormMax = " << MGNormMax << " MGNormMin = " << MGNormMin << std::endl << std::endl;
#endif

    }

    Int ncols1DCol, nrows1DCol, lld1DCol; 
    Int ncols1DRow, nrows1DRow, lld1DRow; 

    Int desc_1DCol[9];
    Int desc_1DRow[9];

    if(contxt1 >= 0){
      nrows1DCol = SCALAPACK(numroc)(&m_MGTemp, &m_MGTemp, &myrow1, &I_ZERO, &nprow1);
      ncols1DCol = SCALAPACK(numroc)(&n_MGTemp, &BlockSizeScaLAPACKTemp, &mycol1, &I_ZERO, &npcol1);
      lld1DCol = std::max( nrows1DCol, 1 );
    }    

    SCALAPACK(descinit)(desc_1DCol, &m_MGTemp, &n_MGTemp, &m_MGTemp, &BlockSizeScaLAPACKTemp, &I_ZERO, 
        &I_ZERO, &contxt1, &lld1DCol, &info1);

    if(contxt11 >= 0){
      nrows1DRow = SCALAPACK(numroc)(&m_MGTemp, &I_ONE, &myrow11, &I_ZERO, &nprow11);
      ncols1DRow = SCALAPACK(numroc)(&n_MGTemp, &n_MGTemp, &mycol11, &I_ZERO, &npcol11);
      lld1DRow = std::max( nrows1DRow, 1 );
    }    

    SCALAPACK(descinit)(desc_1DRow, &m_MGTemp, &n_MGTemp, &I_ONE, &n_MGTemp, &I_ZERO, 
        &I_ZERO, &contxt11, &lld1DRow, &info11);

    DblNumMat MGRow( nrows1DRow, ntot );
    SetValue( MGRow, 0.0 );


    GetTime( timeSta1 );

    SCALAPACK(pdgemr2d)(&m_MGTemp, &n_MGTemp, MGCol.Data(), &I_ONE, &I_ONE, desc_1DCol, 
        MGRow.Data(), &I_ONE, &I_ONE, desc_1DRow, &contxt1 );

    GetTime( timeEnd1 );
#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for computing MG from Col and Row is " <<
      timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
#endif

    DblNumMat MG( nrows1DRow, ntotMG );

    for( Int i = 0; i < nrows1DRow; i++ ){
      for( Int j = 0; j < ntotMG; j++ ){
        MG(i,j) = MGRow(i,MGIdx(j));
      }
    }

    DblNumVec tau(ntotMG);
    pivQR_.Resize(ntot);
    SetValue( pivQR_, 0 ); // Important. Otherwise QRCP uses piv as initial guess
    // Q factor does not need to be used

    for( Int k = 0; k < ntotMG; k++ ){
      tau[k] = 0.0;
    }

    Real timeQRCPSta, timeQRCPEnd;
    GetTime( timeQRCPSta );

    if(0){  
      lapack::QRCP( numPre*numPre, ntotMG, MG.Data(), numPre*numPre, 
          pivQR_.Data(), tau.Data() );
    }//


    if(0){ // ScaLAPACL QRCP
      Int contxt;
      Int nprow, npcol, myrow, mycol, info;
      Cblacs_get(0, 0, &contxt);
      nprow = 1;
      npcol = mpisize;

      Cblacs_gridinit(&contxt, "C", nprow, npcol);
      Cblacs_gridinfo(contxt, &nprow, &npcol, &myrow, &mycol);
      Int desc_MG[9];

      Int irsrc = 0;
      Int icsrc = 0;

      Int mb_MG = numPre*numPre;
      Int nb_MG = ntotLocalMG;

      // FIXME The current routine does not actually allow ntotLocal to be different on different processors.
      // This must be fixed.
      SCALAPACK(descinit)(&desc_MG[0], &mb_MG, &ntotMG, &mb_MG, &nb_MG, &irsrc, 
          &icsrc, &contxt, &mb_MG, &info);

      IntNumVec pivQRTmp(ntotMG), pivQRLocal(ntotMG);
      if( mb_MG > ntot ){
        std::ostringstream msg;
        msg << "numPre*numPre > ntot. The number of grid points is perhaps too small!" << std::endl;
        ErrorHandling( msg.str().c_str() );
      }
      // DiagR is only for debugging purpose
      //        DblNumVec diagRLocal( mb_MG );
      //        DblNumVec diagR( mb_MG );

      SetValue( pivQRTmp, 0 );
      SetValue( pivQRLocal, 0 );
      SetValue( pivQR_, 0 );


      //        SetValue( diagRLocal, 0.0 );
      //        SetValue( diagR, 0.0 );

      if(0) {
        scalapack::QRCPF( mb_MG, ntotMG, MG.Data(), &desc_MG[0], 
            pivQRTmp.Data(), tau.Data() );
      }

      if(1) {
        scalapack::QRCPR( mb_MG, ntotMG, numMu_, MG.Data(), &desc_MG[0], 
            pivQRTmp.Data(), tau.Data(), 80, 40 );
      }


      // Combine the local pivQRTmp to global pivQR_
      for( Int j = 0; j < ntotLocalMG; j++ ){
        pivQRLocal[j + mpirank * ntotLocalMG] = pivQRTmp[j];
      }

      //        std::cout << "diag of MG = " << std::endl;
      //        if(mpirank == 0){
      //          std::cout << pivQRLocal << std::endl;
      //          for( Int j = 0; j < mb_MG; j++ ){
      //            std::cout << MG(j,j) << std::endl;
      //          }
      //        }
      MPI_Allreduce( pivQRLocal.Data(), pivQR_.Data(), 
          ntotMG, MPI_INT, MPI_SUM, domain_.comm );


      if(contxt >= 0) {
        Cblacs_gridexit( contxt );
      }

    } //ScaLAPACL QRCP


    if(1){ //ScaLAPACL QRCP 2D

      Int contxt1D, contxt2D;
      Int nprow1D, npcol1D, myrow1D, mycol1D, info1D;
      Int nprow2D, npcol2D, myrow2D, mycol2D, info2D;

      Int ncols1D, nrows1D, lld1D; 
      Int ncols2D, nrows2D, lld2D; 

      Int desc_MG1D[9];
      Int desc_MG2D[9];

      Int m_MG = numPre*numPre;
      Int n_MG = ntotMG;

      Int mb_MG1D = 1;
      Int nb_MG1D = ntotMG;

      nprow1D = mpisize;
      npcol1D = 1;

      Cblacs_get(0, 0, &contxt1D);
      Cblacs_gridinit(&contxt1D, "C", nprow1D, npcol1D);
      Cblacs_gridinfo(contxt1D, &nprow1D, &npcol1D, &myrow1D, &mycol1D);

      nrows1D = SCALAPACK(numroc)(&m_MG, &mb_MG1D, &myrow1D, &I_ZERO, &nprow1D);
      ncols1D = SCALAPACK(numroc)(&n_MG, &nb_MG1D, &mycol1D, &I_ZERO, &npcol1D);

      lld1D = std::max( nrows1D, 1 );

      SCALAPACK(descinit)(desc_MG1D, &m_MG, &n_MG, &mb_MG1D, &nb_MG1D, &I_ZERO, 
          &I_ZERO, &contxt1D, &lld1D, &info1D);

      for( Int i = std::min(mpisize, IRound(sqrt(double(mpisize*(n_MG/m_MG))))); 
          i <= mpisize; i++){
        npcol2D = i; nprow2D = mpisize / npcol2D;
        if( (npcol2D >= nprow2D) && (nprow2D * npcol2D == mpisize) ) break;
      }

      Cblacs_get(0, 0, &contxt2D);
      //Cblacs_gridinit(&contxt2D, "C", nprow2D, npcol2D);

      IntNumVec pmap(mpisize);
      for ( Int i = 0; i < mpisize; i++ ){
        pmap[i] = i;
      }
      Cblacs_gridmap(&contxt2D, &pmap[0], nprow2D, nprow2D, npcol2D);

      Int m_MG2DBlocksize = BlockSizeScaLAPACK;
      Int n_MG2DBlocksize = BlockSizeScaLAPACK;

      //Int m_MG2Local = m_MG/(m_MG2DBlocksize*nprow2D)*m_MG2DBlocksize;
      //Int n_MG2Local = n_MG/(n_MG2DBlocksize*npcol2D)*n_MG2DBlocksize;
      Int m_MG2Local, n_MG2Local;

      //MPI_Comm rowComm = MPI_COMM_NULL;
      MPI_Comm colComm = MPI_COMM_NULL;

      Int mpirankRow, mpisizeRow, mpirankCol, mpisizeCol;

      //MPI_Comm_split( domain_.comm, mpirank / nprow2D, mpirank, &rowComm );
      MPI_Comm_split( domain_.comm, mpirank % nprow2D, mpirank, &colComm );

      //MPI_Comm_rank(rowComm, &mpirankRow);
      //MPI_Comm_size(rowComm, &mpisizeRow);

      MPI_Comm_rank(colComm, &mpirankCol);
      MPI_Comm_size(colComm, &mpisizeCol);

      if(0){

        if((m_MG % (m_MG2DBlocksize * nprow2D))!= 0){ 
          if(mpirankRow < ((m_MG % (m_MG2DBlocksize*nprow2D)) / m_MG2DBlocksize)){
            m_MG2Local = m_MG2Local + m_MG2DBlocksize;
          }
          if(((m_MG % (m_MG2DBlocksize*nprow2D)) % m_MG2DBlocksize) != 0){
            if(mpirankRow == ((m_MG % (m_MG2DBlocksize*nprow2D)) / m_MG2DBlocksize)){
              m_MG2Local = m_MG2Local + ((m_MG % (m_MG2DBlocksize*nprow2D)) % m_MG2DBlocksize);
            }
          }
        }

        if((n_MG % (n_MG2DBlocksize * npcol2D))!= 0){ 
          if(mpirankCol < ((n_MG % (n_MG2DBlocksize*npcol2D)) / n_MG2DBlocksize)){
            n_MG2Local = n_MG2Local + n_MG2DBlocksize;
          }
          if(((n_MG % (n_MG2DBlocksize*nprow2D)) % n_MG2DBlocksize) != 0){
            if(mpirankCol == ((n_MG % (n_MG2DBlocksize*npcol2D)) / n_MG2DBlocksize)){
              n_MG2Local = n_MG2Local + ((n_MG % (n_MG2DBlocksize*nprow2D)) % n_MG2DBlocksize);
            }
          }
        }

      } // if(0)

      if(contxt2D >= 0){
        Cblacs_gridinfo(contxt2D, &nprow2D, &npcol2D, &myrow2D, &mycol2D);
        nrows2D = SCALAPACK(numroc)(&m_MG, &m_MG2DBlocksize, &myrow2D, &I_ZERO, &nprow2D);
        ncols2D = SCALAPACK(numroc)(&n_MG, &n_MG2DBlocksize, &mycol2D, &I_ZERO, &npcol2D);
        lld2D = std::max( nrows2D, 1 );
      }

      SCALAPACK(descinit)(desc_MG2D, &m_MG, &n_MG, &m_MG2DBlocksize, &n_MG2DBlocksize, &I_ZERO, 
          &I_ZERO, &contxt2D, &lld2D, &info2D);

      m_MG2Local = nrows2D;
      n_MG2Local = ncols2D;

      IntNumVec pivQRTmp1(ntotMG), pivQRTmp2(ntotMG), pivQRLocal(ntotMG);
      if( m_MG > ntot ){
        std::ostringstream msg;
        msg << "numPre*numPre > ntot. The number of grid points is perhaps too small!" << std::endl;
        ErrorHandling( msg.str().c_str() );
      }
      // DiagR is only for debugging purpose
      //        DblNumVec diagRLocal( mb_MG );
      //        DblNumVec diagR( mb_MG );

      DblNumMat&  MG1D = MG;
      DblNumMat  MG2D (m_MG2Local, n_MG2Local);

      SCALAPACK(pdgemr2d)(&m_MG, &n_MG, MG1D.Data(), &I_ONE, &I_ONE, desc_MG1D, 
          MG2D.Data(), &I_ONE, &I_ONE, desc_MG2D, &contxt1D );

      if(contxt2D >= 0){

        Real timeQRCP1, timeQRCP2;
        GetTime( timeQRCP1 );

        SetValue( pivQRTmp1, 0 );
        scalapack::QRCPF( m_MG, n_MG, MG2D.Data(), desc_MG2D, pivQRTmp1.Data(), tau.Data() );

        //scalapack::QRCPR( m_MG, n_MG, numMu_, MG2D.Data(), desc_MG2D, pivQRTmp1.Data(), tau.Data(), BlockSizeScaLAPACK, 32);

        GetTime( timeQRCP2 );

#if ( _DEBUGlevel_ >= 0 )
        statusOFS << "Time only for QRCP is " << timeQRCP2 - timeQRCP1 << " [s]" << std::endl << std::endl;
#endif

      }

      // Redistribute back eigenvectors
      //SetValue(MG1D, 0.0 );

      //SCALAPACK(pdgemr2d)( &m_MG, &n_MG, MG2D.Data(), &I_ONE, &I_ONE, desc_MG2D,
      //    MG1D.Data(), &I_ONE, &I_ONE, desc_MG1D, &contxt1D );

      // Combine the local pivQRTmp to global pivQR_
      SetValue( pivQRLocal, 0 );
      for( Int j = 0; j < n_MG2Local; j++ ){
        pivQRLocal[ (j / n_MG2DBlocksize) * n_MG2DBlocksize * npcol2D + mpirankCol * n_MG2DBlocksize + j % n_MG2DBlocksize] = pivQRTmp1[j];
      }

      SetValue( pivQRTmp2, 0 );
      MPI_Allreduce( pivQRLocal.Data(), pivQRTmp2.Data(), 
          ntotMG, MPI_INT, MPI_SUM, colComm );

      SetValue( pivQR_, 0 );
      for( Int j = 0; j < ntotMG; j++ ){
        pivQR_(j) = MGIdx(pivQRTmp2(j));
      }

      //if( rowComm != MPI_COMM_NULL ) MPI_Comm_free( & rowComm );
      if( colComm != MPI_COMM_NULL ) MPI_Comm_free( & colComm );

      if(contxt2D >= 0) {
        Cblacs_gridexit( contxt2D );
      }

      if(contxt1D >= 0) {
        Cblacs_gridexit( contxt1D );
      }

    } // if(1) ScaLAPACL QRCP

    GetTime( timeQRCPEnd );

    //statusOFS << std::endl<< "All processors exit with abort in spinor.cpp." << std::endl;

#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for QRCP alone is " <<
      timeQRCPEnd - timeQRCPSta << " [s]" << std::endl << std::endl;
#endif

    if(0){
      Real tolR = std::abs(MG(numMu_-1,numMu_-1)/MG(0,0));
      statusOFS << "numMu_ = " << numMu_ << std::endl;
      statusOFS << "|R(numMu-1,numMu-1)/R(0,0)| = " << tolR << std::endl;
    }

    GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for density fitting with QRCP is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

    // Dump out pivQR_
    if(0){
      std::ostringstream muStream;
      serialize( pivQR_, muStream, NO_MASK );
      SharedWrite( "pivQR", muStream );
    }
  }

  // Load pivQR_ file
  if(0){
    statusOFS << "Loading pivQR file.." << std::endl;
    std::istringstream muStream;
    SharedRead( "pivQR", muStream );
    deserialize( pivQR_, muStream, NO_MASK );
  }

  // *********************************************************************
  // Compute the interpolation matrix via the density matrix formulation
  // *********************************************************************

  GetTime( timeSta );

  // PhiMu is scaled by the occupation number to reflect the "true" density matrix
  //IntNumVec pivMu(numMu_);
  IntNumVec pivMu1(numMu_);

  for( Int mu = 0; mu < numMu_; mu++ ){
    pivMu1(mu) = pivQR_(mu);
  }


  //if(ntot % mpisize != 0){
  //  for( Int mu = 0; mu < numMu_; mu++ ){
  //    Int k1 = (pivMu1(mu) / ntotLocalMG) - (ntot % mpisize);
  //    if(k1 > 0){
  //      pivMu1(mu) = pivQR_(mu) - k1;
  //    }
  //  }
  //}

  GetTime( timeSta1 );

  DblNumMat psiMuCol(numStateLocal, numMu_);
  DblNumMat phiMuCol(numStateLocal, numMu_);
  SetValue( psiMuCol, 0.0 );
  SetValue( phiMuCol, 0.0 );

  for (Int k=0; k<numStateLocal; k++) {
    for (Int mu=0; mu<numMu_; mu++) {
      psiMuCol(k, mu) = psiCol(pivMu1(mu),k);
      phiMuCol(k, mu) = phiCol(pivMu1(mu),k) * occupationRate[(k * mpisize + mpirank)];
    }
  }

  DblNumMat psiMu2D(nrowsNeNu2D, ncolsNeNu2D);
  DblNumMat phiMu2D(nrowsNeNu2D, ncolsNeNu2D);
  SetValue( psiMu2D, 0.0 );
  SetValue( phiMu2D, 0.0 );

  SCALAPACK(pdgemr2d)(&Ne, &Nu, psiMuCol.Data(), &I_ONE, &I_ONE, desc_NeNu1D, 
      psiMu2D.Data(), &I_ONE, &I_ONE, desc_NeNu2D, &contxt11 );

  SCALAPACK(pdgemr2d)(&Ne, &Nu, phiMuCol.Data(), &I_ONE, &I_ONE, desc_NeNu1D, 
      phiMu2D.Data(), &I_ONE, &I_ONE, desc_NeNu2D, &contxt11 );

  GetTime( timeEnd1 );

#if ( _DEBUGlevel_ >= 0 )
  statusOFS << "Time for computing psiMuRow and phiMuRow is " <<
    timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
#endif

  GetTime( timeSta1 );

  DblNumMat psi2D(nrowsNgNe2D, ncolsNgNe2D);
  DblNumMat phi2D(nrowsNgNe2D, ncolsNgNe2D);
  SetValue( psi2D, 0.0 );
  SetValue( phi2D, 0.0 );

  SCALAPACK(pdgemr2d)(&Ng, &Ne, psiCol.Data(), &I_ONE, &I_ONE, desc_NgNe1DCol, 
      psi2D.Data(), &I_ONE, &I_ONE, desc_NgNe2D, &contxt1 );
  SCALAPACK(pdgemr2d)(&Ng, &Ne, phiCol.Data(), &I_ONE, &I_ONE, desc_NgNe1DCol, 
      phi2D.Data(), &I_ONE, &I_ONE, desc_NgNe2D, &contxt1 );

  DblNumMat PpsiMu2D(nrowsNgNu2D, ncolsNgNu2D);
  DblNumMat PphiMu2D(nrowsNgNu2D, ncolsNgNu2D);
  SetValue( PpsiMu2D, 0.0 );
  SetValue( PphiMu2D, 0.0 );

  SCALAPACK(pdgemm)("N", "N", &Ng, &Nu, &Ne, 
      &D_ONE,
      psi2D.Data(), &I_ONE, &I_ONE, desc_NgNe2D,
      psiMu2D.Data(), &I_ONE, &I_ONE, desc_NeNu2D, 
      &D_ZERO,
      PpsiMu2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D);

  SCALAPACK(pdgemm)("N", "N", &Ng, &Nu, &Ne, 
      &D_ONE,
      phi2D.Data(), &I_ONE, &I_ONE, desc_NgNe2D,
      phiMu2D.Data(), &I_ONE, &I_ONE, desc_NeNu2D, 
      &D_ZERO,
      PphiMu2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D);

  GetTime( timeEnd1 );

#if ( _DEBUGlevel_ >= 0 )
  statusOFS << "Time for PpsiMu and PphiMu GEMM is " <<
    timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
#endif

  GetTime( timeSta1 );

  DblNumMat Xi2D(nrowsNgNu2D, ncolsNgNu2D);
  SetValue( Xi2D, 0.0 );

  Real* Xi2DPtr = Xi2D.Data();
  Real* PpsiMu2DPtr = PpsiMu2D.Data();
  Real* PphiMu2DPtr = PphiMu2D.Data();

  for( Int g = 0; g < nrowsNgNu2D * ncolsNgNu2D; g++ ){
    Xi2DPtr[g] = PpsiMu2DPtr[g] * PphiMu2DPtr[g];
  }

  DblNumMat Xi1D(nrowsNgNu1D, ncolsNuNu1D);
  SetValue( Xi1D, 0.0 );

  SCALAPACK(pdgemr2d)(&Ng, &Nu, Xi2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D, 
      Xi1D.Data(), &I_ONE, &I_ONE, desc_NgNu1D, &contxt2 );

  DblNumMat PMuNu1D(nrowsNuNu1D, ncolsNuNu1D);
  SetValue( PMuNu1D, 0.0 );

  for (Int mu=0; mu<nrowsNuNu1D; mu++) {
    for (Int nu=0; nu<ncolsNuNu1D; nu++) {
      PMuNu1D(mu, nu) = Xi1D(pivMu1(mu),nu);
    }
  }

  DblNumMat PMuNu2D(nrowsNuNu2D, ncolsNuNu2D);
  SetValue( PMuNu2D, 0.0 );

  SCALAPACK(pdgemr2d)(&Nu, &Nu, PMuNu1D.Data(), &I_ONE, &I_ONE, desc_NuNu1D, 
      PMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D, &contxt1 );

  GetTime( timeEnd1 );

#if ( _DEBUGlevel_ >= 0 )
  statusOFS << "Time for computing PMuNu is " <<
    timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
#endif


  //Method 1
  if(0){

    GetTime( timeSta1 );

    SCALAPACK(pdpotrf)("L", &Nu, PMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D, &info2);

    GetTime( timeEnd1 );

#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for PMuNu Potrf is " <<
      timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
#endif

    GetTime( timeSta1 );

    SCALAPACK(pdtrsm)("R", "L", "T", "N", &Ng, &Nu, &D_ONE,
        PMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D,
        Xi2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D);

    SCALAPACK(pdtrsm)("R", "L", "N", "N", &Ng, &Nu, &D_ONE,
        PMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D,
        Xi2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D);

    GetTime( timeEnd1 );

#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for PMuNu and Xi pdtrsm is " <<
      timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
#endif

  }


  //Method 2
  if(1){

    GetTime( timeSta1 );

    SCALAPACK(pdpotrf)("L", &Nu, PMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D, &info2);
    SCALAPACK(pdpotri)("L", &Nu, PMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D, &info2);

    GetTime( timeEnd1 );

#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for PMuNu Potrf is " <<
      timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
#endif

    GetTime( timeSta1 );

    DblNumMat PMuNu2DTemp(nrowsNuNu2D, ncolsNuNu2D);
    SetValue( PMuNu2DTemp, 0.0 );

    lapack::Lacpy( 'A', nrowsNuNu2D, ncolsNuNu2D, PMuNu2D.Data(), nrowsNuNu2D, PMuNu2DTemp.Data(), nrowsNuNu2D );

    SCALAPACK(pdtradd)("U", "T", &Nu, &Nu, 
        &D_ONE,
        PMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D, 
        &D_ZERO,
        PMuNu2DTemp.Data(), &I_ONE, &I_ONE, desc_NuNu2D);

    DblNumMat Xi2DTemp(nrowsNgNu2D, ncolsNgNu2D);
    SetValue( Xi2DTemp, 0.0 );

    SCALAPACK(pdgemm)("N", "N", &Ng, &Nu, &Nu, 
        &D_ONE,
        Xi2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D,
        PMuNu2DTemp.Data(), &I_ONE, &I_ONE, desc_NuNu2D, 
        &D_ZERO,
        Xi2DTemp.Data(), &I_ONE, &I_ONE, desc_NgNu2D);

    SetValue( Xi2D, 0.0 );
    lapack::Lacpy( 'A', nrowsNgNu2D, ncolsNgNu2D, Xi2DTemp.Data(), nrowsNgNu2D, Xi2D.Data(), nrowsNgNu2D );

    GetTime( timeEnd1 );

#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for PMuNu and Xi pdgemm is " <<
      timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
#endif

  }


  //Method 3
  if(0){

    GetTime( timeSta1 );

    SCALAPACK(pdpotrf)("L", &Nu, PMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D, &info2);
    SCALAPACK(pdpotri)("L", &Nu, PMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D, &info2);

    GetTime( timeEnd1 );

#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for PMuNu Potrf is " <<
      timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
#endif

    GetTime( timeSta1 );

    DblNumMat Xi2DTemp(nrowsNgNu2D, ncolsNgNu2D);
    SetValue( Xi2DTemp, 0.0 );

    SCALAPACK(pdsymm)("R", "L", &Ng, &Nu, 
        &D_ONE,
        PMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D,
        Xi2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D,
        &D_ZERO,
        Xi2DTemp.Data(), &I_ONE, &I_ONE, desc_NgNu2D);

    SetValue( Xi2D, 0.0 );
    lapack::Lacpy( 'A', nrowsNgNu2D, ncolsNgNu2D, Xi2DTemp.Data(), nrowsNgNu2D, Xi2D.Data(), nrowsNgNu2D );

    GetTime( timeEnd1 );

#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for PMuNu and Xi pdsymm is " <<
      timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
#endif

  }


  //Method 4
  if(0){

    GetTime( timeSta1 );

    DblNumMat Xi2DTemp(nrowsNuNg2D, ncolsNuNg2D);
    SetValue( Xi2DTemp, 0.0 );

    DblNumMat PMuNu2DTemp(ncolsNuNu2D, nrowsNuNu2D);
    SetValue( PMuNu2DTemp, 0.0 );

    SCALAPACK(pdgeadd)("T", &Nu, &Ng,
        &D_ONE,
        Xi2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D,
        &D_ZERO,
        Xi2DTemp.Data(), &I_ONE, &I_ONE, desc_NuNg2D);

    SCALAPACK(pdgeadd)("T", &Nu, &Nu,
        &D_ONE,
        PMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D,
        &D_ZERO,
        PMuNu2DTemp.Data(), &I_ONE, &I_ONE, desc_NuNu2D);

    Int lwork=-1, info;
    double dummyWork;

    SCALAPACK(pdgels)("N", &Nu, &Nu, &Ng, 
        PMuNu2DTemp.Data(), &I_ONE, &I_ONE, desc_NuNu2D,
        Xi2DTemp.Data(), &I_ONE, &I_ONE, desc_NuNg2D,
        &dummyWork, &lwork, &info);

    lwork = dummyWork;
    std::vector<double> work(lwork);

    SCALAPACK(pdgels)("N", &Nu, &Nu, &Ng, 
        PMuNu2DTemp.Data(), &I_ONE, &I_ONE, desc_NuNu2D,
        Xi2DTemp.Data(), &I_ONE, &I_ONE, desc_NuNg2D,
        &work[0], &lwork, &info);

    SetValue( Xi2D, 0.0 );
    SCALAPACK(pdgeadd)("T", &Ng, &Nu,
        &D_ONE,
        Xi2DTemp.Data(), &I_ONE, &I_ONE, desc_NuNg2D,
        &D_ZERO,
        Xi2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D);

    GetTime( timeEnd1 );

#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for PMuNu and Xi pdgels is " <<
      timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
#endif

  }


  GetTime( timeEnd );

#if ( _DEBUGlevel_ >= 0 )
  statusOFS << "Time for computing the interpolation vectors is " <<
    timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif



  // *********************************************************************
  // Solve the Poisson equations.
  // Store VXi separately. This is not the most memory efficient
  // implementation
  // *********************************************************************

  DblNumMat VXi2D(nrowsNgNu2D, ncolsNgNu2D);

  {
    GetTime( timeSta );
    // XiCol used for both input and output
    DblNumMat XiCol(ntot, numMuLocal);
    SetValue(XiCol, 0.0 );

    SCALAPACK(pdgemr2d)(&Ng, &Nu, Xi2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D, 
        XiCol.Data(), &I_ONE, &I_ONE, desc_NgNu1D, &contxt2 );

    GetTime( timeSta );
    for( Int mu = 0; mu < numMuLocal; mu++ ){
      blas::Copy( ntot,  XiCol.VecData(mu), 1, fft.inputVecR2C.Data(), 1 );

      FFTWExecute ( fft, fft.forwardPlanR2C );

      for( Int ig = 0; ig < ntotR2C; ig++ ){
        fft.outputVecR2C(ig) *= -exxFraction * exxgkkR2C(ig);
      }

      FFTWExecute ( fft, fft.backwardPlanR2C );

      blas::Copy( ntot, fft.inputVecR2C.Data(), 1, XiCol.VecData(mu), 1 );

    } // for (mu)

    SetValue(VXi2D, 0.0 );
    SCALAPACK(pdgemr2d)(&Ng, &Nu, XiCol.Data(), &I_ONE, &I_ONE, desc_NgNu1D, 
        VXi2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D, &contxt1 );

    GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for solving Poisson-like equations is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
  }

  // Prepare for the computation of the M matrix
  GetTime( timeSta );

  DblNumMat MMatMuNu2D(nrowsNuNu2D, ncolsNuNu2D);
  SetValue(MMatMuNu2D, 0.0 );

  SCALAPACK(pdgemm)("T", "N", &Nu, &Nu, &Ng, 
      &D_MinusONE,
      Xi2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D,
      VXi2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D, 
      &D_ZERO,
      MMatMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D);

  DblNumMat phiMuNu2D(nrowsNuNu2D, ncolsNuNu2D);
  SCALAPACK(pdgemm)("T", "N", &Nu, &Nu, &Ne, 
      &D_ONE,
      phiMu2D.Data(), &I_ONE, &I_ONE, desc_NeNu2D,
      phiMu2D.Data(), &I_ONE, &I_ONE, desc_NeNu2D, 
      &D_ZERO,
      phiMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D);

  Real* MMat2DPtr = MMatMuNu2D.Data();
  Real* phi2DPtr  = phiMuNu2D.Data();

  for( Int g = 0; g < nrowsNuNu2D * ncolsNuNu2D; g++ ){
    MMat2DPtr[g] *= phi2DPtr[g];
  }

  GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
  statusOFS << "Time for preparing the M matrix is " <<
    timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

  // *********************************************************************
  // Compute the exchange potential and the symmetrized inner product
  // *********************************************************************

  GetTime( timeSta );

  // Rewrite VXi by VXi.*PcolPhi
  Real* VXi2DPtr = VXi2D.Data();
  Real* PphiMu2DPtr1 = PphiMu2D.Data();
  for( Int g = 0; g < nrowsNgNu2D * ncolsNgNu2D; g++ ){
    VXi2DPtr[g] *= PphiMu2DPtr1[g];
  }

  // NOTE: a3 must be zero in order to compute the M matrix later
  DblNumMat a32D( nrowsNgNe2D, ncolsNgNe2D );
  SetValue( a32D, 0.0 );

  SCALAPACK(pdgemm)("N", "T", &Ng, &Ne, &Nu, 
      &D_ONE,
      VXi2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D,
      psiMu2D.Data(), &I_ONE, &I_ONE, desc_NeNu2D, 
      &D_ZERO,
      a32D.Data(), &I_ONE, &I_ONE, desc_NgNe2D);

  DblNumMat a3Col( ntot, numStateLocal );
  SetValue(a3Col, 0.0 );

  SCALAPACK(pdgemr2d)(&Ng, &Ne, a32D.Data(), &I_ONE, &I_ONE, desc_NgNe2D, 
      a3Col.Data(), &I_ONE, &I_ONE, desc_NgNe1DCol, &contxt2 );

  lapack::Lacpy( 'A', ntot, numStateLocal, a3Col.Data(), ntot, a3.Data(), ntot );

  GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
  statusOFS << "Time for computing the exchange potential is " <<
    timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

  // Compute the matrix VxMat = -Psi'* vexxPsi and symmetrize
  // vexxPsi (a3) must be zero before entering this routine
  VxMat.Resize( numStateTotal, numStateTotal );

  GetTime( timeSta );

  if(1){

    DblNumMat VxMat2D( nrowsNeNe2D, ncolsNeNe2D );
    DblNumMat VxMatTemp2D( nrowsNuNe2D, ncolsNuNe2D );

    SCALAPACK(pdgemm)("N", "T", &Nu, &Ne, &Nu, 
        &D_ONE,
        MMatMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D,
        psiMu2D.Data(), &I_ONE, &I_ONE, desc_NeNu2D, 
        &D_ZERO,
        VxMatTemp2D.Data(), &I_ONE, &I_ONE, desc_NuNe2D);

    SCALAPACK(pdgemm)("N", "N", &Ne, &Ne, &Nu, 
        &D_ONE,
        psiMu2D.Data(), &I_ONE, &I_ONE, desc_NeNu2D, 
        VxMatTemp2D.Data(), &I_ONE, &I_ONE, desc_NuNe2D, 
        &D_ZERO,
        VxMat2D.Data(), &I_ONE, &I_ONE, desc_NeNe2D);

    SCALAPACK(pdgemr2d)(&Ne, &Ne, VxMat2D.Data(), &I_ONE, &I_ONE, desc_NeNe2D, 
        VxMat.Data(), &I_ONE, &I_ONE, desc_NeNe0D, &contxt2 );

    //if(mpirank == 0){
    //  MPI_Bcast( VxMat.Data(), Ne * Ne, MPI_DOUBLE, 0, domain_.comm );
    //}

  }
  GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
  statusOFS << "Time for computing VxMat in the sym format is " <<
    timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

  if(contxt0 >= 0) {
    Cblacs_gridexit( contxt0 );
  }

  if(contxt1 >= 0) {
    Cblacs_gridexit( contxt1 );
  }

  if(contxt11 >= 0) {
    Cblacs_gridexit( contxt11 );
  }

  if(contxt2 >= 0) {
    Cblacs_gridexit( contxt2 );
  }

  MPI_Barrier(domain_.comm);

  return ;
}        // -----  end of method Spinor::AddMultSpinorEXXDF6  ----- 


// Kmeans for ISDF
// Update: 8/20/2017
void Spinor::AddMultSpinorEXXDF7 ( Fourier& fft, 
    const NumTns<Real>& phi,
    const DblNumVec& exxgkkR2C,
    Real  exxFraction,
    Real  numSpin,
    const DblNumVec& occupationRate,
    std::string hybridDFType,
    Real  hybridDFKmeansTolerance, 
    Int   hybridDFKmeansMaxIter, 
    const Real numMuFac,
    const Real numGaussianRandomFac,
    const Int numProcScaLAPACK,  
    const Real hybridDFTolerance,
    const Int BlockSizeScaLAPACK,  
    NumTns<Real>& a3, 
    NumMat<Real>& VxMat,
    bool isFixColumnDF )
{
  Real timeSta, timeEnd;
  Real timeSta1, timeEnd1;

  if( !fft.isInitialized ){
    ErrorHandling("Fourier is not prepared.");
  }

  MPI_Barrier(domain_.comm);
  int mpirank;  MPI_Comm_rank(domain_.comm, &mpirank);
  int mpisize;  MPI_Comm_size(domain_.comm, &mpisize);

  Index3& numGrid = domain_.numGrid;
  Index3& numGridFine = domain_.numGridFine;

  Int ntot     = domain_.NumGridTotal();
  Int ntotFine = domain_.NumGridTotalFine();
  Int ntotR2C = fft.numGridTotalR2C;
  Int ntotR2CFine = fft.numGridTotalR2CFine;
  Int ncom = wavefun_.n();
  Int numStateLocal = wavefun_.p();
  Int numStateTotal = numStateTotal_;

  Int ncomPhi = phi.n();

  Real vol = domain_.Volume();

  if( ncomPhi != 1 || ncom != 1 ){
    ErrorHandling("Spin polarized case not implemented.");
  }

  if( fft.domain.NumGridTotal() != ntot ){
    ErrorHandling("Domain size does not match.");
  }


  // *********************************************************************
  // Perform interpolative separable density fitting
  // *********************************************************************

  //numMu_ = std::min(IRound(numStateTotal*numMuFac), ntot);
  numMu_ = IRound(numStateTotal*numMuFac);

  Int numPre = IRound(std::sqrt(numMu_*numGaussianRandomFac));
  if( numPre > numStateTotal ){
    ErrorHandling("numMu is too large for interpolative separable density fitting!");
  }

  statusOFS << "ntot          = " << ntot << std::endl;
  statusOFS << "numMu         = " << numMu_ << std::endl;
  statusOFS << "numPre        = " << numPre << std::endl;
  statusOFS << "numPre*numPre = " << numPre * numPre << std::endl;

  // Convert the column partition to row partition
  Int numStateBlocksize = numStateTotal / mpisize;
  //Int ntotBlocksize = ntot / mpisize;

  Int numMuBlocksize = numMu_ / mpisize;

  Int numStateLocal1 = numStateBlocksize;
  //Int ntotLocal = ntotBlocksize;

  Int numMuLocal = numMuBlocksize;

  if(mpirank < (numStateTotal % mpisize)){
    numStateLocal1 = numStateBlocksize + 1;
  }

  if(numStateLocal !=  numStateLocal1){
    statusOFS << "numStateLocal = " << numStateLocal << " numStateLocal1 = " << numStateLocal1 << std::endl;
    ErrorHandling("The size is not right in interpolative separable density fitting!");
  }

  if(mpirank < (numMu_ % mpisize)){
    numMuLocal = numMuBlocksize + 1;
  }

  //if(mpirank < (ntot % mpisize)){
  //  ntotLocal = ntotBlocksize + 1;
  //}

  //huwei 
  //2D MPI commucation for all the matrix

  Int I_ONE = 1, I_ZERO = 0;
  double D_ONE = 1.0;
  double D_ZERO = 0.0;
  double D_MinusONE = -1.0;

  Int contxt0, contxt1, contxt11, contxt2;
  Int nprow0, npcol0, myrow0, mycol0, info0;
  Int nprow1, npcol1, myrow1, mycol1, info1;
  Int nprow11, npcol11, myrow11, mycol11, info11;
  Int nprow2, npcol2, myrow2, mycol2, info2;

  Int ncolsNgNe1DCol, nrowsNgNe1DCol, lldNgNe1DCol; 
  Int ncolsNgNe1DRow, nrowsNgNe1DRow, lldNgNe1DRow; 
  Int ncolsNgNe2D, nrowsNgNe2D, lldNgNe2D; 
  Int ncolsNgNu1D, nrowsNgNu1D, lldNgNu1D; 
  Int ncolsNgNu2D, nrowsNgNu2D, lldNgNu2D; 
  Int ncolsNuNg2D, nrowsNuNg2D, lldNuNg2D; 
  Int ncolsNeNe0D, nrowsNeNe0D, lldNeNe0D; 
  Int ncolsNeNe2D, nrowsNeNe2D, lldNeNe2D; 
  Int ncolsNuNu1D, nrowsNuNu1D, lldNuNu1D; 
  Int ncolsNuNu2D, nrowsNuNu2D, lldNuNu2D; 
  Int ncolsNeNu1D, nrowsNeNu1D, lldNeNu1D; 
  Int ncolsNeNu2D, nrowsNeNu2D, lldNeNu2D; 
  Int ncolsNuNe2D, nrowsNuNe2D, lldNuNe2D; 

  Int desc_NgNe1DCol[9];
  Int desc_NgNe1DRow[9];
  Int desc_NgNe2D[9];
  Int desc_NgNu1D[9];
  Int desc_NgNu2D[9];
  Int desc_NuNg2D[9];
  Int desc_NeNe0D[9];
  Int desc_NeNe2D[9];
  Int desc_NuNu1D[9];
  Int desc_NuNu2D[9];
  Int desc_NeNu1D[9];
  Int desc_NeNu2D[9];
  Int desc_NuNe2D[9];

  Int Ng = ntot;
  Int Ne = numStateTotal_; 
  Int Nu = numMu_; 

  // 0D MPI
  nprow0 = 1;
  npcol0 = mpisize;

  Cblacs_get(0, 0, &contxt0);
  Cblacs_gridinit(&contxt0, "C", nprow0, npcol0);
  Cblacs_gridinfo(contxt0, &nprow0, &npcol0, &myrow0, &mycol0);

  SCALAPACK(descinit)(desc_NeNe0D, &Ne, &Ne, &Ne, &Ne, &I_ZERO, &I_ZERO, &contxt0, &Ne, &info0);

  // 1D MPI
  nprow1 = 1;
  npcol1 = mpisize;

  Cblacs_get(0, 0, &contxt1);
  Cblacs_gridinit(&contxt1, "C", nprow1, npcol1);
  Cblacs_gridinfo(contxt1, &nprow1, &npcol1, &myrow1, &mycol1);

  //desc_NgNe1DCol
  if(contxt1 >= 0){
    nrowsNgNe1DCol = SCALAPACK(numroc)(&Ng, &Ng, &myrow1, &I_ZERO, &nprow1);
    ncolsNgNe1DCol = SCALAPACK(numroc)(&Ne, &I_ONE, &mycol1, &I_ZERO, &npcol1);
    lldNgNe1DCol = std::max( nrowsNgNe1DCol, 1 );
  }    

  SCALAPACK(descinit)(desc_NgNe1DCol, &Ng, &Ne, &Ng, &I_ONE, &I_ZERO, 
      &I_ZERO, &contxt1, &lldNgNe1DCol, &info1);

  nprow11 = mpisize;
  npcol11 = 1;

  Cblacs_get(0, 0, &contxt11);
  Cblacs_gridinit(&contxt11, "C", nprow11, npcol11);
  Cblacs_gridinfo(contxt11, &nprow11, &npcol11, &myrow11, &mycol11);

  Int BlockSizeScaLAPACKTemp = BlockSizeScaLAPACK; 
  //desc_NgNe1DRow
  if(contxt11 >= 0){
    nrowsNgNe1DRow = SCALAPACK(numroc)(&Ng, &BlockSizeScaLAPACKTemp, &myrow11, &I_ZERO, &nprow11);
    ncolsNgNe1DRow = SCALAPACK(numroc)(&Ne, &Ne, &mycol11, &I_ZERO, &npcol11);
    lldNgNe1DRow = std::max( nrowsNgNe1DRow, 1 );
  }    

  SCALAPACK(descinit)(desc_NgNe1DRow, &Ng, &Ne, &BlockSizeScaLAPACKTemp, &Ne, &I_ZERO, 
      &I_ZERO, &contxt11, &lldNgNe1DRow, &info11);

  //desc_NeNu1D
  if(contxt11 >= 0){
    nrowsNeNu1D = SCALAPACK(numroc)(&Ne, &I_ONE, &myrow11, &I_ZERO, &nprow11);
    ncolsNeNu1D = SCALAPACK(numroc)(&Nu, &Nu, &mycol11, &I_ZERO, &npcol11);
    lldNeNu1D = std::max( nrowsNeNu1D, 1 );
  }    

  SCALAPACK(descinit)(desc_NeNu1D, &Ne, &Nu, &I_ONE, &Nu, &I_ZERO, 
      &I_ZERO, &contxt11, &lldNeNu1D, &info11);

  //desc_NgNu1D
  if(contxt1 >= 0){
    nrowsNgNu1D = SCALAPACK(numroc)(&Ng, &Ng, &myrow1, &I_ZERO, &nprow1);
    ncolsNgNu1D = SCALAPACK(numroc)(&Nu, &I_ONE, &mycol1, &I_ZERO, &npcol1);
    lldNgNu1D = std::max( nrowsNgNu1D, 1 );
  }    

  SCALAPACK(descinit)(desc_NgNu1D, &Ng, &Nu, &Ng, &I_ONE, &I_ZERO, 
      &I_ZERO, &contxt1, &lldNgNu1D, &info1);

  //desc_NuNu1D
  if(contxt1 >= 0){
    nrowsNuNu1D = SCALAPACK(numroc)(&Nu, &Nu, &myrow1, &I_ZERO, &nprow1);
    ncolsNuNu1D = SCALAPACK(numroc)(&Nu, &I_ONE, &mycol1, &I_ZERO, &npcol1);
    lldNuNu1D = std::max( nrowsNuNu1D, 1 );
  }    

  SCALAPACK(descinit)(desc_NuNu1D, &Nu, &Nu, &Nu, &I_ONE, &I_ZERO, 
      &I_ZERO, &contxt1, &lldNuNu1D, &info1);


  // 2D MPI
  for( Int i = IRound(sqrt(double(mpisize))); i <= mpisize; i++){
    nprow2 = i; npcol2 = mpisize / nprow2;
    if( (nprow2 >= npcol2) && (nprow2 * npcol2 == mpisize) ) break;
  }

  Cblacs_get(0, 0, &contxt2);
  //Cblacs_gridinit(&contxt2, "C", nprow2, npcol2);

  IntNumVec pmap2(mpisize);
  for ( Int i = 0; i < mpisize; i++ ){
    pmap2[i] = i;
  }
  Cblacs_gridmap(&contxt2, &pmap2[0], nprow2, nprow2, npcol2);

  Int mb2 = BlockSizeScaLAPACK;
  Int nb2 = BlockSizeScaLAPACK;

  //desc_NgNe2D
  if(contxt2 >= 0){
    Cblacs_gridinfo(contxt2, &nprow2, &npcol2, &myrow2, &mycol2);
    nrowsNgNe2D = SCALAPACK(numroc)(&Ng, &mb2, &myrow2, &I_ZERO, &nprow2);
    ncolsNgNe2D = SCALAPACK(numroc)(&Ne, &nb2, &mycol2, &I_ZERO, &npcol2);
    lldNgNe2D = std::max( nrowsNgNe2D, 1 );
  }

  SCALAPACK(descinit)(desc_NgNe2D, &Ng, &Ne, &mb2, &nb2, &I_ZERO, 
      &I_ZERO, &contxt2, &lldNgNe2D, &info2);

  //desc_NgNu2D
  if(contxt2 >= 0){
    Cblacs_gridinfo(contxt2, &nprow2, &npcol2, &myrow2, &mycol2);
    nrowsNgNu2D = SCALAPACK(numroc)(&Ng, &mb2, &myrow2, &I_ZERO, &nprow2);
    ncolsNgNu2D = SCALAPACK(numroc)(&Nu, &nb2, &mycol2, &I_ZERO, &npcol2);
    lldNgNu2D = std::max( nrowsNgNu2D, 1 );
  }

  SCALAPACK(descinit)(desc_NgNu2D, &Ng, &Nu, &mb2, &nb2, &I_ZERO, 
      &I_ZERO, &contxt2, &lldNgNu2D, &info2);

  //desc_NuNg2D
  if(contxt2 >= 0){
    Cblacs_gridinfo(contxt2, &nprow2, &npcol2, &myrow2, &mycol2);
    nrowsNuNg2D = SCALAPACK(numroc)(&Nu, &mb2, &myrow2, &I_ZERO, &nprow2);
    ncolsNuNg2D = SCALAPACK(numroc)(&Ng, &nb2, &mycol2, &I_ZERO, &npcol2);
    lldNuNg2D = std::max( nrowsNuNg2D, 1 );
  }

  SCALAPACK(descinit)(desc_NuNg2D, &Nu, &Ng, &mb2, &nb2, &I_ZERO, 
      &I_ZERO, &contxt2, &lldNuNg2D, &info2);

  //desc_NeNe2D
  if(contxt2 >= 0){
    Cblacs_gridinfo(contxt2, &nprow2, &npcol2, &myrow2, &mycol2);
    nrowsNeNe2D = SCALAPACK(numroc)(&Ne, &mb2, &myrow2, &I_ZERO, &nprow2);
    ncolsNeNe2D = SCALAPACK(numroc)(&Ne, &nb2, &mycol2, &I_ZERO, &npcol2);
    lldNeNe2D = std::max( nrowsNeNe2D, 1 );
  }

  SCALAPACK(descinit)(desc_NeNe2D, &Ne, &Ne, &mb2, &nb2, &I_ZERO, 
      &I_ZERO, &contxt2, &lldNeNe2D, &info2);

  //desc_NuNu2D
  if(contxt2 >= 0){
    Cblacs_gridinfo(contxt2, &nprow2, &npcol2, &myrow2, &mycol2);
    nrowsNuNu2D = SCALAPACK(numroc)(&Nu, &mb2, &myrow2, &I_ZERO, &nprow2);
    ncolsNuNu2D = SCALAPACK(numroc)(&Nu, &nb2, &mycol2, &I_ZERO, &npcol2);
    lldNuNu2D = std::max( nrowsNuNu2D, 1 );
  }

  SCALAPACK(descinit)(desc_NuNu2D, &Nu, &Nu, &mb2, &nb2, &I_ZERO, 
      &I_ZERO, &contxt2, &lldNuNu2D, &info2);

  //desc_NeNu2D
  if(contxt2 >= 0){
    Cblacs_gridinfo(contxt2, &nprow2, &npcol2, &myrow2, &mycol2);
    nrowsNeNu2D = SCALAPACK(numroc)(&Ne, &mb2, &myrow2, &I_ZERO, &nprow2);
    ncolsNeNu2D = SCALAPACK(numroc)(&Nu, &nb2, &mycol2, &I_ZERO, &npcol2);
    lldNeNu2D = std::max( nrowsNeNu2D, 1 );
  }

  SCALAPACK(descinit)(desc_NeNu2D, &Ne, &Nu, &mb2, &nb2, &I_ZERO, 
      &I_ZERO, &contxt2, &lldNeNu2D, &info2);

  //desc_NuNe2D
  if(contxt2 >= 0){
    Cblacs_gridinfo(contxt2, &nprow2, &npcol2, &myrow2, &mycol2);
    nrowsNuNe2D = SCALAPACK(numroc)(&Nu, &mb2, &myrow2, &I_ZERO, &nprow2);
    ncolsNuNe2D = SCALAPACK(numroc)(&Ne, &nb2, &mycol2, &I_ZERO, &npcol2);
    lldNuNe2D = std::max( nrowsNuNe2D, 1 );
  }

  SCALAPACK(descinit)(desc_NuNe2D, &Nu, &Ne, &mb2, &nb2, &I_ZERO, 
      &I_ZERO, &contxt2, &lldNuNe2D, &info2);

  DblNumMat phiCol( ntot, numStateLocal );
  SetValue( phiCol, 0.0 );

  DblNumMat psiCol( ntot, numStateLocal );
  SetValue( psiCol, 0.0 );

  lapack::Lacpy( 'A', ntot, numStateLocal, phi.Data(), ntot, phiCol.Data(), ntot );
  lapack::Lacpy( 'A', ntot, numStateLocal, wavefun_.Data(), ntot, psiCol.Data(), ntot );

  // Computing the indices is optional


  Int ntotLocal = nrowsNgNe1DRow; 
  //Int ntotLocalMG = nrowsNgNe1DRow;
  //Int ntotMG = ntot;
  Int ntotLocalMG, ntotMG;

  //if( (ntot % mpisize) == 0 ){
  //  ntotLocalMG = ntotBlocksize;
  //}
  //else{
  //  ntotLocalMG = ntotBlocksize + 1;
  // }
  //
  //------------by lijl 20200526
  /*
  if ((pivQR_.m_ != ntot) || (hybridDFType == "QRCP")){
    pivQR_.Resize(ntot);
    SetValue( pivQR_, 0 ); // Important. Otherwise QRCP uses piv as initial guess
    // Q factor does not need to be used
  }
  */

  if( (isFixColumnDF == false) && ((hybridDFType == "QRCP") || (hybridDFType == "Kmeans+QRCP"))){

    GetTime( timeSta );

    DblNumMat localphiGRow( ntotLocal, numPre );
    SetValue( localphiGRow, 0.0 );

    DblNumMat localpsiGRow( ntotLocal, numPre );
    SetValue( localpsiGRow, 0.0 );

    DblNumMat G(numStateTotal, numPre);
    SetValue( G, 0.0 );

    DblNumMat phiRow( ntotLocal, numStateTotal );
    SetValue( phiRow, 0.0 );

    DblNumMat psiRow( ntotLocal, numStateTotal );
    SetValue( psiRow, 0.0 );

    SCALAPACK(pdgemr2d)(&Ng, &Ne, psiCol.Data(), &I_ONE, &I_ONE, desc_NgNe1DCol, 
        psiRow.Data(), &I_ONE, &I_ONE, desc_NgNe1DRow, &contxt1 );

    SCALAPACK(pdgemr2d)(&Ng, &Ne, phiCol.Data(), &I_ONE, &I_ONE, desc_NgNe1DCol, 
        phiRow.Data(), &I_ONE, &I_ONE, desc_NgNe1DRow, &contxt1 );

    // Step 1: Pre-compression of the wavefunctions. This uses
    // multiplication with orthonormalized random Gaussian matrices

    GetTime( timeSta1 );
/*
    if (G_.m_!=numStateTotal){
      DblNumMat G(numStateTotal, numPre);
      if ( mpirank == 0 ) {
        GaussianRandom(G);
        lapack::Orth( numStateTotal, numPre, G.Data(), numStateTotal );
        statusOFS << "Random projection initialzied." << std::endl << std::endl;
      }
      MPI_Bcast(G.Data(), numStateTotal * numPre, MPI_DOUBLE, 0, domain_.comm);
      G_ = G;
    } else {
      statusOFS << "Random projection reused." << std::endl;
    }
    statusOFS << "G(0,0) = " << G_(0,0) << std::endl << std::endl;
*/
    if ( mpirank == 0 ) {
      GaussianRandom(G);
      lapack::Orth( numStateTotal, numPre, G.Data(), numStateTotal );
      statusOFS << "Random projection initialzied." << std::endl << std::endl;
    }
    MPI_Bcast(G.Data(), numStateTotal * numPre, MPI_DOUBLE, 0, domain_.comm);
//----------------------------------by lijl 20200526
    GetTime( timeEnd1 );
#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for Gaussian MPI_Bcast is " <<
      timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
#endif

    GetTime( timeSta1 );

    blas::Gemm( 'N', 'N', ntotLocal, numPre, numStateTotal, 1.0, 
        phiRow.Data(), ntotLocal, G.Data(), numStateTotal, 0.0,
        localphiGRow.Data(), ntotLocal );

    blas::Gemm( 'N', 'N', ntotLocal, numPre, numStateTotal, 1.0, 
        psiRow.Data(), ntotLocal, G.Data(), numStateTotal, 0.0,
        localpsiGRow.Data(), ntotLocal );

    GetTime( timeEnd1 );
#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for localphiGRow and localpsiGRow Gemm is " <<
      timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
#endif

    // Step 2: Pivoted QR decomposition  for the Hadamard product of
    // the compressed matrix. Transpose format for QRCP

    // NOTE: All processors should have the same ntotLocalMG


    int m_MGTemp = numPre*numPre;
    int n_MGTemp = ntot;

    DblNumMat MGCol( m_MGTemp, ntotLocal );

    DblNumVec MGNorm(ntot);
    for( Int k = 0; k < ntot; k++ ){
      MGNorm(k) = 0;
    }
    DblNumVec MGNormLocal(ntotLocal);
    for( Int k = 0; k < ntotLocal; k++ ){
      MGNormLocal(k) = 0;
    }

    GetTime( timeSta1 );

    for( Int j = 0; j < numPre; j++ ){
      for( Int i = 0; i < numPre; i++ ){
        for( Int ir = 0; ir < ntotLocal; ir++ ){
          MGCol(i+j*numPre,ir) = localphiGRow(ir,i) * localpsiGRow(ir,j);
          MGNormLocal(ir) += MGCol(i+j*numPre,ir) * MGCol(i+j*numPre,ir);   
        }
      }
    }

    GetTime( timeEnd1 );
#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for computing MG from localphiGRow and localpsiGRow is " <<
      timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
#endif

    //int ntotMGTemp, ntotLocalMGTemp;

    IntNumVec MGIdx(ntot);

    {
      Int ncols0D, nrows0D, lld0D; 
      Int ncols1D, nrows1D, lld1D; 

      Int desc_0D[9];
      Int desc_1D[9];

      Cblacs_get(0, 0, &contxt0);
      Cblacs_gridinit(&contxt0, "C", nprow0, npcol0);
      Cblacs_gridinfo(contxt0, &nprow0, &npcol0, &myrow0, &mycol0);

      SCALAPACK(descinit)(desc_0D, &Ng, &I_ONE, &Ng, &I_ONE, &I_ZERO, &I_ZERO, &contxt0, &Ng, &info0);

      if(contxt11 >= 0){
        nrows1D = SCALAPACK(numroc)(&Ng, &BlockSizeScaLAPACKTemp, &myrow11, &I_ZERO, &nprow11);
        ncols1D = SCALAPACK(numroc)(&I_ONE, &I_ONE, &mycol11, &I_ZERO, &npcol11);
        lld1D = std::max( nrows1D, 1 );
      }    

      SCALAPACK(descinit)(desc_1D, &Ng, &I_ONE, &BlockSizeScaLAPACKTemp, &I_ONE, &I_ZERO, 
          &I_ZERO, &contxt11, &lld1D, &info11);


      SCALAPACK(pdgemr2d)(&Ng, &I_ONE, MGNormLocal.Data(), &I_ONE, &I_ONE, desc_1D, 
          MGNorm.Data(), &I_ONE, &I_ONE, desc_0D, &contxt11 );

      MPI_Bcast( MGNorm.Data(), ntot, MPI_DOUBLE, 0, domain_.comm );


      double MGNormMax = *(std::max_element( MGNorm.Data(), MGNorm.Data() + ntot ) );
      double MGNormMin = *(std::min_element( MGNorm.Data(), MGNorm.Data() + ntot ) );

      ntotMG = 0;
      SetValue( MGIdx, 0 );

      for( Int k = 0; k < ntot; k++ ){
        if(MGNorm(k) > hybridDFTolerance){
          MGIdx(ntotMG) = k; 
          ntotMG = ntotMG + 1;
        }
      }

#if ( _DEBUGlevel_ >= 0 )
      statusOFS << "The col size for MG: " << " ntotMG = " << ntotMG << " ntotMG/ntot = " << Real(ntotMG)/Real(ntot) << std::endl << std::endl;
      statusOFS << "The norm range for MG: " <<  " MGNormMax = " << MGNormMax << " MGNormMin = " << MGNormMin << std::endl << std::endl;
#endif

    }

    Int ncols1DCol, nrows1DCol, lld1DCol; 
    Int ncols1DRow, nrows1DRow, lld1DRow; 

    Int desc_1DCol[9];
    Int desc_1DRow[9];

    if(contxt1 >= 0){
      nrows1DCol = SCALAPACK(numroc)(&m_MGTemp, &m_MGTemp, &myrow1, &I_ZERO, &nprow1);
      ncols1DCol = SCALAPACK(numroc)(&n_MGTemp, &BlockSizeScaLAPACKTemp, &mycol1, &I_ZERO, &npcol1);
      lld1DCol = std::max( nrows1DCol, 1 );
    }    

    SCALAPACK(descinit)(desc_1DCol, &m_MGTemp, &n_MGTemp, &m_MGTemp, &BlockSizeScaLAPACKTemp, &I_ZERO, 
        &I_ZERO, &contxt1, &lld1DCol, &info1);

    if(contxt11 >= 0){
      nrows1DRow = SCALAPACK(numroc)(&m_MGTemp, &I_ONE, &myrow11, &I_ZERO, &nprow11);
      ncols1DRow = SCALAPACK(numroc)(&n_MGTemp, &n_MGTemp, &mycol11, &I_ZERO, &npcol11);
      lld1DRow = std::max( nrows1DRow, 1 );
    }    

    SCALAPACK(descinit)(desc_1DRow, &m_MGTemp, &n_MGTemp, &I_ONE, &n_MGTemp, &I_ZERO, 
        &I_ZERO, &contxt11, &lld1DRow, &info11);

    DblNumMat MGRow( nrows1DRow, ntot );
    SetValue( MGRow, 0.0 );


    GetTime( timeSta1 );

    SCALAPACK(pdgemr2d)(&m_MGTemp, &n_MGTemp, MGCol.Data(), &I_ONE, &I_ONE, desc_1DCol, 
        MGRow.Data(), &I_ONE, &I_ONE, desc_1DRow, &contxt1 );

    GetTime( timeEnd1 );
#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for computing MG from Col and Row is " <<
      timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
#endif

    DblNumMat MG( nrows1DRow, ntotMG );

    for( Int i = 0; i < nrows1DRow; i++ ){
      for( Int j = 0; j < ntotMG; j++ ){
        MG(i,j) = MGRow(i,MGIdx(j));
      }
    }
//----------------by lijl 20200526
    DblNumVec tau(ntotMG);
    pivQR_.Resize(ntot);
    SetValue( pivQR_, 0 ); // Important. Otherwise QRCP uses piv as initial guess
    // Q factor does not need to be used

    for( Int k = 0; k < ntotMG; k++ ){
      tau[k] = 0.0;
    }

    Real timeQRCPSta, timeQRCPEnd;
    GetTime( timeQRCPSta );

    if(0){  
      lapack::QRCP( numPre*numPre, ntotMG, MG.Data(), numPre*numPre, 
          pivQR_.Data(), tau.Data() );
    }//


    if(0){ // ScaLAPACL QRCP
      Int contxt;
      Int nprow, npcol, myrow, mycol, info;
      Cblacs_get(0, 0, &contxt);
      nprow = 1;
      npcol = mpisize;

      Cblacs_gridinit(&contxt, "C", nprow, npcol);
      Cblacs_gridinfo(contxt, &nprow, &npcol, &myrow, &mycol);
      Int desc_MG[9];

      Int irsrc = 0;
      Int icsrc = 0;

      Int mb_MG = numPre*numPre;
      Int nb_MG = ntotLocalMG;

      // FIXME The current routine does not actually allow ntotLocal to be different on different processors.
      // This must be fixed.
      SCALAPACK(descinit)(&desc_MG[0], &mb_MG, &ntotMG, &mb_MG, &nb_MG, &irsrc, 
          &icsrc, &contxt, &mb_MG, &info);

      IntNumVec pivQRTmp(ntotMG), pivQRLocal(ntotMG);
      if( mb_MG > ntot ){
        std::ostringstream msg;
        msg << "numPre*numPre > ntot. The number of grid points is perhaps too small!" << std::endl;
        ErrorHandling( msg.str().c_str() );
      }
      // DiagR is only for debugging purpose
      //        DblNumVec diagRLocal( mb_MG );
      //        DblNumVec diagR( mb_MG );

      SetValue( pivQRTmp, 0 );
      SetValue( pivQRLocal, 0 );
      SetValue( pivQR_, 0 );


      //        SetValue( diagRLocal, 0.0 );
      //        SetValue( diagR, 0.0 );

      if(0) {
        scalapack::QRCPF( mb_MG, ntotMG, MG.Data(), &desc_MG[0], 
            pivQRTmp.Data(), tau.Data() );
      }

      if(1) {
        scalapack::QRCPR( mb_MG, ntotMG, numMu_, MG.Data(), &desc_MG[0], 
            pivQRTmp.Data(), tau.Data(), 80, 40 );
      }


      // Combine the local pivQRTmp to global pivQR_
      for( Int j = 0; j < ntotLocalMG; j++ ){
        pivQRLocal[j + mpirank * ntotLocalMG] = pivQRTmp[j];
      }

      //        std::cout << "diag of MG = " << std::endl;
      //        if(mpirank == 0){
      //          std::cout << pivQRLocal << std::endl;
      //          for( Int j = 0; j < mb_MG; j++ ){
      //            std::cout << MG(j,j) << std::endl;
      //          }
      //        }
      MPI_Allreduce( pivQRLocal.Data(), pivQR_.Data(), 
          ntotMG, MPI_INT, MPI_SUM, domain_.comm );


      if(contxt >= 0) {
        Cblacs_gridexit( contxt );
      }

    } //ScaLAPACL QRCP


    if(1){ //ScaLAPACL QRCP 2D

      Int contxt1D, contxt2D;
      Int nprow1D, npcol1D, myrow1D, mycol1D, info1D;
      Int nprow2D, npcol2D, myrow2D, mycol2D, info2D;

      Int ncols1D, nrows1D, lld1D; 
      Int ncols2D, nrows2D, lld2D; 

      Int desc_MG1D[9];
      Int desc_MG2D[9];

      Int m_MG = numPre*numPre;
      Int n_MG = ntotMG;

      Int mb_MG1D = 1;
      Int nb_MG1D = ntotMG;

      nprow1D = mpisize;
      npcol1D = 1;

      Cblacs_get(0, 0, &contxt1D);
      Cblacs_gridinit(&contxt1D, "C", nprow1D, npcol1D);
      Cblacs_gridinfo(contxt1D, &nprow1D, &npcol1D, &myrow1D, &mycol1D);

      nrows1D = SCALAPACK(numroc)(&m_MG, &mb_MG1D, &myrow1D, &I_ZERO, &nprow1D);
      ncols1D = SCALAPACK(numroc)(&n_MG, &nb_MG1D, &mycol1D, &I_ZERO, &npcol1D);

      lld1D = std::max( nrows1D, 1 );

      SCALAPACK(descinit)(desc_MG1D, &m_MG, &n_MG, &mb_MG1D, &nb_MG1D, &I_ZERO, 
          &I_ZERO, &contxt1D, &lld1D, &info1D);

      for( Int i = std::min(mpisize, IRound(sqrt(double(mpisize*(n_MG/m_MG))))); 
          i <= mpisize; i++){
        npcol2D = i; nprow2D = mpisize / npcol2D;
        if( (npcol2D >= nprow2D) && (nprow2D * npcol2D == mpisize) ) break;
      }

      Cblacs_get(0, 0, &contxt2D);
      //Cblacs_gridinit(&contxt2D, "C", nprow2D, npcol2D);

      IntNumVec pmap(mpisize);
      for ( Int i = 0; i < mpisize; i++ ){
        pmap[i] = i;
      }
      Cblacs_gridmap(&contxt2D, &pmap[0], nprow2D, nprow2D, npcol2D);

      Int m_MG2DBlocksize = BlockSizeScaLAPACK;
      Int n_MG2DBlocksize = BlockSizeScaLAPACK;

      //Int m_MG2Local = m_MG/(m_MG2DBlocksize*nprow2D)*m_MG2DBlocksize;
      //Int n_MG2Local = n_MG/(n_MG2DBlocksize*npcol2D)*n_MG2DBlocksize;
      Int m_MG2Local, n_MG2Local;

      //MPI_Comm rowComm = MPI_COMM_NULL;
      MPI_Comm colComm = MPI_COMM_NULL;

      Int mpirankRow, mpisizeRow, mpirankCol, mpisizeCol;

      //MPI_Comm_split( domain_.comm, mpirank / nprow2D, mpirank, &rowComm );
      MPI_Comm_split( domain_.comm, mpirank % nprow2D, mpirank, &colComm );

      //MPI_Comm_rank(rowComm, &mpirankRow);
      //MPI_Comm_size(rowComm, &mpisizeRow);

      MPI_Comm_rank(colComm, &mpirankCol);
      MPI_Comm_size(colComm, &mpisizeCol);

      if(0){

        if((m_MG % (m_MG2DBlocksize * nprow2D))!= 0){ 
          if(mpirankRow < ((m_MG % (m_MG2DBlocksize*nprow2D)) / m_MG2DBlocksize)){
            m_MG2Local = m_MG2Local + m_MG2DBlocksize;
          }
          if(((m_MG % (m_MG2DBlocksize*nprow2D)) % m_MG2DBlocksize) != 0){
            if(mpirankRow == ((m_MG % (m_MG2DBlocksize*nprow2D)) / m_MG2DBlocksize)){
              m_MG2Local = m_MG2Local + ((m_MG % (m_MG2DBlocksize*nprow2D)) % m_MG2DBlocksize);
            }
          }
        }

        if((n_MG % (n_MG2DBlocksize * npcol2D))!= 0){ 
          if(mpirankCol < ((n_MG % (n_MG2DBlocksize*npcol2D)) / n_MG2DBlocksize)){
            n_MG2Local = n_MG2Local + n_MG2DBlocksize;
          }
          if(((n_MG % (n_MG2DBlocksize*nprow2D)) % n_MG2DBlocksize) != 0){
            if(mpirankCol == ((n_MG % (n_MG2DBlocksize*npcol2D)) / n_MG2DBlocksize)){
              n_MG2Local = n_MG2Local + ((n_MG % (n_MG2DBlocksize*nprow2D)) % n_MG2DBlocksize);
            }
          }
        }

      } // if(0)

      if(contxt2D >= 0){
        Cblacs_gridinfo(contxt2D, &nprow2D, &npcol2D, &myrow2D, &mycol2D);
        nrows2D = SCALAPACK(numroc)(&m_MG, &m_MG2DBlocksize, &myrow2D, &I_ZERO, &nprow2D);
        ncols2D = SCALAPACK(numroc)(&n_MG, &n_MG2DBlocksize, &mycol2D, &I_ZERO, &npcol2D);
        lld2D = std::max( nrows2D, 1 );
      }

      SCALAPACK(descinit)(desc_MG2D, &m_MG, &n_MG, &m_MG2DBlocksize, &n_MG2DBlocksize, &I_ZERO, 
          &I_ZERO, &contxt2D, &lld2D, &info2D);

      m_MG2Local = nrows2D;
      n_MG2Local = ncols2D;

      IntNumVec pivQRTmp1(ntotMG), pivQRTmp2(ntotMG), pivQRLocal(ntotMG);
      if( m_MG > ntot ){
        std::ostringstream msg;
        msg << "numPre*numPre > ntot. The number of grid points is perhaps too small!" << std::endl;
        ErrorHandling( msg.str().c_str() );
      }
      // DiagR is only for debugging purpose
      //        DblNumVec diagRLocal( mb_MG );
      //        DblNumVec diagR( mb_MG );

      DblNumMat&  MG1D = MG;
      DblNumMat  MG2D (m_MG2Local, n_MG2Local);

      SCALAPACK(pdgemr2d)(&m_MG, &n_MG, MG1D.Data(), &I_ONE, &I_ONE, desc_MG1D, 
          MG2D.Data(), &I_ONE, &I_ONE, desc_MG2D, &contxt1D );

      if(contxt2D >= 0){

        Real timeQRCP1, timeQRCP2;
        GetTime( timeQRCP1 );

        SetValue( pivQRTmp1, 0 );
        scalapack::QRCPF( m_MG, n_MG, MG2D.Data(), desc_MG2D, pivQRTmp1.Data(), tau.Data() );

        //scalapack::QRCPR( m_MG, n_MG, numMu_, MG2D.Data(), desc_MG2D, pivQRTmp1.Data(), tau.Data(), BlockSizeScaLAPACK, 32);

        GetTime( timeQRCP2 );

#if ( _DEBUGlevel_ >= 0 )
        statusOFS << "Time only for QRCP is " << timeQRCP2 - timeQRCP1 << " [s]" << std::endl << std::endl;
#endif

      }

      // Redistribute back eigenvectors
      //SetValue(MG1D, 0.0 );

      //SCALAPACK(pdgemr2d)( &m_MG, &n_MG, MG2D.Data(), &I_ONE, &I_ONE, desc_MG2D,
      //    MG1D.Data(), &I_ONE, &I_ONE, desc_MG1D, &contxt1D );

      // Combine the local pivQRTmp to global pivQR_
      SetValue( pivQRLocal, 0 );
      for( Int j = 0; j < n_MG2Local; j++ ){
        pivQRLocal[ (j / n_MG2DBlocksize) * n_MG2DBlocksize * npcol2D + mpirankCol * n_MG2DBlocksize + j % n_MG2DBlocksize] = pivQRTmp1[j];
      }

      SetValue( pivQRTmp2, 0 );
      MPI_Allreduce( pivQRLocal.Data(), pivQRTmp2.Data(), 
          ntotMG, MPI_INT, MPI_SUM, colComm );

      SetValue( pivQR_, 0 );
      for( Int j = 0; j < ntotMG; j++ ){
        pivQR_(j) = MGIdx(pivQRTmp2(j));
      }

      //if( rowComm != MPI_COMM_NULL ) MPI_Comm_free( & rowComm );
      if( colComm != MPI_COMM_NULL ) MPI_Comm_free( & colComm );

      if(contxt2D >= 0) {
        Cblacs_gridexit( contxt2D );
      }

      if(contxt1D >= 0) {
        Cblacs_gridexit( contxt1D );
      }

    } // if(1) ScaLAPACL QRCP

    GetTime( timeQRCPEnd );

    //statusOFS << std::endl<< "All processors exit with abort in spinor.cpp." << std::endl;

#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for QRCP alone is " <<
      timeQRCPEnd - timeQRCPSta << " [s]" << std::endl << std::endl;
#endif
    
    if(0){
      Real tolR = std::abs(MG(numMu_-1,numMu_-1)/MG(0,0));
      statusOFS << "numMu_ = " << numMu_ << std::endl;
      statusOFS << "|R(numMu-1,numMu-1)/R(0,0)| = " << tolR << std::endl;
    }

    GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for density fitting with QRCP is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

    // Dump out pivQR_
    if(0){
      std::ostringstream muStream;
      serialize( pivQR_, muStream, NO_MASK );
      SharedWrite( "pivQR", muStream );
    }

    if (hybridDFType == "Kmeans+QRCP"){
      hybridDFType = "Kmeans";
    }
  } //  if( (isFixColumnDF == false) && (hybridDFType == "QRCP") )

  if( (isFixColumnDF == false) && (hybridDFType == "Kmeans") ){
   
    GetTime( timeSta );

    DblNumVec weight(ntot);
    Real* wp = weight.Data();
    Real timeW1,timeW2;
    Real timeKMEANSta, timeKMEANEnd;

    GetTime(timeW1); 
    DblNumVec phiW(ntot);
    SetValue(phiW,0.0);
    Real* phW = phiW.Data();
    Real* ph = phiCol.Data();

    for (int j = 0; j < numStateLocal; j++){
      for(int i = 0; i < ntot; i++){
        phW[i] += ph[i+j*ntot]*ph[i+j*ntot];
      }
    }
    MPI_Barrier(domain_.comm);
    MPI_Reduce(phW, wp, ntot, MPI_DOUBLE, MPI_SUM, 0, domain_.comm);
    MPI_Bcast(wp, ntot, MPI_DOUBLE, 0, domain_.comm);
    GetTime(timeW2);
    statusOFS << "Time for computing weight in Kmeans: " << timeW2-timeW1 << "[s]" << std::endl << std::endl;

    int rk = numMu_;
    pivQR_.Resize(ntot); // important,should be added
    SetValue( pivQR_, 0 ); // Important. Otherwise QRCP uses piv as initial guess
    GetTime(timeKMEANSta);
    KMEAN(ntot, weight, rk, hybridDFKmeansTolerance, hybridDFKmeansMaxIter, hybridDFTolerance, domain_, pivQR_.Data());
    GetTime(timeKMEANEnd);
    statusOFS << "Time for Kmeans alone is " << timeKMEANEnd-timeKMEANSta << "[s]" << std::endl << std::endl;
 
    GetTime(timeEnd);
    
    statusOFS << "Time for density fitting with Kmeans is " << timeEnd-timeSta << "[s]" << std::endl << std::endl;

    // Dump out pivQR_
    if(0){
      std::ostringstream muStream;
      serialize( pivQR_, muStream, NO_MASK );
      SharedWrite( "pivQR", muStream );
    }

  } //  if( (isFixColumnDF == false) && (hybridDFType == "Kmeans") )

  // Load pivQR_ file
  if(0){
    statusOFS << "Loading pivQR file.." << std::endl;
    std::istringstream muStream;
    SharedRead( "pivQR", muStream );
    deserialize( pivQR_, muStream, NO_MASK );
  }

  // *********************************************************************
  // Compute the interpolation matrix via the density matrix formulation
  // *********************************************************************

  GetTime( timeSta );

  // PhiMu is scaled by the occupation number to reflect the "true" density matrix
  //IntNumVec pivMu(numMu_);
  IntNumVec pivMu1(numMu_);

  for( Int mu = 0; mu < numMu_; mu++ ){
    pivMu1(mu) = pivQR_(mu);
  }


  //if(ntot % mpisize != 0){
  //  for( Int mu = 0; mu < numMu_; mu++ ){
  //    Int k1 = (pivMu1(mu) / ntotLocalMG) - (ntot % mpisize);
  //    if(k1 > 0){
  //      pivMu1(mu) = pivQR_(mu) - k1;
  //    }
  //  }
  //}

  GetTime( timeSta1 );

  DblNumMat psiMuCol(numStateLocal, numMu_);
  DblNumMat phiMuCol(numStateLocal, numMu_);
  SetValue( psiMuCol, 0.0 );
  SetValue( phiMuCol, 0.0 );

  for (Int k=0; k<numStateLocal; k++) {
    for (Int mu=0; mu<numMu_; mu++) {
      psiMuCol(k, mu) = psiCol(pivMu1(mu),k);
      phiMuCol(k, mu) = phiCol(pivMu1(mu),k) * occupationRate[(k * mpisize + mpirank)];
    }
  }

  DblNumMat psiMu2D(nrowsNeNu2D, ncolsNeNu2D);
  DblNumMat phiMu2D(nrowsNeNu2D, ncolsNeNu2D);
  SetValue( psiMu2D, 0.0 );
  SetValue( phiMu2D, 0.0 );

  SCALAPACK(pdgemr2d)(&Ne, &Nu, psiMuCol.Data(), &I_ONE, &I_ONE, desc_NeNu1D, 
      psiMu2D.Data(), &I_ONE, &I_ONE, desc_NeNu2D, &contxt11 );

  SCALAPACK(pdgemr2d)(&Ne, &Nu, phiMuCol.Data(), &I_ONE, &I_ONE, desc_NeNu1D, 
      phiMu2D.Data(), &I_ONE, &I_ONE, desc_NeNu2D, &contxt11 );

  GetTime( timeEnd1 );

#if ( _DEBUGlevel_ >= 0 )
  statusOFS << "Time for computing psiMuRow and phiMuRow is " <<
    timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
#endif

  GetTime( timeSta1 );

  DblNumMat psi2D(nrowsNgNe2D, ncolsNgNe2D);
  DblNumMat phi2D(nrowsNgNe2D, ncolsNgNe2D);
  SetValue( psi2D, 0.0 );
  SetValue( phi2D, 0.0 );

  SCALAPACK(pdgemr2d)(&Ng, &Ne, psiCol.Data(), &I_ONE, &I_ONE, desc_NgNe1DCol, 
      psi2D.Data(), &I_ONE, &I_ONE, desc_NgNe2D, &contxt1 );
  SCALAPACK(pdgemr2d)(&Ng, &Ne, phiCol.Data(), &I_ONE, &I_ONE, desc_NgNe1DCol, 
      phi2D.Data(), &I_ONE, &I_ONE, desc_NgNe2D, &contxt1 );

  DblNumMat PpsiMu2D(nrowsNgNu2D, ncolsNgNu2D);
  DblNumMat PphiMu2D(nrowsNgNu2D, ncolsNgNu2D);
  SetValue( PpsiMu2D, 0.0 );
  SetValue( PphiMu2D, 0.0 );

  SCALAPACK(pdgemm)("N", "N", &Ng, &Nu, &Ne, 
      &D_ONE,
      psi2D.Data(), &I_ONE, &I_ONE, desc_NgNe2D,
      psiMu2D.Data(), &I_ONE, &I_ONE, desc_NeNu2D, 
      &D_ZERO,
      PpsiMu2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D);

  SCALAPACK(pdgemm)("N", "N", &Ng, &Nu, &Ne, 
      &D_ONE,
      phi2D.Data(), &I_ONE, &I_ONE, desc_NgNe2D,
      phiMu2D.Data(), &I_ONE, &I_ONE, desc_NeNu2D, 
      &D_ZERO,
      PphiMu2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D);

  GetTime( timeEnd1 );

#if ( _DEBUGlevel_ >= 0 )
  statusOFS << "Time for PpsiMu and PphiMu GEMM is " <<
    timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
#endif

  GetTime( timeSta1 );

  DblNumMat Xi2D(nrowsNgNu2D, ncolsNgNu2D);
  SetValue( Xi2D, 0.0 );

  Real* Xi2DPtr = Xi2D.Data();
  Real* PpsiMu2DPtr = PpsiMu2D.Data();
  Real* PphiMu2DPtr = PphiMu2D.Data();

  for( Int g = 0; g < nrowsNgNu2D * ncolsNgNu2D; g++ ){
    Xi2DPtr[g] = PpsiMu2DPtr[g] * PphiMu2DPtr[g];
  }

  DblNumMat Xi1D(nrowsNgNu1D, ncolsNuNu1D);
  SetValue( Xi1D, 0.0 );

  SCALAPACK(pdgemr2d)(&Ng, &Nu, Xi2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D, 
      Xi1D.Data(), &I_ONE, &I_ONE, desc_NgNu1D, &contxt2 );

  DblNumMat PMuNu1D(nrowsNuNu1D, ncolsNuNu1D);
  SetValue( PMuNu1D, 0.0 );

  for (Int mu=0; mu<nrowsNuNu1D; mu++) {
    for (Int nu=0; nu<ncolsNuNu1D; nu++) {
      PMuNu1D(mu, nu) = Xi1D(pivMu1(mu),nu);
    }
  }

  DblNumMat PMuNu2D(nrowsNuNu2D, ncolsNuNu2D);
  SetValue( PMuNu2D, 0.0 );

  SCALAPACK(pdgemr2d)(&Nu, &Nu, PMuNu1D.Data(), &I_ONE, &I_ONE, desc_NuNu1D, 
      PMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D, &contxt1 );

  GetTime( timeEnd1 );

#if ( _DEBUGlevel_ >= 0 )
  statusOFS << "Time for computing PMuNu is " <<
    timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
#endif


  //Method 1
  if(0){

    GetTime( timeSta1 );

    SCALAPACK(pdpotrf)("L", &Nu, PMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D, &info2);

    GetTime( timeEnd1 );

#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for PMuNu Potrf is " <<
      timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
#endif

    GetTime( timeSta1 );

    SCALAPACK(pdtrsm)("R", "L", "T", "N", &Ng, &Nu, &D_ONE,
        PMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D,
        Xi2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D);

    SCALAPACK(pdtrsm)("R", "L", "N", "N", &Ng, &Nu, &D_ONE,
        PMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D,
        Xi2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D);

    GetTime( timeEnd1 );

#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for PMuNu and Xi pdtrsm is " <<
      timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
#endif

  }


  //Method 2
  if(1){

    GetTime( timeSta1 );

    SCALAPACK(pdpotrf)("L", &Nu, PMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D, &info2);
    SCALAPACK(pdpotri)("L", &Nu, PMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D, &info2);

    GetTime( timeEnd1 );

#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for PMuNu Potrf is " <<
      timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
#endif

    GetTime( timeSta1 );

    DblNumMat PMuNu2DTemp(nrowsNuNu2D, ncolsNuNu2D);
    SetValue( PMuNu2DTemp, 0.0 );

    lapack::Lacpy( 'A', nrowsNuNu2D, ncolsNuNu2D, PMuNu2D.Data(), nrowsNuNu2D, PMuNu2DTemp.Data(), nrowsNuNu2D );

    SCALAPACK(pdtradd)("U", "T", &Nu, &Nu, 
        &D_ONE,
        PMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D, 
        &D_ZERO,
        PMuNu2DTemp.Data(), &I_ONE, &I_ONE, desc_NuNu2D);

    DblNumMat Xi2DTemp(nrowsNgNu2D, ncolsNgNu2D);
    SetValue( Xi2DTemp, 0.0 );

    SCALAPACK(pdgemm)("N", "N", &Ng, &Nu, &Nu, 
        &D_ONE,
        Xi2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D,
        PMuNu2DTemp.Data(), &I_ONE, &I_ONE, desc_NuNu2D, 
        &D_ZERO,
        Xi2DTemp.Data(), &I_ONE, &I_ONE, desc_NgNu2D);

    SetValue( Xi2D, 0.0 );
    lapack::Lacpy( 'A', nrowsNgNu2D, ncolsNgNu2D, Xi2DTemp.Data(), nrowsNgNu2D, Xi2D.Data(), nrowsNgNu2D );

    GetTime( timeEnd1 );

#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for PMuNu and Xi pdgemm is " <<
      timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
#endif

  }


  //Method 3
  if(0){

    GetTime( timeSta1 );

    SCALAPACK(pdpotrf)("L", &Nu, PMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D, &info2);
    SCALAPACK(pdpotri)("L", &Nu, PMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D, &info2);

    GetTime( timeEnd1 );

#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for PMuNu Potrf is " <<
      timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
#endif

    GetTime( timeSta1 );

    DblNumMat Xi2DTemp(nrowsNgNu2D, ncolsNgNu2D);
    SetValue( Xi2DTemp, 0.0 );

    SCALAPACK(pdsymm)("R", "L", &Ng, &Nu, 
        &D_ONE,
        PMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D,
        Xi2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D,
        &D_ZERO,
        Xi2DTemp.Data(), &I_ONE, &I_ONE, desc_NgNu2D);

    SetValue( Xi2D, 0.0 );
    lapack::Lacpy( 'A', nrowsNgNu2D, ncolsNgNu2D, Xi2DTemp.Data(), nrowsNgNu2D, Xi2D.Data(), nrowsNgNu2D );

    GetTime( timeEnd1 );

#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for PMuNu and Xi pdsymm is " <<
      timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
#endif

  }


  //Method 4
  if(0){

    GetTime( timeSta1 );

    DblNumMat Xi2DTemp(nrowsNuNg2D, ncolsNuNg2D);
    SetValue( Xi2DTemp, 0.0 );

    DblNumMat PMuNu2DTemp(ncolsNuNu2D, nrowsNuNu2D);
    SetValue( PMuNu2DTemp, 0.0 );

    SCALAPACK(pdgeadd)("T", &Nu, &Ng,
        &D_ONE,
        Xi2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D,
        &D_ZERO,
        Xi2DTemp.Data(), &I_ONE, &I_ONE, desc_NuNg2D);

    SCALAPACK(pdgeadd)("T", &Nu, &Nu,
        &D_ONE,
        PMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D,
        &D_ZERO,
        PMuNu2DTemp.Data(), &I_ONE, &I_ONE, desc_NuNu2D);

    Int lwork=-1, info;
    double dummyWork;

    SCALAPACK(pdgels)("N", &Nu, &Nu, &Ng, 
        PMuNu2DTemp.Data(), &I_ONE, &I_ONE, desc_NuNu2D,
        Xi2DTemp.Data(), &I_ONE, &I_ONE, desc_NuNg2D,
        &dummyWork, &lwork, &info);

    lwork = dummyWork;
    std::vector<double> work(lwork);

    SCALAPACK(pdgels)("N", &Nu, &Nu, &Ng, 
        PMuNu2DTemp.Data(), &I_ONE, &I_ONE, desc_NuNu2D,
        Xi2DTemp.Data(), &I_ONE, &I_ONE, desc_NuNg2D,
        &work[0], &lwork, &info);

    SetValue( Xi2D, 0.0 );
    SCALAPACK(pdgeadd)("T", &Ng, &Nu,
        &D_ONE,
        Xi2DTemp.Data(), &I_ONE, &I_ONE, desc_NuNg2D,
        &D_ZERO,
        Xi2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D);

    GetTime( timeEnd1 );

#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for PMuNu and Xi pdgels is " <<
      timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
#endif

  }


  GetTime( timeEnd );

#if ( _DEBUGlevel_ >= 0 )
  statusOFS << "Time for computing the interpolation vectors is " <<
    timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif



  // *********************************************************************
  // Solve the Poisson equations.
  // Store VXi separately. This is not the most memory efficient
  // implementation
  // *********************************************************************

  DblNumMat VXi2D(nrowsNgNu2D, ncolsNgNu2D);

  {
    GetTime( timeSta );
    // XiCol used for both input and output
    DblNumMat XiCol(ntot, numMuLocal);
    SetValue(XiCol, 0.0 );

    SCALAPACK(pdgemr2d)(&Ng, &Nu, Xi2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D, 
        XiCol.Data(), &I_ONE, &I_ONE, desc_NgNu1D, &contxt2 );

    GetTime( timeSta );
    for( Int mu = 0; mu < numMuLocal; mu++ ){
      blas::Copy( ntot,  XiCol.VecData(mu), 1, fft.inputVecR2C.Data(), 1 );

      FFTWExecute ( fft, fft.forwardPlanR2C );

      for( Int ig = 0; ig < ntotR2C; ig++ ){
        fft.outputVecR2C(ig) *= -exxFraction * exxgkkR2C(ig);
      }

      FFTWExecute ( fft, fft.backwardPlanR2C );

      blas::Copy( ntot, fft.inputVecR2C.Data(), 1, XiCol.VecData(mu), 1 );

    } // for (mu)

    SetValue(VXi2D, 0.0 );
    SCALAPACK(pdgemr2d)(&Ng, &Nu, XiCol.Data(), &I_ONE, &I_ONE, desc_NgNu1D, 
        VXi2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D, &contxt1 );

    GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for solving Poisson-like equations is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
  }

  // Prepare for the computation of the M matrix
  GetTime( timeSta );

  DblNumMat MMatMuNu2D(nrowsNuNu2D, ncolsNuNu2D);
  SetValue(MMatMuNu2D, 0.0 );

  SCALAPACK(pdgemm)("T", "N", &Nu, &Nu, &Ng, 
      &D_MinusONE,
      Xi2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D,
      VXi2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D, 
      &D_ZERO,
      MMatMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D);

  DblNumMat phiMuNu2D(nrowsNuNu2D, ncolsNuNu2D);
  SCALAPACK(pdgemm)("T", "N", &Nu, &Nu, &Ne, 
      &D_ONE,
      phiMu2D.Data(), &I_ONE, &I_ONE, desc_NeNu2D,
      phiMu2D.Data(), &I_ONE, &I_ONE, desc_NeNu2D, 
      &D_ZERO,
      phiMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D);

  Real* MMat2DPtr = MMatMuNu2D.Data();
  Real* phi2DPtr  = phiMuNu2D.Data();

  for( Int g = 0; g < nrowsNuNu2D * ncolsNuNu2D; g++ ){
    MMat2DPtr[g] *= phi2DPtr[g];
  }

  GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
  statusOFS << "Time for preparing the M matrix is " <<
    timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

  // *********************************************************************
  // Compute the exchange potential and the symmetrized inner product
  // *********************************************************************

  GetTime( timeSta );

  // Rewrite VXi by VXi.*PcolPhi
  Real* VXi2DPtr = VXi2D.Data();
  Real* PphiMu2DPtr1 = PphiMu2D.Data();
  for( Int g = 0; g < nrowsNgNu2D * ncolsNgNu2D; g++ ){
    VXi2DPtr[g] *= PphiMu2DPtr1[g];
  }

  // NOTE: a3 must be zero in order to compute the M matrix later
  DblNumMat a32D( nrowsNgNe2D, ncolsNgNe2D );
  SetValue( a32D, 0.0 );

  SCALAPACK(pdgemm)("N", "T", &Ng, &Ne, &Nu, 
      &D_ONE,
      VXi2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D,
      psiMu2D.Data(), &I_ONE, &I_ONE, desc_NeNu2D, 
      &D_ZERO,
      a32D.Data(), &I_ONE, &I_ONE, desc_NgNe2D);

  DblNumMat a3Col( ntot, numStateLocal );
  SetValue(a3Col, 0.0 );

  SCALAPACK(pdgemr2d)(&Ng, &Ne, a32D.Data(), &I_ONE, &I_ONE, desc_NgNe2D, 
      a3Col.Data(), &I_ONE, &I_ONE, desc_NgNe1DCol, &contxt2 );

  lapack::Lacpy( 'A', ntot, numStateLocal, a3Col.Data(), ntot, a3.Data(), ntot );

  GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
  statusOFS << "Time for computing the exchange potential is " <<
    timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

  // Compute the matrix VxMat = -Psi'* vexxPsi and symmetrize
  // vexxPsi (a3) must be zero before entering this routine
  VxMat.Resize( numStateTotal, numStateTotal );

  GetTime( timeSta );

  if(1){

    DblNumMat VxMat2D( nrowsNeNe2D, ncolsNeNe2D );
    DblNumMat VxMatTemp2D( nrowsNuNe2D, ncolsNuNe2D );

    SCALAPACK(pdgemm)("N", "T", &Nu, &Ne, &Nu, 
        &D_ONE,
        MMatMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D,
        psiMu2D.Data(), &I_ONE, &I_ONE, desc_NeNu2D, 
        &D_ZERO,
        VxMatTemp2D.Data(), &I_ONE, &I_ONE, desc_NuNe2D);

    SCALAPACK(pdgemm)("N", "N", &Ne, &Ne, &Nu, 
        &D_ONE,
        psiMu2D.Data(), &I_ONE, &I_ONE, desc_NeNu2D, 
        VxMatTemp2D.Data(), &I_ONE, &I_ONE, desc_NuNe2D, 
        &D_ZERO,
        VxMat2D.Data(), &I_ONE, &I_ONE, desc_NeNe2D);

    SCALAPACK(pdgemr2d)(&Ne, &Ne, VxMat2D.Data(), &I_ONE, &I_ONE, desc_NeNe2D, 
        VxMat.Data(), &I_ONE, &I_ONE, desc_NeNe0D, &contxt2 );

    //if(mpirank == 0){
    //  MPI_Bcast( VxMat.Data(), Ne * Ne, MPI_DOUBLE, 0, domain_.comm );
    //}

  }
  GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
  statusOFS << "Time for computing VxMat in the sym format is " <<
    timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

  if(contxt0 >= 0) {
    Cblacs_gridexit( contxt0 );
  }

  if(contxt1 >= 0) {
    Cblacs_gridexit( contxt1 );
  }

  if(contxt11 >= 0) {
    Cblacs_gridexit( contxt11 );
  }

  if(contxt2 >= 0) {
    Cblacs_gridexit( contxt2 );
  }

  MPI_Barrier(domain_.comm);

  return ;
}        // -----  end of method Spinor::AddMultSpinorEXXDF7  ----- 

#endif

}  // namespace dgdft
