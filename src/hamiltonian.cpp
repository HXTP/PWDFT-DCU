/*
   Copyright (c) 2012 The Regents of the University of California,
   through Lawrence Berkeley National Laboratory.  

Author: Lin Lin, Wei Hu, Weile Jia

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
/// @file hamiltonian.cpp
/// @brief Hamiltonian class for planewave basis diagonalization method.
/// @date 2012-09-16
#include  "hamiltonian.hpp"
#include  "blas.hpp"
#include  "lapack.hpp"

namespace dgdft{

using namespace dgdft::PseudoComponent;
using namespace dgdft::DensityComponent;
using namespace dgdft::esdf;

// *********************************************************************
// KohnSham class
// *********************************************************************

KohnSham::KohnSham() {
  XCInitialized_ = false;
}

KohnSham::~KohnSham() {
  if( XCInitialized_ ){
    if( XCId_ == XC_LDA_XC_TETER93 )
    {
      xc_func_end(&XCFuncType_);
    }    
    else if( XId_ == XC_LDA_X && CId_ == XC_LDA_C_PZ )
    {
      xc_func_end(&XFuncType_);
      xc_func_end(&CFuncType_);
    }
    else if( ( XId_ == XC_GGA_X_PBE ) && ( CId_ == XC_GGA_C_PBE ) )
    {
      xc_func_end(&XFuncType_);
      xc_func_end(&CFuncType_);
    }
    else if( XCId_ == XC_HYB_GGA_XC_HSE06 ){
      xc_func_end(&XFuncType_);
      xc_func_end(&CFuncType_);
      xc_func_end(&XCFuncType_);
    }
    else if( XCId_ == XC_HYB_GGA_XC_PBEH ){
      xc_func_end(&XCFuncType_);
    }
    else
      ErrorHandling("Unrecognized exchange-correlation type");
  }
}

void
KohnSham::Setup    (
    const Domain&               dm,
    const std::vector<Atom>&    atomList )
{
  domain_              = dm;
  atomList_            = atomList;
  numExtraState_       = esdfParam.numExtraState;
  XCType_              = esdfParam.XCType;
  
  hybridDFType_                    = esdfParam.hybridDFType;
  hybridDFKmeansWFType_            = esdfParam.hybridDFKmeansWFType;
  hybridDFKmeansWFAlpha_           = esdfParam.hybridDFKmeansWFAlpha;
  hybridDFKmeansTolerance_         = esdfParam.hybridDFKmeansTolerance;
  hybridDFKmeansMaxIter_           = esdfParam.hybridDFKmeansMaxIter;
  hybridDFNumMu_                   = esdfParam.hybridDFNumMu;
  hybridDFNumGaussianRandom_       = esdfParam.hybridDFNumGaussianRandom;
  hybridDFNumProcScaLAPACK_        = esdfParam.hybridDFNumProcScaLAPACK;
  hybridDFTolerance_               = esdfParam.hybridDFTolerance;
  BlockSizeScaLAPACK_      = esdfParam.BlockSizeScaLAPACK;
  exxDivergenceType_   = esdfParam.exxDivergenceType;

  // FIXME Hard coded
  numDensityComponent_ = 1;

  // Since the number of density components is always 1 here, set numSpin = 2.
  numSpin_ = 2;

  // NOTE: NumSpin variable will be determined in derivative classes.

  Int ntotCoarse = domain_.NumGridTotal();
  Int ntotFine = domain_.NumGridTotalFine();

  density_.Resize( ntotFine, numDensityComponent_ );   
  SetValue( density_, 0.0 );

  densityold_.Resize( ntotFine, numDensityComponent_ );
  SetValue( densityold_, 0.0 );

  gradDensity_.resize( DIM );
  for( Int d = 0; d < DIM; d++ ){
    gradDensity_[d].Resize( ntotFine, numDensityComponent_ );
    SetValue (gradDensity_[d], 0.0);
  }

  pseudoCharge_.Resize( ntotFine );
  SetValue( pseudoCharge_, 0.0 );

  if( esdfParam.isUseVLocal == true ){
    vLocalSR_.Resize( ntotFine );
    SetValue( vLocalSR_, 0.0 );
  }
    


  vext_.Resize( ntotFine );
  SetValue( vext_, 0.0 );

  vhart_.Resize( ntotFine );
  SetValue( vhart_, 0.0 );

  vtot_.Resize( ntotFine );
  SetValue( vtot_, 0.0 );

  epsxc_.Resize( ntotFine );
  SetValue( epsxc_, 0.0 );

  vxc_.Resize( ntotFine, numDensityComponent_ );
  SetValue( vxc_, 0.0 );

  // MPI communication 
  MPI_Barrier(domain_.comm);
  int mpirank;  MPI_Comm_rank(domain_.comm, &mpirank);
  int mpisize;  MPI_Comm_size(domain_.comm, &mpisize);
  int dmCol = DIM;
  int dmRow = mpisize / dmCol;
  
  rowComm_ = MPI_COMM_NULL;
  colComm_ = MPI_COMM_NULL;

  if(mpisize >= DIM){

    IntNumVec mpiRowMap(mpisize);
    IntNumVec mpiColMap(mpisize);

    for( Int i = 0; i < mpisize; i++ ){
      mpiRowMap(i) = i / dmCol;
      mpiColMap(i) = i % dmCol;
    } 

    if( mpisize > dmRow * dmCol ){
      for( Int k = dmRow * dmCol; k < mpisize; k++ ){
        mpiRowMap(k) = dmRow - 1;
      }
    } 

    MPI_Comm_split( domain_.comm, mpiRowMap(mpirank), mpirank, &rowComm_ );
    //MPI_Comm_split( domain_.comm, mpiColMap(mpirank), mpirank, &colComm_ );

  }

  // Initialize the XC functionals, only spin-unpolarized case
  // Obtain the exchange-correlation id
  {
    isHybrid_ = false;
    isEXXActive_ = false;

    if( XCType_ == "XC_LDA_XC_TETER93" )
    { 
      XCId_ = XC_LDA_XC_TETER93;
      statusOFS << "XC_LDA_XC_TETER93  XCId = " << XCId_  << std::endl << std::endl;
      if( xc_func_init(&XCFuncType_, XCId_, XC_UNPOLARIZED) != 0 ){
        ErrorHandling( "XC functional initialization error." );
      } 
      // Teter 93
      // S Goedecker, M Teter, J Hutter, Phys. Rev B 54, 1703 (1996) 
    }    
    else if( XCType_ == "XC_GGA_XC_PBE" )
    {
      XId_ = XC_GGA_X_PBE;
      CId_ = XC_GGA_C_PBE;
      statusOFS << "XC_GGA_XC_PBE  XId_ CId_ = " << XId_ << " " << CId_  << std::endl << std::endl;
      // Perdew, Burke & Ernzerhof correlation
      // JP Perdew, K Burke, and M Ernzerhof, Phys. Rev. Lett. 77, 3865 (1996)
      // JP Perdew, K Burke, and M Ernzerhof, Phys. Rev. Lett. 78, 1396(E) (1997)
      if( xc_func_init(&XFuncType_, XId_, XC_UNPOLARIZED) != 0 ){
        ErrorHandling( "X functional initialization error." );
      }
      if( xc_func_init(&CFuncType_, CId_, XC_UNPOLARIZED) != 0 ){
        ErrorHandling( "C functional initialization error." );
      }
    }
    else if( XCType_ == "XC_HYB_GGA_XC_HSE06" )
    {
      XId_ = XC_GGA_X_PBE;
      CId_ = XC_GGA_C_PBE;

      statusOFS << "XC_GGA_XC_PBE  XId_ CId_ = " << XId_ << " " << CId_  << std::endl << std::endl;

      if( xc_func_init(&XFuncType_, XId_, XC_UNPOLARIZED) != 0 ){
        ErrorHandling( "X functional initialization error." );
      }
      if( xc_func_init(&CFuncType_, CId_, XC_UNPOLARIZED) != 0 ){
        ErrorHandling( "C functional initialization error." );
      }
      XCId_ = XC_HYB_GGA_XC_HSE06;
      statusOFS << "XC_HYB_GGA_XC_HSE06  XCId = " << XCId_  << std::endl << std::endl;
      if( xc_func_init(&XCFuncType_, XCId_, XC_UNPOLARIZED) != 0 ){
        ErrorHandling( "XC functional initialization error." );
      } 
      screenMu_ = 0.106;
      isHybrid_ = true;

      // J. Heyd, G. E. Scuseria, and M. Ernzerhof, J. Chem. Phys. 118, 8207 (2003) (doi: 10.1063/1.1564060)
      // J. Heyd, G. E. Scuseria, and M. Ernzerhof, J. Chem. Phys. 124, 219906 (2006) (doi: 10.1063/1.2204597)
      // A. V. Krukau, O. A. Vydrov, A. F. Izmaylov, and G. E. Scuseria, J. Chem. Phys. 125, 224106 (2006) (doi: 10.1063/1.2404663)
      //
      // This is the same as the "hse" functional in QE 5.1
    }
    else if( XCType_ == "XC_HYB_GGA_XC_PBEH" )
    {
      XCId_ = XC_HYB_GGA_XC_PBEH;
      statusOFS << "XC_HYB_GGA_XC_PBEH  XCId = " << XCId_  << std::endl << std::endl;
      if( xc_func_init(&XCFuncType_, XCId_, XC_UNPOLARIZED) != 0 ){
        ErrorHandling( "XC functional initialization error." );
      } 

      isHybrid_ = true;
      screenMu_ = 0.0;

      // C. Adamo and V. Barone, J. Chem. Phys. 110, 6158 (1999) (doi: 10.1063/1.478522)
      // M. Ernzerhof and G. E. Scuseria, J. Chem. Phys. 110, 5029 (1999) (doi: 10.1063/1.478401)  
    }
    else {
      ErrorHandling("Unrecognized exchange-correlation type");
    }
  }

  // ~~~ * ~~~
  // Set up wavefunction filter options: useful for CheFSI in PWDFT, for example
  // Affects the MATVEC operations in MultSpinor
  if(esdfParam.PWSolver == "CheFSI")
    set_wfn_filter(esdfParam.PWDFT_Cheby_apply_wfn_ecut_filt, 1, esdfParam.ecutWavefunction);
  else
    set_wfn_filter(0, 0, esdfParam.ecutWavefunction);



  return ;
}         // -----  end of method KohnSham::Setup  ----- 

void
KohnSham::CalculatePseudoPotential    ( PeriodTable &ptable ){

  MPI_Barrier(domain_.comm);
  int mpirank;  MPI_Comm_rank(domain_.comm, &mpirank);
  int mpisize;  MPI_Comm_size(domain_.comm, &mpisize);

  Int ntotFine = domain_.NumGridTotalFine();
  Int numAtom = atomList_.size();
  Real vol = domain_.Volume();

  pseudo_.clear();
  pseudo_.resize( numAtom );

  std::vector<DblNumVec> gridpos;
  UniformMeshFine ( domain_, gridpos );

  // calculate the number of occupied states
  // need to distinguish the number of charges carried by the ion and that 
  // carried by the electron
  Int nZion = 0, nelec = 0;
  for (Int a=0; a<numAtom; a++) {
    Int atype  = atomList_[a].type;
    if( ptable.ptemap().find(atype) == ptable.ptemap().end() ){
      ErrorHandling( "Cannot find the atom type." );
    }
    nZion = nZion + ptable.Zion(atype);
  }

  // add the extra electron
  nelec = nZion + esdfParam.extraElectron;
 
  // FIXME Deal with the case when this is a buffer calculation and the
  // number of electrons is not a even number.
  //
//  if( nelec % 2 != 0 ){
//    ErrorHandling( "This is spin-restricted calculation. nelec should be even." );
//  }
  numOccupiedState_ = nelec / numSpin_;

#if ( _DEBUGlevel_ >= 0 )
  Print( statusOFS, "Number of electrons in extend element        = ", nelec );
  Print( statusOFS, "Number of Occupied States                    = ", numOccupiedState_ );
#endif

  // Compute pseudocharge

  Real timeSta, timeEnd;
  Real time1, time2;
  Int  iterCharge = 0;
  Real timeCharge = 0.0;
  Int  iterNonlocal = 0;
  Real timeNonlocal = 0.0;
  Real timeSta1, timeEnd1;

  int numAtomBlocksize = numAtom  / mpisize;
  int numAtomLocal = numAtomBlocksize;
  if(mpirank < (numAtom % mpisize)){
    numAtomLocal = numAtomBlocksize + 1;
  }
  IntNumVec numAtomIdx( numAtomLocal );

  if (numAtomBlocksize == 0 ){
    for (Int i = 0; i < numAtomLocal; i++){
      numAtomIdx[i] = mpirank;
    }
  }
  else {
    if ( (numAtom % mpisize) == 0 ){
      for (Int i = 0; i < numAtomLocal; i++){
        numAtomIdx[i] = numAtomBlocksize * mpirank + i;
      }
    }
    else{
      for (Int i = 0; i < numAtomLocal; i++){
        if ( mpirank < (numAtom % mpisize) ){
          numAtomIdx[i] = (numAtomBlocksize + 1) * mpirank + i;
        }
        else{
          numAtomIdx[i] = (numAtomBlocksize + 1) * (numAtom % mpisize) + numAtomBlocksize * (mpirank - (numAtom % mpisize)) + i;
        }
      }
    }
  }

  IntNumVec numAtomMpirank( numAtom );

  if (numAtomBlocksize == 0 ){
    for (Int i = 0; i < numAtom; i++){
      numAtomMpirank[i] = i % mpisize;
    }
  }
  else {
    if ( (numAtom % mpisize) == 0 ){
      for (Int i = 0; i < numAtom; i++){
        numAtomMpirank[i] = i / numAtomBlocksize;
      }
    }
    else{
      for (Int i = 0; i < numAtom; i++){
        if ( i < (numAtom % mpisize) * (numAtomBlocksize + 1) ){
          numAtomMpirank[i] = i / (numAtomBlocksize + 1);
        }
        else{
          numAtomMpirank[i] = numAtom % mpisize + (i - (numAtom % mpisize) * (numAtomBlocksize + 1)) / numAtomBlocksize;
        }
      }
    }
  }

  GetTime( timeSta );

  Print( statusOFS, "Computing the local pseudopotential" );

  if( esdfParam.isUseVLocal == false )
  {

    DblNumVec pseudoChargeLocal(ntotFine);
    SetValue( pseudoChargeLocal, 0.0 );

//    GetTime(time1);
    for (Int i=0; i<numAtomLocal; i++) {
      int a = numAtomIdx[i];

//      GetTime(timeSta1);
      ptable.CalculatePseudoCharge( atomList_[a], domain_, 
          gridpos, pseudo_[a].pseudoCharge );
//      GetTime(timeEnd1);
//      iterCharge = iterCharge + 1;
//      timeCharge = timeCharge + timeEnd1 - timeSta1;

      //accumulate to the global vector
      IntNumVec &idx = pseudo_[a].pseudoCharge.first;
      DblNumMat &val = pseudo_[a].pseudoCharge.second;
      for (Int k=0; k<idx.m(); k++) 
        pseudoChargeLocal[idx(k)] += val(k, VAL);
      // For debug purpose, check the summation of the derivative
      if(0){
        Real sumVDX = 0.0, sumVDY = 0.0, sumVDZ = 0.0;
        for (Int k=0; k<idx.m(); k++) {
          sumVDX += val(k, DX);
          sumVDY += val(k, DY);
          sumVDZ += val(k, DZ);
        }
        sumVDX *= vol / Real(ntotFine);
        sumVDY *= vol / Real(ntotFine);
        sumVDZ *= vol / Real(ntotFine);
        if( std::sqrt(sumVDX * sumVDX + sumVDY * sumVDY + sumVDZ * sumVDZ) 
            > 1e-8 ){
          Print( statusOFS, "Local pseudopotential may not be constructed correctly" );
          Print( statusOFS, "For Atom ", a );
          Print( statusOFS, "Sum dV_a / dx = ", sumVDX );
          Print( statusOFS, "Sum dV_a / dy = ", sumVDY );
          Print( statusOFS, "Sum dV_a / dz = ", sumVDZ );
        }
      }
    }

//    GetTime(time2);

//    statusOFS << "Time for iterCharge        = " << iterCharge          << "  timeCharge        = " << timeCharge << std::endl;
//    statusOFS << "Time for CalculatePseudoCharge  = " << time2- time1 << std::endl;

    SetValue( pseudoCharge_, 0.0 );
    MPI_Allreduce( pseudoChargeLocal.Data(), pseudoCharge_.Data(), ntotFine, MPI_DOUBLE, MPI_SUM, domain_.comm );

//    GetTime(time1);
    for (Int a=0; a<numAtom; a++) {

      std::stringstream vStream;
      std::stringstream vStreamTemp;
      int vStreamSize;

      PseudoPot& pseudott = pseudo_[a]; 

      serialize( pseudott, vStream, NO_MASK );

      if (numAtomMpirank[a] == mpirank){
        vStreamSize = Size( vStream );
      }

      MPI_Bcast( &vStreamSize, 1, MPI_INT, numAtomMpirank[a], domain_.comm );

      std::vector<char> sstr;
      sstr.resize( vStreamSize );

      if (numAtomMpirank[a] == mpirank){
        vStream.read( &sstr[0], vStreamSize );
      }

      MPI_Bcast( &sstr[0], vStreamSize, MPI_BYTE, numAtomMpirank[a], domain_.comm );

      vStreamTemp.write( &sstr[0], vStreamSize );

      deserialize( pseudott, vStreamTemp, NO_MASK );

    }
//    GetTime(time2);
//    statusOFS << "Time for CalculatePseudoCharge comm  = " << time2- time1 << std::endl;


    GetTime( timeEnd );

//    statusOFS << "Time for no Vlocal pseudopotential " << timeEnd - timeSta  << std::endl;

    Real sumrho = 0.0;
    for (Int i=0; i<ntotFine; i++) 
      sumrho += pseudoCharge_[i]; 
    sumrho *= vol / Real(ntotFine);

    Print( statusOFS, "Sum of Pseudocharge                          = ", 
        sumrho );
    Print( statusOFS, "Number of Occupied States                    = ", 
        numOccupiedState_ );
    // adjustment should be multiplicative
    Real fac = nZion / sumrho;
    for (Int i=0; i<ntotFine; i++) 
      pseudoCharge_(i) *= fac; 

    Print( statusOFS, "After adjustment, Sum of Pseudocharge        = ", 
        (Real) nZion );
  } // Use the pseudocharge formulation
  else{
    DblNumVec pseudoChargeLocal(ntotFine);
    DblNumVec vLocalSRLocal(ntotFine);
    SetValue( pseudoChargeLocal, 0.0 );
    SetValue( vLocalSRLocal, 0.0 );

    for (Int i=0; i<numAtomLocal; i++) {
      int a = numAtomIdx[i];
      ptable.CalculateVLocal( atomList_[a], domain_, 
          gridpos, pseudo_[a].vLocalSR, pseudo_[a].pseudoCharge );

      //statusOFS << "Finish the computation of VLocal for atom " << i << std::endl;

      //accumulate to the global vector
      {
        IntNumVec &idx = pseudo_[a].pseudoCharge.first;
        DblNumMat &val = pseudo_[a].pseudoCharge.second;
        for (Int k=0; k<idx.m(); k++) 
          pseudoChargeLocal[idx(k)] += val(k, VAL);

        // For debug purpose, check the summation of the derivative
        if(0){
          Real sumVDX = 0.0, sumVDY = 0.0, sumVDZ = 0.0;
          for (Int k=0; k<idx.m(); k++) {
            sumVDX += val(k, DX);
            sumVDY += val(k, DY);
            sumVDZ += val(k, DZ);
          }
          sumVDX *= vol / Real(ntotFine);
          sumVDY *= vol / Real(ntotFine);
          sumVDZ *= vol / Real(ntotFine);
          if( std::sqrt(sumVDX * sumVDX + sumVDY * sumVDY + sumVDZ * sumVDZ) 
              > 1e-8 ){
            Print( statusOFS, "Local pseudopotential may not be constructed correctly" );
            Print( statusOFS, "For Atom ", a );
            Print( statusOFS, "Sum dV_a / dx = ", sumVDX );
            Print( statusOFS, "Sum dV_a / dy = ", sumVDY );
            Print( statusOFS, "Sum dV_a / dz = ", sumVDZ );
          }
        }
      }
      {
        IntNumVec &idx = pseudo_[a].vLocalSR.first;
        DblNumMat &val = pseudo_[a].vLocalSR.second;
        for (Int k=0; k<idx.m(); k++) 
          vLocalSRLocal[idx(k)] += val(k, VAL);
      }
    }

    SetValue( pseudoCharge_, 0.0 );
    SetValue( vLocalSR_, 0.0 );
    MPI_Allreduce( pseudoChargeLocal.Data(), pseudoCharge_.Data(), ntotFine, MPI_DOUBLE, MPI_SUM, domain_.comm );
    MPI_Allreduce( vLocalSRLocal.Data(), vLocalSR_.Data(), ntotFine, MPI_DOUBLE, MPI_SUM, domain_.comm );

    for (Int a=0; a<numAtom; a++) {

      std::stringstream vStream;
      std::stringstream vStreamTemp;
      int vStreamSize;

      PseudoPot& pseudott = pseudo_[a]; 

      serialize( pseudott, vStream, NO_MASK );

      if (numAtomMpirank[a] == mpirank){
        vStreamSize = Size( vStream );
      }

      MPI_Bcast( &vStreamSize, 1, MPI_INT, numAtomMpirank[a], domain_.comm );

      std::vector<char> sstr;
      sstr.resize( vStreamSize );

      if (numAtomMpirank[a] == mpirank){
        vStream.read( &sstr[0], vStreamSize );
      }

      MPI_Bcast( &sstr[0], vStreamSize, MPI_BYTE, numAtomMpirank[a], domain_.comm );

      vStreamTemp.write( &sstr[0], vStreamSize );

      deserialize( pseudott, vStreamTemp, NO_MASK );

    }

    GetTime( timeEnd );

//    statusOFS << "Time for using local pseudopotential " << timeEnd - timeSta  << std::endl;

    Real sumrho = 0.0;
    for (Int i=0; i<ntotFine; i++){
      sumrho += pseudoCharge_[i]; 
    }
    sumrho *= vol / Real(ntotFine);

    Print( statusOFS, "Sum of Pseudocharge                          = ", 
        sumrho );
    Print( statusOFS, "Number of Occupied States                    = ", 
        numOccupiedState_ );

    // adjustment should be multiplicative
    Real fac = nZion / sumrho;
    for (Int i=0; i<ntotFine; i++) 
      pseudoCharge_(i) *= fac; 

    Print( statusOFS, "After adjustment, Sum of Pseudocharge        = ", 
        (Real) nZion );

//    statusOFS << "vLocalSR = " << vLocalSR_  << std::endl;
//    statusOFS << "pseudoCharge = " << pseudoCharge_ << std::endl;
  } // Use the VLocal formulation
 
  // Nonlocal projectors
  // FIXME. Remove the contribution form the coarse grid

  GetTime( timeSta );

  Print( statusOFS, "Computing the non-local pseudopotential" );

  Int cnt = 0; // the total number of PS used
  Int cntLocal = 0; // the total number of PS used

  GetTime(time1);
  for (Int i=0; i<numAtomLocal; i++) {

//    GetTime(timeSta1);
    int a = numAtomIdx[i];
    // Introduce the nonlocal pseudopotential on the fine grid.
    ptable.CalculateNonlocalPP( atomList_[a], domain_, gridpos,
        pseudo_[a].vnlList ); 
//    GetTime(timeEnd1);
//    iterNonlocal = iterNonlocal + 1;
//    timeNonlocal = timeNonlocal + timeEnd1 - timeSta1;

    cntLocal = cntLocal + pseudo_[a].vnlList.size();

    // For debug purpose, check the summation of the derivative
    if(0){
      std::vector<NonlocalPP>& vnlList = pseudo_[a].vnlList;
      for( Int l = 0; l < vnlList.size(); l++ ){
        SparseVec& bl = vnlList[l].first;
        IntNumVec& idx = bl.first;
        DblNumMat& val = bl.second;
        Real sumVDX = 0.0, sumVDY = 0.0, sumVDZ = 0.0;
        for (Int k=0; k<idx.m(); k++) {
          sumVDX += val(k, DX);
          sumVDY += val(k, DY);
          sumVDZ += val(k, DZ);
        }
        sumVDX *= vol / Real(ntotFine);
        sumVDY *= vol / Real(ntotFine);
        sumVDZ *= vol / Real(ntotFine);
        if( std::sqrt(sumVDX * sumVDX + sumVDY * sumVDY + sumVDZ * sumVDZ) 
            > 1e-8 ){
          Print( statusOFS, "Local pseudopotential may not be constructed correctly" );
          statusOFS << "For atom " << a << ", projector " << l << std::endl;
          Print( statusOFS, "Sum dV_a / dx = ", sumVDX );
          Print( statusOFS, "Sum dV_a / dy = ", sumVDY );
          Print( statusOFS, "Sum dV_a / dz = ", sumVDZ );
        }
      }
    }

  }
//    GetTime(time2);
//    statusOFS << "Time for iterNonlocal        = " << iterNonlocal          << "  timeNonlocal        = " << timeNonlocal << std::endl;
//    statusOFS << "Time for CalculateNonlocalPP  = " << time2- time1 << std::endl;


//  GetTime(time1);
  cnt = 0; // the total number of PS used
  MPI_Allreduce( &cntLocal, &cnt, 1, MPI_INT, MPI_SUM, domain_.comm );
  
  Print( statusOFS, "Total number of nonlocal pseudopotential = ",  cnt );

  for (Int a=0; a<numAtom; a++) {

    std::stringstream vStream1;
    std::stringstream vStream2;
    std::stringstream vStream1Temp;
    std::stringstream vStream2Temp;
    int vStream1Size, vStream2Size;

    std::vector<NonlocalPP>& vnlList = pseudo_[a].vnlList;

    serialize( vnlList, vStream1, NO_MASK );

    if (numAtomMpirank[a] == mpirank){
      vStream1Size = Size( vStream1 );
    }

    MPI_Bcast( &vStream1Size, 1, MPI_INT, numAtomMpirank[a], domain_.comm );

    std::vector<char> sstr1;
    sstr1.resize( vStream1Size );

    if (numAtomMpirank[a] == mpirank){
      vStream1.read( &sstr1[0], vStream1Size );
    }

    MPI_Bcast( &sstr1[0], vStream1Size, MPI_BYTE, numAtomMpirank[a], domain_.comm );

    vStream1Temp.write( &sstr1[0], vStream1Size );

    deserialize( vnlList, vStream1Temp, NO_MASK );
  }
//    GetTime(time2);
//    statusOFS << "Time for CalculateNonlocalPP comm = " << time2- time1 << std::endl;
  GetTime( timeEnd );
  
//  statusOFS << "Time for nonlocal pseudopotential " << timeEnd - timeSta  << std::endl;

  // Calculate other atomic related energies and forces, such as self
  // energy, short range repulsion energy and VdW energies.
  
  this->CalculateIonSelfEnergyAndForce( ptable );

  this->CalculateVdwEnergyAndForce();

  Eext_ = 0.0;
  forceext_.Resize( atomList_.size(), DIM );
  SetValue( forceext_, 0.0 );

  return ;
}         // -----  end of method KohnSham::CalculatePseudoPotential ----- 

void KohnSham::CalculateAtomDensity ( PeriodTable &ptable, Fourier &fft ){

  MPI_Barrier(domain_.comm);
  int mpirank;  MPI_Comm_rank(domain_.comm, &mpirank);
  int mpisize;  MPI_Comm_size(domain_.comm, &mpisize);

  if( esdfParam.pseudoType == "HGH" ){
    ErrorHandling("HGH pseudopotential does not yet support the computation of atomic density!");
  }

  Int ntotFine = domain_.NumGridTotalFine();
  Int numAtom = atomList_.size();
  Real vol = domain_.Volume();
  std::vector<DblNumVec> gridpos;
  UniformMeshFine ( domain_, gridpos );

  // The number of electrons for normalization purpose. 
  Int nelec = 0;
  for (Int a=0; a<numAtom; a++) {
    Int atype  = atomList_[a].type;
    if( ptable.ptemap().find(atype) == ptable.ptemap().end() ){
      ErrorHandling( "Cannot find the atom type." );
    }
    nelec = nelec + ptable.Zion(atype);
  }
  // add the extra electron
  nelec = nelec + esdfParam.extraElectron;
//  if( nelec % 2 != 0 ){
//    ErrorHandling( "This is spin-restricted calculation. nelec should be even." );
//  }
#if ( _DEBUGlevel_ >= 0 )
  Print( statusOFS, "Number of electrons in extend element        = ", nelec );
//  Print( statusOFS, "Number of Occupied States                    = ", numOccupiedState_ );
#endif
 
 
  int numAtomBlocksize = numAtom  / mpisize;
  int numAtomLocal = numAtomBlocksize;
  if(mpirank < (numAtom % mpisize)){
    numAtomLocal = numAtomBlocksize + 1;
  }
  IntNumVec numAtomIdx( numAtomLocal );

  if (numAtomBlocksize == 0 ){
    for (Int i = 0; i < numAtomLocal; i++){
      numAtomIdx[i] = mpirank;
    }
  }
  else {
    if ( (numAtom % mpisize) == 0 ){
      for (Int i = 0; i < numAtomLocal; i++){
        numAtomIdx[i] = numAtomBlocksize * mpirank + i;
      }
    }
    else{
      for (Int i = 0; i < numAtomLocal; i++){
        if ( mpirank < (numAtom % mpisize) ){
          numAtomIdx[i] = (numAtomBlocksize + 1) * mpirank + i;
        }
        else{
          numAtomIdx[i] = (numAtomBlocksize + 1) * (numAtom % mpisize) + numAtomBlocksize * (mpirank - (numAtom % mpisize)) + i;
        }
      }
    }
  }

  Real timeSta, timeEnd, timeSta1, timeEnd1;
  GetTime( timeSta );

  //Print( statusOFS, "Computing the atomic density for initialization" );

  // Search for the number of atom types and build a list of atom types
  std::set<Int> atomTypeSet;
  for( Int a = 0; a < numAtom; a++ ){
    atomTypeSet.insert( atomList_[a].type );
  } // for (a)

  // For each atom type, construct the atomic pseudocharge within the
  // cutoff radius starting from the origin in the real space, and
  // construct the structure factor

  // Origin-centered atomDensity in the real space and Fourier space
  DblNumVec atomDensityR( ntotFine );
  CpxNumVec atomDensityG( ntotFine );
  atomDensity_.Resize( ntotFine );
  SetValue( atomDensity_, 0.0 );
  SetValue( atomDensityR, 0.0 );
  SetValue( atomDensityG, Z_ZERO );

  for( std::set<Int>::iterator itype = atomTypeSet.begin(); 
    itype != atomTypeSet.end(); ++itype ){
    Int atype = *itype;
    Atom fakeAtom;
    fakeAtom.type = atype;
    fakeAtom.pos = domain_.posStart;

    ptable.CalculateAtomDensity( fakeAtom, domain_, gridpos, atomDensityR );

    // Compute the structure factor
    CpxNumVec ccvecLocal(ntotFine);
    CpxNumVec ccvec(ntotFine);
    SetValue( ccvecLocal, Z_ZERO );
    SetValue( ccvec, Z_ZERO );

//    Complex* ccvecPtr = ccvec.Data();
    Complex* ikxPtr = fft.ikFine[0].Data();
    Complex* ikyPtr = fft.ikFine[1].Data();
    Complex* ikzPtr = fft.ikFine[2].Data();
    Real xx, yy, zz;
    Complex phase;

    GetTime( timeSta1 );
    for( Int k = 0; k < numAtomLocal; k++ ){
      int a = numAtomIdx[k];
      if( atomList_[a].type == atype ){
        xx = atomList_[a].pos[0];
        yy = atomList_[a].pos[1];
        zz = atomList_[a].pos[2];
        for( Int i = 0; i < ntotFine; i++ ){
          phase = -(ikxPtr[i] * xx + ikyPtr[i] * yy + ikzPtr[i] * zz);
          ccvecLocal[i] += std::exp( phase );
        }
      }
    }

    MPI_Allreduce( ccvecLocal.Data(), ccvec.Data(), ntotFine, 
        MPI_DOUBLE_COMPLEX, MPI_SUM, domain_.comm );
    GetTime( timeEnd1 );
    // Transfer the atomic charge from real space to Fourier space, and
    // multiply with the structure factor
    for(Int i = 0; i < ntotFine; i++){
      fft.inputComplexVecFine[i] = Complex( atomDensityR[i], 0.0 ); 

    }

    FFTWExecute ( fft, fft.forwardPlanFine );

    for( Int i = 0; i < ntotFine; i++ ){
      // Make it smoother: AGGREESIVELY truncate components beyond EcutWavefunction
      if( fft.gkkFine[i] <= esdfParam.ecutWavefunction * 4.0  ){
        atomDensityG[i] += fft.outputComplexVecFine[i] * ccvec[i];
      }
    }
  }

  // Transfer back to the real space and add to atomDensity_ 
  {
    for(Int i = 0; i < ntotFine; i++){
      fft.outputComplexVecFine[i] = atomDensityG[i];
    }
  
    FFTWExecute ( fft, fft.backwardPlanFine );

    for( Int i = 0; i < ntotFine; i++ ){
      atomDensity_[i] = fft.inputComplexVecFine[i].real();
    }
  }

  Real sumrho = 0.0;
  for (Int i=0; i<ntotFine; i++) 
    sumrho += atomDensity_[i]; 
  sumrho *= vol / Real(ntotFine);

  Print( statusOFS, "Sum of atomic density                        = ", 
      sumrho );

  // adjustment should be multiplicative
  Real fac = nelec / sumrho;
  for (Int i=0; i<ntotFine; i++) 
    atomDensity_[i] *= fac; 

  Print( statusOFS, "After adjustment, Sum of atomic density = ", (Real) nelec );

  return ;
}         // -----  end of method KohnSham::CalculateAtomDensity  ----- 

#ifdef _COMPLEX_

void
KohnSham::CalculateDensity ( const Spinor &psi, const DblNumVec &occrate, Real &val, Fourier &fft)
{
  Int ntot  = psi.NumGridTotal();
  Int ncom  = psi.NumComponent();
  Int nocc  = psi.NumState();
  Real vol  = domain_.Volume();

  Int ntotFine  = fft.domain.NumGridTotalFine();

  MPI_Barrier(domain_.comm);
  int mpirank;  MPI_Comm_rank(domain_.comm, &mpirank);
  int mpisize;  MPI_Comm_size(domain_.comm, &mpisize);

  //  IntNumVec& wavefunIdx = psi.WavefunIdx();

  DblNumMat   densityLocal;
  densityLocal.Resize( ntotFine, ncom );   
  SetValue( densityLocal, 0.0 );

  Real fac;

  SetValue( density_, 0.0 );
  for (Int k=0; k<nocc; k++) {
    for (Int j=0; j<ncom; j++) {

      for( Int i = 0; i < ntot; i++ ){
        fft.inputComplexVec(i) = psi.Wavefun(i,j,k);
      }

      FFTWExecute ( fft, fft.forwardPlan );

      // fft Coarse to Fine 

      SetValue( fft.outputComplexVecFine, Z_ZERO );
      for( Int i = 0; i < ntot; i++ ){
        fft.outputComplexVecFine(fft.idxFineGrid(i)) = fft.outputComplexVec(i) * 
          sqrt( double(ntot) / double(ntotFine) );
      } 

      FFTWExecute ( fft, fft.backwardPlanFine );

      // FIXME Factor to be simplified
      fac = numSpin_ * occrate(psi.WavefunIdx(k));
      for( Int i = 0; i < ntotFine; i++ ){
//        densityLocal(i,RHO) +=  (fft.inputComplexVecFine(i) * std::conj(fft.inputComplexVecFine(i))).real()* fac;
        densityLocal(i,RHO) +=  pow( std::abs(fft.inputComplexVecFine(i).real()), 2.0 ) * fac;
      }
    }
  }

  mpi::Allreduce( densityLocal.Data(), density_.Data(), ntotFine, MPI_SUM, domain_.comm );

  val = 0.0; // sum of density
  for (Int i=0; i<ntotFine; i++) {
    val  += density_(i, RHO);
  }

  Real val1 = val;

  // Scale the density
  blas::Scal( ntotFine, (numSpin_ * Real(numOccupiedState_) * Real(ntotFine)) / ( vol * val ), 
      density_.VecData(RHO), 1 );

//  for(Int p= 0; p < ntotFine; p++){
//    Real DEN = density(p,RHO);
//    if(DEN < 0.0 ) statusOFS << p << "Calculated density " << DEN << std::endl;
//  }

  // Double check (can be neglected)
  val = 0.0; // sum of density
  for (Int i=0; i<ntotFine; i++) {
    val  += density_(i, RHO) * vol / ntotFine;

  }

  Real val2 = val;

#if ( _DEBUGlevel_ >= 2 )
  statusOFS << "Raw data, sum of density          = " << val1 << std::endl;
  statusOFS << "Expected sum of density           = " << numSpin_ * numOccupiedState_ << std::endl;
  statusOFS << "Raw data, sum of adjusted density = " << val2 << std::endl;
#endif


  return ;
}         // -----  end of method KohnSham::CalculateDensity  ----- 

#else

void
KohnSham::CalculateDensity ( const Spinor &psi, const DblNumVec &occrate, Real &val, Fourier &fft)
{
  Int ntot  = psi.NumGridTotal();
  Int ncom  = psi.NumComponent();
  Int nocc  = psi.NumState();
  Real vol  = domain_.Volume();

  Int ntotFine  = fft.domain.NumGridTotalFine();

  MPI_Barrier(domain_.comm);
  int mpirank;  MPI_Comm_rank(domain_.comm, &mpirank);
  int mpisize;  MPI_Comm_size(domain_.comm, &mpisize);

  //  IntNumVec& wavefunIdx = psi.WavefunIdx();

  DblNumMat   densityLocal;
  densityLocal.Resize( ntotFine, ncom );   
  SetValue( densityLocal, 0.0 );

  Real fac;

  blas::Copy( ntotFine, density_.Data(), 1, densityold_.Data(), 1 );
  SetValue( density_, 0.0 );
  for (Int k=0; k<nocc; k++) {
    for (Int j=0; j<ncom; j++) {

      for( Int i = 0; i < ntot; i++ ){
        fft.inputComplexVec(i) = Complex( psi.Wavefun(i,j,k), 0.0 ); 
      }

      FFTWExecute ( fft, fft.forwardPlan );

      // fft Coarse to Fine 

      SetValue( fft.outputComplexVecFine, Z_ZERO );
      for( Int i = 0; i < ntot; i++ ){
        fft.outputComplexVecFine(fft.idxFineGrid(i)) = fft.outputComplexVec(i) * 
          sqrt( double(ntot) / double(ntotFine) );
      } 

      FFTWExecute ( fft, fft.backwardPlanFine );

      // FIXME Factor to be simplified
      fac = numSpin_ * occrate(psi.WavefunIdx(k));
      for( Int i = 0; i < ntotFine; i++ ){
        densityLocal(i,RHO) +=  pow( std::abs(fft.inputComplexVecFine(i).real()), 2.0 ) * fac;
      }
    }
  }

  mpi::Allreduce( densityLocal.Data(), density_.Data(), ntotFine, MPI_SUM, domain_.comm );

  val = 0.0; // sum of density
  for (Int i=0; i<ntotFine; i++) {
    val  += density_(i, RHO);
  }

  Real val1 = val;

  // Scale the density
  blas::Scal( ntotFine, (numSpin_ * Real(numOccupiedState_) * Real(ntotFine)) / ( vol * val ), 
      density_.VecData(RHO), 1 );

  // Double check (can be neglected)
  val = 0.0; // sum of density
  for (Int i=0; i<ntotFine; i++) {
    val  += density_(i, RHO) * vol / ntotFine;
  }

  Real val2 = val;

#if ( _DEBUGlevel_ >= 0 )
  statusOFS << "Raw data, sum of density          = " << val1 << std::endl;
  statusOFS << "Expected sum of density           = " << numSpin_ * numOccupiedState_ << std::endl;
  statusOFS << "Raw data, sum of adjusted density = " << val2 << std::endl;
#endif


  return ;
}         // -----  end of method KohnSham::CalculateDensity  ----- 
#endif //COMPLEX


void
KohnSham::CalculateGradDensity ( Fourier& fft )
{
  Int ntotFine  = fft.domain.NumGridTotalFine();
  Real vol  = domain_.Volume();
  Real EPS = 1e-14;
  MPI_Barrier(domain_.comm);
  int mpirank;  MPI_Comm_rank(domain_.comm, &mpirank);
  int mpisize;  MPI_Comm_size(domain_.comm, &mpisize);
  int dmCol = DIM;
  int dmRow = mpisize / dmCol;

  for( Int i = 0; i < ntotFine; i++ ){
    fft.inputComplexVecFine(i) = Complex( density_(i,RHO), 0.0 ); 
  }

  FFTWExecute ( fft, fft.forwardPlanFine );

  CpxNumVec  cpxVec( ntotFine );
  blas::Copy( ntotFine, fft.outputComplexVecFine.Data(), 1,
      cpxVec.Data(), 1 );

  // Compute the derivative of the Density via Fourier

  if(0){

    for( Int d = 0; d < DIM; d++ ){
      CpxNumVec& ik = fft.ikFine[d];

      for( Int i = 0; i < ntotFine; i++ ){
        if( fft.gkkFine(i) <= EPS || fft.gkkFine(i) > esdfParam.ecutWavefunction * 4.0 ){
          fft.outputComplexVecFine(i) = Z_ZERO;
        }
        else{
          fft.outputComplexVecFine(i) = cpxVec(i) * ik(i); 
        }
      }

      FFTWExecute ( fft, fft.backwardPlanFine );

      DblNumMat& gradDensity = gradDensity_[d];
      for( Int i = 0; i < ntotFine; i++ ){
        gradDensity(i, RHO) = fft.inputComplexVecFine(i).real();
      }
    } // for d

  } //if(0)

  if(1){
    
    if( mpisize < DIM ){ // mpisize < 3
      for( Int d = 0; d < DIM; d++ ){
        DblNumMat& gradDensity = gradDensity_[d];
        CpxNumVec& ik = fft.ikFine[d];
        for( Int i = 0; i < ntotFine; i++ ){
          if( fft.gkkFine(i) <= EPS || fft.gkkFine(i) > esdfParam.ecutWavefunction * 4.0 ){
            fft.outputComplexVecFine(i) = Z_ZERO;
          }
          else{
            fft.outputComplexVecFine(i) = cpxVec(i) * ik(i); 
          }
        }

        FFTWExecute ( fft, fft.backwardPlanFine );

        for( Int i = 0; i < ntotFine; i++ ){
          gradDensity(i, RHO) = fft.inputComplexVecFine(i).real();
        }
      } // for d

    } // mpisize < 3
    else { // mpisize > 3
  
      for( Int d = 0; d < DIM; d++ ){
        DblNumMat& gradDensity = gradDensity_[d];
        if ( d == mpirank % dmCol ){ 
          CpxNumVec& ik = fft.ikFine[d];
          for( Int i = 0; i < ntotFine; i++ ){
            if( fft.gkkFine(i) <= EPS || fft.gkkFine(i) > esdfParam.ecutWavefunction * 4.0  ){
              fft.outputComplexVecFine(i) = Z_ZERO;
            }
            else{
              fft.outputComplexVecFine(i) = cpxVec(i) * ik(i); 
            }
          }

          FFTWExecute ( fft, fft.backwardPlanFine );

          for( Int i = 0; i < ntotFine; i++ ){
            gradDensity(i, RHO) = fft.inputComplexVecFine(i).real();
          }
        } // d == mpirank
      } // for d

      for( Int d = 0; d < DIM; d++ ){
        DblNumMat& gradDensity = gradDensity_[d];
        MPI_Bcast( gradDensity.VecData(RHO), ntotFine, MPI_DOUBLE, d, rowComm_ );
      } // for d

    } // mpisize > 3

  } //if(1)
  
  return ;
}         // -----  end of method KohnSham::CalculateGradDensity  ----- 


void
KohnSham::CalculateXC    ( Real &val, Fourier& fft )
{
  Int ntot = domain_.NumGridTotalFine();
  Real vol = domain_.Volume();

  Real EPS = 1e-14;
  MPI_Barrier(domain_.comm);
  int mpirank;  MPI_Comm_rank(domain_.comm, &mpirank);
  int mpisize;  MPI_Comm_size(domain_.comm, &mpisize);
  int dmCol = DIM;
  int dmRow = mpisize / dmCol;
  
  Int ntotBlocksize = ntot / mpisize;
  Int ntotLocal = ntotBlocksize;
  if(mpirank < (ntot % mpisize)){
    ntotLocal = ntotBlocksize + 1;
  } 
  IntNumVec localSize(mpisize);
  IntNumVec localSizeDispls(mpisize);
  SetValue( localSize, 0 );
  SetValue( localSizeDispls, 0 );
  MPI_Allgather( &ntotLocal, 1, MPI_INT, localSize.Data(), 1, MPI_INT, domain_.comm );

  for (Int i = 1; i < mpisize; i++ ){
    localSizeDispls[i] = localSizeDispls[i-1] + localSize[i-1];
  }

  Real fac;
  // Cutoff 
  Real epsRho = 1e-10, epsRhoGGA = 1e-6, epsGrhoGGA = 1e-10;

  Real timeSta, timeEnd;

  Real timeFFT = 0.00;
  Real timeOther = 0.00;


  if( XCId_ == XC_LDA_XC_TETER93 ) 
  {
    GetTime( timeSta );
//    xc_func_set_dens_threshold( &XCFuncType_, epsRho ); 
    xc_lda_exc_vxc( &XCFuncType_, ntot, density_.VecData(RHO), 
        epsxc_.Data(), vxc_.Data() );
    GetTime( timeEnd );

#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for calling the XC kernel in XC LDA is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

    // Modify "bad points"
      for( Int i = 0; i < ntot; i++ ){
       if(0){
        if( density_(i,RHO) < epsRho ){
          epsxc_(i) = 0.0;
          vxc_( i, RHO ) = 0.0;
        }
       }
//xmqin
        epsxc_(i) *= density_(i,RHO) ;
      }
  }//XC_FAMILY_LDA
  else if( XId_ == XC_LDA_X && CId_ == XC_LDA_C_PZ )
  {
    DblNumVec vx_lxc( ntot );
    DblNumVec vc_lxc( ntot );
    DblNumVec epsx( ntot );
    DblNumVec epsc( ntot );

    SetValue( vx_lxc, 0.0 );
    SetValue( vc_lxc, 0.0 );
    SetValue( epsx, 0.0 );
    SetValue( epsc, 0.0 );
  
    xc_func_set_dens_threshold( &XFuncType_, epsRho );
    xc_lda_exc_vxc( &XFuncType_, ntot, density_.Data(),
      epsx.Data(), vx_lxc.Data() );

    xc_func_set_dens_threshold( &CFuncType_, epsRho );
    xc_lda_exc_vxc( &CFuncType_, ntot, density_.Data(),
      epsc.Data(), vc_lxc.Data() );

    blas::Copy( ntot, &vx_lxc[0], 1, vxc_.Data(), 1 );
    blas::Copy( ntot, epsx.Data(), 1, epsxc_.Data(), 1 );
    blas::Axpy( ntot, 1.0, &vc_lxc[0], 1, vxc_.Data(), 1 );
    blas::Axpy( ntot, 1.0, epsc.Data(), 1, epsxc_.Data(), 1 );

//    for( Int i = 0; i < ntot; i++ ){
//      badpoint = ( density_(i,RHO) < epsRho );
//      if( badpoint ){
//        epsxc_(i) = 0.0;
//        for( Int is = 0; is < numDensityComponent_; is++ ){
//          vxc_( i, is ) = 0.0;
//        }
//      }
//    }

    for( Int i = 0; i < ntot; i++ ){
      epsxc_(i) *= density_(i,RHO) ;
    }

  }
  else if( ( !isEXXActive_ ) && ( XId_ == XC_GGA_X_PBE ) && ( CId_ == XC_GGA_C_PBE ) ) {

    DblNumMat gradDensity( ntotLocal, numDensityComponent_ );
    DblNumMat& gradDensity0 = gradDensity_[0];
    DblNumMat& gradDensity1 = gradDensity_[1];
    DblNumMat& gradDensity2 = gradDensity_[2];

    GetTime( timeSta );
    for(Int i = 0; i < ntotLocal; i++){
      Int ii = i + localSizeDispls(mpirank);
      gradDensity(i, RHO) = gradDensity0(ii, RHO) * gradDensity0(ii, RHO)
        + gradDensity1(ii, RHO) * gradDensity1(ii, RHO)
        + gradDensity2(ii, RHO) * gradDensity2(ii, RHO);
    }
    GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 2 )
    statusOFS << "Time for computing gradDensity in XC GGA-PBE is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

    DblNumMat densityTemp;
    densityTemp.Resize( ntotLocal, numDensityComponent_ );

    for( Int i = 0; i < ntotLocal; i++ ){
      densityTemp(i, RHO) = density_(i + localSizeDispls(mpirank), RHO);
    }

    DblNumVec vxc1(ntotLocal);             
    DblNumVec vxc2(ntotLocal);             
    SetValue( vxc1, 0.0 );
    SetValue( vxc2, 0.0 );
    DblNumVec epsx(ntotLocal);
    DblNumVec epsc(ntotLocal);
    SetValue( epsx, 0.0 );
    SetValue( epsc, 0.0 );

    DblNumVec     epsxcTemp( ntotLocal ); 
    SetValue( epsxcTemp, 0.0 );

    if( esdfParam.isUseLIBXC )
    {
      DblNumVec vxc1Temp(ntotLocal);             
      DblNumVec vxc2Temp(ntotLocal);             
      SetValue( vxc1Temp, 0.0 );
      SetValue( vxc2Temp, 0.0 );
      GetTime( timeSta );

      xc_func_set_dens_threshold( &XFuncType_, epsRhoGGA );
      xc_gga_exc_vxc( &XFuncType_, ntotLocal, densityTemp.VecData(RHO), 
         gradDensity.VecData(RHO), epsx.Data(), vxc1.Data(), vxc2.Data() );

      xc_func_set_dens_threshold( &CFuncType_, epsRhoGGA );
      xc_gga_exc_vxc( &CFuncType_, ntotLocal, densityTemp.VecData(RHO), 
         gradDensity.VecData(RHO), epsc.Data(), vxc1Temp.Data(), vxc2Temp.Data() );

      for( Int i = 0; i < ntotLocal; i++ ){
        epsxcTemp(i) = ( epsx(i) + epsc(i) ) * densityTemp(i,RHO) ;
        vxc1(i) += vxc1Temp( i );
        vxc2(i) += vxc2Temp( i );
      }

      GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 2 )
      statusOFS << "Time for calling the XC kernel in GGA is " <<
        timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif


    }
    else
    {
      Real absrho, grho2, rs, vx, ux, vc, uc;
      Real v1gcx, v2gcx, ugcx;
      Real v1gcc, v2gcc, ugcc;

      for( Int i = 0; i < ntotLocal; i++ ){
        absrho = std::abs( densityTemp(i,RHO) );
        grho2 = gradDensity(i,RHO);

        if( absrho > epsRhoGGA ){
          rs = std::pow(3.0 / 4.0 / PI / absrho, 1.0 / 3.0);
          VExchange_sla(rs, ux, vx);
          VCorrelation_pw(rs, uc, vc);
          
          epsx(i) = epsx(i) + ux;
          epsc(i) = epsc(i) + uc;
          vxc1(i) = vxc1(i) + vx + vc;     
        }

        if( absrho > epsRhoGGA && grho2 > epsGrhoGGA ){
          VGCExchange_pbx(absrho, grho2, ugcx, v1gcx, v2gcx);
          VGCCorrelation_pbc(absrho, grho2, ugcc, v1gcc, v2gcc);
          
          epsx(i) = epsx(i) + ugcx;
          epsc(i) = epsc(i) + ugcc;
          vxc1(i) = vxc1(i) + v1gcx + v1gcc;
          vxc2(i) = vxc2(i) + 0.5 * v2gcx + 0.5 * v2gcc;
        }
      }

      for( Int i = 0; i < ntotLocal; i++ ){
        epsxcTemp(i) = ( epsx(i) + epsc(i) ) * densityTemp(i,RHO) ;
      }
    } //uselibxc

    // Modify "bad points"
    if(0){
      for( Int i = 0; i < ntotLocal; i++ ){
        if( ( densityTemp(i,RHO) < epsRhoGGA ) || 
            ( gradDensity(i,RHO) < epsGrhoGGA )  ){
          epsxcTemp(i) = 0.0;
          vxc1(i) = 0.0;
          vxc2(i) = 0.0;
        }
//xmqin Now espxcTemp = espxcTemp* density
      }
    }

    DblNumVec     vxcTemp( ntot );
    DblNumVec     vxc2Temp2( ntot );
    SetValue( epsxc_, 0.0 );
    SetValue( vxcTemp, 0.0 );
    SetValue( vxc2Temp2, 0.0 );

    GetTime( timeSta );

    MPI_Allgatherv( epsxcTemp.Data(), ntotLocal, MPI_DOUBLE, epsxc_.Data(), 
        localSize.Data(), localSizeDispls.Data(), MPI_DOUBLE, domain_.comm );
    MPI_Allgatherv( vxc1.Data(), ntotLocal, MPI_DOUBLE, vxcTemp.Data(), 
        localSize.Data(), localSizeDispls.Data(), MPI_DOUBLE, domain_.comm );
    MPI_Allgatherv( vxc2.Data(), ntotLocal, MPI_DOUBLE, vxc2Temp2.Data(), 
        localSize.Data(), localSizeDispls.Data(), MPI_DOUBLE, domain_.comm );

    GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 2 )
    statusOFS << "Time for MPI_Allgatherv in XC GGA-PBE is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif


    for( Int i = 0; i < ntot; i++ ){
      vxc_( i, RHO ) = vxcTemp(i);
    }
    Int d;
    if( mpisize < DIM ){ // mpisize < 3

      for( Int d = 0; d < DIM; d++ ){
        DblNumMat& gradDensityd = gradDensity_[d];
        for(Int i = 0; i < ntot; i++){
          fft.inputComplexVecFine(i) = Complex( gradDensityd( i, RHO ) * 2.0 * vxc2Temp2(i), 0.0 ); 
        }

        FFTWExecute ( fft, fft.forwardPlanFine );

        CpxNumVec& ik = fft.ikFine[d];

        for( Int i = 0; i < ntot; i++ ){
          if( fft.gkkFine(i) <= EPS || fft.gkkFine(i) > esdfParam.ecutWavefunction * 4.0 ){
            fft.outputComplexVecFine(i) = Z_ZERO;
          }
          else{
            fft.outputComplexVecFine(i) *= ik(i);
          }
        }

        FFTWExecute ( fft, fft.backwardPlanFine );

        for( Int i = 0; i < ntot; i++ ){
          vxc_( i, RHO ) -= fft.inputComplexVecFine(i).real();
        }

      } // for d

    } // mpisize < 3
    else { // mpisize > 3

      std::vector<DblNumVec>      vxcTemp3d;
      vxcTemp3d.resize( DIM );
      for( Int d = 0; d < DIM; d++ ){
        vxcTemp3d[d].Resize(ntot);
        SetValue (vxcTemp3d[d], 0.0);
      }

      for( Int d = 0; d < DIM; d++ ){
        DblNumMat& gradDensityd = gradDensity_[d];
        DblNumVec& vxcTemp3 = vxcTemp3d[d]; 
        if ( d == mpirank % dmCol ){ 
          for(Int i = 0; i < ntot; i++){
            fft.inputComplexVecFine(i) = Complex( gradDensityd( i, RHO ) * 2.0 * vxc2Temp2(i), 0.0 ); 
          }

          FFTWExecute ( fft, fft.forwardPlanFine );

          CpxNumVec& ik = fft.ikFine[d];

          for( Int i = 0; i < ntot; i++ ){
            if( fft.gkkFine(i) <= EPS || fft.gkkFine(i) > esdfParam.ecutWavefunction * 4.0 ){
              fft.outputComplexVecFine(i) = Z_ZERO;
            }
            else{
              fft.outputComplexVecFine(i) *= ik(i);
            }
          }

          FFTWExecute ( fft, fft.backwardPlanFine );

          for( Int i = 0; i < ntot; i++ ){
            vxcTemp3(i) = fft.inputComplexVecFine(i).real();
          }
        } // d == mpirank
      } // for d

      for( Int d = 0; d < DIM; d++ ){
        DblNumVec& vxcTemp3 = vxcTemp3d[d]; 
        MPI_Bcast( vxcTemp3.Data(), ntot, MPI_DOUBLE, d, rowComm_ );
        for( Int i = 0; i < ntot; i++ ){
          vxc_( i, RHO ) -= vxcTemp3(i);
        }
      } // for d

    } // mpisize > 3



  } // XC_FAMILY_GGA
  else if( XCId_ == XC_HYB_GGA_XC_HSE06 ){
    // FIXME Condensify with the previous

    DblNumMat gradDensity;
    gradDensity.Resize( ntotLocal, numDensityComponent_ );
    SetValue( gradDensity, 0.0 );
    DblNumMat& gradDensity0 = gradDensity_[0];
    DblNumMat& gradDensity1 = gradDensity_[1];
    DblNumMat& gradDensity2 = gradDensity_[2];

    GetTime( timeSta );
    for(Int i = 0; i < ntotLocal; i++){
      Int ii = i + localSizeDispls(mpirank);
      gradDensity(i, RHO) = gradDensity0(ii, RHO) * gradDensity0(ii, RHO)
        + gradDensity1(ii, RHO) * gradDensity1(ii, RHO)
        + gradDensity2(ii, RHO) * gradDensity2(ii, RHO);
    }
    GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 1 )
    statusOFS << " " << std::endl;
    statusOFS << "Time for computing gradDensity in XC HSE06 is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

    DblNumMat densityTemp;
    densityTemp.Resize( ntotLocal, numDensityComponent_ );

    for( Int i = 0; i < ntotLocal; i++ ){
      densityTemp(i, RHO) = density_(i + localSizeDispls(mpirank), RHO);
    }

    DblNumVec epsxcTemp(ntotLocal);
    DblNumVec vxc1Temp(ntotLocal);
    DblNumVec vxc2Temp(ntotLocal);
    SetValue( epsxcTemp, 0.0 );
    SetValue( vxc1Temp, 0.0 );
    SetValue( vxc2Temp, 0.0 );

    DblNumVec epsx( ntotLocal );
    DblNumVec epsc( ntotLocal );

    SetValue( epsx, 0.0 );
    SetValue( epsc, 0.0 );

    GetTime( timeSta );

    if( !esdfParam.isUseLIBXC ){
//      statusOFS << " call QE-XC for PBE XC - external Short X " << std::endl;
      Real absrho, grho2, rs, vx, ux, vc, uc;
      Real v1gcx, v2gcx, ugcx;
      Real v1gcc, v2gcc, ugcc;

      for( Int i = 0; i < ntotLocal; i++ ){
        absrho = std::abs( densityTemp(i,RHO) );
        grho2 = gradDensity(i,RHO);

        if( absrho > epsRhoGGA ){
          rs = std::pow(3.0 / 4.0 / PI / absrho, 1.0 / 3.0);
          VExchange_sla(rs, ux, vx);
          VCorrelation_pw(rs, uc, vc);
          
          epsx(i) = epsx(i) + ux;
          epsc(i) = epsc(i) + uc;
          vxc1Temp(i) = vxc1Temp(i) + vx + vc;     
        }

        if( absrho > epsRhoGGA && grho2 > epsGrhoGGA ){
          VGCExchange_pbx(absrho, grho2, ugcx, v1gcx, v2gcx);
          VGCCorrelation_pbc(absrho, grho2, ugcc, v1gcc, v2gcc);
          
          epsx(i) = epsx(i) + ugcx;
          epsc(i) = epsc(i) + ugcc;
          vxc1Temp(i) = vxc1Temp(i) + v1gcx + v1gcc;
          vxc2Temp(i) = vxc2Temp(i) + 0.5 * v2gcx + 0.5 * v2gcc;
        }
      }

      for( Int i = 0; i < ntotLocal; i++ ){
          epsx(i) *=  densityTemp(i,RHO);
          epsc(i) *=  densityTemp(i,RHO);
      }

    }
    else
    {
      statusOFS << " call LIBXC PBE for HSE06 XC " << std::endl;

      DblNumVec vxc1(ntotLocal);
      DblNumVec vxc2(ntotLocal);
      SetValue( vxc1, 0.0 );
      SetValue( vxc2, 0.0 );

      xc_func_set_dens_threshold( &XFuncType_, epsRhoGGA );
      xc_gga_exc_vxc( &XFuncType_, ntotLocal, densityTemp.VecData(RHO),
         gradDensity.VecData(RHO), epsx.Data(), vxc1.Data(), vxc2.Data() );

      xc_func_set_dens_threshold( &CFuncType_, epsRhoGGA );
      xc_gga_exc_vxc( &CFuncType_, ntotLocal, densityTemp.VecData(RHO),
         gradDensity.VecData(RHO), epsc.Data(), vxc1Temp.Data(), vxc2Temp.Data() );

      for( Int i = 0; i < ntotLocal; i++ ){
          epsx(i) *=  densityTemp(i,RHO);
          epsc(i) *=  densityTemp(i,RHO);
          vxc1Temp(i) += vxc1( i );
          vxc2Temp(i) += vxc2( i );
      }
    } // libxc

    Real omega = 0.106;
    Real frac = 0.25;
    Real v1xsr, v2xsr, epxsr;
    for( Int i = 0; i < ntotLocal; i++ ){
       if( ( densityTemp(i,RHO) > epsRhoGGA ) && ( gradDensity(i,RHO) > epsGrhoGGA ) ){
          pbexsr( densityTemp(i,RHO), gradDensity(i,RHO), omega, epxsr, v1xsr, v2xsr );
          vxc1Temp(i)   -= frac * v1xsr;
          vxc2Temp(i)   -= frac * v2xsr / 2.0;
          epsx(i)  -= frac * epxsr;
       }
    }

    blas::Copy( ntotLocal, epsx.Data(), 1, epsxcTemp.Data(), 1 );
    blas::Axpy( ntotLocal, 1.0, epsc.Data(), 1, epsxcTemp.Data(), 1 );

    if(0){
      for( Int i = 0; i < ntotLocal; i++ ){
        if( ( densityTemp(i,RHO) < epsRhoGGA ) || ( gradDensity(i,RHO) < epsGrhoGGA )  ){
          epsxcTemp(i) = 0.0;
          vxc1Temp(i) = 0.0;
          vxc2Temp(i) = 0.0;
        }
      }
    }

    GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 1 )
    statusOFS << "Time for computing XC kernel XC HSE06 is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

    DblNumVec vxc1(ntot);             
    DblNumVec vxc2(ntot);             

    SetValue( vxc1, 0.0 );
    SetValue( vxc2, 0.0 );
    SetValue( epsxc_, 0.0 );

    GetTime( timeSta );

    MPI_Allgatherv( epsxcTemp.Data(), ntotLocal, MPI_DOUBLE, epsxc_.Data(), 
        localSize.Data(), localSizeDispls.Data(), MPI_DOUBLE, domain_.comm );
    MPI_Allgatherv( vxc1Temp.Data(), ntotLocal, MPI_DOUBLE, vxc1.Data(), 
        localSize.Data(), localSizeDispls.Data(), MPI_DOUBLE, domain_.comm );
    MPI_Allgatherv( vxc2Temp.Data(), ntotLocal, MPI_DOUBLE, vxc2.Data(), 
        localSize.Data(), localSizeDispls.Data(), MPI_DOUBLE, domain_.comm );

    GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 1 )
    statusOFS << "Time for MPI_Allgatherv in XC HSE06 is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

    for( Int i = 0; i < ntot; i++ ){
      vxc_( i, RHO ) = vxc1(i);
    }

    if( mpisize < DIM ){ // mpisize < 3

      for( Int d = 0; d < DIM; d++ ){
        DblNumMat& gradDensityd = gradDensity_[d];
        for(Int i = 0; i < ntot; i++){
          fft.inputComplexVecFine(i) = Complex( gradDensityd( i, RHO ) * 2.0 * vxc2(i), 0.0 ); 
        }

        FFTWExecute ( fft, fft.forwardPlanFine );

        CpxNumVec& ik = fft.ikFine[d];

        for( Int i = 0; i < ntot; i++ ){
          if( fft.gkkFine(i) <= EPS || fft.gkkFine(i) > esdfParam.ecutWavefunction * 4.0 ){
            fft.outputComplexVecFine(i) = Z_ZERO;
          }
          else{
            fft.outputComplexVecFine(i) *= ik(i);
          }
        }

        FFTWExecute ( fft, fft.backwardPlanFine );

        GetTime( timeSta );
        for( Int i = 0; i < ntot; i++ ){
          vxc_( i, RHO ) -= fft.inputComplexVecFine(i).real();
        }
        GetTime( timeEnd );
        timeOther = timeOther + ( timeEnd - timeSta );

      } // for d
    
    } // mpisize < 3
    else { // mpisize > 3

      std::vector<DblNumVec>      vxcTemp3d;
      vxcTemp3d.resize( DIM );
      for( Int d = 0; d < DIM; d++ ){
        vxcTemp3d[d].Resize(ntot);
        SetValue (vxcTemp3d[d], 0.0);
      }

      for( Int d = 0; d < DIM; d++ ){
        DblNumMat& gradDensityd = gradDensity_[d];
        DblNumVec& vxcTemp3 = vxcTemp3d[d]; 
        if ( d == mpirank % dmCol ){ 
          for(Int i = 0; i < ntot; i++){
            fft.inputComplexVecFine(i) = Complex( gradDensityd( i, RHO ) * 2.0 * vxc2(i), 0.0 ); 
          }

          FFTWExecute ( fft, fft.forwardPlanFine );

          CpxNumVec& ik = fft.ikFine[d];

          for( Int i = 0; i < ntot; i++ ){
            if( fft.gkkFine(i) <= EPS || fft.gkkFine(i) > esdfParam.ecutWavefunction * 4.0 ){
              fft.outputComplexVecFine(i) = Z_ZERO;
            }
            else{
              fft.outputComplexVecFine(i) *= ik(i);
            }
          }

          FFTWExecute ( fft, fft.backwardPlanFine );

          for( Int i = 0; i < ntot; i++ ){
            vxcTemp3(i) = fft.inputComplexVecFine(i).real();
          }

        } // d == mpirank
      } // for d

      for( Int d = 0; d < DIM; d++ ){
        DblNumVec& vxcTemp3 = vxcTemp3d[d]; 
        MPI_Bcast( vxcTemp3.Data(), ntot, MPI_DOUBLE, d, rowComm_ );
        for( Int i = 0; i < ntot; i++ ){
          vxc_( i, RHO ) -= vxcTemp3(i);
        }
      } // for d

    } // mpisize > 3

  } // XC_FAMILY Hybrid
  else if( XCId_ == XC_HYB_GGA_XC_PBEH ){
    // FIXME Condensify with the previous

    DblNumMat gradDensity;
    gradDensity.Resize( ntotLocal, numDensityComponent_ );
    SetValue( gradDensity, 0.0 );
    DblNumMat& gradDensity0 = gradDensity_[0];
    DblNumMat& gradDensity1 = gradDensity_[1];
    DblNumMat& gradDensity2 = gradDensity_[2];

    GetTime( timeSta );
    for(Int i = 0; i < ntotLocal; i++){
      Int ii = i + localSizeDispls(mpirank);
      gradDensity(i, RHO) = gradDensity0(ii, RHO) * gradDensity0(ii, RHO)
        + gradDensity1(ii, RHO) * gradDensity1(ii, RHO)
        + gradDensity2(ii, RHO) * gradDensity2(ii, RHO);
    }
    GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 1 )
    statusOFS << " " << std::endl;
    statusOFS << "Time for computing gradDensity in XC PBE0 is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

    DblNumMat densityTemp;
    densityTemp.Resize( ntotLocal, numDensityComponent_ );

    for( Int i = 0; i < ntotLocal; i++ ){
      densityTemp(i, RHO) = density_(i + localSizeDispls(mpirank), RHO);
    }

    DblNumVec vxc1Temp(ntotLocal);
    DblNumVec vxc2Temp(ntotLocal);
    DblNumVec epsxcTemp(ntotLocal);

    SetValue( vxc1Temp, 0.0 );
    SetValue( vxc2Temp, 0.0 );
    SetValue( epsxcTemp, 0.0 );


    if( !esdfParam.isUseLIBXC ){

      statusOFS << " call QE-XC for PBE XC - external Short X " << std::endl;

      DblNumVec epsx(ntotLocal);
      DblNumVec epsc(ntotLocal);
      SetValue( epsx, 0.0 );
      SetValue( epsc, 0.0 );

      Real absrho, grho2, rs, vx, ux, vc, uc;
      Real v1gcx, v2gcx, ugcx;
      Real v1gcc, v2gcc, ugcc;

      for( Int i = 0; i < ntotLocal; i++ ){
        absrho = std::abs( densityTemp(i,RHO) );
        grho2 = gradDensity(i,RHO);

        if( absrho > epsRhoGGA ){
          rs = std::pow(3.0 / 4.0 / PI / absrho, 1.0 / 3.0);
          VExchange_sla(rs, ux, vx);
          VCorrelation_pw(rs, uc, vc);

          epsx(i) = epsx(i) + ux;
          epsc(i) = epsc(i) + uc;
          vxc1Temp(i) = vxc1Temp(i) + 0.75 * vx + vc;     
        }

        if( absrho > epsRhoGGA & grho2 > epsGrhoGGA ){
          VGCExchange_pbx(absrho, grho2, ugcx, v1gcx, v2gcx);
          VGCCorrelation_pbc(absrho, grho2, ugcc, v1gcc, v2gcc);
          
          epsx(i) = epsx(i) + ugcx;
          epsc(i) = epsc(i) + ugcc;
          vxc1Temp(i) = vxc1Temp(i) + 0.75 * v1gcx + v1gcc;
          vxc2Temp(i) = vxc2Temp(i) + 0.75 * 0.5 * v2gcx + 0.5 * v2gcc;
        }
      }

      for( Int i = 0; i < ntotLocal; i++ ){
        if(0){
          if( densityTemp(i,RHO) < epsRhoGGA || gradDensity(i,RHO) < epsGrhoGGA ){
            epsxcTemp(i) = 0.0;
            vxc1Temp(i) = 0.0;
            vxc2Temp(i) = 0.0;
          }
        }
        epsxcTemp(i) = ( 0.75 * epsx(i) + epsc(i) ) * densityTemp(i,RHO) ; 
//        statusOFS << " I " <<i << "  epsxcTemp(i) "<<  epsxcTemp(i) << " density "<< densityTemp(i,RHO) <<std::endl;
      }
    }
    else
    {
      GetTime( timeSta );
      xc_gga_exc_vxc( &XCFuncType_, ntotLocal, densityTemp.VecData(RHO), 
        gradDensity.VecData(RHO), epsxcTemp.Data(), vxc1Temp.Data(), vxc2Temp.Data() );
      GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 1 )
      statusOFS << "Time for computing xc_gga_exc_vxc in XC PBE0 is " <<
        timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
  // Modify "bad points"
      for( Int i = 0; i < ntotLocal; i++ ){
        if(0){
          if( densityTemp(i,RHO) < epsRhoGGA || gradDensity(i,RHO) < epsGrhoGGA ){
            epsxcTemp(i) = 0.0;
            vxc1Temp(i) = 0.0;
            vxc2Temp(i) = 0.0;
          }
        }
/// xmqin
        epsxcTemp(i) *= densityTemp(i, RHO) ;
//        statusOFS << " I " <<i << "  epsxcTemp(i) "<<  epsxcTemp(i) << " density "<< densityTemp(i,RHO) <<std::endl;
//      statusOFS << " I " <<i << "  epsxcTemp(i) "<<  epsxcTemp(i) <<std::endl;
      }
    }

    DblNumVec vxc1(ntot);
    DblNumVec vxc2(ntot);

    SetValue( vxc1, 0.0 );
    SetValue( vxc2, 0.0 );
    SetValue( epsxc_, 0.0 );

    GetTime( timeSta );

    MPI_Allgatherv( epsxcTemp.Data(), ntotLocal, MPI_DOUBLE, epsxc_.Data(), 
        localSize.Data(), localSizeDispls.Data(), MPI_DOUBLE, domain_.comm );
    MPI_Allgatherv( vxc1Temp.Data(), ntotLocal, MPI_DOUBLE, vxc1.Data(), 
        localSize.Data(), localSizeDispls.Data(), MPI_DOUBLE, domain_.comm );
    MPI_Allgatherv( vxc2Temp.Data(), ntotLocal, MPI_DOUBLE, vxc2.Data(), 
        localSize.Data(), localSizeDispls.Data(), MPI_DOUBLE, domain_.comm );

    GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 1 )
    statusOFS << "Time for MPI_Allgatherv in XC PBE0 is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

    for( Int i = 0; i < ntot; i++ ){
      vxc_( i, RHO ) = vxc1(i);
    }

    if( mpisize < DIM ){ // mpisize < 3

      for( Int d = 0; d < DIM; d++ ){
        DblNumMat& gradDensityd = gradDensity_[d];
        for(Int i = 0; i < ntot; i++){
          fft.inputComplexVecFine(i) = Complex( gradDensityd( i, RHO ) * 2.0 * vxc2(i), 0.0 ); 
        }

        FFTWExecute ( fft, fft.forwardPlanFine );

        CpxNumVec& ik = fft.ikFine[d];

        for( Int i = 0; i < ntot; i++ ){
          if( fft.gkkFine(i) <= EPS || fft.gkkFine(i) > esdfParam.ecutWavefunction * 4.0 ){
            fft.outputComplexVecFine(i) = Z_ZERO;
          }
          else{
            fft.outputComplexVecFine(i) *= ik(i);
          }
        }

        FFTWExecute ( fft, fft.backwardPlanFine );

        GetTime( timeSta );
        for( Int i = 0; i < ntot; i++ ){
          vxc_( i, RHO ) -= fft.inputComplexVecFine(i).real();
        }
        GetTime( timeEnd );
        timeOther = timeOther + ( timeEnd - timeSta );

      } // for d
    
    } // mpisize < 3
    else { // mpisize > 3
      
      std::vector<DblNumVec>      vxcTemp3d;
      vxcTemp3d.resize( DIM );
      for( Int d = 0; d < DIM; d++ ){
        vxcTemp3d[d].Resize(ntot);
        SetValue (vxcTemp3d[d], 0.0);
      }

      for( Int d = 0; d < DIM; d++ ){
        DblNumMat& gradDensityd = gradDensity_[d];
        DblNumVec& vxcTemp3 = vxcTemp3d[d]; 
        if ( d == mpirank % dmCol ){ 
          for(Int i = 0; i < ntot; i++){
            fft.inputComplexVecFine(i) = Complex( gradDensityd( i, RHO ) * 2.0 * vxc2(i), 0.0 ); 
          }

          FFTWExecute ( fft, fft.forwardPlanFine );

          CpxNumVec& ik = fft.ikFine[d];

          for( Int i = 0; i < ntot; i++ ){
            if( fft.gkkFine(i) <= EPS || fft.gkkFine(i) > esdfParam.ecutWavefunction * 4.0 ){
              fft.outputComplexVecFine(i) = Z_ZERO;
            }
            else{
              fft.outputComplexVecFine(i) *= ik(i);
            }
          }

          FFTWExecute ( fft, fft.backwardPlanFine );

          for( Int i = 0; i < ntot; i++ ){
            vxcTemp3(i) = fft.inputComplexVecFine(i).real();
          }

        } // d == mpirank
      } // for d

      for( Int d = 0; d < DIM; d++ ){
        DblNumVec& vxcTemp3 = vxcTemp3d[d]; 
        MPI_Bcast( vxcTemp3.Data(), ntot, MPI_DOUBLE, d, rowComm_ );
        for( Int i = 0; i < ntot; i++ ){
          vxc_( i, RHO ) -= vxcTemp3(i);
        }
      } // for d

    } // mpisize > 3

  } // XC_FAMILY Hybrid
    else
      ErrorHandling( "Unsupported XC family!" );

  // Compute the total exchange-correlation energy
  val = 0.0;
  GetTime( timeSta );
  for(Int i = 0; i < ntot; i++){
//    val += density_(i, RHO) * epsxc_(i) * vol / (Real) ntot;
// xmqin
    val += epsxc_(i) * vol / (Real) ntot;    

  }
//  statusOFS << " call LIBXC for Exc " << val << std::endl;
  GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 2 )
  statusOFS << " " << std::endl;
  statusOFS << "Time for computing total xc energy in XC is " <<
    timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

  return ;
}         // -----  end of method KohnSham::CalculateXC  ----- 

void KohnSham::CalculateHartree( Fourier& fft ) {
  if( !fft.isInitialized ){
    ErrorHandling("Fourier is not prepared.");
  }

  Int ntot = domain_.NumGridTotalFine();
  if( fft.domain.NumGridTotalFine() != ntot ){
    ErrorHandling( "Grid size does not match!" );
  }

  // The contribution of the pseudoCharge is subtracted. So the Poisson
  // equation is well defined for neutral system.
  for( Int i = 0; i < ntot; i++ ){
    fft.inputComplexVecFine(i) = Complex( 
        density_(i,RHO) - pseudoCharge_(i), 0.0 );
  }

  FFTWExecute ( fft, fft.forwardPlanFine );

  Real EPS = 1e-14;
  for( Int i = 0; i < ntot; i++ ){
    if( fft.gkkFine(i) <= EPS || fft.gkkFine(i) > esdfParam.ecutWavefunction * 4.0 ){
      fft.outputComplexVecFine(i) = Z_ZERO;
    }
    else{
      // NOTE: gkk already contains the factor 1/2.
      fft.outputComplexVecFine(i) *= 2.0 * PI / fft.gkkFine(i);
    }
  }

  FFTWExecute ( fft, fft.backwardPlanFine );

  for( Int i = 0; i < ntot; i++ ){
    vhart_(i) = fft.inputComplexVecFine(i).real();
    //statusOFS << " i " << i << "  " << vhart_(i) <<std::endl;
  }

  return; 
}  // -----  end of method KohnSham::CalculateHartree ----- 


void
KohnSham::CalculateVtot    ( DblNumVec& vtot )
{
  Int ntot = domain_.NumGridTotalFine();
  if( esdfParam.isUseVLocal == false ){
    for (int i=0; i<ntot; i++) {
      vtot(i) = vext_(i) + vhart_(i) + vxc_(i, RHO);
    }
  }
  else
  {
    for (int i=0; i<ntot; i++) {
      vtot(i) = vext_(i) + vLocalSR_(i) + vhart_(i) + vxc_(i, RHO);
    }
  }

  return ;
}         // -----  end of method KohnSham::CalculateVtot  ----- 


//void
//KohnSham::CalculateForce    ( Spinor& psi, Fourier& fft  )
//{
//
//  //  Int ntot      = fft.numGridTotal;
//  Int ntot      = fft.domain.NumGridTotalFine();
//  Int numAtom   = atomList_.size();
//
//  DblNumMat  force( numAtom, DIM );
//  SetValue( force, 0.0 );
//  DblNumMat  forceLocal( numAtom, DIM );
//  SetValue( forceLocal, 0.0 );
//
//  // *********************************************************************
//  // Compute the derivative of the Hartree potential for computing the 
//  // local pseudopotential contribution to the Hellmann-Feynman force
//  // *********************************************************************
//  DblNumVec               vhart;
//  std::vector<DblNumVec>  vhartDrv(DIM);
//
//  DblNumVec  tempVec(ntot);
//  SetValue( tempVec, 0.0 );
//
//  // tempVec = density_ - pseudoCharge_
//  // FIXME No density
//  blas::Copy( ntot, density_.VecData(0), 1, tempVec.Data(), 1 );
//  blas::Axpy( ntot, -1.0, pseudoCharge_.Data(),1,
//      tempVec.Data(), 1 );
//
//  // cpxVec saves the Fourier transform of 
//  // density_ - pseudoCharge_ 
//  CpxNumVec  cpxVec( tempVec.Size() );
//
//  for( Int i = 0; i < ntot; i++ ){
//    fft.inputComplexVecFine(i) = Complex( 
//        tempVec(i), 0.0 );
//  }
//
//  FFTWExecute ( fft, fft.forwardPlanFine );
//
//  blas::Copy( ntot, fft.outputComplexVecFine.Data(), 1,
//      cpxVec.Data(), 1 );
//
//  // Compute the derivative of the Hartree potential via Fourier
//  // transform 
//  {
//    for( Int i = 0; i < ntot; i++ ){
//      if( fft.gkkFine(i) == 0 ){
//        fft.outputComplexVecFine(i) = Z_ZERO;
//      }
//      else{
//        // NOTE: gkk already contains the factor 1/2.
//        fft.outputComplexVecFine(i) = cpxVec(i) *
//          2.0 * PI / fft.gkkFine(i);
//      }
//    }
//
//    FFTWExecute ( fft, fft.backwardPlanFine );
//
//    vhart.Resize( ntot );
//
//    for( Int i = 0; i < ntot; i++ ){
//      vhart(i) = fft.inputComplexVecFine(i).real();
//    }
//  }
//
//  for( Int d = 0; d < DIM; d++ ){
//    CpxNumVec& ik = fft.ikFine[d];
//    for( Int i = 0; i < ntot; i++ ){
//      if( fft.gkkFine(i) == 0 ){
//        fft.outputComplexVecFine(i) = Z_ZERO;
//      }
//      else{
//        // NOTE: gkk already contains the factor 1/2.
//        fft.outputComplexVecFine(i) = cpxVec(i) *
//          2.0 * PI / fft.gkkFine(i) * ik(i);
//      }
//    }
//
//    FFTWExecute ( fft, fft.backwardPlanFine );
//
//    // vhartDrv saves the derivative of the Hartree potential
//    vhartDrv[d].Resize( ntot );
//
//    for( Int i = 0; i < ntot; i++ ){
//      vhartDrv[d](i) = fft.inputComplexVecFine(i).real();
//    }
//
//  } // for (d)
//
//
//  // *********************************************************************
//  // Compute the force from local pseudopotential
//  // *********************************************************************
//  // Method 1: Using the derivative of the pseudopotential
//  if(0){
//    for (Int a=0; a<numAtom; a++) {
//      PseudoPot& pp = pseudo_[a];
//      SparseVec& sp = pp.pseudoCharge;
//      IntNumVec& idx = sp.first;
//      DblNumMat& val = sp.second;
//
//      Real wgt = domain_.Volume() / domain_.NumGridTotalFine();
//      Real resX = 0.0;
//      Real resY = 0.0;
//      Real resZ = 0.0;
//      for( Int l = 0; l < idx.m(); l++ ){
//        resX -= val(l, DX) * vhart[idx(l)] * wgt;
//        resY -= val(l, DY) * vhart[idx(l)] * wgt;
//        resZ -= val(l, DZ) * vhart[idx(l)] * wgt;
//      }
//      force( a, 0 ) += resX;
//      force( a, 1 ) += resY;
//      force( a, 2 ) += resZ;
//
//    } // for (a)
//  }
//
//  // Method 2: Using integration by parts
//  // This formulation must be used when ONCV pseudopotential is used.
//  if(1)
//  {
//    for (Int a=0; a<numAtom; a++) {
//      PseudoPot& pp = pseudo_[a];
//      SparseVec& sp = pp.pseudoCharge;
//      IntNumVec& idx = sp.first;
//      DblNumMat& val = sp.second;
//
//      Real wgt = domain_.Volume() / domain_.NumGridTotalFine();
//      Real resX = 0.0;
//      Real resY = 0.0;
//      Real resZ = 0.0;
//      for( Int l = 0; l < idx.m(); l++ ){
//        resX += val(l, VAL) * vhartDrv[0][idx(l)] * wgt;
//        resY += val(l, VAL) * vhartDrv[1][idx(l)] * wgt;
//        resZ += val(l, VAL) * vhartDrv[2][idx(l)] * wgt;
//      }
//      force( a, 0 ) += resX;
//      force( a, 1 ) += resY;
//      force( a, 2 ) += resZ;
//
//    } // for (a)
//  }
//
//  // Method 3: Evaluating the derivative by expliciting computing the
//  // derivative of the local pseudopotential
//  if(0){
//    Int ntotFine = domain_.NumGridTotalFine();
//
//    DblNumVec totalCharge(ntotFine);
//    blas::Copy( ntot, density_.VecData(0), 1, totalCharge.Data(), 1 );
//    blas::Axpy( ntot, -1.0, pseudoCharge_.Data(),1,
//        totalCharge.Data(), 1 );
//
//    std::vector<DblNumVec> vlocDrv(DIM);
//    for( Int d = 0; d < DIM; d++ ){
//      vlocDrv[d].Resize(ntotFine);
//    }
//    CpxNumVec cpxTempVec(ntotFine);
//
//    for (Int a=0; a<numAtom; a++) {
//      PseudoPot& pp = pseudo_[a];
//      SparseVec& sp = pp.pseudoCharge;
//      IntNumVec& idx = sp.first;
//      DblNumMat& val = sp.second;
//
//      // Solve the Poisson equation for the pseudo-charge of atom a
//      SetValue( fft.inputComplexVecFine, Z_ZERO );
//
//      for( Int k = 0; k < idx.m(); k++ ){
//        fft.inputComplexVecFine(idx(k)) = Complex( val(k,VAL), 0.0 );
//      }
//
//      FFTWExecute ( fft, fft.forwardPlanFine );
//
//      // Save the vector for multiple differentiation
//      blas::Copy( ntot, fft.outputComplexVecFine.Data(), 1,
//          cpxTempVec.Data(), 1 );
//
//      for( Int d = 0; d < DIM; d++ ){
//        CpxNumVec& ik = fft.ikFine[d];
//
//        for( Int i = 0; i < ntot; i++ ){
//          if( fft.gkkFine(i) == 0 ){
//            fft.outputComplexVecFine(i) = Z_ZERO;
//          }
//          else{
//            // NOTE: gkk already contains the factor 1/2.
//            fft.outputComplexVecFine(i) = cpxVec(i) *
//              2.0 * PI / fft.gkkFine(i) * ik(i);
//          }
//        }
//
//        FFTWExecute ( fft, fft.backwardPlanFine );
//
//        for( Int i = 0; i < ntotFine; i++ ){
//          vlocDrv[d](i) = fft.inputComplexVecFine(i).real();
//        }
//      } // for (d)
//
//
//      Real wgt = domain_.Volume() / domain_.NumGridTotalFine();
//      Real resX = 0.0;
//      Real resY = 0.0;
//      Real resZ = 0.0;
//      for( Int i = 0; i < ntotFine; i++ ){
//        resX -= vlocDrv[0](i) * totalCharge(i) * wgt;
//        resY -= vlocDrv[1](i) * totalCharge(i) * wgt;
//        resZ -= vlocDrv[2](i) * totalCharge(i) * wgt;
//      }
//
//      force( a, 0 ) += resX;
//      force( a, 1 ) += resY;
//      force( a, 2 ) += resZ;
//
//    } // for (a)
//  }
//
//
//  // *********************************************************************
//  // Compute the force from nonlocal pseudopotential
//  // *********************************************************************
//  // Method 1: Using the derivative of the pseudopotential
//  if(0)
//  {
//    // Loop over atoms and pseudopotentials
//    Int numEig = occupationRate_.m();
//    Int numStateTotal = psi.NumStateTotal();
//    Int numStateLocal = psi.NumState();
//
//    MPI_Barrier(domain_.comm);
//    int mpirank;  MPI_Comm_rank(domain_.comm, &mpirank);
//    int mpisize;  MPI_Comm_size(domain_.comm, &mpisize);
//
//    if( numEig != numStateTotal ){
//      ErrorHandling( "numEig != numStateTotal in CalculateForce" );
//    }
//
//    for( Int a = 0; a < numAtom; a++ ){
//      std::vector<NonlocalPP>& vnlList = pseudo_[a].vnlList;
//      for( Int l = 0; l < vnlList.size(); l++ ){
//        SparseVec& bl = vnlList[l].first;
//        Real  gamma   = vnlList[l].second;
//        // FIXME Change to coarse
//        Real wgt = domain_.Volume() / domain_.NumGridTotal();
//        IntNumVec& idx = bl.first;
//        DblNumMat& val = bl.second;
//
//        for( Int g = 0; g < numStateLocal; g++ ){
//          DblNumVec res(4);
//          SetValue( res, 0.0 );
//          Real* psiPtr = psi.Wavefun().VecData(0, g);
//          for( Int i = 0; i < idx.Size(); i++ ){
//            res(VAL) += val(i, VAL ) * psiPtr[ idx(i) ] * sqrt(wgt);
//            res(DX) += val(i, DX ) * psiPtr[ idx(i) ] * sqrt(wgt);
//            res(DY) += val(i, DY ) * psiPtr[ idx(i) ] * sqrt(wgt);
//            res(DZ) += val(i, DZ ) * psiPtr[ idx(i) ] * sqrt(wgt);
//          }
//
//          // forceLocal( a, 0 ) += 4.0 * occupationRate_( g + mpirank * psi.Blocksize() ) * gamma * res[VAL] * res[DX];
//          // forceLocal( a, 1 ) += 4.0 * occupationRate_( g + mpirank * psi.Blocksize() ) * gamma * res[VAL] * res[DY];
//          // forceLocal( a, 2 ) += 4.0 * occupationRate_( g + mpirank * psi.Blocksize() ) * gamma * res[VAL] * res[DZ];
//
//          forceLocal( a, 0 ) += 4.0 * occupationRate_( psi.WavefunIdx(g) ) * gamma * res[VAL] * res[DX];
//          forceLocal( a, 1 ) += 4.0 * occupationRate_( psi.WavefunIdx(g) ) * gamma * res[VAL] * res[DY];
//          forceLocal( a, 2 ) += 4.0 * occupationRate_( psi.WavefunIdx(g) ) * gamma * res[VAL] * res[DZ];
//
//        } // for (g)
//      } // for (l)
//    } // for (a)
//  }
//
//
//
//  // Method 2: Using integration by parts, and throw the derivative to the wavefunctions
//  // FIXME: Assuming real arithmetic is used here.
//  // This formulation must be used when ONCV pseudopotential is used.
//  if(1)
//  {
//    // Compute the derivative of the wavefunctions
//
//    Fourier* fftPtr = &(fft);
//
//    Int ntothalf = fftPtr->numGridTotalR2C;
//    Int ntot  = psi.NumGridTotal();
//    Int ncom  = psi.NumComponent();
//    Int nocc  = psi.NumState(); // Local number of states
//
//    DblNumVec realInVec(ntot);
//    CpxNumVec cpxSaveVec(ntothalf);
//    CpxNumVec cpxOutVec(ntothalf);
//
//    std::vector<DblNumTns>   psiDrv(DIM);
//    for( Int d = 0; d < DIM; d++ ){
//      psiDrv[d].Resize( ntot, ncom, nocc );
//      SetValue( psiDrv[d], 0.0 );
//    }
//
//    for (Int k=0; k<nocc; k++) {
//      for (Int j=0; j<ncom; j++) {
//        // For c2r and r2c transforms, the default is to DESTROY the
//        // input, therefore a copy of the original matrix is necessary. 
//        blas::Copy( ntot, psi.Wavefun().VecData(j, k), 1, 
//            realInVec.Data(), 1 );
//        fftw_execute_dft_r2c(
//            fftPtr->forwardPlanR2C, 
//            realInVec.Data(),
//            reinterpret_cast<fftw_complex*>(cpxOutVec.Data() ));
//
//        cpxSaveVec = cpxOutVec;
//
//        for( Int d = 0; d < DIM; d++ ){
//          Complex* ptr1   = fftPtr->ikR2C[d].Data();
//          Complex* ptr2   = cpxSaveVec.Data();
//          Complex* ptr3   = cpxOutVec.Data();
//          for (Int i=0; i<ntothalf; i++) {
//            *(ptr3++) = (*(ptr1++)) * (*(ptr2++));
//          }
//
//          fftw_execute_dft_c2r(
//              fftPtr->backwardPlanR2C,
//              reinterpret_cast<fftw_complex*>(cpxOutVec.Data() ),
//              realInVec.Data() );
//
//          blas::Axpy( ntot, 1.0 / Real(ntot), realInVec.Data(), 1, 
//              psiDrv[d].VecData(j, k), 1 );
//        }
//      }
//    }
//
//    // Loop over atoms and pseudopotentials
//    Int numEig = occupationRate_.m();
//
//    for( Int a = 0; a < numAtom; a++ ){
//      std::vector<NonlocalPP>& vnlList = pseudo_[a].vnlList;
//      for( Int l = 0; l < vnlList.size(); l++ ){
//        SparseVec& bl = vnlList[l].first;
//        Real  gamma   = vnlList[l].second;
//        Real wgt = domain_.Volume() / domain_.NumGridTotal();
//        IntNumVec& idx = bl.first;
//        DblNumMat& val = bl.second;
//
//        for( Int g = 0; g < nocc; g++ ){
//          DblNumVec res(4);
//          SetValue( res, 0.0 );
//          Real* psiPtr = psi.Wavefun().VecData(0, g);
//          Real* DpsiXPtr = psiDrv[0].VecData(0, g);
//          Real* DpsiYPtr = psiDrv[1].VecData(0, g);
//          Real* DpsiZPtr = psiDrv[2].VecData(0, g);
//          for( Int i = 0; i < idx.Size(); i++ ){
//            res(VAL) += val(i, VAL ) * psiPtr[ idx(i) ] * sqrt(wgt);
//            res(DX)  += val(i, VAL ) * DpsiXPtr[ idx(i) ] * sqrt(wgt);
//            res(DY)  += val(i, VAL ) * DpsiYPtr[ idx(i) ] * sqrt(wgt);
//            res(DZ)  += val(i, VAL ) * DpsiZPtr[ idx(i) ] * sqrt(wgt);
//          }
//
//          forceLocal( a, 0 ) += -4.0 * occupationRate_(psi.WavefunIdx(g)) * gamma * res[VAL] * res[DX];
//          forceLocal( a, 1 ) += -4.0 * occupationRate_(psi.WavefunIdx(g)) * gamma * res[VAL] * res[DY];
//          forceLocal( a, 2 ) += -4.0 * occupationRate_(psi.WavefunIdx(g)) * gamma * res[VAL] * res[DZ];
//        } // for (g)
//      } // for (l)
//    } // for (a)
//  }
//
//
//  // Method 3: Using the derivative of the pseudopotential, but evaluated on a fine grid
//  if(0)
//  {
//    // Loop over atoms and pseudopotentials
//    Int numEig = occupationRate_.m();
//    Int numStateTotal = psi.NumStateTotal();
//    Int numStateLocal = psi.NumState();
//
//    MPI_Barrier(domain_.comm);
//    int mpirank;  MPI_Comm_rank(domain_.comm, &mpirank);
//    int mpisize;  MPI_Comm_size(domain_.comm, &mpisize);
//
//    if( numEig != numStateTotal ){
//      ErrorHandling( "numEig != numStateTotal in CalculateForce" );
//    }
//
//    DblNumVec wfnFine(domain_.NumGridTotalFine());
//
//    for( Int a = 0; a < numAtom; a++ ){
//      // Use nonlocal pseudopotential on the fine grid 
//      std::vector<NonlocalPP>& vnlList = pseudo_[a].vnlList;
//      for( Int l = 0; l < vnlList.size(); l++ ){
//        SparseVec& bl = vnlList[l].first;
//        Real  gamma   = vnlList[l].second;
//        Real wgt = domain_.Volume() / domain_.NumGridTotalFine();
//        IntNumVec& idx = bl.first;
//        DblNumMat& val = bl.second;
//
//        for( Int g = 0; g < numStateLocal; g++ ){
//          DblNumVec res(4);
//          SetValue( res, 0.0 );
//          Real* psiPtr = psi.Wavefun().VecData(0, g);
//
//          // Interpolate the wavefunction from coarse to fine grid
//
//          for( Int i = 0; i < domain_.NumGridTotal(); i++ ){
//            fft.inputComplexVec(i) = Complex( psiPtr[i], 0.0 ); 
//          }
//
//          FFTWExecute ( fft, fft.forwardPlan );
//
//          // fft Coarse to Fine 
//
//          SetValue( fft.outputComplexVecFine, Z_ZERO );
//          for( Int i = 0; i < domain_.NumGridTotal(); i++ ){
//            fft.outputComplexVecFine(fft.idxFineGrid(i)) = fft.outputComplexVec(i) *
//              sqrt( double(domain_.NumGridTotal()) / double(domain_.NumGridTotalFine()) );
//          }
//
//          FFTWExecute ( fft, fft.backwardPlanFine );
//
//          for( Int i = 0; i < domain_.NumGridTotalFine(); i++ ){
//            wfnFine(i) = fft.inputComplexVecFine(i).real();
//          }
//
//          for( Int i = 0; i < idx.Size(); i++ ){
//            res(VAL) += val(i, VAL ) * wfnFine[ idx(i) ] * sqrt(wgt);
//            res(DX) += val(i, DX ) * wfnFine[ idx(i) ] * sqrt(wgt);
//            res(DY) += val(i, DY ) * wfnFine[ idx(i) ] * sqrt(wgt);
//            res(DZ) += val(i, DZ ) * wfnFine[ idx(i) ] * sqrt(wgt);
//          }
//
//          forceLocal( a, 0 ) += 4.0 * occupationRate_( psi.WavefunIdx(g) ) * gamma * res[VAL] * res[DX];
//          forceLocal( a, 1 ) += 4.0 * occupationRate_( psi.WavefunIdx(g) ) * gamma * res[VAL] * res[DY];
//          forceLocal( a, 2 ) += 4.0 * occupationRate_( psi.WavefunIdx(g) ) * gamma * res[VAL] * res[DZ];
//
//        } // for (g)
//      } // for (l)
//    } // for (a)
//  }
//
//  // *********************************************************************
//  // Compute the total force and give the value to atomList
//  // *********************************************************************
//
//  // Sum over the force
//  DblNumMat  forceTmp( numAtom, DIM );
//  SetValue( forceTmp, 0.0 );
//
//  mpi::Allreduce( forceLocal.Data(), forceTmp.Data(), numAtom * DIM, MPI_SUM, domain_.comm );
//
//  for( Int a = 0; a < numAtom; a++ ){
//    force( a, 0 ) = force( a, 0 ) + forceTmp( a, 0 );
//    force( a, 1 ) = force( a, 1 ) + forceTmp( a, 1 );
//    force( a, 2 ) = force( a, 2 ) + forceTmp( a, 2 );
//  }
//
//  for( Int a = 0; a < numAtom; a++ ){
//    atomList_[a].force = Point3( force(a,0), force(a,1), force(a,2) );
//  } 
//
//
//
//  return ;
//}         // -----  end of method KohnSham::CalculateForce  ----- 

#ifdef _COMPLEX_
void
KohnSham::CalculateForce    ( Spinor& psi, Fourier& fft  )
{

  Real timeSta, timeEnd;

  // DEBUG purpose: special time on FFT
  Real timeFFTSta, timeFFTEnd, timeFFTTotal = 0.0;

  Int ntotFine  = fft.domain.NumGridTotalFine();
  Int ntot  = psi.NumGridTotal();
  Int ncom  = psi.NumComponent();
  Int numStateLocal = psi.NumState(); // Local number of states

  Int numAtom   = atomList_.size();

  DblNumMat  force( numAtom, DIM );
  SetValue( force, 0.0 );
  DblNumMat  forceLocal( numAtom, DIM );
  SetValue( forceLocal, 0.0 );

  // *********************************************************************
  // Compute the force from local pseudopotential
  // *********************************************************************
  // Using integration by parts for local pseudopotential.
  // No need to evaluate the derivative of the local pseudopotential.
  // This could potentially save some coding effort, and perhaps better for other 
  // pseudopotential such as Troullier-Martins
  GetTime( timeSta );
  
  if( esdfParam.isUseVLocal == false )
  {
    std::vector<DblNumVec>  vhartDrv(DIM);

    DblNumVec  totalCharge(ntotFine);
    SetValue( totalCharge, 0.0 );

    // totalCharge = density_ - pseudoCharge_
    blas::Copy( ntotFine, density_.VecData(0), 1, totalCharge.Data(), 1 );
    blas::Axpy( ntotFine, -1.0, pseudoCharge_.Data(),1,
        totalCharge.Data(), 1 );

    // Total charge in the Fourier space
    CpxNumVec  totalChargeFourier( ntotFine );

    for( Int i = 0; i < ntotFine; i++ ){
      fft.inputComplexVecFine(i) = Complex( totalCharge(i), 0.0 );
    }

    GetTime( timeFFTSta );
    FFTWExecute ( fft, fft.forwardPlanFine );
    GetTime( timeFFTEnd );
    timeFFTTotal += timeFFTEnd - timeFFTSta;


    blas::Copy( ntotFine, fft.outputComplexVecFine.Data(), 1,
        totalChargeFourier.Data(), 1 );

    // Compute the derivative of the Hartree potential via Fourier
    // transform 
    for( Int d = 0; d < DIM; d++ ){
      CpxNumVec& ikFine = fft.ikFine[d];
      for( Int i = 0; i < ntotFine; i++ ){
        if( fft.gkkFine(i) <= EPS || fft.gkkFine(i) > esdfParam.ecutWavefunction * 4.0 ){ 
          fft.outputComplexVecFine(i) = Z_ZERO;
        }
        else{
          // NOTE: gkk already contains the factor 1/2.
          fft.outputComplexVecFine(i) = totalChargeFourier(i) *
            2.0 * PI / fft.gkkFine(i) * ikFine(i);
        }
      }

      GetTime( timeFFTSta );
      FFTWExecute ( fft, fft.backwardPlanFine );
      GetTime( timeFFTEnd );
      timeFFTTotal += timeFFTEnd - timeFFTSta;

      // vhartDrv saves the derivative of the Hartree potential
      vhartDrv[d].Resize( ntotFine );

      for( Int i = 0; i < ntotFine; i++ ){
        vhartDrv[d](i) = fft.inputComplexVecFine(i).real();
      }

    } // for (d)


    // FIXME This should be parallelized
    for (Int a=0; a<numAtom; a++) {
      PseudoPot& pp = pseudo_[a];
      SparseVec& sp = pp.pseudoCharge;
      IntNumVec& idx = sp.first;
      DblNumMat& val = sp.second;

      Real wgt = domain_.Volume() / domain_.NumGridTotalFine();
      Real resX = 0.0;
      Real resY = 0.0;
      Real resZ = 0.0;
      for( Int l = 0; l < idx.m(); l++ ){
        resX += val(l, VAL) * vhartDrv[0][idx(l)] * wgt;
        resY += val(l, VAL) * vhartDrv[1][idx(l)] * wgt;
        resZ += val(l, VAL) * vhartDrv[2][idx(l)] * wgt;
      }
      force( a, 0 ) += resX;
      force( a, 1 ) += resY;
      force( a, 2 ) += resZ;

    } // for (a)
  } // pseudocharge formulation of the local contribution to the force
  else{
    // First contribution from the pseudocharge
    std::vector<DblNumVec>  vhartDrv(DIM);

    DblNumVec  totalCharge(ntotFine);
    SetValue( totalCharge, 0.0 );

    // totalCharge = density_ - pseudoCharge_
    blas::Copy( ntotFine, density_.VecData(0), 1, totalCharge.Data(), 1 );
    blas::Axpy( ntotFine, -1.0, pseudoCharge_.Data(),1,
        totalCharge.Data(), 1 );

    // Total charge in the Fourier space
    CpxNumVec  totalChargeFourier( ntotFine );

    for( Int i = 0; i < ntotFine; i++ ){
      fft.inputComplexVecFine(i) = Complex( totalCharge(i), 0.0 );
    }

    GetTime( timeFFTSta );
    FFTWExecute ( fft, fft.forwardPlanFine );
    GetTime( timeFFTEnd );
    timeFFTTotal += timeFFTEnd - timeFFTSta;


    blas::Copy( ntotFine, fft.outputComplexVecFine.Data(), 1,
        totalChargeFourier.Data(), 1 );

    // Compute the derivative of the Hartree potential via Fourier
    // transform 
    for( Int d = 0; d < DIM; d++ ){
      CpxNumVec& ikFine = fft.ikFine[d];
      for( Int i = 0; i < ntotFine; i++ ){
        if( fft.gkkFine(i) <= EPS || fft.gkkFine(i) > esdfParam.ecutWavefunction * 4.0 ){ 
          fft.outputComplexVecFine(i) = Z_ZERO;
        }
        else{
          // NOTE: gkk already contains the factor 1/2.
          fft.outputComplexVecFine(i) = totalChargeFourier(i) *
            2.0 * PI / fft.gkkFine(i) * ikFine(i);
        }
      }

      GetTime( timeFFTSta );
      FFTWExecute ( fft, fft.backwardPlanFine );
      GetTime( timeFFTEnd );
      timeFFTTotal += timeFFTEnd - timeFFTSta;

      // vhartDrv saves the derivative of the Hartree potential
      vhartDrv[d].Resize( ntotFine );

      for( Int i = 0; i < ntotFine; i++ ){
        vhartDrv[d](i) = fft.inputComplexVecFine(i).real();
      }

    } // for (d)


    // FIXME This should be parallelized
    for (Int a=0; a<numAtom; a++) {
      PseudoPot& pp = pseudo_[a];
      SparseVec& sp = pp.pseudoCharge;
      IntNumVec& idx = sp.first;
      DblNumMat& val = sp.second;

      Real wgt = domain_.Volume() / domain_.NumGridTotalFine();
      Real resX = 0.0;
      Real resY = 0.0;
      Real resZ = 0.0;
      for( Int l = 0; l < idx.m(); l++ ){
        resX += val(l, VAL) * vhartDrv[0][idx(l)] * wgt;
        resY += val(l, VAL) * vhartDrv[1][idx(l)] * wgt;
        resZ += val(l, VAL) * vhartDrv[2][idx(l)] * wgt;
      }
      force( a, 0 ) += resX;
      force( a, 1 ) += resY;
      force( a, 2 ) += resZ;

    } // for (a)
  
  
    // Second, contribution from the vLocalSR.  
    // The integration by parts formula requires the calculation of the grad density
    this->CalculateGradDensity( fft );

    // FIXME This should be parallelized
    for (Int a=0; a<numAtom; a++) {
      PseudoPot& pp = pseudo_[a];
      SparseVec& sp = pp.vLocalSR;
      IntNumVec& idx = sp.first;
      DblNumMat& val = sp.second;

//      statusOFS << "vLocalSR = " << val << std::endl;
//      statusOFS << "gradDensity_[0] = " << gradDensity_[0] << std::endl;

      Real wgt = domain_.Volume() / domain_.NumGridTotalFine();
      Real resX = 0.0;
      Real resY = 0.0;
      Real resZ = 0.0;
      for( Int l = 0; l < idx.m(); l++ ){
        resX -= val(l, VAL) * gradDensity_[0](idx(l),0) * wgt;
        resY -= val(l, VAL) * gradDensity_[1](idx(l),0) * wgt;
        resZ -= val(l, VAL) * gradDensity_[2](idx(l),0) * wgt;
      }
      force( a, 0 ) += resX;
      force( a, 1 ) += resY;
      force( a, 2 ) += resZ;

    } // for (a)
  
  
  } // VLocal formulation of the local contribution to the force




  if(0){
    // Output the local component of the force for debugging purpose
    for( Int a = 0; a < numAtom; a++ ){
      Point3 ft(force(a,0),force(a,1),force(a,2));
      Print( statusOFS, "atom", a, "localforce ", ft );
    }
  }

  GetTime( timeEnd );

#if ( _DEBUGlevel_ >= 0 )
  statusOFS << "Time for computing the local potential contribution of the force is " <<
    timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif


  // *********************************************************************
  // Compute the force from nonlocal pseudopotential
  // *********************************************************************
  GetTime( timeSta );
  // Method 4: Using integration by parts, and throw the derivative to the wavefunctions
  // No need to evaluate the derivative of the non-local pseudopotential.
  // This could potentially save some coding effort, and perhaps better for other 
  // pseudopotential such as Troullier-Martins
  if(1)
  {
    Int ntot  = psi.NumGridTotal(); 
    Int ncom  = psi.NumComponent();
    Int numStateLocal = psi.NumState(); // Local number of states

    CpxNumVec                psiFine( ntotFine );
    std::vector<CpxNumVec>   psiDrvFine(DIM);
    for( Int d = 0; d < DIM; d++ ){
      psiDrvFine[d].Resize( ntotFine );
    }

    CpxNumVec psiFourier(ntotFine);

    // Loop over atoms and pseudopotentials
    Int numEig = occupationRate_.m();
    for( Int g = 0; g < numStateLocal; g++ ){
      // Compute the derivative of the wavefunctions on a fine grid
      Complex* psiPtr = psi.Wavefun().VecData(0, g);
      for( Int i = 0; i < domain_.NumGridTotal(); i++ ){
        fft.inputComplexVec(i) = psiPtr[i];
      }

      GetTime( timeFFTSta );
      FFTWExecute ( fft, fft.forwardPlan );
      GetTime( timeFFTEnd );
      timeFFTTotal += timeFFTEnd - timeFFTSta;
      // fft Coarse to Fine 

      SetValue( psiFourier, Z_ZERO );
      for( Int i = 0; i < ntot; i++ ){
        psiFourier(fft.idxFineGrid(i)) = fft.outputComplexVec(i);
      }

      // psi on a fine grid
      for( Int i = 0; i < ntotFine; i++ ){
        fft.outputComplexVecFine(i) = psiFourier(i);
      }

      GetTime( timeFFTSta );
      FFTWExecute ( fft, fft.backwardPlanFine );
      GetTime( timeFFTEnd );
      timeFFTTotal += timeFFTEnd - timeFFTSta;

      Real fac = sqrt(double(domain_.NumGridTotal())) / 
        sqrt( double(domain_.NumGridTotalFine()) ); 
      //      for( Int i = 0; i < domain_.NumGridTotalFine(); i++ ){
      //        psiFine(i) = fft.inputComplexVecFine(i).real() * fac;
      //      }
      blas::Copy( ntotFine, fft.inputComplexVecFine.Data(),
          1, psiFine.Data(), 1 );
      blas::Scal( ntotFine, fac, psiFine.Data(), 1 );

      // derivative of psi on a fine grid
      for( Int d = 0; d < DIM; d++ ){
        Complex* ikFinePtr     = fft.ikFine[d].Data();
        Complex* psiFourierPtr = psiFourier.Data();
        Complex* fftOutFinePtr = fft.outputComplexVecFine.Data();
        for( Int i = 0; i < ntotFine; i++ ){
          //          fft.outputComplexVecFine(i) = psiFourier(i) * ikFine(i);
          *(fftOutFinePtr++) = *(psiFourierPtr++) * *(ikFinePtr++);
        }

        GetTime( timeFFTSta );
        FFTWExecute ( fft, fft.backwardPlanFine );
        GetTime( timeFFTEnd );
        timeFFTTotal += timeFFTEnd - timeFFTSta;

        //        for( Int i = 0; i < domain_.NumGridTotalFine(); i++ ){
        //          psiDrvFine[d](i) = fft.inputComplexVecFine(i).real() * fac;
        //        }
        blas::Copy( ntotFine, fft.inputComplexVecFine.Data(),
            1, psiDrvFine[d].Data(), 1 );
        blas::Scal( ntotFine, fac, psiDrvFine[d].Data(), 1 );

      } // for (d)

      // Evaluate the contribution to the atomic force
      for( Int a = 0; a < numAtom; a++ ){
        std::vector<NonlocalPP>& vnlList = pseudo_[a].vnlList;
        for( Int l = 0; l < vnlList.size(); l++ ){
          SparseVec& bl = vnlList[l].first;
          Real  gamma   = vnlList[l].second;
          Real  wgt = domain_.Volume() / domain_.NumGridTotalFine();
          IntNumVec& idx = bl.first;
          DblNumMat& val = bl.second;

          CpxNumVec res(4);
          SetValue( res, Complex(0.0,0.0) );
          Complex* psiPtr = psiFine.Data();
          Complex* DpsiXPtr = psiDrvFine[0].Data();
          Complex* DpsiYPtr = psiDrvFine[1].Data();
          Complex* DpsiZPtr = psiDrvFine[2].Data();
          Real* valPtr   = val.VecData(VAL);
          Int*  idxPtr = idx.Data();
          for( Int i = 0; i < idx.Size(); i++ ){
            res(VAL) += *valPtr * psiPtr[ *idxPtr ] * sqrt(wgt);
            res(DX)  += *valPtr * DpsiXPtr[ *idxPtr ] * sqrt(wgt);
            res(DY)  += *valPtr * DpsiYPtr[ *idxPtr ] * sqrt(wgt);
            res(DZ)  += *valPtr * DpsiZPtr[ *idxPtr ] * sqrt(wgt);
            valPtr++;
            idxPtr++;
          }

          forceLocal( a, 0 ) += -4.0 * occupationRate_(psi.WavefunIdx(g)) * gamma * (res[VAL] * std::conj(res[DX])).real();
          forceLocal( a, 1 ) += -4.0 * occupationRate_(psi.WavefunIdx(g)) * gamma * (res[VAL] * std::conj(res[DY])).real();
          forceLocal( a, 2 ) += -4.0 * occupationRate_(psi.WavefunIdx(g)) * gamma * (res[VAL] * std::conj(res[DZ])).real();
        } // for (l)
      } // for (a)

    } // for (g)
  }

  GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
  statusOFS << "Time for computing the nonlocal potential contribution of the force is " <<
    timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif


#if ( _DEBUGlevel_ >= 0 )
  statusOFS << "Total time for FFT in the computation of the force is " <<
    timeFFTTotal << " [s]" << std::endl << std::endl;
#endif
  // *********************************************************************
  // Compute the total force and give the value to atomList
  // *********************************************************************

  // Sum over the force
  DblNumMat  forceTmp( numAtom, DIM );
  SetValue( forceTmp, 0.0 );

  mpi::Allreduce( forceLocal.Data(), forceTmp.Data(), numAtom * DIM, MPI_SUM, domain_.comm );

  for( Int a = 0; a < numAtom; a++ ){
    force( a, 0 ) = force( a, 0 ) + forceTmp( a, 0 );
    force( a, 1 ) = force( a, 1 ) + forceTmp( a, 1 );
    force( a, 2 ) = force( a, 2 ) + forceTmp( a, 2 );
  }

  for( Int a = 0; a < numAtom; a++ ){
    atomList_[a].force = Point3( force(a,0), force(a,1), force(a,2) );
  } 

  // Add extra contribution to the force
  if( esdfParam.VDWType == "DFT-D2"){
    // Update force
    std::vector<Atom>& atomList = this->AtomList();
    for( Int a = 0; a < atomList.size(); a++ ){
      atomList[a].force += Point3( forceVdw_(a,0), forceVdw_(a,1), forceVdw_(a,2) );
    }
  }

  // Add the contribution from short range interaction
  if( esdfParam.isUseVLocal == true ){
    std::vector<Atom>& atomList = this->AtomList();
    for( Int a = 0; a < atomList.size(); a++ ){
      atomList[a].force += Point3( forceIonSR_(a,0), forceIonSR_(a,1), forceIonSR_(a,2) );
    }
  }

  // Add the contribution from external force
  {
    std::vector<Atom>& atomList = this->AtomList();
    for( Int a = 0; a < atomList.size(); a++ ){
      atomList[a].force += Point3( forceext_(a,0), forceext_(a,1), forceext_(a,2) );
    }
  }

  return ;
}         // -----  end of method KohnSham::CalculateForce  ----- 

#else

void
KohnSham::CalculateForce    ( Spinor& psi, Fourier& fft  )
{

  Real timeSta, timeEnd;

  // DEBUG purpose: special time on FFT
  Real timeFFTSta, timeFFTEnd, timeFFTTotal = 0.0;

  Int ntotFine  = fft.domain.NumGridTotalFine();
  Int ntot  = psi.NumGridTotal();
  Int ncom  = psi.NumComponent();
  Int numStateLocal = psi.NumState(); // Local number of states

  Int numAtom   = atomList_.size();
  Real EPS = 1e-14;
  DblNumMat  force( numAtom, DIM );
  SetValue( force, 0.0 );
  DblNumMat  forceLocal( numAtom, DIM );
  SetValue( forceLocal, 0.0 );

  // *********************************************************************
  // Compute the force from local pseudopotential
  // *********************************************************************
  // Using integration by parts for local pseudopotential.
  // No need to evaluate the derivative of the local pseudopotential.
  // This could potentially save some coding effort, and perhaps better for other 
  // pseudopotential such as Troullier-Martins
  GetTime( timeSta );
  
  if( esdfParam.isUseVLocal == false )
  {
    std::vector<DblNumVec>  vhartDrv(DIM);

    DblNumVec  totalCharge(ntotFine);
    SetValue( totalCharge, 0.0 );

    // totalCharge = density_ - pseudoCharge_
    blas::Copy( ntotFine, density_.VecData(0), 1, totalCharge.Data(), 1 );
    blas::Axpy( ntotFine, -1.0, pseudoCharge_.Data(),1,
        totalCharge.Data(), 1 );

    // Total charge in the Fourier space
    CpxNumVec  totalChargeFourier( ntotFine );

    for( Int i = 0; i < ntotFine; i++ ){
      fft.inputComplexVecFine(i) = Complex( totalCharge(i), 0.0 );
    }

    GetTime( timeFFTSta );
    FFTWExecute ( fft, fft.forwardPlanFine );
    GetTime( timeFFTEnd );
    timeFFTTotal += timeFFTEnd - timeFFTSta;


    blas::Copy( ntotFine, fft.outputComplexVecFine.Data(), 1,
        totalChargeFourier.Data(), 1 );

    // Compute the derivative of the Hartree potential via Fourier
    // transform 
    for( Int d = 0; d < DIM; d++ ){
      CpxNumVec& ikFine = fft.ikFine[d];
      for( Int i = 0; i < ntotFine; i++ ){
        if( fft.gkkFine(i) <= EPS || fft.gkkFine(i) > esdfParam.ecutWavefunction * 4.0 ){ 
          fft.outputComplexVecFine(i) = Z_ZERO;
        }
        else{
          // NOTE: gkk already contains the factor 1/2.
          fft.outputComplexVecFine(i) = totalChargeFourier(i) *
            2.0 * PI / fft.gkkFine(i) * ikFine(i);
        }
      }

      GetTime( timeFFTSta );
      FFTWExecute ( fft, fft.backwardPlanFine );
      GetTime( timeFFTEnd );
      timeFFTTotal += timeFFTEnd - timeFFTSta;

      // vhartDrv saves the derivative of the Hartree potential
      vhartDrv[d].Resize( ntotFine );

      for( Int i = 0; i < ntotFine; i++ ){
        vhartDrv[d](i) = fft.inputComplexVecFine(i).real();
      }

    } // for (d)


    // FIXME This should be parallelized
    for (Int a=0; a<numAtom; a++) {
      PseudoPot& pp = pseudo_[a];
      SparseVec& sp = pp.pseudoCharge;
      IntNumVec& idx = sp.first;
      DblNumMat& val = sp.second;

      Real wgt = domain_.Volume() / domain_.NumGridTotalFine();
      Real resX = 0.0;
      Real resY = 0.0;
      Real resZ = 0.0;
      for( Int l = 0; l < idx.m(); l++ ){
        resX += val(l, VAL) * vhartDrv[0][idx(l)] * wgt;
        resY += val(l, VAL) * vhartDrv[1][idx(l)] * wgt;
        resZ += val(l, VAL) * vhartDrv[2][idx(l)] * wgt;
      }
      force( a, 0 ) += resX;
      force( a, 1 ) += resY;
      force( a, 2 ) += resZ;

    } // for (a)
  } // pseudocharge formulation of the local contribution to the force
  else{
    // First contribution from the pseudocharge
    std::vector<DblNumVec>  vhartDrv(DIM);

    DblNumVec  totalCharge(ntotFine);
    SetValue( totalCharge, 0.0 );

    // totalCharge = density_ - pseudoCharge_
    blas::Copy( ntotFine, density_.VecData(0), 1, totalCharge.Data(), 1 );
    blas::Axpy( ntotFine, -1.0, pseudoCharge_.Data(),1,
        totalCharge.Data(), 1 );

    // Total charge in the Fourier space
    CpxNumVec  totalChargeFourier( ntotFine );

    for( Int i = 0; i < ntotFine; i++ ){
      fft.inputComplexVecFine(i) = Complex( totalCharge(i), 0.0 );
    }

    GetTime( timeFFTSta );
    FFTWExecute ( fft, fft.forwardPlanFine );
    GetTime( timeFFTEnd );
    timeFFTTotal += timeFFTEnd - timeFFTSta;


    blas::Copy( ntotFine, fft.outputComplexVecFine.Data(), 1,
        totalChargeFourier.Data(), 1 );

    // Compute the derivative of the Hartree potential via Fourier
    // transform 
    for( Int d = 0; d < DIM; d++ ){
      CpxNumVec& ikFine = fft.ikFine[d];
      for( Int i = 0; i < ntotFine; i++ ){
        if( fft.gkkFine(i) <= EPS || fft.gkkFine(i) > esdfParam.ecutWavefunction * 4.0 ){ 
          fft.outputComplexVecFine(i) = Z_ZERO;
        }
        else{
          // NOTE: gkk already contains the factor 1/2.
          fft.outputComplexVecFine(i) = totalChargeFourier(i) *
            2.0 * PI / fft.gkkFine(i) * ikFine(i);
        }
      }

      GetTime( timeFFTSta );
      FFTWExecute ( fft, fft.backwardPlanFine );
      GetTime( timeFFTEnd );
      timeFFTTotal += timeFFTEnd - timeFFTSta;

      // vhartDrv saves the derivative of the Hartree potential
      vhartDrv[d].Resize( ntotFine );

      for( Int i = 0; i < ntotFine; i++ ){
        vhartDrv[d](i) = fft.inputComplexVecFine(i).real();
      }

    } // for (d)


    // FIXME This should be parallelized
    for (Int a=0; a<numAtom; a++) {
      PseudoPot& pp = pseudo_[a];
      SparseVec& sp = pp.pseudoCharge;
      IntNumVec& idx = sp.first;
      DblNumMat& val = sp.second;

      Real wgt = domain_.Volume() / domain_.NumGridTotalFine();
      Real resX = 0.0;
      Real resY = 0.0;
      Real resZ = 0.0;
      for( Int l = 0; l < idx.m(); l++ ){
        resX += val(l, VAL) * vhartDrv[0][idx(l)] * wgt;
        resY += val(l, VAL) * vhartDrv[1][idx(l)] * wgt;
        resZ += val(l, VAL) * vhartDrv[2][idx(l)] * wgt;
      }
      force( a, 0 ) += resX;
      force( a, 1 ) += resY;
      force( a, 2 ) += resZ;

    } // for (a)
  
  
    // Second, contribution from the vLocalSR.  
    // The integration by parts formula requires the calculation of the grad density
    this->CalculateGradDensity( fft );

    // FIXME This should be parallelized
    for (Int a=0; a<numAtom; a++) {
      PseudoPot& pp = pseudo_[a];
      SparseVec& sp = pp.vLocalSR;
      IntNumVec& idx = sp.first;
      DblNumMat& val = sp.second;

//      statusOFS << "vLocalSR = " << val << std::endl;
//      statusOFS << "gradDensity_[0] = " << gradDensity_[0] << std::endl;

      Real wgt = domain_.Volume() / domain_.NumGridTotalFine();
      Real resX = 0.0;
      Real resY = 0.0;
      Real resZ = 0.0;
      for( Int l = 0; l < idx.m(); l++ ){
        resX -= val(l, VAL) * gradDensity_[0](idx(l),0) * wgt;
        resY -= val(l, VAL) * gradDensity_[1](idx(l),0) * wgt;
        resZ -= val(l, VAL) * gradDensity_[2](idx(l),0) * wgt;
      }
      force( a, 0 ) += resX;
      force( a, 1 ) += resY;
      force( a, 2 ) += resZ;

    } // for (a)
  
  
  } // VLocal formulation of the local contribution to the force




  if(0){
    // Output the local component of the force for debugging purpose
    for( Int a = 0; a < numAtom; a++ ){
      Point3 ft(force(a,0),force(a,1),force(a,2));
      Print( statusOFS, "atom", a, "localforce ", ft );
    }
  }

  GetTime( timeEnd );

#if ( _DEBUGlevel_ >= 0 )
  statusOFS << "Time for computing the local potential contribution of the force is " <<
    timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif


  // *********************************************************************
  // Compute the force from nonlocal pseudopotential
  // *********************************************************************
  GetTime( timeSta );
  // Method 4: Using integration by parts, and throw the derivative to the wavefunctions
  // No need to evaluate the derivative of the non-local pseudopotential.
  // This could potentially save some coding effort, and perhaps better for other 
  // pseudopotential such as Troullier-Martins
  if(1)
  {
    Int ntot  = psi.NumGridTotal(); 
    Int ncom  = psi.NumComponent();
    Int numStateLocal = psi.NumState(); // Local number of states

    DblNumVec                psiFine( ntotFine );
    std::vector<DblNumVec>   psiDrvFine(DIM);
    for( Int d = 0; d < DIM; d++ ){
      psiDrvFine[d].Resize( ntotFine );
    }

    CpxNumVec psiFourier(ntotFine);

    // Loop over atoms and pseudopotentials
    Int numEig = occupationRate_.m();
    for( Int g = 0; g < numStateLocal; g++ ){
      // Compute the derivative of the wavefunctions on a fine grid
      Real* psiPtr = psi.Wavefun().VecData(0, g);
      for( Int i = 0; i < domain_.NumGridTotal(); i++ ){
        fft.inputComplexVec(i) = Complex( psiPtr[i], 0.0 ); 
      }

      GetTime( timeFFTSta );
      FFTWExecute ( fft, fft.forwardPlan );
      GetTime( timeFFTEnd );
      timeFFTTotal += timeFFTEnd - timeFFTSta;
      // fft Coarse to Fine 

      SetValue( psiFourier, Z_ZERO );
      for( Int i = 0; i < ntot; i++ ){
        psiFourier(fft.idxFineGrid(i)) = fft.outputComplexVec(i);
      }

      // psi on a fine grid
      for( Int i = 0; i < ntotFine; i++ ){
        fft.outputComplexVecFine(i) = psiFourier(i);
      }

      GetTime( timeFFTSta );
      FFTWExecute ( fft, fft.backwardPlanFine );
      GetTime( timeFFTEnd );
      timeFFTTotal += timeFFTEnd - timeFFTSta;

      Real fac = sqrt(double(domain_.NumGridTotal())) / 
        sqrt( double(domain_.NumGridTotalFine()) ); 
      //      for( Int i = 0; i < domain_.NumGridTotalFine(); i++ ){
      //        psiFine(i) = fft.inputComplexVecFine(i).real() * fac;
      //      }
      blas::Copy( ntotFine, reinterpret_cast<Real*>(fft.inputComplexVecFine.Data()),
          2, psiFine.Data(), 1 );
      blas::Scal( ntotFine, fac, psiFine.Data(), 1 );

      // derivative of psi on a fine grid
      for( Int d = 0; d < DIM; d++ ){
        Complex* ikFinePtr = fft.ikFine[d].Data();
        Complex* psiFourierPtr    = psiFourier.Data();
        Complex* fftOutFinePtr = fft.outputComplexVecFine.Data();
        for( Int i = 0; i < ntotFine; i++ ){
          //          fft.outputComplexVecFine(i) = psiFourier(i) * ikFine(i);
          *(fftOutFinePtr++) = *(psiFourierPtr++) * *(ikFinePtr++);
        }

        GetTime( timeFFTSta );
        FFTWExecute ( fft, fft.backwardPlanFine );
        GetTime( timeFFTEnd );
        timeFFTTotal += timeFFTEnd - timeFFTSta;

        //        for( Int i = 0; i < domain_.NumGridTotalFine(); i++ ){
        //          psiDrvFine[d](i) = fft.inputComplexVecFine(i).real() * fac;
        //        }
        blas::Copy( ntotFine, reinterpret_cast<Real*>(fft.inputComplexVecFine.Data()),
            2, psiDrvFine[d].Data(), 1 );
        blas::Scal( ntotFine, fac, psiDrvFine[d].Data(), 1 );

      } // for (d)

      // Evaluate the contribution to the atomic force
      for( Int a = 0; a < numAtom; a++ ){
        std::vector<NonlocalPP>& vnlList = pseudo_[a].vnlList;
        for( Int l = 0; l < vnlList.size(); l++ ){
          SparseVec& bl = vnlList[l].first;
          Real  gamma   = vnlList[l].second;
          Real  wgt = domain_.Volume() / domain_.NumGridTotalFine();
          IntNumVec& idx = bl.first;
          DblNumMat& val = bl.second;

          DblNumVec res(4);
          SetValue( res, 0.0 );
          Real* psiPtr = psiFine.Data();
          Real* DpsiXPtr = psiDrvFine[0].Data();
          Real* DpsiYPtr = psiDrvFine[1].Data();
          Real* DpsiZPtr = psiDrvFine[2].Data();
          Real* valPtr   = val.VecData(VAL);
          Int*  idxPtr = idx.Data();
          for( Int i = 0; i < idx.Size(); i++ ){
            res(VAL) += *valPtr * psiPtr[ *idxPtr ] * sqrt(wgt);
            res(DX)  += *valPtr * DpsiXPtr[ *idxPtr ] * sqrt(wgt);
            res(DY)  += *valPtr * DpsiYPtr[ *idxPtr ] * sqrt(wgt);
            res(DZ)  += *valPtr * DpsiZPtr[ *idxPtr ] * sqrt(wgt);
            valPtr++;
            idxPtr++;
          }

          forceLocal( a, 0 ) += -4.0 * occupationRate_(psi.WavefunIdx(g)) * gamma * res[VAL] * res[DX];
          forceLocal( a, 1 ) += -4.0 * occupationRate_(psi.WavefunIdx(g)) * gamma * res[VAL] * res[DY];
          forceLocal( a, 2 ) += -4.0 * occupationRate_(psi.WavefunIdx(g)) * gamma * res[VAL] * res[DZ];
        } // for (l)
      } // for (a)

    } // for (g)
  }

  GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
  statusOFS << "Time for computing the nonlocal potential contribution of the force is " <<
    timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif


#if ( _DEBUGlevel_ >= 0 )
  statusOFS << "Total time for FFT in the computation of the force is " <<
    timeFFTTotal << " [s]" << std::endl << std::endl;
#endif
  // *********************************************************************
  // Compute the total force and give the value to atomList
  // *********************************************************************

  // Sum over the force
  DblNumMat  forceTmp( numAtom, DIM );
  SetValue( forceTmp, 0.0 );

  mpi::Allreduce( forceLocal.Data(), forceTmp.Data(), numAtom * DIM, MPI_SUM, domain_.comm );

  for( Int a = 0; a < numAtom; a++ ){
    force( a, 0 ) = force( a, 0 ) + forceTmp( a, 0 );
    force( a, 1 ) = force( a, 1 ) + forceTmp( a, 1 );
    force( a, 2 ) = force( a, 2 ) + forceTmp( a, 2 );
  }

  for( Int a = 0; a < numAtom; a++ ){
    atomList_[a].force = Point3( force(a,0), force(a,1), force(a,2) );
  } 

  // Add extra contribution to the force
  if( esdfParam.VDWType == "DFT-D2"){
    // Update force
    std::vector<Atom>& atomList = this->AtomList();
    for( Int a = 0; a < atomList.size(); a++ ){
      atomList[a].force += Point3( forceVdw_(a,0), forceVdw_(a,1), forceVdw_(a,2) );
    }
  }

  // Add the contribution from short range interaction
  if( esdfParam.isUseVLocal == true ){
    std::vector<Atom>& atomList = this->AtomList();
    for( Int a = 0; a < atomList.size(); a++ ){
      atomList[a].force += Point3( forceIonSR_(a,0), forceIonSR_(a,1), forceIonSR_(a,2) );
    }
  }

  // Add the contribution from external force
  {
    std::vector<Atom>& atomList = this->AtomList();
    for( Int a = 0; a < atomList.size(); a++ ){
      atomList[a].force += Point3( forceext_(a,0), forceext_(a,1), forceext_(a,2) );
    }
  }

  return ;
}         // -----  end of method KohnSham::CalculateForce  ----- 

#endif

#ifdef _COMPLEX_
void
KohnSham::MultSpinor    ( Spinor& psi, NumTns<Complex>& a3, Fourier& fft )
{

  MPI_Barrier(domain_.comm);
  int mpirank;  MPI_Comm_rank(domain_.comm, &mpirank);
  int mpisize;  MPI_Comm_size(domain_.comm, &mpisize);

  Int ntot      = fft.domain.NumGridTotal();
  Int ntotFine  = fft.domain.NumGridTotalFine();
  Int numStateTotal = psi.NumStateTotal();
  Int numStateLocal = psi.NumState();
  NumTns<Complex>& wavefun = psi.Wavefun();
  Int ncom = wavefun.n();

  Int ntotR2C = fft.numGridTotal;

  Real timeSta, timeEnd;
  Real timeSta1, timeEnd1;

  Real timeGemm = 0.0;
  Real timeAlltoallv = 0.0;
  Real timeAllreduce = 0.0;

  SetValue( a3, Complex(0.0,0.0) );

  // Apply an initial filter on the wavefunctions, if required
  if((apply_filter_ == 1 && apply_first_ == 1))
  {

    //statusOFS << std::endl << " In here in 1st filter : " << wfn_cutoff_ << std::endl; 
    apply_first_ = 0;

    for (Int k=0; k<numStateLocal; k++) {
      for (Int j=0; j<ncom; j++) {

        SetValue( fft.inputComplexVec,  Z_ZERO);
        SetValue( fft.outputComplexVec, Z_ZERO );

        blas::Copy( ntot, wavefun.VecData(j,k), 1,
            fft.inputComplexVec.Data(), 1 );
        FFTWExecute ( fft, fft.forwardPlan ); // So outputComplexVec contains the FFT result now


        for (Int i=0; i<ntotR2C; i++)
        {
          if(fft.gkk(i) > wfn_cutoff_)
            fft.outputComplexVec(i) = Z_ZERO;
        }

        FFTWExecute ( fft, fft.backwardPlan);
        blas::Copy( ntot,  fft.inputComplexVec.Data(), 1,
            wavefun.VecData(j,k), 1 );

      }
    }
  }


  GetTime( timeSta );
  psi.AddMultSpinorFine( fft, vtot_, pseudo_, a3 );
  GetTime( timeEnd );
  statusOFS << "Time for complex psi.AddMultSpinorFine is " <<
    timeEnd - timeSta << " [s]" << std::endl << std::endl;
#if ( _DEBUGlevel_ >= 1 )
  statusOFS << "Time for complex psi.AddMultSpinorFine is " <<
    timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

#if 1
  if( isHybrid_ && isEXXActive_ ){

    GetTime( timeSta );

    if( esdfParam.isHybridACE ){

      //      if(0)
      //      {
      //        DblNumMat M(numStateTotal, numStateTotal);
      //        blas::Gemm( 'T', 'N', numStateTotal, numStateTotal, ntot, 1.0,
      //            vexxProj_.Data(), ntot, psi.Wavefun().Data(), ntot, 
      //            0.0, M.Data(), M.m() );
      //        // Minus sign comes from that all eigenvalues are negative
      //        blas::Gemm( 'N', 'N', ntot, numStateTotal, numStateTotal, -1.0,
      //            vexxProj_.Data(), ntot, M.Data(), numStateTotal,
      //            1.0, a3.Data(), ntot );
      //      }

      if(1){ // for MPI
        // Convert the column partition to row partition

        Int numStateBlocksize = numStateTotal / mpisize;
        Int ntotBlocksize = ntot / mpisize;

        Int numStateLocal = numStateBlocksize;
        Int ntotLocal = ntotBlocksize;

        if(mpirank < (numStateTotal % mpisize)){
          numStateLocal = numStateBlocksize + 1;
        }

        if(mpirank < (ntot % mpisize)){
          ntotLocal = ntotBlocksize + 1;
        }

        CpxNumMat psiCol( ntot, numStateLocal );
        SetValue( psiCol, Z_ZERO );

        CpxNumMat vexxProjCol( ntot, numStateLocal );
        SetValue( vexxProjCol, Z_ZERO );

        CpxNumMat psiRow( ntotLocal, numStateTotal );
        SetValue( psiRow, Z_ZERO );

        CpxNumMat vexxProjRow( ntotLocal, numStateTotal );
        SetValue( vexxProjRow, Z_ZERO );

        lapack::Lacpy( 'A', ntot, numStateLocal, psi.Wavefun().Data(), ntot, psiCol.Data(), ntot );
        lapack::Lacpy( 'A', ntot, numStateLocal, vexxProj_.Data(), ntot, vexxProjCol.Data(), ntot );

        GetTime( timeSta1 );
        AlltoallForward (psiCol, psiRow, domain_.comm);
        AlltoallForward (vexxProjCol, vexxProjRow, domain_.comm);
        GetTime( timeEnd1 );
        timeAlltoallv = timeAlltoallv + ( timeEnd1 - timeSta1 );

        CpxNumMat MTemp( numStateTotal, numStateTotal );
        SetValue( MTemp, Z_ZERO );

        GetTime( timeSta1 );
        blas::Gemm( 'C', 'N', numStateTotal, numStateTotal, ntotLocal,
            1.0, vexxProjRow.Data(), ntotLocal, 
            psiRow.Data(), ntotLocal, 0.0,
            MTemp.Data(), numStateTotal );
        GetTime( timeEnd1 );
        timeGemm = timeGemm + ( timeEnd1 - timeSta1 );

        CpxNumMat M(numStateTotal, numStateTotal);
        SetValue( M, Z_ZERO );
        GetTime( timeSta1 );
        MPI_Allreduce( MTemp.Data(), M.Data(), 2*numStateTotal * numStateTotal, MPI_DOUBLE, MPI_SUM, domain_.comm );
        GetTime( timeEnd1 );
        timeAllreduce = timeAllreduce + ( timeEnd1 - timeSta1 );

        CpxNumMat a3Col( ntot, numStateLocal );
        SetValue( a3Col, Z_ZERO );

        CpxNumMat a3Row( ntotLocal, numStateTotal );
        SetValue( a3Row, Z_ZERO );

        GetTime( timeSta1 );
        blas::Gemm( 'N', 'N', ntotLocal, numStateTotal, numStateTotal, 
            -1.0, vexxProjRow.Data(), ntotLocal, 
            M.Data(), numStateTotal, 0.0, 
            a3Row.Data(), ntotLocal );
        GetTime( timeEnd1 );
        timeGemm = timeGemm + ( timeEnd1 - timeSta1 );

        GetTime( timeSta1 );
        AlltoallBackward (a3Row, a3Col, domain_.comm);
        GetTime( timeEnd1 );
        timeAlltoallv = timeAlltoallv + ( timeEnd1 - timeSta1 );

        GetTime( timeSta1 );
        for (Int k=0; k<numStateLocal; k++) {
          for (Int j=0; j<ncom; j++) {
            Complex *p1 = a3Col.VecData(k);
            Complex *p2 = a3.VecData(j, k);
            for (Int i=0; i<ntot; i++) { 
              *(p2++) += *(p1++); 
            }
          }
        }
        GetTime( timeEnd1 );
        timeGemm = timeGemm + ( timeEnd1 - timeSta1 );

      } //if(1)

    }
    else
    {
      psi.AddMultSpinorEXX( fft, phiEXX_, exxgkk_,
          exxFraction_,  numSpin_, occupationRate_, a3 );
    }
    GetTime( timeEnd );

#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for updating hybrid Spinor is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;
    statusOFS << "Time for Gemm is " <<
      timeGemm << " [s]" << std::endl << std::endl;
    statusOFS << "Time for Alltoallv is " <<
      timeAlltoallv << " [s]" << std::endl << std::endl;
    statusOFS << "Time for Allreduce is " <<
      timeAllreduce << " [s]" << std::endl << std::endl;
#endif


  }
#endif

  // Apply filter on the wavefunctions before exit, if required
  if((apply_filter_ == 1))
  {
    //statusOFS << std::endl << " In here in 2nd filter : "  << wfn_cutoff_<< std::endl; 
    for (Int k=0; k<numStateLocal; k++) {
      for (Int j=0; j<ncom; j++) {

        SetValue( fft.inputComplexVec, Z_ZERO );
        SetValue( fft.outputComplexVec, Z_ZERO );

        blas::Copy( ntot, a3.VecData(j,k), 1,
            fft.inputComplexVec.Data(), 1 );
        FFTWExecute ( fft, fft.forwardPlan ); // So outputVecR2C contains the FFT result now


        for (Int i=0; i<ntotR2C; i++)
        {
          if(fft.gkk(i) > wfn_cutoff_)
            fft.outputComplexVec(i) = Z_ZERO;
        }

        FFTWExecute ( fft, fft.backwardPlan );
        blas::Copy( ntot,  fft.inputComplexVec.Data(), 1,
            a3.VecData(j,k), 1 );

      }
    }
  }



  return ;
}         // -----  end of method KohnSham::MultSpinor  ----- 
#else

void
KohnSham::MultSpinor    ( Spinor& psi, NumTns<Real>& a3, Fourier& fft )
{

  MPI_Barrier(domain_.comm);
  int mpirank;  MPI_Comm_rank(domain_.comm, &mpirank);
  int mpisize;  MPI_Comm_size(domain_.comm, &mpisize);

  Int ntot      = fft.domain.NumGridTotal();
  Int ntotFine  = fft.domain.NumGridTotalFine();
  Int numStateTotal = psi.NumStateTotal();
  Int numStateLocal = psi.NumState();
  NumTns<Real>& wavefun = psi.Wavefun();
  Int ncom = wavefun.n();

  Int ntotR2C = fft.numGridTotalR2C;

  Real timeSta, timeEnd;
  Real timeSta1, timeEnd1;

  Real timeGemm = 0.0;
  Real timeAlltoallv = 0.0;
  Real timeAllreduce = 0.0;

  SetValue( a3, 0.0 );

  // Apply an initial filter on the wavefunctions, if required
  if((apply_filter_ == 1 && apply_first_ == 1))
  {

    //statusOFS << std::endl << " In here in 1st filter : " << wfn_cutoff_ << std::endl; 
    apply_first_ = 0;

    for (Int k=0; k<numStateLocal; k++) {
      for (Int j=0; j<ncom; j++) {

        SetValue( fft.inputVecR2C, 0.0 );
        SetValue( fft.outputVecR2C, Z_ZERO );

        blas::Copy( ntot, wavefun.VecData(j,k), 1,
            fft.inputVecR2C.Data(), 1 );
        FFTWExecute ( fft, fft.forwardPlanR2C ); // So outputVecR2C contains the FFT result now


        for (Int i=0; i<ntotR2C; i++)
        {
          if(fft.gkkR2C(i) > wfn_cutoff_)
            fft.outputVecR2C(i) = Z_ZERO;
        }

        FFTWExecute ( fft, fft.backwardPlanR2C );
        blas::Copy( ntot,  fft.inputVecR2C.Data(), 1,
            wavefun.VecData(j,k), 1 );

      }
    }
  }


  GetTime( timeSta );
  psi.AddMultSpinorFineR2C( fft, vtot_, pseudo_, a3 );
  GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 1 )
  statusOFS << "Time for psi.AddMultSpinorFineR2C is " <<
    timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

  if( isHybrid_ && isEXXActive_ ){

    GetTime( timeSta );

    if( esdfParam.isHybridACE ){

      //      if(0)
      //      {
      //        DblNumMat M(numStateTotal, numStateTotal);
      //        blas::Gemm( 'T', 'N', numStateTotal, numStateTotal, ntot, 1.0,
      //            vexxProj_.Data(), ntot, psi.Wavefun().Data(), ntot, 
      //            0.0, M.Data(), M.m() );
      //        // Minus sign comes from that all eigenvalues are negative
      //        blas::Gemm( 'N', 'N', ntot, numStateTotal, numStateTotal, -1.0,
      //            vexxProj_.Data(), ntot, M.Data(), numStateTotal,
      //            1.0, a3.Data(), ntot );
      //      }

      if(1){ // for MPI
        // Convert the column partition to row partition

        Int numStateBlocksize = numStateTotal / mpisize;
        Int ntotBlocksize = ntot / mpisize;

        Int numStateLocal = numStateBlocksize;
        Int ntotLocal = ntotBlocksize;

        if(mpirank < (numStateTotal % mpisize)){
          numStateLocal = numStateBlocksize + 1;
        }

        if(mpirank < (ntot % mpisize)){
          ntotLocal = ntotBlocksize + 1;
        }

        DblNumMat psiCol( ntot, numStateLocal );
        SetValue( psiCol, 0.0 );

        DblNumMat vexxProjCol( ntot, numStateLocal );
        SetValue( vexxProjCol, 0.0 );

        DblNumMat psiRow( ntotLocal, numStateTotal );
        SetValue( psiRow, 0.0 );

        DblNumMat vexxProjRow( ntotLocal, numStateTotal );
        SetValue( vexxProjRow, 0.0 );

        lapack::Lacpy( 'A', ntot, numStateLocal, psi.Wavefun().Data(), ntot, psiCol.Data(), ntot );
        lapack::Lacpy( 'A', ntot, numStateLocal, vexxProj_.Data(), ntot, vexxProjCol.Data(), ntot );

        GetTime( timeSta1 );
        AlltoallForward (psiCol, psiRow, domain_.comm);
        AlltoallForward (vexxProjCol, vexxProjRow, domain_.comm);
        GetTime( timeEnd1 );
        timeAlltoallv = timeAlltoallv + ( timeEnd1 - timeSta1 );

        DblNumMat MTemp( numStateTotal, numStateTotal );
        SetValue( MTemp, 0.0 );

        GetTime( timeSta1 );
        blas::Gemm( 'T', 'N', numStateTotal, numStateTotal, ntotLocal,
            1.0, vexxProjRow.Data(), ntotLocal, 
            psiRow.Data(), ntotLocal, 0.0,
            MTemp.Data(), numStateTotal );
        GetTime( timeEnd1 );
        timeGemm = timeGemm + ( timeEnd1 - timeSta1 );

        DblNumMat M(numStateTotal, numStateTotal);
        SetValue( M, 0.0 );
        GetTime( timeSta1 );
        MPI_Allreduce( MTemp.Data(), M.Data(), numStateTotal * numStateTotal, MPI_DOUBLE, MPI_SUM, domain_.comm );
        GetTime( timeEnd1 );
        timeAllreduce = timeAllreduce + ( timeEnd1 - timeSta1 );

        DblNumMat a3Col( ntot, numStateLocal );
        SetValue( a3Col, 0.0 );

        DblNumMat a3Row( ntotLocal, numStateTotal );
        SetValue( a3Row, 0.0 );

        GetTime( timeSta1 );
        blas::Gemm( 'N', 'N', ntotLocal, numStateTotal, numStateTotal, 
            -1.0, vexxProjRow.Data(), ntotLocal, 
            M.Data(), numStateTotal, 0.0, 
            a3Row.Data(), ntotLocal );
        GetTime( timeEnd1 );
        timeGemm = timeGemm + ( timeEnd1 - timeSta1 );

        GetTime( timeSta1 );
        AlltoallBackward (a3Row, a3Col, domain_.comm);
        GetTime( timeEnd1 );
        timeAlltoallv = timeAlltoallv + ( timeEnd1 - timeSta1 );

        GetTime( timeSta1 );
        for (Int k=0; k<numStateLocal; k++) {
          for (Int j=0; j<ncom; j++) {
            Real *p1 = a3Col.VecData(k);
            Real *p2 = a3.VecData(j, k);
            for (Int i=0; i<ntot; i++) { 
              *(p2++) += *(p1++); 
            }
          }
        }
        GetTime( timeEnd1 );
        timeGemm = timeGemm + ( timeEnd1 - timeSta1 );

      } //if(1)

    }
    else{
      psi.AddMultSpinorEXX( fft, phiEXX_, exxgkkR2C_,
          exxFraction_,  numSpin_, occupationRate_, a3 );
    }

    GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 1 )
    statusOFS << "Time for updating hybrid Spinor is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;
    statusOFS << "Time for Gemm is " <<
      timeGemm << " [s]" << std::endl << std::endl;
    statusOFS << "Time for Alltoallv is " <<
      timeAlltoallv << " [s]" << std::endl << std::endl;
    statusOFS << "Time for Allreduce is " <<
      timeAllreduce << " [s]" << std::endl << std::endl;
#endif


  }

  // Apply filter on the wavefunctions before exit, if required
  if((apply_filter_ == 1))
  {
    //statusOFS << std::endl << " In here in 2nd filter : "  << wfn_cutoff_<< std::endl; 
    for (Int k=0; k<numStateLocal; k++) {
      for (Int j=0; j<ncom; j++) {

        SetValue( fft.inputVecR2C, 0.0 );
        SetValue( fft.outputVecR2C, Z_ZERO );

        blas::Copy( ntot, a3.VecData(j,k), 1,
            fft.inputVecR2C.Data(), 1 );
        FFTWExecute ( fft, fft.forwardPlanR2C ); // So outputVecR2C contains the FFT result now


        for (Int i=0; i<ntotR2C; i++)
        {
          if(fft.gkkR2C(i) > wfn_cutoff_)
            fft.outputVecR2C(i) = Z_ZERO;
        }

        FFTWExecute ( fft, fft.backwardPlanR2C );
        blas::Copy( ntot,  fft.inputVecR2C.Data(), 1,
            a3.VecData(j,k), 1 );

      }
    }
  }



  return ;
}         // -----  end of method KohnSham::MultSpinor  ----- 
#endif


#ifdef _COMPLEX_

void KohnSham::InitializeEXX ( Real ecutWavefunction, Fourier& fft )
{

  const Real epsDiv = 1e-8;

  isEXXActive_ = false;

  Int numGridTotalR2C = fft.numGridTotalR2C;
  Int numGridTotal    = fft.numGridTotal;
  exxgkkR2C_.Resize(numGridTotalR2C);
  SetValue( exxgkkR2C_, 0.0 );

  exxgkk_.Resize(numGridTotal);
  SetValue( exxgkk_, 0.0 );


  // extra 2.0 factor for ecutWavefunction compared to QE due to unit difference
  // tpiba2 in QE is just a unit for G^2. Do not include it here
  Real exxAlpha = 10.0 / (ecutWavefunction * 2.0);

  // Gygi-Baldereschi regularization. Currently set to zero and compare
  // with QE without the regularization 
  // Set exxdiv_treatment to "none"
  // NOTE: I do not quite understand the detailed derivation
  // Compute the divergent term for G=0
  Real gkk2;
  if(exxDivergenceType_ == 0){
    exxDiv_ = 0.0;
  }
  else if (exxDivergenceType_ == 1){
    exxDiv_ = 0.0;
    // no q-point
    // NOTE: Compared to the QE implementation, it is easier to do below.
    // Do the integration over the entire G-space rather than just the
    // R2C grid. This is because it is an integration in the G-space.
    // This implementation fully agrees with the QE result.
    for( Int ig = 0; ig < fft.numGridTotal; ig++ ){
      gkk2 = fft.gkk(ig) * 2.0;
      if( gkk2 > epsDiv ){
        if( screenMu_ > 0.0 ){
          exxDiv_ += std::exp(-exxAlpha * gkk2) / gkk2 * 
            (1.0 - std::exp(-gkk2 / (4.0*screenMu_*screenMu_)));
        }
        else{
          exxDiv_ += std::exp(-exxAlpha * gkk2) / gkk2;
        }
      }
    } // for (ig)

    if( screenMu_ > 0.0 ){
      exxDiv_ += 1.0 / (4.0*screenMu_*screenMu_);
    }
    else{
      exxDiv_ -= exxAlpha;
    }
    exxDiv_ *= 4.0 * PI;


    Int nqq = 100000;
    Real dq = 5.0 / std::sqrt(exxAlpha) / nqq;
    Real aa = 0.0;
    Real qt, qt2;
    for( Int iq = 0; iq < nqq; iq++ ){
      qt = dq * (iq+0.5);
      qt2 = qt*qt;
      if( screenMu_ > 0.0 ){
        aa -= std::exp(-exxAlpha *qt2) * 
          std::exp(-qt2 / (4.0*screenMu_*screenMu_)) * dq;
      }
    }
    aa = aa * 2.0 / PI + 1.0 / std::sqrt(exxAlpha*PI);
    exxDiv_ -= domain_.Volume()*aa;
  }

  if(1){
    statusOFS << " exxDivergenceType_= " << exxDivergenceType_ <<  std::endl;
    statusOFS << " computed exxDiv_ =  " << exxDiv_ << std::endl;
  }


  for( Int ig = 0; ig < numGridTotalR2C; ig++ ){
    gkk2 = fft.gkkR2C(ig) * 2.0;
    if( gkk2 > epsDiv ){
      if( screenMu_ > 0 ){
        // 2.0*pi instead 4.0*pi due to gkk includes a factor of 2
        exxgkkR2C_[ig] = 4.0 * PI / gkk2 * (1.0 - 
            std::exp( -gkk2 / (4.0*screenMu_*screenMu_) ));
      }
      else{
        exxgkkR2C_[ig] = 4.0 * PI / gkk2;
      }
    }
    else{
      exxgkkR2C_[ig] = -exxDiv_;
      if( screenMu_ > 0 ){
        exxgkkR2C_[ig] += PI / (screenMu_*screenMu_);
      }
    }
  } // for (ig)


  for( Int ig = 0; ig < numGridTotal; ig++ ){
    gkk2 = fft.gkk(ig) * 2.0;
    if( gkk2 > epsDiv ){
      if( screenMu_ > 0 ){
        // 2.0*pi instead 4.0*pi due to gkk includes a factor of 2
        exxgkk_[ig] = 4.0 * PI / gkk2 * (1.0 - 
            std::exp( -gkk2 / (4.0*screenMu_*screenMu_) ));
      }
      else{
        exxgkk_[ig] = 4.0 * PI / gkk2;
      }
    }
    else{
      exxgkk_[ig] = -exxDiv_;
      if( screenMu_ > 0 ){
        exxgkk_[ig] += PI / (screenMu_*screenMu_);
      }
    }
  } // for (ig)



  if(1){
    statusOFS << "Hybrid mixing parameter  = " << exxFraction_ << std::endl; 
    statusOFS << "Hybrid screening length = " << screenMu_ << std::endl;
  }


  return ;
}        // -----  end of function KohnSham::InitializeEXX  ----- 

void
KohnSham::SetPhiEXX    (const Spinor& psi, Fourier& fft)
{
  // FIXME collect Psi into a globally shared array in the MPI context.
  const NumTns<Complex>& wavefun = psi.Wavefun();
  Int ntot = wavefun.m();
  Int ncom = wavefun.n();
  Int numStateLocal = wavefun.p();
  Int numStateTotal = this->NumStateTotal();
  Int ntotFine  = fft.domain.NumGridTotalFine();
  Real vol = fft.domain.Volume();

  phiEXX_.Resize( ntot, ncom, numStateLocal );
  SetValue( phiEXX_, Z_ZERO );

  // FIXME Put in a more proper place
  for (Int k=0; k<numStateLocal; k++) {
    for (Int j=0; j<ncom; j++) {

      Real fac = std::sqrt( double(ntot) / vol );
      blas::Copy( ntot, wavefun.VecData(j,k), 1, phiEXX_.VecData(j,k), 1 );
      blas::Scal( ntot, fac, phiEXX_.VecData(j,k), 1 );
    } // for (j)
  } // for (k)


  return ;
}         // -----  end of method KohnSham::SetPhiEXX  ----- 

// This comes from exxenergy2() function in exx.f90 in QE.
Real
KohnSham::CalculateEXXEnergy    ( Spinor& psi, Fourier& fft )
{

  MPI_Barrier(domain_.comm);
  int mpirank;  MPI_Comm_rank(domain_.comm, &mpirank);
  int mpisize;  MPI_Comm_size(domain_.comm, &mpisize);

  Real fockEnergy = 0.0;
  Real fockEnergyLocal = 0.0;

  // Repeat the calculation of Vexx
  // FIXME Will be replaced by the stored VPhi matrix in the new
  // algorithm to reduce the cost, but this should be a new function

  // FIXME Should be combined better with the addition of exchange part in spinor
  NumTns<Complex>& wavefun = psi.Wavefun();

  if( !fft.isInitialized ){
    ErrorHandling("Fourier is not prepared.");
  }
  Index3& numGrid = fft.domain.numGrid;
  Index3& numGridFine = fft.domain.numGridFine;
  Int ntot      = fft.domain.NumGridTotal();
  Int ntotFine  = fft.domain.NumGridTotalFine();
  Int numStateTotal = psi.NumStateTotal();
  Int numStateLocal = psi.NumState();
  Real vol = fft.domain.Volume();
  Int ncom = wavefun.n();
  NumTns<Complex>& phi = phiEXX_;
  Int ncomPhi = phi.n();
  if( ncomPhi != 1 || ncom != 1 ){
    ErrorHandling("Spin polarized case not implemented.");
  }
  Int numStateLocalPhi = phi.p();

  if( fft.domain.NumGridTotal() != ntot ){
    ErrorHandling("Domain size does not match.");
  }

  // Directly use the phiEXX_ and vexxProj_ to calculate the exchange energy
  if( esdfParam.isHybridACE ){
    // temporarily just implement here
    // Directly use projector
    Int numProj = vexxProj_.n();
    Int numStateTotal = this->NumStateTotal();
    Int ntot = psi.NumGridTotal();

    /*
    if(0)
    {
      DblNumMat M(numProj, numStateTotal);

      NumTns<Real>  vexxPsi( ntot, 1, numStateLocalPhi );
      SetValue( vexxPsi, 0.0 );

      blas::Gemm( 'T', 'N', numProj, numStateTotal, ntot, 1.0,
          vexxProj_.Data(), ntot, psi.Wavefun().Data(), ntot, 
          0.0, M.Data(), M.m() );
      // Minus sign comes from that all eigenvalues are negative
      blas::Gemm( 'N', 'N', ntot, numStateTotal, numProj, -1.0,
          vexxProj_.Data(), ntot, M.Data(), numProj,
          0.0, vexxPsi.Data(), ntot );

      for( Int k = 0; k < numStateLocalPhi; k++ ){
        for( Int j = 0; j < ncom; j++ ){
          for( Int ir = 0; ir < ntot; ir++ ){
            fockEnergy += vexxPsi(ir,j,k) * wavefun(ir,j,k) * occupationRate_[psi.WavefunIdx(k)];
          }
        }
      }

    }
    */

    if(1) // For MPI
    {
      Int numStateBlocksize = numStateTotal / mpisize;
      Int ntotBlocksize = ntot / mpisize;

      Int numStateLocal = numStateBlocksize;
      Int ntotLocal = ntotBlocksize;

      if(mpirank < (numStateTotal % mpisize)){
        numStateLocal = numStateBlocksize + 1;
      }

      if(mpirank < (ntot % mpisize)){
        ntotLocal = ntotBlocksize + 1;
      }

      CpxNumMat psiCol( ntot, numStateLocal );
      SetValue( psiCol, Z_ZERO );

      CpxNumMat psiRow( ntotLocal, numStateTotal );
      SetValue( psiRow, Z_ZERO );

      CpxNumMat vexxProjCol( ntot, numStateLocal );
      SetValue( vexxProjCol, Z_ZERO );

      CpxNumMat vexxProjRow( ntotLocal, numStateTotal );
      SetValue( vexxProjRow, Z_ZERO );

      CpxNumMat vexxPsiCol( ntot, numStateLocal );
      SetValue( vexxPsiCol, Z_ZERO );

      CpxNumMat vexxPsiRow( ntotLocal, numStateTotal );
      SetValue( vexxPsiRow, Z_ZERO );

      lapack::Lacpy( 'A', ntot, numStateLocal, psi.Wavefun().Data(), ntot, psiCol.Data(), ntot );
      lapack::Lacpy( 'A', ntot, numStateLocal, vexxProj_.Data(), ntot, vexxProjCol.Data(), ntot );

      AlltoallForward (psiCol, psiRow, domain_.comm);
      AlltoallForward (vexxProjCol, vexxProjRow, domain_.comm);

      CpxNumMat MTemp( numStateTotal, numStateTotal );
      SetValue( MTemp, Z_ZERO );

      blas::Gemm( 'C', 'N', numStateTotal, numStateTotal, ntotLocal,
          1.0, vexxProjRow.Data(), ntotLocal, 
          psiRow.Data(), ntotLocal, 0.0,
          MTemp.Data(), numStateTotal );

      CpxNumMat M(numStateTotal, numStateTotal);
      SetValue( M, Z_ZERO );

      MPI_Allreduce( MTemp.Data(), M.Data(), 2*numStateTotal * numStateTotal, MPI_DOUBLE, MPI_SUM, domain_.comm );

      blas::Gemm( 'N', 'N', ntotLocal, numStateTotal, numStateTotal, -1.0,
          vexxProjRow.Data(), ntotLocal, M.Data(), numStateTotal,
          0.0, vexxPsiRow.Data(), ntotLocal );

      AlltoallBackward (vexxPsiRow, vexxPsiCol, domain_.comm);

      fockEnergy = 0.0;
      fockEnergyLocal = 0.0;

      for( Int k = 0; k < numStateLocal; k++ ){
        for( Int j = 0; j < ncom; j++ ){
          for( Int ir = 0; ir < ntot; ir++ ){
            fockEnergyLocal += (vexxPsiCol(ir,k) * std::conj(wavefun(ir,j,k))).real() * occupationRate_[psi.WavefunIdx(k)];
          }
        }
      }
      mpi::Allreduce( &fockEnergyLocal, &fockEnergy, 1, MPI_SUM, domain_.comm );
    } //if(1) 
  }
  else
  {
    NumTns<Complex>  vexxPsi( ntot, 1, numStateLocalPhi );
    SetValue( vexxPsi, Z_ZERO );
    psi.AddMultSpinorEXX( fft, phiEXX_, exxgkk_, 
        exxFraction_,  numSpin_, occupationRate_, 
        vexxPsi );
    // Compute the exchange energy:
    // Note: no additional normalization factor due to the
    // normalization rule of psi, NOT phi!!
    fockEnergy = 0.0;
    fockEnergyLocal = 0.0;
    for( Int k = 0; k < numStateLocalPhi; k++ ){
      for( Int j = 0; j < ncom; j++ ){
        for( Int ir = 0; ir < ntot; ir++ ){
          fockEnergyLocal += (vexxPsi(ir,j,k) * std::conj(wavefun(ir,j,k))).real() * occupationRate_[psi.WavefunIdx(k)];
        }
      }
    }
    mpi::Allreduce( &fockEnergyLocal, &fockEnergy, 1, MPI_SUM, domain_.comm );
  }


  return fockEnergy;
}         // -----  end of method KohnSham::CalculateEXXEnergy  ----- 

void
KohnSham::CalculateVexxACE ( Spinor& psi, Fourier& fft )
{
  // This assumes SetPhiEXX has been called so that phiEXX and psi
  // contain the same information. 

  // Since this is a projector, it should be done on the COARSE grid,
  // i.e. to the wavefunction directly

  MPI_Barrier(domain_.comm);
  int mpirank;  MPI_Comm_rank(domain_.comm, &mpirank);
  int mpisize;  MPI_Comm_size(domain_.comm, &mpisize);

  // Only works for single processor
  Int ntot      = fft.domain.NumGridTotal();
  Int ntotFine  = fft.domain.NumGridTotalFine();
  Int numStateTotal = psi.NumStateTotal();
  Int numStateLocal = psi.NumState();
  NumTns<Complex>  vexxPsi( ntot, 1, numStateLocal );

  // VexxPsi = V_{exx}*Phi.
  SetValue( vexxPsi, Z_ZERO );
  psi.AddMultSpinorEXX( fft, phiEXX_, exxgkk_,
      exxFraction_,  numSpin_, occupationRate_, vexxPsi );

  // Implementation based on SVD
  CpxNumMat  M(numStateTotal, numStateTotal);

  /*
  if(0){
    // FIXME
    Real SVDTolerance = 1e-4;
    // M = Phi'*vexxPsi
    blas::Gemm( 'T', 'N', numStateTotal, numStateTotal, ntot, 
        1.0, psi.Wavefun().Data(), ntot, vexxPsi.Data(), ntot,
        0.0, M.Data(), numStateTotal );

    DblNumMat  U( numStateTotal, numStateTotal );
    DblNumMat VT( numStateTotal, numStateTotal );
    DblNumVec  S( numStateTotal );
    SetValue( S, 0.0 );

    lapack::QRSVD( numStateTotal, numStateTotal, M.Data(), numStateTotal,
        S.Data(), U.Data(), U.m(), VT.Data(), VT.m() );


    for( Int g = 0; g < numStateTotal; g++ ){
      S[g] = std::sqrt( S[g] );
    }

    Int rankM = 0;
    for( Int g = 0; g < numStateTotal; g++ ){
      if( S[g] / S[0] > SVDTolerance ){
        rankM++;
      }
    }
    statusOFS << "rank of Phi'*VPhi matrix = " << rankM << std::endl;
    for( Int g = 0; g < rankM; g++ ){
      blas::Scal( numStateTotal, 1.0 / S[g], U.VecData(g), 1 );
    }

    vexxProj_.Resize( ntot, rankM );
    blas::Gemm( 'N', 'N', ntot, rankM, numStateTotal, 1.0, 
        vexxPsi.Data(), ntot, U.Data(), numStateTotal, 0.0,
        vexxProj_.Data(), ntot );
  }

  // Implementation based on Cholesky
  if(0){
    // M = -Phi'*vexxPsi. The minus sign comes from vexx is a negative
    // semi-definite matrix.
    blas::Gemm( 'T', 'N', numStateTotal, numStateTotal, ntot, 
        -1.0, psi.Wavefun().Data(), ntot, vexxPsi.Data(), ntot,
        0.0, M.Data(), numStateTotal );

    lapack::Potrf('L', numStateTotal, M.Data(), numStateTotal);

    blas::Trsm( 'R', 'L', 'T', 'N', ntot, numStateTotal, 1.0, 
        M.Data(), numStateTotal, vexxPsi.Data(), ntot );

    vexxProj_.Resize( ntot, numStateTotal );
    blas::Copy( ntot * numStateTotal, vexxPsi.Data(), 1, vexxProj_.Data(), 1 );
  }
  */

  if(1){ //For MPI

    // Convert the column partition to row partition
    Int numStateBlocksize = numStateTotal / mpisize;
    Int ntotBlocksize = ntot / mpisize;

    Int numStateLocal = numStateBlocksize;
    Int ntotLocal = ntotBlocksize;

    if(mpirank < (numStateTotal % mpisize)){
      numStateLocal = numStateBlocksize + 1;
    }

    if(mpirank < (ntot % mpisize)){
      ntotLocal = ntotBlocksize + 1;
    }

    CpxNumMat localPsiCol( ntot, numStateLocal );
    SetValue( localPsiCol, Z_ZERO );

    CpxNumMat localVexxPsiCol( ntot, numStateLocal );
    SetValue( localVexxPsiCol, Z_ZERO );

    CpxNumMat localPsiRow( ntotLocal, numStateTotal );
    SetValue( localPsiRow, Z_ZERO );

    CpxNumMat localVexxPsiRow( ntotLocal, numStateTotal );
    SetValue( localVexxPsiRow, Z_ZERO );

    // Initialize
    lapack::Lacpy( 'A', ntot, numStateLocal, psi.Wavefun().Data(), ntot, localPsiCol.Data(), ntot );
    lapack::Lacpy( 'A', ntot, numStateLocal, vexxPsi.Data(), ntot, localVexxPsiCol.Data(), ntot );

    AlltoallForward (localPsiCol, localPsiRow, domain_.comm);
    AlltoallForward (localVexxPsiCol, localVexxPsiRow, domain_.comm);

    CpxNumMat MTemp( numStateTotal, numStateTotal );
    SetValue( MTemp, Z_ZERO );

    blas::Gemm( 'C', 'N', numStateTotal, numStateTotal, ntotLocal,
        -1.0, localPsiRow.Data(), ntotLocal, 
        localVexxPsiRow.Data(), ntotLocal, 0.0,
        MTemp.Data(), numStateTotal );

    SetValue( M, Z_ZERO );
    MPI_Allreduce( MTemp.Data(), M.Data(), numStateTotal * numStateTotal*2, MPI_DOUBLE, MPI_SUM, domain_.comm );

    if ( mpirank == 0) {
      lapack::Potrf('L', numStateTotal, M.Data(), numStateTotal);
    }

    MPI_Bcast(M.Data(), 2*numStateTotal * numStateTotal, MPI_DOUBLE, 0, domain_.comm);

    blas::Trsm( 'R', 'L', 'C', 'N', ntotLocal, numStateTotal, 1.0, 
        M.Data(), numStateTotal, localVexxPsiRow.Data(), ntotLocal );

    vexxProj_.Resize( ntot, numStateLocal );

    AlltoallBackward (localVexxPsiRow, vexxProj_, domain_.comm);
  } //if(1)

  // Sanity check. For debugging only
  //  if(0){
  //  // Make sure U and VT are the same. Should be an identity matrix
  //    blas::Gemm( 'N', 'N', numStateTotal, numStateTotal, numStateTotal, 1.0, 
  //        VT.Data(), numStateTotal, U.Data(), numStateTotal, 0.0,
  //        M.Data(), numStateTotal );
  //    statusOFS << "M = " << M << std::endl;
  //
  //    NumTns<Real> vpsit = psi.Wavefun();
  //    Int numProj = rankM;
  //    DblNumMat Mt(numProj, numStateTotal);
  //    
  //    blas::Gemm( 'T', 'N', numProj, numStateTotal, ntot, 1.0,
  //        vexxProj_.Data(), ntot, psi.Wavefun().Data(), ntot, 
  //        0.0, Mt.Data(), Mt.m() );
  //    // Minus sign comes from that all eigenvalues are negative
  //    blas::Gemm( 'N', 'N', ntot, numStateTotal, numProj, -1.0,
  //        vexxProj_.Data(), ntot, Mt.Data(), numProj,
  //        0.0, vpsit.Data(), ntot );
  //
  //    for( Int k = 0; k < numStateTotal; k++ ){
  //      Real norm = 0.0;
  //      for( Int ir = 0; ir < ntot; ir++ ){
  //        norm = norm + std::pow(vexxPsi(ir,0,k) - vpsit(ir,0,k), 2.0);
  //      }
  //      statusOFS << "Diff of vexxPsi " << std::sqrt(norm) << std::endl;
  //    }
  //  }


  return ;
}         // -----  end of method KohnSham::CalculateVexxACE  ----- 

//2019/10/30
//add Complex CalculateVexxACEDF  -----by lijl
void
KohnSham::CalculateVexxACEDF ( Spinor& psi, Fourier& fft, bool isFixColumnDF )
{
  // This assumes SetPhiEXX has been called so that phiEXX and psi
  // contain the same information. 

  // Since this is a projector, it should be done on the COARSE grid,
  // i.e. to the wavefunction directly

  MPI_Barrier(domain_.comm);
  int mpirank;  MPI_Comm_rank(domain_.comm, &mpirank);
  int mpisize;  MPI_Comm_size(domain_.comm, &mpisize);

  // Only works for single processor
  Int ntot      = fft.domain.NumGridTotal();
  Int ntotFine  = fft.domain.NumGridTotalFine();
  Int numStateTotal = psi.NumStateTotal();
  Int numStateLocal = psi.NumState();
  NumTns<Complex>  vexxPsi( ntot, 1, numStateLocal );

  // VexxPsi = V_{exx}*Phi.
  CpxNumMat  M(numStateTotal, numStateTotal);
  SetValue( vexxPsi, Z_ZERO );
  SetValue( M, Z_ZERO );
  // M = -Phi'*vexxPsi. The minus sign comes from vexx is a negative
  // semi-definite matrix.
  psi.AddMultSpinorEXXDF7( fft, phiEXX_, exxgkk_, exxFraction_,  numSpin_, 
      occupationRate_, hybridDFType_, hybridDFKmeansWFType_, hybridDFKmeansWFAlpha_, hybridDFKmeansTolerance_, 
      hybridDFKmeansMaxIter_, hybridDFNumMu_, hybridDFNumGaussianRandom_,
      hybridDFNumProcScaLAPACK_, hybridDFTolerance_, BlockSizeScaLAPACK_,
      vexxPsi, M, isFixColumnDF );
  
  // Implementation based on Cholesky
  if(0){
    lapack::Potrf('L', numStateTotal, M.Data(), numStateTotal);

    blas::Trsm( 'R', 'L', 'C', 'N', ntot, numStateTotal, 1.0, 
        M.Data(), numStateTotal, vexxPsi.Data(), ntot );

    vexxProj_.Resize( ntot, numStateTotal );
    blas::Copy( ntot * numStateTotal, vexxPsi.Data(), 1, vexxProj_.Data(), 1 );
  }

  if(1){ //For MPI

    // Convert the column partition to row partition
    Int numStateBlocksize = numStateTotal / mpisize;
    Int ntotBlocksize = ntot / mpisize;

    Int numStateLocal = numStateBlocksize;
    Int ntotLocal = ntotBlocksize;

    if(mpirank < (numStateTotal % mpisize)){
      numStateLocal = numStateBlocksize + 1;
    }

    if(mpirank < (ntot % mpisize)){
      ntotLocal = ntotBlocksize + 1;
    }

    CpxNumMat localVexxPsiCol( ntot, numStateLocal );
    SetValue( localVexxPsiCol, Z_ZERO );

    CpxNumMat localVexxPsiRow( ntotLocal, numStateTotal );
    SetValue( localVexxPsiRow, Z_ZERO );

    // Initialize
    lapack::Lacpy( 'A', ntot, numStateLocal, vexxPsi.Data(), ntot, localVexxPsiCol.Data(), ntot );

    AlltoallForward (localVexxPsiCol, localVexxPsiRow, domain_.comm);

    if ( mpirank == 0) {
      statusOFS << "lijl potrf 20200523"<<  std::endl;
      lapack::Potrf('L', numStateTotal, M.Data(), numStateTotal);
    }

    MPI_Bcast(M.Data(), 2*numStateTotal * numStateTotal, MPI_DOUBLE, 0, domain_.comm);

    blas::Trsm( 'R', 'L', 'C', 'N', ntotLocal, numStateTotal, 1.0, 
        M.Data(), numStateTotal, localVexxPsiRow.Data(), ntotLocal );

    vexxProj_.Resize( ntot, numStateLocal );
//20200422
    statusOFS << "lijl calculateVexxACEDF "<<  std::endl;
//
    AlltoallBackward (localVexxPsiRow, vexxProj_, domain_.comm);
  } //if(1)
  return ;
}         // -----  end of method KohnSham::CalculateVexxACEDF  ----- 

#else


void KohnSham::InitializeEXX ( Real ecutWavefunction, Fourier& fft )
{
  const Real epsDiv = 1e-8;

  Real EPS = 1e-14;
  isEXXActive_ = false;

  Int numGridTotalR2C = fft.numGridTotalR2C;
  exxgkkR2C_.Resize(numGridTotalR2C);
  SetValue( exxgkkR2C_, 0.0 );

  // extra 2.0 factor for ecutWavefunction compared to QE due to unit difference
  // tpiba2 in QE is just a unit for G^2. Do not include it here
  Real exxAlpha = 10.0 / (ecutWavefunction * 2.0);

  // Gygi-Baldereschi regularization. Currently set to zero and compare
  // with QE without the regularization 
  // Set exxdiv_treatment to "none"
  // NOTE: I do not quite understand the detailed derivation
  // Compute the divergent term for G=0
  Real gkk2;
  if(exxDivergenceType_ == 0){
    exxDiv_ = 0.0;
  }
  else if (exxDivergenceType_ == 1){
    exxDiv_ = 0.0;
    // no q-point
    // NOTE: Compared to the QE implementation, it is easier to do below.
    // Do the integration over the entire G-space rather than just the
    // R2C grid. This is because it is an integration in the G-space.
    // This implementation fully agrees with the QE result.
    for( Int ig = 0; ig < fft.numGridTotal; ig++ ){
      gkk2 = fft.gkk(ig) * 2.0;
      if( gkk2 > epsDiv ){
        if( screenMu_ > 0.0 ){
          exxDiv_ += std::exp(-exxAlpha * gkk2) / gkk2 * 
            (1.0 - std::exp(-gkk2 / (4.0*screenMu_*screenMu_)));
        }
        else{
          exxDiv_ += std::exp(-exxAlpha * gkk2) / gkk2;
        }
      }
    } // for (ig)

    if( screenMu_ > 0.0 ){
      exxDiv_ += 1.0 / (4.0*screenMu_*screenMu_);
    }
    else{
      exxDiv_ -= exxAlpha;
    }
    exxDiv_ *= 4.0 * PI;

    Real aa = 0.0;

    if( screenMu_ > 0.0 ){
      aa = + 1.0 / std::sqrt(exxAlpha*PI) -
          1.0 / std::sqrt(PI*(exxAlpha + 1.0 / (4.0*screenMu_*screenMu_ )));
    }
    else{
      aa = + 1.0 / std::sqrt(exxAlpha*PI);
    }
    exxDiv_ -= domain_.Volume()*aa;
  }

  statusOFS << "computed exxDiv_ = " << exxDiv_ << std::endl;

  for( Int ig = 0; ig < numGridTotalR2C; ig++ ){
    gkk2 = fft.gkkR2C(ig) * 2.0;
    if( gkk2 > epsDiv ){
      if( screenMu_ > 0 ){
        // 2.0*pi instead 4.0*pi due to gkk includes a factor of 2
        exxgkkR2C_[ig] = 4.0 * PI / gkk2 * (1.0 - 
            std::exp( -gkk2 / (4.0*screenMu_*screenMu_) ));
      }
      else{
        exxgkkR2C_[ig] = 4.0 * PI / gkk2;
      }
    }
    else{
      exxgkkR2C_[ig] = -exxDiv_;
      if( screenMu_ > 0 ){
        exxgkkR2C_[ig] += PI / (screenMu_*screenMu_);
      }
    }
  } // for (ig)

  if(1){
    statusOFS << "Hybrid mixing parameter  = " << exxFraction_ << std::endl; 
    statusOFS << "Hybrid screening length = " << screenMu_ << std::endl;
  }

  return ;
}        // -----  end of function KohnSham::InitializeEXX  ----- 


#endif

#ifndef _COMPLEX_
void
KohnSham::SetPhiEXX    (const Spinor& psi, Fourier& fft)
{
  // FIXME collect Psi into a globally shared array in the MPI context.
  const NumTns<Real>& wavefun = psi.Wavefun();
  Int ntot = wavefun.m();
  Int ncom = wavefun.n();
  Int numStateLocal = wavefun.p();
  Int numStateTotal = this->NumStateTotal();
  Int ntotFine  = fft.domain.NumGridTotalFine();
  Real vol = fft.domain.Volume();

  phiEXX_.Resize( ntot, ncom, numStateLocal );
  SetValue( phiEXX_, 0.0 );

  // FIXME Put in a more proper place
  for (Int k=0; k<numStateLocal; k++) {
    for (Int j=0; j<ncom; j++) {

      Real fac = std::sqrt( double(ntot) / vol );
      blas::Copy( ntot, wavefun.VecData(j,k), 1, phiEXX_.VecData(j,k), 1 );
      blas::Scal( ntot, fac, phiEXX_.VecData(j,k), 1 );

      if(0){
        DblNumVec psiTemp(ntot);
        blas::Copy( ntot, phiEXX_.VecData(j,k), 1, psiTemp.Data(), 1 );
        statusOFS << "int (phi^2) dx = " << Energy(psiTemp)*vol / double(ntot) << std::endl;
      }

    } // for (j)
  } // for (k)


  return ;
}         // -----  end of method KohnSham::SetPhiEXX  ----- 

void
KohnSham::CalculateVexxACE ( Spinor& psi, Fourier& fft )
{
  // This assumes SetPhiEXX has been called so that phiEXX and psi
  // contain the same information. 

  // Since this is a projector, it should be done on the COARSE grid,
  // i.e. to the wavefunction directly

  MPI_Barrier(domain_.comm);
  int mpirank;  MPI_Comm_rank(domain_.comm, &mpirank);
  int mpisize;  MPI_Comm_size(domain_.comm, &mpisize);

  // Only works for single processor
  Int ntot      = fft.domain.NumGridTotal();
  Int ntotFine  = fft.domain.NumGridTotalFine();
  Int numStateTotal = psi.NumStateTotal();
  Int numStateLocal = psi.NumState();
  NumTns<Real>  vexxPsi( ntot, 1, numStateLocal );

  // VexxPsi = V_{exx}*Phi.
  SetValue( vexxPsi, 0.0 );
  psi.AddMultSpinorEXX( fft, phiEXX_, exxgkkR2C_,
      exxFraction_,  numSpin_, occupationRate_, vexxPsi );

  // Implementation based on SVD
  DblNumMat  M(numStateTotal, numStateTotal);

  if(0){
    // FIXME
    Real SVDTolerance = 1e-4;
    // M = Phi'*vexxPsi
    blas::Gemm( 'T', 'N', numStateTotal, numStateTotal, ntot, 
        1.0, psi.Wavefun().Data(), ntot, vexxPsi.Data(), ntot,
        0.0, M.Data(), numStateTotal );

    DblNumMat  U( numStateTotal, numStateTotal );
    DblNumMat VT( numStateTotal, numStateTotal );
    DblNumVec  S( numStateTotal );
    SetValue( S, 0.0 );

    lapack::QRSVD( numStateTotal, numStateTotal, M.Data(), numStateTotal,
        S.Data(), U.Data(), U.m(), VT.Data(), VT.m() );


    for( Int g = 0; g < numStateTotal; g++ ){
      S[g] = std::sqrt( S[g] );
    }

    Int rankM = 0;
    for( Int g = 0; g < numStateTotal; g++ ){
      if( S[g] / S[0] > SVDTolerance ){
        rankM++;
      }
    }
    statusOFS << "rank of Phi'*VPhi matrix = " << rankM << std::endl;
    for( Int g = 0; g < rankM; g++ ){
      blas::Scal( numStateTotal, 1.0 / S[g], U.VecData(g), 1 );
    }

    vexxProj_.Resize( ntot, rankM );
    blas::Gemm( 'N', 'N', ntot, rankM, numStateTotal, 1.0, 
        vexxPsi.Data(), ntot, U.Data(), numStateTotal, 0.0,
        vexxProj_.Data(), ntot );
  }

  // Implementation based on Cholesky
  if(0){
    // M = -Phi'*vexxPsi. The minus sign comes from vexx is a negative
    // semi-definite matrix.
    blas::Gemm( 'T', 'N', numStateTotal, numStateTotal, ntot, 
        -1.0, psi.Wavefun().Data(), ntot, vexxPsi.Data(), ntot,
        0.0, M.Data(), numStateTotal );

    lapack::Potrf('L', numStateTotal, M.Data(), numStateTotal);

    blas::Trsm( 'R', 'L', 'T', 'N', ntot, numStateTotal, 1.0, 
        M.Data(), numStateTotal, vexxPsi.Data(), ntot );

    vexxProj_.Resize( ntot, numStateTotal );
    blas::Copy( ntot * numStateTotal, vexxPsi.Data(), 1, vexxProj_.Data(), 1 );
  }

  if(1){ //For MPI

    // Convert the column partition to row partition
    Int numStateBlocksize = numStateTotal / mpisize;
    Int ntotBlocksize = ntot / mpisize;

    Int numStateLocal = numStateBlocksize;
    Int ntotLocal = ntotBlocksize;

    if(mpirank < (numStateTotal % mpisize)){
      numStateLocal = numStateBlocksize + 1;
    }

    if(mpirank < (ntot % mpisize)){
      ntotLocal = ntotBlocksize + 1;
    }

    DblNumMat localPsiCol( ntot, numStateLocal );
    SetValue( localPsiCol, 0.0 );

    DblNumMat localVexxPsiCol( ntot, numStateLocal );
    SetValue( localVexxPsiCol, 0.0 );

    DblNumMat localPsiRow( ntotLocal, numStateTotal );
    SetValue( localPsiRow, 0.0 );

    DblNumMat localVexxPsiRow( ntotLocal, numStateTotal );
    SetValue( localVexxPsiRow, 0.0 );

    // Initialize
    lapack::Lacpy( 'A', ntot, numStateLocal, psi.Wavefun().Data(), ntot, localPsiCol.Data(), ntot );
    lapack::Lacpy( 'A', ntot, numStateLocal, vexxPsi.Data(), ntot, localVexxPsiCol.Data(), ntot );

    AlltoallForward (localPsiCol, localPsiRow, domain_.comm);
    AlltoallForward (localVexxPsiCol, localVexxPsiRow, domain_.comm);

    DblNumMat MTemp( numStateTotal, numStateTotal );
    SetValue( MTemp, 0.0 );

    blas::Gemm( 'T', 'N', numStateTotal, numStateTotal, ntotLocal,
        -1.0, localPsiRow.Data(), ntotLocal, 
        localVexxPsiRow.Data(), ntotLocal, 0.0,
        MTemp.Data(), numStateTotal );

    SetValue( M, 0.0 );
    MPI_Allreduce( MTemp.Data(), M.Data(), numStateTotal * numStateTotal, MPI_DOUBLE, MPI_SUM, domain_.comm );

    if ( mpirank == 0) {
      lapack::Potrf('L', numStateTotal, M.Data(), numStateTotal);
    }

    MPI_Bcast(M.Data(), numStateTotal * numStateTotal, MPI_DOUBLE, 0, domain_.comm);

    blas::Trsm( 'R', 'L', 'T', 'N', ntotLocal, numStateTotal, 1.0, 
        M.Data(), numStateTotal, localVexxPsiRow.Data(), ntotLocal );

    vexxProj_.Resize( ntot, numStateLocal );

    AlltoallBackward (localVexxPsiRow, vexxProj_, domain_.comm);
  } //if(1)

  // Sanity check. For debugging only
  //  if(0){
  //  // Make sure U and VT are the same. Should be an identity matrix
  //    blas::Gemm( 'N', 'N', numStateTotal, numStateTotal, numStateTotal, 1.0, 
  //        VT.Data(), numStateTotal, U.Data(), numStateTotal, 0.0,
  //        M.Data(), numStateTotal );
  //    statusOFS << "M = " << M << std::endl;
  //
  //    NumTns<Real> vpsit = psi.Wavefun();
  //    Int numProj = rankM;
  //    DblNumMat Mt(numProj, numStateTotal);
  //    
  //    blas::Gemm( 'T', 'N', numProj, numStateTotal, ntot, 1.0,
  //        vexxProj_.Data(), ntot, psi.Wavefun().Data(), ntot, 
  //        0.0, Mt.Data(), Mt.m() );
  //    // Minus sign comes from that all eigenvalues are negative
  //    blas::Gemm( 'N', 'N', ntot, numStateTotal, numProj, -1.0,
  //        vexxProj_.Data(), ntot, Mt.Data(), numProj,
  //        0.0, vpsit.Data(), ntot );
  //
  //    for( Int k = 0; k < numStateTotal; k++ ){
  //      Real norm = 0.0;
  //      for( Int ir = 0; ir < ntot; ir++ ){
  //        norm = norm + std::pow(vexxPsi(ir,0,k) - vpsit(ir,0,k), 2.0);
  //      }
  //      statusOFS << "Diff of vexxPsi " << std::sqrt(norm) << std::endl;
  //    }
  //  }


  return ;
}         // -----  end of method KohnSham::CalculateVexxACE  ----- 

void
KohnSham::CalculateVexxACEDF ( Spinor& psi, Fourier& fft, bool isFixColumnDF )
{
  // This assumes SetPhiEXX has been called so that phiEXX and psi
  // contain the same information. 

  // Since this is a projector, it should be done on the COARSE grid,
  // i.e. to the wavefunction directly

  MPI_Barrier(domain_.comm);
  int mpirank;  MPI_Comm_rank(domain_.comm, &mpirank);
  int mpisize;  MPI_Comm_size(domain_.comm, &mpisize);

  // Only works for single processor
  Int ntot      = fft.domain.NumGridTotal();
  Int ntotFine  = fft.domain.NumGridTotalFine();
  Int numStateTotal = psi.NumStateTotal();
  Int numStateLocal = psi.NumState();
  NumTns<Real>  vexxPsi( ntot, 1, numStateLocal );

  // VexxPsi = V_{exx}*Phi.
  DblNumMat  M(numStateTotal, numStateTotal);
  SetValue( vexxPsi, 0.0 );
  SetValue( M, 0.0 );
  // M = -Phi'*vexxPsi. The minus sign comes from vexx is a negative
  // semi-definite matrix.
  psi.AddMultSpinorEXXDF7( fft, phiEXX_, exxgkkR2C_, exxFraction_,  numSpin_, 
      occupationRate_, hybridDFType_, hybridDFKmeansTolerance_, 
      hybridDFKmeansMaxIter_, hybridDFNumMu_, hybridDFNumGaussianRandom_,
      hybridDFNumProcScaLAPACK_, hybridDFTolerance_, BlockSizeScaLAPACK_,
      vexxPsi, M, isFixColumnDF );
  
  // Implementation based on Cholesky
  if(0){
    lapack::Potrf('L', numStateTotal, M.Data(), numStateTotal);

    blas::Trsm( 'R', 'L', 'T', 'N', ntot, numStateTotal, 1.0, 
        M.Data(), numStateTotal, vexxPsi.Data(), ntot );

    vexxProj_.Resize( ntot, numStateTotal );
    blas::Copy( ntot * numStateTotal, vexxPsi.Data(), 1, vexxProj_.Data(), 1 );
  }

  if(1){ //For MPI

    // Convert the column partition to row partition
    Int numStateBlocksize = numStateTotal / mpisize;
    Int ntotBlocksize = ntot / mpisize;

    Int numStateLocal = numStateBlocksize;
    Int ntotLocal = ntotBlocksize;

    if(mpirank < (numStateTotal % mpisize)){
      numStateLocal = numStateBlocksize + 1;
    }

    if(mpirank < (ntot % mpisize)){
      ntotLocal = ntotBlocksize + 1;
    }

    DblNumMat localVexxPsiCol( ntot, numStateLocal );
    SetValue( localVexxPsiCol, 0.0 );

    DblNumMat localVexxPsiRow( ntotLocal, numStateTotal );
    SetValue( localVexxPsiRow, 0.0 );

    // Initialize
    lapack::Lacpy( 'A', ntot, numStateLocal, vexxPsi.Data(), ntot, localVexxPsiCol.Data(), ntot );

    AlltoallForward (localVexxPsiCol, localVexxPsiRow, domain_.comm);

    if ( mpirank == 0) {
      lapack::Potrf('L', numStateTotal, M.Data(), numStateTotal);
    }

    MPI_Bcast(M.Data(), numStateTotal * numStateTotal, MPI_DOUBLE, 0, domain_.comm);

    blas::Trsm( 'R', 'L', 'T', 'N', ntotLocal, numStateTotal, 1.0, 
        M.Data(), numStateTotal, localVexxPsiRow.Data(), ntotLocal );

    vexxProj_.Resize( ntot, numStateLocal );

    AlltoallBackward (localVexxPsiRow, vexxProj_, domain_.comm);
  } //if(1)
  return ;
}         // -----  end of method KohnSham::CalculateVexxACEDF  ----- 


// This comes from exxenergy2() function in exx.f90 in QE.
Real
KohnSham::CalculateEXXEnergy    ( Spinor& psi, Fourier& fft )
{

  MPI_Barrier(domain_.comm);
  int mpirank;  MPI_Comm_rank(domain_.comm, &mpirank);
  int mpisize;  MPI_Comm_size(domain_.comm, &mpisize);

  Real fockEnergy = 0.0;
  Real fockEnergyLocal = 0.0;

  // Repeat the calculation of Vexx
  // FIXME Will be replaced by the stored VPhi matrix in the new
  // algorithm to reduce the cost, but this should be a new function

  // FIXME Should be combined better with the addition of exchange part in spinor
  NumTns<Real>& wavefun = psi.Wavefun();

  if( !fft.isInitialized ){
    ErrorHandling("Fourier is not prepared.");
  }
  Index3& numGrid = fft.domain.numGrid;
  Index3& numGridFine = fft.domain.numGridFine;
  Int ntot      = fft.domain.NumGridTotal();
  Int ntotFine  = fft.domain.NumGridTotalFine();
  Int numStateTotal = psi.NumStateTotal();
  Int numStateLocal = psi.NumState();
  Real vol = fft.domain.Volume();
  Int ncom = wavefun.n();
  NumTns<Real>& phi = phiEXX_;
  Int ncomPhi = phi.n();
  if( ncomPhi != 1 || ncom != 1 ){
    ErrorHandling("Spin polarized case not implemented.");
  }
  Int numStateLocalPhi = phi.p();

  if( fft.domain.NumGridTotal() != ntot ){
    ErrorHandling("Domain size does not match.");
  }

  // Directly use the phiEXX_ and vexxProj_ to calculate the exchange energy
  if( esdfParam.isHybridACE ){
    // temporarily just implement here
    // Directly use projector
    Int numProj = vexxProj_.n();
    Int numStateTotal = this->NumStateTotal();
    Int ntot = psi.NumGridTotal();

    if(0)
    {
      DblNumMat M(numProj, numStateTotal);

      NumTns<Real>  vexxPsi( ntot, 1, numStateLocalPhi );
      SetValue( vexxPsi, 0.0 );

      blas::Gemm( 'T', 'N', numProj, numStateTotal, ntot, 1.0,
          vexxProj_.Data(), ntot, psi.Wavefun().Data(), ntot, 
          0.0, M.Data(), M.m() );
      // Minus sign comes from that all eigenvalues are negative
      blas::Gemm( 'N', 'N', ntot, numStateTotal, numProj, -1.0,
          vexxProj_.Data(), ntot, M.Data(), numProj,
          0.0, vexxPsi.Data(), ntot );

      for( Int k = 0; k < numStateLocalPhi; k++ ){
        for( Int j = 0; j < ncom; j++ ){
          for( Int ir = 0; ir < ntot; ir++ ){
            fockEnergy += vexxPsi(ir,j,k) * wavefun(ir,j,k) * occupationRate_[psi.WavefunIdx(k)];
          }
        }
      }

    }

    if(1) // For MPI
    {
      Int numStateBlocksize = numStateTotal / mpisize;
      Int ntotBlocksize = ntot / mpisize;

      Int numStateLocal = numStateBlocksize;
      Int ntotLocal = ntotBlocksize;

      if(mpirank < (numStateTotal % mpisize)){
        numStateLocal = numStateBlocksize + 1;
      }

      if(mpirank < (ntot % mpisize)){
        ntotLocal = ntotBlocksize + 1;
      }

      DblNumMat psiCol( ntot, numStateLocal );
      SetValue( psiCol, 0.0 );

      DblNumMat psiRow( ntotLocal, numStateTotal );
      SetValue( psiRow, 0.0 );

      DblNumMat vexxProjCol( ntot, numStateLocal );
      SetValue( vexxProjCol, 0.0 );

      DblNumMat vexxProjRow( ntotLocal, numStateTotal );
      SetValue( vexxProjRow, 0.0 );

      DblNumMat vexxPsiCol( ntot, numStateLocal );
      SetValue( vexxPsiCol, 0.0 );

      DblNumMat vexxPsiRow( ntotLocal, numStateTotal );
      SetValue( vexxPsiRow, 0.0 );

      lapack::Lacpy( 'A', ntot, numStateLocal, psi.Wavefun().Data(), ntot, psiCol.Data(), ntot );
      lapack::Lacpy( 'A', ntot, numStateLocal, vexxProj_.Data(), ntot, vexxProjCol.Data(), ntot );

      AlltoallForward (psiCol, psiRow, domain_.comm);
      AlltoallForward (vexxProjCol, vexxProjRow, domain_.comm);

      DblNumMat MTemp( numStateTotal, numStateTotal );
      SetValue( MTemp, 0.0 );

      blas::Gemm( 'T', 'N', numStateTotal, numStateTotal, ntotLocal,
          1.0, vexxProjRow.Data(), ntotLocal, 
          psiRow.Data(), ntotLocal, 0.0,
          MTemp.Data(), numStateTotal );

      DblNumMat M(numStateTotal, numStateTotal);
      SetValue( M, 0.0 );

      MPI_Allreduce( MTemp.Data(), M.Data(), numStateTotal * numStateTotal, MPI_DOUBLE, MPI_SUM, domain_.comm );

      blas::Gemm( 'N', 'N', ntotLocal, numStateTotal, numStateTotal, -1.0,
          vexxProjRow.Data(), ntotLocal, M.Data(), numStateTotal,
          0.0, vexxPsiRow.Data(), ntotLocal );

      AlltoallBackward (vexxPsiRow, vexxPsiCol, domain_.comm);

      fockEnergy = 0.0;
      fockEnergyLocal = 0.0;

      for( Int k = 0; k < numStateLocal; k++ ){
        for( Int j = 0; j < ncom; j++ ){
          for( Int ir = 0; ir < ntot; ir++ ){
            fockEnergyLocal += vexxPsiCol(ir,k) * wavefun(ir,j,k) * occupationRate_[psi.WavefunIdx(k)];
          }
        }
      }
      mpi::Allreduce( &fockEnergyLocal, &fockEnergy, 1, MPI_SUM, domain_.comm );
    } //if(1) 
  }
  else{
    NumTns<Real>  vexxPsi( ntot, 1, numStateLocalPhi );
    SetValue( vexxPsi, 0.0 );
    psi.AddMultSpinorEXX( fft, phiEXX_, exxgkkR2C_, 
        exxFraction_,  numSpin_, occupationRate_, 
        vexxPsi );
    // Compute the exchange energy:
    // Note: no additional normalization factor due to the
    // normalization rule of psi, NOT phi!!
    fockEnergy = 0.0;
    fockEnergyLocal = 0.0;
    for( Int k = 0; k < numStateLocalPhi; k++ ){
      for( Int j = 0; j < ncom; j++ ){
        for( Int ir = 0; ir < ntot; ir++ ){
          fockEnergyLocal += vexxPsi(ir,j,k) * wavefun(ir,j,k) * occupationRate_[psi.WavefunIdx(k)];
        }
      }
    }
    mpi::Allreduce( &fockEnergyLocal, &fockEnergy, 1, MPI_SUM, domain_.comm );
  }


  return fockEnergy;
}         // -----  end of method KohnSham::CalculateEXXEnergy  ----- 
#endif





void
KohnSham::CalculateVdwEnergyAndForce    ()
{
  std::vector<Atom>&  atomList = this->AtomList();
  EVdw_ = 0.0;
  forceVdw_.Resize( atomList.size(), DIM );
  SetValue( forceVdw_, 0.0 );

  Int numAtom = atomList.size();

  const Domain& dm = domain_;

  if( esdfParam.VDWType == "DFT-D2"){

    const Int vdw_nspecies = 55;
    Int ia,is1,is2,is3,itypat,ja,jtypat,npairs,nshell;
    bool need_gradient,newshell;
    const Real vdw_d = 20.0;
    const Real vdw_tol_default = 1e-10;
    const Real vdw_s_pbe = 0.75, vdw_s_blyp = 1.2, vdw_s_b3lyp = 1.05;
    const Real vdw_s_hse = 0.75, vdw_s_pbe0 = 0.60;
    //Thin Solid Films 535 (2013) 387-389
    //J. Chem. Theory Comput. 2011, 7, 88–96

    Real c6,c6r6,ex,fr,fred1,fred2,fred3,gr,grad,r0,r1,r2,r3,rcart1,rcart2,rcart3;
    //real(dp) :: rcut,rcut2,rsq,rr,sfact,ucvol,vdw_s
    //character(len=500) :: msg
    //type(atomdata_t) :: atom
    //integer,allocatable :: ivdw(:)
    //real(dp) :: gmet(3,3),gprimd(3,3),rmet(3,3)
    //real(dp),allocatable :: vdw_c6(:,:),vdw_r0(:,:),xred01(:,:)
    //DblNumVec vdw_c6_dftd2(vdw_nspecies);

    double vdw_c6_dftd2[vdw_nspecies] = 
    { 0.14, 0.08, 1.61, 1.61, 3.13, 1.75, 1.23, 0.70, 0.75, 0.63,
      5.71, 5.71,10.79, 9.23, 7.84, 5.57, 5.07, 4.61,10.80,10.80,
      10.80,10.80,10.80,10.80,10.80,10.80,10.80,10.80,10.80,10.80,
      16.99,17.10,16.37,12.64,12.47,12.01,24.67,24.67,24.67,24.67,
      24.67,24.67,24.67,24.67,24.67,24.67,24.67,24.67,37.32,38.71,
      38.44,31.74,31.50,29.99, 0.00 };

    // DblNumVec vdw_r0_dftd2(vdw_nspecies);
    double vdw_r0_dftd2[vdw_nspecies] =
    { 1.001,1.012,0.825,1.408,1.485,1.452,1.397,1.342,1.287,1.243,
      1.144,1.364,1.639,1.716,1.705,1.683,1.639,1.595,1.485,1.474,
      1.562,1.562,1.562,1.562,1.562,1.562,1.562,1.562,1.562,1.562,
      1.650,1.727,1.760,1.771,1.749,1.727,1.628,1.606,1.639,1.639,
      1.639,1.639,1.639,1.639,1.639,1.639,1.639,1.639,1.672,1.804,
      1.881,1.892,1.892,1.881,1.000 };

    for(Int i=0; i<vdw_nspecies; i++) {
      vdw_c6_dftd2[i] = vdw_c6_dftd2[i] / 2625499.62 * pow(10/0.52917706, 6);
      vdw_r0_dftd2[i] = vdw_r0_dftd2[i] / 0.52917706;
    }

    DblNumMat vdw_c6(vdw_nspecies, vdw_nspecies);
    DblNumMat vdw_r0(vdw_nspecies, vdw_nspecies);
    SetValue( vdw_c6, 0.0 );
    SetValue( vdw_r0, 0.0 );

    for(Int i=0; i<vdw_nspecies; i++) {
      for(Int j=0; j<vdw_nspecies; j++) {
        vdw_c6(i, j) = std::sqrt( vdw_c6_dftd2[i] * vdw_c6_dftd2[j] );
        vdw_r0(i, j) = vdw_r0_dftd2[i] + vdw_r0_dftd2[j];
      }
    }

    Real vdw_s;

    if (XCType_ == "XC_GGA_XC_PBE") {
      vdw_s = vdw_s_pbe;
    }
    else if (XCType_ == "XC_HYB_GGA_XC_HSE06") {
      vdw_s = vdw_s_hse;
    }
    else if (XCType_ == "XC_HYB_GGA_XC_PBEH") {
      vdw_s = vdw_s_pbe0;
    }
    else {
      ErrorHandling( "Van der Waals DFT-D2 correction in only compatible with GGA-PBE, HSE06, and PBE0!" );
    }


    for(Int ii=-1; ii<2; ii++) {
      for(Int jj=-1; jj<2; jj++) {
        for(Int kk=-1; kk<2; kk++) {

          for(Int i=0; i<atomList.size(); i++) {
            Int iType = atomList[i].type;
            for(Int j=0; j<(i+1); j++) {
              Int jType = atomList[j].type;

              Real rx = atomList[i].pos[0] - atomList[j].pos[0] + ii * dm.length[0];
              Real ry = atomList[i].pos[1] - atomList[j].pos[1] + jj * dm.length[1];
              Real rz = atomList[i].pos[2] - atomList[j].pos[2] + kk * dm.length[2];
              Real rr = std::sqrt( rx * rx + ry * ry + rz * rz );

              if ( ( rr > 0.0001 ) && ( rr < 75.0 ) ) {

                Real sfact = vdw_s;
                if ( i == j ) sfact = sfact * 0.5;

                Real c6 = vdw_c6(iType-1, jType-1);
                Real r0 = vdw_r0(iType-1, jType-1);

                Real ex = exp( -vdw_d * ( rr / r0 - 1 ));
                Real fr = 1.0 / ( 1.0 + ex );
                Real c6r6 = c6 / pow(rr, 6.0);

                // Contribution to energy
                EVdw_ = EVdw_ - sfact * fr * c6r6;

                // Contribution to force
                if( i != j ) {

                  Real gr = ( vdw_d / r0 ) * ( fr * fr ) * ex;
                  Real grad = sfact * ( gr - 6.0 * fr / rr ) * c6r6 / rr; 

                  Real fx = grad * rx;
                  Real fy = grad * ry;
                  Real fz = grad * rz;

                  forceVdw_( i, 0 ) = forceVdw_( i, 0 ) + fx; 
                  forceVdw_( i, 1 ) = forceVdw_( i, 1 ) + fy; 
                  forceVdw_( i, 2 ) = forceVdw_( i, 2 ) + fz; 
                  forceVdw_( j, 0 ) = forceVdw_( j, 0 ) - fx; 
                  forceVdw_( j, 1 ) = forceVdw_( j, 1 ) - fy; 
                  forceVdw_( j, 2 ) = forceVdw_( j, 2 ) - fz; 

                } // end for i != j

              } // end if

            } // end for j
          } // end for i

        } // end for ii
      } // end for jj
    } // end for kk

  } // If DFT-D2

  return ;
}         // -----  end of method KohnSham::CalculateVdwEnergyAndForce  ----- 


void
KohnSham::CalculateIonSelfEnergyAndForce    ( PeriodTable &ptable )
{

  std::vector<Atom>&  atomList = this->AtomList();
  EVdw_ = 0.0;
  forceVdw_.Resize( atomList.size(), DIM );
  SetValue( forceVdw_, 0.0 );
  
  // Self energy part. 
  Eself_ = 0.0;
  for(Int a=0; a< atomList.size() ; a++) {
    Int type = atomList[a].type;
    Eself_ +=  ptable.SelfIonInteraction(type);
  }

  // Short range repulsion part
  EIonSR_ = 0.0;
  forceIonSR_.Resize( atomList.size(), DIM );
  SetValue(forceIonSR_, 0.0);
  if( esdfParam.isUseVLocal == true ){
    const Domain& dm = domain_;

    for(Int a=0; a< atomList.size() ; a++) {
      Int type_a = atomList[a].type;
      Real Zion_a = ptable.Zion(type_a);
      Real RGaussian_a = ptable.RGaussian(type_a);

      for(Int b=a; b< atomList.size() ; b++) {
        // Need to consider the interaction between the same atom and
        // its periodic image. Be sure not to double ocunt
        bool same_atom = (a==b);

        Int type_b = atomList[b].type;
        Real Zion_b = ptable.Zion(type_b);
        Real RGaussian_b = ptable.RGaussian(type_b);

        Real radius_ab = std::sqrt ( RGaussian_a*RGaussian_a + RGaussian_b*RGaussian_b );
        // convergence criterion for lattice sums:
        // facNbr * radius_ab < ncell * d
        const Real facNbr = 8.0;
        const Int ncell0 = (Int) (facNbr * radius_ab / dm.length[0]);
        const Int ncell1 = (Int) (facNbr * radius_ab / dm.length[1]);
        const Int ncell2 = (Int) (facNbr * radius_ab / dm.length[2]);

        Point3 pos_ab = atomList[a].pos - atomList[b].pos;
        for( Int d = 0; d < DIM; d++ ){
          pos_ab[d] = pos_ab[d] - IRound(pos_ab[d] / dm.length[d])*dm.length[d];
        }


        // loop over neighboring cells
        Real fac;
        for ( Int ic0 = -ncell0; ic0 <= ncell0; ic0++ )
          for ( Int ic1 = -ncell1; ic1 <= ncell1; ic1++ )
            for ( Int ic2 = -ncell2; ic2 <= ncell2; ic2++ )
            {
              if ( !same_atom || ic0!=0 || ic1!=0 || ic2!=0 )
              {
                if ( same_atom )
                  fac = 0.5;
                else
                  fac = 1.0;
                
                Point3 pos_ab_image;
                pos_ab_image[0] = pos_ab[0] + ic0*dm.length[0];
                pos_ab_image[1] = pos_ab[1] + ic1*dm.length[1];
                pos_ab_image[2] = pos_ab[2] + ic2*dm.length[2];

                Real r_ab = pos_ab_image.l2();
                Real esr_term = Zion_a * Zion_b * std::erfc(r_ab / radius_ab) / r_ab;
                Real desr_erfc = 2.0 * Zion_a * Zion_b *
                  std::exp(-(r_ab / radius_ab)*(r_ab / radius_ab))/(radius_ab*std::sqrt(PI));
                // desrdr = (1/r) d Esr / dr
                Real desrdr = - fac * (esr_term+desr_erfc) / ( r_ab*r_ab );
                
                EIonSR_ += fac * esr_term;

                forceIonSR_(a,0) -= desrdr * pos_ab_image[0];
                forceIonSR_(b,0) += desrdr * pos_ab_image[0];
                forceIonSR_(a,1) -= desrdr * pos_ab_image[1];
                forceIonSR_(b,1) += desrdr * pos_ab_image[1];
                forceIonSR_(a,2) -= desrdr * pos_ab_image[2];
                forceIonSR_(b,2) += desrdr * pos_ab_image[2];
              }
            }
      } // for (b)
    } // for (a)
  } // if esdfParam.isUseVLocal == true

  return ;
}         // -----  end of method KohnSham::CalculateIonSelfEnergyAndForce  ----- 

void KohnSham::Setup_XC( std::string xc_functional)
{
  if( xc_functional == "XC_GGA_XC_PBE" )
  {
    XId_  = XC_GGA_X_PBE;
    CId_  = XC_GGA_C_PBE;
    isEXXActive_ = false;
    statusOFS << "XC_GGA_XC_PBE  XId_ CId_ = " << XId_ << " " << CId_  << std::endl << std::endl;
    // Perdew, Burke & Ernzerhof correlation
    // JP Perdew, K Burke, and M Ernzerhof, Phys. Rev. Lett. 77, 3865 (1996)
    // JP Perdew, K Burke, and M Ernzerhof, Phys. Rev. Lett. 78, 1396(E) (1997)
    if( xc_func_init(&XFuncType_, XId_, XC_UNPOLARIZED) != 0 ){
      ErrorHandling( "X functional initialization error." );
    }
    if( xc_func_init(&CFuncType_, CId_, XC_UNPOLARIZED) != 0 ){
      ErrorHandling( "C functional initialization error." );
    }
  }
  else if( xc_functional == "XC_HYB_GGA_XC_HSE06" )
  {
    XCId_ = XC_HYB_GGA_XC_HSE06;
    XId_ = XC_GGA_X_PBE;
    CId_ = XC_GGA_C_PBE;
    statusOFS << "XC_HYB_GGA_XC_HSE06  XCId = " << XCId_  << std::endl << std::endl;
    statusOFS << "XC_HYB_GGA_XC_HSE06  XId_ CId_ = " << XId_ << " " << CId_  << std::endl;
    if( xc_func_init(&XCFuncType_, XCId_, XC_UNPOLARIZED) != 0 ){
      ErrorHandling( "XC functional initialization error." );
    }
    if( xc_func_init(&XFuncType_, XId_, XC_UNPOLARIZED) != 0 ){
      ErrorHandling( "X functional initialization error." );
    }
    if( xc_func_init(&CFuncType_, CId_, XC_UNPOLARIZED) != 0 ){
      ErrorHandling( "C functional initialization error." );
    }

    isHybrid_ = true;
  }
  else if( xc_functional == "XC_HYB_GGA_XC_PBEH" )
  {
    XCId_ = XC_HYB_GGA_XC_PBEH;
    statusOFS << "XC_HYB_GGA_XC_PBEH  XCId = " << XCId_  << std::endl ;
    if( xc_func_init(&XCFuncType_, XCId_, XC_UNPOLARIZED) != 0 ){
      ErrorHandling( "XC functional initialization error." );
    } 
    isHybrid_ = true;
  }
}

} // namespace dgdft
