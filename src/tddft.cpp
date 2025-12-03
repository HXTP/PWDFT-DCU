/*
   Copyright (c) 2012 The Regents of the University of California,
   through Lawrence Berkeley National Laboratory.  

Author: Weile Jia, Lin Lin 

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
/// @file tddft.cpp
/// @brief time dependent density functional theory with ehrenfest dynamics.
/// @date 2017-09-05 Initialize
/// @date 2017-12-28 Integrate with master branch

#include "tddft.hpp"
#include "utility.hpp"


#ifdef _COMPLEX_
namespace dgdft{

using namespace dgdft::DensityComponent;
using namespace dgdft::esdf;

void setDefaultEfieldOptions( eField * eF)
{
  // set the polarization to x 
  eF->pol.resize(3);
  eF->pol[0] = esdfParam.TDDFTVextPolx;
  eF->pol[1] = esdfParam.TDDFTVextPoly;
  eF->pol[2] = esdfParam.TDDFTVextPolz;

  // set frequency to 0.0 
  eF->freq = esdfParam.TDDFTVextFreq;

  // set phase to 0.0 
  eF->phase = esdfParam.TDDFTVextPhase;

  // set phase to 0.0 
  eF->env = esdfParam.TDDFTVextEnv;

  // set Amp to 0
  eF->Amp = esdfParam.TDDFTVextAmp;

  // set t0 to 0
  eF->t0 = esdfParam.TDDFTVextT0;

  // set tau to 0
  eF->tau = esdfParam.TDDFTVextTau;
}


void setEfieldPolarization( eField* eF, std::vector<Real> & pol)
{
  Real scale = std::sqrt(pol[0]*pol[0] + pol[1]*pol[1] + pol[2] * pol[2]);

  eF->pol.resize(3);
  for(int i =0; i < 3; i++)
    eF->pol[i] = pol[i]/scale;
}

void setEfieldFrequency( eField* eF, Real freq)
{
  eF->freq = freq;
}

void setEfieldPhase( eField* eF, Real phase)
{
  eF->phase = phase;
}

void setEfieldEnv( eField* eF, std::string env)
{
  if(env != "constant" && env != "gaussian" &&
      env != "erf"      && env != "sinsq"    &&
      env != "hann"     && env != "kick")
    env = "gaussian"; // set gaussian as default

  eF->env = env;
}

void setEfieldAmplitude( eField* eF, Real Amp)
{
  eF->Amp = Amp;
}

void setEfieldT0( eField* eF, Real t0)
{
  eF->t0 = t0;
}

void setEfieldTau( eField* eF, Real tau)
{
  if(tau <= 0.0){
    tau = 1.0;
    statusOFS << " Warning: Tau must be positive number; reset to 1.0 " << std::endl;
  }
  eF->tau = tau;
}

void setDefaultTDDFTOptions( TDDFTOptions * options)
{
  options->method        = esdfParam.TDDFTMethod;
  options->ehrenfest     = esdfParam.isTDDFTEhrenfest;
  options->simulateTime  = esdfParam.TDDFTTotalT;
  options->dt            = esdfParam.TDDFTDeltaT;
  options->phiMaxIter    = esdfParam.TDDFTPhiMaxIter; 
  options->diisMaxIter   = esdfParam.TDDFTDiisMaxIter; 
  options->krylovTol     = esdfParam.TDDFTKrylovTol;
  options->krylovMax     = esdfParam.TDDFTKrylovMax; 
  options->diisTol       = esdfParam.TDDFTDiisTol; 
  options->phiTol        = esdfParam.TDDFTPhiTol; 
  options->isOutputXYZ   = esdfParam.isOutputXYZ;

  //  FIXME 
  options->auto_save     = 0;
  options->load_save     = false;
  options->gmres_restart = 10; // not sure.
  options->adNum         = 20; 
  options->adUpdate      = 1;
  setDefaultEfieldOptions( & options->eField_);
}
void setTDDFTMethod( TDDFTOptions * options, std::string method)
{
  if(method == "RK4" || method == "TPTRAP")
    options->method = method;
  else
    statusOFS << std::endl << " Warning: Method set failed, still use " << options->method << std::endl;
}
void setTDDFTEhrenfest( TDDFTOptions * options, bool ehrenfest)
{
  options->ehrenfest = ehrenfest;
}
void setTDDFTDt( TDDFTOptions * options, Real dT)
{
  options->dt = dT;
}
void setTDDFTTime( TDDFTOptions * options, Real time)
{
  options->simulateTime = time;
}
void setTDDFTkrylovTol( TDDFTOptions * options, Real krylovTol)
{
  options->krylovTol = krylovTol;
}
void setTDDFTkrylovMax( TDDFTOptions *options, int krylovMax)
{
  options->krylovMax = krylovMax;
}
void setTDDFTDiisTol( TDDFTOptions *options, Real diisTol)
{
  options->diisTol = diisTol;
}
void setTDDFTPhiTol( TDDFTOptions *options, Real phiTol)
{
  options->phiTol = phiTol;
}

void TDDFT::Setup(
    Hamiltonian& ham,
    Spinor& psi,
    Fourier& fft,
    std::vector<Atom>& atomList,
    PeriodTable& ptable) {

  hamPtr_ = &ham;
  psiPtr_ = &psi;
  fftPtr_ = &fft;
  atomListPtr_ = &atomList;


  MPI_Comm mpi_comm = fftPtr_->domain.comm;
  Int mpirank, mpisize;
  MPI_Comm_rank( mpi_comm, &mpirank );
  MPI_Comm_size( mpi_comm, &mpisize );

  if( options_.method == "PTTRAP"){
    if( psi.NumStateTotal() % mpisize != 0) 
      ErrorHandling( " Band must be multiples of Np." );
  }

  if( esdfParam.numExtraState != 0 ) 
    ErrorHandling( " ExtraState must be 0. check your pwdft.in ");

  // Grab the supercell info
  supercell_x_ = esdfParam.domain.length[0];
  supercell_y_ = esdfParam.domain.length[1];
  supercell_z_ = esdfParam.domain.length[2];

  // History of atomic position
  maxHist_ = 4;  // hard coded
  atomListHist_.resize(maxHist_);
  for( Int l = 0; l < maxHist_; l++ ){
    atomListHist_[l] = atomList;
  }

  Int numAtom = atomList.size();
  atomMass_.Resize( numAtom );
  for(Int a=0; a < numAtom; a++) {
    Int atype = atomList[a].type;
    if (ptable.ptemap().find(atype)==ptable.ptemap().end() ){
      ErrorHandling( "Cannot find the atom type." );
    }
    atomMass_[a]=amu2au*ptable.Mass(atype);
  }

  setDefaultTDDFTOptions( & options_);// set the default options.

  // init the time list
  int size = options_.simulateTime/ options_.dt + 1;
  tlist_.resize(size);
  for(int i = 0; i < size; i++)
    tlist_[i] = i * options_.dt;

  // might need to setup others, will add here.
  k_ = 0;

  // CHECK CHECK: change this to a input parameter.
  calDipole_ = esdfParam.isTDDFTDipole;
  calVext_ = esdfParam.isTDDFTVext;

  if( calDipole_ ) {
    statusOFS << " ************************ WARNING ******************************** " << std::endl;
    statusOFS << " Warning: Please make sure that your atoms are centered at (0,0,0) " << std::endl;
    statusOFS << " ************************ WARNING ******************************** " << std::endl;

    Xr_.Resize( fft.domain.NumGridTotalFine() );
    Yr_.Resize( fft.domain.NumGridTotalFine() );
    Zr_.Resize( fft.domain.NumGridTotalFine() );
    D_.Resize( fft.domain.NumGridTotalFine() );

    Real * xr = Xr_.Data();
    Real * yr = Yr_.Data();
    Real * zr = Zr_.Data();

    Int  idx;
    Real Xtmp, Ytmp, Ztmp;       

    for( Int k = 0; k < fft.domain.numGridFine[2]; k++ ){
      for( Int j = 0; j < fft.domain.numGridFine[1]; j++ ){
        for( Int i = 0; i < fft.domain.numGridFine[0]; i++ ){

          idx = i + j * fft.domain.numGridFine[0] + k * fft.domain.numGridFine[0] * fft.domain.numGridFine[1];
          Xtmp = (Real(i) - Real( round(Real(i)/Real(fft.domain.numGridFine[0])) * Real(fft.domain.numGridFine[0]) ) ) / Real(fft.domain.numGridFine[0]);
          Ytmp = (Real(j) - Real( round(Real(j)/Real(fft.domain.numGridFine[1])) * Real(fft.domain.numGridFine[1]) ) ) / Real(fft.domain.numGridFine[1]);
          Ztmp = (Real(k) - Real( round(Real(k)/Real(fft.domain.numGridFine[2])) * Real(fft.domain.numGridFine[2]) ) ) / Real(fft.domain.numGridFine[2]);

          // should be AL(0,0) * X + AL(0,1) * Y + AL(0,2) * Z
          // the other parts are zeros. 
          xr[idx] = Xtmp * supercell_x_ ;
          yr[idx] = Ytmp * supercell_y_ ;
          zr[idx] = Ztmp * supercell_z_ ;

          // get the p.D corresponding to the matlab KSSOLV 
          D_[idx] = xr[idx] * options_.eField_.pol[0] + yr[idx] * options_.eField_.pol[1] + zr[idx] * options_.eField_.pol[2]; 
        }
      }
    }
  } 

  Int mixMaxDim_ = esdfParam.mixMaxDim;
  //statusOFS << " mixMaxDim_ " << mixMaxDim_ << std::endl;
  Int ntotFine  = fftPtr_->domain.NumGridTotalFine();
  dfMat_.Resize( ntotFine, mixMaxDim_ ); SetValue( dfMat_, 0.0 );
  dvMat_.Resize( ntotFine, mixMaxDim_ ); SetValue( dvMat_, 0.0 );

  // init the sgmres solvers. 
  sgmres_solver.Setup(ham, psi, fft, fft.domain.NumGridTotal());

  if(mpirank == 0) {
    vextOFS.open( "vext.out");
    vextOFS 
      << std::setw(LENGTH_VAR_DATA) << " Time(fs) " << " " 
      << std::setw(LENGTH_VAR_DATA) << " Vext " << " " << std::endl;

    dipoleOFS.open( "dipole.out");
    dipoleOFS 
      << std::setw(LENGTH_VAR_DATA) << " Time(fs) " 
      << std::setw(LENGTH_VAR_DATA) << " Dipole_X "
      << std::setw(LENGTH_VAR_DATA) << " Dipole_Y "
      << std::setw(LENGTH_VAR_DATA) << " Dipole_Z " << std::endl;

    etotOFS.open( "etot.out");
    etotOFS 
      << std::setw(LENGTH_VAR_DATA) << " Time "    << " " 
      << std::setw(LENGTH_VAR_DATA) << " Etot "    << " "
      << std::setw(LENGTH_VAR_DATA) << " AtomKin " << " "
      << std::setw(LENGTH_VAR_DATA) << " Epot "    << " "
      << std::setw(LENGTH_VAR_DATA) << " Eext "    << " "
      << std::setw(LENGTH_VAR_DATA) << " Eproton " << " "
      << std::setw(LENGTH_VAR_DATA) << " Efock "   << std::endl;

  }

  if(esdfParam.isRestartVelocity){
    statusOFS << std::endl 
      << "Read velocity information from lastVel.out. " << std::endl;

    DblNumVec atomvelRead(3*numAtom);
    if( mpirank == 0 ){
      std::fstream fin;
      fin.open("lastVel.out",std::ios::in);
      if( !fin.good() ){
        ErrorHandling( "Cannot open lastVel.out!" );
      }
      for(Int a=0; a<numAtom; a++){
        fin>> atomvelRead[3*a+0];
        fin>> atomvelRead[3*a+1];
        fin>> atomvelRead[3*a+2];
      }
      fin.close();
    }
    // Broadcast thermostat information
    MPI_Bcast( atomvelRead.Data(), 3*numAtom, MPI_DOUBLE, 0, MPI_COMM_WORLD );

    for(Int a=0; a<numAtom; a++){
      atomList[a].vel = 
        Point3( atomvelRead[3*a], atomvelRead[3*a+1], atomvelRead[3*a+2] );
    }

    if( mpirank == 0 ){
      PrintBlock( statusOFS, "Read in Atomic Velocity" );
      {
        for( Int a = 0; a < numAtom; a++ ){
          Print( statusOFS, "atom", a, "Velocity   ", atomList[a].vel );
        }
      }
    }
  }//restart read in last velocities of atoms


  // Recompute the atomic force
  if(options_.ehrenfest){
    CalculateEfieldExt(ptable, 0.0);
    ham.CalculateForce( psi, fft);
  }


  if( options_.eField_.env == "kick" ){
    // Transform psi to fine grid. Apply the delta kick on the real space
    // fine grid, and then transform back
    Int ntot = fft.domain.NumGridTotal();
    Int ntotFine = fft.domain.NumGridTotalFine();
    Int ncom = psi.NumComponent();
    Int numStateLocal = psi.NumState();
    CpxNumVec psiCoarse(ntot);
    CpxNumVec psiFine(ntotFine);

    for (Int k=0; k<numStateLocal; k++) {
      for (Int j=0; j<ncom; j++) {

        blas::Copy( ntot, psi.Wavefun().VecData(j,k), 1, fft.inputComplexVec.Data(), 1 );
        // Fourier transform of wavefunction saved in fft.outputComplexVec
        fftw_execute( fft.forwardPlan );

        // Interpolate wavefunction from coarse to fine grid
        SetValue( fft.outputComplexVecFine, Z_ZERO ); 
        for( Int i = 0; i < ntot; i++ ){
          fft.outputComplexVecFine[fft.idxFineGrid[i]] = fft.outputComplexVec[i];
        }

        fftw_execute( fft.backwardPlanFine );
        Real fac = 1.0 / std::sqrt( double(fft.domain.NumGridTotal())  *
            double(fft.domain.NumGridTotalFine()) ); 

        blas::Copy( ntotFine, fft.inputComplexVecFine.Data(),
            1, psiFine.Data(), 1 );
        blas::Scal( ntotFine, fac, psiFine.Data(), 1 );

        // Add the contribution from local pseudopotential
        Complex expfac;
        for( Int i = 0; i < ntotFine; i++ ){
          expfac = Complex(0.0,options_.eField_.Amp*D_[i]);
          psiFine[i] *= std::exp(expfac);
        }

        // Restrict psiFine from fine grid in the real space to
        // coarse grid in the Fourier space.
        blas::Copy( ntotFine, psiFine.Data(), 1,
            fft.inputComplexVecFine.Data(), 1 );

        fftw_execute( fft.forwardPlanFine );
        {
          Real fac = std::sqrt(Real(ntot) / (Real(ntotFine)));

          for( Int i = 0; i < ntot; i++ ){
            fft.outputComplexVec[i] = fft.outputComplexVecFine[fft.idxFineGrid[i]] * fac;
          }
        }

        // Inverse Fourier transform to save back to the output vector
        fftw_execute( fft.backwardPlan );
        blas::Copy( ntot, fft.inputComplexVec.Data(), 1,
            psi.Wavefun().VecData(j,k), 1 ); 
        blas::Scal( ntot, 1.0 / Real(ntot), psi.Wavefun().VecData(j,k), 1 );
      } // for (j)
    } // for (k)

  } // apply delta kick

  {
    isCalculateGradRho_ = false;
    if( esdfParam.XCType == "XC_GGA_XC_PBE" || 
        esdfParam.XCType == "XC_HYB_GGA_XC_HSE06" ||
        esdfParam.XCType == "XC_HYB_GGA_XC_PBEH" ) {
      isCalculateGradRho_ = true;
    }
  }

} // TDDFT::Setup function

void TDDFT::AdjustAtomPos( std::vector<Point3> &atomPos)
{
  Fourier&     fft = *fftPtr_;
  int numAtom = atomPos.size();
  Point3 Ls = fft.domain.length;
//  for( int ia = 0; ia < numAtom; ia++)
//  {
//    if( atomPos[ia][0] < 0.0) 
//      atomPos[ia][0] += supercell_x_;
//    if( atomPos[ia][1] < 0.0) 
//      atomPos[ia][1] += supercell_y_;
//    if( atomPos[ia][2] < 0.0) 
//      atomPos[ia][2] += supercell_z_;
//    if( atomPos[ia][0] > supercell_x_) 
//      atomPos[ia][0] -= supercell_x_;
//    if( atomPos[ia][1] > supercell_y_) 
//      atomPos[ia][1] -= supercell_y_;
//    if( atomPos[ia][2] > supercell_z_) 
//      atomPos[ia][2] -= supercell_z_;
//  }
  for( int a = 0; a < numAtom; a++)
  {
    atomPos[a][0] -= IRound(atomPos[a][0] / Ls[0]) * Ls[0];
    atomPos[a][1] -= IRound(atomPos[a][1] / Ls[1]) * Ls[1];
    atomPos[a][2] -= IRound(atomPos[a][2] / Ls[2]) * Ls[2];
  }
}


Real TDDFT::getEfield(Real t)
{
  Real et = 0.0;
  if ( options_.eField_.env == "gaussian" ) {
    Real temp = (t-options_.eField_.t0)/options_.eField_.tau;
    et = options_.eField_.Amp * exp( - temp * temp / 2.0) * sin(options_.eField_.freq * t + options_.eField_.phase);
  }
  else if( options_.eField_.env == "kick" )
    et = 0.0;
  else{
    statusOFS<< " Wrong Efield input, should be constant/gaussian/erf/sinsq/hann/kick" << std::endl;
    exit(0);
  }
  return et;
}

void TDDFT::CalculateEfieldExt(PeriodTable& ptable, Real t)
{
  Hamiltonian& ham = *hamPtr_;
  Fourier&     fft = *fftPtr_;
  std::vector<Atom>&  atomList = ham.AtomList();
  Point3 result;
  Int numAtom = atomList.size();

  Eext_ = 0.0;

  DblNumMat forceext( atomList.size(), DIM );
  SetValue( forceext, 0.0 );

  DblNumVec vext = ham.Vext();

  if(calVext_){
    Real et = getEfield(t);

    // time(fs), et
    vextOFS 
      << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC) << t * au2fs << " " 
      << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC) << et << std::endl;
    // then Vext = Vext0 + et * options_.D
    // Here we suppose the Vext0 are zeros.
    Int idx;
    for( Int k = 0; k < fft.domain.numGridFine[2]; k++ ){
      for( Int j = 0; j < fft.domain.numGridFine[1]; j++ ){
        for( Int i = 0; i < fft.domain.numGridFine[0]; i++ ){
          idx = i + j * fft.domain.numGridFine[0] + k * fft.domain.numGridFine[0] * fft.domain.numGridFine[1];
          vext[idx] = et * D_[idx];
        }
      }
    }

    Int numAtom = atomList.size();
    std::vector<Point3>  atompos(numAtom);
    for( Int a = 0; a < numAtom; a++ ){
      atompos[a] = atomList[a].pos;
    }

    Real xet = options_.eField_.pol[0] * et ;
    Real yet = options_.eField_.pol[1] * et ;
    Real zet = options_.eField_.pol[2] * et ;

    Point3 Ls = fft.domain.length;

    Real dx, dy, dz;
    for (Int a=0; a<numAtom; a++) {
      Int atype  = atomList[a].type;
      if( ptable.ptemap().find(atype) == ptable.ptemap().end() ){
        ErrorHandling( "Cannot find the atom type." );
      }
      dx = atompos[a][0] - IRound(atompos[a][0] / Ls[0]) * Ls[0];
      dy = atompos[a][1] - IRound(atompos[a][1] / Ls[1]) * Ls[1];
      dz = atompos[a][2] - IRound(atompos[a][2] / Ls[2]) * Ls[2];


      Eext_  -= ptable.Zion(atype) * ( xet * dx + yet * dy + zet * dz );

      forceext(a,0) = xet * ptable.Zion(atype);
      forceext(a,1) = yet * ptable.Zion(atype);
      forceext(a,2) = zet * ptable.Zion(atype);
    }



    // For consistency update the info to hamiltonian. This will later
    // be used for CalculateEnergy and CalculateForce
    ham.SetVext( vext );
    ham.SetEext( Eext_ );
    ham.SetForceExt( forceext );
  } // calVext_ 
  return;
}


void TDDFT::Update() {

  Int mixMaxDim_ = esdfParam.mixMaxDim;
  Int ntotFine  = fftPtr_->domain.NumGridTotalFine();
  dfMat_.Resize( ntotFine, mixMaxDim_ ); SetValue( dfMat_, 0.0 );
  dvMat_.Resize( ntotFine, mixMaxDim_ ); SetValue( dvMat_, 0.0 );

  return;
}

void TDDFT::CalculateDipole(Real t)
{
  Hamiltonian& ham = *hamPtr_;
  Fourier&     fft = *fftPtr_;
  Spinor&      psi = *psiPtr_;

  // Di = -∫ Rho(i, j, k) * Xr(i, j, k) *dx
  // Density is not distributed
  DblNumMat& density = ham.Density();

  Real Dx = 0.0;
  Real Dy = 0.0;
  Real Dz = 0.0;
  Real fac = fft.domain.Volume() / fft.domain.NumGridTotalFine();
  
  Dx = -fac * blas::Dot( fft.domain.NumGridTotalFine(), density.VecData(0), 1, 
      Xr_.Data(), 1 );
  Dy = -fac * blas::Dot( fft.domain.NumGridTotalFine(), density.VecData(0), 1, 
      Yr_.Data(), 1 );
  Dz = -fac * blas::Dot( fft.domain.NumGridTotalFine(), density.VecData(0), 1, 
      Zr_.Data(), 1 );

  // Time (fs), Dx, Dy, Dz
  dipoleOFS 
    << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)  << t * au2fs <<  " " 
    << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)  << Dx << " " 
    << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)  << Dy << " " 
    << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)  << Dz << std::endl;

}    // -----  end of method TDDFT::CalculateDipole ---- 



void
TDDFT::AndersonMix    ( 
    Int             iter,
    Real            mixStepLength,
    std::string     mixType,
    DblNumVec&      vMix,
    DblNumVec&      vOld, 
    DblNumVec&      vNew,
    DblNumMat&      dfMat,
    DblNumMat&      dvMat ) {

  Int ntot  = fftPtr_->domain.NumGridTotalFine();

  // Residual 
  DblNumVec res;
  // Optimal input potential in Anderon mixing.
  DblNumVec vOpt; 
  // Optimal residual in Anderson mixing
  DblNumVec resOpt; 
  // Preconditioned optimal residual in Anderson mixing
  DblNumVec precResOpt;

  res.Resize(ntot);
  vOpt.Resize(ntot);
  resOpt.Resize(ntot);
  precResOpt.Resize(ntot);

  Int mixMaxDim_ = esdfParam.mixMaxDim;
  // Number of iterations used, iter should start from 1
  Int iterused = std::min( iter-1, mixMaxDim_ ); 
  // The current position of dfMat, dvMat
  Int ipos = iter - 1 - ((iter-2)/ mixMaxDim_ ) * mixMaxDim_;
  // The next position of dfMat, dvMat
  Int inext = iter - ((iter-1)/ mixMaxDim_) * mixMaxDim_;

  statusOFS << " iter " << iter << " mixMaxDim_ " << mixMaxDim_ << " iterused " << iterused << " ipos  " << ipos << " inext " << inext << std::endl;
  res = vOld;
  // res(:) = vOld(:) - vNew(:) is the residual
  blas::Axpy( ntot, -1.0, vNew.Data(), 1, res.Data(), 1 );

  vOpt = vOld;
  resOpt = res;

  if( iter > 1 ){
    // dfMat(:, ipos-1) = res(:) - dfMat(:, ipos-1);
    // dvMat(:, ipos-1) = vOld(:) - dvMat(:, ipos-1);
    blas::Scal( ntot, -1.0, dfMat.VecData(ipos-1), 1 );
    blas::Axpy( ntot, 1.0, res.Data(), 1, dfMat.VecData(ipos-1), 1 );
    blas::Scal( ntot, -1.0, dvMat.VecData(ipos-1), 1 );
    blas::Axpy( ntot, 1.0, vOld.Data(), 1, dvMat.VecData(ipos-1), 1 );


    // Calculating pseudoinverse
    Int nrow = iterused;
    DblNumMat dfMatTemp;
    DblNumVec gammas, S;

    Int rank;
    // FIXME Magic number
    Real rcond = 1e-12;

    S.Resize(nrow);

    gammas    = res;
    dfMatTemp = dfMat;

    lapack::SVDLeastSquare( ntot, iterused, 1, 
        dfMatTemp.Data(), ntot, gammas.Data(), ntot,
        S.Data(), rcond, &rank );

    Print( statusOFS, "  Rank of dfmat = ", rank );
    Print( statusOFS, "  Rcond = ", rcond );
    // Update vOpt, resOpt. 

    blas::Gemv('N', ntot, nrow, -1.0, dvMat.Data(),
        ntot, gammas.Data(), 1, 1.0, vOpt.Data(), 1 );

    blas::Gemv('N', ntot, iterused, -1.0, dfMat.Data(),
        ntot, gammas.Data(), 1, 1.0, resOpt.Data(), 1 );
  }

  if( mixType == "kerker+anderson" ){
    Print( statusOFS, " Kerker+anderson  is not supported in TDDFT ");
  }
  else if( mixType == "anderson" ){
    precResOpt = resOpt;
  }
  else{
    ErrorHandling("Invalid mixing type.");
  }


  // Update dfMat, dvMat, vMix 
  // dfMat(:, inext-1) = res(:)
  // dvMat(:, inext-1) = vOld(:)
  blas::Copy( ntot, res.Data(), 1, 
      dfMat.VecData(inext-1), 1 );
  blas::Copy( ntot, vOld.Data(),  1, 
      dvMat.VecData(inext-1), 1 );

  // vMix(:) = vOpt(:) - mixStepLength * precRes(:)
  vMix = vOpt;
  blas::Axpy( ntot, -mixStepLength, precResOpt.Data(), 1, vMix.Data(), 1 );


  return ;

}         // -----  end of method SCF::AndersonMix  ----- 

void
TDDFT::CalculateEnergy  ( PeriodTable& ptable, Real t )
{

  Hamiltonian& ham = *hamPtr_;
  Fourier&     fft = *fftPtr_;
  Spinor&      psi = *psiPtr_;


  // Electronic kinetic energy has been computed elsewhere

  // Hartree and xc part
  Int  ntot = fft.domain.NumGridTotalFine();
  Real vol  = fft.domain.Volume();
  DblNumMat&  density      = ham.Density();
  DblNumMat&  vxc          = ham.Vxc();
  DblNumVec&  pseudoCharge = ham.PseudoCharge();
  DblNumVec&  vhart        = ham.Vhart();
  Ehart_ = 0.0;
  EVxc_  = 0.0;
  for (Int i=0; i<ntot; i++) {
    EVxc_  += vxc(i,RHO) * density(i,RHO);
    Ehart_ += 0.5 * vhart(i) * ( density(i,RHO) + pseudoCharge(i) );
  }
  Ehart_ *= vol/Real(ntot);
  EVxc_  *= vol/Real(ntot);

  // Ionic repulsion related energy
  Eself_ = ham.Eself();

  Ecor_ = (Exc_ - EVxc_) - Ehart_ - Eself_;

  if( esdfParam.isUseVLocal == true ){
    EIonSR_ = ham.EIonSR();
    Ecor_ += EIonSR_;
  }

  // Van der Waals energy
  EVdw_ = ham.EVdw();
  Ecor_ += EVdw_;

  // Total energy
  Etot_ = Ekin_ + Ecor_;

  // Helmholtz free energy
  if( ham.NumOccupiedState() == 
      ham.NumStateTotal() ){
    // Zero temperature
    Efree_ = Etot_;
  }
  else{
    ErrorHandling("Fractional occupation not supported for TDDFT.");
  }

  // Atomic kinetic energy
  Real Eproton = 0.0;
  AtomKin_ = 0.0;
  {
    std::vector<Atom>&  atomList = ham.AtomList();
    Int numAtom = atomList.size();
    std::vector<Point3>  atompos(numAtom);
    std::vector<Point3>  atomvel(numAtom);
    std::vector<Point3>  atomforce(numAtom);
    DblNumVec& atomMass = atomMass_;

    for( Int a = 0; a < numAtom; a++ ){
      atomvel[a] = atomList[a].vel;
      atompos[a] = atomList[a].pos;
    }


    for(Int a=0; a<numAtom; a++){
      for(Int j=0; j<3; j++){
        AtomKin_ += atomMass[a]*atomvel[a][j]*atomvel[a][j]/2.;
      }
    }

    Int a = numAtom - 1;
    for(Int j=0; j<3; j++){
      Eproton += atomMass[a]*atomvel[a][j]*atomvel[a][j]/2.;
    }
  }

  // External energy due to the electric field 
  Eext_ = ham.Eext();

#if ( _DEBUGlevel_ >= 2 )
  statusOFS << " Etot_ " << Etot_ << " EVdw " << EVdw_ << " Ecor_ " << Ecor_  << " EIonSR_ " 
    << EIonSR_ << " Eself " << Eself_ << " EVxc_ " << EVxc_ << " Ehart_ " << Ehart_ 
    << " Ekin_ " <<  Ekin_  << std::endl;
#endif
  //  Time(fs), E_tot(eV), E_kin(eV), E_pot(ev), E_field(eV), E_proton

  Real Efork = 0.0; 

  if( ham.IsHybrid() ) {
    // the fock energy calculation must use the ACE operator.
    Efork = ham.CalculateEXXEnergy( psi, fft ); 
  }
  etotOFS 
    << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)  << t * au2fs << " " 
    << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)  << (Etot_ + AtomKin_ + Eext_ - Efork) * au2ev << " "
    << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)  << AtomKin_ * au2ev << " "
    << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)  << Etot_ * au2ev << " "
    << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)  << Eext_ * au2ev<< " "
    << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)  << Eproton * au2ev<< " "
    << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)  << Efork* au2ev<< std::endl;

  return ;
}         // -----  end of method TDDFT::CalculateEnergy  ----- 


void TDDFT::advanceRK4( PeriodTable& ptable ) {

  Int mpirank, mpisize;
  MPI_Comm mpi_comm = fftPtr_->domain.comm;
  MPI_Comm_rank( mpi_comm, &mpirank );
  MPI_Comm_size( mpi_comm, &mpisize );

  Hamiltonian& ham = *hamPtr_;
  Fourier&     fft = *fftPtr_;
  Spinor&      psi = *psiPtr_;

  std::vector<Atom>&   atomList = *atomListPtr_;
  Int numAtom = atomList.size();

  if( ham.IsHybrid() ) {
    ham.SetPhiEXX( psi, fft);
  }
  // print the options_ when first step. 
  if(k_ == 0){
    statusOFS<< std::endl;
    statusOFS<< " ----- TDDFT RK-4 Method Print Options   ----- "     << std::endl;
    statusOFS<< " options.auto_save      " << options_.auto_save      << std::endl;
    statusOFS<< " options.load_save      " << options_.load_save      << std::endl;
    statusOFS<< " options.method         " << options_.method         << std::endl;
    statusOFS<< " options.ehrenfest      " << options_.ehrenfest      << std::endl;
    statusOFS<< " options.simulateTime   " << options_.simulateTime   << std::endl;
    statusOFS<< " options.dt             " << options_.dt             << std::endl;
    statusOFS<< " options.gmres_restart  " << options_.gmres_restart  << std::endl;
    statusOFS<< " options.krylovTol      " << options_.krylovTol      << std::endl;
    statusOFS<< " options.diisTol        " << options_.diisTol        << std::endl;
    statusOFS<< " options.phiTol         " << options_.phiTol         << std::endl;
    statusOFS<< " options.adNum          " << options_.adNum          << std::endl;
    statusOFS<< " options.adUpdate       " << options_.adUpdate       << std::endl;
    statusOFS<< " --------------------------------------------- "     << std::endl;
    statusOFS<< std::endl;
  }

  // Update saved atomList. 0 is the latest one
  for( Int l = maxHist_-1; l > 0; l-- ){
    atomListHist_[l] = atomListHist_[l-1];
  }
  atomListHist_[0] = atomList;

  // do the verlocity verlet algorithm to move ion
  Int ionIter = k_;

  std::vector<Point3>  atompos(numAtom);
  std::vector<Point3>  atomvel(numAtom);
  std::vector<Point3>  atomvel_temp(numAtom);
  std::vector<Point3>  atomforce(numAtom);

  std::vector<Point3>  atompos_mid(numAtom);
  std::vector<Point3>  atompos_fin(numAtom);
  {

    Real& dt = options_.dt;
    DblNumVec& atomMass = atomMass_;
    // do not update force at the beginning

    for( Int a = 0; a < numAtom; a++ ){
      atompos[a]     = atomList[a].pos;
      atompos_mid[a] = atomList[a].pos;
      atompos_fin[a] = atomList[a].pos;
      atomvel[a]     = atomList[a].vel;
      atomforce[a]   = atomList[a].force;
    }

    PrintState( k_ );

    // Update velocity and position when doing ehrenfest dynamics
    if(options_.ehrenfest){

      for(Int a=0; a<numAtom; a++) {
        atomvel_temp[a] = atomvel[a]/2.0 + atomforce[a]*dt/atomMass[a]/8.0; 
        atompos_mid[a]  = atompos[a] + atomvel_temp[a] * dt;

        atomvel_temp[a] = atomvel[a] + atomforce[a]*dt/atomMass[a]/2.0; 
        atompos_fin[a]  = atompos[a] + atomvel_temp[a] * dt;

      }
      AdjustAtomPos( atompos_mid );
      AdjustAtomPos( atompos_fin );
    }
  }

  // have the atompos_mid and atompos_final
  if(options_.ehrenfest){
    for(Int a = 0; a < numAtom; a++){
      MPI_Bcast( &atompos_mid[a][0], 3, MPI_DOUBLE, 0, mpi_comm ); 
      MPI_Bcast( &atompos_fin[a][0], 3, MPI_DOUBLE, 0, mpi_comm ); 
    }
  }

  // k_ is the current K
  Complex i_Z_One = Complex(0.0, 1.0);
  // FIXME k is a confusing variable
  Int k = k_;
  Real ti = tlist_[k];
  Real tf = tlist_[k+1];
  Real dT = tf - ti;
  Real tmid =  (ti + tf)/2.0;
  //statusOFS << " step " << k_ << " ti " << ti << " tf " << tf << " dT " << dT << std::endl;

  // 4-th order Runge-Kutta  Start now 
  DblNumVec &occupationRate = ham.OccupationRate();
  occupationRate.Resize( psi.NumStateTotal() );
  SetValue( occupationRate, 1.0);
  //statusOFS << " Occupation Rate: " << occupationRate << std::endl;
  if(calDipole_)  CalculateDipole(tlist_[k_]);
  //if(k == 0) {
    Real totalCharge_;
    ham.CalculateDensity(
        psi,
        ham.OccupationRate(),
        totalCharge_, 
        fft );

#if ( _DEBUGlevel_ >= 2 )
    statusOFS << " total Charge init " << setw(16) << totalCharge_ << std::endl;
#endif
    //get the new V(r,t+dt) from the rho(r,t+dt)
    if( isCalculateGradRho_ ){
      ham.CalculateGradDensity( fft );
    }
    ham.CalculateXC( Exc_, fft ); 
    ham.CalculateHartree( fft );
    CalculateEfieldExt(ptable, ti);

#if ( _DEBUGlevel_ >= 2 )
    Real et0 = getEfield(0);
    Real eti = getEfield(ti);
    Real etmid = getEfield(tmid);
    Real etf = getEfield(tf);
    statusOFS << " DEBUG INFORMATION ON THE EFILED " << std::endl;
    statusOFS << " et0: " << et0 << " eti " << eti << " etmid " << etmid << " etf " << etf << std::endl;
    statusOFS << " DEBUG INFORMATION ON THE EFILED " << std::endl;
#endif
    ham.CalculateVtot( ham.Vtot() );

  //}


  // HX1 = (H1 * psi)
  Int ntot  = fft.domain.NumGridTotal();
  Int numStateLocal = psi.NumState();
  CpxNumMat Xtemp(ntot, numStateLocal); // X2, X3, X4
  CpxNumMat HX1(ntot, numStateLocal);
  NumTns<Complex> tnsTemp(ntot, 1, numStateLocal, false, HX1.Data());
  ham.MultSpinor( psi, tnsTemp, fft );

  Ekin_ = 0.0;
  {
    Real Ekin_temp = 0.0;	  
    CpxNumMat  XHX( numStateLocal, numStateLocal);
    blas::Gemm( 'C', 'N', numStateLocal, numStateLocal, ntot, 1.0, psi.Wavefun().Data(), 
        ntot, HX1.Data(), ntot, 0.0, XHX.Data(), numStateLocal );
    Complex * ptr = XHX.Data();
    Int numSpin = ham.NumSpin();
    for(int i =0; i < numStateLocal; i++)
      Ekin_temp += numSpin * ptr[i*numStateLocal+i].real();

    MPI_Allreduce( &Ekin_temp, &Ekin_, 1, MPI_DOUBLE_PRECISION, MPI_SUM, mpi_comm );

    CalculateEnergy( ptable, ti );
  }


  // test psi * conj(psi) 
#if ( _DEBUGlevel_ >= 2 )
  statusOFS<< "***************   DEBUG INFOMATION ******************" << std::endl << std::endl;
  statusOFS << " step: " << k_ << " K1 = -i1 * ( H1 * psi ) " << std::endl;
  Int width = numStateLocal;
  Int heightLocal = ntot;
  CpxNumMat  XTXtemp1( width, width );

  blas::Gemm( 'C', 'N', width, width, heightLocal, 1.0, psi.Wavefun().Data(), 
      heightLocal, psi.Wavefun().Data(), heightLocal, 0.0, XTXtemp1.Data(), width );

  for( Int i = 0; i < width; i++)
    statusOFS << " Psi * conjg( Psi) : "  << XTXtemp1(i,i) << std::endl;
  Complex *ptr = psi.Wavefun().Data();
  DblNumVec vtot = ham.Vtot();
  statusOFS << " Psi  0 : " << ptr[0] << " HX1 " << HX1(0,0) << " vtot " << vtot[0] << std::endl;
  statusOFS<< "***************   DEBUG INFOMATION ******************" << std::endl << std::endl;
#endif

  //  1. set up Psi <-- X - i ( HX1 ) * dt/2
  Complex* dataPtr = Xtemp.Data();
  Complex* psiDataPtr = psi.Wavefun().Data();
  Complex* HpsiDataPtr = HX1.Data();
  Int numStateTotal = psi.NumStateTotal();
  Spinor psi2 (fft.domain, 1, numStateTotal, numStateLocal, false, Xtemp.Data() );
  for( Int i = 0; i < numStateLocal; i ++)
    for( Int j = 0; j < ntot; j ++){
      Int index = i* ntot +j;
      dataPtr[index] = psiDataPtr[index] -  i_Z_One * HpsiDataPtr[index] * options_.dt/2.0;
    }

  // 2. if ehrenfest dynamics, re-calculate the Vlocal and Vnonlocal
  if(options_.ehrenfest){
    for( Int a = 0; a < numAtom; a++ ){
      atomList[a].pos  = atompos_mid[a];
    }
    ham.UpdateHamiltonian( atomList );
    ham.CalculatePseudoPotential( ptable );
  }

  if( ham.IsHybrid() ) {
    ham.SetPhiEXX( psi2, fft);
  }
  // 3. Update the H matrix. 
  {
    Real totalCharge_;
    ham.CalculateDensity(
        psi2,
        ham.OccupationRate(),
        totalCharge_, 
        fft );

    if( isCalculateGradRho_ ){
      ham.CalculateGradDensity( fft );
    }
    ham.CalculateXC( Exc_, fft ); 
    ham.CalculateHartree( fft );
    CalculateEfieldExt(ptable, tmid);
    ham.CalculateVtot( ham.Vtot() );
  }

  // 4. Calculate the K2 = H2 * X2
  CpxNumMat HX2(ntot, numStateLocal);
  NumTns<Complex> tnsTemp2(ntot, 1, numStateLocal, false, HX2.Data());
  ham.MultSpinor( psi2, tnsTemp2, fft );

  // check the psi * conj(psi)
#if ( _DEBUGlevel_ >= 2 )
  {
    statusOFS<< "***************   DEBUG INFOMATION ******************" << std::endl << std::endl;
    statusOFS << " step: " << k_ << " K2 = ( H2 * X2 ) " << std::endl;
    Int width = numStateLocal;
    Int heightLocal = ntot;
    CpxNumMat  XTXtemp1( width, width );

    blas::Gemm( 'C', 'N', width, width, heightLocal, 1.0, psi2.Wavefun().Data(), 
        heightLocal, psi2.Wavefun().Data(), heightLocal, 0.0, XTXtemp1.Data(), width );

    for( Int i = 0; i < width; i++)
      statusOFS << " Psi2 * conjg( Psi2) : " << XTXtemp1(i,i) << std::endl;
    Complex *ptr = psi2.Wavefun().Data();
    DblNumVec vtot = ham.Vtot();
    statusOFS << " Psi 2   0 : " << ptr[0]  << " HX2 " << HX2(0,0)<< " vtot " << vtot[0] << std::endl;
  }
  statusOFS<< "***************   DEBUG INFOMATION ******************" << std::endl << std::endl;
#endif 

  //  1. set up Psi <-- X - i ( HX2) * dt/2
  Spinor psi3 (fft.domain, 1, numStateTotal, numStateLocal, false, Xtemp.Data() );
  dataPtr = Xtemp.Data();
  psiDataPtr = psi.Wavefun().Data();
  HpsiDataPtr = HX2.Data();
  for( Int i = 0; i < numStateLocal; i ++)
    for( Int j = 0; j < ntot; j ++)
    {
      Int index = i* ntot +j;
      dataPtr[index] = psiDataPtr[index] -  i_Z_One * HpsiDataPtr[index] * options_.dt/2.0;
    }

  if( ham.IsHybrid() ) {
    ham.SetPhiEXX( psi3, fft);
  }
  // 2. if ehrenfest dynamics, re-calculate the Vlocal and Vnonlocal
  if(options_.ehrenfest){
    for( Int a = 0; a < numAtom; a++ ){
      atomList[a].pos  = atompos_mid[a];
    }
    ham.UpdateHamiltonian( atomList );
    ham.CalculatePseudoPotential( ptable );
  }

  // 3. Update the H matrix. 
  // CHECK CHECK: The Vext is zero, not updated here. 
  {
    Real totalCharge_;
    ham.CalculateDensity(
        psi3,
        ham.OccupationRate(),
        totalCharge_, 
        fft );
    if( isCalculateGradRho_ ){
      ham.CalculateGradDensity( fft );
    }
    ham.CalculateXC( Exc_, fft ); 
    ham.CalculateHartree( fft );
    ham.CalculateVtot( ham.Vtot() );
  }
  // 4. Calculate the K3 = H3 * X3
  CpxNumMat HX3(ntot, numStateLocal);
  NumTns<Complex> tnsTemp3(ntot, 1, numStateLocal, false, HX3.Data());
  ham.MultSpinor( psi3, tnsTemp3, fft );

  // check psi3*conj(psi3)
#if ( _DEBUGlevel_ >= 2 )
  {
    statusOFS<< "***************   DEBUG INFOMATION ******************" << std::endl << std::endl;
    statusOFS << " step: " << k_ << " K3 = -i1 * ( H3 * X3 ) " << std::endl;
    Int width = numStateLocal;
    Int heightLocal = ntot;
    CpxNumMat  XTXtemp1( width, width );

    blas::Gemm( 'C', 'N', width, width, heightLocal, 1.0, psi3.Wavefun().Data(), 
        heightLocal, psi3.Wavefun().Data(), heightLocal, 0.0, XTXtemp1.Data(), width );

    for( Int i = 0; i < width; i++)
      statusOFS << " Psi3 * conjg( Psi3) : " << XTXtemp1(i,i) << std::endl;
    Complex *ptr = psi3.Wavefun().Data();
    DblNumVec vtot = ham.Vtot();
    statusOFS << " Psi 3   0 : " << ptr[0]  << " HX3 " << HX2(0,0) << " vtot " << vtot[0] << std::endl;
    statusOFS<< "***************   DEBUG INFOMATION ******************" << std::endl << std::endl;
  }
#endif


  //  1. set up Psi <-- X - i ( HX3) * dt
  Spinor psi4 (fft.domain, 1, numStateTotal, numStateLocal, false, Xtemp.Data() );
  dataPtr = Xtemp.Data();
  psiDataPtr = psi.Wavefun().Data();
  HpsiDataPtr = HX3.Data();
  for( Int i = 0; i < numStateLocal; i ++)
    for( Int j = 0; j < ntot; j ++)
    {
      Int index = i* ntot +j;
      dataPtr[index] = psiDataPtr[index] -  i_Z_One * HpsiDataPtr[index] * options_.dt;
    }

  if( ham.IsHybrid() ) {
    ham.SetPhiEXX( psi4, fft);
  }
  // 2. if ehrenfest dynamics, re-calculate the Vlocal and Vnonlocal
  if(options_.ehrenfest){
    for( Int a = 0; a < numAtom; a++ ){
      atomList[a].pos  = atompos_fin[a];
    }
    ham.UpdateHamiltonian( atomList );
    ham.CalculatePseudoPotential( ptable );
  }

  // 3. Update the H matrix. 
  {
    Real totalCharge_;
    ham.CalculateDensity(
        psi4,
        ham.OccupationRate(),
        totalCharge_, 
        fft );

    if( isCalculateGradRho_ ){
      ham.CalculateGradDensity( fft );
    }
    ham.CalculateXC( Exc_, fft ); 
    ham.CalculateHartree( fft );
    CalculateEfieldExt(ptable, tf);
    ham.CalculateVtot( ham.Vtot() );
  }

  // 4. Calculate the K3 = H3 * X3
  // Now Hpsi is H2 * X2
  // K2 = -i1(H2 * psi) 
  CpxNumMat HX4(ntot, numStateLocal);
  NumTns<Complex> tnsTemp4(ntot, 1, numStateLocal, false, HX4.Data());
  ham.MultSpinor( psi4, tnsTemp4, fft );

#if ( _DEBUGlevel_ >= 2 )
  {
    statusOFS<< "***************   DEBUG INFOMATION ******************" << std::endl << std::endl;
    statusOFS << " step: " << k_ << " K4 = -i1 * ( H4 * X4 ) " << std::endl;
    Int width = numStateLocal;
    Int heightLocal = ntot;
    CpxNumMat  XTXtemp1( width, width );

    blas::Gemm( 'C', 'N', width, width, heightLocal, 1.0, psi4.Wavefun().Data(), 
        heightLocal, psi4.Wavefun().Data(), heightLocal, 0.0, XTXtemp1.Data(), width );

    for( Int i = 0; i < width; i++)
      statusOFS << " Psi4 * conjg( Psi4) : " << XTXtemp1(i,i) << std::endl;
    Complex *ptr = psi4.Wavefun().Data();
    DblNumVec vtot = ham.Vtot();
    statusOFS << " Psi 4   0 : " << ptr[0]  << " HX4 " << HX4(0,0)<< " vtot " << vtot[0] << std::endl;
    statusOFS<< "***************   DEBUG INFOMATION ******************" << std::endl << std::endl;
  }
#endif 

  // Xf <--  X - i ( K1 + 2K2 + 2K3 + K4) * dt / 6.0
  psiDataPtr = psi.Wavefun().Data();
  Complex *XfPtr = Xtemp.Data();
  Complex *K1    = HX1.Data();
  Complex *K2    = HX2.Data();
  Complex *K3    = HX3.Data();
  Complex *K4    = HX4.Data();
  for( Int i = 0; i < numStateLocal; i ++)
    for( Int j = 0; j < ntot; j ++){
      Int index = i* ntot +j;
      XfPtr[index] =  psiDataPtr[index] 
        - i_Z_One * ( K1[index] + 2.0*K2[index] 
            + 2.0*K3[index] + K4[index]) *options_.dt /6.0;
    }

  Spinor psiFinal (fft.domain, 1, numStateTotal, numStateLocal, false, Xtemp.Data() );

  // 2. if ehrenfest dynamics, re-calculate the Vlocal and Vnonlocal
  if(options_.ehrenfest){
    for( Int a = 0; a < numAtom; a++ ){
      atomList[a].pos  = atompos_fin[a];
    }
    ham.UpdateHamiltonian( atomList );
    ham.CalculatePseudoPotential( ptable );
  }

  //get the new V(r,t+dt) from the rho(r,t+dt)
  {
    Real totalCharge_;
    ham.CalculateDensity(
        psiFinal,
        ham.OccupationRate(),
        totalCharge_, 
        fft );

    if( isCalculateGradRho_ ){
      ham.CalculateGradDensity( fft );
    }
    ham.CalculateXC( Exc_, fft ); 
    ham.CalculateHartree( fft );
    DblNumVec vtot;
    Int ntotFine  = fft.domain.NumGridTotalFine();
    vtot.Resize(ntotFine);
    SetValue(vtot, 0.0);
    ham.CalculateVtot( vtot);
    Real *vtot0 = ham.Vtot().Data() ;
#if ( _DEBUGlevel_ >= 2 )
    statusOFS << "Xf delta vtot " << vtot(0) - vtot0[0] << std::endl;
#endif
    blas::Copy( ntotFine, vtot.Data(), 1, ham.Vtot().Data(), 1 );
  }

  psiDataPtr = psi.Wavefun().Data();
  Complex* psiDataPtrFinal = psiFinal.Wavefun().Data();
  for( Int i = 0; i < numStateLocal; i ++)
    for( Int j = 0; j < ntot; j ++){
      Int index = i* ntot +j;
      psiDataPtr[index] =  psiDataPtrFinal[index];
    }

#if ( _DEBUGlevel_ >= 2 )
  {
    statusOFS<< "***************   DEBUG INFOMATION ******************" << std::endl << std::endl;
    Int width = numStateLocal;
    Int heightLocal = ntot;
    CpxNumMat  XTXtemp1( width, width );

    blas::Gemm( 'C', 'N', width, width, heightLocal, 1.0, psi.Wavefun().Data(), 
        heightLocal, psi.Wavefun().Data(), heightLocal, 0.0, XTXtemp1.Data(), width );

    for( Int i = 0; i < width; i++)
      statusOFS << " Psi End * conjg( Psi End) : " << XTXtemp1(i,i) << std::endl;
    Complex *ptr = psi.Wavefun().Data();
    DblNumVec vtot = ham.Vtot();
    statusOFS << " Psi End 0 : " << ptr[0] << " vtot " << vtot[0] << std::endl;
    statusOFS<< "***************   DEBUG INFOMATION ******************" << std::endl << std::endl;
  }
#endif

  //update Velocity
  if(options_.ehrenfest){
    ham.CalculateForce( psi, fft);

    Real& dt = options_.dt;
    DblNumVec& atomMass = atomMass_;
    for( Int a = 0; a < numAtom; a++ ){
      atomList[a].vel = atomList[a].vel + (atomforce[a]/atomMass[a] + atomList[a].force/atomMass[a])*dt/2.0;
    } 
  }

  ++k_;
}

void TDDFT::advancePTTRAP( PeriodTable& ptable ) {

  Int mpirank, mpisize;
  MPI_Comm mpi_comm = fftPtr_->domain.comm;
  MPI_Comm_rank( mpi_comm, &mpirank );
  MPI_Comm_size( mpi_comm, &mpisize );

  Hamiltonian& ham = *hamPtr_;
  Fourier&     fft = *fftPtr_;
  Spinor&      psi = *psiPtr_;

  std::vector<Atom>&   atomList = *atomListPtr_;
  Int numAtom = atomList.size();

  // print the options_ when first step. 
  if(k_ == 0){
    statusOFS<< std::endl;
    statusOFS<< " -----   TDDFT PT-TRAP Print the Options  ---- "     << std::endl;
    statusOFS<< " options.auto_save      " << options_.auto_save      << std::endl;
    statusOFS<< " options.load_save      " << options_.load_save      << std::endl;
    statusOFS<< " options.method         " << options_.method         << std::endl;
    statusOFS<< " options.ehrenfest      " << options_.ehrenfest      << std::endl;
    statusOFS<< " options.simulateTime   " << options_.simulateTime   << std::endl;
    statusOFS<< " options.dt             " << options_.dt             << std::endl;
    statusOFS<< " options.gmres_restart  " << options_.gmres_restart  << std::endl;
    statusOFS<< " options.krylovTol      " << options_.krylovTol      << std::endl;
    statusOFS<< " options.diisTol        " << options_.diisTol        << std::endl;
    statusOFS<< " options.phiTol         " << options_.phiTol         << std::endl;
    statusOFS<< " options.adNum          " << options_.adNum          << std::endl;
    statusOFS<< " options.adUpdate       " << options_.adUpdate       << std::endl;
    statusOFS<< " --------------------------------------------- "     << std::endl;
    statusOFS<< std::endl;
  }

  // Update saved atomList. 0 is the latest one
  for( Int l = maxHist_-1; l > 0; l-- ){
    atomListHist_[l] = atomListHist_[l-1];
  }
  atomListHist_[0] = atomList;

  // do the verlocity verlet algorithm to move ion
  Int ionIter = k_;

  std::vector<Point3>  atompos(numAtom);
  std::vector<Point3>  atomvel(numAtom);
  std::vector<Point3>  atomforce(numAtom);
  std::vector<Point3>  atompos_fin(numAtom);
  {
    std::vector<Point3>  atomvel_temp(numAtom);
    // do not update force at the beginning

    Real& dt = options_.dt;
    DblNumVec& atomMass = atomMass_;

    for( Int a = 0; a < numAtom; a++ ){
      atompos[a]     = atomList[a].pos;
      atompos_fin[a] = atomList[a].pos;
      atomvel[a]     = atomList[a].vel;
      atomforce[a]   = atomList[a].force;
    }

    PrintState( k_ );

    // Update velocity and position when doing ehrenfest dynamics
    if(options_.ehrenfest){
      for(Int a=0; a<numAtom; a++) {
        atomvel_temp[a] = atomvel[a] + atomforce[a]*dt/atomMass[a]/2.0; 
        atompos_fin[a]  = atompos[a] + atomvel_temp[a] * dt;

      }
      AdjustAtomPos( atompos_fin );
    }
  }

  // have the atompos_final
  if(options_.ehrenfest){
    for(Int a = 0; a < numAtom; a++){
      MPI_Bcast( &atompos_fin[a][0], 3, MPI_DOUBLE, 0, mpi_comm); 
    }
  }

  // k_ is the current K
  Int k = k_;
  Real ti = tlist_[k];
  Real tf = tlist_[k+1];
  Real dT = tf - ti;
  Real tmid =  (ti + tf)/2.0;
  Complex i_Z_One = Complex(0.0, 1.0);

  // PT-TRAP Method starts, note we only use Ne bands
  DblNumVec &occupationRate = ham.OccupationRate();
  occupationRate.Resize( psi.NumStateTotal() );
  SetValue( occupationRate, 1.0);


  // update H when it is first step.
  if(k == esdfParam.restartTDDFTStep) {
    if(!esdfParam.isRestartDensity){
      Real totalCharge_;
      ham.CalculateDensity(
          psi,
          ham.OccupationRate(),
          totalCharge_, 
          fft );
    }
    if( isCalculateGradRho_ ){
      ham.CalculateGradDensity( fft );
    }
    ham.CalculateXC( Exc_, fft ); 
    ham.CalculateHartree( fft );
    CalculateEfieldExt(ptable, ti); // ti is 0
    ham.CalculateVtot( ham.Vtot() );
  }


  // calculate Dipole at the beginning.
  if(calDipole_)  CalculateDipole(tlist_[k_]);

  // 1. Calculate Xmid which appears on the right hand of the equation
  // HPSI = (H1 * psi)
  Int ntot  = fft.domain.NumGridTotal();
  Int numStateLocal = psi.NumState();
  Int ntotLocal = ntot/mpisize;
  if(mpirank < (ntot % mpisize)) ntotLocal++;
  Int numStateTotal = psi.NumStateTotal();
  CpxNumMat HPSI(ntot, numStateLocal);
  NumTns<Complex> tnsTemp(ntot, 1, numStateLocal, false, HPSI.Data());
  ham.MultSpinor( psi, tnsTemp, fft );

  // All X's are in G-parallel
  CpxNumMat X(ntotLocal, numStateTotal); 
  CpxNumMat HX(ntotLocal, numStateTotal); 
  CpxNumMat RX(ntotLocal, numStateTotal); 
  CpxNumMat Xmid(ntotLocal, numStateTotal); 
  CpxNumMat Xfin(ntotLocal, numStateTotal); 
  CpxNumMat Ymid(ntotLocal, numStateTotal); 
  CpxNumMat Yfin(ntotLocal, numStateTotal); 
  CpxNumMat XF  (ntotLocal, numStateTotal);

  // psi and HPSI in Band parallel 
  CpxNumMat psiF  ( ntot, numStateLocal );
  CpxNumMat psiCol( ntot, numStateLocal );
  CpxNumMat psiYmid( ntot, numStateLocal );
  CpxNumMat psiYfin( ntot, numStateLocal );
  lapack::Lacpy( 'A', ntot, numStateLocal, psi.Wavefun().Data(), ntot, psiCol.Data(), ntot );

  // tranfer psi and Hpsi from band-parallel to G-parallel
  AlltoallForward( HPSI,  HX, mpi_comm);
  AlltoallForward( psiCol, X, mpi_comm);
  lapack::Lacpy( 'A', ntotLocal, numStateTotal, X.Data(), ntotLocal, XF.Data(), ntotLocal );

  // RX <-- HX - X*(X'*HX)
  Int width = numStateTotal;
  Int heightLocal = ntotLocal;
  CpxNumMat  XHXtemp( width, width );
  CpxNumMat  XHX( width, width );
  lapack::Lacpy( 'A', ntotLocal, numStateTotal, HX.Data(), ntotLocal, RX.Data(), ntotLocal );
  blas::Gemm( 'C', 'N', width, width, heightLocal, 1.0, X.Data(), 
      heightLocal, HX.Data(), heightLocal, 0.0, XHXtemp.Data(), width );
  MPI_Allreduce( XHXtemp.Data(), XHX.Data(), width*width, MPI_DOUBLE_COMPLEX, MPI_SUM, mpi_comm );
  blas::Gemm( 'N', 'N', heightLocal, width, width, -1.0, 
      X.Data(), heightLocal, XHX.Data(), width, 1.0, RX.Data(), heightLocal );

  // check check
  // E_kin = numSpin * trace( XHX )
  Ekin_ = 0.0;
  {
    Complex * ptr = XHX.Data();
    Int numSpin = ham.NumSpin();
    for(int i =0; i < width; i++)
      Ekin_ += numSpin * ptr[i*width+i].real();
  }

  CalculateEnergy( ptable, ti );

  // Xmid <-- X - li*T/2 * RX  in G-parallel
  {
    Complex * xmidPtr = Xmid.Data();
    Complex * xPtr    = X.Data();
    Complex * rxPtr   = RX.Data();
    for( Int i = 0; i < numStateTotal; i ++)
      for( Int j = 0; j < ntotLocal; j ++){
        Int index = i* ntotLocal +j;
        xmidPtr[index] = xPtr[index] -  i_Z_One * dT/2.0 * rxPtr[index];
      }
  }

  // Xf <-- X
  Spinor psiFinal (fft.domain, 1, numStateTotal, numStateLocal, false, psiF.Data() );
  lapack::Lacpy( 'A', ntot, numStateLocal, psi.Wavefun().Data(), ntot, psiF.Data(), ntot );

  // move the atom from atom_begin to atom_final, then recalculate the
  if(options_.ehrenfest){
    for( Int a = 0; a < numAtom; a++ ){
      atomList[a].pos  = atompos_fin[a];
    }
    ham.UpdateHamiltonian( atomList );
    ham.CalculatePseudoPotential( ptable );
  }

  // get the charge density, but not update the H matrix
  {
    CalculateEfieldExt(ptable, tf); // tf is the current step, calculate only once.

    // get the charge density of the Hf.
    Real totalCharge_;
    ham.CalculateDensity(
        psiFinal,
        ham.OccupationRate(),
        totalCharge_, 
        fft );
  }

  Int maxscfiter = options_.diisMaxIter; 
  int iscf;
  int totalHx = 0;
  for (iscf = 0; iscf < maxscfiter; iscf++){

    // update the Hf matrix, note rho is calculated before.
    {
      if( isCalculateGradRho_ ){
        ham.CalculateGradDensity( fft );
      }
      ham.CalculateXC( Exc_, fft ); 
      ham.CalculateHartree( fft );
      DblNumVec vtot;
      Int ntotFine  = fft.domain.NumGridTotalFine();
      vtot.Resize(ntotFine);
      SetValue(vtot, 0.0);
      ham.CalculateVtot( vtot);
      Real *vtot0 = ham.Vtot().Data() ;
      blas::Copy( ntotFine, vtot.Data(), 1, ham.Vtot().Data(), 1 );
    }

    // HXf <--- Hf * Xf , Now HPSI is HXf
    ham.MultSpinor( psiFinal, tnsTemp, fft );

    // XHXtemp <--- X'HXf
    AlltoallForward( HPSI,  HX, mpi_comm);
    AlltoallForward( psiF, X, mpi_comm);

    blas::Gemm( 'C', 'N', width, width, heightLocal, 1.0, X.Data(), 
        heightLocal, HX.Data(), heightLocal, 0.0, XHXtemp.Data(), width );

    MPI_Allreduce( XHXtemp.Data(), XHX.Data(), width*width, MPI_DOUBLE_COMPLEX, MPI_SUM, mpi_comm );

    // XHX <-- 0.5(XHX + XHX') 
    {
      Complex * xPtr = XHXtemp.Data();
      Complex * yPtr = XHX.Data();
      for(int i = 0; i < width; i++){
        for(int j = 0; j < width; j++){
          xPtr[i*width + j] = 0.5 * ( yPtr[i*width+j] + std::conj(yPtr[j*width+i]) );
        }
      }
    }

    // Diag XHX for the eigen value and eigen vectors
    DblNumVec  eigValS(width);
    lapack::Syevd( 'V', 'U', width, XHXtemp.Data(), width, eigValS.Data() );

    // YpsiMid<-- XpsiMid * XHX 
    blas::Gemm( 'N', 'N', heightLocal, width, width, 1.0, 
        Xmid.Data(), heightLocal, XHXtemp.Data(), width, 0.0, Ymid.Data(), heightLocal );

    // YpsiF <-- Xf * XHX
    blas::Gemm( 'N', 'N', heightLocal, width, width, 1.0, 
        XF.Data(), heightLocal, XHXtemp.Data(), width, 0.0, Yfin.Data(), heightLocal );

    // change from G-para to Band-parallel
    AlltoallBackward( Ymid, psiYmid, mpi_comm);
    AlltoallBackward( Yfin, psiYfin, mpi_comm);


    for( int j = 0; j < numStateLocal; j++){

      int index = j * mpisize + mpirank;
      // psiYmid now is rhs
      Complex omega = eigValS(index) + 2.0 * i_Z_One / dT; 
      blas::Scal( ntot, -2.0*i_Z_One, psiYmid.Data() + j*ntot, 1 );

      // call the SGMRES here.
      sgmres_solver.Solve( psiYmid.Data() + j* ntot, psiYfin.Data() + j* ntot, omega);

    }


    // Change the Parallelization
    AlltoallForward ( psiYfin, XF, mpi_comm);

    // Xf.psi = Ypsif * Cf', where Ypsif is the psi get from GMRES, check check
    blas::Gemm( 'N', 'C', heightLocal, width, width, 1.0, 
        XF.Data(), heightLocal, XHXtemp.Data(), width, 0.0, X.Data(), heightLocal );

    // Change the Parallelization
    AlltoallBackward( X, psiF, mpi_comm);
    lapack::Lacpy( 'A', ntotLocal, numStateTotal, X.Data(), ntotLocal, XF.Data(), ntotLocal );

    // get the density
    {
      Real totalCharge_;
      ham.CalculateDensity(
          psiFinal,
          ham.OccupationRate(),
          totalCharge_, 
          fft );

      Int ntotFine  = fft.domain.NumGridTotalFine();
      DblNumVec vtotNew(ntotFine);
      if( isCalculateGradRho_ ){
        ham.CalculateGradDensity( fft );
      }
      ham.CalculateXC( Exc_, fft ); 
      ham.CalculateHartree( fft );
      ham.CalculateVtot( vtotNew );

      Real normVtotDif = 0.0, normVtotOld = 0.0;
      DblNumVec& vtotOld_ = ham.Vtot();
      Int ntot = vtotOld_.m();
      for( Int i = 0; i < ntot; i++ ){
        normVtotDif += pow( vtotOld_(i) - vtotNew(i), 2.0 );
        normVtotOld += pow( vtotOld_(i), 2.0 );
      }
      normVtotDif = sqrt( normVtotDif );
      normVtotOld = sqrt( normVtotOld );
      Real scfNorm_    = normVtotDif / normVtotOld;

      Print(statusOFS, "norm(out-in)/norm(in) = ", scfNorm_ );
      totalHx += sgmres_solver.iter_;

      if( scfNorm_ < options_.diisTol){
        /* converged */
        statusOFS << "TDDFT step " << k_ << " SCF is converged in " << iscf << " steps !" << std::endl;
        statusOFS << "TDDFT step " << k_ << " used " << totalHx << " H * x operations!" << std::endl;
        break; // break if converged. 
      }

      /*
         statusOFS << " iscf " << iscf + 1 << " mixStepLength " << esdfParam.mixStepLength
         << " " << esdfParam.mixType << std::endl;

         blas::Copy( ntotFine, vtotNew.Data(), 1, ham.Vtot().Data(), 1 );
       */

      AndersonMix(
          iscf+1,
          esdfParam.mixStepLength,
          esdfParam.mixType,
          ham.Vtot(),
          vtotOld_,
          vtotNew,
          dfMat_,
          dvMat_);

    }
  }

  // Reorthogonalize
  {
    blas::Gemm( 'C', 'N', width, width, heightLocal, 1.0, X.Data(), 
        heightLocal, X.Data(), heightLocal, 0.0, XHXtemp.Data(), width );
    MPI_Allreduce( XHXtemp.Data(), XHX.Data(), width*width, MPI_DOUBLE_COMPLEX, MPI_SUM, mpi_comm );

    // XHXtemp = 0.5 * ( XHX + conj ( XHX ) )
    {
      Complex * xPtr = XHXtemp.Data();
      Complex * yPtr = XHX.Data();
      for(int i = 0; i < width; i++){
        for(int j = 0; j < width; j++){
          xPtr[i*width + j] = 0.5 * ( yPtr[i*width+j] + std::conj(yPtr[j*width+i]) );
        }
      }
    }

    DblNumVec  eigValS(width);
    lapack::Syevd( 'V', 'U', width, XHXtemp.Data(), width, eigValS.Data() );

    CpxNumMat temp( width, width );
    SetValue( temp, Complex(0.0, 0.0) );
    for(int i = 0; i < width; i++) {
      temp(i,i) = Complex( 1.0 / sqrt( eigValS[i] ), 0.0);
    }

    blas::Gemm( 'N', 'N', width, width, width, 1.0, XHXtemp.Data(),
        width, temp.Data(), width, 0.0, XHX.Data(), width );

    blas::Gemm( 'N', 'C', width, width, width, 1.0, XHX.Data(),
        width, XHXtemp.Data(), width, 0.0, temp.Data(), width );

    blas::Gemm( 'N', 'N', heightLocal, width, width, 1.0, X.Data(),
        heightLocal, temp.Data(), width, 0.0, HX.Data(), heightLocal );

    AlltoallBackward ( HX, psiF, mpi_comm );
  }

  // psi <--- psiFinal
  blas::Copy( ntot*numStateLocal, psiFinal.Wavefun().Data(), 1, psi.Wavefun().Data(), 1 );

  // Update the atomic position and the force.
  if(options_.ehrenfest){
    ham.CalculateForce( psi, fft);

    Real& dt = options_.dt;
    DblNumVec& atomMass = atomMass_;
    for( Int a = 0; a < numAtom; a++ ){
      atomList[a].vel = atomList[a].vel + (atomforce[a]/atomMass[a] + atomList[a].force/atomMass[a])*dt/2.0;
    } 
  }

  //Update the anderson mixing 
  Update();

  ++k_;

} // TDDFT:: advancePTTRAP



void TDDFT::advancePTTRAPDIIS( PeriodTable& ptable ) {

  Int mpirank, mpisize;
  MPI_Comm mpi_comm = fftPtr_->domain.comm;
  MPI_Comm_rank( mpi_comm, &mpirank );
  MPI_Comm_size( mpi_comm, &mpisize );

  Real timeSta, timeEnd;
  Real timeSta1, timeEnd1;

  Real timeDF = 0.0;
  Real timeSetPhi = 0.0;
  Real timeCalACE = 0.0;
  Real timeCalExxEnergy = 0.0;
  Real timeDIISSCF = 0.0;
  Real timeInit = 0.0;
  Real timeDIIS = 0.0;
  Real timeOrth = 0.0;
  Real timeDensity = 0.0;
  Real timeForce = 0.0;
  Int  iterDF = 0;
  Int  iterSetPhi  = 0;
  Int  iterCalACE  = 0;
  Int  iterDIISSCF = 0;
  Int  iterDIIS    = 0;
  Int  iterOrth    = 0;
  Int  iterDensity = 0;
  Int  iterCalExxEnergy  = 0;
  Int  iterForce   = 0;

  GetTime( timeSta );

  Hamiltonian& ham = *hamPtr_;
  Fourier&     fft = *fftPtr_;
  Spinor&      psi = *psiPtr_;

  if( ham.IsHybrid() ) {
    ham.SetPhiEXX( psi, fft);
  }
#if ( _DEBUGlevel_ >= 0 )
    //statusOFS << "lijl psi = " << psi.Wavefun() << std::endl;
#endif
  std::vector<Atom>&   atomList = *atomListPtr_;
  Int numAtom = atomList.size();

  // print the options_ when first step. 
  if(k_ == 0) {
    statusOFS<< std::endl;
    statusOFS<< " -----   TDDFT PT-TRAP DIIS Print Options ---- "     << std::endl;
    statusOFS<< " options.auto_save      " << options_.auto_save      << std::endl;
    statusOFS<< " options.load_save      " << options_.load_save      << std::endl;
    statusOFS<< " options.method         " << options_.method         << std::endl;
    statusOFS<< " options.ehrenfest      " << options_.ehrenfest      << std::endl;
    statusOFS<< " options.simulateTime   " << options_.simulateTime   << std::endl;
    statusOFS<< " options.dt             " << options_.dt             << std::endl;
    statusOFS<< " options.gmres_restart  " << options_.gmres_restart  << std::endl;
    statusOFS<< " options.krylovTol      " << options_.krylovTol      << std::endl;
    statusOFS<< " options.diisTol        " << options_.diisTol        << std::endl;
    statusOFS<< " options.phiTol         " << options_.phiTol         << std::endl;
    statusOFS<< " options.adNum          " << options_.adNum          << std::endl;
    statusOFS<< " options.adUpdate       " << options_.adUpdate       << std::endl;
    statusOFS<< " -----   TDDFT PT-TRAP DIIS Print Options ---- "     << std::endl;
    statusOFS<< std::endl;
  }

  // Update saved atomList. 0 is the latest one
  for( Int l = maxHist_-1; l > 0; l-- ){
    atomListHist_[l] = atomListHist_[l-1];
  }
  atomListHist_[0] = atomList;

  Int ionIter = k_;

  std::vector<Point3>  atompos(numAtom);
  std::vector<Point3>  atomvel(numAtom);
  std::vector<Point3>  atomforce(numAtom);
  std::vector<Point3>  atompos_fin(numAtom);
  {
    // do not update force at the beginning

    Real& dt = options_.dt;
    DblNumVec& atomMass = atomMass_;

    for( Int a = 0; a < numAtom; a++ ){
      atompos[a]     = atomList[a].pos;
      atompos_fin[a] = atomList[a].pos;
      atomvel[a]     = atomList[a].vel;
      atomforce[a]   = atomList[a].force;
    }

    PrintState( k_ );


    // Update velocity and position when doing ehrenfest dynamics
    if(options_.ehrenfest){
      for(Int a=0; a<numAtom; a++) {
        atompos_fin[a]  = atompos[a] + atomvel[a] * dt + atomforce[a]*(dt*dt)/(atomMass[a]*2.0);
      }  
      AdjustAtomPos( atompos_fin );
    }
  }

  if(options_.ehrenfest){
    for(Int a = 0; a < numAtom; a++){
      MPI_Bcast( &atompos_fin[a][0], 3, MPI_DOUBLE, 0, mpi_comm); 
    }
  }

  // k_ is the current K
  Int k = k_;
  Real ti = tlist_[k];
  Real tf = tlist_[k+1];
  Real dT = tf - ti;
  Real tmid =  (ti + tf)/2.0;
  Complex i_Z_One = Complex(0.0, 1.0);

  // PT-TRAP DIIS Method, Occupation = 1
  DblNumVec &occupationRate = ham.OccupationRate();
  occupationRate.Resize( psi.NumStateTotal() );
  SetValue( occupationRate, 1.0);

  // update H when it is first step. 
  // This can be avoided since it is 
  // already converged in the first SCF.
  if(k == esdfParam.restartTDDFTStep) {
    //if(!esdfParam.isRestartDensity){
    if(1){
	    statusOFS << " always start by calculating Density from WFN " << std::endl;
      Real totalCharge_;
      ham.CalculateDensity(
          psi,
          ham.OccupationRate(),
          totalCharge_, 
          fft );
    }
    if( isCalculateGradRho_ ){
      ham.CalculateGradDensity( fft );
    }
    ham.CalculateXC( Exc_, fft ); 
    ham.CalculateHartree( fft );
    CalculateEfieldExt(ptable, ti); 
    ham.CalculateVtot( ham.Vtot() );
  }

  // calculate Dipole at the beginning.
  if(calDipole_)  CalculateDipole(tlist_[k_]);

  // 1. Calculate Xmid which appears on the right hand of the equation
  // HPSI = (H1 * psi)
  Int ntot  = fft.domain.NumGridTotal();
  Int numStateLocal = psi.NumState();
  Int ntotLocal = ntot/mpisize;
  if(mpirank < (ntot % mpisize)) ntotLocal++;
  Int numStateTotal = psi.NumStateTotal();
  CpxNumMat HPSI(ntot, numStateLocal);
  NumTns<Complex> tnsTemp(ntot, 1, numStateLocal, false, HPSI.Data());
  ham.MultSpinor( psi, tnsTemp, fft );

  //  All X's are in G-parallel
  CpxNumMat X(ntotLocal, numStateTotal); 
  CpxNumMat HX(ntotLocal, numStateTotal); 
  CpxNumMat RX(ntotLocal, numStateTotal); 
  CpxNumMat Xmid(ntotLocal, numStateTotal); 

  // All psi's are in Band-parallel
  CpxNumMat psiF  ( ntot, numStateLocal );
  CpxNumMat psiCol( ntot, numStateLocal );
  CpxNumMat psiRes( ntot, numStateLocal );

  lapack::Lacpy( 'A', ntot, numStateLocal, psi.Wavefun().Data(), ntot, psiCol.Data(), ntot );

  AlltoallForward( HPSI,  HX, mpi_comm);
  AlltoallForward( psiCol, X, mpi_comm);

  // RX <-- HX - X*(X'*HX)
  Int width = numStateTotal;
  Int heightLocal = ntotLocal;
  CpxNumMat  XHXtemp( width, width );
  CpxNumMat  XHX( width, width );
  lapack::Lacpy( 'A', ntotLocal, numStateTotal, HX.Data(), ntotLocal, RX.Data(), ntotLocal );
  blas::Gemm( 'C', 'N', width, width, heightLocal, 1.0, X.Data(), 
      heightLocal, HX.Data(), heightLocal, 0.0, XHXtemp.Data(), width );
  MPI_Allreduce( XHXtemp.Data(), XHX.Data(), width*width, MPI_DOUBLE_COMPLEX, MPI_SUM, mpi_comm );
  blas::Gemm( 'N', 'N', heightLocal, width, width, -1.0, 
      X.Data(), heightLocal, XHX.Data(), width, 1.0, RX.Data(), heightLocal );


  // check check
  // E_kin = numSpin * trace( XHX )
  Ekin_ = 0.0;
  {
    Complex * ptr = XHX.Data();
    Int numSpin = ham.NumSpin();
    for(int i =0; i < width; i++)
      Ekin_ += numSpin * ptr[i*width+i].real();
  }

  CalculateEnergy( ptable, ti );

  // Xmid <-- X - li*T/2 * RX  in G-parallel
  {
    Complex * xmidPtr = Xmid.Data();
    Complex * xPtr    = X.Data();
    Complex * rxPtr   = RX.Data();
    for( Int i = 0; i < numStateTotal; i ++)
      for( Int j = 0; j < ntotLocal; j ++){
        Int index = i* ntotLocal +j;
        xmidPtr[index] = xPtr[index] -  i_Z_One * dT/2.0 * rxPtr[index];
      }
  }

  Spinor psiFinal (fft.domain, 1, numStateTotal, numStateLocal, false, psiF.Data() );
  if(0){
    // psiF <== psi
    lapack::Lacpy( 'A', ntot, numStateLocal, psi.Wavefun().Data(), ntot, psiF.Data(), ntot );
  }
  if(1){
    // psiF <== psi - i * dT * RX
    AlltoallBackward( RX, psiRes, mpi_comm);

    lapack::Lacpy( 'A', ntot, numStateLocal, psi.Wavefun().Data(), ntot, psiF.Data(), ntot );
    blas::Axpy( ntot, - i_Z_One * dT, psiRes.Data(), 1, psiF.Data(), 1 );
  }

  // AtomPos <== AtomPosFinal, then update Vatom
  if(options_.ehrenfest){
    for( Int a = 0; a < numAtom; a++ ){
      atomList[a].pos  = atompos_fin[a];
    }
    ham.UpdateHamiltonian( atomList );
    ham.CalculatePseudoPotential( ptable );
  }

  // rhoF <== update Charge Density.
  DblNumMat          rhoFinal; 
  {
    CalculateEfieldExt(ptable, tf); // tf is the current step, calculate only once.

    // get the charge density of the Hf.
    Real totalCharge_;
    ham.CalculateDensity(
        psiFinal,
        ham.OccupationRate(),
        totalCharge_, 
        fft );
    Int ntotFine  = fftPtr_->domain.NumGridTotalFine();
    rhoFinal.Resize (ntotFine, 1);  

    Real * densityPtr = ham.Density().Data();
    Real * rhoFinalPtr= rhoFinal.Data();
    for(int i = 0; i < ntotFine; i++) {
      rhoFinalPtr[i] = densityPtr[i];
    }
  }

  GetTime( timeEnd );
  timeInit += timeEnd - timeSta;
  //statusOFS << " TDDFT Step " << k_ << " Setting-up Time: " << timeEnd - timeSta << " [s]" << std::endl;


  Int numGridTotal = ntot;

  if(1){

    GetTime( timeSta1 );
    Int maxScfIteration = options_.diisMaxIter;
    Real betaMix = esdfParam.mixStepLength;
    Int  maxDim  = esdfParam.mixMaxDim;

    std::vector<CpxNumMat>   dfMat;
    std::vector<CpxNumMat>   dvMat;
    dfMat.resize( numStateLocal );
    dvMat.resize( numStateLocal );
    for( int i = 0; i < numStateLocal; i++) { 
      dfMat[i].Resize( ntot, maxDim ); 
      dvMat[i].Resize( ntot, maxDim ); 
      SetValue( dfMat[i], Complex(0.0, 0.0) );
      SetValue( dvMat[i], Complex(0.0, 0.0) );
    }

    CpxNumVec vin;
    CpxNumVec vout;
    vin.Resize( ntot);
    vout.Resize( ntot);
    SetValue( vin,  Complex(0,0));
    SetValue( vout, Complex(0,0));

    GetTime( timeEnd1 );
    timeDF += timeEnd1 - timeSta1;
    iterDF ++;

    Real scfNorm = 0.0;
    if( esdfParam.isHybridACE ) {

      statusOFS << "TDDFT is using Hybrid ACE Operator ...." << std::endl;

      Real fock1 = 0.0;
      Real fock2 = 0.0;

      // Two SCF loops, outer and inner SCF.
      // Outer SCF
      int maxPhiIteration = options_.phiMaxIter;

      // new scheme: get E[V] <== V[psi_0] <== psi_0
      GetTime( timeSta1 );
      ham.SetPhiEXX( psiFinal, fft);
      GetTime( timeEnd1 );
      timeSetPhi += timeEnd1 - timeSta1;
      iterSetPhi ++;

      GetTime( timeSta1 );
      bool isFixColumnDF = false;
      //if( esdfParam.isHybridDF ){
      if(0){
        statusOFS << " ACEDF: first iteration " << std::endl;
        ham.CalculateVexxACEDF( psi, fft, isFixColumnDF );
        isFixColumnDF = true;
      }
      else
      {
        ham.CalculateVexxACE ( psi, fft );
      }
      GetTime( timeEnd1 );
      timeCalACE += timeEnd1 - timeSta1;
      iterCalACE ++;
#if ( _DEBUGlevel_ >= 0 )
    //statusOFS << "psi = " << psi.Wavefun() << std::endl;
    //statusOFS << "psiFinal = " << psiFinal.Wavefun() << std::endl;
#endif
      GetTime( timeSta1 );
      fock1 = ham.CalculateEXXEnergy( psiFinal, fft ); 
      GetTime( timeEnd1 );
      statusOFS << " fock1 ACEDF lijl :" << fock1 << std::endl;
      timeCalExxEnergy += timeEnd1 - timeSta1;
      iterCalExxEnergy ++;

      for( int phiIter = 0; phiIter < maxPhiIteration; phiIter++){

        // Inner SCF.
        GetTime( timeSta1 );
	int iscf;
        for( iscf = 1; iscf <= maxScfIteration; iscf++ ) {
          scfNorm = InnerSolve( iscf, psiFinal, tnsTemp, HX, X, HPSI, psiF, XHX, XHXtemp, RX, Xmid, dT, psiRes, vin, vout, dfMat, dvMat, rhoFinal);
          if( scfNorm < options_.diisTol){
            break;
          }
        }
        GetTime( timeEnd1 );
        timeDIISSCF += timeEnd1 - timeSta1;
        iterDIISSCF += iscf;

        if( scfNorm < options_.diisTol)
          statusOFS << "phiStep " << phiIter << " DIIS is  converged in " << iscf << " steps " << " scfNorm " << scfNorm << std::endl;
	else 
          statusOFS << "phiStep " << phiIter << " DIIS NOT converged in " << iscf << " steps " << " scfNorm " << scfNorm << std::endl;

        // new scheme: get E[V] <== V[psi_0] <== psi_0
        GetTime( timeSta1 );
        ham.SetPhiEXX( psiFinal, fft);
        GetTime( timeEnd1 );
        timeSetPhi += timeEnd1 - timeSta1;
        iterSetPhi ++;

        GetTime( timeSta1 );
        if( esdfParam.isHybridDF ){
            if( phiIter == 0){
                isFixColumnDF = false;
                ham.CalculateVexxACEDF( psiFinal, fft, isFixColumnDF );
                isFixColumnDF = true;
            }
            else{
                ham.CalculateVexxACEDF( psiFinal, fft, isFixColumnDF );
                isFixColumnDF = true;
            }
        }
        else{
            ham.CalculateVexxACE ( psiFinal, fft );
        }

        GetTime( timeEnd1 );
        timeCalACE += timeEnd1 - timeSta1;
        iterCalACE ++;

        GetTime( timeSta1 );
        fock2 = ham.CalculateEXXEnergy( psiFinal, fft ); 
        GetTime( timeEnd1 );
        timeCalExxEnergy += timeEnd1 - timeSta1;
        iterCalExxEnergy ++;

        Real dExx = std::abs(fock2 - fock1) / std::abs(fock2);

        statusOFS << " Fock Energy  = " << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< fock2 << " [au]" << std::endl 
                  << " dExx         = " << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< dExx  << " [au]" << std::endl;

	fock1 = fock2;
        if( dExx < options_.phiTol) {
          statusOFS << "TDDFT step " << k_ << " Phi Iteration in " << phiIter + 1<< " steps !" << std::endl;
	  break; 
	}
      }
    } 
    else {

      if( ham.IsHybrid() && !esdfParam.isHybridACE ) 
        statusOFS << "TDDFT screen exchange ... " << std::endl;
      else
        statusOFS << "TDDFT PBE ... " << std::endl;

      // Note, the exact HF and PBE implementation together. 
      for(int iscf = 1; iscf <= maxScfIteration; iscf++){
        scfNorm = InnerSolve( iscf, psiFinal, tnsTemp, HX, X, HPSI, psiF, XHX, XHXtemp, RX, Xmid, dT, psiRes, vin, vout, dfMat, dvMat, rhoFinal);
        if( scfNorm < options_.diisTol){
          statusOFS << "TDDFT step " << k_ << " SCF is converged in " << iscf << " steps !" << std::endl;
          break;
        }
      }
    }
#if 0
      Int iterused = std::min (iscf-1, maxDim);
      Int ipos = iscf - 1 - floor( (iscf-2) / maxDim ) * maxDim;

      // Update Hf <== updateV(molf, rhof)
      Int ntotFine  = fft.domain.NumGridTotalFine();
      {
        if( isCalculateGradRho_ ){
          ham.CalculateGradDensity( fft );
        }
        ham.CalculateXC( Exc_, fft ); 
        ham.CalculateHartree( fft );
        ham.CalculateVtot( ham.Vtot());
      }

      if( ham.IsHybrid() ) {
        ham.SetPhiEXX( psiFinal, fft);
      }
      // HXf <== Hf * Xf, now HPSI is HXf  
      ham.MultSpinor( psiFinal, tnsTemp, fft );

      //  XHX <== XHXtemp <--- X'HXf
      //  PsiF, HPSI are psiF and H*psiF
      AlltoallForward( HPSI, HX, mpi_comm);
      AlltoallForward( psiF, X,  mpi_comm);

      blas::Gemm( 'C', 'N', width, width, heightLocal, 1.0, X.Data(), 
          heightLocal, HX.Data(), heightLocal, 0.0, XHXtemp.Data(), width );

      MPI_Allreduce( XHXtemp.Data(), XHX.Data(), width*width, MPI_DOUBLE_COMPLEX, MPI_SUM, mpi_comm );

      // ResX <== Xf + 1i* dT/2 * ( HXf - Xf * XHXf ) - Xmid
      // Note RX is the ResX
      Complex traceXHX (0.0, 0.0);
      for( int i = 0; i < width; i++)
        traceXHX += *(XHX.Data() + i * width + i);

      {
        // remember:
        // X == X in G-parallel
        // XHX == XHXf 
        // Now Y == Xf * XHXf 
        CpxNumMat Y(ntotLocal, numStateTotal); 
        blas::Gemm( 'N', 'N', heightLocal, width, width, 1.0, 
            X.Data(), heightLocal, XHX.Data(), width, 0.0, Y.Data(), heightLocal );

        // Do things in the G-parallel fashion. 
        // HX is the HXf in G-parallel
        // Xmid is in the G-parallel Fashion
        // X is Xf in G-parallel fashion
        // RX is the ResX 

        Complex * ResPtr = RX.Data();
        Complex * XfPtr  = X.Data();
        Complex * HXfPtr = HX.Data();
        Complex * YPtr   = Y.Data();
        Complex * XmidPtr= Xmid.Data();
        for ( int i = 0; i < width; i++)
          for( int j = 0; j < heightLocal; j++){
            int index = i * heightLocal + j;
            ResPtr[index] = XfPtr[index] + i_Z_One * dT / 2.0 * ( HXfPtr[index] - YPtr[index] ) - XmidPtr[index];
          }
      }

      // Tranpose the ResX to band Parallel
      AlltoallBackward( RX, psiRes, mpi_comm);

      // Check check, still have the pre-conditioner 
      // not done yet.
      CpxNumVec preMat(ntot);
      Complex * precPtr = preMat.Data();
      for( int i = 0; i < ntot; i++){
        precPtr[i] = 1.0/(1.0 + i_Z_One * dT/2.0 * ( fft.gkk[i] - traceXHX / (Real)numStateTotal ));
      }

      // FIXME
      CpxNumMat dfMatTemp( ntot, maxDim ); 

      for( int iband = 0; iband < numStateLocal; iband++ ) {

        Complex *vinPtr = vin.Data();
        Complex *voutPtr= vout.Data();
        Complex *psiFPtr= psiF.Data() + iband * ntot;
        Complex *psiResPtr= psiRes.Data() + iband * ntot;

        for( int i = 0; i < ntot; i++){
          vinPtr[i]  =  psiFPtr  [i];
          voutPtr[i] =  psiResPtr[i];
        }

        if( iscf > 1) {
          Complex * dfMatPtr =  dfMat[iband].Data() + (ipos-1) * ntot;
          Complex * dvMatPtr =  dvMat[iband].Data() + (ipos-1) * ntot;

          for( int i = 0; i < ntot; i ++){
            dfMatPtr[i] = dfMatPtr[i] - psiResPtr[i];
            dvMatPtr[i] = dvMatPtr[i] - psiFPtr[i];
          }

          // Least Square problem here. 
          Real rcond = 1.0E-12;
          CpxNumVec gammas;
          DblNumVec S;
          S.Resize(iterused);
          gammas.Resize(ntot);

          blas::Copy( ntot, psiResPtr, 1, gammas.Data(), 1 );
          Int rank;
          Int nrow = iterused;

          // FIXME
          dfMatTemp = dfMat[iband];

          lapack::SVDLeastSquare( ntot, iterused, 1, 
              dfMatTemp.Data(), ntot, gammas.Data(), ntot,
              S.Data(), rcond, &rank );


          Print( statusOFS, "  Rank of dfmat = ", rank );
          Print( statusOFS, "  Rcond = ", rcond );

          blas::Gemv('N', ntot, nrow, -1.0, dvMat[iband].Data(),
              ntot, gammas.Data(), 1, 1.0, vin.Data(), 1 );

          blas::Gemv('N', ntot, iterused, -1.0, dfMat[iband].Data(),
              ntot, gammas.Data(), 1, 1.0, vout.Data(), 1 );

          // statusOFS << "Gammas = " << std::endl;
          // for(Int i = 0; i < iterused; i++ ){
          //   statusOFS << gammas[i] << std::endl;
          // }
        }


        int inext = iscf - std::floor((iscf - 1) / maxDim) *maxDim;

        Complex * dfMatPtr =  dfMat[iband].Data() + (inext-1) * ntot;
        Complex * dvMatPtr =  dvMat[iband].Data() + (inext-1) * ntot;

        for(int j = 0; j < ntot; j++){
          dfMatPtr[j] = psiResPtr[j];
          dvMatPtr[j] = psiFPtr[j];
        }

        // first FFT the vout to the G-space then do the Preconditioner. 
        {
          blas::Copy( ntot, voutPtr, 1, fft.inputComplexVec.Data(), 1 );
          fftw_execute( fft.forwardPlan );
          Complex * tempPtr = fft.outputComplexVec.Data();
          for(int i = 0; i < ntot; ++i)
            tempPtr[i] = tempPtr[i] * precPtr[i];
          fftw_execute( fft.backwardPlan );
          SetValue( vout, Complex(0,0) );
          blas::Axpy( ntot, 1.0 / Real(ntot), fft.inputComplexVec.Data(), 1, voutPtr, 1 );
        }

        for( int j = 0; j < ntot; j++) {
          psiFPtr[j] = vinPtr[j] + betaMix * voutPtr[j];
        }
      } // for (iband)

      {
        // Get the rhoFnew
        Real totalCharge_;
        ham.CalculateDensity(
            psiFinal,
            ham.OccupationRate(),
            totalCharge_, 
            fft );

        // Norm check 
        Real * densityPtr = ham.Density().Data();
        Real * rhoFinalPtr= rhoFinal.Data();
        Real normRhoF = 0.0;
        Real normRhoDiff = 0.0;

        for(int i = 0; i < ntotFine; i++) {
          normRhoDiff += pow( rhoFinalPtr[i] - densityPtr[i], 2.0 ); 
          normRhoF    += pow( rhoFinalPtr[i], 2.0 ); 
        }
        Real scfNorm = std::sqrt(normRhoDiff / normRhoF);
        //Print(statusOFS, "norm(RhoOut-RhoIn)/norm(RhoIn) = ", scfNorm );
        statusOFS << "SCF " << iscf << " norm(RhoOut-RhoIn)/norm(RhoIn): " << scfNorm << std::endl;


        // rhoF <== rhoFNew
        blas::Copy( ntotFine,  ham.Density().Data(), 1,  rhoFinal.Data(), 1 );

        if( scfNorm < options_.diisTol){
          statusOFS << "TDDFT step " << k_ << " SCF is converged in " << iscf << " steps !" << std::endl;
          //statusOFS << "TDDFT step " << k_ << " used " << totalHx << " H * x operations!" << std::endl;
          break;
        }
      }
#endif
    //} // iscf iteration

  } // if 1

  GetTime( timeEnd );
  timeDIIS += timeEnd - timeSta ;
  iterDIIS ++;
  //statusOFS << " TDDFT Step " << k_ << " DIIS loop used Time: " << timeEnd - timeSta1 << " [s]" << std::endl;

  GetTime( timeSta1 );
  AlltoallForward( psiF,  X, mpi_comm);

  // Reorthogonalize
  {
    blas::Gemm( 'C', 'N', width, width, heightLocal, 1.0, X.Data(), 
        heightLocal, X.Data(), heightLocal, 0.0, XHXtemp.Data(), width );
    MPI_Allreduce( XHXtemp.Data(), XHX.Data(), width*width, MPI_DOUBLE_COMPLEX, MPI_SUM, mpi_comm );

    // XHXtemp = 0.5 * ( XHX + conj ( XHX ) )
    {
      Complex * xPtr = XHXtemp.Data();
      Complex * yPtr = XHX.Data();
      for(int i = 0; i < width; i++){
        for(int j = 0; j < width; j++){
          xPtr[i*width + j] = 0.5 * ( yPtr[i*width+j] + std::conj(yPtr[j*width+i]) );
        }
      }
    }

    DblNumVec  eigValS(width);
    lapack::Syevd( 'V', 'U', width, XHXtemp.Data(), width, eigValS.Data() );

    CpxNumMat temp( width, width );
    SetValue( temp, Complex(0.0, 0.0) );
    for(int i = 0; i < width; i++) {
      temp(i,i) = Complex( 1.0 / sqrt( eigValS[i] ), 0.0);
    }

    blas::Gemm( 'N', 'N', width, width, width, 1.0, XHXtemp.Data(),
        width, temp.Data(), width, 0.0, XHX.Data(), width );

    blas::Gemm( 'N', 'C', width, width, width, 1.0, XHX.Data(),
        width, XHXtemp.Data(), width, 0.0, temp.Data(), width );

    blas::Gemm( 'N', 'N', heightLocal, width, width, 1.0, X.Data(),
        heightLocal, temp.Data(), width, 0.0, HX.Data(), heightLocal );

    AlltoallBackward ( HX, psiF, mpi_comm );
  }

  blas::Copy( ntot*numStateLocal, psiFinal.Wavefun().Data(), 1, psi.Wavefun().Data(), 1 );

  GetTime( timeEnd );
  timeOrth += timeEnd - timeSta1;
  iterOrth ++;

  // Update the density for the renormalized wavefunction
  GetTime( timeSta1 );
  {
    // get the charge density of the Hf.
    Real totalCharge_;
    ham.CalculateDensity(
        psiFinal,
        ham.OccupationRate(),
        totalCharge_, 
        fft );
  }
  GetTime( timeEnd );
  timeDensity += timeEnd - timeSta1;
  iterDensity ++;


  GetTime( timeSta1 );
  if(options_.ehrenfest){
    ham.CalculateForce( psi, fft);

    Real& dt = options_.dt;
    DblNumVec& atomMass = atomMass_;
    for( Int a = 0; a < numAtom; a++ ){
      atomList[a].vel = atomList[a].vel + (atomforce[a]/atomMass[a] + atomList[a].force/atomMass[a])*dt/2.0;
    } 
  }
  GetTime( timeEnd );
  timeForce += timeEnd - timeSta1;
  iterForce ++;

  statusOFS << " ***************************************************************************" << endl;
  if( esdfParam.isHybridACE) {
  statusOFS << "   PHI Setup DF time: " << timeDF           << " [s] " << " iterations " << iterDF      << endl;
  statusOFS << "   PHI Setup     Phi: " << timeSetPhi       << " [s] " << " iterations " << iterSetPhi  << endl;
  statusOFS << "   PHI Calculate ACE: " << timeCalACE       << " [s] " << " iterations " << iterCalACE  << endl;
  statusOFS << "   PHI CalEXX Energy: " << timeCalExxEnergy << " [s] " << " iterations " << iterCalExxEnergy << endl;
  statusOFS << "   DIIS SCF     Time: " << timeDIISSCF      << " [s] " << " iterations " << iterDIISSCF << endl;
  statusOFS << "   Adding Up   Above: " << timeDIISSCF + timeDF + timeSetPhi + timeCalACE + timeCalExxEnergy << " [s] " << endl;
  }
  statusOFS << " initialization    time: " << timeInit      << " [s] " << " iterations " << 1           << endl;
  statusOFS << " SCF calculating   time: " << timeDIIS      << " [s] " << " iterations " << iterDIIS    << endl;
  statusOFS << " othogonalization  time: " << timeOrth      << " [s] " << " iterations " << iterOrth    << endl;
  statusOFS << " calculate density time: " << timeDensity   << " [s] " << " iterations " << iterDensity << endl;
  statusOFS << " calculate Force   time: " << timeForce     << " [s] " << " iterations " << iterForce   << endl;
  statusOFS << " TDDFT Step " << k_ << " total Time: " << timeEnd - timeSta << " [s]" << std::endl;
 
 
  ++k_;
} // TDDFT:: advancePTTRAPDIIS

void TDDFT::advanceCN( PeriodTable& ptable ) {

  Int mpirank, mpisize;
  MPI_Comm mpi_comm = fftPtr_->domain.comm;
  MPI_Comm_rank( mpi_comm, &mpirank );
  MPI_Comm_size( mpi_comm, &mpisize );

  Real timeSta, timeEnd;
  Real timeSta1, timeEnd1;

  Real timeDF = 0.0;
  Real timeSetPhi = 0.0;
  Real timeCalACE = 0.0;
  Real timeCalExxEnergy = 0.0;
  Real timeDIISSCF = 0.0;
  Real timeInit = 0.0;
  Real timeDIIS = 0.0;
  Real timeOrth = 0.0;
  Real timeDensity = 0.0;
  Real timeForce = 0.0;
  Int  iterDF = 0;
  Int  iterSetPhi  = 0;
  Int  iterCalACE  = 0;
  Int  iterDIISSCF = 0;
  Int  iterDIIS    = 0;
  Int  iterOrth    = 0;
  Int  iterDensity = 0;
  Int  iterCalExxEnergy  = 0;
  Int  iterForce   = 0;

  GetTime( timeSta );

  Hamiltonian& ham = *hamPtr_;
  Fourier&     fft = *fftPtr_;
  Spinor&      psi = *psiPtr_;

  if( ham.IsHybrid() ) {
    ham.SetPhiEXX( psi, fft);
  }
#if ( _DEBUGlevel_ >= 0 )
    //statusOFS << "lijl psi = " << psi.Wavefun() << std::endl;
#endif
  std::vector<Atom>&   atomList = *atomListPtr_;
  Int numAtom = atomList.size();

  // print the options_ when first step. 
  if(k_ == 0) {
    statusOFS<< std::endl;
    statusOFS<< " -----   TDDFT Crank–Nicolson DIIS Print Options ---- "     << std::endl;
    statusOFS<< " options.auto_save      " << options_.auto_save      << std::endl;
    statusOFS<< " options.load_save      " << options_.load_save      << std::endl;
    statusOFS<< " options.method         " << options_.method         << std::endl;
    statusOFS<< " options.ehrenfest      " << options_.ehrenfest      << std::endl;
    statusOFS<< " options.simulateTime   " << options_.simulateTime   << std::endl;
    statusOFS<< " options.dt             " << options_.dt             << std::endl;
    statusOFS<< " options.gmres_restart  " << options_.gmres_restart  << std::endl;
    statusOFS<< " options.krylovTol      " << options_.krylovTol      << std::endl;
    statusOFS<< " options.diisTol        " << options_.diisTol        << std::endl;
    statusOFS<< " options.phiTol         " << options_.phiTol         << std::endl;
    statusOFS<< " options.adNum          " << options_.adNum          << std::endl;
    statusOFS<< " options.adUpdate       " << options_.adUpdate       << std::endl;
    statusOFS<< " -----   TDDFT Crank–Nicolson DIIS Print Options ---- "     << std::endl;
    statusOFS<< std::endl;
  }

  // Update saved atomList. 0 is the latest one
  for( Int l = maxHist_-1; l > 0; l-- ){
    atomListHist_[l] = atomListHist_[l-1];
  }
  atomListHist_[0] = atomList;

  Int ionIter = k_;

  std::vector<Point3>  atompos(numAtom);
  std::vector<Point3>  atomvel(numAtom);
  std::vector<Point3>  atomforce(numAtom);
  std::vector<Point3>  atompos_fin(numAtom);
  {
    // do not update force at the beginning

    Real& dt = options_.dt;
    DblNumVec& atomMass = atomMass_;

    for( Int a = 0; a < numAtom; a++ ){
      atompos[a]     = atomList[a].pos;
      atompos_fin[a] = atomList[a].pos;
      atomvel[a]     = atomList[a].vel;
      atomforce[a]   = atomList[a].force;
    }

    PrintState( k_ );


    // Update velocity and position when doing ehrenfest dynamics
    if(options_.ehrenfest){
      for(Int a=0; a<numAtom; a++) {
        atompos_fin[a]  = atompos[a] + atomvel[a] * dt + atomforce[a]*(dt*dt)/(atomMass[a]*2.0);
      }  
      AdjustAtomPos( atompos_fin );
    }
  }

  if(options_.ehrenfest){
    for(Int a = 0; a < numAtom; a++){
      MPI_Bcast( &atompos_fin[a][0], 3, MPI_DOUBLE, 0, mpi_comm); 
    }
  }

  // k_ is the current K
  Int k = k_;
  Real ti = tlist_[k];
  Real tf = tlist_[k+1];
  Real dT = tf - ti;
  Real tmid =  (ti + tf)/2.0;
  Complex i_Z_One = Complex(0.0, 1.0);

  // Crank–Nicolson DIIS Method, Occupation = 1
  DblNumVec &occupationRate = ham.OccupationRate();
  occupationRate.Resize( psi.NumStateTotal() );
  SetValue( occupationRate, 1.0);

  // update H when it is first step. 
  // This can be avoided since it is 
  // already converged in the first SCF.
  if(k == esdfParam.restartTDDFTStep) {
    //if(!esdfParam.isRestartDensity){
    if(1){
	    statusOFS << " always start by calculating Density from WFN " << std::endl;
      Real totalCharge_;
      ham.CalculateDensity(
          psi,
          ham.OccupationRate(),
          totalCharge_, 
          fft );
    }
    if( isCalculateGradRho_ ){
      ham.CalculateGradDensity( fft );
    }
    ham.CalculateXC( Exc_, fft ); 
    ham.CalculateHartree( fft );
    CalculateEfieldExt(ptable, ti); 
    ham.CalculateVtot( ham.Vtot() );
  }

  // calculate Dipole at the beginning.
  if(calDipole_)  CalculateDipole(tlist_[k_]);

  // 1. Calculate Xmid which appears on the right hand of the equation
  // HPSI = (H1 * psi)
  Int ntot  = fft.domain.NumGridTotal();
  Int numStateLocal = psi.NumState();
  Int ntotLocal = ntot/mpisize;
  if(mpirank < (ntot % mpisize)) ntotLocal++;
  Int numStateTotal = psi.NumStateTotal();
  CpxNumMat HPSI(ntot, numStateLocal);
  NumTns<Complex> tnsTemp(ntot, 1, numStateLocal, false, HPSI.Data());
  ham.MultSpinor( psi, tnsTemp, fft );

  //  All X's are in G-parallel
  CpxNumMat X(ntotLocal, numStateTotal); 
  CpxNumMat HX(ntotLocal, numStateTotal); 
  CpxNumMat RX(ntotLocal, numStateTotal);
  // lijl Xmid right hand (I - 0.5i*dt*Hn)Xn 
  CpxNumMat Xmid(ntotLocal, numStateTotal); 

  // All psi's are in Band-parallel
  CpxNumMat psiF  ( ntot, numStateLocal );
  CpxNumMat psiCol( ntot, numStateLocal );
  CpxNumMat psiRes( ntot, numStateLocal );

  lapack::Lacpy( 'A', ntot, numStateLocal, psi.Wavefun().Data(), ntot, psiCol.Data(), ntot );

  AlltoallForward( HPSI,  HX, mpi_comm);
  AlltoallForward( psiCol, X, mpi_comm);

  // RX <-- HX - X*(X'*HX)
  Int width = numStateTotal;
  Int heightLocal = ntotLocal;
  CpxNumMat  XHXtemp( width, width );
  CpxNumMat  XHX( width, width );
  lapack::Lacpy( 'A', ntotLocal, numStateTotal, HX.Data(), ntotLocal, RX.Data(), ntotLocal );
  blas::Gemm( 'C', 'N', width, width, heightLocal, 1.0, X.Data(), 
      heightLocal, HX.Data(), heightLocal, 0.0, XHXtemp.Data(), width );
  MPI_Allreduce( XHXtemp.Data(), XHX.Data(), width*width, MPI_DOUBLE_COMPLEX, MPI_SUM, mpi_comm );
  //blas::Gemm( 'N', 'N', heightLocal, width, width, -1.0, 
  //    X.Data(), heightLocal, XHX.Data(), width, 1.0, RX.Data(), heightLocal );


  // check check
  // E_kin = numSpin * trace( XHX )
  Ekin_ = 0.0;
  {
    Complex * ptr = XHX.Data();
    Int numSpin = ham.NumSpin();
    for(int i =0; i < width; i++)
      Ekin_ += numSpin * ptr[i*width+i].real();
  }

  CalculateEnergy( ptable, ti );

  // Xmid <-- X - li*T/2 * HX  in G-parallel
  {
    Complex * xmidPtr = Xmid.Data();
    Complex * xPtr    = X.Data();
    Complex * rxPtr   = HX.Data();
    for( Int i = 0; i < numStateTotal; i ++)
      for( Int j = 0; j < ntotLocal; j ++){
        Int index = i* ntotLocal +j;
        xmidPtr[index] = xPtr[index] -  i_Z_One * dT/2.0 * rxPtr[index];
      }
  }

  Spinor psiFinal (fft.domain, 1, numStateTotal, numStateLocal, false, psiF.Data() );
  if(0){
    // psiF <== psi
    lapack::Lacpy( 'A', ntot, numStateLocal, psi.Wavefun().Data(), ntot, psiF.Data(), ntot );
  }
  if(1){
    // psiF <== psi - i * dT * HX
    AlltoallBackward( HX, psiRes, mpi_comm);

    lapack::Lacpy( 'A', ntot, numStateLocal, psi.Wavefun().Data(), ntot, psiF.Data(), ntot );
    blas::Axpy( ntot, - i_Z_One * dT, psiRes.Data(), 1, psiF.Data(), 1 );
  }

  // AtomPos <== AtomPosFinal, then update Vatom
  if(options_.ehrenfest){
    for( Int a = 0; a < numAtom; a++ ){
      atomList[a].pos  = atompos_fin[a];
    }
    ham.UpdateHamiltonian( atomList );
    ham.CalculatePseudoPotential( ptable );
  }

  // rhoF <== update Charge Density.
  DblNumMat          rhoFinal; 
  {
    CalculateEfieldExt(ptable, tf); // tf is the current step, calculate only once.

    // get the charge density of the Hf.
    Real totalCharge_;
    ham.CalculateDensity(
        psiFinal,
        ham.OccupationRate(),
        totalCharge_, 
        fft );
    Int ntotFine  = fftPtr_->domain.NumGridTotalFine();
    rhoFinal.Resize (ntotFine, 1);  

    Real * densityPtr = ham.Density().Data();
    Real * rhoFinalPtr= rhoFinal.Data();
    for(int i = 0; i < ntotFine; i++) {
      rhoFinalPtr[i] = densityPtr[i];
    }
  }

  GetTime( timeEnd );
  timeInit += timeEnd - timeSta;
  //statusOFS << " TDDFT Step " << k_ << " Setting-up Time: " << timeEnd - timeSta << " [s]" << std::endl;


  Int numGridTotal = ntot;

  if(1){

    GetTime( timeSta1 );
    Int maxScfIteration = options_.diisMaxIter;
    Real betaMix = esdfParam.mixStepLength;
    Int  maxDim  = esdfParam.mixMaxDim;

    std::vector<CpxNumMat>   dfMat;
    std::vector<CpxNumMat>   dvMat;
    dfMat.resize( numStateLocal );
    dvMat.resize( numStateLocal );
    for( int i = 0; i < numStateLocal; i++) { 
      dfMat[i].Resize( ntot, maxDim ); 
      dvMat[i].Resize( ntot, maxDim ); 
      SetValue( dfMat[i], Complex(0.0, 0.0) );
      SetValue( dvMat[i], Complex(0.0, 0.0) );
    }

    CpxNumVec vin;
    CpxNumVec vout;
    vin.Resize( ntot);
    vout.Resize( ntot);
    SetValue( vin,  Complex(0,0));
    SetValue( vout, Complex(0,0));

    GetTime( timeEnd1 );
    timeDF += timeEnd1 - timeSta1;
    iterDF ++;

    Real scfNorm = 0.0;
    if( esdfParam.isHybridACE ) {

      statusOFS << "TDDFT is using Hybrid ACE Operator ...." << std::endl;

      Real fock1 = 0.0;
      Real fock2 = 0.0;

      // Two SCF loops, outer and inner SCF.
      // Outer SCF
      int maxPhiIteration = options_.phiMaxIter;

      // new scheme: get E[V] <== V[psi_0] <== psi_0
      GetTime( timeSta1 );
      ham.SetPhiEXX( psiFinal, fft);
      GetTime( timeEnd1 );
      timeSetPhi += timeEnd1 - timeSta1;
      iterSetPhi ++;

      GetTime( timeSta1 );
      bool isFixColumnDF = false;
      //if( esdfParam.isHybridDF ){
      if(0){
        statusOFS << " ACEDF: first iteration " << std::endl;
        ham.CalculateVexxACEDF( psi, fft, isFixColumnDF );
        isFixColumnDF = true;
      }
      else
      {
        ham.CalculateVexxACE ( psi, fft );
      }
      GetTime( timeEnd1 );
      timeCalACE += timeEnd1 - timeSta1;
      iterCalACE ++;
#if ( _DEBUGlevel_ >= 0 )
    //statusOFS << "psi = " << psi.Wavefun() << std::endl;
    //statusOFS << "psiFinal = " << psiFinal.Wavefun() << std::endl;
#endif
      GetTime( timeSta1 );
      fock1 = ham.CalculateEXXEnergy( psiFinal, fft ); 
      GetTime( timeEnd1 );
      statusOFS << " fock1 ACEDF lijl :" << fock1 << std::endl;
      timeCalExxEnergy += timeEnd1 - timeSta1;
      iterCalExxEnergy ++;

      for( int phiIter = 0; phiIter < maxPhiIteration; phiIter++){

        // Inner SCF.
        GetTime( timeSta1 );
	int iscf;
        for( iscf = 1; iscf <= maxScfIteration; iscf++ ) {
          scfNorm = InnerSolve_CN( iscf, psiFinal, tnsTemp, HX, X, HPSI, psiF, XHX, XHXtemp, RX, Xmid, dT, psiRes, vin, vout, dfMat, dvMat, rhoFinal);
          if( scfNorm < options_.diisTol){
            break;
          }
        }
        GetTime( timeEnd1 );
        timeDIISSCF += timeEnd1 - timeSta1;
        iterDIISSCF += iscf;

        if( scfNorm < options_.diisTol)
          statusOFS << "phiStep " << phiIter << " DIIS is  converged in " << iscf << " steps " << " scfNorm " << scfNorm << std::endl;
	else 
          statusOFS << "phiStep " << phiIter << " DIIS NOT converged in " << iscf << " steps " << " scfNorm " << scfNorm << std::endl;

        // new scheme: get E[V] <== V[psi_0] <== psi_0
        GetTime( timeSta1 );
        ham.SetPhiEXX( psiFinal, fft);
        GetTime( timeEnd1 );
        timeSetPhi += timeEnd1 - timeSta1;
        iterSetPhi ++;

        GetTime( timeSta1 );
        if( esdfParam.isHybridDF ){
            if( phiIter == 0){
                isFixColumnDF = false;
                ham.CalculateVexxACEDF( psiFinal, fft, isFixColumnDF );
                isFixColumnDF = true;
            }
            else{
                ham.CalculateVexxACEDF( psiFinal, fft, isFixColumnDF );
                isFixColumnDF = true;
            }
        }
        else{
            ham.CalculateVexxACE ( psiFinal, fft );
        }

        GetTime( timeEnd1 );
        timeCalACE += timeEnd1 - timeSta1;
        iterCalACE ++;

        GetTime( timeSta1 );
        fock2 = ham.CalculateEXXEnergy( psiFinal, fft ); 
        GetTime( timeEnd1 );
        timeCalExxEnergy += timeEnd1 - timeSta1;
        iterCalExxEnergy ++;

        Real dExx = std::abs(fock2 - fock1) / std::abs(fock2);

        statusOFS << " Fock Energy  = " << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< fock2 << " [au]" << std::endl 
                  << " dExx         = " << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< dExx  << " [au]" << std::endl;

	fock1 = fock2;
        if( dExx < options_.phiTol) {
          statusOFS << "TDDFT step " << k_ << " Phi Iteration in " << phiIter + 1<< " steps !" << std::endl;
	  break; 
	}
      }
    } 
    else {

      if( ham.IsHybrid() && !esdfParam.isHybridACE ) 
        statusOFS << "TDDFT screen exchange ... " << std::endl;
      else
        statusOFS << "TDDFT PBE ... " << std::endl;

      // Note, the exact HF and PBE implementation together. 
      for(int iscf = 1; iscf <= maxScfIteration; iscf++){
        scfNorm = InnerSolve_CN( iscf, psiFinal, tnsTemp, HX, X, HPSI, psiF, XHX, XHXtemp, RX, Xmid, dT, psiRes, vin, vout, dfMat, dvMat, rhoFinal);
        if( scfNorm < options_.diisTol){
          statusOFS << "TDDFT step " << k_ << " SCF is converged in " << iscf << " steps !" << std::endl;
          break;
        }
      }
    }
#if 0
      Int iterused = std::min (iscf-1, maxDim);
      Int ipos = iscf - 1 - floor( (iscf-2) / maxDim ) * maxDim;

      // Update Hf <== updateV(molf, rhof)
      Int ntotFine  = fft.domain.NumGridTotalFine();
      {
        if( isCalculateGradRho_ ){
          ham.CalculateGradDensity( fft );
        }
        ham.CalculateXC( Exc_, fft ); 
        ham.CalculateHartree( fft );
        ham.CalculateVtot( ham.Vtot());
      }

      if( ham.IsHybrid() ) {
        ham.SetPhiEXX( psiFinal, fft);
      }
      // HXf <== Hf * Xf, now HPSI is HXf  
      ham.MultSpinor( psiFinal, tnsTemp, fft );

      //  XHX <== XHXtemp <--- X'HXf
      //  PsiF, HPSI are psiF and H*psiF
      AlltoallForward( HPSI, HX, mpi_comm);
      AlltoallForward( psiF, X,  mpi_comm);

      blas::Gemm( 'C', 'N', width, width, heightLocal, 1.0, X.Data(), 
          heightLocal, HX.Data(), heightLocal, 0.0, XHXtemp.Data(), width );

      MPI_Allreduce( XHXtemp.Data(), XHX.Data(), width*width, MPI_DOUBLE_COMPLEX, MPI_SUM, mpi_comm );

      // ResX <== Xf + 1i* dT/2 * ( HXf - Xf * XHXf ) - Xmid
      // Note RX is the ResX
      Complex traceXHX (0.0, 0.0);
      for( int i = 0; i < width; i++)
        traceXHX += *(XHX.Data() + i * width + i);

      {
        // remember:
        // X == X in G-parallel
        // XHX == XHXf 
        // Now Y == Xf * XHXf 
        CpxNumMat Y(ntotLocal, numStateTotal); 
        blas::Gemm( 'N', 'N', heightLocal, width, width, 1.0, 
            X.Data(), heightLocal, XHX.Data(), width, 0.0, Y.Data(), heightLocal );

        // Do things in the G-parallel fashion. 
        // HX is the HXf in G-parallel
        // Xmid is in the G-parallel Fashion
        // X is Xf in G-parallel fashion
        // RX is the ResX 

        Complex * ResPtr = RX.Data();
        Complex * XfPtr  = X.Data();
        Complex * HXfPtr = HX.Data();
        Complex * YPtr   = Y.Data();
        Complex * XmidPtr= Xmid.Data();
        for ( int i = 0; i < width; i++)
          for( int j = 0; j < heightLocal; j++){
            int index = i * heightLocal + j;
            ResPtr[index] = XfPtr[index] + i_Z_One * dT / 2.0 * ( HXfPtr[index] - YPtr[index] ) - XmidPtr[index];
          }
      }

      // Tranpose the ResX to band Parallel
      AlltoallBackward( RX, psiRes, mpi_comm);

      // Check check, still have the pre-conditioner 
      // not done yet.
      CpxNumVec preMat(ntot);
      Complex * precPtr = preMat.Data();
      for( int i = 0; i < ntot; i++){
        precPtr[i] = 1.0/(1.0 + i_Z_One * dT/2.0 * ( fft.gkk[i] - traceXHX / (Real)numStateTotal ));
      }

      // FIXME
      CpxNumMat dfMatTemp( ntot, maxDim ); 

      for( int iband = 0; iband < numStateLocal; iband++ ) {

        Complex *vinPtr = vin.Data();
        Complex *voutPtr= vout.Data();
        Complex *psiFPtr= psiF.Data() + iband * ntot;
        Complex *psiResPtr= psiRes.Data() + iband * ntot;

        for( int i = 0; i < ntot; i++){
          vinPtr[i]  =  psiFPtr  [i];
          voutPtr[i] =  psiResPtr[i];
        }

        if( iscf > 1) {
          Complex * dfMatPtr =  dfMat[iband].Data() + (ipos-1) * ntot;
          Complex * dvMatPtr =  dvMat[iband].Data() + (ipos-1) * ntot;

          for( int i = 0; i < ntot; i ++){
            dfMatPtr[i] = dfMatPtr[i] - psiResPtr[i];
            dvMatPtr[i] = dvMatPtr[i] - psiFPtr[i];
          }

          // Least Square problem here. 
          Real rcond = 1.0E-12;
          CpxNumVec gammas;
          DblNumVec S;
          S.Resize(iterused);
          gammas.Resize(ntot);

          blas::Copy( ntot, psiResPtr, 1, gammas.Data(), 1 );
          Int rank;
          Int nrow = iterused;

          // FIXME
          dfMatTemp = dfMat[iband];

          lapack::SVDLeastSquare( ntot, iterused, 1, 
              dfMatTemp.Data(), ntot, gammas.Data(), ntot,
              S.Data(), rcond, &rank );


          Print( statusOFS, "  Rank of dfmat = ", rank );
          Print( statusOFS, "  Rcond = ", rcond );

          blas::Gemv('N', ntot, nrow, -1.0, dvMat[iband].Data(),
              ntot, gammas.Data(), 1, 1.0, vin.Data(), 1 );

          blas::Gemv('N', ntot, iterused, -1.0, dfMat[iband].Data(),
              ntot, gammas.Data(), 1, 1.0, vout.Data(), 1 );

          // statusOFS << "Gammas = " << std::endl;
          // for(Int i = 0; i < iterused; i++ ){
          //   statusOFS << gammas[i] << std::endl;
          // }
        }


        int inext = iscf - std::floor((iscf - 1) / maxDim) *maxDim;

        Complex * dfMatPtr =  dfMat[iband].Data() + (inext-1) * ntot;
        Complex * dvMatPtr =  dvMat[iband].Data() + (inext-1) * ntot;

        for(int j = 0; j < ntot; j++){
          dfMatPtr[j] = psiResPtr[j];
          dvMatPtr[j] = psiFPtr[j];
        }

        // first FFT the vout to the G-space then do the Preconditioner. 
        {
          blas::Copy( ntot, voutPtr, 1, fft.inputComplexVec.Data(), 1 );
          fftw_execute( fft.forwardPlan );
          Complex * tempPtr = fft.outputComplexVec.Data();
          for(int i = 0; i < ntot; ++i)
            tempPtr[i] = tempPtr[i] * precPtr[i];
          fftw_execute( fft.backwardPlan );
          SetValue( vout, Complex(0,0) );
          blas::Axpy( ntot, 1.0 / Real(ntot), fft.inputComplexVec.Data(), 1, voutPtr, 1 );
        }

        for( int j = 0; j < ntot; j++) {
          psiFPtr[j] = vinPtr[j] + betaMix * voutPtr[j];
        }
      } // for (iband)

      {
        // Get the rhoFnew
        Real totalCharge_;
        ham.CalculateDensity(
            psiFinal,
            ham.OccupationRate(),
            totalCharge_, 
            fft );

        // Norm check 
        Real * densityPtr = ham.Density().Data();
        Real * rhoFinalPtr= rhoFinal.Data();
        Real normRhoF = 0.0;
        Real normRhoDiff = 0.0;

        for(int i = 0; i < ntotFine; i++) {
          normRhoDiff += pow( rhoFinalPtr[i] - densityPtr[i], 2.0 ); 
          normRhoF    += pow( rhoFinalPtr[i], 2.0 ); 
        }
        Real scfNorm = std::sqrt(normRhoDiff / normRhoF);
        //Print(statusOFS, "norm(RhoOut-RhoIn)/norm(RhoIn) = ", scfNorm );
        statusOFS << "SCF " << iscf << " norm(RhoOut-RhoIn)/norm(RhoIn): " << scfNorm << std::endl;


        // rhoF <== rhoFNew
        blas::Copy( ntotFine,  ham.Density().Data(), 1,  rhoFinal.Data(), 1 );

        if( scfNorm < options_.diisTol){
          statusOFS << "TDDFT step " << k_ << " SCF is converged in " << iscf << " steps !" << std::endl;
          //statusOFS << "TDDFT step " << k_ << " used " << totalHx << " H * x operations!" << std::endl;
          break;
        }
      }
#endif
    //} // iscf iteration

  } // if 1

  GetTime( timeEnd );
  timeDIIS += timeEnd - timeSta ;
  iterDIIS ++;
  //statusOFS << " TDDFT Step " << k_ << " DIIS loop used Time: " << timeEnd - timeSta1 << " [s]" << std::endl;

  GetTime( timeSta1 );
  AlltoallForward( psiF,  X, mpi_comm);

  // Reorthogonalize
  {
    blas::Gemm( 'C', 'N', width, width, heightLocal, 1.0, X.Data(), 
        heightLocal, X.Data(), heightLocal, 0.0, XHXtemp.Data(), width );
    MPI_Allreduce( XHXtemp.Data(), XHX.Data(), width*width, MPI_DOUBLE_COMPLEX, MPI_SUM, mpi_comm );

    // XHXtemp = 0.5 * ( XHX + conj ( XHX ) )
    {
      Complex * xPtr = XHXtemp.Data();
      Complex * yPtr = XHX.Data();
      for(int i = 0; i < width; i++){
        for(int j = 0; j < width; j++){
          xPtr[i*width + j] = 0.5 * ( yPtr[i*width+j] + std::conj(yPtr[j*width+i]) );
        }
      }
    }

    DblNumVec  eigValS(width);
    lapack::Syevd( 'V', 'U', width, XHXtemp.Data(), width, eigValS.Data() );

    CpxNumMat temp( width, width );
    SetValue( temp, Complex(0.0, 0.0) );
    for(int i = 0; i < width; i++) {
      temp(i,i) = Complex( 1.0 / sqrt( eigValS[i] ), 0.0);
    }

    blas::Gemm( 'N', 'N', width, width, width, 1.0, XHXtemp.Data(),
        width, temp.Data(), width, 0.0, XHX.Data(), width );

    blas::Gemm( 'N', 'C', width, width, width, 1.0, XHX.Data(),
        width, XHXtemp.Data(), width, 0.0, temp.Data(), width );

    blas::Gemm( 'N', 'N', heightLocal, width, width, 1.0, X.Data(),
        heightLocal, temp.Data(), width, 0.0, HX.Data(), heightLocal );

    AlltoallBackward ( HX, psiF, mpi_comm );
  }

  blas::Copy( ntot*numStateLocal, psiFinal.Wavefun().Data(), 1, psi.Wavefun().Data(), 1 );

  GetTime( timeEnd );
  timeOrth += timeEnd - timeSta1;
  iterOrth ++;

  // Update the density for the renormalized wavefunction
  GetTime( timeSta1 );
  {
    // get the charge density of the Hf.
    Real totalCharge_;
    ham.CalculateDensity(
        psiFinal,
        ham.OccupationRate(),
        totalCharge_, 
        fft );
  }
  GetTime( timeEnd );
  timeDensity += timeEnd - timeSta1;
  iterDensity ++;


  GetTime( timeSta1 );
  if(options_.ehrenfest){
    ham.CalculateForce( psi, fft);

    Real& dt = options_.dt;
    DblNumVec& atomMass = atomMass_;
    for( Int a = 0; a < numAtom; a++ ){
      atomList[a].vel = atomList[a].vel + (atomforce[a]/atomMass[a] + atomList[a].force/atomMass[a])*dt/2.0;
    } 
  }
  GetTime( timeEnd );
  timeForce += timeEnd - timeSta1;
  iterForce ++;

  statusOFS << " ***************************************************************************" << endl;
  if( esdfParam.isHybridACE) {
  statusOFS << "   PHI Setup DF time: " << timeDF           << " [s] " << " iterations " << iterDF      << endl;
  statusOFS << "   PHI Setup     Phi: " << timeSetPhi       << " [s] " << " iterations " << iterSetPhi  << endl;
  statusOFS << "   PHI Calculate ACE: " << timeCalACE       << " [s] " << " iterations " << iterCalACE  << endl;
  statusOFS << "   PHI CalEXX Energy: " << timeCalExxEnergy << " [s] " << " iterations " << iterCalExxEnergy << endl;
  statusOFS << "   DIIS SCF     Time: " << timeDIISSCF      << " [s] " << " iterations " << iterDIISSCF << endl;
  statusOFS << "   Adding Up   Above: " << timeDIISSCF + timeDF + timeSetPhi + timeCalACE + timeCalExxEnergy << " [s] " << endl;
  }
  statusOFS << " initialization    time: " << timeInit      << " [s] " << " iterations " << 1           << endl;
  statusOFS << " SCF calculating   time: " << timeDIIS      << " [s] " << " iterations " << iterDIIS    << endl;
  statusOFS << " othogonalization  time: " << timeOrth      << " [s] " << " iterations " << iterOrth    << endl;
  statusOFS << " calculate density time: " << timeDensity   << " [s] " << " iterations " << iterDensity << endl;
  statusOFS << " calculate Force   time: " << timeForce     << " [s] " << " iterations " << iterForce   << endl;
  statusOFS << " TDDFT Step " << k_ << " total Time: " << timeEnd - timeSta << " [s]" << std::endl;
 
 
  ++k_;
} // TDDFT:: advanceCN

void TDDFT::Propagate( PeriodTable& ptable ) {
  Int totalSteps = tlist_.size() - 1;
  int startTime = 0;

  if(hamPtr_->IsHybrid()) {
    statusOFS << " TDDFT with HSE functions. " << std::endl;
    //FIXME, change this when using ACE
    hamPtr_->SetEXXActive(true) ; 
    if(options_.method == "PTTRAP"){
      ErrorHandling( "TDDFT HSE functions only works for PTTRAPDIIS and RK4");
    }
  }

  if(esdfParam.restartTDDFTStep) {
    startTime = esdfParam.restartTDDFTStep;
  }
  k_ = startTime;
  if(options_.method == "RK4"){
    for( Int i = startTime; i < totalSteps; i++){
      advanceRK4( ptable );
      if( (i != 0) && (i % esdfParam.TDDFTautoSaveSteps == 0)) 
        Store4Restart();
    }
  }
  else if( options_.method == "PTTRAP"){
    for( Int i = startTime; i < totalSteps; i++) {
      advancePTTRAP( ptable );
      if( (i != 0) && (i % esdfParam.TDDFTautoSaveSteps == 0)) 
        Store4Restart();
    }
  }
  else if( options_.method == "CN"){
    for( Int i = startTime; i < totalSteps; i++) {
      advanceCN( ptable );
      if( (i != 0) && (i % esdfParam.TDDFTautoSaveSteps == 0)) 
        Store4Restart();
    }
  }
  else if( options_.method == "PTTRAPDIIS"){
    for( Int i = startTime; i < totalSteps; i++) {
      advancePTTRAPDIIS( ptable );
      if( (i != 0) && (i % esdfParam.TDDFTautoSaveSteps == 0)) 
        Store4Restart();
    }
  }

  // at the end of the propagation, write the WFN, DENSITY and Velocity, Atom Pos. 
//  if( esdfParam.save4RestartTDDFT ) {
#if 0
  if( 1 ) {

    statusOFS << std::endl 
      << " ********************** Warning ************************************"<< std::endl;
    statusOFS << " TDDFT now optionally saves the WFN, DEN, Pos, Vel for restart " << std::endl;
    statusOFS << " ********************** Warning ************************************" 
      << std::endl << std::endl;

    MPI_Comm mpi_comm = fftPtr_->domain.comm;
    Int mpirank, mpisize;
    MPI_Comm_rank( mpi_comm, &mpirank );
    MPI_Comm_size( mpi_comm, &mpisize );

    // WFN
    if( esdfParam.isOutputWfn )
    {
      std::ostringstream wfnStream;
      serialize( psiPtr_->Wavefun(), wfnStream, NO_MASK );
      serialize( hamPtr_->OccupationRate(), wfnStream, NO_MASK );
      string restartWfnFileName_     = "WFN";
      SeparateWrite( restartWfnFileName_, wfnStream, mpirank );
    }


    if( mpirank == 0 ){
      std::vector<Atom>&   atomList = *atomListPtr_;
      Int numAtom = atomList.size();
    
      // output density
      if( esdfParam.isOutputDensity ) {
        string restartDensityFileName_ = "DEN";
        std::ofstream rhoStream(restartDensityFileName_.c_str());
        if( !rhoStream.good() ){
          ErrorHandling( "Density file cannot be opened." );
        }

        const Domain& dm =  fftPtr_->domain;
        std::vector<DblNumVec>   gridpos(DIM);
        UniformMeshFine ( dm, gridpos );
        for( Int d = 0; d < DIM; d++ ){
          serialize( gridpos[d], rhoStream, NO_MASK );
        }

        // Only work for the restricted spin case
        DblNumMat& densityMat = hamPtr_->Density();
        DblNumVec densityVec(densityMat.m(), false, densityMat.Data());
        serialize( densityVec, rhoStream, NO_MASK );
        rhoStream.close();
      }

      if(esdfParam.isOutputPosition & options_.ehrenfest){
        std::fstream fout;
        fout.open("lastPos.out",std::ios::out);
        if( !fout.good() ){
          ErrorHandling( "File cannot be opened !" );
        }

        for(Int a=0; a<numAtom; a++){
          fout << std::setiosflags(std::ios::scientific)
            << std::setiosflags(std::ios::showpos)
            << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< atomList[a].pos[0]
            << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< atomList[a].pos[1]
            << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< atomList[a].pos[2]
            << std::resetiosflags(std::ios::scientific)
            << std::resetiosflags(std::ios::showpos)
            << std::endl;
        }
        fout.close();
      } // OutputPosition


      if(esdfParam.isOutputVelocity & options_.ehrenfest){
        std::fstream fout_v;
        fout_v.open("lastVel.out",std::ios::out);
        if( !fout_v.good() ){
          ErrorHandling( "File cannot be opened !" );
        }
        for(Int a=0; a<numAtom; a++){
          fout_v << std::setiosflags(std::ios::scientific)
            << std::setiosflags(std::ios::showpos)
            << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< atomList[a].vel[0]
            << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< atomList[a].vel[1]
            << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< atomList[a].vel[2]
            << std::resetiosflags(std::ios::scientific)
            << std::resetiosflags(std::ios::showpos)
            << std::endl;
        }
        fout_v.close();
      } // OutputVelocity
    } // mpirank == 0


  }
#endif 
}

void TDDFT::Store4Restart()
{
    statusOFS << std::endl 
      << " ********************** Warning ************************************"<< std::endl;
    statusOFS << " TDDFT now optionally saves the WFN, DEN, Pos, Vel for restart " << std::endl;
    statusOFS << " ********************** Warning ************************************" 
      << std::endl << std::endl;

    MPI_Comm mpi_comm = fftPtr_->domain.comm;
    Int mpirank, mpisize;
    MPI_Comm_rank( mpi_comm, &mpirank );
    MPI_Comm_size( mpi_comm, &mpisize );

    // WFN
    if( esdfParam.isOutputWfn )
    {
      std::ostringstream wfnStream;
      serialize( psiPtr_->Wavefun(), wfnStream, NO_MASK );
      serialize( hamPtr_->OccupationRate(), wfnStream, NO_MASK );
      string restartWfnFileName_     = "WFN";
      SeparateWrite( restartWfnFileName_, wfnStream, mpirank );
    }


    if( mpirank == 0 ){
      std::vector<Atom>&   atomList = *atomListPtr_;
      Int numAtom = atomList.size();
    
      // output density
      if( esdfParam.isOutputDensity ) {
        string restartDensityFileName_ = "DEN";
        std::ofstream rhoStream(restartDensityFileName_.c_str());
        if( !rhoStream.good() ){
          ErrorHandling( "Density file cannot be opened." );
        }

        const Domain& dm =  fftPtr_->domain;
        std::vector<DblNumVec>   gridpos(DIM);
        UniformMeshFine ( dm, gridpos );
        for( Int d = 0; d < DIM; d++ ){
          serialize( gridpos[d], rhoStream, NO_MASK );
        }

        // Only work for the restricted spin case
        DblNumMat& densityMat = hamPtr_->Density();
        DblNumVec densityVec(densityMat.m(), false, densityMat.Data());
        serialize( densityVec, rhoStream, NO_MASK );
        rhoStream.close();
      }

      if(esdfParam.isOutputPosition & options_.ehrenfest){
        std::fstream fout;
        fout.open("lastPos.out",std::ios::out);
        if( !fout.good() ){
          ErrorHandling( "File cannot be opened !" );
        }

        for(Int a=0; a<numAtom; a++){
          fout << std::setiosflags(std::ios::scientific)
            << std::setiosflags(std::ios::showpos)
            << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< atomList[a].pos[0]
            << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< atomList[a].pos[1]
            << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< atomList[a].pos[2]
            << std::resetiosflags(std::ios::scientific)
            << std::resetiosflags(std::ios::showpos)
            << std::endl;
        }
        fout.close();
      } // OutputPosition


      if(esdfParam.isOutputVelocity & options_.ehrenfest){
        std::fstream fout_v;
        fout_v.open("lastVel.out",std::ios::out);
        if( !fout_v.good() ){
          ErrorHandling( "File cannot be opened !" );
        }
        for(Int a=0; a<numAtom; a++){
          fout_v << std::setiosflags(std::ios::scientific)
            << std::setiosflags(std::ios::showpos)
            << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< atomList[a].vel[0]
            << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< atomList[a].vel[1]
            << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< atomList[a].vel[2]
            << std::resetiosflags(std::ios::scientific)
            << std::resetiosflags(std::ios::showpos)
            << std::endl;
        }
        fout_v.close();
      } // OutputVelocity
    } // mpirank == 0
}
void TDDFT::PrintState ( Int step ) {
  Int mpirank, mpisize;
  MPI_Comm mpi_comm = fftPtr_->domain.comm;
  MPI_Comm_rank( mpi_comm, &mpirank );
  MPI_Comm_size( mpi_comm, &mpisize );

  Real& dt = options_.dt;
  Real t = step*dt;
  std::vector<Atom>&  atomList = hamPtr_->AtomList();
  Int numAtom = atomList.size();

  statusOFS<< "****************************************************************************" << std::endl << std::endl;
  statusOFS<< "Step " << step << ", time (fs) = " << t*au2fs << std::endl << std::endl;

  if(options_.ehrenfest){ for( Int a = 0; a < atomList.size(); a++ ){
      Print( statusOFS, "atom", a, "pos", atomList[a].pos );
    }
    statusOFS << std::endl;
    for( Int a = 0; a < atomList.size(); a++ ){
      Print( statusOFS, "atom", a, "vel", atomList[a].vel );
    }
    statusOFS << std::endl;
    Point3 forceCM(0.0, 0.0, 0.0);
    for( Int a = 0; a < atomList.size(); a++ ){
      Print( statusOFS, "atom", a, "force", atomList[a].force );
      forceCM += atomList[a].force;
    }
    statusOFS << std::endl;
    Print( statusOFS, "force for centroid  : ", forceCM );
    Print( statusOFS, "Max force magnitude : ", MaxForce(atomList) );
    statusOFS << std::endl;
  }

  if( options_.isOutputXYZ && options_.ehrenfest ){
    if( mpirank == 0 ){
      std::fstream fout;
      fout.open("TD.xyz",std::ios::out | std::ios::app) ;
      if( !fout.good() ){
        ErrorHandling( "Cannot open TD.xyz!" );
      }
      fout << numAtom << std::endl;
      fout << "TD step # "<< k_ << std::endl;
      for(Int a=0; a<numAtom; a++){
        fout << std::setw(6)<< atomList[a].type
          << std::setiosflags(std::ios::scientific)
          << std::setiosflags(std::ios::showpos)
          << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< atomList[a].pos[0]*au2ang
          << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< atomList[a].pos[1]*au2ang
          << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< atomList[a].pos[2]*au2ang
          << std::resetiosflags(std::ios::scientific)
          << std::resetiosflags(std::ios::showpos)
          << std::endl;
      }
      fout.close();
    } // if( mpirank == 0 )
  }


  statusOFS<< "****************************************************************************" << std::endl << std::endl;
  return;
}

Real TDDFT::InnerSolve( int iscf, Spinor & psiFinal, NumTns<Complex> & tnsTemp, CpxNumMat & HX, CpxNumMat &X, CpxNumMat &HPSI, CpxNumMat & psiF, CpxNumMat & XHX, CpxNumMat & XHXtemp, CpxNumMat & RX, CpxNumMat & Xmid, Real & dT, CpxNumMat & psiRes, CpxNumVec & vin, CpxNumVec & vout, std::vector<CpxNumMat> & dfMat, std::vector<CpxNumMat> & dvMat, DblNumMat & rhoFinal )
{
  Hamiltonian& ham = *hamPtr_;
  Fourier&     fft = *fftPtr_;
  Spinor&      psi = *psiPtr_;
  MPI_Comm mpi_comm = fftPtr_->domain.comm;
  Int mpirank, mpisize;
  MPI_Comm_rank( mpi_comm, &mpirank );
  MPI_Comm_size( mpi_comm, &mpisize );
  Complex i_Z_One = Complex(0.0, 1.0);

  Int  maxDim  = esdfParam.mixMaxDim;
  Real betaMix = esdfParam.mixStepLength;

  Int ntot  = fft.domain.NumGridTotal();
  Int numStateLocal = psiFinal.NumState();
  Int ntotLocal = ntot/mpisize;
  if(mpirank < (ntot % mpisize)) ntotLocal++;
  Int numStateTotal = psi.NumStateTotal();

  Int iterused = std::min (iscf-1, maxDim);
  Int ipos = iscf - 1 - floor( (iscf-2) / maxDim ) * maxDim;

  // Update Hf <== updateV(molf, rhof)
  Int ntotFine  = fft.domain.NumGridTotalFine();
  { 
    if( isCalculateGradRho_ ){
      ham.CalculateGradDensity( fft );
    }
    ham.CalculateXC( Exc_, fft ); 
    ham.CalculateHartree( fft );
    ham.CalculateVtot( ham.Vtot());
  }

  if( ham.IsHybrid() && !esdfParam.isHybridACE ) {
    ham.SetPhiEXX( psiFinal, fft);
  }
#if 0
#endif
  // HXf <== Hf * Xf, now HPSI is HXf  
  ham.MultSpinor( psiFinal, tnsTemp, fft );
  Int width = numStateTotal;
  Int heightLocal = ntotLocal;

  //  XHX <== XHXtemp <--- X'HXf
  //  PsiF, HPSI are psiF and H*psiF
  AlltoallForward( HPSI, HX, mpi_comm);
  AlltoallForward( psiF, X,  mpi_comm);

  blas::Gemm( 'C', 'N', width, width, heightLocal, 1.0, X.Data(), 
      heightLocal, HX.Data(), heightLocal, 0.0, XHXtemp.Data(), width );

  MPI_Allreduce( XHXtemp.Data(), XHX.Data(), width*width, MPI_DOUBLE_COMPLEX, MPI_SUM, mpi_comm );

  // ResX <== Xf + 1i* dT/2 * ( HXf - Xf * XHXf ) - Xmid
  // Note RX is the ResX
  Complex traceXHX (0.0, 0.0);
  for( int i = 0; i < width; i++)
    traceXHX += *(XHX.Data() + i * width + i);

  {
    // remember:
    // X == X in G-parallel
    // XHX == XHXf 
    // Now Y == Xf * XHXf 
    CpxNumMat Y(ntotLocal, numStateTotal); 
    blas::Gemm( 'N', 'N', heightLocal, width, width, 1.0, 
        X.Data(), heightLocal, XHX.Data(), width, 0.0, Y.Data(), heightLocal );

    // Do things in the G-parallel fashion. 
    // HX is the HXf in G-parallel
    // Xmid is in the G-parallel Fashion
    // X is Xf in G-parallel fashion
    // RX is the ResX 

    Complex * ResPtr = RX.Data();
    Complex * XfPtr  = X.Data();
    Complex * HXfPtr = HX.Data();
    Complex * YPtr   = Y.Data();
    Complex * XmidPtr= Xmid.Data();
    for ( int i = 0; i < width; i++)
      for( int j = 0; j < heightLocal; j++){
        int index = i * heightLocal + j;
        ResPtr[index] = XfPtr[index] + i_Z_One * dT / 2.0 * ( HXfPtr[index] - YPtr[index] ) - XmidPtr[index];
      }
  }

  // Tranpose the ResX to band Parallel
  AlltoallBackward( RX, psiRes, mpi_comm);

  // Check check, still have the pre-conditioner 
  // not done yet.
  CpxNumVec preMat(ntot);
  Complex * precPtr = preMat.Data();
  for( int i = 0; i < ntot; i++){
    precPtr[i] = 1.0/(1.0 + i_Z_One * dT/2.0 * ( fft.gkk[i] - traceXHX / (Real)numStateTotal ));
  }

  // FIXME
  CpxNumMat dfMatTemp( ntot, maxDim ); 

  for( int iband = 0; iband < numStateLocal; iband++ ) {

    Complex *vinPtr = vin.Data();
    Complex *voutPtr= vout.Data();
    Complex *psiFPtr= psiF.Data() + iband * ntot;
    Complex *psiResPtr= psiRes.Data() + iband * ntot;

    for( int i = 0; i < ntot; i++){
      vinPtr[i]  =  psiFPtr  [i];
      voutPtr[i] =  psiResPtr[i];
    }

    if( iscf > 1) {
      Complex * dfMatPtr =  dfMat[iband].Data() + (ipos-1) * ntot;
      Complex * dvMatPtr =  dvMat[iband].Data() + (ipos-1) * ntot;

      for( int i = 0; i < ntot; i ++){
        dfMatPtr[i] = dfMatPtr[i] - psiResPtr[i];
        dvMatPtr[i] = dvMatPtr[i] - psiFPtr[i];
      }

      // Least Square problem here. 
      Real rcond = 1.0E-12;
      CpxNumVec gammas;
      DblNumVec S;
      S.Resize(iterused);
      gammas.Resize(ntot);

      blas::Copy( ntot, psiResPtr, 1, gammas.Data(), 1 );
      Int rank;
      Int nrow = iterused;

      // FIXME
      dfMatTemp = dfMat[iband];

      lapack::SVDLeastSquare( ntot, iterused, 1, 
          dfMatTemp.Data(), ntot, gammas.Data(), ntot,
          S.Data(), rcond, &rank );


      Print( statusOFS, "  Rank of dfmat = ", rank );
      Print( statusOFS, "  Rcond = ", rcond );

      blas::Gemv('N', ntot, nrow, -1.0, dvMat[iband].Data(),
          ntot, gammas.Data(), 1, 1.0, vin.Data(), 1 );

      blas::Gemv('N', ntot, iterused, -1.0, dfMat[iband].Data(),
          ntot, gammas.Data(), 1, 1.0, vout.Data(), 1 );

      // statusOFS << "Gammas = " << std::endl;
      // for(Int i = 0; i < iterused; i++ ){
      //   statusOFS << gammas[i] << std::endl;
      // }
    }


    int inext = iscf - std::floor((iscf - 1) / maxDim) *maxDim;

    Complex * dfMatPtr =  dfMat[iband].Data() + (inext-1) * ntot;
    Complex * dvMatPtr =  dvMat[iband].Data() + (inext-1) * ntot;

    for(int j = 0; j < ntot; j++){
      dfMatPtr[j] = psiResPtr[j];
      dvMatPtr[j] = psiFPtr[j];
    }

    // first FFT the vout to the G-space then do the Preconditioner. 
    {
      blas::Copy( ntot, voutPtr, 1, fft.inputComplexVec.Data(), 1 );
      fftw_execute( fft.forwardPlan );
      Complex * tempPtr = fft.outputComplexVec.Data();
      for(int i = 0; i < ntot; ++i)
        tempPtr[i] = tempPtr[i] * precPtr[i];
      fftw_execute( fft.backwardPlan );
      SetValue( vout, Complex(0,0) );
      blas::Axpy( ntot, 1.0 / Real(ntot), fft.inputComplexVec.Data(), 1, voutPtr, 1 );
    }

    for( int j = 0; j < ntot; j++) {
      psiFPtr[j] = vinPtr[j] + betaMix * voutPtr[j];
    }
  } // for (iband)

  Real scfNorm = 0.0;
  {
    // Get the rhoFnew
    Real totalCharge_;
    ham.CalculateDensity(
        psiFinal,
        ham.OccupationRate(),
        totalCharge_, 
        fft );

    // Norm check 
    Real * densityPtr = ham.Density().Data();
    Real * rhoFinalPtr= rhoFinal.Data();
    Real normRhoF = 0.0;
    Real normRhoDiff = 0.0;

    for(int i = 0; i < ntotFine; i++) {
      normRhoDiff += pow( rhoFinalPtr[i] - densityPtr[i], 2.0 ); 
      normRhoF    += pow( rhoFinalPtr[i], 2.0 ); 
    }
    Real scfNorm = std::sqrt(normRhoDiff / normRhoF);
    //Print(statusOFS, "norm(RhoOut-RhoIn)/norm(RhoIn) = ", scfNorm );
    statusOFS << "SCF " << iscf << " norm(RhoOut-RhoIn)/norm(RhoIn): " << scfNorm << std::endl;


    // rhoF <== rhoFNew
    blas::Copy( ntotFine,  ham.Density().Data(), 1,  rhoFinal.Data(), 1 );

    
#if 0
    if( scfNorm < options_.diisTol){
      statusOFS << "TDDFT step " << k_ << " SCF is converged in " << iscf << " steps !" << std::endl;
      //statusOFS << "TDDFT step " << k_ << " used " << totalHx << " H * x operations!" << std::endl;
    }
#endif
    return scfNorm;
  }

}
Real TDDFT::InnerSolve_CN( int iscf, Spinor & psiFinal, NumTns<Complex> & tnsTemp, CpxNumMat & HX, CpxNumMat &X, CpxNumMat &HPSI, CpxNumMat & psiF, CpxNumMat & XHX, CpxNumMat & XHXtemp, CpxNumMat & RX, CpxNumMat & Xmid, Real & dT, CpxNumMat & psiRes, CpxNumVec & vin, CpxNumVec & vout, std::vector<CpxNumMat> & dfMat, std::vector<CpxNumMat> & dvMat, DblNumMat & rhoFinal )
{
  Hamiltonian& ham = *hamPtr_;
  Fourier&     fft = *fftPtr_;
  Spinor&      psi = *psiPtr_;
  MPI_Comm mpi_comm = fftPtr_->domain.comm;
  Int mpirank, mpisize;
  MPI_Comm_rank( mpi_comm, &mpirank );
  MPI_Comm_size( mpi_comm, &mpisize );
  Complex i_Z_One = Complex(0.0, 1.0);

  Int  maxDim  = esdfParam.mixMaxDim;
  Real betaMix = esdfParam.mixStepLength;

  Int ntot  = fft.domain.NumGridTotal();
  Int numStateLocal = psiFinal.NumState();
  Int ntotLocal = ntot/mpisize;
  if(mpirank < (ntot % mpisize)) ntotLocal++;
  Int numStateTotal = psi.NumStateTotal();

  Int iterused = std::min (iscf-1, maxDim);
  Int ipos = iscf - 1 - floor( (iscf-2) / maxDim ) * maxDim;

  // Update Hf <== updateV(molf, rhof)
  Int ntotFine  = fft.domain.NumGridTotalFine();
  { 
    if( isCalculateGradRho_ ){
      ham.CalculateGradDensity( fft );
    }
    ham.CalculateXC( Exc_, fft ); 
    ham.CalculateHartree( fft );
    ham.CalculateVtot( ham.Vtot());
  }

  if( ham.IsHybrid() && !esdfParam.isHybridACE ) {
    ham.SetPhiEXX( psiFinal, fft);
  }
#if 0
#endif
  // HXf <== Hf * Xf, now HPSI is HXf  
  ham.MultSpinor( psiFinal, tnsTemp, fft );
  Int width = numStateTotal;
  Int heightLocal = ntotLocal;

  //  XHX <== XHXtemp <--- X'HXf
  //  PsiF, HPSI are psiF and H*psiF
  AlltoallForward( HPSI, HX, mpi_comm);
  AlltoallForward( psiF, X,  mpi_comm);

  blas::Gemm( 'C', 'N', width, width, heightLocal, 1.0, X.Data(), 
      heightLocal, HX.Data(), heightLocal, 0.0, XHXtemp.Data(), width );

  MPI_Allreduce( XHXtemp.Data(), XHX.Data(), width*width, MPI_DOUBLE_COMPLEX, MPI_SUM, mpi_comm );

  // ResX <== Xf + 1i* dT/2 * ( HXf - Xf * XHXf ) - Xmid
  // Note RX is the ResX
  Complex traceXHX (0.0, 0.0);
  for( int i = 0; i < width; i++)
    traceXHX += *(XHX.Data() + i * width + i);

  {
    // remember:
    // X == X in G-parallel
    // XHX == XHXf 
    // Now Y == Xf * XHXf 
    //CpxNumMat Y(ntotLocal, numStateTotal); 
    //blas::Gemm( 'N', 'N', heightLocal, width, width, 1.0, 
    //    X.Data(), heightLocal, XHX.Data(), width, 0.0, Y.Data(), heightLocal );

    // Do things in the G-parallel fashion. 
    // HX is the HXf in G-parallel
    // Xmid is in the G-parallel Fashion
    // X is Xf in G-parallel fashion
    // RX is the ResX 

    Complex * ResPtr = RX.Data();
    Complex * XfPtr  = X.Data();
    Complex * HXfPtr = HX.Data();
    //Complex * YPtr   = Y.Data();
    Complex * XmidPtr= Xmid.Data();
    for ( int i = 0; i < width; i++)
      for( int j = 0; j < heightLocal; j++){
        int index = i * heightLocal + j;
        ResPtr[index] = XfPtr[index] + i_Z_One * dT / 2.0 * HXfPtr[index] - XmidPtr[index];
      }
  }

  // Tranpose the ResX to band Parallel
  AlltoallBackward( RX, psiRes, mpi_comm);

  // Check check, still have the pre-conditioner 
  // not done yet.
  CpxNumVec preMat(ntot);
  Complex * precPtr = preMat.Data();
  for( int i = 0; i < ntot; i++){
    precPtr[i] = 1.0/(1.0 + i_Z_One * dT/2.0 * fft.gkk[i]);
  }

  // FIXME
  CpxNumMat dfMatTemp( ntot, maxDim ); 

  for( int iband = 0; iband < numStateLocal; iband++ ) {

    Complex *vinPtr = vin.Data();
    Complex *voutPtr= vout.Data();
    Complex *psiFPtr= psiF.Data() + iband * ntot;
    Complex *psiResPtr= psiRes.Data() + iband * ntot;

    for( int i = 0; i < ntot; i++){
      vinPtr[i]  =  psiFPtr  [i];
      voutPtr[i] =  psiResPtr[i];
    }

    if( iscf > 1) {
      Complex * dfMatPtr =  dfMat[iband].Data() + (ipos-1) * ntot;
      Complex * dvMatPtr =  dvMat[iband].Data() + (ipos-1) * ntot;

      for( int i = 0; i < ntot; i ++){
        dfMatPtr[i] = dfMatPtr[i] - psiResPtr[i];
        dvMatPtr[i] = dvMatPtr[i] - psiFPtr[i];
      }

      // Least Square problem here. 
      Real rcond = 1.0E-12;
      CpxNumVec gammas;
      DblNumVec S;
      S.Resize(iterused);
      gammas.Resize(ntot);

      blas::Copy( ntot, psiResPtr, 1, gammas.Data(), 1 );
      Int rank;
      Int nrow = iterused;

      // FIXME
      dfMatTemp = dfMat[iband];

      lapack::SVDLeastSquare( ntot, iterused, 1, 
          dfMatTemp.Data(), ntot, gammas.Data(), ntot,
          S.Data(), rcond, &rank );


      Print( statusOFS, "  Rank of dfmat = ", rank );
      Print( statusOFS, "  Rcond = ", rcond );

      blas::Gemv('N', ntot, nrow, -1.0, dvMat[iband].Data(),
          ntot, gammas.Data(), 1, 1.0, vin.Data(), 1 );

      blas::Gemv('N', ntot, iterused, -1.0, dfMat[iband].Data(),
          ntot, gammas.Data(), 1, 1.0, vout.Data(), 1 );

      // statusOFS << "Gammas = " << std::endl;
      // for(Int i = 0; i < iterused; i++ ){
      //   statusOFS << gammas[i] << std::endl;
      // }
    }


    int inext = iscf - std::floor((iscf - 1) / maxDim) *maxDim;

    Complex * dfMatPtr =  dfMat[iband].Data() + (inext-1) * ntot;
    Complex * dvMatPtr =  dvMat[iband].Data() + (inext-1) * ntot;

    for(int j = 0; j < ntot; j++){
      dfMatPtr[j] = psiResPtr[j];
      dvMatPtr[j] = psiFPtr[j];
    }

    // first FFT the vout to the G-space then do the Preconditioner. 
    {
      blas::Copy( ntot, voutPtr, 1, fft.inputComplexVec.Data(), 1 );
      fftw_execute( fft.forwardPlan );
      Complex * tempPtr = fft.outputComplexVec.Data();
      for(int i = 0; i < ntot; ++i)
        tempPtr[i] = tempPtr[i] * precPtr[i];
      fftw_execute( fft.backwardPlan );
      SetValue( vout, Complex(0,0) );
      blas::Axpy( ntot, 1.0 / Real(ntot), fft.inputComplexVec.Data(), 1, voutPtr, 1 );
    }

    for( int j = 0; j < ntot; j++) {
      psiFPtr[j] = vinPtr[j] + betaMix * voutPtr[j];
    }
  } // for (iband)

  Real scfNorm = 0.0;
  {
    // Get the rhoFnew
    Real totalCharge_;
    ham.CalculateDensity(
        psiFinal,
        ham.OccupationRate(),
        totalCharge_, 
        fft );

    // Norm check 
    Real * densityPtr = ham.Density().Data();
    Real * rhoFinalPtr= rhoFinal.Data();
    Real normRhoF = 0.0;
    Real normRhoDiff = 0.0;

    for(int i = 0; i < ntotFine; i++) {
      normRhoDiff += pow( rhoFinalPtr[i] - densityPtr[i], 2.0 ); 
      normRhoF    += pow( rhoFinalPtr[i], 2.0 ); 
    }
    Real scfNorm = std::sqrt(normRhoDiff / normRhoF);
    //Print(statusOFS, "norm(RhoOut-RhoIn)/norm(RhoIn) = ", scfNorm );
    statusOFS << "SCF " << iscf << " norm(RhoOut-RhoIn)/norm(RhoIn): " << scfNorm << std::endl;


    // rhoF <== rhoFNew
    blas::Copy( ntotFine,  ham.Density().Data(), 1,  rhoFinal.Data(), 1 );

    
#if 0
    if( scfNorm < options_.diisTol){
      statusOFS << "TDDFT step " << k_ << " SCF is converged in " << iscf << " steps !" << std::endl;
      //statusOFS << "TDDFT step " << k_ << " used " << totalHx << " H * x operations!" << std::endl;
    }
#endif
    return scfNorm;
  }
}// InnerSolve_CN
}
#endif
