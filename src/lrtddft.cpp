#include  "lrtddft.hpp"
#include  "utility.hpp"
#include  "blas.hpp"
#include  "lapack.hpp"
#include  "scalapack.hpp"
#include  "mpi_interf.hpp"

#define SINGLE_PRECISION_GEMM

#ifdef SINGLE_PRECISION_GEMM
#include  <stdio.h>
#include  <stdlib.h>
#endif

using namespace dgdft::PseudoComponent;
using namespace dgdft::scalapack;
using namespace dgdft::esdf;

namespace dgdft
{
void LRTDDFT::Setup(
    Hamiltonian& ham,
    Spinor& psi,
    Fourier& fft,
    const Domain& dm)
{
  //setup:

  numExtraState_ = ham.NumExtraState();               //maxncbandtol
  nocc_ = ham.NumOccupiedState();                     //maxnvbandtol
  density_ = ham.Density();                           //rho in Fine grid
  eigVal_ = ham.EigVal();                             //ev 

  numcomponent_ = psi.NumComponent();                 //spinor
  ntotR2C_ = fft.numGridTotalR2C;                     //fft Grid number
  ntotR2CFine_ = fft.numGridTotalR2CFine;             //fft Fine Grid number
  ntot_ = psi.NumGridTotal();                         //real space Grid number
  ntotFine_ = dm.NumGridTotalFine();                  //real space Fine Grid number

  vol_ = dm.Volume();
  domain_ = dm;

  //MPI Setup
  int mpirank;  MPI_Comm_rank(domain_.comm, &mpirank);
  int mpisize;  MPI_Comm_size(domain_.comm, &mpisize);

  if (mpirank == 0) {
    std::ostringstream structStream;
#if ( _DEBUGlevel_ >= 0 )
    statusOFS << std::endl
      << "Output the LRTDDFT information"
      << std::endl << std::endl;
#endif
  }

  return;
}         // -----  end of LRTDDFT::Setup  -----

void LRTDDFT::FFTRtoC(
    Fourier& fft,
    Hamiltonian& ham,
    DblNumMat psiphi,
    DblNumMat& temp,
    Int ncvband)
{
  Real          facH = 2.0/ vol_;

#ifdef _USE_OPENMP_
#pragma omp parallel
  {
#endif
    //Int ntothalf = fftPtr->numGridTotalR2C;
    // These two are private variables in the OpenMP context

#ifdef _USE_OPENMP_
#pragma omp for schedule (dynamic,1) nowait
#endif

    for (Int mu = 0; mu < ncvband; mu++) {

      SetValue(fft.inputVecR2C, 0.0);
      SetValue(fft.outputVecR2C, Z_ZERO);

      blas::Copy(ntot_, psiphi.VecData(mu), 1, fft.inputVecR2C.Data(), 1);

      // Fourier transform of wavefunction saved in fft.outputComplexVec

      FFTWExecute(fft, fft.forwardPlanR2C);

      for (Int i = 0; i < ntotR2C_; i++) {
        if (fft.gkkR2C(i) < 1e-8) {
          fft.outputVecR2C(i) = 0;
        }
        else {
          fft.outputVecR2C(i) = fft.outputVecR2C(i) * 4.0 * PI / fft.gkkR2C(i) * facH;
        }
      }

      FFTWExecute(fft, fft.backwardPlanR2C);

      blas::Copy(ntot_, fft.inputVecR2C.Data(), 1, temp.VecData(mu), 1);

    }
#ifdef _USE_OPENMP_
  } //#pragma omp parallel
#endif

  return;
}

void LRTDDFT::Calculatefxc(
    Fourier& fft,
    DblNumVec& fxcPz)
{
  const double  a = 0.0311;
  const double  b = -0.048;
  const double  c = 0.0020;
  const double  d = -0.0116;
  const double  gc = -0.1423;
  const double  b1 = 1.0529;
  const double  b2 = 0.3334;
  const double  falpha = -0.458165293283143;
  const double  e2 = 1;
  Real          rhoxc;
  Real          rs;
  Real          lnrs;
  Real          rs12;
  Real          ox;
  Real          dox;
  Real          drsdrho;
  Real          dvxdrho;
  Real          dvcdrho;
  DblNumMat     density(ntot_, numcomponent_);
  SetValue( density, 0.0);

  //Fine to coarse grid
  SetValue( fft.inputVecR2CFine, 0.0 );
  SetValue( fft.outputVecR2CFine, Z_ZERO);
  blas::Copy( ntotFine_, density_.Data(), 1, fft.inputVecR2CFine.Data(), 1 );

// xmqin ====================================================================
//  fftw_execute( fft.forwardPlanFine );

  FFTWExecute ( fft, fft.forwardPlanR2CFine );

//1. fine to coarse:  even version
  if( esdfParam.FFTtype == "even" || esdfParam.FFTtype == "power")
  {
    Index3& numGrid = fft.domain.numGrid;
    Index3& numGridFine = fft.domain.numGridFine;
    Complex *fftOutFinePtr = fft.outputVecR2CFine.Data();
    Complex *fftOutPtr = fft.outputVecR2C.Data();
    IP_f2c(numGrid.Data(), numGridFine.Data(), fftOutPtr, fftOutFinePtr);
   }
//2. fine to coarse:  odd version
  else if(esdfParam.FFTtype == "odd" )
  {
   Real fac = double(ntotFine_) / double(ntot_);
    Int *idxPtr = fft.idxFineGridR2C.Data();
    Complex *fftOutFinePtr = fft.outputVecR2CFine.Data();
    Complex *fftOutPtr = fft.outputVecR2C.Data();
    for( Int i = 0; i < ntotR2C_; i++ ){
      *(fftOutPtr++) += fftOutFinePtr[*(idxPtr++)] * fac;
    }
  }
   
//============================================================================

  FFTWExecute ( fft, fft.backwardPlanR2C );
  blas::Axpy( ntot_, 1.0, fft.inputVecR2C.Data(), 1, density.Data(), 1 );

  for (Int i = 0; i < ntot_; i++) {
    rhoxc = density(i, 0);
    if (rhoxc > 1e-15) {
      rs = pow(((3 / (4 * PI)) / rhoxc), 1 / 3);
      drsdrho = -pow(rs, 4.0) * 4 * PI / 9;
      dvxdrho = 16 * PI / 27 * falpha * pow(rs, 2);
      if (rs < 1) {
        lnrs = log(rs);
        dvcdrho = a / rs + 2 / 3 * c * (lnrs + 1) + (2 * d - c) / 3;
      }
      else {
        rs12 = sqrt(rs);
        ox = 1 + b1 * rs12 + 4 / 3 * b2 * rs;
        dox = 1 + 7 / 6 * b1 * rs12 + 4 / 3 * b2 * rs;
        dvcdrho = gc / (pow(ox, 2.0)) * (7 / 12 * b1 / rs12 + 4 / 3 * b2 - dox / ox * (b1 / rs12 + 2 * b2));
      }
      fxcPz(i) = e2 * (dvcdrho * drsdrho + dvxdrho);
    }
    else fxcPz(i) = 0;
  }
  return;
}

void LRTDDFT::Spectrum(
    Fourier& fft, 
    DblNumMat psinv,
    DblNumMat psinc, 
    DblNumMat XX, 
    DblNumVec eigValS, 
    Int nkband){

  // setup paramater
  Int nvband = esdfParam.nvband;
  Int ncband = esdfParam.ncband;
  Int ncvband = nvband*ncband;
  Int nroots = nkband;
  Int ntot = ntot_;
  Int ntotR2C = ntotR2C_;
  Real vol = vol_;
  
  // following is g space method
  // Calculate the martix element without electron-hole interaction,
  // ref: Rohlfing & Louie PRB 62 4927 (2000).
  
  // following is real space method
  
  statusOFS << " ************************ WARNING ******************************** " << std::endl;
  statusOFS << " Warning: Please make sure that your atoms are centered at (0,0,0) " << std::endl;
  statusOFS << " ************************ WARNING ******************************** " << std::endl;
  DblNumVec Xr( fft.domain.NumGridTotal() );
  DblNumVec Yr( fft.domain.NumGridTotal() );
  DblNumVec Zr( fft.domain.NumGridTotal() );
  DblNumVec D( fft.domain.NumGridTotal() );
  DblNumVec pol(3);
  // pol dirction
  pol[0] = esdfParam.LRTDDFTVextPolx;
  pol[1] = esdfParam.LRTDDFTVextPoly;
  pol[2] = esdfParam.LRTDDFTVextPolz;
  
  Real * xr = Xr.Data();
  Real * yr = Yr.Data();
  Real * zr = Zr.Data();
  Int  idx; 
  Real Xtmp, Ytmp, Ztmp;     
  for( Int k = 0; k < fft.domain.numGrid[2]; k++ ){
    for( Int j = 0; j < fft.domain.numGrid[1]; j++ ){
      for( Int i = 0; i < fft.domain.numGrid[0]; i++ ){
        idx = i + j * fft.domain.numGrid[0] + k * fft.domain.numGrid[0] * fft.domain.numGrid[1];
        Xtmp = (Real(i) - Real( round(Real(i)/Real(fft.domain.numGrid[0])) * Real(fft.domain.numGrid[0]) ) ) / Real(fft.domain.numGrid[0]);
        Ytmp = (Real(j) - Real( round(Real(j)/Real(fft.domain.numGrid[1])) * Real(fft.domain.numGrid[1]) ) ) / Real(fft.domain.numGrid[1]);
        Ztmp = (Real(k) - Real( round(Real(k)/Real(fft.domain.numGrid[2])) * Real(fft.domain.numGrid[2]) ) ) / Real(fft.domain.numGrid[2]);
        // should be AL(0,0) * X + AL(0,1) * Y + AL(0,2) * Z
        // the other parts are zeros. 
        xr[idx] = Xtmp * esdfParam.domain.length[0]*pol[0] ;
        yr[idx] = Ytmp * esdfParam.domain.length[1]*pol[1] ;
        zr[idx] = Ztmp * esdfParam.domain.length[2]*pol[2] ;
        // get the p.D corresponding to the matlab KSSOLV 
        D[idx] = xr[idx] + yr[idx] + zr[idx];
      }
    }
  }
  
  //statusOFS << "D        = " << D << std::endl;

  // Calculate Delta Dipole
  // D Pi = -∫ Delta Rho(i, j, k) * Ri(i, j, k) *dr
  
  // calculate dpsi and drho
  DblNumMat  Xiv(ncband, nvband);
  SetValue(Xiv, 0.0);
  Int ntotLocal = psinv.m();
  DblNumMat dpsiLocal(ntotLocal,nvband);
  SetValue(dpsiLocal, 0.0);
 
  //MPI init
  int mpirank;  MPI_Comm_rank(domain_.comm, &mpirank);
  int mpisize;  MPI_Comm_size(domain_.comm, &mpisize);
  Int ntotBlocksize = ntot / mpisize;
 
  IntNumVec  weightSizeDispls(mpisize);
  IntNumVec  weightSize(mpisize);
  DblNumVec deltadensityLocal(ntotLocal);
  SetValue(deltadensityLocal, 0.0);
  DblNumVec deltadensity(ntot);
  SetValue(deltadensity, 0.0);
  
  DblNumVec ChiX( nroots );
  DblNumVec ChiY( nroots );
  DblNumVec ChiZ( nroots );
  SetValue(ChiX, 0.0);
  SetValue(ChiY, 0.0);
  SetValue(ChiZ, 0.0);
  DblNumMat Mu(nroots,4);
  SetValue(Mu, 0.0);
  
  if ((ntot % mpisize) == 0) {
    for (Int i = 0; i < mpisize; i++){
      weightSizeDispls[i] = i * ntotBlocksize;
      weightSize[i] = ntotBlocksize;
    }
  }
  else{
    for (Int i = 0; i < mpisize; i++){
      if (i < (ntot % mpisize)) {
        weightSizeDispls[i] = i * (ntotBlocksize + 1);
        weightSize[i] = ntotBlocksize + 1;
      }
      else{
        weightSizeDispls[i] = (ntot%mpisize) * (ntotBlocksize + 1) + (i-(ntot%mpisize)) * (ntotBlocksize);
        weightSize[i] = ntotBlocksize;
      }
    }
  }
  
  //statusOFS << "XX        = " << XX << std::endl;
 
  for( Int i = 0; i < nroots; i++ ){
    Real *wp = XX.Data()+ i*ncvband;
    Real *p = Xiv.Data();  
    for (Int j = 0; j < ncvband; j++){
      *(p++) = *(wp++);
    }

    //statusOFS << "Xiv        = " << Xiv << std::endl;

    blas::Gemm('N', 'N', ntotLocal, nvband, ncband, 1.0, psinc.Data(), ntotLocal, Xiv.Data(), ncband, 0.0, dpsiLocal.Data(), ntotLocal);
    for (Int j = 0; j < ntotLocal; j++){
      for (Int k = 0; k < nvband; k++){
        deltadensityLocal(j) += dpsiLocal(j,k)*psinv(j,k)*sqrt(ntot);
      }
    }
    MPI_Allgatherv(deltadensityLocal.Data(), ntotLocal, MPI_DOUBLE, deltadensity.Data(), weightSize.Data(), weightSizeDispls.Data(), MPI_DOUBLE, domain_.comm);
    for (Int j = 0; j < ntot; j++){
      ChiX(i) += deltadensity(j)*Xr(j)*vol/ntot;
      ChiY(i) += deltadensity(j)*Yr(j)*vol/ntot;
      ChiZ(i) += deltadensity(j)*Zr(j)*vol/ntot;
    }
    Mu(i,0) = ChiX(i)*ChiX(i)*2.0/(3.0*PI);
    Mu(i,1) = ChiY(i)*ChiY(i)*2.0/(3.0*PI);
    Mu(i,2) = ChiZ(i)*ChiZ(i)*2.0/(3.0*PI);
    Mu(i,3) = Mu(i,0) + Mu(i,1) + Mu(i,2);
  }
  std::ofstream DipoleStream("Dipole");
  if( !DipoleStream.good() ){
    ErrorHandling( "Dipole file cannot be opened." );
  }
  statusOFS << "Diopole        = " << Mu << std::endl;

  Real Emax = eigValS[nroots]*13.60569253;
  Real Emin = eigValS[0]*13.60569253;;
  Real wgrid = esdfParam.LRTDDFTOmegagrid;
  Real sigma = esdfParam.LRTDDFTSigma;
  Int nw = floor((Emax - Emin)/wgrid);
  DblNumMat spectrum(nw,5);
  SetValue(spectrum,0.0);
  Real w = 0;
  Real ry2ev = 13.60569253;
  // Spectrum: 1-Energy 2-Total oscillator strength 3-X 4-Y 5-Z
  for( Int i = 0; i < nroots; i++ ){
    for( Int iw = 0; iw < nw; iw++ ){
      w = (iw - 1)*wgrid + Emin - eigValS[i]*13.60569253;
      spectrum(iw,0) = (iw-1)*wgrid + Emin;
      spectrum(iw,1) += sigma/(pow(w,2)+pow(sigma,2))*Mu(i,0);
      spectrum(iw,2) += sigma/(pow(w,2)+pow(sigma,2))*Mu(i,1);
      spectrum(iw,3) += sigma/(pow(w,2)+pow(sigma,2))*Mu(i,2);
      spectrum(iw,4) = spectrum(iw,1) + spectrum(iw,2) + spectrum(iw,3);
    }
  }
  std::ofstream SpecStream("Spectrum");
  if( !SpecStream.good() ){
    ErrorHandling( "Spectrum file cannot be opened." );
  }
  statusOFS << "Spectrum       = " << spectrum << std::endl;
}


void LRTDDFT::CalculateLRTDDFT(
    Hamiltonian& ham,
    Spinor& psi,
    Fourier& fft,
    const Domain& dm) {

  Int nvband  = esdfParam.nvband;
  Int ncband  = esdfParam.ncband;
  Int ncvband = ncband * nvband;                         //Matrix size

  //Final matrix
  DblNumMat  hamTDDFT(ncvband, ncvband);                 //Save all matrix
  SetValue(hamTDDFT, 0.0);
  DblNumVec  eigValS(ncvband);                           //Save energy eig
  SetValue(eigValS, 0.0);

  //main
  int kk = 0;

  //time
  Real timeSta;
  Real timeEnd;
  Real timeStaTotal;
  Real timeEndTotal;
  Real timeFFTRtoC = 0.0;
  Real timeCalculateHartree = 0.0;
  Real timeCalculatefxc = 0.0;

  //MPI init

  MPI_Barrier(domain_.comm);
  int mpirank;  MPI_Comm_rank(domain_.comm, &mpirank);
  int mpisize;  MPI_Comm_size(domain_.comm, &mpisize);

  Int ntot = ntot_;
  Int ntotR2C = ntotR2C_;
  Int ncom = numcomponent_;
  Int numStateTotal = psi.NumStateTotal();
  Real vol = vol_;

  // Convert the column partition to row partition

  Int numStateBlocksize = numStateTotal / mpisize;
  Int ntotBlocksize = ntot / mpisize;

  Int numStateLocal = numStateBlocksize;
  Int ntotLocal = ntotBlocksize;

  Int numStateBlocksizeNcv = ncvband / mpisize;
  Int numStateLocalNcv = numStateBlocksizeNcv;

  if (mpirank < (numStateTotal % mpisize)) {
    numStateLocal = numStateBlocksize + 1;
  }

  if (mpirank < (ncvband % mpisize)) {
    numStateLocalNcv = numStateBlocksizeNcv + 1;
  }

  if (mpirank < (ntot % mpisize)) {
    ntotLocal = ntotBlocksize + 1;
  }

  GetTime(timeStaTotal);

  DblNumMat psiphiRow(ntotLocal, ncvband);                       //Save wavefunproduct Row
  SetValue(psiphiRow, 0.0);

  DblNumMat tempRow(ntotLocal, ncvband);                         //Save HartreeFock wave Row
  SetValue(tempRow, 0.0);

  {
    DblNumMat psiCol(ntot, numStateLocal);                         //Save wavefun Col
    SetValue(psiCol, 0.0);
    DblNumMat psiRow(ntotLocal, numStateTotal);                    //Save wavefun Row  
    SetValue(psiRow, 0.0);

    DblNumMat psiphiCol(ntot, numStateLocalNcv);                   //Save wavefunproduct Col
    SetValue(psiphiCol, 0.0);

    GetTime(timeSta);
    lapack::Lacpy('A', ntot, numStateLocal, psi.Wavefun().Data(), ntot, psiCol.Data(), ntot);
    GetTime(timeEnd);
    statusOFS << "Time for Lacpy                   = " << timeEnd - timeSta << " [s]" << std::endl;

    GetTime(timeSta);
    AlltoallForward(psiCol, psiRow, domain_.comm);

    //SCALAPACK(pdgemr2d)(&Ng, &No, PcRow.Data(), &I_ONE, &I_ONE, desc_NgNo1DRow,
    //	    PcCol.Data(), &I_ONE, &I_ONE, desc_NgNo1DCol, &contxt1DCol );

    GetTime(timeEnd);
    statusOFS << "Time for AlltoallForward         = " << timeEnd - timeSta << " [s]" << std::endl;

    GetTime(timeSta);

// product state begin

#ifdef _USE_OPENMP_
#pragma omp parallel
    {
#endif

#ifdef _USE_OPENMP_
#pragma omp for schedule(dynamic,1)
#endif
      for (Int k = nocc_ - nvband; k < nocc_; k++) {
        for (Int j = nocc_; j < nocc_ + ncband; j++) {
          for (Int i = 0; i < ntotLocal; i++) {
            psiphiRow(i, kk) = (psiRow(i, k)) * (psiRow(i, j));
          }//for i 
          kk++; 
        }//for j
      }//for k

#ifdef _USE_OPENMP_
    } //#pragma omp parallel
#endif

// product state end

    GetTime(timeEnd);
    statusOFS << "Time for psiphiRow               = " << timeEnd - timeSta << " [s]" << std::endl;

    GetTime(timeSta);
    AlltoallBackward(psiphiRow, psiphiCol, domain_.comm);
    GetTime(timeEnd);
    statusOFS << "Time for AlltoallBackward        = " << timeEnd - timeSta << " [s]" << std::endl;

    // Fourier transform

    DblNumMat     tempCol(ntot_, numStateLocalNcv);                  //Save HartreeFock wave
    SetValue(tempCol, 0.0);

    GetTime(timeSta);
    FFTRtoC(fft, ham, psiphiCol, tempCol, numStateLocalNcv);
    GetTime(timeEnd);
    statusOFS << "Time for FFTRtoC                 = " << timeEnd - timeSta << " [s]" << std::endl;
    timeFFTRtoC = timeFFTRtoC + (timeEnd - timeSta);

    //end of FFT

    //get Hartree potential   

    //MPI belong to Hartree Fock

    GetTime(timeSta);
    AlltoallForward(tempCol, tempRow, domain_.comm);
    GetTime(timeEnd);
    statusOFS << "Time for AlltoallForward         = " << timeEnd - timeSta << " [s]" << std::endl;
  }
    
  if(0){
    //Hartree Fock setup
    DblNumMat     HartreeFock(ncvband, ncvband);                             //Save HartreeFock matrix
    SetValue(HartreeFock, 0.0);
    DblNumMat     HartreeFockLocal(ncvband, ncvband);                         //Save Local HartreeFock matrix
    SetValue(HartreeFockLocal, 0.0);

    GetTime(timeSta);
    blas::Gemm('T', 'N', ncvband, ncvband, ntotLocal, 1.0, psiphiRow.Data(), ntotLocal, tempRow.Data(), ntotLocal, 0.0, HartreeFockLocal.Data(), ncvband);
    GetTime(timeEnd);
    statusOFS << "time for Hatree product          = " << timeEnd - timeSta << " [s]" << std::endl;                

    //reduce

    GetTime(timeSta);
    MPI_Allreduce(HartreeFockLocal.Data(), HartreeFock.Data(), ncvband*ncvband, MPI_DOUBLE, MPI_SUM, domain_.comm);
    GetTime(timeEnd);
    statusOFS << "time for MPI_Allreduce           = " << timeEnd - timeSta << " [s]" << std::endl;                
  }
  //end of Hartree Fock

  //get fxc potential

  //fxc setup
  DblNumMat     fxc(ncvband, ncvband);                                   //Save fxc matrix
  SetValue(fxc, 0.0);
  DblNumMat     psiphifxcR2C(ntotLocal, ncvband);                      //Save fxc wavefun
  SetValue(psiphifxcR2C, 0.0);

  //Calculatefxc operator

  DblNumVec     fxcPz(ntot);                                       //Save fxc operator
  SetValue(fxcPz, 0.0); 

  GetTime(timeSta);
  Calculatefxc(fft, fxcPz);
  GetTime(timeEnd);
  statusOFS << "Time for Calculatefxc            = " << timeEnd - timeSta << " [s]" << std::endl;

  //fxcPz send to every MPI 

  GetTime(timeSta);
  MPI_Bcast(fxcPz.Data(), ntot, MPI_DOUBLE, 0, domain_.comm);
  GetTime(timeEnd);
  statusOFS << "Time for MPI_Bcast               = " << timeEnd - timeSta << " [s]" << std::endl;

  DblNumMat     fxcLocal(ncvband, ncvband);
  SetValue(fxcLocal, 0.0);

  //Calculate fxc wavefun;
#ifdef _USE_OPENMP_
#pragma omp parallel
  {
#endif

    double facfxc = 2.0 * vol_ / double(ntot_);

    GetTime(timeSta);

#ifdef _USE_OPENMP_
#pragma omp for schedule(dynamic,1)
#endif
    for (Int mu = 0; mu < ncvband; mu++) {
      for (Int i = 0; i < ntotLocal; i++) {
        psiphifxcR2C(i, mu) = psiphiRow(i, mu) * fxcPz(i + mpirank * ntotLocal) * facfxc + tempRow(i, mu);
      }
    }//for mu  

#ifdef _USE_OPENMP_
  } //#pragma omp parallel
#endif

  GetTime(timeEnd);
  statusOFS << "Time for psiphifxcR2C            = " << timeEnd - timeSta << " [s]" << std::endl;

  //Calculate Hatree-fxc product;

  //DblNumMat     fxcLocal(ncvband, ncvband);
  //SetValue(fxcLocal, 0.0);

  GetTime(timeSta);
  blas::Gemm('T', 'N', ncvband, ncvband, ntotLocal, 1.0, psiphiRow.Data(), ntotLocal, psiphifxcR2C.Data(), ntotLocal, 0.0, fxcLocal.Data(), ncvband);
  GetTime(timeEnd);
  statusOFS << "Time for Hatree-fxc product      = " << timeEnd - timeSta << " [s]" << std::endl;

  //#ifdef _USE_OPENMP_
  //#pragma omp critical
  //    {
  //#endif

  // This is a reduce operation for an array, and should be
  // done in the OMP critical syntax
  GetTime(timeSta);
  MPI_Allreduce(fxcLocal.Data(), fxc.Data(), ncvband*ncvband, MPI_DOUBLE, MPI_SUM, domain_.comm);
  GetTime(timeEnd);
  statusOFS << "Time for MPI_Allreduce           = " << timeEnd - timeSta << " [s]" << std::endl;
  //#ifdef _USE_OPENMP_
  //    } //#pragma omp critical
  //#endif

  //end of get fxc potential

  //Diagonal part

  //Diagonal part setup
  DblNumMat Energydiff(ncvband, ncvband);
  SetValue(Energydiff, 0.0);

  kk = 0;
  GetTime(timeSta);
  for (Int k = nocc_ - nvband; k < nocc_; k++) {
    for (Int j = nocc_; j < nocc_ + ncband; j++) {      
      fxc(kk, kk) += eigVal_(j) - eigVal_(k);
      kk++;
    }//for j
  }//for k
  GetTime(timeEnd);
  statusOFS << "Time for Energydiff              = " << timeEnd - timeSta << " [s]" << std::endl;

  GetTime(timeSta);

  lapack::Lacpy( 'A', ncvband, ncvband, fxc.Data(), ncvband, hamTDDFT.Data(), ncvband );
  //blas::Axpy( ncvband, 1.0, fxc.Data(), 1, hamTDDFT.Data(), 1 );

  // eigensolver part

  if(0){
    for (Int i = 0; i < ncvband; i++) {
      for (Int j = 0; j < ncvband; j++) {
        hamTDDFT(i, j) = Energydiff(i, j) + fxc(i, j);
      }
    }
  }

  if(0){
    Real *TDDFTncv = hamTDDFT.Data();
    Real *Energydiffncv = Energydiff.Data();
    Real *fxcncv = fxc.Data();
    for( Int i = 0; i < ncvband*ncvband; i++ ){
      *(TDDFTncv++) = *(Energydiffncv++) + *(fxcncv++) ;
    }
  }

  GetTime(timeEnd);

  statusOFS << "Time for hamTDDFT                = " << timeEnd - timeSta << " [s]" << std::endl;


  if(1){	

    GetTime(timeSta);
    lapack::Syevd('V', 'U', ncvband, hamTDDFT.Data(), ncvband, eigValS.Data());
    GetTime(timeEnd);
    statusOFS << "Time for Syevd in lapack         = " << timeEnd - timeSta << " [s]" << std::endl;

    statusOFS << "eigValS = " << eigValS << std::endl;
    //statusOFS << "XX      = " << hamTDDFT  << std::endl;

  } //end if(0)


  if(0){

    GetTime(timeSta);

    // Setup BLACS
    Int contxt;
    Int nprow, npcol, myrow, mycol, info;
    Int scaBlockSize      = esdfParam.scaBlockSize;
    Int numProcScaLAPACK  = esdfParam.numProcScaLAPACKPW;

    for(Int i = IRound(sqrt(double(numProcScaLAPACK))); i <= numProcScaLAPACK; i++){
      nprow = i; 
      npcol = int(numProcScaLAPACK / nprow);
      if( nprow * npcol == numProcScaLAPACK ) break;
    }

    Cblacs_get(0, 0, &contxt);
    Cblacs_gridinit(&contxt, "C", nprow, npcol);
    Cblacs_gridinfo(contxt, &nprow, &npcol, &myrow, &mycol);

    //IntNumVec pmap(numProcScaLAPACK);
    //for ( Int i = 0; i < numProcScaLAPACK_; i++ ){
    //  pmap[i] = i;
    //}
    //Cblacs_get(0, 0, &contxt_);
    //Cblacs_gridmap(&contxt_, &pmap[0], nprow_, nprow_, npcol_);

    Int numKeep = ncvband;
    Int lda = ncvband;
    scalapack::ScaLAPACKMatrix<Real> square_mat_scala;
    scalapack::ScaLAPACKMatrix<Real> eigvecs_scala;
    scalapack::Descriptor descReduceSeq, descReducePar;
    Real timeEigScala_sta, timeEigScala_end;

    // Leading dimension provided
    descReduceSeq.Init( numKeep, numKeep, numKeep, numKeep, I_ZERO, I_ZERO, contxt, lda );

    // Automatically comptued Leading Dimension
    descReducePar.Init( numKeep, numKeep, scaBlockSize, scaBlockSize, I_ZERO, I_ZERO, contxt );

    square_mat_scala.SetDescriptor( descReducePar );

    eigvecs_scala.SetDescriptor( descReducePar );

    DblNumMat&  square_mat = hamTDDFT;

    // Redistribute the input matrix over the process grid
    SCALAPACK(pdgemr2d)(&numKeep, &numKeep, square_mat.Data(), &I_ONE, &I_ONE, 
        descReduceSeq.Values(), &square_mat_scala.LocalMatrix()[0],
        &I_ONE, &I_ONE, square_mat_scala.Desc().Values(), &contxt );

    // Make the ScaLAPACK call

    char uplo = 'U';
    std::vector<Real> temp_eigs(lda);
    scalapack::Syevd(uplo, square_mat_scala, temp_eigs, eigvecs_scala );

    for(Int copy_iter = 0; copy_iter < lda; copy_iter ++){
      eigValS[copy_iter] = temp_eigs[copy_iter];
    }

    GetTime(timeEnd);
    statusOFS << "Time for Syevd in scalapack      = " << timeEnd - timeSta << " [s]" << std::endl;
    //statusOFS << "eigValS = " << eigValS << std::endl;

    if(esdfParam.isOutputExcitationWfn) {

      SetValue(square_mat, 0.0 );
      SCALAPACK(pdgemr2d)( &numKeep, &numKeep, eigvecs_scala.Data(), &I_ONE, &I_ONE,
          square_mat_scala.Desc().Values(), square_mat.Data(), &I_ONE, 
          &I_ONE, descReduceSeq.Values(), &contxt );

      DblNumMat excitationWfn(ntotLocal, ncvband);                       //Save wavefunproduct Row
      SetValue(excitationWfn, 0.0);

      GetTime(timeSta);
      blas::Gemm('T', 'N', ntotLocal, ncvband, ncvband, 1.0, psiphiRow.Data(), ntotLocal, 
          square_mat.Data(), ncvband, 0.0, excitationWfn.Data(), ntotLocal);
      GetTime(timeEnd);
      statusOFS << "Time for excitation Wfn          = " << timeEnd - timeSta << " [s]" << std::endl;

    }

    if(contxt >= 0) {
      Cblacs_gridexit( contxt );
    }

    GetTime(timeEndTotal);
    statusOFS << std::endl << "Total time for LDA LRTDDFT       = " << timeEndTotal - timeStaTotal << " [s]" << std::endl;

    if(esdfParam.isOutputExcitationEnergy) {
      statusOFS << std::endl << "Output LRTDDFT excitation energy" <<  std::endl <<  std::endl; 
      for(Int i = 0; i < ncvband; i++){
        statusOFS << "excitation# = " << i << "      " << "eigValS = " << eigValS[i]*2 << " [Ry]" << std::endl; 
      }
    }

  } //end if(1)

  return;

} //CalculateLRTDDFT


void LRTDDFT::CalculateLRTDDFT_ISDF(
    Hamiltonian& ham,
    Spinor& psi,
    Fourier& fft,
    const Domain& dm) {

  //gobal init
  Real numMuFac = esdfParam.numMuFacLRTDDFTISDF;
  Real numGaussianRandomFac = esdfParam.numGaussianRandomFacLRTDDFTISDF;
  Real Tolerance = esdfParam.toleranceLRTDDFT;
  Int hybridDFKmeansMaxIter_ISDF = esdfParam.maxIterKmeansLRTDDFTISDF;
  std::string lrtddftISDFIPType = esdfParam.ipTypeLRTDDFTISDF;
  std::string lrtddftEigenSolver = esdfParam.eigenSolverLRTDDFT;
  Real hybridDFKmeansTolerance = esdfParam.toleranceKmeansLRTDDFTISDF;
  Int nvband = esdfParam.nvband;
  Int ncband = esdfParam.ncband;
  Int nkband = esdfParam.nkband;
  Int eigMaxIter_LRTDDFT = esdfParam.eigMaxIterLRTDDFT;
  Real eigMinTolerance_LRTDDFT = esdfParam.eigMinToleranceLRTDDFT;
  Real eigTolerance_LRTDDFT = esdfParam.eigToleranceLRTDDFT;

  Int ntot = ntot_;
  Int ntotR2C = ntotR2C_;
  Int ncom = numcomponent_;
  Int nocc = nocc_;
  Int numStateTotal = psi.NumStateTotal();
  Real vol = vol_;
  Int  ncvband = ncband * nvband;                 // Matrix size
  Int  ncvbandMu = IRound(std::sqrt(ncvband) * numMuFac);    // N_Mu

  //time init
  Real timeSta;
  Real timeEnd;
  Real timeStaTotal;
  Real timeEndTotal;
  Real timeFFTRtoC = 0.0;
  Real timeCalculateHartree = 0.0;
  Real timeCalculatefxc = 0.0;

  //Final matrix
  DblNumMat  hamTDDFT(ncvband, ncvband);                   // Save all matrix
  SetValue(hamTDDFT, 0.0);
  // DblNumVec  eigValS(ncvband);                             // Save energy eig
  // SetValue(eigValS, 0.0);

  //MPI init

  MPI_Barrier(domain_.comm);
  int mpirank;  MPI_Comm_rank(domain_.comm, &mpirank);
  int mpisize;  MPI_Comm_size(domain_.comm, &mpisize);

  //Part 1 get product state

  GetTime(timeStaTotal);

  //Step 1 Convert the column partition to row partition

  //MPI parameter 
  Int numStateBlocksize = numStateTotal / mpisize;
  Int numStateLocal = numStateBlocksize;
  if (mpirank < (numStateTotal % mpisize)) {
    numStateLocal = numStateBlocksize + 1;
  }

  Int ntotBlocksize = ntot / mpisize;
  Int ntotLocal = ntotBlocksize;
  if (mpirank < (ntot % mpisize)) {
    ntotLocal = ntotBlocksize + 1;
  }

  DblNumMat psiCol(ntot, numStateLocal);                         //Save wavefun Col
  SetValue(psiCol, 0.0);
  DblNumMat psiRow(ntotLocal, numStateTotal);                    //Save wavefun Row  
  SetValue(psiRow, 0.0);

  GetTime(timeSta);
  lapack::Lacpy('A', ntot, numStateLocal, psi.Wavefun().Data(), ntot, psiCol.Data(), ntot);
  GetTime(timeEnd);
  statusOFS << "Time for Lacpy                   = " << timeEnd - timeSta << " [s]" << std::endl;

  double norm = 0.0;
  for(Int i = 0; i < ntot; i++){
    norm += psiCol(i,0)*psiCol(i,0);
  }
  statusOFS << "Norm                   = " << norm  << std::endl;

  GetTime(timeSta);
  AlltoallForward(psiCol, psiRow, domain_.comm);
  GetTime(timeEnd);
  statusOFS << "Time for AlltoallForward         = " << timeEnd - timeSta << " [s]" << std::endl;

  // Convert the column partition to row partition end

  //Step 2 ISDF for product state

  //MPI parameter 

  Int numStateBlocksizeNcv = ncvband / mpisize;
  Int numStateLocalNcv = numStateBlocksizeNcv;
  if (mpirank < (ncvband % mpisize)) {
    numStateLocalNcv = numStateBlocksizeNcv + 1;
  }

  Int numStateBlocksizeNcvMu = ncvbandMu / mpisize;
  Int NcvMuLocal = numStateBlocksizeNcvMu;
  if (mpirank < (ncvbandMu % mpisize)) {
    NcvMuLocal = numStateBlocksizeNcvMu + 1;
  }

  // Perform ISDF

  DblNumMat psiRowLocal(ntotLocal, nvband);
  DblNumMat phiRowLocal(ntotLocal, ncband);  
  GetTime(timeSta);
  lapack::Lacpy('A', ntotLocal, nvband, psiRow.Data() + (nocc - nvband) * ntotLocal, ntotLocal, psiRowLocal.Data(), ntotLocal);
  lapack::Lacpy('A', ntotLocal, ncband, psiRow.Data() + nocc * ntotLocal, ntotLocal, phiRowLocal.Data(), ntotLocal);
  GetTime(timeEnd);
  statusOFS << "Time for Lacpy in ISDF           = " << timeEnd - timeSta << " [s]" << std::endl;
  Real timeStaisdf, timeEndisdf;
  GetTime(timeStaisdf);
  IntNumVec pivQR_(ntot);

  DblNumVec weight(ntot);

  IntNumVec  weightSizeDispls(mpisize);
  IntNumVec  weightSize(mpisize);

  if (lrtddftISDFIPType == "QRCP") {

    Int numPre = IRound(std::sqrt(ncvbandMu * numGaussianRandomFac));

    // Step 1: Pre-compression of the wavefunctions. This uses
    // multiplication with orthonormalized random Gaussian matrices

    DblNumMat G1(ncband, numPre);
    DblNumMat G2(nvband, numPre);
    GetTime(timeSta);
    if (mpirank == 0) {
      GaussianRandom(G1);
      GaussianRandom(G2);
      lapack::Orth(ncband, numPre, G1.Data(), ncband);
      lapack::Orth(nvband, numPre, G2.Data(), nvband);
      statusOFS << "Random projection initialzied!!!!!" << std::endl;
    }
    MPI_Bcast(G1.Data(), ncband * numPre, MPI_DOUBLE, 0, domain_.comm);
    MPI_Bcast(G2.Data(), nvband * numPre, MPI_DOUBLE, 0, domain_.comm);
    GetTime(timeEnd);
    statusOFS << "Time for random initialzied      = " << timeEnd - timeSta << " [s]" << std::endl;
    DblNumMat localphiGRow(ntotLocal, numPre);
    DblNumMat localpsiGRow(ntotLocal, numPre);
    GetTime(timeSta);
    blas::Gemm('N', 'N', ntotLocal, numPre, ncband, 1.0, phiRowLocal.Data(), ntotLocal, G1.Data(), ncband, 0.0, localphiGRow.Data(), ntotLocal);
    blas::Gemm('N', 'N', ntotLocal, numPre, nvband, 1.0, psiRowLocal.Data(), ntotLocal, G2.Data(), nvband, 0.0, localpsiGRow.Data(), ntotLocal);
    Int numPreSquare = numPre * numPre;

    DblNumMat MGCol(ntotLocal, numPreSquare);
    DblNumVec weightLocal(ntotLocal);
    SetValue(weightLocal, 0.0);

#ifdef _USE_OPENMP_
#pragma omp parallel 
    {
#endif

#ifdef _USE_OPENMP_
#pragma omp for schedule(dynamic,1)
#endif
      for (int j = 0; j < numPre; j++) {
        for (int i = 0; i < numPre; i++) {
          for (int kk = 0; kk < ntotLocal; kk++) {
            MGCol(kk, i + j * numPre) = localphiGRow(kk, i) * localpsiGRow(kk, j);
            // MGNormLocal(kk) += MGCol(i + j * numPre, kk) * MGCol(i + j * numPre, kk);
          }
        }
      }

#ifdef _USE_OPENMP_
#pragma omp for schedule(dynamic,1)
#endif
      for (int i = 0; i < numPreSquare; i++) {
        for (int j = 0; j < ntotLocal; j++) {
          weightLocal(j) += MGCol(j, i);
        }
      }

#ifdef _USE_OPENMP_
    }
#endif


//#ifdef _USE_OPENMP_
//#pragma omp parallel 
    {
//#endif

      if ((ntot % mpisize) == 0) {
//#ifdef _USE_OPENMP_
//#pragma omp for schedule(dynamic,1)
//#endif
        for (Int i = 0; i < mpisize; i++){
          weightSizeDispls[i] = i * ntotBlocksize;
          weightSize[i] = ntotBlocksize;
        }

      }
      else{
//#ifdef _USE_OPENMP_
//#pragma omp for schedule(dynamic,1)
//#endif
        for (Int i = 0; i < mpisize; i++){
          if (i < (ntot % mpisize)) {
            weightSizeDispls[i] = i * (ntotBlocksize + 1);
            weightSize[i] = ntotBlocksize + 1;
          }
          else{
            weightSizeDispls[i] = (ntot%mpisize) * (ntotBlocksize + 1) + (i-(ntot%mpisize)) * (ntotBlocksize);
            weightSize[i] = ntotBlocksize;
          }
        }
      }

//#ifdef _USE_OPENMP_
    }
//#endif

    MPI_Allgatherv(weightLocal.Data(), ntotLocal, MPI_DOUBLE, weight.Data(), weightSize.Data(), weightSizeDispls.Data(), MPI_DOUBLE, domain_.comm);
    // MPI_Allgather(weightLocal.Data(), ntotLocal, MPI_DOUBLE, weight.Data(), ntotLocal, MPI_DOUBLE, domain_.comm);

    GetTime(timeEnd);
    statusOFS << "Time for generating MG matrix    = " << timeEnd - timeSta << " [s]" << std::endl;

    // Got MG matrix, begin perform QRCP
    //DblNumVec tau(ntotLocalMG); //Not being used in this module
    //SetValue(tau, 0.0);

    IntNumVec pivQR_(ntot);

    GetTime(timeSta);
    //lapack::QRCP(numPre * numPre, ntotLocalMG, MG.Data(), numPre * numPre, pivQR_.Data(), tau.Data());
    GetTime(timeEnd);
    statusOFS << "Time for pivQR_ with QRCP        = " << timeEnd - timeSta << " [s]" << std::endl;
  }

  if (lrtddftISDFIPType == "Kmeans"){
    GetTime(timeSta);
    // DblNumVec weight(ntotLocalMG);
    // SetValue(weight, 0.0);
    // for (int i = 0; i < ntotLocalMG; i++) {
    //   for (int j = 0; j < numPre * numPre; j++) {
    //     weight[i] +=  MG(j, i) * MG(j, i);
    //   }
    // }

    DblNumVec weightLocal(ntotLocal);
    SetValue(weightLocal, 0.0);

#ifdef _USE_OPENMP_
#pragma omp parallel 
    {
#endif

#ifdef _USE_OPENMP_
#pragma omp for schedule(dynamic,1)
#endif
      for (int j = 0; j < nvband; j++) {
        for (int i = 0; i < ncband; i++) {
          for (int kk = 0; kk < ntotLocal; kk++) {
            weightLocal(kk) += pow(phiRowLocal(kk, i) * psiRowLocal(kk, j),2);
            // MGNormLocal(kk) += MGCol(i + j * numPre, kk) * MGCol(i + j * numPre, kk);
          }
        }
      }

#ifdef _USE_OPENMP_
    }
#endif

//#ifdef _USE_OPENMP_
//#pragma omp parallel 
    {
//#endif

      if ((ntot % mpisize) == 0) {
//#ifdef _USE_OPENMP_
//#pragma omp for schedule(dynamic,1)
//#endif
        for (Int i = 0; i < mpisize; i++){
          weightSizeDispls[i] = i * ntotBlocksize;
          weightSize[i] = ntotBlocksize;
        }

      }
      else{
//#ifdef _USE_OPENMP_
//#pragma omp for schedule(dynamic,1)
//#endif
        for (Int i = 0; i < mpisize; i++){
          if (i < (ntot % mpisize)) {
            weightSizeDispls[i] = i * (ntotBlocksize + 1);
            weightSize[i] = ntotBlocksize + 1;
          }
          else{
            weightSizeDispls[i] = (ntot%mpisize) * (ntotBlocksize + 1) + (i-(ntot%mpisize)) * (ntotBlocksize);
            weightSize[i] = ntotBlocksize;
          }
        }
      }

//#ifdef _USE_OPENMP_
    }
//#endif

    MPI_Allgatherv(weightLocal.Data(), ntotLocal, MPI_DOUBLE, weight.Data(), weightSize.Data(), weightSizeDispls.Data(), MPI_DOUBLE, domain_.comm);

    SetValue( pivQR_, 0 ); // Important. Otherwise QRCP uses piv as initial guess
    int rk = ncvbandMu;
    KMEAN(ntot, weight, rk, hybridDFKmeansTolerance, hybridDFKmeansMaxIter_ISDF, Tolerance, domain_, pivQR_.Data());
    GetTime(timeEnd);
    statusOFS << "Time for pivQR_ with Kmeans      = " << timeEnd - timeSta << " [s]" << std::endl;
  }


  // Real *p = MGCol.Data();
  // if (p) {
  //   delete p;
  //   p = NULL;
  // }
  
  IntNumVec ntotresTotal(mpisize);
  SetValue(ntotresTotal, 0);

  if ((ntot % mpisize) == 0) {
    for (int i = 0; i < mpisize; i++) {
      ntotresTotal(i) =  i * ntotBlocksize;
     }
  }
  else{
    for (Int i = 0; i < mpisize; i++){
      if (i < (ntot % mpisize)) {
        ntotresTotal(i) =  i * (ntotBlocksize+1);
      }
      else{
        ntotresTotal(i) = (ntot%mpisize) * (ntotBlocksize + 1) + (i-(ntot%mpisize)) * (ntotBlocksize);
      }
    }
  }

  Int selectedRow = 0;
  std::vector<Int> localRowIdx;
  GetTime(timeSta);

//#ifdef _USE_OPENMP_
//#pragma omp parallel
  {
//#endif

//#ifdef _USE_OPENMP_
//#pragma omp for schedule(dynamic,1)
//#endif
    for (int i = 0; i < ncvbandMu; i++) {
      if (mpirank < mpisize-1){
        if (pivQR_(i) >= ntotresTotal(mpirank) && pivQR_(i) < ntotresTotal(mpirank+1)) {
          selectedRow++;
          localRowIdx.push_back(pivQR_(i) - ntotresTotal(mpirank));
        }
      }
      else{ 
        if (pivQR_(i) >= ntotresTotal(mpirank)) {
          selectedRow++;
          localRowIdx.push_back(pivQR_(i) - ntotresTotal(mpirank));
        }
      }
    }

//#ifdef _USE_OPENMP_
  } //#pragma omp parallel
//#endif

  DblNumMat psiMuLocal(selectedRow, nvband);
  DblNumMat phiMuLocal(selectedRow, ncband);

//#ifdef _USE_OPENMP_
//#pragma omp parallel
  {
//#endif

//#ifdef _USE_OPENMP_
//#pragma omp for schedule(dynamic,1)
//#endif
    for (int i = 0; i < nvband; i++) {
      for (int j = 0; j < selectedRow; j++) {
        psiMuLocal(j, i) = psiRowLocal(localRowIdx[j], i);
      }
    }

//#ifdef _USE_OPENMP_
//#pragma omp for schedule(dynamic,1)
//#endif
    for (int i = 0; i < ncband; i++) {
      for (int j = 0; j < selectedRow; j++) {
        phiMuLocal(j, i) = phiRowLocal(localRowIdx[j], i);
      }
    } 

//#ifdef _USE_OPENMP_
  } //#pragma omp parallel
//#endif
  GetTime(timeEnd);
  statusOFS << "Time for psiphiMuLocal           = " << timeEnd - timeSta << " [s]" << std::endl;

  DblNumMat psiMu(ncvbandMu, nvband);
  DblNumMat phiMu(ncvbandMu, ncband);

  IntNumVec  localSizeVec(mpisize);
  IntNumVec  localnvbandSizeVec(mpisize);
  IntNumVec  localncbandSizeVec(mpisize);

  IntNumVec  localSizeDispls(mpisize);
  IntNumVec  localnvbandSizeDispls(mpisize);
  IntNumVec  localncbandSizeDispls(mpisize);
  GetTime(timeSta);

  MPI_Allgather(&selectedRow, 1, MPI_INT, localSizeVec.Data(), 1, MPI_INT, domain_.comm);

  localSizeDispls[0] = 0;
//#ifdef _USE_OPENMP_
//#pragma omp parallel 
  {
//#endif
//#ifdef _USE_OPENMP_
//#pragma omp for schedule (dynamic,1)
//#endif
    for(Int i = 1; i < mpisize; i++) {
      localSizeDispls[i] = localSizeDispls[i-1] + localSizeVec[i-1];
    }
//#ifdef _USE_OPENMP_
//#pragma omp for schedule (dynamic,1) 
//#endif
    for(Int i = 0; i < mpisize; i++) {
      localnvbandSizeVec[i] = localSizeVec[i] * nvband;
      localncbandSizeVec[i] = localSizeVec[i] * ncband;
      localnvbandSizeDispls[i] = localSizeDispls[i] * nvband;
      localncbandSizeDispls[i] = localSizeDispls[i] * ncband;
    }
//#ifdef _USE_OPENMP_
  }
//#endif
  MPI_Allgatherv(psiMuLocal.Data(), selectedRow * nvband, MPI_DOUBLE, psiMu.Data(), localnvbandSizeVec.Data(), localnvbandSizeDispls.Data(), MPI_DOUBLE, domain_.comm);
  MPI_Allgatherv(phiMuLocal.Data(), selectedRow * ncband, MPI_DOUBLE, phiMu.Data(), localncbandSizeVec.Data(), localncbandSizeDispls.Data(), MPI_DOUBLE, domain_.comm);
  GetTime(timeEnd);
  statusOFS << "Time for Allgather ncv           = " << timeEnd - timeSta << " [s]" << std::endl;

  GetTime(timeSta);
  DblNumMat psiphiMu(ncvbandMu, ncvband); //coefficient matrix C
#ifdef _USE_OPENMP_
#pragma omp parallel 
  {
#endif
#ifdef _USE_OPENMP_
#pragma omp for schedule (dynamic,1) 
#endif
    for (int i = 0; i < nvband; i++) {
      for (int j = 0; j < ncband; j++) {
        for (int mu = 0; mu < ncvbandMu; mu++) {
          psiphiMu(mu, j + i * ncband) = psiMu(mu, i) * phiMu(mu, j);
        }
      }
    }
#ifdef _USE_OPENMP_
  }
#endif
  GetTime(timeEnd);
  statusOFS << "Time for generating coefficient  = " << timeEnd - timeSta << " [s]" << std::endl;

  DblNumMat PMuNu(ncvbandMu,ncvbandMu);      // save C*C^T temp
  SetValue(PMuNu,0.0);
  GetTime(timeSta);
  blas::Gemm('N', 'T', ncvbandMu, ncvbandMu, ncvband, 1.0, psiphiMu.Data(), ncvbandMu, psiphiMu.Data(), ncvbandMu, 0.0, PMuNu.Data(), ncvbandMu);
  GetTime(timeEnd);
  statusOFS << "Time for C*C^T                   = " << timeEnd - timeSta << " [s]" << std::endl;

  GetTime(timeSta);
  DblNumMat Ppsimu(ntotLocal, ncvbandMu);
  blas::Gemm('N', 'T', ntotLocal, ncvbandMu, nvband, 1.0, psiRowLocal.Data(), ntotLocal, psiMu.Data(), ncvbandMu, 0.0, Ppsimu.Data(), ntotLocal);
  DblNumMat Pphimu(ntotLocal, ncvbandMu);
  blas::Gemm('N', 'T', ntotLocal, ncvbandMu, ncband, 1.0, phiRowLocal.Data(), ntotLocal, phiMu.Data(), ncvbandMu, 0.0, Pphimu.Data(), ntotLocal);
  GetTime(timeEnd);
  statusOFS << "Time for Pphimu and Pphimu       = " << timeEnd - timeSta << " [s]" << std::endl;

  DblNumMat psiphizetaRow(ntotLocal, ncvbandMu);     // save Z*C^T
  GetTime(timeSta);
#ifdef _USE_OPENMP_
#pragma omp parallel 
  {
#endif
#ifdef _USE_OPENMP_
#pragma omp for schedule (dynamic,1) nowait
#endif
    for (int i = 0; i < ncvbandMu; i++) {
      for (int j = 0; j < ntotLocal; j++) {
        psiphizetaRow(j, i) = Ppsimu(j, i) * Pphimu(j, i);
      }
    }
#ifdef _USE_OPENMP_
  }
#endif
  GetTime(timeEnd);
  statusOFS << "Time for Z*C^T                   = " << timeEnd - timeSta << " [s]" << std::endl;

  GetTime(timeSta);

  lapack::Potrf('L', ncvbandMu, PMuNu.Data(), ncvbandMu);
  blas::Trsm('R', 'L', 'T', 'N', ntotLocal, ncvbandMu, 1.0, PMuNu.Data(), ncvbandMu, psiphizetaRow.Data(), ntotLocal);
  blas::Trsm('R', 'L', 'N', 'N', ntotLocal, ncvbandMu, 1.0, PMuNu.Data(), ncvbandMu, psiphizetaRow.Data(), ntotLocal);
  GetTime(timeEnd);
  statusOFS << "Time for Trsm                    = " << timeEnd - timeSta << " [s]" << std::endl;
  GetTime(timeEndisdf);
  statusOFS << "Time for ISDF in LRTDDFT         = " << timeEndisdf - timeStaisdf << " [s]" << std::endl;
  // GetTime(timeEnd);
  // Compute the interpolation matrix via the density matrix formulation  
  // GetTime(timeSta);


  // ISDF for product state end

  // Step 3 Convert the row partition to column partition

  DblNumMat psiphizetaCol(ntot, NcvMuLocal);                   //Save wavefunproduct zeta function Col
  SetValue(psiphizetaCol, 0.0);
  GetTime(timeSta);
  AlltoallBackward(psiphizetaRow, psiphizetaCol, domain_.comm);
  GetTime(timeEnd);
  statusOFS << "Time for AlltoallBackward        = " << timeEnd - timeSta << " [s]" << std::endl;

  // Convert the row partition to column partition end

  // Part 2 get Hartree potential wavfunction

  // Fourier transform get Hartree potential and 
  // sort wavefunction in real space in the end 

  DblNumMat tempRow(ntotLocal, ncvbandMu);                         //Save Hartree-Fock wave Row
  SetValue(tempRow, 0.0);

  DblNumMat tempCol(ntot, NcvMuLocal);                     //Save Hartree-Fock wave Col
  SetValue(tempCol, 0.0);

  GetTime(timeSta);
  FFTRtoC(fft, ham, psiphizetaCol, tempCol, NcvMuLocal);
  GetTime(timeEnd);
  statusOFS << "Time for FFTRtoC                 = " << timeEnd - timeSta << " [s]" << std::endl;
  timeFFTRtoC = timeFFTRtoC + (timeEnd - timeSta);

  // Fourier transform end

  // Convert the column partition to row partition

  GetTime(timeSta);
  AlltoallForward(tempCol, tempRow, domain_.comm);
  GetTime(timeEnd);
  statusOFS << "Time for AlltoallForward         = " << timeEnd - timeSta << " [s]" << std::endl;

  // Convert the column partition to row partition end

  // Part 3 get fxc potential wavfunction

  //get fxc potential and sort wavefunction in real space in the end 

  //fxc setup

  DblNumMat     psiphifxcR2C(ntotLocal, ncvbandMu);                       //Save fxc wavefun
  SetValue(psiphifxcR2C, 0.0);

  //Calculatefxc operator

  DblNumVec     fxcPz(ntot);                                             //Save fxc operator
  SetValue(fxcPz, 0.0);

  GetTime(timeSta);
  Calculatefxc(fft, fxcPz);
  GetTime(timeEnd);
  statusOFS << "Time for Calculatefxc            = " << timeEnd - timeSta << " [s]" << std::endl;

  //fxcPz send to every MPI 

  GetTime(timeSta);
  MPI_Bcast(fxcPz.Data(), ntot, MPI_DOUBLE, 0, domain_.comm);
  GetTime(timeEnd);
  statusOFS << "Time for MPI_Bcast               = " << timeEnd - timeSta << " [s]" << std::endl;

  //Calculate fxc wavefun;

//#ifdef _USE_OPENMP_
//#pragma omp parallel
  {
//#endif

    double facfxc = 2.0 * vol_ / double(ntot_) ;

    GetTime(timeSta);

#ifdef _USE_OPENMP_
#pragma omp for schedule(dynamic,1)
#endif

    for (Int mu = 0; mu < ncvbandMu; mu++) {
      for (Int i = 0; i < ntotLocal; i++) {
        psiphifxcR2C(i, mu) = psiphizetaRow(i, mu) * fxcPz(i + mpirank * ntotLocal) * facfxc + tempRow(i, mu);
      }
    }//for mu  

#ifdef _USE_OPENMP_
  } //#pragma omp parallel
#endif

  GetTime(timeEnd);
  statusOFS << "Time for psiphifxcR2C            = " << timeEnd - timeSta << " [s]" << std::endl;

  if(lrtddftEigenSolver == "LAPACK"){

    //Calculate Hatree-fxc product;
  
    DblNumMat     fxcLocaltemp1(ncvbandMu, ncvbandMu);
    SetValue(fxcLocaltemp1, 0.0);
    DblNumMat     fxcLocaltemp2(ncvbandMu, ncvband);
    SetValue(fxcLocaltemp2, 0.0);
    DblNumMat     fxcLocal(ncvband, ncvband);
    SetValue(fxcLocal, 0.0);
    GetTime(timeSta);
  
    //wanly allreduce
  
    blas::Gemm('T', 'N', ncvbandMu, ncvbandMu, ntotLocal, 1.0, psiphizetaRow.Data(), ntotLocal, psiphifxcR2C.Data(), ntotLocal, 0.0, fxcLocaltemp1.Data(), ncvbandMu);
  
    blas::Gemm('N', 'N', ncvbandMu, ncvband, ncvbandMu, 1.0, fxcLocaltemp1.Data(), ncvbandMu, psiphiMu.Data(), ncvbandMu, 0.0, fxcLocaltemp2.Data(), ncvbandMu);
  
    blas::Gemm('T', 'N', ncvband, ncvband, ncvbandMu, 1.0, psiphiMu.Data(), ncvbandMu, fxcLocaltemp2.Data(), ncvbandMu, 0.0, fxcLocal.Data(), ncvband);
  
  #if ( _DEBUGlevel_ >= 1 )
    statusOFS << "psiphizetaRow     = " << psiphizetaRow << std::endl;
  
    statusOFS << "psiphifxcR2C      = " << psiphifxcR2C << std::endl;
  
    statusOFS << "fxcLocaltemp1     = " << fxcLocaltemp1 << std::endl;
  
    statusOFS << "psiphiMu          = " << psiphiMu << std::endl;
  
    statusOFS << "fxcLocaltemp2     = " << fxcLocaltemp2 << std::endl;
  
    statusOFS << "fxcLocal          = " << fxcLocal << std::endl;
  #endif
  
    GetTime(timeEnd);
    statusOFS << "Time for Hatree-fxc product      = " << timeEnd - timeSta << " [s]" << std::endl;
  
    DblNumMat     fxc(ncvband, ncvband);                                   //Save Hfxc matrix
    SetValue(fxc, 0.0);
    GetTime(timeSta);
    MPI_Allreduce(fxcLocal.Data(), fxc.Data(), ncvband * ncvband, MPI_DOUBLE, MPI_SUM, domain_.comm);
    GetTime(timeEnd);
    statusOFS << "Time for MPI_Allreduce           = " << timeEnd - timeSta << " [s]" << std::endl;
  
  #if ( _DEBUGlevel_ >= 1 )
    if (mpirank == 0) {
      printf("fxc %d of %d\n", mpirank, mpisize);
      for (Int i = 0; i < ncvband; i++) {
        for (Int j = 0; j < ncvband; j++) {
          printf("%f\t", fxc(i, j));
        }
        printf("\n");
      }
    }
  #endif
    //Diagonal part setup
  
    Int kk = 0;
  
    GetTime(timeSta);
  #ifdef _USE_OPENMP_
  #pragma omp parallel 
    {
  #endif
  #ifdef _USE_OPENMP_
  #pragma omp for schedule (dynamic,1)
  #endif
      for (Int k = nocc_ - nvband; k < nocc_; k++) {
        for (Int j = nocc_; j < nocc_ + ncband; j++) {
          fxc(kk, kk) += eigVal_(j) - eigVal_(k);
          kk++;
        }//for j
      }//for k
  
  #ifdef _USE_OPENMP_
    }
  #endif
    GetTime(timeEnd);
    statusOFS << "Time for Energydiff              = " << timeEnd - timeSta << " [s]" << std::endl;
  
    GetTime(timeSta);
  
    lapack::Lacpy('A', ncvband, ncvband, fxc.Data(), ncvband, hamTDDFT.Data(), ncvband);
    //p = fxc.Data();
    //if (p) {
    //  delete p;
    //  p = NULL;
    //}
    GetTime(timeEnd);
  
    statusOFS << "Time for hamTDDFT                = " << timeEnd - timeSta << " [s]" << std::endl;

    DblNumVec  eigValS(ncvband);                           //Save energy eig
    SetValue(eigValS, 0.0);	
    GetTime(timeSta);
    lapack::Syevd('V', 'U', ncvband, hamTDDFT.Data(), ncvband, eigValS.Data());
    GetTime(timeEnd);
    statusOFS << "Time for Syevd in lapack         = " << timeEnd - timeSta << " [s]" << std::endl;

    //statusOFS << "eigValS = " << eigValS << std::endl;
    if(esdfParam.isOutputExcitationEnergy) {
      statusOFS << std::endl << "Output LRTDDFT excitation energy" <<  std::endl <<  std::endl;
      for(Int i = 0; i < ncvband; i++){
        statusOFS << "excitation# = " << i << "      " << "eigValS = " << eigValS[i]*2 << " [Ry]" << std::endl;
      }
    }

  } //end if(lrtddftEigenSolver == "LAPACK")


  if(lrtddftEigenSolver == "ScaLAPACK") { // Parallel Syevd

    //Calculate Hatree-fxc product;
  
    DblNumMat     fxcLocaltemp1(ncvbandMu, ncvbandMu);
    SetValue(fxcLocaltemp1, 0.0);
    DblNumMat     fxcLocaltemp2(ncvbandMu, ncvband);
    SetValue(fxcLocaltemp2, 0.0);
    DblNumMat     fxcLocal(ncvband, ncvband);
    SetValue(fxcLocal, 0.0);
    GetTime(timeSta);
  
    //wanly allreduce
  
    blas::Gemm('T', 'N', ncvbandMu, ncvbandMu, ntotLocal, 1.0, psiphizetaRow.Data(), ntotLocal, psiphifxcR2C.Data(), ntotLocal, 0.0, fxcLocaltemp1.Data(), ncvbandMu);
  
    blas::Gemm('N', 'N', ncvbandMu, ncvband, ncvbandMu, 1.0, fxcLocaltemp1.Data(), ncvbandMu, psiphiMu.Data(), ncvbandMu, 0.0, fxcLocaltemp2.Data(), ncvbandMu);
  
    blas::Gemm('T', 'N', ncvband, ncvband, ncvbandMu, 1.0, psiphiMu.Data(), ncvbandMu, fxcLocaltemp2.Data(), ncvbandMu, 0.0, fxcLocal.Data(), ncvband);
  
  #if ( _DEBUGlevel_ >= 1 )
    statusOFS << "psiphizetaRow     = " << psiphizetaRow << std::endl;
  
    statusOFS << "psiphifxcR2C      = " << psiphifxcR2C << std::endl;
  
    statusOFS << "fxcLocaltemp1     = " << fxcLocaltemp1 << std::endl;
  
    statusOFS << "psiphiMu          = " << psiphiMu << std::endl;
  
    statusOFS << "fxcLocaltemp2     = " << fxcLocaltemp2 << std::endl;
  
    statusOFS << "fxcLocal          = " << fxcLocal << std::endl;
  #endif
  
    GetTime(timeEnd);
    statusOFS << "Time for Hatree-fxc product      = " << timeEnd - timeSta << " [s]" << std::endl;
  
    DblNumMat     fxc(ncvband, ncvband);                                   //Save Hfxc matrix
    SetValue(fxc, 0.0);
    GetTime(timeSta);
    MPI_Allreduce(fxcLocal.Data(), fxc.Data(), ncvband * ncvband, MPI_DOUBLE, MPI_SUM, domain_.comm);
    GetTime(timeEnd);
    statusOFS << "Time for MPI_Allreduce           = " << timeEnd - timeSta << " [s]" << std::endl;
  
  #if ( _DEBUGlevel_ >= 1 )
    if (mpirank == 0) {
      printf("fxc %d of %d\n", mpirank, mpisize);
      for (Int i = 0; i < ncvband; i++) {
        for (Int j = 0; j < ncvband; j++) {
          printf("%f\t", fxc(i, j));
        }
        printf("\n");
      }
    }
  #endif
    //Diagonal part setup
  
    Int kk = 0;
  
    GetTime(timeSta);
  #ifdef _USE_OPENMP_
  #pragma omp parallel 
    {
  #endif
  #ifdef _USE_OPENMP_
  #pragma omp for schedule (dynamic,1)
  #endif
      for (Int k = nocc_ - nvband; k < nocc_; k++) {
        for (Int j = nocc_; j < nocc_ + ncband; j++) {
          fxc(kk, kk) += eigVal_(j) - eigVal_(k);
          kk++;
        }//for j
      }//for k
  
  #ifdef _USE_OPENMP_
    }
  #endif
    GetTime(timeEnd);
    statusOFS << "Time for Energydiff              = " << timeEnd - timeSta << " [s]" << std::endl;
  
    GetTime(timeSta);
  
    lapack::Lacpy('A', ncvband, ncvband, fxc.Data(), ncvband, hamTDDFT.Data(), ncvband);
    //p = fxc.Data();
    //if (p) {
    //  delete p;
    //  p = NULL;
    //}
    GetTime(timeEnd);
 
    statusOFS << "Time for hamTDDFT                = " << timeEnd - timeSta << " [s]" << std::endl;

    GetTime(timeSta);

    // Setup BLACS
    Int contxt;
    Int nprow, npcol, myrow, mycol, info;
    Int scaBlockSize = esdfParam.scaBlockSize;
    Int numProcScaLAPACK = esdfParam.numProcScaLAPACKPW;

    for (Int i = IRound(sqrt(double(numProcScaLAPACK))); i <= numProcScaLAPACK; i++) {
      nprow = i;
      npcol = int(numProcScaLAPACK / nprow);
      if (nprow * npcol == numProcScaLAPACK) break;
    }

    Cblacs_get(0, 0, &contxt);
    Cblacs_gridinit(&contxt, "C", nprow, npcol);
    Cblacs_gridinfo(contxt, &nprow, &npcol, &myrow, &mycol);

    //IntNumVec pmap(numProcScaLAPACK);
    //for ( Int i = 0; i < numProcScaLAPACK_; i++ ){
    //  pmap[i] = i;
    //}
    //Cblacs_get(0, 0, &contxt_);
    //Cblacs_gridmap(&contxt_, &pmap[0], nprow_, nprow_, npcol_);

    Int numKeep = ncvband;
    Int lda = ncvband;
    scalapack::ScaLAPACKMatrix<Real> square_mat_scala;
    scalapack::ScaLAPACKMatrix<Real> eigvecs_scala;
    scalapack::Descriptor descReduceSeq, descReducePar;
    Real timeEigScala_sta, timeEigScala_end;
    DblNumVec  eigValS(lda);
    SetValue(eigValS, 0.0);

    // Leading dimension provided
    descReduceSeq.Init(numKeep, numKeep, numKeep, numKeep, I_ZERO, I_ZERO, contxt, lda);

    // Automatically comptued Leading Dimension
    descReducePar.Init(numKeep, numKeep, scaBlockSize, scaBlockSize, I_ZERO, I_ZERO, contxt);

    square_mat_scala.SetDescriptor(descReducePar);

    eigvecs_scala.SetDescriptor(descReducePar);

    DblNumMat& square_mat = hamTDDFT;

    // Redistribute the input matrix over the process grid
    SCALAPACK(pdgemr2d)(&numKeep, &numKeep, square_mat.Data(), &I_ONE, &I_ONE,
        descReduceSeq.Values(), &square_mat_scala.LocalMatrix()[0],
        &I_ONE, &I_ONE, square_mat_scala.Desc().Values(), &contxt);

    // Make the ScaLAPACK call

    char uplo = 'U';
    std::vector<Real> temp_eigs(lda);
    scalapack::Syevd(uplo, square_mat_scala, temp_eigs, eigvecs_scala);

    for (Int copy_iter = 0; copy_iter < lda; copy_iter++) {
      eigValS[copy_iter] = temp_eigs[copy_iter];
    }

    if (esdfParam.isOutputExcitationWfn) {

      SetValue(square_mat, 0.0);
      SCALAPACK(pdgemr2d)(&numKeep, &numKeep, eigvecs_scala.Data(), &I_ONE, &I_ONE,
          square_mat_scala.Desc().Values(), square_mat.Data(), &I_ONE,
          &I_ONE, descReduceSeq.Values(), &contxt);

      DblNumMat excitationWfn(ntotLocal, ncvband);                       //Save wavefunproduct Row
      SetValue(excitationWfn, 0.0);

      GetTime(timeSta);

    }

    if(contxt >= 0) {
      Cblacs_gridexit( contxt );
    }

    GetTime(timeEndTotal);
    statusOFS << std::endl << "Total time for LDA LRTDDFT       = " << timeEndTotal - timeStaTotal << " [s]" << std::endl;

    if(esdfParam.isOutputExcitationEnergy) {
      statusOFS << std::endl << "Output LRTDDFT excitation energy" <<  std::endl <<  std::endl; 
      for(Int i = 0; i < ncvband; i++){
        statusOFS << "excitation# = " << i << "      " << "eigValS = " << eigValS[i]*2 << " [Ry]" << std::endl; 
      }
    }
  }// end if(lrtddftEigenSolver == "ScaLAPACK")


  if (lrtddftEigenSolver == "LOBPCG") {

if(0){
    //Calculate Hatree-fxc product;
  
    DblNumMat     fxcLocaltemp1(ncvbandMu, ncvbandMu);
    SetValue(fxcLocaltemp1, 0.0);
    DblNumMat     fxcLocaltemp2(ncvbandMu, ncvband);
    SetValue(fxcLocaltemp2, 0.0);
    DblNumMat     fxcLocal(ncvband, ncvband);
    SetValue(fxcLocal, 0.0);
    GetTime(timeSta);
  
    //wanly allreduce
  
    blas::Gemm('T', 'N', ncvbandMu, ncvbandMu, ntotLocal, 1.0, psiphizetaRow.Data(), ntotLocal, psiphifxcR2C.Data(), ntotLocal, 0.0, fxcLocaltemp1.Data(), ncvbandMu);
  
    blas::Gemm('N', 'N', ncvbandMu, ncvband, ncvbandMu, 1.0, fxcLocaltemp1.Data(), ncvbandMu, psiphiMu.Data(), ncvbandMu, 0.0, fxcLocaltemp2.Data(), ncvbandMu);
  
    blas::Gemm('T', 'N', ncvband, ncvband, ncvbandMu, 1.0, psiphiMu.Data(), ncvbandMu, fxcLocaltemp2.Data(), ncvbandMu, 0.0, fxcLocal.Data(), ncvband);
  
  #if ( _DEBUGlevel_ >= 1 )
    statusOFS << "psiphizetaRow     = " << psiphizetaRow << std::endl;
  
    statusOFS << "psiphifxcR2C      = " << psiphifxcR2C << std::endl;
  
    statusOFS << "fxcLocaltemp1     = " << fxcLocaltemp1 << std::endl;
  
    statusOFS << "psiphiMu          = " << psiphiMu << std::endl;
  
    statusOFS << "fxcLocaltemp2     = " << fxcLocaltemp2 << std::endl;
  
    statusOFS << "fxcLocal          = " << fxcLocal << std::endl;
  #endif
  
    GetTime(timeEnd);
    statusOFS << "Time for Hatree-fxc product      = " << timeEnd - timeSta << " [s]" << std::endl;
  
    DblNumMat     fxc(ncvband, ncvband);                                   //Save Hfxc matrix
    SetValue(fxc, 0.0);
    GetTime(timeSta);
    MPI_Allreduce(fxcLocal.Data(), fxc.Data(), ncvband * ncvband, MPI_DOUBLE, MPI_SUM, domain_.comm);
    GetTime(timeEnd);
    statusOFS << "Time for MPI_Allreduce           = " << timeEnd - timeSta << " [s]" << std::endl;
  
  #if ( _DEBUGlevel_ >= 1 )
    if (mpirank == 0) {
      printf("fxc %d of %d\n", mpirank, mpisize);
      for (Int i = 0; i < ncvband; i++) {
        for (Int j = 0; j < ncvband; j++) {
          printf("%f\t", fxc(i, j));
        }
        printf("\n");
      }
    }
  #endif
    //Diagonal part setup
  
    Int kk = 0;
  
    GetTime(timeSta);
  #ifdef _USE_OPENMP_
  #pragma omp parallel 
    {
  #endif
  #ifdef _USE_OPENMP_
  #pragma omp for schedule (dynamic,1)
  #endif
      for (Int k = nocc_ - nvband; k < nocc_; k++) {
        for (Int j = nocc_; j < nocc_ + ncband; j++) {
          fxc(kk, kk) += eigVal_(j) - eigVal_(k);
          kk++;
        }//for j
      }//for k
  
  #ifdef _USE_OPENMP_
    }
  #endif
    GetTime(timeEnd);
    statusOFS << "Time for Energydiff              = " << timeEnd - timeSta << " [s]" << std::endl;
  
    GetTime(timeSta);
  
    lapack::Lacpy('A', ncvband, ncvband, fxc.Data(), ncvband, hamTDDFT.Data(), ncvband);
    //p = fxc.Data();
    //if (p) {
    //  delete p;
    //  p = NULL;
    //}
    GetTime(timeEnd);
  
    statusOFS << "Time for hamTDDFT                = " << timeEnd - timeSta << " [s]" << std::endl;
}

    Real timeStaLOBPCG, timeEndLOBPCG;

    MPI_Comm mpi_comm = domain_.comm;

    Int useLessProcessInLOBPCG  = mpisize - esdfParam.numProcEigenSolverLRTDDFT; // useLessProcessInLOBPCG == 0 if use all process

    if (useLessProcessInLOBPCG != 0){ 
      Int LOBPCGcolor;
      Int numProcessInLOBPCG = esdfParam.numProcEigenSolverLRTDDFT;
      if (mpirank < numProcessInLOBPCG)
        LOBPCGcolor = 0;
      else
        LOBPCGcolor = MPI_UNDEFINED;
      MPI_Comm LOBPCGcomm;
      MPI_Comm_split(domain_.comm, LOBPCGcolor, 0, &LOBPCGcomm);
      Int mpisizeLOBPCG, mpirankLOBPCG;
      MPI_Comm_size(LOBPCGcomm, &mpisizeLOBPCG);
      MPI_Comm_rank(LOBPCGcomm, &mpirankLOBPCG);
      statusOFS << "mpisize of LOBPCG in LRTDDFT     = " << mpisizeLOBPCG << std::endl;
      //MPI_Comm mpi_comm = domain_.comm;
      mpi_comm = LOBPCGcomm;
      mpisize = mpisizeLOBPCG;
      mpirank = mpirankLOBPCG;
    }

    GetTime(timeStaLOBPCG);

    Int height = ncvband;
    Int width = nkband;
    Int lda = 3 * width;
    Int widthBlocksize = width / mpisize;
    Int heightBlocksize = height / mpisize;
    Int widthLocal = widthBlocksize;
    Int heightLocal = heightBlocksize;

    if(mpirank < (width % mpisize)){
      widthLocal = widthBlocksize + 1;
    }

    if(mpirank < (height % mpisize)){
      heightLocal = heightBlocksize + 1;
    }

    // Int eigMaxIter_LRTDDFT = 1; 
    // Real eigMinTolerance_LRTDDFT = 1e-3;
    // Real eigTolerance_LRTDDFT = 1e-10;
    Int numEig_LRTDDFT = width;
    DblNumMat  S(heightLocal, 3*width), AS(heightLocal, 3*width);
    DblNumMat  AMat(3*width, 3*width), BMat(3*width, 3*width);
    DblNumMat  AMatT1(3*width, 3*width);

    DblNumMat  XTX(width, width);
    DblNumMat  XTXtemp(width, width);
    DblNumMat  XTXtemp1(width, width);

    DblNumMat  Xtemp(heightLocal, width);
    Real       resMax, resMin;
    DblNumVec  resNormLocal (width); 
    SetValue(resNormLocal, 0.0);
    DblNumVec  resNorm(width);
    SetValue(resNorm, 0.0);
    DblNumMat  X(heightLocal, width, false, S.VecData(0));
    DblNumMat  W(heightLocal, width, false, S.VecData(width));
    DblNumMat  P(heightLocal, width, false, S.VecData(2*width));
    DblNumMat  AX(heightLocal, width, false, AS.VecData(0));
    DblNumMat  AW(heightLocal, width, false, AS.VecData(width));
    DblNumMat  AP(heightLocal, width, false, AS.VecData(2*width));

    DblNumMat  Xcol(height, widthLocal);
    DblNumMat  Wcol(height, widthLocal);
    DblNumMat AXcol(height, widthLocal);
    DblNumMat AWcol(height, widthLocal);
    // numSet = 2    : Steepest descent (Davidson), only use (X | W)
    //        = 3    : Conjugate gradient, use all the triplet (X | W | P)
    Int numSet = 2;

    // numLocked is the number of converged vectors
    Int numLockedLocal = 0, numLockedSaveLocal = 0;
    Int numLockedTotal = 0, numLockedSaveTotal = 0; 
    Int numLockedSave = 0;
    Int numActiveLocal = 0;
    Int numActiveTotal = 0;

    const Int numLocked = 0;  // Never perform locking in this version
    const Int numActive = width;
    bool isConverged = false;
    SetValue(S, 0.0);
    SetValue(AS, 0.0);

    DblNumVec  eigValS(lda);
    SetValue(eigValS, 0.0);
    //GaussianRandom(X); //random initial of X, can be improved later

    DblNumVec DiagonalE(height); //D matrix
    GetTime(timeSta);
    Int kk = 0;
//#ifdef _USE_OPENMP_
//#pragma omp parallel 
    {
//#endif
//#ifdef _USE_OPENMP_
//#pragma omp for schedule (dynamic,1) 
//#endif
      for (Int k = nocc_ - nvband; k < nocc_; k++) {
        for (Int j = nocc_; j < nocc_ + ncband; j++) {
          DiagonalE(kk) = eigVal_(j) - eigVal_(k);
          kk++;
        }//for j
      }
//#ifdef _USE_OPENMP_
    }
//#endif
    GetTime(timeEnd);
    statusOFS << "Time for D matrix                = " << timeEnd - timeSta << " [s]" << std::endl;
    int resnum = width % mpisize;
    IntNumVec widthresTotal(mpisize);
    SetValue(widthresTotal, 0);

    SetValue(Xcol, 0.0);

    GetTime(timeSta);
    for (int i = 1; i < mpisize; i++) {
      if (i <= resnum) {
        widthresTotal(i) = widthresTotal(i - 1) + widthBlocksize + 1;
      }
      else
        widthresTotal(i) = widthresTotal(i - 1) + widthBlocksize;
    }
//#ifdef _USE_OPENMP_
//#pragma omp parallel 
    {
//#endif
//#ifdef _USE_OPENMP_
//#pragma omp for schedule (dynamic,1) nowait
//#endif
      for (int i = 0; i < widthLocal; i++) {
        int j = i + widthresTotal(mpirank);
        Xcol(j, i) = 1.0;//DiagonalE(widthresTotal(mpirank) + i);
      }
//#ifdef _USE_OPENMP_
    }
//#endif
    GetTime(timeEnd);
    statusOFS << "Time for initial Xcol            = " << timeEnd - timeSta << " [s]" << std::endl;

    GetTime(timeSta);

    AlltoallForward(Xcol, X, mpi_comm);
    GetTime(timeEnd);
    statusOFS << "Time for Alltoall Xcol to X      = " << timeEnd - timeSta << " [s]" << std::endl;

    // *********************************************************************
    // Main loop
    // *********************************************************************
    // Orthogonalization through Cholesky factorization

    GetTime(timeSta);
    blas::Gemm('T', 'N', width, width, heightLocal, 1.0, X.Data(), heightLocal, X.Data(), heightLocal, 0.0, XTXtemp1.Data(), width);
    SetValue(XTX, 0.0);
    MPI_Allreduce(XTXtemp1.Data(), XTX.Data(), width * width, MPI_DOUBLE, MPI_SUM, mpi_comm);
    GetTime(timeEnd);
    statusOFS << "Time for generating XTX          = " << timeEnd - timeSta << " [s]" << std::endl;

    GetTime(timeSta);
    if ( mpirank == 0) {
      GetTime( timeSta );
      lapack::Potrf( 'U', width, XTX.Data(), width );
      GetTime( timeEnd );
    }
    MPI_Bcast(XTX.Data(), width * width, MPI_DOUBLE, 0, mpi_comm);
    GetTime(timeEnd);
    statusOFS << "Time for Potrf XTX and Bcast     = " << timeEnd - timeSta << " [s]" << std::endl;

    // X <- X * U^{-1} is orthogonal
    GetTime(timeSta);
    blas::Trsm('R', 'U', 'N', 'N', heightLocal, width, 1.0, XTX.Data(), width, X.Data(), heightLocal);
    GetTime(timeEnd);
    statusOFS << "Time for Trsm XTX                = " << timeEnd - timeSta << " [s]" << std::endl;
    // *********************************************************************
    // Applying the Hamiltonian matrix, maybe wrong def
    // *********************************************************************
    GetTime(timeSta);
    AlltoallBackward(X, Xcol, mpi_comm);
    GetTime(timeEnd);
    statusOFS << "Time for Alltoall X to Xcol      = " << timeEnd - timeSta << " [s]" << std::endl;

    for (Int i = 0;i < widthLocal; i++){
      for (Int j = 0;j < height; j++){
        AXcol(j,i) = DiagonalE(j) * Xcol(j,i);
      }
    }

    GetTime(timeSta);
    // blas::Gemm('N', 'N', height, widthLocal, height, 1.0, hamTDDFT.Data(), height, Xcol.Data(), height, 0.0, AXcol.Data(), height);
    
    DblNumMat  fxcLocaltemp1(height, height);
    blas::Gemm('T', 'N', ncvbandMu, ncvbandMu, ntotLocal, 1.0, psiphizetaRow.Data(), ntotLocal, psiphifxcR2C.Data(), ntotLocal, 0.0, fxcLocaltemp1.Data(), ncvbandMu);

    DblNumMat fxcLocaltemp3(ncvbandMu, ncvbandMu);
    MPI_Allreduce(fxcLocaltemp1.Data(), fxcLocaltemp3.Data(), ncvbandMu * ncvbandMu, MPI_DOUBLE, MPI_SUM, mpi_comm);

    DblNumMat AXcoltemp(ncvbandMu, widthLocal);

    blas::Gemm('N', 'N', ncvbandMu, widthLocal, height, 1.0, psiphiMu.Data(), ncvbandMu, Xcol.Data(), height, 0.0, AXcoltemp.Data(), ncvbandMu);

    DblNumMat fxcLocal(ncvbandMu, widthLocal);
    blas::Gemm('N', 'N', ncvbandMu, widthLocal, ncvbandMu, 1.0, fxcLocaltemp3.Data(), ncvbandMu, AXcoltemp.Data(), ncvbandMu, 0.0, fxcLocal.Data(), ncvbandMu);

    blas::Gemm('T', 'N', ncvband, widthLocal, ncvbandMu, 1.0, psiphiMu.Data(), ncvbandMu, fxcLocal.Data(), ncvbandMu, 1.0, AXcol.Data(), ncvband);

    //statusOFS << AXcol << std::endl;
    GetTime(timeEnd);
    statusOFS << "Time for generating AXcol        = " << timeEnd - timeSta << " [s]" << std::endl;

    GetTime(timeSta);
    AlltoallForward(AXcol, AX, mpi_comm);
    GetTime(timeEnd);
    statusOFS << "Time for Alltoall AXcol to AX    = " << timeEnd - timeSta << " [s]" << std::endl;
    // Start the main loop
    Int iter = 0; 
    Real timeIterSta, timeIterEnd;
    Real timeMainLoopLOBPCGSta, timeMainLoopLOBPCGEnd;
    GetTime(timeMainLoopLOBPCGSta);
    do{
      iter++;
      if( iter == 1 )
        numSet = 2;
      else
        numSet = 3;

      GetTime(timeIterSta);
      SetValue(AMat, 0.0);
      SetValue(BMat, 0.0);

      // XTX <- X' * (AX)
      blas::Gemm('T', 'N', width, width, heightLocal, 1.0, X.Data(), heightLocal, AX.Data(), heightLocal, 0.0, XTXtemp1.Data(), width);
      SetValue( XTX, 0.0 );
      MPI_Allreduce( XTXtemp1.Data(), XTX.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm);
      lapack::Lacpy('A', width, width, XTX.Data(), width, AMat.Data(), lda);
      // Compute the residual.
      // R <- AX - X*(X'*AX)
      lapack::Lacpy('A', heightLocal, width, AX.Data(), heightLocal, Xtemp.Data(), heightLocal);
      blas::Gemm('N', 'N', heightLocal, width, width, -1.0, X.Data(), heightLocal, AMat.Data(), lda, 1.0, Xtemp.Data(), heightLocal);
      SetValue(resNormLocal, 0.0 );
      for(Int k = 0; k < width; k++){
        resNormLocal(k) = Energy(DblNumVec(heightLocal, false, Xtemp.VecData(k)));
      }
      SetValue( resNorm, 0.0 );
      MPI_Allreduce(resNormLocal.Data(), resNorm.Data(), width, MPI_DOUBLE, MPI_SUM, mpi_comm);

      if(mpirank == 0) {
        for(Int k = 0; k < width; k++){
          resNorm(k) = std::sqrt(resNorm(k)) / std::max(1.0, std::abs(XTX(k, k)));
        }
      }

      MPI_Bcast(resNorm.Data(), width, MPI_DOUBLE, 0, mpi_comm);

      resMax = *(std::max_element(resNorm.Data(), resNorm.Data() + numEig_LRTDDFT));
      resMin = *(std::min_element(resNorm.Data(), resNorm.Data() + numEig_LRTDDFT));
      if(resMax < eigTolerance_LRTDDFT){
        isConverged = true;
        break;
      }
      numActiveTotal = width - numLockedTotal;
      // numActiveLocal = width - numLockedLocal;

      // lapack::Lacpy('A', heightLocal, width, Xtemp.Data(), heightLocal, W.Data(), heightLocal);
      AlltoallBackward(Xtemp, Xcol, mpi_comm);

      // AlltoallBackward(W, Wcol, mpi_comm);

      //statusOFS << "Time for Alltoall AXcol to AX    = " << timeEnd - timeSta << " [s]" << std::endl;

      // *********************************************************************
      // AddTeterPrecond here, should be implemented later
      // *********************************************************************
      // Compute the preconditioned residual W = T*R.
      Real norm = 0.0; 

      lapack::Lacpy('A', height, widthLocal, Xcol.Data(), height, Wcol.Data(), height);
      for (int i = 0; i < widthLocal; i++) {
          int j = i + widthresTotal(mpirank);
          Wcol(j, i) /=  DiagonalE(widthresTotal(mpirank) + i) - eigValS(widthresTotal(mpirank) + i);
       }

      // Normalize the preconditioned residual

#ifdef _USE_OPENMP_
#pragma omp parallel 
      {
#endif
#ifdef _USE_OPENMP_
#pragma omp for schedule (dynamic,1) nowait
#endif
        for(Int k = numLockedLocal; k < widthLocal; k++){
          norm = Energy(DblNumVec(height, false, Wcol.VecData(k)));
          norm = std::sqrt(norm);
          blas::Scal(height, 1.0 / norm, Wcol.VecData(k), 1);
        }
#ifdef _USE_OPENMP_
      }
#endif

      // Normalize the conjugate direction
      Real normPLocal[width]; 
      Real normP[width]; 
      if(numSet == 3){
#ifdef _USE_OPENMP_
#pragma omp parallel 
        {
#endif
#ifdef _USE_OPENMP_
#pragma omp for schedule (dynamic,1) 
#endif
          for(Int k = numLockedLocal; k < width; k++){
            normPLocal[k] = Energy(DblNumVec(heightLocal, false, P.VecData(k)));
            normP[k] = 0.0;
          }
#ifdef _USE_OPENMP_
        }
#endif

        MPI_Allreduce(&normPLocal[0], &normP[0], width, MPI_DOUBLE, MPI_SUM, mpi_comm);
#ifdef _USE_OPENMP_
#pragma omp parallel 
        {
#endif
#ifdef _USE_OPENMP_
#pragma omp for schedule (dynamic,1) nowait
#endif
          for(Int k = numLockedLocal; k < width; k++){
            norm = std::sqrt(normP[k]);
            blas::Scal(heightLocal, 1.0 / norm, P.VecData(k), 1);
            blas::Scal(heightLocal, 1.0 / norm, AP.VecData(k), 1);
          }
#ifdef _USE_OPENMP_
        }
#endif
      }

      // *********************************************************************
      // Compute AW = A*W, likely to be wrong def
      // *********************************************************************

      for (Int i = 0;i < widthLocal; i++){
        for (Int j = 0;j < height; j++){
          AWcol(j,i) = DiagonalE(j) * Wcol(j,i);
        }
      }

      DblNumMat Wcoltemp(ncvbandMu, widthLocal);
      blas::Gemm('N', 'N', ncvbandMu, widthLocal, height, 1.0, psiphiMu.Data(), ncvbandMu, Wcol.Data(), height, 0.0, Wcoltemp.Data(), ncvbandMu);

      DblNumMat Wcoltemp2(ncvbandMu, widthLocal);
      blas::Gemm('N', 'N', ncvbandMu, widthLocal, ncvbandMu, 1.0, fxcLocaltemp3.Data(), ncvbandMu, Wcoltemp.Data(), ncvbandMu, 0.0, Wcoltemp2.Data(), ncvbandMu);

      blas::Gemm('T', 'N', ncvband, widthLocal, ncvbandMu, 1.0, psiphiMu.Data(), ncvbandMu, Wcoltemp2.Data(), ncvbandMu, 1.0, AWcol.Data(), height);

      //blas::Gemm('N', 'N', height, widthLocal, height, 1.0, hamTDDFT.VecData(numLockedLocal), height, Wcol.VecData(numLockedLocal), height, 0.0, AWcol.Data(), height);
      AlltoallForward(Wcol, W, mpi_comm);
      AlltoallForward(AWcol, AW, mpi_comm);

      // Compute X' * (AW)
      blas::Gemm('T', 'N', width, numActive, heightLocal, 1.0, X.Data(), heightLocal, AW.VecData(numLocked), heightLocal, 0.0, XTXtemp1.Data(), width);
      SetValue(XTXtemp, 0.0);
      MPI_Allreduce(XTXtemp1.Data(), XTXtemp.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm);
      lapack::Lacpy('A', width, width, XTXtemp.Data(), width, &AMat(0,width), lda);

      // Compute W' * (AW)
      blas::Gemm('T', 'N', numActive, numActive, heightLocal, 1.0, W.VecData(numLocked), heightLocal, AW.VecData(numLocked), heightLocal, 0.0, XTXtemp1.Data(), width);
      MPI_Allreduce(XTXtemp1.Data(), XTXtemp.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm);
      lapack::Lacpy('A', width, width, XTXtemp.Data(), width, &AMat(width,width), lda);

      if(numSet == 3){ // Compute AMat 
        // Compute X' * (AP)
        blas::Gemm('T', 'N', width, numActive, heightLocal, 1.0, X.Data(), heightLocal, AP.VecData(numLocked), heightLocal, 0.0, XTXtemp1.Data(), width);
        SetValue(XTXtemp, 0.0);
        MPI_Allreduce(XTXtemp1.Data(), XTXtemp.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm);
        lapack::Lacpy('A', width, width, XTXtemp.Data(), width, &AMat(0, width+numActive), lda);

        // Compute W' * (AP)
        blas::Gemm('T', 'N', numActive, numActive, heightLocal, 1.0, W.VecData(numLocked), heightLocal, AP.VecData(numLocked), heightLocal, 0.0, XTXtemp1.Data(), width);
        SetValue(XTXtemp, 0.0);
        MPI_Allreduce(XTXtemp1.Data(), XTXtemp.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm);
        lapack::Lacpy('A', width, width, XTXtemp.Data(), width, &AMat(width, width+numActive), lda);

        // Compute P' * (AP)
        blas::Gemm('T', 'N', numActive, numActive, heightLocal, 1.0, P.VecData(numLocked), heightLocal, AP.VecData(numLocked), heightLocal, 0.0, XTXtemp1.Data(), width);
        SetValue(XTXtemp, 0.0);
        MPI_Allreduce(XTXtemp1.Data(), XTXtemp.Data(), width*width, MPI_DOUBLE, MPI_SUM, domain_.comm);
        lapack::Lacpy('A', width, width, XTXtemp.Data(), width, &AMat(width+numActive, width+numActive), lda);
      }

      // Compute BMat (overlap matrix)
      // Compute X'*X (B is orthogonal matrix)
      blas::Gemm('T', 'N', width, width, heightLocal, 1.0, X.Data(), heightLocal, X.Data(), heightLocal, 0.0, XTXtemp1.Data(), width); 
      SetValue(XTXtemp, 0.0);
      MPI_Allreduce(XTXtemp1.Data(), XTXtemp.Data(), width*width, MPI_DOUBLE, MPI_SUM, domain_.comm);
      lapack::Lacpy('A', width, width, XTXtemp.Data(), width, &BMat(0,0), lda);

      // Compute X'*W
      blas::Gemm('T', 'N', width, numActive, heightLocal, 1.0, X.Data(), heightLocal, W.VecData(numLocked), heightLocal, 0.0, XTXtemp1.Data(), width);
      SetValue(XTXtemp, 0.0);
      MPI_Allreduce(XTXtemp1.Data(), XTXtemp.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm);
      lapack::Lacpy('A', width, width, XTXtemp.Data(), width, &BMat(0,width), lda);     

      // Compute W'*W
      blas::Gemm('T', 'N', numActive, numActive, heightLocal, 1.0, W.VecData(numLocked), heightLocal, W.VecData(numLocked), heightLocal, 0.0, XTXtemp1.Data(), width);
      SetValue(XTXtemp, 0.0);
      MPI_Allreduce(XTXtemp1.Data(), XTXtemp.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm);
      lapack::Lacpy('A', width, width, XTXtemp.Data(), width, &BMat(width, width), lda);  

      if( numSet == 3 ){
        // Compute X'*P
        blas::Gemm('T', 'N', width, numActive, heightLocal, 1.0, X.Data(), heightLocal, P.VecData(numLocked), heightLocal, 0.0, XTXtemp1.Data(), width);
        SetValue(XTXtemp, 0.0);
        MPI_Allreduce(XTXtemp1.Data(), XTXtemp.Data(), width*width, MPI_DOUBLE, MPI_SUM, domain_.comm);
        lapack::Lacpy('A', width, width, XTXtemp.Data(), width, &BMat(0, width+numActive), lda);

        // Compute W'*P

        blas::Gemm( 'T', 'N', numActive, numActive, heightLocal, 1.0, W.VecData(numLocked), heightLocal, P.VecData(numLocked), heightLocal, 0.0, XTXtemp1.Data(), width );
        SetValue( XTXtemp, 0.0 );
        MPI_Allreduce( XTXtemp1.Data(), XTXtemp.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm );
        lapack::Lacpy( 'A', width, width, XTXtemp.Data(), width, &BMat(width, width+numActive), lda );


        // Compute P'*P 
        blas::Gemm('T', 'N', numActive, numActive, heightLocal, 1.0, P.VecData(numLocked), heightLocal, P.VecData(numLocked), heightLocal, 0.0, XTXtemp1.Data(), width);
        SetValue(XTXtemp, 0.0);
        MPI_Allreduce(XTXtemp1.Data(), XTXtemp.Data(), width*width, MPI_DOUBLE, MPI_SUM, domain_.comm);

        lapack::Lacpy('A', width, width, XTXtemp.Data(), width, &BMat(width+numActive, width+numActive), lda);
      }
      Int numCol;
      if(numSet == 3){
        // Conjugate gradient
        numCol = width + 2 * numActiveTotal;
      }
      else{
        numCol = width + numActiveTotal;
      }
      if ( mpirank == 0 ) {
        DblNumVec  sigma2(lda);
        DblNumVec  invsigma(lda);
        SetValue( sigma2, 0.0 );
        SetValue( invsigma, 0.0 );

        // Symmetrize A and B first.  This is important.
#ifdef _USE_OPENMP_
#pragma omp parallel 
        {
#endif
#ifdef _USE_OPENMP_
#pragma omp for schedule (dynamic,1) 
#endif
          for( Int j = 0; j < numCol; j++ ){
            for( Int i = j+1; i < numCol; i++ ){
              AMat(i,j) = AMat(j,i);
              BMat(i,j) = BMat(j,i);
            }
          }
#ifdef _USE_OPENMP_
        }
#endif

        GetTime( timeSta );

        lapack::Syevd( 'V', 'U', numCol, BMat.Data(), lda, sigma2.Data() );
        GetTime( timeEnd );
        // iterMpirank0 = iterMpirank0 + 1;
        // timeMpirank0 = timeMpirank0 + ( timeEnd - timeSta );

        Int numKeep = 0;

        for( Int i = numCol-1; i>=0; i-- ){
          if( sigma2(i) / sigma2(numCol-1) >  1e-8 )
            numKeep++;
          else
            break;
        }


#if ( _DEBUGlevel_ >= 1 )
        statusOFS << "sigma2 = " << sigma2 << std::endl;
#endif

#if ( _DEBUGlevel_ >= 1 )
        statusOFS << "sigma2(0)        = " << sigma2(0) << std::endl;
        statusOFS << "sigma2(numCol-1) = " << sigma2(numCol-1) << std::endl;
        statusOFS << "numKeep          = " << numKeep << std::endl;
#endif

#ifdef _USE_OPENMP_
#pragma omp parallel 
        {
#endif
#ifdef _USE_OPENMP_
#pragma omp for schedule (dynamic,1) 
#endif
          for( Int i = 0; i < numKeep; i++ ){
            invsigma(i) = 1.0 / std::sqrt( sigma2(i+numCol-numKeep) );
          }
#ifdef _USE_OPENMP_
        }
#endif

        if( numKeep < width ){
          std::ostringstream msg;
          msg 
            << "width   = " << width << std::endl
            << "numKeep =  " << numKeep << std::endl
            << "there are not enough number of columns." << std::endl;
          ErrorHandling( msg.str().c_str() );
        }

        SetValue( AMatT1, 0.0 );
        // Evaluate S^{-1/2} (U^T A U) S^{-1/2}
        GetTime( timeSta );
        blas::Gemm( 'N', 'N', numCol, numKeep, numCol, 1.0,
            AMat.Data(), lda, BMat.VecData(numCol-numKeep), lda,
            0.0, AMatT1.Data(), lda );
        GetTime( timeEnd );
        // iterMpirank0 = iterMpirank0 + 1;
        // timeMpirank0 = timeMpirank0 + ( timeEnd - timeSta );

        GetTime( timeSta );
        blas::Gemm( 'T', 'N', numKeep, numKeep, numCol, 1.0,
            BMat.VecData(numCol-numKeep), lda, AMatT1.Data(), lda, 
            0.0, AMat.Data(), lda );
        GetTime( timeEnd );
        // iterMpirank0 = iterMpirank0 + 1;
        // timeMpirank0 = timeMpirank0 + ( timeEnd - timeSta );

        GetTime( timeSta );
#ifdef _USE_OPENMP_
#pragma omp parallel 
        {
#endif
#ifdef _USE_OPENMP_
#pragma omp for schedule (dynamic,1) 
#endif
          for( Int j = 0; j < numKeep; j++ ){
            for( Int i = 0; i < numKeep; i++ ){
              AMat(i,j) *= invsigma(i)*invsigma(j);
            }
          }
#ifdef _USE_OPENMP_
        }
#endif
        GetTime( timeEnd );
        // iterMpirank0 = iterMpirank0 + 1;
        // timeMpirank0 = timeMpirank0 + ( timeEnd - timeSta );

        // Solve the standard eigenvalue problem
        GetTime( timeSta );
        lapack::Syevd( 'V', 'U', numKeep, AMat.Data(), lda,
            eigValS.Data() );
        GetTime( timeEnd );
        // iterMpirank0 = iterMpirank0 + 1;
        // timeMpirank0 = timeMpirank0 + ( timeEnd - timeSta );

        // Compute the correct eitgenvectors and save them in AMat
#ifdef _USE_OPENMP_
#pragma omp parallel 
        {
#endif
#ifdef _USE_OPENMP_
#pragma omp for schedule (dynamic,1) nowait
#endif
          for( Int j = 0; j < numKeep; j++ ){
            for( Int i = 0; i < numKeep; i++ ){
              AMat(i,j) *= invsigma(i);
            }
          }
#ifdef _USE_OPENMP_
        }
#endif

        GetTime( timeSta );
        blas::Gemm( 'N', 'N', numCol, numKeep, numKeep, 1.0,
            BMat.VecData(numCol-numKeep), lda, AMat.Data(), lda,
            0.0, AMatT1.Data(), lda );
        // iterMpirank0 = iterMpirank0 + 1;
        // timeMpirank0 = timeMpirank0 + ( timeEnd - timeSta );

        lapack::Lacpy( 'A', numCol, numKeep, AMatT1.Data(), lda, 
            AMat.Data(), lda );

      } // mpirank ==0
      MPI_Bcast(AMat.Data(), lda*lda, MPI_DOUBLE, 0, mpi_comm);
      MPI_Bcast(BMat.Data(), lda*lda, MPI_DOUBLE, 0, mpi_comm);
      MPI_Bcast(eigValS.Data(), lda, MPI_DOUBLE, 0, mpi_comm);


      if( numSet == 2 ){
        // Update the eigenvectors 
        // X <- X * C_X + W * C_W
        blas::Gemm('N', 'N', heightLocal, width, width, 1.0, X.Data(), heightLocal, &AMat(0,0), lda, 0.0, Xtemp.Data(), heightLocal);

        blas::Gemm('N', 'N', heightLocal, width, numActive, 1.0, W.VecData(numLocked), heightLocal, &AMat(width,0), lda, 1.0, Xtemp.Data(), heightLocal);
        // Save the result into X
        lapack::Lacpy( 'A', heightLocal, width, Xtemp.Data(), heightLocal,  X.Data(), heightLocal);
        // P <- W
        lapack::Lacpy( 'A', heightLocal, numActive, W.VecData(numLocked),  heightLocal, P.VecData(numLocked), heightLocal);
      } 
      else{ //numSet == 3
        // Compute the conjugate direction
        // P <- W * C_W + P * C_P
        blas::Gemm('N', 'N', heightLocal, width, numActive, 1.0, W.VecData(numLocked), heightLocal, &AMat(width, 0), lda, 0.0, Xtemp.Data(), heightLocal);

        blas::Gemm('N', 'N', heightLocal, width, numActive, 1.0, P.VecData(numLocked), heightLocal, &AMat(width+numActive,0), lda, 1.0, Xtemp.Data(), heightLocal);

        lapack::Lacpy('A', heightLocal, numActive, Xtemp.VecData(numLocked), heightLocal, P.VecData(numLocked), heightLocal);
        // Update the eigenvectors
        // X <- X * C_X + P

        blas::Gemm('N', 'N', heightLocal, width, width, 1.0, X.Data(), heightLocal, &AMat(0,0), lda, 1.0, Xtemp.Data(), heightLocal);
        lapack::Lacpy('A', heightLocal, width, Xtemp.Data(), heightLocal, X.Data(), heightLocal);
      }
      // Update AX and AP
      if( numSet == 2 ){
        // AX <- AX * C_X + AW * C_W
        blas::Gemm('N', 'N', heightLocal, width, width, 1.0, AX.Data(), heightLocal, &AMat(0,0), lda, 0.0, Xtemp.Data(), heightLocal);
        blas::Gemm('N', 'N', heightLocal, width, numActive, 1.0, AW.VecData(numLocked), heightLocal, &AMat(width,0), lda, 1.0, Xtemp.Data(), heightLocal);
        lapack::Lacpy('A', heightLocal, width, Xtemp.Data(), heightLocal, AX.Data(), heightLocal);
        // AP <- AW
        lapack::Lacpy( 'A', heightLocal, numActive, AW.VecData(numLocked), heightLocal, AP.VecData(numLocked), heightLocal);
      }
      else{
        // AP <- AW * C_W + A_P * C_P
        blas::Gemm('N', 'N', heightLocal, width, numActive, 1.0, AW.VecData(numLocked), heightLocal, &AMat(width,0), lda, 0.0, Xtemp.Data(), heightLocal);
        blas::Gemm('N', 'N', heightLocal, width, numActive, 1.0, AP.VecData(numLocked), heightLocal, &AMat(width+numActive, 0), lda, 1.0, Xtemp.Data(), heightLocal);
        lapack::Lacpy('A', heightLocal, numActive, Xtemp.VecData(numLocked), heightLocal, AP.VecData(numLocked), heightLocal);

        // AX <- AX * C_X + AP
        blas::Gemm('N', 'N', heightLocal, width, width, 1.0, AX.Data(), heightLocal, &AMat(0,0), lda, 1.0, Xtemp.Data(), heightLocal);
        lapack::Lacpy('A', heightLocal, width, Xtemp.Data(), heightLocal,  AX.Data(), heightLocal);
      } // if ( numSet == 2 )
      GetTime(timeIterEnd);
#if ( _DEBUGlevel_ >= 1 )
      statusOFS << "Time for iter " << iter << " = "<< timeIterEnd - timeIterSta << " [s]" << std::endl;
#endif
    } while((iter < (eigMaxIter_LRTDDFT)) && ( (iter < eigMaxIter_LRTDDFT) || (resMin > eigMinTolerance_LRTDDFT)));

    GetTime(timeMainLoopLOBPCGEnd);

    statusOFS << "Time for main loop in LOBPCG     = " << timeMainLoopLOBPCGEnd - timeMainLoopLOBPCGSta << " [s]" << std::endl;

    if ( mpirank == 0 ){
      lapack::Syevd( 'V', 'U', width, XTX.Data(), width, eigValS.Data() );
    }
    MPI_Bcast(XTX.Data(), width * width, MPI_DOUBLE, 0, mpi_comm);
    MPI_Bcast(eigValS.Data(), width, MPI_DOUBLE, 0, mpi_comm);
    // X <- X*C
    blas::Gemm('N', 'N', heightLocal, width, width, 1.0, X.Data(), heightLocal, XTX.Data(), width, 0.0, Xtemp.Data(), heightLocal);
    lapack::Lacpy( 'A', heightLocal, width, Xtemp.Data(), heightLocal, X.Data(), heightLocal);

    AlltoallBackward(X, Xcol, mpi_comm);

    if(isConverged){
      statusOFS << std::endl << "After " << iter 
        << " iterations, LOBPCG has converged."  << std::endl
        << "The maximum norm of the residual is " 
        << resMax << std::endl << std::endl
        << "The minimum norm of the residual is " 
        << resMin << std::endl << std::endl;
    }
    else{
      statusOFS << std::endl << "After " << iter 
        << " iterations, LOBPCG did not converge. " << std::endl
        << "The maximum norm of the residual is " 
        << resMax << std::endl << std::endl
        << "The minimum norm of the residual is " 
        << resMin << std::endl << std::endl;
    }

    GetTime(timeEndLOBPCG);
    statusOFS << "Time for LOBPCG in LRTDDFT       = " << timeEndLOBPCG - timeStaLOBPCG << " [s]" << std::endl;

    GetTime(timeEndTotal);
    statusOFS << std::endl << "Total time for LDA LRTDDFT       = " << timeEndTotal - timeStaTotal << " [s]" << std::endl;

    if (esdfParam.isOutputExcitationEnergy) {
      statusOFS << std::endl << "Output LRTDDFT excitation energy" << std::endl << std::endl;
      for (Int i = 0; i < width; i++) {
        statusOFS << "excitation# = " << i << "      " << "eigValS = " << std::scientific << std::setw(12) << std::setprecision(5) << eigValS[i]*2 << " [Ry]" << std::endl;
      }
    }

      //gather X from all mpi

      DblNumMat  XX(ncvband, width);
      SetValue(XX, 0.0);

      Int widthXBlocksize = width / mpisize;
      Int widthXLocal = widthXBlocksize;
  
      if(mpirank < (width % mpisize)){
        widthXLocal = widthXBlocksize + 1;
      }
      IntNumVec  localSizeXDispls(mpisize);
      IntNumVec  localSizeXVec(mpisize);

      localSizeXDispls[0] = 0;
      
      for(Int i = 0; i < mpisize; i++) {
        if(i < (width % mpisize)){
          localSizeXVec[i] = (widthXBlocksize + 1)*ncvband;
        }
        else{localSizeXVec[i] = widthXBlocksize*ncvband;}
      }
      for(Int i = 1; i < mpisize; i++) {
        localSizeXDispls[i] = localSizeXVec[i-1] + localSizeXDispls[i-1];
      }
      //statusOFS << "Output localSizeVec" << localSizeVec << std::endl;
      //statusOFS << "Output localSizeDis" << localSizeDispls << std::endl;
      //statusOFS << "Output Xcol" << Xcol << std::endl;
      MPI_Gatherv(Xcol.Data(), widthXLocal*ncvband, MPI_DOUBLE, XX.Data(), localSizeXVec.Data(), localSizeXDispls.Data(), MPI_DOUBLE, 0, domain_.comm);

    if (esdfParam.isOutputExcitationWfn){
      
      statusOFS << std::endl << "Output LRTDDFT excitation Wfn" << std::endl << std::endl;
      std::string         eleDensityFileName;
      std::string         holeDensityFileName;

      DblNumMat  Xiv(ncband, nvband);
      SetValue(Xiv, 0.0);
      DblNumVec weightLocal(ntotLocal);
      SetValue(weightLocal, 0.0);
      DblNumMat  dXiv(ntotLocal, nvband);
      SetValue(dXiv, 0.0);
      DblNumVec eledens(ntot);
      SetValue(eledens, 0.0);
      DblNumVec holedens(ntot);
      SetValue(holedens, 0.0);
      DblNumMat  Xtem(nvband, nvband);
      SetValue(Xtem, 0.0);
      DblNumMat  XXtem(nvband, nvband);
      SetValue(XXtem, 0.0);

      //statusOFS << "Output XX" << XX << std::endl;

      Int start = esdfParam.startband;
      Int end = esdfParam.endband;

      for (Int i = start; i < end; i++){    //output choosen bands;
        Real *wp = XX.Data() + i*ncvband;
        Real *p = Xiv.Data();  
        for (Int j = 0; j < ncvband; j++){
          *(p++) = *(wp++);
        }

        MPI_Bcast(Xiv.Data(), nvband*ncband, MPI_DOUBLE, 0, domain_.comm);
        blas::Gemm('N', 'N', ntotLocal, nvband, ncband, 1.0, phiRowLocal.Data(), ntotLocal, Xiv.Data(), ncband, 0.0, dXiv.Data(), ntotLocal);
        for (Int j = 0; j < nvband; j++) {
          for (Int k = 0; k < ntotLocal; k++) {
            weightLocal(k) += dXiv(k, j)*dXiv(k, j);
          }
        }

        MPI_Allgatherv(weightLocal.Data(), ntotLocal, MPI_DOUBLE, eledens.Data(), weightSize.Data(), weightSizeDispls.Data(), MPI_DOUBLE, domain_.comm);
        double norm = 0;
        for (Int j = 0; j < ntot; j++){
          norm += eledens(j)*eledens(j);
        }
        for (Int j = 0; j < ntot; j++){
          eledens(j) = eledens(j)/norm;
        }
        statusOFS << "Output norm" << norm << std::endl;
        //MPI_Allgather(weightLocal.Data(), ntotLocal, MPI_DOUBLE, eledens.Data(), ntotLocal, MPI_DOUBLE, domain_.comm);

        blas::Gemm('T', 'N', nvband, nvband, ntotLocal, 1.0, dXiv.Data(), ntotLocal, dXiv.Data(), ntotLocal, 0.0, Xtem.Data(), nvband);
        MPI_Allreduce(Xtem.Data(), XXtem.Data(), nvband*nvband, MPI_DOUBLE, MPI_SUM, domain_.comm);
        blas::Gemm('N', 'N', ntotLocal, nvband, nvband, 1.0, psiRowLocal.Data(), ntotLocal, XXtem.Data(), nvband, 0.0, dXiv.Data(), ntotLocal);
        for (Int j = 0; j < nvband; j++) {
          for (Int k = 0; k < ntotLocal; k++) {
            weightLocal(k) += dXiv(k, j)*psiRowLocal(k, j);
          }
        }
        //MPI_Allgather(weightLocal.Data(), ntotLocal, MPI_DOUBLE, holedens.Data(), ntotLocal, MPI_DOUBLE, domain_.comm);

        MPI_Allgatherv(weightLocal.Data(), ntotLocal, MPI_DOUBLE, holedens.Data(), weightSize.Data(), weightSizeDispls.Data(), MPI_DOUBLE, domain_.comm);

        for (Int j = 0; j < ntot; j++){
          norm += holedens(j)*holedens(j);
        }
        statusOFS << "Output norm" << norm << std::endl;

        for (Int j = 0; j < ntot; j++){
          holedens(j) = holedens(j)/norm;
        }
        std::string eleDensityFileName = "ele" + std::to_string(i);
        std::ofstream eleStream;
        //serialize( eledens, eleStream, NO_MASK );

        std::ofstream erhoStream(eleDensityFileName);
        if( !erhoStream.good() ){
          ErrorHandling( "Ele Density file cannot be opened." );
        }
  
        const Domain& dm =  fft.domain;
        std::vector<DblNumVec>   gridpos(DIM);
        UniformMeshFine ( dm, gridpos );
        for( Int d = 0; d < DIM; d++ ){
          serialize( gridpos[d], erhoStream, NO_MASK );
        }
  
        // Only work for the restricted spin case

        serialize( eledens, erhoStream, NO_MASK );
        erhoStream.close();

        std::string holeDensityFileName = "hole" + std::to_string(i);
        //serialize( eledens, eleStream, NO_MASK );

        std::ofstream hrhoStream(holeDensityFileName);
        if( !hrhoStream.good() ){
          ErrorHandling( "Hole Density file cannot be opened." );
        }
  
        UniformMeshFine ( dm, gridpos );
        for( Int d = 0; d < DIM; d++ ){
          serialize( gridpos[d], hrhoStream, NO_MASK );
        }
  
        // Only work for the restricted spin case

        serialize( holedens, hrhoStream, NO_MASK );
        hrhoStream.close();
      }
      statusOFS << std::endl << "Output LRTDDFT excitation Wfn Success !" << std::endl << std::endl;
    }

    if (esdfParam.isOutputExcitationSpectrum){
      Spectrum(fft, psiRowLocal, phiRowLocal, XX, eigValS, nkband);      
    }

  } // end if (lrtddftEigenSolver == "LOBPCG")


  return;

} //CalculateLRTDDFT

}}//namespace
