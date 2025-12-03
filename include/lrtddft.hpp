#ifndef _LRTDDFT_HPP_
#define _LRTDDFT_HPP_

#include  "environment.hpp"
#include  "numvec_impl.hpp"
#include  "numtns_impl.hpp"
#include  "domain.hpp"
#include  "fourier.hpp"
#include  "spinor.hpp"
#include  "hamiltonian.hpp"
#include  "utility.hpp"
#include  "lapack.hpp"
#include  "esdf.hpp"

namespace dgdft {
class LRTDDFT {
private:

  Int               numExtraState_;         //maxncbandtol
  Int               nocc_;                  //maxnvbandtol
  Int               ntot_;                  //real space Corase grid
  Int               ntotR2C_;               //Corase grid	
  Int               ntotFine_;              //real space Fine grid
  Int               ntotR2CFine_;           //Fine grid	
  Int               numcomponent_ ;         //spinor
  DblNumMat         density_;               //rho
  DblNumVec         eigVal_;                //energy
  Real              vol_;                   //Vol
  Domain            domain_;                //domain

public:
  // *********************************************************************
  // Constructor and destructor
  // *********************************************************************
  void Setup(Hamiltonian& ham, Spinor& psi, Fourier& fft, const Domain& dm);

  void Spectrum(Fourier& fft, DblNumMat psinv, DblNumMat psinc, DblNumMat XX, DblNumVec eigValS, Int nkband);

  void CalculateLRTDDFT(Hamiltonian& ham, Spinor& psi, Fourier& fft, const Domain& dm);

  void CalculateLRTDDFT_ISDF(Hamiltonian& ham, Spinor& psi, Fourier& fft, const Domain& dm);

  void FFTRtoC(Fourier& fft, Hamiltonian& ham, DblNumMat psiphi, DblNumMat& temp, Int ncvband);

  void Calculatefxc(Fourier& fft, DblNumVec& fxcPz);

};
}

#endif // _LRTDDFT_HPP_
