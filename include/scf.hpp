/*
   Copyright (c) 2012 The Regents of the University of California,
   through Lawrence Berkeley National Laboratory.  

Author: Lin Lin, Wei Hu and Amartya Banerjee

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
/// @file scf.hpp
/// @brief SCF class for the global domain or extended element.
/// @date 2012-10-25 Initial version
/// @date 2014-02-01 Dual grid implementation
/// @date 2014-08-07 Parallelization for PWDFT
/// @date 2016-01-19 Add hybrid functional
/// @date 2016-04-08 Update mixing

#ifndef _SCF_HPP_ 
#define _SCF_HPP_

#include  "environment.hpp"
#include  "numvec_impl.hpp"
#include  "domain.hpp"
#include  "fourier.hpp"
#include  "hamiltonian.hpp"
#include  "spinor.hpp"
#include  "periodtable.hpp"
#include  "esdf.hpp"
#include  "eigensolver.hpp"

// TODO In the case of K-point, should the occupation rate, the energy
// and the potential subroutines be moved to SCF class rather than
// stay in the Hamiltonian class?

namespace dgdft{

class SCF
{
private:
  // Control parameters
  Int                 mixMaxDim_;
  std::string         mixType_;
  std::string         mixVariable_;
  Real                mixStepLength_;            
  Real                eigMinTolerance_;
  Real                eigTolerance_;
  Int                 eigMaxIter_;
  Real                scfTolerance_;
  Int                 scfMaxIter_;
  Int                 scfPhiMaxIter_;
  Real                scfPhiTolerance_;
  Int                 numUnusedState_;
  bool                isEigToleranceDynamic_;
  Int                 BlockSizeScaLAPACK_;

  std::string         restartDensityFileName_;
  std::string         restartPotentialFileName_;
  std::string         restartWfnFileName_;

  // Physical parameters
  Real                Tbeta_;                    // Inverse of temperature in atomic unit
  Real                Efree_;                    // Helmholtz free energy
  Real                EfreeHarris_;              // Free energy from Harris functional
  Real                Etot_;                     // Total energy
  Real                Ekin_;                     // Kinetic energy
  Real                Ehart_;                    // Hartree (correction) energy
  Real                Ecor_;                     // Nonlinear correction energy
  Real                Exc_;                      // Exchange-correlation energy
  Real                EVdw_;                     // Van der Waals energy
  Real                EVxc_;                     // Exchange-correlation potential energy
  Real                Eself_;                    // Self energy due to the pseudopotential
  Real                EIonSR_;                   // Short range repulsion energy for Gaussian charge
  Real                Eext_;                     // External energy
  Real                fermi_;                    // Fermi energy
  Real                Efock_;                    // Hartree-Fock energy

  Real                totalCharge_;              // Total number of computed electron charge

  EigenSolver*        eigSolPtr_;
  PeriodTable*        ptablePtr_;


  /// @brief Needed for GGA, meta-GGA and hybrid functional calculations
  bool                isCalculateGradRho_; 

  // SCF variables
  DblNumVec           vtotNew_;
  DblNumVec           densityOld_;
  Real                scfNorm_;                 // ||V_{new} - V_{old}|| / ||V_{old}||
  Real                efreeDifPerAtom_;         // Difference between free energy and 
                                                // Harris free energy per atom
  // for Anderson iteration
  DblNumMat           dfMat_;
  DblNumMat           dvMat_;
  // extra term for Broyden mixing
  CpxNumMat           GdfMat_;
  CpxNumMat           GdvMat_;
  CpxNumMat           GcdfMat_;

  /// @brief Work array for the mixing variable in the inner iteration.
  DblNumVec           mixSave_;


  Index3  numGridWavefunctionElem_;
  Index3  numGridDensityElem_;

  std::string         SmearingScheme_;
  Int                 MP_smearing_order_;
  Real                Tsigma_;

  // Chebyshev Filtering variables
  bool Diag_SCF_PWDFT_by_Cheby_;
  Int First_SCF_PWDFT_ChebyFilterOrder_;
  Int First_SCF_PWDFT_ChebyCycleNum_;
  Int General_SCF_PWDFT_ChebyFilterOrder_;
  bool PWDFT_Cheby_use_scala_;
  bool PWDFT_Cheby_apply_wfn_ecut_filt_;


  // Do the usual Chebyshev filtering schedule or work in ionic movement mode
  Int Cheby_iondynamics_schedule_flag_;


public:

  // *********************************************************************
  // Life-cycle
  // *********************************************************************
  SCF();
  ~SCF();

  // *********************************************************************
  // Operations
  // *********************************************************************
  // Basic parameters. Density and wavefunction
  void  Setup( EigenSolver& eigSol, PeriodTable& ptable ); 
  void  Iterate();
  // Solve the linear eigenvalue problem and compute potential etc
  void  InnerSolve( Int iter ); 
  void  Update();

  void  CalculateOccupationRate ( DblNumVec& eigVal, DblNumVec& occupationRate );
  void  CalculateEnergy();
  void  CalculateHarrisEnergy();



  void  PrintState( const Int iter );

  // Mixing
  void  AndersonMix( 
      Int iter,
      Real            mixStepLength,
      std::string     mixType,
      DblNumVec&      vMix,
      DblNumVec&      vOld,
      DblNumVec&      vNew,
      DblNumMat&      dfMat,
      DblNumMat&      dvMat );

  void  KerkerPrecond(
      DblNumVec&  precResidual,
      const DblNumVec&  residual );

  void  BroydenMix(
      Int iter,
      Real            mixStepLength,
      std::string     mixType,
      DblNumVec&      vMix,
      DblNumVec&      vOld,
      DblNumVec&      vNew,
      CpxNumMat&      GdfMat,
      CpxNumMat&      GdvMat,
      CpxNumMat&      GcdfMat );

  Real RhoDdot( Complex* rho1, Complex* rho2 );
  /// @brief Update the parameters for SCF during the MD simulation
  /// This is done by modifying the global esdfParam parameters
  void UpdateMDParameters( );

  /// @brief Update the parameters for SCF during the TDDFT simulation
  /// This is done by modifying the global esdfParam parameters
  void UpdateTDDFTParameters( );

  // *********************************************************************
  // Inquiry
  // *********************************************************************
  // Energy etc.
  Real Etot() const  {return Etot_;};    
  Real Efree() const {return Efree_;};    
  Real Efock() const {return Efock_;};    

  Real Fermi() const {return fermi_;};    

  void UpdateEfock( Real Efock ) {Efock_ = Efock; Etot_ -= Efock; Efree_ -= Efock;}

  // Setup the Cheby-iondynamics flag
  void set_Cheby_iondynamics_schedule_flag(int flag){Cheby_iondynamics_schedule_flag_ = flag;}


}; // -----  end of class  SCF ----- 


} // namespace dgdft
#endif // _SCF_HPP_

