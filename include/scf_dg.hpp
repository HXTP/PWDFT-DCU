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
/// @file scf_dg.hpp
/// @brief Self consistent iteration using the DG method.
/// @date 2013-02-05
/// @date 2014-08-06 Intra-element parallelization
#ifndef _SCF_DG_HPP_ 
#define _SCF_DG_HPP_

#include  "environment.hpp"
#include  "numvec_impl.hpp"
#include  "domain.hpp"
#include  "fourier.hpp"
#include  "hamiltonian.hpp"
#include  "spinor.hpp"
#include  "periodtable.hpp"
#include  "esdf.hpp"
#include  "eigensolver.hpp"
#include  "utility.hpp"
#include  "hamiltonian_dg.hpp"
#include  "hamiltonian_dg_conversion.hpp"

namespace dgdft{

class SCFDG
{
private:
  // Control parameters
  Int                 mixMaxDim_;
  std::string         mixType_;
  Real                mixStepLength_;            
  Int                 HybridmixMaxDim_;
  std::string         HybridmixType_;
  Real                HybridmixStepLength_;
  Real                eigMinTolerance_;
  Real                eigTolerance_;
  Int                 eigMinIter_;
  Int                 eigMaxIter_;
  Real                scfInnerTolerance_;

  bool                DGHFXNestedLoop_;

  Real                scfInnerDMTolerance_;
  Real                scfInnerHamTolerance_;

  Int                 scfInnerMinIter_;
  Int                 scfInnerMaxIter_;
  Real                DFTscfInnerTolerance_;
  Int                 DFTscfInnerMinIter_;
  Int                 DFTscfInnerMaxIter_;
  Real                HybridscfInnerTolerance_;
  Int                 HybridscfInnerMinIter_;
  Int                 HybridscfInnerMaxIter_;
  Int                 PhiMaxIter_;
  Real                PhiTolerance_;

  Real                scfHFXEnergyTolerance_;
  /// @brief Criterion for convergence using Efree rather than the
  /// potential difference.
  Real                scfOuterEnergyTolerance_;  
  Real                scfOuterTolerance_;
  Int                 scfOuterMinIter_;
  Int                 scfOuterMaxIter_;
  Real                DFTscfOuterEnergyTolerance_;
  Real                DFTscfOuterTolerance_;
  Int                 DFTscfOuterMinIter_;
  Int                 DFTscfOuterMaxIter_;
  Real                HybridscfOuterEnergyTolerance_;
  Real                HybridscfOuterTolerance_;
  Int                 HybridscfOuterMinIter_;
  Int                 HybridscfOuterMaxIter_;

  Real                scfNorm_;                 // ||V_{new} - V_{old}|| / ||V_{old}||
  Int                 numUnusedState_;
  Real                SVDBasisTolerance_;
  
  Real                ecutWavefunction_;
  Real                densityGridFactor_;        
  Real                LGLGridFactor_;

  Point3              distancePeriodize_;
  // Bubble function along each dimension
  std::vector<DblNumVec>   vBubble_;

  Real                potentialBarrierW_;
  Real                potentialBarrierS_;
  Real                potentialBarrierR_;
  std::vector<DblNumVec>   vBarrier_;

  Int                 numSVDBasisTotal_;

  std::string         restartDensityFileName_;
  std::string         restartWfnFileName_;
  std::string         XCType_;
  std::string         VDWType_;
  /// @brief Same as @ref esdf::ESDFInputParam::solutionMethod
  std::string         solutionMethod_;

  /// @brief Same as @ref esdf::ESDFInputParam::diagSolutionMethod
  std::string         diagSolutionMethod_;

  std::string         SmearingScheme_;
  Int                 MP_smearing_order_;
  
  // PWDFT solver on extended element
  std::string         PWSolver_;

  // xmqin for Hybrid
  bool                isCalculateGradRho_;

  // LL: 2016/11/26 Some of the control parameters below are not
  // necessary and can be referred to directly as esdfParam.xxx

  // Chebyshev Filtering variables for PWDFT on extended element
  bool Diag_SCF_PWDFT_by_Cheby_;
  Int First_SCF_PWDFT_ChebyFilterOrder_;
  Int First_SCF_PWDFT_ChebyCycleNum_;
  Int General_SCF_PWDFT_ChebyFilterOrder_;
  bool PWDFT_Cheby_use_scala_;
  bool PWDFT_Cheby_apply_wfn_ecut_filt_;

  // PPCG for PWDFT on extended element
  bool Diag_SCF_PWDFT_by_PPCG_;

  // PEXSI parameters
#ifdef _USE_PEXSI_
  PPEXSIPlan          pexsiPlan_;
  PPEXSIPlan          plan_;
  PPEXSIOptions       pexsiOptions_;
  DistDblNumVec       VtotHist_; 
#endif

  // Variables related to Chebyshev Filtered SCF iterations for DG

  // User option variables
  bool Diag_SCFDG_by_Cheby_; // Default: 0
  bool SCFDG_Cheby_use_ScaLAPACK_; // Default: 0

  Int First_SCFDG_ChebyFilterOrder_; // Default 60
  Int First_SCFDG_ChebyCycleNum_; // Default 5

  Int Second_SCFDG_ChebyOuterIter_; // How many SCF steps for 2nd phase, default = 3
  Int Second_SCFDG_ChebyFilterOrder_; // Filter Order for 2nd phase, default = 60
  Int Second_SCFDG_ChebyCycleNum_; // Default 3 

  Int General_SCFDG_ChebyFilterOrder_; // Filter Order for general phase, default = 60
  Int General_SCFDG_ChebyCycleNum_; // Default 1

  // Internal use variables
  // Key for the eigenvector on the current processor
  Index3 my_cheby_eig_vec_key_;

  // Do the usual Chebyshev filtering schedule or work in ionic movement mode
  Int Cheby_iondynamics_schedule_flag_;
  
  // Ionic iteration related parameters
  Int scfdg_ion_dyn_iter_; // Ionic iteration number
  bool useEnergySCFconvergence_; // Whether to use energy based SCF convergence
  Real md_scf_etot_diff_tol_; // Tolerance for SCF total energy for energy based SCF convergence
  Real md_scf_eband_diff_tol_; // Tolerance for SCF band energy for energy based SCF convergence
  Real md_scf_etot_;
  Real md_scf_etot_old_;
  Real md_scf_etot_diff_;
  Real md_scf_eband_;
  Real md_scf_eband_old_; 
  Real md_scf_eband_diff_;

  // Deque for ALBs expressed on the LGL grid
  std::deque<DblNumMat> ALB_LGL_deque_;

  // Variables related to Chebyshev polynomial filtered 
  // complementary subspace iteration strategy in DGDFT
  bool SCFDG_use_comp_subspace_;
  bool SCFDG_comp_subspace_parallel_;
  bool SCFDG_comp_subspace_syrk_; // Currently only available in the parallel version
  bool SCFDG_comp_subspace_syr2k_; // Currently only available in the parallel version
  
  Int SCFDG_comp_subspace_nstates_;
  Int SCFDG_CS_ioniter_regular_cheby_freq_;
  Int SCFDG_CS_bigger_grid_dim_fac_;
  
  // LOBPCG (for top states) related options
  Int SCFDG_comp_subspace_LOBPCG_iter_;
  Real SCFDG_comp_subspace_LOBPCG_tol_;

  // CheFSI (for top states) related options
  bool Hmat_top_states_use_Cheby_;
  Int  Hmat_top_states_ChebyFilterOrder_; 
  Int  Hmat_top_states_ChebyCycleNum_; 
  double Hmat_top_states_Cheby_delta_fudge_;

  // Internal variables   
  Int SCFDG_comp_subspace_N_solve_;
  bool SCFDG_comp_subspace_engaged_;

  double SCFDG_comp_subspace_saved_a_L_; // This is for scaling the top level Chebyshev Filter
  double SCFDG_comp_subspace_trace_Hmat_; 

  DblNumVec SCFDG_comp_subspace_top_eigvals_;
  DblNumVec SCFDG_comp_subspace_top_occupations_;

  DblNumMat SCFDG_comp_subspace_start_guess_; // Used in the serial implementation 
  DblNumMat SCFDG_comp_subspace_matC_; 
  
  
  double SCFDG_comp_subspace_inner_CheFSI_lower_bound_;
  double SCFDG_comp_subspace_inner_CheFSI_upper_bound_;
  double SCFDG_comp_subspace_inner_CheFSI_a_L_;

  /// @brief The total number of processors used by PEXSI.
  /// 
  /// Let npPerPole_ = numProcRowPEXSI_ * numProcColPEXSI_, then
  ///
  /// LL 11/26/2014: In the new version of DGDFT-PEXSI with the
  /// intra-element parallelization, the pexsi communicator is given as
  /// follows:
  ///
  /// If the DG communicator is partitioned into a 2D rectangular grid
  /// as
  ///
  /// numElem * numProcPerElem 
  ///
  /// Then PEXSI uses a subset of this grid with size
  ///
  /// numProcPerPole * min(numPole, numProcPerElem)
  ///
  /// i.e. a upper-left rectangular block of the total number of
  /// processors.
  ///
  /// This greatly simplfies the procedure and the cost for data
  /// communication when the number of processors is large.
  ///
  /// This number is equal to numProcPEXSICommRow_ *
  /// numProcPEXSICommCol_
  Int                 numProcTotalPEXSI_;
  /// @brief The number of processors for each pole.
  ///
  /// This number is equal to numProcRowPEXSI_ * numProcColPEXSI_, and
  /// should be less than or equal to the number of elements.
  Int                 numProcPEXSICommCol_;
  /// @brief The number of processors for pole parallelization.
  ///
  /// This number should be set as the minimum of the number of poles,
  /// and the number of processors for each element in DG.
  Int                 numProcPEXSICommRow_;
  /// @brief Communicator used only by PEXSI.  
  ///
  MPI_Comm            pexsiComm_;
  MPI_Comm            planComm_;
  /// @brief Whether PEXSI has been initialized.
  bool                isPEXSIInitialized_;
  /// @brief The number of row processors used by PEXSI for each pole.
  Int                 numProcRowPEXSI_;
  /// @brief The number of column processors used by PEXSI for each pole.
  Int                 numProcColPEXSI_;

  Int                 inertiaCountSteps_;
  // Minimum of the tolerance for the inertia counting in the
  // dynamically adjustment strategy
  Real                muInertiaToleranceTarget_; 
  // Minimum of the tolerance for the PEXSI solve in the
  // dynamically adjustment strategy
  Real                numElectronPEXSIToleranceTarget_;

  // Physical parameters
  Real                Tbeta_;                    // Inverse of temperature in atomic unit
  Real                Tsigma_;                   // = kB * T in atomic units (i.e. = 1 / Tbeta_
  Real                EfreeHarris_;              // Helmholtz free energy defined through Harris energy functional
  Real                EfreeSecondOrder_;         // Second order accurate Helmholtz free energy 
  Real                Efree_;                    // Helmholtz free energy (KS energy functional)
  Real                Etot_;                     // Total energy (KSenergy functional)
  Real                Ekin_;                     // Kinetic energy
  Real                Ehart_;                    // Hartree energy
  Real                Ecor_;                     // Nonlinear correction energy
  Real                Exc_;                      // Exchange-correlation energy
  // Add for Hybrids xmqin
  Real                Ehfx_;                     // Hartree-Fock exchange energy
  Real                Evdw_;                     // Van der Waals energy
  Real                EVxc_;                     // Exchange-correlation potential energy
  Real                Eself_;                    // Self energy due to the pseudopotential
  Real                fermi_;                    // Fermi energy
  Real                Efock_;                    // Hartree-Fock energy

  // @brief Number of processor rows and columns for ScaLAPACK
  Int                 dmRow_;
  Int                 dmCol_;
  Int                 numProcScaLAPACK_;

  // Density matrices

  DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn>      distDMMat_;
  DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn>      distHFXMat_;
  DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn>      distDMMatSave_;
  DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn>      distEDMMat_;
  DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn>      distFDMMat_;

  PeriodTable*        ptablePtr_;

  HamiltonianDG*      hamDGPtr_;

  DistVec<Index3, EigenSolver, ElemPrtn>*   distEigSolPtr_;

  DistFourier*        distfftPtr_;

  // SCF variables

  // @brief What to mix when calling PWDFT to generate ALBs
  std::string         OutermixVariable_;
  // @brief What to mix in Inner iteration 
  std::string         InnermixVariable_;
  std::string         HybridInnermixVariable_;

  /// @brief Work array for the old mixing variable in the outer iteration.
  DistDblNumVec       mixOuterSave_;
  /// @brief Work array for the old mixing variable in the inner iteration.
  DistDblNumVec       mixInnerSave_;
  // TODO Remove dfOuterMat_, dvOuterMat_
  /// @brief Work array for the Anderson mixing in the outer iteration.
  DistDblNumMat       dfOuterMat_;
  /// @brief Work array for the Anderson mixing in the outer iteration.
  DistDblNumMat       dvOuterMat_;
  /// @brief Work array for the Anderson mixing in the inner iteration.
  DistDblNumMat       dfInnerMat_;
  /// @brief Work array for the Anderson mixing in the inner iteration.
  DistDblNumMat       dvInnerMat_;

  DistDblNumMat       cdfInnerMat_;

  /// @brief Work array for updating the local potential on the LGL
  /// grid.
  DistDblNumVec       vtotLGLSave_;

  // density matrix or Hamiltonian matrix mixing variable xmqin

  DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn>      distHMatSave_;
//  DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn>      distHMatOld_;
  DistVec<ElemMatKey, NumTns<Real>, ElemMatPrtn>      distdfInnerMat_;
  DistVec<ElemMatKey, NumTns<Real>, ElemMatPrtn>      distdvInnerMat_;
  DistVec<ElemMatKey, NumTns<Real>, ElemMatPrtn>      distcdfInnerMat_;

  DblNumMat           forceVdw_;

  Int                 scfTotalInnerIter_;       // For the purpose of Anderson mixing
  Real                scfInnerNorm_;            // ||V_{new} - V_{old}|| / ||V_{old}||
  Real                scfOuterNorm_;            // ||V_{new} - V_{old}|| / ||V_{old}||
  Real                scfInnerHamMaxDif_;          // Maximum Ham difference |M_new(i,j)-M_old(i,j)|
  Real                scfInnerDMMaxDif_;          // Maximum DM difference |M_new(i,j)-M_old(i,j)|
  Real                efreeDifPerAtom_;            

  /// @brief Global domain.
  Domain              domain_;

  Index3              numElem_;

  Index3              extElemRatio_;

  /// @brief Partition of element.
  ElemPrtn            elemPrtn_;

  Int                 scaBlockSize_;

  /// @brief Interpolation matrix from uniform grid in the extended
  /// element with periodic boundary condition to LGL grid in each
  /// element (assuming all the elements are the same).
  std::vector<DblNumMat>    PeriodicUniformToLGLMat_;
  std::vector<DblNumMat>    PeriodicUniformFineToLGLMat_;
#ifdef GPU
  cuDblNumMat cu_PeriodicUniformToLGLMat_x; 
  cuDblNumMat cu_PeriodicUniformToLGLMat_y; 
  cuDblNumMat cu_PeriodicUniformToLGLMat_z; 
#endif

  /// @brief Interpolation matrix from uniform fine grid in the extended
  /// element with periodic boundary condition to fine grid in each
  /// element (assuming all the elements are the same).
  std::vector<DblNumMat>    PeriodicGridExtElemToGridElemMat_;

  /// @brief Context for BLACS.
  Int                 contxt_;

  /// @brief Internal routines used by Chebyshev filtering
  void scfdg_hamiltonian_times_distvec(DistVec<Index3, DblNumMat, ElemPrtn>  &my_dist_vec, 
      DistVec<Index3, DblNumMat, ElemPrtn>  &Hmat_times_my_dist_vec);

  void scfdg_hamiltonian_times_distmat(DistVec<Index3, DblNumMat, ElemPrtn>  &my_dist_mat, 
      DistVec<Index3, DblNumMat, ElemPrtn>  &Hmat_times_my_dist_mat);

  void scfdg_Hamiltonian_times_eigenvectors(DistVec<Index3, DblNumMat, ElemPrtn>  &result_mat);

  double scfdg_Cheby_Upper_bound_estimator(DblNumVec& ritz_values, 
      int Num_Lanczos_Steps);

  void scfdg_Chebyshev_filter_scaled(int m, 
      double a, 
      double b, 
      double a_L);

  // These following routines are useful, for the Lanczos procedure

  // Dot product for conforming distributed vectors
  double scfdg_distvec_dot(DistVec<Index3, DblNumMat, ElemPrtn>  &dist_vec_a,
      DistVec<Index3, DblNumMat, ElemPrtn>  &dist_vec_b);

  // L2 norm
  double scfdg_distvec_nrm2(DistVec<Index3, DblNumMat, ElemPrtn>  &dist_vec_a);

  // Computes b = scal_a * a + scal_b * b for conforming distributed vectors / matrices
  // Set scal_a = 0.0 and use any vector / matrix a to obtain blas::scal on b
  // Set scal_b = 1.0 for blas::axpy with b denoting y, i.e., b = scal_a * a + b;
  void scfdg_distmat_update(DistVec<Index3, DblNumMat, ElemPrtn>  &dist_mat_a, 
      double scal_a,
      DistVec<Index3, DblNumMat, ElemPrtn>  &dist_mat_b,
      double scal_b);

  /// @brief Internal routine used by Chebyshev Filtering for ScaLAPACK based
  /// solution of the subspace problem : converts a distributed eigenvector block to ScaLAPACK format   
  void scfdg_Cheby_convert_eigvec_distmat_to_ScaLAPACK(DistVec<Index3, DblNumMat, ElemPrtn>  &my_dist_mat, 
      MPI_Comm comm,
      dgdft::scalapack::Descriptor &my_scala_descriptor,
      dgdft::scalapack::ScaLAPACKMatrix<Real>  &my_scala_mat);
 
  /// @brief Internal routines related to Chebyshev polynomial filtered 
  /// complementary subspace iteration strategy in DGDFT
  double scfdg_fermi_func_comp_subspc( DblNumVec& top_eigVals, DblNumVec& top_occ, Int num_solve, Real x);
  void scfdg_calc_occ_rate_comp_subspc( DblNumVec& top_eigVals, DblNumVec& top_occ, Int num_solve);

//  // Internal routines for MP (and GB) type smearing 
//  double low_order_hermite_poly(double x, int order); // This returns Hermite polynomials (physicist convention) of order <= 6
//  double mp_occupations(double x, int order); // This returns the occupation as a function of x = (e_i - mu) / sigma
//  double mp_entropy(double x, int order); // This returns the contribution to the electronic entropy as a function of x = (e_i - mu) / sigma
//
//  // Calculate the Methfessel-Paxton (or Gaussian Broadening) occupations for a given set of eigenvalues and a fermi-level
//  void populate_mp_occupations(DblNumVec& input_eigvals, DblNumVec& output_occ, double fermi_mu);
//
//  // Calculate the residual function for use in Fermi-level calculations using bisection method
//  double mp_occupations_residual(DblNumVec& input_eigvals, double fermi_mu, int num_solve);

public:
  // *********************************************************************
  // Life-cycle
  // *********************************************************************
  SCFDG();
  ~SCFDG();

  // *********************************************************************
  // Operations
  // *********************************************************************

  /// @brief Setup the basic parameters for initial SCF iteration.
  void  Setup( 
      HamiltonianDG& hamDG,
      DistVec<Index3, EigenSolver, ElemPrtn>&  distEigSol,
      DistFourier&   distfft,
      PeriodTable&   ptable,
      Int            contxt ); 

  /// @brief Update the basic parameters for SCF interation for MD and
  /// geometry optimization.
  void  Update( ); 

  /// @brief Main self consistent iteration subroutine.
  void  Iterate();

  /// @brief Inner self consistent iteration subroutine without
  /// correcting the basis functions.
  void  InnerIterate( Int outerIter );

  void  InnerSolver( Int outerIter );

  /// @brief Main routines for Chebyshev filtering based SCF
  void scfdg_FirstChebyStep(Int eigMaxIter,
      Int filter_order );
#ifdef GPU
  void scfdg_GeneralChebyStep_GPU(Int eigMaxIter, 
      Int filter_order );    
#endif
  void scfdg_GeneralChebyStep(Int eigMaxIter, 
      Int filter_order );    
  void set_Cheby_iondynamics_schedule_flag(int flag){Cheby_iondynamics_schedule_flag_ = flag;}
  
  void set_iondynamics_iter(int ion_iter){scfdg_ion_dyn_iter_ = ion_iter;}

  /// @brief Routines related to Chebyshev polynomial filtered 
  /// complementary subspace iteration strategy in DGDFT
  void scfdg_complementary_subspace_serial( Int filter_order );
  void scfdg_complementary_subspace_parallel( Int filter_order );
  void scfdg_complementary_subspace_compute_fullDM();
  void scfdg_compute_fullDM();

  /// @brief Update the local potential in the extended element and the element.
  void  UpdateElemLocalPotential();

  /// @brief Calculate the occupation rate and the Fermi energy given
  /// the eigenvalues
  void  CalculateOccupationRate ( DblNumVec& eigVal, DblNumVec&
      occupationRate );

  void  CalculateOccupationRateExtElem ( DblNumVec& eigVal, DblNumVec&
      occupationRate, Int npsi, Int nOccStates );

  /// @brief Interpolate the uniform grid in the periodic extended
  /// element domain to LGL grid in each element.
#ifdef GPU
  void InterpPeriodicUniformToLGL( const Index3& numUniformGrid,
      const Index3& numLGLGrid, cuDblNumMat& cu_psiUniform, cuDblNumMat& cu_psiLGL, cuDblNumVec& cu_tmp1, cuDblNumVec& cu_tmp2 );
#endif
  void InterpPeriodicUniformToLGL( const Index3& numUniformGrid,
      const Index3& numLGLGrid, const Real* psiUniform, Real* psiLGL );

  void InterpPeriodicUniformFineToLGL( const Index3& numUniformGridFine,
      const Index3& numLGLGrid, const Real* rhoUniform, Real* rhoLGL );

  /// @brief Interpolate the uniform fine grid in the periodic extended
  /// element domain to fine grid in each element.
  void InterpPeriodicGridExtElemToGridElem( const Index3& numUniformGridFineExtElem,
      const Index3& numUniformGridFineElem, const Real* rhoUniformExtElem, Real* rhoUniformElem );

  /// @brief Calculate the Kohn-Sham energy and other related energies.
  void  CalculateKSEnergy();

  /// @brief Calculate the Kohn-Sham energy and other related energies
  /// using the energy density matrix and the free energy density matrix.
  ///
  /// FIXME In order to be compatible with PPEXSIDFTDriver3, the
  /// Tr[H*DM] part is directly read from totalEnergyH, and free energy
  /// is the total energy.
  void  CalculateKSEnergyDM(
      Real totalEnergyH,
      DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn>& distEDMMat,
      DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn>& distFDMMat );

  /// @brief Calculate the Harris (free) energy.  
  ///
  /// The difference between the Kohn-Sham energy and the Harris energy
  /// is that the nonlinear correction term in the Harris energy
  /// functional must be computed via the input electron density, rather
  /// than the output electron density or the mixed electron density.
  ///
  /// Reference:
  ///
  /// [Soler et al. "The SIESTA method for ab initio order-N
  /// materials", J. Phys. Condens. Matter. 14, 2745 (2002) pp 18]
  void  CalculateHarrisEnergy();

  /// @brief Calculate the Harris (free) energy using density matrix and
  /// free energy density matrix.  
  ///
  /// FIXME In order to be compatible with PPEXSIDFTDriver3, the
  /// free energy is just not available. Free energy is the total energy
  /// @see CalculateHarrisEnergy
  void  CalculateHarrisEnergyDM(
      Real totalFreeEnergy,
      DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn>& distFDMMat );

  /// @brief Calculate the second order accurate energy that is
  /// applicable to both density and potential mixing.
  ///
  /// Reference:
  ///
  /// Research note, "On the understanding and generalization of Harris
  /// energy functional", 08/26/2013.
  void  CalculateSecondOrderEnergy();

  /// @brief Calculate Van der Waals energy and force 
  ///
  void  CalculateVDW ( Real& VDWEnergy, DblNumMat& VDWForce );

  /// @brief Print out the state variables at each SCF iteration.
  void  PrintState(  );

  /// @brief Parallel preconditioned Anderson mixing. Can be used for
  /// potential mixing or density mixing.
  void  AndersonMix( 
      Int             iter, 
      Real            mixStepLength,
      std::string     mixType,
      DistDblNumVec&  distvMix,
      DistDblNumVec&  distvOld,
      DistDblNumVec&  distvNew,
      DistDblNumMat&  dfMat,
      DistDblNumMat&  dvMat);

  void  AndersonMix2( 
      Int             iter, 
      Real            mixStepLength,
      DistDblNumVec&  distvMix,
      DistDblNumVec&  distvOld,
      DistDblNumVec&  distvNew,
      DistDblNumMat&  dfMat,
      DistDblNumMat&  dvMat);

  /// @brief Parallel preconditioned Anderson mixing. Can be used for
  /// density matrix mixing or hamiltonian mixing
  void  AndersonMix(
      Int             iter,
      Real            mixStepLength,
      std::string     mixType,
      DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn>&  distvMix,
      DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn>&  distvOld,
      DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn>&  distvNew,
      DistVec<ElemMatKey, NumTns<Real>, ElemMatPrtn>&  distdfMat,
      DistVec<ElemMatKey, NumTns<Real>, ElemMatPrtn>&  distdvMat );

  void  AndersonMix2(
      Int             iter,
      Real            mixStepLength,
      DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn>&  distvMix,
      DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn>&  distvOld,
      DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn>&  distvNew,
      DistVec<ElemMatKey, NumTns<Real>, ElemMatPrtn>&  distdfMat,
      DistVec<ElemMatKey, NumTns<Real>, ElemMatPrtn>&  distdvMat );

  void  PulayMix( 
      Int             iter, 
      Real            mixStepLength,
      DistDblNumVec&  distvMix,
      DistDblNumVec&  distvIn,
      DistDblNumVec&  distvOut,
      DistDblNumMat&  dfMat,
      DistDblNumMat&  dvMat);

  void  PulayMix(
      Int             iter,
      Real            mixStepLength,
      DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn>&  distvMix,
      DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn>&  distvIn,
      DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn>&  distvOut,
      DistVec<ElemMatKey, NumTns<Real>, ElemMatPrtn>&  distdfMat,
      DistVec<ElemMatKey, NumTns<Real>, ElemMatPrtn>&  distdvMat );

  void  BroydenMix(
      Int             iter,
      Real            mixStepLength,
      DistDblNumVec&  distvMix,
      DistDblNumVec&  distvOld,
      DistDblNumVec&  distvNew,
      DistDblNumMat&  dfMat,
      DistDblNumMat&  dvMat,
      DistDblNumMat&  cdfMat);

  void  BroydenMix(
      Int             iter,
      Real            mixStepLength,
      DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn>&  distvMix,
      DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn>&  distvOld,
      DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn>&  distvNew,
      DistVec<ElemMatKey, NumTns<Real>, ElemMatPrtn>&  distdfMat,
      DistVec<ElemMatKey, NumTns<Real>, ElemMatPrtn>&  distdvMat,
      DistVec<ElemMatKey, NumTns<Real>, ElemMatPrtn>&  distcdfMat );

  /// @brief Parallel Kerker preconditioner. Can be used for
  /// potential mixing or density mixing.
  void  KerkerPrecond(
      DistDblNumVec&  distPrecResidual,
      const DistDblNumVec&  distResidual );

  /// @brief Update the parameters for SCF during the MD simulation
  /// This is done through modifying the global esdfParam parameters
  void UpdateMDParameters( );

  void SVDLocalizeBasis ( Int iter,
                          Index3 numGridExtElem,
                          Index3 numGridElemFine,
                          Index3 numLGLGrid,
                          Spinor& psi,
                          DblNumMat& basisLGL );

  void SetupDMMix( );

  void ProjectDM ( DistDblNumMat&  Oldbasis,
                   DistDblNumMat&  Newbasis,
                   DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn>& distDMMat ); 

  // *********************************************************************
  // Inquiry
  // *********************************************************************
  Real Efree() const {return Efree_;};    

  Real Fermi() const {return fermi_;};    


  DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn>& HFXMat() {return distHFXMat_;};
  DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn>& DMMat() {return distDMMat_;};
  DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn>& DMMatSave() {return distDMMatSave_;};

}; // -----  end of class  SCFDG ----- 

} // namespace dgdft

#endif // _SCF_HPP_

