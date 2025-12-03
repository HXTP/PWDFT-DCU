/*
   Copyright (c) 2012 The Regents of the University of California,
   through Lawrence Berkeley National Laboratory.  

Authors: Chris J. Pickard and Lin Lin

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
/// @file esdf.hpp
/// @brief Electronic structure data format for reading the input data.
/// @date 2012-08-10
#ifndef _ESDF_HPP_
#define _ESDF_HPP_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <limits.h>
#include "mpi.h"
#include "domain.hpp"
#include "numvec_impl.hpp"
#include "nummat_impl.hpp"
#include "numtns_impl.hpp"
#include "tinyvec_impl.hpp"

namespace dgdft{

// Forward declaration of Atom structure in periodtable.hpp 
struct Atom;

//
// *********************************************************************
// Electronic structure data format
// *********************************************************************
namespace esdf{


/************************************************************ 
 * Main routines
 ************************************************************/
void esdf_bcast();
void esdf_key();
void esdf_init(const char *);
void esdf_string(const char *, const char *, char *);
int esdf_integer(const char *, int);
float esdf_single(const char *, float);
double esdf_double(const char *, double);
double esdf_physical(const char *, double, char *);
bool esdf_defined(const char *);
bool esdf_boolean(const char *, bool);
bool esdf_block(const char *, int *);
char *esdf_reduce(char *);
double esdf_convfac(char *, char *);
int esdf_unit(int *);
void esdf_file(int *, char *, int *);
void esdf_lablchk(char *, char *, int *);
void esdf_die(char *);
void esdf_warn(char *);
void esdf_close();

/************************************************************ 
 * Utilities
 ************************************************************/
void getaline(FILE *, char *);
void getlines(FILE *fp, char **);
char *trim(char *);
void adjustl(char *,char *);
int len_trim(char *);
int indexstr(char *, char *);
int indexch(char *, char);
int countw(char *, char **, int);
char *strlwr(char *);
char *strupr(char *);


// *********************************************************************
// Input interface
// *********************************************************************


/// @struct ESDFInputParam
/// @brief Main structure containing input parameters for the
/// electronic structure calculation.
struct ESDFInputParam{
  /// @brief Global computational domain.
  ///
  /// Not an input parameter by the user.
  Domain              domain;
  /// @brief Types and positions of all atoms in the global
  /// computational domain.
  ///
  /// Not an input parameter by the user.
  std::vector<Atom>   atomList;

  /// @brief Number of atom types
  Int                 numAtomType;

  /// @brief Mixing maximum dimension.
  ///
  /// Default: 9
  ///
  /// This parameter is relevant for Anderson mixing.
  Int                 mixMaxDim;
  Int                 DGHybridmixMaxDim;
  /// @brief Mixing type for self-consistent field iteration.
  ///
  /// Default: anderson
  ///
  /// - = "anderson"           : Anderson mixing
  /// - = "kerker+anderson"    : Anderson mixing with Kerker
  /// preconditioner
  ///
  /// @todo Preconditioner better than Kerker mixing.
  std::string         mixType;
  std::string         DGHybridmixType;
  bool                DGHFXNestedLoop;   
  /// @brief Which variable to mix when calling PWDFT to generate ALBs
  ///
  /// Default: potential
  ///
  /// - = "density"            : Density mixing
  /// - = "potential"          : Potential mixing
  std::string         mixVariable;
  /// @brief Which variable to mix in Inner iteration
  ///
  /// Default: potential
  ///
  /// - = "density"            : Density mixing
  /// - = "potential"          : Potential mixing
  /// - = "hamiltonian"        : Hamiltonian mixing
  /// - = "densitymatrix"      : Density matrix mixing
  std::string         InnermixVariable;
  std::string         HybridInnermixVariable;
  /// Default: potential
  ///
  /// - = "density"            : Density mixing
  /// - = "potential"          : Potential mixing;
  std::string         OutermixVariable;

  /// @brief Coefficient in front of the preconditioned residual to be
  /// mixed with the mixing variable in the previous step.
  ///
  /// Default: 0.8
  ///
  /// For metallic systems or small gapped semiconductors,
  /// mixStepLength is often needed to be smaller than 0.1.  In such
  /// case, a better preconditioner such as Kerker preconditioner can
  /// be helpful.
  Real                mixStepLength;            
  Real                DGHybridmixStepLength;            

  /// @brief Tolerance for inner %SCF iteration in DG calculation.
  ///
  /// Default: 1e-4
  ///
  /// @note When scfInnerMaxIter = 1 (which is used most of the cases
  /// in the current version), this parameter is not useful.
  Real                scfInnerTolerance;
  Real                DFTscfInnerTolerance;
  Real                HybridscfInnerTolerance;

  Real                scfInnerHamTolerance;
  Real                scfInnerDMTolerance;
  /// @brief Tolerance for outer %SCF iteration in DG calculation.
  ///
  /// Default: 1e-6
  ///
  /// The DG calculation stops when
  ///
  /// \f[
  /// \frac{\Vert v^{(k)} - v^{(k-1)} \Vert}{\Vert{v^{(k-1)}}\Vert} <
  /// scfOuterTolerance
  /// \f]
  ///
  /// where the variable \f$v\f$ can be the density 
  /// (mixVariable = "density") or potential 
  /// (mixVariable = "potential").
  Real                scfOuterTolerance;
  Real                DFTscfOuterTolerance;
  Real                HybridscfOuterTolerance;
  /// @brief The DG calculation stops when the difference between free
  /// energy and Harris energy per atom is less than scfOuterEnergyTolerance.
  ///
  /// Default: 1e-4
  Real                scfOuterEnergyTolerance;
  Real                DFTscfOuterEnergyTolerance;
  Real                HybridscfOuterEnergyTolerance;
  /// @brief Minimum number of inner %SCF iterations
  ///
  /// Default: 1
  Int                 scfInnerMinIter;
  Int                 DFTscfInnerMinIter;
  Int                 HybridscfInnerMinIter;

  /// @brief Maximum number of inner %SCF iterations
  ///
  /// Default: 1
  Int                 scfInnerMaxIter;
  Int                 DFTscfInnerMaxIter;
  Int                 HybridscfInnerMaxIter;
  /// @brief Minimum number of outer %SCF iterations
  ///
  /// Default: 3
  Int                 scfOuterMinIter;
  Int                 DFTscfOuterMinIter;
  Int                 HybridscfOuterMinIter;
  /// @brief Maximum number of outer %SCF iterations
  ///
  /// Default: 30
  Int                 scfOuterMaxIter;
  Int                 DFTscfOuterMaxIter;
  Int                 HybridscfOuterMaxIter;
  
  /// @brief From which ion iteration to engage energy based convergence in MD
  /// 
  /// Default: ionMaxIter + 1
  Int MDscfEnergyCriteriaEngageIonIter;
  /// @brief Maximum number of outer SCF iterations in MD
  /// Currently this is interperetd slightly differently in DGDFT and PWDFT  
  /// In DGDFT, this number goes into effect once Energy based SCF convergence is activated
  /// Default: the same as scfOuterMaxIter
  Int                MDscfOuterMaxIter;
  /// @brief Etot tolerance in Energy based convergence
  /// The difference in Etot should be less than this in energy based SCf convergence
  Real               MDscfEtotdiff;
  /// @brief Eband tolerance in Energy based convergence
  /// The difference in Eband should be less than this in energy based SCf convergence
  Real               MDscfEbanddiff;
  
  
  /// @brief Maximum number of iterations for hybrid functional
  /// iterations.
  /// 
  /// Default: 10
  Int                 scfPhiMaxIter;
  Int                 HybridPhiMaxIter;

  /// @brief Maximum number of iterations for hybrid functional
  /// iterations in MD
  /// 
  /// Default: the same as scfPhiMaxIter
  Int                 MDscfPhiMaxIter;
  /// @brief Tolerance for hybrid functional iterations using Fock
  /// energy
  ///
  /// Default: 1e-6
  Real                scfPhiTolerance;

  Real                HybridPhiTolerance;
  /// @brief Mixing type for performing hybrid functional calculations.
  ///
  /// Default: nested
  ///
  /// - = "nested"             : Standard nested two loop procedure
  /// - = "scdiis"             : Selected column DIIS
  /// - = "pcdiis"             : Projected commutator DIIS
  ///
  std::string         hybridMixType;

  std::string         DGhybridMixType;

  Real                DGHFXEnergyTolerance;

  bool                isUseLIBXC;

  /// @brief Whether to initialize hybrid functional in the initial
  /// step.
  ///
  /// Default: 0
  bool                isHybridActiveInit;
  /// @brief Whether to update the ACE operator twice in PCDIIS.
  ///
  /// Default: 1
  ///
  bool                isHybridACETwicePCDIIS;
  /// @brief Whether to use the adaptively compressed exchange (ACE)
  /// formulation for hybrid functional.
  ///
  /// Default: 1
  bool                isHybridACE;
  /// @brief Whether to use the density fitting formalism for
  /// hybrid functional. Currently this must be used with the ACE
  /// formulation.
  ///
  /// Default: 0
  bool                isHybridDF;
  bool                isHybridDFTDDFT;
  bool                isDGHFISDF;
  bool                DGHF_ISDF_Scalapack;

  /// @brief Type for selecting points in density fitting
  ///
  /// Default: "QRCP"
  ///
  /// Currently two types (QRCP and Kmeans) are supported.
  std::string         hybridDFType;

  /// @brief the method to calculate the weight function for complex kmeans
  /// Default: "Add"  psi^2 + phi^2 
  /// Another type : "Multi" (psi^2)(phi^2)
  std::string         hybridDFKmeansWFType;

  /// @brief when hybridDFKmeansWFType==Add, weight function = psi^alpha + phi^alpha
  /// Default: "2.0"
  /// option: "0.5" "1.0" "1.5" "2.0" "2.5" "3.0" "4.0"
  Real         hybridDFKmeansWFAlpha;
 
  /// @brief The tolerance used in K-means method of density fitting
  ///
  /// Default: 1e-3
  Real                hybridDFKmeansTolerance;
  
  /// @brief The max number of iterations used in K-means method of density fitting
  ///
  /// Default: 99
  Int                 hybridDFKmeansMaxIter;
  
  /// @brief Density fitting uses numMu * numStateTotal number of
  /// states for hybrid calculations with density fitting.
  ///
  /// Default: 6.0
  Real                hybridDFNumMu;
  Real                DGHFISDFNumMu;
  /// @brief Density fitting uses numGaussianRandom * numMu * numStateTotal number of
  /// states for GaussianRandom.
  ///
  /// Default: 6.0
  Real                hybridDFNumGaussianRandom;
  
  /// @brief Density fitting uses this number of cores in ScaLAPACAL
  ///
  /// Default: mpisize
  Int                 hybridDFNumProcScaLAPACK;
  
  /// @brief Density fitting uses the tolerance to remove the matrix element
  ///
  /// Default: 1e-20
  Real                hybridDFTolerance;
  
  /// @brief The blocksize of cores in ScaLAPACAL
  ///
  /// Default: 32
  Int                 BlockSizeScaLAPACK;


  /// @brief Treatment of the divergence term in hybrid functional
  /// calculation.
  ///
  /// Default: 1
  ///
  /// - 0    : No regularization
  /// - 1    : Gygi-Baldereschi regularization
  Int                 exxDivergenceType;


  /// @brief Tolerance for minimum of the residual that should be
  /// reached by the eigensolver
  ///
  /// Default: 1e-3
  /// 
  /// Currently if eigMinTolerance is not reached, the LOBPCG
  /// iterations continue regardless of whether eigMaxIter is reached
  Real                eigMinTolerance;

  /// @brief Tolerance for the eigenvalue solver
  ///
  /// Default: 1e-6
  ///
  /// Currently the LOBPCG method is used as the eigenvalue solver for
  /// obtaining the adaptive local basis functions, and eigTolerance
  /// controls the tolerance for the LOBPCG solver.  
  ///
  /// In the case when the eigensolver tolerance is tunned dynamically
  /// (see 
  /// @ref dgdft::esdf::ESDFInputParam::isEigToleranceDynamic "isEigToleranceDynamic"), the tolerance for
  /// the eigensolver is controlled dynamically and can be larger than
  /// eigTolerance.
  Real                eigTolerance;
  /// @brief Minimum number of iterations for the eigensolver.
  ///
  /// Default: 1
  Int                 eigMinIter;
  /// @brief Maximum number of iterations for the eigensolver.
  ///
  /// Default: 10
  Int                 eigMaxIter;
  /// @brief Tolerance for thresholding the adaptive local basis functions.
  ///
  /// Default: 1e-6
  ///
  /// The adaptive local basis functions restricted to the element are
  /// not orthogonal and may be linearly dependent.  An local SVD
  /// decomposition is performed to eliminate the linearly dependent
  /// modes with tolerance SVDBasisTolerance.
  Real                SVDBasisTolerance;


  
  /// @brief Whether to use the sum of atomic density as the initial
  /// guess for electron density.
  ///
  /// Currently only works for ONCV pseudopotential and not for HGH
  /// pseudopotential.
  /// 
  /// Default: 1
  bool                isUseAtomDensity;

  /// @brief Whether to use the saved electron density as the start.
  ///
  /// Default: 0
  bool                isRestartDensity;
  /// @brief Whether to output the total potential.
  ///
  /// Default: 0
  ///
  /// When isOutputPotential = 1, files POT_xxx_yyy will be generated,
  /// where by default xxx is the mpirank (starting from 0), and yyy
  /// is mpisize.
  bool                isOutputPotential;
  /// @brief Whether to use the saved basis functions in extended
  /// element as the start.
  ///
  /// Default: 0
  bool                isRestartWfn;
  /// @brief Whether to output the electron density.
  ///
  /// Default: 1
  ///
  /// When isOutputDensity = 1, files DEN_xxx_yyy will be generated,
  /// where by default xxx is the mpirank (starting from 0), and yyy
  /// is mpisize.
  ///
  /// This option is needed to restart the electron density using 
  /// @ref dgdft::esdf::ESDFInputParam::isRestartDensity "isRestartDensity".
  bool                isOutputDensity;
  /// @brief Whether to output the wavefunctions in the extended
  /// element.
  ///
  /// Default: 0
  ///
  /// When isOutputWfn = 1, the approximate eigenvectors in the
  /// extended element are given in the output, in the form
  /// WFN_xxx_yyy, where by default xxx is the mpirank (starting
  /// from 0), and yyy is mpisize.
  bool                isOutputWfn;
  /// @brief Whether to output the wavefunctions in the element on LGL
  /// grid.
  ///
  /// Default: 0
  ///
  /// This is mainly for debugging and visualization purpose and is
  /// not commonly used.
  bool                isOutputALBElemLGL;
  /// @brief Whether to output the wavefunctions in the element on
  /// uniform grid.
  ///
  /// Default: 0
  ///
  /// This is mainly for debugging and visualization purpose and is
  /// not commonly used.
  bool                isOutputALBElemUniform;
  /// @brief Whether to output the wavefunctions in the extended
  /// element.
  ///
  /// Default: 1
  ///
  /// When isOutputWfnExtElem = 1, the approximate eigenvectors in the
  /// extended element are given in the output, in the form
  /// WFNEXT_xxx_yyy, where by default xxx is the mpirank (starting
  /// from 0), and yyy is mpisize.
  bool                isOutputWfnExtElem;
  /// @brief Whether to output the potential in the extended
  /// element.
  ///
  /// Default: 0
  ///
  /// This is mainly for debugging and visualization purpose and is
  /// not commonly used.
  bool                isOutputPotExtElem;
  /// @brief Whether to output the coefficient for eigenvectors
  /// 
  /// Default: 0
  ///
  /// This is only valid for diagonalization based methods.
  bool                isOutputEigvecCoef;
  /// @brief Whether to calculate a posteriori error estimator for
  /// each %SCF iteration.
  ///
  /// Default: 0
  ///
  /// If this is 0, then the a posteriori error estimator is given in
  /// the output after the %SCF iteration is finished.
  ///
  /// @todo Sharp a posteriori error estimate than the current version.
  bool                isCalculateAPosterioriEachSCF; 
  /// @brief Whether to calculate the force each %SCF iteration.
  ///
  /// @todo This is obsolete and should be removed.
  ///
  /// Default: 1
  bool                isCalculateForceEachSCF; 
  /// @brief Whether to output the DG Hamiltonian matrix in each %SCF
  /// iteration.
  ///
  /// Default: 0
  ///
  /// If isOutputHMatrix = 1, the H matrix is output in the file
  /// `H.csc` using the compressed sparse column (CSC) binary format.
  bool                isOutputHMatrix;


  /// @brief Temperature in the unit of Kelvin
  Real                temperature;
  /// @brief Inverse of temperature.
  ///
  /// Default: 1.0 / (100 K * k_b)
  ///
  /// This parameter is not controlled directly, but through 
  /// "Temperature" in the input file, in the unit of Kelvin.
  Real                Tbeta;       
  /// @brief Number of empty states for finite temperature
  /// calculation.
  ///
  /// Default: 0
  ///
  /// This parameter must be larger than 0 for small gapped systems or
  /// relatively high temperature calculations.
  Int                 numExtraState;
  /// @brief Some states for the planewave solver are unused in order
  /// to accelerate the convergence rate of the eigensolver.
  ///
  /// Default: 0
  Int                 numUnusedState;
  /// @brief Whether to control the tolerance of the eigensolver
  /// dynamically.
  ///
  /// Default: 1
  ///
  /// When isEigToleranceDynamic = 1, the tolerance for the
  /// eigensolver is controlled dynamically and is related to the
  /// error in the current %SCF iteration.  The lower limit of the
  /// tolerance is controlled by
  /// @ref dgdft::esdf::ESDFInputParam::eigTolerance "eigTolerance".
  bool                isEigToleranceDynamic;
  /// @brief File for storing the information of the pseudopotential.
  ///
  /// Default: "HGH.bin"
  ///
  /// The pseudopotential file is currently only for the
  /// Hartwigsen-Goedecker-Hutter pseudopotential, and is generated by
  /// the utility subroutine HGH.m.
  ///
  /// @note Only the master processor (mpirank == 0) reads this table,
  /// and the information is broadcast to other processors.
  std::string         periodTableFile;

  /// @brief File for storing the information of the pseudopotential.
  ///
  /// Default: None
  ///
  /// In the current PWDFT code, we use UPF file pseudopotential, which 
  /// is downloaded from QE website.
  ///
  /// @note Only the master processor (mpirank == 0) reads this table,
  /// and the information is broadcast to other processors.
  std::vector<std::string>    pspFile;
 
  /// @brief Type of the pseudopotential
  ///
  /// Default: "HGH"
  ///
  /// Currently HGH and ONCV are the only supported pseudopotential format.
  std::string         pseudoType;

  /// @brief Whether to use the VLocal generated from the
  /// pseudopotential, and use Gaussian pseudocharge as compensation
  /// charge
  ///
  /// @todo This option will become obsolete in the future when
  /// pseudopotential is read from text files.
  bool                isUseVLocal;

  /// @brief Solver for the planewave problem.  
  ///
  /// @todo out-of-date description.  BLOPEX is to be removed.
  ///
  /// - = "modified_blopex"    : BLOPEX package (modified in the
  ///                            external/ directory using the LOBPCG
  ///                            method but with BLAS3 for linear
  ///                            algebra operations.  (default)
  /// Default: "LOBPCG"
  ///
  /// - = "LOBPCG"       
  /// - = "LOBPCGScaLAPACK"       
  /// - = "PPCG"       
  /// - = "PPCGScaLAPACK"       
  /// - = "CheFSI"       
  std::string         PWSolver;            

  /// @brief Subblock size used in PPCG in PWDFT.  
  ///
  /// Default: "1"
  ///
  /// - = "1"       
  /// - = "Nband+Extra_States" : Equivalent to LOBPCG      
  Int                 PPCGsbSize;            

  /// @brief Method for solving the projected problem in the adaptive
  /// local basis set.
  ///
  /// Default: "diag"
  ///
  /// - = "diag"      : Diagonalization method using ScaLAPACK. 
  /// - = "pexsi"     : Pole expansion and selected inversion method.
  ///                   This option needs to turn on the macro -DPEXSI
  ///                   to support the libraries.
  std::string         solutionMethod; 
  
   /// @brief Method for solving the projected problem in the adaptive
  /// local basis set within the "diag" method
  ///
  /// Default: "scalapack"
  ///
  /// - = "elpa"      : use ELSI-ELPA to diag the H matrix
  /// - = "scalapack" : use scalapack to diag the H matrix
  std::string         diagSolutionMethod; 
  
  std::string         smearing_scheme;
  /// @brief smearing scheme for fractional occupations 
  /// options : FD - Fermi-Dirac distribution
  ///           GB - Gaussian Broadening or Methfessel / Paxton of order 0
  ///           MP - Methfessel-Paxton smearing - order 2 default, up to order 3 supported
  
  /// @brief Type of the exchange-correlation functional.
  ///
  /// Default: "XC_LDA_XC_TETER93"
  ///
  /// The exchange-correlation functional is implemented using the
  /// libxc package. Currently only the LDA and GGA xc functionals is
  /// supported.
  std::string         XCType;
  /// @brief Type of the van der Waals correction.
  ///
  /// Default: "DFT-D2"
  ///
  /// Currently only the DFT-D2 correction is supported.
  std::string         VDWType;

  // DG related
  /// @brief Number of elements along x,y,z directions.
  ///
  /// Default: (1,1,1)
  Index3              numElem;
  /// @brief Number of uniform grids along x,y,z directions in the
  /// extended element.
  ///
  /// This is not directly controlled by the user, but through 
  /// @ref dgdft::esdf::ESDFInputParam::ecutWavefunction "ecutWavefunction".
  Index3              numGridWavefunctionElem;
  /// @brief Number of uniform grids for representing the density
  /// along x,y,z directions in the element.
  ///
  /// The current implementation uses dual-grid implementation, which
  /// uses a denser grid to represent the electron density and
  /// potential than that of the wavefunction on the uniform grid.
  /// This parameter is controlled by
  /// @ref dgdft::esdf::ESDFInputParam::ecutWavefunction "ecutWavefunction"
  /// and
  /// @ref dgdft::esdf::ESDFInputParam::densityGridFactor "densityGridFactor".
  Index3              numGridDensityElem;
  /// @brief Number of Legendre-Gauss-Lobatto (LGL) grids for
  /// representing the basis functions along x,y,z directions in the
  /// element.
  ///
  /// This parameter is controlled by
  /// @ref dgdft::esdf::ESDFInputParam::ecutWavefunction "ecutWavefunction"
  /// and
  /// @ref dgdft::esdf::ESDFInputParam::LGLGridFactor "LGLGridFactor".
  Index3              numGridLGL;
  /// @brief Penalty parameter 
  ///
  /// Default: 100.0
  ///
  /// The same penalty parameter is applied to all faces.
  ///
  /// @todo The automatic choice of penalty parameter is to be
  /// implemented.
  Real                penaltyAlpha;
  /// @brief Number of adaptive local basis functions in each element.
  ///
  /// Default: Must be provided by the user.
  ///
  /// Dimension: numElem[0] * numElem[1] * numElem[2]
  ///
  /// @note In the current implementation, the actual number of basis
  /// functions used in the element (i,j,k) is **no more than**
  /// numALBElem(i,j,k)-numUnusedState.
  IntNumTns           numALBElem;
  /// @brief Block size for ScaLAPACK.
  ///
  /// Default: 16
  ///
  /// Only used when ScaLAPACK is invoked.
  Int                 scaBlockSize;

  // extra electron
  // Default: 0
  Int                 extraElectron;

  // Add a potential barrier in the extended element
  bool                isPotentialBarrier;
  Real                potentialBarrierW;
  Real                potentialBarrierS;
  Real                potentialBarrierR;

  // Periodization of the potential in the extended element
  bool                isPeriodizePotential;
  Point3              distancePeriodize;    

  /// @brief Kinetic energy cutoff for the wavefunction on the uniform
  /// grid.
  ///
  /// Default: 10.0 Ha
  ///
  /// The number of uniform grids for the wavefunction along each
  /// direction i (i=x,y,z) is given by the formula
  /// \f[
  ///    N_i = \sqrt{2*ecutWavefunction}*L_i
  /// \f]
  /// where \f$L_i\f$ is the dimension of the domain
  /// along the direction i. The domain can be the global domain,
  /// extended element or element.
  Real                ecutWavefunction;

  std::string         FFTtype;
  /// @brief The ratio between the number of grids for the density and
  /// the wavefunction in the uniform grid along each dimension.
  ///
  /// Default: 2.0
  ///
  /// The number of uniform grids for the density and potential along
  /// each direction i (i=x,y,z) is given by the formula
  /// \f[
  ///    N_i = densityGridFactor * \sqrt{2*ecutWavefunction}*L_i
  /// \f]
  /// where \f$L_i\f$ is the dimension of the domain
  /// along the direction i. The domain can be the global domain,
  /// extended element or element.
  Real                densityGridFactor;
  /// @brief The ratio between the number of LGL grid and uniform grid
  /// for the basis functions along each dimension.
  ///
  /// Default: 2.0
  ///
  /// The number of LGL grids along each direction i (i=x,y,z) is
  /// given by the formula
  /// \f[
  ///    N_i = LGLGridFactor * \sqrt{2*ecutWavefunction}*L_i
  /// \f]
  /// where \f$L_i\f$ is the dimension of the domain
  /// along the direction i. The domain can be the global domain,
  /// extended element or element.
  Real                LGLGridFactor;

  /// @brief The interp factor for Gaussian function for generating 
  /// the transfer matrix from LGL grid to uniform grid on each
  /// element with the Gaussian convolution interpolation method. 
  /// 
  /// Default: 4.0
  Real                GaussInterpFactor;

  /// @brief The sigma value for Gaussian function for generating 
  /// the transfer matrix from LGL grid to uniform grid on each
  /// element with the Gaussian convolution interpolation method. 
  /// 
  /// Default: 0.001
  Real                GaussSigma;

  /// @brief Number of processors for distributed FFT.
  ///
  Int                 numProcDistFFT;

  /// @brief Number of processors used by ScaLAPACK.
  ///
  /// Default: mpisize
  Int                 numProcScaLAPACK;

  /// @brief Number of processors used by ScaLAPACK in the PW part
  ///
  /// Default: mpisize for PWDFT. mpisizeRow for DGDFT.
  Int                 numProcScaLAPACKPW;

  // PEXSI
  /// @brief Number of processors in the row communication group for
  /// each pole.
  ///
  /// Default: 1
  Int                 numProcRowPEXSI;
  /// @brief Number of processors in the column communication group for
  /// each pole.
  ///
  /// Default: 1
  Int                 numProcColPEXSI;

  /// @brief Number of terms in the pole expansion.
  /// 
  /// Default: 60
  Int                 numPole;
  /// @brief Number of processors for PARMETIS/PT-SCOTCH.
  /// 
  /// Default: 1
  Int                 npSymbFact;
  /// @brief Spectral gap.
  ///
  /// Default: 0.0 Ha
  ///
  /// Setting this value to be 0.0 works for both metallic and
  /// insulating systems. A correct and larger energy gap can reduce
  /// the number of poles.
  Real                energyGap;
  /// @brief Spectral radius of the (H,S) pencil.
  ///
  /// Default: 100.0 Ha
  ///
  /// Often in practice this estimate does not need to be sharp.
  Real                spectralRadius;
  /// @brief Ordering strategy for factorization and selected
  /// inversion.  
  ///
  /// Default: 0
  ///
  /// - = 0   : Parallel ordering using ParMETIS/PT-SCOTCH (PARMETIS
  ///   option in SuperLU_DIST).
  /// - = 1   : Sequential ordering using METIS (METIS_AT_PLUS_A
  ///   option in SuperLU_DIST).
  /// - = 2   : Multiple minimum degree ordering (MMD_AT_PLUS_A
  ///   option in SuperLU_DIST).
  Int                 matrixOrdering;
  /// @brief Number of %SCF iterations before inertia counting is
  /// turned off.
  ///
  /// Default: 10
  Int                 inertiaCountSteps;
  /// @brief Maximum number of PEXSI iteration.
  ///
  /// Default: 5
  Int                 maxPEXSIIter;
  /// @brief Estimate of the lower bound of the chemical potential.
  ///
  /// Default: -2.0 Ha
  Real                muMin;
  /// @brief Estimate of the upper bound of the chemical potential.
  ///
  /// Default: 2.0 Ha
  Real                muMax;
  /// @brief Tolerance for the total number of electrons in the PEXSI
  /// iteration.
  /// 
  /// Default: 0.001
  Real                numElectronPEXSITolerance;
  /// @brief Tolerance for the chemical potential in the inertia
  /// counting procedure.
  ///
  /// Default: 0.05 Ha
  Real                muInertiaTolerance;
  /// @brief Step length for expanding the chemical potential
  /// interval.
  ///
  /// Default: 0.3 Ha
  ///
  /// If the chemical potential is not in the prescribed interval, the
  /// interval is then expanded by the prescribed step length.
  Real                muInertiaExpansion;
  /// @brief Safe guard for the chemical potential update.
  ///
  /// Default: 0.05 Ha
  ///
  /// If the difference of chemical potential between consequetive
  /// steps of the PEXSI iteration exceeds the safe guard value, the
  /// value is not trusted and the inertia count procedure is
  /// re-invoked.
  Real                muPEXSISafeGuard;

  /// @brief Maximum number of steps for geometry optimization /
  /// molecular dynamics
  ///
  /// Default: 0 
  Int                 ionMaxIter;

  /// @brief PEXSI method 1: original pole exapansion 2: Moussa Expansion
  ///
  /// Default: 2
  Int                 pexsiMethod; 

  /// @brief PEXSI method 1: original pole exapansion 2: Moussa Expansion
  ///
  /// Default: 2
  Int                 pexsiNpoint; 

  /// @brief Mode for geometry optimization and molecular dynamics
  ///
  /// Default: NONE
  ///
  /// Geometry optimization routine:
  ///
  /// ionMove = bb               : BarzilaiBorwein 
  ///           cg               : conjugate gradient
  ///           bfgs             : BFGS
  ///           fire             : FIRE optimization method
  ///
  /// Molecular dynamics routine:
  ///
  /// ionMove = verlet           : Velocity-Verlet (NVE)
  ///           nosehoover1      : Nose-Hoover method with chain
  ///                              level 1
  ///           langevin         : Langevin dynamics
  std::string         ionMove;


  /// @brief Maximum force for geometry optimization
  ///
  /// Default: 0.001 
  Real                geoOptMaxForce;
  
  Real                geoOpt_NLCG_sigma; // Line search step length parameter in NLCG for Geo Opt : Default = 0.5
  
  // These variables are related to the FIRE optimizer
  Int FIRE_Nmin; // Set to 10 by default
  Real FIRE_dt; // Set to 40.0 a.u. (= 1 femtosecond) by default
  Real FIRE_atomicmass; // Set to 4.0 by default

  /// @brief Time step for MD simulation.
  ///
  /// Default: 50.0
  Int                                    MDTimeStep; 
  /// @brief Extrapolation type for updating the density or wavefunction
  ///
  /// Default: "linear"
  /// 
  /// = "linear"        : 1-step extrapolation 
  /// = "quadratic"     : 2-step extrapolation
  /// = "aspc2"         : 2-step ASPC extrapolation
  /// = "aspc3"         : 3-step ASPC extrapolation
  /// = "xlbomd"        : wavefunction extrapolation by using XLBOMD
  ///
  std::string         MDExtrapolationType;
  /// @brief Extrapolation variable
  ///
  /// Default: "density"
  ///
  /// = "density"        : density extrapolation
  /// = "wavefun"        : wavefunction extrapolation
  ///                      currently only available in PWDFT. The
  ///                      density is constructed from the
  ///                      wavefunctions
  std::string         MDExtrapolationVariable;
  /// @brief Temperature for ion.
  ///
  /// Default: K
  Real                ionTemperature;       
  /// @brief Inverse of ionTemperature.
  ///
  /// Default: 1.0 / (100 K * k_b)
  ///
  /// This parameter is not controlled directly, but through 
  /// "ionTemperature" in the input file, in the unit of Kelvin.
  Real                TbetaIonTemperature;       
  /// @brief Mass for Nose-Hoover thermostat
  ///
  /// Default: 10.0
  Real                qMass;                                
  /// @brief Dampling factor for Langevin theromostat
  ///
  /// Default: 0.01
  Real                langevinDamping;
  /// @brief Kappa value of XL-BOMD
  ///
  /// Default: 2.0
  Real                kappaXLBOMD;
  /// @brief Whether to use the previous position
  ///
  /// Default: 0
  bool                isRestartPosition;
  /// @brief Whether to use the previous velocity and thermostat
  ///
  /// Default: 0
  bool                isRestartVelocity;
  /// @brief Whether to output position information stored in
  /// lastPos.out
  ///
  /// Default: 1
  bool                isOutputPosition;
  /// @brief Whether to output velocity and thermostat information
  /// stored in lastVel.out
  ///
  /// Default: 1
  bool                isOutputVelocity;
  /// @brief Output the atomic position in XYZ format. Used in MD
  /// simulation and geometry optimization. Outputs MD.xyz. This
  /// only stores the finished configuration.
  ///
  /// Default: 1
  bool                isOutputXYZ;


  // TDDFT parameters
  /// @brief Whether to use the saved WFN, Density, Velocity to restart
  ///
  /// FIXME This option is currently not used
  ///
  /// Default: 0
  bool                save4RestartTDDFT;


  /// @brief Maximum number of outer SCF iterations in TDDFT
  /// This is usually used in PWDFT  
  Int                restartTDDFTStep;
  
  /// @brief Maximum number of outer SCF iterations in TDDFT
  /// This is usually used in PWDFT  
  Int                TDDFTDiisMaxIter;

  /// @brief Maximum number of iterations for hybrid functional
  /// iterations in TDDFT
  /// 
  /// Default: the same as PhiMaxIter
  Int                 TDDFTPhiMaxIter;

  /// @brief use TDDFT or not
  /// 
  /// Default: 1
  bool                isTDDFT;

  /// @brief use TDDFT ehrenfest dynamics
  /// 
  /// Default: 1
  bool                isTDDFTEhrenfest;


  /// @brief use TDDFT  Vexternal or not
  /// 
  /// Default: 1
  bool                isTDDFTVext;

  /// @brief calculate TDDFT dipole or not
  /// 
  /// Default: 1
  bool                isTDDFTDipole;


  /// @brief use TDDFT  Vexternal polorization in the X direction.
  /// 
  /// Default: 1.0
  Real TDDFTVextPolx;

  /// @brief use TDDFT  Vexternal polorization in the X direction.
  /// 
  /// Default: 0.0
  Real TDDFTVextPoly;

  /// @brief use TDDFT  Vexternal polorization in the X direction.
  /// 
  /// Default: 0.0
  Real TDDFTVextPolz;

  /// @brief use TDDFT Vexternal Frequency.
  /// 
  /// Default: 18.0/27.211385
  Real TDDFTVextFreq;

  /// @brief use TDDFT Vexternal Phase
  /// 
  /// Default: 0.0
  Real TDDFTVextPhase;

  /// @brief use TDDFT Vexternal Amplitude 
  /// 
  /// Default: 0.0194
  Real TDDFTVextAmp;

  /// @brief use TDDFT Vexternal T0
  /// 
  /// Default: 13.6056925
  Real TDDFTVextT0;

  /// @brief use TDDFT Vexternal Tau
  /// 
  /// Default: 13.6056925
  Real TDDFTVextTau;

  /// @brief auto save TDDFT WFN and DEN 
  /// 
  /// Default: 20
  Int  TDDFTautoSaveSteps;

  /// @brief types of external fields
  ///
  /// Default: gaussian 
  ///
  /// - = "guassian"             : Gaussian type of External field
  /// - = "constant"             : Constant type of External field
  /// - = "sinsq"                : sin type of the external field
  /// - = "erf"                  : Exp type of the external field
  /// - = "kick "                :                            
  ///
  std::string        TDDFTVextEnv;

  /// @brief types of TDDFT methods
  ///
  /// Default: PTTRAP
  ///
  /// - = "PTTRAP"       : implicit Parallel transport evolution with trapezoidal rule
  /// - = "RK4"          :  Explicit Runge-Kutta 4th order method
  ///
  std::string        TDDFTMethod;

  /// @brief TDDFT delta T
  /// 
  /// Default: 1.0 atomic unit
  Real TDDFTDeltaT;

  /// @brief TDDFT total T
  /// 
  /// Default: 40.0 atomic unit
  Real TDDFTTotalT;

  /// @brief TDDFT Krylov Max iteration number
  /// 
  /// Default: 30
  Int  TDDFTKrylovMax;

  /// @brief TDDFT SCF Max iteration number
  /// 
  /// Default: 30
  Int  TDDFTMaxIter;

  /// @brief TDDFT Krylov Tolerance
  /// 
  /// Default: 1.0E-7
  Real TDDFTKrylovTol;

  /// @brief TDDFT Phi Tolerance
  /// 
  /// Default: 1.0E-8
  Real TDDFTPhiTol;

  /// @brief TDDFT Diis Tolerance
  /// 
  /// Default: 1.0E-6
  Real TDDFTDiisTol;


  // Inputs related to Chebyshev Filtered SCF iterations for DG
  // FIXME To be combined with other input parameters
  // ~~**~~
  bool Diag_SCFDG_by_Cheby; // Default: 0
  bool SCFDG_Cheby_use_ScaLAPACK; // Default: 0

  Int First_SCFDG_ChebyFilterOrder; // Default 60
  Int First_SCFDG_ChebyCycleNum; // Default 5

  Int Second_SCFDG_ChebyOuterIter; // How many SCF steps for 2nd phase, default = 3
  Int Second_SCFDG_ChebyFilterOrder; // Filter Order for 2nd phase, default = 60
  Int Second_SCFDG_ChebyCycleNum; // Default 3 

  Int General_SCFDG_ChebyFilterOrder; // Filter Order for general phase, default = 60
  Int General_SCFDG_ChebyCycleNum; // Default 1

  // **###**
  // Inputs related to Chebyshev polynomial filtered 
  // complementary subspace iteration strategy in DGDFT
  bool scfdg_use_chefsi_complementary_subspace;
  bool scfdg_chefsi_complementary_subspace_syrk;
  bool scfdg_chefsi_complementary_subspace_syr2k;

  Int scfdg_complementary_subspace_nstates;
  Int scfdg_cs_ioniter_regular_cheby_freq;
  
  Int scfdg_cs_bigger_grid_dim_fac; 
  
  // LOBPCG (for top states) related options
  Real scfdg_complementary_subspace_lobpcg_tol;
  Int scfdg_complementary_subspace_lobpcg_iter;
  
  // CheFSI (for top states) related options 
  bool Hmat_top_states_use_Cheby;
  Int  Hmat_top_states_ChebyFilterOrder; 
  Int  Hmat_top_states_ChebyCycleNum; 

  // Inputs related to Chebyshev Filtered SCF iterations for PWDFT
  Int First_SCF_PWDFT_ChebyFilterOrder; // Default 40
  Int First_SCF_PWDFT_ChebyCycleNum; // Default 4
  Int General_SCF_PWDFT_ChebyFilterOrder; // Filter Order for general phase, default = 60
  bool PWDFT_Cheby_use_scala; // Default 1
  bool PWDFT_Cheby_apply_wfn_ecut_filt; // Default 1

  /// @brief This is NOT an input parameter, but records whether
  /// DGDFT is performed.
  bool isDGDFT;

  bool isHFXFineGrid; 

  // Inputs related to LRTDDFT  
  bool isLRTDDFT;                 // Default 0
#if 1
  bool isLRTDDFTISDF;             // Default 0
  bool isOutputExcitationEnergy;  // Default 1
  bool isOutputExcitationWfn;     // Default 0
  Int  nvband;                    // Default 1
  Int  ncband;                    // Default 1
  Int  nkband;                    // LOBPCG
  Int  startband;                 // LOBPCG
  Int  endband;                   // LOBPCG
  Int  numProcEigenSolverLRTDDFT;
  bool isdouble;                  // Default 1
  bool isfloat;                   // Default 0
  Real numMuFacLRTDDFTISDF;
  Real numGaussianRandomFacLRTDDFTISDF;
  Real toleranceLRTDDFT; 
  Int  maxIterKmeansLRTDDFTISDF;
  Real toleranceKmeansLRTDDFTISDF; 
  Int  eigMaxIterLRTDDFT;
  Real eigMinToleranceLRTDDFT;
  Real eigToleranceLRTDDFT; 
  std::string         ipTypeLRTDDFTISDF; 
  std::string         eigenSolverLRTDDFT;

  // Inputs related to Spectrum
  bool isOutputExcitationSpectrum; 
  Real LRTDDFTVextPolx;
  Real LRTDDFTVextPoly;
  Real LRTDDFTVextPolz;
  Real LRTDDFTOmegagrid;
  Real LRTDDFTSigma;
#endif

};


//-------the odd number for FFT 
//Not use it temporarily because of different energy results from that of the old version
/*
// Adjust the grid size to be more optimal for FFT. Grid sizes are currently hard coded
Int AdjustNumGrid( Int numGrid );

// Adjust the grid size to be more optimal for PW part, choose only Odd numbers for converge considerations.
Int AdjustNumGridOdd( Int numGrid );
*/
void ESDFReadInput( const std::string filename );

void ESDFReadInput( const char* filename );

void ESDFPrintInput( );


// Global input parameters
extern ESDFInputParam  esdfParam;


} // namespace esdf
} // namespace dgdft
#endif // _ESDF_HPP_
