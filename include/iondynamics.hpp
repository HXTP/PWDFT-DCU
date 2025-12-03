/*
   Copyright (c) 2012 The Regents of the University of California,
   through Lawrence Berkeley National Laboratory.  

  Author: Lin Lin, Amartya Banerjee and Subhajit Banerjee

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
/// @file iondynamics.hpp
/// @brief Geometry optimization and molecular dynamics for ions
/// @date 2015-03-05 Organize previously implemented methods
#ifndef _IONDYNAMICS_HPP_
#define _IONDYNAMICS_HPP_

#include  "environment.hpp"
#include  "domain.hpp"
#include  "numvec_impl.hpp"
#include  "periodtable.hpp"
#include  "esdf.hpp"
#include  "blas.hpp"
#include  "lapack.hpp"

using namespace std;

namespace dgdft{

extern "C"{
Int F2C(lbfgs)( Int *n, Int *m, double *x, double *f, double* g, 
    int *diagco, double *diag, int *iprint, double *eps,
    double *xtol, double *work, int *iflag );
}



enum NLCG_call_type{NLCG_INIT, NLCG_CALL_TYPE_1, NLCG_CALL_TYPE_2, NLCG_CALL_TYPE_3, NLCG_CALL_TYPE_4};   

class NLCG_internal_vars_type
{
private:

public:

  // User defined variables : read from esdf or use defaults
  int i_max_;
  int j_max_;
  int n_;
  double epsilon_tol_outer_;
  double epsilon_tol_inner_;
  double sigma_0_;

  NLCG_call_type call_type;
  int numAtom;

  int i_;
  int j_;
  int k_;

  std::vector<Point3>  atompos_x_;
  std::vector<Point3>  atomforce_r_;
  std::vector<Point3>  atomforce_s_;
  std::vector<Point3>  atomforce_d_;

  double delta_new_;
  double delta_mid_;
  double delta_old_;

  double delta_0_;
  double delta_d_;

  double eta_prev_;
  double eta_;
  double alpha_;
  double beta_;

  // Initialization routine
  void setup(int i_max, int j_max, int n, 
      double epsilon_tol_outer, double epsilon_tol_inner, double sigma_0,
      std::vector<Atom>& atomList)
  {
    call_type = NLCG_INIT;

    i_max_ = i_max;
    j_max_ = j_max;
    n_ = n;
    epsilon_tol_outer_ = epsilon_tol_outer;
    epsilon_tol_inner_ = epsilon_tol_inner;
    sigma_0_ = sigma_0;

    // Prepare the position and force lists
    numAtom = atomList.size();
    atompos_x_.resize(numAtom);
    atomforce_r_.resize(numAtom);
    atomforce_s_.resize(numAtom);
    atomforce_d_.resize(numAtom);

    i_ = 0;
    j_ = 0;
    k_ = 0;

    // Set r = -f'(x) : Note that  atomList[a].force = - grad E already - so no extra negative required
    for( Int a = 0; a < numAtom; a++ )
      atomforce_r_[a] = atomList[a].force;

    // Also copy the atom positions
    for( Int a = 0; a < numAtom; a++ )
      atompos_x_[a] = atomList[a].pos;


    // Set s = M^{-1} r : M = Identity used here
    for( Int a = 0; a < numAtom; a++ )
      atomforce_s_[a] = atomforce_r_[a];

    // Set d = s
    for( Int a = 0; a < numAtom; a++ )
      atomforce_d_[a] = atomforce_s_[a];

    // Set delta_new = r^T d  
    delta_new_ = atom_ddot(atomforce_r_, atomforce_d_);

    // Set delta_0 = delta_new
    delta_0_ = delta_new_;


    call_type = NLCG_CALL_TYPE_1;

    return;

  }

  // Computes the dot product 
  double atom_ddot(std::vector<Point3>&  list_1, std::vector <Point3>&  list_2)
  {

    double ans = 0.0;

    for( Int a = 0; a < numAtom; a++ )
    {
      for( Int d = 0; d < DIM; d++ )
      {
        ans +=  (list_1[a][d] * list_2[a][d]);
      }
    }

    return ans;    

  }

};

// A class for handling/manipulating internal state of the FIRE optimizer
 class FIRE_internal_vars_type
{
  private:

  public:

  // These variables get assigned through esdf input
  int nMin_;                    // Set to 5 by default, through esdf
  double dt_;                   // Set to 41.3413745758 a.u. (= 1 femtosecond) by default, through esdf
  double mass_;                 // Set to 4.0 by default, through esdf

  // These variables are internal to the working of the fire routines.
  // Hence, hard coded as per: Ref: DOI: 10.1103/PhysRevLett.97.170201
  double fInc_;
  double fDec_;
  double alphaStart_;
  double fAlpha_;
  // cut_ starts at 0 but keeps on getting updated as the 
  // iterations proceed
  int cut_;

  double alpha_;
  double dtMax_;                        // Set this to 10*dt_

  // Position, velocity, and forces at time t = t + dt
  DblNumVec atomPos_;
  DblNumVec atomVel_;
  DblNumVec atomForce_;

  // Position, velocity, and forces at time t = t
  DblNumVec atomPosOld_;
  DblNumVec atomVelOld_;
  DblNumVec atomForceOld_;

  int numAtom;

  // Parameter setup method:
  void setup(int nMin, double dt, double mass, double fInc, double fDec,
             double alphaStart, double fAlpha, double alpha, int cut, double dtMax,
             std::vector<Atom>& atomList, DblNumVec& atomPos, DblNumVec& atomVel, DblNumVec& atomForce, 
	     DblNumVec& atomPosOld, DblNumVec& atomVelOld, DblNumVec& atomForceOld)
  {

    // User controlled parameters, read from esdfParam.
    // Among these dt_ and alpha_ are updated or reset as 
    // decided in the method FIREStepper()
    nMin_  = nMin;
    dt_    = dt;
    mass_  = mass;

    fInc_       = fInc;
    fDec_       = fDec;
    alphaStart_ = alphaStart;
    fAlpha_     = fAlpha;
    cut_        = cut;
    alpha_      = alpha;
    dtMax_      = dtMax;

    numAtom = atomList.size();

    // Prepare the position, velocity, and force lists
    atomPos_       = atomPos;
    atomVel_       = atomVel;
    atomForce_     = atomForce;

    // Required for the first step
    atomPosOld_    = atomPosOld;
    atomVelOld_    = atomVelOld;
    atomForceOld_  = atomForceOld;

    return;
  }

  // Purpose: Compute L2-norm  of a vector of length 3 * numAtom
  // Usage: To compute \hat{F} = F / norm( F ) 
  double atom_l2norm(DblNumVec&  list)
  {

    double accum = 0.0;

    for( Int i = 0; i < 3*numAtom; i++ ){
      accum += list[i]*list[i];
    }

    double l2norm = sqrt(accum);

    return l2norm;

  }

  // Purpose: Compute dot product of two vectors
  // Usage: To compute Power (P) = F . v  
  double atom_ddot(DblNumVec& list1, DblNumVec& list2)
  {

    double ans = 0.0;

    for( Int i = 0; i < 3*numAtom; i++ ){
      ans += list1[i]*list2[i];
    }

    return ans;

  }

  // Purpose: Scale a vector by multiplying its every element by a scalar
  // Usage: Perform (1 - alpha) * v and alpha * \hat{F}
  void atom_scale(DblNumVec& list, double fctr)
  {

    for( Int i = 0; i < 3*numAtom; i++ ){
      list[i] *= fctr;
    }

    return;

  }

  // Add two vectors
  void atom_add(DblNumVec& list1, DblNumVec& list2, DblNumVec& list3)
  {

    for( Int i = 0; i < 3*numAtom; i++ ){
      list3[i] = list1[i] + list2[i];
    }

    return;
  }

  //
  void DblNumVecCopier( DblNumVec& list1, DblNumVec& list2 )
  {

    // list1 gets copies to list2
    for( Int i = 0; i < 3*numAtom; i++ ){
      list2[i] = list1[i];
    }
 
    return;
  }

  //
  void DblNumVecPrinter( DblNumVec list )
  {

    for( Int i = 0; i < 3*numAtom; i++ ){
      statusOFS << std::endl << "Component: " << i << " = " << list[i] << std::endl;
    }  

    return;
  }


  //
  void FIREStepper( const int& it )
  {

    // Compute the Power:

    double power = atom_ddot(atomVel_, atomForce_);

    statusOFS << std::endl << "Power = " << power << std::endl;

    DblNumVec fHat(3*numAtom);

    fHat = atomForce_;
    
    atom_scale(fHat, 1.0/atom_l2norm(atomForce_));

    // FIRE Velocity update formula:
    DblNumVec tmpVel(3*numAtom);

    tmpVel = atomVel_;

    DblNumVec tmpForce(3*numAtom);

    tmpForce = atomForce_;
   
    atom_scale(tmpVel, (1.0 - alpha_));
 
    atom_scale(tmpForce, alpha_*atom_l2norm(atomVel_)/atom_l2norm(atomForce_));

    atom_add(tmpVel, tmpForce, atomVel_);

    //dtMax_ = 10.0*dt_;		// dtMax updated for every new dt. ** !!CHECK!! **
	
    statusOFS << std::endl << "alpha_ before update: " << alpha_ << std::endl;

    statusOFS << std::endl << "dt_ before update : " << dt_ << std::endl;

    statusOFS << std::endl << " cut_ before update: " << cut_ << std::endl;

    if (power < 0.0){
       // Reset the velocities to 0.0
       for( Int i = 0; i < 3*numAtom; i++ ){
           atomVel_[i] = 0.0;
       }
       cut_ = it;                 // cut_ <-- iter # (gets updated everytime P <= 0)
       dt_ = dt_*fDec_;           // slow down
       alpha_ = alphaStart_;      // reset alpha to alphaStart
    }
    else if ((power >= 0.0) && ((it - cut_) > nMin_)) {
       dt_ = std::min(dt_*fInc_, dtMax_);
       alpha_ = fAlpha_*alpha_;
    }

   statusOFS << std::endl << "alpha_ after update: " << alpha_ << std::endl;

    statusOFS << std::endl << "dt_ after update : " << dt_ << std::endl;

    statusOFS << std::endl << " cut_ after update: " << cut_ << std::endl;

    return;
 }

 };


struct GeoOptVars
{
  // Currently the optimization parameters are taken from fminPGBB.m obtained from Zaiwen Wen.
  // Variable names to be organized later
  Real xtol;
  Real gtol;

  // Parameters for controling the linear approximation in line search. 
  // 
  // Used in PGBB
  Real tau;
  Real rhols;
  Real eta;
  Real gamma;
  Real STPEPS;
  Real nt;
  Int callType;
  Int nls;
  // Internal variables
  std::vector<Point3>  atompos;
  std::vector<Point3>  atomforce;
  std::vector<Point3>  atomposOld;
  std::vector<Point3>  atomforceOld;

  // Parameters for LBFGS
  // 
  DblNumVec work;
  Int maxMixingDim;
};


class IonDynamics
{
private:
  /// @brief Syncec with atomic position (and velocity and force) from
  /// input. Currently atomList info is stored in the Hamiltonian
  /// classes (Hamiltonian or HamiltonianDG).
  std::vector<Atom>*   atomListPtr_;

  /// @brief Stores the Positions at the infimum of maxForce 
  /// Useful for restarting any geometry optimization method from 
  /// infimum configuration rather than lastPos
  Real fAtInfimum_;

  /// @brief Max number of history for density extrapolation etc
  Int maxHist_;

  /// @brief atomListHist_[0] stores the lastest atomic configuration
  ///
  /// FIXME
  /// This should be combined with GeoOptVars information later
  std::vector<std::vector<Atom> >   atomListHist_;

  std::string          ionMove_;

  DblNumVec            atomMass_;

  // Taken from input
  std::string          MDExtrapolationType_;

  // Supercell parameters : grab this from esdf
  Real supercell_x_;
  Real supercell_y_;
  Real supercell_z_;
  
  // Molecular dynamics variables
  Real                 Ekinetic_; // kinetic energy for ions
  Real                 Epot_;  // potential energy for ions
  Real                 EconserveInit_;
  Real                 Econserve_;
  Real                 Edrift_;
  Real                 dt_;
  Real                 ionTemperature_; // unit: au

  // Nose-Hoover variables
  Real                 Q1_;
  Real                 xi1_;
  Real                 vxi1_;
  Real                 G1_;
  Real                 scalefac_;
  Int                  phase_; // NH1 has two phases
  // Langevin variables
  Real                 langevinDamping_;
  
  bool                 isGeoOpt_;
  bool                 isMD_;

  GeoOptVars           geoOptVars_;

  /// @brief BarzilaiBorwein method for geometry optimization
  ///
  void BarzilaiBorweinOpt( Int ionIter );

  /// @brief PGBB method for geometry optimization (from Zaiwen Wen)
  ///
  void PGBBOpt( Int ionIter );

  /// @brief Norcedal's implementation of L-BFGS
  ///
  void LBFGSOpt( Int ionIter );

  /// @brief Non-linear Conjugate Gradient with Secant and Polak-Ribiere
  NLCG_internal_vars_type NLCG_vars;
  void NLCG_Opt(Int ionIter );

  /// @brief Fast Inertial Relaxation Engine
  // Subhajit Banerjee
  FIRE_internal_vars_type FIRE_vars;
  void FIREOpt(Int ionIter);

  /// @brief VelocityVerlet for NVE simulation
  ///
  void VelocityVerlet( Int ionIter );

  /// @brief NoseHoover thermostat with chain level 1. The
  /// implementation is consistent with the CORRECT version of
  ///
  /// Frenkel and Smit, Understanding Molecular Simulation, 
  /// Alg. 30 (pp 540)
  ///
  /// The correction is due to the inconsistency between (E.2.4) and the
  /// pos_vel algorithm (Alg. 32). The algorithm in the book is correct,
  /// however, it does not allow one to obtain position and velocity at
  /// the same snapshot.
  /// 
  /// Normally one NH step is (3.2.4)
  ///
  ///   *Time t
  ///   chain (dt/2)
  ///   velocity (dt/2)
  ///   position (dt)
  ///   evaluate force
  ///   velocity (dt/2)
  ///   chain (dt/2)
  ///   *Time t+dt
  ///
  /// Since we do not want to call MoveIons twice, the order is switched
  /// to the following
  ///
  ///   evaluate force
  ///   velocity (dt/2)
  ///   chain (dt/2)
  ///   *Time t+dt
  ///   chain (dt/2)
  ///   velocity (dt/2)
  ///   position (dt)
  ///
  /// This means that after updating the chain variables for the first
  /// time, the position, velocity, energy and forces are synced at time
  /// t+dt, and then chain, velocity and position variables are updated
  /// in preparation for evaluation of the force.  
  ///
  /// If the job is stopped in the middle of the evaluation of the
  /// force, the calculation can be restarted from the stored variables
  /// without repeating the work. (3/6/2016)
  /// 
  void NoseHoover1( Int ionIter );

  // Integrator 
  /// @brief Langevin integrator
  /// Reference 
  ///
  /// N. Gronbech-Jensen, O. Farago, A simple and effective Verlet-type
  /// algorithm for simulating Langevin dynamics, Mol.  Phys.  111
  /// (2013) 983–991
  void Langevin( Int ionIter );



public:

  /// @brief Initial setup from the input parameters
  void Setup( std::vector<Atom>& atomList,
      PeriodTable& ptable );

  /// @brief Main program to move the ions.
  ///
  /// Will determine both geometry optimization and molecular dynamics
  void MoveIons( Int ionIter );

  /// @brief Extrapolating coefficient for density or wavefunction
  ///
  void ExtrapolateCoefficient( Int ionIter, DblNumVec& coef );

  // *********************************************************************
  // Access functions
  // *********************************************************************
  std::vector<Atom>& AtomList() { return *atomListPtr_; }
  void SetEpot(Real Epot) { Epot_ = Epot; }
  Int  MaxHist() { return maxHist_; }

  bool IsGeoOpt() { return isGeoOpt_; }
  bool IsMD()     { return isMD_; }


};


} // namespace dgdft


#endif // _IONDYNAMICS_HPP_
