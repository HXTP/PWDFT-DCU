/*
   Copyright (c) 2012 The Regents of the University of California,
   through Lawrence Berkeley National Laboratory.  

Author: Lin Lin and Weile Jia

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
/// @file periodtable.hpp
/// @brief Periodic table and its entries.
/// @date 2012-08-10
/// @date 2017-12-26 Add support for UPF format
#ifndef _PERIODTABLE_HPP_
#define _PERIODTABLE_HPP_

#include "environment.hpp"
#include "domain.hpp"
#include "tinyvec_impl.hpp"
#include "numvec_impl.hpp"
#include "nummat_impl.hpp"
#include "numtns_impl.hpp"

namespace dgdft{

// The following typedefs must be IDENTIAL to that in utility.hpp (C++
// feature only)
typedef std::pair<IntNumVec, DblNumMat > SparseVec; 
typedef std::pair<SparseVec, Real> NonlocalPP; 


// *********************************************************************
// Atomic information
// *********************************************************************
/// @struct Atom
/// @brief Atom structure.
struct Atom
{
  /// @brief Atomic number.
  Int          type;                            
  /// @brief Position.
  Point3       pos;  
  /// @brief Velocity.
  Point3       vel;  
  /// @brief Force.
  Point3       force; 

  Atom() {;}
  ~Atom() {;}
  Atom( const Int t, const Point3 p, const Point3 v, const Point3 f): 
    type(t), pos(p), vel(v), force(f) {}
};

/// @struct PTParam
/// @brief Index of the parameters in each entry of the periodic table.
/// However, the detailed value may depend on the pseudopotential.
///
/// FIXME: RGAUSSIAN == ESELF
///
/// If isUseVLocal == false, the position ESELF stores the self energy
/// If isUseVLocal == true, the same position stores the radius of the
/// Gaussian pseudocharge
struct PTParam{
  /// @brief Atomic number.
  Int ZNUC;
  /// @brief Nuclear mass.
  Int MASS;
  /// @brief Nuclear charge (valence).
  Int ZION;
  /// @brief Self-interaction energy.
  Int ESELF;
  /// @brief Effective radius of the Gaussian charge
  /// In the future this could be a table storing the default value for
  /// each type of species.
  Int RGAUSSIAN;
};


/// @struct PTType
/// @brief Type of each sample, including radial grid, pseudocharge, and
/// nonlocal projectors of different angular momentum, and the
/// spin-orbit contribution. 
/// However, the detailed value may depend on the pseudopotential.
struct PTType{
  Int RADIAL;
  Int PSEUDO_CHARGE;
  Int VLOCAL;
  Int RHOATOM;
  Int L0;
  Int L1;
  Int L2;
  Int L3;
  Int SPINORBIT_L1;
  Int SPINORBIT_L2;
  Int SPINORBIT_L3;
};


/// @struct PTSample
/// @brief Index of the radial grid, the pseudocharge, derivative
/// of the pseudocharge, model core charge density, derivative of model
/// core charge density, nonlocal pseudopotentials in samples and cuts.
/// However, the detailed value may depend on the pseudopotential.
///
/// FIXME: PSEUDO_CHARGE = VLOCAL, DRV_PSEUDO_CHARGE = DRV_VLOCAL.
///
///
/// If isUseVLocal == false, then ptsample stores the pseudocharge of
/// the local part of the pseudopotential 
///
/// If isUseVLocal == true, then ptsample stores the local part of the
/// pseudopotential  in the same place
struct PTSample{
  Int RADIAL_GRID;
  Int PSEUDO_CHARGE;
  Int DRV_PSEUDO_CHARGE;
  Int VLOCAL;
  Int DRV_VLOCAL;
  Int RHOATOM;
  Int DRV_RHOATOM;
  Int NONLOCAL;
};

/// @brief The start of the radial grid.
const Real MIN_RADIAL = 1e-10;

/// @struct PTEntry
/// @brief Each entry for the periodic table structure.
struct PTEntry
{
  /// @brief Parameters following the order in PTParam.
  DblNumVec params; 
  /// @brief Radial grid, pseudocharge and nonlocal projectors and
  /// their derivatives.
  DblNumMat samples; 
  /// @brief Weight of each sample. Only used for the nonlocal projectors. 
  DblNumVec weights; 
  /// @brief Type of each sample, following PTType.
  IntNumVec types; 

//  DblNumVec soctypes;
  /// @brief Cutoff value for different samples, following PTSample
  DblNumVec cutoffs; 
};

Int serialize(const PTEntry&, std::ostream&, const std::vector<Int>&);
Int deserialize(PTEntry&, std::istream&, const std::vector<Int>&);
Int combine(PTEntry&, PTEntry&);

/// @class PeriodTable
/// @brief Periodic table for pseudopotentials.
class PeriodTable
{
private:
  PTParam  ptparam_;
  PTType   pttype_;
  PTSample ptsample_;

  /// @brief Map from atomic number to PTEntry
  std::map<Int, PTEntry> ptemap_; 
  /// @brief Map from atomic number to splines for pseudopotentials.
  std::map<Int, std::map< Int,std::vector<DblNumVec> > > splmap_;
public:
  PeriodTable() {;}
  ~PeriodTable() {;}

  /// @brief Map from atomic number to PTEntry
  std::map<Int, PTEntry>& ptemap() { return ptemap_; }

  /// @brief Map from atomic number to splines for pseudopotentials.
  std::map<Int, std::map< Int,std::vector<DblNumVec> > > splmap() { return splmap_; }

  /// @brief Read the information of the periodic table from file.  
  ///
  /// All processors can call this routine at the same time. 
  void Setup( );

  /// @brief Generate the pseudo-charge and its derivatives.
  ///
  /// The data are saved in the sparse veector res
  ///   res[0]         : pseudo-charge values
  ///   res[1]--res[3] : x,y,z components of the derivatives of the
  ///             pseudo-charge
  void CalculatePseudoCharge( const Atom& atom, const Domain& dm, 
      const std::vector<DblNumVec>& gridpos,
      SparseVec& res );

  /// @brief Generate the pseudo-charge and its derivatives on a set of
  /// elements.
  ///
  /// The data are saved in the sparse veector res
  ///   res[0]         : pseudo-charge values
  ///   res[1]--res[3] : x,y,z components of the derivatives of the
  ///             pseudo-charge
  void CalculatePseudoCharge    (
      const Atom& atom, 
      const Domain& dm,
      const NumTns<std::vector<DblNumVec> >& gridposElem,
      NumTns<SparseVec>& res );

  /// @brief Generate the nonlocal pseudopotential projectors.
  void CalculateNonlocalPP( const Atom& atom, const Domain& dm, 
      const std::vector<DblNumVec>& gridpos,
      std::vector<NonlocalPP>& vnlList );

  /// @brief Generate the nonlocal pseudopotential projectors over a set
  /// of elements.
  void CalculateNonlocalPP( const Atom& atom, const Domain& dm, 
      const NumTns<std::vector<DblNumVec> >&   gridposElem,
      std::vector<std::pair<NumTns<SparseVec>, Real> >& vnlList );

  /// @brief Generate the atomic density. This is to be used with
  /// structure factor to generate the initial atomic density.
  ///
  /// This can be used both for PWDFT and DGDFT.
  /// 
  /// NOTE: This assumes atom.pos should agree with posStart in domain
  /// (PWDFT) and global domain (DGDFT).
  ///
  void CalculateAtomDensity( const Atom& atom, const Domain& dm, 
      const std::vector<DblNumVec>& gridpos, 
      DblNumVec& atomDensity );


  /// @brief Generate the short range local pseudopotential and its
  /// derivatives, as well as the Gaussian compensation charge and its derivatives.
  ///
  /// The data are saved in the sparse veector res
  ///   res[0]         : values
  ///   res[1]--res[3] : x,y,z components of the derivatives of the
  void CalculateVLocal( const Atom& atom, const Domain& dm, 
      const std::vector<DblNumVec>& gridpos,
      SparseVec& resVLocalSR, 
      SparseVec& resGaussianPseudoCharge );

void  CalculateVLocal(
    const Atom& atom, 
    const Domain& dm,
    const NumTns<std::vector<DblNumVec> >& gridposElem,
    NumTns<SparseVec>& resVLocalSR, 
    NumTns<SparseVec>& resGaussianPseudoCharge);


  /// @brief Whether the atom type has nonlocal pseudopotential
  bool IsNonlocal(Int type) {return ptemap_[type].cutoffs.m()>ptsample_.NONLOCAL;}
  
  /// @brief Cutoff radius for the pseudocharge in the real space
  Real RcutPseudoCharge(Int type)   {return ptemap_[type].cutoffs(ptsample_.PSEUDO_CHARGE);}
 

  /// @brief Cutoff radius for model atomic density in the real space.
  /// This is only used for constructing initial charge density, and
  /// does not need to be very accurate
  Real RcutRhoAtom(Int type)   {return ptemap_[type].cutoffs(ptsample_.RHOATOM);}
  
  /// @brief Cutoff radius for the nonlocal pseudpotental in the real
  /// space. If there are multiple pseudopotentials, Rcut should be the
  /// maximum radius so that ALL nonlocal pseudopotentials are accurate.
  Real RcutNonlocal(Int type)   {return (this->IsNonlocal(type)) ? 
    ptemap_[type].cutoffs(ptsample_.NONLOCAL) : 0.0;}


  /// @brief Atomic mass
  Real Mass(Int type) {return ptemap_[type].params(ptparam_.MASS);}

  /// @brief Valence charge of the ion
  Real Zion(Int type) {return ptemap_[type].params(ptparam_.ZION);}

  /// @brief Self ionic interaction energy.
  ///
  /// This handles the case of isUseVLocal = true or false
  Real SelfIonInteraction (Int type); 
  
  /// @brief Cutoff radius for the pseudocharge in the real space
  Real RGaussian(Int type)   {return ptemap_[type].params(ptparam_.RGAUSSIAN);}
};



// Serialization / Deserialization
Int serialize(const Atom& val, std::ostream& os, const std::vector<Int>& mask);

Int deserialize(Atom& val, std::istream& is, const std::vector<Int>& mask);

Real MaxForce( const std::vector<Atom>& atomList );

/// @brief Read PTEntry from UPF file.
///
/// This is obtained from the UPF2QSO subroutine in Qbox.
Int ReadUPF( std::string file_name, PTEntry* pt, Int * atom);

} // namespace dgdft

#endif // _PERIODTABLE_HPP_
