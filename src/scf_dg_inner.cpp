/*
   Copyright (c) 2012 The Regents of the University of California,
   through Lawrence Berkeley National Laboratory.  

Authors: Lin Lin, Wei Hu, Amartya Banerjee, and Xinming Qin

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
/// @file scf_dg.cpp
/// @brief Self consistent iteration using the DG method.
/// @date 2013-02-05
/// @date 2014-08-06 Add intra-element parallelization.
//
/// Xinming Qin xmqin03@gmail.com
/// @date 2023-10-17 Add Hybrid DFT ALBs
/// @date 2023-11-05 Add HFX hamiltonian matrix
/// @date 2024-01-29 Add different mixing schemes 
// 
#include  "scf_dg.hpp"
#include  "blas.hpp"
#include  "lapack.hpp"
#include  "utility.hpp"
#include  "environment.hpp"
#ifdef ELSI
#include  "elsi.h"
#endif

namespace  dgdft{

using namespace dgdft::DensityComponent;
using namespace dgdft::esdf;
using namespace dgdft::scalapack;

void
SCFDG::InnerIterate    ( Int outerIter )
{
  Int mpirank, mpisize;
  MPI_Comm_rank( domain_.comm, &mpirank );
  MPI_Comm_size( domain_.comm, &mpisize );
  Int mpirankRow;  MPI_Comm_rank(domain_.rowComm, &mpirankRow);
  Int mpisizeRow;  MPI_Comm_size(domain_.rowComm, &mpisizeRow);
  Int mpirankCol;  MPI_Comm_rank(domain_.colComm, &mpirankCol);
  Int mpisizeCol;  MPI_Comm_size(domain_.colComm, &mpisizeCol);

  Real timeSta, timeEnd;
  Real timeIterSta, timeIterEnd;
  Real timeSta0, timeEnd0;

  HamiltonianDG&  hamDG = *hamDGPtr_;

  bool isInnerSCFConverged = false;

  // The first inner iteration does not update the potential, and
  // construct the global Hamiltonian matrix from scratch
//  GetTime(timeSta);

  // *********************************************************************
  // Enter the inner SCF iterations with fixed ALBs.
  // This is similar with SIESTA/CP2K/FHI-aims, but the outer SCF 
  // iteration is not needed since the basis set is given and fixed.
  // A mixing scheme is required to accelerate the SCF convergence.
  // Now we add Anderson/Broyden mixing for density, potential,
  // density matrix and Hamiltonian.
  // Hybrid DFT can only uses density matrix and Hamiltonian mixing.
  // *********************************************************************  

//  if( hamDG.IsHybrid() && hamDG.IsEXXActive() ) {
//    scfInnerMaxIter_   = HybridscfInnerMaxIter_;
//    scfInnerTolerance_ = HybridscfInnerTolerance_;
//    InnermixVariable_  = HybridInnermixVariable_;
//    mixType_ = HybridmixType_;  
//    mixMaxDim_ = HybridmixMaxDim_;
//    mixStepLength_ = HybridmixStepLength_;
//  }
//  else {
    scfInnerMaxIter_   = DFTscfInnerMaxIter_;
    scfInnerTolerance_ = DFTscfInnerTolerance_;
//  }
//
//  statusOFS << " outerIter " << outerIter <<  std::endl;
//  statusOFS << " InnermixVariable_ "<< InnermixVariable_ << std::endl;
//  statusOFS << " mixType_ " << mixType_ << std::endl;
//  statusOFS << " mixStepLength_ "<< mixStepLength_ << std::endl;

//  statusOFS<< " esdfParam.DGhybridMixType "<< esdfParam.DGhybridMixType << std::endl;
//  statusOFS<< "  hamDG.IsHybrid() " <<  hamDG.IsHybrid() << std::endl; 
//  statusOFS<< " hamDG.IsEXXActive() " << hamDG.IsEXXActive() << std::endl;
//  statusOFS<< " DGHFXNestedLoop_" << DGHFXNestedLoop_ << std::endl;

   if( DGHFXNestedLoop_ && hamDG.IsHybrid() && hamDG.IsEXXActive() ){

    statusOFS << " Nested-Loop Scheme for Hybrid DFT Calculations" << std::endl;
    statusOFS << " Mixing variable " << InnermixVariable_ << std::endl;
    statusOFS << " Mixing type     " <<   mixType_ << std::endl;

    Real Ehfx0 = 0.0, Ehfx1 = 0.0, Ehfx2 = 0.0, Ehfxin = 0.0;
    Real dExx; 

    Real timeHFXIterSta, timeHFXIterEnd; 

    bool isHFXIterConverged = false;

//    GetTime( timeSta );
//    hamDG.CalculateDGMatrix ( );
//    GetTime( timeEnd );
//#if ( _DEBUGlevel_ >= 1 )
//    statusOFS << "Time for constructing the DG matrix is " <<
//      timeEnd - timeSta << " [s]" << std::endl << std::endl;
//#endif

    GetTime( timeSta );
    if(esdfParam.isDGHFISDF){
      hamDG.CalculateDGHFXMatrix_ISDF( Ehfx_, distDMMat_ );
    }
    else{
      hamDG.CalculateDGHFXMatrix( distDMMat_ );
    }

    hamDG.CalculateDGHFXEnergy( Ehfx0, distDMMat_, hamDG.HFXMat() );

//    hamDG.MinusDGHFXMatrix( );

    GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 1 )
    statusOFS << "InnerSCF: Time for constructing the DGHFX matrix is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

    Ehfx_ = Ehfx0;
    Ehfx1 = Ehfx0;
    Ehfx2 = Ehfx0;

    for( Int innerHFXIter = 1; innerHFXIter <= HybridscfInnerMaxIter_; innerHFXIter++ ){
  
      statusOFS << std::endl << "Nested DG HFX iteration #"
        << innerHFXIter << " starts." << std::endl << std::endl;
  
      GetTime( timeHFXIterSta );

      bool isSCFConverged = false ;

      for (Int innerDFTIter=1; innerDFTIter <= scfInnerMaxIter_; innerDFTIter++) {

        if( isSCFConverged ) break;
        // *********************************************************************
        // Performing each iteartion
        // *********************************************************************
        {
          std::ostringstream msg;
          msg << "  Nested Inner DFT density matrix iteration # " << innerDFTIter;
          PrintBlock( statusOFS, msg.str() );
        }

        GetTime( timeIterSta );

        if( innerDFTIter == 1 ){
//        GetTime( timeSta );
          hamDG.CalculateDGMatrix ( );
//        GetTime( timeEnd );
//    #if ( _DEBUGlevel_ >= 1 )
//        statusOFS << "Time for constructing the DG matrix is " <<
//          timeEnd - timeSta << " [s]" << std::endl << std::endl;
//    #endif

          hamDG.MinusDGHFXMatrix( );

        }
//        GetTime(timeSta);
        else
        {
          // Save the old potential on the LGL grid
          for( Int k = 0; k < numElem_[2]; k++ )
            for( Int j = 0; j < numElem_[1]; j++ )
              for( Int i = 0; i < numElem_[0]; i++ ){
                Index3 key( i, j, k );
                if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
                  Index3 numLGLGrid     = hamDG.NumLGLGridElem();
                  blas::Copy( numLGLGrid.prod(),
                      hamDG.VtotLGL().LocalMap()[key].Data(), 1,
                      vtotLGLSave_.LocalMap()[key].Data(), 1 );
                } // if (own this element)
          } // for (i)

          // Update the local potential on the extended element and on the
          // element.

          UpdateElemLocalPotential();

          // Save the difference of the potential on the LGL grid into vtotLGLSave_
          for( Int k = 0; k < numElem_[2]; k++ )
            for( Int j = 0; j < numElem_[1]; j++ )
              for( Int i = 0; i < numElem_[0]; i++ ){
                Index3 key( i, j, k );
                if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
                  Index3 numLGLGrid     = hamDG.NumLGLGridElem();
                  Real *ptrNew = hamDG.VtotLGL().LocalMap()[key].Data();
                  Real *ptrDif = vtotLGLSave_.LocalMap()[key].Data();
                  for( Int p = 0; p < numLGLGrid.prod(); p++ ){
                    (*ptrDif) = (*ptrNew) - (*ptrDif);
                    ptrNew++;
                    ptrDif++;
                  }
                } // if (own this element)
          } // for (i)

          GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 1 )
          statusOFS << "InnerSCF:: Time for updating the local potential in the extended element and the element is " <<
            timeEnd - timeSta << " [s]" << std::endl;
#endif

#if 0
          GetTime(timeSta);
          hamDG.UpdateDGMatrix( vtotLGLSave_ );
          GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 1 )
          statusOFS << "Time for updating the DG matrix is " <<
            timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
#else
        GetTime( timeSta );
        hamDG.CalculateDGMatrix ( );
        GetTime( timeEnd );
    #if ( _DEBUGlevel_ >= 1 )
        statusOFS << "Time for constructing the DG matrix is " <<
          timeEnd - timeSta << " [s]" << std::endl << std::endl;
    #endif

        hamDG.MinusDGHFXMatrix( );
#endif
        }

        GetTime(timeSta);
         
        // Save the mixing variable first
        // 
        {
          for( Int k = 0; k < numElem_[2]; k++ )
            for( Int j = 0; j < numElem_[1]; j++ )
              for( Int i = 0; i < numElem_[0]; i++ ){
                Index3 key( i, j, k );
                if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
                  if( InnermixVariable_ == "density" ){
                    DblNumVec& oldVec = hamDG.Density().LocalMap()[key];
                    mixInnerSave_.LocalMap()[key] = oldVec;
                  }
                  else if( InnermixVariable_ == "potential" ){
                    DblNumVec& oldVec = hamDG.Vtot().LocalMap()[key];
                    mixInnerSave_.LocalMap()[key] = oldVec;
                  }
                } // own this element
          } // for (i)
        }

        GetTime( timeSta );
        InnerSolver ( outerIter );
        GetTime( timeEnd );

#if ( _DEBUGlevel_ >= 1 )
          statusOFS << "Time for diagonalizing the DG matrix is " <<
            timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif                       

  // **************************************************************************************
  //   // Compute the error of the mixing variable
  //
//        statusOFS << " InnerDG SCF " << innerDFTIter << " :   Density/Potential converge ?" << std::endl;
        GetTime(timeSta);
        {
          Real normMixDifLocal = 0.0, normMixOldLocal = 0.0;
          Real normMixDif, normMixOld;
      
          for( Int k = 0; k < numElem_[2]; k++ ){
            for( Int j = 0; j < numElem_[1]; j++ ){
              for( Int i = 0; i < numElem_[0]; i++ ){
                Index3 key( i, j, k );
                if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
                  if( InnermixVariable_ == "density" ){
                    DblNumVec& oldVec = mixInnerSave_.LocalMap()[key];
                    DblNumVec& newVec = hamDG.Density().LocalMap()[key];
      
                    for( Int p = 0; p < oldVec.m(); p++ ){
                      normMixDifLocal += pow( oldVec(p) - newVec(p), 2.0 );
                      normMixOldLocal += pow( oldVec(p), 2.0 );
                    }
                  }
                  else if( InnermixVariable_ == "potential" ){
                    DblNumVec& oldVec = mixInnerSave_.LocalMap()[key];
                    DblNumVec& newVec = hamDG.Vtot().LocalMap()[key];
      
                    for( Int p = 0; p < oldVec.m(); p++ ){
                      normMixDifLocal += pow( oldVec(p) - newVec(p), 2.0 );
                      normMixOldLocal += pow( oldVec(p), 2.0 );
                    }
                  }
                } // own this element
              } // for (i)
            } // for(j)
          } // for (k)
      
          mpi::Allreduce( &normMixDifLocal, &normMixDif, 1, MPI_SUM,
            domain_.colComm );
          mpi::Allreduce( &normMixOldLocal, &normMixOld, 1, MPI_SUM,
            domain_.colComm );
      
          normMixDif = std::sqrt( normMixDif );
          normMixOld = std::sqrt( normMixOld );
      
          scfInnerNorm_    = normMixDif / normMixOld;
      
          #if ( _DEBUGlevel_ >= 0 )
            Print(statusOFS, "norm(MixDif)          = ", normMixDif );
            Print(statusOFS, "norm(MixOld)          = ", normMixOld );
            Print(statusOFS, "norm(out-in)/norm(in) = ", scfInnerNorm_ );
          #endif

        }

        if( scfInnerNorm_ < scfInnerTolerance_ ){
          /* converged */
//          Print( statusOFS, "Nested SCF is converged!\n" );
          statusOFS << "nested SCF is converged in " << innerDFTIter << " steps !" << std::endl;
          isSCFConverged = true;
        }
      
        GetTime( timeEnd );
        #if ( _DEBUGlevel_ >= 1 )
     
            timeEnd - timeSta << " [s]" << std::endl << std::endl;
        #endif

//    if(!isSCFConverged){
      if(1){
        // Mixing for the inner SCF iteration.
        GetTime( timeSta );
      
        // The number of iterations used for Anderson mixing
        Int numIter = innerDFTIter;
      
//        statusOFS <<" InnermixVariable_  " << InnermixVariable_ << std::endl;
//        statusOFS << " mixType_  " <<  mixType_ << std::endl;
//        statusOFS << " innerIter " << innerDFTIter  << std::endl;
//        statusOFS << " outerIter " << outerIter  << std::endl;
//        statusOFS << " numIter " <<  numIter << std::endl << std::endl;
      
        if( InnermixVariable_ == "density" ){
          if( mixType_ == "anderson" ){
            //statusOFS << " anderson density mixing " << std::endl;
            AndersonMix2(
                numIter, 
                mixStepLength_,
                hamDG.Density(),
                mixInnerSave_,
                hamDG.Density(),
                dfInnerMat_,
                dvInnerMat_);
          }
          else if( mixType_ == "kerker+anderson"){
            //statusOFS << " kerker+anderson density mixing " << std::endl;
            AndersonMix(
                numIter, 
                mixStepLength_,
                mixType_,
                hamDG.Density(),
                mixInnerSave_,
                hamDG.Density(),
                dfInnerMat_,
                dvInnerMat_);
          }
          else if( mixType_ == "pulay"){
            //statusOFS << " pulay density mixing " << std::endl;
            PulayMix(
                numIter, 
                mixStepLength_,
                hamDG.Density(),
                mixInnerSave_,
                hamDG.Density(),
                dfInnerMat_,
                dvInnerMat_);
          }
          else if( mixType_ == "broyden" ){
            //statusOFS << " broyden density mixing " << std::endl;
            BroydenMix(
                numIter,
                mixStepLength_,
                hamDG.Density(),
                mixInnerSave_,
                hamDG.Density(),
                dfInnerMat_,
                dvInnerMat_,
                cdfInnerMat_);
          }
          else{
            ErrorHandling("Invalid density mixing type.");
          }
        }
        else if( InnermixVariable_ == "potential" ){
          if( mixType_ == "anderson"){
            //statusOFS << " anderson potential mixing " << std::endl;
            AndersonMix2(
                numIter,
                mixStepLength_,
                hamDG.Vtot(),
                mixInnerSave_,
                hamDG.Vtot(),
                dfInnerMat_,
                dvInnerMat_);
          }
          else if( mixType_ == "kerker+anderson"    ){
            //statusOFS << " kerker+anderson potential mixing " << std::endl;
            AndersonMix(
                numIter, 
                mixStepLength_,
                mixType_,
                hamDG.Vtot(),
                mixInnerSave_,
                hamDG.Vtot(),
                dfInnerMat_,
                dvInnerMat_);
          }
          else if( mixType_ == "pulay"    ){
            //statusOFS << " pulay potential mixing " << std::endl;
            PulayMix(
                numIter, 
                mixStepLength_,
                hamDG.Vtot(),
                mixInnerSave_,
                hamDG.Vtot(),
                dfInnerMat_,
                dvInnerMat_);
          }
          else if( mixType_ == "broyden" ){
            //statusOFS << " broyden potential mixing " << std::endl;
            BroydenMix(
                numIter,
                mixStepLength_,
                hamDG.Vtot(),
                mixInnerSave_,
                hamDG.Vtot(),
                dfInnerMat_,
                dvInnerMat_,
                cdfInnerMat_);
          }
          else{
            ErrorHandling("Invalid potential mixing type.");
          }
        }

        MPI_Barrier( domain_.comm );
        GetTime( timeEnd );
        #if ( _DEBUGlevel_ >= 1 )
          statusOFS << "InnerSCF: Time for mixing is " <<
            timeEnd - timeSta << " [s]" << std::endl;
        #endif

        if( InnermixVariable_ == "density" )
        {
          Real sumRhoLocal = 0.0;
          Real sumRho;
          for( Int k = 0; k < numElem_[2]; k++ )
            for( Int j = 0; j < numElem_[1]; j++ )
              for( Int i = 0; i < numElem_[0]; i++ ){
                Index3 key( i, j, k );
                if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
                  DblNumVec&  density      = hamDG.Density().LocalMap()[key];
      
                  for (Int p=0; p < density.Size(); p++) {
                    density(p) = std::max( density(p), 0.0 );
                    sumRhoLocal += density(p);
                  }
                } // own this element
              } // for (i)
      
          mpi::Allreduce( &sumRhoLocal, &sumRho, 1, MPI_SUM, domain_.colComm );
          sumRho *= domain_.Volume() / domain_.NumGridTotalFine();
      
          Real rhofac = hamDG.NumSpin() * hamDG.NumOccupiedState() / sumRho;
          #if ( _DEBUGlevel_ >= 1 )
         //    statusOFS << std::endl;
         //    Print( statusOFS, "Numer of Occupied State ",  hamDG.NumOccupiedState() );
          Print( statusOFS, "Rho factor after mixing (raw data) = ", rhofac );
          Print( statusOFS, "Sum Rho after mixing (raw data)    = ", sumRho );
          //    statusOFS << std::endl;
          #endif
          // Normalize the electron density in the global domain
          #if 1
          for( Int k = 0; k < numElem_[2]; k++ )
            for( Int j = 0; j < numElem_[1]; j++ )
              for( Int i = 0; i < numElem_[0]; i++ ){
                Index3 key( i, j, k );
                  if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
                    DblNumVec& localRho = hamDG.Density().LocalMap()[key];
                    blas::Scal( localRho.Size(), rhofac, localRho.Data(), 1 );
                  } // own this element
          } // for (i)
          #endif
          // Update the potential after mixing for the next iteration.  
          // This is only used for potential mixing
      
          // Compute the exchange-correlation potential and energy from the
          // new density
          if( isCalculateGradRho_  ){
            GetTime( timeSta );
            hamDG.CalculateGradDensity(  *distfftPtr_ );
            GetTime( timeEnd );
          #if ( _DEBUGlevel_ >= 1 )
            statusOFS << "Time for calculating gradient of density is " <<
              timeEnd - timeSta << " [s]" << std::endl << std::endl;
          #endif
          }
      
          GetTime( timeSta );
          hamDG.CalculateXC( Exc_, hamDG.Epsxc(), hamDG.Vxc(), *distfftPtr_ );
          //statusOFS << "Exc after DIAG " << Exc_ << std::endl;
          GetTime( timeEnd );
          #if ( _DEBUGlevel_ >= 1 )
          statusOFS << "Time for computing Exc in the global domain is " <<
              timeEnd - timeSta << " [s]" << std::endl << std::endl;
          #endif
      
          GetTime( timeSta );
          hamDG.CalculateHartree( hamDG.Vhart(), *distfftPtr_ );
          GetTime( timeEnd );
          #if ( _DEBUGlevel_ >= 1 )
          statusOFS << "InnerSCF: Time for computing Vhart in the global domain is " <<
            timeEnd - timeSta << " [s]" << std::endl << std::endl;
          #endif
      
          // Compute the second order accurate energy functional.
          // NOTE: In computing the second order energy, the density and the
          // potential must be the OUTPUT density and potential without ANY
          // MIXING.
          // CalculateSecondOrderEnergy();
          // Compute the KS energy 
          // CalculateKSEnergy();
      
          hamDG.CalculateVtot( hamDG.Vtot() );
          GetTime( timeEnd );
          #if ( _DEBUGlevel_ >= 1 )
          statusOFS << "InnerSCF: Time for computing Vtot in the global domain is " <<
           timeEnd - timeSta << " [s]" << std::endl << std::endl;
          #endif
        
        } // if(density ) 

      } //isSCFConverged

//        if( mpirank == 0 ){
//          PrintState( );
//        }
//        statusOFS << " nested Inner HFX energy with old HFX and updated DM "<< std::endl;        
//        hamDG.CalculateDGHFXEnergy( Ehfxin, distDMMat_, hamDG.HFXMat() );
//        statusOFS << " Ehfxin  " << Ehfxin << std::endl;
 
        GetTime( timeIterEnd );
    
        statusOFS << "Time for this inner SCF iteration = " << timeIterEnd - timeIterSta
          << " [s]" << std::endl;

      }

      GetTime( timeHFXIterEnd );
#if ( _DEBUGlevel_ >= 1 )
      statusOFS << "Total wall clock time for this HFX iteration = " <<
        timeHFXIterEnd - timeHFXIterSta << " [s]" << std::endl;
#endif

      GetTime( timeSta );
//      statusOFS << " nested Outer HFX energy with old HFX and new DM "<< std::endl;
      hamDG.CalculateDGHFXEnergy( Ehfx1, distDMMat_, hamDG.HFXMat() );
      GetTime( timeEnd );
//      statusOFS << " Ehfx1  "<< Ehfx1 << std::endl;

      #if ( _DEBUGlevel_ >= 1 )
      statusOFS << "InnerSCF: Time for computing HFX energy after inner DFT iteration " <<
        timeEnd - timeSta << " [s]" << std::endl << std::endl;
      #endif

      GetTime( timeSta );
      if(esdfParam.isDGHFISDF){
        hamDG.CalculateDGHFXMatrix_ISDF( Ehfx_, distDMMat_ );
      }
      else{
        hamDG.CalculateDGHFXMatrix( distDMMat_ );
      }
      GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 1 )
      statusOFS << "Time for updating HFX matrix from new densty matrix is " <<
          timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

      GetTime( timeSta );
      statusOFS << " nested Outer HFX Energy with new HFX and DM "<< std::endl;
      hamDG.CalculateDGHFXEnergy( Ehfx2, distDMMat_, hamDG.HFXMat() );

      statusOFS << " Ehfx0 " << Ehfx0 << std::endl;
      statusOFS << " Ehfx1 " << Ehfx1 << std::endl;
      statusOFS << " Ehfx2  "<< Ehfx2 << std::endl;

#if ( _DEBUGlevel_ >= 1 )
      statusOFS << "Time for computing HFX energy is " <<
          timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

      dExx = 4.0 * std::abs( Ehfx1 - 0.5 * ( Ehfx0 + Ehfx2 ) );
      // use scfNorm to reflect dExx
//      scfNorm_ = dExx;

      Ehfx0 = Ehfx2;
      Ehfx_ = Ehfx2;

      Etot_ = Etot_ -  2.0 * Ehfx1 + Ehfx_;
      Efree_ = Efree_ - 2.0 * Ehfx1 + Ehfx_;
      EfreeHarris_ = EfreeHarris_ - 2.0 * Ehfx1 + Ehfx_;

      Print(statusOFS, "Fock energy       = ",  Ehfx_, "[au]");
      Print(statusOFS, "Etot(with fock)   = ",  Etot_, "[au]");
      Print(statusOFS, "Efree(with fock)  = ",  Efree_, "[au]");
      Print(statusOFS, "dExx              = ",  dExx, "[au]");

      if( dExx < scfHFXEnergyTolerance_ ){

        statusOFS << " Nested HFX Iteration is Converged in "
          << innerHFXIter << " steps !" << std::endl;

        isHFXIterConverged = true;
      }

      if ( isHFXIterConverged ) break;

    } // for(HFXIter)

    GetTime( timeEnd );
    statusOFS << "Time for using nested method is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;

  }  // hybridMixType == "nested"
  else
  {

    if( hamDG.IsHybrid() && hamDG.IsEXXActive() ) {
      scfInnerMaxIter_   = HybridscfInnerMaxIter_;
      scfInnerTolerance_ = HybridscfInnerTolerance_;
      InnermixVariable_  = HybridInnermixVariable_;
      mixType_ = HybridmixType_;
      mixMaxDim_ = HybridmixMaxDim_;
      mixStepLength_ = HybridmixStepLength_;

      statusOFS << " DIIS Scheme for Hybrid DFT Calculations" << std::endl;
      statusOFS << " Mixing variable " << InnermixVariable_ << std::endl;
      statusOFS << " Mixing type     " << mixType_ << std::endl;
    }

    for( Int innerIter = 1; innerIter <= scfInnerMaxIter_; innerIter++ ){
      if ( isInnerSCFConverged ) break;

      scfTotalInnerIter_++;
  
      statusOFS << std::endl << "Inner SCF iteration #"  
        << innerIter << " starts." << std::endl << std::endl;
  
      GetTime( timeIterSta );
  //    statusOFS << " scfTotalInnerIter_ " << scfTotalInnerIter_ << std::endl;
  
      // *********************************************************************
      // Update potential and construct/update the DG matrix
      // *********************************************************************
  
      if( innerIter == 1 ){
//        statusOFS << " InnerDG SCF " << innerIter  << " : Init DG Hamiltonian Matrix" << std::endl;
        // The first inner iteration does not update the potential, and
        // construct the global Hamiltonian matrix from scratch
        GetTime(timeSta);
        hamDG.CalculateDGMatrix( );
        GetTime( timeEnd );
  
  #if ( _DEBUGlevel_ >= 1 )
        statusOFS << "Time for constructing the DG matrix is " <<
          timeEnd - timeSta << " [s]" << std::endl << std::endl;
  #endif
  
        if( hamDG.IsEXXActive()){
          if(esdfParam.isDGHFISDF){
            hamDG.CalculateDGHFXMatrix_ISDF( Ehfx_, distDMMat_ );
          }
          else{
            //statusOFS << "InnerSCF : Calculate DG HFX Matrix " << std::endl;
            hamDG.CalculateDGHFXMatrix( distDMMat_ );
//            statusOFS << "InnerSCF : Calculate DG HFX Energy " << std::endl;
            hamDG.CalculateDGHFXEnergy( Ehfx_, distDMMat_, hamDG.HFXMat() );
//            statusOFS << "InnerSCF HFX end " << std::endl;
          }

          hamDG.MinusDGHFXMatrix( );

          GetTime( timeEnd );
  #if ( _DEBUGlevel_ >= 1 )
          statusOFS << "InnerSCF: Time for constructing the DGHFX matrix is " <<
            timeEnd - timeSta << " [s]" << std::endl << std::endl;
  #endif
        }// End if ( hamDG.IsHybrid() )
      }
      else{
        // The consequent inner iterations update the potential in the
        // element, and only update the global Hamiltonian matrix
  
        // Update the potential in the element (and the extended element)
   
//        statusOFS << " InnerDG SCF " << innerIter  << " : Update DG Hamiltonian Matrix" << std::endl;
       
        GetTime(timeSta);
        // Save the old potential on the LGL grid
        for( Int k = 0; k < numElem_[2]; k++ )
          for( Int j = 0; j < numElem_[1]; j++ )
            for( Int i = 0; i < numElem_[0]; i++ ){
              Index3 key( i, j, k );
              if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
                Index3 numLGLGrid     = hamDG.NumLGLGridElem();
                blas::Copy( numLGLGrid.prod(),
                    hamDG.VtotLGL().LocalMap()[key].Data(), 1,
                    vtotLGLSave_.LocalMap()[key].Data(), 1 );
              } // if (own this element)
        } // for (i)
  
        // Update the local potential on the extended element and on the
        // element.
        UpdateElemLocalPotential();
  
        // Save the difference of the potential on the LGL grid into vtotLGLSave_
        for( Int k = 0; k < numElem_[2]; k++ )
          for( Int j = 0; j < numElem_[1]; j++ )
            for( Int i = 0; i < numElem_[0]; i++ ){
              Index3 key( i, j, k );
              if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
                Index3 numLGLGrid     = hamDG.NumLGLGridElem();
                Real *ptrNew = hamDG.VtotLGL().LocalMap()[key].Data();
                Real *ptrDif = vtotLGLSave_.LocalMap()[key].Data();
                for( Int p = 0; p < numLGLGrid.prod(); p++ ){
                  (*ptrDif) = (*ptrNew) - (*ptrDif);
                  ptrNew++;
                  ptrDif++;
                }
              } // if (own this element)
            } // for (i)
  
        GetTime( timeEnd );
  #if ( _DEBUGlevel_ >= 1 )
        statusOFS << "InnerSCF:: Time for updating the local potential in the extended element and the element is " <<
          timeEnd - timeSta << " [s]" << std::endl;
  #endif
  
//        statusOFS << " Inner SCF " << innerIter  << " : Save DG Hamiltonian Matrix" << std::endl;
        // Save old hamiltonian before mixing
        if( InnermixVariable_ == "densitymatrix" || InnermixVariable_ == "hamiltonian" ){
  
          for(typename std::map<ElemMatKey, DblNumMat >::iterator
              Ham_iterator = hamDG.HMat().LocalMap().begin();
              Ham_iterator != hamDG.HMat().LocalMap().end();
              ++ Ham_iterator )
          {
            ElemMatKey matkey = (*Ham_iterator).first;
            DblNumMat& oldMat = hamDG.HMat().LocalMap()[matkey];
            std::map<ElemMatKey, DblNumMat>::iterator mi =
              distHMatSave_.LocalMap().find( matkey );
              if( mi == distHMatSave_.LocalMap().end() ){
                distHMatSave_.LocalMap()[matkey] = oldMat;
              }
              else{
                DblNumMat&  mat = (*mi).second;
                blas::Copy( mat.Size(), oldMat.Data(), 1,
                  mat.Data(), 1);
              }
          }
        }
  
  //      // Update the DG Matrix
  //      GetTime(timeSta);
  //      hamDG.UpdateDGMatrix( vtotLGLSave_ );
  //
  ////      MPI_Barrier( domain_.comm );
  //      GetTime( timeEnd );
  //#if ( _DEBUGlevel_ >= 1 )
  //      statusOFS << "InnerSCF: Time for updating the DG matrix is " <<
  //        timeEnd - timeSta << " [s]" << std::endl << std::endl;
  //#endif
  
        // Recalculate DG Matrix
        if( hamDG.IsHybrid() && hamDG.IsEXXActive()){
          GetTime(timeSta);
          hamDG.CalculateDGMatrix( );
          MPI_Barrier( domain_.comm );
          GetTime( timeEnd );
   #if ( _DEBUGlevel_ >= 1 )
          statusOFS << "Time for recalculating the DG matrix is " <<
            timeEnd - timeSta << " [s]" << std::endl << std::endl;
  #endif
          GetTime( timeSta );
          if(esdfParam.isDGHFISDF){
            hamDG.CalculateDGHFXMatrix_ISDF( Ehfx_, distDMMat_ );
          }
          else{
            hamDG.CalculateDGHFXMatrix( distDMMat_ );
            hamDG.CalculateDGHFXEnergy( Ehfx_, distDMMat_, hamDG.HFXMat() );
          }

          hamDG.MinusDGHFXMatrix( );

          GetTime( timeEnd );
  #if ( _DEBUGlevel_ >= 1 )
          statusOFS << "InnerSCF: Time for constructing the DGHFX matrix is " <<
            timeEnd - timeSta << " [s]" << std::endl << std::endl;
  #endif
        }// End if ( hamDG.IsHybrid() )
        else{
          GetTime(timeSta);
          hamDG.UpdateDGMatrix( vtotLGLSave_ );
          MPI_Barrier( domain_.comm );
          GetTime( timeEnd );
   #if ( _DEBUGlevel_ >= 1 )
          statusOFS << "Time for updating the DG matrix is " <<
            timeEnd - timeSta << " [s]" << std::endl << std::endl;
  #endif
        }
  
        // Mixing for the inner SCF iteration.
        if( InnermixVariable_ == "hamiltonian" ){
//        statusOFS << " InnerDG SCF " << innerIter  << " : Mixing DG Hamiltonian Matrix" << std::endl;  
          GetTime( timeSta );
          Int  HamIter = innerIter - 1;
          
//          statusOFS << "  InnerIter " << innerIter << std::endl;
//          statusOFS << " HamIter " << HamIter  << std::endl;
          
          if( mixType_ == "kerker+anderson" ){
            //statusOFS << " anderson hamiltonian mixing " << std::endl;
            AndersonMix(
              HamIter,
              HybridmixStepLength_,
              mixType_,
              hamDG.HMat(),
              distHMatSave_,
              hamDG.HMat(),
              distdfInnerMat_,
              distdvInnerMat_);
          }
          else if( mixType_ == "anderson" ){
           // statusOFS << " anderson hamiltonian mixing " << std::endl;
            AndersonMix2(
              HamIter,
              HybridmixStepLength_,
              hamDG.HMat(),
              distHMatSave_,
              hamDG.HMat(),
              distdfInnerMat_,
              distdvInnerMat_);
          }
          else if( mixType_ == "pulay" ){
           // statusOFS << " pulay hamiltonian mixing " << std::endl;
            PulayMix(
              HamIter,
              HybridmixStepLength_,
              hamDG.HMat(),
              distHMatSave_,
              hamDG.HMat(),
              distdfInnerMat_,
              distdvInnerMat_);
          }
          else if( mixType_ == "broyden" ){
            //statusOFS << " broyden hamiltonian mixing " << std::endl;
            BroydenMix(
              HamIter,
              HybridmixStepLength_,
              hamDG.HMat(),
              distHMatSave_,
              hamDG.HMat(),
              distdfInnerMat_,
              distdvInnerMat_,
              distcdfInnerMat_);
          }
          else{
            ErrorHandling("Invalid hamiltonian mixing type.");
          }
  
          GetTime( timeEnd );
  #if ( _DEBUGlevel_ >= 1 )
          statusOFS << "Time for Hamiltonian mixing is " <<
             timeEnd - timeSta << " [s]" << std::endl << std::endl;
  #endif
        }
      } // InnerIter > 1
  
      // *********************************************************************
      // Write the Hamiltonian matrix to a file (if needed) 
  
      if( esdfParam.isOutputHMatrix ){
        // Only the first processor column participates in the conversion
        if( mpirankRow == 0 ){
          DistSparseMatrix<Real>  HSparseMat;
  
          GetTime(timeSta);
          DistElemMatToDistSparseMat(
              hamDG.HMat(),
              hamDG.NumBasisTotal(),
              HSparseMat,
              hamDG.ElemBasisIdx(),
              domain_.colComm );
          GetTime(timeEnd);
  #if ( _DEBUGlevel_ >= 1 )
          statusOFS << "InnerSCF: Time for converting the DG matrix to DistSparseMatrix format is " <<
            timeEnd - timeSta << " [s]" << std::endl << std::endl;
  #endif
          GetTime(timeSta);
          ParaWriteDistSparseMatrix( "H.csc", HSparseMat );
                //            WriteDistSparseMatrixFormatted( "H.matrix", HSparseMat );
          GetTime(timeEnd);
  #if ( _DEBUGlevel_ >= 1 )
          statusOFS << "InnerSCF: Time for writing the matrix in parallel is " <<
            timeEnd - timeSta << " [s]" << std::endl << std::endl;
  #endif
        }
  
        MPI_Barrier( domain_.comm );
      }
  
     //*********************************************************************   
  
  #if 0
  //    if( (innerIter > 1) && (InnermixVariable_ == "hamiltonian" ) ){
      if( InnermixVariable_ == "hamiltonian" || InnermixVariable_ == "densitymatrix" ){
        Real MaxDifLocal = 0.0, MaxDif = 0.0;
      
        for(typename std::map<ElemMatKey, DblNumMat >::iterator
          Ham_iterator = hamDG.HMat().LocalMap().begin();
          Ham_iterator != hamDG.HMat().LocalMap().end();
            ++ Ham_iterator ) {
          ElemMatKey matkey = (*Ham_iterator).first;
          DblNumMat& oldMat = distHMatSave_.LocalMap()[matkey];
          DblNumMat& newMat = hamDG.HMat().LocalMap()[matkey];
  
          for( Int q = 0; q < oldMat.n(); q++ ){
            for( Int p = 0; p < oldMat.m(); p++ ){
               Real diffMat = std::abs( oldMat(p, q) - newMat(p, q) );
               MaxDifLocal = std::max( MaxDifLocal, diffMat );
            }
          }
        }
  
        mpi::Allreduce( &MaxDifLocal, &MaxDif, 1, MPI_MAX, domain_.colComm );
        scfInnerHamMaxDif_    = MaxDif;
  #if ( _DEBUGlevel_ >= 0 )
        Print(statusOFS, "Inner Ham MaxDiff(out-in) = ", scfInnerHamMaxDif_ );
  #endif
  
        GetTime( timeEnd );
  #if ( _DEBUGlevel_ >= 1 )
        statusOFS << "InnerSCF: Time for computing the Hamiltonian residual is " <<
              timeEnd - timeSta << " [s]" << std::endl << std::endl;
  #endif
  
        if( scfInnerHamMaxDif_ < scfInnerHamTolerance_ ){
           /* converged */
           Print( statusOFS, "Inner SCF is converged!\n" );
           isInnerSCFConverged = true;
        }
  
      }
  #endif
  
      // *********************************************************************
      //  Save the mixing variable first
      {
//        statusOFS << " InnerDG SCF " << innerIter  << " : Save DG Potential/Density/Denisty Matrix" << std::endl;
  
        if( InnermixVariable_ == "density" || InnermixVariable_ == "potential" ){
          for( Int k = 0; k < numElem_[2]; k++ )
            for( Int j = 0; j < numElem_[1]; j++ )
              for( Int i = 0; i < numElem_[0]; i++ ){
                Index3 key( i, j, k );
                if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
                  if( InnermixVariable_ == "density" ){
                    DblNumVec& oldVec = hamDG.Density().LocalMap()[key];
  //                  DblNumVec& newVec = mixInnerSave_.LocalMap()[key];
  //                  blas::Copy( oldVec.Size(), oldVec.Data(), 1, newVec.Data(), 1 );
                    mixInnerSave_.LocalMap()[key] = oldVec;
                  }
                  else if( InnermixVariable_ == "potential" ){
                    DblNumVec& oldVec = hamDG.Vtot().LocalMap()[key];
  //                  DblNumVec& newVec = mixInnerSave_.LocalMap()[key];
  //                  blas::Copy( oldVec.Size(), oldVec.Data(), 1, newVec.Data(), 1 );
                    mixInnerSave_.LocalMap()[key] = oldVec;
                  }
                } // own this element
          } // for (i)
        }
        else if( InnermixVariable_ == "densitymatrix" || InnermixVariable_ == "hamiltonian" ){
          for(typename std::map<ElemMatKey, DblNumMat >::iterator
            DM_iterator =  distDMMat_.LocalMap().begin();
            DM_iterator !=  distDMMat_.LocalMap().end();
            ++ DM_iterator ) 
          {
            ElemMatKey matkey = (*DM_iterator).first;
            DblNumMat& oldMat = distDMMat_.LocalMap()[matkey];
            std::map<ElemMatKey, DblNumMat>::iterator mi =
              distDMMatSave_.LocalMap().find( matkey );
              if( mi == distDMMatSave_.LocalMap().end() ){
                distDMMatSave_.LocalMap()[matkey] = oldMat;
              }
              else{
                DblNumMat&  mat = (*mi).second;
                blas::Copy( mat.Size(), oldMat.Data(), 1,
                  mat.Data(), 1);
              }
          } // ---- End of if( InnermixVariable_ == "densitymatrix" )
        }
      }
  
     
     InnerSolver( outerIter );
  
  
    // **************************************************************************************
    // Compute the error of the mixing variable
  
//    statusOFS << " InnerDG SCF " << innerIter  << " :   Density/Potential/DM/H converge ?" << std::endl;
    GetTime(timeSta);
    {
      Real normMixDifLocal = 0.0, normMixOldLocal = 0.0;
      Real normMixDif, normMixOld;
  
      Real MaxDifLocal = 0.0, MaxDif = 0.0;
  
      if( InnermixVariable_ == "density" || InnermixVariable_ == "potential" ){
        for( Int k = 0; k < numElem_[2]; k++ ){
          for( Int j = 0; j < numElem_[1]; j++ ){
            for( Int i = 0; i < numElem_[0]; i++ ){
              Index3 key( i, j, k );
              if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
                if( InnermixVariable_ == "density" ){
                  DblNumVec& oldVec = mixInnerSave_.LocalMap()[key];
                  DblNumVec& newVec = hamDG.Density().LocalMap()[key];
  
                  for( Int p = 0; p < oldVec.m(); p++ ){
                    normMixDifLocal += pow( oldVec(p) - newVec(p), 2.0 );
                    normMixOldLocal += pow( oldVec(p), 2.0 );
                  }
                }
                else if( InnermixVariable_ == "potential" ){
                  DblNumVec& oldVec = mixInnerSave_.LocalMap()[key];
                  DblNumVec& newVec = hamDG.Vtot().LocalMap()[key];
  
                  for( Int p = 0; p < oldVec.m(); p++ ){
                    normMixDifLocal += pow( oldVec(p) - newVec(p), 2.0 );
                    normMixOldLocal += pow( oldVec(p), 2.0 );
                  }
                }
              } // own this element
            } // for (i)
          } // for(j)
        } // for (k)
  
        mpi::Allreduce( &normMixDifLocal, &normMixDif, 1, MPI_SUM,
          domain_.colComm );
        mpi::Allreduce( &normMixOldLocal, &normMixOld, 1, MPI_SUM,
          domain_.colComm );
    
        normMixDif = std::sqrt( normMixDif );
        normMixOld = std::sqrt( normMixOld );
    
        scfInnerNorm_    = normMixDif / normMixOld;
  
  #if ( _DEBUGlevel_ >= 0 )
        Print(statusOFS, "norm(MixDif)          = ", normMixDif );
        Print(statusOFS, "norm(MixOld)          = ", normMixOld );
        Print(statusOFS, "norm(out-in)/norm(in) = ", scfInnerNorm_ );
  #endif
        if( scfInnerNorm_ < scfInnerTolerance_ ){
          /* converged */
          Print( statusOFS, "Inner SCF is Converged!\n" );
          isInnerSCFConverged = true;
        }
      }
      else if( InnermixVariable_ == "densitymatrix" || InnermixVariable_ == "hamiltonian"){  
        for(typename std::map<ElemMatKey, DblNumMat >::iterator
          DM_iterator = distDMMat_.LocalMap().begin();
          DM_iterator !=  distDMMat_.LocalMap().end();
          ++ DM_iterator ) {
          ElemMatKey matkey = (*DM_iterator).first;
          DblNumMat& oldMat = distDMMatSave_.LocalMap()[matkey];
          DblNumMat& newMat =  distDMMat_.LocalMap()[matkey];
  
          for( Int q = 0; q < oldMat.n(); q++ ){
            for( Int p = 0; p < oldMat.m(); p++ ){
              Real diffMat = std::abs( oldMat(p, q) - newMat(p, q) );
              MaxDifLocal = std::max( MaxDifLocal, diffMat );
            }
          }
        }
        mpi::Allreduce( &MaxDifLocal, &MaxDif, 1, MPI_MAX, domain_.colComm );
        scfInnerDMMaxDif_    = MaxDif;
  #if ( _DEBUGlevel_ >= 0 )
        Print(statusOFS, "Inner DM MaxDiff(out-in) = ", scfInnerDMMaxDif_ );
  #endif
        MaxDifLocal = 0.0;
        MaxDif = 0.0;
        for(typename std::map<ElemMatKey, DblNumMat >::iterator
          Ham_iterator = hamDG.HMat().LocalMap().begin();
          Ham_iterator != hamDG.HMat().LocalMap().end();
            ++ Ham_iterator ) {
          ElemMatKey matkey = (*Ham_iterator).first;
          DblNumMat& oldMat = distHMatSave_.LocalMap()[matkey];
          DblNumMat& newMat = hamDG.HMat().LocalMap()[matkey];
  
          for( Int q = 0; q < oldMat.n(); q++ ){
            for( Int p = 0; p < oldMat.m(); p++ ){
               Real diffMat = std::abs( oldMat(p, q) - newMat(p, q) );
               MaxDifLocal = std::max( MaxDifLocal, diffMat );
            }
          }
        }
  
        mpi::Allreduce( &MaxDifLocal, &MaxDif, 1, MPI_MAX, domain_.colComm );
        scfInnerHamMaxDif_    = MaxDif;
  #if ( _DEBUGlevel_ >= 0 )
        Print(statusOFS, "Inner Ham MaxDiff(out-in) = ", scfInnerHamMaxDif_ );
  #endif
  
        if( scfInnerHamMaxDif_ < scfInnerHamTolerance_ || scfInnerDMMaxDif_ < scfInnerDMTolerance_ ){
           /* converged */
           Print( statusOFS, "Inner SCF is Converged!\n" );
           isInnerSCFConverged = true;
        }
      }
  
  //  MPI_Barrier( domain_.colComm );
  //  MPI_Barrier( domain_.rowComm ); 
  //  MPI_Barrier( domain_.comm ); 
    GetTime( timeEnd );
  #if ( _DEBUGlevel_ >= 1 )
    statusOFS << "Time for computing the SCF residual is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;
  #endif
    }
 
//   if(!isInnerSCFConverged){ 
   {
    // Mixing for the inner SCF iteration.
    GetTime( timeSta );
  
    // The number of iterations used for Anderson mixing
    Int numIter;
  
    if( scfInnerMaxIter_ == 1 ){
      // Maximum inner iteration = 1 means there is no distinction of
      // inner/outer SCF.  Anderson mixing uses the global history
      numIter = scfTotalInnerIter_;
    }
    else{
      // If more than one inner iterations is used, then Anderson only
      // uses local history.  For explanation see 
      numIter = innerIter;
    }
  
  //  if(InnermixVariable_ == "densitymatrix") numIter = innerIter - 1;
  
//    statusOFS << " InnerDG SCF " << innerIter  << " :   Mixing Density/Potential/DM ?" << std::endl;
//    statusOFS <<" InnermixVariable_  " << InnermixVariable_ << std::endl;
//    statusOFS << " mixType_  " <<  mixType_ << std::endl;
//    statusOFS << " innerIter " << innerIter  << std::endl;
//    statusOFS << " outerIter " << outerIter  << std::endl;
//    statusOFS << " scfTotalInnerIter_  " <<  scfTotalInnerIter_ << std::endl;
//    statusOFS << " numIter " <<  numIter << std::endl << std::endl;
  
    if( InnermixVariable_ == "density" ){
      if( mixType_ == "anderson" ){
       // statusOFS << " anderson density mixing " << std::endl;
        AndersonMix2(
            numIter, 
            mixStepLength_,
            hamDG.Density(),
            mixInnerSave_,
            hamDG.Density(),
            dfInnerMat_,
            dvInnerMat_);
      }
      else if( mixType_ == "kerker+anderson"){
       // statusOFS << " kerker+anderson density mixing " << std::endl;
        AndersonMix(
            numIter, 
            mixStepLength_,
            mixType_,
            hamDG.Density(),
            mixInnerSave_,
            hamDG.Density(),
            dfInnerMat_,
            dvInnerMat_);
      }
      else if( mixType_ == "pulay"){
       // statusOFS << " pulay density mixing " << std::endl;
        PulayMix(
            numIter, 
            mixStepLength_,
            hamDG.Density(),
            mixInnerSave_,
            hamDG.Density(),
            dfInnerMat_,
            dvInnerMat_);
      }
      else if( mixType_ == "broyden" ){
       // statusOFS << " broyden density mixing " << std::endl;
        BroydenMix(
            numIter,
            mixStepLength_,
            hamDG.Density(),
            mixInnerSave_,
            hamDG.Density(),
            dfInnerMat_,
            dvInnerMat_,
            cdfInnerMat_);
      }
      else{
        ErrorHandling("Invalid density mixing type.");
      }
    }
    else if( InnermixVariable_ == "potential" ){
      if( mixType_ == "anderson"){
       // statusOFS << " anderson potential mixing " << std::endl;
        AndersonMix2(
            numIter,
            mixStepLength_,
            hamDG.Vtot(),
            mixInnerSave_,
            hamDG.Vtot(),
            dfInnerMat_,
            dvInnerMat_);
      }
      else if( mixType_ == "kerker+anderson"    ){
       // statusOFS << " kerker+anderson potential mixing " << std::endl;
        AndersonMix(
            numIter, 
            mixStepLength_,
            mixType_,
            hamDG.Vtot(),
            mixInnerSave_,
            hamDG.Vtot(),
            dfInnerMat_,
            dvInnerMat_);
      }
      else if( mixType_ == "pulay"    ){
       // statusOFS << " pulay potential mixing " << std::endl;
        PulayMix(
            numIter, 
            mixStepLength_,
            hamDG.Vtot(),
            mixInnerSave_,
            hamDG.Vtot(),
            dfInnerMat_,
            dvInnerMat_);
      }
      else if( mixType_ == "broyden" ){
       // statusOFS << " broyden potential mixing " << std::endl;
        BroydenMix(
            numIter,
            mixStepLength_,
            hamDG.Vtot(),
            mixInnerSave_,
            hamDG.Vtot(),
            dfInnerMat_,
            dvInnerMat_,
            cdfInnerMat_);
      }
      else{
        ErrorHandling("Invalid potential mixing type.");
      }
    }
    else if( InnermixVariable_ == "densitymatrix" ){
      if( mixType_ == "kerker+anderson" ){
      //  statusOFS << " kerker+anderson1 densitymatrix mixing " << std::endl;
        AndersonMix(
              numIter,
              HybridmixStepLength_,
              mixType_,
              distDMMat_,
              distDMMatSave_,
              distDMMat_,
              distdfInnerMat_,
              distdvInnerMat_);
      } 
      else if( mixType_ == "anderson" ){
        //statusOFS << " anderson2 densitymatrix mixing " << std::endl;
        AndersonMix2(
              numIter,
              HybridmixStepLength_,
  //            mixType_,
              distDMMat_,
              distDMMatSave_,
              distDMMat_,
              distdfInnerMat_,
              distdvInnerMat_);
      }
      else if( mixType_ == "pulay" ){
       // statusOFS << " pulay densitymatrix mixing " << std::endl;
        PulayMix(
              numIter,
              HybridmixStepLength_,
              distDMMat_,
              distDMMatSave_,
              distDMMat_,
              distdfInnerMat_,
              distdvInnerMat_);
      }
      else if( mixType_ == "broyden" ){
       // statusOFS << " broyden densitymatrix mixing " << std::endl;
        BroydenMix(
              numIter,
              HybridmixStepLength_,
              distDMMat_,
              distDMMatSave_,
              distDMMat_,
              distdfInnerMat_,
              distdvInnerMat_,
              distcdfInnerMat_);
      }
      else{
        ErrorHandling("Invalid densitymatrix mixing type.");
      } 
    }
  
    MPI_Barrier( domain_.comm );
    GetTime( timeEnd );
  #if ( _DEBUGlevel_ >= 1 )
    statusOFS << "InnerSCF: Time for mixing is " <<
      timeEnd - timeSta << " [s]" << std::endl;
  #endif
  
    // Post processing for the density mixing. Make sure that the
    // density is positive, and compute the potential again. 
    // This is only used for density mixing.
    if( InnermixVariable_ == "densitymatrix" ){
//      statusOFS << " InnerDG SCF " << innerIter  << " :   New Density from mixed DM ?" << std::endl;
  //    statusOFS << "InnerSCF: Recalculate density from mixed density matrix " << std::endl;
      hamDG.CalculateDensityDM2( hamDG.Density(), hamDG.DensityLGL(), distDMMat_ );
    }
  
    if( (InnermixVariable_ == "density") || (InnermixVariable_ == "densitymatrix") )
    {
//      statusOFS << " InnerDG SCF " << innerIter  << " : Normalize mixed Density on Uniform GRID" << std::endl;
      Real sumRhoLocal = 0.0;
      Real sumRho;
      for( Int k = 0; k < numElem_[2]; k++ )
        for( Int j = 0; j < numElem_[1]; j++ )
          for( Int i = 0; i < numElem_[0]; i++ ){
            Index3 key( i, j, k );
            if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
              DblNumVec&  density      = hamDG.Density().LocalMap()[key];
  
              for (Int p=0; p < density.Size(); p++) {
                density(p) = std::max( density(p), 0.0 );
                sumRhoLocal += density(p);
              }
            } // own this element
          } // for (i)
  
      mpi::Allreduce( &sumRhoLocal, &sumRho, 1, MPI_SUM, domain_.colComm );
      sumRho *= domain_.Volume() / domain_.NumGridTotalFine();
  
      Real rhofac = hamDG.NumSpin() * hamDG.NumOccupiedState() / sumRho;
  #if ( _DEBUGlevel_ >= 1 )
  //    statusOFS << std::endl;
  //    Print( statusOFS, "Numer of Occupied State ",  hamDG.NumOccupiedState() );
      Print( statusOFS, "Rho factor after mixing (raw data) = ", rhofac );
      Print( statusOFS, "Sum Rho after mixing (raw data)    = ", sumRho );
  //    statusOFS << std::endl;
  #endif
      // Normalize the electron density in the global domain
  #if 1
      for( Int k = 0; k < numElem_[2]; k++ )
        for( Int j = 0; j < numElem_[1]; j++ )
          for( Int i = 0; i < numElem_[0]; i++ ){
            Index3 key( i, j, k );
              if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
                DblNumVec& localRho = hamDG.Density().LocalMap()[key];
                blas::Scal( localRho.Size(), rhofac, localRho.Data(), 1 );
              } // own this element
      } // for (i)
  #endif
      // Update the potential after mixing for the next iteration.  
      // This is only used for potential mixing
  
      // Compute the exchange-correlation potential and energy from the
      // new density
//      statusOFS << " InnerDG SCF " << innerIter  << " : Update DG Potential " << std::endl;
  
      if( isCalculateGradRho_  ){
        GetTime( timeSta );
        hamDG.CalculateGradDensity(  *distfftPtr_ );
        GetTime( timeEnd );
  #if ( _DEBUGlevel_ >= 1 )
        statusOFS << "Time for calculating gradient of density is " <<
          timeEnd - timeSta << " [s]" << std::endl << std::endl;
  #endif
      }
  
      GetTime( timeSta );
      hamDG.CalculateXC( Exc_, hamDG.Epsxc(), hamDG.Vxc(), *distfftPtr_ );
//      statusOFS << "Exc after DIAG " << Exc_ << std::endl;
      GetTime( timeEnd );
  #if ( _DEBUGlevel_ >= 1 )
      statusOFS << "Time for computing Exc in the global domain is " <<
          timeEnd - timeSta << " [s]" << std::endl << std::endl;
  #endif
  
      GetTime( timeSta );
      hamDG.CalculateHartree( hamDG.Vhart(), *distfftPtr_ );
      GetTime( timeEnd );
  #if ( _DEBUGlevel_ >= 1 )
      statusOFS << "InnerSCF: Time for computing Vhart in the global domain is " <<
        timeEnd - timeSta << " [s]" << std::endl << std::endl;
  #endif
  
      // Compute the second order accurate energy functional.
      // NOTE: In computing the second order energy, the density and the
      // potential must be the OUTPUT density and potential without ANY
      // MIXING.
  //    CalculateSecondOrderEnergy();
      // Compute the KS energy 
  //    CalculateKSEnergy();
  
      hamDG.CalculateVtot( hamDG.Vtot() );
      GetTime( timeEnd );
  #if ( _DEBUGlevel_ >= 1 )
      statusOFS << "InnerSCF: Time for computing Vtot in the global domain is " <<
       timeEnd - timeSta << " [s]" << std::endl << std::endl;
  #endif
    } // if(density + densitymatrix ) 
  
  #ifdef _USE_PEXSI_
    if( solutionMethod_ == "pexsi" )
    {
      Real deltaVmin = 0.0;
      Real deltaVmax = 0.0;
  
      for( Int k=0; k< numElem_[2]; k++ )
        for( Int j=0; j< numElem_[1]; j++ )
          for( Int i=0; i< numElem_[0]; i++ ) {
            Index3 key = Index3(i,j,k);
            if( distEigSolPtr_->Prtn().Owner(key) == (mpirank / dmRow_) ){
              DblNumVec vtotCur;
              vtotCur = hamDG.Vtot().LocalMap()[key];
              DblNumVec& oldVtot = VtotHist_.LocalMap()[key];
              blas::Axpy( vtotCur.m(), -1.0, oldVtot.Data(),
                              1, vtotCur.Data(), 1);
              deltaVmin = std::min( deltaVmin, findMin(vtotCur) );
              deltaVmax = std::max( deltaVmax, findMax(vtotCur) );
            }
      }
  
      {
        Int color = mpirank % dmRow_;
        MPI_Comm elemComm;
        std::vector<Real> vlist(mpisize/dmRow_);
  
        MPI_Comm_split( domain_.comm, color, mpirank, &elemComm );
        MPI_Allgather( &deltaVmin, 1, MPI_DOUBLE, &vlist[0], 1, MPI_DOUBLE, elemComm);
        deltaVmin = 0.0;
        for(Int i =0; i < mpisize/dmRow_; i++)
          if(deltaVmin > vlist[i])
            deltaVmin = vlist[i];
  
        MPI_Allgather( &deltaVmax, 1, MPI_DOUBLE, &vlist[0], 1, MPI_DOUBLE, elemComm);
        deltaVmax = 0.0;
        for(Int i =0; i < mpisize/dmRow_; i++)
          if(deltaVmax < vlist[i])
            deltaVmax = vlist[i];
  
        pexsiOptions_.muMin0 += deltaVmin;
        pexsiOptions_.muMax0 += deltaVmax;
        MPI_Comm_free( &elemComm);
      }
    }
  #endif
    // Print out the state variables of the current iteration

   }
 
    // Only master processor output information containing all atoms
//    if( hamDG.IsHybrid() && hamDG.IsEXXActive() ) {
//      Etot_ = Etot_ - Ehfx_;
//      Efree_ = Efree_  - Ehfx_;
//    }

//    if( mpirank == 0 ){
//      PrintState( );
//    }
  
    GetTime( timeIterEnd );
  
    statusOFS << "Time for this inner SCF iteration = " << timeIterEnd - timeIterSta
      << " [s]" << std::endl;
  
    }  // Inner iter

  } //else nested

  return ;
}         // -----  end of method SCFDG::InnerIterate  ----- 

}
