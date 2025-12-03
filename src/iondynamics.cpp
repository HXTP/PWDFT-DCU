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
/// @file iondynamics.cpp
/// @brief Geometry optimization and molecular dynamics for ions
/// @date 2015-03-05 Organize previously implemented methods
#ifndef _IONDYNAMICS_CPP_
#define _IONDYNAMICS_CPP_

#include  "iondynamics.hpp"
#include  "utility.hpp"
#include  <algorithm>		// std::min

namespace dgdft{

using namespace dgdft::esdf;


void
  IonDynamics::Setup    ( std::vector<Atom>& atomList,
      PeriodTable& ptable )
  {
    Int mpirank, mpisize;
    MPI_Comm_rank( MPI_COMM_WORLD, &mpirank );
    MPI_Comm_size( MPI_COMM_WORLD, &mpisize );

    // Read in input parameters
    atomListPtr_            = &atomList;
    ionMove_                = esdfParam.ionMove;
    ionTemperature_         = 1.0 / esdfParam.TbetaIonTemperature;
    MDExtrapolationType_    = esdfParam.MDExtrapolationType;
    dt_                     = esdfParam.MDTimeStep;
    Q1_                     = esdfParam.qMass;
    langevinDamping_        = esdfParam.langevinDamping;

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

    // Determine the mode of the ionDynamics
    isGeoOpt_ = false;
    isMD_     = false;

    if( ionMove_ == "bb" ||
        ionMove_ == "pgbb" ||
        ionMove_ == "nlcg" ||
        ionMove_ == "lbfgs" ||
        ionMove_ == "fire"){
      isGeoOpt_ = true;
    }

    if( ionMove_ == "verlet" ||
        ionMove_ == "nosehoover1" ||
        ionMove_ == "langevin" ){
      isMD_ = true;
    }

    if (isGeoOpt_ == 1){
      fAtInfimum_ = 10000.0;   // set the maxForce value to be very high to start with
    }   

    // Grab the supercell info
    supercell_x_ = esdfParam.domain.length[0];
    supercell_y_ = esdfParam.domain.length[1];
    supercell_z_ = esdfParam.domain.length[2];

    //if( isGeoOpt_ == false && isMD_ == false ){
    //  ErrorHandling("Neither geometry optimization nor MD is invoked.");
    //}


    // Geometry optimization

    if( ionMove_ == "bb" ){
    }

    if( ionMove_ == "pgbb" ){
      geoOptVars_.xtol = 1e-4;
      geoOptVars_.gtol = 1e-4;

      geoOptVars_.tau  = 1e-1; // Original default is 1e-3
      geoOptVars_.rhols = 1e-4;
      geoOptVars_.eta = 0.1;
      geoOptVars_.gamma = 0.85;
      geoOptVars_.STPEPS = 1e-10;
      geoOptVars_.nt = 5;

      geoOptVars_.callType = 0;


      Int numAtom = atomList.size();
      geoOptVars_.atompos.resize(numAtom);
      geoOptVars_.atomposOld.resize(numAtom);
      geoOptVars_.atomforce.resize(numAtom);
      geoOptVars_.atomforceOld.resize(numAtom);

    } // if( pgbb )

    if( ionMove_ == "lbfgs" ){
      geoOptVars_.gtol = 1e-6;

      geoOptVars_.callType = 0;
      geoOptVars_.maxMixingDim = 7;

      Int numAtom = atomList.size();
      Int lwork = (3*numAtom+1)*(2*geoOptVars_.maxMixingDim + 1);
      geoOptVars_.work.Resize(lwork);
    } // if( lbfgs )


    if( ionMove_ == "nlcg" )
    {
      statusOFS << std::endl << " Setting up Non-linear CG based relaxation ...";
      // Set up the parameters and atom / force list for nlcg
      //       int i_max = 50;
      //       int j_max = 0;
      //       int n = 30;
      //       double epsilon_tol_outer = 1e-6;
      //       double epsilon_tol_inner = 1e-6;
      //       double sigma_0 = 0.5;

      int i_max = esdfParam.ionMaxIter;
      int j_max = 5;
      int n = 30;

      // double epsilon_tol_outer = 0.01 * sqrt(esdfParam.geoOptMaxForce * 3.0 * double(atomListPtr_->size()));
      double epsilon_tol_outer =  esdfParam.geoOptMaxForce * sqrt(3.0 * double(atomListPtr_->size()));
      //double epsilon_tol_inner = 1e-5;
      double epsilon_tol_inner = 1e-4;

      double sigma_0 = esdfParam.geoOpt_NLCG_sigma; // Defaults to 0.02

      NLCG_vars.setup(i_max, j_max, n, 
          epsilon_tol_outer, epsilon_tol_inner, sigma_0,
          *atomListPtr_);

      statusOFS << " Done ." << std::endl;

      statusOFS << std::endl << " NLCG parameters : i_max = " << i_max << " , j_max = " << j_max << " , n = " << n;
      statusOFS << std::endl << "                   epsilon_tol_outer = " << epsilon_tol_outer << " , epsilon_tol_inner = " << epsilon_tol_inner 
        << " , sigma_0 = " << sigma_0 << std::endl;                  

    }	// if( nlcg )


    if( ionMove_ == "fire" )
    {
      statusOFS << std::endl << " Setting up Fast Inertial Relaxation Engine (FIRE) - - - > ";

      // Fetching user controlled parameters from esdfParam
      int nMin    = esdfParam.FIRE_Nmin;
      double dt   = esdfParam.FIRE_dt; 		// this is the starting dt
      double mass = esdfParam.FIRE_atomicmass*amu2au; // ***

      // Hard coded parameters (Ref: DOI: 10.1103/PhysRevLett.97.170201) 
      double fInc	  = 1.1;
      double fDec 	  = 0.5;
      double alphaStart = 0.1;
      double alpha 	  = alphaStart;
      double fAlpha 	  = 0.99;
      double dtMax 	  = 10.0*dt;

      // cut_ starts at 0 but keeps on getting updated as the 
      // iterations proceed
      int cut           = 0;

      // Prepare the position, velocity, and force lists
      Int numAtom;

      numAtom = atomList.size();

      DblNumVec  atomPos(3*numAtom);
      DblNumVec  atomVel(3*numAtom);
      DblNumVec  atomForce(3*numAtom);

      DblNumVec  atomPosOld(3*numAtom);
      DblNumVec  atomVelOld(3*numAtom);
      DblNumVec  atomForceOld(3*numAtom);

      // Copy the position and force from the single-shot 
      // calculation done outside geometry optimization loop:

      // Force:
      for( Int a = 0; a < numAtom; a++ ){
        // Position
        atomPos[3*a]   = atomList[a].pos[0];
        atomPos[3*a+1] = atomList[a].pos[1];
        atomPos[3*a+2] = atomList[a].pos[2];

        // Force:
        atomForce[3*a]   = atomList[a].force[0];
        atomForce[3*a+1] = atomList[a].force[1];
        atomForce[3*a+2] = atomList[a].force[2];

        // Reset velocity to zero for first FIRE velocity Verlet step
        atomList[a].vel[0] = 0.0;
        atomList[a].vel[1] = 0.0;
        atomList[a].vel[2] = 0.0;

      }

      for( Int a = 0; a < numAtom; a++ ){
        // Velocity:
        atomVel[3*a]   = atomList[a].vel[0];
        atomVel[3*a+1] = atomList[a].vel[1];
        atomVel[3*a+2] = atomList[a].vel[2];
      }

      // This is useful only for the first step
      atomPosOld   = atomPos;
      atomVelOld   = atomVel;
      atomForceOld = atomForce;

      // Setting up internal variables of FIRE with user-defined parameters 
      // as well as the default ones. This is done thru the setup method 
      // of the FIRE_vars object which is an instance of the 
      // class FIRE_internal_vars_type, defined in iondynamics.hpp

      FIRE_vars.setup(nMin, dt, mass, fInc, fDec, alphaStart, fAlpha, alpha, 
          cut, dtMax, atomList, atomPos, atomVel, atomForce, 
          atomPosOld, atomVelOld, atomForceOld);

      statusOFS << " Done ." << std::endl;

      statusOFS << std::endl << " <---- FIRE based optimization parameters ---->";
      statusOFS << std::endl << "N_min = " << nMin << ", dt = " << dt << ", Mass = " << mass << std::endl;
      statusOFS << std::endl << "f_inc = " << fInc << ", f_dec = " << fDec << ", alpha_start = " << alphaStart 
        << ", f_alpha = " << fAlpha << ", dt_max = 10*dt = " << dtMax << std::endl;
    }	// if( fire )


    // Molecular dynamics
    Ekinetic_ = 0.0;
    Epot_    = 0.0;
    EconserveInit_ = 0.0;
    Econserve_ = 0.0;
    Edrift_   = 0.0;

    if( ionMove_ == "verlet" || ionMove_ == "langevin" ){
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
      else{
        // Random velocity given by ion temperature
        if( mpirank == 0 ){
          Point3 xi;
          for(Int a=0; a<numAtom; a++) {
            xi[0] = GaussianRandom() * std::sqrt(ionTemperature_/atomMass_[a]);
            xi[1] = GaussianRandom() * std::sqrt(ionTemperature_/atomMass_[a]);
            xi[2] = GaussianRandom() * std::sqrt(ionTemperature_/atomMass_[a]);
            atomList[a].vel = xi;
          }
        }
        for(Int a = 0; a < numAtom; a++){
          MPI_Bcast( &atomList[a].vel[0], 3, MPI_DOUBLE, 0, MPI_COMM_WORLD ); 
        }
      }
    }

    if( ionMove_ == "nosehoover1" ){
      xi1_ = 0.0;
      vxi1_ = 0.0;
      G1_ = 0.0;
      scalefac_ = 0.0;

      if(esdfParam.isRestartVelocity){
        statusOFS << std::endl 
          << "Read velocity and thermostat information from lastVel.out. " << std::endl;

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
          fin >> vxi1_;
          fin >> xi1_;

          fin.close();
        }
        // Broadcast thermostat information
        MPI_Bcast( atomvelRead.Data(), 3*numAtom, MPI_DOUBLE, 0, MPI_COMM_WORLD );
        MPI_Bcast( &vxi1_, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD );
        MPI_Bcast( &xi1_, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD ); 

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

        Print( statusOFS, "vxi1         = ", vxi1_ );
        Print( statusOFS, "xi1          = ", xi1_ );

      }//restart read in last velocities of atoms
      else{
        // Random velocity given by ion temperature
        if( mpirank == 0 ){
          Point3 xi;
          for(Int a=0; a<numAtom; a++) {
            xi[0] = GaussianRandom() * std::sqrt(ionTemperature_/atomMass_[a]);
            xi[1] = GaussianRandom() * std::sqrt(ionTemperature_/atomMass_[a]);
            xi[2] = GaussianRandom() * std::sqrt(ionTemperature_/atomMass_[a]);
            atomList[a].vel = xi;
          }
        }
        xi1_ = 0.0;
        vxi1_ = 0.0;
        for(Int a = 0; a < numAtom; a++){
          MPI_Bcast( &atomList[a].vel[0], 3, MPI_DOUBLE, 0, MPI_COMM_WORLD ); 
        }
      }

    } // nosehoover 1


    // Print out the force
    //  PrintBlock( statusOFS, "Atomic Force" );
    //
    //  Int numAtom = atomListPtr_->size();
    //
    //  for( Int a = 0; a < numAtom; a++ ){
    //    Print( statusOFS, "atom", a, "force", (*atomListPtr_)[a].force );
    //  }
    //  statusOFS << std::endl;


    return ;
  }         // -----  end of method IonDynamics::Setup  ----- 


void
  IonDynamics::MoveIons    ( Int ionIter )
  {
    Int mpirank, mpisize;
    MPI_Comm_rank( MPI_COMM_WORLD, &mpirank );
    MPI_Comm_size( MPI_COMM_WORLD, &mpisize );

    std::vector<Atom>&   atomList = *atomListPtr_;
    Int numAtom = atomList.size();

    // Update saved atomList. 0 is the latest one
    for( Int l = maxHist_-1; l > 0; l-- ){
      atomListHist_[l] = atomListHist_[l-1];
    }
    atomListHist_[0] = atomList;

    // *********************************************************************
    // Geometry optimization methods
    // *********************************************************************

    // Save the position at the infimum configuration
    // This is done here before the ions move since the forces correspond to the unmoved ions
    if ( mpirank == 0 ){
      if ( isGeoOpt_ == 1 ){
        Real fAtThisIter = MaxForce( atomList );

        if ( fAtThisIter < fAtInfimum_ ){
          // Update fAtInfimum
          fAtInfimum_ = fAtThisIter;

          std::fstream fout;
          fout.open("infimumPos.out",std::ios::out);
          if ( !fout.good() ){
            ErrorHandling( "File cannot be opened !" );
          }    
          for(Int i=0; i<numAtom; i++){
            fout << std::setiosflags(std::ios::scientific)
              << std::setiosflags(std::ios::showpos)
              << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< atomList[i].pos[0]
              << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< atomList[i].pos[1]
              << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< atomList[i].pos[2]
              << std::resetiosflags(std::ios::scientific)
              << std::resetiosflags(std::ios::showpos)
              << std::endl;
          }

          fout.close();

          // Output the XYZ format for movie
          fout.open("InfimumMD.xyz",std::ios::out | std::ios::app) ;
          if ( !fout.good() ){
            ErrorHandling( "Cannot open MD.xyz!" );
          }    
          fout << numAtom << std::endl;
          fout << "MD step # "<< ionIter << std::endl;
          for(Int a=0; a<numAtom; a++){
            fout<< std::setw(6)<< atomList[a].type
              << std::setw(16)<< atomList[a].pos[0]*au2ang
              << std::setw(16)<< atomList[a].pos[1]*au2ang
              << std::setw(16)<< atomList[a].pos[2]*au2ang
              << std::endl;
          }

          fout.close();

        } // if ( fAtInfimum < MaxForce( atomList ))
      } // if ( isGeoOpt_ == 1 )
    } // if( mpirank == 0 )   


    if( ionMove_ == "bb" ){
      BarzilaiBorweinOpt( ionIter );
    }

    if( ionMove_ == "pgbb" ){
      PGBBOpt( ionIter );
    }

    if( ionMove_ == "lbfgs" ){
      LBFGSOpt( ionIter );
    }

    if( ionMove_ == "nlcg"){
      NLCG_Opt( ionIter );      
    }

    if( ionMove_ == "fire"){
      FIREOpt( ionIter ); 
    }

    // *********************************************************************
    // Molecular dynamics methods
    // *********************************************************************
    if( ionMove_ == "verlet" ){
      VelocityVerlet( ionIter );
    }

    if( ionMove_ == "nosehoover1" ){
      NoseHoover1( ionIter );
    }

    if( ionMove_ == "langevin" ){
      Langevin( ionIter );
    }


    // Make sure no atom is outside supercell after ionic moves
    // Should be done both for geo opt and MD actually
    if ((isGeoOpt_ == 1)) 
    {
      for (int ii = 0; ii < numAtom; ii ++)
      {
        // X coordinate adjustment
        while ( atomList[ii].pos[0] > supercell_x_ )
          atomList[ii].pos[0] -= supercell_x_ ;

        while ( atomList[ii].pos[0] < 0.0 )
          atomList[ii].pos[0] += supercell_x_ ;

        // Y coordinate adjustment	
        while ( atomList[ii].pos[1] > supercell_y_ )
          atomList[ii].pos[1] -= supercell_y_ ;

        while ( atomList[ii].pos[1] < 0.0 )
          atomList[ii].pos[1] += supercell_y_ ;

        // Z coordinate adjustment
        while ( atomList[ii].pos[2] > supercell_z_ )
          atomList[ii].pos[2] -= supercell_z_ ;

        while ( atomList[ii].pos[2] < 0.0 )
          atomList[ii].pos[2] += supercell_z_ ;		
      }
    }    

    // Output the new coordinates
    {
      Print(statusOFS, ""); 
      Print(statusOFS, "Atom Type and Coordinates");
      Print(statusOFS, ""); 
      for(Int i=0; i < atomList.size(); i++) {
        statusOFS << std::setiosflags(std::ios::left) 
          << std::setw(LENGTH_VAR_NAME) << "Type = "
          << std::setw(6) << atomList[i].type
          << std::setiosflags(std::ios::scientific)
          << std::setiosflags(std::ios::showpos)
          << std::setw(LENGTH_VAR_NAME) << "Pos  = "
          << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC) << atomList[i].pos[0]
          << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC) << atomList[i].pos[1]
          << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC) << atomList[i].pos[2]
          << std::setw(LENGTH_VAR_NAME) << "Vel  = "
          << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC) << atomList[i].vel[0]
          << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC) << atomList[i].vel[1]
          << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC) << atomList[i].vel[2]
          << std::resetiosflags(std::ios::scientific)
          << std::resetiosflags(std::ios::showpos)
          << std::endl;
      }
    }

    // Output the position. Common to all routines
    if( mpirank == 0 ){
      if(esdfParam.isOutputPosition){
        std::fstream fout;
        fout.open("lastPos.out",std::ios::out);
        if( !fout.good() ){
          ErrorHandling( "File cannot be opened !" );
        }
        for(Int i=0; i<numAtom; i++){
          fout << std::setiosflags(std::ios::scientific)
            << std::setiosflags(std::ios::showpos)
            << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< atomList[i].pos[0]
            << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< atomList[i].pos[1]
            << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< atomList[i].pos[2]
            << std::resetiosflags(std::ios::scientific)
            << std::resetiosflags(std::ios::showpos)
            << std::endl;
        }
        fout.close();
      }
    }


    // Output the XYZ format for movie
    if( mpirank == 0 ){
      if( esdfParam.isOutputXYZ ){
        std::fstream fout;
        fout.open("MD.xyz",std::ios::out | std::ios::app) ;
        if( !fout.good() ){
          ErrorHandling( "Cannot open MD.xyz!" );
        }
        fout << numAtom << std::endl;
        fout << "MD step # "<< ionIter << std::endl;
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
      }
    } // if( mpirank == 0 )


    return ;
  }         // -----  end of method IonDynamics::MoveIons  ----- 


void IonDynamics::BarzilaiBorweinOpt    ( Int ionIter )
{
  Int mpirank, mpisize;
  MPI_Comm_rank( MPI_COMM_WORLD, &mpirank );
  MPI_Comm_size( MPI_COMM_WORLD, &mpisize );

  std::vector<Atom>&   atomList    = *atomListPtr_;
  // Note: atomListHist_[0] stores the same info as atomList
  std::vector<Atom>&   atomListOld = atomListHist_[1];

  Int numAtom = atomList.size();

  std::vector<Point3>  atompos(numAtom);
  std::vector<Point3>  atomforce(numAtom);
  std::vector<Point3>  atomposOld(numAtom);
  std::vector<Point3>  atomforceOld(numAtom);

  for( Int a = 0; a < numAtom; a++ ){
    atompos[a]   = atomList[a].pos;
    atomforce[a] = atomList[a].force;
  }

  //     // Output the XYZ format for movie
  //     // Once this is written, all work associated with the current atomic
  //     // position is DONE.
  //     if( mpirank == 0 ){
  //       if( isOutputXYZ_ ){
  // 	std::fstream fout;
  // 	fout.open("MD.xyz",std::ios::out | std::ios::app) ;
  // 	if( !fout.good() ){
  // 	  ErrorHandling( "Cannot open MD.xyz!" );
  // 	}
  // 	fout << numAtom << std::endl;
  // 	fout << "MD step # "<< ionIter << std::endl;
  // 	for(Int a=0; a<numAtom; a++){
  // 	  fout<< std::setw(6)<< atomList[a].type
  // 	      << std::setw(16)<< atompos[a][0]*au2ang
  // 	      << std::setw(16)<< atompos[a][1]*au2ang
  // 	      << std::setw(16)<< atompos[a][2]*au2ang
  // 	      << std::endl;
  // 	}
  // 	fout.close();
  //       }
  //     } // if( mpirank == 0 )


  if( ionIter == 1 ){
    // FIXME 0.1 is a magic number
    for( Int a = 0; a < numAtom; a++ ){
      atompos[a]   = atompos[a] + 0.1 * atomforce[a];
    }
  }
  else{

    for( Int a = 0; a < numAtom; a++ ){
      atomposOld[a]   = atomListOld[a].pos;
      atomforceOld[a] = atomListOld[a].force;
    }

    DblNumVec sVec(DIM*numAtom), yVec(DIM*numAtom);
    SetValue( sVec, 0.0 );
    SetValue( yVec, 0.0 );

    for( Int a = 0; a < numAtom; a++ ){
      for( Int d = 0; d < DIM; d++ ){
        sVec(DIM*a+d) = atompos[a][d] - atomposOld[a][d];
        yVec(DIM*a+d) = atomforce[a][d] - atomforceOld[a][d];
      }
    }
    // Note the minus sign
    Real step = - blas::Dot( DIM*numAtom, sVec.Data(), 1, yVec.Data(), 1 ) / 
      blas::Dot( DIM*numAtom, yVec.Data(), 1, yVec.Data(), 1 );

    for( Int a = 0; a < numAtom; a++ ){
      // Update the atomic position
      atompos[a]   = atompos[a] + step * atomforce[a];
    }
  }

  // Update atomic position to store in atomListPtr_
  for(Int a = 0; a < numAtom; a++){
    atomList[a].pos = atompos[a];
  }


  return ;
}         // -----  end of method IonDynamics::BarzilaiBorweinOpt  ----- 

void IonDynamics::PGBBOpt ( Int ionIter )
{
  Int mpirank, mpisize;
  MPI_Comm_rank( MPI_COMM_WORLD, &mpirank );
  MPI_Comm_size( MPI_COMM_WORLD, &mpisize );

  std::vector<Atom>&   atomList    = *atomListPtr_;

  Int numAtom = atomList.size();

  // Update the current position and force
  for( Int a = 0; a < numAtom; a++ ){
    geoOptVars_.atompos[a]   = atomList[a].pos;
    geoOptVars_.atomforce[a] = atomList[a].force;
  }

  // Should evaluate here

  if( ionIter > 1 ){

    DblNumVec sVec(DIM*numAtom), yVec(DIM*numAtom);
    SetValue( sVec, 0.0 );
    SetValue( yVec, 0.0 );

    for( Int a = 0; a < numAtom; a++ ){
      for( Int d = 0; d < DIM; d++ ){
        sVec(DIM*a+d) = geoOptVars_.atompos[a][d] - geoOptVars_.atomposOld[a][d];
        yVec(DIM*a+d) = geoOptVars_.atomforce[a][d] - geoOptVars_.atomforceOld[a][d];
      }
    }

    Real sy = blas::Dot( DIM*numAtom, sVec.Data(), 1, yVec.Data(), 1 );
    Real ss = blas::Dot( DIM*numAtom, sVec.Data(), 1, sVec.Data(), 1 );
    Real yy = blas::Dot( DIM*numAtom, yVec.Data(), 1, yVec.Data(), 1 );

    if( ionIter % 2 == 0 )
      geoOptVars_.tau = ss / std::abs(sy);
    else
      geoOptVars_.tau = std::abs(sy) / yy;

    statusOFS << "PGBB tau prev = " << geoOptVars_.tau << std::endl;

    geoOptVars_.tau = std::max( std::min( geoOptVars_.tau, 1e10 ), 1e-10 );
  }

  statusOFS << "PGBB tau = " << geoOptVars_.tau << std::endl;

  // Regular step. Update the history information
  if( geoOptVars_.callType == 0 ){
    for( Int a = 0; a < numAtom; a++ ){
      geoOptVars_.atomposOld[a]   = geoOptVars_.atompos[a];
      geoOptVars_.atomforceOld[a] = geoOptVars_.atomforce[a];
    }
    geoOptVars_.nls = 1;
  }

  statusOFS << "Computed tau = " << geoOptVars_.tau << std::endl;

  // Update tau if necessary
  Real posdiffMax = 0.0, posdiff = 0.0;

  for( Int a = 0; a < numAtom; a++ ){
    posdiff = geoOptVars_.tau * geoOptVars_.atomforceOld[a].l2();
    if( posdiffMax < posdiff )
      posdiffMax = posdiff;
  }

  // magic number
  Real possdiffBound = 0.1;
  geoOptVars_.tau *= possdiffBound/std::max(posdiffMax, possdiffBound);

  statusOFS << "Corrected tau = " << geoOptVars_.tau << std::endl;

  // Update atomic position to store in atomListPtr_
  for( Int a = 0; a < numAtom; a++ ){
    atomList[a].pos   = geoOptVars_.atomposOld[a] + 
      geoOptVars_.tau * geoOptVars_.atomforceOld[a];
  }


  return ;
}         // -----  end of method IonDynamics::PGBBOpt  ----- 


void IonDynamics::LBFGSOpt ( Int ionIter )
{
  Int mpirank, mpisize;
  MPI_Comm_rank( MPI_COMM_WORLD, &mpirank );
  MPI_Comm_size( MPI_COMM_WORLD, &mpisize );

  std::vector<Atom>&   atomList    = *atomListPtr_;

  Int numAtom = atomList.size();

  Int N = 3*numAtom;

  DblNumVec pos(N);
  DblNumVec grad(N);
  Int diagco = 0;
  DblNumVec diagdummy(N);
  //  SetValue( diagdummy, 0.01 );
  IntNumVec iPrint(2);
  Real MACH_EPS = 1e-15;

  iPrint[0] = 1;  // output every IPRINT(0) iterations.
  iPrint[1] = 0;  // iteration count, number of function evaluations, 
  // function value, norm of the gradient, and steplength,

  for( Int a = 0; a < numAtom; a++ ){
    pos[3*a]     = atomList[a].pos[0];
    pos[3*a+1]   = atomList[a].pos[1];
    pos[3*a+2]   = atomList[a].pos[2];
    grad[3*a]    = -atomList[a].force[0];
    grad[3*a+1]  = -atomList[a].force[1];
    grad[3*a+2]  = -atomList[a].force[2];
  }

  statusOFS << "geoOptVars_.work.size() = " << geoOptVars_.work.Size() << std::endl;
  statusOFS << "Epot = " << Epot_ << std::endl;

  if( mpirank == 0 ){
    F2C(lbfgs)( &N, &geoOptVars_.maxMixingDim, pos.Data(),
        &Epot_, grad.Data(), &diagco, diagdummy.Data(), 
        iPrint.Data(), &geoOptVars_.gtol, &MACH_EPS, 
        geoOptVars_.work.Data(), &geoOptVars_.callType );
  }
  MPI_Bcast( &pos[0], N, MPI_DOUBLE, 0, MPI_COMM_WORLD ); 

  statusOFS << "callType = " << geoOptVars_.callType << std::endl;

  for( Int a = 0; a < numAtom; a++ ){
    atomList[a].pos[0] = pos[3*a];
    atomList[a].pos[1] = pos[3*a+1];
    atomList[a].pos[2] = pos[3*a+2];
  }

  return ;
}         // -----  end of method IonDynamics::LBFGSOpt  ----- 



// Non-Linear Conjugate Gradients with Secant and Polak-Ribiere
// Page 53 of the online.pdf : "An introduction to the conjugate gradient method without the agonizing pain." -- Jonathan Richard Shewchuk (1994).
// Implemented by Amartya S. Banerjee, May 2016
void
IonDynamics::NLCG_Opt ( Int ionIter)
{

  std::vector<Atom>&   atomList    = *atomListPtr_;

  statusOFS << std::endl << " Inside NLCG stepper routine : Call type = " << NLCG_vars.call_type << std::endl;

  if(NLCG_vars.call_type == NLCG_CALL_TYPE_1)
  { 

    if( NLCG_vars.delta_new_ <= (NLCG_vars.epsilon_tol_outer_ * NLCG_vars.epsilon_tol_outer_ * NLCG_vars.delta_0_))
    {
      // This probably happened because the exit conditions inside here are slightly different from what the caller routines expect.
      // Adjust the exit tolerances 
      statusOFS << std::endl << " Could be stuck in Call type 1 of NLCG !!!" << std::endl; 
      statusOFS << std::endl << " Call type 1 : Will lower NLCG_vars.epsilon_tol_outer_ to ensure exit condition in calling routine is satisfied !! ";

      NLCG_vars.epsilon_tol_outer_ = NLCG_vars.epsilon_tol_outer_ * 0.001;

      statusOFS << std::endl << "Resuming NLCG with Call type = " << NLCG_vars.call_type << std::endl;
    }

    if( ((NLCG_vars.i_ <= NLCG_vars.i_max_) && 
          (NLCG_vars.delta_new_ > (NLCG_vars.epsilon_tol_outer_ * NLCG_vars.epsilon_tol_outer_ * NLCG_vars.delta_0_))) )
    {
      // j = 0
      NLCG_vars.j_ = 0;

      // delta_d = d^T d
      NLCG_vars.delta_d_ = NLCG_vars.atom_ddot(NLCG_vars.atomforce_d_, NLCG_vars.atomforce_d_);

      statusOFS << std::endl << " atomforce_d_ = " << NLCG_vars.atomforce_d_ << std::endl ;

      // alpha = - sigma_0 
      NLCG_vars.alpha_ = -NLCG_vars.sigma_0_;

      statusOFS << std::endl << " sigma_0 = " << NLCG_vars.sigma_0_ << std::endl ;

      // Use x and d to compute new position for force evalation
      // pos = x + sigma_0 * d
      for( Int a = 0; a < NLCG_vars.numAtom; a++ )
      {
        for( Int d = 0; d < DIM; d++ )
        {
          atomList[a].pos[d] = (NLCG_vars.atompos_x_[a][d] + NLCG_vars.sigma_0_ * NLCG_vars.atomforce_d_[a][d]);      
        }
      }


      NLCG_vars.call_type = NLCG_CALL_TYPE_2;

      // Go back to do new evaluations of energy and forces after changing call type.
      statusOFS << std::endl << " Calling back for SCF energy / force evaluation : Call type = " << NLCG_vars.call_type << std::endl;


    } // end of if NLCG_vars.i_ <= NLCG_vars.i_max_ etc..
    else
    {
      // This situation should not arise at all
      statusOFS << std::endl << std::endl << " Reached unexpected portion of NLCG routine with Call type = 1. Will Abort now ! " << std::endl << std::endl;
      exit(1);

    }



  } // end of call_type 1    
  else if(NLCG_vars.call_type == NLCG_CALL_TYPE_2)
  {

    // Compute eta_prev = [f'(x + sigma_0 * d)]^T d
    std::vector<Point3>  atomforce_temp;
    atomforce_temp.resize(NLCG_vars.numAtom);
    for( Int a = 0; a < NLCG_vars.numAtom; a++ )
      atomforce_temp[a] = atomList[a].force;

    NLCG_vars.eta_prev_ = - NLCG_vars.atom_ddot(atomforce_temp, NLCG_vars.atomforce_d_); // Note the negative sign

    // Use x to update position for force evaluation
    for( Int a = 0; a < NLCG_vars.numAtom; a++ )
      atomList[a].pos = NLCG_vars.atompos_x_[a];


    NLCG_vars.call_type = NLCG_CALL_TYPE_3;

    // Go back to do new evaluations of energy and forces after changing call type.
    statusOFS << std::endl << " Calling back for SCF energy / force evaluation : Call type = " << NLCG_vars.call_type << std::endl;

  } // end of call_type 2    
  else if(NLCG_vars.call_type == NLCG_CALL_TYPE_3)
  {

    // Compute eta = [f'(x)]^T d
    std::vector<Point3>  atomforce_temp;
    atomforce_temp.resize(NLCG_vars.numAtom);
    for( Int a = 0; a < NLCG_vars.numAtom; a++ )
      atomforce_temp[a] = atomList[a].force;

    NLCG_vars.eta_ = - NLCG_vars.atom_ddot(atomforce_temp, NLCG_vars.atomforce_d_); // Note the negative sign

    // alpha = alpha * (eta/(eta_prev-eta))
    NLCG_vars.alpha_ =  NLCG_vars.alpha_ * (NLCG_vars.eta_ / (NLCG_vars.eta_prev_ - NLCG_vars.eta_));

    // Set x = x + alpha * d
    for( Int a = 0; a < NLCG_vars.numAtom; a++ )
    {
      for( Int d = 0; d < DIM; d++ )
      {
        NLCG_vars.atompos_x_[a][d]  = (NLCG_vars.atompos_x_[a][d] + NLCG_vars.alpha_ * NLCG_vars.atomforce_d_[a][d]);      
      }
    }

    // Set eta_prev = eta
    NLCG_vars.eta_prev_ = NLCG_vars.eta_;

    // Increment inner counter
    NLCG_vars.j_ = NLCG_vars.j_ + 1;

    // Exit conditions
    if((NLCG_vars.j_ < NLCG_vars.j_max_) && 
        ((NLCG_vars.alpha_ * NLCG_vars.alpha_ * NLCG_vars.delta_d_) > (NLCG_vars.epsilon_tol_inner_ * NLCG_vars.epsilon_tol_inner_)))

      //if((NLCG_vars.j_ < NLCG_vars.j_max_) && 
      //    ((NLCG_vars.alpha_ * NLCG_vars.alpha_ * NLCG_vars.delta_d_) > (NLCG_vars.epsilon_tol_inner_)))
    {
      // Stay for looping : do not change call type
      // Use x to update position for force evaluation
      for( Int a = 0; a < NLCG_vars.numAtom; a++ )
        atomList[a].pos = NLCG_vars.atompos_x_[a];  
    }
    else
    {
      // Exit to pass control to remaining portion of code
      NLCG_vars.call_type = NLCG_CALL_TYPE_4;

      // Use x to update position for force evaluation
      for( Int a = 0; a < NLCG_vars.numAtom; a++ )
        atomList[a].pos = NLCG_vars.atompos_x_[a];  

    }

    statusOFS << std::endl << " Calling back for SCF energy / force evaluation : Call type = " << NLCG_vars.call_type << std::endl;


  } // end of call_type 3    
  else
  {

    // Set r = - f'(x) : Note that  atomList[a].force = - grad E already - so no extra negative required
    for( Int a = 0; a < NLCG_vars.numAtom; a++ )
      NLCG_vars.atomforce_r_[a] = atomList[a].force;

    // Set delta_old = delta_new
    NLCG_vars.delta_old_ = NLCG_vars.delta_new_ ;

    // Set delta_mid = r^T s  
    NLCG_vars.delta_mid_ =  NLCG_vars.atom_ddot( NLCG_vars.atomforce_r_, NLCG_vars.atomforce_s_);

    // Set s = M^{-1} r : M = Identity used here
    for( Int a = 0; a <  NLCG_vars.numAtom; a++ )
      NLCG_vars.atomforce_s_[a] =  NLCG_vars.atomforce_r_[a];

    // Set delta_new = r^T s
    NLCG_vars.delta_new_ =  NLCG_vars.atom_ddot( NLCG_vars.atomforce_r_, NLCG_vars.atomforce_s_);

    // Set beta = (delta_new - delta_mid) / delta_old
    NLCG_vars.beta_ = (NLCG_vars.delta_new_ - NLCG_vars.delta_mid_) / NLCG_vars.delta_old_;

    // Increment counter
    NLCG_vars.k_ = NLCG_vars.k_ + 1;

    if((NLCG_vars.k_ == NLCG_vars.n_) || (NLCG_vars.beta_ <= 0.0))
    {
      // Set d = s
      for( Int a = 0; a < NLCG_vars.numAtom; a++ )
        NLCG_vars.atomforce_d_[a] = NLCG_vars.atomforce_s_[a];

      // set k = 0
      NLCG_vars.k_ = 0;

      statusOFS << std::endl << " Restarting NLCG (k = n or beta < 0 satisfied) !!" << std::endl;

    }
    else
    {
      // Set d = s + beta * d
      for( Int a = 0; a < NLCG_vars.numAtom; a++ )
      {
        for( Int d = 0; d < DIM; d++ )
        {
          NLCG_vars.atomforce_d_[a][d]  = (NLCG_vars.atomforce_s_[a][d] + NLCG_vars.beta_ * NLCG_vars.atomforce_d_[a][d]);      
        }
      }
    }

    // Increment outer counter
    NLCG_vars.i_ = NLCG_vars.i_ + 1;

    NLCG_vars.call_type = NLCG_CALL_TYPE_1;

    // Change the call type to run the outer loop again.
    statusOFS << std::endl << " Calling back : Call type = " << NLCG_vars.call_type << std::endl;

  } // end of call_type 4    



  return;
}   // -----  end of method IonDynamics::NLCG_Opt  -----  


// Routine implementing the Fast Inertial Relaxation Engine Optimizer  
void
IonDynamics::FIREOpt( Int ionIter )
{
  Int mpirank, mpisize;
  MPI_Comm_rank( MPI_COMM_WORLD, &mpirank );
  MPI_Comm_size( MPI_COMM_WORLD, &mpisize );

  // IMPORTANT: iondynamics should only be performed by one
  //            processor, and then the atomic position and velocity are
  //            broadcast to other processors. This is particularly important
  //            for stochastic methods

  std::vector<Atom>& atomList = *atomListPtr_;

  Int numAtom = atomList.size();

  if( mpirank == 0 ){

    if( ionIter > 1 ){

      // Fetch the latest atomPos_, atomVel_, and atomForce_ from scf.Update()
      // via atomListPtr_

      for( Int a = 0; a < numAtom; a++ ){
        // Note the atomForce_[iter] is calculated from 
        // atomPos_[iter-1] (=atomPosOld_[iter - 1]) 
        // which is calculated in step # [iter - 1]
        FIRE_vars.atomForce_[3*a]   = atomList[a].force[0];
        FIRE_vars.atomForce_[3*a+1] = atomList[a].force[1];
        FIRE_vars.atomForce_[3*a+2] = atomList[a].force[2];
      }

      // FIRE_vars.atomVel_ gets copies to FIRE_vars.atomVelOld_
      FIRE_vars.DblNumVecCopier(FIRE_vars.atomVel_, FIRE_vars.atomVelOld_);

      // Update velocity as per velocity Verlet (NOTE: this is the 
      // last step of velocity Verlet update initiated in step # [iter - 1]) 
      statusOFS << std::endl << " Updating velocities using velocity Verlet . . . " << std::endl;

      for( Int a = 0; a < numAtom; a++ ){
        FIRE_vars.atomVel_[3*a]   = FIRE_vars.atomVelOld_[3*a] + 0.5*(FIRE_vars.atomForceOld_[3*a]
            + FIRE_vars.atomForce_[3*a])*FIRE_vars.dt_/FIRE_vars.mass_;

        FIRE_vars.atomVel_[3*a+1] = FIRE_vars.atomVelOld_[3*a+1] + 0.5*(FIRE_vars.atomForceOld_[3*a+1]
            + FIRE_vars.atomForce_[3*a+1])*FIRE_vars.dt_/FIRE_vars.mass_;

        FIRE_vars.atomVel_[3*a+2] = FIRE_vars.atomVelOld_[3*a+2] + 0.5*(FIRE_vars.atomForceOld_[3*a+2]
            + FIRE_vars.atomForce_[3*a+2])*FIRE_vars.dt_/FIRE_vars.mass_;
      }

      // Update the latest atomVel_, dt_, and cut_ using FIRE update algorithm

      statusOFS << std::endl << " Modifying velocities output from Verlet and dt using FIRE . . . " << std::endl;

      FIRE_vars.FIREStepper( ionIter );


      // Limit FIRE-vars.dt_ so no charges change more than dMax (dMax is arbitrary)
      double dMax = 0.1;	// hard-coded MAGIC number
      double vMax = 0.0;

      // Find the maximum component
      for( Int j = 0; j < 3*numAtom; j++ )
        vMax = std::max(fabs(FIRE_vars.atomVel_[j]), vMax);

      // Reset dt here
      if (FIRE_vars.dt_* vMax > dMax){
        statusOFS << std::endl << " Warning! Possible large step in x avoided by readjusting dt. . . " << std::endl;

        FIRE_vars.dt_ = dMax/vMax;		
      }



      /*

      // Update atomic velocity to store in atomListPtr_.
      // This also gets used in velocity Verlet
      for( Int a = 0; a < numAtom; a++ ){
      atomList[a].vel[0] = atomVel_[3*a];
      atomList[a].vel[1] = atomVel_[3*a+1];
      atomList[a].vel[2] = atomVel_[3*a+2];
      }

       */

      // Having modified the atomVel_ and dt_ (from FIREStepper), update atomPos; 

      // Store atomPos_ as atomPosold_ first
      FIRE_vars.DblNumVecCopier(FIRE_vars.atomPos_, FIRE_vars.atomPosOld_);

      statusOFS << std::endl << " Updating positions using velocity Verlet . . . " << std::endl;

      /*

         for ( Int  a = 0; a < numAtom; a++ ){
         FIRE_vars.atomPos_[3*a]   = FIRE_vars.atomPosOld_[3*a] + FIRE_vars.atomVelOld_[3*a]*FIRE_vars.dt_
         + 0.5*FIRE_vars.atomForceOld_[3*a]*FIRE_vars.dt_*FIRE_vars.dt_/FIRE_vars.mass_;

         FIRE_vars.atomPos_[3*a+1] = FIRE_vars.atomPosOld_[3*a+1] + FIRE_vars.atomVelOld_[3*a+1]*FIRE_vars.dt_
         + 0.5*FIRE_vars.atomForceOld_[3*a+1]*FIRE_vars.dt_*FIRE_vars.dt_/FIRE_vars.mass_;

         FIRE_vars.atomPos_[3*a+2] = FIRE_vars.atomPosOld_[3*a+2] + FIRE_vars.atomVelOld_[3*a+2]*FIRE_vars.dt_
         + 0.5*FIRE_vars.atomForceOld_[3*a+2]*FIRE_vars.dt_*FIRE_vars.dt_/FIRE_vars.mass_;
         }

       */

      // x(t + dt) = x(t) + v(t) * dt + 1/2 * f(t)/m * dt^2
      for ( Int  a = 0; a < numAtom; a++ ){
        FIRE_vars.atomPos_[3*a]   = FIRE_vars.atomPosOld_[3*a] + FIRE_vars.atomVel_[3*a]*FIRE_vars.dt_
          + 0.5*FIRE_vars.atomForce_[3*a]*FIRE_vars.dt_*FIRE_vars.dt_/FIRE_vars.mass_;

        FIRE_vars.atomPos_[3*a+1] = FIRE_vars.atomPosOld_[3*a+1] + FIRE_vars.atomVel_[3*a+1]*FIRE_vars.dt_
          + 0.5*FIRE_vars.atomForce_[3*a+1]*FIRE_vars.dt_*FIRE_vars.dt_/FIRE_vars.mass_;

        FIRE_vars.atomPos_[3*a+2] = FIRE_vars.atomPosOld_[3*a+2] + FIRE_vars.atomVel_[3*a+2]*FIRE_vars.dt_
          + 0.5*FIRE_vars.atomForce_[3*a+2]*FIRE_vars.dt_*FIRE_vars.dt_/FIRE_vars.mass_;
      }

      // Also store atomForce  as atomForceOld_ before moving on to next ionIter
      FIRE_vars.DblNumVecCopier(FIRE_vars.atomForce_, FIRE_vars.atomForceOld_);

    }      // if( ionIter > 1)

    else if( ionIter == 1 ){         // for FIRE, starting velocity = 0
      // Use the position and force values from single shot calculation done 
      // outside geometry optimization loop. Note these are already stored 
      // as atomPos_ and atomVel_ in FIRE_vars.setup(). 
      // Alongside, using the same method the initial velocity (stored in atomVel) 
      // is set to be 0.0 as required by FIRE.


      // Update atomPos using velocity Verlet
      statusOFS << std::endl << " Updating positions using velocity Verlet starting from zero initial velocities . . . " << std::endl;

      // Debug statement ****
      // statusOFS << std::endl << " ** Velocity before the POSITION update in Iteration # " << ionIter << std::endl;
      // FIRE_vars.DblNumVecPrinter( FIRE_vars.atomVel_ );

      // Store atomPos_ as atomPosold_ first
      FIRE_vars.DblNumVecCopier(FIRE_vars.atomPos_, FIRE_vars.atomPosOld_);

      /*

      // x(0 + dt) = x(0) + v(0) * dt + 1/2 * f(0)/m * dt^2
      for ( Int  a = 0; a < numAtom; a++ ){
      FIRE_vars.atomPos_[3*a]   = FIRE_vars.atomPosOld_[3*a] + FIRE_vars.atomVelOld_[3*a]*FIRE_vars.dt_
      + 0.5*FIRE_vars.atomForceOld_[3*a]*FIRE_vars.dt_*FIRE_vars.dt_/FIRE_vars.mass_;

      FIRE_vars.atomPos_[3*a+1] = FIRE_vars.atomPosOld_[3*a+1] + FIRE_vars.atomVelOld_[3*a+1]*FIRE_vars.dt_
      + 0.5*FIRE_vars.atomForceOld_[3*a+1]*FIRE_vars.dt_*FIRE_vars.dt_/FIRE_vars.mass_;

      FIRE_vars.atomPos_[3*a+2] = FIRE_vars.atomPosOld_[3*a+2] + FIRE_vars.atomVelOld_[3*a+2]*FIRE_vars.dt_
      + 0.5*FIRE_vars.atomForceOld_[3*a+2]*FIRE_vars.dt_*FIRE_vars.dt_/FIRE_vars.mass_;
      }

       */

      // x(0 + dt) = x(0) + v(0) * dt + 1/2 * f(0)/m * dt^2
      for ( Int  a = 0; a < numAtom; a++ ){
        FIRE_vars.atomPos_[3*a]   = FIRE_vars.atomPosOld_[3*a] + FIRE_vars.atomVel_[3*a]*FIRE_vars.dt_
          + 0.5*FIRE_vars.atomForce_[3*a]*FIRE_vars.dt_*FIRE_vars.dt_/FIRE_vars.mass_;

        FIRE_vars.atomPos_[3*a+1] = FIRE_vars.atomPosOld_[3*a+1] + FIRE_vars.atomVel_[3*a+1]*FIRE_vars.dt_
          + 0.5*FIRE_vars.atomForce_[3*a+1]*FIRE_vars.dt_*FIRE_vars.dt_/FIRE_vars.mass_;

        FIRE_vars.atomPos_[3*a+2] = FIRE_vars.atomPosOld_[3*a+2] + FIRE_vars.atomVel_[3*a+2]*FIRE_vars.dt_
          + 0.5*FIRE_vars.atomForce_[3*a+2]*FIRE_vars.dt_*FIRE_vars.dt_/FIRE_vars.mass_;
      }

      // Also store atomForce  as atomForceOld_ before moving on to next ionIter
      FIRE_vars.DblNumVecCopier(FIRE_vars.atomForce_, FIRE_vars.atomForceOld_);

      statusOFS << std::endl << " Velocity update with velocity Verlet can only be done in the next step . . .  " << std::endl;

    }     // if( ionIter == 1)
  }          // if( mpirank == 0 )

  // Broadcast Position and velocity
  MPI_Bcast( &FIRE_vars.atomPos_[0], 3*numAtom, MPI_DOUBLE, 0, MPI_COMM_WORLD );
  MPI_Bcast( &FIRE_vars.atomVel_[0], 3*numAtom, MPI_DOUBLE, 0, MPI_COMM_WORLD );

  // Update atomic position to store in atomListPtr_
  for( Int a = 0; a < numAtom; a++ ){
    atomList[a].pos[0]    = FIRE_vars.atomPos_[3*a];
    atomList[a].pos[1]    = FIRE_vars.atomPos_[3*a+1];
    atomList[a].pos[2]    = FIRE_vars.atomPos_[3*a+2];
    /*
       atomList[a].vel[0]    = FIRE_vars.atomVel_[3*a];
       atomList[a].vel[1]    = FIRE_vars.atomVel_[3*a+1];
       atomList[a].vel[2]    = FIRE_vars.atomVel_[3*a+2];

       atomList[a].force[0]  = FIRE_vars.atomForce_[3*a];
       atomList[a].force[1]  = FIRE_vars.atomForce_[3*a+1];
       atomList[a].force[2]  = FIRE_vars.atomForce_[3*a+2];
     */
  }

  return;            // exit to next SCF update

}


void
IonDynamics::VelocityVerlet    ( Int ionIter )
{
  Int mpirank, mpisize;
  MPI_Comm_rank( MPI_COMM_WORLD, &mpirank );
  MPI_Comm_size( MPI_COMM_WORLD, &mpisize );

  std::vector<Atom>&   atomList = *atomListPtr_;

  Int numAtom = atomList.size();

  std::vector<Point3>  atompos(numAtom);
  std::vector<Point3>  atomvel(numAtom);
  std::vector<Point3>  atomforce(numAtom);

  if( mpirank == 0 ){

    // some aliasing to be compatible with implementation before
    Real& dt = dt_;
    DblNumVec& atomMass = atomMass_;
    Real  K;

    for( Int a = 0; a < numAtom; a++ ){
      atompos[a]   = atomList[a].pos;
      atomvel[a]   = atomList[a].vel;
      atomforce[a] = atomList[a].force;
    }


    // Propagate velocity. This is the second part of Verlet step

    for( Int a = 0; a < numAtom; a++ ){
      atomvel[a] = atomvel[a] + atomforce[a]*dt*0.5/atomMass[a]; 
    }


    // Propagate the chain. This is due to the remaining update of the
    // chain variables.
    K=0.;
    for(Int a=0; a<numAtom; a++){
      for(Int j=0; j<3; j++){
        K += atomMass[a]*atomvel[a][j]*atomvel[a][j]/2.;
      }
    }

    // At this point, the position, velocity and thermostat variables are
    // synced at the same time step

    Ekinetic_  = K;
    Econserve_ = Ekinetic_ + Epot_;
    if(ionIter == 1)
      EconserveInit_ = Econserve_;
    Edrift_ = (Econserve_-EconserveInit_)/EconserveInit_;

    Print(statusOFS, "MD_Ekin    =  ", Ekinetic_);
    Print(statusOFS, "MD_Epot    =  ", Epot_);
    Print(statusOFS, "MD_Econ    =  ", Econserve_);
    Print(statusOFS, "MD_Edrift  =  ", Edrift_);

    // Output the XYZ format for movie
    // Once this is written, all work associated with the current atomic
    // position is DONE.
    if( esdfParam.isOutputXYZ ){
      std::fstream fout;
      fout.open("MD.xyz",std::ios::out | std::ios::app) ;
      if( !fout.good() ){
        ErrorHandling( "Cannot open MD.xyz!" );
      }
      fout << numAtom << std::endl;
      fout << "MD step # "<< ionIter << std::endl;
      for(Int a=0; a<numAtom; a++){
        fout<< std::setw(6)<< atomList[a].type
          << std::setw(16)<< atompos[a][0]*au2ang
          << std::setw(16)<< atompos[a][1]*au2ang
          << std::setw(16)<< atompos[a][2]*au2ang
          << std::endl;
      }
      fout.close();
    }

    // Update velocity and position
    for(Int a=0; a<numAtom; a++) {
      atomvel[a] = atomvel[a] + atomforce[a]*dt*0.5/atomMass[a]; 
      atompos[a] = atompos[a] + atomvel[a] * dt;
    }

    // Output the position and thermostat variable. 
    // These are the configuration that SCF will work on next. 
    // Hence if the job is stopped in the middle of SCF (which is most
    // likely), the MD job should continue from this configuration
    if(esdfParam.isOutputVelocity){
      std::fstream fout_v;
      fout_v.open("lastVel.out",std::ios::out);
      if( !fout_v.good() ){
        ErrorHandling( "File cannot be opened !" );
      }
      for(Int i=0; i<numAtom; i++){
        fout_v << std::setiosflags(std::ios::scientific)
          << std::setiosflags(std::ios::showpos)
          << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< atomvel[i][0]
          << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< atomvel[i][1]
          << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< atomvel[i][2]
          << std::resetiosflags(std::ios::scientific)
          << std::resetiosflags(std::ios::showpos)
          << std::endl;
      }
      fout_v.close();
    }

  } // if( mpirank == 0 )

  // Sync the atomic position and velocity
  for(Int a = 0; a < numAtom; a++){
    MPI_Bcast( &atompos[a][0], 3, MPI_DOUBLE, 0, MPI_COMM_WORLD ); 
    MPI_Bcast( &atomvel[a][0], 3, MPI_DOUBLE, 0, MPI_COMM_WORLD ); 
  }

  // Update atomic position and velocity to store in atomListPtr_
  // NOTE: Force is NOT consistent with the position yet.
  for(Int a = 0; a < numAtom; a++){
    atomList[a].pos = atompos[a];
    atomList[a].vel = atomvel[a];
  }


  return ;
}         // -----  end of method IonDynamics::VelocityVerlet  ----- 

void
IonDynamics::NoseHoover1    ( Int ionIter )
{
  Int mpirank, mpisize;
  MPI_Comm_rank( MPI_COMM_WORLD, &mpirank );
  MPI_Comm_size( MPI_COMM_WORLD, &mpisize );

  std::vector<Atom>&   atomList = *atomListPtr_;

  Int numAtom = atomList.size();


  std::vector<Point3>  atompos(numAtom);
  std::vector<Point3>  atomvel(numAtom);
  std::vector<Point3>  atomforce(numAtom);

  // some aliasing to be compatible with implementation before
  Real& vxi1 = vxi1_;
  Real& xi1  = xi1_;
  Real& s = scalefac_;
  Real& dt = dt_;
  Real& T  = ionTemperature_;
  Real& Q1 = Q1_;
  DblNumVec& atomMass = atomMass_;
  Real  L =  3.0 * numAtom;  // number of degrees of freedom
  Real  K;

  for( Int a = 0; a < numAtom; a++ ){
    atompos[a]   = atomList[a].pos;
    atomvel[a]   = atomList[a].vel;
    atomforce[a] = atomList[a].force;
  }


  // Propagate velocity. This is the second part of Verlet step

  for( Int a = 0; a < numAtom; a++ ){
    atomvel[a] = atomvel[a] + atomforce[a]*dt*0.5/atomMass[a]; 
  }

  // Propagate the chain. This is due to the remaining update of the
  // chain variables.
  K=0.;
  for(Int a=0; a<numAtom; a++){
    for(Int j=0; j<3; j++){
      K += atomMass[a]*atomvel[a][j]*atomvel[a][j]/2.;
    }
  }

  vxi1 = vxi1+(K*2.-L*T)/Q1*dt/4.;
  xi1  = xi1+vxi1*dt/2.;
  s    = std::exp(-vxi1*dt/2.);
  for(Int a=0;a<numAtom;a++){
    atomvel[a]=s*atomvel[a];
  }
  K=K*s*s;
  vxi1 = vxi1+(2*K-L*T)/Q1*dt/4.;

  // At this point, the position, velocity and thermostat variables are
  // synced at the same time step

  Ekinetic_  = K;
  Econserve_ = Ekinetic_ + Epot_ + Q1*vxi1*vxi1/2. + L*T*xi1;
  if(ionIter == 1)
    EconserveInit_ = Econserve_;
  Edrift_ = (Econserve_-EconserveInit_)/EconserveInit_;

  Print(statusOFS, "MD_Ekin    =  ", Ekinetic_);
  Print(statusOFS, "MD_Epot    =  ", Epot_);
  Print(statusOFS, "MD_Econ    =  ", Econserve_);
  Print(statusOFS, "MD_Edrift  =  ", Edrift_);

  //     // Output the XYZ format for movie
  //     // Once this is written, all work associated with the current atomic
  //     // position is DONE.
  //     if( mpirank == 0 ){
  //       if( isOutputXYZ_ ){
  //         std::fstream fout;
  //         fout.open("MD.xyz",std::ios::out | std::ios::app) ;
  //         if( !fout.good() ){
  //           ErrorHandling( "Cannot open MD.xyz!" );
  //         }
  //         fout << numAtom << std::endl;
  //         fout << "MD step # "<< ionIter << std::endl;
  //         for(Int a=0; a<numAtom; a++){
  //           fout<< std::setw(6)<< atomList[a].type
  // 	      << std::setw(16)<< atompos[a][0]*au2ang
  // 	      << std::setw(16)<< atompos[a][1]*au2ang
  // 	      << std::setw(16)<< atompos[a][2]*au2ang
  // 	      << std::endl;
  //         }
  //         fout.close();
  //       }
  //     } // if( mpirank == 0 )

  // Now prepare the variable for the next evalation of the force

  // Propagate the chain again
  K = Ekinetic_;
  vxi1 = vxi1+(K*2.-L*T)/Q1*dt/4.;
  xi1  = xi1+vxi1*dt/2.;
  s    = std::exp(-vxi1*dt/2.);
  for(Int a=0;a<numAtom;a++){
    atomvel[a]=s*atomvel[a];
  }
  vxi1 = vxi1+(2*K-L*T)/Q1*dt/4.;

  // Update velocity and position
  for(Int a=0; a<numAtom; a++) {
    atomvel[a] = atomvel[a] + atomforce[a]*dt*0.5/atomMass[a]; 
    atompos[a] = atompos[a] + atomvel[a] * dt;
  }

  // Output the velocity and thermostat variable. 
  // These are the configuration that SCF will work on next. 
  // Hence if the job is stopped in the middle of SCF (which is most
  // likely), the MD job should continue from this configuration
  if( mpirank == 0 ){

    if(esdfParam.isOutputVelocity){
      std::fstream fout_v;
      fout_v.open("lastVel.out",std::ios::out);
      if( !fout_v.good() ){
        ErrorHandling( "File cannot be open!" );
      }
      for(Int i=0; i<numAtom; i++){
        fout_v << std::setiosflags(std::ios::scientific)
          << std::setiosflags(std::ios::showpos)
          << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< atomvel[i][0]
          << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< atomvel[i][1]
          << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< atomvel[i][2]
          << std::resetiosflags(std::ios::scientific)
          << std::resetiosflags(std::ios::showpos)
          << std::endl;
      }

      fout_v << std::setiosflags(std::ios::scientific)
        << std::setiosflags(std::ios::showpos)
        << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< vxi1 << std::endl
        << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< xi1 << std::endl
        << std::resetiosflags(std::ios::scientific)
        << std::resetiosflags(std::ios::showpos);

      fout_v.close();


      // Also output into the statfile
      statusOFS << std::endl 
        << std::setiosflags(std::ios::left) 
        << std::setw(LENGTH_VAR_NAME) << "vxi1 = "
        << std::setiosflags(std::ios::scientific)
        << std::setiosflags(std::ios::showpos)
        << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< vxi1 << std::endl
        << std::setw(LENGTH_VAR_NAME) << "xi1 = "
        << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< xi1 << std::endl
        << std::resetiosflags(std::ios::scientific)
        << std::resetiosflags(std::ios::showpos);
    }
  } // if( mpirank == 0 )


  // Update atomic position and velocity to store in atomListPtr_
  // NOTE: Force is NOT consistent with the position yet.
  for(Int a = 0; a < numAtom; a++){
    atomList[a].pos = atompos[a];
    atomList[a].vel = atomvel[a];
  }


  return ;
}         // -----  end of method IonDynamics::NoseHoover1  ----- 

void
IonDynamics::Langevin ( Int ionIter )
{
  Int mpirank, mpisize;
  MPI_Comm_rank( MPI_COMM_WORLD, &mpirank );
  MPI_Comm_size( MPI_COMM_WORLD, &mpisize );
  // IMPORTANT: ion dynamics should only be performed by one
  // processor, and then the atomic position and velocity are
  // broadcast to other processors. This is particularly important
  // for stochastic methods

  std::vector<Atom>&   atomList = *atomListPtr_;
  Int numAtom = atomList.size();
  std::vector<Point3>  atompos(numAtom);
  std::vector<Point3>  atomvel(numAtom);
  std::vector<Point3>  atomforce(numAtom);

  if( mpirank == 0 ){
    // some aliasing to be compatible with implementation before
    Real& damping = langevinDamping_;
    Real& dt = dt_;
    Real& T  = ionTemperature_;
    DblNumVec& atomMass = atomMass_;
    Real  K;

    for( Int a = 0; a < numAtom; a++ ){
      atompos[a]   = atomList[a].pos;
      atomvel[a]   = atomList[a].vel;
      atomforce[a] = atomList[a].force;
    }

    // Propagate velocity. This is the second part of the update

    for( Int a = 0; a < numAtom; a++ ){
      atomvel[a] = atomvel[a] + atomforce[a]*dt*0.5/atomMass[a]; 
    }

    // The position and velocity variables are synced at the same
    // time step
    K=0.;
    for(Int a=0; a<numAtom; a++){
      for(Int j=0; j<3; j++){
        K += atomMass[a]*atomvel[a][j]*atomvel[a][j]/2.;
      }
    }

    Ekinetic_  = K;

    Print(statusOFS, "MD_Ekin    =  ", Ekinetic_);
    Print(statusOFS, "MD_Epot    =  ", Epot_);

    //       // Output the XYZ format for movie
    //       // Once this is written, all work associated with the current atomic
    //       // position is DONE.
    //       if( isOutputXYZ_ ){
    //         std::fstream fout;
    //         fout.open("MD.xyz",std::ios::out | std::ios::app) ;
    //         if( !fout.good() ){
    //           ErrorHandling( "Cannot open MD.xyz!" );
    //         }
    //         fout << numAtom << std::endl;
    //         fout << "MD step # "<< ionIter << std::endl;
    //         for(Int a=0; a<numAtom; a++){
    //           fout<< std::setw(6)<< atomList[a].type
    // 	      << std::setw(16)<< atompos[a][0]*au2ang
    // 	      << std::setw(16)<< atompos[a][1]*au2ang
    // 	      << std::setw(16)<< atompos[a][2]*au2ang
    // 	      << std::endl;
    //         }
    //         fout.close();
    //       }

    // Now prepare the variable for the next evalation of the force

    // Update velocity and position
    Point3 xi;
    for(Int a=0; a<numAtom; a++) {
      // Generate random variable
      Real fac = std::sqrt(2.0*damping*atomMass[a]*T*dt);
      statusOFS << "fac = " << fac << std::endl;
      xi[0] = GaussianRandom() * fac;
      xi[1] = GaussianRandom() * fac;
      xi[2] = GaussianRandom() * fac;
      Real afac, bfac;
      afac = (1.0 - damping*dt*0.5) / (1.0 + damping*dt*0.5);
      bfac = 1.0 / (1.0 + damping*dt*0.5);
      atompos[a] = atompos[a] + bfac * dt * ( atomvel[a] + 
          dt * 0.5 / atomMass[a] * atomforce[a] +
          0.5 / atomMass[a] * xi );
      atomvel[a] = afac * atomvel[a] + 
        afac * dt * 0.5 / atomMass[a] * atomforce[a] +
        bfac / atomMass[a] * xi;
    }

    // Output the velocity variable. 
    // These are the configuration that SCF will work on next. 
    // Hence if the job is stopped in the middle of SCF (which is most
    // likely), the MD job should continue from this configuration
    if(esdfParam.isOutputVelocity){
      std::fstream fout_v;
      fout_v.open("lastVel.out",std::ios::out);
      if( !fout_v.good() ){
        ErrorHandling( "File cannot be open!" );
      }
      for(Int i=0; i<numAtom; i++){
        fout_v << std::setiosflags(std::ios::scientific)
          << std::setiosflags(std::ios::showpos)
          << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< atomvel[i][0]
          << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< atomvel[i][1]
          << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< atomvel[i][2]
          << std::resetiosflags(std::ios::scientific)
          << std::resetiosflags(std::ios::showpos)
          << std::endl;
      }

      fout_v.close();

      // Also output into the statfile
    }
  } // if(mpirank == 0)

  // Sync the atomic position and velocity
  for(Int a = 0; a < numAtom; a++){
    MPI_Bcast( &atompos[a][0], 3, MPI_DOUBLE, 0, MPI_COMM_WORLD ); 
    MPI_Bcast( &atomvel[a][0], 3, MPI_DOUBLE, 0, MPI_COMM_WORLD ); 
  }

  // Update atomic position and velocity to store in atomListPtr_
  // NOTE: Force is NOT consistent with the position yet.
  for(Int a = 0; a < numAtom; a++){
    atomList[a].pos = atompos[a];
    atomList[a].vel = atomvel[a];
  }


  return ;
}


void
IonDynamics::ExtrapolateCoefficient    ( Int ionIter, DblNumVec& coef ) {
  std::vector<Atom>&   atomList = *atomListPtr_;
  Int numAtom = atomList.size();

  coef.Resize( maxHist_ );
  SetValue(coef, 0.0);
  if( MDExtrapolationType_ == "linear" ){
    coef[0] = 2.0;
    coef[1] = -1.0;
  }
  else if( MDExtrapolationType_ == "quadratic" ){
    if( ionIter < 3 ){
      coef[0] = 2.0;
      coef[1] = -1.0;
    }
    else{
      coef[0] = 3.0;
      coef[1] = -3.0;
      coef[2] = 1.0;
    }
  }
  else if( MDExtrapolationType_ == "none"){
    coef[0] = 1.0;
  }

  //        else if( MDExtrapolationType_ == "dario" ){
  //            if( ionIter < 3 ){
  //                coef[0] = 2.0;
  //                coef[1] = -1.0;
  //            }
  //            else{
  //                // FIXME
  //                // Dario extrapolation not working yet
  //
  //
  //                // Update the density through quadratic extrapolation
  //                // Dario CPC 118, 31 (1999)
  //                // huwei 20150923 
  //                // Compute the coefficient a and b
  //                Real a11 = 0.0;
  //                Real a22 = 0.0;
  //                Real a12 = 0.0;
  //                Real a21 = 0.0;
  //                Real b1 = 0.0;
  //                Real b2 = 0.0;
  //
  //                std::vector<Point3>  atemp1(numAtom);
  //                std::vector<Point3>  atemp2(numAtom);
  //                std::vector<Point3>  atemp3(numAtom);
  //
  //                for( Int i = 0; i < numAtom; i++ ){
  //                    atemp1[i] = atomListHist_[0][i].pos - atomListHist_[1][i].pos;
  //                    atemp2[i] = atomListHist_[1][i].pos - atomListHist_[2][i].pos;
  //                    atemp3[i] = atomListHist_[0][i].pos - atomList[i].pos;
  //                }
  //
  //                for( Int i = 0; i < numAtom; i++ ){
  //
  //                    a11 += atemp1[i][0]*atemp1[i][0]+atemp1[i][1]*atemp1[i][1]+atemp1[i][2]*atemp1[i][2];
  //                    a12 += atemp1[i][0]*atemp2[i][0]+atemp1[i][1]*atemp2[i][1]+atemp1[i][2]*atemp2[i][2];
  //                    a22 += atemp2[i][0]*atemp2[i][0]+atemp2[i][1]*atemp2[i][1]+atemp2[i][2]*atemp2[i][2];
  //                    a21 = a12;
  //                    b1 += 0.0-atemp3[i][0]*atemp1[i][0]-atemp3[i][1]*atemp1[i][1]-atemp3[i][2]*atemp1[i][2];
  //                    b2 += 0.0-atemp3[i][0]*atemp2[i][0]-atemp3[i][1]*atemp2[i][1]-atemp3[i][2]*atemp2[i][2];
  //                }
  //
  //
  //                Real detA = a11*a22 - a12*a21;
  //                Real aA = (b1*a22-b2*a12)/detA;
  //                Real bA = (b2*a11-b2*a21)/detA;
  //
  //
  //                //      statusOFS << "info"<< std::endl;
  //                //      statusOFS << a11 << ", " << a12 << ", " << a21 << ", " << a22 << ", " << detA << ", " << b1 
  //                //        << ", " << b2 << std::endl;
  //
  //                //      denCurVec(ii) = den0(ii) + aA * ( den0(ii) - den1(ii) ) + bA * ( den1(ii) - den2(ii) );
  //                coef[0] = 1.0 + aA;
  //                coef[1] = -aA + bA;
  //                coef[2] = -bA;
  //            }
  //        }
  else if ( MDExtrapolationType_ == "aspc2" ){
    /// Reference for ASPC schemes:
    /// J. Kolafa, Time‐reversible always stable
    /// predictor–corrector method for molecular dynamics of
    /// polarizable molecules, J. Comput. Chem. (2004).
    ///
    /// The modification is that the damping is not used since
    /// SCF is typically done relatively accurately
    ///
    /// aspc1 is the same as linear extrapolation
    if( ionIter < 3 ){
      coef[0] = 2.0;
      coef[1] = -1.0;
    }
    else{
      coef[0] = 2.5;
      coef[1] = -2.0;
      coef[2] = 0.5;
    }
  }
  else if ( MDExtrapolationType_ == "aspc3" ){
    if( ionIter < 4 ){
      coef[0] = 2.0;
      coef[1] = -1.0;
    }
    else{
      coef[0] = 2.8;
      coef[1] = -2.8;
      coef[2] = 1.2;
      coef[3] = -0.2;
    }
  }
  else{
    ErrorHandling( "Currently three extrapolation types are supported!" );
  }

  return ;
}         // -----  end of method IonDynamics::ExtrapolateCoefficient  ----- 

} // namespace dgdft



#endif // _IONDYNAMICS_CPP_
