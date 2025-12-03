/*
   Copyright (c) 2012 The Regents of the University of California,
   through Lawrence Berkeley National Laboratory.  

Author: Lin Lin and Wei Hu

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
/// @file dgdft.cpp
/// @brief Main driver for DGDFT for self-consistent field iteration.
/// @date 2012-09-16 Original version
/// @date 2014-02-11 Dual grid implementation
/// @date 2014-08-06 Intra-element parallelization
/// @date 2016-03-07 Refactoring DGDFT to include geometry optimization
/// and molecular dynamics.
#include "dgdft.hpp"

using namespace dgdft;
using namespace std;
using namespace dgdft::esdf;
using namespace dgdft::scalapack;

void Usage(){
  std::cout 
    << "dgdft -in [inFile]" << std::endl
    << "in:             Input file (default: config.yaml)" << std::endl;
}

int main(int argc, char **argv) 
{
  MPI_Init(&argc, &argv);
  int mpirank, mpisize;
  MPI_Comm_rank( MPI_COMM_WORLD, &mpirank );
  MPI_Comm_size( MPI_COMM_WORLD, &mpisize );

  Real timeSta, timeEnd;

  if( mpirank == 0 )
    Usage();

  try
  {
    // *********************************************************************
    // Input parameter
    // *********************************************************************

    // Initialize input parameters
    std::map<std::string,std::string> options;
    OptionsCreate(argc, argv, options);

    std::string inFile;
    if( options.find("-in") != options.end() ){
      inFile = options["-in"];
    }
    else{
      inFile = "config.yaml";
    }

    std::string outFile;
    if( options.find("-out") != options.end() ){
      outFile = options["-out"];
    }
    else{
      outFile = "statfile";
    }

    // Initialize log file
#ifdef _RELEASE_
    // In the release mode, only the master processor outputs information
    if( mpirank == 0 )
    {
      stringstream  ss;
      ss << "statfile." << mpirank;
      statusOFS.open( ss.str().c_str() );
    }
#else
   // Every processor outputs information
    {
      stringstream  ss;
      ss << "statfile." << mpirank;
      statusOFS.open( ss.str().c_str() );
    }
#endif

    // Initialize FFTW
    fftw_mpi_init();

    Print( statusOFS, "mpisize = ", mpisize );
    Print( statusOFS, "mpirank = ", mpirank );

    // Read ESDF input file. Note: esdfParam is a global variable (11/25/2016)
    GetTime( timeSta );

    ESDFReadInput( inFile.c_str() );

    GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 1 )
    statusOFS << "Time for reading the input file is " <<
      timeEnd - timeSta << " [s]" << std::endl;
#endif
    // Print the initial state
    ESDFPrintInput( );

    // *********************************************************************
    // Preparation
    // *********************************************************************

    // IMPORTANT: RandomSeed cannot be the same due to initialization of wavefunction
    SetRandomSeed(mpirank);

    // Setup BLACS
    Int nprow, npcol;
    Int contxt;
    Int dmCol, dmRow;
    Index3  numElem = esdfParam.numElem;
    {
      // Note the usage of notation can be a bit misleading here:
      // dmRow is the number of processors per row, which normally is
      // denoted by number of column processors
      // dmCol is the number of processors per column, which normally is
      // denoted by number of row processors
      dmCol = numElem[0] * numElem[1] * numElem[2];
      dmRow = mpisize / dmCol;
      Int numALBElement = esdfParam.numALBElem(0,0,0);

      if( mpisize > (dmCol * numALBElement) ){
        std::ostringstream msg;
        msg << "Total number of processors is too large! " << std::endl;
        msg << "The maximum number of processors is " << dmCol * numALBElement << std::endl;
        throw std::runtime_error( msg.str().c_str() );
      }

      // Cblacs_gridinit(&contxt, "C", dmRow, dmCol);
      Int numProcScaLAPACK = esdfParam.numProcScaLAPACK;

      // Here nprow, npcol is for the usage of ScaLAPACK in
      // diagonalization

      for( Int i = IRound(sqrt(double(numProcScaLAPACK))); 
          i <= numProcScaLAPACK; i++ ){
        nprow = i; npcol = numProcScaLAPACK / nprow;
        if( nprow * npcol == numProcScaLAPACK ) break;
      }

      Int ldpmap = npcol;
      IntNumVec pmap(numProcScaLAPACK);
      // Take the first numProcScaLAPACK processors for diagonalization
      for ( Int i = 0; i < numProcScaLAPACK; i++ ){
        pmap[i] = i;
      }

      contxt = -1; 
      Cblacs_get( 0, 0, &contxt );
      Cblacs_gridmap(&contxt, &pmap[0], nprow, nprow, npcol);

      if( (mpisize % dmCol) != 0 ){
        std::ostringstream msg;
        msg << "Total number of processors do not fit to the number processors per element." << std::endl;
        throw std::runtime_error( msg.str().c_str() );
      }
    } // BLACS

    GetTime( timeSta );

    Domain&  dm = esdfParam.domain;
    PeriodTable ptable;
    ptable.Setup( );

    GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 1 )
    statusOFS << "Time for setting up the periodic table is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

    // Setup the element and extended element information
    DistVec<Index3, EigenSolver, ElemPrtn>  distEigSol; 
    DistVec<Index3, KohnSham, ElemPrtn>     distHamKS;
    DistVec<Index3, Spinor, ElemPrtn>       distPsi;

    // All extended elements share the same Fourier structure.
    Fourier fftExtElem;

    // Element partition 
    {
      GetTime( timeSta );

      distEigSol.SetComm( dm.comm );
      distHamKS.SetComm( dm.comm );
      distPsi.SetComm( dm.comm );

      IntNumTns& elemPrtnInfo = distEigSol.Prtn().ownerInfo;
      elemPrtnInfo.Resize( numElem[0], numElem[1], numElem[2] );

      Int cnt = 0;
      for( Int k=0; k< numElem[2]; k++ )
        for( Int j=0; j< numElem[1]; j++ )
          for( Int i=0; i< numElem[0]; i++ ) {
            elemPrtnInfo(i,j,k) = cnt++;
          }

      distHamKS.Prtn() = distEigSol.Prtn();
      distPsi.Prtn()   = distEigSol.Prtn(); 

      for( Int k=0; k< numElem[2]; k++ )
        for( Int j=0; j< numElem[1]; j++ )
          for( Int i=0; i< numElem[0]; i++ ) {
            Index3 key (i,j,k);
            if( distEigSol.Prtn().Owner(key) == (mpirank / dmRow) ){
              // Setup the domain in the extended element
              Domain dmExtElem;
              for( Int d = 0; d < DIM; d++ ){

                dmExtElem.comm    = dm.rowComm;
                dmExtElem.rowComm = dm.rowComm;
                dmExtElem.colComm = dm.rowComm;

                // Assume the global domain starts from 0.0
                if( numElem[d] == 1 ){
                  dmExtElem.length[d]      = dm.length[d];
                  dmExtElem.numGrid[d]     = esdfParam.numGridWavefunctionElem[d];
                  dmExtElem.numGridFine[d] = esdfParam.numGridDensityElem[d];
                  dmExtElem.posStart[d]    = 0.0;
                }
                else if ( numElem[d] >= 3 ){
                  dmExtElem.length[d]      = dm.length[d]  / numElem[d] * 3;
                  dmExtElem.numGrid[d]     = esdfParam.numGridWavefunctionElem[d] * 3;
                  dmExtElem.numGridFine[d] = esdfParam.numGridDensityElem[d] * 3;
                  dmExtElem.posStart[d]    = dm.length[d]  / numElem[d] * ( key[d] - 1 );
                }
                else{
                  throw std::runtime_error( "numElem[d] is either 1 or >=3." );
                }

                // Do not specify the communicator for the domain yet
                // since it is not used for parallelization
              } // for d

              // Atoms    
              std::vector<Atom>&  atomList = esdfParam.atomList;
              std::vector<Atom>   atomListExtElem;

              Int numAtom = atomList.size();

              const Real eps = 1e-8;
              for( Int a = 0; a < numAtom; a++ ){
                Point3 pos = atomList[a].pos;
                if( IsInSubdomain( pos, dmExtElem, dm.length ) ){
                  // Update the coordinate relative to the extended
                  // element
                  for( Int d = 0; d < DIM; d++ ){
 //                   pos[d] -= floor( ( pos[d] - dmExtElem.posStart[d] ) / 
//                        dm.length[d] )* dm.length[d]; 
                  // Check how to obtain the correct atmoic positions for each extended element.
                    pos[d] -= floor( ( pos[d] - dmExtElem.posStart[d] + eps ) / 
                      dm.length[d] )* dm.length[d]; 
                  }
                  atomListExtElem.push_back( Atom( atomList[a].type,
                        pos, atomList[a].vel, atomList[a].force ) );
                } // Atom is in the extended element
              }

              statusOFS << " number of atoms in extended elements " << atomListExtElem.size() << std::endl;

              // Fourier
              fftExtElem.Initialize( dmExtElem );
              fftExtElem.InitializeFine( dmExtElem );

              // Wavefunction
              int dmExtElemMpirank, dmExtElemMpisize;
              MPI_Comm_rank( dmExtElem.comm, &dmExtElemMpirank );
              MPI_Comm_size( dmExtElem.comm, &dmExtElemMpisize );
              int numStateTotal = esdfParam.numALBElem(i,j,k) + 
                esdfParam.numUnusedState;
              int numStateLocal, blocksize;

              statusOFS << " numStateTotal " << numStateTotal << std::endl;
              if ( numStateTotal <=  dmExtElemMpisize ) {
                blocksize = 1;

                if ( dmExtElemMpirank < numStateTotal ){
                  numStateLocal = 1; // blocksize == 1;
                }
                else { 
                  // FIXME Throw an error here
                  numStateLocal = 0;
                }
              } 
              else {  // numStateTotal >  mpisize
                if ( numStateTotal % dmExtElemMpisize == 0 ){
                  blocksize = numStateTotal / dmExtElemMpisize;
                  numStateLocal = blocksize ;
                }
                else {
                  // blocksize = ((numStateTotal - 1) / mpisize) + 1;
                  blocksize = numStateTotal / dmExtElemMpisize;
                  numStateLocal = blocksize ;
                  if ( dmExtElemMpirank < ( numStateTotal % dmExtElemMpisize ) ) {
                    numStateLocal = numStateLocal + 1 ;
                  }
                }    
              }

              Spinor& spn = distPsi.LocalMap()[key];

              spn.Setup( dmExtElem, 1, numStateTotal, numStateLocal, 0.0 );

              UniformRandom( spn.Wavefun() );

              // Hamiltonian
              // The exchange-correlation type and numExtraState is not
              // used in the extended element calculation
              KohnSham& hamKS = distHamKS.LocalMap()[key];

              hamKS.Setup( dmExtElem, atomListExtElem );

              hamKS.CalculatePseudoPotential( ptable );

              /// Initia EXX in extended elements
              if( hamKS.IsHybrid() ){
                GetTime( timeSta );
                hamKS.InitializeEXX( esdfParam.ecutWavefunction, fftExtElem );
                GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 1 )
                statusOFS << "Time for setting up the exchange for the extended Hamiltonian  part = "
                  << timeEnd - timeSta << " [s]" << std::endl;
#endif
              }
              
              // Eigensolver class
              EigenSolver& eigSol = distEigSol.LocalMap()[key];
              eigSol.Setup( hamKS, spn, fftExtElem );
            } // own this element
          } // for(i)
      GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 1 )
      statusOFS << "Time for setting up extended element is " <<
        timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
    }

    // Setup distributed Fourier
    GetTime( timeSta );

    DistFourier distfft;
    distfft.Initialize( dm, esdfParam.numProcDistFFT );

    GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 1 )
    statusOFS << "Time for setting up Distributed Fourier is " <<
      timeEnd - timeSta << " [s]" << std::endl;
#endif
    // Setup HamDG
    GetTime( timeSta );

    HamiltonianDG hamDG;
    hamDG.Setup();
    hamDG.CalculatePseudoPotential( ptable );

    if( hamDG.IsHybrid() ) {
       hamDG.InitializeEXX();
    }

    GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 1 )
    statusOFS << "Time for setting up the DG Hamiltonian is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
    // Setup SCFDG
    GetTime( timeSta );

    SCFDG scfDG;
    scfDG.Setup( hamDG, distEigSol, distfft, ptable, contxt ); // This also sets Cheby_iondynamics_schedule_flag_ = 0

    GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 1 )
    statusOFS << "Time for setting up SCFDG is " <<
      timeEnd - timeSta << " [s]" << std::endl;
#endif
    // *********************************************************************
    // Solve
    // *********************************************************************

    // Main SCF iteration
    GetTime( timeSta );
    scfDG.Iterate();
    GetTime( timeEnd );
    statusOFS << "! Total time for the SCF iteration = " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;


    // *********************************************************************
    // Geometry optimization or Molecular dynamics
    // *********************************************************************

    IonDynamics ionDyn;

    ionDyn.Setup( hamDG.AtomList(), ptable ); 

    // For density extrapolation

    Int maxHist = ionDyn.MaxHist();
    // densityHist[0] is the most recent density
    std::vector<DistDblNumVec>    densityHist(maxHist);
    // Initialize the history of density
    for( Int k=0; k< numElem[2]; k++ )
      for( Int j=0; j< numElem[1]; j++ )
        for( Int i=0; i< numElem[0]; i++ ) {
          Index3 key = Index3(i,j,k);
          if( distEigSol.Prtn().Owner(key) == (mpirank / dmRow) ){
            for( Int l = 0; l < maxHist; l++ ){
              DistDblNumVec& den    = densityHist[l];
              DistDblNumVec& denCur = hamDG.Density();
              den.LocalMap()[key]     = denCur.LocalMap()[key];
            } // for (l)
          } // own this element
        }  // for (i)

    // Main loop for geometry optimization or molecular dynamics
    // If ionMaxIter == 1, it is equivalent to single shot calculation
    Int ionMaxIter = esdfParam.ionMaxIter;
    for( Int ionIter = 1; ionIter <= ionMaxIter; ionIter++ ){
      {
        std::ostringstream msg;
        msg << "Ion move step # " << ionIter;
        PrintBlock( statusOFS, msg.str() );
      }

      // Change the SCF convergence criteria if necessary (in MD mode only)
      if( (ionIter == esdfParam.MDscfEnergyCriteriaEngageIonIter) && (ionDyn.IsGeoOpt() == 0) )
      {	
       scfDG.UpdateMDParameters( );
       statusOFS << std::endl << " Switching to energy based SCF convergence on ioniter = " << ionIter << std::endl;
      }
      
      // Make CheFSI work in iondynamics mode  
      if(ionIter >= 1)
      {	
        scfDG.set_Cheby_iondynamics_schedule_flag(1);
        scfDG.set_iondynamics_iter(ionIter); 
      }
      
      // Get the new atomic coordinates
      // NOTE: ionDyn directly updates the coordinates in Hamiltonian
      ionDyn.SetEpot( scfDG.Efree() );
      ionDyn.MoveIons( ionIter );

      // Update atomic position in the extended element
      {
        GetTime(timeSta);
        std::vector<Atom>&  atomList = hamDG.AtomList();

        for( Int k=0; k< numElem[2]; k++ )
          for( Int j=0; j< numElem[1]; j++ )
            for( Int i=0; i< numElem[0]; i++ ) {
              Index3 key (i,j,k);
              if( distEigSol.Prtn().Owner(key) == (mpirank / dmRow) ){
                // Setup the domain in the extended element
                KohnSham& hamKS = distHamKS.LocalMap()[key];
                Domain& dmExtElem = distEigSol.LocalMap()[key].FFT().domain;
                std::vector<Atom> atomListExtElem;

                Int numAtom = atomList.size();

                for( Int a = 0; a < numAtom; a++ ){
                  Point3 pos = atomList[a].pos;
                  if( IsInSubdomain( pos, dmExtElem, dm.length ) ){
                    // Update the coordinate relative to the extended
                    // element
                    for( Int d = 0; d < DIM; d++ ){
                      pos[d] -= floor( ( pos[d] - dmExtElem.posStart[d] ) / 
                          dm.length[d] )* dm.length[d];
                    }
                    atomListExtElem.push_back( Atom( atomList[a].type, 
                          pos, atomList[a].vel, atomList[a].force ) );
                  } // Atom is in the extended element
                }

                // Make a copy and update the atomList in the extended element
                hamKS.UpdateHamiltonian( atomListExtElem );
                hamKS.CalculatePseudoPotential( ptable );

              }//own this element
            }//(i)
        GetTime( timeEnd );
        statusOFS << "Time for updating the Hamiltonian in the extended elements = " 
          << timeEnd - timeSta << " [s]" << std::endl;
      }

      GetTime( timeSta );
      hamDG.UpdateHamiltonianDG( hamDG.AtomList() );
      hamDG.CalculatePseudoPotential( ptable );
      scfDG.Update( );

      // Update the density history through extrapolation
      {
        for( Int k=0; k< numElem[2]; k++ )
          for( Int j=0; j< numElem[1]; j++ )
            for( Int i=0; i< numElem[0]; i++ ) {
              Index3 key = Index3(i,j,k);
              if( distEigSol.Prtn().Owner(key) == (mpirank / dmRow) ){
                for( Int l = maxHist-1; l > 0; l-- ){
                  densityHist[l].LocalMap()[key]     = densityHist[l-1].LocalMap()[key];
                } // for (l)
                densityHist[0].LocalMap()[key] = hamDG.Density().LocalMap()[key];
              } // own this element
            }  // for (i)

        // Compute the extrapolation coefficient
        DblNumVec denCoef;
        ionDyn.ExtrapolateCoefficient( ionIter, denCoef );
        statusOFS << "Extrapolation density coefficient = " << denCoef << std::endl;

        // Update the electron density
        for( Int k=0; k< numElem[2]; k++ )
          for( Int j=0; j< numElem[1]; j++ )
            for( Int i=0; i< numElem[0]; i++ ) {
              Index3 key = Index3(i,j,k);
              if( distEigSol.Prtn().Owner(key) == (mpirank / dmRow) ){
                DblNumVec& denCurVec  = hamDG.Density().LocalMap()[key];
                SetValue( denCurVec, 0.0 );
                for( Int l = 0; l < maxHist; l++ ){
                  DblNumVec& denHistVec = densityHist[l].LocalMap()[key];

                  blas::Axpy( denCurVec.m(), denCoef[l], denHistVec.Data(),
                      1, denCurVec.Data(), 1 );
                } // for (l)
              } // own this element
            }  // for (i)
      } // density extrapolation

      GetTime( timeEnd );
      statusOFS << "Time for updating the Hamiltonian in DG = " 
        << timeEnd - timeSta << " [s]" << std::endl;

      GetTime( timeSta );
      scfDG.Iterate( );
      GetTime( timeEnd );
      statusOFS << "! Total time for the SCF iteration = " << timeEnd - timeSta
        << " [s]" << std::endl;

      // Geometry optimization
      if( ionDyn.IsGeoOpt() ){
        if( MaxForce( hamDG.AtomList() ) < esdfParam.geoOptMaxForce ){
          statusOFS << "Stopping criterion for geometry optimization has been reached." << std::endl
            << "Exit the loops for ions." << std::endl;
          break;
        }
      }
    } // ionIter

    // *********************************************************************
    // Clean up
    // *********************************************************************

    // Finish Cblacs
    if(contxt >= 0) {
      Cblacs_gridexit( contxt );
    }

    // Finish fftw
    fftw_mpi_cleanup();

    MPI_Comm_free( &dm.rowComm );
    MPI_Comm_free( &dm.colComm );

  }
  catch( std::exception& e )
  {
    std::cerr << "Processor " << mpirank << " caught exception with message: "
      << e.what() << std::endl;
  }

  MPI_Finalize();

  return 0;
}
