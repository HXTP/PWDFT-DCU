/*
   Copyright (c) 2012 The Regents of the University of California,
   through Lawrence Berkeley National Laboratory.  

Author: Lin Lin

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
/// @file diagonalize.cpp
/// @brief Read a matrix generated from DGDFT in parallel, convert to
/// block cyclic format, and solve the standard eigenvalue problem, or
/// the generalized eigenvalue problem using ScaLAPACK.
///
/// @date 2013-10-15
#include "sparse_matrix_decl.hpp"
#include "dgdft.hpp"

using namespace dgdft;
using namespace std;
using namespace dgdft::esdf;
using namespace dgdft::scalapack;


void Usage(){
  cout 
    << "Read a matrix generated from DGDFT in parallel, " << endl
    << "convert to block cyclic format, and solve the standard eigenvalue problem, " << endl
    << "or the generalized eigenvalue problem using ScaLAPACK." << endl << endl
    << "Usage: diagonalize -H [Hfile] [-S [Sfile] ] -r [nprow] -c [npcol] -MB [Blocksize] "
    << "-D [Diagonalization method] -NE [numEig] -O [outputOptions] -OE [outputEig]" << endl << endl;
}

int main(int argc, char **argv) 
{
  MPI_Init(&argc, &argv);
  int mpirank, mpisize;
  MPI_Comm_rank( MPI_COMM_WORLD, &mpirank );
  MPI_Comm_size( MPI_COMM_WORLD, &mpisize );

  if( mpirank == 0 )
    Usage();

  try
  {

    Real timeSta, timeEnd;

    // *********************************************************************
    // Input parameter
    // *********************************************************************
    std::map<std::string,std::string> options;

    OptionsCreate(argc, argv, options);

    // Default processor number 
    Int nprow, npcol;
    for( Int i = IRound(sqrt(double(mpisize))); i <= mpisize; i++){
      nprow = i; npcol = mpisize / nprow;
      if( nprow * npcol == mpisize ) break;
    } 

    if( options.find("-r") != options.end() ){
      if( options.find("-c") != options.end() ){
        nprow= atoi(options["-r"].c_str());
        npcol= atoi(options["-c"].c_str());
        if(nprow*npcol > mpisize){
          throw std::runtime_error("The number of used processors cannot be higher than the total number of available processors." );
        } 
      }
      else{
        throw std::runtime_error( "When using -r option, -c also needs to be provided." );
      }
    }
    else if( options.find("-c") != options.end() ){
      if( options.find("-r") != options.end() ){
        nprow= atoi(options["-r"].c_str());
        npcol= atoi(options["-c"].c_str());
        if(nprow*npcol > mpisize){
          throw std::runtime_error("The number of used processors cannot be higher than the total number of available processors." );
        } 
      }
      else{
        throw std::runtime_error( "When using -c option, -r also needs to be provided." );
      }
    }


    string Hfile;
    if( options.find("-H") != options.end() ){ 
      Hfile = options["-H"];
    }
    else{
      throw std::logic_error("Hfile must be provided.");
    }

    string Sfile;
    Int isSIdentity;
    if( options.find("-S") != options.end() ){ 
      Sfile = options["-S"];
      isSIdentity = 0;
    }
    else{
      isSIdentity = 1;
    }


    Int blockSize;
    if( options.find("-MB") != options.end() ){ 
      blockSize = atoi(options["-MB"].c_str());
    }
    else{
      // Default block size for ScaLAPACK
      blockSize = 32;  
    }

    Int numEig;
    if( options.find("-NE") != options.end() ){ 
      numEig = atoi(options["-NE"].c_str());
      if( numEig < 0 ){
        throw std::runtime_error("Number of eigenvalues must be 0 (comput all eigenvalues) or > 0 (partial diagonalization)" );
      }
    }
    else{
      // Default NE for computing all eigenvalues
      numEig = 0;
    }

    Int routine;
    if( options.find("-D") != options.end() ){ 
      routine = atoi(options["-D"].c_str());
      if( routine != 0 && routine != 1 ){
        throw std::runtime_error("Diagonalization must be 0 (Syevr) or 1 (Syevd).");
      }
    }
    else{
      // Default routine: SYEVR
      routine = 0;
    }

    Int outputOptions;
    if( options.find("-O") != options.end() ){ 
      outputOptions = atoi(options["-O"].c_str());
      if( outputOptions != 0 && outputOptions != 1 ){
        throw std::runtime_error("outputOptions must be 0 (single output) or 1 (multiple output).");
      }
    }
    else{
      // Default output options
      outputOptions = 0;
    }


    Int outputNE;
    if( options.find("-OE") != options.end() ){ 
      outputNE = atoi(options["-OE"].c_str());
      if( outputNE != 0 && outputNE != 1 ){
        throw std::runtime_error("outputNE must be 0 or 1.");
      }
    }
    else{
      // Default output options for eigenvalues
      outputNE = 1;
    }


    // Output input parameters

    if( outputOptions == 1 ){
      // Multiple output
      stringstream  ss; ss << "logTest" << mpirank;
      statusOFS.open( ss.str().c_str() );
    }
    else{
      // Single output
      if( mpirank == 0 ){
        stringstream  ss; ss << "logTest";
        statusOFS.open( ss.str().c_str() );
      }
    }



    PrintBlock( statusOFS, "Input parameters" );
    Print( statusOFS, "nprow                   = ", nprow );
    Print( statusOFS, "npcol                   = ", npcol ); 
    Print( statusOFS, "H file                  = ", Hfile );
    if( isSIdentity == 0 ){
      Print( statusOFS, "S file                  = ", Sfile );
    }
    else{
      statusOFS << "S is an identity matrix." << std::endl;
    }
    Print( statusOFS, "Block size              = ", blockSize );
    if( numEig == 0 )
      Print( statusOFS, "Number of eigs          = ALL" );
    else    
      Print( statusOFS, "Number of eigs          = ", numEig );

    if( routine == 0 )
      Print( statusOFS, "Diagonalization routine = PDSYEVR" );
    if( routine == 1 )
      Print( statusOFS, "Diagonalization routine = PDSYEVD" );

    if( outputOptions == 0 )
      Print( statusOFS, "Output options          = single logTest" );
    else
      Print( statusOFS, "Output options          = multiple logTest*" );


    statusOFS << endl;

    // *********************************************************************
    // Read input matrix
    // *********************************************************************


    DistSparseMatrix<Real> HMat;
    DistSparseMatrix<Real> SMat;

    if( isSIdentity == 0 ){
      GetTime( timeSta );
      ParaReadDistSparseMatrix( Hfile.c_str(), HMat, MPI_COMM_WORLD); 
      ParaReadDistSparseMatrix( Sfile.c_str(), SMat, MPI_COMM_WORLD); 
      GetTime( timeEnd );

      statusOFS << "Time for reading H and S is " << timeEnd - timeSta << endl;
      statusOFS << "H.size = " << HMat.size << endl;
      statusOFS << "H.nnz  = " << HMat.nnz  << endl;
      statusOFS << "S.size = " << SMat.size << endl;
      statusOFS << "S.nnz  = " << SMat.nnz  << endl;
    }
    else{
      GetTime( timeSta );
      ParaReadDistSparseMatrix( Hfile.c_str(), HMat, MPI_COMM_WORLD ); 
      GetTime( timeEnd );

      statusOFS << "Time for reading H is " << timeEnd - timeSta << endl;
      statusOFS << "H.size = " << HMat.size << endl;
      statusOFS << "H.nnz  = " << HMat.nnz  << endl;

      SMat.size = 0;
      SMat.nnz  = 0;
      SMat.nnzLocal = 0;
      SMat.comm = HMat.comm; 
    }



    // *********************************************************************
    // Convert the H and S matrix into the ScaLAPACK format
    // *********************************************************************

    // Initialize BLACS
    Int contxt;
    Cblacs_get(0, 0, &contxt);
    Cblacs_gridinit(&contxt, "C", nprow, npcol);

    scalapack::Descriptor descH, descS;

    scalapack::ScaLAPACKMatrix<Real>  scaH, scaS, scaZ;
    std::vector<Real> eigs;

    GetTime( timeSta );
    descH.Init( HMat.size, HMat.size, blockSize, blockSize, 
        0, 0, contxt );
    DistSparseMatToScaMat( HMat, descH, scaH, MPI_COMM_WORLD );

    if( isSIdentity == 0 ){
      descS.Init( SMat.size, SMat.size, blockSize, blockSize, 
          0, 0, contxt );
      DistSparseMatToScaMat( SMat, descS, scaS, MPI_COMM_WORLD );
    }
    GetTime( timeEnd );

    statusOFS << "Time for converting the matrix is " << timeEnd - timeSta << endl;




    // *********************************************************************
    // Solve using ScaLAPACK
    // *********************************************************************


    if( isSIdentity == 0 ){
      // Generalized eigenvalue problem
      // Only use Syevr to solve the standard problem

      if( routine != 0 ){
        std::ostringstream msg;
        msg << "For generalized eigenvalue problem, currently only PDSYEVR " 
          << "(with option -D 0) is supported." << std::endl;
        throw std::runtime_error( msg.str().c_str() ); 
      }

      Real timePartSta, timePartEnd;

      GetTime( timeSta );


      GetTime( timePartSta );
      scalapack::Potrf( 'U', scaS );
      GetTime( timePartEnd );

      statusOFS << "Time for Potrf is " 
        << timePartEnd - timePartSta << endl;

      GetTime( timePartSta );
      scalapack::Sygst( 1, 'U', scaH, scaS );
      GetTime( timePartEnd );

      statusOFS << "Time for Sygst is " 
        << timePartEnd - timePartSta << endl;


      GetTime( timePartSta );
      if( numEig > 0 )
        scalapack::Syevr('U', scaH, eigs, scaZ, 1, numEig );
      else
        scalapack::Syevr('U', scaH, eigs, scaZ );
      GetTime( timePartEnd );

      statusOFS << "Time for Syevr is " 
        << timePartEnd - timePartSta << endl;


      GetTime( timePartSta );
      scalapack::Trsm('L', 'U', 'N', 'N', 1.0, scaS, scaZ);
      GetTime( timePartEnd );

      statusOFS << "Time for Trsm is " 
        << timePartEnd - timePartSta << endl;



      GetTime( timeEnd );

      statusOFS << "Time for solve the generalized eigenvalue problem is "
        << timeEnd - timeSta << endl;
    }
    else{
      // Standard eigenvalue problem
      GetTime( timeSta );

      if( routine == 0 && numEig > 0 )
        scalapack::Syevr('U', scaH, eigs, scaZ, 1, numEig );
      if( routine == 0 && numEig == 0 )
        scalapack::Syevr('U', scaH, eigs, scaZ );
      if( routine == 1 )
        scalapack::Syevd('U', scaH, eigs, scaZ );

      GetTime( timeEnd );

      statusOFS << "Time for solve the standard eigenvalue problem is "
        << timeEnd - timeSta << endl;
    }

    // *********************************************************************
    // Post-processing
    // *********************************************************************

    if( outputNE )
      statusOFS << "Eigenvalues = " << eigs << endl;

    if( outputOptions == 1 || mpirank == 0 ){
      statusOFS.close();
    }

  }
  catch( std::exception& e )
  {
    std::cerr << " caught exception with message: "
      << e.what() << std::endl;
  }

  MPI_Finalize();

  return 0;
}
