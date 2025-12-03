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
/// @file lufact.cpp
/// @brief Read a matrix generated from DGDFT in parallel, convert to
/// block cyclic format, and compute a resolvent using ScaLAPACK's LU
/// routine.
///
/// The correctness is verified for the Lap 2d matrix using 1 core.
///
/// @date 2015-08-17
#include "sparse_matrix_decl.hpp"
#include "dgdft.hpp"

using namespace dgdft;
using namespace std;
using namespace dgdft::esdf;
using namespace dgdft::scalapack;


// FIXME
extern "C"{
// Factorization and triangular solve
void SCALAPACK(pzgetrf)( const Int* m, const Int* n, dcomplex* A,
    const Int* ia, const Int* ja, const Int* desca, Int* ipiv,
    Int* info );

void SCALAPACK(pzgetri)( const Int* n, dcomplex* A, const Int* ia,
    const Int* ja, const Int* desca, const Int* ipiv, 
    dcomplex *work, const Int* lwork, Int *iwork, const Int *liwork, 
    Int* info );
}


void Usage(){
  cout 
    << "Read a matrix generated from DGDFT in parallel, " << endl
    << "convert to block cyclic format, and compute a resolvent using " << endl
    << "ScaLAPACK's LU routine" << endl << endl
    << "Usage: lufact -H [Hfile] [-S [Sfile] ] -r [nprow] -c [npcol] -MB [Blocksize] "
    << "-O [outputOptions] -OAinv [outputAinv]" << endl << endl;
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

    Complex zshift = Complex(0.0, 0.1);

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


    Int outputAinv;
    if( options.find("-OAinv") != options.end() ){ 
      outputAinv = atoi(options["-OAinv"].c_str());
      if( outputAinv != 0 && outputAinv != 1 ){
        throw std::runtime_error("outputAinv must be 0 or 1.");
      }
    }
    else{
      // Default output options for eigenvalues
      outputAinv = 0;
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

    if( outputOptions == 0 )
      Print( statusOFS, "Output options          = single logTest" );
    else
      Print( statusOFS, "Output options          = multiple logTest*" );
    statusOFS << "zshift                  = " << zshift << std::endl;


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


    // Construct the matrix to be inverted
    DistSparseMatrix<Complex>  AMat;
    std::vector<Int>  diagIdxLocal;
    { 
      Int numColLocal      = HMat.colptrLocal.m() - 1;
      Int numColLocalFirst = HMat.size / mpisize;
      Int firstCol         = mpirank * numColLocalFirst;

      diagIdxLocal.clear();

      for( Int j = 0; j < numColLocal; j++ ){
        Int jcol = firstCol + j + 1;
        for( Int i = HMat.colptrLocal(j)-1; 
            i < HMat.colptrLocal(j+1)-1; i++ ){
          Int irow = HMat.rowindLocal(i);
          if( irow == jcol ){
            diagIdxLocal.push_back( i );
          }
        }
      } // for (j)
    }

    AMat.size          = HMat.size;
    AMat.nnz           = HMat.nnz;
    AMat.nnzLocal      = HMat.nnzLocal;
    AMat.colptrLocal   = HMat.colptrLocal;
    AMat.rowindLocal   = HMat.rowindLocal;
    AMat.nzvalLocal.Resize( HMat.nnzLocal );
    AMat.comm = MPI_COMM_WORLD;

    if( SMat.size != 0 ){
      // S is not an identity matrix
      for( Int i = 0; i < HMat.nnzLocal; i++ ){
        AMat.nzvalLocal(i) = HMat.nzvalLocal(i) - zshift * SMat.nzvalLocal(i);
      }
    }
    else{
      // S is an identity matrix
      for( Int i = 0; i < HMat.nnzLocal; i++ ){
        AMat.nzvalLocal(i) = HMat.nzvalLocal(i);
      }

      for( Int i = 0; i < diagIdxLocal.size(); i++ ){
        AMat.nzvalLocal( diagIdxLocal[i] ) -= zshift;
      }
    } // if (SMat.size != 0 )

    // *********************************************************************
    // Convert the A matrix into ScaLAPACK format
    // *********************************************************************

    // Initialize BLACS
    Int contxt;
    Cblacs_get(0, 0, &contxt);
    Cblacs_gridinit(&contxt, "C", nprow, npcol);

    scalapack::Descriptor descA;

    scalapack::ScaLAPACKMatrix<Complex>  scaA;

    GetTime( timeSta );
    descA.Init( AMat.size, AMat.size, blockSize, blockSize, 
        0, 0, contxt );
    DistSparseMatToScaMat( AMat, descA, scaA, MPI_COMM_WORLD );

    GetTime( timeEnd );

    statusOFS << "Time for converting the matrix is " << timeEnd - timeSta << endl;


    // *********************************************************************
    // Solve using ScaLAPACK
    // *********************************************************************

    // Compute the resolvent
    {
      GetTime( timeSta );
      Int N = scaA.Height();

      Int  liwork = -1, lwork = -1, info;
      std::vector<Complex> work(1);
      std::vector<Int>    iwork(1);
      std::vector<Int>    ipiv(N);

      SCALAPACK(pzgetrf)( &N, &N, scaA.Data(), &I_ONE, &I_ONE, 
          scaA.Desc().Values(), &ipiv[0], &info );

      SCALAPACK(pzgetri)( &N, scaA.Data(), &I_ONE, &I_ONE, 
          scaA.Desc().Values(), &ipiv[0], &work[0], &lwork, 
          &iwork[0], &liwork, &info );

      lwork = (Int)work[0].real();
      work.resize(lwork);
      liwork = iwork[0];
      iwork.resize(liwork);

      SCALAPACK(pzgetri)( &N, scaA.Data(), &I_ONE, &I_ONE, 
          scaA.Desc().Values(), &ipiv[0], &work[0], &lwork, 
          &iwork[0], &liwork, &info );

      GetTime( timeEnd );

      statusOFS << "Time for computing the resolvent is "
        << timeEnd - timeSta << endl;

    }

    // *********************************************************************
    // Post-processing
    // *********************************************************************

    // Output the matrix A. Only for 1 processor
    if( (outputAinv == 1) && (mpisize == 1) ){
      Int N = scaA.Height();
      Int lda = scaA.LocalHeight();
      Complex* ptrA = scaA.Data();
      statusOFS << "Output the matrix elements of A^{-1}" << std::endl;
      for(Int j = 0; j < N; j++ ){
        for( Int i = 0; i < N; i++ ){
          statusOFS << ptrA[i+j*lda].real() << " " << ptrA[i+j*lda].imag() << std::endl;
        }
      }
    }

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
