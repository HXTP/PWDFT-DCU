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
/// @file sparse_matrix_impl.hpp
/// @brief Implementation of sparse matrices.
/// @date 2012-11-28
#ifndef _SPARSE_MATRIX_IMPL_HPP_
#define _SPARSE_MATRIX_IMPL_HPP_

#include "sparse_matrix_decl.hpp"
#include "mpi_interf.hpp"

namespace  dgdft{

extern Int SharedRead(std::string name, std::istringstream& is);

//---------------------------------------------------------
template<typename F>
  void ReadSparseMatrix ( const char* filename, SparseMatrix<F>& spmat )
  {

    std::istringstream iss;
    Int dummy;
    SharedRead( std::string(filename), iss );
    deserialize( spmat.size, iss, NO_MASK );
    deserialize( spmat.dummy, iss, NO_MASK );
    deserialize( spmat.nnz,  iss, NO_MASK );
    deserialize( spmat.colptr, iss, NO_MASK );
    deserialize( spmat.rowind, iss, NO_MASK );
    deserialize( spmat.nzval, iss, NO_MASK );


    return ;
  }        // -----  end of function ReadSparseMatrix  ----- 


template <class F> void
  ReadSparseMatrixFormatted    ( const char* filename, SparseMatrix<F>& spmat )
  {
    std::ifstream fin(filename);
    Int dummy;
    fin >> spmat.size >> dummy >> spmat.nnz;

    spmat.colptr.Resize( spmat.size+1 );
    spmat.rowind.Resize( spmat.nnz );
    spmat.nzval.Resize ( spmat.nnz );

    for( Int i = 0; i < spmat.size + 1; i++ ){
      fin >> spmat.colptr(i);
    }

    for( Int i = 0; i < spmat.nnz; i++ ){
      fin >> spmat.rowind(i);
    }

    for( Int i = 0; i < spmat.nnz; i++ ){
      fin >> spmat.nzval(i);
    }

    fin.close();

    return ;
  }        // -----  end of function ReadSparseMatrixFormatted  ----- 

//---------------------------------------------------------
template<typename F>
  void ReadDistSparseMatrix ( const char* filename, DistSparseMatrix<F>& pspmat, MPI_Comm comm )
  {
    // Get the processor information within the current communicator
    MPI_Barrier( comm );
    Int mpirank;  MPI_Comm_rank(comm, &mpirank);
    Int mpisize;  MPI_Comm_size(comm, &mpisize);
    MPI_Status mpistat;
    std::ifstream fin;

    // Read basic information
    if( mpirank == 0 ){
      fin.open(filename);
      if( !fin.good() ){
        ErrorHandling( "File cannot be openeded!" );
      }
      Int dummy;
      fin.read((char*)&pspmat.size, sizeof(Int));
      fin.read((char*)&dummy, sizeof(Int));
      fin.read((char*)&pspmat.nnz,  sizeof(Int));
    }

    pspmat.comm = comm;

    MPI_Bcast(&pspmat.size, 1, MPI_INT, 0, comm);
    MPI_Bcast(&pspmat.nnz,  1, MPI_INT, 0, comm);

    // Read colptr

    IntNumVec  colptr(pspmat.size+1);
    if( mpirank == 0 ){
      Int tmp;
      fin.read((char*)&tmp, sizeof(Int));  
      if( tmp != pspmat.size+1 ){
        ErrorHandling( "colptr is not of the right size." );
      }
      fin.read((char*)colptr.Data(), sizeof(Int)*tmp);
    }

    MPI_Bcast(colptr.Data(), pspmat.size+1, MPI_INT, 0, comm);
    //    std::cout << "Proc " << mpirank << " outputs colptr[end]" << colptr[pspmat.size] << endl;

    // Compute the number of columns on each processor
    IntNumVec numColLocalVec(mpisize);
    Int numColLocal, numColFirst;
    numColFirst = pspmat.size / mpisize;
    SetValue( numColLocalVec, numColFirst );
    numColLocalVec[mpisize-1] = pspmat.size - numColFirst * (mpisize-1);  // Modify the last entry    
    numColLocal = numColLocalVec[mpirank];

    pspmat.colptrLocal.Resize( numColLocal + 1 );
    for( Int i = 0; i < numColLocal + 1; i++ ){
      pspmat.colptrLocal[i] = colptr[mpirank * numColFirst+i] - colptr[mpirank * numColFirst] + 1;
    }

    // Calculate nnz_loc on each processor
    pspmat.nnzLocal = pspmat.colptrLocal[numColLocal] - pspmat.colptrLocal[0];

    pspmat.rowindLocal.Resize( pspmat.nnzLocal );
    pspmat.nzvalLocal.Resize ( pspmat.nnzLocal );

    // Read and distribute the row indices
    if( mpirank == 0 ){
      Int tmp;
      fin.read((char*)&tmp, sizeof(Int));  
      if( tmp != pspmat.nnz ){
        std::ostringstream msg;
        msg 
          << "The number of nonzeros in row indices do not match." << std::endl
          << "nnz = " << pspmat.nnz << std::endl
          << "size of row indices = " << tmp << std::endl;
        ErrorHandling( msg.str().c_str() );
      }
      IntNumVec buf;
      Int numRead;
      for( Int ip = 0; ip < mpisize; ip++ ){
        numRead = colptr[ip*numColFirst + numColLocalVec[ip]] - 
          colptr[ip*numColFirst];
        buf.Resize(numRead);
        fin.read( (char*)buf.Data(), numRead*sizeof(Int) );
        if( ip > 0 ){
          MPI_Send(&numRead, 1, MPI_INT, ip, 0, comm);
          MPI_Send(buf.Data(), numRead, MPI_INT, ip, 1, comm);
        }
        else{
          pspmat.rowindLocal = buf;
        }
      }
    }
    else{
      Int numRead;
      MPI_Recv(&numRead, 1, MPI_INT, 0, 0, comm, &mpistat);
      if( numRead != pspmat.nnzLocal ){
        std::ostringstream msg;
        msg << "The number of columns in row indices do not match." << std::endl
          << "numRead  = " << numRead << std::endl
          << "nnzLocal = " << pspmat.nnzLocal << std::endl;
        ErrorHandling( msg.str().c_str() );
      }

      pspmat.rowindLocal.Resize( numRead );
      MPI_Recv( pspmat.rowindLocal.Data(), numRead, MPI_INT, 0, 1, comm, &mpistat );
    }

    //    std::cout << "Proc " << mpirank << " outputs rowindLocal.size() = " 
    //        << pspmat.rowindLocal.m() << endl;


    // Read and distribute the nonzero values
    if( mpirank == 0 ){
      Int tmp;
      fin.read((char*)&tmp, sizeof(Int));  
      if( tmp != pspmat.nnz ){
        std::ostringstream msg;
        msg 
          << "The number of nonzeros in values do not match." << std::endl
          << "nnz = " << pspmat.nnz << std::endl
          << "size of values = " << tmp << std::endl;
        ErrorHandling( msg.str().c_str() );
      }
      NumVec<F> buf;
      Int numRead;
      for( Int ip = 0; ip < mpisize; ip++ ){
        numRead = colptr[ip*numColFirst + numColLocalVec[ip]] - 
          colptr[ip*numColFirst];
        buf.Resize(numRead);
        fin.read( (char*)buf.Data(), numRead*sizeof(F) );
        if( ip > 0 ){
          std::stringstream sstm;
          serialize( buf, sstm, NO_MASK );
          mpi::Send( sstm, ip, 0, 1, comm );
        }
        else{
          pspmat.nzvalLocal = buf;
        }
      }
    }
    else{
      std::stringstream sstm;
      mpi::Recv( sstm, 0, 0, 1, comm, mpistat, mpistat );
      deserialize( pspmat.nzvalLocal, sstm, NO_MASK );
      if( pspmat.nzvalLocal.m() != pspmat.nnzLocal ){
        std::ostringstream msg;
        msg << "The number of columns in values do not match." << std::endl
          << "numRead  = " << pspmat.nzvalLocal.m() << std::endl
          << "nnzLocal = " << pspmat.nnzLocal << std::endl;
        ErrorHandling( msg.str().c_str() );
      }
    }

    // Close the file
    if( mpirank == 0 ){
      fin.close();
    }



    MPI_Barrier( comm );


    return ;
  }        // -----  end of function ReadDistSparseMatrix  ----- 



template<typename F>
  void ReadDistSparseMatrixFormatted ( const char* filename, DistSparseMatrix<F>& pspmat, MPI_Comm comm )
  {
    // Get the processor information within the current communicator
    MPI_Barrier( comm );
    Int mpirank;  MPI_Comm_rank(comm, &mpirank);
    Int mpisize;  MPI_Comm_size(comm, &mpisize);
    MPI_Status mpistat;
    std::ifstream fin;

    // Read basic information
    if( mpirank == 0 ){
      fin.open(filename);
      if( !fin.good() ){
        ErrorHandling( "File cannot be openeded!" );
      }
      Int dummy;
      fin >> pspmat.size >> dummy;
      fin >> pspmat.nnz;
    }

    pspmat.comm = comm;

    MPI_Bcast(&pspmat.size, 1, MPI_INT, 0, comm);
    MPI_Bcast(&pspmat.nnz,  1, MPI_INT, 0, comm);

    // Read colptr

    IntNumVec  colptr(pspmat.size+1);
    if( mpirank == 0 ){
      Int* ptr = colptr.Data();
      for( Int i = 0; i < pspmat.size+1; i++ )
        fin >> *(ptr++);
    }

    MPI_Bcast(colptr.Data(), pspmat.size+1, MPI_INT, 0, comm);

    // Compute the number of columns on each processor
    IntNumVec numColLocalVec(mpisize);
    Int numColLocal, numColFirst;
    numColFirst = pspmat.size / mpisize;
    SetValue( numColLocalVec, numColFirst );
    numColLocalVec[mpisize-1] = pspmat.size - numColFirst * (mpisize-1);  // Modify the last entry    
    numColLocal = numColLocalVec[mpirank];

    // The first column follows the 1-based (FORTRAN convention) index.
    pspmat.firstCol = mpirank * numColFirst + 1;

    pspmat.colptrLocal.Resize( numColLocal + 1 );
    for( Int i = 0; i < numColLocal + 1; i++ ){
      pspmat.colptrLocal[i] = colptr[mpirank * numColFirst+i] - colptr[mpirank * numColFirst] + 1;
    }

    // Calculate nnz_loc on each processor
    pspmat.nnzLocal = pspmat.colptrLocal[numColLocal] - pspmat.colptrLocal[0];

    pspmat.rowindLocal.Resize( pspmat.nnzLocal );
    pspmat.nzvalLocal.Resize ( pspmat.nnzLocal );

    // Read and distribute the row indices
    if( mpirank == 0 ){
      Int tmp;
      IntNumVec buf;
      Int numRead;
      for( Int ip = 0; ip < mpisize; ip++ ){
        numRead = colptr[ip*numColFirst + numColLocalVec[ip]] - 
          colptr[ip*numColFirst];
        buf.Resize(numRead);
        Int *ptr = buf.Data();
        for( Int i = 0; i < numRead; i++ ){
          fin >> *(ptr++);
        }
        if( ip > 0 ){
          MPI_Send(&numRead, 1, MPI_INT, ip, 0, comm);
          MPI_Send(buf.Data(), numRead, MPI_INT, ip, 1, comm);
        }
        else{
          pspmat.rowindLocal = buf;
        }
      }
    }
    else{
      Int numRead;
      MPI_Recv(&numRead, 1, MPI_INT, 0, 0, comm, &mpistat);
      if( numRead != pspmat.nnzLocal ){
        std::ostringstream msg;
        msg << "The number of columns in row indices do not match." << std::endl
          << "numRead  = " << numRead << std::endl
          << "nnzLocal = " << pspmat.nnzLocal << std::endl;
        ErrorHandling( msg.str().c_str() );
      }

      pspmat.rowindLocal.Resize( numRead );
      MPI_Recv( pspmat.rowindLocal.Data(), numRead, MPI_INT, 0, 1, comm, &mpistat );
    }

#if ( _DEBUGlevel_ >= 2 )
    std::cout << "Proc " << mpirank << " outputs rowindLocal.size() = " 
      << pspmat.rowindLocal.m() << endl;
#endif


    // Read and distribute the nonzero values
    if( mpirank == 0 ){
      Int tmp;
      NumVec<F> buf;
      Int numRead;
      for( Int ip = 0; ip < mpisize; ip++ ){
        numRead = colptr[ip*numColFirst + numColLocalVec[ip]] - 
          colptr[ip*numColFirst];
        buf.Resize(numRead);
        F *ptr = buf.Data();
        for( Int i = 0; i < numRead; i++ ){
          fin >> *(ptr++);
        }
        if( ip > 0 ){
          std::stringstream sstm;
          serialize( buf, sstm, NO_MASK );
          mpi::Send( sstm, ip, 0, 1, comm );
        }
        else{
          pspmat.nzvalLocal = buf;
        }
      }
    }
    else{
      std::stringstream sstm;
      mpi::Recv( sstm, 0, 0, 1, comm, mpistat, mpistat );
      deserialize( pspmat.nzvalLocal, sstm, NO_MASK );
      if( pspmat.nzvalLocal.m() != pspmat.nnzLocal ){
        std::ostringstream msg;
        msg << "The number of columns in values do not match." << std::endl
          << "numRead  = " << pspmat.nzvalLocal.m() << std::endl
          << "nnzLocal = " << pspmat.nnzLocal << std::endl;
        ErrorHandling( msg.str().c_str() );
      }
    }

    // Close the file
    if( mpirank == 0 ){
      fin.close();
    }

    MPI_Barrier( comm );


    return ;
  }        // -----  end of function ReadDistSparseMatrixFormatted  ----- 


template<typename F>
  void WriteDistSparseMatrixFormatted ( 
      const char* filename, 
      DistSparseMatrix<F>& pspmat    )
  {
    // Get the processor information within the current communicator
    MPI_Comm comm = pspmat.comm;
    Int mpirank;  MPI_Comm_rank(comm, &mpirank);
    Int mpisize;  MPI_Comm_size(comm, &mpisize);

    MPI_Status mpistat;
    std::ofstream ofs;

    // Write basic information
    if( mpirank == 0 ){
      ofs.open(filename, std::ios_base::out);
      if( !ofs.good() ){
        ErrorHandling( "File cannot be openeded!" );
      }
      ofs << std::setiosflags(std::ios::left) 
        << std::setw(LENGTH_VAR_DATA) << pspmat.size
        << std::setw(LENGTH_VAR_DATA) << pspmat.size
        << std::setw(LENGTH_VAR_DATA) << pspmat.nnz << std::endl;
      ofs.close();
    }

    // Write colptr information, one processor after another
    IntNumVec colptrSizeLocal(mpisize);
    SetValue( colptrSizeLocal, 0 );
    IntNumVec colptrSize(mpisize);
    SetValue( colptrSize, 0 );
    colptrSizeLocal(mpirank) = pspmat.colptrLocal[pspmat.colptrLocal.Size()-1] - 1;
    mpi::Allreduce( colptrSizeLocal.Data(), colptrSize.Data(),
        mpisize, MPI_SUM, comm );
    IntNumVec colptrStart(mpisize);
    colptrStart[0] = 1;
    for( Int l = 1; l < mpisize; l++ ){
      colptrStart[l] = colptrStart[l-1] + colptrSize[l-1];
    }
    for( Int p = 0; p < mpisize; p++ ){
      if( mpirank == p ){
        ofs.open(filename, std::ios_base::out | std::ios_base::app );
        if( !ofs.good() ){
          ErrorHandling( "File cannot be openeded!" );
        }
        IntNumVec& colptrLocal = pspmat.colptrLocal;
        for( Int i = 0; i < colptrLocal.Size() - 1; i++ ){
          ofs << std::setiosflags(std::ios::left) 
            << colptrLocal[i] + colptrStart[p] - 1 << "  ";
        }
        if( p == mpisize - 1 ){
          ofs << std::setiosflags(std::ios::left) 
            << colptrLocal[colptrLocal.Size()-1] + colptrStart[p] - 1 << std::endl;
        }
        ofs.close();
      }

      MPI_Barrier( comm );
    }    

    // Write rowind information, one processor after another
    for( Int p = 0; p < mpisize; p++ ){
      if( mpirank == p ){
        ofs.open(filename, std::ios_base::out | std::ios_base::app );
        if( !ofs.good() ){
          ErrorHandling( "File cannot be openeded!" );
        }
        IntNumVec& rowindLocal = pspmat.rowindLocal;
        for( Int i = 0; i < rowindLocal.Size(); i++ ){
          ofs << std::setiosflags(std::ios::left) 
            << rowindLocal[i] << "  ";
        }
        if( p == mpisize - 1 ){
          ofs << std::endl;
        }
        ofs.close();
      }

      MPI_Barrier( comm );
    }    

    // Write nzval information, one processor after another
    for( Int p = 0; p < mpisize; p++ ){
      if( mpirank == p ){
        ofs.open(filename, std::ios_base::out | std::ios_base::app );
        if( !ofs.good() ){
          ErrorHandling( "File cannot be openeded!" );
        }
        NumVec<F>& nzvalLocal = pspmat.nzvalLocal;
        for( Int i = 0; i < nzvalLocal.Size(); i++ ){
          ofs << std::setiosflags(std::ios::left) 
            << std::setiosflags(std::ios::scientific)
            << std::setiosflags(std::ios::showpos)
            << std::setprecision(LENGTH_FULL_PREC)
            << nzvalLocal[i] << "  ";
        }
        if( p == mpisize - 1 ){
          ofs << std::endl;
        }
        ofs.close();
      }

      MPI_Barrier( comm );
    }    

    MPI_Barrier( comm );


    return ;
  }        // -----  end of function WriteDistSparseMatrixFormatted  ----- 

template<typename F>
  void ParaReadDistSparseMatrix ( 
      const char* filename, 
      DistSparseMatrix<F>& pspmat,
      MPI_Comm comm    )
  {
    // Get the processor information within the current communicator
    MPI_Barrier( comm );
    Int mpirank;  MPI_Comm_rank(comm, &mpirank);
    Int mpisize;  MPI_Comm_size(comm, &mpisize);
    MPI_Status mpistat;
    MPI_Datatype type;
    Int lens[3];
    MPI_Aint disps[3];
    MPI_Datatype types[3];
    Int err = 0;



    Int filemode = MPI_MODE_RDONLY | MPI_MODE_UNIQUE_OPEN;

    MPI_File fin;
    MPI_Status status;


    err = MPI_File_open(comm,(char*) filename, filemode, MPI_INFO_NULL,  &fin);

    if (err != MPI_SUCCESS) {
      ErrorHandling( "File cannot be opened!" );
    }

    // Read header
    if( mpirank == 0 ){
      err = MPI_File_read_at(fin, 0,(char*)&pspmat.size, 1, MPI_INT, &status);
      err = MPI_File_read_at(fin, sizeof(Int),(char*)&pspmat.nnz, 1, MPI_INT, &status);
    }


    /* define a struct that describes all our data */
    lens[0] = 1;
    lens[1] = 1;
    MPI_Address(&pspmat.size, &disps[0]);
    MPI_Address(&pspmat.nnz, &disps[1]);
    types[0] = MPI_INT;
    types[1] = MPI_INT;
    MPI_Type_struct(2, lens, disps, types, &type);
    MPI_Type_commit(&type);

    /* broadcast the header data to everyone */
    MPI_Bcast(MPI_BOTTOM, 1, type, 0, comm);

    MPI_Type_free(&type);

    // Compute the number of columns on each processor
    IntNumVec numColLocalVec(mpisize);
    Int numColLocal, numColFirst;
    numColFirst = pspmat.size / mpisize;
    SetValue( numColLocalVec, numColFirst );
    numColLocalVec[mpisize-1] = pspmat.size - numColFirst * (mpisize-1);  // Modify the last entry    
    numColLocal = numColLocalVec[mpirank];
    pspmat.colptrLocal.Resize( numColLocal + 1 );



    MPI_Offset myColPtrOffset = (2 + ((mpirank==0)?0:1) )*sizeof(Int) + (mpirank*numColFirst)*sizeof(Int);

    Int np1 = 0;
    lens[0] = (mpirank==0)?1:0;
    lens[1] = numColLocal + 1;

    MPI_Address(&np1, &disps[0]);
    MPI_Address(pspmat.colptrLocal.Data(), &disps[1]);

    MPI_Type_hindexed(2, lens, disps, MPI_INT, &type);
    MPI_Type_commit(&type);

    err= MPI_File_read_at_all(fin, myColPtrOffset, MPI_BOTTOM, 1, type, &status);

    if (err != MPI_SUCCESS) {
      ErrorHandling( "error reading colptr" );
    }
    MPI_Type_free(&type);

    // Calculate nnz_loc on each processor
    pspmat.nnzLocal = pspmat.colptrLocal[numColLocal] - pspmat.colptrLocal[0];


    pspmat.rowindLocal.Resize( pspmat.nnzLocal );
    pspmat.nzvalLocal.Resize ( pspmat.nnzLocal );

    //read rowIdx
    MPI_Offset myRowIdxOffset = (3 + ((mpirank==0)?-1:0) )*sizeof(Int) + (pspmat.size+1 + pspmat.colptrLocal[0])*sizeof(Int);

    lens[0] = (mpirank==0)?1:0;
    lens[1] = pspmat.nnzLocal;

    MPI_Address(&np1, &disps[0]);
    MPI_Address(pspmat.rowindLocal.Data(), &disps[1]);

    MPI_Type_hindexed(2, lens, disps, MPI_INT, &type);
    MPI_Type_commit(&type);

    err= MPI_File_read_at_all(fin, myRowIdxOffset, MPI_BOTTOM, 1, type,&status);

    if (err != MPI_SUCCESS) {
      ErrorHandling( "error reading rowind" );
    }
    MPI_Type_free(&type);


    //read nzval
    MPI_Offset myNzValOffset = (3 + ((mpirank==0)?-1:0) )*sizeof(Int) + (pspmat.size+1 + pspmat.nnz)*sizeof(Int) + pspmat.colptrLocal[0]*sizeof(F);

    lens[0] = (mpirank==0)?1:0;
    lens[1] = pspmat.nnzLocal;

    MPI_Address(&np1, &disps[0]);
    MPI_Address(pspmat.nzvalLocal.Data(), &disps[1]);

    types[0] = MPI_INT;
    // FIXME Currently only support double format
    if( sizeof(F) != sizeof(double) ){
      ErrorHandling("ParaReadDistSparseMatrix only supports double format");
    }

    types[1] = MPI_DOUBLE;

    MPI_Type_create_struct(2, lens, disps, types, &type);
    MPI_Type_commit(&type);

    err = MPI_File_read_at_all(fin, myNzValOffset, MPI_BOTTOM, 1, type,&status);

    if (err != MPI_SUCCESS) {
      ErrorHandling( "error reading nzval" );
    }

    MPI_Type_free(&type);


    //convert to local references
    for( Int i = 1; i < numColLocal + 1; i++ ){
      pspmat.colptrLocal[i] = pspmat.colptrLocal[i] -  pspmat.colptrLocal[0] + 1;
    }
    pspmat.colptrLocal[0]=1;

    MPI_Barrier( comm );

    MPI_File_close(&fin);

    return ;
  }        // -----  end of function ParaReadDistSparseMatrix  ----- 

template<typename F>
  void
  ParaWriteDistSparseMatrix ( 
      const char* filename, 
      DistSparseMatrix<F>& pspmat )
  {
    MPI_Comm  comm = pspmat.comm;
    // Get the processor information within the current communicator
    MPI_Barrier( comm );
    Int mpirank;  MPI_Comm_rank(comm, &mpirank);
    Int mpisize;  MPI_Comm_size(comm, &mpisize);
    MPI_Status mpistat;
    Int err = 0;



    int filemode = MPI_MODE_WRONLY | MPI_MODE_CREATE | MPI_MODE_UNIQUE_OPEN;

    MPI_File fout;
    MPI_Status status;



    err = MPI_File_open(comm,(char*) filename, filemode, MPI_INFO_NULL,  &fout);

    if (err != MPI_SUCCESS) {
      ErrorHandling( "File cannot be opened!" );
    }

    // Write header
    if( mpirank == 0 ){
      err = MPI_File_write_at(fout, 0,(char*)&pspmat.size, 1, MPI_INT, &status);
      err = MPI_File_write_at(fout, sizeof(Int),(char*)&pspmat.nnz, 1, MPI_INT, &status);
    }


    // Compute the number of columns on each processor
    Int numColLocal = pspmat.colptrLocal.m()-1;
    Int numColFirst = pspmat.size / mpisize;
    IntNumVec  colptrChunk(numColLocal+1);

    Int prev_nz = 0;
    MPI_Exscan(&pspmat.nnzLocal, &prev_nz, 1, MPI_INT, MPI_SUM, comm);

    for( Int i = 0; i < numColLocal + 1; i++ ){
      colptrChunk[i] = pspmat.colptrLocal[i] + prev_nz;
    }


    MPI_Datatype memtype, filetype;
    MPI_Aint disps[6];
    int blklens[6];
    // FIXME Currently only support double format
    if( sizeof(F) != sizeof(double) ){
      ErrorHandling("ParaReadDistSparseMatrix only supports double format");
    }

    MPI_Datatype types[6] = {MPI_INT,MPI_INT, MPI_INT,MPI_INT, MPI_INT,MPI_DOUBLE};

    /* set block lengths (same for both types) */
    blklens[0] = (mpirank==0)?1:0;
    blklens[1] = numColLocal+1;
    blklens[2] = (mpirank==0)?1:0;
    blklens[3] = pspmat.nnzLocal;
    blklens[4] = (mpirank==0)?1:0;
    blklens[5] = pspmat.nnzLocal;




    //Calculate offsets
    MPI_Offset myColPtrOffset, myRowIdxOffset, myNzValOffset;
    myColPtrOffset = 3*sizeof(Int) + (mpirank*numColFirst)*sizeof(Int);
    myRowIdxOffset = 3*sizeof(Int) + (pspmat.size +1  +  prev_nz)*sizeof(Int);
    myNzValOffset = 4*sizeof(Int) + (pspmat.size +1 +  pspmat.nnz)*sizeof(Int)+ prev_nz*sizeof(F);
    disps[0] = 2*sizeof(Int);
    disps[1] = myColPtrOffset;
    disps[2] = myRowIdxOffset;
    disps[3] = sizeof(Int)+myRowIdxOffset;
    disps[4] = myNzValOffset;
    disps[5] = sizeof(Int)+myNzValOffset;



#if ( _DEBUGlevel_ >= 1 )
    char msg[200];
    char * tmp = msg;
    tmp += sprintf(tmp,"P%d ",mpirank);
    for(Int i = 0; i<6; ++i){
      if(i==5)
        tmp += sprintf(tmp, "%d [%d - %d] | ",i,disps[i],disps[i]+blklens[i]*sizeof(F));
      else
        tmp += sprintf(tmp, "%d [%d - %d] | ",i,disps[i],disps[i]+blklens[i]*sizeof(Int));
    }
    tmp += sprintf(tmp,"\n");
    printf("%s",msg);
#endif




    MPI_Type_create_struct(6, blklens, disps, types, &filetype);
    MPI_Type_commit(&filetype);

    /* create memory type */
    Int np1 = pspmat.size+1;
    MPI_Address( (void *)&np1,  &disps[0]);
    MPI_Address(colptrChunk.Data(), &disps[1]);
    MPI_Address( (void *)&pspmat.nnz,  &disps[2]);
    MPI_Address((void *)pspmat.rowindLocal.Data(),  &disps[3]);
    MPI_Address( (void *)&pspmat.nnz,  &disps[4]);
    MPI_Address((void *)pspmat.nzvalLocal.Data(),   &disps[5]);

    MPI_Type_create_struct(6, blklens, disps, types, &memtype);
    MPI_Type_commit(&memtype);



    /* set file view */
    err = MPI_File_set_view(fout, 0, MPI_BYTE, filetype, "native",MPI_INFO_NULL);

    /* everyone writes their own row offsets, columns, and 
     * data with one big noncontiguous write (in memory and 
     * file)
     */
    err = MPI_File_write_all(fout, MPI_BOTTOM, 1, memtype, &status);

    MPI_Type_free(&filetype);
    MPI_Type_free(&memtype);





    MPI_Barrier( comm );

    MPI_File_close(&fout);


    return ;
  }        // -----  end of function ParaWriteDistSparseMatrix  ----- 


} // namespace dgdft

#endif // _SPARSE_MATRIX_IMPL_HPP_
