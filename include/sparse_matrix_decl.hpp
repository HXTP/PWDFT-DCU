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
/// @file sparse_matrix_decl.hpp
/// @brief Sparse matrix and Distributed sparse matrix in compressed
/// column format.
/// @date 2012-11-10
#ifndef _SPARSE_MATRIX_DECL_HPP_
#define _SPARSE_MATRIX_DECL_HPP_

#include "environment.hpp"
#include "numvec_impl.hpp"

namespace  dgdft{


/// @struct SparseMatrix
/// 
/// @brief SparseMatrix describes a sequential sparse matrix saved in
/// compressed sparse column format.
///
/// Note
/// ----
///
/// Since in PEXSI and PPEXSI only symmetric matrix is considered, the
/// compressed sparse row format will also be represented by the
/// compressed sparse column format.
template <class F> struct SparseMatrix{
  Int          size;                            // Matrix dimension
  Int          nnz;                             // Number of nonzeros
  IntNumVec    colptr;                          // Column index pointer
  IntNumVec    rowind;                          // Starting row index pointer
  NumVec<F>    nzval;                           // Nonzero values for the sparse matrix
};

// Commonly used
typedef SparseMatrix<Real>       DblSparseMatrix;
typedef SparseMatrix<Complex>    CpxSparseMatrix;

/// @struct DistSparseMatrix
///
/// @brief DistSparseMatrix describes a Sparse matrix in the compressed
/// sparse column format (CSC) and distributed with column major partition. 
///
/// Note
/// ----
/// 
/// Since only symmetric matrix is considered here, the compressed
/// sparse row format will also be represented by the compressed sparse
/// column format.
///
template <class F> struct DistSparseMatrix{
  /// @brief Matrix dimension.
  Int          size;         

  /// @brief Total number of nonzeros elements.
  Int          nnz;                             

  /// @brief Local number of local nonzeros elements on this processor.
  Int          nnzLocal;                        

  /// @brief The starting column on this processor.  The indices are
  /// 1-based (FORTRAN-convention).  Usually firstCol can be computed
  /// directly through the formula
  ///
  /// firstCol = mpirank * (size/mpisize) + 1
  /// 
  /// FIXME: The definition of firstCol is different in DGDFT and in
  /// PEXSI.  In PEXSI, firstCol is 0-based (C-convention).
  Int          firstCol;

  /// @brief Dimension numColLocal + 1, storing the pointers to the
  /// nonzero row indices and nonzero values in rowptrLocal and
  /// nzvalLocal, respectively.  numColLocal is the number
  /// of local columns saved on this processor. The indices are 1-based
  /// (FORTRAN-convention), i.e.  colptrLocal[0] = 1. 
  IntNumVec    colptrLocal;                     

  /// @brief Dimension nnzLocal, storing the nonzero indices.
  /// The indices are 1-based (FORTRAN-convention), i.e. the first row
  /// index is 1. 
  IntNumVec    rowindLocal;                    

  /// @brief Dimension nnzLocal, storing the nonzero values.
  NumVec<F>    nzvalLocal;                      

  /// @brief MPI communicator
  MPI_Comm     comm;        
};

// Commonly used
typedef DistSparseMatrix<Real>       DblDistSparseMatrix;
typedef DistSparseMatrix<Complex>    CpxDistSparseMatrix;

// Utility subroutines

/// @brief Read a sparse matrix from a binary file using one processor.
template<typename F>
void ReadSparseMatrix ( const char* filename, SparseMatrix<F>& spmat );

/// @brief Read a sparse matrix from a formatted file using one processor.
template <class F> void
ReadSparseMatrixFormatted    ( const char* filename, SparseMatrix<F>& spmat );

/// @brief Read a distributed sparse matrix from a binary file.
template<typename F>
void ReadDistSparseMatrix ( const char* filename, 
    DistSparseMatrix<F>& pspmat, MPI_Comm comm );

/// @brief Read a distributed sparse matrix from a formatted file.
template<typename F>
void ReadDistSparseMatrixFormatted ( const char* filename, 
    DistSparseMatrix<F>& pspmat, MPI_Comm comm );

/// @brief Write distributed sparse matrix to a formatted file.
template<typename F>
void WriteDistSparseMatrixFormatted ( 
    const char* filename, 
    DistSparseMatrix<F>& pspmat );


/// @brief Read distributed sparse matrix from an unformatted file using
/// MPI-IO.
template<typename F>
void ParaReadDistSparseMatrix( 
    const char* filename, 
    DistSparseMatrix<F>& pspmat,
    MPI_Comm comm    );


/// @brief Write distributed sparse matrix to an unformatted file using
/// MPI-IO.
template<typename F>
void ParaWriteDistSparseMatrix( 
    const char* filename, 
    DistSparseMatrix<F>& pspmat );



} // namespace dgdft




#endif // _SPARSE_MATRIX_DECL_HPP_
