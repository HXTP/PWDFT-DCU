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
/// @file hamiltonian_dg_conversion.hpp
/// @brief Various data conversion utility routines related to the DG
/// format.
/// @date 2014-03-03
#ifndef _HAMILTONIAN_DG_CONVERSION_HPP_
#define _HAMILTONIAN_DG_CONVERSION_HPP_
#include "hamiltonian_dg.hpp"

//#define _DEBUGlevel_ 0

namespace dgdft{

// *********************************************************************
// Utility routines
// *********************************************************************

/// @brief Convert a DistNumVec structure to the row partitioned vector,
/// which can be used for instance by DistFourier.
template<typename F>
  void DistNumVecToDistRowVec(
      const DistVec<Index3, NumVec<F>, ElemPrtn>&   distVec,
      NumVec<F>&                                    distRowVec,
      const Index3&                                 numGrid,
      const Index3&                                 numElem,
      Int                                           localNzStart,
      Int                                           localNz,
      bool                                          isInGrid,
      MPI_Comm                                      commDistVec )
  {

    Int mpirank, mpisize;
    MPI_Comm_rank( commDistVec, &mpirank );
    MPI_Comm_size( commDistVec, &mpisize );

    Index3  numGridElem;

    for( Int d = 0; d < DIM; d++ ){
      numGridElem[d] = numGrid[d] / numElem[d];
    }

    // Prepare distVecRecv

    DistVec<Index3, NumVec<F>, ElemPrtn>  distVecRecv;
    distVecRecv.Prtn() = distVec.Prtn();
    distVecRecv.SetComm( commDistVec );

    for(typename std::map<Index3, NumVec<F> >::const_iterator 
        mi = distVec.LocalMap().begin();
        mi != distVec.LocalMap().end(); ++mi ){
      Index3 key = (*mi).first;  
      if( distVec.Prtn().Owner( key ) != mpirank ){
        ErrorHandling( "DistNumVec owns a wrong key." );
      }
      distVecRecv.LocalMap()[key] = (*mi).second;
    }


    // Specify the elements of data to receive according to the index of
    // the z-dimension. 
    std::vector<Index3>  getKey;
    if( isInGrid ){
      for( Int k = 0; k < numElem[2]; k++ ){
        if( k     * numGridElem[2] < localNzStart + localNz &&
            (k+1) * numGridElem[2] > localNzStart ){
          // The processor needs these elements
          for( Int j = 0; j < numElem[1]; j++ ){
            for( Int i = 0; i < numElem[0]; i++ ){
              getKey.push_back( Index3(i,j,k) );
            }
          } // for (j)
        }
      } // for (k)
    } // if (isInGrid)

    // Data communication.
    distVecRecv.GetBegin( getKey, NO_MASK );
    distVecRecv.GetEnd( NO_MASK );

    // Unpack the data locally into distRowVec
    if( isInGrid ){
      Int numGridLocal = numGrid[0] * numGrid[1] * localNz;
      distRowVec.Resize( numGridLocal );
      for(typename std::map<Index3, NumVec<F> >::const_iterator 
          mi = distVecRecv.LocalMap().begin();
          mi != distVecRecv.LocalMap().end(); ++mi ){
        Index3 key = (*mi).first;  
        const NumVec<F> & val = (*mi).second;
        Int kStart = std::max( 0, localNzStart - key[2] * numGridElem[2] );
        Int kEnd   = std::min( numGridElem[2], 
            localNzStart + localNz - key[2] * numGridElem[2] );
        for( Int k = kStart; k < kEnd; k++ )
          for( Int j = 0; j < numGridElem[1]; j++ )
            for( Int i = 0; i < numGridElem[0]; i++ ){
              Index3 glbIdx( 
                  key[0] * numGridElem[0] + i,
                  key[1] * numGridElem[1] + j,
                  key[2] * numGridElem[2] + k - localNzStart );

              distRowVec( glbIdx[0] + glbIdx[1] * numGrid[0] + glbIdx[2] *
                  numGrid[0] * numGrid[1] ) = 
                val( i + j * numGridElem[0] + k * numGridElem[0] *
                    numGridElem[1] );
            } // for (i)
      }
    } // if (isInGrid)
  }  // -----  end of function DistNumVecToDistRowVec  ----- 


/// @brief Convert a distributed row partitioned vector to a DistNumVec,
/// which can be used for instance after DistFourier.
template<typename F>
  void DistRowVecToDistNumVec(
      const NumVec<F>&                              distRowVec,
      DistVec<Index3, NumVec<F>, ElemPrtn>&         distVec,
      const Index3&                                 numGrid,
      const Index3&                                 numElem,
      Int                                           localNzStart,
      Int                                           localNz,
      bool                                          isInGrid,
      MPI_Comm                                      commDistVec )
  {

    Int mpirank, mpisize;
    MPI_Comm_rank( commDistVec, &mpirank );
    MPI_Comm_size( commDistVec, &mpisize );

    Index3  numGridElem;

    distVec.LocalMap().clear();

    for( Int d = 0; d < DIM; d++ ){
      numGridElem[d] = numGrid[d] / numElem[d];
    }

    // Collect the element indicies
    std::vector<Index3> putKey;
    if( isInGrid ){
      for( Int k = 0; k < numElem[2]; k++ ){
        if( k     * numGridElem[2] < localNzStart + localNz &&
            (k+1) * numGridElem[2] > localNzStart ){
          // The current processor contains these elements
          for( Int j = 0; j < numElem[1]; j++ ){
            for( Int i = 0; i < numElem[0]; i++ ){
              putKey.push_back( Index3(i,j,k) );
            }
          } 
        }
      } // for (k)
    } // if (isInGrid)

    // Prepare the local data
    for( std::vector<Index3>::iterator vi = putKey.begin();
        vi != putKey.end(); ++vi ){
      Index3 key = (*vi);
      typename std::map<Index3, NumVec<F> >::iterator mi = 
        distVec.LocalMap().find(key);
      if( mi == distVec.LocalMap().end() ){
        NumVec<F> empty( numGridElem[0] * numGridElem[1] * numGridElem[2] );
        SetValue( empty, (F)(0) ); 
        distVec.LocalMap()[key] = empty;
      }
      NumVec<F>& val = distVec.LocalMap()[key];

      Int kStart = std::max( 0, localNzStart - key[2] * numGridElem[2] );
      Int kEnd   = std::min( numGridElem[2], 
          localNzStart + localNz - key[2] * numGridElem[2] );

      for( Int k = kStart; k < kEnd; k++ )
        for( Int j = 0; j < numGridElem[1]; j++ )
          for( Int i = 0; i < numGridElem[0]; i++ ){
            Index3 glbIdx( 
                key[0] * numGridElem[0] + i,
                key[1] * numGridElem[1] + j,
                key[2] * numGridElem[2] + k - localNzStart );

            val( i + j * numGridElem[0] + k * numGridElem[0] *
                numGridElem[1] ) +=
              distRowVec( glbIdx[0] + glbIdx[1] * numGrid[0] +
                  glbIdx[2] * numGrid[0] * numGrid[1] );
          }
    }


    // Data communication.
    distVec.PutBegin( putKey, NO_MASK );
    distVec.PutEnd( NO_MASK, PutMode::COMBINE );

    // Erase the elements distVec does not own
    std::vector<Index3>  eraseKey;
    for(typename std::map<Index3, NumVec<F> >::iterator
        mi = distVec.LocalMap().begin();
        mi != distVec.LocalMap().end(); ++mi ){
      Index3 key = (*mi).first;
      if( distVec.Prtn().Owner( key ) != mpirank ){
        eraseKey.push_back( key );
      }
    }
    for( std::vector<Index3>::iterator vi = eraseKey.begin();
        vi != eraseKey.end(); ++vi ){
      distVec.LocalMap().erase( *vi );
    }


  }  // -----  end of function DistNumVecToDistRowVec  ----- 


/// @brief Convert a matrix distributed according to 2D element indices
/// to a matrix distributed block-cyclicly as in ScaLAPACK.
///
/// NOTE: 
///
/// 1. This subroutine can be considered as one implementation of Gemr2d
/// type process.
///
/// 2. This subroutine assumes that proper descriptor has been given
/// by desc, which agrees with distMat.
///
/// 3. This subroutine is mainly used for converting the DG Hamiltonian
/// matrix to ScaLAPACK format for diagonalization purpose.
///
template<typename F>
  void DistElemMatToScaMat(
      const DistVec<ElemMatKey, NumMat<F>, ElemMatPrtn>&   distMat,
      const scalapack::Descriptor&                         desc, 
      scalapack::ScaLAPACKMatrix<F>&                       scaMat,
      const NumTns<std::vector<Int> >&                     basisIdx,
      MPI_Comm                                               comm ){
    using namespace dgdft::scalapack;

    Int mpirank, mpisize;
    MPI_Comm_rank( comm, &mpirank );
    MPI_Comm_size( comm, &mpisize );

    Int nprow  = desc.NpRow();
    Int npcol  = desc.NpCol();
    Int myprow = desc.MypRow();
    Int mypcol = desc.MypCol();

    scaMat.SetDescriptor( desc );

    {
      std::vector<F>& vec = scaMat.LocalMatrix();
      for( Int i = 0; i < vec.size(); i++ ){
        vec[i] = (F) 0.0;
      }
    }

    Int MB = scaMat.MB(); 
    Int NB = scaMat.NB();

    if( MB != NB ){
      ErrorHandling("MB must be equal to NB.");
    }

    Int numRowBlock = scaMat.NumRowBlocks();
    Int numColBlock = scaMat.NumColBlocks();

    // Get the processor map
    IntNumMat  procGrid( nprow, npcol );
    SetValue( procGrid, 0 );
    {
      IntNumMat  procTmp( nprow, npcol );
      SetValue( procTmp, 0 );
      procTmp( myprow, mypcol ) = mpirank;
      mpi::Allreduce( procTmp.Data(), procGrid.Data(), nprow * npcol,
          MPI_SUM, comm );
    }


    BlockMatPrtn  blockPrtn;
    blockPrtn.ownerInfo.Resize( numRowBlock, numColBlock );
    IntNumMat&    blockOwner = blockPrtn.ownerInfo;
    for( Int jb = 0; jb < numColBlock; jb++ ){
      for( Int ib = 0; ib < numRowBlock; ib++ ){
        blockOwner( ib, jb ) = procGrid( ib % nprow, jb % npcol );
      }
    }

#if ( _DEBUGlevel_ >= 1 )
    statusOFS << "blockPrtn.ownerInfo = " << blockPrtn.ownerInfo << std::endl;
#endif


    // Intermediate variable for constructing the distributed matrix in
    // ScaLAPACK format.  
    DistVec<Index2, NumMat<F>, BlockMatPrtn> distScaMat;
    distScaMat.Prtn() = blockPrtn;
    distScaMat.SetComm(comm);

    // Initialize
    DblNumMat empty( MB, MB );
    SetValue( empty, 0.0 );
    for( Int jb = 0; jb < numColBlock; jb++ )
      for( Int ib = 0; ib < numRowBlock; ib++ ){
        Index2 key( ib, jb );
        if( distScaMat.Prtn().Owner( key ) == mpirank ){
          distScaMat.LocalMap()[key] = empty;
        }
      } // for (ib)

    // Convert matrix distributed according to elements to distScaMat
    for( typename std::map<ElemMatKey, NumMat<F> >::const_iterator 
        mi  = distMat.LocalMap().begin();
        mi != distMat.LocalMap().end(); ++mi ){
      ElemMatKey elemKey = (*mi).first;
      if( distMat.Prtn().Owner( elemKey ) == mpirank ){
        Index3 key1 = elemKey.first;
        Index3 key2 = elemKey.second;
        const std::vector<Int>& idx1 = basisIdx( key1(0), key1(1), key1(2) );
        const std::vector<Int>& idx2 = basisIdx( key2(0), key2(1), key2(2) );
        const NumMat<F>& localMat = (*mi).second;
        // Skip if there is no basis functions.
        if( localMat.Size() == 0 )
          continue;

        if( localMat.m() != idx1.size() ||
            localMat.n() != idx2.size() ){
          std::ostringstream msg;
          msg 
            << "Local matrix size is not consistent." << std::endl
            << "localMat   size : " << localMat.m() << " x " << localMat.n() << std::endl
            << "idx1       size : " << idx1.size() << std::endl
            << "idx2       size : " << idx2.size() << std::endl;
          ErrorHandling( msg.str().c_str() );
        }

        // Reshape the matrix element by element
        Int ib, jb, io, jo;
        for( Int b = 0; b < localMat.n(); b++ ){
          for( Int a = 0; a < localMat.m(); a++ ){
            ib = idx1[a] / MB;
            jb = idx2[b] / MB;
            io = idx1[a] % MB;
            jo = idx2[b] % MB;
            typename std::map<Index2, NumMat<F> >::iterator 
              ni = distScaMat.LocalMap().find( Index2(ib, jb) );
            // Contributes to blocks not owned by the current processor
            if( ni == distScaMat.LocalMap().end() ){
              distScaMat.LocalMap()[Index2(ib, jb)] = empty;
              ni = distScaMat.LocalMap().find( Index2(ib, jb) );
            }
            DblNumMat&  scaMat = (*ni).second;
            scaMat(io, jo) += localMat(a, b);
          } // for (a)
        } // for (b)


      } // own this matrix block
    } // for (mi)

    // Communication of the matrix
    {
      // Prepare
      std::vector<Index2>  keyIdx;
      for( typename std::map<Index2, NumMat<F> >::iterator 
          mi  = distScaMat.LocalMap().begin();
          mi != distScaMat.LocalMap().end(); ++mi ){
        Index2 key = (*mi).first;
        if( distScaMat.Prtn().Owner( key ) != mpirank ){
          keyIdx.push_back( key );
        }
      } // for (mi)

      // Communication
      distScaMat.PutBegin( keyIdx, NO_MASK );
      distScaMat.PutEnd( NO_MASK, PutMode::COMBINE );

      // Clean to save space
      std::vector<Index2>  eraseKey;
      for( typename std::map<Index2, NumMat<F> >::iterator 
          mi  = distScaMat.LocalMap().begin();
          mi != distScaMat.LocalMap().end(); ++mi ){
        Index2 key = (*mi).first;
        if( distScaMat.Prtn().Owner( key ) != mpirank ){
          eraseKey.push_back( key );
        }
      } // for (mi)

      for( std::vector<Index2>::iterator vi = eraseKey.begin();
          vi != eraseKey.end(); ++vi ){
        distScaMat.LocalMap().erase( *vi );
      }    
    }

    // Copy the matrix values from distScaMat to scaMat
    {
      for( typename std::map<Index2, NumMat<F> >::iterator 
          mi  = distScaMat.LocalMap().begin();
          mi != distScaMat.LocalMap().end(); ++mi ){
        Index2 key = (*mi).first;
        if( distScaMat.Prtn().Owner( key ) == mpirank ){
          Int ib = key(0), jb = key(1);
          Int offset = ( jb / npcol ) * MB * scaMat.LocalLDim() + 
            ( ib / nprow ) * MB;
          lapack::Lacpy( 'A', MB, MB, (*mi).second.Data(),
              MB, scaMat.Data() + offset, scaMat.LocalLDim() );
        } // own this block
      } // for (mi)
    }

    return;
  }  // -----  end of function DistElemMatToScaMat  ----- 



///// @brief Convert a matrix distributed according to 2D element indices
///// on one communicator to a matrix distributed block-cyclicly as in
///// ScaLAPACK on another communicator, of which the processors are a
///// subset of the processors on the first communicator.
/////
///// In order to reduce communication volume and time, the data
///// distribution of the distMat should strictly follow the rule below.
/////
///// NOTE: 
/////
///// 1. This subroutine can be considered as one implementation of Gemr2d
///// type process.
/////
///// 2. This subroutine assumes that proper descriptor has been given
///// by desc, which agrees with distMat.
/////
///// 3. This subroutine is mainly used for converting the DG Hamiltonian
///// matrix to ScaLAPACK format for diagonalization purpose.
///// 
///// @param[in] distMat Matrix distributed according to 2D element
///// indices.  The data saved in each processor column of comm1 should be
///// exactly the same.
///// @param[in] desc    Descriptor for the ScaLAPACK matrix.
///// @param[out] scaMat Converted 2D block cyclically distributed
///// ScaLAPACK matrix.
///// @param[in] basisIdx Indices for all basis functions.  The input from
///// each processor column of comm1 should be exactly the same.
///// @param[in] comm Communicator for distMat. This should be the same as
///// the communicator in the global domain.
///// @param[in] rowComm This should be the same as the row communicator in
///// the global domain.
///// @param[in] colComm This should be the same as the column communicator in
///// the global domain.
///// @param[in] commSca Communicator for the scaMat, which should be compatible
///// with desc, and the processors should be the processors in comm with
///// mpirank = 0, ..., mpisizeSca-1.
///// @param[in] mpisizeSca Size of the communicator commSca.
//template<typename F>
//void DistElemMatToScaMat2(
//        const DistVec<ElemMatKey, NumMat<F>, ElemMatPrtn>&   distMat,
//        const scalapack::Descriptor&                         desc, 
//        scalapack::ScaLAPACKMatrix<F>&                       scaMat,
//        const NumTns<std::vector<Int> >&                     basisIdx,
//      MPI_Comm                                               comm,
//    MPI_Comm                                             rowComm,
//    MPI_Comm                                             colComm,
//    MPI_Comm                                             commSca,
//    const Int                                            mpisizeSca ){
//    using namespace dgdft::scalapack;
//
//
//
//    Int mpirankElem, mpisizeElem;
//    MPI_Comm_rank( comm, &mpirankElem );
//    MPI_Comm_size( comm, &mpisizeElem );
//
//  Int nprowElem, npcolElem;
//  MPI_Comm_size( rowComm, &nprowElem );
//  MPI_Comm_size( colComm, &npcolElem );
//
//  if( mpisizeSca > mpisizeElem ){
//    ErrorHandling("mpisizeSca should be <= mpisizeElem.");
//  }
//
//  // Initialize the ScaLAPACK matrix
//  Int mpirankSca;
//  Int nprowSca;
//  Int npcolSca;
//  Int myprowSca;
//  Int mypcolSca;
//  Int MB; 
//  Int NB;
//  BlockMatPrtn  blockPrtn;
//
//  if( mpirankElem < mpisizeSca ){
//    MPI_Comm_rank( commSca, &mpirankSca );
//
//    nprowSca  = desc.NpRow();
//    npcolSca  = desc.NpCol();
//    myprowSca = desc.MypRow();
//    mypcolSca = desc.MypCol();
//
//    if( nprowSca * npcolSca != mpisizeSca ){
//      ErrorHandling("nprowSca * npcolSca != mpisizeSca.");
//    }
//
//    scaMat.SetDescriptor( desc );
//
//    {
//      std::vector<F>& vec = scaMat.LocalMatrix();
//      for( Int i = 0; i < vec.size(); i++ ){
//        vec[i] = (F) 0.0;
//      }
//    }
//
//    MB = scaMat.MB(); 
//    NB = scaMat.NB();
//
//    if( MB != NB ){
//      ErrorHandling("MB must be equal to NB.");
//    }
//
//
//    // Get the processor map
//    IntNumMat  procGrid( nprowSca, npcolSca );
//    SetValue( procGrid, 0 );
//    {
//      IntNumMat  procTmp( nprowSca, npcolSca );
//      SetValue( procTmp, 0 );
//      procTmp( myprowSca, mypcolSca ) = mpirankSca;
//      mpi::Allreduce( procTmp.Data(), procGrid.Data(), nprowSca * npcolSca,
//          MPI_SUM, commSca );
//    }
//
//
//    blockPrtn.ownerInfo.Resize( scaMat.NumRowBlocks(), scaMat.NumColBlocks() );
//    IntNumMat&    blockOwner = blockPrtn.ownerInfo;
//    for( Int jb = 0; jb < scaMat.NumColBlocks(); jb++ ){
//      for( Int ib = 0; ib < scaMat.NumRowBlocks(); ib++ ){
//        blockOwner( ib, jb ) = procGrid( ib % nprowSca, jb % npcolSca );
//      }
//    }
//  } // if( mpirankElem < mpisizeSca )
//
//    // Intermediate variable for constructing the distributed matrix in
//    // ScaLAPACK format.  
//  //
//  // All processors participate in this step, though processors with
//  // mpirankElem >= mpisizeSca will not contain any data.
//  DistVec<Index2, NumMat<F>, BlockMatPrtn> distScaMat;
//    distScaMat.Prtn() = blockPrtn;
//  // Since all processors in the same processor row group have identical
//  // information of the distElemMat, only communication in the same
//  // column communicator is needed.
//  distScaMat.SetComm(colComm);
//
//  // Make sure ALL processors know what MB and an empty matrix block is.
//  MPI_Bcast( &MB, 1, MPI_INT, 0, comm );
//    DblNumMat empty( MB, MB ); 
//  SetValue( empty, 0.0 );
//
//    // Initialize the distScaMat
//  if( mpirankElem < mpisizeSca ){
//    for( Int jb = 0; jb < scaMat.NumColBlocks(); jb++ )
//      for( Int ib = 0; ib < scaMat.NumRowBlocks(); ib++ ){
//        Index2 key( ib, jb );
//        if( distScaMat.Prtn().Owner( key ) == mpirankSca ){
//          distScaMat.LocalMap()[key] = empty;
//        }
//      } // for (ib)
//  }
//
//    // Convert matrix distributed according to elements to distScaMat
//    for( typename std::map<ElemMatKey, NumMat<F> >::const_iterator 
//             mi  = distMat.LocalMap().begin();
//             mi != distMat.LocalMap().end(); mi++ ){
//        ElemMatKey elemKey = (*mi).first;
//    // Processors in different column groups proceed simultaneously
//        if( distMat.Prtn().Owner( elemKey ) == mpirank / nprowElem ){
//            Index3 key1 = elemKey.first;
//            Index3 key2 = elemKey.second;
//            const std::vector<Int>& idx1 = basisIdx( key1(0), key1(1), key1(2) );
//            const std::vector<Int>& idx2 = basisIdx( key2(0), key2(1), key2(2) );
//            const NumMat<F>& localMat = (*mi).second;
//            // Skip if there is no basis functions.
//            if( localMat.Size() == 0 )
//                continue;
//
//            if( localMat.m() != idx1.size() ||
//                    localMat.n() != idx2.size() ){
//                std::ostringstream msg;
//                msg 
//                    << "Local matrix size is not consistent." << std::endl
//                    << "localMat   size : " << localMat.m() << " x " << localMat.n() << std::endl
//                    << "idx1       size : " << idx1.size() << std::endl
//                    << "idx2       size : " << idx2.size() << std::endl;
//                ErrorHandling( msg.str().c_str() );
//            }
//
//            // Reshape the matrix element by element
//            Int ib, jb, io, jo;
//            for( Int b = 0; b < localMat.n(); b++ ){
//                for( Int a = 0; a < localMat.m(); a++ ){
//                    ib = idx1[a] / MB;
//                    jb = idx2[b] / MB;
//                    io = idx1[a] % MB;
//                    jo = idx2[b] % MB;
//                    typename std::map<Index2, NumMat<F> >::iterator 
//                        ni = distScaMat.LocalMap().find( Index2(ib, jb) );
//                    // Contributes to blocks not owned by the current processor
//                    if( ni == distScaMat.LocalMap().end() ){
//                        distScaMat.LocalMap()[Index2(ib, jb)] = empty;
//                        ni = distScaMat.LocalMap().find( Index2(ib, jb) );
//                    }
//                    DblNumMat&  scaMat = (*ni).second;
//                    scaMat(io, jo) += localMat(a, b);
//                } // for (a)
//            } // for (b)
//
//
//        } // own this matrix block
//    } // for (mi)
//
//    // Communication of the matrix
//    {
//        // Prepare
//        std::vector<Index2>  keyIdx;
//        for( typename std::map<Index2, NumMat<F> >::iterator 
//                 mi  = distScaMat.LocalMap().begin();
//                 mi != distScaMat.LocalMap().end(); mi++ ){
//            Index2 key = (*mi).first;
//            if( distScaMat.Prtn().Owner( key ) != mpirank ){
//                keyIdx.push_back( key );
//            }
//        } // for (mi)
//
//        // Communication
//        distScaMat.PutBegin( keyIdx, NO_MASK );
//        distScaMat.PutEnd( NO_MASK, PutMode::COMBINE );
//
//        // Clean to save space
//        std::vector<Index2>  eraseKey;
//        for( typename std::map<Index2, NumMat<F> >::iterator 
//                 mi  = distScaMat.LocalMap().begin();
//                 mi != distScaMat.LocalMap().end(); mi++ ){
//            Index2 key = (*mi).first;
//            if( distScaMat.Prtn().Owner( key ) != mpirank ){
//                eraseKey.push_back( key );
//            }
//        } // for (mi)
//
//        for( std::vector<Index2>::iterator vi = eraseKey.begin();
//                 vi != eraseKey.end(); vi++ ){
//            distScaMat.LocalMap().erase( *vi );
//        }    
//    }
//
//    // Copy the matrix values from distScaMat to scaMat
//    {
//        for( typename std::map<Index2, NumMat<F> >::iterator 
//                 mi  = distScaMat.LocalMap().begin();
//                 mi != distScaMat.LocalMap().end(); mi++ ){
//            Index2 key = (*mi).first;
//            if( distScaMat.Prtn().Owner( key ) == mpirank ){
//                Int ib = key(0), jb = key(1);
//                Int offset = ( jb / npcolSca ) * MB * scaMat.LocalLDim() + 
//                    ( ib / nprowSca ) * MB;
//                lapack::Lacpy( 'A', MB, MB, (*mi).second.Data(),
//                        MB, scaMat.Data() + offset, scaMat.LocalLDim() );
//            } // own this block
//        } // for (mi)
//    }
//
//    return;
//}  // -----  end of function DistElemMatToScaMat2  ----- 


/// @brief Convert a matrix distributed according to 2D element indices
/// on one communicator to a matrix distributed block-cyclicly as in
/// ScaLAPACK on another communicator.
///
///  
///
/// NOTE: 
///
/// 1. There is no restriction on the communicators, i.e. in principle
/// the two communicators can be completely different.  This may not be
/// the most efficient way for implementing this feature, but should be
/// more general and maintainable.
/// 
/// 2. This subroutine can be considered as one implementation of Gemr2d
/// type process.
///
/// 3. This subroutine assumes that proper descriptor has been given
/// by desc, which agrees with scaMat.
///
/// 4. This subroutine is mainly used for converting the DG Hamiltonian
/// matrix to ScaLAPACK format for diagonalization purpose. In the
/// current step with intra-element parallelization, the comm for
/// distElemMat can be colComm.
///
/// 5. commSca is not needed since ScaLAPACK does not use MPI
/// communicators directly.
///
/// 
/// @param[in] distMat (local) Matrix distributed according to 2D element
/// indices. 
/// @param[in] desc    (local) Descriptor for the ScaLAPACK matrix.
/// @param[out] scaMat (local) Converted 2D block cyclically distributed
/// ScaLAPACK matrix.
/// @param[in] basisIdx (global) Indices for all basis functions.  The input from
/// each processor column of comm1 should be exactly the same.
/// @param[in] comm (global) Global communicator. 
/// @param[in] commElem (local) Communicator for distElemMat. In the current step
/// with intra-element parallelization, the comm for distElemMat can be
/// colComm.
/// @param[in] mpirankElemVec (global) mpiranks in the global
/// communicator comm for processors in commElem.  This vector should
/// follow a non-decreasing order.
/// @param[in] mpirankScaVec (global) mpiranks in the global
/// communicator comm for processors using ScaLAPACK.  This vector should
/// follow a non-decreasing order, and should use the 2D column-major
/// processor mapping.
template<typename F>
  void DistElemMatToScaMat2(
      const DistVec<ElemMatKey, NumMat<F>, ElemMatPrtn>&   distMat,
      const scalapack::Descriptor&                         desc, 
      scalapack::ScaLAPACKMatrix<F>&                       scaMat,
      const NumTns<std::vector<Int> >&                     basisIdx,
      MPI_Comm                                               comm,
      MPI_Comm                                               commElem,
      const std::vector<Int>&                              mpirankElemVec,
      const std::vector<Int>&                              mpirankScaVec){
    using namespace dgdft::scalapack;


    // Global rank and size
    Int mpirank, mpisize;
    MPI_Comm_rank( comm, &mpirank);
    MPI_Comm_size( comm, &mpisize);
    bool isInElem, isInSca;
    isInElem = ( find( mpirankElemVec.begin(), 
          mpirankElemVec.end(), mpirank ) != mpirankElemVec.end() );
    isInSca  = ( find( mpirankScaVec.begin(), 
          mpirankScaVec.end(), mpirank ) != mpirankScaVec.end() );

    // commElem rank and size
    Int mpirankElem; 
    Int mpisizeElem = mpirankElemVec.size();
    if( isInElem ){
      Int tmp;
      MPI_Comm_rank( commElem, &mpirankElem );
      MPI_Comm_size( commElem, &tmp );

      if( mpisizeElem != tmp ){
        ErrorHandling("mpisizeElem read from input does not agree with commElem.");
      }
    }

    if( mpisizeElem > mpisize ){
      ErrorHandling("mpisizeElem cannot be larger than mpisize.");
    }

    // commSca size and other information
    Int mpisizeSca = mpirankScaVec.size();
    Int nprowSca;
    Int npcolSca;
    Int MB; 
    Int NB;
    Int numRowBlock;
    Int numColBlock;

    if( isInSca ){

      nprowSca  = desc.NpRow();
      npcolSca  = desc.NpCol();

      if( nprowSca * npcolSca != mpisizeSca ){
        ErrorHandling("nprowSca * npcolSca != mpisizeSca.");
      }

      scaMat.SetDescriptor( desc );

      {
        std::vector<F>& vec = scaMat.LocalMatrix();
        for( Int i = 0; i < vec.size(); i++ ){
          vec[i] = (F) 0.0;
        }
      }

      MB = scaMat.MB(); 
      NB = scaMat.NB();

      if( MB != NB ){
        ErrorHandling("MB must be equal to NB.");
      }

      numRowBlock = scaMat.NumRowBlocks();
      numColBlock = scaMat.NumColBlocks();

    }

    if( mpisizeSca > mpisize ){
      ErrorHandling("mpisizeSca cannot be larger than mpisize.");
    }

    // Make sure ALL processors in comm have some basic information
    // The root is mpirankScaSta
    Int mpirankScaSta = mpirankScaVec[0];
    MPI_Bcast( &MB, 1, MPI_INT, mpirankScaSta, comm );
    MPI_Bcast( &nprowSca, 1, MPI_INT, mpirankScaSta, comm );
    MPI_Bcast( &npcolSca, 1, MPI_INT, mpirankScaSta, comm );
    MPI_Bcast( &numRowBlock, 1, MPI_INT, mpirankScaSta, comm );
    MPI_Bcast( &numColBlock, 1, MPI_INT, mpirankScaSta, comm );

    // Define the data to processor mapping for ScaLAPACK matrix

    BlockMatPrtn  blockPrtn;
    {
      IntNumMat  procGrid( nprowSca, npcolSca );
      SetValue( procGrid, 0 );
      Int count = 0;
      // Note the column-major distribution here, and that mpirankScaVec must
      // be sorted in the correct order.  The communication
      // is later performed in comm instead of commSca
      for( Int jp = 0; jp < npcolSca; jp++ ){
        for( Int ip = 0; ip < nprowSca; ip++ ){
          procGrid( ip, jp ) = mpirankScaVec[count++];
        }
      } // for (ip)

      blockPrtn.ownerInfo.Resize( numRowBlock, numColBlock );
      IntNumMat&    blockOwner = blockPrtn.ownerInfo;
      // 2D block cyclic distribution
      for( Int jb = 0; jb < numColBlock; jb++ ){
        for( Int ib = 0; ib < numRowBlock; ib++ ){
          blockOwner( ib, jb ) = procGrid( ib % nprowSca, jb % npcolSca );
        }
      }
    }

    // Intermediate variable for constructing the distributed matrix in
    // ScaLAPACK format.  
    //
    // All processors participate in this step.
    DistVec<Index2, NumMat<F>, BlockMatPrtn> distScaMat;
    distScaMat.Prtn() = blockPrtn;
    // The communicator is the global communicator
    distScaMat.SetComm(comm);

    // Define the empty matrix
    DblNumMat empty( MB, MB ); 
    SetValue( empty, 0.0 );

#if ( _DEBUGlevel_ >= 1 )
    statusOFS << "blockPrtn.ownerInfo = " << blockPrtn.ownerInfo << std::endl;
#endif


    // Initialize the distScaMat for processors owning ScaLAPACK matrices
    if( isInSca ){
      for( Int jb = 0; jb < numColBlock; jb++ )
        for( Int ib = 0; ib < numRowBlock; ib++ ){
          Index2 key( ib, jb );
          if( distScaMat.Prtn().Owner( key ) == mpirank ){
            distScaMat.LocalMap()[key] = empty;
          }
        } // for (ib)
    }

    // Get the values of distScaMat for processors in commElem
    if( isInElem ){
      // Convert matrix distributed according to distElemMat to distScaMat
      for( typename std::map<ElemMatKey, NumMat<F> >::const_iterator 
          mi  = distMat.LocalMap().begin();
          mi != distMat.LocalMap().end(); ++mi ){
        ElemMatKey elemKey = (*mi).first;
        // Note the mpirank is given according to the commElem
        if( distMat.Prtn().Owner( elemKey ) == mpirankElem ){
          Index3 key1 = elemKey.first;
          Index3 key2 = elemKey.second;
          const std::vector<Int>& idx1 = basisIdx( key1(0), key1(1), key1(2) );
          const std::vector<Int>& idx2 = basisIdx( key2(0), key2(1), key2(2) );
          const NumMat<F>& localMat = (*mi).second;
          // Skip if there is no basis functions.
          if( localMat.Size() == 0 )
            continue;

          if( localMat.m() != idx1.size() ||
              localMat.n() != idx2.size() ){
            std::ostringstream msg;
            msg 
              << "Local matrix size is not consistent." << std::endl
              << "localMat   size : " << localMat.m() << " x " << localMat.n() << std::endl
              << "idx1       size : " << idx1.size() << std::endl
              << "idx2       size : " << idx2.size() << std::endl;
            ErrorHandling( msg.str().c_str() );
          }

          // Reshape the matrix element by element
          Int ib, jb, io, jo;
          for( Int b = 0; b < localMat.n(); b++ ){
            for( Int a = 0; a < localMat.m(); a++ ){
              ib = idx1[a] / MB;
              jb = idx2[b] / MB;
              io = idx1[a] % MB;
              jo = idx2[b] % MB;
              typename std::map<Index2, NumMat<F> >::iterator 
                ni = distScaMat.LocalMap().find( Index2(ib, jb) );
              // Contributes to blocks not owned by the current processor
              if( ni == distScaMat.LocalMap().end() ){
                distScaMat.LocalMap()[Index2(ib, jb)] = empty;
                ni = distScaMat.LocalMap().find( Index2(ib, jb) );
              }
              DblNumMat&  localScaMat = (*ni).second;
              localScaMat(io, jo) += localMat(a, b);
            } // for (a)
          } // for (b)


        } // own this matrix block
      } // for (mi)
    }

    // Communication of the matrix in the global communicator
    {
      // Prepare
      std::vector<Index2>  keyIdx;
      for( typename std::map<Index2, NumMat<F> >::iterator 
          mi  = distScaMat.LocalMap().begin();
          mi != distScaMat.LocalMap().end(); ++mi ){
        Index2 key = (*mi).first;
        if( distScaMat.Prtn().Owner( key ) != mpirank ){
          keyIdx.push_back( key );
        }
      } // for (mi)

      // Communication
      distScaMat.PutBegin( keyIdx, NO_MASK );
      distScaMat.PutEnd( NO_MASK, PutMode::COMBINE );

      // Clean to save space
      std::vector<Index2>  eraseKey;
      for( typename std::map<Index2, NumMat<F> >::iterator 
          mi  = distScaMat.LocalMap().begin();
          mi != distScaMat.LocalMap().end(); ++mi ){
        Index2 key = (*mi).first;
        if( distScaMat.Prtn().Owner( key ) != mpirank ){
          eraseKey.push_back( key );
        }
      } // for (mi)

      for( std::vector<Index2>::iterator vi = eraseKey.begin();
          vi != eraseKey.end(); ++vi ){
        distScaMat.LocalMap().erase( *vi );
      }    
    }

    // Copy the matrix values from distScaMat to scaMat. This is done for
    // processors in commSca
    if( isInSca )
    {
      for( typename std::map<Index2, NumMat<F> >::iterator 
          mi  = distScaMat.LocalMap().begin();
          mi != distScaMat.LocalMap().end(); ++mi ){
        Index2 key = (*mi).first;
        if( distScaMat.Prtn().Owner( key ) == mpirank ){
          Int ib = key(0), jb = key(1);
          Int offset = ( jb / npcolSca ) * MB * scaMat.LocalLDim() + 
            ( ib / nprowSca ) * MB;
          lapack::Lacpy( 'A', MB, MB, (*mi).second.Data(),
              MB, scaMat.Data() + offset, scaMat.LocalLDim() );
        } // own this block
      } // for (mi)
    }

    return;
  }  // -----  end of function DistElemMatToScaMat2  ----- 


/// @brief Convert a matrix distributed block-cyclicly as in ScaLAPACK 
/// back to a matrix distributed according to 1D element indices.
///
/// Note:
///
/// 1.  This routine is mainly used for converting the eigenvector
/// matrix from ScaLAPACK format back to the DG matrix format for
/// constructing density etc.
///
/// @param[in] numColKeep The first numColKeep columns are kept in each
/// matrix of distMat (i.e. distMat.LocalMap()[key].n() ) to save disk
/// space.  However, if numColKeep < 0 or omitted, *all* columns are kept in
/// distMat. 
///
template<typename F>
  void ScaMatToDistNumMat(
      const scalapack::ScaLAPACKMatrix<F>&           scaMat,
      const ElemPrtn&                                prtn,
      DistVec<Index3, NumMat<F>, ElemPrtn>&          distMat,
      const NumTns<std::vector<Int> >&               basisIdx,
      MPI_Comm                                         comm,
      Int                                            numColKeep = -1 ){
    using namespace dgdft::scalapack;

    Int mpirank, mpisize;
    MPI_Comm_rank( comm, &mpirank );
    MPI_Comm_size( comm, &mpisize );

    Descriptor desc = scaMat.Desc();
    distMat.Prtn()  = prtn;

    Index3 numElem;
    numElem[0] = prtn.ownerInfo.m();
    numElem[1] = prtn.ownerInfo.n(); 
    numElem[2] = prtn.ownerInfo.p(); 

    if( numColKeep < 0 )
      numColKeep = scaMat.Width();

    if( numColKeep > scaMat.Width() )
      ErrorHandling("NumColKeep cannot be bigger than the matrix width.");

    Int nprow  = desc.NpRow();
    Int npcol  = desc.NpCol();
    Int myprow = desc.MypRow();
    Int mypcol = desc.MypCol();

    Int MB = scaMat.MB(); 
    Int NB = scaMat.NB();

    if( MB != NB ){
      ErrorHandling("MB must be equal to NB.");
    }

    Int numRowBlock = scaMat.NumRowBlocks();
    Int numColBlock = scaMat.NumColBlocks();

    // Get the processor map
    IntNumMat  procGrid( nprow, npcol );
    SetValue( procGrid, 0 );
    {
      IntNumMat  procTmp( nprow, npcol );
      SetValue( procTmp, 0 );
      procTmp( myprow, mypcol ) = mpirank;
      mpi::Allreduce( procTmp.Data(), procGrid.Data(), nprow * npcol,
          MPI_SUM, comm );
    }


    BlockMatPrtn  blockPrtn;
    blockPrtn.ownerInfo.Resize( numRowBlock, numColBlock );
    IntNumMat&    blockOwner = blockPrtn.ownerInfo;
    for( Int jb = 0; jb < numColBlock; jb++ ){
      for( Int ib = 0; ib < numRowBlock; ib++ ){
        blockOwner( ib, jb ) = procGrid( ib % nprow, jb % npcol );
      }
    }

    // Intermediate variable for constructing the distributed matrix in
    // ScaLAPACK format.  
    DistVec<Index2, NumMat<F>, BlockMatPrtn> distScaMat;
    distScaMat.Prtn() = blockPrtn;
    distScaMat.SetComm(comm);

    // Initialize
    DblNumMat empty( MB, MB );
    SetValue( empty, 0.0 );
    for( Int jb = 0; jb < numColBlock; jb++ )
      for( Int ib = 0; ib < numRowBlock; ib++ ){
        Index2 key( ib, jb );
        if( distScaMat.Prtn().Owner( key ) == mpirank ){
          distScaMat.LocalMap()[key] = empty;
        }
      } // for (ib)

    // Convert ScaLAPACK matrix to distScaMat
    for( Int jb = 0; jb < numColBlock; jb++ )
      for( Int ib = 0; ib < numRowBlock; ib++ ){
        Index2 key( ib, jb );
        if( distScaMat.Prtn().Owner(key) == mpirank ){
          typename std::map<Index2, NumMat<F> >::iterator
            mi = distScaMat.LocalMap().find(key);
          Int offset = ( jb / npcol ) * MB * scaMat.LocalLDim() + 
            ( ib / nprow ) * MB;
          lapack::Lacpy( 'A', MB, MB, scaMat.Data() + offset,
              scaMat.LocalLDim(), (*mi).second.Data(), MB );
        } // own this block
      } // for (ib)


    // Communication of the matrix
    {
      // Prepare
      std::set<Index2>  keySet;
      for( Int k = 0; k < numElem[2]; k++ )
        for( Int j = 0; j < numElem[1]; j++ )
          for( Int i = 0; i < numElem[0]; i++ ){
            Index3 key( i, j, k );
            if( distMat.Prtn().Owner( key ) == mpirank ) {
              const std::vector<Int>&  idx = basisIdx(i, j, k);
              for( Int g = 0; g < idx.size(); g++ ){
                Int ib = idx[g] / MB;
                for( Int jb = 0; jb < numColBlock; jb++ )
                  keySet.insert( Index2( ib, jb ) );
              } // for (g)
            }
          } // for (i)

      std::vector<Index2>  keyIdx;
      keyIdx.insert( keyIdx.begin(), keySet.begin(), keySet.end() );

      // Actual communication
      distScaMat.GetBegin( keyIdx, NO_MASK );
      distScaMat.GetEnd( NO_MASK );
    }


    // Write back to distMat

    for( Int k = 0; k < numElem[2]; k++ )
      for( Int j = 0; j < numElem[1]; j++ )
        for( Int i = 0; i < numElem[0]; i++ ){
          Index3 key( i, j, k );
          if( distMat.Prtn().Owner( key ) == mpirank ) {
            const std::vector<Int>&  idx = basisIdx(i, j, k);
            NumMat<F>& localMat = distMat.LocalMap()[key];
            localMat.Resize( idx.size(), numColKeep );
            // Skip if there is no basis functions.
            if( localMat.Size() == 0 )
              continue;

            SetValue( localMat, (F)0 );
            for( Int g = 0; g < idx.size(); g++ ){
              Int ib = idx[g] / MB;
              Int io = idx[g] % MB;
              for( Int jb = 0; jb < numColBlock; jb++ ){
                NumMat<F>& localScaMat = 
                  distScaMat.LocalMap()[Index2(ib, jb)];
                for( Int jo = 0; jo < MB; jo++ ){
                  Int h = jb * MB + jo;
                  if( h < numColKeep )
                    localMat(g, h) = localScaMat(io, jo);
                }
              }
            } // for (g)
          } // own this element
        } // for (i)


    return;
  }  // -----  end of function ScaMatToDistNumMat  ----- 


/// @brief Convert a matrix distributed block-cyclicly as in ScaLAPACK
/// from one communicator back to a matrix distributed according to 1D
/// element indices on another communicator.
///
/// Note:
///
/// 1. There is no restriction on the communicators, i.e. in principle
/// the two communicators can be completely different.  This may not be
/// the most efficient way for implementing this feature, but should be
/// more general and maintainable.
///
/// 2.  This routine is mainly used for converting the eigenvector
/// matrix from ScaLAPACK format back to the DG matrix format for
/// constructing density etc.
///
/// 3. commSca is not needed since ScaLAPACK does not use MPI
/// communicators directly.
///
/// @param[in] numColKeep The first numColKeep columns are kept in each
/// matrix of distMat (i.e. distMat.LocalMap()[key].n() ) to save disk
/// space.  However, if numColKeep < 0 or omitted, *all* columns are kept in
/// distMat. 
///
template<typename F>
  void ScaMatToDistNumMat2(
      const scalapack::ScaLAPACKMatrix<F>&           scaMat,
      const ElemPrtn&                                prtn,
      DistVec<Index3, NumMat<F>, ElemPrtn>&          distMat,
      const NumTns<std::vector<Int> >&               basisIdx,
      MPI_Comm                                         comm,
      MPI_Comm                                         commElem,
      const std::vector<Int>&                        mpirankElemVec,
      const std::vector<Int>&                        mpirankScaVec,
      Int                                            numColKeep = -1 ){
    using namespace dgdft::scalapack;

    Int mpirank, mpisize;
    MPI_Comm_rank( comm, &mpirank );
    MPI_Comm_size( comm, &mpisize );

    bool isInElem, isInSca;
    isInElem = ( find( mpirankElemVec.begin(), 
          mpirankElemVec.end(), mpirank ) != mpirankElemVec.end() );
    isInSca  = ( find( mpirankScaVec.begin(), 
          mpirankScaVec.end(), mpirank ) != mpirankScaVec.end() );

    // commElem rank and size
    Int mpirankElem; 
    Int mpisizeElem = mpirankElemVec.size();
    if( isInElem ){
      Int tmp;
      MPI_Comm_rank( commElem, &mpirankElem );
      MPI_Comm_size( commElem, &tmp );

      if( mpisizeElem != tmp ){
        ErrorHandling("mpisizeElem read from input does not agree with commElem.");
      }
    }

    if( mpisizeElem > mpisize ){
      ErrorHandling("mpisizeElem cannot be larger than mpisize.");
    }

    // commSca size and other information
    Int mpisizeSca = mpirankScaVec.size();
    Int nprowSca;
    Int npcolSca;
    Int MB; 
    Int NB;
    Int numRowBlock;
    Int numColBlock;

    if( isInSca ){

      Descriptor desc = scaMat.Desc();
      nprowSca  = desc.NpRow();
      npcolSca  = desc.NpCol();

      if( nprowSca * npcolSca != mpisizeSca ){
        ErrorHandling("nprowSca * npcolSca != mpisizeSca.");
      }

      MB = scaMat.MB(); 
      NB = scaMat.NB();

      if( MB != NB ){
        ErrorHandling("MB must be equal to NB.");
      }

      numRowBlock = scaMat.NumRowBlocks();
      numColBlock = scaMat.NumColBlocks();
    }

    if( mpisizeSca > mpisize ){
      ErrorHandling("mpisizeSca cannot be larger than mpisize.");
    }

    // Make sure ALL processors in comm have some basic information
    // The root is mpirankScaSta
    Int mpirankScaSta = mpirankScaVec[0];
    MPI_Bcast( &MB, 1, MPI_INT, mpirankScaSta, comm );
    MPI_Bcast( &nprowSca, 1, MPI_INT, mpirankScaSta, comm );
    MPI_Bcast( &npcolSca, 1, MPI_INT, mpirankScaSta, comm );
    MPI_Bcast( &numRowBlock, 1, MPI_INT, mpirankScaSta, comm );
    MPI_Bcast( &numColBlock, 1, MPI_INT, mpirankScaSta, comm );


    // FIXME
    Index3 numElem;
    if( isInElem ){
      distMat.Prtn()  = prtn;

      numElem[0] = prtn.ownerInfo.m();
      numElem[1] = prtn.ownerInfo.n(); 
      numElem[2] = prtn.ownerInfo.p(); 
    }

#if ( _DEBUGlevel_ >= 1 )
    statusOFS << "numElem = " << numElem << std::endl;
#endif


    // Make sure ALL processors in comm have some basic information
    // The root is mpirankElemSta
    Int mpirankElemSta = mpirankElemVec[0];
    MPI_Bcast( &numElem[0], 3, MPI_INT, mpirankElemSta, comm );


    if( isInSca ){
      if( numColKeep < 0 )
        numColKeep = scaMat.Width();

      if( numColKeep > scaMat.Width() )
        ErrorHandling("NumColKeep cannot be bigger than the matrix width.");
    }
    // Communicate and make sure that numColKeep is globally shared
    MPI_Bcast( &numColKeep, 1, MPI_INT, mpirankScaSta, comm );


    // Define the data to processor mapping for ScaLAPACK matrix
    BlockMatPrtn  blockPrtn;
    {
      IntNumMat  procGrid( nprowSca, npcolSca );
      SetValue( procGrid, 0 );
      Int count = 0;
      // Note the column-major distribution here, and that mpirankScaVec must
      // be sorted in the correct order.  The communication
      // is later performed in comm instead of commSca
      for( Int jp = 0; jp < npcolSca; jp++ ){
        for( Int ip = 0; ip < nprowSca; ip++ ){
          procGrid( ip, jp ) = mpirankScaVec[count++];
        }
      } // for (ip)

      blockPrtn.ownerInfo.Resize( numRowBlock, numColBlock );
      IntNumMat&    blockOwner = blockPrtn.ownerInfo;
      // 2D block cyclic distribution
      for( Int jb = 0; jb < numColBlock; jb++ ){
        for( Int ib = 0; ib < numRowBlock; ib++ ){
          blockOwner( ib, jb ) = procGrid( ib % nprowSca, jb % npcolSca );
        }
      }
    }

    // Intermediate variable for constructing the distributed matrix in
    // ScaLAPACK format.  
    // 
    // All processors participate in this step.
    DistVec<Index2, NumMat<F>, BlockMatPrtn> distScaMat;
    distScaMat.Prtn() = blockPrtn;
    distScaMat.SetComm(comm);

    // Define the empty matrix
    DblNumMat empty( MB, MB ); 
    SetValue( empty, 0.0 );

#if ( _DEBUGlevel_ >= 1 )
    statusOFS << "blockPrtn.ownerInfo = " << blockPrtn.ownerInfo << std::endl;
#endif

    // Convert ScaLAPACK matrix to distScaMat for processors owning
    // ScaLAPACK matrices
    if( isInSca ){
      for( Int jb = 0; jb < numColBlock; jb++ )
        for( Int ib = 0; ib < numRowBlock; ib++ ){
          Index2 key( ib, jb );
          if( distScaMat.Prtn().Owner( key ) == mpirank ){
            // Initialize
            distScaMat.LocalMap()[key] = empty;

            typename std::map<Index2, NumMat<F> >::iterator
              mi = distScaMat.LocalMap().find(key);
            Int offset = ( jb / npcolSca ) * MB * scaMat.LocalLDim() + 
              ( ib / nprowSca ) * MB;
            lapack::Lacpy( 'A', MB, MB, scaMat.Data() + offset,
                scaMat.LocalLDim(), (*mi).second.Data(), MB );

          }
        } // for (ib)
    }


    // Communication of the matrix in the global communicator
    {
      // Prepare
      std::set<Index2>  keySet;
      if( isInElem ){
        for( Int k = 0; k < numElem[2]; k++ )
          for( Int j = 0; j < numElem[1]; j++ )
            for( Int i = 0; i < numElem[0]; i++ ){
              Index3 key( i, j, k );
              // Note the mpirank is given according to the commElem
              if( distMat.Prtn().Owner( key ) == mpirankElem ) {
                const std::vector<Int>&  idx = basisIdx(i, j, k);
                for( Int g = 0; g < idx.size(); g++ ){
                  Int ib = idx[g] / MB;
                  for( Int jb = 0; jb < numColBlock; jb++ )
                    keySet.insert( Index2( ib, jb ) );
                } // for (g)
              }
            } // for (i)
      }

      std::vector<Index2>  keyIdx;
      keyIdx.insert( keyIdx.begin(), keySet.begin(), keySet.end() );

      // Actual communication
      distScaMat.GetBegin( keyIdx, NO_MASK );
      distScaMat.GetEnd( NO_MASK );
    }


    // Write back to distMat

    if( isInElem ){
      for( Int k = 0; k < numElem[2]; k++ )
        for( Int j = 0; j < numElem[1]; j++ )
          for( Int i = 0; i < numElem[0]; i++ ){
            Index3 key( i, j, k );
            if( distMat.Prtn().Owner( key ) == mpirankElem ) {
              const std::vector<Int>&  idx = basisIdx(i, j, k);
              NumMat<F>& localMat = distMat.LocalMap()[key];
              localMat.Resize( idx.size(), numColKeep );
              // Skip if there is no basis functions.
              if( localMat.Size() == 0 )
                continue;

              SetValue( localMat, (F)0 );
              for( Int g = 0; g < idx.size(); g++ ){
                Int ib = idx[g] / MB;
                Int io = idx[g] % MB;
                for( Int jb = 0; jb < numColBlock; jb++ ){
                  NumMat<F>& localScaMat = 
                    distScaMat.LocalMap()[Index2(ib, jb)];
                  for( Int jo = 0; jo < MB; jo++ ){
                    Int h = jb * MB + jo;
                    if( h < numColKeep )
                      localMat(g, h) = localScaMat(io, jo);
                  }
                }
              } // for (g)
            } // own this element
          } // for (i)
    } // if( isInElem )

    return;
  }  // -----  end of function ScaMatToDistNumMat2  ----- 



/// @brief Convert a matrix distributed according to 2D element indices
/// to a DistSparseMatrix.
///
/// Note: 
///
/// 1) This routine assumes that the input matrix (DistElemMat) and the
/// output matrix (DistSparseMat) shares the same MPI communicator.
/// This may not be always the case.  For instance, in PEXSI we might
/// want to use a smaller communicator for distSparseMat so that
/// parallelization can be performed over the number of poles.  This
/// will be done in the future work.
///
/// 2) It is assumed that the matrix is symmetric.  Therefore the
/// position of row and column may be confusing.  This will be a
/// problem later when structurally symmetric matrix is considered.
/// This shall be fixed later.
template<typename F>
  void DistElemMatToDistSparseMat(
      const DistVec<ElemMatKey, NumMat<F>, ElemMatPrtn>&   distMat,
      const Int                                            sizeMat,
      DistSparseMatrix<F>&                                 distSparseMat,
      const NumTns<std::vector<Int> >&                     basisIdx,
      MPI_Comm                                               comm ){
    Int mpirank, mpisize;
    MPI_Comm_rank( comm, &mpirank );
    MPI_Comm_size( comm, &mpisize );

    // Compute the local column partition in DistSparseMatrix
    Int numColFirst = sizeMat / mpisize;
    Int firstCol = mpirank * numColFirst;
    Int numColLocal;
    if( mpirank == mpisize - 1 )
      numColLocal = sizeMat - numColFirst * (mpisize - 1 );
    else
      numColLocal = numColFirst;

    VecPrtn  vecPrtn;
    vecPrtn.ownerInfo.Resize( sizeMat );
    IntNumVec&  vecOwner = vecPrtn.ownerInfo;
    for( Int i = 0; i < sizeMat; i++ ){
      vecOwner(i) = std::min( i / numColFirst, mpisize - 1 );
    }

    // nonzero row indices distributedly saved according to column partitions.
    DistVec<Int, std::vector<Int>, VecPrtn>   distRow;
    // nonzero values distributedly saved according to column partitions.
    DistVec<Int, std::vector<F>,   VecPrtn>   distVal;

    distRow.Prtn() = vecPrtn;
    distVal.Prtn() = vecPrtn;
    distRow.SetComm( comm );
    distVal.SetComm( comm );

    std::set<Int>  ownerSet;

    // Convert DistElemMat to intermediate data structure
    for( typename std::map<ElemMatKey, NumMat<F> >::const_iterator 
        mi  = distMat.LocalMap().begin();
        mi != distMat.LocalMap().end(); ++mi ){
      ElemMatKey elemKey = (*mi).first;
      if( distMat.Prtn().Owner( elemKey ) == mpirank ){
        Index3 key1 = elemKey.first;
        Index3 key2 = elemKey.second;
        const std::vector<Int>& idx1 = basisIdx( key1(0), key1(1), key1(2) );
        const std::vector<Int>& idx2 = basisIdx( key2(0), key2(1), key2(2) );
        const NumMat<F>& localMat = (*mi).second;
        // Skip if there is no basis functions.
        if( localMat.Size() == 0 )
          continue;

        if( localMat.m() != idx1.size() ||
            localMat.n() != idx2.size() ){
          std::ostringstream msg;
          msg 
            << "Local matrix size is not consistent." << std::endl
            << "localMat   size : " << localMat.m() << " x " << localMat.n() << std::endl
            << "idx1       size : " << idx1.size() << std::endl
            << "idx2       size : " << idx2.size() << std::endl;
          ErrorHandling( msg.str().c_str() );
        }

        // Distribute the matrix element and row indices to intermediate
        // data structure in an element by element fashion.

        for( Int a = 0; a < localMat.m(); a++ ){
          Int row = idx1[a];
          ownerSet.insert( row );
          std::vector<Int>& vecRow = distRow.LocalMap()[row];
          std::vector<F>& vecVal = distVal.LocalMap()[row];
          for( Int b = 0; b < localMat.n(); b++ ){
            vecRow.push_back( idx2[b] + 1 );
            vecVal.push_back( localMat( a, b ) );
          }
        } // for (a)
      } // own this matrix block
    } // for (mi)

#if ( _DEBUGlevel_ >= 2 )
    if( mpirank == 0 ){
      for( typename std::map<Int, std::vector<Int> >::const_iterator 
          mi  = distRow.LocalMap().begin();
          mi != distRow.LocalMap().end(); ++mi ){
        std::cout << "Row " << (*mi).first << std::endl << (*mi).second << std::endl;
      }
      for( typename std::map<Int, std::vector<F> >::const_iterator 
          mi  = distVal.LocalMap().begin();
          mi != distVal.LocalMap().end(); ++mi ){
        std::cout << "Row " << (*mi).first << std::endl << (*mi).second << std::endl;
      }
    }
#endif


    // Communication of the matrix
    {

      std::vector<Int>  keyIdx;
      keyIdx.insert( keyIdx.end(), ownerSet.begin(), ownerSet.end() );
      //        std::cout << keyIdx << std::endl;
      //        std::cout << vecOwner << std::endl;

      distRow.PutBegin( keyIdx, NO_MASK );
      distVal.PutBegin( keyIdx, NO_MASK );

      distRow.PutEnd( NO_MASK, PutMode::REPLACE );
      distVal.PutEnd( NO_MASK, PutMode::REPLACE );

      // Clean to save space
      std::vector<Int>  eraseKey;
      for( typename std::map<Int, std::vector<Int> >::iterator 
          mi  = distRow.LocalMap().begin();
          mi != distRow.LocalMap().end(); ++mi ){
        Int key = (*mi).first;
        if( distRow.Prtn().Owner( key ) != mpirank ){
          eraseKey.push_back( key );
        }
      } // for (mi)

      for( std::vector<Int>::iterator vi = eraseKey.begin();
          vi != eraseKey.end(); ++vi){
        distRow.LocalMap().erase( *vi );
        distVal.LocalMap().erase( *vi );
      }    
    }

    // Copy the values from intermediate data structure to distSparseMat
    {
      // Global information
      distSparseMat.size        = sizeMat;
      // Note the 1-based convention
      distSparseMat.firstCol    = firstCol + 1;
      distSparseMat.comm        = comm;

      // Local information
      IntNumVec& colptrLocal = distSparseMat.colptrLocal;
      IntNumVec& rowindLocal = distSparseMat.rowindLocal;
      NumVec<F>& nzvalLocal  = distSparseMat.nzvalLocal;

      Int nnzLocal = 0;
      Int nnz = 0;
      colptrLocal.Resize( numColLocal + 1 );
      colptrLocal[0] = 1;

      for( Int row = firstCol; row < firstCol + numColLocal; row++ ){
        if( distRow.Prtn().Owner( row ) != mpirank ){
          ErrorHandling( "The owner information in distRow is incorrect.");
        }
        std::vector<Int>&  rowVec = distRow.LocalMap()[row];
        nnzLocal += rowVec.size();
        Int cur = row - firstCol;
        colptrLocal[cur+1] = colptrLocal[cur] + rowVec.size();
      }    

      mpi::Allreduce( &nnzLocal, &nnz, 1, MPI_SUM, comm );

      rowindLocal.Resize( nnzLocal );
      nzvalLocal.Resize( nnzLocal );

      for( Int row = firstCol; row < firstCol + numColLocal; row++ ){
        if( distRow.Prtn().Owner( row ) != mpirank ){
          ErrorHandling( "The owner information in distRow is incorrect.");
        }
        std::vector<Int>&  rowVec = distRow.LocalMap()[row];
        std::vector<F>&    valVec = distVal.LocalMap()[row];
        Int cur = row - firstCol;
        std::copy( rowVec.begin(), rowVec.end(), &rowindLocal[colptrLocal[cur]-1] );
        std::copy( valVec.begin(), valVec.end(), &nzvalLocal[colptrLocal[cur]-1] );
      }    


      // Other global and local information
      distSparseMat.nnzLocal = nnzLocal;
      distSparseMat.nnz      = nnz;

    }


    return;
  }  // -----  end of function DistElemMatToDistSparseMat  ----- 

/// @brief Convert a matrix distributed according to 2D element indices
/// from one communicator to a DistSparseMatrix distributed on another
/// communicator.
///
/// LL: 11/26/2014: This function is not used because it is not scalable
/// for a large number of processors
///
/// Note: 
///
/// 1. There is no restriction on the communicators, i.e. in principle
/// the two communicators can be completely different.  This may not be
/// the most efficient way for implementing this feature, but should be
/// more general and maintainable.
///
/// 2. It is assumed that the matrix is symmetric.  Therefore the
/// position of row and column may be confusing.  This will be a
/// problem later when structurally symmetric matrix is considered.
/// This shall be fixed later.
///
///
/// @param[in] distMat (local) Matrix distributed according to 2D element
/// indices.  
/// @param[in] sizeMat (global) Size of the matrix. Owned by all
/// processors in comm.
/// @param[out] distSparseMat (local) Matrix distibuted according to the
/// DistSparseMatrix format.  
/// @param[in] basisIdx (global) Indices for all basis functions.  The input from
/// each processor column of comm1 should be exactly the same.
/// @param[in] comm (global) Global communicator. 
/// @param[in] commElem (local) Communicator for distElemMat. In the current step
/// with intra-element parallelization, the comm for distElemMat can be
/// colComm.
/// @param[in] commSparse (local) Communicator for distSparseMat. For
/// PEXSI, this is the communicator for a single pole
/// @param[in] mpirankElemVec (global) mpiranks in the global
/// communicator comm for processors in commElem.  This vector should
/// follow a non-decreasing order.
/// @param[in] mpirankSparseVec (global) mpiranks in the global
/// communicator comm for processors saving the matrix in
/// DistSparseMatrix format.
template<typename F>
  void DistElemMatToDistSparseMat2(
      const DistVec<ElemMatKey, NumMat<F>, ElemMatPrtn>&   distMat,
      const Int                                            sizeMat,
      DistSparseMatrix<F>&                                 distSparseMat,
      const NumTns<std::vector<Int> >&                     basisIdx,
      MPI_Comm                                               comm,
      MPI_Comm                                             commElem,
      MPI_Comm                                             commSparse,
      const std::vector<Int>&                              mpirankElemVec,
      const std::vector<Int>&                              mpirankSparseVec){
    Int mpirank, mpisize;
    MPI_Comm_rank( comm, &mpirank );
    MPI_Comm_size( comm, &mpisize );

    Real timeSta, timeEnd;

    bool isInElem, isInSparse;
    isInElem = ( find( mpirankElemVec.begin(), 
          mpirankElemVec.end(), mpirank ) != mpirankElemVec.end() );
    isInSparse = ( find( mpirankSparseVec.begin(), 
          mpirankSparseVec.end(), mpirank ) != mpirankSparseVec.end() );

    // commElem rank and size
    Int mpirankElem; 
    Int mpisizeElem = mpirankElemVec.size();
    if( isInElem ){
      Int tmp;
      MPI_Comm_rank( commElem, &mpirankElem );
      MPI_Comm_size( commElem, &tmp );

      if( mpisizeElem != tmp ){
        ErrorHandling("mpisizeElem read from input does not agree with commElem.");
      }
    }

    if( mpisizeElem > mpisize ){
      ErrorHandling("mpisizeElem cannot be larger than mpisize.");
    }

    // commSparse size and other information
    Int mpirankSparse; 
    Int mpisizeSparse = mpirankSparseVec.size();
    Int numColLocal;
    if( isInSparse ){
      Int tmp;
      MPI_Comm_rank( commSparse, &mpirankSparse );
      MPI_Comm_size( commSparse, &tmp );

      if( mpisizeSparse != tmp ){
        ErrorHandling("mpisizeSparse read from input does not agree with commSparse.");
      }
    }

    if( mpisizeSparse > mpisize ){
      ErrorHandling("mpisizeSparse cannot be larger than mpisize.");
    }

    // Compute the local column partition in DistSparseMatrix
    Int numColFirst = sizeMat / mpisizeSparse;

    VecPrtn  vecPrtn;
    vecPrtn.ownerInfo.Resize( sizeMat );
    IntNumVec&  vecOwner = vecPrtn.ownerInfo;
    for( Int i = 0; i < sizeMat; i++ ){
      vecOwner(i) = mpirankSparseVec[std::min( i / numColFirst, mpisizeSparse - 1 )];
    }

    // nonzero row indices distributedly saved according to column partitions.
    DistVec<Int, std::vector<Int>, VecPrtn>   distRow;
    // nonzero values distributedly saved according to column partitions.
    DistVec<Int, std::vector<F>,   VecPrtn>   distVal;

    distRow.Prtn() = vecPrtn;
    distVal.Prtn() = vecPrtn;
    distRow.SetComm( comm );
    distVal.SetComm( comm );

    std::set<Int>  ownerSet;

    // Convert DistElemMat to intermediate data structure
    GetTime( timeSta );
    if( isInElem ){
      for( typename std::map<ElemMatKey, NumMat<F> >::const_iterator 
          mi  = distMat.LocalMap().begin();
//          mi != distMat.LocalMap().end(); mi++ ){
          mi != distMat.LocalMap().end(); ++mi ){
        ElemMatKey elemKey = (*mi).first;
        if( distMat.Prtn().Owner( elemKey ) == mpirankElem ){
          Index3 key1 = elemKey.first;
          Index3 key2 = elemKey.second;
          const std::vector<Int>& idx1 = basisIdx( key1(0), key1(1), key1(2) );
          const std::vector<Int>& idx2 = basisIdx( key2(0), key2(1), key2(2) );
          const NumMat<F>& localMat = (*mi).second;
          // Skip if there is no basis functions.
          if( localMat.Size() == 0 )
            continue;

          if( localMat.m() != idx1.size() ||
              localMat.n() != idx2.size() ){
            std::ostringstream msg;
            msg 
              << "Local matrix size is not consistent." << std::endl
              << "localMat   size : " << localMat.m() << " x " << localMat.n() << std::endl
              << "idx1       size : " << idx1.size() << std::endl
              << "idx2       size : " << idx2.size() << std::endl;
            ErrorHandling( msg.str().c_str() );
          }

          // Distribute the matrix element and row indices to intermediate
          // data structure in an element by element fashion.

          std::vector<Int> rowTmp(localMat.n());
          // NOTE the +1 is used here since DistSparseMat uses 1-based
          // index for legacy reason.
          for( Int b = 0; b < localMat.n(); b++ ){
            rowTmp[b] = idx2[b]+1;
          }
          std::vector<F>   valTmp(localMat.n());

          for( Int a = 0; a < localMat.m(); a++ ){
            // NOTE that the row here is only for the intermediate data
            // structure and therefore do not need the 1-based index.
            Int row = idx1[a];
            ownerSet.insert( row );
            std::vector<Int>& vecRow = distRow.LocalMap()[row];
            vecRow.insert(vecRow.end(), rowTmp.begin(), rowTmp.end());

            std::vector<F>& vecVal = distVal.LocalMap()[row];
            lapack::Lacpy( 'A', 1, localMat.n(), &localMat(a,0), localMat.m(),
                &valTmp[0], 1 );
            vecVal.insert( vecVal.end(), valTmp.begin(), valTmp.end() );
          } // for (a)
        } // own this matrix block
      } // for (mi)
    }
    GetTime(timeEnd);
#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for compressing the information in DistElemMat is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif


    // Communication of the matrix in the global communicator
    GetTime(timeSta);
    {

      std::vector<Int>  keyIdx;
      keyIdx.insert( keyIdx.end(), ownerSet.begin(), ownerSet.end() );

      distRow.PutBegin( keyIdx, NO_MASK );
      distVal.PutBegin( keyIdx, NO_MASK );

      distRow.PutEnd( NO_MASK, PutMode::REPLACE );
      distVal.PutEnd( NO_MASK, PutMode::REPLACE );

      // Clean to save space
      std::vector<Int>  eraseKey;
      for( typename std::map<Int, std::vector<Int> >::iterator 
          mi  = distRow.LocalMap().begin();
//          mi != distRow.LocalMap().end(); mi++ ){
          mi != distRow.LocalMap().end(); ++mi ){
        Int key = (*mi).first;
        if( distRow.Prtn().Owner( key ) != mpirank ){
          eraseKey.push_back( key );
        }
      } // for (mi)

      for( std::vector<Int>::iterator vi = eraseKey.begin();
//          vi != eraseKey.end(); vi++ ){
          vi != eraseKey.end(); ++vi ){
        distRow.LocalMap().erase( *vi );
        distVal.LocalMap().erase( *vi );
      }    
    }
    GetTime(timeEnd);
#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for communication of the information of DistElemMat is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif


    // Copy the values from intermediate data structure to distSparseMat
    GetTime(timeSta);
    if( isInSparse )
    {
      // Global information
      distSparseMat.size        = sizeMat;
      // Note the 1-based convention
      Int firstCol = mpirankSparse * numColFirst;
      distSparseMat.firstCol    = firstCol + 1;
      distSparseMat.comm        = commSparse;

      // Local information
      IntNumVec& colptrLocal = distSparseMat.colptrLocal;
      IntNumVec& rowindLocal = distSparseMat.rowindLocal;
      NumVec<F>& nzvalLocal  = distSparseMat.nzvalLocal;

      Int numColLocal;
      if( mpirankSparse == mpisizeSparse - 1 ) 
        numColLocal = sizeMat - numColFirst * (mpisizeSparse - 1 );
      else
        numColLocal = numColFirst;

      Int nnzLocal = 0;
      Int nnz = 0;
      colptrLocal.Resize( numColLocal + 1 );
      // 1-based index here
      colptrLocal[0] = 1;

      for( Int row = firstCol; row < firstCol + numColLocal; row++ ){
        if( distRow.Prtn().Owner( row ) != mpirank ){
          ErrorHandling( "The owner information in distRow is incorrect.");
        }
        std::vector<Int>&  rowVec = distRow.LocalMap()[row];
        nnzLocal += rowVec.size();
        Int cur = row - firstCol;
        colptrLocal[cur+1] = colptrLocal[cur] + rowVec.size();
      }    

      mpi::Allreduce( &nnzLocal, &nnz, 1, MPI_SUM, commSparse );

      rowindLocal.Resize( nnzLocal );
      nzvalLocal.Resize( nnzLocal );

      for( Int row = firstCol; row < firstCol + numColLocal; row++ ){
        if( distRow.Prtn().Owner( row ) != mpirank ){
          ErrorHandling( "The owner information in distRow is incorrect.");
        }
        std::vector<Int>&  rowVec = distRow.LocalMap()[row];
        std::vector<F>&    valVec = distVal.LocalMap()[row];
        Int cur = row - firstCol;
        std::copy( rowVec.begin(), rowVec.end(), &rowindLocal[colptrLocal[cur]-1] );
        std::copy( valVec.begin(), valVec.end(), &nzvalLocal[colptrLocal[cur]-1] );
      }    


      // Other global and local information
      distSparseMat.nnzLocal = nnzLocal;
      distSparseMat.nnz      = nnz;
    } // if (isInSparse)
    GetTime(timeEnd);
#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for writing the matrix to distSparseMat is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif


    return;
  }  // -----  end of function DistElemMatToDistSparseMat2  ----- 


/// @brief Convert a matrix distributed according to 2D element indices
/// from one communicator to a DistSparseMatrix distributed on another
/// communicator.
///
/// Note: 
/// 
/// 1. Compared to DistElemMatToDistSparseMat2, there is restriction of
/// the communicator that the processors for saving the DistSparseMat
/// must be a subset of the processors for saving the matrix in
/// DistElemMat
///
/// 2. It is assumed that the matrix is symmetric.  Therefore the
/// position of row and column may be confusing.  This will be a
/// problem later when structurally symmetric matrix is considered.
/// This shall be fixed later.
///
/// @param[in] distMat (local) Matrix distributed according to 2D element
/// indices.  
/// @param[in] sizeMat (global) Size of the matrix. Owned by all
/// processors in comm.
/// @param[out] distSparseMat (local) Matrix distibuted according to the
/// DistSparseMatrix format.  
/// @param[in] basisIdx (global) Indices for all basis functions.  The input from
/// each processor column of comm1 should be exactly the same.
/// @param[in] commElem (global) Communicator for distElemMat. In the current step
/// with intra-element parallelization, the communicator for distElemMat can be
/// colComm.
/// @param[in] mpirankSparseVec (global) mpiranks in the local commElem
/// communicator for processors saving the matrix in DistSparseMatrix
/// format.
template<typename F>
  void DistElemMatToDistSparseMat3(
      const DistVec<ElemMatKey, NumMat<F>, ElemMatPrtn>&   distMat,
      const Int                                            sizeMat,
      DistSparseMatrix<F>&                                 distSparseMat,
      const NumTns<std::vector<Int> >&                     basisIdx,
      MPI_Comm                                             commElem,
      const std::vector<Int>&                              mpirankSparseVec){
    Real timeSta, timeEnd;

    // commElem rank and size
    Int mpirankElem; 
    Int mpisizeElem;
    MPI_Comm_rank( commElem, &mpirankElem );
    MPI_Comm_size( commElem, &mpisizeElem );

    bool isInSparse;
    // Note that mpirank refers to the rank in the local communicator
    isInSparse = ( find( mpirankSparseVec.begin(), 
          mpirankSparseVec.end(), mpirankElem ) != mpirankSparseVec.end() );

    Int mpisizeSparse = mpirankSparseVec.size();

    if( mpisizeSparse > mpisizeElem ){
      ErrorHandling("The number of processors for DistSparseMatrix cannot exceed that of DistElemMat.");
    }

    Int mpirankSparse = -1;
    for( Int i = 0; i < mpisizeSparse; i++ ){
      if( mpirankElem == mpirankSparseVec[i] )
        mpirankSparse = mpirankElem;
    }


    // Compute the local column partition in DistSparseMatrix
    Int numColFirst = sizeMat / mpisizeSparse;

    VecPrtn  vecPrtn;
    vecPrtn.ownerInfo.Resize( sizeMat );
    IntNumVec&  vecOwner = vecPrtn.ownerInfo;
    for( Int i = 0; i < sizeMat; i++ ){
      vecOwner(i) = mpirankSparseVec[std::min( i / numColFirst, mpisizeSparse - 1 )];
    }

    // nonzero row indices distributedly saved according to column partitions.
    DistVec<Int, std::vector<Int>, VecPrtn>   distRow;
    // nonzero values distributedly saved according to column partitions.
    DistVec<Int, std::vector<F>,   VecPrtn>   distVal;

    distRow.Prtn() = vecPrtn;
    distVal.Prtn() = vecPrtn;
    distRow.SetComm( commElem );
    distVal.SetComm( commElem );

    std::set<Int>  ownerSet;

    // Convert DistElemMat to intermediate data structure
    GetTime( timeSta );
    for( typename std::map<ElemMatKey, NumMat<F> >::const_iterator 
        mi  = distMat.LocalMap().begin();
//        mi != distMat.LocalMap().end(); mi++ ){
        mi != distMat.LocalMap().end(); ++mi ){
      ElemMatKey elemKey = (*mi).first;
      if( distMat.Prtn().Owner( elemKey ) == mpirankElem ){
        Index3 key1 = elemKey.first;
        Index3 key2 = elemKey.second;
        const std::vector<Int>& idx1 = basisIdx( key1(0), key1(1), key1(2) );
        const std::vector<Int>& idx2 = basisIdx( key2(0), key2(1), key2(2) );
        const NumMat<F>& localMat = (*mi).second;
        // Skip if there is no basis functions.
        if( localMat.Size() == 0 )
          continue;

        if( localMat.m() != idx1.size() ||
            localMat.n() != idx2.size() ){
          std::ostringstream msg;
          msg 
            << "Local matrix size is not consistent." << std::endl
            << "localMat   size : " << localMat.m() << " x " << localMat.n() << std::endl
            << "idx1       size : " << idx1.size() << std::endl
            << "idx2       size : " << idx2.size() << std::endl;
          ErrorHandling( msg.str().c_str() );
        }

        // Distribute the matrix element and row indices to intermediate
        // data structure in an element by element fashion.

        std::vector<Int> rowTmp(localMat.n());
        // NOTE the +1 is used here since DistSparseMat uses 1-based
        // index for legacy reason.
        for( Int b = 0; b < localMat.n(); b++ ){
          rowTmp[b] = idx2[b]+1;
        }
        std::vector<F>   valTmp(localMat.n());

        for( Int a = 0; a < localMat.m(); a++ ){
          // NOTE that the row here is only for the intermediate data
          // structure and therefore do not need the 1-based index.
          Int row = idx1[a];
          ownerSet.insert( row );
          std::vector<Int>& vecRow = distRow.LocalMap()[row];
          vecRow.insert(vecRow.end(), rowTmp.begin(), rowTmp.end());

          std::vector<F>& vecVal = distVal.LocalMap()[row];
          lapack::Lacpy( 'A', 1, localMat.n(), &localMat(a,0), localMat.m(),
              &valTmp[0], 1 );
          vecVal.insert( vecVal.end(), valTmp.begin(), valTmp.end() );
        } // for (a)
      } // own this matrix block
    } // for (mi)
    GetTime(timeEnd);
#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for compressing the information in DistElemMat is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif


    // Communication of the matrix in the global communicator
    GetTime(timeSta);
    {

      std::vector<Int>  keyIdx;
      keyIdx.insert( keyIdx.end(), ownerSet.begin(), ownerSet.end() );

      distRow.PutBegin( keyIdx, NO_MASK );
      distVal.PutBegin( keyIdx, NO_MASK );

      distRow.PutEnd( NO_MASK, PutMode::REPLACE );
      distVal.PutEnd( NO_MASK, PutMode::REPLACE );

      // Clean to save space
      std::vector<Int>  eraseKey;
      for( typename std::map<Int, std::vector<Int> >::iterator 
          mi  = distRow.LocalMap().begin();
//          mi != distRow.LocalMap().end(); mi++ ){
          mi != distRow.LocalMap().end(); ++mi ){
        Int key = (*mi).first;
        if( distRow.Prtn().Owner( key ) != mpirankElem ){
          eraseKey.push_back( key );
        }
      } // for (mi)

      for( std::vector<Int>::iterator vi = eraseKey.begin();
//          vi != eraseKey.end(); vi++ ){
          vi != eraseKey.end(); ++vi ){
        distRow.LocalMap().erase( *vi );
        distVal.LocalMap().erase( *vi );
      }    
    }
    GetTime(timeEnd);
#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for communication of the information of DistElemMat is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif


    // Copy the values from intermediate data structure to distSparseMat
    GetTime(timeSta);

    // Important to have nnzLocal = 0 for all procs
    Int nnzLocal = 0;
    if( isInSparse ) {
      // Global information
      distSparseMat.size        = sizeMat;
      // Note the 1-based convention
      Int firstCol = mpirankSparse * numColFirst;
      distSparseMat.firstCol    = firstCol + 1;
      // No communicator is set
      //        distSparseMat.comm        = commSparse;

      // Local information
      IntNumVec& colptrLocal = distSparseMat.colptrLocal;
      IntNumVec& rowindLocal = distSparseMat.rowindLocal;
      NumVec<F>& nzvalLocal  = distSparseMat.nzvalLocal;

      Int numColLocal;
      if( mpirankSparse == mpisizeSparse - 1 )
        numColLocal = sizeMat - numColFirst * (mpisizeSparse - 1 );
      else
        numColLocal = numColFirst;

      Int nnz = 0;
      colptrLocal.Resize( numColLocal + 1 );
      // 1-based index here
      colptrLocal[0] = 1;

      for( Int row = firstCol; row < firstCol + numColLocal; row++ ){
        if( distRow.Prtn().Owner( row ) != mpirankElem ){
          ErrorHandling( "The owner information in distRow is incorrect.");
        }
        std::vector<Int>&  rowVec = distRow.LocalMap()[row];
        nnzLocal += rowVec.size();
        Int cur = row - firstCol;
        colptrLocal[cur+1] = colptrLocal[cur] + rowVec.size();
      }    

      rowindLocal.Resize( nnzLocal );
      nzvalLocal.Resize( nnzLocal );

      for( Int row = firstCol; row < firstCol + numColLocal; row++ ){
        if( distRow.Prtn().Owner( row ) != mpirankElem ){
          ErrorHandling( "The owner information in distRow is incorrect.");
        }
        std::vector<Int>&  rowVec = distRow.LocalMap()[row];
        std::vector<F>&    valVec = distVal.LocalMap()[row];
        Int cur = row - firstCol;
        std::copy( rowVec.begin(), rowVec.end(), &rowindLocal[colptrLocal[cur]-1] );
        std::copy( valVec.begin(), valVec.end(), &nzvalLocal[colptrLocal[cur]-1] );
      }    
    } // if (isInSparse)

    // Global information
    Int nnz;
    mpi::Allreduce( &nnzLocal, &nnz, 1, MPI_SUM, commElem );

    if( isInSparse ){
      distSparseMat.nnzLocal = nnzLocal;
      distSparseMat.nnz      = nnz;
    }


    GetTime(timeEnd);
#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for writing the matrix to distSparseMat is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif


    return;
  }  // -----  end of function DistElemMatToDistSparseMat3  ----- 



/// @brief Convert a matrix distributed according to 2D element indices
/// to a DistSparseMatrix distributed on the first mpisizeDistSparse processors.
///
/// Note: 
///
/// 1) The communicator corresponding to first mpisizeDistSparse is
/// **not** given. Therefore the communicatior for distSparseMat must be
/// given **outside** this routine.  Similarly the nnz property is not
/// computed.  This should also be done **outside** this routine.
///
/// 2) It is assumed that the matrix is symmetric.  Therefore the
/// position of row and column may be confusing.  This will be a
/// problem later when structurally symmetric matrix is considered.
/// This shall be fixed later.
template<typename F>
  void DistElemMatToDistSparseMat(
      const DistVec<ElemMatKey, NumMat<F>, ElemMatPrtn>&   distMat,
      const Int                                            sizeMat,
      DistSparseMatrix<F>&                                 distSparseMat,
      const NumTns<std::vector<Int> >&                     basisIdx,
      MPI_Comm                                               comm,
      Int                                                  mpisizeDistSparse ){
    Int mpirank, mpisize;
    MPI_Comm_rank( comm, &mpirank );
    MPI_Comm_size( comm, &mpisize );

    if( mpisizeDistSparse > mpisize ){
      std::ostringstream msg;
      msg << std::endl
        << "mpisize = " << mpisize << std::endl
        << "mpisizeDistSparse = " << mpisizeDistSparse << std::endl
        << "This is an illegal value." << std::endl;
      ErrorHandling( msg.str().c_str() );
    }

    // Compute the local column partition in DistSparseMatrix
    Int numColFirst = sizeMat / mpisizeDistSparse;

    VecPrtn  vecPrtn;
    vecPrtn.ownerInfo.Resize( sizeMat );
    IntNumVec&  vecOwner = vecPrtn.ownerInfo;
    for( Int i = 0; i < sizeMat; i++ ){
      vecOwner(i) = std::min( i / numColFirst, mpisizeDistSparse - 1 );
    }

    // nonzero row indices distributedly saved according to column partitions.
    DistVec<Int, std::vector<Int>, VecPrtn>   distRow;
    // nonzero values distributedly saved according to column partitions.
    DistVec<Int, std::vector<F>,   VecPrtn>   distVal;

    distRow.Prtn() = vecPrtn;
    distVal.Prtn() = vecPrtn;

    std::set<Int>  ownerSet;

    // Convert DistElemMat to intermediate data structure
    for( typename std::map<ElemMatKey, NumMat<F> >::const_iterator 
        mi  = distMat.LocalMap().begin();
//        mi != distMat.LocalMap().end(); mi++ ){
        mi != distMat.LocalMap().end(); ++mi ){
      ElemMatKey elemKey = (*mi).first;
      if( distMat.Prtn().Owner( elemKey ) == mpirank ){
        Index3 key1 = elemKey.first;
        Index3 key2 = elemKey.second;
        const std::vector<Int>& idx1 = basisIdx( key1(0), key1(1), key1(2) );
        const std::vector<Int>& idx2 = basisIdx( key2(0), key2(1), key2(2) );
        const NumMat<F>& localMat = (*mi).second;
        // Skip if there is no basis functions.
        if( localMat.Size() == 0 )
          continue;

        if( localMat.m() != idx1.size() ||
            localMat.n() != idx2.size() ){
          std::ostringstream msg;
          msg 
            << "Local matrix size is not consistent." << std::endl
            << "localMat   size : " << localMat.m() << " x " << localMat.n() << std::endl
            << "idx1       size : " << idx1.size() << std::endl
            << "idx2       size : " << idx2.size() << std::endl;
          ErrorHandling( msg.str().c_str() );
        }

        // Distribute the matrix element and row indices to intermediate
        // data structure in an element by element fashion.

        for( Int a = 0; a < localMat.m(); a++ ){
          Int row = idx1[a];
          ownerSet.insert( row );
          std::vector<Int>& vecRow = distRow.LocalMap()[row];
          std::vector<F>& vecVal = distVal.LocalMap()[row];
          for( Int b = 0; b < localMat.n(); b++ ){
            vecRow.push_back( idx2[b] + 1 );
            vecVal.push_back( localMat( a, b ) );
          }
        } // for (a)
      } // own this matrix block
    } // for (mi)

#if ( _DEBUGlevel_ >= 2 )
    if( mpirank == 0 ){
      for( typename std::map<Int, std::vector<Int> >::const_iterator 
          mi  = distRow.LocalMap().begin();
//          mi != distRow.LocalMap().end(); mi++ ){
          mi != distRow.LocalMap().end(); ++mi ){
        std::cout << "Row " << (*mi).first << std::endl << (*mi).second << std::endl;
      }
      for( typename std::map<Int, std::vector<F> >::const_iterator 
          mi  = distVal.LocalMap().begin();
//          mi != distVal.LocalMap().end(); mi++ ){
          mi != distRow.LocalMap().end(); ++mi ){
        std::cout << "Row " << (*mi).first << std::endl << (*mi).second << std::endl;
      }
    }
#endif


    // Communication of the matrix
    {

      std::vector<Int>  keyIdx;
      keyIdx.insert( keyIdx.end(), ownerSet.begin(), ownerSet.end() );
      //        std::cout << keyIdx << std::endl;
      //        std::cout << vecOwner << std::endl;

      distRow.PutBegin( keyIdx, NO_MASK );
      distVal.PutBegin( keyIdx, NO_MASK );

      distRow.PutEnd( NO_MASK, PutMode::REPLACE );
      distVal.PutEnd( NO_MASK, PutMode::REPLACE );

      // Clean to save space
      std::vector<Int>  eraseKey;
      for( typename std::map<Int, std::vector<Int> >::iterator 
          mi  = distRow.LocalMap().begin();
//          mi != distRow.LocalMap().end(); mi++ ){
          mi != distRow.LocalMap().end(); ++mi ){
        Int key = (*mi).first;
        if( distRow.Prtn().Owner( key ) != mpirank ){
          eraseKey.push_back( key );
        }
      } // for (mi)

      for( std::vector<Int>::iterator vi = eraseKey.begin();
//          vi != eraseKey.end(); vi++ ){
          vi != eraseKey.end(); ++vi ){
        distRow.LocalMap().erase( *vi );
        distVal.LocalMap().erase( *vi );
      }    
    }

    // Copy the values from intermediate data structure to distSparseMat
    // This only occur for the first mpisizeDistSparse processors
    if( mpirank < mpisizeDistSparse )
    {
      Int firstCol = mpirank * numColFirst;
      Int numColLocal;
      if( mpirank == mpisizeDistSparse - 1 )
        numColLocal = sizeMat - numColFirst * (mpisizeDistSparse - 1 );
      else
        numColLocal = numColFirst;

      // Global information
      distSparseMat.size        = sizeMat;
      // Note the 1-based convention
      distSparseMat.firstCol    = firstCol + 1;

      // Local information
      IntNumVec& colptrLocal = distSparseMat.colptrLocal;
      IntNumVec& rowindLocal = distSparseMat.rowindLocal;
      NumVec<F>& nzvalLocal  = distSparseMat.nzvalLocal;

      Int nnzLocal = 0;
      Int nnz = 0;
      colptrLocal.Resize( numColLocal + 1 );
      colptrLocal[0] = 1;

      for( Int row = firstCol; row < firstCol + numColLocal; row++ ){
        if( distRow.Prtn().Owner( row ) != mpirank ){
          ErrorHandling( "The owner information in distRow is incorrect.");
        }
        std::vector<Int>&  rowVec = distRow.LocalMap()[row];
        nnzLocal += rowVec.size();
        Int cur = row - firstCol;
        colptrLocal[cur+1] = colptrLocal[cur] + rowVec.size();
      }    

      //        mpi::Allreduce( &nnzLocal, &nnz, 1, MPI_SUM, commDistSparse );

      rowindLocal.Resize( nnzLocal );
      nzvalLocal.Resize( nnzLocal );

      for( Int row = firstCol; row < firstCol + numColLocal; row++ ){
        if( distRow.Prtn().Owner( row ) != mpirank ){
          ErrorHandling( "The owner information in distRow is incorrect.");
        }
        std::vector<Int>&  rowVec = distRow.LocalMap()[row];
        std::vector<F>&    valVec = distVal.LocalMap()[row];
        Int cur = row - firstCol;
        std::copy( rowVec.begin(), rowVec.end(), &rowindLocal[colptrLocal[cur]-1] );
        std::copy( valVec.begin(), valVec.end(), &nzvalLocal[colptrLocal[cur]-1] );
      }    


      // Other global and local information
      distSparseMat.nnzLocal = nnzLocal;
      //        distSparseMat.nnz      = nnz;

    }


    return;
  }  // -----  end of function DistElemMatToDistSparseMat  ----- 


/// @brief Convert a DistSparseMatrix distributed on the first
/// mpisizeDistSparse processors to a DistElemMat format.
///
/// Note: 
///
/// 1) The size of the communicatior for distSparseMat must be
/// the same as mpisizeDistSparse.
///
/// 2) It is assumed that the matrix is symmetric.  Therefore the
/// position of row and column may be confusing.  This will be a
/// problem later when structurally symmetric matrix is considered.
/// This shall be fixed later.
template<typename F>
  void DistSparseMatToDistElemMat(
      const DistSparseMatrix<F>&                           distSparseMat,
      const Int                                            sizeMat,
      const ElemMatPrtn&                                   elemMatPrtn,
      DistVec<ElemMatKey, NumMat<F>, ElemMatPrtn>&         distMat,
      const NumTns<std::vector<Int> >&                     basisIdx,
      MPI_Comm                                               comm,
      Int                                                  mpisizeDistSparse ){
    Int mpirank, mpisize;
    MPI_Comm_rank( comm, &mpirank );
    MPI_Comm_size( comm, &mpisize );

    if( mpisizeDistSparse > mpisize ){
      std::ostringstream msg;
      msg << std::endl
        << "mpisize = " << mpisize << std::endl
        << "mpisizeDistSparse = " << mpisizeDistSparse << std::endl
        << "This is an illegal value." << std::endl;
      ErrorHandling( msg.str().c_str() );
    }


    distMat.LocalMap().clear();

    distMat.Prtn() = elemMatPrtn;

    // Convert the inverse map of basisIdx 
    std::vector<Index3> invBasisIdx( sizeMat );
    Index3 numElem = Index3( basisIdx.m(), basisIdx.n(), basisIdx.p() );
    for( Int g = 0; g < sizeMat; g++ ){
      bool flag = false;
      for( Int k = 0; k < numElem[2]; k++ )
        for( Int j = 0; j < numElem[1]; j++ )
          for( Int i = 0; i < numElem[0]; i++ ){
            if( flag == true ) 
              break;
            const std::vector<Int>& idx = basisIdx( i, j, k );
            if( g >= idx[0] & g <= idx[idx.size()-1] ){
              invBasisIdx[g] = Index3( i, j, k );
              flag = true;
            }
          }
      if( flag == false ){
        std::ostringstream msg;
        msg << std::endl
          << "The element index for the row " << g << " was not found!" << std::endl;
        ErrorHandling( msg.str().c_str() );
      }
    }

#if ( _DEBUGlevel_ >= 2 )
    statusOFS << "The inverse map of basisIdx: " << std::endl;
    for( Int g = 0; g < sizeMat; g++ ){
      statusOFS << g << " : " << invBasisIdx[g] << std::endl;
    }
#endif



    // Convert nonzero values in DistSparseMat. Only processors with rank
    // smaller than mpisizeDistSparse participate

    std::set<ElemMatKey> ownerSet;

    if( mpirank < mpisizeDistSparse )
    {
      Int numColFirst = sizeMat / mpisizeDistSparse;
      Int firstCol = mpirank * numColFirst;
      Int numColLocal;
      if( mpirank == mpisizeDistSparse - 1 )
        numColLocal = sizeMat - numColFirst * (mpisizeDistSparse - 1 );
      else
        numColLocal = numColFirst;

      const IntNumVec&  colptrLocal = distSparseMat.colptrLocal;
      const IntNumVec&  rowindLocal = distSparseMat.rowindLocal;
      const NumVec<F>&  nzvalLocal  = distSparseMat.nzvalLocal;

      for( Int j = 0; j < numColLocal; j++ ){
        Int jcol = firstCol + j;
        Index3 jkey = invBasisIdx[jcol];
        for( Int i = colptrLocal(j) - 1;
            i < colptrLocal(j+1) - 1; i++ ){
          Int irow = rowindLocal(i) - 1;
          Index3 ikey = invBasisIdx[irow];
          ElemMatKey matKey( std::pair<Index3,Index3>(ikey, jkey) );
          typename std::map<ElemMatKey, NumMat<F> >::iterator 
            mi = distMat.LocalMap().find( matKey );
          if( mi == distMat.LocalMap().end() ){
            Int isize = basisIdx( ikey(0), ikey(1), ikey(2) ).size();
            Int jsize = basisIdx( jkey(0), jkey(1), jkey(2) ).size();
            NumMat<F> empty( isize, jsize );
            SetValue( empty, (F)(0) ); 
            distMat.LocalMap()[matKey] = empty;
            mi = distMat.LocalMap().find( matKey );
            ownerSet.insert( matKey );
          }
          NumMat<F>& mat = (*mi).second;
          Int io = irow - basisIdx( ikey(0), ikey(1), ikey(2) )[0];
          Int jo = jcol - basisIdx( jkey(0), jkey(1), jkey(2) )[0];
          mat(io, jo) = nzvalLocal(i);
        }
      }
    } 



    // Communication of the matrix
    {

      std::vector<ElemMatKey>  keyIdx;
      keyIdx.insert( keyIdx.end(), ownerSet.begin(), ownerSet.end() );

      distMat.PutBegin( keyIdx, NO_MASK );
      distMat.PutEnd( NO_MASK, PutMode::COMBINE );

      // Clean to save space
      std::vector<ElemMatKey>  eraseKey;
      for( typename std::map<ElemMatKey, NumMat<F> >::iterator 
          mi  = distMat.LocalMap().begin();
//          mi != distMat.LocalMap().end(); mi++ ){
          mi != distMat.LocalMap().end(); ++mi ){
        ElemMatKey key = (*mi).first;
        if( distMat.Prtn().Owner( key ) != mpirank ){
          eraseKey.push_back( key );
        }
      } // for (mi)

      for( std::vector<ElemMatKey>::iterator vi = eraseKey.begin();
//          vi != eraseKey.end(); vi++ ){
          vi != eraseKey.end(); ++vi ){
        distMat.LocalMap().erase( *vi );
      }    
    }

    return;
  }  // -----  end of function DistSparseMatToDistElemMat  ----- 


/// @brief Convert a DistSparseMatrix distributed from one communicator
/// to a DistElemMat format on another communicator.
///
/// LL: 11/26/2014: This function is not used because it is not scalable
/// for a large number of processors
///
/// Note: 
///
/// 1. There is no restriction on the communicators, i.e. in principle
/// the two communicators can be completely different.  This may not be
/// the most efficient way for implementing this feature, but should be
/// more general and maintainable.
///
/// 2. It is assumed that the matrix is symmetric.  Therefore the
/// position of row and column may be confusing.  This will be a
/// problem later when structurally symmetric matrix is considered.
/// This shall be fixed later.
///
/// @param[in] distSparseMat (local) Matrix distibuted according to the
/// DistSparseMatrix format.  
/// @param[in] sizeMat (global) Size of the matrix. Owned by all
/// processors in comm.
/// @param[in] elemMatPrtn (global) Partition of the element matrix
/// format.
/// @param[out] distMat (local) Matrix distributed according to 2D element
/// indices.  
/// @param[in] basisIdx (global) Indices for all basis functions.  The input from
/// each processor column of comm1 should be exactly the same.
/// @param[in] comm (global) Global communicator. 
/// @param[in] commElem (local) Communicator for distElemMat. In the current step
/// with intra-element parallelization, the comm for distElemMat can be
/// colComm.
/// @param[in] commSparse (local) Communicator for distSparseMat. For
/// PEXSI, this is the communicator for a single pole
/// @param[in] mpirankElemVec (global) mpiranks in the global
/// communicator comm for processors in commElem.  This vector should
/// follow a non-decreasing order.
/// @param[in] mpirankSparseVec (global) mpiranks in the global
/// communicator comm for processors saving the matrix in
/// DistSparseMatrix format.
template<typename F>
  void DistSparseMatToDistElemMat2(
      const DistSparseMatrix<F>&                           distSparseMat,
      const Int                                            sizeMat,
      const ElemMatPrtn&                                   elemMatPrtn,
      DistVec<ElemMatKey, NumMat<F>, ElemMatPrtn>&         distMat,
      const NumTns<std::vector<Int> >&                     basisIdx,
      MPI_Comm                                               comm,
      MPI_Comm                                             commElem,
      MPI_Comm                                             commSparse,
      const std::vector<Int>&                              mpirankElemVec,
      const std::vector<Int>&                              mpirankSparseVec){
    Int mpirank, mpisize;
    MPI_Comm_rank( comm, &mpirank );
    MPI_Comm_size( comm, &mpisize );

    bool isInElem, isInSparse;
    isInElem = ( find( mpirankElemVec.begin(), 
          mpirankElemVec.end(), mpirank ) != mpirankElemVec.end() );
    isInSparse = ( find( mpirankSparseVec.begin(), 
          mpirankSparseVec.end(), mpirank ) != mpirankSparseVec.end() );


    // commElem rank and size
    Int mpirankElem; 
    Int mpisizeElem = mpirankElemVec.size();
    if( isInElem ){
      Int tmp;
      MPI_Comm_rank( commElem, &mpirankElem );
      MPI_Comm_size( commElem, &tmp );

      if( mpisizeElem != tmp ){
        ErrorHandling("mpisizeElem read from input does not agree with commElem.");
      }
    }

    if( mpisizeElem > mpisize ){
      ErrorHandling("mpisizeElem cannot be larger than mpisize.");
    }

    // commSparse size and other information
    Int mpirankSparse; 
    Int mpisizeSparse = mpirankSparseVec.size();
    Int numColLocal;
    if( isInSparse ){
      Int tmp;
      MPI_Comm_rank( commSparse, &mpirankSparse );
      MPI_Comm_size( commSparse, &tmp );

      if( mpisizeSparse != tmp ){
        ErrorHandling("mpisizeSparse read from input does not agree with commSparse.");
      }
    }

    if( mpisizeSparse > mpisize ){
      ErrorHandling("mpisizeSparse cannot be larger than mpisize.");
    }

    // Compute the local column partition in DistSparseMatrix
    // Temporarily change the Prtn to cover the whole communicator

    Index3 numElem = Index3( basisIdx.m(), basisIdx.n(), basisIdx.p() );

    ElemMatPrtn elemMatGlbPrtn;
    elemMatGlbPrtn.ownerInfo.Resize( numElem[0], numElem[1], numElem[2] ); 
    IntNumTns& elemMatGlbOwner = elemMatGlbPrtn.ownerInfo;
    const IntNumTns& elemMatOwner = elemMatPrtn.ownerInfo; 

    for( Int k = 0; k < numElem[2]; k++ )
      for( Int j = 0; j < numElem[1]; j++ )
        for( Int i = 0; i < numElem[0]; i++ ){
          elemMatGlbOwner(i,j,k) = mpirankElemVec[elemMatOwner(i,j,k)];
        }

    // Communication using the global communicator
    DistVec<ElemMatKey, NumMat<F>, ElemMatPrtn> distGlbMat;
    distGlbMat.Prtn() = elemMatGlbPrtn;
    distGlbMat.SetComm( comm );


    // Convert the inverse map of basisIdx 
    std::vector<Index3> invBasisIdx( sizeMat );
    for( Int g = 0; g < sizeMat; g++ ){
      bool flag = false;
      for( Int k = 0; k < numElem[2]; k++ )
        for( Int j = 0; j < numElem[1]; j++ )
          for( Int i = 0; i < numElem[0]; i++ ){
            if( flag == true ) 
              break;
            const std::vector<Int>& idx = basisIdx( i, j, k );
            if( g >= idx[0] & g <= idx[idx.size()-1] ){
              invBasisIdx[g] = Index3( i, j, k );
              flag = true;
            }
          }
      if( flag == false ){
        std::ostringstream msg;
        msg << std::endl
          << "The element index for the row " << g << " was not found!" << std::endl;
        ErrorHandling( msg.str().c_str() );
      }
    }

#if ( _DEBUGlevel_ >= 2 )
    statusOFS << "The inverse map of basisIdx: " << std::endl;
    for( Int g = 0; g < sizeMat; g++ ){
      statusOFS << g << " : " << invBasisIdx[g] << std::endl;
    }
#endif



    // Convert nonzero values in DistSparseMat. Only processors with rank
    // smaller than mpisizeSparse participate

    std::set<ElemMatKey> ownerSet;

    if( isInSparse )
    {
      Int numColFirst = sizeMat / mpisizeSparse;
      Int firstCol = mpirankSparse * numColFirst;
      Int numColLocal;
      if( mpirankSparse == mpisizeSparse - 1 )
        numColLocal = sizeMat - numColFirst * (mpisizeSparse - 1 );
      else
        numColLocal = numColFirst;

      const IntNumVec&  colptrLocal = distSparseMat.colptrLocal;
      const IntNumVec&  rowindLocal = distSparseMat.rowindLocal;
      const NumVec<F>&  nzvalLocal  = distSparseMat.nzvalLocal;

      for( Int j = 0; j < numColLocal; j++ ){
        Int jcol = firstCol + j;
        Index3 jkey = invBasisIdx[jcol];
        for( Int i = colptrLocal(j) - 1;
            i < colptrLocal(j+1) - 1; i++ ){
          Int irow = rowindLocal(i) - 1;
          Index3 ikey = invBasisIdx[irow];
          ElemMatKey matKey( std::pair<Index3,Index3>(ikey, jkey) );
          typename std::map<ElemMatKey, NumMat<F> >::iterator 
            mi = distGlbMat.LocalMap().find( matKey );
          if( mi == distGlbMat.LocalMap().end() ){
            Int isize = basisIdx( ikey(0), ikey(1), ikey(2) ).size();
            Int jsize = basisIdx( jkey(0), jkey(1), jkey(2) ).size();
            NumMat<F> empty( isize, jsize );
            SetValue( empty, (F)(0) ); 
            distGlbMat.LocalMap()[matKey] = empty;
            mi = distGlbMat.LocalMap().find( matKey );
            ownerSet.insert( matKey );
          }
          NumMat<F>& mat = (*mi).second;
          Int io = irow - basisIdx( ikey(0), ikey(1), ikey(2) )[0];
          Int jo = jcol - basisIdx( jkey(0), jkey(1), jkey(2) )[0];
          mat(io, jo) = nzvalLocal(i);
        }
      }
    } // if ( isInSparse ) 


    // Communication of the matrix
    {

      std::vector<ElemMatKey>  keyIdx;
      keyIdx.insert( keyIdx.end(), ownerSet.begin(), ownerSet.end() );

      distGlbMat.PutBegin( keyIdx, NO_MASK );
      distGlbMat.PutEnd( NO_MASK, PutMode::COMBINE );

      // Clean to save space
      std::vector<ElemMatKey>  eraseKey;
      for( typename std::map<ElemMatKey, NumMat<F> >::iterator 
          mi  = distGlbMat.LocalMap().begin();
//          mi != distGlbMat.LocalMap().end(); mi++ ){
          mi != distGlbMat.LocalMap().end(); ++mi ){
        ElemMatKey key = (*mi).first;
        if( distGlbMat.Prtn().Owner( key ) != mpirank ){
          eraseKey.push_back( key );
        }
      } // for (mi)

      for( std::vector<ElemMatKey>::iterator vi = eraseKey.begin();
//          vi != eraseKey.end(); vi++ ){
          vi != eraseKey.end(); ++vi ){
        distGlbMat.LocalMap().erase( *vi );
      }    
    }



    // Copy the values from intermediate data structure to distElemMat
    if( isInElem ){
      distMat.LocalMap().clear();
      distMat.Prtn() = elemMatPrtn;
      for( typename std::map<ElemMatKey, NumMat<F> >::iterator 
          mi  = distGlbMat.LocalMap().begin();
          mi != distGlbMat.LocalMap().end(); mi++ ){
        ElemMatKey key = (*mi).first;
        if( distGlbMat.Prtn().Owner( key ) != mpirank ){
          ErrorHandling( "The owner information in distGlbMat is incorrect.");
        }
        distMat.LocalMap()[key] = (*mi).second;
      }
    }


    return;
  }  // -----  end of function DistSparseMatToDistElemMat2  ----- 


/// @brief Convert a DistSparseMatrix distributed from one communicator
/// to a DistElemMat format on another communicator.
///
/// Note: 
///
/// 1. Compared to DistSparseMatToDistElemMat2, there is restriction of
/// the communicator that the processors for saving the DistSparseMat
/// must be a subset of the processors for saving the matrix in
/// DistElemMat
///
/// 2. It is assumed that the matrix is symmetric.  Therefore the
/// position of row and column may be confusing.  This will be a
/// problem later when structurally symmetric matrix is considered.
/// This shall be fixed later.
///
/// @param[in] distSparseMat (local) Matrix distibuted according to the
/// DistSparseMatrix format.  
/// @param[in] sizeMat (global) Size of the matrix. Owned by all
/// processors in comm.
/// @param[in] elemMatPrtn (global) Partition of the element matrix
/// format.
/// @param[out] distMat (local) Matrix distributed according to 2D element
/// indices.  
/// @param[in] basisIdx (global) Indices for all basis functions.  The input from
/// each processor column of comm1 should be exactly the same.
/// @param[in] commElem (global) Communicator for distElemMat. In the current step
/// with intra-element parallelization, the communicator for distElemMat can be
/// colComm.
/// @param[in] mpirankSparseVec (global) mpiranks in the global
/// communicator commElem for processors saving the matrix in
/// DistSparseMatrix format.
template<typename F>
  void DistSparseMatToDistElemMat3(
      const DistSparseMatrix<F>&                           distSparseMat,
      const Int                                            sizeMat,
      const ElemMatPrtn&                                   elemMatPrtn,
      DistVec<ElemMatKey, NumMat<F>, ElemMatPrtn>&         distMat,
      const NumTns<std::vector<Int> >&                     basisIdx,
      const std::vector<Index3>&                           basisInvIdx,
      MPI_Comm                                             commElem,
      const std::vector<Int>&                              mpirankSparseVec){

    Real timeSta, timeEnd;

    // commElem rank and size
    Int mpirankElem; 
    Int mpisizeElem;
    MPI_Comm_rank( commElem, &mpirankElem );
    MPI_Comm_size( commElem, &mpisizeElem );

    bool isInSparse;
    // Note that mpirank refers to the rank in the local communicator
    isInSparse = ( find( mpirankSparseVec.begin(), 
          mpirankSparseVec.end(), mpirankElem ) != mpirankSparseVec.end() );

    Int mpisizeSparse = mpirankSparseVec.size();

    if( mpisizeSparse > mpisizeElem ){
      ErrorHandling("The number of processors for DistSparseMatrix cannot exceed that of DistElemMat.");
    }

    Int mpirankSparse = -1;
    for( Int i = 0; i < mpisizeSparse; i++ ){
      if( mpirankElem == mpirankSparseVec[i] )
        mpirankSparse = mpirankElem;
    }


    // Compute the local column partition in DistSparseMatrix
    // Temporarily change the Prtn to cover the whole communicator

    Index3 numElem = Index3( basisIdx.m(), basisIdx.n(), basisIdx.p() );

    distMat.LocalMap().clear();
    distMat.Prtn() = elemMatPrtn;
    distMat.SetComm( commElem );

    // Convert nonzero values in DistSparseMat. Only processors with rank
    // smaller than mpisizeSparse participate

    std::set<ElemMatKey> ownerSet;

    GetTime(timeSta);

    if( isInSparse )
    {
      Int numColFirst = sizeMat / mpisizeSparse;
      Int firstCol = mpirankSparse * numColFirst;
      Int numColLocal;
      if( mpirankSparse == mpisizeSparse - 1 )
        numColLocal = sizeMat - numColFirst * (mpisizeSparse - 1 );
      else
        numColLocal = numColFirst;

      const IntNumVec&  colptrLocal = distSparseMat.colptrLocal;
      const IntNumVec&  rowindLocal = distSparseMat.rowindLocal;
      const NumVec<F>&  nzvalLocal  = distSparseMat.nzvalLocal;

      for( Int j = 0; j < numColLocal; j++ ){
        Int jcol = firstCol + j;
        Index3 jkey = basisInvIdx[jcol];
        for( Int i = colptrLocal(j) - 1;
            i < colptrLocal(j+1) - 1; i++ ){
          Int irow = rowindLocal(i) - 1;
          Index3 ikey = basisInvIdx[irow];
          ElemMatKey matKey( std::pair<Index3,Index3>(ikey, jkey) );
          typename std::map<ElemMatKey, NumMat<F> >::iterator 
            mi = distMat.LocalMap().find( matKey );
          if( mi == distMat.LocalMap().end() ){
            Int isize = basisIdx( ikey(0), ikey(1), ikey(2) ).size();
            Int jsize = basisIdx( jkey(0), jkey(1), jkey(2) ).size();
            NumMat<F> empty( isize, jsize );
            SetValue( empty, (F)(0) ); 
            distMat.LocalMap()[matKey] = empty;
            mi = distMat.LocalMap().find( matKey );
            ownerSet.insert( matKey );
          }
          NumMat<F>& mat = (*mi).second;
          Int io = irow - basisIdx( ikey(0), ikey(1), ikey(2) )[0];
          Int jo = jcol - basisIdx( jkey(0), jkey(1), jkey(2) )[0];
          mat(io, jo) = nzvalLocal(i);
        }
      }
    } // if ( isInSparse ) 

    GetTime(timeEnd);
#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for converting the matrix locally is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

    // Communication of the matrix
    GetTime(timeSta);
    {

      std::vector<ElemMatKey>  keyIdx;
      keyIdx.insert( keyIdx.end(), ownerSet.begin(), ownerSet.end() );

      distMat.PutBegin( keyIdx, NO_MASK );
      distMat.PutEnd( NO_MASK, PutMode::COMBINE );

      // Clean to save space
      std::vector<ElemMatKey>  eraseKey;
      for( typename std::map<ElemMatKey, NumMat<F> >::iterator 
          mi  = distMat.LocalMap().begin();
//          mi != distMat.LocalMap().end(); mi++ ){
          mi != distMat.LocalMap().end(); ++mi ){
        ElemMatKey key = (*mi).first;
        if( distMat.Prtn().Owner( key ) != mpirankElem ){
          eraseKey.push_back( key );
        }
      } // for (mi)

      for( std::vector<ElemMatKey>::iterator vi = eraseKey.begin();
//          vi != eraseKey.end(); vi++ ){
          vi != eraseKey.end(); ++vi ){
        distMat.LocalMap().erase( *vi );
      }    
    }
    GetTime(timeEnd);
#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for actual communication is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

    return;
  }  // -----  end of function DistSparseMatToDistElemMat3  ----- 



/// @brief Convert a DistSparseMatrix distributed in compressed sparse
/// column format to a matrix distributed block-cyclicly as in
/// ScaLAPACK.
///
/// NOTE: 
///
/// 1. This subroutine assumes that proper descriptor has been given
/// by desc, which agrees with distMat.
///
/// 2. This subroutine is mainly used for converting a DistSparseMatrix
/// read from input file to ScaLAPACK format for diagonalization
/// purpose.
///
template<typename F>
  void DistSparseMatToScaMat(
      const DistSparseMatrix<F>&                           distMat,
      const scalapack::Descriptor&                         desc, 
      scalapack::ScaLAPACKMatrix<F>&                       scaMat,
      MPI_Comm                                               comm ){
    using namespace dgdft::scalapack;

    Int mpirank, mpisize;
    MPI_Comm_rank( comm, &mpirank );
    MPI_Comm_size( comm, &mpisize );

    Int nprow  = desc.NpRow();
    Int npcol  = desc.NpCol();
    Int myprow = desc.MypRow();
    Int mypcol = desc.MypCol();

    scaMat.SetDescriptor( desc );

    {
      std::vector<F>& vec = scaMat.LocalMatrix();
      for( Int i = 0; i < vec.size(); i++ ){
        vec[i] = (F) 0.0;
      }
    }

    Int MB = scaMat.MB(); 
    Int NB = scaMat.NB();

    if( MB != NB ){
      ErrorHandling("MB must be equal to NB.");
    }

    Int numRowBlock = scaMat.NumRowBlocks();
    Int numColBlock = scaMat.NumColBlocks();

    // Get the processor map
    IntNumMat  procGrid( nprow, npcol );
    SetValue( procGrid, 0 );
    {
      IntNumMat  procTmp( nprow, npcol );
      SetValue( procTmp, 0 );
      procTmp( myprow, mypcol ) = mpirank;
      mpi::Allreduce( procTmp.Data(), procGrid.Data(), nprow * npcol,
          MPI_SUM, comm );
    }


    BlockMatPrtn  blockPrtn;
    blockPrtn.ownerInfo.Resize( numRowBlock, numColBlock );
    IntNumMat&    blockOwner = blockPrtn.ownerInfo;
    for( Int jb = 0; jb < numColBlock; jb++ ){
      for( Int ib = 0; ib < numRowBlock; ib++ ){
        blockOwner( ib, jb ) = procGrid( ib % nprow, jb % npcol );
      }
    }

    // Intermediate variable for constructing the distributed matrix in
    // ScaLAPACK format.  
    DistVec<Index2, NumMat<F>, BlockMatPrtn> distScaMat;
    distScaMat.Prtn() = blockPrtn;

    // Initialize
    NumMat<F> empty( MB, MB );
    SetValue( empty, (F)(0) );
    for( Int jb = 0; jb < numColBlock; jb++ )
      for( Int ib = 0; ib < numRowBlock; ib++ ){
        Index2 key( ib, jb );
        if( distScaMat.Prtn().Owner( key ) == mpirank ){
          distScaMat.LocalMap()[key] = empty;
        }
      } // for (ib)

    // Convert a DistSparseMatrix in Compressed Sparse Column format to distScaMat
    // Compute the local column partition in DistSparseMatrix 
    Int sizeMat = distMat.size;
    Int numColFirst = sizeMat / mpisize;
    Int firstCol = mpirank * numColFirst;
    Int numColLocal;
    if( mpirank == mpisize - 1 )
      numColLocal = sizeMat - numColFirst * (mpisize - 1 );
    else
      numColLocal = numColFirst;

    // TODO Check the consistency of communicator

    // Local information 
    const IntNumVec& colptrLocal = distMat.colptrLocal;
    const IntNumVec& rowindLocal = distMat.rowindLocal;
    const NumVec<F>& nzvalLocal  = distMat.nzvalLocal;

    Int row, col, ib, jb, io, jo;
    F   val;

    for( Int j = 0; j < numColLocal; j++ ){
      col = firstCol + j;
      for( Int i = colptrLocal(j)-1; 
          i < colptrLocal(j+1)-1; i++ ){
        row = rowindLocal(i) - 1;
        val = nzvalLocal(i);

        ib = row / MB;
        jb = col / MB;
        io = row % MB;
        jo = col % MB;
        typename std::map<Index2, NumMat<F> >::iterator 
          ni = distScaMat.LocalMap().find( Index2(ib, jb) );
        // Contributes to blocks not owned by the current processor
        if( ni == distScaMat.LocalMap().end() ){
          distScaMat.LocalMap()[Index2(ib, jb)] = empty;
          ni = distScaMat.LocalMap().find( Index2(ib, jb) );
        }
        NumMat<F>&  mat = (*ni).second;
        mat(io, jo) += val;
      } // for (i)
    } // for (j)

    // Communication of the matrix
    {
      // Prepare
      std::vector<Index2>  keyIdx;
      for( typename std::map<Index2, NumMat<F> >::iterator 
          mi  = distScaMat.LocalMap().begin();
//          mi != distScaMat.LocalMap().end(); mi++ ){
          mi != distScaMat.LocalMap().end(); ++mi ){
        Index2 key = (*mi).first;
        if( distScaMat.Prtn().Owner( key ) != mpirank ){
          keyIdx.push_back( key );
        }
      } // for (mi)

      // Communication
      distScaMat.PutBegin( keyIdx, NO_MASK );
      distScaMat.PutEnd( NO_MASK, PutMode::COMBINE );

      // Clean to save space
      std::vector<Index2>  eraseKey;
      for( typename std::map<Index2, NumMat<F> >::iterator 
          mi  = distScaMat.LocalMap().begin();
//          mi != distScaMat.LocalMap().end(); mi++ ){
          mi != distScaMat.LocalMap().end(); ++mi ){
        Index2 key = (*mi).first;
        if( distScaMat.Prtn().Owner( key ) != mpirank ){
          eraseKey.push_back( key );
        }
      } // for (mi)

      for( std::vector<Index2>::iterator vi = eraseKey.begin();
//          vi != eraseKey.end(); vi++ ){
          vi != eraseKey.end(); ++vi ){
        distScaMat.LocalMap().erase( *vi );
      }    
    }

    // Copy the matrix values from distScaMat to scaMat
    {
      for( typename std::map<Index2, NumMat<F> >::iterator 
          mi  = distScaMat.LocalMap().begin();
//          mi != distScaMat.LocalMap().end(); mi++ ){
          mi != distScaMat.LocalMap().end(); ++mi ){
        Index2 key = (*mi).first;
        if( distScaMat.Prtn().Owner( key ) == mpirank ){
          Int ib = key(0), jb = key(1);
          Int offset = ( jb / npcol ) * MB * scaMat.LocalLDim() + 
            ( ib / nprow ) * MB;
          lapack::Lacpy( 'A', MB, MB, (*mi).second.Data(),
              MB, scaMat.Data() + offset, scaMat.LocalLDim() );
        } // own this block
      } // for (mi)
    }

    return;
  }  // -----  end of function DistElemMatToScaMat  ----- 

} // namespace dgdft

#endif // _HAMILTONIAN_DG_CONVERSION_HPP_
