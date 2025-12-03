/*
   Copyright (c) 2012 The Regents of the University of California,
   through Lawrence Berkeley National Laboratory.  

Authors: Xinming Qin

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
/// @file hamiltonian_dg_hfx.cpp
/// @brief Implementation of the Hamiltonian class for DG calculation.
/// @date 2022-10-5
#include  "hamiltonian_dg.hpp"
#include  "mpi_interf.hpp"
#include  "blas.hpp"
#include  "lapack.hpp"
#include  "scalapack.hpp"
#include  "utility.hpp"

using namespace dgdft::esdf;

using namespace dgdft::scalapack;

namespace dgdft{

// *********************************************************************
// Hamiltonian class for constructing the DG HFX matrix
// *********************************************************************

// *********************************************************************
// Collect the local ALBs of neighbor elements for HFX matrix construction
// Start the communication of the ALBs
// Each rowComm only requires the local ALBs on neigbhor colComm with t
// the same rowComm
// *********************************************************************

void
HamiltonianDG:: CollectNeighborBasis( )
{
  Int mpirank, mpisize;
  MPI_Comm_rank( domain_.comm, &mpirank );
  MPI_Comm_size( domain_.comm, &mpisize );

  Int mpirankRow;  MPI_Comm_rank(domain_.rowComm, &mpirankRow);
  Int mpisizeRow;  MPI_Comm_size(domain_.rowComm, &mpisizeRow);

  Int mpirankCol;  MPI_Comm_rank(domain_.colComm, &mpirankCol);
  Int mpisizeCol;  MPI_Comm_size(domain_.colComm, &mpisizeCol);

//  statusOFS << " HFX::  Collect the local ALBs of neighbor elements for HFX matrix construction " << std::endl;

  Real timeSta, timeEnd;
//  Method 1 :: Collect the local ALBs on uniform grid from column cmmunicators of neighbor elements.
//  (GetBegin/GetEnd/PutBegin/PutEnd) are not directly used. Thus we redifine a neighborbasisUniform_
//  distributed on neigbhor elements (colComm).
//  This implementation requires to transform local ALBs from LGL to uniform grid.

  DblNumTns& LGLWeight3D = LGLWeight3D_;

  GetTime( timeSta );
  for( Int k = 0; k < numElem_[2]; k++ )
    for( Int j = 0; j < numElem_[1]; j++ )
      for( Int i = 0; i < numElem_[0]; i++ ){
        Index3 key(i, j, k);
        if( elemPrtn_.Owner(key) == (mpirank / dmRow_) ){

          DblNumMat& localBasisLGL = basisLGL_.LocalMap()[key];
          Real factor = 0.0;
          DblNumMat localBasisUniform( numUniformGridElemHFX_.prod(), localBasisLGL.n() );

          if( isHFXFineGrid_ ) {
            factor = domain_.Volume() / domain_.NumGridTotalFine();
            for( Int g = 0; g < localBasisLGL.n(); g++ ){
              InterpLGLToUniform(
                numLGLGridElem_,
                numUniformGridElemHFX_,
                localBasisLGL.VecData(g),
                localBasisUniform.VecData(g) );
            }
          }
          else{
            factor = domain_.Volume() / domain_.NumGridTotal();
            for( Int g = 0; g < localBasisLGL.n(); g++ ){
              InterpLGLToUniform2(
                numLGLGridElem_,
                numUniformGridElemHFX_,
                localBasisLGL.VecData(g),
                localBasisUniform.VecData(g) );
            }
          }
           
#if ( _DEBUGlevel_ >= 2 ) //  print overlap
          DblNumMat basisTemp (localBasisLGL.m(), localBasisLGL.n());
          SetValue( basisTemp, 0.0 );

          // This is the same as the FourDotProduct process.
          for( Int g = 0; g < localBasisLGL.n(); g++ ){
            Real *ptr1 = LGLWeight3D.Data();
            Real *ptr2 = localBasisLGL.VecData(g);
            Real *ptr3 = basisTemp.VecData(g);
            for( Int l = 0; l < localBasisLGL.m(); l++ ){
              *(ptr3++) = (*(ptr1++)) * (*(ptr2++)) ;
             }
          }
        
          DblNumMat SmatLGL(localBasisLGL.n(), localBasisLGL.n() );
          SetValue( SmatLGL, 0.0 );
          DblNumMat SmatUniform(localBasisUniform.n(), localBasisUniform.n() );
          SetValue( SmatUniform, 0.0 );

          blas::Gemm( 'T', 'N',localBasisLGL.n() , localBasisLGL.n(), localBasisLGL.m(),
                     1.0, localBasisLGL.Data(), localBasisLGL.m(),
                     basisTemp.Data(), localBasisLGL.m(), 0.0,
                     SmatLGL.Data(), localBasisLGL.n() );

          blas::Gemm( 'T', 'N',localBasisUniform.n() , localBasisUniform.n(), localBasisUniform.m(),
                     factor, localBasisUniform.Data(), localBasisUniform.m(),
                     localBasisUniform.Data(), localBasisUniform.m(), 0.0,
                     SmatUniform.Data(), localBasisUniform.n() );

          for( Int p = 0; p < localBasisLGL.n(); p++ ){
//              for( Int q = 0; q < localBasisLGL.n(); q++ ){
                  statusOFS << " p " << p <<  std::endl; // " q " << q << std::endl;
                  statusOFS << " SmatLGL " << SmatLGL(p,p) << " SmatUniform " <<  SmatUniform(p,p) << std::endl;
//              }
          }
#endif
          neighborbasisUniform_.LocalMap()[key] = localBasisUniform;
        } // if  elemPrtn_.Owner(key)
  } // for i

  GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 1 )
  statusOFS << " HFX: Time for LGL2Unifom ALBs = " << timeEnd - timeSta << " [s]" << std::endl;
#endif
  // ****************************************************************************
  // Second: Comunicate local ALBs between neighbor elements for each column proc
  // ****************************************************************************
  //
  // Scheme 1: Element loop
  //  This is a straightforward but reliable implementation
  //
  GetTime( timeSta );
  std::set<Index3> neighborSet;
  for( Int k = 0; k < numElem_[2]; k++ )
    for( Int j = 0; j < numElem_[1]; j++ )
      for( Int i = 0; i < numElem_[0]; i++ ){
        Index3 key( i, j, k );
        if( elemPrtn_.Owner(key) == (mpirank / dmRow_) ){
          std::vector<Index3>   idx(3);

          for( Int d = 0; d < DIM; d++ ){
            // Previous
            if( key[d] == 0 )
              idx[0][d] = numElem_[d]-1;
            else
              idx[0][d] = key[d]-1;

            // Current
            idx[1][d] = key[d];

            //  We ignore the previous neighbor elements using the symmetry
            //  H(key1,key2) = tanspose H(key2, key1)
            //  Next
            if( key[d] == numElem_[d]-1)
              idx[2][d] = 0;
            else
              idx[2][d] = key[d] + 1;
          } // for (d)

          // Tensor product
          for( Int c = 0; c < 3; c++ )
            for( Int b = 0; b < 3; b++ )
              for( Int a = 0; a < 3; a++ ){
                // Not the element key itself
                if( idx[a][0] != i || idx[b][1] != j || idx[c][2] != k ){
                  neighborSet.insert( Index3( idx[a][0], idx[b][1], idx[c][2] ) );
                }
              } // for (a)
        } // own this element
  } // for (i)

  std::vector<Index3>  neighborIdx;
  neighborIdx.insert( neighborIdx.begin(), neighborSet.begin(), neighborSet.end() );

//    statusOFS << " ALB getKeys_list " << neighborIdx << std::endl;

//       neighborbasisLGL_.GetBegin( neighborIdx, NO_MASK );
//       neighborbasisLGL_.GetEnd( NO_MASK );

//       neighborbasisUniform_.GetBegin( neighborIdx, NO_MASK );
//       neighborbasisUniform_.GetEnd( NO_MASK );

  neighborbasisUniform_.GetBegin( neighborIdx, NO_MASK );
  neighborbasisUniform_.GetEnd( NO_MASK );

  GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 1 )
  statusOFS << " HFX: Time for neighbor local ALBs communcation = " << timeEnd - timeSta << " [s]" << std::endl;
#endif
#if ( _DEBUGlevel_ >= 2)
  for( std::map<Index3, DblNumMat>::iterator
      mi  = neighborbasisUniform_.LocalMap().begin();
      mi != neighborbasisUniform_.LocalMap().end(); ++mi ){
      Index3 key = (*mi).first;
      DblNumMat&  LocalBasis = (*mi).second;
      statusOFS << "key " << key << std::endl;
      statusOFS << LocalBasis.Size() << " LocalBasis.m  " << LocalBasis.m() <<  " LocalBasis.n  " << LocalBasis.n() << std::endl;
//        statusOFS << " neighborbasis " << LocalBasis << std::endl;
  }
#endif

  return;
}

void
HamiltonianDG::CalculateDGHFXMatrix ( DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn>& distDMMat )
{
  Int mpirank, mpisize;
  Int numAtom = atomList_.size();
  MPI_Comm_rank( domain_.comm, &mpirank );
  MPI_Comm_size( domain_.comm, &mpisize );

  Int mpirankRow;  MPI_Comm_rank(domain_.rowComm, &mpirankRow);
  Int mpisizeRow;  MPI_Comm_size(domain_.rowComm, &mpisizeRow);

  Int mpirankCol;  MPI_Comm_rank(domain_.colComm, &mpirankCol);
  Int mpisizeCol;  MPI_Comm_size(domain_.colComm, &mpisizeCol);

  Real timeSta, timeEnd;
  Real timeStaHFX, timeEndHFX;
  Real timeSta1, timeEnd1;
//  statusOFS << " Start HFX calculations " << std::endl;

// *********************************************************************
//
// Initialize the necessary global parameters for calculating HFX matrix:
//
// *********************************************************************

  GetTime( timeStaHFX );
  GetTime( timeSta );

  // FIXME This is a definement only for integrals on uniform Grid
  // 0.5 for the HFX matrix Vum = -1/2 (uv|mn)Dvn
  // domain_.Volume() / domain_.NumGridTotalFine() for integral weight dv
  Real HFX_factor;
  if(isHFXFineGrid_ ) {
    HFX_factor = - 0.5 * 0.25 * domain_.Volume() / domain_.NumGridTotalFine();
  }
  else{
    HFX_factor = - 0.5 * 0.25 * domain_.Volume() / domain_.NumGridTotal();
  }

//  Real EhfxLocal = 0.0;
//  Real EhfxLocal2 = 0.0;
//  Real Ehfx2 = 0.0; 
    // For convenience, I define global variables for local and global number of ALBs
  Int numBasisLocal = 0;
  Int numBasisTotal = 0;

    // variables for LGL and uniform (FFT) grids 
  Index3& numUniformGridExt = fftExtElem_.domain.numGridHFX;
  Index3& numUniformGrid    = numUniformGridElemHFX_;

  Int ntotUniform = numUniformGrid.prod();
  Int ntotUniformExt = numUniformGridExt.prod();
  Int ntotR2CHFX = fftExtElem_.numGridTotalR2CHFX;

#if ( _DEBUGlevel_ >= 2 )
  statusOFS << " numUniformGridExt " << numUniformGridExt << std::endl;
  statusOFS << " numUniformGrid " << numUniformGrid << std::endl;
  statusOFS << " HFX factor " << HFX_factor << std::endl;
  statusOFS << " ntotUniform " << ntotUniform << std::endl;
  statusOFS << " ntotUniformExt " << ntotUniformExt << std::endl;
  statusOFS << " ntotR2CHFX " << ntotR2CHFX << std::endl;
#endif

  // shittIdx of elements relative to the extended element
  // This variable is used in ALB pair needs to be used frequently
  // in the loop of pair potential and HFX calculations 
  Index3 shiftIdx;
  for(Int d = 0; d < DIM; d++ ){
    if( numElem_[d] > 1 ) shiftIdx[d] ++;
    shiftIdx[d] *= numUniformGrid[d];
  }

  for( Int k = 0; k < numElem_[2]; k++ )
    for( Int j = 0; j < numElem_[1]; j++ )
      for( Int i = 0; i < numElem_[0]; i++ ){
        Index3 key( i, j, k );
        if( elemPrtn_.Owner(key) == (mpirank / dmRow_) ){
          DblNumMat& localBasisUniform = neighborbasisUniform_.LocalMap()[key];
          numBasisLocal = localBasisUniform.n();
          mpi::Allreduce( &numBasisLocal, &numBasisTotal, 1, MPI_SUM, domain_.rowComm );
          //statusOFS << " numBasisTotal " << numBasisTotal <<std::endl;
        }
  }

  // *********************************************************************
  // Start to evaluate the DG HFX matrix element :

  // Universal expression: 
  //               H_{Iu,Jv} = -1/2 \sum_{Im,Jn}D_{Im,Jn} ( Iu, Im | Jv, Jn )
  //                         = -1/2 \sum_{Im,Jn}D_{Im,Jn} \int [\int X_Iu(r) v(r,r') X_Im(r) dr] X_Jv(r')X_Jn(r') dr'
  //                         = -1/2 \sum_{Im,Jn}D_{Im,Jn} \int V_{Iu,Im}(r') X_Jv(r')X_Jn(r') dr'
  //                         = -1/2 \sum_{Im,Jn}D_{Im,Jn} \sum_g V_{Iu,Im}(g') X_Jv(g)X_Jn(g)

  //                         = -1/2 \sum_{Im,Jn}\sum_g [D_{Im,Jn} V_{Iu,Im}(g') X_Jv(g)X_Jn(g)]
  //                         = -1/2 \sum_{Im,Jn}\sum_g V_{Iu,Im}(g') [D_{Im,Jn} X_Jn(g)] X_Jv(g)
  //                         = -1/2 \sum_{Im,Jn}\sum_g V_{Iu,Im}(g') Y_Im(g) X_Jv(g)
  //                         = -1/2 \sum_{Im,Jn}\sum_g A_{Iu}(g) X_Jv(g)
  // Matrix expression:
  //              H  = -1/2 {V(Iu,Im,g')*[ D(Im,Jn) * X(Jn,g)]} * X(g,Jv)
  //                 = -1/2 [V * (D*X^T)] * X
  //

  //      V[R, Iu, Iv] <-- FFT--> V[G, Iu, Iv] on extended element QI.
  //       Iu, Iv on element EI,
  //       zero padding from phi(r_EI)phi(r_EI) to phi(r_QI)phi(r_QI)
  //
  //    where I = key1 ( current element ), J = key2 (neighbor elements);
  //              u, v, m, n on ALBs , r is the grid point of element, R is the grid
  //    point of extended element.

  // This requires several steps when using different strategies.
  // Two implementations are designed for local ALBs on LGL and uniform grid.
  // The current version adopts simplier implementation based on uniform grid.
  // Also, the pair potential of ALBs are calculated using two scheme
  // One is to calculate and store all Vxx[r,u,v] in advance, and the other is to
  // calculate each Vxx[r,u,v] and use it to evaluate HFX immediately.
  // The former costs memory, the latter does not( but also require to store new matrice)
  //
  // *********************************************************************

  // *********************************************************************
  //   Strategy 1  :: Calculate and Store the 3-rd potential tensor of ALB pairs
  //                  on extended Grid with FFT, then Compute HFX
  // *********************************************************************
  //  The treatment for normalization constant of ALB is different on LGL and Uniform Grids.
  //  For LGL Grid:
  //               \sum_i \phi[i] \phi[i] w[i] = 1, where w[i] is the LGL grid weight
  //  For Uniform Grid:
  //               \sum_j \phi[j] \phi[j] dv = 1, where dv = Volume/ntot
  //
  //  ALB pair density::  \rho_uv(r) = \phi_u(r) \phi_v(r)
  //
  //  To calculate pair potential with FFT : Vx,uv(r') = \int \rho_uv(r) v(r,r') dr
  //                                         \rho_uv(r) -- FFT--> rho_uv(G)
  //                                         Vx,uv (G) = 4 * PI*\rho_uv(G)/|G|^2 -- FFT--> Vx,uv(r)
  //
  //  Vx,uv(r) is the true potential withou dv.
  //


  // *********************************************************************
  //  Strage 2 :: Calculate and store the ALBs multiplied by Density matrix
  //  for (key1, key2), then calculate each element of 3-rd pair potential tensor
  //  on-the-fly on extended Grid with FFT, pair potential is not stored
  // *********************************************************************

  // This is for global ALBs multiplied by Density matrix
  // H[Iu,Jm] = \sum_Iv( V[r_J, Iu, Iv] D[Iv, Jn] X[r_J,Jn] ) X[r_J, Jm]
  //  
  // disPsi[r_J, Iv] = D[Iv, Jn] X[r_J,Jn]  key pair (I, J) is required
  //
  
  DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn>      distDpsi_;
  distDpsi_.SetComm(domain_.colComm);
  distDpsi_.Prtn()  = distDMMat.Prtn();

//   DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn>      distDpsiRow_;
//   distDpsiRow_.SetComm(domain_.colComm);
//   distDpsiRow_.Prtn()  = distDMMat.Prtn();
  // distAmat[r_J, Iu] =  \sum_Iv( V[r_J, Iu, Iv] disPsi[r_J, Iv])
  // also requires a key pair ElemMatKey(I,J)
  //
  DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn>      distAmat;
  distAmat.SetComm(domain_.colComm);
  distAmat.Prtn()  = distDMMat.Prtn();

  for( Int k = 0; k < numElem_[2]; k++ )
    for( Int j = 0; j < numElem_[1]; j++ )
      for( Int i = 0; i < numElem_[0]; i++ ){
        Index3 key1( i, j, k );
        if( elemPrtn_.Owner(key1) == (mpirank / dmRow_) ){
          DblNumMat& localBasisUniform = neighborbasisUniform_.LocalMap()[key1];
          for( std::map<Index3, DblNumMat>::iterator
            mi  = neighborbasisUniform_.LocalMap().begin();
            mi != neighborbasisUniform_.LocalMap().end(); ++mi ){
            Index3 key2 = (*mi).first;
            DblNumMat&  A_mat = distAmat.LocalMap()[ElemMatKey(key1, key2)];
            A_mat.Resize( localBasisUniform.m() , numBasisLocal );
            SetValue( A_mat, 0.0 );
          }
        }
  }

  GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 1 )
  statusOFS << " HFX: Time for ALB HFX init = " << timeEnd - timeSta << " [s]" << std::endl;
#endif
//   DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn>      distBmat;
//   distBmat.SetComm(domain_.colComm);
//   distBmat.Prtn()  = distDMMat.Prtn();

  // *********************************************************************
  //   Step 1  :: Calculate and store  DX = D[Iv, Jn] X[r_J,Jn]
  //              for eache element pair
  // *********************************************************************



 // *********************************************************************
 //   Scheme 1  :: Dpsi is stored according to the column (grid) pivot 
 //                Dpsi(Nr,Nphi) = Phi(Nr, Nb) * DM(Nb, Nb)
 // *********************************************************************

if(1){
    Real timeGemmT = 0.0;
    Real timeTrans = 0.0;
    Real timeAllreduce = 0.0;
    Real timeCopy = 0.0;
    Int  iterGemmT = 0;
    Int  iterTrans = 0;
    Int  iterAllreduce = 0;
    Int  iterCopy = 0;

    GetTime( timeSta1 );

    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
        for( Int i = 0; i < numElem_[0]; i++ ){
          Index3 key1( i, j, k );
          if( elemPrtn_.Owner(key1) == (mpirank / dmRow_) ){
//           DblNumMat& localBasisLGL = basisLGL_.LocalMap()[key];
            DblNumMat& localBasisUniform = neighborbasisUniform_.LocalMap()[key1];
            for( std::map<Index3, DblNumMat>::iterator
              mi  = neighborbasisUniform_.LocalMap().begin();
              mi != neighborbasisUniform_.LocalMap().end(); ++mi ){
              Index3 key2 = (*mi).first;
              DblNumMat&  neighborBasis = (*mi).second;
              Int ntot = neighborBasis.m();

              Int height = neighborBasis.m();
              Int width = numBasisTotal;

              Int widthBlocksize = width / mpisizeRow;
              Int heightBlocksize = height / mpisizeRow;

              Int widthLocal = widthBlocksize;
              Int heightLocal = heightBlocksize;

              if(mpirankRow < (width % mpisizeRow)){
                widthLocal = widthBlocksize + 1;
              }

              if(mpirankRow < (height % mpisizeRow)){
                heightLocal = heightBlocksize + 1;
              }

              DblNumMat neighborBasisRow( heightLocal, width );
              SetValue( neighborBasisRow, 0.0 );

              GetTime( timeSta );
              AlltoallForward (neighborBasis, neighborBasisRow, domain_.rowComm);
              GetTime( timeEnd );
              iterTrans = iterTrans + 1;
              timeTrans = timeTrans + ( timeEnd - timeSta );

            // Calculate disDpsi[r_J, Iv] = D[Iv, Jn] X[r_J,Jn] in parallel

            // global  D[Iv, Jn]
              DblNumMat& localDM = distDMMat.LocalMap()[ElemMatKey(key1, key2)];

//              statusOFS << " key 1 "<< key1 << " key2 " << key2 << std::endl;
//              statusOFS << "localDM  " << localDM.m() << " x "<< localDM.n() << std::endl;

             // local Dpsi[r_J,Jn] , row distribution
             //
              DblNumMat Y_mat_Row( heightLocal, width );
              SetValue( Y_mat_Row, 0.0 );
            // Local Dpsi[r_local,Jn] = X[r_local,Jn] D[Iv, Jn]  , row distribution by GemmT
            //
              GetTime( timeSta );
              blas::Gemm( 'N', 'T', heightLocal, width, width,
                 1.0, neighborBasisRow.Data(), heightLocal,
                 localDM.Data(), width, 0.0,
                 Y_mat_Row.Data(), heightLocal );
              GetTime( timeEnd );
              iterGemmT = iterGemmT + 1;
              timeGemmT = timeGemmT + ( timeEnd - timeSta );

            //  Local Dpsi[r_local,Jn] to  Local Dpsi[r_J,Jn_local]
              DblNumMat Y_mat_Col( height, widthLocal );
              SetValue( Y_mat_Col, 0.0 );

              GetTime( timeSta );
              AlltoallBackward (Y_mat_Row, Y_mat_Col, domain_.rowComm);
              GetTime( timeEnd );
              iterTrans = iterTrans + 1;
              timeTrans = timeTrans + ( timeEnd - timeSta );

            // Local Dpsi[r_J,Jn_local] to glabal Dpsi[r_J,Jn]
            // It needs to be optimized
            // FIXME
              DblNumMat Ytemp(height,width );
              SetValue( Ytemp, 0.0 );

              GetTime( timeSta );
              for(Int b = 0; b < widthLocal; b++ ) {
                Int Gb =  basisLGLIdx_[b];
                blas::Copy( height, Y_mat_Col.VecData(b), 1, Ytemp.VecData(Gb), 1);
              }

              GetTime( timeEnd );
              iterCopy = iterCopy + 1;
              timeCopy = timeCopy + ( timeEnd - timeSta );

              DblNumMat& localDpsi =  distDpsi_.LocalMap()[ElemMatKey(key1, key2)];
              localDpsi.Resize( height, width );
              SetValue( localDpsi, 0.0 );

              GetTime( timeSta );
              mpi::Allreduce( Ytemp.Data(), localDpsi.Data(), height*width, MPI_SUM, domain_.rowComm );
              GetTime( timeEnd );
              iterAllreduce = iterAllreduce + 1;
              timeAllreduce = timeAllreduce + ( timeEnd - timeSta );
              // Allreduce for glabal Dpsi[r_J,Jn]
            } // mi
          }  // Owner
    }  // if i

    GetTime( timeEnd1 );
#if ( _DEBUGlevel_ >= 1 )
    statusOFS << " Time for iterGemmT      = " << iterGemmT     << "  timeGemmT      = " << timeGemmT << std::endl;
    statusOFS << " Time for iterRow2Col    = " << iterTrans     << "  timeTrans      = " << timeTrans << std::endl;
    statusOFS << " Time for iterAllreduce  = " << iterAllreduce << "  timeAllreduce  = " << timeAllreduce << std::endl;
    statusOFS << " Time for iterCopy       = " << iterCopy      << "  timeCopy       = " << timeCopy << std::endl;

    statusOFS <<  " HFX: Time for Dpsi " << timeEnd1 - timeSta1 << " [s]" << std::endl;
#endif

  } // domain

 // *********************************************************************
 //   Scheme 1  :: Dpsi is stored according to the row (phi) pivot
 //                Dpsi(Nb, Nr) = DM(Nb, Nb) * Phi(Nb, Nr)
 // *********************************************************************

  if(0){
    Real timeGemmT = 0.0;
    Real timeTrans = 0.0;
    Real timeAllreduce = 0.0;
    Real timeCopy = 0.0;
   
    Int  iterGemmT = 0;
    Int  iterTrans = 0;
    Int  iterAllreduce = 0;
    Int  iterCopy = 0;

    GetTime( timeSta1 );

    Int ntotBlocksize = ntotUniform / mpisizeRow;

    IntNumVec  displs(mpisizeRow);
    IntNumVec  sendcounts(mpisizeRow);

    if ((ntotUniform % mpisizeRow) == 0) {
      for (Int i = 0; i < mpisizeRow; i++){
        displs[i] = i * ntotBlocksize * numBasisTotal;
        sendcounts[i] = ntotBlocksize * numBasisTotal;
      }
    }
    else{
      for (Int i = 0; i < mpisizeRow; i++){
        if (i < (ntotUniform % mpisizeRow)) {
         displs[i] = i * (ntotBlocksize + 1) * numBasisTotal;
         sendcounts[i] = (ntotBlocksize + 1) * numBasisTotal;
        }
        else{
          displs[i] = (ntotUniform % mpisizeRow) * (ntotBlocksize + 1) * numBasisTotal 
               + (i-(ntotUniform % mpisizeRow)) * (ntotBlocksize) * numBasisTotal;
          sendcounts[i] = ntotBlocksize * numBasisTotal;
        }
      }
    }

    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
        for( Int i = 0; i < numElem_[0]; i++ ){
          Index3 key1( i, j, k );
          if( elemPrtn_.Owner(key1) == (mpirank / dmRow_) ){
//          DblNumMat& localBasisLGL = basisLGL_.LocalMap()[key];
            DblNumMat& localBasisUniform = neighborbasisUniform_.LocalMap()[key1];

            for( std::map<Index3, DblNumMat>::iterator
              mi  = neighborbasisUniform_.LocalMap().begin();
              mi != neighborbasisUniform_.LocalMap().end(); ++mi ){
              Index3 key2 = (*mi).first;

              //statusOFS << " key 1 "<< key1 << " key2 " << key2 << std::endl;
//              statusOFS << "localDM  " << localDM.m() << " x "<< localDM.n() << std::endl;

              DblNumMat&  neighborBasis = (*mi).second;
              Int ntot = neighborBasis.m();

              //statusOFS << " neighborBasis " << neighborBasis.m() << " x "<< neighborBasis.n() << std::endl;
              Int height = neighborBasis.m();
              Int width = numBasisTotal;
       
              Int widthBlocksize = width / mpisizeRow;
              Int heightBlocksize = height / mpisizeRow;
       
              Int widthLocal = widthBlocksize;
              Int heightLocal = heightBlocksize;
       
              if(mpirankRow < (width % mpisizeRow)){
                widthLocal = widthBlocksize + 1;
              }
       
              if(mpirankRow < (height % mpisizeRow)){
                heightLocal = heightBlocksize + 1;
              }
       
              DblNumMat neighborBasisRow( heightLocal, width );
              SetValue( neighborBasisRow, 0.0 );
       
              GetTime( timeSta );
              AlltoallForward (neighborBasis, neighborBasisRow, domain_.rowComm);
              GetTime( timeEnd );
              iterTrans = iterTrans + 1;
              timeTrans = timeTrans + ( timeEnd - timeSta );
       
              // Calculate disDpsi[r_J, Iv] = D[Iv, Jn] X[r_J,Jn] in parallel

              // global  D[Iv, Jn]  
              DblNumMat& localDM = distDMMat.LocalMap()[ElemMatKey(key1, key2)];
               // local Dpsi[r_J,Jn] , row distribution
               //
              // Local Dpsi[r_local,Jn] = X[r_local,Jn] D[Iv, Jn]  , row distribution by GemmT
              //
              DblNumMat Y_mat( width, heightLocal );
              SetValue( Y_mat, 0.0 );

              GetTime( timeSta );
 
              blas::Gemm( 'N', 'T', width, heightLocal, width,
                    1.0, localDM.Data(),  width, 
                    neighborBasisRow.Data(), heightLocal, 0.0,
                    Y_mat.Data(), width );

              GetTime( timeEnd );
              iterGemmT = iterGemmT + 1;
              timeGemmT = timeGemmT + ( timeEnd - timeSta );


////////////////////////////////////////////////////////////////////////////////////////////
              DblNumMat& localDpsi =  distDpsi_.LocalMap()[ElemMatKey(key1, key2)];
              localDpsi.Resize( width, height );
              SetValue( localDpsi, 0.0 );

              GetTime( timeSta );
              MPI_Allgatherv(Y_mat.Data(), width*heightLocal, MPI_DOUBLE, localDpsi.Data(), 
               sendcounts.Data(), displs.Data(), MPI_DOUBLE, domain_.rowComm);
              GetTime( timeEnd );
              iterAllreduce = iterAllreduce + 1;
              timeAllreduce = timeAllreduce + ( timeEnd - timeSta );

              // Allreduce for glabal Dpsi[r_J,Jn]
            } // mi
          }  // Owner
    }  // if i

    GetTime( timeEnd1 );
#if ( _DEBUGlevel_ >= 1 )
    statusOFS << " Time for iterGemmT      = " << iterGemmT     << "  timeGemmT      = " << timeGemmT << std::endl;
    statusOFS << " Time for iterRow2Col    = " << iterTrans     << "  timeTrans      = " << timeTrans << std::endl;
    statusOFS << " Time for iterAllreduce  = " << iterAllreduce << "  timeAllreduce  = " << timeAllreduce << std::endl;
    statusOFS << " Time for iterCopy       = " << iterCopy      << "  timeCopy       = " << timeCopy << std::endl;
    statusOFS <<  " HFX: Time for Dpsi " << timeEnd1 - timeSta1 << " [s]" << std::endl;
#endif
  } // domain
      
    // *********************************************************************
    //   Step 2  :: Calculate and use each elemnt of 3-rd potential tensor
    //              on-the-fly on extended Grid with FFT
    // *********************************************************************

  {
    Real timeFFT = 0.0;
    Real timeTensor = 0.0;
    Real timeMap = 0.0;
    Real timeBcast = 0.0;
    Real timeCopy = 0.0;
    Real timeOther = 0.0;
 
    Int  iterFFT = 0;
    Int  iterTensor = 0;
    Int  iterMap = 0;
    Int  iterBcast = 0;
    Int  iterCopy = 0;
    Int  iterOther = 0;

    MPI_Barrier(domain_.rowComm);
    MPI_Barrier(domain_.colComm);
    MPI_Barrier(domain_.comm);

    Real EPS = 1e-14;
 
    GetTime( timeSta1 );

    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
        for( Int i = 0; i < numElem_[0]; i++ ){
          Index3 key1( i, j, k );
          if( elemPrtn_.Owner(key1) == (mpirank / dmRow_) ){
//               DblNumMat& localBasisLGL = basisLGL_.LocalMap()[key];
            DblNumMat& localBasisUniform = neighborbasisUniform_.LocalMap()[key1];
            DblNumVec basisTemp(ntotUniform);
            SetValue(basisTemp, 0.0);

            Int numBasisLocalTemp;
            for( Int iproc = 0; iproc < mpisizeRow; iproc++ ){
              if( iproc == mpirankRow )  numBasisLocalTemp = numBasisLocal;
 
              GetTime( timeSta );
              MPI_Bcast( &numBasisLocalTemp, 1, MPI_INT, iproc, domain_.rowComm );
                 
              GetTime( timeEnd );
              iterBcast = iterBcast + 1;
              timeBcast = timeBcast + ( timeEnd - timeSta );


              IntNumVec basisIdxTemp(numBasisLocalTemp);
              SetValue(basisIdxTemp, 0);
              if( iproc == mpirankRow )  basisIdxTemp = basisLGLIdx_;

              GetTime( timeSta );
              MPI_Bcast( basisIdxTemp.Data(), numBasisLocalTemp, MPI_INT, iproc, domain_.rowComm );

              GetTime( timeEnd );
              iterBcast = iterBcast + 1;
              timeBcast = timeBcast + ( timeEnd - timeSta );

              for( Int jphi = 0; jphi < numBasisLocalTemp; jphi++ ){

                Int jgphi = basisIdxTemp(jphi);
                SetValue( basisTemp, 0.0 );

                GetTime( timeSta );
                if( iproc == mpirankRow )
                {
                  Real* basisPtr1 = localBasisUniform.VecData(jphi);

                  for( Int ir = 0; ir < ntotUniform; ir++ ){
                    basisTemp(ir) = basisPtr1[ir];
                  }
                }

                GetTime( timeEnd );
                iterOther = iterOther + 1;
                timeOther = timeOther + ( timeEnd - timeSta );

                GetTime( timeSta );
                MPI_Bcast( basisTemp.Data(), ntotUniform, MPI_DOUBLE, iproc, domain_.rowComm );
                GetTime( timeEnd );
                iterBcast = iterBcast + 1;
                timeBcast = timeBcast + ( timeEnd - timeSta );

                for( Int iphi=0; iphi < numBasisLocal; iphi++ ) {
                  Int igphi = basisLGLIdx_[iphi];
                  Real* basisPtr2 = localBasisUniform.VecData(iphi);
                  SetValue(fftExtElem_.inputVecR2CHFX, 0.0);

                  GetTime( timeSta );
                  for( Int n = 0; n < numUniformGrid[2]; n++ ){
                     for( Int m = 0; m < numUniformGrid[1]; m++ ){
                         Int ptrExtElem, ptrElem;
                         ptrExtElem = shiftIdx[0] + ( shiftIdx[1] + m ) * numUniformGridExt[0] 
                              + ( shiftIdx[2] + n ) * numUniformGridExt[0] * numUniformGridExt[1];
                         ptrElem    =  m * numUniformGrid[0] +
                              n * numUniformGrid[0] * numUniformGrid[1];
                         for( Int l = 0; l < numUniformGrid[0]; l++ ){
                           fftExtElem_.inputVecR2CHFX(ptrExtElem + l) =
                              basisPtr2[ptrElem + l] * basisTemp(ptrElem + l);
                        }
                      }
                  }

                  GetTime( timeEnd );
                  iterMap = iterMap + 1;
                  timeMap = timeMap + ( timeEnd - timeSta );

                  GetTime( timeSta );
                  FFTWExecute ( fftExtElem_, fftExtElem_.forwardPlanR2CHFX );
                  GetTime( timeEnd );
                  iterFFT = iterFFT + 1;
                  timeFFT = timeFFT + ( timeEnd - timeSta );

                  // Solve the Poisson-like problem for exchange
                  GetTime( timeSta );
                  for( Int ig = 0; ig < ntotR2CHFX; ig++ ){
                    if( fftExtElem_.gkkR2CHFX(ig) > esdfParam.ecutWavefunction * 4.0 ){
                      fftExtElem_.outputVecR2CHFX(ig) = Z_ZERO;
                    }
                    else{
                      fftExtElem_.outputVecR2CHFX(ig) *= exxgkkR2CHFX_(ig);
                    }
                  }
                  GetTime( timeEnd );
                  iterOther = iterOther + 1;
                  timeOther = timeOther + ( timeEnd - timeSta );

                  GetTime( timeSta );
                  FFTWExecute ( fftExtElem_, fftExtElem_.backwardPlanR2CHFX );

                  GetTime( timeEnd );
                  iterFFT = iterFFT + 1;
                  timeFFT = timeFFT + ( timeEnd - timeSta );


                  for( std::map<Index3, DblNumMat>::iterator
                    mi  = neighborbasisUniform_.LocalMap().begin();
                    mi != neighborbasisUniform_.LocalMap().end(); ++mi ){
                    Index3 key2 = (*mi).first;
                    DblNumMat&  neighborBasis = (*mi).second;
                    Int ntot = neighborBasis.m();

                    DblNumMat& localDpsi =  distDpsi_.LocalMap()[ElemMatKey(key1, key2)];
                    DblNumMat&  A_mat = distAmat.LocalMap()[ElemMatKey(key1, key2)];
                   // DblNumMat&  B_mat = distBmat.LocalMap()[ElemMatKey(key1, key2)];
           /// Third  compute A[g, Ku] = V[g, Ku, Kv] X[g, Kv]
           /// sub1: Mapping 3-rd Vx tensor from extended-element Grid to neighbor-element Grid.

                    GetTime( timeSta );
                    Index3 shiftIdx2;
                    for( Int d = 0; d < DIM; d++ ){
                      shiftIdx2[d] = key2[d] - key1[d];
                      shiftIdx2[d] = shiftIdx2[d] - IRound( Real(shiftIdx2[d])/numElem_[d] ) * numElem_[d];
                      if( numElem_[d] > 1 ) shiftIdx2[d] ++;
                        shiftIdx2[d] *= numUniformGrid[d];
                    }
                    GetTime( timeEnd );
                    iterOther = iterOther + 1;
                    timeOther = timeOther + ( timeEnd - timeSta );

                    GetTime( timeSta );
                    Int ptrExtElem2, ptrElem2;
                    for( Int n = 0; n < numUniformGrid[2]; n++ ) {
                       for( Int m = 0; m < numUniformGrid[1]; m++ ) {
                         ptrExtElem2 = shiftIdx2[0] + ( shiftIdx2[1] + m ) * numUniformGridExt[0] +
                              ( shiftIdx2[2] + n ) * numUniformGridExt[0] * numUniformGridExt[1];
                         ptrElem2    = m * numUniformGrid[0] + n * numUniformGrid[0] * numUniformGrid[1];
                         for( Int l = 0; l < numUniformGrid[0]; l++ ){
//                           A_mat.VecData(iphi)[ptrElem2 + l] += fftExtElem_.inputVecR2CHFX(ptrExtElem2 + l)
//                                                                * localDpsi(jgphi, ptrElem2 + l);
                           A_mat.VecData(iphi)[ptrElem2 + l] += fftExtElem_.inputVecR2CHFX(ptrExtElem2 + l)
                                                                * localDpsi.VecData(jgphi)[ptrElem2 + l];
                         } // l
                       } // m
                    } // n
                    GetTime( timeEnd );
                    iterTensor = iterTensor + 1;
                    timeTensor = timeTensor + ( timeEnd - timeSta );
                  } //for mi
                } // for jphi
              }   // for iphi
            } //for proc
          }  // for elemPrtn_.Owner
    }  // for i

    MPI_Barrier(domain_.rowComm);
    MPI_Barrier(domain_.colComm);
    MPI_Barrier(domain_.comm);

    GetTime( timeEnd1 );

#if ( _DEBUGlevel_ >= 1 )
    statusOFS << " Time for iterFFT      = " << iterFFT        << "  timeFFT      = " << timeFFT << std::endl;
    statusOFS << " Time for iterTensor   = " << iterTensor     << "  timeTensor   = " << timeTensor << std::endl;
    statusOFS << " Time for iterMap      = " << iterMap        << "  timeMap      = " << timeMap << std::endl;
    statusOFS << " Time for iterBcast    = " << iterBcast      << "  timeBcast    = " << timeBcast << std::endl;
    statusOFS << " Time for iterOther    = " << iterOther      << "  timeOther    = " << timeOther << std::endl;

    statusOFS << " HFX: Time for ALB pair potential with FFT poisson solver " << timeEnd1 - timeSta1 << " [s]" << std::endl;

#endif
  
  }

if(0) {
      for( std::map<ElemMatKey, DblNumMat>::iterator
           mi  = distAmat.LocalMap().begin();
           mi != distAmat.LocalMap().end(); ++mi ){
         ElemMatKey key = (*mi).first;

         statusOFS << key.first << " --f " << key.second << std::endl;
         DblNumMat&  mat = (*mi).second;

         statusOFS <<" Amat " <<  mat << std::endl;
 
       }

}

 if(1){
     Real timeGemmT = 0.0;
     Real timeTrans = 0.0;
     Real timeAllreduce = 0.0;
     Real timeOther = 0.0;
     Real timeAxpy = 0.0;

     Int  iterGemmT = 0;
     Int  iterTrans = 0;
     Int  iterAllreduce = 0;
     Int  iterOther =0;
     Int  iterAxpy = 0;

     GetTime( timeSta1 );

     for( Int k = 0; k < numElem_[2]; k++ )
        for( Int j = 0; j < numElem_[1]; j++ )
           for( Int i = 0; i < numElem_[0]; i++ ){
               Index3 key1( i, j, k );
               if( elemPrtn_.Owner(key1) == (mpirank / dmRow_) ){
                  DblNumMat& localBasisUniform = neighborbasisUniform_.LocalMap()[key1];

                  for( std::map<Index3, DblNumMat>::iterator
                    mi  = neighborbasisUniform_.LocalMap().begin();
                    mi != neighborbasisUniform_.LocalMap().end(); ++mi ){
                    Index3 key2 = (*mi).first;
                    DblNumMat&  neighborBasis = (*mi).second;
                    Int ntot = neighborBasis.m();

                    Int height = neighborBasis.m();
                    Int width = numBasisTotal;

                    Int widthBlocksize = width / mpisizeRow;
                    Int heightBlocksize = height / mpisizeRow;

                    Int widthLocal = widthBlocksize;
                    Int heightLocal = heightBlocksize;

                    if(mpirankRow < (width % mpisizeRow)){
                      widthLocal = widthBlocksize + 1;
                    }

                    if(mpirankRow < (height % mpisizeRow)){
                      heightLocal = heightBlocksize + 1;
                    }

                    DblNumMat neighborBasisRow( heightLocal, width );
                    SetValue( neighborBasisRow, 0.0 );

                    GetTime( timeSta );
                    AlltoallForward (neighborBasis, neighborBasisRow, domain_.rowComm);
                    GetTime( timeEnd );
                    iterTrans = iterTrans + 1;
                    timeTrans = timeTrans + ( timeEnd - timeSta );

                    DblNumMat&  A_mat = distAmat.LocalMap()[ElemMatKey(key1, key2)];

                    DblNumMat A_mat_Row( heightLocal, width );
                    SetValue( A_mat_Row, 0.0 );
                    GetTime( timeSta );
                    AlltoallForward (A_mat, A_mat_Row, domain_.rowComm);
                    GetTime( timeEnd );
                    iterTrans = iterTrans + 1;
                    timeTrans = timeTrans + ( timeEnd - timeSta );

                    DblNumMat HTemp( width, width );
                    SetValue( HTemp, 0.0 );
       
                    GetTime( timeSta );
                    blas::Gemm( 'T', 'N', width, width, heightLocal,
                       HFX_factor, A_mat_Row.Data(), heightLocal,
                       neighborBasisRow.Data(), heightLocal, 0.0,
                       HTemp.Data(), width );
                    GetTime( timeEnd );
                    iterGemmT = iterGemmT + 1;
                    timeGemmT = timeGemmT + ( timeEnd - timeSta );
      
                    DblNumMat HFX_mat ( width, width ); 
                    SetValue( HFX_mat, 0.0 );
                    GetTime( timeSta );
                    MPI_Allreduce( HTemp.Data(), HFX_mat.Data(), width * width , MPI_DOUBLE, MPI_SUM, domain_.rowComm );
                    GetTime( timeEnd );
                    iterAllreduce = iterAllreduce + 1;
                    timeAllreduce = timeAllreduce + ( timeEnd - timeSta );
       
                  /// Calculate EHFX
       
#if 0
                    DblNumMat& localDM = distDMMat.LocalMap()[ElemMatKey(key1, key2)];
//                    DblNumMat& localDM2 = distDMMat.LocalMap()[ElemMatKey(key2, key1)];
                   ///============================================================================
                   //  To calculate the Fock exchange energy Ehfx= Tr( D H )
                   /// Note: Both DM and Ham are dist(key1,key2), their product by gemmm
                   /// DM(key1, key2) * Ham(key2, key1) is the diagonal block of DH
                   /// then, sum of the diagnal elements of DH is the trace. Tr(DH (key, key))
                   /// ===========================================================================
                    SetValue( HTemp, 0.0 );
                    GetTime( timeSta );
                    blas::Gemm( 'N', 'T', width, width, width,
                        1.0, HFX_mat.Data(), width,
                        localDM.Data(), width, 0.0,
                        HTemp.Data(), width );
                    GetTime( timeEnd );
                    iterGemmT = iterGemmT + 1;
                    timeGemmT = timeGemmT + ( timeEnd - timeSta );

                    
                    GetTime( timeSta );
                    for( Int a = 0; a < HTemp.m(); a++ ){
                      EhfxLocal  += 0.5 * HTemp(a,a);
                    }
                    GetTime( timeEnd );
                    iterOther = iterOther + 1;
                    timeOther = timeOther + ( timeEnd - timeSta );
#endif
 
                    GetTime( timeSta );
                    ElemMatKey matKey( key1, key2 );
                    std::map<ElemMatKey, DblNumMat>::iterator HFX_iterator =
                      HFXMat_.LocalMap().find( matKey );
                    
                    if( HFX_iterator == HFXMat_.LocalMap().end() ){
                        HFXMat_.LocalMap()[matKey] = HFX_mat;
                    }
                    else{
                       DblNumMat&  Fmat = (*HFX_iterator).second;
                       blas::Copy( Fmat.Size(), HFX_mat.Data(), 1, Fmat.Data(), 1);
                    }
#if 0
                    std::map<ElemMatKey, DblNumMat>::iterator Ham_iterator =
                      HMat_.LocalMap().find( matKey );

                    if( Ham_iterator == HMat_.LocalMap().end() ){
                        HMat_.LocalMap()[matKey] = HFX_mat;
                    }
                    else{
                       DblNumMat&  mat = (*Ham_iterator).second;
                       blas::Axpy( mat.Size(), 1.0, HFX_mat.Data(), 1,
                        mat.Data(), 1);
                    }

                    GetTime( timeEnd );
                    iterAxpy = iterAxpy + 1;
                    timeAxpy = timeAxpy + ( timeEnd - timeSta );
#endif
          } // for mi
       } // if ElemPrtn_
     } // for i

//     MPI_Barrier(domain_.comm);
//     MPI_Barrier(domain_.colComm);
//     MPI_Barrier(domain_.rowComm);

//     statusOFS << "HFX Local       = " << EhfxLocal <<   std::endl;
//     GetTime( timeSta);
//     mpi::Allreduce( &EhfxLocal, &Ehfx, 1, MPI_SUM, domain_.colComm );
//     GetTime( timeEnd );
//     iterAllreduce = iterAllreduce + 1;
//     timeAllreduce = timeAllreduce + ( timeEnd - timeSta );

//FIXME
//     Print(statusOFS, "Fock energy       = ",  Ehfx, "[au]");

     GetTime( timeEnd1 );
#if ( _DEBUGlevel_ >= 1 )
    statusOFS << " Time for iterGemmT     = " << iterGemmT      << "  timeGemmT      = " << timeGemmT << std::endl;
    statusOFS << " Time for iterRow2Col   = " << iterTrans      << "  timeTrans      = " << timeTrans << std::endl;
    statusOFS << " Time for iterAllreduce = " << iterAllreduce  << "  timeAllreduce  = " << timeAllreduce << std::endl;
    statusOFS << " Time for iterAxpy      = " << iterAxpy       << "  timeAxpy       = " << timeAxpy << std::endl;
    statusOFS << " Time for iterOther     = " << iterOther      << "  timeOther      = " << timeOther << std::endl;
    statusOFS << " HFX: Time for HFX matrix " << timeEnd1 - timeSta1 << " [s]" << std::endl;
#endif

  }

//    statusOFS << std::endl
//      <<"  Results obtained from DGHFX: " << std::endl;
//    Print(statusOFS, "Fock energy       = ",  Ehfx, "[au]");
//    statusOFS << std::endl;

    GetTime( timeEndHFX );

 #if ( _DEBUGlevel_ >= 1 )
    statusOFS <<  " HFX: Time for HFX without interElement communication " << timeEndHFX - timeStaHFX << " [s]" << std::endl;
 #endif
      // Every processor compute all index sets
      //
    // *********************************************************************
    // Collect information and combine HMat_
    //
    // When intra-element parallelization is invoked, at this stage all
    // processors in the same processor row communicator do the same job,
    // and communication is restricted to each column communicator group.
    // *********************************************************************
/*    {
    GetTime( timeSta );
    std::vector<ElemMatKey>  keyIdx;
    for( std::map<ElemMatKey, DblNumMat>::iterator
        mi  = HMat_.LocalMap().begin();
        mi != HMat_.LocalMap().end(); ++mi ){
      ElemMatKey key = (*mi).first;

      if( HMat_.Prtn().Owner(key) != (mpirank / dmRow_) ){
        keyIdx.push_back( key );
      }
    }

    // Communication
    HMat_.PutBegin( keyIdx, NO_MASK );
    HMat_.PutEnd( NO_MASK, PutMode::COMBINE );

    // Clean up
    std::vector<ElemMatKey>  eraseKey;
    for( std::map<ElemMatKey, DblNumMat>::iterator
        mi  = HMat_.LocalMap().begin();
        mi != HMat_.LocalMap().end(); ++mi ){
      ElemMatKey key = (*mi).first;
      if( HMat_.Prtn().Owner(key) != (mpirank / dmRow_) ){
        eraseKey.push_back( key );
      }
    }
    for( std::vector<ElemMatKey>::iterator vi = eraseKey.begin();
        vi != eraseKey.end(); ++vi ){
      HMat_.LocalMap().erase( *vi );
    }

  }
*/

  return ;
}         // -----  end of method HamiltonianDG::CalculateDGMatrix  -----


void
HamiltonianDG::CalculateDGHFXEnergy ( Real& Ehfx, DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn>& distDMMat, DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn>& distHFXMat )
{
  Int mpirank, mpisize;
  MPI_Comm_rank( domain_.comm, &mpirank );
  MPI_Comm_size( domain_.comm, &mpisize );

  Int mpirankRow;  MPI_Comm_rank(domain_.rowComm, &mpirankRow);
  Int mpisizeRow;  MPI_Comm_size(domain_.rowComm, &mpisizeRow);

  Int mpirankCol;  MPI_Comm_rank(domain_.colComm, &mpirankCol);
  Int mpisizeCol;  MPI_Comm_size(domain_.colComm, &mpisizeCol);

  Real EhfxLocal = 0.0;

  Ehfx = 0.0;

//  std::vector<Index3>  getKeys_list;
  for(typename std::map<ElemMatKey, DblNumMat >::iterator
    HFX_iterator = distHFXMat.LocalMap().begin();
    HFX_iterator != distHFXMat.LocalMap().end();  ++HFX_iterator)
  {
    Index3 key1 = (HFX_iterator->first).first;
    Index3 key2 = (HFX_iterator->first).second;

//    if(key2 == key1)
//      continue;
//    else
//    getKeys_list.push_back(key2);

    DblNumMat& localHFX  = distHFXMat.LocalMap()[ElemMatKey(key1, key2)];
    DblNumMat& localDM   = distDMMat.LocalMap()[ElemMatKey(key1, key2)];

//                    DblNumMat& localDM2 = distDMMat.LocalMap()[ElemMatKey(key2, key1)];
///============================================================================
//  To calculate the Fock exchange energy Ehfx= Tr( D H )
/// Note: Both DM and Ham are dist(key1,key2), their product by gemmm
/// DM(key1, key2) * Ham(key2, key1) is the diagonal block of DH
/// then, sum of the diagnal elements of DH is the trace. Tr(DH (key, key))
/// ===========================================================================

    Int width = localHFX.m();

    DblNumMat HTemp(width, width);
    SetValue( HTemp, 0.0 );

//    GetTime( timeSta );
    blas::Gemm( 'N', 'T', width, width, width,
        1.0, localHFX.Data(), width,
        localDM.Data(), width, 0.0,
        HTemp.Data(), width );
//    GetTime( timeEnd );
//    iterGemmT = iterGemmT + 1;
//    timeGemmT = timeGemmT + ( timeEnd - timeSta );

//    GetTime( timeSta );
    for( Int a = 0; a < HTemp.m(); a++ ){
      EhfxLocal  += 0.5 * HTemp(a,a);
    }
//    GetTime( timeEnd );
//    iterOther = iterOther + 1;
//    timeOther = timeOther + ( timeEnd - timeSta );


     MPI_Barrier(domain_.comm);
     MPI_Barrier(domain_.colComm);
     MPI_Barrier(domain_.rowComm);

//     GetTime( timeSta);
     mpi::Allreduce( &EhfxLocal, &Ehfx, 1, MPI_SUM, domain_.colComm );
//     GetTime( timeEnd );
//     iterAllreduce = iterAllreduce + 1;
//     timeAllreduce = timeAllreduce + ( timeEnd - timeSta );

//FIXME
//     Print(statusOFS, "Fock energy       = ",  Ehfx, "[au]");

  }

//    statusOFS << std::endl
//      <<"  Results obtained from DGHFX: " << std::endl;
//    Print(statusOFS, "DG Fock energy       = ",  Ehfx, "[au]");
//    statusOFS << std::endl;

 return;
}

void
HamiltonianDG:: DGHFX_ISDF( )
{
  Int mpirank, mpisize;
  MPI_Comm_rank( domain_.comm, &mpirank );
  MPI_Comm_size( domain_.comm, &mpisize );

  Int mpirankRow;  MPI_Comm_rank(domain_.rowComm, &mpirankRow);
  Int mpisizeRow;  MPI_Comm_size(domain_.rowComm, &mpisizeRow);

  Int mpirankCol;  MPI_Comm_rank(domain_.colComm, &mpirankCol);
  Int mpisizeCol;  MPI_Comm_size(domain_.colComm, &mpisizeCol);

  statusOFS << " HFX:: Interpolation separative density fitting " << std::endl;
  statusOFS << "       for the ALB pairs of each element. " << std::endl;

#if ( _DEBUGlevel_ >= 2 )
  statusOFS << "ISDFType_ = " << ISDFType_ << std::endl;
  statusOFS << "ISDFKmeansWFAlpha_  = " << ISDFKmeansWFAlpha_  << std::endl;
  statusOFS << "ISDFKmeansTolerance_ = " << ISDFKmeansTolerance_ << std::endl;
  statusOFS << "ISDFKmeansMaxIter_  = " << ISDFKmeansMaxIter_ << std::endl;
  statusOFS << "ISDFNumMu_    = " << ISDFNumMu_ << std::endl;
  statusOFS << "ISDFNumGaussianRandom_ = " << ISDFNumGaussianRandom_ << std::endl;
  statusOFS << "ISDFTolerance_      = " << ISDFTolerance_ << std::endl;
#endif

  Real timeSta, timeEnd;
  Real timeSta1, timeEnd1;

  DblNumTns&               LGLWeight3D = LGLWeight3D_;
  Int numBasisTotal = 0;

  DistDblNumMat    neighborAuxbasis_;
  neighborAuxbasis_.SetComm( domain_.colComm );    
  neighborAuxbasis_.Prtn()      = elemPrtn_;

  GetTime( timeSta1 );
  for( Int k = 0; k < numElem_[2]; k++ )
    for( Int j = 0; j < numElem_[1]; j++ )
      for( Int i = 0; i < numElem_[0]; i++ ){
        Index3 key(i, j, k);
        if( elemPrtn_.Owner(key) == (mpirank / dmRow_) ){
          DblNumMat& localBasisLGL = basisLGL_.LocalMap()[key];
          DblNumMat& localBasisUniform =  basisUniform_.LocalMap()[key];
          Real factor = 0.0;            
          localBasisUniform.Resize( numUniformGridElemHFX_.prod(), localBasisLGL.n() );
          SetValue(localBasisUniform, 0.0);

          GetTime( timeSta );
          if( isHFXFineGrid_ ) {
            factor = domain_.Volume() / domain_.NumGridTotalFine();
            for( Int g = 0; g < localBasisLGL.n(); g++ ){
              InterpLGLToUniform(
                numLGLGridElem_,
                numUniformGridElemHFX_,
                localBasisLGL.VecData(g),
                localBasisUniform.VecData(g) );
            }
          }
          else{
            factor = domain_.Volume() / domain_.NumGridTotal();
            for( Int g = 0; g < localBasisLGL.n(); g++ ){
              InterpLGLToUniform2(
                numLGLGridElem_,
                numUniformGridElemHFX_,
                localBasisLGL.VecData(g),
                localBasisUniform.VecData(g) );
            }
          }
          GetTime( timeEnd );

#if ( _DEBUGlevel_ >= 2 ) //  print overlap
          DblNumMat basisTemp (localBasisLGL.m(), localBasisLGL.n());
          SetValue( basisTemp, 0.0 );

          // This is the same as the FourDotProduct process.
          for( Int g = 0; g < localBasisLGL.n(); g++ ){
            Real *ptr1 = LGLWeight3D.Data();
            Real *ptr2 = localBasisLGL.VecData(g);
            Real *ptr3 = basisTemp.VecData(g);
            for( Int l = 0; l < localBasisLGL.m(); l++ ){
              *(ptr3++) = (*(ptr1++)) * (*(ptr2++)) ;
             }
          }

          DblNumMat SmatLGL(localBasisLGL.n(), localBasisLGL.n() );
          SetValue( SmatLGL, 0.0 );
          DblNumMat SmatUniform(localBasisUniform.n(), localBasisUniform.n() );
          SetValue( SmatUniform, 0.0 );

          blas::Gemm( 'T', 'N',localBasisLGL.n() , localBasisLGL.n(), localBasisLGL.m(),
                     1.0, localBasisLGL.Data(), localBasisLGL.m(),
                     basisTemp.Data(), localBasisLGL.m(), 0.0,
                     SmatLGL.Data(), localBasisLGL.n() );

          blas::Gemm( 'T', 'N',localBasisUniform.n() , localBasisUniform.n(), localBasisUniform.m(),
                     factor, localBasisUniform.Data(), localBasisUniform.m(),
                     localBasisUniform.Data(), localBasisUniform.m(), 0.0,
                     SmatUniform.Data(), localBasisUniform.n() );

          for( Int p = 0; p < localBasisLGL.n(); p++ ){
                  statusOFS << " p " << p <<  std::endl; // " q " << q << std::endl;
                  statusOFS << " SmatLGL " << SmatLGL(p,p) << " SmatUniform " <<  SmatUniform(p,p) << std::endl;
          }
#endif

          statusOFS << "ISDF: Time for LGL2Unifom ALBs = " << timeEnd - timeSta << " [s]" << std::endl;

          Int numBasisLocal = localBasisLGL.n();
          mpi::Allreduce( &numBasisLocal, &numBasisTotal, 1, MPI_SUM, domain_.rowComm );

          numMu_ = IRound( numBasisTotal * ISDFNumMu_ );

          Domain& dmElem = domainElem_(i, j, k);

          statusOFS << " ISDF: Kmeans for interpolation points " << std::endl;
          GetTime( timeSta );
          ISDF_getPoints( localBasisUniform, localBasisUniform.m(), numBasisTotal, dmElem );
          GetTime( timeEnd );
          statusOFS << "ISDF: Time for interpolation points by Kmeans = " << timeEnd - timeSta << " [s]" << std::endl;

          DblNumMat localBasisISDF( numMu_, numBasisLocal );
          SetValue(localBasisISDF, 0.0);

          for (Int nb =0; nb < numBasisLocal; nb++) {
            for (Int mu=0; mu < numMu_; mu++) {
              localBasisISDF(mu, nb) = localBasisUniform(pivQR_(mu), nb);
            }
          }

          neighborbasisISDF_.LocalMap()[key] = localBasisISDF;          

          Int numMuBlocksize = numMu_ / mpisizeRow;

          Int numMuLocal = numMuBlocksize;

          if(mpirankRow < (numMu_ % mpisizeRow)){
            numMuLocal = numMuBlocksize + 1;
          }

          DblNumMat AuxBasis(localBasisUniform.m(), numMuLocal);
          SetValue( AuxBasis, 0.0 );

          statusOFS << " ISDF: Least-square for interpolation vectors " << std::endl;
          GetTime( timeSta );
          ISDF_getBasis( localBasisUniform, AuxBasis, localBasisUniform.m(), numBasisTotal, dmElem );
          GetTime( timeEnd );
          statusOFS << "ISDF: Time for interpolation vectors = " << timeEnd - timeSta << " [s]" << std::endl;

#if ( _DEBUGlevel_ >= 2)  //debug ISDF
          Int height = localBasisUniform.m();
          Int width = numBasisTotal;
          Int width2 = numMu_;
          Int height2 = numMu_;

          Int widthBlocksize = width / mpisizeRow;
          Int widthBlocksize2 = width2 / mpisizeRow;
          Int heightBlocksize = height / mpisizeRow;
          Int heightBlocksize2 = height2 / mpisizeRow;

          Int widthLocal = widthBlocksize;
          Int widthLocal2 = widthBlocksize2;
          Int heightLocal = heightBlocksize;
          Int heightLocal2 = heightBlocksize2;
          if(mpirankRow < (width % mpisizeRow)){
            widthLocal = widthBlocksize + 1;
          }
          if(mpirankRow < (width2 % mpisizeRow)){
            widthLocal2 = widthBlocksize2 + 1;
          }
          if(mpirankRow < (height % mpisizeRow)){
            heightLocal = heightBlocksize + 1;
          }
          if(mpirankRow < (height2 % mpisizeRow)){
            heightLocal2 = heightBlocksize2 + 1;
          }
          DblNumMat psiRow(heightLocal, width);
          DblNumMat AuxBasisRow(heightLocal, width2);
          SetValue(psiRow, 0.0);
          SetValue(AuxBasisRow, 0.0);
          AlltoallForward(localBasisUniform, psiRow, domain_.rowComm);
          AlltoallForward(AuxBasis, AuxBasisRow, domain_.rowComm);

          DblNumMat psiISDFAll(height2, width);
          SetValue(psiISDFAll, 0.0);
          DblNumMat psiISDFLocal(height2, width);
          SetValue(psiISDFLocal, 0.0);

          for(Int b = 0; b < widthLocal; b++ ) {
            Int Gb =  basisLGLIdx_[b];
            blas::Copy( height2, localBasisISDF.VecData(b), 1, psiISDFLocal.VecData(Gb), 1);
          }
          mpi::Allreduce( psiISDFLocal.Data(), psiISDFAll.Data(), height2*width, MPI_SUM, domain_.rowComm );

          DblNumMat psipsiISDF(height2, width*width);
          SetValue(psipsiISDF, 0.0);
          for (Int na =0; na < width; na++) {
            for (Int nb =0; nb < width; nb++) {
              for (Int mu=0; mu < height2; mu++) {
                psipsiISDF(mu, width*na+nb) = psiISDFAll(mu, na)*psiISDFAll(mu, nb);
              }
            }
          }

          DblNumMat MatISDF(heightLocal, width*width);
          SetValue(MatISDF, 0.0);
          blas::Gemm( 'N', 'N', heightLocal, width*width, width2,
             1.0, AuxBasisRow.Data(), heightLocal,
             psipsiISDF.Data(), width2, 0.0,
             MatISDF.Data(), heightLocal );
         
          DblNumMat psipsi(heightLocal, width*width);
          SetValue(psipsi, 0.0);
          for (Int na =0; na < width; na++) {
            for (Int nb =0; nb < width; nb++) {
              for (Int ir =0; ir < heightLocal; ir++) {
                psipsi(ir, width*na+nb) = psiRow(ir, na)*psiRow(ir, nb);
                statusOFS << " psipsi " << psipsi(ir, width*na+nb) << "  MatISDF " << MatISDF(ir, width*na+nb) <<std::endl;
              }
            }
          }
         
#endif

            neighborAuxbasis_.LocalMap()[key] = AuxBasis;

          }
    }   
    GetTime( timeEnd1 );
    statusOFS << "DGHF: ISDF Time for ALB-pairs = " << timeEnd1 - timeSta1 << " [s]" << std::endl;

    // ****************************************************************************
    // Second: Comunicate local auxiliary basis between neighbor elements for each column proc
    // ****************************************************************************
    //
    // Scheme 1: Element loop
    //  This is a straightforward but reliable implementation
    //
    GetTime( timeSta );
    std::set<Index3> neighborSet;
    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
        for( Int i = 0; i < numElem_[0]; i++ ){
          Index3 key( i, j, k );
          if( elemPrtn_.Owner(key) == (mpirank / dmRow_) ){
            std::vector<Index3>   idx(3);

            for( Int d = 0; d < DIM; d++ ){
              // Previous
              if( key[d] == 0 )
                idx[0][d] = numElem_[d]-1;
              else
                idx[0][d] = key[d]-1;

              // Current
              idx[1][d] = key[d];

              //  We ignore the previous neighbor elements using the symmetry
              //  H(key1,key2) = tanspose H(key2, key1)
              //  Next
              if( key[d] == numElem_[d]-1)
                idx[2][d] = 0;
              else
                idx[2][d] = key[d] + 1;
            } // for (d)

            // Tensor product
            for( Int c = 0; c < 3; c++ )
              for( Int b = 0; b < 3; b++ )
                for( Int a = 0; a < 3; a++ ){
                  // Not the element key itself
                  if( idx[a][0] != i || idx[b][1] != j || idx[c][2] != k ){
                      neighborSet.insert( Index3( idx[a][0], idx[b][1], idx[c][2] ) );
                  }
                } // for (a)
          } // own this element
    } // for (i)

    std::vector<Index3>  neighborIdx;
    neighborIdx.insert( neighborIdx.begin(), neighborSet.begin(), neighborSet.end() );

    neighborAuxbasis_.GetBegin( neighborIdx, NO_MASK );
    neighborAuxbasis_.GetEnd( NO_MASK );

    neighborbasisISDF_.GetBegin( neighborIdx, NO_MASK );
    neighborbasisISDF_.GetEnd( NO_MASK );

    GetTime( timeEnd );

    statusOFS << "Time for neighbor local auxbasis and basisISDF communcation = " << timeEnd - timeSta << " [s]" << std::endl;

#if ( _DEBUGlevel_ >= 2)
    for( std::map<Index3, DblNumMat>::iterator
        mi  = neighborbasisISDF_.LocalMap().begin();
        mi != neighborbasisISDF_.LocalMap().end(); ++mi ){
        Index3 key = (*mi).first;
        DblNumMat&  LocalBasis = (*mi).second;
        statusOFS << "key " << key << std::endl;
        statusOFS << LocalBasis.Size() << " LocalBasis.m  " << LocalBasis.m() <<  " LocalBasis.n  " << LocalBasis.n() << std::endl;
//        statusOFS << " neighborbasis " << LocalBasis << std::endl;
    }
#endif


  {
    Real timeFFT = 0.0;
    Real timeTensor = 0.0;
    Real timeCopy = 0.0;
    Real timeOther = 0.0;
    Real timeMap = 0.0;
    Real timeGemmT = 0.0;
    Real timeTrans = 0.0;
    Real timeAllreduce = 0.0;

    Int  iterFFT = 0;
    Int  iterTensor = 0;
    Int  iterCopy = 0;
    Int  iterOther = 0;
    Int  iterGemmT = 0;
    Int  iterTrans = 0;
    Int  iterAllreduce = 0;
    Int  iterMap = 0;

    Real EPS = 1e-14;
    Real factor;

    MPI_Barrier(domain_.rowComm);
    MPI_Barrier(domain_.colComm);
    MPI_Barrier(domain_.comm);

    GetTime( timeSta1 );
    if(isHFXFineGrid_ ) {
      factor = domain_.Volume() / domain_.NumGridTotalFine();
    }
    else{
      factor = domain_.Volume() / domain_.NumGridTotal();
    }

    Index3& numUniformGridExt = fftExtElem_.domain.numGridHFX;
    Index3& numUniformGrid    = numUniformGridElemHFX_;

    Int ntotUniform = numUniformGrid.prod();
    Int ntotUniformExt = numUniformGridExt.prod();
    Int ntotR2CHFX = fftExtElem_.numGridTotalR2CHFX;

    Index3 shiftIdx;
    for(Int d = 0; d < DIM; d++ ){
      if( numElem_[d] > 1 ) shiftIdx[d] ++;
      shiftIdx[d] *= numUniformGrid[d];
    }

    DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn>      distVaux;
    distVaux.SetComm(domain_.colComm);
    distVaux.Prtn()  = HMat_.Prtn();

    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
        for( Int i = 0; i < numElem_[0]; i++ ){
          Index3 key1( i, j, k );
          if( elemPrtn_.Owner(key1) == (mpirank / dmRow_) ){
            DblNumMat& localAuxBasis = neighborAuxbasis_.LocalMap()[key1];
            Int numAuxBasisLocal = localAuxBasis.n();
            for( std::map<Index3, DblNumMat>::iterator
              mi  = neighborAuxbasis_.LocalMap().begin();
              mi != neighborAuxbasis_.LocalMap().end(); ++mi ){
              Index3 key2 = (*mi).first;
              DblNumMat&  AuxCoulMax = distAuxCoulMat_.LocalMap()[ElemMatKey(key1, key2)];
              AuxCoulMax.Resize(numMu_, numMu_);
              SetValue(AuxCoulMax, 0.0);
              DblNumMat& Vaux = distVaux.LocalMap()[ElemMatKey(key1, key2)];
              Vaux.Resize(ntotUniform, numAuxBasisLocal);
              SetValue(Vaux, 0.0);
            }
          }
    }
    GetTime( timeEnd1 );
    statusOFS << " ISDF: Initialized time for auxiliary matrix = " << timeEnd1 - timeSta1 << " [s]" << std::endl;


    GetTime( timeSta1 );    
    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
        for( Int i = 0; i < numElem_[0]; i++ ){
          Index3 key1( i, j, k );
          if( elemPrtn_.Owner(key1) == (mpirank / dmRow_) ){
//               DblNumMat& localBasisLGL = basisLGL_.LocalMap()[key];
            DblNumMat& localAuxBasis = neighborAuxbasis_.LocalMap()[key1];
            Int numAuxBasisLocal = localAuxBasis.n();

            for( Int mu = 0; mu < numAuxBasisLocal; mu++ ){
              Real* basisPtr = localAuxBasis.VecData(mu);

              SetValue(fftExtElem_.inputVecR2CHFX, 0.0);

              GetTime( timeSta );
              for( Int n = 0; n < numUniformGrid[2]; n++ ){
                 for( Int m = 0; m < numUniformGrid[1]; m++ ){
                   Int ptrExtElem, ptrElem;
                   ptrExtElem = shiftIdx[0] + ( shiftIdx[1] + m ) * numUniformGridExt[0]
                        + ( shiftIdx[2] + n ) * numUniformGridExt[0] * numUniformGridExt[1];
                   ptrElem    =  m * numUniformGrid[0] +
                        n * numUniformGrid[0] * numUniformGrid[1];
                   for( Int l = 0; l < numUniformGrid[0]; l++ ){
                      fftExtElem_.inputVecR2CHFX(ptrExtElem + l) =
                          basisPtr[ptrElem + l];
                    }
                  }
              }

              GetTime( timeEnd );
              iterMap = iterMap + 1;
              timeMap = timeMap + ( timeEnd - timeSta );

              GetTime( timeSta );
              FFTWExecute ( fftExtElem_, fftExtElem_.forwardPlanR2CHFX );
              GetTime( timeEnd );
              iterFFT = iterFFT + 1;
              timeFFT = timeFFT + ( timeEnd - timeSta );

              // Solve the Poisson-like problem for exchange
              GetTime( timeSta );
              for( Int ig = 0; ig < ntotR2CHFX; ig++ ){
                if( fftExtElem_.gkkR2CHFX(ig) > esdfParam.ecutWavefunction * 4.0 ){
                  fftExtElem_.outputVecR2CHFX(ig) = Z_ZERO;
                }
                else{
                  fftExtElem_.outputVecR2CHFX(ig) *= exxgkkR2CHFX_(ig);
                }
              }
              GetTime( timeEnd );
              iterOther = iterOther + 1;
              timeOther = timeOther + ( timeEnd - timeSta );

              GetTime( timeSta );
              FFTWExecute ( fftExtElem_, fftExtElem_.backwardPlanR2CHFX );
              GetTime( timeEnd );
              iterFFT = iterFFT + 1;
              timeFFT = timeFFT + ( timeEnd - timeSta );

              for( std::map<Index3, DblNumMat>::iterator
                mi  = neighborAuxbasis_.LocalMap().begin();
                mi != neighborAuxbasis_.LocalMap().end(); ++mi ){
                Index3 key2 = (*mi).first;
                DblNumMat&  Vaux = distVaux.LocalMap()[ElemMatKey(key1, key2)];

                GetTime( timeSta );
                Index3 shiftIdx2;
                for( Int d = 0; d < DIM; d++ ){
                  shiftIdx2[d] = key2[d] - key1[d];
                  shiftIdx2[d] = shiftIdx2[d] - IRound( Real(shiftIdx2[d])/numElem_[d] ) * numElem_[d];
                  if( numElem_[d] > 1 ) shiftIdx2[d] ++;
                    shiftIdx2[d] *= numUniformGrid[d];
                }
                GetTime( timeEnd );
                iterOther = iterOther + 1;
                timeOther = timeOther + ( timeEnd - timeSta );

                GetTime( timeSta );
                for( Int n = 0; n < numUniformGrid[2]; n++ ) {
                  for( Int m = 0; m < numUniformGrid[1]; m++ ) {
                   Int ptrExtElem2, ptrElem2;
                    ptrExtElem2 = shiftIdx2[0] + ( shiftIdx2[1] + m ) * numUniformGridExt[0] +
                        ( shiftIdx2[2] + n ) * numUniformGridExt[0] * numUniformGridExt[1];
                    ptrElem2    = m * numUniformGrid[0] + n * numUniformGrid[0] * numUniformGrid[1];
                    for( Int l = 0; l < numUniformGrid[0]; l++ ){
                      Vaux.VecData(mu)[ptrElem2 + l] = fftExtElem_.inputVecR2CHFX(ptrExtElem2 + l);
                    } // l
                  } // m
                } // n
                GetTime( timeEnd );
                iterMap = iterMap + 1;
                timeMap = timeMap + ( timeEnd - timeSta );
              } // mi

            } // for mu

            for( std::map<Index3, DblNumMat>::iterator
              mi  = neighborAuxbasis_.LocalMap().begin();
              mi != neighborAuxbasis_.LocalMap().end(); ++mi ){
              Index3 key2 = (*mi).first;
              DblNumMat&  neighborAuxBasis = (*mi).second;
              DblNumMat&  AuxCoulMax = distAuxCoulMat_.LocalMap()[ElemMatKey(key1, key2)];
              DblNumMat&  Vaux = distVaux.LocalMap()[ElemMatKey(key1, key2)];

              Int height = ntotUniform;
              Int width = numMu_;

              Int widthBlocksize = width / mpisizeRow;
              Int heightBlocksize = height / mpisizeRow;

              Int widthLocal = widthBlocksize;
              Int heightLocal = heightBlocksize;

              if(mpirankRow < (width % mpisizeRow)){
                widthLocal = widthBlocksize + 1;
              }

              if(mpirankRow < (height % mpisizeRow)){
                heightLocal = heightBlocksize + 1;
              }

              DblNumMat VauxRow( heightLocal, width );
              DblNumMat BasisAuxRow( heightLocal, width );
              SetValue( VauxRow, 0.0 );
              SetValue( BasisAuxRow, 0.0);
              GetTime( timeSta );
              AlltoallForward (Vaux, VauxRow, domain_.rowComm);
              AlltoallForward (neighborAuxBasis, BasisAuxRow, domain_.rowComm);
              GetTime( timeEnd );
              iterTrans = iterTrans + 2;
              timeTrans = timeTrans + ( timeEnd - timeSta );

              DblNumMat Mtemp( width, width );
              SetValue( Mtemp, 0.0 );
              GetTime( timeSta );
              blas::Gemm( 'T', 'N', width, width, heightLocal,
                 factor, VauxRow.Data(), heightLocal,
                 BasisAuxRow.Data(), heightLocal, 0.0,
                 Mtemp.Data(), width );
              GetTime( timeEnd );
              iterGemmT = iterGemmT + 1;
              timeGemmT = timeGemmT + ( timeEnd - timeSta );

              GetTime( timeSta );
              mpi::Allreduce( Mtemp.Data(), AuxCoulMax.Data(), width*width, MPI_SUM, domain_.rowComm );
              GetTime( timeEnd );
              iterAllreduce = iterAllreduce + 1;
              timeAllreduce = timeAllreduce + ( timeEnd - timeSta );
            }

          }  // for elemPrtn_.Owner
    }  // for i

    MPI_Barrier(domain_.rowComm);
    MPI_Barrier(domain_.colComm);
    MPI_Barrier(domain_.comm);

    GetTime( timeEnd1 );

#if ( _DEBUGlevel_ >= 1 )
    statusOFS << " Time for iterFFT        = " << iterFFT       << "  timeFFT      = " << timeFFT << std::endl;
    statusOFS << " Time for iterMap        = " << iterMap       << "  timeMap      = " << timeMap << std::endl;
    statusOFS << " Time for iterGemmT      = " << iterGemmT     << "  timeGemmT    = " << timeGemmT << std::endl;
    statusOFS << " Time for iterRow2Col    = " << iterTrans     << "  timeTrans    = " << timeTrans << std::endl;
    statusOFS << " Time for iterAllreduce  = " << iterAllreduce << "  timeAllreduce  = " << timeAllreduce << std::endl;
    statusOFS << " Time for iterOther      = " << iterOther     << "  timeOther      = " << timeOther << std::endl;
    statusOFS << " ISDF: Time for auxiliary Coulomb Matrix with FFT " << timeEnd1 - timeSta1 << " [s]" << std::endl;
#endif
  }
           
    return;

  }


void
  HamiltonianDG::CalculateDGHFXMatrix_ISDF
  ( Real& Ehfx, DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn>& distDMMat )
  {
    Int mpirank, mpisize;
//    Int numAtom = atomList_.size();
    MPI_Comm_rank( domain_.comm, &mpirank );
    MPI_Comm_size( domain_.comm, &mpisize );

    Int mpirankRow;  MPI_Comm_rank(domain_.rowComm, &mpirankRow);
    Int mpisizeRow;  MPI_Comm_size(domain_.rowComm, &mpisizeRow);

    Int mpirankCol;  MPI_Comm_rank(domain_.colComm, &mpirankCol);
    Int mpisizeCol;  MPI_Comm_size(domain_.colComm, &mpisizeCol);

    Real timeSta, timeEnd;
    Real timeStaHFX, timeEndHFX;
    statusOFS << " Start HFX ISDF calculations " << std::endl;

    GetTime( timeStaHFX );

      // FIXME This is a definement only for integrals on uniform Grid
      // 0.5 for the HFX matrix Vum = -1/2 (uv|mn)Dvn
      // domain_.Volume() / domain_.NumGridTotalFine() for integral weight dv
    Real HFX_factor = - 0.5 * 0.25;
    Real EhfxLocal = 0.0;

    Real timeSta1, timeEnd1;
    Real timeGemmN = 0.0;
    Real timeGemmT = 0.0;
    Real timeTrans = 0.0;
    Real timeAllreduce = 0.0;
    Real timeOther = 0.0;
    Real timeAxpy = 0.0;

    Int  iterGemmN = 0;
    Int  iterGemmT = 0;
    Int  iterTrans = 0;
    Int  iterAllreduce = 0;
    Int  iterOther =0;
    Int  iterAxpy = 0;

    GetTime( timeSta1 );

    for( Int k = 0; k < numElem_[2]; k++ )
       for( Int j = 0; j < numElem_[1]; j++ )
          for( Int i = 0; i < numElem_[0]; i++ ){
              Index3 key1( i, j, k );
              if( elemPrtn_.Owner(key1) == (mpirank / dmRow_) ){
                DblNumMat& localBasisISDF = neighborbasisISDF_.LocalMap()[key1];

                Int numBasisLocal = localBasisISDF.n();
                Int numBasisTotal;
                mpi::Allreduce( &numBasisLocal, &numBasisTotal, 1, MPI_SUM, domain_.rowComm );

                Int height = numMu_;
                Int width = numBasisTotal;
 
//                statusOFS << " height " << height << " width " << width << std::endl; 
                Int widthBlocksize = width / mpisizeRow;
                Int heightBlocksize = height / mpisizeRow;
  
                Int widthLocal = widthBlocksize;
                Int heightLocal = heightBlocksize;
  
                if(mpirankRow < (width % mpisizeRow)){
                  widthLocal = widthBlocksize + 1;
                }
  
                if(mpirankRow < (height % mpisizeRow)){
                  heightLocal = heightBlocksize + 1;
                }
 

                DblNumMat localBasisISDFRow( heightLocal, width );
                SetValue( localBasisISDFRow, 0.0 );
                GetTime( timeSta );
                AlltoallForward (localBasisISDF, localBasisISDFRow, domain_.rowComm);
                GetTime( timeEnd );
                iterTrans = iterTrans + 1;
                timeTrans = timeTrans + ( timeEnd - timeSta );
  
                for( std::map<Index3, DblNumMat>::iterator
                  mi  = neighborbasisISDF_.LocalMap().begin();
                  mi != neighborbasisISDF_.LocalMap().end(); ++mi ){
                  Index3 key2 = (*mi).first;
                  DblNumMat&  neighborBasisISDF = (*mi).second;

                  DblNumMat& localDM = distDMMat.LocalMap()[ElemMatKey(key1, key2)];

                  DblNumMat psiDMatRow(heightLocal, width);
                  SetValue( psiDMatRow, 0.0 );
                  GetTime( timeSta );
                  blas::Gemm( 'N', 'N', heightLocal, width, width,
                     1.0, localBasisISDFRow.Data(), heightLocal,
                     localDM.Data(), width, 0.0,
                     psiDMatRow.Data(), heightLocal );
                  GetTime( timeEnd );
                  iterGemmN = iterGemmN + 1;
                  timeGemmN = timeGemmN + ( timeEnd - timeSta );

                  DblNumMat psiDMat(height, widthLocal);
                  SetValue( psiDMat, 0.0 );

                  GetTime( timeSta );
                  AlltoallBackward (psiDMatRow, psiDMat, domain_.rowComm);
                  GetTime( timeEnd );
                  iterTrans = iterTrans + 1;
                  timeTrans = timeTrans + ( timeEnd - timeSta );

                  DblNumMat psiDpsi(height, height);
                  SetValue( psiDpsi, 0.0 );
                  GetTime( timeSta );
                  blas::Gemm( 'N', 'T', height, height, widthLocal,
                     1.0, psiDMat.Data(), height,
                     neighborBasisISDF.Data(), height, 0.0,
                     psiDpsi.Data(), height );
                  GetTime( timeEnd );
                  iterGemmT = iterGemmT + 1;
                  timeGemmT = timeGemmT + ( timeEnd - timeSta );

                  DblNumMat Maux(height, height);
                  SetValue( Maux, 0.0 );
                  GetTime( timeSta );
                  mpi::Allreduce( psiDpsi.Data(), Maux.Data(), height*height, MPI_SUM, domain_.rowComm );
                  GetTime( timeEnd );
                  iterAllreduce = iterAllreduce + 1;
                  timeAllreduce = timeAllreduce + ( timeEnd - timeSta );

                  DblNumMat& AuxCoulMat = distAuxCoulMat_.LocalMap()[ElemMatKey(key1, key2)];

//                 statusOFS << " Maux1  "<< Maux << std::endl;
//                 statusOFS << " AuxColMat " << AuxColMat << std::endl;
                  Real* MatPtr1 = Maux.Data();
                  Real* MatPtr2 = AuxCoulMat.Data();
                  Int MatSize = height*height;
                  for(Int g =0; g < MatSize; g++){
                    *(MatPtr1++) *= *(MatPtr2++);
                  }
//                 statusOFS << " Maux2  "<< Maux << std::endl;

                  DblNumMat Mtemp(height, widthLocal);
                  SetValue( Mtemp, 0.0 );
                  GetTime( timeSta );
                  blas::Gemm( 'T', 'N', height, widthLocal, height ,
                     1.0, Maux.Data(), height, 
                     localBasisISDF.Data(), height, 0.0,
                     Mtemp.Data(), height );
                  GetTime( timeEnd );
                  iterGemmT = iterGemmT + 1;
                  timeGemmT = timeGemmT + ( timeEnd - timeSta );

                  DblNumMat MtempRow(heightLocal, width);
                  SetValue( MtempRow, 0.0 );
                  GetTime( timeSta );
                  AlltoallForward (Mtemp, MtempRow, domain_.rowComm);
                  GetTime( timeEnd );
                  iterTrans = iterTrans + 1;
                  timeTrans = timeTrans + ( timeEnd - timeSta );

                  DblNumMat neighborBasisISDFRow(heightLocal, width);
                  SetValue( neighborBasisISDFRow, 0.0 );
                  GetTime( timeSta );
                  AlltoallForward (neighborBasisISDF, neighborBasisISDFRow, domain_.rowComm);
                  GetTime( timeEnd );
                  iterTrans = iterTrans + 1;
                  timeTrans = timeTrans + ( timeEnd - timeSta );

                  DblNumMat Htemp(width, width);
                  SetValue( Htemp, 0.0 );
                  GetTime( timeSta );
                  blas::Gemm( 'T', 'N', width, width, heightLocal,
                     HFX_factor, MtempRow.Data(), heightLocal,
                     neighborBasisISDFRow.Data(), heightLocal, 0.0,
                     Htemp.Data(), width );
                  GetTime( timeEnd );
                  iterGemmT = iterGemmT + 1;
                  timeGemmT = timeGemmT + ( timeEnd - timeSta );

                  DblNumMat HFX_mat ( width, width );
                  SetValue( HFX_mat, 0.0 );
                  GetTime( timeSta );
                  MPI_Allreduce( Htemp.Data(), HFX_mat.Data(), width*width , MPI_DOUBLE, MPI_SUM, domain_.rowComm );
                  GetTime( timeEnd );
                  iterAllreduce = iterAllreduce + 1;
                  timeAllreduce = timeAllreduce + ( timeEnd - timeSta );

               if(1){
                  ///============================================================================
                  //  To calculate the Fock exchange energy Ehfx= Tr( D H )
                  /// Note: Both DM and Ham are dist(key1,key2), their product by gemmm
                  /// DM(key1, key2) * Ham(key2, key1) is the diagonal block of DH
                  /// then, sum of the diagnal elements of DH is the trace. Tr(DH (key, key))
                  /// ===========================================================================
                  SetValue( Htemp, 0.0 );
                  GetTime( timeSta );
                  blas::Gemm( 'N', 'T', width, width, width,
                      1.0, HFX_mat.Data(), width,
                      localDM.Data(), width, 0.0,
                      Htemp.Data(), width );
                  GetTime( timeEnd );
                  iterGemmT = iterGemmT + 1;
                  timeGemmT = timeGemmT + ( timeEnd - timeSta );

                  GetTime( timeSta );
                  for( Int a = 0; a < Htemp.m(); a++ ){
                    EhfxLocal  += 0.5 * Htemp(a,a);
                  }
                  GetTime( timeEnd );
                  iterOther = iterOther + 1;
                  timeOther = timeOther + ( timeEnd - timeSta );
                }

                  GetTime( timeSta );
                  ElemMatKey matKey( key1, key2 );
                  std::map<ElemMatKey, DblNumMat>::iterator Ham_iterator =
                    HMat_.LocalMap().find( matKey );

                  if( Ham_iterator == HMat_.LocalMap().end() ){
                      HMat_.LocalMap()[matKey] = HFX_mat;
                  }
                  else{
                     DblNumMat&  mat = (*Ham_iterator).second;
                     blas::Axpy( mat.Size(), 1.0, HFX_mat.Data(), 1,
                      mat.Data(), 1);
                  }

                  std::map<ElemMatKey, DblNumMat>::iterator HFX_iterator =
                    HFXMat_.LocalMap().find( matKey );

                  if( HFX_iterator == HFXMat_.LocalMap().end() ){
                      HFXMat_.LocalMap()[matKey] = HFX_mat;
                  }
                  else{
                     DblNumMat&  Fmat = (*HFX_iterator).second;
                     blas::Copy( Fmat.Size(), HFX_mat.Data(), 1, Fmat.Data(), 1);
                  }

                  GetTime( timeEnd );
                  iterAxpy = iterAxpy + 1;
                  timeAxpy = timeAxpy + ( timeEnd - timeSta );

                } // mi
              }  // Prtowner
    } //for ijk
     MPI_Barrier(domain_.comm);
     MPI_Barrier(domain_.colComm);
     MPI_Barrier(domain_.rowComm);

//     statusOFS << "HFX Local       = " << EhfxLocal <<   std::endl;
     GetTime( timeSta);
     mpi::Allreduce( &EhfxLocal, &Ehfx, 1, MPI_SUM, domain_.colComm );
     GetTime( timeEnd );
     iterAllreduce = iterAllreduce + 1;
     timeAllreduce = timeAllreduce + ( timeEnd - timeSta );

//FIXME
//     Print(statusOFS, "Fock energy       = ",  Ehfx, "[au]");

     GetTime( timeEnd1 );
#if ( _DEBUGlevel_ >= 1 )
    statusOFS << " Time for iterGemmT     = " << iterGemmT      << "  timeGemmT      = " << timeGemmT << std::endl;
    statusOFS << " Time for iterRow2Col   = " << iterTrans      << "  timeTrans      = " << timeTrans << std::endl;
    statusOFS << " Time for iterAllreduce = " << iterAllreduce  << "  timeAllreduce  = " << timeAllreduce << std::endl;
    statusOFS << " Time for iterAxpy      = " << iterAxpy       << "  timeAxpy       = " << timeAxpy << std::endl;
    statusOFS << " Time for iterOther     = " << iterOther      << "  timeOther      = " << timeOther << std::endl;

    statusOFS << " HFX-ISDF: Time for HFX matrix and energy " << timeEnd1 - timeSta1 << " [s]" << std::endl;
#endif

    statusOFS << std::endl
      <<"  Results obtained from DGHFX: " << std::endl;
    Print(statusOFS, "Fock energy       = ",  Ehfx, "[au]");
    statusOFS << std::endl;

    GetTime( timeEndHFX );

 #if ( _DEBUGlevel_ >= 0 )
    statusOFS <<  " HFX-ISDF: Time for HFX without interElement communication " << timeEndHFX - timeStaHFX << " [s]" << std::endl;
 #endif

    return; 
  }

void HamiltonianDG:: ISDF_getPoints(  DblNumMat& psi, Int Ng, Int Nb, const Domain& dmElem ) 
  {
    Int mpirank, mpisize;
    MPI_Comm_rank( dmElem.comm, &mpirank );
    MPI_Comm_size( dmElem.comm, &mpisize );

 //   Int mpirankCol;  MPI_Comm_rank(domain_.colComm, &mpirankCol);
//    Int mpisizeCol;  MPI_Comm_size(domain_.colComm, &mpisizeCol);

    //statusOFS << "PSI SIZE  " <<  psi.m() << " -- " << psi.n() << std::endl;

// *localphiGRow;
    Real timeSta, timeEnd;
    Real timeSta1, timeEnd1;
// K-means -------------------------------------------------------------------------------
    if(ISDFType_ == "Kmeans"){

      GetTime( timeSta );

      GetTime(timeSta1);
      DblNumVec weight(Ng);
      Real* wp = weight.Data();
  
      DblNumVec phiW(Ng);
      SetValue(phiW,0.0);
      Real* phW = phiW.Data();
      Real* ph = psi.Data();

      for (Int j = 0; j < psi.n(); j++){
        for(Int i = 0; i < Ng; i++){
          phW[i] += ph[i+j*Ng]*ph[i+j*Ng];
        }
      }

      MPI_Barrier(dmElem.comm);
      MPI_Reduce(phW, wp, Ng, MPI_DOUBLE, MPI_SUM, 0, dmElem.comm);
      MPI_Bcast(wp, Ng, MPI_DOUBLE, 0, dmElem.comm);
      GetTime(timeEnd1);
  //statusOFS << " ISDF: Time for computing weight in Kmeans: " << timeEnd1-timeSta1 << "[s]" << std::endl << std::endl;
  
      Int rk = numMu_;
      pivQR_.Resize( Ng );
      SetValue( pivQR_, 0 ); // Important. Otherwise QRCP uses piv as initial guess
      GetTime(timeSta1);
      //statusOFS << "rk = " << rk << std::endl;
      KMEAN(Ng, weight, rk, ISDFKmeansTolerance_, ISDFKmeansMaxIter_, ISDFTolerance_, dmElem, pivQR_.Data());
      GetTime(timeEnd1);
      statusOFS << " ISDF: Time for Kmeans alone is " << timeEnd1-timeSta1 << "[s]" << std::endl << std::endl;
    }
/*    else{
    Int  I_ONE = 1,   I_ZERO = 0,  I_NEGONE = -1;
    Real D_ONE = 1.0, D_ZERO = 0.0;
    Int Basis_BlockSize = 1;
    Int Grid_BlockSize = 32;

 // -----------------------------------------------------------------
 //  Define process grid and ScaLAPACK descriptors
 // -----------------------------------------------------------------
 //  Define 1D Col BLACS contxt
 //  BLACS & ScaLAPACK 1D Column
    Int contxt1DCol;
    Int nprow1DCol, npcol1DCol, myrow1DCol, mycol1DCol, info1DCol;
    Int nrowsNgNb1DCol, ncolsNgNb1DCol, lldNgNb1DCol;
    Int desc_NgNb1DCol[9];

    nprow1DCol = 1;
    npcol1DCol = mpisize;
    Cblacs_get(0, 0, &contxt1DCol);
    Cblacs_gridinit(&contxt1DCol, "C", nprow1DCol, npcol1DCol);
    Cblacs_gridinfo(contxt1DCol, &nprow1DCol, &npcol1DCol,
                    &myrow1DCol, &mycol1DCol);

    if(contxt1DCol >= 0){
      nrowsNgNb1DCol = SCALAPACK(numroc)(&Ng, &Ng, &myrow1DCol,
                                         &I_ZERO, &nprow1DCol);
      ncolsNgNb1DCol = SCALAPACK(numroc)(&Nb, &Basis_BlockSize, &mycol1DCol,
                                         &I_ZERO, &npcol1DCol);

      lldNgNb1DCol = std::max( nrowsNgNb1DCol, 1 );
    }

    Int NbLocal = ncolsNgNb1DCol;
    statusOFS << psi.n() << " -- " <<  NbLocal << std::endl;

    SCALAPACK(descinit)(desc_NgNb1DCol, &Ng, &Nb, &Ng, &I_ONE,
                        &I_ZERO, &I_ZERO, &contxt1DCol, &lldNgNb1DCol, &info1DCol);

//--------------------------------------------------------------
      // BLACS & ScaLAPACK 1D Row 
      Int contxt1DRow;
      Int nprow1DRow, npcol1DRow, myrow1DRow, mycol1DRow, info1DRow;
      Int nrowsNgNb1DRow, ncolsNgNb1DRow, lldNgNb1DRow;
      Int desc_NgNb1DRow[9];
 
  // -----------------------------------------------------------------
  // Define process grid and ScaLAPACK descriptors
  // -----------------------------------------------------------------
  // Define 1D row BLACS contxt 
      nprow1DRow = mpisize;
      npcol1DRow = 1;
      Cblacs_get(0, 0, &contxt1DRow);
      Cblacs_gridinit(&contxt1DRow, "C", nprow1DRow, npcol1DRow);
      Cblacs_gridinfo(contxt1DRow, &nprow1DRow, &npcol1DRow,
                      &myrow1DRow, &mycol1DRow);
 
      if(contxt1DRow >= 0){
        nrowsNgNb1DRow = SCALAPACK(numroc)(&Ng, &Grid_BlockSize, &myrow1DRow,
                                           &I_ZERO, &nprow1DRow);
        ncolsNgNb1DRow = SCALAPACK(numroc)(&Nb, &Nb, &mycol1DRow,
                                           &I_ZERO, &npcol1DRow);
        lldNgNb1DRow = std::max( nrowsNgNb1DRow, 1 );
      }
 
      Int NgLocal = nrowsNgNb1DRow;
 
      SCALAPACK(descinit)(desc_NgNb1DRow,
                          &Ng, &Nb, &Grid_BlockSize, &Nb,
                          &I_ZERO, &I_ZERO, &contxt1DRow, &lldNgNb1DRow, &info1DRow);

      DblNumMat psiRow( NgLocal, Nb );
      SetValue( psiRow, 0.0 );
 
      SCALAPACK(pdgemr2d)(&Ng, &Nb, psi.Data(), &I_ONE, &I_ONE, desc_NgNb1DCol,
                          psiRow.Data(), &I_ONE, &I_ONE, desc_NgNb1DRow, &contxt1DRow );

//------------------------------------------------------------------------------
    // Step 1: Pre-compression of the wavefunctions. This uses
    // multiplication with orthonormalized random Gaussian matrices
      GetTime( timeSta1 );

      Int numPre = IRound(std::sqrt(numMu_*ISDFNumGaussianRandom_));

      if( numPre > Nb ){
        ErrorHandling("numMu is too large for interpolative separable density fitting!");
      }
      statusOFS << "Ng          = " << Ng << std::endl;
      statusOFS << "numMu         = " << numMu_ << std::endl;
      statusOFS << "numPre        = " << numPre << std::endl;
      statusOFS << "numPre*numPre = " << numPre * numPre << std::endl;

      DblNumMat localpsiGRow( NgLocal, numPre );
      SetValue( localpsiGRow, 0.0 );

      DblNumMat G(Nb, numPre);
      SetValue( G, 0.0 );

      if ( mpirank == 0 ) {
        GaussianRandom(G);
        lapack::Orth( Nb, numPre, G.Data(), Nb );
//        statusOFS << "Random projection initialzied." << std::endl << std::endl;
      }
      MPI_Bcast(G.Data(), Nb * numPre, MPI_DOUBLE, 0, dmElem.comm);

      blas::Gemm( 'N', 'N', NgLocal, numPre, Nb, 1.0,
                  psiRow.Data(), NgLocal, G.Data(), Nb, 0.0,
                  localpsiGRow.Data(), NgLocal );
      GetTime( timeEnd1 );
#if ( _DEBUGlevel_ >= 0 )
      statusOFS << "Time for localpsiGRow Orth + Gemm is " <<
        timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
#endif

    // Step 2: Pivoted QR decomposition  for the Hadamard product of
    // the compressed matrix. Transpose format for QRCP

    // NOTE: All processors should have the same ntotLocalMG

      Int m_MGTemp = numPre*numPre;
 
      DblNumMat MGCol( m_MGTemp, NgLocal );
 
      GetTime( timeSta1 );
 
      for( Int j = 0; j < numPre; j++ ){
        for( Int i = 0; i < numPre; i++ ){
          for( Int ir = 0; ir < NgLocal; ir++ ){
            MGCol(i+j*numPre,ir) = localpsiGRow(ir,i) * localpsiGRow(ir,j);
          }
        }
      }
 
      GetTime( timeEnd1 );
#if ( _DEBUGlevel_ >= 0 )
      statusOFS << "Time for computing MG from localpsiGRow is " <<
        timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
#endif

  // -----------------------------------------------------------------
  // Perform the QRCP for the Hadamard product.
  // -----------------------------------------------------------------
  // BLACS & ScaLAPACK 2D Distributions 
      Int nrowsMG1DCol, ncolsMG1DCol, lldMG1DCol;
      Int desc_MG1DCol[9];
  
      Int contxt2D;
      Int nprow2D, npcol2D, myrow2D, mycol2D, info2D;
      Int nrows2D, ncols2D, lld2D;
      Int desc_MG2D[9];
  
      Int m_MG = numPre*numPre;
      Int n_MG = Ng;
  
   // Define 1D Col BLACS contxt 
      if(contxt1DCol >= 0){
        nrowsMG1DCol = SCALAPACK(numroc)(&m_MG, &m_MG, &myrow1DCol, &I_ZERO, &nprow1DCol);
        ncolsMG1DCol = SCALAPACK(numroc)(&n_MG, &Grid_BlockSize, &mycol1DCol, &I_ZERO, &npcol1DCol);
        lldMG1DCol = std::max( nrowsMG1DCol, 1 );
      }

      SCALAPACK(descinit)(desc_MG1DCol, &m_MG, &n_MG, &m_MG, &Grid_BlockSize, &I_ZERO,
                        &I_ZERO, &contxt1DCol, &lldMG1DCol, &info1DCol);

//   printf("mpirank: %d, mb_MG1D: %d, nb_MG1D: %d\n",mpirank, mb_MG1D, nb_MG1D );

 // Define 2D BLACS contxt 
      for( Int i = std::min(mpisize, IRound(sqrt((double)(mpisize*((double)n_MG/m_MG)))));
           i <= mpisize; i++){
        npcol2D = i; nprow2D = mpisize / npcol2D;
        if( (npcol2D >= nprow2D) && (nprow2D * npcol2D == mpisize) ) break;
      }
  
      Cblacs_get(0, 0, &contxt2D);
//      Cblacs_gridinit(&contxt2D, "C", nprow2D, npcol2D);
//      Cblacs_gridinfo(contxt2D, &nprow2D, &npcol2D, &myrow2D, &mycol2D);

      IntNumVec pmap(mpisize);
      for ( Int i = 0; i < mpisize; i++ ){
        pmap[i] = i;
      }
      Cblacs_gridmap(&contxt2D, &pmap[0], nprow2D, nprow2D, npcol2D);

      Int m_MG2DBlockSize = Grid_BlockSize;
      Int n_MG2DBlockSize = Grid_BlockSize;
  
      if(contxt2D >= 0){
        nrows2D = SCALAPACK(numroc)(&m_MG, &m_MG2DBlockSize, &myrow2D,
                                    &I_ZERO, &nprow2D);
        ncols2D = SCALAPACK(numroc)(&n_MG, &n_MG2DBlockSize, &mycol2D,
                                    &I_ZERO, &npcol2D);
        lld2D = std::max( nrows2D, 1 );
      }

      Int m_MG2Local = nrows2D;
      Int n_MG2Local = ncols2D;

      IntNumVec pivQRTemp( Ng );
      
      DblNumVec tau( Ng );

      SCALAPACK(descinit)(desc_MG2D, &m_MG, &n_MG, &m_MG2DBlockSize,
                          &n_MG2DBlockSize, &I_ZERO, &I_ZERO, &contxt2D, &lld2D, &info2D);

      DblNumMat  MG2D (m_MG2Local, n_MG2Local);

      SCALAPACK(pdgemr2d)(&m_MG, &n_MG, MGCol.Data(), &I_ONE, &I_ONE, desc_MG1DCol,
          MG2D.Data(), &I_ONE, &I_ONE, desc_MG2D, &contxt1DCol );

      if(contxt2D >= 0){

        Real timeQRCP1, timeQRCP2;
        GetTime( timeQRCP1 );

        SetValue( pivQRTemp, 0 );
//        SetValue( tau, 0.0 );
        scalapack::QRCPF( m_MG, n_MG, MG2D.Data(), desc_MG2D, pivQRTemp.Data(), tau.Data() );
        GetTime( timeQRCP2 );

#if ( _DEBUGlevel_ >= 0 )
        statusOFS << "Time only for QRCP is " << timeQRCP2 - timeQRCP1 << " [s]" << std::endl << std::endl;
#endif

      }

      for( Int j = 0; j < n_MG2Local; j++ ){
        pivQR_[ (j / n_MG2DBlockSize) * n_MG2DBlockSize * npcol2D
          + mycol2D * n_MG2DBlockSize + j % n_MG2DBlockSize] = pivQRTemp[j];
      }

    if(contxt1DCol >= 0) {
      Cblacs_gridexit( contxt1DCol );
    }

      if(contxt2D >= 0) {
        Cblacs_gridexit( contxt2D );
      }

      if(contxt1DRow >= 0) {
        Cblacs_gridexit( contxt1DRow );
      }

    } //else QRCP
*/

    return ;
  }


void HamiltonianDG:: ISDF_getBasis( DblNumMat& psi, DblNumMat& Xi, Int Ng, Int Nb, const Domain& dmElem )
  {
    Int mpirank, mpisize;
    MPI_Comm_rank( dmElem.comm, &mpirank );
    MPI_Comm_size( dmElem.comm, &mpisize );

    Int mpirankRow, mpisizeRow;
    MPI_Comm_rank( domain_.rowComm, &mpirankRow );
    MPI_Comm_size( domain_.rowComm, &mpisizeRow );
    Int mpirankCol, mpisizeCol;
    MPI_Comm_rank( domain_.colComm, &mpirankCol );
    MPI_Comm_size( domain_.colComm, &mpisizeCol );

    Real timeSta, timeEnd;
    Real timeSta1, timeEnd1;

    Real timeGemmN = 0.0;
    Real timeGemmT = 0.0;
    Real timeTrans = 0.0;
    Real timeAllreduce = 0.0;
    Real timeOther = 0.0;
    Real timeCopy = 0.0;
    Real timePotrf = 0.0;
    Real timeTrsm = 0.0;


    Int  iterGemmN = 0;
    Int  iterGemmT = 0;
    Int  iterTrans = 0;
    Int  iterAllreduce = 0;
    Int  iterOther = 0;
    Int  iterCopy = 0;
    Int  iterPotrf = 0;
    Int  iterTrsm = 0;

    if(!esdfParam.DGHF_ISDF_Scalapack){

      GetTime( timeSta1 );

      IntNumVec pivMu(numMu_);
      GetTime( timeSta );
      for( Int mu = 0; mu < numMu_; mu++ ){
        pivMu(mu) = pivQR_(mu);
      }
      GetTime( timeEnd );
      iterOther = iterOther + 1;
      timeOther = timeOther + ( timeEnd - timeSta );


      Int height = Ng;
      Int width = Nb;
      Int widthMu = numMu_;
      Int heightMu = numMu_;
    
      Int widthBlocksize = width / mpisizeRow;
      Int widthBlocksizeMu = widthMu / mpisizeRow;
      Int heightBlocksize = height / mpisizeRow;
      Int heightBlocksizeMu = heightMu / mpisizeRow;

      Int widthLocal = widthBlocksize;
      Int widthLocalMu = widthBlocksizeMu;
      Int heightLocal = heightBlocksize;
      Int heightLocalMu = heightBlocksizeMu;
      if(mpirankRow < (width % mpisizeRow)){
        widthLocal = widthBlocksize + 1;
      }
      if(mpirankRow < (widthMu % mpisizeRow)){
        widthLocalMu = widthBlocksizeMu + 1;
      }
      if(mpirankRow < (height % mpisizeRow)){
        heightLocal = heightBlocksize + 1;
      }
      if(mpirankRow < (heightMu % mpisizeRow)){
        heightLocalMu = heightBlocksizeMu + 1;
      }

      DblNumMat psiISDFCol(heightMu, widthLocal);
      SetValue( psiISDFCol, 0.0 );

      GetTime( timeSta );
      for (Int k=0; k < widthLocal; k++) {
        for (Int mu=0; mu < heightMu; mu++) {
          psiISDFCol(mu, k) = psi(pivMu(mu),k);
        }
      }
      GetTime( timeEnd );
      iterOther = iterOther + 1;
      timeOther = timeOther + ( timeEnd - timeSta );

      DblNumMat psiISDFLocal(heightMu, width);
      SetValue(psiISDFLocal, 0.0);

      DblNumMat psiISDF(heightMu, width);
      SetValue(psiISDF, 0.0);

      GetTime( timeSta );
      for(Int b = 0; b < widthLocal; b++ ) {
        Int Gb =  basisLGLIdx_[b];
        blas::Copy( heightMu, psiISDFCol.VecData(b), 1, psiISDFLocal.VecData(Gb), 1);
      }
      GetTime( timeEnd );
      iterCopy = iterCopy + 1;
      timeCopy = timeCopy + ( timeEnd - timeSta );

      GetTime( timeSta );
      mpi::Allreduce( psiISDFLocal.Data(), psiISDF.Data(), width*heightMu, MPI_SUM, domain_.rowComm );
      GetTime( timeEnd );
      iterAllreduce = iterAllreduce + 1;
      timeAllreduce = timeAllreduce + ( timeEnd - timeSta );

      DblNumMat psiRow(heightLocal, width);
      SetValue( psiRow, 0.0 );
      GetTime( timeSta );
      AlltoallForward(psi, psiRow, domain_.rowComm);
      GetTime( timeEnd );
      iterTrans = iterTrans + 1;
      timeTrans = timeTrans + ( timeEnd - timeSta );

      DblNumMat PpsiMuRow(heightLocal, widthMu);
      SetValue( PpsiMuRow, 0.0 );
      GetTime( timeSta );
      blas::Gemm( 'N', 'T', heightLocal, widthMu, width,
         1.0, psiRow.Data(), heightLocal,
         psiISDF.Data(), heightMu, 0.0,
         PpsiMuRow.Data(), heightLocal );
      GetTime( timeEnd );
      iterGemmT = iterGemmT + 1;
      timeGemmT = timeGemmT + ( timeEnd - timeSta );

      Real* PpsiMuPtr = PpsiMuRow.Data();
      GetTime( timeSta );
      for( Int g = 0; g < heightLocal*widthMu; g++ ){
//           *(PMuNuPtr++) *= *(PMuNuPtr++);
           PpsiMuPtr[g] = PpsiMuPtr[g] * PpsiMuPtr[g];
      }
      GetTime( timeEnd );
      iterOther = iterOther + 1;
      timeOther = timeOther + ( timeEnd - timeSta );

      DblNumMat PMuNu(heightMu, widthMu);
      SetValue( PMuNu, 0.0 );
      GetTime( timeSta );
      blas::Gemm( 'N', 'T', heightMu, widthMu, width,
         1.0, psiISDF.Data(), heightMu,
         psiISDF.Data(), heightMu, 0.0,
         PMuNu.Data(), heightMu );
      GetTime( timeEnd );
      iterGemmT = iterGemmT + 1;
      timeGemmT = timeGemmT + ( timeEnd - timeSta );

      Real* PMuNuPtr = PMuNu.Data();
      GetTime( timeSta );
      for( Int g = 0; g < heightMu*widthMu; g++ ){
//           *(PMuNuPtr++) *= *(PMuNuPtr++);           
           PMuNuPtr[g] = PMuNuPtr[g] * PMuNuPtr[g];
      }
      GetTime( timeEnd );
      iterOther = iterOther + 1;
      timeOther = timeOther + ( timeEnd - timeSta );

  //Method 1
    {
      GetTime( timeSta );
      lapack::Potrf( 'L', heightMu, PMuNu.Data(), heightMu );
      GetTime( timeEnd );
      iterPotrf = iterPotrf + 1;
      timePotrf = timePotrf + ( timeEnd - timeSta );

      GetTime( timeSta );
      blas::Trsm('R', 'L', 'T', 'N', heightLocal, widthMu, 1.0, PMuNu.Data(), heightMu, PpsiMuRow.Data(), heightLocal);
      blas::Trsm('R', 'L', 'N', 'N', heightLocal, widthMu, 1.0, PMuNu.Data(), heightMu, PpsiMuRow.Data(), heightLocal);
      GetTime( timeEnd );
      iterTrsm = iterTrsm + 1;
      timeTrsm = timeTrsm + ( timeEnd - timeSta );
    }

      GetTime( timeSta ); 
      AlltoallBackward(PpsiMuRow, Xi, domain_.rowComm);
      GetTime( timeEnd );
      iterTrans = iterTrans + 1;
      timeTrans = timeTrans + ( timeEnd - timeSta );

      GetTime( timeEnd1 );

#if ( _DEBUGlevel_ >= 0 )
    statusOFS << " Time for iterGemmT     = " << iterGemmT      << "  timeGemmT      = " << timeGemmT << std::endl;
    statusOFS << " Time for iterPotrf     = " << iterPotrf      << "  timePotrf      = " << timePotrf << std::endl;
    statusOFS << " Time for iterTrsm      = " << iterTrsm       << "  timeTrsm       = " << timeTrsm << std::endl;
    statusOFS << " Time for iterRow2Col   = " << iterTrans      << "  timeTrans      = " << timeTrans << std::endl;
    statusOFS << " Time for iterAllreduce = " << iterAllreduce  << "  timeAllreduce  = " << timeAllreduce << std::endl;
    statusOFS << " Time for iterCopy      = " << iterCopy       << "  timeCopy       = " << timeCopy << std::endl;
    statusOFS << " Time for iterOther     = " << iterOther      << "  timeOther      = " << timeOther << std::endl;

    statusOFS << " ISDF: MPI Time for IVs " << timeEnd1 - timeSta1 << " [s]" << std::endl;
#endif

    }
    else{  //Scalapack
      
      Int  I_ONE = 1,   I_ZERO = 0,  I_NEGONE = -1;
      Real D_ONE = 1.0, D_ZERO = 0.0;

      Int Nu = numMu_ ;
      Int Basis_BlockSize = 1;
      Int Grid_BlockSize = 32;
      Real timeSta2, timeEnd2;
      Real timeTradd = 0.0;
      Int  iterTradd = 0;

      GetTime( timeSta2 );
     
  // BLACS & ScaLAPACK =================================================

      Int contxt1DRow, contxt1DCol, contxt2D;
      Int nprow1DRow, npcol1DRow, myrow1DRow, mycol1DRow, info1DRow;
      Int nprow1DCol, npcol1DCol, myrow1DCol, mycol1DCol, info1DCol;
      Int nprow2D, npcol2D, myrow2D, mycol2D, info2D;
  
      Int nrowsNgNu1DRow, ncolsNgNu1DRow, lldNgNu1DRow;
      Int nrowsNgNb1DRow, ncolsNgNb1DRow, lldNgNb1DRow;
      Int nrowsNbNu1DRow, ncolsNbNu1DRow, lldNbNu1DRow;
  
      Int nrowsNgNu1DCol, ncolsNgNu1DCol, lldNgNu1DCol;
      Int nrowsNgNb1DCol, ncolsNgNb1DCol, lldNgNb1DCol;
      Int nrowsNuNu1DCol, ncolsNuNu1DCol, lldNuNu1DCol;
  
      Int nrowsNgNu2D, ncolsNgNu2D, lldNgNu2D;
      Int nrowsNgNb2D, ncolsNgNb2D, lldNgNb2D;
      Int nrowsNuNu2D, ncolsNuNu2D, lldNuNu2D;
      Int nrowsNbNu2D, ncolsNbNu2D, lldNbNu2D;
  
      Int desc_NgNb1DRow[9];
      Int desc_NgNu1DRow[9];
      Int desc_NbNu1DRow[9];
  
      Int desc_NgNb1DCol[9];
      Int desc_NgNu1DCol[9];
      Int desc_NuNu1DCol[9];
  
      Int desc_NgNu2D[9];
      Int desc_NuNu2D[9];
      Int desc_NgNb2D[9];
      Int desc_NbNu2D[9];
/// map
//
      GetTime( timeSta );
      IntNumVec isdf_pmap(dmRow_);
  
      for ( Int pmap_iter = 0; pmap_iter < dmRow_; pmap_iter++ ){
        isdf_pmap[pmap_iter] = pmap_iter + mpirankCol*dmRow_;
      }
      // Define 1D BLACS contxts 
      nprow1DCol = 1;
      npcol1DCol = mpisize;
    
      Cblacs_get(0, 0, &contxt1DCol);
      Cblacs_gridmap(&contxt1DCol, &isdf_pmap[0], nprow1DCol, nprow1DCol, npcol1DCol );
      Cblacs_gridinfo(contxt1DCol, &nprow1DCol, &npcol1DCol, &myrow1DCol, &mycol1DCol);
  
      nprow1DRow = mpisize;
      npcol1DRow = 1;
    
      Cblacs_get(0, 0, &contxt1DRow);
      Cblacs_gridmap(&contxt1DRow, &isdf_pmap[0], nprow1DRow, nprow1DRow, npcol1DRow );
      Cblacs_gridinfo(contxt1DRow, &nprow1DRow, &npcol1DRow, &myrow1DRow, &mycol1DRow);
    
      if(contxt1DCol >= 0){
        nrowsNgNu1DCol = SCALAPACK(numroc)(&Ng, &Ng, &myrow1DCol, &I_ZERO, &nprow1DCol);
        ncolsNgNu1DCol = SCALAPACK(numroc)(&Nu, &Basis_BlockSize, &mycol1DCol, &I_ZERO, &npcol1DCol);
        lldNgNu1DCol = std::max( nrowsNgNu1DCol, 1 );
    
        nrowsNgNb1DCol = SCALAPACK(numroc)(&Ng, &Ng, &myrow1DCol, &I_ZERO, &nprow1DCol);
        ncolsNgNb1DCol = SCALAPACK(numroc)(&Nb, &Basis_BlockSize, &mycol1DCol, &I_ZERO, &npcol1DCol);
        lldNgNb1DCol = std::max( nrowsNgNb1DCol, 1 );
    
        nrowsNuNu1DCol = SCALAPACK(numroc)(&Nu, &Nu, &myrow1DCol, &I_ZERO, &nprow1DCol);
        ncolsNuNu1DCol = SCALAPACK(numroc)(&Nu, &Basis_BlockSize, &mycol1DCol, &I_ZERO, &npcol1DCol);
        lldNuNu1DCol = std::max( nrowsNuNu1DCol, 1 );
    
        SCALAPACK(descinit)(desc_NgNb1DCol, &Ng, &Nb, &Ng, &Basis_BlockSize, &I_ZERO,
                          &I_ZERO, &contxt1DCol, &lldNgNb1DCol, &info1DCol);
    
        SCALAPACK(descinit)(desc_NgNu1DCol, &Ng, &Nu, &Ng, &Basis_BlockSize, &I_ZERO,
                          &I_ZERO, &contxt1DCol, &lldNgNu1DCol, &info1DCol);
    
        SCALAPACK(descinit)(desc_NuNu1DCol, &Nu, &Nu, &Nu, &Basis_BlockSize, &I_ZERO,
                          &I_ZERO, &contxt1DCol, &lldNuNu1DCol, &info1DCol);
      }
    
      if(contxt1DRow >= 0){
        nrowsNgNu1DRow = SCALAPACK(numroc)(&Ng, &Grid_BlockSize, &myrow1DRow, &I_ZERO, &nprow1DRow);
        ncolsNgNu1DRow = SCALAPACK(numroc)(&Nu, &Nu, &mycol1DRow, &I_ZERO, &npcol1DRow);
        lldNgNu1DRow = std::max( nrowsNgNu1DRow, 1 );
    
        nrowsNgNb1DRow = SCALAPACK(numroc)(&Ng, &Grid_BlockSize, &myrow1DRow, &I_ZERO, &nprow1DRow);
        ncolsNgNb1DRow = SCALAPACK(numroc)(&Nb, &Nb, &mycol1DRow, &I_ZERO, &npcol1DRow);
        lldNgNb1DRow = std::max( nrowsNgNb1DRow, 1 );
    
        nrowsNbNu1DRow = SCALAPACK(numroc)(&Nb, &Basis_BlockSize, &myrow1DRow, &I_ZERO, &nprow1DRow);
        ncolsNbNu1DRow = SCALAPACK(numroc)(&Nu, &Nu, &mycol1DRow, &I_ZERO, &npcol1DRow);
        lldNbNu1DRow = std::max( nrowsNbNu1DRow, 1 );
    
        SCALAPACK(descinit)(desc_NgNb1DRow, &Ng, &Nb, &Grid_BlockSize, &Nb, &I_ZERO,
                          &I_ZERO, &contxt1DRow, &lldNgNb1DRow, &info1DRow);
    
        SCALAPACK(descinit)(desc_NgNu1DRow, &Ng, &Nu, &Grid_BlockSize, &Nu, &I_ZERO,
                          &I_ZERO, &contxt1DRow, &lldNgNu1DRow, &info1DRow);
    
        SCALAPACK(descinit)(desc_NbNu1DRow, &Nb, &Nu, &Basis_BlockSize, &Nu, &I_ZERO,
                          &I_ZERO, &contxt1DRow, &lldNbNu1DRow, &info1DRow);
      }
      // Define 2D BLACS contxt 
      for( Int i = IRound(sqrt((double)(mpisize))); i <= mpisize; i++){
        nprow2D = i; npcol2D = mpisize / nprow2D;
        if( (nprow2D >= npcol2D) && (nprow2D * npcol2D == mpisize) ) break;
      }
  
    
      Cblacs_get(0, 0, &contxt2D);
    //  Cblacs_gridinit(&contxt2D, "C", nprow2D, npcol2D);
      Cblacs_gridmap(&contxt2D, &isdf_pmap[0], nprow2D, nprow2D, npcol2D );
      Cblacs_gridinfo(contxt2D, &nprow2D, &npcol2D, &myrow2D, &mycol2D);
    
      Int mb2D = Grid_BlockSize;
      Int nb2D = Basis_BlockSize;
    
      // desc_NgNu2D 
      if(contxt2D >= 0){
     //   Cblacs_gridinfo(contxt2D, &nprow2D, &npcol2D, &myrow2D, &mycol2D);
        nrowsNgNu2D = SCALAPACK(numroc)(&Ng, &mb2D, &myrow2D, &I_ZERO, &nprow2D);
        ncolsNgNu2D = SCALAPACK(numroc)(&Nu, &nb2D, &mycol2D, &I_ZERO, &npcol2D);
        lldNgNu2D = std::max( nrowsNgNu2D, 1 );
    
        nrowsNgNb2D = SCALAPACK(numroc)(&Ng, &mb2D, &myrow2D, &I_ZERO, &nprow2D);
        ncolsNgNb2D = SCALAPACK(numroc)(&Nb, &nb2D, &mycol2D, &I_ZERO, &npcol2D);
        lldNgNb2D = std::max( nrowsNgNb2D, 1 );
    
        nrowsNuNu2D = SCALAPACK(numroc)(&Nu, &nb2D, &myrow2D, &I_ZERO, &nprow2D);
        ncolsNuNu2D = SCALAPACK(numroc)(&Nu, &nb2D, &mycol2D, &I_ZERO, &npcol2D);
        lldNuNu2D = std::max( nrowsNuNu2D, 1 );
    
        nrowsNbNu2D = SCALAPACK(numroc)(&Nb, &nb2D, &myrow2D, &I_ZERO, &nprow2D);
        ncolsNbNu2D = SCALAPACK(numroc)(&Nu, &nb2D, &mycol2D, &I_ZERO, &npcol2D);
        lldNbNu2D = std::max( nrowsNbNu2D, 1 );
    
        SCALAPACK(descinit)(desc_NgNb2D, &Ng, &Nb, &mb2D, &nb2D, &I_ZERO,
                          &I_ZERO, &contxt2D, &lldNgNb2D, &info2D);
    
        SCALAPACK(descinit)(desc_NgNu2D, &Ng, &Nu, &mb2D, &nb2D, &I_ZERO,
                          &I_ZERO, &contxt2D, &lldNgNu2D, &info2D);
    
        SCALAPACK(descinit)(desc_NuNu2D, &Nu, &Nu, &nb2D, &nb2D, &I_ZERO,
                          &I_ZERO, &contxt2D, &lldNuNu2D, &info2D);
    
        SCALAPACK(descinit)(desc_NbNu2D, &Nb, &Nu, &nb2D, &nb2D, &I_ZERO,
                          &I_ZERO, &contxt2D, &lldNbNu2D, &info2D);
      }
      GetTime( timeEnd );
      iterOther = iterOther + 1;
      timeOther = timeOther + ( timeEnd - timeSta );

/// ==============================================================================
      IntNumVec pivMu1(numMu_);
      GetTime( timeSta );
      for( Int mu = 0; mu < numMu_; mu++ ){
        pivMu1(mu) = pivQR_(mu);
      }
      GetTime( timeEnd );
      iterOther = iterOther + 1;
      timeOther = timeOther + ( timeEnd - timeSta );

      GetTime( timeSta1 );
  
      Int NbLocal = ncolsNgNb1DCol;
  
      DblNumMat psiMuRow(NbLocal, numMu_);
      SetValue( psiMuRow, 0.0 );

      GetTime( timeSta );  
      for (Int k=0; k < NbLocal; k++) {
        for (Int mu=0; mu < numMu_; mu++) {
          psiMuRow(k, mu) = psi(pivMu1(mu),k);
        }
      }
      GetTime( timeEnd );
      iterOther = iterOther + 1;
      timeOther = timeOther + ( timeEnd - timeSta );
  
      DblNumMat psiMu2D(nrowsNbNu2D, ncolsNbNu2D);
      SetValue( psiMu2D, 0.0 );
      GetTime( timeSta );
      SCALAPACK(pdgemr2d)(&Nb, &Nu, psiMuRow.Data(), &I_ONE, &I_ONE, desc_NbNu1DRow,
          psiMu2D.Data(), &I_ONE, &I_ONE, desc_NbNu2D, &contxt1DCol );
      GetTime( timeEnd );
      iterTrans = iterTrans + 1;
      timeTrans = timeTrans + ( timeEnd - timeSta );
  
      GetTime( timeEnd1 );

#if ( _DEBUGlevel_ >= 2 )
      statusOFS << "Time for computing psiMu2D is " <<
        timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
#endif

      GetTime( timeSta1 );
  
      DblNumMat psi2D(nrowsNgNb2D, ncolsNgNb2D);
      SetValue( psi2D, 0.0 );

      GetTime( timeSta );  
      SCALAPACK(pdgemr2d)(&Ng, &Nb, psi.Data(), &I_ONE, &I_ONE, desc_NgNb1DCol,
          psi2D.Data(), &I_ONE, &I_ONE, desc_NgNb2D, &contxt1DCol );
      GetTime( timeEnd );
      iterTrans = iterTrans + 1;
      timeTrans = timeTrans + ( timeEnd - timeSta );

    
      DblNumMat PpsiMu2D(nrowsNgNu2D, ncolsNgNu2D);
      SetValue( PpsiMu2D, 0.0 );
      GetTime( timeSta );
      SCALAPACK(pdgemm)("N", "N", &Ng, &Nu, &Nb,
          &D_ONE,
          psi2D.Data(), &I_ONE, &I_ONE, desc_NgNb2D,
          psiMu2D.Data(), &I_ONE, &I_ONE, desc_NbNu2D,
          &D_ZERO,
          PpsiMu2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D);
      GetTime( timeEnd );
      iterGemmN = iterGemmN + 1;
      timeGemmN = timeGemmN + ( timeEnd - timeSta );
 
      GetTime( timeEnd1 );

#if ( _DEBUGlevel_ >= 2 )
      statusOFS << "Time for PpsiMu GEMM is " <<
        timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
#endif

      GetTime( timeSta1 );  
      DblNumMat Xi2D(nrowsNgNu2D, ncolsNgNu2D);
      SetValue( Xi2D, 0.0 );
  
      Real* Xi2DPtr = Xi2D.Data();
      Real* PpsiMu2DPtr = PpsiMu2D.Data();
      GetTime( timeSta );  
      for( Int g = 0; g < nrowsNgNu2D * ncolsNgNu2D; g++ ){
        Xi2DPtr[g] = PpsiMu2DPtr[g] * PpsiMu2DPtr[g];
      }
      GetTime( timeEnd );
      iterOther = iterOther + 1;
      timeOther = timeOther + ( timeEnd - timeSta );
 
//    statusOFS << " PpsiMu:  " << Xi2D << std::endl;

      DblNumMat Xi1D(nrowsNgNu1DCol, ncolsNuNu1DCol);
      SetValue( Xi1D, 0.0 );
      GetTime( timeSta );  
      SCALAPACK(pdgemr2d)(&Ng, &Nu, Xi2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D,
          Xi1D.Data(), &I_ONE, &I_ONE, desc_NgNu1DCol, &contxt2D );
      GetTime( timeEnd );
      iterTrans = iterTrans + 1;
      timeTrans = timeTrans + ( timeEnd - timeSta );

      DblNumMat PMuNu1D(nrowsNuNu1DCol, ncolsNuNu1DCol);
      SetValue( PMuNu1D, 0.0 );
 
      GetTime( timeSta ); 
      for (Int mu=0; mu<nrowsNuNu1DCol; mu++) {
        for (Int nu=0; nu<ncolsNuNu1DCol; nu++) {
          PMuNu1D(mu, nu) = Xi1D(pivMu1(mu),nu);
        }
      }
      GetTime( timeEnd );
      iterOther = iterOther + 1;
      timeOther = timeOther + ( timeEnd - timeSta );

      DblNumMat PMuNu2D(nrowsNuNu2D, ncolsNuNu2D);
      SetValue( PMuNu2D, 0.0 );

      GetTime( timeSta );  
      SCALAPACK(pdgemr2d)(&Nu, &Nu, PMuNu1D.Data(), &I_ONE, &I_ONE, desc_NuNu1DCol,
          PMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D, &contxt1DCol );
      GetTime( timeEnd );
      iterTrans = iterTrans + 1;
      timeTrans = timeTrans + ( timeEnd - timeSta );
  
      GetTime( timeEnd1 );

#if ( _DEBUGlevel_ >= 2 )
      statusOFS << "Time for computing PMuNu is " <<
        timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
#endif

  //Method 1
  if(1){

      GetTime( timeSta );
      SCALAPACK(pdpotrf)("L", &Nu, PMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D, &info2D);
      GetTime( timeEnd );
      iterPotrf = iterPotrf + 1;
      timePotrf = timePotrf + ( timeEnd - timeSta );

#if ( _DEBUGlevel_ >= 2 )
      statusOFS << "Time for PMuNu Potrf is " <<
        timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

      GetTime( timeSta );
      SCALAPACK(pdtrsm)("R", "L", "T", "N", &Ng, &Nu, &D_ONE,
          PMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D,
          Xi2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D);
      SCALAPACK(pdtrsm)("R", "L", "N", "N", &Ng, &Nu, &D_ONE,
          PMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D,
          Xi2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D);
       GetTime( timeEnd );
       iterTrsm = iterTrsm + 2;
       timeTrsm = timeTrsm + ( timeEnd - timeSta );

#if ( _DEBUGlevel_ >= 2 )
      statusOFS << "Time for PMuNu and Xi pdtrsm is " <<
        timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

    }

  //Method 2
    if(0){

      GetTime( timeSta );
      SCALAPACK(pdpotrf)("L", &Nu, PMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D, &info2D);
      SCALAPACK(pdpotri)("L", &Nu, PMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D, &info2D);
      GetTime( timeEnd );
      iterPotrf = iterPotrf + 2;
      timePotrf = timePotrf + ( timeEnd - timeSta );

#if ( _DEBUGlevel_ >= 2 )
      statusOFS << "Time for PMuNu Potrf is " <<
        timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

      GetTime( timeSta1 );
      DblNumMat PMuNu2DTemp(nrowsNuNu2D, ncolsNuNu2D);
      SetValue( PMuNu2DTemp, 0.0 );

      GetTime( timeSta );
      lapack::Lacpy( 'A', nrowsNuNu2D, ncolsNuNu2D, PMuNu2D.Data(), nrowsNuNu2D, PMuNu2DTemp.Data(), nrowsNuNu2D );
      GetTime( timeEnd );
      iterCopy = iterCopy + 1;
      timeCopy = timeCopy + ( timeEnd - timeSta );
  
      GetTime( timeSta );
      SCALAPACK(pdtradd)("U", "T", &Nu, &Nu,
          &D_ONE,
          PMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D,
          &D_ZERO,
          PMuNu2DTemp.Data(), &I_ONE, &I_ONE, desc_NuNu2D);
      GetTime( timeEnd );
      iterTradd = iterTradd + 1;
      timeTradd = timeTradd + ( timeEnd - timeSta );

      DblNumMat Xi2DTemp(nrowsNgNu2D, ncolsNgNu2D);
      SetValue( Xi2DTemp, 0.0 );
  
      GetTime( timeSta );
      SCALAPACK(pdgemm)("N", "N", &Ng, &Nu, &Nu,
          &D_ONE,
          Xi2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D,
          PMuNu2DTemp.Data(), &I_ONE, &I_ONE, desc_NuNu2D,
          &D_ZERO,
          Xi2DTemp.Data(), &I_ONE, &I_ONE, desc_NgNu2D);
      GetTime( timeEnd );
      iterGemmN = iterGemmN + 1;
      timeGemmN = timeGemmN + ( timeEnd - timeSta );

      GetTime( timeSta );
      SetValue( Xi2D, 0.0 );
      lapack::Lacpy( 'A', nrowsNgNu2D, ncolsNgNu2D, Xi2DTemp.Data(), nrowsNgNu2D, Xi2D.Data(), nrowsNgNu2D );
      GetTime( timeEnd );
      iterCopy = iterCopy + 1;
      timeCopy = timeCopy + ( timeEnd - timeSta );
      GetTime( timeEnd1 );
#if ( _DEBUGlevel_ >= 2 )
      statusOFS << "Time for PMuNu and Xi pdgemm is " <<
        timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
#endif

    }

      GetTime( timeSta );
      SCALAPACK(pdgemr2d)(&Ng, &Nu, Xi2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D,
         Xi.Data(), &I_ONE, &I_ONE, desc_NgNu1DCol, &contxt2D );
      GetTime( timeEnd );
      iterTrans = iterTrans + 1;
      timeTrans = timeTrans + ( timeEnd - timeSta );

      if(contxt1DCol >= 0) {
        Cblacs_gridexit( contxt1DCol );
      }

      if(contxt1DRow >= 0) {
        Cblacs_gridexit( contxt1DRow );
      }

      if(contxt2D >= 0) {
        Cblacs_gridexit( contxt2D );
      }
      GetTime( timeEnd2 );

#if ( _DEBUGlevel_ >= 0 )
    statusOFS << " Time for iterGemmN     = " << iterGemmN      << "  timeGemmN      = " << timeGemmN << std::endl;
    statusOFS << " Time for iterPotrf     = " << iterPotrf      << "  timePotrf      = " << timePotrf << std::endl;
    statusOFS << " Time for iterTrsm      = " << iterTrsm       << "  timeTrsm       = " << timeTrsm << std::endl;
    statusOFS << " Time for iterRow2Col   = " << iterTrans      << "  timeTrans      = " << timeTrans << std::endl;
    statusOFS << " Time for iterOther     = " << iterOther      << "  timeOther      = " << timeOther << std::endl;

    statusOFS << " ISDF: Scalapck Time for IVs " << timeEnd2 - timeSta2 << " [s]" << std::endl;
#endif

    } //Scalapack for LS

    return;
  }


void HamiltonianDG:: ISDF_KMEAN(Int n, NumVec<Real>& weight, Int& rk, Real KmeansTolerance, 
    Int KmeansMaxIter, Real DFTolerance,  const Domain &dm, Int* piv)
{
  MPI_Barrier(dm.comm);
  int mpirank; MPI_Comm_rank(dm.comm, &mpirank);
  int mpisize; MPI_Comm_size(dm.comm, &mpisize);
  
  Real timeSta, timeEnd;
  Real timeSta2, timeEnd2;
  Real timeDist=0.0;
  Real timeMin=0.0;
  Real timeComm=0.0;
  Real time0 = 0.0;

  GetTime(timeSta);
  Real* wptr = weight.Data();
  int npt;
  std::vector<int> index(n);
  double maxW = 0.0;
  if(DFTolerance > 1e-16){
    maxW = findMax(weight);
    npt = 0;
    for (int i = 0; i < n;i++){
      if (wptr[i] > DFTolerance*maxW){
        index[npt] = i;
        npt++;
      }
    }
    index.resize(npt);
  } else {
    npt = n;
    for (int i = 0; i < n; i++){
      index[i] = i;
    }
  }

  if(npt < rk){
    int k0 = 0;
    int k1 = 0;
    for (int i = 0; i < npt; i++){
      if ( i == index[k0] ){
        piv[k0] = i;
        k0 = std::min(k0+1, rk-1);
      } else {
        piv[npt+k1] = i;
        k1++;
      }
    }
    std::random_shuffle(piv+npt,piv+n);
    return;
  } 

  int nptLocal = n/mpisize;
  int res = n%mpisize;
  if (mpirank < res){
    nptLocal++;
  }
  int indexSta = mpirank*nptLocal;
  if (mpirank >= res){
    indexSta += res;
  }
  std::vector<int> indexLocal(nptLocal);
  DblNumMat GridLocal(nptLocal,3);
  Real* glptr = GridLocal.Data();
  DblNumVec weightLocal(nptLocal);
  Real* wlptr = weightLocal.Data();

  int tmp;
  double len[3];
  double dx[3];
//  int nG[3];
  Index3 nG = numUniformGridElemHFX_;

  for (int i = 0; i < 3; i++){
    len[i] = dm.length[i];
    dx[i] = len[i]/nG[i];
  }
 
//  statusOFS << " mpisize " << mpisize << std::endl; 
//  statusOFS << "nG " << nG[0] << " -- "<< nG[1] << " -- " << nG[2] << std::endl;
//  statusOFS << " len " << len[0] << " -- "<< len[1] << " -- " << len[2] << std::endl;
//  statusOFS  << " dx " << dx[0] << " -- "<< dx[1] << " -- " << dx[2] << std::endl;

  for (int i = 0; i < nptLocal; i++){
    tmp = index[indexSta+i];
    indexLocal[i] = tmp;
    wlptr[i] = wptr[tmp];
    glptr[i] = (tmp%nG[0])*dx[0];
    glptr[i+nptLocal] = (tmp%(nG[0]*nG[1])-glptr[i]/dx[0])/nG[0]*dx[1];
    glptr[i+2*nptLocal] = (tmp-glptr[i]/dx[0]-glptr[i+nptLocal]/dx[1]*nG[0])/(nG[0]*nG[1])*dx[2];
  }
  DblNumMat C(rk,3);
  Real* Cptr = C.Data();
  std::vector<int> Cind = index;
  std::vector<int> Cinit;
  Cinit.reserve(rk);
  std::random_shuffle(Cind.begin(), Cind.end());
  GetTime(timeEnd);
  statusOFS << "After Setup: " << timeEnd-timeSta << "[s]" << std::endl;

  if (piv[0]!= piv[1]){
    statusOFS << "Used previous initialization." << std::endl;
    for (int i = 0; i < rk; i++){
      if(wptr[piv[i]] > DFTolerance*maxW){
        Cinit.push_back(piv[i]);
      }
    }
    statusOFS << "Reusable pivots: " << Cinit.size() << std::endl;
    GetTime(timeEnd);
    statusOFS << "After load: " << timeEnd-timeSta << "[s]" << std::endl;
    int k = 0;
    while(Cinit.size() < rk && k < npt){
      bool flag = 1;
      int it = 0; 
      while (flag && it < Cinit.size()){
        if (Cinit[it] == Cind[k]){
          flag = 0;
        }
        it++;
      }
      if(flag){
        Cinit.push_back(Cind[k]);
      }
      k++;
    }
  } else {
    Cinit = Cind;
    Cinit.resize(rk);
  }
  GetTime(timeEnd);
  statusOFS << "After Initialization: " << timeEnd-timeSta << "[s]" << std::endl;

  for (int i = 0; i < rk; i++){
    tmp = Cinit[i];
    Cptr[i] = (tmp%nG[0])*dx[0];
    Cptr[i+rk] = (tmp%(nG[0]*nG[1])-Cptr[i]/dx[0])/nG[0]*dx[1];
    Cptr[i+2*rk] = (tmp-Cptr[i]/dx[0]-Cptr[i+rk]/dx[1]*nG[0])/(nG[0]*nG[1])*dx[2];
  }

  int s = 0;
  int flag = n;
  int flagrecv = 0;
  IntNumVec label(nptLocal);
  Int* lbptr = label.Data();
  IntNumVec last(nptLocal);
  Int* laptr = last.Data();
  DblNumVec count(rk);
  Real* cptr = count.Data();
  DblNumMat DLocal(nptLocal, rk);
  DblNumMat Crecv(rk,3);
  Real* Crptr = Crecv.Data();
  DblNumVec countrecv(rk);
  Real* crptr = countrecv.Data();

  GetTime(timeSta2);
  pdist2(GridLocal, C, DLocal);
  GetTime(timeEnd2);
  timeDist += (timeEnd2-timeSta2);
  
  GetTime(timeSta2);
  findMin(DLocal, 1, label);
  GetTime(timeEnd2);
  timeMin+=(timeEnd2-timeSta2);
  lbptr = label.Data();

  double maxF = KmeansTolerance*n;
  while (flag > maxF && s < KmeansMaxIter){
    SetValue(count, 0.0);
    SetValue(C, 0.0);
    for (int i = 0; i < nptLocal; i++){
      tmp = lbptr[i];
      cptr[tmp] += wlptr[i];
      Cptr[tmp] += wlptr[i]*glptr[i];
      Cptr[tmp+rk] += wlptr[i]*glptr[i+nptLocal];
      Cptr[tmp+2*rk] += wlptr[i]*glptr[i+2*nptLocal];
    }
    MPI_Barrier(dm.comm);
    GetTime(timeSta2);
    MPI_Reduce(cptr, crptr, rk, MPI_DOUBLE, MPI_SUM, 0, dm.comm);
    MPI_Reduce(Cptr, Crptr, rk*3, MPI_DOUBLE, MPI_SUM, 0, dm.comm);
    GetTime(timeEnd2);
    timeComm += (timeEnd2-timeSta2);

    GetTime(timeSta2);
    if (mpirank == 0){
      tmp = rk;
      for (int i = 0; i < rk; i++){
        if(crptr[i] != 0.0){
          Crptr[i] = Crptr[i]/crptr[i];
          Crptr[i+tmp] = Crptr[i+tmp]/crptr[i];
          Crptr[i+2*tmp] = Crptr[i+2*tmp]/crptr[i];
        } else {
          rk--;
          Crptr[i] = Crptr[rk];
          Crptr[i+tmp] = Crptr[rk+tmp];
          Crptr[i+2*tmp] = Crptr[rk+2*tmp];
          crptr[i] = crptr[rk];
          i--;
        }
      }
      C.Resize(rk,3);
      Cptr = C.Data();
      for (int i = 0; i < rk; i++){
        Cptr[i] = Crptr[i];
        Cptr[i+rk] = Crptr[i+tmp];
        Cptr[i+2*rk] = Crptr[i+2*tmp];
      }
    }
    GetTime(timeEnd2);
    time0 += (timeEnd2-timeSta2);

    MPI_Bcast(&rk, 1, MPI_INT, 0, dm.comm);
    if (mpirank != 0){
      C.Resize(rk,3);
      Cptr= C.Data();
    }
    GetTime(timeSta2);
    MPI_Bcast(Cptr, rk*3, MPI_DOUBLE, 0, dm.comm);
    GetTime(timeEnd2);
    timeComm += (timeEnd2-timeSta2);

    count.Resize(rk);
    GetTime(timeSta2);
    pdist2(GridLocal, C, DLocal);
    GetTime(timeEnd2);
    timeDist += (timeEnd2-timeSta2);

    last = label;
    laptr = last.Data();
    GetTime(timeSta2);
    findMin(DLocal, 1, label);
    GetTime(timeEnd2);
    timeMin +=(timeEnd2-timeSta2);
    lbptr = label.Data();
    flag = 0;
    for (int i = 0; i < label.m_; i++){
      if(laptr[i]!=lbptr[i]){
        flag++;
      }
    }
    MPI_Barrier(dm.comm);
    MPI_Reduce(&flag, &flagrecv, 1, MPI_INT, MPI_SUM, 0, dm.comm);
    MPI_Bcast(&flagrecv, 1, MPI_INT, 0, dm.comm);
    flag = flagrecv;
//    statusOFS<< flag << " ";
    s++;
  }
  statusOFS << std::endl << "Converged in " << s << " iterations." << std::endl;
  GetTime(timeEnd);
  statusOFS << "After iteration: " << timeEnd-timeSta << "[s]" << std::endl;
  IntNumVec Imin(rk);
  Int* imptr = Imin.Data();
  DblNumVec amin(rk);
  findMin(DLocal, 0, Imin, amin);
  for (int i = 0; i < rk; i++){
    imptr[i] = indexLocal[imptr[i]];
  }
  IntNumMat Iminrecv(rk, mpisize);
  Int* imrptr = Iminrecv.Data();
  DblNumMat aminrecv(rk, mpisize);
  MPI_Barrier(dm.comm);
  
  GetTime(timeSta2);
  MPI_Gather(imptr, rk, MPI_INT, imrptr, rk, MPI_INT, 0, dm.comm);
  MPI_Gather(amin.Data(), rk, MPI_DOUBLE, aminrecv.Data(), rk, MPI_DOUBLE, 0, dm.comm);
  GetTime(timeEnd2);
  timeComm += (timeEnd2-timeSta2);
  IntNumVec pivTemp(rk);
  Int* pvptr = pivTemp.Data();
  
  GetTime(timeSta2);
  if (mpirank == 0) {
    findMin(aminrecv,1,pivTemp);
    for (int i = 0; i <rk; i++){
      pvptr[i] = imrptr[i+rk*pvptr[i]];
    }
  }
  GetTime(timeEnd2);
  time0 += (timeEnd2-timeSta2);

  GetTime(timeSta2);
  MPI_Bcast(pvptr, rk, MPI_INT, 0, dm.comm);
  GetTime(timeEnd2);
  timeComm += (timeEnd2-timeSta2);

  unique(pivTemp);
  pvptr = pivTemp.Data();
  rk = pivTemp.m_;
  int k0 = 0;
  int k1 = 0;
  for (int i = 0; i < n; i++){
    if(i == pvptr[k0]){
      piv[k0] = i;
      k0 = std::min(k0+1, rk-1);
    } else {
      piv[rk+k1] = i;
      k1++;
    }
  }
  statusOFS << "Dist time: " << timeDist << "[s]" << std::endl;
  statusOFS << "Min time: " << timeMin << "[s]" << std::endl;
  statusOFS << "Comm time: " << timeComm << "[s]" << std::endl;
  statusOFS << "core0 time: " << time0 << "[s]" << std::endl;
}

/// the following parts are merged from UPFS2QSO package.
} // namespace dgdft
