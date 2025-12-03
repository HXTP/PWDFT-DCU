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
// Xinming Qin xmqin03@gmail.com
// diag_dg.cpp
//
#include  "scf_dg.hpp"
#include  "blas.hpp"
#include  "lapack.hpp"
#include  "utility.hpp"
#include  "environment.hpp"

namespace  dgdft{

using namespace dgdft::DensityComponent;
using namespace dgdft::esdf;
using namespace dgdft::scalapack;

void
SCFDG::AndersonMix    ( 
    Int             iter, 
    Real            mixStepLength,
    std::string     mixType,
    DistDblNumVec&  distvMix,
    DistDblNumVec&  distvOld,
    DistDblNumVec&  distvNew,
    DistDblNumMat&  dfMat,
    DistDblNumMat&  dvMat )
{
  Int mpirank, mpisize;
  MPI_Comm_rank( domain_.comm, &mpirank );
  MPI_Comm_size( domain_.comm, &mpisize );

  Int mpirankRow;  MPI_Comm_rank(domain_.rowComm, &mpirankRow);
  Int mpisizeRow;  MPI_Comm_size(domain_.rowComm, &mpisizeRow);
  Int mpirankCol;  MPI_Comm_rank(domain_.colComm, &mpirankCol);
  Int mpisizeCol;  MPI_Comm_size(domain_.colComm, &mpisizeCol);

  distvMix.SetComm(domain_.colComm);
  distvOld.SetComm(domain_.colComm);
  distvNew.SetComm(domain_.colComm);
  dfMat.SetComm(domain_.colComm);
  dvMat.SetComm(domain_.colComm);

  // Residual 
  DistDblNumVec distRes;
  // Optimal input potential in Anderon mixing.
  DistDblNumVec distvOpt; 
  // Optimal residual in Anderson mixing
  DistDblNumVec distResOpt; 
  // Preconditioned optimal residual in Anderson mixing
  DistDblNumVec distPrecResOpt;

  distRes.SetComm(domain_.colComm);
  distvOpt.SetComm(domain_.colComm);
  distResOpt.SetComm(domain_.colComm);
  distPrecResOpt.SetComm(domain_.colComm);

  // *********************************************************************
  // Initialize
  // *********************************************************************
  Int ntot  = hamDGPtr_->NumUniformGridElemFine().prod();

  // Number of iterations used, iter should start from 1
  Int iterused = std::min( iter-1, mixMaxDim_ ); 
  // The current position of dfMat, dvMat
  Int ipos = iter - 1 - ((iter-2)/ mixMaxDim_ ) * mixMaxDim_;
  // The next position of dfMat, dvMat
  Int inext = iter - ((iter-1)/ mixMaxDim_) * mixMaxDim_;

//  statusOFS << " iter " << iter <<std::endl;
//  statusOFS << " iterused  " << iterused << std::endl;
//  statusOFS << " ipos " << ipos << std::endl;
//  statusOFS << " inext " << inext << std::endl;

  distRes.Prtn()          = elemPrtn_;
  distvOpt.Prtn()         = elemPrtn_;
  distResOpt.Prtn()       = elemPrtn_;
  distPrecResOpt.Prtn()   = elemPrtn_;

  for( Int k = 0; k < numElem_[2]; k++ )
    for( Int j = 0; j < numElem_[1]; j++ )
      for( Int i = 0; i < numElem_[0]; i++ ){
        Index3 key( i, j, k );
        if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
          DblNumVec  emptyVec( ntot );
          SetValue( emptyVec, 0.0 );
          distRes.LocalMap()[key]        = emptyVec;
          distvOpt.LocalMap()[key]       = emptyVec;
          distResOpt.LocalMap()[key]     = emptyVec;
          distPrecResOpt.LocalMap()[key] = emptyVec;
        } // if ( own this element )
      } // for (i)

  // *********************************************************************
  // Anderson mixing
  // *********************************************************************

  for( Int k = 0; k < numElem_[2]; k++ )
    for( Int j = 0; j < numElem_[1]; j++ )
      for( Int i = 0; i < numElem_[0]; i++ ){
        Index3 key( i, j, k );
        if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
          // res(:) = vOld(:) - vNew(:) is the residual
//          distRes.LocalMap()[key] = distvOld.LocalMap()[key];
          blas::Copy( ntot, distvOld.LocalMap()[key].Data(),  1,
              distRes.LocalMap()[key].Data(), 1 );
          blas::Axpy( ntot, -1.0, distvNew.LocalMap()[key].Data(), 1, 
              distRes.LocalMap()[key].Data(), 1 );
//          distvOpt.LocalMap()[key]   = distvOld.LocalMap()[key];
          blas::Copy( ntot, distvOld.LocalMap()[key].Data(),  1,
              distvOpt.LocalMap()[key].Data(), 1 );
          distResOpt.LocalMap()[key] = distRes.LocalMap()[key];


          // dfMat(:, ipos-1) = res(:) - dfMat(:, ipos-1);
          // dvMat(:, ipos-1) = vOld(:) - dvMat(:, ipos-1);
          if( iter > 1 ){
            blas::Scal( ntot, -1.0, dfMat.LocalMap()[key].VecData(ipos-1), 1 );
            blas::Axpy( ntot, 1.0,  distRes.LocalMap()[key].Data(), 1, 
                dfMat.LocalMap()[key].VecData(ipos-1), 1 );
            blas::Scal( ntot, -1.0, dvMat.LocalMap()[key].VecData(ipos-1), 1 );
            blas::Axpy( ntot, 1.0,  distvOld.LocalMap()[key].Data(),  1, 
                dvMat.LocalMap()[key].VecData(ipos-1), 1 );
          }
        } // own this element
      } // for (i)

  // For iter == 1, Anderson mixing is the same as simple mixing. 
  if( iter > 1){

    Int nrow = iterused;

    // Normal matrix FTF = F^T * F
    DblNumMat FTFLocal( nrow, nrow ), FTF( nrow, nrow );
    SetValue( FTFLocal, 0.0 );
    SetValue( FTF, 0.0 );

    // Right hand side FTv = F^T * vout
    DblNumVec FTvLocal( nrow ), FTv( nrow );
    SetValue( FTvLocal, 0.0 );
    SetValue( FTv, 0.0 );

    // Local construction of FTF and FTv
    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
        for( Int i = 0; i < numElem_[0]; i++ ){
          Index3 key( i, j, k );
          if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
            DblNumMat& df     = dfMat.LocalMap()[key];
            DblNumVec& res    = distRes.LocalMap()[key];
            for( Int q = 0; q < nrow; q++ ){
              FTvLocal(q) += blas::Dot( ntot, df.VecData(q), 1,
                  res.Data(), 1 );

              for( Int p = q; p < nrow; p++ ){
                FTFLocal(p, q) += blas::Dot( ntot, df.VecData(p), 1, 
                    df.VecData(q), 1 );
                if( p > q )
                  FTFLocal(q,p) = FTFLocal(p,q);
              } // for (p)
            } // for (q)

          } // own this element
        } // for (i)

    // Reduce the data
    mpi::Allreduce( FTFLocal.Data(), FTF.Data(), nrow * nrow, 
        MPI_SUM, domain_.colComm );
    mpi::Allreduce( FTvLocal.Data(), FTv.Data(), nrow, 
        MPI_SUM, domain_.colComm );

    // All processors solve the least square problem

    // FIXME Magic number for pseudo-inverse
    Real rcond = 1e-12;
    Int rank;

    DblNumVec  S( nrow );

    // FTv = pinv( FTF ) * res
    lapack::SVDLeastSquare( nrow, nrow, 1, 
        FTF.Data(), nrow, FTv.Data(), nrow,
        S.Data(), rcond, &rank );

    statusOFS << "Rank of dfmat = " << rank <<
      ", rcond = " << rcond << std::endl;

    // Update vOpt, resOpt. 
    // FTv = Y^{\dagger} r as in the usual notation.
    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
        for( Int i = 0; i < numElem_[0]; i++ ){
          Index3 key( i, j, k );
          if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
            // vOpt   -= dv * FTv
            blas::Gemv('N', ntot, nrow, -1.0, dvMat.LocalMap()[key].Data(),
                ntot, FTv.Data(), 1, 1.0, 
                distvOpt.LocalMap()[key].Data(), 1 );

            // resOpt -= df * FTv
            blas::Gemv('N', ntot, nrow, -1.0, dfMat.LocalMap()[key].Data(),
                ntot, FTv.Data(), 1, 1.0, 
                distResOpt.LocalMap()[key].Data(), 1 );
          } // own this element
        } // for (i)
  } // (iter > 1)

  if( mixType == "kerker+anderson" ){
    KerkerPrecond( distPrecResOpt, distResOpt );
  }
  else if( mixType == "anderson" ){
    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
        for( Int i = 0; i < numElem_[0]; i++ ){
          Index3 key( i, j, k );
          if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
            distPrecResOpt.LocalMap()[key] = 
              distResOpt.LocalMap()[key];
          } // own this element
        } // for (i)
  }
  else{
    ErrorHandling("Invalid mixing type.");
  }

  // Update dfMat, dvMat, vMix 
  for( Int k = 0; k < numElem_[2]; k++ )
    for( Int j = 0; j < numElem_[1]; j++ )
      for( Int i = 0; i < numElem_[0]; i++ ){
        Index3 key( i, j, k );
        if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
          // dfMat(:, inext-1) = res(:)
          // dvMat(:, inext-1) = vOld(:)
          blas::Copy( ntot, distRes.LocalMap()[key].Data(), 1, 
              dfMat.LocalMap()[key].VecData(inext-1), 1 );
          blas::Copy( ntot, distvOld.LocalMap()[key].Data(),  1, 
              dvMat.LocalMap()[key].VecData(inext-1), 1 );

          // vMix(:) = vOpt(:) - mixStepLength * precRes(:)
          distvMix.LocalMap()[key] = distvOpt.LocalMap()[key];
          blas::Axpy( ntot, -mixStepLength, 
              distPrecResOpt.LocalMap()[key].Data(), 1, 
              distvMix.LocalMap()[key].Data(), 1 );
        } // own this element
  } // for (i)

  return ;
}         // -----  end of method SCFDG::AndersonMix  ----- 

void
SCFDG::AndersonMix2    ( 
    Int             iter, 
    Real            mixStepLength,
    DistDblNumVec&  distvMix,
    DistDblNumVec&  distvOld,
    DistDblNumVec&  distvNew,
    DistDblNumMat&  dfMat,
    DistDblNumMat&  dvMat )
{
  Int mpirank, mpisize;
  MPI_Comm_rank( domain_.comm, &mpirank );
  MPI_Comm_size( domain_.comm, &mpisize );

  Int mpirankRow;  MPI_Comm_rank(domain_.rowComm, &mpirankRow);
  Int mpisizeRow;  MPI_Comm_size(domain_.rowComm, &mpisizeRow);
  Int mpirankCol;  MPI_Comm_rank(domain_.colComm, &mpirankCol);
  Int mpisizeCol;  MPI_Comm_size(domain_.colComm, &mpisizeCol);

  distvMix.SetComm(domain_.colComm);
  distvOld.SetComm(domain_.colComm);
  distvNew.SetComm(domain_.colComm);
  dfMat.SetComm(domain_.colComm);
  dvMat.SetComm(domain_.colComm);

  // Residual 
  DistDblNumVec distRes;
  // Optimal input potential in Anderon mixing.
  DistDblNumVec distvRes; 
  DistDblNumVec distTemp; 
  // Optimal residual in Anderson mixing
  //
  distRes.SetComm(domain_.colComm);
  distvRes.SetComm(domain_.colComm);
  distTemp.SetComm(domain_.colComm);

  // *********************************************************************
  // Initialize
  // *********************************************************************
  Int ntot  = hamDGPtr_->NumUniformGridElemFine().prod();
//  Int numBasis = esdfParam.numALBElem(0, 0, 0);
  Int pos = ((iter - 1) % mixMaxDim_ - 1 + mixMaxDim_ ) % mixMaxDim_;
  Int next = (pos + 1) % mixMaxDim_;

//  statusOFS << " Iter " << iter <<std::endl;
//  statusOFS << " pos " << pos << std::endl;
//  statusOFS << " next " << next << std::endl;

  distRes.Prtn()          = elemPrtn_;
  distvRes.Prtn()         = elemPrtn_;
  distTemp.Prtn()         = elemPrtn_;

  for( Int k = 0; k < numElem_[2]; k++ )
    for( Int j = 0; j < numElem_[1]; j++ )
      for( Int i = 0; i < numElem_[0]; i++ ){
        Index3 key( i, j, k );
        if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
          DblNumVec  emptyVec( ntot );
          SetValue( emptyVec, 0.0 );
          distRes.LocalMap()[key]        = emptyVec;
          distvRes.LocalMap()[key]       = emptyVec;
          distTemp.LocalMap()[key]       = emptyVec;
        } // if ( own this element )
  } // for (i)

  // *********************************************************************
  // Anderson mixing
  // *********************************************************************

  for( Int k = 0; k < numElem_[2]; k++ )
    for( Int j = 0; j < numElem_[1]; j++ )
      for( Int i = 0; i < numElem_[0]; i++ ){
        Index3 key( i, j, k );
        if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
          // vRes(:) = vOld(:) - vNew(:) is the residual
          DblNumVec& vOld = distvOld.LocalMap()[key];
          DblNumVec& Temp = distTemp.LocalMap()[key];
          DblNumVec& vNew = distvNew.LocalMap()[key];
          DblNumVec&  Res = distRes.LocalMap()[key];
          DblNumVec& vRes = distvRes.LocalMap()[key];
          DblNumMat&  dv  = dvMat.LocalMap()[key];
          DblNumMat&  df  = dfMat.LocalMap()[key];
//          statusOFS << " vOld " << vOld << std::endl;
//          statusOFS << " vNew " << vOld << std::endl;
          blas::Copy( ntot, Temp.Data(), 1, Res.Data(), 1);          
          blas::Axpy( ntot, -1.0, Temp.Data(), 1, vNew.Data(), 1 ); // F = Xout(New) - Xin(old)
          blas::Copy( ntot, Temp.Data(), 1, Res.Data(), 1);      // Res = Xin
          blas::Copy( ntot, vNew.Data(), 1, vRes.Data(), 1);     // vRes = F
           
          // \Delta X = Xin(i) - Xin(i-1)         
          // \Delta F = F(i) - F(i-1)
          if( iter > 1 ){
            blas::Axpy( ntot, -1.0,  Temp.Data(), 1, dv.VecData(pos), 1 );
            blas::Axpy( ntot, -1.0,  vNew.Data(), 1, df.VecData(pos), 1 );
          }
        } // own this element
  } // for (i)

  // For iter == 1, Anderson mixing is the same as simple mixing. 
  if( iter > 1)
  {
    Int dim=std::min(iter - 1,mixMaxDim_);
    // Normal matrix FTF = F^T * F
    DblNumMat FTFLocal( dim, dim ), FTF( dim, dim );
    SetValue( FTFLocal, 0.0 );
    SetValue( FTF, 0.0 );

    // Right hand side FTv = F^T * vout
    DblNumVec FTvLocal( dim ), FTv( dim );
    SetValue( FTvLocal, 0.0 );
    SetValue( FTv, 0.0 );

    // Local construction of FTF and FTv
    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
        for( Int i = 0; i < numElem_[0]; i++ ){
          Index3 key( i, j, k );
          if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
            DblNumVec& vNew   = distvNew.LocalMap()[key];
            DblNumMat& df     = dfMat.LocalMap()[key];
            for( Int q = 0; q < dim; q++ ){
              FTvLocal(q) += blas::Dot( ntot, df.VecData(q), 1,
                  vNew.Data(), 1 );

              for( Int p = q; p < dim; p++ ){
                FTFLocal(p, q) += blas::Dot( ntot, df.VecData(p), 1, 
                    df.VecData(q), 1 );
                if( p > q )
                  FTFLocal(q,p) = FTFLocal(p,q);
              } // for (p)
            } // for (q)

          } // own this element
    } // for (i)

    // Reduce the data
    mpi::Allreduce( FTFLocal.Data(), FTF.Data(), dim * dim, 
        MPI_SUM, domain_.colComm );
    mpi::Allreduce( FTvLocal.Data(), FTv.Data(), dim, 
        MPI_SUM, domain_.colComm );

    // All processors solve the least square problem

    lapack::Potrf('L', dim, FTF.Data(), dim );

    lapack::Potrs('L', dim, FTF.Data(), dim, I_ONE, FTv.Data(), dim);

    // Update vOpt, resOpt. 
    // FTv = Y^{\dagger} r as in the usual notation.
    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
        for( Int i = 0; i < numElem_[0]; i++ ){
          Index3 key( i, j, k );
          if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
            DblNumVec& vNew   = distvNew.LocalMap()[key];
            DblNumMat& df     = dfMat.LocalMap()[key];
            DblNumMat& dv     = dvMat.LocalMap()[key];
            DblNumVec& Temp = distTemp.LocalMap()[key];
            for(Int p = 0; p < dim; p++){
              blas::Axpy(ntot, -FTv(p),  dv.VecData(p),
                1, Temp.Data(), 1); // Xopt = Xin - \sum\gamma\Delta X
              blas::Axpy(ntot, -FTv(p),  df.VecData(p),
                1, vNew.Data(), 1); // Fopt = F - \sum\gamma\Delta F
            }
          } // own this element
    } // for (i)
  } // (iter > 1)

  // Update dfMat, dvMat, vMix 
  for( Int k = 0; k < numElem_[2]; k++ )
    for( Int j = 0; j < numElem_[1]; j++ )
      for( Int i = 0; i < numElem_[0]; i++ ){
        Index3 key( i, j, k );
        if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){

          DblNumVec& vNew   = distvNew.LocalMap()[key];
          DblNumVec& vMix   = distvMix.LocalMap()[key];
          DblNumVec& Res    = distRes.LocalMap()[key];
          DblNumVec& vRes   = distvRes.LocalMap()[key];
          DblNumMat& df     = dfMat.LocalMap()[key];
          DblNumMat& dv     = dvMat.LocalMap()[key];
          DblNumVec& Temp = distTemp.LocalMap()[key];

//          blas::Copy(ntot, vOld.Data(), 1, Temp.Data(), 1);
          blas::Axpy(ntot, mixStepLength, vNew.Data(), 1, Temp.Data(), 1);
          blas::Copy(ntot, Temp.Data(), 1, vMix.Data(), 1);

          blas::Copy(ntot, Res.Data(), 1, dv.VecData(next), 1);
          blas::Copy(ntot, vRes.Data(), 1, df.VecData(next), 1);

        } // own this element
  } // for (i)

  return ;
}         // -----  end of method SCFDG::AndersonMix2  ----- 

void
SCFDG::PulayMix  ( 
    Int             iter, 
    Real            mixStepLength,
    DistDblNumVec&  distvMix,
    DistDblNumVec&  distvIn,
    DistDblNumVec&  distvOut,
    DistDblNumMat&  dfMat,
    DistDblNumMat&  dvMat )
{
  Int mpirank, mpisize;
  MPI_Comm_rank( domain_.comm, &mpirank );
  MPI_Comm_size( domain_.comm, &mpisize );

  Int mpirankRow;  MPI_Comm_rank(domain_.rowComm, &mpirankRow);
  Int mpisizeRow;  MPI_Comm_size(domain_.rowComm, &mpisizeRow);
  Int mpirankCol;  MPI_Comm_rank(domain_.colComm, &mpirankCol);
  Int mpisizeCol;  MPI_Comm_size(domain_.colComm, &mpisizeCol);

  distvMix.SetComm(domain_.colComm);
  distvIn.SetComm(domain_.colComm);
  distvOut.SetComm(domain_.colComm);
  dfMat.SetComm(domain_.colComm);
  dvMat.SetComm(domain_.colComm);

  // Residual 
  DistDblNumVec distRes;
// // Optimal input potential in Pulay mixing.
  DistDblNumVec distvRes; 
  DistDblNumVec distTemp; 
  // Optimal residual in Pulay mixing
  //
  distRes.SetComm(domain_.colComm);
  distvRes.SetComm(domain_.colComm);
  distTemp.SetComm(domain_.colComm);

  // *********************************************************************
  // Initialize
  // *********************************************************************
  Int ntot  = hamDGPtr_->NumUniformGridElemFine().prod();
//  Int numBasis = esdfParam.numALBElem(0, 0, 0);
  Int pos = ((iter - 1) % mixMaxDim_ - 1 + mixMaxDim_ ) % mixMaxDim_;
  Int next = (pos + 1) % mixMaxDim_;

//  statusOFS << " Iter " << iter <<std::endl;
//  statusOFS << " pos " << pos << std::endl;
//  statusOFS << " next " << next << std::endl;

  distRes.Prtn()          = elemPrtn_;
  distvRes.Prtn()         = elemPrtn_;
  distTemp.Prtn()         = elemPrtn_;

  for( Int k = 0; k < numElem_[2]; k++ )
    for( Int j = 0; j < numElem_[1]; j++ )
      for( Int i = 0; i < numElem_[0]; i++ ){
        Index3 key( i, j, k );
        if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
          DblNumVec  emptyVec( ntot );
          SetValue( emptyVec, 0.0 );
          distRes.LocalMap()[key]        = emptyVec;
          distvRes.LocalMap()[key]       = emptyVec;
          distTemp.LocalMap()[key]       = emptyVec;
        } // if ( own this element )
  } // for (i)

  // *********************************************************************
  // Pulay mixing
  // *********************************************************************

  for( Int k = 0; k < numElem_[2]; k++ )
    for( Int j = 0; j < numElem_[1]; j++ )
      for( Int i = 0; i < numElem_[0]; i++ ){
        Index3 key( i, j, k );
        if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
          // res(:) = vOld(:) - vNew(:) is the residual
          DblNumVec&  vIn = distvIn.LocalMap()[key];
          DblNumVec& vOut = distvOut.LocalMap()[key];
          DblNumVec&  Res = distRes.LocalMap()[key];
          DblNumVec& vRes = distvRes.LocalMap()[key];
          DblNumVec& Temp = distTemp.LocalMap()[key];
          DblNumMat&  dv  = dvMat.LocalMap()[key];
          DblNumMat&  df  = dfMat.LocalMap()[key];
         
          // vOut = vOut - vIn 
          blas::Copy( ntot, vIn.Data(), 1, Temp.Data(), 1); 

          blas::Axpy( ntot, -1.0, Temp.Data(), 1, vOut.Data(), 1 );  // vOut = vOut - vIn 
          // Res = vIn
          blas::Copy( ntot, Temp.Data(), 1, Res.Data(), 1);   // Res = vIn,  copy vIn ?
          // vRes = vOut
          blas::Copy( ntot, vOut.Data(), 1, vRes.Data(), 1); // Here, vRes = vOut becomes vOut - vIn
          
          // dfMat(:, ipos-1) = Res(:) - dfMat(:, ipos-1);
          // dvMat(:, ipos-1) = vRes(:) - dvMat(:, ipos-1);
          if( iter > 1 ){
            blas::Axpy( ntot, -1.0,  Res.Data(), 1, dv.VecData(pos), 1 ); // delta X = Xin(i) - Xin(i-1)
            blas::Axpy( ntot, -1.0,  vRes.Data(), 1, df.VecData(pos), 1 ); // delta F = F(i) - F(i-1)
          }
        } // own this element
  } // for (i)

  // For iter == 1, Anderson mixing is the same as simple mixing. 
  if( iter > 1)
  {
    Int dim=std::min(iter-1,mixMaxDim_);
    statusOFS << " Iter : "<< iter << "  <F|F>'s dim " << dim << std::endl;
    
    // Normal matrix FTF = F^T * F = sum_pq<vRes_p|vRes_q>
    DblNumMat FTFLocal( dim, dim ), FTF( dim, dim );
    SetValue( FTFLocal, 0.0 );
    SetValue( FTF, 0.0 );

    // Right hand side FTv = F^T * Res = sum_q <vRes^q|Res^iter>
    DblNumVec FTvLocal( dim ), FTv( dim );
    SetValue( FTvLocal, 0.0 );
    SetValue( FTv, 0.0 );

    // Local construction of FTF and FTv
    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
        for( Int i = 0; i < numElem_[0]; i++ ){
          Index3 key( i, j, k );
          if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
            DblNumVec& vOut   = distvOut.LocalMap()[key];
            DblNumMat& df     = dfMat.LocalMap()[key];
            for( Int q = 0; q < dim; q++ ){
              FTvLocal(q) += blas::Dot( ntot, df.VecData(q), 1,
                  vOut.Data(), 1 );

              for( Int p = q; p < dim; p++ ){
                FTFLocal(p, q) += blas::Dot( ntot, df.VecData(p), 1, 
                    df.VecData(q), 1 );
                if( p > q )
                  FTFLocal(q,p) = FTFLocal(p,q);
              } // for (p)
            } // for (q)

          } // own this element
    } // for (i)

    // Reduce the data
    mpi::Allreduce( FTFLocal.Data(), FTF.Data(), dim * dim, 
        MPI_SUM, domain_.colComm );
    mpi::Allreduce( FTvLocal.Data(), FTv.Data(), dim, 
        MPI_SUM, domain_.colComm );

    // All processors solve the least square problem

    lapack::Potrf('L', dim, FTF.Data(), dim ); 

    lapack::Potrs('L', dim, FTF.Data(), dim, I_ONE, FTv.Data(), dim); // FTv = alpha=(LLT)^{−1}(FTv)

//    KerkerPrecond( distPrecResOpt, distResOpt );

    // Update vOpt, resOpt. 
    // FTv = Y^{\dagger} r as in the usual notation.
    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
        for( Int i = 0; i < numElem_[0]; i++ ){
          Index3 key( i, j, k );
          if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
//            DblNumVec& vIn   = distvIn.LocalMap()[key];
            DblNumVec& vOut  = distvOut.LocalMap()[key];
            DblNumMat& df    = dfMat.LocalMap()[key];
            DblNumMat& dv    = dvMat.LocalMap()[key];
            DblNumVec& Temp = distTemp.LocalMap()[key];
            for(Int p = 0; p < dim; p++){
              blas::Axpy(ntot, -FTv(p),  dv.VecData(p),
                1, Temp.Data(), 1); // Xopt = Xin - \sum\gamma\Delta X
              blas::Axpy(ntot, -FTv(p),  df.VecData(p),
                1, vOut.Data(), 1); // Fopt = F - \sum\gamma\Delta F
            }
          } // own this element
    } // for (i)
  } // (iter > 1)

  // Update dfMat, dvMat, vMix 
  for( Int k = 0; k < numElem_[2]; k++ )
    for( Int j = 0; j < numElem_[1]; j++ )
      for( Int i = 0; i < numElem_[0]; i++ ){
        Index3 key( i, j, k );
        if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){

//          DblNumVec& vIn    = distvIn.LocalMap()[key];
          DblNumVec& vOut   = distvOut.LocalMap()[key];
          DblNumVec& vMix   = distvMix.LocalMap()[key];
          DblNumVec& Res    = distRes.LocalMap()[key];
          DblNumVec& vRes   = distvRes.LocalMap()[key];
          DblNumMat& df     = dfMat.LocalMap()[key];
          DblNumMat& dv     = dvMat.LocalMap()[key];
          DblNumVec& Temp = distTemp.LocalMap()[key];

//          blas::Copy(ntot, vIn.Data(), 1, Temp.Data(), 1);
          blas::Axpy(ntot, mixStepLength, vOut.Data(), 1, Temp.Data(), 1);
          blas::Copy(ntot, Temp.Data(), 1, vMix.Data(), 1);

          blas::Copy(ntot, Res.Data(), 1, dv.VecData(next), 1);
          blas::Copy(ntot, vRes.Data(), 1, df.VecData(next), 1);

        } // own this element
  } // for (i)

  return ;
}         // -----  end of method SCFDG::PulayMix  ----- 


void 
SCFDG::AndersonMix    (
        Int             iter,
        Real            mixStepLength,
        std::string     mixType,
        DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn>&  distvMix,
        DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn>&  distvOld,
        DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn>&  distvNew,
        DistVec<ElemMatKey, NumTns<Real>, ElemMatPrtn>&  dfMat,
        DistVec<ElemMatKey, NumTns<Real>, ElemMatPrtn>&  dvMat )
{
  Int mpirank, mpisize;
  MPI_Comm_rank( domain_.comm, &mpirank );
  MPI_Comm_size( domain_.comm, &mpisize );

  Int mpirankRow;  MPI_Comm_rank(domain_.rowComm, &mpirankRow);
  Int mpisizeRow;  MPI_Comm_size(domain_.rowComm, &mpisizeRow);
  Int mpirankCol;  MPI_Comm_rank(domain_.colComm, &mpirankCol);
  Int mpisizeCol;  MPI_Comm_size(domain_.colComm, &mpisizeCol);

  distvMix.SetComm(domain_.colComm);
  distvOld.SetComm(domain_.colComm);
  distvNew.SetComm(domain_.colComm);
  dfMat.SetComm(domain_.colComm);
  dvMat.SetComm(domain_.colComm);

  DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn> distRes;
  DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn> distvOpt;
  DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn> distResOpt;

  distRes.LocalMap().clear();
  distvOpt.LocalMap().clear();
  distResOpt.LocalMap().clear();

  distRes.Prtn()          = distvMix.Prtn();
  distvOpt.Prtn()         = distvMix.Prtn();
  distResOpt.Prtn()       = distvMix.Prtn();

  distRes.SetComm(domain_.colComm);
  distvOpt.SetComm(domain_.colComm);
  distResOpt.SetComm(domain_.colComm);

  // *********************************************************************
  // Initialize
  // *********************************************************************
  Int ntot = esdfParam.numALBElem(0, 0, 0) * esdfParam.numALBElem(0, 0, 0);
  Int numBasis = esdfParam.numALBElem(0, 0, 0);
  Int iterused = std::min( iter-1, HybridmixMaxDim_ );
  Int ipos = iter - 1 - ((iter-2)/ HybridmixMaxDim_ ) * HybridmixMaxDim_;
  Int inext = iter - ((iter-1)/ HybridmixMaxDim_) * HybridmixMaxDim_;

//  statusOFS << " Iter " << iter <<std::endl;
//  statusOFS << "iterused  " << iterused << std::endl;
//  statusOFS << " pos " << ipos << std::endl;
//  statusOFS << " next " << inext << std::endl;
//  statusOFS << " HybridmixMaxDim_" << HybridmixMaxDim_ << std::endl;


  DblNumMat  emptyMat( numBasis,  numBasis );
  SetValue( emptyMat, 0.0 );

  for(typename std::map<ElemMatKey, DblNumMat >::iterator
    My_iterator = distvMix.LocalMap().begin();
    My_iterator != distvMix.LocalMap().end();
    ++ My_iterator )
  {
    ElemMatKey matkey = (*My_iterator).first;
    distRes.LocalMap()[matkey]     = emptyMat;
    distvOpt.LocalMap()[matkey]    = emptyMat;
    distResOpt.LocalMap()[matkey]  = emptyMat;
   }

  // *********************************************************************
  // Anderson mixing
  // *********************************************************************
  for(typename std::map<ElemMatKey, DblNumMat >::iterator
     My_iterator = distvMix.LocalMap().begin();
     My_iterator != distvMix.LocalMap().end();
     ++ My_iterator )
  {
    ElemMatKey matkey = (*My_iterator).first;
    DblNumMat& vOld = distvOld.LocalMap()[matkey];
    DblNumMat& Res = distRes.LocalMap()[matkey];
    DblNumMat& vNew = distvNew.LocalMap()[matkey];
    blas::Copy(ntot, vOld.Data(), 1, Res.Data(), 1 ); 
    blas::Axpy( ntot, -1.0, vNew.Data(), 1, Res.Data(), 1);

    DblNumMat& vOpt = distvOpt.LocalMap()[matkey];
    DblNumMat& ResOpt =  distResOpt.LocalMap()[matkey];

    blas::Copy(ntot, vOld.Data(), 1, vOpt.Data(), 1 );
    blas::Copy(ntot, Res.Data(), 1, ResOpt.Data(), 1 );

    if( iter > 1 ){
      DblNumTns& dfTns = dfMat.LocalMap()[matkey];
      DblNumTns& dvTns = dvMat.LocalMap()[matkey];
      blas::Scal( ntot, -1.0, dfTns.MatData(ipos-1), 1 );
      blas::Axpy( ntot, 1.0, Res.Data(), 1, dfTns.MatData(ipos-1), 1 );
      blas::Scal( ntot, -1.0, dvTns.MatData(ipos-1), 1 );
      blas::Axpy( ntot, 1.0, vOld.Data(), 1, dvTns.MatData(ipos-1), 1 );
    }
   }

  if( iter > 1){

    Int nrow = iterused;

    DblNumMat FTFLocal( nrow, nrow ), FTF( nrow, nrow );
    SetValue( FTFLocal, 0.0 );
    SetValue( FTF, 0.0 );

    DblNumVec FTvLocal( nrow ), FTv( nrow );
    SetValue( FTvLocal, 0.0 );
    SetValue( FTv, 0.0 );

    for(typename std::map<ElemMatKey, DblNumMat >::iterator
      My_iterator = distvMix.LocalMap().begin();
      My_iterator != distvMix.LocalMap().end();
      ++ My_iterator )
    {
      ElemMatKey matkey = (* My_iterator).first;
      DblNumTns& df     = dfMat.LocalMap()[matkey];
      DblNumMat& res    = distRes.LocalMap()[matkey];
      for( Int q = 0; q < nrow; q++ ){
        FTvLocal(q) += blas::Dot( ntot, df.MatData(q), 1,
            res.Data(), 1 );

        for( Int p = q; p < nrow; p++ ){
          FTFLocal(p, q) += blas::Dot( ntot, df.MatData(p), 1,
              df.MatData(q), 1 );
          if( p > q )
            FTFLocal(q,p) = FTFLocal(p,q);
        } // for (p)
      } // for (q)
    }

    mpi::Allreduce( FTFLocal.Data(), FTF.Data(), nrow * nrow,
        MPI_SUM, domain_.colComm );
    mpi::Allreduce( FTvLocal.Data(), FTv.Data(), nrow,
        MPI_SUM, domain_.colComm );

    Real rcond = 1e-12;
    Int rank;

    DblNumVec  S( nrow );

    lapack::SVDLeastSquare( nrow, nrow, 1,
        FTF.Data(), nrow, FTv.Data(), nrow,
        S.Data(), rcond, &rank );

    statusOFS << "Rank of dfmat = " << rank <<
      ", rcond = " << rcond << std::endl;
       
    for(typename std::map<ElemMatKey, DblNumMat >::iterator
      My_iterator = distvMix.LocalMap().begin();
      My_iterator != distvMix.LocalMap().end();
      ++ My_iterator )
    {
      ElemMatKey matkey = (*My_iterator).first;
      blas::Gemv('N', ntot, nrow, -1.0, dvMat.LocalMap()[matkey].Data(),
          ntot, FTv.Data(), 1, 1.0,
          distvOpt.LocalMap()[matkey].Data(), 1 );

      blas::Gemv('N', ntot, nrow, -1.0, dfMat.LocalMap()[matkey].Data(),
          ntot, FTv.Data(), 1, 1.0,
          distResOpt.LocalMap()[matkey].Data(), 1 );
    }
    
  } // ( iter > 1 )

  // Update dfMat, dvMat, vMix 
  for(typename std::map<ElemMatKey, DblNumMat >::iterator
    My_iterator = distvMix.LocalMap().begin();
    My_iterator != distvMix.LocalMap().end();
    ++ My_iterator )
  {
    ElemMatKey matkey = (*My_iterator).first;
    
    blas::Copy( ntot, distRes.LocalMap()[matkey].Data(), 1,
        dfMat.LocalMap()[matkey].MatData(inext-1), 1 );
    blas::Copy( ntot, distvOld.LocalMap()[matkey].Data(),  1,
        dvMat.LocalMap()[matkey].MatData(inext-1), 1 );

    Real fac = -1.0 * mixStepLength;      
    DblNumMat& vOpt = distvOpt.LocalMap()[matkey];
    DblNumMat& ResOpt = distResOpt.LocalMap()[matkey];

    distvMix.LocalMap()[matkey] = vOpt;
    blas::Axpy( ntot, fac, ResOpt.Data(), 1,
         distvMix.LocalMap()[matkey].Data(), 1 );
  } 
  
  return ;
}

void
SCFDG::AndersonMix2    ( 
       Int             iter, 
       Real            mixStepLength,
       DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn>&  distvMix,
       DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn>&  distvOld,
       DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn>&  distvNew,
       DistVec<ElemMatKey, NumTns<Real>, ElemMatPrtn>&  dfMat,
       DistVec<ElemMatKey, NumTns<Real>, ElemMatPrtn>&  dvMat )
{
  Int mpirank, mpisize;
  MPI_Comm_rank( domain_.comm, &mpirank );
  MPI_Comm_size( domain_.comm, &mpisize );

  Int mpirankRow;  MPI_Comm_rank(domain_.rowComm, &mpirankRow);
  Int mpisizeRow;  MPI_Comm_size(domain_.rowComm, &mpisizeRow);
  Int mpirankCol;  MPI_Comm_rank(domain_.colComm, &mpirankCol);
  Int mpisizeCol;  MPI_Comm_size(domain_.colComm, &mpisizeCol);

  distvMix.SetComm(domain_.colComm);
  distvOld.SetComm(domain_.colComm);
  distvNew.SetComm(domain_.colComm);
  dfMat.SetComm(domain_.colComm);
  dvMat.SetComm(domain_.colComm);

  DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn> distRes;
  DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn> distvRes;
  DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn> distTemp;

  distRes.LocalMap().clear();
  distvRes.LocalMap().clear();
  distTemp.LocalMap().clear();

  distRes.Prtn()          = distvMix.Prtn();
  distvRes.Prtn()         = distvMix.Prtn();
  distTemp.Prtn()         = distvMix.Prtn();

  distRes.SetComm(domain_.colComm);
  distvRes.SetComm(domain_.colComm);
  distTemp.SetComm(domain_.colComm);

  Int ntot = esdfParam.numALBElem(0, 0, 0) * esdfParam.numALBElem(0, 0, 0);
  Int numBasis = esdfParam.numALBElem(0, 0, 0);
  Int pos = ((iter - 1) % HybridmixMaxDim_ - 1 + HybridmixMaxDim_ ) % HybridmixMaxDim_;
  Int next = (pos + 1) % HybridmixMaxDim_;

 // statusOFS << " Iter " << iter <<std::endl;
 // statusOFS << " pos " << pos << std::endl;
 // statusOFS << " next " << next << std::endl;
 // statusOFS << " HybridmixMaxDim_" << HybridmixMaxDim_ << std::endl;
  
  DblNumMat  emptyMat( numBasis,  numBasis );
  SetValue( emptyMat, 0.0 );

  for(typename std::map<ElemMatKey, DblNumMat >::iterator
    My_iterator = distvMix.LocalMap().begin();
    My_iterator != distvMix.LocalMap().end();
    ++ My_iterator )
  {
    ElemMatKey matkey = (*My_iterator).first;
    distRes.LocalMap()[matkey]    = emptyMat;
    distvRes.LocalMap()[matkey]   = emptyMat;
    distTemp.LocalMap()[matkey]   = emptyMat;
   }

  // *********************************************************************
  // Anderson mixing
  // *********************************************************************
  for(typename std::map<ElemMatKey, DblNumMat >::iterator
     My_iterator = distvMix.LocalMap().begin();
     My_iterator != distvMix.LocalMap().end();
     ++ My_iterator )
  {
    ElemMatKey matkey = (*My_iterator).first;
    DblNumMat& vOld = distvOld.LocalMap()[matkey];
    DblNumMat& Temp = distTemp.LocalMap()[matkey];
    DblNumMat& vNew = distvNew.LocalMap()[matkey];
    DblNumMat&  Res = distRes.LocalMap()[matkey];
    DblNumMat& vRes = distvRes.LocalMap()[matkey];
    DblNumTns& df = dfMat.LocalMap()[matkey];
    DblNumTns& dv = dvMat.LocalMap()[matkey];

    // F = Xout- Xin
    blas::Copy(ntot, vOld.Data(), 1, Temp.Data(),1); 
    blas::Axpy( ntot, -1.0, Temp.Data(), 1, vNew.Data(), 1);
    // Store the current Xin and F
    blas::Copy(ntot, Temp.Data(), 1, Res.Data(),1);
    blas::Copy(ntot, vNew.Data(), 1, vRes.Data(),1);

    if( iter > 1 ){
      // -\Delta X = Xin(i-1) - Xin(i)
      blas::Axpy(ntot, -1.0, Temp.Data(), 1, dv.MatData(pos), 1);
      // -\Delta F = F(i-1) - F(i)
      blas::Axpy(ntot, -1.0, vNew.Data(), 1, df.MatData(pos), 1);
     }
  }

  if( iter > 1 ){
    Int dim = std::min(iter - 1, HybridmixMaxDim_ );

    DblNumMat FTFLocal( dim, dim ), FTF ( dim, dim );
    SetValue( FTFLocal, 0.0 );
    SetValue( FTF, 0.0 );

    DblNumVec FTvLocal( dim ), FTv( dim );
    SetValue( FTvLocal, 0.0 );
    SetValue( FTv, 0.0 );

    for(typename std::map<ElemMatKey, DblNumMat >::iterator
      My_iterator = distvMix.LocalMap().begin();
      My_iterator != distvMix.LocalMap().end();
      ++ My_iterator )
    {
      ElemMatKey matkey = (* My_iterator).first;
      DblNumMat& vNew   = distvNew.LocalMap()[matkey];
      DblNumTns& df     = dfMat.LocalMap()[matkey];

      for(Int q = 0; q < dim; q++){

        FTvLocal(q) += blas::Dot(ntot, df.MatData(q), 1, vNew.Data(), 1);  // -<\Delta F(i)|F>

        for(Int p = q; p < dim; p++){
//          statusOFS << " i, j " << i << " , " <<j <<std::endl; 
          // <\Delta F(i)|\Delta F(j)>
          FTFLocal(p,q) += blas::Dot(ntot, df.MatData(p), 1, df.MatData(q),1); 
//          statusOFS << " FTFLocal(j,i) "<< FTFLocal(j,i)  << std::endl;
          if( p > q )  FTFLocal(q, p) = FTFLocal(p, q);
        }
      }
    }

    mpi::Allreduce( FTFLocal.Data(), FTF.Data(), dim * dim,
        MPI_SUM, domain_.colComm );
    mpi::Allreduce( FTvLocal.Data(), FTv.Data(), dim,
        MPI_SUM, domain_.colComm );

    lapack::Potrf('L', dim, FTF.Data(), dim );

    lapack::Potrs('L', dim, FTF.Data(), dim, I_ONE, FTv.Data(), dim);

    for(typename std::map<ElemMatKey, DblNumMat >::iterator
      My_iterator = distvMix.LocalMap().begin();
      My_iterator != distvMix.LocalMap().end();
      ++ My_iterator )
    {
      ElemMatKey matkey = (*My_iterator).first;
      DblNumTns& df     = dfMat.LocalMap()[matkey];
      DblNumTns& dv     = dvMat.LocalMap()[matkey];
//      DblNumMat& vOld   = distvOld.LocalMap()[matkey];
      DblNumMat& vNew   = distvNew.LocalMap()[matkey];
      DblNumMat& Temp = distTemp.LocalMap()[matkey];   
      for(Int i = 0; i < dim; i++){ 
        blas::Axpy(ntot, -FTv(i),  dv.MatData(i),
             1, Temp.Data(), 1); // Xopt = Xin - \sum\gamma\Delta X
        blas::Axpy(ntot, -FTv(i),  df.MatData(i),
             1, vNew.Data(), 1); // Fopt = F - \sum\gamma\Delta F
      }
    }

  } // ( iter > 1 )

  for(typename std::map<ElemMatKey, DblNumMat >::iterator
    My_iterator = distvMix.LocalMap().begin();
    My_iterator != distvMix.LocalMap().end();
    ++ My_iterator )
  {
    ElemMatKey matkey = (*My_iterator).first;
   // DblNumMat& vOld = distvOld.LocalMap()[matkey];
    DblNumMat& vNew = distvNew.LocalMap()[matkey];
    DblNumMat&  Res = distRes.LocalMap()[matkey];
    DblNumMat& vRes = distvRes.LocalMap()[matkey];
    DblNumMat& vMix = distvMix.LocalMap()[matkey];
    DblNumMat& Temp = distTemp.LocalMap()[matkey];

    DblNumTns& df   = dfMat.LocalMap()[matkey];
    DblNumTns& dv   = dvMat.LocalMap()[matkey];

    //blas::Copy(ntot, vOld.Data(), 1, Temp.Data(), 1);
    blas::Axpy(ntot, mixStepLength, vNew.Data(), 1, Temp.Data(), 1);
    blas::Copy(ntot, Temp.Data(), 1, vMix.Data(), 1);

    blas::Copy(ntot, Res.Data(), 1, dv.MatData(next), 1);
    blas::Copy(ntot, vRes.Data(), 1, df.MatData(next), 1);
  }

  return ;
}   

void
SCFDG::PulayMix  ( 
       Int             iter, 
       Real            mixStepLength,
       DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn>&  distvMix,
       DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn>&  distvIn,
       DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn>&  distvOut,
       DistVec<ElemMatKey, NumTns<Real>, ElemMatPrtn>&  dfMat,
       DistVec<ElemMatKey, NumTns<Real>, ElemMatPrtn>&  dvMat )
{
  Int mpirank, mpisize;
  MPI_Comm_rank( domain_.comm, &mpirank );
  MPI_Comm_size( domain_.comm, &mpisize );

  Int mpirankRow;  MPI_Comm_rank(domain_.rowComm, &mpirankRow);
  Int mpisizeRow;  MPI_Comm_size(domain_.rowComm, &mpisizeRow);
  Int mpirankCol;  MPI_Comm_rank(domain_.colComm, &mpirankCol);
  Int mpisizeCol;  MPI_Comm_size(domain_.colComm, &mpisizeCol);

  distvMix.SetComm(domain_.colComm);
  distvIn.SetComm(domain_.colComm);
  distvOut.SetComm(domain_.colComm);
  dfMat.SetComm(domain_.colComm);
  dvMat.SetComm(domain_.colComm);

  DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn> distRes;
  DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn> distvRes;
  DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn> distTemp;

  distRes.LocalMap().clear();
  distvRes.LocalMap().clear();
  distTemp.LocalMap().clear();

  distRes.Prtn()          = distvMix.Prtn();
  distvRes.Prtn()         = distvMix.Prtn();
  distTemp.Prtn()         = distvMix.Prtn();

  distRes.SetComm(domain_.colComm);
  distvRes.SetComm(domain_.colComm);
  distTemp.SetComm(domain_.colComm);

  Int numBasis = esdfParam.numALBElem(0, 0, 0);

  Int ntot = numBasis*numBasis;
  Int pos = ((iter - 1) % HybridmixMaxDim_ - 1 + HybridmixMaxDim_ ) % HybridmixMaxDim_;
  Int next = (pos + 1) % HybridmixMaxDim_;

//  statusOFS << " Iter " << iter <<std::endl;
//  statusOFS << " pos " << pos << std::endl;
//  statusOFS << " next " << next << std::endl;
//  statusOFS << " HybridmixMaxDim_" << HybridmixMaxDim_ << std::endl;
  
  DblNumMat  emptyMat( numBasis,  numBasis );
  SetValue( emptyMat, 0.0 );

  for(typename std::map<ElemMatKey, DblNumMat >::iterator
    My_iterator = distvMix.LocalMap().begin();
    My_iterator != distvMix.LocalMap().end();
    ++ My_iterator )
  {
    ElemMatKey matkey = (*My_iterator).first;
    distRes.LocalMap()[matkey]    = emptyMat;
    distvRes.LocalMap()[matkey]   = emptyMat;
    distTemp.LocalMap()[matkey]   = emptyMat;
   }

  // *********************************************************************
  // Pulay mixing
  // *********************************************************************
  for(typename std::map<ElemMatKey, DblNumMat >::iterator
     My_iterator = distvMix.LocalMap().begin();
     My_iterator != distvMix.LocalMap().end();
     ++ My_iterator )
  {
    ElemMatKey matkey = (*My_iterator).first;
    DblNumMat&  vIn = distvIn.LocalMap()[matkey];
    DblNumMat& Temp = distTemp.LocalMap()[matkey];
    DblNumMat& vOut = distvOut.LocalMap()[matkey];
    DblNumMat&  Res = distRes.LocalMap()[matkey];
    DblNumMat& vRes = distvRes.LocalMap()[matkey];
    DblNumTns&   df = dfMat.LocalMap()[matkey];
    DblNumTns&   dv = dvMat.LocalMap()[matkey];

    // F = vOut = vOut- vIn
    blas::Copy(ntot, vIn.Data(), 1, Temp.Data(),1);

    blas::Axpy( ntot, -1.0, Temp.Data(), 1, vOut.Data(), 1);
    // Store the current vIn and F
    // Res = vIn,
    blas::Copy(ntot, Temp.Data(), 1, Res.Data(),1);
     // vRes = vOut -vIn = F ,
    blas::Copy(ntot, vOut.Data(), 1, vRes.Data(),1);

    if( iter > 1 ){
      //statusOFS << " iter  " << iter << " pos " << pos << "  next " << next << std::endl;
      // \Delta X = Xin(i-1) - Xin(i)
      blas::Axpy(ntot, -1.0, Res.Data(), 1, dv.MatData(pos), 1);
      // \Delta F = F(i-1) - F(i)
      blas::Axpy(ntot, -1.0, vRes.Data(), 1, df.MatData(pos), 1);
     }
  }

  if( iter > 1 ){
    Int dim = std::min(iter - 1, HybridmixMaxDim_ );

    DblNumMat FTFLocal( dim, dim ), FTF ( dim, dim );
    SetValue( FTFLocal, 0.0 );
    SetValue( FTF, 0.0 );

    DblNumVec FTvLocal( dim ), FTv( dim );
    SetValue( FTvLocal, 0.0 );
    SetValue( FTv, 0.0 );

    for(typename std::map<ElemMatKey, DblNumMat >::iterator
      My_iterator = distvMix.LocalMap().begin();
      My_iterator != distvMix.LocalMap().end();
      ++ My_iterator )
    {
      ElemMatKey matkey = (* My_iterator).first;
      DblNumMat& vOut   = distvOut.LocalMap()[matkey];
      DblNumTns& df     = dfMat.LocalMap()[matkey];

      for(Int q = 0; q < dim; q++){

        FTvLocal(q) += blas::Dot(ntot, df.MatData(q), 1, vOut.Data(), 1);  // -<\Delta F(i)|F>

        for(Int p = q; p < dim; p++){
//          statusOFS << " i, j " << i << " , " <<j <<std::endl; 
          // <\Delta F(i)|\Delta F(j)>
          FTFLocal(p,q) += blas::Dot(ntot, df.MatData(p), 1, df.MatData(q),1); 
//          statusOFS << " FTFLocal(j,i) "<< FTFLocal(j,i)  << std::endl;
          if( p > q )  FTFLocal(q, p) = FTFLocal(p, q);
        }
      }
    }

    mpi::Allreduce( FTFLocal.Data(), FTF.Data(), dim * dim,
        MPI_SUM, domain_.colComm );
    mpi::Allreduce( FTvLocal.Data(), FTv.Data(), dim,
        MPI_SUM, domain_.colComm );

    lapack::Potrf('L', dim, FTF.Data(), dim );

    lapack::Potrs('L', dim, FTF.Data(), dim, I_ONE, FTv.Data(), dim);

    for(typename std::map<ElemMatKey, DblNumMat >::iterator
      My_iterator = distvMix.LocalMap().begin();
      My_iterator != distvMix.LocalMap().end();
      ++ My_iterator )
    {
      ElemMatKey matkey = (*My_iterator).first;
      DblNumTns& df     = dfMat.LocalMap()[matkey];
      DblNumTns& dv     = dvMat.LocalMap()[matkey];
      DblNumMat& vIn    = distvIn.LocalMap()[matkey];
      DblNumMat& vOut   = distvOut.LocalMap()[matkey];
      DblNumMat& Temp = distTemp.LocalMap()[matkey];

      for(Int i = 0; i < dim; i++){ 
        blas::Axpy(ntot, -FTv(i),  dv.MatData(i),
             1, Temp.Data(), 1); // Xopt = Xin - \sum\gamma\Delta X
        blas::Axpy(ntot, -FTv(i),  df.MatData(i),
             1, vOut.Data(), 1); // Fopt = F - \sum\gamma\Delta F
      }
    }

  } // ( iter > 1 )

  for(typename std::map<ElemMatKey, DblNumMat >::iterator
    My_iterator = distvMix.LocalMap().begin();
    My_iterator != distvMix.LocalMap().end();
    ++ My_iterator )
  {
    ElemMatKey matkey = (*My_iterator).first;
   // DblNumMat& vIn  = distvIn.LocalMap()[matkey];
    DblNumMat& vOut = distvOut.LocalMap()[matkey];
    DblNumMat&  Res = distRes.LocalMap()[matkey];
    DblNumMat& vRes = distvRes.LocalMap()[matkey];
    DblNumMat& vMix = distvMix.LocalMap()[matkey];
    DblNumMat& Temp = distTemp.LocalMap()[matkey];

    DblNumTns& df   = dfMat.LocalMap()[matkey];
    DblNumTns& dv   = dvMat.LocalMap()[matkey];

    //blas::Copy(ntot, vIn.Data(), 1, Temp.Data(), 1);
    blas::Axpy(ntot, mixStepLength, vOut.Data(), 1, Temp.Data(), 1);
    blas::Copy(ntot, Temp.Data(), 1, vMix.Data(), 1);

    blas::Copy(ntot, Res.Data(), 1, dv.MatData(next), 1);
    blas::Copy(ntot, vRes.Data(), 1, df.MatData(next), 1);
  }

  return ;
}   

void
SCFDG::KerkerPrecond ( 
    DistDblNumVec&  distPrecResidual,
    const DistDblNumVec&  distResidual )
{
  Int mpirank, mpisize;
  MPI_Comm_rank( domain_.comm, &mpirank );
  MPI_Comm_size( domain_.comm, &mpisize );

  Int mpirankRow;  MPI_Comm_rank(domain_.rowComm, &mpirankRow);
  Int mpisizeRow;  MPI_Comm_size(domain_.rowComm, &mpisizeRow);
  Int mpirankCol;  MPI_Comm_rank(domain_.colComm, &mpirankCol);
  Int mpisizeCol;  MPI_Comm_size(domain_.colComm, &mpisizeCol);

  DistFourier& fft = *distfftPtr_;
  //DistFourier.SetComm(domain_.colComm);

  Int ntot      = fft.numGridTotal;
  Int ntotLocal = fft.numGridLocal;

  Index3 numUniformGridElem = hamDGPtr_->NumUniformGridElem();

  // Convert distResidual to tempVecLocal in distributed row vector format
  DblNumVec  tempVecLocal;

  DistNumVecToDistRowVec(
      distResidual,
      tempVecLocal,
      domain_.numGridFine,
      numElem_,
      fft.localNzStart,
      fft.localNz,
      fft.isInGrid,
      domain_.colComm );

  // NOTE Fixed KerkerB parameter
  //
  // From the point of view of the elliptic preconditioner
  //
  // (-\Delta + 4 * pi * b) r_p = -Delta r
  //
  // The Kerker preconditioner in the Fourier space is
  //
  // k^2 / (k^2 + 4 * pi * b)
  //
  // or using gkk = k^2 /2 
  //
  // gkk / ( gkk + 2 * pi * b )
  //
  // Here we choose KerkerB to be a fixed number.

  // FIXME hard coded
  Real KerkerB = 0.08; 
  Real Amin = 0.4;

  if( fft.isInGrid ){

    for( Int i = 0; i < ntotLocal; i++ ){
      fft.inputComplexVecLocal(i) = Complex( 
          tempVecLocal(i), 0.0 );
    }
    fftw_execute( fft.forwardPlan );

    for( Int i = 0; i < ntotLocal; i++ ){
      // Do not touch the zero frequency
      // Procedure taken from VASP
      if( fft.gkkLocal(i) != 0 ){
        fft.outputComplexVecLocal(i) *= fft.gkkLocal(i) / 
          ( fft.gkkLocal(i) + 2.0 * PI * KerkerB );
        //                fft.outputComplexVecLocal(i) *= std::min(fft.gkkLocal(i) / 
        //                        ( fft.gkkLocal(i) + 2.0 * PI * KerkerB ), Amin);
      }
    }
    fftw_execute( fft.backwardPlan );

    for( Int i = 0; i < ntotLocal; i++ ){
      tempVecLocal(i) = fft.inputComplexVecLocal(i).real() / ntot;
    }
  } // if (fft.isInGrid)

  // Convert tempVecLocal to distPrecResidual in the DistNumVec format 

  DistRowVecToDistNumVec(
      tempVecLocal,
      distPrecResidual,
      domain_.numGridFine,
      numElem_,
      fft.localNzStart,
      fft.localNz,
      fft.isInGrid,
      domain_.colComm );



  return ;
}         // -----  end of method SCFDG::KerkerPrecond  ----- 


void
SCFDG::BroydenMix    (
       Int             iter,
       Real            mixStepLength,
       DistDblNumVec&  distvMix,
       DistDblNumVec&  distvOld,
       DistDblNumVec&  distvNew,
       DistDblNumMat&  dfMat,
       DistDblNumMat&  dvMat,
       DistDblNumMat&  cdfMat )
{
  Int mpirank, mpisize;
  MPI_Comm_rank( domain_.comm, &mpirank );
  MPI_Comm_size( domain_.comm, &mpisize );

  Int mpirankRow;  MPI_Comm_rank(domain_.rowComm, &mpirankRow);
  Int mpisizeRow;  MPI_Comm_size(domain_.rowComm, &mpisizeRow);
  Int mpirankCol;  MPI_Comm_rank(domain_.colComm, &mpirankCol);
  Int mpisizeCol;  MPI_Comm_size(domain_.colComm, &mpisizeCol);

  distvMix.SetComm(domain_.colComm);
  distvOld.SetComm(domain_.colComm);
  distvNew.SetComm(domain_.colComm);
  dfMat.SetComm(domain_.colComm);
  dvMat.SetComm(domain_.colComm);
  cdfMat.SetComm(domain_.colComm);

  DistDblNumVec distGvMix;
  DistDblNumVec distTemp;
//  DistDblNumVec distGvNew;

  // Create space to store intermediate variables
  distGvMix.SetComm(domain_.colComm);
  distTemp.SetComm(domain_.colComm);
//  distGvNew.SetComm(domain_.colComm);

  distGvMix.Prtn()       = elemPrtn_;
  distTemp.Prtn()       = elemPrtn_;
//  distGvNew.Prtn()       = elemPrtn_;

  Int ntot  = hamDGPtr_->NumUniformGridElemFine().prod();

  for( Int k = 0; k < numElem_[2]; k++ )
    for( Int j = 0; j < numElem_[1]; j++ )
      for( Int i = 0; i < numElem_[0]; i++ ){
        Index3 key( i, j, k );
        if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
          DblNumVec  emptyVec( ntot );
          SetValue( emptyVec, 0.0 );
          distGvMix.LocalMap()[key]     = emptyVec;
          distTemp.LocalMap()[key]     = emptyVec;
//          distGvOld.LocalMap()[key]     = distvOld.LocalMap()[key];
//          distGvNew.LocalMap()[key]     = distvNew.LocalMap()[key];
        } // if ( own this element )
  } // for (i)

  // *********************************************************************
  // Initialize
  // *********************************************************************

  // Number of iterations used, iter should start from 1
  Int iterused = std::min( iter-1, mixMaxDim_ );
  // The current position of dfMat, dvMat
  Int ipos = iter - 1 - ((iter-2)/ mixMaxDim_ ) * mixMaxDim_;

//  statusOFS << " Iter " << iter <<std::endl;
//  statusOFS << " iterused  " << iterused << std::endl;
//  statusOFS << " pos " << ipos << std::endl;
  // *********************************************************************
  // Broyden mixing
  // *********************************************************************
  for( Int k = 0; k < numElem_[2]; k++ )
    for( Int j = 0; j < numElem_[1]; j++ )
      for( Int i = 0; i < numElem_[0]; i++ ){
        Index3 key( i, j, k );
        if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
          DblNumVec& vOld   = distvOld.LocalMap()[key];
          DblNumVec& vNew   = distvNew.LocalMap()[key];
          DblNumMat& cdf    = cdfMat.LocalMap()[key];
          DblNumMat& df     = dfMat.LocalMap()[key];
          DblNumMat& dv     = dvMat.LocalMap()[key];
          DblNumVec& Temp   = distTemp.LocalMap()[key];

          blas::Copy( ntot, vOld.Data(), 1, Temp.Data(), 1 );
          blas::Axpy( ntot, -1.0, Temp.Data(), 1, vNew.Data(), 1 );

          if( iter > 1 ){
            blas::Copy( ntot, cdf.VecData(0), 1, df.VecData(ipos-1), 1);
            blas::Axpy( ntot, -1.0, vNew.Data(), 1, df.VecData(ipos-1), 1);
            blas::Copy( ntot, cdf.VecData(1), 1, dv.VecData(ipos-1), 1 );
            blas::Axpy( ntot, -1.0, Temp.Data(), 1, dv.VecData(ipos-1), 1);
          }

          blas::Copy( ntot, vNew.Data(), 1, cdf.VecData(0), 1 );
          blas::Copy( ntot, Temp.Data(), 1, cdf.VecData(1), 1 );
        } // own this element
      } // for (i)

  if( iterused > 0){

    Int nrow = iterused;

    DblNumMat betamixLocal( nrow, nrow ), betamix( nrow, nrow );
    SetValue( betamixLocal, D_ZERO );
    SetValue( betamix, D_ZERO );

    // Local construction of betamix
    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
        for( Int i = 0; i < numElem_[0]; i++ ){
          Index3 key( i, j, k );
          if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
            DblNumMat& df = dfMat.LocalMap()[key];
            for (Int p=0; p<nrow; p++) {
              for (Int q=p; q<nrow; q++) {
                betamixLocal(p,q) = blas::Dot( ntot, df.VecData(q), 1,
                    df.VecData(p), 1 );
                betamixLocal(q,p) = betamixLocal(p,q);
              }
            }
          } // own this element
        } // for (i)

    // Reduce the data
    mpi::Allreduce( betamixLocal.Data(), betamix.Data(), nrow * nrow,
        MPI_SUM, domain_.colComm );

    // Inverse betamix using the Bunch-Kaufman diagonal pivoting method
    IntNumVec iwork;
    iwork.Resize( nrow ); SetValue( iwork, I_ZERO );

    lapack::Sytrf( 'U', nrow, betamix.Data(), nrow, iwork.Data() );
    lapack::Sytri( 'U', nrow, betamix.Data(), nrow, iwork.Data() );
    for (Int p=0; p<nrow; p++) {
      for (Int q=p+1; q<nrow; q++) {
        betamix(q,p) = betamix(p,q);
      }
    }

    DblNumVec workLocal(nrow), work(nrow);
    Real gamma0 = D_ZERO;
    SetValue( workLocal, D_ZERO ); SetValue( work, D_ZERO );

    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
        for( Int i = 0; i < numElem_[0]; i++ ){
          Index3 key( i, j, k );
          if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
            DblNumMat& df     = dfMat.LocalMap()[key];
            DblNumVec& vNew = distvNew.LocalMap()[key];
            for (Int p=0; p<nrow; p++) {
              workLocal(p) = blas::Dot( ntot, df.VecData(p), 1,
                vNew.Data(), 1 );
            }
          } // own this element
        } // for (i)

    mpi::Allreduce( workLocal.Data(), work.Data(), nrow,
        MPI_SUM, domain_.colComm );

    for (Int p=0; p<nrow; p++){
      gamma0 = blas::Dot( nrow, betamix.VecData(p), 1, work.Data(), 1 );

      for( Int k = 0; k < numElem_[2]; k++ )
        for( Int j = 0; j < numElem_[1]; j++ )
          for( Int i = 0; i < numElem_[0]; i++ ){
            Index3 key( i, j, k );
            if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
              blas::Axpy( ntot, -gamma0, dvMat.LocalMap()[key].VecData(p), 1,
                  distTemp.LocalMap()[key].Data(), 1);
              blas::Axpy( ntot, -gamma0, dfMat.LocalMap()[key].VecData(p), 1,
                  distvNew.LocalMap()[key].Data(), 1);
            } // own this element
          } // for (i)
    }
  } // End of if ( iterused > 0 )

  // Update vMix
  for( Int k = 0; k < numElem_[2]; k++ )
    for( Int j = 0; j < numElem_[1]; j++ )
      for( Int i = 0; i < numElem_[0]; i++ ){
        Index3 key( i, j, k );
        if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
          blas::Copy( ntot, distTemp.LocalMap()[key].Data(), 1,
              distGvMix.LocalMap()[key].Data(), 1 );

          blas::Axpy( ntot, mixStepLength, distvNew.LocalMap()[key].Data(), 1,
              distGvMix.LocalMap()[key].Data(), 1 );

          blas::Copy( ntot, distGvMix.LocalMap()[key].Data(), 1,
              distvMix.LocalMap()[key].Data(), 1 );

        } // own this element
      } // for (i)

  return ;
}         // -----  end of method SCFDG::BroydenMix( distributed vector version )  -----


void
SCFDG::BroydenMix    (
       Int             iter,
       Real            mixStepLength,
       DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn>&  distvMix,
       DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn>&  distvOld,
       DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn>&  distvNew,
       DistVec<ElemMatKey, NumTns<Real>, ElemMatPrtn>&  dfTns,
       DistVec<ElemMatKey, NumTns<Real>, ElemMatPrtn>&  dvTns,
       DistVec<ElemMatKey, NumTns<Real>, ElemMatPrtn>&  cdfTns )
{
  Int mpirank, mpisize;
  MPI_Comm_rank( domain_.comm, &mpirank );
  MPI_Comm_size( domain_.comm, &mpisize );

  Int mpirankRow;  MPI_Comm_rank(domain_.rowComm, &mpirankRow);
  Int mpisizeRow;  MPI_Comm_size(domain_.rowComm, &mpisizeRow);
  Int mpirankCol;  MPI_Comm_rank(domain_.colComm, &mpirankCol);
  Int mpisizeCol;  MPI_Comm_size(domain_.colComm, &mpisizeCol);

  distvMix.SetComm(domain_.colComm);
  distvOld.SetComm(domain_.colComm);
  distvNew.SetComm(domain_.colComm);
  dfTns.SetComm(domain_.colComm);
  dvTns.SetComm(domain_.colComm);
  cdfTns.SetComm(domain_.colComm);

  DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn> distGvMix;
  DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn> distTemp;
  // Create space to store intermediate variables
  distGvMix.SetComm(domain_.colComm);
  distTemp.SetComm(domain_.colComm);
  distGvMix.Prtn()  = distvMix.Prtn();
  distTemp.Prtn()   = distvMix.Prtn();


  Int ntot = esdfParam.numALBElem(0, 0, 0) * esdfParam.numALBElem(0, 0, 0);
  Int numBasis = esdfParam.numALBElem(0, 0, 0);

  DblNumMat  emptyMat( numBasis,  numBasis );
  SetValue( emptyMat, 0.0 );

  for(typename std::map<ElemMatKey, DblNumMat >::iterator
    My_iterator = distvMix.LocalMap().begin();
    My_iterator != distvMix.LocalMap().end();
    ++ My_iterator )
  {
    ElemMatKey matkey = (*My_iterator).first;
    distGvMix.LocalMap()[matkey]    = emptyMat;
    distTemp.LocalMap()[matkey]    = emptyMat;
   }

  // *********************************************************************
  // Initialize
  // *********************************************************************

  // Number of iterations used, iter should start from 1
  Int iterused = std::min( iter-1, HybridmixMaxDim_ );
  // The current position of dfMat, dvMat
  Int ipos = iter - 1 - ((iter-2)/ HybridmixMaxDim_ ) * HybridmixMaxDim_;

//  statusOFS << " Iter " << iter <<std::endl;
//  statusOFS << " iterused  " << iterused << std::endl;
//  statusOFS << " pos " << ipos << std::endl;
  // *********************************************************************
  // Broyden mixing
  // *********************************************************************

  for(typename std::map<ElemMatKey, DblNumMat >::iterator
     My_iterator = distvMix.LocalMap().begin();
     My_iterator != distvMix.LocalMap().end();
     ++ My_iterator )
  {
    ElemMatKey matkey = (*My_iterator).first;
    DblNumMat& vOld  = distvOld.LocalMap()[matkey];
    DblNumMat& vNew  = distvNew.LocalMap()[matkey];
    DblNumTns& cdf    = cdfTns.LocalMap()[matkey];
    DblNumTns& df     = dfTns.LocalMap()[matkey];
    DblNumTns& dv     = dvTns.LocalMap()[matkey];
    DblNumMat& Temp  = distTemp.LocalMap()[matkey];

    blas::Copy( ntot, vOld.Data(), 1, Temp.Data(), 1 );
    blas::Axpy( ntot, -1.0, Temp.Data(), 1, vNew.Data(), 1 );

    if( iter > 1 ){
      blas::Copy( ntot, cdf.MatData(0), 1, df.MatData(ipos-1), 1);
      blas::Axpy( ntot, -1.0, vNew.Data(), 1, df.MatData(ipos-1), 1);
      blas::Copy( ntot, cdf.MatData(1), 1, dv.MatData(ipos-1), 1 );
      blas::Axpy( ntot, -1.0, Temp.Data(), 1, dv.MatData(ipos-1), 1);
    }

    blas::Copy( ntot, vNew.Data(), 1, cdf.MatData(0), 1 );
    blas::Copy( ntot, Temp.Data(), 1, cdf.MatData(1), 1 );
  } // for (i)

  if( iterused > 0){

    Int nrow = iterused;
    DblNumMat betamixLocal( nrow, nrow ), betamix( nrow, nrow );
    SetValue( betamixLocal, D_ZERO );
    SetValue( betamix, D_ZERO );

    for(typename std::map<ElemMatKey, DblNumMat >::iterator
       My_iterator = distvMix.LocalMap().begin();
       My_iterator != distvMix.LocalMap().end();
       ++ My_iterator )
    {
      ElemMatKey matkey = (*My_iterator).first;
      DblNumTns& df     = dfTns.LocalMap()[matkey];

      for (Int p = 0; p < nrow; p++) {
        for (Int q = p; q < nrow; q++) {
          betamixLocal(p, q) = blas::Dot( ntot, df.MatData(q), 1, df.MatData(p), 1 );
          betamixLocal(q, p) = betamixLocal(p, q);
        }
      }
    }
    // Reduce the data
    mpi::Allreduce( betamixLocal.Data(), betamix.Data(), nrow * nrow,
        MPI_SUM, domain_.colComm );

    // Inverse betamix using the Bunch-Kaufman diagonal pivoting method
    IntNumVec iwork;
    iwork.Resize( nrow ); SetValue( iwork, I_ZERO );

    lapack::Sytrf( 'U', nrow, betamix.Data(), nrow, iwork.Data() );
    lapack::Sytri( 'U', nrow, betamix.Data(), nrow, iwork.Data() );
    for (Int p=0; p<nrow; p++) {
      for (Int q=p+1; q<nrow; q++) {
        betamix(q,p) = betamix(p,q);
      }
    }

    DblNumVec workLocal(nrow), work(nrow);
    Real gamma0 = D_ZERO;
    SetValue( workLocal, D_ZERO ); SetValue( work, D_ZERO );

    for(typename std::map<ElemMatKey, DblNumMat >::iterator
       My_iterator = distvMix.LocalMap().begin();
       My_iterator != distvMix.LocalMap().end();
       ++ My_iterator )
    {
      ElemMatKey matkey = (*My_iterator).first;
      DblNumMat& vNew  = distvNew.LocalMap()[matkey];
      DblNumTns& cdf    = cdfTns.LocalMap()[matkey];
      DblNumTns& df     = dfTns.LocalMap()[matkey];

      for (Int p=0; p<nrow; p++) {
        workLocal(p) = blas::Dot( ntot, df.MatData(p), 1,
          vNew.Data(), 1 );
      }
    }

    mpi::Allreduce( workLocal.Data(), work.Data(), nrow,
        MPI_SUM, domain_.colComm );

    for (Int p=0; p<nrow; p++){
      gamma0 = blas::Dot( nrow, betamix.VecData(p), 1, work.Data(), 1 );

      for(typename std::map<ElemMatKey, DblNumMat >::iterator
         My_iterator = distvMix.LocalMap().begin();
         My_iterator != distvMix.LocalMap().end();
         ++ My_iterator )
      {
        ElemMatKey matkey = (*My_iterator).first;
        DblNumMat& vNew  = distvNew.LocalMap()[matkey];
//        DblNumMat& vOld  = distvOld.LocalMap()[matkey];
        DblNumMat& Temp  = distTemp.LocalMap()[matkey];
        DblNumTns& df     = dfTns.LocalMap()[matkey];
        DblNumTns& dv     = dvTns.LocalMap()[matkey];

        blas::Axpy( ntot, -gamma0, dv.MatData(p), 1, Temp.Data(), 1);
        blas::Axpy( ntot, -gamma0, df.MatData(p), 1, vNew.Data(), 1);
      } // for (i)
    }
  } // End of if ( iterused > 0 )

  // Update vMix
  for(typename std::map<ElemMatKey, DblNumMat >::iterator
     My_iterator = distvMix.LocalMap().begin();
     My_iterator != distvMix.LocalMap().end();
     ++ My_iterator )
  {
    ElemMatKey matkey = (*My_iterator).first;
    DblNumMat& vNew   = distvNew.LocalMap()[matkey];
    DblNumMat& Temp   = distTemp.LocalMap()[matkey];
    DblNumMat& vMix   = distvMix.LocalMap()[matkey];
    DblNumMat& GvMix  = distGvMix.LocalMap()[matkey];

    blas::Copy( ntot, Temp.Data(), 1, GvMix.Data(), 1 );
    blas::Axpy( ntot, mixStepLength, vNew.Data(), 1, GvMix.Data(), 1 );
    blas::Copy( ntot, GvMix.Data(), 1, vMix.Data(), 1 );

  } // for (i)

  return ;
}         // -----  end of method SCFDG::BroydenMix( distributed vector version )  -----

void
SCFDG::SetupDMMix( )
{
//  statusOFS << "Init hybrid mixing parameters " <<std::endl;

  HamiltonianDG& hamDG = *hamDGPtr_;

//  distHMatSave_.SetComm(domain_.colComm);
//  distDMMatSave_.SetComm(domain_.colComm);
  distdfInnerMat_.SetComm(domain_.colComm);
  distdvInnerMat_.SetComm(domain_.colComm);
  distcdfInnerMat_.SetComm(domain_.colComm);
//  distHMatSave_.Prtn()  = hamDG.HMat().Prtn();
//  distDMMatSave_.Prtn()  = hamDG.HMat().Prtn();
  distdfInnerMat_.Prtn()  = hamDG.HMat().Prtn();
  distdvInnerMat_.Prtn()  = hamDG.HMat().Prtn();
  distcdfInnerMat_.Prtn()  = hamDG.HMat().Prtn();

  distHMatSave_.LocalMap().clear();
  distDMMatSave_.LocalMap().clear();
  distdfInnerMat_.LocalMap().clear();
  distdvInnerMat_.LocalMap().clear();
  distcdfInnerMat_.LocalMap().clear();

  Int numBasis = esdfParam.numALBElem(0,0,0);
  DblNumMat emptyMat( numBasis, numBasis );
  SetValue( emptyMat, 0.0 );
  DblNumTns emptyTns( numBasis, numBasis, HybridmixMaxDim_ );
  DblNumTns emptyTns2( numBasis, numBasis, 2 );
  SetValue( emptyTns, 0.0 );
  SetValue( emptyTns2, 0.0 );

  for(typename std::map<ElemMatKey, DblNumMat >::iterator
    My_iterator = hamDG.HMat().LocalMap().begin();
    My_iterator != hamDG.HMat().LocalMap().end();
    ++ My_iterator )
  {
    ElemMatKey matkey = (*My_iterator).first;

    std::map<ElemMatKey, DblNumMat>::iterator mi =
        distHMatSave_.LocalMap().find( matkey );

    if( mi == distHMatSave_.LocalMap().end() ){
      distHMatSave_.LocalMap()[matkey] = emptyMat;
    }
    else{
      DblNumMat&  mat = (*mi).second;
      blas::Copy( emptyMat.Size(), emptyMat.Data(), 1, mat.Data(), 1);
    }

    std::map<ElemMatKey, DblNumMat>::iterator mj =
        distDMMatSave_.LocalMap().find( matkey );
    if( mj == distDMMatSave_.LocalMap().end() ){
      distDMMatSave_.LocalMap()[matkey] = emptyMat;
    }
    else{
      DblNumMat&  mat = (*mj).second;
      blas::Copy( emptyMat.Size(), emptyMat.Data(), 1, mat.Data(), 1);
    }
  // Initialize distributed tensor in distdfInnerMat_
  // and distdvInnerMat_ 
    std::map<ElemMatKey, DblNumTns>::iterator ni =
        distdfInnerMat_.LocalMap().find( matkey );
    if( ni == distdfInnerMat_.LocalMap().end() ){
      distdfInnerMat_.LocalMap()[matkey] = emptyTns;
    }
    else{
      DblNumTns&  mixmat = (*ni).second;
      blas::Copy( emptyTns.Size(), emptyTns.Data(), 1, mixmat.Data(), 1);
    }
  
    std::map<ElemMatKey, DblNumTns>::iterator ki =
      distdvInnerMat_.LocalMap().find( matkey );
    if( ki == distdvInnerMat_.LocalMap().end() ){
      distdvInnerMat_.LocalMap()[matkey] = emptyTns;
    }
    else{
      DblNumTns&  mixmat = (*ki).second;
      blas::Copy( emptyTns.Size(), emptyTns.Data(), 1, mixmat.Data(), 1);
    }

    // Additional term for Broyden mixing
    if( HybridmixType_ == "broyden" )//    { 
    { 
      std::map<ElemMatKey, DblNumTns>::iterator bi =
         distcdfInnerMat_.LocalMap().find( matkey );

      if( bi == distcdfInnerMat_.LocalMap().end() ){
        distcdfInnerMat_.LocalMap()[matkey] = emptyTns2;
      }
      else{
        DblNumTns&  mixmat = (*bi).second;
        blas::Copy( emptyTns2.Size(), emptyTns2.Data(), 1, mixmat.Data(), 1);
      }
    }

  }

  return;
} 

} // namespace dgdft
