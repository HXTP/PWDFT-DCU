/*
   Copyright (c) 2012 The Regents of the University of California,
   through Lawrence Berkeley National Laboratory.  

Authors: Lexing Ying and Lin Lin

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
/// @file nummat_impl.hpp
/// @brief Implementation of numerical matrix.
/// @date 2010-09-27
#ifndef _NUMMAT_IMPL_HPP_
#define _NUMMAT_IMPL_HPP_

#include "nummat_decl.hpp"

namespace  dgdft{

template <class F> inline NumMat<F>::NumMat(Int m, Int n): m_(m), n_(n), owndata_(true) {
  if(m_>0 && n_>0) { data_ = new F[m_*n_]; if( data_ == NULL ) ErrorHandling("Cannot allocate memory."); } else data_=NULL;
}

template <class F> inline NumMat<F>::NumMat(Int m, Int n, bool owndata, F* data): m_(m), n_(n), owndata_(owndata) {
  if(owndata_) {
    if(m_>0 && n_>0) { data_ = new F[m_*n_]; if( data_ == NULL ) ErrorHandling("Cannot allocate memory."); } else data_=NULL;
    if(m_>0 && n_>0) { for(Int i=0; i<m_*n_; i++) data_[i] = data[i]; }
  } else {
    data_ = data;
  }
}

template <class F> inline NumMat<F>::NumMat(const NumMat& C): m_(C.m_), n_(C.n_), owndata_(C.owndata_) {
  if(owndata_) {
    if(m_>0 && n_>0) { data_ = new F[m_*n_]; if( data_ == NULL ) ErrorHandling("Cannot allocate memory."); } else data_=NULL;
    if(m_>0 && n_>0) { for(Int i=0; i<m_*n_; i++) data_[i] = C.data_[i]; }
  } else {
    data_ = C.data_;
  }
}

template <class F> inline NumMat<F>::~NumMat() {
  if(owndata_) {
    if(m_>0 && n_>0) { delete[] data_; data_ = NULL; }
  }
}

template <class F> inline NumMat<F>& NumMat<F>::operator=(const NumMat& C) {
  // Do not copy if it is the same matrix.
  if(C.data_ == data_) return *this;
  if(owndata_) {
    if(m_>0 && n_>0) { delete[] data_; data_ = NULL; }
  }
  m_ = C.m_; n_=C.n_; owndata_=C.owndata_;
  if(owndata_) {
    if(m_>0 && n_>0) { data_ = new F[m_*n_]; if( data_ == NULL ) ErrorHandling("Cannot allocate memory."); } else data_=NULL;
    if(m_>0 && n_>0) { for(Int i=0; i<m_*n_; i++) data_[i] = C.data_[i]; }
  } else {
    data_ = C.data_;
  }
  return *this;
}

template <class F> inline void NumMat<F>::Resize(Int m, Int n)  {
  if( owndata_ == false ){
    ErrorHandling("Matrix being resized must own data.");
  }
  if(m_!=m || n_!=n) {
    if(m_>0 && n_>0) { delete[] data_; data_ = NULL; }
    m_ = m; n_ = n;
    if(m_>0 && n_>0) { data_ = new F[m_*n_]; if( data_ == NULL ) ErrorHandling("Cannot allocate memory."); } else data_=NULL;
  }
}

template <class F> 
inline const F& NumMat<F>::operator()(Int i, Int j) const  { 
#if ( _DEBUGlevel_ >= 1 )
  if( i < 0 || i >= m_ ||
      j < 0 || j >= n_ ) {
    std::ostringstream msg;
    msg 
      << "Index is out of bound."  << std::endl
      << "Index bound    ~ (" << m_ << ", " << n_ << ")" << std::endl
      << "This index     ~ (" << i  << ", " << j  << ")" << std::endl;
    ErrorHandling( msg.str().c_str() ); 
  }
#endif
  return data_[i+j*m_];
}

template <class F>
inline F& NumMat<F>::operator()(Int i, Int j)  { 
#if ( _DEBUGlevel_ >= 1 )
  if( i < 0 || i >= m_ ||
      j < 0 || j >= n_ ) {
    std::ostringstream msg;
    msg 
      << "Index is out of bound."  << std::endl
      << "Index bound    ~ (" << m_ << ", " << n_ << ")" << std::endl
      << "This index     ~ (" << i  << ", " << j  << ")" << std::endl;
    ErrorHandling( msg.str().c_str() ); 
  }
#endif
  return data_[i+j*m_];
}

template <class F>
inline F* NumMat<F>::VecData(Int j)  const 
{
#if ( _DEBUGlevel_ >= 1 )
  if( j < 0 || j >= n_ ) {
    std::ostringstream msg;
    msg 
      << "Index is out of bound."  << std::endl
      << "Index bound    ~ (" << n_ << ")" << std::endl
      << "This index     ~ (" << j  << ")" << std::endl;
    ErrorHandling( msg.str().c_str() ); 
  }
#endif
  return &(data_[j*m_]); 
}


template <class F>
void NumMat<F>::ExchangeCols(IntNumVec& colMap) {
    // 这里传入一个和矩阵列数相同维度的数组，对于数组中每个元素，原矩阵中索引号对应的列的数据，应该存储在索引对应的值这一列
    // 例如，假设传入数组：[0, 2, 3, 1]，则表示：第0列所应该存储的数据应该存储在第0列，
    // 所以第0列数据不用移动，第1列应该要存储的数据，应该存储在第2列，因此将第1列和第2列交换数据，同时交换数组中对应位置的值
    // 此时数组变为：[0, 3, 2, 1]，然后此时第1列所存储的数据，应该放置在第3列，因此交换第1列和第3列的数据，同时交换对应数组中的值
    // 此时数组变为：[0, 1, 2, 3]，此时数据变换完毕，每一行对应的数据，均在其对应的列上。
    if (colMap.Size() != n_) {
        std::ostringstream msg;
        msg << "ERROR! Input mapping with wrong size, input size: " << colMap.Size()
            << "matrix column number: ( " << n_ << " )";
        ErrorHandling(msg.str().c_str());
    }
    int col_index = 0;
    while (col_index < colMap.Size()) {
        if (col_index == colMap[col_index]) {
            col_index++;
            continue;
        }
        NumVec<F> tmp_vec(m_);
        int tmp;
        while (col_index != colMap[col_index]) {
            if (col_index >= colMap.Size()){
                std::ostringstream msg;
                msg << "ERROR! Index of colMap exceeds the column dimension of the matrix." << std::endl
                    << " Index: " <<  col_index << std::endl
                    << " Value: " << colMap[col_index] << std::endl
                    << " Matrix column number: ( " << n_ << " )" << std::endl << std::endl;
                ErrorHandling(msg.str().c_str());
            }
            memcpy(tmp_vec.Data(), this->VecData(colMap[col_index]), sizeof(F) * m_);
            tmp = colMap[colMap[col_index]];
            memcpy(this->VecData(colMap[col_index]), this->VecData(col_index), sizeof(F) * m_);
            colMap[colMap[col_index]] = colMap[col_index];
            memcpy(this->VecData(col_index), tmp_vec.Data(), sizeof(F) * m_);
            colMap[col_index] = tmp;
        }
    }
}

template <class F>
void NumMat<F>::Rearrange(Int blockNum) {
    Int cols = n_;
    Int rows = m_;
    Int r = rows % blockNum;
    Int blockSize = rows / blockNum;
    Int subBlockTotal = blockNum * cols;
    NumVec<F> buf(blockSize + 1);
    Int subBlockNum2, tmp;
    IntNumVec index2;
    if (r == 0){
        subBlockNum2 = blockNum * cols;
        index2.Resize(subBlockNum2);
        for (Int i = 0; i < subBlockNum2; i++){
            index2(i) = i / blockNum + (i % blockNum) * cols;
        }
        for(Int i = 0; i < subBlockNum2; i++){
            while(index2(i) != i){
                Int cur_ptr = (i / blockNum)  * rows + (i % blockNum) * blockSize;
                Int target_ptr = (index2(i) / blockNum)  * rows + (index2(i) % blockNum) * blockSize;
                memcpy((void *) buf.Data(), (void *)(data_ + cur_ptr), sizeof(F) * blockSize);
                memcpy((void *)(data_ + cur_ptr), (void *)(data_ + target_ptr), sizeof(F) * blockSize);
                memcpy((void *)(data_ + target_ptr), (void *)buf.Data(), sizeof(F) * blockSize);
                tmp = index2(index2(i));
                index2(index2(i)) = index2(i);
                index2(i) = tmp;
            }
        }
        return;
    }
    subBlockNum2 = subBlockTotal - (rows % blockNum) * cols;
    index2.Resize(subBlockNum2);
    for (Int i = 0; i < subBlockNum2; i++){
        index2(i) = i / (blockNum - r) + (i % (blockNum - r)) * cols;
    }
    for(Int i = 0; i < subBlockNum2; i++){
        while(index2(i) != i){
            Int cur_ptr = (i / (blockNum - r))  * rows + (i % (blockNum - r)) * blockSize + (blockSize + 1) * r;
            Int target_ptr = (index2(i) / (blockNum - r))  * rows + (index2(i) % (blockNum - r)) * blockSize + (blockSize + 1) * r;
            memcpy((void *) buf.Data(), (void *)(data_ + cur_ptr), sizeof(F) * blockSize);
            memcpy((void *)(data_ + cur_ptr), (void *)(data_ + target_ptr), sizeof(F) * blockSize);
            memcpy((void *)(data_ + target_ptr), (void *)buf.Data(), sizeof(F) * blockSize);
            tmp = index2(index2(i));
            index2(index2(i)) = index2(i);
            index2(i) = tmp;
        }
    }
    if (rows % blockNum == 0) {
        return;
    }
    Int subBlockNum1 = (rows % blockNum) * cols;
    IntNumVec index1(subBlockNum1);
    for (Int i = 0; i < subBlockNum1; i++){
        index1(i) = i / r + (i % r) * cols;
    }

    for(Int i = 0; i < subBlockNum1; i++){
        while(index1(i) != i){
            Int cur_ptr = (i / r)  * rows + (i % r) * (blockSize + 1);
            Int target_ptr = (index1(i) / r)  * rows + (index1(i) % r) * (blockSize + 1);
            memcpy((void *) buf.Data(), (void *)(data_ + cur_ptr), sizeof(F) * (blockSize + 1));
            memcpy((void *)(data_ + cur_ptr), (void *)(data_ + target_ptr), sizeof(F) * (blockSize + 1));
            memcpy((void *)(data_ + target_ptr), (void *)buf.Data(), sizeof(F) *(blockSize + 1));
            tmp = index1(index1(i));
            index1(index1(i)) = index1(i);
            index1(i) = tmp;
        }
    }
    Int largerBlockNum = 2;
    Int largerSubBlockNum = largerBlockNum * cols;
    IntNumVec largerBlockSize(largerBlockNum);
    largerBlockSize(0) = (rows%blockNum)*(blockSize+1);
    largerBlockSize(1) = rows-(rows%blockNum)*(blockSize+1);
    Int bufSize = largerBlockSize(0) > largerBlockSize(1) ? largerBlockSize(0) : largerBlockSize(0);
    buf.Resize(bufSize);
    Int ptr = 0;
    for (Int index = 0; index < cols; index++) {
        Int tmpCol = index % cols;
        Int tmpRow = index / cols;
        Int currentSubBlockEles = largerBlockSize(0);
        Int currentSubBlockPtr = tmpCol * (largerBlockSize(0) + largerBlockSize(1));
        if(ptr == currentSubBlockPtr){
            ptr += currentSubBlockEles;
            continue;
        }
        memcpy((void*)buf.Data(), (void*)(data_ + currentSubBlockPtr), sizeof(F) * currentSubBlockEles);
        for (Int k = currentSubBlockPtr - 1; k >= ptr; k--) {
            *(this->data_ + k + currentSubBlockEles) = *(this->data_ + k);
        }
        memcpy((void*)(this->data_ + ptr), buf.Data(), sizeof(F) * currentSubBlockEles);
        ptr += currentSubBlockEles;
    }
}

template <class F>
void NumMat<F>::RevertRearrange(Int blockNum) {
    Int cols = n_;
    Int rows = m_;
    Int r = rows % blockNum;
    Int blockSize = rows / blockNum;
    Int subBlockTotal = blockNum * cols;
    NumVec<F> buf;
    Int tmp;
    if (r != 0){
        Int largerBlockNum = 2;
        Int largerSubBlockNum = largerBlockNum * cols;
        IntNumVec largerBlockSize(largerBlockNum);
        largerBlockSize(0) = (rows%blockNum)*(blockSize+1);
        largerBlockSize(1) = rows-(rows%blockNum)*(blockSize+1);
        Int bufSize = largerBlockSize(0) > largerBlockSize(1) ? largerBlockSize(0) : largerBlockSize(0);
        buf.Resize(bufSize); 
        Int ptr = cols * (largerBlockSize(0) + largerBlockSize(1)) - largerBlockSize(1);
        for(Int index = cols-1; index > 0; index--){
            Int tmpCol = index % cols;
            Int currentSubBlockEles = largerBlockSize(0);
            Int currentSubBlockPtr = tmpCol * largerBlockSize(0);
            memcpy((void*)buf.Data(), (void*)(data_ + currentSubBlockPtr), sizeof(F)*currentSubBlockEles);
            for(Int k = currentSubBlockPtr; k < ptr - currentSubBlockEles; k++){
                *(this->data_ + k) = *(this->data_ + k + currentSubBlockEles);
            }
            memcpy((void*)(this->data_ + ptr - currentSubBlockEles), buf.Data(), sizeof(F) * currentSubBlockEles);
            ptr -= largerBlockSize(0) + largerBlockSize(1);
        }   
        buf.Resize(blockSize + 1);
        Int subBlockNum1 = (rows % blockNum) * cols;
        IntNumVec index1(subBlockNum1);
        for (Int i = 0; i < subBlockNum1; i++){
            index1(i) = (i % cols) * r + i / cols; 
        }
        for(Int i = 0; i < subBlockNum1; i++){
            while(index1(i) != i){ 
                Int cur_ptr = (i / r)  * rows + (i % r) * (blockSize + 1);
                Int target_ptr = (index1(i) / r)  * rows + (index1(i) % r) * (blockSize + 1);
                memcpy((void *) buf.Data(), (void *)(data_ + cur_ptr), sizeof(F) * (blockSize + 1));
                memcpy((void *)(data_ + cur_ptr), (void *)(data_ + target_ptr), sizeof(F) * (blockSize + 1));
                memcpy((void *)(data_ + target_ptr), (void *)buf.Data(), sizeof(F) *(blockSize + 1));
                tmp = index1(index1(i));
                index1(index1(i)) = index1(i);
                index1(i) = tmp;
            }   
        }       
        Int subBlockNum2 = subBlockTotal - (rows%blockNum)*cols ? subBlockTotal - (rows%blockNum)*cols :  rows*blockNum;
        IntNumVec index2(subBlockNum2);
        for (Int i = 0; i < subBlockNum2; i++){
            index2(i) = (i % cols) * (blockNum - r) + i / cols; 
        }
        for(Int i = 0; i < subBlockNum2; i++){
            while(index2(i) != i){
                Int cur_ptr = (i / (blockNum - r))  * rows + (i % (blockNum - r)) * blockSize + (blockSize + 1) * r;
                Int target_ptr = (index2(i) / (blockNum - r))  * rows + (index2(i) % (blockNum - r)) * blockSize + (blockSize + 1) * r;
                 memcpy((void *) buf.Data(), (void *)(data_ + cur_ptr), sizeof(F) * blockSize);
                 memcpy((void *)(data_ + cur_ptr), (void *)(data_ + target_ptr), sizeof(F) * blockSize);
                 memcpy((void *)(data_ + target_ptr), (void *)buf.Data(), sizeof(F) * blockSize); 
                 tmp = index2(index2(i));
                 index2(index2(i)) = index2(i);
                 index2(i) = tmp;
             }   
         }       
     } else{     
         Int subBlockNum2 = cols*blockNum;
         IntNumVec index2(subBlockNum2);
         buf.Resize(blockSize);
         for (Int i = 0; i < subBlockNum2; i++){
             index2(i) = (i % cols) * blockNum + i / cols;
         }
         for(Int i = 0; i < subBlockNum2; i++){
             while(index2(i) != i){
                 Int cur_ptr = (i / blockNum)  * rows + (i % blockNum) * blockSize + (blockSize + 1) * r;
                 Int target_ptr = (index2(i) / blockNum)  * rows + (index2(i) % blockNum) * blockSize + (blockSize + 1) * r;
                 memcpy((void *) buf.Data(), (void *)(data_ + cur_ptr), sizeof(F) * blockSize);
                 memcpy((void *)(data_ + cur_ptr), (void *)(data_ + target_ptr), sizeof(F) * blockSize);
                 memcpy((void *)(data_ + target_ptr), (void *)buf.Data(), sizeof(F) * blockSize);
                 tmp = index2(index2(i));
                 index2(index2(i)) = index2(i);
                 index2(i) = tmp;
             }
         }
     }
 }

// *********************************************************************
// Utilities
// *********************************************************************

template <class F> inline void SetValue(NumMat<F>& M, F val)
{
  F *ptr = M.data_;
  for (Int i=0; i < M.m()*M.n(); i++) *(ptr++) = val;
}

template <class F> inline Real Energy(const NumMat<F>& M)
{
  Real sum = 0;
  F *ptr = M.data_;
  for (Int i=0; i < M.m()*M.n(); i++) 
    sum += std::abs(ptr[i]) * std::abs(ptr[i]);
  return sum;
}


template <class F> inline void
Transpose ( const NumMat<F>& A, NumMat<F>& B )
{
  if( A.m() != B.n() || A.n() != B.m() ){
    B.Resize( A.n(), A.m() );
  }

  F* Adata = A.Data();
  F* Bdata = B.Data();
  Int m = A.m(), n = A.n();

  for( Int i = 0; i < m; i++ ){
    for( Int j = 0; j < n; j++ ){
      Bdata[ j + n*i ] = Adata[ i + j*m ];
    }
  }


  return ;
}        // -----  end of function Transpose  ----- 

template <class F> inline void
Symmetrize( NumMat<F>& A )
{
  if( A.m() != A.n() ){
    ErrorHandling( "The matrix to be symmetrized should be a square matrix." );
  }

  NumMat<F> B;
  Transpose( A, B );

  F* Adata = A.Data();
  F* Bdata = B.Data();

  F  half = (F) 0.5;

  for( Int i = 0; i < A.m() * A.n(); i++ ){
    *Adata = half * (*Adata + *Bdata);
    Adata++; Bdata++;
  }


  return ;
}        // -----  end of function Symmetrize ----- 

} // namespace dgdft

#endif // _NUMMAT_IMPL_HPP_
