/*
   Copyright (c) 2012 The Regents of the University of California,
   through Lawrence Berkeley National Laboratory.  

Author: Lexing Ying and Lin Lin

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
/// @file distvec_decl.hpp
/// @brief General purpose parallel vectors.
/// @date 2013-01-09
#ifndef _DISTVEC_DECL_HPP_
#define _DISTVEC_DECL_HPP_

#include "environment.hpp"

namespace dgdft{

/// @namespace PutMode
/// @brief Mode for put operation of DistVec.
namespace PutMode{
enum {
  REPLACE = 0,
  COMBINE = 1,
};
}

/// @class DistVec
/// @brief General purpose parallel vector interfaces.
///
/// DistVec uses a triplet <Key, Data, Partition> to describe a general
/// parallel vectors.
template <class Key, class Data, class Partition>
class DistVec
{
private:
  std::map<Key,Data> lclmap_;
  Partition          prtn_;   //has function owner:Key->pid
  MPI_Comm           comm_;

  // Data communication variables
  std::vector<Int> snbvec_;
  std::vector<Int> rnbvec_;
  std::vector< std::vector<char> > sbufvec_;
  std::vector< std::vector<char> > rbufvec_;
  MPI_Request *reqs_;
  MPI_Status  *stats_;
public:
  DistVec( MPI_Comm comm = MPI_COMM_WORLD ) : comm_(comm) {;}
  ~DistVec() {;}
  //
  std::map<Key,Data>& LocalMap()             { return lclmap_; }
  const std::map<Key,Data>& LocalMap() const { return lclmap_; }
  Partition& Prtn()                          { return prtn_; }
  const Partition& Prtn() const              { return prtn_; }
  //
  Int Insert(Key, Data&);
  Data& Access(Key);
  //
  void SetComm( MPI_Comm comm ) {comm_ = comm;}
  //
  Int GetBegin(Int (*e2ps)(Key, Data& ,std::vector<Int>&), const std::vector<Int>& mask); //gather all entries st pid contains this proc
  Int GetBegin(std::vector<Key>& keyvec, const std::vector<Int>& mask); //gather all entries with key in keyvec
  Int GetEnd(const std::vector<Int>& mask);
  Int PutBegin(std::vector<Key>& keyvec, const std::vector<Int>& mask); //put data for all entries with key in keyvec
  Int PutEnd(const std::vector<Int>& mask, Int putmode=0);
  //
  Int Expand( std::vector<Key>& keyvec); //allocate space for not-owned entries
  Int Discard(std::vector<Key>& keyvec); //remove non-owned entries
  Int mpirank() const { Int rank; MPI_Comm_rank(comm_, &rank); return rank; }
  Int mpisize() const { Int size; MPI_Comm_size(comm_, &size); return size; }
};

} //  namespace dgdft

#endif // _DISTVEC_DECL_HPP_

