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
/// @file global.cpp
/// @brief Global structure.
/// @date 2012-08-01
/// @date 2016-07-12 Update to the coredumper model for debugging
#include    "environment.hpp"
#include    "esdf.hpp"
#include    "periodtable.hpp"
#ifdef _COREDUMPER_
#include <google/coredumper.h>
#endif

namespace dgdft{

// *********************************************************************
// Input parameters
// *********************************************************************
namespace esdf{
ESDFInputParam  esdfParam;
}

// *********************************************************************
// IO
// *********************************************************************
std::ofstream  statusOFS;

// *********************************************************************
// Error handling
// *********************************************************************
void ErrorHandling( const char * msg ){
  statusOFS << std::endl << "ERROR!" << std::endl 
    << msg << std::endl << std::endl;
#ifdef _COREDUMPER_
  int mpirank, mpisize;
  MPI_Comm_rank( MPI_COMM_WORLD, &mpirank );
  MPI_Comm_size( MPI_COMM_WORLD, &mpisize );
  char filename[100];
  sprintf(filename, "core_%d_%d", mpirank, mpisize);

  if( WriteCoreDump(filename) ==0 ) {   
    statusOFS << "success: WriteCoreDump to " << filename << std::endl;
  } else {  
    statusOFS << "failed:  WriteCoreDump to " << filename << std::endl;
  }     
#endif // #ifdef _COREDUMPER_
  throw std::runtime_error( msg );
}


} // namespace dgdft
