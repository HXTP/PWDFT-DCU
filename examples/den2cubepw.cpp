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
/// @file den2cubepw.cpp
/// @brief Utility routine to convert the density files to Gaussian cube
/// format.
/// 
/// This is a sequential code reading the output from PWDFT.
///
/// The input takes the format of DEN.
///
/// NOTE: all units are atomic units (Bohr) here.  
///
///            Example of Gaussian cube format
///            from http://paulbourke.net/dataformats/cube/ 
///
///            CPMD CUBE FILE.
///            OUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z
///            3    0.000000    0.000000    0.000000
///            40    0.283459    0.000000    0.000000
///            40    0.000000    0.283459    0.000000
///            40    0.000000    0.000000    0.283459
///            8    0.000000    5.570575    5.669178    5.593517
///            1    0.000000    5.562867    5.669178    7.428055
///            1    0.000000    7.340606    5.669178    5.111259
///            -0.25568E-04  0.59213E-05  0.81068E-05  0.10868E-04  0.11313E-04  0.35999E-05
///            xxx
///
/// @date 2016-08-12 Original version
#include "dgdft.hpp"

#define _DEBUGlevel_ 0

using namespace dgdft;
using namespace std;
using namespace dgdft::esdf;
using namespace dgdft::scalapack;

using std::vector;
using std::pair;
using std::map;
using std::set;
using std::cin;
using std::cerr;
using std::cout;
using std::ostream;
using std::istream;
using std::istringstream;
using std::ifstream;
using std::ofstream;
using std::setw;
using std::setprecision;
using std::scientific;
using std::fixed;

int main(int argc, char **argv){
  char infoFileName[100];
  char densityFileName[100];
  string outputFileName;
  Point3 domainSizeGlobal;
  Index3 numGridGlobal;
  Index3 numGridFineGlobal;
  Point3 posStartGlobal;
  Int numAtom;
  DblNumVec  densityVec;
  std::vector<Atom> atomList;


  // Initialization
  MPI_Init(&argc, &argv);
  int mpirank, mpisize;
  MPI_Comm_rank( MPI_COMM_WORLD, &mpirank );
  MPI_Comm_size( MPI_COMM_WORLD, &mpisize );

  if( mpisize != 1 ){
    if( mpirank == 0 ){
      cerr << "The current code only supports serial mode." << endl;
    }
    MPI_Finalize();
    return 0;
  }

  // Structure information 
  {
#if ( _DEBUGlevel_ >= 0 )
    cout << "Reading structure file " << endl;
#endif
    sprintf(infoFileName, "STRUCTURE");
    ifstream inputFileStream(infoFileName); iA( inputFileStream.good());


    // Domain
    deserialize(domainSizeGlobal, inputFileStream, NO_MASK);
    deserialize(numGridGlobal, inputFileStream, NO_MASK); 
    deserialize(numGridFineGlobal, inputFileStream, NO_MASK);
    deserialize(posStartGlobal, inputFileStream, NO_MASK);

    // Atom
    deserialize( atomList, inputFileStream, NO_MASK );

    numAtom = atomList.size();

    inputFileStream.close();

#if ( _DEBUGlevel_ >= 0 )
    cout << "number of fine grids in the global domain " << endl << 
      numGridFineGlobal << endl;
#endif
  }

  // Read information
  {
    outputFileName = "DEN.cub";
    //    cout << "Output file name?" << endl;
    //    cin >> outputFileName;
  }


  Real timeStaTotal, timeEndTotal;

  GetTime( timeStaTotal );

  // Construct the density in the global domain
  {
    // Only works for the case without magnetization
    densityVec.Resize(numGridFineGlobal[0]*numGridFineGlobal[1]*numGridFineGlobal[2]);
    SetValue( densityVec, 0.0 );


    ifstream inputFileStream("DEN");
    iA( inputFileStream.good() );

    // Read the gridpos
    std::vector<DblNumVec> gridpos(DIM);
    for( Int d = 0; d < DIM; d++ ){
      deserialize( gridpos[d], inputFileStream, NO_MASK );
    }

    deserialize(densityVec, inputFileStream, NO_MASK);

    if( inputFileStream.is_open() ) inputFileStream.close();
  }

  // Output the density in the Gaussian cube format
  {
    Point3 gridSizeFineGlobal;
    gridSizeFineGlobal[0] = domainSizeGlobal[0] / numGridFineGlobal[0];
    gridSizeFineGlobal[1] = domainSizeGlobal[1] / numGridFineGlobal[1];
    gridSizeFineGlobal[2] = domainSizeGlobal[2] / numGridFineGlobal[2];

    ofstream outputFileStream(outputFileName.c_str());
    iA( outputFileStream.good() );
    // Header
    outputFileStream << "Gaussian cube format, created by den2cube" << endl;
    outputFileStream << "X: Outer loop Y: Middle loop Z: Inner loop" << endl;
    outputFileStream << fixed << setw(9) << numAtom << " " 
      << setw(12) << setprecision(5) 
      << posStartGlobal[0] << " " 
      << posStartGlobal[1] << " " 
      << posStartGlobal[2] << endl;
    outputFileStream << fixed << setw(9) << numGridFineGlobal[0] << " " << setw(12) 
      << setprecision(5) << gridSizeFineGlobal[0]  << " " 
      << 0.0 << " " << 0.0 << endl;
    outputFileStream << fixed << setw(9) << numGridFineGlobal[1] << " " << setw(12) 
      << setprecision(5) << 0.0 << " " << gridSizeFineGlobal[1]  << " " 
      << 0.0 << endl;
    outputFileStream << fixed << setw(9) << numGridFineGlobal[2] << " " << setw(12) 
      << setprecision(5) << 0.0 << " " << 0.0 << " " << gridSizeFineGlobal[2]  << 
      endl;
    for( Int a = 0; a < numAtom; a++ ){
      outputFileStream << fixed << setw(9) << atomList[a].type << " " << setw(12) 
        << setprecision(5) << 0.0 << " " 
        << atomList[a].pos[0]  << " " 
        << atomList[a].pos[1]  << " " 
        << atomList[a].pos[2]  << endl;
    }

    DblNumTns densityGlobal(
        numGridFineGlobal[0],
        numGridFineGlobal[1],
        numGridFineGlobal[2], 
        false, densityVec.Data() );


    //NOTE the special Z-Y-X order here in the Gaussian cube format.
    Int count = 0;
    for(int i = 0; i < numGridFineGlobal[0]; i++)
      for(int j = 0; j < numGridFineGlobal[1]; j++)
        for(int k = 0; k < numGridFineGlobal[2]; k++){
          outputFileStream << scientific << setw(12) << setprecision(5) <<
            densityGlobal(i,j,k);
          count++;
          if( count % 6 == 0 )
            outputFileStream << endl;
        }
    outputFileStream.close();
  }

  GetTime( timeEndTotal );
  cout << "Total processing time " << timeEndTotal - timeStaTotal 
    << " [s]" << endl;

  MPI_Finalize();

  return 0;
}



