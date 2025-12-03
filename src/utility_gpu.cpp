/*
   Copyright (c) 2012 The Regents of the University of California,
   through Lawrence Berkeley National Laboratory.  

Authors: Lin Lin, Weile Jia

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
/// @file utility.cpp
/// @brief Utility subroutines
/// @date 2012-08-12
#include "utility.hpp"

namespace dgdft{

// *********************************************************************
// Spline functions
// *********************************************************************

void spline(int n, double* x, double* y, double* b, double* c, double* d){
  /* 
     the coefficients b(i), c(i), and d(i), i=1,2,...,n are computed
     for a cubic interpolating spline

     s(x) = y(i) + b(i)*(x-x(i)) + c(i)*(x-x(i))**2 + d(i)*(x-x(i))**3

     for  x(i) .le. x .le. x(i+1)

     input..

     n = the number of data points or knots (n.ge.2)
     x = the abscissas of the knots in strictly increasing order
     y = the ordinates of the knots

     output..

     b, c, d  = arrays of spline coefficients as defined above.

     using  p  to denote differentiation,

     y(i) = s(x(i))
     b(i) = sp(x(i))
     c(i) = spp(x(i))/2
     d(i) = sppp(x(i))/6  (derivative from the right)

     the accompanying function subprogram  seval  can be used
     to evaluate the spline.
   */
  int nm1, i;
  double t;

  for(i = 0; i < n; i++){
    b[i] = 0.0;
    c[i] = 0.0;
    d[i] = 0.0;
  }
  nm1 = n-1;
  if ( n < 2 ) {
    ErrorHandling(" SPLINE REQUIRES N >= 2!" );
  }
  if ( n < 3 ){
    b[0] = (y[1]-y[0])/(x[1]-x[0]);
    c[0] = 0;
    d[0] = 0;
    b[1] = b[0];
    c[1] = 0;
    d[1] = 0;
    return;
  }

  /*
     set up tridiagonal system

     b = diagonal, d = offdiagonal, c = right hand side.
   */ 

  d[0] = x[1] - x[0];
  c[1] = (y[1] - y[0])/d[0];
  for(i = 1; i <  nm1; i++){
    d[i] = x[i+1] - x[i];
    b[i] = 2.*(d[i-1] + d[i]);
    c[i+1] = (y[i+1] - y[i])/d[i];
    c[i] = c[i+1] - c[i];
  }

  /* 
     end  onditions.  third derivatives at  x(1)  and  x(n)
     obtained from divided differences.
   */ 
  b[0] = -d[0];
  b[n-1] = -d[n-2];
  c[0] = 0.;
  c[n-1] = 0.;
  if ( n > 3 ){
    c[0] = c[2]/(x[3]-x[1]) - c[1]/(x[2]-x[0]);
    c[n-1] = c[n-2]/(x[n-1]-x[n-3]) - c[n-3]/(x[n-2]-x[n-4]);
    c[0] = c[0]*d[0]*d[0]/(x[3]-x[0]);
    c[n-1] = -c[n-1]*d[n-2]*d[n-2]/(x[n-1]-x[n-4]);
  }

  /* forward elimination */

  for( i = 1; i < n; i++ ){
    t = d[i-1] / b[i-1];
    b[i] = b[i] - t * d[i-1];
    c[i] = c[i] - t * c[i-1];
  }

  /* backward substitution */
  c[n-1] = c[n-1] / b[n-1];
  for( i = n-2; i >= 0; i-- ){
    c[i] = (c[i] - d[i]*c[i+1]) / b[i];
  }

  /* compute polynomial coefficients */
  b[n-1] = (y[n-1] - y[nm1-1])/d[nm1-1] + d[nm1-1]*(c[nm1-1] + 2.*c[n-1]);
  for(i = 0; i < nm1; i++){
    b[i] = (y[i+1] - y[i])/d[i] - d[i]*(c[i+1] + 2.*c[i]);
    d[i] = (c[i+1] - c[i])/d[i];
    c[i] = 3.*c[i];
  }
  c[n-1] = 3.*c[n-1];
  d[n-1] = d[n-2];

}


void seval(double* v, int m, double* u, int n, double* x, 
    double* y, double* b, double* c, double* d){

  /* ***************************************************
   * This SPLINE function is designed specifically for the interpolation
   * part for pseudopotential generation in the electronic structure
   * calculation.  Therefore if u is outside the range [min(x), max(x)],
   * the corresponding v value will be an extrapolation.
   * ***************************************************

   this subroutine evaluates the  spline function

   seval = y(i) + b(i)*(u-x(i)) +  (i)*(u-x(i))**2 + d(i)*(u-x(i))**3

   where  x(i) .lt. u .lt. x(i+1), using horner's rule

   if  u .lt. x(1) then  i = 1  is used.
   if  u .ge. x(n) then  i = n  is used.

   input..

   m = the number of output data points
   n = the number of input data points
   u = the abs issa at which the spline is to be evaluated
   v = the value of the spline function at u
   x,y = the arrays of data absissas and ordinates
   b,c,d = arrays of spline coefficients computed by spline

   if  u  is not in the same interval as the previous  all, then a
   binary sear h is performed to determine the proper interval.
   */

  int i, j, k, l;
  double dx;
  if( n < 2 ){
    ErrorHandling(" SPLINE REQUIRES N >= 2!" );
  }

  for(l = 0; l < m; l++){
    v[l] = 0.0;
  }

  for(l = 0; l < m; l++){
    i = 0;
    if( u[l] < x[0] ){
      i = 0;
    }
    else if( u[l] > x[n-1] ){
      i = n-1;
    }
    else{
      /* calculate the index of u[l] */
      i = 0;
      j = n;
      while( j > i+1 ) {
        k = (i+j)/2;
        if( u[l] < x[k] ) j = k;
        if( u[l] >= x[k] ) i = k;
      }
    }
    /* evaluate spline */
    dx = u[l] - x[i];
    v[l] = y[i] + dx*(b[i] + dx*(c[i] + dx*d[i]));
  }
  return;
}

// *********************************************************************
// Generating grids in a domain
// *********************************************************************

void GenerateLGLMeshWeightOnly(double* x, double* w, int Nm1)
{
  int i, j;
  double pi = 4.0 * atan(1.0);
  double err, tol = 1e-15;
  std::vector<double> xold;
  int N = Nm1;
  int N1 = N + 1;
  // Only for the three-term recursion
  DblNumMat PMat(N1, 3);
  SetValue( PMat, 0.0 );

  xold.resize(N1);

  double *P0 = PMat.VecData(0);
  double *P1 = PMat.VecData(1);
  double *P2 = PMat.VecData(2);

  for (i=0; i<N1; i++){
    x[i] = cos(pi*(N1-i-1)/(double)N);
  }

  do{
    for (i=0; i<N1; i++){
      xold[i] = x[i];
      P0[i] = 1.0;
      P1[i] = x[i];
    }
    for (j=2; j<N1; j++){
      for (i=0; i<N1; i++){
        P2[i] = ((2.0*j-1.0)*x[i]*P1[i] - (j-1)*P0[i])/j;
        P0[i] = P1[i];
        P1[i] = P2[i];
      }
    }

    for (i=0; i<N1; i++){
      x[i] = xold[i] - (x[i]*P1[i] - P0[i])/(N1*P1[i]);
    }

    err = 0.0;
    for (i=0; i<N1; i++){
      if (err < fabs(xold[i] - x[i])){
        err = fabs(xold[i] - x[i]);
      }
    }
  } while(err>tol);

  for (i=0; i<N1; i++){
    w[i] = 2.0/(N*N1*P1[i]*P1[i]);
  }

  return;
}

void GenerateLGLMeshWeightOnly(
    DblNumVec&         x, 
    DblNumVec&         w, 
    Int                N)
{
  x.Resize( N );
  w.Resize( N );
  GenerateLGLMeshWeightOnly( x.Data(), w.Data(), N-1 );

  return;
}


void GenerateLGL(double* x, double* w, double* P, double* D, int Nm1)
{
  int i, j;
  double pi = 4.0 * atan(1.0);
  double err, tol = 1e-15;
  std::vector<double> xold;
  int N = Nm1;
  int N1 = N + 1;

  xold.resize(N1);

  for (i=0; i<N1; i++){
    x[i] = cos(pi*(N1-i-1)/(double)N);
  }

  for (j=0; j<N1; j++){
    for (i=0; i<N1; i++){
      P[j*N1+i] = 0;
    }
  }

  do{
    for (i=0; i<N1; i++){
      xold[i] = x[i]; 
      P[i] = 1.0; 
      P[N1+i] = x[i];
    }
    for (j=2; j<N1; j++){
      for (i=0; i<N1; i++){
        P[j*N1+i] = ((2*j-1)*x[i]*P[(j-1)*N1+i] - (j-1)*P[(j-2)*N1+i])/j;
      }
    }

    for (i=0; i<N1; i++){
      x[i] = xold[i] - (x[i]*P[N*N1+i] - P[(N-1)*N1+i])/(N1*P[N*N1+i]);
    }

    err = 0.0;
    for (i=0; i<N1; i++){
      if (err < fabs(xold[i] - x[i])){
        err = fabs(xold[i] - x[i]);
      }
    }
  } while(err>tol);

  for (i=0; i<N1; i++){
    w[i] = 2.0/(N*N1*P[N*N1+i]*P[N*N1+i]);
  }

  for (j=0; j<N1; j++){
    for (i=0; i<N1; i++){
      if (i!=j) {
        D[j*N1+i] = P[N*N1+i]/P[N*N1+j]/(x[i] - x[j]);
      }
      else if (i==0){
        D[j*N1+i] = - N*N1/4.0;
      }
      else if (i==N1-1){
        D[j*N1+i] = N*N1/4.0;
      }
      else{
        D[j*N1+i] = 0.0;      
      }
    }
  }

  return;
}

void GenerateLGL(
    DblNumVec&         x, 
    DblNumVec&         w, 
    DblNumMat&         P,
    DblNumMat&         D,
    Int                N)
{
  x.Resize( N );
  w.Resize( N );
  P.Resize( N, N );
  D.Resize( N, N );
  GenerateLGL( x.Data(), w.Data(), P.Data(), D.Data(), N-1 );

  return;
}



void
UniformMesh ( const Domain &dm, std::vector<DblNumVec> &gridpos )
{
  gridpos.resize(DIM);
  for (Int d=0; d<DIM; d++) {
    gridpos[d].Resize(dm.numGrid[d]);
    Real h = dm.length[d] / dm.numGrid[d];
    for (Int i=0; i < dm.numGrid[d]; i++) {
      gridpos[d](i) = dm.posStart[d] + Real(i)*h;
    }
  }

  return ;
}        // -----  end of function UniformMesh  ----- 


void
UniformMeshFine ( const Domain &dm, std::vector<DblNumVec> &gridpos )
{
  gridpos.resize(DIM);
  for (Int d=0; d<DIM; d++) {
    gridpos[d].Resize(dm.numGridFine[d]);
    Real h = dm.length[d] / dm.numGridFine[d];
    for (Int i=0; i < dm.numGridFine[d]; i++) {
      gridpos[d](i) = dm.posStart[d] + Real(i)*h;
    }
  }

  return ;
}        // -----  end of function UniformMesh  ----- 


void
LGLMesh ( const Domain &dm, const Index3& numGrid, std::vector<DblNumVec> &gridpos )
{
  gridpos.resize(DIM);
  for (Int d=0; d<DIM; d++) {
    gridpos[d].Resize( numGrid[d] );

    DblNumVec  mesh;
    DblNumVec  dummyW;
    DblNumMat  dummyP, dummyD;
    GenerateLGL( mesh, dummyW, dummyP, dummyD, numGrid[d] );
    for( Int i = 0; i < numGrid[d]; i++ ){
      gridpos[d][i] = dm.posStart[d] + 
        ( mesh[i] + 1.0 ) * dm.length[d] * 0.5;
    }
  }

  return ;
}        // -----  end of function LGLMesh  ----- 

// *********************************************************************
// IO functions
// *********************************************************************
//---------------------------------------------------------
Int SeparateRead(std::string name, std::istringstream& is)
{
  int mpirank;  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  int mpisize;  MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
  //
  char filename[100];
  sprintf(filename, "%s_%d_%d", name.c_str(), mpirank, mpisize);  
  std::ifstream fin(filename);
  if( !fin.good() ){
    ErrorHandling( "File cannot be open!" );
  }

  is.str( std::string(std::istreambuf_iterator<char>(fin), std::istreambuf_iterator<char>()) );
  fin.close();
  return 0;
}

//---------------------------------------------------------
Int SeparateRead(std::string name, std::istringstream& is, Int outputIndex)
{
  char filename[100];
  sprintf(filename, "%s_%d", name.c_str(), outputIndex);
  std::ifstream fin(filename);
  if( !fin.good() ){
    ErrorHandling( "File cannot be open!" );
  }

  is.str( std::string(std::istreambuf_iterator<char>(fin), std::istreambuf_iterator<char>()) );
  fin.close();
  return 0;
}


//---------------------------------------------------------
Int SeparateWrite(std::string name, std::ostringstream& os)
{
  int mpirank;  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  int mpisize;  MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
  //
  char filename[100];
  sprintf(filename, "%s_%d_%d", name.c_str(), mpirank, mpisize);
  std::ofstream fout(filename);
  if( !fout.good() ){
    ErrorHandling( "File cannot be open!" );
  }
  fout<<os.str();
  fout.close();
  return 0;
}


//---------------------------------------------------------
Int SeparateWrite(std::string name, std::ostringstream& os, Int outputIndex)
{
  char filename[100];
  sprintf(filename, "%s_%d", name.c_str(), outputIndex);
  std::ofstream fout(filename);
  if( !fout.good() ){
    ErrorHandling( "File cannot be open!" );
  }
  fout<<os.str();
  fout.close();
  return 0;
}

//---------------------------------------------------------
Int SharedRead(std::string name, std::istringstream& is)
{
  MPI_Barrier(MPI_COMM_WORLD);
  int mpirank;  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  int mpisize;  MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
  //
  std::vector<char> tmpstr;
  if(mpirank==0) {
    std::ifstream fin(name.c_str());
    if( !fin.good() ){
      ErrorHandling( "File cannot be open!" );
    }
    //std::string str(std::istreambuf_iterator<char>(fin), std::istreambuf_iterator<char>());
    //tmpstr.insert(tmpstr.end(), str.begin(), str.end());
    tmpstr.insert(tmpstr.end(), std::istreambuf_iterator<char>(fin), std::istreambuf_iterator<char>());
    fin.close();
    int size = tmpstr.size();    
    MPI_Bcast((void*)&size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast((void*)&(tmpstr[0]), size, MPI_BYTE, 0, MPI_COMM_WORLD);
  } else {
    int size;
    MPI_Bcast((void*)&size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    tmpstr.resize(size);
    MPI_Bcast((void*)&(tmpstr[0]), size, MPI_BYTE, 0, MPI_COMM_WORLD);
  }
  is.str( std::string(tmpstr.begin(), tmpstr.end()) );
  //
  MPI_Barrier(MPI_COMM_WORLD);
  return 0;
}

//---------------------------------------------------------
Int SharedWrite(std::string name, std::ostringstream& os)
{
  MPI_Barrier(MPI_COMM_WORLD);
  int mpirank;  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  int mpisize;  MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
  //
  if(mpirank==0) {
    std::ofstream fout(name.c_str());
    if( !fout.good() ){
      ErrorHandling( "File cannot be open!" );
    }
    fout<<os.str();
    fout.close();
  }
  MPI_Barrier(MPI_COMM_WORLD);
  return 0;
}


//---------------------------------------------------------
Int SeparateWriteAscii(std::string name, std::ostringstream& os)
{
  MPI_Barrier(MPI_COMM_WORLD);
  int mpirank;  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  int mpisize;  MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
  //
  char filename[100];
  sprintf(filename, "%s_%d_%d", name.c_str(), mpirank, mpisize);
  std::ofstream fout(filename, std::ios::trunc);
  if( !fout.good() ){
    ErrorHandling( "File cannot be open!" );
  }
  fout<<os.str();
  fout.close();
  //
  MPI_Barrier(MPI_COMM_WORLD);
  return 0;
}


void AlltoallForward( DblNumMat& A, DblNumMat& B, MPI_Comm comm )
{

  int mpirank, mpisize;
  MPI_Comm_rank( comm, &mpirank );
  MPI_Comm_size( comm, &mpisize );

  Int height = A.m();
  Int widthTemp = A.n();

  Int width = 0;
  MPI_Allreduce( &widthTemp, &width, 1, MPI_INT, MPI_SUM, comm );

  Int widthBlocksize = width / mpisize;
  Int heightBlocksize = height / mpisize;
  Int widthLocal = widthBlocksize;
  Int heightLocal = heightBlocksize;

  if(mpirank < (width % mpisize)){
    widthLocal = widthBlocksize + 1;
  }
  
  if(mpirank < (height % mpisize)){
    heightLocal = heightBlocksize + 1;
  }
  
  DblNumVec sendbuf(height*widthLocal); 
  DblNumVec recvbuf(heightLocal*width);
  IntNumVec sendcounts(mpisize);
  IntNumVec recvcounts(mpisize);
  IntNumVec senddispls(mpisize);
  IntNumVec recvdispls(mpisize);
  IntNumMat  sendk( height, widthLocal );
  IntNumMat  recvk( heightLocal, width );

  for( Int k = 0; k < mpisize; k++ ){ 
    sendcounts[k] = heightBlocksize * widthLocal;
    if( k < (height % mpisize)){
      sendcounts[k] = sendcounts[k] + widthLocal;  
    }
  }

  for( Int k = 0; k < mpisize; k++ ){ 
    recvcounts[k] = heightLocal * widthBlocksize;
    if( k < (width % mpisize)){
      recvcounts[k] = recvcounts[k] + heightLocal;  
    }
  }

  senddispls[0] = 0;
  recvdispls[0] = 0;
  for( Int k = 1; k < mpisize; k++ ){ 
    senddispls[k] = senddispls[k-1] + sendcounts[k-1];
    recvdispls[k] = recvdispls[k-1] + recvcounts[k-1];
  }

  if((height % mpisize) == 0){
    for( Int j = 0; j < widthLocal; j++ ){ 
      for( Int i = 0; i < height; i++ ){
        sendk(i, j) = senddispls[i / heightBlocksize] + j * heightBlocksize + i % heightBlocksize;
      } 
    }
  }
  else{
    for( Int j = 0; j < widthLocal; j++ ){ 
      for( Int i = 0; i < height; i++ ){
        if( i < ((height % mpisize) * (heightBlocksize+1)) ){
          sendk(i, j) = senddispls[i / (heightBlocksize+1)] + j * (heightBlocksize+1) + i % (heightBlocksize+1);
        }
        else {
          sendk(i, j) = senddispls[(height % mpisize) + (i-(height % mpisize)*(heightBlocksize+1))/heightBlocksize]
            + j * heightBlocksize + (i-(height % mpisize)*(heightBlocksize+1)) % heightBlocksize;
        }
      }
    }
  }

  for( Int j = 0; j < width; j++ ){ 
    for( Int i = 0; i < heightLocal; i++ ){
      recvk(i, j) = recvdispls[j % mpisize] + (j / mpisize) * heightLocal + i;
    }
  }

  for( Int j = 0; j < widthLocal; j++ ){ 
    for( Int i = 0; i < height; i++ ){
      sendbuf[sendk(i, j)] = A(i, j); 
    }
  }
  MPI_Alltoallv( &sendbuf[0], &sendcounts[0], &senddispls[0], MPI_DOUBLE, 
      &recvbuf[0], &recvcounts[0], &recvdispls[0], MPI_DOUBLE, comm );
  for( Int j = 0; j < width; j++ ){ 
    for( Int i = 0; i < heightLocal; i++ ){
      B(i, j) = recvbuf[recvk(i, j)];
    }
  }


  return ;
}        // -----  end of function AlltoallForward ----- 

#ifdef GPU
void GPU_AlltoallForward( cuDblNumMat& cu_A, cuDblNumMat& cu_B, MPI_Comm comm )
{

  int mpirank, mpisize;
  MPI_Comm_rank( comm, &mpirank );
  MPI_Comm_size( comm, &mpisize );

  Int height = cu_A.m();
  Int widthTemp = cu_A.n();

  Int width = 0;
  MPI_Allreduce( &widthTemp, &width, 1, MPI_INT, MPI_SUM, comm );

  Int widthBlocksize = width / mpisize;
  Int heightBlocksize = height / mpisize;
  Int widthLocal = widthBlocksize;
  Int heightLocal = heightBlocksize;

  if(mpirank < (width % mpisize)){
    widthLocal = widthBlocksize + 1;
  }
  
  if(mpirank < (height % mpisize)){
    heightLocal = heightBlocksize + 1;
  }
  
  DblNumVec sendbuf(height*widthLocal); 
  DblNumVec recvbuf(heightLocal*width);
  IntNumVec sendcounts(mpisize);
  IntNumVec recvcounts(mpisize);
  IntNumVec senddispls(mpisize);
  IntNumVec recvdispls(mpisize);
  IntNumMat  sendk( height, widthLocal );
  IntNumMat  recvk( heightLocal, width );

  for( Int k = 0; k < mpisize; k++ ){ 
    sendcounts[k] = heightBlocksize * widthLocal;
    if( k < (height % mpisize)){
      sendcounts[k] = sendcounts[k] + widthLocal;  
    }
  }

  for( Int k = 0; k < mpisize; k++ ){ 
    recvcounts[k] = heightLocal * widthBlocksize;
    if( k < (width % mpisize)){
      recvcounts[k] = recvcounts[k] + heightLocal;  
    }
  }

  senddispls[0] = 0;
  recvdispls[0] = 0;
  for( Int k = 1; k < mpisize; k++ ){ 
    senddispls[k] = senddispls[k-1] + sendcounts[k-1];
    recvdispls[k] = recvdispls[k-1] + recvcounts[k-1];
  }

  cuIntNumMat  cu_sendk( height, widthLocal );
  cuIntNumMat  cu_recvk( heightLocal, width );
  cuIntNumVec  cu_senddispls(mpisize);
  cuIntNumVec  cu_recvdispls(mpisize);
  cuDblNumVec  cu_recvbuf(heightLocal*width);
  cuDblNumVec  cu_sendbuf(height*widthLocal); 

  cu_senddispls.CopyFrom( senddispls );
  cu_recvdispls.CopyFrom( recvdispls );
 
  cuda_cal_sendk( cu_sendk.Data(), cu_senddispls.Data(), widthLocal, height, heightBlocksize, mpisize );
  cuda_cal_recvk( cu_recvk.Data(), cu_recvdispls.Data(), width, heightLocal, mpisize ); 

  cuda_mapping_to_buf( cu_sendbuf.Data(), cu_A.Data(), cu_sendk.Data(), height*widthLocal);
  cu_sendbuf.CopyTo( sendbuf );
  
  MPI_Alltoallv( &sendbuf[0], &sendcounts[0], &senddispls[0], MPI_DOUBLE, 
      &recvbuf[0], &recvcounts[0], &recvdispls[0], MPI_DOUBLE, comm );

  cu_recvbuf.CopyFrom( recvbuf );
  cuda_mapping_from_buf(cu_B.Data(), cu_recvbuf.Data(), cu_recvk.Data(), heightLocal*width);
 

  return ;
}        // -----  end of function GPU_AlltoallForward ----- 


void GPU_AlltoallBackward( cuDblNumMat& cu_A, cuDblNumMat& cu_B, MPI_Comm comm )
{

  int mpirank, mpisize;
  MPI_Comm_rank( comm, &mpirank );
  MPI_Comm_size( comm, &mpisize );

  Int height = cu_B.m();
  Int widthTemp = cu_B.n();

  Int width = 0;
  MPI_Allreduce( &widthTemp, &width, 1, MPI_INT, MPI_SUM, comm );

  Int widthBlocksize = width / mpisize;
  Int heightBlocksize = height / mpisize;
  Int widthLocal = widthBlocksize;
  Int heightLocal = heightBlocksize;

  if(mpirank < (width % mpisize)){
    widthLocal = widthBlocksize + 1;
  }

  if(mpirank < (height % mpisize)){
    heightLocal = heightBlocksize + 1;
  }

  DblNumVec sendbuf(height*widthLocal); 
  DblNumVec recvbuf(heightLocal*width);
  IntNumVec sendcounts(mpisize);
  IntNumVec recvcounts(mpisize);
  IntNumVec senddispls(mpisize);
  IntNumVec recvdispls(mpisize);

  for( Int k = 0; k < mpisize; k++ ){ 
    sendcounts[k] = heightBlocksize * widthLocal;
    if( k < (height % mpisize)){
      sendcounts[k] = sendcounts[k] + widthLocal;  
    }
  }

  for( Int k = 0; k < mpisize; k++ ){ 
    recvcounts[k] = heightLocal * widthBlocksize;
    if( k < (width % mpisize)){
      recvcounts[k] = recvcounts[k] + heightLocal;  
    }
  }

  senddispls[0] = 0;
  recvdispls[0] = 0;
  for( Int k = 1; k < mpisize; k++ ){ 
    senddispls[k] = senddispls[k-1] + sendcounts[k-1];
    recvdispls[k] = recvdispls[k-1] + recvcounts[k-1];
  }

  cuIntNumMat  cu_sendk( height, widthLocal );
  cuIntNumMat  cu_recvk( heightLocal, width );
  cuIntNumVec  cu_senddispls(mpisize);
  cuIntNumVec  cu_recvdispls(mpisize);
  cuDblNumVec  cu_recvbuf(heightLocal*width);
  cuDblNumVec  cu_sendbuf(height*widthLocal); 

  cu_senddispls.CopyFrom( senddispls );
  cu_recvdispls.CopyFrom( recvdispls );
 
  cuda_cal_sendk( cu_sendk.Data(), cu_senddispls.Data(), widthLocal, height, heightBlocksize, mpisize );
  cuda_cal_recvk( cu_recvk.Data(), cu_recvdispls.Data(), width, heightLocal, mpisize ); 

  cuda_mapping_to_buf( cu_recvbuf.Data(), cu_A.Data(), cu_recvk.Data(), heightLocal*width);
  cu_recvbuf.CopyTo( recvbuf );
  
  MPI_Alltoallv( &recvbuf[0], &recvcounts[0], &recvdispls[0], MPI_DOUBLE, 
      &sendbuf[0], &sendcounts[0], &senddispls[0], MPI_DOUBLE, comm );

  cu_sendbuf.CopyFrom( sendbuf );
  cuda_mapping_from_buf(cu_B.Data(), cu_sendbuf.Data(), cu_sendk.Data(), height*widthLocal);
  
  return ;
}        // -----  end of function GPU_AlltoallBackward ----- 


#endif


void AlltoallForward( CpxNumMat& A, CpxNumMat& B, MPI_Comm comm )
{

  int mpirank, mpisize;
  MPI_Comm_rank( comm, &mpirank );
  MPI_Comm_size( comm, &mpisize );

  Int height = A.m();
  Int widthTemp = A.n();

  Int width = 0;
  MPI_Allreduce( &widthTemp, &width, 1, MPI_INT, MPI_SUM, comm );

  Int widthBlocksize = width / mpisize;
  Int heightBlocksize = height / mpisize;
  Int widthLocal = widthBlocksize;
  Int heightLocal = heightBlocksize;

  if(mpirank < (width % mpisize)){
    widthLocal = widthBlocksize + 1;
  }
  
  if(mpirank < (height % mpisize)){
    heightLocal = heightBlocksize + 1;
  }
  
  CpxNumVec sendbuf(height*widthLocal); 
  CpxNumVec recvbuf(heightLocal*width);
  IntNumVec sendcounts(mpisize);
  IntNumVec recvcounts(mpisize);
  IntNumVec senddispls(mpisize);
  IntNumVec recvdispls(mpisize);
  IntNumMat  sendk( height, widthLocal );
  IntNumMat  recvk( heightLocal, width );

  for( Int k = 0; k < mpisize; k++ ){ 
    sendcounts[k] = heightBlocksize * widthLocal;
    if( k < (height % mpisize)){
      sendcounts[k] = sendcounts[k] + widthLocal;  
    }
  }

  for( Int k = 0; k < mpisize; k++ ){ 
    recvcounts[k] = heightLocal * widthBlocksize;
    if( k < (width % mpisize)){
      recvcounts[k] = recvcounts[k] + heightLocal;  
    }
  }

  senddispls[0] = 0;
  recvdispls[0] = 0;
  for( Int k = 1; k < mpisize; k++ ){ 
    senddispls[k] = senddispls[k-1] + sendcounts[k-1];
    recvdispls[k] = recvdispls[k-1] + recvcounts[k-1];
  }

  if((height % mpisize) == 0){
    for( Int j = 0; j < widthLocal; j++ ){ 
      for( Int i = 0; i < height; i++ ){
        sendk(i, j) = senddispls[i / heightBlocksize] + j * heightBlocksize + i % heightBlocksize;
      } 
    }
  }
  else{
    for( Int j = 0; j < widthLocal; j++ ){ 
      for( Int i = 0; i < height; i++ ){
        if( i < ((height % mpisize) * (heightBlocksize+1)) ){
          sendk(i, j) = senddispls[i / (heightBlocksize+1)] + j * (heightBlocksize+1) + i % (heightBlocksize+1);
        }
        else {
          sendk(i, j) = senddispls[(height % mpisize) + (i-(height % mpisize)*(heightBlocksize+1))/heightBlocksize]
            + j * heightBlocksize + (i-(height % mpisize)*(heightBlocksize+1)) % heightBlocksize;
        }
      }
    }
  }

  for( Int j = 0; j < width; j++ ){ 
    for( Int i = 0; i < heightLocal; i++ ){
      recvk(i, j) = recvdispls[j % mpisize] + (j / mpisize) * heightLocal + i;
    }
  }

  for( Int j = 0; j < widthLocal; j++ ){ 
    for( Int i = 0; i < height; i++ ){
      sendbuf[sendk(i, j)] = A(i, j); 
    }
  }
  MPI_Alltoallv( &sendbuf[0], &sendcounts[0], &senddispls[0], MPI_DOUBLE_COMPLEX, 
      &recvbuf[0], &recvcounts[0], &recvdispls[0], MPI_DOUBLE_COMPLEX, comm );
  for( Int j = 0; j < width; j++ ){ 
    for( Int i = 0; i < heightLocal; i++ ){
      B(i, j) = recvbuf[recvk(i, j)];
    }
  }


  return ;
}        // -----  end of function AlltoallForward ----- 


void AlltoallBackward( DblNumMat& A, DblNumMat& B, MPI_Comm comm )
{

  int mpirank, mpisize;
  MPI_Comm_rank( comm, &mpirank );
  MPI_Comm_size( comm, &mpisize );

  Int height = B.m();
  Int widthTemp = B.n();

  Int width = 0;
  MPI_Allreduce( &widthTemp, &width, 1, MPI_INT, MPI_SUM, comm );

  Int widthBlocksize = width / mpisize;
  Int heightBlocksize = height / mpisize;
  Int widthLocal = widthBlocksize;
  Int heightLocal = heightBlocksize;

  if(mpirank < (width % mpisize)){
    widthLocal = widthBlocksize + 1;
  }

  if(mpirank < (height % mpisize)){
    heightLocal = heightBlocksize + 1;
  }

  DblNumVec sendbuf(height*widthLocal); 
  DblNumVec recvbuf(heightLocal*width);
  IntNumVec sendcounts(mpisize);
  IntNumVec recvcounts(mpisize);
  IntNumVec senddispls(mpisize);
  IntNumVec recvdispls(mpisize);
  IntNumMat  sendk( height, widthLocal );
  IntNumMat  recvk( heightLocal, width );

  for( Int k = 0; k < mpisize; k++ ){ 
    sendcounts[k] = heightBlocksize * widthLocal;
    if( k < (height % mpisize)){
      sendcounts[k] = sendcounts[k] + widthLocal;  
    }
  }

  for( Int k = 0; k < mpisize; k++ ){ 
    recvcounts[k] = heightLocal * widthBlocksize;
    if( k < (width % mpisize)){
      recvcounts[k] = recvcounts[k] + heightLocal;  
    }
  }

  senddispls[0] = 0;
  recvdispls[0] = 0;
  for( Int k = 1; k < mpisize; k++ ){ 
    senddispls[k] = senddispls[k-1] + sendcounts[k-1];
    recvdispls[k] = recvdispls[k-1] + recvcounts[k-1];
  }

  if((height % mpisize) == 0){
    for( Int j = 0; j < widthLocal; j++ ){ 
      for( Int i = 0; i < height; i++ ){
        sendk(i, j) = senddispls[i / heightBlocksize] + j * heightBlocksize + i % heightBlocksize;
      } 
    }
  }
  else{
    for( Int j = 0; j < widthLocal; j++ ){ 
      for( Int i = 0; i < height; i++ ){
        if( i < ((height % mpisize) * (heightBlocksize+1)) ){
          sendk(i, j) = senddispls[i / (heightBlocksize+1)] + j * (heightBlocksize+1) + i % (heightBlocksize+1);
        }
        else {
          sendk(i, j) = senddispls[(height % mpisize) + (i-(height % mpisize)*(heightBlocksize+1))/heightBlocksize]
            + j * heightBlocksize + (i-(height % mpisize)*(heightBlocksize+1)) % heightBlocksize;
        }
      }
    }
  }

  for( Int j = 0; j < width; j++ ){ 
    for( Int i = 0; i < heightLocal; i++ ){
      recvk(i, j) = recvdispls[j % mpisize] + (j / mpisize) * heightLocal + i;
    }
  }

  for( Int j = 0; j < width; j++ ){ 
    for( Int i = 0; i < heightLocal; i++ ){
      recvbuf[recvk(i, j)] = A(i, j);
    }
  }
  MPI_Alltoallv( &recvbuf[0], &recvcounts[0], &recvdispls[0], MPI_DOUBLE, 
      &sendbuf[0], &sendcounts[0], &senddispls[0], MPI_DOUBLE, comm );
  for( Int j = 0; j < widthLocal; j++ ){ 
    for( Int i = 0; i < height; i++ ){
      B(i, j) = sendbuf[sendk(i, j)]; 
    }
  }


  return ;
}        // -----  end of function AlltoallBackward ----- 

void AlltoallBackward( CpxNumMat& A, CpxNumMat& B, MPI_Comm comm )
{

  int mpirank, mpisize;
  MPI_Comm_rank( comm, &mpirank );
  MPI_Comm_size( comm, &mpisize );

  Int height = B.m();
  Int widthTemp = B.n();

  Int width = 0;
  MPI_Allreduce( &widthTemp, &width, 1, MPI_INT, MPI_SUM, comm );

  Int widthBlocksize = width / mpisize;
  Int heightBlocksize = height / mpisize;
  Int widthLocal = widthBlocksize;
  Int heightLocal = heightBlocksize;

  if(mpirank < (width % mpisize)){
    widthLocal = widthBlocksize + 1;
  }

  if(mpirank < (height % mpisize)){
    heightLocal = heightBlocksize + 1;
  }

  CpxNumVec sendbuf(height*widthLocal); 
  CpxNumVec recvbuf(heightLocal*width);
  IntNumVec sendcounts(mpisize);
  IntNumVec recvcounts(mpisize);
  IntNumVec senddispls(mpisize);
  IntNumVec recvdispls(mpisize);
  IntNumMat  sendk( height, widthLocal );
  IntNumMat  recvk( heightLocal, width );

  for( Int k = 0; k < mpisize; k++ ){ 
    sendcounts[k] = heightBlocksize * widthLocal;
    if( k < (height % mpisize)){
      sendcounts[k] = sendcounts[k] + widthLocal;  
    }
  }

  for( Int k = 0; k < mpisize; k++ ){ 
    recvcounts[k] = heightLocal * widthBlocksize;
    if( k < (width % mpisize)){
      recvcounts[k] = recvcounts[k] + heightLocal;  
    }
  }

  senddispls[0] = 0;
  recvdispls[0] = 0;
  for( Int k = 1; k < mpisize; k++ ){ 
    senddispls[k] = senddispls[k-1] + sendcounts[k-1];
    recvdispls[k] = recvdispls[k-1] + recvcounts[k-1];
  }

  if((height % mpisize) == 0){
    for( Int j = 0; j < widthLocal; j++ ){ 
      for( Int i = 0; i < height; i++ ){
        sendk(i, j) = senddispls[i / heightBlocksize] + j * heightBlocksize + i % heightBlocksize;
      } 
    }
  }
  else{
    for( Int j = 0; j < widthLocal; j++ ){ 
      for( Int i = 0; i < height; i++ ){
        if( i < ((height % mpisize) * (heightBlocksize+1)) ){
          sendk(i, j) = senddispls[i / (heightBlocksize+1)] + j * (heightBlocksize+1) + i % (heightBlocksize+1);
        }
        else {
          sendk(i, j) = senddispls[(height % mpisize) + (i-(height % mpisize)*(heightBlocksize+1))/heightBlocksize]
            + j * heightBlocksize + (i-(height % mpisize)*(heightBlocksize+1)) % heightBlocksize;
        }
      }
    }
  }

  for( Int j = 0; j < width; j++ ){ 
    for( Int i = 0; i < heightLocal; i++ ){
      recvk(i, j) = recvdispls[j % mpisize] + (j / mpisize) * heightLocal + i;
    }
  }

  for( Int j = 0; j < width; j++ ){ 
    for( Int i = 0; i < heightLocal; i++ ){
      recvbuf[recvk(i, j)] = A(i, j);
    }
  }
  MPI_Alltoallv( &recvbuf[0], &recvcounts[0], &recvdispls[0], MPI_DOUBLE_COMPLEX, 
      &sendbuf[0], &sendcounts[0], &senddispls[0], MPI_DOUBLE_COMPLEX, comm );
  for( Int j = 0; j < widthLocal; j++ ){ 
    for( Int i = 0; i < height; i++ ){
      B(i, j) = sendbuf[sendk(i, j)]; 
    }
  }


  return ;
}        // -----  end of function AlltoallBackward ----- 


// serialize/deserialize the pseudopot

Int serialize(const PseudoPot& val, std::ostream& os, const std::vector<Int>& mask)
{
  serialize( val.pseudoCharge,        os, mask );
  serialize( val.vLocalSR,            os, mask );
  serialize( val.vnlList,             os, mask );
  // No need to serialize the communicator
  return 0;
}

Int deserialize(PseudoPot& val, std::istream& is, const std::vector<Int>& mask)
{
  deserialize( val.pseudoCharge,      is, mask );
  deserialize( val.vLocalSR,          is, mask );
  deserialize( val.vnlList,           is, mask );
  return 0;
}

void findMin(NumMat<Real>& A, const int Dim, NumVec<Int>& Imin){
  int n = A.n_;
  int m = A.m_;
  if (Dim == 0){ 
    Imin.Resize(n);
    Int* Iptr = Imin.Data();
    for (int i = 0; i < n; i++){
      double* temp = A.VecData(i);
      Iptr[i] = std::distance(temp,std::min_element(temp,temp+m));
    }
  } else {
    Real* Aptr = A.Data();
    DblNumVec amin(m,1,Aptr);
    Imin.Resize(m);
    SetValue(Imin,0);
    Int* Iptr = Imin.Data();
    Real* aptr = amin.Data();
    for (int j = 1; j < n; j++){
      for (int i = 0; i < m; i++){
        if (Aptr[i+j*m] < aptr[i]){
          aptr[i] = Aptr[i+j*m];
          Iptr[i] = j;
        }
      }
    }
  }
}

void findMin(NumMat<Real>& A, const int Dim, NumVec<Int>& Imin, NumVec<Real>& amin){
  int n = A.n_;
  int m = A.m_;
  if (Dim == 0){ 
    Imin.Resize(n);
    amin.Resize(n);
    Int* Iptr = Imin.Data();
    Real* aptr = amin.Data();
    Int d;
    for (int i = 0; i < n; i++){
      double* temp = A.VecData(i);
      d = std::distance(temp,std::min_element(temp,temp+m));
      Iptr[i] = d;
      aptr[i] = temp[d];
    }
  } else {
    Real* Aptr = A.Data();
    amin = DblNumVec(m,1,Aptr);
    Imin.Resize(m);
    SetValue(Imin,0);
    Int* Iptr = Imin.Data();
    Real* aptr = amin.Data();
    for (int j = 1; j < n; j++){
      for (int i = 0; i < m; i++){
        if (Aptr[i+j*m] < aptr[i]){
          aptr[i] = Aptr[i+j*m];
          Iptr[i] = j;
        }
      }
    } 
  }
}

void pdist2(NumMat<Real>& A, NumMat<Real>& B, NumMat<Real>& D){
  D.Resize(A.m_, B.m_);
  Int Am = A.m_;
  Int Bm = B.m_;
  Real* Dptr = D.Data();
  Real* Aptr = A.Data();
  Real* Bptr = B.Data();
  
  Real d1,d2,d3;
  for (int j = 0; j < Bm;  j++) {
    for (int i = 0; i < Am; i++) {
      d1 = Aptr[i] - Bptr[j];
      d2 = Aptr[i+Am] - Bptr[j+Bm];
      d3 = Aptr[i+2*Am] - Bptr[j+2*Bm];
      Dptr[j*Am+i] = d1*d1 + d2*d2 + d3*d3;
    }
  }
}

void unique(NumVec<Int>& Index){
  Sort(Index);
  Int* Ipt = Index.Data();
  Int* it = std::unique(Ipt, Ipt + Index.m_);
  std::vector<Int> temp(Ipt, it); 
  delete[] Index.Data();
  Index.m_ = temp.size();
  Index.data_ = new Int[Index.m_];
  Ipt = Index.Data();
  for (int i = 0; i < Index.m_; i++){
    Ipt[i] = temp[i];
  }
}

void KMEAN(Int n, NumVec<Real>& weight, Int& rk, Real KmeansTolerance, 
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
  int nG[3];
  for (int i = 0; i < 3; i++){
    len[i] = dm.length[i];
    nG[i] = dm.numGrid[i];
    dx[i] = len[i]/nG[i];
  }

  for (int i = 0; i < nptLocal; i++){
    tmp = index[indexSta+i];
    indexLocal[i] = tmp;
    wlptr[i] = wptr[tmp];
    glptr[i] = (tmp%nG[1])*dx[0];
    glptr[i+nptLocal] = (tmp%(nG[1]*nG[2])-glptr[i])/nG[1]*dx[2];
    glptr[i+2*nptLocal] = (tmp-glptr[i]-glptr[i+nptLocal]*nG[1])/(nG[1]*nG[2])*dx[2];
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
    Cptr[i] = (tmp%nG[1])*dx[0];
    Cptr[i+rk] = (tmp%(nG[1]*nG[2])-Cptr[i])/nG[1]*dx[2];
    Cptr[i+2*rk] = (tmp-Cptr[i]-Cptr[i+rk]*nG[1])/(nG[1]*nG[2])*dx[2];
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
    statusOFS<< flag << " ";
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

void tridsolve(int n, double* d, double* e, double* f, double* x)
{
  // solve the tridiagonal system Ax=b
  // d[i] = a(i,i)
  // e[i] = a(i,i+1) (superdiagonal of A, e[n-1] not defined)
  // f[i] = a(i,i-1) (subdiagonal of A, f[0] not defined)
  // x[i] = right-hand side b as input
  // x[i] = solution as output

  for ( int i = 1; i < n; i++ )
  {
    f[i] /= d[i-1];
    d[i] -= f[i]*e[i-1];
  }

  for ( int i = 1; i < n; i++ )
    x[i] -= f[i]*x[i-1];

  x[n-1] /= d[n-1];

  for ( int i = n-2; i >= 0; i-- )
    x[i] = (x[i]-e[i]*x[i+1])/d[i];
}

void spline(int n, double *x, double *y, double yp_left, double yp_right,
  int bcnat_left, int bcnat_right, double *y2)
{
  const double third = 1.0/3.0;
  const double sixth = 1.0/6.0;
  double *d = new double[n];
  double *e = new double[n];
  double *f = new double[n];
  if ( bcnat_left == 0 )
  {
    // use derivative yp_left at x[0]
    const double h = x[1]-x[0];
    assert(h>0.0);
    d[0] = third*h;
    e[0] = sixth*h;
    f[0] = 0.0;
    y2[0] = (y[1]-y[0])/h - yp_left;
  }
  else
  {
    // use natural spline at x[0]
    d[0] = 1.0;
    e[0] = 0.0;
    f[0] = 0.0;
    y2[0] = 0.0;
  }
  if ( bcnat_right == 0 )
  {
    // use derivative yp_right at x[n-1]
    const double h = x[n-1]-x[n-2];
    assert(h>0.0);
    d[n-1] = third*h;
    e[n-1] = 0.0;
    f[n-1] = sixth*h;
    y2[n-1] = yp_right - (y[n-1]-y[n-2])/h;
  }
  else
  {
    // use natural spline at x[n-1]
    d[n-1] = 1.0;
    e[n-1] = 0.0;
    f[n-1] = 0.0;
    y2[n-1] = 0.0;
  }

  // tridiagonal matrix
  for ( int i = 1; i < n-1; i++ )
  {
    const double hp = x[i+1]-x[i];
    const double hm = x[i]-x[i-1];
    assert(hp>0.0);
    assert(hm>0.0);
    d[i] = third * (hp+hm);
    e[i] = sixth * hp;
    f[i] = sixth * hm;
    y2[i] = (y[i+1]-y[i])/hp - (y[i]-y[i-1])/hm;
  }

  tridsolve(n,d,e,f,y2);

  delete [] d;
  delete [] e;
  delete [] f;
}

void splint (int n, double *xa, double *ya, double *y2a, double x, double *y)
{
  int k;
  double a,b,h;

  int kl = 0;
  int kh = n-1;

  while ( kh - kl > 1 )
  {
    k = ( kh + kl ) / 2;
    if ( xa[k] > x )
      kh = k;
    else
      kl = k;
  }

  h = xa[kh] - xa[kl];
  assert ( h > 0.0 );

  a = ( xa[kh] - x ) / h;
  b = ( x - xa[kl] ) / h;

  *y = a * ya[kl] + b * ya[kh] + h * h * (1.0/6.0) *
       ( (a*a*a-a) * y2a[kl] + (b*b*b-b) * y2a[kh] );

}

void splintd (int n, double *xa, double *ya, double *y2a,
              double x, double *y, double *dy)
{
  int k;
  double a,b,h;

  int kl = 0;
  int kh = n-1;

  while ( kh - kl > 1 )
  {
    k = ( kh + kl ) / 2;
    if ( xa[k] > x )
      kh = k;
    else
      kl = k;
  }

  h = xa[kh] - xa[kl];
  assert ( h > 0.0 );

  a = ( xa[kh] - x ) / h;
  b = ( x - xa[kl] ) / h;

  *y = a * ya[kl] + b * ya[kh] + h * h * (1.0/6.0) *
       ( (a*a*a-a) * y2a[kl] + (b*b*b-b) * y2a[kh] );

  *dy = ( ya[kh] - ya[kl] ) / h +
        h * ( ( (1.0/6.0) - 0.5 * a * a ) * y2a[kl] +
              ( 0.5 * b * b - (1.0/6.0) ) * y2a[kh] );
}

void splinerad( std::vector<double> & r, std::vector<double> &v, std::vector <double> & out_r, std::vector <double> &out_v, int even)
{
   int n = r.size();
   int size = 0;
   double rmin = 10.0;
   double rmax = 0.0;
   for(int i = 0; i < n; i++)
   {
     if( r[i] > 0.0) {
	     if( r[i] < rmin) rmin = r[i];
	     if( r[i] > rmax) rmax = r[i];
     }
   }
   double dstep = 0.001;
   size = ( rmax - rmin ) / dstep;

   std::vector < double> rtemp;
   std::vector < double> vtemp;
   rtemp.resize(size);
   vtemp.resize(size);
   for(int i = 0; i < size; i++)
     rtemp[i] = rmin + i * dstep;

   out_r.resize(2*size);
   for(int i = 0; i < size; i++)
     out_r[i] = - rtemp[size-1-i];
   for(int i = size; i < 2*size; i++)
     out_r[i] = rtemp[i-size];

   out_v.resize(2*size);

//   tk::spline s;
//   s.set_points(r, v);
//   for(int i = 0; i < size; i++)
//     vtemp[i] = s( rtemp[i] );	   

   DblNumVec spla(n,true,&v[0]); 
   DblNumVec splb(n), splc(n), spld(n);
   spline(n, &r[0], spla.Data(), splb.Data(), splc.Data(), spld.Data());

   seval(&vtemp[0], size, &rtemp[0], n, &r[0], spla.Data(), splb.Data(),
       splc.Data(), spld.Data());

   out_v.resize(2*size);
   if(even)
     for(int i = 0; i < size; i++)
       out_v[i] = vtemp[size-1-i];
   else
     for(int i = 0; i < size; i++)
       out_v[i] = - vtemp[size-1-i];
   for(int i = size; i < 2*size; i++)
     out_v[i] = vtemp[i-size];
}

////////////////////////////////////////////////////////////////////////////////
std::string find_start_element(std::string name, std::ifstream &upfin)
{
  // return the contents of the tag at start of element "name"
  std::string buf, token;
  std::string search_str = "<" + name;
  do
  {
    upfin >> token;
  }
  while ( !upfin.eof() && token.find(search_str) == std::string::npos );
  if ( upfin.eof() )
  {
    std::cerr << " EOF reached before start element " << name << std::endl;
    throw std::invalid_argument(name);
  }

  buf = token;
  if ( buf[buf.length()-1] == '>' )
    return buf;

  // read until ">" is found
  bool found = false;
  char ch;
  do
  {
    upfin.get(ch);
    found = ch == '>';
    buf += ch;
  }
  while ( !upfin.eof() && !found );
  if ( upfin.eof() )
  {
    std::cerr << " EOF reached before > " << name << std::endl;
    throw std::invalid_argument(name);
  }
  return buf;
}

////////////////////////////////////////////////////////////////////////////////
void find_end_element(std::string name, std::ifstream &upfin)
{
  std::string buf, token;
  std::string search_str = "</" + name + ">";
  do
  {
    upfin >> token;
    if ( token.find(search_str) != std::string::npos ) return;
  }
  while ( !upfin.eof() );
  std::cerr << " EOF reached before end element " << name << std::endl;
  throw std::invalid_argument(name);
}

////////////////////////////////////////////////////////////////////////////////
void seek_str(std::string tag, std::ifstream &upfin)
{
  // Read tokens from stdin until tag is found.
  // Throw an exception if tag not found before eof()
  bool done = false;
  std::string token;
  int count = 0;

  do
  {
    upfin >> token;
    if ( token.find(tag) != std::string::npos ) return;
  }
  while ( !upfin.eof() );

  std::cerr << " EOF reached before " << tag << std::endl;
  throw std::invalid_argument(tag);
}

////////////////////////////////////////////////////////////////////////////////
std::string get_attr(std::string buf, std::string attr)
{
  bool done = false;
//  std::string s, search_string = " " + attr + "=";
  std::string s, search_string = attr + "=";
  // find attribute name in buf
  std::string::size_type p = buf.find(search_string);
  if ( p != std::string::npos )
  {
    // process attribute
    std::string::size_type b = buf.find_first_of("\"",p);
    std::string::size_type e = buf.find_first_of("\"",b+1);
    if ( b == std::string::npos || e == std::string::npos )
    {
      std::cerr << " get_attr: attribute not found: " << attr << std::endl;
      throw std::invalid_argument(attr);
    }
    return buf.substr(b+1,e-b-1);
  }
  else
  {
    std::cerr << " get_attr: attribute not found: " << attr << std::endl;
    throw std::invalid_argument(attr);
  }
  return s;
}

////////////////////////////////////////////////////////////////////////////////
void skipln(std::ifstream & upfin )
{
  char ch;
  bool found = false;
  while ( !upfin.eof() && !found )
  {
    upfin.get(ch);
    found = ch == '\n';
  }
}


// fft interpolation method I: each dimension N/2 --> N/2 or -N/2 respectively;
//coarse to fine
/*
void IP_c2f(int* Nc,int* Nf,Complex* tilde,Complex* tildef){
        int iF,jF,kF;
        int ptrc,ptrF;
        int cjF=Nf[1]-Nc[1]/2;
        int ckF=Nf[2]-Nc[2]/2;
        double fac;
        for(int k=0;k<Nc[2];k++){
                for(int j=0;j<Nc[1];j++){
                        for(int i=0;i<Nc[0]/2+1;i++){
                                ptrc=i+j*(Nc[0]/2+1)+k*(Nc[0]/2+1)*Nc[1];
                                iF=i;
                                if(j<=Nc[1]/2){
                                        jF=j;
                                }
                                else{
                                        jF=Nf[1]-Nc[1]+j;
                                }
                                if(k<=Nc[2]/2){
                                        kF=k;
                                }
                                else{
                                        kF=Nf[2]-Nc[2]+k;
                                }
                                ptrF=iF+jF*(Nf[0]/2+1)+kF*(Nf[0]/2+1)*Nf[1];
                                tildef[ptrF]=tilde[ptrc];
                                if((iF==Nc[0]/2)&&(jF==Nc[1]/2)&&(kF==Nc[2]/2)){
                                        fac=8.0;
                                        tildef[ptrF]/=fac;
                                        jF=cjF;
                                        ptrF=iF+jF*(Nf[0]/2+1)+kF*(Nf[0]/2+1)*Nf[1];
                                        tildef[ptrF]=tilde[ptrc]/fac;
                                        kF=ckF;
                                        ptrF=iF+jF*(Nf[0]/2+1)+kF*(Nf[0]/2+1)*Nf[1];
                                        tildef[ptrF]=tilde[ptrc]/fac;
                                        jF=Nc[1]/2;
                                        ptrF=iF+jF*(Nf[0]/2+1)+kF*(Nf[0]/2+1)*Nf[1];
                                        tildef[ptrF]=tilde[ptrc]/fac;
                                }
                                else{
                                        if((iF==Nc[0]/2)&&(jF==Nc[1]/2)){
                                                fac=4.0;
                                                tildef[ptrF]/=fac;
                                            jF=cjF;
                                            ptrF=iF+jF*(Nf[0]/2+1)+kF*(Nf[0]/2+1)*Nf[1];
                                            tildef[ptrF]=tilde[ptrc]/fac;
                                        }
                                        else if((iF==Nc[0]/2)&&(kF==Nc[2]/2)){
                                                fac=4.0;
                                                tildef[ptrF]/=fac;
                                            kF=ckF;
                                            ptrF=iF+jF*(Nf[0]/2+1)+kF*(Nf[0]/2+1)*Nf[1];
                                            tildef[ptrF]=tilde[ptrc]/fac;
                                        }
                                        else if((jF==Nc[1]/2)&&(kF==Nc[2]/2)){
                                                fac=4.0;
                                                tildef[ptrF]/=fac;
                                            jF=cjF;
                                            ptrF=iF+jF*(Nf[0]/2+1)+kF*(Nf[0]/2+1)*Nf[1];
                                            tildef[ptrF]=tilde[ptrc]/fac;
                                            kF=ckF;
                                            ptrF=iF+jF*(Nf[0]/2+1)+kF*(Nf[0]/2+1)*Nf[1];
                                            tildef[ptrF]=tilde[ptrc]/fac;
                                            jF=Nc[1]/2;
                                            ptrF=iF+jF*(Nf[0]/2+1)+kF*(Nf[0]/2+1)*Nf[1];
                                            tildef[ptrF]=tilde[ptrc]/fac;
                                        }
                                        else if(iF==Nc[0]/2){
                                                fac=2.0;
                                                tildef[ptrF]/=fac;
                                        }
                                        else if(jF==Nc[1]/2){
                                                fac=2.0;
                                                tildef[ptrF]/=fac;
                                                jF=cjF;
                                            ptrF=iF+jF*(Nf[0]/2+1)+kF*(Nf[0]/2+1)*Nf[1];
                                            tildef[ptrF]=tilde[ptrc]/fac;
                                        }
                                        else if(kF==Nc[2]/2){
                                                fac=2.0;
                                                tildef[ptrF]/=fac;
                                                kF=ckF;
                                            ptrF=iF+jF*(Nf[0]/2+1)+kF*(Nf[0]/2+1)*Nf[1];
                                            tildef[ptrF]=tilde[ptrc]/fac;
                                        }
                                }
                        }
                }
        }
}

// fine to coarse

void IP_f2c(int* Nc,int* Nf,Complex* tilde,Complex* tildef){
        int i,j,k;
        int iF,jF,kF;
        int cjF=Nf[1]-Nc[1]/2;
        int ckF=Nf[2]-Nc[2]/2;
        int ptrc,f1,f2,f3,f4;
        for(k=0;k<Nc[2];k++){
                for(j=0;j<Nc[1];j++){
                        for(i=0;i<Nc[0]/2+1;i++){
                                ptrc=i+j*(Nc[0]/2+1)+k*(Nc[0]/2+1)*Nc[1];
                                iF=i;
                                if(j<=Nc[1]/2){
                                        jF=j;
                                }
                                else{
                                        jF=Nf[1]-Nc[1]+j;
                                }
                                if(k<=Nc[2]/2){
                                        kF=k;
                                }
                                else{
                                        kF=Nf[2]-Nc[2]+k;
                                }
                                f1=iF+jF*(Nf[0]/2+1)+kF*(Nf[0]/2+1)*Nf[1];
                                if((i==Nc[0]/2)&&(j==Nc[1]/2)&&(k==Nc[2]/2)){
                                        f2=iF+cjF*(Nf[0]/2+1)+kF*(Nf[0]/2+1)*Nf[1];
                                        f3=iF+jF*(Nf[0]/2+1)+ckF*(Nf[0]/2+1)*Nf[1];
                                        f4=iF+cjF*(Nf[0]/2+1)+ckF*(Nf[0]/2+1)*Nf[1];
                                        tilde[ptrc]+=0.25*Complex(tildef[f1].real()+tildef[f2].real()+tildef[f3].real()+tildef[f4].real(),0.0);
                                }
                                else if((i==Nc[0]/2)&&(j==Nc[1]/2)){
                                        f2=iF+cjF*(Nf[0]/2+1)+kF*(Nf[0]/2+1)*Nf[1];
                                        kF=(Nf[2]-kF)%Nf[2];
                                        f3=iF+jF*(Nf[0]/2+1)+kF*(Nf[0]/2+1)*Nf[1];
                                        f4=iF+cjF*(Nf[0]/2+1)+kF*(Nf[0]/2+1)*Nf[1];
                                        tilde[ptrc]+=0.25*(tildef[f1]+tildef[f2]+std::conj(tildef[f3])+std::conj(tildef[f4]));
                                }
                                else if((i==Nc[0]/2)&&(k==Nc[2]/2)){
                                        f2=iF+jF*(Nf[0]/2+1)+ckF*(Nf[0]/2+1)*Nf[1];
                                        jF=(Nf[1]-jF)%Nf[1];
                                        f3=iF+jF*(Nf[0]/2+1)+kF*(Nf[0]/2+1)*Nf[1];
                                        f4=iF+jF*(Nf[0]/2+1)+ckF*(Nf[0]/2+1)*Nf[1];
                                        tilde[ptrc]+=0.25*(tildef[f1]+tildef[f2]+std::conj(tildef[f3])+std::conj(tildef[f4]));
                                }
                                else if((j==Nc[1]/2)&&(k==Nc[2]/2)){
                                        f2=iF+cjF*(Nf[0]/2+1)+kF*(Nf[0]/2+1)*Nf[1];
                                        f3=iF+jF*(Nf[0]/2+1)+ckF*(Nf[0]/2+1)*Nf[1];
                                        f4=iF+cjF*(Nf[0]/2+1)+ckF*(Nf[0]/2+1)*Nf[1];
                                        tilde[ptrc]+=0.25*(tildef[f1]+tildef[f2]+tildef[f3]+tildef[f4]);
                                }
                                else if(i==Nc[0]/2){
                                        jF=(Nf[1]-jF)%Nf[1];
                                        kF=(Nf[2]-kF)%Nf[2];
                                        f2=iF+jF*(Nf[0]/2+1)+kF*(Nf[0]/2+1)*Nf[1];
                                        tilde[ptrc]+=0.5*(tildef[f1]+std::conj(tildef[f2]));
                                }
                                else if(j==Nc[1]/2){
                                        f2=iF+cjF*(Nf[0]/2+1)+kF*(Nf[0]/2+1)*Nf[1];
                                        tilde[ptrc]+=0.5*(tildef[f1]+tildef[f2]);
                                }
                                else if(k==Nc[2]/2){
                                        f2=iF+jF*(Nf[0]/2+1)+ckF*(Nf[0]/2+1)*Nf[1];
                                        tilde[ptrc]+=0.5*(tildef[f1]+tildef[f2]);
                                }
                                else{
                                        tilde[ptrc]+=tildef[f1];
                                }
                        }
                }
        }
}
*/


// fft interpolation method II: all dimension N/2 --> N/2 or -N/2 at the same time;
//coarse to fine, equals to take the real part of Complex(coarse to fine)

void IP_c2f(int* Nc,int* Nf,Complex* tilde,Complex* tildef){
        int iF,jF,kF;
        int ptrc,ptrF;
        int cjF=Nf[1]-Nc[1]/2;
        int ckF=Nf[2]-Nc[2]/2;
        double fac=2.0;
        for(int k=0;k<Nc[2];k++){
                for(int j=0;j<Nc[1];j++){
                        for(int i=0;i<Nc[0]/2+1;i++){
                                ptrc=i+j*(Nc[0]/2+1)+k*(Nc[0]/2+1)*Nc[1];
                                iF=i;
                                if(j<=Nc[1]/2){
                                        jF=j;
                                }
                                else{
                                        jF=Nf[1]-Nc[1]+j;
                                }
                                if(k<=Nc[2]/2){
                                        kF=k;
                                }
                                else{
                                        kF=Nf[2]-Nc[2]+k;
                                }
                                ptrF=iF+jF*(Nf[0]/2+1)+kF*(Nf[0]/2+1)*Nf[1];
                                tildef[ptrF]=tilde[ptrc];
                                if((iF==Nc[0]/2)&&(jF==Nc[1]/2)&&(kF==Nc[2]/2)){
                                        tildef[ptrF]/=fac;
                                }
                                else{
                                        if((iF==Nc[0]/2)&&(jF==Nc[1]/2)){
                                                tildef[ptrF]/=fac;
                                        }
                                        else if((iF==Nc[0]/2)&&(kF==Nc[2]/2)){
                                                tildef[ptrF]/=fac;
                                        }
                                        else if((jF==Nc[1]/2)&&(kF==Nc[2]/2)){
                                                tildef[ptrF]/=fac;
                                            jF=cjF;
                                            kF=ckF;
                                            ptrF=iF+jF*(Nf[0]/2+1)+kF*(Nf[0]/2+1)*Nf[1];
                                            tildef[ptrF]=tilde[ptrc]/fac;
                                        }
                                        else if(iF==Nc[0]/2){
                                                tildef[ptrF]/=fac;
                                        }
                                        else if(jF==Nc[1]/2){
                                                tildef[ptrF]/=fac;
                                                jF=cjF;
                                            ptrF=iF+jF*(Nf[0]/2+1)+kF*(Nf[0]/2+1)*Nf[1];
                                            tildef[ptrF]=tilde[ptrc]/fac;
                                        }
                                        else if(kF==Nc[2]/2){
                                                tildef[ptrF]/=fac;
                                                kF=ckF;
                                            ptrF=iF+jF*(Nf[0]/2+1)+kF*(Nf[0]/2+1)*Nf[1];
                                            tildef[ptrF]=tilde[ptrc]/fac;
                                        }
                                }
                        }
                }
        }
}

//fine to coarse, equals to take the real part of Complex(fine to coarse)

void IP_f2c(int* Nc,int* Nf,Complex* tilde,Complex* tildef){
        int i,j,k;
        int iF,jF,kF;
        int cjF=Nf[1]-Nc[1]/2;
        int ckF=Nf[2]-Nc[2]/2;
        int ptrc,f1,f2,f3,f4;
        for(k=0;k<Nc[2];k++){
                for(j=0;j<Nc[1];j++){
                        for(i=0;i<Nc[0]/2+1;i++){
                                ptrc=i+j*(Nc[0]/2+1)+k*(Nc[0]/2+1)*Nc[1];
                                iF=i;
                                if(j<=Nc[1]/2){
                                        jF=j;
                                }
                                else{
                                        jF=Nf[1]-Nc[1]+j;
                                }
                                if(k<=Nc[2]/2){
                                        kF=k;
                                }
                                else{
                                        kF=Nf[2]-Nc[2]+k;
                                }
                                f1=iF+jF*(Nf[0]/2+1)+kF*(Nf[0]/2+1)*Nf[1];
                                if((i==Nc[0]/2)&&(j==Nc[1]/2)&&(k==Nc[2]/2)){
                                        tilde[ptrc]+=Complex(tildef[f1].real(),0.0);
                                }
                                else if((i==Nc[0]/2)&&(j==Nc[1]/2)){
                                        kF=(Nf[2]-kF)%Nf[2];
                                        f2=iF+jF*(Nf[0]/2+1)+kF*(Nf[0]/2+1)*Nf[1];
                                        tilde[ptrc]+=0.5*(tildef[f1]+std::conj(tildef[f2]));
                                }
                                else if((i==Nc[0]/2)&&(k==Nc[2]/2)){
                                        jF=(Nf[1]-jF)%Nf[1];
                                        f2=iF+jF*(Nf[0]/2+1)+kF*(Nf[0]/2+1)*Nf[1];
                                        tilde[ptrc]+=0.5*(tildef[f1]+std::conj(tildef[f2]));
                                }
                                else if((j==Nc[1]/2)&&(k==Nc[2]/2)){
                                        f2=iF+cjF*(Nf[0]/2+1)+ckF*(Nf[0]/2+1)*Nf[1];
                                        tilde[ptrc]+=0.5*(tildef[f1]+tildef[f2]);
                                }
                                else if(i==Nc[0]/2){
                                        jF=(Nf[1]-jF)%Nf[1];
                                        kF=(Nf[2]-kF)%Nf[2];
                                        f2=iF+jF*(Nf[0]/2+1)+kF*(Nf[0]/2+1)*Nf[1];
                                        tilde[ptrc]+=0.5*(tildef[f1]+std::conj(tildef[f2]));
                                }
                                else if(j==Nc[1]/2){
                                        f2=iF+cjF*(Nf[0]/2+1)+kF*(Nf[0]/2+1)*Nf[1];
                                        tilde[ptrc]+=0.5*(tildef[f1]+tildef[f2]);
                                }
                                else if(k==Nc[2]/2){
                                        f2=iF+jF*(Nf[0]/2+1)+ckF*(Nf[0]/2+1)*Nf[1];
                                        tilde[ptrc]+=0.5*(tildef[f1]+tildef[f2]);
                                }
                                else{
                                        tilde[ptrc]+=tildef[f1];
                                }
                        }
                }
        }
}

Int sph_ind(Int l, Real j, Int m, Int spin){

  Int sph_ind;

  if( std::abs(j-l-0.5) < 1e-8 ){
    if( spin == 0 )
      sph_ind = m;
    else if( spin == 1 )
      sph_ind = m + 1;
  }
  else if( std::abs(j-l+0.5) < 1e-8 ){
    if( m < -l+1 )
      sph_ind = 0;
    else{
      if( spin == 0 )
        sph_ind = m - 1;
      else if( spin == 1 )
        sph_ind = m;
    }
  }

  if( sph_ind<-l || sph_ind>l )
    sph_ind = 0;
  
  return sph_ind;
}

Real spinor(Int l, Real j, Int m, Int spin){

  Real denom = 1.0 / Real( ( 2 * l + 1 ) );
  Real spinor;

  if( std::abs(j-l-0.5) < 1e-8 ){
    if( spin == 0 )
      spinor = std::sqrt( (l + m + 1) * denom );
    else if( spin == 1 )
      spinor = std::sqrt( (l - m) * denom );
  }
  else if( std::abs(j-l+0.5) < 1e-8 ){
    if( m < -l+1 )
      spinor = 0.0;
    else{
      if( spin == 0 )
        spinor = std::sqrt( (l-m+1) * denom );
      else if( spin == 1 )
        spinor = - std::sqrt( (l+m) * denom );
    }
  } 
  return spinor;
}

Real Simpson(Int n, DblNumVec& func, Real rab){

  Real intf = 0.0;

  for( Int i = 0; i < n-2; i=i+2 ){
    intf += ( func(i) + 4 * func(i+1) + func(i+2) ) / 3.0 * rab;
  }

  return intf;
}

void pbexsr( Real RHO, Real GRHO, Real OMEGA, Real &sx, Real &V1X, Real &V2X ){
 
  Real SMALL = 1e-20;
  Real SMAL2 = 1e-08;
  Real US = 0.161620459673995492;
  Real AX = -0.738558766382022406;
  Real UM = 0.2195149727645171;
  Real UK = 0.8040;
  Real UL = UM / UK;
  Real f1 = -1.10783814957303361;
  Real alpha = 2.0 / 3.0;

  Real RS = std::pow(RHO , (1.0 / 3.0) );
  Real VX = (4.0 / 3.0) * f1 * alpha * RS;

  Real AA = GRHO;
  Real RR = 1.0 / (RHO * RS);
  Real EX = AX / RR;
  Real S2 = AA * RR * RR * US * US;

  Real S = std::sqrt(S2);
  if( S > 8.3 )
    S = 8.572844 - 18.796223 / S2;

  Real FX, D1X, D2X;
  wpbe_analy_erfc_approx_grad(RHO,S,OMEGA,FX,D1X,D2X);

  sx = EX * FX;       
  Real DSDN = -4.0 / 3.0 * S / RHO;
  V1X = VX * FX + (DSDN * D2X + D1X) * EX;
  Real DSDG = US * RR;
  V2X = EX * 1.0 / std::sqrt(AA) * DSDG * D2X;

  return;
} 

void wpbe_analy_erfc_approx_grad(Real rho, Real s, Real omega, Real &Fx_wpbe, Real &d1rfx, Real &d1sfx){

  Real    Zero = 0;
  Real    One = 1;
  Real    Two = 2;
  Real    Three = 3;
  Real    Four = 4;
  Real    Five = 5;
  Real    Six = 6;
  Real    Seven = 7;
  Real    Eight = 8;
  Real    Nine = 9;
  Real    Ten = 10;
  Real    Fifteen = 15;
  Real    Sixteen = 16;
       
  Real    r36 = 36;
  Real    r64 = 64;
  Real    r81 = 81;
  Real    r256 = 256;
  Real    r384 = 384;
  Real    r864 = 864;
  Real    r1944 = 1944;
  Real    r4374 = 4374;
       
  Real    r27 = 27;
  Real    r48 = 48;
  Real    r120 = 120;
  Real    r128 = 128;
  Real    r144 = 144;
  Real    r288 = 288;
  Real    r324 = 324;
  Real    r512 = 512;
  Real    r729 = 729;
        
  Real    r20 = 20;
  Real    r32 = 32;
  Real    r243 = 243;
  Real    r2187 = 2187;
  Real    r6561 = 6561;
  Real    r40 = 40;
      
  Real    r12 = 12;
  Real    r25 = 25;
  Real    r30 = 30;
  Real    r54 = 54;
  Real    r75 = 75;
  Real    r105 = 105;
  Real    r135 = 135;
  Real    r1215 = 1215;
  Real    r15309 = 15309; 

  Real    f12 = 0.5;
  Real    f13 = One/Three;
  Real    f14 = 0.25;
  Real    f18 = 0.125;

  Real    f23 = Two * f13;
  Real    f43 = Two * f23;

  Real    f32 = 1.5;
  Real    f72 = 3.5;
  Real    f34 = 0.75;
  Real    f94 = 2.25;
  Real    f98 = 1.125;
  Real    f1516 = Fifteen / Sixteen;

  Real    pi = acos(-One);
  Real    pi2 = pi * pi;
  Real    pi_23 = std::pow(pi2, f13);
  Real    srpi  = sqrt(pi);

  Real    Three_13 = std::pow(Three, f13);

  Real    ea1 = -1.128223946706117;
  Real    ea2 = 1.452736265762971;
  Real    ea3 = -1.243162299390327;
  Real    ea4 = 0.971824836115601;
  Real    ea5 = -0.568861079687373;
  Real    ea6 = 0.246880514820192;
  Real    ea7 = -0.065032363850763;
  Real    ea8 = 0.008401793031216;

  Real    eb1 = 1.455915450052607;

  Real    A      =  1.0161144;
  Real    B      = -3.7170836e-1;
  Real    C      = -7.7215461e-2;
  Real    D      =  5.7786348e-1;
  Real    E      = -5.1955731e-2;
  Real    X      = - Eight / Nine;

  Real    Ha1    = 9.79681e-3;
  Real    Ha2    = 4.10834e-2;
  Real    Ha3    = 1.87440e-1;
  Real    Ha4    = 1.20824e-3;
  Real    Ha5    = 3.47188e-2;

  Real    Fc1    = 6.4753871e0;
  Real    Fc2    = 4.7965830e-1;

  Real    EGa1   = -2.628417880e-2;
  Real    EGa2   = -7.117647788e-2;
  Real    EGa3   =  8.534541323e-2;

  Real    expei1 = 4.03640;
  Real    expei2 = 1.15198;
  Real    expei3 = 5.03627;
  Real    expei4 = 4.19160;
      
  Real    EGscut     = 8.0e-2;
  Real    wcutoff    = 1.4e1;
  Real    expfcutoff = 7.0e2;

  Real    xkf        = std::pow(Three*pi2*rho, f13);
  Real    xkfrho     = xkf * rho;

  Real    A2 = A*A;
  Real    A3 = A2*A;
  Real    A4 = A3*A;
  Real    A12 = std::sqrt(A);
  Real    A32 = A12*A;
  Real    A52 = A32*A;
  Real    A72 = A52*A;

  Real    w     = omega / xkf;
  Real    w2    = w * w;
  Real    w3    = w2 * w;
  Real    w4    = w2 * w2;
  Real    w5    = w3 * w2;
  Real    w6    = w5 * w;
  Real    w7    = w6 * w;
  Real    w8    = w7 * w;

  Real    d1rw  = -(One/(Three*rho))*w;

  X      = - Eight/Nine;

  Real    s2     = s*s;
  Real    s3     = s2*s;
  Real    s4     = s2*s2;
  Real    s5     = s4*s;
  Real    s6     = s5*s;

  Real    Hnum    = Ha1*s2 + Ha2*s4;
  Real    Hden    = One + Ha3*s4 + Ha4*s5 + Ha5*s6;

  Real    H       = Hnum/Hden;

  Real    d1sHnum = Two*Ha1*s + Four*Ha2*s3;
  Real    d1sHden = Four*Ha3*s3 + Five*Ha4*s4 + Six*Ha5*s5;

  Real    d1sH    = (Hden*d1sHnum - Hnum*d1sHden) / (Hden*Hden);

  Real    F      = Fc1*H + Fc2;
  Real    d1sF   = Fc1*d1sH;

  if( w > wcutoff ){
    eb1 = 2.0;
  }

  Real    Hsbw = s2*H + eb1*w2;
  Real    Hsbw2 = Hsbw*Hsbw;
  Real    Hsbw3 = Hsbw2*Hsbw;
  Real    Hsbw4 = Hsbw3*Hsbw;
  Real    Hsbw12 = std::sqrt(Hsbw);
  Real    Hsbw32 = Hsbw12*Hsbw;
  Real    Hsbw52 = Hsbw32*Hsbw;
  Real    Hsbw72 = Hsbw52*Hsbw;

  Real    d1sHsbw  = d1sH*s2 + Two*s*H;
  Real    d1rHsbw  = Two*eb1*d1rw*w;

  Real    DHsbw = D + s2*H + eb1*w2;
  Real    DHsbw2 = DHsbw*DHsbw;
  Real    DHsbw3 = DHsbw2*DHsbw;
  Real    DHsbw4 = DHsbw3*DHsbw;
  Real    DHsbw5 = DHsbw4*DHsbw;
  Real    DHsbw12 = std::sqrt(DHsbw);
  Real    DHsbw32 = DHsbw12*DHsbw;
  Real    DHsbw52 = DHsbw32*DHsbw;
  Real    DHsbw72 = DHsbw52*DHsbw;
  Real   DHsbw92 = DHsbw72*DHsbw;

  Real    HsbwA94   = f94 * Hsbw / A;
  Real    HsbwA942  = HsbwA94*HsbwA94;
  Real    HsbwA943  = HsbwA942*HsbwA94;
  Real    HsbwA945  = HsbwA943*HsbwA942;
  Real    HsbwA9412 = std::sqrt(HsbwA94);

  Real    DHs    = D + s2*H;
  Real    DHs2   = DHs*DHs;
  Real    DHs3   = DHs2*DHs;
  Real    DHs4   = DHs3*DHs;
  Real    DHs72  = DHs3*std::sqrt(DHs);
  Real    DHs92  = DHs72*DHs;

  Real    d1sDHs = Two*s*H + s2*d1sH;

  Real    DHsw   = DHs + w2;
  Real    DHsw2  = DHsw*DHsw;
  Real    DHsw52 = std::sqrt(DHsw)*DHsw2;
  Real    DHsw72 = DHsw52*DHsw;

  Real    d1rDHsw = Two*d1rw*w;
 
  Real EG, d1sEG;
  if( s > EGscut ){
    Real G_a    = srpi * (Fifteen*E + Six*C*(One+F*s2)*DHs + 
                    Four*B*(DHs2) + Eight*A*(DHs3))    
                    * (One / (Sixteen * DHs72))
                    - f34*pi*std::sqrt(A) * std::exp(f94*H*s2/A) * 
                    (One - std::erf(f32*s*std::sqrt(H/A)));

    Real d1sG_a = (One/r32)*srpi *
                    ((r36*(Two*H + d1sH*s) / (A12*std::sqrt(H/A)))
                    + (One/DHs92) *
                    (-Eight*A*d1sDHs*DHs3 - r105*d1sDHs*E
                    -r30*C*d1sDHs*DHs*(One+s2*F)
                    +r12*DHs2*(-B*d1sDHs + C*s*(d1sF*s + Two*F)))
                    - ((r54*std::exp(f94*H*s2/A)*srpi*s*(Two*H+d1sH*s)*
                    std::erfc(f32*std::sqrt(H/A)*s))
                    / A12));

    Real G_b    = (f1516 * srpi * s2) / DHs72;

    Real d1sG_b = (Fifteen*srpi*s*(Four*DHs - Seven*d1sDHs*s))
                    / (r32*DHs92);

    EG     = - (f34*pi + G_a) / G_b;

    d1sEG  = (-Four*d1sG_a*G_b + d1sG_b*(Four*G_a + Three*pi))
                    / (Four*G_b*G_b);
  }
  else{
    EG    = EGa1 + EGa2*s2 + EGa3*s4;
    d1sEG = Two*EGa2*s + Four*EGa3*s3;
  }

  Real    term2 = (DHs2*B + DHs*C + Two*E + DHs*s2*C*F + Two*s2*EG) /
                    (Two*DHs3);

  Real    d1sterm2 = (-Six*d1sDHs*(EG*s2 + E)
                       + DHs2 * (-d1sDHs*B + s*C*(d1sF*s + Two*F))
                       + Two*DHs * (Two*EG*s - d1sDHs*C
                       + s2 * (d1sEG - d1sDHs*C*F)))
                       / (Two*DHs4);

  Real    term3 = - w  * (Four*DHsw2*B + Six*DHsw*C + Fifteen*E
                      + Six*DHsw*s2*C*F + Fifteen*s2*EG) /
                     (Eight*DHs*DHsw52);

  Real    d1sterm3 = w * (Two*d1sDHs*DHsw * (Four*DHsw2*B
                         + Six*DHsw*C + Fifteen*E
                         + Three*s2*(Five*EG + Two*DHsw*C*F))
                         + DHs * (r75*d1sDHs*(EG*s2 + E)
                         + Four*DHsw2*(d1sDHs*B
                         - Three*s*C*(d1sF*s + Two*F))
                         - Six*DHsw*(-Three*d1sDHs*C
                         + s*(Ten*EG + Five*d1sEG*s
                         - Three*d1sDHs*s*C*F))))
                         / (Sixteen*DHs2*DHsw72);

  Real    d1rterm3 = (-Two*d1rw*DHsw * (Four*DHsw2*B            
                         + Six*DHsw*C + Fifteen*E
                         + Three*s2*(Five*EG + Two*DHsw*C*F))
                         + w * d1rDHsw * (r75*(EG*s2 + E)
                         + Two*DHsw*(Two*DHsw*B + Nine*C
                         + Nine*s2*C*F)))
                         / (Sixteen*DHs*DHsw72);

  Real    term4 = - w3 * (DHsw*C + Five*E + DHsw*s2*C*F + Five*s2*EG) /
                   (Two*DHs2*DHsw52);

  Real    d1sterm4 = (w3 * (Four*d1sDHs*DHsw * (DHsw*C + Five*E
                        + s2 * (Five*EG + DHsw*C*F))
                        + DHs * (r25*d1sDHs*(EG*s2 + E)
                        - Two*DHsw2*s*C*(d1sF*s + Two*F)
                        + DHsw * (Three*d1sDHs*C + s*(-r20*EG
                        - Ten*d1sEG*s
                        + Three*d1sDHs*s*C*F)))))
                         / (Four*DHs3*DHsw72);

  Real    d1rterm4 = (w2 * (-Six*d1rw*DHsw * (DHsw*C + Five*E
                        + s2 * (Five*EG + DHsw*C*F))
                        + w * d1rDHsw * (r25*(EG*s2 + E) +
                        Three*DHsw*C*(One + s2*F))))
                         / (Four*DHs2*DHsw72);

  Real    term5 = - w5 * (E + s2*EG) /
                     (DHs3*DHsw52);

  Real    d1sterm5 = (w5 * (Six*d1sDHs*DHsw*(EG*s2 + E)          
                        + DHs * (-Two*DHsw*s * (Two*EG + d1sEG*s)
                        + Five*d1sDHs * (EG*s2 + E))))
                        / (Two*DHs4*DHsw72);

  Real    d1rterm5 = (w4 * Five*(EG*s2 + E) * (-Two*d1rw*DHsw
                          + d1rDHsw * w))
                          / (Two*DHs3*DHsw72);


  Real t10, t10d1, d1st10, d1rt10;
  if( s > 0.0 || w > 0.0 ){
    t10    = (f12)*A*log(Hsbw / DHsbw);
    t10d1  = f12*A*(One/Hsbw - One/DHsbw);
    d1st10 = d1sHsbw*t10d1;
    d1rt10 = d1rHsbw*t10d1;
  }


  Real piexperf, expei;
  if( HsbwA94 < expfcutoff ){
    piexperf = pi*std::exp(HsbwA94)*std::erfc(HsbwA9412);
    expei    = std::exp(HsbwA94)*(-EXPINT(1,HsbwA94));
  }
  else{
    piexperf = pi*(One/(srpi*HsbwA9412)
                         - One/(Two*std::sqrt(pi*HsbwA943))
                         + Three/(Four*std::sqrt(pi*HsbwA945)));

    expei  = - (One/HsbwA94) *                       
                   (HsbwA942 + expei1*HsbwA94 + expei2) /
                   (HsbwA942 + expei3*HsbwA94 + expei4);
  }

  Real    piexperfd1  = - (Three*srpi*std::sqrt(Hsbw/A))/(Two*Hsbw)
                         + (Nine*piexperf)/(Four*A);
  Real    d1spiexperf = d1sHsbw*piexperfd1;
  Real    d1rpiexperf = d1rHsbw*piexperfd1;

  Real    expeid1  = f14*(Four/Hsbw + (Nine*expei)/A);
  Real    d1sexpei = d1sHsbw*expeid1;
  Real    d1rexpei = d1rHsbw*expeid1;

  if( w ==  Zero ){ 
    Real      t1 = -f12*A*expei;
    Real      d1st1 = -f12*A*d1sexpei;
    Real      d1rt1 = -f12*A*d1rexpei;
    if( s > 0.0 ){
      Real    term1    = t1 + t10;
      Real    d1sterm1 = d1st1 + d1st10;
      Real    d1rterm1 = d1rt1 + d1rt10;
      Fx_wpbe = X * (term1 + term2);
      d1sfx = X * (d1sterm1 + d1sterm2);
      d1rfx = X * d1rterm1;
    }
    else{
      Fx_wpbe = 1.0;
      d1sfx   = 0.0;
      d1rfx   = 0.0;
    }
  }          
  else if( w > wcutoff ){
      Real    term1 = -f12*A*(expei+log(DHsbw)-log(Hsbw));
      Real    term1d1  = - A/(Two*DHsbw) - f98*expei;
      Real    d1sterm1 = d1sHsbw*term1d1;
      Real    d1rterm1 = d1rHsbw*term1d1;
      Fx_wpbe = X * (term1 + term2 + term3 + term4 + term5);
      d1sfx = X * (d1sterm1 + d1sterm2 + d1sterm3  
                  + d1sterm4 + d1sterm5);
      d1rfx = X * (d1rterm1 + d1rterm3 + d1rterm4 + d1rterm5);
  }
  else{
      Real    np1    = -f32*ea1*A12*w + r27*ea3*w3/(Eight*A12)
                    - r243*ea5*w5/(r32*A32) + r2187*ea7*w7/(r128*A52);
      Real    d1rnp1 = - f32*ea1*d1rw*A12 + (r81*ea3*d1rw*w2)/(Eight*A12) 
                       - (r1215*ea5*d1rw*w4)/(r32*A32)                   
                       + (r15309*ea7*d1rw*w6)/(r128*A52);
      Real     np2 = -A + f94*ea2*w2 - r81*ea4*w4/(Sixteen*A) 
                       + r729*ea6*w6/(r64*A2) - r6561*ea8*w8/(r256*A3);

      Real    d1rnp2 =   f12*(Nine*ea2*d1rw*w)         
                     - (r81*ea4*d1rw*w3)/(Four*A)    
                     + (r2187*ea6*d1rw*w5)/(r32*A2)  
                     - (r6561*ea8*d1rw*w7)/(r32*A3);

      Real    t1    = f12*(np1*piexperf + np2*expei);
      Real    d1st1 = f12*(d1spiexperf*np1 + d1sexpei*np2);
      Real    d1rt1 = f12*(d1rnp2*expei + d1rpiexperf*np1 + 
                       d1rexpei*np2 + d1rnp1*piexperf);

      Real    f2    = (f12)*ea1*srpi*A / DHsbw12;
      Real    f2d1  = - ea1*srpi*A / (Four*DHsbw32);
      Real    d1sf2 = d1sHsbw*f2d1;
      Real    d1rf2 = d1rHsbw*f2d1;

      Real    f3    = (f12)*ea2*A / DHsbw;
      Real    f3d1  = - ea2*A / (Two*DHsbw2);
      Real    d1sf3 = d1sHsbw*f3d1;
      Real    d1rf3 = d1rHsbw*f3d1;

      Real    f4    =  ea3*srpi*(-f98 / Hsbw12
                   + f14*A / DHsbw32);
      Real    f4d1  = ea3*srpi*((Nine/(Sixteen*Hsbw32))-
                          (Three*A/(Eight*DHsbw52)));
      Real    d1sf4 = d1sHsbw*f4d1;
      Real    d1rf4 = d1rHsbw*f4d1;

      Real    f5    = ea4*(One/r128) * (-r144*(One/Hsbw)   
                   + r64*(One/DHsbw2)*A);
      Real    f5d1  = ea4*((f98/Hsbw2)-(A/DHsbw3));
      Real    d1sf5 = d1sHsbw*f5d1;
      Real    d1rf5 = d1rHsbw*f5d1;

      Real    f6    = ea5*(Three*srpi*(Three*DHsbw52*(Nine*Hsbw-Two*A) 
                   + Four*Hsbw32*A2))                              
                   / (r32*DHsbw52*Hsbw32*A);
      Real    f6d1  = ea5*srpi*((r27/(r32*Hsbw52))-        
                      (r81/(r64*Hsbw32*A))-            
                      ((Fifteen*A)/(Sixteen*DHsbw72)));
      Real    d1sf6 = d1sHsbw*f6d1;
      Real    d1rf6 = d1rHsbw*f6d1;

      Real    f7    = ea6*(((r32*A)/DHsbw3                 
                   + (-r36 + (r81*s2*H)/A)/Hsbw2)) / r32;
      Real    d1sf7 = ea6*(Three*(r27*d1sH*DHsbw4*Hsbw*s2 +           
                  Eight*d1sHsbw*A*(Three*DHsbw4 - Four*Hsbw3*A) + 
                  r54*DHsbw4*s*(Hsbw - d1sHsbw*s)*H))/            
                  (r32*DHsbw4*Hsbw3*A);
      Real    d1rf7 = ea6*d1rHsbw*((f94/Hsbw3)-((Three*A)/DHsbw4)     
                             -((r81*s2*H)/(Sixteen*Hsbw3*A)));

      Real    f8    = ea7*(-Three*srpi*(-r40*Hsbw52*A3                
                   +Nine*DHsbw72*(r27*Hsbw2-Six*Hsbw*A+Four*A2))) 
                   / (r128 * DHsbw72*Hsbw52*A2);
      Real    f8d1  = ea7*srpi*((r135/(r64*Hsbw72)) + (r729/(r256*Hsbw32*A2))  
                           -(r243/(r128*Hsbw52*A))                         
                           -((r105*A)/(r32*DHsbw92)));
      Real    d1sf8 = d1sHsbw*f8d1;
      Real    d1rf8 = d1rHsbw*f8d1;

      Real    f9    = (r324*ea6*eb1*DHsbw4*Hsbw*A                      
                  + ea8*(r384*Hsbw3*A3 + DHsbw4*(-r729*Hsbw2       
                  + r324*Hsbw*A - r288*A2))) / (r128*DHsbw4*Hsbw3*A2);
      Real    f9d1  = -((r81*ea6*eb1)/(Sixteen*Hsbw3*A))               
                  + ea8*((r27/(Four*Hsbw4))+(r729/(r128*Hsbw2*A2)) 
                        -(r81/(Sixteen*Hsbw3*A))                  
                        -((r12*A/DHsbw5)));
      Real    d1sf9 = d1sHsbw*f9d1;
      Real    d1rf9 = d1rHsbw*f9d1;

      Real    t2t9    = f2*w  + f3*w2 + f4*w3 + f5*w4 + f6*w5          
                          + f7*w6 + f8*w7 + f9*w8;
      Real    d1st2t9 = d1sf2*w + d1sf3*w2 + d1sf4*w3 + d1sf5*w4       
                          + d1sf6*w5 + d1sf7*w6 + d1sf8*w7       
                          + d1sf9*w8;
      Real    d1rt2t9 = d1rw*f2 + d1rf2*w + Two*d1rw*f3*w   
                  + d1rf3*w2 + Three*d1rw*f4*w2       
                  + d1rf4*w3 + Four*d1rw*f5*w3        
                  + d1rf5*w4 + Five*d1rw*f6*w4        
                  + d1rf6*w5 + Six*d1rw*f7*w5         
                  + d1rf7*w6 + Seven*d1rw*f8*w6       
                  + d1rf8*w7 + Eight*d1rw*f9*w7 + d1rf9*w8;
      Real    term1 = t1 + t2t9 + t10;

      Real    d1sterm1 = d1st1 + d1st2t9 + d1st10;
      Real    d1rterm1 = d1rt1 + d1rt2t9 + d1rt10;


      Fx_wpbe = X * (term1 + term2 + term3 + term4 + term5);

      d1sfx = X * (d1sterm1 + d1sterm2 + d1sterm3    
                  + d1sterm4 + d1sterm5);

      d1rfx = X * (d1rterm1 + d1rterm3 + d1rterm4 + d1rterm5);
  }

  return;
}

Real EXPINT( Int n, Real x ){

  Int maxit = 200;
  Real eps = 1e-12;
  Real big = 1.797693127679088e296;
  Real euler = 0.577215664901532860606512;
  Real expint;
  
  if (!((n >= 0)&&(x >= 0.0)&&((x > 0.0)||(n > 1))))
    ErrorHandling("error for inputed n and x...");
  
  if( n == 0 ){
    expint = exp(-x) / x;
    return expint;
  }

  Int nm1 = n - 1;
  if( x == 0.0 ){
    expint = 1.0 / nm1;
  }
  else if( x > 1.0 ){
    Real b = x + n;
    Real c = big;
    Real d = 1.0 / b;
    Real h = d;
    for( Int i = 1; i <= maxit; i++ ){
      Real a = -i * ( nm1 + i );
      b = b + 2.0;
      d = 1.0 / (a * d + b);
      c = b + a / c;
      Real del = c * d;
      h = h * del;
      if( abs(del - 1.0) <= eps )
        break;      
    }
    expint = h * std::exp(-x);
  }
  else{
    if( nm1 != 0 )
      expint = 1.0 / nm1;
    else
      expint = - log(x) - euler;
    
    Real fact = 1.0;
    for( Int i = 1; i <= maxit; i++ ){
      fact = - fact * x / i;
      Real del;
      if( i != nm1 )
        del = - fact / (i - nm1);
      else{
        Real iarsum = 0.0;
        for( Int k = 1; k <= nm1; k++ )
          iarsum = iarsum + 1.0 / k;

        del = fact * ( - log(x) - euler + iarsum );
      }

      expint = expint + del;
      if( abs(del) < abs(expint) * eps )
        break;
    }  
  }

  return expint;
}

}  // namespace dgdft
