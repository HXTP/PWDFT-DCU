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
#include  "scf_dg.hpp"
#include  "blas.hpp"
#include  "lapack.hpp"
#include  "utility.hpp"
#include  "environment.hpp"
#ifdef ELSI
#include  "elsi.h"
#endif
#include  "scfdg_upper_end_of_spectrum.hpp"

namespace  dgdft{

using namespace dgdft::DensityComponent;
using namespace dgdft::esdf;
using namespace dgdft::scalapack;

// ---------------------------------------------------------
// ------------ Routines for Chebyshev Filtering -----------
// ---------------------------------------------------------

// Dot product for conforming distributed vectors
// Requires that a distributed vector (and not a distributed matrix) has been sent
double 
SCFDG::scfdg_distvec_dot(DistVec<Index3, DblNumMat, ElemPrtn>  &dist_vec_a,
    DistVec<Index3, DblNumMat, ElemPrtn>  &dist_vec_b)
{
  double temp_result = 0.0, final_result = 0.0;
  DblNumMat& vec_a= (dist_vec_a.LocalMap().begin())->second;
  DblNumMat& vec_b= (dist_vec_b.LocalMap().begin())->second;

  // Conformity check
  if( (dist_vec_a.LocalMap().size() != 1) ||
      (dist_vec_b.LocalMap().size() != 1) ||
      (vec_a.m() != vec_b.m()) ||
      (vec_a.n() != 1) ||
      (vec_b.n() != 1) )
  {
    statusOFS << std::endl << " Non-conforming vectors in dot product !! Aborting ... " << std::endl;
    exit(1);
  }

  // Local result
  temp_result = blas::Dot(vec_a.m(),vec_a.Data(), 1, vec_b.Data(), 1);
  // Global reduce  across colComm since the partion is by DG elements
  mpi::Allreduce( &temp_result, &final_result, 1, MPI_SUM, domain_.colComm );

  return final_result;
} // End of routine scfdg_distvec_dot

// L2 norm : Requires that a distributed vector (and not a distributed matrix) has been sent
double 
SCFDG::scfdg_distvec_nrm2(DistVec<Index3, DblNumMat, ElemPrtn>  &dist_vec_a)
{
  double temp_result = 0.0, final_result = 0.0;
  DblNumMat& vec_a= (dist_vec_a.LocalMap().begin())->second;
  double *ptr = vec_a.Data();

  if((dist_vec_a.LocalMap().size() != 1) ||
      (vec_a.n() != 1))
  {
    statusOFS << std::endl << " Unacceptable vector in norm routine !! Aborting ... " << std::endl;
    exit(1);
  }

  // Local result 
  for(Int iter = 0; iter < vec_a.m(); iter ++)
    temp_result += (ptr[iter] * ptr[iter]);

  // Global reduce  across colComm since the partion is by DG elements
  mpi::Allreduce( &temp_result, &final_result, 1, MPI_SUM, domain_.colComm );
  final_result = sqrt(final_result);

  return  final_result;
} // End of routine scfdg_distvec_nrm2

// Computes b = scal_a * a + scal_b * b for conforming distributed vectors / matrices
// Set scal_a = 0.0 and use any vector / matrix a to obtain blas::scal on b
// Set scal_b = 1.0 for blas::axpy with b denoting y, i.e., b = scal_a * a + b;
void 
SCFDG::scfdg_distmat_update(DistVec<Index3, DblNumMat, ElemPrtn>  &dist_mat_a, 
    double scal_a,
    DistVec<Index3, DblNumMat, ElemPrtn>  &dist_mat_b,
    double scal_b)
{
  DblNumMat& mat_a= (dist_mat_a.LocalMap().begin())->second;
  DblNumMat& mat_b= (dist_mat_b.LocalMap().begin())->second;  

  double *ptr_a = mat_a.Data();
  double *ptr_b = mat_b.Data();

  // Conformity check
  if( (dist_mat_a.LocalMap().size() != 1) ||
      (dist_mat_b.LocalMap().size() != 1) ||
      (mat_a.m() != mat_b.m()) ||
      (mat_a.n() != mat_b.n()) )
  {
    statusOFS << std::endl << " Non-conforming distributed vectors / matrices in update routine !!" 
      << std::endl << " Aborting ... " << std::endl;
    exit(1);
  }

  for(Int iter = 0; iter < (mat_a.m() * mat_a.n()) ; iter ++)
    ptr_b[iter] = scal_a * ptr_a[iter] + scal_b * ptr_b[iter];
} // End of routine scfdg_distvec_update

// This routine computes Hmat_times_my_dist_mat = Hmat_times_my_dist_mat + Hmat * my_dist_mat
void SCFDG::scfdg_hamiltonian_times_distmat(DistVec<Index3, DblNumMat, ElemPrtn>  &my_dist_mat, 
    DistVec<Index3, DblNumMat, ElemPrtn>  &Hmat_times_my_dist_mat)
{
  Int mpirank, mpisize;
  MPI_Comm_rank( domain_.comm, &mpirank );
  MPI_Comm_size( domain_.comm, &mpisize );

  HamiltonianDG&  hamDG = *hamDGPtr_;
  std::vector<Index3>  getKeys_list;

  // Check that vectors provided only contain one entry in the local map
  // This is a safeguard to ensure that we are really dealing with distributed matrices
  if((my_dist_mat.LocalMap().size() != 1) ||
      (Hmat_times_my_dist_mat.LocalMap().size() != 1) ||
      ((my_dist_mat.LocalMap().begin())->first != (Hmat_times_my_dist_mat.LocalMap().begin())->first))
  {
    statusOFS << std::endl << " Vectors in Hmat * vector_block product not formatted correctly !!"
      << std::endl << " Aborting ... " << std::endl;
    exit(1);
  }

  // Obtain key based on my_dist_mat : This assumes that my_dist_mat is formatted correctly
  // based on processor number, etc.
  Index3 key = (my_dist_mat.LocalMap().begin())->first;

  // Obtain keys of neighbors using the Hamiltonian matrix
  // We only use these keys to minimize communication in GetBegin since other parts of the vector
  // block, even if they are non-zero will get multiplied by the zero block of the Hamiltonian
  // anyway.
  for(typename std::map<ElemMatKey, DblNumMat >::iterator 
      get_neighbors_from_Ham_iterator = hamDG.HMat().LocalMap().begin();
      get_neighbors_from_Ham_iterator != hamDG.HMat().LocalMap().end();
      ++get_neighbors_from_Ham_iterator)
  {
    Index3 neighbor_key = (get_neighbors_from_Ham_iterator->first).second;

    if(neighbor_key == key)
      continue;
    else
      getKeys_list.push_back(neighbor_key);
  }

  // Do the communication necessary to get the information from
  // procs holding the neighbors
  // Supposedly, independent row communicators (i.e. colComm)
  //  are being used for this
  my_dist_mat.GetBegin( getKeys_list, NO_MASK ); 
  my_dist_mat.GetEnd( NO_MASK );


  // Obtain a reference to the chunk where we want to store
  DblNumMat& mat_Y_local = Hmat_times_my_dist_mat.LocalMap()[key];

  // Now pluck out relevant chunks of the Hamiltonian and the vector and multiply
  for(typename std::map<Index3, DblNumMat >::iterator 
      mat_X_iterator = my_dist_mat.LocalMap().begin();
      mat_X_iterator != my_dist_mat.LocalMap().end(); ++mat_X_iterator ){

    Index3 iter_key = mat_X_iterator->first;       
    DblNumMat& mat_X_local = mat_X_iterator->second; // Chunk from input block of vectors

    // Create key for looking up Hamiltonian chunk 
    ElemMatKey myelemmatkey = std::make_pair(key, iter_key);

    std::map<ElemMatKey, DblNumMat >::iterator ham_iterator = hamDG.HMat().LocalMap().find(myelemmatkey);

    //statusOFS << std::endl << " Working on key " << key << "   " << iter_key << std::endl;

    // Now do the actual multiplication
    DblNumMat& mat_H_local = ham_iterator->second; // Chunk from Hamiltonian

    Int m = mat_H_local.m(), n = mat_X_local.n(), k = mat_H_local.n();

    blas::Gemm( 'N', 'N', m, n, k, 
        1.0, mat_H_local.Data(), m, 
        mat_X_local.Data(), k, 
        1.0, mat_Y_local.Data(), m);


  } // End of loop using mat_X_iterator

  // Matrix * vector_block product is ready now ... 
  // Need to clean up extra entries in my_dist_mat
  typename std::map<Index3, DblNumMat >::iterator it;
  for(Int delete_iter = 0; delete_iter <  getKeys_list.size(); delete_iter ++)
  {
    it = my_dist_mat.LocalMap().find(getKeys_list[delete_iter]);
    (my_dist_mat.LocalMap()).erase(it);
  }
}

// This routine estimates the spectral bounds using the Lanczos method
double 
SCFDG::scfdg_Cheby_Upper_bound_estimator(DblNumVec& ritz_values, 
    int Num_Lanczos_Steps
    )
{
  Int mpirank, mpisize;
  MPI_Comm_rank( domain_.comm, &mpirank );
  MPI_Comm_size( domain_.comm, &mpisize );

  Real timeSta, timeEnd;
  Real timeIterStart, timeIterEnd;

  HamiltonianDG&  hamDG = *hamDGPtr_;

  // Declare vectors partioned according to DG elements
  // These will be used for Lanczos

  // Input vector
  DistVec<Index3, DblNumMat, ElemPrtn>  dist_vec_v; 
  dist_vec_v.Prtn() = elemPrtn_;
  dist_vec_v.SetComm(domain_.colComm);

  // vector to hold f = H*v
  DistVec<Index3, DblNumMat, ElemPrtn>  dist_vec_f; 
  dist_vec_f.Prtn() = elemPrtn_;
  dist_vec_f.SetComm(domain_.colComm);

  // vector v0
  DistVec<Index3, DblNumMat, ElemPrtn>  dist_vec_v0; 
  dist_vec_v0.Prtn() = elemPrtn_;
  dist_vec_v0.SetComm(domain_.colComm);

  // Step 1 : Generate a random vector v
  // Fill up the vector v using random entries
  // Also, initialize the vectors f and v0 to 0

  // We use this loop for setting up the keys. 
  for( Int k = 0; k < numElem_[2]; k++ )
    for( Int j = 0; j < numElem_[1]; j++ )
      for( Int i = 0; i < numElem_[0]; i++ ){
        Index3 key( i, j, k );

        // If the current processor owns this element
        if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){ 

          // Associate the current key with a vector that contains the stuff 
          // that should reside on this process.
          const std::vector<Int>&  idx = hamDG.ElemBasisIdx()(i, j, k); 

          dist_vec_v.LocalMap()[key].Resize(idx.size(), 1); // Because of this, the LocalMap is of size 1 now        
          dist_vec_f.LocalMap()[key].Resize(idx.size(), 1); // Because of this, the LocalMap is of size 1 now
          dist_vec_v0.LocalMap()[key].Resize(idx.size(), 1); // Because of this, the LocalMap is of size 1 now

          // Initialize the local maps    
          // Vector v is initialized randomly
          UniformRandom(dist_vec_v.LocalMap()[key]);

          // Vector f and v0 are filled with zeros
          SetValue(dist_vec_f.LocalMap()[key], 0.0);
          SetValue(dist_vec_v0.LocalMap()[key], 0.0);

        }
      } // End of vector initializations

  // Normalize the vector v
  double norm_v = scfdg_distvec_nrm2(dist_vec_v);     
  scfdg_distmat_update(dist_vec_v, 0.0, dist_vec_v, (1.0 / norm_v));

  // Step 2a : f = H * v 
  //scfdg_hamiltonian_times_distvec(dist_vec_v,  dist_vec_f); // f has already been set to 0
  scfdg_hamiltonian_times_distmat(dist_vec_v,  dist_vec_f); // f has already been set to 0

  // Step 2b : alpha = f^T * v
  double alpha, beta;
  alpha = scfdg_distvec_dot(dist_vec_f, dist_vec_v);

  // Step 2c : f = f - alpha * v
  scfdg_distmat_update(dist_vec_v, (-alpha), dist_vec_f, 1.0);

  // Step 2d: T(1,1) = alpha
  DblNumMat matT( Num_Lanczos_Steps, Num_Lanczos_Steps );
  SetValue(matT, 0.0);    
  matT(0,0) = alpha;

  // Step 3: Lanczos iteration
  for(Int j = 1; j < Num_Lanczos_Steps; j ++)
  {
    // Step 4 : beta = norm(f)
    beta = scfdg_distvec_nrm2(dist_vec_f);

    // Step 5a : v0 = v
    scfdg_distmat_update(dist_vec_v, 1.0, dist_vec_v0,  0.0);

    // Step 5b : v = f / beta
    scfdg_distmat_update(dist_vec_f, (1.0 / beta), dist_vec_v, 0.0);

    // Step 6a : f = H * v
    SetValue((dist_vec_f.LocalMap().begin())->second, 0.0);// Set f to zero first !
    scfdg_hamiltonian_times_distmat(dist_vec_v,  dist_vec_f);

    // Step 6b : f = f - beta * v0
    scfdg_distmat_update(dist_vec_v0, (-beta) , dist_vec_f,  1.0);

    // Step 7a : alpha = f^T * v
    alpha = scfdg_distvec_dot(dist_vec_f, dist_vec_v);

    // Step 7b : f = f - alpha * v
    scfdg_distmat_update(dist_vec_v, (-alpha) , dist_vec_f,  1.0);

    // Step 8 : Fill up the matrix
    matT(j, j - 1) = beta;
    matT(j - 1, j) = beta;
    matT(j, j) = alpha;
  } // End of loop over Lanczos steps 

  // Step 9 : Compute the Lanczos-Ritz values
  ritz_values.Resize(Num_Lanczos_Steps);
  SetValue( ritz_values, 0.0 );

  // Solve the eigenvalue problem for the Ritz values
  lapack::Syevd( 'N', 'U', Num_Lanczos_Steps, matT.Data(), Num_Lanczos_Steps, ritz_values.Data() );

  // Step 10 : Compute the upper bound on each process
  double b_up = ritz_values(Num_Lanczos_Steps - 1) + scfdg_distvec_nrm2(dist_vec_f);

    statusOFS << std::endl << " Ritz values in estimator here : " << ritz_values ;
    statusOFS << std::endl << " Upper bound of spectrum = " << b_up << std::endl;

  // Need to synchronize the Ritz values and the upper bound across the processes
  MPI_Bcast(&b_up, 1, MPI_DOUBLE, 0, domain_.comm);
  MPI_Bcast(ritz_values.Data(), Num_Lanczos_Steps, MPI_DOUBLE, 0, domain_.comm);

  return b_up;

} // End of scfdg_Cheby_Upper_bound_estimator

// Apply the scaled Chebyshev Filter on the Eigenvectors
// Use a distributor to work on selected bands based on
// number of processors per element (i.e., no. of rows in process grid).
void 
SCFDG::scfdg_Chebyshev_filter_scaled(int m, 
    double a, 
    double b, 
    double a_L)
{
  Int mpirank, mpisize;
  MPI_Comm_rank( domain_.comm, &mpirank );
  MPI_Comm_size( domain_.comm, &mpisize );
  Int mpirankRow;  MPI_Comm_rank(domain_.rowComm, &mpirankRow);
  Int mpisizeRow;  MPI_Comm_size(domain_.rowComm, &mpisizeRow);
  Int mpirankCol;  MPI_Comm_rank(domain_.colComm, &mpirankCol);
  Int mpisizeCol;  MPI_Comm_size(domain_.colComm, &mpisizeCol);

  Real timeSta, timeEnd;
  Real extra_timeSta, extra_timeEnd;
  Real filter_total_time = 0.0;

  HamiltonianDG&  hamDG = *hamDGPtr_;

  // We need to distribute bands according to rowComm since colComm is
  // used for element-wise partition.
  if(mpisizeRow > hamDG.NumStateTotal())
  {
    statusOFS << std::endl << " Number of processors per element exceeds number of bands !!"
      << std::endl << " Cannot continue with band-parallelization. "
      << std::endl << " Aborting ... " << std::endl;
    exit(1);

  }
  simple_distributor band_distributor(hamDG.NumStateTotal(), mpisizeRow, mpirankRow);

  // Create distributed matrices pluck_X, pluck_Y, pluck_Yt for filtering 
  const Index3 key = (hamDG.EigvecCoef().LocalMap().begin())->first; // Will use same key as eigenvectors
  DblNumMat &eigvecs_local = (hamDG.EigvecCoef().LocalMap().begin())->second;

  Int local_width = band_distributor.current_proc_size;
  Int local_height = eigvecs_local.m();
  Int local_pluck_sz = local_height * local_width;

  DistVec<Index3, DblNumMat, ElemPrtn>  pluck_X; 
  pluck_X.Prtn() = elemPrtn_;
  pluck_X.SetComm(domain_.colComm);
  pluck_X.LocalMap()[key].Resize(local_height, local_width);

  DistVec<Index3, DblNumMat, ElemPrtn>  pluck_Y; 
  pluck_Y.Prtn() = elemPrtn_;
  pluck_Y.SetComm(domain_.colComm);
  pluck_Y.LocalMap()[key].Resize(local_height, local_width);

  DistVec<Index3, DblNumMat, ElemPrtn>  pluck_Yt; 
  pluck_Yt.Prtn() = elemPrtn_;
  pluck_Yt.SetComm(domain_.colComm);
  pluck_Yt.LocalMap()[key].Resize(local_height, local_width);


  // Initialize the distributed matrices
  blas::Copy(local_pluck_sz, 
      eigvecs_local.Data() + local_height * band_distributor.current_proc_start, 
      1,
      pluck_X.LocalMap()[key].Data(),
      1);

  SetValue(pluck_Y.LocalMap()[key], 0.0);
  SetValue(pluck_Yt.LocalMap()[key], 0.0);

  // Filtering scalars
  double e = (b - a) / 2.0;
  double c = (a + b) / 2.0;
  double sigma = e / (c - a_L);
  double tau = 2.0 / sigma;
  double sigma_new;

  // Begin the filtering process
  // Y = (H * X - c * X) * (sigma / e)
  // pluck_Y has been initialized to 0 already

  statusOFS << std::endl << " Chebyshev filtering : Process " << mpirank << " working on " 
    << local_width << " of " << eigvecs_local.n() << " bands.";

  statusOFS << std::endl << " Chebyshev filtering : Lower bound = " << a
    << std::endl << "                     : Upper bound = " << b
    << std::endl << "                     : a_L = " << a_L;

  //statusOFS << std::endl << " Chebyshev filtering step 1 of " << m << " ... ";
  GetTime( extra_timeSta );

  scfdg_hamiltonian_times_distmat(pluck_X, pluck_Y); // Y = H * X
  scfdg_distmat_update(pluck_X, (-c) , pluck_Y,  1.0); // Y = -c * X + 1.0 * Y
  scfdg_distmat_update(pluck_Y, 0.0 , pluck_Y,  (sigma / e)); // Y = 0.0 * Y + (sigma / e) * Y

  GetTime( extra_timeEnd );
  //statusOFS << " Done. ( " << (extra_timeEnd - extra_timeSta ) << " s.)";
  filter_total_time += (extra_timeEnd - extra_timeSta );
  for(Int filter_iter = 2; filter_iter < m; filter_iter ++)
  {   
    //statusOFS << std::endl << " Chebyshev filtering step " << filter_iter << " of " << m << " ... ";
    GetTime( extra_timeSta );

    sigma_new = 1.0 / (tau - sigma);

    //Compute Yt = (H * Y - c * Y) * (2 * sigma_new / e) - (sigma * sigma_new) * X
    // Set Yt to 0
    SetValue(pluck_Yt.LocalMap()[key], 0.0);
    scfdg_hamiltonian_times_distmat(pluck_Y, pluck_Yt); // Yt = H * Y
    scfdg_distmat_update(pluck_Y, (-c) , pluck_Yt,  1.0); // Yt = -c * Y + 1.0 * Yt
    scfdg_distmat_update(pluck_Yt, 0.0 , pluck_Yt,  (2.0 * sigma_new / e)); // Yt = 0.0 * Yt + (2.0 * sigma_new / e) * Yt
    scfdg_distmat_update(pluck_X, (-sigma * sigma_new) , pluck_Yt,  1.0 ); // Yt = (-sigma * sigma_new) * X + 1.0 * Yt

    // Update / Re-assign : X = Y, Y = Yt, sigma = sigma_new
    scfdg_distmat_update(pluck_Y, 1.0 , pluck_X,  0.0 ); // X = 0.0 * X + 1.0 * Y
    scfdg_distmat_update(pluck_Yt, 1.0 , pluck_Y,  0.0 ); // Y = 0.0 * Y + 1.0 * Yt

    sigma = sigma_new;   

    GetTime( extra_timeEnd );
    //statusOFS << " Done. ( " << (extra_timeEnd - extra_timeSta ) << " s.)";
    filter_total_time += (extra_timeEnd - extra_timeSta );

  }

  statusOFS << std::endl <<  " Total filtering time for " 
    << m << " filter steps = " << filter_total_time << " s."
    << std::endl <<  " Average per filter step = " << ( filter_total_time / double(m) ) << " s.";

  // pluck_Y contains the results of filtering.
  // Copy back pluck_Y to the eigenvector
  // SetValue(eigvecs_local, 0.0); // All entries set to zero for All-Reduce
  GetTime( extra_timeSta );

  DblNumMat temp_buffer;
  temp_buffer.Resize(eigvecs_local.m(), eigvecs_local.n());
  SetValue(temp_buffer, 0.0);

  blas::Copy(local_pluck_sz, 
      pluck_Y.LocalMap()[key].Data(),
      1,
      temp_buffer.Data() + local_height * band_distributor.current_proc_start,
      1);

  MPI_Allreduce(temp_buffer.Data(),
      eigvecs_local.Data(),
      (eigvecs_local.m() * eigvecs_local.n()),
      MPI_DOUBLE,
      MPI_SUM,
      domain_.rowComm);

  GetTime( extra_timeEnd );
  statusOFS << std::endl << " Eigenvector block rebuild time = " 
    << (extra_timeEnd - extra_timeSta ) << " s.";
} // End of scfdg_Chebyshev_filter

// Apply the Hamiltonian to the Eigenvectors and place result in result_mat
// This routine is actually not used by the Chebyshev filter routine since all 
// the filtering can be done block-wise to reduce communication 
// among processor rows (i.e., rowComm), i.e., we do the full filter and 
// communicate only in the end.
// This routine is used for the Raleigh-Ritz step.
// Uses a distributor to work on selected bands based on
// number of processors per element (i.e., no. of rows in process grid).  
void 
SCFDG::scfdg_Hamiltonian_times_eigenvectors(DistVec<Index3, DblNumMat, ElemPrtn>  &result_mat)
{
  Int mpirank, mpisize;
  MPI_Comm_rank( domain_.comm, &mpirank );
  MPI_Comm_size( domain_.comm, &mpisize );
  Int mpirankRow;  MPI_Comm_rank(domain_.rowComm, &mpirankRow);
  Int mpisizeRow;  MPI_Comm_size(domain_.rowComm, &mpisizeRow);
  Int mpirankCol;  MPI_Comm_rank(domain_.colComm, &mpirankCol);
  Int mpisizeCol;  MPI_Comm_size(domain_.colComm, &mpisizeCol);

  Real timeSta, timeEnd;
  Real extra_timeSta, extra_timeEnd;
  Real filter_total_time = 0.0;

  HamiltonianDG&  hamDG = *hamDGPtr_;

  // We need to distribute bands according to rowComm since colComm is
  // used for element-wise partition.
  if(mpisizeRow > hamDG.NumStateTotal())
  {
    statusOFS << std::endl << " Number of processors per element exceeds number of bands !!"
      << std::endl << " Cannot continue with band-parallelization. "
      << std::endl << " Aborting ... " << std::endl;
    exit(1);

  }
  simple_distributor band_distributor(hamDG.NumStateTotal(), mpisizeRow, mpirankRow);

  // Create distributed matrices pluck_X, pluck_Y, pluck_Yt for filtering 
  const Index3 key = (hamDG.EigvecCoef().LocalMap().begin())->first; // Will use same key as eigenvectors
  DblNumMat &eigvecs_local = (hamDG.EigvecCoef().LocalMap().begin())->second;

  Int local_width = band_distributor.current_proc_size;
  Int local_height = eigvecs_local.m();
  Int local_pluck_sz = local_height * local_width;

  DistVec<Index3, DblNumMat, ElemPrtn>  pluck_X; 
  pluck_X.Prtn() = elemPrtn_;
  pluck_X.SetComm(domain_.colComm);
  pluck_X.LocalMap()[key].Resize(local_height, local_width);

  DistVec<Index3, DblNumMat, ElemPrtn>  pluck_Y; 
  pluck_Y.Prtn() = elemPrtn_;
  pluck_Y.SetComm(domain_.colComm);
  pluck_Y.LocalMap()[key].Resize(local_height, local_width);

  // Initialize the distributed matrices
  blas::Copy(local_pluck_sz, 
      eigvecs_local.Data() + local_height * band_distributor.current_proc_start, 
      1,
      pluck_X.LocalMap()[key].Data(),
      1);

  SetValue(pluck_Y.LocalMap()[key], 0.0); // pluck_Y is initialized to 0 

  GetTime( extra_timeSta );

  scfdg_hamiltonian_times_distmat(pluck_X, pluck_Y); // Y = H * X

  GetTime( extra_timeEnd );

  statusOFS << std::endl << " Hamiltonian times eigenvectors calculation time = " 
    << (extra_timeEnd - extra_timeSta ) << " s.";

  // Copy pluck_Y to result_mat after preparing it
  GetTime( extra_timeSta );

  DblNumMat temp_buffer;
  temp_buffer.Resize(eigvecs_local.m(), eigvecs_local.n());
  SetValue(temp_buffer, 0.0);

  blas::Copy(local_pluck_sz, 
      pluck_Y.LocalMap()[key].Data(),
      1,
      temp_buffer.Data() + local_height * band_distributor.current_proc_start,
      1);

  // Empty out the results and prepare the distributed matrix
  result_mat.LocalMap().clear();
  result_mat.Prtn() = elemPrtn_;
  result_mat.SetComm(domain_.colComm);
  result_mat.LocalMap()[key].Resize(eigvecs_local.m(), eigvecs_local.n());

  MPI_Allreduce(temp_buffer.Data(),
      result_mat.LocalMap()[key].Data(),
      (eigvecs_local.m() * eigvecs_local.n()),
      MPI_DOUBLE,
      MPI_SUM,
      domain_.rowComm);

  GetTime( extra_timeEnd );
  statusOFS << std::endl << " Eigenvector block rebuild time = " 
    << (extra_timeEnd - extra_timeSta ) << " s.";
} // End of scfdg_Hamiltonian_times_eigenvectors

// Given a block of eigenvectors (size:  hamDG.NumBasisTotal() * hamDG.NumStateTotal()),
// convert this to ScaLAPACK format for subsequent use with ScaLAPACK routines
// Should only be called by processors for which context > 0
// This routine is largely based on the Hamiltonian to ScaLAPACK conversion routine which is similar
void 
SCFDG::scfdg_Cheby_convert_eigvec_distmat_to_ScaLAPACK(DistVec<Index3, DblNumMat, ElemPrtn>  &my_dist_mat, 
    MPI_Comm comm,
    dgdft::scalapack::Descriptor &my_scala_descriptor,
    dgdft::scalapack::ScaLAPACKMatrix<Real>  &my_scala_mat)
{
  HamiltonianDG&  hamDG = *hamDGPtr_;

  // Load up the important ScaLAPACK info
  int nprow = my_scala_descriptor.NpRow();
  int npcol = my_scala_descriptor.NpCol();
  int myprow = my_scala_descriptor.MypRow();
  int mypcol = my_scala_descriptor.MypCol();

  // Set the descriptor for the ScaLAPACK matrix
  my_scala_mat.SetDescriptor( my_scala_descriptor );

  // Save the original key for the distributed vector
  const Index3 my_original_key = (my_dist_mat.LocalMap().begin())->first;

  // Get some basic information set up
  Int mpirank, mpisize;
  MPI_Comm_rank( comm, &mpirank );
  MPI_Comm_size( comm, &mpisize );

  Int MB = my_scala_mat.MB(); 
  Int NB = my_scala_mat.NB();

  if( MB != NB )
  {
    ErrorHandling("MB must be equal to NB.");
  }

  // ScaLAPACK block information
  Int numRowBlock = my_scala_mat.NumRowBlocks();
  Int numColBlock = my_scala_mat.NumColBlocks();

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

  // ScaLAPACK block partition 
  BlockMatPrtn  blockPrtn;

  // Fill up the owner information
  IntNumMat&    blockOwner = blockPrtn.ownerInfo;
  blockOwner.Resize( numRowBlock, numColBlock );   
  for( Int jb = 0; jb < numColBlock; jb++ ){
    for( Int ib = 0; ib < numRowBlock; ib++ ){
      blockOwner( ib, jb ) = procGrid( ib % nprow, jb % npcol );
    }
  }

  // Distributed matrix in ScaLAPACK format
  DistVec<Index2, DblNumMat, BlockMatPrtn> distScaMat;
  distScaMat.Prtn() = blockPrtn;
  distScaMat.SetComm(comm);

  // Zero out and initialize
  DblNumMat empty_mat( MB, MB ); // As MB = NB
  SetValue( empty_mat, 0.0 );
  // We need this loop here since we are initializing the distributed 
  // ScaLAPACK matrix. Subsequently, the LocalMap().begin()->first technique can be used
  for( Int jb = 0; jb < numColBlock; jb++ )
    for( Int ib = 0; ib < numRowBlock; ib++ )
    {
      Index2 key( ib, jb );
      if( distScaMat.Prtn().Owner( key ) == mpirank )
      {
        distScaMat.LocalMap()[key] = empty_mat;
      }
    }
  // Copy data from DG distributed matrix to ScaLAPACK distributed matrix
  DblNumMat& localMat = (my_dist_mat.LocalMap().begin())->second;
  const std::vector<Int>& my_idx = hamDG.ElemBasisIdx()( my_original_key(0), my_original_key(1), my_original_key(2) );
  {
    Int ib, jb, io, jo;
    for( Int b = 0; b < localMat.n(); b++ )
    {
      for( Int a = 0; a < localMat.m(); a++ )
      {
        ib = my_idx[a] / MB;
        io = my_idx[a] % MB;

        jb = b / MB;            
        jo = b % MB;

        typename std::map<Index2, DblNumMat >::iterator 
          ni = distScaMat.LocalMap().find( Index2(ib, jb) );
        if( ni == distScaMat.LocalMap().end() )
        {
          distScaMat.LocalMap()[Index2(ib, jb)] = empty_mat;
          ni = distScaMat.LocalMap().find( Index2(ib, jb) );
        }

        DblNumMat&  localScaMat = ni->second;
        localScaMat(io, jo) += localMat(a, b);
      } // for (a)
    } // for (b)

  }     

  // Communication step
  {
    // Prepare
    std::vector<Index2>  keyIdx;
    for( typename std::map<Index2, DblNumMat >::iterator 
        mi  = distScaMat.LocalMap().begin();
        mi != distScaMat.LocalMap().end(); ++mi )
    {
      Index2 key = mi->first;

      // Include all keys which do not reside on current processor
      if( distScaMat.Prtn().Owner( key ) != mpirank )
      {
        keyIdx.push_back( key );
      }
    } // for (mi)

    // Communication
    distScaMat.PutBegin( keyIdx, NO_MASK );
    distScaMat.PutEnd( NO_MASK, PutMode::COMBINE );

    // Clean up
    std::vector<Index2>  eraseKey;
    for( typename std::map<Index2, DblNumMat >::iterator 
        mi  = distScaMat.LocalMap().begin();
        mi != distScaMat.LocalMap().end(); ++mi)
    {
      Index2 key = mi->first;
      if( distScaMat.Prtn().Owner( key ) != mpirank )
      {
        eraseKey.push_back( key );
      }
    } // for (mi)

    for( std::vector<Index2>::iterator vi = eraseKey.begin();
        vi != eraseKey.end(); ++vi )
    {
      distScaMat.LocalMap().erase( *vi );
    }    

  } // End of communication step

  // Final step: Copy from distributed ScaLAPACK matrix to local part of actual
  // ScaLAPACK matrix
  {
    for( typename std::map<Index2, DblNumMat>::iterator 
        mi  = distScaMat.LocalMap().begin();
        mi != distScaMat.LocalMap().end(); ++mi )
    {
      Index2 key = mi->first;
      if( distScaMat.Prtn().Owner( key ) == mpirank )
      {
        Int ib = key(0), jb = key(1);
        Int offset = ( jb / npcol ) * MB * my_scala_mat.LocalLDim() + 
          ( ib / nprow ) * MB;
        lapack::Lacpy( 'A', MB, MB, mi->second.Data(),
            MB, my_scala_mat.Data() + offset, my_scala_mat.LocalLDim() );
      } // own this block
    } // for (mi)
  }
} // End of scfdg_Cheby_convert_eigvec_distmat_to_ScaLAPACK

void 
SCFDG::scfdg_FirstChebyStep(Int MaxIter,
    Int filter_order )
{
  Int mpirank, mpisize;
  MPI_Comm_rank( domain_.comm, &mpirank );
  MPI_Comm_size( domain_.comm, &mpisize );
  Int mpirankRow;  MPI_Comm_rank(domain_.rowComm, &mpirankRow);
  Int mpisizeRow;  MPI_Comm_size(domain_.rowComm, &mpisizeRow);
  Int mpirankCol;  MPI_Comm_rank(domain_.colComm, &mpirankCol);
  Int mpisizeCol;  MPI_Comm_size(domain_.colComm, &mpisizeCol);

  Real timeSta, timeEnd;
  Real extra_timeSta, extra_timeEnd;
  Real cheby_timeSta, cheby_timeEnd;

  HamiltonianDG&  hamDG = *hamDGPtr_;

  // Step 1: Obtain the upper bound and the Ritz values (for the lower bound)
  // using the Lanczos estimator

  DblNumVec Lanczos_Ritz_values;

  statusOFS << std::endl << " Estimating the spectral bounds ... "; 
  GetTime( extra_timeSta );
  const int Num_Lanczos_Steps = 6;
  double b_up = scfdg_Cheby_Upper_bound_estimator(Lanczos_Ritz_values, Num_Lanczos_Steps);  
  GetTime( extra_timeEnd );
  statusOFS << " Done. ( " << (extra_timeEnd - extra_timeSta ) << " s.)";

  double a_L = Lanczos_Ritz_values(0);
  double beta = 0.5; // 0.5 <= beta < 1.0
  double b_low = beta * Lanczos_Ritz_values(0) + (1.0 - beta) * Lanczos_Ritz_values(Num_Lanczos_Steps - 1);
  //b_low = 0.0; // This is probably better than the above estimate based on Lanczos

  // Step 2: Initialize the eigenvectors to random numbers
  statusOFS << std::endl << " Initializing eigenvectors randomly on first SCF step ... "; 

  hamDG.EigvecCoef().Prtn() = elemPrtn_;
  hamDG.EigvecCoef().SetComm(domain_.colComm);

  GetTime( extra_timeSta );

  // This is one of the very few places where we use this loop 
  // for setting up the keys. In other cases, we can simply access
  // the first and second elements of the LocalMap of the distributed vector
  // once it has been set up.
  for( Int k = 0; k < numElem_[2]; k++ )
    for( Int j = 0; j < numElem_[1]; j++ )
      for( Int i = 0; i < numElem_[0]; i++ ){
        Index3 key( i, j, k );

        // If the current processor owns this element
        if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){ 

          // Save the key 
          my_cheby_eig_vec_key_ = key;

          // Associate the current key with a vector that contains the stuff 
          // that should reside on this process.
          const std::vector<Int>&  idx = hamDG.ElemBasisIdx()(i, j, k); 
          hamDG.EigvecCoef().LocalMap()[key].Resize(idx.size(), hamDG.NumStateTotal()); 

          DblNumMat &ref_mat =  hamDG.EigvecCoef().LocalMap()[key];

          // Only the first processor on every column does this
          if(mpirankRow == 0)          
            UniformRandom(ref_mat);
          // Now broadcast this
          MPI_Bcast(ref_mat.Data(), (ref_mat.m() * ref_mat.n()), MPI_DOUBLE, 0, domain_.rowComm);

        }
      } // End of eigenvector initialization
  GetTime( extra_timeEnd );
  statusOFS << " Done. ( " << (extra_timeEnd - extra_timeSta ) << " s.)";

  // Step 3: Main loop
  const Int Iter_Max = MaxIter;
  const Int Filter_Order = filter_order ;

  DblNumVec eig_vals_Raleigh_Ritz;

  for(Int i = 1; i <= Iter_Max; i ++)
  {
    GetTime( cheby_timeSta );
    statusOFS << std::endl << std::endl << " ------------------------------- ";
    statusOFS << std::endl << " First CheFSI step for DGDFT cycle " << i << " of " << Iter_Max << " . ";
    // Filter the eigenvectors
    statusOFS << std::endl << std::endl << " Filtering the eigenvectors ... (Filter order = " << Filter_Order << ")";
    GetTime( timeSta );
    scfdg_Chebyshev_filter_scaled(Filter_Order, b_low, b_up, a_L);
    GetTime( timeEnd );
    statusOFS << std::endl << " Filtering completed. ( " << (timeEnd - timeSta ) << " s.)";

    // Subspace projected problems: Orthonormalization, Raleigh-Ritz and subspace rotation steps
    // This can be done serially or using ScaLAPACK    
    if(SCFDG_Cheby_use_ScaLAPACK_ == 0)
    {
      // Do the subspace problem serially 
      statusOFS << std::endl << std::endl << " Solving subspace problems serially ...";

      // Orthonormalize using Cholesky factorization  
      statusOFS << std::endl << " Orthonormalizing filtered vectors ... ";
      GetTime( timeSta );

      DblNumMat &local_eigvec_mat = (hamDG.EigvecCoef().LocalMap().begin())->second;
      DblNumMat square_mat;
      DblNumMat temp_square_mat;

      Int width = local_eigvec_mat.n();
      Int height_local = local_eigvec_mat.m();

      square_mat.Resize(width, width);
      temp_square_mat.Resize(width, width);

      SetValue(temp_square_mat, 0.0);

      // Compute square_mat = X^T * X for Cholesky    
      blas::Gemm( 'T', 'N', width, width, height_local, 
          1.0, local_eigvec_mat.Data(), height_local,
          local_eigvec_mat.Data(), height_local, 
          0.0, temp_square_mat.Data(), width );

      SetValue( square_mat, 0.0 );
      MPI_Allreduce( temp_square_mat.Data(), square_mat.Data(), width*width, MPI_DOUBLE, MPI_SUM, domain_.colComm );

      // In the following, reduction happens on colComm but the result is broadcast to everyone
      // This can probably be band-parallelized

      // Make the Cholesky factorization call on proc 0
      if ( mpirank == 0) {
        lapack::Potrf( 'U', width, square_mat.Data(), width );
      }
      // Send the Cholesky factor to every process
      MPI_Bcast(square_mat.Data(), width*width, MPI_DOUBLE, 0, domain_.comm);

      // Do a solve with the Cholesky factor : Band parallelization ??
      // X = X * U^{-1} is orthogonal, where U is the Cholesky factor
      blas::Trsm( 'R', 'U', 'N', 'N', height_local, width, 1.0, square_mat.Data(), width, 
          local_eigvec_mat.Data(), height_local );

      GetTime( timeEnd );
      statusOFS << std::endl << " Orthonormalization completed ( " << (timeEnd - timeSta ) << " s.)";

      // Raleigh-Ritz step: This part is non-scalable and needs to be fixed
      // !!!!!!!!!!!!!!!!!!!!!!!!!!!!
      statusOFS << std::endl << std::endl << " Raleigh-Ritz step ... ";
      GetTime( timeSta );

      // Compute H * X
      DistVec<Index3, DblNumMat, ElemPrtn>  result_mat;
      scfdg_Hamiltonian_times_eigenvectors(result_mat);
      DblNumMat &local_result_mat = (result_mat.LocalMap().begin())->second;

      SetValue(temp_square_mat, 0.0);

      // Compute square_mat = X^T * HX 
      blas::Gemm( 'T', 'N', width, width, height_local, 
          1.0, local_eigvec_mat.Data(), height_local,
          local_result_mat.Data(), height_local, 
          0.0, temp_square_mat.Data(), width );

      SetValue( square_mat, 0.0 );
      MPI_Allreduce( temp_square_mat.Data(), square_mat.Data(), width*width, MPI_DOUBLE, MPI_SUM, domain_.colComm );

      eig_vals_Raleigh_Ritz.Resize(width);
      SetValue(eig_vals_Raleigh_Ritz, 0.0);

      if ( mpirank == 0 ) {
        lapack::Syevd( 'V', 'U', width, square_mat.Data(), width, eig_vals_Raleigh_Ritz.Data() );

      }

      // Send the results to every process
      MPI_Bcast(square_mat.Data(), width*width, MPI_DOUBLE, 0, domain_.comm); // Eigen-vectors
      MPI_Bcast(eig_vals_Raleigh_Ritz.Data(), width, MPI_DOUBLE, 0,  domain_.comm); // Eigen-values

      GetTime( timeEnd );
      statusOFS << std::endl << " Raleigh-Ritz step completed ( " << (timeEnd - timeSta ) << " s.)";

      // Subspace rotation step X <- X * Q: This part is non-scalable and needs to be fixed
      // !!!!!!!!!!!!!!!!!!!!!!!!!!!!
      statusOFS << std::endl << std::endl << " Subspace rotation step ... ";
      GetTime( timeSta );

      // So copy X to HX 
      lapack::Lacpy( 'A', height_local, width, local_eigvec_mat.Data(),  height_local, local_result_mat.Data(), height_local );

      // Gemm: X <-- HX (= X) * Q
      blas::Gemm( 'N', 'N', height_local, width, width, 1.0, local_result_mat.Data(),
          height_local, square_mat.Data(), width, 0.0, local_eigvec_mat.Data(), height_local );    

      GetTime( timeEnd );
      statusOFS << std::endl << " Subspace rotation step completed ( " << (timeEnd - timeSta ) << " s.)";

      // Reset the filtering bounds using results of the Raleigh-Ritz step
      b_low = eig_vals_Raleigh_Ritz(width - 1);
      a_L = eig_vals_Raleigh_Ritz(0);

      // Fill up the eigen-values
      DblNumVec& eigval = hamDG.EigVal(); 
      eigval.Resize( hamDG.NumStateTotal() );    

      for(Int i = 0; i < hamDG.NumStateTotal(); i ++)
        eigval[i] =  eig_vals_Raleigh_Ritz[i];
    } // End of if(SCFDG_Cheby_use_ScaLAPACK_ == 0)
    else
    {
      // Do the subspace problems using ScaLAPACK
      statusOFS << std::endl << std::endl << " Solving subspace problems using ScaLAPACK ...";

      statusOFS << std::endl << " Setting up BLACS and ScaLAPACK Process Grid ...";
      GetTime( timeSta );

      double detail_timeSta, detail_timeEnd;

      // Basic ScaLAPACK setup steps
      //Number of ScaLAPACK processors equal to number of DG elements
      const int num_cheby_scala_procs = mpisizeCol; 

      // Figure out the process grid dimensions
      int temp_factor = int(sqrt(double(num_cheby_scala_procs)));
      while(num_cheby_scala_procs % temp_factor != 0 )
        ++temp_factor;

      // temp_factor now contains the process grid height
      const int cheby_scala_num_rows = temp_factor;      
      const int cheby_scala_num_cols = num_cheby_scala_procs / temp_factor;

      // Set up the ScaLAPACK context
      IntNumVec cheby_scala_pmap(num_cheby_scala_procs);

      // Use the first processor from every DG-element 
      for ( Int pmap_iter = 0; pmap_iter < num_cheby_scala_procs; pmap_iter++ )
        cheby_scala_pmap[pmap_iter] = pmap_iter * mpisizeRow; 

      // Set up BLACS for subsequent ScaLAPACK operations
      Int cheby_scala_context = -2;
      dgdft::scalapack::Cblacs_get( 0, 0, &cheby_scala_context );
      dgdft::scalapack::Cblacs_gridmap(&cheby_scala_context, &cheby_scala_pmap[0], cheby_scala_num_rows, cheby_scala_num_rows, cheby_scala_num_cols);

      // Figure out my ScaLAPACK information
      int dummy_np_row, dummy_np_col;
      int my_cheby_scala_proc_row, my_cheby_scala_proc_col;

      dgdft::scalapack::Cblacs_gridinfo(cheby_scala_context, &dummy_np_row, &dummy_np_col, &my_cheby_scala_proc_row, &my_cheby_scala_proc_col);

      GetTime( timeEnd );
      statusOFS << " Done. ( " << (timeEnd - timeSta ) << " s.)";

      statusOFS << std::endl << " ScaLAPACK will use " << num_cheby_scala_procs << " processes.";
      statusOFS << std::endl << " ScaLAPACK process grid = " << cheby_scala_num_rows << " * " << cheby_scala_num_cols << " ."  << std::endl;

      // Eigenvcetors in ScaLAPACK format : this will be used multiple times
      dgdft::scalapack::ScaLAPACKMatrix<Real>  cheby_scala_eigvecs_X; // Declared here for scope, empty constructor invoked

      if(cheby_scala_context >= 0)
      { 
        // Now setup the ScaLAPACK matrix
        statusOFS << std::endl << " Orthonormalization step : ";
        statusOFS << std::endl << " Distributed vector to ScaLAPACK format conversion ... ";
        GetTime( timeSta );

        // The dimensions should be  hamDG.NumBasisTotal() * hamDG.NumStateTotal()
        // But this is not verified here as far as the distributed vector is concerned
        dgdft::scalapack::Descriptor cheby_eigvec_desc( hamDG.NumBasisTotal(), hamDG.NumStateTotal(), 
            scaBlockSize_, scaBlockSize_, 
            0, 0, 
            cheby_scala_context);

        // Make the conversion call         
        scfdg_Cheby_convert_eigvec_distmat_to_ScaLAPACK(hamDG.EigvecCoef(),
            domain_.colComm,
            cheby_eigvec_desc,
            cheby_scala_eigvecs_X);

        GetTime( timeEnd );
        statusOFS << " Done. ( " << (timeEnd - timeSta ) << " s.)";

        statusOFS << std::endl << " Orthonormalizing filtered vectors ... ";
        GetTime( timeSta );

        GetTime( detail_timeSta);

        // Compute C = X^T * X
        dgdft::scalapack::Descriptor cheby_chol_desc( hamDG.NumStateTotal(), hamDG.NumStateTotal(), 
            scaBlockSize_, scaBlockSize_, 
            0, 0, 
            cheby_scala_context);

        dgdft::scalapack::ScaLAPACKMatrix<Real>  cheby_scala_chol_mat;
        cheby_scala_chol_mat.SetDescriptor(cheby_chol_desc);

        dgdft::scalapack::Gemm( 'T', 'N',
            hamDG.NumStateTotal(), hamDG.NumStateTotal(), hamDG.NumBasisTotal(),
            1.0,
            cheby_scala_eigvecs_X.Data(), I_ONE, I_ONE, cheby_scala_eigvecs_X.Desc().Values(), 
            cheby_scala_eigvecs_X.Data(), I_ONE, I_ONE, cheby_scala_eigvecs_X.Desc().Values(),
            0.0,
            cheby_scala_chol_mat.Data(), I_ONE, I_ONE, cheby_scala_chol_mat.Desc().Values(),
            cheby_scala_context);

        GetTime( detail_timeEnd);
        statusOFS << std::endl << " Overlap matrix computed in : " << (detail_timeEnd - detail_timeSta) << " s.";

        GetTime( detail_timeSta);
        // Compute V = Chol(C)
        dgdft::scalapack::Potrf( 'U', cheby_scala_chol_mat);

        GetTime( detail_timeEnd);
        statusOFS << std::endl << " Cholesky factorization computed in : " << (detail_timeEnd - detail_timeSta) << " s.";

        GetTime( detail_timeSta);
        // Compute  X = X * V^{-1}
        dgdft::scalapack::Trsm( 'R', 'U', 'N', 'N', 1.0,
            cheby_scala_chol_mat, 
            cheby_scala_eigvecs_X );

        GetTime( detail_timeEnd);
        statusOFS << std::endl << " TRSM computed in : " << (detail_timeEnd - detail_timeSta) << " s.";

        GetTime( timeEnd );
        statusOFS << std::endl << " Orthonormalization steps completed ( " << (timeEnd - timeSta ) << " s.)" << std::endl;

        statusOFS << std::endl << " ScaLAPACK to Distributed vector format conversion ... ";
        GetTime( timeSta );

        // Now convert this back to DG-distributed matrix format
        ScaMatToDistNumMat(cheby_scala_eigvecs_X ,
            elemPrtn_,
            hamDG.EigvecCoef(),
            hamDG.ElemBasisIdx(), 
            domain_.colComm, 
            hamDG.NumStateTotal() );

        GetTime( timeEnd );
        statusOFS << " Done. ( " << (timeEnd - timeSta ) << " s.)";

      } // End of  if(cheby_scala_context >= 0)
      else
      {
        statusOFS << std::endl << " Waiting for ScaLAPACK solution of subspace problems ...";
      }        

      // Communicate the orthonormalized eigenvectors (to other intra-element processors)
      statusOFS << std::endl << " Communicating orthonormalized filtered vectors ... ";
      GetTime( timeSta );

      DblNumMat &ref_mat_1 =  (hamDG.EigvecCoef().LocalMap().begin())->second;
      MPI_Bcast(ref_mat_1.Data(), (ref_mat_1.m() * ref_mat_1.n()), MPI_DOUBLE, 0, domain_.rowComm);

      GetTime( timeEnd );
      statusOFS << " Done. ( " << (timeEnd - timeSta ) << " s.)" << std::endl;

      // Set up space for eigenvalues in the Hamiltonian object for results of Raleigh-Ritz step
      DblNumVec& eigval = hamDG.EigVal(); 
      eigval.Resize( hamDG.NumStateTotal() );    

      // Compute H * X
      statusOFS << std::endl << " Computing H * X for filtered orthonormal vectors : ";
      GetTime( timeSta );

      DistVec<Index3, DblNumMat, ElemPrtn>  result_mat;
      scfdg_Hamiltonian_times_eigenvectors(result_mat);

      GetTime( timeEnd );
      statusOFS << std::endl << " H * X for filtered orthonormal vectors computed . ( " << (timeEnd - timeSta ) << " s.)";

      // Raleigh-Ritz step
      if(cheby_scala_context >= 0)
      { 
        statusOFS << std::endl << std::endl << " Raleigh - Ritz step : ";

        // Convert HX to ScaLAPACK format
        dgdft::scalapack::ScaLAPACKMatrix<Real>  cheby_scala_HX;
        dgdft::scalapack::Descriptor cheby_HX_desc( hamDG.NumBasisTotal(), hamDG.NumStateTotal(), 
            scaBlockSize_, scaBlockSize_, 
            0, 0, 
            cheby_scala_context);

        statusOFS << std::endl << " Distributed vector to ScaLAPACK format conversion ... ";
        GetTime( timeSta );

        scfdg_Cheby_convert_eigvec_distmat_to_ScaLAPACK(result_mat,
            domain_.colComm,
            cheby_HX_desc,
            cheby_scala_HX);

        GetTime( timeEnd );
        statusOFS << " Done. ( " << (timeEnd - timeSta ) << " s.)";

        statusOFS << std::endl << " Solving the subspace problem ... ";
        GetTime( timeSta );

        dgdft::scalapack::Descriptor cheby_XTHX_desc( hamDG.NumStateTotal(), hamDG.NumStateTotal(), 
            scaBlockSize_, scaBlockSize_, 
            0, 0, 
            cheby_scala_context);

        dgdft::scalapack::ScaLAPACKMatrix<Real>  cheby_scala_XTHX_mat;
        cheby_scala_XTHX_mat.SetDescriptor(cheby_XTHX_desc);

        GetTime( detail_timeSta);

        dgdft::scalapack::Gemm( 'T', 'N',
            hamDG.NumStateTotal(), hamDG.NumStateTotal(), hamDG.NumBasisTotal(),
            1.0,
            cheby_scala_eigvecs_X.Data(), I_ONE, I_ONE, cheby_scala_eigvecs_X.Desc().Values(), 
            cheby_scala_HX.Data(), I_ONE, I_ONE, cheby_scala_HX.Desc().Values(),
            0.0,
            cheby_scala_XTHX_mat.Data(), I_ONE, I_ONE, cheby_scala_XTHX_mat.Desc().Values(),
            cheby_scala_context);

        GetTime( detail_timeEnd);
        statusOFS << std::endl << " X^T(HX) computed in : " << (detail_timeEnd - detail_timeSta) << " s.";

        scalapack::ScaLAPACKMatrix<Real>  scaZ;

        std::vector<Real> eigen_values;

        GetTime( detail_timeSta);

        // Eigenvalue probem solution call
        scalapack::Syevd('U', cheby_scala_XTHX_mat, eigen_values, scaZ);

        // Copy the eigenvalues to the Hamiltonian object          
        for( Int i = 0; i < hamDG.NumStateTotal(); i++ )
          eigval[i] = eigen_values[i];

        GetTime( detail_timeEnd);
        statusOFS << std::endl << " Eigenvalue problem solved in : " << (detail_timeEnd - detail_timeSta) << " s.";

        // Subspace rotation step : X <- X * Q
        GetTime( detail_timeSta);

        // To save memory, copy X to HX
        blas::Copy((cheby_scala_eigvecs_X.LocalHeight() * cheby_scala_eigvecs_X.LocalWidth()), 
            cheby_scala_eigvecs_X.Data(),
            1,
            cheby_scala_HX.Data(),
            1);

        // Now perform X <- HX (=X) * Q (=scaZ)
        dgdft::scalapack::Gemm( 'N', 'N',
            hamDG.NumBasisTotal(), hamDG.NumStateTotal(), hamDG.NumStateTotal(),
            1.0,
            cheby_scala_HX.Data(), I_ONE, I_ONE, cheby_scala_HX.Desc().Values(), 
            scaZ.Data(), I_ONE, I_ONE, scaZ.Desc().Values(),
            0.0,
            cheby_scala_eigvecs_X.Data(), I_ONE, I_ONE, cheby_scala_eigvecs_X.Desc().Values(),
            cheby_scala_context);

        GetTime( detail_timeEnd);
        statusOFS << std::endl << " Subspace rotation step completed in : " << (detail_timeEnd - detail_timeSta) << " s.";

        GetTime( timeEnd );
        statusOFS << std::endl << " All subspace problem steps completed . ( " << (timeEnd - timeSta ) << " s.)" << std::endl ;

        // Convert the eigenvectors back to distributed vector format
        statusOFS << std::endl << " ScaLAPACK to Distributed vector format conversion ... ";
        GetTime( timeSta );

        // Now convert this back to DG-distributed matrix format
        ScaMatToDistNumMat(cheby_scala_eigvecs_X ,
            elemPrtn_,
            hamDG.EigvecCoef(),
            hamDG.ElemBasisIdx(), 
            domain_.colComm, 
            hamDG.NumStateTotal() );

        GetTime( timeEnd );
        statusOFS << " Done. ( " << (timeEnd - timeSta ) << " s.)";
      }  // End of if(cheby_scala_context >= 0)

      // Communicate the final eigenvectors (to other intra-element processors)
      statusOFS << std::endl << " Communicating eigenvalues and eigenvectors ... ";
      GetTime( timeSta );

      DblNumMat &ref_mat_2 =  (hamDG.EigvecCoef().LocalMap().begin())->second;
      MPI_Bcast(ref_mat_2.Data(), (ref_mat_2.m() * ref_mat_2.n()), MPI_DOUBLE, 0, domain_.rowComm);

      // Communicate the final eigenvalues (to other intra-element processors)
      MPI_Bcast(eigval.Data(), ref_mat_2.n(), MPI_DOUBLE, 0,  domain_.rowComm); // Eigen-values

      GetTime( timeEnd );
      statusOFS << " Done. ( " << (timeEnd - timeSta ) << " s.)";

      // Reset the filtering bounds using results of the Raleigh-Ritz step    
      b_low = eigval(ref_mat_2.n() - 1);
      a_L = eigval(0);

      MPI_Barrier(domain_.rowComm);
      MPI_Barrier(domain_.colComm);
      MPI_Barrier(domain_.comm);

      // Clean up BLACS
      if(cheby_scala_context >= 0) {
        dgdft::scalapack::Cblacs_gridexit( cheby_scala_context );
      }

    } // End of if(SCFDG_Cheby_use_ScaLAPACK_ == 0) ... else 

    statusOFS << std::endl << " ------------------------------- ";
    GetTime( cheby_timeEnd );

    //statusOFS << std::endl << " Eigenvalues via Chebyshev filtering : " << std::endl;
    //statusOFS << eig_vals_Raleigh_Ritz << std::endl;

    statusOFS << std::endl << " This Chebyshev cycle took a total of " << (cheby_timeEnd - cheby_timeSta ) << " s.";
    statusOFS << std::endl << " ------------------------------- " << std::endl;

  } // End of loop over inner iteration repeats
}

void 
  SCFDG::scfdg_GeneralChebyStep(Int MaxIter, 
    Int filter_order )
  {
  Int mpirank, mpisize;
  MPI_Comm_rank( domain_.comm, &mpirank );
  MPI_Comm_size( domain_.comm, &mpisize );
  Int mpirankRow;  MPI_Comm_rank(domain_.rowComm, &mpirankRow);
  Int mpisizeRow;  MPI_Comm_size(domain_.rowComm, &mpisizeRow);
  Int mpirankCol;  MPI_Comm_rank(domain_.colComm, &mpirankCol);
  Int mpisizeCol;  MPI_Comm_size(domain_.colComm, &mpisizeCol);

  Real timeSta, timeEnd;
  Real extra_timeSta, extra_timeEnd;
  Real cheby_timeSta, cheby_timeEnd;

  HamiltonianDG&  hamDG = *hamDGPtr_;

  DblNumVec& eigval = hamDG.EigVal(); 

  // Step 0: Safeguard against the eigenvector containing extra keys
  {
    std::map<Index3, DblNumMat>::iterator cleaner_itr = hamDG.EigvecCoef().LocalMap().begin();
    while (cleaner_itr != hamDG.EigvecCoef().LocalMap().end()) 
    {
      if (cleaner_itr->first != my_cheby_eig_vec_key_) 
      {
        std::map<Index3, DblNumMat>::iterator toErase = cleaner_itr;
        ++ cleaner_itr;
        hamDG.EigvecCoef().LocalMap().erase(toErase);
      } 
      else 
      {
        ++ cleaner_itr;
      }
    }
  }

  // Step 1: Obtain the upper bound and the Ritz values (for the lower bound)
  // using the Lanczos estimator

  DblNumVec Lanczos_Ritz_values;

  statusOFS << std::endl << " Estimating the spectral bounds ... "; 
  GetTime( extra_timeSta );
  const int Num_Lanczos_Steps = 6;
  double b_up = scfdg_Cheby_Upper_bound_estimator(Lanczos_Ritz_values, Num_Lanczos_Steps);  
  GetTime( extra_timeEnd );
  statusOFS << " Done. ( " << (extra_timeEnd - extra_timeSta ) << " s.)";

  // statusOFS << std::endl << "Lanczos-Ritz values : " << Lanczos_Ritz_values << std::endl ;

  double a_L = eigval[0];
  double b_low = eigval[hamDG.NumStateTotal() - 1];

  // Step 2: Main loop
  const Int Iter_Max = MaxIter;
  const Int Filter_Order = filter_order ;

  DblNumVec eig_vals_Raleigh_Ritz;

  for(Int i = 1; i <= Iter_Max; i ++)
  {
    GetTime( cheby_timeSta );
    statusOFS << std::endl << std::endl << " ------------------------------- ";
    statusOFS << std::endl << " General CheFSI step for DGDFT cycle " << i << " of " << Iter_Max << " . ";

    // Filter the eigenvectors
    statusOFS << std::endl << std::endl << " Filtering the eigenvectors ... (Filter order = " << Filter_Order << ")";
    GetTime( timeSta );
    scfdg_Chebyshev_filter_scaled(Filter_Order, b_low, b_up, a_L);
    GetTime( timeEnd );
    statusOFS << std::endl << " Filtering completed. ( " << (timeEnd - timeSta ) << " s.)";


    // Subspace projected problems: Orthonormalization, Raleigh-Ritz and subspace rotation steps
    // This can be done serially or using ScaLAPACK    
    if(SCFDG_Cheby_use_ScaLAPACK_ == 0)
    {
      // Do the subspace problem serially 
      statusOFS << std::endl << std::endl << " Solving subspace problems serially ...";

      // Orthonormalize using Cholesky factorization  
      statusOFS << std::endl << " Orthonormalizing filtered vectors ... ";
      GetTime( timeSta );

      DblNumMat &local_eigvec_mat = (hamDG.EigvecCoef().LocalMap().begin())->second;
      DblNumMat square_mat;
      DblNumMat temp_square_mat;

      Int width = local_eigvec_mat.n();
      Int height_local = local_eigvec_mat.m();

      square_mat.Resize(width, width);
      temp_square_mat.Resize(width, width);

      SetValue(temp_square_mat, 0.0);

      // Compute square_mat = X^T * X for Cholesky    
      blas::Gemm( 'T', 'N', width, width, height_local, 
          1.0, local_eigvec_mat.Data(), height_local,
          local_eigvec_mat.Data(), height_local, 
          0.0, temp_square_mat.Data(), width );

      SetValue( square_mat, 0.0 );
      MPI_Allreduce( temp_square_mat.Data(), square_mat.Data(), width*width, MPI_DOUBLE, MPI_SUM, domain_.colComm );

      // In the following, reduction happens on colComm but the result is broadcast to everyone
      // This can probably be band-parallelized

      // Make the Cholesky factorization call on proc 0
      if ( mpirank == 0) {
        lapack::Potrf( 'U', width, square_mat.Data(), width );
      }
      // Send the Cholesky factor to every process
      MPI_Bcast(square_mat.Data(), width*width, MPI_DOUBLE, 0, domain_.comm);

      // Do a solve with the Cholesky factor : Band parallelization ??
      // X = X * U^{-1} is orthogonal, where U is the Cholesky factor
      blas::Trsm( 'R', 'U', 'N', 'N', height_local, width, 1.0, square_mat.Data(), width, 
          local_eigvec_mat.Data(), height_local );

      GetTime( timeEnd );
      statusOFS << std::endl << " Orthonormalization completed ( " << (timeEnd - timeSta ) << " s.)";

      // Raleigh-Ritz step: This part is non-scalable and needs to be fixed
      // !!!!!!!!!!!!!!!!!!!!!!!!!!!!
      statusOFS << std::endl << std::endl << " Raleigh-Ritz step ... ";
      GetTime( timeSta );

      // Compute H * X
      DistVec<Index3, DblNumMat, ElemPrtn>  result_mat;
      scfdg_Hamiltonian_times_eigenvectors(result_mat);
      DblNumMat &local_result_mat = (result_mat.LocalMap().begin())->second;

      SetValue(temp_square_mat, 0.0);

      // Compute square_mat = X^T * HX 
      blas::Gemm( 'T', 'N', width, width, height_local, 
          1.0, local_eigvec_mat.Data(), height_local,
          local_result_mat.Data(), height_local, 
          0.0, temp_square_mat.Data(), width );

      SetValue( square_mat, 0.0 );
      MPI_Allreduce( temp_square_mat.Data(), square_mat.Data(), width*width, MPI_DOUBLE, MPI_SUM, domain_.colComm );


      eig_vals_Raleigh_Ritz.Resize(width);
      SetValue(eig_vals_Raleigh_Ritz, 0.0);

      if ( mpirank == 0 ) {
        lapack::Syevd( 'V', 'U', width, square_mat.Data(), width, eig_vals_Raleigh_Ritz.Data() );

      }
      // Send the results to every process
      MPI_Bcast(square_mat.Data(), width*width, MPI_DOUBLE, 0, domain_.comm); // Eigen-vectors
      MPI_Bcast(eig_vals_Raleigh_Ritz.Data(), width, MPI_DOUBLE, 0,  domain_.comm); // Eigen-values

      GetTime( timeEnd );
      statusOFS << std::endl << " Raleigh-Ritz step completed ( " << (timeEnd - timeSta ) << " s.)";

      // This is for use with the Complementary Subspace strategy in subsequent steps
      if(SCFDG_use_comp_subspace_ == 1)
      {
        GetTime( timeSta );

        statusOFS << std::endl << std::endl << " Copying top states for serial CS strategy ... ";  
        SCFDG_comp_subspace_start_guess_.Resize(width, SCFDG_comp_subspace_N_solve_);

        for(Int copy_iter = 0; copy_iter < SCFDG_comp_subspace_N_solve_; copy_iter ++)
        {

          blas::Copy( width, square_mat.VecData(width - 1 - copy_iter), 1, 
              SCFDG_comp_subspace_start_guess_.VecData(copy_iter), 1 );

          // lapack::Lacpy( 'A', width, 1, square_mat.VecData(width - 1 - copy_iter), width, 
          //      SCFDG_comp_subspace_start_guess_.VecData(copy_iter), width );
        }

        GetTime( timeEnd );
        statusOFS << " Done. ( " << (timeEnd - timeSta ) << " s.)";
      }

      // Subspace rotation step X <- X * Q: This part is non-scalable and needs to be fixed
      // !!!!!!!!!!!!!!!!!!!!!!!!!!!!
      statusOFS << std::endl << std::endl << " Subspace rotation step ... ";
      GetTime( timeSta );

      // ~~ So copy X to HX 
      lapack::Lacpy( 'A', height_local, width, local_eigvec_mat.Data(),  height_local, local_result_mat.Data(), height_local );

      // ~~ Gemm: X <-- HX (= X) * Q
      blas::Gemm( 'N', 'N', height_local, width, width, 1.0, local_result_mat.Data(),
          height_local, square_mat.Data(), width, 0.0, local_eigvec_mat.Data(), height_local );    

      GetTime( timeEnd );
      statusOFS << std::endl << " Subspace rotation step completed ( " << (timeEnd - timeSta ) << " s.)";

      // Reset the filtering bounds using results of the Raleigh-Ritz step
      b_low = eig_vals_Raleigh_Ritz(width - 1);
      a_L = eig_vals_Raleigh_Ritz(0);

      // Fill up the eigen-values
      DblNumVec& eigval = hamDG.EigVal(); 
      eigval.Resize( hamDG.NumStateTotal() );    

      for(Int i = 0; i < hamDG.NumStateTotal(); i ++)
        eigval[i] =  eig_vals_Raleigh_Ritz[i];
    } // End of if(SCFDG_Cheby_use_ScaLAPACK_ == 0)
    else
    {
      // Do the subspace problems using ScaLAPACK
      statusOFS << std::endl << std::endl << " Solving subspace problems using ScaLAPACK ...";

      statusOFS << std::endl << " Setting up BLACS and ScaLAPACK Process Grid ...";
      GetTime( timeSta );

      double detail_timeSta, detail_timeEnd;

      // Basic ScaLAPACK setup steps
      //Number of ScaLAPACK processors equal to number of DG elements
      const int num_cheby_scala_procs = mpisizeCol; 

      // Figure out the process grid dimensions
      int temp_factor = int(sqrt(double(num_cheby_scala_procs)));
      while(num_cheby_scala_procs % temp_factor != 0 )
        ++temp_factor;

      // temp_factor now contains the process grid height
      const int cheby_scala_num_rows = temp_factor;      
      const int cheby_scala_num_cols = num_cheby_scala_procs / temp_factor;

      // Set up the ScaLAPACK context
      IntNumVec cheby_scala_pmap(num_cheby_scala_procs);

      // Use the first processor from every DG-element 
      for ( Int pmap_iter = 0; pmap_iter < num_cheby_scala_procs; pmap_iter++ )
        cheby_scala_pmap[pmap_iter] = pmap_iter * mpisizeRow; 

      // Set up BLACS for subsequent ScaLAPACK operations
      Int cheby_scala_context = -2;
      dgdft::scalapack::Cblacs_get( 0, 0, &cheby_scala_context );
      dgdft::scalapack::Cblacs_gridmap(&cheby_scala_context, &cheby_scala_pmap[0], cheby_scala_num_rows, cheby_scala_num_rows, cheby_scala_num_cols);

      // Figure out my ScaLAPACK information
      int dummy_np_row, dummy_np_col;
      int my_cheby_scala_proc_row, my_cheby_scala_proc_col;

      dgdft::scalapack::Cblacs_gridinfo(cheby_scala_context, &dummy_np_row, &dummy_np_col, &my_cheby_scala_proc_row, &my_cheby_scala_proc_col);

      GetTime( timeEnd );
      statusOFS << " Done. ( " << (timeEnd - timeSta ) << " s.)";

      statusOFS << std::endl << " ScaLAPACK will use " << num_cheby_scala_procs << " processes.";
      statusOFS << std::endl << " ScaLAPACK process grid = " << cheby_scala_num_rows << " * " << cheby_scala_num_cols << " ."  << std::endl;

      // Eigenvcetors in ScaLAPACK format : this will be used multiple times
      dgdft::scalapack::ScaLAPACKMatrix<Real>  cheby_scala_eigvecs_X; // Declared here for scope, empty constructor invoked

      if(cheby_scala_context >= 0)
      { 
        // Now setup the ScaLAPACK matrix
        statusOFS << std::endl << " Orthonormalization step : ";
        statusOFS << std::endl << " Distributed vector to ScaLAPACK format conversion ... ";
        GetTime( timeSta );

        // The dimensions should be  hamDG.NumBasisTotal() * hamDG.NumStateTotal()
        // But this is not verified here as far as the distributed vector is concerned
        dgdft::scalapack::Descriptor cheby_eigvec_desc( hamDG.NumBasisTotal(), hamDG.NumStateTotal(), 
            scaBlockSize_, scaBlockSize_, 
            0, 0, 
            cheby_scala_context);

        // Make the conversion call         
        scfdg_Cheby_convert_eigvec_distmat_to_ScaLAPACK(hamDG.EigvecCoef(),
            domain_.colComm,
            cheby_eigvec_desc,
            cheby_scala_eigvecs_X);
        //         //Older version of conversion call
        //         // Store the important ScaLAPACK information
        //         std::vector<int> my_cheby_scala_info;
        //         my_cheby_scala_info.resize(4,0);
        //         my_cheby_scala_info[0] = cheby_scala_num_rows;
        //         my_cheby_scala_info[1] = cheby_scala_num_cols;
        //         my_cheby_scala_info[2] = my_cheby_scala_proc_row;
        //         my_cheby_scala_info[3] = my_cheby_scala_proc_col;
        //       

        //         scfdg_Cheby_convert_eigvec_distmat_to_ScaLAPACK_old(hamDG.EigvecCoef(),
        //                                     my_cheby_scala_info,
        //                                     cheby_eigvec_desc,
        //                                     cheby_scala_eigvecs_X);
        //         
        GetTime( timeEnd );
        statusOFS << " Done. ( " << (timeEnd - timeSta ) << " s.)";

        statusOFS << std::endl << " Orthonormalizing filtered vectors ... ";
        GetTime( timeSta );

        GetTime( detail_timeSta);

        // Compute C = X^T * X
        dgdft::scalapack::Descriptor cheby_chol_desc( hamDG.NumStateTotal(), hamDG.NumStateTotal(), 
            scaBlockSize_, scaBlockSize_, 
            0, 0, 
            cheby_scala_context);

        dgdft::scalapack::ScaLAPACKMatrix<Real>  cheby_scala_chol_mat;
        cheby_scala_chol_mat.SetDescriptor(cheby_chol_desc);

        dgdft::scalapack::Gemm( 'T', 'N',
            hamDG.NumStateTotal(), hamDG.NumStateTotal(), hamDG.NumBasisTotal(),
            1.0,
            cheby_scala_eigvecs_X.Data(), I_ONE, I_ONE, cheby_scala_eigvecs_X.Desc().Values(), 
            cheby_scala_eigvecs_X.Data(), I_ONE, I_ONE, cheby_scala_eigvecs_X.Desc().Values(),
            0.0,
            cheby_scala_chol_mat.Data(), I_ONE, I_ONE, cheby_scala_chol_mat.Desc().Values(),
            cheby_scala_context);

        GetTime( detail_timeEnd);
        statusOFS << std::endl << " Overlap matrix computed in : " << (detail_timeEnd - detail_timeSta) << " s.";

        GetTime( detail_timeSta);
        // Compute V = Chol(C)
        dgdft::scalapack::Potrf( 'U', cheby_scala_chol_mat);

        GetTime( detail_timeEnd);
        statusOFS << std::endl << " Cholesky factorization computed in : " << (detail_timeEnd - detail_timeSta) << " s.";

        GetTime( detail_timeSta);
        // Compute  X = X * V^{-1}
        dgdft::scalapack::Trsm( 'R', 'U', 'N', 'N', 1.0,
            cheby_scala_chol_mat, 
            cheby_scala_eigvecs_X );

        GetTime( detail_timeEnd);
        statusOFS << std::endl << " TRSM computed in : " << (detail_timeEnd - detail_timeSta) << " s.";

        GetTime( timeEnd );
        statusOFS << std::endl << " Orthonormalization steps completed ( " << (timeEnd - timeSta ) << " s.)" << std::endl;

        statusOFS << std::endl << " ScaLAPACK to Distributed vector format conversion ... ";
        GetTime( timeSta );

        // Now convert this back to DG-distributed matrix format
        ScaMatToDistNumMat(cheby_scala_eigvecs_X ,
            elemPrtn_,
            hamDG.EigvecCoef(),
            hamDG.ElemBasisIdx(), 
            domain_.colComm, 
            hamDG.NumStateTotal() );

        GetTime( timeEnd );
        statusOFS << " Done. ( " << (timeEnd - timeSta ) << " s.)";

      } // End of  if(cheby_scala_context >= 0)
      else
      {
        statusOFS << std::endl << " Waiting for ScaLAPACK solution of subspace problems ...";
      }    

      // Communicate the orthonormalized eigenvectors (to other intra-element processors)
      statusOFS << std::endl << " Communicating orthonormalized filtered vectors ... ";
      GetTime( timeSta );

      DblNumMat &ref_mat_1 =  (hamDG.EigvecCoef().LocalMap().begin())->second;
      MPI_Bcast(ref_mat_1.Data(), (ref_mat_1.m() * ref_mat_1.n()), MPI_DOUBLE, 0, domain_.rowComm);

      GetTime( timeEnd );
      statusOFS << " Done. ( " << (timeEnd - timeSta ) << " s.)" << std::endl;

      // Set up space for eigenvalues in the Hamiltonian object for results of Raleigh-Ritz step
      DblNumVec& eigval = hamDG.EigVal(); 
      eigval.Resize( hamDG.NumStateTotal() );    

      // Compute H * X
      statusOFS << std::endl << " Computing H * X for filtered orthonormal vectors : ";
      GetTime( timeSta );

      DistVec<Index3, DblNumMat, ElemPrtn>  result_mat;
      scfdg_Hamiltonian_times_eigenvectors(result_mat);

      GetTime( timeEnd );
      statusOFS << std::endl << " H * X for filtered orthonormal vectors computed . ( " << (timeEnd - timeSta ) << " s.)";

      // Set up a single process ScaLAPACK matrix for use with parallel CS strategy          
      // This is for copying the relevant portion of scaZ                        
      Int single_proc_context = -1;
      scalapack::Descriptor single_proc_desc;
      scalapack::ScaLAPACKMatrix<Real>  single_proc_scala_mat;

      if(SCFDG_use_comp_subspace_ == 1 && SCFDG_comp_subspace_N_solve_ != 0)
      {  
        // Reserve the serial space : this will actually contain the vectors in the reversed order
        SCFDG_comp_subspace_start_guess_.Resize( hamDG.NumStateTotal(), SCFDG_comp_subspace_N_solve_);

        Int single_proc_pmap[1];   
        single_proc_pmap[0] = 0; // Just using proc. 0 for the job.

        // Set up BLACS for for the single proc context

        dgdft::scalapack::Cblacs_get( 0, 0, &single_proc_context );
        dgdft::scalapack::Cblacs_gridmap(&single_proc_context, &single_proc_pmap[0], 1, 1, 1);

        if( single_proc_context >= 0)
        {
          // For safety, make sure this is MPI Rank zero : throw an error otherwise
          // Fix this in the future ?
          if(mpirank != 0)
          {
            statusOFS << std::endl << std::endl << "  Error !! BLACS rank 0 does not match MPI rank 0 "
              << " Aborting ... " << std::endl << std::endl;
            MPI_Abort(domain_.comm, 0);
          }

          single_proc_desc.Init( hamDG.NumStateTotal(), SCFDG_comp_subspace_N_solve_,
              scaBlockSize_, scaBlockSize_, 
              0, 0,  single_proc_context );              

          single_proc_scala_mat.SetDescriptor( single_proc_desc );

        }
      }

      // Raleigh-Ritz step
      if(cheby_scala_context >= 0)
      { 
        statusOFS << std::endl << std::endl << " Raleigh - Ritz step : ";

        // Convert HX to ScaLAPACK format
        dgdft::scalapack::ScaLAPACKMatrix<Real>  cheby_scala_HX;
        dgdft::scalapack::Descriptor cheby_HX_desc( hamDG.NumBasisTotal(), hamDG.NumStateTotal(), 
            scaBlockSize_, scaBlockSize_, 
            0, 0, 
            cheby_scala_context);

        statusOFS << std::endl << " Distributed vector to ScaLAPACK format conversion ... ";
        GetTime( timeSta );

        // Make the conversion call                 
        scfdg_Cheby_convert_eigvec_distmat_to_ScaLAPACK(result_mat,
            domain_.colComm,
            cheby_HX_desc,
            cheby_scala_HX);

        GetTime( timeEnd );
        statusOFS << " Done. ( " << (timeEnd - timeSta ) << " s.)";

        statusOFS << std::endl << " Solving the subspace problem ... ";
        GetTime( timeSta );

        dgdft::scalapack::Descriptor cheby_XTHX_desc( hamDG.NumStateTotal(), hamDG.NumStateTotal(), 
            scaBlockSize_, scaBlockSize_, 
            0, 0, 
            cheby_scala_context);

        dgdft::scalapack::ScaLAPACKMatrix<Real>  cheby_scala_XTHX_mat;
        cheby_scala_XTHX_mat.SetDescriptor(cheby_XTHX_desc);

        GetTime( detail_timeSta);

        dgdft::scalapack::Gemm( 'T', 'N',
            hamDG.NumStateTotal(), hamDG.NumStateTotal(), hamDG.NumBasisTotal(),
            1.0,
            cheby_scala_eigvecs_X.Data(), I_ONE, I_ONE, cheby_scala_eigvecs_X.Desc().Values(), 
            cheby_scala_HX.Data(), I_ONE, I_ONE, cheby_scala_HX.Desc().Values(),
            0.0,
            cheby_scala_XTHX_mat.Data(), I_ONE, I_ONE, cheby_scala_XTHX_mat.Desc().Values(),
            cheby_scala_context);

        GetTime( detail_timeEnd);
        statusOFS << std::endl << " X^T(HX) computed in : " << (detail_timeEnd - detail_timeSta) << " s.";

        scalapack::ScaLAPACKMatrix<Real>  scaZ;

        std::vector<Real> eigen_values;

        GetTime( detail_timeSta);

        // Eigenvalue probem solution call
        scalapack::Syevd('U', cheby_scala_XTHX_mat, eigen_values, scaZ);

        // Copy the eigenvalues to the Hamiltonian object          
        for( Int i = 0; i < hamDG.NumStateTotal(); i++ )
          eigval[i] = eigen_values[i];

        GetTime( detail_timeEnd);
        statusOFS << std::endl << " Eigenvalue problem solved in : " << (detail_timeEnd - detail_timeSta) << " s.";

        // This is for use with the Complementary Subspace strategy in subsequent steps
        if(SCFDG_use_comp_subspace_ == 1 && SCFDG_comp_subspace_N_solve_ != 0)
        {
          GetTime( detail_timeSta);
          statusOFS << std::endl << std::endl << " Distributing and copying top states for parallel CS strategy ... ";

          const Int M_copy_ = hamDG.NumStateTotal();
          const Int N_copy_ = SCFDG_comp_subspace_N_solve_;
          const Int copy_src_col = M_copy_ - N_copy_ + 1;

          // Note that this is being called inside cheby_scala_context >= 0
          SCALAPACK(pdgemr2d)(&M_copy_, &N_copy_, 
              scaZ.Data(), &I_ONE, &copy_src_col,
              scaZ.Desc().Values(), 
              single_proc_scala_mat.Data(), &I_ONE, &I_ONE, 
              single_proc_scala_mat.Desc().Values(), 
              &cheby_scala_context);    

          // Copy the data from single_proc_scala_mat to SCFDG_comp_subspace_start_guess_
          if( single_proc_context >= 0)
          {
            double *src_ptr, *dest_ptr; 

            //                 statusOFS << std::endl << std::endl 
            //                            << " Ht = " << single_proc_scala_mat.Height()
            //                            << " Width = " << single_proc_scala_mat.Width()
            //                            << " loc. Ht = " << single_proc_scala_mat.LocalHeight()
            //                            << " loc. Width = " << single_proc_scala_mat.LocalWidth()
            //                            << " loc. LD = " << single_proc_scala_mat.LocalLDim();           
            //                 statusOFS << std::endl << std::endl;                        
            //                
            // Do this in the reverse order      
            for(Int copy_iter = 0; copy_iter < SCFDG_comp_subspace_N_solve_; copy_iter ++)
            {
              src_ptr = single_proc_scala_mat.Data() + (SCFDG_comp_subspace_N_solve_ - copy_iter - 1) * single_proc_scala_mat.LocalLDim();
              dest_ptr =  SCFDG_comp_subspace_start_guess_.VecData(copy_iter);

              blas::Copy( M_copy_, src_ptr, 1, dest_ptr, 1 );                          
            }
          }

          GetTime( detail_timeEnd);
          statusOFS << "Done. (" << (detail_timeEnd - detail_timeSta) << " s.)";

        } // end of if(SCFDG_use_comp_subspace_ == 1)

        // Subspace rotation step : X <- X * Q        
        statusOFS << std::endl << std::endl << " Subspace rotation step ...  ";
        GetTime( detail_timeSta);

        // To save memory, copy X to HX
        blas::Copy((cheby_scala_eigvecs_X.LocalHeight() * cheby_scala_eigvecs_X.LocalWidth()), 
            cheby_scala_eigvecs_X.Data(),
            1,
            cheby_scala_HX.Data(),
            1);

        // Now perform X <- HX (=X) * Q (=scaZ)
        dgdft::scalapack::Gemm( 'N', 'N',
            hamDG.NumBasisTotal(), hamDG.NumStateTotal(), hamDG.NumStateTotal(),
            1.0,
            cheby_scala_HX.Data(), I_ONE, I_ONE, cheby_scala_HX.Desc().Values(), 
            scaZ.Data(), I_ONE, I_ONE, scaZ.Desc().Values(),
            0.0,
            cheby_scala_eigvecs_X.Data(), I_ONE, I_ONE, cheby_scala_eigvecs_X.Desc().Values(),
            cheby_scala_context);

        GetTime( detail_timeEnd);
        statusOFS << " Done. (" << (detail_timeEnd - detail_timeSta) << " s.)";

        GetTime( timeEnd );
        statusOFS << std::endl << " All subspace problem steps completed . ( " << (timeEnd - timeSta ) << " s.)" << std::endl ;

        // Convert the eigenvectors back to distributed vector format
        statusOFS << std::endl << " ScaLAPACK to Distributed vector format conversion ... ";
        GetTime( timeSta );

        // Now convert this back to DG-distributed matrix format
        ScaMatToDistNumMat(cheby_scala_eigvecs_X ,
            elemPrtn_,
            hamDG.EigvecCoef(),
            hamDG.ElemBasisIdx(), 
            domain_.colComm, 
            hamDG.NumStateTotal() );

        GetTime( timeEnd );
        statusOFS << " Done. ( " << (timeEnd - timeSta ) << " s.)";
      } // End of if(cheby_scala_context >= 0)
      else
      {
        statusOFS << std::endl << std::endl << " Waiting for ScaLAPACK solution of subspace problems ...";
      }

      // Communicate the final eigenvectors (to other intra-element processors)
      statusOFS << std::endl << " Communicating eigenvalues and eigenvectors ... ";
      GetTime( timeSta );

      DblNumMat &ref_mat_2 =  (hamDG.EigvecCoef().LocalMap().begin())->second;
      MPI_Bcast(ref_mat_2.Data(), (ref_mat_2.m() * ref_mat_2.n()), MPI_DOUBLE, 0, domain_.rowComm);

      // Communicate the final eigenvalues (to other intra-element processors)
      MPI_Bcast(eigval.Data(), ref_mat_2.n(), MPI_DOUBLE, 0,  domain_.rowComm); // Eigen-values

      GetTime( timeEnd );
      statusOFS << " Done. ( " << (timeEnd - timeSta ) << " s.)";

      // Broadcast the guess vectors for the CS strategy from proc 0
      if(SCFDG_use_comp_subspace_ == 1 && SCFDG_comp_subspace_N_solve_ != 0)
      {
        statusOFS << std::endl << std::endl << " Broadcasting guess vectors for parallel CS strategy ... ";
        GetTime( timeSta );

        MPI_Bcast(SCFDG_comp_subspace_start_guess_.Data(),hamDG.NumStateTotal() * SCFDG_comp_subspace_N_solve_, 
            MPI_DOUBLE, 0,  domain_.comm); 

        GetTime( timeEnd );
        statusOFS << " Done. ( " << (timeEnd - timeSta ) << " s.)";
      }    

      // Reset the filtering bounds using results of the Raleigh-Ritz step    
      b_low = eigval(ref_mat_2.n() - 1);
      a_L = eigval(0);

      MPI_Barrier(domain_.rowComm);
      MPI_Barrier(domain_.colComm);
      MPI_Barrier(domain_.comm);

      // Clean up BLACS
      if(cheby_scala_context >= 0) 
      {
        dgdft::scalapack::Cblacs_gridexit( cheby_scala_context );
      }

      if( single_proc_context >= 0)
      {
        dgdft::scalapack::Cblacs_gridexit( single_proc_context );
      }
    } // End of if(SCFDG_Cheby_use_ScaLAPACK_ == 0) ... else 

    // Display the eigenvalues 
    statusOFS << std::endl << " ------------------------------- ";
    GetTime( cheby_timeEnd );

    //statusOFS << std::endl << " Eigenvalues via Chebyshev filtering : " << std::endl;
    //statusOFS << eig_vals_Raleigh_Ritz << std::endl;

    statusOFS << std::endl << " This Chebyshev cycle took a total of " << (cheby_timeEnd - cheby_timeSta ) << " s.";
    statusOFS << std::endl << " ------------------------------- " << std::endl;

  } // End of loop over inner iteration repeats
}

// **###**    
/// @brief Routines related to Chebyshev polynomial filtered 
/// complementary subspace iteration strategy in DGDFT
void SCFDG::scfdg_complementary_subspace_serial(Int filter_order )
{
  statusOFS << std::endl << " ----------------------------------------------------------" ; 
  statusOFS << std::endl << " Complementary Subspace Strategy (Serial Subspace version): " << std::endl ; 

  Int mpirank, mpisize;
  MPI_Comm_rank( domain_.comm, &mpirank );
  MPI_Comm_size( domain_.comm, &mpisize );
  Int mpirankRow;  MPI_Comm_rank(domain_.rowComm, &mpirankRow);
  Int mpisizeRow;  MPI_Comm_size(domain_.rowComm, &mpisizeRow);
  Int mpirankCol;  MPI_Comm_rank(domain_.colComm, &mpirankCol);
  Int mpisizeCol;  MPI_Comm_size(domain_.colComm, &mpisizeCol);

  Real timeSta, timeEnd;
  Real extra_timeSta, extra_timeEnd;
  Real cheby_timeSta, cheby_timeEnd;

  HamiltonianDG&  hamDG = *hamDGPtr_;

  DblNumVec& eigval = hamDG.EigVal(); 

  // Step 0: Safeguard against the eigenvector containing extra keys
  {
    std::map<Index3, DblNumMat>::iterator cleaner_itr = hamDG.EigvecCoef().LocalMap().begin();
    while (cleaner_itr != hamDG.EigvecCoef().LocalMap().end()) 
    {
      if (cleaner_itr->first != my_cheby_eig_vec_key_) 
      {
        std::map<Index3, DblNumMat>::iterator toErase = cleaner_itr;
        ++ cleaner_itr;
        hamDG.EigvecCoef().LocalMap().erase(toErase);
      } 
      else 
      {
        ++ cleaner_itr;
      }
    }
  }

  // Step 1: Obtain the upper bound and the Ritz values (for the lower bound)
  // using the Lanczos estimator

  DblNumVec Lanczos_Ritz_values;

  statusOFS << std::endl << " Estimating the spectral bounds ... "; 
  GetTime( extra_timeSta );
  const int Num_Lanczos_Steps = 6;
  double b_up = scfdg_Cheby_Upper_bound_estimator(Lanczos_Ritz_values, Num_Lanczos_Steps);  
  GetTime( extra_timeEnd );
  statusOFS << " Done. ( " << (extra_timeEnd - extra_timeSta ) << " s.)";

  double a_L;
  double b_low;

  if(SCFDG_comp_subspace_engaged_ == 1)
  {
    a_L = SCFDG_comp_subspace_saved_a_L_;
    b_low = SCFDG_comp_subspace_top_eigvals_[0];
  }
  else
  {
    // First time we are doing this : use earlier ScaLAPACK results 
    a_L = eigval[0];
    SCFDG_comp_subspace_saved_a_L_ = a_L;

    b_low = eigval[hamDG.NumStateTotal() - 1];
  }

  // Step 2: Perform filtering
  const Int Filter_Order = filter_order ;

  GetTime( cheby_timeSta );
  // Filter the subspace
  statusOFS << std::endl << std::endl << " Filtering the subspace ... (Filter order = " << Filter_Order << ")";
  GetTime( timeSta );
  scfdg_Chebyshev_filter_scaled(Filter_Order, b_low, b_up, a_L);
  GetTime( timeEnd );
  statusOFS << std::endl << " Filtering completed. ( " << (timeEnd - timeSta ) << " s.)";

  // Step 3: Perform subspace projected problems
  // Subspace problems serially done here

  statusOFS << std::endl << std::endl << " Solving subspace problems serially ...";
  
  // Some diagnostic info
  statusOFS << std::endl;
  statusOFS << std::endl << " Note : Outer subspace problem dimension = " << hamDG.NumStateTotal() << " * " << hamDG.NumStateTotal();
  statusOFS << std::endl << "        Inner subspace problem dimension = " << SCFDG_comp_subspace_N_solve_ << " * " << SCFDG_comp_subspace_N_solve_;
  statusOFS << std::endl << "        Matrix C has dimension = " << hamDG.NumStateTotal() << " * " << SCFDG_comp_subspace_N_solve_;
  statusOFS << std::endl << "        Matrix C occupies " << (double(hamDG.NumStateTotal() *  SCFDG_comp_subspace_N_solve_ * 8) / double(1048576)) << " MBs per process.";

  statusOFS << std::endl ;

  // Orthonormalize using Cholesky factorization  
  statusOFS << std::endl << " Orthonormalizing filtered vectors ... ";
  GetTime( timeSta );

  DblNumMat &local_eigvec_mat = (hamDG.EigvecCoef().LocalMap().begin())->second;
  DblNumMat square_mat;
  DblNumMat temp_square_mat;

  Int width = local_eigvec_mat.n();
  Int height_local = local_eigvec_mat.m();

  square_mat.Resize(width, width);
  temp_square_mat.Resize(width, width);

  SetValue(temp_square_mat, 0.0);

  // Compute square_mat = X^T * X for Cholesky    
  blas::Gemm( 'T', 'N', width, width, height_local, 
      1.0, local_eigvec_mat.Data(), height_local,
      local_eigvec_mat.Data(), height_local, 
      0.0, temp_square_mat.Data(), width );

  SetValue( square_mat, 0.0 );
  MPI_Allreduce( temp_square_mat.Data(), square_mat.Data(), width*width, MPI_DOUBLE, MPI_SUM, domain_.colComm );

  // In the following, reduction happens on colComm but the result is broadcast to everyone

  // Make the Cholesky factorization call on proc 0
  if ( mpirank == 0) {
    lapack::Potrf( 'U', width, square_mat.Data(), width );
  }
  // Send the Cholesky factor to every process
  MPI_Bcast(square_mat.Data(), width*width, MPI_DOUBLE, 0, domain_.comm);

  // Do a solve with the Cholesky factor : Band parallelization ??
  // X = X * U^{-1} is orthogonal, where U is the Cholesky factor
  blas::Trsm( 'R', 'U', 'N', 'N', height_local, width, 1.0, square_mat.Data(), width, 
      local_eigvec_mat.Data(), height_local );

  GetTime( timeEnd );
  statusOFS << std::endl << " Orthonormalization completed ( " << (timeEnd - timeSta ) << " s.)";

  // Alternate to Raleigh-Ritz step
  // !!!!!!!!!!!!!!!!!!!!!!!!!!!!
  statusOFS << std::endl << std::endl << " Performing alternate to Raleigh-Ritz step ... ";
  GetTime( timeSta );

  // Compute H * X
  DistVec<Index3, DblNumMat, ElemPrtn>  result_mat;
  scfdg_Hamiltonian_times_eigenvectors(result_mat);
  DblNumMat &local_result_mat = (result_mat.LocalMap().begin())->second;

  SetValue(temp_square_mat, 0.0);

  // Compute square_mat = X^T * HX 
  blas::Gemm( 'T', 'N', width, width, height_local, 
      1.0, local_eigvec_mat.Data(), height_local,
      local_result_mat.Data(), height_local, 
      0.0, temp_square_mat.Data(), width );

  SetValue( square_mat, 0.0 );
  MPI_Allreduce( temp_square_mat.Data(), square_mat.Data(), width*width, MPI_DOUBLE, MPI_SUM, domain_.colComm );

  DblNumMat temp_Hmat(square_mat);

  // Space for top few eigenpairs of projected Hamiltonian
  // Note that SCFDG_comp_subspace_start_guess_ should contain the starting guess already
  // This is either from the earlier ScaLAPACK results (when SCFDG_comp_subspace_engaged_ = 0)
  // or from the previous LOBPCG results which are copied

  DblNumMat temp_Xmat;
  temp_Xmat.Resize(width, SCFDG_comp_subspace_N_solve_);
  lapack::Lacpy( 'A', width, SCFDG_comp_subspace_N_solve_, SCFDG_comp_subspace_start_guess_.Data(), width, 
      temp_Xmat.Data(), width );
  // Space for top few eigenvalues of projected Hamiltonian
  DblNumVec temp_eig_vals_Xmat(SCFDG_comp_subspace_N_solve_);
  SetValue( temp_eig_vals_Xmat, 0.0 );

  if(Hmat_top_states_use_Cheby_ == 0)
  {  
    // Use serial LOBPCG to get the top states  
    GetTime(extra_timeSta);

    LOBPCG_Hmat_top_serial(temp_Hmat,
        temp_Xmat,
        temp_eig_vals_Xmat,
        SCFDG_comp_subspace_LOBPCG_iter_, SCFDG_comp_subspace_LOBPCG_tol_); // The tolerance should be dynamic probably

    GetTime(extra_timeEnd);

    statusOFS << std::endl << " Serial LOBPCG completed on " <<  SCFDG_comp_subspace_N_solve_ 
      << " top states ( " << (extra_timeEnd - extra_timeSta ) << " s.)";
  }
  else
  {
    // XXXXXXXXXXXXXXXXXXXXXX
    // Use CheFSI for top states here : Use -H for the matrix

    // Fix the filter bounds
    if(SCFDG_comp_subspace_engaged_ != 1)
    {
      SCFDG_comp_subspace_inner_CheFSI_a_L_ = - eigval[hamDG.NumStateTotal() - 1];

      int state_ind = hamDG.NumStateTotal() - SCFDG_comp_subspace_N_solve_;        
      SCFDG_comp_subspace_inner_CheFSI_lower_bound_ = -0.5 * (eigval[state_ind] + eigval[state_ind - 1]);

      SCFDG_comp_subspace_inner_CheFSI_upper_bound_ = find_comp_subspace_UB_serial(temp_Hmat);

      Hmat_top_states_Cheby_delta_fudge_ = 0.5 * (eigval[state_ind] - eigval[state_ind - 1]);

      statusOFS << std::endl << " Going into inner CheFSI routine for top states ... ";
      statusOFS << std::endl << "   Lower bound = -(average of eigenvalues  " << eigval[state_ind] 
        << " and " <<  eigval[state_ind - 1] << ") = " << SCFDG_comp_subspace_inner_CheFSI_lower_bound_
        << std::endl << "   Lanczos upper bound = " << SCFDG_comp_subspace_inner_CheFSI_upper_bound_ ;
    }
    else
    {
      SCFDG_comp_subspace_inner_CheFSI_lower_bound_ = - (SCFDG_comp_subspace_top_eigvals_[SCFDG_comp_subspace_N_solve_ - 1] - Hmat_top_states_Cheby_delta_fudge_);

      SCFDG_comp_subspace_inner_CheFSI_upper_bound_ = find_comp_subspace_UB_serial(temp_Hmat);

      statusOFS << std::endl << " Going into inner CheFSI routine for top states ... ";
      statusOFS << std::endl << "   Lower bound eigenvalue = " << SCFDG_comp_subspace_top_eigvals_[SCFDG_comp_subspace_N_solve_ - 1] 
        << std::endl << "   delta_fudge = " << Hmat_top_states_Cheby_delta_fudge_
        << std::endl << "   Lanczos upper bound = " << SCFDG_comp_subspace_inner_CheFSI_upper_bound_ ;
    }

    GetTime(extra_timeSta);

    CheFSI_Hmat_top_serial(temp_Hmat,
        temp_Xmat,
        temp_eig_vals_Xmat,
        Hmat_top_states_ChebyFilterOrder_,
        Hmat_top_states_ChebyCycleNum_,
        SCFDG_comp_subspace_inner_CheFSI_lower_bound_,SCFDG_comp_subspace_inner_CheFSI_upper_bound_, SCFDG_comp_subspace_inner_CheFSI_a_L_);

    GetTime(extra_timeEnd);

    statusOFS << std::endl << " Serial CheFSI completed on " <<  SCFDG_comp_subspace_N_solve_ 
      << " top states ( " << (extra_timeEnd - extra_timeSta ) << " s.)";

    //exit(1);                

  }

  // Broadcast the results from proc 0 to ensure all procs are using the same eigenstates
  MPI_Bcast(temp_Xmat.Data(), SCFDG_comp_subspace_N_solve_ * width, MPI_DOUBLE, 0, domain_.comm); // Eigenvectors
  MPI_Bcast(temp_eig_vals_Xmat.Data(), SCFDG_comp_subspace_N_solve_ , MPI_DOUBLE, 0, domain_.comm); // Eigenvalues

  // Copy back the eigenstates to the guess for the next step

  // Eigenvectors
  lapack::Lacpy( 'A', width, SCFDG_comp_subspace_N_solve_, temp_Xmat.Data(), width, 
      SCFDG_comp_subspace_start_guess_.Data(), width );

  // Eigenvalues  
  SCFDG_comp_subspace_top_eigvals_.Resize(SCFDG_comp_subspace_N_solve_);
  for(Int copy_iter = 0; copy_iter < SCFDG_comp_subspace_N_solve_; copy_iter ++)
    SCFDG_comp_subspace_top_eigvals_[copy_iter] = temp_eig_vals_Xmat[copy_iter];

  // Also update the top eigenvalues in hamDG in case we need them
  // For example, they are required if we switch back to regular CheFSI at some stage 
  Int n_top = hamDG.NumStateTotal() - 1;
  for(Int copy_iter = 0; copy_iter < SCFDG_comp_subspace_N_solve_; copy_iter ++)
    eigval[n_top - copy_iter] = SCFDG_comp_subspace_top_eigvals_[copy_iter];

  // Compute the occupations    
  SCFDG_comp_subspace_top_occupations_.Resize(SCFDG_comp_subspace_N_solve_);

  Int howmany_to_calc = (hamDGPtr_->NumOccupiedState() + SCFDG_comp_subspace_N_solve_) - hamDGPtr_->NumStateTotal(); 
  scfdg_calc_occ_rate_comp_subspc(SCFDG_comp_subspace_top_eigvals_,SCFDG_comp_subspace_top_occupations_, howmany_to_calc);


  statusOFS << std::endl << " Fermi level = " << fermi_ << std::endl;
  statusOFS << std::endl << " Top Eigenvalues and occupations (in reverse order) : ";
  
  for(int print_iter = 0; print_iter < SCFDG_comp_subspace_N_solve_; print_iter ++)
    statusOFS << std::endl <<  " " << std::setw(8) << print_iter << std::setw(20) << SCFDG_comp_subspace_top_eigvals_[print_iter] 
                            << '\t' << SCFDG_comp_subspace_top_occupations_[print_iter];

  // Form the matrix C by scaling the eigenvectors with the appropriate occupation related weights
  SCFDG_comp_subspace_matC_.Resize(width, SCFDG_comp_subspace_N_solve_);
  lapack::Lacpy( 'A', width, SCFDG_comp_subspace_N_solve_, temp_Xmat.Data(), width, 
      SCFDG_comp_subspace_matC_.Data(), width );

  double scale_fac;
  for(Int scal_iter = 0; scal_iter < SCFDG_comp_subspace_N_solve_; scal_iter ++)
  {
    scale_fac = sqrt(1.0 - SCFDG_comp_subspace_top_occupations_(scal_iter));
    blas::Scal(width, scale_fac, SCFDG_comp_subspace_matC_.VecData(scal_iter), 1);
  }

  // This calculation is done for computing the band energy later
  SCFDG_comp_subspace_trace_Hmat_ = 0.0;
  for(Int trace_calc = 0; trace_calc < width; trace_calc ++)
    SCFDG_comp_subspace_trace_Hmat_ += temp_Hmat(trace_calc, trace_calc);

  statusOFS << std::endl << std::endl << " ------------------------------- ";

}

/// @brief Routines related to Chebyshev polynomial filtered 
/// complementary subspace iteration strategy in DGDFT in parallel
void SCFDG::scfdg_complementary_subspace_parallel(Int filter_order )
{
  statusOFS << std::endl << " ----------------------------------------------------------" ; 
  statusOFS << std::endl << " Complementary Subspace Strategy (Parallel Subspace version): " << std::endl ; 

  Int mpirank, mpisize;
  MPI_Comm_rank( domain_.comm, &mpirank );
  MPI_Comm_size( domain_.comm, &mpisize );
  Int mpirankRow;  MPI_Comm_rank(domain_.rowComm, &mpirankRow);
  Int mpisizeRow;  MPI_Comm_size(domain_.rowComm, &mpisizeRow);
  Int mpirankCol;  MPI_Comm_rank(domain_.colComm, &mpirankCol);
  Int mpisizeCol;  MPI_Comm_size(domain_.colComm, &mpisizeCol);

  Real timeSta, timeEnd;
  Real extra_timeSta, extra_timeEnd;
  Real cheby_timeSta, cheby_timeEnd;

  HamiltonianDG&  hamDG = *hamDGPtr_;

  DblNumVec& eigval = hamDG.EigVal(); 

  // Step 0: Safeguard against the eigenvector containing extra keys
  {
    std::map<Index3, DblNumMat>::iterator cleaner_itr = hamDG.EigvecCoef().LocalMap().begin();
    while (cleaner_itr != hamDG.EigvecCoef().LocalMap().end()) 
    {
      if (cleaner_itr->first != my_cheby_eig_vec_key_) 
      {
        std::map<Index3, DblNumMat>::iterator toErase = cleaner_itr;
        ++ cleaner_itr;
        hamDG.EigvecCoef().LocalMap().erase(toErase);
      } 
      else 
      {
        ++ cleaner_itr;
      }
    }
  }

  // Step 1: Obtain the upper bound and the Ritz values (for the lower bound)
  // using the Lanczos estimator
  DblNumVec Lanczos_Ritz_values;
  statusOFS << std::endl << " Estimating the spectral bounds ... "; 
  GetTime( extra_timeSta );
  const int Num_Lanczos_Steps = 6;
  double b_up = scfdg_Cheby_Upper_bound_estimator(Lanczos_Ritz_values, Num_Lanczos_Steps);  
  GetTime( extra_timeEnd );
  statusOFS << " Done. ( " << (extra_timeEnd - extra_timeSta ) << " s.)";

  double a_L;
  double b_low;

  if(SCFDG_comp_subspace_engaged_ == 1)
  {
    a_L = SCFDG_comp_subspace_saved_a_L_;
    b_low = SCFDG_comp_subspace_top_eigvals_[0];
  }
  else
  {
    // First time we are doing this : use earlier ScaLAPACK results 
    a_L = eigval[0];
    SCFDG_comp_subspace_saved_a_L_ = a_L;

    b_low = eigval[hamDG.NumStateTotal() - 1];
  }

  // Step 2: Perform filtering
  const Int Filter_Order = filter_order ;

  GetTime( cheby_timeSta );
  // Filter the subspace
  statusOFS << std::endl << std::endl << " Filtering the subspace ... (Filter order = " << Filter_Order << ")";
  GetTime( timeSta );
  scfdg_Chebyshev_filter_scaled(Filter_Order, b_low, b_up, a_L);
  GetTime( timeEnd );
  statusOFS << std::endl << " Filtering completed. ( " << (timeEnd - timeSta ) << " s.)";

  // Step 3: Perform subspace projected problems
  // Subspace problems solved in parallel here

  statusOFS << std::endl << std::endl << " Solving subspace problems in parallel :" << std::endl;

  // YYYY
  // Step a : Convert to ScaLAPACK format
  // Setup BLACS / ScaLAPACK    
  statusOFS << std::endl << " Setting up BLACS Process Grids ...";
  GetTime( timeSta );

  // Set up 3 independent ScaLAPACK contexts 
  // First one is just the regular one that arises in CheFSI.
  // It involves the first row of processes (i.e., 1 processor for every DG element)

  int num_cheby_scala_procs = mpisizeCol; 

  // Figure out the process grid dimensions for the cheby_scala_context
  int temp_factor = int(sqrt(double(num_cheby_scala_procs)));
  while(num_cheby_scala_procs % temp_factor != 0 )
    ++temp_factor;

  // temp_factor now contains the process grid height
  int cheby_scala_num_rows = temp_factor;      
  int cheby_scala_num_cols = num_cheby_scala_procs / temp_factor;

  // We favor process grids which are taller instead of being wider
  if(cheby_scala_num_cols > cheby_scala_num_rows)
  {
    int exchg_temp = cheby_scala_num_cols;
    cheby_scala_num_cols = cheby_scala_num_rows;
    cheby_scala_num_rows = exchg_temp;
  } 

  // Set up the ScaLAPACK context
  IntNumVec cheby_scala_pmap(num_cheby_scala_procs);

  // Use the first processor from every DG-element 
  for ( Int pmap_iter = 0; pmap_iter < num_cheby_scala_procs; pmap_iter++ )
    cheby_scala_pmap[pmap_iter] = pmap_iter * mpisizeRow; 

  // Set up BLACS for subsequent ScaLAPACK operations
  Int cheby_scala_context = -1;
  dgdft::scalapack::Cblacs_get( 0, 0, &cheby_scala_context );
  dgdft::scalapack::Cblacs_gridmap(&cheby_scala_context, &cheby_scala_pmap[0], 
                     cheby_scala_num_rows, cheby_scala_num_rows, cheby_scala_num_cols);

  statusOFS << std::endl << " Cheby-Scala context will use " 
            << num_cheby_scala_procs << " processes.";
  statusOFS << std::endl << " Cheby-Scala process grid dim. = " 
            << cheby_scala_num_rows << " * " << cheby_scala_num_cols << " .";

  // Next one is the "bigger grid" for doing some linear algebra operations
  int bigger_grid_num_procs = num_cheby_scala_procs * SCFDG_CS_bigger_grid_dim_fac_;

  if(bigger_grid_num_procs > mpisize)
  {
    SCFDG_CS_bigger_grid_dim_fac_ = mpisize / num_cheby_scala_procs;
    bigger_grid_num_procs = mpisize;

    statusOFS << std::endl << std::endl << " Warning !! Check input parameter SCFDG_CS_bigger_grid_dim_fac .";
    statusOFS << std::endl << " Requested process grid is bigger than total no. of processes.";
    statusOFS << std::endl << " Using " << bigger_grid_num_procs << " processes instead. ";
    statusOFS << std::endl << " Calculation will now continue without throwing exception.";

    statusOFS << std::endl;      
  }

  int bigger_grid_num_rows = cheby_scala_num_rows * SCFDG_CS_bigger_grid_dim_fac_;
  int bigger_grid_num_cols = cheby_scala_num_cols;

  IntNumVec bigger_grid_pmap(bigger_grid_num_procs);
  int pmap_ctr = 0;
  for (Int pmap_iter_1 = 0; pmap_iter_1 < num_cheby_scala_procs; pmap_iter_1 ++)
  {
    for (Int pmap_iter_2 = 0; pmap_iter_2 < SCFDG_CS_bigger_grid_dim_fac_; pmap_iter_2 ++)
    {
      bigger_grid_pmap[pmap_ctr] =  pmap_iter_1 * mpisizeRow + pmap_iter_2;
      pmap_ctr ++;
    }
  }

  Int bigger_grid_context = -1;
  dgdft::scalapack::Cblacs_get( 0, 0, &bigger_grid_context );
  dgdft::scalapack::Cblacs_gridmap(&bigger_grid_context, &bigger_grid_pmap[0], 
                      bigger_grid_num_rows, bigger_grid_num_rows, bigger_grid_num_cols);   

  statusOFS << std::endl << " Bigger grid context will use "
            << bigger_grid_num_procs << " processes.";
  statusOFS << std::endl << " Bigger process grid dim. = "
            << bigger_grid_num_rows << " * " << bigger_grid_num_cols << " .";

  // Finally, there is the single process case
  Int single_proc_context = -1;
  Int single_proc_pmap[1];  
  single_proc_pmap[0] = 0; // Just using proc. 0 for the job.

  // Set up BLACS for for the single proc context
  dgdft::scalapack::Cblacs_get( 0, 0, &single_proc_context );
  dgdft::scalapack::Cblacs_gridmap(&single_proc_context, &single_proc_pmap[0], 1, 1, 1);

  // For safety, make sure this is MPI Rank zero : throw an error otherwise
  // Fix this in the future ?
  if(single_proc_context >= 0)
  {  
    if(mpirank != 0)
    {
      statusOFS << std::endl << std::endl << "  Error !! BLACS rank 0 does not match MPI rank 0 "
        << " Aborting ... " << std::endl << std::endl;

      MPI_Abort(domain_.comm, 0);
    }
  }

  statusOFS << std::endl << " Single process context works with process 0 .";
  statusOFS << std::endl << " Single process dimension = 1 * 1 ." << std::endl;

  GetTime( timeEnd );
  statusOFS << " BLACS setup done. ( " << (timeEnd - timeSta ) << " s.)";

  // Some diagnostic info
  statusOFS << std::endl;
  statusOFS << std::endl << " Note : On Cheby-Scala grid, scala_block_size * cheby_scala_num_rows = " 
            << (scaBlockSize_ * cheby_scala_num_rows);
  statusOFS << std::endl << "        On Cheby-Scala grid, scala_block_size * cheby_scala_num_cols = "
            << (scaBlockSize_ * cheby_scala_num_cols);
  statusOFS << std::endl << "        On bigger grid, scala_block_size * bigger_grid_num_rows = " 
            << (scaBlockSize_ * bigger_grid_num_rows);
  statusOFS << std::endl << "        On bigger grid, scala_block_size * bigger_grid_num_cols = "
            << (scaBlockSize_ * bigger_grid_num_cols);
  statusOFS << std::endl << "        Outer subspace problem dimension = " 
            << hamDG.NumStateTotal() << " * " << hamDG.NumStateTotal();
  statusOFS << std::endl << "        Inner subspace problem dimension = "
            << SCFDG_comp_subspace_N_solve_ << " * " << SCFDG_comp_subspace_N_solve_;
  statusOFS << std::endl << "        Matrix C has dimension = " 
            << hamDG.NumStateTotal() << " * " << SCFDG_comp_subspace_N_solve_;
  statusOFS << std::endl << "        Matrix C occupies " 
            << (double(hamDG.NumStateTotal() *  SCFDG_comp_subspace_N_solve_ * 8) / double(1048576))
            << " MBs per process.";
  statusOFS << std::endl ;
	
  // Step b: Orthonormalize using "bigger grid"
  dgdft::scalapack::ScaLAPACKMatrix<Real>  cheby_scala_eigvecs_X;
  dgdft::scalapack::ScaLAPACKMatrix<Real>  bigger_grid_eigvecs_X;

  scalapack::Descriptor cheby_eigvec_desc;
  scalapack::Descriptor bigger_grid_eigvec_desc;

  statusOFS << std::endl << std::endl << " Orthonormalization step : " << std::endl;

  // DG to Cheby-Scala format conversion
  if(cheby_scala_context >= 0)
  { 
    statusOFS << std::endl << " Distributed vector X to ScaLAPACK (Cheby-Scala grid) conversion ... ";
    GetTime( timeSta );

    cheby_eigvec_desc.Init( hamDG.NumBasisTotal(), hamDG.NumStateTotal(), 
        scaBlockSize_, scaBlockSize_, 
        0, 0, 
        cheby_scala_context);

    cheby_scala_eigvecs_X.SetDescriptor(cheby_eigvec_desc);

    // Make the conversion call         
    scfdg_Cheby_convert_eigvec_distmat_to_ScaLAPACK(hamDG.EigvecCoef(),
        domain_.colComm,
        cheby_eigvec_desc,
        cheby_scala_eigvecs_X);

    GetTime( timeEnd );
    statusOFS << " Done. ( " << (timeEnd - timeSta ) << " s.)";
  }

  // Cheby-Scala to Big grid format conversion

  Int M_copy_ =  hamDG.NumBasisTotal();
  Int N_copy_ =  hamDG.NumStateTotal();

  if(bigger_grid_context >= 0)
  {
    statusOFS << std::endl << " Cheby-Scala grid to bigger grid pdgemr2d for X... ";
    GetTime( timeSta );

    bigger_grid_eigvec_desc.Init( M_copy_, N_copy_, 
        scaBlockSize_, scaBlockSize_, 
        0, 0, 
        bigger_grid_context);  

    bigger_grid_eigvecs_X.SetDescriptor(bigger_grid_eigvec_desc);

    SCALAPACK(pdgemr2d)(&M_copy_, &N_copy_, 
        cheby_scala_eigvecs_X.Data(), &I_ONE, &I_ONE,
        cheby_scala_eigvecs_X.Desc().Values(), 
        bigger_grid_eigvecs_X.Data(), &I_ONE, &I_ONE, 
        bigger_grid_eigvecs_X.Desc().Values(), 
        &bigger_grid_context);      

    GetTime( timeEnd );
    statusOFS << " Done. ( " << (timeEnd - timeSta ) << " s.)" << std::endl;

  }

  // Make the ScaLAPACK calls for orthonormalization   
  GetTime( timeSta );
  if(bigger_grid_context >= 0)
  {
    GetTime( extra_timeSta );

    // Compute C = X^T * X
    dgdft::scalapack::Descriptor bigger_grid_chol_desc( hamDG.NumStateTotal(), hamDG.NumStateTotal(), 
        scaBlockSize_, scaBlockSize_, 
        0, 0, 
        bigger_grid_context);

    dgdft::scalapack::ScaLAPACKMatrix<Real>  bigger_grid_chol_mat;
    bigger_grid_chol_mat.SetDescriptor(bigger_grid_chol_desc);

    if(SCFDG_comp_subspace_syrk_ == 0)
    {
      dgdft::scalapack::Gemm( 'T', 'N',
          hamDG.NumStateTotal(), hamDG.NumStateTotal(), hamDG.NumBasisTotal(),
          1.0,
          bigger_grid_eigvecs_X.Data(), I_ONE, I_ONE, bigger_grid_eigvecs_X.Desc().Values(), 
          bigger_grid_eigvecs_X.Data(), I_ONE, I_ONE, bigger_grid_eigvecs_X.Desc().Values(),
          0.0,
          bigger_grid_chol_mat.Data(), I_ONE, I_ONE, bigger_grid_chol_mat.Desc().Values(),
          bigger_grid_context);
    }
    else
    {  
      dgdft::scalapack::Syrk('U', 'T',
          hamDG.NumStateTotal(), hamDG.NumBasisTotal(),
          1.0, bigger_grid_eigvecs_X.Data(),
          I_ONE, I_ONE,bigger_grid_eigvecs_X.Desc().Values(),
          0.0, bigger_grid_chol_mat.Data(),
          I_ONE, I_ONE,bigger_grid_chol_mat.Desc().Values());
    }

    GetTime( extra_timeEnd);            
    if(SCFDG_comp_subspace_syrk_ == 0)
      statusOFS << std::endl << " Overlap matrix computed using GEMM in : " << (extra_timeEnd - extra_timeSta) << " s.";
    else
      statusOFS << std::endl << " Overlap matrix computed using SYRK in : " << (extra_timeEnd - extra_timeSta) << " s.";

    GetTime( extra_timeSta);

    // Compute V = Chol(C)
    dgdft::scalapack::Potrf( 'U', bigger_grid_chol_mat);

    GetTime( extra_timeEnd);
    statusOFS << std::endl << " Cholesky factorization computed in : " << (extra_timeEnd - extra_timeSta) << " s.";

    GetTime( extra_timeSta);

    // Compute  X = X * V^{-1}
    dgdft::scalapack::Trsm( 'R', 'U', 'N', 'N', 1.0,
        bigger_grid_chol_mat, 
        bigger_grid_eigvecs_X );

    GetTime( extra_timeEnd);
    statusOFS << std::endl << " TRSM computed in : " << (extra_timeEnd - extra_timeSta) << " s." << std::endl;    
  }

  GetTime( timeEnd );
  statusOFS << " Total time for ScaLAPACK calls during Orthonormalization  = " << (timeEnd - timeSta ) << " s." << std::endl;

  if(bigger_grid_context >= 0)
  {
    // Convert back to Cheby-Scala grid
    statusOFS << std::endl << " Bigger grid to Cheby-Scala grid pdgemr2d ... ";
    GetTime( timeSta );

    SCALAPACK(pdgemr2d)(&M_copy_, &N_copy_, 
        bigger_grid_eigvecs_X.Data(), &I_ONE, &I_ONE,
        bigger_grid_eigvecs_X.Desc().Values(), 
        cheby_scala_eigvecs_X.Data(), &I_ONE, &I_ONE, 
        cheby_scala_eigvecs_X.Desc().Values(), 
        &bigger_grid_context);  

    GetTime( timeEnd );
    statusOFS << " Done. ( " << (timeEnd - timeSta ) << " s.)";
  }    

  if(cheby_scala_context >= 0)
  {
    statusOFS << std::endl << " ScaLAPACK (Cheby-Scala grid) to Distributed vector conversion ... ";
    GetTime( timeSta );

    // Convert to DG-distributed matrix format
    ScaMatToDistNumMat(cheby_scala_eigvecs_X ,
        elemPrtn_,
        hamDG.EigvecCoef(),
        hamDG.ElemBasisIdx(), 
        domain_.colComm, 
        hamDG.NumStateTotal() );

    GetTime( timeEnd );
    statusOFS << " Done. ( " << (timeEnd - timeSta ) << " s.)" << std::endl;     

  }

  // Communicate the orthonormalized eigenvectors (to other intra-element processors)
  statusOFS << std::endl << " Communicating orthonormalized filtered vectors ... ";
  GetTime( timeSta );

  DblNumMat &ref_mat_1 =  (hamDG.EigvecCoef().LocalMap().begin())->second;
  MPI_Bcast(ref_mat_1.Data(), (ref_mat_1.m() * ref_mat_1.n()), MPI_DOUBLE, 0, domain_.rowComm);

  GetTime( timeEnd );
  statusOFS << " Done. ( " << (timeEnd - timeSta ) << " s.)" << std::endl;

  // Step c : Perform alternate to Raleigh-Ritz step
  // Compute H * X
  statusOFS << std::endl << " Computing H * X for filtered orthonormal vectors : ";
  GetTime( timeSta );

  DistVec<Index3, DblNumMat, ElemPrtn>  result_mat_HX;
  scfdg_Hamiltonian_times_eigenvectors(result_mat_HX);

  GetTime( timeEnd );
  statusOFS << std::endl << " H * X for filtered orthonormal vectors computed . ( " << (timeEnd - timeSta ) << " s.)";

  statusOFS << std::endl << std::endl << " Alternate to Raleigh-Ritz step : " << std::endl;

  GetTime(timeSta);
  // Convert HX to ScaLAPACK format on Cheby-Scala grid
  dgdft::scalapack::ScaLAPACKMatrix<Real>  cheby_scala_HX;
  dgdft::scalapack::ScaLAPACKMatrix<Real>  bigger_grid_HX;

  dgdft::scalapack::Descriptor cheby_scala_HX_desc;
  dgdft::scalapack::Descriptor bigger_grid_HX_desc;

  // Convert HX to ScaLAPACK format on Cheby-Scala grid
  if(cheby_scala_context >= 0)
  {
    statusOFS << std::endl << " Distributed vector HX to ScaLAPACK (Cheby-Scala grid) conversion ... ";
    GetTime( extra_timeSta );

    cheby_scala_HX_desc.Init(hamDG.NumBasisTotal(), hamDG.NumStateTotal(), 
        scaBlockSize_, scaBlockSize_, 
        0, 0, 
        cheby_scala_context);

    cheby_scala_HX.SetDescriptor(cheby_scala_HX_desc);   

    scfdg_Cheby_convert_eigvec_distmat_to_ScaLAPACK(result_mat_HX,
        domain_.colComm,
        cheby_scala_HX_desc,
        cheby_scala_HX);

    GetTime( extra_timeEnd );
    statusOFS << " Done. ( " << (extra_timeEnd - extra_timeSta ) << " s.)";            

  }

  // Move to bigger grid from Cheby-Scala grid 
  if(bigger_grid_context >= 0)
  {
    statusOFS << std::endl << " Cheby-Scala grid to bigger grid pdgemr2d for HX ... ";
    GetTime( extra_timeSta );

    bigger_grid_HX_desc.Init(hamDG.NumBasisTotal(), hamDG.NumStateTotal(), 
        scaBlockSize_, scaBlockSize_, 
        0, 0, 
        bigger_grid_context);

    bigger_grid_HX.SetDescriptor(bigger_grid_HX_desc);   

    SCALAPACK(pdgemr2d)(&M_copy_, &N_copy_, 
        cheby_scala_HX.Data(), &I_ONE, &I_ONE,
        cheby_scala_HX.Desc().Values(), 
        bigger_grid_HX.Data(), &I_ONE, &I_ONE, 
        bigger_grid_HX.Desc().Values(), 
        &bigger_grid_context);      

    GetTime( extra_timeEnd );
    statusOFS << " Done. ( " << (extra_timeEnd - extra_timeSta ) << " s.)";

  }
  // Compute X^T * HX on bigger grid
  dgdft::scalapack::Descriptor bigger_grid_square_mat_desc;
  dgdft::scalapack::ScaLAPACKMatrix<Real>  bigger_grid_square_mat;

  if(SCFDG_comp_subspace_syr2k_ == 0)
    statusOFS << std::endl << " Computing X^T * HX on bigger grid using GEMM ... ";
  else
    statusOFS << std::endl << " Computing X^T * HX on bigger grid using SYR2K + TRADD ... ";

  GetTime( extra_timeSta );
  if(bigger_grid_context >= 0)
  {
    bigger_grid_square_mat_desc.Init( hamDG.NumStateTotal(), hamDG.NumStateTotal(), 
        scaBlockSize_, scaBlockSize_, 
        0, 0, 
        bigger_grid_context);


    bigger_grid_square_mat.SetDescriptor(bigger_grid_square_mat_desc);

    if(SCFDG_comp_subspace_syr2k_ == 0)
    {        
      dgdft::scalapack::Gemm('T', 'N',
          hamDG.NumStateTotal(), hamDG.NumStateTotal(), hamDG.NumBasisTotal(),
          1.0,
          bigger_grid_eigvecs_X.Data(), I_ONE, I_ONE, bigger_grid_eigvecs_X.Desc().Values(), 
          bigger_grid_HX.Data(), I_ONE, I_ONE, bigger_grid_HX.Desc().Values(),
          0.0,
          bigger_grid_square_mat.Data(), I_ONE, I_ONE, bigger_grid_square_mat.Desc().Values(),
          bigger_grid_context);
    }
    else
    {
      dgdft::scalapack::Syr2k ('U', 'T',
          hamDG.NumStateTotal(), hamDG.NumBasisTotal(),
          0.5, 
          bigger_grid_eigvecs_X.Data(), I_ONE, I_ONE, bigger_grid_eigvecs_X.Desc().Values(),
          bigger_grid_HX.Data(), I_ONE, I_ONE, bigger_grid_HX.Desc().Values(),
          0.0,
          bigger_grid_square_mat.Data(), I_ONE, I_ONE, bigger_grid_square_mat.Desc().Values());


      // Copy the upper triangle to a temporary location
      dgdft::scalapack::ScaLAPACKMatrix<Real>  bigger_grid_square_mat_copy;
      bigger_grid_square_mat_copy.SetDescriptor(bigger_grid_square_mat_desc);

      char uplo = 'A';
      int ht = hamDG.NumStateTotal();
      dgdft::scalapack::SCALAPACK(pdlacpy)(&uplo, &ht, &ht,
          bigger_grid_square_mat.Data(), &I_ONE, &I_ONE, bigger_grid_square_mat.Desc().Values(), 
          bigger_grid_square_mat_copy.Data(), &I_ONE, &I_ONE, bigger_grid_square_mat_copy.Desc().Values() );

      uplo = 'L';
      char trans = 'T';
      double scalar_one = 1.0, scalar_zero = 0.0;
      dgdft::scalapack::SCALAPACK(pdtradd)(&uplo, &trans, &ht, &ht,
          &scalar_one,
          bigger_grid_square_mat_copy.Data(), &I_ONE, &I_ONE, bigger_grid_square_mat_copy.Desc().Values(), 
          &scalar_zero,
          bigger_grid_square_mat.Data(), &I_ONE, &I_ONE, bigger_grid_square_mat.Desc().Values());
    }        
  } 
  GetTime( extra_timeEnd );
  statusOFS << " Done. ( " << (extra_timeEnd - extra_timeSta ) << " s.)" ;

  // Move square matrix to Cheby-Scala grid for working with top states
  dgdft::scalapack::Descriptor cheby_scala_square_mat_desc;
  dgdft::scalapack::ScaLAPACKMatrix<Real>  cheby_scala_square_mat;

  statusOFS << std::endl << " Moving X^T * HX to Cheby-Scala grid using pdgemr2d ... ";
  GetTime( extra_timeSta );

  if(cheby_scala_context >= 0)
  {
    cheby_scala_square_mat_desc.Init( hamDG.NumStateTotal(), hamDG.NumStateTotal(), 
        scaBlockSize_, scaBlockSize_, 
        0, 0, 
        cheby_scala_context);

    cheby_scala_square_mat.SetDescriptor(cheby_scala_square_mat_desc);     
  }

  if(bigger_grid_context >= 0)
  {
    SCALAPACK(pdgemr2d)(&N_copy_, &N_copy_, 
        bigger_grid_square_mat.Data(), &I_ONE, &I_ONE,
        bigger_grid_square_mat.Desc().Values(), 
        cheby_scala_square_mat.Data(), &I_ONE, &I_ONE, 
        cheby_scala_square_mat.Desc().Values(), 
        &bigger_grid_context);      
  }

  GetTime( extra_timeEnd );
  statusOFS << " Done. ( " << (extra_timeEnd - extra_timeSta ) << " s.)" << std::endl;

  // All ready for doing the inner CheFSI
  if(cheby_scala_context >= 0)
  {
    // Obtain the spectral bounds    

    // Fix the filter bounds using full ScaLAPACK results if inner Cheby has not been engaged
    if(SCFDG_comp_subspace_engaged_ != 1)
    {
      SCFDG_comp_subspace_inner_CheFSI_a_L_ = - eigval[hamDG.NumStateTotal() - 1];

      int state_ind = hamDG.NumStateTotal() - SCFDG_comp_subspace_N_solve_;        
      SCFDG_comp_subspace_inner_CheFSI_lower_bound_ = -0.5 * (eigval[state_ind] + eigval[state_ind - 1]);

      statusOFS << std::endl << " Computing upper bound of projected Hamiltonian (parallel) ... ";
      GetTime( extra_timeSta );
      SCFDG_comp_subspace_inner_CheFSI_upper_bound_ = find_comp_subspace_UB_parallel(cheby_scala_square_mat);
      GetTime( extra_timeEnd );
      statusOFS << " Done. ( " << (extra_timeEnd - extra_timeSta ) << " s.)" << std::endl;

      Hmat_top_states_Cheby_delta_fudge_ = 0.5 * (eigval[state_ind] - eigval[state_ind - 1]);

      statusOFS << std::endl << " Going into inner CheFSI routine (parallel) for top states ... ";
      statusOFS << std::endl << "   Lower bound = -(average of prev. eigenvalues " << eigval[state_ind] 
        << " and " <<  eigval[state_ind - 1] << ") = " << SCFDG_comp_subspace_inner_CheFSI_lower_bound_;
    }
    else
    {
      // Fix the filter bounds using earlier inner CheFSI results
      SCFDG_comp_subspace_inner_CheFSI_lower_bound_ = - (SCFDG_comp_subspace_top_eigvals_[SCFDG_comp_subspace_N_solve_ - 1] - Hmat_top_states_Cheby_delta_fudge_);

      statusOFS << std::endl << " Computing upper bound of projected Hamiltonian (parallel) ... ";
      GetTime( extra_timeSta );
      SCFDG_comp_subspace_inner_CheFSI_upper_bound_ = find_comp_subspace_UB_parallel(cheby_scala_square_mat);
      GetTime( extra_timeEnd );
      statusOFS << " Done. ( " << (extra_timeEnd - extra_timeSta ) << " s.)" << std::endl;

      statusOFS << std::endl << " Going into inner CheFSI routine (parallel) for top states ... ";
      statusOFS << std::endl << "   Lower bound eigenvalue = " << SCFDG_comp_subspace_top_eigvals_[SCFDG_comp_subspace_N_solve_ - 1] 
        << std::endl << "   delta_fudge = " << Hmat_top_states_Cheby_delta_fudge_;
    }    
  }

  // Broadcast the inner filter bounds, etc. to every process. 
  // This is definitely required by the procs participating in cheby_scala_context
  // Some of the info is redundant
  double bounds_array[4];
  bounds_array[0] = SCFDG_comp_subspace_inner_CheFSI_lower_bound_;
  bounds_array[1] = SCFDG_comp_subspace_inner_CheFSI_upper_bound_;
  bounds_array[2] = SCFDG_comp_subspace_inner_CheFSI_a_L_;
  bounds_array[3] = Hmat_top_states_Cheby_delta_fudge_;

  MPI_Bcast(bounds_array, 4, MPI_DOUBLE, 0, domain_.comm); 

  SCFDG_comp_subspace_inner_CheFSI_lower_bound_ = bounds_array[0];
  SCFDG_comp_subspace_inner_CheFSI_upper_bound_ = bounds_array[1];
  SCFDG_comp_subspace_inner_CheFSI_a_L_ = bounds_array[2];
  Hmat_top_states_Cheby_delta_fudge_ = bounds_array[3];

  if(cheby_scala_context >= 0)
    statusOFS << std::endl << "   Lanczos upper bound = " << SCFDG_comp_subspace_inner_CheFSI_upper_bound_ << std::endl;

  // Load up and distibute top eigenvectors from serial storage
  scalapack::Descriptor temp_single_proc_desc;
  scalapack::ScaLAPACKMatrix<Real>  temp_single_proc_scala_mat;

  GetTime( extra_timeSta );
  statusOFS << std::endl << " Loading up and distributing initial guess vectors ... ";

  int M_temp_ =  hamDG.NumStateTotal();
  int N_temp_ =  SCFDG_comp_subspace_N_solve_;

  if(single_proc_context >= 0)
  {
    temp_single_proc_desc.Init(M_temp_, N_temp_,
        scaBlockSize_, scaBlockSize_, 
        0, 0,  single_proc_context );              

    temp_single_proc_scala_mat.SetDescriptor( temp_single_proc_desc );

    // Copy from the serial storage to the single process ScaLAPACK matrix
    double *src_ptr, *dest_ptr; 

    // Copy in the regular order      
    for(Int copy_iter = 0; copy_iter < SCFDG_comp_subspace_N_solve_; copy_iter ++)
    {
      src_ptr = SCFDG_comp_subspace_start_guess_.VecData(copy_iter);
      dest_ptr = temp_single_proc_scala_mat.Data() + copy_iter * temp_single_proc_scala_mat.LocalLDim();

      blas::Copy( M_temp_, src_ptr, 1, dest_ptr, 1 );                                                 
    }
  }

  // Distribute onto Cheby-Scala grid    
  scalapack::Descriptor Xmat_desc;
  dgdft::scalapack::ScaLAPACKMatrix<Real> Xmat;

  if(cheby_scala_context >= 0)
  {
    Xmat_desc.Init(M_temp_, N_temp_,
        scaBlockSize_, scaBlockSize_, 
        0, 0,  cheby_scala_context );          

    Xmat.SetDescriptor(Xmat_desc);     

    SCALAPACK(pdgemr2d)(&M_temp_, &N_temp_, 
        temp_single_proc_scala_mat.Data(), &I_ONE, &I_ONE,
        temp_single_proc_scala_mat.Desc().Values(), 
        Xmat.Data(), &I_ONE, &I_ONE, 
        Xmat.Desc().Values(), 
        &cheby_scala_context);    
  }

  GetTime( extra_timeEnd );
  statusOFS << " Done. ( " << (extra_timeEnd - extra_timeSta ) << " s.)" << std::endl;

  // Call the inner CheFSI routine 
  DblNumVec eig_vals_Xmat;
  eig_vals_Xmat.Resize(SCFDG_comp_subspace_N_solve_);

  if(cheby_scala_context >= 0)
  {
    GetTime(extra_timeSta);

    CheFSI_Hmat_top_parallel(cheby_scala_square_mat,
        Xmat,
        eig_vals_Xmat,
        Hmat_top_states_ChebyFilterOrder_,
        Hmat_top_states_ChebyCycleNum_,
        SCFDG_comp_subspace_inner_CheFSI_lower_bound_,SCFDG_comp_subspace_inner_CheFSI_upper_bound_, SCFDG_comp_subspace_inner_CheFSI_a_L_);

    GetTime(extra_timeEnd);

    statusOFS << std::endl << " Parallel CheFSI completed on " <<  SCFDG_comp_subspace_N_solve_ 
      << " top states ( " << (extra_timeEnd - extra_timeSta ) << " s.)";
  }

  // Redistribute and broadcast top eigenvectors to serial storage
  GetTime( extra_timeSta );
  statusOFS << std::endl << " Distributing back and broadcasting inner CheFSI vectors ... ";

  // Redistribute to single process ScaLAPACK matrix
  if(cheby_scala_context >= 0)
  {    
    SCALAPACK(pdgemr2d)(&M_temp_, &N_temp_, 
        Xmat.Data(), &I_ONE, &I_ONE,
        Xmat.Desc().Values(), 
        temp_single_proc_scala_mat.Data(), &I_ONE, &I_ONE, 
        temp_single_proc_scala_mat.Desc().Values(), 
        &cheby_scala_context);    
  }

  if(single_proc_context >= 0)
  {
    // Copy from the single process ScaLAPACK matrix to serial storage
    double *src_ptr, *dest_ptr; 

    // Copy in the regular order      
    for(Int copy_iter = 0; copy_iter < SCFDG_comp_subspace_N_solve_; copy_iter ++)
    {
      src_ptr = temp_single_proc_scala_mat.Data() + copy_iter * temp_single_proc_scala_mat.LocalLDim();
      dest_ptr = SCFDG_comp_subspace_start_guess_.VecData(copy_iter);

      blas::Copy( M_temp_, src_ptr, 1, dest_ptr, 1 );                                                 
    }
  }

  // Broadcast top eigenvectors to all processes
  MPI_Bcast(SCFDG_comp_subspace_start_guess_.Data(),hamDG.NumStateTotal() * SCFDG_comp_subspace_N_solve_, 
      MPI_DOUBLE, 0,  domain_.comm); 

  GetTime( extra_timeEnd );
  statusOFS << " Done. ( " << (extra_timeEnd - extra_timeSta ) << " s.)" << std::endl;

  GetTime(timeEnd);
  statusOFS << std::endl << std::endl << " Alternate to Raleigh-Ritz step performed in " << (timeEnd - timeSta ) << " s.";


  GetTime( extra_timeSta );
  statusOFS << std::endl << " Adjusting top eigenvalues ... ";

  // Broadcast the top eigenvalues to every processor
  MPI_Bcast(eig_vals_Xmat.Data(), SCFDG_comp_subspace_N_solve_ , MPI_DOUBLE, 0, domain_.comm); 

  // Copy these to native storage
  SCFDG_comp_subspace_top_eigvals_.Resize(SCFDG_comp_subspace_N_solve_);
  for(Int copy_iter = 0; copy_iter < SCFDG_comp_subspace_N_solve_; copy_iter ++)
    SCFDG_comp_subspace_top_eigvals_[copy_iter] = eig_vals_Xmat[copy_iter];

  // Also update the top eigenvalues in hamDG in case we need them
  // For example, they are required if we switch back to regular CheFSI at some stage 
  Int n_top = hamDG.NumStateTotal() - 1;
  for(Int copy_iter = 0; copy_iter < SCFDG_comp_subspace_N_solve_; copy_iter ++)
    eigval[n_top - copy_iter] = SCFDG_comp_subspace_top_eigvals_[copy_iter];

  GetTime( extra_timeEnd );
  statusOFS << " Done. ( " << (extra_timeEnd - extra_timeSta ) << " s.)" << std::endl;

  // Compute the occupations    
  GetTime( extra_timeSta );
  statusOFS << std::endl << " Computing occupation numbers : ";   
  SCFDG_comp_subspace_top_occupations_.Resize(SCFDG_comp_subspace_N_solve_);
  
  Int howmany_to_calc = (hamDGPtr_->NumOccupiedState() + SCFDG_comp_subspace_N_solve_) - hamDGPtr_->NumStateTotal(); 
  scfdg_calc_occ_rate_comp_subspc(SCFDG_comp_subspace_top_eigvals_,SCFDG_comp_subspace_top_occupations_, howmany_to_calc);

  statusOFS << std::endl << " SCFDG_comp_subspace_N_solve_ = " << SCFDG_comp_subspace_N_solve_ << std::endl;
  
  GetTime( extra_timeEnd );
  statusOFS << std::endl << " Completed computing occupations. ( " << (extra_timeEnd - extra_timeSta ) << " s.)" << std::endl;

  statusOFS << std::endl << " Fermi level = " << fermi_ << std::endl;
  statusOFS << std::endl << " Top Eigenvalues and occupations (in reverse order) : ";
  
  for(int print_iter = 0; print_iter < SCFDG_comp_subspace_N_solve_; print_iter ++)
     statusOFS << std::endl <<  " " << std::setw(8) << print_iter << std::setw(20) << SCFDG_comp_subspace_top_eigvals_[print_iter] 
                           << '\t' << SCFDG_comp_subspace_top_occupations_[print_iter];
  
  // Form the matrix C by scaling the eigenvectors with the appropriate occupation related weights

  GetTime( extra_timeSta );
  statusOFS << std::endl << " Forming the occupation number weighted matrix C : ";   

  int wd = hamDGPtr_->NumStateTotal();

  SCFDG_comp_subspace_matC_.Resize(wd, SCFDG_comp_subspace_N_solve_);
  lapack::Lacpy( 'A', wd, SCFDG_comp_subspace_N_solve_, SCFDG_comp_subspace_start_guess_.Data(), wd, 
      SCFDG_comp_subspace_matC_.Data(), wd );

  statusOFS << std::endl << "  Space for matrix C locally allocated. ( " << (double(hamDG.NumStateTotal() *  SCFDG_comp_subspace_N_solve_ * 8) / double(1048576)) << " MBs per process. )";;   

  double scale_fac;
  for(Int scal_iter = 0; scal_iter < SCFDG_comp_subspace_N_solve_; scal_iter ++)
  {
    scale_fac = sqrt(1.0 - SCFDG_comp_subspace_top_occupations_(scal_iter));
    blas::Scal(wd, scale_fac, SCFDG_comp_subspace_matC_.VecData(scal_iter), 1);
  }
  statusOFS << std::endl << "  BLAS Scal operations completed. ";   

  GetTime( extra_timeEnd );
  statusOFS << std::endl  << " Occupation number weighted matrix C formed. ( " << (extra_timeEnd - extra_timeSta ) << " s.)" << std::endl;

  // This calculation is done for computing the band energy later
  GetTime( extra_timeSta );
  statusOFS << std::endl << " Computing the trace of the projected Hamiltonian ... ";   

  SCFDG_comp_subspace_trace_Hmat_ = 0.0;

  if(cheby_scala_context >= 0)
  {  
    SCFDG_comp_subspace_trace_Hmat_ = dgdft::scalapack::SCALAPACK(pdlatra)(&M_temp_ , 
        cheby_scala_square_mat.Data() , &I_ONE , &I_ONE , 
        cheby_scala_square_mat.Desc().Values());

  }

  // Broadcast the trace
  MPI_Bcast(&SCFDG_comp_subspace_trace_Hmat_, 1 , MPI_DOUBLE, 0, domain_.comm); 

  GetTime( extra_timeEnd );
  statusOFS << " Done. ( " << (extra_timeEnd - extra_timeSta ) << " s.)" << std::endl;

  statusOFS << std::endl << std::endl << " ------------------------------- ";

  // Adjust other things... trace, etc. Broadcast necessary parts of these results

  statusOFS << std::endl << std::endl << " ------------------------------- " << std::endl;

  // Clean up BLACS
  if(cheby_scala_context >= 0) 
  {
    dgdft::scalapack::Cblacs_gridexit( cheby_scala_context );
  }

  if(bigger_grid_context >= 0)
  {
    dgdft::scalapack::Cblacs_gridexit( bigger_grid_context );

  }

  if( single_proc_context >= 0)
  {
    dgdft::scalapack::Cblacs_gridexit( single_proc_context );             
  }
} // end of scfdg_complementary_subspace_parallel

void SCFDG::scfdg_complementary_subspace_compute_fullDM()
{
  Int mpirank, mpisize;
  MPI_Comm_rank( domain_.comm, &mpirank );
  MPI_Comm_size( domain_.comm, &mpisize );

  HamiltonianDG&  hamDG = *hamDGPtr_;
  std::vector<Index3>  getKeys_list;

  DistDblNumMat& my_dist_mat = hamDG.EigvecCoef();

  // Check that vectors provided only contain one entry in the local map
  // This is a safeguard to ensure that we are really dealing with distributed matrices
  if((my_dist_mat.LocalMap().size() != 1))
  {
    statusOFS << std::endl << " Eigenvector not formatted correctly !!"
      << std::endl << " Aborting ... " << std::endl;
    exit(1);
  }

  // Obtain key based on my_dist_mat : This assumes that my_dist_mat is formatted correctly
  // based on processor number, etc.
  Index3 key = (my_dist_mat.LocalMap().begin())->first;

  // Obtain keys of neighbors using the Hamiltonian matrix
  for(typename std::map<ElemMatKey, DblNumMat >::iterator 
      get_neighbors_from_Ham_iterator = hamDG.HMat().LocalMap().begin();
      get_neighbors_from_Ham_iterator != hamDG.HMat().LocalMap().end();
      get_neighbors_from_Ham_iterator ++)
  {
    Index3 neighbor_key = (get_neighbors_from_Ham_iterator->first).second;

    if(neighbor_key == key)
      continue;
    else
      getKeys_list.push_back(neighbor_key);
  }

  // Do the communication necessary to get the information from
  // procs holding the neighbors
  my_dist_mat.GetBegin( getKeys_list, NO_MASK ); 
  my_dist_mat.GetEnd( NO_MASK );

  DblNumMat XC_mat;
  // First compute the diagonal block
  {
    DblNumMat &mat_local = my_dist_mat.LocalMap()[key];
    ElemMatKey diag_block_key = std::make_pair(key, key);

    //statusOFS << std::endl << " Diag key = " << diag_block_key.first << "  " << diag_block_key.second << std::endl;

    // First compute the X*X^T portion : adjust for numspin
    distDMMat_.LocalMap()[diag_block_key].Resize( mat_local.m(),  mat_local.m());

    blas::Gemm( 'N', 'T', mat_local.m(), mat_local.m(), mat_local.n(),
        hamDG.NumSpin(), 
        mat_local.Data(), mat_local.m(), 
        mat_local.Data(), mat_local.m(),
        0.0, 
        distDMMat_.LocalMap()[diag_block_key].Data(),  mat_local.m());

    // Now compute the X * C portion
    XC_mat.Resize(mat_local.m(), SCFDG_comp_subspace_N_solve_);        
    blas::Gemm( 'N', 'N', mat_local.m(), SCFDG_comp_subspace_N_solve_, mat_local.n(),
        1.0, 
        mat_local.Data(), mat_local.m(), 
        SCFDG_comp_subspace_matC_.Data(), SCFDG_comp_subspace_matC_.m(),
        0.0, 
        XC_mat.Data(),  XC_mat.m());

    // Subtract XC*XC^T from DM : adjust for numspin
    blas::Gemm( 'N', 'T', XC_mat.m(), XC_mat.m(), XC_mat.n(),
        -hamDG.NumSpin(), 
        XC_mat.Data(), XC_mat.m(), 
        XC_mat.Data(), XC_mat.m(),
        1.0, 
        distDMMat_.LocalMap()[diag_block_key].Data(),  mat_local.m());
  }

  // Now handle the off-diagonal blocks
  {
    DblNumMat &mat_local = my_dist_mat.LocalMap()[key];
    for(Int off_diag_iter = 0; off_diag_iter < getKeys_list.size(); off_diag_iter ++)
    {
      DblNumMat &mat_neighbor = my_dist_mat.LocalMap()[getKeys_list[off_diag_iter]];
      ElemMatKey off_diag_key = std::make_pair(key, getKeys_list[off_diag_iter]);

      //statusOFS << std::endl << " Off Diag key = " << off_diag_key.first << "  " << off_diag_key.second << std::endl;

      // First compute the Xi * Xj^T portion : adjust for numspin
      distDMMat_.LocalMap()[off_diag_key].Resize( mat_local.m(),  mat_neighbor.m());

      blas::Gemm( 'N', 'T', mat_local.m(), mat_neighbor.m(), mat_local.n(),
          hamDG.NumSpin(), 
          mat_local.Data(), mat_local.m(), 
          mat_neighbor.Data(), mat_neighbor.m(),
          0.0, 
          distDMMat_.LocalMap()[off_diag_key].Data(),  mat_local.m());

      // Now compute the XC portion for the off-diagonal block
      DblNumMat XC_neighbor_mat;
      XC_neighbor_mat.Resize(mat_neighbor.m(), SCFDG_comp_subspace_N_solve_);

      blas::Gemm( 'N', 'N', mat_neighbor.m(), SCFDG_comp_subspace_N_solve_, mat_neighbor.n(),
          1.0, 
          mat_neighbor.Data(), mat_neighbor.m(), 
          SCFDG_comp_subspace_matC_.Data(), SCFDG_comp_subspace_matC_.m(),
          0.0, 
          XC_neighbor_mat.Data(),  XC_neighbor_mat.m());

      // Subtract (Xi C)* (Xj C)^T from off diagonal block : adjust for numspin
      blas::Gemm( 'N', 'T', XC_mat.m(), XC_neighbor_mat.m(), XC_mat.n(),
          -hamDG.NumSpin(), 
          XC_mat.Data(), XC_mat.m(), 
          XC_neighbor_mat.Data(), XC_neighbor_mat.m(),
          1.0, 
          distDMMat_.LocalMap()[off_diag_key].Data(),  mat_local.m());
    }
  }
  // Need to clean up extra entries in my_dist_mat
  typename std::map<Index3, DblNumMat >::iterator it;
  for(Int delete_iter = 0; delete_iter <  getKeys_list.size(); delete_iter ++)
  {
    it = my_dist_mat.LocalMap().find(getKeys_list[delete_iter]);
    (my_dist_mat.LocalMap()).erase(it);
  }
}

void 
SCFDG::scfdg_calc_occ_rate_comp_subspc( DblNumVec& top_eigVals, DblNumVec& top_occ, Int num_solve)
{
  // For a given finite temperature, update the occupation number
  // FIXME Magic number here
  Real tol = 1e-10; 
  Int maxiter = 100;  

  Real lb, ub, flb, fub, fx;
  Int  iter;

  Int npsi       = top_eigVals.m();
  Int nOccStates = num_solve;

  top_occ.Resize(npsi);

  if( npsi > nOccStates )  
  { 
    if(SmearingScheme_ == "FD")
    {  
      // The reverse order for the bounds needs to be used because the eigenvalues appear in decreasing order
      lb = top_eigVals(npsi - 1);
      ub = top_eigVals(0);

      flb = scfdg_fermi_func_comp_subspc(top_eigVals, top_occ, num_solve, lb);
      fub = scfdg_fermi_func_comp_subspc(top_eigVals, top_occ, num_solve, ub);

      if(flb * fub > 0.0)
        ErrorHandling( "Bisection method for finding Fermi level cannot proceed !!" );

      fermi_ = (lb+ub)*0.5;

      /* Start bisection iteration */
      iter = 1;
      fx = scfdg_fermi_func_comp_subspc(top_eigVals, top_occ, num_solve, fermi_);


      while( (fabs(fx) > tol) && (iter < maxiter) ) 
      {
        flb = scfdg_fermi_func_comp_subspc(top_eigVals, top_occ, num_solve, lb);
        fub = scfdg_fermi_func_comp_subspc(top_eigVals, top_occ, num_solve, ub);

        if( (flb * fx) < 0.0 )
          ub = fermi_;
        else
          lb = fermi_;

        fermi_ = (lb+ub)*0.5;
        fx = scfdg_fermi_func_comp_subspc(top_eigVals, top_occ, num_solve, fermi_);

        iter++;
      }
    } // end of if (SmearingScheme_ == "fd")
    else
    {
      // GB and MP smearing schemes

      // The reverse order for the bounds needs to be used because the eigenvalues appear in decreasing order
      lb = top_eigVals(npsi - 1);
      ub = top_eigVals(0);

      // Set up the function bounds
      flb = mp_occupations_residual(top_eigVals, lb, num_solve, Tsigma_, MP_smearing_order_ );
      fub = mp_occupations_residual(top_eigVals, ub, num_solve, Tsigma_, MP_smearing_order_ );

      if(flb * fub > 0.0)
        ErrorHandling( "Bisection method for finding Fermi level cannot proceed !!" );

      fermi_ = ( lb + ub ) * 0.5;

      /* Start bisection iteration */
      iter = 1;
      fx = mp_occupations_residual(top_eigVals, fermi_, num_solve, Tsigma_, MP_smearing_order_ );

      while( (fabs(fx) > tol) && (iter < maxiter) ) 
      {
        flb = mp_occupations_residual(top_eigVals, lb, num_solve, Tsigma_, MP_smearing_order_ );
        fub = mp_occupations_residual(top_eigVals, ub, num_solve, Tsigma_, MP_smearing_order_ );

        if( (flb * fx) < 0.0 )
          ub = fermi_;
        else
          lb = fermi_;

        fermi_ = ( lb + ub ) * 0.5;
        fx = mp_occupations_residual(top_eigVals, fermi_, num_solve, Tsigma_, MP_smearing_order_);

        iter++;
      }

      if(iter >= maxiter)
        ErrorHandling( "Bisection method for finding Fermi level does not appear to converge !!" );
      else
      {
        // Bisection method seems to have converged
        // Fill up the occupations
        populate_mp_occupations(top_eigVals, top_occ, fermi_, Tsigma_, MP_smearing_order_);
      }

    } // end of GB and MP smearing cases

  } // End of finite temperature case
  else 
  {
    if (npsi == nOccStates ) 
    {
      for(Int j = 0; j < npsi; j++) 
        top_occ(j) = 1.0;
      fermi_ = top_eigVals(npsi-1);
    }
    else 
    {
      ErrorHandling( "The number of top eigenvalues should be larger than number of top occupied states" );
    }
  }

  return ;
}

double 
SCFDG::scfdg_fermi_func_comp_subspc( DblNumVec& top_eigVals, DblNumVec& top_occ, Int num_solve, Real x)
{
  double occsum = 0.0, retval;
  Int npsi = top_eigVals.m();


  for(Int j = 0; j < npsi; j++) 
  {
    top_occ(j) = 1.0 / (1.0 + exp(Tbeta_*(top_eigVals(j) - x)));
    occsum += top_occ(j);     
  }

  retval = occsum - Real(num_solve);

  return retval;
}
/*
// Internal routines for MP (and GB) type smearing
double SCFDG::low_order_hermite_poly(double x, int order)
{
  double y; 
  switch (order)
  {
    case 0: y = 1; break;
    case 1: y = 2.0 * x; break;
    case 2: y = 4.0 * x * x - 2.0; break;
    case 3: y = 8.0 * x * x * x - 12.0 * x; break;
    case 4: y = 16.0 * x * x * x * x - 48.0 * x * x + 12.0; break;
    case 5: y = 32.0 * x * x * x * x * x - 160.0 * x * x * x + 120.0 * x; break;
    case 6: y = 64.0 * x * x * x * x * x * x - 480.0 * x * x * x * x + 720.0 * x * x - 120.0; 
  }

  return y;
}

double SCFDG::mp_occupations(double x, int order)
{
  const double sqrt_pi = sqrt(M_PI);
  double A_vec[4] = { 1.0 / sqrt_pi, -1.0 / (4.0 * sqrt_pi), 1.0 / (32.0 * sqrt_pi), -1.0 / (384 * sqrt_pi) };
  double y = 0.5 *(1.0 - erf(x));

  for (int m = 1; m <= order; m++)
    y = y + A_vec[m] * low_order_hermite_poly(x, 2 * order - 1) * exp(- x * x);

  return y;

}

double SCFDG::mp_entropy(double x, int order)
{
  const double sqrt_pi = sqrt(M_PI);
  double A_vec[4] = { 1.0 / sqrt_pi, -1.0 / (4.0 * sqrt_pi), 1.0 / (32.0 * sqrt_pi), -1.0 / (384 * sqrt_pi) };

  double y = 0.5 * A_vec[order] * low_order_hermite_poly(x, 2 * order) * exp(- x * x);

  return y;
}

// This fills up the the output_occ occupations using the input eigvals, according to the Methfessel-Paxton recipe 
void 
SCFDG::populate_mp_occupations(DblNumVec& input_eigvals, DblNumVec& output_occ, double fermi_mu)
{
  double x, t;

  for(int ii = 0; ii < input_eigvals.m(); ii ++)
  {
    x = (input_eigvals(ii) - fermi_mu) / Tsigma_ ;
    t = mp_occupations(x, MP_smearing_order_); 

    if(t < 0.0)
      t = 0.0;
    if(t > 1.0)
      t = 1.0;

    output_occ(ii) = t;

  }
}

// This computes the residual of (\sum_i f_i ) - n_e used for computing the Fermi level
double  
SCFDG::mp_occupations_residual(DblNumVec& input_eigvals, double fermi_mu, int num_solve)
{
  double x;
  double y = 0.0, t;

  for(int ii = 0; ii < input_eigvals.m(); ii ++)
  {
    x = (input_eigvals(ii) - fermi_mu) / Tsigma_ ;
    t = mp_occupations(x, MP_smearing_order_);

    if(t < 0.0)
      t = 0.0;
    if(t > 1.0)
      t = 1.0;

    y += t;    
  }

  return (y - double(num_solve));
}
*/

} // namespace dgdft
